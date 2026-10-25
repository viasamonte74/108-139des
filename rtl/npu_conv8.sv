// npu_conv8 - CONV_2D / FULLY_CONNECTED / DEPTHWISE_CONV_2D engine: 8 lanes x 16-wide int8 dot
// products.
//
// Conv / FC: output channels go in groups of 8 (one per lane). Each lane holds its channel's
// weights for one K-chunk (<= 144 bytes) locally; the im2col vector of every output pixel streams
// past once per group and is broadcast to all lanes (128 MACs per activation beat). K > 144 runs as
// several chunk passes; partial sums are parked in a DRAM scratch buffer and come back on S1.
//   acc_j = [bias_j] + sum((in + in_off) * w_j)   ->  requant (2 units)  ->  int8
// The input offset is added once to the shared activation operand, and each lane multiplies with
// radix-4 Booth digits of its weights selecting among the multiples {+-a', +-2a'}.
//
// Depthwise 3x3 (p_dw): channels go in groups of 16. The 3x3 window of 16-channel beats is rebuilt
// tap by tap as in the sliced conv walk (only newly uncovered columns are read); once complete,
// every lane takes the 9 taps of one channel from the window (elements 0..8, its own multiples) and
// two MAC cycles cover channels 0-7 and 8-15. The window may change only after both have issued.
module npu_conv8 #(
    parameter int NL = 8,
    parameter int KE = 9,
    parameter bit RQ8 = 1'b1,    // requant units E..H present: fast octets drain in one cycle
    parameter bit XDR = 1'b1,    // stride-1 dual-row cross issues (9 MAC issues per sliding pair)
    parameter bit DWF = 1'b1,    // (v37_1a) fused depthwise unit (npu_dwf)
    parameter bit CHN = 1'b1     // (v42DS) DWF op chaining (see chain_ok)
) (
    input  logic         clk,
    input  logic         rst_n,
    input  logic         start,
    input  logic         pre,          // (v28A) the start of the next cycle's op is known (registers loaded)
    input  logic [31:0]  p_in, p_wgt, p_bias, p_mult, p_shift, p_out, p_scr,
    // (v37_1a) DWF: a paired-group pass (16 channels, W = 5 raster pixels) feeds a 3x3 stride-1 pad-1 depthwise conv
    // of the same 16 channels (npu_dwf, beside the lanes); only the depthwise outputs are written (NHWC: p_out +
    // 16 g + p_co * pixel). Per pass, after the pass's own parameters: the depthwise weights (144 bytes at p_dwgt +
    // 144 g: channel-major taps), biases, multipliers, shifts (64 bytes at p_dbias / p_dmult / p_dshift + 64 g).
    input  logic         p_dwf_r,
    input  logic         p_rot_r,      // (v47c AG) RG_PG bit 5: line-aligned output channel groups (see ag_d)
    input  logic         p_dwt_r,      // (v46DS) tee: the pointwise results too, NHWC at p_out2 + 16 g + p_co * pixel
    input  logic [19:0]  p_out2,
    input  logic [19:0]  p_dwgt, p_dbias, p_dmult, p_dshift,
    input  logic         p_has_bias,
    input  logic         p_hf,         // (v52BF) RG_HASB bit 1: add in_off * sum(w) per pass in hardware (oc)
    input  logic [3:0]   p_gl,         // (v55_KC) RG_HASB bit 2: at most this many groups (0: no limit; compact conv)
    // (v15 widths, kept by the kernel: Ho, Wo, FH, H, W <= 255, Co, Ci <= 1023, K, L < 8192,
    // Ho*Wo < 4096, strides < 65536, implicit padding <= 3)
    input  logic [7:0]   p_ho, p_wo, p_fh,
    input  logic [9:0]   p_co,
    input  logic [12:0]  p_l,          // FW*Ci  (piece length)
    input  logic [15:0]  p_rowstride,  // W*Ci
    input  logic [15:0]  p_pixstep,    // sw*Ci
    input  logic [15:0]  p_rowstep,    // sh*W*Ci
    input  logic [12:0]  p_k,          // FH*FW*Ci
    input  logic [11:0]  p_npix,       // Ho*Wo
    input  logic         p_contig_r,   // FH=FW=1 and stride 1: activations of all pixels contiguous
    input  logic [4:0]   p_nchunks_r,  // ceil(K/144)
    input  logic [16:0]  p_wglast,     // gc_last*K (weights of the last group, single chunk)
    input  logic [19:0]  p_actlen,     // npix*L   (all activations, contiguous single chunk)
    input  logic [19:0]  p_outlen,     // npix*Co  (all outputs, single group)
    input  logic [8:0]   p_inoff,
    input  logic         p_sliced_r,   // 3x3, Ci%16==0: chunk = 16-channel slice of all 9 taps
    input  logic         p_dw,         // depthwise 3x3, stride 1 or 2
    input  logic         p_pg_r,       // paired groups: im2col conv, K <= 64, 16 channels per pass
    input  logic         p_fs,         // FC stream: npix 1, K <= 720 (see below)
    input  logic         p_fsa,        // (v34P) FC stream: accumulators start from S1 partial sums (p_scr)
    input  logic         p_raw,        // (v34P) FC stream: raw 32-bit accumulators out (no requant, no mult / shift words)
    input  logic         p_dr_r,       // dual-row sliced conv (stride 1): pixels (x, y) [A] and (x, y + 1) [B]
                                       // share one 4-row window walk; A accumulates in acc, B in acc2;
                                       // stream per pair: rows 0-2 new taps, [A residual], row 3 new taps,
                                       // [B residual]; outputs: A at obase, B at obase + p_orow
    input  logic         p_w8,         // (v58_RS) rotated slices (see r_w8.py)
    input  logic         p_pre,        // sliced conv, several chunks: each pass's weights (8 channels x 9 taps
                                       // x 16 bytes, [lane][tap]) pre-arranged contiguously: one read per pass
    input  logic         p_fu,         // FC stream, per-tensor quantisation: multiplier / shift words
                                       // are read for the first group only (every channel's are equal)
    // packed FC (FC stream, p_pk): the layer's weight rows, each group's bias [/ mult / shift] words
    // before its rows, are one compressed stream read by a single command (p_actlen bytes at p_wgt;
    // the output is written by one command). A row's 16-byte beats come as 16-weight blocks of
    // width w (3..8 bits, 2w bytes), singly or in pairs (first beat even) behind a header byte
    // {pair, w1-1, w0-1}; a pair issues both beats at once (slot A: even beat, slot B: odd beat).
    // The activation vector is loaded as beat b -> lane b/8, entry b%8 (K <= 640).
    input  logic         p_pk,
    input  logic [19:0]  p_wg2, p_len2,   // packed FC: the stream continues with p_len2 bytes at p_wg2
    input  logic [127:0] rs0_data2,    // stream 0: the 16 bytes after rs0_data
    output logic         s1_unused,    // no traffic on S1: S0 may use long bursts
    output logic         wlong,        // (v28R) loading a pre-arranged pass's weights: S0 may use long bursts
    output logic         wlong_e,      // (v28R) the issuer is a pass ahead: S0 may go long once S1 has nothing to propose
    input  logic         p_cp,         // gather copy: the im2col vectors of all pixels are written
                                       // out contiguously (no MACs; p_co = 1, single write)
    // strip mode (im2col, single chunk, L <= 16): per strip of up to p_sp pixels of an output row,
    // each of the FH input rows is read once (p_segf bytes, p_segl for a row's last strip); every
    // pixel then takes one beat per filter row (L bytes) at an offset into the stream, and the
    // strip (p_dropf / p_dropl = FH x segment) is popped after its last pixel
    input  logic         p_sm,
    // (v28D) overlapped strips (p_smo, with p_sm): every pixel's window is K contiguous bytes, pixels
    // p_pixstep apart, so a strip is ONE read of p_segf / p_segl bytes; beat b of strip pixel i is
    // at i * p_pixstep + 16 b (beats of 16 bytes, the last c_lastn), the strip is dropped whole
    input  logic         p_smo,
    // (v28D) strip mode with implicit padding (p_spd, plain strips only): the strips are read from the
    // unpadded input (p_in = its padded origin; rows / bytes outside it are read as whatever lies there)
    // and the consumer replaces the window bytes outside the input by the zero point: whole beats of
    // window rows outside [0, p_hin) (beat c_b = filter row c_iy + c_b), the first p_padl bytes of a
    // row's first pixel (pair) and the last p_padr bytes of the beat of a row's last pixel (pair)
    input  logic         p_spd,
    input  logic [4:0]   p_padl, p_padr,
    input  logic         p_spt,        // (v58_DS) gather copy with transposed implicit padding (see r_spt.py)
    input  logic [7:0]   p_spt_t0, p_spt_nv, p_spt_pv,
    input  logic [2:0]   p_spt_l,
    // pixel pairs (strip mode, paired groups, 8 output channels: p_co = 16): two neighbouring pixels
    // of a strip per MAC issue, the first on slot A (even weight entries, acc), the second on slot B
    // (odd entries, acc2) with the same beat shifted by p_pixstep bytes (L + pixstep <= 16); weights,
    // biases, multipliers and shifts are read twice (entries / channels 8..15 = 0..7); results of a
    // pair are the two pixels' 8 bytes, written in order (one contiguous output)
    input  logic         p_pp,
    input  logic         p_v3,         // with p_v2 (transposed planar walk): three output rows per window
                                       // (5 input rows); the walk steps over row triples
    input  logic         p_v2,         // depthwise stride 1, 16-channel groups: two output rows per
                                       // window (4 input rows); the walk steps over row pairs
    input  logic [15:0]  p_orow,       // v2: output row stride (Wo * C)
    input  logic         p_dsp,        // depthwise C == 8, stride 1, pad 1: 16 lanes = 2 pixels x 8
                                       // channels (a "super pixel" of 16 bytes, W/2 columns)
    // 16-channel planes ([C/16][H][W][16]): a paired-group conv writes its output as planes
    // (p_pout: group g at p_out + g * npix * 16, pixels 16 bytes apart); a depthwise conv reads
    // planes walking the TRANSPOSED image (p_ptr): walk rows are input columns (16 bytes apart),
    // walk taps of a row p_colstep (W * 16) apart, the 3x3 filter applied transposed, group g's
    // plane at p_in + g * p_igrp, and the new column of a sliding window (3 or 4 contiguous taps)
    // read with one command; outputs go to their NHWC places, p_opix (Wo * C) apart along a walk
    // row and p_orow (C) between walk rows
    input  logic         p_pout, p_ptr,
    input  logic         p_rp,         // fused multi-chunk conv in column mode: the residual tensor is pair-swizzled
                                       // (A's and B's 8 bytes adjacent at obase(A) + p_rdelta + group offset): one
                                       // 16-byte residual read per pair, issued before the pair's taps
    input  logic         p_p24,        // (v28R) 24-bit partial sums (pixel pairs, see the ps staging)
    input  logic         p_rr,         // (v28R) residual in the partial-sum record (see the sc queue)
    // (v35R) row-pair layout: (y, x) at ((y >> 1) * 2W + 2x + 1 - (y & 1)) * 16 + group bytes (the rows of a pair
    // interleaved pixel by pixel, odd row first). p_rpw: a stride-1 column-mode dual-row conv writes it;
    // p_rpr: a cw2 conv reads it on the vertically flipped image (PIXSTEP / OPIX negative, filter rows flipped
    // at the weight load, top padding row = the bottom one): a pair's new walk columns are one 10-beat read
    input  logic         p_rpw, p_rpr,
    input  logic         p_rpx,        // (v35R) 1x1 stride-2 read of row-pair planes: write with the last read
    output logic         b10,          // (v35R) stream-0 bursts of up to 10 beats (p_rpr)
    input  logic         p_swz,        // paired-group conv writing a pair-swizzled residual: per pair one 32-byte
                                       // writes (pixel x odd: 8 bytes after x - 1's, the second 16 bytes on)
    input  logic         p_pin,        // sliced row walk over 16-channel planes: taps 16 bytes apart, slice k
                                       // at + k * p_igrp
    input  logic         p_po8,        // 8-channel groups write planes: group g at p_out + (g/2) * npix * 16
                                       // + (g%2) * 8, pixels 16 bytes apart
    input  logic [15:0]  p_colstep, p_igrp, p_opix,
    input  logic [7:0]   p_sp, p_segf, p_segl, p_dropf, p_dropl,
    input  logic         p_sm2,        // (v40K) packed strip beats (with p_sm, seg 18, K 27): 2 beats per pixel
    // global average pool (p_pa, a depthwise pass): 3x3 windows of stride 3 tile the input (taps past
    // its edge are zeros: in_off = 0), all-ones weights; every window's sums start from the previous
    // window's (res), and only the last window's leave (1 per cycle on unit A, whose multiplier is the
    // divisor's); one write per group, nothing advances the weight / parameter pointers
    input  logic         p_pa,
    // (v39S) global average pool fused into the drain (p_pf, RG_CP bit 30): the pass's requantised bytes are
    // summed per channel (not written); per pass the 16 sums leave through units A / B (npu.sv: RG_MO, shift 0
    // while pe_on) and go to p_out + 16 * group
    input  logic         p_pf,
    // (v46MB) mirror writes (plain conv, single chunk, one command per pixel): M1 = NHWC copy with a row gap
    // (start + p_m1d, + p_m1x after a row's last pixel), M2 = 16-channel planes copy (planes base + p_m2d)
    input  logic         p_m1, p_m2,
    input  logic [19:0]  p_m1d, p_m2d,
    input  logic [9:0]   p_m1x,
    output logic         wc_keep,      // the write command's bytes are written again by the next command
    input  logic         p_mf,         // (with M1) the PAD's border runs as fill commands in the first group pass
    input  logic [15:0]  p_mtl, p_mbl, // top run (bytes before pixel 0's M1 address) / the last row's run
    output logic         wc_fill,
    output logic         pe_on,
    input  logic         p_fzp,        // (v46RS) pair residual read (column mode, Co 16): A reads A + B (24 bytes)
    input  logic         p_fz2,        // (v46RS) fz2: the conv's own results are written too (write commands 'dup')
    output logic         wc_dup,
    input  logic         p_fz,         // fused ADD: results (1 per cycle, unit A) go to the add with the
    input  logic [31:0]  p_rdelta,     // residual bytes read on S1 at out + p_rdelta (single chunk)
    output logic [7:0]   fz_b, fz_b2,      // fused ADD: residual bytes of the results leaving units C / D
    input  logic [9:0]   p_ci,
    input  logic [1:0]   p_sw,
    input  logic [15:0]  p_fx0off,     // (3-sw)*Ci : first new column when the window slides
    input  logic [1:0]   p_pt, p_pl,   // depthwise implicit padding: top / left (p_in is the
    input  logic [7:0]   p_hin, p_win, // padded origin), and the real input height / width
    // stream S0: params, weights, activations
    output logic         rc0_valid,
    input  logic         rc0_ready,
    output logic [31:0]  rc0_addr,
    output logic [19:0]  rc0_len,
    input  logic [7:0]   rs0_avail,
    input  logic [127:0] rs0_data,
    output logic [7:0]   rs0_pop,
    output logic [7:0]   rs0_peek,     // strip mode: operand offset past the stream head
    // stream S1: partial sums in
    output logic         rc1_valid,
    input  logic         rc1_ready,
    output logic [31:0]  rc1_addr,
    output logic [19:0]  rc1_len,
    input  logic [7:0]   rs1_avail,
    input  logic [127:0] rs1_data,
    output logic [4:0]   rs1_pop,
    // write stream
    output logic         wc_valid,
    input  logic         wc_ready,
    output logic [31:0]  wc_addr,
    output logic [19:0]  wc_len,
    output logic         wc_tee,       // (v49_RS) an output write (I_OUTW, not DWF)
    output logic         wc_pm, wc_sk, // (v49_RS RS3) pair-merged fz2 writes (A / B of a column-mode pair)
    input  logic         p_pm,
    input  logic         p_tee,        // (v49_RS) the op's output writes are teed
    output logic         wd_valid,
    input  logic [7:0]   wd_space,     // free bytes in the write FIFO
    output logic [127:0] wd_data,
    output logic [4:0]   wd_n,
    input  logic         wr_idle,
    // requant units
    output logic         rqa_v, rqb_v, rqc_v, rqd_v,
    output logic [31:0]  rqa_acc, rqa_mult, rqb_acc, rqb_mult, rqc_acc, rqc_mult, rqd_acc, rqd_mult,
    output logic [7:0]   rqa_shift, rqb_shift, rqc_shift, rqd_shift,
    input  logic         rqa_ov, rqb_ov, rqc_ov, rqd_ov,
    input  logic [7:0]   rqa_q, rqb_q, rqc_q, rqd_q,
    // units E..H (npu_drq, conv only): a fast octet (results res_i .. res_i + 7, res_i a multiple
    // of 8, every channel but the first two C / D-capable) drains in one cycle on all eight units
    output logic         rqx_v,
    output logic [127:0] rqx_acc,
    output logic [123:0] rqx_mult,
    output logic [31:0]  rqx_shift,
    input  logic         rqx_ov,
    input  logic [31:0]  rqx_q,
    output logic         chain_ok,     // (v42DS) a compatible queued op may start now (npu.sv: chain_go)
    output logic         bg_yield,     // (v50_RS BGP) the final chunk pass of a multi-chunk conv runs (its outputs
                                       // and fused ADD use the ports): no background PAD copy pieces
    output logic         dwf_bad,      // (v51_D) a DWF depthwise multiplier / shift outside the fast-octet range
    output logic         busy
);
    localparam int KCH = KE * 16;
    wire       p_dwf     = DWF && p_dwf_r && p_pg_r && !p_dw && !p_fs;
    wire       p_dwt     = p_dwf && p_dwt_r;           // (v46DS) DWF tee
    wire       p_pft     = p_pf && p_dwt_r;            // (v46DS) pool tee
    logic      tq_hd;                                  // (v46DS) head piece of the tag queue: 1 = pointwise
    logic [19:0] tp_a;                                 // (v46DS) its address (pointwise stream)
    // (v37_1a) DWF signals (see npu_dwf and the drain / window sections)
    logic        dwc_v;                                    // write command of an output pending
    logic [19:0] dwc_a;                                    // its address
    logic        dwu_busy;                                 // npu_dwf still walks / drains the previous pass
    logic        dpar;                                     // drain slot parity: 0 the pass's results, 1 npu_dwf's
    logic        dwd, dwd_k;                               // npu_dwf drains an octet (channels 8 dwd_k ..)
    logic [8*22-1:0] dwd_acc;
    logic        dwo_st;                                   // npu_dwf starts an output (reserves 16 bytes)
    logic        t_pw, t_dw;                               // (v46DS) tee pieces complete
    logic        t_dmy;                                    // (v46DS) npu_dwf's dummy (tail) shift
    logic [127:0] t_data;                                  // (v46DS) the piece (octet 1, held octet 0)
    // (the depthwise strip walk for 8 channels is superseded by super pixels: tied off)
    wire       p_dsm     = 1'b0;
    // depthwise ignores the conv layout registers (they keep the previous conv's values)
    wire       p_contig  = p_contig_r && !p_dw && !p_fs;
    wire       p_sliced  = p_sliced_r && !p_dw && !p_fs;
    wire       p_pg      = p_pg_r && !p_fs;
    wire       p_dr      = p_dr_r && p_sliced_r && !p_dw && !p_fs && (p_sw == 2'd1 || (p_sw == 2'd2 && p_ptr));
    // dual-row with stride 2 (column mode only, cw2): A takes walk rows 0-2, B rows 2-4
    wire       dr2       = p_dr && (p_sw == 2'd2);
    wire [4:0] p_nchunks = (p_dw || p_fs) ? 5'd1 : p_nchunks_r;
    // FC stream (p_fs): the K-byte activation vector is loaded once into the lane weight banks
    // (beat b -> lane b/9, entry b%9; K <= 720 uses lanes 0..4), then the weight rows of 16-channel
    // groups stream past once, one 16-byte beat per cycle: lane b/9 multiplies the beat with its
    // stored activations, lanes 5 and 6 multiply it with constants c1 + c2 = in_off (the offset
    // term in_off * sum(w)), and the three dot products of a beat add into one accumulator. One
    // output channel per K/16 beats; per-channel bias / mult / shift per group of 16 as usual.
    wire [5:0] fs_kbt = 6'((p_k[9:0] + 10'd15) >> 4);      // beats per weight row (last partial)
    wire [4:0] fs_lastn = (p_k[3:0] == 4'd0) ? 5'd16 : {1'b0, p_k[3:0]};
    wire [7:0] fs_c1  = p_inoff[8:1];                      // in_off >>> 1 (in_off in [-128, 128])
    wire [7:0] fs_c2  = p_inoff[7:0] - p_inoff[8:1];
    wire       pk     = p_pk && p_fs;
    wire       fsa    = p_fsa && p_fs;                    // (v34P)
    wire       fs_raw = p_raw && p_fs;

    // ------------------------------------------------------------------ geometry
    wire        slw     = p_sliced || p_dw;                  // 3x3 window walk
    // (v28A) the window slides at stride 1, and at stride 2 except on the plain depthwise walk (no v2 /
    // transposed walk): that one re-reads all three columns, so window column 0 never takes column 2
    wire        sl12    = (p_sw == 2'd1) || ((p_sw == 2'd2) && (!p_dw || p_v2));
    wire        g16     = p_dw || p_pg || p_fs;              // 16 output channels per group
    wire [7:0]  ngroups0 = g16 ? ((p_co + 16'd15) >> 4) : ((p_co + 16'd7) >> 3);
    wire [7:0]  ngroups  = ((p_gl != 4'd0) && ({4'd0, p_gl} < ngroups0)) ? {4'd0, p_gl} : ngroups0;   // (v55_KC)
    wire [4:0]  gc_last = g16 ? ((p_co[3:0] == 4'd0) ? 5'd16 : {1'b0, p_co[3:0]})
                              : ((p_co[2:0] == 3'd0) ? 5'd8  : {2'b0, p_co[2:0]});
    wire [4:0]  gsize   = g16 ? 5'd16 : 5'd8;
    wire [4:0]  nchunks = p_nchunks;
    // fused ADD over several chunk passes (sliced walk): S1 carries the partial sums, so in the last
    // pass every pixel's residual bytes follow its taps on S0 (popped into the sc queue)
    wire        fz_s0   = p_fz && (nchunks != 8'd1);
    wire owhole = ((p_co <= {11'd0, gsize}) && !p_v2 && !p_dr) || p_fs;   // all outputs form one contiguous range
    wire dbl    = p_dsp || p_pp;                           // per-channel words / taps read twice
    wire pin    = p_pin && p_sliced && !p_ptr;
    wire [9:0] ci_a = pin ? 10'd16 : p_ci;                    // activation tap stride (row walk)
    wire rowc   = (p_sliced && (p_ci == 16'd16 || pin) && !p_ptr) || (p_dw && p_ci <= 16'd16 && !p_ptr); // a row's new taps: one read
    // column mode (p_ptr on a sliced conv): the conv runs on the TRANSPOSED image (walk rows are input
    // columns 16 bytes apart, walk taps p_colstep apart, weights loaded transposed); a sliding pixel's new
    // walk column (3 taps, 4 for a dual-row pair) is contiguous and read with one command
    wire scw    = p_sliced && p_ptr;
    // stride-2 transposed depthwise walk (s2t, v22U s12): two output rows of a walk row group on window rows 0-2 / 2-4
    wire s2t    = p_v2 && (p_sw == 2'd2);

    // ================================================================== issuer
    typedef enum logic [3:0] {I_IDLE, I_BIAS, I_MULT, I_SHIFT, I_WGT, I_BAR, I_PSR, I_PSW, I_OWH,
                              I_PIX, I_OUTW, I_PIXNEXT, I_PASSNEXT, I_PFW, I_CHW, I_OUTM} ist_t;
    logic [1:0]  i_ms;                                     // (v46MB) I_OUTM: 0 M1, 1 M2, 2 top fill, 3 row fill
    logic [19:0] i_m1, i_m2;                               // (v46MB) mirror addresses of the current pixel
    wire         mir = p_m1 || p_m2;
    wire         i_mft = p_m1 && p_mf && (i_g == 8'd0) && (i_px == 12'd0) && (p_mtl != 16'd0);   // top run due
    wire         i_mfr = p_m1 && p_mf && (i_g == 8'd0) && i_rend;                             // row run due
    logic        chn;                                      // (v42DS) a chained op runs while the old op's tail completes
    ist_t        ist;
    logic        i_dl;                                     // (v37_1a) DWF: the pass's depthwise header (I_WGT .. I_SHIFT)
    logic [7:0]  i_g, i_oy, i_ox, i_fy;
    logic [11:0] i_px;
    logic [7:0]  i_sp;                                   // strip mode: pixel within the strip
    wire  [7:0]  i_seg   = ((p_wo - i_ox) <= {8'd0, p_sp}) ? p_segl : p_segf;
    wire         i_spend = (i_sp == p_sp - 8'd1) || (i_ox == p_wo - 16'd1);
    // depthwise strip: valid columns of the strip starting at this pixel
    wire  [2:0]  d_cs    = p_ci[4] ? 3'd4 : 3'd3;
    wire  [15:0] i_dnp   = ((p_wo - i_ox) < {8'd0, p_sp}) ? (p_wo - i_ox) : {8'd0, p_sp};
    wire  signed [16:0] i_dclo = (i_ix < 0) ? 17'sd0 : i_ix;
    wire  signed [16:0] i_dch0 = i_ix + $signed({1'b0, i_dnp}) + 17'sd1;
    wire  signed [16:0] i_dchi = (i_dch0 > $signed({1'b0, p_win}) - 17'sd1) ? ($signed({1'b0, p_win}) - 17'sd1) : i_dch0;
    wire  [7:0]  i_dseg  = 8'(i_dchi - i_dclo + 17'sd1) << d_cs;
    wire  [7:0]  i_dskip = 8'(i_dclo - i_ix) << d_cs;
    logic [4:0]  i_k;
    logic [3:0]  i_j;
    // addresses are DRAM offsets: the low 20 bits (the DMA engines supply the fixed upper bits)
    logic [19:0] i_rowbase, i_pbase, i_ptr, i_obase, i_wg, i_wl;
    // (v26A: i_ptr also serves as the 3x3 walk's row tap (rowtap = ptr + ioff2); i_wl is the one weight
    //  pointer: chunk lanes (K apart), sliced taps (Ci apart: K = 9 Ci, so lane j's tap 8 + Ci = lane j+1's
    //  tap 0) and the pre-arranged blocks; i_ig is the depthwise input group and the column-mode slice offset)
    logic [3:0]  i_t;                                    // tap (sliced weight load)
    logic [19:0] i_tap_s;                                // sliced: address of the next tap read
    logic        i_h;                                    // second command of a pair (dsp / v2)
    // (v25M) v3: when the walk row group's third output row does not exist, window row 4 is neither read nor walked
    wire         i_r4    = !p_v3 || ((i_oy + 16'd2) < p_ho);
    wire         i_hvl;                                  // v3: the position's last output row write
    wire  [15:0] i_lfy   = (p_v2 || p_dr) ? ((p_v3 || s2t) ? (i_r4 ? 16'd4 : 16'd3) : 16'd3) : 16'd2;  // last window row of the 3x3 walk
    logic [1:0]  i_hv;                                   // v3: output row of the position's write
    wire  [15:0] i_ostep = p_v3 ? 16'd3 : 16'd2;         // v2 / v3 / dual-row: output rows per walk row
    wire  [15:0] i_istep = p_v3 ? 16'd3 : s2t ? 16'd4 : 16'd2;   // input rows per walk row group
    wire  [19:0] i_ovs   = p_v3 ? ({3'd0, p_orow, 1'b0} + {4'd0, p_orow}) : {3'd0, p_orow, 1'b0};
    logic [19:0] i_ig, i_ogb, i_orb;                     // input group / output group / output row bases
    logic [19:0] i_ob2;                                  // (v35R) rpw: output position base (row-pair layout)
    // (v35R) rpw: i_ob2's step at a walk row's end (row Ho - 1 -> row 0 two columns on): 80 - (Ho - 2) W 16 (npix = Ho W, OPIX = W 16)
    wire  [19:0] rp_k = 20'd80 - {4'd0, p_npix, 4'd0} + {3'd0, p_opix, 1'b0};
    logic        i_pre;                                  // p_rp: the pair's first write command + residual read, before its taps
    wire  [19:0] i_rgo   = p_po8 ? {16'd0, i_g[0], 3'd0} : {9'd0, i_g, 3'd0};   // the output's group byte offset
    wire  [19:0] i_ioff  = (p_dw || scw || pin) ? i_ig : {7'd0, i_cs};
    wire  [19:0] ioff2   = slw ? i_ioff : 20'd0;
    wire  [19:0] i_cstep = p_ptr ? p_colstep : {10'd0, ci_a};
    wire         dw_wrap = (p_dw && (p_v2 || p_ptr)) || p_dr || p_smo;   // output rows are not contiguous
    wire  [19:0] i_opix  = p_pa ? 20'd0 : p_ptr ? {(p_rpr ? {4{p_opix[15]}} : 4'd0), p_opix} : (p_pout || p_po8) ? 20'd16 : {4'd0, p_co};
    wire  [19:0] i_po8b  = i_ogb + (i_g[0] ? 20'd8 : 20'd0);   // p_po8: the group's base
    wire         i_bok   = (i_oy + 16'd1) < p_ho;            // v2: the pair's second row exists
    assign       i_hvl   = (i_hv == 2'd2) || (i_hv == 2'd1 && !i_r4) || (i_hv == 2'd0 && !i_bok);
    wire  [15:0] i_pxn   = i_px + ((p_dr && i_bok) ? 16'd2 : 16'd1);   // pixels done after this one / pair
    logic [1:0]  i_fx;
    wire         i_slide = slw && sl12 && (i_ox != 16'd0);
    wire  [15:0] i_fx0   = i_slide ? (16'd3 - p_sw) : 16'd0;
    wire         sl_multi = p_sliced && (nchunks != 8'd1);
    wire         wsk      = sl_multi && !p_pre && (p_ci == 10'd32);   // (v58_RS) one long weight read per pass (Ci 32)
    wire         prei     = p_pre || wsk;                           // (v58_RS) one weight command per pass
    wire         w8       = sl_multi && !prei && p_w8;              // (v58_RS) rotated slices
    wire         w8l      = w8 && (i_k == nchunks - 8'd1);          // ... the wrap slice: line-wise reads, one extra
    wire         w8m0     = (i_t == 4'd0) && (i_j == 4'd0) && !i_h; // ... its first command (8 bytes)
    wire         w8n      = (i_t == 4'd8) && (i_j == i_gc[3:0] - 4'd1);   // the pass's 72nd piece
    logic [12:0] i_cs;
    wire  [4:0]  i_gc   = (i_g == ngroups - 16'd1) ? gc_last : gsize;
    wire  [19:0] i_ce   = ((i_cs + 20'(KCH)) > 20'(p_k)) ? 20'(p_k) : (i_cs + 20'(KCH));
    wire  [19:0] i_ck   = i_ce - i_cs;
    wire         i_last = (i_k == nchunks - 8'd1);
    wire         one_act = p_contig && (nchunks == 8'd1);   // one command for every pixel's activations
    // a single output pixel with several chunks (FC, K > 144): the lanes keep accumulating across
    // chunk passes, so there are no partial sums to park
    // (off: a single-pixel multi-chunk conv runs as ordinary chunk passes with partial sums, so
    // every chunk's input-offset correction enters through the accumulator start value)
    wire         onepix  = 1'b0;
    assign s1_unused = ((nchunks == 8'd1) || onepix) && !p_fz && !fsa;   // (v34P: fsa reads S1)
    wire  [1:0]  i_ncol = 2'(16'd3 - i_fx);                // columns left in this row (row read)
    // depthwise implicit padding: input row / column of the window origin; taps outside the
    // input are not read (the consumer feeds zeros, i.e. the zero point after the offset)
    logic signed [9:0]  i_iy, i_ix;
    wire  signed [16:0] i_ry = i_iy + $signed({1'b0, i_fy});
    wire         i_rv = !slw || ((i_ry >= 17'sd0) && (i_ry < $signed({1'b0, p_hin})));
    // pool: the input may be walked as rows of 3 pixels whose last row holds only p_fx0off[1:0]
    // (1 or 2) of them (RG_FX0OFF is free in pool mode: no sliding; v15A2 s7)
    wire         pa_lr  = p_pa && (p_fx0off[1:0] != 2'd0);
    wire         i_lrow = (i_ry[7:0] == p_hin - 8'd1);
    logic [2:0]  i_cv;
    always_comb for (int c = 0; c < 3; c++)
        i_cv[c] = !slw || (((i_ix + 17'(c)) >= 17'sd0) && ((i_ix + 17'(c)) < $signed({1'b0, p_win})) &&
                           !(pa_lr && i_lrow && (2'(c) >= p_fx0off[1:0])));
    wire         i_tv = i_rv && i_cv[i_fx[1:0]];
    // row read (new columns i_fx..2): clip to the valid columns
    wire         rv0 = i_cv[0] && (i_fx == 16'd0), rv1 = i_cv[1] && (i_fx <= 16'd1), rv2 = i_cv[2];
    // column mode, last pass of a fused multi-chunk conv: the residual of A comes before B's row-3 tap,
    // so the column read covers rows 0-2 and row 3's tap is read on its own after A's write command
    wire         rpi     = p_rp && fz_s0 && scw && p_dr && i_last && !p_rr;   // paired residual, this pass
    wire         cw3     = fz_s0 && i_last && scw && !p_rp;
    // planar transposed walk: rows of the new (contiguous) column inside the input
    // column mode, stride 2 (dual-row: A walk rows 0-2, B rows 2-4): every pixel pair reads its new walk
    // columns (1, 2; the first pixel of a walk row also column 0) as 5-tap column reads
    // (depthwise s2t: a sliding position reads its two new columns (1, 2) as 5-tap column reads; v3: 5 rows)
    wire         cw2i    = scw && dr2;
    wire         rpr     = p_rpr && cw2i;                  // (v35R) row-pair read (see the port)
    wire         rpw     = p_rpw && scw && p_dr && !dr2;   // (v35R) row-pair write
    assign       b10     = rpr;
    wire         s2tc    = s2t && (i_fx0 != 16'd0);
    // (v25M) depthwise transposed walk (planes, v2 / v3 / s2t): a position's every new column, also
    // the three of a walk row's first position, is one column read (walked column by column)
    wire         dwc     = p_dw && p_ptr && p_v2;
    logic [4:0]  i_crv;
    always_comb for (int f = 0; f < 5; f++)
        i_crv[f] = ((i_iy + 17'(f)) >= 17'sd0) && ((i_iy + 17'(f)) < $signed({1'b0, p_hin})) &&
                   (f < 3 || (f == 3 && (p_v2 || (p_dr && i_bok && !cw3))) || (f == 4 && ((cw2i && i_bok) || (p_v3 && i_r4) || s2t)));
    wire         i_colrd = p_ptr && (i_fy == 16'd0) && ((i_fx0 == 16'd2) || cw2i || s2tc || dwc);
    wire  [2:0]  cr_lo  = i_crv[0] ? 3'd0 : i_crv[1] ? 3'd1 : i_crv[2] ? 3'd2 : i_crv[3] ? 3'd3 : 3'd4;
    wire  [2:0]  cr_hi  = i_crv[4] ? 3'd4 : i_crv[3] ? 3'd3 : i_crv[2] ? 3'd2 : i_crv[1] ? 3'd1 : 3'd0;
    wire  [2:0]  cr_n   = cr_hi - cr_lo + 3'd1;          // (v35R) taps of the column read (1..5)
    wire         cr_any = (i_crv != 5'd0) && ((cw2i || s2t || dwc) ? i_cv[i_fx] : i_cv[2]);   // (v24M: i_cv[2] outside cw2: the conv8 pool's column reads)
    wire  [1:0]  r_lo = rv0 ? 2'd0 : rv1 ? 2'd1 : 2'd2;
    wire  [1:0]  r_hi = rv2 ? 2'd2 : rv1 ? 2'd1 : 2'd0;
    wire         r_any = i_rv && (rv0 || rv1 || rv2);
    // (v25M) pool over contiguous rows of 3 pixels (rowc): a window whose 3 rows are all inside the
    // input and full is one read of 9 pixels (its rows are consecutive: ROWSTR = 3 pixels)
    wire         pa_w3 = p_pa && rowc && (i_fy == 16'd0) && (i_fx == 2'd0) && (i_iy >= 10'sd0) &&
                         ((i_iy + 17'sd3) <= $signed({1'b0, p_hin})) && rv0 && rv1 && rv2 &&
                         !(pa_lr && ((i_iy + 17'sd3) >= $signed({1'b0, p_hin})));
    wire  [1:0]  r_skip = r_lo - i_fx[1:0];
    wire  [1:0]  r_ncol = r_hi - r_lo + 2'd1;
    wire  [19:0] r_len  = (r_ncol == 2'd3) ? (20'(ci_a) + {9'd0, ci_a, 1'b0}) :
                          (r_ncol == 2'd2) ? {9'd0, ci_a, 1'b0} : 20'(ci_a);
    wire  [31:0] r_off  = (r_skip == 2'd2) ? {21'd0, ci_a, 1'b0} : (r_skip == 2'd1) ? {22'd0, ci_a} : 32'd0;
    // byte offset of group i_g's parameter words: 8 or 16 channels of 4 bytes
    // pool: the all-ones weights are loaded for the first group only
    wire  pa_iw = p_pa && (i_g != 8'd0);
    // (v47c AG) line-aligned output channel groups (RG_PG bit 5): groups 0..n-2 = channels ag_d + 16 g .. + 15 (one
    // aligned line per pixel), the last (wrap) group = channels 0..ag_d-1 (lanes 0..ag_d-1) and Co-16+ag_d..Co-1;
    // its parameter / weight reads are two commands (i_sg: the second); one aligned 16-byte write per pixel (the
    // previous pixel's high piece + this pixel's low piece), the first low / last high piece on their own
    wire  [3:0]  ag_d   = 4'd0 - p_out[3:0];                              // 4, 8 or 12 (16 - ag_d = p_out[3:0])
    wire         rot    = p_rot_r && (p_out[1:0] == 2'd0) && (p_out[3:2] != 2'd0) && p_pg && !p_dw && !p_fs && !p_cp &&
                          !p_sliced && (nchunks == 5'd1) && (p_co[3:0] == 4'd0) && (p_co[9:5] != 5'd0) && !p_fz && !p_pf &&
                          !p_pout && !p_po8 && !p_pp && !p_dsp && !p_v2 && !p_dr && !p_swz && !p_ptr && !p_smo && !p_rpw &&
                          !p_rpx && !p_rp && !p_rr && !p_p24 &&
                          (!p_dwf || (p_dwt && (p_out2[3:0] == p_out[3:0]) && ag_d[2] && (ag_d[1:0] == 2'd0)));   // (v49DS) tee
    wire         ag_tw  = rot && p_dwf;                                     // (v49DS) a rotated tee
    wire  [19:0] ag_wo  = {10'd0, p_co} - 20'd16 + {16'd0, ag_d};           // (v49DS) wrap: high piece offset in a pixel
    wire  [7:0]  ag_d9  = {1'b0, ag_d, 3'd0} + {4'd0, ag_d};                // (v49DS) wrap: depthwise tap bytes of the low piece
    logic        ag_hw, t_h0, t_hf, ps_w, fl_go;                           // (v49DS) tee wrap piece / carry (see t_unit)
    logic [11:0] ps_i;
    logic [127:0] t_unit;
    wire         ag_wi  = rot && (i_g == ngroups - 8'd1);                  // issuer: the wrap group
    logic        i_sg;                                                     // wrap: second command of a split
    wire  [16:0] ag_k4  = {2'd0, p_k, 2'd0};
    wire  [16:0] ag_k8  = {1'd0, p_k, 3'd0};
    wire  [16:0] ag_k12 = ag_k4 + ag_k8;
    wire  [16:0] ag_lo  = (ag_d[3:2] == 2'd1) ? ag_k4 : (ag_d[3:2] == 2'd2) ? ag_k8 : ag_k12;           // ag_d K
    wire  [16:0] ag_hi  = (p_out[3:2] == 2'd1) ? ag_k4 : (p_out[3:2] == 2'd2) ? ag_k8 : ag_k12;        // (16 - ag_d) K
    wire  [31:0] i_pofs = p_pa ? 32'd0 : rot ? ((ag_wi && !i_sg) ? 32'd0 : {18'd0, i_g, ag_d, 2'd0}) :
                          g16 ? {10'd0, i_g, 6'd0} : {11'd0, i_g, 5'd0};

    // v26A: the next pixel's bases (I_PIXNEXT)
    wire         i_rend = (i_ox == p_wo - 16'd1);
    // (v56_AR p_hadd) fa_1 ripple adders (h_rca) for register updates
    wire [19:0] hb_rb; wire hb_rb_co;
    h_rca #(20) u_hb_rb (.a(20'(i_rowbase)), .b(20'({4'd0, p_rowstep})), .ci(1'b0), .s(hb_rb), .co(hb_rb_co));
    wire [19:0] hb_pb; wire hb_pb_co;
    h_rca #(20) u_hb_pb (.a(20'(i_pbase)), .b(20'({(p_rpr ? {4{p_pixstep[15]}} : 4'd0), p_pixstep})), .ci(1'b0), .s(hb_pb), .co(hb_pb_co));
    wire  [19:0] i_nb   = i_rend ? hb_rb : hb_pb;
    // (v56_AR p_hadd) fa_1 ripple adders (h_rca) for register updates
    wire [19:0] hb_nt; wire hb_nt_co;
    h_rca #(20) u_hb_nt (.a(20'(i_nb)), .b(20'(ioff2)), .ci(1'b0), .s(hb_nt), .co(hb_nt_co));
    wire  [19:0] i_nt   = hb_nt;
    wire         i_nsl  = !i_rend && slw && sl12;
    logic bar_ok;
    logic rc1_raw, wc_raw, rc0_raw;
    always_comb begin
        rc0_raw = 1'b0; rc0_addr = '0; rc0_len = '0;
        rc1_raw = 1'b0; rc1_addr = p_scr; rc1_len = (p_p24 && !p_rr) ? ({4'd0, p_npix, 4'd0} + {5'd0, p_npix, 3'd0}) : {3'd0, p_npix, 5'd0};
        wc_raw  = 1'b0; wc_addr = '0; wc_len = '0; wc_tee = 1'b0;
        case (ist)
            // dsp: the 8 channels' words are read twice (results 0-7 and 8-15)
            I_BIAS:  begin rc0_raw = p_has_bias || i_dl; rc0_addr = (i_dl ? {12'd0, p_dbias} : p_bias) + i_pofs; rc0_len = dbl ? 20'd32 : {13'd0, i_gc, 2'd0}; end
            I_MULT:  begin rc0_raw = 1'b1;       rc0_addr = (i_dl ? {12'd0, p_dmult} : p_mult) + i_pofs; rc0_len = dbl ? 20'd32 : {13'd0, i_gc, 2'd0}; end
            I_SHIFT: begin rc0_raw = 1'b1;       rc0_addr = (i_dl ? {12'd0, p_dshift} : p_shift) + i_pofs; rc0_len = dbl ? 20'd32 : {13'd0, i_gc, 2'd0}; end
            I_WGT: begin
                rc0_raw = !p_cp && !pa_iw;
                if (i_dl) begin                                         // (v37_1a) the depthwise weights
                    rc0_addr = {12'd0, p_dwgt} + {17'd0, i_g, 7'd0} + {20'd0, i_g, 4'd0} + (rot ? {24'd0, ag_d9} : 32'd0);
                    rc0_len  = 20'd144;
                end else if (p_fs) begin
                    rc0_addr = p_in;                                    // the activation vector
                    rc0_len  = 20'(p_k);
                end else if (p_dw) begin
                    rc0_addr = i_wg;
                    rc0_len  = dbl ? 20'd72 : ({12'd0, i_gc, 3'd0} + {15'd0, i_gc});   // 9 bytes per channel
                end else if (nchunks == 8'd1) begin
                    rc0_addr = i_wg;
                    rc0_len  = (i_g == ngroups - 16'd1) ? p_wglast : (p_pg ? {p_k, 4'd0} : {1'b0, p_k, 3'd0});
                end else if (p_sliced && p_pre) begin                 // one read per pass
                    rc0_addr = i_wl;
                    rc0_len  = {8'd0, i_gc, 7'd0} + {11'd0, i_gc, 4'd0};   // gc x 144 bytes
                end else if (wsk) begin                               // (v58_RS) the group's filter rows from slice c on
                    rc0_addr = i_wl;
                    rc0_len  = {7'd0, i_gc, 8'd0} + {10'd0, i_gc, 5'd0} - 20'd16;   // gc x 288 - 16 bytes
                end else if (p_sliced) begin
                    rc0_addr = !w8 ? {12'd0, i_wl} : !w8l ? {12'd0, i_wl + 20'd8} :
                               {12'd0, i_wl - {10'd0, p_ci} + (w8m0 ? 20'd16 : 20'd8)};   // (v58_RS w8)
                    rc0_len  = (w8l && (w8m0 || i_h)) ? 20'd8 : 20'd16;
                end else begin
                    rc0_addr = i_wl;
                    rc0_len  = i_ck;
                end
            end
            I_PSR: rc1_raw = 1'b1;
            I_PSW: begin wc_raw = 1'b1; wc_addr = p_scr; wc_len = (p_p24 && !p_rr) ? ({4'd0, p_npix, 4'd0} + {5'd0, p_npix, 3'd0}) : {3'd0, p_npix, 5'd0}; end
            I_OWH: begin                                            // (fs: one command per group)
                wc_raw = 1'b1; wc_addr = (p_fs && !pk) ? (p_out + {12'd0, i_g, 4'd0}) : p_out;
                wc_len = (p_fs && !pk) ? {15'd0, i_gc} : fs_raw ? {8'd0, p_co, 2'b00} : p_outlen;
                rc1_raw = p_fz && !fz_s0; rc1_addr = p_out + p_rdelta; rc1_len = p_outlen;
                if (fsa) begin rc1_raw = 1'b1; rc1_addr = p_scr; rc1_len = {8'd0, p_co, 2'b00}; end
            end
            I_PIX: begin
                if (p_fs) begin                                         // the group's weight rows
                    rc0_raw = 1'b1;
                    rc0_addr  = (pk && i_h) ? {12'd0, p_wg2} : i_wg;
                    rc0_len   = pk ? (i_h ? p_len2 : p_actlen) : (i_g == ngroups - 16'd1) ? p_wglast : {p_k, 4'd0};
                end else if (p_contig) begin
                    rc0_raw = 1'b1;
                    rc0_addr  = one_act ? p_in : {11'd0, hb_pbc_co, hb_pbc};
                    rc0_len   = one_act ? p_actlen : i_ck;
                end else if (p_sm) begin
                    rc0_raw = 1'b1; rc0_addr = i_ptr; rc0_len = {12'd0, i_seg};
                end else if (p_dsm) begin
                    rc0_raw = i_rv; rc0_addr = i_ptr + {24'd0, i_dskip}; rc0_len = {12'd0, i_dseg};
                end else if (i_colrd) begin
                    rc0_raw = cr_any;
                    rc0_addr  = i_tap_s + (rpr ? {12'd0, cr_lo, 5'd0} : {13'd0, cr_lo, 4'd0});
                    rc0_len   = rpr ? {12'd0, cr_n, 5'd0} : {13'd0, cr_n, 4'd0};
                end else if (slw) begin
                    rc0_raw = rowc ? r_any : i_tv;
                    rc0_addr  = rowc ? (i_tap_s + r_off) : i_tap_s;
                    rc0_len   = pa_w3 ? ({r_len[18:0], 1'b0} + r_len) : rowc ? r_len : (p_dw ? {15'd0, i_gc} : 20'd16);
                end else begin
                    rc0_raw = 1'b1;                     // (single chunk: one row piece of L bytes)
                    rc0_addr  = i_ptr;
                    rc0_len   = {7'd0, p_l};
                    if (p_rpx && p_swz) begin            // (v35R) the pair's swizzled write with its first pixel's last read
                        wc_len  = 20'd32;
                        wc_addr = i_obase + (p_pout ? 20'd0 : {8'd0, i_g, 4'd0});
                        wc_raw  = !i_ox[0] && i_last && (i_fy == p_fh - 16'd1);
                    end
                end
            end
            I_OUTW: begin
                wc_raw = i_last && !owhole && (!p_pf || p_pft) && (p_v3 ? ((i_oy + {6'd0, i_hv}) < p_ho) : (!i_h || i_bok || p_swz)) && (!p_pa || i_px == p_npix - 12'd1) &&
                         !p_dwf;                                // (v37_1a) DWF: npu_dwf's outputs, own commands
                wc_addr = p_v3 ? (i_obase + ((i_hv == 2'd2) ? {3'd0, p_orow, 1'b0} : (i_hv == 2'd1) ? {4'd0, p_orow} : 20'd0)) :
                          rpw ? (i_h ? (i_ob2 + 20'd32) : i_ob2) : i_h ? {11'd0, hb_obo_co, hb_obo} : {12'd0, i_obase};
                wc_len = {15'd0, i_gc};
                wc_tee = !p_dwf && !p_swz;           // (v49_RS)
                if (p_swz) begin                       // pair-swizzled residual: one 32-byte write per pair (v28R)
                    wc_raw  = i_last && !i_ox[0];
                    wc_addr = i_obase + (p_pout ? 20'd0 : {8'd0, i_g, 4'd0});   // (NHWC: 16-channel group p at 32 p in the pair)
                    wc_len  = 20'd32;
                end
                rc1_raw = p_fz && !fz_s0 && i_last && !owhole; rc1_addr = (i_h ? {11'd0, hb_obo_co, hb_obo} : {12'd0, i_obase}) + p_rdelta;
                rc1_len = {15'd0, i_gc};
                if (p_fzp) begin rc1_raw = rc1_raw && !i_h; rc1_len = 20'd24; end   // (v46RS) A: A, 8 skipped, B
                rc0_raw = fz_s0 && i_last && (!rpi || i_pre) && !p_rr; rc0_addr = (i_h ? {11'd0, hb_obo_co, hb_obo} : {12'd0, i_obase}) + p_rdelta;
                rc0_len = {15'd0, i_gc};
                if (rpi) begin rc0_addr = i_obase + p_rdelta + i_rgo; rc0_len = 20'd16; end
                // (v28R) rr: the pair's residual (16 bytes) on stream 1 in pass k0, at A's write state
                if (p_rr && !i_last && !i_h) begin rc1_raw = 1'b1; rc1_addr = i_obase + p_rdelta + i_rgo; rc1_len = 20'd16; end
            end
            I_PFW: begin wc_raw = 1'b1; wc_addr = p_out + {12'd0, i_g, 4'd0}; wc_len = 20'd16; end   // (v39S) pooled bytes
            I_OUTM: begin                                       // (v46MB) mirrors / border fills
                wc_raw = 1'b1;
                case (i_ms)
                    2'd0: begin wc_addr = {12'd0, i_m1}; wc_len = {15'd0, i_gc}; end
                    2'd1: begin wc_addr = {12'd0, i_m2}; wc_len = {15'd0, i_gc}; end
                    2'd2: begin wc_addr = {12'd0, hb_m1t}; wc_len = {4'd0, p_mtl}; end
                    default: begin wc_addr = {12'd0, i_m1 + {10'd0, p_co}};
                                   wc_len = ((i_oy + 8'd1) == p_ho) ? {4'd0, p_mbl} : {10'd0, p_m1x}; end
                endcase
            end
            default: ;
        endcase
        // (v37_1a) npu_dwf's output commands
        wc_addr = p_dwf ? {12'd0, ((p_dwt && tq_hd) ? tp_a : dwc_a) - (ag_hw ? (t_h0 ? ag_wo : {10'd0, p_co}) : 20'd0)} : wc_addr;   // (v46DS) tee: the head piece's stream
        wc_len  = p_dwf ? (ag_hw ? (t_h0 ? {16'd0, ag_d} : t_hf ? (20'd16 - {16'd0, ag_d}) : 20'd16) : 20'd16) : wc_len;   // (v49DS) wrap
        if (ag_wi) begin                                        // (v47c AG) the wrap group
            case (ist)
                I_BIAS, I_MULT, I_SHIFT: rc0_len = i_sg ? {14'd0, p_out[3:0], 2'd0} : {14'd0, ag_d, 2'd0};
                I_WGT: if (i_dl) begin                          // (v49DS) tee: the depthwise weights
                    rc0_addr = i_sg ? rc0_addr : {12'd0, p_dwgt};
                    rc0_len  = i_sg ? (20'd144 - {12'd0, ag_d9}) : {12'd0, ag_d9};
                end else begin
                    rc0_addr = i_sg ? rc0_addr : p_wgt;
                    rc0_len  = i_sg ? {3'd0, ag_hi} : {3'd0, ag_lo};
                end
                I_OUTW: if (i_px == 12'd0 && !p_dwf) begin                // pixel 0: its low piece (i_obase = out's line)
                    wc_addr = {12'd0, i_obase[19:4], p_out[3:0]};
                    wc_len  = {16'd0, ag_d};
                end
                I_PASSNEXT: if (!p_dwf) begin                   // the last pixel's high piece
                    wc_raw  = !i_sg;
                    wc_addr = {12'd0, i_obase};
                    wc_len  = {16'd0, p_out[3:0]};
                end
                default: ;
            endcase
        end
    end
    // a fused ADD's residual read and its write command are handed over together
    // (fz_s0: the residual read on S0 and the write command, likewise)
    wire rc0_pair = (ist == I_OUTW) || ((ist == I_PIX) && p_rpx);
    assign wc_dup    = p_fz2 && (ist == I_OUTW);               // (v46RS) output writes of a fz2 op
    // (v49_RS RS3 pm) fz2, Co 16, column mode: in every group but the last, A writes the pair, B drops
    wire   pm_g      = p_pm && p_fz2 && (ist == I_OUTW) && (i_g != {8'd0, ngroups} - 16'd1);
    assign wc_pm     = pm_g && !i_h && i_bok;
    assign wc_sk     = pm_g && i_h;
    assign wc_keep = (ist == I_OUTW) ? mir : (ist == I_OUTM) && (i_ms == 2'd0) && p_m2;   // (v46MB)
    assign wc_fill = (ist == I_OUTM) && i_ms[1];
    assign wc_valid  = p_dwf ? dwc_v : (wc_raw && (!rc1_raw || rc1_ready) && (!(rc0_raw && rc0_pair) || rc0_ready));
    assign rc1_valid = rc1_raw && (!wc_raw || wc_ready);
    assign rc0_valid = rc0_raw && (!(rc0_pair && wc_raw) || wc_ready);
    wire i_adv0 = (!rc0_raw || rc0_ready) && (!rc1_raw || rc1_ready) && (!wc_raw || wc_ready);
    // (v47c AG) wrap group: the first command of a split read holds the state; I_PASSNEXT waits for the final high piece
    wire ag_sg = ag_wi && !i_sg &&
                 ((((ist == I_BIAS) || (ist == I_MULT) || (ist == I_SHIFT) || (ist == I_WGT)) && rc0_raw) ||
                  ((ist == I_PASSNEXT) && !p_dwf));
    wire i_adv = i_adv0 && !ag_sg;
    always_ff @(posedge clk or negedge rst_n)
        if (!rst_n) i_sg <= 1'b0; else if (start) i_sg <= 1'b0; else if (i_adv0) i_sg <= ag_sg;
    // state after a pass's weights (and, in its first chunk, the group's parameters)
    wire  fu_i = (p_fs && p_fu && (i_g != 16'd0)) || p_pa; // uniform / pool: no multiplier / shift reads
    ist_t i_nxt;
    always_comb begin
        if (p_fs)             i_nxt = I_OWH;
        else if (onepix)      i_nxt = (i_last && owhole) ? I_OWH : I_PIX;
        else if (!i_last)     i_nxt = I_PSW;
        else if (owhole)      i_nxt = I_OWH;
        else                  i_nxt = I_PIX;
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ist <= I_IDLE; i_g <= '0; i_k <= '0; i_j <= '0; i_dl <= 1'b0; i_ms <= 2'd0;
            i_t <= '0; i_h <= 1'b0; i_pre <= 1'b0; i_hv <= '0;
        end else begin
            case (ist)
                I_IDLE: if (start) begin
                    i_g <= '0; i_k <= '0; i_j <= '0; i_t <= '0; ; ; ;
                    ; ; i_h <= 1'b0;
                    ist <= I_WGT;
                end
                I_BIAS:  if (i_adv) begin
                    i_h <= dbl && !i_h && p_has_bias;
                    if (!(dbl && !i_h && p_has_bias)) ist <= fu_i ? i_nxt : I_MULT;
                end
                I_MULT:  if (i_adv) begin
                    i_h <= dbl && !i_h;
                    if (!(dbl && !i_h)) ist <= I_SHIFT;
                end
                I_SHIFT: if (i_adv) begin
                    i_h <= dbl && !i_h;
                    if (!(dbl && !i_h)) begin
                        ist <= (p_dwf && !i_dl) ? I_WGT : (CHN && chn) ? I_CHW : i_nxt;     // (v37_1a) DWF: the depthwise header next
                                                                                     // (v42DS) chained: activations after the old op
                        i_dl <= p_dwf && !i_dl;
                    end
                end
                I_WGT:   if (i_adv && dbl && !i_h) i_h <= 1'b1;   // dsp: the 8 channels' taps again
                else if (i_adv) begin
                    i_h <= w8l && !i_h && w8n;   // (v58_RS w8) the wrap slice's extra 73rd read next
                    if (!sl_multi) ;
                    if (sl_multi && prei) ;
                    if (sl_multi && !prei) begin
                        // 9 taps of 16 bytes per lane, stride Ci (the next lane's tap 0 follows tap 8)
                        ;
                        i_t <= (i_t != 4'd8) ? i_t + 4'd1 : 4'd0;
                    end
                    if (p_fs || p_dw || nchunks == 8'd1 || (sl_multi && prei) || (!sl_multi && i_j == i_gc[3:0] - 4'd1) ||
                        (sl_multi && (w8l ? i_h : w8n))) begin
                        i_j <= '0;
                        ; ; ; ; ;
                        ; ; ;
                        ; ;
                        ; ;
                        ;
                        ;
                        if (rpw) ;   // (v35R) row 0: second in its pair
                        // a group's parameters follow its first chunk's weights
                        if (i_k == 8'd0 && !p_cp && !p_pa) ist <= pk ? I_OWH : I_BIAS;
                        else if (i_k != 8'd0 && !onepix) ist <= I_BAR;
                        else ist <= i_nxt;
                    end else if (!sl_multi || (i_t == 4'd8 && !(w8l && w8n))) i_j <= i_j + 4'd1;
                end
                I_BAR: if (bar_ok) ist <= I_PSR;
                I_PSR: if (i_adv) begin
                    ist <= i_last ? (owhole ? I_OWH : rpi ? I_OUTW : I_PIX) : I_PSW;
                    i_pre <= rpi; i_h <= 1'b0;
                end
                I_PSW: if (i_adv) ist <= I_PIX;
                I_OWH: if (i_adv) ist <= I_PIX;
                I_PIX: if (i_adv) begin
                    if (pk && !i_h && p_len2 != 20'd0) i_h <= 1'b1;    // packed: the second piece
                    else if (p_contig || p_fs) ist <= I_OUTW;
                    else if (i_colrd && (cw2i || s2t || dwc) && i_fx != 2'd2 && !(rpr && i_fx != 2'd0)) begin     // stride-2 column mode / s2t: next column
                        ; ;   // (rpr: column 0 is padding, 1 + 2 one read)
                    end
                    else if (i_colrd) begin
                        ist <= I_OUTW;
                        if (rpi) i_h <= 1'b1;                     // (A's write command went first)
                        if (cw3 && p_dr && i_bok) begin ; ; ; end
                    end
                    else if (p_sm || p_dsm) begin
                        ;
                        if (i_fy == (p_dsm ? 16'd2 : p_smo ? 16'd0 : p_fh - 16'd1)) begin ; ist <= I_OUTW; end
                        else ;
                    end
                    else if (pa_w3) begin ; ist <= I_OUTW; end   // pool: the whole window in one read
                    else if (slw) begin
                        if (i_fx == 16'd2 || rowc) begin
                            if (i_fy == i_lfy) begin ; ist <= I_OUTW; if (rpi) i_h <= 1'b1; end
                            else begin
                                if (p_dr && i_fy == 16'd2 && !rpi) ist <= I_OUTW;    // A's write; row 3 follows
                                ;
                                ;
                                ;
                                ;
                            end
                        end else begin
                            ;
                            ;
                        end
                    end else begin
                        ;
                        if (i_fy == p_fh - 16'd1) begin ; ist <= (p_rpx && p_swz) ? I_PIXNEXT : I_OUTW; end
                        else ;
                    end
                end
                I_OUTW: if (i_adv) begin
                    if (i_pre) begin i_pre <= 1'b0; ist <= I_PIX; end   // p_rp: A's command first, then the taps
                    else if (p_swz) ist <= I_PIXNEXT;
                    else if (p_dr) begin                             // A: row 3 next (if B exists)
                        i_h <= !i_h && i_bok;
                        ist <= (!i_h && i_bok) ? (i_colrd ? I_OUTW : I_PIX) : I_PIXNEXT;   // (column read: B's row 3 is in)
                    end else if (p_v3) begin                         // (v25M: rows past the output are skipped)
                        i_hv <= i_hvl ? 2'd0 : i_hv + 2'd1;
                        if (i_hvl) ist <= I_PIXNEXT;
                    end else begin
                        i_h <= p_v2 && !i_h;
                        if (!(p_v2 && !i_h)) ist <= (mir && wc_raw) ? I_OUTM : I_PIXNEXT;   // (v46MB) mirrors next
                        i_ms <= p_m1 ? 2'd0 : 2'd1;
                    end
                end
                I_PIXNEXT: begin
                    ; i_h <= 1'b0;
                    ;
                    ;
                    ;
                    if (rpw) ;   // (v35R) next row / column pair
                    if (i_pxn == p_npix) begin
                        ist <= p_pf ? I_PFW : I_PASSNEXT;
                    end else begin
                        if (i_ox == p_wo - 16'd1) begin
                            ; ;
                            if (dw_wrap) begin                  // next walk row (v2 / v3: skip the group's others)
                                ;
                                ;
                            end
                            ; ;
                            ;
                        end else begin
                            ;
                            ;
                        end
                        // next pixel (v26A: one base select, one + ioff2); a sliding pixel starts at the first new column
                        ;
                        ;
                        ;
                        ;
                        // (v28D) strip mode with one output command for the whole op (owhole): a pixel inside a
                        // strip has nothing to issue, so the next one follows at once (one cycle per pixel; I_OUTW
                        // issues nothing there)
                        ist <= (p_sm && !i_spend && owhole && !p_fz && !p_swz) ? I_PIXNEXT :
                               (one_act || ((p_sm || p_dsm) && !i_spend) || rpi) ? I_OUTW : I_PIX;
                        i_pre <= rpi;
                    end
                end
                I_PFW: if (i_adv) ist <= I_PASSNEXT;
                I_OUTM: if (i_adv) begin                     // (v46MB) M1, M2, top run, row run
                    if (i_ms == 2'd0 && p_m2) i_ms <= 2'd1;
                    else if (i_ms[1] == 1'b0 && i_mft) i_ms <= 2'd2;
                    else if (i_ms != 2'd3 && i_mfr) i_ms <= 2'd3;
                    else ist <= I_PIXNEXT;
                end
                I_PASSNEXT: if (!(ag_wi && !i_sg && !p_dwf)) begin   // (v47c AG) after the wrap pass's last write
                    if (p_fs) ;                     // (the next group skips I_WGT)
                    if (!i_last) begin
                        i_k <= i_k + 8'd1; i_j <= '0; i_t <= '0; ist <= I_WGT;
                        ;
                        ;
                        if (!(sl_multi && p_pre)) ;
                    end else begin
                        i_k <= '0;
                        ;      // depthwise: next 16-channel slice
                        ;
                        if (!p_po8 || i_g[0]) ;   // planar output: next plane
                        if (i_g == ngroups - 16'd1 || pk) ist <= I_IDLE;
                        else begin
                            i_g <= i_g + 16'd1; ist <= p_fs ? I_BIAS : I_WGT; i_j <= '0; i_t <= '0;
                            ;
                            if (!(sl_multi && p_pre))
                                ;
                        end
                    end
                end
                I_CHW: if (!chn) ist <= i_nxt;                  // (v42DS) the chained op's activations wait for the old op
                default: ist <= I_IDLE;
            endcase
        end
    end
    // (v26A: data registers of this FSM are gated-clock flops without reset: next value / write flag)
    logic i_oy_w_i;
    logic [7:0] i_oy_n_i;
    logic i_ox_w_i;
    logic [7:0] i_ox_n_i;
    logic i_fy_w_i;
    logic [7:0] i_fy_n_i;
    logic i_px_w_i;
    logic [11:0] i_px_n_i;
    logic i_rowbase_w_i;
    // (v56_AR p_hadd) fa_1 ripple adders (h_rca) for register updates
    wire [19:0] ha_wlk; wire ha_wlk_co;
    h_rca #(20) u_ha_wlk (.a(20'(i_wl)), .b(20'({7'd0, p_k})), .ci(1'b0), .s(ha_wlk), .co(ha_wlk_co));
    wire [19:0] ha_wlc; wire ha_wlc_co;
    h_rca #(20) u_ha_wlc (.a(20'(i_wl)), .b(20'({10'd0, p_ci})), .ci(1'b0), .s(ha_wlc), .co(ha_wlc_co));
    wire [19:0] ha_ptr; wire ha_ptr_co;
    h_rca #(20) u_ha_ptr (.a(20'(i_ptr)), .b(20'({4'd0, p_rowstride})), .ci(1'b0), .s(ha_ptr), .co(ha_ptr_co));
    wire [19:0] ha_tpa; wire ha_tpa_co;
    h_rca #(20) u_ha_tpa (.a(20'(i_tap_s)), .b(20'((rpr ? 20'd0 : i_cstep))), .ci(1'b0), .s(ha_tpa), .co(ha_tpa_co));
    wire [19:0] ha_tpb; wire ha_tpb_co;
    h_rca #(20) u_ha_tpb (.a(20'(i_tap_s)), .b(20'(i_cstep)), .ci(1'b0), .s(ha_tpb), .co(ha_tpb_co));
    wire [19:0] ha_obp; wire ha_obp_co;
    h_rca #(20) u_ha_obp (.a(20'(i_obase)), .b(20'(i_opix)), .ci(1'b0), .s(ha_obp), .co(ha_obp_co));
    wire [19:0] ha_igg; wire ha_igg_co;
    h_rca #(20) u_ha_igg (.a(20'(i_ig)), .b(20'({4'd0, p_igrp})), .ci(1'b0), .s(ha_igg), .co(ha_igg_co));
    wire [19:0] ha_ogb; wire ha_ogb_co;
    h_rca #(20) u_ha_ogb (.a(20'(i_ogb)), .b(20'({4'd0, p_npix, 4'd0})), .ci(1'b0), .s(ha_ogb), .co(ha_ogb_co));
    wire [19:0] ha_tpc; wire ha_tpc_co;
    h_rca #(20) u_ha_tpc (.a(20'(tp_a)), .b(20'({10'd0, p_co})), .ci(1'b0), .s(ha_tpc), .co(ha_tpc_co));
    wire [19:0] ha_dwc; wire ha_dwc_co;
    h_rca #(20) u_ha_dwc (.a(20'(dwc_a)), .b(20'({10'd0, p_co})), .ci(1'b0), .s(ha_dwc), .co(ha_dwc_co));
    // (v56_AR p_hadd) fa_1 ripple adders (h_rca) for register updates
    wire [19:0] hb_m1a; wire hb_m1a_co;
    h_rca #(20) u_hb_m1a (.a(20'(i_m1)), .b(20'({10'd0, p_co})), .ci(1'b0), .s(hb_m1a), .co(hb_m1a_co));
    wire [19:0] hb_m1b; wire hb_m1b_co;
    h_rca #(20) u_hb_m1b (.a(20'(hb_m1a)), .b(20'((i_rend ? {10'd0, p_m1x} : 20'd0))), .ci(1'b0), .s(hb_m1b), .co(hb_m1b_co));
    wire [19:0] hb_orb; wire hb_orb_co;
    h_rca #(20) u_hb_orb (.a(20'(i_orb)), .b(20'(((p_v2 || p_dr) ? i_ovs : {4'd0, p_orow}))), .ci(1'b0), .s(hb_orb), .co(hb_orb_co));
    wire [19:0] hb_ntf; wire hb_ntf_co;
    h_rca #(20) u_hb_ntf (.a(20'(i_nt)), .b(20'((i_nsl ? {4'd0, p_fx0off} : 20'd0))), .ci(1'b0), .s(hb_ntf), .co(hb_ntf_co));
    wire [19:0] hb_wgk; wire hb_wgk_co;
    h_rca #(20) u_hb_wgk (.a(20'(i_wg)), .b(20'((p_dw ? (p_pa ? 20'd0 : 20'd144) : (p_pg || p_fs) ? {3'd0, p_k, 4'd0} : {4'd0, p_k, 3'd0}))), .ci(1'b0), .s(hb_wgk), .co(hb_wgk_co));
    wire [19:0] hb_obo; wire hb_obo_co;
    h_rca #(20) u_hb_obo (.a(20'(i_obase)), .b(20'({4'd0, p_orow})), .ci(1'b0), .s(hb_obo), .co(hb_obo_co));
    wire [19:0] hb_pbc; wire hb_pbc_co;
    h_rca #(20) u_hb_pbc (.a(20'(i_pbase)), .b(20'(i_cs)), .ci(1'b0), .s(hb_pbc), .co(hb_pbc_co));
    wire [19:0] hb_m1t; wire hb_m1t_co;
    h_rca #(20) u_hb_m1t (.a(20'(i_m1)), .b(20'(~{4'd0, p_mtl})), .ci(1'b1), .s(hb_m1t), .co(hb_m1t_co));
    logic [19:0] i_rowbase_n_i;
    logic i_pbase_w_i;
    logic [19:0] i_pbase_n_i;
    logic i_ptr_w_i;
    logic [19:0] i_ptr_n_i;
    logic i_obase_w_i;
    logic [19:0] i_obase_n_i;
    logic i_cs_w_i;
    logic [12:0] i_cs_n_i;
    logic i_wg_w_i;
    logic [19:0] i_wg_n_i;
    logic i_wl_w_i;
    logic [19:0] i_wl_n_i;
    logic i_fx_w_i;
    logic [1:0] i_fx_n_i;
    logic i_tap_s_w_i;
    logic [19:0] i_tap_s_n_i;
    logic i_iy_w_i;
    logic signed [9:0] i_iy_n_i;
    logic i_ix_w_i;
    logic signed [9:0] i_ix_n_i;
    logic i_sp_w_i;
    logic [7:0] i_sp_n_i;
    logic i_ig_w_i;
    logic [19:0] i_ig_n_i;
    logic i_ogb_w_i;
    logic [19:0] i_ogb_n_i;
    logic i_orb_w_i;
    logic [19:0] i_orb_n_i;
    logic i_ob2_w_i;
    logic [19:0] i_ob2_n_i;
    always_comb begin
        i_oy_w_i = 1'b0; i_oy_n_i = 'x;
        i_ox_w_i = 1'b0; i_ox_n_i = 'x;
        i_fy_w_i = 1'b0; i_fy_n_i = 'x;
        i_px_w_i = 1'b0; i_px_n_i = 'x;
        i_rowbase_w_i = 1'b0; i_rowbase_n_i = 'x;
        i_pbase_w_i = 1'b0; i_pbase_n_i = 'x;
        i_ptr_w_i = 1'b0; i_ptr_n_i = 'x;
        i_obase_w_i = 1'b0; i_obase_n_i = 'x;
        i_cs_w_i = 1'b0; i_cs_n_i = 'x;
        i_wg_w_i = 1'b0; i_wg_n_i = 'x;
        i_wl_w_i = 1'b0; i_wl_n_i = 'x;
        i_fx_w_i = 1'b0; i_fx_n_i = 'x;
        i_tap_s_w_i = 1'b0; i_tap_s_n_i = 'x;
        i_iy_w_i = 1'b0; i_iy_n_i = 'x;
        i_ix_w_i = 1'b0; i_ix_n_i = 'x;
        i_sp_w_i = 1'b0; i_sp_n_i = 'x;
        i_ig_w_i = 1'b0; i_ig_n_i = 'x;
        i_ogb_w_i = 1'b0; i_ogb_n_i = 'x;
        i_orb_w_i = 1'b0; i_orb_n_i = 'x;
        i_ob2_w_i = 1'b0; i_ob2_n_i = 'x;
        begin
            case (ist)
                I_IDLE: if (start) begin
                    ; ; ; ; begin i_cs_n_i = '0; i_cs_w_i = 1'b1; end begin i_ig_n_i = '0; i_ig_w_i = 1'b1; end begin i_ogb_n_i = p_out[19:0]; i_ogb_w_i = 1'b1; end
                    begin i_wg_n_i = p_wgt + (rot ? {15'd0, ag_lo} : 32'd0); i_wg_w_i = 1'b1; end begin i_wl_n_i = p_wgt; i_wl_w_i = 1'b1; end ;
                    ;
                end
                I_BIAS:  if (i_adv) begin
                    ;
                    if (!(dbl && !i_h && p_has_bias)) ;
                end
                I_MULT:  if (i_adv) begin
                    ;
                    if (!(dbl && !i_h)) ;
                end
                I_SHIFT: if (i_adv) begin
                    ;
                    if (!(dbl && !i_h)) begin
                        ;     // (v37_1a) DWF: the depthwise header next
                                                                                     // (v42DS) chained: activations after the old op
                        ;
                    end
                end
                I_WGT:   if (i_adv && dbl && !i_h) ;   // dsp: the 8 channels' taps again
                else if (i_adv) begin
                    ;
                    if (!sl_multi) begin i_wl_n_i = ha_wlk; i_wl_w_i = 1'b1; end
                    if (sl_multi && p_pre) begin i_wl_n_i = i_wl + rc0_len; i_wl_w_i = 1'b1; end
                    if (sl_multi && !prei) begin
                        // 9 taps of 16 bytes per lane, stride Ci (the next lane's tap 0 follows tap 8)
                        begin i_wl_n_i = ha_wlc; i_wl_w_i = 1'b1; end
                        ;
                    end
                    if (p_fs || p_dw || nchunks == 8'd1 || (sl_multi && prei) || (!sl_multi && i_j == i_gc[3:0] - 4'd1) ||
                        (sl_multi && (w8l ? i_h : w8n))) begin
                        ;
                        begin i_oy_n_i = '0; i_oy_w_i = 1'b1; end begin i_ox_n_i = '0; i_ox_w_i = 1'b1; end begin i_fy_n_i = '0; i_fy_w_i = 1'b1; end begin i_px_n_i = '0; i_px_w_i = 1'b1; end begin i_fx_n_i = '0; i_fx_w_i = 1'b1; end
                        begin i_rowbase_n_i = p_in; i_rowbase_w_i = 1'b1; end begin i_pbase_n_i = p_in; i_pbase_w_i = 1'b1; end begin i_ptr_n_i = p_in + ioff2; i_ptr_w_i = 1'b1; end
                        begin i_tap_s_n_i = p_in + ioff2; i_tap_s_w_i = 1'b1; end begin i_sp_n_i = '0; i_sp_w_i = 1'b1; end
                        begin i_iy_n_i = -$signed({9'd0, p_pt}); i_iy_w_i = 1'b1; end begin i_ix_n_i = -$signed({9'd0, p_pl}); i_ix_w_i = 1'b1; end
                        begin i_obase_n_i = p_dw ? (p_out + {12'd0, i_cs}) : p_pout ? i_ogb : p_po8 ? i_po8b :
                                   p_pg ? (p_pft ? ({12'd0, p_out2} + {12'd0, i_g, 4'd0}) : ({p_out[31:4], rot ? 4'd0 : p_out[3:0]} + {12'd0, rot ? (ag_wi ? 8'd0 : (i_g + 8'd1)) : i_g, 4'd0})) : (p_out + {13'd0, i_g, 3'd0}); i_obase_w_i = 1'b1; end
                        begin i_orb_n_i = p_dr ? (p_po8 ? i_po8b : (p_out + {13'd0, i_g, 3'd0})) : p_smo ? i_ogb : (p_out + {12'd0, i_cs}); i_orb_w_i = 1'b1; end
                        if (rpw) begin i_ob2_n_i = (p_po8 ? i_po8b : (p_out[19:0] + {9'd0, i_g, 3'd0})) + 20'd16; i_ob2_w_i = 1'b1; end   // (v35R) row 0: second in its pair
                        // a group's parameters follow its first chunk's weights
                        if (i_k == 8'd0 && !p_cp && !p_pa) ;
                        else if (i_k != 8'd0 && !onepix) ;
                        else ;
                    end else if (!sl_multi || i_t == 4'd8) ;
                end
                I_BAR: if (bar_ok) ;
                I_PSR: if (i_adv) begin
                    ;
                    ; ;
                end
                I_PSW: if (i_adv) ;
                I_OWH: if (i_adv) ;
                I_PIX: if (i_adv) begin
                    if (pk && !i_h && p_len2 != 20'd0) ;    // packed: the second piece
                    else if (p_contig || p_fs) ;
                    else if (i_colrd && (cw2i || s2t || dwc) && i_fx != 2'd2 && !(rpr && i_fx != 2'd0)) begin     // stride-2 column mode / s2t: next column
                        begin i_fx_n_i = i_fx + 2'd1; i_fx_w_i = 1'b1; end begin i_tap_s_n_i = ha_tpa; i_tap_s_w_i = 1'b1; end   // (rpr: column 0 is padding, 1 + 2 one read)
                    end
                    else if (i_colrd) begin
                        ;
                        if (rpi) ;                     // (A's write command went first)
                        if (cw3 && p_dr && i_bok) begin begin i_fy_n_i = 16'd3; i_fy_w_i = 1'b1; end begin i_fx_n_i = 2'd2; i_fx_w_i = 1'b1; end begin i_tap_s_n_i = i_tap_s + 20'd48; i_tap_s_w_i = 1'b1; end end
                    end
                    else if (p_sm || p_dsm) begin
                        begin i_ptr_n_i = ha_ptr; i_ptr_w_i = 1'b1; end
                        if (i_fy == (p_dsm ? 16'd2 : p_smo ? 16'd0 : p_fh - 16'd1)) begin begin i_fy_n_i = '0; i_fy_w_i = 1'b1; end ; end
                        else begin i_fy_n_i = i_fy + 16'd1; i_fy_w_i = 1'b1; end
                    end
                    else if (pa_w3) begin begin i_fy_n_i = '0; i_fy_w_i = 1'b1; end ; end   // pool: the whole window in one read
                    else if (slw) begin
                        if (i_fx == 16'd2 || rowc) begin
                            if (i_fy == i_lfy) begin begin i_fy_n_i = '0; i_fy_w_i = 1'b1; end ; if (rpi) ; end
                            else begin
                                if (p_dr && i_fy == 16'd2 && !rpi) ;    // A's write; row 3 follows
                                begin i_fy_n_i = i_fy + 16'd1; i_fy_w_i = 1'b1; end
                                begin i_fx_n_i = i_fx0; i_fx_w_i = 1'b1; end
                                begin i_ptr_n_i = ha_ptr; i_ptr_w_i = 1'b1; end
                                begin i_tap_s_n_i = ha_ptr + ((i_fx0 != 16'd0) ? p_fx0off : 32'd0); i_tap_s_w_i = 1'b1; end
                            end
                        end else begin
                            begin i_fx_n_i = i_fx + 16'd1; i_fx_w_i = 1'b1; end
                            begin i_tap_s_n_i = ha_tpb; i_tap_s_w_i = 1'b1; end
                        end
                    end else begin
                        begin i_ptr_n_i = ha_ptr; i_ptr_w_i = 1'b1; end
                        if (i_fy == p_fh - 16'd1) begin begin i_fy_n_i = '0; i_fy_w_i = 1'b1; end ; end
                        else begin i_fy_n_i = i_fy + 16'd1; i_fy_w_i = 1'b1; end
                    end
                end
                I_OUTW: if (i_adv) begin
                    if (i_pre) begin ; ; end   // p_rp: A's command first, then the taps
                    else if (p_swz) ;
                    else if (p_dr) begin                             // A: row 3 next (if B exists)
                        ;
                        ;   // (column read: B's row 3 is in)
                    end else if (p_v3) begin                         // (v25M: rows past the output are skipped)
                        ;
                        if (i_hvl) ;
                    end else begin
                        ;
                        if (!(p_v2 && !i_h)) ;
                    end
                end
                I_PIXNEXT: begin
                    begin i_fy_n_i = '0; i_fy_w_i = 1'b1; end ;
                    begin i_sp_n_i = i_spend ? 8'd0 : i_sp + 8'd1; i_sp_w_i = 1'b1; end
                    begin i_px_n_i = i_pxn; i_px_w_i = 1'b1; end
                    begin i_obase_n_i = ha_obp; i_obase_w_i = 1'b1; end
                    if (rpw) begin i_ob2_n_i = i_ob2 + ((i_ox == p_wo - 16'd1) ? rp_k : i_ox[0] ? {3'd0, p_opix[15:4], 1'b1, 4'd0} : 20'hffff0); i_ob2_w_i = 1'b1; end   // (v35R) next row / column pair
                    if (i_pxn == p_npix) begin
                        ;
                    end else begin
                        if (i_ox == p_wo - 16'd1) begin
                            begin i_ox_n_i = '0; i_ox_w_i = 1'b1; end begin i_oy_n_i = i_oy + ((p_v2 || p_dr) ? i_ostep : 16'd1); i_oy_w_i = 1'b1; end
                            if (dw_wrap) begin                  // next walk row (v2 / v3: skip the group's others)
                                begin i_obase_n_i = hb_orb; i_obase_w_i = 1'b1; end
                                begin i_orb_n_i = i_orb + ((p_v2 || p_dr) ? i_ovs : {4'd0, p_orow}); i_orb_w_i = 1'b1; end
                            end
                            begin i_iy_n_i = i_iy + ((dr2 || s2t) ? 17'sd4 : p_v3 ? 17'sd3 : (p_v2 || p_dr) ? 17'sd2 : $signed({1'b0, p_sw})); i_iy_w_i = 1'b1; end begin i_ix_n_i = -$signed({9'd0, p_pl}); i_ix_w_i = 1'b1; end
                            begin i_rowbase_n_i = i_nb; i_rowbase_w_i = 1'b1; end
                        end else begin
                            begin i_ox_n_i = i_ox + 16'd1; i_ox_w_i = 1'b1; end
                            begin i_ix_n_i = i_ix + $signed({1'b0, p_sw}); i_ix_w_i = 1'b1; end
                        end
                        // next pixel (v26A: one base select, one + ioff2); a sliding pixel starts at the first new column
                        begin i_pbase_n_i = i_nb; i_pbase_w_i = 1'b1; end
                        begin i_ptr_n_i = i_nt; i_ptr_w_i = 1'b1; end
                        begin i_tap_s_n_i = hb_ntf; i_tap_s_w_i = 1'b1; end
                        begin i_fx_n_i = i_nsl ? (2'd3 - p_sw) : 2'd0; i_fx_w_i = 1'b1; end
                        // (v28D) strip mode with one output command for the whole op (owhole): a pixel inside a
                        // strip has nothing to issue, so the next one follows at once (one cycle per pixel; I_OUTW
                        // issues nothing there)
                        ;
                        ;
                    end
                end
                I_PFW: if (i_adv) ;
                I_PASSNEXT: if (!(ag_wi && !i_sg && !p_dwf)) begin   // (v47c AG) after the wrap pass's last write
                    if (p_fs) begin i_px_n_i = '0; i_px_w_i = 1'b1; end                     // (the next group skips I_WGT)
                    if (!i_last) begin
                        ; ; ; ;
                        begin i_cs_n_i = i_cs + (p_sliced ? 20'd16 : 20'(KCH)); i_cs_w_i = 1'b1; end
                        begin i_ig_n_i = ha_igg; i_ig_w_i = 1'b1; end
                        if (!(sl_multi && p_pre)) begin i_wl_n_i = i_wg + {12'd0, i_cs} + (p_sliced ? 32'd16 : 32'(KCH)); i_wl_w_i = 1'b1; end
                    end else begin
                        ;
                        begin i_cs_n_i = p_dw ? (i_cs + 20'd16) : 20'd0; i_cs_w_i = 1'b1; end      // depthwise: next 16-channel slice
                        begin i_ig_n_i = p_dw ? ha_igg : 20'd0; i_ig_w_i = 1'b1; end
                        if (!(p_po8 || (p_m2 && !p_pg)) || i_g[0]) begin i_ogb_n_i = ha_ogb; i_ogb_w_i = 1'b1; end   // planar output: next plane
                        if (i_g == ngroups - 16'd1 || pk) ;
                        else begin
                            ; ; ; ;
                            begin i_wg_n_i = hb_wgk; i_wg_w_i = 1'b1; end
                            if (!(sl_multi && p_pre))
                                begin i_wl_n_i = hb_wgk; i_wl_w_i = 1'b1; end
                        end
                    end
                end
                I_CHW: if (!chn) ;                  // (v42DS) the chained op's activations wait for the old op
                default: ;
            endcase
                end
    end
    wire i_oy_g_i = clk | ~i_oy_w_i;
    always_ff @(posedge i_oy_g_i) i_oy <= i_oy_n_i;
    wire i_ox_g_i = clk | ~i_ox_w_i;
    always_ff @(posedge i_ox_g_i) i_ox <= i_ox_n_i;
    wire i_fy_g_i = clk | ~i_fy_w_i;
    always_ff @(posedge i_fy_g_i) i_fy <= i_fy_n_i;
    wire i_px_g_i = clk | ~i_px_w_i;
    always_ff @(posedge i_px_g_i) i_px <= i_px_n_i;
    wire i_rowbase_g_i = clk | ~i_rowbase_w_i;
    always_ff @(posedge i_rowbase_g_i) i_rowbase <= i_rowbase_n_i;
    wire i_pbase_g_i = clk | ~i_pbase_w_i;
    always_ff @(posedge i_pbase_g_i) i_pbase <= i_pbase_n_i;
    wire i_ptr_g_i = clk | ~i_ptr_w_i;
    always_ff @(posedge i_ptr_g_i) i_ptr <= i_ptr_n_i;
    wire i_obase_g_i = clk | ~i_obase_w_i;
    always_ff @(posedge i_obase_g_i) i_obase <= i_obase_n_i;
    // (v46MB) mirror addresses: set with the pass's first output base, stepped with the pixel
    wire m_st  = mir && (ist == I_WGT) && i_obase_w_i;
    wire m_adv = mir && (ist == I_PIXNEXT);
    wire i_m_g = clk | ~(m_st || m_adv);
    always_ff @(posedge i_m_g) begin
        i_m1 <= m_st ? ((p_pg ? (p_out[19:0] + {8'd0, i_g, 4'd0}) : (p_out[19:0] + {9'd0, i_g, 3'd0})) + p_m1d)
                     : hb_m1b;
        i_m2 <= m_st ? ((p_pg ? i_ogb : i_po8b) + p_m2d) : (i_m2 + 20'd16);
    end
    wire i_cs_g_i = clk | ~i_cs_w_i;
    always_ff @(posedge i_cs_g_i) i_cs <= i_cs_n_i;
    wire i_wg_g_i = clk | ~i_wg_w_i;
    always_ff @(posedge i_wg_g_i) i_wg <= i_wg_n_i;
    wire i_wl_g_i = clk | ~i_wl_w_i;
    always_ff @(posedge i_wl_g_i) i_wl <= i_wl_n_i;
    wire i_fx_g_i = clk | ~i_fx_w_i;
    always_ff @(posedge i_fx_g_i) i_fx <= i_fx_n_i;
    wire i_tap_s_g_i = clk | ~i_tap_s_w_i;
    always_ff @(posedge i_tap_s_g_i) i_tap_s <= i_tap_s_n_i;
    wire i_iy_g_i = clk | ~i_iy_w_i;
    always_ff @(posedge i_iy_g_i) i_iy <= i_iy_n_i;
    wire i_ix_g_i = clk | ~i_ix_w_i;
    always_ff @(posedge i_ix_g_i) i_ix <= i_ix_n_i;
    wire i_sp_g_i = clk | ~i_sp_w_i;
    always_ff @(posedge i_sp_g_i) i_sp <= i_sp_n_i;
    wire i_ig_g_i = clk | ~i_ig_w_i;
    always_ff @(posedge i_ig_g_i) i_ig <= i_ig_n_i;
    wire i_ogb_g_i = clk | ~i_ogb_w_i;
    always_ff @(posedge i_ogb_g_i) i_ogb <= i_ogb_n_i;
    wire i_orb_g_i = clk | ~i_orb_w_i;
    always_ff @(posedge i_orb_g_i) i_orb <= i_orb_n_i;
    wire i_ob2_g_i = clk | ~i_ob2_w_i;
    always_ff @(posedge i_ob2_g_i) i_ob2 <= i_ob2_n_i;


    // ================================================================== consumer
    typedef enum logic [2:0] {C_IDLE, C_BIAS, C_MULT, C_SHIFT, C_WGT, C_PIX, C_PASSEND} cst_t;
    cst_t        cst;
    logic        c_dl;                                     // (v37_1a) DWF: loading the pass's depthwise header
    logic [7:0]  c_g;
    logic [11:0] c_px;
    logic [4:0]  c_k;
    logic [4:0]  c_j;
    logic [3:0]  c_e, c_b;
    logic [12:0] c_cs;
    wire  [4:0]  c_gc   = (c_g == ngroups - 16'd1) ? gc_last : gsize;
    wire  [19:0] c_ce   = ((c_cs + 20'(KCH)) > 20'(p_k)) ? 20'(p_k) : (c_cs + 20'(KCH));
    wire  [19:0] c_ck   = c_ce - c_cs;
    wire         sm2    = p_sm && p_sm2;                  // (v40K rs-conv1-pack)
    wire  [3:0]  c_kb   = slw ? 4'd9 : p_sm ? (sm2 ? 4'd2 : p_fh[3:0]) : 4'((c_ck + 20'd15) >> 4);
    wire  [4:0]  c_sml  = (p_l[4:0] == 5'd0) ? 5'd16 : p_l[4:0];   // strip mode: bytes per beat (L)
    logic [7:0]  c_sp, c_pb, c_po;                       // strip mode: pixel, its offset, beat offset
    logic        c_ls;                                   // the current strip is its row's last
    wire  [7:0]  c_seg   = c_ls ? p_segl : p_segf;
    wire         c_spend = (c_sp == p_sp - 8'd1) || (c_ox == p_wo - 16'd1);
    // pixel pairs: the pair's second pixel exists (same strip), the pair ends its strip / row
    wire         pv2     = p_pp && !c_spend;
    wire         c_spe   = c_spend || (pv2 && ((c_sp + 8'd1 == p_sp - 8'd1) || (c_ox + 16'd1 == p_wo - 16'd1)));
    wire         c_rend  = (c_ox == p_wo - 16'd1) || (pv2 && (c_ox + 16'd1 == p_wo - 16'd1));
    wire  [15:0] c_adv   = pv2 ? 16'd2 : 16'd1;
    wire  [7:0]  c_pps   = pv2 ? {p_pixstep[6:0], 1'b0} : p_pixstep[7:0];
    wire  [15:0] nx_oxp  = c_rend ? 16'd0 : c_ox + c_adv;
    // depthwise strip: constants of the current strip (origin column s_ix0, s_np pixels) and the
    // offset of the walked tap; rows above / below the input hold no bytes in the strip
    logic signed [9:0]  s_ix0;
    logic [7:0]  s_np;
    wire  signed [16:0] s_clo = (s_ix0 < 0) ? 17'sd0 : s_ix0;
    wire  signed [16:0] s_ch0 = s_ix0 + $signed({9'd0, s_np}) + 17'sd1;
    wire  signed [16:0] s_chi = (s_ch0 > $signed({1'b0, p_win}) - 17'sd1) ? ($signed({1'b0, p_win}) - 17'sd1) : s_ch0;
    wire  [7:0]  s_seg = 8'(s_chi - s_clo + 17'sd1) << d_cs;
    wire  [1:0]  s_tinv = (c_iy < 0) ? 2'(-c_iy) : 2'd0;
    wire  signed [16:0] s_bx = c_iy + 17'sd3 - $signed({1'b0, p_hin});
    wire  [1:0]  s_binv = (s_bx > 0) ? 2'(s_bx) : 2'd0;
    wire  [1:0]  s_nvr = 2'd3 - s_tinv - s_binv;
    wire  [7:0]  s_drop = (s_nvr == 2'd3) ? (s_seg + {s_seg[6:0], 1'b0}) : (s_nvr == 2'd2) ? {s_seg[6:0], 1'b0} :
                          (s_nvr == 2'd1) ? s_seg : 8'd0;
    wire  [1:0]  s_vrow = c_fy[1:0] - s_tinv;
    wire  [7:0]  d_off = ((s_vrow == 2'd2) ? {s_seg[6:0], 1'b0} : (s_vrow == 2'd1) ? s_seg : 8'd0) +
                         (8'(c_ix + $signed({1'b0, c_fx}) - s_clo) << d_cs);
    wire  [15:0] nx_ixw  = (c_ox == p_wo - 16'd1) ? 16'd0 : c_ox + 16'd1;
    wire  [4:0]  c_lastn= (slw || c_ck[3:0] == 4'd0) ? 5'd16 : {1'b0, c_ck[3:0]};
    // sliced 3x3 window: taps whose column is still in the window are not re-read. The window
    // holds offset-added activations (9 bits per byte).
    logic [7:0]   c_ox;
    logic [2:0]   c_fy;
    logic [1:0]   c_fx;
    logic [7:0]   c_oy;
    // dual-row: B (c_hb) walks window rows 1..3 (c_rr = c_fy + 1); rows 1, 2 were committed by A
    // (their current columns are in place), row 3 still holds the previous pixel's columns
    logic        c_hb;
    wire  [2:0]  c_rr = c_fy + {2'd0, c_hb};
    wire         c_brw = c_hb && (c_rr != 3'd3);           // B's row, all from the window
    wire         c_bok = (c_oy + 16'd1) < p_ho;
    logic [127:0] win  [5][3];                             // row 3: v2 / dual-row only, row 4: v3 (raw int8 bytes)
    logic [127:0] nrow [4];
    wire         c_slide = slw && sl12 && (c_ox != 16'd0);
    wire         win_src = c_brw || (c_slide && (c_fx < (16'd3 - p_sw)));
    // depthwise walks only the new taps of each row (from column fx0); old columns shift in the
    // window when the row is committed
    wire  [15:0] c_fx0   = c_slide ? (16'd3 - p_sw) : 16'd0;
    logic signed [9:0]  c_iy, c_ix;                       // window origin in the real input
    wire  signed [16:0] c_ry = c_iy + $signed({1'b0, c_rr});
    wire  signed [16:0] c_rx = c_ix + $signed({1'b0, c_fx});
    wire         c_tv = !slw || ((c_ry >= 17'sd0) && (c_ry < $signed({1'b0, p_hin})) &&
                                  !(pa_lr && (c_ry[7:0] == p_hin - 8'd1) && (c_fx >= p_fx0off[1:0])) &&
                                  (c_rx >= 17'sd0) && (c_rx < $signed({1'b0, p_win})));
    wire  [15:0] nx_ox   = (c_ox == p_wo - 16'd1) ? 16'd0 : c_ox + 16'd1;
    wire  [15:0] nx_fx0  = (slw && sl12 && nx_ox != 16'd0) ? (16'd3 - p_sw) : 16'd0;
    // paired groups (dp): every beat of a pixel feeds group A (lane j: channel j, even weight entry
    // 2b, slot A) and group B (channel 8+j, odd entry 2b+1, slot B) in the same MAC issue
    // K-split (dp, sliced conv): an issue at an even tap t (slot A) also takes tap t+1 (slot B)
    // when at most one of the two reads the stream; both add into the lane's accumulator
    // stride-1 dual-row cross issues (sliding pair, B present): A's tap 8 (slot A, entry 8) and B's tap 5
    // (slot B, entry 5) are the same window tap (row 2, column 2) and issue together (xiss: acc += slot A,
    // acc2 starts with slot B); B then issues taps 0|1, 2|3, 6|7 and 4|8 (x48: slot B takes entry 8)
    logic        c_x5;                                     // B's tap 5 went with A's tap 8
    wire         xm     = XDR && p_dr && !dr2 && p_sliced && c_slide && c_bok;
    wire         xiss   = xm && !c_hb && (c_b == 4'd8);
    wire         x48    = c_hb && c_x5 && (c_b == 4'd4);
    wire  [15:0] n_fy   = x48 ? 16'd2 : (c_fx == 16'd2) ? (c_fy + 16'd1) : c_fy;    // tap t+1 (x48: tap 8)
    wire  [15:0] n_fx   = x48 ? 16'd2 : (c_fx == 16'd2) ? 16'd0 : (c_fx + 16'd1);
    wire  [2:0]  n_rr   = n_fy[2:0] + {2'd0, c_hb};
    wire         n_brw  = c_hb && (n_rr != 3'd3);
    wire         n_src  = n_brw || (c_slide && (n_fx < (16'd3 - p_sw)));
    wire  signed [16:0] n_ry = c_iy + $signed({1'b0, n_rr});
    wire  signed [16:0] n_rx = c_ix + $signed({1'b0, n_fx});
    wire         n_tv   = (n_ry >= 17'sd0) && (n_ry < $signed({1'b0, p_hin})) &&
                          (n_rx >= 17'sd0) && (n_rx < $signed({1'b0, p_win}));
    wire         a_str  = c_tv && !win_src;                   // tap t reads the stream
    wire         n_str  = n_tv && !n_src;
    // ---- stride-2 column mode (cw2): a pixel pair (A walk rows 0-2, B rows 2-4) is a fixed 11-step
    // schedule; steps 0-9 consume one stream tap each (new walk columns 1, then 2, rows 0..4, in read
    // order). Column 0 of rows 0..3 is kept in win[r][2], of row 4 in nrow[1] (W4); nrow[0] (G) holds a
    // shared tap for B. All window / G writes are staged (+2 cycles): every read is >= 3 steps later.
    //  step: slot A src/entry | slot B src/entry | accumulate (K0: acc, K1: acc2, X: A -> acc, B -> acc2)
    //   0: W0 e0 | S e1  K0 (A starts)      1: S e4 | W1 e3  K0        2: W2 e6 | S e7  K0 (G <- S)
    //   3: S e4 | W3 e3  K1 (B starts)      4: W2 e0 | S e7  K1        5: S e2 | G e1  X  (W0 <- S)
    //   6: -    | S e5   K0 (W1 <- S)       7: S e8 | -      K0 (A ends; G <- S; W2 <- S)
    //   8: W4 e6 | S e5  K1 (W3 <- S)       9: S e8 | -      K1 (W4 <- S)    10: G e2 | -  K1 (B ends)
    // The first pixel of a walk row first fills column 0 (5 steps, no MAC).
    wire         cw2    = scw && dr2;
    logic [3:0]  c_st;                                     // cw2: schedule step
    logic [2:0]  c_fl;                                     // cw2: fill row (5: column 0 in place)
    logic [4:0]  c_pz;                                     // (v35R) rpq: window column 0 of row r still the padding row
    wire         c_fill = cw2 && (c_fl != 3'd5);
    wire         rpq    = p_rpr && cw2;                    // (v35R) row-pair read: stream taps (r, 1), (r, 2) per walk row r
    wire  [2:0]  s_r    = c_fill ? c_fl : rpq ? c_st[3:1] : (c_st >= 4'd5) ? 3'(c_st - 4'd5) : c_st[2:0];
    wire  [1:0]  s_c    = c_fill ? 2'd0 : rpq ? (c_st[0] ? 2'd2 : 2'd1) : (c_st >= 4'd5) ? 2'd2 : 2'd1;
    wire  signed [16:0] s_ry = c_iy + $signed({14'd0, s_r});
    wire  signed [16:0] s_rx = c_ix + $signed({15'd0, s_c});
    wire         s_v    = (s_ry >= 17'sd0) && (s_ry < $signed({1'b0, p_hin})) && (s_rx >= 17'sd0) && (s_rx < $signed({1'b0, p_win}));
    wire         s_in   = c_fill || (c_st != 4'd10);       // the step reads the stream
    logic [2:0]  x_as, x_bs;                               // source: 0..3 W(r), 4 W4, 5 S, 6 G, 7 none
    logic [3:0]  x_ae, x_be;                               // weight entries
    logic [1:0]  x_md;                                     // 0 K0, 1 K1, 2 X
    logic        x_an;                                     // (v35R) slot A reads the stream for a window write only
    // (v35R) rpq schedule (stream order (r, 1), (r, 2), r = 0..4; same slots / entries / staging rules):
    //   0: W0 e0 | S e1  K0 (A starts)      1: S e2 | W1 e3  K0 (W0 <- S)       2: S e4 | W3 e3  X (B starts)
    //   3: W2 e6 | S e5  K0 (W1 <- S)       4: W2 e0 | S e1  K1 (G <- S)        5: S e8 | -      K0 (W2 <- S)
    //   6: S e4 | -      K1                 7: (S) | G e7    K0 (A ends; W3 <- S) 8: W4 e6 | S e7  K1
    //   9: S e8 | -      K1 (W4 <- S)      10: W2 e2 | W3 e5 K1 (B ends)
    always_comb begin
        x_as = 3'd5; x_bs = 3'd7; x_ae = 4'd8; x_be = 4'd1; x_md = 2'd1; x_an = 1'b0;
        if (rpq) case (c_st)
            4'd0: begin x_as = 3'd0; x_ae = 4'd0; x_bs = 3'd5; x_be = 4'd1; x_md = 2'd0; end
            4'd1: begin x_as = 3'd5; x_ae = 4'd2; x_bs = 3'd1; x_be = 4'd3; x_md = 2'd0; end
            4'd2: begin x_as = 3'd5; x_ae = 4'd4; x_bs = 3'd3; x_be = 4'd3; x_md = 2'd2; end
            4'd3: begin x_as = 3'd2; x_ae = 4'd6; x_bs = 3'd5; x_be = 4'd5; x_md = 2'd0; end
            4'd4: begin x_as = 3'd2; x_ae = 4'd0; x_bs = 3'd5; x_be = 4'd1; x_md = 2'd1; end
            4'd5: begin x_as = 3'd5; x_ae = 4'd8; x_bs = 3'd7; x_be = 4'd1; x_md = 2'd0; end
            4'd6: begin x_as = 3'd5; x_ae = 4'd4; x_bs = 3'd7; x_be = 4'd1; x_md = 2'd1; end
            4'd7: begin x_as = 3'd5; x_ae = 4'd8; x_an = 1'b1; x_bs = 3'd6; x_be = 4'd7; x_md = 2'd0; end
            4'd8: begin x_as = 3'd4; x_ae = 4'd6; x_bs = 3'd5; x_be = 4'd7; x_md = 2'd1; end
            4'd9: begin x_as = 3'd5; x_ae = 4'd8; x_bs = 3'd7; x_be = 4'd1; x_md = 2'd1; end
            default: begin x_as = 3'd2; x_ae = 4'd2; x_bs = 3'd3; x_be = 4'd5; x_md = 2'd1; end
        endcase
        else case (c_st)
            4'd0: begin x_as = 3'd0; x_ae = 4'd0; x_bs = 3'd5; x_be = 4'd1; x_md = 2'd0; end
            4'd1: begin x_as = 3'd5; x_ae = 4'd4; x_bs = 3'd1; x_be = 4'd3; x_md = 2'd0; end
            4'd2: begin x_as = 3'd2; x_ae = 4'd6; x_bs = 3'd5; x_be = 4'd7; x_md = 2'd0; end
            4'd3: begin x_as = 3'd5; x_ae = 4'd4; x_bs = 3'd3; x_be = 4'd3; x_md = 2'd1; end
            4'd4: begin x_as = 3'd2; x_ae = 4'd0; x_bs = 3'd5; x_be = 4'd7; x_md = 2'd1; end
            4'd5: begin x_as = 3'd5; x_ae = 4'd2; x_bs = 3'd6; x_be = 4'd1; x_md = 2'd2; end
            4'd6: begin x_as = 3'd7; x_ae = 4'd0; x_bs = 3'd5; x_be = 4'd5; x_md = 2'd0; end
            4'd7: begin x_as = 3'd5; x_ae = 4'd8; x_bs = 3'd7; x_be = 4'd1; x_md = 2'd0; end
            4'd8: begin x_as = 3'd4; x_ae = 4'd6; x_bs = 3'd5; x_be = 4'd5; x_md = 2'd1; end
            4'd9: begin x_as = 3'd5; x_ae = 4'd8; x_bs = 3'd7; x_be = 4'd1; x_md = 2'd1; end
            default: begin x_as = 3'd6; x_ae = 4'd2; x_bs = 3'd7; x_be = 4'd1; x_md = 2'd1; end
        endcase
    end
    wire         ks     = p_sliced && !cw2 && !c_b[0] && (c_b != 4'd8) && !(a_str && n_str);
    wire         c_first= (c_k == 8'd0);
    wire         pa_rw  = p_pa && (c_g != 8'd0);            // pool: weights kept from the first group
    wire         c_last = (c_k == nchunks - 8'd1);
    // FC stream walk: beat in the row, its lane / entry, channel in the group
    logic [5:0]  fs_b;
    logic [3:0]  fs_e, fs_ch;
    logic [2:0]  fs_l;
    wire         fs_lastb = (fs_b == fs_kbt - 6'd1);
    // packed FC decoder: header byte at the stream head, blocks in the next 31 bytes
    wire [2:0]   pk_c0   = rs0_data[2:0], pk_c1 = rs0_data[5:3];      // width - 1
    wire         pk_pair = rs0_data[6];
    wire         pk_two  = pk && pk_pair;
    wire [4:0]   pk_need = 5'd3 + {1'b0, pk_c0, 1'b0} + (pk_pair ? ({1'b0, pk_c1, 1'b0} + 5'd2) : 5'd0);
    wire [255:0] pk_win  = {8'd0, rs0_data2, rs0_data[127:8]};
    logic [127:0] pk_b1, pk_d0, pk_d1;
    always_comb begin
        begin case (pk_c0)
            3'd2: pk_b1 = pk_win[48  +: 128];
            3'd3: pk_b1 = pk_win[64  +: 128];
            3'd4: pk_b1 = pk_win[80  +: 128];
            3'd5: pk_b1 = pk_win[96  +: 128];
            3'd6: pk_b1 = pk_win[112 +: 128];
            default: pk_b1 = pk_win[128 +: 128];
        endcase end
        for (int i = 0; i < 16; i++) begin
            begin case (pk_c0)
                3'd2: pk_d0[i*8 +: 8] = {{5{pk_win[i*3+2]}}, pk_win[i*3 +: 3]};
                3'd3: pk_d0[i*8 +: 8] = {{4{pk_win[i*4+3]}}, pk_win[i*4 +: 4]};
                3'd4: pk_d0[i*8 +: 8] = {{3{pk_win[i*5+4]}}, pk_win[i*5 +: 5]};
                3'd5: pk_d0[i*8 +: 8] = {{2{pk_win[i*6+5]}}, pk_win[i*6 +: 6]};
                3'd6: pk_d0[i*8 +: 8] = {pk_win[i*7+6], pk_win[i*7 +: 7]};
                default: pk_d0[i*8 +: 8] = pk_win[i*8 +: 8];
            endcase end
            begin case (pk_c1)
                3'd2: pk_d1[i*8 +: 8] = {{5{pk_b1[i*3+2]}}, pk_b1[i*3 +: 3]};
                3'd3: pk_d1[i*8 +: 8] = {{4{pk_b1[i*4+3]}}, pk_b1[i*4 +: 4]};
                3'd4: pk_d1[i*8 +: 8] = {{3{pk_b1[i*5+4]}}, pk_b1[i*5 +: 5]};
                3'd5: pk_d1[i*8 +: 8] = {{2{pk_b1[i*6+5]}}, pk_b1[i*6 +: 6]};
                3'd6: pk_d1[i*8 +: 8] = {pk_b1[i*7+6], pk_b1[i*7 +: 7]};
                default: pk_d1[i*8 +: 8] = pk_b1[i*8 +: 8];
            endcase end
        end
    end
    // the issue ends its row: a pair covering the last beat, or the last single
    wire         fs_endb = pk ? ((fs_b + {5'd0, pk_pair}) == fs_kbt - 6'd1) : fs_lastb;
    wire  [3:0]  fs_en   = fs_e + (pk_two ? 4'd2 : 4'd1);
    wire         fu_c = p_fs && p_fu && (c_g != 16'd0);    // uniform: the group skips C_MULT / C_SHIFT

    // staged weight / parameter write (see the gated banks below)
    logic         wg_v;
    logic [2:0]   wg_l;
    logic [3:0]   wg_e;
    logic [127:0] wg_d;
    logic [31:0]  mult_r  [16];
    logic [7:0]   shift_r [16];
    logic [127:0] wreg    [NL][KE];

    logic [4:0] need;
    always_comb begin
        case (cst)
            C_BIAS, C_MULT, C_SHIFT: need = ((c_gc - c_j) >= 5'd4) ? 5'd16 : 5'({c_gc - c_j, 2'b00});
            C_WGT:   need = c_dl ? 5'd16 : p_dw ? 5'd9 : p_fs ? (fs_lastb ? fs_lastn : 5'd16) : (p_sm && !p_smo && !sm2) ? c_sml : ((c_e == c_kb - 4'd1) ? c_lastn : 5'd16);
            C_PIX:   need = pk ? pk_need : p_fs ? (fs_lastb ? fs_lastn : 5'd16) :
                            cw2 ? ((s_in && s_v) ? 5'd16 : 5'd0) :
                            p_sliced ? ((a_str || (ks && n_str)) ? 5'd16 : 5'd0) :
                            (win_src || !c_tv) ? 5'd0 : sm2 ? ((c_b == 4'd0) ? 5'd25 : 5'd20) :
                            (p_sm && !p_smo) ? (c_sml + (pv2 ? p_pixstep[4:0] : 5'd0)) :
                            (p_dw ? c_gc : ((c_b == c_kb - 4'd1) ? c_lastn : 5'd16));
            default: need = 5'd0;
        endcase
    end
    logic [127:0] rmask;
    always_comb for (int i = 0; i < 16; i++) rmask[i*8 +: 8] = (5'(i) < need) ? 8'hff : 8'h00;
    wire [7:0] peek_at = (cst != C_PIX) ? 8'd0 : p_sm ? c_po : p_dsm ? d_off : 8'd0;
    wire       wsk_c   = wsk && (cst == C_WGT) && !((c_e == 4'd8) && (c_j == c_gc - 5'd1));   // (v58_RS)
    wire [8:0] need_at = (need == 5'd0) ? 9'd0 : wsk_c ? 9'd32 : ({1'b0, peek_at} + {4'd0, need});
    assign rs0_peek = peek_at;
    wire have0 = ({1'b0, rs0_avail} >= need_at) && ((need != 5'd0) || (cst == C_PIX && (win_src || !c_tv || p_sliced)));

    // activation operand of the current tap: raw int8 bytes from the window or the stream (the
    // input offset enters as in_off * sum(w) through the accumulator start value, see oc below);
    // a padding tap is the zero point -in_off (zero after the offset; in_off = -zero point is in
    // [-127, 128], so -in_off is an int8)
    logic [127:0] str9, pix9, pixb;
    // (v58_DS) p_spt: the gather copy's bytes P holds as padding -> p_spt_pv (pixel c_ox outside [t0, t0 + nv),
    // the first p_spt_l bytes of output row 0, the last p_padr bytes of the last output row)
    wire          spt_px = (c_ox < p_spt_t0) || ({1'b0, c_ox} >= ({1'b0, p_spt_t0} + {1'b0, p_spt_nv}));
    wire          spt_r0 = (c_oy == 8'd0);
    wire          spt_rl = (c_oy == p_ho - 8'd1);
    logic [127:0] spt_d;   // (bytes 0..3 only: the gather's L is 4)
    always_comb begin
        spt_d = rs0_data;
        for (int i = 0; i < 4; i++)
            if (p_spt && (spt_px || (spt_r0 && (3'(i) < p_spt_l)) || (spt_rl && ((5'(i) + p_padr) >= need)))) spt_d[i*8 +: 8] = p_spt_pv;
    end
    wire  [7:0]   padv = 8'd0 - p_inoff[7:0];
    // (v28D) p_spd: implicit padding of a plain strip walk (see the port); the mask depends on
    // registers only (c_iy, c_b, c_ox, strip state), the byte select sits behind the FIFO read
    wire  signed [10:0] sp_row  = 11'(c_iy) + $signed({7'd0, c_b});
    wire          sp_rpad = (sp_row < 11'sd0) || (sp_row >= $signed({3'd0, p_hin}));
    wire  [4:0]   sp_need = c_sml + (pv2 ? p_pixstep[4:0] : 5'd0);
    // (v40K rs-conv1-pack) sm2: byte i of a packed beat is window row sr, position sq, read at stream byte i or i + 9
    // (beat 0: rows 0 / 1 at bytes 0..8 / 18..24; beat 1: rows 1 / 2 at bytes 0..1 / 11..19; rmask is all ones,
    // need >= 16); its implicit-padding mask uses (sr, sq). Beat 1 bytes 11..15 carry stream bytes 20..24: their
    // weights are 0. Other modes: unchanged (row c_b, position i).
    wire  [255:0] sd2 = {rs0_data2, rs0_data};
    logic [2:0]   sm2_rp;                                  // window row r outside the input
    always_comb for (int r = 0; r < 3; r++)
        sm2_rp[r] = ((11'(c_iy) + 11'(r)) < 11'sd0) || ((11'(c_iy) + 11'(r)) >= $signed({3'd0, p_hin}));
    logic [15:0]  sp_pm;
    always_comb for (int i = 0; i < 16; i++) begin
        logic [1:0] sr; logic [4:0] sq; logic rp;
        sr = (c_b == 4'd0) ? ((i < 9) ? 2'd0 : 2'd1) : ((i < 2) ? 2'd1 : 2'd2);
        sq = !sm2 ? 5'(i) : (c_b == 4'd0) ? 5'((i < 9) ? i : i - 9) : 5'((i < 2) ? i + 7 : i - 2);
        rp = sm2 ? sm2_rp[sr] : sp_rpad;
        sp_pm[i] = p_spd && (sm2 || (5'(i) < sp_need)) &&
                   (rp || ((c_ox == 8'd0) && (sq < p_padl)) || (c_rend && ((sq + p_padr) >= sp_need)));
    end
    always_comb for (int i = 0; i < 16; i++) begin
        logic hi;
        hi = sm2 && (i >= 9 || (c_b[0] && i >= 2));
        str9[i*8 +: 8] = sp_pm[i] ? padv : ((hi ? sd2[(i + 9)*8 +: 8] : rs0_data[i*8 +: 8]) & rmask[i*8 +: 8]);
    end
    // one window mux per slot: cw2 maps its sources onto it (window column 2 of rows 0..3), plus W4 / G
    wire         a_v   = cw2 ? ((c_fill || x_as == 3'd5) ? s_v : !(rpq && x_as <= 3'd4 && c_pz[x_as])) : c_tv;
    wire         a_w   = cw2 ? (!c_fill && x_as[2:2] == 1'b0) : win_src;
    wire  [1:0]  a_r   = cw2 ? x_as[1:0] : c_rr[1:0];
    wire  [1:0]  a_c   = cw2 ? 2'd2 : (c_brw ? 2'(c_fx) : 2'(c_fx + p_sw));
    wire         b_v   = cw2 ? ((x_bs == 3'd5) ? s_v : !(rpq && x_bs <= 3'd3 && c_pz[x_bs[1:0]])) : n_tv;
    wire         b_w   = cw2 ? (x_bs[2:2] == 1'b0) : n_src;
    wire  [1:0]  b_r   = cw2 ? x_bs[1:0] : n_rr[1:0];
    wire  [1:0]  b_c   = cw2 ? 2'd2 : (n_brw ? 2'(n_fx) : 2'(n_fx + p_sw));
    // (v28A) window taps slot A can read: an even tap t of a K-split pair or a single tap from the window
    // (an odd tap issues alone only when it reads the stream; tap 8 never comes from the window): dual-row B
    // rows (1,0) / (1,2) / (2,1), sliding taps (column + stride) (0,1) / (0,2) / (1,2) / (2,1) / (2,2) /
    // (3,1) / (3,2), cw2 W0 / W2 (0,2) / (2,2)
    function automatic logic [127:0] wtap(input logic [1:0] r, input logic [1:0] c, input logic [127:0] w01,
            input logic [127:0] w02, input logic [127:0] w10, input logic [127:0] w12, input logic [127:0] w21,
            input logic [127:0] w22, input logic [127:0] w31, input logic [127:0] w32);
        wtap = (w01 & {128{r == 2'd0 && c == 2'd1}}) | (w02 & {128{r == 2'd0 && c == 2'd2}}) |
               (w10 & {128{r == 2'd1 && c == 2'd0}}) | (w12 & {128{r == 2'd1 && c == 2'd2}}) |
               (w21 & {128{r == 2'd2 && c == 2'd1}}) | (w22 & {128{r == 2'd2 && c == 2'd2}}) |
               (w31 & {128{r == 2'd3 && c == 2'd1}}) | (w32 & {128{r == 2'd3 && c == 2'd2}});
    endfunction
    always_comb begin
        if (!a_v) pix9 = {16{padv}};
        else if (a_w) pix9 = wtap(a_r, a_c, win[0][1], win[0][2], win[1][0], win[1][2], win[2][1], win[2][2], win[3][1],
                                  win[3][2]);
        else if (cw2 && !c_fill && x_as == 3'd4) pix9 = nrow[1];
        else if (cw2 && !c_fill && x_as == 3'd6) pix9 = nrow[0];
        else pix9 = str9;
        // K-split: tap t+1 (slot B)
        if (!b_v) pixb = {16{padv}};
        // (v28A) slot B's window tap is the odd tap t + 1 of a K-split pair (t = 0, 2, 4, 6: taps 1, 3, 5, 7
        // at (0,1), (1,0), (1,2), (2,1)), x48's tap 8, or a cw2 W1 / W3 step: from the window only (1,1) /
        // (2,0) / (2,2) (dual-row B rows), (0,2) / (1,1) / (1,2) / (2,2) / (3,2) (sliding: column + stride),
        // (1,2) / (3,2) (cw2)
        else if (b_w) pixb = (win[0][2] & {128{b_r == 2'd0}}) |
                             (win[1][1] & {128{b_r == 2'd1 && b_c == 2'd1}}) | (win[1][2] & {128{b_r == 2'd1 && b_c == 2'd2}}) |
                             (win[2][0] & {128{b_r == 2'd2 && b_c == 2'd0}}) | (win[2][2] & {128{b_r == 2'd2 && b_c == 2'd2}}) |
                             (win[3][2] & {128{b_r == 2'd3}});
        else if (cw2 && x_bs == 3'd6) pixb = nrow[0];
        else pixb = str9;
        if (xiss) pixb = pix9;                             // (B's tap 5 is A's tap 8)
    end

    // pixel pairs: the second pixel's beat = the stream beat shifted down by p_pixstep (< 8) bytes
    logic [127:0] pixq;
    always_comb begin
        logic [127:0] t1, t2;
        t1   = p_pixstep[0] ? {8'd0, str9[127:8]} : str9;
        t2   = p_pixstep[1] ? {16'd0, t1[127:16]} : t1;
        pixq = p_pixstep[2] ? {32'd0, t2[127:32]} : t2;
    end

    // ---- partial-sum staging for passes k > 0: 8 x int32 = two pops of S1 per pixel ----
    logic [255:0] ps_in;
    logic         ps_half, ps_valid;
    wire          ps_need = (!c_first && !onepix) || fsa;
    logic         ps_hold;                                 // the consumed partial sum is still read (s2)
    logic         ps_b, ps_fmt;                            // (v28R) p24: the next fill is a pair's B / B's format in ps_lo, ps_hi
    logic         rr_pend, rr_got;                         // (v28R) rr: the pair's residual is the next S1 beat / is in sc
    wire          ps_fill = ps_need && !ps_valid && !ps_hold && (rs1_avail >= 8'd16) && (cst == C_PIX) && !rr_pend;
    // fused ADD: the residual bytes of a drain are popped with it
    logic         fz_pop;
    // (v46RS) p_fzp: residual pops of the current pixel (2 bytes each) and A / B; A's last pair also pops the 8
    // skipped bytes
    logic [1:0]   fp_pc;
    logic         fp_b;
    wire          fp_skip = p_fzp && !fp_b && (fp_pc == 2'd3);
    logic [1:0]   fz_k;
    // fz_s0: a finished pixel's residual bytes are next on S0 (sc_pend: taps that need S0 wait);
    // they wait in a 2-entry queue until that pixel has drained
    // p_rp (paired residual): a pair's 16 residual bytes (A, B) are next on S0 before the pair's taps
    // (sc_pd2); they fill two entries at once
    logic         sc_pend, sc_pd2;
    logic [1:0]   sc_wp, sc_rp;
    logic [2:0]   sc_n;
    logic [63:0]  sc_mem [4];
    wire          rpc    = p_rp && fz_s0 && scw && p_dr && c_last && !p_rr;
    wire          sc_v   = (sc_n != 3'd0);
    wire          sc_pop = sc_pend && (sc_n < 3'd2) && (rs0_avail >= {3'd0, c_gc}) &&
                           (cst == C_PIX || cst == C_PASSEND);
    wire          sc_pop2 = sc_pd2 && (sc_n <= 3'd2) && (rs0_avail >= 8'd16) && (cst == C_PIX || cst == C_PASSEND);
    wire          sc_pop2r = rr_pend && (sc_n <= 3'd2) && (rs1_avail >= 8'd16) && (cst == C_PIX || cst == C_PASSEND);
    wire          sc_pp2 = sc_pop2 || sc_pop2r;
    logic         ps_w2, dr_b;                             // (v28R) rr pass k0: B's third push (its pair's residual) / B drains
    wire          drain_rr = ps_w2 && (rs1_avail < 8'd16);  // ... waits for the residual bytes
    assign rs1_pop = (ps_fill || sc_pop2r || (res_v && res_ps && ps_w2 && !drain_rr)) ? 5'd16 :
                     fz_pop ? ({3'd0, fz_k} + (fp_skip ? 5'd8 : 5'd0)) : 5'd0;

    // ---- result slot: accumulators captured at the end of a pixel, drained 2 per cycle ----
    logic signed [31:0] res [16];
    logic         res_v, res_ps;
    logic [4:0]   res_i, res_gc;
    logic         ps_w1;                                   // psum mode: second half pending
    logic [4:0]   gap;                                     // cycles since the last 'last beat'
    logic         pe_v, pf_go, pf_all;                     // (v39S) pool fusion (see the drain)
    logic [14:0]  pf_ea, pf_eb;
    // Units C and D (npu_drq) take only channels with mult == 0 or 2^30 <= mult < 2^31 and
    // -11 <= shift <= 0; a quad of results whose third or fourth channel is another one ("slow")
    // drains on units A and B alone in two cycles.
    logic [15:0]  rk;                                      // channel C..H cannot take (k % 8 >= 2)
    wire  [4:0]   c_gce = (p_pp && !pv2) ? 5'd8 : c_gc;    // results of this pixel (pair)
    logic [3:0]   c_sq;                                    // slow quads of the issuing group
    always_comb for (int q = 0; q < 4; q++)
        c_sq[q] = (rk[4*q+2] && (5'(4*q+2) < c_gce)) || (rk[4*q+3] && (5'(4*q+3) < c_gce));
    // fast octets of the issuing group (all eight results present, channels 2..7 narrow-capable):
    // one drain cycle each, else its quads (one cycle each, two if slow)
    logic [1:0]   c_fo;
    logic [4:0]   c_dq;
    always_comb begin
        c_dq = 5'd0;
        for (int o = 0; o < 2; o++) begin
            c_fo[o] = RQ8 && (5'(8*o+8) <= c_gce) && (rk[8*o +: 8] == 8'd0);
            if (c_fo[o]) c_dq = c_dq + 5'd1;
            else c_dq = c_dq + 5'((5'(8*o) < c_gce) ? 1 : 0) + 5'(c_sq[2*o]) +
                               5'((5'(8*o+4) < c_gce) ? 1 : 0) + 5'(c_sq[2*o+1]);
        end
    end
    wire  [4:0]   dcyc = p_dwf ? 5'd4 : p_fs ? 5'd1 : (p_fz && c_last) ? ((c_gc + 5'd1) >> 1) :        // drain cycles per pixel
                                                     // (a fused conv's earlier chunk passes drain partial sums)
                         (g16 || c_last) ? c_dq : p_rr ? 5'd3 : 5'd2;
    // the last beat of a pixel may only issue when its results will find the slot free and the
    // write FIFO has room for everything still travelling
    // fused ADD: the residual bytes of the in-flight drain and of this pixel must already be in
    // S1, so a drain never waits for them (its results are captured on a fixed schedule)
    wire [5:0]  fz_have = 6'(res_v ? (res_gc - res_i) : 5'd0) + 6'(c_gc);
    // write-FIFO space: conv / depthwise drains reserve their pixel's bytes when its last beat
    // issues (released as they are pushed); the FC stream, the fused ADD and the pool (only its
    // last window's results are written) keep the static bound
    logic [7:0] wres;
    wire        wr_dyn = !p_fs && !p_fz && !p_cp && !p_pa && (!p_pf || p_pft);   // (v46DS) pool tee: pixels reserved
    wire [7:0]  w_need = !c_last ? (p_rr ? (c_hb ? 8'd40 : 8'd24) : p_p24 ? 8'd24 : 8'd32) : {3'd0, c_gce};
    wire        w_room = p_dwf ? ({1'b0, wd_space} >= ({1'b0, wres} + (p_dwt ? 9'd32 : 9'd64))) :   // (v46DS) tee: exact reservations      // (v37_1a) room for npu_dwf's outputs
                         // (v49_RS) a teed op: every push finds 16 more free bytes
                         wr_dyn ? ({1'b0, wd_space} >= ({1'b0, wres} + {1'b0, w_need} + (p_tee ? 9'd16 : 9'd0))) :
                                  (wd_space >= ((p_fz2 ? 8'd80 : 8'd48) + (p_tee ? 8'd16 : 8'd0)));
    wire last_ok  = (gap >= ((p_fz && c_gc < 5'd4) ? 5'd4 : dcyc)) && w_room &&
                    (fz_s0 ? (!c_last ? (!p_rr || !c_hb || (rs1_avail >= 8'd16)) :
                              (!sc_pend && (!p_rr || rr_got) && (!res_v || ((res_gc - res_i) <= ((rpc || p_rr) ? 5'd8 : 5'd4))))) :
                             (!p_fz || (rs1_avail >= ({2'd0, fz_have} + (p_fzp ? 8'd8 : 8'd0)))));
    // (fz_s0: a drain may have waited for its residual bytes; the next pixel's results must not
    // land before it ends: at most 4 results left when the last beat issues)
    wire a_last   = cw2 ? (!c_fill && (c_st == 4'd7 || c_st == 4'd10)) :
                    (c_hb && c_x5) ? (c_b == 4'd4) : (c_b == c_kb - 4'd1);   // last beat of (group A of) a pixel
    wire        c_r4  = !p_v3 || ((c_oy + 16'd2) < p_ho);   // (v25M) v3: window row 4 is walked
    wire  [2:0] c_lfy = p_v2 ? ((p_v3 || s2t) ? (c_r4 ? 3'd4 : 3'd3) : 3'd3) : 3'd2;   // last window row (depthwise)
    wire  [7:0] c_ostep = p_v3 ? 8'd3 : 8'd2;                     // v2 / v3 / dual-row: output rows per walk row
    wire  [7:0] c_istep = p_v3 ? 8'd3 : s2t ? 8'd4 : 8'd2;        // input rows per walk row group
    wire is_lastb = p_dw ? (c_fy == c_lfy && c_fx == 16'd2) : a_last;
    wire fin_beat = p_fs ? fs_endb : (a_last && (!onepix || c_last));   // its results leave the lanes
    wire c_isfst  = p_fs ? (fs_b == 6'd0) : cw2 ? (c_st == 4'd0 || c_st == (rpq ? 4'd2 : 4'd3)) : (xiss || ((c_b == 4'd0) && !(c_hb && c_x5)));   // an accumulator starts (psum / bias)
    wire first_ok = !c_isfst || !ps_need || ps_valid;
    // (v34P) a partial sum is used up by its pixel; FC stream: ps_in holds 8 channels (lane = channel % 8)
    wire ps_cons  = mac_go && c_isfst && ps_need && (!p_fs || fs_ch[2:0] == 3'd7 || {1'b0, fs_ch} == c_gc - 5'd1);
    wire fill_go  = (cst == C_PIX) && c_fill && have0;
    wire mac_go   = (cst == C_PIX) && !p_dw && !p_cp && !c_fill && have0 && first_ok && (!fin_beat || last_ok) && !wg_v && pf_go &&
                    !((sc_pend || sc_pd2) && (need != 5'd0));
    wire cp_go    = (cst == C_PIX) && p_cp && have0 && (wd_space >= 8'd16);

    // depthwise: walk (window rebuild) and the two MAC issues per completed window
    logic dw_pend, dw_h1p;                                 // window complete: h0 / h1 still to issue
    logic pw_first, pw_last;                               // pool: the pending window is the first / last
    logic dw_bp, dw_bp2;                                   // v2 / v3: second / third output row pending
    logic [1:0] dw_ri;                                     // window row offset of the issuing output row
    wire  dw_lock  = dw_pend || dw_h1p || dw_bp || dw_bp2;
    logic dw_lkn;                                          // see below
    // v2 sliding: the new column's taps of rows 0-2 wait in nrow; the row-3 tap commits all four
    wire  v2s      = p_v2 && (c_slide || p_ptr);   // (v25M: p_ptr: also a walk row's first position)
    wire  walk_go  = (cst == C_PIX) && p_dw && have0 &&
                     !(dw_lkn && (v2s ? (c_fy == c_lfy) : (c_fy == 3'd0 && c_fx == 16'd2)));
    // pool: a group's first window waits until the previous group's results have drained (the walk
    // overlaps the drain)
    wire  pa_hold  = p_pa && pw_first && (s1_v || s2_v || fin_v || res_v);
    wire  dw_go0   = (dw_pend || dw_bp || dw_bp2) && !dw_h1p && last_ok && !wg_v && !pa_hold;
    wire  dw_go1   = dw_h1p;
    // the window's commit may step in the cycle of its last MAC issue: the staged write lands at the
    // end of the next cycle, after that issue's s1 has read the window (dw_lkn: issues left after
    // this cycle; an h0 issue of a 16-channel group is followed by its h1)
    assign dw_lkn  = dw_go0 ? ((dw_pend && (dw_bp || dw_bp2)) || (dw_bp && dw_bp2) || (c_gc > 5'd8))
                            : (dw_pend || dw_bp || dw_bp2);
    wire  [1:0] go_nr = dw_pend ? 2'd0 : dw_bp ? 2'd1 : 2'd2; // output row of the next issue
    wire  [1:0] go_ri = dw_go0 ? go_nr : dw_ri;            // the MAC issued uses window rows go_ri .. + 2
    wire  step     = p_dw ? walk_go : p_cp ? cp_go : (mac_go || fill_go); // C_PIX advances one tap
    wire  s1_go    = mac_go || dw_go0 || dw_go1;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) wres <= '0;
        else wres <= wres + ((wr_dyn && (!p_dwf || p_dwt) && ((mac_go && fin_beat) || dw_go0)) ? w_need : 8'd0) +
                            ((p_dwf && dwo_st && (!p_dwt || t_dmy)) ? 8'd16 : 8'd0) +              // (v37_1a) DWF: npu_dwf reserves per output
                            // (v46DS) tee: the depthwise output a pointwise pixel's shift will start (pixel >= W + 1 of
                            // the pass) is reserved with the pixel (only the tail's dummy-shift outputs at their start)
                            ((p_dwt && mac_go && fin_beat && (c_px >= {4'd0, p_wo} + 12'd1)) ? 8'd16 : 8'd0) -
                            ((wr_dyn && wd_valid && !(p_pft && pe_on)) ? {3'd0, wd_n} : 8'd0);   // (v46DS) pooled bytes: unreserved
    end
    // no results of the previous pass still to capture or drain: its multipliers / shifts may be
    // replaced (assigned below)
    logic pidle, pgo;

    always_comb begin
        rs0_pop = 8'd0;
        case (cst)
            C_BIAS, C_MULT, C_SHIFT: if (pgo) rs0_pop = {3'd0, need};
            C_WGT: if (have0 && !pa_rw && !(c_dl && dwu_busy)) rs0_pop = wsk_c ? 8'd32 : {3'd0, need};   // (v58_RS wsk)
            C_PASSEND: if (sc_pop) rs0_pop = {3'd0, c_gc}; else if (sc_pop2) rs0_pop = 8'd16;
            C_PIX: if (sc_pop) rs0_pop = {3'd0, c_gc};
                   else if (sc_pop2) rs0_pop = 8'd16;
                   else if (p_dsm) begin
                       if (step && is_lastb && c_spend) rs0_pop = s_drop;
                   end else if (p_sm) begin
                       if (step && (c_b == c_kb - 4'd1) && c_spe) rs0_pop = c_ls ? p_dropl : p_dropf;
                   end else if (step) rs0_pop = {3'd0, need};
            default: ;
        endcase
    end

    // ---- MAC pipeline (double-pumped lanes): issue -> s1 operands -> slot A (high phase) and
    // slot B (low phase) dot products -> s2 accumulate ----
    // Every lane's dot product (npu_dpd: Booth + Dadda to 3 carry-save rows) is used twice per
    // cycle. Slot A takes an EVEN weight entry (0, 2, 4, 6, 8) and the slot-A activations, slot B
    // an ODD entry (1, 3, 5, 7) and the slot-B activations; an unused slot gets no entry (weights
    // 0, product 0). Issues: one beat on its entry's slot (generic), K-split (sliced conv: taps t
    // and t+1 into the same accumulator), paired groups (one beat, group A on slot A into acc[j],
    // group B on slot B into acc[8+j]); depthwise h0 / h1 on slots A / B.
    // Both slots' operands are posedge registers loaded in the issue cycle (weights per lane: wqa,
    // wqb; activations: s1_a, s1_b); a per-lane phase signal ph (1 in the high phase: a posedge
    // toggle flop XOR its negedge copy, replicated so that each copy drives few selects) picks
    // slot A or B in front of the dot. So every half-cycle path leaves a flop through one select
    // level (plus the per-lane depthwise window select). Slot A's rows are captured at the
    // negedge and summed in the low phase (pa); slot B's rows at the posedge (pb), summed in s2.
    // Rows are kept mod 2^21 (the dot product of raw int8 operands fits 20 bits signed).
    wire [3:0] m_e  = p_fs ? fs_e : p_dw ? {3'd0, dw_go1} : c_b;    // entry of a one-beat issue
    wire       m_ka = p_pg || !m_e[0] || pk;                               // slot A used
    wire       m_kb = p_pg || m_e[0] || ks || xiss || pk;                  // slot B used
    wire [2:0] m_ix = p_pg ? c_b[2:0] : m_e[3:1];                    // entry pair (2i, 2i+1)
    wire [4:0] sela = cw2 ? ((x_as == 3'd7 || x_an) ? 5'd0 : (5'd1 << x_ae[3:1])) : m_ka ? (5'd1 << m_ix) : 5'd0;
    wire [4:0] selb = cw2 ? ((x_bs == 3'd7) ? 5'd0 : (5'd1 << x_be[2:1])) :
                      xiss ? 5'b00100 : x48 ? 5'b10000 : m_kb ? (5'd1 << m_ix[1:0]) : 5'd0;   // entries 1, 3, 5, 7, 8
    // depthwise operand selects (elements 0..8 of a lane): bit 0 = shared activations (not
    // depthwise); per window row offset r (bits 1+8r ..): element class e%3 == 1: lo, hi;
    // e%3 == 0: lo, hi, column-1 lo (super pixel h1); e%3 == 2: lo, hi, column-1 hi (dsp h0)
    wire       d_x  = dw_go1 ^ p_dsp, d_dh = p_dsp && dw_go1, d_dl = p_dsp && !dw_go1;
    logic [24:0] dsel_d;
    always_comb begin
        dsel_d = '0;
        dsel_d[0] = !p_dw;
        for (int r = 0; r < 3; r++) begin
            logic rr;
            rr = p_dw && ((s2t ? {go_ri[0], 1'b0} : go_ri) == 2'(r));   // s2t: window rows 0-2, 2-4
            dsel_d[1+8*r] = rr && !dw_go1;          dsel_d[2+8*r] = rr && dw_go1;
            dsel_d[3+8*r] = rr && !d_dh && !d_x;    dsel_d[4+8*r] = rr && !d_dh && d_x;   dsel_d[5+8*r] = rr && d_dh;
            dsel_d[6+8*r] = rr && !d_dl && !d_x;    dsel_d[7+8*r] = rr && !d_dl && d_x;   dsel_d[8+8*r] = rr && d_dl;
        end
    end
    logic         s1_v, s1_first, s1_last, s1_h, s1_pp, s1_hb, s1_first2, s1_x;
    logic         s1_pf, s1_dr, s2_pf, s2_dr, fin_dr;      // pool: start from res / results leave
    logic [127:0] s1_a, s1_b;                                // slot A / B activations (16 raw int8)
    logic [3:0]   s1_ch, s2_ch, fin_ch;                    // fs: channel in the group
    logic [2:0]   s1_l, s2_l;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) s1_v <= 1'b0; else s1_v <= s1_go;
    end
    always_ff @(posedge clk) begin                         // no enable: s1_v marks validity
        s1_first <= cw2 ? (c_st == 4'd0) : p_fs ? (fs_b == 6'd0) : (p_dw || ((c_b == 4'd0) && (!onepix || c_first)));
        s1_first2 <= cw2 ? (c_st == (rpq ? 4'd2 : 4'd3)) :
                     xiss || (!p_fs && (p_dw || ((c_b == 4'd0) && (!onepix || c_first))) && !(c_hb && c_x5));
        s1_x     <= cw2 ? (x_md == 2'd2) : xiss;
        s1_pf    <= p_pa && !pw_first;
        s1_dr    <= !p_pa || pw_last;
        s1_last  <= p_dw || fin_beat;
        s1_h     <= dw_go1 || (p_fs && fs_ch[3]);
        s1_pp    <= p_pg;
        s1_hb    <= cw2 ? (x_md == 2'd1) : c_hb;
        s1_l     <= fs_l;
        s1_ch    <= fs_ch;
        s1_a     <= pk ? (fs_b[0] ? 128'd0 : pk_d0) : pix9;
        s1_b     <= pk ? (fs_b[0] ? pk_d0 : pk_pair ? pk_d1 : 128'd0) : (ks || xiss || cw2) ? pixb : p_pp ? pixq : pix9;
    end
    // (v26A: rows kept mod 2^20: a slot's dot product of raw int8 operands is in [-260096, 262144])
    logic        [19:0] pa [NL];                           // (v44A) slot A's rows summed, unsigned (= dot + 696320)
    logic [59:0]  pb [NL];
    localparam int NPH = 16;                               // phase copies per lane
    for (genvar j = 0; j < NL; j++) begin : g_lane
        // per-lane copies of the depthwise selects (no select net on a half-cycle path fans out
        // to all lanes; v22U s13 form): element e reads its bits at dslf[e*25 + k] = dsl[k]
        (* keep *) logic [24:0] dsl;
        (* keep *) always_ff @(posedge clk) dsl <= dsel_d;
        wire [9*25-1:0] dslf = {9{dsl}};
        // phase copies: each a posedge toggle flop and its negedge copy (structurally distinct
        // copies, so synthesis cannot merge them); ph = 1 after the posedge, 0 after the negedge
        logic [NPH-1:0] ph, phw;
        for (genvar k = 0; k < NPH; k++) begin : g_ph
            (* keep *) logic tp, tn;
            always_ff @(posedge clk) tp <= !tp;
            always_ff @(negedge clk) tn <= tp;
            assign ph[k] = tp ^ tn;
            (* keep *) wire pw = tp ^ tn;                   // (v26A: the weight phase select's own copy)
            assign phw[k] = pw;
        end
        // slot A / B selects of elements 0..8 as kept nets (each element's dslf bit 0 drives 2 gates, each select 8 bits;
        // without them synthesis folded dsl[0] into every bit's gate: a 144-load buffer chain)
        (* keep *) wire [8:0] sea, seb;
        for (genvar e = 0; e < 9; e++) begin : g_se
            (* keep *) logic d0n;                           // (v28R) element e's own copy of !dsl[0]
            (* keep *) always_ff @(posedge clk) d0n <= !dsel_d[0];
            assign sea[e] = !(d0n || !ph[e]);
            assign seb[e] = !(d0n || ph[e]);
        end
        logic [127:0] la;
        for (genvar i = 9; i < 16; i++) begin : g_lah
            assign la[i*8 +: 8] = ph[i] ? s1_a[i*8 +: 8] : s1_b[i*8 +: 8];
        end
        // balanced operand select (v22U s13): the slot term and each window row offset's terms are
        // kept nets, then one OR of four
        for (genvar e = 0; e < 9; e++) begin : g_la
            localparam int C = e % 3;
            localparam int D = e * 25;
            // (v28A: the window terms are ORed into one kept net vrr first; the phase-critical slot
            // term vs then meets it in one gate, a221o, instead of an OR of four kept nets)
            wire [7:0] vs;
            (* keep *) wire [7:0] vr0, vr1, vr2, vrr;
            assign vs = (s1_a[e*8 +: 8] & {8{sea[e]}}) | (s1_b[e*8 +: 8] & {8{seb[e]}});
            if (C == 1) begin : g_c1
                assign vr0 = (win[e/3][C][j*8 +: 8] & {8{dslf[D+1]}})  | (win[e/3][C][(8+j)*8 +: 8] & {8{dslf[D+2]}});
                assign vr1 = (win[e/3+1][C][j*8 +: 8] & {8{dslf[D+9]}})  | (win[e/3+1][C][(8+j)*8 +: 8] & {8{dslf[D+10]}});
                assign vr2 = (win[e/3+2][C][j*8 +: 8] & {8{dslf[D+17]}}) | (win[e/3+2][C][(8+j)*8 +: 8] & {8{dslf[D+18]}});
            end else if (C == 0) begin : g_c0
                assign vr0 = (win[e/3][C][j*8 +: 8] & {8{dslf[D+3]}})  | (win[e/3][C][(8+j)*8 +: 8] & {8{dslf[D+4]}}) |
                             (win[e/3][1][j*8 +: 8] & {8{dslf[D+5]}});
                assign vr1 = (win[e/3+1][C][j*8 +: 8] & {8{dslf[D+11]}}) | (win[e/3+1][C][(8+j)*8 +: 8] & {8{dslf[D+12]}}) |
                             (win[e/3+1][1][j*8 +: 8] & {8{dslf[D+13]}});
                assign vr2 = (win[e/3+2][C][j*8 +: 8] & {8{dslf[D+19]}}) | (win[e/3+2][C][(8+j)*8 +: 8] & {8{dslf[D+20]}}) |
                             (win[e/3+2][1][j*8 +: 8] & {8{dslf[D+21]}});
            end else begin : g_c2
                assign vr0 = (win[e/3][C][j*8 +: 8] & {8{dslf[D+6]}})  | (win[e/3][C][(8+j)*8 +: 8] & {8{dslf[D+7]}}) |
                             (win[e/3][1][(8+j)*8 +: 8] & {8{dslf[D+8]}});
                assign vr1 = (win[e/3+1][C][j*8 +: 8] & {8{dslf[D+14]}}) | (win[e/3+1][C][(8+j)*8 +: 8] & {8{dslf[D+15]}}) |
                             (win[e/3+1][1][(8+j)*8 +: 8] & {8{dslf[D+16]}});
                assign vr2 = (win[e/3+2][C][j*8 +: 8] & {8{dslf[D+22]}}) | (win[e/3+2][C][(8+j)*8 +: 8] & {8{dslf[D+23]}}) |
                             (win[e/3+2][1][(8+j)*8 +: 8] & {8{dslf[D+24]}});
            end
            assign vrr = vr0 | vr1 | vr2;
            assign la[e*8 +: 8] = vs | vrr;
        end
        // v28A: no weight operand registers (wqa / wqb, 256 flops per lane). The issue's entry selects are
        // registered (one-hot; slot A entries 0, 2, 4, 6, 8, slot B entries 1, 3, 5, 7, 8; NSC kept copies
        // per lane, each serving 16 / NSC bytes) and the two 5:1 selects read the weight banks directly in
        // front of the phase mux (the data had ~0.7 ns to spare over the phase select). The banks are
        // never written while an issued MAC is in s1: C_WGT is entered only from C_IDLE, or from
        // C_PASSEND with s1 empty and no depthwise issue pending; mac_go / dw_go0 wait for !wg_v.
        localparam bit FSL = (j == 5) || (j == 6);         // FC stream offset lanes
        localparam int NSC = 4;
        (* keep *) logic [127:0] wa_s, wb_s;               // (kept: the phase mux stays a separate stage)
        logic [127:0] wv;
        // FC stream offset lanes 5 / 6: their entry 8 holds the in_off constant (written before the op
        // starts, see fs0 / fs1), selected on slot A (packed: both slots)
        for (genvar c = 0; c < NSC; c++) begin : g_sc
            (* keep *) logic [4:0] saq, sbq;
            (* keep *) always_ff @(posedge clk) begin
                saq <= (FSL && p_fs) ? 5'b10000 : sela;
                sbq <= (FSL && p_fs) ? (pk ? 5'b10000 : 5'b00000) : selb;
            end
            localparam int BW = 128 / NSC;
            always_comb begin
                wa_s[c*BW +: BW] = '0; wb_s[c*BW +: BW] = '0;
                for (int k = 0; k < 5; k++) wa_s[c*BW +: BW] = wa_s[c*BW +: BW] | (wreg[j][2*k][c*BW +: BW] & {BW{saq[k]}});
                for (int k = 0; k < 4; k++) wb_s[c*BW +: BW] = wb_s[c*BW +: BW] | (wreg[j][2*k+1][c*BW +: BW] & {BW{sbq[k]}});
                wb_s[c*BW +: BW] = wb_s[c*BW +: BW] | (wreg[j][8][c*BW +: BW] & {BW{sbq[4]}});
            end
        end
        always_comb for (int i = 0; i < 16; i++) wv[i*8 +: 8] = phw[i] ? wa_s[i*8 +: 8] : wb_s[i*8 +: 8];
        logic [68:0] dr;
        logic [59:0] na;
        npu_dpd u_dot (.a(la), .w(wv), .r(dr));
        always_ff @(negedge clk) na <= {dr[46 +: 20], dr[23 +: 20], dr[0 +: 20]};
        // (v44A) the dpd rows are exact unsigned values (< 2^19 each; sum = dot + 696320 < 2^20): one fa_1 3:2 row and a
        // carry-select adder of fa_1 ripple blocks (npu_nas)
        wire [19:0] na_s;
        npu_nas u_nas (.a(na[0 +: 19]), .b(na[20 +: 19]), .c(na[40 +: 19]), .y(na_s));
        always_ff @(posedge clk) begin pa[j] <= na_s; pb[j] <= {dr[46 +: 20], dr[23 +: 20], dr[0 +: 20]}; end
    end
    logic         s2_v, s2_first, s2_last, s2_h, s2_pp, s2_hb, s2_first2, s2_x;
    // (v44A) accumulate as carry-save trees of sky130 fa_1 cells + one carry-select adder per accumulator (npu_acs.sv):
    // no resolved slot-B / slot A+B sums. Slot B's three rows -> one fa_1 3:2 row per lane (bsf: weight 2^i, bcf: 2^(i+1)),
    // shared by acc, acc2 and the FC stream sum; pa is slot A's rows summed unsigned. Both are dot + 696320 (exact).
    logic [NL*19-1:0] bsf, bcf;
    for (genvar j = 0; j < NL; j++) begin : g_bcs
        for (genvar i = 0; i < 19; i++) begin : g_b
            sky130_fd_sc_hd__fa_1 u_fa (.A(pb[j][i]), .B(pb[j][20+i]), .CIN(pb[j][40+i]), .SUM(bsf[j*19+i]), .COUT(bcf[j*19+i]));
        end
    end
    // accumulator start value, formed when the pixel's first beat is in s2: bias / partial sum
    // (bias_r and c_k are stable through a pass's MACs; ps_in is not refilled until that beat
    // has passed s2, see ps_hold); paired groups: acc[8+j] starts with channel 8+j's bias
    // + in_off * sum(w) of the pass (oc; not the FC stream, whose offset lanes add it). Bias and
    // partial sum are never both used (single passes - depthwise, paired groups, FC stream - have
    // c_first), so the start value is one add of oc and a select.
    logic signed [31:0] s2_init [NL];             // (v52BF: 32-bit again; v40K acc26 undone)
    // (v28R) partial sum of lane j: 32-bit words, or (p24) the 24-bit fields of A = {hi[63:0], lo} / B = {lo, hi[127:64]}
    wire  [191:0] ps24a = {ps_in[127:0], ps_in[255:192]}, ps24b = {ps_in[191:128], ps_in[127:0]};   // (v44A) ps_fmt 1 / 0
    logic signed [31:0] ps_v [NL];
    // (v39S) one register per channel (ob) holds the channel's bias word: the kernel's folded start value
    // b + in_off * sum(w) (FC stream: the bias alone). The first chunk pass starts from it, later passes from
    // their partial sums alone; the FC stream / pool without bias words start from 0; (v38M) a v34_P chunk op
    // (fsa, no bias words) from its S1 partial sums alone
    logic signed [31:0] ob [16];                           // lane j (+8: second half)
    // (v52BF) per channel: in_off * sum(w) of the current pass (ops with p_hf; g_hf below), added carry-save at every
    // accumulator start (npu_acsa / npu_acsb .o)
    logic signed [22:0] oc [16];
    // (v44A) per-lane copies of the start value's selects (each flop drives one lane's selects; a shared select
    // reached the trees through a chain of ~30 buffers): s2_h / s2_pf / ps_fmt exactly, the configuration
    // conditions (!c_first || fsa) / ((p_fs || p_pa) && !p_has_bias) and p_p24 one cycle late (constant while beats are in s2)
    logic [NL-1:0] s2_hl, s2_pfl, s2_cc0, s2_cc1, s2_c24, s2_cfm;
    for (genvar j = 0; j < NL; j++) begin : g_s2c
        (* keep *) logic hl, pfl, cc0, cc1;
        (* keep *) always_ff @(posedge clk) begin hl <= s1_h; pfl <= s1_pf; cc0 <= (!c_first || fsa); cc1 <= ((p_fs || p_pa || p_hf) && !p_has_bias); end
        assign s2_hl[j] = hl; assign s2_pfl[j] = pfl; assign s2_cc0[j] = cc0; assign s2_cc1[j] = cc1;
        assign s2_c24[j] = 1'b0; assign s2_cfm[j] = 1'b0;   // (v56_AR r_tie3: p24 tied off)
    end
    always_comb for (int j = 0; j < NL; j++)
        ps_v[j] = s2_c24[j] ? 32'($signed(s2_cfm[j] ? ps24a[j*24 +: 24] : ps24b[j*24 +: 24])) : $signed(ps_in[j*32 +: 32]);
    always_comb for (int j = 0; j < NL; j++)
        s2_init[j] = s2_pfl[j] ? res[{s2_hl[j], 3'(j)}] :
                     s2_cc0[j] ? ps_v[j] : (s2_cc1[j] ? 32'sd0 : ob[{s2_hl[j], 3'(j)}]);
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) s2_v <= 1'b0; else s2_v <= s1_v;
    end
    always_ff @(posedge clk) begin
        s2_first <= s1_first; s2_first2 <= s1_first2; s2_x <= s1_x; s2_last <= s1_last; s2_h <= s1_h; s2_l <= s1_l; s2_ch <= s1_ch; s2_pp <= s1_pp; s2_hb <= s1_hb;
        s2_pf <= s1_pf; s2_dr <= s1_dr;
    end
    logic signed [31:0] acc [NL], acc2 [NL];             // acc2: paired groups' group B
    logic         fin_v, fin_ps, fin_h, fin_pp, fin_hb;
    logic         fin_ppn, fin_hbn;                       // (v40K dmux)
    logic [4:0]   fin_gc;                                  // group size of the pass in the pipeline
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) fin_v <= 1'b0; else fin_v <= s2_v && s2_last;
    end
    always_ff @(posedge clk) begin fin_h <= s2_h; fin_ch <= s2_ch; fin_pp <= s2_pp; fin_hb <= s2_hb; fin_ppn <= !s2_pp; fin_hbn <= !s2_hb; fin_dr <= s2_dr; end
    // FC stream: the active activation lane plus the two offset lanes, into the channel's lane
    // (v44A) FC stream: the three lanes' unsigned rows (both slots) -> one fa_1 tree -> two rows fs_r1 + fs_r2 = their dot
    // products + 4538368 (exact), added carry-save into the channel lane's tree; lane l by registered one-hot selects
    (* keep *) logic [4:0] s2_lq;
    (* keep *) always_ff @(posedge clk) for (int u = 0; u < 5; u++) s2_lq[u] <= (u == 4) ? (s1_l >= 3'd4) : (s1_l == 3'(u));
    logic [57:0] fsl;
    always_comb begin
        fsl = '0;
        for (int u = 0; u < 5; u++) fsl = fsl | ({bcf[u*19 +: 19], bsf[u*19 +: 19], pa[u]} & {58{s2_lq[u]}});
    end
    wire [22:0] fs_r1, fs_r2;
    npu_fsu u_fsu (.p0(fsl[19:0]), .s0(fsl[38:20]), .c0(fsl[57:39]), .p1(pa[5]), .s1(bsf[5*19 +: 19]), .c1(bcf[5*19 +: 19]),
                   .p2(pa[6]), .s2(bsf[6*19 +: 19]), .c2(bcf[6*19 +: 19]), .r1(fs_r1), .r2(fs_r2));
    // accumulator clock gates: each enable is its own flop (one load, glitch-free); acc2 takes paired
    // groups' group B and dual-row B (per-lane copies of the B flag select its operands)
    logic acc_en, acc2_en;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin acc_en <= 1'b0; acc2_en <= 1'b0; end
        else begin acc_en <= s1_v && !s1_hb; acc2_en <= s1_v && (s1_pp || s1_hb || s1_x); end
    end
    wire acc_gclk = clk | ~acc_en;
    wire acc_gclk2 = clk | ~acc2_en;
    // FC stream: channel ch accumulates in lane ch % 8 (initialised with its bias, captured into
    // res[ch]); the other lanes take don't-care sums meanwhile (their channels were captured on the
    // edge the next channel's first beat lands, and are re-initialised before reuse)
    // (v44A) acc += FC stream channel lane: fs sum; paired / x: slot A; else slot A + slot B (npu_acsa:
    // x + (fs ? fs_r1 : 3 * 2^20 + pa) + (fs ? fs_r2 : slot B row (pp / x: 696320)) - 4538368)
    for (genvar j = 0; j < NL; j++) begin : g_acc
        // per-lane registered selects (complementary pairs: the selects are AND-ORs of flops, a22o-type gates that drive
        // the fa_1 inputs without a buffer): fq / fqn = s2_first, fj / fjn = FC stream channel lane (p_fs one cycle late),
        // sb = slot B used, sc = the constant instead (paired / x)
        (* keep *) logic fq, fqn, fj, fjn, sb, sc, fo;
        (* keep *) always_ff @(posedge clk) begin
            fq <= s1_first; fqn <= !s1_first;
            fo <= s1_first && p_hf && !p_fs;                // (v52BF) start + this pass's correction
            fj <= p_fs && (s1_ch[2:0] == 3'(j)); fjn <= !(p_fs && (s1_ch[2:0] == 3'(j)));
            sb <= !(s1_pp || s1_x) && !(p_fs && (s1_ch[2:0] == 3'(j)));
            sc <= (s1_pp || s1_x) && !(p_fs && (s1_ch[2:0] == 3'(j)));
        end
        wire [31:0] x = (s2_init[j] & {32{fq}}) | (acc[j] & {32{fqn}});
        wire [22:0] oa = oc[{s2_hl[j], 3'(j)}] & {23{fo}};   // (v52BF)
        wire [22:0] pv = (fs_r1 & {23{fj}}) | ({3'b011, pa[j]} & {23{fjn}});
        wire [22:0] bs = (fs_r2 & {23{fj}}) | {4'd0, bsf[j*19 +: 19] & {19{sb}}};
        wire [18:0] bc = (bcf[j*19 +: 19] & {19{sb}}) | (19'h55000 & {19{sc}});   // 2 x 19'h55000 = 696320
        wire [31:0] y;
        npu_acsa u_acs (.x(x), .o(oa), .p(pv), .s(bs), .c(bc), .y(y));

        logic signed [31:0] a1;
        always_ff @(posedge acc_gclk) a1 <= y;
        assign acc[j] = a1;
    end
    for (genvar j = 0; j < NL; j++) begin : g_acc2
        // (v44A) per-lane registered selects: hbl / hbn = s2_hb; one-hot start value g1 = s2_first2 && (hb || x),
        // g2 = s2_first2 && !(hb || x), g3 = !s2_first2
        (* keep *) logic hbl, hbn, g1, g2, g3, o1, o2;
        (* keep *) always_ff @(posedge clk) begin
            o1 <= s1_first2 && (s1_hb || s1_x) && p_hf && !p_fs;    // (v52BF) start from s2_init: + its channel's oc
            o2 <= s1_first2 && !(s1_hb || s1_x) && p_hf && !p_fs;   //          start from ob[8 + j]: + oc[8 + j]
            hbl <= s1_hb; hbn <= !s1_hb;
            g1 <= s1_first2 && (s1_hb || s1_x); g2 <= s1_first2 && !(s1_hb || s1_x) && !(p_hf && !p_fs && !p_has_bias); g3 <= !s1_first2;
        end
        logic signed [31:0] a2;
        // (v44A) acc2 += hb: slot A + slot B, else slot B (npu_acsb: x + (hb ? pa : 696320) + slot B row - 1392640)
        wire [19:0] p2 = (pa[j] & {20{hbl}}) | (20'd696320 & {20{hbn}});
        wire [31:0] x2 = (s2_init[j] & {32{g1}}) | (ob[NL+j] & {32{g2}}) | (a2 & {32{g3}});
        wire [31:0] y2;
        wire [22:0] ob2 = (oc[{s2_hl[j], 3'(j)}] & {23{o1}}) | (oc[NL+j] & {23{o2}});   // (v52BF)
        npu_acsb u_acs (.x(x2), .o(ob2), .p(p2), .s(bsf[j*19 +: 19]), .c(bcf[j*19 +: 19]), .y(y2));
        always_ff @(posedge acc_gclk2) a2 <= y2;
        assign acc2[j] = a2;
    end

    // ---- drain ----
    // the upper half (res[8..15], a depthwise / paired second group) is captured later than the
    // lower half: its drain waits for the capture
    logic res_hv;
    // (v37_1a) DWF: the paired-group results drain in the even slots (dpar 0), npu_dwf's octets in the odd ones
    always_ff @(posedge clk or negedge rst_n) if (!rst_n) dpar <= 1'b0; else dpar <= !dpar;
    always_ff @(posedge clk or negedge rst_n) begin                // (v46RS) p_fzp pop position
        if (!rst_n) begin fp_pc <= 2'd0; fp_b <= 1'b0; end
        else if (start) begin fp_pc <= 2'd0; fp_b <= 1'b0; end
        else if (fz_pop && p_fzp) begin fp_pc <= fp_pc + 2'd1; if (fp_pc == 2'd3) fp_b <= !fp_b; end
    end
    wire drain_rq = res_v && !res_ps && (!res_i[3] || res_hv) && !(p_dwf && dpar) &&
                    (!p_fz || (fz_s0 ? sc_v : (rs1_avail >= ({6'd0, fz_k} + (fp_skip ? 8'd8 : 8'd0)))));
    assign fz_k   = (res_i + 5'd1 < res_gc) ? 2'd2 : 2'd1;   // fused ADD: two results per drain cycle
    assign fz_pop = p_fz && !fz_s0 && drain_rq;
    wire   sc_done = fz_s0 && drain_rq && (res_i + 5'd2 >= res_gc);
    wire drain_ps = res_v &&  res_ps;
    assign rqa_v     = dwd || pe_v || (!fs_raw && drain_rq && !p_fz);
    wire   one_dr    = p_fs;                               // FC stream: one result per drain cycle (unit A)
    // (pool: two per cycle on A and B, as a slow quad; both divide with RG_MO)
    // fused ADD: two results per drain cycle on units C and D (every channel is one they can take: the
    // driver checks), A and B requantise the ADD's sums
    // a fast quad (res_i a multiple of 4): unit u takes result res_i + u; a slow quad: A and B
    // take res_i, res_i + 1 (res_i = quad + 0, + 2); fused ADD / FC stream: one per cycle on A
    wire [1:0] rsel = res_i[3:2];
    wire       r_slow = !one_dr && !p_fz && (p_pa || (res_i[1:0] != 2'd0) || (rk[{rsel, 2'd2}] && (res_i + 5'd2 < res_gc)) ||
                                                             (rk[{rsel, 2'd3}] && (res_i + 5'd3 < res_gc)));
    wire       r_oct  = RQ8 && !one_dr && !p_fz && !p_pa && (res_i[2:0] == 3'd0) && ((res_i + 5'd8) <= res_gc) &&
                        (rk[{res_i[3], 3'd0} +: 8] == 8'd0);
    wire [4:0] r_step = one_dr ? 5'd1 : r_oct ? 5'd8 : (r_slow || p_fz) ? 5'd2 : 5'd4;
    assign rqx_v     = (drain_rq && r_oct) || dwd;
    // (v37_1a) npu_dwf's octet: results 22-bit (saturated), multipliers / shifts of its own parameter registers
    logic [30:0] dmult [16];
    logic [3:0]  dshr [16];                                // -shift (0..11)
    logic [31:0] dw_acc [8], dw_mult [8];
    logic [7:0]  dw_shf [8];
    always_comb for (int u = 0; u < 8; u++) begin
        dw_acc[u]  = 32'($signed(dwd_acc[22*u +: 22]));
        dw_mult[u] = {1'b0, dmult[{dwd_k, 3'(u)}]};
        dw_shf[u]  = 8'd0 - {4'd0, dshr[{dwd_k, 3'(u)}]};
    end
    always_comb for (int u = 0; u < 4; u++) begin
        rqx_acc[u*32 +: 32]   = dwd ? dw_acc[4+u] : res[{res_i[3], 3'(4 + u)}];
        rqx_mult[u*31 +: 31]  = dwd ? dw_mult[4+u][30:0] : mult_r[{res_i[3], 3'(4 + u)}][30:0];
        rqx_shift[u*8 +: 8]   = dwd ? dw_shf[4+u] : shift_r[{res_i[3], 3'(4 + u)}];
    end
    assign rqb_v     = (drain_rq && !one_dr && !p_fz && (res_i + 5'd1 < res_gc)) || dwd || pe_v;
    assign rqc_v     = (drain_rq && !one_dr && (p_fz || (!r_slow && (res_i + 5'd2 < res_gc)))) || dwd;
    assign rqd_v     = (drain_rq && !one_dr && (p_fz ? (res_i + 5'd1 < res_gc) : (!r_slow && (res_i + 5'd3 < res_gc)))) || dwd;
    wire [1:0] cq    = p_fz ? {res_i[1], 1'b0} : 2'd2;      // quad position of unit C's result (D: + 1)
    // four results per cycle: unit u takes result res_i + u (res_i a multiple of 4); fused ADD:
    // one per cycle on unit A (result res_i)
    // shared selects: the quad rsel = res_i[3:2] once (four 4:1 selects), then A takes result
    // res_i[1:0] of it, B result {res_i[1], 1}, C / D results cq / cq + 1 (same values as indexing the
    // 16 results directly, fewer select levels in total)
    logic [31:0] q4_acc [4], q4_mult [4];
    logic [7:0]  q4_sh [4];
    always_comb for (int u = 0; u < 4; u++) begin
        q4_acc[u] = res[{rsel, 2'(u)}]; q4_mult[u] = mult_r[{rsel, 2'(u)}]; q4_sh[u] = shift_r[{rsel, 2'(u)}];
    end
    assign rqa_acc   = pe_on ? {17'd0, pf_ea} : dwd ? dw_acc[0]  : q4_acc[res_i[1:0]];
    assign rqa_mult  = dwd ? dw_mult[0] : q4_mult[res_i[1:0]];
    assign rqa_shift = dwd ? dw_shf[0]   : q4_sh[res_i[1:0]];
    assign rqb_acc   = pe_on ? {17'd0, pf_eb} : dwd ? dw_acc[1]  : q4_acc[{res_i[1], 1'b1}];
    assign rqb_mult  = dwd ? dw_mult[1] : q4_mult[{res_i[1], 1'b1}];
    assign rqb_shift = dwd ? dw_shf[1]   : q4_sh[{res_i[1], 1'b1}];
    assign rqc_acc   = dwd ? dw_acc[2]  : q4_acc[cq];
    assign rqc_mult  = dwd ? dw_mult[2] : q4_mult[cq];
    assign rqc_shift = dwd ? dw_shf[2]   : q4_sh[cq];
    assign rqd_acc   = dwd ? dw_acc[3]  : q4_acc[{cq[1], 1'b1}];
    assign rqd_mult  = dwd ? dw_mult[3] : q4_mult[{cq[1], 1'b1}];
    assign rqd_shift = dwd ? dw_shf[3]   : q4_sh[{cq[1], 1'b1}];

    // (v28R) pair swizzle: bytes 8..15 of a pair's result stream (A's high half) wait in sw_hold and
    // leave in front of byte 24's chunk (B's high half): pushed as A.lo, B.lo, A.hi, B.hi (memory order).
    // sw_c: bytes of requantised results pushed in drain order, mod 32 (chunks never straddle 8 bytes)
    logic [4:0]  sw_c;
    logic [63:0] sw_hold;
    wire  [63:0] sw_q = {rqx_q, rqd_q, rqc_q, rqb_q, rqa_q};
    wire  [4:0]  sw_n = (rqa_ov && !p_fz && !drain_ps && !cp_go) ?
                        (rqx_ov ? 5'd8 : rqd_ov ? 5'd4 : rqc_ov ? 5'd3 : rqb_ov ? 5'd2 : 5'd1) : 5'd0;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) sw_c <= '0;
        else if (start) sw_c <= '0;
        else sw_c <= sw_c + sw_n;
    end
    wire sw_hg = clk | ~(p_swz && rqa_ov && sw_c[4:3] == 2'd1);
    always_ff @(posedge sw_hg)
        case ({rqx_ov, rqd_ov, rqc_ov, rqb_ov})
            4'b0000: sw_hold <= {sw_q[7:0],  sw_hold[63:8]};
            4'b0001: sw_hold <= {sw_q[15:0], sw_hold[63:16]};
            4'b0011: sw_hold <= {sw_q[23:0], sw_hold[63:24]};
            4'b0111: sw_hold <= {sw_q[31:0], sw_hold[63:32]};
            default: sw_hold <= sw_q;
        endcase
    // write data: psums (16 B per push) or requantised bytes (1-2 per cycle, in lane order)
    always_comb begin
        wd_valid = 1'b0; wd_data = '0; wd_n = 5'd0;
        if (drain_ps) begin
            wd_valid = 1'b1; wd_n = p_p24 ? 5'd12 : 5'd16;
            for (int q = 0; q < 4; q++) wd_data[q*32 +: 32] = res[{1'b0, ps_w1, 2'(q)}];
            if (p_p24) begin
                wd_data = '0;
                for (int q = 0; q < 4; q++) wd_data[q*24 +: 24] = res[{1'b0, ps_w1, 2'(q)}][23:0];
            end
            if (ps_w2) begin wd_data = rs1_data; wd_n = 5'd16; wd_valid = !drain_rr; end   // (v28R) rr: B's pair residual
        end else if (fs_raw && drain_rq) begin                // (v34P) raw partial sum of one channel
            wd_valid = 1'b1; wd_data[31:0] = rqa_acc; wd_n = 5'd4;
        end else if (cp_go) begin
            wd_valid = 1'b1; wd_data = spt_d; wd_n = need;
        end else if (rqa_ov && !p_fz && (!p_pf || pe_on || p_pft)) begin   // (v46DS) pool tee: the pixels too
            wd_valid = !p_dwf || dpar;                     // (v37_1a) DWF: even-slot octets feed npu_dwf's chain
            wd_data[63:0] = {rqx_q, rqd_q, rqc_q, rqb_q, rqa_q};
            wd_n = rqx_ov ? 5'd8 : rqd_ov ? 5'd4 : rqc_ov ? 5'd3 : rqb_ov ? 5'd2 : 5'd1;
            if (p_dwt) begin                               // (v46DS) tee: whole 16-byte pieces (octet 1 + the held octet 0)
                wd_valid = t_pw || t_dw;
                wd_data  = (ps_w && ps_i != 12'd0) ? t_unit : t_data;          // (v49DS) wrap: {held high, low}
                wd_n     = (ps_w && ps_i == 12'd0) ? {1'b0, ag_d} : 5'd16;     // (v49DS) wrap piece 0: its low piece
            end
            if (p_swz && sw_c[4:3] == 2'd1) wd_valid = 1'b0;                 // (v28R) A.hi: held
            if (p_swz && sw_c == 5'd24) begin wd_data = {sw_q, sw_hold}; wd_n = wd_n + 5'd8; end
        end
        if (fl_go) begin wd_valid = 1'b1; wd_data = t_unit; wd_n = 5'd16 - {1'b0, ag_d}; end   // (v49DS) tee carry flush
    end

    // ---- (v39S) global average pool fused into the drain (p_pf; scope: see the driver) ----
    // a fast octet of the pass's requantised results leaves units A..H (pf_add): lane l adds u = q + 128 into
    // U[l] (first octet of a pixel, pf_pos 0) or U[l + 8] (second); the pass's first pixel starts the sums. After
    // p_npix pixels the sums leave in pairs (pf_ea / pf_eb = U[2i], U[2i+1]) through units A / B (pe_on); pf_eg
    // counts the groups whose pooled bytes are out (the next group's MACs wait for it)
    logic        pf_pos, pf_rdy;
    logic [6:0]  pf_px;
    logic [2:0]  pe_i;
    logic [3:0]  pe_o, pf_eg;
    logic        pe_iv;                                    // pairs still to issue
    wire         pf_add = p_pf && rqa_ov && !pe_on;
    wire         pf_fst = (pf_px == 7'd0);
    wire  [63:0] pf_q   = {rqx_q, rqd_q, rqc_q, rqb_q, rqa_q};
    assign pe_v   = pe_on && pe_iv;
    assign pf_go  = !p_pf || (pf_eg == c_g[3:0]);
    assign pf_all = !p_pf || (pf_eg == ngroups[3:0]);
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pf_pos <= 1'b0; pf_px <= '0; pf_rdy <= 1'b0; pe_on <= 1'b0; pe_iv <= 1'b0; pe_i <= '0; pe_o <= '0; pf_eg <= '0;
        end else if (start) begin
            pf_pos <= 1'b0; pf_px <= '0; pf_rdy <= 1'b0; pe_on <= 1'b0; pe_iv <= 1'b0; pf_eg <= '0;
        end else begin
            if (pf_add) begin
                pf_pos <= !pf_pos;
                if (pf_pos) begin
                    if (pf_px == p_npix[6:0] - 7'd1) begin pf_px <= '0; pf_rdy <= 1'b1; end
                    else pf_px <= pf_px + 7'd1;
                end
            end
            if (pf_rdy && !pe_on && (wd_space >= 8'd16)) begin
                pf_rdy <= 1'b0; pe_on <= 1'b1; pe_iv <= 1'b1; pe_i <= '0; pe_o <= '0;
            end
            if (pe_v) begin pe_i <= pe_i + 3'd1; if (pe_i == 3'd7) pe_iv <= 1'b0; end
            if (pe_on && rqa_ov) begin
                pe_o <= pe_o + 4'd1;
                if (pe_o == 4'd7) begin pe_on <= 1'b0; pf_eg <= pf_eg + 4'd1; end
            end
        end
    end
    // the sums live in the partial-sum staging registers (idle in a single-chunk op): U[l] = ps_lo[16 l +: 15],
    // U[l + 8] = ps_hi[16 l +: 15]; the emission reads them through the lane adders (byte 0): pair i = lanes
    // 2 (i % 4), + 1 of half i / 4
    wire          pf_h   = pe_on ? pe_i[2] : pf_pos;
    logic [14:0]  pf_u [8];
    logic [127:0] pf_wd;
    for (genvar l = 0; l < 8; l++) begin : g_pf
        wire [14:0] b = (pf_fst && !pe_on) ? 15'd0 : (pf_h ? ps_in[128 + 16*l +: 15] : ps_in[16*l +: 15]);
        assign pf_u[l] = b + ({7'd0, ~pf_q[l*8 + 7], pf_q[l*8 +: 7]} & {15{!pe_on}});
        assign pf_wd[16*l +: 16] = {1'b0, pf_u[l]};
    end
    assign pf_ea = pf_u[{pe_i[1:0], 1'b0}];
    assign pf_eb = pf_u[{pe_i[1:0], 1'b1}];

    assign pidle = !s1_v && !s2_v && !fin_v && !res_v && !dw_pend && !dw_h1p && !dw_bp && !dw_bp2;
    assign pgo   = (cst == C_BIAS || cst == C_MULT || cst == C_SHIFT) && have0 && pidle && !p_pa;

    // pass bookkeeping for the issuer barrier
    logic outs_left;                                       // results not yet pushed
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cst <= C_IDLE; c_g <= '0; c_px <= '0; c_k <= '0; c_j <= '0; c_e <= '0; c_b <= '0; c_dl <= 1'b0;
            ps_half <= 1'b0; ps_valid <= 1'b0; ps_hold <= 1'b0; ps_b <= 1'b0; ps_fmt <= 1'b0;
            rr_pend <= 1'b0; rr_got <= 1'b0; ps_w2 <= 1'b0; dr_b <= 1'b0; res_v <= 1'b0; res_hv <= 1'b0; res_ps <= 1'b0; res_i <= '0; res_gc <= '0;
            ps_w1 <= 1'b0; gap <= 5'd31; fin_ps <= 1'b0; fin_gc <= '0; c_cs <= '0; c_ox <= '0; c_fy <= '0; c_fx <= '0;
            dw_pend <= 1'b0; dw_h1p <= 1'b0; c_iy <= '0; c_ix <= '0; pw_first <= 1'b0; pw_last <= 1'b0;
            dw_bp <= 1'b0; dw_bp2 <= 1'b0; dw_ri <= '0; c_oy <= '0; c_hb <= 1'b0; c_x5 <= 1'b0; c_st <= '0; c_fl <= '0; c_pz <= '0;
            fs_b <= '0; fs_e <= '0; fs_l <= '0; fs_ch <= '0;
            c_sp <= '0; c_pb <= '0; c_po <= '0; c_ls <= 1'b0; s_ix0 <= '0; s_np <= '0;
        end else begin
            if (fz_s0 && c_last && step && is_lastb && !rpc && !p_rr) sc_pend <= 1'b1;
            else if (sc_pop) sc_pend <= 1'b0;
            if ((rpc && step && is_lastb && c_hb && (c_px != p_npix - 12'd1)) ||
                (cst == C_WGT && have0 && !p_fs && !p_dw && c_e == c_kb - 4'd1 && c_j == c_gc - 5'd1 && c_k != 8'd0 &&
                 p_rp && fz_s0 && scw && p_dr && c_last && !p_rr)) sc_pd2 <= 1'b1;
            else if (sc_pop2) sc_pd2 <= 1'b0;
            if (sc_pop) sc_wp <= sc_wp + 2'd1;
            if (sc_pp2) sc_wp <= sc_wp + 2'd2;
            if (sc_done) sc_rp <= sc_rp + 2'd1;
            sc_n <= sc_n + (sc_pop ? 3'd1 : 3'd0) + (sc_pp2 ? 3'd2 : 3'd0) - (sc_done ? 3'd1 : 3'd0);
            // (v28R) rr: B's partial sum filled -> its pair's residual beat is next on S1
            if (cst == C_WGT) begin rr_pend <= 1'b0; rr_got <= 1'b0; end
            else begin
                if (ps_fill && p_rr && ps_b) rr_pend <= 1'b1; else if (sc_pop2r) rr_pend <= 1'b0;
                if (sc_pop2r) rr_got <= 1'b1; else if (mac_go && fin_beat && c_hb) rr_got <= 1'b0;
            end
            if (ps_fill) begin
                if (!ps_half && !(p_p24 && ps_b)) ps_half <= 1'b1;
                else begin ps_half <= 1'b0; ps_valid <= 1'b1; end
            end
            if (ps_cons) ps_valid <= 1'b0;
            // (v28R) p24: pixels alternate A / B (complete pairs); B's partial sum is one more beat
            if (start) dr_b <= 1'b0;                     // (pairs complete per pass: dr_b is 0 at every pass end)
            if (cst == C_WGT) ps_b <= 1'b0;
            else if (ps_cons) begin ps_b <= p_p24 && !ps_b; ps_fmt <= ps_b; end
            ps_hold <= ps_cons;
            if ((mac_go && fin_beat) || dw_go0) begin gap <= 5'd1; fin_ps <= !c_last; fin_gc <= c_gce; end
            else if (gap != 5'd31) gap <= gap + 5'd1;
            // depthwise MAC issues
            if (dw_go0) begin
                if (dw_pend) dw_pend <= 1'b0; else if (dw_bp) dw_bp <= 1'b0; else dw_bp2 <= 1'b0;
                dw_ri <= go_nr;
                dw_h1p <= (c_gc > 5'd8);
            end
            if (dw_go1) dw_h1p <= 1'b0;
            // capture / drain (a depthwise second half lands in res[8..15] while the first drains)
            if (fin_v) res_hv <= fin_h || fin_pp;
            if (fin_v && (!fin_h || p_fs) && fin_dr) begin
                res_v <= 1'b1; res_ps <= fin_ps; ps_w1 <= 1'b0; ps_w2 <= 1'b0;
                // FC stream: one channel per result; res_i selects it (and its multiplier / shift)
                res_i <= p_fs ? {1'b0, fin_ch} : 5'd0; res_gc <= p_fs ? ({1'b0, fin_ch} + 5'd1) : fin_gc;
            end else if (drain_ps) begin
                if (!ps_w1) ps_w1 <= 1'b1;
                else if (p_rr && dr_b && !ps_w2) ps_w2 <= 1'b1;
                else if (!drain_rr) begin res_v <= 1'b0; ps_w2 <= 1'b0; dr_b <= !dr_b; end
            end else if (drain_rq) begin
                if (res_i + r_step >= res_gc) res_v <= 1'b0; else res_i <= res_i + r_step;
            end
            // strip mode offsets (A beats of a pixel read the stream at c_po)
            if (cst == C_PIX && p_sm && step) begin
                if (c_b == c_kb - 4'd1) begin
                    if (c_spe) begin
                        c_sp <= '0; c_pb <= '0; c_po <= '0;
                        c_ls <= ((p_wo - nx_oxp) <= {8'd0, p_sp});
                    end else begin
                        c_sp <= c_sp + c_adv[7:0]; c_pb <= c_pb + c_pps; c_po <= c_pb + c_pps;
                    end
                end else c_po <= c_po + (p_smo ? 8'd16 : sm2 ? 8'd25 : c_seg);
            end
            if (cst == C_PIX && p_dsm && step && is_lastb) begin
                c_sp <= c_spend ? 8'd0 : c_sp + 8'd1;
                if (c_spend) begin
                    s_ix0 <= (c_ox == p_wo - 16'd1) ? -$signed({9'd0, p_pl}) : (c_ix + $signed({1'b0, p_sw}));
                    s_np  <= ((p_wo - nx_ixw) < {8'd0, p_sp}) ? 8'(p_wo - nx_ixw) : p_sp;
                end
            end
            if (cst != C_PIX) begin
                c_sp <= '0; c_pb <= '0; c_po <= '0; c_ls <= (p_wo <= {8'd0, p_sp});
                s_ix0 <= -$signed({9'd0, p_pl});
                s_np  <= (p_wo < {8'd0, p_sp}) ? p_wo[7:0] : p_sp;
            end
            case (cst)
                C_IDLE: if (start) begin
                    c_g <= '0; c_k <= '0; c_j <= '0; c_cs <= '0; c_px <= '0; c_b <= '0;
                    c_e <= '0; cst <= p_cp ? C_PIX : C_WGT; fs_b <= '0;
                    if (p_spt) begin c_ox <= '0; c_oy <= '0; end   // (v58_DS r_spt)
                end
                C_BIAS: if (pgo) begin
                    if (c_j + 5'd4 >= c_gc) begin c_j <= '0; cst <= fu_c ? C_PIX : C_MULT; end else c_j <= c_j + 5'd4;
                end
                // (pool: no multiplier / shift words; the group's first window waits for the drain)
                C_MULT: if (p_pa) begin
                    if (pidle) begin c_j <= '0; cst <= C_PIX; end
                end else if (pgo) begin
                    if (c_j + 5'd4 >= c_gc) begin c_j <= '0; cst <= C_SHIFT; end else c_j <= c_j + 5'd4;
                end
                C_SHIFT: if (pgo) begin
                    if (c_j + 5'd4 >= c_gc) begin
                        c_j <= '0; cst <= (p_dwf && !c_dl) ? C_WGT : C_PIX;   // (v37_1a) DWF: the depthwise header next
                        c_dl <= p_dwf && !c_dl; c_e <= '0;
                        fs_b <= '0; fs_e <= '0; fs_l <= '0; fs_ch <= '0;
                    end else c_j <= c_j + 5'd4;
                end
                C_WGT: if (have0 || pa_rw) begin
                    if (c_dl) begin
                        // (v37_1a) depthwise weight beat c_j: 0..7 -> lane c_j entry 8, 8 -> w8x (once npu_dwf is idle)
                        if (!dwu_busy) begin
                            if (c_j[3]) begin c_j <= '0; cst <= C_BIAS; end
                            else c_j <= c_j + 5'd1;
                        end
                    end else if (p_fs) begin
                        // activation beat fs_b -> lane c_j, entry c_e
                        if (fs_lastb) begin
                            fs_b <= '0; c_j <= '0; c_e <= '0; c_px <= '0; cst <= p_has_bias ? C_BIAS : fs_raw ? C_PIX : C_MULT;
                        end else begin
                            fs_b <= fs_b + 6'd1;
                            if (c_e == (pk ? 4'd7 : 4'd8)) begin c_e <= '0; c_j <= c_j + 5'd1; end else c_e <= c_e + 4'd1;
                        end
                    end else if (p_dw) begin
                        // channel c of the group -> lane c%8, entry c/8
                        if (pa_rw || {c_e[0], c_j[2:0]} == 4'(c_gc - 5'd1)) begin
                            c_j <= '0; c_e <= '0; c_px <= '0; c_b <= '0; c_ox <= '0; c_fy <= '0; c_fx <= '0; c_oy <= '0;
                            c_iy <= -$signed({9'd0, p_pt}); c_ix <= -$signed({9'd0, p_pl});
                            cst <= p_pa ? C_PIX : p_has_bias ? C_BIAS : C_MULT;
                        end else if (c_j[2:0] == 3'd7) begin c_j <= '0; c_e <= c_e + 4'd1; end
                        else c_j <= c_j + 5'd1;
                    end else if (c_e == c_kb - 4'd1) begin
                        c_e <= '0;
                        if (c_j == c_gc - 5'd1) begin
                            c_j <= '0; c_px <= '0; c_b <= '0; c_ox <= '0; c_fy <= '0; c_fx <= '0; c_oy <= '0; c_hb <= 1'b0; c_x5 <= 1'b0; c_st <= '0;
                            c_fl <= rpq ? 3'd5 : 3'd0; c_pz <= {5{rpq}};   // (v35R) rpq: no fill, column 0 = padding
                            cst <= (c_k != 8'd0) ? C_PIX : p_has_bias ? C_BIAS : C_MULT;
                            c_iy <= -$signed({9'd0, p_pt}); c_ix <= -$signed({9'd0, p_pl});
                        end
                        else c_j <= c_j + 5'd1;
                    end else c_e <= c_e + 4'd1;
                end
                C_PIX: if (step && p_fs) begin
                    if (fs_endb) begin
                        fs_b <= '0; fs_e <= '0; fs_l <= '0;
                        if ({1'b0, fs_ch} == c_gc - 5'd1) begin fs_ch <= '0; cst <= C_PASSEND; end
                        else fs_ch <= fs_ch + 4'd1;
                    end else begin
                        fs_b <= fs_b + (pk_two ? 6'd2 : 6'd1);
                        if (pk ? fs_en[3] : (fs_e == 4'd8)) begin fs_e <= '0; fs_l <= fs_l + 3'd1; end else fs_e <= fs_en;
                    end
                end else if (step && cw2) begin
                    if (rpq && !c_fill && c_st[0] && c_st <= 4'd9) c_pz[c_st[3:1]] <= 1'b0;
                    if (c_fill) c_fl <= c_fl + 3'd1;
                    else if (c_st != 4'd10) c_st <= c_st + 4'd1;
                    else begin                                 // pair done
                        c_st <= '0;
                        c_ox <= c_rend ? 16'd0 : c_ox + 16'd1;
                        if (c_rend) begin
                            c_iy <= c_iy + 17'sd4; c_ix <= -$signed({9'd0, p_pl}); c_oy <= c_oy + 16'd2;
                            c_fl <= rpq ? 3'd5 : 3'd0; c_pz <= {5{rpq}};
                        end else c_ix <= c_ix + $signed({1'b0, p_sw});
                        if (c_px + 16'd2 == p_npix) begin c_px <= '0; cst <= C_PASSEND; end
                        else c_px <= c_px + 16'd2;
                    end
                end else if (step) begin
                    if (ks) begin                              // K-split: two taps
                        c_fx <= (c_fx == 16'd0) ? 16'd2 : (c_fx == 16'd1) ? 16'd0 : 16'd1;
                        if (c_fx != 16'd0) c_fy <= c_fy + 16'd1;
                    end else if (v2s && (s2t || p_ptr)) begin   // column by column
                        if (c_fy == c_lfy) begin c_fy <= '0; c_fx <= c_fx + 2'd1; end
                        else c_fy <= c_fy + 3'd1;
                    end else if (c_fx == 16'd2) begin c_fx <= p_dw ? c_fx0 : 16'd0; c_fy <= c_fy + 16'd1; end
                    else c_fx <= c_fx + 16'd1;
                    if (is_lastb && p_dr && !c_hb && c_bok) begin    // dual-row: A done, B next
                        c_b <= '0; c_fx <= '0; c_fy <= '0; c_hb <= 1'b1; c_px <= c_px + 16'd1; c_x5 <= xiss;
                    end else if (is_lastb) begin
                        c_b <= '0; c_fx <= p_dw ? nx_fx0 : 16'd0; c_fy <= '0; c_hb <= 1'b0; c_x5 <= 1'b0;
                        if (p_dw) dw_pend <= 1'b1;
                        pw_first <= (c_px == 12'd0); pw_last <= (c_px == p_npix - 12'd1);
                        if (p_v2) dw_bp <= (c_oy + 16'd1) < p_ho;
                        if (p_v3) dw_bp2 <= (c_oy + 16'd2) < p_ho;
                        c_ox <= c_rend ? 16'd0 : c_ox + c_adv;
                        if (c_rend) begin
                            c_iy <= c_iy + ((p_v2 || p_dr) ? $signed({1'b0, c_istep}) : $signed({1'b0, p_sw})); c_ix <= -$signed({9'd0, p_pl});
                            c_oy <= c_oy + ((p_v2 || p_dr) ? c_ostep : 16'd1);
                        end
                        else c_ix <= c_ix + $signed({1'b0, p_sw});
                        if (c_px + c_adv - 16'd1 == p_npix - 16'd1) begin c_px <= '0; cst <= C_PASSEND; end
                        else c_px <= c_px + c_adv;
                    end else if (c_hb && c_x5 && c_b == 4'd2) begin   // cross: B skips 4|5 (5 is done) ...
                        c_b <= 4'd6; c_fy <= 2'd2; c_fx <= 2'd0;
                    end else if (c_hb && c_x5 && c_b == 4'd6) begin   // ... and ends with 4|8
                        c_b <= 4'd4; c_fy <= 2'd1; c_fx <= 2'd1;
                    end else c_b <= c_b + (ks ? 4'd2 : 4'd1);
                end
                // the next pass may load its weights once the last MAC has left s1; its parameters
                // wait for the drain (C_BIAS..C_SHIFT); the op ends once everything has drained
                C_PASSEND: if (CHN && start) begin             // (v42DS) chained op: as C_IDLE's start (the old tail drains on)
                    c_g <= '0; c_k <= '0; c_j <= '0; c_cs <= '0; c_px <= '0; c_b <= '0;
                    c_e <= '0; cst <= p_cp ? C_PIX : C_WGT; fs_b <= '0;
                    if (p_spt) begin c_ox <= '0; c_oy <= '0; end   // (v58_DS r_spt)
                end else if (!sc_pend && !sc_pd2 && !rr_pend && ((c_last && c_g == ngroups - 16'd1) ? (pidle && !outs_left && !sc_v && !dwu_busy && !dwc_v && pf_all) : (!s1_v && !dw_pend && !dw_h1p && !dw_bp && !dw_bp2))) begin
                    if (!c_last) begin c_k <= c_k + 8'd1; c_cs <= c_cs + 20'(KCH); c_e <= '0; c_j <= '0; cst <= C_WGT; end
                    else begin
                        c_k <= '0; c_cs <= '0;
                        if (c_g == ngroups - 16'd1) cst <= C_IDLE;
                        else begin c_g <= c_g + 16'd1; c_j <= '0; c_e <= '0; cst <= p_fs ? (p_has_bias ? C_BIAS : (p_fu || fs_raw) ? C_PIX : C_MULT) : C_WGT; end
                    end
                end
                default: cst <= C_IDLE;
            endcase
        end
    end
    // fused ADD: residual bytes travel alongside the 4-stage requant of their results
    logic [15:0] fzd [2];                                  // (v26A: two stages; the scalers start 2 cycles after the drain)
    // (v26A: residual queue entries written through gated clocks)
    for (genvar q = 0; q < 4; q++) begin : g_sc
        wire hi = sc_pp2 && (sc_wp + 2'd1 == 2'(q));
        wire g  = clk | ~(((sc_pop || sc_pp2) && sc_wp == 2'(q)) || hi);
        logic [63:0] e;
        wire [127:0] sd = !p_rr ? rs0_data : rs1_data;     // (v28R) rr: residual beats come on S1
        wire lo = (sc_pop || sc_pp2) && sc_wp == 2'(q);       // (v40K dmux: exclusive with hi, not its complement)
        always_ff @(posedge g) e <= ({64{hi}} & sd[127:64]) | ({64{lo}} & sd[63:0]);
        assign sc_mem[q] = e;
    end
    always_ff @(posedge clk) begin
        fzd[0] <= fz_s0 ? {sc_mem[sc_rp][{res_i[2:1], 1'b1}*8 +: 8], sc_mem[sc_rp][{res_i[2:1], 1'b0}*8 +: 8]} : rs1_data[15:0];
        fzd[1] <= fzd[0];
    end
    assign fz_b   = fzd[1][7:0];
    assign fz_b2  = fzd[1][15:8];

    // requant results still in the 4-stage requant pipeline
    logic [2:0] rqp;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) rqp <= '0;
        else rqp <= rqp + (((drain_rq && !fs_raw) || dwd) ? 3'd1 : 3'd0) - ((p_fz ? rqc_ov : (rqa_ov && !pe_on)) ? 3'd1 : 3'd0);
    end
    assign outs_left = (rqp != '0);

    // captures
    // (paired groups: both halves at once, the upper from the group-B accumulators)
    for (genvar h = 0; h < 2; h++) begin : g_res
        wire gclk = clk | ~(fin_v && ((h == 0) ? !fin_h : (fin_h || fin_pp)));
        logic signed [31:0] rb [NL];
        always_ff @(posedge gclk) for (int j = 0; j < NL; j++)
            rb[j] <= ({32{(h == 1) ? fin_pp : fin_hb}} & acc2[j]) | ({32{(h == 1) ? fin_ppn : fin_hbn}} & acc[j]);
        for (genvar j = 0; j < NL; j++) begin : g_rj
            assign res[h*NL + j] = rb[j];
        end
    end
    // per-channel bias / multiplier / shift: staged one cycle (the weight stage's data register),
    // written through gated clocks; read only from the first MAC on
    logic       pm_v;
    logic [1:0] pm_a;                                      // 0 bias, 1 mult, 2 shift
    logic [4:0] pm_j;
    logic       pm_d;                                      // (v37_1a) DWF: the depthwise parameters
    logic [4:0] pm_n;                                      // group size, registered: the gated-clock
                                                           // enables below see flops only
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) pm_v <= 1'b0;
        else pm_v <= pgo;
    end
    always_ff @(posedge clk) begin
        pm_a <= (cst == C_BIAS) ? 2'd0 : (cst == C_MULT) ? 2'd1 : 2'd2;
        pm_j <= c_j;
        pm_d <= c_dl;
        pm_n <= c_gc;
    end
    // (v51_D) DWF parameter check: multiplier 0 or in [2^30, 2^31), shift in [-11, 0], for every loaded channel
    logic [3:0] dwf_bl;
    always_comb for (int u = 0; u < 4; u++) begin
        logic [31:0] x;
        x = wg_d[u*32 +: 32];
        dwf_bl[u] = (5'({pm_j[3:2], 2'(u)}) < pm_n) &&
                    ((pm_a == 2'd1) ? (x[31] || (!x[30] && x != 32'd0)) :
                     (pm_a == 2'd2) ? !(x == 32'd0 || (&x[31:4] && x[3:0] >= 4'd5)) : 1'b0);
    end
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) dwf_bad <= 1'b0;
        else dwf_bad <= pm_v && pm_d && (|dwf_bl);
    end
    // (v39S) bias words are final start values (offline / kernel fold): ob loads them as they are
    logic [31:0] bw [4];
    always_comb for (int u = 0; u < 4; u++) bw[u] = wg_d[u*32 +: 32];
    logic [15:0] bwr;                                      // bias write of channel k (this edge)
    for (genvar k = 0; k < 16; k++) begin : g_pm
        wire wr_k = pm_v && !pm_d && (pm_j[3:2] == 2'(k / 4)) && (5'(k) < pm_n);
        assign bwr[k] = wr_k && pm_a == 2'd0;
        wire gm = clk | ~(wr_k && pm_a == 2'd1);
        wire gs = clk | ~(wr_k && pm_a == 2'd2);
        // multipliers are in [0, 2^31) and shifts in [-31, 30] (TFLM): 31 and 6 bits are kept
        logic [30:0] mk;
        logic [5:0]  sk;
        always_ff @(posedge gm) mk <= wg_d[(k % 4)*32 +: 31];
        always_ff @(posedge gs) sk <= wg_d[(k % 4)*32 +: 6];
        assign mult_r[k] = {1'b0, mk}; assign shift_r[k] = {{2{sk[5]}}, sk};
        if (k % 8 >= 2) begin : g_rk
            // channel of unit C / D: flags kept when its multiplier / shift is written
            logic mz, mr, sr;
            always_ff @(posedge gm) begin
                mz <= (wg_d[(k % 4)*32 +: 32] == 32'd0);
                mr <= (wg_d[(k % 4)*32 + 30 +: 2] != 2'b01);
            end
            always_ff @(posedge gs) sr <= wg_d[(k % 4)*32 + 7] ? (wg_d[(k % 4)*32 +: 8] < 8'hf5)
                                                              : (wg_d[(k % 4)*32 +: 8] != 8'd0);
            assign rk[k] = !mz && (mr || sr);
        end else begin : g_nrk
            assign rk[k] = 1'b0;
        end
    end
    // (v26A: the two partial-sum halves are written through gated clocks, no enable muxes)
    // (the gate enables use their own copy of c_k != 0, one cycle late: c_k changes only at a pass
    //  start / end, at least one C_WGT cycle before C_PIX; c_first itself has a large fanout)
    (* keep *) logic ps_nk;
    always_ff @(posedge clk) ps_nk <= (c_k != 5'd0) || fsa;
    wire ps_fg = ps_nk && !ps_valid && !ps_hold && (rs1_avail >= 8'd16) && (cst == C_PIX) && !rr_pend;
    // (v39S) p_pf: the staging registers hold the fused pool's sums (written by the octet pushes)
    wire ps_g0 = clk | ~((ps_fg && !ps_half) || (pf_add && !pf_pos));
    wire ps_g1 = clk | ~((ps_fg && ps_half) || (pf_add && pf_pos));
    logic [127:0] ps_lo, ps_hi, ps_mx;
    h_mx #(128) u_psmx (.a0(rs1_data), .a1(pf_wd), .s(p_pf), .x(ps_mx));   // (v43H) = p_pf ? pf_wd : rs1_data
    always_ff @(posedge ps_g0) ps_lo <= ps_mx;
    always_ff @(posedge ps_g1) ps_hi <= ps_mx;
    assign ps_in = {ps_hi, ps_lo};
    // Window / row-buffer writes are staged one cycle: registered one-hot enables drive gated
    // clocks; the data is s1_a (= pix9 of the previous cycle), a row-buffer entry or a window
    // neighbour, all read one cycle later than unstaged (no source changes in between). Readers:
    // sliced taps come >= 3 beats after a commit, depthwise MACs read the window in s1 (after the
    // walk's commit, which waits for the MAC issues). K-split: tap t+1 (slot B, s1_b) writes its
    // nrow entry or commits its row (column 1 then comes from tap t, s1_a) in the same step.
    logic [3:0]  wq_n;                                      // nrow[k] write
    logic [2:0]  wq_nb, wq_na;                                 // nrow[k] <- s1_b
    logic        wq_f1, wq_b2;                              // win[r][1] <- s1_a, win[r][2] <- s1_b
    logic [14:0] wq_w;                                      // win[r][c] write (bit 3r + c)
    logic        wq_s0;                                     // win[r][0] <- 0 nrow[0], 1 win[r][1]
    logic        wq_s1;                                     // win[r][1] <- 0 nrow[1], 1 win[r][2]
    logic        wq_s2;                                     // win[r][2] (r < 3) <- 0 s1_a, 1 nrow[r]
    logic        wq_l3;                                     // v3 commit at the row-3 tap: win[3][2] <- s1_a
    logic        wq_xn, wq_xa;                             // (v40K dmux) win[r][2] <- nrow[r] / s1_a
    wire         wv2 = step && v2s;
    wire         wsl = step && slw && !v2s && !cw2;
    // cw2 window writes: column 0 of row r (win[r][2]; row 4: nrow[1]) <- the fill tap / the step's
    // column-2 tap; G (nrow[0]) <- the step-2 / step-7 tap
    wire         wc2 = step && cw2 && (c_fill || (rpq ? (c_st[0] && c_st <= 4'd9) : (c_st >= 4'd5 && c_st <= 4'd9)));
    wire  [2:0]  wc2r = c_fill ? c_fl : rpq ? c_st[3:1] : 3'(c_st - 4'd5);
    wire         wcg = step && cw2 && !c_fill && (rpq ? (c_st == 4'd4) : (c_st == 4'd2 || c_st == 4'd7));
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin wq_n <= '0; wq_w <= '0; end
        else begin
            for (int k = 0; k < 3; k++)
                wq_n[k] <= (wv2 && c_fy == 3'(k)) || (wsl && c_fx[1:0] == 2'(k)) ||
                           (wsl && ks && n_fx[1:0] == 2'(k)) || (wcg && k == 0) || (wc2 && wc2r == 3'd4 && k == 1);
            wq_n[3] <= wv2 && (p_v3 || s2t) && (c_fy == 3'd3);
            for (int k = 0; k < 15; k++)
                wq_w[k] <= (wv2 && (c_fy == c_lfy) && (k < 12 || p_v3 || s2t)) || (wsl && c_fx == 16'd2 && c_rr == 3'(k / 3)) ||
                           (wsl && ks && n_fx == 16'd2 && n_rr == 3'(k / 3)) || (wc2 && k % 3 == 2 && wc2r == 3'(k / 3) && k < 12);
        end
    end
    always_ff @(posedge clk) begin
        wq_s0 <= v2s || (p_dw && c_fx0 != 16'd0);         // (v28A: 1 = column 1, 0 = nrow[0])
        wq_s1 <= v2s || (p_dw && c_fx0 == 16'd2);
        wq_s2 <= v2s;
        wq_l3 <= !c_r4;
        for (int k = 0; k < 3; k++) wq_nb[k] <= (ks && n_fx[1:0] == 2'(k)) || (cw2 && k == 0 && c_st == (rpq ? 4'd4 : 4'd2) && !c_fill);
        for (int k = 0; k < 3; k++) wq_na[k] <= !((ks && n_fx[1:0] == 2'(k)) || (cw2 && k == 0 && c_st == (rpq ? 4'd4 : 4'd2) && !c_fill));   // (v40K dmux)
        wq_f1 <= ks && n_fx == 16'd2 && !x48;
        wq_b2 <= (ks && n_fx == 16'd2) || (cw2 && !c_fill && (rpq ? (c_st == 4'd3) : (c_st == 4'd6 || c_st == 4'd8)));
        // (v40K dmux) one-hot selects of win[r][2] (r < 4): separately registered (ABC maps a22o + a21o)
        wq_xn <= !((ks && n_fx == 16'd2) || (cw2 && !c_fill && (rpq ? (c_st == 4'd3) : (c_st == 4'd6 || c_st == 4'd8)))) && (v2s);
        wq_xa <= !((ks && n_fx == 16'd2) || (cw2 && !c_fill && (rpq ? (c_st == 4'd3) : (c_st == 4'd6 || c_st == 4'd8)))) && !(v2s);
    end
    for (genvar k = 0; k < 4; k++) begin : g_nrow
        wire gclk = clk | ~wq_n[k];
        logic [127:0] q;
        if (k < 3) begin : g_nb
            always_ff @(posedge gclk) q <= ({128{wq_nb[k]}} & s1_b) | ({128{wq_na[k]}} & s1_a);
        end else begin : g_n3                              // (v3 walk only)
            always_ff @(posedge gclk) q <= s1_a;
        end
        assign nrow[k] = q;
    end
    // (v37_1a) DWF: the window registers are npu_dwf's 13-pixel chain (position k = win[k / 3][2 - k % 3], 0 = newest):
    // every chain shift moves q2 -> q1 -> q0 -> the next row's q2; row 0's q2 takes the arriving pixel (dw_px)
    wire          dw_sh;                                   // chain shift at the end of this cycle
    wire  [127:0] dw_px;
    for (genvar r = 0; r < 5; r++) begin : g_win
        wire chs = dw_sh && (r < 4);                       // (row 4: only q2, position 12)
        wire gc0 = clk | ~(wq_w[3*r] | chs);
        wire gc1 = clk | ~(wq_w[3*r+1] | chs);
        wire gc2 = clk | ~(wq_w[3*r+2] | dw_sh);
        logic [127:0] q0, q1, q2;
        always_ff @(posedge gc0) q0 <= (wq_s0 || p_dwf) ? q1 : nrow[0];
        always_ff @(posedge gc1) q1 <= (wq_f1 && !p_dwf) ? s1_a : (wq_s1 || p_dwf) ? q2 : nrow[1];
        if (r == 0) begin : g_c20
            always_ff @(posedge gc2) q2 <= ({128{p_dwf}} & dw_px) | ({128{wq_b2 && !p_dwf}} & s1_b) |
                                           ({128{wq_xn && !p_dwf}} & nrow[r]) | ({128{wq_xa && !p_dwf}} & s1_a);
        end else if (r < 3) begin : g_c2
            always_ff @(posedge gc2) q2 <= ({128{p_dwf}} & win[r-1][0]) | ({128{wq_b2 && !p_dwf}} & s1_b) |
                                           ({128{wq_xn && !p_dwf}} & nrow[r]) | ({128{wq_xa && !p_dwf}} & s1_a);
        end else if (r == 3) begin : g_c23
            wire c3 = (p_v3 || s2t) && !wq_l3;
            always_ff @(posedge gc2) q2 <= ({128{p_dwf}} & win[r-1][0]) | ({128{wq_b2 && !p_dwf}} & s1_b) |
                                           ({128{wq_xn && c3 && !p_dwf}} & nrow[3]) | ({128{(wq_xa || (wq_xn && !c3)) && !p_dwf}} & s1_a);
        end else begin : g_c2l
            always_ff @(posedge gc2) q2 <= p_dwf ? win[r-1][0] : s1_a;
        end
        assign win[r][0] = q0; assign win[r][1] = q1; assign win[r][2] = q2;
    end
    // the pass's requantised octets (even slots) in drain order: octet 0 waits in dw_lo, octet 1 shifts the pixel in
    logic [3:0]  dw_oq;                                    // res_i[3] of the drains 1..4 cycles ago
    always_ff @(posedge clk) dw_oq <= {dw_oq[2:0], res_i[3]};
    // (gate enables see flops only: a registered copy of p_dwf; an op's first cycle has no requant output)
    logic        dwf_q;
    always_ff @(posedge clk or negedge rst_n) if (!rst_n) dwf_q <= 1'b0; else dwf_q <= p_dwf;
    wire         dw_ov = dwf_q && rqa_ov && !dpar;
    wire [63:0]  dw_q  = {rqx_q, rqd_q, rqc_q, rqb_q, rqa_q};
    wire         dw_lg = clk | ~(dw_ov && !dw_oq[3]);
    logic [63:0] dw_lo;
    always_ff @(posedge dw_lg) dw_lo <= dw_q;
    assign dw_px = {dw_q, dw_lo} ^ {16{8'h80}};            // offset-added (in_off = 128): padding = 0
    // (v46DS) tee: npu_dwf's octet index travels with its 4-stage requant; octet 0 waits in dt_lo
    logic [3:0]  dk_q;
    always_ff @(posedge clk) dk_q <= {dk_q[2:0], dwd_k};
    wire         dt_lg = clk | ~(dwf_q && rqa_ov && dpar && !dk_q[3]);
    logic [63:0] dt_lo;
    always_ff @(posedge dt_lg) dt_lo <= dw_q;
    assign       t_data = {dw_q, dpar ? dt_lo : dw_lo};
    assign       t_dmy  = dw_shd;
    assign       t_pw  = dwf_q && rqa_ov && !dpar && dw_oq[3];   // pointwise pixel complete (= dw_shp)
    assign       t_dw  = dwf_q && rqa_ov &&  dpar && dk_q[3];    // depthwise output complete
    wire         t_push = p_dwt && (t_pw || t_dw);
    // (v49DS) rotated tee, wrap group through a carry, per stream: piece i >= 1 is pushed as {high piece of pixel i - 1
    // (held in cy_p / cy_d: lanes 4..15 of the stream's last wrap piece), low piece of pixel i} = one aligned line;
    // piece 0 as its low piece (ag_d bytes); after the last one a flush push of the held high piece (16 - ag_d bytes)
    logic [11:0] pp_i, pd_i;
    logic [5:0]  pp_g, pd_g;
    logic [95:0] cy_p, cy_d;
    logic        fl_p, fl_d;
    assign       ps_w   = ag_tw && ({2'd0, (t_pw ? pp_g : pd_g)} == ngroups - 8'd1);
    assign       ps_i   = t_pw ? pp_i : pd_i;
    assign       fl_go  = p_dwt && (fl_p || fl_d) && !t_push;
    wire  [95:0] cy     = (fl_go ? fl_p : t_pw) ? cy_p : cy_d;
    assign       t_unit = ag_d[3] ? {t_data[95:0], cy[95:64]} : {t_data[31:0], cy};
    wire         t_tag  = t_push || fl_go;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin fl_p <= 1'b0; fl_d <= 1'b0; end
        else if (start) begin fl_p <= 1'b0; fl_d <= 1'b0; end
        else begin
            if (t_push && ps_w && (ps_i == p_npix - 12'd1)) begin if (t_pw) fl_p <= 1'b1; else fl_d <= 1'b1; end
            if (fl_go) begin if (fl_p) fl_p <= 1'b0; else fl_d <= 1'b0; end
        end
    end
    always_ff @(posedge clk) begin
        if (start) begin pp_i <= '0; pp_g <= '0; pd_i <= '0; pd_g <= '0; end
        else if (t_push) begin
            if (t_pw) begin if (pp_i == p_npix - 12'd1) begin pp_i <= '0; pp_g <= pp_g + 6'd1; end else pp_i <= pp_i + 12'd1; end
            else begin if (pd_i == p_npix - 12'd1) begin pd_i <= '0; pd_g <= pd_g + 6'd1; end else pd_i <= pd_i + 12'd1; end
        end
        if (t_push && ps_w) begin if (t_pw) cy_p <= t_data[127:32]; else cy_d <= t_data[127:32]; end
    end
    // tag queue (push order of the pieces; at most 16 pieces sit in the 256-byte FIFO without a command)
    logic [15:0] tq;
    logic [3:0]  tq_w, tq_r;
    wire         t_pop = p_dwt && dwc_v && wc_ready;
    assign       tq_hd = tq[tq_r];
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin tq_w <= '0; tq_r <= '0; end
        else begin
            if (t_tag) tq_w <= tq_w + 4'd1;
            if (t_pop) tq_r <= tq_r + 4'd1;
        end
    end
    always_ff @(posedge clk) if (t_tag) tq[tq_w] <= fl_go ? fl_p : !dpar;
    // per-stream position (pixel i of group g, raster order within a group); the op's start sets them
    logic [11:0] tp_i, tw_i;
    logic [5:0]  tp_g, tw_g;
    wire         tp_l = (tp_i == p_npix - ((ag_tw && ({2'd0, tp_g} == ngroups - 8'd1)) ? 12'd0 : 12'd1));   // (v49DS) wrap: + flush
    wire         tw_l = (tw_i == p_npix - ((ag_tw && ({2'd0, tw_g} == ngroups - 8'd1)) ? 12'd0 : 12'd1));
    assign       t_h0 = ((tq_hd ? tp_i : tw_i) == 12'd0);
    assign       t_hf = ((tq_hd ? tp_i : tw_i) == p_npix);
    assign       ag_hw = ag_tw && p_dwt && dwc_v && (tq_hd ? ({2'd0, tp_g} == ngroups - 8'd1) : ({2'd0, tw_g} == ngroups - 8'd1));
    wire         t_pp = t_pop && tq_hd;
    wire         t_pd = t_pop && !tq_hd;
    always_ff @(posedge clk) begin
        if (start) begin
            tp_a <= p_out2 + (ag_tw ? {16'd0, ag_d} : 20'd0); tp_i <= '0; tp_g <= '0; tw_i <= '0; tw_g <= '0;
        end else begin
            if (t_pp) begin
                if (tp_l) begin tp_i <= '0; tp_g <= tp_g + 6'd1; tp_a <= p_out2 + {10'd0, tp_g + 6'd1, 4'd0} + (ag_tw ? {16'd0, ag_d} : 20'd0); end
                else begin tp_i <= tp_i + 12'd1; tp_a <= ha_tpc; end
            end
            if (t_pd) begin
                if (tw_l) begin tw_i <= '0; tw_g <= tw_g + 6'd1; end
                else tw_i <= tw_i + 12'd1;
            end
        end
    end
    wire         dw_shp = dw_ov && dw_oq[3];
    // depthwise parameters (gated): bias saturated to 24 bits (exact: |dot| < 2^19, the result saturates to 22 bits)
    logic [16*24-1:0] dbias;
    logic [23:0] dbs [4];                                  // the staged beat's words: bias saturated, -shift
    logic [3:0]  dsn [4];
    always_comb for (int u = 0; u < 4; u++) begin
        logic [31:0] x;
        logic [7:0]  n;
        x = wg_d[u*32 +: 32];
        n = 8'd0 - x[7:0];
        dbs[u] = (x[31:23] == {9{x[31]}}) ? x[23:0] : {x[31], {23{~x[31]}}};
        dsn[u] = n[3:0];
    end
    for (genvar k = 0; k < 16; k++) begin : g_dpm
        wire wr_k = pm_v && pm_d && (pm_j[3:2] == 2'(k / 4)) && (5'(k) < pm_n);
        wire gb = clk | ~(wr_k && pm_a == 2'd0);
        wire gm = clk | ~(wr_k && pm_a == 2'd1);
        wire gs = clk | ~(wr_k && pm_a == 2'd2);
        wire [31:0] wv = wg_d[(k % 4)*32 +: 32];
        logic [23:0] b;
        logic [30:0] m;
        logic [3:0]  r;
        always_ff @(posedge gb) b <= dbs[k % 4];
        always_ff @(posedge gm) m <= wv[30:0];
        always_ff @(posedge gs) r <= dsn[k % 4];
        assign dbias[24*k +: 24] = b; assign dmult[k] = m; assign dshr[k] = r;
    end
    // depthwise weight byte 128..143 (beat 8)
    logic [127:0] w8x;
    wire w8g = clk | ~(wg_v && wg_x);
    always_ff @(posedge w8g) w8x <= wg_d;
    // pass start: the walk restarts (the depthwise header is in)
    wire dwo_prst = p_dwf && (cst == C_SHIFT) && pgo && (c_j + 5'd4 >= c_gc) && c_dl;
    logic dw_room;
    always_ff @(posedge clk) dw_room <= ({1'b0, wd_space} >= ({1'b0, wres} + 9'd32));
    logic dw_shd, dwo_busy;
    logic [9*128-1:0] dw_tap;
    logic [8*128-1:0] dw_wb;
    assign dw_tap = {win[0][2], win[0][1], win[0][0], win[1][0], win[2][2], win[2][1], win[3][1], win[3][0], win[4][2]};
    for (genvar j = 0; j < NL; j++) begin : g_dwb
        assign dw_wb[128*j +: 128] = wreg[j][8];
    end
    if (DWF) begin : g_dwf
        npu_dwf u_dwf (.clk(clk), .rst_n(rst_n), .prst(dwo_prst), .sh_pw(dw_shp), .p_ho(p_ho), .p_wo(p_wo), .p_npix(p_npix),
                       .tap(dw_tap), .wb(dw_wb), .w8x(w8x), .dbias(dbias), .par(dpar), .room(dw_room),
                       .sh_dm(dw_shd), .ostart(dwo_st), .dr_v(dwd), .dr_k(dwd_k), .dr_acc(dwd_acc), .busy(dwo_busy));
    end else begin : g_ndwf
        assign dw_shd = 1'b0; assign dwo_st = 1'b0; assign dwd = 1'b0; assign dwd_k = 1'b0; assign dwd_acc = '0;
        assign dwo_busy = 1'b0;
    end
    assign dw_sh = dw_shp || dw_shd;
    assign dwu_busy = dwo_busy;
    // write commands: one per output, NHWC (p_out + 16 g + p_co * pixel)
    logic [4:0] dwc_n;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) dwc_n <= '0;
        else dwc_n <= dwc_n + ((p_dwf && (p_dwt ? t_tag : dwo_st)) ? 5'd1 : 5'd0) - ((p_dwf && dwc_v && wc_ready) ? 5'd1 : 5'd0);
    end
    assign dwc_v = (dwc_n != 5'd0);
    always_ff @(posedge clk) begin
        if (p_dwt) begin                                   // (v46DS) tee: depthwise stream counters (op start, wrap per group)
            if (start) dwc_a <= p_out[19:0] + (ag_tw ? {16'd0, ag_d} : 20'd0);
            else if (t_pd) dwc_a <= tw_l ? (p_out[19:0] + {10'd0, tw_g + 6'd1, 4'd0} + (ag_tw ? {16'd0, ag_d} : 20'd0)) : ha_dwc;
        end
        else if (dwo_prst) dwc_a <= p_out[19:0] + {12'd0, c_g, 4'd0};
        else if (p_dwf && dwc_v && wc_ready) dwc_a <= ha_dwc;
    end

    // weight registers: one clock-gated bank of 128 flops per (lane, entry). The write is staged
    // one cycle (registered enable + data), so each bank's gated clock gclk = clk | ~en_q only
    // pulses while its enable, settled during the high phase, is set; no per-bit enable muxes.
    // (v28A) FC stream: before the op starts (the cycle of pre, then the start cycle; the engine is
    // idle, the first C_WGT write lands two cycles later) entry 8 of lanes 5 / 6 takes the in_off
    // constant (fs_c1 / fs_c2 in every byte): the offset lanes then read it as a weight
    wire  fs0 = pre && p_fs;
    logic fs1;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin wg_v <= 1'b0; fs1 <= 1'b0; end
        else begin wg_v <= ((cst == C_WGT) && have0 && !pa_rw && !(c_dl && dwu_busy)) || fs0 || fs1; fs1 <= fs0; end
    end
    logic wg_x;                                            // (v37_1a) depthwise beat 8 (-> w8x) (v38M: wg_dw went with the correction)
    logic wg_h, wg_f, wg_dw;                               // (v52BF) correction: second half / a lane's first beat / DWF beat
    always_ff @(posedge clk) begin
        wg_x <= c_dl && c_j[3];
        wg_h <= p_dw ? c_e[0] : (p_pg && c_j[3]); wg_f <= p_dw || (c_e == 4'd0); wg_dw <= c_dl;   // (v52BF)
        wg_l <= fs0 ? 3'd5 : fs1 ? 3'd6 : c_j[2:0];
        wg_d <= (fs0 || fs1) ? {16{fs0 ? fs_c1 : fs_c2}} : (rs0_data & rmask);
        // column mode: tap t = 3 ky + kx goes to entry 3 kx + ky
        wg_e <= (fs0 || fs1 || c_dl) ? 4'd8 :
                p_pg ? {c_e[2:0], c_j[3]} : !scw ? c_e :
                rpr ? ((c_e == 4'd0) ? 4'd2 : (c_e == 4'd1) ? 4'd5 : (c_e == 4'd2) ? 4'd8 : (c_e == 4'd3) ? 4'd1 :     // (v35R) 3 kx + 2 - ky
                       (c_e == 4'd5) ? 4'd7 : (c_e == 4'd6) ? 4'd0 : (c_e == 4'd7) ? 4'd3 : (c_e == 4'd8) ? 4'd6 : c_e) :
                (c_e == 4'd1) ? 4'd3 : (c_e == 4'd2) ? 4'd6 : (c_e == 4'd3) ? 4'd1 :
                (c_e == 4'd5) ? 4'd7 : (c_e == 4'd6) ? 4'd2 : (c_e == 4'd7) ? 4'd5 : c_e;
        if (p_ptr && p_dw && cst == C_WGT)                         // transposed walk: tap (r, c) takes (c, r)
            for (int e = 0; e < 9; e++) wg_d[e*8 +: 8] <= rs0_data[((e % 3) * 3 + e / 3)*8 +: 8];
    end
    // (v39S) the lanes multiply raw int8 activations; the input offset's share in_off * sum(w) is folded into
    // the bias words by the kernel (no correction in hardware): ob is written by the bias words only
    for (genvar k = 0; k < 16; k++) begin : g_oc
        wire gclk = clk | ~bwr[k];
        logic signed [31:0] q;
        always_ff @(posedge gclk) q <= bw[k % 4];
        assign ob[k] = q;
    end
    // (v52BF) hardware input-offset fold (the n1b-era correction v39_S removed; only on ops with p_hf = RG_HASB bit 1):
    // the lanes multiply raw int8 activations, so every output adds in_off * (sum of its channel's weights in the
    // pass) - computed here from the weight beats the harness's tensors deliver, no table. The weight beats of a
    // (lane, half) arrive back to back during the load: stage 1 sums the staged beat, stage 2 accumulates
    // in_off * sum and writes oc[{h, lane}] (gated). |in_off * sum| <= 128 * 144 * 128 < 2^22. oc is final 3 cycles
    // after the last weight beat, when a pass's first start value is formed at the earliest; it is rewritten only
    // by the next load (after the last MAC left s1). DWF depthwise beats (npu_dwf: offset-added inputs) and the FC
    // stream (offset lanes) take no correction.
    logic        cw_v, cw_f;
    logic [3:0]  cw_k;
    logic signed [11:0] cw_s, wsum;
    logic signed [22:0] cacc, cnext;
    npu_hfs u_hfs (.d(wg_d), .y(wsum));                   // (fa_1 tree: sum of the 16 signed bytes)
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) cw_v <= 1'b0; else cw_v <= wg_v && p_hf && !p_fs && !wg_dw;
    end
    always_ff @(posedge clk) begin
        cw_s <= wsum; cw_k <= {wg_h, wg_l}; cw_f <= wg_f;
    end
    npu_hfm u_hfm (.a(cw_s), .b(p_inoff[8:0]), .c(cacc & {23{!cw_f}}), .y(cnext));   // (fa_1: cacc + cw_s * in_off)
    wire cacc_gclk = clk | ~cw_v;
    always_ff @(posedge cacc_gclk) cacc <= cnext;
    for (genvar k = 0; k < 16; k++) begin : g_hf
        wire gclk = clk | ~(cw_v && cw_k == 4'(k));
        logic signed [22:0] q;
        always_ff @(posedge gclk) q <= cnext;
        assign oc[k] = q;
    end
    for (genvar j = 0; j < NL; j++) begin : g_wb
        for (genvar e = 0; e < KE; e++) begin : g_we
            wire gclk = clk | ~(wg_v && wg_l == 3'(j) && wg_e == 4'(e) && !wg_x);
            logic [127:0] bank;
            always_ff @(posedge gclk) bank <= wg_d;
            assign wreg[j][e] = bank;
        end
    end

    // (v28R) the consumer loads the weights of the issuer's pass (pre-arranged, one read): its remaining
    // bytes may come as one long burst (C_WGT pops every beat as it arrives; registered: the issuer's
    // next weight read is queued only after it has left this pass)
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) wlong <= 1'b0;
        else wlong <= (cst == C_WGT) && (c_k == i_k) && (c_g == i_g) && sl_multi && prei;
    end
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) wlong_e <= 1'b0;
        else wlong_e <= (ist != I_IDLE) && (cst != C_IDLE) && !((c_k == i_k) && (c_g == i_g)) && sl_multi && prei;
    end
    // barrier: the psum read of pass k may start once pass k-1 is fully written back
    assign bar_ok = (c_k == i_k) && (c_g == i_g) && (cst == C_WGT || cst == C_PIX) && wr_idle && !res_v && !outs_left;
    // (v42DS) op chaining: the last pass of a DWF op with its MAC pipeline drained (a pass boundary's condition),
    // issuer idle; chn: a chained op runs while the old op's tail (requant drain, npu_dwf, writes) completes
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) chn <= 1'b0;
        else if (CHN && start && (cst == C_PASSEND)) chn <= 1'b1;
        else if (chn && pidle && !outs_left && !sc_v && !dwu_busy && !dwc_v && wr_idle) chn <= 1'b0;
    end
    assign bg_yield = (nchunks != 5'd1) && c_last;
    assign chain_ok = CHN && p_dwf && !p_dwt && (cst == C_PASSEND) && c_last && (c_g == ngroups - 16'd1) && (ist == I_IDLE) &&
                      !sc_pend && !sc_pd2 && !rr_pend && !s1_v && !dw_pend && !dw_h1p && !dw_bp && !dw_bp2 && !chn;
    assign busy = (ist != I_IDLE) || (cst != C_IDLE);
endmodule

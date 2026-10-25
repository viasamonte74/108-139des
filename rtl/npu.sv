// ChipForge NPU v1 - top level.
//
// The CPU kernel writes a layer descriptor into the register window (0x3000_0000), writes the
// op code to CMD, then reads STATUS: that read is held (RVALID low) until the NPU is idle, so a
// kernel costs one blocking load instead of a polling loop.
//
// Engines: npu_conv8 (conv / fc / depthwise / global average pool), npu_add (elementwise add),
// npu_pad (pad / copy). They share two ordered read streams, one write stream and two
// requantisers.
`include "npu_params.svh"

module npu (
    input  logic                       clk,
    input  logic                       rst_n,
    input  logic [`NPU_SID_W-1:0]      s_awid,
    input  logic [31:0]                s_awaddr,
    input  logic [7:0]                 s_awlen,
    input  logic [2:0]                 s_awsize,
    input  logic [1:0]                 s_awburst,
    input  logic                       s_awvalid,
    output logic                       s_awready,
    input  logic [`NPU_DATA_W-1:0]     s_wdata,
    input  logic [`NPU_DATA_W/8-1:0]   s_wstrb,
    input  logic                       s_wlast,
    input  logic                       s_wvalid,
    output logic                       s_wready,
    output logic [`NPU_SID_W-1:0]      s_bid,
    output logic [1:0]                 s_bresp,
    output logic                       s_bvalid,
    input  logic                       s_bready,
    input  logic [`NPU_SID_W-1:0]      s_arid,
    input  logic [31:0]                s_araddr,
    input  logic [7:0]                 s_arlen,
    input  logic [2:0]                 s_arsize,
    input  logic [1:0]                 s_arburst,
    input  logic                       s_arvalid,
    output logic                       s_arready,
    output logic [`NPU_SID_W-1:0]      s_rid,
    output logic [`NPU_DATA_W-1:0]     s_rdata,
    output logic [1:0]                 s_rresp,
    output logic                       s_rlast,
    output logic                       s_rvalid,
    input  logic                       s_rready,
    output logic [`NPU_MID_W-1:0]      m_awid,
    output logic [31:0]                m_awaddr,
    output logic [7:0]                 m_awlen,
    output logic [2:0]                 m_awsize,
    output logic [1:0]                 m_awburst,
    output logic                       m_awvalid,
    input  logic                       m_awready,
    output logic [`NPU_DATA_W-1:0]     m_wdata,
    output logic [`NPU_DATA_W/8-1:0]   m_wstrb,
    output logic                       m_wlast,
    output logic                       m_wvalid,
    input  logic                       m_wready,
    input  logic [`NPU_MID_W-1:0]      m_bid,
    input  logic [1:0]                 m_bresp,
    input  logic                       m_bvalid,
    output logic                       m_bready,
    output logic [`NPU_MID_W-1:0]      m_arid,
    output logic [31:0]                m_araddr,
    output logic [7:0]                 m_arlen,
    output logic [2:0]                 m_arsize,
    output logic [1:0]                 m_arburst,
    output logic                       m_arvalid,
    input  logic                       m_arready,
    input  logic [`NPU_MID_W-1:0]      m_rid,
    input  logic [`NPU_DATA_W-1:0]     m_rdata,
    input  logic [1:0]                 m_rresp,
    input  logic                       m_rlast,
    input  logic                       m_rvalid,
    output logic                       m_rready
);
    // ------------------------------------------------------------------- register file
    // Two banks. The CPU writes the staging bank (regs) at any time. CMD queues an op; once the
    // engines are idle the staging bank is copied into the active bank (act), which the engines
    // read, and the op starts. So the kernel of op i+1 writes its registers while op i runs.
    // STATUS (word 4) is held while an op is queued or running; SLOT (a read of word 0) only while
    // a command is queued (its registers not yet copied). Registers are not readable.
    // Both banks are written through gated clocks (staged CPU writes; one copy strobe).
    localparam int NREG = 68;              // (v49_RS: 64, 65, 67 write tee; 66 = its arm, below)
    logic [31:0] regs [NREG];
    logic [31:0] act  [NREG];
    // op codes
    localparam logic [3:0] OP_CONV = 4'd1, OP_DW = 4'd2, OP_AVG = 4'd3, OP_MAX = 4'd4,
                           OP_ADD  = 4'd5, OP_PAD = 4'd6, OP_PADI = 4'd7,   // (v49DS) PADI: split PAD
                           OP_PIL  = 4'd8,                                  // (v50_MB) planes -> NHWC interleave
                           OP_KCP  = 4'd9;                                  // (v55_KC) constant-channel probe
    logic        busy;                 // engines / DMA / start pipeline
    logic        cmd_pend, copy_q, start;
    logic        take;                 // (v42DS) the queued op is copied in (idle, or chained onto a DWF op)
    logic [3:0]  op_n, op_q;
    logic        op_cv_q, op_dw_q, op_add_q, op_pad_q;   // (v40K opdec) registered decodes of op_q
    logic        op_pi_q;                                  // (v49DS) the PAD is interior-only (OP_PADI)
    logic        op_il_q;                                  // (v50_MB) OP_PIL
    logic        op_kc_q;                                  // (v55_KC) OP_KCP
    logic        sp_on;                                    // (v49DS) border runs pending (see the filler)
    // (v34P) chunk sequence / input gate (see the end of the file)
    logic        seq_on, seq_adv, seq_raw, go, busy_core;
    logic [2:0]  seq_c;
    logic [19:0] seq_in, seq_wgt;
    logic [31:0] r_scr, r_outc;
    logic [19:0] r_actlen;

    // ---------------- AXI slave: writes ----------------
    logic        aw_act;
    logic [31:0] aw_a;
    logic [3:0]  aw_id;
    assign s_awready = !aw_act && !s_bvalid;
    assign s_wready  = aw_act;
    assign s_bresp   = 2'b00;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            aw_act <= 1'b0; aw_a <= '0; aw_id <= '0; s_bvalid <= 1'b0; s_bid <= '0;
            cmd_pend <= 1'b0; op_n <= '0; copy_q <= 1'b0; start <= 1'b0; op_q <= '0;
            op_cv_q <= 1'b0; op_dw_q <= 1'b0; op_add_q <= 1'b0; op_pad_q <= 1'b0; op_pi_q <= 1'b0; op_il_q <= 1'b0; op_kc_q <= 1'b0;
        end else begin
            // queued command -> copy the registers -> start
            copy_q <= 1'b0;
            start  <= copy_q || seq_adv;
            if (take) begin copy_q <= 1'b1; cmd_pend <= 1'b0; op_q <= op_n;
                op_cv_q <= (op_n == OP_CONV); op_dw_q <= (op_n == OP_DW); op_add_q <= (op_n == OP_ADD);
                op_pad_q <= (op_n == OP_PAD) || (op_n == OP_PADI) || (op_n == OP_PIL); op_pi_q <= 1'b0;   /* (v58_AR) OP_PADI tied off */
                op_il_q <= 1'b0;   /* (v57_MB) OP_PIL tied off */ op_kc_q <= (op_n == OP_KCP); end
            if (s_bvalid && s_bready) s_bvalid <= 1'b0;
            if (s_awvalid && s_awready) begin aw_act <= 1'b1; aw_a <= s_awaddr; aw_id <= s_awid; end
            if (s_wvalid && s_wready) begin
                // register 0 (CMD) queues an op
                if (aw_a[8:4] == 5'd0 && s_wstrb[3:0] != 4'd0) begin
                    cmd_pend <= 1'b1; op_n <= s_wdata[3:0];
                end
                aw_a <= aw_a + 32'd16;
                if (s_wlast) begin aw_act <= 1'b0; s_bvalid <= 1'b1; s_bid <= aw_id; end
            end
        end
    end
    // staged register writes: the CPU stores one 32-bit word per beat (RV32), so only the strobed
    // word and its lane are staged (v15_A2 / v18C s7)
    logic         wst_v;
    logic [4:0]   wst_row;
    logic [1:0]   wst_lane;
    logic [31:0]  wst_d;
    wire  [1:0]   w_lane = (s_wstrb[3:0] != 4'd0) ? 2'd0 : (s_wstrb[7:4] != 4'd0) ? 2'd1 :
                           (s_wstrb[11:8] != 4'd0) ? 2'd2 : 2'd3;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) wst_v <= 1'b0; else wst_v <= s_wvalid && s_wready && (s_wstrb != 16'd0);
    end
    always_ff @(posedge clk) begin
        wst_row <= aw_a[8:4]; wst_lane <= w_lane; wst_d <= s_wdata[w_lane*32 +: 32];
    end
    wire act_gclk = clk | ~copy_q;
    for (genvar k = 0; k < NREG; k++) begin : g_reg
        wire gclk = clk | ~(wst_v && wst_row == 5'(k / 4) && wst_lane == 2'(k % 4));
        logic [31:0] r, a;
        always_ff @(posedge gclk) r <= wst_d;
        always_ff @(posedge act_gclk) a <= r;
        assign regs[k] = r;
        assign act[k]  = a;
    end

    // (v51_D) sticky DWF parameter flag: set by npu_conv8, cleared by a register-66 write with bit 1, read as bit 0
    logic cv_dwfbad, dwf_flag;
    logic kc_ovf;                     // (v55_KC)
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) dwf_flag <= 1'b0;
        else if (cv_dwfbad || kc_ovf) dwf_flag <= 1'b1;   // (v55_KC) or a probe's recorded-channel overflow
        else if (wst_v && wst_row == 5'd16 && wst_lane == 2'd2 && wst_d[1]) dwf_flag <= 1'b0;
    end
    // (v49_RS) write-tee arm (register 66, bit 0): one-shot, taken by the next op that starts
    logic t_arm_s, t_arm_a;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin t_arm_s <= 1'b0; t_arm_a <= 1'b0; end
        else begin
            if (copy_q) begin t_arm_a <= t_arm_s; t_arm_s <= 1'b0; end
            if (wst_v && wst_row == 5'd16 && wst_lane == 2'd2) t_arm_s <= wst_d[0];
        end
    end

    // ---------------- AXI slave: reads (STATUS / SLOT beats are held) ----------------
    logic        ar_act;
    logic [31:0] ar_a;
    logic [7:0]  ar_left;
    logic [3:0]  ar_id;
    assign s_arready = !ar_act;
    assign s_rresp   = 2'b00;
    assign s_rid     = ar_id;
    logic        pk_act, pk_hold, pk_ovf, pk_bgv, pk_brv;     // (v55AD) pack tee (npu_pkw, below)
    logic [19:0] pk_len, pk_bga;
    logic [7:0]  pk_bgl;
    logic [127:0] pk_brd, pk_fpd;
    logic        pk_fpush, pk_frdy;
    logic [4:0]  pk_fpn, pk_fpop;
    logic        pk_uni;
    // (v51_D) bit 0: DWF parameter flag (else 0 = ok); (v55AD) the last pack op: bit 2 overflowed, bit 3 uniform
    // multipliers / shifts (its per-tensor stream is valid), 31:12 its stream length
    logic        st_bg;                // (v56_CY) bit 1: background work (filler / copy / queued burst) at the read
    assign s_rdata   = {{(`NPU_DATA_W-32){1'b0}}, pk_len, 8'd0, pk_uni, pk_ovf, st_bg, dwf_flag};
    // the CPU's bus aligns read addresses to the 16-byte beat: SLOT is row 0, STATUS row 1
    wire   blocked   = (ar_a[7:4] == 4'd0) ? cmd_pend : (ar_a[7:4] == 4'd1) && (cmd_pend || busy || sp_on || bgc_on);
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ar_act <= 1'b0; ar_a <= '0; ar_left <= '0; ar_id <= '0;
            s_rvalid <= 1'b0; s_rlast <= 1'b0;
        end else begin
            if (s_rvalid && s_rready) begin
                s_rvalid <= 1'b0; s_rlast <= 1'b0;
                if (s_rlast) ar_act <= 1'b0;
                else begin ar_a <= ar_a + 32'd16; ar_left <= ar_left - 8'd1; end
            end else if (s_arvalid && s_arready) begin
                ar_act <= 1'b1; ar_a <= s_araddr; ar_left <= s_arlen; ar_id <= s_arid;
            end else if (ar_act && !s_rvalid && !blocked) begin
                s_rvalid <= 1'b1;
                s_rlast  <= (ar_left == 8'd0);
            end
        end
    end

    // ------------------------------------------------------------------- parameters
    wire [31:0] r_in      = {12'd0, act[8][19:0]};    // (v52AD) no chunk sequence: the active bank's IN / WGT
    wire [31:0] r_wgt     = {12'd0, act[9][19:0]};
    wire [31:0] r_bias    = {12'd0, act[10][19:0]};
    wire [31:0] r_mult    = {12'd0, act[11][19:0]};   // conv: parameter address (the average pool's
                                                     // divisor multiplier is in RG_MO)
    wire [31:0] r_shift   = {12'd0, act[12][19:0]};
    wire [31:0] r_out     = {12'd0, act[13][19:0]};
    wire        r_hasb    = act[14][0];
    // (v15 widths: the kernel keeps Ho, Wo, FH, FW, H, W <= 255 (pool: 256 reads as 0 = 256),
    // Co, Ci <= 1023, K, L < 8192, Ho*Wo < 4096, strides and plane sizes < 65536, padding <= 3)
    wire [7:0]  r_ho      = act[15][7:0];
    wire [7:0]  r_wo      = act[16][7:0];
    wire [9:0]  r_co      = act[17][9:0];
    wire [7:0]  r_fh      = act[18][7:0];
    wire [7:0]  r_fw      = act[19][7:0];
    wire [9:0]  r_ci      = act[20][9:0];
    wire [12:0] r_l       = act[21][12:0];
    wire [15:0] r_rowstr  = act[22][15:0];
    wire [15:0] r_pixstep = act[23][15:0];
    wire [15:0] r_rowstep = act[24][15:0];
    wire [12:0] r_k       = act[25][12:0];
    wire [31:0] r_wgstr   = act[26];
    wire [16:0] r_wglast  = act[27][16:0];
    wire [31:0] r_dwadj   = act[28];
    wire [31:0] r_inoff   = act[29];
    wire [8:0]  r_ooff    = act[30][8:0];
    wire [7:0]  r_amin    = act[31][7:0];
    wire [7:0]  r_amax    = act[32][7:0];
    wire [23:0] r_count   = act[33][23:0];
    wire [19:0] r_n       = act[34][19:0];
    wire [31:0] r_er      = act[35];
    wire [31:0] r_m1      = act[36];
    wire [31:0] r_m2      = act[37];
    wire [31:0] r_off1    = act[38];
    wire [31:0] r_off2    = act[39];
    wire [31:0] r_mo      = act[40];
    wire [7:0]  r_sho     = act[41][7:0];
    wire [7:0]  r_prows   = act[42][7:0];        // (pad: rows <= 255, row segments < 65536 bytes)
    wire [7:0]  r_phb     = act[43][7:0];
    wire [7:0]  r_ph      = act[44][7:0];
    wire [19:0] r_pwbc    = act[45][19:0];
    wire [15:0] r_pwc     = act[46][15:0];
    wire [15:0] r_pwac    = act[47][15:0];
    wire [15:0] r_pwoc    = act[48][15:0];
    wire [19:0] r_ptot    = act[49][19:0];
    wire [7:0]  r_pval    = act[50][7:0];
    wire [31:0] r_scr0    = {12'd0, act[51][19:0]};
    wire [11:0] r_npix    = act[52][11:0];
    wire        r_contig  = act[53][0];
    wire [4:0]  r_nchunks = act[54][4:0];
    wire [19:0] r_actlen0 = act[55][19:0];
    wire [19:0] r_outlen  = act[56][19:0];
    wire [1:0]  r_sw      = act[57][1:0];
    wire [15:0] r_fx0off  = act[58][15:0];
    wire        r_sliced  = act[59][0];
    wire        r_pg      = act[60][0];
    wire [1:0]  r_pt      = act[61][1:0];        // pad top / left
    wire [1:0]  r_pl      = act[61][9:8];
    wire [7:0]  r_hin     = act[62][7:0];        // H, W of the real (unpadded) input
    wire [7:0]  r_win     = act[62][23:16];
    wire        r_cp      = act[63][0];          // conv8 gather copy
    wire        r_fz      = act[63][1];          // conv8 with a fused ADD
    wire        r_fs      = act[63][4];          // conv8 FC stream (one output pixel, K <= 720)
    // global average pool: a depthwise pass of conv8 (RG_CP bit 7); bit 8: the input is 16-channel
    // planes (each walked as an H x W x 16 tensor, RG_IGRP apart). No bias, no input offset; unit A
    // divides with the multiplier in RG_MO (shift 0): the multiplier / shift words read are unused.
    wire        pa_on     = (op_dw_q) && act[63][7];
    wire        r_pplane  = act[63][8];
    wire        r_pk      = act[63][2];          // (v55AD) packed FC stream (v52AD had tied it off; act[63][2]: RG_WGT / RG_ACTLEN, then
                                                 // RG_WGSTR / RG_DWADJ: a second piece, length 0 = none)
    wire [8:0]  r_fooff   = act[42][8:0];        // fused ADD: output offset / clamp
    wire [7:0]  r_famin   = act[43][7:0];
    wire [7:0]  r_famax   = act[44][7:0];
    wire [31:0] r_rdelta  = {12'd0, act[45][19:0]};   // fused ADD: residual - output address (mod 2^20)

    wire sel_cv  = (op_cv_q) || (op_dw_q);
    wire sel_add = (op_add_q);
    wire sel_pad = (op_pad_q);
    wire st_cv   = start && sel_cv;
    wire st_add  = start && sel_add;
    wire st_pad  = start && sel_pad;
    // engines see the start one cycle after the CMD write lands, with op_q already valid
    logic st_cv_q, st_add_q, st_pad_q;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin st_cv_q <= 1'b0; st_add_q <= 1'b0; st_pad_q <= 1'b0; end
        else begin st_cv_q <= st_cv; st_add_q <= st_add; st_pad_q <= st_pad; end
    end

    // ------------------------------------------------------------------- shared resources
    // read streams
    logic [1:0]   rc_valid, rc_ready, rc_fill;
    logic [63:0]  rc_addr;
    logic [39:0]  rc_len;
    logic [15:0]  rc_val;
    logic [15:0]  rs_avail;
    logic [255:0] rs_data;
    logic [127:0] rs0_data2;                     // stream 0: the 16 bytes after its window (packed FC)
    logic [15:0]  rs_pop, rs_peek;
    logic         rd_idle;
    logic cv_long, cv_wlong, cv_wlong_e, cv_b10;
    // (v50_RS BGP) background PAD copy (see the engine below the border filler)
    wire         bgx = 1'b0;                   // (v58_AR) tied: no background PAD copy (RS50 / MB_BG off)
    logic        bgc_on, bgd_v, bgd_tk, bgr_ok, bgr_v, bgr_rdy, cv_bgy;
    logic [19:0] bgc_p, bgc_q;                 // destination / source of the next piece
    logic [11:0] bgc_r, bgc_wc, bgc_ac;        // bytes left in the row; row bytes; destination gap between rows
    logic [7:0]  bgc_k;                        // rows left (with the current one)
    wire  [7:0]  bgc_lp = 8'd128 - {1'b0, bgc_p[6:0]};
    wire  [7:0]  bgc_lq = 8'd128 - {1'b0, bgc_q[6:0]};
    wire  [7:0]  bgc_lb = (bgc_lp < bgc_lq) ? bgc_lp : bgc_lq;
    wire  [7:0]  bgc_l  = (bgc_r < {4'd0, bgc_lb}) ? bgc_r[7:0] : bgc_lb;   // piece bytes (1..128)
    wire  [8:0]  bgc_e  = {5'd0, bgc_p[3:0]} + {1'b0, bgc_l} + 9'd15;
    wire  [3:0]  bgc_nb = bgc_e[7:4];                                        // its beats (1..8)
    logic        bgc_cap;                      // a copy piece is due (not in a final chunk pass): short data bursts
    npu_rdma #(.NSTR(2), .CAPB0(256), .CAPB(128), .MAXB0(8), .MAXB(6), .MAXL0(256), .CQD(4)) u_rd (
        .bg_go(bgd_tk && !pk_act), .bg_addr({bgc_q[19:4], 4'd0}), .bg_beats(bgc_nb), .bg_ok(bgr_ok), .bg_beat(bgr_v), .bg_rdy(bgr_rdy && !pk_act),
        .clk(clk), .rst_n(rst_n), .clr(1'b0), .clr1(start), .gate_on(seq_on && !go), .gate_seq(seq_on), .g_lo(act[8][19:0]), .g_len(act[63][29:20]), .long0((sel_cv && (cv_long || cv_wlong)) || (sel_pad && !op_il_q)),   /* (v50_MB r_lp) */ .long0e(sel_cv && cv_wlong_e), .long0f(sel_cv && seq_on), .sm1(sel_pad && op_il_q), .b10_0(sel_cv && cv_b10),
        .cmd_valid(rc_valid), .cmd_ready(rc_ready), .cmd_fill(rc_fill),
        .cmd_addr(rc_addr), .cmd_len(rc_len), .cmd_val(rc_val),
        .out_avail(rs_avail), .out_data(rs_data), .out_data2(rs0_data2), .peek(rs_peek), .idle(rd_idle),
        .pop_n(rs_pop | {3'd0, pk_fpop, 8'd0}), .x_on(pk_act), .x_valid(pk_fpush), .x_data(pk_fpd), .x_n(pk_fpn), .x_rdy(pk_frdy),   // (v55AD)
        .araddr(m_araddr), .arlen(m_arlen), .arvalid(m_arvalid), .arready(m_arready),
        .rdata(m_rdata), .rlast(m_rlast), .rvalid(m_rvalid), .rready(m_rready));
    assign m_arid = '0; assign m_arsize = 3'd4; assign m_arburst = 2'b01;

    // write stream
    logic         wc_valid, wc_ready, wd_valid, wd_ready;
    logic         wc_dup, cv_wcdup;                 // (v46RS) fz2 output write: also at + RG_N
    logic         wc_keep, wc_fill;    // (v46MB)
    logic [31:0]  wc_addr;
    logic [19:0]  wc_len;
    logic [127:0] wd_data;
    logic [4:0]   wd_n;
    logic         wr_idle;
    logic         wr_bginq;            // (v56_CY) background burst queued in the write DMA
    logic         sp_bgv, sp_tk;       // (v49DS) border filler -> write DMA background port
    logic [19:0]  sp_a;
    logic [7:0]   sp_n, sp_val;
    logic [7:0]   wd_space;
    // RQ8: requant units E..H (eight results per cycle) and a 128-byte write FIFO; 0: four units, 64 B
    localparam bit RQ8 = 1'b1;
    // (v49_RS) write tee: the conv engine's output writes (op armed by RG_TARM)
    logic         wc_tee;
    logic         wc_pm, wc_sk;                  // (v49_RS RS3)
    wire          t_on = sel_cv && t_arm_a;
    npu_wdma #(.CAPB(RQ8 ? 128 : 64), .MAXB(64), .CQD(10)) u_wr (   // (v58_AR) 192 -> 128 B: no cycle change after the v57 fusion
        .clk(clk), .rst_n(rst_n), .clr(1'b0),
        .cmd_valid(wc_valid), .cmd_ready(wc_ready), .cmd_addr(wc_addr), .cmd_len(wc_len), .cmd_keep(wc_keep),
        .cmd_fill(1'b0), .fill_val(8'd0),   // (v58_AR fv) no fill bursts: the conv8 mirror border fills are tied off
        .cmd_tee(wc_tee), .t_on(t_on), .t_l64(act[65][17]), .t_b3(act[64][19:0]),
        .t_out(act[67][19:0]), .t_plb(act[65][15:0]), .t_rp(act[65][20]), .t_w(act[65][22:21]), .t_r8(act[65][23]),   // (v49_RS RS3)
        .cmd_pm(1'b0), .cmd_sk(1'b0), .bg_cap(bgc_cap),
        .rp_on(1'b0), .rp_nb(act[60][12]), .rp_al(act[60][13]),   // (v50_MB)
        .rp_da(act[45][19:0]), .rp_db(act[49][19:0]), .rp_tp(act[38][19:0]), .rp_rw(act[46][11:0]),   // (register bits
        .rp_top(act[33][11:0]), .rp_h(act[33][19:12]), .rp_sk(act[34][7:0]), .rp_bot(act[34][19:8]),   //  other ops already
        .rp_ch(8'd96),                                                                                  //  use: no new flops)
        .cmd_dup(1'b0), .dup_delta(20'd0),
        .in_valid(wd_valid), .in_ready(wd_ready), .in_data(wd_data), .in_n(wd_n), .in_space(wd_space),
        .idle(wr_idle), .bg_inq(wr_bginq), .bg_valid(sp_bgv), .bg_addr(sp_a), .bg_len(sp_n), .bg_val(sp_val), .bg_take(sp_tk),
        .bgd_valid(pk_act ? pk_bgv : bgd_v), .bgd_addr(pk_act ? pk_bga : bgc_p), .bgd_len(pk_act ? pk_bgl : bgc_l), .bgd_take(bgd_tk),
        .bgr_v(pk_act ? pk_brv : bgr_v), .bgr_data(pk_act ? pk_brd : m_rdata), .bgr_rdy(bgr_rdy),   // (v55AD) pack tee
        .awaddr(m_awaddr), .awlen(m_awlen), .awvalid(m_awvalid), .awready(m_awready),
        .wdata(m_wdata), .wstrb(m_wstrb), .wlast(m_wlast), .wvalid(m_wvalid), .wready(m_wready),
        .bvalid(m_bvalid), .bready(m_bready));
    assign m_awid = '0; assign m_awsize = 3'd4; assign m_awburst = 2'b01;

    // requant units (A shared by every engine, B used by conv8)
    logic        rq_v, rq_ov, rqb_v, rqb_ov;
    logic [31:0] rq_acc, rq_mult, rqb_acc, rqb_mult;
    logic [7:0]  rq_shift, rq_q, rqb_shift, rqb_q;
    // An ADD fused into a conv (two elements per cycle): C and D requantise the conv results i, i + 1
    // (npu_conv8 routes them; the driver checks that every channel is one npu_drq takes), E-H scale the
    // ADD's inputs (MultiplyByQuantizedMultiplier((x + off) << 20, m, shift), raw; E, G the conv
    // results (scale 2^30, shift 0: a shift), F, H the residual bytes), A and B requantise the two sums.
    logic        cv_rqcv, cv_rqdv, rqc_ov, rqd_ov;
    logic [31:0] cv_rqcacc, cv_rqcmult, cv_rqdacc, cv_rqdmult;
    logic [7:0]  cv_rqcsh, cv_rqdsh, rqc_q, rqd_q;
    logic        cv_pe_on;                       // (v39S) conv8 sends fused-pool sums to A / B: RG_MO, shift 0
    wire         fz_on = sel_cv && r_fz;
    // (v46RS) fz2 (RG_CP bit 28 with bit 1): the conv's own results (C / D) are written too, after the ADD's
    // results of the same group of 8: C / D's pair delayed by npu_rq2's 4 cycles is the pair A / B emit
    wire         r_fz2 = 1'b0;   // (v57_RS r_rsfz) fz2 tied off
    logic [15:0] f2_d [4];
    logic [15:0] f2_c [3];
    logic [1:0]  f2_n;
    always_ff @(posedge clk) begin
        f2_d[0] <= {rqd_q, rqc_q}; f2_d[1] <= f2_d[0]; f2_d[2] <= f2_d[1]; f2_d[3] <= f2_d[2];
    end
    wire         f2_push = r_fz2 && rq_ov && sel_cv;
    for (genvar q = 0; q < 3; q++) begin : g_f2
        wire g = clk | ~(f2_push && f2_n == 2'(q));
        logic [15:0] e;
        always_ff @(posedge g) e <= f2_d[3];
        assign f2_c[q] = e;
    end
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) f2_n <= 2'd0;
        else if (st_cv) f2_n <= 2'd0;
        else if (f2_push) f2_n <= f2_n + 2'd1;
    end
    logic [7:0]  fz_b, fz_b2;
    logic        sce_v, scf_v, scg_v, sch_v;
    logic [31:0] sce_r, scf_r, scg_r, sch_r;
    // the conv results' scale is 2^30 with shift 0 (the driver checks): raw = (x + off) << 19 exactly,
    // formed as C / D's results leave; the residual scalers start 2 cycles after the drain (their
    // 2-cycle result lines up with C / D's 4-cycle one: v26A, no delay registers)
    logic [1:0] fzc_q, fzd_q;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin fzc_q <= 2'b00; fzd_q <= 2'b00; end
        else begin fzc_q <= {fzc_q[0], cv_rqcv && sel_cv}; fzd_q <= {fzd_q[0], cv_rqdv && sel_cv}; end
    end
    wire  [8:0] sce_e = 9'($signed(rqc_q)) + r_off1[8:0];
    wire  [8:0] scg_e = 9'($signed(rqd_q)) + r_off1[8:0];
    assign sce_v = fz_on && rqc_ov; assign sce_r = {{4{sce_e[8]}}, sce_e, 19'd0};
    assign scg_v = fz_on && rqd_ov; assign scg_r = {{4{scg_e[8]}}, scg_e, 19'd0};
    npu_asc u_scf (.clk(clk), .rst_n(rst_n), .in_valid(fz_on && fzc_q[1]), .v(9'($signed(fz_b)) + r_off2[8:0]),
                   .mult(r_m2[30:0]), .shift(r_er[15:8]), .raw_valid(scf_v), .raw(scf_r));
    npu_asc u_sch (.clk(clk), .rst_n(rst_n), .in_valid(fz_on && fzd_q[1]), .v(9'($signed(fz_b2)) + r_off2[8:0]),
                   .mult(r_m2[30:0]), .shift(r_er[15:8]), .raw_valid(sch_v), .raw(sch_r));
    // A, B: one npu_rq2 (two npu_rq units on a double-pumped multiplier)
    npu_rq2 u_rqab (.clk(clk), .rst_n(rst_n), .in_valid0(fz_on ? sce_v : rq_v), .in_valid1(fz_on ? scg_v : rqb_v),
                 .acc0(fz_on ? (sce_r + scf_r) : rq_acc), .acc1(fz_on ? (scg_r + sch_r) : rqb_acc),
                 .mult0(fz_on ? r_mo : rq_mult), .mult1(fz_on ? r_mo : rqb_mult),
                 .shift0(fz_on ? r_sho : rq_shift), .shift1(fz_on ? r_sho : rqb_shift),
                 .ooff0(fz_on ? r_fooff : r_ooff), .amin0(fz_on ? r_famin : r_amin), .amax0(fz_on ? r_famax : r_amax),
                 .ooff1(fz_on ? r_fooff : r_ooff), .amin1(fz_on ? r_famin : r_amin), .amax1(fz_on ? r_famax : r_amax),
                 .out_valid0(rq_ov), .out_valid1(rqb_ov), .out_q0(rq_q), .out_q1(rqb_q));
    // C, D: one npu_drq2 (two npu_drq channels on a double-pumped multiplier; conv: the channels
    // npu_conv8 routes to them)
    npu_drq2 u_rqcd (.clk(clk), .rst_n(rst_n), .in_valid0(cv_rqcv && sel_cv), .in_valid1(cv_rqdv && sel_cv),
                 .acc0(cv_rqcacc), .acc1(cv_rqdacc), .mult0(cv_rqcmult[30:0]), .mult1(cv_rqdmult[30:0]),
                 .shift0(cv_rqcsh), .shift1(cv_rqdsh), .ooff(r_ooff), .amin(r_amin), .amax(r_amax),
                 .out_valid0(rqc_ov), .out_valid1(rqd_ov), .out_q0(rqc_q), .out_q1(rqd_q));

    // E..H: npu_drq, conv only (fast octets: eight results per cycle with A..D)
    logic         cv_rqxv;
    logic [127:0] cv_rqxacc;
    logic [123:0] cv_rqxmult;
    logic [31:0]  cv_rqxsh, rqx_q;
    logic [3:0]   rqx_ov;
    if (RQ8) begin : g_rq8
        // E / F and G / H: npu_drq2 pairs
        logic [1:0] xov;
        for (genvar u = 0; u < 2; u++) begin : g_rqx
            npu_drq2 u_rqx (.clk(clk), .rst_n(rst_n), .in_valid0(cv_rqxv && sel_cv), .in_valid1(cv_rqxv && sel_cv),
                         .acc0(cv_rqxacc[(2*u)*32 +: 32]), .acc1(cv_rqxacc[(2*u+1)*32 +: 32]),
                         .mult0(cv_rqxmult[(2*u)*31 +: 31]), .mult1(cv_rqxmult[(2*u+1)*31 +: 31]),
                         .shift0(cv_rqxsh[(2*u)*8 +: 8]), .shift1(cv_rqxsh[(2*u+1)*8 +: 8]),
                         .ooff(r_ooff), .amin(r_amin), .amax(r_amax),
                         .out_valid0(xov[u]), .out_valid1(),
                         .out_q0(rqx_q[(2*u)*8 +: 8]), .out_q1(rqx_q[(2*u+1)*8 +: 8]));
        end
        assign rqx_ov = {2'b00, xov};
    end else begin : g_rq4
        assign rqx_ov = 4'd0; assign rqx_q = 32'd0;
    end

    // --- conv8 ---
    logic         cv_rc0v, cv_rc1v, cv_wcv, cv_wdv, cv_rqav, cv_rqbv, cv_busy;
    logic         cv_wct;                        // (v49_RS) write tee
    logic         cv_wpm, cv_wsk;                // (v49_RS RS3) pair-merged fz2 writes
    logic         cv_wck, cv_wcf;      // (v46MB) mirror writes: keep the command's bytes / border fill commands
    // (v46MB) mirror registers reuse fields a plain single-chunk conv does not read: M1 start offset = RG_RDELTA,
    // M2 planes base = RG_SCR, border runs: top = RG_COUNT[15:0], last = RG_PTOT[15:0], value = RG_PVAL;
    // RG_PG bits 4 / 5 / 6 = M1 / M2 / fills, bits 8:7 = the row gap in pixels (row gap bytes = that x Co)
    wire  [10:0]  mrow = (act[60][7] ? {1'b0, r_co} : 11'd0) + (act[60][8] ? {r_co, 1'b0} : 11'd0);
    logic         cv_chain;            // (v42DS)
    logic [31:0]  cv_rc0a, cv_rc1a, cv_wca, cv_rqaacc, cv_rqamult, cv_rqbacc, cv_rqbmult;
    logic [19:0]  cv_rc0l, cv_rc1l, cv_wcl;
    logic [7:0]   cv_pop0, cv_peek0;
    logic [4:0]   cv_pop1, cv_wdn;
    logic [127:0] cv_wdd;
    logic [7:0]   cv_rqash, cv_rqbsh;
    npu_conv8 #(.RQ8(RQ8)) u_cv (   // (v57_DS r_chn) CHN = 1 (default): DWF op chaining
   
        .clk(clk), .rst_n(rst_n), .start(st_cv_q), .pre(st_cv),
        .p_in(r_in), .p_wgt(r_wgt), .p_bias(r_bias), .p_mult(r_mult), .p_shift(r_shift), .p_out(r_outc),
        // (v37_1a) fused depthwise (RG_CP bit 30): weights / biases / multipliers / shifts in RG_RDELTA, RG_PTOT, RG_COUNT,
        // RG_SCR (registers a paired-group conv does not read)
        .p_dwf_r(act[63][31] && (op_cv_q)), .p_rot_r(act[60][10] && (op_cv_q)), .p_dwt_r(1'b0 /* (v57_DS r_notee) */), .p_out2(act[38][19:0]), .p_dwgt(act[45][19:0]), .p_dbias(act[49][19:0]), .p_dmult(act[33][19:0]),
        .p_dshift(act[51][19:0]),
        .p_scr(r_scr), .p_has_bias(r_hasb && !pa_on), .p_hf(act[14][1] && !pa_on), .p_gl(act[14][2] ? ((kc_lrec == 7'd0) ? 4'd1 : 4'((kc_lrec + 7'd7) >> 3)) : 4'd0), .p_ho(r_ho), .p_wo(r_wo), .p_co(r_co), .p_fh(r_fh),
        .p_l(r_l), .p_rowstride(r_rowstr), .p_pixstep(r_pixstep), .p_rowstep(r_rowstep), .p_k(r_k),
        .p_npix(r_npix), .p_contig_r(r_contig), .p_nchunks_r(r_nchunks), .p_wglast(r_wglast),
        .p_actlen(r_actlen), .p_outlen(r_outlen), .p_inoff(pa_on ? 9'd0 : r_inoff[8:0]),
        .p_sliced_r(r_sliced), .p_dw(op_dw_q), .p_pg_r(r_pg && (op_cv_q)),
        .p_fs(r_fs && (op_cv_q)), .p_fsa(seq_on), .p_raw(seq_raw), .p_pk(r_pk), .p_wg2(act[26][19:0]), .p_len2(act[28][19:0]), .rs0_data2(rs0_data2), .p_fu(act[63][5]), .p_pre(1'b0), .p_w8(act[63][6] && (op_cv_q)), .p_dr_r(act[57][19] && (op_cv_q)), .s1_unused(cv_long), .wlong(cv_wlong), .wlong_e(cv_wlong_e), .p_cp(r_cp && (op_cv_q)),
        .p_fz(r_fz && (op_cv_q)), .p_fz2(r_fz2 && (op_cv_q)), .p_fzp(r_fz && act[63][27] && (op_cv_q)), .wc_dup(cv_wcdup),
        .p_v2(act[57][16] && (op_dw_q)), .p_v3(act[57][21] && (op_dw_q)), .p_dsp(act[57][17] && (op_dw_q)), .p_orow(act[6][15:0]),
        .p_pout(act[60][1] && (op_cv_q) && !r_fs), .p_po8(act[60][3] && (op_cv_q) && !r_fs), .p_pin(act[57][20] && (op_cv_q)), .p_rp(1'b0), .p_swz(1'b0), .p_p24(1'b0), .p_rr(1'b0), .p_rpw(1'b0), .p_rpr(act[63][14] && (op_cv_q)), .p_rpx(act[63][15] && (op_cv_q)), .b10(cv_b10), .p_ptr(act[57][18] && sel_cv),
        .p_colstep(act[1][15:0]), .p_igrp((act[57][18] || act[57][19] || act[57][20] || (pa_on && r_pplane)) ? act[2][15:0] : 16'd16), .p_opix(act[3][15:0]),
        .p_pa(pa_on), .p_pf(act[63][30] && (op_cv_q)), .pe_on(cv_pe_on),
        .p_m1(1'b0), .p_m2(1'b0), .p_m1d(r_rdelta[19:0]),
        .p_m2d(r_scr0[19:0] - r_out[19:0]), .p_m1x(mrow[9:0]), .wc_keep(cv_wck),
        .p_mf(1'b0), .p_mtl(act[33][15:0]), .p_mbl(act[49][15:0]), .wc_fill(cv_wcf),
        .p_sm(act[63][3] && (op_cv_q)), .p_smo(1'b0), .p_spd(act[63][10] && act[63][3] && (op_cv_q)), .p_padl(5'd0), .p_padr(act[61][20:16]), .p_spt(act[63][12] && act[63][3] && act[63][0] && (op_cv_q)), .p_spt_t0(act[61][31:24]), .p_spt_nv(act[62][31:24]), .p_spt_pv(act[50][7:0]), .p_spt_l(act[61][23:21]), .p_pp(act[60][2] && (op_cv_q) && !r_fs), .p_sp(act[47][7:0]), .p_segf(act[47][15:8]), .p_segl(act[47][23:16]), .p_sm2(act[47][24]),
        .p_dropf(act[48][7:0]), .p_dropl(act[48][15:8]), .p_rdelta(r_rdelta), .fz_b(fz_b), .fz_b2(fz_b2),
        .p_pt(r_pt), .p_pl(r_pl), .p_hin(r_hin), .p_win(r_win), .p_ci(r_ci), .p_sw(r_sw), .p_fx0off(r_fx0off),
        .rc0_valid(cv_rc0v), .rc0_ready(rc_ready[0]), .rc0_addr(cv_rc0a), .rc0_len(cv_rc0l),
        .rs0_avail(pk_hold ? 8'd0 : rs_avail[7:0]), .rs0_data(rs_data[127:0]), .rs0_pop(cv_pop0), .rs0_peek(cv_peek0),   // (v55AD) pk_hold
        .rc1_valid(cv_rc1v), .rc1_ready(rc_ready[1]), .rc1_addr(cv_rc1a), .rc1_len(cv_rc1l),
        .rs1_avail(rs_avail[15:8]), .rs1_data(rs_data[255:128]), .rs1_pop(cv_pop1),
        .wc_valid(cv_wcv), .wc_ready(wc_ready), .wc_addr(cv_wca), .wc_len(cv_wcl),
        .wc_tee(cv_wct), .p_tee(t_on), .wc_pm(cv_wpm), .wc_sk(cv_wsk), .p_pm(1'b0),
        .wd_valid(cv_wdv), .wd_space(wd_space), .wd_data(cv_wdd), .wd_n(cv_wdn), .wr_idle(wr_idle),
        .rqa_v(cv_rqav), .rqb_v(cv_rqbv), .rqa_acc(cv_rqaacc), .rqa_mult(cv_rqamult),
        .rqb_acc(cv_rqbacc), .rqb_mult(cv_rqbmult), .rqa_shift(cv_rqash), .rqb_shift(cv_rqbsh),
        .rqc_v(cv_rqcv), .rqd_v(cv_rqdv), .rqc_acc(cv_rqcacc), .rqc_mult(cv_rqcmult),
        .rqd_acc(cv_rqdacc), .rqd_mult(cv_rqdmult), .rqc_shift(cv_rqcsh), .rqd_shift(cv_rqdsh),
        .rqa_ov(rq_ov && sel_cv), .rqb_ov(rqb_ov && sel_cv), .rqc_ov(rqc_ov && sel_cv), .rqd_ov(rqd_ov && sel_cv),
        .rqa_q(rq_q), .rqb_q(rqb_q), .rqc_q(rqc_q), .rqd_q(rqd_q),
        .rqx_v(cv_rqxv), .rqx_acc(cv_rqxacc), .rqx_mult(cv_rqxmult), .rqx_shift(cv_rqxsh),
        .rqx_ov(rqx_ov[0] && sel_cv), .rqx_q(rqx_q), .chain_ok(cv_chain), .bg_yield(cv_bgy), .dwf_bad(cv_dwfbad), .busy(cv_busy));

    // ------------------------------------------------------------------- engines
    // --- pad / copy ---
    logic         pd_rcv, pd_fill, pd_wcv, pd_wdv, pad_busy;
    logic         pd_rcv1;                 // (v50_MB) OP_PIL: stream 1
    logic [31:0]  pd_rca1;
    logic [4:0]   pd_pop1;
    logic [31:0]  pd_rca, pd_wca;
    logic [19:0]  pd_rcl, pd_wcl;
    logic [4:0]   pd_pop, pd_wdn;
    logic [127:0] pd_wdd;
    npu_pad u_pad (
        .clk(clk), .rst_n(rst_n), .start(st_pad_q && !(op_pi_q && bgx)), .p_pi(op_pi_q),
        .p_il(op_il_q), .rc1_valid(pd_rcv1), .rc1_ready(rc_ready[1]), .rc1_addr(pd_rca1),
        .rs1_avail(rs_avail[15:8]), .rs1_data(rs_data[255:128]), .rs1_pop(pd_pop1),
        .p_in(r_in), .p_out(r_out), .p_rows(r_prows), .p_hb(r_phb), .p_h(r_ph),
        .p_wbc(r_pwbc[15:0]), .p_wc(r_pwc), .p_wac(r_pwac), .p_woc(r_pwoc), .p_total(r_ptot), .p_inlen(r_count[19:0]), .p_val(r_pval),
        .rc_valid(pd_rcv), .rc_ready(rc_ready[0]), .rc_fill(pd_fill), .rc_addr(pd_rca), .rc_len(pd_rcl),
        .rs_avail(rs_avail[7:0]), .rs_data(rs_data[127:0]), .rs_pop(pd_pop),
        .wc_valid(pd_wcv), .wc_ready(wc_ready), .wc_addr(pd_wca), .wc_len(pd_wcl),
        .wd_valid(pd_wdv), .wd_ready(wd_ready), .wd_data(pd_wdd), .wd_n(pd_wdn), .busy(pad_busy));

    // --- (v55_KC) constant-channel probe ---
    logic        st_kc_q, kc_busy, kc_wcv, kc_wdv, kc_rqv;
    logic [1:0]  kc_rcv;
    logic [63:0] kc_rca;
    logic [39:0] kc_rcl;
    logic [5:0]  kc_pop0;
    logic [7:0]  kc_peek0;
    logic [4:0]  kc_pop1, kc_wdn;
    logic [31:0] kc_wca, kc_rqa, kc_rqb, kc_rqm;
    logic [19:0] kc_wcl;
    logic [127:0] kc_wdd;
    logic [7:0]  kc_rqs;
    logic [6:0]  kc_lrec;
    always_ff @(posedge clk or negedge rst_n) if (!rst_n) st_kc_q <= 1'b0; else st_kc_q <= start && op_kc_q;
    npu_kcp u_kcp (
        .clk(clk), .rst_n(rst_n), .start(st_kc_q),
        .p_wgt(act[9][19:0]), .p_bias(act[10][19:0]), .p_mult(act[11][19:0]), .p_shift(act[12][19:0]),
        .p_st(act[13][19:0]), .p_lst(act[8][19:0]), .p_cb(act[36][19:0]), .p_cm(act[37][19:0]), .p_cs(act[38][19:0]),
        .p_cw(act[39][19:0]), .p_oh(act[45][19:0]), .p_lg(act[49][19:0]),
        .p_co(act[17][8:0]), .p_k(act[25][12:0]), .p_kcap(act[21][6:0]),
        .rc_valid(kc_rcv), .rc_ready(rc_ready), .rc_addr(kc_rca), .rc_len(kc_rcl),
        .rs0_avail(rs_avail[7:0]), .rs1_avail(rs_avail[15:8]), .rs0_data(rs_data[127:0]), .rs0_data2(rs0_data2),
        .rs1_data(rs_data[255:128]), .rs0_pop(kc_pop0), .rs0_peek(kc_peek0), .rs1_pop(kc_pop1),
        .wc_valid(kc_wcv), .wc_ready(wc_ready), .wc_addr(kc_wca), .wc_len(kc_wcl),
        .wd_valid(kc_wdv), .wd_ready(wd_ready), .wd_data(kc_wdd), .wd_n(kc_wdn), .wr_idle(wr_idle),
        .rq_v(kc_rqv), .rq_acc(kc_rqa), .rqb_acc(kc_rqb), .rq_mult(kc_rqm), .rq_shift(kc_rqs),
        .rq_ov(rq_ov), .rq_q(rq_q), .rqb_q(rqb_q), .busy(kc_busy), .ovf(kc_ovf), .lrec_o(kc_lrec));

    // ---------------- muxes ----------------
    always_comb begin
        rc_valid = 2'b00; rc_fill = 2'b00; rc_addr = '0; rc_len = '0; rc_val = 16'd0;   /* (v58_AR fv) no fill streams */
        rs_pop = '0; rs_peek = '0;
        wc_valid = 1'b0; wc_addr = '0; wc_len = '0; wc_keep = 1'b0; wc_fill = 1'b0; wc_dup = 1'b0; wc_tee = 1'b0; wc_pm = 1'b0; wc_sk = 1'b0;
        wd_valid = 1'b0; wd_data = '0; wd_n = 5'd0;
        rq_v = 1'b0; rq_acc = '0; rq_mult = '0; rq_shift = '0;
        rqb_v = 1'b0; rqb_acc = '0; rqb_mult = '0; rqb_shift = '0;
        if (sel_cv) begin
            rc_valid = {cv_rc1v, cv_rc0v}; rc_addr = {cv_rc1a, cv_rc0a}; rc_len = {cv_rc1l, cv_rc0l};
            rs_pop = {3'd0, cv_pop1, cv_pop0}; rs_peek[7:0] = cv_peek0;
            wc_valid = cv_wcv; wc_addr = cv_wca; wc_len = cv_wcl; wc_keep = cv_wck; wc_fill = cv_wcf; wc_dup = cv_wcdup; wc_tee = cv_wct; wc_pm = cv_wpm; wc_sk = cv_wsk;
            wd_valid = cv_wdv; wd_data = cv_wdd; wd_n = cv_wdn;
            // fused ADD: the ADD's results (partial sums of earlier chunk passes come from conv8)
            if (r_fz && rq_ov) begin
                wd_valid = 1'b1; wd_data = {112'd0, rqb_q, rq_q}; wd_n = rqb_ov ? 5'd2 : 5'd1;
                if (r_fz2 && f2_n == 2'd3) begin       // (v46RS) the group's last ADD pair + its 8 conv bytes
                    wd_data = {48'd0, f2_d[3], f2_c[2], f2_c[1], f2_c[0], rqb_q, rq_q}; wd_n = 5'd10;
                end
            end
            rq_v = cv_rqav; rq_acc = cv_rqaacc; rq_mult = (pa_on || cv_pe_on) ? r_mo : cv_rqamult; rq_shift = (pa_on || cv_pe_on) ? 8'd0 : cv_rqash;
            rqb_v = cv_rqbv; rqb_acc = cv_rqbacc; rqb_mult = (pa_on || cv_pe_on) ? r_mo : cv_rqbmult; rqb_shift = (pa_on || cv_pe_on) ? 8'd0 : cv_rqbsh;
        end else if (sel_pad) begin
            rc_valid[0] = pd_rcv; rc_fill[0] = pd_fill; rc_addr[31:0] = pd_rca; rc_len[19:0] = pd_rcl;
            rs_pop[7:0] = {3'd0, pd_pop};
            rc_valid[1] = pd_rcv1; rc_addr[63:32] = pd_rca1; rc_len[39:20] = pd_rcl;   // (v50_MB) OP_PIL plane 1
            rs_pop[15:8] = {3'd0, pd_pop1};
            wc_valid = pd_wcv; wc_addr = pd_wca; wc_len = pd_wcl;
            wd_valid = pd_wdv; wd_data = pd_wdd; wd_n = pd_wdn;
        end else if (op_kc_q) begin                    // (v55_KC)
            rc_valid = kc_rcv; rc_addr = kc_rca; rc_len = kc_rcl;
            rs_pop = {3'd0, kc_pop1, 2'd0, kc_pop0}; rs_peek[7:0] = kc_peek0;
            wc_valid = kc_wcv; wc_addr = kc_wca; wc_len = kc_wcl;
            wd_valid = kc_wdv; wd_data = kc_wdd; wd_n = kc_wdn;
            rq_v = kc_rqv; rq_acc = kc_rqa; rq_mult = kc_rqm; rq_shift = kc_rqs;
            rqb_v = kc_rqv; rqb_acc = kc_rqb; rqb_mult = kc_rqm; rqb_shift = kc_rqs;
        end
    end

    assign busy_core = copy_q || start || st_cv_q || st_add_q || st_pad_q ||
                       cv_busy || pad_busy || kc_busy || st_kc_q ||
                       !rd_idle || !wr_idle || pk_act;   // (v55AD)
    assign busy = busy_core || seq_on;
    // (v42DS) chaining: the staged op is a conv with the running DWF op's output geometry and quantisation (RG_CP
    // word, PG bits, HO, WO, CO, NPIX, OOFF, AMIN, AMAX); the running op's last pass has drained its MACs (cv_chain),
    // no start / copy is in flight, the read DMA is idle and no chunk sequence runs
    wire chain_cmp = (regs[63] == act[63]) && (regs[60][10:0] == act[60][10:0]) && (regs[15][7:0] == act[15][7:0]) &&
                     (regs[16][7:0] == act[16][7:0]) && (regs[17][9:0] == act[17][9:0]) && (regs[52][11:0] == act[52][11:0]) &&
                     (regs[30][8:0] == act[30][8:0]) && (regs[31][7:0] == act[31][7:0]) && (regs[32][7:0] == act[32][7:0]);
    wire chain_go  = cv_chain && (op_n == OP_CONV) && (op_q == OP_CONV) && chain_cmp && rd_idle &&
                     !copy_q && !start && !st_cv_q && !seq_on;
    // (v50_RS BGP) an OP_PADI waits for the filler; a BG one also for the copy engine (one BG PAD at a time)
    assign take = cmd_pend && (!busy || chain_go) && !((sp_on || 1'b0) && op_n == OP_PADI);

    // ------------------------------------------------------------------- (v49DS) background border filler
    // An OP_PADI hands its border runs over at its start: run 0 = RG_PTOT bytes at RG_OUT; after each of the RG_PH
    // interior rows (RG_PWC bytes skipped) one run of RG_PWAC bytes, the last one RG_PWBC bytes. The write DMA takes
    // them in pieces inside one 128-byte block when no data burst can start; not offered during a PAD op, nor while an
    // op is queued and the engines are idle (the hand-over), except to a queued OP_PADI (it waits for the filler). A new OP_PADI
    // starts with the write DMA idle (take needs !busy), so sp_val is stable for every fill burst.
    logic [11:0] sp_l, sp_last, sp_mid, sp_skip;   // (lengths < 4096: the kernel checks)
    logic [7:0]  sp_k;
    wire  [7:0]  sp_blk = 8'd128 - {1'b0, sp_a[6:0]};
    assign       sp_n   = (sp_l < {4'd0, sp_blk}) ? sp_l[7:0] : sp_blk;
    assign       sp_bgv = sp_on && (sp_l != '0) && !pad_busy && (cv_busy || !cmd_pend || (op_n == OP_PADI));
    wire         sp_adv = sp_on && (sp_tk || sp_l == '0);
    wire         sp_end = (sp_l <= {4'd0, sp_n});                       // the run's last piece (or an empty run)
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) sp_on <= 1'b0;
        else sp_on <= 1'b0;   // (v58_AR) tied: no OP_PADI
    end
    always_ff @(posedge clk) begin
        if (st_pad_q && op_pi_q) begin
            sp_a <= bgx ? act[28][19:0] : r_out[19:0]; sp_l <= r_ptot[11:0]; sp_k <= bgx ? act[49][19:12] : r_ph;
            sp_mid <= bgx ? act[33][23:12] : r_pwac[11:0]; sp_skip <= r_pwc[11:0];   // (v50_RS BGP: BG registers)
            sp_last <= bgx ? act[33][11:0] : r_pwbc[11:0]; sp_val <= r_pval;
        end else if (sp_adv) begin
            if (!sp_end) begin sp_a <= sp_a + {12'd0, sp_n}; sp_l <= sp_l - {4'd0, sp_n}; end    // next piece
            else if (sp_k != 8'd0) begin                                                        // next run
                sp_a <= sp_a + {8'd0, sp_l} + {8'd0, sp_skip}; sp_k <= sp_k - 8'd1;
                sp_l <= (sp_k == 8'd1) ? sp_last : sp_mid;
            end
        end
    end

    // ------------------------------------------------------------------- (v50_RS BGP) background PAD copy
    // A BG OP_PADI's interior rows (RG 49 / 26: ph rows from in, contiguous; RG 28 / 49: the first at out + ptot, rows
    // RG_PWC + pwac apart) are copied a piece at a time: a read burst (only when no stream proposes one) and a write burst
    // (in the write DMA's idle slots) taken in the same cycle, the read beats forwarded to the write channel. Pieces start
    // while a conv runs (not in a multi-chunk conv's final chunk pass: its outputs / fused ADD use the
    // ports) or nothing is queued (a queued op is taken first; a queued BG PADI waits for the copy).
    assign bgd_v = bgc_on && bgr_ok && !pad_busy && (cv_busy ? !cv_bgy : (!cmd_pend || 1'b0));
    wire   bgc_le = (bgc_r == {4'd0, bgc_l});                 // the row's last piece
    always_ff @(posedge clk or negedge rst_n) if (!rst_n) bgc_cap <= 1'b0; else bgc_cap <= 1'b0;   // (v58_AR) tied
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) bgc_on <= 1'b0;
        else bgc_on <= 1'b0;   // (v58_AR) tied
    end
    always_ff @(posedge clk) begin
        if (st_pad_q && op_pi_q && bgx) begin
            bgc_p <= act[28][19:0] + {8'd0, act[49][11:0]}; bgc_q <= act[26][19:0]; bgc_r <= act[46][11:0]; bgc_k <= act[49][19:12];
            bgc_wc <= act[46][11:0]; bgc_ac <= act[33][23:12];
        end else if (bgd_tk && !pk_act) begin
            if (bgc_le) begin bgc_p <= bgc_p + {12'd0, bgc_l} + {8'd0, bgc_ac}; bgc_r <= bgc_wc; bgc_k <= bgc_k - 8'd1; end
            else begin bgc_p <= bgc_p + {12'd0, bgc_l}; bgc_r <= bgc_r - {4'd0, bgc_l}; end
            bgc_q <= bgc_q + {12'd0, bgc_l};
        end
    end

    // ------------------------------------------------------------------- (v55AD) pack tee
    // A raw FC stream op with RG_CP bit 9 also writes the packed stream of its weights (npu_pkw.sv): destination
    // RG_WGSTR (bit 31 set: explicit, else continue after the previous pack op), limit RG_DWADJ.
    npu_pkw #(.CAPB(128), .NB(4)) u_pk (
        .clk(clk), .rst_n(rst_n), .start(st_cv_q && r_fs && act[63][9]), .k(r_k), .co(r_co), .hasb(r_hasb),
        .dest(act[26][31] ? act[26][19:0] : 20'd0), .lim(act[28][19:0]),
        .pop_n(sel_cv ? cv_pop0 : 8'd0), .pop_d(rs_data[127:0]), .hold(pk_hold),
        .bgd_valid(pk_bgv), .bgd_addr(pk_bga), .bgd_len(pk_bgl), .bgd_take(bgd_tk && pk_act),
        .bgr_v(pk_brv), .bgr_data(pk_brd), .bgr_rdy(bgr_rdy && pk_act),
        .act(pk_act), .len_q(pk_len), .ovf_q(pk_ovf), .uni_q(pk_uni),
        .f_push(pk_fpush), .f_pd(pk_fpd), .f_pn(pk_fpn), .f_ird(pk_frdy), .f_avail_i(rs_avail[15:8]), .f_data(rs_data[255:128]),
        .f_pop(pk_fpop));

    // ------------------------------------------------------------------- (v34P) chunk sequence
    wire         seq_last = (seq_c == act[63][18:16] - 3'd1);   // act[63][18:16]: chunk ops in the sequence
    assign       seq_raw  = seq_on && !seq_last;
    assign       r_actlen = (seq_on && seq_last) ? act[26][19:0] : r_actlen0;
    // partial sums: chunk c reads SCR half (c - 1) & 1 (chunk 0: the bias words), writes half c & 1
    assign r_scr  = !seq_on ? r_scr0 : (seq_c == 3'd0) ? r_bias : {r_scr0[31:10], ~seq_c[0], r_scr0[8:0]};
    assign r_outc = seq_raw ? {r_scr0[31:10], seq_c[0], r_scr0[8:0]} : r_out;
    // GO: a write to register word 4 (lane 0) while a sequence is queued or running
    wire go_w = s_wvalid && s_wready && (aw_a[7:4] == 4'd1) && (s_wstrb[3:0] != 4'd0) &&
                seq_on;                                // (v52AD)
    assign seq_adv = seq_on && !busy_core && !seq_last;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            seq_on <= 1'b0; seq_c <= '0; go <= 1'b0;
        end else begin
            if (go_w) go <= 1'b1;
            if (take) begin                                     // a queued op is copied in
                seq_on <= 1'b0; seq_c <= '0;           // (v52AD) no chunk sequence (was RG_CP bit 19)
            end else if (seq_on && !busy_core) begin
                if (seq_last) begin seq_on <= 1'b0; go <= 1'b0; end
                else seq_c <= seq_c + 3'd1;
            end
        end
    end
    always_ff @(posedge clk) begin
        seq_in <= '0; seq_wgt <= '0;                   // (v52AD) unused
    end
    // (v56_CY) status bit 1 captured when the read data is presented (R(8), row 2, is never held)
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) st_bg <= 1'b0;
        else if (ar_act && !s_rvalid && !blocked) st_bg <= sp_on || bgc_on || wr_bginq;
    end
endmodule

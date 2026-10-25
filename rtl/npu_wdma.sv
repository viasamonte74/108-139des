// npu_wdma - ordered write stream over the AXI master write channel.
// Commands {addr,len} describe where the next len bytes of the input byte stream go. Bytes are
// pushed by the engine into an internal byte FIFO; bursts carry exact byte strobes, so writes at
// any alignment never touch neighbouring bytes. done_all = everything written and acknowledged.
module npu_wdma #(
    parameter int CAPB = 64,
    parameter int MAXB = 16,
    parameter int CQD  = 4,
    parameter int BGI  = 2,             // (v49DS) idle cycles before a background fill burst
    parameter int BGF  = 48,            // (v49DS) ... and at most this many bytes waiting in the FIFO
    parameter int BGD  = 96,            // (v50_RS BGP) background bursts: at most this many bytes waiting in the FIFO
    parameter int CAPN = 3              // (v50_RS BGP) data burst beats while a copy piece is due (bg_cap)
) (
    input  logic         clk,
    input  logic         rst_n,
    input  logic         clr,
    input  logic         cmd_valid,
    output logic         cmd_ready,
    input  logic [31:0]  cmd_addr,
    input  logic [19:0]  cmd_len,
    input  logic         cmd_keep,     // (v46MB) the next command writes the same bytes: do not consume them
    input  logic         cmd_fill,     // (v46MB) constant bytes (fill_val), none taken from the FIFO
    input  logic [7:0]   fill_val,
    input  logic         cmd_dup,      // (v46RS) then as many further bytes at cmd_addr + dup_delta (len < 64)
    input  logic [19:0]  dup_delta,
    // (v49_RS) write tee
    input  logic         cmd_tee,      // an output write of the op (teed while t_on)
    input  logic         t_on,
    input  logic         t_l64,        // the teed tensor has 64 channels (else 32)
    input  logic [19:0]  t_b3,
    input  logic [19:0]  t_out,
    input  logic [15:0]  t_plb,
    input  logic         t_rp,         // (v49_RS RS3) the copy is row-pair planes (W = 8 << t_w)
    input  logic [1:0]   t_w,
    input  logic         t_r8,         // (v58_RS) rotated planes (r_w8.py)
    input  logic         cmd_pm,       // (v49_RS RS3) pair-merged A: 3 * len at addr (keep), 3 * len at addr + dup_delta
    input  logic         cmd_sk,       // (v49_RS RS3) its B: drop len bytes (no write)
    // (v50_MB) replicated whole-output writes (see the expander below)
    input  logic         rp_on,
    input  logic         rp_nb,
    input  logic         rp_al,
    input  logic [19:0]  rp_da, rp_db, rp_tp,
    input  logic [11:0]  rp_rw, rp_top, rp_bot,
    input  logic [7:0]   rp_sk, rp_ch, rp_h,
    // byte input
    input  logic         in_valid,
    output logic         in_ready,
    input  logic [127:0] in_data,
    input  logic [4:0]   in_n,
    output logic [7:0]   in_space,     // free bytes in the input FIFO
    output logic         idle,
    output logic         bg_inq,       // (v56_CY) a background burst (fill / copy) is in the burst queue
    // (v49DS) background fill bursts: bg_len (1..128) bytes of bg_val at bg_addr, inside one 128-byte block
    input  logic         bg_valid,
    input  logic [19:0]  bg_addr,
    input  logic [7:0]   bg_len,
    input  logic [7:0]   bg_val,
    output logic         bg_take,
    // (v50_RS BGP) background PAD copy: a data burst of bgd_len (1..128) bytes at bgd_addr inside one 128-byte
    // block, taken like a fill burst (before one); its beats are the read channel's beats of the matching read burst
    // (bgr_v / bgr_data, lane for lane), which wait for the write channel (bgr_rdy)
    input  logic         bgd_valid,
    input  logic [19:0]  bgd_addr,
    input  logic [7:0]   bgd_len,
    output logic         bgd_take,
    input  logic         bgr_v,
    input  logic [127:0] bgr_data,
    output logic         bgr_rdy,
    // (v50_RS BGP) a copy piece is due: data bursts of at most CAPN beats whose AW waits for all their bytes (a
    // burst whose bytes trickle in - a MAC-bound pass's partial sums - no longer holds the write channel)
    input  logic         bg_cap,
    // AXI write
    output logic [31:0]  awaddr,
    output logic [7:0]   awlen,
    output logic         awvalid,
    input  logic         awready,
    output logic [127:0] wdata,
    output logic [15:0]  wstrb,
    output logic         wlast,
    output logic         wvalid,
    input  logic         wready,
    input  logic         bvalid,
    output logic         bready
);
    localparam int AW = $clog2(CAPB);
    localparam int QW = $clog2(CQD);

    assign bready = 1'b1;

    // ---- byte FIFO ----
    logic [3:0]   bw_lo;                 // AXI byte lane of the next stream byte (see below)
    logic [AW:0]  f_avail, f_used;
    logic [7:0]   pk;                    // (v46MB) bytes of the current keep command already sent
    logic [7:0]   w_pk;                  // (v49_RS RS3) the FIFO window: pk, or 8 for a p8 burst
    logic        wq_p8    [2];          // (v49_RS RS3) burst popping with the window 8 bytes on
    logic        wq_drop  [2];          // (v49_RS RS3) drop entry
    logic [127:0] f_data;
    logic [4:0]   f_pop;
    npu_bsf #(.CAPB(CAPB), .EXACT(1)) u_bsf (
        .clk(clk), .rst_n(rst_n), .clr(clr),
        .in_valid(in_valid), .in_ready(in_ready), .in_data(in_data), .in_lo(4'd0), .in_n(in_n),
        .out_avail(f_avail), .out_used(f_used), .out_data(f_data), .peek(w_pk), .out_rot(bw_lo),
        .pop_n({3'd0, f_pop}));

    // ---- command queue ----
    logic [45:0] cq_mem [CQD];                 // {dup, tee, fill, keep, len, DRAM offset (20 bits)} (v48C: + v46RS dup, v49RS tee)
    logic [QW:0] cq_cnt;
    logic [QW-1:0] cq_rp, cq_wp;
    wire cq_room  = (cq_cnt != (QW+1)'(CQD));
    logic ex_v, ex_keep, ex_fill;
    logic [19:0] ex_addr;
    logic [11:0] ex_len;
    wire ex_push  = ex_v && cq_room;                       // (v50_MB) the expander's command
    wire cq_push  = ex_push || (!rp_on && cmd_valid && cq_room);
    wire cq_empty = (cq_cnt == '0);
    assign cmd_ready = rp_on ? !ex_v : cq_room;
    // (v26A: queue entries written through gated clocks)
    // (v28A: the push is staged one cycle - registered entry index and data - so that each entry's
    // gate enable is a flop compare, not the engine's command logic; a take of the entry being
    // written reads the staged data)
    logic          cs_v;
    logic [QW-1:0] cs_i;
    logic [45:0]   cs_d;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) cs_v <= 1'b0; else cs_v <= cq_push;
    end
    always_ff @(posedge clk) begin cs_i <= cq_wp;
        cs_d <= ex_push ? {4'd0, ex_fill, ex_keep, 8'd0, ex_len, ex_addr}
                        : {cmd_sk, cmd_pm, cmd_dup, cmd_tee && t_on, cmd_fill, cmd_keep, cmd_len, cmd_addr[19:0]}; end

    // ---- (v50_MB) expander: the command taken while rp_on is written in chunks: own (keep), A (keep if B), [B];
    // A = own + ex_da steps over rp_sk fill bytes after each rp_rw-byte row, rp_top fill bytes at rp_tp first and
    // rp_bot after the last row (a PAD output's interior + border); B = own + rp_db is a plain copy
    localparam logic [2:0] EX_TOP = 3'd0, EX_OWN = 3'd1, EX_A = 3'd2, EX_B = 3'd3, EX_SK = 3'd4, EX_BOT = 3'd5;
    logic [2:0]  ex_ph;
    logic [19:0] ex_o, ex_da;
    logic [11:0] ex_c;                                      // bytes of the current row done
    logic [7:0]  ex_r;                                      // rows left (this one included)
    // chunk: rp_ch bytes (rp_al: up to the own range's next 16-byte line), never past the row's end
    wire  [7:0]  ex_cp = rp_al ? (rp_ch - {4'd0, ex_o[3:0]}) : rp_ch;
    wire  [11:0] ex_rl = rp_rw - ex_c;
    wire         ex_re = (ex_rl <= {4'd0, ex_cp});          // the chunk ends the row
    wire  [11:0] ex_ck = ex_re ? ex_rl : {4'd0, ex_cp};
    wire         ex_ls = ex_re && (ex_r == 8'd1);          // the last chunk
    wire  [19:0] ex_aa = ex_o + ex_da;
    always_comb begin
        ex_keep = 1'b0; ex_fill = 1'b0; ex_addr = ex_aa; ex_len = ex_ck;
        case (ex_ph)
            EX_TOP:  begin ex_addr = rp_tp; ex_len = rp_top; ex_fill = 1'b1; end
            EX_OWN:  begin ex_addr = ex_o; ex_keep = 1'b1; end
            EX_A:    begin ex_keep = rp_nb; end
            EX_B:    begin ex_addr = ex_o + rp_db; end
            EX_SK:   begin ex_len = {4'd0, rp_sk}; ex_fill = 1'b1; end
            default: begin ex_len = rp_bot; ex_fill = 1'b1; end
        endcase
    end
    wire ex_take = rp_on && cmd_valid && !ex_v;
    wire ex_adv  = ex_push && ((ex_ph == EX_B) || ((ex_ph == EX_A) && !rp_nb));   // a chunk's last command
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ex_v <= 1'b0; ex_ph <= EX_OWN; ex_o <= '0; ex_da <= '0; ex_c <= '0; ex_r <= '0;
        end else if (ex_take) begin
            ex_v <= 1'b1; ex_o <= cmd_addr[19:0]; ex_da <= rp_da; ex_c <= '0; ex_r <= rp_h;
            ex_ph <= (rp_top != 12'd0) ? EX_TOP : EX_OWN;
        end else if (ex_push) begin
            if (ex_adv) begin                                 // (a chunk's last command; the case below never
                ex_o <= ex_o + {8'd0, ex_ck};                  //  assigns then)
                ex_c <= ex_re ? 12'd0 : (ex_c + ex_ck);
                if (ex_re) ex_r <= ex_r - 8'd1;
                if (ex_ls) begin if (rp_bot != 12'd0) ex_ph <= EX_BOT; else ex_v <= 1'b0; end
                else ex_ph <= (ex_re && (rp_sk != 8'd0)) ? EX_SK : EX_OWN;
            end
            case (ex_ph)
                EX_TOP: ex_ph <= EX_OWN;
                EX_OWN: ex_ph <= EX_A;
                EX_A:   if (rp_nb) ex_ph <= EX_B;
                EX_SK:  begin ex_da <= ex_da + {12'd0, rp_sk}; ex_ph <= EX_OWN; end
                EX_BOT: ex_v <= 1'b0;
                default: ;
            endcase
        end
    end
    for (genvar i = 0; i < CQD; i++) begin : g_cq
        wire g = clk | ~(cs_v && cs_i == QW'(i));
        logic [45:0] e;
        always_ff @(posedge g) e <= cs_d;
        assign cq_mem[i] = e;
    end
    wire [45:0] cq_head = (cs_v && cs_i == cq_rp) ? cs_d : cq_mem[cq_rp];

    // ---- current command / burst state ----
    // AW generation and W beats are decoupled: the next burst's AW is prepared (and presented)
    // while the previous burst's W beats are still being sent; a 2-entry queue carries the
    // geometry of the bursts whose AW has been prepared to the W side.
    logic        cur_v;
    logic [19:0] cur_addr;              // next byte (DRAM offset) to write
    logic [19:0] aw20;
    assign awaddr = {12'h804, aw20};
    logic [19:0] cur_rem;               // bytes not yet assigned to a burst
    logic        cur_keep;              // (v46MB) the current command keeps its bytes
    logic        cur_fill;              // (v46MB) the current command is a fill
    logic        aw_pend;               // AW presented, waiting for handshake
    logic [7:0]  outst;                 // bursts awaiting B
    // W burst queue
    logic [8:0]  wq_beats [2];
    logic [12:0] wq_bytes [2];
    logic [3:0]  wq_lo    [2];
    logic        wq_bd    [2];          // (v50_RS BGP) background copy burst (beats from the read channel)
    logic        wq_bg    [2];          // (v49DS) background fill burst (constant bg_val, no FIFO bytes)
    logic        wq_keep  [2];          // (v46MB) burst of a keep command
    logic        wq_cl    [2];          // (v46MB) its command's last burst
    logic        wq_fill  [2];          // (v46MB) fill burst (constant bytes)
    logic [1:0]  wq_cnt;
    logic        wq_rp, wq_wp;
    logic        wfirst;                // the head burst has not sent a beat yet
    logic [8:0]  bw_left;               // W beats left in the head burst (after its first beat)
    logic [12:0] bb_left;               // bytes left in the head burst (after its first beat)

    wire aw_free = !aw_pend || awready;
    // (v46RS) a dup command's second range: taken when its first range is assigned to bursts
    logic        d_pend;
    logic [19:0] d_addr;
    logic [5:0]  d_len;
    wire take = (!cur_v || (cur_rem == '0 && !d_pend)) && !cq_empty;

    // next burst geometry (v46MB: of the head entry when it is taken this cycle - its first burst starts at once)
    wire        h_pm   = cq_head[44];       // (v49_RS RS3) pair-merged A
    wire        h_sk   = cq_head[45];       // (v49_RS RS3) its B: a drop
    wire [19:0] h_len3 = {cq_head[38:20], 1'b0} + cq_head[39:20];
    logic       cur_p8, cur_pm, cur_drop;   // (v49_RS RS3) the range pops with the window 8 bytes on / pm / drop
    wire [19:0] s_addr = take ? (h_sk ? 20'd0 : cq_head[19:0]) : cur_addr;
    wire [19:0] s_rem  = take ? (h_pm ? h_len3 : cq_head[39:20]) : cur_rem;
    wire        s_p8   = take ? 1'b0 : cur_p8;
    wire        s_drop = take ? h_sk : cur_drop;
    wire        h_tee  = cq_head[42];       // (v49_RS) a teed command: its own range first as a keep pass
    wire        s_keep = take ? (cq_head[40] || h_tee || h_pm) : cur_keep;
    wire        s_fill = take ? cq_head[41] : cur_fill;
    logic [20:0] span;
    logic [8:0]  tot, to4k, nb;
    logic [12:0] nby;
    always_comb begin
        span = 21'(s_addr[3:0]) + 21'(s_rem);
        tot  = (bg_cap ? (span >= 21'(CAPN*16)) : (span >= 21'(MAXB*16))) ? (bg_cap ? 9'(CAPN) : 9'(MAXB)) :
               9'((span + 21'd15) >> 4);
        to4k = 9'd256 - {1'b0, s_addr[11:4]};               // beats to the 4KB boundary (>= 1)
        nb   = (to4k < tot) ? to4k : tot;
        if (({12'd0, nb} << 4) - 21'(s_addr[3:0]) >= 21'(s_rem)) nby = 13'(s_rem);
        else nby = 13'(({12'd0, nb} << 4) - 21'(s_addr[3:0]));
    end
    // (v49DS) a plain data burst's AW goes out once its first beat's bytes (16, or the whole burst) are in the FIFO
    // beyond those of the plain bursts already prepared (keep / fill bursts: as before)
    wire [4:0]  fb_n = (nby >= 13'd16) ? 5'd16 : 5'(nby);
    wire [12:0] w_res = (w_has && !hf && !hk) ? bleft : 13'd0;   // the prepared (head) plain burst's bytes not yet sent
    wire        d_ok = s_keep || s_fill || s_drop || (13'(f_avail) >= w_res + (bg_cap ? nby : {8'd0, fb_n}));
    // (v49_RS RS3) a drop needs no AW
    wire bgx_t;                         // (v50_RS BGP) a background burst (fill or copy) is taken: it goes first
    wire start_burst = (take || cur_v) && (s_rem != '0) && (aw_free || s_drop) && (wq_cnt != 2'd2) && d_ok && !bgx_t;
    // background fill burst: the write port is free (no burst prepared), no data burst could start for BGI cycles
    // (or the previous burst was a fill and still none can) and the FIFO holds at most BGF bytes (the producer has
    // room for the fill burst's <= 9 cycles): pass headers, read- / MAC-bound phases, between ops
    wire [8:0]  bgf_nb = ({5'd0, bg_addr[3:0]} + {1'b0, bg_len} + 9'd15) >> 4;
    wire [8:0]  bgd_nb = ({5'd0, bgd_addr[3:0]} + {1'b0, bgd_len} + 9'd15) >> 4;   // (v50_RS BGP)
    wire [19:0] bgx_a = bgd_valid ? bgd_addr : bg_addr;   // (a fill is taken only without a copy request)
    wire [7:0]  bgx_l = bgd_valid ? bgd_len : bg_len;
    wire [8:0]  bg_nb = bgd_valid ? bgd_nb : bgf_nb;
    logic [3:0] w_idl;                  // cycles with nothing prepared and no data burst starting
    logic       bg_last;                // the last burst prepared was a fill
    // (v50_RS BGP) background bursts (copy pieces, then fills) take the port before a data burst whenever the FIFO
    // holds at most BGD bytes and nothing is prepared - or one burst of the same kind whose AW is out (back to back)
    wire   bgd_ok  = (f_used <= (AW+1)'(BGD)) && ((wq_cnt == 2'd0) || ((wq_cnt == 2'd1) && wq_bd[wq_rp] && !aw_pend));
    wire   bgf_ok  = (f_used <= (AW+1)'(BGD)) && ((wq_cnt == 2'd0) || ((wq_cnt == 2'd1) && wq_fill[wq_rp] && !wq_bd[wq_rp] && !aw_pend));
    assign bgd_take = bgd_valid && bgd_ok;
    assign bg_take = bg_valid && (bg_len != 8'd0) && bgf_ok && !bgd_valid;
    assign bgx_t   = bg_take || bgd_take;
    // ---- (v49_RS) tee: the own-range pass (keep: peeks) and the copy pass (pops) of the current command
    logic        tc;                    // the current command is teed
    logic        tp;                    // its pass: 0 own range, 1 copy
    logic [19:0] t_d3;                  // the copy's address
    logic [4:0]  t_len;                 // the command's length (<= 16)
    wire  [19:0] t_rel = cq_head[19:0] - t_out;
    wire  [2:0]  t_gi  = t_l64 ? t_rel[5:3] : {1'b0, t_rel[4:3]};   // (v58_RS r8) 8-channel group in the pixel
    wire  [2:0]  t_gm  = t_l64 ? 3'd7 : 3'd3;
    wire  [2:0]  t_go  = (t_gi == 3'd0) ? (t_gm - 3'd1) : (t_gi == t_gm) ? t_gm : (t_gi - 3'd1);
    wire  [5:0]  t_rr  = t_r8 ? {t_go, t_rel[2:0]} : t_rel[5:0];
    wire  [1:0]  t_g   = t_l64 ? t_rr[5:4] : {1'b0, t_rr[4]};
    wire  [15:0] t_px  = t_l64 ? {2'd0, t_rel[19:6]} : {1'b0, t_rel[19:5]};
    // (v49_RS RS3) row-pair planes, t_rp: (y, x) of a W = 8 << t_w wide plane at ((y >> 1) * 2W + 2x + 1 - (y & 1)) * 16
    logic [15:0] t_pq;
    always_comb begin
        t_pq = t_px;
        if (t_rp) case (t_w)
            2'd0:    t_pq = {t_px[15:4], t_px[2:0], ~t_px[3]};
            2'd1:    t_pq = {t_px[15:5], t_px[3:0], ~t_px[4]};
            default: t_pq = {t_px[15:6], t_px[4:0], ~t_px[5]};
        endcase
    end
    wire  [19:0] d3_a  = t_b3 + (t_g[0] ? {4'd0, t_plb} : 20'd0) + (t_g[1] ? {3'd0, t_plb, 1'b0} : 20'd0) +
                         {t_pq, t_rr[3:0]};
    wire         s_tc  = take ? h_tee : tc;
    wire         s_tp  = take ? 1'b0 : tp;
    wire  [19:0] s_d3  = take ? d3_a : t_d3;
    wire  [4:0]  s_tl  = take ? cq_head[24:20] : t_len;
    wire         p_last = start_burst && (s_rem == 20'(nby));   // this burst ends the current pass

    // current beat (head of the W queue)
    wire          w_has  = (wq_cnt != 2'd0);
    assign        bw_lo  = wfirst ? wq_lo[wq_rp] : 4'd0;
    wire [12:0]   bleft  = wfirst ? wq_bytes[wq_rp] : bb_left;
    wire [8:0]    bwleft = wfirst ? wq_beats[wq_rp] : bw_left;
    logic [4:0]   beat_n;
    // the FIFO's output is already rotated left by bw_lo bytes: lane (bw_lo + i) has stream byte i
    wire  [127:0] rot = f_data;
    always_comb beat_n = ((13'd16 - 13'(bw_lo)) >= bleft) ? 5'(bleft) : 5'(5'd16 - {1'b0, bw_lo});
    wire hk = wq_keep[wq_rp];             // (v46MB) the head burst keeps its bytes (peek pk past the head)
    assign w_pk = (w_has && wq_p8[wq_rp]) ? 8'd8 : pk;
    wire hf = wq_fill[wq_rp];             // (v46MB) the head burst is a fill
    wire hd = wq_drop[wq_rp];             // (v49_RS RS3) the head entry drops its bytes (no AW / beats)
    wire beat_ok = w_has && (hf ? (!wq_bd[wq_rp] || bgr_v) : ({1'b0, f_avail} >= ((AW+2)'(w_pk) + (AW+2)'(beat_n))));
    assign bgr_rdy = w_has && wq_bd[wq_rp] && wready;   // (v50_RS BGP) the copy beat is written this cycle
    assign wvalid = beat_ok && !hd;
    wire dfire = beat_ok && hd;
    assign wlast  = (bwleft == 9'd1);
    // WDATA must not change while WVALID waits: unused lanes are forced to zero, so only the
    // already-present (stable) bytes appear on the bus.
    always_comb begin
        wstrb = '0;
        // (v26A: one end position instead of a subtraction per lane)
        for (int i = 0; i < 16; i++)
            wstrb[i] = (4'(i) >= bw_lo) && (5'(i) < ({1'b0, bw_lo} + beat_n));
        for (int i = 0; i < 16; i++)
            wdata[i*8 +: 8] = wstrb[i] ? (hf ? bgr_data[i*8 +: 8] : rot[i*8 +: 8]) : 8'd0;   // (v58_AR fv) the only fill bursts left are copy bursts
    end
    wire wfire = wvalid && wready;
    assign f_pop = ((wfire && !hk && !hf) || dfire) ? beat_n : 5'd0;

    always_ff @(posedge clk) begin
        if (start_burst) begin
            wq_beats[wq_wp] <= nb;
            wq_bytes[wq_wp] <= nby;
            wq_lo[wq_wp]    <= s_addr[3:0];
            wq_keep[wq_wp]  <= s_keep && !s_fill;
            wq_fill[wq_wp]  <= s_fill;
            wq_cl[wq_wp]    <= (s_rem == 20'(nby));
            wq_bg[wq_wp]    <= 1'b0;
            wq_bd[wq_wp]    <= 1'b0;
            wq_p8[wq_wp]    <= s_p8;
            wq_drop[wq_wp]  <= s_drop;
        end else if (bgx_t) begin
            wq_beats[wq_wp] <= bg_nb;
            wq_bytes[wq_wp] <= {5'd0, bgx_l};
            wq_lo[wq_wp]    <= bgx_a[3:0];
            wq_bd[wq_wp]    <= bgd_valid;
            wq_keep[wq_wp]  <= 1'b0;
            wq_fill[wq_wp]  <= 1'b1;
            wq_cl[wq_wp]    <= 1'b1;
            wq_bg[wq_wp]    <= 1'b1;
            wq_p8[wq_wp]    <= 1'b0;
            wq_drop[wq_wp]  <= 1'b0;
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cq_cnt <= '0; cq_rp <= '0; cq_wp <= '0;
            cur_v <= 1'b0; cur_addr <= '0; cur_rem <= '0; cur_keep <= 1'b0; cur_fill <= 1'b0; pk <= '0; d_pend <= 1'b0; d_addr <= '0; d_len <= '0;
            tc <= 1'b0; tp <= 1'b0; t_d3 <= '0; t_len <= '0; cur_p8 <= 1'b0; cur_pm <= 1'b0; cur_drop <= 1'b0;
            aw_pend <= 1'b0; aw20 <= '0; awlen <= '0;
            wq_cnt <= '0; wq_rp <= 1'b0; wq_wp <= 1'b0; wfirst <= 1'b1;
            bw_left <= '0; bb_left <= '0; outst <= '0; w_idl <= '0; bg_last <= 1'b0;
        end else if (clr) begin
            cq_cnt <= '0; cq_rp <= '0; cq_wp <= '0;
            cur_v <= 1'b0; cur_rem <= '0; aw_pend <= 1'b0; pk <= '0; d_pend <= 1'b0; tc <= 1'b0; tp <= 1'b0; cur_p8 <= 1'b0; cur_pm <= 1'b0; cur_drop <= 1'b0;
            wq_cnt <= '0; wq_rp <= 1'b0; wq_wp <= 1'b0; wfirst <= 1'b1; outst <= '0;
        end else begin
            if (cq_push) cq_wp <= (cq_wp == QW'(CQD - 1)) ? '0 : cq_wp + 1'b1;
            if (take)    cq_rp <= (cq_rp == QW'(CQD - 1)) ? '0 : cq_rp + 1'b1;
            cq_cnt <= cq_cnt + (QW+1)'(cq_push) - (QW+1)'(take);
            if (aw_pend && awready) aw_pend <= 1'b0;
            if (take) begin cur_v <= 1'b1; cur_keep <= cq_head[40] || h_tee || h_pm; cur_fill <= cq_head[41];
                tc <= h_tee && !h_pm && !h_sk; tp <= 1'b0; t_d3 <= d3_a; t_len <= cq_head[24:20];
                cur_p8 <= 1'b0; cur_pm <= h_pm; cur_drop <= h_sk;   // (v49_RS RS3)
                d_pend <= cq_head[43] && !h_sk; d_addr <= cq_head[19:0] + dup_delta;
                d_len <= h_pm ? 6'(h_len3) : cq_head[25:20]; end
            if (start_burst) begin
                if (!s_drop) begin                // (v49_RS RS3) a drop issues no AW
                    aw_pend  <= 1'b1;
                    aw20     <= {s_addr[19:4], 4'd0};
                    awlen    <= 8'(nb - 9'd1);
                end
                cur_addr <= {s_addr[19:4], 4'd0} + {7'd0, nb, 4'd0};
                cur_rem  <= s_rem - 20'(nby);
                if (s_tc && p_last && !s_tp) begin  // (v49_RS tee) the own pass's last burst: the copy pass next
                    cur_addr <= s_d3; cur_rem <= {15'd0, s_tl}; tp <= 1'b1; cur_keep <= 1'b0;
                end else if (s_tc && p_last) tc <= 1'b0;
            end else if (take) begin
                cur_addr <= s_addr;
                cur_rem  <= s_rem;
            end else if (cur_v && cur_rem == '0) begin
                if (d_pend) begin cur_addr <= d_addr; cur_rem <= {14'd0, d_len}; d_pend <= 1'b0; cur_keep <= 1'b0; cur_fill <= 1'b0;
                    cur_p8 <= cur_pm; end            // (v49_RS RS3) pm: A's second range pops from stream byte 8
                else begin cur_v <= 1'b0; cur_p8 <= 1'b0; cur_pm <= 1'b0; cur_drop <= 1'b0; end
            end
            if (bgx_t) begin
                aw_pend <= 1'b1;
                aw20    <= {bgx_a[19:4], 4'd0};
                awlen   <= 8'(bg_nb - 9'd1);
            end
            if (start_burst || wq_cnt != 2'd0) w_idl <= '0; else if (w_idl != 4'd15) w_idl <= w_idl + 4'd1;
            if (bgx_t) bg_last <= 1'b1; else if (start_burst) bg_last <= 1'b0;
            if (start_burst || bgx_t) wq_wp <= ~wq_wp;
            if (wfire && hk) pk <= ((bwleft == 9'd1) && wq_cl[wq_rp]) ? 8'd0 : (pk + 8'(beat_n));
            if (dfire) begin wq_rp <= ~wq_rp; wfirst <= 1'b1; end   // (v49_RS RS3) a one-'beat' drop
            if (wfire) begin
                if (bwleft == 9'd1) begin wq_rp <= ~wq_rp; wfirst <= 1'b1; end
                else begin
                    wfirst  <= 1'b0;
                    bb_left <= bleft - 13'(beat_n);
                    bw_left <= bwleft - 9'd1;
                end
            end
            wq_cnt <= wq_cnt + ((start_burst || bgx_t) ? 2'd1 : 2'd0) - (((wfire && bwleft == 9'd1) || dfire) ? 2'd1 : 2'd0);
            outst <= outst + ((wfire && wlast) ? 8'd1 : 8'd0) - (bvalid ? 8'd1 : 8'd0);
        end
    end
    assign awvalid = aw_pend;
    wire   [AW:0] isp = (AW+1)'(CAPB) - f_used;                       // (v49DS) saturates at 255 (CAPB 256)
    assign in_space = (isp > (AW+1)'(255)) ? 8'd255 : 8'(isp);
    assign idle = !cur_v && cq_empty && !aw_pend && (wq_cnt == '0) && (outst == '0) && (f_used == '0) && !ex_v;
    assign bg_inq = ((wq_cnt != 2'd0) && wq_bg[wq_rp]) || ((wq_cnt == 2'd2) && wq_bg[~wq_rp]);   // (v56_CY)
endmodule

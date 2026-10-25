// npu_rstream - one ordered read stream: command queue {READ addr,len | FILL val,len},
// burst proposals for the arbiter, and a byte FIFO that receives the data in command order.
// Every proposed burst reserves 16 bytes of FIFO space per beat, so returning beats can
// always be accepted (RREADY is tied high) and streams never block each other.
module npu_rstream #(
    parameter int CAPB = 128,
    parameter int MAXB = 8,
    parameter int MAXL = 8,          // long mode: burst beats (beats are accepted only with room)
    parameter int CQD  = 4,
    parameter int W2   = 0,
    parameter int R2   = 0
) (
    input  logic                  clk,
    input  logic                  rst_n,
    input  logic                  clr,
    input  logic                  long_en,     // long bursts allowed (nothing else needs the R channel)
    input  logic                  small_en,    // (v50_MB) bursts sized to the free space, proposed once 4 beats are free
    input  logic                  lcap,        // (v34P) long bursts of at most 32 beats
    output logic                  occ64,       // (v34P) >= 64 bytes held or reserved
    input  logic                  b10,         // (v35R) short bursts of up to 10 beats
    output logic                  prop_long,
    output logic                  prop_fit,    // (v50_RS BGP) the proposed burst fits the FIFO (its beats never wait)
    output logic                  in_rdy,      // room for one more beat
    input  logic                  cmd_valid,
    output logic                  cmd_ready,
    input  logic                  cmd_fill,
    input  logic [31:0]           cmd_addr,
    input  logic [19:0]           cmd_len,
    input  logic [7:0]            cmd_val,
    output logic [$clog2(CAPB):0] out_avail,
    output logic [127:0]          out_data,
    output logic [127:0]          out_data2,
    input  logic [7:0]            peek,
    input  logic [7:0]            pop_n,
    output logic                  prop_valid,
    output logic [31:0]           prop_addr,
    output logic [8:0]            prop_beats,
    output logic [3:0]            prop_lo,
    output logic [12:0]           prop_bytes,
    input  logic                  grant,
    input  logic                  beat_valid,
    input  logic [127:0]          beat_data,
    input  logic [3:0]            beat_lo,
    input  logic [4:0]            beat_n,
    output logic                  busy,
    output logic                  pend,        // (v28R) bytes still to propose (queued or current command)
    // (v55AD) external pushes into the byte FIFO instead of the AXI beats (x_on: the stream has no commands)
    input  logic                  x_on,
    input  logic                  x_valid,
    input  logic [127:0]          x_data,
    input  logic [4:0]            x_n
);
    localparam int AW = $clog2(CAPB);
    localparam int QW = $clog2(CQD);

    // ---- command queue ----
    // addresses are DRAM offsets (20 bits; the AXI address adds the fixed 0x804 upper bits)
    logic [48:0] cq_mem [CQD];
    logic [QW:0] cq_cnt;
    logic [QW-1:0] cq_rp, cq_wp;
    wire cq_push = cmd_valid && cmd_ready;
    wire cq_empty = (cq_cnt == '0);
    assign cmd_ready = (cq_cnt != (QW+1)'(CQD));

    // (v26A: queue entries written through gated clocks)
    for (genvar i = 0; i < CQD; i++) begin : g_cq
        wire g = clk | ~(cq_push && cq_wp == QW'(i));
        logic [48:0] e;
        always_ff @(posedge g) e <= {cmd_fill, cmd_val, cmd_len, cmd_addr[19:0]};
        assign cq_mem[i] = e;
    end

    // ---- current command ----
    logic        cur_v, cur_fill;
    logic [19:0] cur_addr;
    logic [19:0] cur_rem;
    logic [7:0]  cur_val;
    logic [AW+4:0] resv;                // bytes of granted bursts still in flight
    wire cur_done = !cur_v || (cur_rem == '0);
    wire take     = cur_done && !cq_empty;

    // ---- byte FIFO ----
    logic         b_in_valid, b_in_ready;
    logic [127:0] b_in_data;
    logic [3:0]   b_in_lo;
    logic [4:0]   b_in_n;
    logic [AW:0] b_used;
    if (W2 != 0) begin : g_b32
        // (v26A: the two-window FIFO with 32-byte slots)
        npu_bsf32 #(.CAPB(CAPB)) u_bsf (
            .clk(clk), .rst_n(rst_n), .clr(clr),
            .in_valid(b_in_valid), .in_ready(b_in_ready), .in_data(b_in_data),
            .in_lo(b_in_lo), .in_n(b_in_n),
            .out_avail(out_avail), .out_used(b_used), .out_data(out_data), .out_data2(out_data2), .peek(peek), .pop_n(pop_n));
    end else begin : g_b16
        npu_bsf #(.CAPB(CAPB), .W2(W2), .R2(R2)) u_bsf (
            .clk(clk), .rst_n(rst_n), .clr(clr),
            .in_valid(b_in_valid), .in_ready(b_in_ready), .in_data(b_in_data),
            .in_lo(b_in_lo), .in_n(b_in_n),
            .out_avail(out_avail), .out_used(b_used), .out_data(out_data), .out_data2(out_data2), .peek(peek), .out_rot(4'd0), .pop_n(pop_n));
    end

    wire [4:0] fill_n  = (cur_rem >= 20'd16) ? 5'd16 : cur_rem[4:0];
    wire       fill_go = cur_v && cur_fill && (cur_rem != '0) && (resv == '0) && b_in_ready && !beat_valid;
    assign in_rdy = b_in_ready;

    always_comb begin
        b_in_valid = beat_valid | fill_go;
        b_in_data  = beat_data;   // (v58_AR fv) no fill streams: b_in_valid == beat_valid unless x_on
        b_in_lo    = beat_valid ? beat_lo   : 4'd0;
        b_in_n     = beat_valid ? beat_n    : fill_n;
        if (x_on) begin b_in_valid = x_valid; b_in_data = x_data; b_in_lo = 4'd0; b_in_n = x_n; end   // (v55AD)
    end

    // ---- burst proposal ----
    logic signed [AW+5:0] freeb;
    logic [8:0]   fbeats, tot, to4k, nb;
    wire  [8:0]   mb = long_en ? (lcap ? 9'd32 : 9'(MAXL)) : b10 ? 9'd10 : 9'(MAXB);
    assign occ64 = (freeb <= $signed((AW+6)'(CAPB - 64)));
    logic [20:0]  span;
    always_comb begin
        freeb  = $signed((AW+6)'(CAPB)) - $signed((AW+6)'(b_used)) - $signed((AW+6)'(resv));
        fbeats = (freeb < 0) ? 9'd0 : 9'(freeb >>> 4);
        span   = 21'(cur_addr[3:0]) + 21'(cur_rem);
        tot    = (span >= 21'({12'd0, mb} << 4)) ? mb : 9'((span + 21'd15) >> 4);
        if (small_en && (fbeats < tot)) tot = fbeats;          // (v50_MB small_en: never more than the free space)
        to4k   = 9'd256 - {1'b0, cur_addr[11:4]};           // beats to the 4KB boundary (>= 1)
        nb     = tot;
        if (to4k < nb)   nb = to4k;
        // propose a burst only once all of it fits: no fragments sized by just-freed space.
        // Long mode: a burst larger than half the FIFO once half of it is free (its beats then wait
        // for room on the R channel).
        prop_valid = cur_v && !cur_fill && (cur_rem != '0) && (nb != '0) && (!small_en || (fbeats >= 9'd4)) &&
                     ((fbeats >= nb) || (long_en && (fbeats >= 9'(CAPB / 32))));
        prop_long  = long_en;
        prop_fit   = (fbeats >= nb);
        prop_addr  = {12'h804, cur_addr[19:4], 4'd0};
        prop_beats = nb;
        prop_lo    = cur_addr[3:0];
        if (({12'd0, nb} << 4) - 21'(cur_addr[3:0]) >= 21'(cur_rem))
            prop_bytes = 13'(cur_rem);
        else
            prop_bytes = 13'(({12'd0, nb} << 4) - 21'(cur_addr[3:0]));
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cq_cnt <= '0; cq_rp <= '0; cq_wp <= '0;
            cur_v <= 1'b0; cur_fill <= 1'b0; cur_addr <= '0; cur_rem <= '0; cur_val <= '0;
            resv <= '0;
        end else if (clr) begin
            cq_cnt <= '0; cq_rp <= '0; cq_wp <= '0;
            cur_v <= 1'b0; cur_rem <= '0; resv <= '0;
        end else begin
            if (cq_push) cq_wp <= cq_wp + 1'b1;
            if (take)    cq_rp <= cq_rp + 1'b1;
            cq_cnt <= cq_cnt + (QW+1)'(cq_push) - (QW+1)'(take);
            if (take) begin
                cur_v    <= 1'b1;
                cur_fill <= cq_mem[cq_rp][48];
                cur_val  <= cq_mem[cq_rp][47:40];
                cur_rem  <= cq_mem[cq_rp][39:20];
                cur_addr <= cq_mem[cq_rp][19:0];
            end else begin
                if (cur_done) cur_v <= 1'b0;
                if (fill_go) cur_rem <= cur_rem - 20'(fill_n);
                if (grant) begin
                    cur_rem  <= cur_rem - 20'(prop_bytes);
                    cur_addr <= {cur_addr[19:4], 4'd0} + {7'd0, nb, 4'd0};
                end
            end
            resv <= resv + (grant ? (AW+5)'({nb, 4'd0}) : '0) - (beat_valid ? (AW+5)'(16) : '0);
        end
    end

    assign busy = cur_v || !cq_empty || (resv != '0);
    assign pend = !cq_empty || (cur_v && (cur_rem != '0));
endmodule

// npu_rdma - NSTR ordered read streams sharing the AXI read channel (one burst at a time at
// the slave; the next AR is latched while the current burst streams).
module npu_rdma #(
    parameter int NSTR  = 2,
    parameter int CAPB0 = 256,       // stream 0 FIFO bytes (two full bursts in flight)
    parameter int CAPB  = 128,       // other streams
    parameter int MAXB0 = 8,         // stream 0 burst beats
    parameter int MAXB  = 4,         // other streams (two bursts fit their FIFO)
    parameter int MAXL0 = 8,         // stream 0 burst beats in long mode
    parameter int CQD   = 4
) (
    input  logic                    clk,
    input  logic                    rst_n,
    input  logic                    clr,
    input  logic                    clr1,           // clear stream 1 (v26A: at op start, its FIFO head realigns)
    // (v34P) input gate: while gate_on, a stream-0 burst ending above byte g_lo + wmo is held; wmo (region bytes
    // below it are known written) moves when a poll of the region word at the held burst's end finds it
    // no longer SENT (every byte below it is then written). Bursts that end above the
    // region's end wait for GO (gate_on low).
    input  logic                    gate_on,
    input  logic                    gate_seq,       // a gated sequence is active (else wmo restarts at 0)
    input  logic [19:0]             g_lo,
    input  logic [9:0]              g_len,
    input  logic                    long0,          // stream 0 may use long bursts (stream 1 idle)
    input  logic                    long0e,         // (v28R) ... also once stream 1 has nothing left to propose
    input  logic                    sm1,            // (v50_MB) stream 1: small_en (OP_PIL)
    input  logic                    long0f,         // (v34P) ... capped at 32 beats, while stream 1 holds >= 64 bytes
    input  logic                    b10_0,          // (v35R) stream 0: short bursts of up to 10 beats
    input  logic [NSTR-1:0]         cmd_valid,
    output logic [NSTR-1:0]         cmd_ready,
    input  logic [NSTR-1:0]         cmd_fill,
    input  logic [NSTR*32-1:0]      cmd_addr,
    input  logic [NSTR*20-1:0]      cmd_len,
    input  logic [NSTR*8-1:0]       cmd_val,
    output logic [NSTR*8-1:0]       out_avail,      // bytes readable per stream (saturated at 255)
    output logic [NSTR*128-1:0]     out_data,
    output logic [127:0]            out_data2,      // stream 0: the 16 bytes after its out_data
    input  logic [NSTR*8-1:0]       peek,
    input  logic [NSTR*8-1:0]       pop_n,
    output logic                    idle,
    // (v50_RS BGP) background PAD copy: a read burst latched only when no stream proposes one (bg_ok); its beats
    // go to the write DMA (bg_beat) and are accepted when it writes them (bg_rdy)
    input  logic                    bg_go,
    input  logic [19:0]             bg_addr,
    input  logic [3:0]              bg_beats,
    output logic                    bg_ok,
    output logic                    bg_beat,
    input  logic                    bg_rdy,
    // (v55AD) stream 1's byte FIFO as the pack tee's (npu_pkw): pushes while x_on, x_rdy = room for 16 bytes
    input  logic                    x_on,
    input  logic                    x_valid,
    input  logic [127:0]            x_data,
    input  logic [4:0]              x_n,
    output logic                    x_rdy,
    output logic [31:0]             araddr,
    output logic [7:0]              arlen,
    output logic                    arvalid,
    input  logic                    arready,
    input  logic [127:0]            rdata,
    input  logic                    rlast,
    input  logic                    rvalid,
    output logic                    rready
);
    localparam int SI = (NSTR > 1) ? $clog2(NSTR) : 1;

    logic [NSTR-1:0]      p_fit;              // (v50_RS BGP)
    logic [NSTR-1:0]      p_valid, p_vg, s_occ, s_grant, s_beat, s_busy, p_long, s_rdy, s_pend;
    logic [NSTR*32-1:0]   p_addr;
    logic [NSTR*9-1:0]    p_beats;
    logic [NSTR*4-1:0]    p_lo;
    logic [NSTR*13-1:0]   p_bytes;
    logic [3:0]           bt_lo;
    logic [4:0]           bt_n;

    for (genvar s = 0; s < NSTR; s++) begin : g_s
        localparam int CS = (s == 0) ? CAPB0 : CAPB;
        localparam int MS = (s == 0) ? MAXB0 : MAXB;
        localparam int ML = (s == 0) ? MAXL0 : MAXB;
        localparam int AS = $clog2(CS);
        logic [AS:0] av;
        if (AS >= 8) begin : g_sat
            assign out_avail[s*8 +: 8] = (av[AS:8] != '0) ? 8'd255 : av[7:0];
        end else begin : g_nsat
            assign out_avail[s*8 +: 8] = 8'(av);
        end
        logic [127:0] od2;
        if (s == 0) begin : g_o2
            assign out_data2 = od2;
        end
        npu_rstream #(.CAPB(CS), .MAXB(MS), .MAXL(ML), .CQD((s == 1) ? 2 : CQD), .W2((s == 0) ? 1 : 0), .R2((s == 1) ? 1 : 0)) u_st (
            .clk(clk), .rst_n(rst_n), .clr((s == 1) ? (clr || clr1) : clr),
            .long_en((s == 0) ? (long0 || (long0e && !s_pend[NSTR-1]) || (long0f && (!s_pend[NSTR-1] || s_occ[NSTR-1]))) : 1'b0),
            .b10((s == 0) ? b10_0 : 1'b0), .small_en((s == 1) ? sm1 : 1'b0), .lcap((s == 0) && long0f), .occ64(s_occ[s]), .prop_long(p_long[s]), .prop_fit(p_fit[s]), .in_rdy(s_rdy[s]),
            .cmd_valid(cmd_valid[s]), .cmd_ready(cmd_ready[s]), .cmd_fill(cmd_fill[s]),
            .cmd_addr(cmd_addr[s*32 +: 32]), .cmd_len(cmd_len[s*20 +: 20]), .cmd_val(cmd_val[s*8 +: 8]),
            .out_avail(av), .out_data(out_data[s*128 +: 128]), .out_data2(od2),
            .peek(peek[s*8 +: 8]), .pop_n(pop_n[s*8 +: 8]),
            .prop_valid(p_valid[s]), .prop_addr(p_addr[s*32 +: 32]), .prop_beats(p_beats[s*9 +: 9]),
            .prop_lo(p_lo[s*4 +: 4]), .prop_bytes(p_bytes[s*13 +: 13]),
            .grant(s_grant[s]),
            .beat_valid(s_beat[s]), .beat_data(rdata), .beat_lo(bt_lo), .beat_n(bt_n),
            .busy(s_busy[s]), .pend(s_pend[s]),
            .x_on((s == 1) && x_on), .x_valid(x_valid), .x_data(x_data), .x_n(x_n));   // (v55AD)
    end
    assign x_rdy = s_rdy[NSTR-1];   // (v55AD)

    // ---- R queue (issue order) ----
    logic [SI-1:0] rq_s     [4];
    logic          rq_long  [4];
    logic          rq_poll  [4];                 // (v34P) a gate poll (one beat, not a stream's)
    logic          rq_bg    [4];                 // (v50_RS BGP) a background PAD copy burst
    logic          rq_nf    [4];                 // (v50_RS BGP) a long burst that did not fit its FIFO
    logic [3:0]    rq_lo    [4];
    logic [12:0]   rq_bytes [4];
    logic [2:0]    rq_cnt;
    logic [1:0]    rq_rp, rq_wp;
    logic          rb_first;
    logic [12:0]   rb_left;

    // ---- arbitration ----
    logic          ar_pend;
    logic [19:0]   ar20;
    logic [SI-1:0] rr;
    logic          pick_v;
    logic [SI-1:0] pick_s;
    always_comb begin
        pick_v = 1'b0; pick_s = '0;
        for (int k = 0; k < NSTR; k++) begin
            if (!pick_v) begin
                if (p_vg[(32'(rr) + k) % NSTR]) begin
                    pick_v = 1'b1;
                    pick_s = SI'((32'(rr) + k) % NSTR);
                end
            end
        end
    end
    // ---- (v34P) input gate ----
    localparam logic [31:0] SENT = 32'hA5A5A5A5;
    logic [10:0] wmo;                                // region bytes [0, wmo) are written
    logic [10:0] po;                                 // region offset of the poll in flight
    logic [3:0]  ptmr;
    logic        pout;
    wire  [19:0] ge  = {p_addr[19:4] + {7'd0, p_beats[8:0]}, 4'd0};          // stream 0 burst end
    wire  signed [20:0] re = $signed({1'b0, ge}) - $signed({1'b0, g_lo});    // ... as a region offset
    wire         gblk = gate_on && (re > $signed({10'd0, wmo}));
    wire         preq = gblk && (re < $signed({11'd0, g_len})) && (ptmr == 4'd0) && !pout;
    always_comb begin p_vg = p_valid; p_vg[0] = p_valid[0] && !gblk; end
    wire pgo   = !ar_pend && preq && (rq_cnt < 3'd3);
    wire latch = !ar_pend && pick_v && (rq_cnt < 3'd3) && !preq;
    // (v50_RS BGP) nothing else to issue, and no long burst that did not fit its FIFO queued: its beats wait for its
    // consumer, which may wait for the write channel, which a copy burst holds until its own read beats come
    logic [2:0] rq_nl;                         // such bursts in the R queue
    assign bg_ok = !ar_pend && !pick_v && (rq_cnt < 3'd3) && !preq && (rq_nl == 3'd0);
    always_comb for (int s = 0; s < NSTR; s++) s_grant[s] = latch && (pick_s == SI'(s));

    // ---- R routing ----
    wire         rq_hv  = (rq_cnt != '0);
    // a long burst's beats are accepted only when its stream has room for them
    assign rready = !rq_hv || (rq_bg[rq_rp] ? bg_rdy : (!rq_long[rq_rp] || s_rdy[rq_s[rq_rp]]));
    assign bg_beat = rvalid && rq_hv && rq_bg[rq_rp];
    wire         rfire  = rvalid && rready;
    wire [12:0]  left   = rb_first ? rq_bytes[rq_rp] : rb_left;
    always_comb begin
        bt_lo = rb_first ? rq_lo[rq_rp] : 4'd0;
        bt_n  = ((13'd16 - 13'(bt_lo)) >= left) ? 5'(left) : 5'(5'd16 - {1'b0, bt_lo});
        for (int s = 0; s < NSTR; s++) s_beat[s] = rfire && rq_hv && (rq_s[rq_rp] == SI'(s)) && !rq_poll[rq_rp] && !rq_bg[rq_rp];
    end

    always_ff @(posedge clk) begin
        if (latch || pgo) begin
            rq_s[rq_wp]     <= pick_s;
            rq_long[rq_wp]  <= p_long[pick_s] && !pgo;
            rq_lo[rq_wp]    <= p_lo[pick_s*4 +: 4];
            rq_bytes[rq_wp] <= p_bytes[pick_s*13 +: 13];
            rq_poll[rq_wp]  <= pgo;
            rq_bg[rq_wp]    <= 1'b0;
            rq_nf[rq_wp]    <= p_long[pick_s] && !p_fit[pick_s] && !pgo;
        end else if (bg_go) begin                 // (v50_RS BGP)
            rq_s[rq_wp] <= '0; rq_long[rq_wp] <= 1'b0; rq_poll[rq_wp] <= 1'b0; rq_bg[rq_wp] <= 1'b1; rq_nf[rq_wp] <= 1'b0;
            rq_lo[rq_wp] <= 4'd0; rq_bytes[rq_wp] <= 13'd0;
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ar_pend <= 1'b0; ar20 <= '0; arlen <= '0; rr <= '0;
            rq_cnt <= '0; rq_rp <= '0; rq_wp <= '0; rb_first <= 1'b1; rb_left <= '0; rq_nl <= '0;
        end else if (clr) begin
            ar_pend <= 1'b0; rq_nl <= '0; rq_cnt <= '0; rq_rp <= '0; rq_wp <= '0; rb_first <= 1'b1; rb_left <= '0;
        end else begin
            if (latch) begin
                ar_pend <= 1'b1;
                ar20    <= p_addr[pick_s*32 +: 20];
                arlen   <= 8'(p_beats[pick_s*9 +: 9] - 9'd1);
                rr      <= (32'(pick_s) == NSTR-1) ? '0 : pick_s + SI'(1);
                rq_wp   <= rq_wp + 1'b1;
            end else if (pgo) begin
                ar_pend <= 1'b1; ar20 <= ge; arlen <= 8'd0; rq_wp <= rq_wp + 1'b1;
            end else if (bg_go) begin                 // (v50_RS BGP) only with bg_ok: no latch / poll this cycle
                ar_pend <= 1'b1; ar20 <= bg_addr; arlen <= {4'd0, bg_beats} - 8'd1; rq_wp <= rq_wp + 1'b1;
            end else if (ar_pend && arready) begin
                ar_pend <= 1'b0;
            end
            if (rfire && rq_hv) begin
                if (rlast) begin
                    rq_rp    <= rq_rp + 1'b1;
                    rb_first <= 1'b1;
                end else begin
                    rb_first <= 1'b0;
                    rb_left  <= left - 13'(bt_n);
                end
            end
            rq_nl  <= rq_nl + ((latch && p_long[pick_s] && !p_fit[pick_s]) ? 3'd1 : 3'd0) -
                      ((rfire && rq_hv && rlast && rq_nf[rq_rp]) ? 3'd1 : 3'd0);   // (v50_RS BGP)
            rq_cnt <= rq_cnt + ((latch || pgo || bg_go) ? 3'd1 : 3'd0) - ((rfire && rq_hv && rlast) ? 3'd1 : 3'd0);
        end
    end
    // poll: one in flight, >= 16 cycles apart; if the polled word (region offset po, the held burst's end) is no
    // longer SENT, every region byte below it is written: wmo moves to po (po > wmo when issued, only polls move wmo)
    wire         pbeat = rfire && rq_hv && rq_poll[rq_rp];
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin wmo <= '0; po <= '0; ptmr <= '0; pout <= 1'b0; end
        else begin
            if (pgo) begin po <= re[10:0]; pout <= 1'b1; ptmr <= 4'd15; end
            else if (ptmr != 4'd0) ptmr <= ptmr - 4'd1;
            if (pbeat) pout <= 1'b0;
            if (!gate_seq) wmo <= '0;
            else if (pbeat && (rdata[31:0] != SENT)) wmo <= po;
        end
    end
    assign arvalid = ar_pend;
    assign araddr  = {12'h804, ar20};

    always_comb begin
        idle = !ar_pend && (rq_cnt == '0);
        for (int s = 0; s < NSTR; s++) idle = idle && !s_busy[s];
    end
endmodule

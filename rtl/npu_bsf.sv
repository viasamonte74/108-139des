// npu_bsf - byte stream FIFO.
// Push: up to 16 bytes per cycle taken from in_data[in_lo .. in_lo+in_n-1] (in_lo+in_n <= 16).
// Pop:  out_data holds the next 16 bytes (head-aligned, bytes past out_avail are don't-care);
//       pop_n bytes are consumed this cycle (pop_n <= out_avail; pop_n <= 16, or <= 32 with W2).
// Concatenating unaligned DRAM ranges and constant fills into one contiguous stream is
// what lets every engine see packed operands.
// A push is staged one cycle and written through per-byte gated clocks (gclk = clk | ~en_q, the
// enable decoded from registered tail/count): no per-bit enable muxes. out_avail counts bytes that
// are readable; out_used also counts the staged push (use it for free space).
module npu_bsf #(
    parameter int CAPB = 64,                 // capacity in bytes (power of 2, >= 32)
    parameter int W2   = 0,                  // 1: also out_data2 = the 16 bytes after out_data
    parameter int R2   = 0,                  // 1 (v26A, stream 1): only out_data[15:0] follow the head; bytes
                                             // 2..15 are the byte lanes as stored (in order when head % 16 == 0)
    parameter int EXACT = 0                  // (v46MB) 1: a push is accepted whenever its in_n bytes fit
) (
    input  logic                  clk,
    input  logic                  rst_n,
    input  logic                  clr,
    input  logic                  in_valid,
    output logic                  in_ready,
    input  logic [127:0]          in_data,
    input  logic [3:0]            in_lo,
    input  logic [4:0]            in_n,
    output logic [$clog2(CAPB):0] out_avail,
    output logic [$clog2(CAPB):0] out_used,
    output logic [127:0]          out_data,
    output logic [127:0]          out_data2,
    input  logic [7:0]            peek,        // out_data starts this many bytes past the head
    input  logic [3:0]            out_rot,     // out_data rotated left by this many bytes (lane out_rot = head)
    input  logic [7:0]            pop_n
);
    localparam int AW = $clog2(CAPB);
    localparam int NS = CAPB / 16;
    localparam int SW = (NS > 1) ? $clog2(NS) : 1;
    localparam bit NP2 = (CAPB & (CAPB - 1)) != 0;   // (v49DS) a capacity that is no power of 2 (multiple of 16)

    logic [CAPB*8-1:0] mem;
    logic [AW-1:0]     head, tail;
    logic [AW:0]       count;

    wire       push = in_valid & in_ready;
    wire [4:0] pn   = push ? in_n : 5'd0;

    // staged push
    logic          st_v;
    logic [4:0]    st_n;
    logic [AW-1:0] st_tail;
    logic [127:0]  st_d;
    wire  [AW:0]   used = count + (AW+1)'(st_v ? st_n : 5'd0);

    assign in_ready  = ((AW+1)'(CAPB) - used) >= (AW+1)'(EXACT ? {1'b0, in_n} : 6'd16);
    assign out_avail = count;
    assign out_used  = used;

    // rotate the pushed beat right by r bytes so that byte (tail mod 16) lines up with in_lo
    wire [3:0] r = in_lo - tail[3:0];
    logic [127:0] s0, s1, s2, rot;
    // (v43H) the four levels as h_mx (a22o_2 cells, shared inverted selects)
    h_mx #(128) u_w0 (.a0(in_data), .a1({in_data[7:0],  in_data[127:8]}), .s(r[0]), .x(s0));
    h_mx #(128) u_w1 (.a0(s0),      .a1({s0[15:0],      s0[127:16]}),     .s(r[1]), .x(s1));
    h_mx #(128) u_w2 (.a0(s1),      .a1({s1[31:0],      s1[127:32]}),     .s(r[2]), .x(s2));
    h_mx #(128) u_w3 (.a0(s2),      .a1({s2[63:0],      s2[127:64]}),     .s(r[3]), .x(rot));

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) st_v <= 1'b0;
        else        st_v <= push && (in_n != 5'd0) && !clr;
    end
    // the staged push covers lanes [t, t+n) of slot tail/16 and, past 16, lanes [0, t+n-16) of
    // the next slot: decoded at push time into two lane masks and a slot index
    wire [4:0]  p_end = {1'b0, tail[3:0]} + in_n;
    logic [15:0] ma_d, mb_d, st_ma, st_mb;
    logic [SW-1:0] st_s0;
    always_comb for (int l = 0; l < 16; l++) begin
        ma_d[l] = (4'(l) >= tail[3:0]) && (5'(l) < p_end);
        mb_d[l] = (5'(l) + 5'd16) < p_end;
    end
    always_ff @(posedge clk) begin
        st_n <= in_n; st_tail <= tail; st_d <= rot;
        st_ma <= ma_d; st_mb <= mb_d; st_s0 <= tail[AW-1:4];
    end
    for (genvar q = 0; q < NS; q++) begin : g_slot
        wire sa = st_v && (st_s0 == SW'(q));
        wire sb = st_v && (st_s0 == SW'((q == 0) ? (NS - 1) : (q - 1)));
        for (genvar l = 0; l < 16; l++) begin : g_lane
            wire gclk = clk | ~((sa && st_ma[l]) || (sb && st_mb[l]));
            logic [7:0] b;
            always_ff @(posedge gclk) b <= st_d[l*8 +: 8];
            assign mem[(q*16 + l)*8 +: 8] = b;
        end
    end

    // peek: byte lane q (address mod 16) is read from slot sl0 when q >= ph[3:0], else from the
    // next slot; then the 16 lanes are rotated right by ph[3:0] bytes
    wire [AW:0]   phx = {1'b0, head} + (AW+1)'(peek);
    wire [AW-1:0] ph  = (NP2 && (phx >= (AW+1)'(CAPB))) ? AW'(phx - (AW+1)'(CAPB)) : AW'(phx);
    wire [SW-1:0] sl0 = ph[AW-1:4];
    wire [SW-1:0] sl1 = (NP2 && (sl0 == SW'(NS - 1))) ? '0 : (sl0 + SW'(1));
    // (one NS:1 slot select per byte lane, written out per lane so that synthesis sees a plain mux)
    logic [127:0] lanes, f0, f1, f2, lanes2;
    // one-hot slot selects: sl0 decoded once, sl1 = sl0 + 1 is its rotation (v26A)
    logic [NS-1:0] oh0, oh1;
    always_comb begin oh0 = '0; oh0[sl0] = 1'b1; end
    if (NS > 1) begin : g_oh1
        assign oh1 = {oh0[NS-2:0], oh0[NS-1]};
    end else begin : g_oh1s
        assign oh1 = oh0;
    end
    for (genvar q = 0; q < 16; q++) begin : g_rd
        logic [7:0] lm [NS];
        for (genvar t = 0; t < NS; t++) begin : g_t
            assign lm[t] = mem[(t*16 + q)*8 +: 8];
        end
        wire [SW-1:0] ls = (4'(q) >= ph[3:0]) ? sl0 : sl1;
        wire c = (4'(q) >= ph[3:0]);
        logic [7:0] v;
        always_comb begin
            v = '0;
            for (int t = 0; t < NS; t++) v = v | (lm[t] & {8{c ? oh0[t] : oh1[t]}});
        end
        assign lanes[q*8 +: 8] = v;
        // W2: the same lane one slot further
        if (W2 != 0) begin : g_l2
            wire [SW-1:0] ls2 = (4'(q) >= ph[3:0]) ? sl1 : (sl1 + SW'(1));
            assign lanes2[q*8 +: 8] = lm[ls2];
        end else begin : g_nl2
            assign lanes2[q*8 +: 8] = 8'd0;
        end
    end
    wire [3:0] rr = ph[3:0] - out_rot;                    // net right rotation
    if (R2 == 0) begin : g_rot
        h_mx #(128) u_r0 (.a0(lanes), .a1({lanes[7:0],  lanes[127:8]}), .s(rr[0]), .x(f0));
        h_mx #(128) u_r1 (.a0(f0),    .a1({f0[15:0],    f0[127:16]}),   .s(rr[1]), .x(f1));
        h_mx #(128) u_r2 (.a0(f1),    .a1({f1[31:0],    f1[127:32]}),   .s(rr[2]), .x(f2));
        h_mx #(128) u_r3 (.a0(f2),    .a1({f2[63:0],    f2[127:64]}),   .s(rr[3]), .x(out_data));
    end else begin : g_r2
        logic [7:0] lb [16];
        for (genvar q = 0; q < 16; q++) begin : g_lb
            assign lb[q] = lanes[q*8 +: 8];
        end
        assign f0 = '0; assign f1 = '0; assign f2 = '0;
        assign out_data = {lanes[127:16], lb[4'(rr + 4'd1)], lb[rr]};
    end
    if (W2 != 0) begin : g_w2
        logic [127:0] g0, g1, g2;
        h_mx #(128) u_g0 (.a0(lanes2), .a1({lanes2[7:0], lanes2[127:8]}), .s(rr[0]), .x(g0));
        h_mx #(128) u_g1 (.a0(g0),     .a1({g0[15:0],    g0[127:16]}),    .s(rr[1]), .x(g1));
        h_mx #(128) u_g2 (.a0(g1),     .a1({g1[31:0],    g1[127:32]}),    .s(rr[2]), .x(g2));
        h_mx #(128) u_g3 (.a0(g2),     .a1({g2[63:0],    g2[127:64]}),    .s(rr[3]), .x(out_data2));
    end else begin : g_nw2
        assign out_data2 = '0;
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            head <= '0; tail <= '0; count <= '0;
        end else if (clr) begin
            head <= '0; tail <= '0; count <= '0;
        end else begin
            head  <= hx[AW-1:0];
            tail  <= tx[AW-1:0];
            count <= count + (AW+1)'(st_v ? st_n : 5'd0) - (AW+1)'(pop_n);
        end
    end
    wire [AW:0] hx0 = {1'b0, head} + (AW+1)'(pop_n);
    wire [AW:0] tx0 = {1'b0, tail} + (AW+1)'(pn);
    wire [AW:0] hx  = (NP2 && (hx0 >= (AW+1)'(CAPB))) ? (hx0 - (AW+1)'(CAPB)) : hx0;   // (v49DS) wrap at CAPB
    wire [AW:0] tx  = (NP2 && (tx0 >= (AW+1)'(CAPB))) ? (tx0 - (AW+1)'(CAPB)) : tx0;
endmodule

// npu_bsf32 (v26A) - npu_bsf with W2 = 1 organised as 32-byte slots: the 32-byte window at the head
// (out_data, out_data2) is one NS:1 slot select per byte lane (32 lanes) and one 32-byte rotation,
// instead of two 16-lane selects (16-byte slots) and two 16-byte rotations. Same interface and
// behaviour as npu_bsf #(.W2(1)) with out_rot = 0.
module npu_bsf32 #(
    parameter int CAPB = 256                 // capacity in bytes (power of 2, >= 64)
) (
    input  logic                  clk,
    input  logic                  rst_n,
    input  logic                  clr,
    input  logic                  in_valid,
    output logic                  in_ready,
    input  logic [127:0]          in_data,
    input  logic [3:0]            in_lo,
    input  logic [4:0]            in_n,
    output logic [$clog2(CAPB):0] out_avail,
    output logic [$clog2(CAPB):0] out_used,
    output logic [127:0]          out_data,
    output logic [127:0]          out_data2,
    input  logic [7:0]            peek,
    input  logic [7:0]            pop_n
);
    localparam int AW = $clog2(CAPB);
    localparam int NS = CAPB / 32;
    localparam int SW = $clog2(NS);

    logic [CAPB*8-1:0] mem;
    logic [AW-1:0]     head, tail;
    logic [AW:0]       count;

    wire       push = in_valid & in_ready;
    wire [4:0] pn   = push ? in_n : 5'd0;

    // staged push
    logic          st_v;
    logic [4:0]    st_n;
    logic [127:0]  st_d;
    wire  [AW:0]   used = count + (AW+1)'(st_v ? st_n : 5'd0);

    assign in_ready  = ((AW+1)'(CAPB) - used) >= (AW+1)'(16);
    assign out_avail = count;
    assign out_used  = used;

    // rotate the pushed beat right by r bytes so that byte (tail mod 16) lines up with in_lo
    wire [3:0] r = in_lo - tail[3:0];
    logic [127:0] s0, s1, s2, rot;
    // (v43H) the four levels as h_mx (a22o_2 cells, shared inverted selects)
    h_mx #(128) u_w0 (.a0(in_data), .a1({in_data[7:0],  in_data[127:8]}), .s(r[0]), .x(s0));
    h_mx #(128) u_w1 (.a0(s0),      .a1({s0[15:0],      s0[127:16]}),     .s(r[1]), .x(s1));
    h_mx #(128) u_w2 (.a0(s1),      .a1({s1[31:0],      s1[127:32]}),     .s(r[2]), .x(s2));
    h_mx #(128) u_w3 (.a0(s2),      .a1({s2[63:0],      s2[127:64]}),     .s(r[3]), .x(rot));

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) st_v <= 1'b0;
        else        st_v <= push && (in_n != 5'd0) && !clr;
    end
    // the staged push covers lanes [t, t+n) of slot tail/32 and, past 32, lanes [0, t+n-32) of
    // the next slot (lane l takes byte l % 16 of the rotated beat)
    wire [5:0]  p_end = {1'b0, tail[4:0]} + {1'b0, in_n};
    logic [31:0] ma_d, mb_d, st_ma, st_mb;
    logic [SW-1:0] st_s0;
    always_comb for (int l = 0; l < 32; l++) begin
        ma_d[l] = (5'(l) >= tail[4:0]) && (6'(l) < p_end);
        mb_d[l] = (6'(l) + 6'd32) < p_end;
    end
    always_ff @(posedge clk) begin
        st_n <= in_n; st_d <= rot;
        st_ma <= ma_d; st_mb <= mb_d; st_s0 <= tail[AW-1:5];
    end
    for (genvar q = 0; q < NS; q++) begin : g_slot
        wire sa = st_v && (st_s0 == SW'(q));
        wire sb = st_v && (st_s0 == SW'(q - 1));
        for (genvar l = 0; l < 32; l++) begin : g_lane
            wire gclk = clk | ~((sa && st_ma[l]) || (sb && st_mb[l]));
            logic [7:0] b;
            always_ff @(posedge gclk) b <= st_d[(l % 16)*8 +: 8];
            assign mem[(q*32 + l)*8 +: 8] = b;
        end
    end

    // peek: byte lane q (address mod 32) is read from slot sl0 when q >= ph[4:0], else from the
    // next slot; then the 32 lanes are rotated right by ph[4:0] bytes
    wire [AW-1:0] ph  = head + AW'(peek);
    wire [SW-1:0] sl0 = ph[AW-1:5];
    wire [SW-1:0] sl1 = sl0 + SW'(1);
    logic [255:0] lanes, f0, f1, f2, f3, win;
    // one-hot slot selects: sl0 decoded once, sl1 = sl0 + 1 is its rotation
    logic [NS-1:0] oh0, oh1;
    always_comb begin oh0 = '0; oh0[sl0] = 1'b1; end
    assign oh1 = {oh0[NS-2:0], oh0[NS-1]};
    for (genvar q = 0; q < 32; q++) begin : g_rd
        wire c = (5'(q) >= ph[4:0]);
        logic [7:0] v;
        always_comb begin
            v = '0;
            for (int t = 0; t < NS; t++) v = v | (mem[(t*32 + q)*8 +: 8] & {8{c ? oh0[t] : oh1[t]}});
        end
        assign lanes[q*8 +: 8] = v;
    end
    wire [4:0] rr = ph[4:0];
    always_comb begin
        f0  = rr[0] ? {lanes[7:0],   lanes[255:8]}   : lanes;
        f1  = rr[1] ? {f0[15:0],     f0[255:16]}     : f0;
        f2  = rr[2] ? {f1[31:0],     f1[255:32]}     : f1;
        f3  = rr[3] ? {f2[63:0],     f2[255:64]}     : f2;
        win = rr[4] ? {f3[127:0],    f3[255:128]}    : f3;
    end
    assign out_data  = win[127:0];
    assign out_data2 = win[255:128];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            head <= '0; tail <= '0; count <= '0;
        end else if (clr) begin
            head <= '0; tail <= '0; count <= '0;
        end else begin
            head  <= head + AW'(pop_n);
            tail  <= tail + AW'(pn);
            count <= count + (AW+1)'(st_v ? st_n : 5'd0) - (AW+1)'(pop_n);
        end
    end
endmodule

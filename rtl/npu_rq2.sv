// npu_rq2 - two npu_rq units (exact TFLM requantisation, any multiplier / shift) on one
//   double-pumped 32 x 32 multiplier. Same function and latency as two npu_rq: out_q{0,1} =
//   (int8) clamp(MultiplyByQuantizedMultiplier(acc, mult, shift) + ooff, amin, amax); out_valid{0,1}
//   four cycles after in_valid{0,1}. (npu_rq's raw output is not provided.)
//   Pipeline: c0 left shift + Booth recoding, registered (t1) | H1: radix-4 Booth partial
//   products + a 3:2 tree to sum / carry vectors, one operand set per clock phase (set 0 in the
//   high phase, captured at the negedge; set 1 in the low phase, captured at the posedge) |
//   set 0: re-registered at t2; both sets: carry-propagate add + RoundingDivideByPOT (t3) |
//   offset, clamp (t4). Only H1 is a half-cycle path.
module npu_rq2 (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        in_valid0,
    input  logic        in_valid1,
    input  logic [31:0] acc0,
    input  logic [31:0] acc1,
    input  logic [31:0] mult0,       // >= 0 (bit 31 is ignored)
    input  logic [31:0] mult1,
    input  logic [7:0]  shift0,      // signed
    input  logic [7:0]  shift1,
    input  logic [8:0]  ooff0,       // per-op constants: read directly, not pipelined
    input  logic [7:0]  amin0,
    input  logic [7:0]  amax0,
    input  logic [8:0]  ooff1,
    input  logic [7:0]  amin1,
    input  logic [7:0]  amax1,
    output logic        out_valid0,
    output logic        out_valid1,
    output logic [7:0]  out_q0,
    output logic [7:0]  out_q1
);
    localparam int W  = 32;
    localparam int PW = 63;                // product bits 0 .. 62 (only [62:31] are kept)
    // Booth rows are added as {~s, low W bits} = row + 2^W: K0 removes those offsets; + the
    // SRDHM nudge 2^30
    function automatic logic [PW-1:0] k0f();
        k0f = '0;
        for (int k = 0; k < 16; k++) k0f = k0f - (PW'(1) << (W + 2 * k));
        k0f = k0f + (PW'(1) << 30);
    endfunction
    localparam logic [PW-1:0] K0 = k0f();
    function automatic logic [2*PW-1:0] csa(input logic [PW-1:0] x, input logic [PW-1:0] y, input logic [PW-1:0] z);
        logic [PW-1:0] s, c;
        s = x ^ y ^ z;
        c = ((x & y) | (x & z) | (y & z)) << 1;
        csa = {c, s};
    endfunction
    // RoundingDivideByPOT (npu_rq stage 3), then the clamp-relevant part: q (10 bits), hi, lo
    function automatic logic [11:0] rdq(input logic [31:0] y, input logic [4:0] r);
        logic [31:0] mask, rem, thr, ysh, z;
        mask = (32'd1 << r) - 32'd1;
        rem  = y & mask;
        thr  = (mask >> 1) + {31'd0, y[31]};
        ysh  = $signed(y) >>> r;
        z    = ysh + ((rem > thr) ? 32'd1 : 32'd0);
        rdq  = {!z[31] && (z[30:9] != 22'd0), z[31] && (z[30:9] != 22'h3fffff), z[9:0]};
    endfunction
    // offset + clamp (npu_rq stage 4)
    function automatic logic [7:0] outq(input logic [11:0] h, input logic [8:0] off,
                                        input logic [7:0] mn, input logic [7:0] mx);
        logic [10:0] zs, s4, c4;
        zs = {h[9], h[9:0]};
        s4 = zs + {{2{off[8]}}, off};
        c4 = s4;
        if ($signed(c4) < $signed({{3{mn[7]}}, mn})) c4 = {{3{mn[7]}}, mn};
        if ($signed(c4) > $signed({{3{mx[7]}}, mx})) c4 = {{3{mx[7]}}, mx};
        if (h[11]) c4 = {{3{mx[7]}}, mx};
        if (h[10]) c4 = {{3{mn[7]}}, mn};
        outq = c4[7:0];
    endfunction

    // ---- valid pipeline ----
    logic [1:0] v1, v2, v3;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin v1 <= '0; v2 <= '0; v3 <= '0; out_valid0 <= 1'b0; out_valid1 <= 1'b0; end
        else begin
            v1 <= {in_valid1, in_valid0}; v2 <= v1; v3 <= v2;
            out_valid0 <= v3[0]; out_valid1 <= v3[1];
        end
    end

    // ---- c0 -> t1: left-shifted operands, Booth digits, right shifts ----
    localparam int NC = 1;                 // operand copies (v28A: one; the H1 paths had > 1 ns to spare)
    localparam int NP = 4;                 // phase copies (v28A: each serves 4 Booth rows' digit selects)
    (* keep *) logic [W-1:0] a0q [NC];
    (* keep *) logic [W-1:0] a1q [NC];
    logic [31:0]  m0q, m1q;                // (v28A: multipliers registered, Booth-recoded in H1 after the
                                           //  phase select: one recoder per unit pair, 34 fewer flops)
    logic [4:0]   r0q, r1q;
    wire  [4:0]   l0 = shift0[7] ? 5'd0 : shift0[4:0];
    wire  [4:0]   l1 = shift1[7] ? 5'd0 : shift1[4:0];
    always_ff @(posedge clk) begin
        for (int c = 0; c < NC; c++) begin a0q[c] <= acc0 << l0; a1q[c] <= acc1 << l1; end
        m0q <= {1'b0, mult0[30:0]};
        m1q <= {1'b0, mult1[30:0]};
        r0q <= shift0[7] ? 5'(8'd0 - shift0) : 5'd0;
        r1q <= shift1[7] ? 5'(8'd0 - shift1) : 5'd0;
    end

    // ---- clock phase (1 in the high phase, 0 in the low phase), NP copies ----
    logic ph [NP];
    for (genvar c = 0; c < NP; c++) begin : g_ph
        logic tp, tn;
        always_ff @(posedge clk or negedge rst_n) if (!rst_n) tp <= 1'b0; else tp <= !tp;
        always_ff @(negedge clk or negedge rst_n) if (!rst_n) tn <= 1'b0; else tn <= tp;
        assign ph[c] = tp ^ tn;
    end
    // ---- H1: partial products of the phase's operand set and the 3:2 tree (18 rows -> 2) ----
    logic [PW-1:0] r0 [18];
    always_comb begin
        logic [W-1:0] av;
        logic         b1, b2, bn;
        logic [2:0]   bt;
        logic [W:0]   mg, pp;
        logic [PW-1:0] bnr;
        bnr = '0;
        av = ph[0] ? a0q[0] : a1q[0];      // (v28A: one operand select shared by the 16 rows)
        for (int k = 0; k < 16; k++) begin
            bt = ph[k * NP / 16] ? {m0q[2*k+1], m0q[2*k], (k == 0) ? 1'b0 : m0q[(k == 0) ? 0 : 2*k-1]}
                                 : {m1q[2*k+1], m1q[2*k], (k == 0) ? 1'b0 : m1q[(k == 0) ? 0 : 2*k-1]};
            b1 = bt[1] ^ bt[0];
            b2 = (bt == 3'b011) || (bt == 3'b100);
            bn = bt[2] && !(bt[1] && bt[0]);
            mg = ({(W+1){b1}} & {av[W-1], av}) | ({(W+1){b2}} & {av, 1'b0});
            pp = mg ^ {(W+1){bn}};
            r0[k] = PW'({~pp[W], pp[W-1:0]}) << (2*k);
            bnr[2*k] = bn;
        end
        r0[16] = bnr;
        r0[17] = K0;
    end
    logic [PW-1:0] hs, hc;
    wire h_s0, h_c0; sky130_fd_sc_hd__fa_1 h_u0 (.A(r0[0][24]), .B(r0[1][24]), .CIN(1'b0), .SUM(h_s0), .COUT(h_c0));
    wire h_s1, h_c1; sky130_fd_sc_hd__fa_1 h_u1 (.A(r0[0][25]), .B(r0[1][25]), .CIN(1'b0), .SUM(h_s1), .COUT(h_c1));
    wire h_s2, h_c2; sky130_fd_sc_hd__fa_1 h_u2 (.A(r0[0][26]), .B(r0[1][26]), .CIN(r0[2][26]), .SUM(h_s2), .COUT(h_c2));
    wire h_s3, h_c3; sky130_fd_sc_hd__fa_1 h_u3 (.A(r0[3][26]), .B(r0[4][26]), .CIN(1'b0), .SUM(h_s3), .COUT(h_c3));
    wire h_s4, h_c4; sky130_fd_sc_hd__fa_1 h_u4 (.A(r0[0][27]), .B(r0[1][27]), .CIN(r0[2][27]), .SUM(h_s4), .COUT(h_c4));
    wire h_s5, h_c5; sky130_fd_sc_hd__fa_1 h_u5 (.A(r0[3][27]), .B(r0[4][27]), .CIN(1'b0), .SUM(h_s5), .COUT(h_c5));
    wire h_s6, h_c6; sky130_fd_sc_hd__fa_1 h_u6 (.A(r0[0][28]), .B(r0[1][28]), .CIN(r0[2][28]), .SUM(h_s6), .COUT(h_c6));
    wire h_s7, h_c7; sky130_fd_sc_hd__fa_1 h_u7 (.A(r0[3][28]), .B(r0[4][28]), .CIN(r0[5][28]), .SUM(h_s7), .COUT(h_c7));
    wire h_s8, h_c8; sky130_fd_sc_hd__fa_1 h_u8 (.A(r0[6][28]), .B(r0[7][28]), .CIN(1'b0), .SUM(h_s8), .COUT(h_c8));
    wire h_s9, h_c9; sky130_fd_sc_hd__fa_1 h_u9 (.A(r0[0][29]), .B(r0[1][29]), .CIN(r0[2][29]), .SUM(h_s9), .COUT(h_c9));
    wire h_s10, h_c10; sky130_fd_sc_hd__fa_1 h_u10 (.A(r0[3][29]), .B(r0[4][29]), .CIN(r0[5][29]), .SUM(h_s10), .COUT(h_c10));
    wire h_s11, h_c11; sky130_fd_sc_hd__fa_1 h_u11 (.A(r0[6][29]), .B(r0[7][29]), .CIN(1'b0), .SUM(h_s11), .COUT(h_c11));
    wire h_s12, h_c12; sky130_fd_sc_hd__fa_1 h_u12 (.A(r0[0][30]), .B(r0[1][30]), .CIN(r0[2][30]), .SUM(h_s12), .COUT(h_c12));
    wire h_s13, h_c13; sky130_fd_sc_hd__fa_1 h_u13 (.A(r0[3][30]), .B(r0[4][30]), .CIN(r0[5][30]), .SUM(h_s13), .COUT(h_c13));
    wire h_s14, h_c14; sky130_fd_sc_hd__fa_1 h_u14 (.A(r0[6][30]), .B(r0[7][30]), .CIN(r0[8][30]), .SUM(h_s14), .COUT(h_c14));
    wire h_s15, h_c15; sky130_fd_sc_hd__fa_1 h_u15 (.A(r0[9][30]), .B(r0[10][30]), .CIN(r0[11][30]), .SUM(h_s15), .COUT(h_c15));
    wire h_s16, h_c16; sky130_fd_sc_hd__fa_1 h_u16 (.A(r0[0][31]), .B(r0[1][31]), .CIN(r0[2][31]), .SUM(h_s16), .COUT(h_c16));
    wire h_s17, h_c17; sky130_fd_sc_hd__fa_1 h_u17 (.A(r0[3][31]), .B(r0[4][31]), .CIN(r0[5][31]), .SUM(h_s17), .COUT(h_c17));
    wire h_s18, h_c18; sky130_fd_sc_hd__fa_1 h_u18 (.A(r0[6][31]), .B(r0[7][31]), .CIN(r0[8][31]), .SUM(h_s18), .COUT(h_c18));
    wire h_s19, h_c19; sky130_fd_sc_hd__fa_1 h_u19 (.A(r0[9][31]), .B(r0[10][31]), .CIN(1'b0), .SUM(h_s19), .COUT(h_c19));
    wire h_s20, h_c20; sky130_fd_sc_hd__fa_1 h_u20 (.A(r0[0][32]), .B(r0[1][32]), .CIN(r0[2][32]), .SUM(h_s20), .COUT(h_c20));
    wire h_s21, h_c21; sky130_fd_sc_hd__fa_1 h_u21 (.A(r0[3][32]), .B(r0[4][32]), .CIN(r0[5][32]), .SUM(h_s21), .COUT(h_c21));
    wire h_s22, h_c22; sky130_fd_sc_hd__fa_1 h_u22 (.A(r0[6][32]), .B(r0[7][32]), .CIN(r0[8][32]), .SUM(h_s22), .COUT(h_c22));
    wire h_s23, h_c23; sky130_fd_sc_hd__fa_1 h_u23 (.A(r0[9][32]), .B(r0[10][32]), .CIN(r0[11][32]), .SUM(h_s23), .COUT(h_c23));
    wire h_s24, h_c24; sky130_fd_sc_hd__fa_1 h_u24 (.A(r0[1][33]), .B(r0[2][33]), .CIN(r0[3][33]), .SUM(h_s24), .COUT(h_c24));
    wire h_s25, h_c25; sky130_fd_sc_hd__fa_1 h_u25 (.A(r0[4][33]), .B(r0[5][33]), .CIN(r0[6][33]), .SUM(h_s25), .COUT(h_c25));
    wire h_s26, h_c26; sky130_fd_sc_hd__fa_1 h_u26 (.A(r0[7][33]), .B(r0[8][33]), .CIN(r0[9][33]), .SUM(h_s26), .COUT(h_c26));
    wire h_s27, h_c27; sky130_fd_sc_hd__fa_1 h_u27 (.A(r0[10][33]), .B(r0[11][33]), .CIN(1'b0), .SUM(h_s27), .COUT(h_c27));
    wire h_s28, h_c28; sky130_fd_sc_hd__fa_1 h_u28 (.A(r0[1][34]), .B(r0[2][34]), .CIN(r0[3][34]), .SUM(h_s28), .COUT(h_c28));
    wire h_s29, h_c29; sky130_fd_sc_hd__fa_1 h_u29 (.A(r0[4][34]), .B(r0[5][34]), .CIN(r0[6][34]), .SUM(h_s29), .COUT(h_c29));
    wire h_s30, h_c30; sky130_fd_sc_hd__fa_1 h_u30 (.A(r0[7][34]), .B(r0[8][34]), .CIN(r0[9][34]), .SUM(h_s30), .COUT(h_c30));
    wire h_s31, h_c31; sky130_fd_sc_hd__fa_1 h_u31 (.A(r0[2][35]), .B(r0[3][35]), .CIN(r0[4][35]), .SUM(h_s31), .COUT(h_c31));
    wire h_s32, h_c32; sky130_fd_sc_hd__fa_1 h_u32 (.A(r0[5][35]), .B(r0[6][35]), .CIN(r0[7][35]), .SUM(h_s32), .COUT(h_c32));
    wire h_s33, h_c33; sky130_fd_sc_hd__fa_1 h_u33 (.A(r0[8][35]), .B(r0[9][35]), .CIN(1'b0), .SUM(h_s33), .COUT(h_c33));
    wire h_s34, h_c34; sky130_fd_sc_hd__fa_1 h_u34 (.A(r0[2][36]), .B(r0[3][36]), .CIN(r0[4][36]), .SUM(h_s34), .COUT(h_c34));
    wire h_s35, h_c35; sky130_fd_sc_hd__fa_1 h_u35 (.A(r0[5][36]), .B(r0[6][36]), .CIN(r0[7][36]), .SUM(h_s35), .COUT(h_c35));
    wire h_s36, h_c36; sky130_fd_sc_hd__fa_1 h_u36 (.A(r0[3][37]), .B(r0[4][37]), .CIN(r0[5][37]), .SUM(h_s36), .COUT(h_c36));
    wire h_s37, h_c37; sky130_fd_sc_hd__fa_1 h_u37 (.A(r0[6][37]), .B(r0[7][37]), .CIN(1'b0), .SUM(h_s37), .COUT(h_c37));
    wire h_s38, h_c38; sky130_fd_sc_hd__fa_1 h_u38 (.A(r0[3][38]), .B(r0[4][38]), .CIN(r0[5][38]), .SUM(h_s38), .COUT(h_c38));
    wire h_s39, h_c39; sky130_fd_sc_hd__fa_1 h_u39 (.A(r0[4][39]), .B(r0[5][39]), .CIN(1'b0), .SUM(h_s39), .COUT(h_c39));
    wire h_s40, h_c40; sky130_fd_sc_hd__fa_1 h_u40 (.A(r0[0][16]), .B(r0[1][16]), .CIN(1'b0), .SUM(h_s40), .COUT(h_c40));
    wire h_s41, h_c41; sky130_fd_sc_hd__fa_1 h_u41 (.A(r0[0][17]), .B(r0[1][17]), .CIN(1'b0), .SUM(h_s41), .COUT(h_c41));
    wire h_s42, h_c42; sky130_fd_sc_hd__fa_1 h_u42 (.A(r0[0][18]), .B(r0[1][18]), .CIN(r0[2][18]), .SUM(h_s42), .COUT(h_c42));
    wire h_s43, h_c43; sky130_fd_sc_hd__fa_1 h_u43 (.A(r0[3][18]), .B(r0[4][18]), .CIN(1'b0), .SUM(h_s43), .COUT(h_c43));
    wire h_s44, h_c44; sky130_fd_sc_hd__fa_1 h_u44 (.A(r0[0][19]), .B(r0[1][19]), .CIN(r0[2][19]), .SUM(h_s44), .COUT(h_c44));
    wire h_s45, h_c45; sky130_fd_sc_hd__fa_1 h_u45 (.A(r0[3][19]), .B(r0[4][19]), .CIN(1'b0), .SUM(h_s45), .COUT(h_c45));
    wire h_s46, h_c46; sky130_fd_sc_hd__fa_1 h_u46 (.A(r0[0][20]), .B(r0[1][20]), .CIN(r0[2][20]), .SUM(h_s46), .COUT(h_c46));
    wire h_s47, h_c47; sky130_fd_sc_hd__fa_1 h_u47 (.A(r0[3][20]), .B(r0[4][20]), .CIN(r0[5][20]), .SUM(h_s47), .COUT(h_c47));
    wire h_s48, h_c48; sky130_fd_sc_hd__fa_1 h_u48 (.A(r0[6][20]), .B(r0[7][20]), .CIN(1'b0), .SUM(h_s48), .COUT(h_c48));
    wire h_s49, h_c49; sky130_fd_sc_hd__fa_1 h_u49 (.A(r0[0][21]), .B(r0[1][21]), .CIN(r0[2][21]), .SUM(h_s49), .COUT(h_c49));
    wire h_s50, h_c50; sky130_fd_sc_hd__fa_1 h_u50 (.A(r0[3][21]), .B(r0[4][21]), .CIN(r0[5][21]), .SUM(h_s50), .COUT(h_c50));
    wire h_s51, h_c51; sky130_fd_sc_hd__fa_1 h_u51 (.A(r0[6][21]), .B(r0[7][21]), .CIN(1'b0), .SUM(h_s51), .COUT(h_c51));
    wire h_s52, h_c52; sky130_fd_sc_hd__fa_1 h_u52 (.A(r0[0][22]), .B(r0[1][22]), .CIN(r0[2][22]), .SUM(h_s52), .COUT(h_c52));
    wire h_s53, h_c53; sky130_fd_sc_hd__fa_1 h_u53 (.A(r0[3][22]), .B(r0[4][22]), .CIN(r0[5][22]), .SUM(h_s53), .COUT(h_c53));
    wire h_s54, h_c54; sky130_fd_sc_hd__fa_1 h_u54 (.A(r0[6][22]), .B(r0[7][22]), .CIN(r0[8][22]), .SUM(h_s54), .COUT(h_c54));
    wire h_s55, h_c55; sky130_fd_sc_hd__fa_1 h_u55 (.A(r0[9][22]), .B(r0[10][22]), .CIN(1'b0), .SUM(h_s55), .COUT(h_c55));
    wire h_s56, h_c56; sky130_fd_sc_hd__fa_1 h_u56 (.A(r0[0][23]), .B(r0[1][23]), .CIN(r0[2][23]), .SUM(h_s56), .COUT(h_c56));
    wire h_s57, h_c57; sky130_fd_sc_hd__fa_1 h_u57 (.A(r0[3][23]), .B(r0[4][23]), .CIN(r0[5][23]), .SUM(h_s57), .COUT(h_c57));
    wire h_s58, h_c58; sky130_fd_sc_hd__fa_1 h_u58 (.A(r0[6][23]), .B(r0[7][23]), .CIN(r0[8][23]), .SUM(h_s58), .COUT(h_c58));
    wire h_s59, h_c59; sky130_fd_sc_hd__fa_1 h_u59 (.A(r0[9][23]), .B(r0[10][23]), .CIN(1'b0), .SUM(h_s59), .COUT(h_c59));
    wire h_s60, h_c60; sky130_fd_sc_hd__fa_1 h_u60 (.A(h_s0), .B(r0[2][24]), .CIN(r0[3][24]), .SUM(h_s60), .COUT(h_c60));
    wire h_s61, h_c61; sky130_fd_sc_hd__fa_1 h_u61 (.A(r0[4][24]), .B(r0[5][24]), .CIN(r0[6][24]), .SUM(h_s61), .COUT(h_c61));
    wire h_s62, h_c62; sky130_fd_sc_hd__fa_1 h_u62 (.A(r0[7][24]), .B(r0[8][24]), .CIN(r0[9][24]), .SUM(h_s62), .COUT(h_c62));
    wire h_s63, h_c63; sky130_fd_sc_hd__fa_1 h_u63 (.A(r0[10][24]), .B(r0[11][24]), .CIN(r0[12][24]), .SUM(h_s63), .COUT(h_c63));
    wire h_s64, h_c64; sky130_fd_sc_hd__fa_1 h_u64 (.A(h_c0), .B(h_s1), .CIN(r0[2][25]), .SUM(h_s64), .COUT(h_c64));
    wire h_s65, h_c65; sky130_fd_sc_hd__fa_1 h_u65 (.A(r0[3][25]), .B(r0[4][25]), .CIN(r0[5][25]), .SUM(h_s65), .COUT(h_c65));
    wire h_s66, h_c66; sky130_fd_sc_hd__fa_1 h_u66 (.A(r0[6][25]), .B(r0[7][25]), .CIN(r0[8][25]), .SUM(h_s66), .COUT(h_c66));
    wire h_s67, h_c67; sky130_fd_sc_hd__fa_1 h_u67 (.A(r0[9][25]), .B(r0[10][25]), .CIN(r0[11][25]), .SUM(h_s67), .COUT(h_c67));
    wire h_s68, h_c68; sky130_fd_sc_hd__fa_1 h_u68 (.A(h_c1), .B(h_s2), .CIN(h_s3), .SUM(h_s68), .COUT(h_c68));
    wire h_s69, h_c69; sky130_fd_sc_hd__fa_1 h_u69 (.A(r0[5][26]), .B(r0[6][26]), .CIN(r0[7][26]), .SUM(h_s69), .COUT(h_c69));
    wire h_s70, h_c70; sky130_fd_sc_hd__fa_1 h_u70 (.A(r0[8][26]), .B(r0[9][26]), .CIN(r0[10][26]), .SUM(h_s70), .COUT(h_c70));
    wire h_s71, h_c71; sky130_fd_sc_hd__fa_1 h_u71 (.A(r0[11][26]), .B(r0[12][26]), .CIN(r0[13][26]), .SUM(h_s71), .COUT(h_c71));
    wire h_s72, h_c72; sky130_fd_sc_hd__fa_1 h_u72 (.A(h_c2), .B(h_c3), .CIN(h_s4), .SUM(h_s72), .COUT(h_c72));
    wire h_s73, h_c73; sky130_fd_sc_hd__fa_1 h_u73 (.A(h_s5), .B(r0[5][27]), .CIN(r0[6][27]), .SUM(h_s73), .COUT(h_c73));
    wire h_s74, h_c74; sky130_fd_sc_hd__fa_1 h_u74 (.A(r0[7][27]), .B(r0[8][27]), .CIN(r0[9][27]), .SUM(h_s74), .COUT(h_c74));
    wire h_s75, h_c75; sky130_fd_sc_hd__fa_1 h_u75 (.A(r0[10][27]), .B(r0[11][27]), .CIN(r0[12][27]), .SUM(h_s75), .COUT(h_c75));
    wire h_s76, h_c76; sky130_fd_sc_hd__fa_1 h_u76 (.A(h_c4), .B(h_c5), .CIN(h_s6), .SUM(h_s76), .COUT(h_c76));
    wire h_s77, h_c77; sky130_fd_sc_hd__fa_1 h_u77 (.A(h_s7), .B(h_s8), .CIN(r0[8][28]), .SUM(h_s77), .COUT(h_c77));
    wire h_s78, h_c78; sky130_fd_sc_hd__fa_1 h_u78 (.A(r0[9][28]), .B(r0[10][28]), .CIN(r0[11][28]), .SUM(h_s78), .COUT(h_c78));
    wire h_s79, h_c79; sky130_fd_sc_hd__fa_1 h_u79 (.A(r0[12][28]), .B(r0[13][28]), .CIN(r0[14][28]), .SUM(h_s79), .COUT(h_c79));
    wire h_s80, h_c80; sky130_fd_sc_hd__fa_1 h_u80 (.A(h_c6), .B(h_c7), .CIN(h_c8), .SUM(h_s80), .COUT(h_c80));
    wire h_s81, h_c81; sky130_fd_sc_hd__fa_1 h_u81 (.A(h_s9), .B(h_s10), .CIN(h_s11), .SUM(h_s81), .COUT(h_c81));
    wire h_s82, h_c82; sky130_fd_sc_hd__fa_1 h_u82 (.A(r0[8][29]), .B(r0[9][29]), .CIN(r0[10][29]), .SUM(h_s82), .COUT(h_c82));
    wire h_s83, h_c83; sky130_fd_sc_hd__fa_1 h_u83 (.A(r0[11][29]), .B(r0[12][29]), .CIN(r0[13][29]), .SUM(h_s83), .COUT(h_c83));
    wire h_s84, h_c84; sky130_fd_sc_hd__fa_1 h_u84 (.A(h_c9), .B(h_c10), .CIN(h_c11), .SUM(h_s84), .COUT(h_c84));
    wire h_s85, h_c85; sky130_fd_sc_hd__fa_1 h_u85 (.A(h_s12), .B(h_s13), .CIN(h_s14), .SUM(h_s85), .COUT(h_c85));
    wire h_s86, h_c86; sky130_fd_sc_hd__fa_1 h_u86 (.A(h_s15), .B(r0[12][30]), .CIN(r0[13][30]), .SUM(h_s86), .COUT(h_c86));
    wire h_s87, h_c87; sky130_fd_sc_hd__fa_1 h_u87 (.A(r0[14][30]), .B(r0[15][30]), .CIN(r0[16][30]), .SUM(h_s87), .COUT(h_c87));
    wire h_s88, h_c88; sky130_fd_sc_hd__fa_1 h_u88 (.A(h_c12), .B(h_c13), .CIN(h_c14), .SUM(h_s88), .COUT(h_c88));
    wire h_s89, h_c89; sky130_fd_sc_hd__fa_1 h_u89 (.A(h_c15), .B(h_s16), .CIN(h_s17), .SUM(h_s89), .COUT(h_c89));
    wire h_s90, h_c90; sky130_fd_sc_hd__fa_1 h_u90 (.A(h_s18), .B(h_s19), .CIN(r0[11][31]), .SUM(h_s90), .COUT(h_c90));
    wire h_s91, h_c91; sky130_fd_sc_hd__fa_1 h_u91 (.A(r0[12][31]), .B(r0[13][31]), .CIN(r0[14][31]), .SUM(h_s91), .COUT(h_c91));
    wire h_s92, h_c92; sky130_fd_sc_hd__fa_1 h_u92 (.A(h_c16), .B(h_c17), .CIN(h_c18), .SUM(h_s92), .COUT(h_c92));
    wire h_s93, h_c93; sky130_fd_sc_hd__fa_1 h_u93 (.A(h_c19), .B(h_s20), .CIN(h_s21), .SUM(h_s93), .COUT(h_c93));
    wire h_s94, h_c94; sky130_fd_sc_hd__fa_1 h_u94 (.A(h_s22), .B(h_s23), .CIN(r0[12][32]), .SUM(h_s94), .COUT(h_c94));
    wire h_s95, h_c95; sky130_fd_sc_hd__fa_1 h_u95 (.A(r0[13][32]), .B(r0[14][32]), .CIN(r0[15][32]), .SUM(h_s95), .COUT(h_c95));
    wire h_s96, h_c96; sky130_fd_sc_hd__fa_1 h_u96 (.A(h_c20), .B(h_c21), .CIN(h_c22), .SUM(h_s96), .COUT(h_c96));
    wire h_s97, h_c97; sky130_fd_sc_hd__fa_1 h_u97 (.A(h_c23), .B(h_s24), .CIN(h_s25), .SUM(h_s97), .COUT(h_c97));
    wire h_s98, h_c98; sky130_fd_sc_hd__fa_1 h_u98 (.A(h_s26), .B(h_s27), .CIN(r0[12][33]), .SUM(h_s98), .COUT(h_c98));
    wire h_s99, h_c99; sky130_fd_sc_hd__fa_1 h_u99 (.A(r0[13][33]), .B(r0[14][33]), .CIN(r0[15][33]), .SUM(h_s99), .COUT(h_c99));
    wire h_s100, h_c100; sky130_fd_sc_hd__fa_1 h_u100 (.A(h_c24), .B(h_c25), .CIN(h_c26), .SUM(h_s100), .COUT(h_c100));
    wire h_s101, h_c101; sky130_fd_sc_hd__fa_1 h_u101 (.A(h_c27), .B(h_s28), .CIN(h_s29), .SUM(h_s101), .COUT(h_c101));
    wire h_s102, h_c102; sky130_fd_sc_hd__fa_1 h_u102 (.A(h_s30), .B(r0[10][34]), .CIN(r0[11][34]), .SUM(h_s102), .COUT(h_c102));
    wire h_s103, h_c103; sky130_fd_sc_hd__fa_1 h_u103 (.A(r0[12][34]), .B(r0[13][34]), .CIN(r0[14][34]), .SUM(h_s103), .COUT(h_c103));
    wire h_s104, h_c104; sky130_fd_sc_hd__fa_1 h_u104 (.A(h_c28), .B(h_c29), .CIN(h_c30), .SUM(h_s104), .COUT(h_c104));
    wire h_s105, h_c105; sky130_fd_sc_hd__fa_1 h_u105 (.A(h_s31), .B(h_s32), .CIN(h_s33), .SUM(h_s105), .COUT(h_c105));
    wire h_s106, h_c106; sky130_fd_sc_hd__fa_1 h_u106 (.A(r0[10][35]), .B(r0[11][35]), .CIN(r0[12][35]), .SUM(h_s106), .COUT(h_c106));
    wire h_s107, h_c107; sky130_fd_sc_hd__fa_1 h_u107 (.A(r0[13][35]), .B(r0[14][35]), .CIN(r0[15][35]), .SUM(h_s107), .COUT(h_c107));
    wire h_s108, h_c108; sky130_fd_sc_hd__fa_1 h_u108 (.A(h_c31), .B(h_c32), .CIN(h_c33), .SUM(h_s108), .COUT(h_c108));
    wire h_s109, h_c109; sky130_fd_sc_hd__fa_1 h_u109 (.A(h_s34), .B(h_s35), .CIN(r0[8][36]), .SUM(h_s109), .COUT(h_c109));
    wire h_s110, h_c110; sky130_fd_sc_hd__fa_1 h_u110 (.A(r0[9][36]), .B(r0[10][36]), .CIN(r0[11][36]), .SUM(h_s110), .COUT(h_c110));
    wire h_s111, h_c111; sky130_fd_sc_hd__fa_1 h_u111 (.A(r0[12][36]), .B(r0[13][36]), .CIN(r0[14][36]), .SUM(h_s111), .COUT(h_c111));
    wire h_s112, h_c112; sky130_fd_sc_hd__fa_1 h_u112 (.A(h_c34), .B(h_c35), .CIN(h_s36), .SUM(h_s112), .COUT(h_c112));
    wire h_s113, h_c113; sky130_fd_sc_hd__fa_1 h_u113 (.A(h_s37), .B(r0[8][37]), .CIN(r0[9][37]), .SUM(h_s113), .COUT(h_c113));
    wire h_s114, h_c114; sky130_fd_sc_hd__fa_1 h_u114 (.A(r0[10][37]), .B(r0[11][37]), .CIN(r0[12][37]), .SUM(h_s114), .COUT(h_c114));
    wire h_s115, h_c115; sky130_fd_sc_hd__fa_1 h_u115 (.A(r0[13][37]), .B(r0[14][37]), .CIN(r0[15][37]), .SUM(h_s115), .COUT(h_c115));
    wire h_s116, h_c116; sky130_fd_sc_hd__fa_1 h_u116 (.A(h_c36), .B(h_c37), .CIN(h_s38), .SUM(h_s116), .COUT(h_c116));
    wire h_s117, h_c117; sky130_fd_sc_hd__fa_1 h_u117 (.A(r0[6][38]), .B(r0[7][38]), .CIN(r0[8][38]), .SUM(h_s117), .COUT(h_c117));
    wire h_s118, h_c118; sky130_fd_sc_hd__fa_1 h_u118 (.A(r0[9][38]), .B(r0[10][38]), .CIN(r0[11][38]), .SUM(h_s118), .COUT(h_c118));
    wire h_s119, h_c119; sky130_fd_sc_hd__fa_1 h_u119 (.A(r0[12][38]), .B(r0[13][38]), .CIN(r0[14][38]), .SUM(h_s119), .COUT(h_c119));
    wire h_s120, h_c120; sky130_fd_sc_hd__fa_1 h_u120 (.A(h_c38), .B(h_s39), .CIN(r0[6][39]), .SUM(h_s120), .COUT(h_c120));
    wire h_s121, h_c121; sky130_fd_sc_hd__fa_1 h_u121 (.A(r0[7][39]), .B(r0[8][39]), .CIN(r0[9][39]), .SUM(h_s121), .COUT(h_c121));
    wire h_s122, h_c122; sky130_fd_sc_hd__fa_1 h_u122 (.A(r0[10][39]), .B(r0[11][39]), .CIN(r0[12][39]), .SUM(h_s122), .COUT(h_c122));
    wire h_s123, h_c123; sky130_fd_sc_hd__fa_1 h_u123 (.A(r0[13][39]), .B(r0[14][39]), .CIN(r0[15][39]), .SUM(h_s123), .COUT(h_c123));
    wire h_s124, h_c124; sky130_fd_sc_hd__fa_1 h_u124 (.A(h_c39), .B(r0[4][40]), .CIN(r0[5][40]), .SUM(h_s124), .COUT(h_c124));
    wire h_s125, h_c125; sky130_fd_sc_hd__fa_1 h_u125 (.A(r0[6][40]), .B(r0[7][40]), .CIN(r0[8][40]), .SUM(h_s125), .COUT(h_c125));
    wire h_s126, h_c126; sky130_fd_sc_hd__fa_1 h_u126 (.A(r0[9][40]), .B(r0[10][40]), .CIN(r0[11][40]), .SUM(h_s126), .COUT(h_c126));
    wire h_s127, h_c127; sky130_fd_sc_hd__fa_1 h_u127 (.A(r0[12][40]), .B(r0[13][40]), .CIN(r0[14][40]), .SUM(h_s127), .COUT(h_c127));
    wire h_s128, h_c128; sky130_fd_sc_hd__fa_1 h_u128 (.A(r0[5][41]), .B(r0[6][41]), .CIN(r0[7][41]), .SUM(h_s128), .COUT(h_c128));
    wire h_s129, h_c129; sky130_fd_sc_hd__fa_1 h_u129 (.A(r0[8][41]), .B(r0[9][41]), .CIN(r0[10][41]), .SUM(h_s129), .COUT(h_c129));
    wire h_s130, h_c130; sky130_fd_sc_hd__fa_1 h_u130 (.A(r0[11][41]), .B(r0[12][41]), .CIN(r0[13][41]), .SUM(h_s130), .COUT(h_c130));
    wire h_s131, h_c131; sky130_fd_sc_hd__fa_1 h_u131 (.A(r0[14][41]), .B(r0[15][41]), .CIN(1'b0), .SUM(h_s131), .COUT(h_c131));
    wire h_s132, h_c132; sky130_fd_sc_hd__fa_1 h_u132 (.A(r0[5][42]), .B(r0[6][42]), .CIN(r0[7][42]), .SUM(h_s132), .COUT(h_c132));
    wire h_s133, h_c133; sky130_fd_sc_hd__fa_1 h_u133 (.A(r0[8][42]), .B(r0[9][42]), .CIN(r0[10][42]), .SUM(h_s133), .COUT(h_c133));
    wire h_s134, h_c134; sky130_fd_sc_hd__fa_1 h_u134 (.A(r0[11][42]), .B(r0[12][42]), .CIN(r0[13][42]), .SUM(h_s134), .COUT(h_c134));
    wire h_s135, h_c135; sky130_fd_sc_hd__fa_1 h_u135 (.A(r0[6][43]), .B(r0[7][43]), .CIN(r0[8][43]), .SUM(h_s135), .COUT(h_c135));
    wire h_s136, h_c136; sky130_fd_sc_hd__fa_1 h_u136 (.A(r0[9][43]), .B(r0[10][43]), .CIN(r0[11][43]), .SUM(h_s136), .COUT(h_c136));
    wire h_s137, h_c137; sky130_fd_sc_hd__fa_1 h_u137 (.A(r0[12][43]), .B(r0[13][43]), .CIN(1'b0), .SUM(h_s137), .COUT(h_c137));
    wire h_s138, h_c138; sky130_fd_sc_hd__fa_1 h_u138 (.A(r0[6][44]), .B(r0[7][44]), .CIN(r0[8][44]), .SUM(h_s138), .COUT(h_c138));
    wire h_s139, h_c139; sky130_fd_sc_hd__fa_1 h_u139 (.A(r0[9][44]), .B(r0[10][44]), .CIN(r0[11][44]), .SUM(h_s139), .COUT(h_c139));
    wire h_s140, h_c140; sky130_fd_sc_hd__fa_1 h_u140 (.A(r0[7][45]), .B(r0[8][45]), .CIN(r0[9][45]), .SUM(h_s140), .COUT(h_c140));
    wire h_s141, h_c141; sky130_fd_sc_hd__fa_1 h_u141 (.A(r0[10][45]), .B(r0[11][45]), .CIN(1'b0), .SUM(h_s141), .COUT(h_c141));
    wire h_s142, h_c142; sky130_fd_sc_hd__fa_1 h_u142 (.A(r0[7][46]), .B(r0[8][46]), .CIN(r0[9][46]), .SUM(h_s142), .COUT(h_c142));
    wire h_s143, h_c143; sky130_fd_sc_hd__fa_1 h_u143 (.A(r0[8][47]), .B(r0[9][47]), .CIN(1'b0), .SUM(h_s143), .COUT(h_c143));
    wire h_s144, h_c144; sky130_fd_sc_hd__fa_1 h_u144 (.A(r0[0][10]), .B(r0[1][10]), .CIN(1'b0), .SUM(h_s144), .COUT(h_c144));
    wire h_s145, h_c145; sky130_fd_sc_hd__fa_1 h_u145 (.A(r0[0][11]), .B(r0[1][11]), .CIN(1'b0), .SUM(h_s145), .COUT(h_c145));
    wire h_s146, h_c146; sky130_fd_sc_hd__fa_1 h_u146 (.A(r0[0][12]), .B(r0[1][12]), .CIN(r0[2][12]), .SUM(h_s146), .COUT(h_c146));
    wire h_s147, h_c147; sky130_fd_sc_hd__fa_1 h_u147 (.A(r0[3][12]), .B(r0[4][12]), .CIN(1'b0), .SUM(h_s147), .COUT(h_c147));
    wire h_s148, h_c148; sky130_fd_sc_hd__fa_1 h_u148 (.A(r0[0][13]), .B(r0[1][13]), .CIN(r0[2][13]), .SUM(h_s148), .COUT(h_c148));
    wire h_s149, h_c149; sky130_fd_sc_hd__fa_1 h_u149 (.A(r0[3][13]), .B(r0[4][13]), .CIN(1'b0), .SUM(h_s149), .COUT(h_c149));
    wire h_s150, h_c150; sky130_fd_sc_hd__fa_1 h_u150 (.A(r0[0][14]), .B(r0[1][14]), .CIN(r0[2][14]), .SUM(h_s150), .COUT(h_c150));
    wire h_s151, h_c151; sky130_fd_sc_hd__fa_1 h_u151 (.A(r0[3][14]), .B(r0[4][14]), .CIN(r0[5][14]), .SUM(h_s151), .COUT(h_c151));
    wire h_s152, h_c152; sky130_fd_sc_hd__fa_1 h_u152 (.A(r0[6][14]), .B(r0[7][14]), .CIN(1'b0), .SUM(h_s152), .COUT(h_c152));
    wire h_s153, h_c153; sky130_fd_sc_hd__fa_1 h_u153 (.A(r0[0][15]), .B(r0[1][15]), .CIN(r0[2][15]), .SUM(h_s153), .COUT(h_c153));
    wire h_s154, h_c154; sky130_fd_sc_hd__fa_1 h_u154 (.A(r0[3][15]), .B(r0[4][15]), .CIN(r0[5][15]), .SUM(h_s154), .COUT(h_c154));
    wire h_s155, h_c155; sky130_fd_sc_hd__fa_1 h_u155 (.A(r0[6][15]), .B(r0[7][15]), .CIN(1'b0), .SUM(h_s155), .COUT(h_c155));
    wire h_s156, h_c156; sky130_fd_sc_hd__fa_1 h_u156 (.A(h_s40), .B(r0[2][16]), .CIN(r0[3][16]), .SUM(h_s156), .COUT(h_c156));
    wire h_s157, h_c157; sky130_fd_sc_hd__fa_1 h_u157 (.A(r0[4][16]), .B(r0[5][16]), .CIN(r0[6][16]), .SUM(h_s157), .COUT(h_c157));
    wire h_s158, h_c158; sky130_fd_sc_hd__fa_1 h_u158 (.A(r0[7][16]), .B(r0[8][16]), .CIN(r0[16][16]), .SUM(h_s158), .COUT(h_c158));
    wire h_s159, h_c159; sky130_fd_sc_hd__fa_1 h_u159 (.A(h_c40), .B(h_s41), .CIN(r0[2][17]), .SUM(h_s159), .COUT(h_c159));
    wire h_s160, h_c160; sky130_fd_sc_hd__fa_1 h_u160 (.A(r0[3][17]), .B(r0[4][17]), .CIN(r0[5][17]), .SUM(h_s160), .COUT(h_c160));
    wire h_s161, h_c161; sky130_fd_sc_hd__fa_1 h_u161 (.A(r0[6][17]), .B(r0[7][17]), .CIN(r0[8][17]), .SUM(h_s161), .COUT(h_c161));
    wire h_s162, h_c162; sky130_fd_sc_hd__fa_1 h_u162 (.A(h_c41), .B(h_s42), .CIN(h_s43), .SUM(h_s162), .COUT(h_c162));
    wire h_s163, h_c163; sky130_fd_sc_hd__fa_1 h_u163 (.A(r0[5][18]), .B(r0[6][18]), .CIN(r0[7][18]), .SUM(h_s163), .COUT(h_c163));
    wire h_s164, h_c164; sky130_fd_sc_hd__fa_1 h_u164 (.A(r0[8][18]), .B(r0[9][18]), .CIN(r0[16][18]), .SUM(h_s164), .COUT(h_c164));
    wire h_s165, h_c165; sky130_fd_sc_hd__fa_1 h_u165 (.A(h_c42), .B(h_c43), .CIN(h_s44), .SUM(h_s165), .COUT(h_c165));
    wire h_s166, h_c166; sky130_fd_sc_hd__fa_1 h_u166 (.A(h_s45), .B(r0[5][19]), .CIN(r0[6][19]), .SUM(h_s166), .COUT(h_c166));
    wire h_s167, h_c167; sky130_fd_sc_hd__fa_1 h_u167 (.A(r0[7][19]), .B(r0[8][19]), .CIN(r0[9][19]), .SUM(h_s167), .COUT(h_c167));
    wire h_s168, h_c168; sky130_fd_sc_hd__fa_1 h_u168 (.A(h_c44), .B(h_c45), .CIN(h_s46), .SUM(h_s168), .COUT(h_c168));
    wire h_s169, h_c169; sky130_fd_sc_hd__fa_1 h_u169 (.A(h_s47), .B(h_s48), .CIN(r0[8][20]), .SUM(h_s169), .COUT(h_c169));
    wire h_s170, h_c170; sky130_fd_sc_hd__fa_1 h_u170 (.A(r0[9][20]), .B(r0[10][20]), .CIN(r0[16][20]), .SUM(h_s170), .COUT(h_c170));
    wire h_s171, h_c171; sky130_fd_sc_hd__fa_1 h_u171 (.A(h_c46), .B(h_c47), .CIN(h_c48), .SUM(h_s171), .COUT(h_c171));
    wire h_s172, h_c172; sky130_fd_sc_hd__fa_1 h_u172 (.A(h_s49), .B(h_s50), .CIN(h_s51), .SUM(h_s172), .COUT(h_c172));
    wire h_s173, h_c173; sky130_fd_sc_hd__fa_1 h_u173 (.A(r0[8][21]), .B(r0[9][21]), .CIN(r0[10][21]), .SUM(h_s173), .COUT(h_c173));
    wire h_s174, h_c174; sky130_fd_sc_hd__fa_1 h_u174 (.A(h_c49), .B(h_c50), .CIN(h_c51), .SUM(h_s174), .COUT(h_c174));
    wire h_s175, h_c175; sky130_fd_sc_hd__fa_1 h_u175 (.A(h_s52), .B(h_s53), .CIN(h_s54), .SUM(h_s175), .COUT(h_c175));
    wire h_s176, h_c176; sky130_fd_sc_hd__fa_1 h_u176 (.A(h_s55), .B(r0[11][22]), .CIN(r0[16][22]), .SUM(h_s176), .COUT(h_c176));
    wire h_s177, h_c177; sky130_fd_sc_hd__fa_1 h_u177 (.A(h_c52), .B(h_c53), .CIN(h_c54), .SUM(h_s177), .COUT(h_c177));
    wire h_s178, h_c178; sky130_fd_sc_hd__fa_1 h_u178 (.A(h_c55), .B(h_s56), .CIN(h_s57), .SUM(h_s178), .COUT(h_c178));
    wire h_s179, h_c179; sky130_fd_sc_hd__fa_1 h_u179 (.A(h_s58), .B(h_s59), .CIN(r0[11][23]), .SUM(h_s179), .COUT(h_c179));
    wire h_s180, h_c180; sky130_fd_sc_hd__fa_1 h_u180 (.A(h_c56), .B(h_c57), .CIN(h_c58), .SUM(h_s180), .COUT(h_c180));
    wire h_s181, h_c181; sky130_fd_sc_hd__fa_1 h_u181 (.A(h_c59), .B(h_s60), .CIN(h_s61), .SUM(h_s181), .COUT(h_c181));
    wire h_s182, h_c182; sky130_fd_sc_hd__fa_1 h_u182 (.A(h_s62), .B(h_s63), .CIN(r0[16][24]), .SUM(h_s182), .COUT(h_c182));
    wire h_s183, h_c183; sky130_fd_sc_hd__fa_1 h_u183 (.A(h_c60), .B(h_c61), .CIN(h_c62), .SUM(h_s183), .COUT(h_c183));
    wire h_s184, h_c184; sky130_fd_sc_hd__fa_1 h_u184 (.A(h_c63), .B(h_s64), .CIN(h_s65), .SUM(h_s184), .COUT(h_c184));
    wire h_s185, h_c185; sky130_fd_sc_hd__fa_1 h_u185 (.A(h_s66), .B(h_s67), .CIN(r0[12][25]), .SUM(h_s185), .COUT(h_c185));
    wire h_s186, h_c186; sky130_fd_sc_hd__fa_1 h_u186 (.A(h_c64), .B(h_c65), .CIN(h_c66), .SUM(h_s186), .COUT(h_c186));
    wire h_s187, h_c187; sky130_fd_sc_hd__fa_1 h_u187 (.A(h_c67), .B(h_s68), .CIN(h_s69), .SUM(h_s187), .COUT(h_c187));
    wire h_s188, h_c188; sky130_fd_sc_hd__fa_1 h_u188 (.A(h_s70), .B(h_s71), .CIN(r0[16][26]), .SUM(h_s188), .COUT(h_c188));
    wire h_s189, h_c189; sky130_fd_sc_hd__fa_1 h_u189 (.A(h_c68), .B(h_c69), .CIN(h_c70), .SUM(h_s189), .COUT(h_c189));
    wire h_s190, h_c190; sky130_fd_sc_hd__fa_1 h_u190 (.A(h_c71), .B(h_s72), .CIN(h_s73), .SUM(h_s190), .COUT(h_c190));
    wire h_s191, h_c191; sky130_fd_sc_hd__fa_1 h_u191 (.A(h_s74), .B(h_s75), .CIN(r0[13][27]), .SUM(h_s191), .COUT(h_c191));
    wire h_s192, h_c192; sky130_fd_sc_hd__fa_1 h_u192 (.A(h_c72), .B(h_c73), .CIN(h_c74), .SUM(h_s192), .COUT(h_c192));
    wire h_s193, h_c193; sky130_fd_sc_hd__fa_1 h_u193 (.A(h_c75), .B(h_s76), .CIN(h_s77), .SUM(h_s193), .COUT(h_c193));
    wire h_s194, h_c194; sky130_fd_sc_hd__fa_1 h_u194 (.A(h_s78), .B(h_s79), .CIN(r0[16][28]), .SUM(h_s194), .COUT(h_c194));
    wire h_s195, h_c195; sky130_fd_sc_hd__fa_1 h_u195 (.A(h_c76), .B(h_c77), .CIN(h_c78), .SUM(h_s195), .COUT(h_c195));
    wire h_s196, h_c196; sky130_fd_sc_hd__fa_1 h_u196 (.A(h_c79), .B(h_s80), .CIN(h_s81), .SUM(h_s196), .COUT(h_c196));
    wire h_s197, h_c197; sky130_fd_sc_hd__fa_1 h_u197 (.A(h_s82), .B(h_s83), .CIN(r0[14][29]), .SUM(h_s197), .COUT(h_c197));
    wire h_s198, h_c198; sky130_fd_sc_hd__fa_1 h_u198 (.A(h_c80), .B(h_c81), .CIN(h_c82), .SUM(h_s198), .COUT(h_c198));
    wire h_s199, h_c199; sky130_fd_sc_hd__fa_1 h_u199 (.A(h_c83), .B(h_s84), .CIN(h_s85), .SUM(h_s199), .COUT(h_c199));
    wire h_s200, h_c200; sky130_fd_sc_hd__fa_1 h_u200 (.A(h_s86), .B(h_s87), .CIN(1'b1), .SUM(h_s200), .COUT(h_c200));
    wire h_s201, h_c201; sky130_fd_sc_hd__fa_1 h_u201 (.A(h_c84), .B(h_c85), .CIN(h_c86), .SUM(h_s201), .COUT(h_c201));
    wire h_s202, h_c202; sky130_fd_sc_hd__fa_1 h_u202 (.A(h_c87), .B(h_s88), .CIN(h_s89), .SUM(h_s202), .COUT(h_c202));
    wire h_s203, h_c203; sky130_fd_sc_hd__fa_1 h_u203 (.A(h_s90), .B(h_s91), .CIN(r0[15][31]), .SUM(h_s203), .COUT(h_c203));
    wire h_s204, h_c204; sky130_fd_sc_hd__fa_1 h_u204 (.A(h_c88), .B(h_c89), .CIN(h_c90), .SUM(h_s204), .COUT(h_c204));
    wire h_s205, h_c205; sky130_fd_sc_hd__fa_1 h_u205 (.A(h_c91), .B(h_s92), .CIN(h_s93), .SUM(h_s205), .COUT(h_c205));
    wire h_s206, h_c206; sky130_fd_sc_hd__fa_1 h_u206 (.A(h_s94), .B(h_s95), .CIN(1'b1), .SUM(h_s206), .COUT(h_c206));
    wire h_s207, h_c207; sky130_fd_sc_hd__fa_1 h_u207 (.A(h_c92), .B(h_c93), .CIN(h_c94), .SUM(h_s207), .COUT(h_c207));
    wire h_s208, h_c208; sky130_fd_sc_hd__fa_1 h_u208 (.A(h_c95), .B(h_s96), .CIN(h_s97), .SUM(h_s208), .COUT(h_c208));
    wire h_s209, h_c209; sky130_fd_sc_hd__fa_1 h_u209 (.A(h_s98), .B(h_s99), .CIN(1'b1), .SUM(h_s209), .COUT(h_c209));
    wire h_s210, h_c210; sky130_fd_sc_hd__fa_1 h_u210 (.A(h_c96), .B(h_c97), .CIN(h_c98), .SUM(h_s210), .COUT(h_c210));
    wire h_s211, h_c211; sky130_fd_sc_hd__fa_1 h_u211 (.A(h_c99), .B(h_s100), .CIN(h_s101), .SUM(h_s211), .COUT(h_c211));
    wire h_s212, h_c212; sky130_fd_sc_hd__fa_1 h_u212 (.A(h_s102), .B(h_s103), .CIN(r0[15][34]), .SUM(h_s212), .COUT(h_c212));
    wire h_s213, h_c213; sky130_fd_sc_hd__fa_1 h_u213 (.A(h_c100), .B(h_c101), .CIN(h_c102), .SUM(h_s213), .COUT(h_c213));
    wire h_s214, h_c214; sky130_fd_sc_hd__fa_1 h_u214 (.A(h_c103), .B(h_s104), .CIN(h_s105), .SUM(h_s214), .COUT(h_c214));
    wire h_s215, h_c215; sky130_fd_sc_hd__fa_1 h_u215 (.A(h_s106), .B(h_s107), .CIN(1'b1), .SUM(h_s215), .COUT(h_c215));
    wire h_s216, h_c216; sky130_fd_sc_hd__fa_1 h_u216 (.A(h_c104), .B(h_c105), .CIN(h_c106), .SUM(h_s216), .COUT(h_c216));
    wire h_s217, h_c217; sky130_fd_sc_hd__fa_1 h_u217 (.A(h_c107), .B(h_s108), .CIN(h_s109), .SUM(h_s217), .COUT(h_c217));
    wire h_s218, h_c218; sky130_fd_sc_hd__fa_1 h_u218 (.A(h_s110), .B(h_s111), .CIN(r0[15][36]), .SUM(h_s218), .COUT(h_c218));
    wire h_s219, h_c219; sky130_fd_sc_hd__fa_1 h_u219 (.A(h_c108), .B(h_c109), .CIN(h_c110), .SUM(h_s219), .COUT(h_c219));
    wire h_s220, h_c220; sky130_fd_sc_hd__fa_1 h_u220 (.A(h_c111), .B(h_s112), .CIN(h_s113), .SUM(h_s220), .COUT(h_c220));
    wire h_s221, h_c221; sky130_fd_sc_hd__fa_1 h_u221 (.A(h_s114), .B(h_s115), .CIN(1'b1), .SUM(h_s221), .COUT(h_c221));
    wire h_s222, h_c222; sky130_fd_sc_hd__fa_1 h_u222 (.A(h_c112), .B(h_c113), .CIN(h_c114), .SUM(h_s222), .COUT(h_c222));
    wire h_s223, h_c223; sky130_fd_sc_hd__fa_1 h_u223 (.A(h_c115), .B(h_s116), .CIN(h_s117), .SUM(h_s223), .COUT(h_c223));
    wire h_s224, h_c224; sky130_fd_sc_hd__fa_1 h_u224 (.A(h_s118), .B(h_s119), .CIN(r0[15][38]), .SUM(h_s224), .COUT(h_c224));
    wire h_s225, h_c225; sky130_fd_sc_hd__fa_1 h_u225 (.A(h_c116), .B(h_c117), .CIN(h_c118), .SUM(h_s225), .COUT(h_c225));
    wire h_s226, h_c226; sky130_fd_sc_hd__fa_1 h_u226 (.A(h_c119), .B(h_s120), .CIN(h_s121), .SUM(h_s226), .COUT(h_c226));
    wire h_s227, h_c227; sky130_fd_sc_hd__fa_1 h_u227 (.A(h_s122), .B(h_s123), .CIN(1'b1), .SUM(h_s227), .COUT(h_c227));
    wire h_s228, h_c228; sky130_fd_sc_hd__fa_1 h_u228 (.A(h_c120), .B(h_c121), .CIN(h_c122), .SUM(h_s228), .COUT(h_c228));
    wire h_s229, h_c229; sky130_fd_sc_hd__fa_1 h_u229 (.A(h_c123), .B(h_s124), .CIN(h_s125), .SUM(h_s229), .COUT(h_c229));
    wire h_s230, h_c230; sky130_fd_sc_hd__fa_1 h_u230 (.A(h_s126), .B(h_s127), .CIN(r0[15][40]), .SUM(h_s230), .COUT(h_c230));
    wire h_s231, h_c231; sky130_fd_sc_hd__fa_1 h_u231 (.A(h_c124), .B(h_c125), .CIN(h_c126), .SUM(h_s231), .COUT(h_c231));
    wire h_s232, h_c232; sky130_fd_sc_hd__fa_1 h_u232 (.A(h_c127), .B(h_s128), .CIN(h_s129), .SUM(h_s232), .COUT(h_c232));
    wire h_s233, h_c233; sky130_fd_sc_hd__fa_1 h_u233 (.A(h_s130), .B(h_s131), .CIN(1'b1), .SUM(h_s233), .COUT(h_c233));
    wire h_s234, h_c234; sky130_fd_sc_hd__fa_1 h_u234 (.A(h_c128), .B(h_c129), .CIN(h_c130), .SUM(h_s234), .COUT(h_c234));
    wire h_s235, h_c235; sky130_fd_sc_hd__fa_1 h_u235 (.A(h_c131), .B(h_s132), .CIN(h_s133), .SUM(h_s235), .COUT(h_c235));
    wire h_s236, h_c236; sky130_fd_sc_hd__fa_1 h_u236 (.A(h_s134), .B(r0[14][42]), .CIN(r0[15][42]), .SUM(h_s236), .COUT(h_c236));
    wire h_s237, h_c237; sky130_fd_sc_hd__fa_1 h_u237 (.A(h_c132), .B(h_c133), .CIN(h_c134), .SUM(h_s237), .COUT(h_c237));
    wire h_s238, h_c238; sky130_fd_sc_hd__fa_1 h_u238 (.A(h_s135), .B(h_s136), .CIN(h_s137), .SUM(h_s238), .COUT(h_c238));
    wire h_s239, h_c239; sky130_fd_sc_hd__fa_1 h_u239 (.A(r0[14][43]), .B(r0[15][43]), .CIN(1'b1), .SUM(h_s239), .COUT(h_c239));
    wire h_s240, h_c240; sky130_fd_sc_hd__fa_1 h_u240 (.A(h_c135), .B(h_c136), .CIN(h_c137), .SUM(h_s240), .COUT(h_c240));
    wire h_s241, h_c241; sky130_fd_sc_hd__fa_1 h_u241 (.A(h_s138), .B(h_s139), .CIN(r0[12][44]), .SUM(h_s241), .COUT(h_c241));
    wire h_s242, h_c242; sky130_fd_sc_hd__fa_1 h_u242 (.A(r0[13][44]), .B(r0[14][44]), .CIN(r0[15][44]), .SUM(h_s242), .COUT(h_c242));
    wire h_s243, h_c243; sky130_fd_sc_hd__fa_1 h_u243 (.A(h_c138), .B(h_c139), .CIN(h_s140), .SUM(h_s243), .COUT(h_c243));
    wire h_s244, h_c244; sky130_fd_sc_hd__fa_1 h_u244 (.A(h_s141), .B(r0[12][45]), .CIN(r0[13][45]), .SUM(h_s244), .COUT(h_c244));
    wire h_s245, h_c245; sky130_fd_sc_hd__fa_1 h_u245 (.A(r0[14][45]), .B(r0[15][45]), .CIN(1'b1), .SUM(h_s245), .COUT(h_c245));
    wire h_s246, h_c246; sky130_fd_sc_hd__fa_1 h_u246 (.A(h_c140), .B(h_c141), .CIN(h_s142), .SUM(h_s246), .COUT(h_c246));
    wire h_s247, h_c247; sky130_fd_sc_hd__fa_1 h_u247 (.A(r0[10][46]), .B(r0[11][46]), .CIN(r0[12][46]), .SUM(h_s247), .COUT(h_c247));
    wire h_s248, h_c248; sky130_fd_sc_hd__fa_1 h_u248 (.A(r0[13][46]), .B(r0[14][46]), .CIN(r0[15][46]), .SUM(h_s248), .COUT(h_c248));
    wire h_s249, h_c249; sky130_fd_sc_hd__fa_1 h_u249 (.A(h_c142), .B(h_s143), .CIN(r0[10][47]), .SUM(h_s249), .COUT(h_c249));
    wire h_s250, h_c250; sky130_fd_sc_hd__fa_1 h_u250 (.A(r0[11][47]), .B(r0[12][47]), .CIN(r0[13][47]), .SUM(h_s250), .COUT(h_c250));
    wire h_s251, h_c251; sky130_fd_sc_hd__fa_1 h_u251 (.A(r0[14][47]), .B(r0[15][47]), .CIN(1'b1), .SUM(h_s251), .COUT(h_c251));
    wire h_s252, h_c252; sky130_fd_sc_hd__fa_1 h_u252 (.A(h_c143), .B(r0[8][48]), .CIN(r0[9][48]), .SUM(h_s252), .COUT(h_c252));
    wire h_s253, h_c253; sky130_fd_sc_hd__fa_1 h_u253 (.A(r0[10][48]), .B(r0[11][48]), .CIN(r0[12][48]), .SUM(h_s253), .COUT(h_c253));
    wire h_s254, h_c254; sky130_fd_sc_hd__fa_1 h_u254 (.A(r0[13][48]), .B(r0[14][48]), .CIN(r0[15][48]), .SUM(h_s254), .COUT(h_c254));
    wire h_s255, h_c255; sky130_fd_sc_hd__fa_1 h_u255 (.A(r0[9][49]), .B(r0[10][49]), .CIN(r0[11][49]), .SUM(h_s255), .COUT(h_c255));
    wire h_s256, h_c256; sky130_fd_sc_hd__fa_1 h_u256 (.A(r0[12][49]), .B(r0[13][49]), .CIN(r0[14][49]), .SUM(h_s256), .COUT(h_c256));
    wire h_s257, h_c257; sky130_fd_sc_hd__fa_1 h_u257 (.A(r0[15][49]), .B(1'b1), .CIN(1'b0), .SUM(h_s257), .COUT(h_c257));
    wire h_s258, h_c258; sky130_fd_sc_hd__fa_1 h_u258 (.A(r0[9][50]), .B(r0[10][50]), .CIN(r0[11][50]), .SUM(h_s258), .COUT(h_c258));
    wire h_s259, h_c259; sky130_fd_sc_hd__fa_1 h_u259 (.A(r0[12][50]), .B(r0[13][50]), .CIN(r0[14][50]), .SUM(h_s259), .COUT(h_c259));
    wire h_s260, h_c260; sky130_fd_sc_hd__fa_1 h_u260 (.A(r0[10][51]), .B(r0[11][51]), .CIN(r0[12][51]), .SUM(h_s260), .COUT(h_c260));
    wire h_s261, h_c261; sky130_fd_sc_hd__fa_1 h_u261 (.A(r0[13][51]), .B(r0[14][51]), .CIN(1'b0), .SUM(h_s261), .COUT(h_c261));
    wire h_s262, h_c262; sky130_fd_sc_hd__fa_1 h_u262 (.A(r0[10][52]), .B(r0[11][52]), .CIN(r0[12][52]), .SUM(h_s262), .COUT(h_c262));
    wire h_s263, h_c263; sky130_fd_sc_hd__fa_1 h_u263 (.A(r0[11][53]), .B(r0[12][53]), .CIN(1'b0), .SUM(h_s263), .COUT(h_c263));
    wire h_s264, h_c264; sky130_fd_sc_hd__fa_1 h_u264 (.A(r0[0][6]), .B(r0[1][6]), .CIN(1'b0), .SUM(h_s264), .COUT(h_c264));
    wire h_s265, h_c265; sky130_fd_sc_hd__fa_1 h_u265 (.A(r0[0][7]), .B(r0[1][7]), .CIN(1'b0), .SUM(h_s265), .COUT(h_c265));
    wire h_s266, h_c266; sky130_fd_sc_hd__fa_1 h_u266 (.A(r0[0][8]), .B(r0[1][8]), .CIN(r0[2][8]), .SUM(h_s266), .COUT(h_c266));
    wire h_s267, h_c267; sky130_fd_sc_hd__fa_1 h_u267 (.A(r0[3][8]), .B(r0[4][8]), .CIN(1'b0), .SUM(h_s267), .COUT(h_c267));
    wire h_s268, h_c268; sky130_fd_sc_hd__fa_1 h_u268 (.A(r0[0][9]), .B(r0[1][9]), .CIN(r0[2][9]), .SUM(h_s268), .COUT(h_c268));
    wire h_s269, h_c269; sky130_fd_sc_hd__fa_1 h_u269 (.A(r0[3][9]), .B(r0[4][9]), .CIN(1'b0), .SUM(h_s269), .COUT(h_c269));
    wire h_s270, h_c270; sky130_fd_sc_hd__fa_1 h_u270 (.A(h_s144), .B(r0[2][10]), .CIN(r0[3][10]), .SUM(h_s270), .COUT(h_c270));
    wire h_s271, h_c271; sky130_fd_sc_hd__fa_1 h_u271 (.A(r0[4][10]), .B(r0[5][10]), .CIN(r0[16][10]), .SUM(h_s271), .COUT(h_c271));
    wire h_s272, h_c272; sky130_fd_sc_hd__fa_1 h_u272 (.A(h_c144), .B(h_s145), .CIN(r0[2][11]), .SUM(h_s272), .COUT(h_c272));
    wire h_s273, h_c273; sky130_fd_sc_hd__fa_1 h_u273 (.A(r0[3][11]), .B(r0[4][11]), .CIN(r0[5][11]), .SUM(h_s273), .COUT(h_c273));
    wire h_s274, h_c274; sky130_fd_sc_hd__fa_1 h_u274 (.A(h_c145), .B(h_s146), .CIN(h_s147), .SUM(h_s274), .COUT(h_c274));
    wire h_s275, h_c275; sky130_fd_sc_hd__fa_1 h_u275 (.A(r0[5][12]), .B(r0[6][12]), .CIN(r0[16][12]), .SUM(h_s275), .COUT(h_c275));
    wire h_s276, h_c276; sky130_fd_sc_hd__fa_1 h_u276 (.A(h_c146), .B(h_c147), .CIN(h_s148), .SUM(h_s276), .COUT(h_c276));
    wire h_s277, h_c277; sky130_fd_sc_hd__fa_1 h_u277 (.A(h_s149), .B(r0[5][13]), .CIN(r0[6][13]), .SUM(h_s277), .COUT(h_c277));
    wire h_s278, h_c278; sky130_fd_sc_hd__fa_1 h_u278 (.A(h_c148), .B(h_c149), .CIN(h_s150), .SUM(h_s278), .COUT(h_c278));
    wire h_s279, h_c279; sky130_fd_sc_hd__fa_1 h_u279 (.A(h_s151), .B(h_s152), .CIN(r0[16][14]), .SUM(h_s279), .COUT(h_c279));
    wire h_s280, h_c280; sky130_fd_sc_hd__fa_1 h_u280 (.A(h_c150), .B(h_c151), .CIN(h_c152), .SUM(h_s280), .COUT(h_c280));
    wire h_s281, h_c281; sky130_fd_sc_hd__fa_1 h_u281 (.A(h_s153), .B(h_s154), .CIN(h_s155), .SUM(h_s281), .COUT(h_c281));
    wire h_s282, h_c282; sky130_fd_sc_hd__fa_1 h_u282 (.A(h_c153), .B(h_c154), .CIN(h_c155), .SUM(h_s282), .COUT(h_c282));
    wire h_s283, h_c283; sky130_fd_sc_hd__fa_1 h_u283 (.A(h_s156), .B(h_s157), .CIN(h_s158), .SUM(h_s283), .COUT(h_c283));
    wire h_s284, h_c284; sky130_fd_sc_hd__fa_1 h_u284 (.A(h_c156), .B(h_c157), .CIN(h_c158), .SUM(h_s284), .COUT(h_c284));
    wire h_s285, h_c285; sky130_fd_sc_hd__fa_1 h_u285 (.A(h_s159), .B(h_s160), .CIN(h_s161), .SUM(h_s285), .COUT(h_c285));
    wire h_s286, h_c286; sky130_fd_sc_hd__fa_1 h_u286 (.A(h_c159), .B(h_c160), .CIN(h_c161), .SUM(h_s286), .COUT(h_c286));
    wire h_s287, h_c287; sky130_fd_sc_hd__fa_1 h_u287 (.A(h_s162), .B(h_s163), .CIN(h_s164), .SUM(h_s287), .COUT(h_c287));
    wire h_s288, h_c288; sky130_fd_sc_hd__fa_1 h_u288 (.A(h_c162), .B(h_c163), .CIN(h_c164), .SUM(h_s288), .COUT(h_c288));
    wire h_s289, h_c289; sky130_fd_sc_hd__fa_1 h_u289 (.A(h_s165), .B(h_s166), .CIN(h_s167), .SUM(h_s289), .COUT(h_c289));
    wire h_s290, h_c290; sky130_fd_sc_hd__fa_1 h_u290 (.A(h_c165), .B(h_c166), .CIN(h_c167), .SUM(h_s290), .COUT(h_c290));
    wire h_s291, h_c291; sky130_fd_sc_hd__fa_1 h_u291 (.A(h_s168), .B(h_s169), .CIN(h_s170), .SUM(h_s291), .COUT(h_c291));
    wire h_s292, h_c292; sky130_fd_sc_hd__fa_1 h_u292 (.A(h_c168), .B(h_c169), .CIN(h_c170), .SUM(h_s292), .COUT(h_c292));
    wire h_s293, h_c293; sky130_fd_sc_hd__fa_1 h_u293 (.A(h_s171), .B(h_s172), .CIN(h_s173), .SUM(h_s293), .COUT(h_c293));
    wire h_s294, h_c294; sky130_fd_sc_hd__fa_1 h_u294 (.A(h_c171), .B(h_c172), .CIN(h_c173), .SUM(h_s294), .COUT(h_c294));
    wire h_s295, h_c295; sky130_fd_sc_hd__fa_1 h_u295 (.A(h_s174), .B(h_s175), .CIN(h_s176), .SUM(h_s295), .COUT(h_c295));
    wire h_s296, h_c296; sky130_fd_sc_hd__fa_1 h_u296 (.A(h_c174), .B(h_c175), .CIN(h_c176), .SUM(h_s296), .COUT(h_c296));
    wire h_s297, h_c297; sky130_fd_sc_hd__fa_1 h_u297 (.A(h_s177), .B(h_s178), .CIN(h_s179), .SUM(h_s297), .COUT(h_c297));
    wire h_s298, h_c298; sky130_fd_sc_hd__fa_1 h_u298 (.A(h_c177), .B(h_c178), .CIN(h_c179), .SUM(h_s298), .COUT(h_c298));
    wire h_s299, h_c299; sky130_fd_sc_hd__fa_1 h_u299 (.A(h_s180), .B(h_s181), .CIN(h_s182), .SUM(h_s299), .COUT(h_c299));
    wire h_s300, h_c300; sky130_fd_sc_hd__fa_1 h_u300 (.A(h_c180), .B(h_c181), .CIN(h_c182), .SUM(h_s300), .COUT(h_c300));
    wire h_s301, h_c301; sky130_fd_sc_hd__fa_1 h_u301 (.A(h_s183), .B(h_s184), .CIN(h_s185), .SUM(h_s301), .COUT(h_c301));
    wire h_s302, h_c302; sky130_fd_sc_hd__fa_1 h_u302 (.A(h_c183), .B(h_c184), .CIN(h_c185), .SUM(h_s302), .COUT(h_c302));
    wire h_s303, h_c303; sky130_fd_sc_hd__fa_1 h_u303 (.A(h_s186), .B(h_s187), .CIN(h_s188), .SUM(h_s303), .COUT(h_c303));
    wire h_s304, h_c304; sky130_fd_sc_hd__fa_1 h_u304 (.A(h_c186), .B(h_c187), .CIN(h_c188), .SUM(h_s304), .COUT(h_c304));
    wire h_s305, h_c305; sky130_fd_sc_hd__fa_1 h_u305 (.A(h_s189), .B(h_s190), .CIN(h_s191), .SUM(h_s305), .COUT(h_c305));
    wire h_s306, h_c306; sky130_fd_sc_hd__fa_1 h_u306 (.A(h_c189), .B(h_c190), .CIN(h_c191), .SUM(h_s306), .COUT(h_c306));
    wire h_s307, h_c307; sky130_fd_sc_hd__fa_1 h_u307 (.A(h_s192), .B(h_s193), .CIN(h_s194), .SUM(h_s307), .COUT(h_c307));
    wire h_s308, h_c308; sky130_fd_sc_hd__fa_1 h_u308 (.A(h_c192), .B(h_c193), .CIN(h_c194), .SUM(h_s308), .COUT(h_c308));
    wire h_s309, h_c309; sky130_fd_sc_hd__fa_1 h_u309 (.A(h_s195), .B(h_s196), .CIN(h_s197), .SUM(h_s309), .COUT(h_c309));
    wire h_s310, h_c310; sky130_fd_sc_hd__fa_1 h_u310 (.A(h_c195), .B(h_c196), .CIN(h_c197), .SUM(h_s310), .COUT(h_c310));
    wire h_s311, h_c311; sky130_fd_sc_hd__fa_1 h_u311 (.A(h_s198), .B(h_s199), .CIN(h_s200), .SUM(h_s311), .COUT(h_c311));
    wire h_s312, h_c312; sky130_fd_sc_hd__fa_1 h_u312 (.A(h_c198), .B(h_c199), .CIN(h_c200), .SUM(h_s312), .COUT(h_c312));
    wire h_s313, h_c313; sky130_fd_sc_hd__fa_1 h_u313 (.A(h_s201), .B(h_s202), .CIN(h_s203), .SUM(h_s313), .COUT(h_c313));
    wire h_s314, h_c314; sky130_fd_sc_hd__fa_1 h_u314 (.A(h_c201), .B(h_c202), .CIN(h_c203), .SUM(h_s314), .COUT(h_c314));
    wire h_s315, h_c315; sky130_fd_sc_hd__fa_1 h_u315 (.A(h_s204), .B(h_s205), .CIN(h_s206), .SUM(h_s315), .COUT(h_c315));
    wire h_s316, h_c316; sky130_fd_sc_hd__fa_1 h_u316 (.A(h_c204), .B(h_c205), .CIN(h_c206), .SUM(h_s316), .COUT(h_c316));
    wire h_s317, h_c317; sky130_fd_sc_hd__fa_1 h_u317 (.A(h_s207), .B(h_s208), .CIN(h_s209), .SUM(h_s317), .COUT(h_c317));
    wire h_s318, h_c318; sky130_fd_sc_hd__fa_1 h_u318 (.A(h_c207), .B(h_c208), .CIN(h_c209), .SUM(h_s318), .COUT(h_c318));
    wire h_s319, h_c319; sky130_fd_sc_hd__fa_1 h_u319 (.A(h_s210), .B(h_s211), .CIN(h_s212), .SUM(h_s319), .COUT(h_c319));
    wire h_s320, h_c320; sky130_fd_sc_hd__fa_1 h_u320 (.A(h_c210), .B(h_c211), .CIN(h_c212), .SUM(h_s320), .COUT(h_c320));
    wire h_s321, h_c321; sky130_fd_sc_hd__fa_1 h_u321 (.A(h_s213), .B(h_s214), .CIN(h_s215), .SUM(h_s321), .COUT(h_c321));
    wire h_s322, h_c322; sky130_fd_sc_hd__fa_1 h_u322 (.A(h_c213), .B(h_c214), .CIN(h_c215), .SUM(h_s322), .COUT(h_c322));
    wire h_s323, h_c323; sky130_fd_sc_hd__fa_1 h_u323 (.A(h_s216), .B(h_s217), .CIN(h_s218), .SUM(h_s323), .COUT(h_c323));
    wire h_s324, h_c324; sky130_fd_sc_hd__fa_1 h_u324 (.A(h_c216), .B(h_c217), .CIN(h_c218), .SUM(h_s324), .COUT(h_c324));
    wire h_s325, h_c325; sky130_fd_sc_hd__fa_1 h_u325 (.A(h_s219), .B(h_s220), .CIN(h_s221), .SUM(h_s325), .COUT(h_c325));
    wire h_s326, h_c326; sky130_fd_sc_hd__fa_1 h_u326 (.A(h_c219), .B(h_c220), .CIN(h_c221), .SUM(h_s326), .COUT(h_c326));
    wire h_s327, h_c327; sky130_fd_sc_hd__fa_1 h_u327 (.A(h_s222), .B(h_s223), .CIN(h_s224), .SUM(h_s327), .COUT(h_c327));
    wire h_s328, h_c328; sky130_fd_sc_hd__fa_1 h_u328 (.A(h_c222), .B(h_c223), .CIN(h_c224), .SUM(h_s328), .COUT(h_c328));
    wire h_s329, h_c329; sky130_fd_sc_hd__fa_1 h_u329 (.A(h_s225), .B(h_s226), .CIN(h_s227), .SUM(h_s329), .COUT(h_c329));
    wire h_s330, h_c330; sky130_fd_sc_hd__fa_1 h_u330 (.A(h_c225), .B(h_c226), .CIN(h_c227), .SUM(h_s330), .COUT(h_c330));
    wire h_s331, h_c331; sky130_fd_sc_hd__fa_1 h_u331 (.A(h_s228), .B(h_s229), .CIN(h_s230), .SUM(h_s331), .COUT(h_c331));
    wire h_s332, h_c332; sky130_fd_sc_hd__fa_1 h_u332 (.A(h_c228), .B(h_c229), .CIN(h_c230), .SUM(h_s332), .COUT(h_c332));
    wire h_s333, h_c333; sky130_fd_sc_hd__fa_1 h_u333 (.A(h_s231), .B(h_s232), .CIN(h_s233), .SUM(h_s333), .COUT(h_c333));
    wire h_s334, h_c334; sky130_fd_sc_hd__fa_1 h_u334 (.A(h_c231), .B(h_c232), .CIN(h_c233), .SUM(h_s334), .COUT(h_c334));
    wire h_s335, h_c335; sky130_fd_sc_hd__fa_1 h_u335 (.A(h_s234), .B(h_s235), .CIN(h_s236), .SUM(h_s335), .COUT(h_c335));
    wire h_s336, h_c336; sky130_fd_sc_hd__fa_1 h_u336 (.A(h_c234), .B(h_c235), .CIN(h_c236), .SUM(h_s336), .COUT(h_c336));
    wire h_s337, h_c337; sky130_fd_sc_hd__fa_1 h_u337 (.A(h_s237), .B(h_s238), .CIN(h_s239), .SUM(h_s337), .COUT(h_c337));
    wire h_s338, h_c338; sky130_fd_sc_hd__fa_1 h_u338 (.A(h_c237), .B(h_c238), .CIN(h_c239), .SUM(h_s338), .COUT(h_c338));
    wire h_s339, h_c339; sky130_fd_sc_hd__fa_1 h_u339 (.A(h_s240), .B(h_s241), .CIN(h_s242), .SUM(h_s339), .COUT(h_c339));
    wire h_s340, h_c340; sky130_fd_sc_hd__fa_1 h_u340 (.A(h_c240), .B(h_c241), .CIN(h_c242), .SUM(h_s340), .COUT(h_c340));
    wire h_s341, h_c341; sky130_fd_sc_hd__fa_1 h_u341 (.A(h_s243), .B(h_s244), .CIN(h_s245), .SUM(h_s341), .COUT(h_c341));
    wire h_s342, h_c342; sky130_fd_sc_hd__fa_1 h_u342 (.A(h_c243), .B(h_c244), .CIN(h_c245), .SUM(h_s342), .COUT(h_c342));
    wire h_s343, h_c343; sky130_fd_sc_hd__fa_1 h_u343 (.A(h_s246), .B(h_s247), .CIN(h_s248), .SUM(h_s343), .COUT(h_c343));
    wire h_s344, h_c344; sky130_fd_sc_hd__fa_1 h_u344 (.A(h_c246), .B(h_c247), .CIN(h_c248), .SUM(h_s344), .COUT(h_c344));
    wire h_s345, h_c345; sky130_fd_sc_hd__fa_1 h_u345 (.A(h_s249), .B(h_s250), .CIN(h_s251), .SUM(h_s345), .COUT(h_c345));
    wire h_s346, h_c346; sky130_fd_sc_hd__fa_1 h_u346 (.A(h_c249), .B(h_c250), .CIN(h_c251), .SUM(h_s346), .COUT(h_c346));
    wire h_s347, h_c347; sky130_fd_sc_hd__fa_1 h_u347 (.A(h_s252), .B(h_s253), .CIN(h_s254), .SUM(h_s347), .COUT(h_c347));
    wire h_s348, h_c348; sky130_fd_sc_hd__fa_1 h_u348 (.A(h_c252), .B(h_c253), .CIN(h_c254), .SUM(h_s348), .COUT(h_c348));
    wire h_s349, h_c349; sky130_fd_sc_hd__fa_1 h_u349 (.A(h_s255), .B(h_s256), .CIN(h_s257), .SUM(h_s349), .COUT(h_c349));
    wire h_s350, h_c350; sky130_fd_sc_hd__fa_1 h_u350 (.A(h_c255), .B(h_c256), .CIN(h_c257), .SUM(h_s350), .COUT(h_c350));
    wire h_s351, h_c351; sky130_fd_sc_hd__fa_1 h_u351 (.A(h_s258), .B(h_s259), .CIN(r0[15][50]), .SUM(h_s351), .COUT(h_c351));
    wire h_s352, h_c352; sky130_fd_sc_hd__fa_1 h_u352 (.A(h_c258), .B(h_c259), .CIN(h_s260), .SUM(h_s352), .COUT(h_c352));
    wire h_s353, h_c353; sky130_fd_sc_hd__fa_1 h_u353 (.A(h_s261), .B(r0[15][51]), .CIN(1'b1), .SUM(h_s353), .COUT(h_c353));
    wire h_s354, h_c354; sky130_fd_sc_hd__fa_1 h_u354 (.A(h_c260), .B(h_c261), .CIN(h_s262), .SUM(h_s354), .COUT(h_c354));
    wire h_s355, h_c355; sky130_fd_sc_hd__fa_1 h_u355 (.A(r0[13][52]), .B(r0[14][52]), .CIN(r0[15][52]), .SUM(h_s355), .COUT(h_c355));
    wire h_s356, h_c356; sky130_fd_sc_hd__fa_1 h_u356 (.A(h_c262), .B(h_s263), .CIN(r0[13][53]), .SUM(h_s356), .COUT(h_c356));
    wire h_s357, h_c357; sky130_fd_sc_hd__fa_1 h_u357 (.A(r0[14][53]), .B(r0[15][53]), .CIN(1'b1), .SUM(h_s357), .COUT(h_c357));
    wire h_s358, h_c358; sky130_fd_sc_hd__fa_1 h_u358 (.A(h_c263), .B(r0[11][54]), .CIN(r0[12][54]), .SUM(h_s358), .COUT(h_c358));
    wire h_s359, h_c359; sky130_fd_sc_hd__fa_1 h_u359 (.A(r0[13][54]), .B(r0[14][54]), .CIN(r0[15][54]), .SUM(h_s359), .COUT(h_c359));
    wire h_s360, h_c360; sky130_fd_sc_hd__fa_1 h_u360 (.A(r0[12][55]), .B(r0[13][55]), .CIN(r0[14][55]), .SUM(h_s360), .COUT(h_c360));
    wire h_s361, h_c361; sky130_fd_sc_hd__fa_1 h_u361 (.A(r0[15][55]), .B(1'b1), .CIN(1'b0), .SUM(h_s361), .COUT(h_c361));
    wire h_s362, h_c362; sky130_fd_sc_hd__fa_1 h_u362 (.A(r0[12][56]), .B(r0[13][56]), .CIN(r0[14][56]), .SUM(h_s362), .COUT(h_c362));
    wire h_s363, h_c363; sky130_fd_sc_hd__fa_1 h_u363 (.A(r0[13][57]), .B(r0[14][57]), .CIN(1'b0), .SUM(h_s363), .COUT(h_c363));
    wire h_s364, h_c364; sky130_fd_sc_hd__fa_1 h_u364 (.A(r0[0][4]), .B(r0[1][4]), .CIN(1'b0), .SUM(h_s364), .COUT(h_c364));
    wire h_s365, h_c365; sky130_fd_sc_hd__fa_1 h_u365 (.A(r0[0][5]), .B(r0[1][5]), .CIN(1'b0), .SUM(h_s365), .COUT(h_c365));
    wire h_s366, h_c366; sky130_fd_sc_hd__fa_1 h_u366 (.A(h_s264), .B(r0[2][6]), .CIN(r0[3][6]), .SUM(h_s366), .COUT(h_c366));
    wire h_s367, h_c367; sky130_fd_sc_hd__fa_1 h_u367 (.A(h_c264), .B(h_s265), .CIN(r0[2][7]), .SUM(h_s367), .COUT(h_c367));
    wire h_s368, h_c368; sky130_fd_sc_hd__fa_1 h_u368 (.A(h_c265), .B(h_s266), .CIN(h_s267), .SUM(h_s368), .COUT(h_c368));
    wire h_s369, h_c369; sky130_fd_sc_hd__fa_1 h_u369 (.A(h_c266), .B(h_c267), .CIN(h_s268), .SUM(h_s369), .COUT(h_c369));
    wire h_s370, h_c370; sky130_fd_sc_hd__fa_1 h_u370 (.A(h_c268), .B(h_c269), .CIN(h_s270), .SUM(h_s370), .COUT(h_c370));
    wire h_s371, h_c371; sky130_fd_sc_hd__fa_1 h_u371 (.A(h_c270), .B(h_c271), .CIN(h_s272), .SUM(h_s371), .COUT(h_c371));
    wire h_s372, h_c372; sky130_fd_sc_hd__fa_1 h_u372 (.A(h_c272), .B(h_c273), .CIN(h_s274), .SUM(h_s372), .COUT(h_c372));
    wire h_s373, h_c373; sky130_fd_sc_hd__fa_1 h_u373 (.A(h_c274), .B(h_c275), .CIN(h_s276), .SUM(h_s373), .COUT(h_c373));
    wire h_s374, h_c374; sky130_fd_sc_hd__fa_1 h_u374 (.A(h_c276), .B(h_c277), .CIN(h_s278), .SUM(h_s374), .COUT(h_c374));
    wire h_s375, h_c375; sky130_fd_sc_hd__fa_1 h_u375 (.A(h_c278), .B(h_c279), .CIN(h_s280), .SUM(h_s375), .COUT(h_c375));
    wire h_s376, h_c376; sky130_fd_sc_hd__fa_1 h_u376 (.A(h_c280), .B(h_c281), .CIN(h_s282), .SUM(h_s376), .COUT(h_c376));
    wire h_s377, h_c377; sky130_fd_sc_hd__fa_1 h_u377 (.A(h_c282), .B(h_c283), .CIN(h_s284), .SUM(h_s377), .COUT(h_c377));
    wire h_s378, h_c378; sky130_fd_sc_hd__fa_1 h_u378 (.A(h_c284), .B(h_c285), .CIN(h_s286), .SUM(h_s378), .COUT(h_c378));
    wire h_s379, h_c379; sky130_fd_sc_hd__fa_1 h_u379 (.A(h_c286), .B(h_c287), .CIN(h_s288), .SUM(h_s379), .COUT(h_c379));
    wire h_s380, h_c380; sky130_fd_sc_hd__fa_1 h_u380 (.A(h_c288), .B(h_c289), .CIN(h_s290), .SUM(h_s380), .COUT(h_c380));
    wire h_s381, h_c381; sky130_fd_sc_hd__fa_1 h_u381 (.A(h_c290), .B(h_c291), .CIN(h_s292), .SUM(h_s381), .COUT(h_c381));
    wire h_s382, h_c382; sky130_fd_sc_hd__fa_1 h_u382 (.A(h_c292), .B(h_c293), .CIN(h_s294), .SUM(h_s382), .COUT(h_c382));
    wire h_s383, h_c383; sky130_fd_sc_hd__fa_1 h_u383 (.A(h_c294), .B(h_c295), .CIN(h_s296), .SUM(h_s383), .COUT(h_c383));
    wire h_s384, h_c384; sky130_fd_sc_hd__fa_1 h_u384 (.A(h_c296), .B(h_c297), .CIN(h_s298), .SUM(h_s384), .COUT(h_c384));
    wire h_s385, h_c385; sky130_fd_sc_hd__fa_1 h_u385 (.A(h_c298), .B(h_c299), .CIN(h_s300), .SUM(h_s385), .COUT(h_c385));
    wire h_s386, h_c386; sky130_fd_sc_hd__fa_1 h_u386 (.A(h_c300), .B(h_c301), .CIN(h_s302), .SUM(h_s386), .COUT(h_c386));
    wire h_s387, h_c387; sky130_fd_sc_hd__fa_1 h_u387 (.A(h_c302), .B(h_c303), .CIN(h_s304), .SUM(h_s387), .COUT(h_c387));
    wire h_s388, h_c388; sky130_fd_sc_hd__fa_1 h_u388 (.A(h_c304), .B(h_c305), .CIN(h_s306), .SUM(h_s388), .COUT(h_c388));
    wire h_s389, h_c389; sky130_fd_sc_hd__fa_1 h_u389 (.A(h_c306), .B(h_c307), .CIN(h_s308), .SUM(h_s389), .COUT(h_c389));
    wire h_s390, h_c390; sky130_fd_sc_hd__fa_1 h_u390 (.A(h_c308), .B(h_c309), .CIN(h_s310), .SUM(h_s390), .COUT(h_c390));
    wire h_s391, h_c391; sky130_fd_sc_hd__fa_1 h_u391 (.A(h_c310), .B(h_c311), .CIN(h_s312), .SUM(h_s391), .COUT(h_c391));
    wire h_s392, h_c392; sky130_fd_sc_hd__fa_1 h_u392 (.A(h_c312), .B(h_c313), .CIN(h_s314), .SUM(h_s392), .COUT(h_c392));
    wire h_s393, h_c393; sky130_fd_sc_hd__fa_1 h_u393 (.A(h_c314), .B(h_c315), .CIN(h_s316), .SUM(h_s393), .COUT(h_c393));
    wire h_s394, h_c394; sky130_fd_sc_hd__fa_1 h_u394 (.A(h_c316), .B(h_c317), .CIN(h_s318), .SUM(h_s394), .COUT(h_c394));
    wire h_s395, h_c395; sky130_fd_sc_hd__fa_1 h_u395 (.A(h_c318), .B(h_c319), .CIN(h_s320), .SUM(h_s395), .COUT(h_c395));
    wire h_s396, h_c396; sky130_fd_sc_hd__fa_1 h_u396 (.A(h_c320), .B(h_c321), .CIN(h_s322), .SUM(h_s396), .COUT(h_c396));
    wire h_s397, h_c397; sky130_fd_sc_hd__fa_1 h_u397 (.A(h_c322), .B(h_c323), .CIN(h_s324), .SUM(h_s397), .COUT(h_c397));
    wire h_s398, h_c398; sky130_fd_sc_hd__fa_1 h_u398 (.A(h_c324), .B(h_c325), .CIN(h_s326), .SUM(h_s398), .COUT(h_c398));
    wire h_s399, h_c399; sky130_fd_sc_hd__fa_1 h_u399 (.A(h_c326), .B(h_c327), .CIN(h_s328), .SUM(h_s399), .COUT(h_c399));
    wire h_s400, h_c400; sky130_fd_sc_hd__fa_1 h_u400 (.A(h_c328), .B(h_c329), .CIN(h_s330), .SUM(h_s400), .COUT(h_c400));
    wire h_s401, h_c401; sky130_fd_sc_hd__fa_1 h_u401 (.A(h_c330), .B(h_c331), .CIN(h_s332), .SUM(h_s401), .COUT(h_c401));
    wire h_s402, h_c402; sky130_fd_sc_hd__fa_1 h_u402 (.A(h_c332), .B(h_c333), .CIN(h_s334), .SUM(h_s402), .COUT(h_c402));
    wire h_s403, h_c403; sky130_fd_sc_hd__fa_1 h_u403 (.A(h_c334), .B(h_c335), .CIN(h_s336), .SUM(h_s403), .COUT(h_c403));
    wire h_s404, h_c404; sky130_fd_sc_hd__fa_1 h_u404 (.A(h_c336), .B(h_c337), .CIN(h_s338), .SUM(h_s404), .COUT(h_c404));
    wire h_s405, h_c405; sky130_fd_sc_hd__fa_1 h_u405 (.A(h_c338), .B(h_c339), .CIN(h_s340), .SUM(h_s405), .COUT(h_c405));
    wire h_s406, h_c406; sky130_fd_sc_hd__fa_1 h_u406 (.A(h_c340), .B(h_c341), .CIN(h_s342), .SUM(h_s406), .COUT(h_c406));
    wire h_s407, h_c407; sky130_fd_sc_hd__fa_1 h_u407 (.A(h_c342), .B(h_c343), .CIN(h_s344), .SUM(h_s407), .COUT(h_c407));
    wire h_s408, h_c408; sky130_fd_sc_hd__fa_1 h_u408 (.A(h_c344), .B(h_c345), .CIN(h_s346), .SUM(h_s408), .COUT(h_c408));
    wire h_s409, h_c409; sky130_fd_sc_hd__fa_1 h_u409 (.A(h_c346), .B(h_c347), .CIN(h_s348), .SUM(h_s409), .COUT(h_c409));
    wire h_s410, h_c410; sky130_fd_sc_hd__fa_1 h_u410 (.A(h_c348), .B(h_c349), .CIN(h_s350), .SUM(h_s410), .COUT(h_c410));
    wire h_s411, h_c411; sky130_fd_sc_hd__fa_1 h_u411 (.A(h_c350), .B(h_c351), .CIN(h_s352), .SUM(h_s411), .COUT(h_c411));
    wire h_s412, h_c412; sky130_fd_sc_hd__fa_1 h_u412 (.A(h_c352), .B(h_c353), .CIN(h_s354), .SUM(h_s412), .COUT(h_c412));
    wire h_s413, h_c413; sky130_fd_sc_hd__fa_1 h_u413 (.A(h_c354), .B(h_c355), .CIN(h_s356), .SUM(h_s413), .COUT(h_c413));
    wire h_s414, h_c414; sky130_fd_sc_hd__fa_1 h_u414 (.A(h_c356), .B(h_c357), .CIN(h_s358), .SUM(h_s414), .COUT(h_c414));
    wire h_s415, h_c415; sky130_fd_sc_hd__fa_1 h_u415 (.A(h_c358), .B(h_c359), .CIN(h_s360), .SUM(h_s415), .COUT(h_c415));
    wire h_s416, h_c416; sky130_fd_sc_hd__fa_1 h_u416 (.A(h_c360), .B(h_c361), .CIN(h_s362), .SUM(h_s416), .COUT(h_c416));
    wire h_s417, h_c417; sky130_fd_sc_hd__fa_1 h_u417 (.A(h_c362), .B(h_s363), .CIN(r0[15][57]), .SUM(h_s417), .COUT(h_c417));
    wire h_s418, h_c418; sky130_fd_sc_hd__fa_1 h_u418 (.A(h_c363), .B(r0[13][58]), .CIN(r0[14][58]), .SUM(h_s418), .COUT(h_c418));
    wire h_s419, h_c419; sky130_fd_sc_hd__fa_1 h_u419 (.A(r0[14][59]), .B(r0[15][59]), .CIN(1'b0), .SUM(h_s419), .COUT(h_c419));
    wire h_s420, h_c420; sky130_fd_sc_hd__fa_1 h_u420 (.A(r0[0][2]), .B(r0[1][2]), .CIN(1'b0), .SUM(h_s420), .COUT(h_c420));
    wire h_s421, h_c421; sky130_fd_sc_hd__fa_1 h_u421 (.A(r0[0][3]), .B(r0[1][3]), .CIN(1'b0), .SUM(h_s421), .COUT(h_c421));
    wire h_s422, h_c422; sky130_fd_sc_hd__fa_1 h_u422 (.A(h_s364), .B(r0[2][4]), .CIN(r0[16][4]), .SUM(h_s422), .COUT(h_c422));
    wire h_s423, h_c423; sky130_fd_sc_hd__fa_1 h_u423 (.A(h_c364), .B(h_s365), .CIN(r0[2][5]), .SUM(h_s423), .COUT(h_c423));
    wire h_s424, h_c424; sky130_fd_sc_hd__fa_1 h_u424 (.A(h_c365), .B(h_s366), .CIN(r0[16][6]), .SUM(h_s424), .COUT(h_c424));
    wire h_s425, h_c425; sky130_fd_sc_hd__fa_1 h_u425 (.A(h_c366), .B(h_s367), .CIN(r0[3][7]), .SUM(h_s425), .COUT(h_c425));
    wire h_s426, h_c426; sky130_fd_sc_hd__fa_1 h_u426 (.A(h_c367), .B(h_s368), .CIN(r0[16][8]), .SUM(h_s426), .COUT(h_c426));
    wire h_s427, h_c427; sky130_fd_sc_hd__fa_1 h_u427 (.A(h_c368), .B(h_s369), .CIN(h_s269), .SUM(h_s427), .COUT(h_c427));
    wire h_s428, h_c428; sky130_fd_sc_hd__fa_1 h_u428 (.A(h_c369), .B(h_s370), .CIN(h_s271), .SUM(h_s428), .COUT(h_c428));
    wire h_s429, h_c429; sky130_fd_sc_hd__fa_1 h_u429 (.A(h_c370), .B(h_s371), .CIN(h_s273), .SUM(h_s429), .COUT(h_c429));
    wire h_s430, h_c430; sky130_fd_sc_hd__fa_1 h_u430 (.A(h_c371), .B(h_s372), .CIN(h_s275), .SUM(h_s430), .COUT(h_c430));
    wire h_s431, h_c431; sky130_fd_sc_hd__fa_1 h_u431 (.A(h_c372), .B(h_s373), .CIN(h_s277), .SUM(h_s431), .COUT(h_c431));
    wire h_s432, h_c432; sky130_fd_sc_hd__fa_1 h_u432 (.A(h_c373), .B(h_s374), .CIN(h_s279), .SUM(h_s432), .COUT(h_c432));
    wire h_s433, h_c433; sky130_fd_sc_hd__fa_1 h_u433 (.A(h_c374), .B(h_s375), .CIN(h_s281), .SUM(h_s433), .COUT(h_c433));
    wire h_s434, h_c434; sky130_fd_sc_hd__fa_1 h_u434 (.A(h_c375), .B(h_s376), .CIN(h_s283), .SUM(h_s434), .COUT(h_c434));
    wire h_s435, h_c435; sky130_fd_sc_hd__fa_1 h_u435 (.A(h_c376), .B(h_s377), .CIN(h_s285), .SUM(h_s435), .COUT(h_c435));
    wire h_s436, h_c436; sky130_fd_sc_hd__fa_1 h_u436 (.A(h_c377), .B(h_s378), .CIN(h_s287), .SUM(h_s436), .COUT(h_c436));
    wire h_s437, h_c437; sky130_fd_sc_hd__fa_1 h_u437 (.A(h_c378), .B(h_s379), .CIN(h_s289), .SUM(h_s437), .COUT(h_c437));
    wire h_s438, h_c438; sky130_fd_sc_hd__fa_1 h_u438 (.A(h_c379), .B(h_s380), .CIN(h_s291), .SUM(h_s438), .COUT(h_c438));
    wire h_s439, h_c439; sky130_fd_sc_hd__fa_1 h_u439 (.A(h_c380), .B(h_s381), .CIN(h_s293), .SUM(h_s439), .COUT(h_c439));
    wire h_s440, h_c440; sky130_fd_sc_hd__fa_1 h_u440 (.A(h_c381), .B(h_s382), .CIN(h_s295), .SUM(h_s440), .COUT(h_c440));
    wire h_s441, h_c441; sky130_fd_sc_hd__fa_1 h_u441 (.A(h_c382), .B(h_s383), .CIN(h_s297), .SUM(h_s441), .COUT(h_c441));
    wire h_s442, h_c442; sky130_fd_sc_hd__fa_1 h_u442 (.A(h_c383), .B(h_s384), .CIN(h_s299), .SUM(h_s442), .COUT(h_c442));
    wire h_s443, h_c443; sky130_fd_sc_hd__fa_1 h_u443 (.A(h_c384), .B(h_s385), .CIN(h_s301), .SUM(h_s443), .COUT(h_c443));
    wire h_s444, h_c444; sky130_fd_sc_hd__fa_1 h_u444 (.A(h_c385), .B(h_s386), .CIN(h_s303), .SUM(h_s444), .COUT(h_c444));
    wire h_s445, h_c445; sky130_fd_sc_hd__fa_1 h_u445 (.A(h_c386), .B(h_s387), .CIN(h_s305), .SUM(h_s445), .COUT(h_c445));
    wire h_s446, h_c446; sky130_fd_sc_hd__fa_1 h_u446 (.A(h_c387), .B(h_s388), .CIN(h_s307), .SUM(h_s446), .COUT(h_c446));
    wire h_s447, h_c447; sky130_fd_sc_hd__fa_1 h_u447 (.A(h_c388), .B(h_s389), .CIN(h_s309), .SUM(h_s447), .COUT(h_c447));
    wire h_s448, h_c448; sky130_fd_sc_hd__fa_1 h_u448 (.A(h_c389), .B(h_s390), .CIN(h_s311), .SUM(h_s448), .COUT(h_c448));
    wire h_s449, h_c449; sky130_fd_sc_hd__fa_1 h_u449 (.A(h_c390), .B(h_s391), .CIN(h_s313), .SUM(h_s449), .COUT(h_c449));
    wire h_s450, h_c450; sky130_fd_sc_hd__fa_1 h_u450 (.A(h_c391), .B(h_s392), .CIN(h_s315), .SUM(h_s450), .COUT(h_c450));
    wire h_s451, h_c451; sky130_fd_sc_hd__fa_1 h_u451 (.A(h_c392), .B(h_s393), .CIN(h_s317), .SUM(h_s451), .COUT(h_c451));
    wire h_s452, h_c452; sky130_fd_sc_hd__fa_1 h_u452 (.A(h_c393), .B(h_s394), .CIN(h_s319), .SUM(h_s452), .COUT(h_c452));
    wire h_s453, h_c453; sky130_fd_sc_hd__fa_1 h_u453 (.A(h_c394), .B(h_s395), .CIN(h_s321), .SUM(h_s453), .COUT(h_c453));
    wire h_s454, h_c454; sky130_fd_sc_hd__fa_1 h_u454 (.A(h_c395), .B(h_s396), .CIN(h_s323), .SUM(h_s454), .COUT(h_c454));
    wire h_s455, h_c455; sky130_fd_sc_hd__fa_1 h_u455 (.A(h_c396), .B(h_s397), .CIN(h_s325), .SUM(h_s455), .COUT(h_c455));
    wire h_s456, h_c456; sky130_fd_sc_hd__fa_1 h_u456 (.A(h_c397), .B(h_s398), .CIN(h_s327), .SUM(h_s456), .COUT(h_c456));
    wire h_s457, h_c457; sky130_fd_sc_hd__fa_1 h_u457 (.A(h_c398), .B(h_s399), .CIN(h_s329), .SUM(h_s457), .COUT(h_c457));
    wire h_s458, h_c458; sky130_fd_sc_hd__fa_1 h_u458 (.A(h_c399), .B(h_s400), .CIN(h_s331), .SUM(h_s458), .COUT(h_c458));
    wire h_s459, h_c459; sky130_fd_sc_hd__fa_1 h_u459 (.A(h_c400), .B(h_s401), .CIN(h_s333), .SUM(h_s459), .COUT(h_c459));
    wire h_s460, h_c460; sky130_fd_sc_hd__fa_1 h_u460 (.A(h_c401), .B(h_s402), .CIN(h_s335), .SUM(h_s460), .COUT(h_c460));
    wire h_s461, h_c461; sky130_fd_sc_hd__fa_1 h_u461 (.A(h_c402), .B(h_s403), .CIN(h_s337), .SUM(h_s461), .COUT(h_c461));
    wire h_s462, h_c462; sky130_fd_sc_hd__fa_1 h_u462 (.A(h_c403), .B(h_s404), .CIN(h_s339), .SUM(h_s462), .COUT(h_c462));
    wire h_s463, h_c463; sky130_fd_sc_hd__fa_1 h_u463 (.A(h_c404), .B(h_s405), .CIN(h_s341), .SUM(h_s463), .COUT(h_c463));
    wire h_s464, h_c464; sky130_fd_sc_hd__fa_1 h_u464 (.A(h_c405), .B(h_s406), .CIN(h_s343), .SUM(h_s464), .COUT(h_c464));
    wire h_s465, h_c465; sky130_fd_sc_hd__fa_1 h_u465 (.A(h_c406), .B(h_s407), .CIN(h_s345), .SUM(h_s465), .COUT(h_c465));
    wire h_s466, h_c466; sky130_fd_sc_hd__fa_1 h_u466 (.A(h_c407), .B(h_s408), .CIN(h_s347), .SUM(h_s466), .COUT(h_c466));
    wire h_s467, h_c467; sky130_fd_sc_hd__fa_1 h_u467 (.A(h_c408), .B(h_s409), .CIN(h_s349), .SUM(h_s467), .COUT(h_c467));
    wire h_s468, h_c468; sky130_fd_sc_hd__fa_1 h_u468 (.A(h_c409), .B(h_s410), .CIN(h_s351), .SUM(h_s468), .COUT(h_c468));
    wire h_s469, h_c469; sky130_fd_sc_hd__fa_1 h_u469 (.A(h_c410), .B(h_s411), .CIN(h_s353), .SUM(h_s469), .COUT(h_c469));
    wire h_s470, h_c470; sky130_fd_sc_hd__fa_1 h_u470 (.A(h_c411), .B(h_s412), .CIN(h_s355), .SUM(h_s470), .COUT(h_c470));
    wire h_s471, h_c471; sky130_fd_sc_hd__fa_1 h_u471 (.A(h_c412), .B(h_s413), .CIN(h_s357), .SUM(h_s471), .COUT(h_c471));
    wire h_s472, h_c472; sky130_fd_sc_hd__fa_1 h_u472 (.A(h_c413), .B(h_s414), .CIN(h_s359), .SUM(h_s472), .COUT(h_c472));
    wire h_s473, h_c473; sky130_fd_sc_hd__fa_1 h_u473 (.A(h_c414), .B(h_s415), .CIN(h_s361), .SUM(h_s473), .COUT(h_c473));
    wire h_s474, h_c474; sky130_fd_sc_hd__fa_1 h_u474 (.A(h_c415), .B(h_s416), .CIN(r0[15][56]), .SUM(h_s474), .COUT(h_c474));
    wire h_s475, h_c475; sky130_fd_sc_hd__fa_1 h_u475 (.A(h_c416), .B(h_s417), .CIN(1'b1), .SUM(h_s475), .COUT(h_c475));
    wire h_s476, h_c476; sky130_fd_sc_hd__fa_1 h_u476 (.A(h_c417), .B(h_s418), .CIN(r0[15][58]), .SUM(h_s476), .COUT(h_c476));
    wire h_s477, h_c477; sky130_fd_sc_hd__fa_1 h_u477 (.A(h_c418), .B(h_s419), .CIN(1'b1), .SUM(h_s477), .COUT(h_c477));
    wire h_s478, h_c478; sky130_fd_sc_hd__fa_1 h_u478 (.A(h_c419), .B(r0[14][60]), .CIN(r0[15][60]), .SUM(h_s478), .COUT(h_c478));
    wire h_s479, h_c479; sky130_fd_sc_hd__fa_1 h_u479 (.A(r0[15][61]), .B(1'b1), .CIN(1'b0), .SUM(h_s479), .COUT(h_c479));
    assign hs = {h_c479, h_c478, h_c477, h_c476, h_c475, h_c474, h_c473, h_c472, h_c471, h_c470, h_c469, h_c468, h_c467, h_c466, h_c465, h_c464, h_c463, h_c462, h_c461, h_c460, h_c459, h_c458, h_c457, h_c456, h_c455, h_c454, h_c453, h_c452, h_c451, h_c450, h_c449, h_c448, h_c447, h_c446, h_c445, h_c444, h_c443, h_c442, h_c441, h_c440, h_c439, h_c438, h_c437, h_c436, h_c435, h_c434, h_c433, h_c432, h_c431, h_c430, h_c429, h_c428, h_c427, h_c426, h_c425, h_c424, h_c423, h_c422, h_c421, h_c420, h_s420, r0[0][1], r0[0][0]};
    assign hc = {r0[15][62], h_s479, h_s478, h_s477, h_s476, h_s475, h_s474, h_s473, h_s472, h_s471, h_s470, h_s469, h_s468, h_s467, h_s466, h_s465, h_s464, h_s463, h_s462, h_s461, h_s460, h_s459, h_s458, h_s457, h_s456, h_s455, h_s454, h_s453, h_s452, h_s451, h_s450, h_s449, h_s448, h_s447, h_s446, h_s445, h_s444, h_s443, h_s442, h_s441, h_s440, h_s439, h_s438, h_s437, h_s436, h_s435, h_s434, h_s433, h_s432, h_s431, h_s430, h_s429, h_s428, h_s427, h_s426, h_s425, h_s424, h_s423, h_s422, h_s421, r0[16][2], 1'b0, r0[16][0]};
    logic [PW-1:0] s0n, c0n, s1p, c1p;
    always_ff @(negedge clk) begin s0n <= hs; c0n <= hc; end    // set 0 (high phase)
    always_ff @(posedge clk) begin s1p <= hs; c1p <= hc; end    // set 1 (low phase)

    // ---- set 0: the carry out of its low 31 bits is formed in the low phase (v26A: a 31-bit add in a
    //      half cycle), only the kept bits [62:31] are re-registered at t2; then, as set 1, the
    //      carry-propagate add + RoundingDivideByPOT -> t3 ----
    wire  [31:0]   lo0 = {1'b0, s0n[30:0]} + {1'b0, c0n[30:0]};
    logic [31:0]   s0p, c0p;
    logic          k0p;
    logic [4:0]    r0d, r1d;
    always_ff @(posedge clk) begin
        s0p <= s0n[62:31];
        c0p <= c0n[62:31];
        k0p <= lo0[31];
        r0d <= r0q;
        r1d <= r1q;
    end
    wire  [31:0]   p0h = s0p + c0p + {31'd0, k0p};
    wire  [PW-1:0] p1 = s1p + c1p;
    logic [11:0]   h0, h1;
    always_ff @(posedge clk) begin
        h0 <= rdq(p0h, r0d);
        h1 <= rdq(p1[62:31], r1d);
    end
    // ---- t3 -> t4: offset, clamp ----
    always_ff @(posedge clk) begin
        out_q0 <= outq(h0, ooff0, amin0, amax0);
        out_q1 <= outq(h1, ooff1, amin1, amax1);
    end
endmodule

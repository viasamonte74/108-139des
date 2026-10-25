// npu_drq2 - two npu_drq channels (conv mode, asc = 0) on one double-pumped multiplier.
//   Same function and latency as two npu_drq: out_q{0,1} = clamp(MultiplyByQuantizedMultiplier(
//   acc, mult, shift) + ooff), exact for mult == 0 or 2^30 <= mult < 2^31 with -11 <= shift <= 0
//   (acc saturated to 22 bits, see npu_drq); out_valid{0,1} four cycles after in_valid{0,1}.
//   Pipeline: c0 saturation + Booth recoding, registered (t1) | H1: radix-4 Booth partial
//   products + a 3:2 tree to sum / carry vectors, one operand set per clock phase (set 0 in the
//   high phase, captured at the negedge; set 1 in the low phase, captured at the posedge) |
//   set 0: carry-propagate add in the next low phase (t2), RoundingDivideByPOT (t3);
//   set 1: carry-propagate add + RoundingDivideByPOT (t3) | saturate, offset, clamp (t4).
//   Every path but H1 and set 0's add is a full cycle; those two are half cycles (~6.2 ns).
module npu_drq2 (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        in_valid0,
    input  logic        in_valid1,
    input  logic [31:0] acc0,
    input  logic [31:0] acc1,
    input  logic [30:0] mult0,
    input  logic [30:0] mult1,
    input  logic [7:0]  shift0,
    input  logic [7:0]  shift1,
    input  logic [8:0]  ooff,
    input  logic [7:0]  amin,
    input  logic [7:0]  amax,
    output logic        out_valid0,
    output logic        out_valid1,
    output logic [7:0]  out_q0,
    output logic [7:0]  out_q1
);
    localparam int W  = 22;
    localparam int PW = W + 31;            // product bits 0 .. W + 30 (only [W+30:31] are kept)
    localparam int NC = 1;                 // operand copies (each feeds 16 / NC Booth rows)
    // Booth rows are added as {~s, low W bits} = row + 2^W: K0 removes those offsets; + the
    // SRDHM nudge 2^30
    function automatic logic [PW-1:0] k0f();
        k0f = '0;
        for (int k = 0; k < 16; k++) k0f = k0f - (PW'(1) << (W + 2 * k));
        k0f = k0f + (PW'(1) << 30);
    endfunction
    localparam logic [PW-1:0] K0 = k0f();
    function automatic logic [W-1:0] sat22(input logic [31:0] x);
        sat22 = (x[31:W-1] == {(33-W){x[31]}}) ? x[W-1:0] : {x[31], {(W-1){~x[31]}}};
    endfunction
    // RoundingDivideByPOT by r (as npu_drq stage 2) of the W-bit product, then saturated to
    // [-512, 511] (npu_drq stage 3): r <= 11 for a nonzero multiplier, and a zero multiplier gives
    // y = 0 for every r, so 4 bits of r suffice
    function automatic logic [10:0] rdsat(input logic [W-1:0] y, input logic [3:0] r);
        logic [11:0]  mask, rem, thr;      // (v27F: r <= 11 -> 12-bit remainder / threshold, as v26A s13)
        logic [W-1:0] ysh, z;
        mask = (12'd1 << r) - 12'd1;
        rem  = y[11:0] & mask;
        thr  = (mask >> 1) + {11'd0, y[W-1]};
        ysh  = $signed(y) >>> r;
        z    = ysh + ((rem > thr) ? W'(1) : W'(0));
        rdsat = ($signed(z) > $signed(W'(511))) ? 11'sd511 : ($signed(z) < -$signed(W'(512))) ? -11'sd512 : z[10:0];
    endfunction
    // offset, clamp (npu_drq stages 3 + 4)
    function automatic logic [7:0] outq(input logic [10:0] zs, input logic [8:0] off,
                                        input logic [7:0] mn, input logic [7:0] mx);
        logic [10:0] s3, c4;
        s3 = zs + {{2{off[8]}}, off};
        c4 = s3;
        if ($signed(c4) < $signed({{3{mn[7]}}, mn})) c4 = {{3{mn[7]}}, mn};
        if ($signed(c4) > $signed({{3{mx[7]}}, mx})) c4 = {{3{mx[7]}}, mx};
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

    // ---- c0 -> t1: saturated operands (NC copies), Booth digits, shifts ----
    (* keep *) logic [W-1:0] a0q [NC];
    (* keep *) logic [W-1:0] a1q [NC];
    logic [31:0] m0q, m1q;                 // (v28A: multipliers registered, Booth-recoded in H1 after the
                                           //  phase select: one recoder per unit pair, 34 fewer flops)
    logic [4:0]  r0q, r1q;                 // (only bits 3:0 are used, see rdsat)
    always_ff @(posedge clk) begin
        for (int c = 0; c < NC; c++) begin a0q[c] <= sat22(acc0); a1q[c] <= sat22(acc1); end
        m0q <= {1'b0, mult0};
        m1q <= {1'b0, mult1};
        r0q <= shift0[7] ? 5'(8'd0 - shift0) : 5'd0;
        r1q <= shift1[7] ? 5'(8'd0 - shift1) : 5'd0;
    end

    // ---- clock phase (1 in the high phase, 0 in the low phase) ----
    // (v28A) NP phase copies: copy 0 selects the operand (shared by the 16 rows), copies 1..4 the
    // multiplier bits of 4 rows each (the H1 select fans out less since the rows recode after it)
    localparam int NP = 5;
    logic ph [NP];
    for (genvar c = 0; c < NP; c++) begin : g_ph
        (* keep *) logic tp, tn;
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
        for (int k = 0; k < 16; k++) begin
            av = ph[0] ? a0q[k * NC / 16] : a1q[k * NC / 16];
            bt = ph[1 + k / 4] ? {m0q[2*k+1], m0q[2*k], (k == 0) ? 1'b0 : m0q[(k == 0) ? 0 : 2*k-1]}
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
    // (v44R) H1 tree: bit-level Dadda tree of sky130 full-adder cells (gen: v44_R/tools/p_drq2.py)
    wire h_s0, h_c0; sky130_fd_sc_hd__fa_1 h_u0 (.A(r0[0][22]), .B(r0[1][22]), .CIN(1'b0), .SUM(h_s0), .COUT(h_c0));
    wire h_s1, h_c1; sky130_fd_sc_hd__fa_1 h_u1 (.A(r0[4][30]), .B(r0[5][30]), .CIN(1'b0), .SUM(h_s1), .COUT(h_c1));
    wire h_s2, h_c2; sky130_fd_sc_hd__fa_1 h_u2 (.A(r0[0][16]), .B(r0[1][16]), .CIN(1'b0), .SUM(h_s2), .COUT(h_c2));
    wire h_s3, h_c3; sky130_fd_sc_hd__fa_1 h_u3 (.A(r0[0][17]), .B(r0[1][17]), .CIN(1'b0), .SUM(h_s3), .COUT(h_c3));
    wire h_s4, h_c4; sky130_fd_sc_hd__fa_1 h_u4 (.A(r0[0][18]), .B(r0[1][18]), .CIN(r0[2][18]), .SUM(h_s4), .COUT(h_c4));
    wire h_s5, h_c5; sky130_fd_sc_hd__fa_1 h_u5 (.A(r0[3][18]), .B(r0[4][18]), .CIN(1'b0), .SUM(h_s5), .COUT(h_c5));
    wire h_s6, h_c6; sky130_fd_sc_hd__fa_1 h_u6 (.A(r0[0][19]), .B(r0[1][19]), .CIN(r0[2][19]), .SUM(h_s6), .COUT(h_c6));
    wire h_s7, h_c7; sky130_fd_sc_hd__fa_1 h_u7 (.A(r0[3][19]), .B(r0[4][19]), .CIN(1'b0), .SUM(h_s7), .COUT(h_c7));
    wire h_s8, h_c8; sky130_fd_sc_hd__fa_1 h_u8 (.A(r0[0][20]), .B(r0[1][20]), .CIN(r0[2][20]), .SUM(h_s8), .COUT(h_c8));
    wire h_s9, h_c9; sky130_fd_sc_hd__fa_1 h_u9 (.A(r0[3][20]), .B(r0[4][20]), .CIN(r0[5][20]), .SUM(h_s9), .COUT(h_c9));
    wire h_s10, h_c10; sky130_fd_sc_hd__fa_1 h_u10 (.A(r0[6][20]), .B(r0[7][20]), .CIN(1'b0), .SUM(h_s10), .COUT(h_c10));
    wire h_s11, h_c11; sky130_fd_sc_hd__fa_1 h_u11 (.A(r0[0][21]), .B(r0[1][21]), .CIN(r0[2][21]), .SUM(h_s11), .COUT(h_c11));
    wire h_s12, h_c12; sky130_fd_sc_hd__fa_1 h_u12 (.A(r0[3][21]), .B(r0[4][21]), .CIN(r0[5][21]), .SUM(h_s12), .COUT(h_c12));
    wire h_s13, h_c13; sky130_fd_sc_hd__fa_1 h_u13 (.A(r0[6][21]), .B(r0[7][21]), .CIN(1'b0), .SUM(h_s13), .COUT(h_c13));
    wire h_s14, h_c14; sky130_fd_sc_hd__fa_1 h_u14 (.A(h_s0), .B(r0[2][22]), .CIN(r0[3][22]), .SUM(h_s14), .COUT(h_c14));
    wire h_s15, h_c15; sky130_fd_sc_hd__fa_1 h_u15 (.A(r0[4][22]), .B(r0[5][22]), .CIN(r0[6][22]), .SUM(h_s15), .COUT(h_c15));
    wire h_s16, h_c16; sky130_fd_sc_hd__fa_1 h_u16 (.A(r0[7][22]), .B(r0[8][22]), .CIN(r0[9][22]), .SUM(h_s16), .COUT(h_c16));
    wire h_s17, h_c17; sky130_fd_sc_hd__fa_1 h_u17 (.A(r0[10][22]), .B(r0[11][22]), .CIN(1'b0), .SUM(h_s17), .COUT(h_c17));
    wire h_s18, h_c18; sky130_fd_sc_hd__fa_1 h_u18 (.A(h_c0), .B(r0[1][23]), .CIN(r0[2][23]), .SUM(h_s18), .COUT(h_c18));
    wire h_s19, h_c19; sky130_fd_sc_hd__fa_1 h_u19 (.A(r0[3][23]), .B(r0[4][23]), .CIN(r0[5][23]), .SUM(h_s19), .COUT(h_c19));
    wire h_s20, h_c20; sky130_fd_sc_hd__fa_1 h_u20 (.A(r0[6][23]), .B(r0[7][23]), .CIN(r0[8][23]), .SUM(h_s20), .COUT(h_c20));
    wire h_s21, h_c21; sky130_fd_sc_hd__fa_1 h_u21 (.A(r0[9][23]), .B(r0[10][23]), .CIN(r0[11][23]), .SUM(h_s21), .COUT(h_c21));
    wire h_s22, h_c22; sky130_fd_sc_hd__fa_1 h_u22 (.A(r0[1][24]), .B(r0[2][24]), .CIN(r0[3][24]), .SUM(h_s22), .COUT(h_c22));
    wire h_s23, h_c23; sky130_fd_sc_hd__fa_1 h_u23 (.A(r0[4][24]), .B(r0[5][24]), .CIN(r0[6][24]), .SUM(h_s23), .COUT(h_c23));
    wire h_s24, h_c24; sky130_fd_sc_hd__fa_1 h_u24 (.A(r0[7][24]), .B(r0[8][24]), .CIN(r0[9][24]), .SUM(h_s24), .COUT(h_c24));
    wire h_s25, h_c25; sky130_fd_sc_hd__fa_1 h_u25 (.A(r0[10][24]), .B(r0[11][24]), .CIN(r0[12][24]), .SUM(h_s25), .COUT(h_c25));
    wire h_s26, h_c26; sky130_fd_sc_hd__fa_1 h_u26 (.A(r0[2][25]), .B(r0[3][25]), .CIN(r0[4][25]), .SUM(h_s26), .COUT(h_c26));
    wire h_s27, h_c27; sky130_fd_sc_hd__fa_1 h_u27 (.A(r0[5][25]), .B(r0[6][25]), .CIN(r0[7][25]), .SUM(h_s27), .COUT(h_c27));
    wire h_s28, h_c28; sky130_fd_sc_hd__fa_1 h_u28 (.A(r0[8][25]), .B(r0[9][25]), .CIN(r0[10][25]), .SUM(h_s28), .COUT(h_c28));
    wire h_s29, h_c29; sky130_fd_sc_hd__fa_1 h_u29 (.A(r0[11][25]), .B(r0[12][25]), .CIN(1'b0), .SUM(h_s29), .COUT(h_c29));
    wire h_s30, h_c30; sky130_fd_sc_hd__fa_1 h_u30 (.A(r0[2][26]), .B(r0[3][26]), .CIN(r0[4][26]), .SUM(h_s30), .COUT(h_c30));
    wire h_s31, h_c31; sky130_fd_sc_hd__fa_1 h_u31 (.A(r0[5][26]), .B(r0[6][26]), .CIN(r0[7][26]), .SUM(h_s31), .COUT(h_c31));
    wire h_s32, h_c32; sky130_fd_sc_hd__fa_1 h_u32 (.A(r0[8][26]), .B(r0[9][26]), .CIN(r0[10][26]), .SUM(h_s32), .COUT(h_c32));
    wire h_s33, h_c33; sky130_fd_sc_hd__fa_1 h_u33 (.A(r0[11][26]), .B(r0[12][26]), .CIN(r0[13][26]), .SUM(h_s33), .COUT(h_c33));
    wire h_s34, h_c34; sky130_fd_sc_hd__fa_1 h_u34 (.A(r0[3][27]), .B(r0[4][27]), .CIN(r0[5][27]), .SUM(h_s34), .COUT(h_c34));
    wire h_s35, h_c35; sky130_fd_sc_hd__fa_1 h_u35 (.A(r0[6][27]), .B(r0[7][27]), .CIN(r0[8][27]), .SUM(h_s35), .COUT(h_c35));
    wire h_s36, h_c36; sky130_fd_sc_hd__fa_1 h_u36 (.A(r0[9][27]), .B(r0[10][27]), .CIN(r0[11][27]), .SUM(h_s36), .COUT(h_c36));
    wire h_s37, h_c37; sky130_fd_sc_hd__fa_1 h_u37 (.A(r0[12][27]), .B(r0[13][27]), .CIN(1'b0), .SUM(h_s37), .COUT(h_c37));
    wire h_s38, h_c38; sky130_fd_sc_hd__fa_1 h_u38 (.A(r0[3][28]), .B(r0[4][28]), .CIN(r0[5][28]), .SUM(h_s38), .COUT(h_c38));
    wire h_s39, h_c39; sky130_fd_sc_hd__fa_1 h_u39 (.A(r0[6][28]), .B(r0[7][28]), .CIN(r0[8][28]), .SUM(h_s39), .COUT(h_c39));
    wire h_s40, h_c40; sky130_fd_sc_hd__fa_1 h_u40 (.A(r0[9][28]), .B(r0[10][28]), .CIN(r0[11][28]), .SUM(h_s40), .COUT(h_c40));
    wire h_s41, h_c41; sky130_fd_sc_hd__fa_1 h_u41 (.A(r0[12][28]), .B(r0[13][28]), .CIN(r0[14][28]), .SUM(h_s41), .COUT(h_c41));
    wire h_s42, h_c42; sky130_fd_sc_hd__fa_1 h_u42 (.A(r0[4][29]), .B(r0[5][29]), .CIN(r0[6][29]), .SUM(h_s42), .COUT(h_c42));
    wire h_s43, h_c43; sky130_fd_sc_hd__fa_1 h_u43 (.A(r0[7][29]), .B(r0[8][29]), .CIN(r0[9][29]), .SUM(h_s43), .COUT(h_c43));
    wire h_s44, h_c44; sky130_fd_sc_hd__fa_1 h_u44 (.A(r0[10][29]), .B(r0[11][29]), .CIN(r0[12][29]), .SUM(h_s44), .COUT(h_c44));
    wire h_s45, h_c45; sky130_fd_sc_hd__fa_1 h_u45 (.A(r0[13][29]), .B(r0[14][29]), .CIN(1'b0), .SUM(h_s45), .COUT(h_c45));
    wire h_s46, h_c46; sky130_fd_sc_hd__fa_1 h_u46 (.A(h_s1), .B(r0[6][30]), .CIN(r0[7][30]), .SUM(h_s46), .COUT(h_c46));
    wire h_s47, h_c47; sky130_fd_sc_hd__fa_1 h_u47 (.A(r0[8][30]), .B(r0[9][30]), .CIN(r0[10][30]), .SUM(h_s47), .COUT(h_c47));
    wire h_s48, h_c48; sky130_fd_sc_hd__fa_1 h_u48 (.A(r0[11][30]), .B(r0[12][30]), .CIN(r0[13][30]), .SUM(h_s48), .COUT(h_c48));
    wire h_s49, h_c49; sky130_fd_sc_hd__fa_1 h_u49 (.A(r0[14][30]), .B(r0[15][30]), .CIN(r0[16][30]), .SUM(h_s49), .COUT(h_c49));
    wire h_s50, h_c50; sky130_fd_sc_hd__fa_1 h_u50 (.A(h_c1), .B(r0[5][31]), .CIN(r0[6][31]), .SUM(h_s50), .COUT(h_c50));
    wire h_s51, h_c51; sky130_fd_sc_hd__fa_1 h_u51 (.A(r0[7][31]), .B(r0[8][31]), .CIN(r0[9][31]), .SUM(h_s51), .COUT(h_c51));
    wire h_s52, h_c52; sky130_fd_sc_hd__fa_1 h_u52 (.A(r0[10][31]), .B(r0[11][31]), .CIN(r0[12][31]), .SUM(h_s52), .COUT(h_c52));
    wire h_s53, h_c53; sky130_fd_sc_hd__fa_1 h_u53 (.A(r0[13][31]), .B(r0[14][31]), .CIN(r0[15][31]), .SUM(h_s53), .COUT(h_c53));
    wire h_s54, h_c54; sky130_fd_sc_hd__fa_1 h_u54 (.A(r0[5][32]), .B(r0[6][32]), .CIN(r0[7][32]), .SUM(h_s54), .COUT(h_c54));
    wire h_s55, h_c55; sky130_fd_sc_hd__fa_1 h_u55 (.A(r0[8][32]), .B(r0[9][32]), .CIN(r0[10][32]), .SUM(h_s55), .COUT(h_c55));
    wire h_s56, h_c56; sky130_fd_sc_hd__fa_1 h_u56 (.A(r0[11][32]), .B(r0[12][32]), .CIN(r0[13][32]), .SUM(h_s56), .COUT(h_c56));
    wire h_s57, h_c57; sky130_fd_sc_hd__fa_1 h_u57 (.A(r0[6][33]), .B(r0[7][33]), .CIN(r0[8][33]), .SUM(h_s57), .COUT(h_c57));
    wire h_s58, h_c58; sky130_fd_sc_hd__fa_1 h_u58 (.A(r0[9][33]), .B(r0[10][33]), .CIN(r0[11][33]), .SUM(h_s58), .COUT(h_c58));
    wire h_s59, h_c59; sky130_fd_sc_hd__fa_1 h_u59 (.A(r0[12][33]), .B(r0[13][33]), .CIN(1'b0), .SUM(h_s59), .COUT(h_c59));
    wire h_s60, h_c60; sky130_fd_sc_hd__fa_1 h_u60 (.A(r0[6][34]), .B(r0[7][34]), .CIN(r0[8][34]), .SUM(h_s60), .COUT(h_c60));
    wire h_s61, h_c61; sky130_fd_sc_hd__fa_1 h_u61 (.A(r0[9][34]), .B(r0[10][34]), .CIN(r0[11][34]), .SUM(h_s61), .COUT(h_c61));
    wire h_s62, h_c62; sky130_fd_sc_hd__fa_1 h_u62 (.A(r0[7][35]), .B(r0[8][35]), .CIN(r0[9][35]), .SUM(h_s62), .COUT(h_c62));
    wire h_s63, h_c63; sky130_fd_sc_hd__fa_1 h_u63 (.A(r0[10][35]), .B(r0[11][35]), .CIN(1'b0), .SUM(h_s63), .COUT(h_c63));
    wire h_s64, h_c64; sky130_fd_sc_hd__fa_1 h_u64 (.A(r0[7][36]), .B(r0[8][36]), .CIN(r0[9][36]), .SUM(h_s64), .COUT(h_c64));
    wire h_s65, h_c65; sky130_fd_sc_hd__fa_1 h_u65 (.A(r0[8][37]), .B(r0[9][37]), .CIN(1'b0), .SUM(h_s65), .COUT(h_c65));
    wire h_s66, h_c66; sky130_fd_sc_hd__fa_1 h_u66 (.A(r0[0][10]), .B(r0[1][10]), .CIN(1'b0), .SUM(h_s66), .COUT(h_c66));
    wire h_s67, h_c67; sky130_fd_sc_hd__fa_1 h_u67 (.A(r0[0][11]), .B(r0[1][11]), .CIN(1'b0), .SUM(h_s67), .COUT(h_c67));
    wire h_s68, h_c68; sky130_fd_sc_hd__fa_1 h_u68 (.A(r0[0][12]), .B(r0[1][12]), .CIN(r0[2][12]), .SUM(h_s68), .COUT(h_c68));
    wire h_s69, h_c69; sky130_fd_sc_hd__fa_1 h_u69 (.A(r0[3][12]), .B(r0[4][12]), .CIN(1'b0), .SUM(h_s69), .COUT(h_c69));
    wire h_s70, h_c70; sky130_fd_sc_hd__fa_1 h_u70 (.A(r0[0][13]), .B(r0[1][13]), .CIN(r0[2][13]), .SUM(h_s70), .COUT(h_c70));
    wire h_s71, h_c71; sky130_fd_sc_hd__fa_1 h_u71 (.A(r0[3][13]), .B(r0[4][13]), .CIN(1'b0), .SUM(h_s71), .COUT(h_c71));
    wire h_s72, h_c72; sky130_fd_sc_hd__fa_1 h_u72 (.A(r0[0][14]), .B(r0[1][14]), .CIN(r0[2][14]), .SUM(h_s72), .COUT(h_c72));
    wire h_s73, h_c73; sky130_fd_sc_hd__fa_1 h_u73 (.A(r0[3][14]), .B(r0[4][14]), .CIN(r0[5][14]), .SUM(h_s73), .COUT(h_c73));
    wire h_s74, h_c74; sky130_fd_sc_hd__fa_1 h_u74 (.A(r0[6][14]), .B(r0[7][14]), .CIN(1'b0), .SUM(h_s74), .COUT(h_c74));
    wire h_s75, h_c75; sky130_fd_sc_hd__fa_1 h_u75 (.A(r0[0][15]), .B(r0[1][15]), .CIN(r0[2][15]), .SUM(h_s75), .COUT(h_c75));
    wire h_s76, h_c76; sky130_fd_sc_hd__fa_1 h_u76 (.A(r0[3][15]), .B(r0[4][15]), .CIN(r0[5][15]), .SUM(h_s76), .COUT(h_c76));
    wire h_s77, h_c77; sky130_fd_sc_hd__fa_1 h_u77 (.A(r0[6][15]), .B(r0[7][15]), .CIN(1'b0), .SUM(h_s77), .COUT(h_c77));
    wire h_s78, h_c78; sky130_fd_sc_hd__fa_1 h_u78 (.A(h_s2), .B(r0[2][16]), .CIN(r0[3][16]), .SUM(h_s78), .COUT(h_c78));
    wire h_s79, h_c79; sky130_fd_sc_hd__fa_1 h_u79 (.A(r0[4][16]), .B(r0[5][16]), .CIN(r0[6][16]), .SUM(h_s79), .COUT(h_c79));
    wire h_s80, h_c80; sky130_fd_sc_hd__fa_1 h_u80 (.A(r0[7][16]), .B(r0[8][16]), .CIN(r0[16][16]), .SUM(h_s80), .COUT(h_c80));
    wire h_s81, h_c81; sky130_fd_sc_hd__fa_1 h_u81 (.A(h_c2), .B(h_s3), .CIN(r0[2][17]), .SUM(h_s81), .COUT(h_c81));
    wire h_s82, h_c82; sky130_fd_sc_hd__fa_1 h_u82 (.A(r0[3][17]), .B(r0[4][17]), .CIN(r0[5][17]), .SUM(h_s82), .COUT(h_c82));
    wire h_s83, h_c83; sky130_fd_sc_hd__fa_1 h_u83 (.A(r0[6][17]), .B(r0[7][17]), .CIN(r0[8][17]), .SUM(h_s83), .COUT(h_c83));
    wire h_s84, h_c84; sky130_fd_sc_hd__fa_1 h_u84 (.A(h_c3), .B(h_s4), .CIN(h_s5), .SUM(h_s84), .COUT(h_c84));
    wire h_s85, h_c85; sky130_fd_sc_hd__fa_1 h_u85 (.A(r0[5][18]), .B(r0[6][18]), .CIN(r0[7][18]), .SUM(h_s85), .COUT(h_c85));
    wire h_s86, h_c86; sky130_fd_sc_hd__fa_1 h_u86 (.A(r0[8][18]), .B(r0[9][18]), .CIN(r0[16][18]), .SUM(h_s86), .COUT(h_c86));
    wire h_s87, h_c87; sky130_fd_sc_hd__fa_1 h_u87 (.A(h_c4), .B(h_c5), .CIN(h_s6), .SUM(h_s87), .COUT(h_c87));
    wire h_s88, h_c88; sky130_fd_sc_hd__fa_1 h_u88 (.A(h_s7), .B(r0[5][19]), .CIN(r0[6][19]), .SUM(h_s88), .COUT(h_c88));
    wire h_s89, h_c89; sky130_fd_sc_hd__fa_1 h_u89 (.A(r0[7][19]), .B(r0[8][19]), .CIN(r0[9][19]), .SUM(h_s89), .COUT(h_c89));
    wire h_s90, h_c90; sky130_fd_sc_hd__fa_1 h_u90 (.A(h_c6), .B(h_c7), .CIN(h_s8), .SUM(h_s90), .COUT(h_c90));
    wire h_s91, h_c91; sky130_fd_sc_hd__fa_1 h_u91 (.A(h_s9), .B(h_s10), .CIN(r0[8][20]), .SUM(h_s91), .COUT(h_c91));
    wire h_s92, h_c92; sky130_fd_sc_hd__fa_1 h_u92 (.A(r0[9][20]), .B(r0[10][20]), .CIN(r0[16][20]), .SUM(h_s92), .COUT(h_c92));
    wire h_s93, h_c93; sky130_fd_sc_hd__fa_1 h_u93 (.A(h_c8), .B(h_c9), .CIN(h_c10), .SUM(h_s93), .COUT(h_c93));
    wire h_s94, h_c94; sky130_fd_sc_hd__fa_1 h_u94 (.A(h_s11), .B(h_s12), .CIN(h_s13), .SUM(h_s94), .COUT(h_c94));
    wire h_s95, h_c95; sky130_fd_sc_hd__fa_1 h_u95 (.A(r0[8][21]), .B(r0[9][21]), .CIN(r0[10][21]), .SUM(h_s95), .COUT(h_c95));
    wire h_s96, h_c96; sky130_fd_sc_hd__fa_1 h_u96 (.A(h_c11), .B(h_c12), .CIN(h_c13), .SUM(h_s96), .COUT(h_c96));
    wire h_s97, h_c97; sky130_fd_sc_hd__fa_1 h_u97 (.A(h_s14), .B(h_s15), .CIN(h_s16), .SUM(h_s97), .COUT(h_c97));
    wire h_s98, h_c98; sky130_fd_sc_hd__fa_1 h_u98 (.A(h_s17), .B(r0[16][22]), .CIN(1'b1), .SUM(h_s98), .COUT(h_c98));
    wire h_s99, h_c99; sky130_fd_sc_hd__fa_1 h_u99 (.A(h_c14), .B(h_c15), .CIN(h_c16), .SUM(h_s99), .COUT(h_c99));
    wire h_s100, h_c100; sky130_fd_sc_hd__fa_1 h_u100 (.A(h_c17), .B(h_s18), .CIN(h_s19), .SUM(h_s100), .COUT(h_c100));
    wire h_s101, h_c101; sky130_fd_sc_hd__fa_1 h_u101 (.A(h_s20), .B(h_s21), .CIN(1'b1), .SUM(h_s101), .COUT(h_c101));
    wire h_s102, h_c102; sky130_fd_sc_hd__fa_1 h_u102 (.A(h_c18), .B(h_c19), .CIN(h_c20), .SUM(h_s102), .COUT(h_c102));
    wire h_s103, h_c103; sky130_fd_sc_hd__fa_1 h_u103 (.A(h_c21), .B(h_s22), .CIN(h_s23), .SUM(h_s103), .COUT(h_c103));
    wire h_s104, h_c104; sky130_fd_sc_hd__fa_1 h_u104 (.A(h_s24), .B(h_s25), .CIN(r0[16][24]), .SUM(h_s104), .COUT(h_c104));
    wire h_s105, h_c105; sky130_fd_sc_hd__fa_1 h_u105 (.A(h_c22), .B(h_c23), .CIN(h_c24), .SUM(h_s105), .COUT(h_c105));
    wire h_s106, h_c106; sky130_fd_sc_hd__fa_1 h_u106 (.A(h_c25), .B(h_s26), .CIN(h_s27), .SUM(h_s106), .COUT(h_c106));
    wire h_s107, h_c107; sky130_fd_sc_hd__fa_1 h_u107 (.A(h_s28), .B(h_s29), .CIN(1'b1), .SUM(h_s107), .COUT(h_c107));
    wire h_s108, h_c108; sky130_fd_sc_hd__fa_1 h_u108 (.A(h_c26), .B(h_c27), .CIN(h_c28), .SUM(h_s108), .COUT(h_c108));
    wire h_s109, h_c109; sky130_fd_sc_hd__fa_1 h_u109 (.A(h_c29), .B(h_s30), .CIN(h_s31), .SUM(h_s109), .COUT(h_c109));
    wire h_s110, h_c110; sky130_fd_sc_hd__fa_1 h_u110 (.A(h_s32), .B(h_s33), .CIN(r0[16][26]), .SUM(h_s110), .COUT(h_c110));
    wire h_s111, h_c111; sky130_fd_sc_hd__fa_1 h_u111 (.A(h_c30), .B(h_c31), .CIN(h_c32), .SUM(h_s111), .COUT(h_c111));
    wire h_s112, h_c112; sky130_fd_sc_hd__fa_1 h_u112 (.A(h_c33), .B(h_s34), .CIN(h_s35), .SUM(h_s112), .COUT(h_c112));
    wire h_s113, h_c113; sky130_fd_sc_hd__fa_1 h_u113 (.A(h_s36), .B(h_s37), .CIN(1'b1), .SUM(h_s113), .COUT(h_c113));
    wire h_s114, h_c114; sky130_fd_sc_hd__fa_1 h_u114 (.A(h_c34), .B(h_c35), .CIN(h_c36), .SUM(h_s114), .COUT(h_c114));
    wire h_s115, h_c115; sky130_fd_sc_hd__fa_1 h_u115 (.A(h_c37), .B(h_s38), .CIN(h_s39), .SUM(h_s115), .COUT(h_c115));
    wire h_s116, h_c116; sky130_fd_sc_hd__fa_1 h_u116 (.A(h_s40), .B(h_s41), .CIN(r0[16][28]), .SUM(h_s116), .COUT(h_c116));
    wire h_s117, h_c117; sky130_fd_sc_hd__fa_1 h_u117 (.A(h_c38), .B(h_c39), .CIN(h_c40), .SUM(h_s117), .COUT(h_c117));
    wire h_s118, h_c118; sky130_fd_sc_hd__fa_1 h_u118 (.A(h_c41), .B(h_s42), .CIN(h_s43), .SUM(h_s118), .COUT(h_c118));
    wire h_s119, h_c119; sky130_fd_sc_hd__fa_1 h_u119 (.A(h_s44), .B(h_s45), .CIN(1'b1), .SUM(h_s119), .COUT(h_c119));
    wire h_s120, h_c120; sky130_fd_sc_hd__fa_1 h_u120 (.A(h_c42), .B(h_c43), .CIN(h_c44), .SUM(h_s120), .COUT(h_c120));
    wire h_s121, h_c121; sky130_fd_sc_hd__fa_1 h_u121 (.A(h_c45), .B(h_s46), .CIN(h_s47), .SUM(h_s121), .COUT(h_c121));
    wire h_s122, h_c122; sky130_fd_sc_hd__fa_1 h_u122 (.A(h_s48), .B(h_s49), .CIN(1'b1), .SUM(h_s122), .COUT(h_c122));
    wire h_s123, h_c123; sky130_fd_sc_hd__fa_1 h_u123 (.A(h_c46), .B(h_c47), .CIN(h_c48), .SUM(h_s123), .COUT(h_c123));
    wire h_s124, h_c124; sky130_fd_sc_hd__fa_1 h_u124 (.A(h_c49), .B(h_s50), .CIN(h_s51), .SUM(h_s124), .COUT(h_c124));
    wire h_s125, h_c125; sky130_fd_sc_hd__fa_1 h_u125 (.A(h_s52), .B(h_s53), .CIN(1'b1), .SUM(h_s125), .COUT(h_c125));
    wire h_s126, h_c126; sky130_fd_sc_hd__fa_1 h_u126 (.A(h_c50), .B(h_c51), .CIN(h_c52), .SUM(h_s126), .COUT(h_c126));
    wire h_s127, h_c127; sky130_fd_sc_hd__fa_1 h_u127 (.A(h_c53), .B(h_s54), .CIN(h_s55), .SUM(h_s127), .COUT(h_c127));
    wire h_s128, h_c128; sky130_fd_sc_hd__fa_1 h_u128 (.A(h_s56), .B(r0[14][32]), .CIN(r0[15][32]), .SUM(h_s128), .COUT(h_c128));
    wire h_s129, h_c129; sky130_fd_sc_hd__fa_1 h_u129 (.A(h_c54), .B(h_c55), .CIN(h_c56), .SUM(h_s129), .COUT(h_c129));
    wire h_s130, h_c130; sky130_fd_sc_hd__fa_1 h_u130 (.A(h_s57), .B(h_s58), .CIN(h_s59), .SUM(h_s130), .COUT(h_c130));
    wire h_s131, h_c131; sky130_fd_sc_hd__fa_1 h_u131 (.A(r0[14][33]), .B(r0[15][33]), .CIN(1'b1), .SUM(h_s131), .COUT(h_c131));
    wire h_s132, h_c132; sky130_fd_sc_hd__fa_1 h_u132 (.A(h_c57), .B(h_c58), .CIN(h_c59), .SUM(h_s132), .COUT(h_c132));
    wire h_s133, h_c133; sky130_fd_sc_hd__fa_1 h_u133 (.A(h_s60), .B(h_s61), .CIN(r0[12][34]), .SUM(h_s133), .COUT(h_c133));
    wire h_s134, h_c134; sky130_fd_sc_hd__fa_1 h_u134 (.A(r0[13][34]), .B(r0[14][34]), .CIN(r0[15][34]), .SUM(h_s134), .COUT(h_c134));
    wire h_s135, h_c135; sky130_fd_sc_hd__fa_1 h_u135 (.A(h_c60), .B(h_c61), .CIN(h_s62), .SUM(h_s135), .COUT(h_c135));
    wire h_s136, h_c136; sky130_fd_sc_hd__fa_1 h_u136 (.A(h_s63), .B(r0[12][35]), .CIN(r0[13][35]), .SUM(h_s136), .COUT(h_c136));
    wire h_s137, h_c137; sky130_fd_sc_hd__fa_1 h_u137 (.A(r0[14][35]), .B(r0[15][35]), .CIN(1'b1), .SUM(h_s137), .COUT(h_c137));
    wire h_s138, h_c138; sky130_fd_sc_hd__fa_1 h_u138 (.A(h_c62), .B(h_c63), .CIN(h_s64), .SUM(h_s138), .COUT(h_c138));
    wire h_s139, h_c139; sky130_fd_sc_hd__fa_1 h_u139 (.A(r0[10][36]), .B(r0[11][36]), .CIN(r0[12][36]), .SUM(h_s139), .COUT(h_c139));
    wire h_s140, h_c140; sky130_fd_sc_hd__fa_1 h_u140 (.A(r0[13][36]), .B(r0[14][36]), .CIN(r0[15][36]), .SUM(h_s140), .COUT(h_c140));
    wire h_s141, h_c141; sky130_fd_sc_hd__fa_1 h_u141 (.A(h_c64), .B(h_s65), .CIN(r0[10][37]), .SUM(h_s141), .COUT(h_c141));
    wire h_s142, h_c142; sky130_fd_sc_hd__fa_1 h_u142 (.A(r0[11][37]), .B(r0[12][37]), .CIN(r0[13][37]), .SUM(h_s142), .COUT(h_c142));
    wire h_s143, h_c143; sky130_fd_sc_hd__fa_1 h_u143 (.A(r0[14][37]), .B(r0[15][37]), .CIN(1'b1), .SUM(h_s143), .COUT(h_c143));
    wire h_s144, h_c144; sky130_fd_sc_hd__fa_1 h_u144 (.A(h_c65), .B(r0[8][38]), .CIN(r0[9][38]), .SUM(h_s144), .COUT(h_c144));
    wire h_s145, h_c145; sky130_fd_sc_hd__fa_1 h_u145 (.A(r0[10][38]), .B(r0[11][38]), .CIN(r0[12][38]), .SUM(h_s145), .COUT(h_c145));
    wire h_s146, h_c146; sky130_fd_sc_hd__fa_1 h_u146 (.A(r0[13][38]), .B(r0[14][38]), .CIN(r0[15][38]), .SUM(h_s146), .COUT(h_c146));
    wire h_s147, h_c147; sky130_fd_sc_hd__fa_1 h_u147 (.A(r0[9][39]), .B(r0[10][39]), .CIN(r0[11][39]), .SUM(h_s147), .COUT(h_c147));
    wire h_s148, h_c148; sky130_fd_sc_hd__fa_1 h_u148 (.A(r0[12][39]), .B(r0[13][39]), .CIN(r0[14][39]), .SUM(h_s148), .COUT(h_c148));
    wire h_s149, h_c149; sky130_fd_sc_hd__fa_1 h_u149 (.A(r0[15][39]), .B(1'b1), .CIN(1'b0), .SUM(h_s149), .COUT(h_c149));
    wire h_s150, h_c150; sky130_fd_sc_hd__fa_1 h_u150 (.A(r0[9][40]), .B(r0[10][40]), .CIN(r0[11][40]), .SUM(h_s150), .COUT(h_c150));
    wire h_s151, h_c151; sky130_fd_sc_hd__fa_1 h_u151 (.A(r0[12][40]), .B(r0[13][40]), .CIN(r0[14][40]), .SUM(h_s151), .COUT(h_c151));
    wire h_s152, h_c152; sky130_fd_sc_hd__fa_1 h_u152 (.A(r0[10][41]), .B(r0[11][41]), .CIN(r0[12][41]), .SUM(h_s152), .COUT(h_c152));
    wire h_s153, h_c153; sky130_fd_sc_hd__fa_1 h_u153 (.A(r0[13][41]), .B(r0[14][41]), .CIN(1'b0), .SUM(h_s153), .COUT(h_c153));
    wire h_s154, h_c154; sky130_fd_sc_hd__fa_1 h_u154 (.A(r0[10][42]), .B(r0[11][42]), .CIN(r0[12][42]), .SUM(h_s154), .COUT(h_c154));
    wire h_s155, h_c155; sky130_fd_sc_hd__fa_1 h_u155 (.A(r0[11][43]), .B(r0[12][43]), .CIN(1'b0), .SUM(h_s155), .COUT(h_c155));
    wire h_s156, h_c156; sky130_fd_sc_hd__fa_1 h_u156 (.A(r0[0][6]), .B(r0[1][6]), .CIN(1'b0), .SUM(h_s156), .COUT(h_c156));
    wire h_s157, h_c157; sky130_fd_sc_hd__fa_1 h_u157 (.A(r0[0][7]), .B(r0[1][7]), .CIN(1'b0), .SUM(h_s157), .COUT(h_c157));
    wire h_s158, h_c158; sky130_fd_sc_hd__fa_1 h_u158 (.A(r0[0][8]), .B(r0[1][8]), .CIN(r0[2][8]), .SUM(h_s158), .COUT(h_c158));
    wire h_s159, h_c159; sky130_fd_sc_hd__fa_1 h_u159 (.A(r0[3][8]), .B(r0[4][8]), .CIN(1'b0), .SUM(h_s159), .COUT(h_c159));
    wire h_s160, h_c160; sky130_fd_sc_hd__fa_1 h_u160 (.A(r0[0][9]), .B(r0[1][9]), .CIN(r0[2][9]), .SUM(h_s160), .COUT(h_c160));
    wire h_s161, h_c161; sky130_fd_sc_hd__fa_1 h_u161 (.A(r0[3][9]), .B(r0[4][9]), .CIN(1'b0), .SUM(h_s161), .COUT(h_c161));
    wire h_s162, h_c162; sky130_fd_sc_hd__fa_1 h_u162 (.A(h_s66), .B(r0[2][10]), .CIN(r0[3][10]), .SUM(h_s162), .COUT(h_c162));
    wire h_s163, h_c163; sky130_fd_sc_hd__fa_1 h_u163 (.A(r0[4][10]), .B(r0[5][10]), .CIN(r0[16][10]), .SUM(h_s163), .COUT(h_c163));
    wire h_s164, h_c164; sky130_fd_sc_hd__fa_1 h_u164 (.A(h_c66), .B(h_s67), .CIN(r0[2][11]), .SUM(h_s164), .COUT(h_c164));
    wire h_s165, h_c165; sky130_fd_sc_hd__fa_1 h_u165 (.A(r0[3][11]), .B(r0[4][11]), .CIN(r0[5][11]), .SUM(h_s165), .COUT(h_c165));
    wire h_s166, h_c166; sky130_fd_sc_hd__fa_1 h_u166 (.A(h_c67), .B(h_s68), .CIN(h_s69), .SUM(h_s166), .COUT(h_c166));
    wire h_s167, h_c167; sky130_fd_sc_hd__fa_1 h_u167 (.A(r0[5][12]), .B(r0[6][12]), .CIN(r0[16][12]), .SUM(h_s167), .COUT(h_c167));
    wire h_s168, h_c168; sky130_fd_sc_hd__fa_1 h_u168 (.A(h_c68), .B(h_c69), .CIN(h_s70), .SUM(h_s168), .COUT(h_c168));
    wire h_s169, h_c169; sky130_fd_sc_hd__fa_1 h_u169 (.A(h_s71), .B(r0[5][13]), .CIN(r0[6][13]), .SUM(h_s169), .COUT(h_c169));
    wire h_s170, h_c170; sky130_fd_sc_hd__fa_1 h_u170 (.A(h_c70), .B(h_c71), .CIN(h_s72), .SUM(h_s170), .COUT(h_c170));
    wire h_s171, h_c171; sky130_fd_sc_hd__fa_1 h_u171 (.A(h_s73), .B(h_s74), .CIN(r0[16][14]), .SUM(h_s171), .COUT(h_c171));
    wire h_s172, h_c172; sky130_fd_sc_hd__fa_1 h_u172 (.A(h_c72), .B(h_c73), .CIN(h_c74), .SUM(h_s172), .COUT(h_c172));
    wire h_s173, h_c173; sky130_fd_sc_hd__fa_1 h_u173 (.A(h_s75), .B(h_s76), .CIN(h_s77), .SUM(h_s173), .COUT(h_c173));
    wire h_s174, h_c174; sky130_fd_sc_hd__fa_1 h_u174 (.A(h_c75), .B(h_c76), .CIN(h_c77), .SUM(h_s174), .COUT(h_c174));
    wire h_s175, h_c175; sky130_fd_sc_hd__fa_1 h_u175 (.A(h_s78), .B(h_s79), .CIN(h_s80), .SUM(h_s175), .COUT(h_c175));
    wire h_s176, h_c176; sky130_fd_sc_hd__fa_1 h_u176 (.A(h_c78), .B(h_c79), .CIN(h_c80), .SUM(h_s176), .COUT(h_c176));
    wire h_s177, h_c177; sky130_fd_sc_hd__fa_1 h_u177 (.A(h_s81), .B(h_s82), .CIN(h_s83), .SUM(h_s177), .COUT(h_c177));
    wire h_s178, h_c178; sky130_fd_sc_hd__fa_1 h_u178 (.A(h_c81), .B(h_c82), .CIN(h_c83), .SUM(h_s178), .COUT(h_c178));
    wire h_s179, h_c179; sky130_fd_sc_hd__fa_1 h_u179 (.A(h_s84), .B(h_s85), .CIN(h_s86), .SUM(h_s179), .COUT(h_c179));
    wire h_s180, h_c180; sky130_fd_sc_hd__fa_1 h_u180 (.A(h_c84), .B(h_c85), .CIN(h_c86), .SUM(h_s180), .COUT(h_c180));
    wire h_s181, h_c181; sky130_fd_sc_hd__fa_1 h_u181 (.A(h_s87), .B(h_s88), .CIN(h_s89), .SUM(h_s181), .COUT(h_c181));
    wire h_s182, h_c182; sky130_fd_sc_hd__fa_1 h_u182 (.A(h_c87), .B(h_c88), .CIN(h_c89), .SUM(h_s182), .COUT(h_c182));
    wire h_s183, h_c183; sky130_fd_sc_hd__fa_1 h_u183 (.A(h_s90), .B(h_s91), .CIN(h_s92), .SUM(h_s183), .COUT(h_c183));
    wire h_s184, h_c184; sky130_fd_sc_hd__fa_1 h_u184 (.A(h_c90), .B(h_c91), .CIN(h_c92), .SUM(h_s184), .COUT(h_c184));
    wire h_s185, h_c185; sky130_fd_sc_hd__fa_1 h_u185 (.A(h_s93), .B(h_s94), .CIN(h_s95), .SUM(h_s185), .COUT(h_c185));
    wire h_s186, h_c186; sky130_fd_sc_hd__fa_1 h_u186 (.A(h_c93), .B(h_c94), .CIN(h_c95), .SUM(h_s186), .COUT(h_c186));
    wire h_s187, h_c187; sky130_fd_sc_hd__fa_1 h_u187 (.A(h_s96), .B(h_s97), .CIN(h_s98), .SUM(h_s187), .COUT(h_c187));
    wire h_s188, h_c188; sky130_fd_sc_hd__fa_1 h_u188 (.A(h_c96), .B(h_c97), .CIN(h_c98), .SUM(h_s188), .COUT(h_c188));
    wire h_s189, h_c189; sky130_fd_sc_hd__fa_1 h_u189 (.A(h_s99), .B(h_s100), .CIN(h_s101), .SUM(h_s189), .COUT(h_c189));
    wire h_s190, h_c190; sky130_fd_sc_hd__fa_1 h_u190 (.A(h_c99), .B(h_c100), .CIN(h_c101), .SUM(h_s190), .COUT(h_c190));
    wire h_s191, h_c191; sky130_fd_sc_hd__fa_1 h_u191 (.A(h_s102), .B(h_s103), .CIN(h_s104), .SUM(h_s191), .COUT(h_c191));
    wire h_s192, h_c192; sky130_fd_sc_hd__fa_1 h_u192 (.A(h_c102), .B(h_c103), .CIN(h_c104), .SUM(h_s192), .COUT(h_c192));
    wire h_s193, h_c193; sky130_fd_sc_hd__fa_1 h_u193 (.A(h_s105), .B(h_s106), .CIN(h_s107), .SUM(h_s193), .COUT(h_c193));
    wire h_s194, h_c194; sky130_fd_sc_hd__fa_1 h_u194 (.A(h_c105), .B(h_c106), .CIN(h_c107), .SUM(h_s194), .COUT(h_c194));
    wire h_s195, h_c195; sky130_fd_sc_hd__fa_1 h_u195 (.A(h_s108), .B(h_s109), .CIN(h_s110), .SUM(h_s195), .COUT(h_c195));
    wire h_s196, h_c196; sky130_fd_sc_hd__fa_1 h_u196 (.A(h_c108), .B(h_c109), .CIN(h_c110), .SUM(h_s196), .COUT(h_c196));
    wire h_s197, h_c197; sky130_fd_sc_hd__fa_1 h_u197 (.A(h_s111), .B(h_s112), .CIN(h_s113), .SUM(h_s197), .COUT(h_c197));
    wire h_s198, h_c198; sky130_fd_sc_hd__fa_1 h_u198 (.A(h_c111), .B(h_c112), .CIN(h_c113), .SUM(h_s198), .COUT(h_c198));
    wire h_s199, h_c199; sky130_fd_sc_hd__fa_1 h_u199 (.A(h_s114), .B(h_s115), .CIN(h_s116), .SUM(h_s199), .COUT(h_c199));
    wire h_s200, h_c200; sky130_fd_sc_hd__fa_1 h_u200 (.A(h_c114), .B(h_c115), .CIN(h_c116), .SUM(h_s200), .COUT(h_c200));
    wire h_s201, h_c201; sky130_fd_sc_hd__fa_1 h_u201 (.A(h_s117), .B(h_s118), .CIN(h_s119), .SUM(h_s201), .COUT(h_c201));
    wire h_s202, h_c202; sky130_fd_sc_hd__fa_1 h_u202 (.A(h_c117), .B(h_c118), .CIN(h_c119), .SUM(h_s202), .COUT(h_c202));
    wire h_s203, h_c203; sky130_fd_sc_hd__fa_1 h_u203 (.A(h_s120), .B(h_s121), .CIN(h_s122), .SUM(h_s203), .COUT(h_c203));
    wire h_s204, h_c204; sky130_fd_sc_hd__fa_1 h_u204 (.A(h_c120), .B(h_c121), .CIN(h_c122), .SUM(h_s204), .COUT(h_c204));
    wire h_s205, h_c205; sky130_fd_sc_hd__fa_1 h_u205 (.A(h_s123), .B(h_s124), .CIN(h_s125), .SUM(h_s205), .COUT(h_c205));
    wire h_s206, h_c206; sky130_fd_sc_hd__fa_1 h_u206 (.A(h_c123), .B(h_c124), .CIN(h_c125), .SUM(h_s206), .COUT(h_c206));
    wire h_s207, h_c207; sky130_fd_sc_hd__fa_1 h_u207 (.A(h_s126), .B(h_s127), .CIN(h_s128), .SUM(h_s207), .COUT(h_c207));
    wire h_s208, h_c208; sky130_fd_sc_hd__fa_1 h_u208 (.A(h_c126), .B(h_c127), .CIN(h_c128), .SUM(h_s208), .COUT(h_c208));
    wire h_s209, h_c209; sky130_fd_sc_hd__fa_1 h_u209 (.A(h_s129), .B(h_s130), .CIN(h_s131), .SUM(h_s209), .COUT(h_c209));
    wire h_s210, h_c210; sky130_fd_sc_hd__fa_1 h_u210 (.A(h_c129), .B(h_c130), .CIN(h_c131), .SUM(h_s210), .COUT(h_c210));
    wire h_s211, h_c211; sky130_fd_sc_hd__fa_1 h_u211 (.A(h_s132), .B(h_s133), .CIN(h_s134), .SUM(h_s211), .COUT(h_c211));
    wire h_s212, h_c212; sky130_fd_sc_hd__fa_1 h_u212 (.A(h_c132), .B(h_c133), .CIN(h_c134), .SUM(h_s212), .COUT(h_c212));
    wire h_s213, h_c213; sky130_fd_sc_hd__fa_1 h_u213 (.A(h_s135), .B(h_s136), .CIN(h_s137), .SUM(h_s213), .COUT(h_c213));
    wire h_s214, h_c214; sky130_fd_sc_hd__fa_1 h_u214 (.A(h_c135), .B(h_c136), .CIN(h_c137), .SUM(h_s214), .COUT(h_c214));
    wire h_s215, h_c215; sky130_fd_sc_hd__fa_1 h_u215 (.A(h_s138), .B(h_s139), .CIN(h_s140), .SUM(h_s215), .COUT(h_c215));
    wire h_s216, h_c216; sky130_fd_sc_hd__fa_1 h_u216 (.A(h_c138), .B(h_c139), .CIN(h_c140), .SUM(h_s216), .COUT(h_c216));
    wire h_s217, h_c217; sky130_fd_sc_hd__fa_1 h_u217 (.A(h_s141), .B(h_s142), .CIN(h_s143), .SUM(h_s217), .COUT(h_c217));
    wire h_s218, h_c218; sky130_fd_sc_hd__fa_1 h_u218 (.A(h_c141), .B(h_c142), .CIN(h_c143), .SUM(h_s218), .COUT(h_c218));
    wire h_s219, h_c219; sky130_fd_sc_hd__fa_1 h_u219 (.A(h_s144), .B(h_s145), .CIN(h_s146), .SUM(h_s219), .COUT(h_c219));
    wire h_s220, h_c220; sky130_fd_sc_hd__fa_1 h_u220 (.A(h_c144), .B(h_c145), .CIN(h_c146), .SUM(h_s220), .COUT(h_c220));
    wire h_s221, h_c221; sky130_fd_sc_hd__fa_1 h_u221 (.A(h_s147), .B(h_s148), .CIN(h_s149), .SUM(h_s221), .COUT(h_c221));
    wire h_s222, h_c222; sky130_fd_sc_hd__fa_1 h_u222 (.A(h_c147), .B(h_c148), .CIN(h_c149), .SUM(h_s222), .COUT(h_c222));
    wire h_s223, h_c223; sky130_fd_sc_hd__fa_1 h_u223 (.A(h_s150), .B(h_s151), .CIN(r0[15][40]), .SUM(h_s223), .COUT(h_c223));
    wire h_s224, h_c224; sky130_fd_sc_hd__fa_1 h_u224 (.A(h_c150), .B(h_c151), .CIN(h_s152), .SUM(h_s224), .COUT(h_c224));
    wire h_s225, h_c225; sky130_fd_sc_hd__fa_1 h_u225 (.A(h_s153), .B(r0[15][41]), .CIN(1'b1), .SUM(h_s225), .COUT(h_c225));
    wire h_s226, h_c226; sky130_fd_sc_hd__fa_1 h_u226 (.A(h_c152), .B(h_c153), .CIN(h_s154), .SUM(h_s226), .COUT(h_c226));
    wire h_s227, h_c227; sky130_fd_sc_hd__fa_1 h_u227 (.A(r0[13][42]), .B(r0[14][42]), .CIN(r0[15][42]), .SUM(h_s227), .COUT(h_c227));
    wire h_s228, h_c228; sky130_fd_sc_hd__fa_1 h_u228 (.A(h_c154), .B(h_s155), .CIN(r0[13][43]), .SUM(h_s228), .COUT(h_c228));
    wire h_s229, h_c229; sky130_fd_sc_hd__fa_1 h_u229 (.A(r0[14][43]), .B(r0[15][43]), .CIN(1'b1), .SUM(h_s229), .COUT(h_c229));
    wire h_s230, h_c230; sky130_fd_sc_hd__fa_1 h_u230 (.A(h_c155), .B(r0[11][44]), .CIN(r0[12][44]), .SUM(h_s230), .COUT(h_c230));
    wire h_s231, h_c231; sky130_fd_sc_hd__fa_1 h_u231 (.A(r0[13][44]), .B(r0[14][44]), .CIN(r0[15][44]), .SUM(h_s231), .COUT(h_c231));
    wire h_s232, h_c232; sky130_fd_sc_hd__fa_1 h_u232 (.A(r0[12][45]), .B(r0[13][45]), .CIN(r0[14][45]), .SUM(h_s232), .COUT(h_c232));
    wire h_s233, h_c233; sky130_fd_sc_hd__fa_1 h_u233 (.A(r0[15][45]), .B(1'b1), .CIN(1'b0), .SUM(h_s233), .COUT(h_c233));
    wire h_s234, h_c234; sky130_fd_sc_hd__fa_1 h_u234 (.A(r0[12][46]), .B(r0[13][46]), .CIN(r0[14][46]), .SUM(h_s234), .COUT(h_c234));
    wire h_s235, h_c235; sky130_fd_sc_hd__fa_1 h_u235 (.A(r0[13][47]), .B(r0[14][47]), .CIN(1'b0), .SUM(h_s235), .COUT(h_c235));
    wire h_s236, h_c236; sky130_fd_sc_hd__fa_1 h_u236 (.A(r0[0][4]), .B(r0[1][4]), .CIN(1'b0), .SUM(h_s236), .COUT(h_c236));
    wire h_s237, h_c237; sky130_fd_sc_hd__fa_1 h_u237 (.A(r0[0][5]), .B(r0[1][5]), .CIN(1'b0), .SUM(h_s237), .COUT(h_c237));
    wire h_s238, h_c238; sky130_fd_sc_hd__fa_1 h_u238 (.A(h_s156), .B(r0[2][6]), .CIN(r0[3][6]), .SUM(h_s238), .COUT(h_c238));
    wire h_s239, h_c239; sky130_fd_sc_hd__fa_1 h_u239 (.A(h_c156), .B(h_s157), .CIN(r0[2][7]), .SUM(h_s239), .COUT(h_c239));
    wire h_s240, h_c240; sky130_fd_sc_hd__fa_1 h_u240 (.A(h_c157), .B(h_s158), .CIN(h_s159), .SUM(h_s240), .COUT(h_c240));
    wire h_s241, h_c241; sky130_fd_sc_hd__fa_1 h_u241 (.A(h_c158), .B(h_c159), .CIN(h_s160), .SUM(h_s241), .COUT(h_c241));
    wire h_s242, h_c242; sky130_fd_sc_hd__fa_1 h_u242 (.A(h_c160), .B(h_c161), .CIN(h_s162), .SUM(h_s242), .COUT(h_c242));
    wire h_s243, h_c243; sky130_fd_sc_hd__fa_1 h_u243 (.A(h_c162), .B(h_c163), .CIN(h_s164), .SUM(h_s243), .COUT(h_c243));
    wire h_s244, h_c244; sky130_fd_sc_hd__fa_1 h_u244 (.A(h_c164), .B(h_c165), .CIN(h_s166), .SUM(h_s244), .COUT(h_c244));
    wire h_s245, h_c245; sky130_fd_sc_hd__fa_1 h_u245 (.A(h_c166), .B(h_c167), .CIN(h_s168), .SUM(h_s245), .COUT(h_c245));
    wire h_s246, h_c246; sky130_fd_sc_hd__fa_1 h_u246 (.A(h_c168), .B(h_c169), .CIN(h_s170), .SUM(h_s246), .COUT(h_c246));
    wire h_s247, h_c247; sky130_fd_sc_hd__fa_1 h_u247 (.A(h_c170), .B(h_c171), .CIN(h_s172), .SUM(h_s247), .COUT(h_c247));
    wire h_s248, h_c248; sky130_fd_sc_hd__fa_1 h_u248 (.A(h_c172), .B(h_c173), .CIN(h_s174), .SUM(h_s248), .COUT(h_c248));
    wire h_s249, h_c249; sky130_fd_sc_hd__fa_1 h_u249 (.A(h_c174), .B(h_c175), .CIN(h_s176), .SUM(h_s249), .COUT(h_c249));
    wire h_s250, h_c250; sky130_fd_sc_hd__fa_1 h_u250 (.A(h_c176), .B(h_c177), .CIN(h_s178), .SUM(h_s250), .COUT(h_c250));
    wire h_s251, h_c251; sky130_fd_sc_hd__fa_1 h_u251 (.A(h_c178), .B(h_c179), .CIN(h_s180), .SUM(h_s251), .COUT(h_c251));
    wire h_s252, h_c252; sky130_fd_sc_hd__fa_1 h_u252 (.A(h_c180), .B(h_c181), .CIN(h_s182), .SUM(h_s252), .COUT(h_c252));
    wire h_s253, h_c253; sky130_fd_sc_hd__fa_1 h_u253 (.A(h_c182), .B(h_c183), .CIN(h_s184), .SUM(h_s253), .COUT(h_c253));
    wire h_s254, h_c254; sky130_fd_sc_hd__fa_1 h_u254 (.A(h_c184), .B(h_c185), .CIN(h_s186), .SUM(h_s254), .COUT(h_c254));
    wire h_s255, h_c255; sky130_fd_sc_hd__fa_1 h_u255 (.A(h_c186), .B(h_c187), .CIN(h_s188), .SUM(h_s255), .COUT(h_c255));
    wire h_s256, h_c256; sky130_fd_sc_hd__fa_1 h_u256 (.A(h_c188), .B(h_c189), .CIN(h_s190), .SUM(h_s256), .COUT(h_c256));
    wire h_s257, h_c257; sky130_fd_sc_hd__fa_1 h_u257 (.A(h_c190), .B(h_c191), .CIN(h_s192), .SUM(h_s257), .COUT(h_c257));
    wire h_s258, h_c258; sky130_fd_sc_hd__fa_1 h_u258 (.A(h_c192), .B(h_c193), .CIN(h_s194), .SUM(h_s258), .COUT(h_c258));
    wire h_s259, h_c259; sky130_fd_sc_hd__fa_1 h_u259 (.A(h_c194), .B(h_c195), .CIN(h_s196), .SUM(h_s259), .COUT(h_c259));
    wire h_s260, h_c260; sky130_fd_sc_hd__fa_1 h_u260 (.A(h_c196), .B(h_c197), .CIN(h_s198), .SUM(h_s260), .COUT(h_c260));
    wire h_s261, h_c261; sky130_fd_sc_hd__fa_1 h_u261 (.A(h_c198), .B(h_c199), .CIN(h_s200), .SUM(h_s261), .COUT(h_c261));
    wire h_s262, h_c262; sky130_fd_sc_hd__fa_1 h_u262 (.A(h_c200), .B(h_c201), .CIN(h_s202), .SUM(h_s262), .COUT(h_c262));
    wire h_s263, h_c263; sky130_fd_sc_hd__fa_1 h_u263 (.A(h_c202), .B(h_c203), .CIN(h_s204), .SUM(h_s263), .COUT(h_c263));
    wire h_s264, h_c264; sky130_fd_sc_hd__fa_1 h_u264 (.A(h_c204), .B(h_c205), .CIN(h_s206), .SUM(h_s264), .COUT(h_c264));
    wire h_s265, h_c265; sky130_fd_sc_hd__fa_1 h_u265 (.A(h_c206), .B(h_c207), .CIN(h_s208), .SUM(h_s265), .COUT(h_c265));
    wire h_s266, h_c266; sky130_fd_sc_hd__fa_1 h_u266 (.A(h_c208), .B(h_c209), .CIN(h_s210), .SUM(h_s266), .COUT(h_c266));
    wire h_s267, h_c267; sky130_fd_sc_hd__fa_1 h_u267 (.A(h_c210), .B(h_c211), .CIN(h_s212), .SUM(h_s267), .COUT(h_c267));
    wire h_s268, h_c268; sky130_fd_sc_hd__fa_1 h_u268 (.A(h_c212), .B(h_c213), .CIN(h_s214), .SUM(h_s268), .COUT(h_c268));
    wire h_s269, h_c269; sky130_fd_sc_hd__fa_1 h_u269 (.A(h_c214), .B(h_c215), .CIN(h_s216), .SUM(h_s269), .COUT(h_c269));
    wire h_s270, h_c270; sky130_fd_sc_hd__fa_1 h_u270 (.A(h_c216), .B(h_c217), .CIN(h_s218), .SUM(h_s270), .COUT(h_c270));
    wire h_s271, h_c271; sky130_fd_sc_hd__fa_1 h_u271 (.A(h_c218), .B(h_c219), .CIN(h_s220), .SUM(h_s271), .COUT(h_c271));
    wire h_s272, h_c272; sky130_fd_sc_hd__fa_1 h_u272 (.A(h_c220), .B(h_c221), .CIN(h_s222), .SUM(h_s272), .COUT(h_c272));
    wire h_s273, h_c273; sky130_fd_sc_hd__fa_1 h_u273 (.A(h_c222), .B(h_c223), .CIN(h_s224), .SUM(h_s273), .COUT(h_c273));
    wire h_s274, h_c274; sky130_fd_sc_hd__fa_1 h_u274 (.A(h_c224), .B(h_c225), .CIN(h_s226), .SUM(h_s274), .COUT(h_c274));
    wire h_s275, h_c275; sky130_fd_sc_hd__fa_1 h_u275 (.A(h_c226), .B(h_c227), .CIN(h_s228), .SUM(h_s275), .COUT(h_c275));
    wire h_s276, h_c276; sky130_fd_sc_hd__fa_1 h_u276 (.A(h_c228), .B(h_c229), .CIN(h_s230), .SUM(h_s276), .COUT(h_c276));
    wire h_s277, h_c277; sky130_fd_sc_hd__fa_1 h_u277 (.A(h_c230), .B(h_c231), .CIN(h_s232), .SUM(h_s277), .COUT(h_c277));
    wire h_s278, h_c278; sky130_fd_sc_hd__fa_1 h_u278 (.A(h_c232), .B(h_c233), .CIN(h_s234), .SUM(h_s278), .COUT(h_c278));
    wire h_s279, h_c279; sky130_fd_sc_hd__fa_1 h_u279 (.A(h_c234), .B(h_s235), .CIN(r0[15][47]), .SUM(h_s279), .COUT(h_c279));
    wire h_s280, h_c280; sky130_fd_sc_hd__fa_1 h_u280 (.A(h_c235), .B(r0[13][48]), .CIN(r0[14][48]), .SUM(h_s280), .COUT(h_c280));
    wire h_s281, h_c281; sky130_fd_sc_hd__fa_1 h_u281 (.A(r0[14][49]), .B(r0[15][49]), .CIN(1'b0), .SUM(h_s281), .COUT(h_c281));
    wire h_s282, h_c282; sky130_fd_sc_hd__fa_1 h_u282 (.A(r0[0][2]), .B(r0[1][2]), .CIN(1'b0), .SUM(h_s282), .COUT(h_c282));
    wire h_s283, h_c283; sky130_fd_sc_hd__fa_1 h_u283 (.A(r0[0][3]), .B(r0[1][3]), .CIN(1'b0), .SUM(h_s283), .COUT(h_c283));
    wire h_s284, h_c284; sky130_fd_sc_hd__fa_1 h_u284 (.A(h_s236), .B(r0[2][4]), .CIN(r0[16][4]), .SUM(h_s284), .COUT(h_c284));
    wire h_s285, h_c285; sky130_fd_sc_hd__fa_1 h_u285 (.A(h_c236), .B(h_s237), .CIN(r0[2][5]), .SUM(h_s285), .COUT(h_c285));
    wire h_s286, h_c286; sky130_fd_sc_hd__fa_1 h_u286 (.A(h_c237), .B(h_s238), .CIN(r0[16][6]), .SUM(h_s286), .COUT(h_c286));
    wire h_s287, h_c287; sky130_fd_sc_hd__fa_1 h_u287 (.A(h_c238), .B(h_s239), .CIN(r0[3][7]), .SUM(h_s287), .COUT(h_c287));
    wire h_s288, h_c288; sky130_fd_sc_hd__fa_1 h_u288 (.A(h_c239), .B(h_s240), .CIN(r0[16][8]), .SUM(h_s288), .COUT(h_c288));
    wire h_s289, h_c289; sky130_fd_sc_hd__fa_1 h_u289 (.A(h_c240), .B(h_s241), .CIN(h_s161), .SUM(h_s289), .COUT(h_c289));
    wire h_s290, h_c290; sky130_fd_sc_hd__fa_1 h_u290 (.A(h_c241), .B(h_s242), .CIN(h_s163), .SUM(h_s290), .COUT(h_c290));
    wire h_s291, h_c291; sky130_fd_sc_hd__fa_1 h_u291 (.A(h_c242), .B(h_s243), .CIN(h_s165), .SUM(h_s291), .COUT(h_c291));
    wire h_s292, h_c292; sky130_fd_sc_hd__fa_1 h_u292 (.A(h_c243), .B(h_s244), .CIN(h_s167), .SUM(h_s292), .COUT(h_c292));
    wire h_s293, h_c293; sky130_fd_sc_hd__fa_1 h_u293 (.A(h_c244), .B(h_s245), .CIN(h_s169), .SUM(h_s293), .COUT(h_c293));
    wire h_s294, h_c294; sky130_fd_sc_hd__fa_1 h_u294 (.A(h_c245), .B(h_s246), .CIN(h_s171), .SUM(h_s294), .COUT(h_c294));
    wire h_s295, h_c295; sky130_fd_sc_hd__fa_1 h_u295 (.A(h_c246), .B(h_s247), .CIN(h_s173), .SUM(h_s295), .COUT(h_c295));
    wire h_s296, h_c296; sky130_fd_sc_hd__fa_1 h_u296 (.A(h_c247), .B(h_s248), .CIN(h_s175), .SUM(h_s296), .COUT(h_c296));
    wire h_s297, h_c297; sky130_fd_sc_hd__fa_1 h_u297 (.A(h_c248), .B(h_s249), .CIN(h_s177), .SUM(h_s297), .COUT(h_c297));
    wire h_s298, h_c298; sky130_fd_sc_hd__fa_1 h_u298 (.A(h_c249), .B(h_s250), .CIN(h_s179), .SUM(h_s298), .COUT(h_c298));
    wire h_s299, h_c299; sky130_fd_sc_hd__fa_1 h_u299 (.A(h_c250), .B(h_s251), .CIN(h_s181), .SUM(h_s299), .COUT(h_c299));
    wire h_s300, h_c300; sky130_fd_sc_hd__fa_1 h_u300 (.A(h_c251), .B(h_s252), .CIN(h_s183), .SUM(h_s300), .COUT(h_c300));
    wire h_s301, h_c301; sky130_fd_sc_hd__fa_1 h_u301 (.A(h_c252), .B(h_s253), .CIN(h_s185), .SUM(h_s301), .COUT(h_c301));
    wire h_s302, h_c302; sky130_fd_sc_hd__fa_1 h_u302 (.A(h_c253), .B(h_s254), .CIN(h_s187), .SUM(h_s302), .COUT(h_c302));
    wire h_s303, h_c303; sky130_fd_sc_hd__fa_1 h_u303 (.A(h_c254), .B(h_s255), .CIN(h_s189), .SUM(h_s303), .COUT(h_c303));
    wire h_s304, h_c304; sky130_fd_sc_hd__fa_1 h_u304 (.A(h_c255), .B(h_s256), .CIN(h_s191), .SUM(h_s304), .COUT(h_c304));
    wire h_s305, h_c305; sky130_fd_sc_hd__fa_1 h_u305 (.A(h_c256), .B(h_s257), .CIN(h_s193), .SUM(h_s305), .COUT(h_c305));
    wire h_s306, h_c306; sky130_fd_sc_hd__fa_1 h_u306 (.A(h_c257), .B(h_s258), .CIN(h_s195), .SUM(h_s306), .COUT(h_c306));
    wire h_s307, h_c307; sky130_fd_sc_hd__fa_1 h_u307 (.A(h_c258), .B(h_s259), .CIN(h_s197), .SUM(h_s307), .COUT(h_c307));
    wire h_s308, h_c308; sky130_fd_sc_hd__fa_1 h_u308 (.A(h_c259), .B(h_s260), .CIN(h_s199), .SUM(h_s308), .COUT(h_c308));
    wire h_s309, h_c309; sky130_fd_sc_hd__fa_1 h_u309 (.A(h_c260), .B(h_s261), .CIN(h_s201), .SUM(h_s309), .COUT(h_c309));
    wire h_s310, h_c310; sky130_fd_sc_hd__fa_1 h_u310 (.A(h_c261), .B(h_s262), .CIN(h_s203), .SUM(h_s310), .COUT(h_c310));
    wire h_s311, h_c311; sky130_fd_sc_hd__fa_1 h_u311 (.A(h_c262), .B(h_s263), .CIN(h_s205), .SUM(h_s311), .COUT(h_c311));
    wire h_s312, h_c312; sky130_fd_sc_hd__fa_1 h_u312 (.A(h_c263), .B(h_s264), .CIN(h_s207), .SUM(h_s312), .COUT(h_c312));
    wire h_s313, h_c313; sky130_fd_sc_hd__fa_1 h_u313 (.A(h_c264), .B(h_s265), .CIN(h_s209), .SUM(h_s313), .COUT(h_c313));
    wire h_s314, h_c314; sky130_fd_sc_hd__fa_1 h_u314 (.A(h_c265), .B(h_s266), .CIN(h_s211), .SUM(h_s314), .COUT(h_c314));
    wire h_s315, h_c315; sky130_fd_sc_hd__fa_1 h_u315 (.A(h_c266), .B(h_s267), .CIN(h_s213), .SUM(h_s315), .COUT(h_c315));
    wire h_s316, h_c316; sky130_fd_sc_hd__fa_1 h_u316 (.A(h_c267), .B(h_s268), .CIN(h_s215), .SUM(h_s316), .COUT(h_c316));
    wire h_s317, h_c317; sky130_fd_sc_hd__fa_1 h_u317 (.A(h_c268), .B(h_s269), .CIN(h_s217), .SUM(h_s317), .COUT(h_c317));
    wire h_s318, h_c318; sky130_fd_sc_hd__fa_1 h_u318 (.A(h_c269), .B(h_s270), .CIN(h_s219), .SUM(h_s318), .COUT(h_c318));
    wire h_s319, h_c319; sky130_fd_sc_hd__fa_1 h_u319 (.A(h_c270), .B(h_s271), .CIN(h_s221), .SUM(h_s319), .COUT(h_c319));
    wire h_s320, h_c320; sky130_fd_sc_hd__fa_1 h_u320 (.A(h_c271), .B(h_s272), .CIN(h_s223), .SUM(h_s320), .COUT(h_c320));
    wire h_s321, h_c321; sky130_fd_sc_hd__fa_1 h_u321 (.A(h_c272), .B(h_s273), .CIN(h_s225), .SUM(h_s321), .COUT(h_c321));
    wire h_s322, h_c322; sky130_fd_sc_hd__fa_1 h_u322 (.A(h_c273), .B(h_s274), .CIN(h_s227), .SUM(h_s322), .COUT(h_c322));
    wire h_s323, h_c323; sky130_fd_sc_hd__fa_1 h_u323 (.A(h_c274), .B(h_s275), .CIN(h_s229), .SUM(h_s323), .COUT(h_c323));
    wire h_s324, h_c324; sky130_fd_sc_hd__fa_1 h_u324 (.A(h_c275), .B(h_s276), .CIN(h_s231), .SUM(h_s324), .COUT(h_c324));
    wire h_s325, h_c325; sky130_fd_sc_hd__fa_1 h_u325 (.A(h_c276), .B(h_s277), .CIN(h_s233), .SUM(h_s325), .COUT(h_c325));
    wire h_s326, h_c326; sky130_fd_sc_hd__fa_1 h_u326 (.A(h_c277), .B(h_s278), .CIN(r0[15][46]), .SUM(h_s326), .COUT(h_c326));
    wire h_s327, h_c327; sky130_fd_sc_hd__fa_1 h_u327 (.A(h_c278), .B(h_s279), .CIN(1'b1), .SUM(h_s327), .COUT(h_c327));
    wire h_s328, h_c328; sky130_fd_sc_hd__fa_1 h_u328 (.A(h_c279), .B(h_s280), .CIN(r0[15][48]), .SUM(h_s328), .COUT(h_c328));
    wire h_s329, h_c329; sky130_fd_sc_hd__fa_1 h_u329 (.A(h_c280), .B(h_s281), .CIN(1'b1), .SUM(h_s329), .COUT(h_c329));
    wire h_s330, h_c330; sky130_fd_sc_hd__fa_1 h_u330 (.A(h_c281), .B(r0[14][50]), .CIN(r0[15][50]), .SUM(h_s330), .COUT(h_c330));
    wire h_s331, h_c331; sky130_fd_sc_hd__fa_1 h_u331 (.A(r0[15][51]), .B(1'b1), .CIN(1'b0), .SUM(h_s331), .COUT(h_c331));
    // (v44R) carry out of the low 31 product bits ((G, T) prefix tree, keep nodes): shared by both sets
    wire k_n0 = r0[0][0] & r0[16][0];
    wire k_n1 = h_s282 & r0[16][2];
    wire k_n2 = h_s282 | r0[16][2];
    (* keep *) wire k_n3 = k_n2 & r0[0][1];
    (* keep *) wire k_n4 = k_n1 | (k_n3 & k_n0);
    wire k_n5 = h_c282 & h_s283;
    wire k_n6 = h_c282 | h_s283;
    wire k_n7 = h_c283 & h_s284;
    wire k_n8 = h_c283 | h_s284;
    (* keep *) wire k_n9 = k_n7 | (k_n8 & k_n5);
    (* keep *) wire k_n10 = k_n8 & k_n6;
    wire k_n11 = h_c284 & h_s285;
    wire k_n12 = h_c284 | h_s285;
    wire k_n13 = h_c285 & h_s286;
    wire k_n14 = h_c285 | h_s286;
    (* keep *) wire k_n15 = k_n13 | (k_n14 & k_n11);
    (* keep *) wire k_n16 = k_n14 & k_n12;
    (* keep *) wire k_n17 = k_n15 | (k_n16 & k_n9);
    (* keep *) wire k_n18 = k_n16 & k_n10;
    (* keep *) wire k_n19 = k_n17 | (k_n18 & k_n4);
    wire k_n20 = h_c286 & h_s287;
    wire k_n21 = h_c286 | h_s287;
    wire k_n22 = h_c287 & h_s288;
    wire k_n23 = h_c287 | h_s288;
    (* keep *) wire k_n24 = k_n22 | (k_n23 & k_n20);
    (* keep *) wire k_n25 = k_n23 & k_n21;
    wire k_n26 = h_c288 & h_s289;
    wire k_n27 = h_c288 | h_s289;
    wire k_n28 = h_c289 & h_s290;
    wire k_n29 = h_c289 | h_s290;
    (* keep *) wire k_n30 = k_n28 | (k_n29 & k_n26);
    (* keep *) wire k_n31 = k_n29 & k_n27;
    (* keep *) wire k_n32 = k_n30 | (k_n31 & k_n24);
    (* keep *) wire k_n33 = k_n31 & k_n25;
    wire k_n34 = h_c290 & h_s291;
    wire k_n35 = h_c290 | h_s291;
    wire k_n36 = h_c291 & h_s292;
    wire k_n37 = h_c291 | h_s292;
    (* keep *) wire k_n38 = k_n36 | (k_n37 & k_n34);
    (* keep *) wire k_n39 = k_n37 & k_n35;
    wire k_n40 = h_c292 & h_s293;
    wire k_n41 = h_c292 | h_s293;
    wire k_n42 = h_c293 & h_s294;
    wire k_n43 = h_c293 | h_s294;
    (* keep *) wire k_n44 = k_n42 | (k_n43 & k_n40);
    (* keep *) wire k_n45 = k_n43 & k_n41;
    (* keep *) wire k_n46 = k_n44 | (k_n45 & k_n38);
    (* keep *) wire k_n47 = k_n45 & k_n39;
    (* keep *) wire k_n48 = k_n46 | (k_n47 & k_n32);
    (* keep *) wire k_n49 = k_n47 & k_n33;
    (* keep *) wire k_n50 = k_n48 | (k_n49 & k_n19);
    wire k_n51 = h_c294 & h_s295;
    wire k_n52 = h_c294 | h_s295;
    wire k_n53 = h_c295 & h_s296;
    wire k_n54 = h_c295 | h_s296;
    (* keep *) wire k_n55 = k_n53 | (k_n54 & k_n51);
    (* keep *) wire k_n56 = k_n54 & k_n52;
    wire k_n57 = h_c296 & h_s297;
    wire k_n58 = h_c296 | h_s297;
    wire k_n59 = h_c297 & h_s298;
    wire k_n60 = h_c297 | h_s298;
    (* keep *) wire k_n61 = k_n59 | (k_n60 & k_n57);
    (* keep *) wire k_n62 = k_n60 & k_n58;
    (* keep *) wire k_n63 = k_n61 | (k_n62 & k_n55);
    (* keep *) wire k_n64 = k_n62 & k_n56;
    wire k_n65 = h_c298 & h_s299;
    wire k_n66 = h_c298 | h_s299;
    wire k_n67 = h_c299 & h_s300;
    wire k_n68 = h_c299 | h_s300;
    (* keep *) wire k_n69 = k_n67 | (k_n68 & k_n65);
    (* keep *) wire k_n70 = k_n68 & k_n66;
    wire k_n71 = h_c300 & h_s301;
    wire k_n72 = h_c300 | h_s301;
    wire k_n73 = h_c301 & h_s302;
    wire k_n74 = h_c301 | h_s302;
    (* keep *) wire k_n75 = k_n73 | (k_n74 & k_n71);
    (* keep *) wire k_n76 = k_n74 & k_n72;
    (* keep *) wire k_n77 = k_n75 | (k_n76 & k_n69);
    (* keep *) wire k_n78 = k_n76 & k_n70;
    (* keep *) wire k_n79 = k_n77 | (k_n78 & k_n63);
    (* keep *) wire k_n80 = k_n78 & k_n64;
    wire k_n81 = h_c302 & h_s303;
    wire k_n82 = h_c302 | h_s303;
    wire k_n83 = h_c303 & h_s304;
    wire k_n84 = h_c303 | h_s304;
    (* keep *) wire k_n85 = k_n83 | (k_n84 & k_n81);
    (* keep *) wire k_n86 = k_n84 & k_n82;
    wire k_n87 = h_c304 & h_s305;
    wire k_n88 = h_c304 | h_s305;
    wire k_n89 = h_c305 & h_s306;
    wire k_n90 = h_c305 | h_s306;
    (* keep *) wire k_n91 = k_n89 | (k_n90 & k_n87);
    (* keep *) wire k_n92 = k_n90 & k_n88;
    (* keep *) wire k_n93 = k_n91 | (k_n92 & k_n85);
    (* keep *) wire k_n94 = k_n92 & k_n86;
    wire k_n95 = h_c306 & h_s307;
    wire k_n96 = h_c306 | h_s307;
    wire k_n97 = h_c307 & h_s308;
    wire k_n98 = h_c307 | h_s308;
    (* keep *) wire k_n99 = k_n97 | (k_n98 & k_n95);
    (* keep *) wire k_n100 = k_n98 & k_n96;
    wire k_n101 = h_c308 & h_s309;
    wire k_n102 = h_c308 | h_s309;
    wire k_n103 = h_c309 & h_s310;
    wire k_n104 = h_c309 | h_s310;
    (* keep *) wire k_n105 = k_n103 | (k_n104 & k_n101);
    (* keep *) wire k_n106 = k_n104 & k_n102;
    (* keep *) wire k_n107 = k_n105 | (k_n106 & k_n99);
    (* keep *) wire k_n108 = k_n106 & k_n100;
    (* keep *) wire k_n109 = k_n107 | (k_n108 & k_n93);
    (* keep *) wire k_n110 = k_n108 & k_n94;
    (* keep *) wire k_n111 = k_n109 | (k_n110 & k_n79);
    (* keep *) wire k_n112 = k_n110 & k_n80;
    (* keep *) wire k_n113 = k_n111 | (k_n112 & k_n50);
    wire           hk = k_n113;
    wire  [W-1:0]  hs = {h_c331, h_c330, h_c329, h_c328, h_c327, h_c326, h_c325, h_c324, h_c323, h_c322, h_c321, h_c320, h_c319, h_c318, h_c317, h_c316, h_c315, h_c314, h_c313, h_c312, h_c311, h_c310};
    wire  [W-1:0]  hc = {r0[15][52], h_s331, h_s330, h_s329, h_s328, h_s327, h_s326, h_s325, h_s324, h_s323, h_s322, h_s321, h_s320, h_s319, h_s318, h_s317, h_s316, h_s315, h_s314, h_s313, h_s312, h_s311};
    // (v44R) only the kept product bits [PW-1:31] and the carry into bit 31 are registered
    logic [W-1:0]  s0n, c0n, s1p, c1p;
    logic          k0n, k1p;
    always_ff @(negedge clk) begin s0n <= hs; c0n <= hc; k0n <= hk; end    // set 0 (high phase)
    always_ff @(posedge clk) begin s1p <= hs; c1p <= hc; k1p <= hk; end    // set 1 (low phase)

    // ---- set 0: add (low phase) -> t2; RoundingDivideByPOT + saturation -> t3 ----
    wire  [W-1:0]  p0 = s0n + c0n + W'(k0n);
    logic [W-1:0]  y0;
    logic [3:0]    r0d, r1d;
    always_ff @(posedge clk) begin
        y0  <= p0;
        r0d <= r0q[3:0];
        r1d <= r1q[3:0];
    end
    // ---- set 1: add + RoundingDivideByPOT + saturation -> t3 ----
    wire  [W-1:0]  p1 = s1p + c1p + W'(k1p);
    logic [10:0]   z0, z1;
    always_ff @(posedge clk) begin
        z0 <= rdsat(y0, r0d);
        z1 <= rdsat(p1, r1d);
    end
    // ---- t3 -> t4: offset, clamp ----
    always_ff @(posedge clk) begin
        out_q0 <= outq(z0, ooff, amin, amax);
        out_q1 <= outq(z1, ooff, amin, amax);
    end
endmodule

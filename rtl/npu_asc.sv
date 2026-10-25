// npu_asc - fused-ADD input scaler, 2 stages: raw = MultiplyByQuantizedMultiplier(v << 20, mult, shift)
// for a 9-bit v (int8 + offset), mult >= 0, shift <= 0 (a positive input shift is part of the << 20):
// SRDHM(v << 20, mult) = (v * mult + 2^10) >>> 11 exactly (|v << 20| < 2^31), then RoundingDivideByPOT
// by r = -shift <= 3 (the driver checks). raw is valid 2 cycles after in_valid (as npu_drq's ADD-scale mode).
module npu_asc (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        in_valid,
    input  logic [8:0]  v,
    input  logic [30:0] mult,
    input  logic [7:0]  shift,
    output logic        raw_valid,
    output logic [31:0] raw
);
    // ---- stage 1: v * mult + 2^10, radix-4 Booth digits of v (5 rows of +-m, +-2m) ----
    // (v44R) Booth rows pp_k (33 bits, pp_k[32] = sign) and their +1 corrections, summed by a Dadda tree of sky130
    //  full-adder cells (gen: v44_R/tools/p_asc.py): pr = sum_k sext(pp_k) << 2k + bn_k << 2k + 2^10
    logic [32:0] ap [5];
    logic [4:0]  abn;
    always_comb begin
        logic [2:0] bt;
        logic       b1, b2, bn;
        logic [32:0] mg;
        logic [9:0] vx;
        vx = {v[8], v};                                    // sign-extended to 10 bits (5 digits)
        for (int k = 0; k < 5; k++) begin
            bt = {vx[2*k+1], vx[2*k], (k == 0) ? 1'b0 : vx[(k == 0) ? 0 : 2*k-1]};
            b1 = bt[1] ^ bt[0]; b2 = (bt == 3'b011) || (bt == 3'b100); bn = bt[2] && !(bt[1] && bt[0]);
            mg = ({33{b1}} & {2'b00, mult}) | ({33{b2}} & {1'b0, mult, 1'b0});
            ap[k] = mg ^ {33{bn}};
            abn[k] = bn;
        end
    end
    wire a_s0, a_c0; sky130_fd_sc_hd__fa_1 a_u0 (.A(ap[0][6]), .B(ap[1][4]), .CIN(1'b0), .SUM(a_s0), .COUT(a_c0));
    wire a_s1, a_c1; sky130_fd_sc_hd__fa_1 a_u1 (.A(ap[0][7]), .B(ap[1][5]), .CIN(1'b0), .SUM(a_s1), .COUT(a_c1));
    wire a_s2, a_c2; sky130_fd_sc_hd__fa_1 a_u2 (.A(ap[0][8]), .B(ap[1][6]), .CIN(ap[2][4]), .SUM(a_s2), .COUT(a_c2));
    wire a_s3, a_c3; sky130_fd_sc_hd__fa_1 a_u3 (.A(ap[3][2]), .B(ap[4][0]), .CIN(1'b0), .SUM(a_s3), .COUT(a_c3));
    wire a_s4, a_c4; sky130_fd_sc_hd__fa_1 a_u4 (.A(ap[0][9]), .B(ap[1][7]), .CIN(ap[2][5]), .SUM(a_s4), .COUT(a_c4));
    wire a_s5, a_c5; sky130_fd_sc_hd__fa_1 a_u5 (.A(ap[3][3]), .B(ap[4][1]), .CIN(1'b0), .SUM(a_s5), .COUT(a_c5));
    wire a_s6, a_c6; sky130_fd_sc_hd__fa_1 a_u6 (.A(ap[0][10]), .B(ap[1][8]), .CIN(ap[2][6]), .SUM(a_s6), .COUT(a_c6));
    wire a_s7, a_c7; sky130_fd_sc_hd__fa_1 a_u7 (.A(ap[3][4]), .B(ap[4][2]), .CIN(1'b1), .SUM(a_s7), .COUT(a_c7));
    wire a_s8, a_c8; sky130_fd_sc_hd__fa_1 a_u8 (.A(ap[0][11]), .B(ap[1][9]), .CIN(ap[2][7]), .SUM(a_s8), .COUT(a_c8));
    wire a_s9, a_c9; sky130_fd_sc_hd__fa_1 a_u9 (.A(ap[3][5]), .B(ap[4][3]), .CIN(1'b0), .SUM(a_s9), .COUT(a_c9));
    wire a_s10, a_c10; sky130_fd_sc_hd__fa_1 a_u10 (.A(ap[0][12]), .B(ap[1][10]), .CIN(ap[2][8]), .SUM(a_s10), .COUT(a_c10));
    wire a_s11, a_c11; sky130_fd_sc_hd__fa_1 a_u11 (.A(ap[3][6]), .B(ap[4][4]), .CIN(1'b0), .SUM(a_s11), .COUT(a_c11));
    wire a_s12, a_c12; sky130_fd_sc_hd__fa_1 a_u12 (.A(ap[0][13]), .B(ap[1][11]), .CIN(ap[2][9]), .SUM(a_s12), .COUT(a_c12));
    wire a_s13, a_c13; sky130_fd_sc_hd__fa_1 a_u13 (.A(ap[3][7]), .B(ap[4][5]), .CIN(1'b0), .SUM(a_s13), .COUT(a_c13));
    wire a_s14, a_c14; sky130_fd_sc_hd__fa_1 a_u14 (.A(ap[0][14]), .B(ap[1][12]), .CIN(ap[2][10]), .SUM(a_s14), .COUT(a_c14));
    wire a_s15, a_c15; sky130_fd_sc_hd__fa_1 a_u15 (.A(ap[3][8]), .B(ap[4][6]), .CIN(1'b0), .SUM(a_s15), .COUT(a_c15));
    wire a_s16, a_c16; sky130_fd_sc_hd__fa_1 a_u16 (.A(ap[0][15]), .B(ap[1][13]), .CIN(ap[2][11]), .SUM(a_s16), .COUT(a_c16));
    wire a_s17, a_c17; sky130_fd_sc_hd__fa_1 a_u17 (.A(ap[3][9]), .B(ap[4][7]), .CIN(1'b0), .SUM(a_s17), .COUT(a_c17));
    wire a_s18, a_c18; sky130_fd_sc_hd__fa_1 a_u18 (.A(ap[0][16]), .B(ap[1][14]), .CIN(ap[2][12]), .SUM(a_s18), .COUT(a_c18));
    wire a_s19, a_c19; sky130_fd_sc_hd__fa_1 a_u19 (.A(ap[3][10]), .B(ap[4][8]), .CIN(1'b0), .SUM(a_s19), .COUT(a_c19));
    wire a_s20, a_c20; sky130_fd_sc_hd__fa_1 a_u20 (.A(ap[0][17]), .B(ap[1][15]), .CIN(ap[2][13]), .SUM(a_s20), .COUT(a_c20));
    wire a_s21, a_c21; sky130_fd_sc_hd__fa_1 a_u21 (.A(ap[3][11]), .B(ap[4][9]), .CIN(1'b0), .SUM(a_s21), .COUT(a_c21));
    wire a_s22, a_c22; sky130_fd_sc_hd__fa_1 a_u22 (.A(ap[0][18]), .B(ap[1][16]), .CIN(ap[2][14]), .SUM(a_s22), .COUT(a_c22));
    wire a_s23, a_c23; sky130_fd_sc_hd__fa_1 a_u23 (.A(ap[3][12]), .B(ap[4][10]), .CIN(1'b0), .SUM(a_s23), .COUT(a_c23));
    wire a_s24, a_c24; sky130_fd_sc_hd__fa_1 a_u24 (.A(ap[0][19]), .B(ap[1][17]), .CIN(ap[2][15]), .SUM(a_s24), .COUT(a_c24));
    wire a_s25, a_c25; sky130_fd_sc_hd__fa_1 a_u25 (.A(ap[3][13]), .B(ap[4][11]), .CIN(1'b0), .SUM(a_s25), .COUT(a_c25));
    wire a_s26, a_c26; sky130_fd_sc_hd__fa_1 a_u26 (.A(ap[0][20]), .B(ap[1][18]), .CIN(ap[2][16]), .SUM(a_s26), .COUT(a_c26));
    wire a_s27, a_c27; sky130_fd_sc_hd__fa_1 a_u27 (.A(ap[3][14]), .B(ap[4][12]), .CIN(1'b0), .SUM(a_s27), .COUT(a_c27));
    wire a_s28, a_c28; sky130_fd_sc_hd__fa_1 a_u28 (.A(ap[0][21]), .B(ap[1][19]), .CIN(ap[2][17]), .SUM(a_s28), .COUT(a_c28));
    wire a_s29, a_c29; sky130_fd_sc_hd__fa_1 a_u29 (.A(ap[3][15]), .B(ap[4][13]), .CIN(1'b0), .SUM(a_s29), .COUT(a_c29));
    wire a_s30, a_c30; sky130_fd_sc_hd__fa_1 a_u30 (.A(ap[0][22]), .B(ap[1][20]), .CIN(ap[2][18]), .SUM(a_s30), .COUT(a_c30));
    wire a_s31, a_c31; sky130_fd_sc_hd__fa_1 a_u31 (.A(ap[3][16]), .B(ap[4][14]), .CIN(1'b0), .SUM(a_s31), .COUT(a_c31));
    wire a_s32, a_c32; sky130_fd_sc_hd__fa_1 a_u32 (.A(ap[0][23]), .B(ap[1][21]), .CIN(ap[2][19]), .SUM(a_s32), .COUT(a_c32));
    wire a_s33, a_c33; sky130_fd_sc_hd__fa_1 a_u33 (.A(ap[3][17]), .B(ap[4][15]), .CIN(1'b0), .SUM(a_s33), .COUT(a_c33));
    wire a_s34, a_c34; sky130_fd_sc_hd__fa_1 a_u34 (.A(ap[0][24]), .B(ap[1][22]), .CIN(ap[2][20]), .SUM(a_s34), .COUT(a_c34));
    wire a_s35, a_c35; sky130_fd_sc_hd__fa_1 a_u35 (.A(ap[3][18]), .B(ap[4][16]), .CIN(1'b0), .SUM(a_s35), .COUT(a_c35));
    wire a_s36, a_c36; sky130_fd_sc_hd__fa_1 a_u36 (.A(ap[0][25]), .B(ap[1][23]), .CIN(ap[2][21]), .SUM(a_s36), .COUT(a_c36));
    wire a_s37, a_c37; sky130_fd_sc_hd__fa_1 a_u37 (.A(ap[3][19]), .B(ap[4][17]), .CIN(1'b0), .SUM(a_s37), .COUT(a_c37));
    wire a_s38, a_c38; sky130_fd_sc_hd__fa_1 a_u38 (.A(ap[0][26]), .B(ap[1][24]), .CIN(ap[2][22]), .SUM(a_s38), .COUT(a_c38));
    wire a_s39, a_c39; sky130_fd_sc_hd__fa_1 a_u39 (.A(ap[3][20]), .B(ap[4][18]), .CIN(1'b0), .SUM(a_s39), .COUT(a_c39));
    wire a_s40, a_c40; sky130_fd_sc_hd__fa_1 a_u40 (.A(ap[0][27]), .B(ap[1][25]), .CIN(ap[2][23]), .SUM(a_s40), .COUT(a_c40));
    wire a_s41, a_c41; sky130_fd_sc_hd__fa_1 a_u41 (.A(ap[3][21]), .B(ap[4][19]), .CIN(1'b0), .SUM(a_s41), .COUT(a_c41));
    wire a_s42, a_c42; sky130_fd_sc_hd__fa_1 a_u42 (.A(ap[0][28]), .B(ap[1][26]), .CIN(ap[2][24]), .SUM(a_s42), .COUT(a_c42));
    wire a_s43, a_c43; sky130_fd_sc_hd__fa_1 a_u43 (.A(ap[3][22]), .B(ap[4][20]), .CIN(1'b0), .SUM(a_s43), .COUT(a_c43));
    wire a_s44, a_c44; sky130_fd_sc_hd__fa_1 a_u44 (.A(ap[0][29]), .B(ap[1][27]), .CIN(ap[2][25]), .SUM(a_s44), .COUT(a_c44));
    wire a_s45, a_c45; sky130_fd_sc_hd__fa_1 a_u45 (.A(ap[3][23]), .B(ap[4][21]), .CIN(1'b0), .SUM(a_s45), .COUT(a_c45));
    wire a_s46, a_c46; sky130_fd_sc_hd__fa_1 a_u46 (.A(ap[0][30]), .B(ap[1][28]), .CIN(ap[2][26]), .SUM(a_s46), .COUT(a_c46));
    wire a_s47, a_c47; sky130_fd_sc_hd__fa_1 a_u47 (.A(ap[3][24]), .B(ap[4][22]), .CIN(1'b0), .SUM(a_s47), .COUT(a_c47));
    wire a_s48, a_c48; sky130_fd_sc_hd__fa_1 a_u48 (.A(ap[0][31]), .B(ap[1][29]), .CIN(ap[2][27]), .SUM(a_s48), .COUT(a_c48));
    wire a_s49, a_c49; sky130_fd_sc_hd__fa_1 a_u49 (.A(ap[3][25]), .B(ap[4][23]), .CIN(1'b0), .SUM(a_s49), .COUT(a_c49));
    wire a_s50, a_c50; sky130_fd_sc_hd__fa_1 a_u50 (.A(~ap[0][32]), .B(ap[1][30]), .CIN(ap[2][28]), .SUM(a_s50), .COUT(a_c50));
    wire a_s51, a_c51; sky130_fd_sc_hd__fa_1 a_u51 (.A(ap[3][26]), .B(ap[4][24]), .CIN(1'b1), .SUM(a_s51), .COUT(a_c51));
    wire a_s52, a_c52; sky130_fd_sc_hd__fa_1 a_u52 (.A(ap[1][31]), .B(ap[2][29]), .CIN(ap[3][27]), .SUM(a_s52), .COUT(a_c52));
    wire a_s53, a_c53; sky130_fd_sc_hd__fa_1 a_u53 (.A(ap[4][25]), .B(1'b1), .CIN(1'b0), .SUM(a_s53), .COUT(a_c53));
    wire a_s54, a_c54; sky130_fd_sc_hd__fa_1 a_u54 (.A(~ap[1][32]), .B(ap[2][30]), .CIN(ap[3][28]), .SUM(a_s54), .COUT(a_c54));
    wire a_s55, a_c55; sky130_fd_sc_hd__fa_1 a_u55 (.A(ap[2][31]), .B(ap[3][29]), .CIN(1'b0), .SUM(a_s55), .COUT(a_c55));
    wire a_s56, a_c56; sky130_fd_sc_hd__fa_1 a_u56 (.A(ap[0][4]), .B(ap[1][2]), .CIN(1'b0), .SUM(a_s56), .COUT(a_c56));
    wire a_s57, a_c57; sky130_fd_sc_hd__fa_1 a_u57 (.A(ap[0][5]), .B(ap[1][3]), .CIN(1'b0), .SUM(a_s57), .COUT(a_c57));
    wire a_s58, a_c58; sky130_fd_sc_hd__fa_1 a_u58 (.A(a_s0), .B(ap[2][2]), .CIN(ap[3][0]), .SUM(a_s58), .COUT(a_c58));
    wire a_s59, a_c59; sky130_fd_sc_hd__fa_1 a_u59 (.A(a_c0), .B(a_s1), .CIN(ap[2][3]), .SUM(a_s59), .COUT(a_c59));
    wire a_s60, a_c60; sky130_fd_sc_hd__fa_1 a_u60 (.A(a_c1), .B(a_s2), .CIN(a_s3), .SUM(a_s60), .COUT(a_c60));
    wire a_s61, a_c61; sky130_fd_sc_hd__fa_1 a_u61 (.A(a_c2), .B(a_c3), .CIN(a_s4), .SUM(a_s61), .COUT(a_c61));
    wire a_s62, a_c62; sky130_fd_sc_hd__fa_1 a_u62 (.A(a_c4), .B(a_c5), .CIN(a_s6), .SUM(a_s62), .COUT(a_c62));
    wire a_s63, a_c63; sky130_fd_sc_hd__fa_1 a_u63 (.A(a_c6), .B(a_c7), .CIN(a_s8), .SUM(a_s63), .COUT(a_c63));
    wire a_s64, a_c64; sky130_fd_sc_hd__fa_1 a_u64 (.A(a_c8), .B(a_c9), .CIN(a_s10), .SUM(a_s64), .COUT(a_c64));
    wire a_s65, a_c65; sky130_fd_sc_hd__fa_1 a_u65 (.A(a_c10), .B(a_c11), .CIN(a_s12), .SUM(a_s65), .COUT(a_c65));
    wire a_s66, a_c66; sky130_fd_sc_hd__fa_1 a_u66 (.A(a_c12), .B(a_c13), .CIN(a_s14), .SUM(a_s66), .COUT(a_c66));
    wire a_s67, a_c67; sky130_fd_sc_hd__fa_1 a_u67 (.A(a_c14), .B(a_c15), .CIN(a_s16), .SUM(a_s67), .COUT(a_c67));
    wire a_s68, a_c68; sky130_fd_sc_hd__fa_1 a_u68 (.A(a_c16), .B(a_c17), .CIN(a_s18), .SUM(a_s68), .COUT(a_c68));
    wire a_s69, a_c69; sky130_fd_sc_hd__fa_1 a_u69 (.A(a_c18), .B(a_c19), .CIN(a_s20), .SUM(a_s69), .COUT(a_c69));
    wire a_s70, a_c70; sky130_fd_sc_hd__fa_1 a_u70 (.A(a_c20), .B(a_c21), .CIN(a_s22), .SUM(a_s70), .COUT(a_c70));
    wire a_s71, a_c71; sky130_fd_sc_hd__fa_1 a_u71 (.A(a_c22), .B(a_c23), .CIN(a_s24), .SUM(a_s71), .COUT(a_c71));
    wire a_s72, a_c72; sky130_fd_sc_hd__fa_1 a_u72 (.A(a_c24), .B(a_c25), .CIN(a_s26), .SUM(a_s72), .COUT(a_c72));
    wire a_s73, a_c73; sky130_fd_sc_hd__fa_1 a_u73 (.A(a_c26), .B(a_c27), .CIN(a_s28), .SUM(a_s73), .COUT(a_c73));
    wire a_s74, a_c74; sky130_fd_sc_hd__fa_1 a_u74 (.A(a_c28), .B(a_c29), .CIN(a_s30), .SUM(a_s74), .COUT(a_c74));
    wire a_s75, a_c75; sky130_fd_sc_hd__fa_1 a_u75 (.A(a_c30), .B(a_c31), .CIN(a_s32), .SUM(a_s75), .COUT(a_c75));
    wire a_s76, a_c76; sky130_fd_sc_hd__fa_1 a_u76 (.A(a_c32), .B(a_c33), .CIN(a_s34), .SUM(a_s76), .COUT(a_c76));
    wire a_s77, a_c77; sky130_fd_sc_hd__fa_1 a_u77 (.A(a_c34), .B(a_c35), .CIN(a_s36), .SUM(a_s77), .COUT(a_c77));
    wire a_s78, a_c78; sky130_fd_sc_hd__fa_1 a_u78 (.A(a_c36), .B(a_c37), .CIN(a_s38), .SUM(a_s78), .COUT(a_c78));
    wire a_s79, a_c79; sky130_fd_sc_hd__fa_1 a_u79 (.A(a_c38), .B(a_c39), .CIN(a_s40), .SUM(a_s79), .COUT(a_c79));
    wire a_s80, a_c80; sky130_fd_sc_hd__fa_1 a_u80 (.A(a_c40), .B(a_c41), .CIN(a_s42), .SUM(a_s80), .COUT(a_c80));
    wire a_s81, a_c81; sky130_fd_sc_hd__fa_1 a_u81 (.A(a_c42), .B(a_c43), .CIN(a_s44), .SUM(a_s81), .COUT(a_c81));
    wire a_s82, a_c82; sky130_fd_sc_hd__fa_1 a_u82 (.A(a_c44), .B(a_c45), .CIN(a_s46), .SUM(a_s82), .COUT(a_c82));
    wire a_s83, a_c83; sky130_fd_sc_hd__fa_1 a_u83 (.A(a_c46), .B(a_c47), .CIN(a_s48), .SUM(a_s83), .COUT(a_c83));
    wire a_s84, a_c84; sky130_fd_sc_hd__fa_1 a_u84 (.A(a_c48), .B(a_c49), .CIN(a_s50), .SUM(a_s84), .COUT(a_c84));
    wire a_s85, a_c85; sky130_fd_sc_hd__fa_1 a_u85 (.A(a_c50), .B(a_c51), .CIN(a_s52), .SUM(a_s85), .COUT(a_c85));
    wire a_s86, a_c86; sky130_fd_sc_hd__fa_1 a_u86 (.A(a_c52), .B(a_c53), .CIN(a_s54), .SUM(a_s86), .COUT(a_c86));
    wire a_s87, a_c87; sky130_fd_sc_hd__fa_1 a_u87 (.A(a_c54), .B(a_s55), .CIN(ap[4][27]), .SUM(a_s87), .COUT(a_c87));
    wire a_s88, a_c88; sky130_fd_sc_hd__fa_1 a_u88 (.A(a_c55), .B(~ap[2][32]), .CIN(ap[3][30]), .SUM(a_s88), .COUT(a_c88));
    wire a_s89, a_c89; sky130_fd_sc_hd__fa_1 a_u89 (.A(ap[3][31]), .B(ap[4][29]), .CIN(1'b0), .SUM(a_s89), .COUT(a_c89));
    wire a_s90, a_c90; sky130_fd_sc_hd__fa_1 a_u90 (.A(ap[0][2]), .B(ap[1][0]), .CIN(1'b0), .SUM(a_s90), .COUT(a_c90));
    wire a_s91, a_c91; sky130_fd_sc_hd__fa_1 a_u91 (.A(ap[0][3]), .B(ap[1][1]), .CIN(1'b0), .SUM(a_s91), .COUT(a_c91));
    wire a_s92, a_c92; sky130_fd_sc_hd__fa_1 a_u92 (.A(a_s56), .B(ap[2][0]), .CIN(abn[2]), .SUM(a_s92), .COUT(a_c92));
    wire a_s93, a_c93; sky130_fd_sc_hd__fa_1 a_u93 (.A(a_c56), .B(a_s57), .CIN(ap[2][1]), .SUM(a_s93), .COUT(a_c93));
    wire a_s94, a_c94; sky130_fd_sc_hd__fa_1 a_u94 (.A(a_c57), .B(a_s58), .CIN(abn[3]), .SUM(a_s94), .COUT(a_c94));
    wire a_s95, a_c95; sky130_fd_sc_hd__fa_1 a_u95 (.A(a_c58), .B(a_s59), .CIN(ap[3][1]), .SUM(a_s95), .COUT(a_c95));
    wire a_s96, a_c96; sky130_fd_sc_hd__fa_1 a_u96 (.A(a_c59), .B(a_s60), .CIN(abn[4]), .SUM(a_s96), .COUT(a_c96));
    wire a_s97, a_c97; sky130_fd_sc_hd__fa_1 a_u97 (.A(a_c60), .B(a_s61), .CIN(a_s5), .SUM(a_s97), .COUT(a_c97));
    wire a_s98, a_c98; sky130_fd_sc_hd__fa_1 a_u98 (.A(a_c61), .B(a_s62), .CIN(a_s7), .SUM(a_s98), .COUT(a_c98));
    wire a_s99, a_c99; sky130_fd_sc_hd__fa_1 a_u99 (.A(a_c62), .B(a_s63), .CIN(a_s9), .SUM(a_s99), .COUT(a_c99));
    wire a_s100, a_c100; sky130_fd_sc_hd__fa_1 a_u100 (.A(a_c63), .B(a_s64), .CIN(a_s11), .SUM(a_s100), .COUT(a_c100));
    wire a_s101, a_c101; sky130_fd_sc_hd__fa_1 a_u101 (.A(a_c64), .B(a_s65), .CIN(a_s13), .SUM(a_s101), .COUT(a_c101));
    wire a_s102, a_c102; sky130_fd_sc_hd__fa_1 a_u102 (.A(a_c65), .B(a_s66), .CIN(a_s15), .SUM(a_s102), .COUT(a_c102));
    wire a_s103, a_c103; sky130_fd_sc_hd__fa_1 a_u103 (.A(a_c66), .B(a_s67), .CIN(a_s17), .SUM(a_s103), .COUT(a_c103));
    wire a_s104, a_c104; sky130_fd_sc_hd__fa_1 a_u104 (.A(a_c67), .B(a_s68), .CIN(a_s19), .SUM(a_s104), .COUT(a_c104));
    wire a_s105, a_c105; sky130_fd_sc_hd__fa_1 a_u105 (.A(a_c68), .B(a_s69), .CIN(a_s21), .SUM(a_s105), .COUT(a_c105));
    wire a_s106, a_c106; sky130_fd_sc_hd__fa_1 a_u106 (.A(a_c69), .B(a_s70), .CIN(a_s23), .SUM(a_s106), .COUT(a_c106));
    wire a_s107, a_c107; sky130_fd_sc_hd__fa_1 a_u107 (.A(a_c70), .B(a_s71), .CIN(a_s25), .SUM(a_s107), .COUT(a_c107));
    wire a_s108, a_c108; sky130_fd_sc_hd__fa_1 a_u108 (.A(a_c71), .B(a_s72), .CIN(a_s27), .SUM(a_s108), .COUT(a_c108));
    wire a_s109, a_c109; sky130_fd_sc_hd__fa_1 a_u109 (.A(a_c72), .B(a_s73), .CIN(a_s29), .SUM(a_s109), .COUT(a_c109));
    wire a_s110, a_c110; sky130_fd_sc_hd__fa_1 a_u110 (.A(a_c73), .B(a_s74), .CIN(a_s31), .SUM(a_s110), .COUT(a_c110));
    wire a_s111, a_c111; sky130_fd_sc_hd__fa_1 a_u111 (.A(a_c74), .B(a_s75), .CIN(a_s33), .SUM(a_s111), .COUT(a_c111));
    wire a_s112, a_c112; sky130_fd_sc_hd__fa_1 a_u112 (.A(a_c75), .B(a_s76), .CIN(a_s35), .SUM(a_s112), .COUT(a_c112));
    wire a_s113, a_c113; sky130_fd_sc_hd__fa_1 a_u113 (.A(a_c76), .B(a_s77), .CIN(a_s37), .SUM(a_s113), .COUT(a_c113));
    wire a_s114, a_c114; sky130_fd_sc_hd__fa_1 a_u114 (.A(a_c77), .B(a_s78), .CIN(a_s39), .SUM(a_s114), .COUT(a_c114));
    wire a_s115, a_c115; sky130_fd_sc_hd__fa_1 a_u115 (.A(a_c78), .B(a_s79), .CIN(a_s41), .SUM(a_s115), .COUT(a_c115));
    wire a_s116, a_c116; sky130_fd_sc_hd__fa_1 a_u116 (.A(a_c79), .B(a_s80), .CIN(a_s43), .SUM(a_s116), .COUT(a_c116));
    wire a_s117, a_c117; sky130_fd_sc_hd__fa_1 a_u117 (.A(a_c80), .B(a_s81), .CIN(a_s45), .SUM(a_s117), .COUT(a_c117));
    wire a_s118, a_c118; sky130_fd_sc_hd__fa_1 a_u118 (.A(a_c81), .B(a_s82), .CIN(a_s47), .SUM(a_s118), .COUT(a_c118));
    wire a_s119, a_c119; sky130_fd_sc_hd__fa_1 a_u119 (.A(a_c82), .B(a_s83), .CIN(a_s49), .SUM(a_s119), .COUT(a_c119));
    wire a_s120, a_c120; sky130_fd_sc_hd__fa_1 a_u120 (.A(a_c83), .B(a_s84), .CIN(a_s51), .SUM(a_s120), .COUT(a_c120));
    wire a_s121, a_c121; sky130_fd_sc_hd__fa_1 a_u121 (.A(a_c84), .B(a_s85), .CIN(a_s53), .SUM(a_s121), .COUT(a_c121));
    wire a_s122, a_c122; sky130_fd_sc_hd__fa_1 a_u122 (.A(a_c85), .B(a_s86), .CIN(ap[4][26]), .SUM(a_s122), .COUT(a_c122));
    wire a_s123, a_c123; sky130_fd_sc_hd__fa_1 a_u123 (.A(a_c86), .B(a_s87), .CIN(1'b1), .SUM(a_s123), .COUT(a_c123));
    wire a_s124, a_c124; sky130_fd_sc_hd__fa_1 a_u124 (.A(a_c87), .B(a_s88), .CIN(ap[4][28]), .SUM(a_s124), .COUT(a_c124));
    wire a_s125, a_c125; sky130_fd_sc_hd__fa_1 a_u125 (.A(a_c88), .B(a_s89), .CIN(1'b1), .SUM(a_s125), .COUT(a_c125));
    wire a_s126, a_c126; sky130_fd_sc_hd__fa_1 a_u126 (.A(a_c89), .B(~ap[3][32]), .CIN(ap[4][30]), .SUM(a_s126), .COUT(a_c126));
    wire a_s127, a_c127; sky130_fd_sc_hd__fa_1 a_u127 (.A(ap[4][31]), .B(1'b1), .CIN(1'b0), .SUM(a_s127), .COUT(a_c127));
    wire  [40:0] as_ = {a_c127, a_c126, a_c125, a_c124, a_c123, a_c122, a_c121, a_c120, a_c119, a_c118, a_c117, a_c116, a_c115, a_c114, a_c113, a_c112, a_c111, a_c110, a_c109, a_c108, a_c107, a_c106, a_c105, a_c104, a_c103, a_c102, a_c101, a_c100, a_c99, a_c98, a_c97, a_c96, a_c95, a_c94, a_c93, a_c92, a_c91, a_c90, a_s90, ap[0][1], ap[0][0]};
    wire  [40:0] ac_ = {~ap[4][32], a_s127, a_s126, a_s125, a_s124, a_s123, a_s122, a_s121, a_s120, a_s119, a_s118, a_s117, a_s116, a_s115, a_s114, a_s113, a_s112, a_s111, a_s110, a_s109, a_s108, a_s107, a_s106, a_s105, a_s104, a_s103, a_s102, a_s101, a_s100, a_s99, a_s98, a_s97, a_s96, a_s95, a_s94, a_s93, a_s92, a_s91, abn[1], 1'b0, abn[0]};
    wire  [40:0] pr = as_ + ac_;
    logic        v1, v2;
    logic [29:0] y1;
    logic [1:0]  r1;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin v1 <= 1'b0; v2 <= 1'b0; end
        else begin v1 <= in_valid; v2 <= v1; end
    end
    always_ff @(posedge clk) begin
        y1 <= pr[40:11];
        r1 <= shift[7] ? 2'(8'd0 - shift) : 2'd0;
    end
    // ---- stage 2: RoundingDivideByPOT ----
    logic [29:0] mask, rem, thr, ysh, z, z2;
    always_comb begin
        mask = {28'd0, (r1 == 2'd3), (r1[1])} | {29'd0, (r1 != 2'd0)};   // 2^r1 - 1
        rem  = y1 & mask;
        thr  = (mask >> 1) + {29'd0, y1[29]};
        ysh  = $signed(y1) >>> r1;
        z    = ysh + ((rem > thr) ? 30'd1 : 30'd0);
    end
    always_ff @(posedge clk) z2 <= z;
    assign raw_valid = v2;
    assign raw       = {{2{z2[29]}}, z2};
endmodule

// npu_kcp - (v55_KC) OP_KCP: run-time constant-channel probe of a 1x1 conv layer (owner rule 2: computed at run time
// from the layer's PASSED bias / multiplier / shift / weight tensors, nothing shipped).
//
// Channel c is CONSTANT when its requantised output is the same for every input: with R = 2^15 K (any int8 weight,
// |x + in_off| <= 255 for in_off in [-127, 128]: |sum| <= 128 * 255 * K < R) every accumulator lies in
// [b - R, b + R]; with shift <= 0 (no left shift), multiplier >= 0 and no int32 wrap of b +- R, TFLM's requantisation
// is monotone there, so out(b - R) == out(b + R) (units A / B, the layer's own out_off / clamps) proves it. The rest
// are LIVE. Per layer (Co 128 / 256, K = 128 / 256, Kcap <= 64; the kernel checks):
//   L   the previous probe's one-hot bytes are cleared (log, 144 bytes at p_lg: 16-bit count + up to 64 16-bit offsets)
//   S   starts[c] (int32 at p_st) = constant value, or 0 for a live channel; live channel c, j-th (j < Kcap):
//       list[j] = c (byte at p_lst), one-hot byte 1 at p_oh + c Kcap + j, log entry j = c Kcap + j.
//       More than Kcap live channels: the sticky flag (ovf) is set, the rest is not recorded.
//   G   for every recorded j: the weight row c (K bytes) -> p_cw + j K, b[c] / M[c] / S[c] -> p_cb / p_cm / p_cs + 4 j
// The compact conv (Kcap channels: cw / cb / cm / cs) and the scatter conv (K = Kcap, one-hot rows, starts as biases,
// multiplier 2^31 - 1, shift 0) then write the layer's exact output; rows / entries j >= L hold stale data whose
// compact results no one-hot row selects.
// Streams: 0 = 16-byte b / S chunks of 4 channels (S read through out_data2, the word of the channel by peek), then
// the gathered rows / words; 1 = log, 64-byte M blocks (4 chunks), list. One address adder per command port.
module npu_kcp (
    input  logic         clk,
    input  logic         rst_n,
    input  logic         start,
    input  logic [19:0]  p_wgt, p_bias, p_mult, p_shift,
    input  logic [19:0]  p_st, p_lst, p_cb, p_cm, p_cs, p_cw, p_oh, p_lg,
    input  logic [8:0]   p_co,         // 128 / 256
    input  logic [12:0]  p_k,          // 128 / 256
    input  logic [6:0]   p_kcap,
    output logic [1:0]   rc_valid,
    input  logic [1:0]   rc_ready,
    output logic [63:0]  rc_addr,
    output logic [39:0]  rc_len,
    input  logic [7:0]   rs0_avail, rs1_avail,
    input  logic [127:0] rs0_data, rs0_data2, rs1_data,
    output logic [5:0]   rs0_pop,
    output logic [7:0]   rs0_peek,
    output logic [4:0]   rs1_pop,
    output logic         wc_valid,
    input  logic         wc_ready,
    output logic [31:0]  wc_addr,
    output logic [19:0]  wc_len,
    output logic         wd_valid,
    input  logic         wd_ready,
    output logic [127:0] wd_data,
    output logic [4:0]   wd_n,
    input  logic         wr_idle,
    output logic         rq_v,
    output logic [31:0]  rq_acc, rqb_acc, rq_mult,
    output logic [7:0]   rq_shift,
    input  logic         rq_ov,
    input  logic [7:0]   rq_q, rqb_q,
    output logic         busy,
    output logic         ovf,
    output logic [6:0]   lrec_o       // live channels recorded by the last probe (held until the next one)
);
    typedef enum logic [2:0] {K_IDLE, K_LOG, K_SCAN, K_FIN, K_WAIT, K_GATH} kst_t;
    kst_t st;
    assign busy = (st != K_IDLE);
    assign lrec_o = lrec;
    wire         k8   = p_k[8];                           // K = 256 (else 128)
    wire  [6:0]  nq   = p_co[8:2];                        // chunks of 4 channels

    // ---------------- L ----------------
    logic        lg_cmd, lg_cnt_v;
    logic [6:0]  lg_cnt, lg_e;
    logic [7:0]  lg_drop;
    wire  [15:0] lg_w = rs1_data[15:0];

    // ---------------- S ----------------
    logic [6:0]  qi0, qi1, qc, qe, lrec;
    logic        qi0h, es;
    logic [1:0]  ci, ei, ek;
    logic [15:0] ohq;                                     // qe * 4 Kcap
    logic [3:0]  okd, pd;
    logic [1:0]  sd [4];
    wire         r_cst = okd[3] && (rq_q == rqb_q);
    wire         r_p   = pd[3];
    wire  [1:0]  r_s   = sd[3];
    logic [31:0] rt   [2];
    logic [3:0]  rcst [2];
    // (results: one gated bank per (parity, slot), written by the units' output; no enable flops)
    for (genvar p = 0; p < 2; p++) begin : g_rp
        for (genvar q = 0; q < 4; q++) begin : g_rs
            wire g = clk | ~(rq_ov && r_p == 1'(p) && r_s == 2'(q));
            logic [8:0] v;
            always_ff @(posedge g) v <= {r_cst, rq_q};
            assign rt[p][q*8 +: 8] = v[7:0];
            assign rcst[p][q] = v[8];
        end
    end
    logic [2:0]  rn   [2];
    wire  [31:0] b_w  = rs0_data[31:0];                   // (peek = 4 ci)
    wire  [31:0] s_w  = rs0_data2[31:0];
    wire  [31:0] m_w  = rs1_data[ci*32 +: 32];
    wire  [32:0] R    = k8 ? 33'h0_0080_0000 : 33'h0_0040_0000;
    wire  [32:0] lo   = {b_w[31], b_w} - R;
    wire  [32:0] hi   = {b_w[31], b_w} + R;
    wire         ok   = (lo[32] == lo[31]) && (hi[32] == hi[31]) && !m_w[31] &&
                        (s_w[31] ? (s_w[30:5] == 26'h3ffffff) : (s_w == 32'd0));   // shift in [-31, 0]
    wire         c_go = (st == K_SCAN) && (qc != nq) && (rs0_avail >= 8'd32) && (rs1_avail >= 8'd16) &&
                        ((qc - qe) < 7'd2);
    assign rq_v = c_go; assign rq_acc = lo[31:0]; assign rqb_acc = hi[31:0]; assign rq_mult = m_w; assign rq_shift = s_w[7:0];
    assign rs0_peek = (st == K_SCAN) ? {4'd0, ci, 2'b00} : 8'd0;
    always_ff @(posedge clk) begin
        okd <= {okd[2:0], ok}; pd <= {pd[2:0], qc[0]};
        sd[0] <= ci; sd[1] <= sd[0]; sd[2] <= sd[1]; sd[3] <= sd[2];
    end
    wire         e_full = (rn[qe[0]] == 3'd4);
    wire  [3:0]  e_live = ~rcst[qe[0]];
    logic        e_has;
    logic [1:0]  e_nx;
    always_comb begin
        e_has = 1'b0; e_nx = 2'd0;
        for (int i = 3; i >= 0; i--) if (e_live[i] && (2'(i) >= ei)) begin e_has = 1'b1; e_nx = 2'(i); end
    end
    wire  [7:0]  e_c   = {qe[5:0], e_nx};
    wire  [8:0]  e_ik  = (e_nx[1] ? {1'b0, p_kcap, 1'b0} : 9'd0) + (e_nx[0] ? {2'd0, p_kcap} : 9'd0);
    wire  [15:0] e_off = ohq + {7'd0, e_ik} + {9'd0, lrec};
    wire         e_rec = (lrec != p_kcap);
    logic [127:0] e_beat;
    always_comb
        for (int i = 0; i < 4; i++)
            e_beat[i*32 +: 32] = {32{rcst[qe[0]][i]}} & {{24{rt[qe[0]][i*8+7]}}, rt[qe[0]][i*8 +: 8]};

    // ---------------- G ----------------
    logic        gl_cmd;
    logic [6:0]  gj;
    logic [1:0]  gs;
    logic [20:0] fw;
    wire  [7:0]  g_c  = rs1_data[7:0];
    wire  [4:0]  f_n  = (fw[20:4] != 17'd0) ? 5'd16 : fw[4:0];
    wire  [4:0]  f_av = (rs0_avail[7:4] != 4'd0) ? 5'd16 : {1'b0, rs0_avail[3:0]};
    wire  [4:0]  f_m  = (f_n < f_av) ? f_n : f_av;
    wire         f_go = (st == K_GATH) && (f_m != 5'd0) && wd_ready;
    wire         g_go = (st == K_GATH) && gl_cmd && (gj != lrec) && (rs1_avail != 8'd0) && rc_ready[0] && wc_ready;

    // ---------------- ports: one adder each ----------------
    wire  i0_go = (st == K_SCAN) && (qi0 != nq);
    wire  i1_go = (st == K_SCAN) && (qi1 != nq);
    wire  l_go  = (st == K_LOG) && lg_cnt_v && (lg_e != lg_cnt) && (rs1_avail >= 8'd2) && wc_ready && wd_ready;
    wire  e_go  = (st == K_SCAN) && e_full && wc_ready && wd_ready && (!es || (e_has && e_rec));
    wire  n_go  = (st == K_FIN) && wc_ready && wd_ready;
    wire  gath  = (st == K_GATH);
    // stream 0 address: b / S chunk (scan) or the gathered row / word
    wire  [19:0] r0b = gath ? ((gs == 2'd0) ? p_wgt : (gs == 2'd1) ? p_bias : (gs == 2'd2) ? p_mult : p_shift) :
                       (qi0h ? p_shift : p_bias);
    wire  [19:0] r0o = gath ? ((gs == 2'd0) ? (k8 ? {4'd0, g_c, 8'd0} : {5'd0, g_c, 7'd0}) : {10'd0, g_c, 2'b00}) :
                       {9'd0, qi0, 4'd0};
    // stream 1 address: log, M block, list
    wire  [19:0] r1b = gath ? p_lst : (st == K_LOG) ? p_lg : p_mult;
    wire  [19:0] r1o = (st == K_SCAN) ? {9'd0, qi1, 4'd0} : 20'd0;
    // write address
    logic [19:0] wb, wo;
    always_comb begin
        wb = p_lg; wo = 20'd0;
        case (st)
            K_LOG:  begin wb = p_oh; wo = {4'd0, lg_w}; end
            K_SCAN: if (!es) begin wb = p_st; wo = {9'd0, qe, 4'd0}; end
                    else case (ek)
                        2'd0: begin wb = p_lst; wo = {13'd0, lrec}; end
                        2'd1: begin wb = p_oh; wo = {4'd0, e_off}; end
                        default: begin wb = p_lg; wo = {12'd0, lrec, 1'b0} + 20'd2; end
                    endcase
            K_GATH: begin
                wb = (gs == 2'd0) ? p_cw : (gs == 2'd1) ? p_cb : (gs == 2'd2) ? p_cm : p_cs;
                wo = (gs == 2'd0) ? (k8 ? {5'd0, gj, 8'd0} : {6'd0, gj, 7'd0}) : {11'd0, gj, 2'b00};
            end
            default: ;
        endcase
    end
    always_comb begin
        rc_valid = 2'b00;
        rc_addr = {12'd0, r1b + r1o, 12'd0, r0b + r0o};
        rc_len = {20'd16, 20'd16};
        rs0_pop = 6'd0; rs1_pop = 5'd0;
        wc_valid = 1'b0; wc_addr = {12'd0, wb + wo}; wc_len = 20'd1;
        wd_valid = 1'b0; wd_data = '0; wd_n = 5'd1;
        case (st)
            K_LOG: begin
                rc_valid[1] = !lg_cmd; rc_len[39:20] = 20'd144;
                if (!lg_cnt_v) rs1_pop = (rs1_avail >= 8'd2) ? 5'd2 : 5'd0;
                else if (lg_e != lg_cnt) begin
                    wc_valid = l_go; wd_valid = l_go;
                    rs1_pop = l_go ? 5'd2 : 5'd0;
                end else
                    rs1_pop = (lg_drop == 8'd0) ? 5'd0 : (lg_drop < 8'd16) ? ((rs1_avail >= lg_drop) ? lg_drop[4:0] : 5'd0) :
                              ((rs1_avail >= 8'd16) ? 5'd16 : 5'd0);
            end
            K_SCAN: begin
                rc_valid[0] = i0_go;
                rc_valid[1] = i1_go; rc_len[39:20] = 20'd64;
                if (c_go && ci == 2'd3) begin rs0_pop = 6'd32; rs1_pop = 5'd16; end
                wc_valid = e_go; wd_valid = e_go;
                if (!es) begin wc_len = 20'd16; wd_data = e_beat; wd_n = 5'd16; end
                else case (ek)
                    2'd0: wd_data = {120'd0, e_c};
                    2'd1: wd_data = {120'd0, 8'd1};
                    default: begin wc_len = 20'd2; wd_data = {112'd0, e_off}; wd_n = 5'd2; end
                endcase
            end
            K_FIN: begin
                wc_valid = n_go; wc_len = 20'd2;
                wd_valid = n_go; wd_data = {120'd0, 1'b0, lrec}; wd_n = 5'd2;
            end
            K_GATH: begin
                rc_valid[1] = !gl_cmd && (lrec != 7'd0); rc_len[39:20] = {13'd0, lrec};
                rc_valid[0] = g_go; rc_len[19:0] = (gs == 2'd0) ? {7'd0, p_k} : 20'd4;
                wc_valid = g_go; wc_len = (gs == 2'd0) ? {7'd0, p_k} : 20'd4;
                if (g_go && gs == 2'd3) rs1_pop = 5'd1;
                wd_valid = f_go; wd_data = rs0_data; wd_n = f_m;
                rs0_pop = f_go ? {1'b0, f_m} : 6'd0;
            end
            default: ;
        endcase
    end

    // ---------------- state ----------------
    wire [15:0] ohq_n = ohq + {7'd0, p_kcap, 2'b00};
    // st / ovf (reset); every other register is set at the op start (no reset)
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin st <= K_IDLE; ovf <= 1'b0; end
        else begin
            ovf <= (st == K_SCAN) && e_full && es && e_has && !e_rec;
            case (st)
                K_IDLE: if (start) st <= K_LOG;
                K_LOG:  if (lg_cnt_v && lg_e == lg_cnt && lg_drop == 8'd0) st <= K_SCAN;
                K_SCAN: if (qe == nq) st <= K_FIN;
                K_FIN:  if (n_go) st <= K_WAIT;
                K_WAIT: if (wr_idle) st <= K_GATH;
                K_GATH: if (gj == lrec && fw == 21'd0) st <= K_IDLE;
                default: st <= K_IDLE;
            endcase
        end
    end
    always_ff @(posedge clk) begin
        begin
            case (st)
                K_IDLE: if (start) begin
                    lg_cmd <= 1'b0; lg_cnt_v <= 1'b0; lg_e <= '0; lg_drop <= 8'd142;
                    qi0 <= '0; qi1 <= '0; qi0h <= 1'b0; qc <= '0; ci <= '0; qe <= '0; ei <= '0; es <= 1'b0; ek <= '0;
                    lrec <= '0; ohq <= '0; rn[0] <= '0; rn[1] <= '0;
                    gl_cmd <= 1'b0; gj <= '0; gs <= '0;
                end
                K_LOG: begin
                    if (rc_valid[1] && rc_ready[1]) lg_cmd <= 1'b1;
                    if (!lg_cnt_v) begin
                        if (rs1_avail >= 8'd2) begin
                            lg_cnt_v <= 1'b1;
                            lg_cnt <= (lg_w[15:7] != 9'd0) ? 7'd64 : (lg_w[6:0] > 7'd64) ? 7'd64 : lg_w[6:0];
                        end
                    end else if (lg_e != lg_cnt) begin
                        if (l_go) begin lg_e <= lg_e + 7'd1; lg_drop <= lg_drop - 8'd2; end
                    end else if (lg_drop != 8'd0) begin
                        lg_drop <= lg_drop - {3'd0, rs1_pop};
                    end
                end
                K_SCAN: begin
                    if (i0_go && rc_ready[0]) begin qi0h <= !qi0h; if (qi0h) qi0 <= qi0 + 7'd1; end
                    if (i1_go && rc_ready[1]) qi1 <= qi1 + 7'd4;
                    if (c_go) begin ci <= ci + 2'd1; if (ci == 2'd3) qc <= qc + 7'd1; end
                    if (rq_ov) rn[r_p] <= rn[r_p] + 3'd1;
                    if (e_full) begin
                        if (!es) begin
                            if (e_go) begin es <= 1'b1; ei <= 2'd0; ek <= 2'd0; end
                        end else if (e_has && !e_rec) begin
                            es <= 1'b0; rn[qe[0]] <= '0; qe <= qe + 7'd1; ohq <= ohq_n;
                        end else if (!e_has) begin
                            es <= 1'b0; rn[qe[0]] <= '0; qe <= qe + 7'd1; ohq <= ohq_n;
                        end else if (e_go) begin
                            if (ek == 2'd2) begin
                                ek <= 2'd0; lrec <= lrec + 7'd1;
                                if (e_nx == 2'd3) begin es <= 1'b0; rn[qe[0]] <= '0; qe <= qe + 7'd1; ohq <= ohq_n; end
                                else ei <= e_nx + 2'd1;
                            end else ek <= ek + 2'd1;
                        end
                    end
                end
                K_WAIT: if (wr_idle) begin
                    gl_cmd <= 1'b0;
                    fw <= (k8 ? {6'd0, lrec, 8'd0} : {7'd0, lrec, 7'd0}) + {11'd0, lrec, 3'b000} + {12'd0, lrec, 2'b00};
                end
                K_GATH: begin
                    if (rc_valid[1] && rc_ready[1]) gl_cmd <= 1'b1;
                    if (g_go) begin gs <= gs + 2'd1; if (gs == 2'd3) gj <= gj + 7'd1; end
                    if (f_go) fw <= fw - {16'd0, f_m};
                end
                default: ;
            endcase
        end
    end
endmodule

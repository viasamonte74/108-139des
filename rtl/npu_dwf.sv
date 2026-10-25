// npu_dwf (v37_1a) - fused depthwise unit: a 3x3 / stride-1 / pad-1 depthwise conv over the 16 channels a
// paired-group pass of npu_conv8 produces, computed while that pass runs (DWF mode, see npu_conv8).
//
// The pass's requantised outputs (16 channels per pixel, raster order, W = 5) shift into a 13-pixel chain
// (npu_conv8's window registers, offset-added bytes a + 128); shift k >= W + 2 starts output k - (W + 2), whose
// nine taps then sit at fixed chain positions (tap). An output takes four cycles: one channel quarter (4
// channels) per cycle on two double-pumped 9-tap dot products (slot A: high phase, slot B: low phase), so the
// window must stay put for four cycles: shifts are >= 4 cycles apart (one per pointwise pixel; the tail's
// dummy shifts are spaced by this unit). Taps outside the input are masked to 0 (= the zero point after the
// offset: exact). The results (bias added, saturated to 22 bits: exact for the channels npu_drq takes, which
// the driver checks) drain as fast octets in the odd drain slots (par): every shift happens in an even slot, so
// quarters 0 / 2 are finalised in an even slot (held 1 cycle) and quarters 1 / 3 in the next odd one, which drains
// the octet (held quarter + the quarter being finalised).
module npu_dwf (
    input  logic             clk,
    input  logic             rst_n,
    input  logic             prst,          // pass start: the walk restarts
    input  logic             sh_pw,         // a pointwise pixel enters the chain at the end of this cycle
    input  logic [7:0]       p_ho, p_wo,
    input  logic [11:0]      p_npix,
    input  logic [9*128-1:0] tap,           // chain taps, t = 3 dy + dx
    input  logic [8*128-1:0] wb,            // the lanes' entry 8: weight bytes 0..127 of the group (9 c + t)
    input  logic [127:0]     w8x,           // weight bytes 128..143
    input  logic [16*24-1:0] dbias,
    input  logic             par,           // a depthwise drain slot (odd); shifts come in even slots only
    input  logic             room,          // write FIFO room for one more output (tail outputs)
    output logic             sh_dm,         // a dummy shift (tail) at the end of this cycle
    output logic             ostart,        // an output starts with this cycle's shift
    output logic             dr_v,          // an octet drains this cycle
    output logic             dr_k,          // its channels: 8 dr_k + 0..7
    output logic [8*22-1:0]  dr_acc,
    output logic             busy
);
    localparam logic [19:0] DWK = 20'd656896;   // see npu_dwdot

    // ---------------------------------------------------------------- walk control
    logic [11:0] n_sh;                       // shifts in this pass
    logic [7:0]  xo, yo;                     // next output's position
    logic [3:0]  qv, q1;                     // quarter read by the dots / finalised (one-hot)
    logic        dmr;
    wire         sh   = sh_pw || dmr;
    wire  [11:0] w1   = {4'd0, p_wo} + 12'd1;
    wire  [11:0] tot  = p_npix + w1;         // shifts per pass (W + 1 dummies)
    assign ostart = sh && (n_sh >= w1);
    assign sh_dm  = dmr;
    assign dr_v   = q1[1] || q1[3];          // (odd slots by construction)
    assign dr_k   = q1[3];
    wire  [3:0]  qvn  = ostart ? 4'b0001 : {qv[2:0], 1'b0};
    wire         xl   = (xo == p_wo - 8'd1);
    logic [8:0]  tmn;
    always_comb for (int t = 0; t < 9; t++)
        tmn[t] = !((t / 3 == 0) && (yo == 8'd0)) && !((t / 3 == 2) && (yo == p_ho - 8'd1)) &&
                 !((t % 3 == 0) && (xo == 8'd0)) && !((t % 3 == 2) && xl);
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            n_sh <= '0; xo <= '0; yo <= '0; qv <= '0; q1 <= '0; dmr <= 1'b0;
            busy <= 1'b0;
        end else begin
            // (spacing >= 4: the shift lands after quarter 3's read; in an even slot: registered in an odd one)
            dmr  <= busy && !dmr && par && (n_sh >= p_npix) && (n_sh < tot) && ((qv == 4'd0) || qv[2]) && room;
            qv   <= qvn;
            q1   <= qv;
            if (prst) begin
                n_sh <= '0; xo <= '0; yo <= '0; busy <= 1'b1;
            end else begin
                if (sh) n_sh <= n_sh + 12'd1;
                if (ostart) begin
                    xo <= xl ? 8'd0 : xo + 8'd1;
                    if (xl) yo <= yo + 8'd1;
                end
                if (busy && (n_sh == tot) && (qv == 4'd0) && (q1 == 4'd0) && !dmr) busy <= 1'b0;
            end
        end
    end

    // ---------------------------------------------------------------- operands
    // slot s = 2 d + p (dot d, phase p: 0 = A / high, 1 = B / low) takes channel 4 q + s
    logic [71:0] ac [4], wc [4];
    // (the quarter select has kept copies per slot and tap, one for the activations and one for the weights: each bit
    //  drives 8 gates, no buffer chain on the half-cycle path)
    for (genvar s = 0; s < 4; s++) begin : g_op
        for (genvar t = 0; t < 9; t++) begin : g_t
            (* keep *) logic [3:0] qa, qw;
            // (qa carries the tap's padding mask: a tap outside the input selects nothing, i.e. activation 0)
            (* keep *) always_ff @(posedge clk or negedge rst_n)        // (kept: the copies must not be merged)
                if (!rst_n) begin qa <= '0; qw <= '0; end
                else begin qa <= ostart ? {3'b000, tmn[t]} : {qa[2:0], 1'b0}; qw <= qvn; end
            logic [7:0] av, wv;
            always_comb begin
                av = '0; wv = '0;
                for (int q = 0; q < 4; q++) begin
                    av = av | (tap[128*t + 8*(4*q+s) +: 8] & {8{qa[q]}});
                    wv = wv | (((9*(4*q+s)+t) < 128 ? wb[8*((9*(4*q+s)+t) % 128) +: 8] : w8x[8*((9*(4*q+s)+t) % 16) +: 8]) & {8{qw[q]}});
                end
            end
            assign ac[s][8*t +: 8] = av;
            assign wc[s][8*t +: 8] = wv;
        end
    end

    // ---------------------------------------------------------------- dots (double-pumped)
    logic [19:0] sa [2];
    logic [39:0] rb [2];
    for (genvar d = 0; d < 2; d++) begin : g_dot
        // phase copies (1 in the high phase), one per operand and tap: each drives 8 selects
        logic [71:0] ao, wo;
        for (genvar g = 0; g < 18; g++) begin : g_ph
            (* keep *) logic tp, tn;
            always_ff @(posedge clk) tp <= !tp;
            always_ff @(negedge clk) tn <= tp;
            wire ph = tp ^ tn;
            if (g < 9) begin : g_a
                assign ao[8*g +: 8] = ph ? ac[2*d][8*g +: 8] : ac[2*d+1][8*g +: 8];
            end else begin : g_w
                assign wo[8*(g-9) +: 8] = ph ? wc[2*d][8*(g-9) +: 8] : wc[2*d+1][8*(g-9) +: 8];
            end
        end
        logic [39:0] r, ra;
        npu_dwdot u_dot (.a(ao), .w(wo), .r(r));
        always_ff @(negedge clk) ra <= r;                               // slot A rows
        // (v44A) the dwdot rows are exact unsigned values (sum = dot + 391680 < 2^20): slot A's sum kept unsigned (npu_dwas)
        wire [19:0] sau;
        npu_dwas u_sas (.a(ra[18:0]), .b(ra[37:20]), .y(sau));
        always_ff @(posedge clk) begin
            sa[d] <= sau;                                                // slot A sum (low phase; = dot + 391680)
            rb[d] <= r;                                                  // slot B rows
        end
    end

    // ---------------------------------------------------------------- finalise: + bias, saturate
    function automatic logic [21:0] sat22(input logic signed [24:0] x);
        sat22 = (x[24:21] == {4{x[24]}}) ? x[21:0] : {x[24], {21{~x[24]}}};
    endfunction
    logic signed [23:0] bs [4];
    always_comb for (int s = 0; s < 4; s++) begin
        bs[s] = '0;
        for (int q = 0; q < 4; q++) bs[s] = bs[s] | ($signed(dbias[24*(4*q+s) +: 24]) & {24{q1[q]}});
    end
    logic [21:0] fr [4];
    // (v44A) dot + bias on fa_1 cells: slot A from its unsigned sum, slot B from its two rows (npu_dwfz / npu_dwfb)
    logic [24:0] fz [4];
    for (genvar d = 0; d < 2; d++) begin : g_fz
        npu_dwfz u_za (.u(sa[d]), .b(bs[2*d]), .y(fz[2*d]));
        npu_dwfb u_zb (.r0(rb[d][18:0]), .r1(rb[d][37:20]), .b(bs[2*d+1]), .y(fz[2*d+1]));
    end
    always_comb for (int u = 0; u < 4; u++) fr[u] = sat22($signed(fz[u]));
    // quarters 0 / 2 wait one cycle (channels 8 k + 0..3 of the octet)
    logic [21:0] hq [4];
    wire hg = clk | ~(q1[0] || q1[2]);
    always_ff @(posedge hg) for (int s = 0; s < 4; s++) hq[s] <= fr[s];
    always_comb for (int u = 0; u < 4; u++) begin
        dr_acc[22*u +: 22]     = hq[u];
        dr_acc[22*(4+u) +: 22] = fr[u];
    end
endmodule

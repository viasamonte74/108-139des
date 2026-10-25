// npu_pad - PAD / RESHAPE / COPY. The interior rows of the input are contiguous, so the whole input
// is one read; border bytes are generated here, and the padded tensor leaves as one write stream at
// up to 16 bytes per cycle. Output rows: {pad wbC, copy wC, pad waC} or a full pad row (woC).
module npu_pad (
    input  logic         clk,
    input  logic         rst_n,
    input  logic         start,
    input  logic [31:0]  p_in, p_out,
    input  logic [7:0]   p_rows, p_hb, p_h,         // (v15: rows <= 255, row segments < 65536 bytes)
    input  logic [15:0]  p_wbc, p_wc, p_wac, p_woc,
    input  logic [19:0]  p_total,
    input  logic [19:0]  p_inlen,      // p_h * p_wc: the input bytes (one read)
    input  logic [7:0]   p_val,
    input  logic         p_pi,         // (v49DS) interior only: p_h row commands (p_wc bytes at p_out + p_total, then
                                       // p_woc apart), the p_inlen (< 65536) input bytes, no pad bytes
    input  logic         p_il,         // (v50_MB) OP_PIL: plane 0 on stream 0, plane 1 (p_in + p_inlen) on stream 1,
                                       // 16 bytes of each in turn, one write of p_total bytes
    output logic         rc1_valid,
    input  logic         rc1_ready,
    output logic [31:0]  rc1_addr,
    input  logic [7:0]   rs1_avail,
    input  logic [127:0] rs1_data,
    output logic [4:0]   rs1_pop,
    output logic         rc_valid,
    input  logic         rc_ready,
    output logic         rc_fill,
    output logic [31:0]  rc_addr,
    output logic [19:0]  rc_len,
    input  logic [7:0]   rs_avail,
    input  logic [127:0] rs_data,
    output logic [4:0]   rs_pop,
    output logic         wc_valid,
    input  logic         wc_ready,
    output logic [31:0]  wc_addr,
    output logic [19:0]  wc_len,
    output logic         wd_valid,
    input  logic         wd_ready,
    output logic [127:0] wd_data,
    output logic [4:0]   wd_n,
    output logic         busy
);
    logic        act, r_pend, w_pend;
    logic        r1_pend, il_s;          // (v50_MB) OP_PIL: stream-1 read pending, stream of the next unit
    logic [7:0]  row;
    logic [1:0]  seg;                  // 0 pad-before, 1 data, 2 pad-after, 3 full pad row
    logic [15:0] rem;
    logic [19:0] wa;                   // (v49DS) interior: next row command's address (row counts the commands)
    wire in_body = (row >= p_hb) && (row < p_hb + p_h);

    assign rc_valid = r_pend;
    assign rc1_valid = r1_pend;
    assign rc1_addr  = p_in + {12'd0, p_inlen};
    assign rc_fill  = 1'b0;
    assign rc_addr  = p_in;
    assign rc_len   = p_inlen;
    assign wc_valid = w_pend;
    assign wc_addr  = p_pi ? {12'd0, wa} : p_out;
    assign wc_len   = p_pi ? {4'd0, p_wc} : p_total;

    wire [4:0] cap = (rem >= 16'd16) ? 5'd16 : rem[4:0];
    wire [4:0] dav0 = (rs_avail >= 8'd16) ? 5'd16 : rs_avail[4:0];
    wire [4:0] dav1 = (rs1_avail >= 8'd16) ? 5'd16 : 5'd0;      // (v50_MB) OP_PIL: whole 16-byte units
    wire [4:0] dav  = p_il ? (il_s ? dav1 : ((rs_avail >= 8'd16) ? 5'd16 : 5'd0)) : dav0;
    wire [4:0] n   = (seg == 2'd1) ? ((dav < cap) ? dav : cap) : cap;
    wire go = act && (rem != '0) && wd_ready && (n != 5'd0);
    assign rs_pop   = (go && seg == 2'd1 && !(p_il && il_s)) ? n : 5'd0;
    assign rs1_pop  = (go && p_il && il_s) ? n : 5'd0;
    assign wd_valid = go;
    assign wd_data  = (seg == 2'd1) ? ((p_il && il_s) ? rs1_data : rs_data) : {16{p_val}};
    assign wd_n     = n;

    // segment sequencing (zero-length segments are skipped)
    logic [1:0]  nseg;
    logic [15:0] nrem;
    logic [7:0]  nrow;
    logic        ndone;
    always_comb begin
        nseg = seg; nrem = '0; nrow = row; ndone = 1'b0;
        if (p_pi || p_il) ndone = 1'b1;    // (v49DS) interior: one data segment (v50_MB: OP_PIL too)
        else case (seg)
            2'd0: begin nseg = 2'd1; nrem = p_wc; end
            2'd1: begin nseg = 2'd2; nrem = p_wac; end
            default: begin
                if (row == p_rows - 8'd1) ndone = 1'b1;
                else begin
                    nrow = row + 8'd1;
                    if ((nrow >= p_hb) && (nrow < p_hb + p_h)) begin nseg = 2'd0; nrem = p_wbc; end
                    else begin nseg = 2'd3; nrem = p_woc; end
                end
            end
        endcase
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            act <= 1'b0; r_pend <= 1'b0; w_pend <= 1'b0; row <= '0; seg <= '0; rem <= '0; r1_pend <= 1'b0; il_s <= 1'b0;
        end else begin
            if (r_pend && rc_ready) r_pend <= 1'b0;
            if (r1_pend && rc1_ready) r1_pend <= 1'b0;
            if (go && p_il) il_s <= !il_s;
            if (w_pend && wc_ready) begin
                w_pend <= p_pi && (row != p_h - 8'd1);
                if (p_pi) begin wa <= wa + {4'd0, p_woc}; row <= row + 8'd1; end
            end
            if (start) begin
                act <= 1'b1; r_pend <= p_il || ((p_h != 8'd0) && (p_wc != '0)); w_pend <= !p_pi || ((p_h != 8'd0) && (p_wc != '0)); row <= '0;
                wa <= p_out[19:0] + p_total;
                r1_pend <= p_il; il_s <= 1'b0;
                if (p_il) begin seg <= 2'd1; rem <= p_total[15:0]; end
                else if (p_pi) begin seg <= 2'd1; rem <= p_inlen[15:0]; end
                else if (p_hb == 8'd0 && p_h != 8'd0) begin seg <= 2'd0; rem <= p_wbc; end
                else begin seg <= 2'd3; rem <= p_woc; end
            end else if (act) begin
                if (go) rem <= rem - 16'(n);
                if ((rem == '0) || (go && (rem == 16'(n)))) begin
                    if (ndone) act <= 1'b0;
                    else begin seg <= nseg; rem <= nrem; row <= nrow; end
                end
            end
        end
    end
    assign busy = act || r_pend || w_pend || r1_pend;
endmodule

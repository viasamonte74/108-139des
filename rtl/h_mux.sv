// h_fan: an explicit buf_1 fanout tree (every driver <= 8 loads) for a signal that drives hand-instantiated cells:
// ABC never sees those loads (they are outside its network), so it would not buffer them; this keeps the
// netlist inside the flow's max-fanout rule (MAX_FANOUT_CONSTRAINT 10) and prices the buffers honestly.
// y[i] = a for every i.
module h_fan #(parameter int N = 8) (input logic a, output logic [N-1:0] y);
    localparam int F  = 8;
    localparam int N1 = (N + F - 1) / F;          // first-level buffers (each <= 8 loads)
    localparam int N2 = (N1 + F - 1) / F;         // second level (each <= 8 first-level buffers)
    if (N <= F) begin : g_d
        assign y = {N{a}};
    end else begin : g_t
        logic [N1-1:0] b1;
        if (N1 <= F) begin : g_1
            for (genvar k = 0; k < N1; k++) begin : g_b
                sky130_fd_sc_hd__buf_1 u_b (.A(a), .X(b1[k]));
            end
        end else begin : g_2
            logic [N2-1:0] b2;
            for (genvar k = 0; k < N2; k++) begin : g_c
                sky130_fd_sc_hd__buf_1 u_c (.A(a), .X(b2[k]));
            end
            for (genvar k = 0; k < N1; k++) begin : g_b
                sky130_fd_sc_hd__buf_1 u_b (.A(b2[k / F]), .X(b1[k]));
            end
        end
        for (genvar i = 0; i < N; i++) begin : g_y
            assign y[i] = b1[i / F];
        end
    end
endmodule

// h_mx: W-bit 2:1 mux x = s ? a1 : a0 as hand-instantiated a22o_2 cells (10.01 um2/bit) with the inverted select
// computed once (~s); s and ~s reach the cells through explicit buf_1 trees (h_fan). ABC maps a bus mux to
// mux2_2 (11.26 um2/bit) and, when the mux drives a flop D pin, adds a buf_1 (3.75 um2) behind it.
// Exact: (a1 & s) | (a0 & ~s).
module h_mx #(parameter int W = 8) (input logic [W-1:0] a0, input logic [W-1:0] a1, input logic s,
                                    output logic [W-1:0] x);
    wire sn = ~s;
    logic [W-1:0] sb, snb;
    h_fan #(W) u_fs (.a(s), .y(sb));
    h_fan #(W) u_fn (.a(sn), .y(snb));
    for (genvar i = 0; i < W; i++) begin : g_b
        sky130_fd_sc_hd__a22o_2 u_m (.A1(a1[i]), .A2(sb[i]), .B1(a0[i]), .B2(snb[i]), .X(x[i]));
    end
endmodule

// (v56_AR) h_rca: W-bit ripple-carry adder of hand-instantiated sky130 fa_1 cells (20.02 um2 / bit; ABC maps a
// W-bit add to xor2_2 / xnor2_2 + carry gates at ~35-57 um2 / bit in this flow). s = a + b + ci (mod 2^W). Exact.
// ABC does not see through the cells: keep it off half-cycle / critical paths (about 0.4 ns per bit).
module h_rca #(parameter int W = 20) (input logic [W-1:0] a, input logic [W-1:0] b, input logic ci,
                                      output logic [W-1:0] s, output logic co);
    logic [W:0] c;
    assign c[0] = ci;
    for (genvar i = 0; i < W; i++) begin : g_b
        sky130_fd_sc_hd__fa_1 u_fa (.A(a[i]), .B(b[i]), .CIN(c[i]), .COUT(c[i+1]), .SUM(s[i]));
    end
    assign co = c[W];
endmodule

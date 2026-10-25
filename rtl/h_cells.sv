// Behavioural models of the sky130_fd_sc_hd standard cells this design instantiates by hand (simulation /
// equivalence); `(* blackbox *)` makes yosys keep each instance as the real library cell, priced from the
// liberty by the official `stat -liberty` and timed by OpenSTA with its real arcs. None of these cells is in
// the PDK's no_synth.cells or drc_exclude.cells. Functions copied from the liberty `function` attributes.
(* blackbox *)
module sky130_fd_sc_hd__o2bb2a_2 (input A1_N, input A2_N, input B1, input B2, output X);
    assign X = (!A1_N | !A2_N) & (B1 | B2);
endmodule
(* blackbox *)
module sky130_fd_sc_hd__a2bb2o_2 (input A1_N, input A2_N, input B1, input B2, output X);
    assign X = (!A1_N & !A2_N) | (B1 & B2);
endmodule
(* blackbox *)
module sky130_fd_sc_hd__mux2_2 (input A0, input A1, input S, output X);
    assign X = S ? A1 : A0;
endmodule
(* blackbox *)
module sky130_fd_sc_hd__mux2_1 (input A0, input A1, input S, output X);
    assign X = S ? A1 : A0;
endmodule
(* blackbox *)
module sky130_fd_sc_hd__a22o_2 (input A1, input A2, input B1, input B2, output X);
    assign X = (A1 & A2) | (B1 & B2);
endmodule
(* blackbox *)
module sky130_fd_sc_hd__mux4_2 (input A0, input A1, input A2, input A3, input S0, input S1, output X);
    assign X = S1 ? (S0 ? A3 : A2) : (S0 ? A1 : A0);
endmodule
(* blackbox *)
module sky130_fd_sc_hd__buf_1 (input A, output X);
    assign X = A;
endmodule

// behavioural model of the sky130 full-adder standard cell (simulation); synthesis keeps it as the cell
(* blackbox *)
module sky130_fd_sc_hd__fa_1 (input A, input B, input CIN, output COUT, output SUM);
    assign {COUT, SUM} = {1'b0, A} + {1'b0, B} + {1'b0, CIN};
endmodule

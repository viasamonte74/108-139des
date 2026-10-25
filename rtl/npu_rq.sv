// npu_rq - exact TFLM requantisation, 4-stage pipeline, one result per cycle.
//   y   = MultiplyByQuantizedMultiplier(acc, mult, shift)
//         SRDHM(x, M) == (x*M + 2^30) >>> 31 for every sign of x*M (the reference's negative
//         nudge 1-2^30 followed by "+2^31-1" lands on the same value); x = acc << max(shift,0);
//         then RoundingDivideByPOT(., max(-shift,0)).
//   out = (int8) clamp(y + ooff, amin, amax)
module npu_rq (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        in_valid,
    input  logic [31:0] acc,
    input  logic [31:0] mult,        // >= 0 (bit 31 is ignored): TFLM multipliers, ceil(2^31/n) for n >= 2
    input  logic [7:0]  shift,       // signed
    input  logic [8:0]  ooff,        // per-op constants: read directly, not pipelined
    input  logic [7:0]  amin,        //   (int8 offset and clamp; the driver checks the ranges)
    input  logic [7:0]  amax,
    output logic        out_valid,
    output logic [7:0]  out_q,
    output logic        raw_valid,   // MultiplyByQuantizedMultiplier(acc, mult, shift) itself,
    output logic [31:0] raw          //   one stage earlier (no offset / clamp)
);
    // (an average pool's rounded division sign(s) * floor((|s| + n/2) / n) is this requantisation
    //  with mult = floor(2^31 / n) + 1, shift 0: exact for |s| <= 128 n, n < 2896, see the kernel)
    // ---- stage 1 (combinational, shares the cycle with stage 2): left shift ----
    // (the 8-bit result is delayed one cycle at the end instead, so the latency stays 4 with 9
    //  output flops in place of the 69 stage-1 flops)
    logic [31:0] x1, m1;
    logic [4:0]  r1;
    wire  [4:0]  lsh = shift[7] ? 5'd0 : shift[4:0];
    wire  [7:0]  nsh = 8'd0 - shift;
    always_comb begin
        x1  = acc << lsh;
        m1  = {1'b0, mult[30:0]};
        r1  = shift[7] ? nsh[4:0] : 5'd0;
    end

    // ---- stage 2: 32x32 signed product, radix-4 Booth on m1 (rows added as unsigned
    //      {~s, low 32} = row + 2^32, with the constant correcting the sum), the SRDHM nudge 2^30
    //      folded into the constant; only bits [62:31] are kept ----
    logic        v2, sat2;
    logic [31:0] y2;
    logic [4:0]  r2;
    logic [63:0] pr;
    logic [2:0]  bt;
    logic        b1, b2, bn;
    logic [32:0] mg, pp;
    always_comb begin
        pr = {32'hAAAAAAAB, 2'b01, 30'd0};
        for (int k = 0; k < 16; k++) begin
            bt = {m1[2*k+1], m1[2*k], (k == 0) ? 1'b0 : m1[(k == 0) ? 0 : 2*k-1]};
            b1 = bt[1] ^ bt[0]; b2 = (bt == 3'b011) || (bt == 3'b100); bn = bt[2] && !(bt[1] && bt[0]);
            mg = ({33{b1}} & {x1[31], x1}) | ({33{b2}} & {x1, 1'b0});
            pp = mg ^ {33{bn}};
            pr = pr + (64'({~pp[32], pp[31:0]}) << (2*k)) + (64'(bn) << (2*k));
        end
    end
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) v2 <= 1'b0;
        else        v2 <= in_valid;
    end
    always_ff @(posedge clk) begin
        y2   <= pr[62:31];
        sat2 <= 1'b0;                                       // SRDHM saturation needs m1 = -2^31
        r2 <= r1;
    end

    // ---- stage 3: saturation + RoundingDivideByPOT ----
    logic        v3, hi3, lo3;
    logic [31:0] z3;
    logic [9:0]  q3;
    logic [31:0] zz;
    logic [31:0] y, mask, rem, thr, ysh, z;
    always_comb begin
        y    = sat2 ? 32'h7fff_ffff : y2;
        mask = (32'd1 << r2) - 32'd1;
        rem  = y & mask;
        thr  = (mask >> 1) + {31'd0, y[31]};
        ysh  = $signed(y) >>> r2;
        z    = ysh + ((rem > thr) ? 32'd1 : 32'd0);
    end
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) v3 <= 1'b0;
        else        v3 <= v2;
    end
    // the clamped output only depends on the result within [-512, 511] (beyond: amax / amin)
    assign zz = z;
    always_ff @(posedge clk) begin
        z3  <= zz;                                          // raw: used by units C / D only
        q3  <= zz[9:0];
        hi3 <= !zz[31] && (zz[30:9] != 22'd0);
        lo3 <=  zz[31] && (zz[30:9] != 22'h3fffff);
    end
    assign raw_valid = v3;                                  // (one cycle earlier than before the merge;
    assign raw       = z3;                                  //  units A / B leave raw unconnected)

    // ---- stage 4: offset + clamp ----
    logic [10:0] zs, s4, c4;
    always_comb begin
        zs = {q3[9], q3};
        s4 = zs + {{2{ooff[8]}}, ooff};
        c4 = s4;
        if ($signed(c4) < $signed({{3{amin[7]}}, amin})) c4 = {{3{amin[7]}}, amin};
        if ($signed(c4) > $signed({{3{amax[7]}}, amax})) c4 = {{3{amax[7]}}, amax};
        if (hi3) c4 = {{3{amax[7]}}, amax};
        if (lo3) c4 = {{3{amin[7]}}, amin};
    end
    logic       v4;
    logic [7:0] q4;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin v4 <= 1'b0; out_valid <= 1'b0; end
        else        begin v4 <= v3; out_valid <= v4; end
    end
    always_ff @(posedge clk) begin q4 <= c4[7:0]; out_q <= q4; end
endmodule

// npu_drq - dual requant unit (units C, D), 4-stage like npu_rq.
//   conv mode (asc = 0): out_q = clamp(MultiplyByQuantizedMultiplier(acc, mult, shift) + ooff), exact
//     for the channels npu_conv8 gives it: mult == 0, or 2^30 <= mult < 2^31 with -11 <= shift <= 0.
//     With r = -shift: z = RDP((a * mult + 2^30) >>> 31, r), a = acc saturated to 22 bits; the
//     saturation changes acc only when |acc| >= 2^21, and then |z| >= (2^21 - 1) * 2^30 / 2^(31 + r) - 1
//     >= 511 for both values (r <= 11): both saturate the int8 clamp (|ooff| <= 256) on the same side.
//   ADD-scale mode (asc = 1, fused ADD): raw = MultiplyByQuantizedMultiplier(v << 20, mult, shift) for
//     acc = v (9 bits, sign-extended), mult >= 0: SRDHM(v << 20, mult) = (v * mult + 2^10) >>> 11 exactly
//     (|v << 20| < 2^31), then RoundingDivideByPOT by r = -shift <= 31 (a positive input shift is part
//     of the driver's << 20). raw is valid 2 cycles after in_valid.
module npu_drq (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        in_valid,
    input  logic        asc,
    input  logic [31:0] acc,
    input  logic [30:0] mult,
    input  logic [7:0]  shift,
    input  logic [8:0]  ooff,
    input  logic [7:0]  amin,
    input  logic [7:0]  amax,
    output logic        out_valid,
    output logic [7:0]  out_q,
    output logic        raw_valid,
    output logic [31:0] raw
);
    localparam int W  = 22;
    localparam int PW = W + 32;
    // Booth rows are added as {~s, low W bits} = row + 2^W: K0 removes those offsets
    function automatic logic [PW-1:0] k0f();
        k0f = '0;
        for (int k = 0; k < 16; k++) k0f = k0f - (PW'(1) << (W + 2 * k));
    endfunction
    localparam logic [PW-1:0] K0 = k0f();

    // ---- stage 1: a * mult + nudge, radix-4 Booth digits of mult (16 rows of +-a, +-2a) ----
    wire          fits = (acc[31:W-1] == {(33-W){acc[31]}});
    wire [W-1:0]  a    = fits ? acc[W-1:0] : {acc[31], {(W-1){~acc[31]}}};
    wire [31:0]   m    = {1'b0, mult};
    logic [PW-1:0] pr;
    always_comb begin
        logic [2:0] bt;
        logic       b1, b2, bn;
        logic [W:0] mg, pp;
        pr = K0 + (asc ? (PW'(1) << 10) : (PW'(1) << 30));
        for (int k = 0; k < 16; k++) begin
            bt = {m[2*k+1], m[2*k], (k == 0) ? 1'b0 : m[(k == 0) ? 0 : 2*k-1]};
            b1 = bt[1] ^ bt[0]; b2 = (bt == 3'b011) || (bt == 3'b100); bn = bt[2] && !(bt[1] && bt[0]);
            mg = ({(W+1){b1}} & {a[W-1], a}) | ({(W+1){b2}} & {a, 1'b0});
            pp = mg ^ {(W+1){bn}};
            pr = pr + (PW'({~pp[W], pp[W-1:0]}) << (2*k)) + (PW'(bn) << (2*k));
        end
    end
    logic        v1, v2, v3;
    logic [29:0] y1;
    logic [4:0]  r1;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin v1 <= 1'b0; v2 <= 1'b0; v3 <= 1'b0; out_valid <= 1'b0; end
        else begin v1 <= in_valid; v2 <= v1; v3 <= v2; out_valid <= v3; end
    end
    always_ff @(posedge clk) begin
        y1 <= asc ? pr[40:11] : {{(30-W){pr[W+30]}}, pr[W+30:31]};
        r1 <= shift[7] ? 5'(8'd0 - shift) : 5'd0;
    end

    // ---- stage 2: RoundingDivideByPOT ----
    logic [29:0] mask, rem, thr, ysh, z, z2;
    always_comb begin
        mask = (30'd1 << r1) - 30'd1;
        rem  = y1 & mask;
        thr  = (mask >> 1) + {29'd0, y1[29]};
        ysh  = $signed(y1) >>> r1;
        z    = ysh + ((rem > thr) ? 30'd1 : 30'd0);
    end
    always_ff @(posedge clk) z2 <= z;
    assign raw_valid = v2;
    assign raw       = {{2{z2[29]}}, z2};

    // ---- stage 3: saturate + offset; stage 4: clamp ----
    logic [10:0] zs, s3, c4;
    always_comb zs = ($signed(z2) > 30'sd511) ? 11'sd511 : ($signed(z2) < -30'sd512) ? -11'sd512 : z2[10:0];
    always_ff @(posedge clk) s3 <= zs + {{2{ooff[8]}}, ooff};
    always_comb begin
        c4 = s3;
        if ($signed(c4) < $signed({{3{amin[7]}}, amin})) c4 = {{3{amin[7]}}, amin};
        if ($signed(c4) > $signed({{3{amax[7]}}, amax})) c4 = {{3{amax[7]}}, amax};
    end
    always_ff @(posedge clk) out_q <= c4[7:0];
endmodule

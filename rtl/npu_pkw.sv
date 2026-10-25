// (v55AD) npu_pkw - run-time FC weight packer ("pack tee").
// While a raw FC stream op with RG_CP bit 9 runs, it watches the conv engine's stream-0 pops (the bytes the op
// consumes, in order: the K-byte activation vector, then per 16-channel group the bias [if any] / multiplier / shift
// words and the group's weight rows) and writes the stream a packed FC op (RG_CP bit 2, npu_conv8's v22_P decoder)
// reads with RG_CP bit 5 (per-tensor words): the activation vector is dropped, group 0's bias / multiplier / shift
// words and later groups' bias words are copied as they are, later groups' multiplier / shift words are dropped and
// compared with channel 0's (uni_q: every channel's multiplier and shift equal channel 0's; else the stream must not
// be used), every row's 16-byte beats become
// 16-weight blocks of width w (3..8 bits, 2w bytes, two's complement LSB first) behind a header byte
// {pair, w1-1, w0-1}; an even beat and the odd beat after it form a pair when w0 + w1 <= 15 (the driver packs only
// K % 32 == 0). Exact for any weights. The packed bytes go through a byte FIFO (the read DMA's stream 1, idle during
// an FC stream op: f_* ports; pushes <= 16 bytes at lane 0, pops of 16 keep its head 16-byte aligned, so its R2
// output is in lane order) and leave as background write bursts of the write DMA (bgd: up to NB beats inside one
// 128-byte block, only bytes already in the FIFO).
// Destination RG_WGSTR (0: continue at the previous pack op's end rounded up to 16 bytes), limit RG_DWADJ
// (exclusive): a stream that does not fit stops writing and is flagged (ovf); continuing ops after an overflow are
// dropped too (sticky until an op with an explicit destination). After the op: len_q = its stream length in bytes,
// ovf_q its flag (held until the next pack op ends). hold (registered) stops the conv engine's stream-0 pops while
// a piece would be overwritten or the FIFO is short of room. act covers the op until its last burst is handed over.
module npu_pkw #(
    parameter int CAPB = 128,
    parameter int NB   = 4
) (
    input  logic         clk,
    input  logic         rst_n,
    input  logic         start,          // a pack-tee op starts (one cycle; the active bank is valid)
    input  logic [12:0]  k,              // K
    input  logic [9:0]   co,             // output channels
    input  logic         hasb,           // bias words in the stream
    input  logic [19:0]  dest,           // start address (0: continue)
    input  logic [19:0]  lim,            // first address past the region
    input  logic [7:0]   pop_n,          // stream-0 bytes the conv engine pops this cycle
    input  logic [127:0] pop_d,          // stream-0 head (16 bytes)
    output logic         hold,
    output logic         bgd_valid,
    output logic [19:0]  bgd_addr,
    output logic [7:0]   bgd_len,
    input  logic         bgd_take,
    output logic         bgr_v,
    output logic [127:0] bgr_data,
    input  logic         bgr_rdy,
    output logic         act,
    // the byte FIFO (CAPB bytes; in_ready = room for 16 bytes; out_avail excludes a staged push of <= 16 bytes)
    output logic         f_push,
    output logic [127:0] f_pd,
    output logic [4:0]   f_pn,
    input  logic         f_ird,
    input  logic [7:0]   f_avail_i,
    input  logic [127:0] f_data,
    output logic [4:0]   f_pop,
    output logic [19:0]  len_q,
    output logic         ovf_q,
    output logic         uni_q           // the last pack op's multipliers / shifts are uniform (per-tensor stream valid)
);
    localparam int AW = $clog2(CAPB);
    // ---------------------------------------------------------------- stream position
    logic [1:0]  ph;                     // 0 activation vector, 1 words, 2 rows, 3 consumed
    logic [12:0] rem;                    // bytes (ph 0 / 1) or beats (ph 2) left in the phase
    logic [9:0]  cleft;                  // channels after the current group
    logic [4:0]  gc;                     // channels of the current group
    logic        par;                    // ph 2: the next beat is odd
    logic [8:0]  kb;                     // beats per row (K / 16)
    wire  [4:0]  gc_n     = (cleft >= 10'd16) ? 5'd16 : cleft[4:0];
    wire  [12:0] wbytes_n = hasb ? ({5'd0, gc_n, 3'd0} + {6'd0, gc_n, 2'd0}) : {5'd0, gc_n, 3'd0};   // 12 / 8 gc
    wire  [12:0] bbytes_n = hasb ? {6'd0, gc_n, 2'd0} : 13'd0;   // the bias words among them
    wire  [12:0] gbytes   = {6'd0, gc, 2'd0};                    // one kind's words of the current group
    logic        g0;                    // the current group is group 0
    logic [12:0] wdone;                 // word bytes of the current group already popped
    // a word pop's kind: bias (wdone < bias bytes), multipliers, shifts
    wire         w_b   = hasb && (wdone < gbytes);
    wire         w_m   = !w_b && (wdone < (hasb ? {gbytes[11:0], 1'b0} : gbytes));
    logic [31:0] m0, s0;                // channel 0's multiplier / shift (group 0's first words)
    logic        uni;
    wire         w_cmp = !w_b && (w_m ? (wdone == (hasb ? gbytes : 13'd0)) : (wdone == (hasb ? {gbytes[11:0], 1'b0} : gbytes)));
    wire         pop   = act && (pop_n != 8'd0) && (ph != 2'd3);
    wire         p_wa  = pop && (ph == 2'd1);               // a word pop
    wire         p_w   = p_wa && (g0 || w_b);               // ... that is copied (group 0, or a bias word)
    wire         p_e   = pop && (ph == 2'd2) && !par;       // an even beat
    wire         p_o   = pop && (ph == 2'd2) && par;        // an odd beat
    wire         lastb = (rem == 13'd1);
    wire         lastp = (rem <= {5'd0, pop_n});
    // phase / parity after this cycle (for hold)
    logic [1:0]  ph_x;
    logic        par_x;
    always_comb begin
        ph_x = ph; par_x = par;
        if (pop) case (ph)
            2'd0: if (lastp) ph_x = 2'd1;
            2'd1: if (lastp) begin ph_x = 2'd2; par_x = 1'b0; end
            2'd2: begin par_x = !par; if (lastb) ph_x = (cleft == 10'd0) ? 2'd3 : 2'd1; end
            default: ;
        endcase
    end

    // ---------------------------------------------------------------- block width and packing of the popped beat
    logic [6:0] mag;
    always_comb begin
        mag = '0;
        for (int i = 0; i < 16; i++) mag = mag | (pop_d[i*8+7] ? ~pop_d[i*8 +: 7] : pop_d[i*8 +: 7]);
    end
    wire [2:0] wc = mag[6] ? 3'd7 : mag[5] ? 3'd6 : mag[4] ? 3'd5 : mag[3] ? 3'd4 : mag[2] ? 3'd3 : 3'd2;   // w - 1
    logic [127:0] pk;
    always_comb begin
        pk = '0;
        case (wc)
            3'd2:    begin for (int p = 0; p < 48;  p++) pk[p] = pop_d[(p / 3) * 8 + (p % 3)]; end
            3'd3:    begin for (int p = 0; p < 64;  p++) pk[p] = pop_d[(p / 4) * 8 + (p % 4)]; end
            3'd4:    begin for (int p = 0; p < 80;  p++) pk[p] = pop_d[(p / 5) * 8 + (p % 5)]; end
            3'd5:    begin for (int p = 0; p < 96;  p++) pk[p] = pop_d[(p / 6) * 8 + (p % 6)]; end
            3'd6:    begin for (int p = 0; p < 112; p++) pk[p] = pop_d[(p / 7) * 8 + (p % 7)]; end
            default: pk = pop_d;
        endcase
    end

    // ---------------------------------------------------------------- pieces (one pushed per cycle)
    // P0 / c0: the even beat's block; P1 / c1: the odd beat's block, or a word pop's bytes (wn of them)
    logic [127:0] P0, P1;
    logic [2:0]   c0, c1;
    logic [4:0]   wn;
    localparam logic [2:0] Q_IDLE = 3'd0, Q_A = 3'd1, Q_P0 = 3'd2, Q_B = 3'd3, Q_P1 = 3'd4, Q_W = 3'd5;
    logic [2:0]   pq, pq_adv, pq_n;
    logic      pair;                                             // w0 + w1 <= 15 (latched with the odd beat: the
                                                                 // next even beat may replace c0 before Q_B)
    wire [7:0] h_a   = pair ? {2'b01, c1, c0} : {5'd0, c0};
    wire [7:0] h_1   = {5'd0, c1};
    logic         pu_v;
    logic [127:0] pu_d;
    logic [4:0]   pu_n;
    always_comb begin
        pu_v = 1'b1; pu_d = P1; pu_n = 5'd16; pq_adv = Q_IDLE;
        case (pq)
            Q_A:  if (c0 == 3'd7) begin pu_d = {120'd0, h_a}; pu_n = 5'd1; pq_adv = Q_P0; end
                  else begin pu_d = {P0[119:0], h_a}; pu_n = {1'b0, c0, 1'b0} + 5'd3; pq_adv = Q_B; end   // 1 + 2 w0
            Q_P0: begin pu_d = P0; pq_adv = Q_B; end
            Q_B:  if (pair) pu_n = {1'b0, c1, 1'b0} + 5'd2;                                         // 2 w1
                  else if (c1 == 3'd7) begin pu_d = {120'd0, h_1}; pu_n = 5'd1; pq_adv = Q_P1; end
                  else begin pu_d = {P1[119:0], h_1}; pu_n = {1'b0, c1, 1'b0} + 5'd3; end
            Q_P1: ;
            Q_W:  pu_n = wn;
            default: pu_v = 1'b0;
        endcase
    end
    wire  [AW:0]  f_avail = f_avail_i[AW:0];
    logic         drop;                  // overflow: bytes are dropped, not written
    logic         stg;                   // a push was made last cycle (staged in the FIFO, not yet readable)
    wire          push  = pu_v && f_ird && !drop;
    wire          pdone = pu_v && (f_ird || drop);   // the piece leaves (written to the FIFO or dropped)
    assign f_push = push; assign f_pd = pu_d; assign f_pn = pu_n;
    always_comb begin
        pq_n = pdone ? pq_adv : pq;
        if (p_o) pq_n = Q_A;
        else if (p_w) pq_n = Q_W;
    end

    // ---------------------------------------------------------------- write-out
    logic [19:0] wp;                     // next write address
    logic [19:0] lim_q;
    logic [3:0]  bl;                     // requested beats not yet handed over
    logic [AW:0] rqb;                    // requested bytes not yet handed over
    logic [19:0] plen;                   // bytes pushed (the stream length)
    logic        sticky;                 // continuing ops after an overflow are dropped
    wire  [AW:0] unreq = f_avail - rqb;                    // FIFO bytes not yet requested
    wire  [3:0]  blk_b = 4'd8 - {1'b0, wp[6:4]};          // beats to the 128-byte block's end
    wire  [3:0]  nb    = (blk_b < 4'(NB)) ? blk_b : 4'(NB);
    wire         fin   = (ph == 2'd3) && (pq == Q_IDLE) && !stg;   // every byte of the op is readable
                                                           // in the FIFO (no staged push) or dropped
    wire         r_full = (unreq >= (AW+1)'({nb, 4'd0}));
    wire         r_last = fin && (unreq != '0) && !r_full;  // the final (partial) burst
    wire  [7:0]  r_len = r_full ? {nb, 4'd0} : 8'(unreq);
    wire  [20:0] r_end = {1'b0, wp} + {13'd0, r_len};
    wire         r_fit = (r_end <= {1'b0, lim_q});
    wire         r_any = act && !drop && (r_full || r_last) && (bl <= 4'(8 - NB));
    assign bgd_valid = r_any && r_fit;
    assign bgd_addr  = wp;
    assign bgd_len   = r_len;
    assign bgr_v     = (bl != 4'd0);
    assign bgr_data  = f_data;
    wire   bt        = bgr_v && bgr_rdy;                   // a beat is written
    wire   [4:0] f_one = (f_avail > (AW+1)'(16)) ? 5'd16 : 5'(f_avail);
    assign f_pop     = (act && (bt || (drop && bl == 4'd0))) ? f_one : 5'd0;   // (stream 1 is other ops' otherwise)
    wire   ovf_now   = r_any && !r_fit;
    wire   done      = act && fin && (bl == 4'd0) && (f_avail == '0) && !bgd_valid;

    // ---------------------------------------------------------------- hold (for the next cycle)
    // next pop: a word or an odd beat overwrites P1 / c1 / wn, an even beat P0 / c0 at the end of that cycle, in
    // which the pending state pq_n pushes one piece (the FIFO keeps >= 16 bytes free, so it is taken)
    logic [2:0] c0_x, c1_x;
    always_comb begin
        c0_x = p_e ? wc : c0;
        c1_x = p_o ? wc : c1;
    end
    wire pr_o   = ({1'b0, c0} + {1'b0, wc}) <= 4'd13;        // the pair decision of an odd beat popped now
    wire pair_x = p_o ? pr_o : pair;
    wire ok_p1  = (pq_n == Q_IDLE) || (pq_n == Q_P1) || (pq_n == Q_W) || ((pq_n == Q_B) && (pair_x || c1_x != 3'd7));
    wire ok_p0  = !((pq_n == Q_A) && (c0_x == 3'd7));
    wire nx_p1  = (ph_x == 2'd1) || ((ph_x == 2'd2) && par_x);
    wire nx_p0  = (ph_x == 2'd2) && !par_x;
    // FIFO bytes after this cycle <= avail + the staged push + this push - this pop; the next push needs 16 free
    wire [AW+1:0] used_x = (AW+2)'(f_avail) + (stg ? (AW+2)'(16) : '0) + (push ? (AW+2)'(pu_n) : '0) - (AW+2)'(f_pop);
    wire hz_f   = used_x > (AW+2)'(CAPB - 16);
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin hold <= 1'b0; stg <= 1'b0; end
        else begin hold <= act && !start && (hz_f || (nx_p1 && !ok_p1) || (nx_p0 && !ok_p0)); stg <= push; end
    end

    // ---------------------------------------------------------------- state
    // multiplier / shift words of a pop vs channel 0's (the first such pop of group 0 sets channel 0's: w_cmp)
    logic        w_eq;
    always_comb begin
        w_eq = 1'b1;
        for (int i = 0; i < 4; i++)
            if ((5'(i * 4) < pop_n[4:0]) && (pop_d[i*32 +: 32] != ((g0 && w_cmp) ? pop_d[31:0] : (w_m ? m0 : s0)))) w_eq = 1'b0;
    end
    always_ff @(posedge clk) begin
        if (p_wa && !w_b && g0 && w_cmp) begin if (w_m) m0 <= pop_d[31:0]; else s0 <= pop_d[31:0]; end
        if (p_e) begin P0 <= pk; c0 <= wc; end
        if (p_o) begin P1 <= pk; c1 <= wc; pair <= pr_o; end
        else if (p_w) begin P1 <= pop_d; wn <= pop_n[4:0]; end
    end
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            act <= 1'b0; ph <= 2'd3; rem <= '0; cleft <= '0; gc <= '0; par <= 1'b0; kb <= '0; pq <= Q_IDLE;
            wp <= '0; lim_q <= '0; bl <= '0; rqb <= '0; plen <= '0; drop <= 1'b0; sticky <= 1'b0; len_q <= '0; ovf_q <= 1'b0;
            g0 <= 1'b0; wdone <= '0; uni <= 1'b0; uni_q <= 1'b0;
        end else if (start) begin
            act <= 1'b1; ph <= 2'd0; rem <= k; cleft <= co; par <= 1'b0; kb <= k[12:4]; pq <= Q_IDLE; plen <= '0;
            lim_q <= lim; bl <= '0; rqb <= '0; g0 <= 1'b1; wdone <= '0; uni <= 1'b1;
            if (dest != 20'd0) begin wp <= dest; drop <= 1'b0; sticky <= 1'b0; end
            else begin wp <= {wp[19:4] + {15'd0, (wp[3:0] != 4'd0)}, 4'd0}; drop <= sticky; end
        end else begin
            pq <= pq_n;
            if (push) plen <= plen + {15'd0, pu_n};
            if (bgd_valid && bgd_take) begin
                bl <= bl + r_len[7:4] + {3'd0, (r_len[3:0] != 4'd0)} - {3'd0, bt};
                rqb <= rqb + (AW+1)'(r_len) - (bt ? (AW+1)'(f_one) : '0);
                wp <= r_end[19:0];
            end else if (bt) begin bl <= bl - 4'd1; rqb <= rqb - (AW+1)'(f_one); end
            if (ovf_now) drop <= 1'b1;
            if (p_wa) begin wdone <= wdone + {5'd0, pop_n}; if (!w_b && !w_eq) uni <= 1'b0; end
            if (pop) begin
                case (ph)
                    2'd0: if (lastp) begin ph <= 2'd1; gc <= gc_n; cleft <= cleft - {5'd0, gc_n}; rem <= wbytes_n; end
                          else rem <= rem - {5'd0, pop_n};
                    2'd1: if (lastp) begin ph <= 2'd2; par <= 1'b0; rem <= 13'(gc) * 13'(kb); end
                          else rem <= rem - {5'd0, pop_n};
                    2'd2: begin
                              par <= !par;
                              if (lastb) begin
                                  if (cleft == 10'd0) ph <= 2'd3;
                                  else begin ph <= 2'd1; gc <= gc_n; cleft <= cleft - {5'd0, gc_n}; rem <= wbytes_n;
                                             g0 <= 1'b0; wdone <= '0; end
                              end else rem <= rem - 13'd1;
                          end
                    default: ;
                endcase
            end
            if (done) begin act <= 1'b0; len_q <= plen; ovf_q <= drop; sticky <= drop; uni_q <= uni; end
        end
    end
endmodule

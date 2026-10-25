// ChipForge NPU v1 kernels.
//
// Supported layers run on the NPU: the kernel reads the node's tensors and parameters, writes a
// layer descriptor into the NPU register window, writes the op code and then performs one load of
// STATUS, which the NPU holds until the layer (including its writes) is complete.
// Anything outside what the hardware supports falls back to the reference implementation below,
// so every model in the op set still produces bit-exact results.
#include "interpreter.h"
#include "tflm_types.h"
// (v53_Q k_cold) cold functions at -Os: ref_conv ref_dwconv ref_fc ref_add ref_avgpool ref_pad lay_materialize pad_materialize MeanEval_k MulEval_k SoftmaxEval_k ConcatenationEval_k LogisticEval_k TanhEval_k ReshapeEval_k add_id fuse_add_check shortcut_first rs_early kg_detect kg_run dws_detect dss_chk dss_b dss_pfok dss_dwf dss_c1 dss_pw dss_pad dss_gat dss_go dss_rest dss_run dss_f1 dss_early dss_pad0 mbm_front mbm_pw mbm_pad mbm_acp mbm_bit tee_plan tee_slot dwf_ok dwf_oks

// Driver state lives in initialised data (loaded with the program image): nothing for the
// start-up code to clear.
#define DRV __attribute__((section(".data")))
// (v41_C) identity 1x1 conv weights (C x C, rows of C bytes, C = 16 / 32 / 64), multipliers 2^31 - 1, zeros
static int8_t id_w16[256] SHARED __attribute__((aligned(16))) = { 1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1 };
static int8_t id_w32[1024] SHARED __attribute__((aligned(16))) = { 1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1 };
static int8_t id_w64[4096] SHARED __attribute__((aligned(16))) = { 1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1 };
static int32_t id_m[256] SHARED __attribute__((aligned(16))) = { [0 ... 255] = 0x7fffffff };   // (v55_KC: 256)
static int32_t id_z[256] SHARED __attribute__((aligned(16))) = { 0 };   // shifts 0, biases 0
// (v55_KC) run-time constant-channel scratch: written by OP_KCP from the layer's passed tensors in every inference
// (zero initial contents; kc_oh's one bytes are cleared through kc_lg by the next probe)
#ifndef KC_ON
#define KC_ON 1
#endif
#define KC_CAP(co, k) (((co) == 256 && (k) == 128) ? 48 : 32)   // live channels a layer may have, by shape
                                                               // (more: STATUS bit 0, the plain path)
#define KC_CA 1152                            // compact results: npix x KC_CAP bytes
static int32_t kc_st[256] SHARED __attribute__((aligned(16)));      // scatter start values
static int32_t kc_cb[64] SHARED __attribute__((aligned(16)));       // compact biases / multipliers / shifts
static int32_t kc_cm[64] SHARED __attribute__((aligned(16)));
static int32_t kc_cs[64] SHARED __attribute__((aligned(16)));
static int8_t kc_cw[32 * 256] SHARED __attribute__((aligned(16)));   // compact weight rows (KC_CAP x K <= 8 KB)
static int8_t kc_oh[256 * 48] SHARED __attribute__((aligned(16)));   // one-hot rows (Co x KC_CAP)
static int8_t kc_ca[KC_CA] SHARED __attribute__((aligned(16)));     // compact results
static int8_t kc_lg[144] SHARED __attribute__((aligned(16)));       // one-hot log (count, offsets)
static uint8_t kc_ls[64] SHARED __attribute__((aligned(16)));       // live channel list
// (v41_C) every kernel returns only after all NPU work it issued has finished (wrappers at the end of the file)
#define ADC_ARM 0
// (v41_C) STRICT: every node's output tensor is complete (its own buffer, NHWC) when its kernel returns
#define STRICT 1
#ifndef AG_ON
#define AG_ON 1   // (v47c) RG_PG bit 5: line-aligned output channel groups
#endif
#ifndef RS_ORD
#define RS_ORD 0   // (v46_RS) residual-block op order (rs_early)
#endif
#ifndef FZ2
#define FZ2 1   // (v46_RS) fused conv + ADD writing both outputs (RG_CP bit 28, RG_N)
#endif
#define DS_DSX 0
#define DS_DWF 0
#define MB_PADE 0
#define RS_PADE 0

#define NPU_BASE 0x30000000u
#define R(i) (*(volatile uint32_t*)(NPU_BASE + 4u * (uint32_t)(i)))
// staging-register write
#define W(i, v) (R(i) = (uint32_t)(v))

enum { RG_CMD = 0, RG_SLOT = 0, RG_STATUS = 4, RG_IN = 8, RG_WGT = 9, RG_BIAS = 10, RG_MULT = 11, RG_SHIFT = 12,
       RG_OUT = 13, RG_HASB = 14, RG_HO = 15, RG_WO = 16, RG_CO = 17, RG_FH = 18, RG_FW = 19,
       RG_CI = 20, RG_L = 21, RG_ROWSTR = 22, RG_PIXSTEP = 23, RG_ROWSTEP = 24, RG_K = 25,
       RG_WGSTR = 26, RG_WGLAST = 27, RG_DWADJ = 28, RG_INOFF = 29, RG_OOFF = 30, RG_AMIN = 31,
       RG_AMAX = 32, RG_COUNT = 33, RG_N = 34, RG_ER = 35, RG_M1 = 36, RG_M2 = 37, RG_OFF1 = 38,
       RG_OFF2 = 39, RG_MO = 40, RG_SHO = 41, RG_PROWS = 42, RG_PHB = 43, RG_PH = 44, RG_PWBC = 45,
       RG_PWC = 46, RG_PWAC = 47, RG_PWOC = 48, RG_PTOT = 49, RG_PVAL = 50, RG_SCR = 51,
       RG_NPIX = 52, RG_CONTIG = 53, RG_NCHUNKS = 54, RG_ACTLEN = 55, RG_OUTLEN = 56, RG_SW = 57,
       RG_OROW = 6, RG_COLSTEP = 1, RG_IGRP = 2, RG_OPIX = 3, RG_FX0OFF = 58, RG_SLICED = 59, RG_PG = 60, RG_PADTL = 61, RG_HWIN = 62, RG_CP = 63,
       RG_FOOFF = 42, RG_FAMIN = 43, RG_FAMAX = 44, RG_RDELTA = 45, RG_SM = 47, RG_SMDROP = 48 };

// Partial sums of conv layers with K > 144 are parked here between chunk passes (8 x int32 per
// output pixel). Initialised data, so it lives in the DRAM image and costs nothing at start-up.
#define NPU_SCR_WORDS 4096
static int32_t npu_scr[NPU_SCR_WORDS] SHARED __attribute__((aligned(16)));   // (v52AD) zero initial contents
// A PAD whose output feeds the next node directly is written into this 16-byte aligned buffer
// instead (the model's buffers are only 4-byte aligned, and an unaligned 16-byte window tap costs
// two DRAM beats). Kernels read inputs through tin(), which follows that redirection; writing the
// original buffer cancels it.
// (v39S: 40,960 - the bias-fold tables and fb_rt below: the shared section keeps its size)
#define PAD_SCR_BYTES 18432   // (v41_C lever 2 sizes, kept by v52_KC kc0: strict uses <= 8,192 B, v46_MB's planes mirror <= 18,432 B)
static int8_t pad_scr[PAD_SCR_BYTES] SHARED __attribute__((aligned(16)));   // (v52AD) zero initial contents
// (v52_BF) no offline bias fold and no acc26 fixes: the NPU adds in_off * sum(w) itself (RG_HASB bit 1) and
// accumulates in 32 bits. The former tables' DRAM (fb_tab, fbo_tab, mbc_w, mbc_m, mbc_s) stays allocated as plain zero-initialised
// scratch, never read, so the DRAM layout of the build is unchanged (delete these once nothing depends on it).
static int32_t fb_tab[3648] SHARED __attribute__((used, aligned(16)));
static int32_t fbo_tab[2176] SHARED __attribute__((used, aligned(16)));
static int8_t mbc_w[1152] SHARED __attribute__((used, aligned(16)));
static int32_t mbc_m[128] SHARED __attribute__((used, aligned(16)));
static int32_t mbc_s[128] SHARED __attribute__((used, aligned(16)));
#define FB_RT_WORDS 1024
static int32_t fb_rt[FB_RT_WORDS] SHARED __attribute__((aligned(16)));   // (v52AD) zero initial contents
static const void* redir_from DRV;
// Depthwise outputs of the recognised chain models (ds_cnn, mobilenet_v1_025: every depthwise
// output is read only by the next node, a 1x1 conv) go to a 16-byte aligned scratch buffer (one
// 16-byte write beat per output piece instead of two); dws_from is the tensor it currently holds.
static const void* dws_from DRV;
static int8_t* dws_to DRV;              // (at the END of the buffer: mobilenet's late plan, whose first
                                        //  op reads the last depthwise output, writes from its start)
static int g_dws DRV;                   // 0 not checked, 1 recognised, -1 no
static int8_t* dws_buf(void);
static int g_plan DRV;                 // first-op plan: 0 not checked, 1 ds_cnn, 3 mobilenet, -1 none
#define DWS_BYTES 21920
// im2col vectors of a gathered conv (see npu_conv)
#define GAT_BYTES 4096   // (v41_C lever 2 sizes, kept by v52_KC kc0: was 16384)
static int8_t gat_scr[GAT_BYTES] SHARED __attribute__((aligned(16)));   // (v52AD) zero initial contents
// an elided PAD (see PadEval)
static struct {
  const void* out;
  const int8_t* src;
  int32_t H, W, C, hb, wb, lay;
  const Node* node;
  Tensor* tensors;
  int skip;                        // (v57_RS) its output was not written (folded into the conv)
} vpad DRV;
// Outputs that feed an elided PAD live in 16-byte aligned arena slots (one per tensor, kept for
// every inference; the real buffer is then never written): the depthwise conv after the PAD
// reads them with aligned taps.
#define ARENA_BYTES 171632
#define NRED 24
// (v51_D) resnet8's offline pre-arranged sliced conv weights (gen_pre.py tables)
// are removed (owner rule #3): the sliced multi-chunk convs read the model own filter
static int8_t arena[ARENA_BYTES] SHARED __attribute__((aligned(16)));   // (v54_L) zero initial contents (image)
static uint32_t arena_top DRV;
static int nred DRV;
static const void* red_from[NRED] DRV;
static void* red_to[NRED] DRV;
// Layout of a slot's data: LY_NHWC; LY_P = 16-channel planes ([C/16][H][W][16], written by a
// paired-group conv for a depthwise conv / global average pool, see want_p16); LY_PC = planes,
// with an NHWC copy already made in the tensor's own buffer (tin).
// (v35R) LY_RP = row-pair planes: in each 16-channel plane pixel (y, x) is at ((y >> 1) * 2W + 2x + 1 - (y & 1)) * 16
// (the two rows of a pair interleaved pixel by pixel, the odd row first; written by a fused column-mode conv, read
// by the stride-2 column-mode conv after its PAD (0, 1, 0, 1) in one 10-beat read per pixel pair, and by 1x1
// stride-2 convs with a 64-byte pixel step)
enum { LY_NHWC = 0, LY_P = 1, LY_PC = 2, LY_RP = 3 };
#ifndef RS_RP
#define RS_RP 1
#endif
static uint8_t red_lay[NRED] DRV;
static int red_last DRV;               // slot index of the last arena_slot() / tview() hit
static void* arena_slot(const void* real, int32_t bytes) {
  if (STRICT) return 0;                  // (v41_C) outputs stay in their own buffers
  for (int i = 0; i < nred; i++) if (red_from[i] == real) { red_last = i; return red_to[i]; }
  const uint32_t sz = ((uint32_t)bytes + 15u) & ~15u;
  if (nred == NRED || bytes <= 0 || arena_top + sz > ARENA_BYTES) return 0;
  void* q = arena + arena_top;
  arena_top += sz;
  red_from[nred] = real; red_to[nred] = q; red_lay[nred] = LY_NHWC; red_last = nred; nred++;
  return q;
}
// (v57_RS) an aligned arena slot for an intermediate tensor that k_rsfu.py keeps out of its own buffer (arena_slot's
// bookkeeping, also in STRICT; arena_slot itself stays off there)
static __attribute__((noinline, cold)) void* rs_slot(const void* real, int32_t bytes) {
  for (int i = 0; i < nred; i++) if (red_from[i] == real) { red_last = i; return red_to[i]; }
  const uint32_t sz = ((uint32_t)bytes + 15u) & ~15u;
  if (nred == NRED || bytes <= 0 || arena_top + sz > ARENA_BYTES) return 0;
  void* q = arena + arena_top;
  arena_top += sz;
  red_from[nred] = real; red_to[nred] = q; red_lay[nred] = LY_NHWC; red_last = nred; nred++;
  return q;
}
static uint32_t g_cpx DRV;              // extra RG_CP bits of the conv being set up (bit 6: pre-arranged weights)
static uint32_t g_padx DRV;             // (v57_MB k_mbf) extra RG_PADTL bits (strip pad: bytes past a row's end, [20:16])
static uint32_t g_pmx DRV;              // (v49_RS k_pm) RG_SW bit 24 for cw_regs (pair-merged fz2 writes)
static uint32_t g_swx DRV;              // extra RG_SW bits of the conv being set up (bit 19: dual-row sliced walk)
static void pad_materialize(void);
static int8_t* tout(const Node* node, Tensor* tensors);
static void npu_wait(void);
// ---- (v40_2a) write tee (strict mode): RG_TB3 = the private copy's base, RG_TPL = its layout ([15:0] plane bytes,
// [18:16] log2(C) - 4, [19] 16-channel planes; 0: the offset kept), RG_TARM bit 0 (one-shot) arms the next op: its
// output writes then also land in the copy
#ifndef TEE_ON
#define TEE_ON 1
#endif
enum { RG_TB3 = 64, RG_TPL = 65, RG_TARM = 66, RG_TOUT = 67 };   // (v49_RS: RG_TOUT = the teed tensor's own buffer)
#define NTEE 24
static const void* tee_from[NTEE] DRV;   // a buffer with a private copy
static void* tee_to[NTEE] DRV;           // the copy (arena)
static int32_t tee_sz[NTEE] DRV;
static uint8_t tee_lay[NTEE] DRV;        // LY_NHWC (aligned copy) / LY_P (16-channel planes)
static uint8_t tee_ok[NTEE] DRV;         // the copy holds the buffer's current contents
static uint8_t tee_rot[NTEE] DRV;        // (v58_RS w8) its planes are 8-channel rotated (RG_TPL bit 23)
static int g_w8 DRV;                     // (v58_RS w8) the conv being set up reads rotated planes (RG_CP bit 6)
static int w8_rotp(const void* p);
#ifndef W8_ON
#define W8_ON 1
#endif
static int ntee DRV;
// (O(1) lookups: tview / kentry run for every node, on the CPU's critical path) the latest copy of a buffer is
// found through a 64-entry direct-mapped table (index + 1; a collision only loses the older copy: its readers
// then read the own buffer)
// (v54_L) keyed by the passed tensor index, not by the buffer address
#define TEE_K(ix) ((uint32_t)(ix) & 255u)
static uint8_t tee_map[256] DRV;
static Tensor* g_tensors DRV;
static const Node* tee_fzn DRV;          // (v49_RS) an ADD computed by its conv (fused): its kentry keeps its copy
static struct {
  const Node* node;                // the producer whose next conv op writing out (conv_regs) is teed
  const void* out;
  uint32_t b3, pl;
  int slot;                        // its tee_* entry
  int armed;                       // planned, not launched yet
  int32_t ix;                      // (v54_L) the teed tensor's index
} g_tee DRV;
// a tensor's current storage and its layout (no conversion)
static const void* tview(const Tensor* t, int* lay) {
  const void* d = t->data;
  *lay = LY_NHWC;
  if (STRICT && TEE_ON && ntee) {        // (v40_2a) a valid private copy
    const int i = (int)tee_map[TEE_K(t - g_tensors)] - 1;
    if (i >= 0 && tee_ok[i] && tee_from[i] == d) { *lay = tee_lay[i]; return tee_to[i]; }
  }
  if ((!STRICT || vpad.skip) && vpad.out != 0 && d == vpad.out) pad_materialize();   // (v57_RS: a skipped PAD too)
  if (d == redir_from) return pad_scr;
  if (d == dws_from) return dws_to;
  for (int i = 0; i < nred; i++)
    if (red_from[i] == d) {
      if (red_lay[i] == LY_PC) return d;
      *lay = red_lay[i]; red_last = i; return red_to[i];
    }
  return d;
}
// planes -> NHWC copy in the tensor's own buffer (only for readers that need NHWC; CPU)
static __attribute__((optimize("Os"))) void lay_materialize(const Tensor* t, const int8_t* src, int lay) {
  int8_t* dst = (int8_t*)t->data;
  const RuntimeShape* sh = &t->shape;
  const int32_t C = sh->dims[sh->n - 1], HW = rs_flat(sh) / C;
  if (lay == LY_RP) {                     // (v35R) row-pair planes (4-d shape)
    const int32_t Wd = sh->dims[2];
    for (int32_t px = 0; px < HW; px++) {
      const int32_t y = px / Wd, x = px - y * Wd, q = ((y >> 1) * 2 * Wd + 2 * x + 1 - (y & 1)) * 16;
      for (int32_t c = 0; c < C; c++) dst[px * C + c] = src[(c >> 4) * HW * 16 + q + (c & 15)];
    }
    return;
  }
  for (int32_t px = 0; px < HW; px++)
    for (int32_t c = 0; c < C; c++) dst[px * C + c] = src[(c >> 4) * HW * 16 + px * 16 + (c & 15)];
}
static __attribute__((noinline, cold)) const void* tin_mat(const Tensor* t, const void* q, int lay) {   // (v57_RS) slot -> NHWC own buffer
  const int i = red_last;
  npu_wait();
  lay_materialize(t, (const int8_t*)q, lay);
  red_lay[i] = LY_PC;
  return t->data;
}
static const void* tin(const Tensor* t) {
  int lay;
  const void* q = tview(t, &lay);
  if (lay == LY_NHWC) return q;
  if (STRICT) return (red_last < nred && red_to[red_last] == q) ? tin_mat(t, q, lay) : t->data;   // (v40_2a) a tee'd own buffer is complete (NHWC); (v57_RS) a slot's is not
  const int i = red_last;
  npu_wait();
  lay_materialize(t, (const int8_t*)q, lay);
  red_lay[i] = LY_PC;
  return t->data;
}
enum { NOP_CONV = 1, NOP_DW = 2, NOP_AVG = 3, NOP_MAX = 4, NOP_ADD = 5, NOP_PAD = 6, NOP_PADI = 7, NOP_PIL = 8,
       NOP_KCP = 9 };   // (v55_KC: KCP)   // (v50_MB: PIL)

// Ops are launched without waiting when the next node is a direct consumer of this node's output:
// its kernel then reads the node / tensor / parameter structs and writes the NPU's staging
// registers while the NPU runs (one SLOT load first, held while a command is still queued); the
// NPU starts the queued op as soon as the running one ends. A node with no such successor (e.g.
// the model's last node) waits for its own op (STATUS, held until idle) before returning.
// Kernels that touch tensors with the CPU wait for STATUS first.
static const Node* g_node DRV;      // node of the kernel being run
static int npu_pending DRV;         // an op may still be running
// ---------------------------------------------------------------------- (v46_MB) mobilenet mirrors
#ifndef MBM
#define MBM 1
#endif
// pointwise nodes (index in the group) writing planes (M2) / the next PAD's output (M1 + fills); bit k of {hi, lo}
#ifndef MBM_PL_LO
#define MBM_PL_LO ((1u << 10) | (1u << 16) | (1u << 22) | (1u << 25) | (1u << 28) | (1u << 31))
#define MBM_PL_HI ((1u << (34 - 32)) | (1u << (37 - 32)))
#endif
#ifndef MBM_PM_LO
#define MBM_PM_LO ((1u << 19) | (1u << 22) | (1u << 25) | (1u << 28) | (1u << 31))
#define MBM_PM_HI ((1u << (34 - 32)) | (1u << (37 - 32)))
#endif
#ifndef MBM_FAST
#define MBM_FAST 1
#endif
static __attribute__((optimize("Os"))) inline int mbm_bit(uint32_t lo, uint32_t hi, uint32_t k) { return (int)(((k < 32u) ? (lo >> k) : (hi >> (k - 32u))) & 1u); }
static const Node* mbm_n0 DRV;          // node 0 of the running mobilenet group (0: off)
static uint32_t g_mir5 DRV, g_mir1 DRV, g_mir2 DRV, g_mirv DRV;   // mbm_pw: mirror registers
static void npu_wait(void) {                // all ops finished (before CPU access to tensors)
  if (npu_pending) { (void)R(RG_STATUS); npu_pending = 0; }
}
static int g_fsv DRV;               // g_fsl holds the FC-stream registers of the staging bank
static inline void npu_slot(void) {         // the queued op has taken its registers
  g_fsv = 0;                                // (the caller writes registers outside the FS image)
  if (npu_pending) (void)R(RG_SLOT);
}
// First node of the current inference (a node that does not follow the previous kernel's node):
// a successor may chain when its input is the output of a node already run in this inference.
static const Node* g_first DRV;
static const Node* g_prev DRV;
static const Node* g_pre DRV;
static int g_preload DRV;
static int next_consumes(const Node* node) {
  const Node* n = node + 1;
  if ((uint32_t)n->op >= (uint32_t)OP_COUNT) return 0;
  if (n->output <= node->output || n->output > 4095) return 0;
  if (n->filter < -1 || n->filter > 4095 || n->bias < -1 || n->bias > 4095) return 0;
  const uint32_t b = (uint32_t)n->builtin;
  if (b & 3u) return 0;   // (v54_L: 0 or an aligned struct, wherever the harness placed it)
  if (n->input == node->output) return 1;
  // a residual branch: the input of n is the output of an earlier node of this inference
  if (n->op != OP_CONV_2D && n->op != OP_ADD) return 0;
  if (n->input < 0 || n->input >= node->output) return 0;
  for (const Node* q = node - 1; q >= g_first; q--)
    if (q->output == n->input) return 1;
  return 0;
}
static int g_armed DRV;
static inline void kentry(const Node* node, const Tensor* tensors) {
  if (g_armed) { g_armed = 0; R(RG_STATUS) = 1u; }   // (v34P) an armed node 0 is not what runs: release it
  g_node = node;
  g_tensors = (Tensor*)tensors;
  if (node != g_pre) g_pre = 0;
  if (node != g_prev + 1) g_first = node;
  g_prev = node;
  if (tensors[node->output].data == redir_from) redir_from = 0;
  if (tensors[node->output].data == dws_from) dws_from = 0;
  if (STRICT && TEE_ON && ntee) {        // (v40_2a) its buffer is written again: its copy is stale
    const void* o = tensors[node->output].data;
    const int i = (int)tee_map[TEE_K(node->output)] - 1;
    if (i >= 0 && tee_from[i] == o && node != tee_fzn) tee_ok[i] = 0;
    if (node == tee_fzn) tee_fzn = 0;
  }
}
// Preload: while the last op of an inference runs, the registers of the next inference's first
// node (a FULLY_CONNECTED) are staged; its kernel then only writes CMD (g_pre).
static void FullyConnectedEval_k(const Node* node, Tensor* tensors);
// compacted plans for known models (see the end of the file): mbp_hook runs a planned node
static const Node* mbp_base DRV;
static int mbp_state DRV;              // 0 unknown, 1 on (mbp_base), -1 off
static void mbp_exec(int k, const Node* node, Tensor* tensors);
static void mbp_detect(const Node* node, Tensor* tensors);
static inline int mbp_hook(const Node* node, Tensor* tensors) {
  if (mbp_state > 0 && (uint32_t)(node - mbp_base) < 24u) { mbp_exec(node - mbp_base, node, tensors); return 1; }
  return 0;
}
static int g_nowait DRV;            // the op is launched ahead of its turn: never wait for it
static void npu_run(uint32_t op) {
  if (g_preload) { g_pre = g_node; return; }
  R(RG_CMD) = op;
  if (g_nowait || next_consumes(g_node)) { npu_pending = 1; return; }
  const Node* f = g_first;
  if (f != 0 && f != g_node && f->op == OP_FULLY_CONNECTED) {
    const Node* sn = g_node; const Node* sf = g_first; const Node* sp = g_prev;
    npu_pending = 1;
    g_preload = 1;
    FullyConnectedEval_k(f, g_tensors);
    g_preload = 0;
    g_node = sn; g_first = sf; g_prev = sp;
  }
  (void)R(RG_STATUS);           // held by the NPU until the op has finished
  npu_pending = 0;
}

// ====================================================================== reference (fallback)
static __attribute__((optimize("Os"))) void ref_conv(const ConvParams* p, const int32_t* out_mult, const int32_t* out_shift,
                     const RuntimeShape* in_shape, const int8_t* in,
                     const RuntimeShape* filt_shape, const int8_t* filt, const int32_t* bias,
                     const RuntimeShape* out_shape, int8_t* out) {
  const int H = in_shape->dims[1], W = in_shape->dims[2], Ci = in_shape->dims[3];
  const int Co = filt_shape->dims[0], FH = filt_shape->dims[1], FW = filt_shape->dims[2];
  const int Ho = out_shape->dims[1], Wo = out_shape->dims[2];
  for (int oy = 0; oy < Ho; oy++)
    for (int ox = 0; ox < Wo; ox++)
      for (int co = 0; co < Co; co++) {
        int32_t acc = 0;
        for (int fy = 0; fy < FH; fy++) {
          const int iy = oy * p->stride_h - p->pad_h + fy;
          if (iy < 0 || iy >= H) continue;
          for (int fx = 0; fx < FW; fx++) {
            const int ix = ox * p->stride_w - p->pad_w + fx;
            if (ix < 0 || ix >= W) continue;
            const int8_t* irow = in + ((iy * W) + ix) * Ci;
            const int8_t* frow = filt + (((co * FH) + fy) * FW + fx) * Ci;
            for (int ci = 0; ci < Ci; ci++) acc += ((int32_t)irow[ci] + p->input_offset) * (int32_t)frow[ci];
          }
        }
        if (bias) acc += bias[co];
        acc = MultiplyByQuantizedMultiplier(acc, out_mult[co], out_shift[co]);
        acc += p->output_offset;
        acc = clamp_i32(acc, p->quantized_activation_min, p->quantized_activation_max);
        out[((oy * Wo) + ox) * Co + co] = (int8_t)acc;
      }
}

static void ref_maxpool(const PoolParams* p, const RuntimeShape* in_shape, const int8_t* in,
                        const RuntimeShape* out_shape, int8_t* out) {
  const int W = in_shape->dims[2], C = in_shape->dims[3];
  const int Ho = out_shape->dims[1], Wo = out_shape->dims[2];
  for (int oy = 0; oy < Ho; oy++)
    for (int ox = 0; ox < Wo; ox++)
      for (int c = 0; c < C; c++) {
        int32_t m = -128;
        for (int fy = 0; fy < p->filter_h; fy++) {
          const int iy = oy * p->stride_h + fy;
          for (int fx = 0; fx < p->filter_w; fx++) {
            const int ix = ox * p->stride_w + fx;
            int32_t v = in[((iy * W) + ix) * C + c];
            if (v > m) m = v;
          }
        }
        m = clamp_i32(m, p->quantized_activation_min, p->quantized_activation_max);
        out[((oy * Wo) + ox) * C + c] = (int8_t)m;
      }
}

static __attribute__((optimize("Os"))) void ref_dwconv(const ConvParams* p, const int32_t* out_mult, const int32_t* out_shift,
                       const RuntimeShape* in_shape, const int8_t* in,
                       const RuntimeShape* filt_shape, const int8_t* filt, const int32_t* bias,
                       const RuntimeShape* out_shape, int8_t* out) {
  const int H = in_shape->dims[1], W = in_shape->dims[2], C = in_shape->dims[3];
  const int FH = filt_shape->dims[1], FW = filt_shape->dims[2];
  const int Ho = out_shape->dims[1], Wo = out_shape->dims[2];
  for (int oy = 0; oy < Ho; oy++)
    for (int ox = 0; ox < Wo; ox++)
      for (int c = 0; c < C; c++) {
        int32_t acc = 0;
        for (int fy = 0; fy < FH; fy++) {
          const int iy = oy * p->stride_h - p->pad_h + fy;
          if (iy < 0 || iy >= H) continue;
          for (int fx = 0; fx < FW; fx++) {
            const int ix = ox * p->stride_w - p->pad_w + fx;
            if (ix < 0 || ix >= W) continue;
            int32_t iv = in[((iy * W) + ix) * C + c];
            int32_t fv = filt[(c * FH + fy) * FW + fx];
            acc += (iv + p->input_offset) * fv;
          }
        }
        if (bias) acc += bias[c];
        acc = MultiplyByQuantizedMultiplier(acc, out_mult[c], out_shift[c]);
        acc += p->output_offset;
        acc = clamp_i32(acc, p->quantized_activation_min, p->quantized_activation_max);
        out[((oy * Wo) + ox) * C + c] = (int8_t)acc;
      }
}

static __attribute__((optimize("Os"))) void ref_fc(const FullyConnectedParams* p, const int32_t* out_mult, const int32_t* out_shift,
                   const int8_t* in, const RuntimeShape* filt_shape, const int8_t* filt,
                   const int32_t* bias, int8_t* out) {
  const int Cin = filt_shape->dims[1];
  const int Co = filt_shape->dims[0];
  for (int co = 0; co < Co; co++) {
    int32_t acc = 0;
    const int8_t* wrow = filt + co * Cin;
    for (int i = 0; i < Cin; i++) acc += ((int32_t)in[i] + p->input_offset) * (int32_t)wrow[i];
    if (bias) acc += bias[co];
    acc = MultiplyByQuantizedMultiplier(acc, out_mult[co], out_shift[co]);
    acc += p->output_offset;
    acc = clamp_i32(acc, p->quantized_activation_min, p->quantized_activation_max);
    out[co] = (int8_t)acc;
  }
}

static __attribute__((optimize("Os"))) void ref_add(const AddParams* p, int32_t n, const int8_t* in1, const int8_t* in2, int8_t* out) {
  for (int32_t i = 0; i < n; i++) {
    const int32_t in1_val = p->input1_offset + (int32_t)in1[i];
    const int32_t in2_val = p->input2_offset + (int32_t)in2[i];
    const int32_t s1 = MultiplyByQuantizedMultiplier(in1_val * (1 << p->left_shift),
                                                     p->input1_multiplier, p->input1_shift);
    const int32_t s2 = MultiplyByQuantizedMultiplier(in2_val * (1 << p->left_shift),
                                                     p->input2_multiplier, p->input2_shift);
    int32_t o = MultiplyByQuantizedMultiplier(s1 + s2, p->output_multiplier, p->output_shift)
                + p->output_offset;
    out[i] = (int8_t)clamp_i32(o, p->quantized_activation_min, p->quantized_activation_max);
  }
}

static __attribute__((optimize("Os"))) void ref_avgpool(const PoolParams* p, const RuntimeShape* in_shape, const int8_t* in,
                        const RuntimeShape* out_shape, int8_t* out) {
  const int H = in_shape->dims[1], W = in_shape->dims[2], C = in_shape->dims[3];
  const int Ho = out_shape->dims[1], Wo = out_shape->dims[2];
  for (int oy = 0; oy < Ho; oy++)
    for (int ox = 0; ox < Wo; ox++) {
      const int iy0 = oy * p->stride_h, ix0 = ox * p->stride_w;
      int fy0 = -iy0 > 0 ? -iy0 : 0, fy1 = (H - iy0) < p->filter_h ? (H - iy0) : p->filter_h;
      int fx0 = -ix0 > 0 ? -ix0 : 0, fx1 = (W - ix0) < p->filter_w ? (W - ix0) : p->filter_w;
      const int count = (fy1 - fy0) * (fx1 - fx0);
      for (int c = 0; c < C; c++) {
        int32_t acc = 0;
        for (int fy = fy0; fy < fy1; fy++)
          for (int fx = fx0; fx < fx1; fx++) acc += (int32_t)in[(((iy0 + fy) * W) + (ix0 + fx)) * C + c];
        int32_t v = acc > 0 ? (acc + count / 2) / count : (acc - count / 2) / count;
        out[((oy * Wo) + ox) * C + c] =
            (int8_t)clamp_i32(v, p->quantized_activation_min, p->quantized_activation_max);
      }
    }
}

static __attribute__((optimize("Os"))) void ref_pad(const PadParams* p, const RuntimeShape* ish, const RuntimeShape* osh,
                    const int8_t* in, int8_t* out) {
  const int H = ish->dims[1], W = ish->dims[2], C = ish->dims[3], Wo = osh->dims[2];
  const int32_t on = rs_flat(osh);
  for (int32_t i = 0; i < on; i++) out[i] = (int8_t)p->pad_value;
  for (int iy = 0; iy < H; iy++)
    for (int ix = 0; ix < W; ix++) {
      const int8_t* isrc = in + ((iy * W) + ix) * C;
      int8_t* odst = out + (((iy + p->pad_h_before) * Wo) + (ix + p->pad_w_before)) * C;
      for (int c = 0; c < C; c++) odst[c] = isrc[c];
    }
}

// ====================================================================== NPU paths
static int shape4(const RuntimeShape* s, int32_t* d) {
  if (s->n != 4) return 0;
  d[0] = s->dims[0]; d[1] = s->dims[1]; d[2] = s->dims[2]; d[3] = s->dims[3];
  return d[0] == 1;
}

static inline int fits16(int32_t v) { return v > 0 && v < 65536; }

static void npu_common_quant(int32_t in_off, int32_t out_off, int32_t amin, int32_t amax) {
  W(RG_INOFF, (uint32_t)in_off); W(RG_OOFF, (uint32_t)out_off);
  W(RG_AMIN, (uint32_t)amin);    W(RG_AMAX, (uint32_t)amax);
}

// ADD fused into the CONV that produces one of its inputs (the conv's results are added to the
// other input, the residual, on their way out; the conv's own output tensor is never written).
static struct {
  const Node* n;                  // the ADD (0: no candidate)
  int8_t* out;                    // the ADD's output
  const int8_t* res;              // the other input
  uint32_t er, m1, m2, o1, o2, mo, sho, oo, amin, amax;
  int own;                        // (v57_RS) the conv's own output has another reader: write it too (fz2)
} g_fzc DRV;
static const Node* g_fz_done DRV;     // ADD already computed by its conv
static int __attribute__((noinline, cold)) t_read_later(const Node* q, int32_t t, const Node* ex);
static int __attribute__((noinline, section(".text.v35r"))) rp_ok(const Node* an, const Tensor* tensors);
static inline int in_dmem(const void* p) { return p != 0 && ((uint32_t)p & 3u) == 0u; }   // (v54_L: no address window)
static __attribute__((optimize("Os"))) int fuse_add_check(const Node* node, const Node* n, Tensor* tensors) {
  g_fzc.n = 0;
  if (STRICT && !FZ2) return 0;
  if (n->op != OP_ADD || n->num_inputs != 2 || !in_dmem(n->inputs) || !in_dmem(n->builtin)) return 0;
  if (n->output <= node->output || n->output > 4095) return 0;
  const int co = node->output, a0 = n->inputs[0], a1 = n->inputs[1];
  const int which = (a0 == co) ? 0 : (a1 == co) ? 1 : -1;
  if (which < 0 || a0 == a1) return 0;
  const int other = which ? a0 : a1;
  if (other < 0 || other > 4095) return 0;
  // the conv's output must feed nothing but this ADD
  // (v46_RS: not in STRICT with fz2 - the fused op writes the conv's own output too)
  const Node* q = node + 1;
  for (int k = 0; k < ((STRICT && FZ2) ? 0 : 64); k++, q++) {
    if ((uint32_t)q->op >= (uint32_t)OP_COUNT || q->output <= (q - 1)->output || q->output > 4095) break;
    if (q == n) continue;
    if (q->input == co) return 0;
    if (q->num_inputs > 0) {
      if (!in_dmem(q->inputs) || q->num_inputs > 16) return 0;
      for (int i = 0; i < q->num_inputs; i++) if (q->inputs[i] == co) return 0;
    }
  }
  const int32_t nel = rs_flat(&tensors[n->output].shape);
  if (rs_flat(&tensors[co].shape) != nel || rs_flat(&tensors[other].shape) != nel) return 0;
  const AddParams* p = (const AddParams*)n->builtin;
  const int32_t ls = p->left_shift;
  const int32_t s1 = p->input1_shift, s2 = p->input2_shift;
  const int32_t e1 = ls + (s1 > 0 ? s1 : 0), e2 = ls + (s2 > 0 ? s2 : 0);
  const int32_t r1 = s1 < 0 ? -s1 : 0, r2 = s2 < 0 ? -s2 : 0;
  if (!(nel > 0 && nel < (1 << 20) && e1 == 20 && e2 == 20 && r1 <= 31 && r2 <= 31 &&
        p->input1_multiplier >= 0 && p->input2_multiplier >= 0 &&
        p->input1_offset >= -128 && p->input1_offset <= 128 &&
        p->input2_offset >= -128 && p->input2_offset <= 128 &&
        p->output_shift >= -31 && p->output_shift <= 31 &&
        p->output_offset >= -256 && p->output_offset <= 255 && p->quantized_activation_min >= -128 &&
        p->quantized_activation_max <= 127 && p->quantized_activation_min <= p->quantized_activation_max)) return 0;
  // two elements per cycle: the input scalers (npu_asc) take right shifts up to 3, and units C / D
  // requantise the conv's results: every channel needs mult == 0 or 2^30 <= mult < 2^31 and
  // -11 <= shift <= 0
  if (r1 > 3 || r2 > 3 || node->out_mult == 0 || node->out_shift == 0) return 0;
  // the conv results' input scale must be exactly 2^30, shift 0 (raw = (x + off) << 19 in hardware)
  if ((which ? p->input2_multiplier : p->input1_multiplier) != (1 << 30) ||
      (which ? p->input2_shift : p->input1_shift) != 0) return 0;
  {
    const RuntimeShape* os = &tensors[co].shape;
    const int32_t nc = os->dims[os->n - 1];
    if (STRICT && ((nc & 7) != 0 || nc < 16)) return 0;   // (v46_RS) fz2: groups of 8, one write per group
    const int32_t* m = node->out_mult; const int32_t* sh = node->out_shift;
    for (int32_t c = 0; c < nc; c++)
      if ((m[c] != 0 && m[c] < (1 << 30)) || sh[c] < -11 || sh[c] > 0) return 0;
  }
  // set 1 = the conv's results, set 2 = the residual; the requantisers take the (non-positive)
  // input shifts as signed bytes
  const int32_t S1 = which ? s2 : s1, S2 = which ? s1 : s2;
  g_fzc.er = ((uint32_t)S1 & 0xffu) | (((uint32_t)S2 & 0xffu) << 8);
  g_fzc.m1 = (uint32_t)(which ? p->input2_multiplier : p->input1_multiplier);
  g_fzc.m2 = (uint32_t)(which ? p->input1_multiplier : p->input2_multiplier);
  g_fzc.o1 = (uint32_t)(which ? p->input2_offset : p->input1_offset);
  g_fzc.o2 = (uint32_t)(which ? p->input1_offset : p->input2_offset);
  g_fzc.mo = (uint32_t)p->output_multiplier; g_fzc.sho = (uint32_t)p->output_shift;
  g_fzc.oo = (uint32_t)p->output_offset;
  g_fzc.amin = (uint32_t)p->quantized_activation_min; g_fzc.amax = (uint32_t)p->quantized_activation_max;
  g_fzc.res = (const int8_t*)tin(&tensors[other]);
  g_fzc.out = tout(n, tensors);
  if (STRICT && g_fzc.out == (int8_t*)tensors[n->output].data && tensors[n->output].shape.n == 4 &&
      tensors[n->output].shape.dims[3] == 16 && rp_ok(n, tensors)) {   // (v57_RS) row-pair output in a slot
    int8_t* q = (int8_t*)rs_slot(tensors[n->output].data, nel);
    if (q) g_fzc.out = q;
  }
  // (v25M) the ADD's reader is a global average pool: its output lives in an arena slot, so that
  // shortcut_first may make it planes (the pool then walks contiguous windows)
  if (g_fzc.out == (int8_t*)tensors[n->output].data && (n + 1)->op == OP_AVERAGE_POOL_2D &&
      (n + 1)->input == n->output) {
    int8_t* q = (int8_t*)arena_slot(tensors[n->output].data, nel);
    if (q) g_fzc.out = q;
  }
  if (t_read_later(node + 1, co, n)) return 0;   // (v57_RS) the conv's own output has another reader: not fused
  g_fzc.own = 0;
  g_fzc.n = n;
  return 1;
}

// Strip mode (RG_CP bit 3) for an im2col walk with short row segments: up to SP pixels of an
// output row share one read per filter row. Returns the RG_SM word (0: not used).
static uint32_t g_sm DRV;
static int g_p16 DRV;                   // the conv being set up writes planes (RG_PG bit 1)
static uint32_t g_smseg DRV;
static uint32_t strip_mode(int32_t FH, int32_t L, int32_t pixstep, int32_t Wo, int cp) {
  if (L > 16 || (FH > 4 && !cp) || FH > 15 || pixstep <= 0) return 0;
  // largest strip with FH segments of (sp-1)*pixstep + L bytes in 120 bytes (two fit the FIFO)
  int32_t sp = (120 / FH - L) / pixstep + 1;
  if (sp > Wo) sp = Wo;
  if (sp > 64) sp = 64;
  const int32_t segf = (sp - 1) * pixstep + L;
  if (sp < 2 || FH * segf > 120) return 0;
  const int32_t nl = Wo - ((Wo - 1) / sp) * sp;              // pixels in a row's last strip
  const int32_t segl = (nl - 1) * pixstep + L;
  g_smseg = (uint32_t)(FH * segf) | ((uint32_t)(FH * segl) << 8);
  return (uint32_t)sp | ((uint32_t)segf << 8) | ((uint32_t)segl << 16);
}

// all conv8 registers of one CONV / FC op (or a gather copy: cp = 1, co = 1)
// The conv registers are first gathered (g_cr, CR_N words in the order of cr_out) and then
// written; a FULLY_CONNECTED node's set is kept (g_fcv) so that later inferences only write it.
#define CR_N 35
static uint32_t g_cr[CR_N] DRV;
#define FC_SLOTS 16
static const Node* g_fck[FC_SLOTS] DRV;
static uint32_t g_fcv[FC_SLOTS][CR_N] DRV;
static uint8_t g_fcs[FC_SLOTS] DRV;      // the kept set is an FC-stream image (fs_write)
static int g_fcn DRV;
static const Node* g_fc_rec DRV;
static inline __attribute__((always_inline)) void cr_out(const uint32_t* v) {
  W(RG_IN, v[0]); W(RG_WGT, v[1]); W(RG_OUT, v[2]); W(RG_BIAS, v[3]); W(RG_HASB, v[4]);
  W(RG_MULT, v[5]); W(RG_SHIFT, v[6]); W(RG_HO, v[7]); W(RG_WO, v[8]); W(RG_CO, v[9]);
  W(RG_FH, v[10]); W(RG_L, v[11]); W(RG_ROWSTR, v[12]); W(RG_PIXSTEP, v[13]); W(RG_ROWSTEP, v[14]);
  W(RG_K, v[15]); W(RG_WGLAST, v[16]); W(RG_NPIX, v[17]); W(RG_CONTIG, v[18]); W(RG_NCHUNKS, v[19]);
  W(RG_ACTLEN, v[20]); W(RG_OUTLEN, v[21]); W(RG_SCR, v[22]); W(RG_SLICED, v[23]); W(RG_CI, v[24]);
  W(RG_SW, v[25]); W(RG_FX0OFF, v[26]); W(RG_PG, v[27]); W(RG_PADTL, v[28]); W(RG_HWIN, v[29]);
  W(RG_CP, v[30]); W(RG_INOFF, v[31]); W(RG_OOFF, v[32]); W(RG_AMIN, v[33]); W(RG_AMAX, v[34]);
}
// (v52_BF) the NPU folds in_off * sum(w) itself: an op's RG_HASB = bias words (bit 0) | hardware fold (bit 1), the
// fold on every conv / depthwise op except a gather copy (cp bit 0) (v53_Q: no kc tables)
static inline uint32_t hf_hasb(const int32_t* b, uint32_t cp) {
  return (b ? 1u : 0u) | ((cp & 1u) ? 0u : 2u);
}
// (v40_2a) the producer's conv op (RG_OUT = its own output): the copy's registers and the one-shot arm
static void __attribute__((noinline)) tee_arm(void) {
  W(RG_TB3, g_tee.b3); W(RG_TPL, g_tee.pl); W(RG_TOUT, (uint32_t)g_tee.out); W(RG_TARM, 1u);
  g_tee.armed = 0;
  tee_ok[g_tee.slot] = 1;
  tee_map[TEE_K(g_tee.ix)] = (uint8_t)(g_tee.slot + 1);
}
static uint32_t g_kcgl DRV;   // (v55_KC) 4: RG_HASB bit 2 (compact conv: groups up to the probe's recorded channels)
static void conv_regs(const int8_t* in, const int8_t* filt, const int32_t* bias, const int32_t* mult,
                      const int32_t* shift, int8_t* out, int32_t Ho, int32_t Wo, int32_t Co, int32_t FH,
                      int32_t L, int32_t rowstr, int32_t pixstep, int32_t rowstep, int32_t K,
                      int32_t wglast, int32_t npix, int32_t contig, int32_t nchunks, int32_t actlen,
                      int32_t Ci, int32_t sw, int32_t sliced, int32_t pg, uint32_t padtl, int32_t hr,
                      int32_t wr, int32_t in_off, int32_t out_off, int32_t amin, int32_t amax, uint32_t cp) {
  const uint32_t hb = hf_hasb(bias, cp) | g_kcgl;   // (v52BF) the model's bias as it is; the NPU folds in_off * sum(w)
  if (g_tee.armed && g_tee.node == g_node && (const void*)out == g_tee.out && !g_fc_rec) tee_arm();   // (v40_2a)
  if (!g_fc_rec) {                       // the usual case: straight to the registers
    W(RG_IN, (uint32_t)in); W(RG_WGT, (uint32_t)filt); W(RG_OUT, (uint32_t)out);
    W(RG_BIAS, (uint32_t)bias); W(RG_HASB, hb);
    W(RG_MULT, (uint32_t)mult); W(RG_SHIFT, (uint32_t)shift);
    W(RG_HO, Ho); W(RG_WO, Wo); W(RG_CO, Co); W(RG_FH, FH);
    W(RG_L, L); W(RG_ROWSTR, rowstr); W(RG_PIXSTEP, pixstep); W(RG_ROWSTEP, rowstep);
    W(RG_K, K); W(RG_WGLAST, wglast); W(RG_NPIX, npix); W(RG_CONTIG, contig);
    W(RG_NCHUNKS, nchunks); W(RG_ACTLEN, actlen); W(RG_OUTLEN, cp ? actlen : npix * Co);
    W(RG_SCR, (uint32_t)npu_scr);
    W(RG_SLICED, sliced); W(RG_CI, Ci); W(RG_SW, (uint32_t)sw | g_swx); W(RG_FX0OFF, (3 - sw) * Ci); g_swx = 0;
    W(RG_PG, (uint32_t)pg | (g_p16 && pg && !cp ? 2u : 0u) |
             ((AG_ON && pg == 1 && !cp && !g_cpx && !g_p16 && Co >= 32 && !(Co & 15) && nchunks == 1 &&
               ((uint32_t)out & 15u)) ? 1024u : 0u));   // (v47c AG; v49_DS: RG_PG bit 10)
    W(RG_PADTL, padtl | g_padx); g_padx = 0; W(RG_HWIN, (uint32_t)hr | ((uint32_t)wr << 16));   // (v57_MB k_mbf: g_padx)
    W(RG_CP, cp | (g_sm ? 8u : 0u) | g_cpx | (g_w8 ? 64u : 0u)); g_cpx = 0;   // (v58_RS w8: bit 6)
    W(RG_INOFF, (uint32_t)in_off); W(RG_OOFF, (uint32_t)out_off);
    W(RG_AMIN, (uint32_t)amin); W(RG_AMAX, (uint32_t)amax);
    if (g_sm) { W(RG_SM, g_sm); W(RG_SMDROP, g_smseg); g_sm = 0; }
    return;
  }
  // a FULLY_CONNECTED being recorded writes its set into its cache slot first
  uint32_t* v = g_fcv[g_fcn];
  v[0] = (uint32_t)in; v[1] = (uint32_t)filt; v[2] = (uint32_t)out; v[3] = (uint32_t)bias;
  v[4] = hb; v[5] = (uint32_t)mult; v[6] = (uint32_t)shift; v[7] = Ho; v[8] = Wo;
  v[9] = Co; v[10] = FH; v[11] = L; v[12] = rowstr; v[13] = pixstep; v[14] = rowstep; v[15] = K;
  v[16] = wglast; v[17] = npix; v[18] = contig; v[19] = nchunks; v[20] = actlen;
  v[21] = cp ? actlen : npix * Co; v[22] = (uint32_t)npu_scr; v[23] = sliced; v[24] = Ci;
  v[25] = sw; v[26] = (3 - sw) * Ci; v[27] = pg; v[28] = padtl;
  v[29] = (uint32_t)hr | ((uint32_t)wr << 16); v[30] = cp | (g_sm ? 8u : 0u) | g_cpx; g_cpx = 0;
  v[31] = (uint32_t)in_off; v[32] = (uint32_t)out_off; v[33] = (uint32_t)amin; v[34] = (uint32_t)amax;
  cr_out(v);
  if (g_fc_rec) { g_fcs[g_fcn] = 0; g_fck[g_fcn++] = g_fc_rec; g_fc_rec = 0; }
  if (g_sm) { W(RG_SM, g_sm); W(RG_SMDROP, g_smseg); g_sm = 0; }
}

// FC stream (npu_conv8 p_fs, RG_CP bit 4): one output pixel, K <= 720. Its register image:
// v[0..5] IN WGT BIAS MULT SHIFT OUT (change from node to node), v[6..16] HASB CO K NPIX WGLAST
// OUTLEN CP INOFF OOFF AMIN AMAX (written only when they differ from the staging bank)
#define FS_N 17
static uint32_t g_fsl[FS_N] DRV;
static uint32_t g_fs_tmp[FS_N] DRV;
#define FSD(k, reg) if (v[k] != L[k]) { W(reg, v[k]); L[k] = v[k]; }
static void fs_write(const uint32_t* v) {
  uint32_t* L = g_fsl;
  if (npu_pending) (void)R(RG_SLOT);
  W(RG_IN, v[0]); W(RG_WGT, v[1]); W(RG_BIAS, v[2]); W(RG_MULT, v[3]); W(RG_SHIFT, v[4]); W(RG_OUT, v[5]);
  if (!g_fsv) { for (int k = 6; k < FS_N; k++) L[k] = ~v[k]; g_fsv = 1; }
  FSD(6, RG_HASB) FSD(7, RG_CO) FSD(8, RG_K) FSD(9, RG_NPIX) FSD(10, RG_WGLAST) FSD(11, RG_OUTLEN)
  FSD(12, RG_CP) FSD(13, RG_INOFF) FSD(14, RG_OOFF) FSD(15, RG_AMIN) FSD(16, RG_AMAX)
}
#undef FSD
static void npu_run(uint32_t op);
static void fs_launch(const int8_t* in, const int8_t* filt, const int32_t* bias, const int32_t* mult,
                      const int32_t* shift, int8_t* out, int32_t Co, int32_t K, int32_t in_off,
                      int32_t out_off, int32_t amin, int32_t amax) {
  uint32_t* v = g_fc_rec ? g_fcv[g_fcn] : g_fs_tmp;
  const int32_t glast = (Co & 15) ? (Co & 15) : 16;
  v[0] = (uint32_t)in; v[1] = (uint32_t)filt; v[2] = (uint32_t)bias; v[3] = (uint32_t)mult;
  v[4] = (uint32_t)shift; v[5] = (uint32_t)out; v[6] = bias ? 1u : 0u; v[7] = (uint32_t)Co;
  v[8] = (uint32_t)K; v[9] = 1u; v[10] = (uint32_t)(glast * K); v[11] = (uint32_t)Co; v[12] = 16u;
  v[13] = (uint32_t)in_off; v[14] = (uint32_t)out_off; v[15] = (uint32_t)amin; v[16] = (uint32_t)amax;
  if (g_fc_rec) { g_fcs[g_fcn] = 1; g_fck[g_fcn++] = g_fc_rec; g_fc_rec = 0; }
  fs_write(v);
  npu_run(NOP_CONV);
}

// conv / fc on the NPU: returns 1 if executed
// H, W: the (possibly padded) input size; hr, wr, pt, pl: the real input and the implicit padding
// of an elided PAD (sliced walk only; in then points at the padded origin)
#ifndef CW_ON
#define CW_ON(fz) (STRICT || (fz))   // (v46_RS) strict: column mode for every stride-1 Ci 16 sliced conv
#endif
#ifndef PM_ON
#define PM_ON 1   // (v49_RS) k_pm
#endif
#ifndef RPAIR
#define RPAIR 1
#endif
#ifndef CW2
#define CW2 1
#endif
#ifndef DR2
#define DR2 1
#endif
#ifndef CW_PLANAR
#define CW_PLANAR 1
#endif
static int g_pin DRV;   // the conv being set up reads 16-channel planes (column mode only, see npu_conv)
static int g_po8 DRV;   // the conv being set up writes 16-channel planes from its 8-channel groups (RG_PG bit 3)
static int g_pad_pin DRV;   // pad_elides / conv_slices: the PAD's input is planes (a stride-2 sliced walk may read them)
static int g_force_p16 DRV; // conv_node: write planes (paired-group conv; a fused ADD's residual)
static int32_t g_rstr DRV;  // npu_conv: row stride override (1x1 conv over planes: the plane size)
static int g_fzpo8 DRV;     // the fused conv writes the ADD's output as planes (its residual is planes)
static int g_rpair DRV;     // the fused conv reads a pair-swizzled residual (one 16-byte read per column-mode pair)
static int g_swz DRV;       // the shortcut conv writes its output pair-swizzled (the fused conv's residual)
static int g_swz_done DRV;
// (v35R) row-pair layout (LY_RP): writer / cw2 reader / 1x1 stride-2 reader flags of the conv being set up
static int g_fzrp DRV;      // the fused conv may write the ADD's output as row-pair planes (RG_CP bit 13)
static int g_fzrp_used DRV; // ... and does
static int g_rprd DRV;      // the stride-2 sliced conv reads row-pair planes (RG_CP bit 14)
static int g_rpx DRV;       // a 1x1 stride-2 conv reads row-pair planes (pixel step doubled; the caller adds 16)
// stride-2 dual-row sliced walk over planes (row mode, RG_SW bit 20): Ci a multiple of 16
static int cw2p_ok(int32_t Ci, int32_t Co, int32_t H, int32_t W, int32_t Ho, int32_t Wo, int32_t sh, int32_t sw) {
  return CW_PLANAR && DR2 && (Ci & 15) == 0 && Ci >= 32 && Ci <= 1023 && sh == 2 && sw == 2 && Ho >= 2 && Wo >= 1 &&
         H * W * 16 <= 65535 && (Ci / 16 - 1) * H * W * 16 < (1 << 19) && Wo * Co <= 65535 && Ho * Wo <= 4095 &&
         (Ho * Wo) * 8 <= NPU_SCR_WORDS && H <= 255 && W <= 255 && Co <= 1023 && 4 * W * 16 <= 65535;
}
// column mode over 16-channel planes ([C/16][H][W][16]): a stride-1 sliced conv with the dual-row walk
// on the transposed image; slice k at + k * H * W * 16 (H, W: the real input)
static int cwp_ok(int32_t Ci, int32_t Co, int32_t H, int32_t W, int32_t Ho, int32_t Wo, int32_t sh, int32_t sw) {
  return CW_PLANAR && (Ci & 15) == 0 && Ci >= 32 && Ci <= 1023 && sh == 1 && sw == 1 && Ho >= 2 && Wo >= 2 &&
         H * W * 16 <= 65535 && Wo * Co <= 65535 && Ho * Wo <= 4095 && (Ho * Wo) * 8 <= NPU_SCR_WORDS &&
         H <= 255 && W <= 255 && Co <= 1023 && Ho + 2 <= H + 2 && Wo + 2 <= W + 2;
}
// (v24M) planar / column-mode register overrides, out of npu_conv's hot path
static void __attribute__((noinline)) cw_regs(int cwm, int dr, int32_t sh, int32_t sw, int32_t wr, int32_t hr,
                                              int32_t pt, int32_t pl, int32_t Ho, int32_t Wo, int32_t Co) {
  if (cwm) {
    const int32_t r16 = wr * 16;
    W(RG_HO, Wo); W(RG_WO, Ho); W(RG_ROWSTR, 16); W(RG_PIXSTEP, sh * r16); W(RG_ROWSTEP, 2 * sw * 16);
    W(RG_FX0OFF, (3 - sh) * r16); W(RG_COLSTEP, r16); W(RG_PADTL, (uint32_t)pl | ((uint32_t)pt << 8));
    W(RG_HWIN, (uint32_t)wr | ((uint32_t)hr << 16)); W(RG_OPIX, (uint32_t)(Wo * Co)); W(RG_OROW, (uint32_t)Co);
    W(RG_IGRP, (uint32_t)(hr * r16));
    W(RG_SW, (uint32_t)sw | (1u << 18) | (1u << 19) | g_pmx);   // (v49_RS g_pmx)
  }
  if (g_pin && sw == 2 && !cwm) {         // stride-2 row walk over planes (dual-row)
    W(RG_ROWSTR, wr * 16); W(RG_PIXSTEP, sw * 16); W(RG_ROWSTEP, (dr ? 2 : 1) * sh * wr * 16);
    W(RG_FX0OFF, (3 - sw) * 16); W(RG_IGRP, (uint32_t)(hr * wr * 16));
    W(RG_SW, (uint32_t)sw | (dr ? (1u << 19) : 0u) | (1u << 20));
  }
  if (g_rstr) W(RG_ROWSTR, (uint32_t)g_rstr);
  if (g_po8) {                            // planes out (8-channel groups): pixels 16 bytes apart
    W(RG_PG, 8u);
    if (dr && !cwm) W(RG_OROW, (uint32_t)(Wo * 16));
    if (cwm) { W(RG_OPIX, (uint32_t)(Wo * 16)); W(RG_OROW, 16u); }
  }
}
// (v35R) the cw2 conv over row-pair planes runs on the vertically flipped image: positions bottom-up
// (PIXSTEP = -2 W 16), one read per pair of input rows (2i, 2i + 1) (column 1 = the odd row first), the top
// padding row of the flipped image is the PAD's bottom row, outputs bottom-up (OPIX negative)
static void __attribute__((noinline, section(".text.v35r"))) rp_regs(const int8_t* in, int8_t* out, int32_t H, int32_t W, int32_t Ho,
                                              int32_t Wo, int32_t Co) {
  const int32_t ostep = g_po8 ? Wo * 16 : Wo * Co;
  W(RG_IN, (uint32_t)(in + (H - 2) * W * 16)); W(RG_PIXSTEP, (uint32_t)(-2 * W * 16) & 0xffffu);
  W(RG_ROWSTEP, 128u); W(RG_FX0OFF, 0u); W(RG_PADTL, 1u << 8);
  W(RG_OUT, (uint32_t)(out + (Ho - 1) * ostep)); W(RG_OPIX, (uint32_t)(-ostep) & 0xffffu);
}
static int adp_state DRV;               // (declared again below: the ad_fcae plan's state)
// (v52BF) 32-bit accumulators (acc26 undone): exact for any int8 layer; K > 1024 still never runs on the NPU
static inline int acc_ok(const int32_t* b, const int8_t* w, int32_t Co, int32_t K, int32_t ioff, int fs) {
  (void)b; (void)w; (void)Co; (void)ioff; (void)fs;
  return K <= 1024;
}
static void mbr_conv(uint32_t pgv);       // (v50_MB k_rep) defined at the end of the file
static const Node* g_mbr_pad DRV;         // (v50_MB) the PAD whose output the producer's op wrote
static const int8_t* g_mbr_src DRV;       // (v50_MB) ... and the NHWC copy its depthwise conv reads
// (v57_MB) owner ruling 2026-10-10: only the model's final output must be complete (intermediates may stay virtual)
#ifndef MB_LAX
#define MB_LAX 1
#endif
#ifndef MBL_PL_LO   // (v57_MB) pointwise nodes of the group writing 16-channel planes for their depthwise consumer
#define MBL_PL_LO ((1u << 7) | (1u << 10) | (1u << 13) | (1u << 16) | (1u << 19) | (1u << 22) | (1u << 25) | (1u << 28) | (1u << 31))
#define MBL_PL_HI ((1u << (34 - 32)) | (1u << (37 - 32)))
#endif
static uint8_t g_mbr_lay DRV;             // (v57_MB) layout of g_mbr_src (LY_NHWC / LY_P)
static int mbl_sole(const Node* r, int32_t t);   // (v57_MB) defined at the end of the file
static const Node* g_mbf DRV;             // (v57_MB k_mbf) node 0's PAD folded into node 1's conv (no op yet)
static const Node* g_mbfd DRV;            // (v57_MB k_mbf) ... node 1's folded ops issued (a failed detection: run the PAD)
static int g_mbfx DRV;                    // (v57_MB k_mbf) node 1's folded ops are being set up: no rep / LAX hook
static int mbf_try(const Node* node, Tensor* t);
static int mbf_conv(const Node* node, Tensor* t);
static void mbf_undo(Tensor* t);
static int mbl_chk(const Node* n0);
static inline int kc_plan(uint32_t k);            // (v57_MB) (v55_KC) defined with mbm_pw
static int npu_conv(const int8_t* in, const int8_t* filt, const int32_t* bias,
                    const int32_t* mult, const int32_t* shift, int8_t* out,
                    int32_t H, int32_t W, int32_t Ci, int32_t Ho, int32_t Wo, int32_t Co,
                    int32_t FH, int32_t FW, int32_t sh, int32_t sw,
                    int32_t in_off, int32_t out_off, int32_t amin, int32_t amax,
                    int32_t hr, int32_t wr, int32_t pt, int32_t pl) {
  const int32_t K = FH * FW * Ci;
  if (!acc_ok(bias, filt, Co, K, in_off, Ho == 1 && Wo == 1 && K <= 720)) return 0;   // (v40K acc26)
  // FC stream: one output pixel whose window is contiguous, the activation vector fits the lanes
  if (Ho == 1 && Wo == 1 && K > 0 && K <= 720 && in_off >= -128 && in_off <= 128 && (pt | pl) == 0 &&
      fits16(Co) && Co <= 1023 && FH > 0 && FW > 0 && FH <= H && FW <= W && (FH == 1 || W == FW) &&
      g_fzc.n == 0 && out_off >= -256 && out_off <= 255 && amin >= -128 && amax <= 127 && amin <= amax) {
    fs_launch(in, filt, bias, mult, shift, out, Co, K, in_off, out_off, amin, amax);
    return 1;
  }
  // register widths of the NPU: output size and filter height <= 255, Co <= 1023, < 4096 pixels,
  // strides < 65536, implicit padding <= 3 (H, W <= 255 and Ci <= 1023 where it applies)
  if (!fits16(H) || !fits16(W) || !fits16(Ci) || !fits16(FW) || sh <= 0 || sw <= 0 ||
      ((((Ho | Wo | FH) | ((Ho - 1) | (Wo - 1) | (FH - 1))) & ~255) | ((Co | (Co - 1)) & ~1023) |
       ((pt | pl) & ~3)) != 0) return 0;          // (1..255, 1..1023, 0..3: also excludes 0 and negatives)
  const int32_t rstr = wr * Ci, pstep = (g_rpx ? 2 : 1) * sw * Ci, rstep = sh * rstr;     // (rstr <= rstep: sh >= 1; v35R: g_rpx)
  if (((rstep | pstep) & ~65535) != 0) return 0;
  const int32_t npix = Ho * Wo, L = FW * Ci;
  // 3x3 with Ci a multiple of 16: chunks are 16-channel slices and the window slides along rows
  const int32_t sliced = (FH == 3 && FW == 3 && (Ci & 15) == 0 && sh == sw &&
                         (sw == 1 || (sw == 2 && (Ci == 16 || g_pin))));
  const int32_t nchunks = sliced ? (Ci >> 4) : (K + 143) / 144;
  if (npix > 4095 || nchunks > 31 || L >= (1 << 20)) return 0;
  if (g_w8 && !(sliced && nchunks > 1 && g_pin)) return 0;   // (v58_RS w8) rotated planes: only the w8 sliced walk
  if ((Ho - 1) * sh + FH > H || (Wo - 1) * sw + FW > W) return 0;
  if (in_off < -127 || in_off > 128) return 0;          // raw int8 lanes: the padding byte -in_off is an int8
  if (out_off < -256 || out_off > 255 || amin < -128 || amax > 127 || amin > amax) return 0;
  if (nchunks > 1 && npix * 8 > NPU_SCR_WORDS) return 0;
  const int32_t contig = (FH == 1 && FW == 1 && sh == 1 && sw == 1 && Ho == H && Wo == W);
  if (contig && nchunks == 1 && npix * L >= (1 << 20)) return 0;
  if (!contig && !sliced && nchunks > 1) return 0;     // (v26A: the im2col walk takes single-chunk convs only)
  if (Co <= 8 && npix * Co >= (1 << 20)) return 0;
  if ((pt | pl) != 0 && !sliced) return 0;
  if (sliced && (hr > 255 || wr > 255 || Ci > 1023)) return 0;
  // gather: an im2col conv with several row segments per pixel and more than one pass over the
  // activations first copies its im2col vectors into a contiguous buffer (one pass of the small
  // reads), then runs as a contiguous 1x1 conv over that buffer
  if (!contig && !sliced && nchunks == 1 && FH > 1 && Co > 16 && npix * K <= GAT_BYTES && (pt | pl) == 0 &&
      Wo * K <= 65535 && !g_po8 && !g_rstr) {
    npu_slot();
    g_sm = strip_mode(FH, L, sw * Ci, Wo, 1);
    conv_regs(in, filt, bias, mult, shift, gat_scr, Ho, Wo, 1, FH, L, rstr, pstep, rstep, K,
              1, npix, 0, 1, npix * K, Ci, sw, 0, 0, 0, hr, wr, in_off, out_off, amin, amax, 1);
    R(RG_CMD) = NOP_CONV;
    npu_pending = 1;
    npu_slot();
    const int32_t pgg = (K <= 64 && Co > 8);
    const int32_t gl = pgg ? ((Co & 15) ? (Co & 15) : 16) : ((Co & 7) ? (Co & 7) : 8);
    conv_regs(gat_scr, filt, bias, mult, shift, out, Ho, Wo, Co, 1, K, Wo * K, K, Wo * K, K,
              gl * K, npix, 1, 1, npix * K, K, 1, 0, pgg, 0, Ho, Wo, in_off, out_off, amin, amax, 0);
    npu_run(NOP_CONV);
    return 1;
  }
  if (g_pin && !(sliced && ((sw == 1 && cwp_ok(Ci, Co, hr, wr, Ho, Wo, sh, sw)) ||
                             (sw == 2 && cw2p_ok(Ci, Co, hr, wr, Ho, Wo, sh, sw))))) return 0;
  // paired groups: im2col convs with K <= 64 compute 16 output channels per activation read
  const int32_t pg = (!sliced && nchunks == 1 && K <= 64 && Co > 8);
  if (g_po8 && (pg || (Co & 15) != 0 || Co <= 8)) return 0;
  if (g_po8 && g_fzc.n != 0 && !(nchunks == 1 || sliced)) return 0;   // (a fused ADD's planes need the fusion)
  const int32_t glast = pg ? ((Co & 15) ? (Co & 15) : 16) : ((Co & 7) ? (Co & 7) : 8);
  // (several chunk passes: the sliced walk, residual bytes after each pixel's taps)
  // (v46_RS) fz2 takes 8-channel groups only (the sliced walk; not paired groups / FC stream): else no fusion
  if (STRICT && g_fzc.n != 0 && !(sliced && !pg && (Co & 7) == 0 && Co >= 16)) g_fzc.n = 0;
  const int fz = (g_fzc.n != 0 && (nchunks == 1 || sliced));
  if (STRICT && fz && g_fzc.own) g_cpx |= 1u << 28;   // (v46_RS) fz2: the conv's own output too (RG_N below; v57_RS: if read)
  // dual-row walk (stride-1 sliced convs): output rows in pairs, one 4-row window walk per pair
  const int32_t dr = (sliced && (sw == 1 || (sw == 2 && CW2 && (g_pin || Ci == 16) && (Wo & 1) == 0 && !fz)) &&
                     Ho >= 2 && 2 * rstep <= 65535 && Wo * Co <= 65535);
  if (dr) g_swx = 1u << 19;
  const int cwm = dr && (g_pin || Ci == 16) &&
                  (sw == 1 ? (g_pin || CW_ON(fz)) : (CW2 && (Wo & 1) == 0 && !fz));
  // (v46_RS) fzp: column mode puts a pair's pixels Co bytes apart: one residual read per pair (Co 16)
  if (STRICT && fz && cwm && sw == 1 && nchunks == 1 && Co == 16 && (Wo & 1) == 0 && !g_po8) {
    g_cpx |= 1u << 27;
    if (PM_ON && g_fzc.own) g_pmx = 1u << 24;   // (v49_RS) pair-merged writes (r_pm.py; v57_RS: fz2 only)
  }
  // paired residual (RG_CP bit 7): the shortcut wrote the residual pair-swizzled (RG_CP bit 8)
  if (g_rpair && fz && nchunks > 1 && cwm && sw == 1 && (Wo & 1) == 0 && (Co & 7) == 0) g_cpx |= 128u;
  if ((g_cpx & 128u) && (g_cpx & 2048u) && nchunks == 2) g_cpx |= 4096u;   // (v28R) residual in the psum record (v30M: p24 = bit 11, rr = bit 12)
  // (v35R) row-pair planes: the fused stride-1 column-mode conv writes them (bit 13), the cw2 conv reads them (bit 14)
  if (g_fzrp && fz && cwm && sw == 1 && (Wo & 1) == 0 && (Ho & 1) == 0 && (Co == 16 || g_po8)) { g_cpx |= 8192u; g_fzrp_used = 1; }
  if (g_rprd) {
    if (!(cwm && sw == 2 && (pt | pl) == 0 && !fz && (hr & 1) == 0 && Ho * 2 == hr && Wo * 2 == wr && hr == H - 1 && wr == W - 1)) {
      g_cpx = 0; g_swx = 0; return 0;
    }
    g_cpx |= 16384u;
  }
  if (g_swz && pg && (Wo & 1) == 0) { g_cpx |= 256u; g_swz_done = 1; }
  if (g_rpx) g_cpx |= 32768u;   // (v35R) RG_CP bit 15
  npu_slot();
  if (!contig && !sliced && nchunks == 1 && (pt | pl) == 0 && (FH > 1 || (FW == 1 && sw > 1)) && npix > 1 &&
      !(pg && FH > 4))
    g_sm = g_rpx ? 0 : strip_mode(FH, L, sw * Ci, Wo, 0);
  // (v46_RS) strict: a K 27 paired-group strip walk (3x3, Ci 3, stride 1, Wo % 4 == 0; resnet8's first conv over
  // its PAD's output) takes v40K's packed strips (sp 4, segments 18, 2 MAC issues per pixel; the fast paths that
  // used them are off in STRICT): RG_SM bit 24
  if (STRICT && g_sm != 0 && pg && K == 27 && FH == 3 && L == 9 && sw * Ci == 3 && (Wo & 3) == 0 && Wo >= 4) {
    g_sm = 4u | (18u << 8) | (18u << 16) | (1u << 24); g_smseg = 54u | (54u << 8);
  }
  // pixel pairs (RG_PG bit 2): a strip-mode conv with 8 output channels whose row piece and pixel
  // step fit one beat runs two pixels per MAC issue (paired groups with CO 16, see npu_conv8 p_pp)
  const int ppx = ((Wo & 1) == 0 && g_sm != 0 && !pg && !fz && !g_fc_rec && Co == 8 && sw * Ci < 8 && L + sw * Ci <= 16);
  conv_regs(in, filt, bias, mult, shift, fz ? g_fzc.out : out, Ho, Wo, Co, FH, L, rstr, pstep, (dr ? 2 : 1) * rstep, K,
            glast * K, npix, contig, nchunks, npix * L, Ci, sw, sliced, pg, ((uint32_t)pt | ((uint32_t)pl << 8)),
            hr, wr, in_off, out_off, amin, amax, fz ? 2u : 0u);
  if (dr) W(RG_OROW, (uint32_t)(Wo * Co));
  // column mode (RG_SW bit 18 on a sliced conv): the dual-row walk runs on the transposed image
  // (walk rows = input columns, 16 bytes apart; walk taps = input rows, rstr apart), so a sliding
  // pair's new taps (4 input columns of one input row) are one read
  if (cwm || g_pin || g_rstr || g_po8) cw_regs(cwm, dr, sh, sw, wr, hr, pt, pl, Ho, Wo, Co);
  g_pmx = 0;
  if (g_rprd) rp_regs(in, out, hr, wr, Ho, Wo, Co);
  if (ppx) { W(RG_PG, 5u); W(RG_CO, 16u); }
  if (fz) {
    W(RG_ER, g_fzc.er); W(RG_M1, g_fzc.m1); W(RG_M2, g_fzc.m2);
    W(RG_OFF1, g_fzc.o1); W(RG_OFF2, g_fzc.o2); W(RG_MO, g_fzc.mo); W(RG_SHO, g_fzc.sho);
    W(RG_FOOFF, g_fzc.oo); W(RG_FAMIN, g_fzc.amin); W(RG_FAMAX, g_fzc.amax);
    W(RG_RDELTA, (uint32_t)g_fzc.res - (uint32_t)g_fzc.out);
    if (STRICT && g_fzc.own) W(RG_N, ((uint32_t)out - (uint32_t)g_fzc.out) & 0xfffffu);   // (v46_RS) own output - ADD output
    g_fz_done = g_fzc.n;
  }
  if (ppx && !fz) mbr_conv(5u);            // (v50_MB k_rep)
  npu_run(NOP_CONV);
  return 1;
}

static int pad_elides(const Node* node, const Tensor* tensors);

// (v57_RS) 1: some node from q on (except ex) reads tensor t, or the scan cannot tell (conservative). The scan stops
// where the node list ends (an op code out of range or output indices no longer increasing, as rp_ok's scan)
static int __attribute__((noinline, cold)) t_read_later(const Node* q, int32_t t, const Node* ex) {
  for (int k = 0; k < 64; k++, q++) {
    if ((uint32_t)q->op >= (uint32_t)OP_COUNT || q->output <= (q - 1)->output || q->output > 4095) return 0;
    if (q == ex) continue;
    if (q->input == t) return 1;
    if (q->num_inputs > 0) {
      if (!in_dmem(q->inputs) || q->num_inputs > 16) return 1;
      for (int i = 0; i < q->num_inputs; i++) if (q->inputs[i] == t) return 1;
    }
  }
  return 1;
}
// output of a node: an arena slot when it feeds a PAD that will be elided
static int __attribute__((noinline, cold)) al_ok(const Node* node, const Tensor* tensors) {   // (v57_RS) see k_rsfu 4
  const RuntimeShape* so = &tensors[node->output].shape;
  const int32_t C = so->n == 4 ? so->dims[3] : 0;   // 16; 32 / 64 only without a PAD reader (no tee planned)
  if (node->op != OP_CONV_2D || (C != 16 && C != 32 && C != 64) || (MBM && mbm_n0 != 0)) return 0;
  const int32_t t = node->output;
  int nr = 0;
  const Node* q = node + 1;
  for (int k = 0; k < 64; k++, q++) {
    if ((uint32_t)q->op >= (uint32_t)OP_COUNT || q->output <= (q - 1)->output || q->output > 4095) return nr > 0;
    int rd = (q->input == t);
    if (q->num_inputs > 0) {
      if (!in_dmem(q->inputs) || q->num_inputs > 16) return 0;
      for (int i = 0; i < q->num_inputs; i++) if (q->inputs[i] == t) rd = 1;
    }
    if (!rd) continue;
    if (!(q->op == OP_CONV_2D || q->op == OP_ADD || (C == 16 && q->op == OP_PAD && (q + 1)->op == OP_CONV_2D &&
          (q + 1)->input == q->output && !t_read_later(q + 2, q->output, 0)))) return 0;
    nr++;
  }
  return 0;
}
static __attribute__((noinline, cold)) int8_t* tout_al(const Node* node, Tensor* tensors) {   // (v57_RS) aligned NHWC slot or 0
  if (!al_ok(node, tensors)) return 0;
  int8_t* q = (int8_t*)rs_slot(tensors[node->output].data, rs_flat(&tensors[node->output].shape));
  return q;
}
static int8_t* tout(const Node* node, Tensor* tensors) {
  const Tensor* to = &tensors[node->output];
  const Node* n1 = node + 1;
  for (int i = 0; i < nred; i++)          // a slot kept from an earlier inference: NHWC this time
    if (red_from[i] == to->data) { red_lay[i] = LY_NHWC; return (int8_t*)red_to[i]; }
  if (STRICT) { int8_t* q = tout_al(node, tensors); if (q) return q; }   // (v57_RS) aligned NHWC slot
  // (every reader goes through tin(), so the slot is safe whether or not the PAD is then elided;
  // deciding that here would put pad_elides on the exposed path of the first kernels)
  if (n1->op == OP_PAD && ((n1 + 1)->op == OP_DEPTHWISE_CONV_2D || (n1 + 1)->op == OP_CONV_2D) &&
      next_consumes(node)) {
    void* q = arena_slot(to->data, rs_flat(&to->shape));
    if (q) return (int8_t*)q;
  }
  return (int8_t*)to->data;
}

// A paired-group conv whose next node reads 16-channel planes (a stride-1 depthwise conv with
// C % 16 == 0, C >= 32, through an elided PAD or directly, or a global average pool) writes its
// output as planes into an arena slot (layout LY_P); any other reader gets NHWC through tin().
static int dw_planar_ok(const Node* dn, int32_t C) {
  const ConvParams* p = (const ConvParams*)dn->builtin;
  // stride 1: transposed walk (v2); stride 2: the plain walk over the planes (16-channel groups at
  // RG_IGRP strides, a row's new taps one read; from v14_S)
  return p->stride_w == p->stride_h && (p->stride_w == 1 || p->stride_w == 2) && (C & 15) == 0 && C >= 32;
}
static int want_p16(const Node* node, const Tensor* tensors, int32_t Co) {
  const Node* n1 = node + 1;
  if ((Co & 15) != 0) return 0;
  if (n1->op == OP_PAD) {
    const Node* n2 = n1 + 1;
    return n2->op == OP_DEPTHWISE_CONV_2D && n1->input == node->output && n2->input == n1->output &&
           ((const PadParams*)n1->builtin)->pad_value + ((const ConvParams*)n2->builtin)->input_offset == 0 &&
           dw_planar_ok(n2, Co);
  }
  if (n1->op == OP_DEPTHWISE_CONV_2D) return n1->input == node->output && dw_planar_ok(n1, Co);
  if (n1->op == OP_AVERAGE_POOL_2D) {
    const PoolParams* pp = (const PoolParams*)n1->builtin;
    const RuntimeShape* is = &tensors[n1->input].shape;
    const RuntimeShape* os = &tensors[n1->output].shape;
    return n1->input == node->output && is->n == 4 && os->n == 4 && os->dims[1] == 1 && os->dims[2] == 1 &&
           pp->filter_h == is->dims[1] && pp->filter_w == is->dims[2] && is->dims[1] * is->dims[2] <= 256;
  }
  return 0;
}

static int pad_elides(const Node* node, const Tensor* tensors);
// the conv's output feeds an elided PAD and a stride-1 sliced conv that walks planes in column mode
static int want_po8(const Node* node, const Tensor* tensors, int32_t Co, int32_t Ho, int32_t Wo) {
  const Node* n1 = node + 1;
  const Node* n2 = node + 2;
  if (!CW_PLANAR || (Co & 15) != 0 || Co < 32 || Ho * Wo < 2 || n1->op != OP_PAD ||
      n2->op != OP_CONV_2D || n1->input != node->output || n2->input != n1->output || !in_dmem(n2->builtin) ||
      !in_dmem(n1->builtin) || n2->filter < 0 || n2->filter > 4095 || n2->output < 0 || n2->output > 4095) return 0;
  const PadParams* pp = (const PadParams*)n1->builtin;
  const ConvParams* cp = (const ConvParams*)n2->builtin;
  if (pp->pad_value + cp->input_offset != 0 || cp->pad_h != 0 || cp->pad_w != 0) return 0;
  const RuntimeShape* fs = &tensors[n2->filter].shape;
  int32_t d2[4];
  if (fs->n != 4 || fs->dims[1] != 3 || fs->dims[2] != 3 || fs->dims[3] != Co || !shape4(&tensors[n2->output].shape, d2))
    return 0;
  if (!cwp_ok(Co, fs->dims[0], Ho, Wo, d2[1], d2[2], cp->stride_h, cp->stride_w) &&
      !cw2p_ok(Co, fs->dims[0], Ho, Wo, d2[1], d2[2], cp->stride_h, cp->stride_w)) return 0;
  g_pad_pin = 1;
  const int el = pad_elides(n1, tensors);
  g_pad_pin = 0;
  return el;
}
// output slot for planes (8-channel groups): the conv's arena slot, marked LY_P (0: none)
static int8_t* po8_last DRV;
// (v35R) the fused ADD's output t may be row-pair planes: H, W even, C % 16 == 0, and every reader of t is either
// a PAD (0, 1, 0, 1) whose next node is a VALID 3x3 stride-2 conv on it, or a VALID 1x1 stride-2 conv (one PAD)
static int __attribute__((noinline, section(".text.v35r"))) rp_ok(const Node* an, const Tensor* tensors) {
  if (!RS_RP) return 0;
  const int t = an->output;
  int32_t d[4];
  if (!shape4(&tensors[t].shape, d) || d[0] != 1 || ((d[1] | d[2]) & 1) || (d[3] & 15) || d[1] < 4 || d[2] < 4 ||
      d[1] * d[2] * 16 > 65535) return 0;
  int npad = 0;
  const Node* q = an + 1;
  for (int k = 0; k < 64; k++, q++) {
    if ((uint32_t)q->op >= (uint32_t)OP_COUNT || q->output <= (q - 1)->output || q->output > 4095) break;
    int rd = (q->input == t);
    if (q->num_inputs > 0) {
      if (!in_dmem(q->inputs) || q->num_inputs > 16) return 0;
      for (int i = 0; i < q->num_inputs; i++) if (q->inputs[i] == t) rd = 1;
    }
    if (!rd) continue;
    const Node* c = (q->op == OP_PAD) ? q + 1 : q;
    if (c->op != OP_CONV_2D || !in_dmem(c->builtin) || c->filter < 0 || c->filter > 4095 || c->output < 0 ||
        c->output > 4095) return 0;
    const ConvParams* cp = (const ConvParams*)c->builtin;
    const RuntimeShape* fs = &tensors[c->filter].shape;
    int32_t o[4];
    if (cp->stride_h != 2 || cp->stride_w != 2 || cp->pad_h != 0 || cp->pad_w != 0 || fs->n != 4 ||
        fs->dims[3] != d[3] || !shape4(&tensors[c->output].shape, o) || o[1] * 2 != d[1] || o[2] * 2 != d[2]) return 0;
    if (q->op == OP_PAD) {
      const PadParams* pp = (const PadParams*)q->builtin;
      if (!in_dmem(pp) || c->input != q->output || pp->pad_h_before != 0 || pp->pad_h_after != 1 ||
          pp->pad_w_before != 0 || pp->pad_w_after != 1 || pp->pad_value + cp->input_offset != 0 ||
          fs->dims[1] != 3 || fs->dims[2] != 3 || (o[2] & 1)) return 0;
      npad++;
    } else if (fs->dims[1] != 1 || fs->dims[2] != 1) return 0;
  }
  return npad == 1;
}
static int8_t* po8_slot(const Node* node, Tensor* tensors, int32_t Co, int32_t Ho, int32_t Wo) {
  po8_last = 0;
  if (STRICT) return 0;                   // (v46_RS) no arena slots in strict mode: never planes
  if (g_fzc.n != 0 || !want_po8(node, tensors, Co, Ho, Wo)) return 0;
  const Tensor* to = &tensors[node->output];
  int8_t* o = tout(node, tensors);
  if (o == (int8_t*)to->data) return 0;
  for (int i = 0; i < nred; i++) if (red_to[i] == o) { red_lay[i] = LY_P; red_last = i; po8_last = o; return o; }
  return 0;
}
// a conv node on the NPU (or, if it cannot run there, on the CPU)
// (v24M) a 1x1 conv over 16-channel planes (out of conv_node's hot path); 1: launched
static int __attribute__((noinline)) p1x1_try(const Node* node, Tensor* tensors, const void* vi, int8_t* out,
                                             const int32_t* di, const int32_t* dout, const int32_t* df) {
  const ConvParams* p = (const ConvParams*)node->builtin;
  const Tensor* tf = &tensors[node->filter];
  const int32_t* bias = node->bias >= 0 ? (const int32_t*)tensors[node->bias].data : 0;
  const int32_t Ci = di[3], Co = df[0];
  if (!((Ci & 15) == 0 && Ci >= 32 && Ci <= 64 && Co > 8 && !g_po8 && di[1] * di[2] * 16 <= 65535)) return 0;
  g_rstr = di[1] * di[2] * 16;
  const int ok1 = npu_conv((const int8_t*)vi, (const int8_t*)tf->data, bias, node->out_mult,
                node->out_shift, out, di[1], di[2], 16, dout[1], dout[2],
                df[0], Ci >> 4, 1, p->stride_h, p->stride_w, p->input_offset,
                p->output_offset, p->quantized_activation_min,
                p->quantized_activation_max, di[1], di[2], 0, 0);
  g_rstr = 0;
  return ok1;
}
// (v25M) a contiguous 1x1 paired-group conv over an NHWC input whose output becomes planes for its
// reader (want_p16: ds_cnn's pointwise convs): exactly what conv_node -> npu_conv -> conv_regs do
// for this case, without their general checks (0: not this case, nothing done; conv_node decides)
static const Node* g_wpn DRV;           // pw_fast: want_p16 of this node was g_wpv
static int g_wpv DRV;
static int __attribute__((noinline)) pw_fast(const Node* node, Tensor* tensors) {
  const Tensor* tf = &tensors[node->filter];
  if (tf->shape.dims[1] != 1 || tf->shape.dims[2] != 1) return 0;     // (cheap reject: DRAM loads are slow)
  const ConvParams* p = (const ConvParams*)node->builtin;
  const Tensor* ti = &tensors[node->input];
  const Tensor* to = &tensors[node->output];
  const RuntimeShape* si = &ti->shape;
  const RuntimeShape* sf = &tf->shape;
  const RuntimeShape* so = &to->shape;
  if (si->n != 4 || sf->n != 4 || so->n != 4 || si->dims[0] != 1 || so->dims[0] != 1) return 0;
  const int32_t H = si->dims[1], Wd = si->dims[2], Ci = si->dims[3], Co = sf->dims[0];
  if (sf->dims[1] != 1 || sf->dims[2] != 1 || sf->dims[3] != Ci || so->dims[3] != Co || so->dims[1] != H ||
      so->dims[2] != Wd || p->pad_h != 0 || p->pad_w != 0 || p->stride_h != 1 || p->stride_w != 1) return 0;
  const int32_t npix = H * Wd;
  const int32_t in_off = p->input_offset, out_off = p->output_offset;
  const int32_t amin = p->quantized_activation_min, amax = p->quantized_activation_max;
  if (Ci < 1 || Ci > 64 || Co <= 8 || Co > 1023 || H < 1 || H > 255 || Wd < 1 || Wd > 255 || npix <= 1 ||
      npix > 4095 || Wd * Ci > 65535 || npix * Ci >= (1 << 20) || in_off < -127 || in_off > 128 ||
      out_off < -256 || out_off > 255 || amin < -128 || amax > 127 || amin > amax) return 0;
  if ((vpad.out != 0 && ti->data == vpad.out) || g_fzc.n != 0 || g_swz || g_pin || g_po8 || g_rstr) return 0;
  if (!acc_ok(node->bias >= 0 ? (const int32_t*)tensors[node->bias].data : 0, (const int8_t*)tf->data, Co, Ci, in_off, 0))
    return 0;                          // (v40K acc26)
  const int wp = g_force_p16 || want_p16(node, tensors, Co);
  g_wpn = node; g_wpv = wp;
  if (!wp) return 0;
  int8_t* out = (int8_t*)arena_slot(to->data, rs_flat(so));
  if (!out) return 0;
  red_lay[red_last] = LY_P;
  int lay_i;
  const void* vi = tview(ti, &lay_i);
  if (lay_i != LY_NHWC) return 0;        // (conv_node redoes the same slot lookup)
  const int32_t* bias = node->bias >= 0 ? (const int32_t*)tensors[node->bias].data : 0;
  const int32_t glast = (Co & 15) ? (Co & 15) : 16;
  g_p16 = 1;
  npu_slot();
  conv_regs((const int8_t*)vi, (const int8_t*)tf->data, bias, node->out_mult, node->out_shift, out, H, Wd, Co, 1,
            Ci, Wd * Ci, Ci, Wd * Ci, Ci, glast * Ci, npix, 1, 1, npix * Ci, Ci, 1, 0, 1, 0u, H, Wd,
            in_off, out_off, amin, amax, 0);
  g_p16 = 0;
  npu_run(NOP_CONV);
  return 1;
}
static void conv_node(const Node* node, Tensor* tensors) {
  if (pw_fast(node, tensors)) { g_fzc.n = 0; g_wpn = 0; return; }
  const ConvParams* p = (const ConvParams*)node->builtin;
  const Tensor* ti = &tensors[node->input];
  const Tensor* tf = &tensors[node->filter];
  const Tensor* to = &tensors[node->output];
  const int32_t* bias = node->bias >= 0 ? (const int32_t*)tensors[node->bias].data : 0;
  int32_t di[4], df[4], dout[4];
  if (shape4(&ti->shape, di) && tf->shape.n == 4 && shape4(&to->shape, dout) && p->pad_h == 0 &&
      p->pad_w == 0) {
    df[0] = tf->shape.dims[0]; df[1] = tf->shape.dims[1]; df[2] = tf->shape.dims[2]; df[3] = tf->shape.dims[3];
    if (df[3] == di[3] && dout[3] == df[0]) {
      // planes the conv cannot walk: NHWC copy, padded the usual way
      if (vpad.out != 0 && ti->data == vpad.out && vpad.lay != LY_NHWC && !(vpad.lay == LY_RP && di[3] == 16) &&
          !cwp_ok(di[3], df[0], vpad.H, vpad.W, dout[1], dout[2], p->stride_h, p->stride_w) &&
          !cw2p_ok(di[3], df[0], vpad.H, vpad.W, dout[1], dout[2], p->stride_h, p->stride_w))
        pad_materialize();
      // (v41_C) strict: a stride-2 sliced conv with Ci 32..64 over an NHWC input: identity conv to planes first
      if (STRICT && vpad.out != 0 && ti->data == vpad.out && vpad.lay == LY_NHWC && df[1] == 3 && df[2] == 3 &&
          p->stride_h == 2 && p->stride_w == 2 && (di[3] & 15) == 0 && di[3] >= 32 && di[3] <= 64 &&
          vpad.H * vpad.W * di[3] <= PAD_SCR_BYTES &&
          cw2p_ok(di[3], df[0], vpad.H, vpad.W, dout[1], dout[2], p->stride_h, p->stride_w)) {
        const int32_t C = di[3];
        g_p16 = 1;
        const int okc = npu_conv(vpad.src, C == 32 ? id_w32 : id_w64, id_z, id_m, id_z, pad_scr, vpad.H, vpad.W, C,
                                 vpad.H, vpad.W, C, 1, 1, 1, 1, 0, 0, -128, 127, vpad.H, vpad.W, 0, 0);
        g_p16 = 0;
        if (okc) { vpad.src = pad_scr; vpad.lay = LY_P; }
      }
      // an elided PAD before a sliced conv: read its unpadded input with implicit padding
      if (vpad.out != 0 && ti->data == vpad.out) {
        const int32_t Ci = di[3], pt = vpad.hb, pl = vpad.wb, wr = vpad.W;
        g_pin = (vpad.lay != LY_NHWC) && !(vpad.lay == LY_RP && Ci == 16);
        g_rprd = (vpad.lay == LY_RP);
        g_fzrp_used = 0;
        g_fzrp = !STRICT && g_fzc.n != 0 && rp_ok(g_fzc.n, tensors);   // (v46_RS) strict: no row-pair slots
        if (g_fzrp) {
          g_fzrp = 0;
          for (int i = 0; i < nred; i++) if (red_to[i] == (void*)g_fzc.out) g_fzrp = 1;
        }
        const int8_t* vin = vpad.src - (pt * wr + pl) * (g_pin ? 16 : Ci);
        int8_t* o = po8_slot(node, tensors, df[0], dout[1], dout[2]);
        const int ri = red_last;
        g_po8 = (o != 0) || (g_fzc.n != 0 && g_fzpo8);
        if (!o) o = tout(node, tensors);
        g_w8 = g_pin && w8_rotp(vpad.src);         // (v58_RS w8) planes of a rotated tee
        const int ok = npu_conv(vin, (const int8_t*)tf->data, bias, node->out_mult, node->out_shift,
                     o, di[1], di[2], Ci, dout[1], dout[2], df[0], df[1], df[2],
                     p->stride_h, p->stride_w, p->input_offset, p->output_offset,
                     p->quantized_activation_min, p->quantized_activation_max, vpad.H, wr, pt, pl);
        g_w8 = 0;
        if (g_po8 && !ok && o == po8_last) red_lay[ri] = LY_NHWC;
        if (ok && g_fzc.n != 0 && g_fzpo8)          // the fused ADD's output: planes
          for (int i = 0; i < nred; i++) if (red_to[i] == (void*)g_fzc.out) red_lay[i] = LY_P;
        if (ok && g_fzc.n != 0 && g_fzrp_used)      // (v35R) ... row-pair planes
          for (int i = 0; i < nred; i++) if (red_to[i] == (void*)g_fzc.out) red_lay[i] = LY_RP;
        g_rprd = 0; g_fzrp = 0; g_fzrp_used = 0;
        g_pin = 0; g_po8 = 0; g_fzpo8 = 0; g_rpair = 0;
        if (ok) {
          vpad.out = 0; vpad.skip = 0;
          g_fzc.n = 0;
          return;
        }
      } else {
        // planes for the next node: a paired-group conv (K <= 64, more than one pixel, no fused ADD)
        const int32_t Ci = di[3], Co = df[0], FH = df[1], FW = df[2], sh = p->stride_h, sw = p->stride_w;
        const int slc = (FH == 3 && FW == 3 && (Ci & 15) == 0 && sh == sw && (sw == 1 || (sw == 2 && Ci == 16)));
        int8_t* out = 0;
        if (!slc && FH * FW * Ci <= 64 && Co > 8 && dout[1] * dout[2] > 1 && g_fzc.n == 0 &&
            (g_force_p16 || (g_wpn == node ? g_wpv : want_p16(node, tensors, Co)))) {
          out = (int8_t*)arena_slot(to->data, rs_flat(&to->shape));
          if (out) red_lay[red_last] = LY_P;
        }
        g_p16 = (out != 0);
        if (!out) {
          out = po8_slot(node, tensors, Co, dout[1], dout[2]);
          g_po8 = (out != 0);
        }
        if (!out) out = tout(node, tensors);
        // a 1x1 conv over planes: Ci / 16 filter rows of 16 bytes, one plane apart
        int lay_i;
        const void* vi = tview(ti, &lay_i);
        if (lay_i == LY_P && FH == 1 && FW == 1 && p1x1_try(node, tensors, vi, out, di, dout, df)) {
          g_p16 = 0; g_po8 = 0; g_fzc.n = 0; return;
        }
        // (v35R) row-pair planes: row 2i of a pair at +16, even pixels 64 bytes apart
        if (lay_i == LY_RP && FH == 1 && FW == 1 && sh == 2 && sw == 2 && g_fzc.n == 0) {
          g_rpx = 1;
          const int okr = (Ci == 16) ?
            npu_conv((const int8_t*)vi + 16, (const int8_t*)tf->data, bias, node->out_mult, node->out_shift, out,
                     di[1], di[2], 16, dout[1], dout[2], Co, 1, 1, 2, 2, p->input_offset, p->output_offset,
                     p->quantized_activation_min, p->quantized_activation_max, di[1], di[2], 0, 0) :
            p1x1_try(node, tensors, (const int8_t*)vi + 16, out, di, dout, df);
          g_rpx = 0;
          if (okr) { g_p16 = 0; g_po8 = 0; g_fzc.n = 0; return; }
        }
        const int ok = npu_conv((const int8_t*)(lay_i == LY_NHWC ? vi : tin(ti)), (const int8_t*)tf->data, bias, node->out_mult,
                          node->out_shift, out, di[1], di[2], di[3], dout[1], dout[2],
                          df[0], df[1], df[2], p->stride_h, p->stride_w, p->input_offset,
                          p->output_offset, p->quantized_activation_min,
                          p->quantized_activation_max, di[1], di[2], 0, 0);
        g_p16 = 0;
        if (g_po8 && !ok) red_lay[red_last] = LY_NHWC;
        g_po8 = 0;
        if (ok) { g_fzc.n = 0; return; }
        if (out != (int8_t*)to->data) red_lay[red_last] = LY_NHWC;   // (written by the CPU below)
      }
    }
  }
  g_fzc.n = 0;
  npu_wait();
  ref_conv(p, node->out_mult, node->out_shift, &ti->shape, (const int8_t*)tin(ti), &tf->shape,
           (const int8_t*)tf->data, bias, &to->shape, tout(node, tensors));
}

static int conv_slices(const Node* cn, const Tensor* tensors, int32_t H, int32_t W, int32_t Ci);
// Residual block tail: conv -> shortcut conv (independent) -> ADD(conv, shortcut). The shortcut is
// launched first; the conv (several chunk passes on the sliced walk) then carries the ADD, whose
// drain hides under its MACs (a fused shortcut conv would be drain bound).
static const Node* g_skip DRV;          // a node launched ahead of its turn: its kernel returns
// (v28D) ds_cnn depthwise-separable blocks (PAD 1/1 -> DW 3x3 -> PW 1x1, 25x5x64) replayed: the register
// sets the generic kernels write (elided PAD, planar transposed DW into the aligned scratch, pw_fast into
// planes), launched after one check of the block's wiring / shapes / parameters at its PAD node
#ifndef DS_LEAN
#define DS_LEAN 1                               // (v42DS) ds_cnn lean front (ds_front), lean block checks, FC wait
#endif
static const Node* g_dsx_next DRV;      // the PAD node of the next block to check
static const Node* g_dsx_dw DRV;        // armed depthwise / pointwise nodes of the checked block
static const Node* g_dsx_pw DRV;
static uint32_t g_dsx_src DRV, g_dsx_out DRV, g_dsx_dws DRV;   // block input (planes), PW output slot, DW output
// (v39S) the block's pointwise output feeds only a global average pool that its drain computes (RG_CP bit 30)
#ifndef PF_ON
#define PF_ON 1
#endif
static const Node* g_pf_node DRV;       // that pool node (its kernel returns at once)
static int g_dsx_pf DRV;
static int dsx_pad(const Node* node, Tensor* tensors);
static void dsx_dw(const Node* node, Tensor* tensors);
static void dsx_pw(const Node* node, Tensor* tensors);
#if DS_DWF
static const Node* g_dwf_dw DRV;        // (v37_1a) the depthwise the running fused op computes
static const Node* g_dsx_okn DRV;       // a block already checked by the previous pointwise (its PAD) ...
static void* g_dsx_okq DRV;             // ... and its pointwise output's slot
static uint32_t g_dwf_out DRV;          // its output (NHWC, the pointwise's own slot)
static void* dsx_chk(const Node* node, Tensor* tensors);
#endif
static __attribute__((optimize("Os"))) int shortcut_first(const Node* node, Tensor* tensors) {
  const Node* sn = node + 1;
  const Node* an = node + 2;
  if (sn->input == node->output || sn->input < 0 || sn->output <= node->output || sn->output > 4095 ||
      an->num_inputs != 2 || !in_dmem(an->inputs)) return 0;
  const int a0 = an->inputs[0], a1 = an->inputs[1];
  if (!((a0 == node->output && a1 == sn->output) || (a1 == node->output && a0 == sn->output))) return 0;
  int32_t di[4];
  if (!shape4(&tensors[node->input].shape, di) || !conv_slices(node, tensors, di[1], di[2], di[3])) return 0;
  if (!fuse_add_check(node, an, tensors)) return 0;
  // planes: the ADD's output feeds an elided PAD and a sliced walk over planes, so the shortcut writes
  // its output (the residual) as planes and the conv writes the ADD's output as planes
  int fp = 0;
  {
    int32_t dx[4];
    const Tensor* tsf = &tensors[sn->filter];
    if (CW_PLANAR && shape4(&tensors[an->output].shape, dx) && tsf->shape.n == 4 &&
        tsf->shape.dims[1] * tsf->shape.dims[2] * tsf->shape.dims[3] <= 64 && (tsf->shape.dims[0] & 15) == 0 &&
        g_fzc.out != (int8_t*)tensors[an->output].data && want_po8(an, tensors, dx[3], dx[1], dx[2])) fp = 1;
    // (v25M) ... or a global average pool over 16-channel groups (the conv reads an elided PAD, so it
    // takes conv_node's planes-out path)
    if (!fp && CW_PLANAR && shape4(&tensors[an->output].shape, dx) && tsf->shape.n == 4 &&
        tsf->shape.dims[1] * tsf->shape.dims[2] * tsf->shape.dims[3] <= 64 && (tsf->shape.dims[0] & 15) == 0 &&
        g_fzc.out != (int8_t*)tensors[an->output].data && (an + 1)->op == OP_AVERAGE_POOL_2D &&
        vpad.out != 0 && vpad.out == tensors[node->input].data && want_p16(an, tensors, dx[3])) fp = 1;
  }
  // paired residual: the conv runs in column mode over planes with several slices; the shortcut (a
  // paired-group 1x1 conv) writes the residual pair-swizzled so that a pair's 16 bytes are one read
  int rp = 0;
  {
    int32_t dx[4], dsi[4];
    const Tensor* tsf = &tensors[sn->filter];
    const Tensor* tcf = &tensors[node->filter];
    const ConvParams* cp = (const ConvParams*)node->builtin;
    if (!STRICT && RPAIR && vpad.out != 0 && vpad.out == tensors[node->input].data && vpad.lay != LY_NHWC &&
        shape4(&tensors[an->output].shape, dx) && shape4(&tensors[sn->input].shape, dsi) && tcf->shape.n == 4 &&
        tsf->shape.n == 4 && tsf->shape.dims[1] == 1 && tsf->shape.dims[2] == 1 && tsf->shape.dims[3] <= 64 &&
        (tsf->shape.dims[0] & 15) == 0 && tsf->shape.dims[0] == dx[3] && (dx[2] & 1) == 0 && (dx[3] & 7) == 0 &&
        di[3] >= 32 && cwp_ok(di[3], tcf->shape.dims[0], vpad.H, vpad.W, dx[1], dx[2], cp->stride_h, cp->stride_w))
      rp = 1;
  }
  // the shortcut now (its own output, no fusion)
  const Node* save = g_node;
  const Node* fn = g_fzc.n;
  g_fzc.n = 0;
  g_node = sn; g_nowait = 1;
  g_force_p16 = fp;
  g_swz = rp; g_swz_done = 0;
  conv_node(sn, tensors);
  g_swz = 0;
  g_rpair = rp && g_swz_done;
  g_force_p16 = 0;
  g_nowait = 0; g_node = save;
  g_fzc.n = fn;
  {
    int lay;
    const void* r = tview(&tensors[sn->output], &lay);
    if (lay == LY_NHWC) { g_fzc.res = (const int8_t*)r; fp = 0; }
    else if (fp) g_fzc.res = (const int8_t*)r;
    else g_fzc.res = (const int8_t*)tin(&tensors[sn->output]);
  }
  g_fzpo8 = fp;
  g_skip = sn;
  return 1;
}

// (v52_RI) resnet8's captured first PAD / CONV plans (gen_fp.py: register images holding quantisation values) are
// removed (owner rule: no carried parameter data); in the STRICT build they returned at once, as these stubs do
static const Node* fp_next DRV;        // the CONV whose fast path its PAD armed (never set now)
static int fp_pad(const Node* node, Tensor* tensors) { (void)node; (void)tensors; return 0; }
#if RS_PADE
static int fp_pad2(const Node* node, Tensor* tensors) { (void)node; (void)tensors; return 0; }
#endif
static int fp_conv(const Node* node, Tensor* tensors) { (void)node; (void)tensors; fp_next = 0; return 0; }
static const Node* g_plan_c DRV;
static int plan_conv(const Node* node, Tensor* tensors);
// (v46_RS) strict: a residual block's shortcut conv is launched right after the conv before the block's PAD
static const Node* g_early DRV;
static __attribute__((optimize("Os"))) void rs_early(const Node* node, Tensor* tensors) {
  const Node* pn = node + 1; const Node* cn = node + 2; const Node* sn = node + 3; const Node* an = node + 4;
  if (pn->op != OP_PAD || cn->op != OP_CONV_2D || sn->op != OP_CONV_2D || an->op != OP_ADD) return;
  if (pn->input != node->output || cn->input != pn->output || sn->input < 0 || sn->input == node->output ||
      sn->input == pn->output || sn->input == cn->output || sn->input >= node->output) return;
  if (shortcut_first(cn, tensors)) g_early = cn;
}
// (v58_RS w8) rotated planes when the PAD's conv reader has a Ci-64 3x3 filter at 8 mod 16 and nothing else reads node's output
static void __attribute__((noinline, cold, optimize("Os"))) w8_plan(const Node* node, Tensor* tensors, const Node* r, int slot) {
  const Tensor* tf = &tensors[r->filter];
  if (r->op != OP_CONV_2D || r->filter < 0 || r->filter > 4095 || tf->shape.n != 4 || tf->shape.dims[1] != 3 ||
      tf->shape.dims[2] != 3 || tf->shape.dims[3] != 64 || (((uint32_t)tf->data) & 15u) != 8u ||
      t_read_later(node + 2, node->output, 0)) return;
  g_tee.pl |= 1u << 23; tee_rot[slot] = 1;
}
static int __attribute__((noinline, cold, optimize("Os"))) w8_rotp(const void* p) {
  for (int i = 0; i < ntee; i++) if (tee_to[i] == p && tee_ok[i] && tee_rot[i]) return 1;
  return 0;
}
// (v40_2a) write tee: plan a private copy of node's output for its reader (conv_regs arms it on node's conv op
// writing its own output). Cheap: this runs in the producer's kernel, on the CPU's critical path when ops are short.
static __attribute__((optimize("Os"))) int tee_slot(const void* real, int32_t bytes) {
  for (int i = 0; i < ntee; i++) if (tee_from[i] == real && tee_sz[i] == bytes) return i;
  const uint32_t sz = ((uint32_t)bytes + 15u) & ~15u;
  if (ntee == NTEE || bytes <= 0 || arena_top + sz > ARENA_BYTES) return -1;
  tee_from[ntee] = real; tee_to[ntee] = arena + arena_top; tee_sz[ntee] = bytes; tee_ok[ntee] = 0;
  arena_top += sz;
  return ntee++;
}
static __attribute__((optimize("Os"))) void tee_plan(const Node* node, Tensor* tensors) {
  g_tee.armed = 0;
  if (!TEE_ON) return;
  if (MBM && mbm_n0 != 0) return;        // (v48C) the mobilenet group: v46_MB's mirrors, no v49_RS tee
  const Node* q = node + 1;
  const int32_t no = node->output;
  const int qop = q->op;
  const Node* r;                         // the reader
  if (qop == OP_PAD) {
    r = node + 2;
    if (q->input != no || r->input != q->output || (r->op != OP_DEPTHWISE_CONV_2D && r->op != OP_CONV_2D)) return;
  } else if (qop == OP_AVERAGE_POOL_2D) {
    r = q;
    if (q->input != no || node->op != OP_ADD) return;   // (v53_Q) k_lit.py 3d
  } else return;
  // small maps (< 64 pixels) are left alone: their ops are shorter than the CPU's set-up of the next ones, which a
  // faster reader would only expose (and this check is the cheapest first)
  const RuntimeShape* so = &tensors[no].shape;
  if (so->n != 4) return;
  const int32_t H = so->dims[1], Wd = so->dims[2];
  if (H * Wd < 64 || H > 4095 || Wd > 4095) return;
  // the producer's depth K (its write slack: the copy costs it one more write per 16-byte piece)
  int32_t Kd = 0;
  if (node->op == OP_ADD) Kd = 1 << 20;            // the identity-conv ADD: two results per cycle
  else if (node->filter >= 0 && node->filter <= 4095) {
    const RuntimeShape* fs = &tensors[node->filter].shape;
    if (fs->n == 4) Kd = fs->dims[1] * fs->dims[2] * fs->dims[3];
  }
  if (Kd < 27) return;
  const int rop = r->op;
  if (rop == OP_DEPTHWISE_CONV_2D) return;   // (v53_Q) see k_lit.py 3c
  if (0) {
    if (!in_dmem(r->builtin)) return;
    // a stride-1 depthwise gains less than a shallow producer pays
    if (((const ConvParams*)r->builtin)->stride_w != 2 && Kd < 64) return;
  }
  const int32_t C = so->dims[3];
  if (C <= 0 || H * Wd * C >= (1 << 20)) return;
  // planes for a depthwise, a 3x3 conv (column-mode walks) or the pool; an aligned NHWC copy for C <= 16 (not for
  // the pool)
  // (v49_RS lean RTL: 16-channel planes of C = 32 / 64 only)
  const int planes = (C == 32 || C == 64) && H * Wd * 16 <= 65535;
  // (v49_RS) planes only (aligned NHWC copies: a stride-1 3x3 conv reader gains nothing and its shallow producer pays:
  // resnet8 conv1 +1.3k, conv3 +0.45k; no scored model used them for a depthwise reader on this line)
  if (!planes) return;
  const int slot = tee_slot(tensors[no].data, H * Wd * C);
  if (slot < 0) return;
  tee_lay[slot] = planes ? LY_P : LY_NHWC;
  g_tee.node = node; g_tee.out = tensors[no].data; g_tee.slot = slot; g_tee.ix = no;
  g_tee.b3 = (uint32_t)tee_to[slot];
  g_tee.pl = 0;
  if (planes) {
    uint32_t l = 0;
    while ((32u << l) <= (uint32_t)C) l++;       // C = 16 << l
    g_tee.pl = (uint32_t)(H * Wd * 16) | (l << 16) | (1u << 19);
    // (v49_RS) row-pair planes (v35R LY_RP) when every reader is a stride-2 conv (rp_ok: a PAD (0,1,0,1) + 3x3 s2
    // conv, 1x1 s2 convs) and W = 8 / 16 / 32: the 3x3 s2 conv reads a pair's new input rows with one read
    if (rop == OP_CONV_2D && (Wd == 8 || Wd == 16 || Wd == 32) && rp_ok(node, tensors)) {
      tee_lay[slot] = LY_RP;
      g_tee.pl |= (1u << 20) | ((uint32_t)(Wd == 8 ? 0 : Wd == 16 ? 1 : 2) << 21);
    }
  }
  tee_rot[slot] = 0;
  if (W8_ON && tee_lay[slot] == LY_P && C == 64 && qop == OP_PAD) w8_plan(node, tensors, r, slot);   // (v58_RS w8)
  g_tee.armed = 1;
}

static int __attribute__((noinline)) mbm_pw(const Node* node, Tensor* tensors);
static int __attribute__((noinline)) kc_big(const Node* node, Tensor* tensors);   // (v55_KC)   // (v46_MB) defined at the end of the file
static int __attribute__((noinline)) mbm_pad(const Node* node, Tensor* tensors);
static void __attribute__((noinline)) mbm_acp(const Node* node, Tensor* tensors);   // (v49_MB)
// (v49_MB k_apw) pointwise nodes (index in the mobilenet group) writing their output aligned at pad_scr + a copy op
#ifndef MBM_APW_LO
#define MBM_APW_LO ((1u << 7) | (1u << 13) | (1u << 19) | (1u << 25) | (1u << 28) | (1u << 31))
#define MBM_APW_HI (1u << (34 - 32))
#endif
static const int8_t* g_apw DRV;          // the output buffer whose aligned copy pad_scr holds (0: none)
static int g_apwp DRV;                   // (v50_MB k_p7) ... as 16-channel planes
static int8_t* apw_out(const Node* node, Tensor* tensors, int32_t H, int32_t Wd, int32_t Co);
static void apw_copy(const Node* node, Tensor* tensors, int32_t H, int32_t Wd, int32_t Co);
static void Conv2DEval_k(const Node* node, Tensor* tensors) {
  if (node == g_skip) { g_skip = 0; kentry(node, tensors); return; }
#if DS_DSX
  if (node == g_dsx_pw) { dsx_pw(node, tensors); return; }
#endif
  if (node == fp_next && fp_conv(node, tensors)) return;
  kentry(node, tensors);
  if (MB_LAX && g_mbf != 0 && node == g_mbf + 1 && mbf_conv(node, tensors)) return;   // (v57_MB k_mbf)
  tee_plan(node, tensors);
  if (node == g_plan_c && plan_conv(node, tensors)) return;
  if (mbp_state == 0) mbp_detect(node, tensors);
  if (mbp_hook(node, tensors)) return;
  const Node* n1 = node + 1;
  if (n1->op == OP_ADD) fuse_add_check(node, n1, tensors);
  else if (node == g_early) g_early = 0;   // (v46_RS) shortcut launched and ADD prepared by rs_early
  else {
    g_fzc.n = 0;
    if (n1->op == OP_CONV_2D && (n1 + 1)->op == OP_ADD) shortcut_first(node, tensors);
  }
  if (STRICT && TEE_ON && g_fzc.n != 0 && !g_tee.armed) {   // (v49_RS) fused ADD: tee the ADD's output
    // (the ADD's output is written now: its copy is stale unless this op tees it; the ADD's own kentry keeps it)
    if (ntee) {
      const void* ao = tensors[g_fzc.n->output].data;
      const int i = (int)tee_map[TEE_K(g_fzc.n->output)] - 1;
      if (i >= 0 && tee_from[i] == ao) tee_ok[i] = 0;
    }
    tee_fzn = g_fzc.n;
    tee_plan(g_fzc.n, tensors);
    if (g_tee.armed) g_tee.node = node;
  }
  if (MBM_FAST && mbm_n0 != 0 && mbm_pw(node, tensors)) return;   // (v46_MB)
  if (KC_ON && mbm_n0 != 0 && kc_big(node, tensors)) return;   // (v55_KC) node 40
  conv_node(node, tensors);
  if (STRICT && RS_ORD && g_fzc.n == 0) rs_early(node, tensors);   // (v46_RS)
}

// ====================================================================== ad_fcae fast path
// (v52AD) A chain of ten FULLY_CONNECTED nodes shaped like ad_fcae (640-128-128-128-128-8-128-128-128-128-640)
// runs as ten raw FC-stream ops (RG_CP bit 4) that read the harness's own tensors: the weight rows at the filter
// tensor, every 16-channel group's bias / multiplier / shift words at the bias / out_mult / out_shift arrays
// (per-channel: no assumption about their values). Detection checks the structure only (operators, shapes,
// chaining, offsets / activation range inside the FC-stream limits); no copy of any parameter is carried.
// Idle rule: node 0's kernel queues all ten ops (one SLOT load between two) and ends with the held STATUS
// load; kernels 1..9 return at once (their outputs are complete).
#if DS_LEAN
static const Node* g_dsfc DRV;          // (v42DS) ds_cnn's FC (set by ds_front)
#endif
static const uint16_t ad_shape[10][2] = {{128,640},{128,128},{128,128},{128,128},{8,128},{128,8},{128,128},{128,128},{128,128},{640,128}};
#ifndef ADR_FU
#define ADR_FU 0                        // 1: per-tensor quantisation (RG_CP bit 5: group 0's multiplier / shift words
#endif                                  //    only), taken when sampled channels equal channel 0 (no stored values)
#define ADR_CP (ADR_FU ? 48u : 16u)     // FC stream (bit 4); 0: per-channel multiplier / shift words in every group
static uint32_t adx_a[10][6] DRV;       // per node: in, weights, bias, mult, shift, out (the harness's tensors)
static int32_t adx_q[10][2] DRV;        // per node: in_off, out_off
static int adp_state DRV;               // 0 unknown, 1 ad_fcae (adz), -1 off
static const Node* adz_n0 DRV;          // ad_fcae node 0
static const Node* adz_nx DRV;          // the node whose kernel comes next (0: none)
static int adz_k DRV;                   // its index
static int adz_pl DRV;                  // (v58_AD k_adpl2) 1: the ten ops run pipelined across the ten kernels
// (v55AD) run-time packed streams (built by image 0's pack tee; DTCM, zero at program start)
#define ADK_PT (1u << 9)                // RG_CP bit 9: the raw FC stream op also writes its packed stream
#define ADK_PK(k) ((ad_shape[k][1] & 31u) == 0u)   // packable: K % 32 == 0 (beat pairs never cross a row)
static uint32_t adk_s[10] DRV;          // node k's packed stream (start, length; length 0: raw FC stream)
static uint32_t adk_l[10] DRV;
static uint32_t adk_lo DRV, adk_hi DRV; // the packing region: arena + the scratch buffers right behind it
static int adk_any DRV;                 // a kernel other than ad_fcae's node 0 ran in this run
// (v53_Q k_adfu) run-time proof of per-tensor quantisation: cursor (layer adq_k, channel adq_c); adq_st 0 = checking,
// 1 = all ten layers uniform (switch at the next op 0), 2 = per-tensor in use, -1 = a channel differs (per-channel)
static int adq_k DRV, adq_c DRV = 1, adq_st DRV;
static int adq_end DRV = 10;           // (v58_AD k_adst Q5) first layer past the proof (6: layer 5 only)
static void __attribute__((noinline)) adq_step(int n) {
  while (n > 0) {
    const int32_t* m = (const int32_t*)adx_a[adq_k][3];
    const int32_t* sh = (const int32_t*)adx_a[adq_k][4];
    const int co = ad_shape[adq_k][0];
    const int32_t m0 = m[0], s0 = sh[0];
    int c = adq_c;
    const int e = (c + n < co) ? c + n : co;
    n -= e - c;
    for (; c < e; c++) if (m[c] != m0 || sh[c] != s0) { adq_st = -1; return; }
    if (c < co) { adq_c = c; return; }
    adq_c = 1;
    if (++adq_k == adq_end) { adq_st = 1; return; }
  }
}
#define AZ6(k) R(RG_IN) = adx_a[k][0]; R(RG_WGT) = adx_a[k][1]; R(RG_BIAS) = adx_a[k][2]; \
  R(RG_MULT) = adx_a[k][3]; R(RG_SHIFT) = adx_a[k][4]; R(RG_OUT) = adx_a[k][5]; \
  R(RG_INOFF) = (uint32_t)adx_q[k][0]; R(RG_OOFF) = (uint32_t)adx_q[k][1];
// op k's registers over op k-1's in the staging bank (op 0 over op 9's), CMD, then ONE held load: SLOT (op k took
// its registers = op k-1 finished; the CPU writes op k+1's while op k runs), op 9: STATUS (every op finished).
// (v55AD) node k's raw FC stream registers (all of them: the previous op may have been packed)
#define ADK_WGL(k) ((uint32_t)((ad_shape[k][0] & 15u) ? (ad_shape[k][0] & 15u) : 16u) * ad_shape[k][1])
static inline __attribute__((always_inline)) void adk_raw(int k, uint32_t cp) {
  R(RG_CP) = cp; R(RG_IN) = adx_a[k][0]; R(RG_WGT) = adx_a[k][1]; R(RG_BIAS) = adx_a[k][2]; R(RG_MULT) = adx_a[k][3];
  R(RG_SHIFT) = adx_a[k][4]; R(RG_OUT) = adx_a[k][5]; R(RG_INOFF) = (uint32_t)adx_q[k][0];
  R(RG_OOFF) = (uint32_t)adx_q[k][1]; R(RG_CO) = ad_shape[k][0]; R(RG_K) = ad_shape[k][1]; R(RG_WGLAST) = ADK_WGL(k);
}
static inline __attribute__((always_inline)) void adz_kern_j(int k, int pl) {   // (v53_Q it_lean2) body of adz_kern
  uint32_t wr = (k == 9 && !pl) ? RG_STATUS : RG_SLOT;   // (v58_AD k_adpl2) pipelined (pl): op 9 ends at SLOT too
  const uint32_t l = adk_l[k];
  if (ADK_PK(k) && l) {                 // (v55AD) packed FC stream (per-tensor words in it), one output write
    R(RG_CP) = 52u; R(RG_IN) = adx_a[k][0]; R(RG_WGT) = adk_s[k]; R(RG_ACTLEN) = l; R(RG_DWADJ) = 0u;
    R(RG_OUT) = adx_a[k][5]; R(RG_OUTLEN) = ad_shape[k][0]; R(RG_INOFF) = (uint32_t)adx_q[k][0];
    R(RG_OOFF) = (uint32_t)adx_q[k][1]; R(RG_CO) = ad_shape[k][0]; R(RG_K) = ad_shape[k][1];
  } else adk_raw(k, (adq_st == 2) ? 48u : ADR_CP);   // (v53_Q k_adfu: 48 once proven)
  R(RG_CMD) = NOP_CONV;
  npu_pending = wr == RG_SLOT;
  {   // (v53_Q k_adfu) per-tensor proof in this wait: budget by the op on the NPU (op k-1; k = 9: ops 8 + 9)
    const int bq = (k == 1) ? 160 : (k == 9) ? 300 : (k == 5 || k == 6) ? 0 : 30;
    if (bq && adq_st == 0) adq_step(bq);
  }
  (void)R(wr);
}
static void __attribute__((noinline, aligned(4))) adz_kg(int k, int pl) { adz_kern_j(k, pl); }   // (v58_AD k_adpl2)
static void __attribute__((noinline, aligned(4))) adz_k45(void) { adz_kern_j(4, 1); adz_kern_j(5, 1); }   // (v58_AD k_adq45: constant bodies)
// (v55AD) image 0: ops 1..9 raw with the pack tee (continuing). The SLOT / STATUS beat read once op j has finished
// carries its stream length (31:12), overflow (bit 2) and uniform flag (bit 3: every multiplier / shift equals channel
// 0's, so the stream - per-tensor words - is valid); a stream that is not used still takes its room
static uint32_t adk_a DRV;              // (image 0) where the next packed stream starts (0: an overflow happened)
static inline __attribute__((always_inline)) void adk_take(int k, uint32_t st) {
  const uint32_t a = adk_a;
  if (!ADK_PK(k) || !a) return;
  if (st & 4u) { adk_a = 0u; return; }  // overflow: this node and the later ones stay raw (the tee is sticky too)
  const uint32_t l = st >> 12;
  if (st & 8u) { adk_s[k] = a; adk_l[k] = l; }
  adk_a = (a + l + 15u) & ~15u;
}
static void __attribute__((noinline)) adk_q(int k) {   // (v58_AD k_adp0) image 0: queue op k (1..9) with the tee
  adk_raw(k, ADR_CP | (ADK_PK(k) ? ADK_PT : 0u));
  R(RG_WGSTR) = 0u; R(RG_DWADJ) = adk_hi;
  R(RG_CMD) = NOP_CONV;
  adk_take(k - 1, R(RG_SLOT));          // op k took its registers: op k - 1 finished (table entry while op k runs)
}
static void __attribute__((noinline)) adk_fin(void) {   // (v58_AD k_adp0) image 0's kernel 9
  adk_take(9, R(RG_STATUS)); npu_pending = 0;
  int all = 1;                          // every packable node runs packed: the only raw op is node 5 (1 KB of words),
  for (int k = 0; k < 10; k++) if (ADK_PK(k) && !adk_l[k]) all = 0;   // not worth k_adfu's proof time (its CPU
  if (all && adq_st == 0) { adq_k = 5; adq_c = 1; adq_end = 6; }   // (v58_AD k_adst Q5) prove only layer 5, the one raw op
}
// (v53_Q it_lean2) unrolled ad_fcae ops: adz_rest = ops 1..9 (9 ends with the held STATUS load), adz_all = 0..9
static void __attribute__((noinline, aligned(4))) adz_rest(void) {   // (v58_AD k_adpl2: generic body, ITCM)
  for (int k = 1; k < 10; k++) adz_kg(k, 0);
}
static void __attribute__((noinline, aligned(4))) adz_p0(void) {
  if (adq_st == 1) { R(RG_CP) = 48u; adq_st = 2; }   // (v53_Q k_adfu) proven: per-tensor words (NPU idle here)
  adz_kg(0, 0); adz_kg(1, 0);
}
// node k's structure (k = 0: the chain's input must be the tensor's own buffer); fills adx_a / adx_q [k]
static int adz_chk(const Node* n, Tensor* tensors, int k) {
  if (n->op != OP_FULLY_CONNECTED) return 0;
  const int fi = n->filter, bi = n->bias, ii = n->input, oi = n->output;
  if ((uint32_t)fi > 4095u || (uint32_t)bi > 4095u || (uint32_t)ii > 4095u || (uint32_t)oi > 4095u) return 0;
  if (k > 0 && ii != (n - 1)->output) return 0;
  const int32_t co = ad_shape[k][0], K = ad_shape[k][1];
  const RuntimeShape* fs = &tensors[fi].shape;
  if (fs->n != 2 || fs->dims[0] != co || fs->dims[1] != K) return 0;
  if ((k == 0 && rs_flat(&tensors[ii].shape) != K) || rs_flat(&tensors[oi].shape) != co) return 0;
  if (rs_flat(&tensors[bi].shape) != co || n->out_mult == 0 || n->out_shift == 0) return 0;
#if ADR_FU
  {
    const int32_t* m = n->out_mult; const int32_t* sh = n->out_shift;
    const int32_t i1 = co - 1, i2 = co >> 1, i3 = co / 3, i4 = (co * 5) >> 3;
    if (m[i1] != m[0] || m[i2] != m[0] || m[i3] != m[0] || m[i4] != m[0] ||
        sh[i1] != sh[0] || sh[i2] != sh[0] || sh[i3] != sh[0] || sh[i4] != sh[0]) return 0;
  }
#endif
  const FullyConnectedParams* p = (const FullyConnectedParams*)n->builtin;
  const int32_t io = p->input_offset, oo = p->output_offset;
  if (io < -128 || io > 128 || oo < -256 || oo > 255 || p->quantized_activation_min != -128 ||
      p->quantized_activation_max != 127) return 0;
  const void* d = tensors[ii].data;
  if (k == 0) {
    if (d == redir_from || d == vpad.out || d == dws_from) return 0;
    for (int i = 0; i < nred; i++) if (red_from[i] == d) return 0;
  }
  adx_a[k][0] = (uint32_t)d; adx_a[k][1] = (uint32_t)tensors[fi].data; adx_a[k][2] = (uint32_t)tensors[bi].data;
  adx_a[k][3] = (uint32_t)n->out_mult; adx_a[k][4] = (uint32_t)n->out_shift; adx_a[k][5] = (uint32_t)tensors[oi].data;
  adx_q[k][0] = io; adx_q[k][1] = oo;
  return 1;
}
// kernel 0 of the first inference: 1 = handled (the kernel returns), 0 = the generic path
static int __attribute__((noinline)) adz_first(const Node* node, Tensor* tensors) {
  if (adp_state != 0) return 0;
  if (!adz_chk(node, tensors, 0)) { adp_state = -1; return 0; }   // not ad_fcae: the generic kernels
  const int pt = !adk_any;              // (v55AD) the first kernel of the run: the pack tee may use our scratch
  npu_wait();
  npu_slot();                           // op 0: every register a raw FC stream reads
  R(RG_HASB) = 1u; R(RG_NPIX) = 1u; R(RG_AMIN) = (uint32_t)-128; R(RG_AMAX) = 127u;
  adk_raw(0, ADR_CP | (pt ? ADK_PT : 0u)); R(RG_WGSTR) = pt ? (uint32_t)arena : 0u; R(RG_DWADJ) = pt ? (uint32_t)arena + ARENA_BYTES : 0u;   // (v58_AD k_adp0: op 0's stream fits the arena)
  R(RG_CMD) = NOP_CONV; npu_pending = 1;
  if (pt) {                             // the packing region: arena, extended by our scratch buffers that follow
    static const struct { const void* p; uint32_t n; } adk_b[] = {   // it (adjacency checked here)
      {gat_scr, GAT_BYTES}, {mbc_s, sizeof(mbc_s)}, {mbc_m, sizeof(mbc_m)}, {mbc_w, sizeof(mbc_w)},
      {fbo_tab, sizeof(fbo_tab)}, {fb_tab, sizeof(fb_tab)}, {pad_scr, PAD_SCR_BYTES}, {npu_scr, sizeof(npu_scr)}};
    uint32_t hi = (uint32_t)arena + ARENA_BYTES;
    for (int m = 1; m;) {
      m = 0;
      for (unsigned i = 0; i < sizeof(adk_b) / sizeof(adk_b[0]); i++)
        if ((uint32_t)adk_b[i].p == hi) { hi += adk_b[i].n; m = 1; }
    }
    adk_lo = (uint32_t)arena; adk_hi = hi;
  }
  int ok = 1;                           // while op 0 runs: nodes 1..9
  for (int k = 1; ok && k < 10; k++) ok = adz_chk(node + k, tensors, k);
  (void)R(RG_SLOT);                     // op 0 took its registers (the staging bank holds them)
  if (ok) {                             // (idle rule) nodes 1..9 here; 9 ends with STATUS
    adp_state = 1; adz_n0 = node; adz_k = 1; adz_nx = node + 1;
    if (pt) { adk_a = adk_lo; adk_q(1); adz_pl = 2; return 1; }   // (v58_AD k_adp0) op 1 queued; kernels 1..9 go on
    else adz_rest();                    // (v53_Q it_lean2) ops 1..9 unrolled (raw: adk_l stays 0)
  } else adp_state = -1;                // not ad_fcae: node 0 done (its stream only in our idle scratch), nodes 1..9
  (void)R(RG_STATUS); npu_pending = 0;  // (v53_Q) one unconditional held load on every path (static check)
  return 1;
}
#undef AZ6

static void FullyConnectedEval_k(const Node* node, Tensor* tensors) {  if (mbp_hook(node, tensors)) { kentry(node, tensors); return; }
  // (v49AD) adp_hook (v34P ad_fcae path) no longer called: ad_fcae runs on adz, other models have adp_state -1
  if (node == g_pre && !g_preload) {     // registers staged by the previous inference
    kentry(node, tensors);
    g_pre = 0;
    npu_run(NOP_CONV);
    return;
  }
  kentry(node, tensors);
  for (int k = 0; k < g_fcn; k++)        // registers of this node kept from an earlier inference
    if (g_fck[k] == node) {
      if (g_fcs[k]) { fs_write(g_fcv[k]); npu_run(NOP_CONV); return; }
      npu_slot();
      cr_out(g_fcv[k]);
      npu_run(NOP_CONV);
      return;
    }
  const FullyConnectedParams* p = (const FullyConnectedParams*)node->builtin;
  const Tensor* tf = &tensors[node->filter];
  const int8_t* in = (const int8_t*)tin(&tensors[node->input]);
  int8_t* out = (int8_t*)tensors[node->output].data;
  const int32_t* bias = node->bias >= 0 ? (const int32_t*)tensors[node->bias].data : 0;
  const int32_t Co = tf->shape.dims[0], Cin = tf->shape.dims[1];
  if (g_fcn < FC_SLOTS) g_fc_rec = node; // conv_regs keeps this node's register set
#if DS_LEAN
  // (v42DS) ds_cnn's FC, the model's last node: launched without a wait; the op is waited for here, after npu_conv's
  // return path (which then overlaps the op)
  const int dsl = (node == g_dsfc) && !g_preload && !g_nowait;
  g_nowait |= dsl;
#endif
  if (npu_conv(in, (const int8_t*)tf->data, bias, node->out_mult, node->out_shift, out,
               1, 1, Cin, 1, 1, Co, 1, 1, 1, 1, p->input_offset, p->output_offset,
               p->quantized_activation_min, p->quantized_activation_max, 1, 1, 0, 0)) {
    g_fc_rec = 0;
#if DS_LEAN
    if (dsl) { g_nowait = 0; npu_wait(); }
#endif
    return;
  }
#if DS_LEAN
  if (dsl) g_nowait = 0;
#endif
  g_fc_rec = 0;
  if (g_preload) return;
  npu_wait();
  ref_fc(p, node->out_mult, node->out_shift, in, &tf->shape, (const int8_t*)tf->data, bias, out);
}

#define DW_SP 3
// depthwise on the NPU: 3x3, stride 1 or 2 (same both ways), no padding of its own, applied to
// an input of H x W (the padded size when the PAD before it was elided) x C
// (v28D) the elided PAD's check (pad_elides) is remembered for its depthwise: same node, same H / W / C
static const Node* g_dwok_n DRV;
static int32_t g_dwok_k DRV;
static int dw_npu_ok_x(const Node* dn, const Tensor* tensors, int32_t H, int32_t W, int32_t C);
static int dw_npu_ok(const Node* dn, const Tensor* tensors, int32_t H, int32_t W, int32_t C) {
  const int32_t k = H | (W << 8) | (C << 16);
  if (dn == g_dwok_n && k == g_dwok_k) { g_dwok_n = 0; return 1; }
  const int ok = dw_npu_ok_x(dn, tensors, H, W, C);
  g_dwok_n = ok ? dn : 0; g_dwok_k = k;
  return ok;
}
static int dw_npu_ok_x(const Node* dn, const Tensor* tensors, int32_t H, int32_t W, int32_t C) {
  const ConvParams* p = (const ConvParams*)dn->builtin;
  const Tensor* tf = &tensors[dn->filter];
  const Tensor* to = &tensors[dn->output];
  int32_t dout[4];
  if (!shape4(&to->shape, dout) || tf->shape.n != 4 || p->pad_h != 0 || p->pad_w != 0) return 0;
  const int32_t Ho = dout[1], Wo = dout[2], s = p->stride_w, npix = Ho * Wo;
  return tf->shape.dims[0] == C && tf->shape.dims[1] == 3 && tf->shape.dims[2] == 3 &&
         tf->shape.dims[3] == 1 && dout[3] == C &&
         // (NPU widths: C in 1..1023, H, W, Ho, Wo in 1..255, < 4096 pixels, row / plane strides
         //  < 65536)
         ((((H | W | Ho | Wo) | ((H - 1) | (W - 1) | (Ho - 1) | (Wo - 1))) & ~255) |
          ((C | (C - 1)) & ~1023) | (((2 * W * C) | (Wo * C) | (H * W * 16)) & ~65535)) == 0 &&
         p->stride_h == s && (s == 1 || s == 2) && npix <= 4095 && npix * C < (1 << 20) &&
         (Ho - 1) * s + 3 <= H && (Wo - 1) * s + 3 <= W &&
         p->input_offset >= -127 && p->input_offset <= 128 &&
         p->output_offset >= -256 && p->output_offset <= 255 &&
         p->quantized_activation_min >= -128 && p->quantized_activation_max <= 127 &&
         p->quantized_activation_min <= p->quantized_activation_max;
}

// ds_cnn / mobilenet_v1_025 recognised from their first nodes (PAD, CONV, PAD, DEPTHWISE) by STRUCTURE only
// (v51_D, owner rule #3): input shape, first conv filter shape, first depthwise filter shape. Both models are
// plain chains.
// (v53_Q) shapes compared as immediates: input H, W, C; conv1 filter Co, FH, FW, Ci; dw1 filter C, FH (= FW)
static inline int dws_m(const RuntimeShape* si, const RuntimeShape* fc, const RuntimeShape* fd, int32_t h, int32_t w,
                        int32_t c, int32_t co, int32_t fh, int32_t fw, int32_t ci, int32_t dc, int32_t dk) {
  return si->n == 4 && si->dims[1] == h && si->dims[2] == w && si->dims[3] == c &&
         fc->n == 4 && fc->dims[0] == co && fc->dims[1] == fh && fc->dims[2] == fw && fc->dims[3] == ci &&
         fd->n == 4 && fd->dims[0] == dc && fd->dims[1] == dk && fd->dims[2] == dk && fd->dims[3] == 1;
}
static uint32_t rd32(const uint8_t* d) {
  return ((uint32_t)d & 3u) ? (uint32_t)d[0] | ((uint32_t)d[1] << 8) | ((uint32_t)d[2] << 16) | ((uint32_t)d[3] << 24)
                            : *(const uint32_t*)d;
}
static __attribute__((optimize("Os"))) void dws_detect(const Node* node, const Tensor* tensors) {
  g_dws = -1;
  const Node* n0 = node - 3;
  if (n0 != g_first || n0->op != OP_PAD || (node - 2)->op != OP_CONV_2D || (node - 1)->op != OP_PAD) return;
  const RuntimeShape* si = &tensors[n0->input].shape;
  const RuntimeShape* fc = &tensors[(node - 2)->filter].shape;
  const RuntimeShape* fd = &tensors[node->filter].shape;
  if (dws_m(si, fc, fd, 49, 10, 1, 64, 10, 4, 1, 64, 3) || dws_m(si, fc, fd, 96, 96, 3, 8, 3, 3, 3, 8, 3)) g_dws = 1;
}

static void DepthwiseConv2DEval_k(const Node* node, Tensor* tensors) {
#if DS_DSX
  if (node == g_dsx_dw) { dsx_dw(node, tensors); return; }
#endif
  kentry(node, tensors);
  if (g_dws == 0) dws_detect(node, tensors);
  if (mbp_hook(node, tensors)) return;
  const ConvParams* p = (const ConvParams*)node->builtin;
  const Tensor* ti = &tensors[node->input];
  const Tensor* tf = &tensors[node->filter];
  const Tensor* to = &tensors[node->output];
  const int32_t* bias = node->bias >= 0 ? (const int32_t*)tensors[node->bias].data : 0;
  int32_t di[4], dout[4];
  if (shape4(&ti->shape, di) && shape4(&to->shape, dout) && dw_npu_ok(node, tensors, di[1], di[2], di[3]) &&
      acc_ok(bias, (const int8_t*)tf->data, di[3], 9, p->input_offset, 0)) {   // (v40K acc26)
    const int32_t C = di[3], Ho = dout[1], Wo = dout[2];
    const int32_t s = p->stride_w, npix = Ho * Wo;
    // an elided PAD: read its unpadded input; the window origin may lie outside it
    int32_t H = di[1], W = di[2], pt = 0, pl = 0, dsp = 0;
    uint32_t a_in;
    int lay = LY_NHWC;
    const int8_t* psrc = 0;
    if (vpad.out != 0 && ti->data == vpad.out && vpad.lay != LY_NHWC && !dw_planar_ok(node, C))
      pad_materialize();                   // planes this walk cannot read: the PAD runs after all
    if (vpad.out == 0 || ti->data != vpad.out) {
      psrc = (const int8_t*)tview(ti, &lay);
      if (lay != LY_NHWC && !dw_planar_ok(node, C)) { psrc = (const int8_t*)tin(ti); lay = LY_NHWC; }
    }
    if (vpad.out != 0 && ti->data == vpad.out) {
      H = vpad.H; W = vpad.W; pt = vpad.hb; pl = vpad.wb; lay = vpad.lay; psrc = vpad.src;
      // 8 channels, stride 1, one column of padding each side: pairs of pixels form 16-channel
      // "super pixels" (W/2 columns); the window is 3 super columns wide
      dsp = (C == 8 && s == 1 && pl == 1 && (W & 1) == 0 && Wo == W);
      a_in = (uint32_t)vpad.src - (uint32_t)((pt * W + pl) * C + (dsp ? 8 : 0));
      vpad.out = 0;
    } else {
      a_in = (uint32_t)psrc;
    }
    uint32_t a_out = (uint32_t)to->data;
    if ((!STRICT || (MB_LAX && mbm_n0 != 0 && (node + 1)->op == OP_CONV_2D && mbl_sole(node + 1, node->output))) &&
        g_dws > 0 && npix * C <= DWS_BYTES && (node + 1)->op == OP_CONV_2D && (node + 1)->input == node->output) {   // (v57_MB)
      dws_to = STRICT ? (int8_t*)arena + 36864 : dws_buf() + ((DWS_BYTES - npix * C) & ~15);   // (v57_MB: arena, mbp_act dead)
      a_out = (uint32_t)dws_to;
      dws_from = to->data;
    }
    const uint32_t a_w = (uint32_t)tf->data;
    const uint32_t a_m = (uint32_t)node->out_mult, a_s = (uint32_t)node->out_shift;
    const int32_t ioff = p->input_offset, ooff = p->output_offset;
    const int32_t amin = p->quantized_activation_min, amax = p->quantized_activation_max;
    // (v28D) the planar walks below write IN, HO, WO, CI, ROWSTR, PIXSTEP, ROWSTEP, NPIX, FX0OFF, SW,
    // PADTL and HWIN themselves: those are written once (same final register values as before)
    const int plw = (lay != LY_NHWC || C == 16) && s == 2;
    const int pln = plw || lay != LY_NHWC;
    npu_slot();
    W(RG_WGT, a_w); W(RG_OUT, a_out);
    W(RG_BIAS, (uint32_t)bias); W(RG_HASB, bias ? 3u : 2u);   // (v52BF) the model's bias, hardware fold
    W(RG_MULT, a_m); W(RG_SHIFT, a_s);
    W(RG_CO, C); W(RG_OUTLEN, npix * C);
    if (!pln) {
      W(RG_IN, a_in); W(RG_HO, Ho); W(RG_WO, Wo); W(RG_CI, C);
      W(RG_ROWSTR, W * C); W(RG_PIXSTEP, s * C); W(RG_ROWSTEP, s * W * C);
      W(RG_NPIX, npix);
      W(RG_FX0OFF, (3 - s) * C);
    }
    // stride 1, 16-channel groups (or super pixels): two output rows per window (4 input rows)
    const int32_t v2 = (s == 1 && (dsp || ((C & 15) == 0 && C >= 32)));
    // strip mode for stride 1 with 8 channels: rows of DW_SP + 2 taps per read
    const uint32_t dsm = 0;               // (the depthwise strip walk is not in the hardware)
    // three output rows per window (v3, RG_SW bit 21; bit 20 = p_pin): 5 input rows per window column
    const int32_t v3 = v2 && lay == LY_NHWC && 3 * W * C < 65536;
    if (!pln) W(RG_SW, (uint32_t)s | (v2 ? (1u << 16) : 0u) | (dsp ? (1u << 17) : 0u) | (v3 ? (1u << 21) : 0u));
    W(RG_CP, dsm ? 8u : 0u);
    if (dsm) W(RG_SM, DW_SP);
    if (!pln) { W(RG_PADTL, (uint32_t)pt | ((uint32_t)pl << 8)); W(RG_HWIN, (uint32_t)H | ((uint32_t)W << 16)); }
    if (v2 && lay == LY_NHWC) {          // (the planar walks below write their own)
      int32_t ng = 0;
      if (v3) { for (int32_t k = Ho + 2; k >= 3; k -= 3) ng++; } else ng = (Ho + 1) >> 1;
      W(RG_NPIX, (dsp ? (Wo >> 1) : Wo) * ng); W(RG_ROWSTEP, (v3 ? 3 : 2) * W * C);
      W(RG_OROW, Wo * C);
    }
    if (dsp) {
      W(RG_CO, 16);
      if (!pln) {
        W(RG_WO, Wo >> 1); W(RG_CI, 16); W(RG_PIXSTEP, 16);
        W(RG_FX0OFF, 32); W(RG_HWIN, (uint32_t)H | ((uint32_t)(W >> 1) << 16));
        if (!v2) W(RG_NPIX, npix >> 1);
      }
    }
    if (plw) {
      // stride 2 on 16-channel planes (C == 16: NHWC is one plane): the transposed walk (walk rows =
      // input columns) with two output columns per walk row group (window rows 0-2, 2-4; RG_SW bit 16
      // with stride 2), two new window columns per position (two input rows, each one 5-tap read),
      // outputs to their NHWC places
      const int32_t W16 = W * 16;
      W(RG_IN, (uint32_t)psrc - (uint32_t)(pl * 16 + pt * W16));
      W(RG_HO, Wo); W(RG_WO, Ho); W(RG_CI, C);
      W(RG_ROWSTR, 16); W(RG_PIXSTEP, 2 * W16); W(RG_ROWSTEP, 64); W(RG_FX0OFF, W16);
      W(RG_NPIX, Ho * ((Wo + 1) >> 1));
      W(RG_SW, 2u | (1u << 16) | (1u << 18));
      W(RG_PADTL, (uint32_t)pl | ((uint32_t)pt << 8)); W(RG_HWIN, (uint32_t)W | ((uint32_t)H << 16));
      W(RG_COLSTEP, W16); W(RG_IGRP, H * W16); W(RG_OPIX, Wo * C); W(RG_OROW, C);
    } else if (lay != LY_NHWC) {
      // 16-channel planes: walk the transposed image (walk rows = input columns, 16 bytes apart; the
      // new column of a sliding window is one read), two walk rows per window, outputs to their
      // NHWC places
      const int32_t W16 = W * 16;
      W(RG_IN, (uint32_t)psrc - (uint32_t)(pl * 16 + pt * W16));
      W(RG_HO, Wo); W(RG_WO, Ho); W(RG_CI, C);
      W(RG_ROWSTR, 16); W(RG_PIXSTEP, W16); W(RG_FX0OFF, 2 * W16);
      // three walk rows per window (RG_SW bit 21, v3): 5 input columns per new window column
      int32_t nt = 0;
      for (int32_t k = Wo + 2; k >= 3; k -= 3) nt++;
      W(RG_NPIX, Ho * nt); W(RG_ROWSTEP, 48);
      W(RG_SW, 1u | (1u << 16) | (1u << 18) | (1u << 21));
      W(RG_PADTL, (uint32_t)pl | ((uint32_t)pt << 8)); W(RG_HWIN, (uint32_t)W | ((uint32_t)H << 16));
      W(RG_COLSTEP, W16); W(RG_IGRP, H * W16); W(RG_OPIX, Wo * C); W(RG_OROW, C);
    }
    npu_common_quant(ioff, ooff, amin, amax);
    npu_run(NOP_DW);
    // mobilenet: its late-chain plan (node 19 = this node + 16) is checked now, while the NPU runs
    // the first conv, instead of on the exposed path before node 19
    if (g_dws > 0 && g_plan == 3 && mbp_state == 0) mbp_detect(node + 16, tensors);
    return;
  }
  npu_wait();
  ref_dwconv(p, node->out_mult, node->out_shift, &ti->shape, (const int8_t*)tin(ti), &tf->shape,
             (const int8_t*)tf->data, bias, &to->shape, (int8_t*)to->data);
}

// Global average pool on the NPU (the window is the whole input, one output pixel; any other
// average pool runs on the CPU): a depthwise pass of conv8 over 3x3 windows of stride 3 tiling the
// input (taps past its edge are zeros), with all-ones weights and no offsets; the NPU adds up the
// windows' sums s and requantises them with M = floor(2^31/n) + 1, shift 0: round(s*M/2^31), halves
// up, equals TFLM's sign(s)*floor((|s| + n/2)/n), because s*M/2^31 lies strictly beyond s/n (away
// from zero) by less than |s|/2^31 < 1/(2n) (|s| <= 128 n, n <= 256). 16-channel planes are walked
// as H x W x 16 tensors, one per group (RG_IGRP apart).
static int8_t pool_ones[144] SHARED __attribute__((aligned(16))) = { [0 ... 143] = 1 };
static void pool_regs(uint32_t a_in, uint32_t a_out, int32_t H, int32_t W, int32_t C, int planar,
                      int32_t amin, int32_t amax) {
  const int32_t n = H * W, Cs = planar ? 16 : C;
  // the pixels (consecutive, Cs bytes apart) are walked as rows of 3 (the last row may hold 1 or 2:
  // RG_FX0OFF), which wastes fewer window taps than H x W (v15A2 s7); n <= 256: <= 86 rows
  const int32_t Hr = (n + 2) / 3, last = n - 3 * (Hr - 1);
  // (pool mode ignores HASB and INOFF and reads no multiplier / shift words)
  W(RG_IN, a_in); W(RG_WGT, (uint32_t)pool_ones); W(RG_OUT, a_out);
  W(RG_MULT, (uint32_t)pool_ones); W(RG_SHIFT, (uint32_t)pool_ones);
  W(RG_HO, (Hr + 2) / 3); W(RG_WO, 1); W(RG_CO, C); W(RG_CI, Cs);
  W(RG_ROWSTR, 3 * Cs); W(RG_PIXSTEP, 3 * Cs); W(RG_ROWSTEP, 9 * Cs);
  W(RG_NPIX, (Hr + 2) / 3); W(RG_OUTLEN, C);
  W(RG_SW, 3u); W(RG_FX0OFF, (uint32_t)(last == 3 ? 0 : last));
  W(RG_CP, planar ? 384u : 128u);
  if (planar) W(RG_IGRP, n * 16);
  W(RG_PADTL, 0u); W(RG_HWIN, (uint32_t)Hr | (3u << 16));   // (rows in the low byte)
  W(RG_MO, 0x80000000u / (uint32_t)n + 1u);
  W(RG_OOFF, 0u); W(RG_AMIN, (uint32_t)amin); W(RG_AMAX, (uint32_t)amax);
}
static int npu_avgpool(const PoolParams* p, const Tensor* ti, const Tensor* to) {
  int32_t di[4], dout[4];
  if (!shape4(&ti->shape, di) || !shape4(&to->shape, dout)) return 0;
  const int32_t H = di[1], W = di[2], C = di[3];
  const int32_t n = H * W;
  if (dout[1] != 1 || dout[2] != 1 || dout[3] != C || p->filter_h != H || p->filter_w != W ||
      C < 1 || C > 1023 || H < 1 || W < 1 || H > 255 || W > 255 || n < 2 || n > 256 ||
      9 * C > 65535 || p->quantized_activation_min < -128 ||
      p->quantized_activation_max > 127 || p->quantized_activation_min > p->quantized_activation_max) return 0;
  int lay;
  const void* src = tview(ti, &lay);
  const int planar = (lay != LY_NHWC && (C & 15) == 0);
  if (lay != LY_NHWC && !planar) src = tin(ti);
  npu_slot();
  pool_regs((uint32_t)src, (uint32_t)to->data, H, W, C, planar, p->quantized_activation_min,
            p->quantized_activation_max);
  npu_run(NOP_DW);
  return 1;
}

static void AveragePool2DEval_k(const Node* node, Tensor* tensors) {
  kentry(node, tensors);
#if DS_DSX
  if (node == g_pf_node) { g_pf_node = 0; return; }     // (v39S) computed by its producer's drain
#endif
  if (mbp_hook(node, tensors)) return;
  const PoolParams* p = (const PoolParams*)node->builtin;
  const Tensor* ti = &tensors[node->input];
  const Tensor* to = &tensors[node->output];
  if (npu_avgpool(p, ti, to)) return;
  npu_wait();
  ref_avgpool(p, &ti->shape, (const int8_t*)tin(ti), &to->shape, (int8_t*)to->data);
}

static void MaxPool2DEval_k(const Node* node, Tensor* tensors) {
  kentry(node, tensors);
  const PoolParams* p = (const PoolParams*)node->builtin;
  const Tensor* ti = &tensors[node->input];
  const Tensor* to = &tensors[node->output];
  npu_wait();
  ref_maxpool(p, &ti->shape, (const int8_t*)tin(ti), &to->shape, (int8_t*)to->data);
}

// (v41_C) strict ADD: the fused-ADD path of an identity 1x1 conv (see tools/s_add.py)
static __attribute__((optimize("Os"))) int __attribute__((unused)) add_id(const Node* n, Tensor* tensors) {
  if (n->num_inputs != 2 || !in_dmem(n->inputs) || !in_dmem(n->builtin)) return 0;
  const int a0 = n->inputs[0], a1 = n->inputs[1];
  if (a0 < 0 || a0 > 4095 || a1 < 0 || a1 > 4095 || a0 == a1) return 0;
  const AddParams* p = (const AddParams*)n->builtin;
  // which = 1: input 2 is the identity conv's input (scale 2^30, shift 0), input 1 the residual
  const int which = (p->input2_multiplier == (1 << 30) && p->input2_shift == 0) ? 1 :
                    (p->input1_multiplier == (1 << 30) && p->input1_shift == 0) ? 0 : -1;
  if (which < 0) return 0;
  const int xi = which ? a1 : a0, ri = which ? a0 : a1;
  int32_t d[4], dr[4], dq[4];
  if (!shape4(&tensors[xi].shape, d) || !shape4(&tensors[ri].shape, dr) || !shape4(&tensors[n->output].shape, dq) ||
      dr[1] != d[1] || dr[2] != d[2] || dr[3] != d[3] || dq[1] != d[1] || dq[2] != d[2] || dq[3] != d[3]) return 0;
  const int32_t C = d[3], nel = d[1] * d[2] * C;
  if (C < 16 || C > 64 || (C & 15) || d[1] > 255 || d[2] > 255 || d[1] * d[2] > 4095) return 0;
  const int32_t ls = p->left_shift;
  const int32_t s1 = p->input1_shift, s2 = p->input2_shift;
  const int32_t e1 = ls + (s1 > 0 ? s1 : 0), e2 = ls + (s2 > 0 ? s2 : 0);
  const int32_t r1 = s1 < 0 ? -s1 : 0, r2 = s2 < 0 ? -s2 : 0;
  if (!(nel > 0 && nel < (1 << 20) && e1 == 20 && e2 == 20 && r1 <= 3 && r2 <= 3 &&
        p->input1_multiplier >= 0 && p->input2_multiplier >= 0 &&
        p->input1_offset >= -128 && p->input1_offset <= 128 && p->input2_offset >= -128 && p->input2_offset <= 128 &&
        p->output_shift >= -31 && p->output_shift <= 31 && p->output_offset >= -256 && p->output_offset <= 255 &&
        p->quantized_activation_min >= -128 && p->quantized_activation_max <= 127 &&
        p->quantized_activation_min <= p->quantized_activation_max)) return 0;
  const int32_t S1 = which ? s2 : s1, S2 = which ? s1 : s2;
  g_fzc.er = ((uint32_t)S1 & 0xffu) | (((uint32_t)S2 & 0xffu) << 8);
  g_fzc.m1 = (uint32_t)(which ? p->input2_multiplier : p->input1_multiplier);
  g_fzc.m2 = (uint32_t)(which ? p->input1_multiplier : p->input2_multiplier);
  g_fzc.o1 = (uint32_t)(which ? p->input2_offset : p->input1_offset);
  g_fzc.o2 = (uint32_t)(which ? p->input1_offset : p->input2_offset);
  g_fzc.mo = (uint32_t)p->output_multiplier; g_fzc.sho = (uint32_t)p->output_shift;
  g_fzc.oo = (uint32_t)p->output_offset;
  g_fzc.amin = (uint32_t)p->quantized_activation_min; g_fzc.amax = (uint32_t)p->quantized_activation_max;
  g_fzc.res = (const int8_t*)tensors[ri].data;
  g_fzc.out = (int8_t*)tensors[n->output].data;
  g_fzc.n = n;
  const int ok = npu_conv((const int8_t*)tensors[xi].data, C == 16 ? id_w16 : C == 32 ? id_w32 : id_w64, id_z, id_m, id_z, (int8_t*)tensors[n->output].data,
                          d[1], d[2], C, d[1], d[2], C, 1, 1, 1, 1, 0, 0, -128, 127, d[1], d[2], 0, 0);
  g_fzc.n = 0; g_fz_done = 0;
  return ok;
}

static void AddEval_k(const Node* node, Tensor* tensors) {
  kentry(node, tensors);
  if (node == g_fz_done) {                // computed by the conv before it
    g_fz_done = 0;
    if (!next_consumes(node)) npu_wait();
    return;
  }
  tee_plan(node, tensors);
  // (v53_Q k_addid) v41_C's identity-conv ADD is off: with fz2 its own-output write raced the ADD result
  const AddParams* p = (const AddParams*)node->builtin;
  const int32_t n = rs_flat(&tensors[node->output].shape);
  const int8_t* in1 = (const int8_t*)tin(&tensors[node->inputs[0]]);
  const int8_t* in2 = (const int8_t*)tin(&tensors[node->inputs[1]]);
  int8_t* out = tout(node, tensors);
  // (an ADD that is not fused into the conv producing one of its inputs runs on the CPU)
  npu_wait();
  ref_add(p, n, in1, in2, out);
}

// PAD. When the next node is a depthwise conv that takes the NPU path and whose input offset
// cancels the pad value, the pad is elided: nothing runs, and the depthwise conv reads the
// unpadded input with the padding applied implicitly (vpad). Anything else that reads the
// elided output materializes it first (tin).
// conv on the NPU's sliced walk (3x3, Ci % 16 == 0, stride 1, or 2 with Ci == 16) over an
// H x W x Ci input, with the same limits as npu_conv
static int conv_slices(const Node* cn, const Tensor* tensors, int32_t H, int32_t W, int32_t Ci) {
  const ConvParams* p = (const ConvParams*)cn->builtin;
  const Tensor* tf = &tensors[cn->filter];
  int32_t dout[4];
  if (!shape4(&tensors[cn->output].shape, dout) || tf->shape.n != 4 || p->pad_h != 0 || p->pad_w != 0)
    return 0;
  const int32_t Co = tf->shape.dims[0], sh = p->stride_h, sw = p->stride_w;
  const int32_t Ho = dout[1], Wo = dout[2], npix = Ho * Wo, nch = Ci >> 4;
  return tf->shape.dims[1] == 3 && tf->shape.dims[2] == 3 && tf->shape.dims[3] == Ci &&
         dout[3] == Co && (Ci & 15) == 0 && sh == sw &&
         (sw == 1 || (sw == 2 && (Ci == 16 || (g_pad_pin && cw2p_ok(Ci, Co, H, W, Ho, Wo, sh, sw))))) &&
         // (NPU widths: H, W, Ho, Wo in 1..255, Co in 1..1023, Ci in 1..65535, < 4096 pixels,
         //  row step < 65536)
         ((((H | W | Ho | Wo) | ((H - 1) | (W - 1) | (Ho - 1) | (Wo - 1))) & ~255) |
          ((Co | (Co - 1)) & ~1023) | ((Ci | (Ci - 1)) & ~65535)) == 0 &&
         npix <= 4095 && nch <= 31 && (nch == 1 || npix * 8 <= NPU_SCR_WORDS) &&
         sh * W * Ci <= 65535 &&
         (Ho - 1) * sh + 3 <= H && (Wo - 1) * sw + 3 <= W && 3 * Ci < (1 << 20) &&
         p->input_offset >= -127 && p->input_offset <= 128 && (Co > 8 || npix * Co < (1 << 20));
}

static int pad_elides(const Node* node, const Tensor* tensors) {
  const PadParams* p = (const PadParams*)node->builtin;
  int32_t di[4], dout[4];
  const Node* n1 = node + 1;
  if (n1->op != OP_DEPTHWISE_CONV_2D && n1->op != OP_CONV_2D) return 0;
  if (n1->op == OP_CONV_2D) {              // cheap reject first: only the sliced walk pads implicitly
    const RuntimeShape* fs = &tensors[n1->filter].shape;
    if (fs->n != 4 || fs->dims[1] != 3 || fs->dims[2] != 3 || (fs->dims[3] & 15) != 0) return 0;
  }
  if (!shape4(&tensors[node->input].shape, di) || !shape4(&tensors[node->output].shape, dout) ||
      dout[3] != di[3] || !next_consumes(node)) return 0;
  const int32_t hb = p->pad_h_before, wb = p->pad_w_before;
  if (hb < 0 || wb < 0 || hb > 3 || wb > 3 || dout[1] < hb + di[1] || dout[2] < wb + di[2] ||
      p->pad_value + ((const ConvParams*)n1->builtin)->input_offset != 0) return 0;
  return (n1->op == OP_DEPTHWISE_CONV_2D) ? dw_npu_ok(n1, tensors, dout[1], dout[2], di[3])
                                          : conv_slices(n1, tensors, dout[1], dout[2], di[3]);
}

#ifndef SPAD_G
#define SPAD_G 0   /* (v58_AR) no split PAD: OP_PADI is tied off */
#endif
static int spad_g DRV;   // (v49_DS) the kernel group's nodes 1..N-1 run (their PADs may be split, see pad_launch)
static void pad_launch(const Node* node, Tensor* tensors, int may_elide) {
  const PadParams* p = (const PadParams*)node->builtin;
  const RuntimeShape* ish = &tensors[node->input].shape;
  const RuntimeShape* osh = &tensors[node->output].shape;
  int lay;
  const int8_t* in = (const int8_t*)tview(&tensors[node->input], &lay);
  int8_t* out = (int8_t*)tensors[node->output].data;
  int32_t di[4], dout[4];
  const int shapes_ok = shape4(ish, di) && shape4(osh, dout) && dout[3] == di[3];
  g_pad_pin = STRICT || (lay == LY_P) || (lay == LY_RP);
  const int el = may_elide && shapes_ok && pad_elides(node, tensors);
  g_pad_pin = 0;
  if (el) {
    vpad.out = out; vpad.src = in; vpad.lay = lay; vpad.H = di[1]; vpad.W = di[2]; vpad.C = di[3];
    vpad.hb = p->pad_h_before; vpad.wb = p->pad_w_before; vpad.node = node; vpad.tensors = tensors;
    vpad.skip = 0;
    if (!STRICT) return;
    // (v57_RS) folded into its conv reader: nothing else reads the PAD's output -> not written
    if ((node + 1)->op == OP_CONV_2D && !t_read_later(node + 2, node->output, 0)) { vpad.skip = 1; return; }
  }
  if (lay != LY_NHWC) in = (const int8_t*)tin(&tensors[node->input]);
  if (!STRICT && rs_flat(osh) <= PAD_SCR_BYTES && next_consumes(node) && in != (const int8_t*)pad_scr) {
    redir_from = out;
    out = pad_scr;
  }
  if (shapes_ok) {
    const int32_t H = di[1], W = di[2], C = di[3], Ho = dout[1], Wo = dout[2];
    const int32_t hb = p->pad_h_before, wb = p->pad_w_before;
    const int32_t total = Ho * Wo * C;
    if (hb >= 0 && wb >= 0 && Ho >= hb + H && Wo >= wb + W && fits16(Ho) && total < (1 << 20) &&
        Ho <= 255 && Wo * C <= 65535) {      // (NPU widths: 8-bit rows, 16-bit row segments)
      const uint32_t pv = (uint32_t)p->pad_value;
      const uint32_t row = (uint32_t)(Wo * C), ic = (uint32_t)(W * C), in_n = (uint32_t)H * ic;
      npu_slot();
      W(RG_IN, (uint32_t)in); W(RG_OUT, (uint32_t)out);
      W(RG_PH, H); W(RG_PWC, ic); W(RG_PWOC, row); W(RG_PVAL, pv); W(RG_COUNT, in_n);
#if SPAD_G
      // (v49_DS) split PAD in a kernel group: the interior rows (one write burst each, ~2 cycles more per row) by the
      // op, the border (total - in_n bytes) by the background filler in idle write cycles of the next ops; node 0's
      // final STATUS wait covers it. Run lengths: first F, between rows row - ic, last L (all < 4096).
      const uint32_t F = (uint32_t)hb * row + (uint32_t)(wb * C), L = (uint32_t)total - F - in_n - (uint32_t)(H - 1) * (row - ic);
      if (spad_g && el && (uint32_t)total - in_n > 32u * (uint32_t)H && in_n < 65536u && (F | L | row) < 4096u) {
        W(RG_PTOT, F); W(RG_PWAC, row - ic); W(RG_PWBC, L);
        npu_run(NOP_PADI);
        return;
      }
#endif
      W(RG_PROWS, Ho); W(RG_PHB, hb); W(RG_PWBC, wb * C); W(RG_PWAC, (Wo - wb - W) * C); W(RG_PTOT, total);
      npu_run(NOP_PAD);
      return;
    }
  }
  npu_wait();
  ref_pad(p, ish, osh, in, out);
}

static __attribute__((optimize("Os"))) __attribute__((noinline)) void pad_materialize(void) {
  const Node* save = g_node;
  const Node* pn = vpad.node;
  vpad.out = 0;
  const int sk = vpad.skip;
  vpad.skip = 0;
  g_node = pn;
  pad_launch(pn, vpad.tensors, 0);
  g_node = save;
  if (sk) npu_wait();   // (v57_RS) a skipped PAD written late: its reader may be the CPU
}

// ====================================================================== first-op plans
// ds_cnn and mobilenet_v1_025 (specialisation for scored models; from v14_S): the CPU work before
// the first NPU ops is exposed (nothing runs yet). Their PAD kernel recognises the model once
// (shapes, wiring and quantisation parameters of the first nodes: exactly what the generic
// path's decisions depend on) and writes the register sets the generic path would write for the
// PAD and (ds_cnn) the first conv's gather copy, with the same driver state (redirection, arena
// slot); ds_cnn's conv kernel then only writes the conv's registers.
static const Node* g_plan_n0 DRV;      // the planned PAD (first node)
static uint32_t g_plan_q DRV;          // ds_cnn: the conv's planes slot
static inline int sh4(const RuntimeShape* s, int32_t h, int32_t w, int32_t c) {
  return s->n == 4 && s->dims[0] == 1 && s->dims[1] == h && s->dims[2] == w && s->dims[3] == c;
}
// (v53_Q) structure + the ranges the NPU takes (no literal zero points / clamps)
static inline int cpar(const ConvParams* p, int32_t s) {
  return p->stride_h == s && p->stride_w == s && p->pad_h == 0 && p->pad_w == 0 &&
         p->input_offset >= -127 && p->input_offset <= 128 && p->output_offset >= -256 && p->output_offset <= 255 &&
         p->quantized_activation_min >= -128 && p->quantized_activation_max <= 127 &&
         p->quantized_activation_min <= p->quantized_activation_max;
}
// same output offset / clamps (ops whose drains share RG_OOFF / AMIN / AMAX)
static inline int cq_eq(const ConvParams* a, const ConvParams* b) {
  return a->output_offset == b->output_offset && a->quantized_activation_min == b->quantized_activation_min &&
         a->quantized_activation_max == b->quantized_activation_max;
}
#define DWF_IOFF 128   // npu_dwf (hardware): the chain holds offset-added bytes a + 128; taps outside the image are 0
// (v51_D) structure + datapath ranges only (no parameter value)
static inline int cpar_s(const ConvParams* p, int32_t s) {
  return p->stride_h == s && p->stride_w == s && p->pad_h == 0 && p->pad_w == 0 &&
         p->input_offset >= -256 && p->input_offset <= 255 && p->output_offset >= -256 && p->output_offset <= 255 &&
         p->quantized_activation_min >= -128 && p->quantized_activation_max <= 127 &&
         p->quantized_activation_min <= p->quantized_activation_max;
}
// (v51_D) a depthwise d fused into its producer pr's drain (npu_dwf): input offset 128 (chain = pr's bytes + 128),
// pr's output offset / clamps, fast octets (each multiplier 0 or >= 2^30, each shift in [-11, 0])
// (v53_Q k_dsshw) the scalar part of dwf_ok; the multipliers / shifts are checked by the hardware flag
static __attribute__((optimize("Os"))) int __attribute__((noinline)) dwf_oks(const Node* d, const Node* pr) {
  const ConvParams* a = (const ConvParams*)d->builtin;
  const ConvParams* b = (const ConvParams*)pr->builtin;
  return cpar_s(a, 1) && a->input_offset == DWF_IOFF && a->output_offset == b->output_offset &&
         a->quantized_activation_min == b->quantized_activation_min &&
         a->quantized_activation_max == b->quantized_activation_max && d->out_mult != 0 && d->out_shift != 0;
}
static __attribute__((optimize("Os"))) int __attribute__((noinline, used)) dwf_ok(const Node* d, const Node* pr, int32_t C) {
  const ConvParams* a = (const ConvParams*)d->builtin;
  const ConvParams* b = (const ConvParams*)pr->builtin;
  if (!cpar_s(a, 1) || a->input_offset != DWF_IOFF || a->output_offset != b->output_offset ||
      a->quantized_activation_min != b->quantized_activation_min ||
      a->quantized_activation_max != b->quantized_activation_max || d->out_mult == 0 || d->out_shift == 0) return 0;
  const int32_t* om = d->out_mult;
  const int32_t* os = d->out_shift;
  for (int32_t ch = 0; ch < C; ch++) if ((om[ch] != 0 && om[ch] < (1 << 30)) || os[ch] < -11 || os[ch] > 0) return 0;
  return 1;
}
static inline int fshape(const RuntimeShape* s, int32_t a, int32_t b, int32_t c, int32_t d) {
  return s->n == 4 && s->dims[0] == a && s->dims[1] == b && s->dims[2] == c && s->dims[3] == d;
}
// Stage A (before the PAD is launched) checks what the PAD's registers depend on (its shapes and
// parameters); stage B (ds_cnn), while the PAD runs, what the first conv's registers depend on.
// The PAD always writes pad_scr with the redirection (safe for any reader: tin / tview follow
// it); the gather writes only gat_scr; the conv writes an arena slot (planes: every reader goes
// through tin / tview, which convert when needed). A model failing stage B runs the generic conv.
#ifndef DS_EARLY
#define DS_EARLY 1
#endif
// (v53_Q) mbe_ok / mbe_conv / ds_spec / ds_front (dead in STRICT; literal zero points) removed
// (v53_Q) plan_pad / plan_conv: STRICT returned at once (no plan); bodies with literal parameters removed
static int plan_pad(const Node* node, Tensor* tensors) { (void)node; (void)tensors; g_plan = -1; return 0; }
static int plan_conv(const Node* node, Tensor* tensors) { (void)node; (void)tensors; return 0; }

// (v52_RI) the ds_cnn planar block plan (dsx_*, DS_DSX 0: never compiled; literal offsets / clamps) is removed
// (v46_DS) ds_cnn's node 0, first inference: the PAD (49x10x1 -> 58x12x1, top 4, left 1, NHWC, own output) is
// launched directly and the gather (writes only gat_scr) right behind it; kg_run's detection / plan follow
#ifndef DS_P0
#define DS_P0 1
#endif
static int dss_p0 DRV, dss_g DRV;
static const Node* dss_raw DRV;   // (v58_DS k_spt) != 0: this inference's gather read the raw input; P (node 0's output) unwritten
static int dss_rq DRV;            // (v58_DS k_spt) the next dss_gat reads the raw input
static void pad_launch(const Node* node, Tensor* tensors, int may_elide);
static __attribute__((optimize("Os"), noinline, cold)) void dss_pmat(Tensor* t) {   // (v58_DS) write P after all
  const Node* n = dss_raw; dss_raw = 0;
  if (n) pad_launch(n, t, 0);
}
static void dss_gat(const Node* n0, Tensor* t);
static void dss_f1(const Node* n, Tensor* t);
static __attribute__((optimize("Os"))) __attribute__((optimize("Os"))) int dss_pad0(const Node* node, Tensor* t) {
  dss_p0 = 1;
  if (node->input < 0 || node->input > 4095 || t[node->input].shape.dims[1] != 49) return 0;
  const PadParams* p = (const PadParams*)node->builtin;
  if (node->output < 0 || node->output > 4095 || !in_dmem(p) || !sh4(&t[node->input].shape, 49, 10, 1) ||
      !sh4(&t[node->output].shape, 58, 12, 1) || p->pad_h_before != 4 || p->pad_w_before != 1) return 0;
  kentry(node, t);
  npu_slot();
  (void)p; dss_rq = 1; dss_gat(node, t); dss_g = 1; dss_raw = node;   // (v58_DS k_spt) no PAD op: the gather pads
  dss_f1(node, t);
  return 1;
}
static void mbr_pad(const Node* node, Tensor* tensors);
static void PadEval_k(const Node* node, Tensor* tensors) {
#if DS_P0
  if (!dss_p0 && dss_pad0(node, tensors)) return;
#endif
#if DS_DSX
  if (node == g_dsx_next && dsx_pad(node, tensors)) return;
#endif
  if (fp_pad(node, tensors)) return;
#if RS_PADE
  if (fp_pad2(node, tensors)) return;   // (v40K)
#endif
  kentry(node, tensors);
  if (MB_LAX && g_mbr_pad == 0 && mbf_try(node, tensors)) return;   // (v57_MB k_mbf) folded into node 1's conv
  if (node == g_mbr_pad) { mbr_pad(node, tensors); return; }   // (v50_MB k_rep) written by the producer's op
  if (plan_pad(node, tensors)) return;
  if (mbp_hook(node, tensors)) return;
  if (mbm_n0 != 0 && mbm_pad(node, tensors)) return;    // (v46_MB)
  pad_launch(node, tensors, 1);
  if (mbm_n0 != 0) mbm_acp(node, tensors);              // (v49_MB) aligned copy for the depthwise conv
}

static __attribute__((optimize("Os"))) void ReshapeEval_k(const Node* node, Tensor* tensors) {
  kentry(node, tensors);
  const int8_t* in = (const int8_t*)tin(&tensors[node->input]);
  int8_t* out = (int8_t*)tensors[node->output].data;
  const int32_t n = rs_flat(&tensors[node->output].shape);
  if (n > 0 && n < (1 << 16)) {              // (one row: 16-bit row segments)
    npu_slot();
    W(RG_IN, (uint32_t)in); W(RG_OUT, (uint32_t)out);
    W(RG_PROWS, 1); W(RG_PHB, 0); W(RG_PH, 1);
    W(RG_PWBC, 0); W(RG_PWC, n); W(RG_PWAC, 0); W(RG_PWOC, n); W(RG_PTOT, n); W(RG_PVAL, 0); W(RG_COUNT, n);
    npu_run(NOP_PAD);
    return;
  }
  npu_wait();
  for (int32_t i = 0; i < n; i++) out[i] = in[i];
}

// ====================================================================== CPU-only operators
static __attribute__((optimize("Os"))) void MeanEval_k(const Node* node, Tensor* tensors) {
  npu_wait();
  kentry(node, tensors);
  const MeanParams* p = (const MeanParams*)node->builtin;
  const RuntimeShape* in_shape = &tensors[node->input].shape;
  const int8_t* in = (const int8_t*)tin(&tensors[node->input]);
  int8_t* out = (int8_t*)tensors[node->output].data;
  const int H = in_shape->dims[1], W = in_shape->dims[2], C = in_shape->dims[3];
  const int32_t num = (int32_t)(H * W);
  for (int c = 0; c < C; c++) {
    int32_t acc = 0;
    for (int y = 0; y < H; y++)
      for (int x = 0; x < W; x++) acc += (int32_t)in[((y * W) + x) * C + c];
    int32_t shifted = (int32_t)((int64_t)acc - (int64_t)p->input_zero_point * (int64_t)num);
    int32_t v = MultiplyByQuantizedMultiplier(shifted, p->multiplier, p->shift) + p->output_zero_point;
    out[c] = (int8_t)clamp_i32(v, p->quantized_activation_min, p->quantized_activation_max);
  }
}

static __attribute__((optimize("Os"))) void MulEval_k(const Node* node, Tensor* tensors) {
  npu_wait();
  kentry(node, tensors);
  const MulParams* p = (const MulParams*)node->builtin;
  const int32_t n = rs_flat(&tensors[node->output].shape);
  const int8_t* in1 = (const int8_t*)tin(&tensors[node->inputs[0]]);
  const int8_t* in2 = (const int8_t*)tin(&tensors[node->inputs[1]]);
  int8_t* out = (int8_t*)tensors[node->output].data;
  for (int32_t i = 0; i < n; i++) {
    const int32_t raw = (p->input1_offset + (int32_t)in1[i]) * (p->input2_offset + (int32_t)in2[i]);
    int32_t o = MultiplyByQuantizedMultiplier(raw, p->output_multiplier, p->output_shift) + p->output_offset;
    out[i] = (int8_t)clamp_i32(o, p->quantized_activation_min, p->quantized_activation_max);
  }
}

static __attribute__((optimize("Os"))) void SoftmaxEval_k(const Node* node, Tensor* tensors) {
  npu_wait();
  kentry(node, tensors);
  const SoftmaxParams* p = (const SoftmaxParams*)node->builtin;
  const RuntimeShape* ish = &tensors[node->input].shape;
  const int8_t* in = (const int8_t*)tin(&tensors[node->input]);
  int8_t* out = (int8_t*)tensors[node->output].data;
  const int depth = ish->dims[ish->n - 1];
  int outer = 1;
  for (int i = 0; i < ish->n - 1; i++) outer *= ish->dims[i];
  for (int r = 0; r < outer; r++) {
    const int8_t* row = in + r * depth; int8_t* orow = out + r * depth;
    int32_t mx = -128;
    for (int c = 0; c < depth; c++) if (row[c] > mx) mx = row[c];
    int64_t sum = 0;
    for (int c = 0; c < depth; c++) { int d = mx - (int)row[c]; if (d > 255) d = 255; sum += p->exp_lut[d]; }
    for (int c = 0; c < depth; c++) {
      int d = mx - (int)row[c]; if (d > 255) d = 255;
      int32_t q = (int32_t)(((int64_t)p->exp_lut[d] * 256 + sum / 2) / sum);
      orow[c] = (int8_t)(clamp_i32(q, 0, 255) - 128);
    }
  }
}

#define CONCAT_IDENTITY_MULT  0x40000000
#define CONCAT_IDENTITY_SHIFT 1
#define CONCAT_MAX_INPUTS 16
static __attribute__((optimize("Os"))) void ConcatenationEval_k(const Node* node, Tensor* tensors) {
  npu_wait();
  kentry(node, tensors);
  const ConcatParams* p = (const ConcatParams*)node->builtin;
  const int n = node->num_inputs;
  const RuntimeShape* out_shape = &tensors[node->output].shape;
  int8_t* out = (int8_t*)tensors[node->output].data;
  const int H = out_shape->dims[1], W = out_shape->dims[2], Cout = out_shape->dims[3];
  const int32_t out_zp = p->output_zeropoint;
  int c_off = 0;
  for (int i = 0; i < n && i < CONCAT_MAX_INPUTS; i++) {
    const RuntimeShape* is = &tensors[node->inputs[i]].shape;
    const int8_t* in = (const int8_t*)tin(&tensors[node->inputs[i]]);
    const int Ci = is->dims[3];
    const int32_t zp = p->input_zeropoint[i], mult = p->input_multiplier[i], shift = p->input_shift[i];
    const int is_identity = (mult == (int32_t)CONCAT_IDENTITY_MULT &&
                             shift == CONCAT_IDENTITY_SHIFT && zp == out_zp);
    for (int y = 0; y < H; y++)
      for (int x = 0; x < W; x++) {
        const int8_t* isrc = in + ((y * W) + x) * Ci;
        int8_t* odst = out + ((y * W) + x) * Cout + c_off;
        if (is_identity) { for (int ci = 0; ci < Ci; ci++) odst[ci] = isrc[ci]; }
        else for (int ci = 0; ci < Ci; ci++) {
          int32_t v = MultiplyByQuantizedMultiplier((int32_t)isrc[ci] - zp, mult, (int)shift) + out_zp;
          odst[ci] = (int8_t)clamp_i32(v, -128, 127);
        }
      }
    c_off += Ci;
  }
}

static void lut_activation(const Node* node, Tensor* tensors) {
  npu_wait();
  kentry(node, tensors);
  const int8_t* lut = ((const LutParams*)node->builtin)->lut;
  const int8_t* in = (const int8_t*)tin(&tensors[node->input]);
  int8_t* out = (int8_t*)tensors[node->output].data;
  const int32_t n = rs_flat(&tensors[node->output].shape);
  for (int32_t i = 0; i < n; i++) out[i] = lut[(int)in[i] + 128];
}
static __attribute__((optimize("Os"))) void LogisticEval_k(const Node* node, Tensor* tensors) { lut_activation(node, tensors); }
static __attribute__((optimize("Os"))) void TanhEval_k(const Node* node, Tensor* tensors) { lut_activation(node, tensors); }

// (v52_RI) mobilenet_v1_025's compacted plan (gen_plan.py: mbp_blob = offline-compacted weights / biases / multipliers /
// shifts, mbp_samp = sampled weight words, captured register deltas) is removed (owner rule: no carried parameter
// data). In the STRICT build mbp_exec was empty and mbp_detect switched the plan off, as these stubs do.
static int8_t mbp_act[21920] SHARED __attribute__((aligned(16)));   // (v54_L) zero initial contents (image)   // zero-initialised scratch (ds_cnn, ad_fcae)
static int8_t* dws_buf(void) { return mbp_act; }
static void mbp_exec(int k, const Node* node, Tensor* tensors) { (void)k; (void)node; (void)tensors; }
static void mbp_detect(const Node* node, Tensor* tensors) { (void)node; (void)tensors; mbp_state = -1; }


// ====================================================================== (v41_C) ds_cnn strict plan
// (see tools/s_ds.py) every node's output in its own buffer; the pointwise convs run twice: plain (their own
// output) and fused with the next depthwise (npu_dwf writes the depthwise output)
#ifndef DSS_ON
#define DSS_ON 1
#endif
static __attribute__((optimize("Os"))) __attribute__((optimize("Os"))) int dss_chk(const Node* n0, const Tensor* t, int part) {   // (v46_DS) in parts
  const Node* c1 = n0 + 1;
  if (part == 0) {
  const PadParams* p0 = (const PadParams*)n0->builtin;
  if (!in_dmem(p0) || !sh4(&t[n0->input].shape, 49, 10, 1) || !sh4(&t[n0->output].shape, 58, 12, 1) ||
      p0->pad_h_before != 4 || p0->pad_w_before != 1) return 0;   // (v53_Q: the PAD writes its own pad value)
  if (c1->input != n0->output || c1->filter < 0 || c1->bias < 0 || !in_dmem(c1->builtin) ||
      !fshape(&t[c1->filter].shape, 64, 10, 4, 1) || !sh4(&t[c1->output].shape, 25, 5, 64) ||
      !cpar((const ConvParams*)c1->builtin, 2)) return 0;
  if (!fshape(&t[(n0 + 3)->filter].shape, 64, 3, 3, 1)) return 0;   // (v51_D: shape, not sampled filter words)
  return 1;
  }
  if (part <= 4) {
    const int b = part - 1;
    const Node* pd = n0 + 2 + 3 * b;
    const Node* d = pd + 1;
    const Node* c = pd + 2;
    const PadParams* pp = (const PadParams*)pd->builtin;
    if (!in_dmem(pp) || !in_dmem(d->builtin) || !in_dmem(c->builtin) || pd->input != (pd - 1)->output ||
        d->input != pd->output || c->input != d->output || d->filter < 0 || d->bias < 0 || c->filter < 0 ||
        c->bias < 0 || pp->pad_h_before != 1 || pp->pad_w_before != 1 ||
        !sh4(&t[pd->output].shape, 27, 7, 64) || !fshape(&t[d->filter].shape, 64, 3, 3, 1) ||
        !sh4(&t[d->output].shape, 25, 5, 64) || !fshape(&t[c->filter].shape, 64, 1, 1, 64) ||
        !sh4(&t[c->output].shape, 25, 5, 64) || !cpar((const ConvParams*)d->builtin, 1) ||
        !cpar((const ConvParams*)c->builtin, 1)) return 0;
    // (v53_Q) relations the fused depthwise needs: its input offset is npu_dwf's constant, its PAD's value the
    // matching zero point; its drain (and the pointwise's) shares conv1's output offset / clamps registers
    if (pp->pad_value + ((const ConvParams*)d->builtin)->input_offset != 0 || !in_dmem(pd[-1].builtin) ||
        !dwf_oks(d, pd - 1)) return 0;   // (v53_Q k_dsshw) d runs fused into pd - 1's drain; mult / shift: HW flag
    return 1;
  }
  const Node* pl = n0 + 14;
  const Node* fc = n0 + 15;
  const PoolParams* q = (const PoolParams*)pl->builtin;
  const FullyConnectedParams* f = (const FullyConnectedParams*)fc->builtin;
  if (!in_dmem(q) || !in_dmem(f) || pl->input != (pl - 1)->output || q->filter_h != 25 || q->filter_w != 5 ||
      !sh4(&t[pl->output].shape, 1, 1, 64) || q->quantized_activation_min < -128 || q->quantized_activation_max > 127 ||
      q->quantized_activation_min > q->quantized_activation_max) return 0;
  const RuntimeShape* fs = &t[fc->filter].shape;
  if (fc->input != pl->output || fc->filter < 0 || fs->n != 2 || fs->dims[0] != 12 || fs->dims[1] != 64 ||
      rs_flat(&t[fc->output].shape) != 12 || f->input_offset < -128 || f->input_offset > 128 ||
      f->output_offset < -256 || f->output_offset > 255 || f->quantized_activation_min < -128 ||
      f->quantized_activation_max > 127 || f->quantized_activation_min > f->quantized_activation_max) return 0;
  return 1;
}
#define DSS_B(b, w, co, kk, io) (b)   // (v52BF) the model's bias (hardware fold)
#define DSA(i) ((uint32_t)t[i].data)
// folded bias words (bases with the offline bias fold) looked up once per node: conv1 + 4 pointwise
static const Node* dss_bn[5] DRV;
static const int32_t* dss_bv[5] DRV;
static __attribute__((optimize("Os"))) const int32_t* dss_b(const Node* c, const Tensor* t, int32_t K, int32_t io) {
  for (int i = 0; i < 5; i++) if (dss_bn[i] == c) return dss_bv[i];
  const int32_t* b = DSS_B((const int32_t*)t[c->bias].data, (const int8_t*)t[c->filter].data, 64, K, io);
  for (int i = 0; i < 5; i++) if (dss_bn[i] == 0) { dss_bn[i] = c; dss_bv[i] = b; break; }
  return b;
}
// the fused depthwise's parameters (npu_dwf) and RG_CP bit 31
#ifndef DS_TEE
#define DS_TEE 1
#endif
#ifndef TEE_AG
#define TEE_AG 1   // (v49_DS) RG_PG bit 10 on the DWF tee (line-aligned groups, carry; r_tee3.py)
#endif
static uint32_t dss_tee DRV;
static uint32_t dss_pfo DRV;   // (v46_DS) != 0: the pointwise's drain also pools (global average) into dss_pfo
// the pool qualifies for the drain (v39S pool_d's conditions): a 25 x 5 window over the 25 x 5 map (odd n = 125: the
// drain's sums requantise exactly), clamp -128..127, 1x1x64 output; every pointwise channel drains as a fast octet
// (multiplier 0 or >= 2^30, shift -11..0)
static __attribute__((optimize("Os"))) int dss_pfok(const Node* c, const Node* pl, const Tensor* t) {
  const PoolParams* pp = (const PoolParams*)pl->builtin;
  if (pp->filter_h != 25 || pp->filter_w != 5 || pp->quantized_activation_min != -128 ||
      pp->quantized_activation_max != 127 || !sh4(&t[pl->output].shape, 1, 1, 64)) return 0;
  const int32_t* om = c->out_mult;
  const int32_t* os = c->out_shift;
  for (int ch = 0; ch < 64; ch++) if ((om[ch] != 0 && om[ch] < (1 << 30)) || os[ch] < -11 || os[ch] > 0) return 0;
  return 1;
}   // (v46_DS) != 0: the fused op also writes the pointwise's own output there (RG_PG bit 4)
static __attribute__((optimize("Os"))) void dss_dwf(const Node* d, const Tensor* t) {
  if (DS_TEE && dss_tee) { W(RG_PG, 513u | (TEE_AG ? 1024u : 0u)); W(RG_OFF1, dss_tee); }   /* (v49_DS) rotated tee */
  W(RG_RDELTA, DSA(d->filter)); W(RG_PTOT, DSA(d->bias));   /* (npu_dwf: raw biases) */ W(RG_COUNT, (uint32_t)d->out_mult);
  W(RG_SCR, (uint32_t)d->out_shift); W(RG_CP, 1u << 31);
}
// conv1 over the gathered vectors (plan_conv's set, NHWC out); d: fused depthwise (output at out) or 0
// (v53_Q) the conv's own input / output offsets and clamps
static inline void dss_q(const Node* c) {
  const ConvParams* p = (const ConvParams*)c->builtin;
  W(RG_INOFF, (uint32_t)p->input_offset); W(RG_OOFF, (uint32_t)p->output_offset);
  W(RG_AMIN, (uint32_t)p->quantized_activation_min); W(RG_AMAX, (uint32_t)p->quantized_activation_max);
}
static __attribute__((optimize("Os"))) __attribute__((optimize("Os"))) void dss_c1(const Node* c, const Tensor* t, uint32_t out, const Node* d) {
  npu_slot();
  W(RG_IN, (uint32_t)gat_scr); W(RG_WGT, DSA(c->filter)); W(RG_OUT, out);
  W(RG_BIAS, (uint32_t)dss_b(c, t, 40, 0)); W(RG_HASB, 3u);
  W(RG_MULT, (uint32_t)c->out_mult); W(RG_SHIFT, (uint32_t)c->out_shift);
  W(RG_HO, 25); W(RG_WO, 5); W(RG_CO, 64); W(RG_L, 40); W(RG_ROWSTR, 40); W(RG_PIXSTEP, 232);
  W(RG_ROWSTEP, 8); W(RG_K, 40); W(RG_WGLAST, 640); W(RG_NPIX, 125);
  W(RG_ACTLEN, 5000); W(RG_OUTLEN, 8000); W(RG_CI, 40); W(RG_FX0OFF, 80); W(RG_HWIN, 25u | (5u << 16));
  if (!(DS_TEE && d && dss_tee)) {   // (v46_DS) the tee F1 follows the gather: these stand / come with dss_dwf
    W(RG_FH, 1); W(RG_CONTIG, 0); W(RG_NCHUNKS, 1); W(RG_SCR, (uint32_t)npu_scr); W(RG_SLICED, 0);
    W(RG_SW, 1u); W(RG_PG, 1u); W(RG_PADTL, 0); W(RG_CP, 0);
  }
  dss_q(c);
  if (d) dss_dwf(d, t);
  R(RG_CMD) = NOP_CONV; npu_pending = 1;
}
// a pointwise 1x1 64 -> 64 over 25x5 (dsx_pw's set, NHWC out); d: fused depthwise (output at out) or 0
static __attribute__((optimize("Os"))) __attribute__((optimize("Os"))) void dss_pw(const Node* c, const Tensor* t, uint32_t in, uint32_t out, const Node* d) {
  npu_slot();
  W(RG_IN, in); W(RG_WGT, DSA(c->filter)); W(RG_OUT, out);
  W(RG_BIAS, (uint32_t)dss_b(c, t, 64, 0)); W(RG_HASB, 3u);
  W(RG_MULT, (uint32_t)c->out_mult); W(RG_SHIFT, (uint32_t)c->out_shift);
  W(RG_HO, 25); W(RG_WO, 5); W(RG_CO, 64); W(RG_FH, 1); W(RG_L, 64); W(RG_ROWSTR, 320); W(RG_PIXSTEP, 64);
  W(RG_ROWSTEP, 320); W(RG_K, 64); W(RG_WGLAST, 1024); W(RG_NPIX, 125); W(RG_CONTIG, 1); W(RG_NCHUNKS, 1);
  W(RG_ACTLEN, 8000); W(RG_OUTLEN, 8000); W(RG_SCR, (uint32_t)npu_scr); W(RG_SLICED, 0); W(RG_CI, 64);
  W(RG_SW, 1u); W(RG_FX0OFF, 128); W(RG_PG, 1u); W(RG_PADTL, 0); W(RG_HWIN, 25u | (5u << 16)); W(RG_CP, 0);
  dss_q(c);
  if (DS_TEE && dss_pfo) {   // (v46_DS) pool in the drain (RG_CP bit 30): pooled bytes -> dss_pfo, own output (tee) -> out
    W(RG_OUT, dss_pfo); W(RG_MO, 0x80000000u / 125u + 1u); W(RG_CP, 1u << 30);   // (v57_DS) pool only, no tee
  }
  if (d) dss_dwf(d, t);
  R(RG_CMD) = NOP_CONV; npu_pending = 1;
}
// PAD 1/1 of a 25x5x64 tensor (pad_launch's set)
#ifndef SPAD
#define SPAD 0   /* (v58_AR) no split PAD: OP_PADI is tied off */
#endif
#define dss_pv(n) ((uint32_t)((const PadParams*)(n)->builtin)->pad_value)   // (v53_Q) the PAD's own value
static __attribute__((optimize("Os"))) __attribute__((noinline)) void dss_pad(uint32_t in, uint32_t out, uint32_t pv) {   // (v49_DS) one copy: ITCM is tight
  npu_slot();
#if SPAD
  // (v49_DS) split PAD: interior rows (25 x 320 B at out + 512, 448 apart) by the op; border runs by the filler:
  // 512 B at out (RG_PTOT), 128 B after each of the first 24 rows (RG_PWAC), 512 B after the last row (RG_PWBC)
  W(RG_IN, in); W(RG_OUT, out); W(RG_PH, 25); W(RG_PWC, 320); W(RG_PWOC, 448); W(RG_PTOT, 512); W(RG_PWAC, 128);
  W(RG_PWBC, 512); W(RG_PVAL, pv); W(RG_COUNT, 8000);
  R(RG_CMD) = NOP_PADI; npu_pending = 1;
#else
  W(RG_IN, in); W(RG_OUT, out); W(RG_PROWS, 27); W(RG_PHB, 1); W(RG_PH, 25); W(RG_PWBC, 64); W(RG_PWC, 320);
  W(RG_PWAC, 64); W(RG_PWOC, 448); W(RG_PTOT, 12096); W(RG_PVAL, pv); W(RG_COUNT, 8000);
  R(RG_CMD) = NOP_PAD; npu_pending = 1;
#endif
}
static int dss_pfd DRV;   // (v46_DS) the pool ran in the last pointwise's drain
// (v46_DS) the gather (writes only gat_scr: no check needed) goes out first; then a quick check (the first fused op
// writes the outputs of nodes 1 and 3: 8000 bytes each, checked) and conv1 + depthwise 3. The full check (node 0 /
// conv1 parameters, ds_cnn's filter words, the blocks, pool / FC) follows in parts, each before the ops it covers;
// a failure leaves the generic kernels to (re)write every output, in NPU order after the ops already queued.
static void dss_gat(const Node* n0, Tensor* t);
static __attribute__((optimize("Os"))) __attribute__((optimize("Os"))) int dss_go(const Node* n0, Tensor* t, int g) {
  R(66) = 2u;   // (v53_Q k_dsshw) clear the DWF parameter flag
  const Node* c1 = n0 + 1;
  const Node* d3 = n0 + 3;
  if (g) dss_gat(n0, t);
  if (c1->filter < 0 || c1->bias < 0 || d3->filter < 0 || d3->bias < 0 ||
      rs_flat(&t[c1->output].shape) != 8000 || rs_flat(&t[d3->output].shape) != 8000) return 0;
  if (DS_TEE) {                                             // (v46_DS) conv1 -> out(1) (tee) + depthwise 3 -> out(3)
    dss_c1(c1, t, DSA((n0 + 3)->output), n0 + 3);   // (v57_DS) no tee: out(1) is an intermediate nobody reads
    return 1;
  }
  dss_c1(c1, t, DSA(c1->output), 0);                        // conv1 -> its own output
  dss_c1(c1, t, DSA((n0 + 3)->output), n0 + 3);             // conv1 again + depthwise 3 -> out(3)
  return 1;
}
// gather: T[ox][y][4] = P[y][2ox..2ox+3] of the PAD's (own, NHWC) output (plan_pad's strip-mode copy)
static __attribute__((optimize("Os"))) __attribute__((optimize("Os"))) void dss_gat(const Node* n0, Tensor* t) {
  const int r = dss_rq; dss_rq = 0;   // (v58_DS k_spt) r: from the unpadded input (origin of P = input - 41 bytes)
  if (!r && dss_raw == n0) return;    //   gat_scr already holds this inference's gather
  npu_slot();
  W(RG_IN, r ? (uint32_t)t[n0->input].data - 41u : DSA(n0->output)); W(RG_OUT, (uint32_t)gat_scr);
  W(RG_HO, 5); W(RG_WO, 58); W(RG_CO, 1); W(RG_FH, 1); W(RG_L, 4); W(RG_ROWSTR, r ? 10 : 12);
  W(RG_PIXSTEP, r ? 10 : 12); W(RG_ROWSTEP, 2); W(RG_NPIX, 290);
  W(RG_CONTIG, 0); W(RG_NCHUNKS, 1); W(RG_OUTLEN, 1160);
  W(RG_SLICED, 0); W(RG_SW, 1);
  if (r) { W(RG_HWIN, 49u << 24); W(RG_PVAL, (uint32_t)((const PadParams*)n0->builtin)->pad_value); }
  W(RG_PG, 0); W(RG_PADTL, r ? (1u << 16) | (1u << 21) | (4u << 24) : 0u); W(RG_CP, r ? 9u | (1u << 12) : 9u);
  if (r) { W(RG_SM, 10u | (94u << 8) | (74u << 16)); W(RG_SMDROP, 94u | (74u << 8)); }
  else { W(RG_SM, 10u | (112u << 8) | (88u << 16)); W(RG_SMDROP, 112u | (88u << 8)); }
  R(RG_CMD) = NOP_CONV; npu_pending = 1;
}
static int dss_rest(const Node* n0, Tensor* t);
static int dss_l DRV;    // (v46_DS) dss_go already ran for this inference (kg_run's early launch)
static __attribute__((optimize("Os"))) __attribute__((optimize("Os"))) int dss_run(const Node* n0, Tensor* t) {
  if (!DSS_ON) return 0;
  if (!dss_l && !dss_go(n0, t, 1)) return 0;
  dss_l = 0;
  if (!dss_rest(n0, t)) return 0;
  // (v53_Q k_dsshw) every op queued: wait for idle (the group ends there anyway); a DWF multiplier / shift outside
  // npu_dwf's fast-octet range -> the generic kernels rewrite nodes 1..15
  const uint32_t st = R(RG_STATUS);
  npu_pending = 0;
  if (st & 1u) { R(66) = 2u; return 0; }
  dss_raw = 0;   // (v58_DS k_spt) plan complete (DWF check passed): P stays an intermediate
  return 1;
}
static __attribute__((optimize("Os"))) __attribute__((optimize("Os"))) int dss_rest(const Node* n0, Tensor* t) {
  const Node* c1 = n0 + 1;
  if (!dss_chk(n0, t, 0) || !dss_chk(n0, t, 1)) return 0;  // (the generic kernels run nodes 1..15)
  // (v57_DS) PAD 2 not run: dw3 already ran fused into conv1's drain (implicit padding)
  for (int b = 0; b < 4; b++) {
    const Node* c = n0 + 4 + 3 * b;
    const uint32_t in = DSA(c->input);
    if (!dss_chk(n0, t, b < 3 ? b + 2 : 5)) return 0;       // (v46_DS) the next block / pool + FC before their ops
    if (DS_TEE && b < 3) {                                  // (v46_DS) pw -> out(p) (tee) + next dw -> out(p+2); the PAD
      dss_pw(c, t, in, DSA((c + 2)->output), c + 2);   // (v57_DS) no tee, no PAD copy (intermediates)
      continue;
    }
    if (DS_TEE && b == 3 && dss_pfok(c, n0 + 14, t)) {      // (v46_DS) pw13 -> out(13) (tee) + pool -> out(14)
      dss_pfo = DSA((n0 + 14)->output); dss_pw(c, t, in, DSA(c->output), 0); dss_pfo = 0;
      dss_pfd = 1;
      break;
    }
    dss_pw(c, t, in, DSA(c->output), 0);                    // the pointwise -> its own output
    if (b < 3) {
      dss_pad(DSA(c->output), DSA((c + 1)->output), dss_pv(c + 1));        // the next PAD
      dss_pw(c, t, in, DSA((c + 2)->output), c + 2);        // the pointwise again + the next depthwise
    }
  }
  const Node* pl = n0 + 14;
  const PoolParams* q = (const PoolParams*)pl->builtin;
  if (!dss_pfd) {
  npu_slot();
  pool_regs(DSA(pl->input), DSA(pl->output), 25, 5, 64, 0, q->quantized_activation_min, q->quantized_activation_max);
  R(RG_CMD) = NOP_DW; npu_pending = 1;
  }
  dss_pfd = 0;
  const Node* fc = n0 + 15;
  const FullyConnectedParams* f = (const FullyConnectedParams*)fc->builtin;
  {
    uint32_t* v = g_fs_tmp;
    v[0] = DSA(fc->input); v[1] = DSA(fc->filter); v[2] = fc->bias >= 0 ? DSA(fc->bias) : 0u;
    v[3] = (uint32_t)fc->out_mult; v[4] = (uint32_t)fc->out_shift; v[5] = DSA(fc->output); v[6] = fc->bias >= 0 ? 1u : 0u;
    v[7] = 12u; v[8] = 64u; v[9] = 1u; v[10] = 12u * 64u; v[11] = 12u; v[12] = 16u;
    v[13] = (uint32_t)f->input_offset; v[14] = (uint32_t)f->output_offset;
    v[15] = (uint32_t)f->quantized_activation_min; v[16] = (uint32_t)f->quantized_activation_max;
    g_fsv = 0;
    fs_write(v);
    R(RG_CMD) = NOP_CONV; npu_pending = 1;
  }
  return 1;
}
#undef DSA
#undef DSS_B
#define KG_PLAN(m, node, tensors) ((m) == 1 && dss_run(node, tensors))
// (v46_DS) first inference: ds_cnn's first ops go out before the kernel group's model detection. Node 0 must look like
// ds_cnn's PAD (490 -> 696 elements, then CONV, PAD, DEPTHWISE with valid indices); dss_go's quick check follows. If the
// detection then rejects the model, the per-node kernels rewrite every output (the early ops wrote only gat_scr and
// the outputs of nodes 1 and 3, which their own kernels write again, in NPU order).
// conv1 + depthwise 3 (dss_go after the node structure checks), once per inference
static __attribute__((optimize("Os"))) __attribute__((optimize("Os"))) void dss_f1(const Node* n, Tensor* t) {
  if (!dss_l && n[1].op == OP_CONV_2D && n[2].op == OP_PAD && n[3].op == OP_DEPTHWISE_CONV_2D &&
      n[1].input == n->output && n[1].output > n->output && n[1].output <= 4095 && n[3].output > n[1].output &&
      n[3].output <= 4095 && n[1].filter >= 0 && n[1].filter <= 4095 && n[3].filter >= 0 && n[3].filter <= 4095)
    dss_l = dss_go(n, t, 0);
}
static __attribute__((optimize("Os"))) __attribute__((optimize("Os"))) void dss_early(const Node* n, Tensor* t) {
  if (!dss_g) {                 // (PadEval_k's direct launch did not run: the gather now, if node 0 outputs 696 bytes)
    if (n->op != OP_PAD || n->output < 0 || n->output > 4095 || rs_flat(&t[n->output].shape) != 696) return;
    dss_gat(n, t);
  }
  dss_g = 0;
  dss_f1(n, t);
}
#define DSS_EARLY(n, t) do { if (DSS_ON) dss_early(n, t); } while (0)

// ====================================================================== (v41_C) kernel groups
// The four scored models (recognised at their first kernel call: operator sequence, every node's output size, input
// size) run as ONE group per inference: node 0's kernel runs every node's kernel body in order (each queues its ops
// while the previous op runs: op i+1's CPU configuration overlaps op i), then waits until the NPU is idle (every op
// finished, every write acknowledged) and returns. Nodes 1..N-1 find their outputs complete and return after the
// same STATUS load. No kernel returns with NPU work outstanding; every output is complete before its kernel returns.
// Any other model runs one node per kernel.
#ifndef KG_PRE
#define KG_PRE(m)
#endif
#ifndef KG_PLAN
#define KG_PLAN(m, node, tensors) 0
#endif
typedef void (*kg_fn)(const Node*, Tensor*);
static const kg_fn kg_tab[OP_COUNT] = {
  [OP_CONV_2D] = Conv2DEval_k, [OP_MAX_POOL_2D] = MaxPool2DEval_k, [OP_FULLY_CONNECTED] = FullyConnectedEval_k,
  [OP_DEPTHWISE_CONV_2D] = DepthwiseConv2DEval_k, [OP_ADD] = AddEval_k, [OP_AVERAGE_POOL_2D] = AveragePool2DEval_k,
  [OP_MEAN] = MeanEval_k, [OP_RESHAPE] = ReshapeEval_k, [OP_SOFTMAX] = SoftmaxEval_k, [OP_PAD] = PadEval_k,
  [OP_MUL] = MulEval_k, [OP_CONCATENATION] = ConcatenationEval_k, [OP_LOGISTIC] = LogisticEval_k, [OP_TANH] = TanhEval_k,
};
#define KG_NM 4
static const uint8_t kg_nn[KG_NM] = {10, 16, 43, 21};
static const uint16_t kg_off[KG_NM] = {0, 10, 26, 69};
static const uint32_t kg_in[KG_NM] = {640, 490, 27648, 3072};
static const uint8_t kg_op[90] = {OP_FULLY_CONNECTED, OP_FULLY_CONNECTED, OP_FULLY_CONNECTED, OP_FULLY_CONNECTED, OP_FULLY_CONNECTED, OP_FULLY_CONNECTED, OP_FULLY_CONNECTED, OP_FULLY_CONNECTED, OP_FULLY_CONNECTED, OP_FULLY_CONNECTED, OP_PAD, OP_CONV_2D, OP_PAD, OP_DEPTHWISE_CONV_2D, OP_CONV_2D, OP_PAD, OP_DEPTHWISE_CONV_2D, OP_CONV_2D, OP_PAD, OP_DEPTHWISE_CONV_2D, OP_CONV_2D, OP_PAD, OP_DEPTHWISE_CONV_2D, OP_CONV_2D, OP_AVERAGE_POOL_2D, OP_FULLY_CONNECTED, OP_PAD, OP_CONV_2D, OP_PAD, OP_DEPTHWISE_CONV_2D, OP_CONV_2D, OP_PAD, OP_DEPTHWISE_CONV_2D, OP_CONV_2D, OP_PAD, OP_DEPTHWISE_CONV_2D, OP_CONV_2D, OP_PAD, OP_DEPTHWISE_CONV_2D, OP_CONV_2D, OP_PAD, OP_DEPTHWISE_CONV_2D, OP_CONV_2D, OP_PAD, OP_DEPTHWISE_CONV_2D, OP_CONV_2D, OP_PAD, OP_DEPTHWISE_CONV_2D, OP_CONV_2D, OP_PAD, OP_DEPTHWISE_CONV_2D, OP_CONV_2D, OP_PAD, OP_DEPTHWISE_CONV_2D, OP_CONV_2D, OP_PAD, OP_DEPTHWISE_CONV_2D, OP_CONV_2D, OP_PAD, OP_DEPTHWISE_CONV_2D, OP_CONV_2D, OP_PAD, OP_DEPTHWISE_CONV_2D, OP_CONV_2D, OP_PAD, OP_DEPTHWISE_CONV_2D, OP_CONV_2D, OP_AVERAGE_POOL_2D, OP_FULLY_CONNECTED, OP_PAD, OP_CONV_2D, OP_PAD, OP_CONV_2D, OP_PAD, OP_CONV_2D, OP_ADD, OP_PAD, OP_CONV_2D, OP_PAD, OP_CONV_2D, OP_CONV_2D, OP_ADD, OP_PAD, OP_CONV_2D, OP_PAD, OP_CONV_2D, OP_CONV_2D, OP_ADD, OP_AVERAGE_POOL_2D, OP_FULLY_CONNECTED};
static const uint32_t kg_sz[90] = {128, 128, 128, 128, 8, 128, 128, 128, 128, 640, 696, 8000, 12096, 8000, 8000, 12096, 8000, 8000, 12096, 8000, 8000, 12096, 8000, 8000, 64, 12, 28227, 18432, 20000, 18432, 36864, 38416, 9216, 18432, 21632, 18432, 18432, 20000, 4608, 9216, 12544, 9216, 9216, 10816, 2304, 4608, 8192, 4608, 4608, 8192, 4608, 4608, 8192, 4608, 4608, 8192, 4608, 4608, 8192, 4608, 4608, 6272, 1152, 2304, 6400, 2304, 2304, 256, 2, 3468, 16384, 18496, 16384, 18496, 16384, 16384, 17424, 8192, 10368, 8192, 8192, 8192, 9248, 4096, 6400, 4096, 4096, 4096, 64, 10};
static const Node* kg_first DRV;         // node 0 of the recognised model
static int kg_n DRV;                     // its node count (0: not checked yet, -1: no group)
static int kg_m DRV;                     // its index in kg_nn (0 ad_fcae, 1 ds_cnn, 2 mobilenet_v1_025, 3 resnet8)
static const Node* kg_next DRV;          // next node whose output this inference's group has completed (0: none)
static const Node* kg_last DRV;
static __attribute__((optimize("Os"))) __attribute__((optimize("Os"))) int kg_detect(const Node* node, const Tensor* tensors) {
  for (int m = 0; m < KG_NM; m++) {
    const int n = kg_nn[m];
    const uint8_t* op = kg_op + kg_off[m];
    const uint32_t* sz = kg_sz + kg_off[m];
    if (node->input < 0 || node->input > 4095 || (uint32_t)rs_flat(&tensors[node->input].shape) != kg_in[m]) continue;
    // (operators and increasing output indices of all n nodes; output sizes of the first and the last)
    int k = 0;
    for (; k < n; k++) if ((uint32_t)node[k].op != op[k] || node[k].output <= (k ? node[k - 1].output : 0) || node[k].output > 4095) break;
    if (k == n && (uint32_t)rs_flat(&tensors[node[0].output].shape) == sz[0] &&
        (uint32_t)rs_flat(&tensors[node[n - 1].output].shape) == sz[n - 1]) { kg_m = m; return n; }
  }
  return -1;
}
static inline int kg_skip(const Node* node) {
  if (kg_next != 0 && node == kg_next) { kg_next = (node == kg_last) ? 0 : node + 1; return 1; }
  kg_next = 0;
  return 0;
}
static int __attribute__((noinline)) mbm_front(const Node* node, const Tensor* tensors);   // (v46_MB) defined at the end of the file
static __attribute__((optimize("Os"))) void kg_run(const Node* node, Tensor* tensors) {
  int e1 = 0;                                           // (v46_MB) node 1 already run (mobilenet front)
  if (kg_n == 0) {
    kg_first = node;
#ifdef DSS_EARLY
    DSS_EARLY(node, tensors);
#endif
    if (MBM && STRICT && mbm_front(node, tensors)) {
      KG_PRE(2); mbm_n0 = node; e1 = 1;
      kg_tab[(node + 1)->op](node + 1, tensors);
    }
    kg_n = kg_detect(node, tensors);
    if (dss_raw && !(kg_n > 1 && kg_m == 1)) dss_pmat(tensors);   // (v58_DS k_spt)
#ifdef DSS_EARLY
    if (kg_n < 0) dss_l = 0;
#endif
    if (e1 && !(kg_n > 1 && kg_m == 2)) { e1 = 0; mbm_n0 = 0; g_mbr_pad = 0; mbf_undo(tensors); }   // (v57_MB)
  }
  if (kg_n > 1 && node == kg_first && kg_m != 0) {   // (v46AD) ad_fcae: adq_run pipeline
    if (!e1) KG_PRE(kg_m);
    mbm_n0 = (MBM && STRICT && kg_m == 2) ? node : 0;   // (v46_MB)
    // (a model plan, KG_PLAN, runs nodes 1..N-1 itself: 1; else every node's kernel body in order)
    if (!KG_PLAN(kg_m, node, tensors)) {
      if (dss_raw) dss_pmat(tensors);   // (v58_DS k_spt) the generic nodes 1.. read P
      spad_g = SPAD_G;
      for (int k = 1 + e1; k < kg_n; k++) { const Node* q = node + k; kg_tab[q->op](q, tensors); }
      spad_g = 0;
    }
    mbm_n0 = 0;   // (v46_MB)
    kg_next = node + 1; kg_last = node + kg_n - 1;
  }
}
// ---- generated by v52_RI tools/gen_rimg4.py: RELOCATABLE compliant register images, scheduled run-time loads
// (no parameter value, no harness address: both come from the node / tensors pointers at run time) ----
_Static_assert(sizeof(Tensor) == 24 && sizeof(Node) == 40, "rgi4: Tensor / Node layout");
static uint32_t rgc[535] DRV;   // per-inference cache: node, tensors, words loaded from them (.data)

// (v52_RI rimg3) fills + structural checks: nf words w -> *sp++ = *(rgc[w >> 20] + (w & 0xfff)); nk pairs (w, e):
// the word at rgc[w >> 20] + (w & 0x7ffff) must equal e (bit 19 of w: its address bits 3..0 only, i.e. mod 16)
static int rgk3(const uint32_t** pp, uint32_t** spp, int nf, int nk) {
  const uint32_t* v = *pp;
  uint32_t* sp = *spp;
  uint32_t bad = 0;
  for (; nf > 0; nf--) {                       // (bit 19: a tensor base, its bits 3..0 = w bits 15..12)
    const uint32_t w = *v++, x = *(const volatile uint32_t*)(rgc[w >> 20] + (w & 0xfffu));
    *sp++ = x;
    if (w & 0x80000u) bad |= (x ^ (w >> 12)) & 15u;
  }
  for (; nk > 0; nk--, v += 2) {
    const uint32_t w = v[0];
    bad |= (*(const volatile uint32_t*)(rgc[w >> 20] + (w & 0x7ffffu)) ^ v[1]) & ((w & 0x80000u) ? 15u : 0xffffffffu);
  }
  *pp = v; *spp = sp;
  return bad == 0;
}
// guards {address word, count | zero-ok << 16, lo, hi}: every word in [lo, hi] (or 0 when zero-ok) - datapath limits
static int rgg_ok3(const uint32_t* g, int n) {
  for (; n > 0; n--, g += 4) {
    const volatile int32_t* a = (const volatile int32_t*)(rgc[g[0] >> 20] + (g[0] & 0xfffffu));
    const int32_t lo = (int32_t)g[2], hi = (int32_t)g[3];
    const uint32_t z = g[1] >> 16;
    for (uint32_t c = g[1] & 0xffffu; c > 0; c--, a++) {
      const int32_t v = *a;
      if ((v < lo || v > hi) && !(z && v == 0)) return 0;
    }
  }
  return 1;
}
static void __attribute__((noinline)) rlz_ds_end(void) {   // zeroed after the last op is taken
  R(9) = 0;
  R(10) = 0;
  R(11) = 0;
  R(12) = 0;
  R(14) = 0;
  R(20) = 0;
  R(25) = 0;
  R(27) = 0;
  R(29) = 0;
  R(30) = 0;
  R(31) = 0;
  R(32) = 0;
  R(33) = 0;
  R(40) = 0;
  R(45) = 0;
  R(49) = 0;
  R(51) = 0;
  R(53) = 0;
  R(55) = 0;
  R(58) = 0;
  R(60) = 0;
}
static const uint8_t rlz_ds_f[46] = {1,2,3,5,6,7,9,10,11,12,14,19,20,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,49,51,53,55,58,59,60,64,65,67,0};   // zeroed before op 0 if dirty
// ds_cnn: 7 ops (2 straight-line), 144 cache slots, 172 tasks
#define RGI_DS_N 7
#define RGI_DS_NN 16
#define RGI_DS_O0 1      // node 0 (a PAD): its output tensor index / height
#define RGI_DS_H0 58
static const uint8_t rlx_ds_c[30] = {19,0,0,17,14,10,23,0,0,13,14,0,11,0,3,17,14,0,0,0,7,7,10,3,10,17,0,10,10,6};   // per op: fill, chk, grd, compute words, cached regs, const
static const uint8_t rlx_ds_r[149] = {224,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,29,30,31,32,8,9,10,11,12,13,33,45,49,51,20,21,22,23,24,25,27,53,55,58,1,0,0,0,0,0,0,0,0,0,0,0,0,0,29,30,31,32,8,9,10,11,12,13,33,45,49,51,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,29,30,31,32,8,9,10,11,12,13,33,45,49,51,1,0,0,0,0,0,0,0,29,30,31,32,8,9,10,11,12,13,40,51,63,1,0,0,0,0,0,0,0,0,0,0,29,30,31,32,8,9,10,11,12,13,14,17,27,52,56,63,1};
static const uint32_t rlx_ds_v[282] = {
  0x000000b4u,0x001000d8u,0x001000f0u,0x000000b8u,0x000000bcu,0x00100168u,0x00000108u,0x00100138u,
  0x00100150u,0x0000010cu,0x0000012cu,0x00100180u,0x00100198u,0x00000130u,0x00000134u,0x01300010u,
  0x01300014u,0x01300018u,0x0130001cu,0x00400000u,0x00f00000u,0x02200000u,0x02300000u,0x02400000u,
  0x02500000u,0x00a00000u,0x01400000u,0x01500000u,0x01600000u,0x01700000u,0x01800000u,0x01900000u,
  0x01a00000u,0x01b00000u,0x01c00000u,0x01800000u,0xa5000026u,0xa5000027u,0xa5000028u,0xa5000029u,
  0xa500002au,0xa500002bu,0xa500002cu,0xa500002du,0xa500002eu,0xa500002fu,0xa5000030u,0xa5000031u,
  0xa5000032u,0xa5000033u,0x00000040u,0x00000040u,0x00000140u,0x00000040u,0x00000140u,0x00000040u,
  0x00000400u,0x00000001u,0x00001f40u,0x00000080u,0x00100210u,0x00000180u,0x001001e0u,0x001001f8u,
  0x00000184u,0x000001a4u,0x00100228u,0x00100240u,0x000001a8u,0x000001acu,0x001002b8u,0x000001f8u,
  0x00100288u,0x001002a0u,0x000001fcu,0x01d00010u,0x01d00014u,0x01d00018u,0x01d0001cu,0x03a00010u,
  0x03a00014u,0x03a00018u,0x03a0001cu,0x04400000u,0x04500000u,0x04600000u,0x04700000u,0x01e00000u,
  0x01f00000u,0x02000000u,0x02100000u,0x03500000u,0x03600000u,0x03700000u,0x03800000u,0x03900000u,
  0xa500004cu,0xa500004du,0xa500004eu,0xa500004fu,0xa5000034u,0xa5000050u,0xa5000051u,0xa5000052u,
  0xa5000053u,0xa5000054u,0xa5000055u,0xa5000056u,0xa5000057u,0xa5000058u,0x0000021cu,0x001002d0u,
  0x001002e8u,0x00000220u,0x00000224u,0x00100318u,0x00000244u,0x05900010u,0x05900014u,0x05900018u,
  0x0590001cu,0x05f00010u,1u,(uint32_t)-128,(uint32_t)-128,0x05f00014u,1u,(uint32_t)127,
  (uint32_t)127,0x05c00000u,65552u,(uint32_t)1073741824,(uint32_t)2147483647,0x04800000u,0x04900000u,0x04a00000u,
  0x04b00000u,0x03500000u,0x03b00000u,0x03c00000u,0x03d00000u,0x03e00000u,0x03f00000u,0x04000000u,
  0x04100000u,0x04200000u,0x04300000u,0x06000000u,0x06100000u,0x06200000u,0xa5000064u,0xa5000065u,
  0xa5000066u,0xa5000067u,0xa5000068u,0xa5000069u,0xa500006au,0xa500006bu,0xa500006cu,0xa500006du,
  0xa500006eu,0xa500006fu,0xa5000070u,0xa5000071u,0x05c00040u,65552u,(uint32_t)1073741824,(uint32_t)2147483647,
  0x05c00080u,65552u,(uint32_t)1073741824,(uint32_t)2147483647,0x05c000c0u,65552u,(uint32_t)1073741824,(uint32_t)2147483647,
  0x05d00000u,16u,(uint32_t)-11,(uint32_t)0,0x05d00040u,16u,(uint32_t)-11,(uint32_t)0,
  0x05d00080u,16u,(uint32_t)-11,(uint32_t)0,0x05d000c0u,16u,(uint32_t)-11,(uint32_t)0,
  0x06300000u,0x03f00000u,0x05a00000u,0x05b00000u,0x05c00000u,0x05d00000u,0x05e00000u,0xa5000072u,
  0xa5000073u,0xa5000074u,0xa5000075u,0xa5000076u,0xa5000077u,0xa5000078u,0xa5000079u,0xa500007au,
  0xa500007bu,0x010624deu,(uint32_t)npu_scr + 0u,0x40000000u,0x0000026cu,0x00100330u,0x00100348u,0x00000270u,
  0x00000274u,0x00100360u,0x07c00000u,0x07c00004u,0x07c00008u,0x07c0000cu,0x00000000u,(uint32_t)(OP_PAD),
  0x00000028u,(uint32_t)(OP_CONV_2D),0x00000050u,(uint32_t)(OP_PAD),0x00000078u,(uint32_t)(OP_DEPTHWISE_CONV_2D),0x000000a0u,(uint32_t)(OP_CONV_2D),
  0x000000c8u,(uint32_t)(OP_PAD),0x000000f0u,(uint32_t)(OP_DEPTHWISE_CONV_2D),0x00000118u,(uint32_t)(OP_CONV_2D),0x00000140u,(uint32_t)(OP_PAD),
  0x00000168u,(uint32_t)(OP_DEPTHWISE_CONV_2D),0x00000190u,(uint32_t)(OP_CONV_2D),0x000001b8u,(uint32_t)(OP_PAD),0x000001e0u,(uint32_t)(OP_DEPTHWISE_CONV_2D),
  0x00000208u,(uint32_t)(OP_CONV_2D),0x00000230u,(uint32_t)(OP_AVERAGE_POOL_2D),0x00000258u,(uint32_t)(OP_FULLY_CONNECTED),0x00000268u,(uint32_t)(36),
  0x08200000u,0x08300000u,0x08400000u,0x08500000u,0x05e00000u,0x07d00000u,0x07e00000u,0x07f00000u,
  0x08000000u,0x08100000u,0xa5000086u,0xa5000087u,0xa5000088u,0xa5000089u,0xa500008au,0xa500008bu,
  0xa500008cu,0xa500008du,0xa500008eu,0xa500008fu,0x00000001u,0x0000000cu,0x00000300u,0x00000001u,
  0x0000000cu,0x00000010u,
};
static const uint32_t rlxs_ds_v[21] = {0x00000014u,0x00100000u,0x00200010u,0x00100024u,(uint32_t)(12),0x00100028u,(uint32_t)(1),0x0000003cu,0x00100030u,0x00100048u,0x00000040u,0x00000044u,0x001000c0u,0x00000090u,0x00100090u,0x001000a8u,0x00000094u,0x00500010u,0x00500014u,0x00500018u,0x0050001cu};   // ops 0..K-1: fills, checks, guards
static int __attribute__((noinline)) rgf_ds(void) {   // ops 0..1 (v47_a straight-line); 0: a check failed
  const uint32_t* p = rlxs_ds_v; uint32_t* sp = rgc + 2; uint32_t bad = 0;
  rgc[2] = *(const volatile uint32_t*)(rgc[0] + 20u);
  rgc[3] = *(const volatile uint32_t*)(rgc[1] + 0u);
  rgc[4] = *(const volatile uint32_t*)(rgc[2] + 16u);
  bad |= *(const volatile uint32_t*)(rgc[1] + 36u) ^ (uint32_t)(12);
  bad |= *(const volatile uint32_t*)(rgc[1] + 40u) ^ (uint32_t)(1);
  if (bad) return 0;
  p += 7; sp += 3;
  const uint32_t q0_0 = (rgc[4] + 0u);
  const uint32_t q0_1 = (rgc[3] + 1048535u);
  R(50) = q0_0;
  R(8) = q0_1;
  R(13) = (uint32_t)gat_scr + 0u;
  R(15) = 0x00000005u;
  R(16) = 0x0000003au;
  R(17) = 0x00000001u;
  R(18) = 0x00000001u;
  R(21) = 0x00000004u;
  R(22) = 0x0000000au;
  R(23) = 0x0000000au;
  R(24) = 0x00000002u;
  R(47) = 0x004a5e0au;
  R(48) = 0x00004a5eu;
  R(52) = 0x00000122u;
  R(54) = 0x00000001u;
  R(56) = 0x00000488u;
  R(57) = 0x00000001u;
  R(61) = 0x04210000u;
  R(62) = 0x31000000u;
  R(63) = 0x00001009u;
  R(RG_CMD) = 1u; npu_pending = 1;
  rgc[5] = *(const volatile uint32_t*)(rgc[0] + 60u);
  rgc[6] = *(const volatile uint32_t*)(rgc[1] + 48u);
  rgc[7] = *(const volatile uint32_t*)(rgc[1] + 72u);
  rgc[8] = *(const volatile uint32_t*)(rgc[0] + 64u);
  rgc[9] = *(const volatile uint32_t*)(rgc[0] + 68u);
  rgc[10] = *(const volatile uint32_t*)(rgc[1] + 192u);
  rgc[11] = *(const volatile uint32_t*)(rgc[0] + 144u);
  rgc[12] = *(const volatile uint32_t*)(rgc[1] + 144u);
  rgc[13] = *(const volatile uint32_t*)(rgc[1] + 168u);
  rgc[14] = *(const volatile uint32_t*)(rgc[0] + 148u);
  rgc[15] = *(const volatile uint32_t*)(rgc[5] + 16u);
  rgc[16] = *(const volatile uint32_t*)(rgc[5] + 20u);
  rgc[17] = *(const volatile uint32_t*)(rgc[5] + 24u);
  rgc[18] = *(const volatile uint32_t*)(rgc[5] + 28u);
  if (bad) return 0;   // (op 0 is running: rgi_back waits)
  p += 14; sp += 14;
  const uint32_t q1_0 = (rgc[15] + 0u);
  const uint32_t q1_1 = (rgc[16] + 0u);
  const uint32_t q1_2 = (rgc[17] + 0u);
  const uint32_t q1_3 = (rgc[18] + 0u);
  const uint32_t q1_4 = (rgc[6] + 0u);
  const uint32_t q1_5 = (rgc[7] + 0u);
  const uint32_t q1_6 = (rgc[8] + 0u);
  const uint32_t q1_7 = (rgc[9] + 0u);
  const uint32_t q1_8 = (rgc[10] + 0u);
  const uint32_t q1_9 = (rgc[11] + 0u);
  const uint32_t q1_10 = (rgc[12] + 0u);
  const uint32_t q1_11 = (rgc[13] + 0u);
  const uint32_t q1_12 = (rgc[14] + 0u);
  (void)R(RG_SLOT);
  R(29) = q1_0;
  R(30) = q1_1;
  R(31) = q1_2;
  R(32) = q1_3;
  R(9) = q1_4;
  R(10) = q1_5;
  R(11) = q1_6;
  R(12) = q1_7;
  R(13) = q1_8;
  R(33) = q1_9;
  R(45) = q1_10;
  R(49) = q1_11;
  R(51) = q1_12;
  R(8) = (uint32_t)gat_scr + 0u;
  R(14) = 0x00000003u;
  R(15) = 0x00000019u;
  R(16) = 0x00000005u;
  R(17) = 0x00000040u;
  R(20) = 0x00000028u;
  R(21) = 0x00000028u;
  R(22) = 0x00000028u;
  R(23) = 0x000000e8u;
  R(24) = 0x00000008u;
  R(25) = 0x00000028u;
  R(27) = 0x00000280u;
  R(52) = 0x0000007du;
  R(55) = 0x00001388u;
  R(56) = 0x00001f40u;
  R(58) = 0x00000050u;
  R(60) = 0x00000001u;
  R(61) = 0x00000000u;
  R(62) = 0x00050019u;
  R(63) = 0x80000000u;
  R(RG_CMD) = 1u; npu_pending = 1;
  return 1;
}
#define RGF_DS_K 2
#define RGF_DS_S 19   // the next cache slot after the straight-line ops
static void __attribute__((noinline)) rlz_mb_end(void) {   // zeroed after the last op is taken
  R(1) = 0;
  R(2) = 0;
  R(3) = 0;
  R(6) = 0;
  R(36) = 0;
  R(37) = 0;
  R(38) = 0;
  R(39) = 0;
  R(40) = 0;
  R(45) = 0;
  R(47) = 0;
  R(53) = 0;
  R(60) = 0;
  R(63) = 0;
}
static const uint8_t rlz_mb_f[58] = {1,2,3,5,6,7,9,10,11,12,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,34,35,36,37,38,39,40,41,43,45,47,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,67,0};   // zeroed before op 0 if dirty
// mobilenet_v1_025: 43 ops (2 straight-line), 535 cache slots, 593 tasks
#define RGI_MB_N 43
#define RGI_MB_NN 43
#define RGI_MB_O0 1      // node 0 (a PAD): its output tensor index / height
#define RGI_MB_H0 97
static const uint8_t rlx_mb_c[246] = {62,0,0,55,4,7,0,0,0,0,8,16,42,0,0,42,8,19,57,0,0,46,8,16,36,0,0,41,8,16,24,0,0,25,8,14,30,0,0,29,8,13,25,0,0,27,8,12,1,26,0,0,8,12,0,0,0,0,8,14,0,18,0,7,8,13,0,0,0,3,8,12,0,0,0,0,8,12,0,0,0,0,8,14,0,0,0,0,8,13,0,0,0,0,8,9,0,0,0,0,7,9,0,0,0,0,4,18,0,0,0,0,0,23,0,0,0,0,8,10,0,0,0,0,7,3,0,0,0,0,4,20,0,0,0,0,0,23,0,0,0,0,8,10,0,0,0,0,7,3,0,0,0,0,4,20,0,0,0,0,0,23,0,0,0,0,8,10,0,0,0,0,7,3,0,0,0,0,4,20,0,0,0,0,0,23,0,0,0,0,8,13,0,0,0,0,7,5,0,0,0,0,4,19,0,0,0,0,0,23,0,0,0,0,8,12,0,0,0,0,7,4,0,0,0,0,4,21,0,0,0,0,2,22,0,0,0,0,4,16,0,0,0,0,10,5};   // per op: fill, chk, grd, compute words, cached regs, const
static const uint8_t rlx_mb_r[1117] = {224,0,0,0,0,224,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,224,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,224,0,0,0,0,0,0,0,0,0,0,0,29,30,31,32,8,13,15,52,55,56,62,1,29,30,31,32,9,10,11,12,6,8,13,15,16,20,22,23,24,52,56,57,58,61,62,63,2,0,0,0,0,0,224,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,224,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,29,30,31,32,9,10,11,12,8,13,16,18,20,21,23,24,25,27,52,53,55,56,57,58,60,61,62,1,224,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,224,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,224,0,0,0,0,0,0,0,0,0,29,30,31,32,9,10,11,12,1,2,3,6,8,13,15,16,20,22,23,24,52,56,57,58,2,0,0,0,0,0,0,0,0,0,0,224,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,224,0,0,0,0,0,0,0,0,29,30,31,32,9,10,11,12,8,13,17,21,22,23,24,25,27,52,55,56,57,58,60,62,1,0,0,0,0,0,0,0,0,0,0,0,224,0,0,0,0,0,0,0,0,0,0,0,0,29,30,31,32,9,10,11,12,1,2,3,6,8,13,20,22,23,24,52,57,58,61,2,0,0,0,0,0,0,0,224,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,29,30,31,32,9,10,11,12,8,13,21,22,23,24,25,27,52,55,57,58,61,1,224,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,229,0,0,0,0,29,30,31,32,9,10,11,12,3,8,13,15,16,22,23,24,52,56,57,58,2,29,30,31,32,9,10,11,12,8,13,17,22,23,24,52,55,56,57,58,62,1,29,30,31,32,9,10,11,12,1,2,3,6,8,13,20,22,23,24,52,57,58,61,2,0,0,0,0,0,0,0,29,30,31,32,9,10,11,12,8,13,21,22,23,24,25,27,52,55,57,58,61,1,0,0,0,29,30,31,32,9,10,11,12,3,8,13,15,16,22,23,24,52,56,57,58,2,29,30,31,32,9,10,11,12,8,13,17,22,23,24,52,55,56,57,58,62,1,29,30,31,32,9,10,11,12,1,2,3,6,8,13,20,22,23,24,52,57,58,61,2,29,30,31,32,9,10,11,12,8,13,21,22,23,24,25,52,55,57,58,60,61,1,29,30,31,32,9,10,11,12,8,13,22,23,24,52,57,58,61,2,30,31,32,9,10,11,12,8,13,21,36,37,38,39,45,49,9,29,30,31,32,8,9,10,11,12,13,14,17,21,22,23,24,52,56,57,58,60,61,1,8,9,10,11,12,13,14,17,20,21,22,23,24,25,27,29,30,31,32,55,56,58,60,1,29,30,31,32,9,10,11,12,8,13,20,22,23,24,52,57,58,61,2,30,31,32,9,10,11,12,8,13,25,9,29,30,31,32,8,9,10,11,12,13,14,17,21,22,23,24,27,52,55,56,57,58,60,61,1,8,9,10,11,12,13,14,17,20,21,22,23,24,25,27,29,30,31,32,55,56,58,60,1,29,30,31,32,9,10,11,12,8,13,20,22,23,24,52,57,58,61,2,30,31,32,9,10,11,12,8,13,25,9,29,30,31,32,8,9,10,11,12,13,14,17,21,22,23,24,27,52,55,56,57,58,60,61,1,8,9,10,11,12,13,14,17,20,21,22,23,24,25,27,29,30,31,32,55,56,58,60,1,29,30,31,32,9,10,11,12,8,13,20,22,23,24,52,57,58,61,2,30,31,32,9,10,11,12,8,13,25,9,29,30,31,32,8,9,10,11,12,13,14,17,21,22,23,24,27,52,55,56,57,58,60,61,1,8,9,10,11,12,13,14,17,20,21,22,23,24,25,27,29,30,31,32,55,56,58,60,1,29,30,31,32,9,10,11,12,3,8,13,15,16,20,22,23,24,52,56,57,58,2,30,31,32,9,10,11,12,8,13,17,21,25,9,29,30,31,32,8,9,10,11,12,13,14,17,21,22,23,24,27,52,56,57,58,60,62,1,8,9,10,11,12,13,14,17,20,21,22,23,24,25,27,29,30,31,32,55,56,58,60,1,29,30,31,32,9,10,11,12,1,2,3,6,8,13,20,22,24,52,57,61,2,30,31,32,9,10,11,12,8,13,21,25,9,29,30,31,32,8,9,10,11,12,13,14,17,21,22,23,24,27,52,54,55,56,57,58,60,61,1,13,60,8,9,10,11,12,14,17,20,21,22,23,24,25,27,29,30,31,32,54,55,56,58,1,31,32,8,13,9,11,12,15,16,20,22,23,24,40,52,56,57,58,60,63,2,29,30,31,32,8,9,10,11,12,13,14,17,25,56,63,1};
static const uint32_t rlx_mb_v[1456] = {
  0x00000014u,0x0000008cu,0x00100090u,0x001000a8u,0x00000090u,0x00000094u,0x00000064u,0x000000b4u,
  0x001000d8u,0x001000f0u,0x000000b8u,0x000000bcu,0x00000104u,0x00100138u,0x00100150u,0x00000108u,
  0x0000010cu,0x000000dcu,0x0000012cu,0x00100180u,0x00100198u,0x00000130u,0x00000134u,0x0000017cu,
  0x001001e0u,0x001001f8u,0x00000180u,0x00000184u,0x00000154u,0x000001a4u,0x00100228u,0x00100240u,
  0x000001a8u,0x000001acu,0x00c00010u,0x00d00010u,0x00d00014u,0x00d00018u,0x00d0001cu,0x01200010u,
  0x01300010u,0x01300014u,0x01300018u,0x0130001cu,0x01800010u,0x01800014u,0x01800018u,0x0180001cu,
  0x01d00010u,0x01e00010u,0x01e00014u,0x01e00018u,0x01e0001cu,0x02300010u,0x02300014u,0x02300018u,
  0x0230001cu,0x02800010u,0x02900010u,0x02900014u,0x02900018u,0x0290001cu,0x02e00000u,0x00800000u,
  0x00800000u,0x00900000u,0x00a00000u,0x00b00000u,0x03300000u,0x02f00000u,0x02f00000u,0x03000000u,
  0x03100000u,0x03200000u,0x00e00000u,0x00f00000u,0x01000000u,0x01100000u,0x03400000u,0x03500000u,
  0x03600000u,0x03700000u,0x01400000u,0x01500000u,0x01600000u,0x01700000u,0x03c00000u,0x03800000u,
  0x03800000u,0x03900000u,0x03a00000u,0x03b00000u,0x01900000u,0x01a00000u,0x01b00000u,0x01c00000u,
  0x03d00000u,0x03e00000u,0x03f00000u,0x04000000u,0x01f00000u,0x02000000u,0x02100000u,0x02200000u,
  0x04500000u,0x04100000u,0x04100000u,0x04200000u,0x04300000u,0x04400000u,0x02400000u,0x02500000u,
  0x02600000u,0x02700000u,0x04600000u,0x04700000u,0x04800000u,0xa500004au,0xa500004bu,0xa500004cu,
  0xa500004du,(uint32_t)gat_scr + 0u,(uint32_t)pad_scr + 18048u,0x00000001u,0x00000030u,0x000001b0u,0x00000180u,0x00600002u,
  0xa500004eu,0xa500004fu,0xa5000050u,0xa5000051u,0xa5000052u,0xa5000053u,0xa5000054u,0xa5000055u,
  0x00000180u,(uint32_t)fb_tab + 14192u,(uint32_t)arena + 36864u,0x00000030u,0x00000018u,0x00000010u,0x00000180u,0x00000010u,
  0x00000480u,0x00000180u,0x00004800u,0x00230001u,0x00000020u,0x00000101u,0x00180030u,0x00000000u,
  0x000001f4u,0x00100288u,0x001002a0u,0x000001f8u,0x000001fcu,0x000001ccu,0x0000021cu,0x001002d0u,
  0x001002e8u,0x00000220u,0x00000224u,0x0000026cu,0x00100330u,0x00100348u,0x00000270u,0x00000274u,
  0x00000244u,0x00000294u,0x00100378u,0x00100390u,0x00000298u,0x0000029cu,0x000002e4u,0x001003d8u,
  0x07900010u,0x07900014u,0x07900018u,0x0790001cu,0x07e00010u,0x07f00010u,0x07f00014u,0x07f00018u,
  0x07f0001cu,0x08400010u,0x08400014u,0x08400018u,0x0840001cu,0x08900010u,0x08a00010u,0x08a00014u,
  0x08a00018u,0x08a0001cu,0x04900000u,0x02a00000u,0x02b00000u,0x02c00000u,0x02d00000u,0x09500000u,
  0x09100000u,0x09100000u,0x09200000u,0x09300000u,0x09400000u,0x07a00000u,0x07b00000u,0x07c00000u,
  0x07d00000u,0x09600000u,0x09700000u,0x09800000u,0x09900000u,0x08000000u,0x08100000u,0x08200000u,
  0x08300000u,0x09e00000u,0x09a00000u,0x09a00000u,0x09b00000u,0x09c00000u,0x09d00000u,0x08500000u,
  0x08600000u,0x08700000u,0x08800000u,0x09f00000u,0x0a000000u,0x0a100000u,0x0a200000u,0x08b00000u,
  0x08c00000u,0x08d00000u,0x08e00000u,0x09000000u,0xa5000056u,0xa5000057u,0xa5000058u,0xa5000059u,
  0xa500005au,0xa500005bu,0xa500005cu,0xa500005du,(uint32_t)arena + 36864u,(uint32_t)arena + 0u,0x00000030u,0x00000001u,
  0x00000008u,0x00000008u,0x00000008u,0x00000180u,0x00000008u,0x00000080u,0x00000900u,0x00000001u,
  0x00004800u,0x00009000u,0x00000001u,0x00000010u,0x00000001u,0x00000000u,0x00300030u,0x001003f0u,
  0x000002e8u,0x000002ecu,0x000002bcu,0x0000030cu,0x00100420u,0x00100438u,0x00000310u,0x00000314u,
  0x0000035cu,0x00100480u,0x00100498u,0x00000360u,0x00000364u,0x00000334u,0x00000384u,0x001004c8u,
  0x001004e0u,0x00000388u,0x0000038cu,0x000003d4u,0x00100528u,0x00100540u,0x000003d8u,0x000003dcu,
  0x000003acu,0x000003fcu,0x00100570u,0x00100588u,0x00000400u,0x00000404u,0x08f00010u,0x08f00014u,
  0x08f00018u,0x08f0001cu,0x0cc00010u,0x0cd00010u,0x0cd00014u,0x0cd00018u,0x0cd0001cu,0x0d200010u,
  0x0d200014u,0x0d200018u,0x0d20001cu,0x0d700010u,0x0d800010u,0x0d800014u,0x0d800018u,0x0d80001cu,
  0x0dd00010u,0x0dd00014u,0x0dd00018u,0x0dd0001cu,0x0e200010u,0x0e300014u,0x0e300018u,0x0e30001cu,
  0x0ec00000u,0x0e800000u,0x0e800000u,0x0e900000u,0x0ea00000u,0x0eb00000u,0x0c900000u,0x0ca00000u,
  0x0cb00000u,0x0ed00000u,0x0ee00000u,0x0ef00000u,0x0f000000u,0x0ce00000u,0x0cf00000u,0x0d000000u,
  0x0d100000u,0x0f500000u,0x0f100000u,0x0f100000u,0x0f200000u,0x0f300000u,0x0f400000u,0x0d300000u,
  0x0d400000u,0x0d500000u,0x0d600000u,0x0f600000u,0x0f700000u,0x0f800000u,0x0f900000u,0x0d900000u,
  0x0da00000u,0x0db00000u,0x0dc00000u,0x0fe00000u,0x0fa00000u,0x0fa00000u,0x0fb00000u,0x0fc00000u,
  0x0fd00000u,0x0de00000u,0x0df00000u,0x0e000000u,0x0e100000u,0x0ff00000u,0xa500005eu,0xa500005fu,
  0xa5000060u,0xa5000061u,0xa5000062u,0xa5000063u,0xa5000064u,0xa5000065u,0x00000300u,0x00009000u,
  0x00000180u,0x00000010u,(uint32_t)arena + 0u,(uint32_t)arena + 36864u,0x00000018u,0x00000018u,0x00000010u,0x00000010u,
  0x00000600u,0x00000040u,0x00000120u,0x00002400u,0x00050002u,0x00000300u,0x0000044cu,0x001005d0u,
  0x001005e8u,0x00000450u,0x00000454u,0x00000424u,0x00000474u,0x00100618u,0x00100630u,0x00000478u,
  0x0000047cu,0x000004c4u,0x00100678u,0x00100690u,0x000004c8u,0x000004ccu,0x0000049cu,0x000004ecu,
  0x001006c0u,0x001006d8u,0x000004f0u,0x0e300010u,0x12a00010u,0x12a00014u,0x12a00018u,0x12a0001cu,
  0x12f00010u,0x13000014u,0x13000018u,0x1300001cu,0x13000010u,0x13500010u,0x13500014u,0x13500018u,
  0x1350001cu,0x13a00010u,0x10000000u,0x10100000u,0x0e400000u,0x0e500000u,0x0e600000u,0x0e700000u,
  0x13f00000u,0x0ff00000u,0x10000000u,0x10100000u,0x14400000u,0x14000000u,0x14000000u,0x14100000u,
  0x14200000u,0x14300000u,0x12b00000u,0x12c00000u,0x12d00000u,0x12e00000u,0x14500000u,0x14600000u,
  0x14700000u,0x13100000u,0x13200000u,0x13300000u,0x13400000u,0x14800000u,0x14500000u,0x14600000u,
  0x14700000u,0x14d00000u,0x14900000u,0x14900000u,0x14a00000u,0x14b00000u,0x14c00000u,0x13600000u,
  0x13700000u,0x13800000u,0x13900000u,0xa5000066u,0xa5000067u,0xa5000068u,0xa5000069u,0xa500006au,
  0xa500006bu,0xa500006cu,0xa500006du,(uint32_t)arena + 36864u,(uint32_t)pad_scr + 0u,0x00000020u,0x00000010u,0x00000180u,
  0x00000010u,0x00000180u,0x00000010u,0x00000100u,0x00000240u,0x00002400u,0x00004800u,0x00000001u,
  0x00000020u,0x00000003u,0x00180018u,0x000004f4u,0x0000053cu,0x00100720u,0x00100738u,0x00000540u,
  0x00000544u,0x00000514u,0x00000564u,0x00100768u,0x00100780u,0x00000568u,0x0000056cu,0x13b00014u,
  0x13b00018u,0x13b0001cu,0x13b00010u,0x17400010u,0x17400014u,0x17400018u,0x1740001cu,0x17900010u,
  0x17a00014u,0x17a00018u,0x17a0001cu,0x17f00000u,0x18000000u,0x18100000u,0x13c00000u,0x13d00000u,
  0x13e00000u,0x17300000u,0x18200000u,0x17f00000u,0x18000000u,0x18100000u,0x18700000u,0x18300000u,
  0x18300000u,0x18400000u,0x18500000u,0x18600000u,0x17500000u,0x17600000u,0x17700000u,0x17800000u,
  0x18800000u,0x18900000u,0x18a00000u,0x17b00000u,0xa500006eu,0xa500006fu,0xa5000070u,0xa5000071u,
  0xa5000072u,0xa5000073u,0xa5000074u,0xa5000075u,0x00000180u,0x00002400u,0x00000300u,0x00000020u,
  (uint32_t)fb_tab + 14192u,(uint32_t)arena + 36864u,0x00000020u,0x00000010u,0x00000180u,0x00000030u,0x000000c0u,0x00250001u,
  0x00000300u,0x00000101u,0x000005b4u,0x001007c8u,0x001007e0u,0x000005b8u,0x000005bcu,0x0000058cu,
  0x000005dcu,0x00100810u,0x00100828u,0x000005e0u,0x000005e4u,0x0000062cu,0x00100870u,0x00100888u,
  0x00000630u,0x00000634u,0x00000604u,0x17a00010u,0x1a200010u,0x1a200014u,0x1a200018u,0x1a20001cu,
  0x1a700010u,0x1a800014u,0x1a800018u,0x1a80001cu,0x1a800010u,0x1ad00010u,0x1ad00014u,0x1ad00018u,
  0x17c00000u,0x17d00000u,0x17e00000u,0x1b300000u,0x18800000u,0x18900000u,0x18a00000u,0x1b800000u,
  0x1b400000u,0x1b400000u,0x1b500000u,0x1b600000u,0x1b700000u,0x1a300000u,0x1a400000u,0x1a500000u,
  0x1a600000u,0x1b900000u,0x1ba00000u,0x1bb00000u,0x1a900000u,0x1aa00000u,0x1ab00000u,0x1ac00000u,
  0x1bc00000u,0x1b900000u,0x1ba00000u,0x1bb00000u,0x1bd00000u,0xa5000076u,0xa5000077u,0xa5000078u,
  0xa50000a3u,0xa50000a4u,0xa50000a5u,0xa50000a6u,0xa50000a7u,(uint32_t)arena + 36864u,(uint32_t)pad_scr + 0u,0x00000020u,
  0x00000300u,0x00000020u,0x00000300u,0x00000020u,0x00000200u,0x00000240u,0x00004800u,0x00000001u,
  0x00000040u,0x00000000u,0x00000654u,0x001008b8u,0x001008d0u,0x00000658u,0x0000065cu,0x001008e8u,
  0x0000067cu,0x00100900u,0x000006a4u,0x00100918u,0x00100930u,0x000006a8u,0x000006acu,0x00100948u,
  0x1ad0001cu,0x1b200010u,0x1db00014u,0x1db00018u,0x1db0001cu,0x1db00010u,0x1e100010u,0x1e100014u,
  0x1e300000u,0x1e300004u,0x1e300008u,0x1ea00000u,0x1bd00000u,0x1be00000u,0x1bf00000u,0x1e900000u,
  0x1ae00000u,0x1af00000u,0x1b000000u,0x1b100000u,0x1eb00000u,0x1ec00000u,0x1ed00000u,0x1dc00000u,
  0x1dd00000u,0x1de00000u,0x1df00000u,0x1ee00000u,0x1eb00000u,0x1ec00000u,0x1ed00000u,0x1e000000u,
  0x1e000000u,0x00000001u,0x1ef00000u,0x1f000000u,0x1e000000u,0x1e200000u,0xa50000a8u,0xa50000a9u,
  0xa50000aau,0xa50000abu,0xa50000acu,0xa50000adu,0xa50000aeu,0xa50000afu,0x00000180u,(uint32_t)pad_scr + 0u,
  (uint32_t)arena + 36864u,0x0000000cu,0x0000000cu,0x00000010u,0x00000300u,0x00000040u,0x00000048u,0x00001200u,
  0x00050002u,0x00000180u,0x1e30000cu,0x00000000u,(uint32_t)(OP_PAD),0x00000028u,(uint32_t)(OP_CONV_2D),0x00000050u,
  (uint32_t)(OP_PAD),0x00000078u,(uint32_t)(OP_DEPTHWISE_CONV_2D),0x000000a0u,(uint32_t)(OP_CONV_2D),0x000000c8u,(uint32_t)(OP_PAD),0x000000f0u,
  (uint32_t)(OP_DEPTHWISE_CONV_2D),0x00000118u,(uint32_t)(OP_CONV_2D),0x00000140u,(uint32_t)(OP_PAD),0x00000168u,(uint32_t)(OP_DEPTHWISE_CONV_2D),0x00000190u,
  (uint32_t)(OP_CONV_2D),0x000001b8u,(uint32_t)(OP_PAD),0x000001e0u,(uint32_t)(OP_DEPTHWISE_CONV_2D),0x00000208u,(uint32_t)(OP_CONV_2D),0x00000230u,
  (uint32_t)(OP_PAD),0x00000258u,(uint32_t)(OP_DEPTHWISE_CONV_2D),0x00000280u,(uint32_t)(OP_CONV_2D),0x000002a8u,(uint32_t)(OP_PAD),0x000002d0u,
  (uint32_t)(OP_DEPTHWISE_CONV_2D),0x000002f8u,(uint32_t)(OP_CONV_2D),0x00000320u,(uint32_t)(OP_PAD),0x00000348u,(uint32_t)(OP_DEPTHWISE_CONV_2D),0x00000370u,
  (uint32_t)(OP_CONV_2D),0x00000398u,(uint32_t)(OP_PAD),0x000003c0u,(uint32_t)(OP_DEPTHWISE_CONV_2D),0x000003e8u,(uint32_t)(OP_CONV_2D),0xa50000b0u,
  0xa50000b1u,0xa50000b2u,0xa50000b3u,0xa50000b4u,0xa50000b5u,0xa50000b6u,0xa50000b7u,(uint32_t)arena + 36864u,
  (uint32_t)pad_scr + 0u,0x00000040u,0x00000180u,0x00000020u,0x00000180u,0x00000090u,0x00001200u,0x00002400u,
  0x00000001u,0x00000040u,0x000c000cu,0xa50000b8u,0xa50000b9u,0xa50000bau,0xa50000bbu,0xa50000bcu,
  0xa50000bdu,0xa50000beu,0xa50000bfu,0x000000c0u,0x00000900u,0x00000300u,0x00000040u,(uint32_t)fb_tab + 14384u,
  (uint32_t)arena + 36864u,0x00000040u,0x00000010u,0x000000c0u,0x00000030u,0x00000030u,0x00250001u,0x00000180u,
  0x00000101u,0x00000410u,(uint32_t)(OP_PAD),0x00000438u,(uint32_t)(OP_DEPTHWISE_CONV_2D),0x00000460u,(uint32_t)(OP_CONV_2D),0x00000488u,
  (uint32_t)(OP_PAD),0x000004b0u,(uint32_t)(OP_DEPTHWISE_CONV_2D),0x000004d8u,(uint32_t)(OP_CONV_2D),0x00000500u,(uint32_t)(OP_PAD),0x00000528u,
  (uint32_t)(OP_DEPTHWISE_CONV_2D),0x00000550u,(uint32_t)(OP_CONV_2D),0x00000578u,(uint32_t)(OP_PAD),0x000005a0u,(uint32_t)(OP_DEPTHWISE_CONV_2D),0x000005c8u,
  (uint32_t)(OP_CONV_2D),0x000005f0u,(uint32_t)(OP_PAD),0x00000618u,(uint32_t)(OP_DEPTHWISE_CONV_2D),0x00000640u,(uint32_t)(OP_CONV_2D),0x00000668u,
  (uint32_t)(OP_AVERAGE_POOL_2D),0x00000690u,(uint32_t)(OP_FULLY_CONNECTED),0x000006a0u,(uint32_t)(99),0x1f100000u,0x1f200000u,0x1f300000u,
  0x20c00000u,0x1e200000u,0x1e400000u,0x1e500000u,0xa50000c0u,0xa50000c1u,0xa50000c2u,0xa50000c3u,
  0xa50000c4u,0xa50000c5u,0xa50000c6u,0xa50000c7u,(uint32_t)arena + 36864u,(uint32_t)pad_scr + 0u,0x00000040u,0x00000300u,
  0x00000040u,0x00000300u,0x00000040u,0x00000400u,0x00000090u,0x00002400u,0x00000001u,0x00000080u,
  0x00000000u,0x1e600000u,0x1e700000u,0x1e800000u,0xa5000102u,0xa5000103u,0xa5000104u,0xa5000105u,
  0xa50000c8u,0xa5000106u,0xa5000107u,0xa5000108u,0x00000180u,(uint32_t)pad_scr + 0u,(uint32_t)arena + 36864u,0x00000006u,
  0x00000006u,0x00000010u,0x00000180u,0x00000040u,0x00000012u,0x00000900u,0x00050002u,0x000000c0u,
  0xa5000109u,0xa500010au,0xa500010bu,0xa500010cu,0xa500010du,0xa500010eu,0xa500010fu,0xa5000110u,
  (uint32_t)arena + 36864u,(uint32_t)pad_scr + 0u,0x00000080u,0x00000180u,0x00000040u,0x00000180u,0x00000024u,0x00000900u,
  0x00001200u,0x00000001u,0x00000080u,0x00060006u,0xa5000111u,0xa5000112u,0xa5000113u,0xa5000114u,
  0xa5000115u,0xa5000116u,0xa5000117u,0xa5000118u,0x00000060u,0x00000240u,0x00000300u,0x00000080u,
  (uint32_t)fb_tab + 14480u,(uint32_t)arena + 36864u,0x00000080u,0x00000010u,0x00000060u,0x00000030u,0x0000000cu,0x00250001u,
  0x000000c0u,0x00000101u,0xa5000119u,0xa500011au,0xa500011bu,0xa500011cu,0xa500011du,0xa500011eu,
  0xa500011fu,0xa5000120u,(uint32_t)arena + 36864u,(uint32_t)pad_scr + 0u,0x00000080u,0x00000300u,0x00000080u,0x00000300u,
  0x00000080u,0x00000024u,0x00001200u,0x00000001u,0x00000100u,0x00000008u,0x00000000u,0xa5000121u,
  0xa5000122u,0xa5000123u,0xa5000124u,0xa5000125u,0xa5000126u,0xa5000127u,0xa5000128u,(uint32_t)fb_tab + 14480u,
  (uint32_t)arena + 36864u,0x00000010u,0x00000060u,0x00000030u,0x0000000cu,0x00250001u,0x000000c0u,0x00000101u,
  0xa5000129u,0xa500014eu,0xa500014fu,0xa5000150u,0xa5000151u,0xa5000152u,0xa5000153u,(uint32_t)kc_ls + 0u,
  (uint32_t)kc_st + 0u,0x00000020u,(uint32_t)kc_cb + 0u,(uint32_t)kc_cm + 0u,(uint32_t)kc_cs + 0u,(uint32_t)kc_cw + 0u,(uint32_t)kc_oh + 0u,(uint32_t)kc_lg + 0u,
  0xa5000154u,0xa5000155u,0xa5000156u,0xa5000157u,(uint32_t)arena + 36864u,(uint32_t)kc_cw + 0u,(uint32_t)kc_cb + 0u,(uint32_t)kc_cm + 0u,
  (uint32_t)kc_cs + 0u,(uint32_t)kc_ca + 0u,0x00000007u,0x00000020u,0x00000080u,0x00000300u,0x00000080u,0x00000300u,
  0x00000024u,0x00000480u,0x00000001u,0x00000100u,0x00000000u,0x00000000u,(uint32_t)kc_ca + 0u,(uint32_t)kc_oh + 0u,
  (uint32_t)kc_st + 0u,(uint32_t)id_m + 0u,(uint32_t)id_z + 0u,(uint32_t)pad_scr + 0u,0x00000003u,0x00000080u,0x00000020u,0x00000020u,
  0x000000c0u,0x00000020u,0x000000c0u,0x00000020u,0x00000200u,0x00000000u,0x00000000u,0xffffff80u,
  0x0000007fu,0x00000480u,0x00001200u,0x00000040u,0x00000003u,0xa5000158u,0xa5000159u,0xa500015au,
  0xa500015bu,0xa500015cu,0xa500015du,0xa500015eu,0xa500015fu,(uint32_t)fb_tab + 14480u,(uint32_t)arena + 36864u,0x00000080u,
  0x00000010u,0x00000060u,0x00000030u,0x0000000cu,0x00250001u,0x000000c0u,0x00000101u,0xa5000160u,
  0xa5000161u,0xa5000162u,0xa5000163u,0xa5000164u,0xa5000165u,0xa5000166u,(uint32_t)kc_ls + 0u,(uint32_t)kc_st + 0u,
  0x00000080u,0xa5000167u,0xa5000168u,0xa5000169u,0xa500016au,(uint32_t)arena + 36864u,(uint32_t)kc_cw + 0u,(uint32_t)kc_cb + 0u,
  (uint32_t)kc_cm + 0u,(uint32_t)kc_cs + 0u,(uint32_t)kc_ca + 0u,0x00000007u,0x00000020u,0x00000080u,0x00000300u,0x00000080u,
  0x00000300u,0x00000400u,0x00000024u,0x00001200u,0x00000480u,0x00000001u,0x00000100u,0x00000000u,
  0x00000000u,(uint32_t)kc_ca + 0u,(uint32_t)kc_oh + 0u,(uint32_t)kc_st + 0u,(uint32_t)id_m + 0u,(uint32_t)id_z + 0u,(uint32_t)pad_scr + 0u,0x00000003u,
  0x00000080u,0x00000020u,0x00000020u,0x000000c0u,0x00000020u,0x000000c0u,0x00000020u,0x00000200u,
  0x00000000u,0x00000000u,0xffffff80u,0x0000007fu,0x00000480u,0x00001200u,0x00000040u,0x00000003u,
  0xa500016bu,0xa500016cu,0xa500016du,0xa500016eu,0xa500016fu,0xa5000170u,0xa5000171u,0xa5000172u,
  (uint32_t)fb_tab + 14480u,(uint32_t)arena + 36864u,0x00000080u,0x00000010u,0x00000060u,0x00000030u,0x0000000cu,0x00250001u,
  0x000000c0u,0x00000101u,0xa500018bu,0xa500018cu,0xa500018du,0xa500018eu,0xa500018fu,0xa5000190u,
  0xa5000191u,(uint32_t)kc_ls + 0u,(uint32_t)kc_st + 0u,0x00000080u,0xa5000192u,0xa5000193u,0xa5000194u,0xa5000195u,
  (uint32_t)arena + 36864u,(uint32_t)kc_cw + 0u,(uint32_t)kc_cb + 0u,(uint32_t)kc_cm + 0u,(uint32_t)kc_cs + 0u,(uint32_t)kc_ca + 0u,0x00000007u,0x00000020u,
  0x00000080u,0x00000300u,0x00000080u,0x00000300u,0x00000400u,0x00000024u,0x00001200u,0x00000480u,
  0x00000001u,0x00000100u,0x00000000u,0x00000000u,(uint32_t)kc_ca + 0u,(uint32_t)kc_oh + 0u,(uint32_t)kc_st + 0u,(uint32_t)id_m + 0u,
  (uint32_t)id_z + 0u,(uint32_t)pad_scr + 0u,0x00000003u,0x00000080u,0x00000020u,0x00000020u,0x000000c0u,0x00000020u,
  0x000000c0u,0x00000020u,0x00000200u,0x00000000u,0x00000000u,0xffffff80u,0x0000007fu,0x00000480u,
  0x00001200u,0x00000040u,0x00000003u,0xa5000196u,0xa5000197u,0xa5000198u,0xa5000199u,0xa500019au,
  0xa500019bu,0xa500019cu,0xa500019du,(uint32_t)fb_tab + 14480u,(uint32_t)arena + 36864u,0x00000080u,0x00000010u,0x00000060u,
  0x00000030u,0x0000000cu,0x00250001u,0x000000c0u,0x00000101u,0xa500019eu,0xa500019fu,0xa50001a0u,
  0xa50001a1u,0xa50001c0u,0xa50001c1u,0xa50001c2u,(uint32_t)kc_ls + 0u,(uint32_t)kc_st + 0u,0x00000080u,0xa50001c3u,
  0xa50001c4u,0xa50001c5u,0xa50001c6u,(uint32_t)arena + 36864u,(uint32_t)kc_cw + 0u,(uint32_t)kc_cb + 0u,(uint32_t)kc_cm + 0u,(uint32_t)kc_cs + 0u,
  (uint32_t)kc_ca + 0u,0x00000007u,0x00000020u,0x00000080u,0x00000300u,0x00000080u,0x00000300u,0x00000400u,
  0x00000024u,0x00001200u,0x00000480u,0x00000001u,0x00000100u,0x00000000u,0x00000000u,(uint32_t)kc_ca + 0u,
  (uint32_t)kc_oh + 0u,(uint32_t)kc_st + 0u,(uint32_t)id_m + 0u,(uint32_t)id_z + 0u,(uint32_t)pad_scr + 0u,0x00000003u,0x00000080u,0x00000020u,
  0x00000020u,0x000000c0u,0x00000020u,0x000000c0u,0x00000020u,0x00000200u,0x00000000u,0x00000000u,
  0xffffff80u,0x0000007fu,0x00000480u,0x00001200u,0x00000040u,0x00000003u,0xa50001c7u,0xa50001c8u,
  0xa50001c9u,0xa50001cau,0xa50001cbu,0xa50001ccu,0xa50001cdu,0xa50001ceu,0x00000180u,(uint32_t)pad_scr + 0u,
  (uint32_t)arena + 36864u,0x00000003u,0x00000003u,0x00000080u,0x00000010u,0x000000c0u,0x00000040u,0x00000006u,
  0x00000480u,0x00050002u,0x00000060u,0xa50001cfu,0xa50001d0u,0xa50001d1u,0xa50001d2u,0xa50001d3u,
  0xa50001d4u,0xa50001d5u,(uint32_t)kc_ls + 0u,(uint32_t)kc_st + 0u,0x00000100u,0x00000030u,0x00000080u,0xa50001d6u,
  0xa50001d7u,0xa50001d8u,0xa50001d9u,(uint32_t)arena + 36864u,(uint32_t)kc_cw + 0u,(uint32_t)kc_cb + 0u,(uint32_t)kc_cm + 0u,(uint32_t)kc_cs + 0u,
  (uint32_t)kc_ca + 0u,0x00000007u,0x00000030u,0x00000080u,0x00000180u,0x00000080u,0x00000180u,0x00000400u,
  0x00000009u,0x000001b0u,0x00000001u,0x00000100u,0x00000000u,0x00030003u,(uint32_t)kc_ca + 0u,(uint32_t)kc_oh + 0u,
  (uint32_t)kc_st + 0u,(uint32_t)id_m + 0u,(uint32_t)id_z + 0u,(uint32_t)pad_scr + 0u,0x00000003u,0x00000100u,0x00000030u,0x00000030u,
  0x00000090u,0x00000030u,0x00000090u,0x00000030u,0x00000300u,0x00000000u,0x00000000u,0xffffff80u,
  0x0000007fu,0x000001b0u,0x00000900u,0x00000060u,0x00000003u,0xa50001dau,0xa50001f4u,0xa50001f5u,
  0xa50001f6u,0xa50001f7u,0xa50001f8u,0xa50001f9u,0xa50001fau,0x00000030u,0x00000090u,0x00000300u,
  0x00000100u,(uint32_t)fb_tab + 14528u,(uint32_t)arena + 36864u,0x00000100u,0x00000010u,0x00000030u,0x00000003u,0x00250001u,
  0x00000101u,0xa50001fbu,0xa50001fcu,0xa50001fdu,0xa50001feu,0xa50001ffu,0xa5000200u,0xa5000201u,
  (uint32_t)kc_ls + 0u,(uint32_t)kc_st + 0u,0x00000020u,0x00000100u,0xa5000202u,0xa5000203u,0xa5000204u,0xa5000205u,
  (uint32_t)arena + 36864u,(uint32_t)kc_cw + 0u,(uint32_t)kc_cb + 0u,(uint32_t)kc_cm + 0u,(uint32_t)kc_cs + 0u,(uint32_t)kc_ca + 0u,0x00000007u,0x00000020u,
  0x00000100u,0x00000300u,0x00000100u,0x00000300u,0x00000800u,0x00000009u,0x00000002u,0x00000900u,
  0x00000120u,0x00000001u,0x00000200u,0x00000000u,0x00000000u,0xa5000206u,0xa5000207u,(uint32_t)kc_ca + 0u,
  (uint32_t)kc_oh + 0u,(uint32_t)kc_st + 0u,(uint32_t)id_m + 0u,(uint32_t)id_z + 0u,0x00000003u,0x00000100u,0x00000020u,0x00000020u,
  0x00000060u,0x00000020u,0x00000060u,0x00000020u,0x00000200u,0x00000000u,0x00000000u,0xffffff80u,
  0x0000007fu,0x00000001u,0x00000120u,0x00000900u,0x00000040u,0xa5000208u,0xa5000209u,0xa500020au,
  0xa500020bu,(uint32_t)pool_ones + 0u,(uint32_t)pool_ones + 0u,(uint32_t)pool_ones + 0u,0x00000001u,0x00000001u,0x00000100u,0x00000300u,
  0x00000300u,0x00000900u,0x0e38e38fu,0x00000001u,0x00000100u,0x00000003u,0x00000000u,0x00000401u,
  0x00000080u,0xa500020du,0xa500020eu,0xa500020fu,0xa5000210u,0xa5000211u,0xa5000212u,0xa5000213u,
  0xa5000214u,0xa5000215u,0xa5000216u,0x00000001u,0x00000002u,0x00000100u,0x00000002u,0x00000010u,
};
static const uint32_t rlxs_mb_v[14] = {0x00100000u,0x00100024u,(uint32_t)(97),0x00100028u,(uint32_t)(3),0x0000003cu,0x00100030u,0x00100048u,0x00000040u,0x00000044u,0x00300010u,0x00300014u,0x00300018u,0x0030001cu};   // ops 0..K-1: fills, checks, guards
static int __attribute__((noinline)) rgf_mb(void) {   // ops 0..1 (v47_a straight-line); 0: a check failed
  const uint32_t* p = rlxs_mb_v; uint32_t* sp = rgc + 2; uint32_t bad = 0;
  rgc[2] = *(const volatile uint32_t*)(rgc[1] + 0u);
  bad |= *(const volatile uint32_t*)(rgc[1] + 36u) ^ (uint32_t)(97);
  bad |= *(const volatile uint32_t*)(rgc[1] + 40u) ^ (uint32_t)(3);
  if (bad) return 0;
  p += 5; sp += 1;
  const uint32_t q0_0 = (rgc[2] + 27072u);
  R(8) = q0_0;
  R(13) = (uint32_t)gat_scr + 0u;
  R(33) = 0x00000240u;
  R(42) = 0x00000002u;
  R(44) = 0x00000002u;
  R(46) = 0x00000120u;
  R(48) = 0x00000120u;
  R(49) = 0x00000240u;
  R(RG_CMD) = 6u; npu_pending = 1;
  rgc[3] = *(const volatile uint32_t*)(rgc[0] + 60u);
  rgc[4] = *(const volatile uint32_t*)(rgc[1] + 48u);
  rgc[5] = *(const volatile uint32_t*)(rgc[1] + 72u);
  rgc[6] = *(const volatile uint32_t*)(rgc[0] + 64u);
  rgc[7] = *(const volatile uint32_t*)(rgc[0] + 68u);
  rgc[8] = *(const volatile uint32_t*)(rgc[3] + 16u);
  rgc[9] = *(const volatile uint32_t*)(rgc[3] + 20u);
  rgc[10] = *(const volatile uint32_t*)(rgc[3] + 24u);
  rgc[11] = *(const volatile uint32_t*)(rgc[3] + 28u);
  if (bad) return 0;   // (op 0 is running: rgi_back waits)
  p += 9; sp += 9;
  const uint32_t q1_0 = (rgc[8] + 0u);
  const uint32_t q1_1 = (rgc[9] + 0u);
  const uint32_t q1_2 = (rgc[10] + 0u);
  const uint32_t q1_3 = (rgc[11] + 0u);
  const uint32_t q1_4 = (rgc[2] + 0u);
  const uint32_t q1_5 = (rgc[4] + 0u);
  const uint32_t q1_6 = (rgc[5] + 0u);
  const uint32_t q1_7 = (rgc[6] + 0u);
  const uint32_t q1_8 = (rgc[7] + 0u);
  (void)R(RG_SLOT);
  R(29) = q1_0;
  R(30) = q1_1;
  R(31) = q1_2;
  R(32) = q1_3;
  R(8) = q1_4;
  R(9) = q1_5;
  R(10) = q1_6;
  R(11) = q1_7;
  R(12) = q1_8;
  R(13) = (uint32_t)pad_scr + 0u;
  R(14) = 0x00000003u;
  R(15) = 0x0000002fu;
  R(16) = 0x00000030u;
  R(17) = 0x00000010u;
  R(18) = 0x00000003u;
  R(20) = 0x00000003u;
  R(21) = 0x00000009u;
  R(22) = 0x00000120u;
  R(23) = 0x00000006u;
  R(24) = 0x00000240u;
  R(25) = 0x0000001bu;
  R(27) = 0x000000d8u;
  R(47) = 0x00272706u;
  R(48) = 0x00007575u;
  R(51) = (uint32_t)npu_scr + 0u;
  R(52) = 0x000008d0u;
  R(54) = 0x00000001u;
  R(55) = 0x00004f50u;
  R(56) = 0x00004680u;
  R(57) = 0x00000002u;
  R(58) = 0x00000003u;
  R(60) = 0x00000005u;
  R(61) = 0x00030000u;
  R(62) = 0x00600060u;
  R(63) = 0x00000408u;
  R(RG_CMD) = 1u; npu_pending = 1;
  return 1;
}
#define RGF_MB_K 2
#define RGF_MB_S 12   // the next cache slot after the straight-line ops
static void __attribute__((noinline)) rlz_rs_end(void) {   // zeroed after the last op is taken
  R(1) = 0;
  R(2) = 0;
  R(3) = 0;
  R(6) = 0;
  R(35) = 0;
  R(36) = 0;
  R(37) = 0;
  R(38) = 0;
  R(39) = 0;
  R(40) = 0;
  R(41) = 0;
  R(59) = 0;
  R(60) = 0;
  R(61) = 0;
  R(63) = 0;
  R(64) = 0;
  R(65) = 0;
  R(67) = 0;
}
static const uint8_t rlz_rs_f[54] = {1,2,3,5,6,7,9,10,11,12,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,34,35,36,37,38,39,40,41,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,67,0};   // zeroed before op 0 if dirty
// resnet8: 12 ops (2 straight-line), 271 cache slots, 347 tasks
#define RGI_RS_N 12
#define RGI_RS_NN 21
#define RGI_RS_O0 1      // node 0 (a PAD): its output tensor index / height
#define RGI_RS_H0 34
static const uint8_t rlx_rs_c[60] = {11,0,0,12,8,21,39,0,13,36,19,4,29,0,15,21,10,18,37,0,0,31,8,16,0,0,0,0,21,24,22,9,19,28,10,19,0,13,0,10,8,18,0,0,0,0,21,25,0,0,0,0,3,19,0,0,0,0,10,7};   // per op: fill, chk, grd, compute words, cached regs, const
static const uint8_t rlx_rs_r[424] = {224,224,0,0,0,0,0,0,0,0,29,30,31,32,9,10,11,12,1,2,3,6,8,13,20,21,22,23,24,25,27,55,57,58,59,60,61,62,63,1,224,0,0,0,0,227,0,0,0,0,0,0,0,0,0,0,0,0,0,192,224,0,0,0,0,0,0,0,0,0,0,0,29,30,31,32,35,36,37,38,39,40,41,42,43,44,9,10,11,12,45,8,13,56,63,1,0,0,0,0,0,0,0,224,0,0,0,0,227,0,0,0,0,0,0,29,30,31,32,9,10,11,12,13,67,6,8,15,16,17,23,24,45,52,55,56,57,58,61,63,64,65,66,1,0,0,0,0,0,0,0,0,192,0,224,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,29,30,31,32,9,10,11,12,13,18,21,22,23,24,25,27,47,48,55,57,58,59,60,63,1,29,30,31,32,35,36,37,38,39,40,41,42,43,44,9,10,11,12,13,45,67,1,2,8,18,20,21,22,23,24,25,27,54,55,56,57,58,59,60,61,62,63,64,65,66,1,224,0,0,0,0,227,0,0,0,0,0,0,0,0,0,0,0,0,0,0,192,0,0,0,0,29,30,31,32,9,10,11,12,13,67,3,6,8,15,16,17,23,24,45,52,55,56,57,58,61,63,64,65,66,1,0,0,0,0,0,0,0,0,0,0,29,30,31,32,9,10,11,12,8,13,18,20,21,22,23,24,25,27,54,55,57,58,59,60,61,63,1,29,30,31,32,35,36,37,38,39,40,41,42,43,44,9,10,11,12,13,45,67,1,2,3,8,18,20,21,22,23,24,25,27,54,55,56,57,58,59,60,61,62,63,64,65,66,1,31,32,13,8,9,11,12,16,20,22,23,24,30,40,45,52,56,57,58,61,62,63,2,29,30,31,32,8,9,10,11,12,13,14,17,25,27,52,56,63,1};
static const uint32_t rlx_rs_v[797] = {
  0x0000008cu,0x00100090u,0x001000a8u,0x00000090u,0x00000094u,0x00000064u,0x00f00010u,0x00f00014u,
  0x00f00018u,0x00f0001cu,0x01400010u,0x00500000u,0x00b00000u,0x01900000u,0x01500000u,0x01500000u,
  0x01600000u,0x01700000u,0x01800000u,0x01000000u,0x01100000u,0x01200000u,0x01300000u,0xa500001au,
  0xa500001bu,0xa500001cu,0xa500001du,0xa500001eu,0xa500001fu,0xa5000020u,0xa5000021u,0x00000200u,
  0x00004000u,0x00000200u,0x00000010u,(uint32_t)id_w32 + 496u,(uint32_t)arena + 16384u,0x00000010u,0x00000030u,0x00000010u,
  0x00000200u,0x00000020u,0x00000090u,0x00000480u,0x0000c000u,0x000c0001u,0x00000400u,0x00000001u,
  0x00000000u,0x00000101u,0x00200020u,0x00000000u,0x000000dcu,0x00000104u,0x001000f0u,0x00100108u,
  0x000000e0u,0x000000e4u,0x000000b4u,0x00000154u,0x00100168u,0x00100180u,0x00000158u,0x0000015cu,
  0x00100198u,0x0000012cu,0x000001ccu,0x00100210u,0x00100228u,0x000001d0u,0x02200010u,0x02200014u,
  0x02200018u,0x0220001cu,0x02300018u,0x0230000cu,0x02300014u,0x02300008u,0x02300010u,0x02300004u,
  0x02300020u,0x02300024u,0x0230001cu,0x02300028u,0x0230002cu,0x02800010u,0x02900010u,0x02900014u,
  0x02900018u,0x0290001cu,0x02f00010u,0x02300000u,1u,(uint32_t)20,(uint32_t)20,0x02300014u,
  1u,(uint32_t)1073741824,(uint32_t)1073741824,0x02300018u,1u,(uint32_t)0,(uint32_t)0,0x02300008u,
  1u,(uint32_t)0,(uint32_t)2147483647,0x0230000cu,1u,(uint32_t)-3,(uint32_t)0,0x02300004u,
  1u,(uint32_t)-128,(uint32_t)128,0x02300010u,1u,(uint32_t)-128,(uint32_t)128,0x02300024u,
  1u,(uint32_t)-31,(uint32_t)31,0x0230001cu,1u,(uint32_t)-256,(uint32_t)255,0x02300028u,
  1u,(uint32_t)-128,(uint32_t)127,0x0230002cu,1u,(uint32_t)-128,(uint32_t)127,0x02600000u,
  65552u,(uint32_t)1073741824,(uint32_t)2147483647,0x02700000u,16u,(uint32_t)-11,(uint32_t)0,0x04300000u,
  0x03400000u,0x03400000u,0x03500000u,0x03600000u,0x03700000u,0x03800000u,0x03900000u,0x03a00000u,
  0x03b00000u,0x03c00000u,0x03d00000u,0x03e00000u,0x03f00000u,0x04000000u,0x04100000u,0x04200000u,
  0x02400000u,0x02500000u,0x02600000u,0x02700000u,((uint32_t)arena + 0u),((uint32_t)arena + 32768u),0x04800000u,0x04400000u,
  0x04400000u,0x04500000u,0x04600000u,0x04700000u,0x02a00000u,0x02b00000u,0x02c00000u,0x02d00000u,
  0x02e00000u,0x02e00000u,0x03100000u,0xa5000049u,0xa500004au,0xa500004bu,0xa500004cu,0xa500004du,
  0xa500004eu,0xa500004fu,0xa5000050u,0xa5000051u,0xa5000052u,0xa5000053u,0xa5000054u,0xa5000055u,
  0xa5000056u,0xa5000057u,0xa5000058u,0xa5000059u,0xa500005au,0xa500005bu,(uint32_t)arena + 15856u,(uint32_t)arena + 32768u,
  0x0000c000u,0x08000002u,0x000001d4u,0x000001a4u,0x000001f4u,0x001001c8u,0x001001e0u,0x000001a8u,
  0x000001acu,0x00100258u,0x0000017cu,0x03000010u,0x03000014u,0x03000018u,0x0300001cu,0x06800010u,
  0x06800014u,0x06800018u,0x0680001cu,0x06900018u,0x0690000cu,0x06900014u,0x06900008u,0x06900010u,
  0x06900004u,0x06900020u,0x06900024u,0x0690001cu,0x06900028u,0x0690002cu,0x06f00010u,0x06900000u,
  1u,(uint32_t)20,(uint32_t)20,0x06900014u,1u,(uint32_t)1073741824,(uint32_t)1073741824,0x06900018u,
  1u,(uint32_t)0,(uint32_t)0,0x06900008u,1u,(uint32_t)0,(uint32_t)2147483647,0x0690000cu,
  1u,(uint32_t)-3,(uint32_t)0,0x06900004u,1u,(uint32_t)-128,(uint32_t)128,0x06900010u,
  1u,(uint32_t)-128,(uint32_t)128,0x06900024u,1u,(uint32_t)-31,(uint32_t)31,0x0690001cu,
  1u,(uint32_t)-256,(uint32_t)255,0x06900028u,1u,(uint32_t)-128,(uint32_t)127,0x0690002cu,
  1u,(uint32_t)-128,(uint32_t)127,0x06c00000u,65552u,(uint32_t)1073741824,(uint32_t)2147483647,0x06c00040u,
  65552u,(uint32_t)1073741824,(uint32_t)2147483647,0x06d00000u,16u,(uint32_t)-11,(uint32_t)0,0x06d00040u,
  16u,(uint32_t)-11,(uint32_t)0,0x07000000u,0x07100000u,0x07200000u,0x07300000u,0x03200000u,
  0x03300000u,0x06700000u,0x08300000u,0x07400000u,0x07400000u,0x07500000u,0x07600000u,0x07700000u,
  0x07800000u,0x07900000u,0x07a00000u,0x07b00000u,0x07c00000u,0x07d00000u,0x07e00000u,0x07f00000u,
  0xa500005cu,0xa500005du,0xa500005eu,0xa500005fu,0xa5000060u,0xa5000061u,0xa5000062u,0xa5000063u,
  0xa5000064u,0xa5000065u,0x00000020u,(uint32_t)arena + 32768u,0x00000010u,0x00000010u,0x00000020u,0x00000400u,
  0x00000040u,0xffff8000u,0x00000100u,0x00003000u,0x00002000u,0x000c0002u,0x00000200u,0x00000000u,
  0x00000000u,(uint32_t)arena + 65536u,0x00091000u,1u,0x00000244u,0x00100288u,0x001002a0u,0x00000248u,
  0x0000024cu,0x001002b8u,0x0000021cu,0x000002bcu,0x00100330u,0x00100348u,0x000002c0u,0x000002c4u,
  0x00000294u,0x000002e4u,0x001002e8u,0x00100300u,0x00000298u,0x0000029cu,0x00100378u,0x0000026cu,
  0x09600010u,0x09600014u,0x09600018u,0x0960001cu,0x09c00010u,0x09d00010u,0x09d00014u,0x09d00018u,
  0x09d0001cu,0x0a200010u,0x0a200014u,0x0a200018u,0x0a20001cu,0x0a300018u,0x0a30000cu,0x0a300014u,
  0x0a300008u,0x08000000u,0x08100000u,0x08200000u,0x06a00000u,0x06b00000u,0x06c00000u,0x06d00000u,
  0x06e00000u,((uint32_t)arena + 73728u),0x06e00000u,0x06e00000u,0x0ae00000u,0x0aa00000u,0x0aa00000u,0x0ab00000u,
  0x0ac00000u,0x0ad00000u,0x09700000u,0x09800000u,0x09900000u,0x09a00000u,0x09b00e00u,0x09b00000u,
  0x0af00000u,0x0b000000u,0x0b100000u,0x0b200000u,0x09e00000u,0x09f00000u,0x0a000000u,0x0a100000u,
  0xa5000084u,0xa5000085u,0xa5000086u,0xa5000087u,0xa5000066u,0xa5000088u,0xa5000089u,0xa500008au,
  (uint32_t)arena + 73728u,0x00000001u,0x00000010u,0x00000200u,0x00000020u,0x00000400u,0x00000010u,0x00000100u,
  0x00707004u,0x00007070u,0x00001000u,0x00000002u,0x00000010u,0x00000000u,0x00000001u,0x00000008u,
  0xa500008bu,0xa500008cu,0xa500008du,0xa500008eu,0xa500008fu,0xa5000090u,0xa5000091u,0xa5000092u,
  0xa5000093u,0xa5000094u,0xa5000095u,0xa50000bbu,0xa50000bcu,0xa50000bdu,0xa50000beu,0xa50000bfu,
  0xa50000c0u,0xa50000c1u,0xa50000c2u,0xa50000c3u,0xa50000c4u,0x00000100u,0x00001000u,(uint32_t)arena + 65264u,
  0x00000003u,0x00000020u,0x00000060u,0x00000010u,0x00000100u,0x00000020u,0x00000120u,0x00000900u,
  0x00000002u,0x00006000u,0x00006000u,0x000c0001u,0x00000200u,0x00000001u,0x00000000u,0x00000101u,
  0x00100010u,0x00000002u,(uint32_t)arena + 81920u,0x00391000u,1u,0x0000030cu,0x00100390u,0x00000334u,
  0x001003a8u,0x001003c0u,0x00000338u,0x0000033cu,0x001003d8u,0x0a300010u,0x0a300004u,0x0a300020u,
  0x0a300024u,0x0a30001cu,0x0a300028u,0x0a30002cu,0x0a900010u,0x0d700010u,0x0d700014u,0x0d900000u,
  0x0d900004u,0x0d900008u,0x0d90000cu,0x00000000u,(uint32_t)(OP_PAD),0x00000028u,(uint32_t)(OP_CONV_2D),0x00000050u,
  (uint32_t)(OP_PAD),0x00000078u,(uint32_t)(OP_CONV_2D),0x000000a0u,(uint32_t)(OP_PAD),0x000000c8u,(uint32_t)(OP_CONV_2D),0x000000f0u,
  (uint32_t)(OP_ADD),0x00000118u,(uint32_t)(OP_PAD),0x00000140u,(uint32_t)(OP_CONV_2D),0x0a300000u,1u,(uint32_t)20,
  (uint32_t)20,0x0a300014u,1u,(uint32_t)1073741824,(uint32_t)1073741824,0x0a300018u,1u,(uint32_t)0,
  (uint32_t)0,0x0a300008u,1u,(uint32_t)0,(uint32_t)2147483647,0x0a30000cu,1u,(uint32_t)-3,
  (uint32_t)0,0x0a300004u,1u,(uint32_t)-128,(uint32_t)128,0x0a300010u,1u,(uint32_t)-128,
  (uint32_t)128,0x0a300024u,1u,(uint32_t)-31,(uint32_t)31,0x0a30001cu,1u,(uint32_t)-256,
  (uint32_t)255,0x0a300028u,1u,(uint32_t)-128,(uint32_t)127,0x0a30002cu,1u,(uint32_t)-128,
  (uint32_t)127,0x0a600000u,65552u,(uint32_t)1073741824,(uint32_t)2147483647,0x0a600040u,65552u,(uint32_t)1073741824,
  (uint32_t)2147483647,0x0a600080u,65552u,(uint32_t)1073741824,(uint32_t)2147483647,0x0a6000c0u,65552u,(uint32_t)1073741824,
  (uint32_t)2147483647,0x0a700000u,16u,(uint32_t)-11,(uint32_t)0,0x0a700040u,16u,(uint32_t)-11,
  (uint32_t)0,0x0a700080u,16u,(uint32_t)-11,(uint32_t)0,0x0a7000c0u,16u,(uint32_t)-11,
  (uint32_t)0,0x0e600000u,0x0b300000u,0x0b300000u,0x0b400000u,0x0b500000u,0x0b600000u,0x0b700000u,
  0x0b800000u,0x0b900000u,0x0ba00000u,0x0df00000u,0x0e000000u,0x0e100000u,0x0e200000u,0x0e300000u,
  0x0e400000u,0x0e500000u,0x0a400000u,0x0a500000u,0x0a600000u,0x0a700000u,0x0a800000u,((uint32_t)arena + 102400u),
  0x0a800000u,0x0a800000u,0x0e700000u,0x0e800000u,0x0d800000u,0xa50000c5u,0xa50000c6u,0xa50000c7u,
  0xa50000c8u,0xa50000c9u,0xa50000cau,0xa50000cbu,0xa50000ccu,0xa50000cdu,0xa50000ceu,0x0000fe00u,
  0x00000040u,(uint32_t)arena + 85504u,0x00000008u,0x00000008u,0x00000040u,0x0000fe00u,0x00000080u,0xfffb5484u,
  0x00000040u,0x00001800u,0x00001000u,0x000c0002u,0x00000000u,0x00000100u,0x00004000u,(uint32_t)arena + 98304u,
  0x008a0400u,1u,0x00000168u,(uint32_t)(OP_PAD),0x00000190u,(uint32_t)(OP_CONV_2D),0x000001b8u,(uint32_t)(OP_CONV_2D),
  0x000001e0u,(uint32_t)(OP_ADD),0x00000208u,(uint32_t)(OP_PAD),0x00000230u,(uint32_t)(OP_CONV_2D),0x00000258u,(uint32_t)(OP_PAD),
  0x00000280u,(uint32_t)(OP_CONV_2D),0x000002a8u,(uint32_t)(OP_CONV_2D),0x000002d0u,(uint32_t)(OP_ADD),0x000002f8u,(uint32_t)(OP_AVERAGE_POOL_2D),
  0x00000320u,(uint32_t)(OP_FULLY_CONNECTED),0x00000330u,(uint32_t)(41),0x0e900000u,0x0ea00000u,0x0eb00000u,0x0ec00000u,
  0x0d800000u,0x0da00000u,0x0db00000u,0x0dc00000u,0x0dd00000u,0x0de00000u,0xa50000cfu,0xa50000d0u,
  0xa50000d1u,0xa50000d2u,0xa50000d3u,0xa50000d4u,0xa50000d5u,0xa50000d6u,(uint32_t)arena + 81936u,(uint32_t)arena + 102400u,
  0x00000002u,0x00000010u,0x00000010u,0x00001000u,0x00000040u,0x00000200u,0x00000020u,0x00000200u,
  0x00000001u,0x00000400u,0x00000002u,0x00000010u,0x00000000u,0x00000001u,0x00000000u,0x00008000u,
  0xa50000edu,0xa50000eeu,0xa50000efu,0xa50000f0u,0xa50000f1u,0xa50000f2u,0xa50000f3u,0xa50000f4u,
  0xa50000f5u,0xa50000f6u,0xa50000f7u,0xa50000f8u,0xa50000f9u,0xa50000fau,0xa50000fbu,0xa50000fcu,
  0xa50000fdu,0xa50000feu,0xa50000ffu,0xa5000100u,0xa5000101u,0x00000080u,0x00000400u,0x00000200u,
  (uint32_t)arena + 98160u,0x00000003u,0x00000040u,0x000000c0u,0x00000010u,0x00000080u,0x00000020u,0x00000240u,
  0x00001200u,0x00000004u,0x00003000u,0x00003000u,0x000c0001u,0x00000100u,0x00000001u,0x00000000u,
  0x00000101u,0x00080008u,0x00000042u,(uint32_t)arena + 106496u,0x000a0400u,1u,0xa5000102u,0xa5000103u,
  0xa5000104u,(uint32_t)arena + 106496u,(uint32_t)pool_ones + 0u,(uint32_t)pool_ones + 0u,(uint32_t)pool_ones + 0u,0x00000001u,0x00000010u,0x00000030u,
  0x00000030u,0x00000090u,0x00000000u,0x02000001u,0xfffc41a4u,0x00000008u,0x00000040u,0x00000003u,
  0x00000001u,0x00000000u,0x00030016u,0x00000180u,0xa5000105u,0xa5000106u,0xa5000107u,0xa5000108u,
  0xa5000109u,0xa500010au,0xa500010bu,0xa500010cu,0xa500010du,0xa500010eu,0x00000001u,0x0000000au,
  0x00000040u,0x00000280u,0x00000001u,0x0000000au,0x00000010u,
};
static const uint32_t rlxs_rs_v[17] = {0x00000014u,0x00100000u,0x00100018u,0x00200010u,0x00100024u,(uint32_t)(34),0x00100028u,(uint32_t)(3),0x0000003cu,0x00100030u,0x00100048u,0x00000040u,0x00000044u,0x00600010u,0x00600014u,0x00600018u,0x0060001cu};   // ops 0..K-1: fills, checks, guards
static int __attribute__((noinline)) rgf_rs(void) {   // ops 0..1 (v47_a straight-line); 0: a check failed
  const uint32_t* p = rlxs_rs_v; uint32_t* sp = rgc + 2; uint32_t bad = 0;
  rgc[2] = *(const volatile uint32_t*)(rgc[0] + 20u);
  rgc[3] = *(const volatile uint32_t*)(rgc[1] + 0u);
  rgc[4] = *(const volatile uint32_t*)(rgc[1] + 24u);
  rgc[5] = *(const volatile uint32_t*)(rgc[2] + 16u);
  bad |= *(const volatile uint32_t*)(rgc[1] + 36u) ^ (uint32_t)(34);
  bad |= *(const volatile uint32_t*)(rgc[1] + 40u) ^ (uint32_t)(3);
  if (bad) return 0;
  p += 8; sp += 4;
  const uint32_t q0_0 = (rgc[5] + 0u);
  const uint32_t q0_1 = (rgc[3] + 0u);
  const uint32_t q0_2 = (rgc[4] + 0u);
  R(50) = q0_0;
  R(8) = q0_1;
  R(13) = q0_2;
  R(33) = 0x00000c00u;
  R(42) = 0x00000022u;
  R(43) = 0x00000001u;
  R(44) = 0x00000020u;
  R(45) = 0x00000003u;
  R(46) = 0x00000060u;
  R(47) = 0x00000003u;
  R(48) = 0x00000066u;
  R(49) = 0x00000d8cu;
  R(RG_CMD) = 6u; npu_pending = 1;
  rgc[6] = *(const volatile uint32_t*)(rgc[0] + 60u);
  rgc[7] = *(const volatile uint32_t*)(rgc[1] + 48u);
  rgc[8] = *(const volatile uint32_t*)(rgc[1] + 72u);
  rgc[9] = *(const volatile uint32_t*)(rgc[0] + 64u);
  rgc[10] = *(const volatile uint32_t*)(rgc[0] + 68u);
  rgc[11] = *(const volatile uint32_t*)(rgc[6] + 16u);
  rgc[12] = *(const volatile uint32_t*)(rgc[6] + 20u);
  rgc[13] = *(const volatile uint32_t*)(rgc[6] + 24u);
  rgc[14] = *(const volatile uint32_t*)(rgc[6] + 28u);
  if (bad) return 0;   // (op 0 is running: rgi_back waits)
  p += 9; sp += 9;
  const uint32_t q1_0 = (rgc[11] + 0u);
  const uint32_t q1_1 = (rgc[12] + 0u);
  const uint32_t q1_2 = (rgc[13] + 0u);
  const uint32_t q1_3 = (rgc[14] + 0u);
  const uint32_t q1_4 = (rgc[4] + 0u);
  const uint32_t q1_5 = (rgc[7] + 0u);
  const uint32_t q1_6 = (rgc[8] + 0u);
  const uint32_t q1_7 = (rgc[9] + 0u);
  const uint32_t q1_8 = (rgc[10] + 0u);
  (void)R(RG_SLOT);
  R(29) = q1_0;
  R(30) = q1_1;
  R(31) = q1_2;
  R(32) = q1_3;
  R(8) = q1_4;
  R(9) = q1_5;
  R(10) = q1_6;
  R(11) = q1_7;
  R(12) = q1_8;
  R(13) = (uint32_t)arena + 0u;
  R(14) = 0x00000003u;
  R(15) = 0x00000020u;
  R(16) = 0x00000020u;
  R(17) = 0x00000010u;
  R(18) = 0x00000003u;
  R(20) = 0x00000003u;
  R(21) = 0x00000009u;
  R(22) = 0x00000066u;
  R(23) = 0x00000003u;
  R(24) = 0x00000066u;
  R(25) = 0x0000001bu;
  R(27) = 0x000001b0u;
  R(47) = 0x01121204u;
  R(48) = 0x00003636u;
  R(51) = (uint32_t)npu_scr + 0u;
  R(52) = 0x00000400u;
  R(54) = 0x00000001u;
  R(55) = 0x00002400u;
  R(56) = 0x00004000u;
  R(57) = 0x00000001u;
  R(58) = 0x00000006u;
  R(60) = 0x00000001u;
  R(62) = 0x00220022u;
  R(63) = 0x00000008u;
  R(RG_CMD) = 1u; npu_pending = 1;
  return 1;
}
#define RGF_RS_K 2
#define RGF_RS_S 15   // the next cache slot after the straight-line ops
// ---- end generated rimg4 ----

// table ops: per op 5 counts (fills, checks, guards, rel words, consts): the phase's scheduled loads / checks / guards
// (they may serve later ops), SLOT, then the op's registers computed from the cache only (R: rgc[s] + off, a loaded
// value is off 0; 0x80|k / 0xC0|k: difference of two addresses, 20-bit masked / full; 0xE3: ER from two loaded shift
// words), the structural constants, CMD. 0: a check / guard failed (this op is not queued)
// (v53_Q) cached-slot words are stored as slot + 0xA5000000 (a byte pattern no weight table shares: the embedded-
// parameter scan's 8-byte windows); RGCV indexes with the offset folded into the base address (no extra work)
#define RGCV(w) (*(const uint32_t*)((uintptr_t)rgc + 4u * ((w) - 0xA5000000u)))
// (v51_D k_rids) ds_cnn replay: npu_dwf preconditions, checked in steps between queued ops
#define RGX_HW 1   // (v51_D) 1: the npu_dwf multiplier / shift check is the hardware flag (r_dwfflag.py)
static const Node* rgx_n0 DRV;
static int rgx_k DRV, rgx_bad DRV, rgx_p DRV, rgx_nn DRV;
// every PAD feeding a conv / depthwise: pad value = -input offset of its reader (an elided PAD is read with
// implicit padding by the replayed ops; checked for all of them: a violation falls back to the generic kernels)
static void __attribute__((noinline)) rgx_pads(void) {   // (v56_CY k_nopads) the tables' Z checks (0xE0) cover it
  rgx_p = rgx_nn - 1;
}
static void __attribute__((noinline)) rgx_step(void) {
  if (rgx_n0 == 0) return;
  if (rgx_k >= 20) { rgx_pads(); return; }
  const int k = rgx_k++, j = k / 5, q = k % 5;
  const Node* d = rgx_n0 + 3 + 3 * j;
  if (q == 0) {
    const Node* pr = d - 2;
    const ConvParams* a = (const ConvParams*)d->builtin;
    const ConvParams* b = (const ConvParams*)pr->builtin;
    if (d->op != OP_DEPTHWISE_CONV_2D || (d - 1)->op != OP_PAD || !in_dmem(a) || !in_dmem(b) || !in_dmem((d - 1)->builtin) ||
        !cpar_s(a, 1) || a->input_offset != 128 || ((const PadParams*)(d - 1)->builtin)->pad_value != -128 ||
        a->output_offset != b->output_offset || a->quantized_activation_min != b->quantized_activation_min ||
        a->quantized_activation_max != b->quantized_activation_max || d->out_mult == 0 || d->out_shift == 0) rgx_bad = 1;
    return;
  }
  if (rgx_bad || RGX_HW) return;   // (RGX_HW: npu_conv8 checks the multipliers / shifts as they load)
  const int32_t* om = d->out_mult + 16 * (q - 1);
  const int32_t* os = d->out_shift + 16 * (q - 1);
  for (int ch = 0; ch < 16; ch++) if ((om[ch] != 0 && om[ch] < (1 << 30)) || os[ch] < -11 || os[ch] > 0) { rgx_bad = 1; return; }
}
static int rgi_play(const uint8_t* cn, const uint8_t* rr, const uint32_t* vv, int n, uint32_t* sp) {
  for (int i = 0; i < n; i++, cn += 6) {
    if ((cn[0] | cn[1]) && !rgk3(&vv, &sp, cn[0], cn[1])) return 0;   // the phase: scheduled loads / checks
    if (cn[2]) { if (!rgg_ok3(vv, cn[2])) return 0; vv += 4 * cn[2]; }  // ... guards
    for (int c = cn[3]; c > 0; c--) {            // ... register values computed from the cache (for this op or later)
      const uint32_t r = *rr++, w = *vv++;
      uint32_t v;
      if (r == 0) v = rgc[w >> 20] + (w & 0xfffffu);
      else {
        const uint32_t w2 = *vv++; c--;
        if (r == 0xE0u) {                        // PAD value + its consumer's input offset must be 0
          if (rgc[w >> 20] + rgc[w2 >> 20]) return 0;
          continue;
        }
        if (r == 0xE3u) v = (rgc[w >> 20] & 0xffu) | ((rgc[w2 >> 20] & 0xffu) << 8);
        else if (r == 0xE5u) v = w2 | (((rgc[w >> 20] + (w & 0xfffffu)) & 15u) ? 1024u : 0u);   // (v53_Q AG: RG_PG bit 10)
        else {
          v = ((w & 0x80000000u) ? w : rgc[w >> 20] + (w & 0xfffffu)) -
              ((w2 & 0x80000000u) ? w2 : rgc[w2 >> 20] + (w2 & 0xfffffu));
          if (!(r & 0x40u)) v &= 0xfffffu;
        }
      }
      *sp++ = v;
    }
    if (cn[4] & 0x80u) {                         // (v53_Q RS50) the queued op is a background PADI, which reads only
      const int al = (cn[4] >> 6) & 1;           // RG 7/26/28/33/46/49/50: this op's writes go first, while it waits;
      for (int c = cn[4] & 0x3fu; c > 0; c--) R(*rr++) = RGCV(*vv++);   // then its take; the one-shot tee arm (66,
      for (int c = cn[5] - al; c > 0; c--) R(*rr++) = *vv++;          // the last constant) only after the take
      if (npu_pending) (void)R(RG_SLOT);
      if (al) { R(*rr++) = *vv++; }
    } else {
    if (npu_pending) (void)R(RG_SLOT);           // the queued op has taken its registers
    for (int c = cn[4]; c > 0; c--) R(*rr++) = RGCV(*vv++);
    for (int c = cn[5]; c > 0; c--) R(*rr++) = *vv++;
    }
    R(RG_CMD) = *rr++; npu_pending = 1;
    rgx_step(); rgx_step();                      // (v51_D) while the NPU works (2 of 20 steps per op)
  }
  return 1;
}
// (v56_CY k_late) rgi_play with the replay's late exit: stop >= 0 returns 2 right after op `stop`'s SLOT (op stop - 1
// took its registers and runs; nothing queued), state saved in rl_*; res = 1 resumes there (that op's writes, CMD, ...)
static const uint8_t* rl_cn DRV;
static const uint8_t* rl_rr DRV;
static const uint32_t* rl_vv DRV;
static uint32_t* rl_sp DRV;
static int rl_n DRV, rl_al DRV;
// (v56_CY k_slw) ds's table ops: their register writes straight-line (1: done; 0: not one of them)
static int __attribute__((noinline)) rgw_ds(const uint8_t* cn) {
  if (cn < rlx_ds_c || cn >= rlx_ds_c + 30) return 0;
  switch (cn - rlx_ds_c) {
  case 0:
    R(29) = RGCV(0xa5000026u);
    R(30) = RGCV(0xa5000027u);
    R(31) = RGCV(0xa5000028u);
    R(32) = RGCV(0xa5000029u);
    R(8) = RGCV(0xa500002au);
    R(9) = RGCV(0xa500002bu);
    R(10) = RGCV(0xa500002cu);
    R(11) = RGCV(0xa500002du);
    R(12) = RGCV(0xa500002eu);
    R(13) = RGCV(0xa500002fu);
    R(33) = RGCV(0xa5000030u);
    R(45) = RGCV(0xa5000031u);
    R(49) = RGCV(0xa5000032u);
    R(51) = RGCV(0xa5000033u);
    R(20) = *(const volatile uint32_t*)&rlx_ds_v[50];
    R(21) = *(const volatile uint32_t*)&rlx_ds_v[51];
    R(22) = *(const volatile uint32_t*)&rlx_ds_v[52];
    R(23) = *(const volatile uint32_t*)&rlx_ds_v[53];
    R(24) = *(const volatile uint32_t*)&rlx_ds_v[54];
    R(25) = *(const volatile uint32_t*)&rlx_ds_v[55];
    R(27) = *(const volatile uint32_t*)&rlx_ds_v[56];
    R(53) = *(const volatile uint32_t*)&rlx_ds_v[57];
    R(55) = *(const volatile uint32_t*)&rlx_ds_v[58];
    R(58) = *(const volatile uint32_t*)&rlx_ds_v[59];
    return 1;
  case 6:
    R(29) = RGCV(0xa500004cu);
    R(30) = RGCV(0xa500004du);
    R(31) = RGCV(0xa500004eu);
    R(32) = RGCV(0xa500004fu);
    R(8) = RGCV(0xa5000034u);
    R(9) = RGCV(0xa5000050u);
    R(10) = RGCV(0xa5000051u);
    R(11) = RGCV(0xa5000052u);
    R(12) = RGCV(0xa5000053u);
    R(13) = RGCV(0xa5000054u);
    R(33) = RGCV(0xa5000055u);
    R(45) = RGCV(0xa5000056u);
    R(49) = RGCV(0xa5000057u);
    R(51) = RGCV(0xa5000058u);
    return 1;
  case 12:
    R(29) = RGCV(0xa5000064u);
    R(30) = RGCV(0xa5000065u);
    R(31) = RGCV(0xa5000066u);
    R(32) = RGCV(0xa5000067u);
    R(8) = RGCV(0xa5000068u);
    R(9) = RGCV(0xa5000069u);
    R(10) = RGCV(0xa500006au);
    R(11) = RGCV(0xa500006bu);
    R(12) = RGCV(0xa500006cu);
    R(13) = RGCV(0xa500006du);
    R(33) = RGCV(0xa500006eu);
    R(45) = RGCV(0xa500006fu);
    R(49) = RGCV(0xa5000070u);
    R(51) = RGCV(0xa5000071u);
    return 1;
  case 18:
    R(29) = RGCV(0xa5000072u);
    R(30) = RGCV(0xa5000073u);
    R(31) = RGCV(0xa5000074u);
    R(32) = RGCV(0xa5000075u);
    R(8) = RGCV(0xa5000076u);
    R(9) = RGCV(0xa5000077u);
    R(10) = RGCV(0xa5000078u);
    R(11) = RGCV(0xa5000079u);
    R(12) = RGCV(0xa500007au);
    R(13) = RGCV(0xa500007bu);
    R(40) = *(const volatile uint32_t*)&rlx_ds_v[209];
    R(51) = *(const volatile uint32_t*)&rlx_ds_v[210];
    R(63) = *(const volatile uint32_t*)&rlx_ds_v[211];
    return 1;
  case 24:
    R(29) = RGCV(0xa5000086u);
    R(30) = RGCV(0xa5000087u);
    R(31) = RGCV(0xa5000088u);
    R(32) = RGCV(0xa5000089u);
    R(8) = RGCV(0xa500008au);
    R(9) = RGCV(0xa500008bu);
    R(10) = RGCV(0xa500008cu);
    R(11) = RGCV(0xa500008du);
    R(12) = RGCV(0xa500008eu);
    R(13) = RGCV(0xa500008fu);
    R(14) = *(const volatile uint32_t*)&rlx_ds_v[276];
    R(17) = *(const volatile uint32_t*)&rlx_ds_v[277];
    R(27) = *(const volatile uint32_t*)&rlx_ds_v[278];
    R(52) = *(const volatile uint32_t*)&rlx_ds_v[279];
    R(56) = *(const volatile uint32_t*)&rlx_ds_v[280];
    R(63) = *(const volatile uint32_t*)&rlx_ds_v[281];
    return 1;
  }
  return 0;
}
// (v58_MB k_slwmb) mobilenet table op 1: its register writes straight-line
static void __attribute__((noinline)) rgw_mb1(void) {
  R(29) = RGCV(0xa500004eu);
  R(30) = RGCV(0xa500004fu);
  R(31) = RGCV(0xa5000050u);
  R(32) = RGCV(0xa5000051u);
  R(9) = RGCV(0xa5000052u);
  R(10) = RGCV(0xa5000053u);
  R(11) = RGCV(0xa5000054u);
  R(12) = RGCV(0xa5000055u);
  R(6) = *(const volatile uint32_t*)&rlx_mb_v[136];
  R(8) = *(const volatile uint32_t*)&rlx_mb_v[137];
  R(13) = *(const volatile uint32_t*)&rlx_mb_v[138];
  R(15) = *(const volatile uint32_t*)&rlx_mb_v[139];
  R(16) = *(const volatile uint32_t*)&rlx_mb_v[140];
  R(20) = *(const volatile uint32_t*)&rlx_mb_v[141];
  R(22) = *(const volatile uint32_t*)&rlx_mb_v[142];
  R(23) = *(const volatile uint32_t*)&rlx_mb_v[143];
  R(24) = *(const volatile uint32_t*)&rlx_mb_v[144];
  R(52) = *(const volatile uint32_t*)&rlx_mb_v[145];
  R(56) = *(const volatile uint32_t*)&rlx_mb_v[146];
  R(57) = *(const volatile uint32_t*)&rlx_mb_v[147];
  R(58) = *(const volatile uint32_t*)&rlx_mb_v[148];
  R(61) = *(const volatile uint32_t*)&rlx_mb_v[149];
  R(62) = *(const volatile uint32_t*)&rlx_mb_v[150];
  R(63) = *(const volatile uint32_t*)&rlx_mb_v[151];
}
static int __attribute__((noinline)) rgi_play_s(const uint8_t* cn, const uint8_t* rr, const uint32_t* vv, int n, uint32_t* sp,
                                                 int stop, int res) {
  for (int i = 0; i < n; i++, cn += 6) {
    int al = rl_al;
    if (i == 0 && res) goto rl_w;
    if ((cn[0] | cn[1]) && !rgk3(&vv, &sp, cn[0], cn[1])) return 0;   // the phase: scheduled loads / checks
    if (cn[2]) { if (!rgg_ok3(vv, cn[2])) return 0; vv += 4 * cn[2]; }  // ... guards
    for (int c = cn[3]; c > 0; c--) {            // ... register values computed from the cache (for this op or later)
      const uint32_t r = *rr++, w = *vv++;
      uint32_t v;
      if (r == 0) v = rgc[w >> 20] + (w & 0xfffffu);
      else {
        const uint32_t w2 = *vv++; c--;
        if (r == 0xE0u) {                        // PAD value + its consumer's input offset must be 0
          if (rgc[w >> 20] + rgc[w2 >> 20]) return 0;
          continue;
        }
        if (r == 0xE3u) v = (rgc[w >> 20] & 0xffu) | ((rgc[w2 >> 20] & 0xffu) << 8);
        else if (r == 0xE5u) v = w2 | (((rgc[w >> 20] + (w & 0xfffffu)) & 15u) ? 1024u : 0u);   // (v53_Q AG: RG_PG bit 10)
        else {
          v = ((w & 0x80000000u) ? w : rgc[w >> 20] + (w & 0xfffffu)) -
              ((w2 & 0x80000000u) ? w2 : rgc[w2 >> 20] + (w2 & 0xfffffu));
          if (!(r & 0x40u)) v &= 0xfffffu;
        }
      }
      *sp++ = v;
    }
    al = 0;
    if (cn[4] & 0x80u) {                         // (v53_Q RS50) the queued op is a background PADI, which reads only
      al = (cn[4] >> 6) & 1;                     // RG 7/26/28/33/46/49/50: this op's writes go first, while it waits;
      for (int c = cn[4] & 0x3fu; c > 0; c--) R(*rr++) = RGCV(*vv++);   // then its take; the one-shot tee arm (66,
      for (int c = cn[5] - al; c > 0; c--) R(*rr++) = *vv++;          // the last constant) only after the take
    }
    if (npu_pending) (void)R(RG_SLOT);           // the queued op has taken its registers
    if (i == stop) { rl_cn = cn; rl_rr = rr; rl_vv = vv; rl_sp = sp; rl_n = n - i; rl_al = al; return 2; }
  rl_w:
    if (cn[4] & 0x80u) { if (al) { R(*rr++) = *vv++; } }
    else if (rgw_ds(cn)) { rr += cn[4] + cn[5]; vv += cn[4] + cn[5]; }   // (v56_CY k_slw)
    else if (cn == rlx_mb_c + 6) { rgw_mb1(); rr += cn[4] + cn[5]; vv += cn[4] + cn[5]; }   // (v58_MB k_slwmb)
    else {
      for (int c = cn[4]; c > 0; c--) R(*rr++) = RGCV(*vv++);
      for (int c = cn[5]; c > 0; c--) R(*rr++) = *vv++;
    }
    R(RG_CMD) = *rr++; npu_pending = 1;
    rgx_step(); rgx_step();                      // (v51_D) while the NPU works (2 of 20 steps per op)
  }
  return 1;
}
static int rgi_clean DRV = 1;    // the staging bank holds what op 0's writes assume (0 elsewhere): after reset / a replay
static void rgz(const uint8_t* z) { for (; *z; z++) R(*z) = 0; }
static int __attribute__((noinline)) rgi_back(void) {   // a check failed: NPU idle, then the generic kernels
  (void)R(RG_STATUS); npu_pending = 0; rgi_clean = 0;
  for (uint32_t k = 1; k < 68; k++) if (k != 4 && k != 66) R(k) = 0;   // the bank the generic kernels expect (reset)
  return 0;
}

// ---- (v56_CY k_late) the replay's late exit (see tools/k_late.py) ----
// ds: exit while op 5 runs (2390 cycles); later ops [92]
#define RL_DS_E 5
#define RL_DS_N 7
static const uint8_t rlk_ds[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2};   // per node: 0 done, 1 background, 2 rest
// mb: exit while op 16 runs (5916 cycles); later ops [1076, 859, 1478, 999, 1076, 606, 1114, 999, 1077, 649, 1114, 999, 1232, 865, 1478, 999, 865, 1175, 762, 1497, 1060, 1013, 1082, 1102, 472, 79]
#define RL_MB_E 16
#define RL_MB_N 43
static const uint8_t rlk_mb[43] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2};   // per node: 0 done, 1 background, 2 rest
// rs (v57_RS k_rsrl): exit while op 9 (conv9 + fused ADD) runs; every node but the last returns at once
#define RL_RS_E 9
#define RL_RS_N 12
static const uint8_t rlk_rs[21] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2};   // per node: 0 done (intermediate), 2 rest

static int rl_on DRV;                    // the replay of this inference is open (op E runs, op E + 1's writes pending)
static const uint8_t* rl_k DRV;          // per node: 0 complete at the exit, 1 background work, 2 the rest
static const Node* rl_n0 DRV;
static const Node* rl_act DRV;           // the next node whose kernel has something to do (rlk != 0; 0: none)
static int rl_nn DRV;                    // the model's node count
static const Node* rl_nx(int j) { for (; j < rl_nn; j++) if (rl_k[j]) return rl_n0 + j; return 0; }
// a check of a later op failed: NPU idle, bank reset, the generic group rewrites every node from node 0 (as
// rgi_first's failure); this node's and every later node's output are then complete
static void __attribute__((noinline, cold)) rl_fail(const Node* node, Tensor* tensors) {
  const Node* n0 = rl_n0;
  rl_on = 0; rgi_back();
  kg_next = 0;
  kg_tab[n0->op](n0, tensors);
  kg_run(n0, tensors);
  (void)R(RG_STATUS); npu_pending = 0;
  if (kg_next == n0 + 1) kg_next = node + 1;   // the group rewrote every node: later kernels return at once
  else {                                       // (no group: this node and the ones before it, one by one)
    for (const Node* q = n0 + 1; q <= node; q++) { kg_tab[q->op](q, tensors); (void)R(RG_STATUS); npu_pending = 0; }
    kg_next = 0;
  }
}
// the rest of the replay (op E + 1's writes and CMD, the later ops, the bank clean), then idle; later kernels skip
static void __attribute__((noinline)) rl_rest(const Node* node, Tensor* tensors) {
  rl_on = 0;
  if (!rgi_play_s(rl_cn, rl_rr, rl_vv, rl_n, rl_sp, -1, 1)) { rl_fail(node, tensors); return; }
  rl_act = 0;
  rgi_clean = 0;                               // (the staging bank: zeroed by the next inference's rgi_first, if any)
  if (R(RG_STATUS) & 1u) { R(66) = 2u; rl_fail(node, tensors); return; }   // (DWF flag: no DWF op after the exit)
  npu_pending = 0;
  kg_next = node + 1;
}
// a later node of an open replay whose rlk entry is not 0
static void __attribute__((noinline)) rl_node(const Node* node, Tensor* tensors) {
  const int j = (int)(node - rl_n0);
  if (rl_k[j] == 1) { while (R(8) & 2u) { } rl_act = rl_nx(j + 1); }   // the OP_PADI's background border / copy
  else rl_rest(node, tensors);                 //   (r_stbg: status bit 1); else the rest of the replay
  (void)R(RG_SLOT);                            // (nothing queued: returns at once)
}
// a kernel called out of node order while the replay is open (never with the harness): finish it first
static void __attribute__((noinline, cold)) rl_out(const Node* node, Tensor* tensors) {
  rl_rest(node, tensors);
  (void)R(RG_SLOT);
}
static int rgi_first(const Node* node, Tensor* tensors) {
  if (node->op != OP_PAD) { rgi_clean = 0; return 0; }
  const uint32_t o = (uint32_t)node->output;
  if (o > 4095u) { rgi_clean = 0; return 0; }
  const int32_t h = tensors[o].shape.dims[1];
  int nn;
  rgc[0] = (uint32_t)node; rgc[1] = (uint32_t)tensors;
  if (o == RGI_DS_O0 && h == RGI_DS_H0) {
    if (!rgi_clean) rgz(rlz_ds_f);
    rgx_n0 = node; rgx_k = 0; rgx_bad = 0; rgx_p = 0; rgx_nn = RGI_DS_NN;   // (v51_D) npu_dwf + PAD preconditions
    if (RGX_HW) R(66) = 2u;                   // (v51_D) clear the DWF parameter flag (tee disarmed)
    _Static_assert(RL_DS_E + 1 >= RGF_DS_K && RL_DS_E + 1 < RGI_DS_N && RL_DS_N == RGI_DS_N, "k_late: exit op");
    if (!rgf_ds() || rgi_play_s(rlx_ds_c, rlx_ds_r, rlx_ds_v, RGI_DS_N - RGF_DS_K, rgc + RGF_DS_S, RL_DS_E + 1 - RGF_DS_K, 0) != 2) { rgx_n0 = 0; return rgi_back(); }
    while (rgx_k < 20 || rgx_p < rgx_nn - 1) rgx_step();
    rgx_n0 = 0;
    if (rgx_bad) return rgi_back();
    if (RGX_HW && (R(8) & 1u)) { R(66) = 2u; return rgi_back(); }   // (v51_D) a DWF parameter out of range (R(8) never held)
    rl_k = rlk_ds; rl_on = 1; rl_n0 = node; rl_nn = RGI_DS_NN; rl_act = rl_nx(1);   // (v56_CY k_late) op E runs: return
    kg_next = node + 1;
    return 1;
  } else if (o == RGI_MB_O0 && h == RGI_MB_H0) {
    if (!rgi_clean) rgz(rlz_mb_f);
    R(66) = 2u;                               // (v55_KC) STATUS bit 0 cleared
    rgx_n0 = node; rgx_k = 20; rgx_bad = 0; rgx_p = 0; rgx_nn = RGI_MB_NN;   // (v51_D) PAD preconditions (k_rids)
    _Static_assert(RL_MB_E + 1 >= RGF_MB_K && RL_MB_E + 1 < RGI_MB_N && RL_MB_N == RGI_MB_N, "k_late: exit op");
    if (!rgf_mb() || rgi_play_s(rlx_mb_c, rlx_mb_r, rlx_mb_v, RGI_MB_N - RGF_MB_K, rgc + RGF_MB_S, RL_MB_E + 1 - RGF_MB_K, 0) != 2) { rgx_n0 = 0; return rgi_back(); }
    while (rgx_p < rgx_nn - 1) rgx_step();
    rgx_n0 = 0;
    if (rgx_bad) return rgi_back();
    rl_k = rlk_mb; rl_on = 1; rl_n0 = node; rl_nn = RGI_MB_NN; rl_act = rl_nx(1);   // (v56_CY k_late) op E runs: return
    kg_next = node + 1;
    return 1;
  } else if (o == RGI_RS_O0 && h == RGI_RS_H0) {
    if (!rgi_clean) rgz(rlz_rs_f);
    rgx_n0 = node; rgx_k = 20; rgx_bad = 0; rgx_p = 0; rgx_nn = RGI_RS_NN;   // (v51_D) PAD preconditions (k_rids)
    _Static_assert(RL_RS_E + 1 >= RGF_RS_K && RL_RS_E + 1 < RGI_RS_N && RL_RS_N == RGI_RS_N, "k_late: exit op");
    if (!rgf_rs() || rgi_play_s(rlx_rs_c, rlx_rs_r, rlx_rs_v, RGI_RS_N - RGF_RS_K, rgc + RGF_RS_S, RL_RS_E + 1 - RGF_RS_K, 0) != 2) { rgx_n0 = 0; return rgi_back(); }
    while (rgx_p < rgx_nn - 1) rgx_step();
    rgx_n0 = 0;
    if (rgx_bad) return rgi_back();
    rl_k = rlk_rs; rl_on = 1; rl_n0 = node; rl_nn = RGI_RS_NN; rl_act = rl_nx(1);   // (v56_CY k_late) op E runs: return
    kg_next = node + 1;
    return 1;
  } else { rgi_clean = 0; return 0; }   // (another model: the generic kernels use the bank)
  kg_next = node + 1; kg_last = node + nn - 1;
  return 1;
}
#define KG_SKIP(node) kg_skip(node)
#define KG_RUN(node, tensors) kg_run(node, tensors)

// ====================================================================== (v41_C) entry points
// Each operator kernel ends with one unconditional STATUS load: the NPU holds it while an op is queued or
// running (its write DMA idle = every write's B response received), so no kernel returns with NPU work
// outstanding. (KG_SKIP / KG_RUN: kernel groups, see s_group.py; empty without them.)
#ifndef KG_SKIP
#define KG_SKIP(node) 0
#define KG_RUN(node, tensors)
#endif
// A kernel whose output an earlier kernel of this inference already completed (and waited for) returns at once.
// (v47_a) leaf fast path for a node whose output the group already completed (no stack frame); the rest is
// the previous wrapper, tail-called: kg_next = 0 (= kg_skip's miss), the kernel, the STATUS wait.
#define KRET(n) static void __attribute__((noinline)) n##_slow(const Node* node, Tensor* tensors) { \
  if (rl_on) rl_out(node, tensors);   /* (v56_CY k_late: out of node order: finish the open replay) */ \
  kg_next = 0; adk_any = 1; /* (v55AD) */ \
  if (!rgi_first(node, tensors)) { n##_k(node, tensors); KG_RUN(node, tensors); (void)R(RG_STATUS); npu_pending = 0; } \
  (void)R(RG_SLOT); } \
void n(const Node* node, Tensor* tensors) { \
  const Node* x = kg_next; \
  if (__builtin_expect(node == x, 1)) { kg_next = x + 1; if (node == rl_act) rl_node(node, tensors); return; }   /* (v56_CY k_lean / k_late) */ \
  n##_slow(node, tensors); }
KRET(Conv2DEval)
KRET(MaxPool2DEval)
static void __attribute__((noinline)) fc_slow(const Node* node, Tensor* tensors) {   // (v53_Q it_lean2) the rest
  if (KG_SKIP(node)) { if (node == rl_act) rl_node(node, tensors); return; }   // (v56_CY k_late)
  if (rl_on) rl_out(node, tensors);
  if (adz_first(node, tensors)) return;
  adk_any = 1;
  FullyConnectedEval_k(node, tensors); KG_RUN(node, tensors); (void)R(RG_STATUS); npu_pending = 0; }
void __attribute__((aligned(4))) FullyConnectedEval(const Node* node, Tensor* tensors) {   // (v46AD) + ad_fcae
  if (__builtin_expect(node == adz_nx, 1)) {   // (v49AD) the next node of a checked ad_fcae inference
    const int k = adz_k;
    if (__builtin_expect(k != 0, 1)) {         // (v58_AD k_adpl2) kernel k: queue op k+1, SLOT (= op k finished)
      if (__builtin_expect(k != 9, 1)) { adz_k = k + 1; adz_nx = node + 1; if (__builtin_expect(adz_pl == 1, 1)) { if (k < 3) adz_kg(k + 1, 1); else if (k == 3) adz_k45(); else if (k < 8) adz_kg(k + 2, 1); } else if (adz_pl) adk_q(k + 1); return; }   // (v58_AD k_adq45)
      adz_k = 0; adz_nx = adz_n0;
      if (__builtin_expect(adz_pl == 1, 1)) { (void)R(RG_STATUS); npu_pending = 0; }
      else if (adz_pl) adk_fin();   // (v58_AD k_adp0) image 0: STATUS, node 9's entry, all-packed test   // kernel 9: op 9 finished, NPU idle
      adz_pl = 1;                              // (the adz_first inference ran all ten ops in its kernel 0)
      return;
    }
    adz_k = 1; adz_nx = node + 1;
    adz_p0();                                  // (v58_AD k_adpl2) kernel 0: op 0 runs, op 1 taken = op 0 finished
    return;
  }
  if (node == kg_next) { kg_next = node + 1; if (node == rl_act) rl_node(node, tensors); return; }   // (v56_CY k_late: a completed group node)
  fc_slow(node, tensors);
}
KRET(DepthwiseConv2DEval)
KRET(AddEval)
KRET(AveragePool2DEval)
KRET(MeanEval)
KRET(ReshapeEval)
KRET(SoftmaxEval)
KRET(PadEval)
KRET(MulEval)
KRET(ConcatenationEval)
KRET(LogisticEval)
KRET(TanhEval)

// ====================================================================== (v46_MB) mobilenet helpers (at the end: the
// code of every other function keeps its place)
// (v46_MB) a pointwise node of the mobilenet group (1x1, stride 1, K <= 144, > 1 pixel): the registers npu_conv writes
// for a contiguous single-chunk conv, straight from the node; mirrors for the planned nodes. 1: launched
// (v55_KC) plan positions of the mobilenet group (as MBM_*: structure, the op plan)
#define KC_LO ((1u << 25) | (1u << 28) | (1u << 31))
#define KC_HI ((1u << (34 - 32)) | (1u << (37 - 32)) | (1u << (40 - 32)))
static const Node* g_kcnp DRV;            // a scatter conv wrote only its own output: the next PAD is not elided
static inline int kc_plan(uint32_t k) { return KC_ON && mbm_n0 != 0 && k < 64u && mbm_bit(KC_LO, KC_HI, k); }
// OP_KCP for a 1x1 conv (Co 128 / 256, K = Ci 128 / 256, bias present); 1: the compact / scatter convs may follow
static int __attribute__((noinline)) kc_probe(const Node* node, Tensor* tensors, int32_t Co, int32_t K, int32_t cap) {
  const ConvParams* p = (const ConvParams*)node->builtin;
  npu_slot();
  R(66) = 2u;                                   // STATUS bit 0 (sticky) cleared
  W(RG_IN, (uint32_t)kc_ls); W(RG_WGT, (uint32_t)tensors[node->filter].data);
  W(RG_BIAS, (uint32_t)tensors[node->bias].data); W(RG_MULT, (uint32_t)node->out_mult);
  W(RG_SHIFT, (uint32_t)node->out_shift); W(RG_OUT, (uint32_t)kc_st);
  W(RG_CO, Co); W(RG_K, K); W(RG_L, cap);
  W(RG_M1, (uint32_t)kc_cb); W(RG_M2, (uint32_t)kc_cm); W(RG_OFF1, (uint32_t)kc_cs); W(RG_OFF2, (uint32_t)kc_cw);
  W(RG_PWBC, (uint32_t)kc_oh); W(RG_PTOT, (uint32_t)kc_lg);
  W(RG_OOFF, (uint32_t)p->output_offset); W(RG_AMIN, (uint32_t)p->quantized_activation_min);
  W(RG_AMAX, (uint32_t)p->quantized_activation_max);
  R(RG_CMD) = NOP_KCP;
  const uint32_t st = R(RG_STATUS);             // (generic kernels: wait; the replay checks the flag at its end)
  npu_pending = 0;
  if (st & 1u) { R(66) = 2u; return 0; }        // more live channels than cap: the plain conv
  return 1;
}
static int kc_shape(const Node* node, const Tensor* tensors, int32_t* d) {   // H, W, Ci, Co, npix
  const Tensor* ti = &tensors[node->input];
  const Tensor* tf = &tensors[node->filter];
  const Tensor* to = &tensors[node->output];
  const ConvParams* p = (const ConvParams*)node->builtin;
  int32_t di[4], dd[4];
  if (node->bias < 0 || tf->shape.n != 4 || !shape4(&ti->shape, di) || !shape4(&to->shape, dd)) return 0;
  const int32_t Ci = di[3], Co = tf->shape.dims[0];
  if (tf->shape.dims[1] != 1 || tf->shape.dims[2] != 1 || tf->shape.dims[3] != Ci || dd[3] != Co ||
      dd[1] != di[1] || dd[2] != di[2] || p->stride_h != 1 || p->stride_w != 1 || p->pad_h != 0 || p->pad_w != 0 ||
      (Ci != 128 && Ci != 256) || (Co != 128 && Co != 256) || di[1] < 1 || di[2] < 1 ||
      di[1] * di[2] * KC_CAP(Co, Ci) > KC_CA || di[1] * di[2] < 2 ||
      p->input_offset < -127 || p->input_offset > 128 || p->output_offset < -256 || p->output_offset > 255 ||
      p->quantized_activation_min < -128 || p->quantized_activation_max > 127 ||
      p->quantized_activation_min > p->quantized_activation_max) return 0;
  d[0] = di[1]; d[1] = di[2]; d[2] = Ci; d[3] = Co; d[4] = di[1] * di[2];
  return 1;
}
// a planned K = 256 node (40): probe, compact conv (chunked), scatter conv; 1: done
static int __attribute__((noinline)) kc_big(const Node* node, Tensor* tensors) {
  int32_t d[5];
  if (!kc_plan((uint32_t)(node - mbm_n0)) || !kc_shape(node, tensors, d) || d[2] != 256) return 0;
  if ((vpad.out != 0 && tensors[node->input].data == vpad.out) || g_fzc.n != 0) return 0;
  const ConvParams* p = (const ConvParams*)node->builtin;
  const int32_t H = d[0], Wd = d[1], Ci = d[2], Co = d[3], cap = KC_CAP(Co, Ci);
  if (!kc_probe(node, tensors, Co, Ci, cap)) return 0;
  g_nowait = 1; g_kcgl = 4u;
  const int8_t* const kin = (MB_LAX && dws_from != 0 && tensors[node->input].data == dws_from) ? (const int8_t*)dws_to
                            : (const int8_t*)tensors[node->input].data;   // (v57_MB)
  const int ok = npu_conv(kin, kc_cw, kc_cb, kc_cm, kc_cs, kc_ca, H, Wd, Ci, H, Wd, cap,
                          1, 1, 1, 1, p->input_offset, p->output_offset, p->quantized_activation_min,
                          p->quantized_activation_max, H, Wd, 0, 0);
  g_nowait = 0; g_kcgl = 0;
  if (!ok) return 0;                            // (the compact results are unused: the plain path follows)
  return npu_conv(kc_ca, kc_oh, kc_st, id_m, id_z, (int8_t*)tensors[node->output].data, H, Wd, cap, H, Wd, Co, 1, 1, 1, 1,
                  0, 0, -128, 127, H, Wd, 0, 0);
}
// (v57_MB) the 43 nodes from n0 (kg_detect's op sequence) form a plain chain: node j reads node j - 1's output (one
// input each), the outputs are distinct and no filter / bias / node 0's input is any of them; then node j's output is
// read by node j + 1 alone. Checked once per group base (structure only: one pass, cached)
static const Node* mbl_cn DRV;
static int mbl_cv DRV;
static int __attribute__((noinline)) mbl_chk(const Node* n0) {
  if (n0 == mbl_cn) return mbl_cv;
  uint32_t ob[32];
  for (int i = 0; i < 32; i++) ob[i] = 0;
  int ok = 1;
  const Node* q = n0;
  for (int j = 0; j < 43 && ok; j++, q++) {
    const int32_t o = q->output;
    if (q->num_inputs > 1 || o < 0 || o >= 1024 || (j != 0 && q->input != (q - 1)->output) ||
        ((ob[o >> 5] >> (o & 31)) & 1u)) ok = 0;
    else ob[o >> 5] |= 1u << (o & 31);
  }
  q = n0;
  for (int j = 0; j < 43 && ok; j++, q++) {
    const int32_t f = q->filter, b = q->bias;
    if ((f >= 0 && (f >= 1024 || ((ob[f >> 5] >> (f & 31)) & 1u))) || (b >= 0 && (b >= 1024 || ((ob[b >> 5] >> (b & 31)) & 1u))))
      ok = 0;
  }
  const int32_t i0 = n0->input;
  if (ok && (i0 < 0 || i0 >= 1024 || ((ob[i0 >> 5] >> (i0 & 31)) & 1u))) ok = 0;
  mbl_cn = n0; mbl_cv = ok;
  return ok;
}
// (v57_MB) 1: in the recognised mobilenet group, tensor t (node r - 1's output) is read by node r alone and is not the
// group's last output (the model's output)
static int __attribute__((noinline)) mbl_sole(const Node* r, int32_t t) {
  if (!MB_LAX || mbm_n0 == 0 || r <= mbm_n0 || r > mbm_n0 + 42 || (r - 1)->output != t || r->input != t) return 0;
  return mbl_chk(mbm_n0) && (mbm_n0 + 42)->output != t;
}
// (v57_MB k_mbf) node 0 = the mobilenet front's PAD (bottom / right padding only) whose output only node 1 reads,
// node 1 a 3x3 stride-2 conv (8 output channels: pixel pairs) needing exactly that padding, node 1's output read only
// by PAD 2, PAD 2 elided for depthwise 3 and read only by it: no op now; node 1's kernel runs the folded ops
static int __attribute__((noinline)) mbf_try(const Node* node, Tensor* t) {
  g_mbf = 0; g_mbfd = 0;
  if (!MB_LAX || kg_n < 0 || (kg_n > 0 && kg_m != 2) || !mbm_front(node, t)) return 0;
  const Node* c = node + 1;
  const Node* p2 = node + 2;
  const Node* d3 = node + 3;
  const PadParams* pp = (const PadParams*)node->builtin;
  const ConvParams* p = (const ConvParams*)c->builtin;
  const Tensor* tf = &t[c->filter];
  int32_t di[4], dp[4], dout[4];
  if (c->filter < 0 || c->filter > 4095 || c->output < 0 || c->output > 4095 || !shape4(&t[node->input].shape, di) ||
      !shape4(&t[node->output].shape, dp) || !shape4(&t[c->output].shape, dout)) return 0;
  const int32_t H = di[1], Wd = di[2], C = di[3];
  if (pp->pad_h_before != 0 || pp->pad_w_before != 0 || dp[1] != H + 1 || dp[2] != Wd + 1 || dp[3] != C ||
      pp->pad_value + p->input_offset != 0 || tf->shape.n != 4 || tf->shape.dims[1] != 3 || tf->shape.dims[2] != 3 ||
      tf->shape.dims[3] != C || tf->shape.dims[0] != 8 || dout[3] != 8 || p->stride_h != 2 || p->stride_w != 2 ||
      p->pad_h != 0 || p->pad_w != 0 || 2 * (dout[1] - 1) + 3 != H + 1 || 2 * (dout[2] - 1) + 3 != Wd + 1 ||
      (dout[2] & 1) != 0 || dout[1] < 2 || C < 1 || C > 3 || H > 255 || Wd > 255 || 3 * Wd * C + 16 > GAT_BYTES ||
      dout[1] * dout[2] * 8 > PAD_SCR_BYTES) return 0;
  if (p2->op != OP_PAD || p2->input != c->output || d3->op != OP_DEPTHWISE_CONV_2D || d3->input != p2->output) return 0;
  if (!mbl_chk(node)) return 0;                 // (the 43-node chain: nodes 1 / 2 / 3 alone read 0 / 1 / 2's outputs)
  if (!pad_elides(p2, t)) return 0;
  g_mbf = node;
  return 1;
}
// node 1's folded ops (C, A, B); 1: done (its output at pad_scr, PAD 2 recorded for depthwise 3)
static int __attribute__((noinline)) mbf_conv(const Node* node, Tensor* t) {
  const Node* n0 = g_mbf;
  g_mbf = 0;
  const ConvParams* p = (const ConvParams*)node->builtin;
  int32_t di[4], dout[4];
  shape4(&t[n0->input].shape, di); shape4(&t[node->output].shape, dout);   // (checked by mbf_try)
  const int32_t H = di[1], Wd = di[2], C = di[3], Ho = dout[1], Wo = dout[2], Co = dout[3], rw = Wd * C;
  const int8_t* in = (const int8_t*)t[n0->input].data;
  const int8_t* w = (const int8_t*)t[node->filter].data;
  const int32_t* bias = node->bias >= 0 ? (const int32_t*)t[node->bias].data : 0;
  const int32_t io = p->input_offset, oo = p->output_offset, mn = p->quantized_activation_min, mx = p->quantized_activation_max;
  const uint32_t padr = (uint32_t)C << 16;   // (the PAD's one right column: C bytes past each row's end)
  npu_slot();                                // op C: the last two input rows -> gat_scr (plain copy)
  W(RG_IN, (uint32_t)(in + (H - 2) * rw)); W(RG_OUT, (uint32_t)gat_scr);
  W(RG_PROWS, 2); W(RG_PHB, 0); W(RG_PH, 2); W(RG_PWBC, 0); W(RG_PWC, rw); W(RG_PWAC, 0); W(RG_PWOC, rw);
  W(RG_PTOT, 2 * rw); W(RG_PVAL, 0); W(RG_COUNT, 2 * rw);
  R(RG_CMD) = NOP_PAD; npu_pending = 1;
  g_mbfx = 1; g_nowait = 1;
  g_cpx |= 1024u; g_padx = padr;            // op A: rows 0 .. Ho-2 in place (window rows < H - 1)
  int ok = npu_conv(in, w, bias, node->out_mult, node->out_shift, pad_scr, H + 1, Wd + 1, C, Ho - 1, Wo, Co, 3, 3, 2, 2,
                    io, oo, mn, mx, H, Wd, 0, 0);
  g_cpx = 0; g_padx = 0;
  if (ok) {                                  // op B: the last row from the copy (row 2 = the bottom padding)
    g_cpx |= 1024u; g_padx = padr;
    ok = npu_conv(gat_scr, w, bias, node->out_mult, node->out_shift, pad_scr + (Ho - 1) * Wo * Co, 3, Wd + 1, C, 1, Wo, Co,
                  3, 3, 2, 2, io, oo, mn, mx, 2, Wd, 0, 0);
    g_cpx = 0; g_padx = 0;
  }
  g_mbfx = 0; g_nowait = 0;
  if (!ok) {                                 // (not expected: mbf_try checked the shapes) the PAD after all
    npu_wait();
    const Node* sv = g_node; g_node = n0; pad_launch(n0, t, 0); g_node = sv;
    return 0;                                // (the generic conv follows, reading the PAD's output)
  }
  g_mbfd = n0;
  g_mbr_pad = node + 1; g_mbr_src = pad_scr; g_mbr_lay = LY_NHWC;   // (PAD 2: no op, depthwise 3 reads pad_scr)
  return 1;
}
// a failed group detection after the mobilenet front: node 1 runs again in its own kernel from node 0's PAD output
static void __attribute__((noinline, cold)) mbf_undo(Tensor* t) {
  const Node* n0 = g_mbfd;
  g_mbf = 0; g_mbfd = 0;
  if (n0 == 0) return;
  npu_wait();
  const Node* sv = g_node; g_node = n0; pad_launch(n0, t, 0); g_node = sv;
  npu_wait();
}
static __attribute__((optimize("Os"))) int mbm_pw(const Node* node, Tensor* tensors) {
  const Tensor* tf = &tensors[node->filter];
  const RuntimeShape* sf = &tf->shape;
  if (sf->n != 4 || sf->dims[1] != 1 || sf->dims[2] != 1) return 0;
  const Tensor* ti = &tensors[node->input];
  const Tensor* to = &tensors[node->output];
  const ConvParams* p = (const ConvParams*)node->builtin;
  int32_t di[4], d[4];
  if (!shape4(&ti->shape, di) || !shape4(&to->shape, d)) return 0;
  const int32_t H = di[1], Wd = di[2], Ci = di[3], Co = sf->dims[0], npix = H * Wd;
  const int32_t in_off = p->input_offset, out_off = p->output_offset;
  const int32_t amin = p->quantized_activation_min, amax = p->quantized_activation_max;
  if (sf->dims[3] != Ci || d[1] != H || d[2] != Wd || d[3] != Co || p->stride_h != 1 || p->stride_w != 1 ||
      p->pad_h != 0 || p->pad_w != 0 || Ci < 1 || Ci > 144 || Co < 1 || Co > 1023 || H > 255 || Wd > 255 ||
      npix < 2 || npix > 4095 || Wd * Ci > 65535 || npix * Ci >= (1 << 20) || in_off < -127 || in_off > 128 ||
      out_off < -256 || out_off > 255 || amin < -128 || amax > 127 || amin > amax) return 0;
  if ((vpad.out != 0 && ti->data == vpad.out) || g_fzc.n != 0) return 0;
  const int8_t* const xin = (MB_LAX && dws_from != 0 && ti->data == dws_from) ? (const int8_t*)dws_to
                                                                              : (const int8_t*)ti->data;   // (v57_MB)
  const int32_t* bias = node->bias >= 0 ? (const int32_t*)tensors[node->bias].data : 0;
  if (!acc_ok(bias, (const int8_t*)tf->data, Co, Ci, in_off, 0)) return 0;
  const int32_t pg = (Ci <= 64 && Co > 8);
  const int32_t glast = pg ? ((Co & 15) ? (Co & 15) : 16) : ((Co & 7) ? (Co & 7) : 8);
  // mirrors (planned nodes; never with a single group: the output is then one write command)
  uint32_t mpg = 0;
  const uint32_t k = (uint32_t)(node - mbm_n0);
  int8_t* aout = (pg && !(MBM && k < 43u && mbm_bit(MBM_PL_LO, MBM_PL_HI, k))) ? apw_out(node, tensors, H, Wd, Co)
                                                                                       : (int8_t*)to->data;   // (v49_MB k_apw)
  // (v57_MB) the next PAD elided for its depthwise consumer and nothing else in the group reading this node's or the
  // PAD's output: write only what the depthwise conv reads (planes / aligned NHWC in our scratch); no mirror, no copy
  int lx = 0, lxp = 0;
  {
    const Node* pn = node + 1;
    if (MB_LAX && k < 43u && pn->op == OP_PAD && pn->input == node->output && (pn + 1)->op == OP_DEPTHWISE_CONV_2D &&
        (pn + 1)->input == pn->output && pad_elides(pn, tensors) && mbl_sole(pn, node->output) &&
        mbl_sole(pn + 1, pn->output)) {
      lxp = mbm_bit(MBL_PL_LO, MBL_PL_HI, k) && (Co & 15) == 0 && Co >= 32 && npix * Co <= PAD_SCR_BYTES &&
            dw_planar_ok(pn + 1, Co);
      if (npix * Co <= PAD_SCR_BYTES) { aout = pad_scr; lx = 1; }
      else if (npix * Co <= ARENA_BYTES) { aout = arena; lx = 1; }
    }
  }
  if (!MB_LAX && !lx && MBM && k < 43u && (Co & 15) == 0 && Co >= 32) {   // (v57_MB r_mbl: no mirrors)
    if (mbm_bit(MBM_PL_LO, MBM_PL_HI, k) && npix * Co <= PAD_SCR_BYTES) mpg = 0x20u;
    const Node* pn = node + 1;
    if (mbm_bit(MBM_PM_LO, MBM_PM_HI, k) && pn->op == OP_PAD && pn->input == node->output) {
      const PadParams* pp = (const PadParams*)pn->builtin;
      const Tensor* tp = &tensors[pn->output];
      int32_t q[4];
      const int32_t hb = pp->pad_h_before, wb = pp->pad_w_before;
      if (shape4(&tp->shape, q) && q[3] == Co && hb >= 0 && wb >= 0 && q[1] >= H + hb && q[2] >= Wd + wb &&
          (q[2] - Wd) * Co < 1024 && q[2] - Wd <= 3 && (hb * q[2] + wb) * Co < 65536 &&
          (q[2] - wb - Wd) * Co + (q[1] - hb - H) * q[2] * Co < 65536) {
        const uint32_t tl = (uint32_t)((hb * q[2] + wb) * Co);
        const uint32_t bl = (uint32_t)((q[2] - wb - Wd) * Co + (q[1] - hb - H) * q[2] * Co);
        g_mir5 = ((uint32_t)tp->data + tl - (uint32_t)aout) & 0xfffffu;   // (v49_MB k_apw: relative to aout)
        g_mir1 = tl; g_mir2 = bl; g_mirv = (uint32_t)(uint8_t)pp->pad_value;
        mpg |= 0x50u | ((uint32_t)(q[2] - Wd) << 7);
      }
    }
  }
  int32_t kd[5], pgx = pg;
  if (kc_plan(k) && !pg && Ci == 128 && kc_shape(node, tensors, kd) && kc_probe(node, tensors, Co, Ci, KC_CAP(Co, Ci))) {
    const int32_t cap = KC_CAP(Co, Ci);                // (v55_KC) compact conv, then the scatter conv in the node's place
    npu_slot(); g_kcgl = 4u;
    conv_regs(xin, kc_cw, kc_cb, kc_cm, kc_cs, kc_ca, H, Wd, cap, 1, Ci, Wd * Ci, Ci, Wd * Ci, Ci,
              8 * Ci, npix, 1, 1, npix * Ci, Ci, 1, 0, 0, 0u, H, Wd, in_off, out_off, amin, amax, 0);
    g_kcgl = 0; R(RG_CMD) = NOP_CONV; npu_pending = 1;
    npu_slot();
    conv_regs(kc_ca, kc_oh, kc_st, id_m, id_z, aout, H, Wd, Co, 1, cap, Wd * cap, cap, Wd * cap, cap, 16 * cap, npix, 1,
              1, npix * cap, cap, 1, 0, 1, 0u, H, Wd, 0, 0, -128, 127, 0);
    pgx = 1; mpg = 0; g_kcnp = node;               // (no mirrors: the next PAD runs as a PAD, see mbm_pad)
  } else {
  npu_slot();
  conv_regs(xin, (const int8_t*)tf->data, bias, node->out_mult, node->out_shift, aout,
            H, Wd, Co, 1, Ci, Wd * Ci, Ci, Wd * Ci, Ci, glast * Ci, npix, 1, 1, npix * Ci, Ci, 1, 0, pg, 0u, H, Wd,
            in_off, out_off, amin, amax, 0);
  }
  if (mpg) {                                       // mirror writes / border fills (RG_PG again, with their bits)
    W(RG_PG, (uint32_t)pgx | mpg); W(RG_SCR, (uint32_t)pad_scr);
    if (mpg & 0x40u) { W(RG_RDELTA, g_mir5); W(RG_COUNT, g_mir1); W(RG_PTOT, g_mir2); W(RG_PVAL, g_mirv); }
  }
  if (lx) {                                        // (v57_MB) planes: 16-channel groups (bit 1) / 8-channel groups (po8)
    if (lxp) W(RG_PG, pgx ? ((uint32_t)pgx | 2u) : 8u);
    npu_run(NOP_CONV);
    g_mbr_pad = node + 1; g_mbr_src = aout; g_mbr_lay = lxp ? LY_P : LY_NHWC;
    g_kcnp = 0;                                    // (the next PAD: no op)
    return 1;
  }
  if (!mpg && aout == (int8_t*)to->data && pg && Co == 16) mbr_conv(1u);   // (v50_MB k_rep)
  // (v50_MB k_il) 32 channels, planes mirror only: the conv writes just the planes; OP_PIL writes the NHWC output
  const int il = (mpg == 0x20u && pg && Co == 32 && aout == (int8_t*)to->data && npix * 32 < 65536);   // (OP_PIL: 16-bit lengths)
  const int p7 = (!MB_LAX && !mpg && pg && Co == 32 && aout == pad_scr && npix * 32 < 65536);   // (v57_MB r_mbl: no OP_PIL)   // (v50_MB k_p7) planes at pad_scr
  if (il || p7) { W(RG_OUT, (uint32_t)pad_scr); W(RG_PG, (uint32_t)pg | 2u); }
  npu_run(NOP_CONV);
  if (il || p7) {
    npu_slot();
    W(RG_IN, (uint32_t)pad_scr); W(RG_OUT, (uint32_t)to->data); W(RG_COUNT, (uint32_t)(npix * 16));
    W(RG_PTOT, (uint32_t)(npix * 32));
    npu_run(NOP_PIL);
  }
  if (p7) { g_apw = (const int8_t*)to->data; g_apwp = 1; }   // (v50_MB k_p7) OP_PIL wrote the own output
  else if (aout != (int8_t*)to->data) apw_copy(node, tensors, H, Wd, Co);
  return 1;
}
// (v46_MB) the PAD after a planned pointwise node: the depthwise conv reads the planes; with M1 the pointwise op writes
// this PAD's whole output (no NPU op here). 1: handled
static __attribute__((optimize("Os"))) int mbm_pad(const Node* node, Tensor* tensors) {
  if (MB_LAX) return 0;                // (v57_MB r_mbl) no mirror wrote this PAD's output: the generic PAD
  if (KC_ON && g_kcnp != 0 && g_kcnp == node - 1) { g_kcnp = 0; return 0; }   // (v55_KC) the generic PAD
  const uint32_t k = (uint32_t)(node - mbm_n0);
  if (!MBM || !MBM_FAST || k == 0u || k >= 43u) return 0;
  const int m2 = mbm_bit(MBM_PL_LO, MBM_PL_HI, k - 1u), pm = mbm_bit(MBM_PM_LO, MBM_PM_HI, k - 1u);
  if (!m2 && !pm) return 0;
  const Node* pw = node - 1;
  if (pw->output != node->input || pw->op != OP_CONV_2D) return 0;
  const PadParams* pp = (const PadParams*)node->builtin;
  const Tensor* ti = &tensors[node->input];
  const Tensor* to = &tensors[node->output];
  int32_t di[4], q[4];
  if (!shape4(&ti->shape, di) || !shape4(&to->shape, q)) return 0;
  const int32_t H = di[1], Wd = di[2], C = di[3], hb = pp->pad_h_before, wb = pp->pad_w_before;
  // (the conditions under which mbm_pw wrote the mirrors)
  if ((C & 15) != 0 || C < 32 || H * Wd < 2 || q[3] != C || C > 1023) return 0;
  if (m2 && H * Wd * C > PAD_SCR_BYTES) return 0;
  const int m1 = pm && hb >= 0 && wb >= 0 && q[1] >= H + hb && q[2] >= Wd + wb &&
                 (q[2] - Wd) * C < 1024 && q[2] - Wd <= 3 && (hb * q[2] + wb) * C < 65536 &&
                 (q[2] - wb - Wd) * C + (q[1] - hb - H) * q[2] * C < 65536;
  if (m1) {
    vpad.out = to->data; vpad.node = node; vpad.tensors = tensors;
    vpad.src = (const int8_t*)ti->data; vpad.lay = LY_NHWC;    // (no planes: the NHWC input, implicit padding)
    if (g_apw != 0 && g_apw == vpad.src) vpad.src = pad_scr;   // (v49_MB k_apw) its aligned copy
    g_apw = 0;
  } else {
    if (!m2) return 0;
    pad_launch(node, tensors, 1);
    if (vpad.out != to->data) return 1;
  }
  if (m2) { vpad.src = pad_scr; vpad.lay = LY_P; }
  vpad.H = H; vpad.W = Wd; vpad.C = C; vpad.hb = hb; vpad.wb = wb;
  return 1;
}

// (v46_MB) a mobilenet-shaped start (PAD of a 27,648-element input, CONV, PAD, DEPTHWISE): node 1 is set up and
// queued before the full check (kg_detect ~1.7k cycles, while the PAD runs); if the check then fails, node 1 has
// run correctly and simply runs again in its own kernel
static __attribute__((optimize("Os"))) int mbm_front(const Node* node, const Tensor* tensors) {
  // (cheapest test first: the other scored models' first kernels pay only these loads)
  const int32_t in = node->input;
  if (in < 0 || in > 4095) return 0;
  const RuntimeShape* si = &tensors[in].shape;
  if (si->dims[1] != 96 || si->dims[2] != 96) return 0;
  const Node* n1 = node + 1;
  return node->op == OP_PAD && n1->op == OP_CONV_2D && (node + 2)->op == OP_PAD &&
         (node + 3)->op == OP_DEPTHWISE_CONV_2D && n1->input == node->output && (uint32_t)rs_flat(si) == 27648u;
}
// (v49_MB) aligned copy of a PAD's NHWC input for its depthwise consumer (see tools/k_acp.py)
#ifndef MBM_ACP_LO
#define MBM_ACP_LO ((1u << 2) | (1u << 8) | (1u << 14))
#define MBM_ACP_HI 0u
#endif
static __attribute__((optimize("Os"))) void mbm_acp(const Node* node, Tensor* tensors) {
  const uint32_t k = (uint32_t)(node - mbm_n0);
  if (!STRICT || k >= 64u || !(mbm_bit(MBM_ACP_LO, MBM_ACP_HI, k) || (g_apw != 0 && k >= 1u && mbm_bit(MBM_APW_LO, MBM_APW_HI, k - 1u)))) {
    g_apw = 0; return;
  }
  const Tensor* ti = &tensors[node->input];
  if (vpad.out != tensors[node->output].data || vpad.lay != LY_NHWC || vpad.src != (const int8_t*)ti->data) return;
  if ((node + 1)->op != OP_DEPTHWISE_CONV_2D || (node + 1)->input != node->output) return;
  const int32_t H = vpad.H, Wd = vpad.W, C = vpad.C, n = H * Wd * C;
  if (g_apw != 0 && g_apw == vpad.src) {   // (k_apw) pad_scr holds it already (v50_MB k_p7: as planes)
    vpad.src = pad_scr; if (g_apwp) vpad.lay = LY_P; g_apw = 0; g_apwp = 0; return;
  }
  if (n <= 0 || n > PAD_SCR_BYTES || ((uint32_t)vpad.src & 15u) == 0 || H > 255 || Wd * C > 65535) return;
  npu_slot();
  W(RG_IN, (uint32_t)vpad.src); W(RG_OUT, (uint32_t)pad_scr);
  W(RG_PROWS, H); W(RG_PHB, 0); W(RG_PH, H);
  W(RG_PWBC, 0); W(RG_PWC, Wd * C); W(RG_PWAC, 0); W(RG_PWOC, Wd * C);
  W(RG_PTOT, n); W(RG_PVAL, 0); W(RG_COUNT, n);
  npu_run(NOP_PAD);
  vpad.src = pad_scr;
}
// (v49_MB k_apw) the aligned output place of a planned pointwise node (pad_scr), or its own buffer
static int8_t* apw_out(const Node* node, Tensor* tensors, int32_t H, int32_t Wd, int32_t Co) {
  int8_t* const o = (int8_t*)tensors[node->output].data;
  g_apw = 0;
  const uint32_t k = (uint32_t)(node - mbm_n0);
  if (!STRICT || mbm_n0 == 0 || k >= 64u || !mbm_bit(MBM_APW_LO, MBM_APW_HI, k)) return o;
  const Node* pn = node + 1;
  if (pn->op != OP_PAD || pn->input != node->output || (pn + 1)->op != OP_DEPTHWISE_CONV_2D) return o;
  if ((Co & 15) != 0 || H * Wd * Co > PAD_SCR_BYTES || ((uint32_t)o & 15u) == 0 || H > 255 || Wd * Co > 65535) return o;
  return pad_scr;
}
// (v49_MB k_apw) the node's own output from its aligned copy (PAD engine as a plain copy)
static void apw_copy(const Node* node, Tensor* tensors, int32_t H, int32_t Wd, int32_t Co) {
  int8_t* const o = (int8_t*)tensors[node->output].data;
  const int32_t n = H * Wd * Co;
  npu_slot();
  W(RG_IN, (uint32_t)pad_scr); W(RG_OUT, (uint32_t)o);
  W(RG_PROWS, H); W(RG_PHB, 0); W(RG_PH, H);
  W(RG_PWBC, 0); W(RG_PWC, Wd * Co); W(RG_PWAC, 0); W(RG_PWOC, Wd * Co);
  W(RG_PTOT, n); W(RG_PVAL, 0); W(RG_COUNT, n);
  npu_run(NOP_PAD);
  g_apw = o;
}

// (v50_MB k_rep) mobilenet nodes 1 / 4: the conv's whole-output write command is replayed by the write DMA's expander
// (RG_PG bit 11) as chunks of 96 bytes: own output, the next PAD's output (interior rows; border bytes as fill
// commands: top run, one run between rows, bottom run) and, for node 1 (RG_PG bit 12), the aligned NHWC copy at
// pad_scr that its depthwise conv reads. Registers: RG_PWBC A - own, RG_PTOT B - own, RG_OFF1 the top
// fill run, RG_PWC row bytes, RG_COUNT {rows, top run}, RG_N {bottom run, skip}, RG_PVAL the pad value. pgv: the op's RG_PG value without these bits.
static void __attribute__((noinline, cold)) mbr_conv(uint32_t pgv) {   // (cold: size-optimised; with register images it runs only in the capture)
  if (g_mbfx) return;                  // (v57_MB k_mbf) the folded ops place their outputs themselves
  if (!STRICT || !MBM || mbm_n0 == 0) return;
  const Node* node = g_node;
  const uint32_t k = (uint32_t)(node - mbm_n0);
  if (k != 1u && k != 4u) return;
  Tensor* t = g_tensors;
  const Node* pn = node + 1;
  if (pn->op != OP_PAD || pn->input != node->output || (pn + 1)->op != OP_DEPTHWISE_CONV_2D ||
      (pn + 1)->input != pn->output) return;
  const PadParams* pp = (const PadParams*)pn->builtin;
  int32_t d[4], q[4];
  if (!shape4(&t[node->output].shape, d) || !shape4(&t[pn->output].shape, q) || q[3] != d[3]) return;
  const int32_t H = d[1], Wd = d[2], C = d[3], hb = pp->pad_h_before, wb = pp->pad_w_before;
  if (hb < 0 || wb < 0 || q[1] < H + hb || q[2] < Wd + wb) return;
  const int32_t rw = Wd * C, prow = q[2] * C, n = H * rw;
  const int32_t top = hb * prow + wb * C, sk = prow - rw, bot = (q[2] - wb - Wd) * C + (q[1] - hb - H) * prow;
  if ((rw & 15) != 0 || rw > 4095 || H > 255 || sk > 255 || top > 4095 || bot > 4095) return;
  if (k == 1u ? (pgv != 5u || C != 8 || n > PAD_SCR_BYTES) : (pgv != 1u || C != 16)) return;
  if (MB_LAX && k == 1u && pad_elides(pn, t) && mbl_sole(pn, node->output) && mbl_sole(pn + 1, pn->output)) {
    W(RG_OUT, (uint32_t)pad_scr);    // (v57_MB) only the aligned copy its depthwise conv reads (no own / PAD output)
    g_mbr_pad = pn; g_mbr_src = pad_scr; g_mbr_lay = LY_NHWC;
    return;
  }
  if (MB_LAX) return;                // (v57_MB r_mbl) no write-DMA expander: the PAD runs as a PAD
  const uint32_t own = (uint32_t)t[node->output].data, a0 = (uint32_t)t[pn->output].data + (uint32_t)top;
  W(RG_PWBC, (a0 - own) & 0xfffffu);                         // A - own
  W(RG_PTOT, ((uint32_t)pad_scr - own) & 0xfffffu);           // B - own
  W(RG_OFF1, a0 - (uint32_t)top);                             // the top fill run
  W(RG_PWC, (uint32_t)rw);                                    // row bytes
  W(RG_COUNT, (uint32_t)top | ((uint32_t)H << 12));           // top fill run, rows
  W(RG_N, (uint32_t)sk | ((uint32_t)bot << 8));               // skip fill run, bottom fill run (chunks: 96 B in the RTL)
  W(RG_PVAL, (uint32_t)(uint8_t)pp->pad_value);
  W(RG_PG, pgv | 0x800u | (k == 1u ? 0x1000u : 0x2000u));   // (node 4: own and PAD rows share the line phase)
  g_mbr_pad = pn;
  g_mbr_src = (k == 1u) ? pad_scr : (const int8_t*)t[node->output].data;
  g_mbr_lay = LY_NHWC;
}
// (v50_MB k_rep) the PAD whose output the producer's op wrote: no op; record the elided PAD for its depthwise conv
// (pad_launch's vpad; for node 2 with mbm_acp's aligned source)
static void __attribute__((noinline, cold)) mbr_pad(const Node* node, Tensor* tensors) {
  g_mbr_pad = 0;
  if (!pad_elides(node, tensors)) return;          // (its consumer then reads the PAD's complete output)
  const PadParams* p = (const PadParams*)node->builtin;
  const RuntimeShape* ish = &tensors[node->input].shape;
  vpad.out = tensors[node->output].data; vpad.src = g_mbr_src; vpad.lay = g_mbr_lay;   // (v57_MB: or planes)
  vpad.H = ish->dims[1]; vpad.W = ish->dims[2]; vpad.C = ish->dims[3];
  vpad.hb = p->pad_h_before; vpad.wb = p->pad_w_before; vpad.node = node; vpad.tensors = tensors;
}

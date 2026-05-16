/* ******************************************************************************
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include <ops/OpTraitTable.h>
#include <ops/declarable/OpRegistrator.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <string>
#include <unordered_map>

namespace sd {
namespace ops {

// ─── Trait table ───────────────────────────────────────────────────────────────
//
// SINGLE SOURCE OF TRUTH for op classification.
//
// Each entry maps a lowercase op name to the bitwise OR of its OpTraits.
// Traits auto-derived from the class hierarchy (BroadcastableOp, BroadcastableBoolOp,
// DeclarableReductionOp) are already set in constructors and are preserved here
// because initOpTraits() uses addTraits() (OR), not setTraits() (replace).
//
// To add a new op: add ONE entry here. All consumers (NativePlanCompiler,
// FusionPass, NativeDynamicShapePlan, etc.) will pick it up automatically.
//
// ─────────────────────────────────────────────────────────────────────────────

// Shorthand constants for common trait combinations
static constexpr uint32_t UNARY_EW = OP_TRAIT_UNARY_ELEMENTWISE | OP_TRAIT_FULLY_WRITING;
static constexpr uint32_t UNARY_ACT = UNARY_EW | OP_TRAIT_ACTIVATION;
static constexpr uint32_t BINARY_EW = OP_TRAIT_BINARY_ELEMENTWISE | OP_TRAIT_FULLY_WRITING;
static constexpr uint32_t BINARY_CMP = BINARY_EW | OP_TRAIT_COMPARISON;
static constexpr uint32_t BINARY_LOG = BINARY_EW | OP_TRAIT_LOGICAL;
static constexpr uint32_t TERNARY_EW = OP_TRAIT_TERNARY_ELEMENTWISE | OP_TRAIT_FULLY_WRITING;
static constexpr uint32_t REDUCE = OP_TRAIT_REDUCTION | OP_TRAIT_FULLY_WRITING;
static constexpr uint32_t NORM = OP_TRAIT_NORMALIZATION | OP_TRAIT_FULLY_WRITING;
static constexpr uint32_t MATMUL = OP_TRAIT_MATMUL | OP_TRAIT_FULLY_WRITING;
// VIEW_SHAPE_DEP: shape-manipulating view whose output shape is fully determined by
// input shapes + iArgs (no tensor-value read). Examples: expand_dims, squeeze,
// flatten, permute. These ops MUST NOT carry VALUE_DEPENDENT_SHAPE — otherwise
// the SHAPES_FROZEN slot executor treats input-shape changes as value-dep errors
// instead of taking the "shape changed" recovery path (POST_SEAL_SHAPE_CHANGE).
static constexpr uint32_t VIEW_SHAPE_DEP = OP_TRAIT_VIEW_PRODUCING;
// VIEW: view-producing AND value-dependent (shape comes from an input TENSOR,
// e.g., reshape(x, shapeTensor), reshape_no_copy, strided_slice). Use this only
// when the shape fn actually dereferences input tensor data.
static constexpr uint32_t VIEW = OP_TRAIT_VIEW_PRODUCING | OP_TRAIT_VALUE_DEPENDENT_SHAPE;
static constexpr uint32_t VALDEP = OP_TRAIT_VALUE_DEPENDENT_SHAPE;
static constexpr uint32_t DATADEP = OP_TRAIT_DATA_DEPENDENT;
static constexpr uint32_t SHAPE_ONLY = OP_TRAIT_SHAPE_ONLY_OUTPUT;
static constexpr uint32_t IDENT = OP_TRAIT_IDENTITY | OP_TRAIT_UNARY_ELEMENTWISE | OP_TRAIT_FULLY_WRITING;
static constexpr uint32_t DATA_MOVE = OP_TRAIT_DATA_MOVEMENT | OP_TRAIT_FULLY_WRITING;
static constexpr uint32_t DATA_MOVE_VALDEP = DATA_MOVE | OP_TRAIT_VALUE_DEPENDENT_SHAPE;
static constexpr uint32_t CONST_GEN = OP_TRAIT_CONSTANT_GENERATION | OP_TRAIT_FULLY_WRITING;
static constexpr uint32_t CONST_GEN_VALDEP = CONST_GEN | OP_TRAIT_VALUE_DEPENDENT_SHAPE;
static constexpr uint32_t ATTN = OP_TRAIT_ATTENTION | OP_TRAIT_FULLY_WRITING;
// GATHER / GATHER_ND: DATA_MOVE already includes FULLY_WRITING.
// The gather kernel iterates exactly over output length (numIndices * TAD_size),
// writing every element of the allocated output buffer. In frozen replay, output
// shapes are identical between steps, so the buffer size always matches the logical
// output shape — no stale tail data can leak. needsZeroedOutput=false is correct.
//
// IMPORTANT: gather's output shape is determined by input SHAPES ONLY
// (output = indices.shape + params.shape[1:]). The shape fn does not dereference
// indices tensor DATA. Therefore gather/gather_nd must NOT carry VALUE_DEPENDENT_SHAPE —
// when input shapes change (e.g., batch size 1→18), the SHAPES_FROZEN path should
// hit the non-value-dep branch (POST_SEAL_SHAPE_CHANGE recovery), NOT the value-dep
// shape-match assertion. Same applies to repeat (repeats come from iArgs).
static constexpr uint32_t GATHER = DATA_MOVE | OP_TRAIT_GATHER;
static constexpr uint32_t GATHER_ND = DATA_MOVE | OP_TRAIT_GATHER_ND;
static constexpr uint32_t CONCAT = DATA_MOVE | OP_TRAIT_CONCAT;
static constexpr uint32_t SPLIT = DATA_MOVE | OP_TRAIT_SPLIT;
static constexpr uint32_t SPLIT_V = DATA_MOVE | OP_TRAIT_SPLIT_V;
static constexpr uint32_t STACK = DATA_MOVE | OP_TRAIT_STACK;
static constexpr uint32_t SLICE = DATA_MOVE_VALDEP | OP_TRAIT_SLICE;
static constexpr uint32_t TILE = DATA_MOVE_VALDEP | OP_TRAIT_TILE;
// Partial writers: only write at scatter indices, leave other positions stale.
// Must be zeroed before execution if downstream reads the whole buffer.
static constexpr uint32_t SCATTER_PARTIAL = OP_TRAIT_DATA_MOVEMENT;
static constexpr uint32_t SCATTER_ND = SCATTER_PARTIAL | OP_TRAIT_SCATTER_ND;
// scatter_nd_update is a FULL writer: it does output->assign(input) first (copies
// every element), THEN scatters updates at specific indices. The assign step fully
// writes the output, so no prezero is needed. FULLY_WRITING reflects this.
static constexpr uint32_t SCATTER_ND_UPDATE = SCATTER_PARTIAL | OP_TRAIT_SCATTER_ND_UPDATE | OP_TRAIT_FULLY_WRITING;
// BP: modifier applied to backward / gradient ops. Combined with the primary trait so
// that the category lookup still returns the correct TritonOpCategory while profiling
// and diagnostic code can distinguish forward from backward via OP_TRAIT_BACKWARD.
static constexpr uint32_t BP = OP_TRAIT_BACKWARD;

static const std::unordered_map<std::string, uint32_t>& getTraitTable() {
    static const std::unordered_map<std::string, uint32_t> TABLE = {
        // ── Unary elementwise math ─────────────────────────────────────────
        {"abs",            UNARY_EW},
        {"neg",            UNARY_EW},
        {"exp",            UNARY_EW},
        {"log",            UNARY_EW},
        {"log1p",          UNARY_EW},
        {"sqrt",           UNARY_EW},
        {"rsqrt",          UNARY_EW},
        {"square",         UNARY_EW},
        {"reciprocal",     UNARY_EW},
        {"ceil",           UNARY_EW},
        {"floor",          UNARY_EW},
        {"round",          UNARY_EW},
        {"sign",           UNARY_EW},
        {"erf",            UNARY_EW},
        {"erfc",           UNARY_EW},
        {"sin",            UNARY_EW},
        {"cos",            UNARY_EW},
        {"asin",           UNARY_EW},
        {"acos",           UNARY_EW},
        {"atan",           UNARY_EW},
        {"sinh",           UNARY_EW},
        {"cosh",           UNARY_EW},
        {"asinh",          UNARY_EW},
        {"acosh",          UNARY_EW},
        {"atanh",          UNARY_EW},

        // ── Activation functions (unary elementwise + activation tag) ──────
        {"relu",           UNARY_ACT},
        {"relu6",          UNARY_ACT},
        {"leakyrelu",      UNARY_ACT},
        {"elu",            UNARY_ACT},
        {"selu",           UNARY_ACT},
        {"gelu",           UNARY_ACT},
        {"sigmoid",        UNARY_ACT},
        {"tanh",           UNARY_ACT},
        {"softsign",       UNARY_ACT},
        {"softplus",       UNARY_ACT},
        {"swish",          UNARY_ACT},
        {"silu",           UNARY_ACT},
        {"mish",           UNARY_ACT},
        {"hard_sigmoid",   UNARY_ACT},
        {"hardtanh",       UNARY_ACT},
        {"fused_gelu",     UNARY_ACT},

        // ── Logical unary ──────────────────────────────────────────────────
        {"boolean_not",    UNARY_EW | OP_TRAIT_LOGICAL},
        {"logical_not",    UNARY_EW | OP_TRAIT_LOGICAL},

        // ── Identity / copy ────────────────────────────────────────────────
        {"identity",       IDENT},
        {"assign",         IDENT},

        // ── Cast ───────────────────────────────────────────────────────────
        {"cast",           UNARY_EW | OP_TRAIT_CAST},

        // ── Clip ───────────────────────────────────────────────────────────
        {"clipbyvalue",    UNARY_EW},

        // ── Binary elementwise ─────────────────────────────────────────────
        // NOTE: Most of these are BroadcastableOp subclasses and get
        // BINARY_ELEMENTWISE | FULLY_WRITING auto-derived from the constructor.
        // Entries here ensure they also show up for ops registered via
        // DECLARE_CUSTOM_OP or other non-BroadcastableOp paths.
        {"add",            BINARY_EW},
        {"subtract",       BINARY_EW},
        {"multiply",       BINARY_EW},
        {"divide",         BINARY_EW},
        {"floormod",       BINARY_EW},
        {"floordiv",       BINARY_EW},
        {"reversedivide",  BINARY_EW},
        {"reversesubtract", BINARY_EW},
        {"squaredsubtract", BINARY_EW},
        {"add_scalar",     BINARY_EW},
        {"subtract_scalar", BINARY_EW},
        {"multiply_scalar", BINARY_EW},
        {"divide_scalar",  BINARY_EW},
        {"pow",            BINARY_EW},
        {"min_pairwise",   BINARY_EW},
        {"max_pairwise",   BINARY_EW},
        {"atan2",          BINARY_EW},
        {"maximum",        BINARY_EW},
        {"minimum",        BINARY_EW},
        {"mod",            BINARY_EW},
        {"multiply_no_nan", BINARY_EW},
        {"swish_mul",      BINARY_EW},

        // ── Binary comparison ──────────────────────────────────────────────
        // NOTE: These are BroadcastableBoolOp subclasses → auto-derived.
        {"equals",         BINARY_CMP},
        {"not_equals",     BINARY_CMP},
        {"less",           BINARY_CMP},
        {"less_equal",     BINARY_CMP},
        {"greater",        BINARY_CMP},
        {"greater_equal",  BINARY_CMP},

        // ── Binary logical ─────────────────────────────────────────────────
        {"boolean_and",    BINARY_LOG},
        {"boolean_or",     BINARY_LOG},
        {"boolean_xor",    BINARY_LOG},
        {"logical_and",    BINARY_LOG},
        {"logical_or",     BINARY_LOG},

        // ── Ternary elementwise ────────────────────────────────────────────
        {"where",          TERNARY_EW | OP_TRAIT_DATA_DEPENDENT},
        // NOTE: "where" with 3 inputs is ternary elementwise (select).
        //       "where" with 1 input is data-dependent (coordinate extraction).
        //       The DATA_DEPENDENT trait is set here; NativePlanCompiler clears it
        //       for the 3-input case at compile time (runtime input count check).
        {"select",         TERNARY_EW},

        // ── Matrix ops ─────────────────────────────────────────────────────
        {"matmul",         MATMUL},
        {"mmul",           MATMUL},
        {"batched_gemm",   MATMUL},
        {"tensormmul",     MATMUL},
        {"fp8_matmul",     MATMUL},
        {"smooth_quant",   MATMUL},
        {"awq_matmul",     MATMUL},
        {"column_parallel_linear", MATMUL},
        {"row_parallel_linear",    MATMUL},
        {"multi_lora_matmul",      MATMUL},
        {"xw_plus_b",              MATMUL},
        {"fused_gemm_swiglu",      MATMUL},
        {"fused_two_layer_mlp",    MATMUL},

        // ── Reduction ops ──────────────────────────────────────────────────
        // NOTE: Many are DeclarableReductionOp subclasses → auto-derived.
        {"reduce_sum",     REDUCE},
        {"reduce_mean",    REDUCE},
        {"reduce_max",     REDUCE},
        {"reduce_min",     REDUCE},
        {"reduce_prod",    REDUCE},
        {"reduce_norm1",   REDUCE},
        {"reduce_norm2",   REDUCE},
        {"reduce_logsumexp", REDUCE},
        {"reduce_variance", REDUCE},
        {"reduce_stdev",   REDUCE},
        {"sum",            REDUCE},
        {"mean",           REDUCE},
        {"max",            REDUCE},
        {"min",            REDUCE},
        {"prod",           REDUCE},
        {"norm1",          REDUCE},
        {"norm2",          REDUCE},
        {"normmax",        REDUCE},
        {"argmax",         REDUCE | DATADEP},  // shape fn reads axes from INPUT_VARIABLE(1) when 2-input form
        {"argmin",         REDUCE | DATADEP},  // shape fn reads axes from INPUT_VARIABLE(1) when 2-input form

        // ── Normalization ops ──────────────────────────────────────────────
        {"softmax",        NORM},
        {"log_softmax",    NORM},
        {"layer_norm",     NORM},
        {"fused_layer_norm", NORM},
        {"batch_norm",     NORM},
        {"batchnorm",      NORM},
        {"rms_norm",       NORM},
        {"rms_norm_linear", MATMUL},
        {"skip_rms_norm",  NORM},
        {"normalize_moments", NORM},
        {"fused_rope",     NORM},

        // ── Attention ops ──────────────────────────────────────────────────
        {"onnx_multi_head_attention",       ATTN},
        {"dot_product_attention_v2",        ATTN},
        {"flash_attention",                 ATTN},
        {"multi_head_dot_product_attention", ATTN},
        {"multi_head_attention",            ATTN},

        // ── Token sampling ─────────────────────────────────────────────────
        {"token_sample",   OP_TRAIT_FULLY_WRITING},

        // ── View-producing ops ─────────────────────────────────────────────
        // VIEW (VALDEP): shape fn dereferences an input TENSOR's data.
        //   reshape/reshape_no_copy take a shape tensor; strided_slice reads begin/end/stride tensors.
        // VIEW_SHAPE_DEP: shape fn uses only input shapes + iArgs — no data read.
        //   expand_dims/squeeze/flatten/flatten_2d/permute all fall in this class.
        {"reshape",        VIEW | DATADEP},     // shape fn calls INPUT_VARIABLE(1)->asVectorT() for shape tensor
        {"reshape_no_copy", VIEW | DATADEP},   // same as reshape
        {"strided_slice",  VIEW | OP_TRAIT_SLICE},
        {"expand_dims",    VIEW_SHAPE_DEP | DATADEP},  // shape fn reads INPUT_VARIABLE(1)->e<>() when no INT_ARG
        {"squeeze",        VIEW_SHAPE_DEP},
        {"flatten",        VIEW_SHAPE_DEP},
        {"flatten_2d",     VIEW_SHAPE_DEP},
        {"permute",        VIEW_SHAPE_DEP},

        // ── Shape-determined data movement (shape fn reads input shapes + iArgs only) ─
        // These MUST NOT carry VALUE_DEPENDENT_SHAPE — the SHAPES_FROZEN check relies on
        // the flag being accurate. A false positive causes the value-dep shape-match
        // branch to report "value-dependent output shape changed" when in fact the input
        // shape changed (e.g., gather indices went [1,512] → [18,512]).
        {"gather",         GATHER},              // output shape = indices.shape + params.shape[axis+1:]; axis is a structural scalar, not value-dep
        {"gather_nd",      GATHER_ND},
        {"concat",         CONCAT | DATADEP},   // shape fn reads INPUT_VARIABLE(last)->e<>() for axis when isAxisInLastArr
        {"stack",          STACK},
        {"split",          SPLIT},
        {"split_v",        SPLIT_V},
        {"repeat",         DATA_MOVE},

        // ── Value-dependent data movement (shape fn reads tensor DATA) ─────
        {"slice",          SLICE},           // dual-mode: iArg or tensor begin/size; conservative VALDEP
        {"tile",           TILE},            // dual-mode: iArg or tensor multiples; conservative VALDEP
        {"pad",            DATA_MOVE_VALDEP},
        {"fill",           DATA_MOVE_VALDEP | DATADEP},  // shape fn reads shape array elements from INPUT_VARIABLE(0)
        {"broadcast_to",   DATA_MOVE_VALDEP},
        {"scatter_nd",     SCATTER_ND},
        {"scatter_nd_update", SCATTER_ND_UPDATE},
        {"range",          CONST_GEN_VALDEP | DATADEP},    // shape fn reads start/limit/delta element values
        {"linspace",       CONST_GEN_VALDEP | DATADEP},   // shape fn reads steps value from INPUT_VARIABLE(2)
        // create/ConstantOfShape materializes a real output buffer. Some instances feed
        // shape-control ladders, but that must be inferred from runtime tensor semantics,
        // not baked into the op as globally shape-only.
        {"create",         CONST_GEN_VALDEP},

        // ── Data-dependent ops (variable-length output or shape reads tensor data) ──
        {"unique",                      DATADEP},
        {"non_max_suppression",         DATADEP},
        {"non_max_suppression_v3",      DATADEP},
        {"non_max_suppression_overlaps", DATADEP},  // shape fn runs full NMS to determine output length
        // Space/batch transforms: shape fn reads block shape, crop, and pad tensor values
        {"batch_to_space",    DATA_MOVE | DATADEP},
        {"space_to_batch",    DATA_MOVE | DATADEP},
        {"batch_to_space_nd", DATA_MOVE | DATADEP},
        {"space_to_batch_nd", DATA_MOVE | DATADEP},
        // Generator ops whose shape depends on tensor data values
        {"randomuniform",          CONST_GEN_VALDEP | DATADEP},  // shape fn reads shape tensor via asVectorT()
        {"lin_space",              CONST_GEN_VALDEP | DATADEP},  // shape fn reads steps from INPUT_VARIABLE(2)
        {"evaluate_reduction_shape", SHAPE_ONLY | CONST_GEN | DATADEP},  // reads axes from INPUT_VARIABLE(1)->asVectorT()
        // Ops whose output shape depends on k/depth/class-count tensor values
        {"top_k",       OP_TRAIT_FULLY_WRITING | DATADEP},  // shape fn reads k from INPUT_VARIABLE(1)
        {"onehot",      CONST_GEN_VALDEP | DATADEP},        // shape fn reads depth from INPUT_VARIABLE(1)
        {"bincount",    REDUCE | DATADEP},                  // shape fn calls argMax() + reads min/max element values
        // Unsorted segment ops: shape fn reads numOfClasses from INPUT_VARIABLE(2)
        {"unsorted_segment_max",    REDUCE | DATADEP},
        {"unsorted_segment_mean",   REDUCE | DATADEP},
        {"unsorted_segment_min",    REDUCE | DATADEP},
        {"unsorted_segment_prod",   REDUCE | DATADEP},
        {"unsorted_segment_sqrt_n", REDUCE | DATADEP},
        {"unsorted_segment_sum",    REDUCE | DATADEP},
        // Conv backward ops whose output shape is read from an input tensor
        {"conv2d_input_bp", OP_TRAIT_FULLY_WRITING | OP_TRAIT_BACKWARD | DATADEP},  // reads gradIShape from INPUT_VARIABLE(0)
        {"deconv2d_tf",     OP_TRAIT_FULLY_WRITING | OP_TRAIT_BACKWARD | DATADEP},  // reads gradIShape from INPUT_VARIABLE(0)

        // ── Shape-only output (output depends only on input shapes) ────────
        {"shape_of",       SHAPE_ONLY | CONST_GEN},
        {"size_at",        SHAPE_ONLY | CONST_GEN},
        {"rank",           SHAPE_ONLY | CONST_GEN},
        {"zeros_like",     SHAPE_ONLY | CONST_GEN},
        {"zeros_as",       SHAPE_ONLY | CONST_GEN},
        {"zeroslike",      SHAPE_ONLY | CONST_GEN},
        {"ones_like",      SHAPE_ONLY | CONST_GEN},
        {"ones_as",        SHAPE_ONLY | CONST_GEN},
        {"oneslike",       SHAPE_ONLY | CONST_GEN},

        // ── LLM attention ops (forward + backprop) ─────────────────────────
        {"dot_product_attention",               ATTN},
        {"dot_product_attention_bp",            ATTN | BP},
        {"dot_product_attention_v2_bp",         ATTN | BP},
        {"multi_head_dot_product_attention_bp", ATTN | BP},
        {"flash_attention_bp",                  ATTN | BP},
        {"grouped_query_attention",             ATTN},
        {"grouped_query_attention_bp",          ATTN | BP},
        {"sliding_window_attention",            ATTN},
        {"shared_kv_attention",                 ATTN},
        {"windowed_attention",                  ATTN},
        {"paged_attention_forward",             ATTN},
        {"turbo_quant_attention",               ATTN},
        {"two_way_cross_attention",             ATTN},
        {"two_way_cross_attention_bp",          ATTN | BP},
        {"vlm_cross_attention",                 ATTN},
        {"cross_attention",                     ATTN},
        {"vision_language_cross_attention",      ATTN},
        {"apply_alibi",                         ATTN},
        {"relative_position_bias",              ATTN},

        // ── KV cache management ────────────────────────────────────────────
        {"kv_cache_update",     DATA_MOVE_VALDEP},
        {"kv_cache_quantize",   CONST_GEN_VALDEP},
        {"kv_cache_dequantize", UNARY_EW},
        {"paged_kv_append",     DATA_MOVE_VALDEP},

        // ── Rotary / positional embedding ──────────────────────────────────
        {"rope",         NORM},
        {"rope_bp",      NORM | BP},
        {"fused_rope_bp", NORM | BP},
        {"dual_rope",    NORM},

        // ── Normalization backprop / fused variants ────────────────────────
        {"rms_norm_bp",             NORM | BP},
        {"rms_norm_linear_bp",      NORM | BP},
        {"fused_layer_norm_bp",     NORM | BP},
        {"fused_rms_norm_swiglu",   NORM},
        {"fused_rms_norm_swiglu_bp", NORM | BP},

        // ── Fused GEMM / SwiGLU ────────────────────────────────────────────
        {"fused_gemm_swiglu_bp", MATMUL | BP},

        // ── Activation backprop + novel activations ────────────────────────
        {"silu_bp",         UNARY_ACT | BP},
        {"fused_gelu_bp",   UNARY_ACT | BP},
        {"squared_relu",    UNARY_ACT},
        {"squared_relu_bp", UNARY_ACT | BP},
        {"gated_delta_rule", UNARY_ACT},

        // ── Mamba / selective scan / SSM / causal conv ─────────────────────
        {"gated_delta_net_block", REDUCE},
        {"selective_scan",        REDUCE},
        {"mamba2_ssm",            REDUCE},
        {"causal_conv1d",         UNARY_EW},

        // ── Fused training kernels ─────────────────────────────────────────
        {"fused_bias_dropout_residual", UNARY_EW},
        {"fused_elementwise_chain",     UNARY_EW},
        {"swish_mul_bp",                BINARY_EW | BP},
        {"center_and_sharpen",          UNARY_EW},
        {"center_and_sharpen_bp",       UNARY_EW | BP},
        {"ema_update",                  DATA_MOVE},
        {"ema_update_bp",               DATA_MOVE | BP},

        // ── Quantization / adapter matmuls ─────────────────────────────────
        {"quantized_matmul", MATMUL},
        {"dora_matmul",      MATMUL},
        {"dora_matmul_bp",   MATMUL | BP},
        {"lora_matmul",      MATMUL},
        {"lora_matmul_bp",   MATMUL | BP},
        {"loha_matmul",      MATMUL},
        {"loha_matmul_bp",   MATMUL | BP},
        {"lokr_matmul",      MATMUL},
        {"lokr_matmul_bp",   MATMUL | BP},

        // ── GGML / per-layer embedding / misc ──────────────────────────────
        {"ggml_dequantize",     UNARY_EW},
        {"per_layer_embedding", DATA_MOVE_VALDEP},

        // ── Convolution forward + backward ────────────────────────────────
        // No OP_TRAIT_CONVOLUTION exists — use FULLY_WRITING as the minimal
        // structural trait. The Triton layer routes these via buildOpTable()
        // to TritonOpCategory::CONVOLUTION explicitly.
        {"conv2d",    OP_TRAIT_FULLY_WRITING},
        {"deconv2d",  OP_TRAIT_FULLY_WRITING},
        {"im2col",    OP_TRAIT_FULLY_WRITING},
        {"col2im",    OP_TRAIT_FULLY_WRITING},
        {"conv2d_bp",    OP_TRAIT_FULLY_WRITING | BP},
        {"deconv2d_bp",  OP_TRAIT_FULLY_WRITING | BP},
        {"im2col_bp",    OP_TRAIT_FULLY_WRITING | BP},
        {"col2im_bp",    OP_TRAIT_FULLY_WRITING | BP},

        // ── Loss ops ───────────────────────────────────────────────────────
        {"absolute_difference_loss",                                REDUCE},
        {"absolute_difference_loss_grad",                           REDUCE | BP},
        {"attention_distillation_loss",                             REDUCE},
        {"attention_distillation_loss_grad",                        REDUCE | BP},
        {"contrastive_loss",                                        REDUCE},
        {"contrastive_loss_grad",                                   REDUCE | BP},
        {"cosine_distance_loss",                                    REDUCE},
        {"cosine_distance_loss_grad",                               REDUCE | BP},
        {"ctc_loss",                                                REDUCE},
        {"ctc_loss_grad",                                           REDUCE | BP},
        {"distillation_kl_loss",                                    REDUCE},
        {"distillation_kl_loss_grad",                               REDUCE | BP},
        {"feature_distillation_loss",                               REDUCE},
        {"feature_distillation_loss_grad",                          REDUCE | BP},
        {"hinge_loss",                                              REDUCE},
        {"hinge_loss_grad",                                         REDUCE | BP},
        {"huber_loss",                                              REDUCE},
        {"huber_loss_grad",                                         REDUCE | BP},
        {"kl_divergence_per_layer",                                 REDUCE},
        {"l2_loss",                                                 REDUCE},
        {"log_loss",                                                REDUCE},
        {"log_loss_grad",                                           REDUCE | BP},
        {"log_poisson_loss",                                        REDUCE},
        {"log_poisson_loss_grad",                                   REDUCE | BP},
        {"mean_pairwssqerr_loss",                                   REDUCE},
        {"mean_pairwssqerr_loss_grad",                              REDUCE | BP},
        {"mean_sqerr_loss",                                         REDUCE},
        {"mean_sqerr_loss_grad",                                    REDUCE | BP},
        {"sigm_cross_entropy_loss",                                 REDUCE},
        {"sigm_cross_entropy_loss_grad",                            REDUCE | BP},
        {"simpo_loss",                                              REDUCE},
        {"simpo_loss_bp",                                           REDUCE | BP},
        {"softmax_cross_entropy_loss",                              REDUCE},
        {"softmax_cross_entropy_loss_grad",                         REDUCE | BP},
        {"softmax_cross_entropy_loss_with_logits",                  REDUCE},
        {"softmax_cross_entropy_loss_with_logits_grad",             REDUCE | BP},
        {"sparse_softmax_cross_entropy_loss_with_logits",           REDUCE},
        {"sparse_softmax_cross_entropy_loss_with_logits_grad",      REDUCE | BP},
        {"weighted_cross_entropy_with_logits",                      REDUCE},

        // ── Activation backward ops ────────────────────────────────────────
        {"alpha_dropout_bp",    UNARY_ACT | BP},
        {"crelu",               UNARY_ACT},
        {"crelu_bp",            UNARY_ACT | BP},
        {"cube",                UNARY_EW},
        {"cube_bp",             UNARY_EW | BP},
        {"dropout",             UNARY_EW},
        {"dropout_bp",          UNARY_EW | BP},
        {"elu_bp",              UNARY_ACT | BP},
        {"hardsigmoid",         UNARY_ACT},
        {"hardsigmoid_bp",      UNARY_ACT | BP},
        {"hardtanh_bp",         UNARY_ACT | BP},
        {"identity_bp",         IDENT | BP},
        {"lrelu",               UNARY_ACT},
        {"lrelu_bp",            UNARY_ACT | BP},
        {"log_softmax_bp",      NORM | BP},
        {"prelu",               UNARY_ACT},
        {"prelu_bp",            UNARY_ACT | BP},
        {"rationaltanh",        UNARY_ACT},
        {"rationaltanh_bp",     UNARY_ACT | BP},
        {"rectifiedtanh",       UNARY_ACT},
        {"rectifiedtanh_bp",    UNARY_ACT | BP},
        {"relu6_bp",            UNARY_ACT | BP},
        {"relu_bp",             UNARY_ACT | BP},
        {"relugrad",            UNARY_ACT | BP},
        {"selu_bp",             UNARY_ACT | BP},
        {"sigmoid_bp",          UNARY_ACT | BP},
        {"softmax_bp",          NORM | BP},
        {"softplus_bp",         UNARY_ACT | BP},
        {"softsign_bp",         UNARY_ACT | BP},
        {"tanh_bp",             UNARY_ACT | BP},
        {"thresholdedrelu",     UNARY_ACT},
        {"thresholdedrelu_bp",  UNARY_ACT | BP},

        // ── Fused activation ops ───────────────────────────────────────────
        {"gelu_and_mul",        BINARY_EW},
        {"silu_and_mul",        BINARY_EW},

        // ── Unary math with backward ───────────────────────────────────────
        {"digamma",             UNARY_EW},
        {"lgamma",              UNARY_EW},
        {"polygamma",           BINARY_EW},
        {"rint",                UNARY_EW},
        {"zeta",                BINARY_EW},
        {"betainc",             BINARY_EW},

        // ── Type conversion (cast variants) ───────────────────────────────
        {"to_double",           UNARY_EW | OP_TRAIT_CAST},
        {"to_float16",          UNARY_EW | OP_TRAIT_CAST},
        {"to_float32",          UNARY_EW | OP_TRAIT_CAST},
        {"to_int32",            UNARY_EW | OP_TRAIT_CAST},
        {"to_int64",            UNARY_EW | OP_TRAIT_CAST},
        {"to_uint32",           UNARY_EW | OP_TRAIT_CAST},
        {"to_uint64",           UNARY_EW | OP_TRAIT_CAST},
        {"bitcast",             UNARY_EW | OP_TRAIT_CAST},
        {"toggle_bits",         UNARY_EW},
        {"min_max_datatype",    CONST_GEN},
        {"cast_and_scale",      UNARY_EW | OP_TRAIT_CAST},

        // ── Binary backward ops ────────────────────────────────────────────
        {"add_bp",              BINARY_EW | BP},
        {"divide_bp",           BINARY_EW | BP},
        {"floordiv_bp",         BINARY_EW | BP},
        {"floormod_bp",         BINARY_EW | BP},
        {"maximum_bp",          BINARY_EW | BP},
        {"minimum_bp",          BINARY_EW | BP},
        {"mod_bp",              BINARY_EW | BP},
        {"multiply_bp",         BINARY_EW | BP},
        {"pow_bp",              BINARY_EW | BP},
        {"realdiv_bp",          BINARY_EW | BP},
        {"reversedivide_bp",    BINARY_EW | BP},
        {"reversemod_bp",       BINARY_EW | BP},
        {"reversesubtract_bp",  BINARY_EW | BP},
        {"squaredsubtract_bp",  BINARY_EW | BP},
        {"subtract_bp",         BINARY_EW | BP},
        {"bias_add",            BINARY_EW},
        {"biasadd",             BINARY_EW},
        {"biasadd_bp",          BINARY_EW | BP},
        {"biasaddgrad",         BINARY_EW | BP},

        // ── Comparison scalar ops ──────────────────────────────────────────
        {"eq_scalar",           BINARY_CMP},
        {"neq_scalar",          BINARY_CMP},
        {"gt_scalar",           BINARY_CMP},
        {"gte_scalar",          BINARY_CMP},
        {"lt_scalar",           BINARY_CMP},
        {"lte_scalar",          BINARY_CMP},
        {"greaterorequals",     BINARY_CMP},
        {"lessorequals",        BINARY_CMP},
        {"notequals",           BINARY_CMP},

        // ── Reduction backward ops ─────────────────────────────────────────
        {"reduce_dot_bp",          REDUCE | BP},
        {"reduce_max_bp",          REDUCE | BP},
        {"reduce_mean_bp",         REDUCE | BP},
        {"reduce_min_bp",          REDUCE | BP},
        {"reduce_norm1_bp",        REDUCE | BP},
        {"reduce_norm2_bp",        REDUCE | BP},
        {"reduce_norm_max",        REDUCE},
        {"reduce_norm_max_bp",     REDUCE | BP},
        {"reduce_prod_bp",         REDUCE | BP},
        {"reduce_sqnorm",          REDUCE},
        {"reduce_sqnorm_bp",       REDUCE | BP},
        {"reduce_stdev_bp",        REDUCE | BP},
        {"reduce_sum_bp",          REDUCE | BP},
        {"reduce_variance_bp",     REDUCE | BP},
        {"moments",                REDUCE},
        {"mean_square",            REDUCE},
        {"mean_square_bp",         REDUCE | BP},
        {"sufficient_statistics",  REDUCE},
        {"zero_fraction",          REDUCE},

        // ── Accumulate / merge ops ─────────────────────────────────────────
        {"accumulate_n",    REDUCE},
        {"accumulaten",     REDUCE},
        {"add_n",           REDUCE},
        {"addn",            REDUCE},
        {"mergeadd",        REDUCE},
        {"mergeadd_bp",     REDUCE | BP},
        {"mergeavg",        REDUCE},
        {"mergeavg_bp",     REDUCE | BP},
        {"mergemax",        REDUCE},
        {"mergemax_bp",     REDUCE | BP},
        {"mergemaxindex",   REDUCE | DATADEP},
        {"mergesum",        REDUCE},
        {"dot",             REDUCE},
        {"cross",           BINARY_EW},
        {"tensordot",       MATMUL},

        // ── Matrix ops backward ───────────────────────────────────────────
        {"matmul_bp",          MATMUL | BP},
        {"tensormmul_bp",      MATMUL | BP},
        {"batched_gemm_bp",    MATMUL | BP},
        {"xw_plus_b_bp",       MATMUL | BP},
        {"gemm",               MATMUL},
        {"gemv",               MATMUL},
        {"axpy",               BINARY_EW},
        {"linear",             MATMUL},
        {"lineargrad",         MATMUL | BP},
        {"relu_layer",         UNARY_ACT},

        // ── Normalization backprop ─────────────────────────────────────────
        {"batchnorm_bp",                NORM | BP},
        {"fused_batch_norm",            NORM},
        {"layer_norm_bp",               NORM | BP},
        {"lrn",                         NORM},
        {"lrn_bp",                      NORM | BP},
        {"local_response_normalization", NORM},
        {"standardize",                 NORM},
        {"standardize_bp",              NORM | BP},

        // ── Clip ops ──────────────────────────────────────────────────────
        {"clipbyavgnorm",       UNARY_EW},
        {"clipbyavgnorm_bp",    UNARY_EW | BP},
        {"clip_by_global_norm", UNARY_EW},
        {"clipbynorm",          UNARY_EW},
        {"clipbynorm_bp",       UNARY_EW | BP},

        // ── Pooling forward ────────────────────────────────────────────────
        {"avgpool",             OP_TRAIT_FULLY_WRITING},
        {"avgpool2d",           OP_TRAIT_FULLY_WRITING},
        {"avgpool3dnew",        OP_TRAIT_FULLY_WRITING},
        {"adaptive_avgpool2d",  OP_TRAIT_FULLY_WRITING},
        {"adaptive_avgpool3d",  OP_TRAIT_FULLY_WRITING},
        {"adaptive_maxpool2d",  OP_TRAIT_FULLY_WRITING},
        {"maxpool",             OP_TRAIT_FULLY_WRITING},
        {"maxpool2d",           OP_TRAIT_FULLY_WRITING},
        {"maxpool3dnew",        OP_TRAIT_FULLY_WRITING},
        {"max_pool_with_argmax", OP_TRAIT_FULLY_WRITING},
        {"pnormpool",           OP_TRAIT_FULLY_WRITING},
        {"pnormpool2d",         OP_TRAIT_FULLY_WRITING},
        {"sconv2d",             OP_TRAIT_FULLY_WRITING},

        // ── Pooling backward ───────────────────────────────────────────────
        {"avgpool2d_bp",        OP_TRAIT_FULLY_WRITING | BP},
        {"avgpool3dnew_bp",     OP_TRAIT_FULLY_WRITING | BP},
        {"adaptive_avgpool2d_bp", OP_TRAIT_FULLY_WRITING | BP},
        {"adaptive_maxpool2d_bp", OP_TRAIT_FULLY_WRITING | BP},
        {"maxpool2d_bp",        OP_TRAIT_FULLY_WRITING | BP},
        {"maxpool3dnew_bp",     OP_TRAIT_FULLY_WRITING | BP},
        {"pnormpool2d_bp",      OP_TRAIT_FULLY_WRITING | BP},
        {"sconv2d_bp",          OP_TRAIT_FULLY_WRITING | BP},
        {"pointwise_conv2d",    OP_TRAIT_FULLY_WRITING},

        // ── Convolution additional ─────────────────────────────────────────
        {"conv1d",              OP_TRAIT_FULLY_WRITING},
        {"conv1d_bp",           OP_TRAIT_FULLY_WRITING | BP},
        {"conv3dnew",           OP_TRAIT_FULLY_WRITING},
        {"conv3dnew_bp",        OP_TRAIT_FULLY_WRITING | BP},
        {"deconv3d",            OP_TRAIT_FULLY_WRITING},
        {"deconv3d_bp",         OP_TRAIT_FULLY_WRITING | BP},
        {"deformable_conv2d",   OP_TRAIT_FULLY_WRITING},
        {"depthwise_conv2d",    OP_TRAIT_FULLY_WRITING},
        {"depthwise_conv2d_bp", OP_TRAIT_FULLY_WRITING | BP},
        {"dilation2d",          OP_TRAIT_FULLY_WRITING},

        // ── Upsampling ops ────────────────────────────────────────────────
        {"upsampling",          OP_TRAIT_FULLY_WRITING},
        {"upsampling2d",        OP_TRAIT_FULLY_WRITING},
        {"upsampling3d",        OP_TRAIT_FULLY_WRITING},
        {"upsampling2d_bp",     OP_TRAIT_FULLY_WRITING | BP},
        {"upsampling3d_bp",     OP_TRAIT_FULLY_WRITING | BP},
        {"upsampling_bp",       OP_TRAIT_FULLY_WRITING | BP},

        // ── RNN / LSTM / GRU ──────────────────────────────────────────────
        {"gru",                 OP_TRAIT_FULLY_WRITING},
        {"gru_bp",              OP_TRAIT_FULLY_WRITING | BP},
        {"grucell",             OP_TRAIT_FULLY_WRITING},
        {"grucell_bp",          OP_TRAIT_FULLY_WRITING | BP},
        {"lstm",                OP_TRAIT_FULLY_WRITING},
        {"lstmblock",           OP_TRAIT_FULLY_WRITING},
        {"lstmblockcell",       OP_TRAIT_FULLY_WRITING},
        {"lstmcell",            OP_TRAIT_FULLY_WRITING},
        {"lstmlayer",           OP_TRAIT_FULLY_WRITING},
        {"lstmlayer_bp",        OP_TRAIT_FULLY_WRITING | BP},
        {"lstmlayercell",       OP_TRAIT_FULLY_WRITING},
        {"lstmlayercellbp",     OP_TRAIT_FULLY_WRITING | BP},
        {"sru",                 OP_TRAIT_FULLY_WRITING},
        {"sru_bi",              OP_TRAIT_FULLY_WRITING},
        {"sru_bi_bp",           OP_TRAIT_FULLY_WRITING | BP},
        {"sru_bp",              OP_TRAIT_FULLY_WRITING | BP},
        {"srucell",             OP_TRAIT_FULLY_WRITING},
        {"static_bidirectional_rnn", OP_TRAIT_FULLY_WRITING},
        {"static_rnn",          OP_TRAIT_FULLY_WRITING},
        {"dynamic_bidirectional_rnn", OP_TRAIT_FULLY_WRITING},
        {"dynamic_rnn",         OP_TRAIT_FULLY_WRITING},

        // ── Image processing ops ───────────────────────────────────────────
        {"adjust_contrast",         UNARY_EW},
        {"adjust_contrast_v2",      UNARY_EW},
        {"adjust_hue",              UNARY_EW},
        {"adjust_saturation",       UNARY_EW},
        {"affine_grid",             DATA_MOVE_VALDEP},
        {"crop_and_resize",         OP_TRAIT_FULLY_WRITING},
        {"depth_to_space",          DATA_MOVE},
        {"draw_bounding_boxes",     OP_TRAIT_FULLY_WRITING},
        {"extract_image_patches",   OP_TRAIT_FULLY_WRITING},
        {"grid_sample",             OP_TRAIT_FULLY_WRITING},
        {"hsv_to_rgb",              UNARY_EW},
        {"rgb_to_grs",              UNARY_EW},
        {"rgb_to_hsv",              UNARY_EW},
        {"rgb_to_yiq",              UNARY_EW},
        {"rgb_to_yuv",              UNARY_EW},
        {"yiq_to_rgb",              UNARY_EW},
        {"yuv_to_rgb",              UNARY_EW},
        {"image_resize",            OP_TRAIT_FULLY_WRITING},
        {"resize_area",             OP_TRAIT_FULLY_WRITING},
        {"resize_bicubic",          OP_TRAIT_FULLY_WRITING},
        {"resize_bilinear",         OP_TRAIT_FULLY_WRITING},
        {"resize_images",           OP_TRAIT_FULLY_WRITING},
        {"resize_nearest_neighbor", OP_TRAIT_FULLY_WRITING},
        {"space_to_depth",          DATA_MOVE},
        {"mirror_pad",              DATA_MOVE_VALDEP},
        {"pad_input",               DATA_MOVE_VALDEP},
        {"pad_input_bp",            DATA_MOVE_VALDEP | BP},

        // ── Updater ops (optimizer parameter updates) ──────────────────────
        // Multi-input/multi-output updaters must NOT use BINARY_EW (which implies
        // exactly 2 inputs, 1 output with broadcast semantics). Using BINARY_EW
        // causes the native plan to wire wrong slot counts, corrupting weight updates.
        // sgd_updater: 1-in-1-out (grad → scaled_grad)
        {"sgd_updater",             UNARY_EW},
        // apply_sgd: 2-in-1-out (val, update → val-lr*update) — true binary elementwise
        {"apply_sgd",               BINARY_EW},
        // applygradientdescent: 2-in-1-out — true binary elementwise
        {"applygradientdescent",    BINARY_EW},
        // nesterovs_updater: 2-in-2-out (grad, v → updated_grad, new_v)
        {"nesterovs_updater",       OP_TRAIT_FULLY_WRITING},
        // adam_updater: 3-in-3-out (grad, v, m → updated_grad, new_v, new_m)
        {"adam_updater",            OP_TRAIT_FULLY_WRITING},
        // adabelief_updater: 3-in-3-out
        {"adabelief_updater",       OP_TRAIT_FULLY_WRITING},
        // ada_delta_updater: 3-in-3-out
        {"ada_delta_updater",       OP_TRAIT_FULLY_WRITING},
        // ada_grad_updater: 2-in-2-out (grad, state → updated_grad, new_state)
        {"ada_grad_updater",        OP_TRAIT_FULLY_WRITING},
        // ada_max_updater: 3-in-3-out
        {"ada_max_updater",         OP_TRAIT_FULLY_WRITING},
        // ams_grad_updater: 3-in-4-out
        {"ams_grad_updater",        OP_TRAIT_FULLY_WRITING},
        // nadam_updater: 3-in-3-out
        {"nadam_updater",           OP_TRAIT_FULLY_WRITING},
        // rms_prop_updater: 3-in-3-out (grad, v, m → updated_grad, new_v, new_m)
        {"rms_prop_updater",        OP_TRAIT_FULLY_WRITING},

        // ── Random generator ops ───────────────────────────────────────────
        {"random_bernoulli",    CONST_GEN_VALDEP | DATADEP},
        {"random_crop",         DATA_MOVE | DATADEP},
        {"random_exponential",  CONST_GEN_VALDEP | DATADEP},
        {"random_gamma",        CONST_GEN_VALDEP | DATADEP},
        {"random_multinomial",  CONST_GEN_VALDEP | DATADEP},
        {"random_normal",       CONST_GEN_VALDEP | DATADEP},
        {"randomnormal",        CONST_GEN_VALDEP | DATADEP},
        {"random_poisson",      CONST_GEN_VALDEP | DATADEP},
        {"random_shuffle",      DATA_MOVE | DATADEP},

        // ── Scatter ops ────────────────────────────────────────────────────
        {"scatter_add",         SCATTER_PARTIAL},
        {"scatter_div",         SCATTER_PARTIAL},
        {"scatter_max",         SCATTER_PARTIAL},
        {"scatter_min",         SCATTER_PARTIAL},
        {"scatter_mul",         SCATTER_PARTIAL},
        {"scatter_sub",         SCATTER_PARTIAL},
        {"scatter_upd",         SCATTER_PARTIAL},
        {"scatter_update",      SCATTER_PARTIAL},
        {"scatterupdate",       SCATTER_PARTIAL},
        {"scatter_nd_add",      SCATTER_PARTIAL | OP_TRAIT_SCATTER_ND},
        {"scatter_nd_sub",      SCATTER_PARTIAL | OP_TRAIT_SCATTER_ND},

        // ── Segment ops ────────────────────────────────────────────────────
        {"segment_gemm",        MATMUL},
        {"segment_max",         REDUCE | DATADEP},
        {"segment_max_bp",      REDUCE | BP | DATADEP},
        {"segment_mean",        REDUCE | DATADEP},
        {"segment_mean_bp",     REDUCE | BP | DATADEP},
        {"segment_min",         REDUCE | DATADEP},
        {"segment_min_bp",      REDUCE | BP | DATADEP},
        {"segment_prod",        REDUCE | DATADEP},
        {"segment_prod_bp",     REDUCE | BP | DATADEP},
        {"segment_sum",         REDUCE | DATADEP},
        {"segment_sum_bp",      REDUCE | BP | DATADEP},
        {"unsorted_segment_max_bp",     REDUCE | BP | DATADEP},
        {"unsorted_segment_mean_bp",    REDUCE | BP | DATADEP},
        {"unsorted_segment_min_bp",     REDUCE | BP | DATADEP},
        {"unsorted_segment_prod_bp",    REDUCE | BP | DATADEP},
        {"unsorted_segment_sqrt_n_bp",  REDUCE | BP | DATADEP},
        {"unsorted_segment_sum_bp",     REDUCE | BP | DATADEP},

        // ── Data movement ops ──────────────────────────────────────────────
        {"reverse",             DATA_MOVE},
        {"reverse_bp",          DATA_MOVE | BP},
        {"reverse_v2",          DATA_MOVE},
        {"reverse_sequence",    DATA_MOVE},
        {"roll",                DATA_MOVE},
        {"transpose",           DATA_MOVE},
        {"concat_bp",           CONCAT | BP},
        {"concat_v2",           CONCAT | DATADEP},
        {"concatv2",            CONCAT | DATADEP},
        {"unstack",             SPLIT},
        {"unpack",              SPLIT},
        {"pack",                STACK},
        {"parallel_stack",      STACK},
        {"dynamic_stitch",      DATA_MOVE | DATADEP},
        {"dynamic_partition",   SPLIT | DATADEP},
        {"dynamic_partition_bp", SPLIT | BP | DATADEP},
        {"diag",                DATA_MOVE},
        {"diag_part",           DATA_MOVE},
        {"matrix_diag",         DATA_MOVE},
        {"matrix_diag_part",    DATA_MOVE},
        {"matrix_set_diag",     DATA_MOVE},
        {"matrix_band_part",    DATA_MOVE},
        {"band_part",           DATA_MOVE},
        {"tile_bp",             TILE | BP},
        {"tile_to_shape",       DATA_MOVE_VALDEP},
        {"tile_to_shape_bp",    DATA_MOVE_VALDEP | BP},
        {"slice_bp",            SLICE | BP},
        {"linear_copy",         DATA_MOVE},
        {"fill_as",             CONST_GEN_VALDEP},
        {"fill_like",           CONST_GEN_VALDEP},
        {"filllike",            CONST_GEN_VALDEP},
        {"reshape_as",          VIEW | DATADEP},
        {"reshapeas",           VIEW | DATADEP},
        {"create_view",         VIEW | DATADEP},
        {"eye",                 CONST_GEN_VALDEP | DATADEP},
        {"sequence_mask",       CONST_GEN_VALDEP | DATADEP},

        // ── Shape / size ops ───────────────────────────────────────────────
        {"shape",               SHAPE_ONLY | CONST_GEN},
        {"shape_n",             SHAPE_ONLY | CONST_GEN},
        {"shapes_of",           SHAPE_ONLY | CONST_GEN},
        {"size",                SHAPE_ONLY | CONST_GEN},
        {"set_shape",           SHAPE_ONLY | CONST_GEN},
        {"broadcast_dynamic_shape", SHAPE_ONLY | CONST_GEN | DATADEP},
        {"broadcastgradientargs",   SHAPE_ONLY | CONST_GEN | DATADEP},
        {"invert_permutation",  DATA_MOVE | DATADEP},
        {"order",               DATA_MOVE},
        {"ismax",               UNARY_EW | DATADEP},
        {"nth_element",         REDUCE | DATADEP},
        {"in_top_k",            BINARY_CMP | DATADEP},
        {"listdiff",            DATA_MOVE | DATADEP},
        {"percentile",          REDUCE | DATADEP},
        {"unique_with_counts",  DATADEP},
        {"hashcode",            REDUCE | DATADEP},
        {"confusion_matrix",    CONST_GEN_VALDEP | DATADEP},
        {"histogram",           REDUCE | DATADEP},
        {"histogram_fixed_width", REDUCE | DATADEP},
        {"bits_hamming_distance", REDUCE},
        {"compare_and_bitpack", BINARY_EW},
        {"tri",                 CONST_GEN_VALDEP | DATADEP},

        // ── Linear algebra ────────────────────────────────────────────────
        {"cholesky",            OP_TRAIT_FULLY_WRITING},
        {"eig",                 OP_TRAIT_FULLY_WRITING | DATADEP},
        {"lu",                  OP_TRAIT_FULLY_WRITING},
        {"lstsq",               OP_TRAIT_FULLY_WRITING},
        {"matrix_determinant",  REDUCE},
        {"matrix_inverse",      OP_TRAIT_FULLY_WRITING},
        {"matrixsolvels",       OP_TRAIT_FULLY_WRITING},
        {"solve",               OP_TRAIT_FULLY_WRITING},
        {"solve_ls",            OP_TRAIT_FULLY_WRITING},
        {"sqrtm",               OP_TRAIT_FULLY_WRITING},
        {"svd",                 OP_TRAIT_FULLY_WRITING | DATADEP},
        {"logdet",              REDUCE},
        {"log_matrix_determinant", REDUCE},
        {"triangular_solve",    OP_TRAIT_FULLY_WRITING},
        {"trace",               REDUCE},
        {"einsum",              OP_TRAIT_FULLY_WRITING | DATADEP},
        {"meshgrid",            CONST_GEN_VALDEP},

        // ── CTC / sequence ops ────────────────────────────────────────────
        {"ctc_beam",            OP_TRAIT_FULLY_WRITING | DATADEP},
        {"ctc_greedy_decoder",  OP_TRAIT_FULLY_WRITING | DATADEP},
        {"cumprod",             REDUCE},
        {"cumprod_bp",          REDUCE | BP},
        {"cumsum",              REDUCE},
        {"cumsum_bp",           REDUCE | BP},

        // ── Misc math ─────────────────────────────────────────────────────
        {"fake_quant_with_min_max_args",              UNARY_EW},
        {"fake_quant_with_min_max_args_per_channel",  UNARY_EW},
        {"fake_quant_with_min_max_vars",              UNARY_EW},
        {"fake_quant_with_min_max_vars_per_channel",  UNARY_EW},
        {"fp8_dequantize",      UNARY_EW},
        {"fp8_quantize",        UNARY_EW},
        {"loftq_init",          OP_TRAIT_FULLY_WRITING},
        {"dft",                 OP_TRAIT_FULLY_WRITING},
        {"stft",                OP_TRAIT_FULLY_WRITING},
        {"fused_attention_projection", ATTN},

        // ── Audio / signal processing ──────────────────────────────────────
        {"audio_normalize",     OP_TRAIT_FULLY_WRITING},
        {"audio_resample",      OP_TRAIT_FULLY_WRITING},
        {"a_weighting",         OP_TRAIT_FULLY_WRITING},
        {"blackman_window",     CONST_GEN_VALDEP},
        {"chroma_features",     OP_TRAIT_FULLY_WRITING},
        {"griffin_lim",         OP_TRAIT_FULLY_WRITING},
        {"hamming_window",      CONST_GEN_VALDEP},
        {"hann_window",         CONST_GEN_VALDEP},
        {"mel_filterbank",      OP_TRAIT_FULLY_WRITING},
        {"mel_spectrogram",     OP_TRAIT_FULLY_WRITING},
        {"mfcc",                OP_TRAIT_FULLY_WRITING},
        {"pitch_detection",     OP_TRAIT_FULLY_WRITING | DATADEP},
        {"pre_emphasis",        UNARY_EW},
        {"spectral_centroid",   REDUCE},
        {"spectral_rolloff",    REDUCE},
        {"whisper_mel_spectrogram", OP_TRAIT_FULLY_WRITING},
        {"zero_crossing_rate",  REDUCE},

        // ── VLM / vision ops ──────────────────────────────────────────────
        {"autoregressive_decode",     OP_TRAIT_FULLY_WRITING | DATADEP},
        {"vision_encode_patches",     OP_TRAIT_FULLY_WRITING},
        {"vlm_2d_position_encode",    OP_TRAIT_FULLY_WRITING},
        {"vlm_image_embed",           OP_TRAIT_FULLY_WRITING},
        {"vlm_image_preprocess",      OP_TRAIT_FULLY_WRITING},
        {"vlm_multimodal_fusion",     OP_TRAIT_FULLY_WRITING},
        {"vlm_patch_embed",           OP_TRAIT_FULLY_WRITING},
        {"vlm_vision_encode",         OP_TRAIT_FULLY_WRITING},
        {"vlm_vision_projection",     OP_TRAIT_FULLY_WRITING},
        {"kv_scatter",                SCATTER_PARTIAL},
        {"sampling_penalties",        BINARY_EW},
        {"top_k_renorm",              OP_TRAIT_FULLY_WRITING | DATADEP},
        {"top_p_renorm",              OP_TRAIT_FULLY_WRITING | DATADEP},
        {"lightning_attention",       ATTN},
        {"linear_attention_decode",   ATTN},
        {"mixture_of_experts",        OP_TRAIT_FULLY_WRITING},
        {"moe_shared_experts",        OP_TRAIT_FULLY_WRITING},
        {"cascade_attention",         ATTN},
        {"checkpoint_offload_d2h",    DATA_MOVE},
        {"checkpoint_prefetch_h2d",   DATA_MOVE},

        // ── Sparse / compat ops ────────────────────────────────────────────
        {"compat_sparse_to_dense",    DATA_MOVE | DATADEP},
        {"compat_string_split",       DATADEP},
        {"split_string",              DATADEP},
        {"embedding_lookup",          GATHER},
        {"cbow",                      OP_TRAIT_FULLY_WRITING},
        {"cbow_inference",            OP_TRAIT_FULLY_WRITING},
        {"skipgram",                  OP_TRAIT_FULLY_WRITING},
        {"skipgram_inference",        OP_TRAIT_FULLY_WRITING},
        {"knn_mindistance",           REDUCE | DATADEP},

        // ── Miscellaneous utility ops ─────────────────────────────────────
        {"assert",              OP_TRAIT_FULLY_WRITING | DATADEP},
        {"check_numerics",      OP_TRAIT_FULLY_WRITING | DATADEP},
        {"choose",              DATA_MOVE | DATADEP},
        {"expose",              DATA_MOVE},
        {"get_seed",            CONST_GEN_VALDEP | DATADEP},
        {"set_seed",            OP_TRAIT_FULLY_WRITING},
        {"is_non_decreasing",   BINARY_CMP | DATADEP},
        {"is_numeric_tensor",   BINARY_CMP},
        {"is_strictly_increasing", BINARY_CMP | DATADEP},
        {"noop",                OP_TRAIT_FULLY_WRITING},
        {"print_affinity",      OP_TRAIT_FULLY_WRITING},
        {"print_variable",      OP_TRAIT_FULLY_WRITING},
        {"stop_gradient",       DATA_MOVE},
        {"tear",                DATA_MOVE},
        {"unpad_input",         DATA_MOVE | DATADEP},

        // ── Control flow ops ──────────────────────────────────────────────
        {"cond",                DATADEP},
        {"enter",               DATA_MOVE},
        {"if",                  DATADEP},
        {"invoke",              DATADEP},
        {"opscope",             DATADEP},
        {"return",              DATA_MOVE},
        {"scope",               DATADEP},
        {"switch",              DATADEP},
        {"while",               DATADEP},
        {"identity_n",          DATA_MOVE},

        // ── TensorArray ops ───────────────────────────────────────────────
        {"tensorarrayconcatv3",     DATADEP},
        {"tensorarraycreatev3",     DATADEP},
        {"tensorarraygatherv3",     DATADEP},
        {"tensorarrayidentityv3",   DATADEP},
        {"tensorarrayreadv3",       DATADEP},
        {"tensorarrayscatterv3",    DATADEP},
        {"tensorarraysizev3",       DATADEP},
        {"tensorarraysplitv3",      DATADEP},
        {"tensorarrayv3",           DATADEP},
        {"tensorarraywritev3",      DATADEP},

        // ── List ops ──────────────────────────────────────────────────────
        {"clone_list",          DATADEP},
        {"create_list",         DATADEP},
        {"delete_list",         DATADEP},
        {"gather_list",         GATHER | DATADEP},
        {"pick_list",           DATADEP},
        {"read_list",           DATADEP},
        {"scatter_list",        DATADEP},
        {"size_list",           SHAPE_ONLY | CONST_GEN | DATADEP},
        {"split_list",          DATADEP},
        {"stack_list",          STACK | DATADEP},
        {"unstack_list",        SPLIT | DATADEP},
        {"write_list",          DATADEP},
        {"cell_contains",       DATADEP},

        // ── Test / debug ops ──────────────────────────────────────────────
        {"testcustom",          OP_TRAIT_FULLY_WRITING},
        {"testop2i2o",          OP_TRAIT_FULLY_WRITING},
        {"test_output_reshape", OP_TRAIT_FULLY_WRITING},
        {"test_scalar",         OP_TRAIT_FULLY_WRITING},

        // ── Misc structural / quantization ────────────────────────────────
        {"assign_bp",           IDENT | BP},
        {"barnes_edge_forces",  OP_TRAIT_FULLY_WRITING},
        {"barnes_gains",        OP_TRAIT_FULLY_WRITING},
        {"barnes_symmetrized",  OP_TRAIT_FULLY_WRITING},
        {"argamax",             REDUCE | DATADEP},
        {"argamin",             REDUCE | DATADEP},

        // ── Aliases (registered with different casing/spacing) ─────────────
        // These are needed because normalizeOpName lowercases but doesn't
        // insert underscores, so e.g. "DiagPart" → "diagpart" ≠ "diag_part".
        {"conditional",         DATADEP},                           // alias: Conditional
        {"diagpart",            DATA_MOVE},                         // alias: DiagPart
        {"invertpermutation",   DATA_MOVE | DATADEP},               // alias: InvertPermutation
        {"matrixdiag",          DATA_MOVE},                         // alias: MatrixDiag
        {"matrixsetdiag",       DATA_MOVE},                         // alias: MatrixSetDiag
        {"maxpool_bp",          OP_TRAIT_FULLY_WRITING | BP},       // alias: MaxPool_bp
        {"parallelconcat",      CONCAT},                            // alias: ParallelConcat
        {"qr",                  OP_TRAIT_FULLY_WRITING},            // QR decomposition
        {"scatteradd",          SCATTER_PARTIAL},                   // alias: ScatterAdd
        {"scatterdiv",          SCATTER_PARTIAL},                   // alias: ScatterDiv
        {"scattermax",          SCATTER_PARTIAL},                   // alias: ScatterMax
        {"scattermin",          SCATTER_PARTIAL},                   // alias: ScatterMin
        {"scattermul",          SCATTER_PARTIAL},                   // alias: ScatterMul
        {"scattersub",          SCATTER_PARTIAL},                   // alias: ScatterSub
        {"softplusgrad",        UNARY_ACT | BP},                    // alias: SoftplusGrad
        {"softsigngrad",        UNARY_ACT | BP},                    // alias: SoftsignGrad
        {"stopgradient",        DATA_MOVE},                         // alias: StopGradient
        {"stridedslice",        VIEW | OP_TRAIT_SLICE},             // alias: no-underscore form
        {"strided_slice_bp",    VIEW | OP_TRAIT_SLICE | BP},        // strided_slice backward
        {"tanhgrad",            UNARY_ACT | BP},                    // alias: TanhGrad
        {"triu",                DATA_MOVE},                         // upper triangular
        {"triu_bp",             DATA_MOVE | BP},                    // upper triangular backward
        {"where_np",            DATADEP},                           // numpy-style where
    };
    return TABLE;
}

// ─── Structural iArg table ─────────────────────────────────────────────────
static const std::unordered_map<std::string, int>& getStructuralIArgTable() {
    static const std::unordered_map<std::string, int> TABLE = {
        {"strided_slice", 5},   // 5 mask bits (begin/end/shrink/new_axis/ellipsis)
        {"concat", 1},          // axis
        {"split", 1},           // num_splits
        {"split_v", 1},         // axis
        {"one_hot", 2},         // axis, depth
        {"top_k", 1},           // k
    };
    return TABLE;
}

static std::string normalizeOpName(const std::string& opName) {
    std::string normalized = opName;
    std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return normalized;
}

// ─── Init function ─────────────────────────────────────────────────────────

static std::atomic<bool> traitsInitialized{false};

void initOpTraits() {
    // Idempotent: safe to call multiple times
    if (traitsInitialized.load(std::memory_order_acquire)) return;

    auto& table = getTraitTable();
    auto& registrator = OpRegistrator::getInstance();

    // Iterate all registered op names and apply traits by normalized (lowercase) name lookup.
    // This is necessary because ops can be registered with mixed case (e.g., "Where" capital W)
    // while the trait table uses lowercase keys (e.g., "where").  A direct lookup by the
    // table key would miss ops whose registered name has a different case.
    for (const auto& opName : registrator.getAllRegisteredOpNames()) {
        std::string normalized = normalizeOpName(opName);
        auto it = table.find(normalized);
        if (it != table.end()) {
            auto* op = registrator.getOperation(opName.c_str());
            if (op != nullptr) {
                // addTraits preserves any traits already set by the class hierarchy
                op->getOpDescriptor()->addTraits(it->second);
            }
        }
    }

    traitsInitialized.store(true, std::memory_order_release);
}

uint32_t getOpTraitsByName(const std::string& opName) {
    auto& table = getTraitTable();
    auto it = table.find(normalizeOpName(opName));
    return (it != table.end()) ? it->second : 0;
}

int getStructuralIArgCount(const std::string& opName) {
    auto& table = getStructuralIArgTable();
    auto it = table.find(normalizeOpName(opName));
    return (it != table.end()) ? it->second : -1;
}

}  // namespace ops
}  // namespace sd

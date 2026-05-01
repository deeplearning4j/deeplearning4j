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
        {"gather",         GATHER | DATADEP},   // shape fn reads INPUT_VARIABLE(2)->e<>() for axis when 3-input form
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

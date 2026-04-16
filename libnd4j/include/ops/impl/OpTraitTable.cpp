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
static constexpr uint32_t SCATTER_ND_UPDATE = SCATTER_PARTIAL | OP_TRAIT_SCATTER_ND_UPDATE;

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
        {"fused_gemm_swiglu",      MATMUL},

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
        {"argmax",         REDUCE},
        {"argmin",         REDUCE},

        // ── Normalization ops ──────────────────────────────────────────────
        {"softmax",        NORM},
        {"log_softmax",    NORM},
        {"layer_norm",     NORM},
        {"fused_layer_norm", NORM},
        {"batch_norm",     NORM},
        {"batchnorm",      NORM},
        {"rms_norm",       NORM},
        {"rms_norm_linear", NORM},
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
        {"reshape",        VIEW},
        {"reshape_no_copy", VIEW},
        {"strided_slice",  VIEW | OP_TRAIT_SLICE},
        {"expand_dims",    VIEW_SHAPE_DEP},
        {"squeeze",        VIEW_SHAPE_DEP},
        {"flatten",        VIEW_SHAPE_DEP},
        {"flatten_2d",     VIEW_SHAPE_DEP},
        {"permute",        VIEW_SHAPE_DEP},

        // ── Shape-determined data movement (shape fn reads input shapes + iArgs only) ─
        // These MUST NOT carry VALUE_DEPENDENT_SHAPE — the SHAPES_FROZEN check relies on
        // the flag being accurate. A false positive causes the value-dep shape-match
        // branch to report "value-dependent output shape changed" when in fact the input
        // shape changed (e.g., gather indices went [1,512] → [18,512]).
        {"gather",         GATHER},
        {"gather_nd",      GATHER_ND},
        {"concat",         CONCAT},
        {"stack",          STACK},
        {"split",          SPLIT},
        {"split_v",        SPLIT_V},
        {"repeat",         DATA_MOVE},

        // ── Value-dependent data movement (shape fn reads tensor DATA) ─────
        {"slice",          SLICE},           // dual-mode: iArg or tensor begin/size; conservative VALDEP
        {"tile",           TILE},            // dual-mode: iArg or tensor multiples; conservative VALDEP
        {"pad",            DATA_MOVE_VALDEP},
        {"fill",           DATA_MOVE_VALDEP},
        {"broadcast_to",   DATA_MOVE_VALDEP},
        {"scatter_nd",     SCATTER_ND},
        {"scatter_nd_update", SCATTER_ND_UPDATE},
        {"range",          CONST_GEN_VALDEP},
        {"linspace",       CONST_GEN_VALDEP},
        // create/ConstantOfShape materializes a real output buffer. Some instances feed
        // shape-control ladders, but that must be inferred from runtime tensor semantics,
        // not baked into the op as globally shape-only.
        {"create",         CONST_GEN_VALDEP},

        // ── Data-dependent ops (variable-length output) ────────────────────
        {"unique",                 DATADEP},
        {"non_max_suppression",    DATADEP},
        {"non_max_suppression_v3", DATADEP},

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
        {"dot_product_attention_bp",            ATTN},
        {"dot_product_attention_v2_bp",         ATTN},
        {"multi_head_dot_product_attention_bp", ATTN},
        {"flash_attention_bp",                  ATTN},
        {"grouped_query_attention",             ATTN},
        {"grouped_query_attention_bp",          ATTN},
        {"sliding_window_attention",            ATTN},
        {"shared_kv_attention",                 ATTN},
        {"windowed_attention",                  ATTN},
        {"paged_attention_forward",             ATTN},
        {"turbo_quant_attention",               ATTN},
        {"two_way_cross_attention",             ATTN},
        {"two_way_cross_attention_bp",          ATTN},
        {"vlm_cross_attention",                 ATTN},
        {"apply_alibi",                         ATTN},
        {"relative_position_bias",              ATTN},

        // ── KV cache management ────────────────────────────────────────────
        {"kv_cache_update",     DATA_MOVE_VALDEP},
        {"kv_cache_quantize",   CONST_GEN_VALDEP},
        {"kv_cache_dequantize", UNARY_EW},
        {"paged_kv_append",     DATA_MOVE_VALDEP},

        // ── Rotary / positional embedding ──────────────────────────────────
        {"rope",         NORM},
        {"rope_bp",      NORM},
        {"fused_rope_bp", NORM},
        {"dual_rope",    NORM},

        // ── Normalization backprop / fused variants ────────────────────────
        {"rms_norm_bp",             NORM},
        {"rms_norm_linear_bp",      NORM},
        {"fused_layer_norm_bp",     NORM},
        {"fused_rms_norm_swiglu",   NORM},
        {"fused_rms_norm_swiglu_bp", NORM},

        // ── Fused GEMM / SwiGLU ────────────────────────────────────────────
        {"fused_gemm_swiglu_bp", MATMUL},

        // ── Activation backprop + novel activations ────────────────────────
        {"silu_bp",         UNARY_ACT},
        {"fused_gelu_bp",   UNARY_ACT},
        {"squared_relu",    UNARY_ACT},
        {"squared_relu_bp", UNARY_ACT},
        {"gated_delta_rule", UNARY_ACT},

        // ── Mamba / selective scan / SSM / causal conv ─────────────────────
        {"gated_delta_net_block", REDUCE},
        {"selective_scan",        REDUCE},
        {"mamba2_ssm",            REDUCE},
        {"causal_conv1d",         UNARY_EW},

        // ── Fused training kernels ─────────────────────────────────────────
        {"fused_bias_dropout_residual", UNARY_EW},
        {"fused_elementwise_chain",     UNARY_EW},
        {"swish_mul_bp",                BINARY_EW},
        {"center_and_sharpen",          UNARY_EW},
        {"center_and_sharpen_bp",       UNARY_EW},
        {"ema_update",                  DATA_MOVE},
        {"ema_update_bp",               DATA_MOVE},

        // ── Quantization / adapter matmuls ─────────────────────────────────
        {"quantized_matmul", MATMUL},
        {"dora_matmul",      MATMUL},
        {"dora_matmul_bp",   MATMUL},
        {"lora_matmul",      MATMUL},
        {"lora_matmul_bp",   MATMUL},
        {"loha_matmul",      MATMUL},
        {"loha_matmul_bp",   MATMUL},
        {"lokr_matmul",      MATMUL},
        {"lokr_matmul_bp",   MATMUL},

        // ── GGML / per-layer embedding / misc ──────────────────────────────
        {"ggml_dequantize",     UNARY_EW},
        {"per_layer_embedding", DATA_MOVE_VALDEP},
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

    for (auto& entry : table) {
        auto* op = registrator.getOperation(entry.first.c_str());
        if (op != nullptr) {
            // addTraits preserves any traits already set by the class hierarchy
            op->getOpDescriptor()->addTraits(entry.second);
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

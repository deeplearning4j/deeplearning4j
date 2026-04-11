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
static constexpr uint32_t GATHER = DATA_MOVE_VALDEP | OP_TRAIT_GATHER;
static constexpr uint32_t GATHER_ND = DATA_MOVE_VALDEP | OP_TRAIT_GATHER_ND;
static constexpr uint32_t CONCAT = DATA_MOVE | OP_TRAIT_CONCAT;
static constexpr uint32_t SPLIT = DATA_MOVE | OP_TRAIT_SPLIT;
static constexpr uint32_t SPLIT_V = DATA_MOVE | OP_TRAIT_SPLIT_V;
static constexpr uint32_t STACK = DATA_MOVE | OP_TRAIT_STACK;
static constexpr uint32_t SLICE = DATA_MOVE_VALDEP | OP_TRAIT_SLICE;
static constexpr uint32_t TILE = DATA_MOVE_VALDEP | OP_TRAIT_TILE;
static constexpr uint32_t SCATTER_ND = DATA_MOVE | OP_TRAIT_SCATTER_ND;
static constexpr uint32_t SCATTER_ND_UPDATE = DATA_MOVE | OP_TRAIT_SCATTER_ND_UPDATE;

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

        // ── View-producing ops (output shape depends on input VALUES) ──────
        {"reshape",        VIEW},
        {"reshape_no_copy", VIEW},
        {"expand_dims",    VIEW},
        {"squeeze",        VIEW},
        {"flatten",        VIEW},
        {"flatten_2d",     VIEW},
        {"permute",        OP_TRAIT_VIEW_PRODUCING},  // permute shape depends on iArgs, not input values
        {"strided_slice",  VIEW | OP_TRAIT_SLICE},

        // ── Value-dependent shape (non-view) ───────────────────────────────
        {"slice",          SLICE},
        {"gather",         GATHER},
        {"gather_nd",      GATHER_ND},
        {"concat",         CONCAT},
        {"stack",          STACK},
        {"split",          SPLIT},
        {"split_v",        SPLIT_V},
        {"tile",           TILE},
        {"repeat",         DATA_MOVE_VALDEP},
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

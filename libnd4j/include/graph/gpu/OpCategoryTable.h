/* ******************************************************************************
 *
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

#ifndef LIBND4J_OP_CATEGORY_TABLE_H
#define LIBND4J_OP_CATEGORY_TABLE_H

#ifdef SD_CUDA

#include <string>
#include <unordered_map>
#include <system/common.h>

namespace sd {
namespace graph {

/**
 * Classification of how a libnd4j op maps to GPU IR (shared by Triton and NVRTC paths).
 */
enum class TritonOpCategory {
  BINARY_ELEMENTWISE,     // add, sub, mul, div, min, max
  UNARY_ELEMENTWISE,      // relu, sigmoid, tanh, exp, log, sqrt, etc.
  COMPARISON,             // greater, less, equals, etc. -> per-element bool
  LOGICAL,                // boolean_and, boolean_or, etc. -> per-element bool
  TERNARY,                // where, select -> 3-input per-element
  IDENTITY,               // identity, assign -> SSA value forwarding
  MATMUL,                 // matmul, batch_matmul -> tt.dot (2D tiled kernel)
  REDUCTION,              // reduce_sum, reduce_max, etc. -> tree reduction
  NORMALIZATION,          // softmax, layer_norm -> multi-op fused pattern
  CAST,                   // type cast -> arith cast ops
  FUSED_ATTENTION,        // onnx_multi_head_attention -> Flash Attention kernel
  SHAPE_MANIPULATION,     // reshape, permute, expand_dims, squeeze, flatten -> view/stride ops
  DATA_MOVEMENT,          // gather, concat, split, stack, slice, tile, scatter -> data copy
  CONSTANT_GENERATION,    // shape_of, create, set_scalar, ones_as, zeros_like, range -> constant fill
  CONVOLUTION,            // conv2d, conv3d -> spatial convolution kernel
  UNSUPPORTED             // cannot be mapped
};

/**
 * Lightweight op-name -> category mapping.
 * Does NOT depend on MLIR/Triton -- usable from both NVRTC and Triton paths.
 */
inline const std::unordered_map<std::string, TritonOpCategory>& getOpCategoryTable() {
  static const std::unordered_map<std::string, TritonOpCategory> table = {
    // ── Binary element-wise ──
    {"add",               TritonOpCategory::BINARY_ELEMENTWISE},
    {"Add",               TritonOpCategory::BINARY_ELEMENTWISE},
    {"subtract",          TritonOpCategory::BINARY_ELEMENTWISE},
    {"Sub",               TritonOpCategory::BINARY_ELEMENTWISE},
    {"multiply",          TritonOpCategory::BINARY_ELEMENTWISE},
    {"Mul",               TritonOpCategory::BINARY_ELEMENTWISE},
    {"divide",            TritonOpCategory::BINARY_ELEMENTWISE},
    {"Div",               TritonOpCategory::BINARY_ELEMENTWISE},
    {"RealDiv",           TritonOpCategory::BINARY_ELEMENTWISE},
    {"minimum",           TritonOpCategory::BINARY_ELEMENTWISE},
    {"Min",               TritonOpCategory::BINARY_ELEMENTWISE},
    {"maximum",           TritonOpCategory::BINARY_ELEMENTWISE},
    {"Max",               TritonOpCategory::BINARY_ELEMENTWISE},
    {"mod",               TritonOpCategory::BINARY_ELEMENTWISE},
    {"Mod",               TritonOpCategory::BINARY_ELEMENTWISE},
    {"floormod",          TritonOpCategory::BINARY_ELEMENTWISE},
    {"FloorMod",          TritonOpCategory::BINARY_ELEMENTWISE},
    {"atan2",             TritonOpCategory::BINARY_ELEMENTWISE},
    {"Atan2",             TritonOpCategory::BINARY_ELEMENTWISE},
    {"floordiv",          TritonOpCategory::BINARY_ELEMENTWISE},
    {"FloorDiv",          TritonOpCategory::BINARY_ELEMENTWISE},
    {"reversedivide",     TritonOpCategory::BINARY_ELEMENTWISE},
    {"ReverseDivide",     TritonOpCategory::BINARY_ELEMENTWISE},
    {"reversesubtract",   TritonOpCategory::BINARY_ELEMENTWISE},
    {"ReverseSubtract",   TritonOpCategory::BINARY_ELEMENTWISE},
    {"squaredsubtract",   TritonOpCategory::BINARY_ELEMENTWISE},
    {"SquaredSubtract",   TritonOpCategory::BINARY_ELEMENTWISE},
    {"multiply_no_nan",   TritonOpCategory::BINARY_ELEMENTWISE},
    {"MultiplyNoNan",     TritonOpCategory::BINARY_ELEMENTWISE},
    {"min_pairwise",      TritonOpCategory::BINARY_ELEMENTWISE},
    {"MinPairwise",       TritonOpCategory::BINARY_ELEMENTWISE},
    {"max_pairwise",      TritonOpCategory::BINARY_ELEMENTWISE},
    {"MaxPairwise",       TritonOpCategory::BINARY_ELEMENTWISE},
    {"pow",               TritonOpCategory::BINARY_ELEMENTWISE},
    {"Pow",               TritonOpCategory::BINARY_ELEMENTWISE},
    // SwiGLU: swish_mul(x, y) = x * sigmoid(x) * y
    {"swish_mul",         TritonOpCategory::BINARY_ELEMENTWISE},
    {"SwishMul",          TritonOpCategory::BINARY_ELEMENTWISE},

    // ── Unary element-wise ──
    {"relu",              TritonOpCategory::UNARY_ELEMENTWISE},
    {"Relu",              TritonOpCategory::UNARY_ELEMENTWISE},
    {"sigmoid",           TritonOpCategory::UNARY_ELEMENTWISE},
    {"Sigmoid",           TritonOpCategory::UNARY_ELEMENTWISE},
    {"tanh",              TritonOpCategory::UNARY_ELEMENTWISE},
    {"Tanh",              TritonOpCategory::UNARY_ELEMENTWISE},
    {"gelu",              TritonOpCategory::UNARY_ELEMENTWISE},
    {"Gelu",              TritonOpCategory::UNARY_ELEMENTWISE},
    {"exp",               TritonOpCategory::UNARY_ELEMENTWISE},
    {"Exp",               TritonOpCategory::UNARY_ELEMENTWISE},
    {"log",               TritonOpCategory::UNARY_ELEMENTWISE},
    {"Log",               TritonOpCategory::UNARY_ELEMENTWISE},
    {"abs",               TritonOpCategory::UNARY_ELEMENTWISE},
    {"Abs",               TritonOpCategory::UNARY_ELEMENTWISE},
    {"sqrt",              TritonOpCategory::UNARY_ELEMENTWISE},
    {"Sqrt",              TritonOpCategory::UNARY_ELEMENTWISE},
    {"square",            TritonOpCategory::UNARY_ELEMENTWISE},
    {"Square",            TritonOpCategory::UNARY_ELEMENTWISE},
    {"neg",               TritonOpCategory::UNARY_ELEMENTWISE},
    {"Neg",               TritonOpCategory::UNARY_ELEMENTWISE},
    {"reciprocal",        TritonOpCategory::UNARY_ELEMENTWISE},
    {"Reciprocal",        TritonOpCategory::UNARY_ELEMENTWISE},
    {"rsqrt",             TritonOpCategory::UNARY_ELEMENTWISE},
    {"Rsqrt",             TritonOpCategory::UNARY_ELEMENTWISE},
    {"sign",              TritonOpCategory::UNARY_ELEMENTWISE},
    {"Sign",              TritonOpCategory::UNARY_ELEMENTWISE},
    {"erf",               TritonOpCategory::UNARY_ELEMENTWISE},
    {"Erf",               TritonOpCategory::UNARY_ELEMENTWISE},
    {"erfc",              TritonOpCategory::UNARY_ELEMENTWISE},
    {"Erfc",              TritonOpCategory::UNARY_ELEMENTWISE},
    {"log1p",             TritonOpCategory::UNARY_ELEMENTWISE},
    {"Log1p",             TritonOpCategory::UNARY_ELEMENTWISE},
    {"ceil",              TritonOpCategory::UNARY_ELEMENTWISE},
    {"Ceil",              TritonOpCategory::UNARY_ELEMENTWISE},
    {"floor",             TritonOpCategory::UNARY_ELEMENTWISE},
    {"Floor",             TritonOpCategory::UNARY_ELEMENTWISE},
    {"round",             TritonOpCategory::UNARY_ELEMENTWISE},
    {"Round",             TritonOpCategory::UNARY_ELEMENTWISE},
    {"sin",               TritonOpCategory::UNARY_ELEMENTWISE},
    {"Sin",               TritonOpCategory::UNARY_ELEMENTWISE},
    {"cos",               TritonOpCategory::UNARY_ELEMENTWISE},
    {"Cos",               TritonOpCategory::UNARY_ELEMENTWISE},
    {"leakyrelu",         TritonOpCategory::UNARY_ELEMENTWISE},
    {"LeakyRelu",         TritonOpCategory::UNARY_ELEMENTWISE},
    {"silu",              TritonOpCategory::UNARY_ELEMENTWISE},
    {"Silu",              TritonOpCategory::UNARY_ELEMENTWISE},
    {"swish",             TritonOpCategory::UNARY_ELEMENTWISE},
    {"Swish",             TritonOpCategory::UNARY_ELEMENTWISE},
    {"mish",              TritonOpCategory::UNARY_ELEMENTWISE},
    {"Mish",              TritonOpCategory::UNARY_ELEMENTWISE},
    {"elu",               TritonOpCategory::UNARY_ELEMENTWISE},
    {"Elu",               TritonOpCategory::UNARY_ELEMENTWISE},
    {"selu",              TritonOpCategory::UNARY_ELEMENTWISE},
    {"Selu",              TritonOpCategory::UNARY_ELEMENTWISE},
    {"softplus",          TritonOpCategory::UNARY_ELEMENTWISE},
    {"Softplus",          TritonOpCategory::UNARY_ELEMENTWISE},
    {"softsign",          TritonOpCategory::UNARY_ELEMENTWISE},
    {"Softsign",          TritonOpCategory::UNARY_ELEMENTWISE},
    {"hard_sigmoid",      TritonOpCategory::UNARY_ELEMENTWISE},
    {"HardSigmoid",       TritonOpCategory::UNARY_ELEMENTWISE},
    {"hardtanh",          TritonOpCategory::UNARY_ELEMENTWISE},
    {"HardTanh",          TritonOpCategory::UNARY_ELEMENTWISE},
    {"relu6",             TritonOpCategory::UNARY_ELEMENTWISE},
    {"Relu6",             TritonOpCategory::UNARY_ELEMENTWISE},
    {"clamp",             TritonOpCategory::UNARY_ELEMENTWISE},
    {"ClipByValue",       TritonOpCategory::UNARY_ELEMENTWISE},
    {"clipbyvalue",       TritonOpCategory::UNARY_ELEMENTWISE},
    {"clip_by_value",     TritonOpCategory::UNARY_ELEMENTWISE},
    // Scalar ops (second operand from tArgs[0])
    {"add_scalar",        TritonOpCategory::UNARY_ELEMENTWISE},
    {"subtract_scalar",   TritonOpCategory::UNARY_ELEMENTWISE},
    {"multiply_scalar",   TritonOpCategory::UNARY_ELEMENTWISE},
    {"divide_scalar",     TritonOpCategory::UNARY_ELEMENTWISE},

    // ── Comparison ──
    {"greater",           TritonOpCategory::COMPARISON},
    {"Greater",           TritonOpCategory::COMPARISON},
    {"greater_equal",     TritonOpCategory::COMPARISON},
    {"GreaterEqual",      TritonOpCategory::COMPARISON},
    {"less",              TritonOpCategory::COMPARISON},
    {"Less",              TritonOpCategory::COMPARISON},
    {"less_equal",        TritonOpCategory::COMPARISON},
    {"LessEqual",         TritonOpCategory::COMPARISON},
    {"equals",            TritonOpCategory::COMPARISON},
    {"Equals",            TritonOpCategory::COMPARISON},
    {"not_equals",        TritonOpCategory::COMPARISON},
    {"NotEquals",         TritonOpCategory::COMPARISON},

    // ── Logical ──
    {"boolean_and",       TritonOpCategory::LOGICAL},
    {"BooleanAnd",        TritonOpCategory::LOGICAL},
    {"logical_and",       TritonOpCategory::LOGICAL},
    {"LogicalAnd",        TritonOpCategory::LOGICAL},
    {"boolean_or",        TritonOpCategory::LOGICAL},
    {"BooleanOr",         TritonOpCategory::LOGICAL},
    {"logical_or",        TritonOpCategory::LOGICAL},
    {"LogicalOr",         TritonOpCategory::LOGICAL},
    {"boolean_not",       TritonOpCategory::LOGICAL},
    {"BooleanNot",        TritonOpCategory::LOGICAL},
    {"bool_not",          TritonOpCategory::LOGICAL},
    {"logical_not",       TritonOpCategory::LOGICAL},
    {"LogicalNot",        TritonOpCategory::LOGICAL},
    {"boolean_xor",       TritonOpCategory::LOGICAL},
    {"BooleanXor",        TritonOpCategory::LOGICAL},

    // ── Ternary (3 inputs) ──
    {"where",             TritonOpCategory::TERNARY},
    {"Where",             TritonOpCategory::TERNARY},
    {"select",            TritonOpCategory::TERNARY},
    {"Select",            TritonOpCategory::TERNARY},

    // ── Identity ──
    {"identity",          TritonOpCategory::IDENTITY},
    {"Identity",          TritonOpCategory::IDENTITY},
    {"assign",            TritonOpCategory::IDENTITY},
    {"Assign",            TritonOpCategory::IDENTITY},

    // ── Cast ──
    {"cast",              TritonOpCategory::CAST},
    {"Cast",              TritonOpCategory::CAST},

    // ── Matmul ──
    {"matmul",            TritonOpCategory::MATMUL},
    {"MatMul",            TritonOpCategory::MATMUL},
    {"mmul",              TritonOpCategory::MATMUL},
    {"batch_matmul",      TritonOpCategory::MATMUL},
    {"BatchMatMul",       TritonOpCategory::MATMUL},
    {"tensormmul",        TritonOpCategory::MATMUL},
    {"TensorMmul",        TritonOpCategory::MATMUL},
    {"batched_gemm",      TritonOpCategory::MATMUL},
    {"BatchedGemm",       TritonOpCategory::MATMUL},
    {"xw_plus_b",         TritonOpCategory::MATMUL},
    {"XwPlusB",           TritonOpCategory::MATMUL},

    // ── Reduction ──
    {"reduce_sum",        TritonOpCategory::REDUCTION},
    {"ReduceSum",         TritonOpCategory::REDUCTION},
    {"reduce_max",        TritonOpCategory::REDUCTION},
    {"ReduceMax",         TritonOpCategory::REDUCTION},
    {"reduce_min",        TritonOpCategory::REDUCTION},
    {"ReduceMin",         TritonOpCategory::REDUCTION},
    {"reduce_mean",       TritonOpCategory::REDUCTION},
    {"ReduceMean",        TritonOpCategory::REDUCTION},
    {"reduce_prod",       TritonOpCategory::REDUCTION},
    {"ReduceProd",        TritonOpCategory::REDUCTION},
    {"reduce_norm1",      TritonOpCategory::REDUCTION},
    {"ReduceNorm1",       TritonOpCategory::REDUCTION},
    {"reduce_norm2",      TritonOpCategory::REDUCTION},
    {"ReduceNorm2",       TritonOpCategory::REDUCTION},
    {"reduce_logsumexp",  TritonOpCategory::REDUCTION},
    {"ReduceLogSumExp",   TritonOpCategory::REDUCTION},
    {"reduce_variance",   TritonOpCategory::REDUCTION},
    {"ReduceVariance",    TritonOpCategory::REDUCTION},
    {"reduce_stdev",      TritonOpCategory::REDUCTION},
    {"ReduceStdev",       TritonOpCategory::REDUCTION},
    {"sum",               TritonOpCategory::REDUCTION},
    {"Sum",               TritonOpCategory::REDUCTION},
    {"mean",              TritonOpCategory::REDUCTION},
    {"Mean",              TritonOpCategory::REDUCTION},
    {"max",               TritonOpCategory::REDUCTION},
    {"min",               TritonOpCategory::REDUCTION},
    {"prod",              TritonOpCategory::REDUCTION},
    {"Prod",              TritonOpCategory::REDUCTION},
    {"norm1",             TritonOpCategory::REDUCTION},
    {"norm2",             TritonOpCategory::REDUCTION},
    {"normmax",           TritonOpCategory::REDUCTION},
    {"argmax",            TritonOpCategory::REDUCTION},
    {"Argmax",            TritonOpCategory::REDUCTION},
    {"argmin",            TritonOpCategory::REDUCTION},
    {"Argmin",            TritonOpCategory::REDUCTION},

    // ── Normalization ──
    {"softmax",           TritonOpCategory::NORMALIZATION},
    {"Softmax",           TritonOpCategory::NORMALIZATION},
    {"log_softmax",       TritonOpCategory::NORMALIZATION},
    {"LogSoftmax",        TritonOpCategory::NORMALIZATION},
    {"layer_norm",        TritonOpCategory::NORMALIZATION},
    {"LayerNorm",         TritonOpCategory::NORMALIZATION},
    {"batch_norm",        TritonOpCategory::NORMALIZATION},
    {"BatchNorm",         TritonOpCategory::NORMALIZATION},
    {"rms_norm",          TritonOpCategory::NORMALIZATION},
    {"RmsNorm",           TritonOpCategory::NORMALIZATION},
    {"normalize_moments", TritonOpCategory::NORMALIZATION},
    {"NormalizeMoments",  TritonOpCategory::NORMALIZATION},

    // ── Fused attention ──
    {"onnx_multi_head_attention",  TritonOpCategory::FUSED_ATTENTION},
    {"OnnxMultiHeadAttention",     TritonOpCategory::FUSED_ATTENTION},
    {"multi_head_attention",       TritonOpCategory::FUSED_ATTENTION},
    {"MultiHeadAttention",         TritonOpCategory::FUSED_ATTENTION},
    {"dot_product_attention_v2",   TritonOpCategory::FUSED_ATTENTION},
    {"DotProductAttentionV2",      TritonOpCategory::FUSED_ATTENTION},

    // ── Shape manipulation ──
    {"reshape",           TritonOpCategory::SHAPE_MANIPULATION},
    {"Reshape",           TritonOpCategory::SHAPE_MANIPULATION},
    {"permute",           TritonOpCategory::SHAPE_MANIPULATION},
    {"Permute",           TritonOpCategory::SHAPE_MANIPULATION},
    {"expand_dims",       TritonOpCategory::SHAPE_MANIPULATION},
    {"ExpandDims",        TritonOpCategory::SHAPE_MANIPULATION},
    {"squeeze",           TritonOpCategory::SHAPE_MANIPULATION},
    {"Squeeze",           TritonOpCategory::SHAPE_MANIPULATION},
    {"flatten_2d",        TritonOpCategory::SHAPE_MANIPULATION},
    {"Flatten2d",         TritonOpCategory::SHAPE_MANIPULATION},
    {"flatten",           TritonOpCategory::SHAPE_MANIPULATION},
    {"Flatten",           TritonOpCategory::SHAPE_MANIPULATION},

    // ── Data movement ──
    {"gather",            TritonOpCategory::DATA_MOVEMENT},
    {"Gather",            TritonOpCategory::DATA_MOVEMENT},
    {"gather_nd",         TritonOpCategory::DATA_MOVEMENT},
    {"GatherNd",          TritonOpCategory::DATA_MOVEMENT},
    {"concat",            TritonOpCategory::DATA_MOVEMENT},
    {"Concat",            TritonOpCategory::DATA_MOVEMENT},
    {"split",             TritonOpCategory::DATA_MOVEMENT},
    {"Split",             TritonOpCategory::DATA_MOVEMENT},
    {"split_v",           TritonOpCategory::DATA_MOVEMENT},
    {"SplitV",            TritonOpCategory::DATA_MOVEMENT},
    {"stack",             TritonOpCategory::DATA_MOVEMENT},
    {"Stack",             TritonOpCategory::DATA_MOVEMENT},
    {"strided_slice",     TritonOpCategory::DATA_MOVEMENT},
    {"StridedSlice",      TritonOpCategory::DATA_MOVEMENT},
    {"tile",              TritonOpCategory::DATA_MOVEMENT},
    {"Tile",              TritonOpCategory::DATA_MOVEMENT},
    {"scatter_nd_update", TritonOpCategory::DATA_MOVEMENT},
    {"ScatterNdUpdate",   TritonOpCategory::DATA_MOVEMENT},
    {"scatter_nd",        TritonOpCategory::DATA_MOVEMENT},
    {"ScatterNd",         TritonOpCategory::DATA_MOVEMENT},

    // ── Constant generation ──
    {"shape_of",          TritonOpCategory::CONSTANT_GENERATION},
    {"ShapeOf",           TritonOpCategory::CONSTANT_GENERATION},
    {"create",            TritonOpCategory::CONSTANT_GENERATION},
    {"Create",            TritonOpCategory::CONSTANT_GENERATION},
    {"set_scalar",        TritonOpCategory::CONSTANT_GENERATION},
    {"SetScalar",         TritonOpCategory::CONSTANT_GENERATION},
    {"ones_as",           TritonOpCategory::CONSTANT_GENERATION},
    {"OnesAs",            TritonOpCategory::CONSTANT_GENERATION},
    {"ones_like",         TritonOpCategory::CONSTANT_GENERATION},
    {"oneslike",          TritonOpCategory::CONSTANT_GENERATION},
    {"zeros_like",        TritonOpCategory::CONSTANT_GENERATION},
    {"zeroslike",         TritonOpCategory::CONSTANT_GENERATION},
    {"ZerosLike",         TritonOpCategory::CONSTANT_GENERATION},
    {"zeros_as",          TritonOpCategory::CONSTANT_GENERATION},
    {"range",             TritonOpCategory::CONSTANT_GENERATION},
    {"Range",             TritonOpCategory::CONSTANT_GENERATION},
    {"min_max_datatype",  TritonOpCategory::CONSTANT_GENERATION},
    {"MinMaxDatatype",    TritonOpCategory::CONSTANT_GENERATION},

    // ── Convolution ──
    {"conv2d",            TritonOpCategory::CONVOLUTION},
    {"Conv2d",            TritonOpCategory::CONVOLUTION},
    {"conv2D",            TritonOpCategory::CONVOLUTION},
    {"conv3d",            TritonOpCategory::CONVOLUTION},
    {"Conv3d",            TritonOpCategory::CONVOLUTION},
    {"depthwise_conv2d",  TritonOpCategory::CONVOLUTION},
    {"DepthwiseConv2d",   TritonOpCategory::CONVOLUTION},

    // ── im2col / col2im (convolution helpers) ──
    {"im2col",            TritonOpCategory::CONVOLUTION},
    {"Im2col",            TritonOpCategory::CONVOLUTION},
    {"im2col_bp",         TritonOpCategory::CONVOLUTION},
    {"col2im",            TritonOpCategory::CONVOLUTION},
    {"Col2im",            TritonOpCategory::CONVOLUTION},
    {"col2im_bp",         TritonOpCategory::CONVOLUTION},
  };
  return table;
}

/**
 * Look up the op category for a libnd4j op name.
 * Throws if the op is not in the table — every op must be manually categorized.
 */
inline TritonOpCategory getOpCategoryFromName(const std::string& opName) {
  const auto& table = getOpCategoryTable();
  auto it = table.find(opName);
  if (it != table.end()) return it->second;
  std::string msg = "getOpCategoryFromName: op '" + opName + "' is missing from OpCategoryTable. "
                    "Every op MUST be manually categorized. Add it now.";
  THROW_EXCEPTION(msg.c_str());
  return TritonOpCategory::UNSUPPORTED;
}

/**
 * Check if an op category can be JIT-compiled by NVRTC into CUDA C.
 * Accepts all per-element categories. Rejects MATMUL, FUSED_ATTENTION,
 * SHAPE_MANIPULATION, DATA_MOVEMENT, CONSTANT_GENERATION, UNSUPPORTED.
 */
inline bool isNvrtcJittable(TritonOpCategory cat) {
  switch (cat) {
    case TritonOpCategory::BINARY_ELEMENTWISE:
    case TritonOpCategory::UNARY_ELEMENTWISE:
    case TritonOpCategory::COMPARISON:
    case TritonOpCategory::LOGICAL:
    case TritonOpCategory::TERNARY:
    case TritonOpCategory::IDENTITY:
    case TritonOpCategory::CAST:
      return true;
    default:
      return false;
  }
}

/**
 * Check if an op category can be fused into a 1D element-wise Triton kernel.
 * Includes categories that have actual IR emission support in buildModule():
 * - Per-element ops (binary, unary, comparison, logical, ternary, identity, cast)
 * - Reduction and normalization (emitReductionOp/emitNormalizationOp)
 * - Shape manipulation (SSA forwarding — views don't change data in 1D)
 * - Constant generation (splat constants)
 *
 * Does NOT include categories that need their own kernel structure:
 * - MATMUL (needs 2D tiled kernel with K-loop via buildMatmulModule)
 * - FUSED_ATTENTION (needs Flash Attention kernel via emitFusedAttentionKernel)
 * - DATA_MOVEMENT (gather/concat/split need indexed access patterns)
 * - CONVOLUTION (needs spatial tiling)
 */
inline bool isElementwiseCompatible(TritonOpCategory cat) {
  switch (cat) {
    case TritonOpCategory::BINARY_ELEMENTWISE:
    case TritonOpCategory::UNARY_ELEMENTWISE:
    case TritonOpCategory::COMPARISON:
    case TritonOpCategory::LOGICAL:
    case TritonOpCategory::TERNARY:
    case TritonOpCategory::IDENTITY:
    case TritonOpCategory::CAST:
    case TritonOpCategory::REDUCTION:
    case TritonOpCategory::NORMALIZATION:
    case TritonOpCategory::SHAPE_MANIPULATION:
    case TritonOpCategory::CONSTANT_GENERATION:
      return true;
    default:
      return false;
  }
}

/**
 * Determine number of tensor inputs for an op category.
 * Scalar ops (add_scalar, etc.) are UNARY_ELEMENTWISE with scalar from tArgs.
 */
inline int categoryInputCount(TritonOpCategory cat) {
  switch (cat) {
    case TritonOpCategory::UNARY_ELEMENTWISE:
    case TritonOpCategory::CAST:
      return 1;
    case TritonOpCategory::BINARY_ELEMENTWISE:
    case TritonOpCategory::COMPARISON:
    case TritonOpCategory::LOGICAL:
      return 2;
    case TritonOpCategory::TERNARY:
      return 3;
    case TritonOpCategory::IDENTITY:
      return 1;
    default:
      return -1;
  }
}

}  // namespace graph
}  // namespace sd

#endif  // SD_CUDA
#endif  // LIBND4J_OP_CATEGORY_TABLE_H

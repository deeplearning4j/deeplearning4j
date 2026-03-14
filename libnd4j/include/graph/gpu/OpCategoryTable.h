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

#include <config.h>

#if defined(SD_CUDA) || defined(HAVE_MLIR) || HAVE_TRITON || HAVE_MLX

#include <string>
#include <unordered_map>
#include <system/common.h>
#include <graph/DspDiagnostics.h>

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
  ROPE,                   // rope, fused_rope -> rotary position embedding
  FUSED_LLM,              // fused_rms_norm_swiglu, fused_bias_dropout_residual -> mega-fused LLM kernels
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
    {"mul_scalar",        TritonOpCategory::UNARY_ELEMENTWISE},
    {"divide_scalar",     TritonOpCategory::UNARY_ELEMENTWISE},
    {"div_scalar",        TritonOpCategory::UNARY_ELEMENTWISE},
    {"sub_scalar",        TritonOpCategory::UNARY_ELEMENTWISE},

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
    // argmax/argmin require index-aware reduction — no Triton emitter, fall back to native
    {"argmax",            TritonOpCategory::UNSUPPORTED},
    {"Argmax",            TritonOpCategory::UNSUPPORTED},
    {"argmin",            TritonOpCategory::UNSUPPORTED},
    {"Argmin",            TritonOpCategory::UNSUPPORTED},

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
    // normalize_moments has two outputs — unsupported in single-output Triton emission
    {"normalize_moments", TritonOpCategory::UNSUPPORTED},
    {"NormalizeMoments",  TritonOpCategory::UNSUPPORTED},

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

    // ── Pooling (convolution category) ──
    {"maxpool2d",         TritonOpCategory::CONVOLUTION},
    {"MaxPool2d",         TritonOpCategory::CONVOLUTION},
    {"avgpool2d",         TritonOpCategory::CONVOLUTION},
    {"AvgPool2d",         TritonOpCategory::CONVOLUTION},
    {"maxpool3d",         TritonOpCategory::CONVOLUTION},
    {"MaxPool3d",         TritonOpCategory::CONVOLUTION},
    {"avgpool3d",         TritonOpCategory::CONVOLUTION},
    {"AvgPool3d",         TritonOpCategory::CONVOLUTION},
    {"conv1d",            TritonOpCategory::CONVOLUTION},
    {"Conv1d",            TritonOpCategory::CONVOLUTION},
    {"conv3dnew",         TritonOpCategory::CONVOLUTION},
    {"Conv3dNew",         TritonOpCategory::CONVOLUTION},
    {"sconv2d",           TritonOpCategory::CONVOLUTION},
    {"deconv2d",          TritonOpCategory::CONVOLUTION},
    {"deconv3d",          TritonOpCategory::CONVOLUTION},

    // ── Loss ops — no Triton emitter, fall back to native ──
    {"softmax_cross_entropy_loss_with_logits", TritonOpCategory::UNSUPPORTED},
    {"sparse_softmax_cross_entropy_loss_with_logits", TritonOpCategory::UNSUPPORTED},
    {"sigm_cross_entropy_loss",               TritonOpCategory::UNSUPPORTED},
    {"mean_sqerr_loss",                       TritonOpCategory::UNSUPPORTED},
    {"mean_pairwssqerr_loss",                 TritonOpCategory::UNSUPPORTED},
    {"huber_loss",                            TritonOpCategory::UNSUPPORTED},
    {"hinge_loss",                            TritonOpCategory::UNSUPPORTED},
    {"log_loss",                              TritonOpCategory::UNSUPPORTED},
    {"cosine_distance_loss",                  TritonOpCategory::UNSUPPORTED},
    {"ctc_loss",                              TritonOpCategory::UNSUPPORTED},

    // ── Embedding / segment ops ──
    {"embedding_lookup",  TritonOpCategory::DATA_MOVEMENT},
    {"EmbeddingLookup",   TritonOpCategory::DATA_MOVEMENT},
    // Segment ops have no Triton emitter — fall back to native
    {"segment_sum",       TritonOpCategory::UNSUPPORTED},
    {"segment_mean",      TritonOpCategory::UNSUPPORTED},
    {"segment_max",       TritonOpCategory::UNSUPPORTED},
    {"segment_min",       TritonOpCategory::UNSUPPORTED},
    {"unsorted_segment_sum", TritonOpCategory::UNSUPPORTED},
    {"unique",            TritonOpCategory::DATA_MOVEMENT},
    {"top_k",             TritonOpCategory::DATA_MOVEMENT},
    {"in_top_k",          TritonOpCategory::DATA_MOVEMENT},

    // ── One-hot (constant generation) ──
    {"onehot",            TritonOpCategory::CONSTANT_GENERATION},
    {"OneHot",            TritonOpCategory::CONSTANT_GENERATION},

    // ── Attention ops ──
    {"dot_product_attention",  TritonOpCategory::FUSED_ATTENTION},
    {"DotProductAttention",    TritonOpCategory::FUSED_ATTENTION},
    {"self_attention",         TritonOpCategory::FUSED_ATTENTION},
    {"SelfAttention",          TritonOpCategory::FUSED_ATTENTION},

    // ── Normalization ──
    {"batchnorm",         TritonOpCategory::NORMALIZATION},
    {"BatchNorm",         TritonOpCategory::NORMALIZATION},
    {"layer_normalization", TritonOpCategory::NORMALIZATION},

    // ── RNN ops — multi-step decomposed, not simple matmul. Fall back to native. ──
    {"lstmLayer",         TritonOpCategory::UNSUPPORTED},
    {"gruCell",           TritonOpCategory::UNSUPPORTED},
    {"gru",               TritonOpCategory::UNSUPPORTED},
    {"simple_rnn",        TritonOpCategory::UNSUPPORTED},
    {"lstmCell",          TritonOpCategory::UNSUPPORTED},

    // ── Image ops (data movement category) ──
    {"resize_bilinear",        TritonOpCategory::DATA_MOVEMENT},
    {"resize_nearest_neighbor", TritonOpCategory::DATA_MOVEMENT},
    {"resize_bicubic",         TritonOpCategory::DATA_MOVEMENT},
    {"crop_and_resize",        TritonOpCategory::DATA_MOVEMENT},
    // Color space conversions operate across channels, not per-element — NOT elementwise
    {"rgb_to_hsv",             TritonOpCategory::DATA_MOVEMENT},
    {"hsv_to_rgb",             TritonOpCategory::DATA_MOVEMENT},
    {"rgb_to_grs",             TritonOpCategory::DATA_MOVEMENT},
    {"adjust_contrast",        TritonOpCategory::DATA_MOVEMENT},
    {"adjust_hue",             TritonOpCategory::DATA_MOVEMENT},
    {"adjust_saturation",      TritonOpCategory::DATA_MOVEMENT},
    {"image_resize",           TritonOpCategory::DATA_MOVEMENT},

    // ── Extended activations (unary elementwise) ──
    {"hardswish",         TritonOpCategory::UNARY_ELEMENTWISE},
    {"HardSwish",         TritonOpCategory::UNARY_ELEMENTWISE},
    {"hardsigmoid",       TritonOpCategory::UNARY_ELEMENTWISE},
    {"HardSigmoid2",      TritonOpCategory::UNARY_ELEMENTWISE},
    {"celu",              TritonOpCategory::UNARY_ELEMENTWISE},
    {"Celu",              TritonOpCategory::UNARY_ELEMENTWISE},
    {"thresholdedrelu",   TritonOpCategory::UNARY_ELEMENTWISE},
    {"ThresholdedRelu",   TritonOpCategory::UNARY_ELEMENTWISE},
    {"prelu",             TritonOpCategory::BINARY_ELEMENTWISE},
    {"Prelu",             TritonOpCategory::BINARY_ELEMENTWISE},

    // ── LLM: Rotary Position Embedding ──
    {"rope",              TritonOpCategory::ROPE},
    {"Rope",              TritonOpCategory::ROPE},
    {"rope_bp",           TritonOpCategory::ROPE},
    {"fused_rope",        TritonOpCategory::ROPE},
    {"FusedRope",         TritonOpCategory::ROPE},
    {"fused_rope_bp",     TritonOpCategory::ROPE},

    // ── LLM: Attention variants ──
    {"flash_attention",            TritonOpCategory::FUSED_ATTENTION},
    {"FlashAttention",             TritonOpCategory::FUSED_ATTENTION},
    {"grouped_query_attention",    TritonOpCategory::FUSED_ATTENTION},
    {"GroupedQueryAttention",      TritonOpCategory::FUSED_ATTENTION},
    {"sliding_window_attention",   TritonOpCategory::FUSED_ATTENTION},
    {"SlidingWindowAttention",     TritonOpCategory::FUSED_ATTENTION},
    {"apply_alibi",                TritonOpCategory::FUSED_ATTENTION},
    {"ApplyAlibi",                 TritonOpCategory::FUSED_ATTENTION},

    // ── LLM: KV cache management ──
    {"kv_cache_update",   TritonOpCategory::DATA_MOVEMENT},
    {"KvCacheUpdate",     TritonOpCategory::DATA_MOVEMENT},
    {"kv_scatter",        TritonOpCategory::DATA_MOVEMENT},
    {"KvScatter",         TritonOpCategory::DATA_MOVEMENT},

    // ── LLM: Fused mega-kernels ──
    {"fused_rms_norm_swiglu",      TritonOpCategory::FUSED_LLM},
    {"FusedRmsNormSwiglu",         TritonOpCategory::FUSED_LLM},
    {"fused_bias_dropout_residual", TritonOpCategory::FUSED_LLM},
    {"FusedBiasDropoutResidual",   TritonOpCategory::FUSED_LLM},
    {"fused_elementwise_chain",    TritonOpCategory::FUSED_LLM},
    {"FusedElementwiseChain",      TritonOpCategory::FUSED_LLM},

    // ── LLM: Fused activations/normalization ──
    {"fused_gelu",        TritonOpCategory::UNARY_ELEMENTWISE},
    {"FusedGelu",         TritonOpCategory::UNARY_ELEMENTWISE},
    {"fused_layer_norm",  TritonOpCategory::NORMALIZATION},
    {"FusedLayerNorm",    TritonOpCategory::NORMALIZATION},

    // ── LLM: Quantized matmul ──
    {"quantized_matmul",  TritonOpCategory::MATMUL},
    {"QuantizedMatmul",   TritonOpCategory::MATMUL},

    // ── LLM: Mean square (RMS norm building block) — no standalone Triton emitter ──
    {"mean_square",       TritonOpCategory::UNSUPPORTED},
    {"MeanSquare",        TritonOpCategory::UNSUPPORTED},

    // ── Additional data movement ops ──
    {"repeat",            TritonOpCategory::DATA_MOVEMENT},
    {"Repeat",            TritonOpCategory::DATA_MOVEMENT},
    {"pad",               TritonOpCategory::DATA_MOVEMENT},
    {"Pad",               TritonOpCategory::DATA_MOVEMENT},
    {"mirror_pad",        TritonOpCategory::DATA_MOVEMENT},
    {"MirrorPad",         TritonOpCategory::DATA_MOVEMENT},
    {"slice",             TritonOpCategory::DATA_MOVEMENT},
    {"Slice",             TritonOpCategory::DATA_MOVEMENT},

    // ── Additional shape manipulation ──
    {"transpose",         TritonOpCategory::SHAPE_MANIPULATION},
    {"Transpose",         TritonOpCategory::SHAPE_MANIPULATION},
    {"reshape_no_copy",   TritonOpCategory::SHAPE_MANIPULATION},
    {"ReshapeNoCopy",     TritonOpCategory::SHAPE_MANIPULATION},
    {"triu",              TritonOpCategory::SHAPE_MANIPULATION},
    {"Triu",              TritonOpCategory::SHAPE_MANIPULATION},
    {"tril",              TritonOpCategory::SHAPE_MANIPULATION},
    {"Tril",              TritonOpCategory::SHAPE_MANIPULATION},

    // ── Prefix-sum ops — not tree reductions, no Triton emitter ──
    {"cumsum",            TritonOpCategory::UNSUPPORTED},
    {"Cumsum",            TritonOpCategory::UNSUPPORTED},
    {"cumprod",           TritonOpCategory::UNSUPPORTED},
    {"Cumprod",           TritonOpCategory::UNSUPPORTED},

    // ── Additional constant generation ──
    {"eye",               TritonOpCategory::CONSTANT_GENERATION},
    {"Eye",               TritonOpCategory::CONSTANT_GENERATION},
    {"linspace",          TritonOpCategory::CONSTANT_GENERATION},
    {"Linspace",          TritonOpCategory::CONSTANT_GENERATION},
    {"lin_space",         TritonOpCategory::CONSTANT_GENERATION},
    {"LinSpace",          TritonOpCategory::CONSTANT_GENERATION},
    {"fill",              TritonOpCategory::CONSTANT_GENERATION},
    {"Fill",              TritonOpCategory::CONSTANT_GENERATION},

    // ── Ops commonly needed by transformer/VLM models ──
    // Shape/indexing
    {"size_at",           TritonOpCategory::CONSTANT_GENERATION},
    {"SizeAt",            TritonOpCategory::CONSTANT_GENERATION},
    {"size",              TritonOpCategory::CONSTANT_GENERATION},
    {"Size",              TritonOpCategory::CONSTANT_GENERATION},
    {"rank",              TritonOpCategory::CONSTANT_GENERATION},
    {"Rank",              TritonOpCategory::CONSTANT_GENERATION},
    {"broadcast_to",      TritonOpCategory::SHAPE_MANIPULATION},
    {"BroadcastTo",       TritonOpCategory::SHAPE_MANIPULATION},

    // Ternary
    {"where_np",          TritonOpCategory::TERNARY},
    {"WhereNp",           TritonOpCategory::TERNARY},

    // Binary elementwise
    {"biasadd",           TritonOpCategory::BINARY_ELEMENTWISE},
    {"BiasAdd",           TritonOpCategory::BINARY_ELEMENTWISE},

    // Bitwise ops
    {"bitwise_and",       TritonOpCategory::BINARY_ELEMENTWISE},
    {"BitwiseAnd",        TritonOpCategory::BINARY_ELEMENTWISE},
    {"bitwise_or",        TritonOpCategory::BINARY_ELEMENTWISE},
    {"BitwiseOr",         TritonOpCategory::BINARY_ELEMENTWISE},
    {"bitwise_xor",       TritonOpCategory::BINARY_ELEMENTWISE},
    {"BitwiseXor",        TritonOpCategory::BINARY_ELEMENTWISE},
    {"shift_bits",        TritonOpCategory::BINARY_ELEMENTWISE},
    {"ShiftBits",         TritonOpCategory::BINARY_ELEMENTWISE},
    {"rshift_bits",       TritonOpCategory::BINARY_ELEMENTWISE},
    {"RShiftBits",        TritonOpCategory::BINARY_ELEMENTWISE},
    {"toggle_bits",       TritonOpCategory::UNARY_ELEMENTWISE},
    {"ToggleBits",        TritonOpCategory::UNARY_ELEMENTWISE},

    // Inference/sampling
    {"token_sample",      TritonOpCategory::DATA_MOVEMENT},
    {"TokenSample",       TritonOpCategory::DATA_MOVEMENT},

    // Additional data movement
    {"unstack",           TritonOpCategory::DATA_MOVEMENT},
    {"Unstack",           TritonOpCategory::DATA_MOVEMENT},
    {"reverse",           TritonOpCategory::DATA_MOVEMENT},
    {"Reverse",           TritonOpCategory::DATA_MOVEMENT},
    {"reverse_v2",        TritonOpCategory::DATA_MOVEMENT},
    {"ReverseV2",         TritonOpCategory::DATA_MOVEMENT},
    {"sequence_mask",     TritonOpCategory::CONSTANT_GENERATION},
    {"SequenceMask",      TritonOpCategory::CONSTANT_GENERATION},

    // Additional shape manipulation
    {"reshapeas",         TritonOpCategory::SHAPE_MANIPULATION},
    {"ReshapeAs",         TritonOpCategory::SHAPE_MANIPULATION},

    // Fused LLM ops
    {"fused_gelu",        TritonOpCategory::UNARY_ELEMENTWISE},
    {"FusedGelu",         TritonOpCategory::UNARY_ELEMENTWISE},
  };
  return table;
}

/**
 * Look up the op category for a libnd4j op name.
 * Returns UNSUPPORTED for ops not in the table — they will fall back to native execution.
 * Logs a warning so missing ops can be detected and added to the table.
 */
inline TritonOpCategory getOpCategoryFromName(const std::string& opName) {
  const auto& table = getOpCategoryTable();
  auto it = table.find(opName);
  if (it != table.end()) return it->second;
  // Return UNSUPPORTED instead of throwing — allows graceful fallback to native execution.
  // The op will be executed via the standard native path rather than crashing.
  DSP_DIAG(FALLBACK, "getOpCategoryFromName: op '%s' is missing from OpCategoryTable. "
            "Falling back to native execution. Add it to OpCategoryTable.h for proper categorization.",
            opName.c_str());
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

#endif  // defined(SD_CUDA) || defined(HAVE_MLIR) || HAVE_TRITON || HAVE_MLX
#endif  // LIBND4J_OP_CATEGORY_TABLE_H

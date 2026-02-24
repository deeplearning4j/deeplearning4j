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

#include <config.h>

#if HAVE_TRITON

#include <graph/gpu/TritonIRBuilder.h>
#include <array/ArrayOptions.h>
#include <execution/cuda/LaunchDims.h>
#include <helpers/logger.h>
#include <helpers/shape.h>
#include <system/common.h>

#include <algorithm>
#include <cmath>
#include <sstream>

// MLIR core
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>

// Triton MLIR dialect
#include <triton/Dialect/Triton/IR/Dialect.h>
#include <triton/Dialect/Triton/IR/Types.h>

// Standard MLIR dialects
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/SCF/IR/SCF.h>

namespace sd {
namespace graph {

// Maximum number of direct function arguments before switching to indirect
// argument passing via a pointer array. LLVM/Triton crash with an ArrayRef
// assertion when a tt.func has more than ~250 parameters. With indirect passing,
// the kernel receives (argArray* : !tt.ptr<i64>, n_elements : i32) and unpacks
// buffer pointers with indexed loads from the array.
static constexpr int TRITON_DIRECT_ARG_LIMIT = 200;

// ─── Op mapping table ───────────────────────────────────────────────────────

static std::unordered_map<std::string, TritonOpMapping> buildOpTable() {
  std::unordered_map<std::string, TritonOpMapping> table;

  // Binary element-wise
  table["add"]       = {"add",       TritonOpCategory::BINARY_ELEMENTWISE, "arith.addf",     false};
  table["Add"]       = {"Add",       TritonOpCategory::BINARY_ELEMENTWISE, "arith.addf",     false};
  table["subtract"]  = {"subtract",  TritonOpCategory::BINARY_ELEMENTWISE, "arith.subf",     false};
  table["Sub"]       = {"Sub",       TritonOpCategory::BINARY_ELEMENTWISE, "arith.subf",     false};
  table["multiply"]  = {"multiply",  TritonOpCategory::BINARY_ELEMENTWISE, "arith.mulf",     false};
  table["Mul"]       = {"Mul",       TritonOpCategory::BINARY_ELEMENTWISE, "arith.mulf",     false};
  table["divide"]    = {"divide",    TritonOpCategory::BINARY_ELEMENTWISE, "arith.divf",     false};
  table["Div"]       = {"Div",       TritonOpCategory::BINARY_ELEMENTWISE, "arith.divf",     false};
  table["RealDiv"]   = {"RealDiv",   TritonOpCategory::BINARY_ELEMENTWISE, "arith.divf",     false};
  table["minimum"]   = {"minimum",   TritonOpCategory::BINARY_ELEMENTWISE, "arith.minimumf", false};
  table["Min"]       = {"Min",       TritonOpCategory::BINARY_ELEMENTWISE, "arith.minimumf", false};
  table["maximum"]   = {"maximum",   TritonOpCategory::BINARY_ELEMENTWISE, "arith.maximumf", false};
  table["Max"]       = {"Max",       TritonOpCategory::BINARY_ELEMENTWISE, "arith.maximumf", false};
  table["mod"]       = {"mod",       TritonOpCategory::BINARY_ELEMENTWISE, "arith.remf",     false};
  table["Mod"]       = {"Mod",       TritonOpCategory::BINARY_ELEMENTWISE, "arith.remf",     false};
  table["floormod"]  = {"floormod",  TritonOpCategory::BINARY_ELEMENTWISE, "arith.remf",     false};
  table["FloorMod"]  = {"FloorMod",  TritonOpCategory::BINARY_ELEMENTWISE, "arith.remf",     false};

  // Unary element-wise
  table["relu"]      = {"relu",      TritonOpCategory::UNARY_ELEMENTWISE,  "arith.maximumf", true};
  table["Relu"]      = {"Relu",      TritonOpCategory::UNARY_ELEMENTWISE,  "arith.maximumf", true};
  table["sigmoid"]   = {"sigmoid",   TritonOpCategory::UNARY_ELEMENTWISE,  "math.exp",       true};
  table["Sigmoid"]   = {"Sigmoid",   TritonOpCategory::UNARY_ELEMENTWISE,  "math.exp",       true};
  table["tanh"]      = {"tanh",      TritonOpCategory::UNARY_ELEMENTWISE,  "math.tanh",      false};
  table["Tanh"]      = {"Tanh",      TritonOpCategory::UNARY_ELEMENTWISE,  "math.tanh",      false};
  table["gelu"]      = {"gelu",      TritonOpCategory::UNARY_ELEMENTWISE,  "math.erf",       true};
  table["Gelu"]      = {"Gelu",      TritonOpCategory::UNARY_ELEMENTWISE,  "math.erf",       true};
  table["exp"]       = {"exp",       TritonOpCategory::UNARY_ELEMENTWISE,  "math.exp",       false};
  table["Exp"]       = {"Exp",       TritonOpCategory::UNARY_ELEMENTWISE,  "math.exp",       false};
  table["log"]       = {"log",       TritonOpCategory::UNARY_ELEMENTWISE,  "math.log",       false};
  table["Log"]       = {"Log",       TritonOpCategory::UNARY_ELEMENTWISE,  "math.log",       false};
  table["abs"]       = {"abs",       TritonOpCategory::UNARY_ELEMENTWISE,  "math.absf",      false};
  table["Abs"]       = {"Abs",       TritonOpCategory::UNARY_ELEMENTWISE,  "math.absf",      false};
  table["sqrt"]      = {"sqrt",      TritonOpCategory::UNARY_ELEMENTWISE,  "math.sqrt",      false};
  table["Sqrt"]      = {"Sqrt",      TritonOpCategory::UNARY_ELEMENTWISE,  "math.sqrt",      false};
  table["square"]    = {"square",    TritonOpCategory::UNARY_ELEMENTWISE,  "arith.mulf",     true};
  table["Square"]    = {"Square",    TritonOpCategory::UNARY_ELEMENTWISE,  "arith.mulf",     true};
  table["pow"]       = {"pow",       TritonOpCategory::UNARY_ELEMENTWISE,  "custom.pow",     true};
  table["Pow"]       = {"Pow",       TritonOpCategory::UNARY_ELEMENTWISE,  "custom.pow",     true};
  table["clamp"]     = {"clamp",     TritonOpCategory::UNARY_ELEMENTWISE,  "arith.maximumf", true};
  table["ClipByValue"] = {"ClipByValue", TritonOpCategory::UNARY_ELEMENTWISE, "arith.maximumf", true};
  table["clipbyvalue"] = {"clipbyvalue", TritonOpCategory::UNARY_ELEMENTWISE, "arith.maximumf", true};
  table["neg"]       = {"neg",       TritonOpCategory::UNARY_ELEMENTWISE,  "arith.negf",     false};
  table["Neg"]       = {"Neg",       TritonOpCategory::UNARY_ELEMENTWISE,  "arith.negf",     false};
  table["reciprocal"] = {"reciprocal", TritonOpCategory::UNARY_ELEMENTWISE, "custom.reciprocal", true};
  table["Reciprocal"] = {"Reciprocal", TritonOpCategory::UNARY_ELEMENTWISE, "custom.reciprocal", true};
  table["rsqrt"]     = {"rsqrt",     TritonOpCategory::UNARY_ELEMENTWISE,  "math.rsqrt",     false};
  table["Rsqrt"]     = {"Rsqrt",     TritonOpCategory::UNARY_ELEMENTWISE,  "math.rsqrt",     false};
  table["sign"]      = {"sign",      TritonOpCategory::UNARY_ELEMENTWISE,  "custom.sign",    true};
  table["Sign"]      = {"Sign",      TritonOpCategory::UNARY_ELEMENTWISE,  "custom.sign",    true};
  table["erf"]       = {"erf",       TritonOpCategory::UNARY_ELEMENTWISE,  "math.erf",       false};
  table["Erf"]       = {"Erf",       TritonOpCategory::UNARY_ELEMENTWISE,  "math.erf",       false};
  table["log1p"]     = {"log1p",     TritonOpCategory::UNARY_ELEMENTWISE,  "math.log1p",     false};
  table["Log1p"]     = {"Log1p",     TritonOpCategory::UNARY_ELEMENTWISE,  "math.log1p",     false};
  table["ceil"]      = {"ceil",      TritonOpCategory::UNARY_ELEMENTWISE,  "math.ceil",      false};
  table["Ceil"]      = {"Ceil",      TritonOpCategory::UNARY_ELEMENTWISE,  "math.ceil",      false};
  table["floor"]     = {"floor",     TritonOpCategory::UNARY_ELEMENTWISE,  "math.floor",     false};
  table["Floor"]     = {"Floor",     TritonOpCategory::UNARY_ELEMENTWISE,  "math.floor",     false};
  table["round"]     = {"round",     TritonOpCategory::UNARY_ELEMENTWISE,  "math.roundeven", false};
  table["Round"]     = {"Round",     TritonOpCategory::UNARY_ELEMENTWISE,  "math.roundeven", false};
  table["sin"]       = {"sin",       TritonOpCategory::UNARY_ELEMENTWISE,  "math.sin",       false};
  table["Sin"]       = {"Sin",       TritonOpCategory::UNARY_ELEMENTWISE,  "math.sin",       false};
  table["cos"]       = {"cos",       TritonOpCategory::UNARY_ELEMENTWISE,  "math.cos",       false};
  table["Cos"]       = {"Cos",       TritonOpCategory::UNARY_ELEMENTWISE,  "math.cos",       false};
  table["leakyrelu"] = {"leakyrelu", TritonOpCategory::UNARY_ELEMENTWISE,  "custom.leakyrelu", true};
  table["LeakyRelu"] = {"LeakyRelu", TritonOpCategory::UNARY_ELEMENTWISE,  "custom.leakyrelu", true};
  table["silu"]      = {"silu",      TritonOpCategory::UNARY_ELEMENTWISE,  "custom.silu",    true};
  table["Silu"]      = {"Silu",      TritonOpCategory::UNARY_ELEMENTWISE,  "custom.silu",    true};
  table["swish"]     = {"swish",     TritonOpCategory::UNARY_ELEMENTWISE,  "custom.silu",    true};
  table["Swish"]     = {"Swish",     TritonOpCategory::UNARY_ELEMENTWISE,  "custom.silu",    true};
  table["mish"]      = {"mish",      TritonOpCategory::UNARY_ELEMENTWISE,  "custom.mish",    true};
  table["Mish"]      = {"Mish",      TritonOpCategory::UNARY_ELEMENTWISE,  "custom.mish",    true};
  table["elu"]       = {"elu",       TritonOpCategory::UNARY_ELEMENTWISE,  "custom.elu",     true};
  table["Elu"]       = {"Elu",       TritonOpCategory::UNARY_ELEMENTWISE,  "custom.elu",     true};
  table["selu"]      = {"selu",      TritonOpCategory::UNARY_ELEMENTWISE,  "custom.selu",    true};
  table["Selu"]      = {"Selu",      TritonOpCategory::UNARY_ELEMENTWISE,  "custom.selu",    true};
  table["softplus"]  = {"softplus",  TritonOpCategory::UNARY_ELEMENTWISE,  "custom.softplus", true};
  table["Softplus"]  = {"Softplus",  TritonOpCategory::UNARY_ELEMENTWISE,  "custom.softplus", true};
  table["softsign"]  = {"softsign",  TritonOpCategory::UNARY_ELEMENTWISE,  "custom.softsign", true};
  table["Softsign"]  = {"Softsign",  TritonOpCategory::UNARY_ELEMENTWISE,  "custom.softsign", true};
  table["hard_sigmoid"] = {"hard_sigmoid", TritonOpCategory::UNARY_ELEMENTWISE, "custom.hard_sigmoid", true};
  table["HardSigmoid"] = {"HardSigmoid", TritonOpCategory::UNARY_ELEMENTWISE, "custom.hard_sigmoid", true};
  table["hardtanh"]  = {"hardtanh",  TritonOpCategory::UNARY_ELEMENTWISE,  "custom.hardtanh", true};
  table["HardTanh"]  = {"HardTanh",  TritonOpCategory::UNARY_ELEMENTWISE,  "custom.hardtanh", true};
  table["relu6"]     = {"relu6",     TritonOpCategory::UNARY_ELEMENTWISE,  "custom.relu6",   true};
  table["Relu6"]     = {"Relu6",     TritonOpCategory::UNARY_ELEMENTWISE,  "custom.relu6",   true};

  // Matrix ops
  table["matmul"]        = {"matmul",        TritonOpCategory::MATMUL, "tt.dot", false};
  table["MatMul"]        = {"MatMul",        TritonOpCategory::MATMUL, "tt.dot", false};
  table["mmul"]          = {"mmul",          TritonOpCategory::MATMUL, "tt.dot", false};
  table["batch_matmul"]  = {"batch_matmul",  TritonOpCategory::MATMUL, "tt.dot", false};
  table["BatchMatMul"]   = {"BatchMatMul",   TritonOpCategory::MATMUL, "tt.dot", false};

  // Reductions
  table["reduce_sum"]    = {"reduce_sum",    TritonOpCategory::REDUCTION, "tt.reduce", false};
  table["ReduceSum"]     = {"ReduceSum",     TritonOpCategory::REDUCTION, "tt.reduce", false};
  table["reduce_max"]    = {"reduce_max",    TritonOpCategory::REDUCTION, "tt.reduce", false};
  table["ReduceMax"]     = {"ReduceMax",     TritonOpCategory::REDUCTION, "tt.reduce", false};
  table["reduce_min"]    = {"reduce_min",    TritonOpCategory::REDUCTION, "tt.reduce", false};
  table["ReduceMin"]     = {"ReduceMin",     TritonOpCategory::REDUCTION, "tt.reduce", false};
  table["reduce_mean"]   = {"reduce_mean",   TritonOpCategory::REDUCTION, "tt.reduce", true};
  table["ReduceMean"]    = {"ReduceMean",    TritonOpCategory::REDUCTION, "tt.reduce", true};
  table["reduce_prod"]   = {"reduce_prod",   TritonOpCategory::REDUCTION, "tt.reduce", false};
  table["ReduceProd"]    = {"ReduceProd",    TritonOpCategory::REDUCTION, "tt.reduce", false};

  // Normalization (compound patterns)
  table["softmax"]       = {"softmax",       TritonOpCategory::NORMALIZATION, "tt.reduce", true};
  table["Softmax"]       = {"Softmax",       TritonOpCategory::NORMALIZATION, "tt.reduce", true};
  table["log_softmax"]   = {"log_softmax",   TritonOpCategory::NORMALIZATION, "tt.reduce", true};
  table["LogSoftmax"]    = {"LogSoftmax",    TritonOpCategory::NORMALIZATION, "tt.reduce", true};
  table["layer_norm"]    = {"layer_norm",    TritonOpCategory::NORMALIZATION, "tt.reduce", true};
  table["LayerNorm"]     = {"LayerNorm",     TritonOpCategory::NORMALIZATION, "tt.reduce", true};

  // SwiGLU: swish_mul(x, y) = x * sigmoid(x) * y — 30 instances in decoder
  table["swish_mul"]     = {"swish_mul",     TritonOpCategory::BINARY_ELEMENTWISE, "custom.swish_mul",   true};
  table["SwishMul"]      = {"SwishMul",      TritonOpCategory::BINARY_ELEMENTWISE, "custom.swish_mul",   true};

  // Scalar binary ops (second operand from tArgs)
  table["add_scalar"]      = {"add_scalar",      TritonOpCategory::UNARY_ELEMENTWISE,  "custom.add_scalar",  true};
  table["subtract_scalar"] = {"subtract_scalar", TritonOpCategory::UNARY_ELEMENTWISE,  "custom.sub_scalar",  true};
  table["multiply_scalar"] = {"multiply_scalar", TritonOpCategory::UNARY_ELEMENTWISE,  "custom.mul_scalar",  true};
  table["divide_scalar"]   = {"divide_scalar",   TritonOpCategory::UNARY_ELEMENTWISE,  "custom.div_scalar",  true};

  // Missing unary element-wise
  table["erfc"]          = {"erfc",          TritonOpCategory::UNARY_ELEMENTWISE,  "custom.erfc",        true};
  table["Erfc"]          = {"Erfc",          TritonOpCategory::UNARY_ELEMENTWISE,  "custom.erfc",        true};
  table["clip_by_value"] = {"clip_by_value", TritonOpCategory::UNARY_ELEMENTWISE,  "arith.maximumf",     true};

  // Missing binary element-wise
  table["atan2"]             = {"atan2",             TritonOpCategory::BINARY_ELEMENTWISE, "math.atan2",         false};
  table["Atan2"]             = {"Atan2",             TritonOpCategory::BINARY_ELEMENTWISE, "math.atan2",         false};
  table["floordiv"]          = {"floordiv",          TritonOpCategory::BINARY_ELEMENTWISE, "custom.floordiv",    true};
  table["FloorDiv"]          = {"FloorDiv",          TritonOpCategory::BINARY_ELEMENTWISE, "custom.floordiv",    true};
  table["reversedivide"]     = {"reversedivide",     TritonOpCategory::BINARY_ELEMENTWISE, "custom.reversediv",  true};
  table["ReverseDivide"]     = {"ReverseDivide",     TritonOpCategory::BINARY_ELEMENTWISE, "custom.reversediv",  true};
  table["reversesubtract"]   = {"reversesubtract",   TritonOpCategory::BINARY_ELEMENTWISE, "custom.reversesub",  true};
  table["ReverseSubtract"]   = {"ReverseSubtract",   TritonOpCategory::BINARY_ELEMENTWISE, "custom.reversesub",  true};
  table["squaredsubtract"]   = {"squaredsubtract",   TritonOpCategory::BINARY_ELEMENTWISE, "custom.squaredsub",  true};
  table["SquaredSubtract"]   = {"SquaredSubtract",   TritonOpCategory::BINARY_ELEMENTWISE, "custom.squaredsub",  true};
  table["multiply_no_nan"]   = {"multiply_no_nan",   TritonOpCategory::BINARY_ELEMENTWISE, "custom.mul_no_nan",  true};
  table["MultiplyNoNan"]     = {"MultiplyNoNan",     TritonOpCategory::BINARY_ELEMENTWISE, "custom.mul_no_nan",  true};
  table["min_pairwise"]      = {"min_pairwise",      TritonOpCategory::BINARY_ELEMENTWISE, "arith.minimumf",     false};
  table["MinPairwise"]       = {"MinPairwise",       TritonOpCategory::BINARY_ELEMENTWISE, "arith.minimumf",     false};
  table["max_pairwise"]      = {"max_pairwise",      TritonOpCategory::BINARY_ELEMENTWISE, "arith.maximumf",     false};
  table["MaxPairwise"]       = {"MaxPairwise",       TritonOpCategory::BINARY_ELEMENTWISE, "arith.maximumf",     false};

  // Comparison ops (element-wise, return bool)
  table["greater"]           = {"greater",           TritonOpCategory::COMPARISON, "arith.cmpf.ogt",     false};
  table["Greater"]           = {"Greater",           TritonOpCategory::COMPARISON, "arith.cmpf.ogt",     false};
  table["greater_equal"]     = {"greater_equal",     TritonOpCategory::COMPARISON, "arith.cmpf.oge",     false};
  table["GreaterEqual"]      = {"GreaterEqual",      TritonOpCategory::COMPARISON, "arith.cmpf.oge",     false};
  table["less"]              = {"less",              TritonOpCategory::COMPARISON, "arith.cmpf.olt",     false};
  table["Less"]              = {"Less",              TritonOpCategory::COMPARISON, "arith.cmpf.olt",     false};
  table["less_equal"]        = {"less_equal",        TritonOpCategory::COMPARISON, "arith.cmpf.ole",     false};
  table["LessEqual"]         = {"LessEqual",         TritonOpCategory::COMPARISON, "arith.cmpf.ole",     false};
  table["equals"]            = {"equals",            TritonOpCategory::COMPARISON, "arith.cmpf.oeq",     false};
  table["Equals"]            = {"Equals",            TritonOpCategory::COMPARISON, "arith.cmpf.oeq",     false};
  table["not_equals"]        = {"not_equals",        TritonOpCategory::COMPARISON, "arith.cmpf.one",     false};
  table["NotEquals"]         = {"NotEquals",         TritonOpCategory::COMPARISON, "arith.cmpf.one",     false};

  // Logical ops (element-wise, bool→bool)
  table["boolean_and"]       = {"boolean_and",       TritonOpCategory::LOGICAL, "arith.andi",          false};
  table["BooleanAnd"]        = {"BooleanAnd",        TritonOpCategory::LOGICAL, "arith.andi",          false};
  table["logical_and"]       = {"logical_and",       TritonOpCategory::LOGICAL, "arith.andi",          false};
  table["LogicalAnd"]        = {"LogicalAnd",        TritonOpCategory::LOGICAL, "arith.andi",          false};
  table["boolean_or"]        = {"boolean_or",        TritonOpCategory::LOGICAL, "arith.ori",           false};
  table["BooleanOr"]         = {"BooleanOr",         TritonOpCategory::LOGICAL, "arith.ori",           false};
  table["logical_or"]        = {"logical_or",        TritonOpCategory::LOGICAL, "arith.ori",           false};
  table["LogicalOr"]         = {"LogicalOr",         TritonOpCategory::LOGICAL, "arith.ori",           false};
  table["bool_not"]          = {"bool_not",          TritonOpCategory::LOGICAL, "custom.not",          true};
  table["boolean_not"]       = {"boolean_not",       TritonOpCategory::LOGICAL, "custom.not",          true};
  table["BooleanNot"]        = {"BooleanNot",        TritonOpCategory::LOGICAL, "custom.not",          true};
  table["logical_not"]       = {"logical_not",       TritonOpCategory::LOGICAL, "custom.not",          true};
  table["LogicalNot"]        = {"LogicalNot",        TritonOpCategory::LOGICAL, "custom.not",          true};
  table["boolean_xor"]       = {"boolean_xor",       TritonOpCategory::LOGICAL, "arith.xori",          false};
  table["BooleanXor"]        = {"BooleanXor",        TritonOpCategory::LOGICAL, "arith.xori",          false};

  // Select/where (ternary element-wise)
  table["where"]             = {"where",             TritonOpCategory::TERNARY, "arith.select",        false};
  table["Where"]             = {"Where",             TritonOpCategory::TERNARY, "arith.select",        false};
  table["select"]            = {"select",            TritonOpCategory::TERNARY, "arith.select",        false};
  table["Select"]            = {"Select",            TritonOpCategory::TERNARY, "arith.select",        false};

  // Identity/copy (SSA value forwarding)
  table["identity"]          = {"identity",          TritonOpCategory::IDENTITY, "identity",            false};
  table["Identity"]          = {"Identity",          TritonOpCategory::IDENTITY, "identity",            false};
  table["assign"]            = {"assign",            TritonOpCategory::IDENTITY, "identity",            false};
  table["Assign"]            = {"Assign",            TritonOpCategory::IDENTITY, "identity",            false};

  // Cast — reclassified from UNSUPPORTED to CAST for Triton IR fusion
  table["cast"]              = {"cast",              TritonOpCategory::CAST, "arith.cast",              false};
  table["Cast"]              = {"Cast",              TritonOpCategory::CAST, "arith.cast",              false};

  // Additional reduction ops
  table["reduce_norm1"]      = {"reduce_norm1",      TritonOpCategory::REDUCTION, "tt.reduce", true};
  table["ReduceNorm1"]       = {"ReduceNorm1",       TritonOpCategory::REDUCTION, "tt.reduce", true};
  table["reduce_norm2"]      = {"reduce_norm2",      TritonOpCategory::REDUCTION, "tt.reduce", true};
  table["ReduceNorm2"]       = {"ReduceNorm2",       TritonOpCategory::REDUCTION, "tt.reduce", true};
  table["reduce_logsumexp"]  = {"reduce_logsumexp",  TritonOpCategory::REDUCTION, "tt.reduce", true};
  table["ReduceLogSumExp"]   = {"ReduceLogSumExp",   TritonOpCategory::REDUCTION, "tt.reduce", true};
  table["reduce_variance"]   = {"reduce_variance",   TritonOpCategory::REDUCTION, "tt.reduce", true};
  table["ReduceVariance"]    = {"ReduceVariance",    TritonOpCategory::REDUCTION, "tt.reduce", true};
  table["reduce_stdev"]      = {"reduce_stdev",      TritonOpCategory::REDUCTION, "tt.reduce", true};
  table["ReduceStdev"]       = {"ReduceStdev",       TritonOpCategory::REDUCTION, "tt.reduce", true};
  table["sum"]               = {"sum",               TritonOpCategory::REDUCTION, "tt.reduce", false};
  table["Sum"]               = {"Sum",               TritonOpCategory::REDUCTION, "tt.reduce", false};
  table["mean"]              = {"mean",              TritonOpCategory::REDUCTION, "tt.reduce", true};
  table["Mean"]              = {"Mean",              TritonOpCategory::REDUCTION, "tt.reduce", true};
  table["max"]               = {"max",               TritonOpCategory::REDUCTION, "tt.reduce", false};
  table["min"]               = {"min",               TritonOpCategory::REDUCTION, "tt.reduce", false};
  table["prod"]              = {"prod",              TritonOpCategory::REDUCTION, "tt.reduce", false};
  table["Prod"]              = {"Prod",              TritonOpCategory::REDUCTION, "tt.reduce", false};
  table["norm1"]             = {"norm1",             TritonOpCategory::REDUCTION, "tt.reduce", true};
  table["norm2"]             = {"norm2",             TritonOpCategory::REDUCTION, "tt.reduce", true};
  table["normmax"]           = {"normmax",           TritonOpCategory::REDUCTION, "tt.reduce", false};
  table["argmax"]            = {"argmax",            TritonOpCategory::REDUCTION, "tt.reduce", true};
  table["Argmax"]            = {"Argmax",            TritonOpCategory::REDUCTION, "tt.reduce", true};
  table["argmin"]            = {"argmin",            TritonOpCategory::REDUCTION, "tt.reduce", true};
  table["Argmin"]            = {"Argmin",            TritonOpCategory::REDUCTION, "tt.reduce", true};

  // Additional normalization ops
  table["batch_norm"]        = {"batch_norm",        TritonOpCategory::NORMALIZATION, "tt.reduce", true};
  table["BatchNorm"]         = {"BatchNorm",         TritonOpCategory::NORMALIZATION, "tt.reduce", true};
  table["rms_norm"]          = {"rms_norm",          TritonOpCategory::NORMALIZATION, "tt.reduce", true};
  table["RmsNorm"]           = {"RmsNorm",           TritonOpCategory::NORMALIZATION, "tt.reduce", true};
  table["normalize_moments"] = {"normalize_moments", TritonOpCategory::NORMALIZATION, "tt.reduce", true};
  table["NormalizeMoments"]  = {"NormalizeMoments",  TritonOpCategory::NORMALIZATION, "tt.reduce", true};

  // Fused attention (Flash Attention pattern)
  table["onnx_multi_head_attention"]       = {"onnx_multi_head_attention",       TritonOpCategory::FUSED_ATTENTION, "custom.flash_attn", true};
  table["OnnxMultiHeadAttention"]          = {"OnnxMultiHeadAttention",          TritonOpCategory::FUSED_ATTENTION, "custom.flash_attn", true};
  table["multi_head_attention"]            = {"multi_head_attention",            TritonOpCategory::FUSED_ATTENTION, "custom.flash_attn", true};
  table["MultiHeadAttention"]              = {"MultiHeadAttention",              TritonOpCategory::FUSED_ATTENTION, "custom.flash_attn", true};
  table["dot_product_attention_v2"]        = {"dot_product_attention_v2",        TritonOpCategory::FUSED_ATTENTION, "custom.flash_attn", true};
  table["DotProductAttentionV2"]           = {"DotProductAttentionV2",           TritonOpCategory::FUSED_ATTENTION, "custom.flash_attn", true};

  // Matrix ops (additional)
  table["tensormmul"]        = {"tensormmul",        TritonOpCategory::MATMUL, "tt.dot", false};
  table["TensorMmul"]        = {"TensorMmul",        TritonOpCategory::MATMUL, "tt.dot", false};
  table["batched_gemm"]      = {"batched_gemm",      TritonOpCategory::MATMUL, "tt.dot", false};
  table["BatchedGemm"]       = {"BatchedGemm",       TritonOpCategory::MATMUL, "tt.dot", false};
  table["xw_plus_b"]         = {"xw_plus_b",         TritonOpCategory::MATMUL, "tt.dot", true};
  table["XwPlusB"]           = {"XwPlusB",            TritonOpCategory::MATMUL, "tt.dot", true};

  // Shape manipulation ops (zero-cost views / stride reinterpretation)
  table["reshape"]           = {"reshape",           TritonOpCategory::SHAPE_MANIPULATION, "view", false};
  table["Reshape"]           = {"Reshape",           TritonOpCategory::SHAPE_MANIPULATION, "view", false};
  table["permute"]           = {"permute",           TritonOpCategory::SHAPE_MANIPULATION, "view", false};
  table["Permute"]           = {"Permute",           TritonOpCategory::SHAPE_MANIPULATION, "view", false};
  table["expand_dims"]       = {"expand_dims",       TritonOpCategory::SHAPE_MANIPULATION, "view", false};
  table["ExpandDims"]        = {"ExpandDims",        TritonOpCategory::SHAPE_MANIPULATION, "view", false};
  table["squeeze"]           = {"squeeze",           TritonOpCategory::SHAPE_MANIPULATION, "view", false};
  table["Squeeze"]           = {"Squeeze",           TritonOpCategory::SHAPE_MANIPULATION, "view", false};

  // Data movement ops (actual data copies / indexed reads)
  table["gather"]            = {"gather",            TritonOpCategory::DATA_MOVEMENT, "tt.load", true};
  table["Gather"]            = {"Gather",            TritonOpCategory::DATA_MOVEMENT, "tt.load", true};
  table["concat"]            = {"concat",            TritonOpCategory::DATA_MOVEMENT, "tt.store", true};
  table["Concat"]            = {"Concat",            TritonOpCategory::DATA_MOVEMENT, "tt.store", true};
  table["split"]             = {"split",             TritonOpCategory::DATA_MOVEMENT, "tt.load", true};
  table["Split"]             = {"Split",             TritonOpCategory::DATA_MOVEMENT, "tt.load", true};
  table["stack"]             = {"stack",             TritonOpCategory::DATA_MOVEMENT, "tt.store", true};
  table["Stack"]             = {"Stack",             TritonOpCategory::DATA_MOVEMENT, "tt.store", true};
  table["strided_slice"]     = {"strided_slice",     TritonOpCategory::DATA_MOVEMENT, "tt.load", true};
  table["StridedSlice"]      = {"StridedSlice",      TritonOpCategory::DATA_MOVEMENT, "tt.load", true};
  table["tile"]              = {"tile",              TritonOpCategory::DATA_MOVEMENT, "tt.store", true};
  table["Tile"]              = {"Tile",              TritonOpCategory::DATA_MOVEMENT, "tt.store", true};

  // Constant generation ops (produce fixed values from shape/metadata)
  table["shape_of"]          = {"shape_of",          TritonOpCategory::CONSTANT_GENERATION, "arith.constant", false};
  table["ShapeOf"]           = {"ShapeOf",           TritonOpCategory::CONSTANT_GENERATION, "arith.constant", false};
  table["create"]            = {"create",            TritonOpCategory::CONSTANT_GENERATION, "arith.constant", false};
  table["Create"]            = {"Create",            TritonOpCategory::CONSTANT_GENERATION, "arith.constant", false};
  table["set_scalar"]        = {"set_scalar",        TritonOpCategory::CONSTANT_GENERATION, "arith.constant", false};
  table["SetScalar"]         = {"SetScalar",         TritonOpCategory::CONSTANT_GENERATION, "arith.constant", false};
  table["ones_as"]           = {"ones_as",           TritonOpCategory::CONSTANT_GENERATION, "arith.constant", false};
  table["OnesAs"]            = {"OnesAs",            TritonOpCategory::CONSTANT_GENERATION, "arith.constant", false};
  table["ones_like"]         = {"ones_like",         TritonOpCategory::CONSTANT_GENERATION, "arith.constant", false};
  table["oneslike"]          = {"oneslike",          TritonOpCategory::CONSTANT_GENERATION, "arith.constant", false};
  table["zeros_like"]        = {"zeros_like",        TritonOpCategory::CONSTANT_GENERATION, "arith.constant", false};
  table["zeroslike"]         = {"zeroslike",         TritonOpCategory::CONSTANT_GENERATION, "arith.constant", false};
  table["ZerosLike"]         = {"ZerosLike",         TritonOpCategory::CONSTANT_GENERATION, "arith.constant", false};
  table["zeros_as"]          = {"zeros_as",          TritonOpCategory::CONSTANT_GENERATION, "arith.constant", false};
  table["min_max_datatype"]  = {"min_max_datatype",  TritonOpCategory::CONSTANT_GENERATION, "arith.constant", false};
  table["MinMaxDatatype"]    = {"MinMaxDatatype",    TritonOpCategory::CONSTANT_GENERATION, "arith.constant", false};
  table["range"]             = {"range",             TritonOpCategory::CONSTANT_GENERATION, "tt.make_range", false};
  table["Range"]             = {"Range",             TritonOpCategory::CONSTANT_GENERATION, "tt.make_range", false};

  // Shape manipulation ops — additional entries
  table["flatten"]           = {"flatten",           TritonOpCategory::SHAPE_MANIPULATION, "view", false};
  table["Flatten"]           = {"Flatten",           TritonOpCategory::SHAPE_MANIPULATION, "view", false};
  table["flatten_2d"]        = {"flatten_2d",        TritonOpCategory::SHAPE_MANIPULATION, "view", false};
  table["Flatten2d"]         = {"Flatten2d",         TritonOpCategory::SHAPE_MANIPULATION, "view", false};

  // Data movement ops — additional entries
  table["gather_nd"]         = {"gather_nd",         TritonOpCategory::DATA_MOVEMENT, "tt.load", true};
  table["GatherNd"]          = {"GatherNd",          TritonOpCategory::DATA_MOVEMENT, "tt.load", true};
  table["scatter_nd"]        = {"scatter_nd",        TritonOpCategory::DATA_MOVEMENT, "tt.store", true};
  table["ScatterNd"]         = {"ScatterNd",         TritonOpCategory::DATA_MOVEMENT, "tt.store", true};
  table["scatter_nd_update"] = {"scatter_nd_update", TritonOpCategory::DATA_MOVEMENT, "tt.store", true};
  table["ScatterNdUpdate"]   = {"ScatterNdUpdate",   TritonOpCategory::DATA_MOVEMENT, "tt.store", true};
  table["split_v"]           = {"split_v",           TritonOpCategory::DATA_MOVEMENT, "tt.load", true};
  table["SplitV"]            = {"SplitV",            TritonOpCategory::DATA_MOVEMENT, "tt.load", true};

  // Convolution ops
  table["conv2d"]            = {"conv2d",            TritonOpCategory::CONVOLUTION, "custom.conv2d", true};
  table["Conv2d"]            = {"Conv2d",            TritonOpCategory::CONVOLUTION, "custom.conv2d", true};
  table["conv2D"]            = {"conv2D",            TritonOpCategory::CONVOLUTION, "custom.conv2d", true};
  table["conv3d"]            = {"conv3d",            TritonOpCategory::CONVOLUTION, "custom.conv3d", true};
  table["Conv3d"]            = {"Conv3d",            TritonOpCategory::CONVOLUTION, "custom.conv3d", true};
  table["depthwise_conv2d"]  = {"depthwise_conv2d",  TritonOpCategory::CONVOLUTION, "custom.dw_conv2d", true};
  table["DepthwiseConv2d"]   = {"DepthwiseConv2d",   TritonOpCategory::CONVOLUTION, "custom.dw_conv2d", true};

  // im2col / col2im (convolution helpers)
  table["im2col"]            = {"im2col",            TritonOpCategory::CONVOLUTION, "custom.im2col", true};
  table["Im2col"]            = {"Im2col",            TritonOpCategory::CONVOLUTION, "custom.im2col", true};
  table["im2col_bp"]         = {"im2col_bp",         TritonOpCategory::CONVOLUTION, "custom.im2col_bp", true};
  table["col2im"]            = {"col2im",            TritonOpCategory::CONVOLUTION, "custom.col2im", true};
  table["Col2im"]            = {"Col2im",            TritonOpCategory::CONVOLUTION, "custom.col2im", true};
  table["col2im_bp"]         = {"col2im_bp",         TritonOpCategory::CONVOLUTION, "custom.col2im_bp", true};

  return table;
}

const std::unordered_map<std::string, TritonOpMapping>& TritonIRBuilder::getOpTable() {
  static auto table = buildOpTable();
  return table;
}

// ─── Public API ─────────────────────────────────────────────────────────────

TritonIRBuilder::TritonIRBuilder() = default;
TritonIRBuilder::~TritonIRBuilder() = default;

bool TritonIRBuilder::isTritonMappable(const std::string& opName) {
  const auto& table = getOpTable();
  auto it = table.find(opName);
  if (it == table.end()) {
    std::string msg = "TritonIRBuilder::isTritonMappable: op '" + opName + "' is missing from buildOpTable(). "
                      "Every op MUST be manually categorized in the table. Add it now.";
    THROW_EXCEPTION(msg.c_str());
  }
  return true;
}

TritonOpCategory TritonIRBuilder::getOpCategory(const std::string& opName) {
  const auto& table = getOpTable();
  auto it = table.find(opName);
  if (it == table.end()) {
    std::string msg = "TritonIRBuilder::getOpCategory: op '" + opName + "' is missing from buildOpTable(). "
                      "Every op MUST be manually categorized in the table. Add it now.";
    THROW_EXCEPTION(msg.c_str());
  }
  return it->second.category;
}

bool TritonIRBuilder::isElementwiseCompatible(TritonOpCategory cat) {
  return sd::graph::isElementwiseCompatible(cat);
}

// ─── Pass 1: Segment Profiling ──────────────────────────────────────────────

SegmentProfile TritonIRBuilder::profileSegment(NativeSlot* slots, int startSlot, int endSlot,
                                                NDArray** outputSlots, int totalOutputSlots) {
  SegmentProfile profile;
  int segSize = endSlot - startSlot + 1;
  profile.totalOps = segSize;
  profile.nodes.resize(segSize);

  // Build slotIndex → localIndex map and outputSlot → producer local index map
  std::unordered_map<int, int> slotToLocal;
  std::unordered_map<int, int> outputSlotToProducer;  // output slot idx → local index that produces it

  for (int i = 0; i < segSize; i++) {
    int absSlot = startSlot + i;
    slotToLocal[absSlot] = i;

    auto& node = profile.nodes[i];
    node.slotIndex = absSlot;
    node.localIndex = i;
    node.opName = slots[absSlot].opName;
    node.category = getOpCategory(slots[absSlot].opName);
    node.hasExternalInput = false;

    // Register outputs produced by this node
    for (int o = 0; o < slots[absSlot].numOutputs; o++) {
      outputSlotToProducer[slots[absSlot].outputSlotIndices[o]] = i;
    }

    // Populate output shape from DSP's pre-calculated cache or live outputSlots
    if (slots[absSlot].numOutputs > 0) {
      int outIdx = slots[absSlot].outputSlotIndices[0];

      // Priority 1: Use NativeSlot's cached shape info (pre-calculated by DSP)
      if (slots[absSlot].shapeCacheValid && !slots[absSlot].cachedOutputShapes.empty()) {
        const LongType* shapeInfo = slots[absSlot].cachedOutputShapes[0];
        if (shapeInfo) {
          LongType rank = shape::rank(shapeInfo);
          node.outputShape.resize(rank);
          for (int d = 0; d < rank; d++) {
            node.outputShape[d] = shapeInfo[d + 1];
          }
          node.hasOutputShape = true;
        }
      }

      // Priority 2: Fall back to live outputSlots array
      if (!node.hasOutputShape && outputSlots && outIdx >= 0 && outIdx < totalOutputSlots && outputSlots[outIdx]) {
        auto& arr = *outputSlots[outIdx];
        node.outputShape.resize(arr.rankOf());
        for (int d = 0; d < arr.rankOf(); d++) {
          node.outputShape[d] = arr.sizeAt(d);
        }
        node.outputDtype = arr.dataType();
        node.hasOutputShape = true;
      }
    }

    // Count categories
    int catIdx = static_cast<int>(node.category);
    if (catIdx >= 0 && catIdx < 16) profile.categoryCounts[catIdx]++;
  }

  // Build dataflow edges and consumer lists
  std::unordered_set<int> externalInputSet;
  std::unordered_map<int, std::vector<int>> outputToConsumers;  // output slot → list of consuming local indices

  for (int i = 0; i < segSize; i++) {
    int absSlot = startSlot + i;
    auto& node = profile.nodes[i];

    for (int inp = 0; inp < slots[absSlot].numInputs; inp++) {
      int srcIdx = slots[absSlot].inputSourceIndices[inp];

      if (srcIdx < 0) {
        // External input
        node.inputLocalIndices.push_back(-1);
        node.hasExternalInput = true;
        externalInputSet.insert(srcIdx);
      } else {
        // Check if this source is produced within the segment
        auto producerIt = outputSlotToProducer.find(srcIdx);
        if (producerIt != outputSlotToProducer.end()) {
          int producerLocal = producerIt->second;
          node.inputLocalIndices.push_back(producerLocal);
          outputToConsumers[srcIdx].push_back(i);
        } else {
          // Pre-segment output — treat as external
          node.inputLocalIndices.push_back(-1);
          node.hasExternalInput = true;
          externalInputSet.insert(srcIdx);
        }
      }
    }
  }

  // Fill consumer lists
  for (int i = 0; i < segSize; i++) {
    int absSlot = startSlot + i;
    for (int o = 0; o < slots[absSlot].numOutputs; o++) {
      int outIdx = slots[absSlot].outputSlotIndices[o];
      auto it = outputToConsumers.find(outIdx);
      if (it != outputToConsumers.end()) {
        for (int consumer : it->second) {
          profile.nodes[i].consumerLocalIndices.push_back(consumer);
        }
      }
    }
  }

  // Count unique outputs (produced within segment)
  std::unordered_set<int> outputSet;
  for (int i = 0; i < segSize; i++) {
    int absSlot = startSlot + i;
    for (int o = 0; o < slots[absSlot].numOutputs; o++) {
      outputSet.insert(slots[absSlot].outputSlotIndices[o]);
    }
  }

  profile.numUniqueExternalInputs = static_cast<int>(externalInputSet.size());
  profile.numUniqueOutputs = static_cast<int>(outputSet.size());

  // Set summary flags from category counts
  profile.hasMatmul = profile.categoryCounts[static_cast<int>(TritonOpCategory::MATMUL)] > 0;
  profile.hasReduction = profile.categoryCounts[static_cast<int>(TritonOpCategory::REDUCTION)] > 0;
  profile.hasNormalization = profile.categoryCounts[static_cast<int>(TritonOpCategory::NORMALIZATION)] > 0;
  profile.hasFusedAttention = profile.categoryCounts[static_cast<int>(TritonOpCategory::FUSED_ATTENTION)] > 0;
  profile.hasShapeManip = profile.categoryCounts[static_cast<int>(TritonOpCategory::SHAPE_MANIPULATION)] > 0;
  profile.hasDataMovement = profile.categoryCounts[static_cast<int>(TritonOpCategory::DATA_MOVEMENT)] > 0;
  // No UNSUPPORTED category — getOpCategory() throws if any op is missing from the table.

  return profile;
}

// ─── Pass 2: Pattern Detection ──────────────────────────────────────────────

namespace {

// --- Concrete pattern detectors (file-local) ---

class FusedAttentionOpDetector : public PatternDetector {
 public:
  const char* name() const override { return "FusedAttentionOp"; }
  std::vector<PatternMatch> detect(const SegmentProfile& profile,
                                    NativeSlot* slots, int startSlot) override {
    std::vector<PatternMatch> results;
    if (!profile.hasFusedAttention) return results;
    for (auto& node : profile.nodes) {
      if (node.category == TritonOpCategory::FUSED_ATTENTION) {
        PatternMatch m;
        m.type = PatternMatch::FUSED_ATTENTION_OP;
        m.priority = 100;
        m.localIndices.push_back(node.localIndex);
        m.description = "fused attention op at slot " + std::to_string(node.slotIndex);
        results.push_back(m);
      }
    }
    return results;
  }
};

class AttentionPatternDetector : public PatternDetector {
 public:
  const char* name() const override { return "AttentionQKV"; }
  std::vector<PatternMatch> detect(const SegmentProfile& profile,
                                    NativeSlot* slots, int startSlot) override {
    std::vector<PatternMatch> results;
    int matmulCount = profile.categoryCounts[static_cast<int>(TritonOpCategory::MATMUL)];
    if (matmulCount < 2 || (!profile.hasNormalization && !profile.hasReduction)) return results;

    // Find pairs: matmul → (elementwise chain) → softmax/reduction → (elementwise chain) → matmul
    std::vector<int> matmulLocals;
    for (auto& node : profile.nodes) {
      if (node.category == TritonOpCategory::MATMUL) {
        matmulLocals.push_back(node.localIndex);
      }
    }

    for (size_t mi = 0; mi + 1 < matmulLocals.size(); mi++) {
      int firstMatmul = matmulLocals[mi];
      int secondMatmul = matmulLocals[mi + 1];
      // Check for softmax/reduction between the two matmuls
      bool hasSoftmaxBetween = false;
      for (int j = firstMatmul + 1; j < secondMatmul; j++) {
        auto cat = profile.nodes[j].category;
        if (cat == TritonOpCategory::NORMALIZATION || cat == TritonOpCategory::REDUCTION) {
          hasSoftmaxBetween = true;
          break;
        }
      }
      if (hasSoftmaxBetween) {
        PatternMatch m;
        m.type = PatternMatch::ATTENTION_QKV;
        m.priority = 90;
        for (int j = firstMatmul; j <= secondMatmul; j++) {
          m.localIndices.push_back(j);
        }
        m.description = "attention pattern: matmul[" + std::to_string(profile.nodes[firstMatmul].slotIndex) +
                         "] → softmax → matmul[" + std::to_string(profile.nodes[secondMatmul].slotIndex) + "]";
        results.push_back(m);
      }
    }
    return results;
  }
};

class FFNBlockDetector : public PatternDetector {
 public:
  const char* name() const override { return "FFNBlock"; }
  std::vector<PatternMatch> detect(const SegmentProfile& profile,
                                    NativeSlot* slots, int startSlot) override {
    std::vector<PatternMatch> results;
    int matmulCount = profile.categoryCounts[static_cast<int>(TritonOpCategory::MATMUL)];
    if (matmulCount < 2) return results;

    // Find pairs: matmul → activation (elementwise) → matmul (with no reduction/norm between)
    std::vector<int> matmulLocals;
    for (auto& node : profile.nodes) {
      if (node.category == TritonOpCategory::MATMUL) {
        matmulLocals.push_back(node.localIndex);
      }
    }

    for (size_t mi = 0; mi + 1 < matmulLocals.size(); mi++) {
      int firstMatmul = matmulLocals[mi];
      int secondMatmul = matmulLocals[mi + 1];
      bool hasActivation = false;
      bool hasHeavyweight = false;
      for (int j = firstMatmul + 1; j < secondMatmul; j++) {
        auto cat = profile.nodes[j].category;
        if (TritonIRBuilder::isElementwiseCompatible(cat)) hasActivation = true;
        if (cat == TritonOpCategory::REDUCTION || cat == TritonOpCategory::NORMALIZATION) {
          hasHeavyweight = true;
        }
      }
      if (hasActivation && !hasHeavyweight) {
        PatternMatch m;
        m.type = PatternMatch::FFN_BLOCK;
        m.priority = 85;
        for (int j = firstMatmul; j <= secondMatmul; j++) {
          m.localIndices.push_back(j);
        }
        m.description = "FFN block: matmul[" + std::to_string(profile.nodes[firstMatmul].slotIndex) +
                         "] → activation → matmul[" + std::to_string(profile.nodes[secondMatmul].slotIndex) + "]";
        results.push_back(m);
      }
    }
    return results;
  }
};

class DecomposedSoftmaxDetector : public PatternDetector {
 public:
  const char* name() const override { return "DecomposedSoftmax"; }
  std::vector<PatternMatch> detect(const SegmentProfile& profile,
                                    NativeSlot* slots, int startSlot) override {
    std::vector<PatternMatch> results;
    if (!profile.hasReduction) return results;

    // Mirror FusionPass Pass 6: reduce_max → sub → exp → reduce_sum → div
    for (int i = 0; i < profile.totalOps; i++) {
      if (profile.nodes[i].opName != "reduce_max" && profile.nodes[i].opName != "ReduceMax") continue;

      int absI = startSlot + i;
      if (slots[absI].numOutputs != 1) continue;
      int out0 = slots[absI].outputSlotIndices[0];

      // Find sub consuming reduce_max output
      int subLocal = -1;
      for (int j = i + 1; j < profile.totalOps; j++) {
        auto& name = profile.nodes[j].opName;
        if (name != "subtract" && name != "Sub") continue;
        int absJ = startSlot + j;
        for (int k = 0; k < slots[absJ].numInputs; k++) {
          if (slots[absJ].inputSourceIndices[k] == out0) { subLocal = j; break; }
        }
        if (subLocal >= 0) break;
      }
      if (subLocal < 0) continue;

      int outSub = slots[startSlot + subLocal].outputSlotIndices[0];
      // Find exp consuming sub output
      int expLocal = -1;
      for (int j = subLocal + 1; j < profile.totalOps; j++) {
        auto& name = profile.nodes[j].opName;
        if (name != "exp" && name != "Exp") continue;
        int absJ = startSlot + j;
        if (slots[absJ].numInputs >= 1 && slots[absJ].inputSourceIndices[0] == outSub) {
          expLocal = j; break;
        }
      }
      if (expLocal < 0) continue;

      int outExp = slots[startSlot + expLocal].outputSlotIndices[0];
      // Find reduce_sum consuming exp output
      int sumLocal = -1;
      for (int j = expLocal + 1; j < profile.totalOps; j++) {
        auto& name = profile.nodes[j].opName;
        if (name != "reduce_sum" && name != "ReduceSum") continue;
        int absJ = startSlot + j;
        if (slots[absJ].numInputs >= 1 && slots[absJ].inputSourceIndices[0] == outExp) {
          sumLocal = j; break;
        }
      }
      if (sumLocal < 0) continue;

      int outSum = slots[startSlot + sumLocal].outputSlotIndices[0];
      // Find div consuming exp and sum outputs
      int divLocal = -1;
      for (int j = sumLocal + 1; j < profile.totalOps; j++) {
        auto& name = profile.nodes[j].opName;
        if (name != "divide" && name != "Div" && name != "RealDiv") continue;
        int absJ = startSlot + j;
        bool hasExp = false, hasSum = false;
        for (int k = 0; k < slots[absJ].numInputs; k++) {
          if (slots[absJ].inputSourceIndices[k] == outExp) hasExp = true;
          if (slots[absJ].inputSourceIndices[k] == outSum) hasSum = true;
        }
        if (hasExp && hasSum) { divLocal = j; break; }
      }
      if (divLocal < 0) continue;

      PatternMatch m;
      m.type = PatternMatch::SOFTMAX_DECOMPOSED;
      m.priority = 80;
      m.localIndices = {i, subLocal, expLocal, sumLocal, divLocal};
      m.description = "decomposed softmax: reduce_max[" + std::to_string(startSlot + i) +
                       "] → sub → exp → reduce_sum → div[" + std::to_string(startSlot + divLocal) + "]";
      results.push_back(m);
    }
    return results;
  }
};

class MatmulEpilogueDetector : public PatternDetector {
 public:
  const char* name() const override { return "MatmulEpilogue"; }
  std::vector<PatternMatch> detect(const SegmentProfile& profile,
                                    NativeSlot* slots, int startSlot) override {
    std::vector<PatternMatch> results;
    if (!profile.hasMatmul) return results;

    for (int i = 0; i < profile.totalOps; i++) {
      if (profile.nodes[i].category != TritonOpCategory::MATMUL) continue;
      // BFS forward through elementwise-compatible consumers
      std::vector<int> epilogueOps = {i};
      for (int j = i + 1; j < profile.totalOps; j++) {
        if (profile.nodes[j].category == TritonOpCategory::MATMUL) break;
        if (TritonIRBuilder::isElementwiseCompatible(profile.nodes[j].category)) {
          epilogueOps.push_back(j);
        } else {
          break;  // Stop at non-elementwise
        }
      }
      if (epilogueOps.size() > 1) {
        PatternMatch m;
        m.type = PatternMatch::MATMUL_EPILOGUE;
        m.priority = 70;
        m.localIndices = epilogueOps;
        m.description = "matmul epilogue: matmul[" + std::to_string(startSlot + i) +
                         "] + " + std::to_string(epilogueOps.size() - 1) + " elementwise ops";
        results.push_back(m);
      }
    }
    return results;
  }
};

class ElementwisePatternDetector : public PatternDetector {
 public:
  const char* name() const override { return "PureElementwise"; }
  std::vector<PatternMatch> detect(const SegmentProfile& profile,
                                    NativeSlot* slots, int startSlot) override {
    std::vector<PatternMatch> results;
    bool allElementwise = true;
    for (auto& node : profile.nodes) {
      if (!TritonIRBuilder::isElementwiseCompatible(node.category)) {
        allElementwise = false;
        break;
      }
    }
    if (allElementwise && profile.totalOps > 0) {
      PatternMatch m;
      m.type = PatternMatch::PURE_ELEMENTWISE;
      m.priority = 10;
      for (int i = 0; i < profile.totalOps; i++) m.localIndices.push_back(i);
      m.description = "pure elementwise chain (" + std::to_string(profile.totalOps) + " ops)";
      results.push_back(m);
    }
    return results;
  }
};

class MegaSegmentDetector : public PatternDetector {
 public:
  const char* name() const override { return "MegaSegment"; }
  std::vector<PatternMatch> detect(const SegmentProfile& profile,
                                    NativeSlot* slots, int startSlot) override {
    std::vector<PatternMatch> results;
    if (profile.totalOps <= 50) return results;

    // Count heavyweight categories present
    int heavyweightCount = 0;
    if (profile.hasMatmul) heavyweightCount++;
    if (profile.hasReduction) heavyweightCount++;
    if (profile.hasNormalization) heavyweightCount++;
    if (profile.hasFusedAttention) heavyweightCount++;
    if (profile.hasShapeManip) heavyweightCount++;
    if (profile.hasDataMovement) heavyweightCount++;

    if (heavyweightCount >= 2) {
      PatternMatch m;
      m.type = PatternMatch::MIXED_MEGA_SEGMENT;
      m.priority = 5;
      for (int i = 0; i < profile.totalOps; i++) m.localIndices.push_back(i);
      m.description = "mixed mega-segment (" + std::to_string(profile.totalOps) + " ops, " +
                       std::to_string(heavyweightCount) + " heavyweight categories)";
      results.push_back(m);
    }
    return results;
  }
};

// Singleton registry of pattern detectors
class PatternRegistry {
 public:
  static PatternRegistry& instance() {
    static PatternRegistry reg;
    return reg;
  }

  std::vector<PatternDetector*>& detectors() { return detectors_; }

 private:
  PatternRegistry() {
    // Order doesn't matter — matches are ranked by priority
    detectors_.push_back(new FusedAttentionOpDetector());
    detectors_.push_back(new AttentionPatternDetector());
    detectors_.push_back(new FFNBlockDetector());
    detectors_.push_back(new DecomposedSoftmaxDetector());
    detectors_.push_back(new MatmulEpilogueDetector());
    detectors_.push_back(new ElementwisePatternDetector());
    detectors_.push_back(new MegaSegmentDetector());
  }

  ~PatternRegistry() {
    for (auto* d : detectors_) delete d;
  }

  std::vector<PatternDetector*> detectors_;
};

}  // anonymous namespace

MatchedPatterns TritonIRBuilder::matchPatterns(const SegmentProfile& profile,
                                                NativeSlot* slots, int startSlot) {
  MatchedPatterns matched;
  auto& detectors = PatternRegistry::instance().detectors();

  for (auto* detector : detectors) {
    auto hits = detector->detect(profile, slots, startSlot);
    for (auto& hit : hits) {
      matched.matches.push_back(std::move(hit));
    }
  }

  // Sort by priority (descending)
  std::sort(matched.matches.begin(), matched.matches.end(),
            [](const PatternMatch& a, const PatternMatch& b) { return a.priority > b.priority; });

  return matched;
}

// ─── Helper: compute set of output slots that are externally visible ─────────
// An output needs a kernel arg (global memory store) only if it's consumed
// outside [startSlot, endSlot] or is a final requested graph output.
// Purely internal intermediates (produced and consumed entirely within the
// segment) are SSA-forwarded in the kernel and need no kernel arg.
static std::unordered_set<int> computeExternallyVisibleOutputs(
    NativeSlot* slots, int startSlot, int endSlot, int totalSlots,
    int* requestedOutputSlotIndices, int numRequestedOutputs) {

  // 1. Collect all output slot indices produced within the segment
  std::unordered_set<int> segmentOutputs;
  for (int i = startSlot; i <= endSlot; i++) {
    for (int o = 0; o < slots[i].numOutputs; o++) {
      segmentOutputs.insert(slots[i].outputSlotIndices[o]);
    }
  }

  // 2. Find outputs consumed by slots OUTSIDE [startSlot, endSlot]
  std::unordered_set<int> externallyConsumed;
  for (int i = 0; i < totalSlots; i++) {
    if (i >= startSlot && i <= endSlot) continue;  // Skip slots within segment
    for (int inp = 0; inp < slots[i].numInputs; inp++) {
      int srcIdx = slots[i].inputSourceIndices[inp];
      if (srcIdx >= 0 && segmentOutputs.count(srcIdx)) {
        externallyConsumed.insert(srcIdx);
      }
    }
  }

  // 3. Add all requested/final graph outputs
  for (int r = 0; r < numRequestedOutputs; r++) {
    int reqSlot = requestedOutputSlotIndices[r];
    if (segmentOutputs.count(reqSlot)) {
      externallyConsumed.insert(reqSlot);
    }
  }

  // 4. Add outputs NOT consumed by ANY slot (neither internal nor external).
  //    These might be side-effect outputs or terminal values.
  std::unordered_set<int> consumedAnywhere;
  for (int i = 0; i < totalSlots; i++) {
    for (int inp = 0; inp < slots[i].numInputs; inp++) {
      int srcIdx = slots[i].inputSourceIndices[inp];
      if (srcIdx >= 0) consumedAnywhere.insert(srcIdx);
    }
  }
  for (int outIdx : segmentOutputs) {
    if (!consumedAnywhere.count(outIdx)) {
      // Not consumed by anything — could be a final output or side-effect
      externallyConsumed.insert(outIdx);
    }
  }

  return externallyConsumed;
}

// ─── Pass 3: Classify and Analyze ───────────────────────────────────────────

SegmentAnalysis TritonIRBuilder::classifyAndAnalyze(const SegmentProfile& profile,
                                                     const MatchedPatterns& patterns,
                                                     NativeSlot* slots, int startSlot, int endSlot,
                                                     int totalSlots,
                                                     NDArray** externalInputs, int numExternalInputs,
                                                     NDArray** outputSlots, int totalOutputSlots,
                                                     int* requestedOutputSlotIndices,
                                                     int numRequestedOutputs) {
  SegmentAnalysis analysis;

  // Fill category counts from profile
  analysis.numElementwise = profile.categoryCounts[static_cast<int>(TritonOpCategory::BINARY_ELEMENTWISE)] +
                             profile.categoryCounts[static_cast<int>(TritonOpCategory::UNARY_ELEMENTWISE)];
  analysis.numMatmul = profile.categoryCounts[static_cast<int>(TritonOpCategory::MATMUL)];
  analysis.numReduction = profile.categoryCounts[static_cast<int>(TritonOpCategory::REDUCTION)];
  analysis.numNormalization = profile.categoryCounts[static_cast<int>(TritonOpCategory::NORMALIZATION)];
  analysis.numAttention = profile.categoryCounts[static_cast<int>(TritonOpCategory::FUSED_ATTENTION)];
  analysis.numShapeManip = profile.categoryCounts[static_cast<int>(TritonOpCategory::SHAPE_MANIPULATION)];
  analysis.numDataMovement = profile.categoryCounts[static_cast<int>(TritonOpCategory::DATA_MOVEMENT)];
  analysis.numConstGen = profile.categoryCounts[static_cast<int>(TritonOpCategory::CONSTANT_GENERATION)];
  analysis.numIdentity = profile.categoryCounts[static_cast<int>(TritonOpCategory::IDENTITY)];
  analysis.numCast = profile.categoryCounts[static_cast<int>(TritonOpCategory::CAST)];
  // No UNSUPPORTED category — getOpCategory() throws if any op is missing from the table.

  // Count unique input/output args (same logic as buildModule lines 2036-2099, but no MLIR)
  std::unordered_set<int> internalSlotOutputs;
  for (int i = startSlot; i <= endSlot; i++) {
    for (int o = 0; o < slots[i].numOutputs; o++) {
      internalSlotOutputs.insert(slots[i].outputSlotIndices[o]);
    }
  }

  std::unordered_set<int> seenInputs;
  int inputArgCount = 0;
  for (int i = startSlot; i <= endSlot; i++) {
    for (int inp = 0; inp < slots[i].numInputs; inp++) {
      int srcIdx = slots[i].inputSourceIndices[inp];
      if (seenInputs.count(srcIdx)) continue;
      seenInputs.insert(srcIdx);

      if (srcIdx < 0) {
        int extIdx = -(srcIdx + 1);
        if (extIdx < numExternalInputs && externalInputs && externalInputs[extIdx]) {
          inputArgCount++;
        }
      } else if (!internalSlotOutputs.count(srcIdx)) {
        if (srcIdx < totalOutputSlots && outputSlots && outputSlots[srcIdx]) {
          inputArgCount++;
        }
      }
    }
  }

  // Compute externally-visible outputs: only these need kernel args.
  // Purely internal intermediates (produced and consumed entirely within the
  // segment) are SSA-forwarded in the kernel — no global memory store needed.
  auto externalOutputs = computeExternallyVisibleOutputs(
      slots, startSlot, endSlot, totalSlots,
      requestedOutputSlotIndices, numRequestedOutputs);

  int outputArgCount = 0;
  int skippedInternalOutputs = 0;
  std::unordered_set<int> seenOutputs;
  for (int i = startSlot; i <= endSlot; i++) {
    for (int o = 0; o < slots[i].numOutputs; o++) {
      int outIdx = slots[i].outputSlotIndices[o];
      if (outIdx < 0 || outIdx >= totalOutputSlots) continue;
      if (seenOutputs.count(outIdx)) continue;  // Deduplicate
      seenOutputs.insert(outIdx);
      if (!externalOutputs.count(outIdx)) {
        skippedInternalOutputs++;
        continue;  // Purely internal — SSA forwarded, no kernel arg needed
      }
      outputArgCount++;
    }
  }
  if (skippedInternalOutputs > 0) {
    sd_printf("TritonIRBuilder::classifyAndAnalyze: eliminated %d internal intermediate outputs "
              "(keeping %d externally-visible output args)\n",
              skippedInternalOutputs, outputArgCount);
  }

  analysis.totalInputArgs = inputArgCount;
  analysis.totalOutputArgs = outputArgCount;
  analysis.totalArgs = inputArgCount + outputArgCount + 1;  // +1 for n_elements

  // Map best pattern type to SegmentKernelPattern
  const PatternMatch* best = patterns.bestMatch();
  if (best) {
    switch (best->type) {
      case PatternMatch::FUSED_ATTENTION_OP:
        analysis.pattern = SegmentKernelPattern::FUSED_ATTENTION;
        break;
      case PatternMatch::ATTENTION_QKV:
        analysis.pattern = SegmentKernelPattern::WHOLE_GRAPH;
        break;
      case PatternMatch::FFN_BLOCK:
        analysis.pattern = SegmentKernelPattern::WHOLE_GRAPH;
        break;
      case PatternMatch::SOFTMAX_DECOMPOSED:
        analysis.pattern = SegmentKernelPattern::NORMALIZATION;
        break;
      case PatternMatch::MATMUL_EPILOGUE:
        analysis.pattern = SegmentKernelPattern::MATMUL_EPILOGUE;
        break;
      case PatternMatch::PURE_MATMUL:
        analysis.pattern = SegmentKernelPattern::MATMUL_2D;
        break;
      case PatternMatch::PURE_REDUCTION:
        analysis.pattern = SegmentKernelPattern::REDUCTION_1D;
        break;
      case PatternMatch::PURE_NORMALIZATION:
        analysis.pattern = SegmentKernelPattern::NORMALIZATION;
        break;
      case PatternMatch::MIXED_MEGA_SEGMENT:
        analysis.pattern = SegmentKernelPattern::WHOLE_GRAPH;
        break;
      case PatternMatch::PURE_ELEMENTWISE:
      default:
        analysis.pattern = SegmentKernelPattern::ELEMENTWISE_1D;
        break;
    }
  } else {
    // No pattern matched — check if all ops are elementwise-compatible
    bool allEw = true;
    for (auto& node : profile.nodes) {
      if (!isElementwiseCompatible(node.category)) { allEw = false; break; }
    }
    analysis.pattern = allEw ? SegmentKernelPattern::ELEMENTWISE_1D
                             : SegmentKernelPattern::WHOLE_GRAPH;
  }

  // Validate feasibility — reject ops with known-buggy Triton IR emitters.
  {
    analysis.canCompile = true;

    // scatter_nd / scatter_nd_update: now properly handles multi-dimensional
    // scatter indexing with correct sliceSize decomposition and bounds checking.
    // With output dedup and indirect argument passing (pointer array), we can handle
    // segments with many unique buffers. The LLVM function arg limit of ~250 is avoided
    // by packing all buffer pointers into a single global memory array when the count
    // exceeds TRITON_DIRECT_ARG_LIMIT.
    if (analysis.canCompile && analysis.totalArgs > TRITON_DIRECT_ARG_LIMIT) {
      sd_printf("TritonIRBuilder::classifyAndAnalyze: segment will use indirect arg passing "
                "(%d args > %d direct limit)\n", analysis.totalArgs, TRITON_DIRECT_ARG_LIMIT);
    }
  }

  return analysis;
}

// ─── Combined analysis entry point ──────────────────────────────────────────

SegmentAnalysis TritonIRBuilder::analyzeSegment(NativeSlot* slots, int startSlot, int endSlot,
                                                 int totalSlots,
                                                 NDArray** externalInputs, int numExternalInputs,
                                                 NDArray** outputSlots, int totalOutputSlots,
                                                 int* requestedOutputSlotIndices,
                                                 int numRequestedOutputs) {
  auto profile = profileSegment(slots, startSlot, endSlot, outputSlots, totalOutputSlots);
  auto matched = matchPatterns(profile, slots, startSlot);

  // Log diagnostics
  sd_printf("TritonIRBuilder::analyzeSegment [%d-%d]: %d ops, %d ext inputs, %d outputs\n",
            startSlot, endSlot, profile.totalOps, profile.numUniqueExternalInputs, profile.numUniqueOutputs);
  sd_printf("  categories: elem=%d matmul=%d reduce=%d norm=%d attn=%d shape=%d data=%d const=%d id=%d cast=%d\n",
            profile.categoryCounts[0] + profile.categoryCounts[1],
            profile.categoryCounts[static_cast<int>(TritonOpCategory::MATMUL)],
            profile.categoryCounts[static_cast<int>(TritonOpCategory::REDUCTION)],
            profile.categoryCounts[static_cast<int>(TritonOpCategory::NORMALIZATION)],
            profile.categoryCounts[static_cast<int>(TritonOpCategory::FUSED_ATTENTION)],
            profile.categoryCounts[static_cast<int>(TritonOpCategory::SHAPE_MANIPULATION)],
            profile.categoryCounts[static_cast<int>(TritonOpCategory::DATA_MOVEMENT)],
            profile.categoryCounts[static_cast<int>(TritonOpCategory::CONSTANT_GENERATION)],
            profile.categoryCounts[static_cast<int>(TritonOpCategory::IDENTITY)],
            profile.categoryCounts[static_cast<int>(TritonOpCategory::CAST)]);

  for (auto& m : matched.matches) {
    sd_printf("  pattern: %s (priority=%d, %d ops)\n",
              m.description.c_str(), m.priority, static_cast<int>(m.localIndices.size()));
  }

  auto analysis = classifyAndAnalyze(profile, matched, slots, startSlot, endSlot,
                                      totalSlots, externalInputs, numExternalInputs,
                                      outputSlots, totalOutputSlots,
                                      requestedOutputSlotIndices, numRequestedOutputs);

  sd_printf("  result: pattern=%d, %d inputs, %d outputs, %d total args, canCompile=%d%s\n",
            static_cast<int>(analysis.pattern), analysis.totalInputArgs, analysis.totalOutputArgs,
            analysis.totalArgs, analysis.canCompile,
            analysis.canCompile ? "" : (", reason: " + analysis.failureReason).c_str());

  return analysis;
}

// ─── classifySegment — now delegates to 3-pass pipeline ─────────────────────

SegmentKernelPattern TritonIRBuilder::classifySegment(NativeSlot* slots, int startSlot, int endSlot) {
  auto profile = profileSegment(slots, startSlot, endSlot);
  auto matched = matchPatterns(profile, slots, startSlot);
  auto* best = matched.bestMatch();

  if (!best) {
    // Fallback: check if all ops are elementwise-compatible
    bool allEw = true;
    for (auto& node : profile.nodes) {
      if (!isElementwiseCompatible(node.category)) { allEw = false; break; }
    }
    return allEw ? SegmentKernelPattern::ELEMENTWISE_1D : SegmentKernelPattern::WHOLE_GRAPH;
  }

  switch (best->type) {
    case PatternMatch::FUSED_ATTENTION_OP:
      return SegmentKernelPattern::FUSED_ATTENTION;
    case PatternMatch::ATTENTION_QKV:
    case PatternMatch::FFN_BLOCK:
    case PatternMatch::MIXED_MEGA_SEGMENT:
      return SegmentKernelPattern::WHOLE_GRAPH;
    case PatternMatch::SOFTMAX_DECOMPOSED:
      return SegmentKernelPattern::NORMALIZATION;
    case PatternMatch::MATMUL_EPILOGUE:
      return SegmentKernelPattern::MATMUL_EPILOGUE;
    case PatternMatch::PURE_MATMUL:
      return SegmentKernelPattern::MATMUL_2D;
    case PatternMatch::PURE_REDUCTION:
      return SegmentKernelPattern::REDUCTION_1D;
    case PatternMatch::PURE_NORMALIZATION:
      return SegmentKernelPattern::NORMALIZATION;
    case PatternMatch::PURE_ELEMENTWISE:
    default:
      return SegmentKernelPattern::ELEMENTWISE_1D;
  }
}

// ─── Tile configuration ─────────────────────────────────────────────────────
//
// Uses the LaunchDims infrastructure to derive Triton tile config from the
// existing CUDA kernel launch dimension registry, rather than hardcoding.
//
// LaunchDims dim3 convention: x=gridBlocks, y=threadsPerBlock, z=sharedMemBytes
// Triton convention: blockSize=elements per program, numWarps=warps per CTA
//
// We use threadsPerBlock from LaunchDims to derive numWarps (threads/32),
// and use the registry's recommendations as the tile size baseline.

void TritonIRBuilder::selectTileConfig(const std::vector<TritonOpCategory>& categories,
                                       const std::vector<std::vector<LongType>>& shapes,
                                       int& blockSize, int& numWarps, int& numStages) {
  bool hasMatmul = false;
  bool hasReduction = false;
  bool hasFusedAttention = false;
  bool hasNormalization = false;

  // Compute total output length for dynamic dim functions
  LongType maxOutputLen = 0;
  for (auto& shape : shapes) {
    LongType len = 1;
    for (auto d : shape) len *= d;
    if (len > maxOutputLen) maxOutputLen = len;
  }

  for (auto cat : categories) {
    if (cat == TritonOpCategory::MATMUL) hasMatmul = true;
    if (cat == TritonOpCategory::REDUCTION) hasReduction = true;
    if (cat == TritonOpCategory::NORMALIZATION) hasNormalization = true;
    if (cat == TritonOpCategory::FUSED_ATTENTION) hasFusedAttention = true;
  }

  if (hasFusedAttention) {
    // Flash Attention: use softmax dims as baseline (attention is softmax-heavy)
    // getSoftmaxDims(numTads, tadLen) → dim3(grid, threads, sharedMem)
    LongType numTads = maxOutputLen > 0 ? maxOutputLen : 1;
    LongType tadLen = 64;  // headDim estimate; actual from shape if available
    for (auto& shape : shapes) {
      if (shape.size() >= 2) { tadLen = shape.back(); break; }
    }
    dim3 dims = getSoftmaxDims(numTads, tadLen);
    blockSize = 64;  // BLOCK_M for attention tiling
    numWarps = std::max(1, static_cast<int>(dims.y) / 32);
    numStages = 2;
  } else if (hasMatmul) {
    // Use getMMulDims for matmul — derives threads from output length
    int length = static_cast<int>(std::min(maxOutputLen, static_cast<LongType>(INT_MAX)));
    dim3 dims = getMMulDims(length > 0 ? length : 1, sizeof(float));
    blockSize = 128;  // BLOCK_M/BLOCK_N for 2D tiling
    numWarps = std::max(1, static_cast<int>(dims.y) / 32);
    numStages = 3;
  } else if (hasReduction || hasNormalization) {
    // Use getReduceDims for reduction-heavy segments
    int xLength = static_cast<int>(std::min(maxOutputLen, static_cast<LongType>(INT_MAX)));
    dim3 dims = getReduceDims(xLength > 0 ? xLength : 1);
    blockSize = static_cast<int>(dims.y);  // Use reduction block width as tile size
    numWarps = std::max(1, blockSize / 32);
    numStages = 2;
  } else {
    // Pure elementwise — use pairwiseTransforms dims from registry
    try {
      dim3 dims = getLaunchDims("pairwiseTransforms");
      blockSize = static_cast<int>(dims.y);  // threadsPerBlock as tile size
      numWarps = std::max(1, blockSize / 32);
    } catch (...) {
      // Fallback if key not in registry
      blockSize = 1024;
      numWarps = 4;
    }
    numStages = 3;
  }

  // Ensure blockSize is power of 2 (Triton requirement for efficient tiling)
  if (blockSize > 0 && (blockSize & (blockSize - 1)) != 0) {
    int p = 1;
    while (p < blockSize) p <<= 1;
    blockSize = p;
  }

  // Clamp to reasonable Triton tile range
  blockSize = std::max(64, std::min(blockSize, 4096));
  numWarps = std::max(1, std::min(numWarps, 16));
}

// ─── Kernel name generation ─────────────────────────────────────────────────

std::string TritonIRBuilder::generateKernelName(NativeSlot* slots, int startSlot, int endSlot) {
  std::ostringstream ss;
  ss << "triton_fused";
  for (int i = startSlot; i <= endSlot; i++) {
    ss << "_" << slots[i].opName;
  }
  std::string name = ss.str();
  if (name.size() > 200) {
    name = name.substr(0, 190) + "_seg" + std::to_string(startSlot) + "_" + std::to_string(endSlot);
  }
  return name;
}

// ─── MLIR emission helpers ──────────────────────────────────────────────────

mlir::Type TritonIRBuilder::getMLIRType(mlir::OpBuilder& builder, DataType dtype) {
  switch (dtype) {
    case FLOAT32:  return builder.getF32Type();
    case HALF:     return builder.getF16Type();
    case BFLOAT16: return builder.getBF16Type();
    case DOUBLE:   return builder.getF64Type();
    case INT8:     return builder.getIntegerType(8);
    case UINT8:    return builder.getIntegerType(8);
    case INT16:    return builder.getIntegerType(16);
    case UINT16:   return builder.getIntegerType(16);
    case INT32:    return builder.getI32Type();
    case UINT32:   return builder.getI32Type();
    case INT64:    return builder.getI64Type();
    case UINT64:   return builder.getI64Type();
    case BOOL:     return builder.getIntegerType(8);  // Use i8, not i1: Triton's LLVM lowering
                   // generates invalid bitcast (i8 to vector<1xi1>) for i1 ptr args.
                   // BOOL is stored as 1 byte in memory. castTo() handles i8→i1 when needed.
    default:       return builder.getF32Type();
  }
}

mlir::Value TritonIRBuilder::splatConstantF32(mlir::OpBuilder& builder, mlir::Location loc,
                                               mlir::RankedTensorType tensorType, float val) {
  auto elemType = tensorType.getElementType();
  if (mlir::isa<mlir::FloatType>(elemType)) {
    auto scalarAttr = builder.getFloatAttr(elemType, static_cast<double>(val));
    auto scalar = builder.create<mlir::arith::ConstantOp>(loc, elemType, scalarAttr);
    return builder.create<mlir::triton::SplatOp>(loc, tensorType, scalar);
  } else if (elemType.isSignlessInteger()) {
    auto scalarAttr = builder.getIntegerAttr(elemType, static_cast<int64_t>(val));
    auto scalar = builder.create<mlir::arith::ConstantOp>(loc, elemType, scalarAttr);
    return builder.create<mlir::triton::SplatOp>(loc, tensorType, scalar);
  }
  // Fallback: assume float
  auto scalarAttr = builder.getFloatAttr(builder.getF32Type(), static_cast<double>(val));
  auto scalar = builder.create<mlir::arith::ConstantOp>(loc, builder.getF32Type(), scalarAttr);
  return builder.create<mlir::triton::SplatOp>(loc, tensorType, scalar);
}

// ─── Type classification helpers ────────────────────────────────────────────

static mlir::Type getElementType(mlir::Value val) {
  if (auto tensorTy = mlir::dyn_cast<mlir::RankedTensorType>(val.getType()))
    return tensorTy.getElementType();
  return val.getType();
}

static bool isFloatType(mlir::Type type) {
  if (auto tensorTy = mlir::dyn_cast<mlir::RankedTensorType>(type))
    type = tensorTy.getElementType();
  return mlir::isa<mlir::FloatType>(type);
}

static bool isIntegerType(mlir::Type type) {
  if (auto tensorTy = mlir::dyn_cast<mlir::RankedTensorType>(type))
    type = tensorTy.getElementType();
  return type.isSignlessInteger();
}

static bool isBoolType(mlir::Type type) {
  if (auto tensorTy = mlir::dyn_cast<mlir::RankedTensorType>(type))
    type = tensorTy.getElementType();
  return type.isInteger(1);
}

static int getFloatBitWidth(mlir::Type type) {
  if (auto ft = mlir::dyn_cast<mlir::FloatType>(type)) return ft.getWidth();
  return 0;
}

// NegFOp and TanhOp are now legal in Triton via our patch to
// TritonToTritonGPUPass.cpp and ElementwiseOpToLLVM.cpp.
// Use the standard MLIR ops directly.

// ─── Universal type cast: cast any value to target element type ────────────

static mlir::Value castTo(mlir::OpBuilder& builder, mlir::Location loc,
                           mlir::Value val, mlir::Type targetElemType) {
  auto tensorTy = mlir::dyn_cast<mlir::RankedTensorType>(val.getType());
  if (!tensorTy) return val;
  auto srcElemType = tensorTy.getElementType();
  if (srcElemType == targetElemType) return val;

  auto targetTensorType = mlir::RankedTensorType::get(tensorTy.getShape(), targetElemType);
  bool srcIsFloat = mlir::isa<mlir::FloatType>(srcElemType);
  bool dstIsFloat = mlir::isa<mlir::FloatType>(targetElemType);
  bool srcIsBool = srcElemType.isInteger(1);
  bool dstIsBool = targetElemType.isInteger(1);
  bool srcIsInt = srcElemType.isIntOrIndex() && !srcIsBool;
  bool dstIsInt = targetElemType.isIntOrIndex() && !dstIsBool;

  if (srcIsFloat && dstIsFloat) {
    // float → float: widen or narrow
    int srcBits = getFloatBitWidth(srcElemType);
    int dstBits = getFloatBitWidth(targetElemType);
    if (srcBits == dstBits) {
      // Same bit width but different float types (e.g. f16 vs bf16):
      // go through f32 to avoid invalid TruncFOp/ExtFOp on same-width types
      auto f32Ty = builder.getF32Type();
      auto f32TensorType = mlir::RankedTensorType::get(tensorTy.getShape(), f32Ty);
      auto widened = builder.create<mlir::arith::ExtFOp>(loc, f32TensorType, val);
      return builder.create<mlir::arith::TruncFOp>(loc, targetTensorType, widened);
    } else if (dstBits > srcBits) {
      return builder.create<mlir::arith::ExtFOp>(loc, targetTensorType, val);
    } else {
      return builder.create<mlir::arith::TruncFOp>(loc, targetTensorType, val);
    }
  } else if (srcIsFloat && !dstIsFloat) {
    // float → integer/bool
    if (dstIsBool) {
      // float → bool: != 0.0
      auto zeroTy = mlir::RankedTensorType::get(tensorTy.getShape(), srcElemType);
      auto zeroAttr = builder.getFloatAttr(srcElemType, 0.0);
      auto zeroScalar = builder.create<mlir::arith::ConstantOp>(loc, srcElemType, zeroAttr);
      auto zero = builder.create<mlir::triton::SplatOp>(loc, zeroTy, zeroScalar);
      return builder.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::UNE, val, zero);
    } else {
      // float → integer (non-bool)
      // Avoid direct FPToSIOp for >32-bit targets — LLVM assertion fails on f32→i64.
      // Go through i32 intermediate: FPToSI(f32→i32) then ExtSI(i32→i64).
      int dstBits = targetElemType.getIntOrFloatBitWidth();
      if (dstBits > 32) {
        auto i32Type = builder.getI32Type();
        auto i32TensorType = mlir::RankedTensorType::get(tensorTy.getShape(), i32Type);
        auto toI32 = builder.create<mlir::arith::FPToSIOp>(loc, i32TensorType, val);
        return builder.create<mlir::arith::ExtSIOp>(loc, targetTensorType, toI32);
      } else {
        return builder.create<mlir::arith::FPToSIOp>(loc, targetTensorType, val);
      }
    }
  } else if (!srcIsFloat && dstIsFloat) {
    // integer/bool → float
    if (srcIsBool) {
      return builder.create<mlir::arith::UIToFPOp>(loc, targetTensorType, val);
    } else {
      // Avoid direct SIToFPOp for >32-bit source — same LLVM assertion issue.
      // Go through i32 intermediate: TruncI(i64→i32) then SIToFP(i32→f32).
      int srcBits = srcElemType.getIntOrFloatBitWidth();
      if (srcBits > 32) {
        auto i32Type = builder.getI32Type();
        auto i32TensorType = mlir::RankedTensorType::get(tensorTy.getShape(), i32Type);
        auto toI32 = builder.create<mlir::arith::TruncIOp>(loc, i32TensorType, val);
        return builder.create<mlir::arith::SIToFPOp>(loc, targetTensorType, toI32);
      } else {
        return builder.create<mlir::arith::SIToFPOp>(loc, targetTensorType, val);
      }
    }
  } else {
    // integer → integer
    if (srcIsBool && !dstIsBool) {
      // bool → int: zero-extend
      return builder.create<mlir::arith::ExtUIOp>(loc, targetTensorType, val);
    } else if (!srcIsBool && dstIsBool) {
      // int → bool: != 0
      auto zeroAttr = builder.getIntegerAttr(srcElemType, 0);
      auto zeroScalar = builder.create<mlir::arith::ConstantOp>(loc, srcElemType, zeroAttr);
      auto zeroTy = mlir::RankedTensorType::get(tensorTy.getShape(), srcElemType);
      auto zero = builder.create<mlir::triton::SplatOp>(loc, zeroTy, zeroScalar);
      return builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ne, val, zero);
    } else {
      int srcBits = srcElemType.getIntOrFloatBitWidth();
      int dstBits = targetElemType.getIntOrFloatBitWidth();
      if (srcBits == dstBits) {
        return val;  // no-op for same-width integer cast
      } else if (dstBits > srcBits) {
        return builder.create<mlir::arith::ExtSIOp>(loc, targetTensorType, val);
      } else {
        return builder.create<mlir::arith::TruncIOp>(loc, targetTensorType, val);
      }
    }
  }
}

// Promote a value to at least f32 for math ops. Leaves f64 as-is.
static mlir::Value promoteToFloat(mlir::OpBuilder& builder, mlir::Location loc,
                                   mlir::Value val) {
  auto elemType = getElementType(val);
  if (mlir::isa<mlir::FloatType>(elemType)) {
    // Already float — widen f16/bf16 to f32 for precision
    if (getFloatBitWidth(elemType) < 32) {
      return castTo(builder, loc, val, builder.getF32Type());
    }
    return val;
  }
  // Integer/bool → f32
  return castTo(builder, loc, val, builder.getF32Type());
}

// Find the common float type for binary ops (promote both to the wider float)
static mlir::Type commonFloatType(mlir::OpBuilder& builder, mlir::Value lhs, mlir::Value rhs) {
  auto lhsElem = getElementType(lhs);
  auto rhsElem = getElementType(rhs);
  bool lhsF = mlir::isa<mlir::FloatType>(lhsElem);
  bool rhsF = mlir::isa<mlir::FloatType>(rhsElem);

  if (lhsF && rhsF) {
    int lhsBits = getFloatBitWidth(lhsElem);
    int rhsBits = getFloatBitWidth(rhsElem);
    return lhsBits >= rhsBits ? lhsElem : rhsElem;
  } else if (lhsF) {
    return getFloatBitWidth(lhsElem) >= 32 ? lhsElem : builder.getF32Type();
  } else if (rhsF) {
    return getFloatBitWidth(rhsElem) >= 32 ? rhsElem : builder.getF32Type();
  }
  return builder.getF32Type();
}

// Find the common integer type for binary int ops
static mlir::Type commonIntType(mlir::Value lhs, mlir::Value rhs) {
  auto lhsElem = getElementType(lhs);
  auto rhsElem = getElementType(rhs);
  int lhsBits = lhsElem.getIntOrFloatBitWidth();
  int rhsBits = rhsElem.getIntOrFloatBitWidth();
  return lhsBits >= rhsBits ? lhsElem : rhsElem;
}

mlir::Value TritonIRBuilder::emitBinaryElementwise(mlir::OpBuilder& builder, mlir::Location loc,
                                                    const TritonOpMapping& mapping,
                                                    mlir::Value lhs, mlir::Value rhs) {
  auto opIr = mapping.tritonIrOp;
  bool lhsIsFloat = isFloatType(lhs.getType());
  bool rhsIsFloat = isFloatType(rhs.getType());
  bool bothInt = !lhsIsFloat && !rhsIsFloat;
  bool lhsIsBool = isBoolType(lhs.getType());
  bool rhsIsBool = isBoolType(rhs.getType());

  // Integer/bool path: stay in integer domain when both operands are integer
  if (bothInt) {
    // Coerce to same integer width (widen narrower operand)
    if (!lhsIsBool && !rhsIsBool) {
      auto intTy = commonIntType(lhs, rhs);
      lhs = castTo(builder, loc, lhs, intTy);
      rhs = castTo(builder, loc, rhs, intTy);
    } else if (lhsIsBool && rhsIsBool) {
      // Both bool — use logical ops
      if (opIr == "arith.mulf") return builder.create<mlir::arith::AndIOp>(loc, lhs, rhs);
      if (opIr == "arith.addf") return builder.create<mlir::arith::OrIOp>(loc, lhs, rhs);
      if (opIr == "arith.maximumf") return builder.create<mlir::arith::OrIOp>(loc, lhs, rhs);
      if (opIr == "arith.minimumf") return builder.create<mlir::arith::AndIOp>(loc, lhs, rhs);
      // For sub/div on bools, promote to i32
      lhs = castTo(builder, loc, lhs, builder.getI32Type());
      rhs = castTo(builder, loc, rhs, builder.getI32Type());
    } else {
      // Mixed bool + int: promote bool to the int type
      auto intTy = lhsIsBool ? getElementType(rhs) : getElementType(lhs);
      lhs = castTo(builder, loc, lhs, intTy);
      rhs = castTo(builder, loc, rhs, intTy);
    }

    // Integer arithmetic (skip if we already returned for bool ops above)
    if (opIr == "arith.addf") return builder.create<mlir::arith::AddIOp>(loc, lhs, rhs);
    if (opIr == "arith.subf") return builder.create<mlir::arith::SubIOp>(loc, lhs, rhs);
    if (opIr == "arith.mulf") return builder.create<mlir::arith::MulIOp>(loc, lhs, rhs);
    if (opIr == "arith.divf") return builder.create<mlir::arith::DivSIOp>(loc, lhs, rhs);
    if (opIr == "arith.maximumf") {
      auto cmp = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::sgt, lhs, rhs);
      return builder.create<mlir::arith::SelectOp>(loc, cmp, lhs, rhs);
    }
    if (opIr == "arith.minimumf") {
      auto cmp = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt, lhs, rhs);
      return builder.create<mlir::arith::SelectOp>(loc, cmp, lhs, rhs);
    }
    if (opIr == "arith.remf") return builder.create<mlir::arith::RemSIOp>(loc, lhs, rhs);
  }

  // Float path: promote both operands to a common float type
  auto floatTy = commonFloatType(builder, lhs, rhs);
  lhs = castTo(builder, loc, lhs, floatTy);
  rhs = castTo(builder, loc, rhs, floatTy);

  if (opIr == "arith.addf") return builder.create<mlir::arith::AddFOp>(loc, lhs, rhs);
  if (opIr == "arith.subf") return builder.create<mlir::arith::SubFOp>(loc, lhs, rhs);
  if (opIr == "arith.mulf") return builder.create<mlir::arith::MulFOp>(loc, lhs, rhs);
  if (opIr == "arith.divf") return builder.create<mlir::arith::DivFOp>(loc, lhs, rhs);
  if (opIr == "arith.maximumf") return builder.create<mlir::arith::MaximumFOp>(loc, lhs, rhs);
  if (opIr == "arith.minimumf") return builder.create<mlir::arith::MinimumFOp>(loc, lhs, rhs);
  if (opIr == "arith.remf") return builder.create<mlir::arith::RemFOp>(loc, lhs, rhs);
  if (opIr == "math.atan2") return builder.create<mlir::math::Atan2Op>(loc, lhs, rhs);

  // Custom compound binary ops
  if (opIr == "custom.floordiv") {
    // floordiv(a, b) = floor(a / b)
    auto div = builder.create<mlir::arith::DivFOp>(loc, lhs, rhs);
    return builder.create<mlir::math::FloorOp>(loc, div);
  }
  if (opIr == "custom.reversediv") {
    // reversedivide(a, b) = b / a (swapped operands)
    return builder.create<mlir::arith::DivFOp>(loc, rhs, lhs);
  }
  if (opIr == "custom.reversesub") {
    // reversesubtract(a, b) = b - a (swapped operands)
    return builder.create<mlir::arith::SubFOp>(loc, rhs, lhs);
  }
  if (opIr == "custom.squaredsub") {
    // squaredsubtract(a, b) = (a - b)^2
    auto diff = builder.create<mlir::arith::SubFOp>(loc, lhs, rhs);
    return builder.create<mlir::arith::MulFOp>(loc, diff, diff);
  }
  if (opIr == "custom.swish_mul") {
    // swish_mul(x, y) = x * sigmoid(x) * y  (SwiGLU activation)
    auto negX = builder.create<mlir::arith::NegFOp>(loc, lhs);
    auto expNegX = builder.create<mlir::math::ExpOp>(loc, negX);
    auto tensorTy = mlir::cast<mlir::RankedTensorType>(lhs.getType());
    auto one = splatConstantF32(builder, loc, tensorTy, 1.0f);
    auto onePlusExp = builder.create<mlir::arith::AddFOp>(loc, one, expNegX);
    auto sigmoid = builder.create<mlir::arith::DivFOp>(loc, one, onePlusExp);
    auto xTimesSigmoid = builder.create<mlir::arith::MulFOp>(loc, lhs, sigmoid);
    return builder.create<mlir::arith::MulFOp>(loc, xTimesSigmoid, rhs);
  }
  if (opIr == "custom.mul_no_nan") {
    // multiply_no_nan(a, b) = b == 0 ? 0 : a * b
    auto tensorTy = mlir::cast<mlir::RankedTensorType>(lhs.getType());
    auto zero = splatConstantF32(builder, loc, tensorTy, 0.0f);
    auto product = builder.create<mlir::arith::MulFOp>(loc, lhs, rhs);
    auto isZero = builder.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OEQ, rhs, zero);
    return builder.create<mlir::arith::SelectOp>(loc, isZero, zero, product);
  }

  sd_printf("TritonIRBuilder::emitBinaryElementwise: unknown op '%s'\n", opIr.c_str());
  return lhs;
}

mlir::Value TritonIRBuilder::emitUnaryElementwise(mlir::OpBuilder& builder, mlir::Location loc,
                                                   const TritonOpMapping& mapping,
                                                   const NativeSlot& slot, mlir::Value input,
                                                   int blockSize) {
  auto tensorType = mlir::cast<mlir::RankedTensorType>(input.getType());
  auto opName = mapping.opName;

  // Math ops require float inputs — promote integer/bool/f16/bf16 to at least f32
  input = promoteToFloat(builder, loc, input);
  tensorType = mlir::cast<mlir::RankedTensorType>(input.getType());

  // Convert to lowercase for matching
  std::string opLower = opName;
  std::transform(opLower.begin(), opLower.end(), opLower.begin(), ::tolower);

  if (opLower == "relu") {
    // relu(x) = max(x, 0.0)
    auto zero = splatConstantF32(builder, loc, tensorType, 0.0f);
    return builder.create<mlir::arith::MaximumFOp>(loc, input, zero);
  }

  if (opLower == "sigmoid") {
    // sigmoid(x) = 1.0 / (1.0 + exp(-x))
    auto negX = builder.create<mlir::arith::NegFOp>(loc, input);
    auto expNegX = builder.create<mlir::math::ExpOp>(loc, negX);
    auto one = splatConstantF32(builder, loc, tensorType, 1.0f);
    auto onePlusExp = builder.create<mlir::arith::AddFOp>(loc, one, expNegX);
    return builder.create<mlir::arith::DivFOp>(loc, one, onePlusExp);
  }

  if (opLower == "tanh") {
    // Compound: tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
    // Avoids reliance on Triton's math.tanh legalization patch which is unreliable
    // due to ccache interactions and TanhOp being marked illegal in some builds.
    auto two = splatConstantF32(builder, loc, tensorType, 2.0f);
    auto one = splatConstantF32(builder, loc, tensorType, 1.0f);
    auto twoX = builder.create<mlir::arith::MulFOp>(loc, two, input);
    auto exp2x = builder.create<mlir::math::ExpOp>(loc, twoX);
    auto num = builder.create<mlir::arith::SubFOp>(loc, exp2x, one);
    auto den = builder.create<mlir::arith::AddFOp>(loc, exp2x, one);
    return builder.create<mlir::arith::DivFOp>(loc, num, den);
  }

  if (opLower == "gelu") {
    // gelu(x) = 0.5 * x * (1.0 + erf(x / sqrt(2.0)))
    auto half = splatConstantF32(builder, loc, tensorType, 0.5f);
    auto sqrtTwo = splatConstantF32(builder, loc, tensorType, static_cast<float>(std::sqrt(2.0)));
    auto one = splatConstantF32(builder, loc, tensorType, 1.0f);
    auto xDivSqrt2 = builder.create<mlir::arith::DivFOp>(loc, input, sqrtTwo);
    auto erfVal = builder.create<mlir::math::ErfOp>(loc, xDivSqrt2);
    auto onePlusErf = builder.create<mlir::arith::AddFOp>(loc, one, erfVal);
    auto halfX = builder.create<mlir::arith::MulFOp>(loc, half, input);
    return builder.create<mlir::arith::MulFOp>(loc, halfX, onePlusErf);
  }

  if (opLower == "exp") {
    return builder.create<mlir::math::ExpOp>(loc, input);
  }

  if (opLower == "log") {
    return builder.create<mlir::math::LogOp>(loc, input);
  }

  if (opLower == "abs") {
    return builder.create<mlir::math::AbsFOp>(loc, input);
  }

  if (opLower == "sqrt") {
    return builder.create<mlir::math::SqrtOp>(loc, input);
  }

  if (opLower == "square") {
    // square(x) = x * x
    return builder.create<mlir::arith::MulFOp>(loc, input, input);
  }

  if (opLower == "pow") {
    // pow(x, exponent) — avoid math.PowFOp because Triton's NVIDIA backend
    // fails to legalize it during TTGIR→LLVM lowering.
    // Instead, use special cases for common exponents and exp(e*log(x)) for general case.
    float exponent = 2.0f;
    if (slot.numTArgs > 0 && slot.tArgs) {
      exponent = static_cast<float>(slot.tArgs[0]);
    }
    // Special cases that avoid log/exp entirely
    if (exponent == 0.0f) {
      return splatConstantF32(builder, loc, tensorType, 1.0f);
    }
    if (exponent == 1.0f) {
      return input;
    }
    if (exponent == 2.0f) {
      return builder.create<mlir::arith::MulFOp>(loc, input, input);
    }
    if (exponent == 0.5f) {
      return builder.create<mlir::math::SqrtOp>(loc, input);
    }
    if (exponent == -0.5f) {
      auto sq = builder.create<mlir::math::SqrtOp>(loc, input);
      auto one = splatConstantF32(builder, loc, tensorType, 1.0f);
      return builder.create<mlir::arith::DivFOp>(loc, one, sq);
    }
    if (exponent == -1.0f) {
      auto one = splatConstantF32(builder, loc, tensorType, 1.0f);
      return builder.create<mlir::arith::DivFOp>(loc, one, input);
    }
    if (exponent == 3.0f) {
      auto x2 = builder.create<mlir::arith::MulFOp>(loc, input, input);
      return builder.create<mlir::arith::MulFOp>(loc, x2, input);
    }
    // General case: pow(x, e) = exp(e * log(x))
    auto logX = builder.create<mlir::math::LogOp>(loc, input);
    auto expVal = splatConstantF32(builder, loc, tensorType, exponent);
    auto eLogX = builder.create<mlir::arith::MulFOp>(loc, expVal, logX);
    return builder.create<mlir::math::ExpOp>(loc, eLogX);
  }

  if (opLower == "clamp" || opLower == "clipbyvalue") {
    // clamp(x, min, max) = min(max(x, minVal), maxVal)
    float minVal = -3.4028235e+38f;
    float maxVal = 3.4028235e+38f;
    if (slot.numTArgs >= 2 && slot.tArgs) {
      minVal = static_cast<float>(slot.tArgs[0]);
      maxVal = static_cast<float>(slot.tArgs[1]);
    }
    auto minSplat = splatConstantF32(builder, loc, tensorType, minVal);
    auto maxSplat = splatConstantF32(builder, loc, tensorType, maxVal);
    auto clamped = builder.create<mlir::arith::MaximumFOp>(loc, input, minSplat);
    return builder.create<mlir::arith::MinimumFOp>(loc, clamped, maxSplat);
  }

  if (opLower == "neg") {
    return builder.create<mlir::arith::NegFOp>(loc, input);
  }

  if (opLower == "reciprocal") {
    // reciprocal(x) = 1.0 / x
    auto one = splatConstantF32(builder, loc, tensorType, 1.0f);
    return builder.create<mlir::arith::DivFOp>(loc, one, input);
  }

  if (opLower == "rsqrt") {
    // rsqrt(x) = 1.0 / sqrt(x)
    auto sq = builder.create<mlir::math::SqrtOp>(loc, input);
    auto one = splatConstantF32(builder, loc, tensorType, 1.0f);
    return builder.create<mlir::arith::DivFOp>(loc, one, sq);
  }

  if (opLower == "sign") {
    // sign(x) = x > 0 ? 1 : (x < 0 ? -1 : 0)
    auto zero = splatConstantF32(builder, loc, tensorType, 0.0f);
    auto one = splatConstantF32(builder, loc, tensorType, 1.0f);
    auto negOne = splatConstantF32(builder, loc, tensorType, -1.0f);
    auto gtZero = builder.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OGT, input, zero);
    auto ltZero = builder.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OLT, input, zero);
    auto negPart = builder.create<mlir::arith::SelectOp>(loc, ltZero, negOne, zero);
    return builder.create<mlir::arith::SelectOp>(loc, gtZero, one, negPart);
  }

  if (opLower == "erf") {
    return builder.create<mlir::math::ErfOp>(loc, input);
  }

  if (opLower == "erfc") {
    // erfc(x) = 1.0 - erf(x)
    auto erfVal = builder.create<mlir::math::ErfOp>(loc, input);
    auto one = splatConstantF32(builder, loc, tensorType, 1.0f);
    return builder.create<mlir::arith::SubFOp>(loc, one, erfVal);
  }

  if (opLower == "clip_by_value") {
    // clip_by_value(x, min, max) — alias of clipbyvalue
    float minVal = -3.4028235e+38f;
    float maxVal = 3.4028235e+38f;
    if (slot.numTArgs >= 2 && slot.tArgs) {
      minVal = static_cast<float>(slot.tArgs[0]);
      maxVal = static_cast<float>(slot.tArgs[1]);
    }
    auto minSplat = splatConstantF32(builder, loc, tensorType, minVal);
    auto maxSplat = splatConstantF32(builder, loc, tensorType, maxVal);
    auto clamped = builder.create<mlir::arith::MaximumFOp>(loc, input, minSplat);
    return builder.create<mlir::arith::MinimumFOp>(loc, clamped, maxSplat);
  }

  if (opLower == "log1p") {
    return builder.create<mlir::math::Log1pOp>(loc, input);
  }

  if (opLower == "ceil") {
    return builder.create<mlir::math::CeilOp>(loc, input);
  }

  if (opLower == "floor") {
    return builder.create<mlir::math::FloorOp>(loc, input);
  }

  if (opLower == "round") {
    return builder.create<mlir::math::RoundEvenOp>(loc, input);
  }

  if (opLower == "sin") {
    return builder.create<mlir::math::SinOp>(loc, input);
  }

  if (opLower == "cos") {
    return builder.create<mlir::math::CosOp>(loc, input);
  }

  if (opLower == "leakyrelu") {
    // leakyrelu(x) = x > 0 ? x : alpha * x, default alpha = 0.01
    float alpha = 0.01f;
    if (slot.numTArgs > 0 && slot.tArgs) {
      alpha = static_cast<float>(slot.tArgs[0]);
    }
    auto zero = splatConstantF32(builder, loc, tensorType, 0.0f);
    auto alphaSplat = splatConstantF32(builder, loc, tensorType, alpha);
    auto alphaX = builder.create<mlir::arith::MulFOp>(loc, alphaSplat, input);
    auto cmp = builder.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OGT, input, zero);
    return builder.create<mlir::arith::SelectOp>(loc, cmp, input, alphaX);
  }

  if (opLower == "silu" || opLower == "swish") {
    // silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
    auto negX = builder.create<mlir::arith::NegFOp>(loc, input);
    auto expNegX = builder.create<mlir::math::ExpOp>(loc, negX);
    auto one = splatConstantF32(builder, loc, tensorType, 1.0f);
    auto onePlusExp = builder.create<mlir::arith::AddFOp>(loc, one, expNegX);
    return builder.create<mlir::arith::DivFOp>(loc, input, onePlusExp);
  }

  if (opLower == "mish") {
    // mish(x) = x * tanh(softplus(x)) = x * tanh(log(1 + exp(x)))
    // Uses compound tanh: tanh(sp) = (exp(2*sp) - 1) / (exp(2*sp) + 1)
    auto expX = builder.create<mlir::math::ExpOp>(loc, input);
    auto one = splatConstantF32(builder, loc, tensorType, 1.0f);
    auto onePlusExp = builder.create<mlir::arith::AddFOp>(loc, one, expX);
    auto sp = builder.create<mlir::math::LogOp>(loc, onePlusExp);
    // Compound tanh on softplus result
    auto two = splatConstantF32(builder, loc, tensorType, 2.0f);
    auto twoSp = builder.create<mlir::arith::MulFOp>(loc, two, sp);
    auto exp2sp = builder.create<mlir::math::ExpOp>(loc, twoSp);
    auto numMish = builder.create<mlir::arith::SubFOp>(loc, exp2sp, one);
    auto denMish = builder.create<mlir::arith::AddFOp>(loc, exp2sp, one);
    auto tanhSp = builder.create<mlir::arith::DivFOp>(loc, numMish, denMish);
    return builder.create<mlir::arith::MulFOp>(loc, input, tanhSp);
  }

  if (opLower == "elu") {
    // elu(x) = x > 0 ? x : alpha * (exp(x) - 1), default alpha = 1.0
    float alpha = 1.0f;
    if (slot.numTArgs > 0 && slot.tArgs) {
      alpha = static_cast<float>(slot.tArgs[0]);
    }
    auto zero = splatConstantF32(builder, loc, tensorType, 0.0f);
    auto one = splatConstantF32(builder, loc, tensorType, 1.0f);
    auto alphaSplat = splatConstantF32(builder, loc, tensorType, alpha);
    auto expX = builder.create<mlir::math::ExpOp>(loc, input);
    auto expXMinusOne = builder.create<mlir::arith::SubFOp>(loc, expX, one);
    auto negPart = builder.create<mlir::arith::MulFOp>(loc, alphaSplat, expXMinusOne);
    auto cmp = builder.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OGT, input, zero);
    return builder.create<mlir::arith::SelectOp>(loc, cmp, input, negPart);
  }

  if (opLower == "selu") {
    // selu(x) = lambda * (x > 0 ? x : alpha * (exp(x) - 1))
    // lambda = 1.0507, alpha = 1.67326
    float lambda = 1.0507009873554804934193349852946f;
    float alpha = 1.6732632423543772848170429916717f;
    auto zero = splatConstantF32(builder, loc, tensorType, 0.0f);
    auto one = splatConstantF32(builder, loc, tensorType, 1.0f);
    auto alphaSplat = splatConstantF32(builder, loc, tensorType, alpha);
    auto lambdaSplat = splatConstantF32(builder, loc, tensorType, lambda);
    auto expX = builder.create<mlir::math::ExpOp>(loc, input);
    auto expXMinusOne = builder.create<mlir::arith::SubFOp>(loc, expX, one);
    auto negPart = builder.create<mlir::arith::MulFOp>(loc, alphaSplat, expXMinusOne);
    auto cmp = builder.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OGT, input, zero);
    auto selected = builder.create<mlir::arith::SelectOp>(loc, cmp, input, negPart);
    return builder.create<mlir::arith::MulFOp>(loc, lambdaSplat, selected);
  }

  if (opLower == "softplus") {
    // softplus(x) = log(1 + exp(x))
    auto expX = builder.create<mlir::math::ExpOp>(loc, input);
    auto one = splatConstantF32(builder, loc, tensorType, 1.0f);
    auto onePlusExp = builder.create<mlir::arith::AddFOp>(loc, one, expX);
    return builder.create<mlir::math::LogOp>(loc, onePlusExp);
  }

  if (opLower == "softsign") {
    // softsign(x) = x / (1 + |x|)
    auto absX = builder.create<mlir::math::AbsFOp>(loc, input);
    auto one = splatConstantF32(builder, loc, tensorType, 1.0f);
    auto denom = builder.create<mlir::arith::AddFOp>(loc, one, absX);
    return builder.create<mlir::arith::DivFOp>(loc, input, denom);
  }

  if (opLower == "hard_sigmoid") {
    // hard_sigmoid(x) = clip(x/6 + 0.5, 0, 1)
    auto sixth = splatConstantF32(builder, loc, tensorType, 1.0f / 6.0f);
    auto half = splatConstantF32(builder, loc, tensorType, 0.5f);
    auto zero = splatConstantF32(builder, loc, tensorType, 0.0f);
    auto one = splatConstantF32(builder, loc, tensorType, 1.0f);
    auto scaled = builder.create<mlir::arith::MulFOp>(loc, input, sixth);
    auto shifted = builder.create<mlir::arith::AddFOp>(loc, scaled, half);
    auto clamped = builder.create<mlir::arith::MaximumFOp>(loc, shifted, zero);
    return builder.create<mlir::arith::MinimumFOp>(loc, clamped, one);
  }

  if (opLower == "hardtanh") {
    // hardtanh(x) = clip(x, -1, 1)
    auto negOne = splatConstantF32(builder, loc, tensorType, -1.0f);
    auto one = splatConstantF32(builder, loc, tensorType, 1.0f);
    auto clamped = builder.create<mlir::arith::MaximumFOp>(loc, input, negOne);
    return builder.create<mlir::arith::MinimumFOp>(loc, clamped, one);
  }

  if (opLower == "relu6") {
    // relu6(x) = clip(x, 0, 6)
    auto zero = splatConstantF32(builder, loc, tensorType, 0.0f);
    auto six = splatConstantF32(builder, loc, tensorType, 6.0f);
    auto clamped = builder.create<mlir::arith::MaximumFOp>(loc, input, zero);
    return builder.create<mlir::arith::MinimumFOp>(loc, clamped, six);
  }

  // Scalar binary ops: second operand comes from tArgs[0]
  if (opLower == "add_scalar") {
    float scalar = (slot.numTArgs > 0 && slot.tArgs) ? static_cast<float>(slot.tArgs[0]) : 0.0f;
    auto scalarSplat = splatConstantF32(builder, loc, tensorType, scalar);
    return builder.create<mlir::arith::AddFOp>(loc, input, scalarSplat);
  }

  if (opLower == "subtract_scalar") {
    float scalar = (slot.numTArgs > 0 && slot.tArgs) ? static_cast<float>(slot.tArgs[0]) : 0.0f;
    auto scalarSplat = splatConstantF32(builder, loc, tensorType, scalar);
    return builder.create<mlir::arith::SubFOp>(loc, input, scalarSplat);
  }

  if (opLower == "multiply_scalar") {
    float scalar = (slot.numTArgs > 0 && slot.tArgs) ? static_cast<float>(slot.tArgs[0]) : 1.0f;
    auto scalarSplat = splatConstantF32(builder, loc, tensorType, scalar);
    return builder.create<mlir::arith::MulFOp>(loc, input, scalarSplat);
  }

  if (opLower == "divide_scalar") {
    float scalar = (slot.numTArgs > 0 && slot.tArgs) ? static_cast<float>(slot.tArgs[0]) : 1.0f;
    auto scalarSplat = splatConstantF32(builder, loc, tensorType, scalar);
    return builder.create<mlir::arith::DivFOp>(loc, input, scalarSplat);
  }

  sd_printf("TritonIRBuilder::emitUnaryElementwise: unhandled op '%s'\n", opName.c_str());
  return input;
}

// ─── Comparison op emission ─────────────────────────────────────────────────

mlir::Value TritonIRBuilder::emitComparisonOp(mlir::OpBuilder& builder, mlir::Location loc,
                                               const std::string& opName,
                                               mlir::Value lhs, mlir::Value rhs, int blockSize) {
  // Normalize op name to lowercase
  std::string opLower = opName;
  std::transform(opLower.begin(), opLower.end(), opLower.begin(), ::tolower);

  bool lhsIsFloat = isFloatType(lhs.getType());
  bool rhsIsFloat = isFloatType(rhs.getType());

  // If either operand is float, promote both to a common float type
  if (lhsIsFloat || rhsIsFloat) {
    auto floatTy = commonFloatType(builder, lhs, rhs);
    lhs = castTo(builder, loc, lhs, floatTy);
    rhs = castTo(builder, loc, rhs, floatTy);

    mlir::arith::CmpFPredicate pred;
    if (opLower == "greater")            pred = mlir::arith::CmpFPredicate::OGT;
    else if (opLower == "greater_equal") pred = mlir::arith::CmpFPredicate::OGE;
    else if (opLower == "less")          pred = mlir::arith::CmpFPredicate::OLT;
    else if (opLower == "less_equal")    pred = mlir::arith::CmpFPredicate::OLE;
    else if (opLower == "equals")        pred = mlir::arith::CmpFPredicate::OEQ;
    else if (opLower == "not_equals")    pred = mlir::arith::CmpFPredicate::ONE;
    else {
      sd_printf("TritonIRBuilder::emitComparisonOp: unknown float comparison '%s'\n", opName.c_str());
      pred = mlir::arith::CmpFPredicate::OEQ;
    }
    return builder.create<mlir::arith::CmpFOp>(loc, pred, lhs, rhs);
  } else {
    // Both integer — coerce to same width
    auto intTy = commonIntType(lhs, rhs);
    lhs = castTo(builder, loc, lhs, intTy);
    rhs = castTo(builder, loc, rhs, intTy);

    mlir::arith::CmpIPredicate pred;
    if (opLower == "greater")            pred = mlir::arith::CmpIPredicate::sgt;
    else if (opLower == "greater_equal") pred = mlir::arith::CmpIPredicate::sge;
    else if (opLower == "less")          pred = mlir::arith::CmpIPredicate::slt;
    else if (opLower == "less_equal")    pred = mlir::arith::CmpIPredicate::sle;
    else if (opLower == "equals")        pred = mlir::arith::CmpIPredicate::eq;
    else if (opLower == "not_equals")    pred = mlir::arith::CmpIPredicate::ne;
    else {
      sd_printf("TritonIRBuilder::emitComparisonOp: unknown int comparison '%s'\n", opName.c_str());
      pred = mlir::arith::CmpIPredicate::eq;
    }
    return builder.create<mlir::arith::CmpIOp>(loc, pred, lhs, rhs);
  }
}

// ─── Logical op emission ────────────────────────────────────────────────────

mlir::Value TritonIRBuilder::emitLogicalOp(mlir::OpBuilder& builder, mlir::Location loc,
                                            const std::string& opName,
                                            mlir::Value lhs, mlir::Value rhs, int blockSize) {
  std::string opLower = opName;
  std::transform(opLower.begin(), opLower.end(), opLower.begin(), ::tolower);

  // Coerce both inputs to i1 (bool)
  lhs = castTo(builder, loc, lhs, builder.getI1Type());

  // Unary logical_not / boolean_not — single input XOR with all-ones
  if (opLower == "boolean_not" || opLower == "logical_not") {
    auto tensorTy = mlir::cast<mlir::RankedTensorType>(lhs.getType());
    auto trueAttr = builder.getIntegerAttr(builder.getI1Type(), 1);
    auto trueScalar = builder.create<mlir::arith::ConstantOp>(loc, builder.getI1Type(), trueAttr);
    auto allOnes = builder.create<mlir::triton::SplatOp>(loc, tensorTy, trueScalar);
    return builder.create<mlir::arith::XOrIOp>(loc, lhs, allOnes);
  }

  // Binary logical ops
  rhs = castTo(builder, loc, rhs, builder.getI1Type());

  if (opLower == "boolean_and" || opLower == "logical_and") {
    return builder.create<mlir::arith::AndIOp>(loc, lhs, rhs);
  }
  if (opLower == "boolean_or" || opLower == "logical_or") {
    return builder.create<mlir::arith::OrIOp>(loc, lhs, rhs);
  }
  if (opLower == "boolean_xor") {
    return builder.create<mlir::arith::XOrIOp>(loc, lhs, rhs);
  }

  sd_printf("TritonIRBuilder::emitLogicalOp: unknown logical op '%s'\n", opName.c_str());
  return lhs;
}

// ─── Ternary select/where emission ──────────────────────────────────────────

mlir::Value TritonIRBuilder::emitTernaryOp(mlir::OpBuilder& builder, mlir::Location loc,
                                            mlir::Value condition, mlir::Value trueVal,
                                            mlir::Value falseVal, int blockSize) {
  // Condition must be i1 (bool)
  condition = castTo(builder, loc, condition, builder.getI1Type());

  // trueVal and falseVal must have same type — promote to common type
  auto trueElem = getElementType(trueVal);
  auto falseElem = getElementType(falseVal);

  if (trueElem != falseElem) {
    bool trueIsFloat = mlir::isa<mlir::FloatType>(trueElem);
    bool falseIsFloat = mlir::isa<mlir::FloatType>(falseElem);
    if (trueIsFloat || falseIsFloat) {
      auto floatTy = commonFloatType(builder, trueVal, falseVal);
      trueVal = castTo(builder, loc, trueVal, floatTy);
      falseVal = castTo(builder, loc, falseVal, floatTy);
    } else {
      auto intTy = commonIntType(trueVal, falseVal);
      trueVal = castTo(builder, loc, trueVal, intTy);
      falseVal = castTo(builder, loc, falseVal, intTy);
    }
  }

  return builder.create<mlir::arith::SelectOp>(loc, condition, trueVal, falseVal);
}

// ─── Reduction op emission ───────────────────────────────────────────────────

mlir::Value TritonIRBuilder::emitReductionOp(mlir::OpBuilder& builder, mlir::Location loc,
                                              const std::string& opName,
                                              mlir::Value input, int reductionAxis,
                                              mlir::RankedTensorType outputType) {
  std::string opLower = opName;
  std::transform(opLower.begin(), opLower.end(), opLower.begin(), ::tolower);

  // Promote input to float for math ops
  input = promoteToFloat(builder, loc, input);
  auto tensorTy = mlir::cast<mlir::RankedTensorType>(input.getType());
  auto elemType = tensorTy.getElementType();

  // For reduce_norm2: square input first, then reduce_sum, then sqrt
  if (opLower == "reduce_norm2" || opLower == "norm2") {
    input = builder.create<mlir::arith::MulFOp>(loc, input, input);
  }
  // For reduce_norm1: abs input first, then reduce_sum
  if (opLower == "reduce_norm1" || opLower == "norm1") {
    input = builder.create<mlir::math::AbsFOp>(loc, input);
  }

  // Clamp reduction axis to valid range for the tensor's actual rank.
  // In the 1D kernel skeleton, tensors are rank-1 so axis must be 0.
  int64_t rank = tensorTy.getRank();
  if (reductionAxis < 0) reductionAxis += static_cast<int>(rank);
  if (reductionAxis < 0 || reductionAxis >= static_cast<int>(rank)) reductionAxis = 0;

  // Create tt.reduce op with combiner region
  // tt.reduce takes a tensor and reduces along one axis using a combiner
  auto reduceOp = builder.create<mlir::triton::ReduceOp>(loc,
      mlir::ValueRange{input}, reductionAxis);

  // Build combiner region — two block arguments (accumulator, element)
  auto& combinerRegion = reduceOp.getCombineOp();
  auto* combinerBlock = builder.createBlock(&combinerRegion, {}, {elemType, elemType},
                                             {loc, loc});
  auto acc = combinerBlock->getArgument(0);
  auto elem = combinerBlock->getArgument(1);

  builder.setInsertionPointToEnd(combinerBlock);

  mlir::Value combined;
  if (opLower == "reduce_sum" || opLower == "sum" ||
      opLower == "reduce_mean" || opLower == "mean" ||
      opLower == "reduce_norm1" || opLower == "norm1" ||
      opLower == "reduce_norm2" || opLower == "norm2" ||
      opLower == "reduce_variance" || opLower == "reduce_stdev" ||
      opLower == "reduce_logsumexp") {
    combined = builder.create<mlir::arith::AddFOp>(loc, acc, elem);
  } else if (opLower == "reduce_max" || opLower == "max" || opLower == "normmax") {
    combined = builder.create<mlir::arith::MaximumFOp>(loc, acc, elem);
  } else if (opLower == "reduce_min" || opLower == "min") {
    combined = builder.create<mlir::arith::MinimumFOp>(loc, acc, elem);
  } else if (opLower == "reduce_prod" || opLower == "prod") {
    combined = builder.create<mlir::arith::MulFOp>(loc, acc, elem);
  } else {
    // Default to sum
    combined = builder.create<mlir::arith::AddFOp>(loc, acc, elem);
  }

  builder.create<mlir::triton::ReduceReturnOp>(loc, mlir::ValueRange{combined});

  // Restore insertion point to after the reduce op (was inside combiner region)
  builder.setInsertionPointAfter(reduceOp);

  // Get the reduction result
  mlir::Value result = reduceOp->getResult(0);

  // Post-processing for compound reductions
  if (opLower == "reduce_mean" || opLower == "mean") {
    // Divide by reduction dimension size
    int64_t reductionSize = tensorTy.getShape()[reductionAxis];
    auto countVal = builder.create<mlir::arith::ConstantOp>(
        loc, elemType, builder.getFloatAttr(elemType, static_cast<double>(reductionSize)));
    result = builder.create<mlir::arith::DivFOp>(loc, result, countVal);
  } else if (opLower == "reduce_norm2" || opLower == "norm2") {
    result = builder.create<mlir::math::SqrtOp>(loc, result);
  } else if (opLower == "reduce_logsumexp") {
    result = builder.create<mlir::math::LogOp>(loc, result);
  } else if (opLower == "reduce_stdev") {
    // stdev = sqrt(variance) — variance is mean of squares minus square of mean
    // Simplified: assume result is already variance, just sqrt
    result = builder.create<mlir::math::SqrtOp>(loc, result);
  }

  return result;
}

// ─── Normalization op emission ───────────────────────────────────────────────

mlir::Value TritonIRBuilder::emitNormalizationOp(mlir::OpBuilder& builder, mlir::Location loc,
                                                  const std::string& opName,
                                                  mlir::Value input, int axis,
                                                  mlir::RankedTensorType outputType) {
  std::string opLower = opName;
  std::transform(opLower.begin(), opLower.end(), opLower.begin(), ::tolower);

  // Promote input to float
  input = promoteToFloat(builder, loc, input);
  auto tensorTy = mlir::cast<mlir::RankedTensorType>(input.getType());
  auto elemType = tensorTy.getElementType();
  // Clamp axis to valid range for tensor's actual rank (1D kernel → always 0)
  int64_t normRank = tensorTy.getRank();
  if (axis < 0) axis += static_cast<int>(normRank);
  if (axis < 0 || axis >= static_cast<int>(normRank)) axis = 0;
  int64_t reductionSize = tensorTy.getShape()[axis];

  // Helper lambda: create a reduce op with combiner, restore insertion point after
  auto makeReduce = [&](mlir::Value src, int reduceAxis, auto combinerFn) -> mlir::Value {
    auto op = builder.create<mlir::triton::ReduceOp>(loc, mlir::ValueRange{src}, reduceAxis);
    {
      auto& region = op.getCombineOp();
      auto* block = builder.createBlock(&region, {}, {elemType, elemType}, {loc, loc});
      builder.setInsertionPointToEnd(block);
      auto result = combinerFn(builder, loc, block->getArgument(0), block->getArgument(1));
      builder.create<mlir::triton::ReduceReturnOp>(loc, mlir::ValueRange{result});
    }
    builder.setInsertionPointAfter(op);
    return op->getResult(0);
  };

  auto addCombiner = [](mlir::OpBuilder& b, mlir::Location l, mlir::Value a, mlir::Value e) {
    return b.create<mlir::arith::AddFOp>(l, a, e).getResult();
  };
  auto maxCombiner = [](mlir::OpBuilder& b, mlir::Location l, mlir::Value a, mlir::Value e) {
    return b.create<mlir::arith::MaximumFOp>(l, a, e).getResult();
  };

  if (opLower == "softmax") {
    // softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
    auto maxResult = makeReduce(input, axis, maxCombiner);
    auto maxSplat = builder.create<mlir::triton::SplatOp>(loc, tensorTy, maxResult);
    auto shifted = builder.create<mlir::arith::SubFOp>(loc, input, maxSplat);
    auto expShifted = builder.create<mlir::math::ExpOp>(loc, shifted);
    auto sumResult = makeReduce(expShifted, axis, addCombiner);
    auto sumSplat = builder.create<mlir::triton::SplatOp>(loc, tensorTy, sumResult);
    return builder.create<mlir::arith::DivFOp>(loc, expShifted, sumSplat);

  } else if (opLower == "log_softmax") {
    // log_softmax(x) = x - max(x) - log(sum(exp(x - max(x))))
    auto maxResult = makeReduce(input, axis, maxCombiner);
    auto maxSplat = builder.create<mlir::triton::SplatOp>(loc, tensorTy, maxResult);
    auto shifted = builder.create<mlir::arith::SubFOp>(loc, input, maxSplat);
    auto expShifted = builder.create<mlir::math::ExpOp>(loc, shifted);
    auto sumResult = makeReduce(expShifted, axis, addCombiner);
    auto logSum = builder.create<mlir::math::LogOp>(loc, sumResult);
    auto logSumSplat = builder.create<mlir::triton::SplatOp>(loc, tensorTy, logSum);
    return builder.create<mlir::arith::SubFOp>(loc, shifted, logSumSplat);

  } else if (opLower == "rms_norm") {
    // rms_norm(x) = x * rsqrt(mean(x^2) + eps)
    float eps = 1e-6f;
    auto squared = builder.create<mlir::arith::MulFOp>(loc, input, input);
    auto sumSquared = makeReduce(squared, axis, addCombiner);
    auto countVal = builder.create<mlir::arith::ConstantOp>(
        loc, elemType, builder.getFloatAttr(elemType, static_cast<double>(reductionSize)));
    auto meanSquared = builder.create<mlir::arith::DivFOp>(loc, sumSquared, countVal);
    auto epsVal = builder.create<mlir::arith::ConstantOp>(
        loc, elemType, builder.getFloatAttr(elemType, static_cast<double>(eps)));
    auto meanPlusEps = builder.create<mlir::arith::AddFOp>(loc, meanSquared, epsVal);
    auto rsqrtVal = builder.create<mlir::math::RsqrtOp>(loc, meanPlusEps);
    auto rsqrtSplat = builder.create<mlir::triton::SplatOp>(loc, tensorTy, rsqrtVal);
    return builder.create<mlir::arith::MulFOp>(loc, input, rsqrtSplat);

  } else if (opLower == "layer_norm") {
    // layer_norm(x) = (x - mean(x)) * rsqrt(var(x) + eps)
    float eps = 1e-5f;
    auto countVal = builder.create<mlir::arith::ConstantOp>(
        loc, elemType, builder.getFloatAttr(elemType, static_cast<double>(reductionSize)));

    // Mean
    auto sumResult = makeReduce(input, axis, addCombiner);
    auto meanVal = builder.create<mlir::arith::DivFOp>(loc, sumResult, countVal);
    auto meanSplat = builder.create<mlir::triton::SplatOp>(loc, tensorTy, meanVal);
    auto centered = builder.create<mlir::arith::SubFOp>(loc, input, meanSplat);

    // Variance
    auto centeredSq = builder.create<mlir::arith::MulFOp>(loc, centered, centered);
    auto varSum = makeReduce(centeredSq, axis, addCombiner);
    auto varianceVal = builder.create<mlir::arith::DivFOp>(loc, varSum, countVal);
    auto epsVal = builder.create<mlir::arith::ConstantOp>(
        loc, elemType, builder.getFloatAttr(elemType, static_cast<double>(eps)));
    auto varPlusEps = builder.create<mlir::arith::AddFOp>(loc, varianceVal, epsVal);
    auto rsqrtVal = builder.create<mlir::math::RsqrtOp>(loc, varPlusEps);
    auto rsqrtSplat = builder.create<mlir::triton::SplatOp>(loc, tensorTy, rsqrtVal);
    return builder.create<mlir::arith::MulFOp>(loc, centered, rsqrtSplat);
  }

  sd_printf("TritonIRBuilder::emitNormalizationOp: normalization '%s' not fully implemented\n", opName.c_str());
  return input;
}

// ─── Matmul op emission ─────────────────────────────────────────────────────

void TritonIRBuilder::emitMatmulKernel(mlir::OpBuilder& builder, mlir::Location loc,
                                        mlir::Value aPtr, mlir::Value bPtr, mlir::Value cPtr,
                                        int M, int N, int K,
                                        int blockM, int blockN, int blockK) {
  auto f32Type = builder.getF32Type();
  auto i32Type = builder.getI32Type();
  auto i1Type = builder.getI1Type();

  // Extract element types from pointer args for mixed-precision support.
  // Inputs (A, B) may be f16/bf16/int8; accumulator is always f32;
  // output (C) stores in its native type with cast from f32 if needed.
  auto aPtrType = mlir::cast<mlir::triton::PointerType>(aPtr.getType());
  auto bPtrType = mlir::cast<mlir::triton::PointerType>(bPtr.getType());
  auto cPtrType = mlir::cast<mlir::triton::PointerType>(cPtr.getType());
  auto aElemType = aPtrType.getPointeeType();
  auto bElemType = bPtrType.getPointeeType();
  auto cElemType = cPtrType.getPointeeType();

  // Determine InputPrecision for DotOp based on input types
  auto dotPrecision = mlir::triton::InputPrecision::TF32;  // default for f32 inputs
  bool inputIsF32 = mlir::isa<mlir::Float32Type>(aElemType);
  if (!inputIsF32) {
    // f16, bf16, int8 use IEEE — TF32 only applies to f32 inputs
    dotPrecision = mlir::triton::InputPrecision::IEEE;
  }

  sd_printf("TritonIRBuilder::emitMatmulKernel: A elem=%s, B elem=%s, C elem=%s, precision=%s\n",
            inputIsF32 ? "f32" : "non-f32", inputIsF32 ? "f32" : "non-f32",
            mlir::isa<mlir::Float32Type>(cElemType) ? "f32" : "non-f32",
            inputIsF32 ? "TF32" : "IEEE");

  // Program IDs for 2D grid
  auto pidM = builder.create<mlir::triton::GetProgramIdOp>(
      loc, i32Type, mlir::triton::ProgramIDDim::X);
  auto pidN = builder.create<mlir::triton::GetProgramIdOp>(
      loc, i32Type, mlir::triton::ProgramIDDim::Y);

  // Tile index offsets
  auto blockMConst = builder.create<mlir::arith::ConstantIntOp>(loc, blockM, 32);
  auto blockNConst = builder.create<mlir::arith::ConstantIntOp>(loc, blockN, 32);
  auto mOffset = builder.create<mlir::arith::MulIOp>(loc, pidM, blockMConst);
  auto nOffset = builder.create<mlir::arith::MulIOp>(loc, pidN, blockNConst);

  // Create range vectors for tile offsets
  auto i32BmType = mlir::RankedTensorType::get({blockM}, i32Type);
  auto i32BnType = mlir::RankedTensorType::get({blockN}, i32Type);
  auto i32BkType = mlir::RankedTensorType::get({blockK}, i32Type);

  auto rangeM = builder.create<mlir::triton::MakeRangeOp>(loc, i32BmType, 0, blockM);
  auto rangeN = builder.create<mlir::triton::MakeRangeOp>(loc, i32BnType, 0, blockN);
  auto rangeK = builder.create<mlir::triton::MakeRangeOp>(loc, i32BkType, 0, blockK);

  auto splatMOffset = builder.create<mlir::triton::SplatOp>(loc, i32BmType, mOffset);
  auto mIndices = builder.create<mlir::arith::AddIOp>(loc, splatMOffset, rangeM);
  auto splatNOffset = builder.create<mlir::triton::SplatOp>(loc, i32BnType, nOffset);
  auto nIndices = builder.create<mlir::arith::AddIOp>(loc, splatNOffset, rangeN);

  // Initialize accumulator to zeros: always f32 (tensor cores accumulate in f32)
  auto accType = mlir::RankedTensorType::get({blockM, blockN}, f32Type);
  auto zeroAttr = builder.getFloatAttr(f32Type, 0.0);
  auto zeroScalar = builder.create<mlir::arith::ConstantOp>(loc, f32Type, zeroAttr);
  auto accInit = builder.create<mlir::triton::SplatOp>(loc, accType, zeroScalar);

  // K-loop bounds (i32 — Triton convention, NOT index type)
  auto kStart = builder.create<mlir::arith::ConstantIntOp>(loc, 0, 32);
  auto kEnd = builder.create<mlir::arith::ConstantIntOp>(loc, K, 32);
  auto kStep = builder.create<mlir::arith::ConstantIntOp>(loc, blockK, 32);

  // K-loop via scf.for (i32 bounds)
  auto forOp = builder.create<mlir::scf::ForOp>(
      loc, kStart, kEnd, kStep, mlir::ValueRange{accInit});

  // Inside the K-loop body
  builder.setInsertionPointToStart(forOp.getBody());
  auto kIdxI32 = forOp.getInductionVar();  // i32 induction variable
  auto accIter = forOp.getBody()->getArgument(1);  // loop-carried accumulator

  // Splat k offset for pointer arithmetic
  auto splatKOffset = builder.create<mlir::triton::SplatOp>(loc, i32BkType, kIdxI32);
  auto kIndices = builder.create<mlir::arith::AddIOp>(loc, splatKOffset, rangeK);

  // Load A tile [BM, BK] in native dtype
  auto kConst = builder.create<mlir::arith::ConstantIntOp>(loc, K, 32);

  // Compute 2D pointer offsets for A: mIndices[:, None] * K + kIndices[None, :]
  auto mExpanded = builder.create<mlir::triton::ExpandDimsOp>(loc, mIndices, 1);  // [BM, 1]
  auto kExpanded = builder.create<mlir::triton::ExpandDimsOp>(loc, kIndices, 0);  // [1, BK]

  auto i32BmBkType = mlir::RankedTensorType::get({blockM, blockK}, i32Type);
  auto kSplat = builder.create<mlir::triton::SplatOp>(loc, mlir::RankedTensorType::get({blockM, 1}, i32Type), kConst);
  auto mTimesK = builder.create<mlir::arith::MulIOp>(loc, mExpanded, kSplat);
  auto mTimesKBroadcast = builder.create<mlir::triton::BroadcastOp>(loc, i32BmBkType, mTimesK);
  auto kBroadcast = builder.create<mlir::triton::BroadcastOp>(loc, i32BmBkType, kExpanded);
  auto aOffsets = builder.create<mlir::arith::AddIOp>(loc, mTimesKBroadcast, kBroadcast);

  auto aPtrTensorType = mlir::RankedTensorType::get({blockM, blockK},
      mlir::triton::PointerType::get(aElemType, 1));
  auto aSplat = builder.create<mlir::triton::SplatOp>(loc, aPtrTensorType, aPtr);
  auto aPtrs = builder.create<mlir::triton::AddPtrOp>(loc, aPtrTensorType, aSplat, aOffsets);

  // Create 2D mask for A tile: mIndices < M && kIndices < K
  auto mConst = builder.create<mlir::arith::ConstantIntOp>(loc, M, 32);
  auto kConst2 = builder.create<mlir::arith::ConstantIntOp>(loc, K, 32);
  auto mConstSplat = builder.create<mlir::triton::SplatOp>(loc, i32BmType, mConst);
  auto kConstSplat = builder.create<mlir::triton::SplatOp>(loc, i32BkType, kConst2);
  auto mMask1D = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt,
      builder.create<mlir::arith::AddIOp>(loc, splatMOffset, rangeM), mConstSplat);
  auto kMask1D = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt,
      kIndices, kConstSplat);
  auto i1BmBkType = mlir::RankedTensorType::get({blockM, blockK}, i1Type);
  auto mMaskExp = builder.create<mlir::triton::ExpandDimsOp>(loc, mMask1D, 1);
  auto kMaskExp = builder.create<mlir::triton::ExpandDimsOp>(loc, kMask1D, 0);
  auto mMask2D = builder.create<mlir::triton::BroadcastOp>(loc, i1BmBkType, mMaskExp);
  auto kMask2D = builder.create<mlir::triton::BroadcastOp>(loc, i1BmBkType, kMaskExp);
  auto aMask = builder.create<mlir::arith::AndIOp>(loc, mMask2D, kMask2D);

  auto aLoaded = builder.create<mlir::triton::LoadOp>(loc,
      /*ptr=*/aPtrs.getResult(), /*mask=*/aMask.getResult(), /*other=*/mlir::Value(),
      /*cache=*/mlir::triton::CacheModifier::NONE,
      /*evict=*/mlir::triton::EvictionPolicy::NORMAL,
      /*isVolatile=*/false);

  // Load B tile [BK, BN] in native dtype
  auto nConst = builder.create<mlir::arith::ConstantIntOp>(loc, N, 32);

  auto kExpandedB = builder.create<mlir::triton::ExpandDimsOp>(loc, kIndices, 1);  // [BK, 1]
  auto nExpandedB = builder.create<mlir::triton::ExpandDimsOp>(loc, nIndices, 0);  // [1, BN]

  auto i32BkBnType = mlir::RankedTensorType::get({blockK, blockN}, i32Type);
  auto nSplat = builder.create<mlir::triton::SplatOp>(loc, mlir::RankedTensorType::get({blockK, 1}, i32Type), nConst);
  auto kTimesN = builder.create<mlir::arith::MulIOp>(loc, kExpandedB, nSplat);
  auto kTimesNBroadcast = builder.create<mlir::triton::BroadcastOp>(loc, i32BkBnType, kTimesN);
  auto nBroadcastB = builder.create<mlir::triton::BroadcastOp>(loc, i32BkBnType, nExpandedB);
  auto bOffsets = builder.create<mlir::arith::AddIOp>(loc, kTimesNBroadcast, nBroadcastB);

  auto bPtrTensorType = mlir::RankedTensorType::get({blockK, blockN},
      mlir::triton::PointerType::get(bElemType, 1));
  auto bSplat = builder.create<mlir::triton::SplatOp>(loc, bPtrTensorType, bPtr);
  auto bPtrs = builder.create<mlir::triton::AddPtrOp>(loc, bPtrTensorType, bSplat, bOffsets);

  // Create 2D mask for B tile: kIndices < K && nIndices < N
  auto nConstSplat = builder.create<mlir::triton::SplatOp>(loc, i32BnType, nConst);
  auto nMask1D = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt,
      builder.create<mlir::arith::AddIOp>(loc, splatNOffset, rangeN), nConstSplat);
  auto i1BkBnType = mlir::RankedTensorType::get({blockK, blockN}, i1Type);
  auto kMaskExpB = builder.create<mlir::triton::ExpandDimsOp>(loc, kMask1D, 1);
  auto nMaskExpB = builder.create<mlir::triton::ExpandDimsOp>(loc, nMask1D, 0);
  auto kMask2DB = builder.create<mlir::triton::BroadcastOp>(loc, i1BkBnType, kMaskExpB);
  auto nMask2DB = builder.create<mlir::triton::BroadcastOp>(loc, i1BkBnType, nMaskExpB);
  auto bMask = builder.create<mlir::arith::AndIOp>(loc, kMask2DB, nMask2DB);

  auto bLoaded = builder.create<mlir::triton::LoadOp>(loc,
      /*ptr=*/bPtrs.getResult(), /*mask=*/bMask.getResult(), /*other=*/mlir::Value(),
      /*cache=*/mlir::triton::CacheModifier::NONE,
      /*evict=*/mlir::triton::EvictionPolicy::NORMAL,
      /*isVolatile=*/false);

  // Matrix multiply: acc += dot(A_tile, B_tile)
  // tt.dot requires A and B to have same element bit width.
  // Accumulator is always f32. Tensor cores handle f16/bf16→f32 natively.
  auto dotResult = builder.create<mlir::triton::DotOp>(
      loc, accType, aLoaded, bLoaded, accIter,
      dotPrecision, /*maxNumImpreciseAcc=*/0);

  // Yield accumulator for next K-iteration
  builder.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{dotResult});

  // After the K-loop — store result C tile
  builder.setInsertionPointAfter(forOp);
  auto finalAcc = forOp.getResult(0);  // f32 accumulator

  // Cast f32 accumulator to output type if needed
  mlir::Value storeVal = finalAcc;
  if (cElemType != f32Type) {
    auto cTileType = mlir::RankedTensorType::get({blockM, blockN}, cElemType);
    if (mlir::isa<mlir::FloatType>(cElemType)) {
      storeVal = builder.create<mlir::arith::TruncFOp>(loc, cTileType, finalAcc);
    } else if (mlir::isa<mlir::IntegerType>(cElemType)) {
      storeVal = builder.create<mlir::arith::FPToSIOp>(loc, cTileType, finalAcc);
    }
  }

  // Compute C pointers: c_ptr + mIndices * N + nIndices
  auto mExpandedC = builder.create<mlir::triton::ExpandDimsOp>(loc, mIndices, 1);  // [BM, 1]
  auto nExpandedC = builder.create<mlir::triton::ExpandDimsOp>(loc, nIndices, 0);  // [1, BN]

  auto i32BmBnType = mlir::RankedTensorType::get({blockM, blockN}, i32Type);
  auto nSplatC = builder.create<mlir::triton::SplatOp>(loc, mlir::RankedTensorType::get({blockM, 1}, i32Type), nConst);
  auto mTimesNC = builder.create<mlir::arith::MulIOp>(loc, mExpandedC, nSplatC);
  auto mTimesNCBroadcast = builder.create<mlir::triton::BroadcastOp>(loc, i32BmBnType, mTimesNC);
  auto nBroadcastC = builder.create<mlir::triton::BroadcastOp>(loc, i32BmBnType, nExpandedC);
  auto cOffsets = builder.create<mlir::arith::AddIOp>(loc, mTimesNCBroadcast, nBroadcastC);

  auto cPtrTensorType = mlir::RankedTensorType::get({blockM, blockN},
      mlir::triton::PointerType::get(cElemType, 1));
  auto cSplat = builder.create<mlir::triton::SplatOp>(loc, cPtrTensorType, cPtr);
  auto cPtrs = builder.create<mlir::triton::AddPtrOp>(loc, cPtrTensorType, cSplat, cOffsets);

  // Create 2D mask for C tile: mIndices < M && nIndices < N
  auto mConstC = builder.create<mlir::arith::ConstantIntOp>(loc, M, 32);
  auto nConstC = builder.create<mlir::arith::ConstantIntOp>(loc, N, 32);
  auto mConstSplatC = builder.create<mlir::triton::SplatOp>(loc, i32BmType, mConstC);
  auto nConstSplatC = builder.create<mlir::triton::SplatOp>(loc, i32BnType, nConstC);
  auto mMask1DC = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt, mIndices, mConstSplatC);
  auto nMask1DC = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt, nIndices, nConstSplatC);
  auto i1BmBnType = mlir::RankedTensorType::get({blockM, blockN}, i1Type);
  auto mMaskExpC = builder.create<mlir::triton::ExpandDimsOp>(loc, mMask1DC, 1);
  auto nMaskExpC = builder.create<mlir::triton::ExpandDimsOp>(loc, nMask1DC, 0);
  auto mMask2DC = builder.create<mlir::triton::BroadcastOp>(loc, i1BmBnType, mMaskExpC);
  auto nMask2DC = builder.create<mlir::triton::BroadcastOp>(loc, i1BmBnType, nMaskExpC);
  auto cMask = builder.create<mlir::arith::AndIOp>(loc, mMask2DC, nMask2DC);

  builder.create<mlir::triton::StoreOp>(loc, cPtrs, storeVal, cMask,
                                         mlir::triton::CacheModifier::NONE,
                                         mlir::triton::EvictionPolicy::NORMAL);

  sd_printf("TritonIRBuilder: emitted matmul kernel M=%d N=%d K=%d BM=%d BN=%d BK=%d\n",
            M, N, K, blockM, blockN, blockK);
}

// ─── Fused attention (Flash Attention) emission ─────────────────────────────

void TritonIRBuilder::emitFusedAttentionKernel(mlir::OpBuilder& builder, mlir::Location loc,
                                                mlir::Value qPtr, mlir::Value kPtr,
                                                mlir::Value vPtr, mlir::Value outPtr,
                                                int batchSize, int numHeads, int seqQ, int seqK,
                                                int headDim, float scale,
                                                int blockM, int blockN) {
  auto f32Type = builder.getF32Type();
  auto i32Type = builder.getI32Type();

  // Triton requires all tensor dimensions (MakeRangeOp) to be power-of-2.
  // Round headDim up and use masking for the padded region.
  int headDimPadded = headDim;
  if (headDimPadded > 0 && (headDimPadded & (headDimPadded - 1)) != 0) {
    int p = 1;
    while (p < headDimPadded) p <<= 1;
    headDimPadded = p;
  }
  bool needsHdMask = (headDimPadded != headDim);

  // Program IDs: pid0 = batch * num_heads, pid1 = query tile index
  auto pid0 = builder.create<mlir::triton::GetProgramIdOp>(
      loc, i32Type, mlir::triton::ProgramIDDim::X);
  auto pid1 = builder.create<mlir::triton::GetProgramIdOp>(
      loc, i32Type, mlir::triton::ProgramIDDim::Y);

  // Decompose pid0 into batch and head indices
  auto numHeadsConst = builder.create<mlir::arith::ConstantIntOp>(loc, numHeads, 32);
  auto headIdx = builder.create<mlir::arith::RemSIOp>(loc, pid0, numHeadsConst);
  auto batchIdx = builder.create<mlir::arith::DivSIOp>(loc, pid0, numHeadsConst);

  // Query tile offset
  auto blockMConst = builder.create<mlir::arith::ConstantIntOp>(loc, blockM, 32);
  auto qOffset = builder.create<mlir::arith::MulIOp>(loc, pid1, blockMConst);

  // Create range vectors — use headDimPadded (power-of-2) for tensor sizes
  auto i32BmType = mlir::RankedTensorType::get({blockM}, i32Type);
  auto i32BnType = mlir::RankedTensorType::get({blockN}, i32Type);
  auto i32HdType = mlir::RankedTensorType::get({headDimPadded}, i32Type);

  auto rangeM = builder.create<mlir::triton::MakeRangeOp>(loc, i32BmType, 0, blockM);
  auto rangeN = builder.create<mlir::triton::MakeRangeOp>(loc, i32BnType, 0, blockN);
  auto rangeHd = builder.create<mlir::triton::MakeRangeOp>(loc, i32HdType, 0, headDimPadded);

  auto splatQOffset = builder.create<mlir::triton::SplatOp>(loc, i32BmType, qOffset);
  auto qIndices = builder.create<mlir::arith::AddIOp>(loc, splatQOffset, rangeM);

  // Compute base offset into Q/K/V/Out buffers:
  // Layout is [batch, heads, seq, headDim] (BHSD)
  // base = (batchIdx * numHeads * seqLen * headDim) + (headIdx * seqLen * headDim)
  auto seqQConst = builder.create<mlir::arith::ConstantIntOp>(loc, seqQ, 32);
  auto seqKConst = builder.create<mlir::arith::ConstantIntOp>(loc, seqK, 32);
  auto headDimConst = builder.create<mlir::arith::ConstantIntOp>(loc, headDim, 32);

  // Q base: batch * numHeads * seqQ * headDim + head * seqQ * headDim
  auto qStride0 = builder.create<mlir::arith::MulIOp>(loc, numHeadsConst,
      builder.create<mlir::arith::MulIOp>(loc, seqQConst, headDimConst));
  auto qStride1 = builder.create<mlir::arith::MulIOp>(loc, seqQConst, headDimConst);
  auto qBase = builder.create<mlir::arith::AddIOp>(loc,
      builder.create<mlir::arith::MulIOp>(loc, batchIdx, qStride0),
      builder.create<mlir::arith::MulIOp>(loc, headIdx, qStride1));

  // K/V base: batch * numHeads * seqK * headDim + head * seqK * headDim
  auto kvStride0 = builder.create<mlir::arith::MulIOp>(loc, numHeadsConst,
      builder.create<mlir::arith::MulIOp>(loc, seqKConst, headDimConst));
  auto kvStride1 = builder.create<mlir::arith::MulIOp>(loc, seqKConst, headDimConst);
  auto kvBase = builder.create<mlir::arith::AddIOp>(loc,
      builder.create<mlir::arith::MulIOp>(loc, batchIdx, kvStride0),
      builder.create<mlir::arith::MulIOp>(loc, headIdx, kvStride1));

  // Load Q tile [BLOCK_M, headDim]
  // Q pointer offsets: qBase + qIndices[:, None] * headDim + rangeHd[None, :]
  auto qMExpanded = builder.create<mlir::triton::ExpandDimsOp>(loc, qIndices, 1);  // [BM, 1]
  auto hdExpanded = builder.create<mlir::triton::ExpandDimsOp>(loc, rangeHd, 0);   // [1, HD]

  auto i32BmHdType = mlir::RankedTensorType::get({blockM, headDimPadded}, i32Type);
  auto f32BmHdType = mlir::RankedTensorType::get({blockM, headDimPadded}, f32Type);
  auto hdSplat = builder.create<mlir::triton::SplatOp>(loc,
      mlir::RankedTensorType::get({blockM, 1}, i32Type), headDimConst);
  auto qRowOffsets = builder.create<mlir::arith::MulIOp>(loc, qMExpanded, hdSplat);
  auto qRowBroadcast = builder.create<mlir::triton::BroadcastOp>(loc, i32BmHdType, qRowOffsets);
  auto hdBroadcast = builder.create<mlir::triton::BroadcastOp>(loc, i32BmHdType, hdExpanded);
  auto qOffsets2D = builder.create<mlir::arith::AddIOp>(loc, qRowBroadcast, hdBroadcast);

  auto qBaseSplat = builder.create<mlir::triton::SplatOp>(loc, i32BmHdType, qBase);
  auto qFinalOffsets = builder.create<mlir::arith::AddIOp>(loc, qBaseSplat, qOffsets2D);

  // Derive pointer types from actual arguments (NOT hardcoded f32)
  auto qPtrType = mlir::cast<mlir::triton::PointerType>(qPtr.getType());
  auto kPtrTypeAttn = mlir::cast<mlir::triton::PointerType>(kPtr.getType());
  auto vPtrTypeAttn = mlir::cast<mlir::triton::PointerType>(vPtr.getType());
  auto outPtrTypeAttn = mlir::cast<mlir::triton::PointerType>(outPtr.getType());
  auto qPtrTensorType = mlir::RankedTensorType::get({blockM, headDimPadded}, qPtrType);
  auto qSplat = builder.create<mlir::triton::SplatOp>(loc, qPtrTensorType, qPtr);
  auto qPtrs = builder.create<mlir::triton::AddPtrOp>(loc, qPtrTensorType, qSplat, qFinalOffsets);

  // Q mask: qIndices < seqQ (AND rangeHd < headDim if padded)
  auto i1Type = builder.getI1Type();
  auto seqQSplat = builder.create<mlir::triton::SplatOp>(loc, i32BmType, seqQConst);
  auto qMask1D = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt,
      qIndices, seqQSplat);
  auto qMaskExp = builder.create<mlir::triton::ExpandDimsOp>(loc, qMask1D, 1);  // [BM, 1]
  auto i1BmHdType = mlir::RankedTensorType::get({blockM, headDimPadded}, i1Type);
  auto qMask2D_row = builder.create<mlir::triton::BroadcastOp>(loc, i1BmHdType, qMaskExp);
  mlir::Value qMask2D = qMask2D_row;
  if (needsHdMask) {
    auto headDimSplatHd = builder.create<mlir::triton::SplatOp>(loc, i32HdType, headDimConst);
    auto hdMask1D = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt,
        rangeHd, headDimSplatHd);
    auto hdMaskExp = builder.create<mlir::triton::ExpandDimsOp>(loc, hdMask1D, 0);  // [1, HD]
    auto hdMask2DBm = builder.create<mlir::triton::BroadcastOp>(loc, i1BmHdType, hdMaskExp);
    qMask2D = builder.create<mlir::arith::AndIOp>(loc, qMask2D_row, hdMask2DBm);
  }

  mlir::Value qPtrsVal = qPtrs;
  auto qLoadedRaw = builder.create<mlir::triton::LoadOp>(loc,
      qPtrsVal, qMask2D, mlir::Value(),
      mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL, false);
  // Cast Q to f32 for computation
  auto qLoaded = castTo(builder, loc, qLoadedRaw, f32Type);

  // Apply scale to Q: q_scaled = q * scale
  auto scaleSplat = splatConstantF32(builder, loc, f32BmHdType, scale);
  auto qScaled = builder.create<mlir::arith::MulFOp>(loc, qLoaded, scaleSplat);

  // Initialize accumulators for online softmax:
  // acc = zeros([BLOCK_M, headDim]) — accumulated weighted values
  // m_i = splat(-inf, [BLOCK_M]) — running max
  // l_i = zeros([BLOCK_M]) — running sum of exp
  auto f32BmType = mlir::RankedTensorType::get({blockM}, f32Type);
  auto accInit = splatConstantF32(builder, loc, f32BmHdType, 0.0f);
  auto mInit = splatConstantF32(builder, loc, f32BmType, -3.4028235e+38f);
  auto lInit = splatConstantF32(builder, loc, f32BmType, 0.0f);

  // K-V loop: for j in range(0, seqK, BLOCK_N) — i32 bounds (Triton convention)
  auto jStart = builder.create<mlir::arith::ConstantIntOp>(loc, 0, 32);
  auto jEnd = builder.create<mlir::arith::ConstantIntOp>(loc, seqK, 32);
  auto jStep = builder.create<mlir::arith::ConstantIntOp>(loc, blockN, 32);

  auto forOp = builder.create<mlir::scf::ForOp>(
      loc, jStart, jEnd, jStep,
      mlir::ValueRange{accInit, mInit, lInit});

  // Inside KV loop
  builder.setInsertionPointToStart(forOp.getBody());
  auto jIdxI32 = forOp.getInductionVar();  // i32 induction variable
  auto accIter = forOp.getBody()->getArgument(1);
  auto mIter = forOp.getBody()->getArgument(2);
  auto lIter = forOp.getBody()->getArgument(3);

  // Compute K indices for this tile
  auto splatJOffset = builder.create<mlir::triton::SplatOp>(loc, i32BnType, jIdxI32);
  auto kIndices = builder.create<mlir::arith::AddIOp>(loc, splatJOffset, rangeN);

  // Load K tile [BLOCK_N, headDim]
  auto kNExpanded = builder.create<mlir::triton::ExpandDimsOp>(loc, kIndices, 1);  // [BN, 1]
  auto hdExpandedK = builder.create<mlir::triton::ExpandDimsOp>(loc, rangeHd, 0);  // [1, HD]

  auto i32BnHdType = mlir::RankedTensorType::get({blockN, headDimPadded}, i32Type);
  auto f32BnHdType = mlir::RankedTensorType::get({blockN, headDimPadded}, f32Type);
  auto hdSplatK = builder.create<mlir::triton::SplatOp>(loc,
      mlir::RankedTensorType::get({blockN, 1}, i32Type), headDimConst);
  auto kRowOffsets = builder.create<mlir::arith::MulIOp>(loc, kNExpanded, hdSplatK);
  auto kRowBroadcast = builder.create<mlir::triton::BroadcastOp>(loc, i32BnHdType, kRowOffsets);
  auto hdBroadcastK = builder.create<mlir::triton::BroadcastOp>(loc, i32BnHdType, hdExpandedK);
  auto kOffsets2D = builder.create<mlir::arith::AddIOp>(loc, kRowBroadcast, hdBroadcastK);

  auto kvBaseSplat = builder.create<mlir::triton::SplatOp>(loc, i32BnHdType, kvBase);
  auto kFinalOffsets = builder.create<mlir::arith::AddIOp>(loc, kvBaseSplat, kOffsets2D);

  auto kPtrTensorType = mlir::RankedTensorType::get({blockN, headDimPadded}, kPtrTypeAttn);
  auto kSplat = builder.create<mlir::triton::SplatOp>(loc, kPtrTensorType, kPtr);
  auto kPtrs = builder.create<mlir::triton::AddPtrOp>(loc, kPtrTensorType, kSplat, kFinalOffsets);

  // K mask: kIndices < seqK (AND rangeHd < headDim if padded)
  auto seqKSplat = builder.create<mlir::triton::SplatOp>(loc, i32BnType, seqKConst);
  auto kMask1D = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt,
      kIndices, seqKSplat);
  auto kMaskExp = builder.create<mlir::triton::ExpandDimsOp>(loc, kMask1D, 1);
  auto i1BnHdType = mlir::RankedTensorType::get({blockN, headDimPadded}, i1Type);
  auto kMask2D_row = builder.create<mlir::triton::BroadcastOp>(loc, i1BnHdType, kMaskExp);
  mlir::Value kMask2D = kMask2D_row;
  if (needsHdMask) {
    auto headDimSplatHdK = builder.create<mlir::triton::SplatOp>(loc, i32HdType, headDimConst);
    auto hdMask1DK = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt,
        rangeHd, headDimSplatHdK);
    auto hdMaskExpK = builder.create<mlir::triton::ExpandDimsOp>(loc, hdMask1DK, 0);  // [1, HD]
    auto hdMask2DBn = builder.create<mlir::triton::BroadcastOp>(loc, i1BnHdType, hdMaskExpK);
    kMask2D = builder.create<mlir::arith::AndIOp>(loc, kMask2D_row, hdMask2DBn);
  }

  mlir::Value kPtrsVal = kPtrs;
  auto kLoadedRaw = builder.create<mlir::triton::LoadOp>(loc,
      kPtrsVal, kMask2D, mlir::Value(),
      mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL, false);
  // Cast K to f32 for computation
  auto kLoaded = castTo(builder, loc, kLoadedRaw, f32Type);

  // QK^T = dot(q_scaled [BM, HD], k^T [HD, BN]) -> [BM, BN]
  auto transposeOrder = builder.getDenseI32ArrayAttr({1, 0});
  auto kTransposed = builder.create<mlir::triton::TransOp>(loc, kLoaded, transposeOrder);

  auto f32BmBnType = mlir::RankedTensorType::get({blockM, blockN}, f32Type);
  auto qkZeroInit = splatConstantF32(builder, loc, f32BmBnType, 0.0f);
  auto qk = builder.create<mlir::triton::DotOp>(
      loc, f32BmBnType, qScaled, kTransposed, qkZeroInit,
      mlir::triton::InputPrecision::TF32, /*maxNumImpreciseAcc=*/0);

  // Apply key mask: set qk to -inf where kIndices >= seqK
  auto negInfSplat = splatConstantF32(builder, loc, f32BmBnType, -3.4028235e+38f);
  auto kMask1DExp = builder.create<mlir::triton::ExpandDimsOp>(loc, kMask1D, 0);  // [1, BN]
  auto i1BmBnType = mlir::RankedTensorType::get({blockM, blockN}, i1Type);
  auto kMaskBmBn = builder.create<mlir::triton::BroadcastOp>(loc, i1BmBnType, kMask1DExp);
  auto qkMasked = builder.create<mlir::arith::SelectOp>(loc, kMaskBmBn, qk, negInfSplat);

  // Online softmax update:
  // m_new = max(m_i, row_max(qk))
  // correction = exp(m_i - m_new)
  // p = exp(qk - splat(m_new))
  // l_i = l_i * correction + row_sum(p)
  // acc = acc * splat(correction) + dot(p, V)

  // row_max(qk) -> reduce along axis 1
  mlir::Value qkMaskedVal = qkMasked;
  auto rowMaxOp = builder.create<mlir::triton::ReduceOp>(loc,
      mlir::ValueRange{qkMaskedVal}, /*axis=*/1);
  {
    auto& region = rowMaxOp.getCombineOp();
    auto* block = builder.createBlock(&region, {}, {f32Type, f32Type}, {loc, loc});
    builder.setInsertionPointToEnd(block);
    auto maxed = builder.create<mlir::arith::MaximumFOp>(loc, block->getArgument(0), block->getArgument(1));
    builder.create<mlir::triton::ReduceReturnOp>(loc, mlir::ValueRange{maxed.getResult()});
  }
  builder.setInsertionPointAfter(rowMaxOp);
  auto rowMax = rowMaxOp->getResult(0);  // [BM]

  // m_new = max(m_i, rowMax)
  auto mNew = builder.create<mlir::arith::MaximumFOp>(loc, mIter, rowMax);

  // correction = exp(m_i - m_new)
  auto mDiff = builder.create<mlir::arith::SubFOp>(loc, mIter, mNew);
  auto correction = builder.create<mlir::math::ExpOp>(loc, mDiff);

  // p = exp(qk - splat(m_new)) -> [BM, BN]
  auto mNewSplat = builder.create<mlir::triton::ExpandDimsOp>(loc, mNew, 1);  // [BM, 1]
  auto mNewBroadcast = builder.create<mlir::triton::BroadcastOp>(loc, f32BmBnType, mNewSplat);
  auto qkShifted = builder.create<mlir::arith::SubFOp>(loc, qkMasked, mNewBroadcast);
  auto p = builder.create<mlir::math::ExpOp>(loc, qkShifted);

  // row_sum(p) -> reduce along axis 1
  auto rowSumOp = builder.create<mlir::triton::ReduceOp>(loc,
      mlir::ValueRange{p.getResult()}, /*axis=*/1);
  {
    auto& region = rowSumOp.getCombineOp();
    auto* block = builder.createBlock(&region, {}, {f32Type, f32Type}, {loc, loc});
    builder.setInsertionPointToEnd(block);
    auto summed = builder.create<mlir::arith::AddFOp>(loc, block->getArgument(0), block->getArgument(1));
    builder.create<mlir::triton::ReduceReturnOp>(loc, mlir::ValueRange{summed.getResult()});
  }
  builder.setInsertionPointAfter(rowSumOp);
  auto rowSum = rowSumOp->getResult(0);  // [BM]

  // l_new = l_i * correction + rowSum
  auto lScaled = builder.create<mlir::arith::MulFOp>(loc, lIter, correction);
  auto lNew = builder.create<mlir::arith::AddFOp>(loc, lScaled, rowSum);

  // Load V tile [BN, headDim]
  auto vPtrTensorType = mlir::RankedTensorType::get({blockN, headDimPadded}, vPtrTypeAttn);
  auto vSplat = builder.create<mlir::triton::SplatOp>(loc, vPtrTensorType, vPtr);
  auto vPtrs = builder.create<mlir::triton::AddPtrOp>(loc, vPtrTensorType, vSplat, kFinalOffsets);

  mlir::Value vPtrsVal = vPtrs;
  auto vLoadedRaw = builder.create<mlir::triton::LoadOp>(loc,
      vPtrsVal, kMask2D, mlir::Value(),
      mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL, false);
  // Cast V to f32 for computation
  auto vLoaded = castTo(builder, loc, vLoadedRaw, f32Type);

  // acc_new = acc * splat(correction) + dot(p, V)
  // correction is [BM], need to broadcast to [BM, HD]
  auto correctionExp = builder.create<mlir::triton::ExpandDimsOp>(loc, correction, 1);  // [BM, 1]
  auto correctionBroadcast = builder.create<mlir::triton::BroadcastOp>(loc, f32BmHdType, correctionExp);
  auto accScaled = builder.create<mlir::arith::MulFOp>(loc, accIter, correctionBroadcast);

  // dot(p[BM,BN], V[BN,HD]) -> [BM, HD]
  auto pv = builder.create<mlir::triton::DotOp>(
      loc, f32BmHdType, p, vLoaded, accScaled,
      mlir::triton::InputPrecision::TF32, /*maxNumImpreciseAcc=*/0);

  // Yield for next iteration
  mlir::Value pvVal = pv, mNewVal = mNew, lNewVal = lNew;
  mlir::Value yieldVals[] = {pvVal, mNewVal, lNewVal};
  builder.create<mlir::scf::YieldOp>(loc, mlir::ValueRange(yieldVals));

  // After the KV loop
  builder.setInsertionPointAfter(forOp);
  auto finalAcc = forOp.getResult(0);   // [BM, HD]
  auto finalL = forOp.getResult(2);     // [BM]

  // Normalize: result = acc / splat(l_i)
  auto lExp = builder.create<mlir::triton::ExpandDimsOp>(loc, finalL, 1);  // [BM, 1]
  auto lBroadcast = builder.create<mlir::triton::BroadcastOp>(loc, f32BmHdType, lExp);
  auto normalized = builder.create<mlir::arith::DivFOp>(loc, finalAcc, lBroadcast);

  // Store output [BM, headDim]
  // Out base is same as Q base (same layout)
  auto outBaseSplat = builder.create<mlir::triton::SplatOp>(loc, i32BmHdType, qBase);
  auto outFinalOffsets = builder.create<mlir::arith::AddIOp>(loc, outBaseSplat, qOffsets2D);

  auto outPtrTensorTypeAttn = mlir::RankedTensorType::get({blockM, headDimPadded}, outPtrTypeAttn);
  auto outSplatPtr = builder.create<mlir::triton::SplatOp>(loc, outPtrTensorTypeAttn, outPtr);
  auto outPtrs = builder.create<mlir::triton::AddPtrOp>(loc, outPtrTensorTypeAttn, outSplatPtr, outFinalOffsets);

  // Cast normalized f32 result to output element type
  mlir::Value outStoreVal = castTo(builder, loc, normalized, outPtrTypeAttn.getPointeeType());
  builder.create<mlir::triton::StoreOp>(loc, outPtrs, outStoreVal, qMask2D,
                                         mlir::triton::CacheModifier::NONE,
                                         mlir::triton::EvictionPolicy::NORMAL);

  sd_printf("TritonIRBuilder: emitted fused attention kernel batch=%d heads=%d seqQ=%d seqK=%d "
            "headDim=%d scale=%f BM=%d BN=%d\n",
            batchSize, numHeads, seqQ, seqK, headDim, scale, blockM, blockN);
}

// ─── Module construction ────────────────────────────────────────────────────

TritonIRModule TritonIRBuilder::buildModule(NativeSlot* slots, int startSlot, int endSlot,
                                            int totalSlots,
                                            NDArray** externalInputs, int numExternalInputs,
                                            NDArray** outputSlots, int totalOutputSlots,
                                            int* requestedOutputSlotIndices,
                                            int numRequestedOutputs) {
  TritonIRModule result;
  int segSize = endSlot - startSlot + 1;

  // Pre-compilation feasibility check — bail before MLIR allocation if infeasible
  auto analysis = analyzeSegment(slots, startSlot, endSlot, totalSlots,
                                  externalInputs, numExternalInputs,
                                  outputSlots, totalOutputSlots,
                                  requestedOutputSlotIndices, numRequestedOutputs);
  if (!analysis.canCompile) {
    sd_printf("TritonIRBuilder::buildModule: segment [%d-%d] failed pre-check: %s\n",
              startSlot, endSlot, analysis.failureReason.c_str());
    return result;  // result.valid = false
  }

  // Route small, pure matmul segments to the dedicated 2D tiled builder.
  auto pattern = analysis.pattern;
  bool isSmallPureMatmul = (pattern == SegmentKernelPattern::MATMUL_2D ||
                             pattern == SegmentKernelPattern::MATMUL_EPILOGUE) && segSize <= 10;
  if (isSmallPureMatmul) {
    return buildMatmulModule(slots, startSlot, endSlot, totalSlots,
                              externalInputs, numExternalInputs,
                              outputSlots, totalOutputSlots,
                              requestedOutputSlotIndices, numRequestedOutputs);
  }

  // Mixed segments with non-element-wise ops → sectioned cooperative kernel.
  // This handles mega-segments (WHOLE_GRAPH) and segments containing matmul,
  // attention, data movement, convolution, or permute ops that need their own
  // grid mapping and cannot be fused into the 1D element-wise skeleton.
  {
    bool hasNonElementwiseOps = false;
    for (int i = startSlot; i <= endSlot; i++) {
      auto cat = getOpCategory(slots[i].opName);
      if (cat == TritonOpCategory::MATMUL || cat == TritonOpCategory::FUSED_ATTENTION ||
          cat == TritonOpCategory::DATA_MOVEMENT || cat == TritonOpCategory::CONVOLUTION) {
        hasNonElementwiseOps = true;
        break;
      }
      if (cat == TritonOpCategory::SHAPE_MANIPULATION) {
        std::string opLower = slots[i].opName;
        std::transform(opLower.begin(), opLower.end(), opLower.begin(), ::tolower);
        if (opLower == "permute" || opLower == "transpose") {
          hasNonElementwiseOps = true;
          break;
        }
      }
    }
    if (hasNonElementwiseOps) {
      sd_debug("TritonIRBuilder::buildModule: segment [%d-%d] (%d ops) has non-elementwise ops, "
                "routing to buildSectionedModule()\n", startSlot, endSlot, segSize);
      return buildSectionedModule(slots, startSlot, endSlot, totalSlots,
                                   externalInputs, numExternalInputs,
                                   outputSlots, totalOutputSlots,
                                   requestedOutputSlotIndices, numRequestedOutputs);
    }
  }

  // Pure element-wise/reduction/normalization/cast/comparison/logical/ternary/identity segments
  // → existing 1D skeleton (already works)
  sd_printf("TritonIRBuilder::buildModule: segment [%d-%d] (%d ops), pattern=%d\n",
            startSlot, endSlot, segSize, static_cast<int>(pattern));
  result.kernelName = generateKernelName(slots, startSlot, endSlot);
  sd_printf("TritonIRBuilder::buildModule: kernel name generated, collecting categories...\n");

  // Build cached shape info map for shape resolution when outputSlots may be released
  std::unordered_map<int, const LongType*> cachedShapeInfoMap;
  for (int i = 0; i < totalSlots; i++) {
    if (slots[i].shapeCacheValid && !slots[i].cachedOutputShapes.empty()) {
      for (int o = 0; o < slots[i].numOutputs; o++) {
        int outIdx = slots[i].outputSlotIndices[o];
        if (outIdx >= 0 && o < static_cast<int>(slots[i].cachedOutputShapes.size()) &&
            slots[i].cachedOutputShapes[o] != nullptr) {
          cachedShapeInfoMap[outIdx] = slots[i].cachedOutputShapes[o];
        }
      }
    }
  }

  // Shape resolution helpers (cached shape info first, then live outputSlots)
  auto resolveShapeLocal = [&](int srcIdx) -> std::vector<LongType> {
    if (srcIdx < 0) {
      int extIdx = -(srcIdx + 1);
      if (extIdx < numExternalInputs && externalInputs && externalInputs[extIdx]) {
        auto& arr = *externalInputs[extIdx];
        std::vector<LongType> s(arr.rankOf());
        for (int d = 0; d < arr.rankOf(); d++) s[d] = arr.sizeAt(d);
        return s;
      }
      return {};
    }
    auto cit = cachedShapeInfoMap.find(srcIdx);
    if (cit != cachedShapeInfoMap.end() && cit->second) {
      LongType rank = shape::rank(cit->second);
      std::vector<LongType> s(rank);
      for (int d = 0; d < rank; d++) s[d] = shape::shapeOf(cit->second)[d];
      return s;
    }
    if (srcIdx < totalOutputSlots && outputSlots && outputSlots[srcIdx]) {
      auto& arr = *outputSlots[srcIdx];
      std::vector<LongType> s(arr.rankOf());
      for (int d = 0; d < arr.rankOf(); d++) s[d] = arr.sizeAt(d);
      return s;
    }
    return {};
  };

  auto resolveDtypeLocal = [&](int srcIdx) -> DataType {
    if (srcIdx < 0) {
      int extIdx = -(srcIdx + 1);
      if (extIdx < numExternalInputs && externalInputs && externalInputs[extIdx])
        return externalInputs[extIdx]->dataType();
      return FLOAT32;
    }
    auto cit = cachedShapeInfoMap.find(srcIdx);
    if (cit != cachedShapeInfoMap.end() && cit->second)
      return ArrayOptions::dataType(cit->second);
    if (srcIdx < totalOutputSlots && outputSlots && outputSlots[srcIdx])
      return outputSlots[srcIdx]->dataType();
    return FLOAT32;
  };

  // Collect op categories and shapes for tile config
  std::vector<TritonOpCategory> categories;
  std::vector<std::vector<LongType>> shapes;

  for (int i = startSlot; i <= endSlot; i++) {
    auto cat = getOpCategory(slots[i].opName);
    // Every op must be in the table. getOpCategory() throws if missing.
    categories.push_back(cat);

    if (slots[i].numOutputs > 0) {
      int outIdx = slots[i].outputSlotIndices[0];
      shapes.push_back(resolveShapeLocal(outIdx));
    } else {
      shapes.push_back({});
    }
  }
  sd_printf("TritonIRBuilder::buildModule: collected %d categories, selecting tile config...\n",
            (int)categories.size());

  // Select tile configuration
  int blockSize, numWarps, numStages;
  selectTileConfig(categories, shapes, blockSize, numWarps, numStages);
  result.numWarps = numWarps;
  result.numStages = numStages;

  // Create MLIR context and register dialects
  auto mlirContext = new mlir::MLIRContext();
  mlirContext->loadDialect<mlir::triton::TritonDialect>();
  mlirContext->loadDialect<mlir::arith::ArithDialect>();
  mlirContext->loadDialect<mlir::math::MathDialect>();
  mlirContext->loadDialect<mlir::scf::SCFDialect>();

  mlir::OpBuilder builder(mlirContext);
  auto loc = builder.getUnknownLoc();

  // Create module
  auto moduleOp = mlir::ModuleOp::create(loc);
  builder.setInsertionPointToEnd(moduleOp.getBody());

  // ── Collect unique buffer references ──
  std::unordered_set<int> internalSlotOutputs;
  for (int i = startSlot; i <= endSlot; i++) {
    for (int o = 0; o < slots[i].numOutputs; o++) {
      internalSlotOutputs.insert(slots[i].outputSlotIndices[o]);
    }
  }

  // Inputs: external inputs or outputs from slots BEFORE this segment
  std::vector<TritonKernelArg> inputArgs;
  std::unordered_set<int> seenInputs;

  for (int i = startSlot; i <= endSlot; i++) {
    for (int inp = 0; inp < slots[i].numInputs; inp++) {
      int srcIdx = slots[i].inputSourceIndices[inp];
      if (seenInputs.count(srcIdx)) continue;
      seenInputs.insert(srcIdx);

      if (srcIdx < 0) {
        int extIdx = -(srcIdx + 1);
        if (extIdx < numExternalInputs && externalInputs[extIdx]) {
          TritonKernelArg arg;
          arg.slotIndex = srcIdx;
          arg.outputIndex = 0;
          arg.isOutput = false;
          arg.dtype = externalInputs[extIdx]->dataType();
          auto& arr = *externalInputs[extIdx];
          for (int d = 0; d < arr.rankOf(); d++) arg.shape.push_back(arr.sizeAt(d));
          inputArgs.push_back(arg);
        }
      } else if (!internalSlotOutputs.count(srcIdx)) {
        auto shape = resolveShapeLocal(srcIdx);
        auto dtype = resolveDtypeLocal(srcIdx);
        bool hasLiveArr = (srcIdx < totalOutputSlots && outputSlots && outputSlots[srcIdx]);
        if (hasLiveArr || !shape.empty()) {
          TritonKernelArg arg;
          arg.slotIndex = srcIdx;
          arg.outputIndex = 0;
          arg.isOutput = false;
          arg.dtype = dtype;
          arg.shape = shape;
          inputArgs.push_back(arg);
        }
      }
    }
  }

  // Outputs: only externally-visible outputs need kernel args.
  // Purely internal intermediates are SSA-forwarded — no global store needed.
  // Deduplicate: same output slot written by multiple ops only needs one kernel arg.
  auto externalOutputs = computeExternallyVisibleOutputs(
      slots, startSlot, endSlot, totalSlots,
      requestedOutputSlotIndices, numRequestedOutputs);

  std::vector<TritonKernelArg> outputArgs;
  {
    std::unordered_set<int> seenOutputSlots;
    int skippedInternal = 0;
    for (int i = startSlot; i <= endSlot; i++) {
      for (int o = 0; o < slots[i].numOutputs; o++) {
        int outIdx = slots[i].outputSlotIndices[o];
        if (outIdx < 0 || outIdx >= totalOutputSlots) continue;
        if (seenOutputSlots.count(outIdx)) continue;  // Deduplicate
        seenOutputSlots.insert(outIdx);
        if (!externalOutputs.count(outIdx)) {
          skippedInternal++;
          continue;  // Purely internal — SSA forwarded
        }

        TritonKernelArg arg;
        arg.slotIndex = outIdx;
        arg.outputIndex = o;
        arg.isOutput = true;
        if (outputSlots && outIdx < totalOutputSlots && outputSlots[outIdx]) {
          arg.dtype = outputSlots[outIdx]->dataType();
          auto& arr = *outputSlots[outIdx];
          for (int d = 0; d < arr.rankOf(); d++) arg.shape.push_back(arr.sizeAt(d));
        } else {
          // Fall back to cached shape info when live array is not available
          auto cit = cachedShapeInfoMap.find(outIdx);
          if (cit != cachedShapeInfoMap.end() && cit->second) {
            arg.dtype = ArrayOptions::dataType(cit->second);
            LongType rank = shape::rank(cit->second);
            for (int d = 0; d < rank; d++) arg.shape.push_back(shape::shapeOf(cit->second)[d]);
          }
        }
        outputArgs.push_back(arg);
      }
    }
    if (skippedInternal > 0) {
      sd_printf("TritonIRBuilder::buildModule: eliminated %d internal outputs, keeping %d external\n",
                skippedInternal, (int)outputArgs.size());
    }
  }

  // Combine: inputs first, then outputs
  result.args.insert(result.args.end(), inputArgs.begin(), inputArgs.end());
  result.args.insert(result.args.end(), outputArgs.begin(), outputArgs.end());

  int totalBufferArgs = static_cast<int>(result.args.size());
  bool useIndirectArgs = (totalBufferArgs + 1) > TRITON_DIRECT_ARG_LIMIT;  // +1 for n_elements

  sd_printf("TritonIRBuilder::buildModule: %d input args, %d output args, %d total buffer args%s\n",
            (int)inputArgs.size(), (int)outputArgs.size(), totalBufferArgs,
            useIndirectArgs ? " (INDIRECT arg passing)" : " (direct)");

  // ── Build function signature ──
  // Direct mode: each arg is a tt.ptr<dtype>, plus n_elements : i32
  // Indirect mode: (argArray : !tt.ptr<i64>, n_elements : i32) — all buffer pointers
  //   are packed into a device-side array of int64 (pointer-sized values).
  //   The kernel unpacks them with scalar loads: ptr_i = load(argArray + i*8)
  std::vector<mlir::Type> funcArgTypes;
  if (!useIndirectArgs) {
    for (auto& arg : result.args) {
      auto elemType = getMLIRType(builder, arg.dtype);
      funcArgTypes.push_back(mlir::triton::PointerType::get(elemType, 1));
    }
  } else {
    // Indirect: single pointer to array of i64 (each holding a buffer pointer)
    auto i64Type = builder.getI64Type();
    funcArgTypes.push_back(mlir::triton::PointerType::get(i64Type, 1));  // argArray*
  }
  funcArgTypes.push_back(builder.getI32Type());  // n_elements

  sd_printf("TritonIRBuilder::buildModule: creating MLIR function with %d params (%d buffer args)...\n",
            (int)funcArgTypes.size(), totalBufferArgs);

  auto funcType = builder.getFunctionType(funcArgTypes, {});
  auto funcOp = builder.create<mlir::triton::FuncOp>(loc, result.kernelName, funcType);
  funcOp.setPublic();

  auto* entryBlock = funcOp.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  // If using indirect args, unpack buffer pointers from the arg array.
  // argUnpacked[i] holds the mlir::Value for the i-th buffer pointer, equivalent
  // to what entryBlock->getArgument(i) would return in direct mode.
  std::vector<mlir::Value> argUnpacked;
  if (useIndirectArgs) {
    auto i64Type = builder.getI64Type();
    auto argArrayPtr = entryBlock->getArgument(0);  // !tt.ptr<i64>
    for (int a = 0; a < totalBufferArgs; a++) {
      // Compute pointer to argArray[a]: argArrayPtr + a
      auto idxConst = builder.create<mlir::arith::ConstantIntOp>(loc, a, 64);
      auto elemPtr = builder.create<mlir::triton::AddPtrOp>(
          loc, argArrayPtr.getType(), argArrayPtr, idxConst);

      // Scalar load: i64 value = *elemPtr
      auto rawVal = builder.create<mlir::triton::LoadOp>(
          loc, /*ptr=*/elemPtr,
          /*cache=*/mlir::triton::CacheModifier::NONE,
          /*evict=*/mlir::triton::EvictionPolicy::NORMAL,
          /*isVolatile=*/false);

      // inttoptr: i64 -> tt.ptr<elemType>
      auto& argDesc = result.args[a];
      auto elemType = getMLIRType(builder, argDesc.dtype);
      auto targetPtrType = mlir::triton::PointerType::get(elemType, 1);
      auto castPtr = builder.create<mlir::triton::IntToPtrOp>(loc, targetPtrType, rawVal);
      argUnpacked.push_back(castPtr);
    }
    sd_printf("TritonIRBuilder::buildModule: unpacked %d buffer pointers from indirect arg array\n",
              totalBufferArgs);
  }

  // Helper lambda: get the mlir::Value for buffer arg at index 'a'
  auto getBufferArg = [&](int a) -> mlir::Value {
    if (useIndirectArgs) {
      return argUnpacked[a];
    } else {
      return entryBlock->getArgument(a);
    }
  };

  sd_printf("TritonIRBuilder::buildModule: MLIR function created, building kernel body...\n");

  // ── Grid configuration ──
  bool hasMatmul = std::find(categories.begin(), categories.end(), TritonOpCategory::MATMUL) != categories.end();

  if (hasMatmul) {
    result.gridX = 1;
    result.gridY = 1;
    result.gridZ = 1;
    result.blockX = blockSize;
    result.blockY = 1;
    result.blockZ = 1;
  } else {
    result.gridX = 1;  // Set at launch: ceil(n_elements / BLOCK_SIZE)
    result.gridY = 1;
    result.gridZ = 1;
    result.blockX = blockSize;
    result.blockY = 1;
    result.blockZ = 1;
  }

  // ── Kernel body: 1D element-wise pattern ──
  //
  //   pid = tt.get_program_id(0)
  //   offset_base = pid * BLOCK_SIZE
  //   offsets = offset_base + tl.arange(0, BLOCK_SIZE)
  //   mask = offsets < n_elements
  //   [load inputs]
  //   [fused ops via SSA]
  //   [store outputs]

  auto i32Type = builder.getI32Type();
  auto f32Type = builder.getF32Type();
  auto i32TensorType = mlir::RankedTensorType::get({blockSize}, i32Type);
  auto f32TensorType = mlir::RankedTensorType::get({blockSize}, f32Type);
  auto i1TensorType = mlir::RankedTensorType::get({blockSize}, builder.getI1Type());

  auto nElementsArg = entryBlock->getArgument(funcArgTypes.size() - 1);

  // 2a: Prologue — pid, offsets, mask
  auto pid = builder.create<mlir::triton::GetProgramIdOp>(
      loc, i32Type, mlir::triton::ProgramIDDim::X);

  auto blockSizeConst = builder.create<mlir::arith::ConstantIntOp>(loc, blockSize, 32);
  auto offsetBase = builder.create<mlir::arith::MulIOp>(loc, pid, blockSizeConst);

  auto range = builder.create<mlir::triton::MakeRangeOp>(loc, i32TensorType, 0, blockSize);

  auto splatBase = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, offsetBase);
  auto offsets = builder.create<mlir::arith::AddIOp>(loc, splatBase, range);

  auto splatN = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, nElementsArg);
  auto mask = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::slt, offsets, splatN);

  // ── SSA value map: slotIndex/sourceIndex -> mlir::Value ──
  // This is the core fusion mechanism: ops share SSA values instead of going
  // through global memory stores/loads.
  std::unordered_map<int, mlir::Value> ssaValues;

  // Map: kernel arg index -> slotIndex for reverse lookup
  std::unordered_map<int, int> slotToArgIdx;
  for (int a = 0; a < static_cast<int>(result.args.size()); a++) {
    slotToArgIdx[result.args[a].slotIndex] = a;
  }

  // 2b: Load inputs — tt.load for each external input arg
  // Compute max output element count for broadcasting detection
  LongType maxOutputElements = 0;
  for (auto& oarg : outputArgs) {
    LongType oElems = 1;
    for (auto d : oarg.shape) oElems *= d;
    if (oElems > maxOutputElements) maxOutputElements = oElems;
  }

  for (int a = 0; a < static_cast<int>(inputArgs.size()); a++) {
    auto& arg = inputArgs[a];
    auto funcArg = getBufferArg(a);  // tt.ptr<elemType>

    auto elemType = getMLIRType(builder, arg.dtype);
    auto ptrType = mlir::triton::PointerType::get(elemType, 1);
    auto ptrTensorType = mlir::RankedTensorType::get({blockSize}, ptrType);
    auto dataTensorType = mlir::RankedTensorType::get({blockSize}, elemType);

    // Compute this input's total element count for broadcast-aware indexing
    LongType inputElements = 1;
    for (auto d : arg.shape) inputElements *= d;

    // If input is smaller than output, use modular indexing: offsets % inputSize
    // This handles broadcasting (e.g., [1,8] broadcast to [2,8])
    mlir::Value loadOffsets = offsets;
    mlir::Value loadMask = mask;
    if (inputElements > 0 && inputElements < maxOutputElements) {
      auto inputSizeConst = builder.create<mlir::arith::ConstantIntOp>(loc, static_cast<int>(inputElements), 32);
      auto splatInputSize = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, inputSizeConst);
      // offsets_mod = offsets % inputSize (unsigned remainder for non-negative indices)
      loadOffsets = builder.create<mlir::arith::RemUIOp>(loc, offsets, splatInputSize);
      // Mask still uses original offsets vs n_elements (output bounds), not input bounds
      sd_printf("TritonIRBuilder::buildModule: input arg %d (slot %d) uses broadcast indexing: "
                "%lld elements -> %lld output elements\n",
                a, arg.slotIndex, inputElements, maxOutputElements);
    }

    auto splatPtr = builder.create<mlir::triton::SplatOp>(loc, ptrTensorType, funcArg);
    auto ptrs = builder.create<mlir::triton::AddPtrOp>(loc, ptrTensorType, splatPtr, loadOffsets);
    mlir::Value ptrVal = ptrs.getResult();
    auto loaded = builder.create<mlir::triton::LoadOp>(loc,
                                                        /*ptr=*/ptrVal,
                                                        /*mask=*/loadMask,
                                                        /*other=*/mlir::Value(),
                                                        /*cache=*/mlir::triton::CacheModifier::NONE,
                                                        /*evict=*/mlir::triton::EvictionPolicy::NORMAL,
                                                        /*isVolatile=*/false);
    ssaValues[arg.slotIndex] = loaded;
  }

  // 2c: Fused op emission — iterate over slots, resolve inputs from ssaValues
  const auto& opTable = getOpTable();
  int catIdx = 0;
  int opsEmitted = 0;

  // Helper lambda: resolve source index to NDArray* for shape inspection
  auto resolveArr = [&](int srcIdx) -> NDArray* {
    if (srcIdx < 0) {
      int extIdx = -(srcIdx + 1);
      return (extIdx < numExternalInputs && externalInputs) ? externalInputs[extIdx] : nullptr;
    }
    return (srcIdx >= 0 && srcIdx < totalOutputSlots && outputSlots) ? outputSlots[srcIdx] : nullptr;
  };

  // Helper lambda: get kernel arg pointer for a given slot index
  auto getSlotArgPtr = [&](int slotIdx) -> mlir::Value {
    auto it = slotToArgIdx.find(slotIdx);
    if (it != slotToArgIdx.end()) {
      return getBufferArg(it->second);
    }
    return mlir::Value();
  };

  // Helper: load result back from output buffer into SSA for downstream consumers
  auto loadBackFromBuffer = [&](int outSlot, DataType /*dtype*/) -> mlir::Value {
    auto outArgPtr = getSlotArgPtr(outSlot);
    if (!outArgPtr) return mlir::Value();
    // Derive pointer type from actual MLIR arg (NOT from dtype parameter)
    auto ptrType = mlir::cast<mlir::triton::PointerType>(outArgPtr.getType());
    auto ptrTensorType = mlir::RankedTensorType::get({blockSize}, ptrType);
    auto splatPtr = builder.create<mlir::triton::SplatOp>(loc, ptrTensorType, outArgPtr);
    auto ptrs = builder.create<mlir::triton::AddPtrOp>(loc, ptrTensorType, splatPtr, offsets);
    return builder.create<mlir::triton::LoadOp>(loc, ptrs.getResult(), mask.getResult(),
        mlir::Value(), mlir::triton::CacheModifier::NONE,
        mlir::triton::EvictionPolicy::NORMAL, false);
  };

  for (int si = startSlot; si <= endSlot; si++, catIdx++) {
    auto& slot = slots[si];
    auto cat = categories[catIdx];
    auto it = opTable.find(slot.opName);
    if (it == opTable.end()) continue;
    const auto& mapping = it->second;
    opsEmitted++;

    if (cat == TritonOpCategory::BINARY_ELEMENTWISE) {
      // Binary: needs two inputs
      if (slot.numInputs < 2) {
        sd_printf("TritonIRBuilder: binary op '%s' at slot %d has < 2 inputs\n",
                  slot.opName.c_str(), si);
        continue;
      }

      int lhsSrc = slot.inputSourceIndices[0];
      int rhsSrc = slot.inputSourceIndices[1];

      auto lhsIt = ssaValues.find(lhsSrc);
      auto rhsIt = ssaValues.find(rhsSrc);

      if (lhsIt == ssaValues.end() || rhsIt == ssaValues.end()) {
        sd_printf("TritonIRBuilder: missing SSA value for binary op '%s' at slot %d "
                  "(lhs=%d:%s, rhs=%d:%s)\n",
                  slot.opName.c_str(), si,
                  lhsSrc, lhsIt != ssaValues.end() ? "found" : "MISSING",
                  rhsSrc, rhsIt != ssaValues.end() ? "found" : "MISSING");
        continue;
      }

      auto opResult = emitBinaryElementwise(builder, loc, mapping, lhsIt->second, rhsIt->second);

      // Store result SSA value for each output slot
      for (int o = 0; o < slot.numOutputs; o++) {
        ssaValues[slot.outputSlotIndices[o]] = opResult;
      }

    } else if (cat == TritonOpCategory::UNARY_ELEMENTWISE) {
      // Unary: needs one input
      if (slot.numInputs < 1) {
        sd_printf("TritonIRBuilder: unary op '%s' at slot %d has no inputs\n",
                  slot.opName.c_str(), si);
        continue;
      }

      int inputSrc = slot.inputSourceIndices[0];
      auto inputIt = ssaValues.find(inputSrc);
      if (inputIt == ssaValues.end()) {
        sd_printf("TritonIRBuilder: missing SSA value for unary op '%s' at slot %d (src=%d)\n",
                  slot.opName.c_str(), si, inputSrc);
        continue;
      }

      auto opResult = emitUnaryElementwise(builder, loc, mapping, slot, inputIt->second, blockSize);

      for (int o = 0; o < slot.numOutputs; o++) {
        ssaValues[slot.outputSlotIndices[o]] = opResult;
      }

    } else if (cat == TritonOpCategory::COMPARISON) {
      // Comparison: needs two inputs, produces bool tensor
      if (slot.numInputs < 2) {
        sd_printf("TritonIRBuilder: comparison op '%s' at slot %d has < 2 inputs\n",
                  slot.opName.c_str(), si);
        continue;
      }
      int lhsSrc = slot.inputSourceIndices[0];
      int rhsSrc = slot.inputSourceIndices[1];
      auto lhsIt = ssaValues.find(lhsSrc);
      auto rhsIt = ssaValues.find(rhsSrc);
      if (lhsIt == ssaValues.end() || rhsIt == ssaValues.end()) {
        sd_printf("TritonIRBuilder: missing SSA value for comparison op '%s' at slot %d\n",
                  slot.opName.c_str(), si);
        continue;
      }
      auto opResult = emitComparisonOp(builder, loc, slot.opName, lhsIt->second, rhsIt->second, blockSize);
      for (int o = 0; o < slot.numOutputs; o++) {
        ssaValues[slot.outputSlotIndices[o]] = opResult;
      }

    } else if (cat == TritonOpCategory::LOGICAL) {
      // Logical: 1 or 2 inputs depending on op
      if (slot.numInputs < 1) {
        sd_printf("TritonIRBuilder: logical op '%s' at slot %d has no inputs\n",
                  slot.opName.c_str(), si);
        continue;
      }
      int lhsSrc = slot.inputSourceIndices[0];
      auto lhsIt = ssaValues.find(lhsSrc);
      if (lhsIt == ssaValues.end()) {
        sd_printf("TritonIRBuilder: missing SSA value for logical op '%s' at slot %d\n",
                  slot.opName.c_str(), si);
        continue;
      }
      // For NOT ops, rhs is unused (emitLogicalOp handles it internally)
      mlir::Value rhsVal = lhsIt->second;  // dummy for unary
      if (slot.numInputs >= 2) {
        int rhsSrc = slot.inputSourceIndices[1];
        auto rhsIt = ssaValues.find(rhsSrc);
        if (rhsIt != ssaValues.end()) rhsVal = rhsIt->second;
      }
      auto opResult = emitLogicalOp(builder, loc, slot.opName, lhsIt->second, rhsVal, blockSize);
      for (int o = 0; o < slot.numOutputs; o++) {
        ssaValues[slot.outputSlotIndices[o]] = opResult;
      }

    } else if (cat == TritonOpCategory::TERNARY) {
      // Ternary: where/select needs 3 inputs (condition, true_val, false_val)
      if (slot.numInputs < 3) {
        sd_printf("TritonIRBuilder: ternary op '%s' at slot %d has < 3 inputs\n",
                  slot.opName.c_str(), si);
        continue;
      }
      int condSrc = slot.inputSourceIndices[0];
      int trueSrc = slot.inputSourceIndices[1];
      int falseSrc = slot.inputSourceIndices[2];
      auto condIt = ssaValues.find(condSrc);
      auto trueIt = ssaValues.find(trueSrc);
      auto falseIt = ssaValues.find(falseSrc);
      if (condIt == ssaValues.end() || trueIt == ssaValues.end() || falseIt == ssaValues.end()) {
        sd_printf("TritonIRBuilder: missing SSA value for ternary op '%s' at slot %d\n",
                  slot.opName.c_str(), si);
        continue;
      }
      auto opResult = emitTernaryOp(builder, loc, condIt->second, trueIt->second, falseIt->second, blockSize);
      for (int o = 0; o < slot.numOutputs; o++) {
        ssaValues[slot.outputSlotIndices[o]] = opResult;
      }

    } else if (cat == TritonOpCategory::IDENTITY) {
      // Identity/assign: SSA value forwarding, no IR op needed
      if (slot.numInputs < 1) {
        sd_printf("TritonIRBuilder: identity op '%s' at slot %d has no inputs\n",
                  slot.opName.c_str(), si);
        continue;
      }
      // For assign(target, source): output = source = input[1]
      // For identity(x): output = x = input[0]
      int inputIdx = (slot.numInputs >= 2) ? 1 : 0;
      int inputSrc = slot.inputSourceIndices[inputIdx];
      auto inputIt = ssaValues.find(inputSrc);
      if (inputIt == ssaValues.end()) {
        sd_printf("TritonIRBuilder: missing SSA value for identity op '%s' at slot %d\n",
                  slot.opName.c_str(), si);
        continue;
      }
      // Forward the SSA value directly — no computation needed
      for (int o = 0; o < slot.numOutputs; o++) {
        ssaValues[slot.outputSlotIndices[o]] = inputIt->second;
      }

    } else if (cat == TritonOpCategory::CAST) {
      // Cast: type conversion using the castTo() helper
      if (slot.numInputs < 1) {
        sd_printf("TritonIRBuilder: cast op '%s' at slot %d has no inputs\n",
                  slot.opName.c_str(), si);
        continue;
      }
      int inputSrc = slot.inputSourceIndices[0];
      auto inputIt = ssaValues.find(inputSrc);
      if (inputIt == ssaValues.end()) {
        sd_printf("TritonIRBuilder: missing SSA value for cast op '%s' at slot %d\n",
                  slot.opName.c_str(), si);
        continue;
      }
      // Determine target type from the output slot's dtype (dArgs[0])
      DataType targetDtype = FLOAT32;  // default
      if (slot.numDArgs > 0 && slot.dArgs) {
        targetDtype = slot.dArgs[0];
      } else if (slot.numOutputs > 0) {
        int outIdx = slot.outputSlotIndices[0];
        targetDtype = resolveDtypeLocal(outIdx);
      }
      auto targetElemType = getMLIRType(builder, targetDtype);
      auto opResult = castTo(builder, loc, inputIt->second, targetElemType);
      for (int o = 0; o < slot.numOutputs; o++) {
        ssaValues[slot.outputSlotIndices[o]] = opResult;
      }

    } else if (cat == TritonOpCategory::REDUCTION) {
      // Reduction: load input from SSA, call emitReductionOp, store result
      if (slot.numInputs < 1) {
        sd_printf("TritonIRBuilder: reduction op '%s' at slot %d has no inputs\n",
                  slot.opName.c_str(), si);
        continue;
      }
      int inputSrc = slot.inputSourceIndices[0];
      auto inputIt = ssaValues.find(inputSrc);
      if (inputIt == ssaValues.end()) {
        sd_printf("TritonIRBuilder: missing SSA value for reduction op '%s' at slot %d\n",
                  slot.opName.c_str(), si);
        continue;
      }
      // In the 1D kernel skeleton, all tensors are rank-1 (tensor<BLOCK>).
      // The original reduction axis from the multi-dimensional op is irrelevant —
      // we always reduce along axis 0 (the only axis in the 1D tensor).
      // Using the original axis would index out of bounds in tensorTy.getShape().
      int reductionAxis = 0;
      // Get output type from output slot shape
      auto outSlotIdx = slot.outputSlotIndices[0];
      mlir::RankedTensorType outputType;
      {
        auto outShape = resolveShapeLocal(outSlotIdx);
        if (!outShape.empty()) {
          auto elemType = getElementType(inputIt->second);
          std::vector<int64_t> outShape64;
          for (auto d : outShape) outShape64.push_back(static_cast<int64_t>(d));
          outputType = mlir::RankedTensorType::get(outShape64, elemType);
        }
      }
      auto opResult = emitReductionOp(builder, loc, slot.opName, inputIt->second, reductionAxis, outputType);
      // tt.reduce on a rank-1 tensor produces a scalar (not a tensor).
      // Downstream element-wise ops expect RankedTensorType inputs.
      // Splat the scalar result back to a block-sized tensor so the SSA
      // value chain remains homogeneous (all tensors, no scalars).
      if (!mlir::isa<mlir::RankedTensorType>(opResult.getType())) {
        auto splatElemType = opResult.getType();
        auto splatTensorType = mlir::RankedTensorType::get({blockSize}, splatElemType);
        opResult = builder.create<mlir::triton::SplatOp>(loc, splatTensorType, opResult);
      }
      for (int o = 0; o < slot.numOutputs; o++) {
        ssaValues[slot.outputSlotIndices[o]] = opResult;
      }

    } else if (cat == TritonOpCategory::NORMALIZATION) {
      // Normalization: load input from SSA, call emitNormalizationOp, store result
      if (slot.numInputs < 1) {
        sd_printf("TritonIRBuilder: normalization op '%s' at slot %d has no inputs\n",
                  slot.opName.c_str(), si);
        continue;
      }
      int inputSrc = slot.inputSourceIndices[0];
      auto inputIt = ssaValues.find(inputSrc);
      if (inputIt == ssaValues.end()) {
        sd_printf("TritonIRBuilder: missing SSA value for normalization op '%s' at slot %d\n",
                  slot.opName.c_str(), si);
        continue;
      }
      // In the 1D kernel skeleton, all tensors are rank-1 (tensor<BLOCK>).
      // Always normalize along axis 0 — the only axis in the 1D tensor.
      int axis = 0;

      auto outSlotIdx = slot.outputSlotIndices[0];
      mlir::RankedTensorType outputType;
      {
        auto outShape = resolveShapeLocal(outSlotIdx);
        if (!outShape.empty()) {
          auto elemType = getElementType(inputIt->second);
          std::vector<int64_t> outShape64;
          for (auto d : outShape) outShape64.push_back(static_cast<int64_t>(d));
          outputType = mlir::RankedTensorType::get(outShape64, elemType);
        }
      }
      auto opResult = emitNormalizationOp(builder, loc, slot.opName, inputIt->second, axis, outputType);
      // Safety: if normalization somehow returns a scalar, splat it back to tensor
      if (!mlir::isa<mlir::RankedTensorType>(opResult.getType())) {
        auto splatElemType = opResult.getType();
        auto splatTensorType = mlir::RankedTensorType::get({blockSize}, splatElemType);
        opResult = builder.create<mlir::triton::SplatOp>(loc, splatTensorType, opResult);
      }
      for (int o = 0; o < slot.numOutputs; o++) {
        ssaValues[slot.outputSlotIndices[o]] = opResult;
      }

    } else if (cat == TritonOpCategory::MATMUL) {
      // ─── MATMUL: per-element scalar K-loop matmul (correct, no tensor cores) ───
      // For standalone matmul ops within a 1D element-wise segment.
      // Small pure-matmul segments go through buildMatmulModule instead.
      if (slot.numInputs >= 2 && slot.numOutputs >= 1) {
        int aSrc = slot.inputSourceIndices[0];
        int bSrc = slot.inputSourceIndices[1];
        int cSlot = slot.outputSlotIndices[0];

        NDArray* aArr = resolveArr(aSrc);
        NDArray* bArr = resolveArr(bSrc);

        int M = 0, N = 0, K = 0;
        if (aArr && aArr->rankOf() >= 2) {
          M = static_cast<int>(aArr->sizeAt(aArr->rankOf() - 2));
          K = static_cast<int>(aArr->sizeAt(aArr->rankOf() - 1));
        }
        if (bArr && bArr->rankOf() >= 2) {
          N = static_cast<int>(bArr->sizeAt(bArr->rankOf() - 1));
          if (K == 0) K = static_cast<int>(bArr->sizeAt(bArr->rankOf() - 2));
        }

        if (M > 0 && N > 0 && K > 0) {
          auto aPtr = getSlotArgPtr(aSrc);
          auto bPtr = getSlotArgPtr(bSrc);
          auto cPtr = getSlotArgPtr(cSlot);

          if (aPtr && bPtr && cPtr) {
            emitPerElementMatmul(builder, loc, pid, blockSize, aPtr, bPtr, cPtr, M, N, K);

            // Load result back for downstream SSA consumers
            DataType outDtype = FLOAT32;
            NDArray* cArr = resolveArr(cSlot);
            if (cArr) outDtype = cArr->dataType();
            auto loaded = loadBackFromBuffer(cSlot, outDtype);
            if (loaded) {
              for (int o = 0; o < slot.numOutputs; o++) {
                ssaValues[slot.outputSlotIndices[o]] = loaded;
              }
            }
          } else {
            std::string msg = "TritonIRBuilder: matmul '" + slot.opName + "' at slot " + std::to_string(si) +
                " — missing kernel arg ptrs for A(" + std::to_string(aSrc) + ")/B(" + std::to_string(bSrc) +
                ")/C(" + std::to_string(cSlot) + "). Cannot compile.";
            THROW_EXCEPTION(msg.c_str());
          }
        } else {
          std::string msg = "TritonIRBuilder: matmul '" + slot.opName + "' at slot " + std::to_string(si) +
              " — M=" + std::to_string(M) + " N=" + std::to_string(N) + " K=" + std::to_string(K) +
              " invalid dimensions. Cannot compile.";
          THROW_EXCEPTION(msg.c_str());
        }
      } else {
        std::string msg = "TritonIRBuilder: matmul '" + slot.opName + "' at slot " + std::to_string(si) +
            " — needs >=2 inputs and >=1 output, has " + std::to_string(slot.numInputs) + "/" +
            std::to_string(slot.numOutputs) + ". Cannot compile.";
        THROW_EXCEPTION(msg.c_str());
      }

    } else if (cat == TritonOpCategory::FUSED_ATTENTION) {
      // ─── FUSED ATTENTION: Q@K^T + scale + softmax + @V in one kernel ───
      // Resolves Q, K, V inputs, extracts dimensions, calls emitFusedAttentionKernel.
      // emitFusedAttentionKernel creates its own 2D program IDs internally.
      if (slot.numInputs >= 3 && slot.numOutputs >= 1) {
        int qSrc = slot.inputSourceIndices[0];
        int kSrc = slot.inputSourceIndices[1];
        int vSrc = slot.inputSourceIndices[2];
        int outSlot = slot.outputSlotIndices[0];

        NDArray* qArr = resolveArr(qSrc);
        NDArray* kArr = resolveArr(kSrc);
        NDArray* vArr = resolveArr(vSrc);

        // Extract attention dimensions: [batch, heads, seq, headDim]
        int batchSize = 1, numHeads = 1, seqQ = 1, seqK = 1, headDim = 64;
        if (qArr && qArr->rankOf() >= 4) {
          batchSize = static_cast<int>(qArr->sizeAt(0));
          numHeads = static_cast<int>(qArr->sizeAt(1));
          seqQ = static_cast<int>(qArr->sizeAt(2));
          headDim = static_cast<int>(qArr->sizeAt(3));
        } else if (qArr && qArr->rankOf() == 3) {
          // [batch, seq, headDim] — single head
          batchSize = static_cast<int>(qArr->sizeAt(0));
          seqQ = static_cast<int>(qArr->sizeAt(1));
          headDim = static_cast<int>(qArr->sizeAt(2));
        }
        if (kArr && kArr->rankOf() >= 4) {
          seqK = static_cast<int>(kArr->sizeAt(2));
        } else if (kArr && kArr->rankOf() == 3) {
          seqK = static_cast<int>(kArr->sizeAt(1));
        }
        float scale = 1.0f / std::sqrt(static_cast<float>(headDim));
        int blockM = 64, blockN = 64;
        if (headDim <= 32) { blockM = 32; blockN = 32; }

        auto qPtr = getSlotArgPtr(qSrc);
        auto kPtr = getSlotArgPtr(kSrc);
        auto vPtr = getSlotArgPtr(vSrc);
        auto outPtr = getSlotArgPtr(outSlot);

        if (qPtr && kPtr && vPtr && outPtr) {
          emitFusedAttentionKernel(builder, loc, qPtr, kPtr, vPtr, outPtr,
                                   batchSize, numHeads, seqQ, seqK, headDim,
                                   scale, blockM, blockN);

          // Load result back for downstream SSA consumers
          DataType outDtype = FLOAT32;
          NDArray* outArr = resolveArr(outSlot);
          if (outArr) outDtype = outArr->dataType();
          auto loaded = loadBackFromBuffer(outSlot, outDtype);
          if (loaded) {
            for (int o = 0; o < slot.numOutputs; o++) {
              ssaValues[slot.outputSlotIndices[o]] = loaded;
            }
          }
        } else {
          std::string msg = "TritonIRBuilder: fused attention '" + slot.opName + "' at slot " + std::to_string(si) +
              " — missing kernel arg ptrs. Cannot compile.";
          THROW_EXCEPTION(msg.c_str());
        }
      } else {
        std::string msg = "TritonIRBuilder: fused attention '" + slot.opName + "' at slot " + std::to_string(si) +
            " — needs >=3 inputs and >=1 output, has " + std::to_string(slot.numInputs) + "/" +
            std::to_string(slot.numOutputs) + ". Cannot compile.";
        THROW_EXCEPTION(msg.c_str());
      }

    } else if (cat == TritonOpCategory::SHAPE_MANIPULATION) {
      // ─── SHAPE MANIPULATION ───
      // reshape/squeeze/expand_dims/flatten: SSA forwarding (same data, different view)
      // permute/transpose: need actual data reindexing via emitShapeManipulationSection
      std::string opLower = slot.opName;
      std::transform(opLower.begin(), opLower.end(), opLower.begin(), ::tolower);

      bool isPermute = (opLower == "permute" || opLower == "transpose");

      if (isPermute && slot.numInputs >= 1 && slot.numOutputs >= 1) {
        // Permute/transpose requires actual data movement
        int inputSrc = slot.inputSourceIndices[0];
        int outSlot = slot.outputSlotIndices[0];
        NDArray* inArr = resolveArr(inputSrc);
        NDArray* outArr = resolveArr(outSlot);

        auto inPtr = getSlotArgPtr(inputSrc);
        auto outPtr = getSlotArgPtr(outSlot);

        if (inPtr && outPtr && inArr && outArr) {
          std::vector<LongType> inputShape, outputShape;
          for (int d = 0; d < inArr->rankOf(); d++) inputShape.push_back(inArr->sizeAt(d));
          for (int d = 0; d < outArr->rankOf(); d++) outputShape.push_back(outArr->sizeAt(d));

          // Derive permutation from input/output shapes
          // Default: reverse dims (transpose)
          std::vector<int> permutation;
          for (int d = static_cast<int>(inputShape.size()) - 1; d >= 0; d--) {
            permutation.push_back(d);
          }

          int nElements = 1;
          for (auto dim : outputShape) nElements *= static_cast<int>(dim);

          emitShapeManipulationSection(builder, loc, pid, blockSize,
                                        inPtr, outPtr, opLower,
                                        inputShape, outputShape, permutation, nElements);

          // Load result back for downstream SSA consumers
          DataType outDtype = outArr->dataType();
          auto loaded = loadBackFromBuffer(outSlot, outDtype);
          if (loaded) {
            for (int o = 0; o < slot.numOutputs; o++) {
              ssaValues[slot.outputSlotIndices[o]] = loaded;
            }
          }
        } else {
          std::string msg = "TritonIRBuilder: permute/transpose '" + slot.opName + "' at slot " + std::to_string(si) +
              " — missing kernel arg ptrs. Cannot compile.";
          THROW_EXCEPTION(msg.c_str());
        }
      } else if (slot.numInputs >= 1) {
        // reshape/squeeze/expand_dims/flatten: pure SSA forwarding (same data buffer)
        int inputSrc = slot.inputSourceIndices[0];
        auto inputIt = ssaValues.find(inputSrc);
        if (inputIt != ssaValues.end()) {
          for (int o = 0; o < slot.numOutputs; o++) {
            ssaValues[slot.outputSlotIndices[o]] = inputIt->second;
          }
        } else {
          sd_printf("TritonIRBuilder: missing SSA value for shape op '%s' at slot %d (src=%d)\n",
                    slot.opName.c_str(), si, inputSrc);
        }
      }

    } else if (cat == TritonOpCategory::DATA_MOVEMENT) {
      // ─── DATA MOVEMENT: dispatch to appropriate section emitter ───
      std::string opLower = slot.opName;
      std::transform(opLower.begin(), opLower.end(), opLower.begin(), ::tolower);

      if (slot.numInputs < 1 || slot.numOutputs < 1) {
        sd_printf("TritonIRBuilder: data movement '%s' at slot %d — insufficient inputs(%d)/outputs(%d)\n",
                  slot.opName.c_str(), si, slot.numInputs, slot.numOutputs);
      } else if (opLower == "gather" || opLower == "gather_nd") {
        // ─── GATHER ───
        int dataSrc = slot.inputSourceIndices[0];
        int idxSrc = (slot.numInputs >= 2) ? slot.inputSourceIndices[1] : dataSrc;
        int outSlot = slot.outputSlotIndices[0];

        auto dataPtr = getSlotArgPtr(dataSrc);
        auto idxPtr = getSlotArgPtr(idxSrc);
        auto outPtr = getSlotArgPtr(outSlot);
        NDArray* dataArr = resolveArr(dataSrc);
        NDArray* idxArr = resolveArr(idxSrc);
        NDArray* outArr = resolveArr(outSlot);

        if (dataPtr && idxPtr && outPtr && dataArr && outArr) {
          std::vector<LongType> dataShape, indicesShape;
          for (int d = 0; d < dataArr->rankOf(); d++) dataShape.push_back(dataArr->sizeAt(d));
          if (idxArr) {
            for (int d = 0; d < idxArr->rankOf(); d++) indicesShape.push_back(idxArr->sizeAt(d));
          }
          int nElements = static_cast<int>(outArr->lengthOf());
          int axis = 0;
          if (slot.numIArgs > 0 && slot.iArgs) {
            axis = static_cast<int>(slot.iArgs[0]);
          }

          emitGatherSection(builder, loc, pid, blockSize,
                            dataPtr, idxPtr, outPtr, axis,
                            dataShape, indicesShape, nElements);

          auto loaded = loadBackFromBuffer(outSlot, outArr->dataType());
          if (loaded) {
            for (int o = 0; o < slot.numOutputs; o++) ssaValues[slot.outputSlotIndices[o]] = loaded;
          }
        } else {
          std::string msg = "TritonIRBuilder: gather '" + slot.opName + "' at slot " + std::to_string(si) +
              " — missing kernel arg ptrs/arrays. Cannot compile.";
          THROW_EXCEPTION(msg.c_str());
        }

      } else if (opLower == "concat") {
        // ─── CONCAT ───
        int outSlot = slot.outputSlotIndices[0];
        auto outPtr = getSlotArgPtr(outSlot);
        NDArray* outArr = resolveArr(outSlot);

        std::vector<mlir::Value> inPtrs;
        std::vector<std::vector<LongType>> inShapes;
        bool allValid = outPtr && outArr;

        for (int inp = 0; inp < slot.numInputs && allValid; inp++) {
          int src = slot.inputSourceIndices[inp];
          auto ptr = getSlotArgPtr(src);
          NDArray* arr = resolveArr(src);
          if (ptr && arr) {
            inPtrs.push_back(ptr);
            std::vector<LongType> shape;
            for (int d = 0; d < arr->rankOf(); d++) shape.push_back(arr->sizeAt(d));
            inShapes.push_back(shape);
          } else {
            allValid = false;
          }
        }

        if (allValid && !inPtrs.empty()) {
          int nElements = static_cast<int>(outArr->lengthOf());
          int axis = 0;

          emitConcatSection(builder, loc, pid, blockSize,
                            inPtrs, outPtr, axis, inShapes, nElements);

          auto loaded = loadBackFromBuffer(outSlot, outArr->dataType());
          if (loaded) {
            for (int o = 0; o < slot.numOutputs; o++) ssaValues[slot.outputSlotIndices[o]] = loaded;
          }
        } else {
          std::string msg = "TritonIRBuilder: concat '" + slot.opName + "' at slot " + std::to_string(si) +
              " — missing kernel arg ptrs/arrays. Cannot compile.";
          THROW_EXCEPTION(msg.c_str());
        }

      } else if (opLower == "split" || opLower == "split_v") {
        // ─── SPLIT ───
        int dataSrc = slot.inputSourceIndices[0];
        auto dataPtr = getSlotArgPtr(dataSrc);
        NDArray* dataArr = resolveArr(dataSrc);

        std::vector<mlir::Value> outPtrs;
        bool allValid = dataPtr && dataArr;
        for (int o = 0; o < slot.numOutputs && allValid; o++) {
          int oSlot = slot.outputSlotIndices[o];
          auto ptr = getSlotArgPtr(oSlot);
          if (ptr) {
            outPtrs.push_back(ptr);
          } else {
            allValid = false;
          }
        }

        if (allValid && !outPtrs.empty()) {
          std::vector<LongType> dataShape;
          for (int d = 0; d < dataArr->rankOf(); d++) dataShape.push_back(dataArr->sizeAt(d));
          int numSplits = slot.numOutputs;
          int nElements = static_cast<int>(dataArr->lengthOf());

          emitSplitSection(builder, loc, pid, blockSize,
                           dataPtr, outPtrs, 0, numSplits, dataShape, nElements);

          // Load back each output for downstream SSA
          for (int o = 0; o < slot.numOutputs; o++) {
            int oSlot = slot.outputSlotIndices[o];
            NDArray* oArr = resolveArr(oSlot);
            DataType dt = oArr ? oArr->dataType() : FLOAT32;
            auto loaded = loadBackFromBuffer(oSlot, dt);
            if (loaded) ssaValues[oSlot] = loaded;
          }
        } else {
          std::string msg = "TritonIRBuilder: split '" + slot.opName + "' at slot " + std::to_string(si) +
              " — missing kernel arg ptrs. Cannot compile.";
          THROW_EXCEPTION(msg.c_str());
        }

      } else if (opLower == "tile") {
        // ─── TILE ───
        int dataSrc = slot.inputSourceIndices[0];
        int outSlot = slot.outputSlotIndices[0];
        auto dataPtr = getSlotArgPtr(dataSrc);
        auto outPtr = getSlotArgPtr(outSlot);
        NDArray* dataArr = resolveArr(dataSrc);
        NDArray* outArr = resolveArr(outSlot);

        if (dataPtr && outPtr && dataArr && outArr) {
          std::vector<LongType> inputShape;
          for (int d = 0; d < dataArr->rankOf(); d++) inputShape.push_back(dataArr->sizeAt(d));
          // Derive repeats from output/input shape ratio
          std::vector<int> repeats;
          for (int d = 0; d < outArr->rankOf() && d < dataArr->rankOf(); d++) {
            repeats.push_back(static_cast<int>(outArr->sizeAt(d) / std::max(dataArr->sizeAt(d), (LongType)1)));
          }
          int nElements = static_cast<int>(outArr->lengthOf());

          emitTileSection(builder, loc, pid, blockSize,
                          dataPtr, outPtr, inputShape, repeats, nElements);

          auto loaded = loadBackFromBuffer(outSlot, outArr->dataType());
          if (loaded) {
            for (int o = 0; o < slot.numOutputs; o++) ssaValues[slot.outputSlotIndices[o]] = loaded;
          }
        } else {
          std::string msg = "TritonIRBuilder: tile '" + slot.opName + "' at slot " + std::to_string(si) +
              " — missing kernel arg ptrs. Cannot compile.";
          THROW_EXCEPTION(msg.c_str());
        }

      } else if (opLower == "strided_slice") {
        // ─── STRIDED SLICE ───
        int dataSrc = slot.inputSourceIndices[0];
        int outSlot = slot.outputSlotIndices[0];
        auto dataPtr = getSlotArgPtr(dataSrc);
        auto outPtr = getSlotArgPtr(outSlot);
        NDArray* dataArr = resolveArr(dataSrc);
        NDArray* outArr = resolveArr(outSlot);

        if (dataPtr && outPtr && dataArr && outArr) {
          std::vector<LongType> inputShape;
          for (int d = 0; d < dataArr->rankOf(); d++) inputShape.push_back(dataArr->sizeAt(d));
          // Default: slice from 0 with stride 1, length = output length
          std::vector<int> begins(dataArr->rankOf(), 0);
          std::vector<int> ends;
          for (int d = 0; d < outArr->rankOf() && d < dataArr->rankOf(); d++) {
            ends.push_back(static_cast<int>(outArr->sizeAt(d)));
          }
          std::vector<int> strides(dataArr->rankOf(), 1);
          int nElements = static_cast<int>(outArr->lengthOf());

          emitSliceSection(builder, loc, pid, blockSize,
                           dataPtr, outPtr, begins, ends, strides,
                           inputShape, nElements);

          auto loaded = loadBackFromBuffer(outSlot, outArr->dataType());
          if (loaded) {
            for (int o = 0; o < slot.numOutputs; o++) ssaValues[slot.outputSlotIndices[o]] = loaded;
          }
        } else {
          std::string msg = "TritonIRBuilder: strided_slice '" + slot.opName + "' at slot " + std::to_string(si) +
              " — missing kernel arg ptrs. Cannot compile.";
          THROW_EXCEPTION(msg.c_str());
        }

      } else if (opLower == "stack") {
        // ─── STACK: treat as concat (stack = unsqueeze + concat along new axis) ───
        int outSlot = slot.outputSlotIndices[0];
        auto outPtr = getSlotArgPtr(outSlot);
        NDArray* outArr = resolveArr(outSlot);

        std::vector<mlir::Value> inPtrs;
        std::vector<std::vector<LongType>> inShapes;
        bool allValid = outPtr && outArr;

        for (int inp = 0; inp < slot.numInputs && allValid; inp++) {
          int src = slot.inputSourceIndices[inp];
          auto ptr = getSlotArgPtr(src);
          NDArray* arr = resolveArr(src);
          if (ptr && arr) {
            inPtrs.push_back(ptr);
            std::vector<LongType> shape;
            for (int d = 0; d < arr->rankOf(); d++) shape.push_back(arr->sizeAt(d));
            inShapes.push_back(shape);
          } else {
            allValid = false;
          }
        }

        if (allValid && !inPtrs.empty()) {
          int nElements = static_cast<int>(outArr->lengthOf());
          emitConcatSection(builder, loc, pid, blockSize,
                            inPtrs, outPtr, 0, inShapes, nElements);

          auto loaded = loadBackFromBuffer(outSlot, outArr->dataType());
          if (loaded) {
            for (int o = 0; o < slot.numOutputs; o++) ssaValues[slot.outputSlotIndices[o]] = loaded;
          }
        } else {
          std::string msg = "TritonIRBuilder: stack '" + slot.opName + "' at slot " + std::to_string(si) +
              " — missing kernel arg ptrs. Cannot compile.";
          THROW_EXCEPTION(msg.c_str());
        }

      } else if (opLower == "scatter_nd" || opLower == "scatter_nd_update") {
        // ─── SCATTER_ND: copy data + scatter updates at indexed positions ───
        // scatter_nd needs 3 inputs: data, indices, updates
        // Output = copy of data with updates scattered at indexed positions
        if (slot.numInputs >= 3 && slot.numOutputs >= 1) {
          int dataSrc = slot.inputSourceIndices[0];
          int idxSrc = slot.inputSourceIndices[1];
          int updSrc = slot.inputSourceIndices[2];
          int outSlot = slot.outputSlotIndices[0];

          auto dataArgIt = slotToArgIdx.find(dataSrc);
          auto idxArgIt = slotToArgIdx.find(idxSrc);
          auto updArgIt = slotToArgIdx.find(updSrc);
          auto outArgIt = slotToArgIdx.find(outSlot);

          NDArray* dataArr = resolveArr(dataSrc);
          int nElem = dataArr ? static_cast<int>(dataArr->lengthOf()) : 0;

          if (dataArgIt != slotToArgIdx.end() && idxArgIt != slotToArgIdx.end() &&
              updArgIt != slotToArgIdx.end() && outArgIt != slotToArgIdx.end() && nElem > 0) {
            auto dPtr = getBufferArg(dataArgIt->second);
            auto iPtr = getBufferArg(idxArgIt->second);
            auto uPtr = getBufferArg(updArgIt->second);
            auto oPtr = getBufferArg(outArgIt->second);

            std::vector<LongType> dataShape;
            if (dataArr) {
              for (int d = 0; d < dataArr->rankOf(); d++) dataShape.push_back(dataArr->sizeAt(d));
            }
            emitScatterNdSection(builder, loc, pid, blockSize, dPtr, iPtr, uPtr, oPtr, dataShape, nElem);

            // Load result back for downstream SSA consumers
            auto result = loadBackFromBuffer(outSlot, FLOAT32);
            if (result) {
              for (int o = 0; o < slot.numOutputs; o++) ssaValues[slot.outputSlotIndices[o]] = result;
            }
          } else {
            std::string msg = "TritonIRBuilder: scatter_nd '" + slot.opName + "' at slot " + std::to_string(si) +
                " — missing kernel arg ptrs. Cannot compile.";
            THROW_EXCEPTION(msg.c_str());
          }
        } else if (slot.numInputs >= 1) {
          auto inputIt = ssaValues.find(slot.inputSourceIndices[0]);
          if (inputIt != ssaValues.end()) {
            for (int o = 0; o < slot.numOutputs; o++) ssaValues[slot.outputSlotIndices[o]] = inputIt->second;
          }
        }

      } else {
        // Unknown data movement op — fail compilation instead of producing garbage
        std::string msg = "TritonIRBuilder: unhandled data movement op '" + slot.opName + "' at slot " +
            std::to_string(si) + ". No emitter available. Cannot compile.";
        THROW_EXCEPTION(msg.c_str());
      }

    } else if (cat == TritonOpCategory::CONSTANT_GENERATION) {
      // Constant generation ops (shape_of, create, set_scalar, ones_as, range):
      // These produce constant or computed values independent of input data.
      // In the 1D kernel, emit appropriate constant splats or ranges.
      DataType outDtype = FLOAT32;
      if (slot.numOutputs > 0) {
        int outIdx = slot.outputSlotIndices[0];
        outDtype = resolveDtypeLocal(outIdx);
      }
      auto elemType = getMLIRType(builder, outDtype);
      auto tensorType = mlir::RankedTensorType::get({blockSize}, elemType);

      std::string opLower = slot.opName;
      std::transform(opLower.begin(), opLower.end(), opLower.begin(), ::tolower);

      mlir::Value opResult;
      if (opLower == "ones_as" || opLower == "oneslike" || opLower == "ones_like") {
        // Fill with 1.0 / 1
        opResult = splatConstantF32(builder, loc, tensorType, 1.0f);
      } else if (opLower == "create" || opLower == "set_scalar") {
        // create/set_scalar: produce constant fill value.
        // Try tArgs first, then fall back to reading from the warmup output array.
        float fillVal = 0.0f;
        bool foundVal = false;
        if (slot.numTArgs > 0 && slot.tArgs) {
          fillVal = static_cast<float>(slot.tArgs[0]);
          foundVal = true;
        }
        if (!foundVal && slot.numOutputs > 0) {
          int outIdx = slot.outputSlotIndices[0];
          auto* arr = resolveArr(outIdx);
          if (arr && arr->lengthOf() > 0) {
            arr->syncToHost();
            fillVal = arr->e<float>(0);
            foundVal = true;
          }
        }
        opResult = splatConstantF32(builder, loc, tensorType, fillVal);
      } else if (opLower == "range") {
        // range(start, stop, step): produce broadcast-safe values using global offsets.
        // The range output has rangeLen elements; when downstream ops have more elements,
        // we use modular indexing: value[i] = start + step * (offsets % rangeLen).
        float start = 0.0f, step = 1.0f;
        if (slot.numTArgs >= 1 && slot.tArgs) start = static_cast<float>(slot.tArgs[0]);
        if (slot.numTArgs >= 3 && slot.tArgs) step = static_cast<float>(slot.tArgs[2]);

        // Determine range output length from the output array's shape
        int rangeLen = blockSize;
        if (slot.numOutputs > 0) {
          int outIdx = slot.outputSlotIndices[0];
          auto* arr = resolveArr(outIdx);
          if (arr) rangeLen = static_cast<int>(arr->lengthOf());
        }

        auto i32TensorTy = mlir::RankedTensorType::get({blockSize}, builder.getI32Type());
        auto f32TensorTy = mlir::RankedTensorType::get({blockSize}, builder.getF32Type());

        // offsets % rangeLen → position within the range (broadcast-safe)
        auto rangeLenConst = builder.create<mlir::arith::ConstantIntOp>(loc, rangeLen, 32);
        auto splatRangeLen = builder.create<mlir::triton::SplatOp>(loc, i32TensorTy, rangeLenConst);
        auto modOffsets = builder.create<mlir::arith::RemUIOp>(loc, offsets, splatRangeLen);

        // start + step * modOffsets
        auto floatOffsets = builder.create<mlir::arith::SIToFPOp>(loc, f32TensorTy, modOffsets);
        auto startSplat = splatConstantF32(builder, loc, f32TensorTy, start);
        auto stepSplat = splatConstantF32(builder, loc, f32TensorTy, step);
        auto scaled = builder.create<mlir::arith::MulFOp>(loc, floatOffsets, stepSplat);
        opResult = builder.create<mlir::arith::AddFOp>(loc, startSplat, scaled);
        opResult = castTo(builder, loc, opResult, elemType);
      } else if (opLower == "shape_of") {
        // shape_of(x): output = shape dimensions of x as a tensor.
        // Read the pre-computed values from the warmup output array and use
        // broadcast-safe indexing (offsets % outputLen) since the output is tiny.
        bool emitted = false;
        if (slot.numOutputs > 0) {
          int outIdx = slot.outputSlotIndices[0];
          auto* arr = resolveArr(outIdx);
          if (arr && arr->lengthOf() > 0) {
            arr->syncToHost();
            int outLen = static_cast<int>(arr->lengthOf());
            // Emit the shape values as: load from constant index within [0, outLen)
            // Use the same broadcast-safe pattern as range: offsets % outLen
            auto i32TensorTy = mlir::RankedTensorType::get({blockSize}, builder.getI32Type());
            auto outLenConst = builder.create<mlir::arith::ConstantIntOp>(loc, outLen, 32);
            auto splatOutLen = builder.create<mlir::triton::SplatOp>(loc, i32TensorTy, outLenConst);
            auto modOffsets = builder.create<mlir::arith::RemUIOp>(loc, offsets, splatOutLen);

            // Build a lookup table: for each dimension d, shape_val[d]
            // Since outLen is small (typically 2-6), use chained selects
            auto f32TensorTy = mlir::RankedTensorType::get({blockSize}, builder.getF32Type());
            opResult = splatConstantF32(builder, loc, f32TensorTy, 0.0f);
            for (int d = outLen - 1; d >= 0; d--) {
              float dimVal = static_cast<float>(arr->e<float>(d));
              auto dimConst = builder.create<mlir::arith::ConstantIntOp>(loc, d, 32);
              auto splatDim = builder.create<mlir::triton::SplatOp>(loc, i32TensorTy, dimConst);
              auto cmp = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq,
                                                               modOffsets, splatDim);
              auto dimValSplat = splatConstantF32(builder, loc, f32TensorTy, dimVal);
              opResult = builder.create<mlir::arith::SelectOp>(loc, cmp, dimValSplat, opResult);
            }
            opResult = castTo(builder, loc, opResult, elemType);
            emitted = true;
          }
        }
        if (!emitted) {
          opResult = splatConstantF32(builder, loc, tensorType, 0.0f);
        }
      } else {
        // Default: zero fill
        opResult = splatConstantF32(builder, loc, tensorType, 0.0f);
      }

      for (int o = 0; o < slot.numOutputs; o++) {
        ssaValues[slot.outputSlotIndices[o]] = opResult;
      }
    }
  }

  // 2d: Store outputs — tt.store for each output arg
  int outputArgBase = static_cast<int>(inputArgs.size());
  for (int a = 0; a < static_cast<int>(outputArgs.size()); a++) {
    auto& arg = outputArgs[a];
    auto funcArg = getBufferArg(outputArgBase + a);

    auto ssaIt = ssaValues.find(arg.slotIndex);
    if (ssaIt == ssaValues.end()) {
      sd_printf("TritonIRBuilder: no SSA value for output slot %d — skipping store\n",
                arg.slotIndex);
      continue;
    }

    auto elemType = getMLIRType(builder, arg.dtype);
    auto ptrType = mlir::triton::PointerType::get(elemType, 1);
    auto ptrTensorType = mlir::RankedTensorType::get({blockSize}, ptrType);

    auto splatPtr = builder.create<mlir::triton::SplatOp>(loc, ptrTensorType, funcArg);
    auto ptrs = builder.create<mlir::triton::AddPtrOp>(loc, ptrTensorType, splatPtr, offsets);

    // Cast SSA value to match output element type if needed
    mlir::Value storeVal = castTo(builder, loc, ssaIt->second, elemType);

    builder.create<mlir::triton::StoreOp>(loc, ptrs, storeVal, mask,
                                           mlir::triton::CacheModifier::NONE,
                                           mlir::triton::EvictionPolicy::NORMAL);
  }

  // Return
  builder.create<mlir::triton::ReturnOp>(loc);

  result.mlirModule = new mlir::ModuleOp(moduleOp);
  result.mlirContext = mlirContext;  // Store for proper cleanup
  result.valid = true;
  result.useIndirectArgs = useIndirectArgs;

  // Dump TTIR module for diagnostics (before Triton pipeline)
  {
    std::string ttirDump;
    llvm::raw_string_ostream os(ttirDump);
    moduleOp.print(os);
    sd_printf("TritonIRBuilder: built module '%s' with %d ops, %d input args, %d output args, "
              "BLOCK_SIZE=%d\n",
              result.kernelName.c_str(), (endSlot - startSlot + 1),
              static_cast<int>(inputArgs.size()), static_cast<int>(outputArgs.size()),
              blockSize);
    // Write TTIR to file for indirect-args kernels (large output)
    if (useIndirectArgs) {
      FILE* df = fopen("/tmp/triton_ttir_indirect.mlir", "w");
      if (df) {
        fprintf(df, "%s\n", ttirDump.c_str());
        fflush(df); fclose(df);
      }
    }
  }

  return result;
}

// ─── Sectioned cooperative mega-kernel builder ──────────────────────────────
//
// Breaks a mixed segment into typed sections (elementwise, matmul, attention,
// data movement, etc.) and emits each section with the appropriate emitter.
// Cooperative grid sync barriers are inserted between sections that have
// cross-block data dependencies (i.e., a section reads another section's output).

TritonIRModule TritonIRBuilder::buildSectionedModule(
    NativeSlot* slots, int startSlot, int endSlot,
    int totalSlots,
    NDArray** externalInputs, int numExternalInputs,
    NDArray** outputSlots, int totalOutputSlots,
    int* requestedOutputSlotIndices,
    int numRequestedOutputs) {

  TritonIRModule result;
  int segSize = endSlot - startSlot + 1;
  result.kernelName = generateKernelName(slots, startSlot, endSlot);

  sd_debug("TritonIRBuilder::buildSectionedModule: segment [%d-%d] (%d ops)\n",
            startSlot, endSlot, segSize);

  // ── Step 1: Identify sections ──
  auto sections = identifySections(slots, startSlot, endSlot,
                                    outputSlots, totalOutputSlots,
                                    externalInputs, numExternalInputs);
  if (sections.empty()) {
    sd_debug("TritonIRBuilder::buildSectionedModule: no sections identified for seg [%d-%d]\n",
              startSlot, endSlot);
    return result;
  }

  sd_debug("TritonIRBuilder::buildSectionedModule: identified %d sections\n",
            static_cast<int>(sections.size()));

  // ── Step 1b: Build cached shape info map ──
  // Maps outputSlotIndex → cached shapeInfo pointer from NativeSlot's shape cache.
  // This survives even when outputSlots[idx] has been released (set to nullptr).
  std::unordered_map<int, const LongType*> cachedShapeInfoMap;
  for (int i = 0; i < totalSlots; i++) {
    if (slots[i].shapeCacheValid && !slots[i].cachedOutputShapes.empty()) {
      for (int o = 0; o < slots[i].numOutputs; o++) {
        int outIdx = slots[i].outputSlotIndices[o];
        if (outIdx >= 0 && o < static_cast<int>(slots[i].cachedOutputShapes.size()) &&
            slots[i].cachedOutputShapes[o] != nullptr) {
          cachedShapeInfoMap[outIdx] = slots[i].cachedOutputShapes[o];
        }
      }
    }
  }

  sd_debug("TritonIRBuilder::buildSectionedModule: cached shape info map has %d entries\n",
            static_cast<int>(cachedShapeInfoMap.size()));

  // Helper: resolve shape for a source index.
  // Priority 1: cached shape info (survives outputSlot release)
  // Priority 2: live outputSlots array
  // Priority 3: external inputs
  auto resolveShape = [&](int srcIdx) -> std::vector<LongType> {
    if (srcIdx < 0) {
      int extIdx = -(srcIdx + 1);
      if (extIdx < numExternalInputs && externalInputs && externalInputs[extIdx]) {
        auto& arr = *externalInputs[extIdx];
        std::vector<LongType> s(arr.rankOf());
        for (int d = 0; d < arr.rankOf(); d++) s[d] = arr.sizeAt(d);
        return s;
      }
      return {};
    }
    // Priority 1: cached shape info
    auto cit = cachedShapeInfoMap.find(srcIdx);
    if (cit != cachedShapeInfoMap.end() && cit->second) {
      LongType rank = shape::rank(cit->second);
      std::vector<LongType> s(rank);
      for (int d = 0; d < rank; d++) s[d] = shape::shapeOf(cit->second)[d];
      return s;
    }
    // Priority 2: live outputSlots
    if (srcIdx < totalOutputSlots && outputSlots && outputSlots[srcIdx]) {
      auto& arr = *outputSlots[srcIdx];
      std::vector<LongType> s(arr.rankOf());
      for (int d = 0; d < arr.rankOf(); d++) s[d] = arr.sizeAt(d);
      return s;
    }
    return {};
  };

  // Helper: resolve dtype for a source index (same priority as resolveShape)
  auto resolveDtype = [&](int srcIdx) -> DataType {
    if (srcIdx < 0) {
      int extIdx = -(srcIdx + 1);
      if (extIdx < numExternalInputs && externalInputs && externalInputs[extIdx])
        return externalInputs[extIdx]->dataType();
      return FLOAT32;
    }
    auto cit = cachedShapeInfoMap.find(srcIdx);
    if (cit != cachedShapeInfoMap.end() && cit->second)
      return ArrayOptions::dataType(cit->second);
    if (srcIdx < totalOutputSlots && outputSlots && outputSlots[srcIdx])
      return outputSlots[srcIdx]->dataType();
    return FLOAT32;
  };

  // Helper: compute total length from shape
  auto shapeLength = [](const std::vector<LongType>& s) -> LongType {
    if (s.empty()) return 0;
    LongType len = 1;
    for (auto d : s) len *= d;
    return len;
  };

  // ── Step 2: Collect kernel args ──
  // For sectioned kernels, ALL outputs need kernel args (not just externally visible ones)
  // because cross-section data flows through global memory buffers.
  // Internal intermediates within a single ELEMENTWISE section are still SSA-forwarded.

  // Collect all internal slot outputs
  std::unordered_set<int> internalSlotOutputs;
  for (int i = startSlot; i <= endSlot; i++) {
    for (int o = 0; o < slots[i].numOutputs; o++) {
      internalSlotOutputs.insert(slots[i].outputSlotIndices[o]);
    }
  }

  // Determine which outputs are cross-section intermediates:
  // produced in one section, consumed in a different section
  std::unordered_set<int> crossSectionIntermediates;
  for (size_t si = 0; si < sections.size(); si++) {
    auto& sec = sections[si];
    for (int i = sec.startSlot; i <= sec.endSlot; i++) {
      for (int inp = 0; inp < slots[i].numInputs; inp++) {
        int srcIdx = slots[i].inputSourceIndices[inp];
        if (srcIdx < 0) continue;  // External input
        // Check if this source is produced in a DIFFERENT section
        bool producedInThisSection = false;
        for (int j = sec.startSlot; j <= sec.endSlot; j++) {
          for (int o = 0; o < slots[j].numOutputs; o++) {
            if (slots[j].outputSlotIndices[o] == srcIdx) {
              producedInThisSection = true;
              break;
            }
          }
          if (producedInThisSection) break;
        }
        if (!producedInThisSection && internalSlotOutputs.count(srcIdx)) {
          crossSectionIntermediates.insert(srcIdx);
        }
      }
    }
  }

  sd_debug("TritonIRBuilder::buildSectionedModule: %d cross-section intermediates\n",
            static_cast<int>(crossSectionIntermediates.size()));

  // Input args: external inputs or outputs from slots BEFORE this segment
  std::vector<TritonKernelArg> inputArgs;
  std::unordered_set<int> seenInputs;
  for (int i = startSlot; i <= endSlot; i++) {
    for (int inp = 0; inp < slots[i].numInputs; inp++) {
      int srcIdx = slots[i].inputSourceIndices[inp];
      if (seenInputs.count(srcIdx)) continue;
      seenInputs.insert(srcIdx);

      if (srcIdx < 0) {
        int extIdx = -(srcIdx + 1);
        if (extIdx < numExternalInputs && externalInputs[extIdx]) {
          TritonKernelArg arg;
          arg.slotIndex = srcIdx;
          arg.outputIndex = 0;
          arg.isOutput = false;
          arg.dtype = externalInputs[extIdx]->dataType();
          auto& arr = *externalInputs[extIdx];
          for (int d = 0; d < arr.rankOf(); d++) arg.shape.push_back(arr.sizeAt(d));
          inputArgs.push_back(arg);
        }
      } else if (!internalSlotOutputs.count(srcIdx)) {
        auto shape = resolveShape(srcIdx);
        auto dtype = resolveDtype(srcIdx);
        bool hasLiveArr = (srcIdx < totalOutputSlots && outputSlots && outputSlots[srcIdx]);
        if (hasLiveArr || !shape.empty()) {
          TritonKernelArg arg;
          arg.slotIndex = srcIdx;
          arg.outputIndex = 0;
          arg.isOutput = false;
          arg.dtype = dtype;
          arg.shape = shape;
          inputArgs.push_back(arg);
        }
      }
    }
  }

  // Output args: externally visible outputs + cross-section intermediates
  auto externalOutputs = computeExternallyVisibleOutputs(
      slots, startSlot, endSlot, totalSlots,
      requestedOutputSlotIndices, numRequestedOutputs);

  // Merge cross-section intermediates into external outputs set
  for (int idx : crossSectionIntermediates) {
    externalOutputs.insert(idx);
  }

  std::vector<TritonKernelArg> outputArgs;
  {
    std::unordered_set<int> seenOutputSlots;
    for (int i = startSlot; i <= endSlot; i++) {
      for (int o = 0; o < slots[i].numOutputs; o++) {
        int outIdx = slots[i].outputSlotIndices[o];
        if (outIdx < 0 || outIdx >= totalOutputSlots) continue;
        if (seenOutputSlots.count(outIdx)) continue;
        seenOutputSlots.insert(outIdx);
        if (!externalOutputs.count(outIdx)) continue;

        TritonKernelArg arg;
        arg.slotIndex = outIdx;
        arg.outputIndex = o;
        arg.isOutput = true;
        if (outputSlots && outIdx < totalOutputSlots && outputSlots[outIdx]) {
          arg.dtype = outputSlots[outIdx]->dataType();
          auto& arr = *outputSlots[outIdx];
          for (int d = 0; d < arr.rankOf(); d++) arg.shape.push_back(arr.sizeAt(d));
        } else {
          // Fall back to cached shape info when live array is not available
          auto cit = cachedShapeInfoMap.find(outIdx);
          if (cit != cachedShapeInfoMap.end() && cit->second) {
            arg.dtype = ArrayOptions::dataType(cit->second);
            LongType rank = shape::rank(cit->second);
            for (int d = 0; d < rank; d++) arg.shape.push_back(shape::shapeOf(cit->second)[d]);
          }
        }
        outputArgs.push_back(arg);
      }
    }
  }

  // Combine: inputs first, then outputs, then sync counter
  result.args.insert(result.args.end(), inputArgs.begin(), inputArgs.end());
  result.args.insert(result.args.end(), outputArgs.begin(), outputArgs.end());

  int totalBufferArgs = static_cast<int>(result.args.size());
  bool useIndirectArgs = (totalBufferArgs + 2) > TRITON_DIRECT_ARG_LIMIT;  // +1 n_elements, +1 sync counter

  sd_debug("TritonIRBuilder::buildSectionedModule: %d input args, %d output args, %d total buffer args%s\n",
            static_cast<int>(inputArgs.size()), static_cast<int>(outputArgs.size()),
            totalBufferArgs, useIndirectArgs ? " (INDIRECT)" : " (direct)");

  // ── Step 3: Create MLIR module and function ──
  auto mlirContext = new mlir::MLIRContext();
  mlirContext->loadDialect<mlir::triton::TritonDialect>();
  mlirContext->loadDialect<mlir::arith::ArithDialect>();
  mlirContext->loadDialect<mlir::math::MathDialect>();
  mlirContext->loadDialect<mlir::scf::SCFDialect>();

  mlir::OpBuilder builder(mlirContext);
  auto loc = builder.getUnknownLoc();
  auto moduleOp = mlir::ModuleOp::create(loc);
  builder.setInsertionPointToEnd(moduleOp.getBody());

  // Function signature: buffer args + n_elements (i32) + sync_counter_ptr (ptr<i32>)
  std::vector<mlir::Type> funcArgTypes;
  auto i32Type = builder.getI32Type();

  if (!useIndirectArgs) {
    for (auto& arg : result.args) {
      auto elemType = getMLIRType(builder, arg.dtype);
      funcArgTypes.push_back(mlir::triton::PointerType::get(elemType, 1));
    }
  } else {
    auto i64Type = builder.getI64Type();
    funcArgTypes.push_back(mlir::triton::PointerType::get(i64Type, 1));
  }
  funcArgTypes.push_back(i32Type);  // n_elements
  // Sync counter pointer for grid sync barriers (only if multiple sections)
  bool needsGridSync = sections.size() > 1;
  if (needsGridSync) {
    funcArgTypes.push_back(mlir::triton::PointerType::get(i32Type, 1));  // sync_counter_ptr
  }

  auto funcType = builder.getFunctionType(funcArgTypes, {});
  auto funcOp = builder.create<mlir::triton::FuncOp>(loc, result.kernelName, funcType);
  funcOp.setPublic();
  auto* entryBlock = funcOp.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  // Unpack indirect args if needed
  std::vector<mlir::Value> argUnpacked;
  if (useIndirectArgs) {
    auto i64Type = builder.getI64Type();
    auto argArrayPtr = entryBlock->getArgument(0);
    for (int a = 0; a < totalBufferArgs; a++) {
      auto idxConst = builder.create<mlir::arith::ConstantIntOp>(loc, a, 64);
      auto elemPtr = builder.create<mlir::triton::AddPtrOp>(
          loc, argArrayPtr.getType(), argArrayPtr, idxConst);
      auto rawVal = builder.create<mlir::triton::LoadOp>(
          loc, elemPtr, mlir::triton::CacheModifier::NONE,
          mlir::triton::EvictionPolicy::NORMAL, false);
      auto& argDesc = result.args[a];
      auto elemType = getMLIRType(builder, argDesc.dtype);
      auto targetPtrType = mlir::triton::PointerType::get(elemType, 1);
      auto castPtr = builder.create<mlir::triton::IntToPtrOp>(loc, targetPtrType, rawVal);
      argUnpacked.push_back(castPtr);
    }
  }

  auto getBufferArg = [&](int a) -> mlir::Value {
    if (useIndirectArgs) return argUnpacked[a];
    return entryBlock->getArgument(a);
  };

  int nElementsArgIdx = useIndirectArgs ? 1 : totalBufferArgs;
  auto nElementsArg = entryBlock->getArgument(nElementsArgIdx);
  mlir::Value syncCounterPtr;
  if (needsGridSync) {
    int syncArgIdx = nElementsArgIdx + 1;
    syncCounterPtr = entryBlock->getArgument(syncArgIdx);
  }

  // ── Step 4: Compute max section grid and tile config from segment content ──
  int maxSectionGrid = 1;
  for (auto& sec : sections) {
    if (sec.gridRequirement > maxSectionGrid) maxSectionGrid = sec.gridRequirement;
  }

  // Derive blockSize/numWarps/numStages from actual op categories and shapes
  // via selectTileConfig() which consults LaunchDims.h
  std::vector<TritonOpCategory> categories;
  std::vector<std::vector<LongType>> shapes;
  for (int i = startSlot; i <= endSlot; i++) {
    categories.push_back(getOpCategory(slots[i].opName));
    if (slots[i].numOutputs > 0) {
      int outIdx = slots[i].outputSlotIndices[0];
      shapes.push_back(resolveShape(outIdx));
    } else {
      shapes.push_back({});
    }
  }
  int blockSize, numWarps, numStages;
  selectTileConfig(categories, shapes, blockSize, numWarps, numStages);

  auto i32TensorType = mlir::RankedTensorType::get({blockSize}, i32Type);

  auto pid = builder.create<mlir::triton::GetProgramIdOp>(
      loc, i32Type, mlir::triton::ProgramIDDim::X);

  // ── Step 5: SSA value map and arg lookup ──
  std::unordered_map<int, mlir::Value> ssaValues;
  std::unordered_map<int, int> slotToArgIdx;
  for (int a = 0; a < static_cast<int>(result.args.size()); a++) {
    slotToArgIdx[result.args[a].slotIndex] = a;
  }

  auto getSlotArgPtr = [&](int slotIdx) -> mlir::Value {
    auto it = slotToArgIdx.find(slotIdx);
    if (it != slotToArgIdx.end()) return getBufferArg(it->second);
    return mlir::Value();
  };

  // Helper: resolve source index to NDArray*
  auto resolveArr = [&](int srcIdx) -> NDArray* {
    if (srcIdx < 0) {
      int extIdx = -(srcIdx + 1);
      return (extIdx < numExternalInputs && externalInputs) ? externalInputs[extIdx] : nullptr;
    }
    return (srcIdx >= 0 && srcIdx < totalOutputSlots && outputSlots) ? outputSlots[srcIdx] : nullptr;
  };

  // Helper: load a buffer into a 1D block-sized tensor
  auto loadBlock = [&](int slotIdx, DataType /*dtype*/) -> mlir::Value {
    auto argPtr = getSlotArgPtr(slotIdx);
    if (!argPtr) return mlir::Value();
    // Derive pointer type from the actual MLIR arg (NOT from dtype parameter)
    auto ptrType = mlir::cast<mlir::triton::PointerType>(argPtr.getType());
    auto ptrTensorType = mlir::RankedTensorType::get({blockSize}, ptrType);
    auto blockSizeConst = builder.create<mlir::arith::ConstantIntOp>(loc, blockSize, 32);
    auto offsetBase = builder.create<mlir::arith::MulIOp>(loc, pid, blockSizeConst);
    auto range = builder.create<mlir::triton::MakeRangeOp>(loc, i32TensorType, 0, blockSize);
    auto splatBase = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, offsetBase);
    auto offsets = builder.create<mlir::arith::AddIOp>(loc, splatBase, range);
    auto splatN = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, nElementsArg);
    auto mask = builder.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::slt, offsets, splatN);
    auto splatPtr = builder.create<mlir::triton::SplatOp>(loc, ptrTensorType, argPtr);
    auto ptrs = builder.create<mlir::triton::AddPtrOp>(loc, ptrTensorType, splatPtr, offsets);
    return builder.create<mlir::triton::LoadOp>(loc, ptrs.getResult(), mask.getResult(),
        mlir::Value(), mlir::triton::CacheModifier::NONE,
        mlir::triton::EvictionPolicy::NORMAL, false);
  };

  // ── Step 6: Emit sections ──
  const auto& opTable = getOpTable();
  int sectionBarrierCount = 0;

  for (size_t secIdx = 0; secIdx < sections.size(); secIdx++) {
    auto& sec = sections[secIdx];

    sd_debug("TritonIRBuilder::buildSectionedModule: emitting section %d/%d type=%d slots[%d-%d]\n",
              static_cast<int>(secIdx), static_cast<int>(sections.size()),
              static_cast<int>(sec.type), sec.startSlot, sec.endSlot);

    // Before each section (except the first), insert grid sync barrier
    // if this section reads outputs from a previous section
    if (secIdx > 0 && needsGridSync) {
      bool needsBarrier = false;
      for (int i = sec.startSlot; i <= sec.endSlot; i++) {
        for (int inp = 0; inp < slots[i].numInputs; inp++) {
          int srcIdx = slots[i].inputSourceIndices[inp];
          if (crossSectionIntermediates.count(srcIdx)) {
            needsBarrier = true;
            break;
          }
        }
        if (needsBarrier) break;
      }
      if (needsBarrier) {
        // Grid sync counter accumulates across barriers (never reset within a launch).
        // Barrier N must wait for counter >= (N+1) * maxSectionGrid:
        //   Barrier 0: all K blocks increment → counter = K, check counter >= K ✓
        //   Barrier 1: all K blocks increment → counter = 2K, check counter >= 2K ✓
        int threshold = (sectionBarrierCount + 1) * maxSectionGrid;
        auto numBlocksVal = builder.create<mlir::arith::ConstantIntOp>(loc, threshold, 32);
        emitGridSync(builder, loc, syncCounterPtr, numBlocksVal);
        sectionBarrierCount++;
        sd_debug("TritonIRBuilder::buildSectionedModule: inserted grid sync barrier before section %d\n",
                  static_cast<int>(secIdx));
      }
    }

    // Emit section body based on type
    switch (sec.type) {
      case KernelSectionType::ELEMENTWISE:
      case KernelSectionType::IDENTITY:
      case KernelSectionType::CONSTANT_GENERATION:
      case KernelSectionType::REDUCTION:
      case KernelSectionType::NORMALIZATION: {
        // ── Element-wise section: 1D skeleton for the ops in this section ──
        auto blockSizeConst = builder.create<mlir::arith::ConstantIntOp>(loc, blockSize, 32);
        auto offsetBase = builder.create<mlir::arith::MulIOp>(loc, pid, blockSizeConst);
        auto range = builder.create<mlir::triton::MakeRangeOp>(loc, i32TensorType, 0, blockSize);
        auto splatBase = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, offsetBase);
        auto offsets = builder.create<mlir::arith::AddIOp>(loc, splatBase, range);
        auto splatN = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, nElementsArg);
        auto mask = builder.create<mlir::arith::CmpIOp>(
            loc, mlir::arith::CmpIPredicate::slt, offsets, splatN);

        // Load inputs that aren't already in SSA map
        for (int si = sec.startSlot; si <= sec.endSlot; si++) {
          for (int inp = 0; inp < slots[si].numInputs; inp++) {
            int srcIdx = slots[si].inputSourceIndices[inp];
            if (ssaValues.count(srcIdx)) continue;
            auto argIt = slotToArgIdx.find(srcIdx);
            if (argIt == slotToArgIdx.end()) continue;
            auto funcArg = getBufferArg(argIt->second);
            auto& argDesc = result.args[argIt->second];
            auto elemType = getMLIRType(builder, argDesc.dtype);
            auto ptrType = mlir::triton::PointerType::get(elemType, 1);
            auto ptrTensorType = mlir::RankedTensorType::get({blockSize}, ptrType);
            auto splatPtr = builder.create<mlir::triton::SplatOp>(loc, ptrTensorType, funcArg);
            auto ptrs = builder.create<mlir::triton::AddPtrOp>(loc, ptrTensorType, splatPtr, offsets);
            auto loaded = builder.create<mlir::triton::LoadOp>(loc, ptrs.getResult(), mask.getResult(),
                mlir::Value(), mlir::triton::CacheModifier::NONE,
                mlir::triton::EvictionPolicy::NORMAL, false);
            ssaValues[srcIdx] = loaded;
          }
        }

        // Emit ops in this section
        for (int si = sec.startSlot; si <= sec.endSlot; si++) {
          auto& slot = slots[si];
          auto cat = getOpCategory(slot.opName);
          auto it = opTable.find(slot.opName);
          if (it == opTable.end()) continue;
          const auto& mapping = it->second;

          if (cat == TritonOpCategory::BINARY_ELEMENTWISE) {
            if (slot.numInputs < 2) continue;
            auto lhsIt = ssaValues.find(slot.inputSourceIndices[0]);
            auto rhsIt = ssaValues.find(slot.inputSourceIndices[1]);
            if (lhsIt == ssaValues.end() || rhsIt == ssaValues.end()) continue;
            auto opResult = emitBinaryElementwise(builder, loc, mapping, lhsIt->second, rhsIt->second);
            for (int o = 0; o < slot.numOutputs; o++) ssaValues[slot.outputSlotIndices[o]] = opResult;
          } else if (cat == TritonOpCategory::UNARY_ELEMENTWISE) {
            if (slot.numInputs < 1) continue;
            auto inputIt = ssaValues.find(slot.inputSourceIndices[0]);
            if (inputIt == ssaValues.end()) continue;
            auto opResult = emitUnaryElementwise(builder, loc, mapping, slot, inputIt->second, blockSize);
            for (int o = 0; o < slot.numOutputs; o++) ssaValues[slot.outputSlotIndices[o]] = opResult;
          } else if (cat == TritonOpCategory::COMPARISON) {
            if (slot.numInputs < 2) continue;
            auto lhsIt = ssaValues.find(slot.inputSourceIndices[0]);
            auto rhsIt = ssaValues.find(slot.inputSourceIndices[1]);
            if (lhsIt == ssaValues.end() || rhsIt == ssaValues.end()) continue;
            auto opResult = emitComparisonOp(builder, loc, slot.opName, lhsIt->second, rhsIt->second, blockSize);
            for (int o = 0; o < slot.numOutputs; o++) ssaValues[slot.outputSlotIndices[o]] = opResult;
          } else if (cat == TritonOpCategory::LOGICAL) {
            if (slot.numInputs < 1) continue;
            auto lhsIt = ssaValues.find(slot.inputSourceIndices[0]);
            if (lhsIt == ssaValues.end()) continue;
            mlir::Value rhsVal = lhsIt->second;
            if (slot.numInputs >= 2) {
              auto rhsIt = ssaValues.find(slot.inputSourceIndices[1]);
              if (rhsIt != ssaValues.end()) rhsVal = rhsIt->second;
            }
            auto opResult = emitLogicalOp(builder, loc, slot.opName, lhsIt->second, rhsVal, blockSize);
            for (int o = 0; o < slot.numOutputs; o++) ssaValues[slot.outputSlotIndices[o]] = opResult;
          } else if (cat == TritonOpCategory::TERNARY) {
            if (slot.numInputs < 3) continue;
            auto condIt = ssaValues.find(slot.inputSourceIndices[0]);
            auto trueIt = ssaValues.find(slot.inputSourceIndices[1]);
            auto falseIt = ssaValues.find(slot.inputSourceIndices[2]);
            if (condIt == ssaValues.end() || trueIt == ssaValues.end() || falseIt == ssaValues.end()) continue;
            auto opResult = emitTernaryOp(builder, loc, condIt->second, trueIt->second, falseIt->second, blockSize);
            for (int o = 0; o < slot.numOutputs; o++) ssaValues[slot.outputSlotIndices[o]] = opResult;
          } else if (cat == TritonOpCategory::IDENTITY) {
            if (slot.numInputs < 1) continue;
            // assign(target, source): forward input[1]; identity(x): forward input[0]
            int identIdx = (slot.numInputs >= 2) ? 1 : 0;
            auto inputIt = ssaValues.find(slot.inputSourceIndices[identIdx]);
            if (inputIt == ssaValues.end()) continue;
            for (int o = 0; o < slot.numOutputs; o++) ssaValues[slot.outputSlotIndices[o]] = inputIt->second;
          } else if (cat == TritonOpCategory::CAST) {
            if (slot.numInputs < 1) continue;
            auto inputIt = ssaValues.find(slot.inputSourceIndices[0]);
            if (inputIt == ssaValues.end()) continue;
            DataType targetDtype = FLOAT32;
            if (slot.numDArgs > 0 && slot.dArgs) {
              targetDtype = slot.dArgs[0];
            } else if (slot.numOutputs > 0) {
              int outIdx = slot.outputSlotIndices[0];
              targetDtype = resolveDtype(outIdx);
            }
            auto targetElemType = getMLIRType(builder, targetDtype);
            auto opResult = castTo(builder, loc, inputIt->second, targetElemType);
            for (int o = 0; o < slot.numOutputs; o++) ssaValues[slot.outputSlotIndices[o]] = opResult;
          } else if (cat == TritonOpCategory::REDUCTION) {
            if (slot.numInputs < 1) continue;
            auto inputIt = ssaValues.find(slot.inputSourceIndices[0]);
            if (inputIt == ssaValues.end()) continue;
            int reductionAxis = 0;
            auto outSlotIdx = slot.outputSlotIndices[0];
            mlir::RankedTensorType outputType;
            {
              auto outShape = resolveShape(outSlotIdx);
              if (!outShape.empty()) {
                auto elemType = getElementType(inputIt->second);
                std::vector<int64_t> outShape64;
                for (auto d : outShape) outShape64.push_back(static_cast<int64_t>(d));
                outputType = mlir::RankedTensorType::get(outShape64, elemType);
              }
            }
            auto opResult = emitReductionOp(builder, loc, slot.opName, inputIt->second, reductionAxis, outputType);
            if (!mlir::isa<mlir::RankedTensorType>(opResult.getType())) {
              auto splatTensorType = mlir::RankedTensorType::get({blockSize}, opResult.getType());
              opResult = builder.create<mlir::triton::SplatOp>(loc, splatTensorType, opResult);
            }
            for (int o = 0; o < slot.numOutputs; o++) ssaValues[slot.outputSlotIndices[o]] = opResult;
          } else if (cat == TritonOpCategory::NORMALIZATION) {
            if (slot.numInputs < 1) continue;
            auto inputIt = ssaValues.find(slot.inputSourceIndices[0]);
            if (inputIt == ssaValues.end()) continue;
            int axis = 0;
            auto outSlotIdx = slot.outputSlotIndices[0];
            mlir::RankedTensorType outputType;
            {
              auto outShape = resolveShape(outSlotIdx);
              if (!outShape.empty()) {
                auto elemType = getElementType(inputIt->second);
                std::vector<int64_t> outShape64;
                for (auto d : outShape) outShape64.push_back(static_cast<int64_t>(d));
                outputType = mlir::RankedTensorType::get(outShape64, elemType);
              }
            }
            auto opResult = emitNormalizationOp(builder, loc, slot.opName, inputIt->second, axis, outputType);
            if (!mlir::isa<mlir::RankedTensorType>(opResult.getType())) {
              auto splatTensorType = mlir::RankedTensorType::get({blockSize}, opResult.getType());
              opResult = builder.create<mlir::triton::SplatOp>(loc, splatTensorType, opResult);
            }
            for (int o = 0; o < slot.numOutputs; o++) ssaValues[slot.outputSlotIndices[o]] = opResult;
          } else if (cat == TritonOpCategory::CONSTANT_GENERATION) {
            // Constant generation: forward SSA value or generate constant
            if (slot.numInputs >= 1) {
              auto inputIt = ssaValues.find(slot.inputSourceIndices[0]);
              if (inputIt != ssaValues.end()) {
                for (int o = 0; o < slot.numOutputs; o++) ssaValues[slot.outputSlotIndices[o]] = inputIt->second;
              }
            }
          } else if (cat == TritonOpCategory::SHAPE_MANIPULATION) {
            // Non-permute shape ops: SSA forwarding
            if (slot.numInputs >= 1) {
              auto inputIt = ssaValues.find(slot.inputSourceIndices[0]);
              if (inputIt != ssaValues.end()) {
                for (int o = 0; o < slot.numOutputs; o++) ssaValues[slot.outputSlotIndices[o]] = inputIt->second;
              }
            }
          }
        }

        // Store cross-section intermediate outputs to global memory
        for (int si = sec.startSlot; si <= sec.endSlot; si++) {
          for (int o = 0; o < slots[si].numOutputs; o++) {
            int outIdx = slots[si].outputSlotIndices[o];
            if (!externalOutputs.count(outIdx)) continue;
            auto ssaIt = ssaValues.find(outIdx);
            if (ssaIt == ssaValues.end()) continue;
            auto argIt = slotToArgIdx.find(outIdx);
            if (argIt == slotToArgIdx.end()) continue;

            DataType dt = resolveDtype(outIdx);

            auto funcArg = getBufferArg(argIt->second);
            auto elemType = getMLIRType(builder, dt);
            auto ptrType = mlir::triton::PointerType::get(elemType, 1);
            auto ptrTensorType = mlir::RankedTensorType::get({blockSize}, ptrType);
            auto splatPtr = builder.create<mlir::triton::SplatOp>(loc, ptrTensorType, funcArg);
            auto ptrs = builder.create<mlir::triton::AddPtrOp>(loc, ptrTensorType, splatPtr, offsets);
            mlir::Value storeVal = castTo(builder, loc, ssaIt->second, elemType);
            builder.create<mlir::triton::StoreOp>(loc, ptrs, storeVal, mask,
                                                   mlir::triton::CacheModifier::NONE,
                                                   mlir::triton::EvictionPolicy::NORMAL);
          }
        }
        break;
      }

      case KernelSectionType::MATMUL: {
        // ── Matmul section: per-element scalar K-loop ──
        // For each matmul op in this section, emit scalar matmul and store/load back
        for (int si = sec.startSlot; si <= sec.endSlot; si++) {
          auto& slot = slots[si];
          if (getOpCategory(slot.opName) != TritonOpCategory::MATMUL) continue;
          if (slot.numInputs < 2 || slot.numOutputs < 1) continue;

          int aSrc = slot.inputSourceIndices[0];
          int bSrc = slot.inputSourceIndices[1];
          int cSlot = slot.outputSlotIndices[0];

          auto aShape = resolveShape(aSrc);
          auto bShape = resolveShape(bSrc);
          int M = 0, N = 0, K = 0;
          if (aShape.size() >= 2) {
            M = static_cast<int>(aShape[aShape.size() - 2]);
            K = static_cast<int>(aShape[aShape.size() - 1]);
          }
          if (bShape.size() >= 2) {
            N = static_cast<int>(bShape[bShape.size() - 1]);
            if (K == 0) K = static_cast<int>(bShape[bShape.size() - 2]);
          }

          auto aPtr = getSlotArgPtr(aSrc);
          auto bPtr = getSlotArgPtr(bSrc);
          auto cPtr = getSlotArgPtr(cSlot);

          if (M > 0 && N > 0 && K > 0 && aPtr && bPtr && cPtr) {
            emitPerElementMatmul(builder, loc, pid, blockSize, aPtr, bPtr, cPtr, M, N, K);
            DataType outDtype = resolveDtype(cSlot);
            auto loaded = loadBlock(cSlot, outDtype);
            if (loaded) {
              for (int o = 0; o < slot.numOutputs; o++) ssaValues[slot.outputSlotIndices[o]] = loaded;
            }
          } else {
            std::string msg = "TritonIRBuilder::buildSectionedModule: matmul at slot " + std::to_string(si) +
                " op='" + slot.opName + "'"
                " aSrc=" + std::to_string(aSrc) + " bSrc=" + std::to_string(bSrc) + " cSlot=" + std::to_string(cSlot) +
                " aShape=[";
            for (size_t d = 0; d < aShape.size(); d++) { if (d) msg += ","; msg += std::to_string(aShape[d]); }
            msg += "] bShape=[";
            for (size_t d = 0; d < bShape.size(); d++) { if (d) msg += ","; msg += std::to_string(bShape[d]); }
            msg += "] M=" + std::to_string(M) + " N=" + std::to_string(N) + " K=" + std::to_string(K) +
                " aPtr=" + (aPtr ? "OK" : "NULL") + " bPtr=" + (bPtr ? "OK" : "NULL") + " cPtr=" + (cPtr ? "OK" : "NULL") +
                " — invalid dimensions or missing args. Cannot compile.";
            THROW_EXCEPTION(msg.c_str());
          }
        }
        break;
      }

      case KernelSectionType::FUSED_ATTENTION: {
        // ── Attention section: emit fused attention kernel ──
        for (int si = sec.startSlot; si <= sec.endSlot; si++) {
          auto& slot = slots[si];
          if (getOpCategory(slot.opName) != TritonOpCategory::FUSED_ATTENTION) continue;
          if (slot.numInputs < 3 || slot.numOutputs < 1) continue;

          int qSrc = slot.inputSourceIndices[0];
          int kSrc = slot.inputSourceIndices[1];
          int vSrc = slot.inputSourceIndices[2];
          int outSlot = slot.outputSlotIndices[0];

          auto qShape = resolveShape(qSrc);
          auto kShape = resolveShape(kSrc);
          int batchSize = 1, numHeads = 1, seqQ = 1, seqK = 1, headDim = 1;
          if (qShape.size() >= 4) {
            batchSize = static_cast<int>(qShape[0]);
            numHeads = static_cast<int>(qShape[1]);
            seqQ = static_cast<int>(qShape[2]);
            headDim = static_cast<int>(qShape[3]);
          } else if (qShape.size() == 3) {
            batchSize = static_cast<int>(qShape[0]);
            seqQ = static_cast<int>(qShape[1]);
            headDim = static_cast<int>(qShape[2]);
          }
          if (kShape.size() >= 4) seqK = static_cast<int>(kShape[2]);
          else if (kShape.size() == 3) seqK = static_cast<int>(kShape[1]);
          float scale = 1.0f / std::sqrt(static_cast<float>(std::max(headDim, 1)));
          // Derive attention tile sizes from LaunchDims softmax heuristic
          LongType numTads = static_cast<LongType>(batchSize) * numHeads * seqQ;
          dim3 attDims = getSoftmaxDims(numTads, static_cast<LongType>(headDim));
          int blockM = std::max(32, static_cast<int>(attDims.y) / 32 * 2);  // round to tile
          // Ensure power of 2 and reasonable range
          if (blockM > 128) blockM = 128;
          if (blockM < 32) blockM = 32;
          if (blockM & (blockM - 1)) { int p = 1; while (p < blockM) p <<= 1; blockM = p; }
          int blockN = blockM;  // Symmetric tiles for QK^T and P@V

          auto qPtr = getSlotArgPtr(qSrc);
          auto kPtr = getSlotArgPtr(kSrc);
          auto vPtr = getSlotArgPtr(vSrc);
          auto outPtr = getSlotArgPtr(outSlot);

          if (qPtr && kPtr && vPtr && outPtr) {
            emitFusedAttentionKernel(builder, loc, qPtr, kPtr, vPtr, outPtr,
                                     batchSize, numHeads, seqQ, seqK, headDim,
                                     scale, blockM, blockN);
            DataType outDtype = resolveDtype(outSlot);
            auto loaded = loadBlock(outSlot, outDtype);
            if (loaded) {
              for (int o = 0; o < slot.numOutputs; o++) ssaValues[slot.outputSlotIndices[o]] = loaded;
            }
          } else {
            std::string msg = "TritonIRBuilder::buildSectionedModule: attention at slot " + std::to_string(si) +
                " — missing args. Cannot compile.";
            THROW_EXCEPTION(msg.c_str());
          }
        }
        break;
      }

      case KernelSectionType::GATHER:
      case KernelSectionType::GATHER_ND: {
        for (int si = sec.startSlot; si <= sec.endSlot; si++) {
          auto& slot = slots[si];
          if (slot.numInputs < 1 || slot.numOutputs < 1) continue;
          int dataSrc = slot.inputSourceIndices[0];
          int idxSrc = (slot.numInputs >= 2) ? slot.inputSourceIndices[1] : dataSrc;
          int outSlot = slot.outputSlotIndices[0];
          auto dataPtr = getSlotArgPtr(dataSrc);
          auto idxPtr = getSlotArgPtr(idxSrc);
          auto outPtr = getSlotArgPtr(outSlot);
          auto dataShape = resolveShape(dataSrc);
          auto indicesShape = resolveShape(idxSrc);
          auto outShape = resolveShape(outSlot);
          if (dataPtr && idxPtr && outPtr && !dataShape.empty() && !outShape.empty()) {
            int nElements = static_cast<int>(shapeLength(outShape));
            int axis = (slot.numIArgs > 0 && slot.iArgs) ? static_cast<int>(slot.iArgs[0]) : 0;
            emitGatherSection(builder, loc, pid, blockSize, dataPtr, idxPtr, outPtr, axis,
                              dataShape, indicesShape, nElements);
            auto loaded = loadBlock(outSlot, resolveDtype(outSlot));
            if (loaded) for (int o = 0; o < slot.numOutputs; o++) ssaValues[slot.outputSlotIndices[o]] = loaded;
          }
        }
        break;
      }

      case KernelSectionType::CONCAT: {
        for (int si = sec.startSlot; si <= sec.endSlot; si++) {
          auto& slot = slots[si];
          if (slot.numInputs < 1 || slot.numOutputs < 1) continue;
          int outSlot = slot.outputSlotIndices[0];
          auto outPtr = getSlotArgPtr(outSlot);
          auto outShape = resolveShape(outSlot);
          std::vector<mlir::Value> inPtrs;
          std::vector<std::vector<LongType>> inShapes;
          bool allValid = outPtr && !outShape.empty();
          for (int inp = 0; inp < slot.numInputs && allValid; inp++) {
            int src = slot.inputSourceIndices[inp];
            auto ptr = getSlotArgPtr(src);
            auto shape = resolveShape(src);
            if (ptr && !shape.empty()) {
              inPtrs.push_back(ptr);
              inShapes.push_back(shape);
            } else allValid = false;
          }
          if (allValid && !inPtrs.empty()) {
            int nElements = static_cast<int>(shapeLength(outShape));
            int axis = (slot.numIArgs > 0 && slot.iArgs) ? static_cast<int>(slot.iArgs[0]) : 0;
            emitConcatSection(builder, loc, pid, blockSize, inPtrs, outPtr, axis, inShapes, nElements);
            auto loaded = loadBlock(outSlot, resolveDtype(outSlot));
            if (loaded) for (int o = 0; o < slot.numOutputs; o++) ssaValues[slot.outputSlotIndices[o]] = loaded;
          }
        }
        break;
      }

      case KernelSectionType::SPLIT:
      case KernelSectionType::SPLIT_V: {
        for (int si = sec.startSlot; si <= sec.endSlot; si++) {
          auto& slot = slots[si];
          if (slot.numInputs < 1 || slot.numOutputs < 1) continue;
          int dataSrc = slot.inputSourceIndices[0];
          auto dataPtr = getSlotArgPtr(dataSrc);
          auto dataShape = resolveShape(dataSrc);
          std::vector<mlir::Value> outPtrs;
          bool allValid = dataPtr && !dataShape.empty();
          for (int o = 0; o < slot.numOutputs && allValid; o++) {
            int oSlot = slot.outputSlotIndices[o];
            auto ptr = getSlotArgPtr(oSlot);
            if (ptr) outPtrs.push_back(ptr);
            else allValid = false;
          }
          if (allValid && !outPtrs.empty()) {
            int nElements = static_cast<int>(shapeLength(dataShape));
            emitSplitSection(builder, loc, pid, blockSize, dataPtr, outPtrs, 0, slot.numOutputs, dataShape, nElements);
            for (int o = 0; o < slot.numOutputs; o++) {
              int oSlot = slot.outputSlotIndices[o];
              DataType dt = resolveDtype(oSlot);
              auto loaded = loadBlock(oSlot, dt);
              if (loaded) ssaValues[oSlot] = loaded;
            }
          }
        }
        break;
      }

      case KernelSectionType::TILE: {
        for (int si = sec.startSlot; si <= sec.endSlot; si++) {
          auto& slot = slots[si];
          if (slot.numInputs < 1 || slot.numOutputs < 1) continue;
          int dataSrc = slot.inputSourceIndices[0];
          int outSlot = slot.outputSlotIndices[0];
          auto dataPtr = getSlotArgPtr(dataSrc);
          auto outPtr = getSlotArgPtr(outSlot);
          auto inputShape = resolveShape(dataSrc);
          auto outShape = resolveShape(outSlot);
          if (dataPtr && outPtr && !inputShape.empty() && !outShape.empty()) {
            std::vector<int> repeats;
            for (size_t d = 0; d < outShape.size() && d < inputShape.size(); d++)
              repeats.push_back(static_cast<int>(outShape[d] / std::max(inputShape[d], (LongType)1)));
            int nElements = static_cast<int>(shapeLength(outShape));
            emitTileSection(builder, loc, pid, blockSize, dataPtr, outPtr, inputShape, repeats, nElements);
            auto loaded = loadBlock(outSlot, resolveDtype(outSlot));
            if (loaded) for (int o = 0; o < slot.numOutputs; o++) ssaValues[slot.outputSlotIndices[o]] = loaded;
          }
        }
        break;
      }

      case KernelSectionType::STRIDED_SLICE: {
        for (int si = sec.startSlot; si <= sec.endSlot; si++) {
          auto& slot = slots[si];
          if (slot.numInputs < 1 || slot.numOutputs < 1) continue;
          int dataSrc = slot.inputSourceIndices[0];
          int outSlot = slot.outputSlotIndices[0];
          auto dataPtr = getSlotArgPtr(dataSrc);
          auto outPtr = getSlotArgPtr(outSlot);
          auto inputShape = resolveShape(dataSrc);
          auto outShape = resolveShape(outSlot);
          if (dataPtr && outPtr && !inputShape.empty() && !outShape.empty()) {
            std::vector<int> begins(inputShape.size(), 0);
            std::vector<int> ends;
            for (size_t d = 0; d < outShape.size() && d < inputShape.size(); d++)
              ends.push_back(static_cast<int>(outShape[d]));
            std::vector<int> strides(inputShape.size(), 1);
            int nElements = static_cast<int>(shapeLength(outShape));
            emitSliceSection(builder, loc, pid, blockSize, dataPtr, outPtr, begins, ends, strides, inputShape, nElements);
            auto loaded = loadBlock(outSlot, resolveDtype(outSlot));
            if (loaded) for (int o = 0; o < slot.numOutputs; o++) ssaValues[slot.outputSlotIndices[o]] = loaded;
          }
        }
        break;
      }

      case KernelSectionType::SCATTER_ND:
      case KernelSectionType::SCATTER_ND_UPDATE: {
        for (int si = sec.startSlot; si <= sec.endSlot; si++) {
          auto& slot = slots[si];
          if (slot.numInputs < 3 || slot.numOutputs < 1) continue;
          int dataSrc = slot.inputSourceIndices[0];
          int idxSrc = slot.inputSourceIndices[1];
          int updSrc = slot.inputSourceIndices[2];
          int outSlot = slot.outputSlotIndices[0];
          auto dataPtr = getSlotArgPtr(dataSrc);
          auto idxPtr = getSlotArgPtr(idxSrc);
          auto updPtr = getSlotArgPtr(updSrc);
          auto outPtr = getSlotArgPtr(outSlot);
          auto dataShape = resolveShape(dataSrc);
          auto outShape = resolveShape(outSlot);
          if (dataPtr && idxPtr && updPtr && outPtr && !dataShape.empty() && !outShape.empty()) {
            int nElements = static_cast<int>(shapeLength(outShape));
            emitScatterNdSection(builder, loc, pid, blockSize, dataPtr, idxPtr, updPtr, outPtr, dataShape, nElements);
            auto loaded = loadBlock(outSlot, resolveDtype(outSlot));
            if (loaded) for (int o = 0; o < slot.numOutputs; o++) ssaValues[slot.outputSlotIndices[o]] = loaded;
          }
        }
        break;
      }

      case KernelSectionType::SHAPE_MANIPULATION: {
        // Permute/transpose section
        for (int si = sec.startSlot; si <= sec.endSlot; si++) {
          auto& slot = slots[si];
          if (slot.numInputs < 1 || slot.numOutputs < 1) continue;
          int inputSrc = slot.inputSourceIndices[0];
          int outSlot = slot.outputSlotIndices[0];
          auto inPtr = getSlotArgPtr(inputSrc);
          auto outPtr = getSlotArgPtr(outSlot);
          auto inputShape = resolveShape(inputSrc);
          auto outputShape = resolveShape(outSlot);
          if (inPtr && outPtr && !inputShape.empty() && !outputShape.empty()) {
            std::vector<int> permutation;
            for (int d = static_cast<int>(inputShape.size()) - 1; d >= 0; d--) permutation.push_back(d);
            int nElements = static_cast<int>(shapeLength(outputShape));
            std::string opLower = slot.opName;
            std::transform(opLower.begin(), opLower.end(), opLower.begin(), ::tolower);
            emitShapeManipulationSection(builder, loc, pid, blockSize, inPtr, outPtr, opLower,
                                          inputShape, outputShape, permutation, nElements);
            auto loaded = loadBlock(outSlot, resolveDtype(outSlot));
            if (loaded) for (int o = 0; o < slot.numOutputs; o++) ssaValues[slot.outputSlotIndices[o]] = loaded;
          }
        }
        break;
      }

      case KernelSectionType::CONVOLUTION: {
        for (int si = sec.startSlot; si <= sec.endSlot; si++) {
          auto& slot = slots[si];
          if (slot.numOutputs < 1) continue;

          std::string opLower = slot.opName;
          std::transform(opLower.begin(), opLower.end(), opLower.begin(), ::tolower);

          bool isIm2col = (opLower == "im2col");
          bool isCol2im = (opLower == "col2im");
          bool isIm2colBp = (opLower == "im2col_bp");
          // col2im_bp is not a standard op — col2im has no backprop variant
          // im2col_bp calls col2im internally

          if (isIm2col) {
            // im2col: 1 input (4D image) → 1 output (6D columns)
            // iArgs: [kH, kW, sH, sW, pH, pW, dH, dW, isSameMode]
            if (slot.numInputs < 1) continue;
            int inputSrc = slot.inputSourceIndices[0];
            int outSlot = slot.outputSlotIndices[0];
            auto inPtr = getSlotArgPtr(inputSrc);
            auto outPtr = getSlotArgPtr(outSlot);
            auto inputShape = resolveShape(inputSrc);
            auto outputShape = resolveShape(outSlot);
            if (inPtr && outPtr && !inputShape.empty() && !outputShape.empty()) {
              int kH = (slot.numIArgs > 0 && slot.iArgs) ? static_cast<int>(slot.iArgs[0]) : 1;
              int kW = (slot.numIArgs > 1 && slot.iArgs) ? static_cast<int>(slot.iArgs[1]) : 1;
              int sH = (slot.numIArgs > 2 && slot.iArgs) ? static_cast<int>(slot.iArgs[2]) : 1;
              int sW = (slot.numIArgs > 3 && slot.iArgs) ? static_cast<int>(slot.iArgs[3]) : 1;
              int pH = (slot.numIArgs > 4 && slot.iArgs) ? static_cast<int>(slot.iArgs[4]) : 0;
              int pW = (slot.numIArgs > 5 && slot.iArgs) ? static_cast<int>(slot.iArgs[5]) : 0;
              int dH = (slot.numIArgs > 6 && slot.iArgs) ? static_cast<int>(slot.iArgs[6]) : 1;
              int dW = (slot.numIArgs > 7 && slot.iArgs) ? static_cast<int>(slot.iArgs[7]) : 1;
              int nElements = static_cast<int>(shapeLength(outputShape));
              emitIm2colSection(builder, loc, pid, blockSize, inPtr, outPtr,
                                inputShape, outputShape, kH, kW, sH, sW, pH, pW, dH, dW, nElements);
              auto loaded = loadBlock(outSlot, resolveDtype(outSlot));
              if (loaded) for (int o = 0; o < slot.numOutputs; o++) ssaValues[slot.outputSlotIndices[o]] = loaded;
            }
          } else if (isCol2im || isIm2colBp) {
            // col2im: 1 input (6D columns) → 1 output (4D image)
            //   iArgs: [sY, sX, pY, pX, inY, inX, dY, dX, isSameMode]
            // im2col_bp: 2 inputs (4D image, 6D grad) → 1 output (4D grad)
            //   iArgs: [kH, kW, sH, sW, pH, pW, dH, dW, isSameMode]
            //   The 6D grad (input[1]) is the column data, output is the image-space grad
            if (slot.numInputs < 1) continue;

            // For col2im: input[0] is the 6D column data
            // For im2col_bp: input[1] is the 6D gradient (column data), input[0] is original image
            int colSrc, outSlotIdx;
            if (isCol2im) {
              colSrc = slot.inputSourceIndices[0];
            } else {
              // im2col_bp: second input is the 6D gradient
              if (slot.numInputs < 2) continue;
              colSrc = slot.inputSourceIndices[1];
            }
            outSlotIdx = slot.outputSlotIndices[0];
            auto colPtr = getSlotArgPtr(colSrc);
            auto outPtr = getSlotArgPtr(outSlotIdx);
            auto colShape = resolveShape(colSrc);
            auto outShape = resolveShape(outSlotIdx);
            if (colPtr && outPtr && !colShape.empty() && !outShape.empty()) {
              int kH, kW, sH, sW, pH, pW, dH, dW;
              if (isCol2im) {
                // col2im iArgs: [sY, sX, pY, pX, inY, inX, dY, dX, isSameMode]
                sH = (slot.numIArgs > 0 && slot.iArgs) ? static_cast<int>(slot.iArgs[0]) : 1;
                sW = (slot.numIArgs > 1 && slot.iArgs) ? static_cast<int>(slot.iArgs[1]) : 1;
                pH = (slot.numIArgs > 2 && slot.iArgs) ? static_cast<int>(slot.iArgs[2]) : 0;
                pW = (slot.numIArgs > 3 && slot.iArgs) ? static_cast<int>(slot.iArgs[3]) : 0;
                dH = (slot.numIArgs > 6 && slot.iArgs) ? static_cast<int>(slot.iArgs[6]) : 1;
                dW = (slot.numIArgs > 7 && slot.iArgs) ? static_cast<int>(slot.iArgs[7]) : 1;
                // kH, kW derived from column shape: col[bS, iC, kH, kW, oH, oW]
                kH = (colShape.size() > 2) ? static_cast<int>(colShape[2]) : 1;
                kW = (colShape.size() > 3) ? static_cast<int>(colShape[3]) : 1;
              } else {
                // im2col_bp iArgs: [kH, kW, sH, sW, pH, pW, dH, dW, isSameMode]
                kH = (slot.numIArgs > 0 && slot.iArgs) ? static_cast<int>(slot.iArgs[0]) : 1;
                kW = (slot.numIArgs > 1 && slot.iArgs) ? static_cast<int>(slot.iArgs[1]) : 1;
                sH = (slot.numIArgs > 2 && slot.iArgs) ? static_cast<int>(slot.iArgs[2]) : 1;
                sW = (slot.numIArgs > 3 && slot.iArgs) ? static_cast<int>(slot.iArgs[3]) : 1;
                pH = (slot.numIArgs > 4 && slot.iArgs) ? static_cast<int>(slot.iArgs[4]) : 0;
                pW = (slot.numIArgs > 5 && slot.iArgs) ? static_cast<int>(slot.iArgs[5]) : 0;
                dH = (slot.numIArgs > 6 && slot.iArgs) ? static_cast<int>(slot.iArgs[6]) : 1;
                dW = (slot.numIArgs > 7 && slot.iArgs) ? static_cast<int>(slot.iArgs[7]) : 1;
              }

              int nElements = static_cast<int>(shapeLength(outShape));
              emitCol2imSection(builder, loc, pid, blockSize, colPtr, outPtr,
                                colShape, outShape, kH, kW, sH, sW, pH, pW, dH, dW, nElements);
              auto loaded = loadBlock(outSlotIdx, resolveDtype(outSlotIdx));
              if (loaded) for (int o = 0; o < slot.numOutputs; o++) ssaValues[slot.outputSlotIndices[o]] = loaded;
            }
          } else {
            // conv2d and other convolution ops: 2+ inputs (image + filter)
            if (slot.numInputs < 2) continue;
            int inputSrc = slot.inputSourceIndices[0];
            int filterSrc = slot.inputSourceIndices[1];
            int outSlot = slot.outputSlotIndices[0];
            auto inPtr = getSlotArgPtr(inputSrc);
            auto filterPtr = getSlotArgPtr(filterSrc);
            auto outPtr = getSlotArgPtr(outSlot);
            auto inputShape = resolveShape(inputSrc);
            auto filterShape = resolveShape(filterSrc);
            auto outputShape = resolveShape(outSlot);
            if (inPtr && filterPtr && outPtr && !inputShape.empty() && !filterShape.empty() && !outputShape.empty()) {
              // Extract stride/padding from iArgs: [strideH, strideW, padH, padW, ...]
              int strideH = (slot.numIArgs > 0 && slot.iArgs) ? static_cast<int>(slot.iArgs[0]) : 1;
              int strideW = (slot.numIArgs > 1 && slot.iArgs) ? static_cast<int>(slot.iArgs[1]) : 1;
              int padH = (slot.numIArgs > 2 && slot.iArgs) ? static_cast<int>(slot.iArgs[2]) : 0;
              int padW = (slot.numIArgs > 3 && slot.iArgs) ? static_cast<int>(slot.iArgs[3]) : 0;
              int nElements = static_cast<int>(shapeLength(outputShape));
              emitConvolutionSection(builder, loc, pid, blockSize, inPtr, filterPtr, outPtr,
                                      inputShape, filterShape, outputShape, strideH, strideW, padH, padW, nElements);
              auto loaded = loadBlock(outSlot, resolveDtype(outSlot));
              if (loaded) for (int o = 0; o < slot.numOutputs; o++) ssaValues[slot.outputSlotIndices[o]] = loaded;
            }
          }
        }
        break;
      }

      case KernelSectionType::STACK: {
        // Stack = unsqueeze + concat
        for (int si = sec.startSlot; si <= sec.endSlot; si++) {
          auto& slot = slots[si];
          if (slot.numInputs < 1 || slot.numOutputs < 1) continue;
          int outSlot = slot.outputSlotIndices[0];
          auto outPtr = getSlotArgPtr(outSlot);
          auto outShape = resolveShape(outSlot);
          std::vector<mlir::Value> inPtrs;
          std::vector<std::vector<LongType>> inShapes;
          bool allValid = outPtr && !outShape.empty();
          for (int inp = 0; inp < slot.numInputs && allValid; inp++) {
            int src = slot.inputSourceIndices[inp];
            auto ptr = getSlotArgPtr(src);
            auto shape = resolveShape(src);
            if (ptr && !shape.empty()) {
              inPtrs.push_back(ptr);
              inShapes.push_back(shape);
            } else allValid = false;
          }
          if (allValid && !inPtrs.empty()) {
            int nElements = static_cast<int>(shapeLength(outShape));
            int axis = (slot.numIArgs > 0 && slot.iArgs) ? static_cast<int>(slot.iArgs[0]) : 0;
            emitConcatSection(builder, loc, pid, blockSize, inPtrs, outPtr, axis, inShapes, nElements);
            auto loaded = loadBlock(outSlot, resolveDtype(outSlot));
            if (loaded) for (int o = 0; o < slot.numOutputs; o++) ssaValues[slot.outputSlotIndices[o]] = loaded;
          }
        }
        break;
      }

      default:
        sd_debug("TritonIRBuilder::buildSectionedModule: unsupported section type %d, skipping\n",
                  static_cast<int>(sec.type));
        break;
    }
  }

  // Return
  builder.create<mlir::triton::ReturnOp>(loc);

  // ── Grid and launch configuration ──
  result.gridX = maxSectionGrid;
  result.gridY = 1;
  result.gridZ = 1;
  result.blockX = blockSize;
  result.blockY = 1;
  result.blockZ = 1;
  result.numWarps = numWarps;
  result.numStages = numStages;
  result.useIndirectArgs = useIndirectArgs;
  result.useCooperativeLaunch = needsGridSync;
  result.requiredGrid = maxSectionGrid;
  result.sections = sections;

  result.mlirModule = new mlir::ModuleOp(moduleOp);
  result.mlirContext = mlirContext;
  result.valid = true;

  // Dump TTIR module for diagnostics
  {
    std::string ttirDump;
    llvm::raw_string_ostream os(ttirDump);
    moduleOp.print(os);
    sd_debug("TritonIRBuilder: built sectioned module '%s' with %d sections, %d ops, "
              "%d input args, %d output args, maxGrid=%d, cooperative=%s\nTTIR:\n%s\n",
              result.kernelName.c_str(), static_cast<int>(sections.size()),
              segSize, static_cast<int>(inputArgs.size()),
              static_cast<int>(outputArgs.size()), maxSectionGrid,
              needsGridSync ? "YES" : "NO", ttirDump.c_str());
    // Write TTIR to file for indirect-args kernels
    if (useIndirectArgs) {
      FILE* df = fopen("/tmp/triton_ttir_indirect.mlir", "w");
      if (df) {
        fprintf(df, "// Sectioned module: %s\n// Sections: %d, Ops: %d, Args: %d (indirect)\n%s\n",
                result.kernelName.c_str(), static_cast<int>(sections.size()),
                segSize, totalBufferArgs, ttirDump.c_str());
        fflush(df); fclose(df);
      }
    }
  }

  return result;
}

// ─── Dedicated matmul module builder ─────────────────────────────────────────

TritonIRModule TritonIRBuilder::buildMatmulModule(NativeSlot* slots, int startSlot, int endSlot,
                                                   int totalSlots,
                                                   NDArray** externalInputs, int numExternalInputs,
                                                   NDArray** outputSlots, int totalOutputSlots,
                                                   int* requestedOutputSlotIndices,
                                                   int numRequestedOutputs) {
  TritonIRModule result;
  result.kernelName = generateKernelName(slots, startSlot, endSlot);

  // Find the matmul op and extract M, N, K from input shapes.
  // For matmul A[..., M, K] @ B[..., K, N] = C[..., M, N]:
  //   M = A.shape[-2], K = A.shape[-1] = B.shape[-2], N = B.shape[-1]
  // We derive from INPUTS (A, B) rather than output C, because output arrays
  // may not be allocated yet at compilation time.
  int matmulSlot = -1;
  int matmulM = 0, matmulN = 0, matmulK = 0;

  // Helper lambda: resolve a source index to an NDArray*
  auto resolveArray = [&](int srcIdx) -> NDArray* {
    if (srcIdx < 0) {
      int extIdx = -(srcIdx + 1);
      if (extIdx < numExternalInputs) return externalInputs[extIdx];
    } else if (srcIdx < totalOutputSlots) {
      return outputSlots[srcIdx];
    }
    return nullptr;
  };

  for (int i = startSlot; i <= endSlot; i++) {
    auto cat = getOpCategory(slots[i].opName);
    if (cat == TritonOpCategory::MATMUL) {
      matmulSlot = i;

      // Strategy 1: Extract from input arrays A and B (preferred — always available)
      if (slots[i].numInputs >= 2) {
        NDArray* aArr = resolveArray(slots[i].inputSourceIndices[0]);
        NDArray* bArr = resolveArray(slots[i].inputSourceIndices[1]);

        if (aArr && aArr->rankOf() >= 2) {
          matmulM = static_cast<int>(aArr->sizeAt(aArr->rankOf() - 2));
          matmulK = static_cast<int>(aArr->sizeAt(aArr->rankOf() - 1));
        }
        if (bArr && bArr->rankOf() >= 2) {
          matmulN = static_cast<int>(bArr->sizeAt(bArr->rankOf() - 1));
          // Cross-validate K from B
          int bK = static_cast<int>(bArr->sizeAt(bArr->rankOf() - 2));
          if (matmulK == 0) matmulK = bK;
        }
      }

      // Strategy 2: Fallback to output array if available
      if ((matmulM == 0 || matmulN == 0) && slots[i].numOutputs > 0) {
        int outIdx = slots[i].outputSlotIndices[0];
        if (outIdx >= 0 && outIdx < totalOutputSlots && outputSlots[outIdx]) {
          auto& outArr = *outputSlots[outIdx];
          int rank = outArr.rankOf();
          if (rank >= 2) {
            if (matmulM == 0) matmulM = static_cast<int>(outArr.sizeAt(rank - 2));
            if (matmulN == 0) matmulN = static_cast<int>(outArr.sizeAt(rank - 1));
          }
        }
      }

      // Strategy 3: Fallback to cachedOutputShapes from slot shape cache
      if ((matmulM == 0 || matmulN == 0) && slots[i].shapeCacheValid &&
          !slots[i].cachedOutputShapes.empty()) {
        const LongType* shapeInfo = slots[i].cachedOutputShapes[0];
        if (shapeInfo) {
          int rank = static_cast<int>(shape::rank(shapeInfo));
          if (rank >= 2) {
            const LongType* shapeArr = shape::shapeOf(shapeInfo);
            if (matmulM == 0) matmulM = static_cast<int>(shapeArr[rank - 2]);
            if (matmulN == 0) matmulN = static_cast<int>(shapeArr[rank - 1]);
          }
        }
      }

      // Strategy 4: For K, also try input slot's cached shapes
      if (matmulK == 0 && slots[i].numInputs >= 1) {
        int aSrc = slots[i].inputSourceIndices[0];
        if (aSrc >= 0 && aSrc < static_cast<int>(totalOutputSlots)) {
          // Check if the input slot has cached output shapes
          // (aSrc is another slot's output index, search for the slot that produces it)
          for (int s = 0; s < startSlot; s++) {
            for (int o = 0; o < slots[s].numOutputs; o++) {
              if (slots[s].outputSlotIndices[o] == aSrc &&
                  slots[s].shapeCacheValid && !slots[s].cachedOutputShapes.empty()) {
                const LongType* shapeInfo = slots[s].cachedOutputShapes[o];
                if (shapeInfo) {
                  int rank = static_cast<int>(shape::rank(shapeInfo));
                  if (rank >= 2) {
                    matmulK = static_cast<int>(shape::shapeOf(shapeInfo)[rank - 1]);
                  }
                }
              }
            }
          }
        }
      }

      break;
    }
  }

  if (matmulSlot < 0 || matmulM == 0 || matmulN == 0 || matmulK == 0) {
    // Diagnostic: show what arrays are available for the matmul inputs
    if (matmulSlot >= 0 && slots[matmulSlot].numInputs >= 2) {
      int aSrc = slots[matmulSlot].inputSourceIndices[0];
      int bSrc = slots[matmulSlot].inputSourceIndices[1];
      NDArray* aArr = resolveArray(aSrc);
      NDArray* bArr = resolveArray(bSrc);
      sd_printf("TritonIRBuilder::buildMatmulModule: could not extract M/N/K from slot %d "
                "(M=%d, N=%d, K=%d). Input A[src=%d]: %s (rank=%d), Input B[src=%d]: %s (rank=%d)\n",
                matmulSlot, matmulM, matmulN, matmulK,
                aSrc, aArr ? "present" : "NULL", aArr ? aArr->rankOf() : -1,
                bSrc, bArr ? "present" : "NULL", bArr ? bArr->rankOf() : -1);
    } else {
      sd_printf("TritonIRBuilder::buildMatmulModule: could not extract M/N/K from matmul slot %d "
                "(M=%d, N=%d, K=%d)\n", matmulSlot, matmulM, matmulN, matmulK);
    }
    return result;
  }
  sd_printf("TritonIRBuilder::buildMatmulModule: extracted M=%d, N=%d, K=%d from slot %d\n",
            matmulM, matmulN, matmulK, matmulSlot);

  int blockM = 128, blockN = 128, blockK = 32;
  int numWarps = 4, numStages = 3;
  result.numWarps = numWarps;
  result.numStages = numStages;

  // Create MLIR context and register dialects
  auto mlirContext = new mlir::MLIRContext();
  mlirContext->loadDialect<mlir::triton::TritonDialect>();
  mlirContext->loadDialect<mlir::arith::ArithDialect>();
  mlirContext->loadDialect<mlir::math::MathDialect>();
  mlirContext->loadDialect<mlir::scf::SCFDialect>();

  mlir::OpBuilder builder(mlirContext);
  auto loc = builder.getUnknownLoc();

  // Create module
  auto moduleOp = mlir::ModuleOp::create(loc);
  builder.setInsertionPointToEnd(moduleOp.getBody());

  // ── Collect unique buffer references (same logic as buildModule) ──
  std::unordered_set<int> internalSlotOutputs;
  for (int i = startSlot; i <= endSlot; i++) {
    for (int o = 0; o < slots[i].numOutputs; o++) {
      internalSlotOutputs.insert(slots[i].outputSlotIndices[o]);
    }
  }

  std::vector<TritonKernelArg> inputArgs;
  std::unordered_set<int> seenInputs;
  for (int i = startSlot; i <= endSlot; i++) {
    for (int inp = 0; inp < slots[i].numInputs; inp++) {
      int srcIdx = slots[i].inputSourceIndices[inp];
      if (seenInputs.count(srcIdx)) continue;
      seenInputs.insert(srcIdx);
      if (srcIdx < 0) {
        int extIdx = -(srcIdx + 1);
        if (extIdx < numExternalInputs && externalInputs[extIdx]) {
          TritonKernelArg arg;
          arg.slotIndex = srcIdx;
          arg.outputIndex = 0;
          arg.isOutput = false;
          arg.dtype = externalInputs[extIdx]->dataType();
          auto& arr = *externalInputs[extIdx];
          for (int d = 0; d < arr.rankOf(); d++) arg.shape.push_back(arr.sizeAt(d));
          inputArgs.push_back(arg);
        }
      } else if (!internalSlotOutputs.count(srcIdx)) {
        if (srcIdx < totalOutputSlots && outputSlots[srcIdx]) {
          TritonKernelArg arg;
          arg.slotIndex = srcIdx;
          arg.outputIndex = 0;
          arg.isOutput = false;
          arg.dtype = outputSlots[srcIdx]->dataType();
          auto& arr = *outputSlots[srcIdx];
          for (int d = 0; d < arr.rankOf(); d++) arg.shape.push_back(arr.sizeAt(d));
          inputArgs.push_back(arg);
        }
      }
    }
  }

  // Deduplicate output args and eliminate purely internal intermediates
  auto externalOutputs = computeExternallyVisibleOutputs(
      slots, startSlot, endSlot, totalSlots,
      requestedOutputSlotIndices, numRequestedOutputs);

  std::vector<TritonKernelArg> outputArgs;
  {
    std::unordered_set<int> seenOutputSlots;
    for (int i = startSlot; i <= endSlot; i++) {
      for (int o = 0; o < slots[i].numOutputs; o++) {
        int outIdx = slots[i].outputSlotIndices[o];
        if (outIdx < 0 || outIdx >= totalOutputSlots) continue;
        if (seenOutputSlots.count(outIdx)) continue;  // Deduplicate
        seenOutputSlots.insert(outIdx);
        if (!externalOutputs.count(outIdx)) continue;  // Internal — SSA forwarded

        TritonKernelArg arg;
        arg.slotIndex = outIdx;
        arg.outputIndex = o;
        arg.isOutput = true;
        if (outputSlots && outIdx < totalOutputSlots && outputSlots[outIdx]) {
          arg.dtype = outputSlots[outIdx]->dataType();
          auto& arr = *outputSlots[outIdx];
          for (int d = 0; d < arr.rankOf(); d++) arg.shape.push_back(arr.sizeAt(d));
        }
        outputArgs.push_back(arg);
      }
    }
  }

  result.args.insert(result.args.end(), inputArgs.begin(), inputArgs.end());
  result.args.insert(result.args.end(), outputArgs.begin(), outputArgs.end());

  int totalBufferArgs = static_cast<int>(result.args.size());
  bool useIndirectArgs = (totalBufferArgs + 1) > TRITON_DIRECT_ARG_LIMIT;

  sd_printf("TritonIRBuilder::buildMatmulModule: %d input args, %d output args, %d total%s\n",
            (int)inputArgs.size(), (int)outputArgs.size(), totalBufferArgs,
            useIndirectArgs ? " (INDIRECT)" : " (direct)");

  // ── Build function signature ──
  // Buffer pointers + n_elements (same convention as element-wise kernels).
  // M, N, K are baked as constants into the IR since the kernel is compiled
  // per-shape-key — no need for runtime dimension arguments.
  std::vector<mlir::Type> funcArgTypes;
  auto f32Type = builder.getF32Type();
  auto i32Type = builder.getI32Type();

  if (!useIndirectArgs) {
    for (auto& arg : result.args) {
      auto elemType = getMLIRType(builder, arg.dtype);
      funcArgTypes.push_back(mlir::triton::PointerType::get(elemType, 1));
    }
  } else {
    auto i64Type = builder.getI64Type();
    funcArgTypes.push_back(mlir::triton::PointerType::get(i64Type, 1));  // argArray*
  }
  funcArgTypes.push_back(i32Type);  // n_elements (unused by matmul but expected by launch convention)

  auto funcType = builder.getFunctionType(funcArgTypes, {});
  auto funcOp = builder.create<mlir::triton::FuncOp>(loc, result.kernelName, funcType);
  funcOp.setPublic();

  auto* entryBlock = funcOp.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  // Unpack indirect args if needed (same pattern as buildModule)
  std::vector<mlir::Value> argUnpacked;
  if (useIndirectArgs) {
    auto i64Type = builder.getI64Type();
    auto argArrayPtr = entryBlock->getArgument(0);
    for (int a = 0; a < totalBufferArgs; a++) {
      auto idxConst = builder.create<mlir::arith::ConstantIntOp>(loc, a, 64);
      auto elemPtr = builder.create<mlir::triton::AddPtrOp>(
          loc, argArrayPtr.getType(), argArrayPtr, idxConst);
      auto rawVal = builder.create<mlir::triton::LoadOp>(
          loc, elemPtr,
          mlir::triton::CacheModifier::NONE,
          mlir::triton::EvictionPolicy::NORMAL, false);
      auto& argDesc = result.args[a];
      auto elemType = getMLIRType(builder, argDesc.dtype);
      auto targetPtrType = mlir::triton::PointerType::get(elemType, 1);
      auto castPtr = builder.create<mlir::triton::IntToPtrOp>(loc, targetPtrType, rawVal);
      argUnpacked.push_back(castPtr);
    }
  }

  auto getBufferArg = [&](int a) -> mlir::Value {
    if (useIndirectArgs) return argUnpacked[a];
    return entryBlock->getArgument(a);
  };

  // ── Identify matmul inputs (A, B) and output (C) ──
  // Find the A and B pointer args and the C pointer arg
  int aArgIdx = -1, bArgIdx = -1, cArgIdx = -1;

  // The matmul's input source indices tell us which args correspond to A and B
  auto& matmulOp = slots[matmulSlot];
  if (matmulOp.numInputs >= 2) {
    int aSrc = matmulOp.inputSourceIndices[0];
    int bSrc = matmulOp.inputSourceIndices[1];
    for (int a = 0; a < static_cast<int>(result.args.size()); a++) {
      if (result.args[a].slotIndex == aSrc && !result.args[a].isOutput) aArgIdx = a;
      if (result.args[a].slotIndex == bSrc && !result.args[a].isOutput) bArgIdx = a;
    }
  }
  if (matmulOp.numOutputs >= 1) {
    int cSlot = matmulOp.outputSlotIndices[0];
    for (int a = 0; a < static_cast<int>(result.args.size()); a++) {
      if (result.args[a].slotIndex == cSlot && result.args[a].isOutput) cArgIdx = a;
    }
  }

  if (aArgIdx < 0 || bArgIdx < 0 || cArgIdx < 0) {
    sd_printf("TritonIRBuilder::buildMatmulModule: could not map matmul A/B/C to kernel args "
              "(aArgIdx=%d, bArgIdx=%d, cArgIdx=%d)\n", aArgIdx, bArgIdx, cArgIdx);
    delete mlirContext;
    return result;
  }

  auto aPtr = getBufferArg(aArgIdx);
  auto bPtr = getBufferArg(bArgIdx);
  auto cPtr = getBufferArg(cArgIdx);

  // Emit the matmul kernel body (2D tiled with K-loop)
  emitMatmulKernel(builder, loc, aPtr, bPtr, cPtr,
                    matmulM, matmulN, matmulK, blockM, blockN, blockK);

  // Return
  builder.create<mlir::triton::ReturnOp>(loc);

  // Grid configuration: 2D grid for matmul
  result.gridX = (matmulM + blockM - 1) / blockM;
  result.gridY = (matmulN + blockN - 1) / blockN;
  result.gridZ = 1;
  result.blockX = blockM;
  result.blockY = 1;
  result.blockZ = 1;

  result.mlirModule = new mlir::ModuleOp(moduleOp);
  result.mlirContext = mlirContext;  // Store for proper cleanup
  result.valid = true;
  result.useIndirectArgs = useIndirectArgs;

  // Dump TTIR module for diagnostics
  {
    std::string ttirDump;
    llvm::raw_string_ostream os(ttirDump);
    moduleOp.print(os);
    sd_printf("TritonIRBuilder: built matmul module '%s' M=%d N=%d K=%d, "
              "grid=(%d,%d), %d input args, %d output args\nTTIR:\n%s\n",
              result.kernelName.c_str(), matmulM, matmulN, matmulK,
              result.gridX, result.gridY,
              static_cast<int>(inputArgs.size()), static_cast<int>(outputArgs.size()),
              ttirDump.c_str());
  }

  return result;
}

// ─── Section identification ─────────────────────────────────────────────────
// Walk ops in the segment and group into sections. A new section starts when:
// - Op type changes from element-wise to non-element-wise or vice versa
// - A non-element-wise op appears (matmul, attention each get their own section)
// - Contiguous element-wise ops fuse into one section

std::vector<KernelSection> TritonIRBuilder::identifySections(
    NativeSlot* slots, int startSlot, int endSlot,
    NDArray** outputSlots, int totalOutputSlots,
    NDArray** externalInputs, int numExternalInputs) {

  std::vector<KernelSection> sections;
  int segSize = endSlot - startSlot + 1;
  if (segSize == 0) return sections;

  // Helper: resolve source index to NDArray
  auto resolveArray = [&](int srcIdx) -> NDArray* {
    if (srcIdx < 0) {
      int extIdx = -(srcIdx + 1);
      if (extIdx < numExternalInputs && externalInputs) return externalInputs[extIdx];
    } else if (srcIdx < totalOutputSlots && outputSlots) {
      return outputSlots[srcIdx];
    }
    return nullptr;
  };

  // Helper: classify a category into a section type
  auto categoryToSectionType = [](TritonOpCategory cat, const std::string& opName) -> KernelSectionType {
    switch (cat) {
      case TritonOpCategory::BINARY_ELEMENTWISE:
      case TritonOpCategory::UNARY_ELEMENTWISE:
      case TritonOpCategory::COMPARISON:
      case TritonOpCategory::LOGICAL:
      case TritonOpCategory::TERNARY:
      case TritonOpCategory::CAST:
        return KernelSectionType::ELEMENTWISE;
      case TritonOpCategory::IDENTITY:
        return KernelSectionType::IDENTITY;
      case TritonOpCategory::MATMUL:
        return KernelSectionType::MATMUL;
      case TritonOpCategory::FUSED_ATTENTION:
        return KernelSectionType::FUSED_ATTENTION;
      case TritonOpCategory::REDUCTION:
        return KernelSectionType::REDUCTION;
      case TritonOpCategory::NORMALIZATION:
        return KernelSectionType::NORMALIZATION;
      case TritonOpCategory::SHAPE_MANIPULATION:
        return KernelSectionType::SHAPE_MANIPULATION;
      case TritonOpCategory::CONSTANT_GENERATION:
        return KernelSectionType::CONSTANT_GENERATION;
      case TritonOpCategory::CONVOLUTION:
        return KernelSectionType::CONVOLUTION;
      case TritonOpCategory::DATA_MOVEMENT: {
        // Sub-classify data movement ops
        std::string lower = opName;
        std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
        if (lower.find("gather_nd") != std::string::npos || lower == "gathernd")
          return KernelSectionType::GATHER_ND;
        if (lower.find("gather") != std::string::npos)
          return KernelSectionType::GATHER;
        if (lower.find("concat") != std::string::npos)
          return KernelSectionType::CONCAT;
        if (lower.find("split_v") != std::string::npos || lower == "splitv")
          return KernelSectionType::SPLIT_V;
        if (lower.find("split") != std::string::npos)
          return KernelSectionType::SPLIT;
        if (lower.find("stack") != std::string::npos)
          return KernelSectionType::STACK;
        if (lower.find("strided_slice") != std::string::npos)
          return KernelSectionType::STRIDED_SLICE;
        if (lower.find("tile") != std::string::npos)
          return KernelSectionType::TILE;
        if (lower.find("scatter_nd_update") != std::string::npos)
          return KernelSectionType::SCATTER_ND_UPDATE;
        if (lower.find("scatter_nd") != std::string::npos)
          return KernelSectionType::SCATTER_ND;
        return KernelSectionType::GATHER;  // Default data movement
      }
      default:
        return KernelSectionType::ELEMENTWISE;
    }
  };

  // Helper: check if a section type can be merged with element-wise
  auto canMergeWithElementwise = [](KernelSectionType type) -> bool {
    switch (type) {
      case KernelSectionType::ELEMENTWISE:
      case KernelSectionType::IDENTITY:
      case KernelSectionType::CONSTANT_GENERATION:
      case KernelSectionType::SHAPE_MANIPULATION:
      case KernelSectionType::REDUCTION:
      case KernelSectionType::NORMALIZATION:
        return true;
      default:
        return false;
    }
  };

  KernelSection currentSection;
  currentSection.startSlot = startSlot;
  currentSection.endSlot = startSlot;
  currentSection.numOps = 0;

  auto firstCat = getOpCategory(slots[startSlot].opName);
  currentSection.type = categoryToSectionType(firstCat, slots[startSlot].opName);

  for (int i = startSlot; i <= endSlot; i++) {
    auto cat = getOpCategory(slots[i].opName);
    auto sectionType = categoryToSectionType(cat, slots[i].opName);

    bool startNewSection = false;

    if (i == startSlot) {
      // First op — always part of current section
      startNewSection = false;
    } else if (sectionType == KernelSectionType::MATMUL ||
               sectionType == KernelSectionType::FUSED_ATTENTION ||
               sectionType == KernelSectionType::CONVOLUTION) {
      // Non-element-wise ops always get their own section
      startNewSection = true;
    } else if (currentSection.type == KernelSectionType::MATMUL ||
               currentSection.type == KernelSectionType::FUSED_ATTENTION ||
               currentSection.type == KernelSectionType::CONVOLUTION) {
      // After a non-element-wise section, start a new one
      startNewSection = true;
    } else if (!canMergeWithElementwise(sectionType) && currentSection.type != sectionType) {
      // Data movement ops that don't merge with element-wise
      startNewSection = true;
    } else if (canMergeWithElementwise(currentSection.type) && canMergeWithElementwise(sectionType)) {
      // Both are element-wise compatible — merge
      startNewSection = false;
    }

    if (startNewSection && currentSection.numOps > 0) {
      // Finalize current section and start new one
      sections.push_back(currentSection);
      currentSection = KernelSection();
      currentSection.startSlot = i;
      currentSection.type = sectionType;
    }

    currentSection.endSlot = i;
    currentSection.numOps++;

    // Extract matmul dimensions
    if (sectionType == KernelSectionType::MATMUL) {
      currentSection.type = KernelSectionType::MATMUL;
      if (slots[i].numInputs >= 2) {
        NDArray* aArr = resolveArray(slots[i].inputSourceIndices[0]);
        NDArray* bArr = resolveArray(slots[i].inputSourceIndices[1]);
        if (aArr && aArr->rankOf() >= 2) {
          currentSection.matmulM = static_cast<int>(aArr->sizeAt(aArr->rankOf() - 2));
          currentSection.matmulK = static_cast<int>(aArr->sizeAt(aArr->rankOf() - 1));
        }
        if (bArr && bArr->rankOf() >= 2) {
          currentSection.matmulN = static_cast<int>(bArr->sizeAt(bArr->rankOf() - 1));
          if (currentSection.matmulK == 0)
            currentSection.matmulK = static_cast<int>(bArr->sizeAt(bArr->rankOf() - 2));
        }
      }
    }

    // Extract attention dimensions
    if (sectionType == KernelSectionType::FUSED_ATTENTION) {
      currentSection.type = KernelSectionType::FUSED_ATTENTION;
      if (slots[i].numInputs >= 1) {
        NDArray* qArr = resolveArray(slots[i].inputSourceIndices[0]);
        if (qArr && qArr->rankOf() >= 3) {
          int rank = qArr->rankOf();
          currentSection.headDim = static_cast<int>(qArr->sizeAt(rank - 1));
          currentSection.seqQ = static_cast<int>(qArr->sizeAt(rank - 2));
          if (rank >= 4) {
            currentSection.numHeads = static_cast<int>(qArr->sizeAt(rank - 3));
            currentSection.batchSize = 1;
            for (int d = 0; d < rank - 3; d++)
              currentSection.batchSize *= static_cast<int>(qArr->sizeAt(d));
          } else {
            currentSection.numHeads = 1;
            currentSection.batchSize = static_cast<int>(qArr->sizeAt(0));
          }
          currentSection.attentionScale = 1.0f / std::sqrt(static_cast<float>(currentSection.headDim));
        }
        if (slots[i].numInputs >= 2) {
          NDArray* kArr = resolveArray(slots[i].inputSourceIndices[1]);
          if (kArr && kArr->rankOf() >= 2) {
            currentSection.seqK = static_cast<int>(kArr->sizeAt(kArr->rankOf() - 2));
          }
        }
      }
    }

    // Extract gather axis from iArgs
    if (sectionType == KernelSectionType::GATHER || sectionType == KernelSectionType::GATHER_ND) {
      if (slots[i].numIArgs > 0 && slots[i].iArgs) {
        currentSection.gatherAxis = static_cast<int>(slots[i].iArgs[0]);
      }
    }

    // Extract concat axis
    if (sectionType == KernelSectionType::CONCAT) {
      if (slots[i].numIArgs > 0 && slots[i].iArgs) {
        currentSection.concatAxis = static_cast<int>(slots[i].iArgs[0]);
      }
    }
  }

  // Don't forget the last section
  if (currentSection.numOps > 0) {
    sections.push_back(currentSection);
  }

  // Compute grid requirement for each section
  int defaultBlockSize = 1024;
  for (auto& sec : sections) {
    sec.gridRequirement = computeSectionGrid(sec, defaultBlockSize);
  }

  return sections;
}

// ─── Section grid computation ───────────────────────────────────────────────

int TritonIRBuilder::computeSectionGrid(const KernelSection& section, int blockSize) {
  switch (section.type) {
    case KernelSectionType::MATMUL: {
      int gridM = (section.matmulM + section.blockM - 1) / section.blockM;
      int gridN = (section.matmulN + section.blockN - 1) / section.blockN;
      return gridM * gridN;
    }
    case KernelSectionType::FUSED_ATTENTION: {
      int batchHeads = section.batchSize * section.numHeads;
      int blockM = 64;
      int gridQ = (section.seqQ + blockM - 1) / blockM;
      return batchHeads * gridQ;
    }
    case KernelSectionType::ELEMENTWISE:
    case KernelSectionType::IDENTITY:
    case KernelSectionType::CONSTANT_GENERATION:
    case KernelSectionType::SHAPE_MANIPULATION:
    case KernelSectionType::REDUCTION:
    case KernelSectionType::NORMALIZATION:
    case KernelSectionType::GATHER:
    case KernelSectionType::GATHER_ND:
    case KernelSectionType::CONCAT:
    case KernelSectionType::SPLIT:
    case KernelSectionType::SPLIT_V:
    case KernelSectionType::STACK:
    case KernelSectionType::STRIDED_SLICE:
    case KernelSectionType::TILE:
    case KernelSectionType::SCATTER_ND:
    case KernelSectionType::SCATTER_ND_UPDATE:
    case KernelSectionType::CONVOLUTION:
    default:
      // 1D grid — estimate from output element count
      // Conservative: assume largest output in the section
      return 256;  // Will be recomputed at launch time based on actual n_elements
  }
}

// ─── Grid sync emission ─────────────────────────────────────────────────────
// Emit a cooperative grid-wide synchronization barrier using inline PTX.
// Uses a global atomic counter + spin loop: each block atomically increments
// the counter, then spins until the counter reaches numBlocks.

static int gridSyncCounter_ = 0;

void TritonIRBuilder::emitGridSync(mlir::OpBuilder& builder, mlir::Location loc,
                                    mlir::Value syncCounterPtr, mlir::Value numBlocksVal) {
  // Cooperative grid sync using atomic counter + spin barrier in inline PTX.
  // Each block's thread 0 atomically increments a global counter, then spins
  // until all blocks have arrived. Requires cuLaunchCooperativeKernel for
  // co-residency guarantee.
  //
  // Protocol:
  //   membar.gl;                            // Flush pending global stores
  //   bar.sync 0;                           // CTA barrier (all threads done)
  //   if (threadIdx.x == 0):
  //     atom.global.add counter, 1          // Arrive
  //     while (load(counter) < numBlocks);  // Spin wait
  //   bar.sync 0;                           // Propagate to all threads

  // Emit inline PTX for the full grid sync protocol.
  // syncCounterPtr is a pointer to a global u32 counter (passed as kernel arg).
  // numBlocksVal is the total number of blocks in the cooperative grid.
  int syncId = gridSyncCounter_++;
  std::string labelName = "GRID_SYNC_SPIN_" + std::to_string(syncId);
  // Operand numbering: $0 = dummy output (=r), $1 = syncCounterPtr (l), $2 = numBlocks (r)
  std::string asmStr =
      "{\n"
      "  .reg .pred %p_t0, %p_loop;\n"
      "  .reg .b32 %r_tid, %r_cnt;\n"
      "  membar.gl;\n"
      "  bar.sync 0;\n"
      "  mov.u32 %r_tid, %tid.x;\n"
      "  setp.eq.u32 %p_t0, %r_tid, 0;\n"
      // CRITICAL: Initialize %p_loop to false for ALL threads.
      // PTX registers are NOT zero-initialized (per PTX ISA spec).
      // Without this, non-thread-0 threads have undefined %p_loop,
      // and if it's stale-true they enter an infinite spin loop
      // (the setp inside is predicated on %p_t0 so never updates %p_loop
      // for non-thread-0 threads) → bar.sync 0 deadlocks.
      "  setp.eq.u32 %p_loop, 0, 1;\n"  // 0 == 1 is false → %p_loop = false
      "  @%p_t0 atom.global.add.u32 %r_cnt, [$1], 1;\n"
      "  @%p_t0 add.u32 %r_cnt, %r_cnt, 1;\n"  // atom returns old value, add 1
      + labelName + ":\n"
      "  @%p_t0 ld.global.acquire.gpu.u32 %r_cnt, [$1];\n"
      "  @%p_t0 setp.lt.u32 %p_loop, %r_cnt, $2;\n"
      "  @%p_loop bra " + labelName + ";\n"
      "  bar.sync 0;\n"
      "}\n";

  // Use tt.elementwise_inline_asm with the counter pointer and numBlocks as operands.
  // ElementwiseInlineAsmOp requires at least 1 result, so we provide a dummy scalar i32 output.
  // Constraints: "=r" for dummy output, "l" for 64-bit pointer, "r" for 32-bit integer
  auto i32Type = builder.getI32Type();
  builder.create<mlir::triton::ElementwiseInlineAsmOp>(
      loc, /*resultTypes=*/mlir::TypeRange{i32Type},
      asmStr,
      /*constraints=*/"=r,l,r",
      /*isPure=*/false, /*pack=*/1,
      mlir::ValueRange{syncCounterPtr, numBlocksVal});
}


// ─── Gather section emitter ─────────────────────────────────────────────────
//
// Multi-dimensional gather on axis k:
//   data shape:    [D0, ..., D_{k-1}, D_k, D_{k+1}, ..., D_n]
//   indices shape: [I0, I1, ..., I_m]
//   output shape:  [D0, ..., D_{k-1}, I0, ..., I_m, D_{k+1}, ..., D_n]
//
// For each flat output element i:
//   innerDim = D_{k+1} * ... * D_n
//   numIndices = I0 * I1 * ... * I_m  (total number of index values)
//   indexSliceSize = numIndices * innerDim  (one "outer" slice of the output)
//
//   For axis=0: outerIdx=0, idxPos = i / innerDim, innerIdx = i % innerDim
//   General:    outerIdx = i / indexSliceSize
//               remaining = i % indexSliceSize
//               idxPos = remaining / innerDim
//               innerIdx = remaining % innerDim
//
//   dataOffset = outerIdx * (D_k * innerDim) + indices[idxPos] * innerDim + innerIdx

void TritonIRBuilder::emitGatherSection(mlir::OpBuilder& builder, mlir::Location loc,
                                         mlir::Value pid, int blockSize,
                                         mlir::Value dataPtr, mlir::Value indicesPtr,
                                         mlir::Value outputPtr, int axis,
                                         const std::vector<LongType>& dataShape,
                                         const std::vector<LongType>& indicesShape,
                                         int nElements) {
  auto i32Type = builder.getI32Type();
  auto i32TensorType = mlir::RankedTensorType::get({blockSize}, i32Type);

  // Derive element types from actual pointer arguments
  auto idxPtrType = mlir::cast<mlir::triton::PointerType>(indicesPtr.getType());
  auto dataPtrType = mlir::cast<mlir::triton::PointerType>(dataPtr.getType());
  auto outPtrType = mlir::cast<mlir::triton::PointerType>(outputPtr.getType());

  // Compute flat output element offsets for this block
  auto blockSizeConst = builder.create<mlir::arith::ConstantIntOp>(loc, blockSize, 32);
  auto offsetBase = builder.create<mlir::arith::MulIOp>(loc, pid, blockSizeConst);
  auto range = builder.create<mlir::triton::MakeRangeOp>(loc, i32TensorType, 0, blockSize);
  auto splatBase = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, offsetBase);
  auto offsets = builder.create<mlir::arith::AddIOp>(loc, splatBase, range);

  auto nElemConst = builder.create<mlir::arith::ConstantIntOp>(loc, nElements, 32);
  auto splatN = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, nElemConst);
  auto mask = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::slt, offsets, splatN);

  // Compute innerDim = product of data dimensions AFTER the gather axis
  LongType innerDim = 1;
  if (axis < static_cast<int>(dataShape.size())) {
    for (int d = axis + 1; d < static_cast<int>(dataShape.size()); d++) {
      innerDim *= dataShape[d];
    }
  }
  // Guard: if innerDim is 0 (empty shape or 0-sized dim), treat as scalar (1)
  if (innerDim <= 0) innerDim = 1;

  // Compute numIndices = total number of index values (product of indices shape)
  LongType numIndices = 1;
  for (auto s : indicesShape) numIndices *= s;
  if (numIndices <= 0) numIndices = 1;

  // Use fast path (flat 1D gather) when:
  //  - innerDim is 1 and axis is 0 (simple element-wise gather)
  //  - axis is out of bounds for dataShape (can't decompose)
  //  - dataShape is empty (scalar data)
  //  - nElements equals numIndices (output is 1:1 with indices, no inner stride)
  bool useFastPath = (innerDim == 1 && axis == 0) ||
                     (axis >= static_cast<int>(dataShape.size())) ||
                     dataShape.empty() ||
                     (nElements == static_cast<int>(numIndices));

  if (useFastPath) {
    // Fast path: 1D gather (scalar elements), no decomposition needed.
    // idxPos = offsets (each output element maps 1:1 to an index)
    // dataOffset = indices[idxPos]
    auto idxPtrTensorType = mlir::RankedTensorType::get({blockSize}, idxPtrType);
    auto splatIdxPtr = builder.create<mlir::triton::SplatOp>(loc, idxPtrTensorType, indicesPtr);
    auto idxPtrs = builder.create<mlir::triton::AddPtrOp>(loc, idxPtrTensorType, splatIdxPtr, offsets);
    auto rawIndices = builder.create<mlir::triton::LoadOp>(loc,
        idxPtrs.getResult(), mask.getResult(), mlir::Value(),
        mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL, false);
    mlir::Value indices = castTo(builder, loc, rawIndices, i32Type);

    auto dataPtrTensorType = mlir::RankedTensorType::get({blockSize}, dataPtrType);
    auto splatDataPtr = builder.create<mlir::triton::SplatOp>(loc, dataPtrTensorType, dataPtr);
    auto dataPtrs = builder.create<mlir::triton::AddPtrOp>(loc, dataPtrTensorType, splatDataPtr, indices);
    auto gathered = builder.create<mlir::triton::LoadOp>(loc,
        dataPtrs.getResult(), mask.getResult(), mlir::Value(),
        mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL, false);

    auto outPtrTensorType = mlir::RankedTensorType::get({blockSize}, outPtrType);
    auto splatOutPtr = builder.create<mlir::triton::SplatOp>(loc, outPtrTensorType, outputPtr);
    auto outPtrs = builder.create<mlir::triton::AddPtrOp>(loc, outPtrTensorType, splatOutPtr, offsets);
    mlir::Value storeVal = castTo(builder, loc, gathered, outPtrType.getPointeeType());
    builder.create<mlir::triton::StoreOp>(loc, outPtrs, storeVal, mask,
                                           mlir::triton::CacheModifier::NONE,
                                           mlir::triton::EvictionPolicy::NORMAL);
  } else {
    // Multi-dimensional gather: decompose flat output index into components.
    //
    // For axis=0:
    //   idxPos = offsets / innerDim
    //   innerIdx = offsets % innerDim
    //   dataOffset = indices[idxPos] * innerDim + innerIdx
    //
    // General axis:
    //   indexSliceSize = numIndices * innerDim
    //   outerIdx = offsets / indexSliceSize
    //   remaining = offsets % indexSliceSize
    //   idxPos = remaining / innerDim
    //   innerIdx = remaining % innerDim
    //   dataOffset = outerIdx * (dataShape[axis] * innerDim) + indices[idxPos] * innerDim + innerIdx

    auto innerDimConst = builder.create<mlir::arith::ConstantIntOp>(loc, static_cast<int>(innerDim), 32);
    auto splatInnerDim = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, innerDimConst);

    mlir::Value idxPos;
    mlir::Value innerIdx;
    mlir::Value outerContrib;  // outerIdx * (D_k * innerDim), or 0 for axis=0

    if (axis == 0) {
      // axis=0: no outer dimension
      idxPos = builder.create<mlir::arith::DivSIOp>(loc, offsets, splatInnerDim);
      innerIdx = builder.create<mlir::arith::RemSIOp>(loc, offsets, splatInnerDim);
      // Zero constant for outer contribution
      auto zeroConst = builder.create<mlir::arith::ConstantIntOp>(loc, 0, 32);
      outerContrib = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, zeroConst);
    } else {
      LongType indexSliceSize = numIndices * innerDim;
      auto indexSliceSizeConst = builder.create<mlir::arith::ConstantIntOp>(
          loc, static_cast<int>(indexSliceSize), 32);
      auto splatISS = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, indexSliceSizeConst);
      auto outerIdx = builder.create<mlir::arith::DivSIOp>(loc, offsets, splatISS);
      auto remaining = builder.create<mlir::arith::RemSIOp>(loc, offsets, splatISS);
      idxPos = builder.create<mlir::arith::DivSIOp>(loc, remaining, splatInnerDim);
      innerIdx = builder.create<mlir::arith::RemSIOp>(loc, remaining, splatInnerDim);

      // outerContrib = outerIdx * (dataShape[axis] * innerDim)
      LongType axisDimSize = (axis < static_cast<int>(dataShape.size())) ? dataShape[axis] : 1;
      LongType axisStride = axisDimSize * innerDim;
      auto axisStrideConst = builder.create<mlir::arith::ConstantIntOp>(
          loc, static_cast<int>(axisStride), 32);
      auto splatAxisStride = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, axisStrideConst);
      outerContrib = builder.create<mlir::arith::MulIOp>(loc, outerIdx, splatAxisStride);
    }

    // Load indices: indices[idxPos]
    auto idxPtrTensorType = mlir::RankedTensorType::get({blockSize}, idxPtrType);
    auto splatIdxPtr = builder.create<mlir::triton::SplatOp>(loc, idxPtrTensorType, indicesPtr);
    auto idxPtrs = builder.create<mlir::triton::AddPtrOp>(loc, idxPtrTensorType, splatIdxPtr, idxPos);
    auto rawIndices = builder.create<mlir::triton::LoadOp>(loc,
        idxPtrs.getResult(), mask.getResult(), mlir::Value(),
        mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL, false);
    mlir::Value gatheredIdx = castTo(builder, loc, rawIndices, i32Type);

    // dataOffset = outerContrib + gatheredIdx * innerDim + innerIdx
    auto scaledIdx = builder.create<mlir::arith::MulIOp>(loc, gatheredIdx, splatInnerDim);
    auto partialOffset = builder.create<mlir::arith::AddIOp>(loc, outerContrib, scaledIdx);
    auto dataOffset = builder.create<mlir::arith::AddIOp>(loc, partialOffset, innerIdx);

    // Load gathered data: data[dataOffset]
    auto dataPtrTensorType = mlir::RankedTensorType::get({blockSize}, dataPtrType);
    auto splatDataPtr = builder.create<mlir::triton::SplatOp>(loc, dataPtrTensorType, dataPtr);
    auto dataPtrs = builder.create<mlir::triton::AddPtrOp>(loc, dataPtrTensorType, splatDataPtr, dataOffset);
    auto gathered = builder.create<mlir::triton::LoadOp>(loc,
        dataPtrs.getResult(), mask.getResult(), mlir::Value(),
        mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL, false);

    // Store result
    auto outPtrTensorType = mlir::RankedTensorType::get({blockSize}, outPtrType);
    auto splatOutPtr = builder.create<mlir::triton::SplatOp>(loc, outPtrTensorType, outputPtr);
    auto outPtrs = builder.create<mlir::triton::AddPtrOp>(loc, outPtrTensorType, splatOutPtr, offsets);
    mlir::Value storeVal = castTo(builder, loc, gathered, outPtrType.getPointeeType());
    builder.create<mlir::triton::StoreOp>(loc, outPtrs, storeVal, mask,
                                           mlir::triton::CacheModifier::NONE,
                                           mlir::triton::EvictionPolicy::NORMAL);
  }
}

// ─── Concat section emitter ─────────────────────────────────────────────────

void TritonIRBuilder::emitConcatSection(mlir::OpBuilder& builder, mlir::Location loc,
                                         mlir::Value pid, int blockSize,
                                         const std::vector<mlir::Value>& inputPtrs,
                                         mlir::Value outputPtr, int axis,
                                         const std::vector<std::vector<LongType>>& inputShapes,
                                         int nElements) {
  auto i32Type = builder.getI32Type();
  auto i32TensorType = mlir::RankedTensorType::get({blockSize}, i32Type);

  // Derive element type from the output pointer (NOT hardcoded f32)
  auto outPtrType = mlir::cast<mlir::triton::PointerType>(outputPtr.getType());
  auto elemType = outPtrType.getPointeeType();
  auto elemTensorType = mlir::RankedTensorType::get({blockSize}, elemType);

  auto blockSizeConst = builder.create<mlir::arith::ConstantIntOp>(loc, blockSize, 32);
  auto offsetBase = builder.create<mlir::arith::MulIOp>(loc, pid, blockSizeConst);
  auto range = builder.create<mlir::triton::MakeRangeOp>(loc, i32TensorType, 0, blockSize);
  auto splatBase = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, offsetBase);
  auto offsets = builder.create<mlir::arith::AddIOp>(loc, splatBase, range);

  auto nElemConst = builder.create<mlir::arith::ConstantIntOp>(loc, nElements, 32);
  auto splatN = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, nElemConst);
  auto mask = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::slt, offsets, splatN);

  // For concat along axis=0 (flattened), just copy sequentially from each input.
  // Start with zeros, then conditionally select from each input
  mlir::Value result = splatConstantF32(builder, loc, elemTensorType, 0.0f);

  int cumulativeOffset = 0;
  for (size_t inp = 0; inp < inputPtrs.size(); inp++) {
    int inputLen = 1;
    if (inp < inputShapes.size()) {
      inputLen = 1;
      for (auto dim : inputShapes[inp]) inputLen *= static_cast<int>(dim);
    }

    auto startConst = builder.create<mlir::arith::ConstantIntOp>(loc, cumulativeOffset, 32);
    auto endConst = builder.create<mlir::arith::ConstantIntOp>(loc, cumulativeOffset + inputLen, 32);
    auto splatStart = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, startConst);
    auto splatEnd = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, endConst);

    auto geStart = builder.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::sge, offsets, splatStart);
    auto ltEnd = builder.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::slt, offsets, splatEnd);
    auto inRange = builder.create<mlir::arith::AndIOp>(loc, geStart, ltEnd);
    auto loadMask = builder.create<mlir::arith::AndIOp>(loc, mask, inRange);

    // Compute local offset within this input
    auto localOffsets = builder.create<mlir::arith::SubIOp>(loc, offsets, splatStart);

    // Derive pointer type from each input (may differ from output type)
    auto inPtrType = mlir::cast<mlir::triton::PointerType>(inputPtrs[inp].getType());
    auto inPtrTensorType = mlir::RankedTensorType::get({blockSize}, inPtrType);
    auto splatInPtr = builder.create<mlir::triton::SplatOp>(loc, inPtrTensorType, inputPtrs[inp]);
    auto inPtrs = builder.create<mlir::triton::AddPtrOp>(loc, inPtrTensorType, splatInPtr, localOffsets);
    auto loaded = builder.create<mlir::triton::LoadOp>(loc,
        inPtrs.getResult(), loadMask.getResult(), mlir::Value(),
        mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL, false);

    // Cast to output element type if input type differs
    mlir::Value castLoaded = castTo(builder, loc, loaded, elemType);

    // Select: if in range, use loaded value; otherwise keep current result
    result = builder.create<mlir::arith::SelectOp>(loc, inRange, castLoaded, result);

    cumulativeOffset += inputLen;
  }

  // Store result
  auto outPtrTensorType = mlir::RankedTensorType::get({blockSize}, outPtrType);
  auto splatOutPtr = builder.create<mlir::triton::SplatOp>(loc, outPtrTensorType, outputPtr);
  auto outPtrs = builder.create<mlir::triton::AddPtrOp>(loc, outPtrTensorType, splatOutPtr, offsets);
  builder.create<mlir::triton::StoreOp>(loc, outPtrs, result, mask,
                                         mlir::triton::CacheModifier::NONE,
                                         mlir::triton::EvictionPolicy::NORMAL);
}

// ─── Slice section emitter ──────────────────────────────────────────────────

void TritonIRBuilder::emitSliceSection(mlir::OpBuilder& builder, mlir::Location loc,
                                        mlir::Value pid, int blockSize,
                                        mlir::Value inputPtr, mlir::Value outputPtr,
                                        const std::vector<int>& begins,
                                        const std::vector<int>& ends,
                                        const std::vector<int>& strides,
                                        const std::vector<LongType>& inputShape,
                                        int nElements) {
  auto i32Type = builder.getI32Type();
  auto i32TensorType = mlir::RankedTensorType::get({blockSize}, i32Type);

  auto blockSizeConst = builder.create<mlir::arith::ConstantIntOp>(loc, blockSize, 32);
  auto offsetBase = builder.create<mlir::arith::MulIOp>(loc, pid, blockSizeConst);
  auto range = builder.create<mlir::triton::MakeRangeOp>(loc, i32TensorType, 0, blockSize);
  auto splatBase = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, offsetBase);
  auto offsets = builder.create<mlir::arith::AddIOp>(loc, splatBase, range);

  auto nElemConst = builder.create<mlir::arith::ConstantIntOp>(loc, nElements, 32);
  auto splatN = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, nElemConst);
  auto mask = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::slt, offsets, splatN);

  // For 1D slice: result = load(input + begin + offsets * stride)
  int begin = begins.empty() ? 0 : begins[0];
  int stride = strides.empty() ? 1 : strides[0];

  auto beginConst = builder.create<mlir::arith::ConstantIntOp>(loc, begin, 32);
  auto splatBegin = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, beginConst);

  mlir::Value srcOffsets;
  if (stride == 1) {
    srcOffsets = builder.create<mlir::arith::AddIOp>(loc, offsets, splatBegin);
  } else {
    auto strideConst = builder.create<mlir::arith::ConstantIntOp>(loc, stride, 32);
    auto splatStride = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, strideConst);
    auto strided = builder.create<mlir::arith::MulIOp>(loc, offsets, splatStride);
    srcOffsets = builder.create<mlir::arith::AddIOp>(loc, strided, splatBegin);
  }

  // Derive pointer types from actual arguments (NOT hardcoded f32)
  auto inPtrType = mlir::cast<mlir::triton::PointerType>(inputPtr.getType());
  auto outPtrType = mlir::cast<mlir::triton::PointerType>(outputPtr.getType());
  auto inPtrTensorType = mlir::RankedTensorType::get({blockSize}, inPtrType);
  auto outPtrTensorType = mlir::RankedTensorType::get({blockSize}, outPtrType);

  auto splatInPtr = builder.create<mlir::triton::SplatOp>(loc, inPtrTensorType, inputPtr);
  auto inPtrs = builder.create<mlir::triton::AddPtrOp>(loc, inPtrTensorType, splatInPtr, srcOffsets);
  auto loaded = builder.create<mlir::triton::LoadOp>(loc,
      inPtrs.getResult(), mask.getResult(), mlir::Value(),
      mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL, false);

  // Cast loaded data to output element type if needed
  mlir::Value storeVal = castTo(builder, loc, loaded, outPtrType.getPointeeType());
  auto splatOutPtr = builder.create<mlir::triton::SplatOp>(loc, outPtrTensorType, outputPtr);
  auto outPtrs = builder.create<mlir::triton::AddPtrOp>(loc, outPtrTensorType, splatOutPtr, offsets);
  builder.create<mlir::triton::StoreOp>(loc, outPtrs, storeVal, mask,
                                         mlir::triton::CacheModifier::NONE,
                                         mlir::triton::EvictionPolicy::NORMAL);
}

// ─── Tile section emitter ───────────────────────────────────────────────────

void TritonIRBuilder::emitTileSection(mlir::OpBuilder& builder, mlir::Location loc,
                                       mlir::Value pid, int blockSize,
                                       mlir::Value inputPtr, mlir::Value outputPtr,
                                       const std::vector<LongType>& inputShape,
                                       const std::vector<int>& repeats,
                                       int nElements) {
  auto i32Type = builder.getI32Type();
  auto i32TensorType = mlir::RankedTensorType::get({blockSize}, i32Type);

  auto blockSizeConst = builder.create<mlir::arith::ConstantIntOp>(loc, blockSize, 32);
  auto offsetBase = builder.create<mlir::arith::MulIOp>(loc, pid, blockSizeConst);
  auto range = builder.create<mlir::triton::MakeRangeOp>(loc, i32TensorType, 0, blockSize);
  auto splatBase = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, offsetBase);
  auto offsets = builder.create<mlir::arith::AddIOp>(loc, splatBase, range);

  auto nElemConst = builder.create<mlir::arith::ConstantIntOp>(loc, nElements, 32);
  auto splatN = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, nElemConst);
  auto mask = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::slt, offsets, splatN);

  // Tile: result = load(input + (offsets % input_size))
  int inputSize = 1;
  for (auto dim : inputShape) inputSize *= static_cast<int>(dim);
  if (inputSize == 0) inputSize = 1;

  auto inputSizeConst = builder.create<mlir::arith::ConstantIntOp>(loc, inputSize, 32);
  auto splatInputSize = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, inputSizeConst);
  auto modOffsets = builder.create<mlir::arith::RemSIOp>(loc, offsets, splatInputSize);

  // Derive pointer types from actual arguments (NOT hardcoded f32)
  auto inPtrType = mlir::cast<mlir::triton::PointerType>(inputPtr.getType());
  auto outPtrType = mlir::cast<mlir::triton::PointerType>(outputPtr.getType());
  auto inPtrTensorType = mlir::RankedTensorType::get({blockSize}, inPtrType);
  auto outPtrTensorType = mlir::RankedTensorType::get({blockSize}, outPtrType);

  auto splatInPtr = builder.create<mlir::triton::SplatOp>(loc, inPtrTensorType, inputPtr);
  auto inPtrs = builder.create<mlir::triton::AddPtrOp>(loc, inPtrTensorType, splatInPtr, modOffsets);
  auto loaded = builder.create<mlir::triton::LoadOp>(loc,
      inPtrs.getResult(), mask.getResult(), mlir::Value(),
      mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL, false);

  mlir::Value storeVal = castTo(builder, loc, loaded, outPtrType.getPointeeType());
  auto splatOutPtr = builder.create<mlir::triton::SplatOp>(loc, outPtrTensorType, outputPtr);
  auto outPtrs = builder.create<mlir::triton::AddPtrOp>(loc, outPtrTensorType, splatOutPtr, offsets);
  builder.create<mlir::triton::StoreOp>(loc, outPtrs, storeVal, mask,
                                         mlir::triton::CacheModifier::NONE,
                                         mlir::triton::EvictionPolicy::NORMAL);
}

// ─── Shape manipulation section emitter ─────────────────────────────────────
// For contiguous data, reshape/flatten/expand_dims/squeeze are just copies.
// Permute/transpose requires stride recomputation.

void TritonIRBuilder::emitShapeManipulationSection(mlir::OpBuilder& builder, mlir::Location loc,
                                                    mlir::Value pid, int blockSize,
                                                    mlir::Value inputPtr, mlir::Value outputPtr,
                                                    const std::string& opName,
                                                    const std::vector<LongType>& inputShape,
                                                    const std::vector<LongType>& outputShape,
                                                    const std::vector<int>& permutation,
                                                    int nElements) {
  auto i32Type = builder.getI32Type();
  auto i32TensorType = mlir::RankedTensorType::get({blockSize}, i32Type);

  auto blockSizeConst = builder.create<mlir::arith::ConstantIntOp>(loc, blockSize, 32);
  auto offsetBase = builder.create<mlir::arith::MulIOp>(loc, pid, blockSizeConst);
  auto range = builder.create<mlir::triton::MakeRangeOp>(loc, i32TensorType, 0, blockSize);
  auto splatBase = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, offsetBase);
  auto offsets = builder.create<mlir::arith::AddIOp>(loc, splatBase, range);

  auto nElemConst = builder.create<mlir::arith::ConstantIntOp>(loc, nElements, 32);
  auto splatN = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, nElemConst);
  auto mask = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::slt, offsets, splatN);

  // Derive pointer types from actual arguments
  auto inPtrType = mlir::cast<mlir::triton::PointerType>(inputPtr.getType());
  auto outPtrType = mlir::cast<mlir::triton::PointerType>(outputPtr.getType());
  auto inPtrTensorType = mlir::RankedTensorType::get({blockSize}, inPtrType);
  auto outPtrTensorType = mlir::RankedTensorType::get({blockSize}, outPtrType);

  std::string opLower = opName;
  std::transform(opLower.begin(), opLower.end(), opLower.begin(), ::tolower);

  bool isPermute = (opLower == "permute" || opLower == "transpose");

  if (isPermute && !permutation.empty() && inputShape.size() >= 2) {
    if (inputShape.size() == 2 && permutation.size() == 2 &&
        permutation[0] == 1 && permutation[1] == 0) {
      int rows = static_cast<int>(inputShape[0]);
      int cols = static_cast<int>(inputShape[1]);

      auto rowsConst = builder.create<mlir::arith::ConstantIntOp>(loc, rows, 32);
      auto colsConst = builder.create<mlir::arith::ConstantIntOp>(loc, cols, 32);
      auto splatRows = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, rowsConst);
      auto splatCols = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, colsConst);

      auto rOut = builder.create<mlir::arith::DivSIOp>(loc, offsets, splatRows);
      auto cOut = builder.create<mlir::arith::RemSIOp>(loc, offsets, splatRows);
      auto cTimesInputCols = builder.create<mlir::arith::MulIOp>(loc, cOut, splatCols);
      auto srcOffsets = builder.create<mlir::arith::AddIOp>(loc, cTimesInputCols, rOut);

      auto splatInPtr = builder.create<mlir::triton::SplatOp>(loc, inPtrTensorType, inputPtr);
      auto inPtrs = builder.create<mlir::triton::AddPtrOp>(loc, inPtrTensorType, splatInPtr, srcOffsets);
      auto loaded = builder.create<mlir::triton::LoadOp>(loc,
          inPtrs.getResult(), mask.getResult(), mlir::Value(),
          mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL, false);

      mlir::Value storeVal = castTo(builder, loc, loaded, outPtrType.getPointeeType());
      auto splatOutPtr = builder.create<mlir::triton::SplatOp>(loc, outPtrTensorType, outputPtr);
      auto outPtrs = builder.create<mlir::triton::AddPtrOp>(loc, outPtrTensorType, splatOutPtr, offsets);
      builder.create<mlir::triton::StoreOp>(loc, outPtrs, storeVal, mask,
                                             mlir::triton::CacheModifier::NONE,
                                             mlir::triton::EvictionPolicy::NORMAL);
      return;
    }
  }

  // Default: straight copy (reshape, flatten, expand_dims, squeeze, or general permute fallback)
  auto splatInPtr = builder.create<mlir::triton::SplatOp>(loc, inPtrTensorType, inputPtr);
  auto inPtrs = builder.create<mlir::triton::AddPtrOp>(loc, inPtrTensorType, splatInPtr, offsets);
  auto loaded = builder.create<mlir::triton::LoadOp>(loc,
      inPtrs.getResult(), mask.getResult(), mlir::Value(),
      mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL, false);

  mlir::Value storeVal = castTo(builder, loc, loaded, outPtrType.getPointeeType());
  auto splatOutPtr = builder.create<mlir::triton::SplatOp>(loc, outPtrTensorType, outputPtr);
  auto outPtrs = builder.create<mlir::triton::AddPtrOp>(loc, outPtrTensorType, splatOutPtr, offsets);
  builder.create<mlir::triton::StoreOp>(loc, outPtrs, storeVal, mask,
                                         mlir::triton::CacheModifier::NONE,
                                         mlir::triton::EvictionPolicy::NORMAL);
}

// ─── Per-element matmul fallback ────────────────────────────────────────────
// When cooperative launch is infeasible, compute matmul per-element without tt.dot.

void TritonIRBuilder::emitPerElementMatmul(mlir::OpBuilder& builder, mlir::Location loc,
                                            mlir::Value pid, int blockSize,
                                            mlir::Value aPtr, mlir::Value bPtr, mlir::Value cPtr,
                                            int M, int N, int K) {
  auto i32Type = builder.getI32Type();
  auto f32Type = builder.getF32Type();
  auto i32TensorType = mlir::RankedTensorType::get({blockSize}, i32Type);
  auto f32TensorType = mlir::RankedTensorType::get({blockSize}, f32Type);

  auto blockSizeConst = builder.create<mlir::arith::ConstantIntOp>(loc, blockSize, 32);
  auto offsetBase = builder.create<mlir::arith::MulIOp>(loc, pid, blockSizeConst);
  auto range = builder.create<mlir::triton::MakeRangeOp>(loc, i32TensorType, 0, blockSize);
  auto splatBase = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, offsetBase);
  auto offsets = builder.create<mlir::arith::AddIOp>(loc, splatBase, range);

  int totalElements = M * N;
  auto nElemConst = builder.create<mlir::arith::ConstantIntOp>(loc, totalElements, 32);
  auto splatN = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, nElemConst);
  auto mask = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::slt, offsets, splatN);

  // Each element of C[i,j] = sum_k A[i,k] * B[k,j]
  // Decompose offsets into row and column
  auto nConst = builder.create<mlir::arith::ConstantIntOp>(loc, N, 32);
  auto splatNConst = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, nConst);
  auto rowIndices = builder.create<mlir::arith::DivSIOp>(loc, offsets, splatNConst);
  auto colIndices = builder.create<mlir::arith::RemSIOp>(loc, offsets, splatNConst);

  // K-loop: accumulate A[row, k] * B[k, col] for k in [0, K)
  auto accInit = splatConstantF32(builder, loc, f32TensorType, 0.0f);
  auto kStart = builder.create<mlir::arith::ConstantIntOp>(loc, 0, 32);
  auto kEnd = builder.create<mlir::arith::ConstantIntOp>(loc, K, 32);
  auto kStep = builder.create<mlir::arith::ConstantIntOp>(loc, 1, 32);

  auto forOp = builder.create<mlir::scf::ForOp>(
      loc, kStart, kEnd, kStep, mlir::ValueRange{accInit});

  builder.setInsertionPointToStart(forOp.getBody());
  auto kIdx = forOp.getInductionVar();
  auto accIter = forOp.getBody()->getArgument(1);

  // A offset: row * K + k
  auto kConst = builder.create<mlir::arith::ConstantIntOp>(loc, K, 32);
  auto splatKConst = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, kConst);
  auto splatK = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, kIdx);
  auto aOffset = builder.create<mlir::arith::AddIOp>(loc,
      builder.create<mlir::arith::MulIOp>(loc, rowIndices, splatKConst), splatK);

  // B offset: k * N + col
  auto bOffset = builder.create<mlir::arith::AddIOp>(loc,
      builder.create<mlir::arith::MulIOp>(loc, splatK, splatNConst), colIndices);

  // Derive pointer types from actual arguments (NOT hardcoded f32)
  auto aPtrType = mlir::cast<mlir::triton::PointerType>(aPtr.getType());
  auto bPtrType = mlir::cast<mlir::triton::PointerType>(bPtr.getType());
  auto cPtrType = mlir::cast<mlir::triton::PointerType>(cPtr.getType());
  auto aPtrTensorType = mlir::RankedTensorType::get({blockSize}, aPtrType);
  auto bPtrTensorType = mlir::RankedTensorType::get({blockSize}, bPtrType);
  auto cPtrTensorType = mlir::RankedTensorType::get({blockSize}, cPtrType);

  auto splatAPtr = builder.create<mlir::triton::SplatOp>(loc, aPtrTensorType, aPtr);
  auto aPtrs = builder.create<mlir::triton::AddPtrOp>(loc, aPtrTensorType, splatAPtr, aOffset);
  auto aLoaded = builder.create<mlir::triton::LoadOp>(loc,
      aPtrs.getResult(), mask.getResult(), mlir::Value(),
      mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL, false);

  auto splatBPtr = builder.create<mlir::triton::SplatOp>(loc, bPtrTensorType, bPtr);
  auto bPtrs = builder.create<mlir::triton::AddPtrOp>(loc, bPtrTensorType, splatBPtr, bOffset);
  auto bLoaded = builder.create<mlir::triton::LoadOp>(loc,
      bPtrs.getResult(), mask.getResult(), mlir::Value(),
      mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL, false);

  // Cast loaded values to f32 for accumulation
  auto aVal = castTo(builder, loc, aLoaded, f32Type);
  auto bVal = castTo(builder, loc, bLoaded, f32Type);

  // acc += a * b
  auto prod = builder.create<mlir::arith::MulFOp>(loc, aVal, bVal);
  auto newAcc = builder.create<mlir::arith::AddFOp>(loc, accIter, prod);

  builder.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{newAcc});

  // After K-loop: store result (cast f32 accumulator to output type)
  builder.setInsertionPointAfter(forOp);
  auto finalAcc = forOp.getResult(0);
  mlir::Value storeVal = castTo(builder, loc, finalAcc, cPtrType.getPointeeType());

  auto splatCPtr = builder.create<mlir::triton::SplatOp>(loc, cPtrTensorType, cPtr);
  auto cPtrs = builder.create<mlir::triton::AddPtrOp>(loc, cPtrTensorType, splatCPtr, offsets);
  builder.create<mlir::triton::StoreOp>(loc, cPtrs, storeVal, mask,
                                         mlir::triton::CacheModifier::NONE,
                                         mlir::triton::EvictionPolicy::NORMAL);
}

// ─── Matmul section emitter (inline in mega-kernel) ─────────────────────────
// Adapted emitMatmulKernel for use within a sectioned kernel. Uses pid to
// derive 2D tile coordinates instead of GetProgramIdOp.

void TritonIRBuilder::emitMatmulSection(mlir::OpBuilder& builder, mlir::Location loc,
                                         mlir::Value pid, const KernelSection& section,
                                         mlir::Value aPtr, mlir::Value bPtr, mlir::Value cPtr) {
  int M = section.matmulM, N = section.matmulN, K = section.matmulK;
  int blockM = section.blockM, blockN = section.blockN, blockK = section.blockK;

  if (M == 0 || N == 0 || K == 0) {
    sd_printf("TritonIRBuilder::emitMatmulSection: invalid dimensions M=%d N=%d K=%d\n", M, N, K);
    return;
  }

  // Derive 2D tile indices from 1D pid
  // gridN = ceil(N / blockN)
  auto i32Type = builder.getI32Type();
  int gridN = (N + blockN - 1) / blockN;
  int gridM = (M + blockM - 1) / blockM;

  auto gridNConst = builder.create<mlir::arith::ConstantIntOp>(loc, gridN, 32);
  auto pidM = builder.create<mlir::arith::DivSIOp>(loc, pid, gridNConst);
  auto pidN = builder.create<mlir::arith::RemSIOp>(loc, pid, gridNConst);

  // Guard: only execute if pidM < gridM && pidN < gridN
  auto gridMConst = builder.create<mlir::arith::ConstantIntOp>(loc, gridM, 32);
  auto validM = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt, pidM, gridMConst);
  auto validN = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt, pidN, gridNConst);
  auto valid = builder.create<mlir::arith::AndIOp>(loc, validM, validN);

  auto ifOp = builder.create<mlir::scf::IfOp>(loc, valid, /*withElse=*/false);
  builder.setInsertionPointToStart(&ifOp.getThenRegion().front());

  // Now emit the matmul body using pidM/pidN as tile coordinates.
  // This is the same logic as emitMatmulKernel but without GetProgramIdOp.
  auto f32Type = builder.getF32Type();
  auto i1Type = builder.getI1Type();

  auto aPtrType = mlir::cast<mlir::triton::PointerType>(aPtr.getType());
  auto bPtrType = mlir::cast<mlir::triton::PointerType>(bPtr.getType());
  auto cPtrType = mlir::cast<mlir::triton::PointerType>(cPtr.getType());
  auto aElemType = aPtrType.getPointeeType();
  auto bElemType = bPtrType.getPointeeType();
  auto cElemType = cPtrType.getPointeeType();

  auto dotPrecision = mlir::triton::InputPrecision::TF32;
  if (!mlir::isa<mlir::Float32Type>(aElemType)) {
    dotPrecision = mlir::triton::InputPrecision::IEEE;
  }

  auto blockMConst = builder.create<mlir::arith::ConstantIntOp>(loc, blockM, 32);
  auto blockNConst = builder.create<mlir::arith::ConstantIntOp>(loc, blockN, 32);
  auto mOffset = builder.create<mlir::arith::MulIOp>(loc, pidM, blockMConst);
  auto nOffset = builder.create<mlir::arith::MulIOp>(loc, pidN, blockNConst);

  auto i32BmType = mlir::RankedTensorType::get({blockM}, i32Type);
  auto i32BnType = mlir::RankedTensorType::get({blockN}, i32Type);
  auto i32BkType = mlir::RankedTensorType::get({blockK}, i32Type);

  auto rangeM = builder.create<mlir::triton::MakeRangeOp>(loc, i32BmType, 0, blockM);
  auto rangeN = builder.create<mlir::triton::MakeRangeOp>(loc, i32BnType, 0, blockN);
  auto rangeK = builder.create<mlir::triton::MakeRangeOp>(loc, i32BkType, 0, blockK);

  auto splatMOffset = builder.create<mlir::triton::SplatOp>(loc, i32BmType, mOffset);
  auto mIndices = builder.create<mlir::arith::AddIOp>(loc, splatMOffset, rangeM);
  auto splatNOffset = builder.create<mlir::triton::SplatOp>(loc, i32BnType, nOffset);
  auto nIndices = builder.create<mlir::arith::AddIOp>(loc, splatNOffset, rangeN);

  auto accType = mlir::RankedTensorType::get({blockM, blockN}, f32Type);
  auto zeroAttr = builder.getFloatAttr(f32Type, 0.0);
  auto zeroScalar = builder.create<mlir::arith::ConstantOp>(loc, f32Type, zeroAttr);
  auto accInit = builder.create<mlir::triton::SplatOp>(loc, accType, zeroScalar);

  auto kStart = builder.create<mlir::arith::ConstantIntOp>(loc, 0, 32);
  auto kEnd = builder.create<mlir::arith::ConstantIntOp>(loc, K, 32);
  auto kStep = builder.create<mlir::arith::ConstantIntOp>(loc, blockK, 32);

  auto forOp = builder.create<mlir::scf::ForOp>(
      loc, kStart, kEnd, kStep, mlir::ValueRange{accInit});

  builder.setInsertionPointToStart(forOp.getBody());
  auto kIdxI32 = forOp.getInductionVar();
  auto accIter = forOp.getBody()->getArgument(1);

  auto splatKOffset = builder.create<mlir::triton::SplatOp>(loc, i32BkType, kIdxI32);
  auto kIndices = builder.create<mlir::arith::AddIOp>(loc, splatKOffset, rangeK);

  // Load A tile [BM, BK]
  auto kConst = builder.create<mlir::arith::ConstantIntOp>(loc, K, 32);
  auto mExpanded = builder.create<mlir::triton::ExpandDimsOp>(loc, mIndices, 1);
  auto kExpanded = builder.create<mlir::triton::ExpandDimsOp>(loc, kIndices, 0);

  auto i32BmBkType = mlir::RankedTensorType::get({blockM, blockK}, i32Type);
  auto kSplat = builder.create<mlir::triton::SplatOp>(loc, mlir::RankedTensorType::get({blockM, 1}, i32Type), kConst);
  auto mTimesK = builder.create<mlir::arith::MulIOp>(loc, mExpanded, kSplat);
  auto mTimesKBroadcast = builder.create<mlir::triton::BroadcastOp>(loc, i32BmBkType, mTimesK);
  auto kBroadcast = builder.create<mlir::triton::BroadcastOp>(loc, i32BmBkType, kExpanded);
  auto aOffsets = builder.create<mlir::arith::AddIOp>(loc, mTimesKBroadcast, kBroadcast);

  auto aPtrTensorType = mlir::RankedTensorType::get({blockM, blockK},
      mlir::triton::PointerType::get(aElemType, 1));
  auto aSplat = builder.create<mlir::triton::SplatOp>(loc, aPtrTensorType, aPtr);
  auto aPtrs = builder.create<mlir::triton::AddPtrOp>(loc, aPtrTensorType, aSplat, aOffsets);

  auto mConst = builder.create<mlir::arith::ConstantIntOp>(loc, M, 32);
  auto kConst2 = builder.create<mlir::arith::ConstantIntOp>(loc, K, 32);
  auto mConstSplat = builder.create<mlir::triton::SplatOp>(loc, i32BmType, mConst);
  auto kConstSplat = builder.create<mlir::triton::SplatOp>(loc, i32BkType, kConst2);
  auto mMask1D = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt,
      builder.create<mlir::arith::AddIOp>(loc, splatMOffset, rangeM), mConstSplat);
  auto kMask1D = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt,
      kIndices, kConstSplat);
  auto i1BmBkType = mlir::RankedTensorType::get({blockM, blockK}, i1Type);
  auto mMaskExp = builder.create<mlir::triton::ExpandDimsOp>(loc, mMask1D, 1);
  auto kMaskExp = builder.create<mlir::triton::ExpandDimsOp>(loc, kMask1D, 0);
  auto mMask2D = builder.create<mlir::triton::BroadcastOp>(loc, i1BmBkType, mMaskExp);
  auto kMask2D = builder.create<mlir::triton::BroadcastOp>(loc, i1BmBkType, kMaskExp);
  auto aMask = builder.create<mlir::arith::AndIOp>(loc, mMask2D, kMask2D);

  auto aLoaded = builder.create<mlir::triton::LoadOp>(loc,
      aPtrs.getResult(), aMask.getResult(), mlir::Value(),
      mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL, false);

  // Load B tile [BK, BN]
  auto nConst = builder.create<mlir::arith::ConstantIntOp>(loc, N, 32);
  auto kExpandedB = builder.create<mlir::triton::ExpandDimsOp>(loc, kIndices, 1);
  auto nExpandedB = builder.create<mlir::triton::ExpandDimsOp>(loc, nIndices, 0);
  auto i32BkBnType = mlir::RankedTensorType::get({blockK, blockN}, i32Type);
  auto nSplat = builder.create<mlir::triton::SplatOp>(loc, mlir::RankedTensorType::get({blockK, 1}, i32Type), nConst);
  auto kTimesN = builder.create<mlir::arith::MulIOp>(loc, kExpandedB, nSplat);
  auto kTimesNBroadcast = builder.create<mlir::triton::BroadcastOp>(loc, i32BkBnType, kTimesN);
  auto nBroadcastB = builder.create<mlir::triton::BroadcastOp>(loc, i32BkBnType, nExpandedB);
  auto bOffsets = builder.create<mlir::arith::AddIOp>(loc, kTimesNBroadcast, nBroadcastB);

  auto bPtrTensorType = mlir::RankedTensorType::get({blockK, blockN},
      mlir::triton::PointerType::get(bElemType, 1));
  auto bSplat = builder.create<mlir::triton::SplatOp>(loc, bPtrTensorType, bPtr);
  auto bPtrs = builder.create<mlir::triton::AddPtrOp>(loc, bPtrTensorType, bSplat, bOffsets);

  auto nConstSplat = builder.create<mlir::triton::SplatOp>(loc, i32BnType, nConst);
  auto nMask1D = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt,
      builder.create<mlir::arith::AddIOp>(loc, splatNOffset, rangeN), nConstSplat);
  auto i1BkBnType = mlir::RankedTensorType::get({blockK, blockN}, i1Type);
  auto kMaskExpB = builder.create<mlir::triton::ExpandDimsOp>(loc, kMask1D, 1);
  auto nMaskExpB = builder.create<mlir::triton::ExpandDimsOp>(loc, nMask1D, 0);
  auto kMask2DB = builder.create<mlir::triton::BroadcastOp>(loc, i1BkBnType, kMaskExpB);
  auto nMask2DB = builder.create<mlir::triton::BroadcastOp>(loc, i1BkBnType, nMaskExpB);
  auto bMask = builder.create<mlir::arith::AndIOp>(loc, kMask2DB, nMask2DB);

  auto bLoaded = builder.create<mlir::triton::LoadOp>(loc,
      bPtrs.getResult(), bMask.getResult(), mlir::Value(),
      mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL, false);

  auto dotResult = builder.create<mlir::triton::DotOp>(
      loc, accType, aLoaded, bLoaded, accIter,
      dotPrecision, 0);

  builder.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{dotResult});

  builder.setInsertionPointAfter(forOp);
  auto finalAcc = forOp.getResult(0);

  // Cast and store C
  mlir::Value storeVal = finalAcc;
  if (cElemType != f32Type) {
    auto cTileType = mlir::RankedTensorType::get({blockM, blockN}, cElemType);
    if (mlir::isa<mlir::FloatType>(cElemType)) {
      storeVal = builder.create<mlir::arith::TruncFOp>(loc, cTileType, finalAcc);
    }
  }

  auto i32BmBnType = mlir::RankedTensorType::get({blockM, blockN}, i32Type);
  auto mExpandedC = builder.create<mlir::triton::ExpandDimsOp>(loc, mIndices, 1);
  auto nExpandedC = builder.create<mlir::triton::ExpandDimsOp>(loc, nIndices, 0);
  auto nSplatC = builder.create<mlir::triton::SplatOp>(loc, mlir::RankedTensorType::get({blockM, 1}, i32Type), nConst);
  auto mTimesNC = builder.create<mlir::arith::MulIOp>(loc, mExpandedC, nSplatC);
  auto mTimesNCBroadcast = builder.create<mlir::triton::BroadcastOp>(loc, i32BmBnType, mTimesNC);
  auto nBroadcastC = builder.create<mlir::triton::BroadcastOp>(loc, i32BmBnType, nExpandedC);
  auto cOffsets = builder.create<mlir::arith::AddIOp>(loc, mTimesNCBroadcast, nBroadcastC);

  auto cPtrTensorType = mlir::RankedTensorType::get({blockM, blockN},
      mlir::triton::PointerType::get(cElemType, 1));
  auto cSplat = builder.create<mlir::triton::SplatOp>(loc, cPtrTensorType, cPtr);
  auto cPtrs = builder.create<mlir::triton::AddPtrOp>(loc, cPtrTensorType, cSplat, cOffsets);

  auto mConstC = builder.create<mlir::arith::ConstantIntOp>(loc, M, 32);
  auto nConstC = builder.create<mlir::arith::ConstantIntOp>(loc, N, 32);
  auto mConstSplatC = builder.create<mlir::triton::SplatOp>(loc, i32BmType, mConstC);
  auto nConstSplatC = builder.create<mlir::triton::SplatOp>(loc, i32BnType, nConstC);
  auto mMask1DC = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt, mIndices, mConstSplatC);
  auto nMask1DC = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt, nIndices, nConstSplatC);
  auto i1BmBnType = mlir::RankedTensorType::get({blockM, blockN}, i1Type);
  auto mMaskExpC = builder.create<mlir::triton::ExpandDimsOp>(loc, mMask1DC, 1);
  auto nMaskExpC = builder.create<mlir::triton::ExpandDimsOp>(loc, nMask1DC, 0);
  auto mMask2DC = builder.create<mlir::triton::BroadcastOp>(loc, i1BmBnType, mMaskExpC);
  auto nMask2DC = builder.create<mlir::triton::BroadcastOp>(loc, i1BmBnType, nMaskExpC);
  auto cMask = builder.create<mlir::arith::AndIOp>(loc, mMask2DC, nMask2DC);

  builder.create<mlir::triton::StoreOp>(loc, cPtrs, storeVal, cMask,
                                         mlir::triton::CacheModifier::NONE,
                                         mlir::triton::EvictionPolicy::NORMAL);

  // Move insertion point after the if block
  builder.setInsertionPointAfter(ifOp);
}

// ─── Diagnostics: section breakdown dump ────────────────────────────────────

void TritonIRBuilder::dumpSectionBreakdown(const std::vector<KernelSection>& sections,
                                            int startSlot, int endSlot,
                                            int maxSectionGrid, bool cooperativeLaunch) {
  static bool dumpEnabled = getenv("ND4J_TRITON_DUMP_SECTIONS") || getenv("ND4J_TRITON_VERBOSE");
  if (!dumpEnabled) return;

  sd_printf("=== Triton Kernel: seg[%d-%d] ===\n", startSlot, endSlot);
  for (size_t i = 0; i < sections.size(); i++) {
    auto& sec = sections[i];
    const char* typeName = "UNKNOWN";
    switch (sec.type) {
      case KernelSectionType::ELEMENTWISE:        typeName = "ELEMENTWISE"; break;
      case KernelSectionType::MATMUL:             typeName = "MATMUL"; break;
      case KernelSectionType::FUSED_ATTENTION:    typeName = "ATTENTION"; break;
      case KernelSectionType::REDUCTION:          typeName = "REDUCTION"; break;
      case KernelSectionType::NORMALIZATION:      typeName = "NORMALIZATION"; break;
      case KernelSectionType::GATHER:             typeName = "GATHER"; break;
      case KernelSectionType::GATHER_ND:          typeName = "GATHER_ND"; break;
      case KernelSectionType::CONCAT:             typeName = "CONCAT"; break;
      case KernelSectionType::SPLIT:              typeName = "SPLIT"; break;
      case KernelSectionType::SPLIT_V:            typeName = "SPLIT_V"; break;
      case KernelSectionType::STACK:              typeName = "STACK"; break;
      case KernelSectionType::STRIDED_SLICE:      typeName = "STRIDED_SLICE"; break;
      case KernelSectionType::TILE:               typeName = "TILE"; break;
      case KernelSectionType::SCATTER_ND:         typeName = "SCATTER_ND"; break;
      case KernelSectionType::SCATTER_ND_UPDATE:  typeName = "SCATTER_ND_UPDATE"; break;
      case KernelSectionType::SHAPE_MANIPULATION: typeName = "SHAPE_MANIP"; break;
      case KernelSectionType::CONSTANT_GENERATION:typeName = "CONST_GEN"; break;
      case KernelSectionType::CONVOLUTION:        typeName = "CONVOLUTION"; break;
      case KernelSectionType::IDENTITY:           typeName = "IDENTITY"; break;
    }
    sd_printf("Section %d: %-15s slots[%d-%d]  %d ops, grid=%d",
              (int)i, typeName, sec.startSlot, sec.endSlot, sec.numOps, sec.gridRequirement);
    if (sec.type == KernelSectionType::MATMUL) {
      sd_printf(", M=%d N=%d K=%d", sec.matmulM, sec.matmulN, sec.matmulK);
    }
    if (sec.type == KernelSectionType::FUSED_ATTENTION) {
      sd_printf(", B=%d H=%d seqQ=%d seqKV=%d headDim=%d",
                sec.batchSize, sec.numHeads, sec.seqQ, sec.seqK, sec.headDim);
    }
    sd_printf("\n", "");
  }
  sd_printf("Max section grid: %d\nCooperative launch: %s\n",
            maxSectionGrid, cooperativeLaunch ? "YES" : "NO");
}

// ─── Diagnostics: arg mapping dump ──────────────────────────────────────────

void TritonIRBuilder::dumpArgMapping(const std::vector<TritonKernelArg>& args,
                                      int startSlot, int endSlot,
                                      int eliminatedCount) {
  static bool dumpEnabled = getenv("ND4J_TRITON_DUMP_ARGS") || getenv("ND4J_TRITON_VERBOSE");
  if (!dumpEnabled) return;

  sd_printf("=== Arg Mapping: seg[%d-%d] ===\n", startSlot, endSlot);
  for (size_t i = 0; i < args.size(); i++) {
    auto& arg = args[i];
    sd_printf("Arg %3d: slot %4d %s dtype=%d shape=[",
              (int)i, arg.slotIndex, arg.isOutput ? "OUT" : "IN ", (int)arg.dtype);
    for (size_t d = 0; d < arg.shape.size(); d++) {
      sd_printf("%s%lld", d > 0 ? "," : "", (long long)arg.shape[d]);
    }
    sd_printf("]\n", "");
  }
  sd_printf("Eliminated: %d internal intermediates\nTotal args: %d%s\n",
            eliminatedCount, (int)args.size(),
            args.size() > 200 ? " (indirect)" : " (direct)");
}

// ─── Section emitter implementations ────────────────────────────────────────

void TritonIRBuilder::emitAttentionSection(mlir::OpBuilder& builder, mlir::Location loc,
                                            mlir::Value pid, const KernelSection& section,
                                            mlir::Value qPtr, mlir::Value kPtr,
                                            mlir::Value vPtr, mlir::Value outPtr) {
  // Delegate to the existing emitFusedAttentionKernel, which creates its own
  // GetProgramIdOp. For the sectioned kernel, this is called within an scf.if
  // guard so only blocks in the attention section's pid range execute it.
  // Note: emitFusedAttentionKernel uses its own pid0/pid1 from GetProgramIdOp.
  // In the cooperative kernel, we remap pid to the attention section's range.
  emitFusedAttentionKernel(builder, loc, qPtr, kPtr, vPtr, outPtr,
                            section.batchSize, section.numHeads,
                            section.seqQ, section.seqK,
                            section.headDim, section.attentionScale,
                            64, 64);
}

void TritonIRBuilder::emitSplitSection(mlir::OpBuilder& builder, mlir::Location loc,
                                        mlir::Value pid, int blockSize,
                                        mlir::Value inputPtr,
                                        const std::vector<mlir::Value>& outputPtrs,
                                        int axis, int numSplits,
                                        const std::vector<LongType>& inputShape,
                                        int nElements) {
  // Split is the inverse of concat: partition input into N equal chunks
  // For now, handle as N separate copy operations
  // Each output gets inputSize / numSplits elements
  if (numSplits <= 0 || inputShape.empty()) return;

  int totalInputSize = 1;
  for (auto dim : inputShape) totalInputSize *= static_cast<int>(dim);
  int chunkSize = totalInputSize / numSplits;

  for (int s = 0; s < numSplits && s < static_cast<int>(outputPtrs.size()); s++) {
    std::vector<int> begins = {s * chunkSize};
    std::vector<int> ends = {(s + 1) * chunkSize};
    std::vector<int> strides = {1};
    emitSliceSection(builder, loc, pid, blockSize, inputPtr, outputPtrs[s],
                     begins, ends, strides, inputShape, chunkSize);
  }
}

void TritonIRBuilder::emitScatterNdSection(mlir::OpBuilder& builder, mlir::Location loc,
                                            mlir::Value pid, int blockSize,
                                            mlir::Value dataPtr, mlir::Value indicesPtr,
                                            mlir::Value updatesPtr, mlir::Value outputPtr,
                                            const std::vector<LongType>& dataShape,
                                            int nElements) {
  // ScatterNd: copy data to output, then scatter updates at indexed positions.
  //
  // data:    [D0, D1, ..., Dn]       — base tensor (same shape as output)
  // indices: [numUpdates, indexDepth] — scatter positions
  // updates: [numUpdates, S0, S1, ...] — values to scatter (S = slice shape)
  // output:  [D0, D1, ..., Dn]       — result
  //
  // Phase 1: Copy all data[i] -> output[i]  (nElements = output length)
  // Phase 2: For each update element j (0..totalUpdateElems-1):
  //   updateIdx = j / sliceSize
  //   slicePos  = j % sliceSize
  //   flatIdx   = indices[updateIdx * indexDepth + 0] * stride0 + ... + slicePos
  //   output[flatIdx] = updates[j]
  //
  // For the simple 1D index case (indexDepth=1):
  //   flatIdx = indices[updateIdx] * sliceSize + slicePos

  auto i32Type = builder.getI32Type();
  auto i32TensorType = mlir::RankedTensorType::get({blockSize}, i32Type);

  auto dataPtrType = mlir::cast<mlir::triton::PointerType>(dataPtr.getType());
  auto idxPtrType = mlir::cast<mlir::triton::PointerType>(indicesPtr.getType());
  auto updPtrType = mlir::cast<mlir::triton::PointerType>(updatesPtr.getType());
  auto outPtrType = mlir::cast<mlir::triton::PointerType>(outputPtr.getType());
  auto dataPtrTensorType = mlir::RankedTensorType::get({blockSize}, dataPtrType);
  auto outPtrTensorType = mlir::RankedTensorType::get({blockSize}, outPtrType);
  auto idxPtrTensorType = mlir::RankedTensorType::get({blockSize}, idxPtrType);
  auto updPtrTensorType = mlir::RankedTensorType::get({blockSize}, updPtrType);

  auto blockSizeConst = builder.create<mlir::arith::ConstantIntOp>(loc, blockSize, 32);
  auto offsetBase = builder.create<mlir::arith::MulIOp>(loc, pid, blockSizeConst);
  auto range = builder.create<mlir::triton::MakeRangeOp>(loc, i32TensorType, 0, blockSize);
  auto splatBase = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, offsetBase);
  auto offsets = builder.create<mlir::arith::AddIOp>(loc, splatBase, range);

  // Phase 1: Copy data to output (nElements = output length)
  auto nElemConst = builder.create<mlir::arith::ConstantIntOp>(loc, nElements, 32);
  auto splatN = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, nElemConst);
  auto mask = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::slt, offsets, splatN);

  auto splatDataPtr = builder.create<mlir::triton::SplatOp>(loc, dataPtrTensorType, dataPtr);
  auto dataPtrs = builder.create<mlir::triton::AddPtrOp>(loc, dataPtrTensorType, splatDataPtr, offsets);
  auto dataLoaded = builder.create<mlir::triton::LoadOp>(loc,
      dataPtrs.getResult(), mask.getResult(), mlir::Value(),
      mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL, false);

  mlir::Value dataStoreVal = castTo(builder, loc, dataLoaded, outPtrType.getPointeeType());
  auto splatOutPtr = builder.create<mlir::triton::SplatOp>(loc, outPtrTensorType, outputPtr);
  auto outPtrs = builder.create<mlir::triton::AddPtrOp>(loc, outPtrTensorType, splatOutPtr, offsets);
  builder.create<mlir::triton::StoreOp>(loc, outPtrs, dataStoreVal, mask,
                                         mlir::triton::CacheModifier::NONE,
                                         mlir::triton::EvictionPolicy::NORMAL);

  // Phase 2: Scatter updates
  // Compute indexDepth (last dim of indices shape) and sliceSize (product of data dims after indexDepth)
  // For simplicity, assume indexDepth=1 for now (covers most common cases).
  // sliceSize = product of dataShape[1:]
  LongType sliceSize = 1;
  for (size_t d = 1; d < dataShape.size(); d++) sliceSize *= dataShape[d];
  if (sliceSize <= 0) sliceSize = 1;

  // totalUpdateElems = numUpdates * sliceSize (this is updates.lengthOf())
  // numUpdates is unknown at compile time but we can derive it:
  // numUpdates * sliceSize must equal updates.length, which we get from the mask.
  // For the kernel, we iterate over update elements using the same grid as output,
  // but with a separate mask based on the total update element count.
  // We don't have updates.length directly, so we express the scatter in terms of
  // the output grid: for each flat update element j, compute output position.

  // The key insight: we use a SECOND pass over the SAME grid but with a different mask.
  // totalUpdateElems = (nElements is output, but updates is typically smaller)
  // We pass the total update elements as a separate constant.
  // Since we don't have it at IR build time, derive from the grid:
  // Actually we DO know it at compile time from the arrays. But the function signature
  // only receives dataShape and nElements. We need the updates length.
  //
  // Alternative approach: iterate over ALL output elements, and for each element,
  // check if it should be overwritten. This is O(nElements) per update, too expensive.
  //
  // Better: iterate over update elements. We know sliceSize from dataShape.
  // We'll use the grid to cover update elements: the grid covers max(nElements, updateElems).
  // For Phase 2, mask with updateElems limit.
  // Since we don't have updateElems explicitly, compute it in the kernel:
  // updateElems = (we'd need it passed in)
  //
  // Simplest correct approach: Phase 2 iterates over the same nElements grid,
  // but only activates for positions that correspond to update elements.
  // For scatter_nd with indexDepth=1:
  //   For flat output position p: check if p's "row" (p / sliceSize) matches any index
  //   This is O(nElements * numUpdates) - not great.
  //
  // Most practical: Phase 2 is a separate grid over totalUpdateElems.
  // Since Triton requires a single grid, we use the SAME grid (nElements) and
  // only process elements within the update range.
  //
  // Simple 1D-index scatter: indices are [numUpdates] (or [numUpdates,1])
  // Each update i writes a slice of sliceSize elements starting at indices[i] * sliceSize
  // Total update elements = numUpdates * sliceSize
  // For flat j in [0, totalUpdateElems):
  //   updateIdx = j / sliceSize
  //   slicePos = j % sliceSize
  //   outPos = indices[updateIdx] * sliceSize + slicePos
  //   output[outPos] = updates[j]
  //
  // We process this using offsets (0..blockSize-1 per block), masked to totalUpdateElems.
  // But we need totalUpdateElems at compile time... we'll use nElements as upper bound
  // and add bounds checking.

  // For correctness: Phase 2 iterates over indices directly.
  // Load index for the current position, compute scatter target.
  auto sliceSizeConst = builder.create<mlir::arith::ConstantIntOp>(loc, static_cast<int>(sliceSize), 32);
  auto splatSliceSize = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, sliceSizeConst);

  // updateIdx = offsets / sliceSize
  auto updateIdx = builder.create<mlir::arith::DivSIOp>(loc, offsets, splatSliceSize);
  // slicePos = offsets % sliceSize
  auto slicePos = builder.create<mlir::arith::RemSIOp>(loc, offsets, splatSliceSize);

  // Load index: indices[updateIdx] (treat indices as flat array of index values)
  auto splatIdxPtr = builder.create<mlir::triton::SplatOp>(loc, idxPtrTensorType, indicesPtr);
  auto idxPtrs = builder.create<mlir::triton::AddPtrOp>(loc, idxPtrTensorType, splatIdxPtr, updateIdx);
  auto rawIndices = builder.create<mlir::triton::LoadOp>(loc,
      idxPtrs.getResult(), mask.getResult(), mlir::Value(),
      mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL, false);
  mlir::Value indices = castTo(builder, loc, rawIndices, i32Type);

  // outPos = indices[updateIdx] * sliceSize + slicePos
  auto scaledIdx = builder.create<mlir::arith::MulIOp>(loc, indices, splatSliceSize);
  auto outPos = builder.create<mlir::arith::AddIOp>(loc, scaledIdx, slicePos);

  // Bounds check: outPos must be in [0, nElements)
  auto outPosBoundsCheck = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::slt, outPos, splatN);
  auto outPosGe0 = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::sge, outPos,
      builder.create<mlir::triton::SplatOp>(loc, i32TensorType,
          builder.create<mlir::arith::ConstantIntOp>(loc, 0, 32)));
  auto scatterMask = builder.create<mlir::arith::AndIOp>(loc,
      builder.create<mlir::arith::AndIOp>(loc, mask, outPosBoundsCheck),
      outPosGe0);

  // Load update values: updates[offsets] (flat indexing)
  auto splatUpdPtr = builder.create<mlir::triton::SplatOp>(loc, updPtrTensorType, updatesPtr);
  auto updPtrs = builder.create<mlir::triton::AddPtrOp>(loc, updPtrTensorType, splatUpdPtr, offsets);
  auto updateVals = builder.create<mlir::triton::LoadOp>(loc,
      updPtrs.getResult(), mask.getResult(), mlir::Value(),
      mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL, false);

  // Scatter: output[outPos] = updates[offsets]
  mlir::Value updStoreVal = castTo(builder, loc, updateVals, outPtrType.getPointeeType());
  auto scatterPtrs = builder.create<mlir::triton::AddPtrOp>(loc, outPtrTensorType, splatOutPtr, outPos);
  builder.create<mlir::triton::StoreOp>(loc, scatterPtrs, updStoreVal, scatterMask,
                                         mlir::triton::CacheModifier::NONE,
                                         mlir::triton::EvictionPolicy::NORMAL);
}

void TritonIRBuilder::emitConvolutionSection(mlir::OpBuilder& builder, mlir::Location loc,
                                              mlir::Value pid, int blockSize,
                                              mlir::Value inputPtr, mlir::Value filterPtr,
                                              mlir::Value outputPtr,
                                              const std::vector<LongType>& inputShape,
                                              const std::vector<LongType>& filterShape,
                                              const std::vector<LongType>& outputShape,
                                              int strideH, int strideW,
                                              int padH, int padW,
                                              int nElements) {
  // Direct conv2d: each output element independently computes its value by
  // iterating over the filter spatial dimensions and input channels.
  //
  // Input shape: [N, IC, IH, IW]
  // Filter shape: [OC, IC, KH, KW]
  // Output shape: [N, OC, OH, OW]
  //
  // out[n,oc,oh,ow] = sum_{ic,kh,kw} input[n,ic,oh*sH-pH+kh,ow*sW-pW+kw] * filter[oc,ic,kh,kw]

  if (inputShape.size() < 4 || filterShape.size() < 4 || outputShape.size() < 4) {
    sd_debug("TritonIRBuilder::emitConvolutionSection: shapes must be 4D, got input=%d filter=%d output=%d\n",
              (int)inputShape.size(), (int)filterShape.size(), (int)outputShape.size());
    return;
  }

  int N  = static_cast<int>(inputShape[0]);
  int IC = static_cast<int>(inputShape[1]);
  int IH = static_cast<int>(inputShape[2]);
  int IW = static_cast<int>(inputShape[3]);
  int OC = static_cast<int>(filterShape[0]);
  int KH = static_cast<int>(filterShape[2]);
  int KW = static_cast<int>(filterShape[3]);
  int OH = static_cast<int>(outputShape[2]);
  int OW = static_cast<int>(outputShape[3]);

  auto i32Type = builder.getI32Type();
  auto f32Type = builder.getF32Type();
  auto i32TensorType = mlir::RankedTensorType::get({blockSize}, i32Type);
  auto f32TensorType = mlir::RankedTensorType::get({blockSize}, f32Type);

  // Derive pointer types from actual arguments (NOT hardcoded f32)
  auto inPtrType = mlir::cast<mlir::triton::PointerType>(inputPtr.getType());
  auto filtPtrType = mlir::cast<mlir::triton::PointerType>(filterPtr.getType());
  auto outPtrType = mlir::cast<mlir::triton::PointerType>(outputPtr.getType());
  auto inPtrTensorType = mlir::RankedTensorType::get({blockSize}, inPtrType);
  auto filtPtrTensorType = mlir::RankedTensorType::get({blockSize}, filtPtrType);
  auto outPtrTensorType = mlir::RankedTensorType::get({blockSize}, outPtrType);

  // Standard 1D offsets
  auto blockSizeConst = builder.create<mlir::arith::ConstantIntOp>(loc, blockSize, 32);
  auto offsetBase = builder.create<mlir::arith::MulIOp>(loc, pid, blockSizeConst);
  auto range = builder.create<mlir::triton::MakeRangeOp>(loc, i32TensorType, 0, blockSize);
  auto splatBase = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, offsetBase);
  auto offsets = builder.create<mlir::arith::AddIOp>(loc, splatBase, range);

  auto nElemConst = builder.create<mlir::arith::ConstantIntOp>(loc, nElements, 32);
  auto splatN = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, nElemConst);
  auto mask = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::slt, offsets, splatN);

  // Unravel linear index to (n, oc, oh, ow):
  //   ow = offsets % OW
  //   oh = (offsets / OW) % OH
  //   oc = (offsets / (OW * OH)) % OC
  //   n  = offsets / (OW * OH * OC)
  auto owConst = builder.create<mlir::arith::ConstantIntOp>(loc, OW, 32);
  auto ohConst = builder.create<mlir::arith::ConstantIntOp>(loc, OH, 32);
  auto ocConst = builder.create<mlir::arith::ConstantIntOp>(loc, OC, 32);
  auto owSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, owConst);
  auto ohSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, ohConst);
  auto ocSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, ocConst);

  auto ow_idx = builder.create<mlir::arith::RemSIOp>(loc, offsets, owSplat);
  auto tmp1 = builder.create<mlir::arith::DivSIOp>(loc, offsets, owSplat);
  auto oh_idx = builder.create<mlir::arith::RemSIOp>(loc, tmp1.getResult(), ohSplat);
  auto tmp2 = builder.create<mlir::arith::DivSIOp>(loc, tmp1.getResult(), ohSplat);
  auto oc_idx = builder.create<mlir::arith::RemSIOp>(loc, tmp2.getResult(), ocSplat);
  auto n_idx = builder.create<mlir::arith::DivSIOp>(loc, tmp2.getResult(), ocSplat);

  // Compute base positions in input space
  // oh_base = oh_idx * strideH - padH
  // ow_base = ow_idx * strideW - padW
  auto sHConst = builder.create<mlir::arith::ConstantIntOp>(loc, strideH, 32);
  auto sWConst = builder.create<mlir::arith::ConstantIntOp>(loc, strideW, 32);
  auto pHConst = builder.create<mlir::arith::ConstantIntOp>(loc, padH, 32);
  auto pWConst = builder.create<mlir::arith::ConstantIntOp>(loc, padW, 32);
  auto sHSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, sHConst);
  auto sWSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, sWConst);
  auto pHSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, pHConst);
  auto pWSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, pWConst);

  auto oh_base = builder.create<mlir::arith::SubIOp>(loc,
      builder.create<mlir::arith::MulIOp>(loc, oh_idx, sHSplat), pHSplat);
  auto ow_base = builder.create<mlir::arith::SubIOp>(loc,
      builder.create<mlir::arith::MulIOp>(loc, ow_idx, sWSplat), pWSplat);

  // Initialize accumulator to 0.0
  auto accInit = splatConstantF32(builder, loc, f32TensorType, 0.0f);

  // Triple nested loop: for ic in [0, IC): for kh in [0, KH): for kw in [0, KW):
  auto zero = builder.create<mlir::arith::ConstantIntOp>(loc, 0, 32);
  auto one = builder.create<mlir::arith::ConstantIntOp>(loc, 1, 32);
  auto icEnd = builder.create<mlir::arith::ConstantIntOp>(loc, IC, 32);
  auto khEnd = builder.create<mlir::arith::ConstantIntOp>(loc, KH, 32);
  auto kwEnd = builder.create<mlir::arith::ConstantIntOp>(loc, KW, 32);

  // Outer loop: ic
  auto icLoop = builder.create<mlir::scf::ForOp>(
      loc, zero, icEnd, one, mlir::ValueRange{accInit});
  builder.setInsertionPointToStart(icLoop.getBody());
  auto ic_val = icLoop.getInductionVar();
  auto acc_ic = icLoop.getBody()->getArgument(1);

  // Middle loop: kh
  auto khLoop = builder.create<mlir::scf::ForOp>(
      loc, zero, khEnd, one, mlir::ValueRange{acc_ic});
  builder.setInsertionPointToStart(khLoop.getBody());
  auto kh_val = khLoop.getInductionVar();
  auto acc_kh = khLoop.getBody()->getArgument(1);

  // Inner loop: kw
  auto kwLoop = builder.create<mlir::scf::ForOp>(
      loc, zero, kwEnd, one, mlir::ValueRange{acc_kh});
  builder.setInsertionPointToStart(kwLoop.getBody());
  auto kw_val = kwLoop.getInductionVar();
  auto acc_kw = kwLoop.getBody()->getArgument(1);

  // Compute input position: h_in = oh_base + kh, w_in = ow_base + kw
  auto khSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, kh_val);
  auto kwSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, kw_val);
  auto h_in = builder.create<mlir::arith::AddIOp>(loc, oh_base, khSplat);
  auto w_in = builder.create<mlir::arith::AddIOp>(loc, ow_base, kwSplat);

  // Bounds check: 0 <= h_in < IH && 0 <= w_in < IW
  auto ihConst = builder.create<mlir::arith::ConstantIntOp>(loc, IH, 32);
  auto iwConst = builder.create<mlir::arith::ConstantIntOp>(loc, IW, 32);
  auto ihSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, ihConst);
  auto iwSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, iwConst);
  auto zeroSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, zero);

  auto h_ge_0 = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::sge, h_in, zeroSplat);
  auto h_lt_IH = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt, h_in, ihSplat);
  auto w_ge_0 = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::sge, w_in, zeroSplat);
  auto w_lt_IW = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt, w_in, iwSplat);
  auto h_valid = builder.create<mlir::arith::AndIOp>(loc, h_ge_0, h_lt_IH);
  auto w_valid = builder.create<mlir::arith::AndIOp>(loc, w_ge_0, w_lt_IW);
  auto in_bounds = builder.create<mlir::arith::AndIOp>(loc, h_valid, w_valid);

  // Input offset: n * IC*IH*IW + ic * IH*IW + h_in * IW + w_in
  auto icIhIw = builder.create<mlir::arith::ConstantIntOp>(loc, IC * IH * IW, 32);
  auto ihIw = builder.create<mlir::arith::ConstantIntOp>(loc, IH * IW, 32);
  auto icIhIwSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, icIhIw);
  auto ihIwSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, ihIw);
  auto icSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, ic_val);

  auto inOffset = builder.create<mlir::arith::AddIOp>(loc,
      builder.create<mlir::arith::AddIOp>(loc,
          builder.create<mlir::arith::MulIOp>(loc, n_idx, icIhIwSplat),
          builder.create<mlir::arith::MulIOp>(loc, icSplat, ihIwSplat)),
      builder.create<mlir::arith::AddIOp>(loc,
          builder.create<mlir::arith::MulIOp>(loc, h_in, iwSplat),
          w_in));

  // Load input value (masked by bounds check AND element mask)
  auto combinedMask = builder.create<mlir::arith::AndIOp>(loc, in_bounds, mask);
  auto splatInPtr = builder.create<mlir::triton::SplatOp>(loc, inPtrTensorType, inputPtr);
  auto inPtrs = builder.create<mlir::triton::AddPtrOp>(loc, inPtrTensorType, splatInPtr, inOffset);
  auto inLoaded = builder.create<mlir::triton::LoadOp>(loc,
      inPtrs.getResult(), combinedMask.getResult(), mlir::Value(),
      mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL, false);
  auto inVal = castTo(builder, loc, inLoaded, f32Type);

  // Filter offset: oc * IC*KH*KW + ic * KH*KW + kh * KW + kw
  auto icKhKw = builder.create<mlir::arith::ConstantIntOp>(loc, IC * KH * KW, 32);
  auto khKw = builder.create<mlir::arith::ConstantIntOp>(loc, KH * KW, 32);
  auto icKhKwSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, icKhKw);
  auto khKwSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, khKw);

  // kh * KW needs KW as a tensor splat (kwEnd is a scalar constant for KW)
  auto kwConstSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, kwEnd);

  auto filterOffset = builder.create<mlir::arith::AddIOp>(loc,
      builder.create<mlir::arith::AddIOp>(loc,
          builder.create<mlir::arith::MulIOp>(loc, oc_idx, icKhKwSplat),
          builder.create<mlir::arith::MulIOp>(loc, icSplat, khKwSplat)),
      builder.create<mlir::arith::AddIOp>(loc,
          builder.create<mlir::arith::MulIOp>(loc, khSplat, kwConstSplat),
          kwSplat));

  // Load filter value (masked by element mask only, filter is always in bounds)
  auto splatFilterPtr = builder.create<mlir::triton::SplatOp>(loc, filtPtrTensorType, filterPtr);
  auto filterPtrs = builder.create<mlir::triton::AddPtrOp>(loc, filtPtrTensorType, splatFilterPtr, filterOffset);
  auto filterLoaded = builder.create<mlir::triton::LoadOp>(loc,
      filterPtrs.getResult(), mask.getResult(), mlir::Value(),
      mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL, false);
  auto filterVal = castTo(builder, loc, filterLoaded, f32Type);

  // Accumulate: acc += input * filter (zero out-of-bounds input)
  // inVal is already zero for out-of-bounds due to masked load default
  auto prod = builder.create<mlir::arith::MulFOp>(loc, inVal, filterVal);
  auto newAcc = builder.create<mlir::arith::AddFOp>(loc, acc_kw, prod);

  // Yield from inner loop
  builder.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{newAcc});

  // Yield from middle loop
  builder.setInsertionPointAfter(kwLoop);
  builder.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{kwLoop.getResult(0)});

  // Yield from outer loop
  builder.setInsertionPointAfter(khLoop);
  builder.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{khLoop.getResult(0)});

  // Store result
  builder.setInsertionPointAfter(icLoop);
  auto finalAcc = icLoop.getResult(0);

  mlir::Value outStoreVal = castTo(builder, loc, finalAcc, outPtrType.getPointeeType());
  auto splatOutPtr = builder.create<mlir::triton::SplatOp>(loc, outPtrTensorType, outputPtr);
  auto outPtrs = builder.create<mlir::triton::AddPtrOp>(loc, outPtrTensorType, splatOutPtr, offsets);
  builder.create<mlir::triton::StoreOp>(loc, outPtrs, outStoreVal, mask,
                                         mlir::triton::CacheModifier::NONE,
                                         mlir::triton::EvictionPolicy::NORMAL);

  sd_debug("TritonIRBuilder::emitConvolutionSection: conv2d N=%d IC=%d IH=%d IW=%d OC=%d KH=%d KW=%d "
            "OH=%d OW=%d sH=%d sW=%d pH=%d pW=%d\n",
            N, IC, IH, IW, OC, KH, KW, OH, OW, strideH, strideW, padH, padW);
}

// ─── im2col emission ─────────────────────────────────────────────────────────
// Rearranges image patches into columns for convolution.
// Input: [bS, iC, iH, iW] (4D)  →  Output: [bS, iC, kH, kW, oH, oW] (6D)
//
// For each output element at (b, c, kRow, kCol, colH, colW):
//   imRow = (-pH + kRow * dH) + colH * sH
//   imCol = (-pW + kCol * dW) + colW * sW
//   out = (in_bounds) ? input[b, c, imRow, imCol] : 0

void TritonIRBuilder::emitIm2colSection(mlir::OpBuilder& builder, mlir::Location loc,
                                         mlir::Value pid, int blockSize,
                                         mlir::Value inputPtr, mlir::Value outputPtr,
                                         const std::vector<LongType>& inputShape,
                                         const std::vector<LongType>& outputShape,
                                         int kH, int kW,
                                         int sH, int sW,
                                         int pH, int pW,
                                         int dH, int dW,
                                         int nElements) {
  // Input: [bS, iC, iH, iW], Output: [bS, iC, kH, kW, oH, oW]
  if (inputShape.size() < 4 || outputShape.size() < 6) {
    sd_debug("TritonIRBuilder::emitIm2colSection: input must be 4D (got %d) and output must be 6D (got %d)\n",
              (int)inputShape.size(), (int)outputShape.size());
    return;
  }

  int bS = static_cast<int>(inputShape[0]);
  int iC = static_cast<int>(inputShape[1]);
  int iH = static_cast<int>(inputShape[2]);
  int iW = static_cast<int>(inputShape[3]);
  int oH = static_cast<int>(outputShape[4]);
  int oW = static_cast<int>(outputShape[5]);

  auto i32Type = builder.getI32Type();
  auto i32TensorType = mlir::RankedTensorType::get({blockSize}, i32Type);
  // Derive pointer types from actual MLIR args (NOT hardcoded f32)
  auto inPtrType = mlir::cast<mlir::triton::PointerType>(inputPtr.getType());
  auto outPtrType = mlir::cast<mlir::triton::PointerType>(outputPtr.getType());
  auto inPtrTensorType = mlir::RankedTensorType::get({blockSize}, inPtrType);
  auto outPtrTensorType = mlir::RankedTensorType::get({blockSize}, outPtrType);

  // 1D offsets into output (6D linearized)
  auto blockSizeConst = builder.create<mlir::arith::ConstantIntOp>(loc, blockSize, 32);
  auto offsetBase = builder.create<mlir::arith::MulIOp>(loc, pid, blockSizeConst);
  auto range = builder.create<mlir::triton::MakeRangeOp>(loc, i32TensorType, 0, blockSize);
  auto splatBase = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, offsetBase);
  auto offsets = builder.create<mlir::arith::AddIOp>(loc, splatBase, range);

  auto nElemConst = builder.create<mlir::arith::ConstantIntOp>(loc, nElements, 32);
  auto splatN = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, nElemConst);
  auto mask = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::slt, offsets, splatN);

  // Unravel linear index to 6D: (b, c, kRow, kCol, colH, colW)
  // Layout: [bS, iC, kH, kW, oH, oW] in row-major
  // colW = offset % oW
  // colH = (offset / oW) % oH
  // kCol = (offset / (oW * oH)) % kW
  // kRow = (offset / (oW * oH * kW)) % kH
  // c    = (offset / (oW * oH * kW * kH)) % iC
  // b    = offset / (oW * oH * kW * kH * iC)
  auto oWConst = builder.create<mlir::arith::ConstantIntOp>(loc, oW, 32);
  auto oHConst = builder.create<mlir::arith::ConstantIntOp>(loc, oH, 32);
  auto kWConst = builder.create<mlir::arith::ConstantIntOp>(loc, kW, 32);
  auto kHConst = builder.create<mlir::arith::ConstantIntOp>(loc, kH, 32);
  auto iCConst = builder.create<mlir::arith::ConstantIntOp>(loc, iC, 32);
  auto oWSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, oWConst);
  auto oHSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, oHConst);
  auto kWSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, kWConst);
  auto kHSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, kHConst);
  auto iCSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, iCConst);

  auto colW_idx = builder.create<mlir::arith::RemSIOp>(loc, offsets, oWSplat);
  auto t1 = builder.create<mlir::arith::DivSIOp>(loc, offsets, oWSplat);
  auto colH_idx = builder.create<mlir::arith::RemSIOp>(loc, t1.getResult(), oHSplat);
  auto t2 = builder.create<mlir::arith::DivSIOp>(loc, t1.getResult(), oHSplat);
  auto kCol_idx = builder.create<mlir::arith::RemSIOp>(loc, t2.getResult(), kWSplat);
  auto t3 = builder.create<mlir::arith::DivSIOp>(loc, t2.getResult(), kWSplat);
  auto kRow_idx = builder.create<mlir::arith::RemSIOp>(loc, t3.getResult(), kHSplat);
  auto t4 = builder.create<mlir::arith::DivSIOp>(loc, t3.getResult(), kHSplat);
  auto c_idx = builder.create<mlir::arith::RemSIOp>(loc, t4.getResult(), iCSplat);
  auto b_idx = builder.create<mlir::arith::DivSIOp>(loc, t4.getResult(), iCSplat);

  // Compute input coordinates:
  // imRow = (-pH + kRow * dH) + colH * sH
  // imCol = (-pW + kCol * dW) + colW * sW
  auto dHConst = builder.create<mlir::arith::ConstantIntOp>(loc, dH, 32);
  auto dWConst = builder.create<mlir::arith::ConstantIntOp>(loc, dW, 32);
  auto sHConst = builder.create<mlir::arith::ConstantIntOp>(loc, sH, 32);
  auto sWConst = builder.create<mlir::arith::ConstantIntOp>(loc, sW, 32);
  auto pHConst = builder.create<mlir::arith::ConstantIntOp>(loc, pH, 32);
  auto pWConst = builder.create<mlir::arith::ConstantIntOp>(loc, pW, 32);
  auto dHSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, dHConst);
  auto dWSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, dWConst);
  auto sHSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, sHConst);
  auto sWSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, sWConst);
  auto pHSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, pHConst);
  auto pWSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, pWConst);

  // kRow * dH
  auto kRowDH = builder.create<mlir::arith::MulIOp>(loc, kRow_idx, dHSplat);
  // colH * sH
  auto colHSH = builder.create<mlir::arith::MulIOp>(loc, colH_idx, sHSplat);
  // imRow = kRow * dH + colH * sH - pH
  auto imRow = builder.create<mlir::arith::SubIOp>(loc,
      builder.create<mlir::arith::AddIOp>(loc, kRowDH, colHSH), pHSplat);

  // kCol * dW
  auto kColDW = builder.create<mlir::arith::MulIOp>(loc, kCol_idx, dWSplat);
  // colW * sW
  auto colWSW = builder.create<mlir::arith::MulIOp>(loc, colW_idx, sWSplat);
  // imCol = kCol * dW + colW * sW - pW
  auto imCol = builder.create<mlir::arith::SubIOp>(loc,
      builder.create<mlir::arith::AddIOp>(loc, kColDW, colWSW), pWSplat);

  // Bounds check: 0 <= imRow < iH && 0 <= imCol < iW
  auto zero = builder.create<mlir::arith::ConstantIntOp>(loc, 0, 32);
  auto iHConst = builder.create<mlir::arith::ConstantIntOp>(loc, iH, 32);
  auto iWConst = builder.create<mlir::arith::ConstantIntOp>(loc, iW, 32);
  auto zeroSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, zero);
  auto iHSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, iHConst);
  auto iWSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, iWConst);

  auto h_ge_0 = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::sge, imRow, zeroSplat);
  auto h_lt_iH = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt, imRow, iHSplat);
  auto w_ge_0 = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::sge, imCol, zeroSplat);
  auto w_lt_iW = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt, imCol, iWSplat);
  auto h_valid = builder.create<mlir::arith::AndIOp>(loc, h_ge_0, h_lt_iH);
  auto w_valid = builder.create<mlir::arith::AndIOp>(loc, w_ge_0, w_lt_iW);
  auto inBounds = builder.create<mlir::arith::AndIOp>(loc, h_valid, w_valid);

  // Combined mask: element in range AND in bounds
  auto combinedMask = builder.create<mlir::arith::AndIOp>(loc, inBounds, mask);

  // Input offset: b * (iC*iH*iW) + c * (iH*iW) + imRow * iW + imCol
  auto iCiHiW = builder.create<mlir::arith::ConstantIntOp>(loc, iC * iH * iW, 32);
  auto iHiW = builder.create<mlir::arith::ConstantIntOp>(loc, iH * iW, 32);
  auto iCiHiWSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, iCiHiW);
  auto iHiWSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, iHiW);

  auto inOffset = builder.create<mlir::arith::AddIOp>(loc,
      builder.create<mlir::arith::AddIOp>(loc,
          builder.create<mlir::arith::MulIOp>(loc, b_idx, iCiHiWSplat),
          builder.create<mlir::arith::MulIOp>(loc, c_idx, iHiWSplat)),
      builder.create<mlir::arith::AddIOp>(loc,
          builder.create<mlir::arith::MulIOp>(loc, imRow, iWSplat),
          imCol));

  // Load from input with bounds mask (out-of-bounds → 0)
  auto splatInPtr = builder.create<mlir::triton::SplatOp>(loc, inPtrTensorType, inputPtr);
  auto inPtrs = builder.create<mlir::triton::AddPtrOp>(loc, inPtrTensorType, splatInPtr, inOffset);
  auto inVal = builder.create<mlir::triton::LoadOp>(loc,
      inPtrs.getResult(), combinedMask.getResult(), mlir::Value(),
      mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL, false);

  // Store to output — cast if input and output element types differ
  mlir::Value storeVal = castTo(builder, loc, inVal, outPtrType.getPointeeType());
  auto splatOutPtr = builder.create<mlir::triton::SplatOp>(loc, outPtrTensorType, outputPtr);
  auto outPtrs = builder.create<mlir::triton::AddPtrOp>(loc, outPtrTensorType, splatOutPtr, offsets);
  builder.create<mlir::triton::StoreOp>(loc, outPtrs, storeVal, mask,
                                         mlir::triton::CacheModifier::NONE,
                                         mlir::triton::EvictionPolicy::NORMAL);

  sd_debug("TritonIRBuilder::emitIm2colSection: bS=%d iC=%d iH=%d iW=%d kH=%d kW=%d "
            "oH=%d oW=%d sH=%d sW=%d pH=%d pW=%d dH=%d dW=%d\n",
            bS, iC, iH, iW, kH, kW, oH, oW, sH, sW, pH, pW, dH, dW);
}

// ─── col2im emission ─────────────────────────────────────────────────────────
// Rearranges columns back to image (inverse of im2col).
// Input: [bS, iC, kH, kW, oH, oW] (6D)  →  Output: [bS, iC, iH, iW] (4D)
//
// For each output pixel at (b, c, h, w):
//   Iterate over kRow in [0, kH) and kCol in [0, kW):
//     colH = (h + pH - kRow * dH)
//     if colH >= 0 && colH % sH == 0: colH /= sH
//       colW = (w + pW - kCol * dW)
//       if colW >= 0 && colW % sW == 0: colW /= sW
//         if colH < oH && colW < oW: val += col[b, c, kRow, kCol, colH, colW]
//   out[b, c, h, w] = val

void TritonIRBuilder::emitCol2imSection(mlir::OpBuilder& builder, mlir::Location loc,
                                         mlir::Value pid, int blockSize,
                                         mlir::Value inputPtr, mlir::Value outputPtr,
                                         const std::vector<LongType>& inputShape,
                                         const std::vector<LongType>& outputShape,
                                         int kH, int kW,
                                         int sH, int sW,
                                         int pH, int pW,
                                         int dH, int dW,
                                         int nElements) {
  // Input (columns): [bS, iC, kH, kW, oH, oW], Output (image): [bS, iC, iH, iW]
  if (inputShape.size() < 6 || outputShape.size() < 4) {
    sd_debug("TritonIRBuilder::emitCol2imSection: input must be 6D (got %d) and output must be 4D (got %d)\n",
              (int)inputShape.size(), (int)outputShape.size());
    return;
  }

  int bS = static_cast<int>(outputShape[0]);
  int iC = static_cast<int>(outputShape[1]);
  int iH = static_cast<int>(outputShape[2]);
  int iW = static_cast<int>(outputShape[3]);
  int oH = static_cast<int>(inputShape[4]);
  int oW = static_cast<int>(inputShape[5]);

  auto i32Type = builder.getI32Type();
  auto f32Type = builder.getF32Type();
  auto i32TensorType = mlir::RankedTensorType::get({blockSize}, i32Type);
  auto f32TensorType = mlir::RankedTensorType::get({blockSize}, f32Type);
  // Derive pointer types from actual MLIR args (NOT hardcoded f32)
  auto inPtrType = mlir::cast<mlir::triton::PointerType>(inputPtr.getType());
  auto outPtrType = mlir::cast<mlir::triton::PointerType>(outputPtr.getType());
  auto inPtrTensorType = mlir::RankedTensorType::get({blockSize}, inPtrType);
  auto outPtrTensorType = mlir::RankedTensorType::get({blockSize}, outPtrType);

  // 1D offsets into output (4D linearized)
  auto blockSizeConst = builder.create<mlir::arith::ConstantIntOp>(loc, blockSize, 32);
  auto offsetBase = builder.create<mlir::arith::MulIOp>(loc, pid, blockSizeConst);
  auto range = builder.create<mlir::triton::MakeRangeOp>(loc, i32TensorType, 0, blockSize);
  auto splatBase = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, offsetBase);
  auto offsets = builder.create<mlir::arith::AddIOp>(loc, splatBase, range);

  auto nElemConst = builder.create<mlir::arith::ConstantIntOp>(loc, nElements, 32);
  auto splatN = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, nElemConst);
  auto mask = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::slt, offsets, splatN);

  // Unravel linear index to 4D: (b, c, h, w)
  // w = offset % iW
  // h = (offset / iW) % iH
  // c = (offset / (iW * iH)) % iC
  // b = offset / (iW * iH * iC)
  auto iWConst = builder.create<mlir::arith::ConstantIntOp>(loc, iW, 32);
  auto iHConst = builder.create<mlir::arith::ConstantIntOp>(loc, iH, 32);
  auto iCConst = builder.create<mlir::arith::ConstantIntOp>(loc, iC, 32);
  auto iWSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, iWConst);
  auto iHSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, iHConst);
  auto iCSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, iCConst);

  auto w_idx = builder.create<mlir::arith::RemSIOp>(loc, offsets, iWSplat);
  auto u1 = builder.create<mlir::arith::DivSIOp>(loc, offsets, iWSplat);
  auto h_idx = builder.create<mlir::arith::RemSIOp>(loc, u1.getResult(), iHSplat);
  auto u2 = builder.create<mlir::arith::DivSIOp>(loc, u1.getResult(), iHSplat);
  auto c_idx = builder.create<mlir::arith::RemSIOp>(loc, u2.getResult(), iCSplat);
  auto b_idx = builder.create<mlir::arith::DivSIOp>(loc, u2.getResult(), iCSplat);

  // Padded coordinates: imH = h + pH, imW = w + pW
  auto pHConst = builder.create<mlir::arith::ConstantIntOp>(loc, pH, 32);
  auto pWConst = builder.create<mlir::arith::ConstantIntOp>(loc, pW, 32);
  auto pHSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, pHConst);
  auto pWSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, pWConst);
  auto imH = builder.create<mlir::arith::AddIOp>(loc, h_idx, pHSplat);
  auto imW = builder.create<mlir::arith::AddIOp>(loc, w_idx, pWSplat);

  // Column buffer strides for 6D [bS, iC, kH, kW, oH, oW]
  // colStride5 = 1 (oW dim)
  // colStride4 = oW (oH dim)
  // colStride3 = oW * oH (kW dim)
  // colStride2 = oW * oH * kW (kH dim)
  // colStride1 = oW * oH * kW * kH (iC dim)
  // colStride0 = oW * oH * kW * kH * iC (bS dim)
  int colStride4 = oW;
  int colStride3 = oW * oH;
  int colStride2 = oW * oH * kW;
  int colStride1 = oW * oH * kW * kH;
  int colStride0 = oW * oH * kW * kH * iC;

  // Base offset into column buffer: b * colStride0 + c * colStride1
  auto colStr0Const = builder.create<mlir::arith::ConstantIntOp>(loc, colStride0, 32);
  auto colStr1Const = builder.create<mlir::arith::ConstantIntOp>(loc, colStride1, 32);
  auto colStr0Splat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, colStr0Const);
  auto colStr1Splat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, colStr1Const);
  auto bsOffset = builder.create<mlir::arith::MulIOp>(loc, b_idx, colStr0Splat);
  auto cOffset = builder.create<mlir::arith::MulIOp>(loc, c_idx, colStr1Splat);
  auto bcOffset = builder.create<mlir::arith::AddIOp>(loc, bsOffset, cOffset);

  // Initialize accumulator to 0.0
  auto accInit = splatConstantF32(builder, loc, f32TensorType, 0.0f);

  // Nested loops over kRow in [0, kH) and kCol in [0, kW)
  // These are compile-time-constant loop bounds, uniform across all elements
  auto zeroScalar = builder.create<mlir::arith::ConstantIntOp>(loc, 0, 32);
  auto oneScalar = builder.create<mlir::arith::ConstantIntOp>(loc, 1, 32);
  auto kHEnd = builder.create<mlir::arith::ConstantIntOp>(loc, kH, 32);
  auto kWEnd = builder.create<mlir::arith::ConstantIntOp>(loc, kW, 32);

  // Stride/dilation constants for vectorized computation
  auto dHConst = builder.create<mlir::arith::ConstantIntOp>(loc, dH, 32);
  auto dWConst = builder.create<mlir::arith::ConstantIntOp>(loc, dW, 32);
  auto sHConst = builder.create<mlir::arith::ConstantIntOp>(loc, sH, 32);
  auto sWConst = builder.create<mlir::arith::ConstantIntOp>(loc, sW, 32);
  auto dHSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, dHConst);
  auto dWSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, dWConst);
  auto sHSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, sHConst);
  auto sWSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, sWConst);
  auto oHConst = builder.create<mlir::arith::ConstantIntOp>(loc, oH, 32);
  auto oWConst = builder.create<mlir::arith::ConstantIntOp>(loc, oW, 32);
  auto oHSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, oHConst);
  auto oWSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, oWConst);
  auto zeroSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, zeroScalar);
  auto colStr2Const = builder.create<mlir::arith::ConstantIntOp>(loc, colStride2, 32);
  auto colStr3Const = builder.create<mlir::arith::ConstantIntOp>(loc, colStride3, 32);
  auto colStr4Const = builder.create<mlir::arith::ConstantIntOp>(loc, colStride4, 32);
  auto colStr2Splat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, colStr2Const);
  auto colStr3Splat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, colStr3Const);
  auto colStr4Splat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, colStr4Const);

  // Outer loop: kRow
  auto kRowLoop = builder.create<mlir::scf::ForOp>(
      loc, zeroScalar, kHEnd, oneScalar, mlir::ValueRange{accInit});
  builder.setInsertionPointToStart(kRowLoop.getBody());
  auto kRow_val = kRowLoop.getInductionVar();
  auto acc_kr = kRowLoop.getBody()->getArgument(1);

  // colH_raw = imH - kRow * dH  (per-element, signed)
  auto kRowSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, kRow_val);
  auto kRowDH = builder.create<mlir::arith::MulIOp>(loc, kRowSplat, dHSplat);
  auto colH_raw = builder.create<mlir::arith::SubIOp>(loc, imH, kRowDH);

  // Valid if colH_raw >= 0 && colH_raw % sH == 0
  auto colH_ge0 = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::sge, colH_raw, zeroSplat);
  auto colH_mod = builder.create<mlir::arith::RemSIOp>(loc, colH_raw, sHSplat);
  auto colH_aligned = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::eq, colH_mod, zeroSplat);
  auto colH_valid1 = builder.create<mlir::arith::AndIOp>(loc, colH_ge0, colH_aligned);

  // colH = colH_raw / sH (only meaningful where valid)
  auto colH = builder.create<mlir::arith::DivSIOp>(loc, colH_raw, sHSplat);
  // Additional check: colH < oH
  auto colH_lt_oH = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::slt, colH, oHSplat);
  auto colH_valid = builder.create<mlir::arith::AndIOp>(loc, colH_valid1, colH_lt_oH);

  // Inner loop: kCol
  auto kColLoop = builder.create<mlir::scf::ForOp>(
      loc, zeroScalar, kWEnd, oneScalar, mlir::ValueRange{acc_kr});
  builder.setInsertionPointToStart(kColLoop.getBody());
  auto kCol_val = kColLoop.getInductionVar();
  auto acc_kc = kColLoop.getBody()->getArgument(1);

  // colW_raw = imW - kCol * dW
  auto kColSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, kCol_val);
  auto kColDW = builder.create<mlir::arith::MulIOp>(loc, kColSplat, dWSplat);
  auto colW_raw = builder.create<mlir::arith::SubIOp>(loc, imW, kColDW);

  // Valid if colW_raw >= 0 && colW_raw % sW == 0
  auto colW_ge0 = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::sge, colW_raw, zeroSplat);
  auto colW_mod = builder.create<mlir::arith::RemSIOp>(loc, colW_raw, sWSplat);
  auto colW_aligned = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::eq, colW_mod, zeroSplat);
  auto colW_valid1 = builder.create<mlir::arith::AndIOp>(loc, colW_ge0, colW_aligned);

  // colW = colW_raw / sW
  auto colW = builder.create<mlir::arith::DivSIOp>(loc, colW_raw, sWSplat);
  // Additional check: colW < oW
  auto colW_lt_oW = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::slt, colW, oWSplat);
  auto colW_valid = builder.create<mlir::arith::AndIOp>(loc, colW_valid1, colW_lt_oW);

  // Combined validity: colH valid AND colW valid AND element mask
  auto hw_valid = builder.create<mlir::arith::AndIOp>(loc, colH_valid, colW_valid);
  auto loadMask = builder.create<mlir::arith::AndIOp>(loc, hw_valid, mask);

  // Column buffer offset: bcOffset + kRow * colStride2 + kCol * colStride3 + colH * colStride4 + colW
  auto colOffset = builder.create<mlir::arith::AddIOp>(loc, bcOffset,
      builder.create<mlir::arith::AddIOp>(loc,
          builder.create<mlir::arith::AddIOp>(loc,
              builder.create<mlir::arith::MulIOp>(loc, kRowSplat, colStr2Splat),
              builder.create<mlir::arith::MulIOp>(loc, kColSplat, colStr3Splat)),
          builder.create<mlir::arith::AddIOp>(loc,
              builder.create<mlir::arith::MulIOp>(loc, colH, colStr4Splat),
              colW)));

  // Load column value (masked: invalid positions get 0)
  auto splatColPtr = builder.create<mlir::triton::SplatOp>(loc, inPtrTensorType, inputPtr);
  auto colPtrs = builder.create<mlir::triton::AddPtrOp>(loc, inPtrTensorType, splatColPtr, colOffset);
  auto colVal = builder.create<mlir::triton::LoadOp>(loc,
      colPtrs.getResult(), loadMask.getResult(), mlir::Value(),
      mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL, false);

  // Cast loaded value to f32 for accumulation if needed
  auto colValF32 = castTo(builder, loc, colVal, f32Type);

  // Accumulate
  auto newAcc = builder.create<mlir::arith::AddFOp>(loc, acc_kc, colValF32);

  // Yield from inner loop (kCol)
  builder.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{newAcc});

  // Yield from outer loop (kRow)
  builder.setInsertionPointAfter(kColLoop);
  builder.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{kColLoop.getResult(0)});

  // Store accumulated result
  builder.setInsertionPointAfter(kRowLoop);
  auto finalAcc = kRowLoop.getResult(0);

  // Cast f32 accumulator to output element type for store
  mlir::Value storeVal = castTo(builder, loc, finalAcc, outPtrType.getPointeeType());
  auto splatOutPtr = builder.create<mlir::triton::SplatOp>(loc, outPtrTensorType, outputPtr);
  auto outPtrs = builder.create<mlir::triton::AddPtrOp>(loc, outPtrTensorType, splatOutPtr, offsets);
  builder.create<mlir::triton::StoreOp>(loc, outPtrs, storeVal, mask,
                                         mlir::triton::CacheModifier::NONE,
                                         mlir::triton::EvictionPolicy::NORMAL);

  sd_debug("TritonIRBuilder::emitCol2imSection: bS=%d iC=%d iH=%d iW=%d kH=%d kW=%d "
            "oH=%d oW=%d sH=%d sW=%d pH=%d pW=%d dH=%d dW=%d\n",
            bS, iC, iH, iW, kH, kW, oH, oW, sH, sW, pH, pW, dH, dW);
}

}  // namespace graph
}  // namespace sd

#endif  // HAVE_TRITON

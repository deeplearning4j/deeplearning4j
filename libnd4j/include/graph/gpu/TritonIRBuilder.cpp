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

  // Matrix ops (additional)
  table["tensormmul"]        = {"tensormmul",        TritonOpCategory::MATMUL, "tt.dot", false};
  table["TensorMmul"]        = {"TensorMmul",        TritonOpCategory::MATMUL, "tt.dot", false};
  table["batched_gemm"]      = {"batched_gemm",      TritonOpCategory::MATMUL, "tt.dot", false};
  table["BatchedGemm"]       = {"BatchedGemm",       TritonOpCategory::MATMUL, "tt.dot", false};
  table["xw_plus_b"]         = {"xw_plus_b",         TritonOpCategory::MATMUL, "tt.dot", true};
  table["XwPlusB"]           = {"XwPlusB",            TritonOpCategory::MATMUL, "tt.dot", true};

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
  return it != table.end() && it->second.category != TritonOpCategory::UNSUPPORTED;
}

TritonOpCategory TritonIRBuilder::getOpCategory(const std::string& opName) {
  const auto& table = getOpTable();
  auto it = table.find(opName);
  if (it != table.end()) return it->second.category;
  return TritonOpCategory::UNSUPPORTED;
}

bool TritonIRBuilder::isElementwiseCompatible(TritonOpCategory cat) {
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

SegmentKernelPattern TritonIRBuilder::classifySegment(NativeSlot* slots, int startSlot, int endSlot) {
  bool hasMatmul = false;
  bool hasReduction = false;
  bool hasNormalization = false;

  for (int i = startSlot; i <= endSlot; i++) {
    auto cat = getOpCategory(slots[i].opName);
    if (cat == TritonOpCategory::MATMUL) hasMatmul = true;
    if (cat == TritonOpCategory::REDUCTION) hasReduction = true;
    if (cat == TritonOpCategory::NORMALIZATION) hasNormalization = true;
  }

  if (hasMatmul) {
    // Check if there are element-wise ops after matmul (epilogue fusion)
    bool hasElementwiseAfterMatmul = false;
    bool seenMatmul = false;
    for (int i = startSlot; i <= endSlot; i++) {
      auto cat = getOpCategory(slots[i].opName);
      if (cat == TritonOpCategory::MATMUL) seenMatmul = true;
      else if (seenMatmul && isElementwiseCompatible(cat)) hasElementwiseAfterMatmul = true;
    }
    return hasElementwiseAfterMatmul ? SegmentKernelPattern::MATMUL_EPILOGUE
                                     : SegmentKernelPattern::MATMUL_2D;
  }
  if (hasNormalization) return SegmentKernelPattern::NORMALIZATION;
  if (hasReduction) return SegmentKernelPattern::REDUCTION_1D;
  return SegmentKernelPattern::ELEMENTWISE_1D;
}

// ─── Tile configuration ─────────────────────────────────────────────────────

void TritonIRBuilder::selectTileConfig(const std::vector<TritonOpCategory>& categories,
                                       const std::vector<std::vector<LongType>>& shapes,
                                       int& blockSize, int& numWarps, int& numStages) {
  bool hasMatmul = false;
  bool hasReduction = false;

  for (auto cat : categories) {
    if (cat == TritonOpCategory::MATMUL) hasMatmul = true;
    if (cat == TritonOpCategory::REDUCTION || cat == TritonOpCategory::NORMALIZATION) hasReduction = true;
  }

  if (hasMatmul) {
    blockSize = 128;
    numWarps = 8;
    numStages = 3;
  } else if (hasReduction) {
    blockSize = 1024;
    numWarps = 4;
    numStages = 2;
  } else {
    blockSize = 1024;
    numWarps = 4;
    numStages = 3;
  }
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
    case BOOL:     return builder.getI1Type();
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

  if (srcIsFloat && dstIsFloat) {
    // float → float: widen or narrow
    int srcBits = getFloatBitWidth(srcElemType);
    int dstBits = getFloatBitWidth(targetElemType);
    if (dstBits > srcBits) {
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
      return builder.create<mlir::arith::FPToSIOp>(loc, targetTensorType, val);
    }
  } else if (!srcIsFloat && dstIsFloat) {
    // integer/bool → float
    if (srcIsBool) {
      return builder.create<mlir::arith::UIToFPOp>(loc, targetTensorType, val);
    } else {
      return builder.create<mlir::arith::SIToFPOp>(loc, targetTensorType, val);
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
      if (dstBits > srcBits) {
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
    return builder.create<mlir::math::TanhOp>(loc, input);
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
    auto expX = builder.create<mlir::math::ExpOp>(loc, input);
    auto one = splatConstantF32(builder, loc, tensorType, 1.0f);
    auto onePlusExp = builder.create<mlir::arith::AddFOp>(loc, one, expX);
    auto sp = builder.create<mlir::math::LogOp>(loc, onePlusExp);
    auto tanhSp = builder.create<mlir::math::TanhOp>(loc, sp);
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

  // Program IDs for 2D grid
  auto pidM = builder.create<mlir::triton::GetProgramIdOp>(
      loc, i32Type, mlir::triton::ProgramIDDim::X);
  auto pidN = builder.create<mlir::triton::GetProgramIdOp>(
      loc, i32Type, mlir::triton::ProgramIDDim::Y);

  // Tile index offsets
  auto blockMConst = builder.create<mlir::arith::ConstantIntOp>(loc, blockM, 32);
  auto blockNConst = builder.create<mlir::arith::ConstantIntOp>(loc, blockN, 32);
  auto blockKConst = builder.create<mlir::arith::ConstantIntOp>(loc, blockK, 32);
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

  // Initialize accumulator to zeros: tensor<BM x BN x f32>
  auto accType = mlir::RankedTensorType::get({blockM, blockN}, f32Type);
  auto zeroAttr = builder.getFloatAttr(f32Type, 0.0);
  auto zeroScalar = builder.create<mlir::arith::ConstantOp>(loc, f32Type, zeroAttr);
  auto accInit = builder.create<mlir::triton::SplatOp>(loc, accType, zeroScalar);

  // K-loop bounds
  auto kStart = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
  auto kEnd = builder.create<mlir::arith::ConstantIndexOp>(loc, K);
  auto kStep = builder.create<mlir::arith::ConstantIndexOp>(loc, blockK);

  // K-loop via scf.for
  auto forOp = builder.create<mlir::scf::ForOp>(
      loc, kStart, kEnd, kStep, mlir::ValueRange{accInit});

  // Inside the K-loop body
  builder.setInsertionPointToStart(forOp.getBody());
  auto kIdx = forOp.getInductionVar();
  auto accIter = forOp.getBody()->getArgument(1);  // loop-carried accumulator

  // Convert k index to i32 for pointer arithmetic
  auto kIdxI32 = builder.create<mlir::arith::IndexCastOp>(loc, i32Type, kIdx);
  auto splatKOffset = builder.create<mlir::triton::SplatOp>(loc, i32BkType, kIdxI32);
  auto kIndices = builder.create<mlir::arith::AddIOp>(loc, splatKOffset, rangeK);

  // Load A tile [BM, BK]: A[mIndices, kIndices]
  // Compute pointers: a_ptr + mIndices * K + kIndices (strided access)
  auto kConst = builder.create<mlir::arith::ConstantIntOp>(loc, K, 32);
  auto aTileType = mlir::RankedTensorType::get({blockM, blockK}, f32Type);

  // Compute 2D pointer offsets for A: mIndices[:, None] * K + kIndices[None, :]
  auto mExpanded = builder.create<mlir::triton::ExpandDimsOp>(loc, mIndices, 1);  // [BM, 1]
  auto kExpanded = builder.create<mlir::triton::ExpandDimsOp>(loc, kIndices, 0);  // [1, BK]

  auto i32BmBkType = mlir::RankedTensorType::get({blockM, blockK}, i32Type);
  auto kSplat = builder.create<mlir::triton::SplatOp>(loc, mlir::RankedTensorType::get({blockM, 1}, i32Type), kConst);
  auto mTimesK = builder.create<mlir::arith::MulIOp>(loc, mExpanded, kSplat);
  auto mTimesKBroadcast = builder.create<mlir::triton::BroadcastOp>(loc, i32BmBkType, mTimesK);
  auto kBroadcast = builder.create<mlir::triton::BroadcastOp>(loc, i32BmBkType, kExpanded);
  auto aOffsets = builder.create<mlir::arith::AddIOp>(loc, mTimesKBroadcast, kBroadcast);

  auto aPtrType = mlir::triton::PointerType::get(f32Type, 1);
  auto aPtrTensorType = mlir::RankedTensorType::get({blockM, blockK}, aPtrType);
  auto aSplat = builder.create<mlir::triton::SplatOp>(loc, aPtrTensorType, aPtr);
  auto aPtrs = builder.create<mlir::triton::AddPtrOp>(loc, aPtrTensorType, aSplat, aOffsets);

  // Create 2D mask for A tile: mIndices < M && kIndices < K
  auto i1Type = builder.getI1Type();
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

  // Load B tile [BK, BN]: B[kIndices, nIndices]
  auto nConst = builder.create<mlir::arith::ConstantIntOp>(loc, N, 32);
  auto bTileType = mlir::RankedTensorType::get({blockK, blockN}, f32Type);

  auto kExpandedB = builder.create<mlir::triton::ExpandDimsOp>(loc, kIndices, 1);  // [BK, 1]
  auto nExpandedB = builder.create<mlir::triton::ExpandDimsOp>(loc, nIndices, 0);  // [1, BN]

  auto i32BkBnType = mlir::RankedTensorType::get({blockK, blockN}, i32Type);
  auto nSplat = builder.create<mlir::triton::SplatOp>(loc, mlir::RankedTensorType::get({blockK, 1}, i32Type), nConst);
  auto kTimesN = builder.create<mlir::arith::MulIOp>(loc, kExpandedB, nSplat);
  auto kTimesNBroadcast = builder.create<mlir::triton::BroadcastOp>(loc, i32BkBnType, kTimesN);
  auto nBroadcastB = builder.create<mlir::triton::BroadcastOp>(loc, i32BkBnType, nExpandedB);
  auto bOffsets = builder.create<mlir::arith::AddIOp>(loc, kTimesNBroadcast, nBroadcastB);

  auto bPtrTensorType = mlir::RankedTensorType::get({blockK, blockN}, aPtrType);
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
  auto dotResult = builder.create<mlir::triton::DotOp>(loc, accType, aLoaded, bLoaded, accIter);

  // Yield accumulator for next K-iteration
  builder.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{dotResult});

  // After the K-loop — store result C tile
  builder.setInsertionPointAfter(forOp);
  auto finalAcc = forOp.getResult(0);

  // Compute C pointers: c_ptr + mIndices * N + nIndices
  auto mExpandedC = builder.create<mlir::triton::ExpandDimsOp>(loc, mIndices, 1);  // [BM, 1]
  auto nExpandedC = builder.create<mlir::triton::ExpandDimsOp>(loc, nIndices, 0);  // [1, BN]

  auto i32BmBnType = mlir::RankedTensorType::get({blockM, blockN}, i32Type);
  auto nSplatC = builder.create<mlir::triton::SplatOp>(loc, mlir::RankedTensorType::get({blockM, 1}, i32Type), nConst);
  auto mTimesNC = builder.create<mlir::arith::MulIOp>(loc, mExpandedC, nSplatC);
  auto mTimesNCBroadcast = builder.create<mlir::triton::BroadcastOp>(loc, i32BmBnType, mTimesNC);
  auto nBroadcastC = builder.create<mlir::triton::BroadcastOp>(loc, i32BmBnType, nExpandedC);
  auto cOffsets = builder.create<mlir::arith::AddIOp>(loc, mTimesNCBroadcast, nBroadcastC);

  auto cPtrTensorType = mlir::RankedTensorType::get({blockM, blockN}, aPtrType);
  auto cSplat = builder.create<mlir::triton::SplatOp>(loc, cPtrTensorType, cPtr);
  auto cPtrs = builder.create<mlir::triton::AddPtrOp>(loc, cPtrTensorType, cSplat, cOffsets);

  // Create 2D mask for C tile: mIndices < M && nIndices < N
  auto i1TypeC = builder.getI1Type();
  auto mConstC = builder.create<mlir::arith::ConstantIntOp>(loc, M, 32);
  auto nConstC = builder.create<mlir::arith::ConstantIntOp>(loc, N, 32);
  auto mConstSplatC = builder.create<mlir::triton::SplatOp>(loc, i32BmType, mConstC);
  auto nConstSplatC = builder.create<mlir::triton::SplatOp>(loc, i32BnType, nConstC);
  auto mMask1DC = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt, mIndices, mConstSplatC);
  auto nMask1DC = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt, nIndices, nConstSplatC);
  auto i1BmBnType = mlir::RankedTensorType::get({blockM, blockN}, i1TypeC);
  auto mMaskExpC = builder.create<mlir::triton::ExpandDimsOp>(loc, mMask1DC, 1);
  auto nMaskExpC = builder.create<mlir::triton::ExpandDimsOp>(loc, nMask1DC, 0);
  auto mMask2DC = builder.create<mlir::triton::BroadcastOp>(loc, i1BmBnType, mMaskExpC);
  auto nMask2DC = builder.create<mlir::triton::BroadcastOp>(loc, i1BmBnType, nMaskExpC);
  auto cMask = builder.create<mlir::arith::AndIOp>(loc, mMask2DC, nMask2DC);

  builder.create<mlir::triton::StoreOp>(loc, cPtrs, finalAcc, cMask,
                                         mlir::triton::CacheModifier::NONE,
                                         mlir::triton::EvictionPolicy::NORMAL);

  sd_printf("TritonIRBuilder: emitted matmul kernel M=%d N=%d K=%d BM=%d BN=%d BK=%d\n",
            M, N, K, blockM, blockN, blockK);
}

// ─── Module construction ────────────────────────────────────────────────────

TritonIRModule TritonIRBuilder::buildModule(NativeSlot* slots, int startSlot, int endSlot,
                                            NDArray** externalInputs, int numExternalInputs,
                                            NDArray** outputSlots, int totalOutputSlots) {
  TritonIRModule result;
  result.kernelName = generateKernelName(slots, startSlot, endSlot);

  // Collect op categories and shapes for tile config
  std::vector<TritonOpCategory> categories;
  std::vector<std::vector<LongType>> shapes;

  for (int i = startSlot; i <= endSlot; i++) {
    auto cat = getOpCategory(slots[i].opName);
    if (cat == TritonOpCategory::UNSUPPORTED) {
      sd_printf("TritonIRBuilder::buildModule: unsupported op '%s' at slot %d\n",
                slots[i].opName.c_str(), i);
      return result;
    }
    categories.push_back(cat);

    if (slots[i].numOutputs > 0) {
      int outIdx = slots[i].outputSlotIndices[0];
      if (outIdx >= 0 && outIdx < totalOutputSlots && outputSlots[outIdx]) {
        auto& arr = *outputSlots[outIdx];
        std::vector<LongType> shape(arr.rankOf());
        for (int d = 0; d < arr.rankOf(); d++) shape[d] = arr.sizeAt(d);
        shapes.push_back(shape);
      } else {
        shapes.push_back({});
      }
    } else {
      shapes.push_back({});
    }
  }

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

  // Outputs: slot outputs that are consumed AFTER the segment or are final outputs
  std::vector<TritonKernelArg> outputArgs;
  for (int i = startSlot; i <= endSlot; i++) {
    for (int o = 0; o < slots[i].numOutputs; o++) {
      int outIdx = slots[i].outputSlotIndices[o];
      if (outIdx >= 0 && outIdx < totalOutputSlots) {
        TritonKernelArg arg;
        arg.slotIndex = outIdx;
        arg.outputIndex = o;
        arg.isOutput = true;
        if (outputSlots[outIdx]) {
          arg.dtype = outputSlots[outIdx]->dataType();
          auto& arr = *outputSlots[outIdx];
          for (int d = 0; d < arr.rankOf(); d++) arg.shape.push_back(arr.sizeAt(d));
        }
        outputArgs.push_back(arg);
      }
    }
  }

  // Combine: inputs first, then outputs
  result.args.insert(result.args.end(), inputArgs.begin(), inputArgs.end());
  result.args.insert(result.args.end(), outputArgs.begin(), outputArgs.end());

  // ── Build function signature ──
  // Each arg is a tt.ptr<dtype>, plus n_elements : i32
  std::vector<mlir::Type> funcArgTypes;
  for (auto& arg : result.args) {
    auto elemType = getMLIRType(builder, arg.dtype);
    funcArgTypes.push_back(mlir::triton::PointerType::get(elemType, 1));
  }
  funcArgTypes.push_back(builder.getI32Type());  // n_elements

  auto funcType = builder.getFunctionType(funcArgTypes, {});
  auto funcOp = builder.create<mlir::triton::FuncOp>(loc, result.kernelName, funcType);
  funcOp.setPublic();

  auto* entryBlock = funcOp.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

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
  for (int a = 0; a < static_cast<int>(inputArgs.size()); a++) {
    auto& arg = inputArgs[a];
    auto funcArg = entryBlock->getArgument(a);  // tt.ptr<f32>

    auto elemType = getMLIRType(builder, arg.dtype);
    auto ptrType = mlir::triton::PointerType::get(elemType, 1);
    auto ptrTensorType = mlir::RankedTensorType::get({blockSize}, ptrType);
    auto dataTensorType = mlir::RankedTensorType::get({blockSize}, elemType);

    auto splatPtr = builder.create<mlir::triton::SplatOp>(loc, ptrTensorType, funcArg);
    auto ptrs = builder.create<mlir::triton::AddPtrOp>(loc, ptrTensorType, splatPtr, offsets);
    mlir::Value ptrVal = ptrs.getResult();
    mlir::Value maskVal = mask;
    auto loaded = builder.create<mlir::triton::LoadOp>(loc,
                                                        /*ptr=*/ptrVal,
                                                        /*mask=*/maskVal,
                                                        /*other=*/mlir::Value(),
                                                        /*cache=*/mlir::triton::CacheModifier::NONE,
                                                        /*evict=*/mlir::triton::EvictionPolicy::NORMAL,
                                                        /*isVolatile=*/false);
    ssaValues[arg.slotIndex] = loaded;
  }

  // 2c: Fused op emission — iterate over slots, resolve inputs from ssaValues
  const auto& opTable = getOpTable();
  int catIdx = 0;
  for (int si = startSlot; si <= endSlot; si++, catIdx++) {
    auto& slot = slots[si];
    auto cat = categories[catIdx];
    auto it = opTable.find(slot.opName);
    if (it == opTable.end()) continue;
    const auto& mapping = it->second;

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
      int inputSrc = slot.inputSourceIndices[0];
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
        if (outIdx >= 0 && outIdx < totalOutputSlots && outputSlots[outIdx]) {
          targetDtype = outputSlots[outIdx]->dataType();
        }
      }
      auto targetElemType = getMLIRType(builder, targetDtype);
      auto opResult = castTo(builder, loc, inputIt->second, targetElemType);
      for (int o = 0; o < slot.numOutputs; o++) {
        ssaValues[slot.outputSlotIndices[o]] = opResult;
      }

    } else {
      // MATMUL, REDUCTION, NORMALIZATION — not yet supported in element-wise kernel
      // canFuseSegment() should have filtered these out already
    }
  }

  // 2d: Store outputs — tt.store for each output arg
  int outputArgBase = static_cast<int>(inputArgs.size());
  for (int a = 0; a < static_cast<int>(outputArgs.size()); a++) {
    auto& arg = outputArgs[a];
    auto funcArg = entryBlock->getArgument(outputArgBase + a);

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

    // Cast SSA value to match output element type if needed (handles all type combos)
    mlir::Value storeVal = castTo(builder, loc, ssaIt->second, elemType);

    builder.create<mlir::triton::StoreOp>(loc, ptrs, storeVal, mask,
                                           mlir::triton::CacheModifier::NONE,
                                           mlir::triton::EvictionPolicy::NORMAL);
  }

  // Return
  builder.create<mlir::triton::ReturnOp>(loc);

  result.mlirModule = new mlir::ModuleOp(moduleOp);
  result.valid = true;

  sd_printf("TritonIRBuilder: built module '%s' with %d ops, %d input args, %d output args, "
            "BLOCK_SIZE=%d\n",
            result.kernelName.c_str(), (endSlot - startSlot + 1),
            static_cast<int>(inputArgs.size()), static_cast<int>(outputArgs.size()),
            blockSize);

  return result;
}

}  // namespace graph
}  // namespace sd

#endif  // HAVE_TRITON

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

//
// TritonIRBuilder — Core:
//   Constructor/destructor, setSectionedBlockSizeOverride, clearSectionedBlockSizeOverride,
//   isTritonMappable, getOpCategory, isElementwiseCompatible,
//   generateKernelName, dumpSectionBreakdown, dumpArgMapping.
//   Op mapping table (buildOpTable, getOpTable).
//
// Split files:
//   TritonIRBuilder_analysis.cpp  — profileSegment, matchPatterns, classifyAndAnalyze,
//                                   analyzeSegment, classifySegment, selectTileConfig
//   TritonIRBuilder_types.cpp     — getMLIRType, splatConstantF32, splatConstantI32
//   TritonIRBuilder_emitters.cpp  — emitBinaryElementwise, emitUnaryElementwise,
//                                   emitComparisonOp, emitLogicalOp, emitTernaryOp,
//                                   emitReductionOp, emitNormalizationOp
//   TritonIRBuilder_kernels.cpp   — emitMatmulKernel, emitFusedAttentionKernel
//   TritonIRBuilder_module.cpp    — buildModule, buildSectionedModule, buildMatmulModule
//   TritonIRBuilder_sections.cpp  — identifySections, computeSectionGrid, emitGridSync,
//                                   emitThreadfenceBarrier, emit*Section, dump*
//   TritonIRBuilder_internal.h    — Shared inline helpers (type helpers, utility, CUDA)
//

#include <config.h>

#if HAVE_TRITON

#include <graph/gpu/TritonIRBuilder.h>
#include <graph/DspDiagnostics.h>
#include <graph/DspHashUtils.h>
#include <graph/gpu/TritonIRBuilder_internal.h>
#include <graph/gpu/OpCategoryTable.h>
#include <helpers/logger.h>
#include <ops/declarable/OpDescriptor.h>
#include <ops/declarable/OpRegistrator.h>
#include <system/Environment.h>
#include <system/common.h>

#include <cstdint>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>


namespace sd {
namespace graph {

using namespace ir_builder_internal;

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
  table["maximum"]   = {"maximum",   TritonOpCategory::BINARY_ELEMENTWISE, "arith.maximumf", false};
  table["Max"]       = {"Max",       TritonOpCategory::BINARY_ELEMENTWISE, "arith.maximumf", false};
  table["minimum"]   = {"minimum",   TritonOpCategory::BINARY_ELEMENTWISE, "arith.minimumf", false};
  table["Min"]       = {"Min",       TritonOpCategory::BINARY_ELEMENTWISE, "arith.minimumf", false};
  table["floormod"]  = {"floormod",  TritonOpCategory::BINARY_ELEMENTWISE, "arith.remf",     false};
  table["FloorMod"]  = {"FloorMod",  TritonOpCategory::BINARY_ELEMENTWISE, "arith.remf",     false};
  table["mod"]       = {"mod",       TritonOpCategory::BINARY_ELEMENTWISE, "arith.remf",     false};
  table["Mod"]       = {"Mod",       TritonOpCategory::BINARY_ELEMENTWISE, "arith.remf",     false};
  table["floordiv"]  = {"floordiv",  TritonOpCategory::BINARY_ELEMENTWISE, "custom.floordiv", true};
  table["FloorDiv"]  = {"FloorDiv",  TritonOpCategory::BINARY_ELEMENTWISE, "custom.floordiv", true};
  table["atan2"]     = {"atan2",     TritonOpCategory::BINARY_ELEMENTWISE, "math.atan2",     false};
  table["Atan2"]     = {"Atan2",     TritonOpCategory::BINARY_ELEMENTWISE, "math.atan2",     false};

  // Custom binary ops
  table["reversedivide"]    = {"reversedivide",    TritonOpCategory::BINARY_ELEMENTWISE, "custom.reversediv",   true};
  table["ReverseDiv"]       = {"ReverseDiv",       TritonOpCategory::BINARY_ELEMENTWISE, "custom.reversediv",   true};
  table["reversesubtract"]  = {"reversesubtract",  TritonOpCategory::BINARY_ELEMENTWISE, "custom.reversesub",   true};
  table["ReverseSub"]       = {"ReverseSub",       TritonOpCategory::BINARY_ELEMENTWISE, "custom.reversesub",   true};
  table["squaredsubtract"]  = {"squaredsubtract",  TritonOpCategory::BINARY_ELEMENTWISE, "custom.squaredsub",   true};
  table["SquaredSub"]       = {"SquaredSub",       TritonOpCategory::BINARY_ELEMENTWISE, "custom.squaredsub",   true};
  table["multiply_no_nan"]  = {"multiply_no_nan",  TritonOpCategory::BINARY_ELEMENTWISE, "custom.mul_no_nan",   true};
  table["MulNoNan"]         = {"MulNoNan",         TritonOpCategory::BINARY_ELEMENTWISE, "custom.mul_no_nan",   true};
  table["Pow"]              = {"Pow",              TritonOpCategory::BINARY_ELEMENTWISE, "custom.pow",          true};
  table["swish_mul"]        = {"swish_mul",        TritonOpCategory::BINARY_ELEMENTWISE, "custom.swish_mul",    true};

  // Unary element-wise
  table["relu"]       = {"relu",       TritonOpCategory::UNARY_ELEMENTWISE, "custom.relu",     true};
  table["Relu"]       = {"Relu",       TritonOpCategory::UNARY_ELEMENTWISE, "custom.relu",     true};
  table["sigmoid"]    = {"sigmoid",    TritonOpCategory::UNARY_ELEMENTWISE, "custom.sigmoid",  true};
  table["Sigmoid"]    = {"Sigmoid",    TritonOpCategory::UNARY_ELEMENTWISE, "custom.sigmoid",  true};
  table["tanh"]       = {"tanh",       TritonOpCategory::UNARY_ELEMENTWISE, "custom.tanh",     true};
  table["Tanh"]       = {"Tanh",       TritonOpCategory::UNARY_ELEMENTWISE, "custom.tanh",     true};
  table["gelu"]       = {"gelu",       TritonOpCategory::UNARY_ELEMENTWISE, "custom.gelu",     true};
  table["Gelu"]       = {"Gelu",       TritonOpCategory::UNARY_ELEMENTWISE, "custom.gelu",     true};
  table["exp"]        = {"exp",        TritonOpCategory::UNARY_ELEMENTWISE, "math.exp",        false};
  table["Exp"]        = {"Exp",        TritonOpCategory::UNARY_ELEMENTWISE, "math.exp",        false};
  table["log"]        = {"log",        TritonOpCategory::UNARY_ELEMENTWISE, "math.log",        false};
  table["Log"]        = {"Log",        TritonOpCategory::UNARY_ELEMENTWISE, "math.log",        false};
  table["abs"]        = {"abs",        TritonOpCategory::UNARY_ELEMENTWISE, "math.absf",       false};
  table["Abs"]        = {"Abs",        TritonOpCategory::UNARY_ELEMENTWISE, "math.absf",       false};
  table["sqrt"]       = {"sqrt",       TritonOpCategory::UNARY_ELEMENTWISE, "math.sqrt",       false};
  table["Sqrt"]       = {"Sqrt",       TritonOpCategory::UNARY_ELEMENTWISE, "math.sqrt",       false};
  table["square"]     = {"square",     TritonOpCategory::UNARY_ELEMENTWISE, "custom.square",   true};
  table["Square"]     = {"Square",     TritonOpCategory::UNARY_ELEMENTWISE, "custom.square",   true};
  table["pow"]        = {"pow",        TritonOpCategory::UNARY_ELEMENTWISE, "custom.pow",      true};
  table["neg"]        = {"neg",        TritonOpCategory::UNARY_ELEMENTWISE, "arith.negf",      false};
  table["Neg"]        = {"Neg",        TritonOpCategory::UNARY_ELEMENTWISE, "arith.negf",      false};
  table["reciprocal"] = {"reciprocal", TritonOpCategory::UNARY_ELEMENTWISE, "custom.reciprocal", true};
  table["Reciprocal"] = {"Reciprocal", TritonOpCategory::UNARY_ELEMENTWISE, "custom.reciprocal", true};
  table["rsqrt"]      = {"rsqrt",      TritonOpCategory::UNARY_ELEMENTWISE, "custom.rsqrt",    true};
  table["Rsqrt"]      = {"Rsqrt",      TritonOpCategory::UNARY_ELEMENTWISE, "custom.rsqrt",    true};
  table["sign"]       = {"sign",       TritonOpCategory::UNARY_ELEMENTWISE, "custom.sign",     true};
  table["Sign"]       = {"Sign",       TritonOpCategory::UNARY_ELEMENTWISE, "custom.sign",     true};
  table["erf"]        = {"erf",        TritonOpCategory::UNARY_ELEMENTWISE, "math.erf",        false};
  table["Erf"]        = {"Erf",        TritonOpCategory::UNARY_ELEMENTWISE, "math.erf",        false};
  table["erfc"]       = {"erfc",       TritonOpCategory::UNARY_ELEMENTWISE, "custom.erfc",     true};
  table["Erfc"]       = {"Erfc",       TritonOpCategory::UNARY_ELEMENTWISE, "custom.erfc",     true};
  table["clamp"]      = {"clamp",      TritonOpCategory::UNARY_ELEMENTWISE, "custom.clamp",    true};
  table["ClipByValue"]= {"ClipByValue",TritonOpCategory::UNARY_ELEMENTWISE, "custom.clamp",    true};
  table["clip_by_value"]={"clip_by_value",TritonOpCategory::UNARY_ELEMENTWISE, "custom.clamp",  true};
  table["log1p"]      = {"log1p",      TritonOpCategory::UNARY_ELEMENTWISE, "math.log1p",      false};
  table["Log1p"]      = {"Log1p",      TritonOpCategory::UNARY_ELEMENTWISE, "math.log1p",      false};
  table["ceil"]       = {"ceil",       TritonOpCategory::UNARY_ELEMENTWISE, "math.ceil",       false};
  table["Ceil"]       = {"Ceil",       TritonOpCategory::UNARY_ELEMENTWISE, "math.ceil",       false};
  table["floor"]      = {"floor",      TritonOpCategory::UNARY_ELEMENTWISE, "math.floor",      false};
  table["Floor"]      = {"Floor",      TritonOpCategory::UNARY_ELEMENTWISE, "math.floor",      false};
  table["round"]      = {"round",      TritonOpCategory::UNARY_ELEMENTWISE, "math.roundeven",  false};
  table["Round"]      = {"Round",      TritonOpCategory::UNARY_ELEMENTWISE, "math.roundeven",  false};
  table["sin"]        = {"sin",        TritonOpCategory::UNARY_ELEMENTWISE, "math.sin",        false};
  table["Sin"]        = {"Sin",        TritonOpCategory::UNARY_ELEMENTWISE, "math.sin",        false};
  table["cos"]        = {"cos",        TritonOpCategory::UNARY_ELEMENTWISE, "math.cos",        false};
  table["Cos"]        = {"Cos",        TritonOpCategory::UNARY_ELEMENTWISE, "math.cos",        false};

  // Activations
  table["leakyrelu"]   = {"leakyrelu",   TritonOpCategory::UNARY_ELEMENTWISE, "custom.leakyrelu",   true};
  table["LeakyRelu"]   = {"LeakyRelu",   TritonOpCategory::UNARY_ELEMENTWISE, "custom.leakyrelu",   true};
  table["silu"]        = {"silu",        TritonOpCategory::UNARY_ELEMENTWISE, "custom.silu",        true};
  table["swish"]       = {"swish",       TritonOpCategory::UNARY_ELEMENTWISE, "custom.silu",        true};
  table["Swish"]       = {"Swish",       TritonOpCategory::UNARY_ELEMENTWISE, "custom.silu",        true};
  table["mish"]        = {"mish",        TritonOpCategory::UNARY_ELEMENTWISE, "custom.mish",        true};
  table["Mish"]        = {"Mish",        TritonOpCategory::UNARY_ELEMENTWISE, "custom.mish",        true};
  table["elu"]         = {"elu",         TritonOpCategory::UNARY_ELEMENTWISE, "custom.elu",         true};
  table["Elu"]         = {"Elu",         TritonOpCategory::UNARY_ELEMENTWISE, "custom.elu",         true};
  table["selu"]        = {"selu",        TritonOpCategory::UNARY_ELEMENTWISE, "custom.selu",        true};
  table["Selu"]        = {"Selu",        TritonOpCategory::UNARY_ELEMENTWISE, "custom.selu",        true};
  table["softplus"]    = {"softplus",    TritonOpCategory::UNARY_ELEMENTWISE, "custom.softplus",    true};
  table["SoftPlus"]    = {"SoftPlus",    TritonOpCategory::UNARY_ELEMENTWISE, "custom.softplus",    true};
  table["softsign"]    = {"softsign",    TritonOpCategory::UNARY_ELEMENTWISE, "custom.softsign",    true};
  table["SoftSign"]    = {"SoftSign",    TritonOpCategory::UNARY_ELEMENTWISE, "custom.softsign",    true};
  table["hard_sigmoid"]= {"hard_sigmoid",TritonOpCategory::UNARY_ELEMENTWISE, "custom.hard_sigmoid",true};
  table["HardSigmoid"] = {"HardSigmoid",TritonOpCategory::UNARY_ELEMENTWISE, "custom.hard_sigmoid",true};
  table["hardtanh"]    = {"hardtanh",    TritonOpCategory::UNARY_ELEMENTWISE, "custom.hardtanh",    true};
  table["HardTanh"]    = {"HardTanh",    TritonOpCategory::UNARY_ELEMENTWISE, "custom.hardtanh",    true};
  table["relu6"]       = {"relu6",       TritonOpCategory::UNARY_ELEMENTWISE, "custom.relu6",       true};
  table["Relu6"]       = {"Relu6",       TritonOpCategory::UNARY_ELEMENTWISE, "custom.relu6",       true};

  // Activation backward ops — binary: input[0]=x, input[1]=dy, output=dx
  table["relu_bp"]          = {"relu_bp",          TritonOpCategory::BINARY_ELEMENTWISE, "custom.relu_bp",          true};
  table["relu6_bp"]         = {"relu6_bp",         TritonOpCategory::BINARY_ELEMENTWISE, "custom.relu6_bp",         true};
  table["thresholdedrelu_bp"] = {"thresholdedrelu_bp", TritonOpCategory::BINARY_ELEMENTWISE, "custom.thresholdedrelu_bp", true};
  table["sigmoid_bp"]       = {"sigmoid_bp",       TritonOpCategory::BINARY_ELEMENTWISE, "custom.sigmoid_bp",       true};
  table["tanh_bp"]          = {"tanh_bp",          TritonOpCategory::BINARY_ELEMENTWISE, "custom.tanh_bp",          true};
  table["elu_bp"]           = {"elu_bp",           TritonOpCategory::BINARY_ELEMENTWISE, "custom.elu_bp",           true};
  table["selu_bp"]          = {"selu_bp",          TritonOpCategory::BINARY_ELEMENTWISE, "custom.selu_bp",          true};
  table["lrelu_bp"]         = {"lrelu_bp",         TritonOpCategory::BINARY_ELEMENTWISE, "custom.lrelu_bp",         true};
  table["softplus_bp"]      = {"softplus_bp",      TritonOpCategory::BINARY_ELEMENTWISE, "custom.softplus_bp",      true};
  table["softsign_bp"]      = {"softsign_bp",      TritonOpCategory::BINARY_ELEMENTWISE, "custom.softsign_bp",      true};
  table["hardsigmoid_bp"]   = {"hardsigmoid_bp",   TritonOpCategory::BINARY_ELEMENTWISE, "custom.hardsigmoid_bp",   true};
  table["hardtanh_bp"]      = {"hardtanh_bp",      TritonOpCategory::BINARY_ELEMENTWISE, "custom.hardtanh_bp",      true};
  table["silu_bp"]          = {"silu_bp",          TritonOpCategory::BINARY_ELEMENTWISE, "custom.silu_bp",          true};
  table["fused_gelu_bp"]    = {"fused_gelu_bp",    TritonOpCategory::BINARY_ELEMENTWISE, "custom.fused_gelu_bp",    true};
  table["squared_relu_bp"]  = {"squared_relu_bp",  TritonOpCategory::BINARY_ELEMENTWISE, "custom.squared_relu_bp",  true};
  table["rectifiedtanh_bp"] = {"rectifiedtanh_bp", TritonOpCategory::BINARY_ELEMENTWISE, "custom.rectifiedtanh_bp", true};
  table["swish_mul_bp"]     = {"swish_mul_bp",     TritonOpCategory::BINARY_ELEMENTWISE, "custom.swish_mul_bp",     true};

  // Scalar binary ops (treated as unary with tArgs)
  table["add_scalar"]      = {"add_scalar",      TritonOpCategory::UNARY_ELEMENTWISE, "custom.add_scalar",      true};
  table["subtract_scalar"] = {"subtract_scalar", TritonOpCategory::UNARY_ELEMENTWISE, "custom.subtract_scalar", true};
  table["multiply_scalar"] = {"multiply_scalar", TritonOpCategory::UNARY_ELEMENTWISE, "custom.multiply_scalar", true};
  table["mul_scalar"]      = {"mul_scalar",      TritonOpCategory::UNARY_ELEMENTWISE, "custom.multiply_scalar", true};
  table["divide_scalar"]   = {"divide_scalar",   TritonOpCategory::UNARY_ELEMENTWISE, "custom.divide_scalar",   true};
  table["div_scalar"]      = {"div_scalar",      TritonOpCategory::UNARY_ELEMENTWISE, "custom.divide_scalar",   true};
  table["sub_scalar"]      = {"sub_scalar",      TritonOpCategory::UNARY_ELEMENTWISE, "custom.subtract_scalar", true};
  table["rsub_scalar"]     = {"rsub_scalar",     TritonOpCategory::UNARY_ELEMENTWISE, "custom.rsub_scalar",     true};
  table["rdiv_scalar"]     = {"rdiv_scalar",     TritonOpCategory::UNARY_ELEMENTWISE, "custom.rdiv_scalar",     true};

  // Comparison ops
  table["greater"]      = {"greater",      TritonOpCategory::COMPARISON,  "arith.cmpf OGT", false};
  table["Greater"]      = {"Greater",      TritonOpCategory::COMPARISON,  "arith.cmpf OGT", false};
  table["greater_equal"]= {"greater_equal",TritonOpCategory::COMPARISON,  "arith.cmpf OGE", false};
  table["GreaterEqual"] = {"GreaterEqual", TritonOpCategory::COMPARISON,  "arith.cmpf OGE", false};
  table["less"]         = {"less",         TritonOpCategory::COMPARISON,  "arith.cmpf OLT", false};
  table["Less"]         = {"Less",         TritonOpCategory::COMPARISON,  "arith.cmpf OLT", false};
  table["less_equal"]   = {"less_equal",   TritonOpCategory::COMPARISON,  "arith.cmpf OLE", false};
  table["LessEqual"]    = {"LessEqual",    TritonOpCategory::COMPARISON,  "arith.cmpf OLE", false};
  table["equals"]       = {"equals",       TritonOpCategory::COMPARISON,  "arith.cmpf OEQ", false};
  table["Equal"]        = {"Equal",        TritonOpCategory::COMPARISON,  "arith.cmpf OEQ", false};
  table["not_equals"]   = {"not_equals",   TritonOpCategory::COMPARISON,  "arith.cmpf ONE", false};
  table["NotEqual"]     = {"NotEqual",     TritonOpCategory::COMPARISON,  "arith.cmpf ONE", false};

  // Scalar comparison ops
  table["greaterthan_scalar"]        = {"greaterthan_scalar",        TritonOpCategory::COMPARISON,  "arith.cmpf OGT", false};
  table["greaterthanorequal_scalar"] = {"greaterthanorequal_scalar", TritonOpCategory::COMPARISON,  "arith.cmpf OGE", false};
  table["lessthan_scalar"]           = {"lessthan_scalar",           TritonOpCategory::COMPARISON,  "arith.cmpf OLT", false};
  table["lessthanorequal_scalar"]    = {"lessthanorequal_scalar",    TritonOpCategory::COMPARISON,  "arith.cmpf OLE", false};
  table["equals_scalar"]             = {"equals_scalar",             TritonOpCategory::COMPARISON,  "arith.cmpf OEQ", false};
  table["notequals_scalar"]          = {"notequals_scalar",          TritonOpCategory::COMPARISON,  "arith.cmpf ONE", false};

  // Logical ops
  table["boolean_and"]  = {"boolean_and",  TritonOpCategory::LOGICAL, "arith.andi", false};
  table["And"]          = {"And",          TritonOpCategory::LOGICAL, "arith.andi", false};
  table["boolean_or"]   = {"boolean_or",   TritonOpCategory::LOGICAL, "arith.ori",  false};
  table["Or"]           = {"Or",           TritonOpCategory::LOGICAL, "arith.ori",  false};
  table["boolean_xor"]  = {"boolean_xor",  TritonOpCategory::LOGICAL, "arith.xori", false};
  table["boolean_not"]  = {"boolean_not",  TritonOpCategory::LOGICAL, "custom.not", true};
  table["BooleanNot"]   = {"BooleanNot",   TritonOpCategory::LOGICAL, "custom.not", true};
  table["bool_not"]     = {"bool_not",     TritonOpCategory::LOGICAL, "custom.not", true};

  // Ternary/select ops
  table["Where"]    = {"Where",    TritonOpCategory::TERNARY, "arith.select", false};
  table["where_np"] = {"where_np", TritonOpCategory::TERNARY, "arith.select", false};
  table["select"]   = {"select",   TritonOpCategory::TERNARY, "arith.select", false};

  // Cast / type conversion
  table["cast"]       = {"cast",       TritonOpCategory::CAST, "custom.cast", true};
  table["Cast"]       = {"Cast",       TritonOpCategory::CAST, "custom.cast", true};

  // Identity / no-op (SSA forwarding)
  table["identity"]   = {"identity",   TritonOpCategory::IDENTITY, "identity", false};
  table["Identity"]   = {"Identity",   TritonOpCategory::IDENTITY, "identity", false};
  table["assign"]     = {"assign",     TritonOpCategory::IDENTITY, "identity", false};
  table["Assign"]     = {"Assign",     TritonOpCategory::IDENTITY, "identity", false};
  table["stop_gradient"] = {"stop_gradient", TritonOpCategory::IDENTITY, "identity", false};

  // Matmul
  table["matmul"]     = {"matmul",     TritonOpCategory::MATMUL, "tt.dot",     false};
  table["MatMul"]     = {"MatMul",     TritonOpCategory::MATMUL, "tt.dot",     false};
  table["mmul"]       = {"mmul",       TritonOpCategory::MATMUL, "tt.dot",     false};
  table["Mmul"]       = {"Mmul",       TritonOpCategory::MATMUL, "tt.dot",     false};
  table["batched_gemm"]  = {"batched_gemm",  TritonOpCategory::MATMUL, "tt.dot", false};
  table["BatchedGemm"]   = {"BatchedGemm",   TritonOpCategory::MATMUL, "tt.dot", false};

  // Fused attention
  table["dot_product_attention"]    = {"dot_product_attention",    TritonOpCategory::FUSED_ATTENTION, "custom.flash_attn", true};
  table["dot_product_attention_v2"] = {"dot_product_attention_v2", TritonOpCategory::FUSED_ATTENTION, "custom.flash_attn", true};
  table["DotProductAttention"]      = {"DotProductAttention",      TritonOpCategory::FUSED_ATTENTION, "custom.flash_attn", true};
  table["DotProductAttentionV2"]    = {"DotProductAttentionV2",    TritonOpCategory::FUSED_ATTENTION, "custom.flash_attn", true};
  // Cross-attention: Q from one modality, K/V from another (FastVLA pattern).
  // Maps to the same Flash Attention kernel — structurally identical to self-attention.
  table["cross_attention"]          = {"cross_attention",          TritonOpCategory::FUSED_ATTENTION, "custom.flash_attn", true};
  table["CrossAttention"]           = {"CrossAttention",           TritonOpCategory::FUSED_ATTENTION, "custom.flash_attn", true};
  table["vision_language_cross_attention"] = {"vision_language_cross_attention", TritonOpCategory::FUSED_ATTENTION, "custom.flash_attn", true};
  table["VisionLanguageCrossAttention"]   = {"VisionLanguageCrossAttention",   TritonOpCategory::FUSED_ATTENTION, "custom.flash_attn", true};

  // Reductions
  table["reduce_sum"]    = {"reduce_sum",    TritonOpCategory::REDUCTION, "tt.reduce sum",  false};
  table["ReduceSum"]     = {"ReduceSum",     TritonOpCategory::REDUCTION, "tt.reduce sum",  false};
  table["reduce_mean"]   = {"reduce_mean",   TritonOpCategory::REDUCTION, "tt.reduce mean", true};
  table["ReduceMean"]    = {"ReduceMean",    TritonOpCategory::REDUCTION, "tt.reduce mean", true};
  table["reduce_max"]    = {"reduce_max",    TritonOpCategory::REDUCTION, "tt.reduce max",  false};
  table["ReduceMax"]     = {"ReduceMax",     TritonOpCategory::REDUCTION, "tt.reduce max",  false};
  table["reduce_min"]    = {"reduce_min",    TritonOpCategory::REDUCTION, "tt.reduce min",  false};
  table["ReduceMin"]     = {"ReduceMin",     TritonOpCategory::REDUCTION, "tt.reduce min",  false};
  table["reduce_prod"]   = {"reduce_prod",   TritonOpCategory::REDUCTION, "tt.reduce prod", false};
  table["ReduceProd"]    = {"ReduceProd",    TritonOpCategory::REDUCTION, "tt.reduce prod", false};
  table["reduce_norm2"]  = {"reduce_norm2",  TritonOpCategory::REDUCTION, "custom.norm2",   true};
  table["ReduceNorm2"]   = {"ReduceNorm2",   TritonOpCategory::REDUCTION, "custom.norm2",   true};
  table["reduce_norm1"]  = {"reduce_norm1",  TritonOpCategory::REDUCTION, "custom.norm1",   true};
  table["ReduceNorm1"]   = {"ReduceNorm1",   TritonOpCategory::REDUCTION, "custom.norm1",   true};
  table["reduce_logsumexp"] = {"reduce_logsumexp", TritonOpCategory::REDUCTION, "custom.logsumexp", true};
  table["argmax"]        = {"argmax",        TritonOpCategory::REDUCTION, "custom.argmax",  true};
  table["Argmax"]        = {"Argmax",        TritonOpCategory::REDUCTION, "custom.argmax",  true};
  table["argmin"]        = {"argmin",        TritonOpCategory::REDUCTION, "custom.argmin",  true};
  table["Argmin"]        = {"Argmin",        TritonOpCategory::REDUCTION, "custom.argmin",  true};
  table["norm_max"]      = {"norm_max",      TritonOpCategory::REDUCTION, "tt.reduce max",  false};
  table["NormMax"]       = {"NormMax",       TritonOpCategory::REDUCTION, "tt.reduce max",  false};
  table["reduce_variance"]= {"reduce_variance",TritonOpCategory::REDUCTION, "custom.variance", true};
  table["reduce_stdev"]  = {"reduce_stdev",  TritonOpCategory::REDUCTION, "custom.stdev",   true};

  // Normalization ops
  table["softmax"]              = {"softmax",              TritonOpCategory::NORMALIZATION, "custom.softmax",      true};
  table["Softmax"]              = {"Softmax",              TritonOpCategory::NORMALIZATION, "custom.softmax",      true};
  table["log_softmax"]          = {"log_softmax",          TritonOpCategory::NORMALIZATION, "custom.log_softmax",  true};
  table["LogSoftmax"]           = {"LogSoftmax",           TritonOpCategory::NORMALIZATION, "custom.log_softmax",  true};
  table["layer_norm"]           = {"layer_norm",           TritonOpCategory::NORMALIZATION, "custom.layer_norm",   true};
  table["LayerNorm"]            = {"LayerNorm",            TritonOpCategory::NORMALIZATION, "custom.layer_norm",   true};
  table["layer_normalization"]  = {"layer_normalization",  TritonOpCategory::NORMALIZATION, "custom.layer_norm",   true};
  table["LayerNormalization"]   = {"LayerNormalization",   TritonOpCategory::NORMALIZATION, "custom.layer_norm",   true};
  table["rms_norm"]             = {"rms_norm",             TritonOpCategory::NORMALIZATION, "custom.rms_norm",     true};
  table["RmsNorm"]              = {"RmsNorm",              TritonOpCategory::NORMALIZATION, "custom.rms_norm",     true};
  table["skip_rms_norm"]        = {"skip_rms_norm",        TritonOpCategory::NORMALIZATION, "custom.skip_rms_norm", true};
  table["SkipRmsNorm"]          = {"SkipRmsNorm",          TritonOpCategory::NORMALIZATION, "custom.skip_rms_norm", true};
  // rms_norm_linear: fused norm+matmul with single-pass Triton kernel
  table["rms_norm_linear"]      = {"rms_norm_linear",      TritonOpCategory::MATMUL,        "custom.rms_norm_linear", true};
  table["RmsNormLinear"]        = {"RmsNormLinear",        TritonOpCategory::MATMUL,        "custom.rms_norm_linear", true};
  // fused_gemm_swiglu: GatedMLP, treated as MATMUL for segment/section handling
  table["fused_gemm_swiglu"]    = {"fused_gemm_swiglu",    TritonOpCategory::MATMUL,        "custom.fused_gemm_swiglu", true};
  table["FusedGemmSwiglu"]      = {"FusedGemmSwiglu",      TritonOpCategory::MATMUL,        "custom.fused_gemm_swiglu", true};
  // fused_two_layer_mlp: Two chained matmuls with intermediate activation (FastVLA pattern).
  // activation2(activation1(x @ W1 + b1) @ W2 + b2) — intermediate stays in registers.
  table["fused_two_layer_mlp"]  = {"fused_two_layer_mlp",  TritonOpCategory::MATMUL,        "custom.fused_two_layer_mlp", true};
  table["FusedTwoLayerMlp"]     = {"FusedTwoLayerMlp",     TritonOpCategory::MATMUL,        "custom.fused_two_layer_mlp", true};
  table["batch_norm"]           = {"batch_norm",           TritonOpCategory::NORMALIZATION, "custom.batch_norm",   true};
  table["BatchNorm"]            = {"BatchNorm",            TritonOpCategory::NORMALIZATION, "custom.batch_norm",   true};
  table["normalize_moments"]    = {"normalize_moments",    TritonOpCategory::NORMALIZATION, "custom.normalize_moments", true};
  table["NormalizeMoments"]     = {"NormalizeMoments",     TritonOpCategory::NORMALIZATION, "custom.normalize_moments", true};

  // Normalization backward ops — multi-output (dx + dgamma [+ dbeta])
  // These use emitNormalizationBackwardSection which writes outputs directly via tt.store.
  table["rms_norm_bp"]          = {"rms_norm_bp",          TritonOpCategory::NORMALIZATION, "custom.rms_norm_bp",       true};
  table["RmsNormBp"]            = {"RmsNormBp",            TritonOpCategory::NORMALIZATION, "custom.rms_norm_bp",       true};
  table["layer_norm_bp"]        = {"layer_norm_bp",        TritonOpCategory::NORMALIZATION, "custom.layer_norm_bp",     true};
  table["LayerNormBp"]          = {"LayerNormBp",          TritonOpCategory::NORMALIZATION, "custom.layer_norm_bp",     true};
  table["fused_layer_norm_bp"]  = {"fused_layer_norm_bp",  TritonOpCategory::NORMALIZATION, "custom.layer_norm_bp",     true};
  table["FusedLayerNormBp"]     = {"FusedLayerNormBp",     TritonOpCategory::NORMALIZATION, "custom.layer_norm_bp",     true};

  // Rotary position embedding ops
  table["fused_rope"]           = {"fused_rope",           TritonOpCategory::ROPE, "custom.rope",     true};
  table["FusedRope"]            = {"FusedRope",            TritonOpCategory::ROPE, "custom.rope",     true};
  table["rope"]                 = {"rope",                 TritonOpCategory::ROPE, "custom.rope",     true};
  table["Rope"]                 = {"Rope",                 TritonOpCategory::ROPE, "custom.rope",     true};

  // Shape manipulation (logical transforms, no data movement in Triton IR)
  table["reshape"]       = {"reshape",       TritonOpCategory::SHAPE_MANIPULATION, "custom.reshape",     true};
  table["Reshape"]       = {"Reshape",       TritonOpCategory::SHAPE_MANIPULATION, "custom.reshape",     true};
  table["permute"]       = {"permute",       TritonOpCategory::SHAPE_MANIPULATION, "custom.permute",     true};
  table["Permute"]       = {"Permute",       TritonOpCategory::SHAPE_MANIPULATION, "custom.permute",     true};
  table["transpose"]     = {"transpose",     TritonOpCategory::SHAPE_MANIPULATION, "custom.transpose",   true};
  table["Transpose"]     = {"Transpose",     TritonOpCategory::SHAPE_MANIPULATION, "custom.transpose",   true};
  table["expand_dims"]   = {"expand_dims",   TritonOpCategory::SHAPE_MANIPULATION, "custom.expand_dims", true};
  table["ExpandDims"]    = {"ExpandDims",    TritonOpCategory::SHAPE_MANIPULATION, "custom.expand_dims", true};
  table["squeeze"]       = {"squeeze",       TritonOpCategory::SHAPE_MANIPULATION, "custom.squeeze",     true};
  table["Squeeze"]       = {"Squeeze",       TritonOpCategory::SHAPE_MANIPULATION, "custom.squeeze",     true};
  table["flatten"]       = {"flatten",       TritonOpCategory::SHAPE_MANIPULATION, "custom.flatten",     true};
  table["Flatten"]       = {"Flatten",       TritonOpCategory::SHAPE_MANIPULATION, "custom.flatten",     true};

  // Data movement ops
  table["gather"]        = {"gather",        TritonOpCategory::DATA_MOVEMENT, "custom.gather",      true};
  table["Gather"]        = {"Gather",        TritonOpCategory::DATA_MOVEMENT, "custom.gather",      true};
  table["gather_nd"]     = {"gather_nd",     TritonOpCategory::DATA_MOVEMENT, "custom.gather_nd",   true};
  table["GatherNd"]      = {"GatherNd",      TritonOpCategory::DATA_MOVEMENT, "custom.gather_nd",   true};
  table["concat"]        = {"concat",        TritonOpCategory::DATA_MOVEMENT, "custom.concat",      true};
  table["Concat"]        = {"Concat",        TritonOpCategory::DATA_MOVEMENT, "custom.concat",      true};
  table["strided_slice"]       = {"strided_slice",       TritonOpCategory::DATA_MOVEMENT, "custom.strided_slice", true};
  table["StridedSlice"]        = {"StridedSlice",        TritonOpCategory::DATA_MOVEMENT, "custom.strided_slice", true};
  table["slice"]               = {"slice",               TritonOpCategory::DATA_MOVEMENT, "custom.slice",         true};
  table["Slice"]               = {"Slice",               TritonOpCategory::DATA_MOVEMENT, "custom.slice",         true};
  table["split"]               = {"split",               TritonOpCategory::DATA_MOVEMENT, "custom.split",         true};
  table["Split"]               = {"Split",               TritonOpCategory::DATA_MOVEMENT, "custom.split",         true};
  table["split_v"]             = {"split_v",             TritonOpCategory::DATA_MOVEMENT, "custom.split_v",       true};
  table["SplitV"]              = {"SplitV",              TritonOpCategory::DATA_MOVEMENT, "custom.split_v",       true};
  table["stack"]               = {"stack",               TritonOpCategory::DATA_MOVEMENT, "custom.stack",         true};
  table["Stack"]               = {"Stack",               TritonOpCategory::DATA_MOVEMENT, "custom.stack",         true};
  table["tile"]                = {"tile",                TritonOpCategory::DATA_MOVEMENT, "custom.tile",          true};
  table["Tile"]                = {"Tile",                TritonOpCategory::DATA_MOVEMENT, "custom.tile",          true};
  table["repeat"]              = {"repeat",              TritonOpCategory::DATA_MOVEMENT, "custom.tile",          true};
  table["Repeat"]              = {"Repeat",              TritonOpCategory::DATA_MOVEMENT, "custom.tile",          true};
  // broadcast_to: classified as IDENTITY so it fuses into elementwise sections.
  // The elementwise preloader handles N-D broadcast indexing (unravel→mod→ravel)
  // instead of flat modular indexing, which is correct for arbitrary dimension broadcasting.
  table["broadcast_to"]          = {"broadcast_to",          TritonOpCategory::IDENTITY,       "identity",             false};
  table["BroadcastTo"]           = {"BroadcastTo",           TritonOpCategory::IDENTITY,       "identity",             false};
  table["scatter_nd"]          = {"scatter_nd",          TritonOpCategory::DATA_MOVEMENT, "custom.scatter_nd",    true};
  table["ScatterNd"]           = {"ScatterNd",           TritonOpCategory::DATA_MOVEMENT, "custom.scatter_nd",    true};
  table["scatter_nd_update"]   = {"scatter_nd_update",   TritonOpCategory::DATA_MOVEMENT, "custom.scatter_nd_update", true};
  table["ScatterNdUpdate"]     = {"ScatterNdUpdate",     TritonOpCategory::DATA_MOVEMENT, "custom.scatter_nd_update", true};

  // Constant generation ops
  table["ones_like"]           = {"ones_like",           TritonOpCategory::CONSTANT_GENERATION, "custom.ones_like",  true};
  table["OnesLike"]            = {"OnesLike",            TritonOpCategory::CONSTANT_GENERATION, "custom.ones_like",  true};
  table["ones_as"]             = {"ones_as",             TritonOpCategory::CONSTANT_GENERATION, "custom.ones_like",  true};
  table["zeros_like"]          = {"zeros_like",          TritonOpCategory::CONSTANT_GENERATION, "custom.zeros_like", true};
  table["ZerosLike"]           = {"ZerosLike",           TritonOpCategory::CONSTANT_GENERATION, "custom.zeros_like", true};
  table["zeros_as"]            = {"zeros_as",            TritonOpCategory::CONSTANT_GENERATION, "custom.zeros_like", true};
  table["zeroslike"]           = {"zeroslike",           TritonOpCategory::CONSTANT_GENERATION, "custom.zeros_like", true};
  table["oneslike"]            = {"oneslike",            TritonOpCategory::CONSTANT_GENERATION, "custom.ones_like",  true};
  table["range"]               = {"range",               TritonOpCategory::CONSTANT_GENERATION, "custom.range",      true};
  table["Range"]               = {"Range",               TritonOpCategory::CONSTANT_GENERATION, "custom.range",      true};
  table["shape_of"]            = {"shape_of",            TritonOpCategory::CONSTANT_GENERATION, "custom.shape_of",   true};
  table["ShapeOf"]             = {"ShapeOf",             TritonOpCategory::CONSTANT_GENERATION, "custom.shape_of",   true};
  table["size_at"]             = {"size_at",             TritonOpCategory::CONSTANT_GENERATION, "custom.size_at",    true};
  table["SizeAt"]              = {"SizeAt",              TritonOpCategory::CONSTANT_GENERATION, "custom.size_at",    true};
  table["rank"]                = {"rank",                TritonOpCategory::CONSTANT_GENERATION, "custom.rank",       true};
  table["Rank"]                = {"Rank",                TritonOpCategory::CONSTANT_GENERATION, "custom.rank",       true};
  table["create"]              = {"create",              TritonOpCategory::CONSTANT_GENERATION, "custom.create",     true};
  table["Create"]              = {"Create",              TritonOpCategory::CONSTANT_GENERATION, "custom.create",     true};

  // Convolution ops
  table["conv2d"]              = {"conv2d",              TritonOpCategory::CONVOLUTION, "custom.conv2d", true};
  table["Conv2d"]              = {"Conv2d",              TritonOpCategory::CONVOLUTION, "custom.conv2d", true};
  table["conv2d_bp"]           = {"conv2d_bp",           TritonOpCategory::CONVOLUTION, "custom.conv2d_bp", true};
  table["deconv2d"]            = {"deconv2d",            TritonOpCategory::CONVOLUTION, "custom.deconv2d", true};
  table["deconv2d_bp"]         = {"deconv2d_bp",         TritonOpCategory::CONVOLUTION, "custom.deconv2d_bp", true};
  table["im2col"]              = {"im2col",              TritonOpCategory::CONVOLUTION, "custom.im2col", true};
  table["Im2col"]              = {"Im2col",              TritonOpCategory::CONVOLUTION, "custom.im2col", true};
  table["im2col_bp"]           = {"im2col_bp",           TritonOpCategory::CONVOLUTION, "custom.im2col_bp", true};
  table["col2im"]              = {"col2im",              TritonOpCategory::CONVOLUTION, "custom.col2im", true};
  table["Col2im"]              = {"Col2im",              TritonOpCategory::CONVOLUTION, "custom.col2im", true};
  table["col2im_bp"]           = {"col2im_bp",           TritonOpCategory::CONVOLUTION, "custom.col2im_bp", true};

  return table;
}

const std::unordered_map<std::string, TritonOpMapping>& TritonIRBuilder::getOpTable() {
  // -fno-threadsafe-statics is set in CompilerFlags.cmake, so C++11 magic statics
  // are NOT thread-safe. Use std::call_once to protect concurrent initialization.
  static std::once_flag flag;
  static std::unordered_map<std::string, TritonOpMapping> table;
  std::call_once(flag, [&]() { table = buildOpTable(); });
  return table;
}

// ─── Public API ─────────────────────────────────────────────────────────────

TritonIRBuilder::TritonIRBuilder() = default;
TritonIRBuilder::~TritonIRBuilder() = default;

void TritonIRBuilder::setSectionedBlockSizeOverride(int blockSize) {
  if (blockSize <= 0) {
    sectionedBlockSizeOverride_ = 0;
    return;
  }
  // Ensure power of 2 and reasonable range
  int p = 64;
  while (p < blockSize && p < 4096) p <<= 1;
  sectionedBlockSizeOverride_ = p;
  DSP_DIAG(JIT, "TritonIRBuilder: sectioned block size override set to %d", sectionedBlockSizeOverride_);
}

void TritonIRBuilder::clearSectionedBlockSizeOverride() {
  sectionedBlockSizeOverride_ = 0;
}

// getSectionedCooperativeTargetBlocks() moved to TritonIRBuilder_cuda.cu
// (requires NVCC for CUDA device queries, kept separate from MLIR code)

// Derive a TritonOpCategory from the op trait bitfield stored on OpDescriptor.
// This is the trait-first fallback path so that newly-added ops don't need manual
// entries in OpCategoryTable.h / buildOpTable() to be recognized — as long as
// OpTraitTable.cpp is populated (the single source of truth), routing works.
static TritonOpCategory categoryFromTraits(uint32_t traits) {
  using sd::ops::OpTraits;
  if (traits == 0) return TritonOpCategory::UNSUPPORTED;
  // Specific / fused traits first — they override generic elementwise classification.
  if (traits & OpTraits::OP_TRAIT_ATTENTION)          return TritonOpCategory::FUSED_ATTENTION;
  if (traits & OpTraits::OP_TRAIT_MATMUL)             return TritonOpCategory::MATMUL;
  if (traits & OpTraits::OP_TRAIT_NORMALIZATION)      return TritonOpCategory::NORMALIZATION;
  if (traits & OpTraits::OP_TRAIT_REDUCTION)          return TritonOpCategory::REDUCTION;
  if (traits & OpTraits::OP_TRAIT_IDENTITY)           return TritonOpCategory::IDENTITY;
  if (traits & OpTraits::OP_TRAIT_CAST)               return TritonOpCategory::CAST;
  if (traits & OpTraits::OP_TRAIT_COMPARISON)         return TritonOpCategory::COMPARISON;
  if (traits & OpTraits::OP_TRAIT_LOGICAL)            return TritonOpCategory::LOGICAL;
  if (traits & OpTraits::OP_TRAIT_TERNARY_ELEMENTWISE) return TritonOpCategory::TERNARY;
  if (traits & OpTraits::OP_TRAIT_VIEW_PRODUCING)     return TritonOpCategory::SHAPE_MANIPULATION;
  if (traits & OpTraits::OP_TRAIT_SHAPE_ONLY_OUTPUT)  return TritonOpCategory::CONSTANT_GENERATION;
  if (traits & OpTraits::OP_TRAIT_CONSTANT_GENERATION) return TritonOpCategory::CONSTANT_GENERATION;
  if (traits & (OpTraits::OP_TRAIT_GATHER | OpTraits::OP_TRAIT_GATHER_ND |
                OpTraits::OP_TRAIT_CONCAT | OpTraits::OP_TRAIT_SPLIT |
                OpTraits::OP_TRAIT_SPLIT_V | OpTraits::OP_TRAIT_STACK |
                OpTraits::OP_TRAIT_SLICE | OpTraits::OP_TRAIT_TILE |
                OpTraits::OP_TRAIT_SCATTER_ND | OpTraits::OP_TRAIT_SCATTER_ND_UPDATE |
                OpTraits::OP_TRAIT_DATA_MOVEMENT))
    return TritonOpCategory::DATA_MOVEMENT;
  // Generic elementwise classifications (activation is a subtype of unary).
  if (traits & OpTraits::OP_TRAIT_UNARY_ELEMENTWISE)  return TritonOpCategory::UNARY_ELEMENTWISE;
  if (traits & OpTraits::OP_TRAIT_BINARY_ELEMENTWISE) return TritonOpCategory::BINARY_ELEMENTWISE;
  // Data-dependent ops can't be reliably mapped — stay UNSUPPORTED so the segment
  // falls back to slot-by-slot execution.
  return TritonOpCategory::UNSUPPORTED;
}

// Look up op traits from the live op registry (OpRegistrator → OpDescriptor).
// Returns 0 if op isn't registered or has no traits set. This lets the Triton
// layer consult the same trait bitfield that FusionPass / NativePlanCompiler use.
static uint32_t lookupRegistryTraits(const std::string& opName) {
  auto* op = sd::ops::OpRegistrator::getInstance().getOperation(opName.c_str());
  if (op == nullptr) return 0;
  auto* desc = op->getOpDescriptor();
  return desc != nullptr ? desc->getTraits() : 0;
}

bool TritonIRBuilder::isTritonMappable(const std::string& opName) {
  const auto& table = getOpTable();
  if (table.find(opName) != table.end()) return true;
  // Fall back to OpCategoryTable.h (shared category-only table with broader coverage)
  const auto& catTable = getOpCategoryTable();
  if (catTable.find(opName) != catTable.end()) return true;
  // OpTraitTable.cpp traits are NOT sufficient for mappability. An op having traits
  // means it's classified for DSP segmentation, but that does NOT mean the Triton IR
  // builder can emit MLIR code for it. Only ops explicitly listed in getOpTable() or
  // OpCategoryTable.h have corresponding IR emission logic. The trait-based fallback
  // caused false positives (e.g. thresholdedrelu_bp classified as UNARY_ELEMENTWISE
  // but with no IR emission code), leading to KERNEL_FAILURE on compilation.
  DSP_DIAG(FALLBACK,
           "TritonIRBuilder::isTritonMappable: op '%s' has no entry in buildOpTable() or "
           "OpCategoryTable.h — routing to native execution. "
           "Add to OpCategoryTable.h for Triton compilation support.",
           opName.c_str());
  return false;
}

TritonOpCategory TritonIRBuilder::getOpCategory(const std::string& opName) {
  const auto& table = getOpTable();
  auto it = table.find(opName);
  if (it != table.end()) return it->second.category;
  // Fall back to OpCategoryTable.h (shared category-only table with broader coverage).
  const auto& catTable = getOpCategoryTable();
  auto catIt = catTable.find(opName);
  if (catIt != catTable.end()) return catIt->second;
  // Do NOT use OpTraitTable.cpp trait-based fallback here. OpTraitTable traits classify
  // ops for DSP segmentation, but the Triton IR builder can only emit code for ops
  // explicitly listed in getOpTable() or OpCategoryTable.h. Using trait fallback causes
  // ops to be placed in ELEMENTWISE sections without IR emission support, leading to
  // empty kernels and KERNEL_FAILURE.
  DSP_DIAG(FALLBACK, "TritonIRBuilder::getOpCategory: op '%s' not found in buildOpTable() "
            "or OpCategoryTable.h — classifying as UNSUPPORTED for Triton. "
            "Add to OpCategoryTable.h for Triton compilation support.", opName.c_str());
  return TritonOpCategory::UNSUPPORTED;
}

bool TritonIRBuilder::isElementwiseCompatible(TritonOpCategory cat) {
  return sd::graph::isElementwiseCompatible(cat);
}

// ─── Kernel name generation ─────────────────────────────────────────────────

static uint64_t hashKernelNameFNV1a(const std::string& text) {
  return sd::graph::dsp::fnv1a64String(text);
}

std::string TritonIRBuilder::generateKernelName(NativeSlot* slots, int startSlot, int endSlot) {
  std::ostringstream ss;
  ss << "triton_fused";
  for (int i = startSlot; i <= endSlot; i++) {
    ss << "_" << slots[i].ident.opName;
  }
  std::string name = ss.str();
  if (name.size() > 200) {
    uint64_t suffixHash = hashKernelNameFNV1a(name);
    name = name.substr(0, 176) + "_h" + std::to_string(static_cast<unsigned long long>(suffixHash));
  }
  return name;
}

}  // namespace graph
}  // namespace sd

#endif  // HAVE_TRITON

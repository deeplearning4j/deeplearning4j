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
#include <system/Environment.h>
#include <system/common.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <sstream>

#ifdef SD_CUDA
#include <cuda_runtime.h>
#endif

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

namespace {

inline int nextPow2AtLeastOne(int value) {
  if (value <= 1) return 1;
  int p = 1;
  while (p < value && p < (1 << 30)) p <<= 1;
  return p;
}

inline int clampPow2(int value, int minPow2, int maxPow2) {
  int v = std::max(minPow2, std::min(value, maxPow2));
  if (v & (v - 1)) {
    int p = 1;
    while (p < v && p < (1 << 30)) p <<= 1;
    v = p;
  }
  if (v > maxPow2) v = maxPow2;
  if (v < minPow2) v = minPow2;
  return v;
}

inline int queryCudaSharedMemLimitBytes() {
  // Conservative fallback when runtime limits are unavailable.
  int limit = 49152;
#ifdef SD_CUDA
  int currentDevice = -1;
  auto devErr = cudaGetDevice(&currentDevice);
  if (devErr != cudaSuccess || currentDevice < 0) {
    cudaGetLastError();
    return limit;
  }

  int optIn = 0;
  auto optErr = cudaDeviceGetAttribute(
      &optIn, cudaDevAttrMaxSharedMemoryPerBlockOptin, currentDevice);
  if (optErr == cudaSuccess && optIn > 0) {
    return optIn;
  }
  cudaGetLastError();

  int defaultLimit = 0;
  auto defErr = cudaDeviceGetAttribute(
      &defaultLimit, cudaDevAttrMaxSharedMemoryPerBlock, currentDevice);
  if (defErr == cudaSuccess && defaultLimit > 0) {
    return defaultLimit;
  }
  cudaGetLastError();
#endif
  return limit;
}

inline int estimateFusedAttentionSharedMemBytes(int headDim, int blockM, int blockN) {
  // Approximate dominant shared-memory footprint of the current fused-attention
  // kernel structure:
  //   Q  tile: [BM, HD]
  //   K  tile: [BN, HD]
  //   V  tile: [BN, HD]
  // plus a fixed overhead for softmax/reduction temporaries.
  //
  // HD is rounded to power-of-2 by emitFusedAttentionKernel().
  constexpr int kBytesPerF32 = 4;
  constexpr int kFixedOverheadBytes = 6144;
  const int headDimPadded = nextPow2AtLeastOne(std::max(1, headDim));
  long long bytes = static_cast<long long>(kBytesPerF32) * headDimPadded *
                    (static_cast<long long>(blockM) + 2LL * blockN);
  bytes += kFixedOverheadBytes;
  if (bytes > static_cast<long long>(std::numeric_limits<int>::max())) {
    return std::numeric_limits<int>::max();
  }
  return static_cast<int>(bytes);
}

struct AttentionTileChoice {
  int blockM = 32;
  int blockN = 32;
  int estimatedSharedMemBytes = 0;
  int sharedMemLimitBytes = 0;
  bool adjustedForSharedMem = false;
  bool fitsSharedMem = true;
};

inline AttentionTileChoice chooseFusedAttentionTileConfig(int batchSize, int numHeads,
                                                          int seqQ, int seqK, int headDim,
                                                          int sharedMemLimitBytes = -1) {
  (void)seqK;
  AttentionTileChoice choice;
  const int limit = (sharedMemLimitBytes > 0) ? sharedMemLimitBytes : queryCudaSharedMemLimitBytes();

  LongType numTads = static_cast<LongType>(std::max(1, batchSize)) *
                     std::max(1, numHeads) *
                     std::max(1, seqQ);
  dim3 attDims = getSoftmaxDims(numTads, static_cast<LongType>(std::max(1, headDim)));
  int preferredM = std::max(32, (static_cast<int>(attDims.y) / 32) * 2);
  preferredM = clampPow2(preferredM, 32, 128);
  int preferredN = preferredM;

  int chosenM = preferredM;
  int chosenN = preferredN;
  int chosenBytes = estimateFusedAttentionSharedMemBytes(headDim, chosenM, chosenN);

  if (chosenBytes > limit) {
    bool found = false;
    for (int m = preferredM; m >= 4 && !found; m >>= 1) {
      int nStart = std::min(preferredN, m);
      if (nStart & (nStart - 1)) {
        int p = 1;
        while (p < nStart && p < (1 << 30)) p <<= 1;
        nStart = std::min(p, m);
      }
      for (int n = nStart; n >= 4; n >>= 1) {
        int bytes = estimateFusedAttentionSharedMemBytes(headDim, m, n);
        if (bytes <= limit) {
          chosenM = m;
          chosenN = n;
          chosenBytes = bytes;
          found = true;
          break;
        }
      }
    }
    if (!found) {
      chosenM = 4;
      chosenN = 4;
      chosenBytes = estimateFusedAttentionSharedMemBytes(headDim, chosenM, chosenN);
    }
  }

  choice.blockM = chosenM;
  choice.blockN = chosenN;
  choice.estimatedSharedMemBytes = chosenBytes;
  choice.sharedMemLimitBytes = limit;
  choice.adjustedForSharedMem = (chosenM != preferredM) || (chosenN != preferredN);
  choice.fitsSharedMem = (chosenBytes <= limit);
  return choice;
}

}  // namespace

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

void TritonIRBuilder::setSectionedBlockSizeOverride(int blockSize) {
  if (blockSize <= 0) {
    sectionedBlockSizeOverride_ = 0;
    return;
  }
  int rounded = 1;
  while (rounded < blockSize && rounded < 16384) rounded <<= 1;
  if (rounded < 64) rounded = 64;
  if (rounded > 16384) rounded = 16384;
  sectionedBlockSizeOverride_ = rounded;
}

void TritonIRBuilder::clearSectionedBlockSizeOverride() {
  sectionedBlockSizeOverride_ = 0;
}

static int getSectionedCooperativeTargetBlocks() {
  int configured = sd::Environment::getInstance().tritonCoopTargetBlocks();
  if (configured > 0) return configured;

#ifdef SD_CUDA
  int device = 0;
  if (cudaGetDevice(&device) == cudaSuccess) {
    cudaDeviceProp props;
    if (cudaGetDeviceProperties(&props, device) == cudaSuccess &&
        props.multiProcessorCount > 0) {
      // One cooperative block per SM is the strictest guaranteed residency target.
      return props.multiProcessorCount;
    }
  }
#endif

  // Conservative default when device query is unavailable.
  return 128;
}

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

// Registry of pattern detectors.
// NOTE: libnd4j is compiled with -fno-threadsafe-statics, so function-local
// static initialization is not synchronized. Keep this registry as a fixed
// global object with immutable detector pointers to avoid racey lazy init.
class PatternRegistry {
 public:
  PatternRegistry()
      : detectors_{&fusedAttentionOpDetector_,
                   &attentionPatternDetector_,
                   &ffnBlockDetector_,
                   &decomposedSoftmaxDetector_,
                   &matmulEpilogueDetector_,
                   &elementwisePatternDetector_,
                   &megaSegmentDetector_} {}

  const std::array<PatternDetector*, 7>& detectors() const { return detectors_; }

 private:
  FusedAttentionOpDetector fusedAttentionOpDetector_;
  AttentionPatternDetector attentionPatternDetector_;
  FFNBlockDetector ffnBlockDetector_;
  DecomposedSoftmaxDetector decomposedSoftmaxDetector_;
  MatmulEpilogueDetector matmulEpilogueDetector_;
  ElementwisePatternDetector elementwisePatternDetector_;
  MegaSegmentDetector megaSegmentDetector_;
  std::array<PatternDetector*, 7> detectors_;
};

PatternRegistry gPatternRegistry;

}  // anonymous namespace

MatchedPatterns TritonIRBuilder::matchPatterns(const SegmentProfile& profile,
                                                NativeSlot* slots, int startSlot) {
  MatchedPatterns matched;
  const auto& detectors = gPatternRegistry.detectors();
  auto& env = Environment::getInstance();
  const bool collectAllMatches = env.tritonLogAllPatterns() || env.tritonVerbose();
  constexpr int MAX_PATTERN_PRIORITY = 100;  // FUSED_ATTENTION_OP

  int bestPriority = std::numeric_limits<int>::min();
  for (auto* detector : detectors) {
    auto hits = detector->detect(profile, slots, startSlot);

    if (collectAllMatches) {
      for (auto& hit : hits) {
        if (hit.priority > bestPriority) bestPriority = hit.priority;
        matched.matches.push_back(std::move(hit));
      }
    } else {
      for (const auto& hit : hits) {
        if (hit.priority > bestPriority) {
          bestPriority = hit.priority;
          matched.matches.clear();
          matched.matches.push_back(hit);
        }
      }
      // Cannot beat max priority; skip lower-priority detector passes.
      if (bestPriority >= MAX_PATTERN_PRIORITY) break;
    }
  }

  if (collectAllMatches) {
    // Sort by priority (descending)
    std::sort(matched.matches.begin(), matched.matches.end(),
              [](const PatternMatch& a, const PatternMatch& b) { return a.priority > b.priority; });
  }

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

  auto& env = Environment::getInstance();
  const bool logAllPatterns = env.tritonLogAllPatterns() || env.tritonVerbose();
  if (logAllPatterns) {
    for (auto& m : matched.matches) {
      sd_printf("  pattern: %s (priority=%d, %d ops)\n",
                m.description.c_str(), m.priority, static_cast<int>(m.localIndices.size()));
    }
  } else if (!matched.matches.empty()) {
    const auto* best = matched.bestMatch();
    if (best != nullptr) {
      sd_printf("  pattern(best): %s (priority=%d, %d ops)%s\n",
                best->description.c_str(), best->priority,
                static_cast<int>(best->localIndices.size()),
                matched.matches.size() > 1 ? " [set ND4J_TRITON_LOG_ALL_PATTERNS=1 for full list]" : "");
    }
  } else {
    sd_printf("  pattern(best): none\n", "");
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
    int suggestedWarps = std::max(1, static_cast<int>(dims.y) / 32);
    // Fused attention kernels in this backend use reductions that become
    // invalid when CTA size grows beyond 256 threads. Keep attention launch
    // width conservative to avoid out-of-bounds shared-memory accesses.
    numWarps = std::max(1, std::min(suggestedWarps, 8));
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

static uint64_t hashKernelNameFNV1a(const std::string& text) {
  uint64_t hash = 1469598103934665603ULL;  // FNV-1a 64-bit offset basis
  for (unsigned char c : text) {
    hash ^= static_cast<uint64_t>(c);
    hash *= 1099511628211ULL;  // FNV prime
  }
  return hash;
}

std::string TritonIRBuilder::generateKernelName(NativeSlot* slots, int startSlot, int endSlot) {
  std::ostringstream ss;
  ss << "triton_fused";
  for (int i = startSlot; i <= endSlot; i++) {
    ss << "_" << slots[i].opName;
  }
  std::string name = ss.str();
  if (name.size() > 200) {
    uint64_t suffixHash = hashKernelNameFNV1a(name);
    name = name.substr(0, 176) + "_h" + std::to_string(static_cast<unsigned long long>(suffixHash));
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
  // Fallback: cast to actual element type to avoid tt.splat type mismatch
  // The tensor element type must match the scalar type exactly
  auto scalarAttr = builder.getFloatAttr(builder.getF32Type(), static_cast<double>(val));
  mlir::Value scalarVal = builder.create<mlir::arith::ConstantOp>(loc, builder.getF32Type(), scalarAttr);
  // If tensorType element type isn't f32, we need to cast
  if (elemType != builder.getF32Type()) {
    if (mlir::isa<mlir::FloatType>(elemType)) {
      scalarVal = builder.create<mlir::arith::ExtFOp>(loc, elemType, scalarVal, nullptr);
    } else if (elemType.isSignlessInteger()) {
      scalarVal = builder.create<mlir::arith::FPToSIOp>(loc, elemType, scalarVal);
    } else {
      // Last resort: rebuild tensorType with f32 element type
      tensorType = mlir::RankedTensorType::get(tensorType.getShape(), builder.getF32Type());
    }
  }
  return builder.create<mlir::triton::SplatOp>(loc, tensorType, scalarVal);
}

mlir::Value TritonIRBuilder::splatConstantI32(mlir::OpBuilder& builder, mlir::Location loc,
                                               mlir::RankedTensorType tensorType, int val) {
  auto elemType = tensorType.getElementType();
  if (elemType.isSignlessInteger()) {
    // Create scalar matching the tensor's actual integer bit width
    int bitWidth = elemType.getIntOrFloatBitWidth();
    auto scalar = builder.create<mlir::arith::ConstantIntOp>(loc, val, bitWidth);
    return builder.create<mlir::triton::SplatOp>(loc, tensorType, scalar);
  }
  // Fallback: i32 (and rebuild tensorType to match)
  auto scalar = builder.create<mlir::arith::ConstantIntOp>(loc, val, 32);
  tensorType = mlir::RankedTensorType::get(tensorType.getShape(), builder.getI32Type());
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
      auto widened = builder.create<mlir::arith::ExtFOp>(loc, f32TensorType, val, nullptr);
      return builder.create<mlir::arith::TruncFOp>(loc, targetTensorType, widened);
    } else if (dstBits > srcBits) {
      return builder.create<mlir::arith::ExtFOp>(loc, targetTensorType, val, nullptr);
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
                                                int batchSize, int numQHeads, int numKvHeads,
                                                int seqQ, int seqK,
                                                int headDim, float scale,
                                                int blockM, int blockN,
                                                bool qIsBSHD, bool kIsBSHD,
                                                mlir::Value biasPtr,
                                                const std::vector<LongType>& biasShape) {
  // GQA: numQHeads >= numKvHeads, each KV head serves (numQHeads/numKvHeads) Q heads
  if (numKvHeads <= 0) numKvHeads = numQHeads;
  int kvGroupSize = (numKvHeads > 0) ? (numQHeads / numKvHeads) : 1;
  if (kvGroupSize < 1) kvGroupSize = 1;
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

  // Program IDs: pid0 = batch * numQHeads + qHeadIdx, pid1 = query tile index
  auto pid0 = builder.create<mlir::triton::GetProgramIdOp>(
      loc, i32Type, mlir::triton::ProgramIDDim::X);
  auto pid1 = builder.create<mlir::triton::GetProgramIdOp>(
      loc, i32Type, mlir::triton::ProgramIDDim::Y);

  // Decompose pid0 into batch and Q head indices
  auto numQHeadsConst = builder.create<mlir::arith::ConstantIntOp>(loc, numQHeads, 32);
  auto numKvHeadsConst = builder.create<mlir::arith::ConstantIntOp>(loc, numKvHeads, 32);
  auto headIdx = builder.create<mlir::arith::RemSIOp>(loc, pid0, numQHeadsConst);   // Q head index [0, numQHeads)
  auto batchIdx = builder.create<mlir::arith::DivSIOp>(loc, pid0, numQHeadsConst);
  // GQA: map Q head to KV head — kvHeadIdx = headIdx / kvGroupSize
  auto kvGroupSizeConst = builder.create<mlir::arith::ConstantIntOp>(loc, kvGroupSize, 32);
  auto kvHeadIdx = builder.create<mlir::arith::DivSIOp>(loc, headIdx, kvGroupSizeConst);

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

  // Compute base offset into Q/K/V/Out buffers.
  // BHSD (4D): [batch, heads, seq, headDim] — base = batch*NH*S*HD + head*S*HD, rowStride=HD
  // BSHD (3D): [batch, seq, NH*HD]         — base = batch*S*NH*HD + head*HD,   rowStride=NH*HD
  auto seqQConst = builder.create<mlir::arith::ConstantIntOp>(loc, seqQ, 32);
  auto seqKConst = builder.create<mlir::arith::ConstantIntOp>(loc, seqK, 32);
  auto headDimConst = builder.create<mlir::arith::ConstantIntOp>(loc, headDim, 32);

  mlir::Value qBase, qRowStride;
  if (qIsBSHD) {
    // BSHD: [batch, seqQ, numQHeads*headDim]
    auto nhTimesHd = builder.create<mlir::arith::MulIOp>(loc, numQHeadsConst, headDimConst);
    auto qStride0 = builder.create<mlir::arith::MulIOp>(loc, seqQConst, nhTimesHd);
    qBase = builder.create<mlir::arith::AddIOp>(loc,
        builder.create<mlir::arith::MulIOp>(loc, batchIdx, qStride0),
        builder.create<mlir::arith::MulIOp>(loc, headIdx, headDimConst));
    qRowStride = nhTimesHd;
  } else {
    // BHSD: [batch, numQHeads, seqQ, headDim]
    auto qStride0 = builder.create<mlir::arith::MulIOp>(loc, numQHeadsConst,
        builder.create<mlir::arith::MulIOp>(loc, seqQConst, headDimConst));
    auto qStride1 = builder.create<mlir::arith::MulIOp>(loc, seqQConst, headDimConst);
    qBase = builder.create<mlir::arith::AddIOp>(loc,
        builder.create<mlir::arith::MulIOp>(loc, batchIdx, qStride0),
        builder.create<mlir::arith::MulIOp>(loc, headIdx, qStride1));
    qRowStride = headDimConst;
  }

  mlir::Value kvBase, kvRowStride;
  if (kIsBSHD) {
    // BSHD: [batch, seqK, numKvHeads*headDim] — use kvHeadIdx for GQA
    auto kvNhTimesHd = builder.create<mlir::arith::MulIOp>(loc, numKvHeadsConst, headDimConst);
    auto kvStride0 = builder.create<mlir::arith::MulIOp>(loc, seqKConst, kvNhTimesHd);
    kvBase = builder.create<mlir::arith::AddIOp>(loc,
        builder.create<mlir::arith::MulIOp>(loc, batchIdx, kvStride0),
        builder.create<mlir::arith::MulIOp>(loc, kvHeadIdx, headDimConst));
    kvRowStride = kvNhTimesHd;
  } else {
    // BHSD: [batch, numKvHeads, seqK, headDim] — use kvHeadIdx for GQA
    auto kvStride0 = builder.create<mlir::arith::MulIOp>(loc, numKvHeadsConst,
        builder.create<mlir::arith::MulIOp>(loc, seqKConst, headDimConst));
    auto kvStride1 = builder.create<mlir::arith::MulIOp>(loc, seqKConst, headDimConst);
    kvBase = builder.create<mlir::arith::AddIOp>(loc,
        builder.create<mlir::arith::MulIOp>(loc, batchIdx, kvStride0),
        builder.create<mlir::arith::MulIOp>(loc, kvHeadIdx, kvStride1));
    kvRowStride = headDimConst;
  }

  // Load Q tile [BLOCK_M, headDim]
  // Q pointer offsets: qBase + qIndices[:, None] * headDim + rangeHd[None, :]
  auto qMExpanded = builder.create<mlir::triton::ExpandDimsOp>(loc, qIndices, 1);  // [BM, 1]
  auto hdExpanded = builder.create<mlir::triton::ExpandDimsOp>(loc, rangeHd, 0);   // [1, HD]

  auto i32BmHdType = mlir::RankedTensorType::get({blockM, headDimPadded}, i32Type);
  auto f32BmHdType = mlir::RankedTensorType::get({blockM, headDimPadded}, f32Type);
  auto qRowStrideSplat = builder.create<mlir::triton::SplatOp>(loc,
      mlir::RankedTensorType::get({blockM, 1}, i32Type), qRowStride);
  auto qRowOffsets = builder.create<mlir::arith::MulIOp>(loc, qMExpanded, qRowStrideSplat);
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
  auto kvRowStrideSplat = builder.create<mlir::triton::SplatOp>(loc,
      mlir::RankedTensorType::get({blockN, 1}, i32Type), kvRowStride);
  auto kRowOffsets = builder.create<mlir::arith::MulIOp>(loc, kNExpanded, kvRowStrideSplat);
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
  auto i32BmBnType = mlir::RankedTensorType::get({blockM, blockN}, i32Type);
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

  // Apply attention bias/mask if provided
  // Bias shape: [B, H, seqQ, seqK] (rank 4 per-head) or [B, seqQ, seqK] (rank 3)
  // Load bias tile [BM, BN] and add to QK scores — this applies the attention mask
  // (valid positions have bias=0.0, masked/padding positions have bias=-inf)
  mlir::Value qkWithBias = qkMasked;
  if (biasPtr) {
    int biasRank = static_cast<int>(biasShape.size());
    // Determine bias strides based on rank:
    // Rank 4: [B, H, seqQ, seqK] → offset = b*H*seqQ*seqK + h*seqQ*seqK + q*seqK + k
    // Rank 3: [B, seqQ, seqK]    → offset = b*seqQ*seqK + q*seqK + k (no head dim)
    int biasNumHeads = (biasRank >= 4) ? static_cast<int>(biasShape[1]) : 0;
    int biasSeqQ = (biasRank >= 4) ? static_cast<int>(biasShape[2]) :
                   (biasRank >= 3) ? static_cast<int>(biasShape[1]) : seqQ;
    int biasSeqK = (biasRank >= 4) ? static_cast<int>(biasShape[3]) :
                   (biasRank >= 3) ? static_cast<int>(biasShape[2]) : seqK;

    auto biasSeqKConst = builder.create<mlir::arith::ConstantIntOp>(loc, biasSeqK, 32);

    // Compute scalar base offset for this (batch, head)
    // headSliceSize = biasSeqQ * biasSeqK
    auto biasSeqQConst = builder.create<mlir::arith::ConstantIntOp>(loc, biasSeqQ, 32);
    auto headSliceSize = builder.create<mlir::arith::MulIOp>(loc, biasSeqQConst, biasSeqKConst);

    mlir::Value biasBaseScalar;
    if (biasRank >= 4 && biasNumHeads > 1) {
      // 4D per-head: offset = batch * (H * seqQ * seqK) + head * (seqQ * seqK)
      auto biasNumHeadsConst = builder.create<mlir::arith::ConstantIntOp>(loc, biasNumHeads, 32);
      auto batchSliceSize = builder.create<mlir::arith::MulIOp>(loc, biasNumHeadsConst, headSliceSize);
      auto batchOffset = builder.create<mlir::arith::MulIOp>(loc, batchIdx, batchSliceSize);
      auto headOffset = builder.create<mlir::arith::MulIOp>(loc, headIdx, headSliceSize);
      biasBaseScalar = builder.create<mlir::arith::AddIOp>(loc, batchOffset, headOffset);
    } else {
      // 3D or 4D with H=1: offset = batch * (seqQ * seqK)
      biasBaseScalar = builder.create<mlir::arith::MulIOp>(loc, batchIdx, headSliceSize);
    }

    // Q row offsets within bias: qIndices * biasSeqK  → [BM, 1] → broadcast to [BM, BN]
    auto qBiasRowExpanded = builder.create<mlir::triton::ExpandDimsOp>(loc, qIndices, 1);  // [BM, 1]
    auto biasSeqKSplat = builder.create<mlir::triton::SplatOp>(loc,
        mlir::RankedTensorType::get({blockM, 1}, i32Type), biasSeqKConst);
    auto biasRowOffsets = builder.create<mlir::arith::MulIOp>(loc, qBiasRowExpanded, biasSeqKSplat);
    auto biasRowBroadcast = builder.create<mlir::triton::BroadcastOp>(loc, i32BmBnType, biasRowOffsets);

    // K column offsets: kIndices → [1, BN] → broadcast to [BM, BN]
    auto kBiasColExpanded = builder.create<mlir::triton::ExpandDimsOp>(loc, kIndices, 0);  // [1, BN]
    auto kBiasColBroadcast = builder.create<mlir::triton::BroadcastOp>(loc, i32BmBnType, kBiasColExpanded);

    // Final: base + qRow*seqK + kCol
    auto biasBaseOffsets = builder.create<mlir::arith::AddIOp>(loc, biasRowBroadcast, kBiasColBroadcast);
    auto biasBaseSplat = builder.create<mlir::triton::SplatOp>(loc, i32BmBnType, biasBaseScalar);
    auto biasFinalOffsets = builder.create<mlir::arith::AddIOp>(loc, biasBaseSplat, biasBaseOffsets);

    // Create bias pointer tensor and load
    auto biasPtrType = mlir::cast<mlir::triton::PointerType>(biasPtr.getType());
    auto biasPtrTensorType = mlir::RankedTensorType::get({blockM, blockN}, biasPtrType);
    auto biasSplat = builder.create<mlir::triton::SplatOp>(loc, biasPtrTensorType, biasPtr);
    auto biasPtrs = builder.create<mlir::triton::AddPtrOp>(loc, biasPtrTensorType, biasSplat, biasFinalOffsets);

    // Bias mask: same as kMaskBmBn (valid Q and K positions)
    auto biasLoadedRaw = builder.create<mlir::triton::LoadOp>(loc,
        biasPtrs, kMaskBmBn, mlir::Value(),
        mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL, false);
    auto biasLoaded = castTo(builder, loc, biasLoadedRaw, f32Type);

    // Add bias to QK scores: qk_biased = qk_masked + bias
    qkWithBias = builder.create<mlir::arith::AddFOp>(loc, qkMasked, biasLoaded);
  }

  // Online softmax update:
  // m_new = max(m_i, row_max(qk))
  // correction = exp(m_i - m_new)
  // p = exp(qk - splat(m_new))
  // l_i = l_i * correction + row_sum(p)
  // acc = acc * splat(correction) + dot(p, V)

  // row_max(qk) -> reduce along axis 1
  mlir::Value qkFinalVal = qkWithBias;
  auto rowMaxOp = builder.create<mlir::triton::ReduceOp>(loc,
      mlir::ValueRange{qkFinalVal}, /*axis=*/1);
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
  auto qkShifted = builder.create<mlir::arith::SubFOp>(loc, qkWithBias, mNewBroadcast);
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

  sd_printf("TritonIRBuilder: emitted fused attention kernel batch=%d qHeads=%d kvHeads=%d seqQ=%d seqK=%d "
            "headDim=%d scale=%f BM=%d BN=%d kvGroupSize=%d hasBias=%d\n",
            batchSize, numQHeads, numKvHeads, seqQ, seqK, headDim, scale, blockM, blockN, kvGroupSize,
            biasPtr ? 1 : 0);
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
  // EXCEPTION: internal intermediates that are inputs to REDUCTION ops within this
  // segment need a buffer (SSA tensors can't be randomly indexed for segmented reduction).
  // Deduplicate: same output slot written by multiple ops only needs one kernel arg.
  auto externalOutputs = computeExternallyVisibleOutputs(
      slots, startSlot, endSlot, totalSlots,
      requestedOutputSlotIndices, numRequestedOutputs);

  // Find internal intermediates consumed by reduction ops
  std::unordered_set<int> reductionInputSlots;
  for (int i = startSlot; i <= endSlot; i++) {
    auto cat = getOpCategory(slots[i].opName);
    if (cat == TritonOpCategory::REDUCTION) {
      for (int inp = 0; inp < slots[i].numInputs; inp++) {
        int srcIdx = slots[i].inputSourceIndices[inp];
        if (srcIdx >= 0 && internalSlotOutputs.count(srcIdx) && !externalOutputs.count(srcIdx)) {
          reductionInputSlots.insert(srcIdx);
        }
      }
    }
  }

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
        if (!externalOutputs.count(outIdx) && !reductionInputSlots.count(outIdx)) {
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
  // Fallback: if output shapes are unavailable (empty), use max input element count
  // This ensures broadcast indexing works even when output shapes aren't populated at compile time
  if (maxOutputElements <= 1) {
    for (auto& iarg : inputArgs) {
      LongType iElems = 1;
      for (auto d : iarg.shape) iElems *= d;
      if (iElems > maxOutputElements) maxOutputElements = iElems;
    }
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
      // Segmented reduction: for each output element, accumulate over the reduction axis.
      // Unlike elementwise ops, reduction changes tensor size, so we can't use the SSA value
      // (which was loaded using output offsets). Instead, directly load from input buffer.
      if (slot.numInputs < 1) {
        sd_printf("TritonIRBuilder: reduction op '%s' at slot %d has no inputs\n",
                  slot.opName.c_str(), si);
        continue;
      }
      int inputSrc = slot.inputSourceIndices[0];
      // Get reduction axis from iArgs
      int reductionAxis = (slot.numIArgs > 0 && slot.iArgs) ? static_cast<int>(slot.iArgs[0]) : -1;

      // Resolve input shape
      auto inputShape = resolveShapeLocal(inputSrc);
      int inputRank = static_cast<int>(inputShape.size());
      if (inputRank == 0) {
        sd_printf("TritonIRBuilder: reduction op '%s' has no input shape info\n", slot.opName.c_str());
        continue;
      }
      // Handle negative axis
      if (reductionAxis < 0) reductionAxis += inputRank;
      if (reductionAxis < 0 || reductionAxis >= inputRank) reductionAxis = inputRank - 1;

      int reductionSize = static_cast<int>(inputShape[reductionAxis]);

      // Compute input strides (row-major)
      std::vector<int> inStrides(inputRank, 1);
      for (int d = inputRank - 2; d >= 0; d--)
        inStrides[d] = inStrides[d + 1] * static_cast<int>(inputShape[d + 1]);

      // Compute output shape (input shape with reduction axis removed)
      std::vector<int> outShape;
      for (int d = 0; d < inputRank; d++)
        if (d != reductionAxis) outShape.push_back(static_cast<int>(inputShape[d]));
      int outRank = static_cast<int>(outShape.size());
      if (outRank == 0) { outShape.push_back(1); outRank = 1; } // scalar output

      // Compute output strides (row-major)
      std::vector<int> outStrides(outRank, 1);
      for (int d = outRank - 2; d >= 0; d--)
        outStrides[d] = outStrides[d + 1] * outShape[d + 1];

      // Find the input arg for this input source.
      // If the input is an internal intermediate with a forced output buffer
      // (reductionInputSlots), store the SSA value to the buffer first.
      auto inputArgIt = slotToArgIdx.find(inputSrc);
      if (inputArgIt == slotToArgIdx.end()) {
        sd_printf("TritonIRBuilder: reduction input slot %d not found in kernel args — cannot compile segmented reduction\n", inputSrc);
        continue;
      }
      // If this is a reduction input slot (internal intermediate forced to have a buffer),
      // store the SSA value to the buffer NOW so we can load from it with proper offsets
      if (reductionInputSlots.count(inputSrc)) {
        auto ssaIt = ssaValues.find(inputSrc);
        if (ssaIt != ssaValues.end()) {
          int midArgIdx = inputArgIt->second;
          auto midFuncArg = getBufferArg(midArgIdx);
          // Derive pointer type from actual function arg (consistent with load side)
          auto midPtrType = mlir::cast<mlir::triton::PointerType>(midFuncArg.getType());
          auto midElemType = midPtrType.getPointeeType();
          auto midPtrTensorType = mlir::RankedTensorType::get({blockSize}, midPtrType);
          auto midSplatPtr = builder.create<mlir::triton::SplatOp>(loc, midPtrTensorType, midFuncArg);
          auto midPtrs = builder.create<mlir::triton::AddPtrOp>(loc, midPtrTensorType, midSplatPtr, offsets);
          mlir::Value midStoreVal = castTo(builder, loc, ssaIt->second, midElemType);
          builder.create<mlir::triton::StoreOp>(loc, midPtrs, midStoreVal, mask,
              mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL);
          sd_printf("TritonIRBuilder: stored reduction input slot %d to buffer for segmented reduction\n", inputSrc);
          // Memory fence + block barrier to ensure all threads' stores are visible
          // before any thread loads from the buffer for reduction.
          // tt.elementwise_inline_asm with a tensor input runs the ASM on all threads,
          // which is required for bar.sync 0 to not deadlock.
          // "=r,r" declares one output register (per thread) and one input register.
          {
            auto dummyTensorType = mlir::RankedTensorType::get({blockSize}, builder.getI32Type());
            auto dummyZero = builder.create<mlir::arith::ConstantIntOp>(loc, 0, 32);
            auto dummyTensor = builder.create<mlir::triton::SplatOp>(loc, dummyTensorType, dummyZero);
            builder.create<mlir::triton::ElementwiseInlineAsmOp>(
                loc, mlir::TypeRange{dummyTensorType},
                "membar.gl; bar.sync 0; mov.b32 $0, $1;",
                "=r,r", /*isPure=*/false,
                /*pack=*/1, mlir::ValueRange{dummyTensor});
          }
        }
      }
      int argIdx = inputArgIt->second;
      auto inputPtrArg = getBufferArg(argIdx);

      // Derive pointer/element types from the ACTUAL function arg type, NOT from
      // result.args[argIdx].dtype which can disagree with the function signature
      // (e.g., when the output slot's live array has been released and dtype
      // resolution falls back to a different source).
      auto ptrType = mlir::cast<mlir::triton::PointerType>(inputPtrArg.getType());
      auto elemType = ptrType.getPointeeType();
      auto f32Type = builder.getF32Type();
      auto f32TensorType = mlir::RankedTensorType::get({blockSize}, f32Type);
      auto ptrTensorType = mlir::RankedTensorType::get({blockSize}, ptrType);

      // Segmented reduction: for each output offset i (from the block's offsets vector),
      // accumulate: acc = identity_val; for k=0..reductionSize-1: acc = combine(acc, input[inputOffset(i, k)])
      // Where inputOffset(i, k) unravels i to output ND coords, inserts k at reductionAxis, ravels to flat.

      // Determine reduction identity value and combine op
      std::string opLower = slot.opName;
      std::transform(opLower.begin(), opLower.end(), opLower.begin(), ::tolower);
      bool isMean = (opLower == "reduce_mean" || opLower == "mean");
      bool isMax = (opLower == "reduce_max" || opLower == "max");
      bool isMin = (opLower == "reduce_min" || opLower == "min");
      bool isProd = (opLower == "reduce_prod" || opLower == "prod");

      float identityVal = 0.0f;
      if (isMax) identityVal = -3.4028235e+38f;
      else if (isMin) identityVal = 3.4028235e+38f;
      else if (isProd) identityVal = 1.0f;

      mlir::Value acc = splatConstantF32(builder, loc, f32TensorType, identityVal);

      // Loop over reduction axis
      for (int k = 0; k < reductionSize; k++) {
        // Compute input flat offset for each output position with reduction index k
        // Unravel offsets (output flat idx) to output coords, map to input coords
        mlir::Value inputOffset = splatConstantI32(builder, loc, i32TensorType, 0);
        mlir::Value rem = offsets;
        int inputDimIdx = 0;
        for (int d = 0; d < inputRank; d++) {
          if (d == reductionAxis) {
            // Add k * inputStride[reductionAxis]
            auto contrib = splatConstantI32(builder, loc, i32TensorType, k * inStrides[d]);
            inputOffset = builder.create<mlir::arith::AddIOp>(loc, inputOffset, contrib);
          } else {
            // Get output coord for this dimension
            auto oStrideConst = splatConstantI32(builder, loc, i32TensorType, outStrides[inputDimIdx]);
            auto coord = builder.create<mlir::arith::DivSIOp>(loc, rem, oStrideConst);
            if (inputDimIdx < outRank - 1)
              rem = builder.create<mlir::arith::RemSIOp>(loc, rem, oStrideConst);
            // Map to input flat offset
            auto inStrideConst = splatConstantI32(builder, loc, i32TensorType, inStrides[d]);
            auto contrib = builder.create<mlir::arith::MulIOp>(loc, coord, inStrideConst);
            inputOffset = builder.create<mlir::arith::AddIOp>(loc, inputOffset, contrib);
            inputDimIdx++;
          }
        }

        // Load input at computed offsets
        auto splatPtr = builder.create<mlir::triton::SplatOp>(loc, ptrTensorType, inputPtrArg);
        auto ptrs = builder.create<mlir::triton::AddPtrOp>(loc, ptrTensorType, splatPtr, inputOffset);
        auto loaded = builder.create<mlir::triton::LoadOp>(loc,
            ptrs.getResult(), mask.getResult(), mlir::Value(),
            mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL, false);
        // Cast to f32 for accumulation
        mlir::Value val = castTo(builder, loc, loaded, f32Type);

        // Combine
        if (isMax)
          acc = builder.create<mlir::arith::MaximumFOp>(loc, acc, val);
        else if (isMin)
          acc = builder.create<mlir::arith::MinimumFOp>(loc, acc, val);
        else if (isProd)
          acc = builder.create<mlir::arith::MulFOp>(loc, acc, val);
        else // sum, mean
          acc = builder.create<mlir::arith::AddFOp>(loc, acc, val);
      }

      // For mean: divide by reduction size
      if (isMean && reductionSize > 0) {
        auto countSplat = splatConstantF32(builder, loc, f32TensorType,
            static_cast<float>(reductionSize));
        acc = builder.create<mlir::arith::DivFOp>(loc, acc, countSplat);
      }

      // Cast back to output element type
      auto outSlotIdx = slot.outputSlotIndices[0];
      auto outDtype = resolveDtypeLocal(outSlotIdx);
      auto outElemType = getMLIRType(builder, outDtype);
      mlir::Value opResult = castTo(builder, loc, acc, outElemType);

      // Ensure result is a tensor (should be, since acc was a tensor)
      if (!mlir::isa<mlir::RankedTensorType>(opResult.getType())) {
        auto splatTy = mlir::RankedTensorType::get({blockSize}, opResult.getType());
        opResult = builder.create<mlir::triton::SplatOp>(loc, splatTy, opResult);
      }

      // Broadcast expansion: only needed when downstream fused ops consume the result
      // at input-sized offsets. For standalone reduction (no downstream consumer), the
      // output-indexed result is stored directly and no broadcast is needed.
      int nInputElements = 1;
      for (auto d : inputShape) nInputElements *= static_cast<int>(d);
      int nOutputElements = 1;
      for (auto d : outShape) nOutputElements *= static_cast<int>(d);

      bool hasDownstreamConsumer = false;
      for (int si2 = si + 1; si2 <= endSlot; si2++) {
        for (int inp2 = 0; inp2 < slots[si2].numInputs; inp2++) {
          for (int o = 0; o < slot.numOutputs; o++) {
            if (slots[si2].inputSourceIndices[inp2] == slot.outputSlotIndices[o])
              hasDownstreamConsumer = true;
          }
        }
      }
      if (hasDownstreamConsumer && nInputElements > nOutputElements && nOutputElements > 0) {
        // Build mapping: for each position in [0, blockSize), compute the output index
        // that should be broadcast to that position.
        // outIdx[i] = (i / (product of dims after reductionAxis in input)) % nOutputElements
        // For axis=last: outIdx = i / reductionSize
        // For axis=first: outIdx = i % (product of remaining dims)
        // General: unravel i with input strides, skip reduction axis, ravel with output strides
        mlir::Value broadcastIdx = splatConstantI32(builder, loc, i32TensorType, 0);
        mlir::Value rem2 = offsets;
        int oDimIdx = 0;
        for (int d = 0; d < inputRank; d++) {
          auto iStrConst = splatConstantI32(builder, loc, i32TensorType, inStrides[d]);
          auto coord2 = builder.create<mlir::arith::DivSIOp>(loc, rem2, iStrConst);
          if (d < inputRank - 1)
            rem2 = builder.create<mlir::arith::RemSIOp>(loc, rem2, iStrConst);
          if (d != reductionAxis) {
            auto oStrConst = splatConstantI32(builder, loc, i32TensorType, outStrides[oDimIdx]);
            auto contrib2 = builder.create<mlir::arith::MulIOp>(loc, coord2, oStrConst);
            broadcastIdx = builder.create<mlir::arith::AddIOp>(loc, broadcastIdx, contrib2);
            oDimIdx++;
          }
        }
        // Now gather from the reduction result using broadcastIdx
        // opResult[broadcastIdx[i]] → broadcast value
        // Since opResult is stored at output positions 0..nOut-1, we need to
        // store the reduction result to a buffer, then reload with broadcast indices.
        // But we don't have a buffer. Instead, recompute: the reduction already produced
        // correct values at positions 0..nOut-1 in the tensor. We need to shuffle them.
        // Alternative: re-emit the accumulation with input-sized offsets.
        // Simplest approach: the result at position outIdx should be at position broadcastIdx.
        // We can use the broadcastIdx to re-index: for each thread, re-accumulate from scratch.
        // But that's wasteful. Better: store result to output buffer, then reload with broadcast.
        // Actually, since we're in SSA-land, the cleanest approach is to just redo the
        // reduction indexed by input offsets: for input position i, the reduced value
        // is sum(input[outIdx * reductionSize + k]) for the right k range.

        // Re-compute with input-indexed offsets
        mlir::Value broadcastAcc = splatConstantF32(builder, loc, f32TensorType, identityVal);
        for (int k = 0; k < reductionSize; k++) {
          mlir::Value inputOff = splatConstantI32(builder, loc, i32TensorType, 0);
          mlir::Value rem3 = offsets;
          int oIdx = 0;
          for (int d = 0; d < inputRank; d++) {
            if (d == reductionAxis) {
              auto contrib3 = splatConstantI32(builder, loc, i32TensorType, k * inStrides[d]);
              inputOff = builder.create<mlir::arith::AddIOp>(loc, inputOff, contrib3);
            } else {
              auto iStrConst3 = splatConstantI32(builder, loc, i32TensorType, inStrides[d]);
              auto coord3 = builder.create<mlir::arith::DivSIOp>(loc, rem3, iStrConst3);
              if (d < inputRank - 1)
                rem3 = builder.create<mlir::arith::RemSIOp>(loc, rem3, iStrConst3);
              auto contrib3 = builder.create<mlir::arith::MulIOp>(loc, coord3, iStrConst3);
              inputOff = builder.create<mlir::arith::AddIOp>(loc, inputOff, contrib3);
              oIdx++;
            }
          }
          auto splatPtr2 = builder.create<mlir::triton::SplatOp>(loc, ptrTensorType, inputPtrArg);
          auto ptrs2 = builder.create<mlir::triton::AddPtrOp>(loc, ptrTensorType, splatPtr2, inputOff);
          // Use mask based on input element count (not output)
          auto nInputConst = builder.create<mlir::arith::ConstantIntOp>(loc, nInputElements, 32);
          auto splatNInput = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, nInputConst);
          auto inputMask = builder.create<mlir::arith::CmpIOp>(
              loc, mlir::arith::CmpIPredicate::slt, offsets, splatNInput);
          auto loaded2 = builder.create<mlir::triton::LoadOp>(loc,
              ptrs2.getResult(), inputMask.getResult(), mlir::Value(),
              mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL, false);
          mlir::Value val2 = castTo(builder, loc, loaded2, f32Type);
          if (isMax)
            broadcastAcc = builder.create<mlir::arith::MaximumFOp>(loc, broadcastAcc, val2);
          else if (isMin)
            broadcastAcc = builder.create<mlir::arith::MinimumFOp>(loc, broadcastAcc, val2);
          else if (isProd)
            broadcastAcc = builder.create<mlir::arith::MulFOp>(loc, broadcastAcc, val2);
          else
            broadcastAcc = builder.create<mlir::arith::AddFOp>(loc, broadcastAcc, val2);
        }
        if (isMean && reductionSize > 0) {
          auto countSplat2 = splatConstantF32(builder, loc, f32TensorType,
              static_cast<float>(reductionSize));
          broadcastAcc = builder.create<mlir::arith::DivFOp>(loc, broadcastAcc, countSplat2);
        }
        opResult = castTo(builder, loc, broadcastAcc, outElemType);
        if (!mlir::isa<mlir::RankedTensorType>(opResult.getType())) {
          auto splatTy = mlir::RankedTensorType::get({blockSize}, opResult.getType());
          opResult = builder.create<mlir::triton::SplatOp>(loc, splatTy, opResult);
        }
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
      // Handles past_key/past_value (inputs 4-5) and BSHD (3D) vs BHSD (4D) layout.
      if (slot.numInputs >= 3 && slot.numOutputs >= 1) {
        int qSrc = slot.inputSourceIndices[0];
        int kSrc = slot.inputSourceIndices[1];
        int vSrc = slot.inputSourceIndices[2];
        int outSlot = slot.outputSlotIndices[0];

        // Check for past_key/past_value (inputs 4 and 5)
        bool hasPastKv = false;
        int pastKeySrc = -1, pastValueSrc = -1;
        if (slot.numInputs > 4) {
          pastKeySrc = slot.inputSourceIndices[4];
          NDArray* pastKeyArr = resolveArr(pastKeySrc);
          if (pastKeyArr && pastKeyArr->rankOf() == 4 && pastKeyArr->lengthOf() > 1) {
            hasPastKv = true;
          }
        }
        if (hasPastKv && slot.numInputs > 5) {
          pastValueSrc = slot.inputSourceIndices[5];
        }

        int effectiveKSrc = hasPastKv ? pastKeySrc : kSrc;
        int effectiveVSrc = (hasPastKv && pastValueSrc >= 0) ? pastValueSrc : vSrc;

        NDArray* qArr = resolveArr(qSrc);
        NDArray* effectiveKArr = resolveArr(effectiveKSrc);

        // Extract attention dimensions
        int batchSize = 1, numQHeads = 1, numKvHeads = 0, seqQ = 1, seqK = 1, headDim = 64;
        bool qIsBSHD = false;
        if (qArr && qArr->rankOf() >= 4) {
          batchSize = static_cast<int>(qArr->sizeAt(0));
          numQHeads = static_cast<int>(qArr->sizeAt(1));
          seqQ = static_cast<int>(qArr->sizeAt(2));
          headDim = static_cast<int>(qArr->sizeAt(3));
        } else if (qArr && qArr->rankOf() == 3) {
          // 3D BSHD: [batch, seq, numQHeads*headDim]
          batchSize = static_cast<int>(qArr->sizeAt(0));
          seqQ = static_cast<int>(qArr->sizeAt(1));
          int hidden = static_cast<int>(qArr->sizeAt(2));
          numQHeads = (slot.numIArgs > 0 && slot.iArgs) ? static_cast<int>(slot.iArgs[0]) : 1;
          if (numQHeads <= 0) numQHeads = 1;
          headDim = hidden / numQHeads;
          qIsBSHD = true;
        }
        // Extract KV head count from past_key (4D BHSD: [B, KvHeads, seqK, HD])
        if (hasPastKv && effectiveKArr && effectiveKArr->rankOf() == 4) {
          numKvHeads = static_cast<int>(effectiveKArr->sizeAt(1));
          headDim = static_cast<int>(effectiveKArr->sizeAt(3));
        }
        if (numKvHeads <= 0) numKvHeads = numQHeads;
        if (effectiveKArr && effectiveKArr->rankOf() >= 4) {
          seqK = static_cast<int>(effectiveKArr->sizeAt(2));
        } else if (effectiveKArr && effectiveKArr->rankOf() == 3) {
          seqK = static_cast<int>(effectiveKArr->sizeAt(1));
        }

        bool kIsBSHD = hasPastKv ? false : qIsBSHD;
        float scale = 1.0f / std::sqrt(static_cast<float>(headDim));
        auto attnTile = chooseFusedAttentionTileConfig(
            batchSize, numQHeads, seqQ, seqK, headDim);
        if (!attnTile.fitsSharedMem) {
          std::string msg = "TritonIRBuilder: fused attention '" + slot.opName + "' at slot " +
                            std::to_string(si) + " cannot fit shared memory (headDim=" +
                            std::to_string(headDim) + ", BM=" + std::to_string(attnTile.blockM) +
                            ", BN=" + std::to_string(attnTile.blockN) + ", estimated=" +
                            std::to_string(attnTile.estimatedSharedMemBytes) + ", limit=" +
                            std::to_string(attnTile.sharedMemLimitBytes) + ")";
          THROW_EXCEPTION(msg.c_str());
        }
        int blockM = attnTile.blockM;
        int blockN = attnTile.blockN;

        auto qPtr = getSlotArgPtr(qSrc);
        auto kPtr = getSlotArgPtr(effectiveKSrc);
        auto vPtr = getSlotArgPtr(effectiveVSrc);
        auto outPtr = getSlotArgPtr(outSlot);

        if (qPtr && kPtr && vPtr && outPtr) {
          emitFusedAttentionKernel(builder, loc, qPtr, kPtr, vPtr, outPtr,
                                   batchSize, numQHeads, numKvHeads, seqQ, seqK, headDim,
                                   scale, blockM, blockN, qIsBSHD, kIsBSHD,
                                   mlir::Value(), std::vector<LongType>());

          // output[0] = attention result
          DataType outDtype = FLOAT32;
          NDArray* outArr = resolveArr(outSlot);
          if (outArr) outDtype = outArr->dataType();
          auto loaded = loadBackFromBuffer(outSlot, outDtype);
          if (loaded) ssaValues[outSlot] = loaded;

          // output[1] = present_key (pass-through effective key)
          if (slot.numOutputs >= 2) {
            if (ssaValues.count(effectiveKSrc)) {
              ssaValues[slot.outputSlotIndices[1]] = ssaValues[effectiveKSrc];
            } else {
              DataType kDtype = FLOAT32;
              NDArray* kArr2 = resolveArr(effectiveKSrc);
              if (kArr2) kDtype = kArr2->dataType();
              auto kLoaded = loadBackFromBuffer(effectiveKSrc, kDtype);
              if (kLoaded) ssaValues[slot.outputSlotIndices[1]] = kLoaded;
            }
          }
          // output[2] = present_value (pass-through effective value)
          if (slot.numOutputs >= 3) {
            if (ssaValues.count(effectiveVSrc)) {
              ssaValues[slot.outputSlotIndices[2]] = ssaValues[effectiveVSrc];
            } else {
              DataType vDtype = FLOAT32;
              NDArray* vArr2 = resolveArr(effectiveVSrc);
              if (vArr2) vDtype = vArr2->dataType();
              auto vLoaded = loadBackFromBuffer(effectiveVSrc, vDtype);
              if (vLoaded) ssaValues[slot.outputSlotIndices[2]] = vLoaded;
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

          // Get permutation from iArgs; fall back to reverse if not provided
          std::vector<int> permutation;
          if (slot.numIArgs > 0 && slot.iArgs) {
            for (int d = 0; d < slot.numIArgs; d++)
              permutation.push_back(static_cast<int>(slot.iArgs[d]));
          }
          if (permutation.empty()) {
            for (int d = static_cast<int>(inputShape.size()) - 1; d >= 0; d--)
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
          int axis = (slot.numIArgs > 0 && slot.iArgs) ? static_cast<int>(slot.iArgs[0]) : 0;

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
  result.useDynamicGrid = false;
  result.requiredGrid = static_cast<int>(
      std::min<LongType>(static_cast<LongType>(result.gridX) * result.gridY,
                         static_cast<LongType>(2147483647)));

  // Estimate shared memory for basic module (elementwise + matmul fusion).
  // This module never uses cooperative launch, but set the estimate for consistency.
  {
    bool hasMatmulCat = false;
    bool hasReductionCat = false;
    bool hasNormCat = false;
    for (auto cat : categories) {
      if (cat == TritonOpCategory::MATMUL) hasMatmulCat = true;
      if (cat == TritonOpCategory::REDUCTION) hasReductionCat = true;
      if (cat == TritonOpCategory::NORMALIZATION) hasNormCat = true;
    }
    if (hasMatmulCat) {
      // Basic matmul fusion: BLOCK_SIZE^2 * elemSize * numStages (simplified)
      result.estimatedSharedMemBytes = blockSize * blockSize * 2 * numStages;
    } else if (hasNormCat) {
      result.estimatedSharedMemBytes = blockSize * 4 * 2;
    } else if (hasReductionCat) {
      result.estimatedSharedMemBytes = blockSize * 4;
    } else {
      result.estimatedSharedMemBytes = 0;
    }
  }

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
    // Write TTIR to file for all kernels
    {
      const char* fname = useIndirectArgs ? "/tmp/triton_ttir_indirect.mlir" : "/tmp/triton_ttir_direct.mlir";
      FILE* df = fopen(fname, "w");
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

  // Pre-compute which section boundaries truly require a grid-wide barrier.
  // Many sections share the same 1D pid mapping and can stream values block-local
  // without cooperative synchronization.
  std::unordered_map<int, int> producerSectionByOutput;
  std::vector<LongType> sectionMaxOutputElements(sections.size(), 0);
  auto computeSectionMaxOutputElements = [&](const KernelSection& sec) -> LongType {
    LongType maxElements = 0;
    for (int si = sec.startSlot; si <= sec.endSlot; si++) {
      for (int o = 0; o < slots[si].numOutputs; o++) {
        int outIdx = slots[si].outputSlotIndices[o];
        auto outShape = resolveShape(outIdx);
        LongType elems = shapeLength(outShape);
        if (elems > maxElements) maxElements = elems;
      }
    }
    return maxElements;
  };

  for (size_t secIdx = 0; secIdx < sections.size(); secIdx++) {
    sectionMaxOutputElements[secIdx] = computeSectionMaxOutputElements(sections[secIdx]);
    for (int si = sections[secIdx].startSlot; si <= sections[secIdx].endSlot; si++) {
      for (int o = 0; o < slots[si].numOutputs; o++) {
        int outIdx = slots[si].outputSlotIndices[o];
        if (internalSlotOutputs.count(outIdx)) {
          producerSectionByOutput[outIdx] = static_cast<int>(secIdx);
        }
      }
    }
  }

  auto sectionNeedsGlobalBarrier = [](KernelSectionType type) -> bool {
    switch (type) {
      case KernelSectionType::FUSED_ATTENTION:
      case KernelSectionType::REDUCTION:
      case KernelSectionType::NORMALIZATION:
      case KernelSectionType::SCATTER_ND:
      case KernelSectionType::SCATTER_ND_UPDATE:
      case KernelSectionType::SHAPE_MANIPULATION:
        // SHAPE_MANIPULATION (permute/transpose) reads cross-section intermediates
        // with permuted indices — thread N reads data written by thread M, so a
        // global barrier is required to ensure all stores complete before permuted loads.
        return true;
      default:
        return false;
    }
  };

  std::vector<uint8_t> sectionNeedsBarrier(sections.size(), 0);
  for (size_t secIdx = 1; secIdx < sections.size(); secIdx++) {
    bool needsBarrier = false;
    const auto& consumerSection = sections[secIdx];
    for (int si = consumerSection.startSlot; si <= consumerSection.endSlot && !needsBarrier; si++) {
      for (int inp = 0; inp < slots[si].numInputs; inp++) {
        int srcIdx = slots[si].inputSourceIndices[inp];
        if (srcIdx < 0 || !crossSectionIntermediates.count(srcIdx)) continue;

        auto producerIt = producerSectionByOutput.find(srcIdx);
        if (producerIt == producerSectionByOutput.end()) continue;

        int producerSectionIdx = producerIt->second;
        if (producerSectionIdx == static_cast<int>(secIdx)) continue;
        if (producerSectionIdx < 0 || producerSectionIdx >= static_cast<int>(sections.size())) continue;

        const auto& producerSection = sections[producerSectionIdx];
        if (sectionNeedsGlobalBarrier(producerSection.type) ||
            sectionNeedsGlobalBarrier(consumerSection.type)) {
          needsBarrier = true;
          break;
        }

        LongType producedElements = shapeLength(resolveShape(srcIdx));
        LongType consumerElements = sectionMaxOutputElements[secIdx];
        if (producedElements <= 0 || consumerElements <= 0 || producedElements != consumerElements) {
          needsBarrier = true;
          break;
        }
      }
    }

    if (needsBarrier) {
      sectionNeedsBarrier[secIdx] = 1;
    }
  }

  bool needsGridSync = std::any_of(sectionNeedsBarrier.begin(), sectionNeedsBarrier.end(),
                                   [](uint8_t v) { return v != 0; });

  // When cooperative launch is disabled (default) and cross-section barriers
  // are needed, use multi-phase launch: the kernel gets a phase_id argument,
  // and the host launches the kernel once per phase. Each phase is a maximal
  // group of consecutive sections that don't need cross-block sync. The kernel
  // launch itself provides implicit global synchronization between phases.
  // This allows arbitrary grid sizes and each phase uses its optimal grid size.
  auto& envRef = sd::Environment::getInstance();
  bool useMultiPhaseLaunch = false;
  std::vector<TritonIRModule::LaunchPhase> launchPhases;

  if (needsGridSync && !envRef.tritonCooperativeLaunch()) {
    int numBarriers = static_cast<int>(std::count(sectionNeedsBarrier.begin(),
                                                   sectionNeedsBarrier.end(), 1));
    // Build phases: group consecutive sections between barriers
    int phaseStart = 0;
    for (size_t secIdx = 1; secIdx <= sections.size(); secIdx++) {
      if (secIdx == sections.size() || sectionNeedsBarrier[secIdx]) {
        // End current phase at secIdx-1
        TritonIRModule::LaunchPhase phase;
        phase.startSection = phaseStart;
        phase.endSection = static_cast<int>(secIdx) - 1;
        // Grid size for this phase = max grid across contained sections
        int phaseGrid = 1;
        for (int s = phase.startSection; s <= phase.endSection; s++) {
          if (sections[s].gridRequirement > phaseGrid)
            phaseGrid = sections[s].gridRequirement;
        }
        phase.gridX = phaseGrid;
        launchPhases.push_back(phase);
        phaseStart = static_cast<int>(secIdx);
      }
    }
    useMultiPhaseLaunch = true;
    needsGridSync = false;  // No in-kernel barriers needed

    sd_printf("TritonIRBuilder::buildSectionedModule: cooperative launch disabled; "
              "using multi-phase launch with %d phases (%d barriers) for [%d-%d]\n",
              static_cast<int>(launchPhases.size()), numBarriers, startSlot, endSlot);
  }

  int requiredBarriers = 0;
  for (auto v : sectionNeedsBarrier) {
    if (v != 0) requiredBarriers++;
  }
  sd_debug("TritonIRBuilder::buildSectionedModule: %d/%d section boundaries require barriers "
            "(gridSync=%d, multiPhase=%d)\n",
            requiredBarriers, std::max(0, static_cast<int>(sections.size()) - 1),
            needsGridSync ? 1 : 0, useMultiPhaseLaunch ? 1 : 0);

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

  // NOTE: K/V projection external output forcing removed — attention ops now run via
  // cuBLAS fallback (isFallbackSection) and handle their own present_key/present_value outputs.

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
  // Extra scalar args: n_elements (always) + sync_counter_ptr (cooperative) + phase_id (multi-phase)
  int extraScalarArgs = 1;  // n_elements
  if (needsGridSync) extraScalarArgs++;  // sync_counter_ptr
  if (useMultiPhaseLaunch) extraScalarArgs++;  // phase_id
  bool useIndirectArgs = (totalBufferArgs + extraScalarArgs) > TRITON_DIRECT_ARG_LIMIT;

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
  // Sync counter pointer for section boundaries that require grid sync.
  if (needsGridSync) {
    funcArgTypes.push_back(mlir::triton::PointerType::get(i32Type, 1));  // sync_counter_ptr
  }
  // Phase ID for multi-phase launch (controls which sections execute)
  if (useMultiPhaseLaunch) {
    funcArgTypes.push_back(i32Type);  // phase_id
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
  int nextArgIdx = nElementsArgIdx + 1;
  mlir::Value syncCounterPtr;
  if (needsGridSync) {
    syncCounterPtr = entryBlock->getArgument(nextArgIdx++);
  }
  mlir::Value phaseIdArg;
  if (useMultiPhaseLaunch) {
    phaseIdArg = entryBlock->getArgument(nextArgIdx++);
  }

  // ── Step 4: Derive tile config and recompute section launch grid ──
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
  if (sectionedBlockSizeOverride_ > 0) {
    if (blockSize != sectionedBlockSizeOverride_) {
      sd_debug("TritonIRBuilder::buildSectionedModule: overriding block size %d -> %d\n",
               blockSize, sectionedBlockSizeOverride_);
    }
    blockSize = sectionedBlockSizeOverride_;
  }
  const int attentionSharedMemLimitBytes = queryCudaSharedMemLimitBytes();

  auto sectionMaxElements = [&](const KernelSection& sec) -> LongType {
    LongType maxElements = 0;
    for (int si = sec.startSlot; si <= sec.endSlot; si++) {
      for (int o = 0; o < slots[si].numOutputs; o++) {
        int outIdx = slots[si].outputSlotIndices[o];
        auto outShape = resolveShape(outIdx);
        LongType elems = shapeLength(outShape);
        if (elems > maxElements) maxElements = elems;
      }
    }
    if (maxElements <= 0) {
      // Fallback for shape-only/meta ops: derive from consumed inputs.
      for (int si = sec.startSlot; si <= sec.endSlot; si++) {
        for (int inp = 0; inp < slots[si].numInputs; inp++) {
          int srcIdx = slots[si].inputSourceIndices[inp];
          auto inShape = resolveShape(srcIdx);
          LongType elems = shapeLength(inShape);
          if (elems > maxElements) maxElements = elems;
        }
      }
    }
    return maxElements;
  };

  auto deriveAttentionGrid = [&](const KernelSection& sec) -> std::pair<int, int> {
    int batchSize = std::max(1, sec.batchSize);
    int numHeads = std::max(1, sec.numHeads);
    int seqQ = std::max(1, sec.seqQ);
    int seqK = std::max(1, sec.seqK);
    int headDim = std::max(1, sec.headDim);

    // Recover dimensions from runtime shapes when section metadata is incomplete.
    if (sec.batchSize <= 0 || sec.numHeads <= 0 || sec.seqQ <= 0 || sec.headDim <= 0) {
      for (int si = sec.startSlot; si <= sec.endSlot; si++) {
        auto& slot = slots[si];
        if (getOpCategory(slot.opName) != TritonOpCategory::FUSED_ATTENTION ||
            slot.numInputs < 1) {
          continue;
        }
        auto qShape = resolveShape(slot.inputSourceIndices[0]);
        if (qShape.size() >= 4) {
          batchSize = static_cast<int>(std::max<LongType>(1, qShape[0]));
          numHeads = static_cast<int>(std::max<LongType>(1, qShape[1]));
          seqQ = static_cast<int>(std::max<LongType>(1, qShape[2]));
          headDim = static_cast<int>(std::max<LongType>(1, qShape[3]));
          if (slot.numInputs >= 2) {
            auto kShape = resolveShape(slot.inputSourceIndices[1]);
            if (kShape.size() >= 3) {
              seqK = static_cast<int>(std::max<LongType>(1, kShape[2]));
            }
          }
        } else if (qShape.size() == 3) {
          batchSize = static_cast<int>(std::max<LongType>(1, qShape[0]));
          numHeads = 1;
          seqQ = static_cast<int>(std::max<LongType>(1, qShape[1]));
          headDim = static_cast<int>(std::max<LongType>(1, qShape[2]));
          if (slot.numInputs >= 2) {
            auto kShape = resolveShape(slot.inputSourceIndices[1]);
            if (kShape.size() >= 2) {
              seqK = static_cast<int>(std::max<LongType>(1, kShape[1]));
            }
          }
        }
        break;
      }
    }

    auto attnTile = chooseFusedAttentionTileConfig(
        batchSize, numHeads, seqQ, seqK, headDim, attentionSharedMemLimitBytes);
    int blockMForAttn = std::max(1, attnTile.blockM);

    int gridX = std::max(1, batchSize * numHeads);
    int gridY = std::max(1, (seqQ + blockMForAttn - 1) / blockMForAttn);
    return {gridX, gridY};
  };

  auto computeSectionBlocks = [&](const KernelSection& sec) -> int {
    if (sec.type == KernelSectionType::FUSED_ATTENTION) {
      auto attnGrid = deriveAttentionGrid(sec);
      LongType blocks64 = static_cast<LongType>(attnGrid.first) * attnGrid.second;
      if (blocks64 > static_cast<LongType>(2147483647)) blocks64 = static_cast<LongType>(2147483647);
      return static_cast<int>(std::max<LongType>(1, blocks64));
    }

    LongType maxElements = sectionMaxElements(sec);
    if (maxElements <= 0) {
      return std::max(1, sec.gridRequirement);
    }

    LongType blocks64 = (maxElements + blockSize - 1) / blockSize;
    if (blocks64 > static_cast<LongType>(2147483647)) blocks64 = static_cast<LongType>(2147483647);
    return static_cast<int>(std::max<LongType>(1, blocks64));
  };

  auto recomputeSectionGridRequirements = [&]() -> int {
    int maxGrid = 1;
    for (auto& sec : sections) {
      sec.gridRequirement = computeSectionBlocks(sec);
      if (sec.gridRequirement > maxGrid) maxGrid = sec.gridRequirement;
    }
    return maxGrid;
  };

  int maxSectionGrid = recomputeSectionGridRequirements();
  if (needsGridSync) {
    const int coopTargetBlocks = std::max(1, getSectionedCooperativeTargetBlocks());
    const int initialBlockSize = blockSize;
    while (maxSectionGrid > coopTargetBlocks && blockSize < 16384) {
      blockSize <<= 1;
      maxSectionGrid = recomputeSectionGridRequirements();
    }
    if (blockSize != initialBlockSize) {
      sd_printf("TritonIRBuilder::buildSectionedModule: tuned cooperative block size %d -> %d "
                "(targetBlocks=%d, resultingGrid=%d)\n",
                initialBlockSize, blockSize, coopTargetBlocks, maxSectionGrid);
    }
  }

  unsigned int fixedGridX = static_cast<unsigned int>(std::max(1, maxSectionGrid));
  unsigned int fixedGridY = 1;
  unsigned int fixedGridZ = 1;
  if (sections.size() == 1 && sections[0].type == KernelSectionType::FUSED_ATTENTION) {
    auto attnGrid = deriveAttentionGrid(sections[0]);
    fixedGridX = static_cast<unsigned int>(std::max(1, attnGrid.first));
    fixedGridY = static_cast<unsigned int>(std::max(1, attnGrid.second));
    LongType totalBlocks = static_cast<LongType>(fixedGridX) * fixedGridY;
    if (totalBlocks > maxSectionGrid) {
      maxSectionGrid = static_cast<int>(std::min<LongType>(totalBlocks, static_cast<LongType>(2147483647)));
    }
  }

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

    // Use per-slot element count when available.
    // A global n_elements can be larger than this slot (e.g., concat/split chains),
    // which would allow out-of-bounds loads while populating section SSA values.
    mlir::Value slotNValue = nElementsArg;
    auto slotShape = resolveShape(slotIdx);
    LongType slotElements = shapeLength(slotShape);
    if (slotElements > 0) {
      if (slotElements > static_cast<LongType>(2147483647)) {
        slotElements = static_cast<LongType>(2147483647);
      }
      slotNValue = builder.create<mlir::arith::ConstantIntOp>(
          loc, static_cast<int>(slotElements), 32);
    }

    auto splatN = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, slotNValue);
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

  // Build section-to-phase mapping for multi-phase launch
  std::vector<int> sectionPhase(sections.size(), 0);  // phase index per section
  if (useMultiPhaseLaunch) {
    for (size_t p = 0; p < launchPhases.size(); p++) {
      for (int s = launchPhases[p].startSection; s <= launchPhases[p].endSection; s++) {
        sectionPhase[s] = static_cast<int>(p);
      }
    }
  }

  for (size_t secIdx = 0; secIdx < sections.size(); secIdx++) {
    auto& sec = sections[secIdx];

    sd_debug("TritonIRBuilder::buildSectionedModule: emitting section %d/%d type=%d slots[%d-%d]\n",
              static_cast<int>(secIdx), static_cast<int>(sections.size()),
              static_cast<int>(sec.type), sec.startSlot, sec.endSlot);

    // Before each section (except the first), insert a cooperative grid sync
    // barrier if needed. Multi-phase launch doesn't need in-kernel barriers
    // (kernel launch provides implicit global sync between phases).
    if (secIdx > 0 && needsGridSync && sectionNeedsBarrier[secIdx]) {
      LongType threshold64 =
          static_cast<LongType>(sectionBarrierCount + 1) * static_cast<LongType>(maxSectionGrid);
      if (threshold64 > static_cast<LongType>(2147483647)) {
        threshold64 = static_cast<LongType>(2147483647);
      }
      auto numBlocksVal = builder.create<mlir::arith::ConstantIntOp>(
          loc, static_cast<int>(threshold64), 32);
      emitGridSync(builder, loc, syncCounterPtr, numBlocksVal);
      sectionBarrierCount++;
      sd_debug("TritonIRBuilder::buildSectionedModule: inserted grid sync barrier before section %d\n",
                static_cast<int>(secIdx));
    }

    // For multi-phase launch: guard each section by its phase_id.
    // Sections only execute when the host-supplied phase_id matches their phase.
    mlir::scf::IfOp phaseIf;
    if (useMultiPhaseLaunch) {
      auto phaseConst = builder.create<mlir::arith::ConstantIntOp>(
          loc, sectionPhase[secIdx], 32);
      auto phaseMatch = builder.create<mlir::arith::CmpIOp>(
          loc, mlir::arith::CmpIPredicate::eq, phaseIdArg, phaseConst);
      phaseIf = builder.create<mlir::scf::IfOp>(loc, phaseMatch, /*withElseRegion=*/false);
      builder.setInsertionPointToStart(&phaseIf.getThenRegion().front());
    }

    // Guard each section by its own grid requirement. Blocks outside this
    // section's range must no-op.
    auto secGridConst = builder.create<mlir::arith::ConstantIntOp>(
        loc, std::max(1, sec.gridRequirement), 32);
    auto secActive = builder.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::slt, pid, secGridConst);
    auto secIf = builder.create<mlir::scf::IfOp>(loc, secActive, /*withElseRegion=*/false);
    builder.setInsertionPointToStart(&secIf.getThenRegion().front());

    // Section bodies are emitted in distinct scf.if regions. Values from one
    // section region do not dominate sibling section regions, so keep this map
    // section-local and force cross-section values through explicit buffers.
    ssaValues.clear();

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

        // Load inputs that aren't already in SSA map, with broadcast indexing
        // Compute max output elements for this section (for broadcast detection)
        LongType secMaxOutputElements = 0;
        for (int si = sec.startSlot; si <= sec.endSlot; si++) {
          for (int o = 0; o < slots[si].numOutputs; o++) {
            int outIdx = slots[si].outputSlotIndices[o];
            auto outShape = resolveShape(outIdx);
            LongType oElems = 1;
            for (auto d : outShape) oElems *= d;
            if (oElems > secMaxOutputElements) secMaxOutputElements = oElems;
          }
        }
        // Fallback: if output shapes unavailable, use max input elements
        if (secMaxOutputElements <= 1) {
          for (int si = sec.startSlot; si <= sec.endSlot; si++) {
            for (int inp = 0; inp < slots[si].numInputs; inp++) {
              int srcIdx = slots[si].inputSourceIndices[inp];
              auto argIt = slotToArgIdx.find(srcIdx);
              if (argIt == slotToArgIdx.end()) continue;
              auto& argDesc = result.args[argIt->second];
              LongType iElems = 1;
              for (auto d : argDesc.shape) iElems *= d;
              if (iElems > secMaxOutputElements) secMaxOutputElements = iElems;
            }
          }
        }

        mlir::Value sectionNValue = nElementsArg;
        if (secMaxOutputElements > 0) {
          if (secMaxOutputElements > static_cast<LongType>(2147483647)) {
            secMaxOutputElements = static_cast<LongType>(2147483647);
          }
          sectionNValue = builder.create<mlir::arith::ConstantIntOp>(
              loc, static_cast<int>(secMaxOutputElements), 32);
        }
        auto splatN = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, sectionNValue);
        auto mask = builder.create<mlir::arith::CmpIOp>(
            loc, mlir::arith::CmpIPredicate::slt, offsets, splatN);

        for (int si = sec.startSlot; si <= sec.endSlot; si++) {
          for (int inp = 0; inp < slots[si].numInputs; inp++) {
            int srcIdx = slots[si].inputSourceIndices[inp];
            if (ssaValues.count(srcIdx)) continue;
            auto argIt = slotToArgIdx.find(srcIdx);
            if (argIt == slotToArgIdx.end()) continue;
            auto funcArg = getBufferArg(argIt->second);
            auto& argDesc = result.args[argIt->second];
            // Derive pointer type from actual function arg (avoids dtype mismatch)
            auto ptrType = mlir::cast<mlir::triton::PointerType>(funcArg.getType());
            auto elemType = ptrType.getPointeeType();
            auto ptrTensorType = mlir::RankedTensorType::get({blockSize}, ptrType);

            // Broadcast indexing: if input is smaller than max output, use modular offsets
            LongType inputElements = 1;
            for (auto d : argDesc.shape) inputElements *= d;
            mlir::Value loadOffsets = offsets;
            if (inputElements > 0 && inputElements < secMaxOutputElements) {
              auto inputSizeConst = builder.create<mlir::arith::ConstantIntOp>(
                  loc, static_cast<int>(inputElements), 32);
              auto splatInputSize = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, inputSizeConst);
              loadOffsets = builder.create<mlir::arith::RemUIOp>(loc, offsets, splatInputSize);
            }

            auto splatPtr = builder.create<mlir::triton::SplatOp>(loc, ptrTensorType, funcArg);
            auto ptrs = builder.create<mlir::triton::AddPtrOp>(loc, ptrTensorType, splatPtr, loadOffsets);
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
            // Shape ops (reshape, permute, etc.): SSA forwarding.
            // For the autoregressive decode step (seq=1), permute [0,2,1,3] on
            // [1,1,heads,dim] is an identity (dim 1 and 2 are both 1), so SSA
            // forwarding is correct. For seq > 1, this would need actual reordering.
            // Non-permute shape ops (reshape, squeeze, expand_dims): always SSA-forward.
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

            auto funcArg = getBufferArg(argIt->second);
            // Derive pointer type from actual function arg (avoids dtype mismatch)
            auto ptrType = mlir::cast<mlir::triton::PointerType>(funcArg.getType());
            auto elemType = ptrType.getPointeeType();
            auto ptrTensorType = mlir::RankedTensorType::get({blockSize}, ptrType);
            auto splatPtr = builder.create<mlir::triton::SplatOp>(loc, ptrTensorType, funcArg);
            auto ptrs = builder.create<mlir::triton::AddPtrOp>(loc, ptrTensorType, splatPtr, offsets);
            mlir::Value storeVal = castTo(builder, loc, ssaIt->second, elemType);
            mlir::Value outMask = mask;
            auto outShape = resolveShape(outIdx);
            LongType outElements = shapeLength(outShape);
            if (outElements > 0) {
              if (outElements > static_cast<LongType>(2147483647)) {
                outElements = static_cast<LongType>(2147483647);
              }
              auto outN = builder.create<mlir::arith::ConstantIntOp>(
                  loc, static_cast<int>(outElements), 32);
              auto splatOutN = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, outN);
              outMask = builder.create<mlir::arith::CmpIOp>(
                  loc, mlir::arith::CmpIPredicate::slt, offsets, splatOutN);
            }
            builder.create<mlir::triton::StoreOp>(loc, ptrs, storeVal, outMask,
                                                   mlir::triton::CacheModifier::NONE,
                                                   mlir::triton::EvictionPolicy::NORMAL);
          }
        }

        // ── Trailing permute fusion: transposed store ──
        // If this section absorbed a trailing permute, store the permute's input SSA
        // value to the permute's OUTPUT buffer using permuted offsets.
        if (sec.hasTrailingPermute && sec.trailingPermuteInputSlotIdx >= 0) {
          auto ssaIt = ssaValues.find(sec.trailingPermuteInputSlotIdx);
          auto outArgIt = slotToArgIdx.find(sec.trailingPermuteOutputSlotIdx);
          if (ssaIt != ssaValues.end() && outArgIt != slotToArgIdx.end()) {
            auto& perm = sec.trailingPermutation;
            auto& inShape = sec.trailingPermuteInputShape;
            auto& outShape = sec.trailingPermuteOutputShape;
            int rank = static_cast<int>(inShape.size());
            int nElements = 1;
            for (auto d : outShape) nElements *= static_cast<int>(d);

            // Compute output strides (row-major)
            std::vector<int> outStrides(rank, 1);
            for (int d = rank - 2; d >= 0; d--)
              outStrides[d] = outStrides[d + 1] * static_cast<int>(outShape[d + 1]);

            // Compute input strides (row-major)
            std::vector<int> inStrides(rank, 1);
            for (int d = rank - 2; d >= 0; d--)
              inStrides[d] = inStrides[d + 1] * static_cast<int>(inShape[d + 1]);

            // Compute permuted store offsets: for each input flat index (offsets),
            // unravel to input coords, apply forward permutation, ravel with output strides.
            // input[d0,d1,...] → output[d_perm_inv[0], d_perm_inv[1], ...] = input[d0,d1,...]
            // We're scattering: for input flat index, compute output flat index.
            mlir::Value dstOffsets = splatConstantI32(builder, loc, i32TensorType, 0);
            mlir::Value remaining = offsets;
            for (int d = 0; d < rank; d++) {
              auto strideConst = splatConstantI32(builder, loc, i32TensorType, inStrides[d]);
              auto coord = builder.create<mlir::arith::DivSIOp>(loc, remaining, strideConst);
              if (d < rank - 1) {
                remaining = builder.create<mlir::arith::RemSIOp>(loc, remaining, strideConst);
              }
              // coord is the d-th coordinate in input space
              // In output space, this coord appears at position perm[d]
              auto outStrideConst = splatConstantI32(builder, loc, i32TensorType, outStrides[perm[d]]);
              auto contrib = builder.create<mlir::arith::MulIOp>(loc, coord, outStrideConst);
              dstOffsets = builder.create<mlir::arith::AddIOp>(loc, dstOffsets, contrib);
            }

            // Store to the permute's output buffer using permuted offsets
            auto outFuncArg = getBufferArg(outArgIt->second);
            DataType dt = resolveDtype(sec.trailingPermuteOutputSlotIdx);
            auto elemType = getMLIRType(builder, dt);
            auto ptrType = mlir::triton::PointerType::get(elemType, 1);
            auto ptrTensorType = mlir::RankedTensorType::get({blockSize}, ptrType);
            auto splatPtr = builder.create<mlir::triton::SplatOp>(loc, ptrTensorType, outFuncArg);
            auto ptrs = builder.create<mlir::triton::AddPtrOp>(loc, ptrTensorType, splatPtr, dstOffsets);
            mlir::Value storeVal = castTo(builder, loc, ssaIt->second, elemType);

            // Mask: only store for valid input indices
            auto nElemConst = builder.create<mlir::arith::ConstantIntOp>(loc, nElements, 32);
            auto splatN2 = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, nElemConst);
            auto permMask = builder.create<mlir::arith::CmpIOp>(
                loc, mlir::arith::CmpIPredicate::slt, offsets, splatN2);
            builder.create<mlir::triton::StoreOp>(loc, ptrs, storeVal, permMask,
                                                   mlir::triton::CacheModifier::NONE,
                                                   mlir::triton::EvictionPolicy::NORMAL);
            sd_debug("TritonIRBuilder::buildSectionedModule: emitted transposed store for "
                      "trailing permute (input slot %d -> output slot %d, nElements=%d)\n",
                      sec.trailingPermuteInputSlotIdx, sec.trailingPermuteOutputSlotIdx, nElements);
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
        // Handles past_key/past_value (inputs 4-5) and BSHD (3D) vs BHSD (4D) layout.
        bool loggedAttnTileAdjust = false;
        for (int si = sec.startSlot; si <= sec.endSlot; si++) {
          auto& slot = slots[si];
          if (getOpCategory(slot.opName) != TritonOpCategory::FUSED_ATTENTION) continue;
          if (slot.numInputs < 3 || slot.numOutputs < 1) continue;

          int qSrc = slot.inputSourceIndices[0];
          int kSrc = slot.inputSourceIndices[1];
          int vSrc = slot.inputSourceIndices[2];
          int outSlot = slot.outputSlotIndices[0];

          // Check for past_key/past_value (inputs 4 and 5)
          bool hasPastKv = false;
          int pastKeySrc = -1, pastValueSrc = -1;
          if (slot.numInputs > 4) {
            pastKeySrc = slot.inputSourceIndices[4];
            auto pastKeyShape = resolveShape(pastKeySrc);
            if (pastKeyShape.size() == 4 && shapeLength(pastKeyShape) > 1) {
              hasPastKv = true;
            }
          }
          if (hasPastKv && slot.numInputs > 5) {
            pastValueSrc = slot.inputSourceIndices[5];
          }

          // Use past_key as effective K source when available (has full KV cache positions)
          int effectiveKSrc = hasPastKv ? pastKeySrc : kSrc;
          int effectiveVSrc = (hasPastKv && pastValueSrc >= 0) ? pastValueSrc : vSrc;

          auto qShape = resolveShape(qSrc);
          auto effectiveKShape = resolveShape(effectiveKSrc);
          int batchSize = 1, numQHeads = 1, numKvHeads = 0, seqQ = 1, seqK = 1, headDim = 1;
          bool isBSHD = false;

          if (qShape.size() >= 4) {
            // 4D BHSD: [batch, numQHeads, seqQ, headDim]
            batchSize = static_cast<int>(qShape[0]);
            numQHeads = static_cast<int>(qShape[1]);
            seqQ = static_cast<int>(qShape[2]);
            headDim = static_cast<int>(qShape[3]);
          } else if (qShape.size() == 3) {
            // 3D BSHD: [batch, seqQ, numQHeads*headDim]
            batchSize = static_cast<int>(qShape[0]);
            seqQ = static_cast<int>(qShape[1]);
            int hidden = static_cast<int>(qShape[2]);
            numQHeads = (slot.numIArgs > 0 && slot.iArgs) ? static_cast<int>(slot.iArgs[0]) : 1;
            if (numQHeads <= 0) numQHeads = 1;
            headDim = hidden / numQHeads;
            isBSHD = true;
          }
          // Extract KV head count from past_key shape (4D BHSD: [B,KvHeads,seqK,HD])
          if (hasPastKv && effectiveKShape.size() == 4) {
            numKvHeads = static_cast<int>(effectiveKShape[1]);
            headDim = static_cast<int>(effectiveKShape[3]);
          }
          // Default: MHA (KV heads = Q heads)
          if (numKvHeads <= 0) numKvHeads = numQHeads;
          // seqK from effective K source
          if (effectiveKShape.size() >= 4) seqK = static_cast<int>(effectiveKShape[2]);
          else if (effectiveKShape.size() == 3) seqK = static_cast<int>(effectiveKShape[1]);

          // past_key is always 4D BHSD; current key follows Q layout
          bool kIsBSHD = hasPastKv ? false : isBSHD;

          float scale = 1.0f / std::sqrt(static_cast<float>(std::max(headDim, 1)));
          auto attnTile = chooseFusedAttentionTileConfig(
              batchSize, numQHeads, seqQ, seqK, headDim, attentionSharedMemLimitBytes);
          if (!attnTile.fitsSharedMem) {
            std::string msg = "TritonIRBuilder::buildSectionedModule: attention at slot " +
                              std::to_string(si) + " cannot fit shared memory (headDim=" +
                              std::to_string(headDim) + ", BM=" + std::to_string(attnTile.blockM) +
                              ", BN=" + std::to_string(attnTile.blockN) + ", estimated=" +
                              std::to_string(attnTile.estimatedSharedMemBytes) + ", limit=" +
                              std::to_string(attnTile.sharedMemLimitBytes) + ")";
            THROW_EXCEPTION(msg.c_str());
          }
          int blockM = attnTile.blockM;
          int blockN = attnTile.blockN;
          if (attnTile.adjustedForSharedMem && !loggedAttnTileAdjust) {
            sd_printf("TritonIRBuilder::buildSectionedModule: adjusted attention tiles for section [%d-%d] "
                      "to BM=%d BN=%d (headDim=%d, seqQ=%d, seqK=%d, estimatedSmem=%d, limit=%d) "
                      "(hasPastKv=%d, numQHeads=%d, numKvHeads=%d, isBSHD=%d)\n",
                      sec.startSlot, sec.endSlot,
                      blockM, blockN, headDim, seqQ, seqK,
                      attnTile.estimatedSharedMemBytes, attnTile.sharedMemLimitBytes,
                      hasPastKv ? 1 : 0, numQHeads, numKvHeads, isBSHD ? 1 : 0);
            loggedAttnTileAdjust = true;
          }

          auto qPtr = getSlotArgPtr(qSrc);
          auto kPtr = getSlotArgPtr(effectiveKSrc);
          auto vPtr = getSlotArgPtr(effectiveVSrc);
          auto outPtr = getSlotArgPtr(outSlot);

          // Extract attention bias/mask from input[3] if available and non-scalar
          mlir::Value attnBiasPtr;
          std::vector<LongType> attnBiasShape;
          if (slot.numInputs > 3) {
            int biasSrc = slot.inputSourceIndices[3];
            auto bShape = resolveShape(biasSrc);
            // Only use bias if it's a real tensor (rank >= 3, length > 1)
            if (bShape.size() >= 3 && shapeLength(bShape) > 1) {
              attnBiasPtr = getSlotArgPtr(biasSrc);
              attnBiasShape = bShape;
              if (attnBiasPtr) {
                sd_printf("TritonIRBuilder: attention bias at slot %d: shape=[", si);
                for (size_t d = 0; d < bShape.size(); d++) {
                  if (d) sd_printf(",");
                  sd_printf("%lld", bShape[d]);
                }
                sd_printf("] — will apply to QK scores\n");
              }
            }
          }

          if (qPtr && kPtr && vPtr && outPtr) {
            emitFusedAttentionKernel(builder, loc, qPtr, kPtr, vPtr, outPtr,
                                     batchSize, numQHeads, numKvHeads, seqQ, seqK, headDim,
                                     scale, blockM, blockN, isBSHD, kIsBSHD,
                                     attnBiasPtr, attnBiasShape);
            // output[0] = attention result (loaded from output buffer)
            DataType outDtype = resolveDtype(outSlot);
            auto loaded = loadBlock(outSlot, outDtype);
            if (loaded) ssaValues[outSlot] = loaded;

            // output[1] = present_key (pass-through effective key SSA)
            // output[2] = present_value (pass-through effective value SSA)
            if (slot.numOutputs >= 2) {
              if (ssaValues.count(effectiveKSrc)) {
                ssaValues[slot.outputSlotIndices[1]] = ssaValues[effectiveKSrc];
              } else {
                auto kLoaded = loadBlock(effectiveKSrc, resolveDtype(effectiveKSrc));
                if (kLoaded) ssaValues[slot.outputSlotIndices[1]] = kLoaded;
              }
            }
            if (slot.numOutputs >= 3) {
              if (ssaValues.count(effectiveVSrc)) {
                ssaValues[slot.outputSlotIndices[2]] = ssaValues[effectiveVSrc];
              } else {
                auto vLoaded = loadBlock(effectiveVSrc, resolveDtype(effectiveVSrc));
                if (vLoaded) ssaValues[slot.outputSlotIndices[2]] = vLoaded;
              }
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
            int rank = static_cast<int>(dataShape.size());

            std::string opLower = slot.opName;
            std::transform(opLower.begin(), opLower.end(), opLower.begin(), ::tolower);
            bool isSplitV = (opLower.find("split_v") != std::string::npos ||
                             opLower.find("splitv") != std::string::npos);

            int splitAxis = 0;
            if (isSplitV) {
              // SplitV iArgs: [splitDim, numSplit]
              if (slot.numIArgs > 0 && slot.iArgs) splitAxis = static_cast<int>(slot.iArgs[0]);
            } else {
              // Split iArgs: [numSplit, splitDim] (most common) or [splitDim]
              if (slot.numIArgs > 1 && slot.iArgs) splitAxis = static_cast<int>(slot.iArgs[1]);
              else if (slot.numIArgs > 0 && slot.iArgs) splitAxis = static_cast<int>(slot.iArgs[0]);
            }
            if (splitAxis < 0) splitAxis += rank;
            if (splitAxis < 0 || splitAxis >= rank) splitAxis = 0;

            if (isSplitV && slot.numInputs >= 2) {
              // SplitV: variable chunk sizes stored in input[1] (a constant int tensor)
              int sizesSrc = slot.inputSourceIndices[1];
              NDArray* sizesArr = resolveArr(sizesSrc);
              if (sizesArr && !dataShape.empty()) {
                // Build per-output slice with variable axis sizes
                int axisOffset = 0;
                for (int o = 0; o < slot.numOutputs && o < static_cast<int>(outPtrs.size()); o++) {
                  int chunkAxisSize = (o < static_cast<int>(sizesArr->lengthOf()))
                      ? static_cast<int>(sizesArr->e<int>(o)) : 1;
                  std::vector<int> begins(rank, 0);
                  std::vector<int> ends;
                  for (int d = 0; d < rank; d++) ends.push_back(static_cast<int>(dataShape[d]));
                  begins[splitAxis] = axisOffset;
                  ends[splitAxis] = axisOffset + chunkAxisSize;
                  std::vector<int> strides(rank, 1);
                  int chunkTotalElements = 1;
                  for (int d = 0; d < rank; d++)
                    chunkTotalElements *= (d == splitAxis) ? chunkAxisSize : static_cast<int>(dataShape[d]);
                  emitSliceSection(builder, loc, pid, blockSize, dataPtr, outPtrs[o],
                                   begins, ends, strides, dataShape, chunkTotalElements);
                  axisOffset += chunkAxisSize;
                }
              } else {
                // Fallback: equal splits if sizes not available
                int nElements = static_cast<int>(shapeLength(dataShape));
                emitSplitSection(builder, loc, pid, blockSize, dataPtr, outPtrs, splitAxis, slot.numOutputs, dataShape, nElements);
              }
            } else {
              // Equal split
              int nElements = static_cast<int>(shapeLength(dataShape));
              emitSplitSection(builder, loc, pid, blockSize, dataPtr, outPtrs, splitAxis, slot.numOutputs, dataShape, nElements);
            }

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
            // Get permutation from iArgs; fall back to reverse if not provided
            std::vector<int> permutation;
            if (slot.numIArgs > 0 && slot.iArgs) {
              for (int d = 0; d < slot.numIArgs; d++)
                permutation.push_back(static_cast<int>(slot.iArgs[d]));
            }
            if (permutation.empty()) {
              for (int d = static_cast<int>(inputShape.size()) - 1; d >= 0; d--)
                permutation.push_back(d);
            }
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
              // Conv2D iArgs: [kH, kW, sH, sW, pH, pW, dH, dW, paddingMode, dataFormat, weightsFormat]
              int strideH = (slot.numIArgs > 2 && slot.iArgs) ? static_cast<int>(slot.iArgs[2]) : 1;
              int strideW = (slot.numIArgs > 3 && slot.iArgs) ? static_cast<int>(slot.iArgs[3]) : 1;
              int padH = (slot.numIArgs > 4 && slot.iArgs) ? static_cast<int>(slot.iArgs[4]) : 0;
              int padW = (slot.numIArgs > 5 && slot.iArgs) ? static_cast<int>(slot.iArgs[5]) : 0;
              int wFormat = (slot.numIArgs > 10 && slot.iArgs) ? static_cast<int>(slot.iArgs[10]) : 0;

              int nElements = static_cast<int>(shapeLength(outputShape));
              emitConvolutionSection(builder, loc, pid, blockSize, inPtr, filterPtr, outPtr,
                                      inputShape, filterShape, outputShape, strideH, strideW, padH, padW, nElements, wFormat);
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

    // Continue emitting after the section guard.
    builder.setInsertionPointAfter(secIf);
    // Close multi-phase guard if present
    if (useMultiPhaseLaunch) {
      builder.setInsertionPointAfter(phaseIf);
    }
  }

  // Return
  builder.create<mlir::triton::ReturnOp>(loc);

  // ── Grid and launch configuration ──
  result.gridX = fixedGridX;
  result.gridY = fixedGridY;
  result.gridZ = fixedGridZ;
  result.blockX = blockSize;
  result.blockY = 1;
  result.blockZ = 1;
  result.numWarps = numWarps;
  result.numStages = numStages;
  result.useIndirectArgs = useIndirectArgs;
  result.useCooperativeLaunch = needsGridSync;
  result.useDynamicGrid = false;
  result.requiredGrid = maxSectionGrid;
  result.sections = sections;
  result.useMultiPhaseLaunch = useMultiPhaseLaunch;
  result.launchPhases = launchPhases;

  // Estimate shared memory from section types and tile sizes.
  // This is used for early cooperative launch capacity rejection BEFORE the
  // expensive TTIR→PTX compilation. The actual value (set by AllocateSharedMemoryPass
  // during TTGIR lowering) may differ, but this estimate is conservative enough
  // to catch clearly impossible cooperative launch configurations.
  //
  // Section type shared memory breakdown:
  //   MATMUL:           A+B tiles in shared memory, multi-buffered by numStages
  //   FUSED_ATTENTION:  Q+K+V tiles, uses estimateFusedAttentionSharedMemBytes()
  //   REDUCTION:        Tree reduction scratch: BLOCK_SIZE * sizeof(float)
  //   NORMALIZATION:    Multi-pass reduction (max + exp-sum + norm): 2x reduction
  //   CONVOLUTION:      Scalar 1D loop (no tiled tt.dot), no shared memory
  //   ELEMENTWISE:      Pure register ops, no shared memory
  //   GATHER/SCATTER:   1D indexed load/store, no shared memory
  //   CONCAT/SPLIT:     1D cascading select/partition, no shared memory
  //   STACK:            1D like concat, no shared memory
  //   TILE:             1D modular indexing, no shared memory
  //   STRIDED_SLICE:    1D strided load, no shared memory
  //   SHAPE_MANIPULATION: Stride recomputation, no shared memory
  //   CONSTANT_GENERATION: Immediate stores, no shared memory
  //   IDENTITY:         SSA forwarding, no IR ops
  {
    int maxSmem = 0;
    for (const auto& sec : sections) {
      int secSmem = 0;
      switch (sec.type) {
        case KernelSectionType::MATMUL: {
          // Tiled matmul with K-loop: tiles A[BM,BK] and B[BK,BN] in shared mem,
          // double/triple-buffered by numStages. fp16/bf16 → 2 bytes per element.
          int bm = std::max(1, sec.blockM);
          int bn = std::max(1, sec.blockN);
          int bk = std::max(1, sec.blockK);
          secSmem = (bm * bk + bk * bn) * 2 * numStages;
          break;
        }
        case KernelSectionType::FUSED_ATTENTION: {
          // Flash attention: use the same estimator as the tile selection code
          // which accounts for Q[BM,HD] + K[BN,HD] + V[BN,HD] + overhead.
          int hd = std::max(1, sec.headDim);
          int sq = std::max(1, sec.seqQ);
          int sk = std::max(1, sec.seqK);
          auto attnTile = chooseFusedAttentionTileConfig(
              sec.batchSize, sec.numHeads, sq, sk, hd);
          secSmem = attnTile.estimatedSharedMemBytes;
          break;
        }
        case KernelSectionType::REDUCTION: {
          // Triton tt.reduce: tree reduction using shared memory shuffle.
          // AllocateSharedMemoryPass allocates BLOCK_SIZE * elemSize for the
          // reduction scratch. We assume fp32 (4 bytes) as worst case.
          secSmem = blockSize * 4;
          break;
        }
        case KernelSectionType::NORMALIZATION: {
          // Softmax/LayerNorm/RMSNorm: multiple reduction passes
          // (e.g., max → exp-sum → divide for softmax). Each pass needs
          // BLOCK_SIZE * 4 bytes. Two concurrent reduction buffers worst case.
          secSmem = blockSize * 4 * 2;
          break;
        }
        case KernelSectionType::CONVOLUTION: {
          // Conv2d uses scalar element-wise loops (no tiled tt.dot in the
          // current backend), so no shared memory beyond what Triton
          // allocates for cross-warp communication.
          // Conservative estimate: blockSize * 4 for potential internal shuffles.
          secSmem = blockSize * 4;
          break;
        }
        // All remaining section types are 1D element-wise patterns that
        // operate purely on registers and global memory:
        case KernelSectionType::ELEMENTWISE:
        case KernelSectionType::IDENTITY:
        case KernelSectionType::CONSTANT_GENERATION:
        case KernelSectionType::SHAPE_MANIPULATION:
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
          // No shared memory needed. Triton may allocate a small amount
          // for internal communication but it's negligible (<256 bytes).
          secSmem = 0;
          break;
      }
      maxSmem = std::max(maxSmem, secSmem);
    }
    // Cooperative kernels need additional shared memory for grid sync barriers.
    // The Triton cooperative launch protocol uses a shared counter + flags.
    // 16KB is a safe lower bound for the barrier infrastructure.
    if (needsGridSync) {
      maxSmem = std::max(maxSmem, 16384);
    }
    result.estimatedSharedMemBytes = maxSmem;
  }

  dumpSectionBreakdown(sections, startSlot, endSlot, maxSectionGrid, needsGridSync);

  result.mlirModule = new mlir::ModuleOp(moduleOp);
  result.mlirContext = mlirContext;
  result.valid = true;

  // Dump TTIR module for diagnostics
  {
    std::string ttirDump;
    llvm::raw_string_ostream os(ttirDump);
    moduleOp.print(os);
    sd_debug("TritonIRBuilder: built sectioned module '%s' with %d sections, %d ops, "
              "%d input args, %d output args, maxGrid=%d, cooperative=%s, multiPhase=%s(%d phases)\nTTIR:\n%s\n",
              result.kernelName.c_str(), static_cast<int>(sections.size()),
              segSize, static_cast<int>(inputArgs.size()),
              static_cast<int>(outputArgs.size()), maxSectionGrid,
              needsGridSync ? "YES" : "NO",
              useMultiPhaseLaunch ? "YES" : "NO",
              static_cast<int>(launchPhases.size()), ttirDump.c_str());
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

      // Strategy 4: For M and K, try input slot's shape cache (cachedOutputShapes)
      if (slots[i].numInputs >= 1) {
        int aSrc = slots[i].inputSourceIndices[0];
        if (aSrc >= 0 && aSrc < static_cast<int>(totalOutputSlots)) {
          // Find the producing slot's cached output shape for aSrc
          if ((matmulM == 0 || matmulK == 0)) {
            for (int s = 0; s < static_cast<int>(totalSlots); s++) {
              for (int o = 0; o < slots[s].numOutputs; o++) {
                if (slots[s].outputSlotIndices[o] == aSrc &&
                    slots[s].shapeCacheValid && !slots[s].cachedOutputShapes.empty() &&
                    o < static_cast<int>(slots[s].cachedOutputShapes.size())) {
                  const LongType* si = slots[s].cachedOutputShapes[o];
                  if (si) {
                    int rank = static_cast<int>(shape::rank(si));
                    if (rank >= 2) {
                      if (matmulM == 0) matmulM = static_cast<int>(shape::shapeOf(si)[rank - 2]);
                      if (matmulK == 0) matmulK = static_cast<int>(shape::shapeOf(si)[rank - 1]);
                    }
                  }
                }
              }
              if (matmulM > 0 && matmulK > 0) break;
            }
          }
          // Also check cachedOutputShapes of the producing slot
          if (matmulK == 0 || matmulM == 0) {
            for (int s = 0; s < startSlot; s++) {
              for (int o = 0; o < slots[s].numOutputs; o++) {
                if (slots[s].outputSlotIndices[o] == aSrc &&
                    slots[s].shapeCacheValid && !slots[s].cachedOutputShapes.empty()) {
                  const LongType* shapeInfo = slots[s].cachedOutputShapes[o];
                  if (shapeInfo) {
                    int rank = static_cast<int>(shape::rank(shapeInfo));
                    if (rank >= 2) {
                      if (matmulM == 0) matmulM = static_cast<int>(shape::shapeOf(shapeInfo)[rank - 2]);
                      if (matmulK == 0) matmulK = static_cast<int>(shape::shapeOf(shapeInfo)[rank - 1]);
                    }
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

  // Matmul shared memory: tiles A[BM,BK] + B[BK,BN] in shared mem, multi-buffered.
  // fp16/bf16 → 2 bytes per element.
  result.estimatedSharedMemBytes = (blockM * blockK + blockK * blockN) * 2 * numStages;

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

  // Helper: check if a section type can be merged with element-wise.
  // SHAPE_MANIPULATION (reshape, squeeze, expand_dims, permute) is merged into
  // element-wise sections and SSA-forwarded. For autoregressive decode (seq=1),
  // permute [0,2,1,3] on [1,1,H,D] is identity; for general seq>1, permute
  // would need actual reordering via a transposed store at the section boundary.
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
    } else if (!canMergeWithElementwise(currentSection.type)) {
      // Non-mergeable section types can still absorb consecutive ops of the SAME type.
      // This reduces section/barrier count without changing per-op emission semantics.
      startNewSection = (sectionType != currentSection.type);
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
        // Fallback: resolve M from input A's producing slot's cached output shape
        if (currentSection.matmulM == 0) {
          int aSrc = slots[i].inputSourceIndices[0];
          if (aSrc >= 0 && aSrc < totalOutputSlots) {
            for (int s = startSlot; s <= endSlot; s++) {
              for (int o = 0; o < slots[s].numOutputs; o++) {
                if (slots[s].outputSlotIndices[o] == aSrc &&
                    slots[s].shapeCacheValid && !slots[s].cachedOutputShapes.empty() &&
                    o < static_cast<int>(slots[s].cachedOutputShapes.size())) {
                  const LongType* si = slots[s].cachedOutputShapes[o];
                  if (si && shape::rank(si) >= 2) {
                    currentSection.matmulM = static_cast<int>(shape::shapeOf(si)[shape::rank(si) - 2]);
                    break;
                  }
                }
              }
              if (currentSection.matmulM > 0) break;
            }
          }
        }
        // Fallback: resolve M from output shape (output of matmul is [..., M, N])
        if (currentSection.matmulM == 0 && slots[i].numOutputs > 0) {
          int outIdx = slots[i].outputSlotIndices[0];
          NDArray* outArr = resolveArray(outIdx);
          if (outArr && outArr->rankOf() >= 2) {
            currentSection.matmulM = static_cast<int>(outArr->sizeAt(outArr->rankOf() - 2));
            if (currentSection.matmulN == 0)
              currentSection.matmulN = static_cast<int>(outArr->sizeAt(outArr->rankOf() - 1));
          } else if (outIdx >= 0 && outIdx < totalOutputSlots) {
            // Try the producing slot's cached output shape for outIdx
            for (int s = startSlot; s <= endSlot; s++) {
              for (int o = 0; o < slots[s].numOutputs; o++) {
                if (slots[s].outputSlotIndices[o] == outIdx &&
                    slots[s].shapeCacheValid && !slots[s].cachedOutputShapes.empty() &&
                    o < static_cast<int>(slots[s].cachedOutputShapes.size())) {
                  const LongType* si = slots[s].cachedOutputShapes[o];
                  if (si && shape::rank(si) >= 2) {
                    currentSection.matmulM = static_cast<int>(shape::shapeOf(si)[shape::rank(si) - 2]);
                    if (currentSection.matmulN == 0)
                      currentSection.matmulN = static_cast<int>(shape::shapeOf(si)[shape::rank(si) - 1]);
                  }
                }
              }
              if (currentSection.matmulM > 0) break;
            }
          }
        }
        // Fallback: resolve K from input A's producing slot's cached output shape
        if (currentSection.matmulK == 0) {
          int aSrc = slots[i].inputSourceIndices[0];
          if (aSrc >= 0 && aSrc < totalOutputSlots) {
            for (int s = startSlot; s <= endSlot; s++) {
              for (int o = 0; o < slots[s].numOutputs; o++) {
                if (slots[s].outputSlotIndices[o] == aSrc &&
                    slots[s].shapeCacheValid && !slots[s].cachedOutputShapes.empty() &&
                    o < static_cast<int>(slots[s].cachedOutputShapes.size())) {
                  const LongType* si = slots[s].cachedOutputShapes[o];
                  if (si && shape::rank(si) >= 2) {
                    currentSection.matmulK = static_cast<int>(shape::shapeOf(si)[shape::rank(si) - 1]);
                  }
                }
              }
              if (currentSection.matmulK > 0) break;
            }
          }
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
          if (rank >= 4) {
            // 4D BHSD: [batch, numHeads, seqQ, headDim]
            currentSection.headDim = static_cast<int>(qArr->sizeAt(rank - 1));
            currentSection.seqQ = static_cast<int>(qArr->sizeAt(rank - 2));
            currentSection.numHeads = static_cast<int>(qArr->sizeAt(rank - 3));
            currentSection.batchSize = 1;
            for (int d = 0; d < rank - 3; d++)
              currentSection.batchSize *= static_cast<int>(qArr->sizeAt(d));
            currentSection.attnQIsBSHD = false;
          } else {
            // 3D BSHD: [batch, seqQ, numHeads*headDim]
            currentSection.batchSize = static_cast<int>(qArr->sizeAt(0));
            currentSection.seqQ = static_cast<int>(qArr->sizeAt(1));
            int hidden = static_cast<int>(qArr->sizeAt(2));
            int nh = (slots[i].numIArgs > 0 && slots[i].iArgs) ? static_cast<int>(slots[i].iArgs[0]) : 1;
            if (nh <= 0) nh = 1;
            currentSection.numHeads = nh;
            currentSection.headDim = hidden / nh;
            currentSection.attnQIsBSHD = true;
          }
          currentSection.attentionScale = 1.0f / std::sqrt(static_cast<float>(currentSection.headDim));
        }

        // Determine effective K source: use past_key (input 4) if available
        bool hasPastKv = false;
        int effectiveKInputIdx = 1;
        if (slots[i].numInputs > 4) {
          NDArray* pastKeyArr = resolveArray(slots[i].inputSourceIndices[4]);
          if (pastKeyArr && pastKeyArr->rankOf() == 4) {
            auto pastKeyLen = pastKeyArr->sizeAt(pastKeyArr->rankOf() - 2);
            if (pastKeyLen > 1) {
              hasPastKv = true;
              effectiveKInputIdx = 4;
            }
          }
        }

        NDArray* kArr = (effectiveKInputIdx < slots[i].numInputs) ?
            resolveArray(slots[i].inputSourceIndices[effectiveKInputIdx]) : nullptr;
        if (kArr && kArr->rankOf() >= 2) {
          currentSection.seqK = static_cast<int>(kArr->sizeAt(kArr->rankOf() - 2));
          currentSection.attnKIsBSHD = hasPastKv ? false : currentSection.attnQIsBSHD;
          // GQA: extract KV head count from past_key (4D: [B, KvHeads, seqK, HD])
          if (hasPastKv && kArr->rankOf() == 4) {
            currentSection.numKvHeads = static_cast<int>(kArr->sizeAt(1));
            currentSection.headDim = static_cast<int>(kArr->sizeAt(3));
          }
        }
        // Default: MHA (numKvHeads = numHeads)
        if (currentSection.numKvHeads <= 0) currentSection.numKvHeads = currentSection.numHeads;
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
      int batchSize = std::max(1, section.batchSize);
      int numHeads = std::max(1, section.numHeads);
      int seqQ = std::max(1, section.seqQ);
      int seqK = std::max(1, section.seqK);
      int headDim = std::max(1, section.headDim);
      auto attnTile = chooseFusedAttentionTileConfig(
          batchSize, numHeads, seqQ, seqK, headDim);
      int batchHeads = batchSize * numHeads;
      int blockM = std::max(1, attnTile.blockM);
      int gridQ = (seqQ + blockM - 1) / blockM;
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
      // Placeholder. buildSectionedModule() recomputes launch grid from resolved
      // section shapes after tile selection.
      return 1;
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


void TritonIRBuilder::emitThreadfenceBarrier(mlir::OpBuilder& builder, mlir::Location loc) {
  // Lightweight inter-section barrier using membar.gl + bar.sync.
  // membar.gl flushes all pending global memory stores, making them visible
  // to all other blocks on the GPU. bar.sync 0 synchronizes threads within
  // the CTA so all threads in this block see the flushed state.
  //
  // Unlike emitGridSync(), this does NOT wait for other blocks to arrive.
  // The GPU scheduler will eventually run all blocks. Later sections that
  // read cross-block data will see committed writes because membar.gl
  // guarantees global visibility ordering.
  //
  // This allows arbitrary grid sizes without the cooperative launch
  // co-residency requirement (no cuLaunchCooperativeKernel needed).
  std::string asmStr =
      "{\n"
      "  membar.gl;\n"
      "  bar.sync 0;\n"
      "}\n";

  auto i32Type = builder.getI32Type();
  builder.create<mlir::triton::ElementwiseInlineAsmOp>(
      loc, /*resultTypes=*/mlir::TypeRange{i32Type},
      asmStr,
      /*constraints=*/"=r",
      /*isPure=*/false, /*pack=*/1,
      mlir::ValueRange{});
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
//
// Concatenates inputs along `axis` into output.  Supports N-D arrays.
//
// Strategy: For each output element at linear index `out_off`:
//   1. Unravel out_off into N-D coordinates using the output shape.
//   2. Determine which input owns this element by inspecting the concat axis coordinate.
//   3. Compute the corresponding input linear index and load from that input.
//
// This is fully unrolled across inputs (no ForOp) to avoid Triton compiler bugs.

void TritonIRBuilder::emitConcatSection(mlir::OpBuilder& builder, mlir::Location loc,
                                         mlir::Value pid, int blockSize,
                                         const std::vector<mlir::Value>& inputPtrs,
                                         mlir::Value outputPtr, int axis,
                                         const std::vector<std::vector<LongType>>& inputShapes,
                                         int nElements) {
  if (inputPtrs.empty() || inputShapes.empty()) return;

  auto i32Type = builder.getI32Type();
  auto i32TensorType = mlir::RankedTensorType::get({blockSize}, i32Type);

  // Derive element type from the output pointer
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

  // Compute output shape from input shapes
  int ndim = static_cast<int>(inputShapes[0].size());
  // Normalize axis to positive
  int normAxis = (axis < 0) ? (ndim + axis) : axis;
  if (normAxis < 0 || normAxis >= ndim) normAxis = 0;  // fallback

  // Compute output shape
  std::vector<int> outShape(ndim);
  for (int d = 0; d < ndim; d++) outShape[d] = static_cast<int>(inputShapes[0][d]);
  for (size_t inp = 1; inp < inputShapes.size(); inp++) {
    outShape[normAxis] += static_cast<int>(inputShapes[inp][normAxis]);
  }

  // Compute output strides (C-order: stride[d] = product(outShape[d+1..ndim-1]))
  std::vector<int> outStrides(ndim, 1);
  for (int d = ndim - 2; d >= 0; d--) outStrides[d] = outStrides[d + 1] * outShape[d + 1];

  // Unravel out linear index to N-D coordinates
  // coord[d] = (offsets / outStrides[d]) % outShape[d]
  std::vector<mlir::Value> coords(ndim);
  mlir::Value rem = offsets;
  for (int d = 0; d < ndim; d++) {
    if (outStrides[d] > 1) {
      auto strideConst = builder.create<mlir::arith::ConstantIntOp>(loc, outStrides[d], 32);
      auto strideSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, strideConst);
      coords[d] = builder.create<mlir::arith::DivSIOp>(loc, rem, strideSplat);
      rem = builder.create<mlir::arith::RemSIOp>(loc, rem, strideSplat);
    } else {
      // Last dimension: coord = rem
      coords[d] = rem;
    }
    if (d < ndim - 1 && outShape[d] > 0) {
      // Clamp/mod to outShape[d] (for safety, in case blockSize padding elements are out of range)
      auto shapeConst = builder.create<mlir::arith::ConstantIntOp>(loc, outShape[d], 32);
      auto shapeSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, shapeConst);
      coords[d] = builder.create<mlir::arith::RemSIOp>(loc, coords[d], shapeSplat);
    }
  }

  // For each input, compute: is this output element from this input?
  // The concat axis coordinate tells us: cumAxisStart[i] <= coord[normAxis] < cumAxisEnd[i]
  // If yes, compute the input linear index and load from input[i].
  mlir::Value result = splatConstantF32(builder, loc, elemTensorType, 0.0f);

  int cumAxisOffset = 0;
  for (size_t inp = 0; inp < inputPtrs.size(); inp++) {
    if (inp >= inputShapes.size()) break;
    const auto& inShape = inputShapes[inp];
    int inAxisSize = static_cast<int>(inShape[normAxis]);

    // Compute input strides (C-order)
    std::vector<int> inStrides(ndim, 1);
    for (int d = ndim - 2; d >= 0; d--) inStrides[d] = inStrides[d + 1] * static_cast<int>(inShape[d + 1]);

    // Determine if this element belongs to input[inp]:
    // cumAxisOffset <= coord[normAxis] < cumAxisOffset + inAxisSize
    auto startConst = builder.create<mlir::arith::ConstantIntOp>(loc, cumAxisOffset, 32);
    auto endConst = builder.create<mlir::arith::ConstantIntOp>(loc, cumAxisOffset + inAxisSize, 32);
    auto splatStart = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, startConst);
    auto splatEnd = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, endConst);
    auto geStart = builder.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::sge, coords[normAxis], splatStart);
    auto ltEnd = builder.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::slt, coords[normAxis], splatEnd);
    auto inRange = builder.create<mlir::arith::AndIOp>(loc, geStart, ltEnd);
    auto loadMask = builder.create<mlir::arith::AndIOp>(loc, mask, inRange);

    // Compute the local axis coordinate within input[inp]
    // localAxisCoord = coord[normAxis] - cumAxisOffset
    mlir::Value localAxisCoord;
    if (cumAxisOffset == 0) {
      localAxisCoord = coords[normAxis];
    } else {
      auto startSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, startConst);
      localAxisCoord = builder.create<mlir::arith::SubIOp>(loc, coords[normAxis], startSplat);
    }

    // Compute input linear offset: sum over dims of coord[d] * inStrides[d]
    // Use localAxisCoord for normAxis dimension
    mlir::Value inOffset = nullptr;
    for (int d = 0; d < ndim; d++) {
      mlir::Value coordD = (d == normAxis) ? localAxisCoord : coords[d];
      if (inStrides[d] == 0) continue;
      mlir::Value contribution;
      if (inStrides[d] == 1) {
        contribution = coordD;
      } else {
        auto strideConst = builder.create<mlir::arith::ConstantIntOp>(loc, inStrides[d], 32);
        auto strideSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, strideConst);
        contribution = builder.create<mlir::arith::MulIOp>(loc, coordD, strideSplat);
      }
      if (!inOffset) {
        inOffset = contribution;
      } else {
        inOffset = builder.create<mlir::arith::AddIOp>(loc, inOffset, contribution);
      }
    }
    if (!inOffset) {
      auto zero = builder.create<mlir::arith::ConstantIntOp>(loc, 0, 32);
      inOffset = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, zero);
    }

    // Load from input[inp] at inOffset (using loadMask so out-of-range elements load zeros)
    auto inPtrType = mlir::cast<mlir::triton::PointerType>(inputPtrs[inp].getType());
    auto inPtrTensorType = mlir::RankedTensorType::get({blockSize}, inPtrType);
    auto splatInPtr = builder.create<mlir::triton::SplatOp>(loc, inPtrTensorType, inputPtrs[inp]);
    auto inPtrsOp = builder.create<mlir::triton::AddPtrOp>(loc, inPtrTensorType, splatInPtr, inOffset);
    auto loaded = builder.create<mlir::triton::LoadOp>(loc,
        inPtrsOp.getResult(), loadMask.getResult(), mlir::Value(),
        mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL, false);

    // Cast to output element type
    mlir::Value castLoaded = castTo(builder, loc, loaded, elemType);

    // Select: if this element belongs to input[inp], use loaded; otherwise keep current result
    result = builder.create<mlir::arith::SelectOp>(loc, inRange, castLoaded, result);

    cumAxisOffset += inAxisSize;
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

  // ND strided slice: for each output flat index, unravel to output ND coords,
  // compute input ND coords as: in_coord[d] = begin[d] + out_coord[d] * stride[d],
  // then ravel to input flat offset.
  int rank = static_cast<int>(inputShape.size());

  // Compute output shape per dimension: ceil((end[d] - begin[d]) / stride[d])
  std::vector<int> outShape(rank);
  for (int d = 0; d < rank; d++) {
    int b = (d < static_cast<int>(begins.size())) ? begins[d] : 0;
    int e = (d < static_cast<int>(ends.size())) ? ends[d] : static_cast<int>(inputShape[d]);
    int s = (d < static_cast<int>(strides.size())) ? strides[d] : 1;
    if (s == 0) s = 1;
    outShape[d] = (e - b + s - 1) / s;  // ceil division
    if (outShape[d] <= 0) outShape[d] = 1;
  }

  // Compute output strides (row-major)
  std::vector<int> outStrides(rank, 1);
  for (int d = rank - 2; d >= 0; d--)
    outStrides[d] = outStrides[d + 1] * outShape[d + 1];

  // Compute input strides (row-major)
  std::vector<int> inStrides(rank, 1);
  for (int d = rank - 2; d >= 0; d--)
    inStrides[d] = inStrides[d + 1] * static_cast<int>(inputShape[d + 1]);

  // Unravel output flat offset to ND coords, apply begin+stride, ravel to input offset
  mlir::Value srcOffsets = splatConstantI32(builder, loc, i32TensorType, 0);
  mlir::Value remaining = offsets;
  for (int d = 0; d < rank; d++) {
    auto oStrideConst = splatConstantI32(builder, loc, i32TensorType, outStrides[d]);
    auto coord = builder.create<mlir::arith::DivSIOp>(loc, remaining, oStrideConst);
    if (d < rank - 1)
      remaining = builder.create<mlir::arith::RemSIOp>(loc, remaining, oStrideConst);
    // input_coord = begin[d] + coord * stride[d]
    int b = (d < static_cast<int>(begins.size())) ? begins[d] : 0;
    int s = (d < static_cast<int>(strides.size())) ? strides[d] : 1;
    mlir::Value inCoord;
    if (s == 1) {
      auto beginSplat = splatConstantI32(builder, loc, i32TensorType, b);
      inCoord = builder.create<mlir::arith::AddIOp>(loc, coord, beginSplat);
    } else {
      auto strideSplat = splatConstantI32(builder, loc, i32TensorType, s);
      auto scaled = builder.create<mlir::arith::MulIOp>(loc, coord, strideSplat);
      auto beginSplat = splatConstantI32(builder, loc, i32TensorType, b);
      inCoord = builder.create<mlir::arith::AddIOp>(loc, scaled, beginSplat);
    }
    auto inStrideSplat = splatConstantI32(builder, loc, i32TensorType, inStrides[d]);
    auto contrib = builder.create<mlir::arith::MulIOp>(loc, inCoord, inStrideSplat);
    srcOffsets = builder.create<mlir::arith::AddIOp>(loc, srcOffsets, contrib);
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

  // Tile: for each output flat index, unravel to ND output coords,
  // apply modulo per-dimension (coord_d % inputShape[d]) to get input coords,
  // then ravel to input flat offset.
  // outputShape[d] = inputShape[d] * repeats[d]
  int rank = static_cast<int>(inputShape.size());

  // Derive pointer types from actual arguments (NOT hardcoded f32)
  auto inPtrType = mlir::cast<mlir::triton::PointerType>(inputPtr.getType());
  auto outPtrType = mlir::cast<mlir::triton::PointerType>(outputPtr.getType());
  auto inPtrTensorType = mlir::RankedTensorType::get({blockSize}, inPtrType);
  auto outPtrTensorType = mlir::RankedTensorType::get({blockSize}, outPtrType);

  // Compute output shape = inputShape * repeats
  std::vector<int> outShape(rank);
  for (int d = 0; d < rank; d++) {
    int rep = (d < static_cast<int>(repeats.size())) ? repeats[d] : 1;
    outShape[d] = static_cast<int>(inputShape[d]) * rep;
  }

  // Compute output strides (row-major)
  std::vector<int> outStrides(rank, 1);
  for (int d = rank - 2; d >= 0; d--)
    outStrides[d] = outStrides[d + 1] * outShape[d + 1];

  // Compute input strides (row-major)
  std::vector<int> inStrides(rank, 1);
  for (int d = rank - 2; d >= 0; d--)
    inStrides[d] = inStrides[d + 1] * static_cast<int>(inputShape[d + 1]);

  // Unravel output flat offset, mod each coord by inputShape[d], ravel to input offset
  mlir::Value srcOffsets = splatConstantI32(builder, loc, i32TensorType, 0);
  mlir::Value remaining = offsets;
  for (int d = 0; d < rank; d++) {
    auto strideConst = splatConstantI32(builder, loc, i32TensorType, outStrides[d]);
    auto coord = builder.create<mlir::arith::DivSIOp>(loc, remaining, strideConst);
    if (d < rank - 1)
      remaining = builder.create<mlir::arith::RemSIOp>(loc, remaining, strideConst);
    // Wrap to input dimension
    auto inDimConst = splatConstantI32(builder, loc, i32TensorType, static_cast<int>(inputShape[d]));
    auto wrappedCoord = builder.create<mlir::arith::RemSIOp>(loc, coord, inDimConst);
    auto inStrideConst = splatConstantI32(builder, loc, i32TensorType, inStrides[d]);
    auto contrib = builder.create<mlir::arith::MulIOp>(loc, wrappedCoord, inStrideConst);
    srcOffsets = builder.create<mlir::arith::AddIOp>(loc, srcOffsets, contrib);
  }

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
    // General ND permute: for each output flat offset, unravel to output coords,
    // apply inverse permutation to get input coords, ravel to input flat offset.
    // output[d0,d1,...] = input[d_perm[0], d_perm[1], ...]
    // For output flat index: unravel with outputShape strides, then
    // srcOffset = sum(coord[perm[i]] * inputStride[i])
    int rank = static_cast<int>(inputShape.size());

    // Compute output strides (row-major): stride[i] = product of outputShape[i+1..rank-1]
    std::vector<int> outStrides(rank, 1);
    for (int d = rank - 2; d >= 0; d--)
      outStrides[d] = outStrides[d + 1] * static_cast<int>(outputShape[d + 1]);

    // Compute input strides (row-major): stride[i] = product of inputShape[i+1..rank-1]
    std::vector<int> inStrides(rank, 1);
    for (int d = rank - 2; d >= 0; d--)
      inStrides[d] = inStrides[d + 1] * static_cast<int>(inputShape[d + 1]);

    // Unravel output flat offset into ND coords, then compute input flat offset
    // srcOffset = 0
    // for each output dim d: coord_d = (flat / outStride[d]) % outShape[d]
    //   srcOffset += coord_d * inStride[perm[d]]
    mlir::Value srcOffsets = splatConstantI32(builder, loc, i32TensorType, 0);
    mlir::Value remaining = offsets;
    for (int d = 0; d < rank; d++) {
      // coord_d = remaining / outStride[d]
      auto strideConst = splatConstantI32(builder, loc, i32TensorType, outStrides[d]);
      auto coord = builder.create<mlir::arith::DivSIOp>(loc, remaining, strideConst);
      if (d < rank - 1) {
        // remaining = remaining % outStride[d]  (for next dimension)
        remaining = builder.create<mlir::arith::RemSIOp>(loc, remaining, strideConst);
      }
      // coord_d in output corresponds to perm[d] in input
      int inputDim = permutation[d];
      auto inStrideConst = splatConstantI32(builder, loc, i32TensorType, inStrides[inputDim]);
      auto contrib = builder.create<mlir::arith::MulIOp>(loc, coord, inStrideConst);
      srcOffsets = builder.create<mlir::arith::AddIOp>(loc, srcOffsets, contrib);
    }

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
  auto& env = Environment::getInstance();
  const bool dumpEnabled = env.tritonDumpSections() || env.tritonVerbose();
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
  auto& env = Environment::getInstance();
  const bool dumpEnabled = env.tritonDumpArgs() || env.tritonVerbose();
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
  (void)pid;
  // Delegate to the existing emitFusedAttentionKernel, which creates its own
  // GetProgramIdOp. For the sectioned kernel, this is called within an scf.if
  // guard so only blocks in the attention section's pid range execute it.
  // Note: emitFusedAttentionKernel uses its own pid0/pid1 from GetProgramIdOp.
  // In the cooperative kernel, we remap pid to the attention section's range.
  auto attnTile = chooseFusedAttentionTileConfig(
      std::max(1, section.batchSize),
      std::max(1, section.numHeads),
      std::max(1, section.seqQ),
      std::max(1, section.seqK),
      std::max(1, section.headDim));
  int blockM = attnTile.blockM;
  int blockN = attnTile.blockN;
  emitFusedAttentionKernel(builder, loc, qPtr, kPtr, vPtr, outPtr,
                            section.batchSize, section.numHeads,
                            section.numKvHeads > 0 ? section.numKvHeads : section.numHeads,
                            section.seqQ, section.seqK,
                            section.headDim, section.attentionScale,
                            blockM, blockN,
                            section.attnQIsBSHD, section.attnKIsBSHD,
                            mlir::Value(), std::vector<LongType>());
}

void TritonIRBuilder::emitSplitSection(mlir::OpBuilder& builder, mlir::Location loc,
                                        mlir::Value pid, int blockSize,
                                        mlir::Value inputPtr,
                                        const std::vector<mlir::Value>& outputPtrs,
                                        int axis, int numSplits,
                                        const std::vector<LongType>& inputShape,
                                        int nElements) {
  // Split input along `axis` into numSplits equal chunks.
  // Each chunk s gets a slice: begin[axis]=s*chunkAlongAxis, end[axis]=(s+1)*chunkAlongAxis.
  if (numSplits <= 0 || inputShape.empty()) return;
  int rank = static_cast<int>(inputShape.size());
  if (axis < 0) axis += rank;
  if (axis < 0 || axis >= rank) axis = 0;

  int axisSize = static_cast<int>(inputShape[axis]);
  int chunkAlongAxis = axisSize / numSplits;
  if (chunkAlongAxis <= 0) chunkAlongAxis = 1;

  // Compute per-chunk output size
  int chunkTotalElements = 1;
  for (int d = 0; d < rank; d++) {
    chunkTotalElements *= (d == axis) ? chunkAlongAxis : static_cast<int>(inputShape[d]);
  }

  for (int s = 0; s < numSplits && s < static_cast<int>(outputPtrs.size()); s++) {
    // Build begins/ends: only the axis dimension differs
    std::vector<int> begins(rank, 0);
    std::vector<int> ends;
    for (int d = 0; d < rank; d++) ends.push_back(static_cast<int>(inputShape[d]));
    begins[axis] = s * chunkAlongAxis;
    ends[axis] = (s + 1) * chunkAlongAxis;
    std::vector<int> strides(rank, 1);
    emitSliceSection(builder, loc, pid, blockSize, inputPtr, outputPtrs[s],
                     begins, ends, strides, inputShape, chunkTotalElements);
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
                                              int nElements, int wFormat) {
  // Direct conv2d: each output element independently computes its value by
  // iterating over the filter spatial dimensions and input channels.
  //
  // Input shape: [N, IC, IH, IW] (NCHW)
  // Filter shape: depends on wFormat:
  //   0=[kH,kW,iC,oC], 1=[oC,iC,kH,kW], 2=[oC,kH,kW,iC]
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

  // Extract OC, KH, KW based on weight format
  int OC, KH, KW;
  if (wFormat == 1) {
    // [oC, iC, kH, kW]
    OC = static_cast<int>(filterShape[0]);
    KH = static_cast<int>(filterShape[2]);
    KW = static_cast<int>(filterShape[3]);
  } else if (wFormat == 2) {
    // [oC, kH, kW, iC]
    OC = static_cast<int>(filterShape[0]);
    KH = static_cast<int>(filterShape[1]);
    KW = static_cast<int>(filterShape[2]);
  } else {
    // wFormat == 0: [kH, kW, iC, oC]
    KH = static_cast<int>(filterShape[0]);
    KW = static_cast<int>(filterShape[1]);
    OC = static_cast<int>(filterShape[3]);
  }

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

  // Unravel linear index to (n, oc, oh, ow).
  // ND4J NCHW conv2d output uses OH as the INNER (faster-varying) spatial dimension
  // and OW as the OUTER spatial dimension. So:
  //   oh = offsets % OH          (inner / fast)
  //   ow = (offsets / OH) % OW   (outer / slow)
  //   oc = (offsets / (OH * OW)) % OC
  //   n  = offsets / (OH * OW * OC)
  auto owConst = builder.create<mlir::arith::ConstantIntOp>(loc, OW, 32);
  auto ohConst = builder.create<mlir::arith::ConstantIntOp>(loc, OH, 32);
  auto ocConst = builder.create<mlir::arith::ConstantIntOp>(loc, OC, 32);
  auto owSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, owConst);
  auto ohSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, ohConst);
  auto ocSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, ocConst);

  auto oh_idx = builder.create<mlir::arith::RemSIOp>(loc, offsets, ohSplat);
  auto tmp1 = builder.create<mlir::arith::DivSIOp>(loc, offsets, ohSplat);
  auto ow_idx = builder.create<mlir::arith::RemSIOp>(loc, tmp1.getResult(), owSplat);
  auto tmp2 = builder.create<mlir::arith::DivSIOp>(loc, tmp1.getResult(), owSplat);
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
  mlir::Value acc = splatConstantF32(builder, loc, f32TensorType, 0.0f);

  // Fully unrolled convolution loop: eliminates scf::ForOp entirely to avoid
  // Triton compiler pass pipeline bugs that corrupt accumulator element ordering.
  // Each iteration is emitted as independent SSA operations.
  int totalIters = IC * KH * KW;

  auto zero = builder.create<mlir::arith::ConstantIntOp>(loc, 0, 32);
  auto ihConst = builder.create<mlir::arith::ConstantIntOp>(loc, IH, 32);
  auto iwConst = builder.create<mlir::arith::ConstantIntOp>(loc, IW, 32);
  auto ihSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, ihConst);
  auto iwSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, iwConst);
  auto zeroSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, zero);
  auto icIhIw = builder.create<mlir::arith::ConstantIntOp>(loc, IC * IH * IW, 32);
  auto ihIw = builder.create<mlir::arith::ConstantIntOp>(loc, IH * IW, 32);
  auto icIhIwSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, icIhIw);
  auto ihIwSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, ihIw);

  for (int iter = 0; iter < totalIters; iter++) {
    int ic_i = iter / (KH * KW);
    int kh_i = (iter / KW) % KH;
    int kw_i = iter % KW;

    auto kh_val = builder.create<mlir::arith::ConstantIntOp>(loc, kh_i, 32);
    auto kw_val = builder.create<mlir::arith::ConstantIntOp>(loc, kw_i, 32);
    auto khSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, kh_val);
    auto kwSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, kw_val);

    // Compute input position: h_in = oh_base + kh, w_in = ow_base + kw
    auto h_in = builder.create<mlir::arith::AddIOp>(loc, oh_base, khSplat);
    auto w_in = builder.create<mlir::arith::AddIOp>(loc, ow_base, kwSplat);

    // Bounds check: 0 <= h_in < IH && 0 <= w_in < IW
    auto h_ge_0 = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::sge, h_in, zeroSplat);
    auto h_lt_IH = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt, h_in, ihSplat);
    auto w_ge_0 = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::sge, w_in, zeroSplat);
    auto w_lt_IW = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt, w_in, iwSplat);
    auto h_valid = builder.create<mlir::arith::AndIOp>(loc, h_ge_0, h_lt_IH);
    auto w_valid = builder.create<mlir::arith::AndIOp>(loc, w_ge_0, w_lt_IW);
    auto in_bounds = builder.create<mlir::arith::AndIOp>(loc, h_valid, w_valid);

    // Input offset: n * IC*IH*IW + ic * IH*IW + h_in * IW + w_in
    auto ic_val = builder.create<mlir::arith::ConstantIntOp>(loc, ic_i, 32);
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

    // Filter offset depends on weight format:
    // wFormat 0: [kH,kW,iC,oC] → kh*kW*iC*oC + kw*iC*oC + ic*oC + oc
    // wFormat 1: [oC,iC,kH,kW] → oc*iC*kH*kW + ic*kH*kW + kh*kW + kw
    // wFormat 2: [oC,kH,kW,iC] → oc*kH*kW*iC + kh*kW*iC + kw*iC + ic
    mlir::Value filterOffset;
    if (wFormat == 0) {
      // [kH, kW, iC, oC]
      int fOff = kh_i * KW * IC * OC + kw_i * IC * OC + ic_i * OC;
      auto fOffConst = builder.create<mlir::arith::ConstantIntOp>(loc, fOff, 32);
      auto fOffSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, fOffConst);
      filterOffset = builder.create<mlir::arith::AddIOp>(loc, fOffSplat, oc_idx);
    } else if (wFormat == 2) {
      // [oC, kH, kW, iC]
      int fOff = kh_i * KW * IC + kw_i * IC + ic_i;
      auto fOffConst = builder.create<mlir::arith::ConstantIntOp>(loc, fOff, 32);
      auto fOffSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, fOffConst);
      // oc_idx * KH * KW * IC + fOff
      auto khKwIc = builder.create<mlir::arith::ConstantIntOp>(loc, KH * KW * IC, 32);
      auto khKwIcSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, khKwIc);
      filterOffset = builder.create<mlir::arith::AddIOp>(loc,
          builder.create<mlir::arith::MulIOp>(loc, oc_idx, khKwIcSplat), fOffSplat);
    } else {
      // wFormat 1: [oC, iC, kH, kW]
      // oc * IC*KH*KW + (ic*KH*KW + kh*KW + kw) — ic/kh/kw are compile-time constants
      int fOff = ic_i * KH * KW + kh_i * KW + kw_i;
      auto fOffConst = builder.create<mlir::arith::ConstantIntOp>(loc, fOff, 32);
      auto fOffSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, fOffConst);
      auto icKhKw = builder.create<mlir::arith::ConstantIntOp>(loc, IC * KH * KW, 32);
      auto icKhKwSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, icKhKw);
      filterOffset = builder.create<mlir::arith::AddIOp>(loc,
          builder.create<mlir::arith::MulIOp>(loc, oc_idx, icKhKwSplat), fOffSplat);
    }

    // Load filter value (masked by element mask only, filter is always in bounds)
    auto splatFilterPtr = builder.create<mlir::triton::SplatOp>(loc, filtPtrTensorType, filterPtr);
    auto filterPtrs = builder.create<mlir::triton::AddPtrOp>(loc, filtPtrTensorType, splatFilterPtr, filterOffset);
    auto filterLoaded = builder.create<mlir::triton::LoadOp>(loc,
        filterPtrs.getResult(), mask.getResult(), mlir::Value(),
        mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL, false);
    auto filterVal = castTo(builder, loc, filterLoaded, f32Type);

    // Accumulate: acc += input * filter (zero out-of-bounds input)
    auto prod = builder.create<mlir::arith::MulFOp>(loc, inVal, filterVal);
    acc = builder.create<mlir::arith::AddFOp>(loc, acc, prod);
  }

  auto finalAcc = acc;

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

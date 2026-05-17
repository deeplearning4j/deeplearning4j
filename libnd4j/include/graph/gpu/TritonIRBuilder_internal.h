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

#ifndef LIBND4J_TRITON_IR_BUILDER_INTERNAL_H
#define LIBND4J_TRITON_IR_BUILDER_INTERNAL_H

#include <config.h>

#if HAVE_TRITON

#include <array/NDArray.h>
#include <graph/NativeDynamicShapePlan.h>
#include <system/Environment.h>
#include <system/common.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <string>
#include <unordered_set>
#include <vector>

#ifdef SD_CUDA
#include <cuda_runtime.h>
#endif

// MLIR core
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>

// Triton MLIR dialect
#include <triton/Dialect/Triton/IR/Dialect.h>
#include <triton/Dialect/Triton/IR/Types.h>

// Standard MLIR dialects
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Math/IR/Math.h>

namespace sd {
namespace graph {
namespace ir_builder_internal {

// ─── Utility helpers ────────────────────────────────────────────────────────

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

inline std::string normalizeOpToken(const std::string& opName) {
  std::string out;
  out.reserve(opName.size());
  for (unsigned char ch : opName) {
    if (std::isalnum(ch)) {
      out.push_back(static_cast<char>(std::tolower(ch)));
    }
  }
  return out;
}

inline std::vector<int> readHostIntVector(NDArray* arr) {
  std::vector<int> out;
  if (arr == nullptr) return out;
  LongType len = arr->lengthOf();
  if (len <= 0) return out;
  if (len > static_cast<LongType>(std::numeric_limits<int>::max())) {
    len = static_cast<LongType>(std::numeric_limits<int>::max());
  }
  out.reserve(static_cast<size_t>(len));
  arr->syncToHost();
  for (int i = 0; i < static_cast<int>(len); i++) {
    out.push_back(static_cast<int>(arr->e<LongType>(i)));
  }
  return out;
}

template <typename ResolveArrayFn>
inline void resolveStridedSliceParams(const NativeSlot& slot,
                                      const std::vector<LongType>& inputShape,
                                      ResolveArrayFn resolveArray,
                                      std::vector<int>& begins,
                                      std::vector<int>& ends,
                                      std::vector<int>& strides) {
  int rank = static_cast<int>(inputShape.size());
  begins.assign(rank, 0);
  ends.resize(rank);
  strides.assign(rank, 1);
  for (int d = 0; d < rank; d++) {
    ends[d] = static_cast<int>(inputShape[d]);
  }
  if (rank <= 0) return;

  // iArgs fallback: [begin... end... stride...] or [begin... end...]
  if (slot.args.numIArgs > 0 && slot.args.iArgs != nullptr) {
    if (slot.args.numIArgs >= rank * 3) {
      for (int d = 0; d < rank; d++) {
        begins[d] = static_cast<int>(slot.args.iArgs[d]);
        ends[d] = static_cast<int>(slot.args.iArgs[d + rank]);
        int s = static_cast<int>(slot.args.iArgs[d + (2 * rank)]);
        strides[d] = (s == 0 ? 1 : s);
      }
    } else if (slot.args.numIArgs >= rank * 2) {
      for (int d = 0; d < rank; d++) {
        begins[d] = static_cast<int>(slot.args.iArgs[d]);
        ends[d] = static_cast<int>(slot.args.iArgs[d + rank]);
      }
    }
  }

  // Tensor-input slice form (ONNX Slice / dynamic strided slice):
  // data, starts, ends, [axes], [steps]
  if (slot.wiring.numInputs >= 3) {
    auto starts = readHostIntVector(resolveArray(slot.wiring.inputSourceIndices[1]));
    auto endVals = readHostIntVector(resolveArray(slot.wiring.inputSourceIndices[2]));
    std::vector<int> axes;
    std::vector<int> steps;

    if (slot.wiring.numInputs >= 5) {
      axes = readHostIntVector(resolveArray(slot.wiring.inputSourceIndices[3]));
      steps = readHostIntVector(resolveArray(slot.wiring.inputSourceIndices[4]));
    } else if (slot.wiring.numInputs >= 4) {
      auto fourth = readHostIntVector(resolveArray(slot.wiring.inputSourceIndices[3]));
      bool inRange = !fourth.empty();
      std::unordered_set<int> seen;
      bool unique = true;
      for (int v : fourth) {
        if (v < -rank || v >= rank) inRange = false;
        int key = (v < 0 ? v + rank : v);
        if (!seen.insert(key).second) unique = false;
      }
      // Ambiguous 4-input case (axes vs steps): prefer axes only when values look
      // like a valid axis set.
      if (inRange && unique) {
        axes = fourth;
      } else {
        steps = fourth;
      }
    }

    if (!starts.empty() && !endVals.empty()) {
      if (axes.empty()) {
        int count = std::min(static_cast<int>(starts.size()), rank);
        axes.reserve(static_cast<size_t>(count));
        for (int i = 0; i < count; i++) axes.push_back(i);
      }
      int count = std::min(static_cast<int>(starts.size()),
                           std::min(static_cast<int>(endVals.size()),
                                    static_cast<int>(axes.size())));
      for (int i = 0; i < count; i++) {
        int ax = axes[i];
        if (ax < 0) ax += rank;
        if (ax < 0 || ax >= rank) continue;
        begins[ax] = starts[i];
        ends[ax] = endVals[i];
        if (i < static_cast<int>(steps.size()) && steps[i] != 0) {
          strides[ax] = steps[i];
        }
      }
    }
  }
}

// ─── CUDA helpers ───────────────────────────────────────────────────────────

inline int queryCudaSharedMemLimitBytes() {
#ifdef SD_CUDA
  int deviceId = 0;
  cudaGetDevice(&deviceId);
  int smemPerBlock = 0;
  cudaDeviceGetAttribute(&smemPerBlock, cudaDevAttrMaxSharedMemoryPerBlockOptin, deviceId);
  if (smemPerBlock <= 0) {
    // Fallback: query without optin
    cudaDeviceGetAttribute(&smemPerBlock, cudaDevAttrMaxSharedMemoryPerBlock, deviceId);
  }
  // Apply safety margin (reserve 1KB for driver/runtime overhead)
  const int safetyMargin = 1024;
  return (smemPerBlock > safetyMargin) ? (smemPerBlock - safetyMargin) : smemPerBlock;
#else
  return 49152 - 1024;  // Default to 48KB - 1KB (common GPU shared memory limit)
#endif
}

inline int estimateFusedAttentionSharedMemBytes(int headDim, int blockM, int blockN) {
  // Flash Attention pattern uses:
  //   Q tile:  BLOCK_M × headDim × sizeof(float)
  //   K tile:  BLOCK_N × headDim × sizeof(float)
  //   V tile:  BLOCK_N × headDim × sizeof(float)
  //   S tile:  BLOCK_M × BLOCK_N × sizeof(float)
  //   m/l vectors: BLOCK_M × sizeof(float) × 2
  const int headDimPadded = nextPow2AtLeastOne(std::max(1, headDim));
  const int qBytes = blockM * headDimPadded * 4;
  const int kBytes = blockN * headDimPadded * 4;
  const int vBytes = blockN * headDimPadded * 4;
  const int sBytes = blockM * blockN * 4;
  const int mlBytes = blockM * 4 * 2;
  const int totalBytes = qBytes + kBytes + vBytes + sBytes + mlBytes;
  // Add 10% overhead for alignment and padding
  return static_cast<int>(totalBytes * 1.1);
}

int getSectionedCooperativeTargetBlocks();

struct AttentionTileChoice {
  int blockM;
  int blockN;
  int estimatedSharedMemBytes;
  bool fitsSharedMem;
  bool adjustedForSharedMem;
  int sharedMemLimitBytes;
};

inline AttentionTileChoice chooseFusedAttentionTileConfig(int batchSize, int numHeads,
                                                          int seqQ, int seqK,
                                                          int headDim,
                                                          int sharedMemLimitBytes = 0) {
  const int limit = (sharedMemLimitBytes > 0) ? sharedMemLimitBytes : queryCudaSharedMemLimitBytes();
  AttentionTileChoice choice;

  // Check for explicit blockN override from environment
  int blockNOverride = sd::Environment::getInstance().tritonAttentionBlockN();

  // DECODE OPTIMIZATION: For seqQ=1 (single-token decode), use minimal blockM
  // to avoid wasting compute on masked-out positions. Standard flash attention
  // uses blockM=32-128, but decode only needs blockM=1-8.
  if (seqQ <= 4) {
    // Decode or very short prefix: blockM matches actual seqQ, blockN based on seqK
    choice.blockM = (seqQ <= 1) ? 1 : ((seqQ <= 2) ? 2 : 4);
    if (blockNOverride > 0) {
      choice.blockN = blockNOverride;
    } else {
      choice.blockN = (seqK <= 32) ? 32 : ((seqK <= 64) ? 64 : 128);
    }
  } else {
    // Prefill or longer sequences: standard tile selection
    int preferredM = (seqQ <= 32) ? 32 : ((seqQ <= 64) ? 64 : 128);
    int preferredN = (seqK <= 32) ? 32 : ((seqK <= 64) ? 64 : 128);
    preferredM = clampPow2(preferredM, 32, 128);
    preferredN = clampPow2(preferredN, 32, 128);
    choice.blockM = preferredM;
    choice.blockN = preferredN;
  }

  // Check shared memory and scale down if needed
  int chosenBytes = estimateFusedAttentionSharedMemBytes(headDim, choice.blockM, choice.blockN);
  if (chosenBytes > limit) {
    // Scale down blockN first (blockM is already minimal for decode)
    bool found = false;
    for (int n = choice.blockN; n >= 16; n /= 2) {
      int bytes = estimateFusedAttentionSharedMemBytes(headDim, choice.blockM, n);
      if (bytes <= limit) {
        choice.blockN = n;
        chosenBytes = bytes;
        found = true;
        break;
      }
    }
    if (!found) {
      // Minimum possible tiles
      choice.blockM = std::min(choice.blockM, 16);
      choice.blockN = 16;
      chosenBytes = estimateFusedAttentionSharedMemBytes(headDim, choice.blockM, choice.blockN);
    }
  }

  choice.estimatedSharedMemBytes = chosenBytes;
  choice.fitsSharedMem = (chosenBytes <= limit);
  choice.adjustedForSharedMem = true;  // Always report as adjusted for decode path
  choice.sharedMemLimitBytes = limit;
  return choice;
}

// ─── Externally-visible output computation ──────────────────────────────────

inline std::unordered_set<int> computeExternallyVisibleOutputs(
    NativeSlot* slots, int startSlot, int endSlot, int totalSlots,
    int* requestedOutputSlotIndices, int numRequestedOutputs) {

  // 1. Collect all output slot indices produced within the segment
  std::unordered_set<int> segmentOutputs;
  for (int i = startSlot; i <= endSlot; i++) {
    for (int o = 0; o < slots[i].wiring.numOutputs; o++) {
      segmentOutputs.insert(slots[i].wiring.outputSlotIndices[o]);
    }
  }

  // 2. Find outputs consumed by slots OUTSIDE [startSlot, endSlot]
  std::unordered_set<int> externallyConsumed;
  for (int i = 0; i < totalSlots; i++) {
    if (i >= startSlot && i <= endSlot) continue;  // Skip slots within segment
    for (int inp = 0; inp < slots[i].wiring.numInputs; inp++) {
      int srcIdx = slots[i].wiring.inputSourceIndices[inp];
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
    for (int inp = 0; inp < slots[i].wiring.numInputs; inp++) {
      int srcIdx = slots[i].wiring.inputSourceIndices[inp];
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

// ─── Type classification helpers ────────────────────────────────────────────

inline mlir::Type getElementType(mlir::Value val) {
  if (auto tensorTy = mlir::dyn_cast<mlir::RankedTensorType>(val.getType()))
    return tensorTy.getElementType();
  return val.getType();
}

inline bool isFloatType(mlir::Type type) {
  if (auto tensorTy = mlir::dyn_cast<mlir::RankedTensorType>(type))
    type = tensorTy.getElementType();
  return mlir::isa<mlir::FloatType>(type);
}

inline bool isIntegerType(mlir::Type type) {
  if (auto tensorTy = mlir::dyn_cast<mlir::RankedTensorType>(type))
    type = tensorTy.getElementType();
  return type.isSignlessInteger();
}

inline bool isBoolType(mlir::Type type) {
  if (auto tensorTy = mlir::dyn_cast<mlir::RankedTensorType>(type))
    type = tensorTy.getElementType();
  return type.isInteger(1);
}

inline int getFloatBitWidth(mlir::Type type) {
  if (auto ft = mlir::dyn_cast<mlir::FloatType>(type)) return ft.getWidth();
  return 0;
}

// ─── Universal type cast: cast any value to target element type ─────────────

inline mlir::Value castTo(mlir::OpBuilder& builder, mlir::Location loc,
                           mlir::Value val, mlir::Type targetElemType) {
  auto tensorTy = mlir::dyn_cast<mlir::RankedTensorType>(val.getType());
  if (!tensorTy) {
    // Scalar (non-tensor) cast path — e.g. scalar i64 loaded from a !tt.ptr<i64>
    // must be cast to f32 before use in arith.addf / arith.mulf etc.
    auto srcType = val.getType();
    if (srcType == targetElemType) return val;
    bool srcIsFloat = mlir::isa<mlir::FloatType>(srcType);
    bool dstIsFloat = mlir::isa<mlir::FloatType>(targetElemType);
    bool srcIsInt   = srcType.isIntOrIndex();
    bool dstIsInt   = targetElemType.isIntOrIndex();
    if (srcIsFloat && dstIsFloat) {
      int srcBits = mlir::cast<mlir::FloatType>(srcType).getWidth();
      int dstBits = mlir::cast<mlir::FloatType>(targetElemType).getWidth();
      if (dstBits > srcBits)
        return builder.create<mlir::arith::ExtFOp>(loc, targetElemType, val, nullptr);
      else
        return builder.create<mlir::arith::TruncFOp>(loc, targetElemType, val);
    } else if (srcIsInt && dstIsFloat) {
      int srcBits = srcType.getIntOrFloatBitWidth();
      int dstFloatBits = mlir::cast<mlir::FloatType>(targetElemType).getWidth();
      if (srcBits > 32 && dstFloatBits < 64) {
        // i64 → f64 → f32 to avoid precision loss
        auto f64Ty = builder.getF64Type();
        auto toF64 = builder.create<mlir::arith::SIToFPOp>(loc, f64Ty, val);
        return builder.create<mlir::arith::TruncFOp>(loc, targetElemType, toF64);
      }
      return builder.create<mlir::arith::SIToFPOp>(loc, targetElemType, val);
    } else if (srcIsFloat && dstIsInt) {
      return builder.create<mlir::arith::FPToSIOp>(loc, targetElemType, val);
    } else if (srcIsInt && dstIsInt) {
      int srcBits = srcType.getIntOrFloatBitWidth();
      int dstBits = targetElemType.getIntOrFloatBitWidth();
      if (srcBits == dstBits) return val;
      if (dstBits > srcBits)
        return builder.create<mlir::arith::ExtSIOp>(loc, targetElemType, val);
      else
        return builder.create<mlir::arith::TruncIOp>(loc, targetElemType, val);
    }
    // Unknown scalar type combination — return as-is and let MLIR verifier catch it
    return val;
  }
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
      int dstBits = targetElemType.getIntOrFloatBitWidth();
      if (dstBits > 32) {
        int srcFloatBits = getFloatBitWidth(srcElemType);
        if (srcFloatBits < 64) {
          auto f64Type = builder.getF64Type();
          auto f64TensorType = mlir::RankedTensorType::get(tensorTy.getShape(), f64Type);
          auto widened = builder.create<mlir::arith::ExtFOp>(loc, f64TensorType, val, nullptr);
          return builder.create<mlir::arith::FPToSIOp>(loc, targetTensorType, widened);
        } else {
          return builder.create<mlir::arith::FPToSIOp>(loc, targetTensorType, val);
        }
      } else {
        return builder.create<mlir::arith::FPToSIOp>(loc, targetTensorType, val);
      }
    }
  } else if (!srcIsFloat && dstIsFloat) {
    // integer/bool → float
    if (srcIsBool) {
      return builder.create<mlir::arith::UIToFPOp>(loc, targetTensorType, val);
    } else {
      int srcBits = srcElemType.getIntOrFloatBitWidth();
      int dstFloatBits = getFloatBitWidth(targetElemType);
      if (srcBits > 32) {
        if (dstFloatBits < 64) {
          auto f64Type = builder.getF64Type();
          auto f64TensorType = mlir::RankedTensorType::get(tensorTy.getShape(), f64Type);
          auto toF64 = builder.create<mlir::arith::SIToFPOp>(loc, f64TensorType, val);
          return builder.create<mlir::arith::TruncFOp>(loc, targetTensorType, toF64);
        } else {
          return builder.create<mlir::arith::SIToFPOp>(loc, targetTensorType, val);
        }
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
inline mlir::Value promoteToFloat(mlir::OpBuilder& builder, mlir::Location loc,
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
inline mlir::Type commonFloatType(mlir::OpBuilder& builder, mlir::Value lhs, mlir::Value rhs) {
  auto lhsElem = getElementType(lhs);
  auto rhsElem = getElementType(rhs);
  bool lhsF = mlir::isa<mlir::FloatType>(lhsElem);
  bool rhsF = mlir::isa<mlir::FloatType>(rhsElem);

  if (lhsF && rhsF) {
    int lhsBits = getFloatBitWidth(lhsElem);
    int rhsBits = getFloatBitWidth(rhsElem);
    int maxBits = lhsBits >= rhsBits ? lhsBits : rhsBits;
    // Match native precision: nd4j's float16/bfloat16 operators always promote
    // to float32 for computation (SD_NATIVE_HALFS is not defined). Promote to
    // at least f32 so Triton binary ops match native per-op precision.
    if (maxBits < 32) return builder.getF32Type();
    return lhsBits >= rhsBits ? lhsElem : rhsElem;
  } else if (lhsF) {
    return getFloatBitWidth(lhsElem) >= 32 ? lhsElem : builder.getF32Type();
  } else if (rhsF) {
    return getFloatBitWidth(rhsElem) >= 32 ? rhsElem : builder.getF32Type();
  }
  return builder.getF32Type();
}

// Find the common integer type for binary int ops
inline mlir::Type commonIntType(mlir::Value lhs, mlir::Value rhs) {
  auto lhsElem = getElementType(lhs);
  auto rhsElem = getElementType(rhs);
  int lhsBits = lhsElem.getIntOrFloatBitWidth();
  int rhsBits = rhsElem.getIntOrFloatBitWidth();
  return lhsBits >= rhsBits ? lhsElem : rhsElem;
}

}  // namespace ir_builder_internal
}  // namespace graph
}  // namespace sd

// Forward declaration for the multi-output normalization backward emitter.
// Implemented in TritonIRBuilder_emitters.cpp; called from TritonIRBuilder_module.cpp.
// Returns false if the op cannot be emitted (wrong opName/missing inputs).
//
// Inputs (loaded as padded row-sized tensors, matching the forward norm convention):
//   dy          — upstream gradient,      shape [paddedRowLen]
//   x           — forward input,          shape [paddedRowLen]
//   gamma       — scale (gamma/gain),     shape [paddedRowLen] (nullptr if absent)
//   mean        — precomputed row mean,   scalar mlir::Value (nullptr if absent)
//   invStd      — precomputed 1/std,      scalar mlir::Value (nullptr if absent)
//   epsilon     — small stabilizer for variance
//   logicalRowLen — actual un-padded row length (for correct reductions)
//
// Outputs written directly via tt.store to their respective buffer pointers:
//   dxPtr       — gradient w.r.t. input  (always required)
//   dgammaPtr   — gradient w.r.t. gamma  (nullptr → skip dgamma store)
//   dbetaPtr    — gradient w.r.t. beta   (nullptr → skip dbeta store)
//
// All pointers are raw base pointers (scalar mlir::Value of PointerType).
// The emitter constructs the per-row range and mask internally using rowBase.
namespace sd {
namespace graph {

bool emitNormalizationBackwardSection(
    mlir::OpBuilder& builder, mlir::Location loc,
    const std::string& opName,
    mlir::Value dy, mlir::Value x,
    mlir::Value gamma,
    mlir::Value mean, mlir::Value invStd,
    float epsilon,
    int64_t logicalRowLen,
    mlir::Value rowBase,
    mlir::Value dxPtr,
    mlir::Value dgammaPtr,
    mlir::Value dbetaPtr);

}  // namespace graph
}  // namespace sd

#endif  // HAVE_TRITON
#endif  // LIBND4J_TRITON_IR_BUILDER_INTERNAL_H

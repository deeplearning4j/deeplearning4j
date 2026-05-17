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
// TritonIRBuilder — Main module builders:
//   buildModule, buildSectionedModule, buildMatmulModule
//

#include <config.h>

#if HAVE_TRITON

#include <graph/gpu/TritonIRBuilder.h>
#include <graph/gpu/TritonIRBuilder_internal.h>
#include <graph/gpu/SectionTypeConfig.h>
#include <graph/DspAnalysisUtils.h>
#include <graph/DspDiagnostics.h>
#include <array/ArrayOptions.h>
#include <helpers/logger.h>
#include <helpers/shape.h>
#include <system/Environment.h>
#include <system/common.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <mutex>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

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

using namespace ir_builder_internal;

// Maximum number of direct function arguments before switching to indirect
// argument passing via a pointer array.
static constexpr int TRITON_DIRECT_ARG_LIMIT = 200;

// ── Thread-safe MLIR context factory ──────────────────────────────────────────
// MLIR's global DialectRegistry is NOT thread-safe for concurrent context
// creation + loadDialect calls from parallel compilation workers.
// Serialize all MLIRContext creation behind a single mutex.
static std::mutex& getMlirContextMutex() {
  static std::mutex mtx;
  return mtx;
}

static mlir::MLIRContext* createMlirContextWithDialects() {
  std::lock_guard<std::mutex> lock(getMlirContextMutex());
  auto* ctx = new mlir::MLIRContext();
  ctx->loadDialect<mlir::triton::TritonDialect>();
  ctx->loadDialect<mlir::arith::ArithDialect>();
  ctx->loadDialect<mlir::math::MathDialect>();
  ctx->loadDialect<mlir::scf::SCFDialect>();
  return ctx;
}

// ── View-producing op stride computation ────────────────────────────────────
// When a Triton section consumes the output of a view-producing op (permute,
// transpose, reshape_no_copy), the live NDArray is often null at compilation
// time (cleanup nulls outputSlots between calls), and cachedShapeInfoMap has
// C-contiguous strides (from shape calculation, not actual view layout).
//
// This helper computes the ACTUAL non-contiguous strides by inspecting the
// source op's type and parameters.  For example, permute([0,2,1]) on input
// strides [32,8,1] produces output strides [32,1,8].
//
// Returns empty vector if the op is not a view-producing op or strides can't
// be computed (caller should fall back to C-contiguous assumption).
static std::vector<LongType> computeViewStridesFromOp(
    NativeSlot* slots, int slotIdx, int totalSlots,
    NDArray** outputSlots, int totalOutputSlots,
    const std::unordered_map<int, const LongType*>& cachedShapeInfoMap) {

  const auto& opName = slots[slotIdx].ident.opName;

  // Only handle permute and transpose — these are the view ops that produce
  // non-contiguous strides.  reshape_no_copy changes shape but preserves
  // contiguity ordering, so C-contiguous strides from shape calc are correct.
  bool isPermute = (opName == "permute");
  bool isTranspose = (opName == "transpose");
  if (!isPermute && !isTranspose) return {};

  // Get the source op's input shape (the array being permuted/transposed)
  if (slots[slotIdx].wiring.numInputs < 1) return {};
  int inputSrcIdx = slots[slotIdx].wiring.inputSourceIndices[0];

  // Resolve the input shape
  std::vector<LongType> inputShape;
  if (inputSrcIdx >= 0 && inputSrcIdx < totalOutputSlots &&
      outputSlots && outputSlots[inputSrcIdx]) {
    auto& arr = *outputSlots[inputSrcIdx];
    for (int d = 0; d < arr.rankOf(); d++)
      inputShape.push_back(arr.sizeAt(d));
  } else if (inputSrcIdx >= 0) {
    auto cit = cachedShapeInfoMap.find(inputSrcIdx);
    if (cit != cachedShapeInfoMap.end() && cit->second) {
      LongType rank = shape::rank(cit->second);
      for (int d = 0; d < rank; d++)
        inputShape.push_back(shape::shapeOf(cit->second)[d]);
    }
  }
  if (inputShape.empty()) return {};

  int rank = static_cast<int>(inputShape.size());

  // Compute C-contiguous strides of the INPUT array
  std::vector<LongType> inputStrides(rank);
  inputStrides[rank - 1] = 1;
  for (int d = rank - 2; d >= 0; d--) {
    inputStrides[d] = inputStrides[d + 1] * inputShape[d + 1];
  }

  // Get the permutation order
  std::vector<int> perm;
  if (isPermute) {
    // permute: iArgs contains the permutation order
    for (int i = 0; i < slots[slotIdx].args.numIArgs && i < rank; i++) {
      perm.push_back(static_cast<int>(slots[slotIdx].args.iArgs[i]));
    }
  } else if (isTranspose) {
    // transpose with no iArgs = full reverse (equivalent to permute(N-1, N-2, ..., 0))
    if (slots[slotIdx].args.numIArgs == 0) {
      for (int d = rank - 1; d >= 0; d--) perm.push_back(d);
    } else {
      for (int i = 0; i < slots[slotIdx].args.numIArgs && i < rank; i++) {
        perm.push_back(static_cast<int>(slots[slotIdx].args.iArgs[i]));
      }
    }
  }

  if (static_cast<int>(perm.size()) != rank) return {};

  // Output strides = input strides permuted by the permutation order
  // output_stride[i] = input_stride[perm[i]]
  std::vector<LongType> outputStrides(rank);
  for (int d = 0; d < rank; d++) {
    if (perm[d] < 0 || perm[d] >= rank) return {};  // Invalid perm
    outputStrides[d] = inputStrides[perm[d]];
  }

  DSP_DIAG(COMPILE, "computeViewStridesFromOp: slot %d op='%s' inputShape=[%s] "
           "inputStrides=[%s] perm=[%s] outputStrides=[%s]",
           slotIdx, opName.c_str(),
           [&]() { std::string s; for (int d = 0; d < rank; d++) { if (d) s += ","; s += std::to_string(inputShape[d]); } return s; }().c_str(),
           [&]() { std::string s; for (int d = 0; d < rank; d++) { if (d) s += ","; s += std::to_string(inputStrides[d]); } return s; }().c_str(),
           [&]() { std::string s; for (int d = 0; d < rank; d++) { if (d) s += ","; s += std::to_string(perm[d]); } return s; }().c_str(),
           [&]() { std::string s; for (int d = 0; d < rank; d++) { if (d) s += ","; s += std::to_string(outputStrides[d]); } return s; }().c_str());

  return outputStrides;
}

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
    DSP_DIAG(COMPILE, "TritonIRBuilder::buildModule: segment [%d-%d] failed pre-check: %s",
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
  // attention, data movement, convolution, constant generation, or shape
  // manipulation ops that need their own grid mapping and cannot be fused
  // into the 1D element-wise skeleton.
  // ALL shape manipulation ops (permute, transpose, reshape, squeeze,
  // expand_dims, flatten) are treated as non-elementwise because they cause
  // zero-output bugs when SSA-forwarded in the 1D skeleton due to buffer
  // pointer resolution issues with ND4J view-based DataBuffers.
  {
    bool hasNonElementwiseOps = false;
    for (int i = startSlot; i <= endSlot; i++) {
      auto cat = getOpCategory(slots[i].ident.opName);
      if (cat == TritonOpCategory::MATMUL || cat == TritonOpCategory::FUSED_ATTENTION ||
          cat == TritonOpCategory::DATA_MOVEMENT || cat == TritonOpCategory::CONVOLUTION ||
          cat == TritonOpCategory::SHAPE_MANIPULATION) {
        hasNonElementwiseOps = true;
        break;
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
  DSP_DIAG(COMPILE, "TritonIRBuilder::buildModule: segment [%d-%d] (%d ops), pattern=%d",
            startSlot, endSlot, segSize, static_cast<int>(pattern));
  result.kernelName = generateKernelName(slots, startSlot, endSlot);
  DSP_DIAG(COMPILE, "TritonIRBuilder::buildModule: kernel name generated, collecting categories...");

  // Build cached shape info map for shape resolution when outputSlots may be released.
  // Iterate ALL slots (not just current segment) so cross-segment inputs are included.
  // Also include slots where shapeCacheValid() is false but cachedOutputShapes is
  // non-empty — this handles cases where cleanup reset the slot state but shapes
  // were populated from a prior execution.
  std::unordered_map<int, const LongType*> cachedShapeInfoMap;
  for (int i = 0; i < totalSlots; i++) {
    if (!slots[i].shapeCache.cachedOutputShapes.empty()) {
      for (int o = 0; o < slots[i].wiring.numOutputs; o++) {
        int outIdx = slots[i].wiring.outputSlotIndices[o];
        if (outIdx >= 0 && o < static_cast<int>(slots[i].shapeCache.cachedOutputShapes.size()) &&
            slots[i].shapeCache.cachedOutputShapes[o] != nullptr) {
          cachedShapeInfoMap[outIdx] = slots[i].shapeCache.cachedOutputShapes[o];
        }
      }
    }
  }

  // Shape resolution helpers.
  // Prefer the live warmup arrays when they exist: they reflect the actual runtime
  // metadata after slot-by-slot execution, while cached slot metadata can lag for
  // dynamic INT64 helper tensors that are later consumed by fused Triton kernels.
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
    if (srcIdx < totalOutputSlots && outputSlots && outputSlots[srcIdx]) {
      auto& arr = *outputSlots[srcIdx];
      std::vector<LongType> s(arr.rankOf());
      for (int d = 0; d < arr.rankOf(); d++) s[d] = arr.sizeAt(d);
      return s;
    }
    auto cit = cachedShapeInfoMap.find(srcIdx);
    if (cit != cachedShapeInfoMap.end() && cit->second) {
      LongType rank = shape::rank(cit->second);
      std::vector<LongType> s(rank);
      for (int d = 0; d < rank; d++) s[d] = shape::shapeOf(cit->second)[d];
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
    if (srcIdx < totalOutputSlots && outputSlots && outputSlots[srcIdx])
      return outputSlots[srcIdx]->dataType();
    auto cit = cachedShapeInfoMap.find(srcIdx);
    if (cit != cachedShapeInfoMap.end() && cit->second)
      return ArrayOptions::dataType(cit->second);
    return FLOAT32;
  };

  // Collect op categories and shapes for tile config
  std::vector<TritonOpCategory> categories;
  std::vector<std::vector<LongType>> shapes;

  for (int i = startSlot; i <= endSlot; i++) {
    auto cat = getOpCategory(slots[i].ident.opName);
    // Every op must be in the table. getOpCategory() throws if missing.
    categories.push_back(cat);

    if (slots[i].wiring.numOutputs > 0) {
      int outIdx = slots[i].wiring.outputSlotIndices[0];
      shapes.push_back(resolveShapeLocal(outIdx));
    } else {
      shapes.push_back({});
    }
  }
  DSP_DIAG(COMPILE, "TritonIRBuilder::buildModule: collected %d categories, selecting tile config...",
            (int)categories.size());

  // Select tile configuration
  int blockSize, numWarps, numStages;
  selectTileConfig(categories, shapes, blockSize, numWarps, numStages);
  result.numWarps = numWarps;
  result.numStages = numStages;

  // Create MLIR context and register dialects (thread-safe)
  auto mlirContext = createMlirContextWithDialects();

  mlir::OpBuilder builder(mlirContext);
  auto loc = builder.getUnknownLoc();

  // Create module
  auto moduleOp = mlir::ModuleOp::create(loc);
  builder.setInsertionPointToEnd(moduleOp.getBody());

  // ── Collect unique buffer references ──
  std::unordered_set<int> internalSlotOutputs;
  for (int i = startSlot; i <= endSlot; i++) {
    for (int o = 0; o < slots[i].wiring.numOutputs; o++) {
      internalSlotOutputs.insert(slots[i].wiring.outputSlotIndices[o]);
    }
  }

  // ── Force-external inputs for ROPE ops ──
  // ROPE cos/sin inputs require kernel arg pointers even when produced by upstream
  // ops in the same segment (internal intermediates). This is because the ROPE emitter
  // does custom 2D indexing on cos/sin buffers (by sequence position and head dimension)
  // that cannot be expressed through the 1D SSA pipeline.
  std::unordered_set<int> forceExternalInputs;
  for (int i = startSlot; i <= endSlot; i++) {
    auto cat = getOpCategory(slots[i].ident.opName);
    if (cat == TritonOpCategory::ROPE && slots[i].wiring.numInputs >= 3) {
      int cosSrc = slots[i].wiring.inputSourceIndices[1];
      int sinSrc = slots[i].wiring.inputSourceIndices[2];
      if (internalSlotOutputs.count(cosSrc)) {
        forceExternalInputs.insert(cosSrc);
        DSP_DIAG(COMPILE, "TritonIRBuilder: ROPE '%s' at slot %d: forcing internal cos src %d to kernel arg",
                  slots[i].ident.opName.c_str(), i, cosSrc);
      }
      if (internalSlotOutputs.count(sinSrc)) {
        forceExternalInputs.insert(sinSrc);
        DSP_DIAG(COMPILE, "TritonIRBuilder: ROPE '%s' at slot %d: forcing internal sin src %d to kernel arg",
                  slots[i].ident.opName.c_str(), i, sinSrc);
      }
    }
  }

  // ── Frozen constant slot handling ──
  // Frozen constant slots have outputs that were computed during the SBS warmup
  // and must NOT be recomputed by Triton (their inputs may have changed since
  // warmup, but the frozen value is correct by definition). Instead, frozen
  // slot outputs are treated as INPUTS to the kernel — loaded via tt.load from
  // the GPU buffer that already holds the correct warmup value. The op itself
  // is skipped in the IR emission loop (ssaValues already populated by tt.load).
  std::unordered_set<int> frozenSlotOutputs;
  for (int i = startSlot; i <= endSlot; i++) {
    if (slots[i].frozenConstantSlot()) {
      for (int o = 0; o < slots[i].wiring.numOutputs; o++) {
        frozenSlotOutputs.insert(slots[i].wiring.outputSlotIndices[o]);
      }
    }
  }
  if (!frozenSlotOutputs.empty()) {
    DSP_DIAG(COMPILE, "TritonIRBuilder::buildModule: %d frozen constant outputs in [%d-%d] "
              "will be loaded as inputs (not recomputed)",
              static_cast<int>(frozenSlotOutputs.size()), startSlot, endSlot);
  }

  // Diagnostic: print slot wiring for small sub-kernels (or first sub-kernel in segment)
  if (DSP_DIAG_ENABLED(COMPILE) && endSlot - startSlot < 100) {
    for (int i = startSlot; i <= endSlot; i++) {
      std::string inSrcs;
      for (int inp = 0; inp < slots[i].wiring.numInputs; inp++) {
        if (!inSrcs.empty()) inSrcs += ",";
        inSrcs += std::to_string(slots[i].wiring.inputSourceIndices[inp]);
      }
      std::string outSlots2;
      for (int o = 0; o < slots[i].wiring.numOutputs; o++) {
        if (!outSlots2.empty()) outSlots2 += ",";
        outSlots2 += std::to_string(slots[i].wiring.outputSlotIndices[o]);
      }
      bool allInternal = true;
      for (int inp = 0; inp < slots[i].wiring.numInputs; inp++) {
        int src = slots[i].wiring.inputSourceIndices[inp];
        if (src < 0 || !internalSlotOutputs.count(src)) { allInternal = false; break; }
      }
      // Also show resolved array info for non-internal inputs
      std::string inputInfo;
      for (int inp = 0; inp < slots[i].wiring.numInputs; inp++) {
        int src = slots[i].wiring.inputSourceIndices[inp];
        if (src < 0) {
          int extIdx = -(src + 1);
          if (extIdx < numExternalInputs && externalInputs[extIdx]) {
            auto* a = externalInputs[extIdx];
            inputInfo += " ext[" + std::to_string(src) + "]:len=" +
                         std::to_string(a->lengthOf()) + ",dt=" +
                         std::to_string(static_cast<int>(a->dataType()));
          }
        } else if (!internalSlotOutputs.count(src)) {
          if (src < totalSlots && outputSlots && outputSlots[src]) {
            auto* a = outputSlots[src];
            inputInfo += " cross[" + std::to_string(src) + "]:len=" +
                         std::to_string(a->lengthOf()) + ",dt=" +
                         std::to_string(static_cast<int>(a->dataType()));
          }
        }
      }
      DSP_DIAG_SLOT(COMPILE, i, "SLOT WIRING: [%d-%d] slot %d op='%s' inputs=[%s] outputs=[%s] allInternal=%d%s",
                startSlot, endSlot, i, slots[i].ident.opName.c_str(), inSrcs.c_str(),
                outSlots2.c_str(), allInternal ? 1 : 0, inputInfo.c_str());
    }
  }

  // Inputs: external inputs or outputs from slots BEFORE this segment
  std::vector<TritonKernelArg> inputArgs;
  std::unordered_set<int> seenInputs;

  // Pre-scan: identify external inputs that are ONLY consumed by CONST_GEN ops.
  // CONST_GEN ops (shape_of, create, zeros_like, etc.) generate output from metadata
  // (shape/dtype/tArgs) without reading input data buffers. Their external inputs may
  // have null DataBuffers at execution time (freed by Java GC after metadata was captured
  // at compile time). Skip adding these as kernel args since the generated MLIR won't
  // actually load from them.
  std::unordered_set<int> constGenOnlyInputs;
  {
    std::unordered_map<int, bool> inputHasNonConstGenConsumer;
    for (int i = startSlot; i <= endSlot; i++) {
      auto cat = getOpCategory(slots[i].ident.opName);
      bool isConstGen = (cat == TritonOpCategory::CONSTANT_GENERATION);
      for (int inp = 0; inp < slots[i].wiring.numInputs; inp++) {
        int srcIdx = slots[i].wiring.inputSourceIndices[inp];
        if (srcIdx >= 0) continue;  // Only care about external inputs here
        auto it = inputHasNonConstGenConsumer.find(srcIdx);
        if (it == inputHasNonConstGenConsumer.end()) {
          inputHasNonConstGenConsumer[srcIdx] = !isConstGen;
        } else if (!isConstGen) {
          it->second = true;
        }
      }
    }
    for (auto& kv : inputHasNonConstGenConsumer) {
      if (!kv.second) constGenOnlyInputs.insert(kv.first);
    }
    if (!constGenOnlyInputs.empty()) {
      DSP_DIAG(COMPILE, "TritonIRBuilder::buildModule: skipping %d external inputs consumed only by CONST_GEN ops",
                static_cast<int>(constGenOnlyInputs.size()));
    }
  }

  for (int i = startSlot; i <= endSlot; i++) {
    for (int inp = 0; inp < slots[i].wiring.numInputs; inp++) {
      int srcIdx = slots[i].wiring.inputSourceIndices[inp];
      if (seenInputs.count(srcIdx)) continue;

      if (srcIdx < 0) {
        seenInputs.insert(srcIdx);
        // Skip external inputs only consumed by CONST_GEN ops — they don't need data buffers
        if (constGenOnlyInputs.count(srcIdx)) continue;
        int extIdx = -(srcIdx + 1);
        if (extIdx < numExternalInputs && externalInputs[extIdx]) {
          TritonKernelArg arg;
          arg.slotIndex = srcIdx;
          arg.outputIndex = 0;
          arg.isOutput = false;
          arg.dtype = externalInputs[extIdx]->dataType();
          auto& arr = *externalInputs[extIdx];
          for (int d = 0; d < arr.rankOf(); d++) {
            arg.shape.push_back(arr.sizeAt(d));
            arg.strides.push_back(arr.strideAt(d));
          }
          inputArgs.push_back(arg);
          // Diagnostic: print compile-time values of small external inputs
          if (DSP_DIAG_ENABLED(COMPILE) && arr.lengthOf() <= 10) {
            arr.syncToHost();
            std::string vals;
            for (LongType e = 0; e < arr.lengthOf(); e++) {
              if (e > 0) vals += ",";
              char buf[64];
              if (arr.dataType() == INT64 || arr.dataType() == INT32) {
                snprintf(buf, sizeof(buf), "%lld", (long long)arr.e<LongType>(e));
              } else {
                snprintf(buf, sizeof(buf), "%.6f", arr.e<double>(e));
              }
              vals += buf;
            }
            DSP_DIAG(COMPILE, "COMPILE-TIME INPUT: segment [%d-%d] ext slot %d: values=[%s] len=%lld dt=%d",
                      startSlot, endSlot, srcIdx, vals.c_str(), (long long)arr.lengthOf(),
                      static_cast<int>(arr.dataType()));
          }
        }
      } else if (!internalSlotOutputs.count(srcIdx) || forceExternalInputs.count(srcIdx)) {
        seenInputs.insert(srcIdx);
        auto shape = resolveShapeLocal(srcIdx);
        auto dtype = resolveDtypeLocal(srcIdx);
        bool hasLiveArr = (srcIdx < totalOutputSlots && outputSlots && outputSlots[srcIdx]);
        // shape.empty() can mean "no shape found" OR "scalar (rank 0)".
        // Check cachedShapeInfoMap for valid scalar shapes to avoid false negatives.
        bool hasShape = !shape.empty() || cachedShapeInfoMap.count(srcIdx);
        if (hasLiveArr || hasShape) {
          TritonKernelArg arg;
          arg.slotIndex = srcIdx;
          arg.outputIndex = 0;
          arg.isOutput = false;
          arg.dtype = dtype;
          arg.shape = shape;
          // Extract actual strides from live NDArray for non-contiguous view detection
          if (hasLiveArr) {
            auto& arr = *outputSlots[srcIdx];
            for (int d = 0; d < arr.rankOf(); d++) {
              arg.strides.push_back(arr.strideAt(d));
            }
          } else {
            // Live array not available (cleanup nulled it between calls).
            // For view-producing ops (permute/transpose), compute actual strides
            // from the op's parameters — cachedShapeInfoMap has C-contiguous strides
            // from shape calculation, which is WRONG for views.
            if (srcIdx >= 0 && srcIdx < totalSlots) {
              auto viewStrides = computeViewStridesFromOp(
                  slots, srcIdx, totalSlots, outputSlots, totalOutputSlots, cachedShapeInfoMap);
              if (!viewStrides.empty()) {
                arg.strides = viewStrides;
              }
            }
          }
          inputArgs.push_back(arg);
        } else {
          // Cross-segment input has neither live array nor cached shape.
          // This is a compilation error — the kernel cannot run without this input.
          DSP_DIAG(COMPILE, "TritonIRBuilder::buildModule: MISSING cross-segment input slot %d "
                    "for segment [%d-%d] (hasLiveArr=%d, shape.empty=%d, cachedShapeInfoMap.count=%d, "
                    "srcIdx<totalSlots=%d, srcIdx<totalOutputSlots=%d)",
                    srcIdx, startSlot, endSlot, hasLiveArr ? 1 : 0, shape.empty() ? 1 : 0,
                    static_cast<int>(cachedShapeInfoMap.count(srcIdx)),
                    srcIdx < totalSlots ? 1 : 0, srcIdx < totalOutputSlots ? 1 : 0);
          // Scan all slots for the producing op to diagnose why shape is missing
          for (int si2 = 0; si2 < totalSlots; si2++) {
            for (int o2 = 0; o2 < slots[si2].wiring.numOutputs; o2++) {
              if (slots[si2].wiring.outputSlotIndices[o2] == srcIdx) {
                DSP_DIAG(COMPILE, "  -> produced by op[%d] '%s' state=%d shapeCacheValid=%d "
                          "cachedOutputShapes.size=%d numOutputs=%d",
                          si2, slots[si2].ident.opName.c_str(), slots[si2].slotPhase.toLegacyCode(),
                          slots[si2].shapeCacheValid() ? 1 : 0,
                          static_cast<int>(slots[si2].shapeCache.cachedOutputShapes.size()),
                          slots[si2].wiring.numOutputs);
              }
            }
          }
          // Fail this compilation — return empty module. The segment will
          // fall back to slot-by-slot or raw CUDA graph capture.
          return result;
        }
      }
    }
  }

  // Add frozen constant slot outputs as INPUT args.
  // These slots' output buffers already contain the correct warmup values on GPU.
  // The kernel will tt.load them instead of recomputing, preserving the frozen values.
  for (int outIdx : frozenSlotOutputs) {
    if (seenInputs.count(outIdx)) continue;  // Already added as input (cross-segment ref)
    seenInputs.insert(outIdx);
    auto shape = resolveShapeLocal(outIdx);
    auto dtype = resolveDtypeLocal(outIdx);
    bool hasLiveArr = (outIdx < totalOutputSlots && outputSlots && outputSlots[outIdx]);
    bool hasShape = !shape.empty() || cachedShapeInfoMap.count(outIdx);
    if (!hasLiveArr && !hasShape) {
      // Frozen constant output has no live array and no cached shape — this happens
      // when post-freeze recompile occurs after releaseGpuIntermediates freed the slot.
      // The frozen constant cannot be loaded without shape info. Fail the build.
      DSP_DIAG(FALLBACK, "TritonIRBuilder: frozen constant output slot %d has no live array "
               "and no cached shape — cannot generate tt.load input arg. Failing module build.",
               outIdx);
      return TritonIRModule();  // valid = false
    }
    TritonKernelArg arg;
    arg.slotIndex = outIdx;
    arg.outputIndex = 0;
    arg.isOutput = false;
    arg.dtype = dtype;
    arg.shape = shape;
    // Extract actual strides for frozen slot outputs (may be views)
    if (hasLiveArr) {
      auto& arr = *outputSlots[outIdx];
      for (int d = 0; d < arr.rankOf(); d++) {
        arg.strides.push_back(arr.strideAt(d));
      }
    } else if (outIdx >= 0 && outIdx < totalSlots) {
      auto viewStrides = computeViewStridesFromOp(
          slots, outIdx, totalSlots, outputSlots, totalOutputSlots, cachedShapeInfoMap);
      if (!viewStrides.empty()) {
        arg.strides = viewStrides;
      }
    }
    inputArgs.push_back(arg);
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
    auto cat = getOpCategory(slots[i].ident.opName);
    if (cat == TritonOpCategory::REDUCTION) {
      for (int inp = 0; inp < slots[i].wiring.numInputs; inp++) {
        int srcIdx = slots[i].wiring.inputSourceIndices[inp];
        if (srcIdx >= 0 && internalSlotOutputs.count(srcIdx) && !externalOutputs.count(srcIdx)) {
          reductionInputSlots.insert(srcIdx);
        }
      }
    }
  }

  // Build set of input buffer addresses for aliasing detection.
  auto inputBufferAddrs = dsp::buildInputBufferAddressSet(
      inputArgs, outputSlots, totalOutputSlots, externalInputs, numExternalInputs);

  std::vector<TritonKernelArg> outputArgs;
  {
    std::unordered_set<int> seenOutputSlots;
    int skippedInternal = 0;
    int skippedAliased = 0;
    for (int i = startSlot; i <= endSlot; i++) {
      for (int o = 0; o < slots[i].wiring.numOutputs; o++) {
        int outIdx = slots[i].wiring.outputSlotIndices[o];
        if (outIdx < 0 || outIdx >= totalOutputSlots) continue;
        if (seenOutputSlots.count(outIdx)) continue;  // Deduplicate
        seenOutputSlots.insert(outIdx);
        // Frozen constant outputs are loaded as inputs — never stored by the kernel.
        if (frozenSlotOutputs.count(outIdx)) {
          skippedInternal++;
          continue;
        }
        if (!externalOutputs.count(outIdx) && !reductionInputSlots.count(outIdx)) {
          skippedInternal++;
          continue;  // Purely internal — SSA forwarded
        }

        // Skip outputs whose GPU buffer aliases an input buffer.
        if (dsp::isOutputAliasedWithInput(outIdx, outputSlots, totalOutputSlots, inputBufferAddrs)) {
          skippedAliased++;
          DSP_DIAG(COMPILE, "TritonIRBuilder: skipping aliased output slot %d in segment [%d-%d]",
                   outIdx, startSlot, endSlot);
          continue;
        }

        TritonKernelArg arg;
        arg.slotIndex = outIdx;
        arg.outputIndex = o;
        arg.isOutput = true;
        if (outputSlots && outIdx < totalOutputSlots && outputSlots[outIdx]) {
          arg.dtype = outputSlots[outIdx]->dataType();
          auto& arr = *outputSlots[outIdx];
          for (int d = 0; d < arr.rankOf(); d++) {
            arg.shape.push_back(arr.sizeAt(d));
            arg.strides.push_back(arr.strideAt(d));
          }
        } else {
          // Fall back to cached shape info when live array is not available
          auto cit = cachedShapeInfoMap.find(outIdx);
          if (cit != cachedShapeInfoMap.end() && cit->second) {
            arg.dtype = ArrayOptions::dataType(cit->second);
            LongType rank = shape::rank(cit->second);
            for (int d = 0; d < rank; d++) arg.shape.push_back(shape::shapeOf(cit->second)[d]);
          } else {
            // No live array and no cached shape — resolve dtype/shape from the producing op.
            // This happens when the output slot belongs to a view-capable op (reshape,
            // strided_slice, etc.) that doesn't pre-allocate its output. The producing op
            // (e.g. cast) needs the correct dtype to generate proper store instructions.
            auto producerCat = getOpCategory(slots[i].ident.opName);
            if (producerCat == TritonOpCategory::CAST && slots[i].args.numIArgs > 0 && slots[i].args.iArgs) {
              // Cast ops store target dtype in iArgs[0]
              arg.dtype = static_cast<DataType>(slots[i].args.iArgs[0]);
              DSP_DIAG(COMPILE, "TritonIRBuilder: output slot %d dtype resolved from cast iArgs[0]=%lld → dtype=%d",
                       outIdx, (long long)slots[i].args.iArgs[0], (int)arg.dtype);
            } else {
              // For non-cast ops, output dtype matches the primary input's dtype
              if (slots[i].wiring.numInputs > 0) {
                int inputSrc = slots[i].wiring.inputSourceIndices[0];
                arg.dtype = resolveDtypeLocal(inputSrc);
              }
            }
            // Derive shape from the primary input (most ops preserve shape)
            if (arg.shape.empty() && slots[i].wiring.numInputs > 0) {
              int inputSrc = slots[i].wiring.inputSourceIndices[0];
              auto inputShape = resolveShapeLocal(inputSrc);
              if (!inputShape.empty()) {
                arg.shape = inputShape;
              }
            }
          }
        }
        outputArgs.push_back(arg);
      }
    }
    if (skippedInternal > 0 || skippedAliased > 0) {
      DSP_DIAG(FUSION, "TritonIRBuilder::buildModule: eliminated %d internal + %d aliased outputs, "
                "keeping %d external",
                skippedInternal, skippedAliased, (int)outputArgs.size());
    }
  }

  // Combine: inputs first, then outputs
  result.args.insert(result.args.end(), inputArgs.begin(), inputArgs.end());
  result.args.insert(result.args.end(), outputArgs.begin(), outputArgs.end());

  int totalBufferArgs = static_cast<int>(result.args.size());
  bool useIndirectArgs = (totalBufferArgs + 1) > TRITON_DIRECT_ARG_LIMIT;  // +1 for n_elements
  // When CUDA graph capture is enabled, force indirect arg passing for ALL kernels.
  // Direct-arg kernels bake buffer pointers into CUDA graph kernel nodes, making them
  // impossible to update on replay (refreshArgTablesForReplay skips direct-arg kernels).
  // Indirect args use a pinned host buffer → H2D memcpy → device arg table pattern,
  // where the H2D memcpy is a graph node that copies fresh pointers each replay.
  if (!useIndirectArgs && sd::Environment::getInstance().tritonGraphCapture()) {
    useIndirectArgs = true;
    DSP_DIAG(COMPILE, "TritonIRBuilder::buildModule: forcing INDIRECT arg passing for graph capture compatibility "
              "(%d buffer args)", totalBufferArgs);
  }

  DSP_DIAG(COMPILE, "TritonIRBuilder::buildModule: %d input args, %d output args, %d total buffer args%s",
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

  DSP_DIAG(COMPILE, "TritonIRBuilder::buildModule: creating MLIR function with %d params (%d buffer args)...",
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
    DSP_DIAG(COMPILE, "TritonIRBuilder::buildModule: unpacked %d buffer pointers from indirect arg array",
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

  DSP_DIAG(COMPILE, "TritonIRBuilder::buildModule: MLIR function created, building kernel body...");

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

  // Segmented reductions require a SINGLE grid block: the bar.sync barrier
  // used to synchronize intermediate buffer writes before the reduction loop
  // is block-local and cannot synchronize across grid blocks.  With dynamic
  // grid the launch would create ceil(nElements/BLOCK_SIZE) blocks, and
  // blocks that finish writing their slice of the intermediate would race
  // ahead into the reduction loop before other blocks finish, reading stale
  // or zeroed data for the unwritten portion.
  bool hasReduction = std::find(categories.begin(), categories.end(),
      TritonOpCategory::REDUCTION) != categories.end();
  bool hasNormalization = std::find(categories.begin(), categories.end(),
      TritonOpCategory::NORMALIZATION) != categories.end();
  // Track multi-row normalization state for row-indexed addressing
  int64_t normLogicalRowLen = 0;  // logical elements per row (last dim)
  int64_t normNumRows = 0;       // number of rows (product of leading dims)
  int64_t normPaddedRowLen = 0;  // paddedRowLen for normalization
  bool normMultiRow = false;     // true if normalization needs per-row processing

  if (hasReduction || hasNormalization) {
    result.useDynamicGrid = false;

    // For normalization-only kernels with multi-row inputs (seqLen > 1),
    // each block processes one row: blockSize = paddedRowLen, gridX = numRows.
    // Without this, blockSize = totalPaddedElements but the emitter only produces
    // paddedRowLen-sized tensors, causing MLIR type mismatch on tt.store.
    if (hasNormalization && !hasReduction && !hasMatmul) {
      for (int i = startSlot; i <= endSlot; i++) {
        if (slots[i].wiring.numInputs >= 1) {
          auto inputShape = resolveShapeLocal(slots[i].wiring.inputSourceIndices[0]);
          if (!inputShape.empty()) {
            int64_t rowLen = inputShape.back();
            int64_t totalElements = 1;
            for (auto d : inputShape) totalElements *= static_cast<int64_t>(d);
            int64_t nRows = totalElements / rowLen;
            if (nRows > 1) {
              normLogicalRowLen = rowLen;
              normNumRows = nRows;
              normPaddedRowLen = 1;
              while (normPaddedRowLen < rowLen) normPaddedRowLen <<= 1;
              normMultiRow = true;
              blockSize = static_cast<int>(std::min(normPaddedRowLen, (int64_t)4096));
              result.gridX = static_cast<int>(nRows);
              DSP_DIAG(COMPILE, "TritonIRBuilder::buildModule: multi-row normalization "
                        "blockSize=%d (paddedRowLen), gridX=%d (numRows), logicalRowLen=%lld",
                        blockSize, result.gridX, (long long)normLogicalRowLen);
            }
          }
          break;  // Only check first normalization op
        }
      }
    }
    if (!normMultiRow) {
      result.gridX = 1;
    }
    DSP_DIAG(COMPILE, "TritonIRBuilder::buildModule: %s grid for %s (BLOCK_SIZE=%d, gridX=%d)",
              normMultiRow ? "multi-block" : "single-block",
              hasNormalization ? "normalization" : "segmented reduction",
              blockSize, result.gridX);
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

  // Outputs that were stored inline by normalization (skip in generic store loop)
  std::unordered_set<int> normInlineStoredOutputs;

  // Map: kernel arg index -> slotIndex for reverse lookup
  std::unordered_map<int, int> slotToArgIdx;
  for (int a = 0; a < static_cast<int>(result.args.size()); a++) {
    slotToArgIdx[result.args[a].slotIndex] = a;
  }

  // 2b: Load inputs — tt.load for each external input arg
  // For segments with normalization ops, skip preloading external inputs.
  // Normalization handlers will load them directly with actual tensor shapes
  // (preloading flattens to blockSize which breaks reduction semantics).
  bool skipExternalPreload = hasNormalization;
  DSP_DIAG(COMPILE, "TritonIRBuilder::buildModule: skipExternalPreload=%d hasNormalization=%d inputArgs.size=%d",
            skipExternalPreload ? 1 : 0, hasNormalization ? 1 : 0, (int)inputArgs.size());
  
  // Compute max output element count and reference shape for broadcasting detection
  LongType maxOutputElements = 0;
  std::vector<LongType> refOutputShape;  // Shape with most elements (used as broadcast reference)
  for (auto& oarg : outputArgs) {
    LongType oElems = 1;
    for (auto d : oarg.shape) oElems *= d;
    if (oElems > maxOutputElements) {
      maxOutputElements = oElems;
      refOutputShape = oarg.shape;
    }
  }
  // Fallback: if output shapes are unavailable (empty), use max input element count/shape
  if (maxOutputElements <= 1 && !skipExternalPreload) {
    for (auto& iarg : inputArgs) {
      LongType iElems = 1;
      for (auto d : iarg.shape) iElems *= d;
      if (iElems > maxOutputElements) {
        maxOutputElements = iElems;
        refOutputShape = iarg.shape;
      }
    }
  }
  for (int a = 0; a < static_cast<int>(inputArgs.size()); a++) {
    auto& arg = inputArgs[a];
    
    // Skip preloading for ALL inputs in normalization segments
    // (normalization handlers will load them directly with proper masking)
    if (skipExternalPreload) {
      DSP_DIAG(COMPILE, "TritonIRBuilder::buildModule: SKIPPING preload for normalization input arg %d slotIndex=%d",
                a, arg.slotIndex);
      continue;
    }
    
    auto funcArg = getBufferArg(a);  // tt.ptr<elemType>

    auto elemType = getMLIRType(builder, arg.dtype);
    auto ptrType = mlir::triton::PointerType::get(elemType, 1);
    auto ptrTensorType = mlir::RankedTensorType::get({blockSize}, ptrType);
    auto dataTensorType = mlir::RankedTensorType::get({blockSize}, elemType);

    // Compute this input's total element count for broadcast-aware indexing
    LongType inputElements = 1;
    for (auto d : arg.shape) inputElements *= d;

    mlir::Value loadOffsets = offsets;
    mlir::Value loadMask = mask;
    // Non-contiguous view detection: if this input has non-C-contiguous strides
    // (e.g., from a permute view), we MUST use stride-based decomposition even
    // when no broadcasting is needed. The flat offset path assumes C-contiguous.
    bool inputIsNonContiguous = arg.isNonContiguous();
    if (inputIsNonContiguous) {
      DSP_DIAG(COMPILE, "TritonIRBuilder: input slot %d is NON-CONTIGUOUS (view), "
               "forcing stride-based indexing", arg.slotIndex);
    }
    if ((inputElements > 0 && inputElements < maxOutputElements && !refOutputShape.empty())
        || (inputIsNonContiguous && !refOutputShape.empty())) {
      // Broadcasting or non-contiguous input requires stride-based indexing.
      // Determine if simple modular indexing suffices (broadcast on outermost dims only)
      // or if we need stride-based indexing (broadcast on inner dims, or non-contiguous).

      // Left-pad input shape with 1s to match output rank.
      // When inRank > outRank, take only the RIGHTMOST outRank dims from input.
      int outRank = static_cast<int>(refOutputShape.size());
      int inRank = static_cast<int>(arg.shape.size());
      std::vector<LongType> paddedInputShape(outRank, 1);
      for (int d = 0; d < outRank; d++) {
        int srcDim = d + (inRank - outRank);  // maps output dim → input dim
        if (srcDim >= 0 && srcDim < inRank) {
          paddedInputShape[d] = arg.shape[srcDim];
        }
        // else: stays 1 (left-pad for inRank < outRank)
      }

      // Check if broadcast is only on outermost dimensions (simple case)
      // Simple = all broadcast dims are contiguous from the left
      // Non-contiguous views always need stride-based indexing.
      bool needsStrideBroadcast = inputIsNonContiguous;
      bool seenNonBroadcast = false;
      for (int d = 0; d < outRank && !needsStrideBroadcast; d++) {
        bool isBroadcast = (paddedInputShape[d] == 1 && refOutputShape[d] > 1);
        if (seenNonBroadcast && isBroadcast) {
          needsStrideBroadcast = true;  // Inner-dimension broadcast
          break;
        }
        if (!isBroadcast) seenNonBroadcast = true;
      }

      if (needsStrideBroadcast) {
        // Stride-based indexing: decompose flat output offset into N-dim coords,
        // then recompute input offset using actual input strides.
        // For non-contiguous views, actual strides reflect the view layout.
        // For broadcasts, broadcast dims have stride 0.

        // Compute output strides (row-major — output is always C-contiguous)
        std::vector<LongType> outStrides(outRank);
        outStrides[outRank - 1] = 1;
        for (int d = outRank - 2; d >= 0; d--) {
          outStrides[d] = outStrides[d + 1] * refOutputShape[d + 1];
        }

        // Compute input strides: use ACTUAL strides from the NDArray when available,
        // falling back to C-contiguous computation when no stride info exists.
        // Map output dims to input dims: srcDim = d + (inRank - outRank).
        std::vector<LongType> inStrides(outRank);
        bool hasActualStrides = (!arg.strides.empty() && static_cast<int>(arg.strides.size()) == inRank);
        for (int d = 0; d < outRank; d++) {
          int srcDim = d + (inRank - outRank);  // same mapping as paddedInputShape
          if (paddedInputShape[d] <= 1) {
            inStrides[d] = 0;  // broadcast dim (or rank-pad dim)
          } else if (hasActualStrides && srcDim >= 0 && srcDim < inRank) {
            inStrides[d] = arg.strides[srcDim];
          } else {
            // Fallback: C-contiguous stride = product of dims below
            LongType stride = 1;
            for (int dd = d + 1; dd < outRank; dd++) {
              stride *= paddedInputShape[dd];
            }
            inStrides[d] = stride;
          }
        }

        // Generate MLIR: decompose flat offset into per-dimension indices,
        // then recompute flat input index using input strides.
        // inputOffset = 0
        // for each dim d:
        //   dimIdx = (offsets / outStride[d]) % outDim[d]
        //   if inStride[d] > 0: inputOffset += dimIdx * inStride[d]
        mlir::Value inputOffset = splatConstantI32(builder, loc, i32TensorType, 0);
        for (int d = 0; d < outRank; d++) {
          if (inStrides[d] == 0) continue;  // broadcast dim, skip

          // dimIdx = (offsets / outStride[d]) % outDim[d]
          mlir::Value dimIdx = offsets;
          if (outStrides[d] > 1) {
            auto strideConst = splatConstantI32(builder, loc, i32TensorType,
                                                 static_cast<int>(outStrides[d]));
            dimIdx = builder.create<mlir::arith::DivUIOp>(loc, dimIdx, strideConst);
          }
          if (refOutputShape[d] > 1) {  // mod by dim size (skip if dim=1)
            auto dimConst = splatConstantI32(builder, loc, i32TensorType,
                                              static_cast<int>(refOutputShape[d]));
            dimIdx = builder.create<mlir::arith::RemUIOp>(loc, dimIdx, dimConst);
          }
          // inputOffset += dimIdx * inStride[d]
          if (inStrides[d] == 1) {
            inputOffset = builder.create<mlir::arith::AddIOp>(loc, inputOffset, dimIdx);
          } else {
            auto inStrideConst = splatConstantI32(builder, loc, i32TensorType,
                                                   static_cast<int>(inStrides[d]));
            auto contrib = builder.create<mlir::arith::MulIOp>(loc, dimIdx, inStrideConst);
            inputOffset = builder.create<mlir::arith::AddIOp>(loc, inputOffset, contrib);
          }
        }
        loadOffsets = inputOffset;
      } else {
        // Simple broadcast: outermost dims only. Modular indexing is correct.
        auto inputSizeConst = builder.create<mlir::arith::ConstantIntOp>(
            loc, static_cast<int>(inputElements), 32);
        auto splatInputSize = builder.create<mlir::triton::SplatOp>(
            loc, i32TensorType, inputSizeConst);
        loadOffsets = builder.create<mlir::arith::RemUIOp>(loc, offsets, splatInputSize);
      }
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

  // ─── Precision-matching truncation ───────────────────────────────────────────
  // Native execution writes each op's output to memory in its native dtype
  // (typically FP16 for half-precision models).  When Triton fuses multiple ops
  // into a single kernel, intermediates stay in FP32 registers, avoiding the
  // FP16 truncation that happens between native ops.  This makes the fused
  // result MORE precise but DIFFERENT from native — accumulated differences
  // cause divergent token selection in autoregressive decoding.
  //
  // Fix: after each op, if the native output would be FP16/BF16, insert
  // FP32→FP16→FP32 round-trip to emulate native per-op memory truncation.
  // The final store to memory performs the real FP16 truncation.
  auto emulateNativePrecision = [&](mlir::Value val, int slotIdx) -> mlir::Value {
    if (!val) return val;
    auto elemType = getElementType(val);
    // Only truncate if the current SSA type is wider than the native output type
    if (!mlir::isa<mlir::FloatType>(elemType) || getFloatBitWidth(elemType) <= 16)
      return val;  // Already FP16 or narrower — no truncation needed
    // Determine native output dtype from the output NDArray
    for (int o = 0; o < slots[slotIdx].wiring.numOutputs; o++) {
      int outIdx = slots[slotIdx].wiring.outputSlotIndices[o];
      if (outIdx >= 0 && outIdx < totalOutputSlots && outputSlots[outIdx]) {
        auto nativeDtype = outputSlots[outIdx]->dataType();
        if (nativeDtype == DataType::HALF || nativeDtype == DataType::BFLOAT16) {
          // Round-trip: FP32 → FP16/BF16 (truncation) → FP32 (re-promote)
          auto narrowType = getMLIRType(builder, nativeDtype);
          val = castTo(builder, loc, val, narrowType);   // truncate
          val = castTo(builder, loc, val, builder.getF32Type());  // re-promote
        }
        break;  // Use first output's type (all outputs typically have the same dtype)
      }
    }
    return val;
  };

  for (int si = startSlot; si <= endSlot; si++, catIdx++) {
    auto& slot = slots[si];
    auto cat = categories[catIdx];

    // ── Comprehensive per-slot compile-time diagnostics (JIT path) ──
    if (DSP_DIAG_ENABLED(COMPILE)) {
      auto fmtShpJit = [](const std::vector<LongType>& s) -> std::string {
        std::string r;
        for (size_t i = 0; i < s.size(); i++) { if (i) r += ","; r += std::to_string(s[i]); }
        return r.empty() ? "empty" : r;
      };
      DSP_DIAG(COMPILE, "JIT SLOT[%d] op='%s' cat=%d inputs=%d outputs=%d iArgs=%d tArgs=%d bArgs=%d "
                "identity=%d view=%d fused=%d zeroOut=%d frozen=%d",
                si, slot.ident.opName.c_str(), static_cast<int>(cat),
                slot.wiring.numInputs, slot.wiring.numOutputs,
                slot.args.numIArgs, slot.args.numTArgs, slot.args.numBArgs,
                slot.isIdentityOp() ? 1 : 0, slot.isViewCapableOp() ? 1 : 0,
                slot.flags.inPlaceFused ? 1 : 0, slot.needsZeroedOutput() ? 1 : 0,
                slot.frozenConstantSlot() ? 1 : 0);
      for (int inp = 0; inp < slot.wiring.numInputs; inp++) {
        int srcIdx = slot.wiring.inputSourceIndices[inp];
        auto srcShape = resolveShapeLocal(srcIdx);
        const char* srcOp = "EXT";
        if (srcIdx >= 0 && srcIdx < totalSlots) srcOp = slots[srcIdx].ident.opName.c_str();
        bool exists = false;
        LongType len = -1;
        if (srcIdx >= 0 && srcIdx < totalOutputSlots && outputSlots && outputSlots[srcIdx]) {
          exists = true; len = outputSlots[srcIdx]->lengthOf();
        } else if (srcIdx < 0) {
          int ei = -(srcIdx + 1);
          if (ei < numExternalInputs && externalInputs && externalInputs[ei]) {
            exists = true; len = externalInputs[ei]->lengthOf();
          }
        }
        DSP_DIAG(COMPILE, "  input[%d] src=%d op='%s' shape=[%s] exists=%d len=%lld",
                  inp, srcIdx, srcOp, fmtShpJit(srcShape).c_str(),
                  exists, (long long)len);
      }
      for (int outp = 0; outp < slot.wiring.numOutputs; outp++) {
        int outIdx = slot.wiring.outputSlotIndices[outp];
        auto outShape = resolveShapeLocal(outIdx);
        DSP_DIAG(COMPILE, "  output[%d] slot=%d shape=[%s]",
                  outp, outIdx, fmtShpJit(outShape).c_str());
      }
      if (slot.args.numIArgs > 0 && slot.args.iArgs) {
        std::string iStr;
        for (int a = 0; a < slot.args.numIArgs && a < 20; a++) {
          if (a) iStr += ","; iStr += std::to_string(slot.args.iArgs[a]);
        }
        if (slot.args.numIArgs > 20) iStr += "...";
        DSP_DIAG(COMPILE, "  iArgs=[%s] (%d)", iStr.c_str(), slot.args.numIArgs);
      }
      if (slot.args.numTArgs > 0 && slot.args.tArgs) {
        char tBuf[256] = {0};
        int toff = 0;
        for (int a = 0; a < slot.args.numTArgs && a < 10 && toff < 240; a++) {
          toff += snprintf(tBuf + toff, sizeof(tBuf) - toff, "%s%.6g", a > 0 ? "," : "", slot.args.tArgs[a]);
        }
        DSP_DIAG(COMPILE, "  tArgs=[%s] (%d)", tBuf, slot.args.numTArgs);
      }
    }

    auto it = opTable.find(slot.ident.opName);
    if (it == opTable.end()) continue;
    const auto& mapping = it->second;
    opsEmitted++;

    // Frozen constant slots: their output SSA values were already populated
    // by tt.load from the input args (the GPU buffer holds the correct warmup
    // value). Skip recomputation entirely — downstream ops will use the loaded
    // values via ssaValues[].
    if (slot.frozenConstantSlot()) {
      // Verify SSA values are available for all outputs (set by tt.load loop above)
      bool allSet = true;
      for (int o = 0; o < slot.wiring.numOutputs; o++) {
        if (ssaValues.find(slot.wiring.outputSlotIndices[o]) == ssaValues.end()) {
          allSet = false;
          break;
        }
      }
      if (allSet) continue;
      // If SSA values are missing (frozen output not in inputArgs), this is a
      // compilation bug — the frozen constant should have been included in the
      // kernel's input argument list. Fail the entire module build so the segment
      // compilation path detects the failure and prevents execution with a broken
      // kernel (which would cause KERNEL_FAILURE at runtime).
      DSP_DIAG_SLOT(FALLBACK, si, "TritonIRBuilder: frozen slot %d (%s) has missing SSA values — "
                    "compilation BUG (frozen constant not in inputArgs). Failing module build.",
                    si, slot.ident.opName.c_str());
      return TritonIRModule();  // valid = false
    }

    if (cat == TritonOpCategory::BINARY_ELEMENTWISE) {
      // Binary: needs two inputs (except pow which can be unary with scalar exponent in tArgs)
      if (slot.wiring.numInputs < 2) {
        // Special case: pow with scalar exponent in tArgs (unary mode)
        std::string opLower2 = slot.ident.opName;
        std::transform(opLower2.begin(), opLower2.end(), opLower2.begin(), ::tolower);
        if (opLower2 == "pow" && slot.wiring.numInputs >= 1) {
          int inputSrc = slot.wiring.inputSourceIndices[0];
          auto inputIt = ssaValues.find(inputSrc);
          if (inputIt != ssaValues.end()) {
            auto opResult = emitUnaryElementwise(builder, loc, mapping, slot, inputIt->second, blockSize);
            opResult = emulateNativePrecision(opResult, si);
            for (int o = 0; o < slot.wiring.numOutputs; o++) {
              ssaValues[slot.wiring.outputSlotIndices[o]] = opResult;
            }
          } else {
            DSP_DIAG_SLOT(FALLBACK, si, "TritonIRBuilder: missing SSA value for unary pow at slot %d (src=%d)",
                      si, inputSrc);
          }
          continue;
        }
        DSP_DIAG_SLOT(FALLBACK, si, "TritonIRBuilder: binary op '%s' at slot %d has < 2 inputs",
                  slot.ident.opName.c_str(), si);
        continue;
      }

      int lhsSrc = slot.wiring.inputSourceIndices[0];
      int rhsSrc = slot.wiring.inputSourceIndices[1];

      auto lhsIt = ssaValues.find(lhsSrc);
      auto rhsIt = ssaValues.find(rhsSrc);

      if (lhsIt == ssaValues.end() || rhsIt == ssaValues.end()) {
        DSP_DIAG_SLOT(FALLBACK, si, "TritonIRBuilder: missing SSA value for binary op '%s' at slot %d "
                  "(lhs=%d:%s, rhs=%d:%s)",
                  slot.ident.opName.c_str(), si,
                  lhsSrc, lhsIt != ssaValues.end() ? "found" : "MISSING",
                  rhsSrc, rhsIt != ssaValues.end() ? "found" : "MISSING");
        continue;
      }

      auto opResult = emitBinaryElementwise(builder, loc, mapping, slot, lhsIt->second, rhsIt->second);
      opResult = emulateNativePrecision(opResult, si);

      // Store result SSA value for each output slot
      for (int o = 0; o < slot.wiring.numOutputs; o++) {
        ssaValues[slot.wiring.outputSlotIndices[o]] = opResult;
      }

    } else if (cat == TritonOpCategory::UNARY_ELEMENTWISE) {
      // Unary: needs one input
      if (slot.wiring.numInputs < 1) {
        DSP_DIAG_SLOT(FALLBACK, si, "TritonIRBuilder: unary op '%s' at slot %d has no inputs",
                  slot.ident.opName.c_str(), si);
        continue;
      }

      int inputSrc = slot.wiring.inputSourceIndices[0];
      auto inputIt = ssaValues.find(inputSrc);
      if (inputIt == ssaValues.end()) {
        DSP_DIAG_SLOT(FALLBACK, si, "TritonIRBuilder: missing SSA value for unary op '%s' at slot %d (src=%d)",
                  slot.ident.opName.c_str(), si, inputSrc);
        continue;
      }

      mlir::Value opResult;
      // ClipByValue with tensor min/max: use loaded SSA inputs, not tArgs
      {
        std::string ol = slot.ident.opName;
        std::transform(ol.begin(), ol.end(), ol.begin(), ::tolower);
        if ((ol == "clipbyvalue" || ol == "clip_by_value" || ol == "clamp")
            && slot.wiring.numInputs >= 3) {
          auto minIt = ssaValues.find(slot.wiring.inputSourceIndices[1]);
          auto maxIt = ssaValues.find(slot.wiring.inputSourceIndices[2]);
          if (minIt != ssaValues.end() && maxIt != ssaValues.end()) {
            auto elTy = mlir::cast<mlir::RankedTensorType>(
                inputIt->second.getType()).getElementType();
            if (mlir::isa<mlir::FloatType>(elTy)) {
              auto cl = builder.create<mlir::arith::MaximumFOp>(loc, inputIt->second, minIt->second);
              opResult = builder.create<mlir::arith::MinimumFOp>(loc, cl, maxIt->second);
            } else {
              auto cl = builder.create<mlir::arith::MaxSIOp>(loc, inputIt->second, minIt->second);
              opResult = builder.create<mlir::arith::MinSIOp>(loc, cl, maxIt->second);
            }
          }
        }
      }
      if (!opResult)
        opResult = emitUnaryElementwise(builder, loc, mapping, slot, inputIt->second, blockSize);
      opResult = emulateNativePrecision(opResult, si);

      for (int o = 0; o < slot.wiring.numOutputs; o++) {
        ssaValues[slot.wiring.outputSlotIndices[o]] = opResult;
      }

    } else if (cat == TritonOpCategory::COMPARISON) {
      // Comparison: produces bool tensor from 2 inputs or 1 input + scalar tArg
      if (slot.wiring.numInputs < 1) {
        DSP_DIAG_SLOT(FALLBACK, si, "TritonIRBuilder: comparison op '%s' at slot %d has no inputs",
                  slot.ident.opName.c_str(), si);
        continue;
      }
      int lhsSrc = slot.wiring.inputSourceIndices[0];
      auto lhsIt = ssaValues.find(lhsSrc);
      if (lhsIt == ssaValues.end()) {
        DSP_DIAG_SLOT(FALLBACK, si, "TritonIRBuilder: missing SSA value for comparison op '%s' at slot %d",
                  slot.ident.opName.c_str(), si);
        continue;
      }

      mlir::Value opResult;
      if (slot.wiring.numInputs >= 2) {
        // Standard 2-input comparison
        int rhsSrc = slot.wiring.inputSourceIndices[1];
        auto rhsIt = ssaValues.find(rhsSrc);
        if (rhsIt == ssaValues.end()) {
          DSP_DIAG_SLOT(FALLBACK, si, "TritonIRBuilder: missing SSA value for comparison op '%s' at slot %d",
                    slot.ident.opName.c_str(), si);
          continue;
        }
        opResult = emitComparisonOp(builder, loc, slot.ident.opName, lhsIt->second, rhsIt->second, blockSize);
      } else if (slot.args.numTArgs > 0 && slot.args.tArgs) {
        // Scalar comparison: second operand comes from tArgs[0]
        auto lhsTensorType = mlir::cast<mlir::RankedTensorType>(lhsIt->second.getType());
        auto lhsElemType = lhsTensorType.getElementType();
        double scalarVal = slot.args.tArgs[0];
        mlir::Value rhsSplat;
        if (mlir::isa<mlir::FloatType>(lhsElemType)) {
          rhsSplat = splatConstantF32(builder, loc, lhsTensorType, static_cast<float>(scalarVal));
        } else {
          rhsSplat = splatConstantI32(builder, loc, lhsTensorType, static_cast<int>(scalarVal));
        }
        opResult = emitComparisonOp(builder, loc, slot.ident.opName, lhsIt->second, rhsSplat, blockSize);
      } else {
        DSP_DIAG_SLOT(FALLBACK, si, "TritonIRBuilder: scalar comparison op '%s' at slot %d has no tArgs",
                  slot.ident.opName.c_str(), si);
        continue;
      }
      for (int o = 0; o < slot.wiring.numOutputs; o++) {
        ssaValues[slot.wiring.outputSlotIndices[o]] = opResult;
      }

    } else if (cat == TritonOpCategory::LOGICAL) {
      // Logical: 1 or 2 inputs depending on op
      if (slot.wiring.numInputs < 1) {
        DSP_DIAG_SLOT(FALLBACK, si, "TritonIRBuilder: logical op '%s' at slot %d has no inputs",
                  slot.ident.opName.c_str(), si);
        continue;
      }
      int lhsSrc = slot.wiring.inputSourceIndices[0];
      auto lhsIt = ssaValues.find(lhsSrc);
      if (lhsIt == ssaValues.end()) {
        DSP_DIAG_SLOT(FALLBACK, si, "TritonIRBuilder: missing SSA value for logical op '%s' at slot %d",
                  slot.ident.opName.c_str(), si);
        continue;
      }
      // For NOT ops, rhs is unused (emitLogicalOp handles it internally)
      mlir::Value rhsVal = lhsIt->second;  // dummy for unary
      if (slot.wiring.numInputs >= 2) {
        int rhsSrc = slot.wiring.inputSourceIndices[1];
        auto rhsIt = ssaValues.find(rhsSrc);
        if (rhsIt != ssaValues.end()) rhsVal = rhsIt->second;
      }
      auto opResult = emitLogicalOp(builder, loc, slot.ident.opName, lhsIt->second, rhsVal, blockSize);
      for (int o = 0; o < slot.wiring.numOutputs; o++) {
        ssaValues[slot.wiring.outputSlotIndices[o]] = opResult;
      }

    } else if (cat == TritonOpCategory::TERNARY) {
      // Ternary: where/select needs 3 inputs (condition, true_val, false_val)
      if (slot.wiring.numInputs < 3) {
        DSP_DIAG_SLOT(FALLBACK, si, "TritonIRBuilder: ternary op '%s' at slot %d has < 3 inputs",
                  slot.ident.opName.c_str(), si);
        continue;
      }
      int condSrc = slot.wiring.inputSourceIndices[0];
      int trueSrc = slot.wiring.inputSourceIndices[1];
      int falseSrc = slot.wiring.inputSourceIndices[2];
      auto condIt = ssaValues.find(condSrc);
      auto trueIt = ssaValues.find(trueSrc);
      auto falseIt = ssaValues.find(falseSrc);
      if (condIt == ssaValues.end() || trueIt == ssaValues.end() || falseIt == ssaValues.end()) {
        DSP_DIAG_SLOT(FALLBACK, si, "TritonIRBuilder: missing SSA value for ternary op '%s' at slot %d",
                  slot.ident.opName.c_str(), si);
        continue;
      }
      auto opResult = emitTernaryOp(builder, loc, condIt->second, trueIt->second, falseIt->second, blockSize);
      opResult = emulateNativePrecision(opResult, si);
      for (int o = 0; o < slot.wiring.numOutputs; o++) {
        ssaValues[slot.wiring.outputSlotIndices[o]] = opResult;
      }

    } else if (cat == TritonOpCategory::IDENTITY) {
      // Identity/assign: SSA value forwarding, no IR op needed
      if (slot.wiring.numInputs < 1) {
        DSP_DIAG_SLOT(FALLBACK, si, "TritonIRBuilder: identity op '%s' at slot %d has no inputs",
                  slot.ident.opName.c_str(), si);
        continue;
      }
      // For assign(target, source): output = source = input[1]
      // For identity(x): output = x = input[0]
      int inputIdx = (slot.wiring.numInputs >= 2) ? 1 : 0;
      int inputSrc = slot.wiring.inputSourceIndices[inputIdx];
      auto inputIt = ssaValues.find(inputSrc);
      if (inputIt == ssaValues.end()) {
        DSP_DIAG_SLOT(FALLBACK, si, "TritonIRBuilder: missing SSA value for identity op '%s' at slot %d",
                  slot.ident.opName.c_str(), si);
        continue;
      }
      // Forward the SSA value directly — no computation needed
      DSP_DIAG_SLOT(COMPILE, si, "TritonIRBuilder: IDENTITY op '%s' at slot %d: numInputs=%d inputIdx=%d inputSrc=%d → forwarded to %d output(s)",
                slot.ident.opName.c_str(), si, slot.wiring.numInputs, inputIdx, inputSrc, slot.wiring.numOutputs);
      for (int o = 0; o < slot.wiring.numOutputs; o++) {
        ssaValues[slot.wiring.outputSlotIndices[o]] = inputIt->second;
      }

    } else if (cat == TritonOpCategory::CAST) {
      // Cast: type conversion using the castTo() helper
      if (slot.wiring.numInputs < 1) {
        DSP_DIAG_SLOT(FALLBACK, si, "TritonIRBuilder: cast op '%s' at slot %d has no inputs",
                  slot.ident.opName.c_str(), si);
        continue;
      }
      int inputSrc = slot.wiring.inputSourceIndices[0];
      auto inputIt = ssaValues.find(inputSrc);
      if (inputIt == ssaValues.end()) {
        DSP_DIAG_SLOT(FALLBACK, si, "TritonIRBuilder: missing SSA value for cast op '%s' at slot %d",
                  slot.ident.opName.c_str(), si);
        continue;
      }
      // If cast elimination marked this as identity, forward the SSA value
      // without computing the cast (consistent with slot-by-slot identity path).
      if (slot.isIdentityOp()) {
        for (int o = 0; o < slot.wiring.numOutputs; o++) {
          ssaValues[slot.wiring.outputSlotIndices[o]] = inputIt->second;
        }
      } else {
        // Determine target type from the output slot's dtype.
        // Priority: dArgs[0] > iArgs[0] > output slot dtype.
        // Cast ops in libnd4j store the target dtype in iArgs[0] as an integer.
        // dArgs (extraTypes) may or may not be populated depending on the model format.
        // resolveDtypeLocal is the last resort but can return FLOAT32 (wrong) when
        // the output slot belongs to a view-capable op with no pre-allocated array.
        DataType targetDtype = FLOAT32;  // default
        if (slot.args.numDArgs > 0 && slot.args.dArgs) {
          targetDtype = slot.args.dArgs[0];
        } else if (slot.args.numIArgs > 0 && slot.args.iArgs) {
          // Cast ops store the target dtype in iArgs[0]
          targetDtype = static_cast<DataType>(slot.args.iArgs[0]);
        } else if (slot.wiring.numOutputs > 0) {
          int outIdx = slot.wiring.outputSlotIndices[0];
          targetDtype = resolveDtypeLocal(outIdx);
        }
        DSP_DIAG_SLOT(COMPILE, si, "TritonIRBuilder: cast at slot %d: numDArgs=%d numIArgs=%d targetDtype=%d (%s)",
                  si, slot.args.numDArgs, slot.args.numIArgs, (int)targetDtype,
                  DataTypeUtils::asString(targetDtype).c_str());
        auto targetElemType = getMLIRType(builder, targetDtype);
        auto opResult = castTo(builder, loc, inputIt->second, targetElemType);
        for (int o = 0; o < slot.wiring.numOutputs; o++) {
          ssaValues[slot.wiring.outputSlotIndices[o]] = opResult;
        }
      }

    } else if (cat == TritonOpCategory::REDUCTION) {
      // Segmented reduction: for each output element, accumulate over the reduction axis.
      // Unlike elementwise ops, reduction changes tensor size, so we can't use the SSA value
      // (which was loaded using output offsets). Instead, directly load from input buffer.
      if (slot.wiring.numInputs < 1) {
        DSP_DIAG_SLOT(FALLBACK, si, "TritonIRBuilder: reduction op '%s' at slot %d has no inputs",
                  slot.ident.opName.c_str(), si);
        continue;
      }
      int inputSrc = slot.wiring.inputSourceIndices[0];
      // Get reduction axis from iArgs
      int reductionAxis = (slot.args.numIArgs > 0 && slot.args.iArgs) ? static_cast<int>(slot.args.iArgs[0]) : -1;

      // Resolve input shape
      auto inputShape = resolveShapeLocal(inputSrc);
      int inputRank = static_cast<int>(inputShape.size());
      if (inputRank == 0) {
        DSP_DIAG_SLOT(SHAPE, si, "TritonIRBuilder: reduction op '%s' has no input shape info", slot.ident.opName.c_str());
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
        DSP_DIAG_SLOT(FALLBACK, si, "TritonIRBuilder: reduction input slot %d not found in kernel args — cannot compile segmented reduction", inputSrc);
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

          // Per-buffer mask: the reduction input buffer may be SMALLER than the global
          // n_elements (e.g., cast output [2] in a kernel with n_elements=64). Using the
          // global mask would write past the buffer boundary, corrupting adjacent memory.
          mlir::Value midStoreMask = mask;  // default: global mask
          LongType midBufElements = 1;
          auto& midArg = result.args[midArgIdx];
          for (auto d : midArg.shape) midBufElements *= d;
          if (!midArg.shape.empty() && midBufElements > 0 &&
              midBufElements < static_cast<LongType>(maxOutputElements)) {
            auto midN = builder.create<mlir::arith::ConstantIntOp>(
                loc, static_cast<int>(std::min(midBufElements, static_cast<LongType>(2147483647))), 32);
            auto splatMidN = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, midN);
            midStoreMask = builder.create<mlir::arith::CmpIOp>(
                loc, mlir::arith::CmpIPredicate::slt, offsets, splatMidN);
            DSP_DIAG_SLOT(COMPILE, si, "TritonIRBuilder: reduction input slot %d: per-buffer mask (%lld elems vs %lld global)",
                      inputSrc, (long long)midBufElements, (long long)maxOutputElements);
          }
          builder.create<mlir::triton::StoreOp>(loc, midPtrs, midStoreVal, midStoreMask,
              mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL);
          DSP_DIAG_SLOT(COMPILE, si, "TritonIRBuilder: stored reduction input slot %d to buffer for segmented reduction", inputSrc);
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
      std::string opLower = slot.ident.opName;
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

      // Kahan compensation for sum/mean — reduces accumulation error from O(n*eps) to O(eps),
      // making result independent of accumulation order (matches native tree reduction).
      bool useKahan = !isMax && !isMin && !isProd;
      mlir::Value kahanComp;
      if (useKahan) {
        kahanComp = splatConstantF32(builder, loc, f32TensorType, 0.0f);
      }

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
        else if (useKahan) {
          // Kahan compensated summation: y = val - comp; t = acc + y; comp = (t - acc) - y; acc = t
          auto y = builder.create<mlir::arith::SubFOp>(loc, val, kahanComp);
          auto t = builder.create<mlir::arith::AddFOp>(loc, acc, y);
          auto tMinusAcc = builder.create<mlir::arith::SubFOp>(loc, t, acc);
          kahanComp = builder.create<mlir::arith::SubFOp>(loc, tMinusAcc, y);
          acc = t;
        }
      }

      // For mean: divide by reduction size
      if (isMean && reductionSize > 0) {
        auto countSplat = splatConstantF32(builder, loc, f32TensorType,
            static_cast<float>(reductionSize));
        acc = builder.create<mlir::arith::DivFOp>(loc, acc, countSplat);
      }

      // Cast back to output element type
      auto outSlotIdx = slot.wiring.outputSlotIndices[0];
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
        for (int inp2 = 0; inp2 < slots[si2].wiring.numInputs; inp2++) {
          for (int o = 0; o < slot.wiring.numOutputs; o++) {
            if (slots[si2].wiring.inputSourceIndices[inp2] == slot.wiring.outputSlotIndices[o])
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
        mlir::Value broadcastKahanComp;
        if (useKahan) {
          broadcastKahanComp = splatConstantF32(builder, loc, f32TensorType, 0.0f);
        }
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
          else if (useKahan) {
            auto y2 = builder.create<mlir::arith::SubFOp>(loc, val2, broadcastKahanComp);
            auto t2 = builder.create<mlir::arith::AddFOp>(loc, broadcastAcc, y2);
            auto tMinusAcc2 = builder.create<mlir::arith::SubFOp>(loc, t2, broadcastAcc);
            broadcastKahanComp = builder.create<mlir::arith::SubFOp>(loc, tMinusAcc2, y2);
            broadcastAcc = t2;
          }
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

      opResult = emulateNativePrecision(opResult, si);
      for (int o = 0; o < slot.wiring.numOutputs; o++) {
        ssaValues[slot.wiring.outputSlotIndices[o]] = opResult;
      }

    } else if (cat == TritonOpCategory::NORMALIZATION) {
      // Normalization: load input with actual shape (not flattened), call emitNormalizationOp
      if (slot.wiring.numInputs < 1) {
        DSP_DIAG_SLOT(FALLBACK, si, "TritonIRBuilder: normalization op '%s' at slot %d has no inputs",
                  slot.ident.opName.c_str(), si);
        continue;
      }
      int inputSrc = slot.wiring.inputSourceIndices[0];
      mlir::Value inputValue;
      mlir::RankedTensorType inputTensorType;
      
      // Get actual input shape for proper normalization semantics
      auto inputShape = resolveShapeLocal(inputSrc);
      if (inputShape.empty()) {
        DSP_DIAG_SLOT(FALLBACK, si, "TritonIRBuilder: cannot resolve shape for normalization input at slot %d", si);
        continue;
      }
      
      // Compute total elements in the actual tensor
      int64_t actualNumElements = 1;
      std::vector<int64_t> actualShape64;
      for (auto d : inputShape) {
        actualShape64.push_back(static_cast<int64_t>(d));
        actualNumElements *= static_cast<int64_t>(d);
      }
      
      // For rms_norm, compute logical row length (last dimension) for correct reduction
      int64_t logicalRowLen = actualShape64.back();
      int64_t numRows = actualNumElements / logicalRowLen;
      // Pad row length to next power of 2 for Triton make_range requirement
      int64_t paddedRowLen = 1;
      while (paddedRowLen < logicalRowLen) paddedRowLen *= 2;

      // For multi-row normalization: compute row offset = pid * logicalRowLen
      // Each block processes one row; pid selects which row.
      mlir::Value normRowOffset;
      if (normMultiRow) {
        auto rowLenVal = builder.create<mlir::arith::ConstantIntOp>(loc, normLogicalRowLen, 32);
        normRowOffset = builder.create<mlir::arith::MulIOp>(loc, pid, rowLenVal);
      }

      // Helper: add row offset to padded range for multi-row addressing
      // Input data is contiguous: row i starts at ptr + i * logicalRowLen
      auto makeRowOffsetRange = [&](mlir::Value paddedRange) -> mlir::Value {
        if (!normMultiRow) return paddedRange;
        auto splatOffset = builder.create<mlir::triton::SplatOp>(loc,
            paddedRange.getType(), normRowOffset);
        return builder.create<mlir::arith::AddIOp>(loc, paddedRange, splatOffset);
      };

      // For normalization, ALWAYS load external inputs directly from buffer with actual shape
      // (SSA values from preloading are flattened to blockSize, which breaks reduction semantics)
      
      if (inputSrc < 0) {
        // External input - load with ACTUAL shape, not flattened (bypass SSA cache)
        int extIdx = -(inputSrc + 1);
        if (extIdx >= numExternalInputs || !externalInputs[extIdx]) {
          DSP_DIAG_SLOT(FALLBACK, si, "TritonIRBuilder: missing external input %d for normalization op '%s' at slot %d",
                    extIdx, slot.ident.opName.c_str(), si);
          continue;
        }
        // Find the arg for this external input
        auto extArgIt = std::find_if(result.args.begin(), result.args.end(),
            [inputSrc](const TritonKernelArg& arg) {
                return arg.slotIndex == inputSrc && !arg.isOutput;
            });
        if (extArgIt == result.args.end()) {
          DSP_DIAG_SLOT(FALLBACK, si, "TritonIRBuilder: external input %d not in result.args for normalization op '%s' at slot %d",
                    inputSrc, slot.ident.opName.c_str(), si);
          continue;
        }
        int argIdx = extArgIt - result.args.begin();
        auto bufferArg = getBufferArg(argIdx);
        if (!bufferArg) {
          DSP_DIAG_SLOT(FALLBACK, si, "TritonIRBuilder: null bufferArg for external input %d in normalization op '%s' at slot %d",
                    inputSrc, slot.ident.opName.c_str(), si);
          continue;
        }
        // Load with padded tensor shape (power of 2 for Triton make_range)
        auto elemType = getMLIRType(builder, externalInputs[extIdx]->dataType());
        auto ptrType = mlir::cast<mlir::triton::PointerType>(bufferArg.getType());

        // Create offsets for padded row size (power of 2)
        auto paddedRange = builder.create<mlir::triton::MakeRangeOp>(loc,
            mlir::RankedTensorType::get({paddedRowLen}, builder.getI32Type()), 0, paddedRowLen);
        // For multi-row: add row offset so each block reads the correct row
        auto rowOffsetRange = makeRowOffsetRange(paddedRange);
        auto paddedPtrTensorType = mlir::RankedTensorType::get({paddedRowLen}, ptrType);
        auto splatPtr = builder.create<mlir::triton::SplatOp>(loc, paddedPtrTensorType, bufferArg);
        auto ptrs = builder.create<mlir::triton::AddPtrOp>(loc, paddedPtrTensorType, splatPtr, rowOffsetRange);

        // Create mask for actual row length (not padded)
        auto rowLenConst = builder.create<mlir::arith::ConstantIntOp>(loc, logicalRowLen, 32);
        auto splatRowLen = builder.create<mlir::triton::SplatOp>(loc,
            mlir::RankedTensorType::get({paddedRowLen}, builder.getI32Type()), rowLenConst);
        auto fullMask = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt, paddedRange, splatRowLen);

        inputValue = builder.create<mlir::triton::LoadOp>(loc, ptrs.getResult(), fullMask,
            builder.create<mlir::triton::SplatOp>(loc,
                mlir::RankedTensorType::get({paddedRowLen}, elemType),
                builder.create<mlir::arith::ConstantOp>(loc, builder.getFloatAttr(elemType, 0.0))).getResult(),
            mlir::triton::CacheModifier::NONE,
            mlir::triton::EvictionPolicy::NORMAL, false);
        inputTensorType = mlir::RankedTensorType::get({paddedRowLen}, elemType);

        // Update SSA cache with correctly-shaped value for downstream consumers
        ssaValues[inputSrc] = inputValue;
      } else {
        // Internal cross-section input - load directly from its kernel arg buffer.
        // Preloading is disabled for normalization leaves, so this value may not
        // exist in ssaValues even though it is present as an input arg.
        // Find the buffer arg for this internal input
        auto intArgIt = std::find_if(result.args.begin(), result.args.end(),
            [inputSrc](const TritonKernelArg& arg) {
                return arg.slotIndex == inputSrc && !arg.isOutput;
            });
        if (intArgIt == result.args.end()) {
          DSP_DIAG_SLOT(FALLBACK, si, "TritonIRBuilder: missing buffer arg for normalization internal input %d", inputSrc);
          continue;
        }
        int intArgIdx = intArgIt - result.args.begin();
        auto intBufferArg = getBufferArg(intArgIdx);
        if (!intBufferArg) {
          DSP_DIAG_SLOT(FALLBACK, si, "TritonIRBuilder: null bufferArg for normalization internal input %d", inputSrc);
          continue;
        }
        // Load with padded tensor shape (power of 2 for Triton make_range)
        auto ptrType = mlir::cast<mlir::triton::PointerType>(intBufferArg.getType());
        auto elemType = ptrType.getPointeeType();

        // Create offsets for padded row size (power of 2)
        auto paddedRange = builder.create<mlir::triton::MakeRangeOp>(loc,
            mlir::RankedTensorType::get({paddedRowLen}, builder.getI32Type()), 0, paddedRowLen);
        // For multi-row: add row offset so each block reads the correct row
        auto rowOffsetRange = makeRowOffsetRange(paddedRange);
        auto paddedPtrTensorType = mlir::RankedTensorType::get({paddedRowLen}, ptrType);
        auto splatPtr = builder.create<mlir::triton::SplatOp>(loc, paddedPtrTensorType, intBufferArg);
        auto ptrs = builder.create<mlir::triton::AddPtrOp>(loc, paddedPtrTensorType, splatPtr, rowOffsetRange);

        // Create mask for actual row length (not padded)
        auto rowLenConst = builder.create<mlir::arith::ConstantIntOp>(loc, logicalRowLen, 32);
        auto splatRowLen = builder.create<mlir::triton::SplatOp>(loc,
            mlir::RankedTensorType::get({paddedRowLen}, builder.getI32Type()), rowLenConst);
        auto fullMask = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt, paddedRange, splatRowLen);

        inputValue = builder.create<mlir::triton::LoadOp>(loc, ptrs.getResult(), fullMask,
            builder.create<mlir::triton::SplatOp>(loc, 
                mlir::RankedTensorType::get({paddedRowLen}, elemType),
                builder.create<mlir::arith::ConstantOp>(loc, builder.getFloatAttr(elemType, 0.0))).getResult(),
            mlir::triton::CacheModifier::NONE,
            mlir::triton::EvictionPolicy::NORMAL, false);
        inputTensorType = mlir::RankedTensorType::get({paddedRowLen}, elemType);
        
        // Update SSA cache with correctly-shaped value
        ssaValues[inputSrc] = inputValue;
      }

      // Compute normalization axis (last dimension)
      int axis = inputTensorType.getRank() - 1;
      if (axis < 0) axis = 0;

      auto outSlotIdx = slot.wiring.outputSlotIndices[0];
      mlir::RankedTensorType outputType;
      {
        auto outShape = resolveShapeLocal(outSlotIdx);
        if (!outShape.empty()) {
          auto elemType = getElementType(inputValue);
          std::vector<int64_t> outShape64;
          for (auto d : outShape) outShape64.push_back(static_cast<int64_t>(d));
          outputType = mlir::RankedTensorType::get(outShape64, elemType);
        }
      }
      std::string normKey = normalizeOpToken(slot.ident.opName);

      // ── Normalization backward ops (rms_norm_bp, layer_norm_bp, fused_layer_norm_bp) ──
      // These produce MULTIPLE outputs (dx + dgamma [+ dbeta]) and write them directly
      // via tt.store/tt.atomic_add.  They do NOT go through the forward emitNormalizationOp path.
      {
        // normalizeOpToken strips underscores and lowercases: rms_norm_bp -> rmsnormbp
        bool isNormBp = (normKey == "rmsnormbp"           ||
                         normKey == "layernormbp"          ||
                         normKey == "layernormalizationbp" ||
                         normKey == "fusedlayernormbp");
        if (isNormBp) {
          // Helper: load a row-tensor from a slot index.
          // rowWise=true  -> x / dy (offset by pid * logicalRowLen)
          // rowWise=false -> gamma / beta (column-indexed, no row offset)
          auto loadBpInput = [&](int inputPos, bool rowWise) -> mlir::Value {
            if (inputPos >= slot.wiring.numInputs) return mlir::Value();
            int src = slot.wiring.inputSourceIndices[inputPos];

            auto buildLoad = [&](mlir::Value bufArg, mlir::Type elemTy, int64_t logLen) -> mlir::Value {
              int64_t padLen = 1;
              while (padLen < logLen) padLen <<= 1;
              auto i32Ty    = builder.getI32Type();
              auto idxTy    = mlir::RankedTensorType::get({padLen}, i32Ty);
              auto ptrTy    = mlir::cast<mlir::triton::PointerType>(bufArg.getType());
              auto ptrTenTy = mlir::RankedTensorType::get({padLen}, ptrTy);
              auto dataTy   = mlir::RankedTensorType::get({padLen}, elemTy);

              auto range = builder.create<mlir::triton::MakeRangeOp>(loc, idxTy, 0, padLen);
              mlir::Value baseOff;
              if (rowWise && normMultiRow && normRowOffset) {
                auto splatOff = builder.create<mlir::triton::SplatOp>(loc, idxTy, normRowOffset);
                baseOff = builder.create<mlir::arith::AddIOp>(loc, splatOff, range);
              } else {
                baseOff = range;
              }

              auto lenConst = builder.create<mlir::arith::ConstantIntOp>(loc, logLen, 32);
              auto lenSplat = builder.create<mlir::triton::SplatOp>(loc, idxTy, lenConst);
              auto localMask = builder.create<mlir::arith::CmpIOp>(
                  loc, mlir::arith::CmpIPredicate::slt, range, lenSplat);

              auto splatPtr = builder.create<mlir::triton::SplatOp>(loc, ptrTenTy, bufArg);
              auto ptrs    = builder.create<mlir::triton::AddPtrOp>(loc, ptrTenTy, splatPtr, baseOff);
              // Build a zero constant compatible with the element type
              mlir::Value zero;
              if (mlir::isa<mlir::FloatType>(elemTy)) {
                zero = builder.create<mlir::arith::ConstantOp>(
                    loc, builder.getFloatAttr(elemTy, 0.0));
              } else {
                zero = builder.create<mlir::arith::ConstantOp>(
                    loc, builder.getIntegerAttr(elemTy, 0));
              }
              auto zeroTen = builder.create<mlir::triton::SplatOp>(loc, dataTy, zero);
              return builder.create<mlir::triton::LoadOp>(
                  loc, ptrs, localMask, zeroTen,
                  mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL, false);
            };

            // External input
            if (src < 0) {
              int extIdx2 = -(src + 1);
              if (extIdx2 >= numExternalInputs || !externalInputs[extIdx2]) return mlir::Value();
              auto argIt = std::find_if(result.args.begin(), result.args.end(),
                  [src](const TritonKernelArg& a){ return a.slotIndex == src; });
              if (argIt == result.args.end()) return mlir::Value();
              auto bufArg = getBufferArg(static_cast<int>(argIt - result.args.begin()));
              if (!bufArg) return mlir::Value();
              auto sideShape = resolveShapeLocal(src);
              if (sideShape.empty()) return mlir::Value();
              // rowWise (x/dy): use logicalRowLen.
              // non-rowWise (gamma/beta): use last dim (1-D column vector length = logicalRowLen).
              int64_t logLen = rowWise ? logicalRowLen : static_cast<int64_t>(sideShape.back());
              auto elemTy2 = getMLIRType(builder, externalInputs[extIdx2]->dataType());
              return buildLoad(bufArg, elemTy2, logLen);
            }
            // Internal SSA
            auto ssaIt = ssaValues.find(src);
            if (ssaIt != ssaValues.end()) return ssaIt->second;
            // Try from kernel args
            auto argIt = slotToArgIdx.find(src);
            if (argIt == slotToArgIdx.end()) return mlir::Value();
            auto bufArg = getBufferArg(argIt->second);
            if (!bufArg) return mlir::Value();
            auto ptrTy = mlir::cast<mlir::triton::PointerType>(bufArg.getType());
            return buildLoad(bufArg, ptrTy.getPointeeType(), logicalRowLen);
          };

          // Input positions:
          //   rms_norm_bp:         x=input[0], dy=input[1]
          //   layer_norm_bp:       x=input[0], gamma=input[1], dy=last
          //   fused_layer_norm_bp: x=input[0], gamma=input[1], dy=input[2]
          bool isRmsBp = (normKey == "rmsnormbp");
          mlir::Value bpX     = inputValue;                         // input[0] always = x
          mlir::Value bpDy    = loadBpInput(isRmsBp ? 1 : 2, /*rowWise=*/true);
          mlir::Value bpGamma = isRmsBp ? mlir::Value() : loadBpInput(1, /*rowWise=*/false);

          if (!bpX || !bpDy) {
            DSP_DIAG_SLOT(FALLBACK, si, "TritonIRBuilder: backward norm '%s' at slot %d: missing dy or x",
                          slot.ident.opName.c_str(), si);
            continue;
          }

          // Output pointers: dx=output[0], dgamma=output[1], dbeta=output[2]
          auto getOutPtr = [&](int outPos) -> mlir::Value {
            if (outPos >= slot.wiring.numOutputs) return mlir::Value();
            return getSlotArgPtr(slot.wiring.outputSlotIndices[outPos]);
          };
          mlir::Value bpDxPtr     = getOutPtr(0);
          mlir::Value bpDgammaPtr = getOutPtr(1);
          mlir::Value bpDbetaPtr  = getOutPtr(2);

          float bpEps = (slot.args.numTArgs > 0 && slot.args.tArgs)
                        ? static_cast<float>(slot.args.tArgs[0]) : 1e-5f;

          // rowBase: for multiRow = pid * logicalRowLen; else 0
          mlir::Value bpRowBase = (normMultiRow && normRowOffset)
              ? normRowOffset
              : builder.create<mlir::arith::ConstantIntOp>(loc, 0, 32).getResult();

          bool bpOk = emitNormalizationBackwardSection(
              builder, loc, slot.ident.opName,
              bpDy, bpX, bpGamma,
              /*mean=*/mlir::Value(), /*invStd=*/mlir::Value(),
              bpEps, logicalRowLen, bpRowBase,
              bpDxPtr, bpDgammaPtr, bpDbetaPtr);

          if (!bpOk) {
            DSP_DIAG_SLOT(FALLBACK, si, "TritonIRBuilder: emitNormalizationBackwardSection failed for '%s' at slot %d",
                          slot.ident.opName.c_str(), si);
            continue;
          }

          // Mark all outputs as inline-stored (skip generic store loop)
          for (int o = 0; o < slot.wiring.numOutputs; o++) {
            int outS = slot.wiring.outputSlotIndices[o];
            normInlineStoredOutputs.insert(outS);
            // Provide a placeholder SSA value for downstream consumers
            ssaValues[outS] = inputValue;  // same shape as x/dx
          }
          continue;  // Skip the forward normalization path below
        }
      }

      // For normalization side inputs (gamma, beta), ALWAYS load external inputs directly
      // (SSA values from preloading are flattened to blockSize, which breaks reduction semantics)
      auto getNormInput = [&](int inputPos) -> mlir::Value {
        if (inputPos >= slot.wiring.numInputs) return mlir::Value();
        int src = slot.wiring.inputSourceIndices[inputPos];

        // For external inputs, load directly with padded shape (power of 2)
        if (src < 0) {
          int extIdx = -(src + 1);
          if (extIdx < numExternalInputs && externalInputs && externalInputs[extIdx]) {
            auto extArgIt = std::find_if(result.args.begin(), result.args.end(),
                [src](const TritonKernelArg& arg) {
                    return arg.slotIndex == src;
                });
            if (extArgIt != result.args.end()) {
              int argIdx = extArgIt - result.args.begin();
              auto bufferArg = getBufferArg(argIdx);
              if (!bufferArg) return mlir::Value();

              // Get actual shape for this input
              auto sideShape = resolveShapeLocal(src);
              if (sideShape.empty()) return mlir::Value();

              int64_t sideNumElements = 1;
              std::vector<int64_t> sideShape64;
              for (auto d : sideShape) {
                sideShape64.push_back(static_cast<int64_t>(d));
                sideNumElements *= static_cast<int64_t>(d);
              }
              
              // Pad to power of 2 for Triton make_range
              int64_t sidePaddedLen = 1;
              while (sidePaddedLen < sideNumElements) sidePaddedLen *= 2;

              auto elemType = getMLIRType(builder, externalInputs[extIdx]->dataType());
              auto ptrType = mlir::cast<mlir::triton::PointerType>(bufferArg.getType());

              auto sideRange = builder.create<mlir::triton::MakeRangeOp>(loc,
                  mlir::RankedTensorType::get({sidePaddedLen}, builder.getI32Type()), 0, sidePaddedLen);
              auto sidePtrTensorType = mlir::RankedTensorType::get({sidePaddedLen}, ptrType);
              auto splatPtr = builder.create<mlir::triton::SplatOp>(loc, sidePtrTensorType, bufferArg);
              auto ptrs = builder.create<mlir::triton::AddPtrOp>(loc, sidePtrTensorType, splatPtr, sideRange);

              // Create mask for actual size
              auto sideLenConst = builder.create<mlir::arith::ConstantIntOp>(loc, sideNumElements, 32);
              auto splatSideLen = builder.create<mlir::triton::SplatOp>(loc,
                  mlir::RankedTensorType::get({sidePaddedLen}, builder.getI32Type()), sideLenConst);
              auto sideMask = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt, sideRange, splatSideLen);

              auto loaded = builder.create<mlir::triton::LoadOp>(loc, ptrs.getResult(), sideMask,
                  builder.create<mlir::triton::SplatOp>(loc, 
                      mlir::RankedTensorType::get({sidePaddedLen}, elemType),
                      builder.create<mlir::arith::ConstantOp>(loc, builder.getFloatAttr(elemType, 0.0))).getResult(),
                  mlir::triton::CacheModifier::NONE,
                  mlir::triton::EvictionPolicy::NORMAL, false);
              // Update SSA cache with correctly-shaped value
              ssaValues[src] = loaded;
              return loaded;
            }
          }
          return mlir::Value();
        }
        
        // Internal SSA input - use cached value
        auto it = ssaValues.find(src);
        if (it != ssaValues.end()) return it->second;
        return mlir::Value();
      };

      mlir::Value scaleVal, biasVal, meanVal, varianceVal;
      mlir::Value hiddenValue;  // pre-norm hidden for skip_rms_norm output[1]
      if (normKey == "batchnorm") {
        meanVal = getNormInput(1);
        varianceVal = getNormInput(2);
        scaleVal = getNormInput(3);
        biasVal = getNormInput(4);
      } else if (normKey == "skiprmsnorm") {
        // skip_rms_norm: input[0]=x, input[1]=skip, input[2]=gamma, input[3]=bias(optional)
        // The skip tensor has the SAME shape as x (full row), so it must be loaded
        // with row-wise addressing (paddedRowLen, makeRowOffsetRange), NOT via
        // getNormInput which uses sideNumElements (suitable for gamma/bias vectors).
        {
          int skipSrc = slot.wiring.inputSourceIndices[1];
          mlir::Value skipVal;
          // First check SSA cache (skip may come from a previous op in this kernel)
          auto skipSsaIt = ssaValues.find(skipSrc);
          if (skipSsaIt != ssaValues.end()) {
            skipVal = skipSsaIt->second;
          } else if (skipSrc < 0) {
            // External input — load with row-wise addressing like the primary input
            int skipExtIdx = -(skipSrc + 1);
            if (skipExtIdx < numExternalInputs && externalInputs && externalInputs[skipExtIdx]) {
              auto skipArgIt = std::find_if(result.args.begin(), result.args.end(),
                  [skipSrc](const TritonKernelArg& arg) {
                      return arg.slotIndex == skipSrc && !arg.isOutput;
                  });
              if (skipArgIt != result.args.end()) {
                int skipArgIdx = skipArgIt - result.args.begin();
                auto skipBufArg = getBufferArg(skipArgIdx);
                if (skipBufArg) {
                  auto skipElemType = getMLIRType(builder, externalInputs[skipExtIdx]->dataType());
                  auto skipPtrType = mlir::cast<mlir::triton::PointerType>(skipBufArg.getType());
                  auto skipRange = builder.create<mlir::triton::MakeRangeOp>(loc,
                      mlir::RankedTensorType::get({paddedRowLen}, builder.getI32Type()), 0, paddedRowLen);
                  auto skipRowOffsets = makeRowOffsetRange(skipRange);
                  auto skipPtrTensorType = mlir::RankedTensorType::get({paddedRowLen}, skipPtrType);
                  auto skipSplatPtr = builder.create<mlir::triton::SplatOp>(loc, skipPtrTensorType, skipBufArg);
                  auto skipPtrs = builder.create<mlir::triton::AddPtrOp>(loc, skipPtrTensorType, skipSplatPtr, skipRowOffsets);
                  auto skipRowLenConst = builder.create<mlir::arith::ConstantIntOp>(loc, logicalRowLen, 32);
                  auto skipSplatRowLen = builder.create<mlir::triton::SplatOp>(loc,
                      mlir::RankedTensorType::get({paddedRowLen}, builder.getI32Type()), skipRowLenConst);
                  auto skipMask = builder.create<mlir::arith::CmpIOp>(loc,
                      mlir::arith::CmpIPredicate::slt, skipRange, skipSplatRowLen);
                  skipVal = builder.create<mlir::triton::LoadOp>(loc, skipPtrs.getResult(), skipMask,
                      builder.create<mlir::triton::SplatOp>(loc,
                          mlir::RankedTensorType::get({paddedRowLen}, skipElemType),
                          builder.create<mlir::arith::ConstantOp>(loc, builder.getFloatAttr(skipElemType, 0.0))).getResult(),
                      mlir::triton::CacheModifier::NONE,
                      mlir::triton::EvictionPolicy::NORMAL, false);
                  ssaValues[skipSrc] = skipVal;
                }
              }
            }
          } else {
            // Internal input from another section — load from its buffer arg
            auto intArgIt = std::find_if(result.args.begin(), result.args.end(),
                [skipSrc](const TritonKernelArg& arg) {
                    return arg.slotIndex == skipSrc && !arg.isOutput;
                });
            if (intArgIt != result.args.end()) {
              int intArgIdx = intArgIt - result.args.begin();
              auto intBufArg = getBufferArg(intArgIdx);
              if (intBufArg) {
                auto skipElemType = getElementType(inputValue);
                auto skipPtrType = mlir::cast<mlir::triton::PointerType>(intBufArg.getType());
                auto skipRange = builder.create<mlir::triton::MakeRangeOp>(loc,
                    mlir::RankedTensorType::get({paddedRowLen}, builder.getI32Type()), 0, paddedRowLen);
                auto skipRowOffsets = makeRowOffsetRange(skipRange);
                auto skipPtrTensorType = mlir::RankedTensorType::get({paddedRowLen}, skipPtrType);
                auto skipSplatPtr = builder.create<mlir::triton::SplatOp>(loc, skipPtrTensorType, intBufArg);
                auto skipPtrs = builder.create<mlir::triton::AddPtrOp>(loc, skipPtrTensorType, skipSplatPtr, skipRowOffsets);
                auto skipRowLenConst = builder.create<mlir::arith::ConstantIntOp>(loc, logicalRowLen, 32);
                auto skipSplatRowLen = builder.create<mlir::triton::SplatOp>(loc,
                    mlir::RankedTensorType::get({paddedRowLen}, builder.getI32Type()), skipRowLenConst);
                auto skipMask = builder.create<mlir::arith::CmpIOp>(loc,
                    mlir::arith::CmpIPredicate::slt, skipRange, skipSplatRowLen);
                skipVal = builder.create<mlir::triton::LoadOp>(loc, skipPtrs.getResult(), skipMask,
                    builder.create<mlir::triton::SplatOp>(loc,
                        mlir::RankedTensorType::get({paddedRowLen}, skipElemType),
                        builder.create<mlir::arith::ConstantOp>(loc, builder.getFloatAttr(skipElemType, 0.0))).getResult(),
                    mlir::triton::CacheModifier::NONE,
                    mlir::triton::EvictionPolicy::NORMAL, false);
                ssaValues[skipSrc] = skipVal;
              }
            }
          }
          if (skipVal) {
            // Cast both operands to a common float type before adding.
            // inputValue and skipVal may be integer (e.g. i64) when the skip
            // connection comes from a strided_slice output of an INT64 array.
            // arith.addf requires floating-point operands — promote to f32 if needed.
            auto commonFTy = mlir::isa<mlir::FloatType>(getElementType(inputValue))
                                 ? getElementType(inputValue)
                                 : builder.getF32Type();
            inputValue = castTo(builder, loc, inputValue, commonFTy);
            skipVal = castTo(builder, loc, skipVal, commonFTy);
            inputValue = builder.create<mlir::arith::AddFOp>(loc, inputValue, skipVal);
          }
        }
        hiddenValue = inputValue;  // save pre-norm hidden for output[1]
        scaleVal = getNormInput(2);
        biasVal = getNormInput(3);
      } else {
        scaleVal = getNormInput(1);
        biasVal = getNormInput(2);
      }

      // Read epsilon from tArgs (first float argument), default 1e-5 if not set
      float epsilon = (slot.args.numTArgs > 0 && slot.args.tArgs) ? static_cast<float>(slot.args.tArgs[0]) : 1e-5f;

      // Pass logical row length to emitNormalizationOp for correct reduction
      // (tensor may be padded to power of 2, but we divide by logical width)
      auto opResult = emitNormalizationOp(builder, loc, slot.ident.opName, inputValue, axis, outputType,
                                          scaleVal, biasVal, meanVal, varianceVal, epsilon, logicalRowLen);
      // Safety: if normalization somehow returns a scalar, splat it back to tensor
      if (!mlir::isa<mlir::RankedTensorType>(opResult.getType())) {
        auto splatElemType = opResult.getType();
        auto splatTensorType = mlir::RankedTensorType::get({paddedRowLen}, splatElemType);
        opResult = builder.create<mlir::triton::SplatOp>(loc, splatTensorType, opResult);
      }
      opResult = emulateNativePrecision(opResult, si);

      if (normMultiRow) {
        // Multi-row normalization: store output inline with row-indexed addressing.
        // The generic store uses blockSize-wide tensors with contiguous offsets,
        // which doesn't match the per-row paddedRowLen-wide normalization output
        // when data rows are packed at logicalRowLen intervals (not paddedRowLen).
        for (int o = 0; o < slot.wiring.numOutputs; o++) {
          auto outSlot = slot.wiring.outputSlotIndices[o];
          // skip_rms_norm: output[0]=normed, output[1]=hidden (pre-norm input+skip)
          mlir::Value outVal = (o == 1 && hiddenValue) ? hiddenValue : opResult;
          // Find the output buffer arg
          auto outArgIt = std::find_if(result.args.begin(), result.args.end(),
              [outSlot](const TritonKernelArg& arg) {
                  return arg.slotIndex == outSlot && arg.isOutput;
              });
          if (outArgIt != result.args.end()) {
            int outArgIdx = outArgIt - result.args.begin();
            auto outBufArg = getBufferArg(outArgIdx);
            if (outBufArg) {
              auto elemType = getElementType(outVal);
              auto ptrType = mlir::triton::PointerType::get(elemType, 1);
              auto ptrTensorType = mlir::RankedTensorType::get({paddedRowLen}, ptrType);

              // Offsets: row offset + range[0..paddedRowLen)
              auto storeRange = builder.create<mlir::triton::MakeRangeOp>(loc,
                  mlir::RankedTensorType::get({paddedRowLen}, builder.getI32Type()), 0, paddedRowLen);
              auto storeOffsets = makeRowOffsetRange(storeRange);
              auto splatOutPtr = builder.create<mlir::triton::SplatOp>(loc, ptrTensorType, outBufArg);
              auto outPtrs = builder.create<mlir::triton::AddPtrOp>(loc, ptrTensorType, splatOutPtr, storeOffsets);

              // Mask: range < logicalRowLen (protect against padding overrun)
              auto storeMaskLenConst = builder.create<mlir::arith::ConstantIntOp>(loc, logicalRowLen, 32);
              auto storeMaskSplat = builder.create<mlir::triton::SplatOp>(loc,
                  mlir::RankedTensorType::get({paddedRowLen}, builder.getI32Type()), storeMaskLenConst);
              auto storeMask = builder.create<mlir::arith::CmpIOp>(loc,
                  mlir::arith::CmpIPredicate::slt, storeRange, storeMaskSplat);

              builder.create<mlir::triton::StoreOp>(loc, outPtrs, outVal, storeMask,
                  mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL);
              normInlineStoredOutputs.insert(outSlot);
            }
          }
          // Still put in SSA for any downstream consumers in the same kernel
          ssaValues[outSlot] = outVal;
        }
      } else {
        // Single-row: use generic store loop at end of buildModule
        for (int o = 0; o < slot.wiring.numOutputs; o++) {
          // skip_rms_norm: output[0]=normed, output[1]=hidden (pre-norm input+skip)
          mlir::Value outVal = (o == 1 && hiddenValue) ? hiddenValue : opResult;
          ssaValues[slot.wiring.outputSlotIndices[o]] = outVal;
        }
      }

    } else if (cat == TritonOpCategory::MATMUL) {
      // ── Check for fused_two_layer_mlp before generic matmul path ──
      {
        std::string opLower = slot.ident.opName;
        std::transform(opLower.begin(), opLower.end(), opLower.begin(), ::tolower);
        if ((opLower == "fused_two_layer_mlp" || opLower == "fusedtwolayermlp") &&
            slot.wiring.numInputs >= 5 && slot.wiring.numOutputs >= 1) {
          // This op needs the sectioned module builder; mark unsupported in 1D path
          // so it falls through to native execution rather than crashing.
          DSP_DIAG(FALLBACK, "fused_two_layer_mlp at slot %d — requires sectioned module, "
                   "skipping in 1D elementwise builder", si);
          goto next_slot;
        }
      }
      // ─── MATMUL: per-element scalar K-loop matmul (correct, no tensor cores) ───
      // For standalone matmul ops within a 1D element-wise segment.
      // Small pure-matmul segments go through buildMatmulModule instead.
      if (slot.wiring.numInputs >= 2 && slot.wiring.numOutputs >= 1) {
        int aSrc = slot.wiring.inputSourceIndices[0];
        int bSrc = slot.wiring.inputSourceIndices[1];
        int cSlot = slot.wiring.outputSlotIndices[0];

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
              for (int o = 0; o < slot.wiring.numOutputs; o++) {
                ssaValues[slot.wiring.outputSlotIndices[o]] = loaded;
              }
            }
          } else {
            std::string msg = "TritonIRBuilder: matmul '" + slot.ident.opName + "' at slot " + std::to_string(si) +
                " — missing kernel arg ptrs for A(" + std::to_string(aSrc) + ")/B(" + std::to_string(bSrc) +
                ")/C(" + std::to_string(cSlot) + "). Cannot compile.";
            THROW_EXCEPTION(msg.c_str());
          }
        } else {
          std::string msg = "TritonIRBuilder: matmul '" + slot.ident.opName + "' at slot " + std::to_string(si) +
              " — M=" + std::to_string(M) + " N=" + std::to_string(N) + " K=" + std::to_string(K) +
              " invalid dimensions. Cannot compile.";
          THROW_EXCEPTION(msg.c_str());
        }
      } else {
        std::string msg = "TritonIRBuilder: matmul '" + slot.ident.opName + "' at slot " + std::to_string(si) +
            " — needs >=2 inputs and >=1 output, has " + std::to_string(slot.wiring.numInputs) + "/" +
            std::to_string(slot.wiring.numOutputs) + ". Cannot compile.";
        THROW_EXCEPTION(msg.c_str());
      }

    } else if (cat == TritonOpCategory::FUSED_ATTENTION) {
      // ─── FUSED ATTENTION BACKWARD: Flash Attention 2 backward pass ─────────
      // Backward ops are identified by the _bp suffix. Input layout (BHSD):
      //   input[0] = dO [B, H, seqQ, HD]  — upstream gradient of output
      //   input[1] = Q  [B, H, seqQ, HD]
      //   input[2] = K  [B, H, seqK,  HD]
      //   input[3] = V  [B, H, seqK,  HD]
      //   input[4] = O  [B, H, seqQ, HD]  — forward output
      //   input[5] = L  [B, H, seqQ]      — log-sum-exp from forward softmax
      //   output[0] = dQ [B, H, seqQ, HD]
      //   output[1] = dK [B, H, seqK,  HD]
      //   output[2] = dV [B, H, seqK,  HD]
      {
        std::string opLowerBp = slot.ident.opName;
        std::transform(opLowerBp.begin(), opLowerBp.end(), opLowerBp.begin(), ::tolower);
        bool isBackward = (opLowerBp.find("_bp") != std::string::npos);
        if (isBackward) {
          // Need 6 inputs (dO, Q, K, V, O, L) and 3 outputs (dQ, dK, dV)
          if (slot.wiring.numInputs >= 6 && slot.wiring.numOutputs >= 3) {
            int doSrc  = slot.wiring.inputSourceIndices[0];  // dO
            int qSrc   = slot.wiring.inputSourceIndices[1];  // Q
            int kSrc   = slot.wiring.inputSourceIndices[2];  // K
            int vSrc   = slot.wiring.inputSourceIndices[3];  // V
            int oSrc   = slot.wiring.inputSourceIndices[4];  // O (forward output)
            int lseSrc = slot.wiring.inputSourceIndices[5];  // L (log-sum-exp)
            int dqSlot = slot.wiring.outputSlotIndices[0];   // dQ
            int dkSlot = slot.wiring.outputSlotIndices[1];   // dK
            int dvSlot = slot.wiring.outputSlotIndices[2];   // dV

            NDArray* qArr = resolveArr(qSrc);
            NDArray* kArr = resolveArr(kSrc);
            int batchSize = 1, numQHeads = 1, seqQ = 1, seqK = 1, headDim = 64;
            if (qArr && qArr->rankOf() >= 4) {
              // BHSD: [batch, heads, seqQ, headDim]
              batchSize = static_cast<int>(qArr->sizeAt(0));
              numQHeads = static_cast<int>(qArr->sizeAt(1));
              seqQ      = static_cast<int>(qArr->sizeAt(2));
              headDim   = static_cast<int>(qArr->sizeAt(3));
            }
            if (kArr && kArr->rankOf() >= 4) {
              seqK = static_cast<int>(kArr->sizeAt(2));
            }
            if (seqQ <= 0 || seqK <= 0 || headDim <= 0) {
              DSP_DIAG(COMPILE, "FUSED_ATTENTION_BP at slot %d: invalid dims "
                        "seqQ=%d seqK=%d headDim=%d — deferring to C++ native", si, seqQ, seqK, headDim);
              return result;
            }

            float scale = 1.0f / std::sqrt(static_cast<float>(headDim));
            auto bpTile = chooseFusedAttentionTileConfig(batchSize, numQHeads, seqQ, seqK, headDim);
            if (!bpTile.fitsSharedMem) {
              std::string msg = "TritonIRBuilder: fused attention backward '" + slot.ident.opName +
                  "' at slot " + std::to_string(si) + " cannot fit shared memory";
              THROW_EXCEPTION(msg.c_str());
            }
            int blockM = bpTile.blockM;
            int blockN = bpTile.blockN;

            auto dOPtrV  = getSlotArgPtr(doSrc);
            auto qPtrV   = getSlotArgPtr(qSrc);
            auto kPtrV   = getSlotArgPtr(kSrc);
            auto vPtrV   = getSlotArgPtr(vSrc);
            auto oPtrV   = getSlotArgPtr(oSrc);
            auto lsePtrV = getSlotArgPtr(lseSrc);
            auto dQPtrV  = getSlotArgPtr(dqSlot);
            auto dKPtrV  = getSlotArgPtr(dkSlot);
            auto dVPtrV  = getSlotArgPtr(dvSlot);

            if (dOPtrV && qPtrV && kPtrV && vPtrV && oPtrV && lsePtrV &&
                dQPtrV && dKPtrV && dVPtrV) {
              emitFusedAttentionBackwardKernel(
                  builder, loc,
                  dOPtrV, qPtrV, kPtrV, vPtrV, oPtrV, lsePtrV,
                  dQPtrV, dKPtrV, dVPtrV,
                  batchSize, numQHeads, seqQ, seqK, headDim, scale, blockM, blockN);

              // Load results back into SSA value table so downstream consumers
              // can reference them (the backward writes to output slot buffers directly)
              for (int outIdx = 0; outIdx < slot.wiring.numOutputs; outIdx++) {
                int outSlotIdx = slot.wiring.outputSlotIndices[outIdx];
                NDArray* outArr = resolveArr(outSlotIdx);
                DataType outDtype = FLOAT32;
                if (outArr) outDtype = outArr->dataType();
                auto loaded = loadBackFromBuffer(outSlotIdx, outDtype);
                if (loaded) ssaValues[outSlotIdx] = loaded;
              }
            } else {
              std::string msg = "TritonIRBuilder: fused attention backward '" +
                  slot.ident.opName + "' at slot " + std::to_string(si) +
                  " — missing kernel arg ptrs. Cannot compile.";
              THROW_EXCEPTION(msg.c_str());
            }
          } else {
            DSP_DIAG(COMPILE, "FUSED_ATTENTION_BP at slot %d: needs >=6 inputs and >=3 outputs, "
                      "has %d/%d — deferring to C++ native", si,
                      slot.wiring.numInputs, slot.wiring.numOutputs);
            return result;  // result.valid = false → C++ fallback
          }
          goto next_slot;  // skip the forward attention handler below
        }
      }

      // ─── FUSED ATTENTION (forward): Q@K^T + scale + softmax + @V ───────────
      // Handles past_key/past_value (inputs 4-5) and BSHD (3D) vs BHSD (4D) layout.
      if (slot.wiring.numInputs >= 3 && slot.wiring.numOutputs >= 1) {
        int qSrc = slot.wiring.inputSourceIndices[0];
        // dot_product_attention_v2 input order: (Q=0, V=1, K=2)
        // onnx_multi_head_attention input order: (Q=0, K=1, V=2)
        // Detect op name to swap K/V source indices for DPA v2.
        std::string opLowerKV = slot.ident.opName;
        std::transform(opLowerKV.begin(), opLowerKV.end(), opLowerKV.begin(), ::tolower);
        bool isDpaV2 = (opLowerKV.find("dot_product_attention") != std::string::npos);

        int kSrc = isDpaV2 ? slot.wiring.inputSourceIndices[2] : slot.wiring.inputSourceIndices[1];
        int vSrc = isDpaV2 ? slot.wiring.inputSourceIndices[1] : slot.wiring.inputSourceIndices[2];
        int outSlot = slot.wiring.outputSlotIndices[0];

        // onnx_multi_head_attention is a compound op: it takes 3D Q/K/V [B,S,H*D],
        // internally reshapes to 4D, concatenates past_key/past_value with current K/V,
        // runs attention, then reshapes output back to 3D. When this op appears as a
        // single-slot segment, we handle the 3D→4D reshape via dual-buffer mode:
        // the Triton kernel reads past_key/past_value (4D BHSD) as the main K/V buffers
        // and the current 3D K/V as secondary buffers with implicit reshape.
        NDArray* qArr = resolveArr(qSrc);
        bool qIs3D = (qArr && qArr->rankOf() == 3);

        // Extract attention dimensions
        int batchSize = 1, numQHeads = 1, numKvHeads = 0, seqQ = 1, seqK = 1, headDim = 64;
        bool qIsBSHD = false;

        // Detect BSHD vs BHSD layout from op name.
        // dot_product_attention_v2 uses BSHD: [batch, seq, heads, headDim]
        bool opUsesBSHD = isDpaV2;

        if (qArr && qArr->rankOf() >= 4) {
          batchSize = static_cast<int>(qArr->sizeAt(0));
          if (opUsesBSHD) {
            // BSHD: [batch, seq, heads, headDim]
            seqQ = static_cast<int>(qArr->sizeAt(1));
            numQHeads = static_cast<int>(qArr->sizeAt(2));
            headDim = static_cast<int>(qArr->sizeAt(3));
            qIsBSHD = true;
          } else {
            // BHSD: [batch, heads, seq, headDim]
            numQHeads = static_cast<int>(qArr->sizeAt(1));
            seqQ = static_cast<int>(qArr->sizeAt(2));
            headDim = static_cast<int>(qArr->sizeAt(3));
          }
        } else if (qIs3D) {
          // 3D Q: [B, seqQ, H*D] — compound attention (onnx_multi_head_attention)
          batchSize = static_cast<int>(qArr->sizeAt(0));
          seqQ = static_cast<int>(qArr->sizeAt(1));
          int hidden = static_cast<int>(qArr->sizeAt(2));
          // numQHeads from iArgs[0] (INT_ARG(0) in onnx_multi_head_attention)
          numQHeads = (slot.args.numIArgs > 0 && slot.args.iArgs) ? static_cast<int>(slot.args.iArgs[0]) : 1;
          if (numQHeads <= 0) numQHeads = 1;
          headDim = hidden / numQHeads;
          qIsBSHD = true;
          DSP_DIAG(JIT, "TritonIRBuilder: fused attention '%s' at slot %d has 3D Q [%lld,%lld,%lld] "
                    "— compound op, numQHeads=%d (from iArgs[0]), headDim=%d, using dual-buffer mode",
                    slot.ident.opName, si,
                    (long long)qArr->sizeAt(0), (long long)qArr->sizeAt(1),
                    (long long)qArr->sizeAt(2), numQHeads, headDim);
        }

        // Detect past_key/past_value by scanning ALL inputs for 4D KV-cache-like shapes.
        // A past_key tensor is 4D BHSD: [batch, kvHeads, seqK, headDim] where headDim
        // matches Q's headDim. This distinguishes it from attention masks [B,H,S,S].
        bool hasPastKv = false;
        int pastKeySrc = -1, pastValueSrc = -1;

        for (int inp = 3; inp < slot.wiring.numInputs && !hasPastKv; inp++) {
          int candidateSrc = slot.wiring.inputSourceIndices[inp];
          NDArray* candidateArr = resolveArr(candidateSrc);
          if (candidateArr && candidateArr->rankOf() == 4) {
            int candidateHD = static_cast<int>(candidateArr->sizeAt(3));
            int candidateKvH = static_cast<int>(candidateArr->sizeAt(1));
            // GQA constraint: KV heads must divide Q heads evenly and be <= numQHeads
            if (candidateHD == headDim && candidateKvH > 0 &&
                candidateKvH <= numQHeads && numQHeads % candidateKvH == 0) {
              pastKeySrc = candidateSrc;
              hasPastKv = true;
              DSP_DIAG(JIT, "ATTN slot=%d found past_key at input[%d] src=%d shape=[%lld,%lld,%lld,%lld] headDim=%d",
                        si, inp, candidateSrc,
                        (long long)candidateArr->sizeAt(0), (long long)candidateArr->sizeAt(1),
                        (long long)candidateArr->sizeAt(2), (long long)candidateArr->sizeAt(3), headDim);
              if (inp + 1 < slot.wiring.numInputs) {
                int pvCandidate = slot.wiring.inputSourceIndices[inp + 1];
                NDArray* pvArr = resolveArr(pvCandidate);
                if (pvArr && pvArr->rankOf() == 4 && static_cast<int>(pvArr->sizeAt(3)) == headDim) {
                  pastValueSrc = pvCandidate;
                }
              }
            }
          }
        }

        if (!hasPastKv) {
          DSP_DIAG(JIT, "ATTN slot=%d no past_key found (headDim=%d) in %d inputs",
                    si, headDim, slot.wiring.numInputs);
        }

        // Determine if we need dual-buffer mode (3D Q with past_key)
        bool useDualBuffer = (qIs3D && hasPastKv);
        int pastSeqLen = 0, seqKVCur = 0;

        // Use past_key as effective K source when available
        int effectiveKSrc = hasPastKv ? pastKeySrc : kSrc;
        int effectiveVSrc = (hasPastKv && pastValueSrc >= 0) ? pastValueSrc : vSrc;

        NDArray* effectiveKArr = resolveArr(effectiveKSrc);

        // Extract KV head count from effective K shape (4D BHSD: [B, KvHeads, seqK, HD])
        if (effectiveKArr && effectiveKArr->rankOf() == 4) {
          if (hasPastKv) {
            numKvHeads = static_cast<int>(effectiveKArr->sizeAt(1));
            headDim = static_cast<int>(effectiveKArr->sizeAt(3));
          } else {
            // No past KV — K shape is same layout as Q (BHSD or BSHD)
            if (qIsBSHD) {
              numKvHeads = static_cast<int>(effectiveKArr->sizeAt(2));
            } else {
              numKvHeads = static_cast<int>(effectiveKArr->sizeAt(1));
            }
          }
        } else if (effectiveKArr && effectiveKArr->rankOf() == 3) {
          // ONNX MHA K/V stay 3D for GQA: [B, seqK, kvHeads * headDim].
          // Infer KV heads from kvHidden instead of assuming numQHeads.
          int kvHidden = static_cast<int>(effectiveKArr->sizeAt(2));
          if (headDim > 0 && kvHidden > 0 && kvHidden % headDim == 0) {
            int inferredKvHeads = kvHidden / headDim;
            if (inferredKvHeads > 0 &&
                inferredKvHeads <= numQHeads &&
                numQHeads % inferredKvHeads == 0) {
              numKvHeads = inferredKvHeads;
            }
          }
        }
        if (numKvHeads <= 0) numKvHeads = numQHeads;

        if (useDualBuffer) {
          // past_key shape is 4D BHSD: [B, kvH, pastSeq, D]
          if (effectiveKArr && effectiveKArr->rankOf() == 4) {
            pastSeqLen = static_cast<int>(effectiveKArr->sizeAt(2));
          }
          NDArray* curKArr = resolveArr(kSrc);
          seqKVCur = (curKArr && curKArr->rankOf() == 3) ? static_cast<int>(curKArr->sizeAt(1)) : 1;
          seqK = pastSeqLen + seqKVCur;
          DSP_DIAG(JIT, "ATTN slot=%d dual-buffer: pastSeqLen=%d seqKVCur=%d seqK=%d numKvHeads=%d",
                    si, pastSeqLen, seqKVCur, seqK, numKvHeads);
        } else {
          // seqK from effective K source
          if (effectiveKArr && effectiveKArr->rankOf() >= 4) {
            if (opUsesBSHD && !hasPastKv) {
              // BSHD K/V: [batch, seqK, heads, headDim]
              seqK = static_cast<int>(effectiveKArr->sizeAt(1));
            } else {
              // BHSD K/V: [batch, heads, seqK, headDim]
              seqK = static_cast<int>(effectiveKArr->sizeAt(2));
            }
          } else if (effectiveKArr && effectiveKArr->rankOf() == 3) {
            seqK = static_cast<int>(effectiveKArr->sizeAt(1));
          }
        }

        // past_key is always 4D BHSD; current key follows Q layout
        bool kIsBSHD = hasPastKv ? false : qIsBSHD;

        // seqK=0 means shapes are stale (cached from warmup).
        // Try deriving seqK from actual external inputs (same strategies as sectioned path).
        bool seqKDerivedFromExternalJit = false;
        if (seqK <= 0) {
          int derivedSeqK = 0;
          int derivedKvHeads = 0;

          // Strategy 1: Walk back from K source to find KV cache external inputs.
          if (kSrc >= 0) {
            bool kProducerFoundJit = false;
            for (int s = startSlot; s <= endSlot && !kProducerFoundJit; s++) {
              for (int o = 0; o < slots[s].wiring.numOutputs && !kProducerFoundJit; o++) {
                if (slots[s].wiring.outputSlotIndices[o] == kSrc) {
                  for (int pi = 0; pi < slots[s].wiring.numInputs; pi++) {
                    int psrc = slots[s].wiring.inputSourceIndices[pi];
                    if (psrc < 0) {
                      int extIdx = -(psrc + 1);
                      if (extIdx < numExternalInputs && externalInputs && externalInputs[extIdx]) {
                        auto& ext = *externalInputs[extIdx];
                        if (ext.rankOf() == 4 && !ext.isEmpty() &&
                            (ext.dataType() == FLOAT32 || ext.dataType() == HALF || ext.dataType() == BFLOAT16)) {
                          int extSeqK = static_cast<int>(ext.sizeAt(2));
                          int extHD = static_cast<int>(ext.sizeAt(3));
                          int extKvH = static_cast<int>(ext.sizeAt(1));
                          if (extHD == headDim && extSeqK > derivedSeqK &&
                              extKvH > 0 && extKvH <= numQHeads && numQHeads % extKvH == 0) {
                            derivedSeqK = extSeqK;
                            derivedKvHeads = extKvH;
                          }
                        }
                      }
                    }
                  }
                  kProducerFoundJit = true;
                }
              }
            }
          }

          // Strategy 2: Scan ALL external inputs for 4D KV cache pattern.
          if (derivedSeqK == 0 && externalInputs) {
            for (int ei = 0; ei < numExternalInputs; ei++) {
              if (!externalInputs[ei]) continue;
              auto& ext = *externalInputs[ei];
              if (ext.rankOf() != 4 || ext.isEmpty()) continue;
              if (ext.dataType() != FLOAT32 && ext.dataType() != HALF && ext.dataType() != BFLOAT16) continue;
              int extBatch = static_cast<int>(ext.sizeAt(0));
              int extHD = static_cast<int>(ext.sizeAt(3));
              int extSeqK = static_cast<int>(ext.sizeAt(2));
              if (extBatch == batchSize && extHD == headDim && extSeqK > 0) {
                int extHeads = static_cast<int>(ext.sizeAt(1));
                if (extHeads > 0 && extHeads <= numQHeads && numQHeads % extHeads == 0 && extSeqK > derivedSeqK) {
                  derivedSeqK = extSeqK;
                  derivedKvHeads = extHeads;
                }
              }
            }
          }

          if (derivedSeqK > 0) {
            seqK = derivedSeqK + seqQ;
            seqKDerivedFromExternalJit = true;
            if (derivedKvHeads > 0 && derivedKvHeads != numKvHeads) {
              DSP_DIAG(JIT, "ATTN slot=%d (JIT) correcting numKvHeads from %d to %d",
                        si, numKvHeads, derivedKvHeads);
              numKvHeads = derivedKvHeads;
            }
            DSP_DIAG(JIT, "ATTN slot=%d (JIT) derived seqK=%d from external inputs (pastSeqK=%d + seqQ=%d)",
                      si, seqK, derivedSeqK, seqQ);
          } else {
            DSP_DIAG(COMPILE, "FUSED_ATTENTION at slot %d: seqK=%d — "
                      "deferring to C++ native (shapes not yet resolved)", si, seqK);
            return result;  // result.valid = false → C++ fallback
          }
        }

        float scale = 1.0f / std::sqrt(static_cast<float>(headDim));
        auto attnTile = chooseFusedAttentionTileConfig(
            batchSize, numQHeads, seqQ, seqK, headDim);
        if (!attnTile.fitsSharedMem) {
          std::string msg = "TritonIRBuilder: fused attention '" + slot.ident.opName + "' at slot " +
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
        auto outPtr = getSlotArgPtr(outSlot);

        // For dual-buffer: kPtr/vPtr = past_key/past_value (BHSD), curKPtr/curVPtr = current key/value (BSHD)
        mlir::Value kPtr, vPtr, curKPtr, curVPtr;
        if (useDualBuffer) {
          // past_key/value are the main K/V buffers (BHSD layout)
          kPtr = getSlotArgPtr(pastKeySrc);
          vPtr = getSlotArgPtr(pastValueSrc);
          // current key/value are the secondary buffers (3D BSHD layout)
          curKPtr = getSlotArgPtr(kSrc);
          curVPtr = getSlotArgPtr(vSrc);
        } else {
          kPtr = getSlotArgPtr(effectiveKSrc);
          vPtr = getSlotArgPtr(effectiveVSrc);
        }

        // Resolve attention bias/mask (input[3]) if present and non-empty
        mlir::Value biasPtr;
        std::vector<LongType> biasShape;
        if (slot.wiring.numInputs > 3) {
          int biasSrc = slot.wiring.inputSourceIndices[3];
          NDArray* biasArr = resolveArr(biasSrc);
          // Only use bias if it's a real tensor (not empty/scalar placeholder)
          if (biasArr && !biasArr->isEmpty() && biasArr->rankOf() >= 2 && biasArr->lengthOf() > 1) {
            biasPtr = getSlotArgPtr(biasSrc);
            for (int d = 0; d < biasArr->rankOf(); d++) {
              biasShape.push_back(biasArr->sizeAt(d));
            }
            DSP_DIAG(JIT, "TritonIRBuilder: fused attention bias: slot=%d rank=%d len=%lld",
                      biasSrc, biasArr->rankOf(),
                      (long long)biasArr->lengthOf());
          }
        }

        // Validate K buffer is non-empty when seqK > 0.
        // When K buffer is empty, the kernel would read from empty buffers causing
        // illegal memory access (CUDA error 700). Always fall back to C++.
        //
        // For dual-buffer mode (static KV cache), the past_key buffer is pre-allocated
        // to max sequence length and reused across decode steps. In this case, we check
        // the buffer shape/capacity rather than content length, since the buffer may
        // contain stale data from previous steps but is still valid for attention.
        bool kBufferValidJit = true;
        {
          NDArray* effKArr = resolveArr(effectiveKSrc);
          if (!effKArr && seqK > 0) {
            kBufferValidJit = false;
            DSP_DIAG(JIT, "TritonIRBuilder: skipping FUSED_ATTENTION at slot %d (JIT path) — "
                      "effective K buffer (src=%d) is null but seqK=%d%s",
                      si, effectiveKSrc, seqK,
                      seqKDerivedFromExternalJit ? " (seqK derived from external inputs)" : "");
          } else if (effKArr && seqK > 0) {
            // For dual-buffer mode (static KV cache), check shape capacity not content
            if (useDualBuffer) {
              // past_key is 4D BHSD: [B, KvHeads, maxSeqLen, headDim]
              // Valid if rank == 4 and seq dimension (dim 2) > 0
              bool shapeValid = (effKArr->rankOf() == 4 && effKArr->sizeAt(2) > 0);
              if (!shapeValid) {
                kBufferValidJit = false;
                DSP_DIAG(JIT, "TritonIRBuilder: skipping FUSED_ATTENTION at slot %d (JIT path) — "
                          "past_key buffer (src=%d) has invalid shape: rank=%d, seqDim=%d (expected 4D BHSD with seqDim>0)",
                          si, effectiveKSrc, effKArr->rankOf(),
                          effKArr->rankOf() >= 3 ? static_cast<int>(effKArr->sizeAt(2)) : -1);
              }
            } else {
              // Non-dual-buffer: check content length (original behavior)
              if (effKArr->isEmpty() || effKArr->lengthOf() == 0) {
                kBufferValidJit = false;
                DSP_DIAG(JIT, "TritonIRBuilder: skipping FUSED_ATTENTION at slot %d (JIT path) — "
                          "effective K buffer (src=%d) is empty but seqK=%d%s",
                          si, effectiveKSrc, seqK,
                          seqKDerivedFromExternalJit ? " (seqK derived from external inputs)" : "");
              }
            }
          }
        }

        if (!kBufferValidJit) {
          DSP_DIAG(JIT, "ATTN slot=%d: K buffer invalid (JIT path), returning as non-compilable", si);
          return result;  // result.valid = false → C++ fallback
        }

        if (qPtr && kPtr && vPtr && outPtr) {
          emitFusedAttentionKernel(builder, loc, qPtr, kPtr, vPtr, outPtr,
                                   batchSize, numQHeads, numKvHeads, seqQ, seqK, headDim,
                                   scale, blockM, blockN, qIsBSHD, kIsBSHD,
                                   biasPtr, biasShape,
                                   curKPtr, curVPtr, pastSeqLen, seqKVCur);

          // output[0] = attention result
          DataType outDtype = FLOAT32;
          NDArray* outArr = resolveArr(outSlot);
          if (outArr) outDtype = outArr->dataType();
          auto loaded = loadBackFromBuffer(outSlot, outDtype);
          if (loaded) ssaValues[outSlot] = loaded;

          // output[1] = present_key, output[2] = present_value
          if (useDualBuffer && slot.wiring.numOutputs >= 2) {
            // Dual-buffer: present_key/value output may need write of current K/V
            // at pastSeqLen offset. Check if the output buffer can hold it.
            int presentKeySlot = slot.wiring.outputSlotIndices[1];
            NDArray* pkOutArr = resolveArr(presentKeySlot);
            int pkSeqCapacity = (pkOutArr && pkOutArr->rankOf() == 4) ? static_cast<int>(pkOutArr->sizeAt(2)) : 0;
            int requiredSeq = pastSeqLen + seqKVCur;
            bool pkFits = (pkSeqCapacity >= requiredSeq);
            if (!pkFits) {
              DSP_DIAG(JIT, "TritonIRBuilder: skipping present_key/value write at slot %d — "
                        "output buffer seqDim=%d < required=%d (pastSeqLen=%d + seqKVCur=%d). "
                        "Static KV cache detected; caller handles cache updates.",
                        si, pkSeqCapacity, requiredSeq, pastSeqLen, seqKVCur);
            }
            if (pkFits) {
              auto presentKeyPtr = getSlotArgPtr(presentKeySlot);
              if (presentKeyPtr && curKPtr) {
                int totalSeq = pastSeqLen + seqKVCur;
                emitPresentKvWrite(builder, loc, curKPtr, presentKeyPtr,
                                   batchSize, numQHeads, numKvHeads,
                                   pastSeqLen, seqKVCur, totalSeq, headDim);
                DataType pkDtype = FLOAT32;
                NDArray* pkArr = resolveArr(presentKeySlot);
                if (pkArr) pkDtype = pkArr->dataType();
                auto pkLoaded = loadBackFromBuffer(presentKeySlot, pkDtype);
                if (pkLoaded) ssaValues[presentKeySlot] = pkLoaded;
              }
            }
            if (slot.wiring.numOutputs >= 3 && pkFits) {
              int presentValSlot = slot.wiring.outputSlotIndices[2];
              auto presentValPtr = getSlotArgPtr(presentValSlot);
              if (presentValPtr && curVPtr) {
                int totalSeq = pastSeqLen + seqKVCur;
                emitPresentKvWrite(builder, loc, curVPtr, presentValPtr,
                                   batchSize, numQHeads, numKvHeads,
                                   pastSeqLen, seqKVCur, totalSeq, headDim);
                DataType pvDtype = FLOAT32;
                NDArray* pvArr = resolveArr(presentValSlot);
                if (pvArr) pvDtype = pvArr->dataType();
                auto pvLoaded = loadBackFromBuffer(presentValSlot, pvDtype);
                if (pvLoaded) ssaValues[presentValSlot] = pvLoaded;
              }
            }
          } else {
            // Non-dual-buffer: pass-through effective key/value SSA
            // output[1] = present_key (pass-through effective key)
            if (slot.wiring.numOutputs >= 2) {
              if (ssaValues.count(effectiveKSrc)) {
                ssaValues[slot.wiring.outputSlotIndices[1]] = ssaValues[effectiveKSrc];
              } else {
                DataType kDtype = FLOAT32;
                NDArray* kArr2 = resolveArr(effectiveKSrc);
                if (kArr2) kDtype = kArr2->dataType();
                auto kLoaded = loadBackFromBuffer(effectiveKSrc, kDtype);
                if (kLoaded) ssaValues[slot.wiring.outputSlotIndices[1]] = kLoaded;
              }
            }
            // output[2] = present_value (pass-through effective value)
            if (slot.wiring.numOutputs >= 3) {
              if (ssaValues.count(effectiveVSrc)) {
                ssaValues[slot.wiring.outputSlotIndices[2]] = ssaValues[effectiveVSrc];
              } else {
                DataType vDtype = FLOAT32;
                NDArray* vArr2 = resolveArr(effectiveVSrc);
                if (vArr2) vDtype = vArr2->dataType();
                auto vLoaded = loadBackFromBuffer(effectiveVSrc, vDtype);
                if (vLoaded) ssaValues[slot.wiring.outputSlotIndices[2]] = vLoaded;
              }
            }
          }
        } else {
          std::string msg = "TritonIRBuilder: fused attention '" + slot.ident.opName + "' at slot " + std::to_string(si) +
              " — missing kernel arg ptrs. Cannot compile.";
          THROW_EXCEPTION(msg.c_str());
        }
      } else {
        std::string msg = "TritonIRBuilder: fused attention '" + slot.ident.opName + "' at slot " + std::to_string(si) +
            " — needs >=3 inputs and >=1 output, has " + std::to_string(slot.wiring.numInputs) + "/" +
            std::to_string(slot.wiring.numOutputs) + ". Cannot compile.";
        THROW_EXCEPTION(msg.c_str());
      }

    } else if (cat == TritonOpCategory::SHAPE_MANIPULATION) {
      // ─── SHAPE MANIPULATION ───
      // reshape/squeeze/expand_dims/flatten: SSA forwarding (same data, different view)
      // permute/transpose: need actual data reindexing via emitShapeManipulationSection
      std::string opLower = slot.ident.opName;
      std::transform(opLower.begin(), opLower.end(), opLower.begin(), ::tolower);

      bool isPermute = (opLower == "permute" || opLower == "transpose");

      if (isPermute && slot.wiring.numInputs >= 1 && slot.wiring.numOutputs >= 1) {
        // Permute/transpose requires actual data movement
        int inputSrc = slot.wiring.inputSourceIndices[0];
        int outSlot = slot.wiring.outputSlotIndices[0];
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
          if (slot.args.numIArgs > 0 && slot.args.iArgs) {
            for (int d = 0; d < slot.args.numIArgs; d++)
              permutation.push_back(static_cast<int>(slot.args.iArgs[d]));
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
            for (int o = 0; o < slot.wiring.numOutputs; o++) {
              ssaValues[slot.wiring.outputSlotIndices[o]] = loaded;
            }
          }
        } else {
          std::string msg = "TritonIRBuilder: permute/transpose '" + slot.ident.opName + "' at slot " + std::to_string(si) +
              " — missing kernel arg ptrs. Cannot compile.";
          THROW_EXCEPTION(msg.c_str());
        }
      } else if (slot.wiring.numInputs >= 1) {
        // reshape/squeeze/expand_dims/flatten: pure SSA forwarding (same data buffer)
        int inputSrc = slot.wiring.inputSourceIndices[0];
        auto inputIt = ssaValues.find(inputSrc);
        if (inputIt != ssaValues.end()) {
          for (int o = 0; o < slot.wiring.numOutputs; o++) {
            ssaValues[slot.wiring.outputSlotIndices[o]] = inputIt->second;
          }
        } else {
          DSP_DIAG_SLOT(FALLBACK, si, "TritonIRBuilder: missing SSA value for shape op '%s' at slot %d (src=%d)",
                    slot.ident.opName.c_str(), si, inputSrc);
        }
      }

    } else if (cat == TritonOpCategory::ROPE) {
      // ─── ROPE: paired elementwise rotation with precomputed cos/sin ───
      if (slot.wiring.numInputs >= 3 && slot.wiring.numOutputs >= 1) {
        int inputSrc = slot.wiring.inputSourceIndices[0];
        int cosSrc = slot.wiring.inputSourceIndices[1];
        int sinSrc = slot.wiring.inputSourceIndices[2];
        int outSlot = slot.wiring.outputSlotIndices[0];

        auto inPtr = getSlotArgPtr(inputSrc);
        auto cosArgPtr = getSlotArgPtr(cosSrc);
        auto sinArgPtr = getSlotArgPtr(sinSrc);
        auto outPtr = getSlotArgPtr(outSlot);
        NDArray* inArr = resolveArr(inputSrc);
        NDArray* cosArr = resolveArr(cosSrc);
        NDArray* outArr = resolveArr(outSlot);

        // cos/sin MUST have kernel arg pointers (force-external ensures this).
        // inPtr/outPtr may be null if input/output are internal SSA intermediates.
        // inArr may be null if input is internal — fall back to resolveShapeLocal.
        // outArr may be null if output is internal — derive from input shape.
        std::vector<LongType> inShape, cosShapeVec;

        // Resolve input shape: prefer live array, fall back to cached shape info
        if (inArr) {
          for (int d = 0; d < inArr->rankOf(); d++) inShape.push_back(inArr->sizeAt(d));
        } else {
          inShape = resolveShapeLocal(inputSrc);
        }

        // Resolve cos shape
        if (cosArr) {
          for (int d = 0; d < cosArr->rankOf(); d++) cosShapeVec.push_back(cosArr->sizeAt(d));
        } else {
          cosShapeVec = resolveShapeLocal(cosSrc);
        }

        // Compute nElements from outArr if available, else from input shape
        int nElements = 0;
        if (outArr) {
          nElements = static_cast<int>(outArr->lengthOf());
        } else if (!inShape.empty()) {
          nElements = 1;
          for (auto d : inShape) nElements *= static_cast<int>(d);
        }

        if (cosArgPtr && sinArgPtr && !inShape.empty() && !cosShapeVec.empty() && nElements > 0) {
          int ropeType = (slot.args.numIArgs > 0 && slot.args.iArgs) ? static_cast<int>(slot.args.iArgs[0]) : 0;

          // Extract headDim and numHeads from input shape [B, S, H, D]
          int inputRank = static_cast<int>(inShape.size());
          int headDim = (inputRank > 0) ? static_cast<int>(inShape[inputRank - 1]) : 0;
          int numHeads = (inputRank >= 3) ? static_cast<int>(inShape[inputRank - 2]) : 1;

          // Try SSA register-level path: requires blockSize divisible by headDim
          // and all elements in block from same seq position (blockSize <= numHeads * headDim)
          auto ssaIt = ssaValues.find(inputSrc);
          bool canUseSSA = ssaIt != ssaValues.end()
                           && headDim > 0 && (headDim % 2 == 0)
                           && (blockSize % headDim == 0)
                           && (blockSize <= numHeads * headDim);

          if (canUseSSA) {
            // Register-based ROPE — no store/reload needed
            auto result = emitRoPESSA(builder, loc, ssaIt->second,
                                       cosArgPtr, sinArgPtr, pid, blockSize,
                                       headDim, numHeads, cosShapeVec, nElements);
            result = emulateNativePrecision(result, si);
            for (int o = 0; o < slot.wiring.numOutputs; o++)
              ssaValues[slot.wiring.outputSlotIndices[o]] = result;
          } else if (inPtr && outPtr) {
            // Pointer-based emitter (flush SSA → global memory → reload)
            auto maybeStoreSSA = [&](int srcIdx) {
              auto ssaIt2 = ssaValues.find(srcIdx);
              if (ssaIt2 != ssaValues.end()) {
                auto argPtr = getSlotArgPtr(srcIdx);
                if (argPtr) {
                  auto ptrType = mlir::cast<mlir::triton::PointerType>(argPtr.getType());
                  auto elemType = ptrType.getPointeeType();
                  auto ptrTensorType = mlir::RankedTensorType::get({blockSize}, ptrType);
                  auto splatPtr = builder.create<mlir::triton::SplatOp>(loc, ptrTensorType, argPtr);
                  auto ptrs = builder.create<mlir::triton::AddPtrOp>(loc, ptrTensorType, splatPtr, offsets);
                  auto storeVal = castTo(builder, loc, ssaIt2->second, elemType);
                  builder.create<mlir::triton::StoreOp>(loc, ptrs, storeVal, mask,
                      mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL);
                }
              }
            };
            maybeStoreSSA(inputSrc);

            emitRoPESection(builder, loc, pid, blockSize,
                            inPtr, cosArgPtr, sinArgPtr, outPtr,
                            inShape, cosShapeVec, ropeType, nElements);

            DataType outDtype = outArr ? outArr->dataType() : resolveDtypeLocal(outSlot);
            auto loaded = loadBackFromBuffer(outSlot, outDtype);
            if (loaded) {
              loaded = emulateNativePrecision(loaded, si);
              for (int o = 0; o < slot.wiring.numOutputs; o++) ssaValues[slot.wiring.outputSlotIndices[o]] = loaded;
            }
          } else {
            // Neither SSA path nor pointer path is available — this is a genuine error.
            // Input must be either in SSA (for register path) or have a buffer pointer
            // (for pointer path). If neither, the segment structure is broken.
            std::string msg = "TritonIRBuilder: ROPE '" + slot.ident.opName + "' at slot " + std::to_string(si) +
                " — cannot emit: SSA input not available (canUseSSA=false) and "
                "buffer pointers missing (inPtr=" + std::to_string(inPtr ? 1 : 0) +
                " outPtr=" + std::to_string(outPtr ? 1 : 0) +
                " cosArgPtr=" + std::to_string(cosArgPtr ? 1 : 0) +
                " sinArgPtr=" + std::to_string(sinArgPtr ? 1 : 0) +
                "). Cannot compile.";
            THROW_EXCEPTION(msg.c_str());
          }
        } else {
          std::string msg = "TritonIRBuilder: ROPE '" + slot.ident.opName + "' at slot " + std::to_string(si) +
              " — missing required data: cosArgPtr=" + std::to_string(cosArgPtr ? 1 : 0) +
              " sinArgPtr=" + std::to_string(sinArgPtr ? 1 : 0) +
              " inShape.size=" + std::to_string(inShape.size()) +
              " cosShape.size=" + std::to_string(cosShapeVec.size()) +
              " nElements=" + std::to_string(nElements) +
              ". Cannot compile.";
          THROW_EXCEPTION(msg.c_str());
        }
      } else if (slot.wiring.numInputs <= 2 && slot.wiring.numOutputs >= 1) {
        // Position-offset ROPE (1 or 2 inputs): compute cos/sin inline from position.
        // Input[0] = data, Input[1] = position offset (scalar, may be ext input).
        // iArgs: [0]=ropeType, [2]=rotaryDims. tArgs: [0]=freqBase, [1]=freqScale.
        int inputSrc = slot.wiring.inputSourceIndices[0];
        int outSlot = slot.wiring.outputSlotIndices[0];

        // Extract parameters from iArgs/tArgs (same as native fused_rope op)
        int ropeType = (slot.args.numIArgs > 0 && slot.args.iArgs)
                       ? static_cast<int>(slot.args.iArgs[0]) : 0;
        float freqBase = (slot.args.numTArgs > 0 && slot.args.tArgs)
                         ? static_cast<float>(slot.args.tArgs[0]) : 10000.0f;
        float freqScale = (slot.args.numTArgs > 1 && slot.args.tArgs)
                          ? static_cast<float>(slot.args.tArgs[1]) : 1.0f;

        // Resolve input shape and dimensions
        NDArray* inArr = resolveArr(inputSrc);
        NDArray* outArr = resolveArr(outSlot);
        std::vector<LongType> inShape;
        if (inArr) {
          for (int d = 0; d < inArr->rankOf(); d++) inShape.push_back(inArr->sizeAt(d));
        } else {
          inShape = resolveShapeLocal(inputSrc);
        }

        int inputRank = static_cast<int>(inShape.size());
        int headDim = (inputRank > 0) ? static_cast<int>(inShape[inputRank - 1]) : 0;
        int numHeads = (inputRank >= 3) ? static_cast<int>(inShape[inputRank - 2]) : 1;
        int nElements = 0;
        if (outArr) {
          nElements = static_cast<int>(outArr->lengthOf());
        } else if (!inShape.empty()) {
          nElements = 1;
          for (auto d : inShape) nElements *= static_cast<int>(d);
        }

        // Get position pointer (input[1] if present, otherwise position comes from iArgs)
        mlir::Value posPtr;
        if (slot.wiring.numInputs >= 2) {
          int posSrc = slot.wiring.inputSourceIndices[1];
          posPtr = getSlotArgPtr(posSrc);
        }

        // Try SSA path: needs input in SSA and valid dimensions.
        // blockSize must fit within one seq position: blockSize <= numHeads * headDim.
        // For small inputs (single head), adjust effective block params for the emitter.
        auto ssaIt = ssaValues.find(inputSrc);
        int effectiveBlockSize = blockSize;
        int effectiveNumHeads = numHeads;
        // When blockSize exceeds data in one sequence position, clamp to actual data size.
        // The emitter will operate on nElements-worth of data with masking from the main loop.
        if (blockSize > numHeads * headDim && numHeads * headDim > 0) {
          effectiveBlockSize = numHeads * headDim;
          effectiveNumHeads = numHeads;
        }
        bool canUseSSA = ssaIt != ssaValues.end()
                         && headDim > 0 && (headDim % 2 == 0)
                         && (effectiveBlockSize % headDim == 0)
                         && (effectiveBlockSize <= effectiveNumHeads * headDim)
                         && posPtr;  // need position pointer

        if (canUseSSA) {
          // If the block is larger than actual data, we need to handle partial blocks.
          // For now, use effectiveBlockSize for the RoPE computation.
          auto inputVal = ssaIt->second;

          // If blockSize was clamped, slice the SSA value to effective size
          if (effectiveBlockSize < blockSize) {
            // Use tt.reshape to extract the first effectiveBlockSize elements,
            // apply RoPE, then pad back. For simplicity with small test inputs,
            // just use the pointer-based fallback path.
            auto inPtr = getSlotArgPtr(inputSrc);
            auto outPtr = getSlotArgPtr(outSlot);
            if (inPtr && outPtr && posPtr) {
              // Pointer-based RoPE: store SSA → global, compute RoPE via pointer path, reload
              auto maybeStoreSSA = [&](int srcIdx) {
                auto ssaIt2 = ssaValues.find(srcIdx);
                if (ssaIt2 != ssaValues.end()) {
                  auto argPtr = getSlotArgPtr(srcIdx);
                  if (argPtr) {
                    auto ptrType = mlir::cast<mlir::triton::PointerType>(argPtr.getType());
                    auto elemType = ptrType.getPointeeType();
                    auto ptrTensorType = mlir::RankedTensorType::get({blockSize}, ptrType);
                    auto splatPtr = builder.create<mlir::triton::SplatOp>(loc, ptrTensorType, argPtr);
                    auto ptrs = builder.create<mlir::triton::AddPtrOp>(loc, ptrTensorType, splatPtr, offsets);
                    auto storeVal = castTo(builder, loc, ssaIt2->second, elemType);
                    builder.create<mlir::triton::StoreOp>(loc, ptrs, storeVal, mask,
                        mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL);
                  }
                }
              };
              maybeStoreSSA(inputSrc);

              // Emit pointer-based position-offset RoPE
              emitRoPEPositionSection(builder, loc, pid, blockSize,
                                       inPtr, posPtr, outPtr,
                                       inShape, ropeType, freqBase, freqScale, nElements);

              DataType outDtype = outArr ? outArr->dataType() : resolveDtypeLocal(outSlot);
              auto loaded = loadBackFromBuffer(outSlot, outDtype);
              if (loaded) {
                loaded = emulateNativePrecision(loaded, si);
                for (int o = 0; o < slot.wiring.numOutputs; o++)
                  ssaValues[slot.wiring.outputSlotIndices[o]] = loaded;
              }
            } else {
              std::string msg = "TritonIRBuilder: ROPE '" + slot.ident.opName +
                  "' at slot " + std::to_string(si) +
                  " — position-offset ROPE pointer path failed: inPtr=" + std::to_string(inPtr ? 1 : 0) +
                  " outPtr=" + std::to_string(outPtr ? 1 : 0) +
                  " posPtr=" + std::to_string(posPtr ? 1 : 0);
              THROW_EXCEPTION(msg.c_str());
            }
          } else {
            auto result = emitRoPEPositionSSA(builder, loc, inputVal,
                                               posPtr, pid, effectiveBlockSize,
                                               headDim, effectiveNumHeads,
                                               freqBase, freqScale,
                                               ropeType, nElements);
            result = emulateNativePrecision(result, si);
            for (int o = 0; o < slot.wiring.numOutputs; o++)
              ssaValues[slot.wiring.outputSlotIndices[o]] = result;
          }
        } else {
          std::string msg = "TritonIRBuilder: ROPE '" + slot.ident.opName +
              "' at slot " + std::to_string(si) +
              " — position-offset ROPE emit failed: headDim=" + std::to_string(headDim) +
              " numHeads=" + std::to_string(numHeads) +
              " blockSize=" + std::to_string(blockSize) +
              " hasSSA=" + std::to_string(ssaIt != ssaValues.end() ? 1 : 0) +
              " hasPosPtr=" + std::to_string(posPtr ? 1 : 0);
          THROW_EXCEPTION(msg.c_str());
        }
      } else {
        DSP_DIAG_SLOT(FALLBACK, si, "TritonIRBuilder: ROPE '%s' at slot %d — insufficient inputs(%d)",
                  slot.ident.opName.c_str(), si, slot.wiring.numInputs);
      }

    } else if (cat == TritonOpCategory::DATA_MOVEMENT) {
      // ─── DATA MOVEMENT: dispatch to appropriate section emitter ───
      std::string opLower = slot.ident.opName;
      std::transform(opLower.begin(), opLower.end(), opLower.begin(), ::tolower);
      std::string opKey = normalizeOpToken(slot.ident.opName);

      if (slot.wiring.numInputs < 1 || slot.wiring.numOutputs < 1) {
        DSP_DIAG_SLOT(FALLBACK, si, "TritonIRBuilder: data movement '%s' at slot %d — insufficient inputs(%d)/outputs(%d)",
                  slot.ident.opName.c_str(), si, slot.wiring.numInputs, slot.wiring.numOutputs);
      } else if (opKey == "gather" || opKey == "gathernd") {
        // ─── GATHER ───
        int dataSrc = slot.wiring.inputSourceIndices[0];
        int idxSrc = (slot.wiring.numInputs >= 2) ? slot.wiring.inputSourceIndices[1] : dataSrc;
        int outSlot = slot.wiring.outputSlotIndices[0];

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
          if (slot.args.numIArgs > 0 && slot.args.iArgs) {
            axis = static_cast<int>(slot.args.iArgs[0]);
          }
          bool gatherNd = (opKey == "gathernd");

          emitGatherSection(builder, loc, pid, blockSize,
                            dataPtr, idxPtr, outPtr, axis,
                            dataShape, indicesShape, nElements, gatherNd);

          auto loaded = loadBackFromBuffer(outSlot, outArr->dataType());
          if (loaded) {
            for (int o = 0; o < slot.wiring.numOutputs; o++) ssaValues[slot.wiring.outputSlotIndices[o]] = loaded;
          }
        } else {
          std::string msg = "TritonIRBuilder: gather '" + slot.ident.opName + "' at slot " + std::to_string(si) +
              " — missing kernel arg ptrs/arrays. Cannot compile.";
          THROW_EXCEPTION(msg.c_str());
        }

      } else if (opLower == "concat") {
        // ─── CONCAT ───
        int outSlot = slot.wiring.outputSlotIndices[0];
        auto outPtr = getSlotArgPtr(outSlot);
        NDArray* outArr = resolveArr(outSlot);

        std::vector<mlir::Value> inPtrs;
        std::vector<std::vector<LongType>> inShapes;
        bool allValid = outPtr && outArr;

        for (int inp = 0; inp < slot.wiring.numInputs && allValid; inp++) {
          int src = slot.wiring.inputSourceIndices[inp];
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
          int axis = (slot.args.numIArgs > 0 && slot.args.iArgs) ? static_cast<int>(slot.args.iArgs[0]) : 0;

          emitConcatSection(builder, loc, pid, blockSize,
                            inPtrs, outPtr, axis, inShapes, nElements);

          auto loaded = loadBackFromBuffer(outSlot, outArr->dataType());
          if (loaded) {
            for (int o = 0; o < slot.wiring.numOutputs; o++) ssaValues[slot.wiring.outputSlotIndices[o]] = loaded;
          }
        } else {
          std::string msg = "TritonIRBuilder: concat '" + slot.ident.opName + "' at slot " + std::to_string(si) +
              " — missing kernel arg ptrs/arrays. Cannot compile.";
          THROW_EXCEPTION(msg.c_str());
        }

      } else if (opLower == "split" || opLower == "split_v") {
        // ─── SPLIT ───
        int dataSrc = slot.wiring.inputSourceIndices[0];
        auto dataPtr = getSlotArgPtr(dataSrc);
        NDArray* dataArr = resolveArr(dataSrc);

        std::vector<mlir::Value> outPtrs;
        bool allValid = dataPtr && dataArr;
        for (int o = 0; o < slot.wiring.numOutputs && allValid; o++) {
          int oSlot = slot.wiring.outputSlotIndices[o];
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
          int numSplits = slot.wiring.numOutputs;
          int nElements = static_cast<int>(dataArr->lengthOf());
          int rank = static_cast<int>(dataShape.size());

          bool isSplitV = (opLower.find("split_v") != std::string::npos ||
                           opLower.find("splitv") != std::string::npos);
          int splitAxis = 0;
          if (isSplitV) {
            // SplitV iArgs: [splitDim, numSplit]
            if (slot.args.numIArgs > 0 && slot.args.iArgs) splitAxis = static_cast<int>(slot.args.iArgs[0]);
          } else {
            // Split iArgs: [numSplit, splitDim] (most common) or [splitDim]
            if (slot.args.numIArgs > 1 && slot.args.iArgs) splitAxis = static_cast<int>(slot.args.iArgs[1]);
            else if (slot.args.numIArgs > 0 && slot.args.iArgs) splitAxis = static_cast<int>(slot.args.iArgs[0]);
          }
          if (splitAxis < 0) splitAxis += rank;
          if (splitAxis < 0 || splitAxis >= rank) splitAxis = 0;

          if (isSplitV && slot.wiring.numInputs >= 2) {
            // SplitV: variable chunk sizes from input[1]
            int sizesSrc = slot.wiring.inputSourceIndices[1];
            NDArray* sizesArr = resolveArr(sizesSrc);
            if (sizesArr && !dataShape.empty()) {
              int axisOffset = 0;
              for (int o2 = 0; o2 < slot.wiring.numOutputs && o2 < static_cast<int>(outPtrs.size()); o2++) {
                int chunkAxisSize = (o2 < static_cast<int>(sizesArr->lengthOf()))
                    ? static_cast<int>(sizesArr->e<int>(o2)) : 1;
                std::vector<int> begins(rank, 0);
                std::vector<int> ends;
                for (int d = 0; d < rank; d++) ends.push_back(static_cast<int>(dataShape[d]));
                begins[splitAxis] = axisOffset;
                ends[splitAxis] = axisOffset + chunkAxisSize;
                std::vector<int> strides(rank, 1);
                int chunkTotalElements = 1;
                for (int d = 0; d < rank; d++)
                  chunkTotalElements *= (d == splitAxis) ? chunkAxisSize : static_cast<int>(dataShape[d]);
                emitSliceSection(builder, loc, pid, blockSize, dataPtr, outPtrs[o2],
                                 begins, ends, strides, dataShape, chunkTotalElements);
                axisOffset += chunkAxisSize;
              }
            } else {
              emitSplitSection(builder, loc, pid, blockSize,
                               dataPtr, outPtrs, splitAxis, numSplits, dataShape, nElements);
            }
          } else {
            emitSplitSection(builder, loc, pid, blockSize,
                             dataPtr, outPtrs, splitAxis, numSplits, dataShape, nElements);
          }

          // Load back each output for downstream SSA
          for (int o = 0; o < slot.wiring.numOutputs; o++) {
            int oSlot = slot.wiring.outputSlotIndices[o];
            NDArray* oArr = resolveArr(oSlot);
            DataType dt = oArr ? oArr->dataType() : FLOAT32;
            auto loaded = loadBackFromBuffer(oSlot, dt);
            if (loaded) ssaValues[oSlot] = loaded;
          }
        } else {
          std::string msg = "TritonIRBuilder: split '" + slot.ident.opName + "' at slot " + std::to_string(si) +
              " — missing kernel arg ptrs. Cannot compile.";
          THROW_EXCEPTION(msg.c_str());
        }

      } else if (opLower == "tile") {
        // ─── TILE ───
        int dataSrc = slot.wiring.inputSourceIndices[0];
        int outSlot = slot.wiring.outputSlotIndices[0];
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
            for (int o = 0; o < slot.wiring.numOutputs; o++) ssaValues[slot.wiring.outputSlotIndices[o]] = loaded;
          }
        } else {
          std::string msg = "TritonIRBuilder: tile '" + slot.ident.opName + "' at slot " + std::to_string(si) +
              " — missing kernel arg ptrs. Cannot compile.";
          THROW_EXCEPTION(msg.c_str());
        }

      } else if (opLower == "strided_slice") {
        // ─── STRIDED SLICE ───
        int dataSrc = slot.wiring.inputSourceIndices[0];
        int outSlot = slot.wiring.outputSlotIndices[0];
        auto dataPtr = getSlotArgPtr(dataSrc);
        auto outPtr = getSlotArgPtr(outSlot);
        NDArray* dataArr = resolveArr(dataSrc);
        NDArray* outArr = resolveArr(outSlot);

        if (dataPtr && outPtr && dataArr && outArr) {
          std::vector<LongType> inputShape;
          for (int d = 0; d < dataArr->rankOf(); d++) inputShape.push_back(dataArr->sizeAt(d));
          std::vector<int> begins, ends, strides;
          resolveStridedSliceParams(slot, inputShape, resolveArr, begins, ends, strides);
          int nElements = static_cast<int>(outArr->lengthOf());

          emitSliceSection(builder, loc, pid, blockSize,
                           dataPtr, outPtr, begins, ends, strides,
                           inputShape, nElements);

          auto loaded = loadBackFromBuffer(outSlot, outArr->dataType());
          if (loaded) {
            for (int o = 0; o < slot.wiring.numOutputs; o++) ssaValues[slot.wiring.outputSlotIndices[o]] = loaded;
          }
        } else {
          std::string msg = "TritonIRBuilder: strided_slice '" + slot.ident.opName + "' at slot " + std::to_string(si) +
              " — missing kernel arg ptrs. Cannot compile.";
          THROW_EXCEPTION(msg.c_str());
        }

      } else if (opLower == "stack") {
        // ─── STACK: treat as concat (stack = unsqueeze + concat along new axis) ───
        int outSlot = slot.wiring.outputSlotIndices[0];
        auto outPtr = getSlotArgPtr(outSlot);
        NDArray* outArr = resolveArr(outSlot);

        std::vector<mlir::Value> inPtrs;
        std::vector<std::vector<LongType>> inShapes;
        bool allValid = outPtr && outArr;

        for (int inp = 0; inp < slot.wiring.numInputs && allValid; inp++) {
          int src = slot.wiring.inputSourceIndices[inp];
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
          int axis = (slot.args.numIArgs > 0 && slot.args.iArgs) ? static_cast<int>(slot.args.iArgs[0]) : 0;
          // Stack = unsqueeze + concat: insert dim=1 at axis for each input shape
          for (auto& shape : inShapes) {
            shape.insert(shape.begin() + axis, 1);
          }
          emitConcatSection(builder, loc, pid, blockSize,
                            inPtrs, outPtr, axis, inShapes, nElements);

          auto loaded = loadBackFromBuffer(outSlot, outArr->dataType());
          if (loaded) {
            for (int o = 0; o < slot.wiring.numOutputs; o++) ssaValues[slot.wiring.outputSlotIndices[o]] = loaded;
          }
        } else {
          std::string msg = "TritonIRBuilder: stack '" + slot.ident.opName + "' at slot " + std::to_string(si) +
              " — missing kernel arg ptrs. Cannot compile.";
          THROW_EXCEPTION(msg.c_str());
        }

      } else if (opLower == "scatter_nd" || opLower == "scatter_nd_update") {
        // ─── SCATTER_ND: copy data + scatter updates at indexed positions ───
        // scatter_nd needs 3 inputs: data, indices, updates
        // Output = copy of data with updates scattered at indexed positions
        if (slot.wiring.numInputs >= 3 && slot.wiring.numOutputs >= 1) {
          int dataSrc = slot.wiring.inputSourceIndices[0];
          int idxSrc = slot.wiring.inputSourceIndices[1];
          int updSrc = slot.wiring.inputSourceIndices[2];
          int outSlot = slot.wiring.outputSlotIndices[0];

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
            DataType outDtype = resolveDtypeLocal(outSlot);
            auto result = loadBackFromBuffer(outSlot, outDtype);
            if (result) {
              for (int o = 0; o < slot.wiring.numOutputs; o++) ssaValues[slot.wiring.outputSlotIndices[o]] = result;
            }
          } else {
            std::string msg = "TritonIRBuilder: scatter_nd '" + slot.ident.opName + "' at slot " + std::to_string(si) +
                " — missing kernel arg ptrs. Cannot compile.";
            THROW_EXCEPTION(msg.c_str());
          }
        } else if (slot.wiring.numInputs >= 1) {
          auto inputIt = ssaValues.find(slot.wiring.inputSourceIndices[0]);
          if (inputIt != ssaValues.end()) {
            for (int o = 0; o < slot.wiring.numOutputs; o++) ssaValues[slot.wiring.outputSlotIndices[o]] = inputIt->second;
          }
        }

      } else {
        // Unknown data movement op — fail compilation instead of producing garbage
        std::string msg = "TritonIRBuilder: unhandled data movement op '" + slot.ident.opName + "' at slot " +
            std::to_string(si) + ". No emitter available. Cannot compile.";
        THROW_EXCEPTION(msg.c_str());
      }

    } else if (cat == TritonOpCategory::CONSTANT_GENERATION) {
      // Constant generation ops (shape_of, create, set_scalar, ones_as, range):
      // These produce constant or computed values independent of input data.
      // In the 1D kernel, emit appropriate constant splats or ranges.
      DataType outDtype = FLOAT32;
      if (slot.wiring.numOutputs > 0) {
        int outIdx = slot.wiring.outputSlotIndices[0];
        outDtype = resolveDtypeLocal(outIdx);
      }
      auto elemType = getMLIRType(builder, outDtype);
      auto tensorType = mlir::RankedTensorType::get({blockSize}, elemType);

      std::string opLower = slot.ident.opName;
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
        if (slot.args.numTArgs > 0 && slot.args.tArgs) {
          fillVal = static_cast<float>(slot.args.tArgs[0]);
          foundVal = true;
        }
        if (!foundVal && slot.wiring.numOutputs > 0) {
          int outIdx = slot.wiring.outputSlotIndices[0];
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
        if (slot.args.numTArgs >= 1 && slot.args.tArgs) start = static_cast<float>(slot.args.tArgs[0]);
        if (slot.args.numTArgs >= 3 && slot.args.tArgs) step = static_cast<float>(slot.args.tArgs[2]);

        // Determine range output length from the output array's shape
        int rangeLen = blockSize;
        if (slot.wiring.numOutputs > 0) {
          int outIdx = slot.wiring.outputSlotIndices[0];
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
        if (slot.wiring.numOutputs > 0) {
          int outIdx = slot.wiring.outputSlotIndices[0];
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

      for (int o = 0; o < slot.wiring.numOutputs; o++) {
        ssaValues[slot.wiring.outputSlotIndices[o]] = opResult;
      }
    }
    next_slot: ;  // NOLINT — target for goto in backward attention dispatch
  }

  // 2d: Store outputs — tt.store for each output arg
  // Use per-output masks when outputs have different element counts to prevent
  // buffer overflows. In a fused kernel covering multiple independent chains,
  // different outputs can have different sizes (e.g., main hidden state [1,960]
  // vs RoPE frequencies [1,480]). Without per-output masks, the global n_elements
  // from the largest output would allow writes past smaller buffers.
  bool skippedAnyStore = false;
  int outputArgBase = static_cast<int>(inputArgs.size());
  for (int a = 0; a < static_cast<int>(outputArgs.size()); a++) {
    auto& arg = outputArgs[a];
    auto funcArg = getBufferArg(outputArgBase + a);

    // Skip outputs already stored inline by multi-row normalization
    if (normInlineStoredOutputs.count(arg.slotIndex)) {
      continue;
    }

    auto ssaIt = ssaValues.find(arg.slotIndex);
    if (ssaIt == ssaValues.end()) {
      DSP_DIAG_SLOT(FALLBACK, arg.slotIndex, "TritonIRBuilder: no SSA value for output slot %d — skipping store, invalidating module",
                arg.slotIndex);
      skippedAnyStore = true;
      continue;
    }

    auto elemType = getMLIRType(builder, arg.dtype);
    auto ptrType = mlir::triton::PointerType::get(elemType, 1);
    auto ptrTensorType = mlir::RankedTensorType::get({blockSize}, ptrType);

    auto splatPtr = builder.create<mlir::triton::SplatOp>(loc, ptrTensorType, funcArg);
    auto ptrs = builder.create<mlir::triton::AddPtrOp>(loc, ptrTensorType, splatPtr, offsets);

    // Cast SSA value to match output element type if needed
    mlir::Value storeVal = castTo(builder, loc, ssaIt->second, elemType);

    // Per-output mask: use the output's actual element count to prevent buffer overflow
    mlir::Value storeMask = mask;  // Default: global mask (offsets < n_elements)
    LongType outElements = 1;
    for (auto d : arg.shape) outElements *= d;
    // Only apply per-output mask when shape is known (non-empty) AND smaller than max.
    // Empty shape means shape was unknown at compile time — use global mask.
    if (!arg.shape.empty() && outElements > 0 && outElements < static_cast<LongType>(maxOutputElements)) {
      // This output is smaller than the largest — use a tighter mask
      auto outN = builder.create<mlir::arith::ConstantIntOp>(
          loc, static_cast<int>(std::min(outElements, static_cast<LongType>(2147483647))), 32);
      auto splatOutN = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, outN);
      storeMask = builder.create<mlir::arith::CmpIOp>(
          loc, mlir::arith::CmpIPredicate::slt, offsets, splatOutN);
    }

    builder.create<mlir::triton::StoreOp>(loc, ptrs, storeVal, storeMask,
                                           mlir::triton::CacheModifier::NONE,
                                           mlir::triton::EvictionPolicy::NORMAL);
  }

  // Return
  builder.create<mlir::triton::ReturnOp>(loc);

  result.mlirModule = new mlir::ModuleOp(moduleOp);
  result.mlirContext = mlirContext;  // Store for proper cleanup
  // If any output slot had no SSA value and its store was skipped, the compiled
  // kernel would leave that output buffer with stale data (NaN propagation).
  // Mark the module invalid so the caller falls back to slot-by-slot execution.
  if (skippedAnyStore) {
    DSP_DIAG(FALLBACK, "TritonIRBuilder: module invalid — %d output stores skipped (no SSA values)",
             (int)outputArgs.size());
    result.valid = false;
    return result;
  }
  result.valid = true;
  result.useIndirectArgs = useIndirectArgs;

  // Element-wise kernels MUST use dynamic grid: the grid size depends on n_elements
  // passed at launch time. A fixed grid of 1 block only processes BLOCK_SIZE elements,
  // leaving larger outputs partially computed (stale data from previous step).
  // Reductions/normalizations stay fixed at 1 block (set earlier at line ~3589)
  // because they use block-local bar.sync barriers.
  bool hasReductionOrNorm = false;
  for (auto cat : categories) {
    if (cat == TritonOpCategory::REDUCTION || cat == TritonOpCategory::NORMALIZATION) {
      hasReductionOrNorm = true;
      break;
    }
  }
  result.useDynamicGrid = !hasReductionOrNorm;
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
    DSP_DIAG(COMPILE, "TritonIRBuilder: built module '%s' with %d ops, %d input args, %d output args, "
              "BLOCK_SIZE=%d",
              result.kernelName.c_str(), (endSlot - startSlot + 1),
              static_cast<int>(inputArgs.size()), static_cast<int>(outputArgs.size()),
              blockSize);
    // Write TTIR to file per sub-kernel range
    {
      char fname[256];
      snprintf(fname, sizeof(fname), "/tmp/triton_ttir_%d_%d.mlir", startSlot, endSlot);
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
  // Include slots even if shapeCacheValid() is false — cleanup may have reset
  // the state but shapes could still be populated from a prior execution.
  std::unordered_map<int, const LongType*> cachedShapeInfoMap;
  for (int i = 0; i < totalSlots; i++) {
    if (!slots[i].shapeCache.cachedOutputShapes.empty()) {
      for (int o = 0; o < slots[i].wiring.numOutputs; o++) {
        int outIdx = slots[i].wiring.outputSlotIndices[o];
        if (outIdx >= 0 && o < static_cast<int>(slots[i].shapeCache.cachedOutputShapes.size()) &&
            slots[i].shapeCache.cachedOutputShapes[o] != nullptr) {
          cachedShapeInfoMap[outIdx] = slots[i].shapeCache.cachedOutputShapes[o];
        }
      }
    }
  }

  sd_debug("TritonIRBuilder::buildSectionedModule: cached shape info map has %d entries\n",
            static_cast<int>(cachedShapeInfoMap.size()));

  // Helper: resolve shape for a source index.
  // Priority 1: live outputSlots array from warmup execution
  // Priority 2: cached shape info (fallback when the live array was released)
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
    // Priority 1: live outputSlots
    if (srcIdx < totalOutputSlots && outputSlots && outputSlots[srcIdx]) {
      auto& arr = *outputSlots[srcIdx];
      std::vector<LongType> s(arr.rankOf());
      for (int d = 0; d < arr.rankOf(); d++) s[d] = arr.sizeAt(d);
      return s;
    }
    // Priority 2: cached shape info
    auto cit = cachedShapeInfoMap.find(srcIdx);
    if (cit != cachedShapeInfoMap.end() && cit->second) {
      LongType rank = shape::rank(cit->second);
      std::vector<LongType> s(rank);
      for (int d = 0; d < rank; d++) s[d] = shape::shapeOf(cit->second)[d];
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
    if (srcIdx < totalOutputSlots && outputSlots && outputSlots[srcIdx])
      return outputSlots[srcIdx]->dataType();
    auto cit = cachedShapeInfoMap.find(srcIdx);
    if (cit != cachedShapeInfoMap.end() && cit->second)
      return ArrayOptions::dataType(cit->second);
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
    for (int o = 0; o < slots[i].wiring.numOutputs; o++) {
      internalSlotOutputs.insert(slots[i].wiring.outputSlotIndices[o]);
    }
  }

  // Determine which outputs are cross-section intermediates:
  // produced in one section, consumed in a different section
  auto crossSectionIntermediates = dsp::identifyCrossSectionIntermediates(
      slots, startSlot, endSlot, sections);

  sd_debug("TritonIRBuilder::buildSectionedModule: %d cross-section intermediates\n",
            static_cast<int>(crossSectionIntermediates.size()));

  // Pre-compute which section boundaries truly require a grid-wide barrier.
  // Many sections share the same 1D pid mapping and can stream values block-local
  // without cooperative synchronization.
  auto producerSectionByOutput = dsp::buildProducerSectionMap(slots, sections, internalSlotOutputs);

  std::vector<LongType> sectionMaxOutputElements(sections.size(), 0);
  auto computeSectionMaxOutputElements = [&](const KernelSection& sec) -> LongType {
    LongType maxElements = 0;
    for (int si = sec.startSlot; si <= sec.endSlot; si++) {
      for (int o = 0; o < slots[si].wiring.numOutputs; o++) {
        int outIdx = slots[si].wiring.outputSlotIndices[o];
        auto outShape = resolveShape(outIdx);
        LongType elems = shapeLength(outShape);
        if (elems > maxElements) maxElements = elems;
      }
    }
    return maxElements;
  };

  for (size_t secIdx = 0; secIdx < sections.size(); secIdx++) {
    sectionMaxOutputElements[secIdx] = computeSectionMaxOutputElements(sections[secIdx]);
  }

  std::vector<uint8_t> sectionNeedsBarrier(sections.size(), 0);
  for (size_t secIdx = 1; secIdx < sections.size(); secIdx++) {
    bool needsBarrier = false;
    const auto& consumerSection = sections[secIdx];
    for (int si = consumerSection.startSlot; si <= consumerSection.endSlot && !needsBarrier; si++) {
      for (int inp = 0; inp < slots[si].wiring.numInputs; inp++) {
        int srcIdx = slots[si].wiring.inputSourceIndices[inp];
        if (srcIdx < 0 || !crossSectionIntermediates.count(srcIdx)) continue;

        auto producerIt = producerSectionByOutput.find(srcIdx);
        if (producerIt == producerSectionByOutput.end()) continue;

        int producerSectionIdx = producerIt->second;
        if (producerSectionIdx == static_cast<int>(secIdx)) continue;
        if (producerSectionIdx < 0 || producerSectionIdx >= static_cast<int>(sections.size())) continue;

        const auto& producerSection = sections[producerSectionIdx];
        if (getSectionTypeConfig(producerSection.type).needsGlobalBarrier ||
            getSectionTypeConfig(consumerSection.type).needsGlobalBarrier) {
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

    DSP_DIAG(COMPILE, "TritonIRBuilder::buildSectionedModule: cooperative launch disabled; "
              "using multi-phase launch with %d phases (%d barriers) for [%d-%d]",
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

  // ── Force-external inputs for ROPE ops (same as buildModule) ──
  std::unordered_set<int> forceExternalInputsSectioned;
  for (int i = startSlot; i <= endSlot; i++) {
    auto cat = getOpCategory(slots[i].ident.opName);
    if (cat == TritonOpCategory::ROPE && slots[i].wiring.numInputs >= 3) {
      int cosSrc = slots[i].wiring.inputSourceIndices[1];
      int sinSrc = slots[i].wiring.inputSourceIndices[2];
      if (internalSlotOutputs.count(cosSrc)) {
        forceExternalInputsSectioned.insert(cosSrc);
        DSP_DIAG(COMPILE, "TritonIRBuilder(sectioned): ROPE '%s' at slot %d: forcing internal cos src %d to kernel arg",
                  slots[i].ident.opName.c_str(), i, cosSrc);
      }
      if (internalSlotOutputs.count(sinSrc)) {
        forceExternalInputsSectioned.insert(sinSrc);
        DSP_DIAG(COMPILE, "TritonIRBuilder(sectioned): ROPE '%s' at slot %d: forcing internal sin src %d to kernel arg",
                  slots[i].ident.opName.c_str(), i, sinSrc);
      }
    }
  }

  // Input args: external inputs or outputs from slots BEFORE this segment
  std::vector<TritonKernelArg> inputArgs;
  std::unordered_set<int> seenInputs;
  for (int i = startSlot; i <= endSlot; i++) {
    for (int inp = 0; inp < slots[i].wiring.numInputs; inp++) {
      int srcIdx = slots[i].wiring.inputSourceIndices[inp];
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
          for (int d = 0; d < arr.rankOf(); d++) {
            arg.shape.push_back(arr.sizeAt(d));
            arg.strides.push_back(arr.strideAt(d));
          }
          inputArgs.push_back(arg);
        }
      } else if (!internalSlotOutputs.count(srcIdx) || forceExternalInputsSectioned.count(srcIdx)) {
        auto shape = resolveShape(srcIdx);
        auto dtype = resolveDtype(srcIdx);
        bool hasLiveArr = (srcIdx < totalOutputSlots && outputSlots && outputSlots[srcIdx]);
        bool hasShape = !shape.empty() || cachedShapeInfoMap.count(srcIdx);
        if (hasLiveArr || hasShape) {
          TritonKernelArg arg;
          arg.slotIndex = srcIdx;
          arg.outputIndex = 0;
          arg.isOutput = false;
          arg.dtype = dtype;
          arg.shape = shape;
          // Extract actual strides from live NDArray for non-contiguous view detection
          if (hasLiveArr) {
            auto& arr = *outputSlots[srcIdx];
            for (int d = 0; d < arr.rankOf(); d++) {
              arg.strides.push_back(arr.strideAt(d));
            }
          } else if (srcIdx >= 0 && srcIdx < totalSlots) {
            // Live array not available — compute view strides from op parameters
            auto viewStrides = computeViewStridesFromOp(
                slots, srcIdx, totalSlots, outputSlots, totalOutputSlots, cachedShapeInfoMap);
            if (!viewStrides.empty()) {
              arg.strides = viewStrides;
            }
          }
          inputArgs.push_back(arg);
        } else {
          // Cross-segment input has neither live array nor cached shape — fail compilation
          DSP_DIAG(COMPILE, "TritonIRBuilder::buildSectionedModule: MISSING cross-segment input slot %d "
                    "for segment [%d-%d] (hasLiveArr=%d, shape.empty=%d, cachedShapeInfoMap.count=%d)",
                    srcIdx, startSlot, endSlot, hasLiveArr ? 1 : 0, shape.empty() ? 1 : 0,
                    static_cast<int>(cachedShapeInfoMap.count(srcIdx)));
          for (int si2 = 0; si2 < totalSlots; si2++) {
            for (int o2 = 0; o2 < slots[si2].wiring.numOutputs; o2++) {
              if (slots[si2].wiring.outputSlotIndices[o2] == srcIdx) {
                DSP_DIAG(COMPILE, "  -> produced by op[%d] '%s' state=%d shapeCacheValid=%d "
                          "cachedOutputShapes.size=%d",
                          si2, slots[si2].ident.opName.c_str(), slots[si2].slotPhase.toLegacyCode(),
                          slots[si2].shapeCacheValid() ? 1 : 0,
                          static_cast<int>(slots[si2].shapeCache.cachedOutputShapes.size()));
              }
            }
          }
          return result;
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

  // Matmul output slots MUST be in the arg list even if internal to the segment.
  // emitPerElementMatmul stores results via getSlotArgPtr which requires an arg entry.
  // Without this, matmul outputs in fused sections get cPtr=NULL and compilation fails.
  for (int i = startSlot; i <= endSlot; i++) {
    const auto& opName = slots[i].ident.opName;
    if (opName == "matmul" || opName == "mmul" || opName == "batched_gemm") {
      for (int o = 0; o < slots[i].wiring.numOutputs; o++) {
        int outIdx = slots[i].wiring.outputSlotIndices[o];
        if (outIdx >= 0 && outIdx < totalSlots) {
          externalOutputs.insert(outIdx);
        }
      }
    }
  }

  // NOTE: K/V projection external output forcing removed — attention ops now run via
  // cuBLAS fallback (isFallbackSection) and handle their own present_key/present_value outputs.

  // Build set of input buffer addresses for aliasing detection
  auto inputBufferAddrsSectioned = dsp::buildInputBufferAddressSet(
      inputArgs, outputSlots, totalOutputSlots, externalInputs, numExternalInputs);

  std::vector<TritonKernelArg> outputArgs;
  {
    std::unordered_set<int> seenOutputSlots;
    int skippedAliased = 0;
    for (int i = startSlot; i <= endSlot; i++) {
      for (int o = 0; o < slots[i].wiring.numOutputs; o++) {
        int outIdx = slots[i].wiring.outputSlotIndices[o];
        if (outIdx < 0 || outIdx >= totalOutputSlots) continue;
        if (seenOutputSlots.count(outIdx)) continue;
        seenOutputSlots.insert(outIdx);
        if (!externalOutputs.count(outIdx)) continue;

        // Skip outputs whose GPU buffer aliases an input buffer
        if (dsp::isOutputAliasedWithInput(outIdx, outputSlots, totalOutputSlots, inputBufferAddrsSectioned)) {
          skippedAliased++;
          DSP_DIAG(COMPILE, "TritonIRBuilder::buildSectionedModule: skipping aliased output slot %d "
                   "in segment [%d-%d]", outIdx, startSlot, endSlot);
          continue;
        }

        TritonKernelArg arg;
        arg.slotIndex = outIdx;
        arg.outputIndex = o;
        arg.isOutput = true;
        if (outputSlots && outIdx < totalOutputSlots && outputSlots[outIdx]) {
          arg.dtype = outputSlots[outIdx]->dataType();
          auto& arr = *outputSlots[outIdx];
          for (int d = 0; d < arr.rankOf(); d++) {
            arg.shape.push_back(arr.sizeAt(d));
            arg.strides.push_back(arr.strideAt(d));
          }
        } else {
          // Fall back to cached shape info when live array is not available
          auto cit = cachedShapeInfoMap.find(outIdx);
          if (cit != cachedShapeInfoMap.end() && cit->second) {
            arg.dtype = ArrayOptions::dataType(cit->second);
            LongType rank = shape::rank(cit->second);
            for (int d = 0; d < rank; d++) arg.shape.push_back(shape::shapeOf(cit->second)[d]);
          } else {
            // No live array and no cached shape — resolve from producing op
            auto producerCat = getOpCategory(slots[i].ident.opName);
            if (producerCat == TritonOpCategory::CAST && slots[i].args.numIArgs > 0 && slots[i].args.iArgs) {
              arg.dtype = static_cast<DataType>(slots[i].args.iArgs[0]);
            } else {
              if (slots[i].wiring.numInputs > 0) {
                int inputSrc = slots[i].wiring.inputSourceIndices[0];
                arg.dtype = resolveDtype(inputSrc);
              }
            }
            if (arg.shape.empty() && slots[i].wiring.numInputs > 0) {
              int inputSrc = slots[i].wiring.inputSourceIndices[0];
              auto inputShape = resolveShape(inputSrc);
              if (!inputShape.empty()) {
                arg.shape = inputShape;
              }
            }
          }
        }
        outputArgs.push_back(arg);
      }
    }
    if (skippedAliased > 0) {
      DSP_DIAG(FUSION, "TritonIRBuilder::buildSectionedModule: eliminated %d aliased outputs, "
                "keeping %d external", skippedAliased, (int)outputArgs.size());
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
  // Force indirect args when CUDA graph capture is enabled (see buildModule comment for rationale)
  if (!useIndirectArgs && sd::Environment::getInstance().tritonGraphCapture()) {
    useIndirectArgs = true;
    DSP_DIAG(COMPILE, "TritonIRBuilder::buildSectionedModule: forcing INDIRECT arg passing for graph capture "
              "compatibility (%d buffer args)", totalBufferArgs);
  }

  sd_debug("TritonIRBuilder::buildSectionedModule: %d input args, %d output args, %d total buffer args%s\n",
            static_cast<int>(inputArgs.size()), static_cast<int>(outputArgs.size()),
            totalBufferArgs, useIndirectArgs ? " (INDIRECT)" : " (direct)");

  // ── Step 3: Create MLIR module and function ──
  // Use thread-safe factory (loadDialect touches global DialectRegistry)
  auto* mlirContext = createMlirContextWithDialects();

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
    categories.push_back(getOpCategory(slots[i].ident.opName));
    if (slots[i].wiring.numOutputs > 0) {
      int outIdx = slots[i].wiring.outputSlotIndices[0];
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
      for (int o = 0; o < slots[si].wiring.numOutputs; o++) {
        int outIdx = slots[si].wiring.outputSlotIndices[o];
        auto outShape = resolveShape(outIdx);
        LongType elems = shapeLength(outShape);
        if (elems > maxElements) maxElements = elems;
      }
    }
    if (maxElements <= 0) {
      // Fallback for shape-only/meta ops: derive from consumed inputs.
      for (int si = sec.startSlot; si <= sec.endSlot; si++) {
        for (int inp = 0; inp < slots[si].wiring.numInputs; inp++) {
          int srcIdx = slots[si].wiring.inputSourceIndices[inp];
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
        if (getOpCategory(slot.ident.opName) != TritonOpCategory::FUSED_ATTENTION ||
            slot.wiring.numInputs < 1) {
          continue;
        }
        // Detect DPA v2 (input order Q,V,K) vs standard (Q,K,V)
        std::string opLowerGrid = slot.ident.opName;
        std::transform(opLowerGrid.begin(), opLowerGrid.end(), opLowerGrid.begin(), ::tolower);
        bool isDpaV2Grid = (opLowerGrid.find("dot_product_attention") != std::string::npos);
        bool opUsesBSHDGrid = isDpaV2Grid;
        // For DPA v2: input[1]=V, input[2]=K; resolve K for seqK
        int kInputIdx = isDpaV2Grid ? 2 : 1;

        auto qShape = resolveShape(slot.wiring.inputSourceIndices[0]);
        if (qShape.size() >= 4) {
          batchSize = static_cast<int>(std::max<LongType>(1, qShape[0]));
          if (opUsesBSHDGrid) {
            // BSHD: [batch, seqQ, numHeads, headDim]
            seqQ = static_cast<int>(std::max<LongType>(1, qShape[1]));
            numHeads = static_cast<int>(std::max<LongType>(1, qShape[2]));
          } else {
            // BHSD: [batch, numHeads, seqQ, headDim]
            numHeads = static_cast<int>(std::max<LongType>(1, qShape[1]));
            seqQ = static_cast<int>(std::max<LongType>(1, qShape[2]));
          }
          headDim = static_cast<int>(std::max<LongType>(1, qShape[3]));
          if (slot.wiring.numInputs > kInputIdx) {
            auto kShape = resolveShape(slot.wiring.inputSourceIndices[kInputIdx]);
            if (kShape.size() >= 3) {
              // seqK is at dim[1] for BSHD, dim[2] for BHSD
              int seqKDim = opUsesBSHDGrid ? 1 : 2;
              if (static_cast<int>(kShape.size()) > seqKDim) {
                seqK = static_cast<int>(std::max<LongType>(1, kShape[seqKDim]));
              }
            }
          }
        } else if (qShape.size() == 3) {
          // 3D Q: [B, seqQ, H*D] — compound attention (onnx_multi_head_attention)
          batchSize = static_cast<int>(std::max<LongType>(1, qShape[0]));
          seqQ = static_cast<int>(std::max<LongType>(1, qShape[1]));
          int hidden = static_cast<int>(std::max<LongType>(1, qShape[2]));
          // numHeads from iArgs[0] (INT_ARG(0) in onnx_multi_head_attention)
          numHeads = (slot.args.numIArgs > 0 && slot.args.iArgs) ? static_cast<int>(slot.args.iArgs[0]) : 1;
          if (numHeads <= 0) numHeads = 1;
          headDim = hidden / numHeads;

          // Scan optional inputs for a real past_key tensor.
          bool hasPastKvGrid = false;
          for (int inp = 3; inp < slot.wiring.numInputs && !hasPastKvGrid; inp++) {
            auto candidateShape = resolveShape(slot.wiring.inputSourceIndices[inp]);
            if (candidateShape.size() == 4) {
              int candidateKvHeads = static_cast<int>(candidateShape[1]);
              int candidatePastSeq = static_cast<int>(candidateShape[2]);
              int candidateHeadDim = static_cast<int>(candidateShape[3]);
              if (candidateShape[0] > 0 &&
                  candidatePastSeq > 0 &&
                  candidateHeadDim == headDim &&
                  candidateKvHeads > 0 &&
                  candidateKvHeads <= numHeads &&
                  numHeads % candidateKvHeads == 0) {
                hasPastKvGrid = true;
                int seqKV = 1;
                if (slot.wiring.numInputs > kInputIdx) {
                  auto curKShape = resolveShape(slot.wiring.inputSourceIndices[kInputIdx]);
                  if (curKShape.size() == 3) {
                    seqKV = static_cast<int>(std::max<LongType>(1, curKShape[1]));
                  }
                }
                seqK = candidatePastSeq + seqKV;  // total sequence for attention
              }
            }
          }
          if (!hasPastKvGrid && slot.wiring.numInputs > kInputIdx) {
            auto kShape = resolveShape(slot.wiring.inputSourceIndices[kInputIdx]);
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

    if (sec.type == KernelSectionType::NORMALIZATION) {
      LongType numRows = 0;
      for (int si = sec.startSlot; si <= sec.endSlot && numRows <= 0; si++) {
        if (slots[si].wiring.numInputs < 1) continue;
        auto inputShape = resolveShape(slots[si].wiring.inputSourceIndices[0]);
        LongType totalElements = shapeLength(inputShape);
        LongType logicalRowLen = inputShape.empty() ? 0 : inputShape.back();
        if (totalElements > 0 && logicalRowLen > 0) {
          numRows = std::max<LongType>(1, (totalElements + logicalRowLen - 1) / logicalRowLen);
        }
      }
      if (numRows > static_cast<LongType>(2147483647)) {
        numRows = static_cast<LongType>(2147483647);
      }
      if (numRows > 0) {
        return static_cast<int>(numRows);
      }
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
      DSP_DIAG(COMPILE, "TritonIRBuilder::buildSectionedModule: tuned cooperative block size %d -> %d "
                "(targetBlocks=%d, resultingGrid=%d)",
                initialBlockSize, blockSize, coopTargetBlocks, maxSectionGrid);
    }
  }

  if (useMultiPhaseLaunch) {
    // Phase launch grids must reflect the final per-section grid requirements
    // after the section grid pass above, not the provisional values seen when
    // the phase list was first constructed.
    for (auto& phase : launchPhases) {
      int phaseGrid = 1;
      for (int s = phase.startSection; s <= phase.endSection; s++) {
        if (sections[s].gridRequirement > phaseGrid) {
          phaseGrid = sections[s].gridRequirement;
        }
      }
      phase.gridX = phaseGrid;
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
  std::unordered_set<int> customStoredOutputs;
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

    // ── Comprehensive per-section compile-time diagnostics ──
    // Logs EVERY section with ALL slots, inputs, outputs, shapes, args.
    // Zero cost when DSP_DIAG is disabled.
    if (DSP_DIAG_ENABLED(COMPILE)) {
      auto sectionTypeName = [](KernelSectionType t) -> const char* {
        switch (t) {
          case KernelSectionType::ELEMENTWISE: return "ELEMENTWISE";
          case KernelSectionType::IDENTITY: return "IDENTITY";
          case KernelSectionType::CONSTANT_GENERATION: return "CONSTANT_GENERATION";
          case KernelSectionType::REDUCTION: return "REDUCTION";
          case KernelSectionType::NORMALIZATION: return "NORMALIZATION";
          case KernelSectionType::MATMUL: return "MATMUL";
          case KernelSectionType::FUSED_ATTENTION: return "FUSED_ATTENTION";
          case KernelSectionType::GATHER: return "GATHER";
          case KernelSectionType::GATHER_ND: return "GATHER_ND";
          case KernelSectionType::CONCAT: return "CONCAT";
          case KernelSectionType::SPLIT: return "SPLIT";
          case KernelSectionType::SPLIT_V: return "SPLIT_V";
          case KernelSectionType::STACK: return "STACK";
          case KernelSectionType::TILE: return "TILE";
          case KernelSectionType::STRIDED_SLICE: return "STRIDED_SLICE";
          case KernelSectionType::SCATTER_ND: return "SCATTER_ND";
          case KernelSectionType::SCATTER_ND_UPDATE: return "SCATTER_ND_UPDATE";
          case KernelSectionType::SHAPE_MANIPULATION: return "SHAPE_MANIPULATION";
          case KernelSectionType::CONVOLUTION: return "CONVOLUTION";
          default: return "UNKNOWN";
        }
      };
      auto fmtShapeVec = [](const std::vector<LongType>& s) -> std::string {
        std::string r;
        for (size_t i = 0; i < s.size(); i++) { if (i) r += ","; r += std::to_string(s[i]); }
        return r.empty() ? "empty" : r;
      };

      // Section header
      DSP_DIAG(COMPILE, "SECTION[%d] type=%s slots=[%d-%d] grid=%d phase=%d",
                static_cast<int>(secIdx), sectionTypeName(sec.type),
                sec.startSlot, sec.endSlot, sec.gridRequirement,
                useMultiPhaseLaunch ? sectionPhase[secIdx] : -1);

      // Every slot in this section
      for (int si = sec.startSlot; si <= sec.endSlot; si++) {
        auto& slot = slots[si];
        DSP_DIAG(COMPILE, "  SLOT[%d] op='%s' inputs=%d outputs=%d iArgs=%d tArgs=%d bArgs=%d "
                  "identity=%d view=%d fused=%d zeroOut=%d",
                  si, slot.ident.opName.c_str(), slot.wiring.numInputs, slot.wiring.numOutputs,
                  slot.args.numIArgs, slot.args.numTArgs, slot.args.numBArgs,
                  slot.isIdentityOp() ? 1 : 0, slot.isViewCapableOp() ? 1 : 0,
                  slot.flags.inPlaceFused ? 1 : 0, slot.needsZeroedOutput() ? 1 : 0);

        // Every input for this slot
        for (int inp = 0; inp < slot.wiring.numInputs; inp++) {
          int srcIdx = slot.wiring.inputSourceIndices[inp];
          auto srcShape = resolveShape(srcIdx);
          const char* srcOp = "EXT";
          if (srcIdx >= 0 && srcIdx < totalSlots) srcOp = slots[srcIdx].ident.opName.c_str();
          bool exists = false;
          LongType len = -1;
          if (srcIdx >= 0 && srcIdx < totalOutputSlots && outputSlots && outputSlots[srcIdx]) {
            exists = true; len = outputSlots[srcIdx]->lengthOf();
          } else if (srcIdx < 0) {
            int ei = -(srcIdx + 1);
            if (ei < numExternalInputs && externalInputs && externalInputs[ei]) {
              exists = true; len = externalInputs[ei]->lengthOf();
            }
          }
          std::string cachedStr = "none";
          auto cit = cachedShapeInfoMap.find(srcIdx);
          if (cit != cachedShapeInfoMap.end() && cit->second) {
            LongType cRank = shape::rank(cit->second);
            cachedStr = "[";
            for (int d = 0; d < cRank; d++) {
              if (d) cachedStr += ",";
              cachedStr += std::to_string(shape::shapeOf(cit->second)[d]);
            }
            cachedStr += "]";
          }
          DSP_DIAG(COMPILE, "    input[%d] src=%d op='%s' shape=[%s] exists=%d len=%lld cached=%s",
                    inp, srcIdx, srcOp, fmtShapeVec(srcShape).c_str(),
                    exists, (long long)len, cachedStr.c_str());
        }

        // Every output for this slot
        for (int outp = 0; outp < slot.wiring.numOutputs; outp++) {
          int outIdx = slot.wiring.outputSlotIndices[outp];
          auto outShape = resolveShape(outIdx);
          bool outExists = (outIdx >= 0 && outIdx < totalOutputSlots && outputSlots && outputSlots[outIdx]);
          LongType outLen = outExists ? outputSlots[outIdx]->lengthOf() : -1;
          DSP_DIAG(COMPILE, "    output[%d] slot=%d shape=[%s] exists=%d len=%lld",
                    outp, outIdx, fmtShapeVec(outShape).c_str(),
                    outExists, (long long)outLen);
        }

        // iArgs
        if (slot.args.numIArgs > 0 && slot.args.iArgs) {
          std::string iStr;
          for (int a = 0; a < slot.args.numIArgs && a < 20; a++) {
            if (a) iStr += ",";
            iStr += std::to_string(slot.args.iArgs[a]);
          }
          if (slot.args.numIArgs > 20) iStr += "...";
          DSP_DIAG(COMPILE, "    iArgs=[%s] (%d total)", iStr.c_str(), slot.args.numIArgs);
        }

        // tArgs
        if (slot.args.numTArgs > 0 && slot.args.tArgs) {
          char tBuf[512] = {0};
          int toff = 0;
          for (int a = 0; a < slot.args.numTArgs && a < 10 && toff < 480; a++) {
            toff += snprintf(tBuf + toff, sizeof(tBuf) - toff, "%s%.6g", a > 0 ? "," : "", slot.args.tArgs[a]);
          }
          if (slot.args.numTArgs > 10) strncat(tBuf, "...", sizeof(tBuf) - strlen(tBuf) - 1);
          DSP_DIAG(COMPILE, "    tArgs=[%s] (%d total)", tBuf, slot.args.numTArgs);
        }

        // bArgs
        if (slot.args.numBArgs > 0 && slot.args.bArgs) {
          std::string bStr;
          for (int a = 0; a < slot.args.numBArgs && a < 10; a++) {
            if (a) bStr += ",";
            bStr += slot.args.bArgs[a] ? "true" : "false";
          }
          DSP_DIAG(COMPILE, "    bArgs=[%s] (%d total)", bStr.c_str(), slot.args.numBArgs);
        }
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

        // Load inputs that aren't already in SSA map, with broadcast indexing
        // Compute max output elements for this section (for broadcast detection)
        LongType secMaxOutputElements = 0;
        for (int si = sec.startSlot; si <= sec.endSlot; si++) {
          for (int o = 0; o < slots[si].wiring.numOutputs; o++) {
            int outIdx = slots[si].wiring.outputSlotIndices[o];
            auto outShape = resolveShape(outIdx);
            LongType oElems = 1;
            for (auto d : outShape) oElems *= d;
            if (oElems > secMaxOutputElements) secMaxOutputElements = oElems;
          }
        }
        // Fallback: if output shapes unavailable, use max input elements
        if (secMaxOutputElements <= 1) {
          for (int si = sec.startSlot; si <= sec.endSlot; si++) {
            for (int inp = 0; inp < slots[si].wiring.numInputs; inp++) {
              int srcIdx = slots[si].wiring.inputSourceIndices[inp];
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

        bool skipGenericPreload = (sec.type == KernelSectionType::NORMALIZATION);
        if (!skipGenericPreload) {
          for (int si = sec.startSlot; si <= sec.endSlot; si++) {
            for (int inp = 0; inp < slots[si].wiring.numInputs; inp++) {
              int srcIdx = slots[si].wiring.inputSourceIndices[inp];
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
        }

        // Emit ops in this section
        for (int si = sec.startSlot; si <= sec.endSlot; si++) {
          auto& slot = slots[si];
          auto cat = getOpCategory(slot.ident.opName);
          auto it = opTable.find(slot.ident.opName);
          if (it == opTable.end()) continue;
          const auto& mapping = it->second;

          if (cat == TritonOpCategory::BINARY_ELEMENTWISE) {
            if (slot.wiring.numInputs < 2) {
              // Unary pow fallback: pow with scalar exponent in tArgs
              std::string opLower2 = slot.ident.opName;
              std::transform(opLower2.begin(), opLower2.end(), opLower2.begin(), ::tolower);
              if (opLower2 == "pow" && slot.wiring.numInputs >= 1) {
                auto inputIt = ssaValues.find(slot.wiring.inputSourceIndices[0]);
                if (inputIt != ssaValues.end()) {
                  auto opResult = emitUnaryElementwise(builder, loc, mapping, slot, inputIt->second, blockSize);
                  for (int o = 0; o < slot.wiring.numOutputs; o++) ssaValues[slot.wiring.outputSlotIndices[o]] = opResult;
                }
              }
              continue;
            }
            auto lhsIt = ssaValues.find(slot.wiring.inputSourceIndices[0]);
            auto rhsIt = ssaValues.find(slot.wiring.inputSourceIndices[1]);
            if (lhsIt == ssaValues.end() || rhsIt == ssaValues.end()) continue;
            auto opResult = emitBinaryElementwise(builder, loc, mapping, slot, lhsIt->second, rhsIt->second);
            for (int o = 0; o < slot.wiring.numOutputs; o++) ssaValues[slot.wiring.outputSlotIndices[o]] = opResult;
          } else if (cat == TritonOpCategory::UNARY_ELEMENTWISE) {
            if (slot.wiring.numInputs < 1) continue;
            auto inputIt = ssaValues.find(slot.wiring.inputSourceIndices[0]);
            if (inputIt == ssaValues.end()) continue;
            mlir::Value opResult;
            // ClipByValue with tensor min/max: use loaded SSA inputs, not tArgs
            {
              std::string ol = slot.ident.opName;
              std::transform(ol.begin(), ol.end(), ol.begin(), ::tolower);
              if ((ol == "clipbyvalue" || ol == "clip_by_value" || ol == "clamp")
                  && slot.wiring.numInputs >= 3) {
                auto minIt = ssaValues.find(slot.wiring.inputSourceIndices[1]);
                auto maxIt = ssaValues.find(slot.wiring.inputSourceIndices[2]);
                if (minIt != ssaValues.end() && maxIt != ssaValues.end()) {
                  auto elTy = mlir::cast<mlir::RankedTensorType>(
                      inputIt->second.getType()).getElementType();
                  if (mlir::isa<mlir::FloatType>(elTy)) {
                    auto cl = builder.create<mlir::arith::MaximumFOp>(loc, inputIt->second, minIt->second);
                    opResult = builder.create<mlir::arith::MinimumFOp>(loc, cl, maxIt->second);
                  } else {
                    auto cl = builder.create<mlir::arith::MaxSIOp>(loc, inputIt->second, minIt->second);
                    opResult = builder.create<mlir::arith::MinSIOp>(loc, cl, maxIt->second);
                  }
                }
              }
            }
            if (!opResult)
              opResult = emitUnaryElementwise(builder, loc, mapping, slot, inputIt->second, blockSize);
            for (int o = 0; o < slot.wiring.numOutputs; o++) ssaValues[slot.wiring.outputSlotIndices[o]] = opResult;
          } else if (cat == TritonOpCategory::COMPARISON) {
            if (slot.wiring.numInputs < 1) continue;
            auto lhsIt = ssaValues.find(slot.wiring.inputSourceIndices[0]);
            if (lhsIt == ssaValues.end()) continue;
            mlir::Value opResult;
            if (slot.wiring.numInputs >= 2) {
              auto rhsIt = ssaValues.find(slot.wiring.inputSourceIndices[1]);
              if (rhsIt == ssaValues.end()) continue;
              opResult = emitComparisonOp(builder, loc, slot.ident.opName, lhsIt->second, rhsIt->second, blockSize);
            } else if (slot.args.numTArgs > 0 && slot.args.tArgs) {
              auto lhsTensorType = mlir::cast<mlir::RankedTensorType>(lhsIt->second.getType());
              auto lhsElemType = lhsTensorType.getElementType();
              double scalarVal = slot.args.tArgs[0];
              mlir::Value rhsSplat;
              if (mlir::isa<mlir::FloatType>(lhsElemType)) {
                rhsSplat = splatConstantF32(builder, loc, lhsTensorType, static_cast<float>(scalarVal));
              } else {
                rhsSplat = splatConstantI32(builder, loc, lhsTensorType, static_cast<int>(scalarVal));
              }
              opResult = emitComparisonOp(builder, loc, slot.ident.opName, lhsIt->second, rhsSplat, blockSize);
            } else {
              continue;
            }
            for (int o = 0; o < slot.wiring.numOutputs; o++) ssaValues[slot.wiring.outputSlotIndices[o]] = opResult;
          } else if (cat == TritonOpCategory::LOGICAL) {
            if (slot.wiring.numInputs < 1) continue;
            auto lhsIt = ssaValues.find(slot.wiring.inputSourceIndices[0]);
            if (lhsIt == ssaValues.end()) continue;
            mlir::Value rhsVal = lhsIt->second;
            if (slot.wiring.numInputs >= 2) {
              auto rhsIt = ssaValues.find(slot.wiring.inputSourceIndices[1]);
              if (rhsIt != ssaValues.end()) rhsVal = rhsIt->second;
            }
            auto opResult = emitLogicalOp(builder, loc, slot.ident.opName, lhsIt->second, rhsVal, blockSize);
            for (int o = 0; o < slot.wiring.numOutputs; o++) ssaValues[slot.wiring.outputSlotIndices[o]] = opResult;
          } else if (cat == TritonOpCategory::TERNARY) {
            if (slot.wiring.numInputs < 3) continue;
            auto condIt = ssaValues.find(slot.wiring.inputSourceIndices[0]);
            auto trueIt = ssaValues.find(slot.wiring.inputSourceIndices[1]);
            auto falseIt = ssaValues.find(slot.wiring.inputSourceIndices[2]);
            if (condIt == ssaValues.end() || trueIt == ssaValues.end() || falseIt == ssaValues.end()) continue;
            auto opResult = emitTernaryOp(builder, loc, condIt->second, trueIt->second, falseIt->second, blockSize);
            for (int o = 0; o < slot.wiring.numOutputs; o++) ssaValues[slot.wiring.outputSlotIndices[o]] = opResult;
          } else if (cat == TritonOpCategory::IDENTITY) {
            if (slot.wiring.numInputs < 1) continue;
            // assign(target, source): forward input[1]; identity(x): forward input[0]
            int identIdx = (slot.wiring.numInputs >= 2) ? 1 : 0;
            auto inputIt = ssaValues.find(slot.wiring.inputSourceIndices[identIdx]);
            if (inputIt == ssaValues.end()) continue;
            for (int o = 0; o < slot.wiring.numOutputs; o++) ssaValues[slot.wiring.outputSlotIndices[o]] = inputIt->second;
          } else if (cat == TritonOpCategory::CAST) {
            if (slot.wiring.numInputs < 1) continue;
            auto inputIt = ssaValues.find(slot.wiring.inputSourceIndices[0]);
            if (inputIt == ssaValues.end()) continue;
            // If cast elimination marked this slot as identity, forward the SSA
            // value without computing the cast. This keeps the Triton path
            // consistent with the slot-by-slot identity fast path.
            if (slot.isIdentityOp()) {
              for (int o = 0; o < slot.wiring.numOutputs; o++)
                ssaValues[slot.wiring.outputSlotIndices[o]] = inputIt->second;
            } else {
              DataType targetDtype = FLOAT32;
              if (slot.args.numDArgs > 0 && slot.args.dArgs) {
                targetDtype = slot.args.dArgs[0];
              } else if (slot.args.numIArgs > 0 && slot.args.iArgs) {
                // Cast ops store target dtype in iArgs[0]
                targetDtype = static_cast<DataType>(slot.args.iArgs[0]);
              } else if (slot.wiring.numOutputs > 0) {
                int outIdx = slot.wiring.outputSlotIndices[0];
                targetDtype = resolveDtype(outIdx);
              }
              auto targetElemType = getMLIRType(builder, targetDtype);
              auto opResult = castTo(builder, loc, inputIt->second, targetElemType);
              for (int o = 0; o < slot.wiring.numOutputs; o++) ssaValues[slot.wiring.outputSlotIndices[o]] = opResult;
            }
          } else if (cat == TritonOpCategory::REDUCTION) {
            // Segmented reduction: same approach as buildModule (lines 4159-4449).
            // Cannot use emitReductionOp/tt.reduce because sectioned module uses flat 1D
            // tensors — tt.reduce would reduce the entire tensor to a scalar, but we need
            // partial reduction along a specific axis of the original multi-dimensional shape.
            if (slot.wiring.numInputs < 1) continue;
            int inputSrc = slot.wiring.inputSourceIndices[0];

            // Get reduction axis from iArgs
            int reductionAxis = (slot.args.numIArgs > 0 && slot.args.iArgs) ? static_cast<int>(slot.args.iArgs[0]) : -1;

            // Resolve original input shape (multi-dimensional)
            auto inputShape = resolveShape(inputSrc);
            int inputRank = static_cast<int>(inputShape.size());
            if (inputRank == 0) {
              DSP_DIAG_SLOT(SHAPE, si, "TritonIRBuilder::buildSectionedModule: reduction op '%s' at slot %d has no input shape",
                        slot.ident.opName.c_str(), si);
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
            std::vector<int> redOutShape;
            for (int d = 0; d < inputRank; d++)
              if (d != reductionAxis) redOutShape.push_back(static_cast<int>(inputShape[d]));
            int outRank = static_cast<int>(redOutShape.size());
            if (outRank == 0) { redOutShape.push_back(1); outRank = 1; }

            // Compute output strides (row-major)
            std::vector<int> redOutStrides(outRank, 1);
            for (int d = outRank - 2; d >= 0; d--)
              redOutStrides[d] = redOutStrides[d + 1] * redOutShape[d + 1];

            // Find input buffer arg
            auto inputArgIt = slotToArgIdx.find(inputSrc);
            if (inputArgIt == slotToArgIdx.end()) {
              DSP_DIAG_SLOT(FALLBACK, si, "TritonIRBuilder::buildSectionedModule: reduction input slot %d not in kernel args", inputSrc);
              continue;
            }

            // If input is from SSA (internal intermediate), store it to buffer first
            if (ssaValues.count(inputSrc)) {
              auto ssaVal = ssaValues[inputSrc];
              int midArgIdx = inputArgIt->second;
              auto midFuncArg = getBufferArg(midArgIdx);
              auto midPtrType = mlir::cast<mlir::triton::PointerType>(midFuncArg.getType());
              auto midElemType = midPtrType.getPointeeType();
              auto midPtrTensorType = mlir::RankedTensorType::get({blockSize}, midPtrType);
              auto midSplatPtr = builder.create<mlir::triton::SplatOp>(loc, midPtrTensorType, midFuncArg);
              auto midPtrs = builder.create<mlir::triton::AddPtrOp>(loc, midPtrTensorType, midSplatPtr, offsets);
              mlir::Value midStoreVal = castTo(builder, loc, ssaVal, midElemType);
              builder.create<mlir::triton::StoreOp>(loc, midPtrs, midStoreVal, mask,
                  mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL);
              // Memory fence + block barrier
              {
                auto dummyTensorType = mlir::RankedTensorType::get({blockSize}, builder.getI32Type());
                auto dummyZero = builder.create<mlir::arith::ConstantIntOp>(loc, 0, 32);
                auto dummyTensor = builder.create<mlir::triton::SplatOp>(loc, dummyTensorType, dummyZero);
                builder.create<mlir::triton::ElementwiseInlineAsmOp>(
                    loc, mlir::TypeRange{dummyTensorType},
                    "membar.gl; bar.sync 0; mov.b32 $0, $1;",
                    "=r,r", /*isPure=*/false, /*pack=*/1, mlir::ValueRange{dummyTensor});
              }
            }

            auto inputPtrArg = getBufferArg(inputArgIt->second);
            auto ptrType = mlir::cast<mlir::triton::PointerType>(inputPtrArg.getType());
            auto elemType = ptrType.getPointeeType();
            auto f32Type = builder.getF32Type();
            auto f32TensorType = mlir::RankedTensorType::get({blockSize}, f32Type);
            auto ptrTensorType = mlir::RankedTensorType::get({blockSize}, ptrType);

            // Determine reduction op type
            std::string opLower = slot.ident.opName;
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

            // Kahan compensation variable for sum/mean — reduces accumulation error
            // from O(n*eps) to O(eps), making result independent of accumulation order.
            // This is critical because native CUDA uses tree reduction (different order).
            bool useKahan = !isMax && !isMin && !isProd;  // sum and mean
            mlir::Value kahanComp;
            if (useKahan) {
              kahanComp = splatConstantF32(builder, loc, f32TensorType, 0.0f);
            }

            // Segmented reduction: for each output offset, accumulate over reduction axis
            for (int k = 0; k < reductionSize; k++) {
              // Compute input flat offset: unravel output coords, insert k at reduction axis, ravel
              mlir::Value inputOffset = splatConstantI32(builder, loc, i32TensorType, 0);
              mlir::Value rem = offsets;
              int oDimIdx = 0;
              for (int d = 0; d < inputRank; d++) {
                if (d == reductionAxis) {
                  auto contrib = splatConstantI32(builder, loc, i32TensorType, k * inStrides[d]);
                  inputOffset = builder.create<mlir::arith::AddIOp>(loc, inputOffset, contrib);
                } else {
                  auto oStrideConst = splatConstantI32(builder, loc, i32TensorType, redOutStrides[oDimIdx]);
                  auto coord = builder.create<mlir::arith::DivSIOp>(loc, rem, oStrideConst);
                  if (oDimIdx < outRank - 1)
                    rem = builder.create<mlir::arith::RemSIOp>(loc, rem, oStrideConst);
                  auto inStrideConst = splatConstantI32(builder, loc, i32TensorType, inStrides[d]);
                  auto contrib = builder.create<mlir::arith::MulIOp>(loc, coord, inStrideConst);
                  inputOffset = builder.create<mlir::arith::AddIOp>(loc, inputOffset, contrib);
                  oDimIdx++;
                }
              }

              // Load input at computed offsets
              auto splatPtr = builder.create<mlir::triton::SplatOp>(loc, ptrTensorType, inputPtrArg);
              auto ptrs = builder.create<mlir::triton::AddPtrOp>(loc, ptrTensorType, splatPtr, inputOffset);
              auto loaded = builder.create<mlir::triton::LoadOp>(loc,
                  ptrs.getResult(), mask.getResult(), mlir::Value(),
                  mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL, false);
              mlir::Value val = castTo(builder, loc, loaded, f32Type);

              // Combine
              if (isMax)
                acc = builder.create<mlir::arith::MaximumFOp>(loc, acc, val);
              else if (isMin)
                acc = builder.create<mlir::arith::MinimumFOp>(loc, acc, val);
              else if (isProd)
                acc = builder.create<mlir::arith::MulFOp>(loc, acc, val);
              else if (useKahan) {
                // Kahan compensated summation: y = val - comp; t = acc + y; comp = (t - acc) - y; acc = t
                auto y = builder.create<mlir::arith::SubFOp>(loc, val, kahanComp);
                auto t = builder.create<mlir::arith::AddFOp>(loc, acc, y);
                auto tMinusAcc = builder.create<mlir::arith::SubFOp>(loc, t, acc);
                kahanComp = builder.create<mlir::arith::SubFOp>(loc, tMinusAcc, y);
                acc = t;
              }
            }

            // Post-processing: mean divides by reduction size
            if (isMean && reductionSize > 0) {
              auto countSplat = splatConstantF32(builder, loc, f32TensorType,
                  static_cast<float>(reductionSize));
              acc = builder.create<mlir::arith::DivFOp>(loc, acc, countSplat);
            }

            // Cast to output element type
            auto outSlotIdx = slot.wiring.outputSlotIndices[0];
            auto outDtype = resolveDtype(outSlotIdx);
            auto outElemType = getMLIRType(builder, outDtype);
            mlir::Value opResult = castTo(builder, loc, acc, outElemType);
            if (!mlir::isa<mlir::RankedTensorType>(opResult.getType())) {
              auto splatTy = mlir::RankedTensorType::get({blockSize}, opResult.getType());
              opResult = builder.create<mlir::triton::SplatOp>(loc, splatTy, opResult);
            }
            for (int o = 0; o < slot.wiring.numOutputs; o++) ssaValues[slot.wiring.outputSlotIndices[o]] = opResult;
          } else if (cat == TritonOpCategory::NORMALIZATION) {
            // Normalization uses one program per logical row so RMS/softmax/layer-norm
            // operate over the last dimension instead of flattening the whole tensor.
            if (slot.wiring.numInputs < 1) continue;
            int inputSrc = slot.wiring.inputSourceIndices[0];
            auto inputShape = resolveShape(inputSrc);
            if (inputShape.empty()) continue;

            int64_t actualNumElements = shapeLength(inputShape);
            int64_t logicalRowLen = inputShape.back();
            if (actualNumElements <= 0 || logicalRowLen <= 0) continue;

            int64_t numRows = std::max<int64_t>(1, (actualNumElements + logicalRowLen - 1) / logicalRowLen);
            int64_t paddedRowLen = 1;
            while (paddedRowLen < logicalRowLen) paddedRowLen <<= 1;

            auto rowIdxType = builder.getI32Type();
            auto rowRangeType = mlir::RankedTensorType::get({paddedRowLen}, rowIdxType);
            auto rowMaskType = mlir::RankedTensorType::get({paddedRowLen}, builder.getI1Type());
            auto rowBase = builder.create<mlir::arith::MulIOp>(
                loc, pid, builder.create<mlir::arith::ConstantIntOp>(loc, logicalRowLen, 32));

            auto loadRowFromBuffer = [&](mlir::Value bufferArg,
                                         mlir::Type elemType,
                                         mlir::Value baseOffset,
                                         int64_t logicalLen) -> mlir::Value {
              int64_t paddedLen = 1;
              while (paddedLen < logicalLen) paddedLen <<= 1;

              auto ptrType = mlir::cast<mlir::triton::PointerType>(bufferArg.getType());
              auto idxTensorType = mlir::RankedTensorType::get({paddedLen}, rowIdxType);
              auto ptrTensorType = mlir::RankedTensorType::get({paddedLen}, ptrType);
              auto dataTensorType = mlir::RankedTensorType::get({paddedLen}, elemType);

              auto range = builder.create<mlir::triton::MakeRangeOp>(loc, idxTensorType, 0, paddedLen);
              auto baseSplat = builder.create<mlir::triton::SplatOp>(loc, idxTensorType, baseOffset);
              auto vecOffsets = builder.create<mlir::arith::AddIOp>(loc, baseSplat, range);

              auto lenConst = builder.create<mlir::arith::ConstantIntOp>(loc, logicalLen, 32);
              auto lenSplat = builder.create<mlir::triton::SplatOp>(loc, idxTensorType, lenConst);
              auto localMask = builder.create<mlir::arith::CmpIOp>(
                  loc, mlir::arith::CmpIPredicate::slt, range, lenSplat);

              auto splatPtr = builder.create<mlir::triton::SplatOp>(loc, ptrTensorType, bufferArg);
              auto ptrs = builder.create<mlir::triton::AddPtrOp>(loc, ptrTensorType, splatPtr, vecOffsets);
              auto zero = builder.create<mlir::arith::ConstantOp>(
                  loc, builder.getFloatAttr(elemType, 0.0));
              auto zeroTensor = builder.create<mlir::triton::SplatOp>(loc, dataTensorType, zero);
              return builder.create<mlir::triton::LoadOp>(
                  loc, ptrs.getResult(), localMask.getResult(), zeroTensor.getResult(),
                  mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL, false);
            };

            auto loadNormInput = [&](int src, bool rowWise) -> mlir::Value {
              auto ssaIt = ssaValues.find(src);
              if (ssaIt != ssaValues.end()) {
                return ssaIt->second;
              }

              auto argIt = slotToArgIdx.find(src);
              if (argIt == slotToArgIdx.end()) return mlir::Value();
              auto bufferArg = getBufferArg(argIt->second);
              if (!bufferArg) return mlir::Value();

              DataType dtype = resolveDtype(src);
              auto elemType = getMLIRType(builder, dtype);
              auto sideShape = resolveShape(src);
              if (sideShape.empty()) return mlir::Value();

              int64_t logicalLen = rowWise ? logicalRowLen : shapeLength(sideShape);
              if (logicalLen <= 0) return mlir::Value();

              mlir::Value baseOffset = rowBase;
              if (!rowWise) {
                baseOffset = builder.create<mlir::arith::ConstantIntOp>(loc, 0, 32).getResult();
              }
              return loadRowFromBuffer(bufferArg, elemType, baseOffset, logicalLen);
            };

            auto numRowsConst = builder.create<mlir::arith::ConstantIntOp>(loc, numRows, 32);
            auto rowActive = builder.create<mlir::arith::CmpIOp>(
                loc, mlir::arith::CmpIPredicate::slt, pid, numRowsConst);
            auto rowIf = builder.create<mlir::scf::IfOp>(loc, rowActive, /*withElseRegion=*/false);
            builder.setInsertionPointToStart(&rowIf.getThenRegion().front());

            mlir::Value inputValue = loadNormInput(inputSrc, true);
            if (!inputValue) {
              builder.create<mlir::scf::YieldOp>(loc);
              builder.setInsertionPointAfter(rowIf);
              continue;
            }

            auto inputTensorType = mlir::cast<mlir::RankedTensorType>(inputValue.getType());
            int axis = inputTensorType.getRank() - 1;
            if (axis < 0) axis = 0;
            auto outSlotIdx = slot.wiring.outputSlotIndices[0];

            std::string normKey = normalizeOpToken(slot.ident.opName);

            // ── Normalization backward ops ──────────────────────────────────────────────
            // rms_norm_bp, layer_norm_bp, fused_layer_norm_bp produce multiple outputs
            // (dx + dgamma [+ dbeta]).  Dispatch to emitNormalizationBackwardSection
            // which writes all outputs directly via tt.store / tt.atomic_add.
            {
              // normalizeOpToken strips underscores: rms_norm_bp -> rmsnormbp
              bool isNormBp = (normKey == "rmsnormbp"           ||
                               normKey == "layernormbp"          ||
                               normKey == "layernormalizationbp" ||
                               normKey == "fusedlayernormbp");
              if (isNormBp) {
                bool isRmsBp = (normKey == "rmsnormbp");

                // Load inputs using the existing loadNormInput helper:
                //   rms_norm_bp:         x=input[0] (already in inputValue), dy=input[1]
                //   layer/fused_norm_bp: x=input[0] (already in inputValue),
                //                        gamma=input[1], dy=input[2]
                mlir::Value bpX     = inputValue;
                mlir::Value bpDy    = loadNormInput(
                    slot.wiring.inputSourceIndices[isRmsBp ? 1 : 2], /*rowWise=*/true);
                mlir::Value bpGamma = isRmsBp ? mlir::Value()
                                              : loadNormInput(
                                                    slot.wiring.inputSourceIndices[1], /*rowWise=*/false);

                if (bpX && bpDy) {
                  // Output pointers: dx=output[0], dgamma=output[1], dbeta=output[2]
                  auto getSecOutPtr = [&](int outPos) -> mlir::Value {
                    if (outPos >= slot.wiring.numOutputs) return mlir::Value();
                    auto it = slotToArgIdx.find(slot.wiring.outputSlotIndices[outPos]);
                    if (it == slotToArgIdx.end()) return mlir::Value();
                    return getBufferArg(it->second);
                  };

                  float bpEps = (slot.args.numTArgs > 0 && slot.args.tArgs)
                                ? static_cast<float>(slot.args.tArgs[0]) : 1e-5f;

                  bool bpOk = emitNormalizationBackwardSection(
                      builder, loc, slot.ident.opName,
                      bpDy, bpX, bpGamma,
                      /*mean=*/mlir::Value(), /*invStd=*/mlir::Value(),
                      bpEps, logicalRowLen, rowBase,
                      getSecOutPtr(0), getSecOutPtr(1), getSecOutPtr(2));

                  if (bpOk) {
                    // Mark all outputs as custom-stored
                    for (int o = 0; o < slot.wiring.numOutputs; o++) {
                      int outS = slot.wiring.outputSlotIndices[o];
                      customStoredOutputs.insert(outS);
                      ssaValues[outS] = inputValue;  // placeholder for downstream SSA
                    }
                    builder.create<mlir::scf::YieldOp>(loc);
                    builder.setInsertionPointAfter(rowIf);
                    continue;  // Skip forward emitNormalizationOp below
                  } else {
                    DSP_DIAG_SLOT(FALLBACK, si, "buildSectionedModule: emitNormalizationBackwardSection failed for '%s'",
                                  slot.ident.opName.c_str());
                  }
                } else {
                  DSP_DIAG_SLOT(FALLBACK, si, "buildSectionedModule: backward norm '%s' at slot %d: missing dy or x",
                                slot.ident.opName.c_str(), si);
                }
                // Fall through to forward path on failure — will throw from emitNormalizationOp
              }
            }

            auto getNormInput = [&](int inputPos) -> mlir::Value {
              if (inputPos >= slot.wiring.numInputs) return mlir::Value();
              int src = slot.wiring.inputSourceIndices[inputPos];
              return loadNormInput(src, false);
            };

            mlir::Value scaleVal, biasVal, meanVal, varianceVal;
            mlir::Value hiddenValue;  // pre-norm hidden for skip_rms_norm output[1]
            if (normKey == "batchnorm") {
              meanVal = getNormInput(1);
              varianceVal = getNormInput(2);
              scaleVal = getNormInput(3);
              biasVal = getNormInput(4);
            } else if (normKey == "skiprmsnorm") {
              // skip_rms_norm: input[0]=x, input[1]=skip, input[2]=gamma, input[3]=bias(optional)
              // Load skip with rowWise=true (same shape as input)
              auto skipVal = loadNormInput(slot.wiring.inputSourceIndices[1], /*rowWise=*/true);
              if (skipVal) {
                // Promote both operands to a common float type before adding.
                // arith.addf requires floating-point operands.
                auto commonFTy = mlir::isa<mlir::FloatType>(getElementType(inputValue))
                                     ? getElementType(inputValue)
                                     : builder.getF32Type();
                inputValue = castTo(builder, loc, inputValue, commonFTy);
                skipVal    = castTo(builder, loc, skipVal, commonFTy);
                inputValue = builder.create<mlir::arith::AddFOp>(loc, inputValue, skipVal);
              }
              hiddenValue = inputValue;  // save pre-norm hidden for output[1]
              // gamma is input[2], bias is input[3]
              scaleVal = loadNormInput(slot.wiring.inputSourceIndices[2], /*rowWise=*/false);
              biasVal = (slot.wiring.numInputs > 3)
                      ? loadNormInput(slot.wiring.inputSourceIndices[3], /*rowWise=*/false)
                      : mlir::Value();
            } else {
              scaleVal = getNormInput(1);
              biasVal = getNormInput(2);
            }

            float epsilon2 = (slot.args.numTArgs > 0 && slot.args.tArgs) ? static_cast<float>(slot.args.tArgs[0]) : 1e-5f;
            auto opResult = emitNormalizationOp(builder, loc, slot.ident.opName, inputValue, axis,
                                                inputTensorType, scaleVal, biasVal,
                                                meanVal, varianceVal, epsilon2, logicalRowLen);
            if (!mlir::isa<mlir::RankedTensorType>(opResult.getType())) {
              auto splatTensorType = mlir::RankedTensorType::get({paddedRowLen}, opResult.getType());
              opResult = builder.create<mlir::triton::SplatOp>(loc, splatTensorType, opResult);
            }

            // Store all outputs: output[0]=normed, output[1]=hidden for skip_rms_norm
            for (int o = 0; o < slot.wiring.numOutputs; o++) {
              auto curOutSlot = slot.wiring.outputSlotIndices[o];
              // skip_rms_norm: output[1] is the pre-norm hidden (input+skip)
              mlir::Value outVal = (o == 1 && hiddenValue) ? hiddenValue : opResult;
              auto outArgIt = slotToArgIdx.find(curOutSlot);
              if (outArgIt != slotToArgIdx.end()) {
                auto outFuncArg = getBufferArg(outArgIt->second);
                auto outPtrType = mlir::cast<mlir::triton::PointerType>(outFuncArg.getType());
                auto outElemType = outPtrType.getPointeeType();
                auto outPtrTensorType = mlir::RankedTensorType::get({paddedRowLen}, outPtrType);
                auto rowRange = builder.create<mlir::triton::MakeRangeOp>(loc, rowRangeType, 0, paddedRowLen);
                auto rowBaseSplat = builder.create<mlir::triton::SplatOp>(loc, rowRangeType, rowBase);
                auto rowOffsets = builder.create<mlir::arith::AddIOp>(loc, rowBaseSplat, rowRange);
                auto rowLenConst = builder.create<mlir::arith::ConstantIntOp>(loc, logicalRowLen, 32);
                auto rowLenSplat = builder.create<mlir::triton::SplatOp>(loc, rowRangeType, rowLenConst);
                auto rowMask = builder.create<mlir::arith::CmpIOp>(
                    loc, mlir::arith::CmpIPredicate::slt, rowRange, rowLenSplat);
                auto splatPtr = builder.create<mlir::triton::SplatOp>(loc, outPtrTensorType, outFuncArg);
                auto ptrs = builder.create<mlir::triton::AddPtrOp>(loc, outPtrTensorType, splatPtr, rowOffsets);
                mlir::Value storeVal = castTo(builder, loc, outVal, outElemType);
                builder.create<mlir::triton::StoreOp>(loc, ptrs, storeVal, rowMask,
                    mlir::triton::CacheModifier::NONE,
                    mlir::triton::EvictionPolicy::NORMAL);
                customStoredOutputs.insert(curOutSlot);
              }
              ssaValues[curOutSlot] = outVal;
            }

            // ── Normalization epilogue: emit absorbed elementwise ops ──
            // These ops (residual add, dropout, cast, etc.) operate on the
            // normalization output which is now in the SSA value map.
            // They run per-row within the same scf.if block, so data stays
            // in registers (no global memory round-trip for the intermediate).
            if (sec.normEpilogueCount > 0 && sec.normEpilogueStartSlot >= 0) {
              DSP_DIAG(COMPILE, "buildSectionedModule: emitting %d norm epilogue ops [%d-%d]",
                        sec.normEpilogueCount, sec.normEpilogueStartSlot, sec.normEpilogueEndSlot);
              for (int epi = sec.normEpilogueStartSlot; epi <= sec.normEpilogueEndSlot; epi++) {
                auto& epiSlot = slots[epi];
                auto epiCat = getOpCategory(epiSlot.ident.opName);
                auto epiIt = opTable.find(epiSlot.ident.opName);
                if (epiIt == opTable.end()) continue;
                const auto& epiMapping = epiIt->second;

                // Ensure all inputs are in SSA value map (load per-row from global memory if needed)
                for (int inp = 0; inp < epiSlot.wiring.numInputs; inp++) {
                  int srcIdx = epiSlot.wiring.inputSourceIndices[inp];
                  if (!ssaValues.count(srcIdx)) {
                    mlir::Value loaded = loadNormInput(srcIdx, true);
                    if (loaded) ssaValues[srcIdx] = loaded;
                  }
                }

                mlir::Value epiResult;
                if (epiCat == TritonOpCategory::BINARY_ELEMENTWISE && epiSlot.wiring.numInputs >= 2) {
                  auto lhsIt = ssaValues.find(epiSlot.wiring.inputSourceIndices[0]);
                  auto rhsIt = ssaValues.find(epiSlot.wiring.inputSourceIndices[1]);
                  if (lhsIt != ssaValues.end() && rhsIt != ssaValues.end()) {
                    epiResult = emitBinaryElementwise(builder, loc, epiMapping, epiSlot, lhsIt->second, rhsIt->second);
                  }
                } else if (epiCat == TritonOpCategory::UNARY_ELEMENTWISE && epiSlot.wiring.numInputs >= 1) {
                  auto inputIt = ssaValues.find(epiSlot.wiring.inputSourceIndices[0]);
                  if (inputIt != ssaValues.end()) {
                    epiResult = emitUnaryElementwise(builder, loc, epiMapping, epiSlot, inputIt->second, static_cast<int>(paddedRowLen));
                  }
                } else if ((epiCat == TritonOpCategory::IDENTITY || epiCat == TritonOpCategory::CAST)
                           && epiSlot.wiring.numInputs >= 1) {
                  auto inputIt = ssaValues.find(epiSlot.wiring.inputSourceIndices[0]);
                  if (inputIt != ssaValues.end()) epiResult = inputIt->second;
                }

                if (epiResult) {
                  for (int o = 0; o < epiSlot.wiring.numOutputs; o++) {
                    ssaValues[epiSlot.wiring.outputSlotIndices[o]] = epiResult;
                  }
                  // Store epilogue result per-row using the same row addressing
                  auto epiOutSlotIdx = epiSlot.wiring.outputSlotIndices[0];
                  auto epiOutArgIt = slotToArgIdx.find(epiOutSlotIdx);
                  if (epiOutArgIt != slotToArgIdx.end()) {
                    auto epiOutFuncArg = getBufferArg(epiOutArgIt->second);
                    auto epiOutPtrType = mlir::cast<mlir::triton::PointerType>(epiOutFuncArg.getType());
                    auto epiOutElemType = epiOutPtrType.getPointeeType();
                    auto epiOutPtrTensorType = mlir::RankedTensorType::get({paddedRowLen}, epiOutPtrType);
                    auto epiRowRange = builder.create<mlir::triton::MakeRangeOp>(loc, rowRangeType, 0, paddedRowLen);
                    auto epiRowBaseSplat = builder.create<mlir::triton::SplatOp>(loc, rowRangeType, rowBase);
                    auto epiRowOffsets = builder.create<mlir::arith::AddIOp>(loc, epiRowBaseSplat, epiRowRange);
                    auto epiRowLenConst = builder.create<mlir::arith::ConstantIntOp>(loc, logicalRowLen, 32);
                    auto epiRowLenSplat = builder.create<mlir::triton::SplatOp>(loc, rowRangeType, epiRowLenConst);
                    auto epiRowMask = builder.create<mlir::arith::CmpIOp>(
                        loc, mlir::arith::CmpIPredicate::slt, epiRowRange, epiRowLenSplat);
                    auto epiSplatPtr = builder.create<mlir::triton::SplatOp>(loc, epiOutPtrTensorType, epiOutFuncArg);
                    auto epiPtrs = builder.create<mlir::triton::AddPtrOp>(loc, epiOutPtrTensorType, epiSplatPtr, epiRowOffsets);
                    auto epiStoreVal = castTo(builder, loc, epiResult, epiOutElemType);
                    builder.create<mlir::triton::StoreOp>(loc, epiPtrs, epiStoreVal, epiRowMask,
                        mlir::triton::CacheModifier::NONE,
                        mlir::triton::EvictionPolicy::NORMAL);
                    customStoredOutputs.insert(epiOutSlotIdx);
                  }
                  DSP_DIAG(COMPILE, "  norm epilogue op '%s' at slot %d emitted successfully",
                            epiSlot.ident.opName.c_str(), epi);
                }
              }
            }

            builder.create<mlir::scf::YieldOp>(loc);
            builder.setInsertionPointAfter(rowIf);
          } else if (cat == TritonOpCategory::ROPE) {
            // ─── ROPE: paired elementwise rotation ───
            if (slot.wiring.numInputs >= 3 && slot.wiring.numOutputs >= 1) {
              int inputSrc2 = slot.wiring.inputSourceIndices[0];
              int cosSrc = slot.wiring.inputSourceIndices[1];
              int sinSrc = slot.wiring.inputSourceIndices[2];
              int outSlot2 = slot.wiring.outputSlotIndices[0];

              auto inPtr2 = getSlotArgPtr(inputSrc2);
              auto cosPtr2 = getSlotArgPtr(cosSrc);
              auto sinPtr2 = getSlotArgPtr(sinSrc);
              auto outPtr2 = getSlotArgPtr(outSlot2);

              NDArray* inArr2 = resolveArr(inputSrc2);
              NDArray* cosArr2 = resolveArr(cosSrc);
              NDArray* outArr2 = resolveArr(outSlot2);

              // Resolve shapes: prefer live arrays, fall back to cached shape info
              std::vector<LongType> inShapeVec, cosShapeVec2;
              if (inArr2) {
                for (int d = 0; d < inArr2->rankOf(); d++) inShapeVec.push_back(inArr2->sizeAt(d));
              } else {
                inShapeVec = resolveShape(inputSrc2);
              }
              if (cosArr2) {
                for (int d = 0; d < cosArr2->rankOf(); d++) cosShapeVec2.push_back(cosArr2->sizeAt(d));
              } else {
                cosShapeVec2 = resolveShape(cosSrc);
              }

              int nElems = 0;
              if (outArr2) {
                nElems = static_cast<int>(outArr2->lengthOf());
              } else if (!inShapeVec.empty()) {
                nElems = 1;
                for (auto d : inShapeVec) nElems *= static_cast<int>(d);
              }

              if (cosPtr2 && sinPtr2 && !inShapeVec.empty() && !cosShapeVec2.empty() && nElems > 0) {
                // Extract headDim and numHeads from input shape
                int inputRank2 = static_cast<int>(inShapeVec.size());
                int headDim2 = (inputRank2 > 0) ? static_cast<int>(inShapeVec[inputRank2 - 1]) : 0;
                int numHeads2 = (inputRank2 >= 3) ? static_cast<int>(inShapeVec[inputRank2 - 2]) : 1;

                // Try SSA register-level path
                auto ssaIt2 = ssaValues.find(inputSrc2);
                bool canUseSSA2 = ssaIt2 != ssaValues.end()
                                  && headDim2 > 0 && (headDim2 % 2 == 0)
                                  && (blockSize % headDim2 == 0)
                                  && (blockSize <= numHeads2 * headDim2);

                if (canUseSSA2) {
                  // Register-based ROPE — no store/reload needed
                  auto result2 = emitRoPESSA(builder, loc, ssaIt2->second,
                                              cosPtr2, sinPtr2, pid, blockSize,
                                              headDim2, numHeads2, cosShapeVec2, nElems);
                  for (int o = 0; o < slot.wiring.numOutputs; o++)
                    ssaValues[slot.wiring.outputSlotIndices[o]] = result2;
                } else if (inPtr2 && outPtr2) {
                  // Pointer-based emitter (requires both input and output buffer pointers)
                  auto maybeStoreSSA = [&](int srcIdx) {
                    auto ssaIt3 = ssaValues.find(srcIdx);
                    auto argIt2 = slotToArgIdx.find(srcIdx);
                    if (ssaIt3 != ssaValues.end() && argIt2 != slotToArgIdx.end()) {
                      auto funcArg2 = getBufferArg(argIt2->second);
                      auto ptrType2 = mlir::cast<mlir::triton::PointerType>(funcArg2.getType());
                      auto elemType2 = ptrType2.getPointeeType();
                      auto ptrTensorType2 = mlir::RankedTensorType::get({blockSize}, ptrType2);
                      auto splatPtr2 = builder.create<mlir::triton::SplatOp>(loc, ptrTensorType2, funcArg2);
                      auto ptrs2 = builder.create<mlir::triton::AddPtrOp>(loc, ptrTensorType2, splatPtr2, offsets);
                      auto storeVal2 = castTo(builder, loc, ssaIt3->second, elemType2);
                      builder.create<mlir::triton::StoreOp>(loc, ptrs2, storeVal2, mask,
                          mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL);
                    }
                  };
                  maybeStoreSSA(inputSrc2);

                  int ropeType2 = (slot.args.numIArgs > 0 && slot.args.iArgs) ? static_cast<int>(slot.args.iArgs[0]) : 0;
                  emitRoPESection(builder, loc, pid, blockSize,
                                  inPtr2, cosPtr2, sinPtr2, outPtr2,
                                  inShapeVec, cosShapeVec2, ropeType2, nElems);

                  // Reload result from buffer to SSA
                  auto outArgIt2 = slotToArgIdx.find(outSlot2);
                  if (outArgIt2 != slotToArgIdx.end()) {
                    auto outFuncArg = getBufferArg(outArgIt2->second);
                    auto outPtrType2 = mlir::cast<mlir::triton::PointerType>(outFuncArg.getType());
                    auto outPtrTensorType2 = mlir::RankedTensorType::get({blockSize}, outPtrType2);
                    auto splatOutPtr2 = builder.create<mlir::triton::SplatOp>(loc, outPtrTensorType2, outFuncArg);
                    auto outPtrs2 = builder.create<mlir::triton::AddPtrOp>(loc, outPtrTensorType2, splatOutPtr2, offsets);
                    auto reloaded = builder.create<mlir::triton::LoadOp>(loc,
                        outPtrs2.getResult(), mask.getResult(), mlir::Value(),
                        mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL, false);
                    for (int o = 0; o < slot.wiring.numOutputs; o++) ssaValues[slot.wiring.outputSlotIndices[o]] = reloaded;
                  }
                } else {
                  DSP_DIAG_SLOT(FALLBACK, si, "TritonIRBuilder(sectioned): ROPE '%s' at slot %d — "
                            "SSA path unavailable and missing buffer pointers (inPtr=%d outPtr=%d)",
                            slot.ident.opName.c_str(), si, inPtr2 ? 1 : 0, outPtr2 ? 1 : 0);
                }
              }
            }
          } else if (cat == TritonOpCategory::CONSTANT_GENERATION) {
            // Constant generation ops produce values independent of input data.
            // Must emit proper constants — NOT forward input SSA values.
            DataType outDtype = FLOAT32;
            if (slot.wiring.numOutputs > 0) {
              int outIdx = slot.wiring.outputSlotIndices[0];
              outDtype = resolveDtype(outIdx);
            }
            auto cgElemType = getMLIRType(builder, outDtype);
            auto cgTensorType = mlir::RankedTensorType::get({blockSize}, cgElemType);

            std::string opLower3 = slot.ident.opName;
            std::transform(opLower3.begin(), opLower3.end(), opLower3.begin(), ::tolower);

            mlir::Value cgResult;
            // Helper to resolve NDArray* for a slot index in the sectioned module scope
            auto secResolveArr = [&](int idx) -> NDArray* {
              if (idx < 0) {
                int extIdx = -(idx + 1);
                if (extIdx < numExternalInputs && externalInputs && externalInputs[extIdx])
                  return externalInputs[extIdx];
                return nullptr;
              }
              if (idx < totalOutputSlots && outputSlots && outputSlots[idx])
                return outputSlots[idx];
              return nullptr;
            };

            if (opLower3 == "ones_as" || opLower3 == "oneslike" || opLower3 == "ones_like") {
              cgResult = splatConstantF32(builder, loc, cgTensorType, 1.0f);
            } else if (opLower3 == "create" || opLower3 == "set_scalar") {
              float fillVal = 0.0f;
              if (slot.args.numTArgs > 0 && slot.args.tArgs) {
                fillVal = static_cast<float>(slot.args.tArgs[0]);
              } else if (slot.wiring.numOutputs > 0) {
                int outIdx = slot.wiring.outputSlotIndices[0];
                auto* arr = secResolveArr(outIdx);
                if (arr && arr->lengthOf() > 0) {
                  arr->syncToHost();
                  fillVal = arr->e<float>(0);
                }
              }
              cgResult = splatConstantF32(builder, loc, cgTensorType, fillVal);
            } else if (opLower3 == "range") {
              float start = 0.0f, step = 1.0f;
              if (slot.args.numTArgs >= 1 && slot.args.tArgs) start = static_cast<float>(slot.args.tArgs[0]);
              if (slot.args.numTArgs >= 3 && slot.args.tArgs) step = static_cast<float>(slot.args.tArgs[2]);
              int rangeLen = blockSize;
              if (slot.wiring.numOutputs > 0) {
                int outIdx = slot.wiring.outputSlotIndices[0];
                auto outShape = resolveShape(outIdx);
                LongType outLen = shapeLength(outShape);
                if (outLen > 0) rangeLen = static_cast<int>(outLen);
              }
              auto cgI32TensorTy = mlir::RankedTensorType::get({blockSize}, builder.getI32Type());
              auto cgF32TensorTy = mlir::RankedTensorType::get({blockSize}, builder.getF32Type());
              auto rangeLenConst = builder.create<mlir::arith::ConstantIntOp>(loc, rangeLen, 32);
              auto splatRangeLen = builder.create<mlir::triton::SplatOp>(loc, cgI32TensorTy, rangeLenConst);
              auto modOffsets = builder.create<mlir::arith::RemUIOp>(loc, offsets, splatRangeLen);
              auto floatOffsets = builder.create<mlir::arith::SIToFPOp>(loc, cgF32TensorTy, modOffsets);
              auto startSplat = splatConstantF32(builder, loc, cgF32TensorTy, start);
              auto stepSplat = splatConstantF32(builder, loc, cgF32TensorTy, step);
              auto scaled = builder.create<mlir::arith::MulFOp>(loc, floatOffsets, stepSplat);
              cgResult = builder.create<mlir::arith::AddFOp>(loc, startSplat, scaled);
              cgResult = castTo(builder, loc, cgResult, cgElemType);
            } else if (opLower3 == "shape_of") {
              bool emitted = false;
              if (slot.wiring.numOutputs > 0) {
                int outIdx = slot.wiring.outputSlotIndices[0];
                auto* arr = secResolveArr(outIdx);
                if (arr && arr->lengthOf() > 0) {
                  arr->syncToHost();
                  int outLen = static_cast<int>(arr->lengthOf());
                  auto cgI32TensorTy = mlir::RankedTensorType::get({blockSize}, builder.getI32Type());
                  auto cgF32TensorTy = mlir::RankedTensorType::get({blockSize}, builder.getF32Type());
                  auto outLenConst = builder.create<mlir::arith::ConstantIntOp>(loc, outLen, 32);
                  auto splatOutLen = builder.create<mlir::triton::SplatOp>(loc, cgI32TensorTy, outLenConst);
                  auto modOffs = builder.create<mlir::arith::RemUIOp>(loc, offsets, splatOutLen);
                  cgResult = splatConstantF32(builder, loc, cgF32TensorTy, 0.0f);
                  for (int d = outLen - 1; d >= 0; d--) {
                    float dimVal = static_cast<float>(arr->e<float>(d));
                    auto dimConst = builder.create<mlir::arith::ConstantIntOp>(loc, d, 32);
                    auto splatDim = builder.create<mlir::triton::SplatOp>(loc, cgI32TensorTy, dimConst);
                    auto cmp = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq,
                                                                     modOffs, splatDim);
                    auto dimValSplat = splatConstantF32(builder, loc, cgF32TensorTy, dimVal);
                    cgResult = builder.create<mlir::arith::SelectOp>(loc, cmp, dimValSplat, cgResult);
                  }
                  cgResult = castTo(builder, loc, cgResult, cgElemType);
                  emitted = true;
                }
              }
              if (!emitted) {
                cgResult = splatConstantF32(builder, loc, cgTensorType, 0.0f);
              }
            } else if (opLower3 == "min_max_datatype") {
              float val = 0.0f;
              if (slot.wiring.numOutputs > 0) {
                int outIdx = slot.wiring.outputSlotIndices[0];
                auto* arr = secResolveArr(outIdx);
                if (arr && arr->lengthOf() > 0) {
                  arr->syncToHost();
                  val = arr->e<float>(0);
                }
              }
              cgResult = splatConstantF32(builder, loc, cgTensorType, val);
            } else {
              // Default: zero fill (zeros_like, zeros_as, zeroslike, unknown)
              cgResult = splatConstantF32(builder, loc, cgTensorType, 0.0f);
            }

            if (cgResult) {
              for (int o = 0; o < slot.wiring.numOutputs; o++) ssaValues[slot.wiring.outputSlotIndices[o]] = cgResult;
            }
          } else if (cat == TritonOpCategory::SHAPE_MANIPULATION) {
            // Shape ops (reshape, squeeze, expand_dims, permute, transpose) are
            // now isolated into their own SHAPE_MANIPULATION sections by
            // identifySections() and always run via native fallback.  This SSA
            // forwarding code should never be reached, but is kept as defensive
            // fallback.  It is correct only for non-permute ops (identity view).
            if (slot.wiring.numInputs >= 1) {
              auto inputIt = ssaValues.find(slot.wiring.inputSourceIndices[0]);
              if (inputIt != ssaValues.end()) {
                for (int o = 0; o < slot.wiring.numOutputs; o++) ssaValues[slot.wiring.outputSlotIndices[o]] = inputIt->second;
              }
            }
          }
        }

        // Store cross-section intermediate outputs to global memory
        for (int si = sec.startSlot; si <= sec.endSlot; si++) {
          for (int o = 0; o < slots[si].wiring.numOutputs; o++) {
            int outIdx = slots[si].wiring.outputSlotIndices[o];
            if (!externalOutputs.count(outIdx)) continue;
            if (customStoredOutputs.count(outIdx)) continue;
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
        // ── Matmul section: per-element scalar K-loop + optional epilogue ──
        // For each matmul op in this section, emit scalar matmul and store/load back.
        // If the section has absorbed epilogue ops (bias add, activation), those are
        // processed as elementwise ops after the matmul, reading the matmul output
        // from the SSA value map.
        int matmulEndSlot = (sec.matmulEpilogueStartSlot >= 0)
                              ? sec.matmulEpilogueStartSlot - 1
                              : sec.endSlot;
        for (int si = sec.startSlot; si <= matmulEndSlot; si++) {
          auto& slot = slots[si];
          if (getOpCategory(slot.ident.opName) != TritonOpCategory::MATMUL) continue;
          if (slot.wiring.numInputs < 2 || slot.wiring.numOutputs < 1) continue;

          // ── Check for fused_gemm_swiglu: use dedicated GatedMLP kernel ──
          {
            std::string opLower = slot.ident.opName;
            std::transform(opLower.begin(), opLower.end(), opLower.begin(), ::tolower);
            if (opLower == "fused_gemm_swiglu" && slot.wiring.numInputs >= 3) {
              int xSrc = slot.wiring.inputSourceIndices[0];
              int wGateSrc = slot.wiring.inputSourceIndices[1];
              int wUpSrc = slot.wiring.inputSourceIndices[2];
              int outSlot = slot.wiring.outputSlotIndices[0];
              auto xPtr = getSlotArgPtr(xSrc);
              auto wGatePtr = getSlotArgPtr(wGateSrc);
              auto wUpPtr = getSlotArgPtr(wUpSrc);
              auto outArgPtr = getSlotArgPtr(outSlot);
              auto xShape = resolveShape(xSrc);
              auto wShape = resolveShape(wGateSrc);
              if (xPtr && wGatePtr && wUpPtr && outArgPtr &&
                  xShape.size() >= 2 && wShape.size() >= 2) {
                int gM = static_cast<int>(xShape[xShape.size() - 2]);
                int gK = static_cast<int>(xShape[xShape.size() - 1]);
                int gN = static_cast<int>(wShape[wShape.size() - 1]);
                if (gM > 0 && gN > 0 && gK > 0) {
                  emitGatedMLPKernel(builder, loc, xPtr, wGatePtr, wUpPtr, outArgPtr,
                                      gM, gN, gK, 128, 128, 32);
                  DataType outDtype = resolveDtype(outSlot);
                  auto loaded = loadBlock(outSlot, outDtype);
                  if (loaded) {
                    for (int o = 0; o < slot.wiring.numOutputs; o++)
                      ssaValues[slot.wiring.outputSlotIndices[o]] = loaded;
                  }
                  continue;
                }
              }
            }
          }

          // ── Check for fused_two_layer_mlp: use dedicated two-layer MLP kernel (FastVLA) ──
          // Computes: tanh(ReLU(x @ W1 + b1) @ W2 + b2) in a single kernel.
          // Inputs: x [M,D], W1 [D,H], b1 [H], W2 [H,A], b2 [A]
          {
            std::string opLower = slot.ident.opName;
            std::transform(opLower.begin(), opLower.end(), opLower.begin(), ::tolower);
            if ((opLower == "fused_two_layer_mlp" || opLower == "fusedtwolayermlp") && slot.wiring.numInputs >= 5) {
              int xSrc = slot.wiring.inputSourceIndices[0];
              int w1Src = slot.wiring.inputSourceIndices[1];
              int b1Src = slot.wiring.inputSourceIndices[2];
              int w2Src = slot.wiring.inputSourceIndices[3];
              int b2Src = slot.wiring.inputSourceIndices[4];
              int outSlot = slot.wiring.outputSlotIndices[0];
              auto xArgPtr = getSlotArgPtr(xSrc);
              auto w1ArgPtr = getSlotArgPtr(w1Src);
              auto b1ArgPtr = getSlotArgPtr(b1Src);
              auto w2ArgPtr = getSlotArgPtr(w2Src);
              auto b2ArgPtr = getSlotArgPtr(b2Src);
              auto outArgPtr = getSlotArgPtr(outSlot);
              auto xShape = resolveShape(xSrc);
              auto w1Shape = resolveShape(w1Src);
              auto w2Shape = resolveShape(w2Src);
              if (xArgPtr && w1ArgPtr && b1ArgPtr && w2ArgPtr && b2ArgPtr && outArgPtr &&
                  xShape.size() >= 2 && w1Shape.size() >= 2 && w2Shape.size() >= 2) {
                int mlpM = static_cast<int>(xShape[xShape.size() - 2]);    // batch
                int mlpD = static_cast<int>(xShape[xShape.size() - 1]);    // input dim
                int mlpH = static_cast<int>(w1Shape[w1Shape.size() - 1]);  // hidden dim
                int mlpA = static_cast<int>(w2Shape[w2Shape.size() - 1]);  // output dim
                if (mlpM > 0 && mlpD > 0 && mlpH > 0 && mlpA > 0) {
                  // Block sizes: blockM=16 (small batch typical for VLA/decode),
                  // blockH=256 (hidden tile stays in registers), blockK=32, blockA=32
                  int bM = std::min(16, mlpM);
                  int bH = std::min(256, mlpH);
                  int bK = 32;
                  int bA = (mlpA <= 16) ? 16 : 32;
                  emitFusedTwoLayerMLPKernel(builder, loc, xArgPtr, w1ArgPtr, b1ArgPtr,
                                              w2ArgPtr, b2ArgPtr, outArgPtr,
                                              mlpM, mlpD, mlpH, mlpA,
                                              bM, bH, bK, bA);
                  DataType outDtype = resolveDtype(outSlot);
                  auto loaded = loadBlock(outSlot, outDtype);
                  if (loaded) {
                    for (int o = 0; o < slot.wiring.numOutputs; o++)
                      ssaValues[slot.wiring.outputSlotIndices[o]] = loaded;
                  }
                  continue;
                }
              }
            }
          }

          // ── Check for rms_norm_linear: use dedicated single-pass kernel ──
          {
            std::string opLower = slot.ident.opName;
            std::transform(opLower.begin(), opLower.end(), opLower.begin(), ::tolower);
            if ((opLower == "rms_norm_linear" || opLower == "rmsnormlinear") && slot.wiring.numInputs >= 3) {
              int xSrc = slot.wiring.inputSourceIndices[0];
              int gammaSrc = slot.wiring.inputSourceIndices[1];
              int wSrc = slot.wiring.inputSourceIndices[2];
              int outSlot = slot.wiring.outputSlotIndices[0];
              auto xArgPtr = getSlotArgPtr(xSrc);
              auto gammaArgPtr = getSlotArgPtr(gammaSrc);
              auto wArgPtr = getSlotArgPtr(wSrc);
              auto outArgPtr = getSlotArgPtr(outSlot);
              auto xShape = resolveShape(xSrc);
              auto wShape = resolveShape(wSrc);
              float eps = (slot.args.numTArgs > 0 && slot.args.tArgs) ? static_cast<float>(slot.args.tArgs[0]) : 1e-6f;
              if (xArgPtr && gammaArgPtr && wArgPtr && outArgPtr &&
                  xShape.size() >= 2 && wShape.size() >= 2) {
                int rM = static_cast<int>(xShape[xShape.size() - 2]);
                int rK = static_cast<int>(xShape[xShape.size() - 1]);
                int rN = static_cast<int>(wShape[wShape.size() - 1]);
                if (rM > 0 && rN > 0 && rK > 0) {
                  emitRmsNormLinearKernel(builder, loc, xArgPtr, gammaArgPtr, wArgPtr, outArgPtr,
                                          rM, rN, rK, eps, 128, 128, 32);
                  DataType outDtype = resolveDtype(outSlot);
                  auto loaded = loadBlock(outSlot, outDtype);
                  if (loaded) {
                    for (int o = 0; o < slot.wiring.numOutputs; o++)
                      ssaValues[slot.wiring.outputSlotIndices[o]] = loaded;
                  }
                  continue;
                }
              }
            }
          }

          int aSrc = slot.wiring.inputSourceIndices[0];
          int bSrc = slot.wiring.inputSourceIndices[1];
          int cSlot = slot.wiring.outputSlotIndices[0];

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
              for (int o = 0; o < slot.wiring.numOutputs; o++) ssaValues[slot.wiring.outputSlotIndices[o]] = loaded;
            }
          } else {
            std::string msg = "TritonIRBuilder::buildSectionedModule: matmul at slot " + std::to_string(si) +
                " op='" + slot.ident.opName + "'"
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

        // ── Matmul epilogue: emit absorbed elementwise ops ──
        // These ops (bias add, activation, cast) operate on the matmul output
        // which is already in the SSA value map from the store/load above.
        // Uses the same emitter functions as the main elementwise section.
        if (sec.matmulEpilogueCount > 0 && sec.matmulEpilogueStartSlot >= 0) {
          DSP_DIAG(COMPILE, "buildSectionedModule: emitting %d matmul epilogue ops [%d-%d]",
                    sec.matmulEpilogueCount, sec.matmulEpilogueStartSlot, sec.matmulEpilogueEndSlot);

          // Create 1D offsets and mask for epilogue loads/stores.
          // The matmul section uses a 2D grid (pid_m, pid_n) but the epilogue
          // operates on the linearized output, so we create a 1D block for it.
          auto epiI32Type = builder.getI32Type();
          auto epiI1Type = builder.getI1Type();
          auto epiI32BsType = mlir::RankedTensorType::get({blockSize}, epiI32Type);
          auto epiI1BsType = mlir::RankedTensorType::get({blockSize}, epiI1Type);
          auto epiPidScalar = builder.create<mlir::triton::GetProgramIdOp>(
              loc, epiI32Type, mlir::triton::ProgramIDDim::X);
          auto epiBlockSizeConst = builder.create<mlir::arith::ConstantIntOp>(loc, blockSize, 32);
          auto epiBlockOffset = builder.create<mlir::arith::MulIOp>(loc, epiPidScalar, epiBlockSizeConst);
          auto epiRange = builder.create<mlir::triton::MakeRangeOp>(loc, epiI32BsType, 0, blockSize);
          auto epiBlockOffsetSplat = builder.create<mlir::triton::SplatOp>(loc, epiI32BsType, epiBlockOffset);
          auto epiOffsets = builder.create<mlir::arith::AddIOp>(loc, epiBlockOffsetSplat, epiRange);

          // Determine n_elements for the epilogue (from first epilogue slot's output)
          int64_t epiNElements = 0;
          for (int esi = sec.matmulEpilogueStartSlot; esi <= sec.matmulEpilogueEndSlot && epiNElements == 0; esi++) {
            if (slots[esi].wiring.numOutputs > 0) {
              int epiOutIdx = slots[esi].wiring.outputSlotIndices[0];
              NDArray* epiOutArr = (epiOutIdx >= 0 && epiOutIdx < totalOutputSlots) ? outputSlots[epiOutIdx] : nullptr;
              if (epiOutArr) epiNElements = epiOutArr->lengthOf();
            }
          }
          if (epiNElements <= 0) epiNElements = static_cast<int64_t>(sec.matmulM) * sec.matmulN;
          auto epiNElemConst = builder.create<mlir::arith::ConstantIntOp>(loc, static_cast<int>(epiNElements), 32);
          auto epiNElemSplat = builder.create<mlir::triton::SplatOp>(loc, epiI32BsType, epiNElemConst);
          auto epiMask = builder.create<mlir::arith::CmpIOp>(
              loc, mlir::arith::CmpIPredicate::slt, epiOffsets, epiNElemSplat);

          for (int si = sec.matmulEpilogueStartSlot; si <= sec.matmulEpilogueEndSlot; si++) {
            auto& epiSlot = slots[si];
            auto epiCat = getOpCategory(epiSlot.ident.opName);
            auto epiIt = opTable.find(epiSlot.ident.opName);
            if (epiIt == opTable.end()) continue;
            const auto& epiMapping = epiIt->second;

            // Ensure all inputs are in SSA value map (load from global memory if needed)
            for (int inp = 0; inp < epiSlot.wiring.numInputs; inp++) {
              int srcIdx = epiSlot.wiring.inputSourceIndices[inp];
              if (!ssaValues.count(srcIdx)) {
                auto argIt = slotToArgIdx.find(srcIdx);
                if (argIt != slotToArgIdx.end()) {
                  auto bufferArg = getBufferArg(argIt->second);
                  if (bufferArg) {
                    auto ptrType = mlir::cast<mlir::triton::PointerType>(bufferArg.getType());
                    auto elemType = ptrType.getPointeeType();
                    auto ptrTensorType = mlir::RankedTensorType::get({blockSize}, ptrType);
                    auto splatPtr = builder.create<mlir::triton::SplatOp>(loc, ptrTensorType, bufferArg);
                    auto ptrs = builder.create<mlir::triton::AddPtrOp>(loc, ptrTensorType, splatPtr, epiOffsets);
                    auto loaded = builder.create<mlir::triton::LoadOp>(loc,
                        ptrs.getResult(), epiMask.getResult(), mlir::Value(),
                        mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL, false);
                    ssaValues[srcIdx] = loaded;
                  }
                }
              }
            }

            mlir::Value epiResult;
            if (epiCat == TritonOpCategory::BINARY_ELEMENTWISE && epiSlot.wiring.numInputs >= 2) {
              auto lhsIt = ssaValues.find(epiSlot.wiring.inputSourceIndices[0]);
              auto rhsIt = ssaValues.find(epiSlot.wiring.inputSourceIndices[1]);
              if (lhsIt != ssaValues.end() && rhsIt != ssaValues.end()) {
                epiResult = emitBinaryElementwise(builder, loc, epiMapping, epiSlot, lhsIt->second, rhsIt->second);
              }
            } else if (epiCat == TritonOpCategory::UNARY_ELEMENTWISE && epiSlot.wiring.numInputs >= 1) {
              auto inputIt = ssaValues.find(epiSlot.wiring.inputSourceIndices[0]);
              if (inputIt != ssaValues.end()) {
                epiResult = emitUnaryElementwise(builder, loc, epiMapping, epiSlot, inputIt->second, blockSize);
              }
            } else if (epiCat == TritonOpCategory::COMPARISON && epiSlot.wiring.numInputs >= 1) {
              auto lhsIt = ssaValues.find(epiSlot.wiring.inputSourceIndices[0]);
              if (lhsIt != ssaValues.end()) {
                if (epiSlot.wiring.numInputs >= 2) {
                  auto rhsIt = ssaValues.find(epiSlot.wiring.inputSourceIndices[1]);
                  if (rhsIt != ssaValues.end()) {
                    epiResult = emitComparisonOp(builder, loc, epiSlot.ident.opName, lhsIt->second, rhsIt->second, blockSize);
                  }
                } else if (epiSlot.args.numTArgs > 0 && epiSlot.args.tArgs) {
                  auto lhsTensorType = mlir::cast<mlir::RankedTensorType>(lhsIt->second.getType());
                  auto lhsElemType = lhsTensorType.getElementType();
                  double scalarVal = epiSlot.args.tArgs[0];
                  mlir::Value rhsSplat;
                  if (mlir::isa<mlir::FloatType>(lhsElemType)) {
                    rhsSplat = splatConstantF32(builder, loc, lhsTensorType, static_cast<float>(scalarVal));
                  } else {
                    rhsSplat = splatConstantI32(builder, loc, lhsTensorType, static_cast<int>(scalarVal));
                  }
                  epiResult = emitComparisonOp(builder, loc, epiSlot.ident.opName, lhsIt->second, rhsSplat, blockSize);
                }
              }
            } else if ((epiCat == TritonOpCategory::IDENTITY || epiCat == TritonOpCategory::CAST)
                       && epiSlot.wiring.numInputs >= 1) {
              auto inputIt = ssaValues.find(epiSlot.wiring.inputSourceIndices[0]);
              if (inputIt != ssaValues.end()) epiResult = inputIt->second;
            }

            if (epiResult) {
              for (int o = 0; o < epiSlot.wiring.numOutputs; o++) {
                ssaValues[epiSlot.wiring.outputSlotIndices[o]] = epiResult;
              }
              // Store epilogue result to output buffer
              auto outSlotIdx = epiSlot.wiring.outputSlotIndices[0];
              auto outArgIt = slotToArgIdx.find(outSlotIdx);
              if (outArgIt != slotToArgIdx.end()) {
                auto outFuncArg = getBufferArg(outArgIt->second);
                auto outPtrType = mlir::cast<mlir::triton::PointerType>(outFuncArg.getType());
                auto outElemType = outPtrType.getPointeeType();
                auto outPtrTensorType = mlir::RankedTensorType::get({blockSize}, outPtrType);
                auto splatOutPtr = builder.create<mlir::triton::SplatOp>(loc, outPtrTensorType, outFuncArg);
                auto outPtrs = builder.create<mlir::triton::AddPtrOp>(loc, outPtrTensorType, splatOutPtr, epiOffsets);
                mlir::Value storeVal = castTo(builder, loc, epiResult, outElemType);
                builder.create<mlir::triton::StoreOp>(loc, outPtrs, storeVal, epiMask,
                    mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL);
                customStoredOutputs.insert(outSlotIdx);
              }
              DSP_DIAG(COMPILE, "  matmul epilogue op '%s' at slot %d emitted successfully",
                        epiSlot.ident.opName.c_str(), si);
            }
          }
        }
        break;
      }

      case KernelSectionType::FUSED_ATTENTION: {
        // ── Attention section: emit fused attention kernel ──
        // Handles past_key/past_value (inputs 4-5) and BSHD (3D) vs BHSD (4D) layout.
        // Also handles backward (_bp) ops via emitFusedAttentionBackwardKernel.
        bool loggedAttnTileAdjust = false;
        for (int si = sec.startSlot; si <= sec.endSlot; si++) {
          auto& slot = slots[si];
          if (getOpCategory(slot.ident.opName) != TritonOpCategory::FUSED_ATTENTION) continue;

          // ── Flash Attention Backward: _bp suffix ops ─────────────────────────
          {
            std::string opLowerBpSec = slot.ident.opName;
            std::transform(opLowerBpSec.begin(), opLowerBpSec.end(), opLowerBpSec.begin(), ::tolower);
            if (opLowerBpSec.find("_bp") != std::string::npos) {
              // Backward: needs 6 inputs (dO, Q, K, V, O, L) and 3 outputs (dQ, dK, dV)
              if (slot.wiring.numInputs >= 6 && slot.wiring.numOutputs >= 3) {
                int doSrcSec  = slot.wiring.inputSourceIndices[0];  // dO
                int qSrcSec   = slot.wiring.inputSourceIndices[1];  // Q
                int kSrcSec   = slot.wiring.inputSourceIndices[2];  // K
                int vSrcSec   = slot.wiring.inputSourceIndices[3];  // V
                int oSrcSec   = slot.wiring.inputSourceIndices[4];  // O
                int lseSrcSec = slot.wiring.inputSourceIndices[5];  // L (log-sum-exp)
                int dqSlotSec = slot.wiring.outputSlotIndices[0];   // dQ
                int dkSlotSec = slot.wiring.outputSlotIndices[1];   // dK
                int dvSlotSec = slot.wiring.outputSlotIndices[2];   // dV

                auto qShapeSec2 = resolveShape(qSrcSec);
                auto kShapeSec2 = resolveShape(kSrcSec);
                int batchSizeBp = 1, numQHeadsBp = 1, seqQBp = 1, seqKBp = 1, headDimBp = 64;
                if (qShapeSec2.size() >= 4) {
                  batchSizeBp = static_cast<int>(qShapeSec2[0]);
                  numQHeadsBp = static_cast<int>(qShapeSec2[1]);
                  seqQBp      = static_cast<int>(qShapeSec2[2]);
                  headDimBp   = static_cast<int>(qShapeSec2[3]);
                }
                if (kShapeSec2.size() >= 4) {
                  seqKBp = static_cast<int>(kShapeSec2[2]);
                }
                if (seqQBp <= 0 || seqKBp <= 0 || headDimBp <= 0) {
                  DSP_DIAG(COMPILE, "FUSED_ATTENTION_BP sectioned at slot %d: invalid dims "
                            "seqQ=%d seqK=%d headDim=%d — deferring to C++ native",
                            si, seqQBp, seqKBp, headDimBp);
                  result.valid = false;
                  return result;
                }
                float scaleBp = 1.0f / std::sqrt(static_cast<float>(headDimBp));
                auto bpTileSec = chooseFusedAttentionTileConfig(
                    batchSizeBp, numQHeadsBp, seqQBp, seqKBp, headDimBp);
                if (!bpTileSec.fitsSharedMem) {
                  std::string bpMsg = "TritonIRBuilder::buildSectionedModule: fused attention backward '"
                      + slot.ident.opName + "' at slot " + std::to_string(si) + " cannot fit shared memory";
                  THROW_EXCEPTION(bpMsg.c_str());
                }
                auto dOPtrSec  = getSlotArgPtr(doSrcSec);
                auto qPtrSec   = getSlotArgPtr(qSrcSec);
                auto kPtrSec   = getSlotArgPtr(kSrcSec);
                auto vPtrSec   = getSlotArgPtr(vSrcSec);
                auto oPtrSec   = getSlotArgPtr(oSrcSec);
                auto lsePtrSec = getSlotArgPtr(lseSrcSec);
                auto dQPtrSec  = getSlotArgPtr(dqSlotSec);
                auto dKPtrSec  = getSlotArgPtr(dkSlotSec);
                auto dVPtrSec  = getSlotArgPtr(dvSlotSec);
                if (dOPtrSec && qPtrSec && kPtrSec && vPtrSec && oPtrSec && lsePtrSec &&
                    dQPtrSec && dKPtrSec && dVPtrSec) {
                  emitFusedAttentionBackwardKernel(
                      builder, loc,
                      dOPtrSec, qPtrSec, kPtrSec, vPtrSec, oPtrSec, lsePtrSec,
                      dQPtrSec, dKPtrSec, dVPtrSec,
                      batchSizeBp, numQHeadsBp, seqQBp, seqKBp, headDimBp,
                      scaleBp, bpTileSec.blockM, bpTileSec.blockN);
                  for (int outIdx2 = 0; outIdx2 < slot.wiring.numOutputs; outIdx2++) {
                    int outSlotIdx2 = slot.wiring.outputSlotIndices[outIdx2];
                    auto loaded2 = loadBlock(outSlotIdx2, resolveDtype(outSlotIdx2));
                    if (loaded2) ssaValues[outSlotIdx2] = loaded2;
                  }
                } else {
                  std::string bpMsg2 = "TritonIRBuilder::buildSectionedModule: fused attention backward '"
                      + slot.ident.opName + "' at slot " + std::to_string(si) +
                      " — missing kernel arg ptrs. Cannot compile.";
                  THROW_EXCEPTION(bpMsg2.c_str());
                }
              } else {
                DSP_DIAG(COMPILE, "FUSED_ATTENTION_BP sectioned at slot %d: needs >=6 inputs and "
                          ">=3 outputs, has %d/%d — deferring to C++ native", si,
                          slot.wiring.numInputs, slot.wiring.numOutputs);
                result.valid = false;
                return result;
              }
              continue;  // skip forward attention handler below
            }
          }

          if (slot.wiring.numInputs < 3 || slot.wiring.numOutputs < 1) continue;

          int qSrc = slot.wiring.inputSourceIndices[0];
          // Detect DPA v2 (input order Q,V,K) vs standard (Q,K,V)
          std::string opLowerSec = slot.ident.opName;
          std::transform(opLowerSec.begin(), opLowerSec.end(), opLowerSec.begin(), ::tolower);
          bool isDpaV2Sec = (opLowerSec.find("dot_product_attention") != std::string::npos);

          // 3D Q: compound attention (onnx_multi_head_attention) — handled via dual-buffer kernel
          auto qShapeSec = resolveShape(qSrc);

          int kSrc = isDpaV2Sec ? slot.wiring.inputSourceIndices[2] : slot.wiring.inputSourceIndices[1];
          int vSrc = isDpaV2Sec ? slot.wiring.inputSourceIndices[1] : slot.wiring.inputSourceIndices[2];
          int outSlot = slot.wiring.outputSlotIndices[0];

          // ── Step 1: Extract Q shape to get headDim (needed for past_key detection) ──
          auto qShape = resolveShape(qSrc);
          int batchSize = 1, numQHeads = 1, numKvHeads = 0, seqQ = 1, seqK = 1, headDim = 1;
          bool isBSHD = false;
          bool opUsesBSHDSec = isDpaV2Sec;

          if (qShape.size() >= 4) {
            batchSize = static_cast<int>(qShape[0]);
            if (opUsesBSHDSec) {
              seqQ = static_cast<int>(qShape[1]);
              numQHeads = static_cast<int>(qShape[2]);
              isBSHD = true;
            } else {
              numQHeads = static_cast<int>(qShape[1]);
              seqQ = static_cast<int>(qShape[2]);
            }
            headDim = static_cast<int>(qShape[3]);
          } else if (qShape.size() == 3) {
            batchSize = static_cast<int>(qShape[0]);
            seqQ = static_cast<int>(qShape[1]);
            int hidden = static_cast<int>(qShape[2]);
            numQHeads = (slot.args.numIArgs > 0 && slot.args.iArgs) ? static_cast<int>(slot.args.iArgs[0]) : 1;
            if (numQHeads <= 0) numQHeads = 1;
            headDim = hidden / numQHeads;
            isBSHD = true;
          }

          // ── Step 2: Detect past_key/past_value inputs ──
          // Different attention op types have different input orderings:
          //   ONNX MultiHeadAttention: Q,K,V,bias,key_mask,attn_bias,past_key,past_value
          //   ONNX GroupQueryAttention: Q,K,V,past_key,past_value,seqlen_k,...
          //   DPA v2: Q,V,K,...
          // Instead of hardcoding positions, scan ALL inputs for 4D KV-cache-like shapes.
          // A past_key tensor is 4D BHSD: [batch, kvHeads, seqK, headDim] where headDim
          // matches Q's headDim. This distinguishes it from attention masks [B,H,S,S].
          bool hasPastKv = false;
          int pastKeySrc = -1, pastValueSrc = -1;
          bool pastKeyIsExternal = false;

          for (int inp = 3; inp < slot.wiring.numInputs && !hasPastKv; inp++) {
            int candidateSrc = slot.wiring.inputSourceIndices[inp];
            auto candidateShape = resolveShape(candidateSrc);
            // Accept 4D KV cache shapes, including empty ones [B,H,0,D] at warmup.
            // GQA constraint: KV heads must divide Q heads evenly and be <= numQHeads.
            if (candidateShape.size() == 4) {
              int candidateHD = static_cast<int>(candidateShape[3]);
              int candidateKvH = static_cast<int>(candidateShape[1]);
              if (candidateHD == headDim && candidateKvH > 0 &&
                  candidateKvH <= numQHeads && numQHeads % candidateKvH == 0) {
                pastKeySrc = candidateSrc;
                pastKeyIsExternal = (pastKeySrc < 0);
                hasPastKv = true;
                DSP_DIAG(COMPILE, "ATTN slot=%d found past_key at input[%d] src=%d shape=[%lld,%lld,%lld,%lld] headDim=%d",
                          si, inp, candidateSrc,
                          (long long)candidateShape[0], (long long)candidateShape[1],
                          (long long)candidateShape[2], (long long)candidateShape[3], headDim);
                if (inp + 1 < slot.wiring.numInputs) {
                  int pvCandidate = slot.wiring.inputSourceIndices[inp + 1];
                  auto pvShape = resolveShape(pvCandidate);
                  if (pvShape.size() == 4 && static_cast<int>(pvShape[3]) == headDim) {
                    pastValueSrc = pvCandidate;
                  }
                }
              }
            }
          }

          if (!hasPastKv) {
            DSP_DIAG(COMPILE, "ATTN slot=%d no past_key found (headDim=%d) in %d inputs",
                      si, headDim, slot.wiring.numInputs);
          }

          // Use past_key as effective K source when available
          int effectiveKSrc = hasPastKv ? pastKeySrc : kSrc;
          int effectiveVSrc = (hasPastKv && pastValueSrc >= 0) ? pastValueSrc : vSrc;

          auto effectiveKShape = resolveShape(effectiveKSrc);

          // ── Comprehensive attention compilation diagnostics ──
          if (DSP_DIAG_ENABLED(COMPILE)) {
            auto fmtShape = [](const std::vector<LongType>& s) -> std::string {
              std::string r; for (size_t i = 0; i < s.size(); i++) { if (i) r += ","; r += std::to_string(s[i]); } return r.empty() ? "empty" : r;
            };

            // Summary line
            DSP_DIAG(COMPILE, "ATTN slot=%d op='%s' qSrc=%d kSrc=%d vSrc=%d effectiveKSrc=%d effectiveVSrc=%d "
                      "outSlot=%d hasPastKv=%d pastKeySrc=%d pastValueSrc=%d numInputs=%d numOutputs=%d",
                      si, slot.ident.opName.c_str(), qSrc, kSrc, vSrc, effectiveKSrc, effectiveVSrc,
                      outSlot, hasPastKv, pastKeySrc, pastValueSrc,
                      slot.wiring.numInputs, slot.wiring.numOutputs);

            // Shapes: Q, K (raw), K (effective), V
            auto kShapeDbg = resolveShape(kSrc);
            auto vShapeDbg = resolveShape(vSrc);
            DSP_DIAG(COMPILE, "ATTN slot=%d shapes: Q=[%s] K=[%s] effK=[%s] V=[%s]",
                      si, fmtShape(qShape).c_str(), fmtShape(kShapeDbg).c_str(),
                      fmtShape(effectiveKShape).c_str(), fmtShape(vShapeDbg).c_str());

            // Every input: source index, shape, op name, whether slot array exists and its length
            for (int inp = 0; inp < slot.wiring.numInputs; inp++) {
              int srcIdx = slot.wiring.inputSourceIndices[inp];
              auto srcShape = resolveShape(srcIdx);
              const char* srcOp = "EXT";
              if (srcIdx >= 0 && srcIdx < totalSlots) srcOp = slots[srcIdx].ident.opName.c_str();
              bool slotExists = false;
              LongType slotLen = -1;
              if (srcIdx >= 0 && srcIdx < totalOutputSlots && outputSlots && outputSlots[srcIdx]) {
                slotExists = true;
                slotLen = outputSlots[srcIdx]->lengthOf();
              } else if (srcIdx < 0) {
                int ei = -(srcIdx + 1);
                if (ei < numExternalInputs && externalInputs && externalInputs[ei]) {
                  slotExists = true;
                  slotLen = externalInputs[ei]->lengthOf();
                }
              }
              // Check cachedShapeInfoMap for this source
              std::string cachedShapeStr = "none";
              auto cit = cachedShapeInfoMap.find(srcIdx);
              if (cit != cachedShapeInfoMap.end() && cit->second) {
                LongType cRank = shape::rank(cit->second);
                cachedShapeStr = "[";
                for (int d = 0; d < cRank; d++) {
                  if (d) cachedShapeStr += ",";
                  cachedShapeStr += std::to_string(shape::shapeOf(cit->second)[d]);
                }
                cachedShapeStr += "]";
              }
              DSP_DIAG(COMPILE, "ATTN slot=%d   input[%d] src=%d op='%s' shape=[%s] exists=%d len=%lld cached=%s",
                        si, inp, srcIdx, srcOp, fmtShape(srcShape).c_str(),
                        slotExists, (long long)slotLen, cachedShapeStr.c_str());
            }

            // Every output slot
            for (int outp = 0; outp < slot.wiring.numOutputs; outp++) {
              int outIdx = slot.wiring.outputSlotIndices[outp];
              auto outShape = resolveShape(outIdx);
              DSP_DIAG(COMPILE, "ATTN slot=%d   output[%d] slot=%d shape=[%s]",
                        si, outp, outIdx, fmtShape(outShape).c_str());
            }

            // iArgs and tArgs
            if (slot.args.numIArgs > 0) {
              std::string iStr;
              for (int a = 0; a < slot.args.numIArgs && a < 16; a++) {
                if (a) iStr += ",";
                iStr += std::to_string(slot.args.iArgs[a]);
              }
              DSP_DIAG(COMPILE, "ATTN slot=%d   iArgs=[%s] (%d total)", si, iStr.c_str(), slot.args.numIArgs);
            }
            if (slot.args.numTArgs > 0) {
              char tBuf[256] = {0};
              int toff = 0;
              for (int a = 0; a < slot.args.numTArgs && a < 8 && toff < 240; a++) {
                toff += snprintf(tBuf + toff, sizeof(tBuf) - toff, "%s%.6g", a > 0 ? "," : "", slot.args.tArgs[a]);
              }
              DSP_DIAG(COMPILE, "ATTN slot=%d   tArgs=[%s] (%d total)", si, tBuf, slot.args.numTArgs);
            }
          }
          // Extract KV head count from effective K shape (4D BHSD: [B,KvHeads,seqK,HD])
          if (effectiveKShape.size() == 4) {
            if (hasPastKv) {
              numKvHeads = static_cast<int>(effectiveKShape[1]);
              headDim = static_cast<int>(effectiveKShape[3]);
            } else {
              // No past KV — K shape is same layout as Q (BHSD or BSHD)
              if (isBSHD) {
                // BSHD: [B, seqK, heads, HD]
                numKvHeads = static_cast<int>(effectiveKShape[2]);
              } else {
                // BHSD: [B, heads, seqK, HD]
                numKvHeads = static_cast<int>(effectiveKShape[1]);
              }
            }
          } else if (effectiveKShape.size() == 3) {
            // ONNX MHA K/V stay 3D for GQA: [B, seqK, kvHeads * headDim].
            // Infer KV heads from kvHidden instead of assuming numQHeads.
            int kvHidden = static_cast<int>(effectiveKShape[2]);
            if (headDim > 0 && kvHidden > 0 && kvHidden % headDim == 0) {
              int inferredKvHeads = kvHidden / headDim;
              if (inferredKvHeads > 0 &&
                  inferredKvHeads <= numQHeads &&
                  numQHeads % inferredKvHeads == 0) {
                numKvHeads = inferredKvHeads;
              }
            }
          }
          // Default: MHA (KV heads = Q heads)
          if (numKvHeads <= 0) numKvHeads = numQHeads;

          // Determine if we need dual-buffer mode (3D Q with past_key)
          bool useDualBuffer = (qShape.size() == 3 && hasPastKv);
          int pastSeqLen = 0, seqKVCur = 0;

          if (useDualBuffer) {
            // past_key shape is 4D BHSD: [B, kvH, pastSeq, D]
            pastSeqLen = static_cast<int>(effectiveKShape[2]);
            auto curKShape = resolveShape(kSrc);
            seqKVCur = (curKShape.size() == 3) ? static_cast<int>(curKShape[1]) : 1;
            seqK = pastSeqLen + seqKVCur;
          } else {
            // seqK from effective K source
            if (effectiveKShape.size() >= 4) {
              int seqKDim = (opUsesBSHDSec && !hasPastKv) ? 1 : 2;
              seqK = static_cast<int>(effectiveKShape[seqKDim]);
            } else if (effectiveKShape.size() == 3) {
              seqK = static_cast<int>(effectiveKShape[1]);
            }
          }

          // past_key is always 4D BHSD; current key follows Q layout
          bool kIsBSHD = hasPastKv ? false : isBSHD;

          // seqK=0 means slot output shapes are stale (cached from warmup before
          // KV cache was populated).  Try deriving seqK from actual external inputs
          // which Java passes with correct shapes.
          bool seqKDerivedFromExternal = false;
          if (seqK <= 0) {
            int derivedSeqK = 0;
            int derivedKvHeads = 0;
            std::string derivedSource;

            // Strategy 1: Walk back from K source to find KV cache external inputs.
            // The K input often comes from a Concat(past_key, current_key) op.
            // Find that concat's past_key external input and use its seqK.
            if (kSrc >= 0) {
              bool kProducerFound = false;
              for (int s = 0; s < totalSlots && !kProducerFound; s++) {
                for (int o = 0; o < slots[s].wiring.numOutputs && !kProducerFound; o++) {
                  if (slots[s].wiring.outputSlotIndices[o] == kSrc) {
                    // Found the producer of K. Check its inputs for external KV cache.
                    for (int pi = 0; pi < slots[s].wiring.numInputs; pi++) {
                      int psrc = slots[s].wiring.inputSourceIndices[pi];
                      if (psrc < 0) {
                        // External input — check if it looks like a KV cache
                        int extIdx = -(psrc + 1);
                        if (extIdx < numExternalInputs && externalInputs && externalInputs[extIdx]) {
                          auto& ext = *externalInputs[extIdx];
                          if (ext.rankOf() == 4 && !ext.isEmpty() &&
                              (ext.dataType() == FLOAT32 || ext.dataType() == HALF || ext.dataType() == BFLOAT16)) {
                            int extSeqK = static_cast<int>(ext.sizeAt(2));
                            int extHD = static_cast<int>(ext.sizeAt(3));
                            int extKvH = static_cast<int>(ext.sizeAt(1));
                            // GQA constraint: KV heads must divide Q heads evenly
                            if (extHD == headDim && extSeqK > derivedSeqK &&
                                extKvH > 0 && extKvH <= numQHeads && numQHeads % extKvH == 0) {
                              derivedSeqK = extSeqK;
                              derivedKvHeads = extKvH;
                              derivedSource = "K-producer-ext[" + std::to_string(extIdx) + "]";
                            }
                          }
                        }
                      } else {
                        // Slot output — check resolved shape for 4D KV cache
                        auto pshape = resolveShape(psrc);
                        int candidateKvH = (pshape.size() == 4) ? static_cast<int>(pshape[1]) : 0;
                        // GQA constraint: KV heads must divide Q heads evenly
                        if (pshape.size() == 4 && pshape[3] == headDim && pshape[2] > 0 &&
                            candidateKvH > 0 && candidateKvH <= numQHeads && numQHeads % candidateKvH == 0) {
                          int candidateSeqK = static_cast<int>(pshape[2]);
                          if (candidateSeqK > derivedSeqK) {
                            derivedSeqK = candidateSeqK;
                            derivedKvHeads = candidateKvH;
                            derivedSource = "K-producer-slot[" + std::to_string(psrc) + "]";
                          }
                        }
                      }
                    }
                    kProducerFound = true;
                  }
                }
              }
            }

            // Strategy 2: Scan ALL attention op inputs for 4D KV cache shapes.
            // Covers cases where past_key is a direct input to the attention op.
            if (derivedSeqK == 0) {
              for (int inp = 0; inp < slot.wiring.numInputs; inp++) {
                int src = slot.wiring.inputSourceIndices[inp];
                auto shape = resolveShape(src);
                int candidateKvH = (shape.size() == 4) ? static_cast<int>(shape[1]) : 0;
                // GQA constraint: KV heads must divide Q heads evenly
                if (shape.size() == 4 && shape[3] == headDim && shape[2] > 0 &&
                    candidateKvH > 0 && candidateKvH <= numQHeads && numQHeads % candidateKvH == 0) {
                  int candidateSeqK = static_cast<int>(shape[2]);
                  if (candidateSeqK > derivedSeqK) {
                    derivedSeqK = candidateSeqK;
                    derivedKvHeads = candidateKvH;
                    derivedSource = "attn-input[" + std::to_string(inp) + "]";
                  }
                }
              }
            }

            // Strategy 3: Broad scan of ALL external inputs for 4D FP arrays
            // matching KV cache pattern [batch, heads, seqK, headDim].
            if (derivedSeqK == 0 && externalInputs) {
              for (int ei = 0; ei < numExternalInputs; ei++) {
                if (!externalInputs[ei]) continue;
                auto& ext = *externalInputs[ei];
                if (ext.rankOf() != 4 || ext.isEmpty()) continue;
                if (ext.dataType() != FLOAT32 && ext.dataType() != HALF && ext.dataType() != BFLOAT16) continue;
                int extBatch = static_cast<int>(ext.sizeAt(0));
                int extHD = static_cast<int>(ext.sizeAt(3));
                int extSeqK = static_cast<int>(ext.sizeAt(2));
                // Match: same batch, same headDim, non-zero seqK, heads divides Q heads
                if (extBatch == batchSize && extHD == headDim && extSeqK > 0) {
                  int extHeads = static_cast<int>(ext.sizeAt(1));
                  if (extHeads > 0 && extHeads <= numQHeads && numQHeads % extHeads == 0 && extSeqK > derivedSeqK) {
                    derivedSeqK = extSeqK;
                    derivedKvHeads = extHeads;
                    derivedSource = "ext-scan[" + std::to_string(ei) + "]";
                  }
                }
              }
            }

            if (derivedSeqK > 0) {
              // For concat-based K (K source is a slot that concatenates past+current),
              // the attention's total seqK = past_seqK + seqQ
              seqK = derivedSeqK + seqQ;
              seqKDerivedFromExternal = true;
              // Also correct numKvHeads if we got it from a 4D KV cache shape
              if (derivedKvHeads > 0 && derivedKvHeads != numKvHeads) {
                DSP_DIAG(COMPILE, "ATTN slot=%d correcting numKvHeads from %d to %d (from %s)",
                          si, numKvHeads, derivedKvHeads, derivedSource.c_str());
                numKvHeads = derivedKvHeads;
              }
              DSP_DIAG(COMPILE, "ATTN slot=%d derived seqK=%d from %s (pastSeqK=%d + seqQ=%d, headDim=%d, numKvHeads=%d)",
                        si, seqK, derivedSource.c_str(), derivedSeqK, seqQ, headDim, numKvHeads);
            } else {
              // Truly unresolvable — fall back to C++
              DSP_DIAG(COMPILE, "ATTN slot=%d seqK=0 and no KV cache shapes found — deferring to C++ native", si);
              result.valid = false;
              return result;
            }
          }

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
            DSP_DIAG(COMPILE, "TritonIRBuilder::buildSectionedModule: adjusted attention tiles for section [%d-%d] "
                      "to BM=%d BN=%d (headDim=%d, seqQ=%d, seqK=%d, estimatedSmem=%d, limit=%d) "
                      "(hasPastKv=%d, numQHeads=%d, numKvHeads=%d, isBSHD=%d, dualBuffer=%d)",
                      sec.startSlot, sec.endSlot,
                      blockM, blockN, headDim, seqQ, seqK,
                      attnTile.estimatedSharedMemBytes, attnTile.sharedMemLimitBytes,
                      hasPastKv ? 1 : 0, numQHeads, numKvHeads, isBSHD ? 1 : 0,
                      useDualBuffer ? 1 : 0);
            loggedAttnTileAdjust = true;
          }

          auto qPtr = getSlotArgPtr(qSrc);
          auto outPtr = getSlotArgPtr(outSlot);

          // For dual-buffer: kPtr/vPtr = past_key/past_value (BHSD), curKPtr/curVPtr = current key/value (BSHD)
          mlir::Value kPtr, vPtr, curKPtr, curVPtr;
          if (useDualBuffer) {
            // past_key/value are the main K/V buffers (BHSD layout)
            kPtr = getSlotArgPtr(pastKeySrc);
            vPtr = getSlotArgPtr(pastValueSrc);
            // current key/value are the secondary buffers (3D BSHD layout)
            curKPtr = getSlotArgPtr(kSrc);
            curVPtr = getSlotArgPtr(vSrc);
          } else {
            kPtr = getSlotArgPtr(effectiveKSrc);
            vPtr = getSlotArgPtr(effectiveVSrc);
          }

          // Extract attention bias/mask from input[3] if available and non-scalar
          mlir::Value attnBiasPtr;
          std::vector<LongType> attnBiasShape;
          {
            auto fmtShp = [](const std::vector<LongType>& v) -> std::string {
              std::string r; for (size_t i = 0; i < v.size(); i++) { if (i) r += ","; r += std::to_string(v[i]); } return r.empty() ? "empty" : r;
            };
            std::string idxStr;
            for (int i = 0; i < slot.wiring.numInputs; i++) { if (i) idxStr += ","; idxStr += std::to_string(slot.wiring.inputSourceIndices[i]); }
            DSP_DIAG(COMPILE, "TritonIRBuilder: attention slot=%d op=%s numInputs=%d numOutputs=%d "
                      "seqQ=%d seqK=%d heads=%d/%d hd=%d hasPastKv=%d dualBuf=%d "
                      "pastKeySrc=%d qShape=[%s] kShape=[%s] effectiveKShape=[%s] inputSrcs=[%s]",
                      si, slot.ident.opName.c_str(), slot.wiring.numInputs, slot.wiring.numOutputs,
                      seqQ, seqK, numQHeads, numKvHeads, headDim,
                      hasPastKv ? 1 : 0, useDualBuffer ? 1 : 0,
                      pastKeySrc,
                      fmtShp(qShape).c_str(), fmtShp(resolveShape(kSrc)).c_str(),
                      fmtShp(effectiveKShape).c_str(), idxStr.c_str());
          }
          if (slot.wiring.numInputs > 3) {
            int biasSrc = slot.wiring.inputSourceIndices[3];
            auto bShape = resolveShape(biasSrc);
            // Accept bias if rank >= 2 and non-scalar (rank 2 = [B, seqK] padding mask)
            if (bShape.size() >= 2 && shapeLength(bShape) > 1) {
              // Verify the bias buffer's seqK dimension matches the kernel's seqK.
              // The bias shape may be stale from warmup (cached slot output) while
              // seqK was derived from external inputs with current shapes.
              // If biasSeqK < seqK, the kernel would read past the bias buffer → crash.
              int biasSeqKDim = static_cast<int>(bShape[bShape.size() - 1]);
              if (biasSeqKDim >= seqK) {
                attnBiasPtr = getSlotArgPtr(biasSrc);
                attnBiasShape = bShape;
              } else {
                DSP_DIAG(COMPILE, "TritonIRBuilder: skipping attention bias at slot %d — "
                          "bias seqK=%d < kernel seqK=%d (stale shape from warmup)",
                          si, biasSeqKDim, seqK);
              }
            }
          }

          // Validate: the effective K buffer must have enough elements for the derived seqK.
          // When the K buffer is empty (stale from warmup or genuinely empty), the kernel
          // would read from empty buffers causing illegal memory access (CUDA error 700).
          // Always fall back to C++ native execution in this case — even if seqK was derived
          // from external inputs, the actual K/V slot data may still be empty at execution time.
          //
          // For dual-buffer mode (static KV cache), the past_key buffer is pre-allocated
          // to max sequence length and reused across decode steps. In this case, we check
          // the buffer shape/capacity rather than total length, since the buffer may
          // contain stale data from previous steps but is still valid for attention.
          bool kBufferValid = true;
          {
            auto effKShape = resolveShape(effectiveKSrc);
            if (effKShape.empty() && seqK > 0) {
              kBufferValid = false;
              DSP_DIAG(COMPILE, "TritonIRBuilder: skipping FUSED_ATTENTION at slot %d — "
                        "effective K buffer (src=%d) shape is empty but seqK=%d%s. "
                        "Falling back to C++ native.",
                        si, effectiveKSrc, seqK,
                        seqKDerivedFromExternal ? " (seqK derived from external inputs)" : "");
            } else if (!effKShape.empty() && seqK > 0) {
              // For dual-buffer mode (static KV cache), check shape capacity not total length
              if (useDualBuffer) {
                // past_key is 4D BHSD: [B, KvHeads, maxSeqLen, headDim]
                // Valid if rank == 4 and seq dimension (dim 2) > 0
                bool shapeValid = (effKShape.size() == 4 && effKShape[2] > 0);
                if (!shapeValid) {
                  kBufferValid = false;
                  DSP_DIAG(COMPILE, "TritonIRBuilder: skipping FUSED_ATTENTION at slot %d — "
                            "past_key buffer (src=%d) has invalid shape: rank=%zu, seqDim=%d (expected 4D BHSD with seqDim>0)",
                            si, effectiveKSrc, effKShape.size(),
                            effKShape.size() >= 3 ? static_cast<int>(effKShape[2]) : -1);
                }
              } else {
                // Non-dual-buffer: check total length (original behavior)
                LongType effKLen = 1;
                for (auto d : effKShape) effKLen *= d;
                if (effKLen == 0) {
                  kBufferValid = false;
                  DSP_DIAG(COMPILE, "TritonIRBuilder: skipping FUSED_ATTENTION at slot %d — "
                            "effective K buffer (src=%d) is empty but seqK=%d%s. "
                            "Falling back to C++ native.",
                            si, effectiveKSrc, seqK,
                            seqKDerivedFromExternal ? " (seqK derived from external inputs)" : "");
                }
              }
            }
          }

          if (!kBufferValid) {
            DSP_DIAG(COMPILE, "ATTN slot=%d: K buffer invalid, returning section as non-compilable", si);
            result.valid = false;
            return result;
          }

          if (qPtr && kPtr && vPtr && outPtr) {
            emitFusedAttentionKernel(builder, loc, qPtr, kPtr, vPtr, outPtr,
                                     batchSize, numQHeads, numKvHeads, seqQ, seqK, headDim,
                                     scale, blockM, blockN, isBSHD, kIsBSHD,
                                     attnBiasPtr, attnBiasShape,
                                     curKPtr, curVPtr, pastSeqLen, seqKVCur);
            // output[0] = attention result (loaded from output buffer)
            DataType outDtype = resolveDtype(outSlot);
            auto loaded = loadBlock(outSlot, outDtype);
            if (loaded) ssaValues[outSlot] = loaded;

            // output[1] = present_key, output[2] = present_value
            if (useDualBuffer && slot.wiring.numOutputs >= 2) {
              // Dual-buffer: write current K/V to present_key/value output at position pastSeq.
              // Guard: verify the present_key output buffer can hold pastSeqLen + seqKVCur positions.
              // For static KV caches, the past_key buffer is pre-allocated to maxKvLen but the
              // present_key output was allocated during warmup with a much smaller shape.
              // Writing at pastSeqLen offset would be out-of-bounds. In static KV cache mode,
              // the caller handles cache updates — skip the write.
              int presentKeySlot = slot.wiring.outputSlotIndices[1];
              auto pkOutShape = resolveShape(presentKeySlot);
              int pkSeqCapacity = (pkOutShape.size() == 4) ? static_cast<int>(pkOutShape[2]) : 0;
              int requiredSeq = pastSeqLen + seqKVCur;
              bool pkFits = (pkSeqCapacity >= requiredSeq);
              if (!pkFits) {
                DSP_DIAG(COMPILE, "TritonIRBuilder: skipping present_key/value write at slot %d — "
                          "output buffer seqDim=%d < required=%d (pastSeqLen=%d + seqKVCur=%d). "
                          "Static KV cache detected; caller handles cache updates.",
                          si, pkSeqCapacity, requiredSeq, pastSeqLen, seqKVCur);
              }
              if (pkFits) {
                auto presentKeyPtr = getSlotArgPtr(presentKeySlot);
                if (presentKeyPtr && curKPtr) {
                  int totalSeq = pastSeqLen + seqKVCur;
                  emitPresentKvWrite(builder, loc, curKPtr, presentKeyPtr,
                                     batchSize, numQHeads, numKvHeads,
                                     pastSeqLen, seqKVCur, totalSeq, headDim);
                  auto pkLoaded = loadBlock(presentKeySlot, resolveDtype(presentKeySlot));
                  if (pkLoaded) ssaValues[presentKeySlot] = pkLoaded;
                }
              }
              if (slot.wiring.numOutputs >= 3 && pkFits) {
                int presentValSlot = slot.wiring.outputSlotIndices[2];
                auto presentValPtr = getSlotArgPtr(presentValSlot);
                if (presentValPtr && curVPtr) {
                  int totalSeq = pastSeqLen + seqKVCur;
                  emitPresentKvWrite(builder, loc, curVPtr, presentValPtr,
                                     batchSize, numQHeads, numKvHeads,
                                     pastSeqLen, seqKVCur, totalSeq, headDim);
                  auto pvLoaded = loadBlock(presentValSlot, resolveDtype(presentValSlot));
                  if (pvLoaded) ssaValues[presentValSlot] = pvLoaded;
                }
              }
            } else {
              // Non-dual-buffer: pass-through effective key/value SSA
              if (slot.wiring.numOutputs >= 2) {
                if (ssaValues.count(effectiveKSrc)) {
                  ssaValues[slot.wiring.outputSlotIndices[1]] = ssaValues[effectiveKSrc];
                } else {
                  auto kLoaded = loadBlock(effectiveKSrc, resolveDtype(effectiveKSrc));
                  if (kLoaded) ssaValues[slot.wiring.outputSlotIndices[1]] = kLoaded;
                }
              }
              if (slot.wiring.numOutputs >= 3) {
                if (ssaValues.count(effectiveVSrc)) {
                  ssaValues[slot.wiring.outputSlotIndices[2]] = ssaValues[effectiveVSrc];
                } else {
                  auto vLoaded = loadBlock(effectiveVSrc, resolveDtype(effectiveVSrc));
                  if (vLoaded) ssaValues[slot.wiring.outputSlotIndices[2]] = vLoaded;
                }
              }
            }
          } else {
            std::string msg = "TritonIRBuilder::buildSectionedModule: attention at slot " + std::to_string(si) +
                " — missing args."
                " qSrc=" + std::to_string(qSrc) + "(ptr=" + (qPtr ? "OK" : "NULL") + ")" +
                " kSrc=" + std::to_string(kSrc) + " vSrc=" + std::to_string(vSrc) +
                " outSlot=" + std::to_string(outSlot) + "(ptr=" + (outPtr ? "OK" : "NULL") + ")" +
                " kPtr=" + (kPtr ? "OK" : "NULL") + " vPtr=" + (vPtr ? "OK" : "NULL") +
                " hasPastKv=" + std::to_string(hasPastKv) + " dualBuf=" + std::to_string(useDualBuffer) +
                " numArgs=" + std::to_string(result.args.size()) +
                ". Cannot compile.";
            THROW_EXCEPTION(msg.c_str());
          }
        }
        break;
      }

      case KernelSectionType::GATHER:
      case KernelSectionType::GATHER_ND: {
        for (int si = sec.startSlot; si <= sec.endSlot; si++) {
          auto& slot = slots[si];
          if (slot.wiring.numInputs < 1 || slot.wiring.numOutputs < 1) continue;
          int dataSrc = slot.wiring.inputSourceIndices[0];
          int idxSrc = (slot.wiring.numInputs >= 2) ? slot.wiring.inputSourceIndices[1] : dataSrc;
          int outSlot = slot.wiring.outputSlotIndices[0];
          auto dataPtr = getSlotArgPtr(dataSrc);
          auto idxPtr = getSlotArgPtr(idxSrc);
          auto outPtr = getSlotArgPtr(outSlot);
          auto dataShape = resolveShape(dataSrc);
          auto indicesShape = resolveShape(idxSrc);
          auto outShape = resolveShape(outSlot);
          if (dataPtr && idxPtr && outPtr && !dataShape.empty() && !outShape.empty()) {
            int nElements = static_cast<int>(shapeLength(outShape));
            int axis = (slot.args.numIArgs > 0 && slot.args.iArgs) ? static_cast<int>(slot.args.iArgs[0]) : 0;
            bool gatherNd = (sec.type == KernelSectionType::GATHER_ND);
            emitGatherSection(builder, loc, pid, blockSize, dataPtr, idxPtr, outPtr, axis,
                              dataShape, indicesShape, nElements, gatherNd);
            auto loaded = loadBlock(outSlot, resolveDtype(outSlot));
            if (loaded) for (int o = 0; o < slot.wiring.numOutputs; o++) ssaValues[slot.wiring.outputSlotIndices[o]] = loaded;
          }
        }
        break;
      }

      case KernelSectionType::CONCAT: {
        for (int si = sec.startSlot; si <= sec.endSlot; si++) {
          auto& slot = slots[si];
          if (slot.wiring.numInputs < 1 || slot.wiring.numOutputs < 1) continue;
          int outSlot = slot.wiring.outputSlotIndices[0];
          auto outPtr = getSlotArgPtr(outSlot);
          auto outShape = resolveShape(outSlot);
          std::vector<mlir::Value> inPtrs;
          std::vector<std::vector<LongType>> inShapes;
          bool allValid = outPtr && !outShape.empty();
          for (int inp = 0; inp < slot.wiring.numInputs && allValid; inp++) {
            int src = slot.wiring.inputSourceIndices[inp];
            auto ptr = getSlotArgPtr(src);
            auto shape = resolveShape(src);
            if (ptr && !shape.empty()) {
              inPtrs.push_back(ptr);
              inShapes.push_back(shape);
            } else allValid = false;
          }
          if (allValid && !inPtrs.empty()) {
            int nElements = static_cast<int>(shapeLength(outShape));
            int axis = (slot.args.numIArgs > 0 && slot.args.iArgs) ? static_cast<int>(slot.args.iArgs[0]) : 0;
            emitConcatSection(builder, loc, pid, blockSize, inPtrs, outPtr, axis, inShapes, nElements);
            auto loaded = loadBlock(outSlot, resolveDtype(outSlot));
            if (loaded) for (int o = 0; o < slot.wiring.numOutputs; o++) ssaValues[slot.wiring.outputSlotIndices[o]] = loaded;
          }
        }
        break;
      }

      case KernelSectionType::SPLIT:
      case KernelSectionType::SPLIT_V: {
        for (int si = sec.startSlot; si <= sec.endSlot; si++) {
          auto& slot = slots[si];
          if (slot.wiring.numInputs < 1 || slot.wiring.numOutputs < 1) continue;
          int dataSrc = slot.wiring.inputSourceIndices[0];
          auto dataPtr = getSlotArgPtr(dataSrc);
          auto dataShape = resolveShape(dataSrc);
          std::vector<mlir::Value> outPtrs;
          bool allValid = dataPtr && !dataShape.empty();
          for (int o = 0; o < slot.wiring.numOutputs && allValid; o++) {
            int oSlot = slot.wiring.outputSlotIndices[o];
            auto ptr = getSlotArgPtr(oSlot);
            if (ptr) outPtrs.push_back(ptr);
            else allValid = false;
          }
          if (allValid && !outPtrs.empty()) {
            int rank = static_cast<int>(dataShape.size());

            std::string opLower = slot.ident.opName;
            std::transform(opLower.begin(), opLower.end(), opLower.begin(), ::tolower);
            bool isSplitV = (opLower.find("split_v") != std::string::npos ||
                             opLower.find("splitv") != std::string::npos);

            int splitAxis = 0;
            if (isSplitV) {
              // SplitV iArgs: [splitDim, numSplit]
              if (slot.args.numIArgs > 0 && slot.args.iArgs) splitAxis = static_cast<int>(slot.args.iArgs[0]);
            } else {
              // Split iArgs: [numSplit, splitDim] (most common) or [splitDim]
              if (slot.args.numIArgs > 1 && slot.args.iArgs) splitAxis = static_cast<int>(slot.args.iArgs[1]);
              else if (slot.args.numIArgs > 0 && slot.args.iArgs) splitAxis = static_cast<int>(slot.args.iArgs[0]);
            }
            if (splitAxis < 0) splitAxis += rank;
            if (splitAxis < 0 || splitAxis >= rank) splitAxis = 0;

            if (isSplitV && slot.wiring.numInputs >= 2) {
              // SplitV: variable chunk sizes stored in input[1] (a constant int tensor)
              int sizesSrc = slot.wiring.inputSourceIndices[1];
              NDArray* sizesArr = resolveArr(sizesSrc);
              if (sizesArr && !dataShape.empty()) {
                // Build per-output slice with variable axis sizes
                int axisOffset = 0;
                for (int o = 0; o < slot.wiring.numOutputs && o < static_cast<int>(outPtrs.size()); o++) {
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
                emitSplitSection(builder, loc, pid, blockSize, dataPtr, outPtrs, splitAxis, slot.wiring.numOutputs, dataShape, nElements);
              }
            } else {
              // Equal split
              int nElements = static_cast<int>(shapeLength(dataShape));
              emitSplitSection(builder, loc, pid, blockSize, dataPtr, outPtrs, splitAxis, slot.wiring.numOutputs, dataShape, nElements);
            }

            for (int o = 0; o < slot.wiring.numOutputs; o++) {
              int oSlot = slot.wiring.outputSlotIndices[o];
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
          if (slot.wiring.numInputs < 1 || slot.wiring.numOutputs < 1) continue;
          int dataSrc = slot.wiring.inputSourceIndices[0];
          int outSlot = slot.wiring.outputSlotIndices[0];
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
            if (loaded) for (int o = 0; o < slot.wiring.numOutputs; o++) ssaValues[slot.wiring.outputSlotIndices[o]] = loaded;
          }
        }
        break;
      }

      case KernelSectionType::STRIDED_SLICE: {
        for (int si = sec.startSlot; si <= sec.endSlot; si++) {
          auto& slot = slots[si];
          if (slot.wiring.numInputs < 1 || slot.wiring.numOutputs < 1) continue;
          int dataSrc = slot.wiring.inputSourceIndices[0];
          int outSlot = slot.wiring.outputSlotIndices[0];
          auto dataPtr = getSlotArgPtr(dataSrc);
          auto outPtr = getSlotArgPtr(outSlot);
          auto inputShape = resolveShape(dataSrc);
          auto outShape = resolveShape(outSlot);
          if (dataPtr && outPtr && !inputShape.empty() && !outShape.empty()) {
            std::vector<int> begins, ends, strides;
            resolveStridedSliceParams(slot, inputShape, resolveArr, begins, ends, strides);
            int nElements = static_cast<int>(shapeLength(outShape));
            emitSliceSection(builder, loc, pid, blockSize, dataPtr, outPtr, begins, ends, strides, inputShape, nElements);
            auto loaded = loadBlock(outSlot, resolveDtype(outSlot));
            if (loaded) for (int o = 0; o < slot.wiring.numOutputs; o++) ssaValues[slot.wiring.outputSlotIndices[o]] = loaded;
          }
        }
        break;
      }

      case KernelSectionType::SCATTER_ND:
      case KernelSectionType::SCATTER_ND_UPDATE: {
        for (int si = sec.startSlot; si <= sec.endSlot; si++) {
          auto& slot = slots[si];
          if (slot.wiring.numInputs < 3 || slot.wiring.numOutputs < 1) continue;
          int dataSrc = slot.wiring.inputSourceIndices[0];
          int idxSrc = slot.wiring.inputSourceIndices[1];
          int updSrc = slot.wiring.inputSourceIndices[2];
          int outSlot = slot.wiring.outputSlotIndices[0];
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
            if (loaded) for (int o = 0; o < slot.wiring.numOutputs; o++) ssaValues[slot.wiring.outputSlotIndices[o]] = loaded;
          }
        }
        break;
      }

      case KernelSectionType::SHAPE_MANIPULATION: {
        // Permute/transpose section
        for (int si = sec.startSlot; si <= sec.endSlot; si++) {
          auto& slot = slots[si];
          if (slot.wiring.numInputs < 1 || slot.wiring.numOutputs < 1) continue;
          int inputSrc = slot.wiring.inputSourceIndices[0];
          int outSlot = slot.wiring.outputSlotIndices[0];
          auto inPtr = getSlotArgPtr(inputSrc);
          auto outPtr = getSlotArgPtr(outSlot);
          auto inputShape = resolveShape(inputSrc);
          auto outputShape = resolveShape(outSlot);
          if (inPtr && outPtr && !inputShape.empty() && !outputShape.empty()) {
            // Get permutation from iArgs; fall back to reverse if not provided
            std::vector<int> permutation;
            if (slot.args.numIArgs > 0 && slot.args.iArgs) {
              for (int d = 0; d < slot.args.numIArgs; d++)
                permutation.push_back(static_cast<int>(slot.args.iArgs[d]));
            }
            if (permutation.empty()) {
              for (int d = static_cast<int>(inputShape.size()) - 1; d >= 0; d--)
                permutation.push_back(d);
            }
            int nElements = static_cast<int>(shapeLength(outputShape));
            std::string opLower = slot.ident.opName;
            std::transform(opLower.begin(), opLower.end(), opLower.begin(), ::tolower);
            emitShapeManipulationSection(builder, loc, pid, blockSize, inPtr, outPtr, opLower,
                                          inputShape, outputShape, permutation, nElements);
            auto loaded = loadBlock(outSlot, resolveDtype(outSlot));
            if (loaded) for (int o = 0; o < slot.wiring.numOutputs; o++) ssaValues[slot.wiring.outputSlotIndices[o]] = loaded;
          }
        }
        break;
      }

      case KernelSectionType::CONVOLUTION: {
        for (int si = sec.startSlot; si <= sec.endSlot; si++) {
          auto& slot = slots[si];
          if (slot.wiring.numOutputs < 1) continue;

          std::string opLower = slot.ident.opName;
          std::transform(opLower.begin(), opLower.end(), opLower.begin(), ::tolower);

          bool isIm2col = (opLower == "im2col");
          bool isCol2im = (opLower == "col2im");
          bool isIm2colBp = (opLower == "im2col_bp");
          // col2im_bp is not a standard op — col2im has no backprop variant
          // im2col_bp calls col2im internally

          if (isIm2col) {
            // im2col: 1 input (4D image) → 1 output (6D columns)
            // iArgs: [kH, kW, sH, sW, pH, pW, dH, dW, isSameMode]
            if (slot.wiring.numInputs < 1) continue;
            int inputSrc = slot.wiring.inputSourceIndices[0];
            int outSlot = slot.wiring.outputSlotIndices[0];
            auto inPtr = getSlotArgPtr(inputSrc);
            auto outPtr = getSlotArgPtr(outSlot);
            auto inputShape = resolveShape(inputSrc);
            auto outputShape = resolveShape(outSlot);
            if (inPtr && outPtr && !inputShape.empty() && !outputShape.empty()) {
              int kH = (slot.args.numIArgs > 0 && slot.args.iArgs) ? static_cast<int>(slot.args.iArgs[0]) : 1;
              int kW = (slot.args.numIArgs > 1 && slot.args.iArgs) ? static_cast<int>(slot.args.iArgs[1]) : 1;
              int sH = (slot.args.numIArgs > 2 && slot.args.iArgs) ? static_cast<int>(slot.args.iArgs[2]) : 1;
              int sW = (slot.args.numIArgs > 3 && slot.args.iArgs) ? static_cast<int>(slot.args.iArgs[3]) : 1;
              int pH = (slot.args.numIArgs > 4 && slot.args.iArgs) ? static_cast<int>(slot.args.iArgs[4]) : 0;
              int pW = (slot.args.numIArgs > 5 && slot.args.iArgs) ? static_cast<int>(slot.args.iArgs[5]) : 0;
              int dH = (slot.args.numIArgs > 6 && slot.args.iArgs) ? static_cast<int>(slot.args.iArgs[6]) : 1;
              int dW = (slot.args.numIArgs > 7 && slot.args.iArgs) ? static_cast<int>(slot.args.iArgs[7]) : 1;
              int nElements = static_cast<int>(shapeLength(outputShape));
              emitIm2colSection(builder, loc, pid, blockSize, inPtr, outPtr,
                                inputShape, outputShape, kH, kW, sH, sW, pH, pW, dH, dW, nElements);
              auto loaded = loadBlock(outSlot, resolveDtype(outSlot));
              if (loaded) for (int o = 0; o < slot.wiring.numOutputs; o++) ssaValues[slot.wiring.outputSlotIndices[o]] = loaded;
            }
          } else if (isCol2im || isIm2colBp) {
            // col2im: 1 input (6D columns) → 1 output (4D image)
            //   iArgs: [sY, sX, pY, pX, inY, inX, dY, dX, isSameMode]
            // im2col_bp: 2 inputs (4D image, 6D grad) → 1 output (4D grad)
            //   iArgs: [kH, kW, sH, sW, pH, pW, dH, dW, isSameMode]
            //   The 6D grad (input[1]) is the column data, output is the image-space grad
            if (slot.wiring.numInputs < 1) continue;

            // For col2im: input[0] is the 6D column data
            // For im2col_bp: input[1] is the 6D gradient (column data), input[0] is original image
            int colSrc, outSlotIdx;
            if (isCol2im) {
              colSrc = slot.wiring.inputSourceIndices[0];
            } else {
              // im2col_bp: second input is the 6D gradient
              if (slot.wiring.numInputs < 2) continue;
              colSrc = slot.wiring.inputSourceIndices[1];
            }
            outSlotIdx = slot.wiring.outputSlotIndices[0];
            auto colPtr = getSlotArgPtr(colSrc);
            auto outPtr = getSlotArgPtr(outSlotIdx);
            auto colShape = resolveShape(colSrc);
            auto outShape = resolveShape(outSlotIdx);
            if (colPtr && outPtr && !colShape.empty() && !outShape.empty()) {
              int kH, kW, sH, sW, pH, pW, dH, dW;
              if (isCol2im) {
                // col2im iArgs: [sY, sX, pY, pX, inY, inX, dY, dX, isSameMode]
                sH = (slot.args.numIArgs > 0 && slot.args.iArgs) ? static_cast<int>(slot.args.iArgs[0]) : 1;
                sW = (slot.args.numIArgs > 1 && slot.args.iArgs) ? static_cast<int>(slot.args.iArgs[1]) : 1;
                pH = (slot.args.numIArgs > 2 && slot.args.iArgs) ? static_cast<int>(slot.args.iArgs[2]) : 0;
                pW = (slot.args.numIArgs > 3 && slot.args.iArgs) ? static_cast<int>(slot.args.iArgs[3]) : 0;
                dH = (slot.args.numIArgs > 6 && slot.args.iArgs) ? static_cast<int>(slot.args.iArgs[6]) : 1;
                dW = (slot.args.numIArgs > 7 && slot.args.iArgs) ? static_cast<int>(slot.args.iArgs[7]) : 1;
                // kH, kW derived from column shape: col[bS, iC, kH, kW, oH, oW]
                kH = (colShape.size() > 2) ? static_cast<int>(colShape[2]) : 1;
                kW = (colShape.size() > 3) ? static_cast<int>(colShape[3]) : 1;
              } else {
                // im2col_bp iArgs: [kH, kW, sH, sW, pH, pW, dH, dW, isSameMode]
                kH = (slot.args.numIArgs > 0 && slot.args.iArgs) ? static_cast<int>(slot.args.iArgs[0]) : 1;
                kW = (slot.args.numIArgs > 1 && slot.args.iArgs) ? static_cast<int>(slot.args.iArgs[1]) : 1;
                sH = (slot.args.numIArgs > 2 && slot.args.iArgs) ? static_cast<int>(slot.args.iArgs[2]) : 1;
                sW = (slot.args.numIArgs > 3 && slot.args.iArgs) ? static_cast<int>(slot.args.iArgs[3]) : 1;
                pH = (slot.args.numIArgs > 4 && slot.args.iArgs) ? static_cast<int>(slot.args.iArgs[4]) : 0;
                pW = (slot.args.numIArgs > 5 && slot.args.iArgs) ? static_cast<int>(slot.args.iArgs[5]) : 0;
                dH = (slot.args.numIArgs > 6 && slot.args.iArgs) ? static_cast<int>(slot.args.iArgs[6]) : 1;
                dW = (slot.args.numIArgs > 7 && slot.args.iArgs) ? static_cast<int>(slot.args.iArgs[7]) : 1;
              }

              int nElements = static_cast<int>(shapeLength(outShape));
              emitCol2imSection(builder, loc, pid, blockSize, colPtr, outPtr,
                                colShape, outShape, kH, kW, sH, sW, pH, pW, dH, dW, nElements);
              auto loaded = loadBlock(outSlotIdx, resolveDtype(outSlotIdx));
              if (loaded) for (int o = 0; o < slot.wiring.numOutputs; o++) ssaValues[slot.wiring.outputSlotIndices[o]] = loaded;
            }
          } else {
            // conv2d and other convolution ops: 2+ inputs (image + filter)
            if (slot.wiring.numInputs < 2) continue;
            int inputSrc = slot.wiring.inputSourceIndices[0];
            int filterSrc = slot.wiring.inputSourceIndices[1];
            int outSlot = slot.wiring.outputSlotIndices[0];
            auto inPtr = getSlotArgPtr(inputSrc);
            auto filterPtr = getSlotArgPtr(filterSrc);
            auto outPtr = getSlotArgPtr(outSlot);
            auto inputShape = resolveShape(inputSrc);
            auto filterShape = resolveShape(filterSrc);
            auto outputShape = resolveShape(outSlot);
            if (inPtr && filterPtr && outPtr && !inputShape.empty() && !filterShape.empty() && !outputShape.empty()) {
              // Conv2D iArgs: [kH, kW, sH, sW, pH, pW, dH, dW, paddingMode, dataFormat, weightsFormat]
              int strideH = (slot.args.numIArgs > 2 && slot.args.iArgs) ? static_cast<int>(slot.args.iArgs[2]) : 1;
              int strideW = (slot.args.numIArgs > 3 && slot.args.iArgs) ? static_cast<int>(slot.args.iArgs[3]) : 1;
              int padH = (slot.args.numIArgs > 4 && slot.args.iArgs) ? static_cast<int>(slot.args.iArgs[4]) : 0;
              int padW = (slot.args.numIArgs > 5 && slot.args.iArgs) ? static_cast<int>(slot.args.iArgs[5]) : 0;
              int wFormat = (slot.args.numIArgs > 10 && slot.args.iArgs) ? static_cast<int>(slot.args.iArgs[10]) : 0;

              int nElements = static_cast<int>(shapeLength(outputShape));
              emitConvolutionSection(builder, loc, pid, blockSize, inPtr, filterPtr, outPtr,
                                      inputShape, filterShape, outputShape, strideH, strideW, padH, padW, nElements, wFormat);
              auto loaded = loadBlock(outSlot, resolveDtype(outSlot));
              if (loaded) for (int o = 0; o < slot.wiring.numOutputs; o++) ssaValues[slot.wiring.outputSlotIndices[o]] = loaded;
            }
          }
        }
        break;
      }

      case KernelSectionType::STACK: {
        // Stack = unsqueeze + concat
        for (int si = sec.startSlot; si <= sec.endSlot; si++) {
          auto& slot = slots[si];
          if (slot.wiring.numInputs < 1 || slot.wiring.numOutputs < 1) continue;
          int outSlot = slot.wiring.outputSlotIndices[0];
          auto outPtr = getSlotArgPtr(outSlot);
          auto outShape = resolveShape(outSlot);
          std::vector<mlir::Value> inPtrs;
          std::vector<std::vector<LongType>> inShapes;
          bool allValid = outPtr && !outShape.empty();
          for (int inp = 0; inp < slot.wiring.numInputs && allValid; inp++) {
            int src = slot.wiring.inputSourceIndices[inp];
            auto ptr = getSlotArgPtr(src);
            auto shape = resolveShape(src);
            if (ptr && !shape.empty()) {
              inPtrs.push_back(ptr);
              inShapes.push_back(shape);
            } else allValid = false;
          }
          if (allValid && !inPtrs.empty()) {
            int nElements = static_cast<int>(shapeLength(outShape));
            int axis = (slot.args.numIArgs > 0 && slot.args.iArgs) ? static_cast<int>(slot.args.iArgs[0]) : 0;
            // Stack = unsqueeze + concat: insert dim=1 at axis for each input shape
            for (auto& shape : inShapes) {
              shape.insert(shape.begin() + axis, 1);
            }
            emitConcatSection(builder, loc, pid, blockSize, inPtrs, outPtr, axis, inShapes, nElements);
            auto loaded = loadBlock(outSlot, resolveDtype(outSlot));
            if (loaded) for (int o = 0; o < slot.wiring.numOutputs; o++) ssaValues[slot.wiring.outputSlotIndices[o]] = loaded;
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
    auto cat = getOpCategory(slots[i].ident.opName);
    if (cat == TritonOpCategory::MATMUL) {
      matmulSlot = i;

      // Strategy 1: Extract from input arrays A and B (preferred — always available)
      if (slots[i].wiring.numInputs >= 2) {
        NDArray* aArr = resolveArray(slots[i].wiring.inputSourceIndices[0]);
        NDArray* bArr = resolveArray(slots[i].wiring.inputSourceIndices[1]);

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
      if ((matmulM == 0 || matmulN == 0) && slots[i].wiring.numOutputs > 0) {
        int outIdx = slots[i].wiring.outputSlotIndices[0];
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
      if ((matmulM == 0 || matmulN == 0) && slots[i].shapeCacheValid() &&
          !slots[i].shapeCache.cachedOutputShapes.empty()) {
        const LongType* shapeInfo = slots[i].shapeCache.cachedOutputShapes[0];
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
      if (slots[i].wiring.numInputs >= 1) {
        int aSrc = slots[i].wiring.inputSourceIndices[0];
        if (aSrc >= 0 && aSrc < static_cast<int>(totalOutputSlots)) {
          // Find the producing slot's cached output shape for aSrc
          if ((matmulM == 0 || matmulK == 0)) {
            for (int s = 0; s < static_cast<int>(totalSlots); s++) {
              for (int o = 0; o < slots[s].wiring.numOutputs; o++) {
                if (slots[s].wiring.outputSlotIndices[o] == aSrc &&
                    slots[s].shapeCacheValid() && !slots[s].shapeCache.cachedOutputShapes.empty() &&
                    o < static_cast<int>(slots[s].shapeCache.cachedOutputShapes.size())) {
                  const LongType* si = slots[s].shapeCache.cachedOutputShapes[o];
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
              for (int o = 0; o < slots[s].wiring.numOutputs; o++) {
                if (slots[s].wiring.outputSlotIndices[o] == aSrc &&
                    slots[s].shapeCacheValid() && !slots[s].shapeCache.cachedOutputShapes.empty()) {
                  const LongType* shapeInfo = slots[s].shapeCache.cachedOutputShapes[o];
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
    if (matmulSlot >= 0 && slots[matmulSlot].wiring.numInputs >= 2) {
      int aSrc = slots[matmulSlot].wiring.inputSourceIndices[0];
      int bSrc = slots[matmulSlot].wiring.inputSourceIndices[1];
      NDArray* aArr = resolveArray(aSrc);
      NDArray* bArr = resolveArray(bSrc);
      DSP_DIAG(FALLBACK, "TritonIRBuilder::buildMatmulModule: could not extract M/N/K from slot %d "
                "(M=%d, N=%d, K=%d). Input A[src=%d]: %s (rank=%d), Input B[src=%d]: %s (rank=%d)",
                matmulSlot, matmulM, matmulN, matmulK,
                aSrc, aArr ? "present" : "NULL", aArr ? aArr->rankOf() : -1,
                bSrc, bArr ? "present" : "NULL", bArr ? bArr->rankOf() : -1);
    } else {
      DSP_DIAG(FALLBACK, "TritonIRBuilder::buildMatmulModule: could not extract M/N/K from matmul slot %d "
                "(M=%d, N=%d, K=%d)", matmulSlot, matmulM, matmulN, matmulK);
    }
    return result;
  }
  DSP_DIAG(COMPILE, "TritonIRBuilder::buildMatmulModule: extracted M=%d, N=%d, K=%d from slot %d",
            matmulM, matmulN, matmulK, matmulSlot);

  int blockM = 128, blockN = 128, blockK = 32;
  int numWarps = 4, numStages = 3;
  result.numWarps = numWarps;
  result.numStages = numStages;

  // Create MLIR context and register dialects (thread-safe)
  auto mlirContext = createMlirContextWithDialects();

  mlir::OpBuilder builder(mlirContext);
  auto loc = builder.getUnknownLoc();

  // Create module
  auto moduleOp = mlir::ModuleOp::create(loc);
  builder.setInsertionPointToEnd(moduleOp.getBody());

  // ── Collect unique buffer references (same logic as buildModule) ──
  std::unordered_set<int> internalSlotOutputs;
  for (int i = startSlot; i <= endSlot; i++) {
    for (int o = 0; o < slots[i].wiring.numOutputs; o++) {
      internalSlotOutputs.insert(slots[i].wiring.outputSlotIndices[o]);
    }
  }

  // ── Force-external inputs for ROPE ops (same as buildModule) ──
  std::unordered_set<int> forceExternalInputsMatmul;
  for (int i = startSlot; i <= endSlot; i++) {
    auto cat = getOpCategory(slots[i].ident.opName);
    if (cat == TritonOpCategory::ROPE && slots[i].wiring.numInputs >= 3) {
      int cosSrc = slots[i].wiring.inputSourceIndices[1];
      int sinSrc = slots[i].wiring.inputSourceIndices[2];
      if (internalSlotOutputs.count(cosSrc)) forceExternalInputsMatmul.insert(cosSrc);
      if (internalSlotOutputs.count(sinSrc)) forceExternalInputsMatmul.insert(sinSrc);
    }
  }

  // Pre-scan: skip external inputs consumed only by CONST_GEN ops (same as buildModule)
  std::unordered_set<int> constGenOnlyInputs;
  {
    std::unordered_map<int, bool> inputHasNonConstGenConsumer;
    for (int i = startSlot; i <= endSlot; i++) {
      auto cat = getOpCategory(slots[i].ident.opName);
      bool isConstGen = (cat == TritonOpCategory::CONSTANT_GENERATION);
      for (int inp = 0; inp < slots[i].wiring.numInputs; inp++) {
        int srcIdx = slots[i].wiring.inputSourceIndices[inp];
        if (srcIdx >= 0) continue;
        auto it = inputHasNonConstGenConsumer.find(srcIdx);
        if (it == inputHasNonConstGenConsumer.end()) {
          inputHasNonConstGenConsumer[srcIdx] = !isConstGen;
        } else if (!isConstGen) {
          it->second = true;
        }
      }
    }
    for (auto& kv : inputHasNonConstGenConsumer) {
      if (!kv.second) constGenOnlyInputs.insert(kv.first);
    }
  }

  std::vector<TritonKernelArg> inputArgs;
  std::unordered_set<int> seenInputs;
  for (int i = startSlot; i <= endSlot; i++) {
    for (int inp = 0; inp < slots[i].wiring.numInputs; inp++) {
      int srcIdx = slots[i].wiring.inputSourceIndices[inp];
      if (seenInputs.count(srcIdx)) continue;
      seenInputs.insert(srcIdx);
      if (srcIdx < 0) {
        if (constGenOnlyInputs.count(srcIdx)) continue;
        int extIdx = -(srcIdx + 1);
        if (extIdx < numExternalInputs && externalInputs[extIdx]) {
          TritonKernelArg arg;
          arg.slotIndex = srcIdx;
          arg.outputIndex = 0;
          arg.isOutput = false;
          arg.dtype = externalInputs[extIdx]->dataType();
          auto& arr = *externalInputs[extIdx];
          for (int d = 0; d < arr.rankOf(); d++) {
            arg.shape.push_back(arr.sizeAt(d));
            arg.strides.push_back(arr.strideAt(d));
          }
          inputArgs.push_back(arg);
        }
      } else if (!internalSlotOutputs.count(srcIdx) || forceExternalInputsMatmul.count(srcIdx)) {
        bool hasLiveArr = (srcIdx < totalOutputSlots && outputSlots && outputSlots[srcIdx]);
        if (hasLiveArr) {
          TritonKernelArg arg;
          arg.slotIndex = srcIdx;
          arg.outputIndex = 0;
          arg.isOutput = false;
          arg.dtype = outputSlots[srcIdx]->dataType();
          auto& arr = *outputSlots[srcIdx];
          for (int d = 0; d < arr.rankOf(); d++) {
            arg.shape.push_back(arr.sizeAt(d));
            arg.strides.push_back(arr.strideAt(d));
          }
          inputArgs.push_back(arg);
        } else {
          // Live array not available — resolve shape from cached slot metadata,
          // strides from view op parameters
          std::vector<LongType> shape;
          DataType dtype = FLOAT32;
          if (srcIdx >= 0 && srcIdx < totalSlots &&
              slots[srcIdx].shapeCacheValid() && !slots[srcIdx].shapeCache.cachedOutputShapes.empty() &&
              slots[srcIdx].shapeCache.cachedOutputShapes[0] != nullptr) {
            auto* si = slots[srcIdx].shapeCache.cachedOutputShapes[0];
            LongType rank = shape::rank(si);
            for (int d = 0; d < rank; d++) shape.push_back(shape::shapeOf(si)[d]);
            dtype = ArrayOptions::dataType(si);
          }
          if (!shape.empty()) {
            TritonKernelArg arg;
            arg.slotIndex = srcIdx;
            arg.outputIndex = 0;
            arg.isOutput = false;
            arg.dtype = dtype;
            arg.shape = shape;
            if (srcIdx >= 0 && srcIdx < totalSlots) {
              // Build a minimal cachedShapeInfoMap for the helper
              std::unordered_map<int, const LongType*> localCacheMap;
              for (int si2 = 0; si2 < totalSlots; si2++) {
                if (slots[si2].shapeCacheValid() && !slots[si2].shapeCache.cachedOutputShapes.empty()) {
                  for (int o2 = 0; o2 < slots[si2].wiring.numOutputs; o2++) {
                    int oIdx = slots[si2].wiring.outputSlotIndices[o2];
                    if (oIdx >= 0 && o2 < static_cast<int>(slots[si2].shapeCache.cachedOutputShapes.size()) &&
                        slots[si2].shapeCache.cachedOutputShapes[o2] != nullptr) {
                      localCacheMap[oIdx] = slots[si2].shapeCache.cachedOutputShapes[o2];
                    }
                  }
                }
              }
              auto viewStrides = computeViewStridesFromOp(
                  slots, srcIdx, totalSlots, outputSlots, totalOutputSlots, localCacheMap);
              if (!viewStrides.empty()) {
                arg.strides = viewStrides;
              }
            }
            inputArgs.push_back(arg);
          }
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
      for (int o = 0; o < slots[i].wiring.numOutputs; o++) {
        int outIdx = slots[i].wiring.outputSlotIndices[o];
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
          for (int d = 0; d < arr.rankOf(); d++) {
            arg.shape.push_back(arr.sizeAt(d));
            arg.strides.push_back(arr.strideAt(d));
          }
        } else {
          // No live array — resolve from producing op (same logic as buildModule)
          auto producerCat = getOpCategory(slots[i].ident.opName);
          if (producerCat == TritonOpCategory::CAST && slots[i].args.numIArgs > 0 && slots[i].args.iArgs) {
            arg.dtype = static_cast<DataType>(slots[i].args.iArgs[0]);
          } else if (slots[i].wiring.numInputs > 0) {
            int inputSrc = slots[i].wiring.inputSourceIndices[0];
            NDArray* inputArr = resolveArray(inputSrc);
            if (inputArr) arg.dtype = inputArr->dataType();
          }
          if (arg.shape.empty() && slots[i].wiring.numInputs > 0) {
            int inputSrc = slots[i].wiring.inputSourceIndices[0];
            NDArray* inputArr = resolveArray(inputSrc);
            if (inputArr) {
              for (int d = 0; d < inputArr->rankOf(); d++) {
                arg.shape.push_back(inputArr->sizeAt(d));
              }
            }
          }
        }
        outputArgs.push_back(arg);
      }
    }
  }

  result.args.insert(result.args.end(), inputArgs.begin(), inputArgs.end());
  result.args.insert(result.args.end(), outputArgs.begin(), outputArgs.end());

  int totalBufferArgs = static_cast<int>(result.args.size());
  bool useIndirectArgs = (totalBufferArgs + 1) > TRITON_DIRECT_ARG_LIMIT;
  // Force indirect args when CUDA graph capture is enabled (see buildModule comment for rationale)
  if (!useIndirectArgs && sd::Environment::getInstance().tritonGraphCapture()) {
    useIndirectArgs = true;
    DSP_DIAG(COMPILE, "TritonIRBuilder::buildMatmulModule: forcing INDIRECT arg passing for graph capture "
              "compatibility (%d buffer args)", totalBufferArgs);
  }

  DSP_DIAG(COMPILE, "TritonIRBuilder::buildMatmulModule: %d input args, %d output args, %d total%s",
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
  if (matmulOp.wiring.numInputs >= 2) {
    int aSrc = matmulOp.wiring.inputSourceIndices[0];
    int bSrc = matmulOp.wiring.inputSourceIndices[1];
    for (int a = 0; a < static_cast<int>(result.args.size()); a++) {
      if (result.args[a].slotIndex == aSrc && !result.args[a].isOutput) aArgIdx = a;
      if (result.args[a].slotIndex == bSrc && !result.args[a].isOutput) bArgIdx = a;
    }
  }
  if (matmulOp.wiring.numOutputs >= 1) {
    int cSlot = matmulOp.wiring.outputSlotIndices[0];
    for (int a = 0; a < static_cast<int>(result.args.size()); a++) {
      if (result.args[a].slotIndex == cSlot && result.args[a].isOutput) cArgIdx = a;
    }
  }

  if (aArgIdx < 0 || bArgIdx < 0 || cArgIdx < 0) {
    DSP_DIAG(FALLBACK, "TritonIRBuilder::buildMatmulModule: could not map matmul A/B/C to kernel args "
              "(aArgIdx=%d, bArgIdx=%d, cArgIdx=%d)", aArgIdx, bArgIdx, cArgIdx);
    delete mlirContext;
    return result;
  }

  auto aPtr = getBufferArg(aArgIdx);
  auto bPtr = getBufferArg(bArgIdx);
  auto cPtr = getBufferArg(cArgIdx);

  // ── Build epilogue ops from post-matmul elementwise slots ──
  // If slots after the matmul are elementwise ops that consume only the matmul
  // output, they become in-register epilogue ops (true mega-kernel fusion).
  std::vector<EpilogueOp> epilogueOps;
  std::vector<mlir::Value> epiloguePtrs;

  if (matmulSlot >= 0 && matmulSlot < endSlot) {
    int matmulOutSlot = slots[matmulSlot].wiring.outputSlotIndices[0];
    for (int ei = matmulSlot + 1; ei <= endSlot; ei++) {
      auto epiCat = getOpCategory(slots[ei].ident.opName);
      if (epiCat != TritonOpCategory::UNARY_ELEMENTWISE &&
          epiCat != TritonOpCategory::BINARY_ELEMENTWISE) break;

      std::string epiLower = slots[ei].ident.opName;
      std::transform(epiLower.begin(), epiLower.end(), epiLower.begin(), ::tolower);

      if (epiCat == TritonOpCategory::BINARY_ELEMENTWISE && epiLower == "add") {
        // Two cases:
        //   1D bias [N]      → BIAS_ADD epilogue (broadcast across M)
        //   2D residual [M,N] → RESIDUAL_ADD epilogue (element-wise)
        // Detect by checking the rank of the non-matmul operand.
        int biasSrc = -1;
        for (int inp = 0; inp < slots[ei].wiring.numInputs; inp++) {
          if (slots[ei].wiring.inputSourceIndices[inp] != matmulOutSlot) {
            biasSrc = slots[ei].wiring.inputSourceIndices[inp];
            break;
          }
        }
        if (biasSrc >= 0) {
          // Determine bias operand rank (1D vector vs 2D residual)
          int biasRank = -1;
          if (biasSrc < totalOutputSlots && outputSlots && outputSlots[biasSrc]) {
            biasRank = outputSlots[biasSrc]->rankOf();
          }

          // Find bias buffer arg
          for (int a = 0; a < static_cast<int>(result.args.size()); a++) {
            if (result.args[a].slotIndex == biasSrc && !result.args[a].isOutput) {
              EpilogueOp op;
              // 1D vector → BIAS_ADD (broadcast). Anything else (2D residual,
              // unknown rank) → RESIDUAL_ADD (element-wise [M, N] add).
              op.type = (biasRank == 1) ? EpilogueOp::BIAS_ADD : EpilogueOp::RESIDUAL_ADD;
              op.biasArgIndex = static_cast<int>(epiloguePtrs.size());
              epilogueOps.push_back(op);
              epiloguePtrs.push_back(getBufferArg(a));
              break;
            }
          }
        }
        // Update matmulOutSlot to this op's output for chaining
        if (slots[ei].wiring.numOutputs > 0) matmulOutSlot = slots[ei].wiring.outputSlotIndices[0];
      } else if (epiLower == "relu") {
        epilogueOps.push_back({EpilogueOp::RELU, "relu", -1});
        if (slots[ei].wiring.numOutputs > 0) matmulOutSlot = slots[ei].wiring.outputSlotIndices[0];
      } else if (epiLower == "gelu") {
        epilogueOps.push_back({EpilogueOp::GELU, "gelu", -1});
        if (slots[ei].wiring.numOutputs > 0) matmulOutSlot = slots[ei].wiring.outputSlotIndices[0];
      } else if (epiLower == "silu" || epiLower == "swish") {
        epilogueOps.push_back({EpilogueOp::SILU, "silu", -1});
        if (slots[ei].wiring.numOutputs > 0) matmulOutSlot = slots[ei].wiring.outputSlotIndices[0];
      } else if (epiLower == "tanh") {
        epilogueOps.push_back({EpilogueOp::TANH, "tanh", -1});
        if (slots[ei].wiring.numOutputs > 0) matmulOutSlot = slots[ei].wiring.outputSlotIndices[0];
      } else if (epiLower == "sigmoid") {
        epilogueOps.push_back({EpilogueOp::SIGMOID, "sigmoid", -1});
        if (slots[ei].wiring.numOutputs > 0) matmulOutSlot = slots[ei].wiring.outputSlotIndices[0];
      } else {
        break;  // Unknown epilogue op — stop chaining
      }
    }
  }

  if (!epilogueOps.empty()) {
    DSP_DIAG(COMPILE, "TritonIRBuilder::buildMatmulModule: %d in-register epilogue ops",
              static_cast<int>(epilogueOps.size()));
  }

  // Emit the matmul kernel body (2D tiled with K-loop + in-register epilogue)
  emitMatmulKernel(builder, loc, aPtr, bPtr, cPtr,
                    matmulM, matmulN, matmulK, blockM, blockN, blockK,
                    epilogueOps, epiloguePtrs);

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
    DSP_DIAG(JIT, "TritonIRBuilder: built matmul module '%s' M=%d N=%d K=%d, "
              "grid=(%d,%d), %d input args, %d output args",
              result.kernelName.c_str(), matmulM, matmulN, matmulK,
              result.gridX, result.gridY,
              static_cast<int>(inputArgs.size()), static_cast<int>(outputArgs.size()));
  }

  return result;
}

}  // namespace graph
}  // namespace sd

#endif  // HAVE_TRITON

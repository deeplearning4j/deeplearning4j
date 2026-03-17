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
#include <graph/DspDiagnostics.h>
#include <array/ArrayOptions.h>
#include <helpers/logger.h>
#include <helpers/shape.h>
#include <system/Environment.h>
#include <system/common.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
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
      auto cat = getOpCategory(slots[i].opName);
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
  DSP_DIAG(COMPILE, "TritonIRBuilder::buildModule: collected %d categories, selecting tile config...",
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

  // ── Frozen constant slot handling ──
  // Frozen constant slots have outputs that were computed during the SBS warmup
  // and must NOT be recomputed by Triton (their inputs may have changed since
  // warmup, but the frozen value is correct by definition). Instead, frozen
  // slot outputs are treated as INPUTS to the kernel — loaded via tt.load from
  // the GPU buffer that already holds the correct warmup value. The op itself
  // is skipped in the IR emission loop (ssaValues already populated by tt.load).
  std::unordered_set<int> frozenSlotOutputs;
  for (int i = startSlot; i <= endSlot; i++) {
    if (slots[i].frozenConstantSlot) {
      for (int o = 0; o < slots[i].numOutputs; o++) {
        frozenSlotOutputs.insert(slots[i].outputSlotIndices[o]);
      }
    }
  }
  if (!frozenSlotOutputs.empty()) {
    DSP_DIAG(COMPILE, "TritonIRBuilder::buildModule: %d frozen constant outputs in [%d-%d] "
              "will be loaded as inputs (not recomputed)",
              static_cast<int>(frozenSlotOutputs.size()), startSlot, endSlot);
  }

  // Diagnostic: print slot wiring for small sub-kernels (or first sub-kernel in segment)
  if (endSlot - startSlot < 100) {
    for (int i = startSlot; i <= endSlot; i++) {
      std::string inSrcs;
      for (int inp = 0; inp < slots[i].numInputs; inp++) {
        if (!inSrcs.empty()) inSrcs += ",";
        inSrcs += std::to_string(slots[i].inputSourceIndices[inp]);
      }
      std::string outSlots2;
      for (int o = 0; o < slots[i].numOutputs; o++) {
        if (!outSlots2.empty()) outSlots2 += ",";
        outSlots2 += std::to_string(slots[i].outputSlotIndices[o]);
      }
      bool allInternal = true;
      for (int inp = 0; inp < slots[i].numInputs; inp++) {
        int src = slots[i].inputSourceIndices[inp];
        if (src < 0 || !internalSlotOutputs.count(src)) { allInternal = false; break; }
      }
      // Also show resolved array info for non-internal inputs
      std::string inputInfo;
      for (int inp = 0; inp < slots[i].numInputs; inp++) {
        int src = slots[i].inputSourceIndices[inp];
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
                startSlot, endSlot, i, slots[i].opName.c_str(), inSrcs.c_str(),
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
      auto cat = getOpCategory(slots[i].opName);
      bool isConstGen = (cat == TritonOpCategory::CONSTANT_GENERATION);
      for (int inp = 0; inp < slots[i].numInputs; inp++) {
        int srcIdx = slots[i].inputSourceIndices[inp];
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
    for (int inp = 0; inp < slots[i].numInputs; inp++) {
      int srcIdx = slots[i].inputSourceIndices[inp];
      if (seenInputs.count(srcIdx)) continue;
      seenInputs.insert(srcIdx);

      if (srcIdx < 0) {
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
          for (int d = 0; d < arr.rankOf(); d++) arg.shape.push_back(arr.sizeAt(d));
          inputArgs.push_back(arg);
          // Diagnostic: print compile-time values of small external inputs
          if (arr.lengthOf() <= 10) {
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

  // Add frozen constant slot outputs as INPUT args.
  // These slots' output buffers already contain the correct warmup values on GPU.
  // The kernel will tt.load them instead of recomputing, preserving the frozen values.
  for (int outIdx : frozenSlotOutputs) {
    if (seenInputs.count(outIdx)) continue;  // Already added as input (cross-segment ref)
    seenInputs.insert(outIdx);
    auto shape = resolveShapeLocal(outIdx);
    auto dtype = resolveDtypeLocal(outIdx);
    bool hasLiveArr = (outIdx < totalOutputSlots && outputSlots && outputSlots[outIdx]);
    if (hasLiveArr || !shape.empty()) {
      TritonKernelArg arg;
      arg.slotIndex = outIdx;
      arg.outputIndex = 0;
      arg.isOutput = false;
      arg.dtype = dtype;
      arg.shape = shape;
      inputArgs.push_back(arg);
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

  // Build set of input buffer addresses for aliasing detection.
  // When an output slot shares the same GPU buffer as an input slot (e.g., identity
  // cast with in-place allocation), emitting a tt.store for it creates a data race:
  // different warps may execute the store before other warps read the aliased input.
  // Skip such outputs — the data is already correct in the shared buffer.
  std::unordered_set<uintptr_t> inputBufferAddrs;
  for (auto& inArg : inputArgs) {
    NDArray* inArr = nullptr;
    if (inArg.slotIndex < 0) {
      int ei = -(inArg.slotIndex + 1);
      if (ei < numExternalInputs && externalInputs[ei]) inArr = externalInputs[ei];
    } else {
      if (inArg.slotIndex < totalOutputSlots && outputSlots && outputSlots[inArg.slotIndex])
        inArr = outputSlots[inArg.slotIndex];
    }
    if (inArr && inArr->specialBuffer()) {
      inputBufferAddrs.insert(reinterpret_cast<uintptr_t>(inArr->specialBuffer()));
    }
  }

  std::vector<TritonKernelArg> outputArgs;
  {
    std::unordered_set<int> seenOutputSlots;
    int skippedInternal = 0;
    int skippedAliased = 0;
    for (int i = startSlot; i <= endSlot; i++) {
      for (int o = 0; o < slots[i].numOutputs; o++) {
        int outIdx = slots[i].outputSlotIndices[o];
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
        // The kernel loads from the input arg and the output shares the same memory,
        // so storing would either write the same data (identity cast) or create a race
        // (non-identity op where output pointer = input pointer).
        if (outputSlots && outIdx < totalOutputSlots && outputSlots[outIdx] &&
            outputSlots[outIdx]->specialBuffer()) {
          uintptr_t outAddr = reinterpret_cast<uintptr_t>(outputSlots[outIdx]->specialBuffer());
          if (inputBufferAddrs.count(outAddr)) {
            skippedAliased++;
            DSP_DIAG(COMPILE, "TritonIRBuilder: skipping aliased output slot %d (addr=%p matches input buffer) "
                     "in segment [%d-%d]",
                     outIdx, (void*)outAddr, startSlot, endSlot);
            continue;
          }
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
          } else {
            // No live array and no cached shape — resolve dtype/shape from the producing op.
            // This happens when the output slot belongs to a view-capable op (reshape,
            // strided_slice, etc.) that doesn't pre-allocate its output. The producing op
            // (e.g. cast) needs the correct dtype to generate proper store instructions.
            auto producerCat = getOpCategory(slots[i].opName);
            if (producerCat == TritonOpCategory::CAST && slots[i].numIArgs > 0 && slots[i].iArgs) {
              // Cast ops store target dtype in iArgs[0]
              arg.dtype = static_cast<DataType>(slots[i].iArgs[0]);
              DSP_DIAG(COMPILE, "TritonIRBuilder: output slot %d dtype resolved from cast iArgs[0]=%lld → dtype=%d",
                       outIdx, (long long)slots[i].iArgs[0], (int)arg.dtype);
            } else {
              // For non-cast ops, output dtype matches the primary input's dtype
              if (slots[i].numInputs > 0) {
                int inputSrc = slots[i].inputSourceIndices[0];
                arg.dtype = resolveDtypeLocal(inputSrc);
              }
            }
            // Derive shape from the primary input (most ops preserve shape)
            if (arg.shape.empty() && slots[i].numInputs > 0) {
              int inputSrc = slots[i].inputSourceIndices[0];
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
  if (hasReduction || hasNormalization) {
    result.useDynamicGrid = false;
    result.gridX = 1;
    DSP_DIAG(COMPILE, "TritonIRBuilder::buildModule: forced single-block grid for segmented reduction (BLOCK_SIZE=%d)", blockSize);
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
  if (maxOutputElements <= 1) {
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
    if (inputElements > 0 && inputElements < maxOutputElements && !refOutputShape.empty()) {
      // Broadcasting required: input is smaller than output.
      // Determine if simple modular indexing suffices (broadcast on outermost dims only)
      // or if we need stride-based indexing (broadcast on inner dims).

      // Left-pad input shape with 1s to match output rank
      int outRank = static_cast<int>(refOutputShape.size());
      int inRank = static_cast<int>(arg.shape.size());
      std::vector<LongType> paddedInputShape(outRank, 1);
      for (int d = 0; d < inRank && d < outRank; d++) {
        paddedInputShape[outRank - inRank + d] = arg.shape[d];
      }

      // Check if broadcast is only on outermost dimensions (simple case)
      // Simple = all broadcast dims are contiguous from the left
      bool needsStrideBroadcast = false;
      bool seenNonBroadcast = false;
      for (int d = 0; d < outRank; d++) {
        bool isBroadcast = (paddedInputShape[d] == 1 && refOutputShape[d] > 1);
        if (seenNonBroadcast && isBroadcast) {
          needsStrideBroadcast = true;  // Inner-dimension broadcast
          break;
        }
        if (!isBroadcast) seenNonBroadcast = true;
      }

      if (needsStrideBroadcast) {
        // Stride-based broadcast indexing for inner-dimension broadcasting.
        // input_idx = sum((idx / out_stride[d]) % out_dim[d]) * in_stride[d])
        //   where in_stride[d] = 0 for broadcast dims (input_dim[d] == 1)

        // Compute output strides (row-major)
        std::vector<LongType> outStrides(outRank);
        outStrides[outRank - 1] = 1;
        for (int d = outRank - 2; d >= 0; d--) {
          outStrides[d] = outStrides[d + 1] * refOutputShape[d + 1];
        }

        // Compute input strides: product of all input dims below, 0 for broadcast dims
        std::vector<LongType> inStrides(outRank);
        inStrides[outRank - 1] = (paddedInputShape[outRank - 1] > 1) ? 1 : 0;
        for (int d = outRank - 2; d >= 0; d--) {
          if (paddedInputShape[d] <= 1) {
            inStrides[d] = 0;  // broadcast dim
          } else {
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
    for (int o = 0; o < slots[slotIdx].numOutputs; o++) {
      int outIdx = slots[slotIdx].outputSlotIndices[o];
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
                si, slot.opName.c_str(), static_cast<int>(cat),
                slot.numInputs, slot.numOutputs,
                slot.numIArgs, slot.numTArgs, slot.numBArgs,
                slot.isIdentityOp ? 1 : 0, slot.isViewCapableOp ? 1 : 0,
                slot.inPlaceFused ? 1 : 0, slot.needsZeroedOutput ? 1 : 0,
                slot.frozenConstantSlot ? 1 : 0);
      for (int inp = 0; inp < slot.numInputs; inp++) {
        int srcIdx = slot.inputSourceIndices[inp];
        auto srcShape = resolveShapeLocal(srcIdx);
        const char* srcOp = "EXT";
        if (srcIdx >= 0 && srcIdx < totalSlots) srcOp = slots[srcIdx].opName.c_str();
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
      for (int outp = 0; outp < slot.numOutputs; outp++) {
        int outIdx = slot.outputSlotIndices[outp];
        auto outShape = resolveShapeLocal(outIdx);
        DSP_DIAG(COMPILE, "  output[%d] slot=%d shape=[%s]",
                  outp, outIdx, fmtShpJit(outShape).c_str());
      }
      if (slot.numIArgs > 0 && slot.iArgs) {
        std::string iStr;
        for (int a = 0; a < slot.numIArgs && a < 20; a++) {
          if (a) iStr += ","; iStr += std::to_string(slot.iArgs[a]);
        }
        if (slot.numIArgs > 20) iStr += "...";
        DSP_DIAG(COMPILE, "  iArgs=[%s] (%d)", iStr.c_str(), slot.numIArgs);
      }
      if (slot.numTArgs > 0 && slot.tArgs) {
        char tBuf[256] = {0};
        int toff = 0;
        for (int a = 0; a < slot.numTArgs && a < 10 && toff < 240; a++) {
          toff += snprintf(tBuf + toff, sizeof(tBuf) - toff, "%s%.6g", a > 0 ? "," : "", slot.tArgs[a]);
        }
        DSP_DIAG(COMPILE, "  tArgs=[%s] (%d)", tBuf, slot.numTArgs);
      }
    }

    auto it = opTable.find(slot.opName);
    if (it == opTable.end()) continue;
    const auto& mapping = it->second;
    opsEmitted++;

    // Frozen constant slots: their output SSA values were already populated
    // by tt.load from the input args (the GPU buffer holds the correct warmup
    // value). Skip recomputation entirely — downstream ops will use the loaded
    // values via ssaValues[].
    if (slot.frozenConstantSlot) {
      // Verify SSA values are available for all outputs (set by tt.load loop above)
      bool allSet = true;
      for (int o = 0; o < slot.numOutputs; o++) {
        if (ssaValues.find(slot.outputSlotIndices[o]) == ssaValues.end()) {
          allSet = false;
          break;
        }
      }
      if (allSet) continue;
      // If SSA values are missing (frozen output not in inputArgs), fall through
      // to normal computation. This shouldn't happen but provides safety.
      DSP_DIAG_SLOT(FALLBACK, si, "TritonIRBuilder: frozen slot %d has missing SSA values, falling through to compute", si);
    }

    if (cat == TritonOpCategory::BINARY_ELEMENTWISE) {
      // Binary: needs two inputs (except pow which can be unary with scalar exponent in tArgs)
      if (slot.numInputs < 2) {
        // Special case: pow with scalar exponent in tArgs (unary mode)
        std::string opLower2 = slot.opName;
        std::transform(opLower2.begin(), opLower2.end(), opLower2.begin(), ::tolower);
        if (opLower2 == "pow" && slot.numInputs >= 1) {
          int inputSrc = slot.inputSourceIndices[0];
          auto inputIt = ssaValues.find(inputSrc);
          if (inputIt != ssaValues.end()) {
            auto opResult = emitUnaryElementwise(builder, loc, mapping, slot, inputIt->second, blockSize);
            opResult = emulateNativePrecision(opResult, si);
            for (int o = 0; o < slot.numOutputs; o++) {
              ssaValues[slot.outputSlotIndices[o]] = opResult;
            }
          } else {
            DSP_DIAG_SLOT(FALLBACK, si, "TritonIRBuilder: missing SSA value for unary pow at slot %d (src=%d)",
                      si, inputSrc);
          }
          continue;
        }
        DSP_DIAG_SLOT(FALLBACK, si, "TritonIRBuilder: binary op '%s' at slot %d has < 2 inputs",
                  slot.opName.c_str(), si);
        continue;
      }

      int lhsSrc = slot.inputSourceIndices[0];
      int rhsSrc = slot.inputSourceIndices[1];

      auto lhsIt = ssaValues.find(lhsSrc);
      auto rhsIt = ssaValues.find(rhsSrc);

      if (lhsIt == ssaValues.end() || rhsIt == ssaValues.end()) {
        DSP_DIAG_SLOT(FALLBACK, si, "TritonIRBuilder: missing SSA value for binary op '%s' at slot %d "
                  "(lhs=%d:%s, rhs=%d:%s)",
                  slot.opName.c_str(), si,
                  lhsSrc, lhsIt != ssaValues.end() ? "found" : "MISSING",
                  rhsSrc, rhsIt != ssaValues.end() ? "found" : "MISSING");
        continue;
      }

      auto opResult = emitBinaryElementwise(builder, loc, mapping, lhsIt->second, rhsIt->second);
      opResult = emulateNativePrecision(opResult, si);

      // Store result SSA value for each output slot
      for (int o = 0; o < slot.numOutputs; o++) {
        ssaValues[slot.outputSlotIndices[o]] = opResult;
      }

    } else if (cat == TritonOpCategory::UNARY_ELEMENTWISE) {
      // Unary: needs one input
      if (slot.numInputs < 1) {
        DSP_DIAG_SLOT(FALLBACK, si, "TritonIRBuilder: unary op '%s' at slot %d has no inputs",
                  slot.opName.c_str(), si);
        continue;
      }

      int inputSrc = slot.inputSourceIndices[0];
      auto inputIt = ssaValues.find(inputSrc);
      if (inputIt == ssaValues.end()) {
        DSP_DIAG_SLOT(FALLBACK, si, "TritonIRBuilder: missing SSA value for unary op '%s' at slot %d (src=%d)",
                  slot.opName.c_str(), si, inputSrc);
        continue;
      }

      auto opResult = emitUnaryElementwise(builder, loc, mapping, slot, inputIt->second, blockSize);
      opResult = emulateNativePrecision(opResult, si);

      for (int o = 0; o < slot.numOutputs; o++) {
        ssaValues[slot.outputSlotIndices[o]] = opResult;
      }

    } else if (cat == TritonOpCategory::COMPARISON) {
      // Comparison: needs two inputs, produces bool tensor
      if (slot.numInputs < 2) {
        DSP_DIAG_SLOT(FALLBACK, si, "TritonIRBuilder: comparison op '%s' at slot %d has < 2 inputs",
                  slot.opName.c_str(), si);
        continue;
      }
      int lhsSrc = slot.inputSourceIndices[0];
      int rhsSrc = slot.inputSourceIndices[1];
      auto lhsIt = ssaValues.find(lhsSrc);
      auto rhsIt = ssaValues.find(rhsSrc);
      if (lhsIt == ssaValues.end() || rhsIt == ssaValues.end()) {
        DSP_DIAG_SLOT(FALLBACK, si, "TritonIRBuilder: missing SSA value for comparison op '%s' at slot %d",
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
        DSP_DIAG_SLOT(FALLBACK, si, "TritonIRBuilder: logical op '%s' at slot %d has no inputs",
                  slot.opName.c_str(), si);
        continue;
      }
      int lhsSrc = slot.inputSourceIndices[0];
      auto lhsIt = ssaValues.find(lhsSrc);
      if (lhsIt == ssaValues.end()) {
        DSP_DIAG_SLOT(FALLBACK, si, "TritonIRBuilder: missing SSA value for logical op '%s' at slot %d",
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
        DSP_DIAG_SLOT(FALLBACK, si, "TritonIRBuilder: ternary op '%s' at slot %d has < 3 inputs",
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
        DSP_DIAG_SLOT(FALLBACK, si, "TritonIRBuilder: missing SSA value for ternary op '%s' at slot %d",
                  slot.opName.c_str(), si);
        continue;
      }
      auto opResult = emitTernaryOp(builder, loc, condIt->second, trueIt->second, falseIt->second, blockSize);
      opResult = emulateNativePrecision(opResult, si);
      for (int o = 0; o < slot.numOutputs; o++) {
        ssaValues[slot.outputSlotIndices[o]] = opResult;
      }

    } else if (cat == TritonOpCategory::IDENTITY) {
      // Identity/assign: SSA value forwarding, no IR op needed
      if (slot.numInputs < 1) {
        DSP_DIAG_SLOT(FALLBACK, si, "TritonIRBuilder: identity op '%s' at slot %d has no inputs",
                  slot.opName.c_str(), si);
        continue;
      }
      // For assign(target, source): output = source = input[1]
      // For identity(x): output = x = input[0]
      int inputIdx = (slot.numInputs >= 2) ? 1 : 0;
      int inputSrc = slot.inputSourceIndices[inputIdx];
      auto inputIt = ssaValues.find(inputSrc);
      if (inputIt == ssaValues.end()) {
        DSP_DIAG_SLOT(FALLBACK, si, "TritonIRBuilder: missing SSA value for identity op '%s' at slot %d",
                  slot.opName.c_str(), si);
        continue;
      }
      // Forward the SSA value directly — no computation needed
      DSP_DIAG_SLOT(COMPILE, si, "TritonIRBuilder: IDENTITY op '%s' at slot %d: numInputs=%d inputIdx=%d inputSrc=%d → forwarded to %d output(s)",
                slot.opName.c_str(), si, slot.numInputs, inputIdx, inputSrc, slot.numOutputs);
      for (int o = 0; o < slot.numOutputs; o++) {
        ssaValues[slot.outputSlotIndices[o]] = inputIt->second;
      }

    } else if (cat == TritonOpCategory::CAST) {
      // Cast: type conversion using the castTo() helper
      if (slot.numInputs < 1) {
        DSP_DIAG_SLOT(FALLBACK, si, "TritonIRBuilder: cast op '%s' at slot %d has no inputs",
                  slot.opName.c_str(), si);
        continue;
      }
      int inputSrc = slot.inputSourceIndices[0];
      auto inputIt = ssaValues.find(inputSrc);
      if (inputIt == ssaValues.end()) {
        DSP_DIAG_SLOT(FALLBACK, si, "TritonIRBuilder: missing SSA value for cast op '%s' at slot %d",
                  slot.opName.c_str(), si);
        continue;
      }
      // Determine target type from the output slot's dtype.
      // Priority: dArgs[0] > iArgs[0] > output slot dtype.
      // Cast ops in libnd4j store the target dtype in iArgs[0] as an integer.
      // dArgs (extraTypes) may or may not be populated depending on the model format.
      // resolveDtypeLocal is the last resort but can return FLOAT32 (wrong) when
      // the output slot belongs to a view-capable op with no pre-allocated array.
      DataType targetDtype = FLOAT32;  // default
      if (slot.numDArgs > 0 && slot.dArgs) {
        targetDtype = slot.dArgs[0];
      } else if (slot.numIArgs > 0 && slot.iArgs) {
        // Cast ops store the target dtype in iArgs[0]
        targetDtype = static_cast<DataType>(slot.iArgs[0]);
      } else if (slot.numOutputs > 0) {
        int outIdx = slot.outputSlotIndices[0];
        targetDtype = resolveDtypeLocal(outIdx);
      }
      DSP_DIAG_SLOT(COMPILE, si, "TritonIRBuilder: cast at slot %d: numDArgs=%d numIArgs=%d targetDtype=%d (%s)",
                si, slot.numDArgs, slot.numIArgs, (int)targetDtype,
                DataTypeUtils::asString(targetDtype).c_str());
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
        DSP_DIAG_SLOT(FALLBACK, si, "TritonIRBuilder: reduction op '%s' at slot %d has no inputs",
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
        DSP_DIAG_SLOT(SHAPE, si, "TritonIRBuilder: reduction op '%s' has no input shape info", slot.opName.c_str());
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
      for (int o = 0; o < slot.numOutputs; o++) {
        ssaValues[slot.outputSlotIndices[o]] = opResult;
      }

    } else if (cat == TritonOpCategory::NORMALIZATION) {
      // Normalization: load input from SSA, call emitNormalizationOp, store result
      if (slot.numInputs < 1) {
        DSP_DIAG_SLOT(FALLBACK, si, "TritonIRBuilder: normalization op '%s' at slot %d has no inputs",
                  slot.opName.c_str(), si);
        continue;
      }
      int inputSrc = slot.inputSourceIndices[0];
      auto inputIt = ssaValues.find(inputSrc);
      if (inputIt == ssaValues.end()) {
        DSP_DIAG_SLOT(FALLBACK, si, "TritonIRBuilder: missing SSA value for normalization op '%s' at slot %d",
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
      std::string normKey = normalizeOpToken(slot.opName);
      auto getNormInput = [&](int inputPos) -> mlir::Value {
        if (inputPos >= slot.numInputs) return mlir::Value();
        int src = slot.inputSourceIndices[inputPos];
        auto it = ssaValues.find(src);
        return (it != ssaValues.end()) ? it->second : mlir::Value();
      };

      mlir::Value scaleVal, biasVal, meanVal, varianceVal;
      if (normKey == "batchnorm") {
        meanVal = getNormInput(1);
        varianceVal = getNormInput(2);
        scaleVal = getNormInput(3);
        biasVal = getNormInput(4);
      } else {
        scaleVal = getNormInput(1);
        biasVal = getNormInput(2);
      }

      // Read epsilon from tArgs (first float argument), default 1e-5 if not set
      float epsilon = (slot.numTArgs > 0 && slot.tArgs) ? static_cast<float>(slot.tArgs[0]) : 1e-5f;

      auto opResult = emitNormalizationOp(builder, loc, slot.opName, inputIt->second, axis, outputType,
                                          scaleVal, biasVal, meanVal, varianceVal, epsilon);
      // Safety: if normalization somehow returns a scalar, splat it back to tensor
      if (!mlir::isa<mlir::RankedTensorType>(opResult.getType())) {
        auto splatElemType = opResult.getType();
        auto splatTensorType = mlir::RankedTensorType::get({blockSize}, splatElemType);
        opResult = builder.create<mlir::triton::SplatOp>(loc, splatTensorType, opResult);
      }
      opResult = emulateNativePrecision(opResult, si);
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
        // dot_product_attention_v2 input order: (Q=0, V=1, K=2)
        // onnx_multi_head_attention input order: (Q=0, K=1, V=2)
        // Detect op name to swap K/V source indices for DPA v2.
        std::string opLowerKV = slot.opName;
        std::transform(opLowerKV.begin(), opLowerKV.end(), opLowerKV.begin(), ::tolower);
        bool isDpaV2 = (opLowerKV.find("dot_product_attention") != std::string::npos);

        int kSrc = isDpaV2 ? slot.inputSourceIndices[2] : slot.inputSourceIndices[1];
        int vSrc = isDpaV2 ? slot.inputSourceIndices[1] : slot.inputSourceIndices[2];
        int outSlot = slot.outputSlotIndices[0];

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
          numQHeads = (slot.numIArgs > 0 && slot.iArgs) ? static_cast<int>(slot.iArgs[0]) : 1;
          if (numQHeads <= 0) numQHeads = 1;
          headDim = hidden / numQHeads;
          qIsBSHD = true;
          DSP_DIAG(JIT, "TritonIRBuilder: fused attention '%s' at slot %d has 3D Q [%lld,%lld,%lld] "
                    "— compound op, numQHeads=%d (from iArgs[0]), headDim=%d, using dual-buffer mode",
                    slot.opName, si,
                    (long long)qArr->sizeAt(0), (long long)qArr->sizeAt(1),
                    (long long)qArr->sizeAt(2), numQHeads, headDim);
        }

        // Detect past_key/past_value by scanning ALL inputs for 4D KV-cache-like shapes.
        // A past_key tensor is 4D BHSD: [batch, kvHeads, seqK, headDim] where headDim
        // matches Q's headDim. This distinguishes it from attention masks [B,H,S,S].
        bool hasPastKv = false;
        int pastKeySrc = -1, pastValueSrc = -1;

        for (int inp = 3; inp < slot.numInputs && !hasPastKv; inp++) {
          int candidateSrc = slot.inputSourceIndices[inp];
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
              if (inp + 1 < slot.numInputs) {
                int pvCandidate = slot.inputSourceIndices[inp + 1];
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
                    si, headDim, slot.numInputs);
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
          // 3D key: [B, seqK, kvHeads*headDim] — infer from iArgs
          if (slot.numIArgs > 0 && slot.iArgs) {
            int totalQHeads = static_cast<int>(slot.iArgs[0]);
            if (totalQHeads > 0) numKvHeads = totalQHeads;  // will be refined below
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
            for (int s = startSlot; s <= endSlot; s++) {
              for (int o = 0; o < slots[s].numOutputs; o++) {
                if (slots[s].outputSlotIndices[o] == kSrc) {
                  for (int pi = 0; pi < slots[s].numInputs; pi++) {
                    int psrc = slots[s].inputSourceIndices[pi];
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
                  goto kProducerSearchDoneJit;
                }
              }
            }
          }
          kProducerSearchDoneJit:

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
        if (slot.numInputs > 3) {
          int biasSrc = slot.inputSourceIndices[3];
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
        bool kBufferValidJit = true;
        {
          NDArray* effKArr = resolveArr(effectiveKSrc);
          if ((!effKArr || effKArr->isEmpty() || effKArr->lengthOf() == 0) && seqK > 0) {
            kBufferValidJit = false;
            DSP_DIAG(JIT, "TritonIRBuilder: skipping FUSED_ATTENTION at slot %d (JIT path) — "
                      "effective K buffer (src=%d) is empty but seqK=%d%s",
                      si, effectiveKSrc, seqK,
                      seqKDerivedFromExternalJit ? " (seqK derived from external inputs)" : "");
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
          if (useDualBuffer && slot.numOutputs >= 2) {
            // Dual-buffer: present_key/value output may need write of current K/V
            // at pastSeqLen offset. Check if the output buffer can hold it.
            int presentKeySlot = slot.outputSlotIndices[1];
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
            if (slot.numOutputs >= 3 && pkFits) {
              int presentValSlot = slot.outputSlotIndices[2];
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
          DSP_DIAG_SLOT(FALLBACK, si, "TritonIRBuilder: missing SSA value for shape op '%s' at slot %d (src=%d)",
                    slot.opName.c_str(), si, inputSrc);
        }
      }

    } else if (cat == TritonOpCategory::ROPE) {
      // ─── ROPE: paired elementwise rotation with precomputed cos/sin ───
      if (slot.numInputs >= 3 && slot.numOutputs >= 1) {
        int inputSrc = slot.inputSourceIndices[0];
        int cosSrc = slot.inputSourceIndices[1];
        int sinSrc = slot.inputSourceIndices[2];
        int outSlot = slot.outputSlotIndices[0];

        auto inPtr = getSlotArgPtr(inputSrc);
        auto cosArgPtr = getSlotArgPtr(cosSrc);
        auto sinArgPtr = getSlotArgPtr(sinSrc);
        auto outPtr = getSlotArgPtr(outSlot);
        NDArray* inArr = resolveArr(inputSrc);
        NDArray* cosArr = resolveArr(cosSrc);
        NDArray* outArr = resolveArr(outSlot);

        if (inPtr && cosArgPtr && sinArgPtr && outPtr && inArr && cosArr && outArr) {
          std::vector<LongType> inShape, cosShapeVec;
          for (int d = 0; d < inArr->rankOf(); d++) inShape.push_back(inArr->sizeAt(d));
          for (int d = 0; d < cosArr->rankOf(); d++) cosShapeVec.push_back(cosArr->sizeAt(d));
          int nElements = static_cast<int>(outArr->lengthOf());
          int ropeType = (slot.numIArgs > 0 && slot.iArgs) ? static_cast<int>(slot.iArgs[0]) : 0;

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
            for (int o = 0; o < slot.numOutputs; o++)
              ssaValues[slot.outputSlotIndices[o]] = result;
          } else {
            // Fallback: pointer-based emitter (flush SSA → global memory → reload)
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

            auto loaded = loadBackFromBuffer(outSlot, outArr->dataType());
            if (loaded) {
              loaded = emulateNativePrecision(loaded, si);
              for (int o = 0; o < slot.numOutputs; o++) ssaValues[slot.outputSlotIndices[o]] = loaded;
            }
          }
        } else {
          std::string msg = "TritonIRBuilder: ROPE '" + slot.opName + "' at slot " + std::to_string(si) +
              " — missing kernel arg ptrs/arrays. Cannot compile.";
          THROW_EXCEPTION(msg.c_str());
        }
      } else if (slot.numInputs == 1 && slot.numOutputs >= 1) {
        // Non-cached ROPE (computes sin/cos internally) — pass through as identity for now
        int inputSrc = slot.inputSourceIndices[0];
        auto inputIt = ssaValues.find(inputSrc);
        if (inputIt != ssaValues.end()) {
          for (int o = 0; o < slot.numOutputs; o++) ssaValues[slot.outputSlotIndices[o]] = inputIt->second;
        }
      } else {
        DSP_DIAG_SLOT(FALLBACK, si, "TritonIRBuilder: ROPE '%s' at slot %d — insufficient inputs(%d)",
                  slot.opName.c_str(), si, slot.numInputs);
      }

    } else if (cat == TritonOpCategory::DATA_MOVEMENT) {
      // ─── DATA MOVEMENT: dispatch to appropriate section emitter ───
      std::string opLower = slot.opName;
      std::transform(opLower.begin(), opLower.end(), opLower.begin(), ::tolower);
      std::string opKey = normalizeOpToken(slot.opName);

      if (slot.numInputs < 1 || slot.numOutputs < 1) {
        DSP_DIAG_SLOT(FALLBACK, si, "TritonIRBuilder: data movement '%s' at slot %d — insufficient inputs(%d)/outputs(%d)",
                  slot.opName.c_str(), si, slot.numInputs, slot.numOutputs);
      } else if (opKey == "gather" || opKey == "gathernd") {
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
          bool gatherNd = (opKey == "gathernd");

          emitGatherSection(builder, loc, pid, blockSize,
                            dataPtr, idxPtr, outPtr, axis,
                            dataShape, indicesShape, nElements, gatherNd);

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
          int rank = static_cast<int>(dataShape.size());

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
            // SplitV: variable chunk sizes from input[1]
            int sizesSrc = slot.inputSourceIndices[1];
            NDArray* sizesArr = resolveArr(sizesSrc);
            if (sizesArr && !dataShape.empty()) {
              int axisOffset = 0;
              for (int o2 = 0; o2 < slot.numOutputs && o2 < static_cast<int>(outPtrs.size()); o2++) {
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
          std::vector<int> begins, ends, strides;
          resolveStridedSliceParams(slot, inputShape, resolveArr, begins, ends, strides);
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
            DataType outDtype = resolveDtypeLocal(outSlot);
            auto result = loadBackFromBuffer(outSlot, outDtype);
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
  // Use per-output masks when outputs have different element counts to prevent
  // buffer overflows. In a fused kernel covering multiple independent chains,
  // different outputs can have different sizes (e.g., main hidden state [1,960]
  // vs RoPE frequencies [1,480]). Without per-output masks, the global n_elements
  // from the largest output would allow writes past smaller buffers.
  int outputArgBase = static_cast<int>(inputArgs.size());
  for (int a = 0; a < static_cast<int>(outputArgs.size()); a++) {
    auto& arg = outputArgs[a];
    auto funcArg = getBufferArg(outputArgBase + a);

    auto ssaIt = ssaValues.find(arg.slotIndex);
    if (ssaIt == ssaValues.end()) {
      DSP_DIAG_SLOT(FALLBACK, arg.slotIndex, "TritonIRBuilder: no SSA value for output slot %d — skipping store",
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
      // DATA_MOVEMENT ops use non-contiguous access patterns (indexed, strided,
      // cascading-select). When consuming cross-section intermediates, thread N
      // may read data written by thread M in a prior section, so a global barrier
      // is required. Without this, gather reads partially-written data → corruption.
      case KernelSectionType::GATHER:
      case KernelSectionType::GATHER_ND:
      case KernelSectionType::CONCAT:
      case KernelSectionType::SPLIT:
      case KernelSectionType::SPLIT_V:
      case KernelSectionType::STACK:
      case KernelSectionType::TILE:
      case KernelSectionType::STRIDED_SLICE:
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

  // Build set of input buffer addresses for aliasing detection (same as buildModule)
  std::unordered_set<uintptr_t> inputBufferAddrsSectioned;
  for (auto& inArg : inputArgs) {
    NDArray* inArr = nullptr;
    if (inArg.slotIndex < 0) {
      int ei = -(inArg.slotIndex + 1);
      if (ei < numExternalInputs && externalInputs[ei]) inArr = externalInputs[ei];
    } else {
      if (inArg.slotIndex < totalOutputSlots && outputSlots && outputSlots[inArg.slotIndex])
        inArr = outputSlots[inArg.slotIndex];
    }
    if (inArr && inArr->specialBuffer()) {
      inputBufferAddrsSectioned.insert(reinterpret_cast<uintptr_t>(inArr->specialBuffer()));
    }
  }

  std::vector<TritonKernelArg> outputArgs;
  {
    std::unordered_set<int> seenOutputSlots;
    int skippedAliased = 0;
    for (int i = startSlot; i <= endSlot; i++) {
      for (int o = 0; o < slots[i].numOutputs; o++) {
        int outIdx = slots[i].outputSlotIndices[o];
        if (outIdx < 0 || outIdx >= totalOutputSlots) continue;
        if (seenOutputSlots.count(outIdx)) continue;
        seenOutputSlots.insert(outIdx);
        if (!externalOutputs.count(outIdx)) continue;

        // Skip outputs whose GPU buffer aliases an input buffer (same as buildModule)
        if (outputSlots && outIdx < totalOutputSlots && outputSlots[outIdx] &&
            outputSlots[outIdx]->specialBuffer()) {
          uintptr_t outAddr = reinterpret_cast<uintptr_t>(outputSlots[outIdx]->specialBuffer());
          if (inputBufferAddrsSectioned.count(outAddr)) {
            skippedAliased++;
            DSP_DIAG(COMPILE, "TritonIRBuilder::buildSectionedModule: skipping aliased output slot %d "
                     "(addr=%p matches input buffer) in segment [%d-%d]",
                     outIdx, (void*)outAddr, startSlot, endSlot);
            continue;
          }
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
          } else {
            // No live array and no cached shape — resolve from producing op
            auto producerCat = getOpCategory(slots[i].opName);
            if (producerCat == TritonOpCategory::CAST && slots[i].numIArgs > 0 && slots[i].iArgs) {
              arg.dtype = static_cast<DataType>(slots[i].iArgs[0]);
            } else {
              if (slots[i].numInputs > 0) {
                int inputSrc = slots[i].inputSourceIndices[0];
                arg.dtype = resolveDtype(inputSrc);
              }
            }
            if (arg.shape.empty() && slots[i].numInputs > 0) {
              int inputSrc = slots[i].inputSourceIndices[0];
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
        // Detect DPA v2 (input order Q,V,K) vs standard (Q,K,V)
        std::string opLowerGrid = slot.opName;
        std::transform(opLowerGrid.begin(), opLowerGrid.end(), opLowerGrid.begin(), ::tolower);
        bool isDpaV2Grid = (opLowerGrid.find("dot_product_attention") != std::string::npos);
        bool opUsesBSHDGrid = isDpaV2Grid;
        // For DPA v2: input[1]=V, input[2]=K; resolve K for seqK
        int kInputIdx = isDpaV2Grid ? 2 : 1;

        auto qShape = resolveShape(slot.inputSourceIndices[0]);
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
          if (slot.numInputs > kInputIdx) {
            auto kShape = resolveShape(slot.inputSourceIndices[kInputIdx]);
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
          numHeads = (slot.numIArgs > 0 && slot.iArgs) ? static_cast<int>(slot.iArgs[0]) : 1;
          if (numHeads <= 0) numHeads = 1;
          headDim = hidden / numHeads;

          // Check for past_key (input 4) — 4D BHSD [B,H,pastSeq,D]
          bool hasPastKvGrid = false;
          if (slot.numInputs > 4) {
            auto pastKeyShape = resolveShape(slot.inputSourceIndices[4]);
            if (pastKeyShape.size() == 4 && pastKeyShape[0] > 0 && pastKeyShape[2] > 0) {
              hasPastKvGrid = true;
              int pastSeq = static_cast<int>(pastKeyShape[2]);
              // current K is input[1], 3D [B, seqKV, H*D]
              int seqKV = 1;
              if (slot.numInputs > kInputIdx) {
                auto curKShape = resolveShape(slot.inputSourceIndices[kInputIdx]);
                if (curKShape.size() == 3) {
                  seqKV = static_cast<int>(std::max<LongType>(1, curKShape[1]));
                }
              }
              seqK = pastSeq + seqKV;  // total sequence for attention
            }
          }
          if (!hasPastKvGrid && slot.numInputs > kInputIdx) {
            auto kShape = resolveShape(slot.inputSourceIndices[kInputIdx]);
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
      DSP_DIAG(COMPILE, "TritonIRBuilder::buildSectionedModule: tuned cooperative block size %d -> %d "
                "(targetBlocks=%d, resultingGrid=%d)",
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
                  si, slot.opName.c_str(), slot.numInputs, slot.numOutputs,
                  slot.numIArgs, slot.numTArgs, slot.numBArgs,
                  slot.isIdentityOp ? 1 : 0, slot.isViewCapableOp ? 1 : 0,
                  slot.inPlaceFused ? 1 : 0, slot.needsZeroedOutput ? 1 : 0);

        // Every input for this slot
        for (int inp = 0; inp < slot.numInputs; inp++) {
          int srcIdx = slot.inputSourceIndices[inp];
          auto srcShape = resolveShape(srcIdx);
          const char* srcOp = "EXT";
          if (srcIdx >= 0 && srcIdx < totalSlots) srcOp = slots[srcIdx].opName.c_str();
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
        for (int outp = 0; outp < slot.numOutputs; outp++) {
          int outIdx = slot.outputSlotIndices[outp];
          auto outShape = resolveShape(outIdx);
          bool outExists = (outIdx >= 0 && outIdx < totalOutputSlots && outputSlots && outputSlots[outIdx]);
          LongType outLen = outExists ? outputSlots[outIdx]->lengthOf() : -1;
          DSP_DIAG(COMPILE, "    output[%d] slot=%d shape=[%s] exists=%d len=%lld",
                    outp, outIdx, fmtShapeVec(outShape).c_str(),
                    outExists, (long long)outLen);
        }

        // iArgs
        if (slot.numIArgs > 0 && slot.iArgs) {
          std::string iStr;
          for (int a = 0; a < slot.numIArgs && a < 20; a++) {
            if (a) iStr += ",";
            iStr += std::to_string(slot.iArgs[a]);
          }
          if (slot.numIArgs > 20) iStr += "...";
          DSP_DIAG(COMPILE, "    iArgs=[%s] (%d total)", iStr.c_str(), slot.numIArgs);
        }

        // tArgs
        if (slot.numTArgs > 0 && slot.tArgs) {
          char tBuf[512] = {0};
          int toff = 0;
          for (int a = 0; a < slot.numTArgs && a < 10 && toff < 480; a++) {
            toff += snprintf(tBuf + toff, sizeof(tBuf) - toff, "%s%.6g", a > 0 ? "," : "", slot.tArgs[a]);
          }
          if (slot.numTArgs > 10) strncat(tBuf, "...", sizeof(tBuf) - strlen(tBuf) - 1);
          DSP_DIAG(COMPILE, "    tArgs=[%s] (%d total)", tBuf, slot.numTArgs);
        }

        // bArgs
        if (slot.numBArgs > 0 && slot.bArgs) {
          std::string bStr;
          for (int a = 0; a < slot.numBArgs && a < 10; a++) {
            if (a) bStr += ",";
            bStr += slot.bArgs[a] ? "true" : "false";
          }
          DSP_DIAG(COMPILE, "    bArgs=[%s] (%d total)", bStr.c_str(), slot.numBArgs);
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
            if (slot.numInputs < 2) {
              // Unary pow fallback: pow with scalar exponent in tArgs
              std::string opLower2 = slot.opName;
              std::transform(opLower2.begin(), opLower2.end(), opLower2.begin(), ::tolower);
              if (opLower2 == "pow" && slot.numInputs >= 1) {
                auto inputIt = ssaValues.find(slot.inputSourceIndices[0]);
                if (inputIt != ssaValues.end()) {
                  auto opResult = emitUnaryElementwise(builder, loc, mapping, slot, inputIt->second, blockSize);
                  for (int o = 0; o < slot.numOutputs; o++) ssaValues[slot.outputSlotIndices[o]] = opResult;
                }
              }
              continue;
            }
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
            } else if (slot.numIArgs > 0 && slot.iArgs) {
              // Cast ops store target dtype in iArgs[0]
              targetDtype = static_cast<DataType>(slot.iArgs[0]);
            } else if (slot.numOutputs > 0) {
              int outIdx = slot.outputSlotIndices[0];
              targetDtype = resolveDtype(outIdx);
            }
            auto targetElemType = getMLIRType(builder, targetDtype);
            auto opResult = castTo(builder, loc, inputIt->second, targetElemType);
            for (int o = 0; o < slot.numOutputs; o++) ssaValues[slot.outputSlotIndices[o]] = opResult;
          } else if (cat == TritonOpCategory::REDUCTION) {
            // Segmented reduction: same approach as buildModule (lines 4159-4449).
            // Cannot use emitReductionOp/tt.reduce because sectioned module uses flat 1D
            // tensors — tt.reduce would reduce the entire tensor to a scalar, but we need
            // partial reduction along a specific axis of the original multi-dimensional shape.
            if (slot.numInputs < 1) continue;
            int inputSrc = slot.inputSourceIndices[0];

            // Get reduction axis from iArgs
            int reductionAxis = (slot.numIArgs > 0 && slot.iArgs) ? static_cast<int>(slot.iArgs[0]) : -1;

            // Resolve original input shape (multi-dimensional)
            auto inputShape = resolveShape(inputSrc);
            int inputRank = static_cast<int>(inputShape.size());
            if (inputRank == 0) {
              DSP_DIAG_SLOT(SHAPE, si, "TritonIRBuilder::buildSectionedModule: reduction op '%s' at slot %d has no input shape",
                        slot.opName.c_str(), si);
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
            auto outSlotIdx = slot.outputSlotIndices[0];
            auto outDtype = resolveDtype(outSlotIdx);
            auto outElemType = getMLIRType(builder, outDtype);
            mlir::Value opResult = castTo(builder, loc, acc, outElemType);
            if (!mlir::isa<mlir::RankedTensorType>(opResult.getType())) {
              auto splatTy = mlir::RankedTensorType::get({blockSize}, opResult.getType());
              opResult = builder.create<mlir::triton::SplatOp>(loc, splatTy, opResult);
            }
            for (int o = 0; o < slot.numOutputs; o++) ssaValues[slot.outputSlotIndices[o]] = opResult;
          } else if (cat == TritonOpCategory::NORMALIZATION) {
            if (slot.numInputs < 1) continue;
            auto inputIt = ssaValues.find(slot.inputSourceIndices[0]);
            if (inputIt == ssaValues.end()) continue;
            // Read normalization axis from iArgs (same pattern as reduction)
            int axis = (slot.numIArgs > 0 && slot.iArgs) ? static_cast<int>(slot.iArgs[0]) : -1;
            // Handle negative axis
            if (auto tensorTy = mlir::dyn_cast<mlir::RankedTensorType>(inputIt->second.getType())) {
              int inputRank = static_cast<int>(tensorTy.getRank());
              if (axis < 0) axis += inputRank;
              if (axis < 0 || axis >= inputRank) axis = inputRank - 1;
            } else {
              if (axis < 0) axis = 0;
            }
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
            std::string normKey = normalizeOpToken(slot.opName);
            auto getNormInput = [&](int inputPos) -> mlir::Value {
              if (inputPos >= slot.numInputs) return mlir::Value();
              int src = slot.inputSourceIndices[inputPos];
              auto it = ssaValues.find(src);
              return (it != ssaValues.end()) ? it->second : mlir::Value();
            };

            mlir::Value scaleVal, biasVal, meanVal, varianceVal;
            if (normKey == "batchnorm") {
              meanVal = getNormInput(1);
              varianceVal = getNormInput(2);
              scaleVal = getNormInput(3);
              biasVal = getNormInput(4);
            } else {
              scaleVal = getNormInput(1);
              biasVal = getNormInput(2);
            }

            float epsilon2 = (slot.numTArgs > 0 && slot.tArgs) ? static_cast<float>(slot.tArgs[0]) : 1e-5f;
            auto opResult = emitNormalizationOp(builder, loc, slot.opName, inputIt->second, axis, outputType,
                                                scaleVal, biasVal, meanVal, varianceVal, epsilon2);
            if (!mlir::isa<mlir::RankedTensorType>(opResult.getType())) {
              auto splatTensorType = mlir::RankedTensorType::get({blockSize}, opResult.getType());
              opResult = builder.create<mlir::triton::SplatOp>(loc, splatTensorType, opResult);
            }
            for (int o = 0; o < slot.numOutputs; o++) ssaValues[slot.outputSlotIndices[o]] = opResult;
          } else if (cat == TritonOpCategory::ROPE) {
            // ─── ROPE: paired elementwise rotation ───
            if (slot.numInputs >= 3 && slot.numOutputs >= 1) {
              int inputSrc2 = slot.inputSourceIndices[0];
              int cosSrc = slot.inputSourceIndices[1];
              int sinSrc = slot.inputSourceIndices[2];
              int outSlot2 = slot.outputSlotIndices[0];

              auto inPtr2 = getSlotArgPtr(inputSrc2);
              auto cosPtr2 = getSlotArgPtr(cosSrc);
              auto sinPtr2 = getSlotArgPtr(sinSrc);
              auto outPtr2 = getSlotArgPtr(outSlot2);

              NDArray* inArr2 = resolveArr(inputSrc2);
              NDArray* cosArr2 = resolveArr(cosSrc);
              NDArray* outArr2 = resolveArr(outSlot2);

              if (inPtr2 && cosPtr2 && sinPtr2 && outPtr2 && inArr2 && cosArr2 && outArr2) {
                std::vector<LongType> inShapeVec, cosShapeVec2;
                for (int d = 0; d < inArr2->rankOf(); d++) inShapeVec.push_back(inArr2->sizeAt(d));
                for (int d = 0; d < cosArr2->rankOf(); d++) cosShapeVec2.push_back(cosArr2->sizeAt(d));
                int nElems = static_cast<int>(outArr2->lengthOf());

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
                  for (int o = 0; o < slot.numOutputs; o++)
                    ssaValues[slot.outputSlotIndices[o]] = result2;
                } else {
                  // Fallback: pointer-based emitter
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

                  int ropeType2 = (slot.numIArgs > 0 && slot.iArgs) ? static_cast<int>(slot.iArgs[0]) : 0;
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
                    for (int o = 0; o < slot.numOutputs; o++) ssaValues[slot.outputSlotIndices[o]] = reloaded;
                  }
                }
              }
            }
          } else if (cat == TritonOpCategory::CONSTANT_GENERATION) {
            // Constant generation ops produce values independent of input data.
            // Must emit proper constants — NOT forward input SSA values.
            DataType outDtype = FLOAT32;
            if (slot.numOutputs > 0) {
              int outIdx = slot.outputSlotIndices[0];
              outDtype = resolveDtype(outIdx);
            }
            auto cgElemType = getMLIRType(builder, outDtype);
            auto cgTensorType = mlir::RankedTensorType::get({blockSize}, cgElemType);

            std::string opLower3 = slot.opName;
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
              if (slot.numTArgs > 0 && slot.tArgs) {
                fillVal = static_cast<float>(slot.tArgs[0]);
              } else if (slot.numOutputs > 0) {
                int outIdx = slot.outputSlotIndices[0];
                auto* arr = secResolveArr(outIdx);
                if (arr && arr->lengthOf() > 0) {
                  arr->syncToHost();
                  fillVal = arr->e<float>(0);
                }
              }
              cgResult = splatConstantF32(builder, loc, cgTensorType, fillVal);
            } else if (opLower3 == "range") {
              float start = 0.0f, step = 1.0f;
              if (slot.numTArgs >= 1 && slot.tArgs) start = static_cast<float>(slot.tArgs[0]);
              if (slot.numTArgs >= 3 && slot.tArgs) step = static_cast<float>(slot.tArgs[2]);
              int rangeLen = blockSize;
              if (slot.numOutputs > 0) {
                int outIdx = slot.outputSlotIndices[0];
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
              if (slot.numOutputs > 0) {
                int outIdx = slot.outputSlotIndices[0];
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
              if (slot.numOutputs > 0) {
                int outIdx = slot.outputSlotIndices[0];
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
              for (int o = 0; o < slot.numOutputs; o++) ssaValues[slot.outputSlotIndices[o]] = cgResult;
            }
          } else if (cat == TritonOpCategory::SHAPE_MANIPULATION) {
            // Shape ops (reshape, squeeze, expand_dims, permute, transpose) are
            // now isolated into their own SHAPE_MANIPULATION sections by
            // identifySections() and always run via native fallback.  This SSA
            // forwarding code should never be reached, but is kept as defensive
            // fallback.  It is correct only for non-permute ops (identity view).
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
          // Detect DPA v2 (input order Q,V,K) vs standard (Q,K,V)
          std::string opLowerSec = slot.opName;
          std::transform(opLowerSec.begin(), opLowerSec.end(), opLowerSec.begin(), ::tolower);
          bool isDpaV2Sec = (opLowerSec.find("dot_product_attention") != std::string::npos);

          // 3D Q: compound attention (onnx_multi_head_attention) — handled via dual-buffer kernel
          auto qShapeSec = resolveShape(qSrc);

          int kSrc = isDpaV2Sec ? slot.inputSourceIndices[2] : slot.inputSourceIndices[1];
          int vSrc = isDpaV2Sec ? slot.inputSourceIndices[1] : slot.inputSourceIndices[2];
          int outSlot = slot.outputSlotIndices[0];

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
            numQHeads = (slot.numIArgs > 0 && slot.iArgs) ? static_cast<int>(slot.iArgs[0]) : 1;
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

          for (int inp = 3; inp < slot.numInputs && !hasPastKv; inp++) {
            int candidateSrc = slot.inputSourceIndices[inp];
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
                if (inp + 1 < slot.numInputs) {
                  int pvCandidate = slot.inputSourceIndices[inp + 1];
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
                      si, headDim, slot.numInputs);
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
                      si, slot.opName.c_str(), qSrc, kSrc, vSrc, effectiveKSrc, effectiveVSrc,
                      outSlot, hasPastKv, pastKeySrc, pastValueSrc,
                      slot.numInputs, slot.numOutputs);

            // Shapes: Q, K (raw), K (effective), V
            auto kShapeDbg = resolveShape(kSrc);
            auto vShapeDbg = resolveShape(vSrc);
            DSP_DIAG(COMPILE, "ATTN slot=%d shapes: Q=[%s] K=[%s] effK=[%s] V=[%s]",
                      si, fmtShape(qShape).c_str(), fmtShape(kShapeDbg).c_str(),
                      fmtShape(effectiveKShape).c_str(), fmtShape(vShapeDbg).c_str());

            // Every input: source index, shape, op name, whether slot array exists and its length
            for (int inp = 0; inp < slot.numInputs; inp++) {
              int srcIdx = slot.inputSourceIndices[inp];
              auto srcShape = resolveShape(srcIdx);
              const char* srcOp = "EXT";
              if (srcIdx >= 0 && srcIdx < totalSlots) srcOp = slots[srcIdx].opName.c_str();
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
            for (int outp = 0; outp < slot.numOutputs; outp++) {
              int outIdx = slot.outputSlotIndices[outp];
              auto outShape = resolveShape(outIdx);
              DSP_DIAG(COMPILE, "ATTN slot=%d   output[%d] slot=%d shape=[%s]",
                        si, outp, outIdx, fmtShape(outShape).c_str());
            }

            // iArgs and tArgs
            if (slot.numIArgs > 0) {
              std::string iStr;
              for (int a = 0; a < slot.numIArgs && a < 16; a++) {
                if (a) iStr += ",";
                iStr += std::to_string(slot.iArgs[a]);
              }
              DSP_DIAG(COMPILE, "ATTN slot=%d   iArgs=[%s] (%d total)", si, iStr.c_str(), slot.numIArgs);
            }
            if (slot.numTArgs > 0) {
              char tBuf[256] = {0};
              int toff = 0;
              for (int a = 0; a < slot.numTArgs && a < 8 && toff < 240; a++) {
                toff += snprintf(tBuf + toff, sizeof(tBuf) - toff, "%s%.6g", a > 0 ? "," : "", slot.tArgs[a]);
              }
              DSP_DIAG(COMPILE, "ATTN slot=%d   tArgs=[%s] (%d total)", si, tBuf, slot.numTArgs);
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
            // 3D key: [B, seqK, kvHeads*headDim] — infer from iArgs
            if (slot.numIArgs > 0 && slot.iArgs) {
              int totalQHeads = static_cast<int>(slot.iArgs[0]);
              if (totalQHeads > 0) numKvHeads = totalQHeads;  // will be refined below
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
              for (int s = 0; s < totalSlots; s++) {
                for (int o = 0; o < slots[s].numOutputs; o++) {
                  if (slots[s].outputSlotIndices[o] == kSrc) {
                    // Found the producer of K. Check its inputs for external KV cache.
                    for (int pi = 0; pi < slots[s].numInputs; pi++) {
                      int psrc = slots[s].inputSourceIndices[pi];
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
                    goto kProducerSearchDone;
                  }
                }
              }
            }
            kProducerSearchDone:

            // Strategy 2: Scan ALL attention op inputs for 4D KV cache shapes.
            // Covers cases where past_key is a direct input to the attention op.
            if (derivedSeqK == 0) {
              for (int inp = 0; inp < slot.numInputs; inp++) {
                int src = slot.inputSourceIndices[inp];
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
            for (int i = 0; i < slot.numInputs; i++) { if (i) idxStr += ","; idxStr += std::to_string(slot.inputSourceIndices[i]); }
            DSP_DIAG(COMPILE, "TritonIRBuilder: attention slot=%d op=%s numInputs=%d numOutputs=%d "
                      "seqQ=%d seqK=%d heads=%d/%d hd=%d hasPastKv=%d dualBuf=%d "
                      "pastKeySrc=%d qShape=[%s] kShape=[%s] effectiveKShape=[%s] inputSrcs=[%s]",
                      si, slot.opName.c_str(), slot.numInputs, slot.numOutputs,
                      seqQ, seqK, numQHeads, numKvHeads, headDim,
                      hasPastKv ? 1 : 0, useDualBuffer ? 1 : 0,
                      pastKeySrc,
                      fmtShp(qShape).c_str(), fmtShp(resolveShape(kSrc)).c_str(),
                      fmtShp(effectiveKShape).c_str(), idxStr.c_str());
          }
          if (slot.numInputs > 3) {
            int biasSrc = slot.inputSourceIndices[3];
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
          bool kBufferValid = true;
          {
            auto effKShape = resolveShape(effectiveKSrc);
            LongType effKLen = 1;
            for (auto d : effKShape) effKLen *= d;
            if (effKLen == 0 && seqK > 0) {
              kBufferValid = false;
              DSP_DIAG(COMPILE, "TritonIRBuilder: skipping FUSED_ATTENTION at slot %d — "
                        "effective K buffer (src=%d) is empty but seqK=%d%s. "
                        "Falling back to C++ native.",
                        si, effectiveKSrc, seqK,
                        seqKDerivedFromExternal ? " (seqK derived from external inputs)" : "");
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
            if (useDualBuffer && slot.numOutputs >= 2) {
              // Dual-buffer: write current K/V to present_key/value output at position pastSeq.
              // Guard: verify the present_key output buffer can hold pastSeqLen + seqKVCur positions.
              // For static KV caches, the past_key buffer is pre-allocated to maxKvLen but the
              // present_key output was allocated during warmup with a much smaller shape.
              // Writing at pastSeqLen offset would be out-of-bounds. In static KV cache mode,
              // the caller handles cache updates — skip the write.
              int presentKeySlot = slot.outputSlotIndices[1];
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
              if (slot.numOutputs >= 3 && pkFits) {
                int presentValSlot = slot.outputSlotIndices[2];
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
            bool gatherNd = (sec.type == KernelSectionType::GATHER_ND);
            emitGatherSection(builder, loc, pid, blockSize, dataPtr, idxPtr, outPtr, axis,
                              dataShape, indicesShape, nElements, gatherNd);
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
            std::vector<int> begins, ends, strides;
            resolveStridedSliceParams(slot, inputShape, resolveArr, begins, ends, strides);
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

  // Pre-scan: skip external inputs consumed only by CONST_GEN ops (same as buildModule)
  std::unordered_set<int> constGenOnlyInputs;
  {
    std::unordered_map<int, bool> inputHasNonConstGenConsumer;
    for (int i = startSlot; i <= endSlot; i++) {
      auto cat = getOpCategory(slots[i].opName);
      bool isConstGen = (cat == TritonOpCategory::CONSTANT_GENERATION);
      for (int inp = 0; inp < slots[i].numInputs; inp++) {
        int srcIdx = slots[i].inputSourceIndices[inp];
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
    for (int inp = 0; inp < slots[i].numInputs; inp++) {
      int srcIdx = slots[i].inputSourceIndices[inp];
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
        } else {
          // No live array — resolve from producing op (same logic as buildModule)
          auto producerCat = getOpCategory(slots[i].opName);
          if (producerCat == TritonOpCategory::CAST && slots[i].numIArgs > 0 && slots[i].iArgs) {
            arg.dtype = static_cast<DataType>(slots[i].iArgs[0]);
          } else if (slots[i].numInputs > 0) {
            int inputSrc = slots[i].inputSourceIndices[0];
            NDArray* inputArr = resolveArray(inputSrc);
            if (inputArr) arg.dtype = inputArr->dataType();
          }
          if (arg.shape.empty() && slots[i].numInputs > 0) {
            int inputSrc = slots[i].inputSourceIndices[0];
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
    DSP_DIAG(FALLBACK, "TritonIRBuilder::buildMatmulModule: could not map matmul A/B/C to kernel args "
              "(aArgIdx=%d, bArgIdx=%d, cArgIdx=%d)", aArgIdx, bArgIdx, cArgIdx);
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

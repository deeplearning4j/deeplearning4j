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

#include <graph/gpu/TritonGraphBackend.h>
#include <graph/gpu/CapturedModuleRegistry.h>
#include <graph/gpu/TritonGraphBackend_internal.h>
#include <graph/gpu/TritonIRBuilder.h>
#include <graph/gpu/TritonIRBuilder_internal.h>
#include <graph/gpu/OpCategoryTable.h>
#include <graph/gpu/TritonTargetDispatch.h>
#include <graph/gpu/SectionTypeConfig.h>
#include <graph/gpu/FusionScoring.h>
#include <graph/DspAnalysisUtils.h>
#include <graph/DspDiagnostics.h>
#include <system/Environment.h>
#include <helpers/shape.h>
#include <helpers/logger.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#ifdef _WIN32
#include <windows.h>
#else
#include <pthread.h>
#endif
#include <sstream>
#include <thread>
#include <vector>

// Stack size for Triton compile worker threads (64 MB).
// MLIR's CoalescePass uses recursive getBackwardSliceImpl/getForwardSliceImpl
// that recurse to the full depth of the SSA def-use chain. For large fused
// segments (768 ops), this can exceed 1000 recursive frames, each consuming
// several KB of stack. The default std::thread stack (8 MB on Linux) is
// insufficient. 64 MB accommodates chains up to ~8000 ops with margin.
static constexpr size_t TRITON_COMPILE_WORKER_STACK_SIZE = 64 * 1024 * 1024;

namespace sd {
namespace graph {

using namespace triton_internal;
using namespace ir_builder_internal;

bool TritonGraphBackend::compileSegment(GraphSegment& seg, NativeSlot* slots,
                                        NDArray** externalInputs, int numExternalInputs,
                                        NDArray** outputSlots, int totalOutputSlots,
                                        LongType shapeKey,
                                        int totalSlots,
                                        int* requestedOutputSlotIndices,
                                        int numRequestedOutputs) {
  // Compile-time contract: outputSlots entries are either dereferenceable or
  // null. Requested-output slots can hold uninitialized placeholder NDArrays
  // (poison shape descriptors) before their producer runs; every consumer below
  // (dtype hashing, identifySections, IR build) already falls back to cached
  // slot metadata for null entries, so present placeholders as null here instead
  // of guarding each dereference site.
  std::vector<NDArray*> sanitizedOutputSlots;
  if (outputSlots != nullptr && totalOutputSlots > 0) {
    sanitizedOutputSlots.assign(outputSlots, outputSlots + totalOutputSlots);
    int placeholderCount = 0;
    for (auto& entry : sanitizedOutputSlots) {
      if (entry != nullptr && !entry->hasValidShapeInfo()) {
        entry = nullptr;
        placeholderCount++;
      }
    }
    if (placeholderCount > 0) {
      DSP_DIAG(COMPILE, "TritonGraphBackend::compileSegment: seg[%d-%d] nulled %d uninitialized "
               "output-slot placeholder(s); cached slot metadata will be used",
               seg.def.startSlot, seg.def.endSlot, placeholderCount);
    }
    outputSlots = sanitizedOutputSlots.data();
  }

  int activeDevice = -1;
  cudaError_t activeDeviceErr = cudaGetDevice(&activeDevice);
  if (activeDeviceErr != cudaSuccess) {
    DSP_DIAG(BACKEND, "TritonGraphBackend::compileSegment: cudaGetDevice failed for segment [%d-%d]: %s",
              seg.def.startSlot, seg.def.endSlot, cudaGetErrorString(activeDeviceErr));
    cudaGetLastError();
    return false;
  }

  int targetDevice = -1;
  if (seg.def.startSlot >= 0) {
    targetDevice = slots[seg.def.startSlot].targetDeviceId;
  }

  int compileDevice = activeDevice;
  if (targetDevice >= 0) {
    int deviceCount = 0;
    cudaError_t countErr = cudaGetDeviceCount(&deviceCount);
    if (countErr != cudaSuccess || deviceCount <= 0) {
      DSP_DIAG(BACKEND, "TritonGraphBackend::compileSegment: failed to query CUDA device count "
                "for segment [%d-%d] targetDeviceId=%d: %s",
                seg.def.startSlot, seg.def.endSlot, targetDevice, cudaGetErrorString(countErr));
      cudaGetLastError();
      return false;
    }
    if (targetDevice >= deviceCount) {
      DSP_DIAG(BACKEND, "TritonGraphBackend::compileSegment: invalid targetDeviceId=%d for segment [%d-%d] "
                "(deviceCount=%d)",
                targetDevice, seg.def.startSlot, seg.def.endSlot, deviceCount);
      return false;
    }
    compileDevice = targetDevice;
  }

  // Always call cudaSetDevice — not just when compileDevice != activeDevice.
  // On std::async precompilation threads the CUDA primary context may not be
  // initialized at all (cudaGetDevice returns 0 by default, but no context is
  // bound). Without an explicit cudaSetDevice, subsequent driver API calls
  // (cuModuleLoadDataEx, cuFuncSetAttribute) operate on an uninitialized or
  // wrong context, causing misaligned-address and kernel-launch failures.
  {
    cudaError_t setDeviceErr = cudaSetDevice(compileDevice);
    if (setDeviceErr != cudaSuccess) {
      DSP_DIAG(BACKEND, "TritonGraphBackend::compileSegment: failed to set CUDA device %d for segment [%d-%d]: %s",
                compileDevice, seg.def.startSlot, seg.def.endSlot, cudaGetErrorString(setDeviceErr));
      cudaGetLastError();
      return false;
    }
  }

    auto& cacheEnv = Environment::getInstance();
  bool cacheCompileAll = cacheEnv.tritonCompileAll();
  size_t cacheExcludeHash = std::hash<std::string>()(cacheEnv.tritonExcludeOps());
  size_t cacheIncludeHash = std::hash<std::string>()(cacheEnv.tritonIncludeTypes());
  bool cacheGraphCapture = cacheEnv.tritonGraphCapture();
  // Include a hash of segment-internal output dtypes so that FLOAT32 and INT8
  // kernels for the same shapes produce distinct cache entries.
  // computeSegmentShapeKey() only hashes dtype for cross-segment inputs; this
  // covers the segment-owned intermediates (e.g. dequantised INT8 K/V arrays).
  size_t cacheDtypeHash = computeSegInternalDtypeHash(slots, seg.def.startSlot, seg.def.endSlot,
                                                       externalInputs, numExternalInputs,
                                                       outputSlots, totalOutputSlots);
  SegmentCacheKey key{seg.def.startSlot, seg.def.endSlot, shapeKey, compileDevice,
                      cacheCompileAll, cacheExcludeHash, cacheIncludeHash,
                      cacheGraphCapture, &seg};
  key.segInternalDtypeHash = cacheDtypeHash;

  {
    std::lock_guard<std::mutex> lock(cacheMtx_);
    auto it = cache_.find(key);
    if (it != cache_.end()) {
      // Already compiled for this shape
      lastCompilationAudit_ = it->second.audit;
      totalCacheHits_++;
      // Modules are still resident on cache hit — check the warn threshold
      // here too so the diagnostic fires even when no fresh module load
      // happened during this compileSegment call.
      const int64_t warnBytes =
          sd::Environment::getInstance().triton().moduleResidencyWarnBytes();
      if (warnBytes > 0) {
        const size_t devBytes = getTritonModuleMemory(compileDevice);
        if (static_cast<int64_t>(devBytes) > warnBytes) {
          bool expected = false;
          if (residencyWarned_[compileDevice].compare_exchange_strong(
                  expected, true, std::memory_order_acq_rel)) {
            sd::Environment::getInstance().triton().incrementModuleResidencyWarnFireCount();
            sd_printf("TritonGraphBackend: module residency on device %d (%zu bytes) "
                      "exceeds warn threshold (%lld bytes). Consider raising "
                      "ND4J_TRITON_MODULE_RESIDENCY_BUDGET_BYTES or the warn ceiling.\n",
                      compileDevice, devBytes, static_cast<long long>(warnBytes));
          }
        }
      }
      return true;
    }
    if (failedCache_.find(key) != failedCache_.end()) {
      DSP_DIAG(COMPILE, "TritonGraphBackend::compileSegment: skipping previously failed segment [%d-%d] "
               "(shapeKey=%lld, deviceId=%d)",
               seg.def.startSlot, seg.def.endSlot, shapeKey, compileDevice);
      return false;
    }
  }

  int segmentOps = seg.def.endSlot - seg.def.startSlot + 1;
  CompiledSegment compiledSeg;

  // Use section boundaries for splitting: identify natural boundaries where
  // the op category changes (e.g., element-wise -> matmul -> element-wise).
  // Each sub-kernel handles one section or a group of compatible sections.
  // This produces correct kernels because each section type needs different
  // grid dimensions, shared memory, and execution patterns.
  auto sections = TritonIRBuilder::identifySections(slots, seg.def.startSlot, seg.def.endSlot,
                                                      outputSlots, totalOutputSlots,
                                                      externalInputs, numExternalInputs);

  if (sections.empty()) {
    // All ops in this segment are natively-handled gap ops (cuBLAS/native CUDA).
    // Triton has nothing to compile, but this is NOT an error — the CUDA graph
    // capture will execute these ops natively via executeSlot() on the capture
    // stream (forceNativeCapture / nativeOnlyGraphCapture path).
    // Mark every op as isNativeHandled=true so segDispatchCompile's audit check
    // at the "compiledCount==0 && failedCount==0" branch can return Status::OK
    // and advance the lifecycle to CAPTURE_PENDING via markCompiled().
    // Returning false here was the bug: it caused segDispatchCompile to return
    // KERNEL_FAILURE, preventing capture from ever being reached for cuBLAS-only
    // segments in GEM_AUTO mode after resetModelState() cleared the Triton cache.
    lastCompilationAudit_.clear();
    for (int s = seg.def.startSlot; s <= seg.def.endSlot; s++) {
      CompilationAuditEntry entry;
      entry.slotIndex = s;
      entry.opName = slots[s].ident.opName;
      entry.wasCompiled = false;
      entry.isNativeHandled = true;
      entry.reason = "all-gap segment: op is natively-executed (cuBLAS/CUDA native), no Triton kernel needed";
      lastCompilationAudit_.push_back(entry);
    }
    DSP_DIAG(COMPILE, "TritonGraphBackend::compileSegment: seg[%d-%d] is all-gap (native-only) — "
             "marking %d ops as isNativeHandled, returning true for capture eligibility",
             seg.def.startSlot, seg.def.endSlot, seg.def.endSlot - seg.def.startSlot + 1);
    return true;
  }

  DSP_DIAG_SEG(COMPILE, seg.def.startSlot, "TritonGraphBackend: segment [%d-%d] has %d ops, %d sections (deviceId=%d)",
               seg.def.startSlot, seg.def.endSlot, segmentOps, static_cast<int>(sections.size()),
               compileDevice);

  auto shapeInfoToVector = [](const LongType* shapeInfo) -> std::vector<LongType> {
    std::vector<LongType> shapeVec;
    if (shapeInfo == nullptr) return shapeVec;
    int rank = shape::rank(shapeInfo);
    shapeVec.reserve(rank);
    const LongType* dims = shape::shapeOf(shapeInfo);
    for (int d = 0; d < rank; d++) {
      shapeVec.push_back(dims[d]);
    }
    return shapeVec;
  };

  auto resolveShape = [&](int slotIdx) -> std::vector<LongType> {
    if (slotIdx < 0) {
      int extIdx = -(slotIdx + 1);
      if (extIdx >= 0 && extIdx < numExternalInputs && externalInputs && externalInputs[extIdx]) {
        std::vector<LongType> shapeVec;
        auto* arr = externalInputs[extIdx];
        shapeVec.reserve(arr->rankOf());
        for (int d = 0; d < arr->rankOf(); d++) {
          shapeVec.push_back(arr->sizeAt(d));
        }
        return shapeVec;
      }
      return {};
    }

    if (slotIdx < totalOutputSlots && outputSlots && outputSlots[slotIdx]) {
      std::vector<LongType> shapeVec;
      auto* arr = outputSlots[slotIdx];
      shapeVec.reserve(arr->rankOf());
      for (int d = 0; d < arr->rankOf(); d++) {
        shapeVec.push_back(arr->sizeAt(d));
      }
      return shapeVec;
    }

    for (int s = seg.def.startSlot; s <= seg.def.endSlot; s++) {
      auto& producerSlot = slots[s];
      if (!producerSlot.shapeCacheValid() || producerSlot.shapeCache.cachedOutputShapes.empty()) continue;
      for (int o = 0; o < producerSlot.wiring.numOutputs; o++) {
        if (o >= static_cast<int>(producerSlot.shapeCache.cachedOutputShapes.size())) break;
        if (producerSlot.wiring.outputSlotIndices[o] == slotIdx) {
          return shapeInfoToVector(producerSlot.shapeCache.cachedOutputShapes[o]);
        }
      }
    }

    return {};
  };

  auto shapeLength = [](const std::vector<LongType>& s) -> LongType {
    if (s.empty()) return 0;
    LongType len = 1;
    for (auto d : s) len *= d;
    return len;
  };

  auto resolveArray = [&](int slotIdx) -> NDArray* {
    if (slotIdx < 0) {
      int extIdx = -(slotIdx + 1);
      if (extIdx >= 0 && extIdx < numExternalInputs && externalInputs) {
        return externalInputs[extIdx];
      }
      return nullptr;
    }

    if (slotIdx >= 0 && slotIdx < totalOutputSlots && outputSlots) {
      return outputSlots[slotIdx];
    }

    return nullptr;
  };

  auto resolveSlotTraits = [](const NativeSlot& slot) -> uint32_t {
    return dsp::resolveSlotTraits(slot);
  };

  auto isIntegralOrBoolType = [](DataType dt) -> bool {
    switch (dt) {
      case DataType::BOOL:
      case DataType::INT8:
      case DataType::UINT8:
      case DataType::INT16:
      case DataType::UINT16:
      case DataType::INT32:
      case DataType::UINT32:
      case DataType::INT64:
      case DataType::UINT64:
        return true;
      default:
        return false;
    }
  };

  auto isShapeControlArray = [&](NDArray* arr) -> bool {
    if (arr == nullptr) return false;
    if (!isIntegralOrBoolType(arr->dataType())) return false;
    return arr->lengthOf() >= 0 && arr->lengthOf() <= 64 && arr->rankOf() <= 8;
  };

  auto slotOutputsShapeControl = [&](const NativeSlot& slot) -> bool {
    if (slot.wiring.numOutputs <= 0) return false;

    bool sawOutput = false;
    for (int o = 0; o < slot.wiring.numOutputs; o++) {
      NDArray* out = resolveArray(slot.wiring.outputSlotIndices[o]);
      if (out == nullptr || !isShapeControlArray(out)) {
        return false;
      }
      sawOutput = true;
    }

    return sawOutput;
  };

  auto slotAllInputsShapeControl = [&](const NativeSlot& slot) -> bool {
    if (slot.wiring.numInputs <= 0) return false;

    bool sawInput = false;
    for (int i = 0; i < slot.wiring.numInputs; i++) {
      NDArray* in = resolveArray(slot.wiring.inputSourceIndices[i]);
      if (in == nullptr || !isShapeControlArray(in)) {
        return false;
      }
      sawInput = true;
    }

    return sawInput;
  };

  auto slotLooksLikeShapeControl = [&](const NativeSlot& slot) -> bool {
    uint32_t traits = resolveSlotTraits(slot);
    // SHAPE_ONLY and VIEW_PRODUCING are always host-only / shape-control.
    if ((traits & (sd::ops::OP_TRAIT_SHAPE_ONLY_OUTPUT |
                   sd::ops::OP_TRAIT_VIEW_PRODUCING)) != 0) {
      return true;
    }
    // CONST_GEN is only shape-control when it also carries VALUE_DEPENDENT_SHAPE
    // (range, linspace, create — read host tensor values to determine output size).
    // Plain fill ops (zeros_as, ones_as) have CONST_GEN but are CUDA-kernel-safe
    // and should be placed inside Triton islands, not relegated to gap slots.
    if ((traits & sd::ops::OP_TRAIT_CONSTANT_GENERATION) != 0 &&
        (traits & sd::ops::OP_TRAIT_VALUE_DEPENDENT_SHAPE) != 0) {
      return true;
    }

    if (!slotOutputsShapeControl(slot)) {
      return false;
    }

    if ((traits & (sd::ops::OP_TRAIT_VALUE_DEPENDENT_SHAPE |
                   sd::ops::OP_TRAIT_DATA_DEPENDENT)) != 0) {
      return slotAllInputsShapeControl(slot);
    }

    return slotAllInputsShapeControl(slot);
  };

  auto sectionLooksLikeShapeControl = [&](const KernelSection& section) -> bool {
    bool sawSlot = false;
    for (int si = section.startSlot; si <= section.endSlot; si++) {
      if (si < seg.def.startSlot || si > seg.def.endSlot) return false;
      if (!slotLooksLikeShapeControl(slots[si])) {
        return false;
      }
      sawSlot = true;
    }
    return sawSlot;
  };

  {
    TritonIRBuilder planningBuilder;
    std::vector<TritonOpCategory> categories;
    std::vector<std::vector<LongType>> shapes;
    categories.reserve(seg.def.endSlot - seg.def.startSlot + 1);
    shapes.reserve(seg.def.endSlot - seg.def.startSlot + 1);
    for (int i = seg.def.startSlot; i <= seg.def.endSlot; i++) {
      categories.push_back(getOpCategoryFromName(slots[i].ident.opName));
      if (slots[i].wiring.numOutputs > 0) {
        shapes.push_back(resolveShape(slots[i].wiring.outputSlotIndices[0]));
      } else {
        shapes.push_back({});
      }
    }

    int planningBlockSize = 1024;
    int planningWarps = 4;
    int planningStages = 1;
    planningBuilder.selectTileConfig(categories, shapes, planningBlockSize, planningWarps, planningStages);
    (void) planningWarps;
    (void) planningStages;

    auto sectionMaxElements = [&](const KernelSection& sec) -> LongType {
      LongType maxElements = 0;
      for (int si = sec.startSlot; si <= sec.endSlot; si++) {
        for (int o = 0; o < slots[si].wiring.numOutputs; o++) {
          int outIdx = slots[si].wiring.outputSlotIndices[o];
          LongType elems = shapeLength(resolveShape(outIdx));
          if (elems > maxElements) maxElements = elems;
        }
      }
      if (maxElements <= 0) {
        for (int si = sec.startSlot; si <= sec.endSlot; si++) {
          for (int inp = 0; inp < slots[si].wiring.numInputs; inp++) {
            int srcIdx = slots[si].wiring.inputSourceIndices[inp];
            LongType elems = shapeLength(resolveShape(srcIdx));
            if (elems > maxElements) maxElements = elems;
          }
        }
      }
      return maxElements;
    };

    auto deriveAttentionBlocks = [&](const KernelSection& sec) -> int {
      int batchSize = std::max(1, sec.batchSize);
      int numHeads = std::max(1, sec.numHeads);
      int seqQ = std::max(1, sec.seqQ);
      int seqK = std::max(1, sec.seqK);
      int headDim = std::max(1, sec.headDim);

      if (sec.batchSize <= 0 || sec.numHeads <= 0 || sec.seqQ <= 0 || sec.headDim <= 0) {
        for (int si = sec.startSlot; si <= sec.endSlot; si++) {
          auto& slot = slots[si];
          if (getOpCategoryFromName(slot.ident.opName) != TritonOpCategory::FUSED_ATTENTION || slot.wiring.numInputs < 1) {
            continue;
          }

          std::string opLower = slot.ident.opName;
          std::transform(opLower.begin(), opLower.end(), opLower.begin(), ::tolower);
          bool isDpaV2 = (opLower.find("dot_product_attention") != std::string::npos);
          bool qIsBSHD = isDpaV2;
          int kInputIdx = isDpaV2 ? 2 : 1;

          auto qShape = resolveShape(slot.wiring.inputSourceIndices[0]);
          if (qShape.size() >= 4) {
            batchSize = static_cast<int>(std::max<LongType>(1, qShape[0]));
            if (qIsBSHD) {
              seqQ = static_cast<int>(std::max<LongType>(1, qShape[1]));
              numHeads = static_cast<int>(std::max<LongType>(1, qShape[2]));
            } else {
              numHeads = static_cast<int>(std::max<LongType>(1, qShape[1]));
              seqQ = static_cast<int>(std::max<LongType>(1, qShape[2]));
            }
            headDim = static_cast<int>(std::max<LongType>(1, qShape[3]));
            if (slot.wiring.numInputs > kInputIdx) {
              auto kShape = resolveShape(slot.wiring.inputSourceIndices[kInputIdx]);
              if (kShape.size() >= 3) {
                int seqKDim = qIsBSHD ? 1 : 2;
                if (static_cast<int>(kShape.size()) > seqKDim) {
                  seqK = static_cast<int>(std::max<LongType>(1, kShape[seqKDim]));
                }
              }
            }
          } else if (qShape.size() == 3) {
            batchSize = static_cast<int>(std::max<LongType>(1, qShape[0]));
            seqQ = static_cast<int>(std::max<LongType>(1, qShape[1]));
            int hidden = static_cast<int>(std::max<LongType>(1, qShape[2]));
            numHeads = (slot.args.numIArgs > 0 && slot.args.iArgs) ? static_cast<int>(slot.args.iArgs[0]) : 1;
            if (numHeads <= 0) numHeads = 1;
            headDim = hidden / numHeads;

            bool hasPastKv = false;
            for (int inp = 3; inp < slot.wiring.numInputs && !hasPastKv; inp++) {
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
                  hasPastKv = true;
                  int seqKV = 1;
                  if (slot.wiring.numInputs > kInputIdx) {
                    auto curKShape = resolveShape(slot.wiring.inputSourceIndices[kInputIdx]);
                    if (curKShape.size() == 3) {
                      seqKV = static_cast<int>(std::max<LongType>(1, curKShape[1]));
                    }
                  }
                  seqK = candidatePastSeq + seqKV;
                }
              }
            }
            if (!hasPastKv && slot.wiring.numInputs > kInputIdx) {
              auto kShape = resolveShape(slot.wiring.inputSourceIndices[kInputIdx]);
              if (kShape.size() >= 2) {
                seqK = static_cast<int>(std::max<LongType>(1, kShape[1]));
              }
            }
          }
          break;
        }
      }

      auto attnTile = chooseFusedAttentionTileConfig(batchSize, numHeads, seqQ, seqK, headDim);
      int blockM = std::max(1, attnTile.blockM);
      int batchHeads = std::max(1, batchSize * numHeads);
      int gridQ = std::max(1, (seqQ + blockM - 1) / blockM);
      LongType blocks64 = static_cast<LongType>(batchHeads) * gridQ;
      if (blocks64 > static_cast<LongType>(2147483647)) {
        blocks64 = static_cast<LongType>(2147483647);
      }
      return static_cast<int>(std::max<LongType>(1, blocks64));
    };

    auto computePlanningGrid = [&](const KernelSection& sec) -> int {
      if (sec.type == KernelSectionType::FUSED_ATTENTION) {
        return deriveAttentionBlocks(sec);
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
      LongType blocks64 = (maxElements + planningBlockSize - 1) / planningBlockSize;
      if (blocks64 > static_cast<LongType>(2147483647)) {
        blocks64 = static_cast<LongType>(2147483647);
      }
      return static_cast<int>(std::max<LongType>(1, blocks64));
    };

    for (auto& sec : sections) {
      sec.gridRequirement = computePlanningGrid(sec);
    }
  }

  // -- Step 1: Build adaptive compile ranges from section graph --
  struct SubSegmentRange {
    int startSlot;
    int endSlot;
    int opsCount;
    int startSectionIdx;
    int endSectionIdx;
  };
  std::deque<SubSegmentRange> pendingRanges;

  bool sectionFusionEnabled = Environment::getInstance().tritonSectionFusion();

  auto isStandaloneSection = [&](const KernelSection& section) -> bool {
    const auto& cfg = getSectionTypeConfig(section.type);
    return shouldBeStandalone(cfg, sectionFusionEnabled, section);
  };

  // Determine which sections are compiled as Triton kernels vs native ordered ranges.
  // Default mode: only ELEMENTWISE/IDENTITY compiled, everything else uses cuBLAS/native.
  // When tritonCompileAll=true: compile ALL section types EXCEPT those containing
  // ops in the exclusion list (ND4J_TRITON_EXCLUDE_OPS). This allows fine-grained
  // control, e.g. keeping matmul on cuBLAS while compiling reductions/norms via Triton.
  auto& compileEnv = Environment::getInstance();
  bool compileAll = compileEnv.tritonCompileAll();

  // Parse tritonIncludeTypes into a set of allowed section types for compileAll mode
  std::unordered_set<KernelSectionType> includedTypes;
  {
    std::string includeStr = compileEnv.tritonIncludeTypes();
    if (!includeStr.empty()) {
      std::istringstream iss(includeStr);
      std::string token;
      while (std::getline(iss, token, ',')) {
        // Trim whitespace
        size_t start = token.find_first_not_of(" \t");
        size_t end = token.find_last_not_of(" \t");
        if (start == std::string::npos) continue;
        token = token.substr(start, end - start + 1);
        // Map type name to enum
        if (token == "ELEMENTWISE")
          includedTypes.insert(KernelSectionType::ELEMENTWISE);
        else if (token == "IDENTITY")
          includedTypes.insert(KernelSectionType::IDENTITY);
        else if (token == "CONST_GEN" || token == "CONSTANT_GENERATION")
          includedTypes.insert(KernelSectionType::CONSTANT_GENERATION);
        else if (token == "SHAPE_MANIP" || token == "SHAPE_MANIPULATION")
          includedTypes.insert(KernelSectionType::SHAPE_MANIPULATION);
        else if (token == "GATHER")
          includedTypes.insert(KernelSectionType::GATHER);
        else if (token == "GATHER_ND")
          includedTypes.insert(KernelSectionType::GATHER_ND);
        else if (token == "CONCAT")
          includedTypes.insert(KernelSectionType::CONCAT);
        else if (token == "SPLIT")
          includedTypes.insert(KernelSectionType::SPLIT);
        else if (token == "SPLIT_V")
          includedTypes.insert(KernelSectionType::SPLIT_V);
        else if (token == "STACK")
          includedTypes.insert(KernelSectionType::STACK);
        else if (token == "REDUCTION")
          includedTypes.insert(KernelSectionType::REDUCTION);
        else if (token == "NORMALIZATION")
          includedTypes.insert(KernelSectionType::NORMALIZATION);
        else if (token == "ATTENTION" || token == "FUSED_ATTENTION")
          includedTypes.insert(KernelSectionType::FUSED_ATTENTION);
        else if (token == "MATMUL")
          includedTypes.insert(KernelSectionType::MATMUL);
        else if (token == "TILE")
          includedTypes.insert(KernelSectionType::TILE);
        else if (token == "STRIDED_SLICE")
          includedTypes.insert(KernelSectionType::STRIDED_SLICE);
        else if (token == "SCATTER_ND")
          includedTypes.insert(KernelSectionType::SCATTER_ND);
        else if (token == "SCATTER_ND_UPDATE")
          includedTypes.insert(KernelSectionType::SCATTER_ND_UPDATE);
        else if (token == "CONVOLUTION")
          includedTypes.insert(KernelSectionType::CONVOLUTION);
        else
          DSP_DIAG(COMPILE, "TritonGraphBackend: unknown include type '%s'", token.c_str());
      }
      if (!includedTypes.empty()) {
        DSP_DIAG(COMPILE, "TritonGraphBackend: compileAll with include types filter (%d types)",
                 static_cast<int>(includedTypes.size()));
      }
    }
  }

  auto isNativeOrderedSection = [&](const KernelSection& section) -> bool {
    const auto& cfg = getSectionTypeConfig(section.type);
    if (shouldStayNativeOrdered(cfg, compileAll, includedTypes,
                                    sd::Environment::getInstance().tritonGraphCapture())) return true;

    // Shape-control heuristic: catches small-integer bookkeeping ops (position IDs,
    // axis scalars) that are cheaper to run natively than compile.  BUT: if the user
    // explicitly listed this section type in includedTypes, respect that intent —
    // the heuristic must not override an explicit compile request.  In LLM decode
    // (batch=1, seq=1), legitimate data-movement GATHERs produce int64 length-1
    // tensors that trip the heuristic even though they should be compiled.
    bool explicitlyIncluded = !includedTypes.empty() &&
                              includedTypes.find(cfg.type) != includedTypes.end();
    if (!explicitlyIncluded && sectionLooksLikeShapeControl(section)) {
      return true;
    }

    // Check op-level exclusion list (not table-driven -- per-op granularity)
    for (int si = section.startSlot; si <= section.endSlot; si++) {
      if (si >= 0 && si < totalSlots && !slots[si].ident.opName.empty()) {
        if (compileEnv.isTritonExcludedOp(slots[si].ident.opName)) {
          return true;
        }
      }
    }
    return false;
  };

  auto makeRange = [&](int startSec, int endSec) -> SubSegmentRange {
    SubSegmentRange r;
    r.startSectionIdx = startSec;
    r.endSectionIdx = endSec;
    r.startSlot = sections[startSec].startSlot;
    r.endSlot = sections[endSec].endSlot;
    r.opsCount = r.endSlot - r.startSlot + 1;
    return r;
  };

  auto makeSlotRange = [&](int startSlot, int endSlot,
                           int startSec, int endSec) -> SubSegmentRange {
    SubSegmentRange r;
    r.startSectionIdx = startSec;
    r.endSectionIdx = endSec;
    r.startSlot = startSlot;
    r.endSlot = endSlot;
    r.opsCount = r.endSlot - r.startSlot + 1;
    return r;
  };

  auto splitRange = [&](const SubSegmentRange& range) {
    const int sectionCount = range.endSectionIdx - range.startSectionIdx + 1;
    if (sectionCount > 1) {
      int midSec = range.startSectionIdx + (range.endSectionIdx - range.startSectionIdx) / 2;
      auto left = makeRange(range.startSectionIdx, midSec);
      auto right = makeRange(midSec + 1, range.endSectionIdx);
      // Process left first to preserve slot order.
      pendingRanges.push_front(right);
      pendingRanges.push_front(left);
      return;
    }

    if (range.opsCount <= 1) return;
    int midSlot = range.startSlot + (range.endSlot - range.startSlot) / 2;
    auto left = makeSlotRange(range.startSlot, midSlot, range.startSectionIdx, range.endSectionIdx);
    auto right = makeSlotRange(midSlot + 1, range.endSlot, range.startSectionIdx, range.endSectionIdx);
    DSP_DIAG(COMPILE, "TritonGraphBackend: splitting single-section range [%d-%d] by slots -> [%d-%d] + [%d-%d]",
             range.startSlot, range.endSlot,
             left.startSlot, left.endSlot, right.startSlot, right.endSlot);
    pendingRanges.push_front(right);
    pendingRanges.push_front(left);
  };

  // Pre-compute cooperative launch capacity for this device.
  // Only needed when cooperative launch is enabled -- when disabled (default),
  // grid sync barriers are suppressed in the IR builder and no splitting is needed.
  const bool cooperativeEnabled = Environment::getInstance().tritonCooperativeLaunch();
  int preCheckSmCount = 0;
  int preCheckMaxThreadsPerSM = 0;
  int preCheckMaxSharedPerSM = 0;
  if (cooperativeEnabled) {
    int currentDevice = 0;
    cudaError_t devErr = cudaGetDevice(&currentDevice);
    if (devErr == cudaSuccess) {
      CUdevice cuDevice = 0;
      CUresult cuDevErr = cuDeviceGet(&cuDevice, currentDevice);
      if (cuDevErr == CUDA_SUCCESS) {
        cuDeviceGetAttribute(&preCheckSmCount, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, cuDevice);
        cuDeviceGetAttribute(&preCheckMaxThreadsPerSM, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, cuDevice);
        cuDeviceGetAttribute(&preCheckMaxSharedPerSM,
            CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR, cuDevice);
      }
    }
  }

  // Estimate max cooperative blocks for a typical Triton kernel (4 warps = 128 threads).
  // This is conservative -- actual capacity depends on shared memory and register usage.
  auto estimateMaxCooperativeBlocks = [&](int numWarps, int estimatedSharedMem) -> unsigned long long {
    int threadsPerBlock = std::max(1, numWarps) * 32;
    int blocksPerSmByThreads = (preCheckMaxThreadsPerSM > 0 && threadsPerBlock > 0)
        ? (preCheckMaxThreadsPerSM / threadsPerBlock) : 16;
    int blocksPerSmBySmem = 16;
    if (estimatedSharedMem > 0 && preCheckMaxSharedPerSM > 0) {
      blocksPerSmBySmem = preCheckMaxSharedPerSM / estimatedSharedMem;
    }
    int blocksPerSm = std::max(1, std::min(blocksPerSmByThreads, blocksPerSmBySmem));
    return static_cast<unsigned long long>(preCheckSmCount) * blocksPerSm;
  };

  for (int secIdx = 0; secIdx < static_cast<int>(sections.size());) {
    // Non-elementwise sections are skipped --
    // they become ordered native ranges executed between Triton islands.
    if (isNativeOrderedSection(sections[secIdx])) {
      DSP_DIAG(COMPILE, "TritonGraphBackend: section %d [%d-%d] type=%s routed to native ordered execution",
               secIdx, sections[secIdx].startSlot, sections[secIdx].endSlot,
               sectionTypeName(sections[secIdx].type));
      secIdx++;
      continue;
    }

    if (isStandaloneSection(sections[secIdx])) {
      DSP_DIAG(COMPILE, "TritonGraphBackend: section %d [%d-%d] type=%s STANDALONE (Triton compile)",
               secIdx, sections[secIdx].startSlot, sections[secIdx].endSlot,
               sectionTypeName(sections[secIdx].type));
      pendingRanges.push_back(makeRange(secIdx, secIdx));
      secIdx++;
      continue;
    }

    // Log each compiled section
    DSP_DIAG(COMPILE, "TritonGraphBackend: section %d [%d-%d] type=%s COMPILED (Triton)",
             secIdx, sections[secIdx].startSlot, sections[secIdx].endSlot,
             sectionTypeName(sections[secIdx].type));

    // Merge consecutive non-standalone Triton sections into one compile range.
    int runStart = secIdx;
    int runEnd = secIdx;

    bool fusionScoringEnabled = Environment::getInstance().tritonFusionScoring();
    float fusionMinScore = Environment::getInstance().tritonFusionMinScore();

    while (runEnd + 1 < static_cast<int>(sections.size()) &&
           !isStandaloneSection(sections[runEnd + 1]) &&
           !isNativeOrderedSection(sections[runEnd + 1])) {
      if (fusionScoringEnabled) {
        float score = scoreSectionFusionRange(sections, runStart, runEnd, runEnd + 1,
                                             slots, seg.def.startSlot, seg.def.endSlot,
                                             outputSlots, totalOutputSlots);
        if (score < fusionMinScore) {
          DSP_DIAG(FUSION, "TritonGraphBackend: section %d [%d-%d] NOT merged (score=%.2f < min=%.2f)",
                   runEnd + 1, sections[runEnd + 1].startSlot, sections[runEnd + 1].endSlot,
                   score, fusionMinScore);
          break;
        }
        DSP_DIAG(FUSION, "TritonGraphBackend: section %d [%d-%d] type=%s SCORED merge (score=%.2f) into range starting at section %d",
                 runEnd + 1, sections[runEnd + 1].startSlot, sections[runEnd + 1].endSlot,
                 sectionTypeName(sections[runEnd + 1].type), score, runStart);
      } else {
        DSP_DIAG(FUSION, "TritonGraphBackend: section %d [%d-%d] type=%s MERGED into range starting at section %d",
                 runEnd + 1, sections[runEnd + 1].startSlot, sections[runEnd + 1].endSlot,
                 sectionTypeName(sections[runEnd + 1].type), runStart);
      }
      runEnd++;
    }

    pendingRanges.push_back(makeRange(runStart, runEnd));
    secIdx = runEnd + 1;
  }

  if (pendingRanges.empty()) {
    DSP_DIAG(COMPILE, "TritonGraphBackend: no sub-segments to compile for segment [%d-%d]",
             seg.def.startSlot, seg.def.endSlot);
    lastCompilationAudit_ = compiledSeg.audit;
    compiledSeg.segInternalDtypeHash = cacheDtypeHash;
    {
      std::lock_guard<std::mutex> lock(cacheMtx_);
      failedCache_.erase(key);
      cache_[key] = std::move(compiledSeg);
      // Populate secondary dtype index so lookup sites without outputSlots can
      // find the right dtype hash for this live segment instance.
      dtypeIndex_[DtypeIndexKey{seg.def.startSlot, seg.def.endSlot, shapeKey,
                                compileDevice, &seg}] = cacheDtypeHash;
    }
    return true;
  }

  auto& env = Environment::getInstance();
  const int envOpsCap = std::max(0, env.tritonMaxSubsegmentOps());
  // When the environment cap is 0 (disabled), use MAX_COMPILABLE_OPS as default.
  // Huge IR modules (>512 ops) cause LLVM register pressure explosions (441K virtual
  // regs for 3840 ops) leading to SIGABRT. Set ND4J_TRITON_MAX_SUBSEGMENT_OPS=0
  // to truly disable (not recommended).
  const int maxOpsCap = (envOpsCap > 0) ? envOpsCap : MAX_COMPILABLE_OPS;
  const int maxSectionsCap = std::max(0, env.tritonMaxSubsegmentSections());
  const int maxParallelCompiles = std::max(1, getMaxParallelCompilations());
  DSP_DIAG(COMPILE, "TritonGraphBackend: adaptive section packing for [%d-%d] "
           "(initialRanges=%d, opsCap=%d(env=%d), sectionsCap=%d, compileThreads=%d)",
           seg.def.startSlot, seg.def.endSlot,
           static_cast<int>(pendingRanges.size()),
           maxOpsCap, envOpsCap, maxSectionsCap, maxParallelCompiles);
  if (maxSectionsCap <= 0) {
    DSP_DIAG(COMPILE, "TritonGraphBackend: section cap disabled for [%d-%d] (runtime control); "
             "set tritonMaxSubsegmentSections>0 to force additional splitting",
             seg.def.startSlot, seg.def.endSlot);
  }

  int activeCompileDevice = -1;
  cudaError_t compileDeviceErr = cudaGetDevice(&activeCompileDevice);
  if (compileDeviceErr != cudaSuccess) {
    DSP_DIAG(BACKEND, "TritonGraphBackend: cudaGetDevice failed before adaptive compilation "
              "for segment [%d-%d]: %s",
              seg.def.startSlot, seg.def.endSlot, cudaGetErrorString(compileDeviceErr));
    cudaGetLastError();
    return false;
  }
  DSP_DIAG(BACKEND, "TritonGraphBackend: compile device binding seg[%d-%d] targetDeviceId=%d activeDevice=%d cacheDeviceId=%d",
           seg.def.startSlot, seg.def.endSlot, targetDevice, activeCompileDevice, compileDevice);

  struct CompileRangeResult {
    SubSegmentRange range;
    CompiledKernel compiled;
  };

  std::atomic<long long> launchedRanges{0};
  std::atomic<long long> completedRanges{0};
  std::atomic<long long> successfulRanges{0};
  std::atomic<long long> failedRanges{0};
  std::atomic<long long> inFlightRanges{0};
  long long splitRetryCount = 0;
  int batchIndex = 0;

  auto compileRange = [&](const SubSegmentRange& range) -> CompileRangeResult {
    CompileRangeResult result;
    result.range = range;
    const auto rangeStart = std::chrono::steady_clock::now();
    const long long launchIndex = launchedRanges.fetch_add(1) + 1;
    const long long inflightNow = inFlightRanges.fetch_add(1) + 1;
    if (compileDevice >= 0) {
      cudaError_t setDeviceErr = cudaSetDevice(compileDevice);
      if (setDeviceErr != cudaSuccess) {
        DSP_DIAG(BACKEND, "TritonGraphBackend: failed to set CUDA device %d in compile worker "
                  "for range [%d-%d]: %s",
                  compileDevice, range.startSlot, range.endSlot,
                  cudaGetErrorString(setDeviceErr));
        const long long completedNow = completedRanges.fetch_add(1) + 1;
        const long long failedNow = failedRanges.fetch_add(1) + 1;
        const long long inflightAfter = inFlightRanges.fetch_sub(1) - 1;
        DSP_DIAG(COMPILE, "TritonGraphBackend: compile progress seg[%d-%d] launch#%lld range[%d-%d] "
                 "status=FAILED(set-device) completed=%lld success=%lld failed=%lld inflight=%lld",
                 seg.def.startSlot, seg.def.endSlot, launchIndex, range.startSlot, range.endSlot,
                 completedNow, successfulRanges.load(), failedNow, inflightAfter);
        cudaGetLastError();
        return result;
      }
    }
    const int sectionCount = range.endSectionIdx - range.startSectionIdx + 1;
    DSP_DIAG(COMPILE, "TritonGraphBackend: compile progress seg[%d-%d] launch#%lld range[%d-%d] "
             "START (ops=%d, sections=%d, inflight=%lld)",
             seg.def.startSlot, seg.def.endSlot, launchIndex, range.startSlot, range.endSlot,
             range.opsCount, sectionCount, inflightNow);
    result.compiled = compileToGpuBinary(slots, range.startSlot, range.endSlot,
                                         totalSlots,
                                         externalInputs, numExternalInputs,
                                         outputSlots, totalOutputSlots,
                                         requestedOutputSlotIndices,
                                         numRequestedOutputs);
    const bool success = (result.compiled.gpuModule && result.compiled.kernelFunction);
    if (result.compiled.gpuModule && result.compiled.kernelFunction) {
      result.compiled.startSlot_ = range.startSlot;
      result.compiled.endSlot_ = range.endSlot;
    }
    const long long completedNow = completedRanges.fetch_add(1) + 1;
    const long long successNow = success ? (successfulRanges.fetch_add(1) + 1) : successfulRanges.load();
    const long long failedNow = success ? failedRanges.load() : (failedRanges.fetch_add(1) + 1);
    const long long inflightAfter = inFlightRanges.fetch_sub(1) - 1;
    const auto elapsedMs = static_cast<long long>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - rangeStart).count());
    DSP_DIAG(COMPILE, "TritonGraphBackend: compile progress seg[%d-%d] launch#%lld range[%d-%d] "
             "DONE status=%s elapsedMs=%lld completed=%lld success=%lld failed=%lld inflight=%lld",
             seg.def.startSlot, seg.def.endSlot, launchIndex, range.startSlot, range.endSlot,
             success ? "OK" : "FAILED", elapsedMs,
             completedNow, successNow, failedNow, inflightAfter);
    return result;
  };

  // -- Step 2: Work-stealing compile loop --
  // Instead of batch-sequential dispatch (launch N, wait ALL, repeat), use a
  // shared work queue with condition variables so workers pick up new work
  // (including split retries) as soon as they finish -- eliminating tail latency.

  // Pre-split any ranges that exceed caps before entering the work queue.
  {
    std::deque<SubSegmentRange> preSplit;
    while (!pendingRanges.empty()) {
      SubSegmentRange range = pendingRanges.front();
      pendingRanges.pop_front();
      const int sectionCount = range.endSectionIdx - range.startSectionIdx + 1;
      const bool canSplit = (sectionCount > 1) || (range.opsCount > 1);
      const bool exceedsOpsCap = (maxOpsCap > 0 && range.opsCount > maxOpsCap);
      const bool exceedsSectionsCap = (maxSectionsCap > 0 && sectionCount > maxSectionsCap);
      if (canSplit && (exceedsOpsCap || exceedsSectionsCap)) {
        DSP_DIAG(COMPILE, "TritonGraphBackend: pre-splitting range [%d-%d] (%d ops, %d sections) "
                 "to honor caps (opsCap=%d, sectionsCap=%d)",
                 range.startSlot, range.endSlot, range.opsCount, sectionCount,
                 maxOpsCap, maxSectionsCap);
        splitRange(range);
      } else {
        preSplit.push_back(range);
      }
    }
    pendingRanges = std::move(preSplit);
  }

  const int totalInitialRanges = static_cast<int>(pendingRanges.size());
  DSP_DIAG(COMPILE, "TritonGraphBackend: work-stealing compile seg[%d-%d] "
           "(ranges=%d, workers=%d)",
           seg.def.startSlot, seg.def.endSlot, totalInitialRanges, maxParallelCompiles);

  // Shared state for the work-stealing pool
  std::mutex workMtx;
  std::condition_variable workCv;
  std::vector<CompileRangeResult> allResults;
  std::vector<SlotRange> leafOrderedRanges;  // Leaf ranges that stay native-ordered
  std::atomic<int> activeWorkers{0};

  auto workerLoop = [&]() {
    while (true) {
      SubSegmentRange range;
      {
        std::unique_lock<std::mutex> lock(workMtx);
        workCv.wait(lock, [&] {
          return !pendingRanges.empty() ||
                 (pendingRanges.empty() && activeWorkers.load() == 0);
        });
        if (pendingRanges.empty()) {
          if (activeWorkers.load() == 0) return;  // All work done
          continue;
        }
        range = pendingRanges.front();
        pendingRanges.pop_front();
        activeWorkers.fetch_add(1);
      }

      auto result = compileRange(range);
      const bool success = (result.compiled.gpuModule && result.compiled.kernelFunction);

      {
        std::lock_guard<std::mutex> lock(workMtx);
        if (success) {
          allResults.push_back(std::move(result));
        } else {
          cudaGetLastError();
          const int sectionCount = range.endSectionIdx - range.startSectionIdx + 1;
          const bool canSplit = (sectionCount > 1) || (range.opsCount > 1);
          if (canSplit) {
            DSP_DIAG(COMPILE, "TritonGraphBackend: adaptive range [%d-%d] compile failed; "
                     "splitting by section graph",
                     range.startSlot, range.endSlot);
            splitRetryCount++;
            splitRange(range);
            // New sub-ranges are in pendingRanges; wake other workers
          } else {
            // Leaf range failed — keep it as a native ordered range instead of aborting
            std::string opNames;
            for (int s = range.startSlot; s <= range.endSlot; s++) {
              if (!opNames.empty()) opNames += ",";
              opNames += slots[s].ident.opName;
            }
            DSP_DIAG(COMPILE, "TritonGraphBackend: leaf range [%d-%d] not compilable, "
                     "keeping as native ordered range (ops: %s)",
                     range.startSlot, range.endSlot, opNames.c_str());
            leafOrderedRanges.push_back({range.startSlot, range.endSlot});
          }
        }
        activeWorkers.fetch_sub(1);
      }
      workCv.notify_all();
    }
  };

  if (maxParallelCompiles > 1 && pendingRanges.size() > 1) {
    // Launch worker threads with enlarged stack for MLIR pass recursion.
    // MLIR's CoalescePass recurses through getBackwardSliceImpl/getForwardSliceImpl
    // to the full SSA def-use chain depth. The default 8MB std::thread stack is
    // insufficient for large fused segments (768+ ops → 1000+ recursive frames).
    // Use pthread_create with explicit 64MB stack size.
    const int numWorkers = std::min(maxParallelCompiles,
                                    static_cast<int>(pendingRanges.size()));

#ifdef _WIN32
    struct WinWorkerArg {
      std::function<void()>* fn;
    };

    std::vector<HANDLE> threadHandles(numWorkers);
    std::vector<WinWorkerArg> args(numWorkers);
    std::vector<std::thread> fallbackThreads;
    auto workerFn = std::function<void()>(workerLoop);

    for (int i = 0; i < numWorkers; i++) {
      args[i].fn = &workerFn;
      threadHandles[i] = CreateThread(
          nullptr,
          TRITON_COMPILE_WORKER_STACK_SIZE,
          [](LPVOID arg) -> DWORD {
            auto* wa = static_cast<WinWorkerArg*>(arg);
            (*wa->fn)();
            return 0;
          }, &args[i], STACK_SIZE_PARAM_IS_A_RESERVATION, nullptr);
      if (threadHandles[i] == nullptr) {
        DSP_DIAG(COMPILE, "TritonGraphBackend: CreateThread failed for worker %d (err=%lu), "
                 "falling back to std::thread", i, GetLastError());
        fallbackThreads.emplace_back(workerLoop);
      }
    }

    for (int i = 0; i < numWorkers; i++) {
      if (threadHandles[i] != nullptr) {
        WaitForSingleObject(threadHandles[i], INFINITE);
        CloseHandle(threadHandles[i]);
      }
    }
    for (auto& ft : fallbackThreads) {
      ft.join();
    }
#else
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, TRITON_COMPILE_WORKER_STACK_SIZE);

    struct PthreadWorkerArg {
      std::function<void()>* fn;
    };

    std::vector<pthread_t> threads(numWorkers);
    std::vector<PthreadWorkerArg> args(numWorkers);
    int inlineFallbackCount = 0;  // workers that must run on calling thread
    auto workerFn = std::function<void()>(workerLoop);

    for (int i = 0; i < numWorkers; i++) {
      args[i].fn = &workerFn;
      int rc = pthread_create(&threads[i], &attr,
          [](void* arg) -> void* {
            auto* wa = static_cast<PthreadWorkerArg*>(arg);
            (*wa->fn)();
            return nullptr;
          }, &args[i]);
      if (rc != 0) {
        DSP_DIAG(COMPILE, "TritonGraphBackend: pthread_create failed for worker %d (rc=%d), "
                 "will run inline on calling thread", i, rc);
        threads[i] = 0;
        inlineFallbackCount++;
      }
    }
    pthread_attr_destroy(&attr);

    for (int i = 0; i < numWorkers; i++) {
      if (threads[i] != 0) {
        pthread_join(threads[i], nullptr);
      }
    }
    // Run inline fallback iterations on the calling thread instead of spawning
    // std::thread (which throws std::system_error when resources are exhausted,
    // crashing the process).  workerLoop drains the shared task queue.
    for (int i = 0; i < inlineFallbackCount; i++) {
      workerLoop();
    }
#endif
  } else {
    // Single-threaded path: still needs enlarged stack because MLIR's
    // CoalescePass recursion can blow the calling thread's stack (JVM threads
    // default to 1MB on Linux). Spawn a single worker with 64MB stack.
    auto singleThreadWork = [&]() {
      while (!pendingRanges.empty()) {
        SubSegmentRange range = pendingRanges.front();
        pendingRanges.pop_front();

        auto result = compileRange(range);
        const bool success = (result.compiled.gpuModule && result.compiled.kernelFunction);
        if (success) {
          allResults.push_back(std::move(result));
        } else {
          cudaGetLastError();
          const int sectionCount = range.endSectionIdx - range.startSectionIdx + 1;
          const bool canSplit = (sectionCount > 1) || (range.opsCount > 1);
          if (canSplit) {
            DSP_DIAG(COMPILE, "TritonGraphBackend: adaptive range [%d-%d] compile failed; "
                     "splitting by section graph",
                     range.startSlot, range.endSlot);
            splitRetryCount++;
            splitRange(range);
          } else {
            // Leaf range failed — keep it as a native ordered range instead of aborting
            std::string opNames;
            for (int s = range.startSlot; s <= range.endSlot; s++) {
              if (!opNames.empty()) opNames += ",";
              opNames += slots[s].ident.opName;
            }
            DSP_DIAG(COMPILE, "TritonGraphBackend: leaf range [%d-%d] not compilable, "
                     "keeping as native ordered range (ops: %s)",
                     range.startSlot, range.endSlot, opNames.c_str());
            leafOrderedRanges.push_back({range.startSlot, range.endSlot});
          }
        }
      }
    };

#ifdef _WIN32
    struct SingleWinArg {
      std::function<void()>* fn;
    };
    auto workFn = std::function<void()>(singleThreadWork);
    SingleWinArg swa{&workFn};
    HANDLE singleThread = CreateThread(
        nullptr,
        TRITON_COMPILE_WORKER_STACK_SIZE,
        [](LPVOID arg) -> DWORD {
          auto* wa = static_cast<SingleWinArg*>(arg);
          (*wa->fn)();
          return 0;
        }, &swa, STACK_SIZE_PARAM_IS_A_RESERVATION, nullptr);

    if (singleThread != nullptr) {
      WaitForSingleObject(singleThread, INFINITE);
      CloseHandle(singleThread);
    } else {
      DSP_DIAG(COMPILE, "TritonGraphBackend: CreateThread failed for single-threaded compile "
               "(err=%lu), running on calling thread (risk of stack overflow)", GetLastError());
      singleThreadWork();
    }
#else
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, TRITON_COMPILE_WORKER_STACK_SIZE);

    struct SingleWorkerArg {
      std::function<void()>* fn;
    };
    auto workFn = std::function<void()>(singleThreadWork);
    SingleWorkerArg swa{&workFn};
    pthread_t singleThread;
    int rc = pthread_create(&singleThread, &attr,
        [](void* arg) -> void* {
          auto* wa = static_cast<SingleWorkerArg*>(arg);
          (*wa->fn)();
          return nullptr;
        }, &swa);
    pthread_attr_destroy(&attr);

    if (rc == 0) {
      pthread_join(singleThread, nullptr);
    } else {
      DSP_DIAG(COMPILE, "TritonGraphBackend: pthread_create failed for single-threaded compile "
               "(rc=%d), running on calling thread (risk of stack overflow)", rc);
      singleThreadWork();
    }
#endif
  }

  // Merge leaf native ordered ranges into compiledSeg
  if (!leafOrderedRanges.empty()) {
    for (auto& fb : leafOrderedRanges) {
      compiledSeg.orderedRanges.push_back(fb);
      // Add audit entries for native-ordered slots
      for (int s = fb.startSlot; s <= fb.endSlot; s++) {
        CompilationAuditEntry entry;
        entry.slotIndex = s;
        entry.opName = slots[s].ident.opName;
        entry.wasCompiled = false;
        entry.isNativeHandled = true;
        entry.reason = "leaf range not Triton-compilable, using native ordered execution";
        compiledSeg.audit.push_back(entry);
      }
    }
    DSP_DIAG(COMPILE, "TritonGraphBackend: %d leaf ranges added to native ordered execution for [%d-%d]",
             static_cast<int>(leafOrderedRanges.size()), seg.def.startSlot, seg.def.endSlot);
  }

  // Move successful results into compiledSeg
  for (auto& r : allResults) {
    compiledSeg.subKernels.push_back(std::move(r.compiled));
  }

  DSP_DIAG(COMPILE, "TritonGraphBackend: compile progress seg[%d-%d] summary "
           "(launched=%lld, completed=%lld, success=%lld, failed=%lld, splitRetries=%lld)",
           seg.def.startSlot, seg.def.endSlot,
           launchedRanges.load(), completedRanges.load(),
           successfulRanges.load(), failedRanges.load(), splitRetryCount);

  std::sort(compiledSeg.subKernels.begin(), compiledSeg.subKernels.end(),
            [](const CompiledKernel& a, const CompiledKernel& b) {
              return a.startSlot_ < b.startSlot_;
            });

  if (compiledSeg.subKernels.empty() && compiledSeg.orderedRanges.empty()) {
    // Populate audit for the full segment so the error message includes op names
    lastCompilationAudit_.clear();
    for (int s = seg.def.startSlot; s <= seg.def.endSlot; s++) {
      CompilationAuditEntry entry;
      entry.slotIndex = s;
      entry.opName = slots[s].ident.opName;
      entry.wasCompiled = false;
      entry.reason = "all compilation attempts failed — no sub-kernels or native ranges produced";
      lastCompilationAudit_.push_back(entry);
    }
    DSP_DIAG(COMPILE, "TritonGraphBackend: no compiled sub-kernels AND no native ordered "
             "ranges for segment [%d-%d] — genuine compilation failure",
             seg.def.startSlot, seg.def.endSlot);
    {
      std::lock_guard<std::mutex> lock(cacheMtx_);
      failedCache_.insert(key);
    }
    return false;
  }

  for (auto& kernel : compiledSeg.subKernels) {
    compiledSeg.audit.insert(compiledSeg.audit.end(),
                             kernel.audit.begin(),
                             kernel.audit.end());
  }

  DSP_DIAG(COMPILE, "TritonGraphBackend: adaptive compilation produced %d sub-segments "
           "and %d native ordered ranges for [%d-%d]",
           static_cast<int>(compiledSeg.subKernels.size()),
           static_cast<int>(compiledSeg.orderedRanges.size()),
           seg.def.startSlot, seg.def.endSlot);

  // Pre-allocate launch workspace outside runtime execution/capture.
  // This ensures the first captured Triton execution does not perform allocations.
  // Skip entirely when there are no compiled sub-kernels (all-native segment).
  if (compileDevice >= 0 && !compiledSeg.subKernels.empty()) {
    auto setDevErr = cudaSetDevice(compileDevice);
    if (setDevErr != cudaSuccess) {
      DSP_DIAG(BACKEND, "TritonGraphBackend: failed to set CUDA device %d for launch workspace pre-allocation: %s",
                compileDevice, cudaGetErrorString(setDevErr));
      cudaGetLastError();
      for (auto& kernel : compiledSeg.subKernels) {
        if (kernel.gpuModule) {
          recordModuleFree(kernel.loadedDeviceId >= 0 ? kernel.loadedDeviceId : compileDevice,
                           kernel.estimatedModuleBytes);
          if (sd::graph::modreg::releaseFromOwner(kernel.gpuModule)) {
          TritonTargetDispatch::unloadModule(kernel.gpuModule);
        }
        kernel.moduleCaptureOwned = false;
          kernel.gpuModule = nullptr;
          kernel.kernelFunction = nullptr;
          kernel.loadedDeviceId = -1;
        }
      }
      return false;
    }
    // Ensure zero-length argument kernels have a valid per-device dummy pointer
    // before any stream capture path executes.
    if (getDummyDevicePtrForDevice(compileDevice, false) == nullptr) {
      DSP_DIAG(MEMORY, "TritonGraphBackend: failed pre-allocating dummy arg buffer on device %d for segment [%d-%d]",
                compileDevice, seg.def.startSlot, seg.def.endSlot);
    }
  }

  cudaStream_t preallocStream = nullptr;
  bool ownPreallocStream = false;
  // Try to trim pool first — release unused reserved memory so stream creation can succeed
  memory::CudaMemoryPool::getInstance().trimPool(compileDevice);
  cudaGetLastError();  // clear any trim error
  auto streamErr = cudaStreamCreateWithFlags(&preallocStream, cudaStreamNonBlocking);
  if (streamErr != cudaSuccess) {
    // Fall back to default stream (0) when memory is too constrained for a new stream.
    // Stream creation needs a small device memory allocation for control structures,
    // which can fail when the CUDA memory pool has reserved nearly all device memory.
    DSP_DIAG(COMPILE, "TritonGraphBackend: pre-allocation stream creation failed for segment [%d-%d]: %s — using default stream",
              seg.def.startSlot, seg.def.endSlot, cudaGetErrorString(streamErr));
    cudaGetLastError();
    preallocStream = 0;  // default stream
    ownPreallocStream = false;
  } else {
    ownPreallocStream = true;
  }

  const bool useConsolidatedArgTable = Environment::getInstance().tritonConsolidatedArgTable();

  auto cleanupCompiledWorkspace = [&]() {
    if (compiledSeg.consolidatedArgTableDevice != nullptr) {
      auto freeErr = freeDeviceBufferAsync(compiledSeg.consolidatedArgTableDevice, preallocStream);
      if (freeErr != cudaSuccess) {
        DSP_DIAG(MEMORY, "TritonGraphBackend: failed freeing consolidated arg table for [%d-%d]: %s",
                  seg.def.startSlot, seg.def.endSlot, cudaGetErrorString(freeErr));
        cudaGetLastError();
      } else {
        recordModuleFree(compiledSeg.consolidatedArgTableDeviceId >= 0 ? compiledSeg.consolidatedArgTableDeviceId : compileDevice,
                         compiledSeg.consolidatedArgTableBytes);
      }
      compiledSeg.consolidatedArgTableDevice = nullptr;
      compiledSeg.consolidatedArgTableBytes = 0;
      compiledSeg.consolidatedArgTableDeviceId = -1;
      compiledSeg.useConsolidatedArgTable = false;
    }
    if (compiledSeg.consolidatedArgTableHostPinned != nullptr) {
      auto& memPool = sd::memory::CudaMemoryPool::getInstance();
      if (!compiledSeg.consolidatedArgTableCaptureOwned) {
        memPool.freePinnedHost(compiledSeg.consolidatedArgTableHostPinned);
      }
      compiledSeg.consolidatedArgTableHostPinned = nullptr;
      compiledSeg.consolidatedArgTableCaptureOwned = false;
    }
    for (auto& k : compiledSeg.subKernels) {
      if (!useConsolidatedArgTable && k.cachedArgTableDevice) {
        auto freeErr = freeDeviceBufferAsync(k.cachedArgTableDevice, preallocStream);
        if (freeErr != cudaSuccess) {
          DSP_DIAG(MEMORY, "TritonGraphBackend: failed freeing pre-allocated arg table for [%d-%d]: %s",
                    k.startSlot_, k.endSlot_, cudaGetErrorString(freeErr));
          cudaGetLastError();
        } else {
          recordModuleFree(k.cachedArgTableDeviceId >= 0 ? k.cachedArgTableDeviceId : compileDevice,
                           k.cachedArgTableBytes);
        }
      }
      if (k.cachedSyncCounterDevice) {
        auto freeErr = freeDeviceBufferAsync(k.cachedSyncCounterDevice, preallocStream);
        if (freeErr != cudaSuccess) {
          DSP_DIAG(MEMORY, "TritonGraphBackend: failed freeing pre-allocated sync counter for [%d-%d]: %s",
                    k.startSlot_, k.endSlot_, cudaGetErrorString(freeErr));
          cudaGetLastError();
        } else {
          recordModuleFree(k.cachedSyncCounterDeviceId >= 0 ? k.cachedSyncCounterDeviceId : compileDevice,
                           sizeof(int));
        }
      }
      if (k.cachedGlobalScratchDevice) {
        auto freeErr = freeDeviceBufferAsync(k.cachedGlobalScratchDevice, preallocStream);
        if (freeErr != cudaSuccess) {
          DSP_DIAG(MEMORY, "TritonGraphBackend: failed freeing pre-allocated global scratch for [%d-%d]: %s",
                    k.startSlot_, k.endSlot_, cudaGetErrorString(freeErr));
          cudaGetLastError();
        } else {
          recordModuleFree(k.cachedGlobalScratchDeviceId >= 0 ? k.cachedGlobalScratchDeviceId : compileDevice,
                           k.cachedGlobalScratchBytes);
        }
      }
      k.cachedArgTableDevice = nullptr;
      k.cachedArgTableBytes = 0;
      k.cachedArgTableDeviceId = -1;
      k.cachedSyncCounterDevice = nullptr;
      k.cachedSyncCounterDeviceId = -1;
      k.cachedGlobalScratchDevice = nullptr;
      k.cachedGlobalScratchBytes = 0;
      k.cachedGlobalScratchDeviceId = -1;
      if (k.gpuModule) {
        recordModuleFree(k.loadedDeviceId >= 0 ? k.loadedDeviceId : compileDevice,
                         k.estimatedModuleBytes);
        if (sd::graph::modreg::releaseFromOwner(k.gpuModule)) {
          TritonTargetDispatch::unloadModule(k.gpuModule);
        }
        k.moduleCaptureOwned = false;
        k.gpuModule = nullptr;
        k.kernelFunction = nullptr;
        k.loadedDeviceId = -1;
      }
    }
    if (ownPreallocStream && preallocStream != nullptr) {
      cudaStreamDestroy(preallocStream);
    }
    preallocStream = nullptr;
  };

  for (auto& kernel : compiledSeg.subKernels) {
    if (kernel.useIndirectArgs && !useConsolidatedArgTable) {
      size_t tableBytes = kernel.argSlotMapping.size() * sizeof(int64_t);
      if (tableBytes == 0) tableBytes = sizeof(int64_t);
      if (kernel.cachedArgTableDevice == nullptr ||
          kernel.cachedArgTableBytes < tableBytes ||
          kernel.cachedArgTableDeviceId != compileDevice) {
        if (kernel.cachedArgTableDevice != nullptr) {
          // Use synchronous cudaFree for stale buffers — they were allocated on a
          // previous compile's preallocStream which may have been destroyed since then.
          // cudaFreeAsync requires the stream to be alive; cudaFree always works
          // (CUDA docs: "Memory allocated by cudaMallocAsync can be freed by cudaFree").
          auto freeErr = cudaFree(kernel.cachedArgTableDevice);
          if (freeErr != cudaSuccess) {
            DSP_DIAG(MEMORY, "TritonGraphBackend: failed freeing stale arg table for sub-kernel [%d-%d]: %s",
                      kernel.startSlot_, kernel.endSlot_, cudaGetErrorString(freeErr));
            cudaGetLastError();
            cleanupCompiledWorkspace();
            return false;
          }
          recordModuleFree(kernel.cachedArgTableDeviceId >= 0 ? kernel.cachedArgTableDeviceId : compileDevice,
                           kernel.cachedArgTableBytes);
          kernel.cachedArgTableDevice = nullptr;
          kernel.cachedArgTableBytes = 0;
          kernel.cachedArgTableDeviceId = -1;
        }
        auto allocErr = allocateDeviceBufferAsync(&kernel.cachedArgTableDevice, tableBytes, preallocStream);
        if (allocErr != cudaSuccess) {
          DSP_DIAG(MEMORY, "TritonGraphBackend: failed pre-allocating indirect arg table (%zu bytes) for sub-kernel [%d-%d]: %s",
                    tableBytes, kernel.startSlot_, kernel.endSlot_, cudaGetErrorString(allocErr));
          cudaGetLastError();
          cleanupCompiledWorkspace();
          return false;
        }
        kernel.cachedArgTableBytes = tableBytes;
        kernel.cachedArgTableDeviceId = compileDevice;
        recordModuleAlloc(compileDevice, tableBytes);
      }

      // Pre-allocate the pinned host buffer alongside the device buffer.
      // CUDA graph capture requires both to be pre-allocated — cudaMallocHost
      // is a synchronous API that poisons the capture stream if called during
      // graph capture, causing endCapture to fail with "previous error."
      if (kernel.cachedArgTableHostPinned == nullptr ||
          kernel.cachedArgTableHostPinnedBytes < tableBytes) {
        auto& memPool = sd::memory::CudaMemoryPool::getInstance();
        if (kernel.cachedArgTableHostPinned != nullptr) {
          if (!kernel.cachedArgTableCaptureOwned) {
            memPool.freePinnedHost(kernel.cachedArgTableHostPinned);
          }
          kernel.cachedArgTableHostPinned = nullptr;
          kernel.cachedArgTableHostPinnedBytes = 0;
          kernel.cachedArgTableCaptureOwned = false;
        }
        kernel.cachedArgTableHostPinned = static_cast<char*>(memPool.allocatePinnedHost(tableBytes));
        if (kernel.cachedArgTableHostPinned == nullptr) {
          DSP_DIAG(MEMORY, "TritonGraphBackend: failed pre-allocating pinned arg table host (%zu bytes) for sub-kernel [%d-%d]",
                    tableBytes, kernel.startSlot_, kernel.endSlot_);
          cleanupCompiledWorkspace();
          return false;
        }
        kernel.cachedArgTableHostPinnedBytes = tableBytes;
      }
    }

    if (kernel.useCooperativeLaunch) {
      if (kernel.cachedSyncCounterDevice == nullptr ||
          kernel.cachedSyncCounterDeviceId != compileDevice) {
        if (kernel.cachedSyncCounterDevice != nullptr) {
          // Use synchronous cudaFree — stale buffer from a previous compile's stream.
          auto freeErr = cudaFree(kernel.cachedSyncCounterDevice);
          if (freeErr != cudaSuccess) {
            DSP_DIAG(MEMORY, "TritonGraphBackend: failed freeing stale sync counter for sub-kernel [%d-%d]: %s",
                      kernel.startSlot_, kernel.endSlot_, cudaGetErrorString(freeErr));
            cudaGetLastError();
            cleanupCompiledWorkspace();
            return false;
          }
          recordModuleFree(kernel.cachedSyncCounterDeviceId >= 0 ? kernel.cachedSyncCounterDeviceId : compileDevice,
                           sizeof(int));
          kernel.cachedSyncCounterDevice = nullptr;
          kernel.cachedSyncCounterDeviceId = -1;
        }
        auto allocErr = allocateDeviceBufferAsync(&kernel.cachedSyncCounterDevice, sizeof(int), preallocStream);
        if (allocErr != cudaSuccess) {
          DSP_DIAG(MEMORY, "TritonGraphBackend: failed pre-allocating cooperative sync counter for sub-kernel [%d-%d]: %s",
                    kernel.startSlot_, kernel.endSlot_, cudaGetErrorString(allocErr));
          cudaGetLastError();
          cleanupCompiledWorkspace();
          return false;
        }
        kernel.cachedSyncCounterDeviceId = compileDevice;
        recordModuleAlloc(compileDevice, sizeof(int));
      }
    }

    // Pre-allocate Triton 3.6 global scratch buffer (per-program scratch memory).
    // Must be done OUTSIDE capture -- raw cudaMallocAsync/cudaFreeAsync during capture
    // creates MemAlloc/MemFree graph nodes with addresses that become stale on replay,
    // causing SIGSEGV on cudaGraphLaunch.
    if (kernel.globalScratchBytes > 0) {
      unsigned int numPrograms = std::max(1u, kernel.gridX) *
                                  std::max(1u, kernel.gridY) *
                                  std::max(1u, kernel.gridZ);
      size_t totalScratchBytes = static_cast<size_t>(kernel.globalScratchBytes) * numPrograms;
      if (kernel.cachedGlobalScratchDevice == nullptr ||
          kernel.cachedGlobalScratchBytes < totalScratchBytes ||
          kernel.cachedGlobalScratchDeviceId != compileDevice) {
        if (kernel.cachedGlobalScratchDevice != nullptr) {
          // Use synchronous cudaFree — stale buffer from a previous compile's stream.
          auto freeErr = cudaFree(kernel.cachedGlobalScratchDevice);
          if (freeErr != cudaSuccess) {
            DSP_DIAG(MEMORY, "TritonGraphBackend: failed freeing stale global scratch for sub-kernel [%d-%d]: %s",
                      kernel.startSlot_, kernel.endSlot_, cudaGetErrorString(freeErr));
            cudaGetLastError();
            cleanupCompiledWorkspace();
            return false;
          }
          recordModuleFree(kernel.cachedGlobalScratchDeviceId >= 0 ? kernel.cachedGlobalScratchDeviceId : compileDevice,
                           kernel.cachedGlobalScratchBytes);
          kernel.cachedGlobalScratchDevice = nullptr;
          kernel.cachedGlobalScratchBytes = 0;
          kernel.cachedGlobalScratchDeviceId = -1;
        }
        auto allocErr = allocateDeviceBufferAsync(&kernel.cachedGlobalScratchDevice,
                                                   totalScratchBytes, preallocStream);
        if (allocErr != cudaSuccess) {
          DSP_DIAG(MEMORY, "TritonGraphBackend: failed pre-allocating global scratch (%zu bytes) for sub-kernel [%d-%d]: %s",
                    totalScratchBytes, kernel.startSlot_, kernel.endSlot_, cudaGetErrorString(allocErr));
          cudaGetLastError();
          cleanupCompiledWorkspace();
          return false;
        }
        kernel.cachedGlobalScratchBytes = totalScratchBytes;
        kernel.cachedGlobalScratchDeviceId = compileDevice;
        recordModuleAlloc(compileDevice, totalScratchBytes);
      }
    }
  }

  // Consolidated arg table: allocate one large buffer for all indirect-args sub-kernels.
  // Each kernel gets an offset into this single buffer, replacing N individual H2D copies
  // with one copy during graph capture (N fewer graph nodes).
  if (useConsolidatedArgTable) {
    size_t totalArgTableBytes = 0;
    compiledSeg.consolidatedArgTableOffsets.resize(compiledSeg.subKernels.size(), 0);
    for (size_t ki = 0; ki < compiledSeg.subKernels.size(); ki++) {
      auto& kernel = compiledSeg.subKernels[ki];
      if (kernel.useIndirectArgs) {
        // Align each sub-kernel's offset to 256 bytes for GPU cache line efficiency
        size_t align = 256;
        totalArgTableBytes = (totalArgTableBytes + align - 1) & ~(align - 1);
        compiledSeg.consolidatedArgTableOffsets[ki] = totalArgTableBytes;
        size_t kernelTableBytes = kernel.argSlotMapping.size() * sizeof(int64_t);
        if (kernelTableBytes == 0) kernelTableBytes = sizeof(int64_t);
        totalArgTableBytes += kernelTableBytes;
      }
    }

    if (totalArgTableBytes > 0) {
      // Allocate consolidated device buffer
      auto allocErr = allocateDeviceBufferAsync(&compiledSeg.consolidatedArgTableDevice,
                                                  totalArgTableBytes, preallocStream);
      if (allocErr != cudaSuccess) {
        DSP_DIAG(MEMORY, "TritonGraphBackend: failed allocating consolidated arg table (%zu bytes): %s",
                  totalArgTableBytes, cudaGetErrorString(allocErr));
        cudaGetLastError();
        cleanupCompiledWorkspace();
        return false;
      }
      compiledSeg.consolidatedArgTableBytes = totalArgTableBytes;
      compiledSeg.consolidatedArgTableDeviceId = compileDevice;
      recordModuleAlloc(compileDevice, totalArgTableBytes);
      // Allocate consolidated pinned host buffer via pool tracking
      auto& memPool = sd::memory::CudaMemoryPool::getInstance();
      compiledSeg.consolidatedArgTableHostPinned = static_cast<char*>(memPool.allocatePinnedHost(totalArgTableBytes));
      if (compiledSeg.consolidatedArgTableHostPinned == nullptr) {
        DSP_DIAG(MEMORY, "TritonGraphBackend: failed allocating consolidated pinned arg table (%zu bytes) via pool",
                  totalArgTableBytes);
        cudaGetLastError();
        cleanupCompiledWorkspace();
        return false;
      }
      compiledSeg.useConsolidatedArgTable = true;

      // Point each sub-kernel's cachedArgTableDevice to its offset in the consolidated buffer
      for (size_t ki = 0; ki < compiledSeg.subKernels.size(); ki++) {
        auto& kernel = compiledSeg.subKernels[ki];
        if (kernel.useIndirectArgs) {
          size_t offset = compiledSeg.consolidatedArgTableOffsets[ki];
          kernel.cachedArgTableDevice = static_cast<char*>(compiledSeg.consolidatedArgTableDevice) + offset;
          kernel.cachedArgTableDeviceId = compileDevice;
          size_t kernelTableBytes = kernel.argSlotMapping.size() * sizeof(int64_t);
          if (kernelTableBytes == 0) kernelTableBytes = sizeof(int64_t);
          kernel.cachedArgTableBytes = kernelTableBytes;
          // Point host pinned to offset in consolidated buffer too
          kernel.cachedArgTableHostPinned = static_cast<char*>(compiledSeg.consolidatedArgTableHostPinned) + offset;
          kernel.cachedArgTableHostPinnedBytes = kernelTableBytes;
        }
      }

      DSP_DIAG(COMPILE, "TritonGraphBackend: consolidated arg table %zu bytes for %d sub-kernels in [%d-%d]",
                totalArgTableBytes, static_cast<int>(compiledSeg.subKernels.size()),
                seg.def.startSlot, seg.def.endSlot);
    }
  }

  // Dirty tracking: classify each sub-kernel's args as static vs dynamic.
  // Must pass slots[] so the classification uses wiring.outputSlotIndices
  // (global output slot indices) rather than the op-slot range — these
  // numbering spaces diverge in multi-output/wide ops, causing intermediates
  // to be misclassified as static weights if only the op range is checked.
  if (Environment::getInstance().tritonArgDirtyTracking()) {
    compiledSeg.classifyDirtyTracking(slots, seg.def.startSlot, seg.def.endSlot);
    DSP_DIAG(COMPILE, "TritonGraphBackend: dirty tracking: %d/%d sub-kernels have static-only args (skip refresh)",
              compiledSeg.countStaticSubKernels(), static_cast<int>(compiledSeg.subKernels.size()));
  }

  cudaEvent_t preallocReadyEvent = nullptr;
  auto preallocEventCreateErr = cudaEventCreateWithFlags(&preallocReadyEvent, cudaEventDisableTiming);
  if (preallocEventCreateErr != cudaSuccess) {
    DSP_DIAG(COMPILE, "TritonGraphBackend: pre-allocation ready event create failed for segment [%d-%d]: %s",
              seg.def.startSlot, seg.def.endSlot, cudaGetErrorString(preallocEventCreateErr));
    cudaGetLastError();
    cleanupCompiledWorkspace();
    return false;
  }
  auto preallocEventRecordErr = cudaEventRecord(preallocReadyEvent, preallocStream);
  if (preallocEventRecordErr != cudaSuccess) {
    DSP_DIAG(COMPILE, "TritonGraphBackend: pre-allocation ready event record failed for segment [%d-%d]: %s",
              seg.def.startSlot, seg.def.endSlot, cudaGetErrorString(preallocEventRecordErr));
    cudaGetLastError();
    cudaEventDestroy(preallocReadyEvent);
    cleanupCompiledWorkspace();
    return false;
  }
  compiledSeg.preallocReadyEvent = preallocReadyEvent;
  if (ownPreallocStream && preallocStream != nullptr) {
    auto preallocDestroyErr = cudaStreamDestroy(preallocStream);
	    if (preallocDestroyErr != cudaSuccess) {
	      DSP_DIAG(COMPILE, "TritonGraphBackend: failed destroying pre-allocation stream for segment [%d-%d]: %s",
	                seg.def.startSlot, seg.def.endSlot, cudaGetErrorString(preallocDestroyErr));
	      cudaGetLastError();
	      cudaEventDestroy(preallocReadyEvent);
	      compiledSeg.preallocReadyEvent = nullptr;
	      cleanupCompiledWorkspace();
	      return false;
	    }
  }

  lastCompilationAudit_ = compiledSeg.audit;
  const int compiledKernelCount = static_cast<int>(compiledSeg.subKernels.size());

  // Store dtype hash in the segment so lookup sites without outputSlots can
  // reconstruct the exact SegmentCacheKey via CompiledSegment::segInternalDtypeHash.
  compiledSeg.segInternalDtypeHash = cacheDtypeHash;

  // Stable references to register with ModuleResidencyCache after the move.
  CompiledSegment* installedSeg = nullptr;
  {
    std::lock_guard<std::mutex> lock(cacheMtx_);
    failedCache_.erase(key);
    cache_[key] = std::move(compiledSeg);
    installedSeg = &cache_[key];
    // Populate secondary dtype index so lookup sites without outputSlots can
    // find the right dtype hash for this live segment instance.
    dtypeIndex_[DtypeIndexKey{seg.def.startSlot, seg.def.endSlot, shapeKey,
                              compileDevice, &seg}] = cacheDtypeHash;
  }

  // Register each freshly-installed sub-kernel with the residency cache.
  // This MUST happen after the std::move into cache_ so the pointer is stable
  // for the lifetime of the cache entry — the per-device tracking list stores
  // raw CompiledKernel* and would dangle if we registered the moved-from copy.
  if (installedSeg != nullptr) {
    for (auto& kernel : installedSeg->subKernels) {
      if (kernel.gpuModule == nullptr) continue;
      const int kernDev = kernel.loadedDeviceId >= 0 ? kernel.loadedDeviceId : compileDevice;
      registerLoadedKernel(&kernel, kernDev);
    }
  }

  DSP_DIAG(COMPILE, "TritonGraphBackend: compiled segment [%d-%d] (%d sub-kernels, shape key %lld, deviceId=%d)",
            seg.def.startSlot, seg.def.endSlot, compiledKernelCount, shapeKey, compileDevice);
  return true;
}

}  // namespace graph
}  // namespace sd

#endif // HAVE_TRITON

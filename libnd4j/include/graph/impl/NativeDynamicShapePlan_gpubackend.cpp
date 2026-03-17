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

// GPU graph backend (Triton/NVRTC/PTX) execution methods.
//
// Contains getGpuGraphBackend() which selects the best available GPU compiler
// backend (Triton > NVRTC > PTX) based on the configured GraphExecutionMode,
// and executeSegmentWithGpuGraph() which drives segment compilation, CUDA graph
// capture/replay for Triton fused kernels, and fallback orchestration.

#include <graph/NativeDynamicShapePlan.h>
#include <graph/DspDiagnostics.h>
#include <graph/DspVerifyUtils.h>
#include <helpers/MmulHelper.h>
#include <system/Environment.h>
#include <config.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <unordered_map>
#include <unordered_set>

// GPU graph backends (conditional)
#if HAVE_TRITON
#include <graph/gpu/TritonGraphBackend.h>
#endif
#ifdef SD_CUDA
#include <graph/gpu/NvrtcGraphBackend.h>
#include <graph/gpu/PtxGraphBackend.h>
#include <graph/gpu/CaptureBufferRegistry.h>
#include <graph/cuda/CudaGraphReplayHandle.h>
#endif

namespace sd {
namespace graph {

// Local helper: convert Status enum to human-readable string for diagnostics.
static const char* statusName_gpu(Status status) {
  switch (status) {
    case Status::OK: return "OK";
    case Status::BAD_INPUT: return "BAD_INPUT";
    case Status::BAD_SHAPE: return "BAD_SHAPE";
    case Status::BAD_RANK: return "BAD_RANK";
    case Status::BAD_PARAMS: return "BAD_PARAMS";
    case Status::BAD_OUTPUT: return "BAD_OUTPUT";
    case Status::BAD_RNG: return "BAD_RNG";
    case Status::BAD_EPSILON: return "BAD_EPSILON";
    case Status::BAD_GRADIENTS: return "BAD_GRADIENTS";
    case Status::BAD_BIAS: return "BAD_BIAS";
    case Status::VALIDATION: return "VALIDATION";
    case Status::BAD_GRAPH: return "BAD_GRAPH";
    case Status::BAD_LENGTH: return "BAD_LENGTH";
    case Status::BAD_DIMENSIONS: return "BAD_DIMENSIONS";
    case Status::BAD_ORDER: return "BAD_ORDER";
    case Status::BAD_ARGUMENTS: return "BAD_ARGUMENTS";
    case Status::DOUBLE_WRITE: return "DOUBLE_WRITE";
    case Status::DOUBLE_READ: return "DOUBLE_READ";
    case Status::KERNEL_FAILURE: return "KERNEL_FAILURE";
    case Status::EQ_TRUE: return "EQ_TRUE";
    case Status::EQ_FALSE: return "EQ_FALSE";
    case Status::MAYBE: return "MAYBE";
    default: return "UNKNOWN";
  }
}

// Helper: extract specialBuffer() device addresses from NDArray** into void** for
// address snapshot diagnostics. Thread-local to avoid repeated allocation.
static void extractDeviceAddrs(NDArray** arrays, int count, std::vector<void*>& out) {
  out.resize(count);
  for (int i = 0; i < count; i++) {
    out[i] = (arrays != nullptr && arrays[i] != nullptr)
             ? arrays[i]->specialBuffer() : nullptr;
  }
}

// Strict mode: fail fast instead of silently degrading to slot-by-slot.
static bool isStrictNoFallbackMode_gpu(GraphExecutionMode mode) {
  return mode == GraphExecutionMode::GEM_TRITON;
}

// ─── DSP Verify Helpers ────────────────────────────────────────────────────

// Source type name for diagnostics
static const char* sourceTypeName(int8_t st) {
  switch (static_cast<NativeSourceType>(st)) {
    case SOURCE_CONSTANT: return "CONSTANT";
    case SOURCE_VARIABLE: return "VARIABLE";
    case SOURCE_PLACEHOLDER: return "PLACEHOLDER";
    case SOURCE_OP_OUTPUT: return "OP_OUTPUT";
    default: return "UNKNOWN";
  }
}

#ifdef SD_CUDA
// Templated helpers in DspVerifyUtils.h (dspVerifyCopyValues, dspMaxDiff, dspFormatValues, etc.)
#endif  // SD_CUDA

void NativeDynamicShapePlan::clearGpuBackendFailedCache() {
#if HAVE_TRITON
  TritonGraphBackend::getInstance().clearFailedSegmentCache();
#endif
}

GraphBackend* NativeDynamicShapePlan::getGpuGraphBackend() {
  if (gpuGraphBackendChecked_) return gpuGraphBackend_;
  gpuGraphBackendChecked_ = true;

  // If a specific backend is forced via setGraphExecutionMode(), use only that one.
  // SLOT_BY_SLOT and CUDA_GRAPHS don't use a GPU graph backend.
  if (graphExecutionMode_ == GraphExecutionMode::GEM_SLOT_BY_SLOT ||
      graphExecutionMode_ == GraphExecutionMode::GEM_CUDA_GRAPHS) {
    gpuGraphBackend_ = nullptr;
    return nullptr;
  }

#if HAVE_TRITON
  if (graphExecutionMode_ == GraphExecutionMode::GEM_TRITON ||
      graphExecutionMode_ == GraphExecutionMode::GEM_AUTO) {
    auto& triton = TritonGraphBackend::getInstance();
    if (triton.isAvailable()) {
      gpuGraphBackend_ = &triton;
      DSP_DIAG(BACKEND, "using Triton GPU compiler backend");
      return gpuGraphBackend_;
    }
    if (graphExecutionMode_ == GraphExecutionMode::GEM_TRITON) {
      DSP_DIAG(BACKEND, "Triton backend requested but not available");
      gpuGraphBackend_ = nullptr;
      return nullptr;
    }
    DSP_DIAG(BACKEND, "Triton unavailable in AUTO mode, trying NVRTC/PTX backends");
  }
#else
  if (graphExecutionMode_ == GraphExecutionMode::GEM_TRITON) {
    DSP_DIAG(BACKEND, "Triton backend requested but not compiled (HAVE_TRITON=0)");
    gpuGraphBackend_ = nullptr;
    return nullptr;
  }
  if (graphExecutionMode_ == GraphExecutionMode::GEM_AUTO) {
    DSP_DIAG(BACKEND, "Triton not compiled (HAVE_TRITON=0); AUTO mode will try NVRTC/PTX/CUDA graphs");
  }
#endif

#ifdef SD_CUDA
  if (graphExecutionMode_ == GraphExecutionMode::GEM_NVRTC_JIT ||
      graphExecutionMode_ == GraphExecutionMode::GEM_AUTO) {
    auto& nvrtc = NvrtcGraphBackend::getInstance();
    if (nvrtc.isAvailable()) {
      gpuGraphBackend_ = &nvrtc;
      DSP_DIAG(BACKEND, "using NVRTC GPU compiler backend");
      return gpuGraphBackend_;
    }
    if (graphExecutionMode_ == GraphExecutionMode::GEM_NVRTC_JIT) {
      DSP_DIAG(BACKEND, "NVRTC backend requested but not available");
      gpuGraphBackend_ = nullptr;
      return nullptr;
    }
  }

  if (graphExecutionMode_ == GraphExecutionMode::GEM_PTX_JIT ||
      graphExecutionMode_ == GraphExecutionMode::GEM_AUTO) {
    auto& ptx = PtxGraphBackend::getInstance();
    if (ptx.isAvailable()) {
      gpuGraphBackend_ = &ptx;
      DSP_DIAG(BACKEND, "using PTX template GPU compiler backend");
      return gpuGraphBackend_;
    }
    if (graphExecutionMode_ == GraphExecutionMode::GEM_PTX_JIT) {
      DSP_DIAG(BACKEND, "PTX backend requested but not available");
      gpuGraphBackend_ = nullptr;
      return nullptr;
    }
  }
#endif

  gpuGraphBackend_ = nullptr;
  return nullptr;
}

Status NativeDynamicShapePlan::executeSegmentWithGpuGraph(
    GraphSegment& seg, NDArray** externalArrays, int numExt, void* stream) {

#ifdef SD_CUDA
  // ── Segment lifecycle: SEG_ENTER ──────────────────────────────────────
  if (Environment::getInstance().tritonVerifyKernels()) {
    // Ensure VERIFY diagnostic category is enabled and output level is FULL
    // when tritonVerifyKernels is on (may be set at runtime via Java, after
    // DspDiagnostics constructor)
    if (!DSP_DIAG_ENABLED(VERIFY)) {
      sd::graph::DspDiagnostics::getInstance().enableCategories(sd::graph::DSP_DIAG_VERIFY);
      sd::graph::DspDiagnostics::getInstance().setLevel(sd::graph::DSP_LEVEL_FULL);
    }
    const char* mode = "unknown";
    if (seg.executionCount == 0) mode = "warmup";
    else if (seg.executionCount == 1) mode = "compile";
    else if (seg.replayHandle != nullptr) mode = "replay";
    else if (seg.captureFailed) mode = "slot-by-slot";
    else mode = "capture";
    DSP_DIAG(VERIFY, "SEG_ENTER seg[%d-%d] execCount=%d mode=%s",
              seg.startSlot, seg.endSlot, seg.executionCount, mode);
    // Dump external input actuality flags for first N inputs
    int dumpCount = std::min(numExt, 8);
    for (int i = 0; i < dumpCount; i++) {
      if (externalArrays[i] != nullptr && externalArrays[i]->dataBuffer() != nullptr) {
        auto* db = externalArrays[i]->dataBuffer();
        DSP_DIAG(VERIFY, "  EXT_INPUT[%d] dtype=%s len=%lld pAct=%d sAct=%d addr=%p",
                  i, DataTypeUtils::asString(externalArrays[i]->dataType()).c_str(),
                  (long long)externalArrays[i]->lengthOf(),
                  db->isPrimaryActual() ? 1 : 0, db->isSpecialActual() ? 1 : 0,
                  externalArrays[i]->specialBuffer());
      }
    }
    if (numExt > 8) {
      DSP_DIAG(VERIFY, "  ... and %d more external inputs", numExt - 8);
    }
  }
#endif

  auto* backend = getGpuGraphBackend();
  if (backend == nullptr) {
    DSP_DIAG(BACKEND, "executeSegmentWithGpuGraph: no GPU backend selected for seg[%d-%d]",
              seg.startSlot, seg.endSlot);
    return Status::KERNEL_FAILURE;
  }
  const char* backendName = backend->name();

  // If compilation previously failed validation, never try again
  if (seg.captureFailed) {
    return Status::KERNEL_FAILURE;
  }

  // Check if this segment can be compiled by the selected GPU backend
  if (!backend->canFuseSegment(slots_, seg.startSlot, seg.endSlot)) {
    DSP_DIAG(BACKEND, "executeSegmentWithGpuGraph: backend=%s cannot fuse seg[%d-%d]",
              backendName, seg.startSlot, seg.endSlot);
    return Status::KERNEL_FAILURE;  // Caller will fall back to CUDA Graphs
  }

  // First execution: run slot-by-slot warmup BEFORE compilation.
  if (seg.executionCount == 0) {
#ifdef SD_CUDA
    // ── Plan structure dump (one-time, on first segment execution) ─────────
    if (Environment::getInstance().tritonVerifyKernels()) {
      DSP_DIAG(VERIFY, "=== PLAN STRUCTURE ===");
      DSP_DIAG(VERIFY, "Plan: %d steps, %d output slots, %d external inputs, %d segments",
                numSlots_, totalOutputSlots_, numExternalInputs_, (int)segments_.size());
      for (int si = 0; si < (int)segments_.size(); si++) {
        auto& s = segments_[si];
        DSP_DIAG(VERIFY, "Segment %d: slots [%d..%d] (%d ops) %s",
                  si, s.startSlot, s.endSlot, s.endSlot - s.startSlot + 1,
                  s.isCapturable ? "capturable" : "non-capturable");
      }
      // Per-step wiring
      std::unordered_map<std::string, int> opHistogram;
      for (int s = 0; s < numSlots_; s++) {
        auto& sl = slots_[s];
        opHistogram[sl.opName]++;
        // Build input description
        std::string inputsStr;
        for (int i = 0; i < sl.numInputs; i++) {
          if (i > 0) inputsStr += ", ";
          int srcIdx = sl.inputSourceIndices[i];
          if (srcIdx >= 0) {
            inputsStr += "slot#" + std::to_string(srcIdx);
          } else {
            int extIdx = -(srcIdx + 1);
            inputsStr += "ext#" + std::to_string(extIdx);
            if (extIdx < (int)externalInputNames_.size() && !externalInputNames_[extIdx].empty()) {
              inputsStr += ":\"" + externalInputNames_[extIdx] + "\"";
            }
            if (sl.inputSourceTypes != nullptr) {
              inputsStr += ":";
              inputsStr += sourceTypeName(sl.inputSourceTypes[i]);
            }
          }
        }
        // Build output description
        std::string outputsStr;
        for (int i = 0; i < sl.numOutputs; i++) {
          if (i > 0) outputsStr += ",";
          outputsStr += std::to_string(sl.outputSlotIndices[i]);
        }
        DSP_DIAG(VERIFY, "STEP %4d: %-20s inputs:[%s] -> outputs:[%s]%s%s%s",
                  s, sl.opName.c_str(), inputsStr.c_str(), outputsStr.c_str(),
                  sl.isIdentityOp ? " [IDENTITY]" : "",
                  sl.frozenConstantSlot ? " [FROZEN]" : "",
                  sl.isFusedChainTail ? " [FUSED_TAIL]" : "");
      }
      // Op histogram
      std::string histStr;
      std::vector<std::pair<std::string, int>> sorted(opHistogram.begin(), opHistogram.end());
      std::sort(sorted.begin(), sorted.end(),
                [](const auto& a, const auto& b) { return b.second < a.second; });
      for (auto& p : sorted) {
        if (!histStr.empty()) histStr += ", ";
        histStr += p.first + "=" + std::to_string(p.second);
      }
      DSP_DIAG(VERIFY, "Op histogram: %s", histStr.c_str());
      DSP_DIAG(VERIFY, "=== END PLAN STRUCTURE ===");
    }
#endif

    auto warmupStatus = executeSegmentSlotBySlot(seg, externalArrays, numExt, stream);
    if (warmupStatus == Status::OK) {
      seg.executionCount++;
    }
    return warmupStatus;
  }

  // Compute shape key for cache lookup.
  // When shapes are frozen and the key was already computed, reuse it — the shapes
  // cannot change so the hash is stable. Saves iterating all cross-segment inputs.
  LongType segShapeKey;
  if (shapesFrozen_ && seg.cachedShapeKey != 0) {
    segShapeKey = seg.cachedShapeKey;
  } else {
    segShapeKey = computeSegmentShapeKey(seg, externalArrays, numExt);
  }

  // Diagnostic: scan all slotArrayCache_ entries for freed DataBuffers.
  // Java may have closed DSP output arrays between steps (e.g., prefill KV outputs via
  // setCloseable(true)+close()), deleting the C++ NDArray and leaving dangling pointers.
  //
  // OPTIMIZATION: Skip this scan when shapes are frozen and we have a captured graph.
  // In steady-state decode, no Java code closes DSP output arrays — the static KV buffers
  // and frozen slots are stable. The scan is only needed during dynamic-shape execution
  // (prefill, transitions) when arrays may be freed between steps.
  bool skipStaleBufferScan = shapesFrozen_ && seg.executionCount > 2;
  if (!skipStaleBufferScan) {
    int invalidCount = 0;
    for (int si = seg.startSlot; si <= seg.endSlot && si < totalOutputSlots_; si++) {
      NDArray* cached = slotArrayCache_[si];
      if (cached != nullptr && !cached->isEmpty()) {
        auto* db = cached->dataBuffer();
        if (db == nullptr || !db->isValid()) {
          DSP_DIAG_SLOT(MEMORY, si, "STALE slotArrayCache_[%d] detected "
                    "(arr=%p, db=%p, dbValid=%d, frozenConst=%d). Invalidating.",
                    si, (void*)cached, (void*)db, db ? (db->isValid() ? 1 : 0) : -1,
                    slots_[si].frozenConstantSlot ? 1 : 0);
          slotArrayCache_[si] = nullptr;
          if (outputSlots_[si] == cached) outputSlots_[si] = nullptr;
          if (si < numSlots_ && slots_[si].frozenConstantSlot) {
            slots_[si].frozenConstantSlot = false;
          }
          invalidCount++;
        }
      }
    }
    for (int ei = 0; ei < numExt; ei++) {
      NDArray* ext = externalArrays[ei];
      if (ext != nullptr && !ext->isEmpty()) {
        auto* db = ext->dataBuffer();
        if (db == nullptr || !db->isValid()) {
          DSP_DIAG(MEMORY, "STALE externalInput[%d] detected "
                    "(arr=%p, db=%p, dbValid=%d)",
                    ei, (void*)ext, (void*)db, db ? (db->isValid() ? 1 : 0) : -1);
          invalidCount++;
        }
      }
    }
    if (invalidCount > 0) {
      DSP_DIAG(MEMORY, "executeSegmentWithGpuGraph: found %d stale entries in slot/external arrays",
                invalidCount);
#ifdef SD_CUDA
      seg.replayHandle.reset();
      seg.argTableStable = false;
      batchD2DCount_ = 0;
      seg.cachedShapeKey = 0;
#endif
      seg.captureFailed = false;
      DSP_DIAG(FALLBACK, "invalidated graph for seg[%d-%d] "
                "due to %d stale entries - executing slot-by-slot this step",
                seg.startSlot, seg.endSlot, invalidCount);
      return executeSegmentSlotBySlot(seg, externalArrays, numExt, stream);
    }
  }

  // Pre-execution: ensure all output slots in the segment have live arrays.
  // The Triton kernel's arg mapping references outputSlots_ for both inputs
  // (from prior ops) and outputs (to write results). Slot-by-slot warmup may
  // have released intermediate arrays via releaseAtStep_, leaving entries null.
  // First restore from slotArrayCache_, then allocate any remaining nulls
  // using cached shape info from warmup.
  //
  // CRITICAL: This MUST happen BEFORE compilation. The compiler resolves
  // arg mappings from outputSlots_ — if intermediate slots are null (released
  // after warmup), the compiler omits them from the arg table, producing
  // sub-kernels with missing inputs that read stale/garbage data on first
  // execution. By populating all slots before compilation, the compiler sees
  // all arrays and builds correct arg mappings.
  //
  // IMPORTANT: Java may close() output arrays between execution steps (e.g.,
  // prefill KV outputs via setCloseable(true)+close()). This frees the underlying
  // DataBuffer while slotArrayCache_ still holds the NDArray*. Validate the
  // DataBuffer before reusing — invalidate entries pointing to freed buffers.
  //
  // CRITICAL: If any output slot within the segment is allocated at a NEW address
  // (different from capture time), the cached CUDA graph becomes invalid. Triton
  // arg tables are refreshed with new addresses, but native ops (cuBLAS matmul)
  // have addresses baked into the graph. This address inconsistency causes the
  // graph to read stale data from old addresses while Triton writes to new ones.
  // Track any new allocations and invalidate the graph if needed.
  //
  // OPTIMIZATION: Skip when shapes are frozen and we've already done this
  // restoration at least once (executionCount > 2). In steady-state decode,
  // outputSlots_ are stable — no arrays are released or freed between steps.
  int preExecAllocCount = 0;
  if (!(shapesFrozen_ && seg.executionCount > 2)) {
  for (int stepIdx = seg.startSlot; stepIdx <= seg.endSlot; stepIdx++) {
    NativeSlot& slot = slots_[stepIdx];
    // Restore input slot entries that may have been released
    for (int i = 0; i < slot.numInputs; i++) {
      int srcIdx = slot.inputSourceIndices[i];
      if (srcIdx >= 0 && srcIdx < totalOutputSlots_ &&
          outputSlots_[srcIdx] == nullptr && slotArrayCache_[srcIdx] != nullptr &&
          !slotArrayCache_[srcIdx]->isEmpty()) {
        // Validate DataBuffer before restoring — Java close() may have freed it
        auto* db = slotArrayCache_[srcIdx]->dataBuffer();
        if (db == nullptr || !db->isValid()) {
          slotArrayCache_[srcIdx] = nullptr;
          // Clear frozenConstantSlot on the SOURCE slot so its op re-executes
          if (srcIdx < numSlots_ && slots_[srcIdx].frozenConstantSlot) {
            slots_[srcIdx].frozenConstantSlot = false;
          }
        } else {
          outputSlots_[srcIdx] = slotArrayCache_[srcIdx];
          if (Environment::getInstance().tritonVerifyKernels()) {
            auto* arr = slotArrayCache_[srcIdx];
            DSP_DIAG(VERIFY, "SLOT_WRITE slot=%d tag=CACHE_RESTORE(input) dtype=%s len=%lld addr=%p",
                      srcIdx, DataTypeUtils::asString(arr->dataType()).c_str(),
                      (long long)arr->lengthOf(), arr->specialBuffer());
          }
        }
      }
    }
    // Restore or allocate output slot entries
    for (int i = 0; i < slot.numOutputs; i++) {
      int slotIdx = slot.outputSlotIndices[i];
      if (slotIdx < 0 || slotIdx >= totalOutputSlots_) continue;
      if (slotArrayCache_[slotIdx] != nullptr && !slotArrayCache_[slotIdx]->isEmpty()) {
        // Validate DataBuffer before restoring — Java close() may have freed it
        auto* db = slotArrayCache_[slotIdx]->dataBuffer();
        if (db == nullptr || !db->isValid()) {
          slotArrayCache_[slotIdx] = nullptr;
          // Clear frozenConstantSlot so the op re-executes instead of
          // returning OK with a null/stale output
          if (stepIdx < numSlots_ && slots_[stepIdx].frozenConstantSlot) {
            slots_[stepIdx].frozenConstantSlot = false;
          }
        } else {
          outputSlots_[slotIdx] = slotArrayCache_[slotIdx];
          if (Environment::getInstance().tritonVerifyKernels()) {
            auto* arr = slotArrayCache_[slotIdx];
            DSP_DIAG(VERIFY, "SLOT_WRITE slot=%d tag=CACHE_RESTORE(output) dtype=%s len=%lld addr=%p",
                      slotIdx, DataTypeUtils::asString(arr->dataType()).c_str(),
                      (long long)arr->lengthOf(), arr->specialBuffer());
          }
        }
      } else if (outputSlots_[slotIdx] == nullptr) {
        // Allocate from cached shape info (populated during warmup)
        const LongType* shapeInfo = nullptr;
        if (i < static_cast<int>(slot.cachedOutputShapes.size()) && slot.cachedOutputShapes[i]) {
          shapeInfo = slot.cachedOutputShapes[i];
        }
        // Fallback: for identity/view-like ops that don't cache output shapes,
        // derive the shape from the first input source's existing array
        if (!shapeInfo && slot.numInputs > 0) {
          int srcIdx = slot.inputSourceIndices[0];
          NDArray* srcArr = nullptr;
          if (srcIdx < 0) {
            int extIdx = -(srcIdx + 1);
            if (extIdx < numExt) srcArr = externalArrays[extIdx];
          } else if (srcIdx < totalOutputSlots_) {
            srcArr = outputSlots_[srcIdx];
            if (!srcArr && slotArrayCache_[srcIdx] != nullptr) {
              auto* db = slotArrayCache_[srcIdx]->dataBuffer();
              if (db != nullptr && db->isValid()) {
                srcArr = slotArrayCache_[srcIdx];
              } else {
                slotArrayCache_[srcIdx] = nullptr;
              }
            }
          }
          if (srcArr) shapeInfo = srcArr->shapeInfo();
        }
        if (shapeInfo) {
          auto dt = ArrayOptions::dataType(shapeInfo);
          auto order = shape::order(shapeInfo);
          LongType rank = shape::rank(shapeInfo);
          std::vector<LongType> shapeVec(rank);
          for (int d = 0; d < rank; d++) shapeVec[d] = shapeInfo[d + 1];
          auto* arr = new NDArray(order, shapeVec, dt);
          outputSlots_[slotIdx] = arr;
          slotArrayCache_[slotIdx] = arr;
          preExecAllocCount++;
          if (Environment::getInstance().tritonVerifyKernels()) {
            DSP_DIAG(VERIFY, "SLOT_WRITE slot=%d tag=ALLOC dtype=%s len=%lld addr=%p",
                      slotIdx, DataTypeUtils::asString(dt).c_str(),
                      (long long)arr->lengthOf(), arr->specialBuffer());
          }
        }
      }
    }
  }
  } // end if (!(shapesFrozen_ && executionCount > 2))

  // Compile once per stable shape; skip cache probe on steady-state replay.
  // This keeps the hot path focused on dispatch instead of repeated compile checks.
  // NOTE: Pre-exec output slot allocation above ensures all slots are populated
  // before the compiler resolves arg mappings. Without this ordering, intermediate
  // slots released after warmup are null and get omitted from the arg table,
  // causing sub-kernels to read stale data on their first execution.
  bool needsCompile = (seg.executionCount == 1) || (seg.shapeKey != segShapeKey);
  if (needsCompile) {
    if (!backend->compileSegment(seg, slots_, externalArrays, numExt,
                                 outputSlots_, totalOutputSlots_, segShapeKey,
                                 numSlots_)) {
      DSP_DIAG(COMPILE, "executeSegmentWithGpuGraph: backend=%s compile failed for seg[%d-%d]",
                backendName, seg.startSlot, seg.endSlot);
      return Status::KERNEL_FAILURE;
    }
  }

  // On first compilation, validate coverage
  if (seg.executionCount == 1) {
    auto audit = backend->getLastCompilationAudit();
    lastCompilationAudit_ = audit;
    int compiledCount = 0;
    int failedCount = 0;
    for (const auto& entry : audit) {
      if (entry.wasCompiled) {
        compiledCount++;
      } else {
        failedCount++;
        DSP_DIAG_SLOT(COMPILE, entry.slotIndex, "%s VALIDATION: slot %d (%s) was NOT compiled: %s",
                  backendName, entry.slotIndex, entry.opName.c_str(), entry.reason.c_str());
      }
    }
    if (compiledCount == 0 && failedCount > 0) {
      // All ops FAILED compilation — real failure.
      if (isStrictNoFallbackMode_gpu(graphExecutionMode_)) {
        DSP_DIAG(COMPILE, "%s VALIDATION FAILURE: segment [%d-%d] has zero compiled ops "
                  "(failed=%d). Forced backend mode prohibits fallback.",
                  backendName, seg.startSlot, seg.endSlot, failedCount);
      } else {
        DSP_DIAG(FALLBACK, "%s VALIDATION FAILURE: segment [%d-%d] has zero compiled ops "
                  "(failed=%d). Falling back to CUDA Graphs.",
                  backendName, seg.startSlot, seg.endSlot, failedCount);
      }
      seg.captureFailed = true;
      return Status::KERNEL_FAILURE;
    }
    if (compiledCount == 0 && failedCount == 0) {
      // All sections are intentional fallback (e.g., all non-elementwise).
      // The compiled segment has 0 sub-kernels; executeSegment will run
      // everything via fallbackRangeExecutor_.
      DSP_DIAG(FALLBACK, "%s: segment [%d-%d] has only fallback sections (no compilation needed). "
                "Will run entirely via slot-by-slot fallback.",
                backendName, seg.startSlot, seg.endSlot);
    }
    if (failedCount > 0) {
      if (isStrictNoFallbackMode_gpu(graphExecutionMode_)) {
        DSP_DIAG(COMPILE, "%s VALIDATION FAILURE: segment [%d-%d] partial compile detected "
                  "(compiled=%d failed=%d). Forced backend mode prohibits fallback.",
                  backendName, seg.startSlot, seg.endSlot, compiledCount, failedCount);
        seg.captureFailed = true;
        return Status::KERNEL_FAILURE;
      }
      DSP_DIAG(COMPILE, "%s VALIDATION: segment [%d-%d] partial compile accepted "
                "(compiled=%d failed=%d); failed ranges will run slot-by-slot.",
                backendName, seg.startSlot, seg.endSlot, compiledCount, failedCount);
    }
  }

  // Execute via selected GPU backend
  seg.shapeKey = segShapeKey;

#ifdef SD_CUDA
  cudaStream_t cudaStr = (stream != nullptr)
      ? *static_cast<cudaStream_t*>(stream) : nullptr;

  // If any output slots were re-allocated at new addresses, the cached CUDA graph
  // is invalid — native ops (cuBLAS) have the old addresses baked in while Triton
  // arg tables were refreshed with new addresses. Invalidate and re-capture.
  if (preExecAllocCount > 0 && seg.replayHandle != nullptr) {
    DSP_DIAG(EXECUTE, "GRAPH INVALIDATED: %d output slots re-allocated at new addresses "
              "(cache entries freed by Java). seg[%d-%d] will re-capture.",
              preExecAllocCount, seg.startSlot, seg.endSlot);
    for (auto* ptr : seg.replayHandle->getCapturedHostPtrs()) { cudaFreeHost(ptr); }
    seg.replayHandle->getCapturedHostPtrs().clear();
    seg.replayHandle->clearExternalAddresses();
    seg.replayHandle.reset();
    seg.argTableStable = false;
    batchD2DCount_ = 0;
    seg.capturedInputAddrKey = 0;
    // Reset execution count to trigger warmup→capture flow
    seg.executionCount = 0;
    seg.captureFailed = false;
  }

  bool allowTritonCudaGraphReplay = Environment::getInstance().tritonGraphCapture() &&
                                    shapesFrozen_;

  // BLATANT DIAGNOSTIC: Log the capture decision factors
  int captureMinExec = Environment::getInstance().tritonCaptureMinExec();
  bool forceRecaptureEnabled = Environment::getInstance().tritonForceRecapture();
  bool hasReplayHandle = (seg.replayHandle != nullptr);
  bool hasCaptureBuffers = hasReplayHandle && !seg.replayHandle->getCaptureBuffers().empty();
  bool replayHandleNull = (seg.replayHandle == nullptr);
  bool notCaptureFailed = !seg.captureFailed;
  bool execCountInWindow = (seg.executionCount >= captureMinExec) && 
                           (forceRecaptureEnabled || seg.executionCount <= (captureMinExec + 2));
  bool hasCudaStream = (cudaStr != nullptr);
  
  DSP_DIAG(EXECUTE, "=== CAPTURE DECISION CHECK seg[%d-%d] ===", seg.startSlot, seg.endSlot);
  DSP_DIAG(EXECUTE, "  tritonGraphCapture()=%d, shapesFrozen_=%d => allowTritonCudaGraphReplay=%d",
           Environment::getInstance().tritonGraphCapture() ? 1 : 0,
           shapesFrozen_ ? 1 : 0, allowTritonCudaGraphReplay ? 1 : 0);
  DSP_DIAG(EXECUTE, "  seg.executionCount=%d, captureMinExec=%d, window=[%d,%d], inWindow=%d",
           seg.executionCount, captureMinExec, captureMinExec, captureMinExec + 2,
           execCountInWindow ? 1 : 0);
  DSP_DIAG(EXECUTE, "  hasReplayHandle=%d, hasCaptureBuffers=%d, replayHandleNull=%d",
           hasReplayHandle ? 1 : 0, hasCaptureBuffers ? 1 : 0, replayHandleNull ? 1 : 0);
  DSP_DIAG(EXECUTE, "  captureFailed=%d, cudaStr!=nullptr=%d",
           seg.captureFailed ? 1 : 0, hasCudaStream ? 1 : 0);
  
  bool shouldCaptureTritonGraph = allowTritonCudaGraphReplay &&
                                  (!hasReplayHandle || !hasCaptureBuffers) &&
                                  replayHandleNull &&
                                  notCaptureFailed &&
                                  execCountInWindow &&
                                  hasCudaStream;
  
  DSP_DIAG(EXECUTE, "  => shouldCaptureTritonGraph=%d", shouldCaptureTritonGraph ? 1 : 0);
  if (!shouldCaptureTritonGraph) {
    if (!allowTritonCudaGraphReplay) 
      DSP_DIAG(EXECUTE, "  BLOCKED: allowTritonCudaGraphReplay=false (tritonGraphCapture=%d OR shapesFrozen_=%d)",
               Environment::getInstance().tritonGraphCapture() ? 1 : 0, shapesFrozen_ ? 1 : 0);
    if (!replayHandleNull)
      DSP_DIAG(EXECUTE, "  BLOCKED: replayHandle already exists (capture already done or in progress)");
    if (seg.captureFailed)
      DSP_DIAG(EXECUTE, "  BLOCKED: captureFailed=true (previous capture failed, falling back to slot-by-slot)");
    if (!execCountInWindow)
      DSP_DIAG(EXECUTE, "  BLOCKED: executionCount=%d outside capture window [%d,%d]",
               seg.executionCount, captureMinExec, captureMinExec + 2);
    if (!hasCudaStream)
      DSP_DIAG(EXECUTE, "  BLOCKED: cudaStr=nullptr (no CUDA stream available)");
  } else {
    DSP_DIAG(EXECUTE, "  >>> CAPTURE WILL BE ATTEMPTED <<<");
  }
  DSP_DIAG(EXECUTE, "=== END CAPTURE DECISION CHECK ===");
  
  // NOTE: shouldCaptureTritonGraph is ONLY checked when we don't have a captured graph.
  // Once captured, we use useFastReplay based on argTableStable, not executionCount.
  // The executionCount window check prevents repeated capture attempts after success.
  
  // OPTIMIZATION: When argTableStable, addresses and create-op values haven't changed
  // since last refresh — skip the expensive hash/comparison loops over all external inputs.
  LongType segInputAddrKey;
  bool extAddrsStable;
  LongType createValueKey;
  if (seg.argTableStable && allowTritonCudaGraphReplay) {
    // Fast path: arg table is stable, all addresses are known-good
    segInputAddrKey = seg.capturedInputAddrKey;
    extAddrsStable = true;
    createValueKey = seg.capturedCreateValueKey;
  } else {
    segInputAddrKey = computeSegmentInputAddrKey(seg, externalArrays, numExt);
    extAddrsStable = (seg.replayHandle && !seg.replayHandle->getCapturedExternalAddresses().empty())
        ? externalAddrsMatch(seg, externalArrays, numExt)
        : (seg.capturedInputAddrKey != 0 && seg.capturedInputAddrKey == segInputAddrKey);
    createValueKey = computeCreateOpValueKey(seg, externalArrays, numExt);
  }
  bool createValuesStable = (createValueKey == 0) ||  // no create ops
                            (seg.capturedCreateValueKey == createValueKey);
  if (!createValuesStable && seg.replayHandle) {
    DSP_DIAG(EXECUTE, "CREATE_VALUE_KEY mismatch: captured=%lld current=%lld → invalidating graph seg[%d-%d]",
             (long long)seg.capturedCreateValueKey, (long long)createValueKey, seg.startSlot, seg.endSlot);
    for (auto* ptr : seg.replayHandle->getCapturedHostPtrs()) { cudaFreeHost(ptr); }
    seg.replayHandle->getCapturedHostPtrs().clear();
    seg.replayHandle->clearExternalAddresses();
    seg.replayHandle.reset();
    seg.argTableStable = false;
    batchD2DCount_ = 0;
    seg.capturedInputAddrKey = 0;
    seg.capturedCreateValueKey = 0;
    seg.executionCount = 0;
    seg.captureFailed = false;
    extAddrsStable = false;  // Force re-capture path
  }

  // Triton graph replay conditions:
  // 1. Shape key matches (frozen shapes)
  // 2. Create op input values stable (ConstantOfShape shapes unchanged)
  // 3. Either: addresses stable OR capture buffers handle data freshness
  //
  // With capture buffers (for PLACEHOLDER inputs), we D2D copy fresh data
  // before replay. The graph reads from capture buffer addresses (baked in)
  // and gets current placeholder values (position_ids, attention_mask, etc.).
  bool hasTritonCaptureBuffers = seg.replayHandle != nullptr &&
                                  !seg.replayHandle->getCaptureBuffers().empty();
  if (allowTritonCudaGraphReplay &&
      seg.replayHandle != nullptr &&
      seg.replayHandle->isReady() &&
      seg.cachedShapeKey == segShapeKey &&
      createValuesStable &&
      (hasTritonCaptureBuffers || extAddrsStable)) {

    // ── Lineage validation: verify directReference addresses haven't drifted ──
    // DirectReference entries (weights, KV cache) assume the graph reads from
    // the original buffer address. If the address changed (freed/reallocated),
    // the graph reads garbage. Detect and invalidate.
    // OPTIMIZATION: Skip when argTableStable is true — all addresses are known-stable.
    bool lineageInvalidated = false;
    if (hasTritonCaptureBuffers && !(seg.argTableStable && allowTritonCudaGraphReplay)) {
      bool addressDrift = false;
      for (auto& cb : seg.replayHandle->getCaptureBuffers()) {
        if (!cb.directReference) continue;
        const void* currentPtr = nullptr;
        if (cb.externalInputIndex >= 0 && cb.externalInputIndex < numExt) {
          NDArray* current = externalArrays[cb.externalInputIndex];
          currentPtr = current ? current->specialBuffer() : nullptr;
        } else if (cb.crossSegmentSlotIdx >= 0 && cb.crossSegmentSlotIdx < totalOutputSlots_) {
          NDArray* current = outputSlots_[cb.crossSegmentSlotIdx];
          currentPtr = current ? current->specialBuffer() : nullptr;
        }
        if (currentPtr != cb.lastSourcePtr) {
          DSP_DIAG(EXECUTE, "LINEAGE_DRIFT: %s#%d addr changed %p → %p → invalidate seg[%d-%d]",
                   cb.externalInputIndex >= 0 ? "ext" : "slot",
                   cb.externalInputIndex >= 0 ? cb.externalInputIndex : cb.crossSegmentSlotIdx,
                   cb.lastSourcePtr, currentPtr, seg.startSlot, seg.endSlot);
          addressDrift = true;
          break;
        }
      }
      if (addressDrift) {
        for (auto& cb2 : seg.replayHandle->getCaptureBuffers()) {
          if (!cb2.directReference) delete cb2.buffer;
        }
        seg.replayHandle->getCaptureBuffers().clear();
        for (auto* ptr : seg.replayHandle->getCapturedHostPtrs()) { cudaFreeHost(ptr); }
        seg.replayHandle->getCapturedHostPtrs().clear();
        seg.replayHandle->clearExternalAddresses();
        seg.replayHandle.reset();
        seg.argTableStable = false;
        batchD2DCount_ = 0;
        seg.capturedInputAddrKey = 0;
        seg.executionCount = 0;
        seg.captureFailed = false;
        hasTritonCaptureBuffers = false;
        lineageInvalidated = true;
      }
    }

    // Fast-replay: when arg table pointers are stable (all unchanged since last
    // refresh), skip the arg table refresh loop and EXT_INPUT_SYNC entirely.
    // Only D2D capture buffer copies + graph launch needed.
    bool useFastReplay = hasTritonCaptureBuffers && seg.argTableStable
                         && !Environment::getInstance().tritonVerifyKernels();

    // CRITICAL FIX: Set tl_dspExecutionStream for ALL Triton executions, not just capture replay.
    // Without this, syncToSpecial() calls fall back to stream 0 and do full cudaStreamSynchronize,
    // causing 657k sync calls per decode step. Setting tl_dspExecutionStream allows async H2D
    // copies on the same stream as compute, with stream ordering guaranteeing correctness.
    tl_dspExecutionStream = cudaStr;

    // Update capture buffers with fresh data (D2D copy).
    // Handles BOTH placeholder external inputs AND cross-segment output slots.
    // Use tl_dspExecutionStream for any syncToDevice calls inside the loop.
    bool crossSegSizeMismatch = false;
    if (hasTritonCaptureBuffers) {
      auto& captureBuffers = seg.replayHandle->getCaptureBuffers();

      // D2D copies for ALL capture buffers (placeholders + cross-segment).
      {
        int cbExtUpdated = 0, cbSlotUpdated = 0;
        for (auto& cb : captureBuffers) {
          if (cb.directReference) continue;
          if (cb.buffer == nullptr) continue;

          NDArray* src = nullptr;
          if (cb.externalInputIndex >= 0 && cb.externalInputIndex < numExt) {
            // Placeholder external input
            src = externalArrays[cb.externalInputIndex];
          } else if (cb.crossSegmentSlotIdx >= 0 && cb.crossSegmentSlotIdx < totalOutputSlots_) {
            // Cross-segment output slot
            src = outputSlots_[cb.crossSegmentSlotIdx];
          }
          if (src == nullptr) continue;

          size_t srcBytes = src->lengthOf() * src->sizeOfT();
          if (srcBytes == 0) continue;
          if (srcBytes != cb.capturedSize) {
            // Size mismatch — data-dependent shape changed. Must invalidate graph.
            if (cb.crossSegmentSlotIdx >= 0) {
              DSP_DIAG(EXECUTE, "CROSS_SEG_SIZE_MISMATCH: slot#%d captured=%zu current=%zu → invalidate",
                       cb.crossSegmentSlotIdx, cb.capturedSize, srcBytes);
              crossSegSizeMismatch = true;
              break;
            }
            continue;  // External input size mismatch — skip (existing behavior)
          }

          // Cross-segment outputs were just written on cudaStr by the previous
          // segment's slot-by-slot execution — already on device, no sync needed.
          // Only external (placeholder) inputs need syncToDevice() for H2D.
          if (cb.externalInputIndex >= 0) {
            src->syncToDevice();
          }
          const void* srcPtr = src->specialBuffer();
          if (!srcPtr || !cb.buffer->specialBuffer()) continue;

          // Always copy — GPU memory pools reuse addresses, so pointer comparison
          // cannot detect stale data. Cost is negligible (typically <10KB per buffer).
          cudaMemcpyAsync(cb.buffer->specialBuffer(), srcPtr,
                          srcBytes, cudaMemcpyDeviceToDevice, cudaStr);
          cb.lastSourcePtr = srcPtr;
          cb.initialCopyDone = true;
          if (cb.externalInputIndex >= 0) cbExtUpdated++;
          else cbSlotUpdated++;
        }
        DSP_DIAG(EXECUTE, "CAPTURE_BUFFER_UPDATE: ext=%d slot=%d "
                 "fastReplay=%d execCount=%d", cbExtUpdated, cbSlotUpdated,
                 useFastReplay ? 1 : 0, seg.executionCount);
      }
    }
    // NOTE: tl_dspExecutionStream is cleared at end of execute() in NativeDynamicShapePlan.cpp
    // Keep it set during replay for syncToSpecial() to use the correct stream.

    // Cross-segment size mismatch: invalidate graph and fall through to re-capture
    if (crossSegSizeMismatch && seg.replayHandle) {
      DSP_DIAG(EXECUTE, "GRAPH INVALIDATED: cross-segment data size changed for seg[%d-%d]",
               seg.startSlot, seg.endSlot);
      for (auto& cb : seg.replayHandle->getCaptureBuffers()) {
        if (!cb.directReference) delete cb.buffer;
      }
      seg.replayHandle->getCaptureBuffers().clear();
      for (auto* ptr : seg.replayHandle->getCapturedHostPtrs()) { cudaFreeHost(ptr); }
      seg.replayHandle->getCapturedHostPtrs().clear();
      seg.replayHandle->clearExternalAddresses();
      seg.replayHandle.reset();
      seg.argTableStable = false;
      batchD2DCount_ = 0;
      seg.capturedInputAddrKey = 0;
      seg.executionCount = 0;
      seg.captureFailed = false;
    } else

    if (useFastReplay) {
      // Fast path: arg table pointers are stable so skip refresh.
      // Only sync VARIABLE (PLACEHOLDER) external inputs — model weights and
      // constants are already on device and never change. This reduces the
      // sync loop from ~63 inputs to ~3 (input_ids, attention_mask, position_ids).
      cudaGetLastError();
      // tl_dspExecutionStream already set in execute() - don't clear it here
      int fastSynced = 0, fastSkipped = 0;
      for (int ei = 0; ei < numExt; ei++) {
        if (externalArrays[ei] == nullptr) continue;
        // Only sync variable (PLACEHOLDER) inputs — constants/weights skip
        if (ei < static_cast<int>(externalInputIsVariable_.size()) &&
            !externalInputIsVariable_[ei]) {
          fastSkipped++;
          continue;
        }
        auto* db = externalArrays[ei]->dataBuffer();
        bool pAct = db ? db->isPrimaryActual() : false;
        bool sAct = db ? db->isSpecialActual() : false;
        if (pAct && !sAct) fastSynced++;
        else fastSkipped++;
        externalArrays[ei]->syncToDevice();
      }
      DSP_DIAG(EXECUTE, "FAST_REPLAY_EXT_SYNC: %d H2D, %d skip execCount=%d",
               fastSynced, fastSkipped, seg.executionCount);
    } else {
    // Standard replay: sync ext inputs, refresh arg tables, diagnostics.
    // tl_dspExecutionStream already set in execute() - don't clear it here
    {
      DspDiagnostics::ExtInputSyncResult syncResult = {0, 0, 0};
      DSP_DIAG_DUMP_EXT_INPUTS(externalArrays, numExt, seg.executionCount, syncResult);
      int synced = 0, skipped = 0;
      for (int ei = 0; ei < numExt; ei++) {
        if (externalArrays[ei] != nullptr) {
          auto* db = externalArrays[ei]->dataBuffer();
          bool pAct = db ? db->isPrimaryActual() : false;
          bool sAct = db ? db->isSpecialActual() : false;
          if (pAct && !sAct) synced++;
          else skipped++;
          externalArrays[ei]->syncToDevice();
        }
      }
      DSP_DIAG(EXECUTE, "EXT_INPUT_SYNC replay: %d H2D, %d skip (device up-to-date) execCount=%d",
               synced, skipped, seg.executionCount);

      // Dump SMALL variable external inputs (verify mode only)
      if (Environment::getInstance().tritonVerifyKernels()) {
      cudaDeviceSynchronize();
      for (int ei = 0; ei < numExt; ei++) {
        if (externalArrays[ei] == nullptr) continue;
        auto* arr = externalArrays[ei];
        bool isSmall = arr->lengthOf() <= 16;
        std::string name = (ei < (int)externalInputNames_.size()) ? externalInputNames_[ei] : "?";
        std::string vals = "?";
        if (isSmall && arr->specialBuffer()) {
          int n = std::min((int)arr->lengthOf(), 4);
          int elemSize = DataTypeUtils::sizeOf(arr->dataType());
          std::vector<uint8_t> devBytes(n * elemSize);
          cudaMemcpy(devBytes.data(), arr->specialBuffer(), n * elemSize, cudaMemcpyDeviceToHost);
          vals = "";
          for (int j = 0; j < n; j++) {
            if (j > 0) vals += ",";
            if (arr->dataType() == INT64 || arr->dataType() == DataType::INT64) {
              int64_t v; std::memcpy(&v, devBytes.data() + j * 8, 8);
              vals += std::to_string(v);
            } else if (arr->dataType() == INT32) {
              int32_t v; std::memcpy(&v, devBytes.data() + j * 4, 4);
              vals += std::to_string(v);
            } else if (arr->dataType() == FLOAT32) {
              float v; std::memcpy(&v, devBytes.data() + j * 4, 4);
              vals += std::to_string(v);
            } else {
              vals += "?";
            }
          }
        }
        if (!isSmall || name.find("input") != std::string::npos ||
            name.find("position") != std::string::npos ||
            name.find("attention") != std::string::npos ||
            name.find("embed") != std::string::npos ||
            name.find("past") != std::string::npos) {
          DSP_DIAG(EXECUTE, "EXT_DATA[%d]:\"%s\" type=%d rank=%d len=%lld addr=%p vals=[%s] execCount=%d",
                   ei, name.c_str(), (int)arr->dataType(), (int)arr->rankOf(),
                   (long long)arr->lengthOf(),
                   arr->specialBuffer(), vals.c_str(), seg.executionCount);
        }
      }
      } // end tritonVerifyKernels() EXT_DATA dump
    }
    // Snapshot buffer addresses BEFORE replay for comparison with capture-time addresses
    {
      std::vector<void*> outAddrs, extAddrs;
      extractDeviceAddrs(outputSlots_, totalOutputSlots_, outAddrs);
      extractDeviceAddrs(externalArrays, numExt, extAddrs);
      DSP_DIAG_SNAPSHOT_ADDRS("replay-entry", outAddrs.data(), totalOutputSlots_,
                               extAddrs.data(), numExt);
      int mismatches = DSP_DIAG_COMPARE_ADDRS("capture-entry", "replay-entry");
      if (mismatches > 0) {
        DSP_DIAG(EXECUTE, "WARNING: %d address mismatches between capture and replay!", mismatches);
      }
    }

    // Refresh Triton arg table pinned buffers before replay.
    // When capture buffers exist, temporarily swap externalArrays AND outputSlots_
    // to capture buffer addresses so the arg table gets the addresses baked into
    // the graph. This covers both placeholder external inputs AND cross-segment
    // output slots.
#if HAVE_TRITON
    {
      std::vector<std::pair<int, NDArray*>> savedForArgRefresh;
      std::vector<std::pair<int, NDArray*>> savedSlotsForArgRefresh;
      if (hasTritonCaptureBuffers) {
        for (auto& cb : seg.replayHandle->getCaptureBuffers()) {
          if (cb.externalInputIndex >= 0 && cb.externalInputIndex < numExt && cb.buffer) {
            savedForArgRefresh.push_back({cb.externalInputIndex, externalArrays[cb.externalInputIndex]});
            externalArrays[cb.externalInputIndex] = cb.buffer;
          }
          if (cb.crossSegmentSlotIdx >= 0 && cb.crossSegmentSlotIdx < totalOutputSlots_ && cb.buffer) {
            savedSlotsForArgRefresh.push_back({cb.crossSegmentSlotIdx, outputSlots_[cb.crossSegmentSlotIdx]});
            outputSlots_[cb.crossSegmentSlotIdx] = cb.buffer;
          }
        }
      }
      auto* tritonBackend = dynamic_cast<TritonGraphBackend*>(backend);
      if (tritonBackend != nullptr) {
        tritonBackend->refreshArgTablesForReplay(seg, externalArrays, numExt,
                                                 outputSlots_, totalOutputSlots_,
                                                 stream);
      }
      for (auto& [extIdx, origArr] : savedForArgRefresh) {
        externalArrays[extIdx] = origArr;
      }
      for (auto& [slotIdx, origArr] : savedSlotsForArgRefresh) {
        outputSlots_[slotIdx] = origArr;
      }
    }
#endif
    // All H2D copies (ext input sync) and D2D copies (capture buffers) are on
    // cudaStr. Graph launch on cudaStr is ordered after them — no explicit sync needed.
    cudaGetLastError();  // Clear any sticky errors
    } // end standard replay path (else branch of useFastReplay)

    // DIAGNOSTIC: Zero capture workspace before replay to test stale-data hypothesis.
    // If zeroing the workspace fixes divergence, stale workspace data is the root cause.
    // This is gated on tritonVerifyKernels to avoid performance impact in production.
    if (Environment::getInstance().tritonVerifyKernels() &&
        seg.replayHandle && seg.replayHandle->getWorkspacePtr() != nullptr &&
        seg.replayHandle->getWorkspaceBytes() > 0) {
      cudaMemsetAsync(seg.replayHandle->getWorkspacePtr(), 0,
                      seg.replayHandle->getWorkspaceBytes(), cudaStr);
      cudaStreamSynchronize(cudaStr);
      DSP_DIAG(VERIFY, "REPLAY_DIAG: zeroed capture workspace (%zuMB) before replay execCount=%d",
               seg.replayHandle->getWorkspaceBytes() / (1024*1024), seg.executionCount);
    }

    // DIAGNOSTIC: Dump specific VARIABLE external inputs before replay to trace stale data.
    if (Environment::getInstance().tritonVerifyKernels()) {
      cudaDeviceSynchronize();
      for (int ei = 0; ei < numExt; ei++) {
        if (ei < (int)externalInputIsVariable_.size() && externalInputIsVariable_[ei] &&
            externalArrays[ei] != nullptr && externalArrays[ei]->lengthOf() <= 8) {
          auto* arr = externalArrays[ei];
          auto* db = arr->dataBuffer();
          int n = std::min((int)arr->lengthOf(), 8);
          int elemSize = DataTypeUtils::sizeOf(arr->dataType());
          std::vector<uint8_t> hostBytes(n * elemSize), devBytes(n * elemSize);
          if (db && db->primary()) std::memcpy(hostBytes.data(), static_cast<char*>(arr->buffer()), n * elemSize);
          if (arr->specialBuffer()) cudaMemcpy(devBytes.data(), arr->specialBuffer(), n * elemSize, cudaMemcpyDeviceToHost);
          float hv[8]={0}, dv[8]={0};
          dspBytesToFloat(hostBytes.data(), arr->dataType(), hv, n);
          dspBytesToFloat(devBytes.data(), arr->dataType(), dv, n);
          std::string name = (ei < (int)externalInputNames_.size()) ? externalInputNames_[ei] : "?";
          DSP_DIAG(VERIFY, "PRE_REPLAY ext#%d:\"%s\" len=%d pAct=%d sAct=%d host=[%.0f,%.0f,%.0f,%.0f] dev=[%.0f,%.0f,%.0f,%.0f]",
                    ei, name.c_str(), n,
                    db ? (db->isPrimaryActual()?1:0) : -1,
                    db ? (db->isSpecialActual()?1:0) : -1,
                    hv[0],hv[1],hv[2],hv[3], dv[0],dv[1],dv[2],dv[3]);
        }
      }
    }

    // Pre-replay batch-zero: zero all output buffers OUTSIDE the graph.
    // Individual cudaMemsetAsync calls use dedicated fill engines (not SMs),
    // pipeline efficiently, and add 0 graph nodes (they run before cudaGraphLaunch).
    // Stream ordering guarantees all zeroing completes before graph launch.
    // NOTE: Do NOT use batchZeroKernel here — it runs on SMs (competition with
    // compute kernels) and has alignment requirements that cause accuracy issues.
    if (Environment::getInstance().dspBatchZero() && !batchZeroEntries_.empty()) {
      for (auto& entry : batchZeroEntries_) {
        if (entry.ptr != nullptr && entry.bytes > 0) {
          cudaMemsetAsync(entry.ptr, 0, entry.bytes, cudaStr);
        }
      }
      DSP_DIAG(MEMORY, "pre-replay batch-zero: %d buffers zeroed via cudaMemsetAsync (fill engines, outside graph)",
                static_cast<int>(batchZeroEntries_.size()));
    }

    // Replay strategy: configurable via ND4J_TRITON_GRAPH_REINSTANTIATE.
    // Default (OFF): direct replay of existing graphExec.
    // ON: destroy and re-instantiate graphExec from graph template before each replay.
    // Skip entirely if lineage validation or cross-segment size mismatch invalidated the graph.
    {
      bool replayOk = false;
      if (lineageInvalidated || crossSegSizeMismatch || !seg.replayHandle) {
        // Graph was invalidated — skip replay, fall through to re-capture/slot-by-slot
        DSP_DIAG(EXECUTE, "REPLAY_SKIPPED: lineage=%d sizeMismatch=%d handle=%p seg[%d-%d]",
                 lineageInvalidated ? 1 : 0, crossSegSizeMismatch ? 1 : 0,
                 (void*)seg.replayHandle.get(), seg.startSlot, seg.endSlot);
      } else if (Environment::getInstance().tritonGraphReinstantiate()) {
        auto* cudaReplay = static_cast<CudaGraphReplayHandle*>(seg.replayHandle.get());
        if (!cudaReplay->getNativeHandle()->reInstantiate()) {
          DSP_DIAG_SEG(EXECUTE, seg.startSlot, "Triton graph reInstantiate FAILED for seg[%d-%d]",
                    seg.startSlot, seg.endSlot);
        } else {
          replayOk = seg.replayHandle->replay(stream);
        }
      } else {
        replayOk = seg.replayHandle->replay(stream);
      }
      if (replayOk) {
        // Find the ACTUAL final output slot index (not the step index)
        int finalOutputSlot = -1;
        if (seg.endSlot < numSlots_ && slots_[seg.endSlot].numOutputs > 0) {
          finalOutputSlot = slots_[seg.endSlot].outputSlotIndices[0];
        }
        // Fallback to seg.endSlot if output slot lookup fails
        if (finalOutputSlot < 0 || finalOutputSlot >= totalOutputSlots_) {
          finalOutputSlot = seg.endSlot;
        }

        if (finalOutputSlot >= 0 && finalOutputSlot < totalOutputSlots_ &&
            outputSlots_[finalOutputSlot] != nullptr) {
          auto* finalOut = outputSlots_[finalOutputSlot];
          if (finalOut->dataType() == FLOAT32) {
            DSP_DIAG_DUMP_SLOT("replay", finalOutputSlot,
                               finalOut->specialBuffer(), finalOut->lengthOf());
          }
          // Dump top logit from cached graph replay via DSP_DIAG
          if (finalOut->dataType() == FLOAT32 && finalOut->lengthOf() > 0) {
            DSP_DIAG_DUMP_SEG_OUTPUT("GRAPH_REPLAY", finalOutputSlot, finalOut->specialBuffer(),
                                     finalOut->lengthOf(), seg.executionCount, stream);
          }
        }

        // Log argmax after replay for divergence tracking (diagnostic only).
        // The sync + D2H copy for argmax is expensive (~5-10ms). Only do it
        // when DSP diagnostics are enabled. The lineage framework (capture
        // buffers + D2D copies) ensures correctness without this sync.
        if (DSP_DIAG_ENABLED(EXECUTE)) {
          cudaStreamSynchronize(cudaStr);
          if (finalOutputSlot >= 0 && finalOutputSlot < totalOutputSlots_ &&
              outputSlots_[finalOutputSlot] != nullptr) {
            auto* finalOut = outputSlots_[finalOutputSlot];
            int replayArgmax = dspArgmax(finalOut->specialBuffer(), finalOut->dataType(),
                                         finalOut->lengthOf());
            std::string firstVals = dspDumpSlotValues(finalOut->specialBuffer(), finalOut->dataType(),
                                                       finalOut->lengthOf(), 4);
            DSP_DIAG(EXECUTE, "GRAPH_REPLAY ARGMAX: slot=%d argmax=%d len=%lld vals=%s execCount=%d",
                     finalOutputSlot, replayArgmax, (long long)finalOut->lengthOf(),
                     firstVals.c_str(), seg.executionCount);
          }
        }

        seg.executionCount++;
        totalGraphReplays_++;

        // ── REPLAY VERIFICATION ─────────────────────────────────────────
        if (Environment::getInstance().tritonVerifyKernels()) {
          cudaStreamSynchronize(cudaStr);
          performReplayVerify(seg, externalArrays, numExt, stream, "TRITON");
        }

        // Force re-capture every step (diagnostic mode).
        // Invalidates the cached graph after each replay so the next step
        // re-captures with fresh data.  Correct but slow.
        if (Environment::getInstance().tritonForceRecapture()) {
          if (seg.replayHandle) {
            for (auto* ptr : seg.replayHandle->getCapturedHostPtrs()) { cudaFreeHost(ptr); }
            seg.replayHandle->getCapturedHostPtrs().clear();
            seg.replayHandle->clearExternalAddresses();
          }
          seg.replayHandle.reset();
          seg.argTableStable = false;
          batchD2DCount_ = 0;
          seg.capturedInputAddrKey = 0;
          DSP_DIAG(EXECUTE, "FORCE_RECAPTURE: invalidated after replay execCount=%d", seg.executionCount);
        }

        if (Environment::getInstance().tritonVerifyKernels()) {
          DSP_DIAG(VERIFY, "SEG_EXIT seg[%d-%d] status=OK(replay) execCount=%d",
                    seg.startSlot, seg.endSlot, seg.executionCount);
        }
        return Status::OK;
      }
      // Launch failed — clear stale graph and fall through to direct Triton execution.
      DSP_DIAG_SEG(FALLBACK, seg.startSlot, "Triton graph replay FAILED for seg[%d-%d], "
                "falling back to direct execution", seg.startSlot, seg.endSlot);
      if (seg.replayHandle) {
        for (auto* ptr : seg.replayHandle->getCapturedHostPtrs()) {
          cudaFreeHost(ptr);
        }
        seg.replayHandle->getCapturedHostPtrs().clear();
        seg.replayHandle->clearExternalAddresses();
      }
      seg.replayHandle.reset();
      seg.argTableStable = false;
      batchD2DCount_ = 0;
      seg.capturedInputAddrKey = 0;
      seg.captureFailed = true;
      cudaGetLastError();
    }
  }

  if (allowTritonCudaGraphReplay &&
      (!seg.replayHandle || seg.replayHandle->getCaptureBuffers().empty()) &&
      seg.replayHandle != nullptr &&
      seg.cachedShapeKey == segShapeKey &&
      !extAddrsStable) {
    if (seg.replayHandle) {
      for (auto* ptr : seg.replayHandle->getCapturedHostPtrs()) { cudaFreeHost(ptr); }
      seg.replayHandle->getCapturedHostPtrs().clear();
      seg.replayHandle->clearExternalAddresses();
    }
    seg.replayHandle.reset();
    seg.argTableStable = false;
    batchD2DCount_ = 0;
    seg.capturedInputAddrKey = 0;
  }

  if (allowTritonCudaGraphReplay &&
      (!seg.replayHandle || seg.replayHandle->getCaptureBuffers().empty()) &&
      seg.replayHandle != nullptr &&
      seg.cachedShapeKey != segShapeKey) {
    if (seg.replayHandle) {
      for (auto* ptr : seg.replayHandle->getCapturedHostPtrs()) { cudaFreeHost(ptr); }
      seg.replayHandle->getCapturedHostPtrs().clear();
      seg.replayHandle->clearExternalAddresses();
    }
    seg.replayHandle.reset();
    seg.argTableStable = false;
    batchD2DCount_ = 0;
    seg.capturedInputAddrKey = 0;
  }
#endif

#if HAVE_TRITON
  struct TritonFallbackGuard {
    bool active = false;
    ~TritonFallbackGuard() {
      if (active) TritonGraphBackend::clearFallbackRangeExecutor();
    }
  } tritonFallbackGuard;

  auto* tritonBackend = dynamic_cast<TritonGraphBackend*>(backend);
  if (tritonBackend != nullptr) {
    TritonGraphBackend::setFallbackRangeExecutor(
        [this, &seg, externalArrays, numExt, stream](int startSlot, int endSlot) -> Status {
          if (startSlot > endSlot) return Status::OK;

          GraphSegment gapSeg;
          gapSeg.startSlot = startSlot;
          gapSeg.endSlot = endSlot;
          gapSeg.executionCount = seg.executionCount;
          gapSeg.captureFailed = seg.captureFailed;

          // Check if the stream is currently being captured (CUDA graph recording).
          // During capture: keep tl_graphExecutionActive=true so fallback ops use the
          // pre-allocated capture workspace for any allocations. The workspace must be
          // set up before beginCapture (see shouldCaptureTritonGraph block below).
          // Outside capture: set tl_graphExecutionActive=false so fallback ops use
          // normal allocation paths (cudaMallocAsync) and sync guards work normally.
          bool streamIsCapturing = false;
#ifdef SD_CUDA
          if (stream != nullptr) {
            cudaStreamCaptureStatus capStat = cudaStreamCaptureStatusNone;
            cudaStreamIsCapturing(*static_cast<cudaStream_t*>(stream), &capStat);
            streamIsCapturing = (capStat != cudaStreamCaptureStatusNone);
          }

          // Synchronize between the Triton execution stream and the gap ops' stream.
          // Triton kernels use the explicit stream parameter; native fallback ops use
          // the thread-local LaunchContext stream (a different CUDA stream). Without
          // synchronization, gap ops can read stale data from before the preceding
          // Triton kernel completes, and subsequent Triton kernels can read stale
          // gap op outputs.
          //
          // Outside capture: use cudaStreamSynchronize (simple, no overhead concern
          // since gap ops are already the bottleneck).
          // During capture: use CUDA events to create graph dependency edges between
          // the capture stream and the gap ops' stream. cudaStreamSynchronize cannot
          // be used during capture.
          cudaStream_t tritonStr = *static_cast<cudaStream_t*>(stream);
          auto* lcStream = LaunchContext::defaultContext()->getCudaStream();
          cudaStream_t gapStr = lcStream ? *lcStream : nullptr;
          bool streamsMatch = (tritonStr == gapStr);

          // One-time diagnostic: log whether streams match
          static bool streamDiagDone = false;
          if (!streamDiagDone) {
            DSP_DIAG(BACKEND, "stream diag: tritonStr=%p gapStr=%p match=%d capturing=%d",
                     (void*)tritonStr, (void*)gapStr, streamsMatch ? 1 : 0,
                     streamIsCapturing ? 1 : 0);
            streamDiagDone = true;
          }

          if (!streamIsCapturing && stream != nullptr) {
            if (!streamsMatch) {
              cudaStreamSynchronize(tritonStr);
            }
          } else if (streamIsCapturing && !streamsMatch && gapStr != nullptr) {
            // During capture: record event on Triton stream, make gap stream wait.
            // This creates a dependency edge in the CUDA graph.
            cudaEvent_t syncEvent;
            cudaEventCreateWithFlags(&syncEvent, cudaEventDisableTiming);
            cudaEventRecord(syncEvent, tritonStr);
            cudaStreamWaitEvent(gapStr, syncEvent, 0);
            cudaEventDestroy(syncEvent);
          }
#endif
          bool savedGraphActive = tl_graphExecutionActive;
          if (!streamIsCapturing) {
            tl_graphExecutionActive = false;
          }
          // When capturing, tl_graphExecutionActive stays true — ops allocate from
          // capture workspace and cuBLAS/cuDNN calls get recorded into the graph.
          auto gapStatus = executeSegmentSlotBySlot(gapSeg, externalArrays, numExt, stream);
#ifdef SD_CUDA
          // After gap ops: synchronize gap stream → Triton stream.
          if (!streamIsCapturing) {
            if (!streamsMatch && gapStr != nullptr) {
              cudaStreamSynchronize(gapStr);
            }
          } else if (streamIsCapturing && !streamsMatch && gapStr != nullptr) {
            // During capture: record event on gap stream, make Triton stream wait.
            cudaEvent_t syncEvent;
            cudaEventCreateWithFlags(&syncEvent, cudaEventDisableTiming);
            cudaEventRecord(syncEvent, gapStr);
            cudaStreamWaitEvent(tritonStr, syncEvent, 0);
            cudaEventDestroy(syncEvent);
          }
#endif
          tl_graphExecutionActive = savedGraphActive;
          return gapStatus;
        });
    tritonFallbackGuard.active = true;
  }
#endif

  Status status = Status::KERNEL_FAILURE;
  bool usedTritonGraphCapture = false;

#ifdef SD_CUDA
  // Recompute shouldCaptureTritonGraph here (same logic as CAPTURE DECISION CHECK above)
  // This is the actual capture point - the diagnostic above just logs the decision.
  bool hasReplayHandleNow = (seg.replayHandle != nullptr);
  bool hasCaptureBuffersNow = hasReplayHandleNow && !seg.replayHandle->getCaptureBuffers().empty();
  bool replayHandleNullNow = (seg.replayHandle == nullptr);
  bool execCountInWindowNow = (seg.executionCount >= captureMinExec) &&
                              (forceRecaptureEnabled || seg.executionCount <= (captureMinExec + 2));
  bool shouldCaptureTritonGraphNow = allowTritonCudaGraphReplay &&
                                     (!hasReplayHandleNow || !hasCaptureBuffersNow) &&
                                     replayHandleNullNow &&
                                     !seg.captureFailed &&
                                     execCountInWindowNow &&
                                     hasCudaStream;
  if (shouldCaptureTritonGraphNow) {
    // Set up capture workspace BEFORE beginCapture — cudaMalloc must be outside capture.
    // Fallback ops (matmul, attention, concat) need temporary buffers during execution.
    // With tl_graphExecutionActive=true, CudaMemoryPool allocates from this workspace
    // instead of calling cudaMallocAsync (which fails during capture).
    static size_t TRITON_CAPTURE_WORKSPACE_SIZE = []() -> size_t {
      const char* envVal = std::getenv("ND4J_DSP_CAPTURE_WORKSPACE_MB");
      size_t mb = 512;  // default
      if (envVal != nullptr) {
        int parsed = std::atoi(envVal);
        if (parsed > 0 && parsed <= 4096) mb = static_cast<size_t>(parsed);
      }
      return mb * 1024ULL * 1024ULL;
    }();

    // Create the replayHandle BEFORE capture — it must exist to store workspace, host ptrs, etc.
    {
      int deviceId = 0;
      cudaGetDevice(&deviceId);
      seg.replayHandle = GraphReplayFactory::create(deviceId);
    }

    if (seg.replayHandle->getWorkspacePtr() == nullptr) {
      int deviceId = 0;
      cudaGetDevice(&deviceId);
      bool allocated = seg.replayHandle->allocateWorkspace(
          TRITON_CAPTURE_WORKSPACE_SIZE, deviceId, captureBufferRegistry_, seg.startSlot);
      if (allocated) {
        DSP_DIAG_SEG(MEMORY, seg.startSlot, "allocated %zuMB Triton capture workspace for seg[%d-%d]",
                  TRITON_CAPTURE_WORKSPACE_SIZE / (1024*1024), seg.startSlot, seg.endSlot);
      } else {
        DSP_DIAG(MEMORY, "WARNING - Triton capture workspace allocation failed for seg[%d-%d]",
                  seg.startSlot, seg.endSlot);
      }
    }

    // Set thread-local workspace for CudaMemoryPool during capture
    tl_captureWorkspace = seg.replayHandle->getWorkspacePtr();
    tl_captureWorkspaceSize = seg.replayHandle->getWorkspaceBytes();
    tl_captureWorkspaceOffset = 0;
    tl_capturedHostPtrs.clear();
    tl_captureReplicateCache.clear();
    // Set capture stream so captureSafeStreamOrDefault() routes ops to the correct stream
    cudaStream_t prevCaptureStream = tl_graphCaptureStream;
    tl_graphCaptureStream = cudaStr;

    // Pre-allocate cuBLAS workspace to prevent internal cudaMalloc during capture.
    // cuBLAS internally allocates workspace on stream 0 for GEMM operations. During
    // graph capture on a named stream, this cross-stream allocation breaks capture,
    // producing invalid graph nodes that SIGSEGV on cudaGraphLaunch.
    static const size_t CUBLAS_WORKSPACE_SIZE = 256 * 1024 * 1024;  // 256 MB
    ensureCublasWorkspace(CUBLAS_WORKSPACE_SIZE);
    // NOTE: setCublasWorkspaceForCapture is deferred to AFTER warmup (see below).
    // Calling it here sets cublasSetStream_v2 to the capture stream, which causes
    // cuBLAS matmuls in gap ops during warmup to run on tritonStr instead of gapStr.
    // This stream mismatch creates data races: cast ops on gapStr write matmul
    // inputs, but cuBLAS on tritonStr starts before gapStr completes.

    // Reset MmulHelper cast cache indices so capture reuses pre-allocated HALF buffers
    // in the same order as the warmup execution (avoids capture workspace temporaries)
    MmulHelper::resetCastCacheIndices();

    // ── Batch-zero preparation (OUTSIDE capture) ─────────────────────────
    // Use the registration-based approach: batchZeroEntries_ was populated
    // by finishBatchZeroRegistration() during the warmup execution (execCount==1).
    // This contains ONLY the buffers that were actually nullified during warmup,
    // avoiding the ~143 extra buffers that collectBatchZeroTargets() would include
    // for slots that don't actually execute (identity ops, fused chains, etc.).
    //
    // If registration didn't happen (e.g., capture retry), fall back to
    // collectBatchZeroTargets for the pre-scan approach.
    if (Environment::getInstance().dspBatchZero()) {
      if (!batchZeroEntries_.empty()) {
        // Registration-based: entries already populated from warmup
        DSP_DIAG(MEMORY, "batch-zero using %d REGISTERED buffers (from warmup observation)",
                  static_cast<int>(batchZeroEntries_.size()));
      } else {
        // Fallback: pre-scan approach (may include extra buffers)
        DSP_DIAG(MEMORY, "batch-zero registration empty, falling back to collectBatchZeroTargets");
        std::unordered_set<int> gapSlots;
        if (Environment::getInstance().dspBatchZeroGapOnly()) {
#if HAVE_TRITON
          auto* tritonBE = dynamic_cast<TritonGraphBackend*>(backend);
          if (tritonBE != nullptr) {
            gapSlots = tritonBE->getGapSlots(seg, slots_);
          } else
#endif
          {
            for (int s = seg.startSlot; s <= seg.endSlot; s++) gapSlots.insert(s);
          }
        } else {
          for (int s = seg.startSlot; s <= seg.endSlot; s++) gapSlots.insert(s);
        }
        collectBatchZeroTargets(gapSlots);
      }
      prepareBatchZeroDevice(cudaStr);
    }

    // Sync external inputs to device before capture — same rationale as non-capture path.
    // Java may have modified host buffers (putScalar + tagLocation(HOST)) between steps.
    // specialBuffer() in arg table population doesn't check for stale device data.
    for (int ei = 0; ei < numExt; ei++) {
      if (externalArrays[ei] != nullptr) {
        if (Environment::getInstance().tritonVerifyKernels()) {
          auto* db = externalArrays[ei]->dataBuffer();
          DSP_DIAG(VERIFY, "SLOT_WRITE slot=%d tag=EXT_SYNC(capture) extIdx=%d pAct=%d sAct=%d len=%lld addr=%p",
                    -(ei + 1), ei,
                    db ? (db->isPrimaryActual() ? 1 : 0) : -1,
                    db ? (db->isSpecialActual() ? 1 : 0) : -1,
                    (long long)externalArrays[ei]->lengthOf(),
                    externalArrays[ei]->specialBuffer());
        }
        externalArrays[ei]->syncToDevice();
      }
    }

    // Synchronize before capture to ensure all prior async work is complete
    cudaStreamSynchronize(cudaStr);
    // Clear any sticky CUDA error before capture — stale errors from prior operations
    // (e.g., cudaFuncGetName on driver-API functions) contaminate capture and launch.
    cudaGetLastError();

    // Configurable: push primary CUDA context during capture.
    // Default OFF — the non-Triton path works without it. Pushing and then popping
    // after capture may cause SIGSEGV on replay (null pointer inside libcuda.so).
    // Enable via ND4J_TRITON_GRAPH_CTX_PUSH=1 for debugging.
    int tritonCaptureDevice = 0;
    cudaGetDevice(&tritonCaptureDevice);
    CUcontext primaryCtx = nullptr;
    CUcontext prevCtx = nullptr;
    bool didPushCtx = false;
    if (Environment::getInstance().tritonGraphCtxPush()) {
      CUdevice cuDev;
      cuDeviceGet(&cuDev, tritonCaptureDevice);
      cuDevicePrimaryCtxRetain(&primaryCtx, cuDev);
      cuCtxGetCurrent(&prevCtx);
      if (prevCtx != primaryCtx) {
        cuCtxPushCurrent(primaryCtx);
        didPushCtx = true;
        DSP_DIAG(EXECUTE, "Triton capture pushed primary ctx %p (was %p) for device %d",
                  (void*)primaryCtx, (void*)prevCtx, tritonCaptureDevice);
      }
    }

    // ── PRE-CAPTURE WARMUP EXECUTION ────────────────────────────────────────
    // During CUDA graph capture, GPU operations are NOT executed — they are only
    // recorded into the graph.  The capture step's output buffers retain whatever
    // values they had BEFORE capture started.  Without a warmup, those values are
    // from the PREVIOUS step's execution, producing a stale/wrong token that
    // corrupts the entire decode sequence.
    //
    // Fix: run a non-capture execution BEFORE capture to produce correct output
    // for this step.  The capture then records the same operations (for replay),
    // but the output buffers already have the correct values from the warmup.
    // This matches the non-Triton CUDA graph path (NativeDynamicShapePlan_cudagraph.cu
    // line 488-490) which runs executeSegmentSlotBySlot() before capture.
    {
      DSP_DIAG_SEG(EXECUTE, seg.startSlot, "Triton pre-capture warmup for seg[%d-%d] execCount=%d",
                seg.startSlot, seg.endSlot, seg.executionCount);

      // Disable frozen fast path for warmup — same rationale as capture below.
      std::vector<bool> savedFrozenWarmup(seg.endSlot - seg.startSlot + 1);
      for (int s = seg.startSlot; s <= seg.endSlot; s++) {
        savedFrozenWarmup[s - seg.startSlot] = slots_[s].frozenContextReady;
        slots_[s].frozenContextReady = false;
      }

      auto warmupStatus = backend->executeSegment(seg, slots_, externalArrays, numExt,
                                                   outputSlots_, totalOutputSlots_, stream);
      // Restore frozen state after warmup
      for (int s = seg.startSlot; s <= seg.endSlot; s++) {
        slots_[s].frozenContextReady = savedFrozenWarmup[s - seg.startSlot];
      }

      if (warmupStatus != Status::OK) {
        DSP_DIAG(EXECUTE, "Triton pre-capture warmup FAILED for seg[%d-%d], falling back",
                  seg.startSlot, seg.endSlot);
        seg.captureFailed = true;
        return warmupStatus;
      }
      // Decrement executionCount — the warmup was an extra execution that should
      // not count toward the capture threshold.
      if (seg.executionCount > 0) seg.executionCount--;

      // Synchronize before capture to ensure warmup results are visible
      cudaStreamSynchronize(cudaStr);
      cudaGetLastError();

      // Reset MmulHelper cast cache indices again — warmup consumed them,
      // capture needs them in the same order for consistent graph recording
      MmulHelper::resetCastCacheIndices();

      DSP_DIAG_SEG(EXECUTE, seg.startSlot, "Triton pre-capture warmup DONE for seg[%d-%d]",
                seg.startSlot, seg.endSlot);

      // DIAGNOSTIC: dump warmup's final output argmax for comparison with replay
      {
        int finalOutputSlot = -1;
        if (seg.endSlot < numSlots_ && slots_[seg.endSlot].numOutputs > 0) {
          finalOutputSlot = slots_[seg.endSlot].outputSlotIndices[0];
        }
        if (finalOutputSlot < 0 || finalOutputSlot >= totalOutputSlots_)
          finalOutputSlot = seg.endSlot;
        if (finalOutputSlot >= 0 && finalOutputSlot < totalOutputSlots_ &&
            outputSlots_[finalOutputSlot] != nullptr) {
          auto* warmupOut = outputSlots_[finalOutputSlot];
          if (warmupOut->dataType() == FLOAT32 && warmupOut->lengthOf() > 0) {
            int warmupArgmax = dspArgmax(warmupOut->specialBuffer(), warmupOut->dataType(),
                                          warmupOut->lengthOf());
            std::string warmupVals = dspDumpSlotValues(warmupOut->specialBuffer(), warmupOut->dataType(),
                                                        warmupOut->lengthOf(), 4);
            DSP_DIAG(EXECUTE, "WARMUP ARGMAX: slot=%d argmax=%d len=%lld vals=%s execCount=%d",
                     finalOutputSlot, warmupArgmax, (long long)warmupOut->lengthOf(),
                     warmupVals.c_str(), seg.executionCount);
          }
        }
      }

      // ── RESTORE NULL OUTPUT SLOTS FROM CACHE ─────────────────────────────
      // The warmup execution may clear some outputSlots_ entries (e.g. control
      // flow CF_SWITCH dead outputs, or segment cleanup paths).  The values
      // were captured into slotArrayCache_ during execution, so restore any
      // null entries from the cache.  Without this, the subsequent CUDA graph
      // capture sees null inputs for sub-kernels and fails (captureStatus=50).
      // This mirrors the non-Triton CUDA graph path which saves/restores
      // outputSlots_ around warmup (NativeDynamicShapePlan_cudagraph.cu:488-496).
      {
        int restoredCount = 0;
        for (int s = seg.startSlot; s <= seg.endSlot; s++) {
          for (int o = 0; o < slots_[s].numOutputs; o++) {
            int si = slots_[s].outputSlotIndices[o];
            if (si >= 0 && si < totalOutputSlots_ &&
                outputSlots_[si] == nullptr && slotArrayCache_[si] != nullptr) {
              outputSlots_[si] = slotArrayCache_[si];
              restoredCount++;
            }
          }
        }
        if (restoredCount > 0) {
          DSP_DIAG(EXECUTE, "POST-WARMUP: restored %d null outputSlots from cache in seg[%d-%d]",
                    restoredCount, seg.startSlot, seg.endSlot);
        }
      }
    }

    // DIAGNOSTIC: warmup-only mode — skip capture, use warmup result directly.
    // Enables bisection: if warmup-only produces correct output but capture+replay
    // does not, the bug is in capture/replay. Set ND4J_TRITON_WARMUP_ONLY=1.
    {
      static bool warmupOnly = (std::getenv("ND4J_TRITON_WARMUP_ONLY") != nullptr &&
                                 std::string(std::getenv("ND4J_TRITON_WARMUP_ONLY")) == "1");
      if (warmupOnly) {
        DSP_DIAG(EXECUTE, "WARMUP_ONLY: skipping capture for seg[%d-%d], using warmup result",
                  seg.startSlot, seg.endSlot);
        // Clean up thread-local state
        tl_captureWorkspace = nullptr;
        tl_captureWorkspaceSize = 0;
        tl_captureWorkspaceOffset = 0;
        tl_graphCaptureStream = prevCaptureStream;
        // Don't create a replay handle — fall through to non-capture path next time
        seg.captureFailed = true;
        if (didPushCtx) {
          CUcontext dummy;
          cuCtxPopCurrent(&dummy);
          CUdevice cuDev;
          cuDeviceGet(&cuDev, tritonCaptureDevice);
          cuDevicePrimaryCtxRelease(cuDev);
        }
        restoreCublasWorkspaceAfterCapture(stream);
        return Status::OK;
      }
    }

    // NOW set cuBLAS handle to capture stream — AFTER warmup completed.
    // During warmup, gap ops must use their default stream (gapStr) for cuBLAS.
    // Only during actual capture do we switch cuBLAS to tritonStr so GEMM nodes
    // are recorded into the CUDA graph on the correct stream.
    setCublasWorkspaceForCapture(stream);

    // Disable frozen fast path during capture. Same rationale as non-Triton path:
    // capture may re-create views, and the frozen context has stale input/output pointers
    // from the prior non-capture execution. Using the full (non-frozen) path during capture
    // is a one-time cost — all context pointers are properly reconfigured with capture-time
    // arrays, including correct nullify() calls to zero output buffers.
    // Save and restore frozenContextReady after capture so replay uses frozen fast path.
    std::vector<bool> savedFrozenContextReadyTriton(seg.endSlot - seg.startSlot + 1);
    for (int s = seg.startSlot; s <= seg.endSlot; s++) {
      savedFrozenContextReadyTriton[s - seg.startSlot] = slots_[s].frozenContextReady;
      slots_[s].frozenContextReady = false;
    }

    // ── Create capture buffers for PLACEHOLDER external inputs ─────────────
    // Only buffer the dynamic inputs (position_ids, attention_mask, input_ids,
    // inputs_embeds) that Java updates between decode steps. Model weights and
    // ConstantOfShape intermediates are NOT buffered — their data doesn't change
    // or is handled by the createValueKey mechanism.
    //
    // During capture, the graph bakes in capture buffer addresses. Before each
    // replay, we D2D copy fresh data from Java's arrays to capture buffers.
    std::vector<std::pair<int, NDArray*>> savedExtForCapture;
    {
      std::unordered_set<int> capturedExtIndices;
      for (int ei = 0; ei < numExt; ei++) {
        if (ei >= static_cast<int>(externalInputIsVariable_.size())) break;
        if (!externalInputIsVariable_[ei]) continue;  // Only PLACEHOLDER inputs
        if (capturedExtIndices.count(ei)) continue;
        NDArray* src = externalArrays[ei];
        if (src == nullptr || src->lengthOf() == 0) continue;

        capturedExtIndices.insert(ei);
        src->syncToDevice();
        size_t srcBytes = src->lengthOf() * src->sizeOfT();

        // Check if this is a KV cache input — these use directReference
        // (the graph reads/writes the original buffer, no copy needed)
        bool isKvCacheInput = false;
        if (kvCacheRetentionEnabled_) {
          for (int km = 0; km < kvCacheNumMappings_; km++) {
            if (kvCacheMappings_[km].pastInputExternalIdx == ei) {
              isKvCacheInput = true;
              break;
            }
          }
        }

        if (isKvCacheInput) {
          // KV cache: graph uses the actual buffer — no copy needed on replay
          ReplayCaptureBuffer cb;
          cb.buffer = src;
          cb.externalInputIndex = ei;
          cb.crossSegmentSlotIdx = -1;
          cb.capturedSize = srcBytes;
          cb.directReference = true;
          cb.initialCopyDone = true;
          cb.lastSourcePtr = src->specialBuffer();
          seg.replayHandle->addCaptureBuffer(std::move(cb));
          // Do NOT save/replace externalArrays — graph uses src directly
        } else {
          // Regular placeholder: create a fixed-address capture buffer
          auto srcShapeVec = *src->getShapeAsVector();
          auto* capBuf = new NDArray(src->ordering(), srcShapeVec, src->dataType(),
                                     sd::LaunchContext::defaultContext());
          if (srcBytes > 0 && src->specialBuffer() && capBuf->specialBuffer()) {
            cudaMemcpyAsync(capBuf->specialBuffer(), src->specialBuffer(),
                            srcBytes, cudaMemcpyDeviceToDevice, cudaStr);
          }
          // Also mirror host buffer for ops that read from host
          if (srcBytes > 0 && src->buffer() && capBuf->buffer()) {
            std::memcpy(capBuf->buffer(), src->buffer(), srcBytes);
            capBuf->dataBuffer()->readPrimary();
            capBuf->dataBuffer()->writeSpecial();
          }

          ReplayCaptureBuffer cb;
          cb.buffer = capBuf;
          cb.externalInputIndex = ei;
          cb.crossSegmentSlotIdx = -1;
          cb.capturedSize = srcBytes;
          cb.neverSkipCopy = true;
          seg.replayHandle->addCaptureBuffer(std::move(cb));

          savedExtForCapture.push_back({ei, externalArrays[ei]});
          externalArrays[ei] = capBuf;
        }
      }
      if (!capturedExtIndices.empty()) {
        cudaStreamSynchronize(cudaStr);
        DSP_DIAG(EXECUTE, "CAPTURE_BUFFERS: created %zu buffers for PLACEHOLDER ext inputs",
                 capturedExtIndices.size());
      }
    }

    // ── Create capture buffers for CROSS-SEGMENT output slot inputs ─────────
    // When a non-capturable segment (data-dependent ops like Where, NonZero)
    // precedes this capturable segment, its output slots feed into this segment
    // as inputs. The graph bakes in capture-time addresses. If the non-capturable
    // segment reallocates output arrays (data-dependent shape changes), the graph
    // reads stale data from the old address. Capture buffers provide fixed-address
    // staging areas, with D2D copies of fresh data before each replay.
    std::vector<std::pair<int, NDArray*>> savedSlotsForCapture;
    {
      std::unordered_set<int> crossSegSlots;
      for (int s = seg.startSlot; s <= seg.endSlot; s++) {
        for (int i = 0; i < slots_[s].numInputs; i++) {
          int srcIdx = slots_[s].inputSourceIndices[i];
          if (srcIdx >= 0 && (srcIdx < seg.startSlot || srcIdx > seg.endSlot)) {
            crossSegSlots.insert(srcIdx);
          }
        }
      }
      int crossSegCreated = 0;
      for (int slotIdx : crossSegSlots) {
        if (slotIdx >= totalOutputSlots_ || outputSlots_[slotIdx] == nullptr) continue;
        NDArray* src = outputSlots_[slotIdx];
        src->syncToDevice();
        size_t srcBytes = src->lengthOf() * src->sizeOfT();
        if (srcBytes == 0) continue;

        // Create fixed-address capture buffer for this cross-segment input
        auto srcShapeVec = *src->getShapeAsVector();
        auto* capBuf = new NDArray(src->ordering(), srcShapeVec, src->dataType(),
                                   sd::LaunchContext::defaultContext());
        if (src->specialBuffer() && capBuf->specialBuffer()) {
          cudaMemcpyAsync(capBuf->specialBuffer(), src->specialBuffer(),
                          srcBytes, cudaMemcpyDeviceToDevice, cudaStr);
        }

        ReplayCaptureBuffer cb;
        cb.buffer = capBuf;
        cb.externalInputIndex = -1;
        cb.crossSegmentSlotIdx = slotIdx;
        cb.capturedSize = srcBytes;
        cb.neverSkipCopy = true;
        seg.replayHandle->addCaptureBuffer(std::move(cb));

        // Swap outputSlots_ so graph captures with capture buffer addresses
        savedSlotsForCapture.push_back({slotIdx, outputSlots_[slotIdx]});
        outputSlots_[slotIdx] = capBuf;
        crossSegCreated++;
      }
      if (crossSegCreated > 0) {
        cudaStreamSynchronize(cudaStr);
        DSP_DIAG(EXECUTE, "CAPTURE_BUFFERS: created %d buffers for CROSS-SEGMENT slot inputs",
                 crossSegCreated);
      }
    }

    // Pre-capture batch-zero: zero all registered buffers BEFORE beginCapture.
    // These cudaMemsetAsync calls execute normally on the stream (not captured).
    // This ensures ops get zeroed outputs during the capture run for correct results.
    // During capture, individual nullify() calls are suppressed (no memset graph nodes).
    // On replay, the same zeroing happens via pre-replay batch-zero above.
    if (Environment::getInstance().dspBatchZero() && !batchZeroEntries_.empty()) {
      for (auto& entry : batchZeroEntries_) {
        if (entry.ptr != nullptr && entry.bytes > 0) {
          cudaMemsetAsync(entry.ptr, 0, entry.bytes, cudaStr);
        }
      }
      DSP_DIAG(MEMORY, "pre-capture batch-zero: %d buffers zeroed via cudaMemsetAsync (fill engines, before beginCapture)",
                static_cast<int>(batchZeroEntries_.size()));
    }

    auto* cudaReplay = static_cast<CudaGraphReplayHandle*>(seg.replayHandle.get());
    auto handle = cudaReplay->getNativeHandle();
    bool captureOk = handle->beginCapture(cudaStr, cudaStreamCaptureModeRelaxed);
    if (captureOk) {
      DSP_DIAG_SEG(EXECUTE, seg.startSlot, "Triton graph capture started for seg[%d-%d] execCount=%d",
                seg.startSlot, seg.endSlot, seg.executionCount);
      tl_graphExecutionActive = true;

      // Batch-zero during capture: DON'T launch inside the graph — instead,
      // suppress individual nullify() calls so no memset nodes get captured.
      // The actual zeroing happens OUTSIDE the graph before each replay() call
      // using cudaMemsetAsync (fill engines, no SM competition).
      // This removes ~700 memset graph nodes while keeping fill-engine efficiency.
      if (Environment::getInstance().dspBatchZero() && !batchZeroEntries_.empty()) {
        setBatchZeroActive(true);
        DSP_DIAG(MEMORY, "batch-zero CAPTURE-SKIP: suppressing %d individual nullify() calls, "
                  "zeroing will happen outside graph before replay",
                  static_cast<int>(batchZeroEntries_.size()));

        // CRITICAL: Mark ALL output slot DataBuffers as device-actual (sAct=1)
        // after batch-zero.  Batch-zero zeroes device memory directly via a GPU
        // kernel, bypassing NDArray's actuality tracking.  Without this,
        // DataBuffer::syncToSpecial() inside native gap ops sees sAct=0 (stale
        // from a previous step) and generates an H2D memcpy that gets RECORDED
        // in the CUDA graph.  On replay, that H2D copies STALE host data
        // (from capture time) over the freshly batch-zeroed device buffer,
        // corrupting inputs to downstream ops.
        //
        // By marking sAct=1 here, syncToSpecial() during capture becomes a
        // no-op for internal buffers (device is already "actual" — it has
        // zeros, which is the correct initial state).  This matches the
        // standard CUDA graph path which uses capture buffers with correct
        // actuality.
        int markedCount = 0;
        for (int si = seg.startSlot; si <= seg.endSlot; si++) {
          for (int o = 0; o < slots_[si].numOutputs; o++) {
            int outIdx = slots_[si].outputSlotIndices[o];
            if (outIdx >= 0 && outIdx < totalOutputSlots_ && outputSlots_[outIdx]) {
              auto* db = outputSlots_[outIdx]->dataBuffer();
              if (db) {
                db->writeSpecial();
                markedCount++;
              }
            }
          }
        }
        DSP_DIAG(MEMORY, "batch-zero actuality: marked %d output DataBuffers as device-actual",
                  markedCount);
        if (Environment::getInstance().tritonVerifyKernels()) {
          DSP_DIAG(VERIFY, "SLOT_WRITE tag=BATCH_ZERO seg[%d-%d] %d buffers suppressed (nullify skipped), %d marked sAct=1",
                    seg.startSlot, seg.endSlot, static_cast<int>(batchZeroEntries_.size()), markedCount);
        }
      } else {
        DSP_DIAG(MEMORY, "batch-zero DISABLED (dspBatchZero=%d, entries=%d)",
                  (int)Environment::getInstance().dspBatchZero(), static_cast<int>(batchZeroEntries_.size()));
      }

      // Query node count mid-capture to verify operations are being recorded
      size_t midCaptureNodes = handle->getNumNodesDuringCapture(cudaStr);
      DSP_DIAG(EXECUTE, "Triton capture mid-check: %zu nodes recorded before executeSegment (batchZero=%d entries, outside-graph)",
                midCaptureNodes, static_cast<int>(batchZeroEntries_.size()));

      // Snapshot all buffer addresses at capture entry — compare with replay to detect stale pointers
      {
        std::vector<void*> outAddrs, extAddrs;
        extractDeviceAddrs(outputSlots_, totalOutputSlots_, outAddrs);
        extractDeviceAddrs(externalArrays, numExt, extAddrs);
        DspDiagnostics::getInstance().clearAddressSnapshots();
        DSP_DIAG_SNAPSHOT_ADDRS("capture-entry", outAddrs.data(), totalOutputSlots_,
                                 extAddrs.data(), numExt);
      }

      auto captureStatus = backend->executeSegment(seg, slots_, externalArrays, numExt,
                                                   outputSlots_, totalOutputSlots_, stream);
      setBatchZeroActive(false);
      tl_graphExecutionActive = false;

      // Snapshot addresses AFTER capture execution to detect pointer changes during capture
      {
        std::vector<void*> outAddrs, extAddrs;
        extractDeviceAddrs(outputSlots_, totalOutputSlots_, outAddrs);
        extractDeviceAddrs(externalArrays, numExt, extAddrs);
        DSP_DIAG_SNAPSHOT_ADDRS("capture-exit", outAddrs.data(), totalOutputSlots_,
                                 extAddrs.data(), numExt);
        int changed = DSP_DIAG_COMPARE_ADDRS("capture-entry", "capture-exit");
        if (changed > 0) {
          DSP_DIAG(EXECUTE, "WARNING: %d buffer addresses CHANGED during capture execution!", changed);
        }
      }

      // Diagnostic: capture workspace usage
      DSP_DIAG(MEMORY, "capture workspace used: %zu / %zu bytes (%.1f%%)",
               tl_captureWorkspaceOffset, seg.replayHandle->getWorkspaceBytes(),
               seg.replayHandle->getWorkspaceBytes() > 0 ? (100.0 * tl_captureWorkspaceOffset / seg.replayHandle->getWorkspaceBytes()) : 0.0);
      // Check for CUDA errors generated during capture — these become invalid graph nodes.
      // Don't use cudaGetLastError (which clears) — peek first for diagnostics.
      {
        cudaError_t capPhaseErr = cudaPeekAtLastError();
        if (capPhaseErr != cudaSuccess) {
          DSP_DIAG(BACKEND, "WARNING - CUDA error during Triton capture phase: %s (%d)",
                    cudaGetErrorString(capPhaseErr), (int)capPhaseErr);
          // Clear it so endCapture can proceed (the graph may still be partially valid)
          cudaGetLastError();
        }
      }

      // Query node count after execution to see how many ops were captured
      size_t postExecNodes = 0;
      {
        cudaStreamCaptureStatus capStat = cudaStreamCaptureStatusNone;
        cudaGraph_t capGraph = nullptr;
        unsigned long long capId = 0;
        auto capErr = cudaStreamGetCaptureInfo_v2(cudaStr, &capStat, &capId, &capGraph, nullptr, nullptr);
        if (capErr == cudaSuccess && capGraph != nullptr) {
          cudaGraphGetNodes(capGraph, nullptr, &postExecNodes);
        }
      }
      DSP_DIAG(EXECUTE, "Triton capture post-exec: %zu nodes, captureStatus=%d",
                postExecNodes, static_cast<int>(captureStatus));
      fflush(stdout); fflush(stderr);

      bool endOk = false;
      if (captureStatus == Status::OK) {
        endOk = handle->endCapture(cudaStr);
      } else {
        DSP_DIAG(FALLBACK, "Triton capture execution FAILED status=%d for seg[%d-%d]",
                  static_cast<int>(captureStatus), seg.startSlot, seg.endSlot);
        fflush(stdout); fflush(stderr);
        if (handle->isCapturing()) {
          handle->endCapture(cudaStr);
        }
      }

      if (endOk) {
        size_t numGraphNodes = handle->getNumNodes();
        DSP_DIAG(EXECUTE, "Triton capture endOk: graph has %zu nodes", numGraphNodes);

        // Sample final output AFTER endCapture (stream no longer capturing, safe)
        if (seg.endSlot < totalOutputSlots_ && outputSlots_[seg.endSlot] != nullptr) {
          auto* finalOut = outputSlots_[seg.endSlot];
          if (finalOut->dataType() == FLOAT32) {
            DSP_DIAG_DUMP_SLOT("capture-post-endCapture", seg.endSlot,
                               finalOut->specialBuffer(), finalOut->lengthOf());
          }
        }
        // Dump top logit from capture execution via DSP_DIAG
        // Use outputSlotIndices[0] to get the ACTUAL final output slot
        // (matches GRAPH_REPLAY logic for apples-to-apples comparison)
        {
          int captureOutputSlot = -1;
          if (seg.endSlot < numSlots_ && slots_[seg.endSlot].numOutputs > 0) {
            captureOutputSlot = slots_[seg.endSlot].outputSlotIndices[0];
          }
          if (captureOutputSlot < 0 || captureOutputSlot >= totalOutputSlots_) {
            captureOutputSlot = seg.endSlot;
          }
          if (captureOutputSlot >= 0 && captureOutputSlot < totalOutputSlots_ &&
              outputSlots_[captureOutputSlot] != nullptr) {
            auto* out = outputSlots_[captureOutputSlot];
            if (out->dataType() == FLOAT32 && out->lengthOf() > 0) {
              DSP_DIAG_DUMP_SEG_OUTPUT("CAPTURE_EXEC", captureOutputSlot, out->specialBuffer(),
                                       out->lengthOf(), seg.executionCount, stream);
            }
          }
        }
      }

      if (endOk) {
        auto stats = handle->getStatistics();
        DSP_DIAG(EXECUTE, "Triton graph stats: %d kernels, %d memcpys, %d memsets, "
                  "%d memAllocs, %d memFrees, %d hostCallbacks, %d events, %d empty",
                  stats.numKernels, stats.numMemcpyH2D, stats.numMemsets,
                  stats.numMemAllocs, stats.numMemFrees,
                  stats.numHostCallbacks, stats.numEvents, stats.numEmpty);
        fflush(stdout); fflush(stderr);
        if (stats.numMemAllocs > 0 || stats.numMemFrees > 0) {
          DSP_DIAG(EXECUTE, "Triton graph has %d MemAlloc + %d MemFree nodes "
                    "(paired alloc/free from cuBLAS internal workspace - CUDA 12+ handles these on replay).",
                    stats.numMemAllocs, stats.numMemFrees);
        }
        if (stats.numHostCallbacks > 0) {
          DSP_DIAG(BACKEND, "WARNING - Graph has %d host callback nodes!",
                    stats.numHostCallbacks);
        }
      }

      // Skip DOT dump by default for Triton graphs — cudaGraphDebugDotPrint with verbose
      // flags may also call cudaGraphKernelNodeGetParams internally, causing the same
      // cudaErrorInvalidDeviceFunction poisoning as getDetailedNodeInfo().
      if (endOk && Environment::getInstance().tritonDumpGraphDot()) {
        cudaGraphDebugDotPrint(handle->getGraph(), "/tmp/triton_graph_debug.dot", 0);
        DSP_DIAG(EXECUTE, "Triton graph dumped to /tmp/triton_graph_debug.dot");
        fflush(stdout); fflush(stderr);
      }

      // Skip getDetailedNodeInfo() for Triton graphs — it calls cudaFuncGetName on each
      // kernel node, which returns cudaErrorInvalidDeviceFunction (error 98) for Triton
      // kernels loaded via cuModuleLoadDataEx (driver API). The 658+ consecutive errors
      // poison the CUDA error state and cause cudaGraphLaunch to SIGSEGV.
      // Use getNumNodes() for basic stats instead (no per-node introspection).
      bool allKernelsValid = true;
      if (endOk) {
#ifdef SD_CUDA
        size_t totalNodes = handle->getNumNodes();
        DSP_DIAG(EXECUTE, "Triton graph has %zu nodes (skipping per-node inspection to avoid error-98 poisoning)",
                  totalNodes);
        fflush(stdout); fflush(stderr);
        // Ensure no sticky errors before instantiation
        cudaGetLastError();
#endif
      }

      bool instantiateOk = endOk && allKernelsValid && handle->instantiate();
      if (instantiateOk) {
        DSP_DIAG(EXECUTE, "Triton graph instantiated OK (graphExec=%p), about to launch...",
                  handle->getGraphExec());
        fflush(stdout); fflush(stderr);
      }

      // Try launch with pre/post sync to catch async errors
      bool launchOk = false;
      if (instantiateOk) {
        // Sync before launch to ensure no pending errors
        cudaStreamSynchronize(cudaStr);
        auto preErr = cudaGetLastError();
        if (preErr != cudaSuccess) {
          DSP_DIAG(BACKEND, "WARNING - pre-launch CUDA error: %s",
                    cudaGetErrorString(preErr));
        }

        // Refresh Triton arg table pinned buffers before first launch.
        // During capture, the arg tables were populated from outputSlots_.
        // Between endCapture and launch, async frees on other streams may
        // have invalidated some device pointers. Re-resolve all pointers
        // from the current outputSlots_ (slotArrayCache_) to ensure the
        // graph's H2D memcpy nodes transfer valid addresses.
#if HAVE_TRITON
        {
          // Temporarily store shape key so refreshArgTablesForReplay can find
          // the compiled segment. seg hasn't been stored yet, so use local.
          LongType savedShapeKey = seg.shapeKey;
          seg.shapeKey = segShapeKey;
          auto* tritonBackend = dynamic_cast<TritonGraphBackend*>(backend);
          if (tritonBackend != nullptr) {
            tritonBackend->refreshArgTablesForReplay(seg, externalArrays, numExt,
                                                     outputSlots_, totalOutputSlots_,
                                                     stream);
          }
          seg.shapeKey = savedShapeKey;
        }
#endif

        // Launch on the capture stream.
        DSP_DIAG(EXECUTE, "Triton graph launching on capture stream %p (device=%d)",
                  (void*)cudaStr, tritonCaptureDevice);
        launchOk = handle->launchAsync(cudaStr);
      }

      if (launchOk) {
        if (seg.endSlot < totalOutputSlots_ && outputSlots_[seg.endSlot] != nullptr) {
          auto* finalOut = outputSlots_[seg.endSlot];
          if (finalOut->dataType() == FLOAT32) {
            DSP_DIAG_DUMP_SLOT("capture-post-launch", seg.endSlot,
                               finalOut->specialBuffer(), finalOut->lengthOf());
          }
        }
        // Dump top logit from first replay (graph launch after capture) via DSP_DIAG
        if (seg.endSlot < totalOutputSlots_ && outputSlots_[seg.endSlot] != nullptr) {
          auto* out = outputSlots_[seg.endSlot];
          if (out->dataType() == FLOAT32 && out->lengthOf() > 0) {
            DSP_DIAG_DUMP_SEG_OUTPUT("REPLAY_LAUNCH", seg.endSlot, out->specialBuffer(),
                                     out->lengthOf(), seg.executionCount, stream);
          }
        }
        // replayHandle already set (created before capture began)
        seg.cachedShapeKey = segShapeKey;
        seg.capturedInputAddrKey = segInputAddrKey;
        seg.capturedCreateValueKey = createValueKey;
        snapshotExternalAddrs(seg, externalArrays, numExt);

        // Export graph stats and DOT file for diagnostics
        auto stats = handle->getStatistics();
        DSP_DIAG_SEG(EXECUTE, seg.startSlot, "Triton graph CAPTURED and launched for seg[%d-%d]: "
                  "%d kernels, %d memcpy, %d memset, %d memAlloc, %d memFree "
                  "(workspace=%zuMB, offset=%zu)",
                  seg.startSlot, seg.endSlot,
                  stats.numKernels, stats.numMemcpyH2D + stats.numMemcpyD2H + stats.numMemcpyD2D,
                  stats.numMemsets, stats.numMemAllocs, stats.numMemFrees,
                  seg.replayHandle->getWorkspaceBytes() / (1024*1024), tl_captureWorkspaceOffset);
        // Write DOT file for offline analysis.
        // Default: non-verbose (flag 0). Verbose queries kernel node params via
        // cudaFuncGetName, which returns cudaErrorInvalidDeviceFunction for
        // Triton CUfunction handles and may poison driver state.
        // Enable via ND4J_TRITON_GRAPH_DOT_VERBOSE=1 for debugging.
        {
          std::string dotPath = "/tmp/triton_graph_captured.dot";
          unsigned int dotFlags = Environment::getInstance().tritonGraphDotVerbose()
              ? cudaGraphDebugDotFlagsVerbose : 0;
          auto dotErr = cudaGraphDebugDotPrint(handle->getGraph(), dotPath.c_str(), dotFlags);
          if (dotErr == cudaSuccess) {
            DSP_DIAG(EXECUTE, "Exported Triton graph DOT to %s (verbose=%d)",
                      dotPath.c_str(), dotFlags != 0);
          }
          cudaGetLastError(); // Clear any error from dot print
        }
        // Write stats to a file the test can read
        {
          FILE* f = fopen("/tmp/triton_graph_stats.txt", "w");
          if (f) {
            fprintf(f, "segment=%d-%d\n", seg.startSlot, seg.endSlot);
            fprintf(f, "kernels=%d\n", stats.numKernels);
            fprintf(f, "memcpyH2D=%d\n", stats.numMemcpyH2D);
            fprintf(f, "memcpyD2H=%d\n", stats.numMemcpyD2H);
            fprintf(f, "memcpyD2D=%d\n", stats.numMemcpyD2D);
            fprintf(f, "memsets=%d\n", stats.numMemsets);
            fprintf(f, "memAllocs=%d\n", stats.numMemAllocs);
            fprintf(f, "memFrees=%d\n", stats.numMemFrees);
            fprintf(f, "hostCallbacks=%d\n", stats.numHostCallbacks);
            fprintf(f, "events=%d\n", stats.numEvents);
            fprintf(f, "childGraphs=%d\n", stats.numChildGraphs);
            fprintf(f, "totalNodes=%zu\n", handle->getNumNodes());
            fclose(f);
          }
        }
        status = Status::OK;
        usedTritonGraphCapture = true;

        // Update slotArrayCache_ to point to the capture-time arrays.
        // During capture, gap ops may have allocated output buffers from the
        // workspace (tl_graphExecutionActive=true → CudaMemoryPool allocates
        // from workspace). These workspace-allocated arrays are stored in
        // outputSlots_[]. The graph's kernel nodes write to these workspace
        // addresses on replay. If slotArrayCache_ still points to the
        // pre-capture (warmup) arrays, the pre-execution slot restoration
        // (lines 467-559) would overwrite outputSlots_[] with old pointers,
        // causing graph replay to write to invisible workspace addresses
        // while outputSlots_[] points elsewhere.
        for (int s = seg.startSlot; s <= seg.endSlot; s++) {
          NativeSlot& slot = slots_[s];
          for (int o = 0; o < slot.numOutputs; o++) {
            int si = slot.outputSlotIndices[o];
            if (si >= 0 && si < totalOutputSlots_ && outputSlots_[si] != nullptr) {
              slotArrayCache_[si] = outputSlots_[si];
            }
          }
        }

        // FORCE_RECAPTURE: invalidate graph immediately after capture+launch
        // so the NEXT step also re-captures instead of replaying a stale graph.
        // This ensures every single step is a fresh capture+launch with zero replays.
        if (Environment::getInstance().tritonForceRecapture()) {
          if (seg.replayHandle) {
            for (auto* ptr : seg.replayHandle->getCapturedHostPtrs()) { cudaFreeHost(ptr); }
            seg.replayHandle->getCapturedHostPtrs().clear();
            seg.replayHandle->clearExternalAddresses();
          }
          seg.replayHandle.reset();
          seg.argTableStable = false;
          batchD2DCount_ = 0;
          seg.capturedInputAddrKey = 0;
          DSP_DIAG(EXECUTE, "FORCE_RECAPTURE: invalidated after capture+launch execCount=%d", seg.executionCount);
        }
      } else {
        DSP_DIAG(FALLBACK, "Triton graph capture/instantiate/launch FAILED for seg[%d-%d] "
                  "(endOk=%d instantiateOk=%d launchOk=%d)",
                  seg.startSlot, seg.endSlot,
                  static_cast<int>(endOk), static_cast<int>(instantiateOk),
                  static_cast<int>(launchOk));
        cudaGetLastError();
      }
    } else {
      DSP_DIAG(FALLBACK, "Triton graph capture beginCapture FAILED for seg[%d-%d]",
                seg.startSlot, seg.endSlot);
      cudaGetLastError();
    }

    // Restore original external arrays after capture (undo capture buffer wiring)
    for (auto& [extIdx, origArr] : savedExtForCapture) {
      externalArrays[extIdx] = origArr;
    }

    // Restore original output slots after capture (undo cross-segment capture buffer wiring)
    for (auto& [slotIdx, origArr] : savedSlotsForCapture) {
      outputSlots_[slotIdx] = origArr;
    }

    // Restore primary CUDA context if we pushed it
    if (didPushCtx) {
      CUcontext dummy;
      cuCtxPopCurrent(&dummy);
      CUdevice cuDev;
      cuDeviceGet(&cuDev, tritonCaptureDevice);
      cuDevicePrimaryCtxRelease(cuDev);
    }

    // Restore cuBLAS workspace to default (undo setCublasWorkspaceForCapture)
    restoreCublasWorkspaceAfterCapture(stream);

    // Reset thread-local state after capture attempt
    tl_captureWorkspace = nullptr;
    tl_captureWorkspaceSize = 0;
    tl_captureWorkspaceOffset = 0;
    tl_graphCaptureStream = prevCaptureStream;
    // Pinned host ptrs: graph's H2D memcpy nodes reference these on replay.
    // On success: move to segment so they persist for graph lifetime.
    // On failure: free immediately (no graph to replay).
    if (usedTritonGraphCapture && seg.replayHandle) {
      for (auto* ptr : tl_capturedHostPtrs) {
        seg.replayHandle->addCapturedHostPtr(ptr);
      }
      DSP_DIAG(MEMORY, "preserved %zu pinned host ptrs for Triton graph replay",
                seg.replayHandle->getCapturedHostPtrs().size());
    } else {
      // No graph captured — free pinned host ptrs immediately
      for (auto* ptr : tl_capturedHostPtrs) {
        cudaFreeHost(ptr);
      }
    }
    tl_capturedHostPtrs.clear();
    tl_captureReplicateCache.clear();

    // Flush arrays that accumulated in pendingClose_ during capture.
    // flushPendingClose was skipped inside capture to avoid recording
    // cudaFreeAsync MemFree graph nodes for external memory. Safe to free now.
    if (!pendingClose_.empty()) {
      flushPendingClose(stream);
    }

    // Restore frozen context state so subsequent executions (including graph replay
    // steps that fall through to direct execution) use the frozen fast path.
    for (int s = seg.startSlot; s <= seg.endSlot; s++) {
      slots_[s].frozenContextReady = savedFrozenContextReadyTriton[s - seg.startSlot];
    }
  }
#endif

  if (!usedTritonGraphCapture) {
    // ── Batch-zero registration: learn which buffers actually get nullified ──
    // On the execution right before capture (executionCount == 1 → next call is
    // executionCount == 2 which triggers capture), enable registration mode.
    // Each nullify() site calls registerBatchZeroBuffer() when registering,
    // building the exact set of buffers that need zeroing.
    // This replaces the pre-scan approach (collectBatchZeroTargets) which
    // collected ~143 EXTRA buffers for slots that don't actually execute,
    // including buffers whose GPU addresses alias external KV cache inputs.
    bool batchZeroRegistrationActive = false;
#ifdef SD_CUDA
    {
      // Check the same conditions as shouldCaptureTritonGraph but for executionCount==1
      // (the warmup step right BEFORE capture). We register which buffers get nullified
      // so the batch-zero kernel during capture zeros EXACTLY the right set.
      // Registration doesn't require shapesFrozen_ — shapes may freeze after
      // this execution but before capture. We just need to be the pre-capture
      // warmup step (executionCount == 1) with no existing graph.
      bool wouldCaptureNextStep =
          Environment::getInstance().tritonGraphCapture() &&
          (!seg.replayHandle || seg.replayHandle->getCaptureBuffers().empty()) &&
          seg.replayHandle == nullptr &&
          !seg.captureFailed &&
          seg.executionCount == 1;
      if (Environment::getInstance().dspBatchZero() && wouldCaptureNextStep) {
        startBatchZeroRegistration();
        batchZeroRegistrationActive = true;
        DSP_DIAG_SEG(MEMORY, seg.startSlot, "batch-zero registration enabled for warmup execution (seg[%d-%d] execCount=%d)",
                  seg.startSlot, seg.endSlot, seg.executionCount);
      }
    }
#endif

    // ── Sync external inputs to device BEFORE setting tl_graphExecutionActive ──
    // Triton's arg table population uses specialBuffer() to resolve GPU pointers.
    // specialBuffer() only calls syncToDevice() when the device buffer is nullptr
    // or on the wrong device — it does NOT check if the device data is stale.
    // Java modifies external inputs (attention_mask, position_ids, input_ids) on the
    // host via putScalar() + tagLocation(HOST), making the device data stale.
    // Native ops handle this via prepareSpecialUse() which calls syncToDevice()
    // unconditionally, but Triton bypasses native ops and reads device buffers directly.
    // We must sync BEFORE setting tl_graphExecutionActive because that flag changes
    // syncToSpecial() to use an async path that skips cudaStreamSynchronize.
    for (int ei = 0; ei < numExt; ei++) {
      if (externalArrays[ei] != nullptr) {
        if (Environment::getInstance().tritonVerifyKernels()) {
          auto* db = externalArrays[ei]->dataBuffer();
          DSP_DIAG(VERIFY, "SLOT_WRITE slot=%d tag=EXT_SYNC(direct) extIdx=%d pAct=%d sAct=%d len=%lld addr=%p",
                    -(ei + 1), ei,
                    db ? (db->isPrimaryActual() ? 1 : 0) : -1,
                    db ? (db->isSpecialActual() ? 1 : 0) : -1,
                    (long long)externalArrays[ei]->lengthOf(),
                    externalArrays[ei]->specialBuffer());
        }
        externalArrays[ei]->syncToDevice();
      }
    }

    // NOTE: Do NOT set tl_graphExecutionActive=true here for non-capture Triton execution.
    // That flag suppresses syncToPrimary (D2H transfers), error checking, and
    // PointersManager sync -- behaviors only appropriate during CUDA graph capture.
    // The fallback lambda (gap ops) already handles capture detection independently:
    // it checks cudaStreamIsCapturing() and only sets tl_graphExecutionActive=true
    // when actually capturing.  Setting it unconditionally here caused gap ops
    // (matmul, gather, etc.) to read stale host data, producing wrong output.

    // Disable frozen fast path for gap ops during Triton segment execution.
    // Same rationale as the capture path (lines 5325-5329): the pre-execution
    // slot restoration at lines 4955-5032 may replace NDArray objects in
    // outputSlots_[], making the frozen context's cached input/output pointers
    // stale. Without clearing frozenContextReady, gap ops write to old arrays
    // while downstream ops read from new arrays, producing wrong output.
    // Save and restore so subsequent executions still benefit from frozen fast path.
    std::vector<bool> savedFrozenContextReadyNonCapture(seg.endSlot - seg.startSlot + 1);
    for (int s = seg.startSlot; s <= seg.endSlot; s++) {
      savedFrozenContextReadyNonCapture[s - seg.startSlot] = slots_[s].frozenContextReady;
      slots_[s].frozenContextReady = false;
    }

    // Snapshot addresses for direct execution (baseline for comparison with capture/replay)
    {
      std::vector<void*> outAddrs, extAddrs;
      extractDeviceAddrs(outputSlots_, totalOutputSlots_, outAddrs);
      extractDeviceAddrs(externalArrays, numExt, extAddrs);
      DSP_DIAG_SNAPSHOT_ADDRS("direct-entry", outAddrs.data(), totalOutputSlots_,
                               extAddrs.data(), numExt);
    }

    try {
      status = backend->executeSegment(seg, slots_, externalArrays, numExt,
                                       outputSlots_, totalOutputSlots_, stream);
    } catch (...) {
      // Restore frozenContextReady on exception
      for (int s = seg.startSlot; s <= seg.endSlot; s++) {
        slots_[s].frozenContextReady = savedFrozenContextReadyNonCapture[s - seg.startSlot];
      }
#ifdef SD_CUDA
      if (batchZeroRegistrationActive) {
        finishBatchZeroRegistration();
      }
#endif
      throw;  // Re-throw after cleanup
    }

    // Restore frozen context state so subsequent calls use the frozen fast path
    // once context pointers are re-established by the normal path above.
    for (int s = seg.startSlot; s <= seg.endSlot; s++) {
      slots_[s].frozenContextReady = savedFrozenContextReadyNonCapture[s - seg.startSlot];
    }

#ifdef SD_CUDA
    if (batchZeroRegistrationActive) {
      finishBatchZeroRegistration();
    }
#endif
  }

  // Dump final output for direct Triton path (baseline comparison)
  if (status == Status::OK && seg.endSlot < totalOutputSlots_ &&
      outputSlots_[seg.endSlot] != nullptr) {
    auto* finalOut = outputSlots_[seg.endSlot];
    if (finalOut->dataType() == FLOAT32) {
      DSP_DIAG_DUMP_SLOT("direct", seg.endSlot,
                         finalOut->specialBuffer(), finalOut->lengthOf());
    }
  }
  // Always-on diagnostic: dump top logit for non-capture Triton execution
  if (!usedTritonGraphCapture && status == Status::OK &&
      seg.endSlot < totalOutputSlots_ && outputSlots_[seg.endSlot] != nullptr) {
    auto* out = outputSlots_[seg.endSlot];
    if (out->dataType() == FLOAT32 && out->lengthOf() > 0) {
      DSP_DIAG_DUMP_SEG_OUTPUT("DIRECT_TRITON", seg.endSlot, out->specialBuffer(),
                               out->lengthOf(), seg.executionCount, stream);
    }
  }

  DSP_DIAG(EXECUTE, "executeSegmentWithGpuGraph: exec%d seg[%d-%d]: backend=%s %s status=%d(%s) "
            "executionCount=%d captureFailed=%d usedCapture=%d",
            seg.executionCount, seg.startSlot, seg.endSlot,
            backendName, status == Status::OK ? "OK" : "FAILED",
            static_cast<int>(status), statusName_gpu(status),
            seg.executionCount,
            seg.captureFailed ? 1 : 0, usedTritonGraphCapture ? 1 : 0);

  if (status == Status::OK) {
    seg.executionCount++;
    totalGraphReplays_++;
    if (seg.compiledByBackend.empty()) {
      seg.compiledByBackend = backendName;
    }
  }

#ifdef SD_CUDA
  if (Environment::getInstance().tritonVerifyKernels()) {
    DSP_DIAG(VERIFY, "SEG_EXIT seg[%d-%d] status=%s execCount=%d",
              seg.startSlot, seg.endSlot, statusName_gpu(status), seg.executionCount);
  }
#endif

  return status;
}

}  // namespace graph
}  // namespace sd

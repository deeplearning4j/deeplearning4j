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

  // Compute shape key for cache lookup
  LongType segShapeKey = computeSegmentShapeKey(seg, externalArrays, numExt);

  // Diagnostic: scan all slotArrayCache_ entries for freed DataBuffers.
  // Java may have closed DSP output arrays between steps (e.g., prefill KV outputs via
  // setCloseable(true)+close()), deleting the C++ NDArray and leaving dangling pointers.
  // Run on EVERY execution — stale entries can appear between any two steps.
  {
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
          // If this was a frozenConstantSlot (e.g. shape_of, zeros_like), its
          // executeSlot() returns OK immediately without re-computing.  With its
          // cached array gone the output stays nullptr, breaking downstream ops.
          // Clear the flag so the op re-executes and re-populates the cache.
          if (si < numSlots_ && slots_[si].frozenConstantSlot) {
            slots_[si].frozenConstantSlot = false;
          }
          invalidCount++;
        }
      }
    }
    // Also scan external inputs (skip empty arrays — they legitimately have no DataBuffer)
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
      // Stale external inputs have null DataBuffers — proceeding would SIGSEGV
      // when the GPU kernel tries to read from a freed buffer. Invalidate the
      // cached graph so it gets re-captured with fresh pointers on the next call.
#ifdef SD_CUDA
      seg.replayHandle.reset();
      seg.cachedShapeKey = 0;
#endif
      seg.captureFailed = false;
      DSP_DIAG(FALLBACK, "invalidated graph for seg[%d-%d] "
                "due to %d stale entries - executing slot-by-slot this step",
                seg.startSlot, seg.endSlot, invalidCount);
      // Execute slot-by-slot as a one-time fallback instead of returning
      // KERNEL_FAILURE (which aborts in forced-backend modes like TRITON).
      return executeSegmentSlotBySlot(seg, externalArrays, numExt, stream);
    }
  }

  // Compile once per stable shape; skip cache probe on steady-state replay.
  // This keeps the hot path focused on dispatch instead of repeated compile checks.
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

  // Pre-execution: ensure all output slots in the segment have live arrays.
  // The Triton kernel's arg mapping references outputSlots_ for both inputs
  // (from prior ops) and outputs (to write results). Slot-by-slot warmup may
  // have released intermediate arrays via releaseAtStep_, leaving entries null.
  // First restore from slotArrayCache_, then allocate any remaining nulls
  // using cached shape info from warmup.
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
  int preExecAllocCount = 0;
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
    seg.capturedInputAddrKey = 0;
    // Reset execution count to trigger warmup→capture flow
    seg.executionCount = 0;
    seg.captureFailed = false;
  }

  bool allowTritonCudaGraphReplay = Environment::getInstance().tritonGraphCapture() &&
                                    shapesFrozen_;
  LongType segInputAddrKey = computeSegmentInputAddrKey(seg, externalArrays, numExt);

  // Per-address comparison: catches address changes the hash may miss.
  bool extAddrsStable = (seg.replayHandle && !seg.replayHandle->getCapturedExternalAddresses().empty())
      ? externalAddrsMatch(seg, externalArrays, numExt)
      : (seg.capturedInputAddrKey != 0 && seg.capturedInputAddrKey == segInputAddrKey);

  // Create (ConstantOfShape) op value validation: their output shapes depend on
  // input DATA values.  If values changed since capture, the baked-in memset is wrong.
  LongType createValueKey = computeCreateOpValueKey(seg, externalArrays, numExt);
  bool createValuesStable = (createValueKey == 0) ||  // no create ops
                            (seg.capturedCreateValueKey == createValueKey);
  if (!createValuesStable && seg.replayHandle) {
    DSP_DIAG(EXECUTE, "CREATE_VALUE_KEY mismatch: captured=%lld current=%lld → invalidating graph seg[%d-%d]",
             (long long)seg.capturedCreateValueKey, (long long)createValueKey, seg.startSlot, seg.endSlot);
    for (auto* ptr : seg.replayHandle->getCapturedHostPtrs()) { cudaFreeHost(ptr); }
    seg.replayHandle->getCapturedHostPtrs().clear();
    seg.replayHandle->clearExternalAddresses();
    seg.replayHandle.reset();
    seg.capturedInputAddrKey = 0;
    seg.capturedCreateValueKey = 0;
    seg.executionCount = 0;
    seg.captureFailed = false;
    extAddrsStable = false;  // Force re-capture path
  }

  // Triton-captured graphs use direct external buffer addresses (no capture buffers),
  // so replay is valid only when both shape key and input-address key match.
  if (allowTritonCudaGraphReplay &&
      (!seg.replayHandle || seg.replayHandle->getCaptureBuffers().empty()) &&
      seg.replayHandle != nullptr &&
      seg.replayHandle->isReady() &&
      seg.cachedShapeKey == segShapeKey &&
      extAddrsStable) {
    // Sync external inputs to device before graph replay.
    // Java's DynamicShapePlanExecutor already calls syncToSpecial() for placeholder inputs
    // (line 4213), so device buffers should have current values by the time we get here.
    // Use syncToDevice() which respects DataBuffer actuality flags — if Java already synced,
    // this is a no-op (correct). If something needs sync, it will be done.
    // NOTE: Do NOT force H2D for model weights (SOURCE_VARIABLE) — their device buffers
    // are authoritative after model load. Host buffers may have stale data.
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

      // Dump SMALL variable external inputs (position_ids, input_ids, attention_mask head)
      // to verify data is actually changing between replays (verify mode only)
      if (Environment::getInstance().tritonVerifyKernels()) {
      cudaDeviceSynchronize();
      for (int ei = 0; ei < numExt; ei++) {
        if (externalArrays[ei] == nullptr) continue;
        auto* arr = externalArrays[ei];
        // Dump small arrays fully, large arrays just metadata
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
        // Only log if it's a model variable input (not constant shape params)
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

    // Refresh Triton arg table pinned buffers before replay
#if HAVE_TRITON
    {
      auto* tritonBackend = dynamic_cast<TritonGraphBackend*>(backend);
      if (tritonBackend != nullptr) {
        tritonBackend->refreshArgTablesForReplay(seg, externalArrays, numExt,
                                                 outputSlots_, totalOutputSlots_,
                                                 stream);
      }
    }
#endif
    // Ensure all prior async work completes before replay.
    // Use cudaDeviceSynchronize instead of cudaStreamSynchronize to guarantee
    // that Java-side H2D copies (done on stream 0 by DataBuffer::syncToSpecial)
    // are complete before graph replay starts on the execution stream.
    cudaDeviceSynchronize();
    cudaGetLastError();  // Clear any sticky errors

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

    // Replay strategy: configurable via ND4J_TRITON_GRAPH_REINSTANTIATE.
    // Default (OFF): direct replay of existing graphExec.
    // ON: destroy and re-instantiate graphExec from graph template before each replay.
    {
      bool replayOk = false;
      if (Environment::getInstance().tritonGraphReinstantiate()) {
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

        // Always log argmax after replay (not just in verify mode) for divergence tracking
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
  // Try capturing the fused Triton launch sequence starting from the SECOND
  // Triton execution (executionCount >= 2). executionCount=1 is the first
  // Triton execution after slot-by-slot warmup — it serves as a Triton warmup
  // that validates arg resolution and triggers any lazy device allocations
  // (e.g., indirect arg tables) OUTSIDE stream capture. Attempting capture on
  // the very first Triton execution causes cudaMallocAsync to fail with
  // "operation not permitted when stream is capturing" if any allocation
  // was not fully pre-allocated during compileSegment().
  // Environment property controls minimum execution count for capture.
  // Default=2 (capture on 3rd Triton execution). Set to 9999 to effectively disable capture.
  int captureMinExec = Environment::getInstance().tritonCaptureMinExec();
  bool forceRecaptureEnabled = Environment::getInstance().tritonForceRecapture();
  bool shouldCaptureTritonGraph = allowTritonCudaGraphReplay &&
                                  (!seg.replayHandle || seg.replayHandle->getCaptureBuffers().empty()) &&
                                  seg.replayHandle == nullptr &&
                                  !seg.captureFailed &&
                                  seg.executionCount >= captureMinExec &&
                                  (forceRecaptureEnabled || seg.executionCount <= (captureMinExec + 2)) &&
                                  cudaStr != nullptr;
  if (shouldCaptureTritonGraph) {
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
    setCublasWorkspaceForCapture(stream);

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

    auto* cudaReplay = static_cast<CudaGraphReplayHandle*>(seg.replayHandle.get());
    auto handle = cudaReplay->getNativeHandle();
    bool captureOk = handle->beginCapture(cudaStr, cudaStreamCaptureModeRelaxed);
    if (captureOk) {
      DSP_DIAG_SEG(EXECUTE, seg.startSlot, "Triton graph capture started for seg[%d-%d] execCount=%d",
                seg.startSlot, seg.endSlot, seg.executionCount);
      tl_graphExecutionActive = true;

      // Launch batch-zero kernel INSIDE capture — this becomes a single graph node
      // that replaces ~1000 individual memset nodes. The kernel zeros all output
      // buffers for native fallback ops (matmul, cast, shape_manip, etc.).
      // Controlled by ND4J_DSP_BATCH_ZERO env var (default: disabled)
      if (Environment::getInstance().dspBatchZero() && batchZeroDeviceCount_ > 0) {
        launchBatchZero(cudaStr);
        setBatchZeroActive(true);
        DSP_DIAG(MEMORY, "batch-zero ENABLED (%d buffers)", batchZeroDeviceCount_);

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
          DSP_DIAG(VERIFY, "SLOT_WRITE tag=BATCH_ZERO seg[%d-%d] %d buffers zeroed, %d marked sAct=1",
                    seg.startSlot, seg.endSlot, batchZeroDeviceCount_, markedCount);
        }
      } else {
        DSP_DIAG(MEMORY, "batch-zero DISABLED (dspBatchZero=%d, buffers=%d)",
                  (int)Environment::getInstance().dspBatchZero(), batchZeroDeviceCount_);
      }

      // Query node count mid-capture to verify operations are being recorded
      size_t midCaptureNodes = handle->getNumNodesDuringCapture(cudaStr);
      DSP_DIAG(EXECUTE, "Triton capture mid-check: %zu nodes recorded before executeSegment (batchZero=%d buffers)",
                midCaptureNodes, batchZeroDeviceCount_);

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
        if (seg.endSlot < totalOutputSlots_ && outputSlots_[seg.endSlot] != nullptr) {
          auto* out = outputSlots_[seg.endSlot];
          if (out->dataType() == FLOAT32 && out->lengthOf() > 0) {
            DSP_DIAG_DUMP_SEG_OUTPUT("CAPTURE_EXEC", seg.endSlot, out->specialBuffer(),
                                     out->lengthOf(), seg.executionCount, stream);
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
      bool wouldCaptureNextStep =
          Environment::getInstance().tritonGraphCapture() &&
          shapesFrozen_ &&
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

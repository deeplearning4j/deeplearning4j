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

/**
 * NativeDynamicShapePlan — CUDA Platform Dispatch
 *
 * Contains all CUDA-specific platform dispatch implementations extracted from
 * NativeDynamicShapePlan.cpp. These functions are called by the platform-neutral
 * main .cpp file. On CPU builds, _cuda_stubs.cpp provides no-op/fallback
 * implementations instead.
 *
 * Also contains the CUDA graph capture audit methods (getHostOnlyOps,
 * printCaptureAudit, validateCapturedGraph).
 */

#ifdef SD_CUDA

#include <graph/NativeDynamicShapePlan.h>
#include <graph/ModeContract.h>
#include <graph/NativePlanCompiler.h>
#include <graph/DspDiagnostics.h>
#include <graph/DspPhaseUtils.h>
#include <graph/DspHashUtils.h>
#include <graph/DspVerifyUtils.h>
#include <graph/DspSegmentLifecycle.h>
#include <graph/cuda/CudaGraphReplayHandle.h>
#include <graph/DspStreamGuard.h>
#include <graph/PlanExecutionContext.h>
#include <helpers/ConstantShapeHelper.h>
#include <helpers/ConstantTadHelper.h>
#include <helpers/MmulHelper.h>
#include <helpers/cublasHelper.h>
#include <cublas_v2.h>
#include <memory/cuda/CudaMemoryPool.h>
#include <helpers/AttentionWorkspace.h>
#include <graph/gpu/NvrtcKernelBuilder.h>
#include <graph/gpu/NvrtcKernelCache.h>
#include <ops/declarable/helpers/kv_scatter.h>
#include <system/Environment.h>
// Forward-declare clearCache to avoid circular includes through CudaGraphScheduler.h → graph/Context.h
namespace sd { namespace cuda { void clearCudaGraphSchedulerCache(); } }

#if HAVE_TRITON
#include <graph/gpu/TritonGraphBackend.h>
#include <graph/gpu/OpCategoryTable.h>
#endif

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <future>
#include <thread>
#include <atomic>
#include <numeric>
#include <unordered_set>

// Thread-local cuBLAS workspace — reset in platformFreePlanResources() to
// prevent stale pointer after cublasWorkspaceBuffer_ is freed.
extern SD_TLS_EXPORT thread_local void*  tl_cublasWorkspacePtr;
extern SD_TLS_EXPORT thread_local size_t tl_cublasWorkspaceSize;

// Global TLS that must be reset on plan teardown to prevent cross-plan contamination.
// These are defined in DataBuffer.cu and LaunchContext.cu.
extern SD_TLS_EXPORT thread_local bool tl_graphExecutionActive;
extern SD_TLS_EXPORT thread_local bool tl_dspReplayActive;
extern SD_TLS_EXPORT thread_local cudaStream_t tl_dspExecutionStream;
extern thread_local cudaStream_t tl_dspGapStream;
extern SD_TLS_EXPORT thread_local cudaStream_t tl_graphCaptureStream;
extern SD_TLS_EXPORT thread_local void* tl_captureWorkspace;
extern SD_TLS_EXPORT thread_local size_t tl_captureWorkspaceSize;
extern SD_TLS_EXPORT thread_local size_t tl_captureWorkspaceOffset;
extern SD_TLS_EXPORT thread_local int tl_islandSlotMin;
extern SD_TLS_EXPORT thread_local int tl_islandSlotMax;

namespace sd {

// Disable cublasLt for CUDA_GRAPHS — defined in DataBuffer.cu
extern SD_TLS_EXPORT thread_local bool tl_cublasLtDisabled;

namespace graph {
// File-static TLS in gpubackend.cu — reset via helper function.
extern void resetMergedCaptureTLS();

using SegmentLifecycleState = GraphSegmentExec::SegmentLifecycleState;

using namespace SegmentLifecycle;

namespace {

LongType computeSlotAddrHash(NDArray** outputSlots, int startSlot, int endSlot, int totalSlots) {
  return dsp::computeSlotAddrHash(outputSlots, startSlot, endSlot, totalSlots,
      [](NDArray* a) -> void* { return a->specialBuffer(); });
}

bool bindSegmentCudaDevice(const GraphSegment& segment,
                           NativeSlot* slots,
                           int numSlots,
                           const char* phase) {
  int targetDevice = -1;
  if (segment.def.startSlot >= 0 && segment.def.startSlot < numSlots) {
    targetDevice = slots[segment.def.startSlot].targetDeviceId;
  }
  if (targetDevice < 0) return true;

  // REPLAY OPTIMIZATION: Cache device count and current device to avoid
  // calling cudaGetDeviceCount + cudaGetDevice for every segment (656 per step).
  // Each CUDA runtime call has ~5-10us overhead. Caching saves ~6-13ms per step.
  // Device count never changes during a process lifetime. Current device is
  // tracked via the cached value and only refreshed after cudaSetDevice.
  static thread_local int cachedDeviceCount = -1;
  static thread_local int cachedCurrentDevice = -1;

  if (cachedDeviceCount < 0) {
    int deviceCount = 0;
    cudaError_t countErr = cudaGetDeviceCount(&deviceCount);
    if (countErr != cudaSuccess || deviceCount <= 0) {
      DSP_DIAG(BACKEND, "NativeDSP::execute: %s seg[%d-%d] targetDeviceId=%d but CUDA device query failed: %s",
               phase, segment.def.startSlot, segment.def.endSlot, targetDevice,
               cudaGetErrorString(countErr));
      cudaGetLastError();
      return false;
    }
    cachedDeviceCount = deviceCount;
  }
  if (targetDevice >= cachedDeviceCount) {
    DSP_DIAG(BACKEND, "NativeDSP::execute: %s seg[%d-%d] invalid targetDeviceId=%d (deviceCount=%d)",
             phase, segment.def.startSlot, segment.def.endSlot, targetDevice, cachedDeviceCount);
    return false;
  }

  if (cachedCurrentDevice < 0) {
    int currentDevice = -1;
    cudaError_t getErr = cudaGetDevice(&currentDevice);
    if (getErr != cudaSuccess) {
      DSP_DIAG(BACKEND, "NativeDSP::execute: %s seg[%d-%d] failed to query current CUDA device: %s",
               phase, segment.def.startSlot, segment.def.endSlot, cudaGetErrorString(getErr));
      cudaGetLastError();
      return false;
    }
    cachedCurrentDevice = currentDevice;
  }

  if (cachedCurrentDevice != targetDevice) {
    cudaError_t setErr = cudaSetDevice(targetDevice);
    if (setErr != cudaSuccess) {
      DSP_DIAG(BACKEND, "NativeDSP::execute: %s seg[%d-%d] failed to switch CUDA device %d->%d: %s",
               phase, segment.def.startSlot, segment.def.endSlot,
               cachedCurrentDevice, targetDevice, cudaGetErrorString(setErr));
      cudaGetLastError();
      return false;
    }
    DSP_DIAG(BACKEND, "NativeDSP::execute: %s seg[%d-%d] switched CUDA device %d->%d",
             phase, segment.def.startSlot, segment.def.endSlot, cachedCurrentDevice, targetDevice);
    cachedCurrentDevice = targetDevice;
  } else {
    DSP_DIAG(BACKEND, "NativeDSP::execute: %s seg[%d-%d] using CUDA device %d",
             phase, segment.def.startSlot, segment.def.endSlot, cachedCurrentDevice);
  }
  return true;
}

}  // namespace

// ═══════════════════════════════════════════════════════════════════════════════
// Platform dispatch: Frozen graph fast path
// ═══════════════════════════════════════════════════════════════════════════════

Status NativeDynamicShapePlan::platformTryFrozenFastPath(
    NDArray** externalInputs, int numExternalInputs,
    NDArray** requestedOutputs, int numRequestedOutputs, void* stream) {

  if (ModeContract::forMode(graphExecutionMode_).isSlotBySlot ||
      Environment::getInstance().tritonSkipKernels()) {
    DSP_DIAG(EXECUTE, "FROZEN_FAST_PATH: disabled for slot-by-slot/skip-kernels mode");
    return Status::MAYBE;
  }

  // Ensure VERIFY diagnostics are enabled and at FULL level when tritonVerifyKernels is on.
  // This mirrors the same logic in executeSegmentWithGpuGraph (gpubackend.cpp) but is needed
  // here because the CUDA_GRAPHS mode uses this path directly (no GPU graph backend).
  if (Environment::getInstance().tritonVerifyKernels()) {
    if (!DSP_DIAG_ENABLED(VERIFY)) {
      sd::graph::DspDiagnostics::getInstance().enableCategories(sd::graph::DSP_DIAG_VERIFY);
      sd::graph::DspDiagnostics::getInstance().setLevel(sd::graph::DSP_LEVEL_FULL);
    }
  }

  bool allowFrozenGraphFastPath =
      ModeContract::forMode(graphExecutionMode_).allowsFrozenFastPath;
  bool frozenFastPathInputStable = true;
  bool frozenFastPathSlotsStable = true;
  if (allowFrozenGraphFastPath && !planLifecycle_.isSlotBySlot() && executeCount_ >= 1 && segments_.size() == 1) {
    auto& seg0 = segments_[0];
    // Per-address comparison: catches address changes that the hash may miss
    // (e.g. CUDA pool reuses an address for a different allocation, changing
    // only a subset of the hashed values in a way that produces a collision).
    if (planLifecycle_.pointersStable()) {
      // All pointers (inputs + slots) confirmed stable — skip both hashes
      frozenFastPathInputStable = true;
    } else if (seg0.exec.capturedInputAddrKey != 0) {
      // Prefer the filtered/staged address key when available. Triton decode
      // stabilizes variable placeholder inputs via plan-owned staging buffers;
      // comparing raw external addresses first would reject replay on every
      // step even though the captured graph is bound to the stable staged
      // pointers.
      frozenFastPathInputStable =
          (computeSegmentInputAddrKey(seg0, externalInputs, numExternalInputs) == seg0.exec.capturedInputAddrKey);
    } else if (seg0.exec.replayHandle && !seg0.exec.replayHandle->getCapturedExternalAddresses().empty()) {
      frozenFastPathInputStable = externalAddrsMatch(seg0, externalInputs, numExternalInputs);
    }
    if (planLifecycle_.pointersStable()) {
      frozenFastPathSlotsStable = true;
    } else if (seg0.exec.capturedSlotAddrHash != 0) {
      frozenFastPathSlotsStable =
          (computeSlotAddrHash(outputSlots_, seg0.def.startSlot, seg0.def.endSlot, totalOutputSlots_) ==
           seg0.exec.capturedSlotAddrHash);
    }
  }
  bool fastPathApplicable = allowFrozenGraphFastPath && !planLifecycle_.isSlotBySlot() && executeCount_ >= 1 && segments_.size() == 1 &&
        frozenFastPathInputStable && frozenFastPathSlotsStable &&
        segments_[0].exec.replayHandle != nullptr &&
        segments_[0].exec.replayHandle->isReady();
  bool compositeFastPathApplicable = false;
  if (!fastPathApplicable && allowFrozenGraphFastPath && !planLifecycle_.isSlotBySlot() && executeCount_ >= 1 && segments_.size() == 1 &&
      frozenFastPathInputStable && frozenFastPathSlotsStable &&
      !Environment::getInstance().tritonVerifyKernels()) {
    compositeFastPathApplicable = hasCompositeHandles(segments_[0]);
    // Don't short-circuit via composite fast path while the segment is in the
    // CUDA graph capture window and hasn't been captured yet. The capture
    // decision lives in executeSegmentWithGpuGraph → segDispatchCaptureOrDirect,
    // which is only reachable through the full phaseReplay path. If we return
    // OK here, the capture window passes without capture being attempted.
    if (compositeFastPathApplicable && Environment::getInstance().tritonGraphCapture()) {
      auto& seg0 = segments_[0];
      bool hasCapturedGraph = (seg0.exec.replayHandle != nullptr);
      if (!hasCapturedGraph) {
        int captureMinExec = Environment::getInstance().tritonCaptureMinExec();
        bool inCaptureWindow = (seg0.exec.executionCount >= captureMinExec) &&
                               (seg0.exec.executionCount <= (captureMinExec + 2));
        if (inCaptureWindow) {
          DSP_DIAG(EXECUTE, "FROZEN_FAST_PATH: composite ready but deferring to full path for "
                   "CUDA graph capture (execCount=%d window=[%d,%d])",
                   seg0.exec.executionCount, captureMinExec, captureMinExec + 2);
          compositeFastPathApplicable = false;
        }
      }
    }
  }
  DSP_DIAG(EXECUTE, "FROZEN_FAST_PATH: allow=%d frozen=%d execCount=%d segs=%d inputStable=%d slotStable=%d hasHandle=%d ready=%d compositeReady=%d -> %s",
           (int)allowFrozenGraphFastPath, (int)shapesFrozen_, (int)executeCount_, (int)segments_.size(),
           (int)frozenFastPathInputStable, (int)frozenFastPathSlotsStable,
           (int)(segments_[0].exec.replayHandle != nullptr),
           (int)(segments_[0].exec.replayHandle ? segments_[0].exec.replayHandle->isReady() : 0),
           (int)compositeFastPathApplicable,
           (fastPathApplicable || compositeFastPathApplicable) ? "OK" : "MAYBE");
  if (!fastPathApplicable && !compositeFastPathApplicable) {
    return Status::MAYBE;  // Fast path not applicable
  }

  using Clock = std::chrono::high_resolution_clock;
  auto t0 = executionTimingEnabled_ ? Clock::now() : Clock::time_point{};

  // Stale CUDA error clearing is handled by executeSteadyState (conditional)
  // or execute() before this call. Only clear the sticky CUDA driver error.
  cudaGetLastError();

  GraphSegment& seg = segments_[0];
  if (!bindSegmentCudaDevice(seg, slots_, numSlots_, "frozenFastPath")) {
    return Status::KERNEL_FAILURE;
  }
  cudaStream_t cudaStr = (stream != nullptr)
      ? *static_cast<cudaStream_t*>(stream) : nullptr;

  // Set DSP execution stream BEFORE the ext input sync loop so that
  // syncToSpecial() routes H2D copies to the DSP stream (async, no per-call
  // cudaStreamSynchronize). Without this, each syncToSpecial uses stream 0
  // with a blocking sync per call — N pipeline drains for N variable inputs.
  // The compositeReplay path in gpubackend.cu already does this correctly.
  sd::graph::DspStreamGuard dspStreamGuard(cudaStr);

  if (fastPathApplicable) {
    bool ok = true;

  // Unified pre-replay sync: cross-stream ordering + H2D variable inputs + D2D staging.
  // Replaces the three separate inline sync blocks that were previously here.
  // performPreReplaySync is idempotent (PlanExecutionContext dedup flags).
  externalInputs = performPreReplaySync(externalInputs, numExternalInputs, stream, "frozen_fast_path");

  if (ok) {
    auto tCopyDone = executionTimingEnabled_ ? Clock::now() : Clock::time_point{};

#if HAVE_TRITON
    // Gate arg table refresh on generation counter (replaces argTableStable boolean)
    if (seg.exec.needsArgRefresh()) {
      auto* tritonBackend = dynamic_cast<TritonGraphBackend*>(getGpuGraphBackend());
      if (tritonBackend != nullptr) {
        tritonBackend->refreshArgTablesForReplay(seg, externalInputs, numExternalInputs,
                                                 outputSlots_, totalOutputSlots_,
                                                 stream);
        // Copy consolidated arg table to device after refresh (was missing)
        tritonBackend->copyConsolidatedArgTableToDevice(seg, stream);
      }
      seg.exec.markArgsCurrent();
    } else {
      DSP_DIAG(EXECUTE, "FROZEN_FAST_PATH: args current — skip refresh seg[%d-%d]",
               seg.def.startSlot, seg.def.endSlot);
    }
#endif

    // Zero slot outputs that require it before each graph replay. The prezero
    // was intended to be captured into the CUDA graph during the capture phase,
    // but the per-slot prezeroThisSlot lambda in executeSlot was never invoked
    // (dead code). Without this, ops whose output is not fully overwritten
    // (reductions, scatter, GDN state) read stale data from the previous
    // replay step. The error accumulates through recurrent state (GDN/conv)
    // and eventually causes token divergence vs SLOT_BY_SLOT.
    prezeroSegmentOutputs(seg, stream);

    // Zero cuBLAS workspace before each graph replay. Captured cuBLAS kernels may
    // use workspace as scratch space; stale data from the previous replay step can
    // cause tiny FP differences that accumulate through recurrent state (GDN).
    // TRITON composite runs cuBLAS live (fresh workspace each call) and matches
    // SLOT_BY_SLOT perfectly — this zeroing makes monolithic replay match too.
    if (cublasWorkspaceBuffer_ != nullptr && cublasWorkspaceSize_ > 0) {
      cudaMemsetAsync(cublasWorkspaceBuffer_, 0, cublasWorkspaceSize_, cudaStr);
    }

    if (seg.exec.replayHandle->replay(stream)) {
      auto tLaunchDone = executionTimingEnabled_ ? Clock::now() : Clock::time_point{};
      for (int i = 0; i < numRequestedOutputs_; i++) {
        int slotIdx = requestedOutputSlotIndices_[i];
        if (slotIdx >= 0 && slotIdx < totalOutputSlots_) {
          requestedOutputs[i] = outputSlots_[slotIdx];
        } else {
          requestedOutputs[i] = nullptr;
        }
      }
      totalGraphReplays_++;
      seg.exec.executionCount++;
      executeCount_++;

      // markCaptured() transitions directly to REPLAYING — no separate step needed.

      // Tick actuality: CUDA graph replay writes device memory directly without
      // registerSpecialUse. Without this tick, syncToHost sees stale host data.
      // Skip frozen constant slots — they were excluded from graph capture and
      // their device buffers are already authoritative from warmup execution.
      for (int s = seg.def.startSlot; s <= seg.def.endSlot && s < totalOutputSlots_; s++) {
        if (s < numSlots_ && slots_[s].frozenConstantSlot()) continue;
        NDArray* arr = outputSlots_[s];
        if (arr != nullptr && arr->dataBuffer() != nullptr && !arr->dataBuffer()->isClosed()) {
          arr->tickWriteDevice();
        }
      }

      // No blocking cudaStreamSynchronize here — platformEndExecution provides
      // event-based cross-stream ordering (record on dspStream, wait on default
      // stream) which is non-blocking and sufficient for correctness.

      // Diagnostic: dump argmax from replay output at divergence steps
      if (Environment::getInstance().isDebug() && executeCount_ >= 10 && executeCount_ <= 22) {
        // Sync to get accurate read
        if (cudaStr != nullptr) cudaStreamSynchronize(cudaStr);
        // Find last output slot (logits)
        int lastOutSlot = seg.def.endSlot;
        if (lastOutSlot >= 0 && lastOutSlot < totalOutputSlots_ && outputSlots_[lastOutSlot] != nullptr) {
          NDArray* logitsArr = outputSlots_[lastOutSlot];
          if (logitsArr->lengthOf() > 0 && logitsArr->specialBuffer() != nullptr) {
            int len = static_cast<int>(logitsArr->lengthOf());
            // Find argmax via D2H copy of top values
            int sampleSize = std::min(len, 8);
            if (logitsArr->dataType() == DataType::FLOAT32) {
              std::vector<float> topVals(len);
              cudaMemcpy(topVals.data(), logitsArr->specialBuffer(),
                         len * sizeof(float), cudaMemcpyDeviceToHost);
              int argmax = 0;
              float maxVal = topVals[0];
              for (int v = 1; v < len; v++) {
                if (topVals[v] > maxVal) { maxVal = topVals[v]; argmax = v; }
              }
              sd_printf("REPLAY_DIAG[exec=%d] logits slot=%d len=%d argmax=%d maxVal=%.4f "
                        "first8: %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n",
                        executeCount_, lastOutSlot, len, argmax, maxVal,
                        topVals[0], topVals[std::min(1,len-1)], topVals[std::min(2,len-1)],
                        topVals[std::min(3,len-1)], topVals[std::min(4,len-1)],
                        topVals[std::min(5,len-1)], topVals[std::min(6,len-1)],
                        topVals[std::min(7,len-1)]);
            }
          }
        }
      }

      // ── REPLAY VERIFICATION (CUDA_GRAPHS path) ──────────────────────────
      if (Environment::getInstance().tritonVerifyKernels()) {
        performReplayVerify(seg, externalInputs, numExternalInputs, stream, "CUDA_GRAPHS");
      }

      if (executionTimingEnabled_) {
        auto tDone = Clock::now();
        auto copyUs = std::chrono::duration_cast<std::chrono::microseconds>(tCopyDone - t0).count();
        auto launchUs = std::chrono::duration_cast<std::chrono::microseconds>(tLaunchDone - tCopyDone).count();
        auto scatterUs = std::chrono::duration_cast<std::chrono::microseconds>(tDone - tLaunchDone).count();
        auto totalUs = std::chrono::duration_cast<std::chrono::microseconds>(tDone - t0).count();
        DSP_DIAG(TIMING, "DSP timing: copy=%lldus launch=%lldus scatter=%lldus total=%lldus",
                 copyUs, launchUs, scatterUs, totalUs);
      }
      return Status::OK;
    }
  }

    DSP_DIAG(EXECUTE, "NativeDSP::execute: frozen fast path not applicable (ok=%d), using full path",
             static_cast<int>(ok));
  } else {
    // Composite fast path: bypass phaseReplay/executeSegmentWithGpuGraph overhead
    // and call compositeReplay directly. All H2D sync, cross-stream ordering,
    // arg table refresh, and prezero are handled inside compositeReplay.
    sd::graph::DspStreamGuard dspStreamGuard(cudaStr);

    auto replayStatus = compositeReplay(seg, seg.exec.compositeReplaySchedule,
                                        externalInputs, numExternalInputs, stream);
    if (replayStatus == Status::OK) {
      for (int i = 0; i < numRequestedOutputs_; i++) {
        int slotIdx = requestedOutputSlotIndices_[i];
        if (slotIdx >= 0 && slotIdx < totalOutputSlots_) {
          requestedOutputs[i] = outputSlots_[slotIdx];
        } else {
          requestedOutputs[i] = nullptr;
        }
      }
      totalGraphReplays_++;
      seg.exec.executionCount++;
      executeCount_++;

      // markCaptured() transitions directly to REPLAYING — no separate step needed.

      // compositeReplay ticks slots per-unit inside the replay loop.
      // No need to double-tick here.

      if (executionTimingEnabled_) {
        auto tDone = Clock::now();
        auto totalUs = std::chrono::duration_cast<std::chrono::microseconds>(tDone - t0).count();
        DSP_DIAG(TIMING, "DSP timing: composite_fast_path total=%lldus", totalUs);
      }
      return Status::OK;
    }
    DSP_DIAG(EXECUTE, "NativeDSP::execute: composite fast path failed, using full path");
  }
  return Status::MAYBE;  // Not applicable — use full execution path
}

// ═══════════════════════════════════════════════════════════════════════════════
// Platform dispatch: Pre-execute setup
// ═══════════════════════════════════════════════════════════════════════════════

void NativeDynamicShapePlan::platformPreExecuteSetup(
    NDArray** externalInputs, int numExternalInputs, void* stream) {

  // Clear stale CUDA errors
  cudaGetLastError();

  // Clear attention workspace when no graphs are cached yet.
  {
    bool anyGraphCached = false;
    for (const auto& seg : segments_) {
      if (seg.exec.replayHandle != nullptr && seg.exec.replayHandle->isReady()) { anyGraphCached = true; break; }
    }
    if (!anyGraphCached) {
      AttentionWorkspace::getInstance()->clear();
    }
  }

  // Clear any CUDA errors from workspace clear
  cudaGetLastError();

  // Free captured graphs for segments whose shapes have changed
  if (planLifecycle_.isSlotBySlot() || executeCount_ == 0) {
    for (auto& segment : segments_) {
      if (segment.exec.replayHandle) {
        LongType segShapeKey = computeSegmentShapeKey(segment, externalInputs, numExternalInputs);
        if (segment.exec.cachedShapeKey != segShapeKey) {
          platformCleanupSegmentForRebuild(segment);
        }
      }
    }
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Platform dispatch: Segment cache retention check
// ═══════════════════════════════════════════════════════════════════════════════

bool NativeDynamicShapePlan::platformShouldKeepSegmentCache(const GraphSegment& seg) const {
  // Keep caches for segments with an instantiated graph that can replay.
  // compilationFailed now throws immediately (no silent fallback), so a segment
  // with compilationFailed=true will never reach cache retention checks.
  if (seg.exec.replayHandle != nullptr) return true;
  return false;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Platform dispatch: Parallel precompilation
// ═══════════════════════════════════════════════════════════════════════════════

void NativeDynamicShapePlan::platformPrecompileSegments(
    NDArray** externalInputs, int numExternalInputs) {
  DSP_REQUIRE_PLAN_PHASE_AT_MOST(PlanPhase::SHAPES_FROZEN, "platformPrecompileSegments");
  using Clock = std::chrono::high_resolution_clock;

  // Guard: require at least one warmup execution (executeCount_ >= 1) so that
  // slot shape caches are populated before Triton IR build tries to read them.
  // Without this, cross-segment inputs have empty shapes → all IR builds fail.
  if (planLifecycle_.compilationDone || executeCount_ < 1 ||
      ModeContract::forMode(graphExecutionMode_).isSlotBySlot ||
      Environment::getInstance().tritonSkipKernels()) {
    DSP_DIAG(COMPILE, "platformPrecompileSegments: skipped (compilationDone=%d execCount=%d mode=%d)",
             planLifecycle_.compilationDone ? 1 : 0, executeCount_, static_cast<int>(graphExecutionMode_));
    return;
  }

  auto* gpuBackend = getGpuGraphBackend();
  if (gpuBackend == nullptr) {
    DSP_DIAG(COMPILE, "platformPrecompileSegments: no GPU backend available");
    return;
  }

  struct PrecompileTask {
    int segIdx;
    LongType shapeKey;
    int targetDevice;
  };
  std::vector<PrecompileTask> tasks;
  for (int si = 0; si < static_cast<int>(segments_.size()); si++) {
    auto& seg = segments_[si];
    if (seg.exec.compilationFailed) continue;
    // Skip data-dependent segments — they require host sync during execution
    // which is incompatible with CUDA graph capture.
    bool hasDataDep = false;
    for (int s = seg.def.startSlot; s <= seg.def.endSlot; s++) {
      if (slots_[s].flags.isDataDependent) { hasDataDep = true; break; }
    }
    if (hasDataDep) continue;
    bool tryCapture = seg.def.isCapturable || (!planLifecycle_.isSlotBySlot() && executeCount_ > 0);
    if (!tryCapture) continue;
    if (!gpuBackend->canFuseSegment(slots_, seg.def.startSlot, seg.def.endSlot)) continue;
    LongType segShapeKey = computeSegmentShapeKey(seg, externalInputs, numExternalInputs);
    int currentDev = 0;
    cudaGetDevice(&currentDev);
    int segTargetDevice = currentDev;
    if (seg.def.startSlot >= 0 && seg.def.startSlot < numSlots_) {
      segTargetDevice = slots_[seg.def.startSlot].targetDeviceId;
      if (segTargetDevice < 0) segTargetDevice = currentDev;
    }
    tasks.push_back({si, segShapeKey, segTargetDevice});
  }

  if (tasks.empty()) return;

  // Determine thread count for parallel precompilation.
  // Inner sub-segment parallelism is handled by compileSegment (DEFAULT_MAX_PARALLEL_COMPILATIONS).
  // Outer segment-level parallelism is safe because:
  // - Each compilation creates its own MLIRContext (via getMlirContextMutex-protected factory)
  // - cuModuleLoadDataEx is serialized via loadModuleMtx
  // - LLVM init is done via std::once_flag
  //
  // NOTE: The previous `tasks.size() <= 1` guard skipped compilation entirely
  // for single-segment plans (common at decode: one mega-segment after
  // freeze-merge), causing Triton islands to only be compiled lazily during
  // execution rather than eagerly at seal time.  Now we handle all task counts.
  int numThreads = std::min(8, static_cast<int>(tasks.size()));
  DSP_DIAG(COMPILE, "NativeDSP::execute: parallel precompilation of %d segments using %d threads "
           "(executeCount=%d)",
           static_cast<int>(tasks.size()), numThreads, executeCount_);
  auto precompileStart = Clock::now();

  // Force-initialize static singleton tables on the main thread BEFORE
  // any worker threads start.
  (void)sd::graph::getOpCategoryTable();

  std::atomic<int> precompileOk{0};
  std::atomic<int> precompileFail{0};
  std::atomic<size_t> nextTask{0};

  auto workerFn = [&]() {
    while (true) {
      size_t i = nextTask.fetch_add(1);
      if (i >= tasks.size()) break;

      const auto& task = tasks[i];
      cudaError_t setDevErr = cudaSetDevice(task.targetDevice);
      if (setDevErr != cudaSuccess) {
        DSP_DIAG(COMPILE, "NativeDSP::precompile: cudaSetDevice(%d) failed for segment %d: %s",
                 task.targetDevice, task.segIdx, cudaGetErrorString(setDevErr));
        cudaGetLastError();
        precompileFail++;
        continue;
      }
      auto& seg = segments_[task.segIdx];
      bool ok = gpuBackend->compileSegment(seg, slots_, externalInputs, numExternalInputs,
                                            outputSlots_, totalOutputSlots_, task.shapeKey,
                                            numSlots_);
      if (ok) {
        segments_[task.segIdx].def.shapeKeyState.markCompiled(task.shapeKey);
        precompileOk++;
      } else {
        precompileFail++;
      }
    }
  };

  // Launch worker threads
  std::vector<std::thread> workers;
  for (int t = 0; t < numThreads; t++) {
    workers.emplace_back(workerFn);
  }
  for (auto& w : workers) {
    w.join();
  }

  auto precompileMs = std::chrono::duration_cast<std::chrono::milliseconds>(
      Clock::now() - precompileStart).count();
  DSP_DIAG(COMPILE, "NativeDSP::execute: parallel precompilation done in %lld ms "
           "(ok=%d, failed=%d)",
           static_cast<long long>(precompileMs), precompileOk.load(), precompileFail.load());

#if HAVE_TRITON
  // Batched module preload (task #4): walk the cache once and make sure every
  // CompiledKernel has a live CUmodule loaded into GPU memory.  This avoids
  // paying lazy-load latency on the first replay of each segment and gives us
  // a single checkpoint where the projected per-device residency is compared
  // against env.triton().moduleResidencyBudgetBytes().  Preload happens on
  // every device that any task targeted so cross-device caches are warmed up
  // before execution begins.
  {
    auto* tritonBackend = dynamic_cast<TritonGraphBackend*>(gpuBackend);
    if (tritonBackend != nullptr) {
      std::unordered_set<int> devicesToPreload;
      for (const auto& task : tasks) {
        devicesToPreload.insert(task.targetDevice);
      }
      int prevDev = 0;
      cudaGetDevice(&prevDev);
      for (int d : devicesToPreload) {
        if (d < 0) continue;
        cudaError_t setDevErr = cudaSetDevice(d);
        if (setDevErr != cudaSuccess) {
          cudaSetDevice(prevDev);
          DSP_THROW_CUDA(COMPILE, setDevErr,
                         "NativeDSP::precompile: cudaSetDevice(%d) failed before preloadAllModules",
                         d);
        }
        // Trim the memory pool before loading Triton modules to reclaim cached
        // buffers. Module loading allocates GPU memory for cubin modules, and on
        // memory-constrained GPUs this can fail if the pool holds reclaimable memory.
        memory::CudaMemoryPool::getInstance().trimPool(d);
        Status preloadStatus = tritonBackend->preloadAllModules(d);
        if (preloadStatus != Status::OK) {
          cudaSetDevice(prevDev);
          DSP_THROW(COMPILE, "NativeDSP::precompile: preloadAllModules(device=%d) failed", d);
        }
      }
      cudaSetDevice(prevDev);
    }
  }

  // Report per-device Triton module memory budget
  {
    auto* tritonBackend = dynamic_cast<TritonGraphBackend*>(gpuBackend);
    if (tritonBackend != nullptr) {
      int numDevices = 0;
      cudaGetDeviceCount(&numDevices);
      for (int d = 0; d < std::min(numDevices, TritonGraphBackend::kMaxTritonDevices); d++) {
        size_t tritonMem = tritonBackend->getTritonModuleMemory(d);
        if (tritonMem == 0) continue;
        size_t gpuFree = 0, gpuTotal = 0;
        int prevDev; cudaGetDevice(&prevDev);
        cudaSetDevice(d);
        cudaMemGetInfo(&gpuFree, &gpuTotal);
        cudaSetDevice(prevDev);
        DSP_DIAG(MEMORY, "TRITON_BUDGET device=%d: modules=%zuMB gpuFree=%zuMB gpuTotal=%zuMB",
                 d, tritonMem / (1024 * 1024), gpuFree / (1024 * 1024), gpuTotal / (1024 * 1024));
      }
    }
  }
#endif
}

// ═══════════════════════════════════════════════════════════════════════════════
// Platform dispatch: Segment device binding
// ═══════════════════════════════════════════════════════════════════════════════

bool NativeDynamicShapePlan::platformBindSegmentDevice(const GraphSegment& segment) {
  return bindSegmentCudaDevice(segment, slots_, numSlots_, "segmentExec");
}

// ═══════════════════════════════════════════════════════════════════════════════
// Platform dispatch: Cross-device input migration
// ═══════════════════════════════════════════════════════════════════════════════

void NativeDynamicShapePlan::platformMigrateSegmentInputs(
    const GraphSegment& seg, NDArray** externalInputs, int numExternalInputs) {
  // Get target device for this segment
  int targetDevice = -1;
  if (seg.def.startSlot >= 0 && seg.def.startSlot < numSlots_) {
    targetDevice = slots_[seg.def.startSlot].targetDeviceId;
  }
  if (targetDevice < 0) return;  // Auto device — no migration needed

  migratedInputs_.clear();

  // Collect unique input slot indices that this segment reads from prior segments
  std::unordered_set<int> neededInputSlots;
  for (int s = seg.def.startSlot; s <= seg.def.endSlot && s < numSlots_; s++) {
    const NativeSlot& slot = slots_[s];
    for (int i = 0; i < slot.wiring.numInputs; i++) {
      int srcIdx = slot.wiring.inputSourceIndices[i];
      if (srcIdx >= 0 && srcIdx < totalOutputSlots_) {
        // This is an internal input from a prior slot's output
        // Only migrate if the source slot is on a different device
        if (outputSlots_[srcIdx] != nullptr) {
          neededInputSlots.insert(srcIdx);
        }
      }
      // External inputs (srcIdx < 0) are handled by the caller
    }
  }

  int migrated = 0;
  for (int slotIdx : neededInputSlots) {
    NDArray* arr = outputSlots_[slotIdx];
    if (arr == nullptr || arr->isEmpty()) continue;

    // Check if this array's GPU data is on a different device
    auto* db = arr->dataBuffer();
    if (db == nullptr) continue;

    // The array may be on a different device. We check by trying to determine
    // where the special (GPU) buffer lives. If targetDevice differs from where
    // the data was produced, we need to migrate.
    // Find which device produced this output by checking the source slot's targetDeviceId
    int sourceDevice = -1;
    // Walk backwards to find which slot produced this output
    for (int s = 0; s < numSlots_; s++) {
      const NativeSlot& srcSlot = slots_[s];
      for (int o = 0; o < srcSlot.wiring.numOutputs; o++) {
        if (srcSlot.wiring.outputSlotIndices[o] == slotIdx) {
          sourceDevice = srcSlot.targetDeviceId;
          break;
        }
      }
      if (sourceDevice >= 0) break;
    }

    if (sourceDevice < 0) {
      // External or auto — use the current active device, not hardcoded 0
      int activeDev = 0;
      cudaGetDevice(&activeDev);
      sourceDevice = activeDev;
    }
    if (sourceDevice == targetDevice) continue;  // Same device, no migration needed

    // Migrate: sync to host on source device, create copy on target device
    // Save current device, switch to source to sync
    int savedDevice = -1;
    cudaGetDevice(&savedDevice);

    // Step 1: Ensure data is on host (sync from source device)
    cudaSetDevice(sourceDevice);
    arr->syncToHost();

    // Step 2: Switch to target device and create a copy
    cudaSetDevice(targetDevice);

    // Create new array on target device with same shape and data type
    std::vector<LongType> shapeVec(*arr->getShapeAsVector());
    auto* copy = new NDArray(arr->ordering(), shapeVec, arr->dataType(),
                             LaunchContext::defaultContext());

    // Copy host data to the new array's host buffer, then sync to target device
    auto srcLen = arr->lengthOf() * DataTypeUtils::sizeOf(arr->dataType());
    if (srcLen > 0 && arr->buffer() != nullptr && copy->buffer() != nullptr) {
      std::memcpy(copy->buffer(), arr->buffer(), srcLen);
    }
    copy->tickWriteHost();
    copy->syncToDevice();

    // Restore to target device (should already be there)
    if (savedDevice != targetDevice) {
      cudaSetDevice(targetDevice);
    }

    // Record migration and replace in outputSlots_
    MigratedInput mi;
    mi.outputSlotIdx = slotIdx;
    mi.original = arr;
    mi.migrated = copy;
    migratedInputs_.push_back(mi);

    outputSlots_[slotIdx] = copy;
    migrated++;
  }

  if (migrated > 0) {
    DSP_DIAG(EXECUTE, "NativeDSP::execute: migrated %d input arrays from device(s) to device %d "
             "for seg[%d-%d] (host-staged D→H→D)",
             migrated, targetDevice, seg.def.startSlot, seg.def.endSlot);
  }
}

void NativeDynamicShapePlan::platformCleanupMigratedInputs() {
  if (migratedInputs_.empty()) return;
  // Restore original arrays in outputSlots_ and delete migrated copies
  for (auto& mi : migratedInputs_) {
    if (outputSlots_ != nullptr && mi.outputSlotIdx >= 0 && mi.outputSlotIdx < totalOutputSlots_) {
      outputSlots_[mi.outputSlotIdx] = mi.original;
    }
    if (mi.migrated != nullptr) {
      delete mi.migrated;
      mi.migrated = nullptr;
    }
  }
  migratedInputs_.clear();
}

// ═══════════════════════════════════════════════════════════════════════════════
// Platform dispatch: Graph eligibility check
// ═══════════════════════════════════════════════════════════════════════════════

bool NativeDynamicShapePlan::platformShouldUseGraph(const GraphSegment& segment) {
  if (ModeContract::forMode(graphExecutionMode_).isSlotBySlot) {
    DSP_DIAG_SEG(EXECUTE, segment.def.startSlot, "platformShouldUseGraph: false (slot-by-slot mode)");
    return false;
  }

  // When tritonSkipKernels=true, route directly to slot-by-slot execution.
  // Going through executeSegmentWithGpuGraph would demote FROZEN→SHAPE_CACHED
  // (breaking the prezero isFullyWriting guard) and fragment execution into
  // per-subkernel ranges (double-zero risk, lost executionCount increments).
  if (Environment::getInstance().tritonSkipKernels()) {
    DSP_DIAG_SEG(EXECUTE, segment.def.startSlot, "platformShouldUseGraph: false (tritonSkipKernels=true, routing to slot-by-slot)");
    return false;
  }

  // Non-frozen execution: use slot-by-slot to avoid memory leaks.
  // Graph capture with tl_graphExecutionActive=true suppresses cudaFreeAsync,
  // leaking temporary NDArrays. When shapes change each step, new captures
  // compound the leak. Slot-by-slot execution properly frees all temporaries.
  if (planLifecycle_.isSlotBySlot()) return false;

  bool tryCapture = (segment.def.isCapturable || (!planLifecycle_.isSlotBySlot() && executeCount_ > 0))
                    && !segment.exec.compilationFailed;
  // Use selectedBackend to determine if graph capture is possible — no cascade check needed.
  bool hasGraphBackend = (segment.def.selectedBackend == SelectedBackend::GPU_COMPILER ||
                          segment.def.selectedBackend == SelectedBackend::CUDA_GRAPHS);
  bool result = tryCapture && hasGraphBackend;
  if (!result) {
    DSP_DIAG_SEG(EXECUTE, segment.def.startSlot,
                 "platformShouldUseGraph: false (frozen=%d execCount=%d capturable=%d "
                 "compilFailed=%d backend=%d tryCapture=%d hasBackend=%d)",
                 shapesFrozen_ ? 1 : 0, executeCount_,
                 segment.def.isCapturable ? 1 : 0,
                 segment.exec.compilationFailed ? 1 : 0,
                 static_cast<int>(segment.def.selectedBackend),
                 tryCapture ? 1 : 0, hasGraphBackend ? 1 : 0);
  }
  return result;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Platform dispatch: Switch-based backend dispatch (hard error on failure)
// ═══════════════════════════════════════════════════════════════════════════════

Status NativeDynamicShapePlan::platformExecuteSegmentWithBackends(
    GraphSegment& segment, NDArray** externalInputs, int numExternalInputs,
    void* stream, bool& usedGraph) {
  usedGraph = false;

  DSP_DIAG(EXECUTE, "NativeDSP::execute: seg[%d-%d] selectedBackend=%d isCapturable=%d executionCount=%d phase=%s",
           segment.def.startSlot, segment.def.endSlot,
           static_cast<int>(segment.def.selectedBackend), static_cast<int>(segment.def.isCapturable),
           segment.exec.executionCount, segment.exec.displayPhaseName());

  switch (segment.def.selectedBackend) {
    case SelectedBackend::GPU_COMPILER: {
      auto* gpuBackend = getGpuGraphBackend();
      if (!gpuBackend) {
        // GPU_COMPILER was selected but no backend is available at runtime.
        // This is a configuration error — throw rather than silently degrading.
        DSP_THROW_SEG(COMPILE, segment.def.startSlot,
                      "NativeDSP::execute: seg[%d-%d] selectedBackend=GPU_COMPILER but "
                      "getGpuGraphBackend() returned null. No GPU backend available.",
                      segment.def.startSlot, segment.def.endSlot);
      }

      // Compilation has permanently failed for this segment. This means a prior
      // attempt to compile or capture failed and was recorded. Throw immediately
      // so the failure is visible rather than silently producing wrong results.
      if (segment.exec.compilationFailed) {
        DSP_THROW_SEG(COMPILE, segment.def.startSlot,
                      "NativeDSP::execute: exec%d seg[%d-%d] gpuBackend=%s "
                      "compilationFailed=true — prior compilation/capture failed permanently. "
                      "Fix the root cause.",
                      executeCount_, segment.def.startSlot, segment.def.endSlot, gpuBackend->name());
      }

      // ── Fusibility pre-check ──────────────────────────────────────────────
      // canFuseSegment() returns false when the segment contains ops that cannot
      // be expressed in the JIT backend's 1D element-wise kernel model (e.g.
      // reshape, permute, gather, concat — any op not in isNvrtcJittable).
      // Such segments must fall back to slot-by-slot execution; this is NOT a
      // compilation failure and must NOT set compilationFailed. If it did, the
      // segment would be permanently dead and throw on every subsequent call.
      //
      // This check is placed here (not inside executeSegmentWithGpuGraph) so
      // that the "not fusible" case does NOT propagate through the generic
      // KERNEL_FAILURE path below (which marks compilationFailed and throws).
      if (!gpuBackend->canFuseSegment(slots_, segment.def.startSlot, segment.def.endSlot)) {
        DSP_DIAG(BACKEND,
                 "platformExecuteSegmentWithBackends: backend=%s cannot fuse seg[%d-%d] "
                 "(segment contains non-JIT-compatible ops — falling back to slot-by-slot)",
                 gpuBackend->name(), segment.def.startSlot, segment.def.endSlot);
        DSP_SET_SEG_PHASE(segment, ExecutionPhase::SLOT_BY_SLOT, "gpu_compiler_not_fusible_fallback");
        return executeSegmentSlotBySlot(segment, externalInputs, numExternalInputs, stream);
      }

      auto status = executeSegmentWithGpuGraph(segment, externalInputs, numExternalInputs, stream);
      if (status == Status::OK) {
        usedGraph = true;
        if (segment.exec.executionCount <= 1) {
          DSP_SET_SEG_PHASE(segment, ExecutionPhase::COMPILING, "gpu_graph_first_exec");
        } else {
          // Check merged + composite replay handles — Triton island+gap segments use
          // these instead of a monolithic replayHandle. The sentinel replayHandle
          // created during composite capture is NOT a captured graph (isReady()=false).
          bool hasComposite = false;
          // Check merged handles first (island-merged capture groups)
          for (auto& h : segment.exec.compositeReplaySchedule.mergedReplayHandles) {
            if (h != nullptr && h->isReady()) {
              hasComposite = true;
              break;
            }
          }
          // Fallback: check individual composite handles
          if (!hasComposite) {
            for (auto& u : segment.exec.compositeReplaySchedule.units) {
              if (u.kind == REPLAY_UNIT_TRITON_ISLAND && u.mergedGroupId < 0) {
                int idx = u.islandIndex;
                if (idx >= 0 && idx < static_cast<int>(segment.exec.compositeReplaySchedule.compositeReplayHandles.size()) &&
                    segment.exec.compositeReplaySchedule.compositeReplayHandles[idx] != nullptr &&
                    segment.exec.compositeReplaySchedule.compositeReplayHandles[idx]->isReady()) {
                  hasComposite = true;
                  break;
                }
              }
            }
          }
          if (hasComposite) {
            DSP_SET_SEG_PHASE(segment, ExecutionPhase::REPLAYING, "gpu_graph_composite_replay_ready");
          } else if (segment.exec.replayHandle && segment.exec.replayHandle->isReady()) {
            DSP_SET_SEG_PHASE(segment, ExecutionPhase::REPLAYING, "gpu_graph_replay_ready");
          } else {
            DSP_SET_SEG_PHASE(segment, ExecutionPhase::COMPILED, "gpu_graph_compiled_no_replay");
          }
        }
        return Status::OK;
      }

      // GPU backend execution failed — propagate the error. Do NOT fall back to
      // slot-by-slot; fix the root cause (e.g., ensure model is closed+reloaded
      // between configs to free GPU memory before capture).
      segment.exec.compilationFailed = true;
      DSP_THROW_SEG(COMPILE, segment.def.startSlot,
                    "NativeDSP::execute: exec%d seg[%d-%d] gpuBackend=%s FAILED status=%d. "
                    "GPU compilation/capture failed — fix the root cause.",
                    executeCount_, segment.def.startSlot, segment.def.endSlot, gpuBackend->name(),
                    static_cast<int>(status));
    }

    case SelectedBackend::CUDA_GRAPHS: {
      auto status = executeSegmentWithGraph(segment, externalInputs, numExternalInputs, stream);
      if (status != Status::OK) {
        // Graph capture failed — propagate the error. Do NOT fall back to
        // slot-by-slot; fix the root cause (memory management during capture).
        segment.exec.compilationFailed = true;
        DSP_THROW_SEG(COMPILE, segment.def.startSlot,
                      "NativeDSP::execute: CUDA graph capture failed for seg[%d-%d] status=%d. "
                      "Fix capture memory management — do NOT fall back to slot-by-slot.",
                      segment.def.startSlot, segment.def.endSlot, static_cast<int>(status));
      }
      usedGraph = (segment.exec.replayHandle != nullptr && segment.exec.replayHandle->isReady() && !segment.exec.compilationFailed);
      if (usedGraph) {
        DSP_SET_SEG_PHASE(segment, ExecutionPhase::REPLAYING, "cuda_graph_replay_ready");
      } else if (segment.exec.executionCount <= 1) {
        DSP_SET_SEG_PHASE(segment, ExecutionPhase::COMPILING, "cuda_graph_first_exec");
      } else {
        DSP_SET_SEG_PHASE(segment, ExecutionPhase::COMPILED, "cuda_graph_compiled_no_replay");
      }
      return Status::OK;
    }

    case SelectedBackend::SLOT_BY_SLOT:
      DSP_SET_SEG_PHASE(segment, ExecutionPhase::SLOT_BY_SLOT, "backend_slot_by_slot");
      return executeSegmentSlotBySlot(segment, externalInputs, numExternalInputs, stream);

    case SelectedBackend::CPU_GRAPH:
      // CPU graph backend not applicable on CUDA build — treat as slot-by-slot
      DSP_SET_SEG_PHASE(segment, ExecutionPhase::SLOT_BY_SLOT, "cpu_graph_on_cuda_build");
      return executeSegmentSlotBySlot(segment, externalInputs, numExternalInputs, stream);

    default:
      DSP_THROW_SEG(EXECUTE, segment.def.startSlot,
                    "NativeDSP::execute: seg[%d-%d] unknown selectedBackend=%d",
                    segment.def.startSlot, segment.def.endSlot,
                    static_cast<int>(segment.def.selectedBackend));
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Platform dispatch: Post-segment error check
// ═══════════════════════════════════════════════════════════════════════════════

Status NativeDynamicShapePlan::platformCheckPostSegment(GraphSegment& segment) {
  auto lastErr = cudaGetLastError();
  if (lastErr != cudaSuccess) {
    DSP_THROW_CUDA(EXECUTE, lastErr,
                   "CUDA error after segment [%d-%d] (execCount=%d shapesFrozen=%d): %d",
                   segment.def.startSlot, segment.def.endSlot,
                   executeCount_, static_cast<int>(shapesFrozen_),
                   static_cast<int>(lastErr));
  }
  return Status::OK;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Platform dispatch: Segment cleanup for rebuild
// ═══════════════════════════════════════════════════════════════════════════════

void NativeDynamicShapePlan::platformCleanupSegmentForRebuild(GraphSegment& seg) {
  DSP_DIAG_SEG(GRAPH_REPLAY, seg.def.startSlot,
               "platformCleanupSegmentForRebuild: seg[%d-%d] hasReplay=%d compositeHandles=%d",
               seg.def.startSlot, seg.def.endSlot, seg.exec.replayHandle ? 1 : 0,
               static_cast<int>(seg.exec.compositeReplaySchedule.compositeReplayHandles.size()));
  // Clear monolithic replay handle
  if (seg.exec.replayHandle) {
    if (seg.exec.replayHandle->getWorkspacePtr() != nullptr) {
      seg.exec.replayHandle->releaseWorkspace(nullptr, seg.def.startSlot);
    }
    seg.exec.replayHandle->freeHostPointers();
    seg.exec.replayHandle->clearExternalAddresses();
    seg.exec.replayHandle.reset();
  }
  // Clear merged replay handles (island-merged capture groups)
  for (auto& h : seg.exec.compositeReplaySchedule.mergedReplayHandles) {
    if (h) {
      if (h->getWorkspacePtr() != nullptr) {
        h->releaseWorkspace(nullptr, seg.def.startSlot);
      }
      h->freeHostPointers();
      h->clearExternalAddresses();
      h.reset();
    }
  }
  seg.exec.compositeReplaySchedule.mergedReplayHandles.clear();
  // Clear merged group tags on schedule units
  for (auto& u : seg.exec.compositeReplaySchedule.units) {
    u.mergedGroupId = -1;
    u.isMergedLeader = false;
  }
  // Clear composite (per-island) replay handles
  for (auto& h : seg.exec.compositeReplaySchedule.compositeReplayHandles) {
    if (h) {
      if (h->getWorkspacePtr() != nullptr) {
        h->releaseWorkspace(nullptr, seg.def.startSlot);
      }
      h->freeHostPointers();
      h->clearExternalAddresses();
      h.reset();
    }
  }
  seg.exec.gapOpsCapturedInGraph = false;
  seg.exec.bumpArgGeneration();
  seg.exec.argTableStable = false;  // Invalidate fast-replay when handles are cleared
  seg.exec.addrKeyStableCount = 0;
  seg.exec.slotAddrStableCount = 0;
  seg.resolvedCpuBackend = nullptr;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Platform dispatch: Plan resource cleanup (destructor)
// ═══════════════════════════════════════════════════════════════════════════════

void NativeDynamicShapePlan::platformFreePlanResources() {
  DSP_DIAG(MEMORY, "platformFreePlanResources: segments=%d slots=%d outputs=%d",
           (int)segments_.size(), numSlots_, totalOutputSlots_);

  // Clear AttentionWorkspace — holds named GPU buffers (attention scratch, softmax
  // intermediate) that persist across plan lifetimes. Without this, the next plan's
  // CUDA graph capture records addresses of the old workspace buffers.
  AttentionWorkspace::getInstance()->clear();

  // Free CUDA event used for cross-stream sync
  if (executionCompleteEvent_ != nullptr) {
    cudaEvent_t evt = *static_cast<cudaEvent_t*>(executionCompleteEvent_);
    cudaEventDestroy(evt);
    delete static_cast<cudaEvent_t*>(executionCompleteEvent_);
    executionCompleteEvent_ = nullptr;
  }

  // Free cached steady-state execution context
  if (steadyStateCrossStreamEvent_ != nullptr) {
    cudaEvent_t evt = *static_cast<cudaEvent_t*>(steadyStateCrossStreamEvent_);
    cudaEventDestroy(evt);
    delete static_cast<cudaEvent_t*>(steadyStateCrossStreamEvent_);
    steadyStateCrossStreamEvent_ = nullptr;
  }
  if (steadyStateExecCtx_ != nullptr) {
    delete static_cast<PlanExecutionContext*>(steadyStateExecCtx_);
    steadyStateExecCtx_ = nullptr;
  }

  // Always invalidate Triton singleton cache entries for this plan's segments.
  // Each compiled CUmodule, arg table device buffer, sync counter, and global
  // scratch allocation stays in the singleton cache across plan lifetimes.
  // Without cleanup, sequential plan creation/destruction (e.g. test matrix
  // running 6 configs) leaks ~GB of GPU memory per plan since the cache entries
  // from destroyed plans are never freed.  The disk cache retains compiled PTX
  // so reloading after eviction is fast (no recompilation needed).
#if HAVE_TRITON
  {
    std::vector<std::pair<int,int>> segRanges;
    segRanges.reserve(segments_.size());
    for (auto& seg : segments_) {
      segRanges.emplace_back(seg.def.startSlot, seg.def.endSlot);
    }
    if (!segRanges.empty()) {
      TritonGraphBackend::getInstance().invalidateCacheForSegments(segRanges);
    }
  }
#endif

  // Free replay workspaces and JIT kernels from all segments.
  // Must explicitly clean up monolithic, merged, AND composite replay handles
  // with proper pool deregistration (releaseWorkspace) before RAII destruction.
  for (auto& seg : segments_) {
    if (seg.exec.replayHandle) {
      if (seg.exec.replayHandle->getWorkspacePtr() != nullptr) {
        seg.exec.replayHandle->releaseWorkspace(nullptr, seg.def.startSlot);
      }
      seg.exec.replayHandle->freeHostPointers();
      seg.exec.replayHandle->clearExternalAddresses();
      seg.exec.replayHandle.reset();
    }
    // Clean up merged replay handles (island-merged capture groups)
    for (auto& h : seg.exec.compositeReplaySchedule.mergedReplayHandles) {
      if (h) {
        if (h->getWorkspacePtr() != nullptr) {
          h->releaseWorkspace(nullptr, seg.def.startSlot);
        }
        h->freeHostPointers();
        h->clearExternalAddresses();
        h.reset();
      }
    }
    seg.exec.compositeReplaySchedule.mergedReplayHandles.clear();
    // Clean up composite (per-island) replay handles
    for (auto& h : seg.exec.compositeReplaySchedule.compositeReplayHandles) {
      if (h) {
        if (h->getWorkspacePtr() != nullptr) {
          h->releaseWorkspace(nullptr, seg.def.startSlot);
        }
        h->freeHostPointers();
        h->clearExternalAddresses();
        h.reset();
      }
    }
    seg.exec.bumpArgGeneration();
    seg.exec.argTableStable = false;  // Invalidate fast-replay on plan teardown
    seg.exec.addrKeyStableCount = 0;
    seg.exec.slotAddrStableCount = 0;
    seg.exec.gapOpsCapturedInGraph = false;
    seg.resolvedCpuBackend = nullptr;
    delete seg.exec.jitKernel;
    seg.exec.jitKernel = nullptr;
  }

  // Free shared capture workspace (allocated once, shared across all segments)
  if (sharedCaptureWorkspace_ != nullptr) {
    memory::CudaMemoryPool::getInstance().unregisterCaptureWorkspace(sharedCaptureWorkspace_);
    cudaFree(sharedCaptureWorkspace_);
    DSP_DIAG(MEMORY, "platformFreePlanResources: freed SHARED capture workspace %zuMB on device %d",
             sharedCaptureWorkspaceBytes_ / (1024*1024), sharedCaptureWorkspaceDevice_);
    sharedCaptureWorkspace_ = nullptr;
    sharedCaptureWorkspaceBytes_ = 0;
    sharedCaptureWorkspaceDevice_ = -1;
  }
  // Free pre-allocated cuBLAS workspace
  if (cublasWorkspaceBuffer_ != nullptr) {
    cudaFree(cublasWorkspaceBuffer_);
    cublasWorkspaceBuffer_ = nullptr;
    cublasWorkspaceSize_ = 0;
  }
  // Reset thread-local cuBLAS workspace pointer — it may still reference the
  // just-freed cublasWorkspaceBuffer_. Without this, MmulHelper::reapplyCublasWorkspace()
  // would set the freed pointer on the singleton cuBLAS handle for the next plan's GEMM ops.
  tl_cublasWorkspacePtr = nullptr;
  tl_cublasWorkspaceSize = 0;

  // Clear thread-local cast cache in MmulHelper — the cached NDArray* pointers
  // reference arrays owned by this plan's model. After plan destruction, those
  // arrays are freed. If another plan (e.g. next config in a sequential test run)
  // reuses CUDA graph capture on the same thread, the stale cast cache entries
  // cause GEMM to read from freed/corrupted memory, producing wrong output.
  MmulHelper::clearCastCache();

  // ── Reset ALL DSP thread-local state to prevent cross-plan contamination ──
  // When sequential configs run on the same thread (e.g. test matrix),
  // stale TLS from the previous plan can corrupt the next plan's execution:
  // - tl_graphExecutionActive stuck true → DataBuffer skips host sync
  // - tl_mergedCaptureActive stuck true → gap ops execute in wrong mode
  // - tl_graphCaptureStream stale → capture records on wrong stream
  // - tl_captureWorkspace stale → allocations use freed workspace
  // - tl_dspExecutionStream/tl_dspGapStream stale → ops route to dead streams
  // - tl_islandSlotMin/Max stale → wrong slot range for island bounds
  tl_graphExecutionActive = false;
  tl_dspReplayActive = false;
  tl_graphCaptureStream = nullptr;
  tl_captureWorkspace = nullptr;
  tl_captureWorkspaceSize = 0;
  tl_captureWorkspaceOffset = 0;
  tl_dspExecutionStream = nullptr;
  tl_dspGapStream = nullptr;
  tl_islandSlotMin = INT_MAX;
  tl_islandSlotMax = INT_MIN;
  // Reset file-static merged capture TLS in gpubackend.cu
  resetMergedCaptureTLS();

  // Clear the CudaGraphScheduler graph cache — cached CudaGraphHandle objects
  // contain baked-in device addresses from this plan's allocations. If another plan
  // is created and the pool recycles those addresses, stale cached graphs would
  // replay against wrong buffers causing accuracy regression.
  sd::cuda::clearCudaGraphSchedulerCache();

  // Free batch D2D resources
  freeBatchD2DResources();

  // Free batched GEMM resources
  freeBatchedGemmResources();
}

// ═══════════════════════════════════════════════════════════════════════════════
// Platform dispatch: Statistics
// ═══════════════════════════════════════════════════════════════════════════════

int NativeDynamicShapePlan::platformCountCapturedGraphSegments() const {
  int count = 0;
  for (const auto& seg : segments_) {
    // Check monolithic replay handle (raw CUDA graph capture)
    if (seg.exec.replayHandle && seg.exec.replayHandle->isReady()) {
      count++;
      continue;
    }
#if HAVE_TRITON
    // Check composite replay handles (per-island Triton capture)
    if (hasCompositeHandles(seg)) {
      count++;
    }
#endif
  }
  return count;
}

void NativeDynamicShapePlan::platformMaybeSplitIfEnabled() {
  // Adaptive splitting removed — segments with shape instability simply recompile
  // via the shape key cache. No physical splitting needed.
}

// ═══════════════════════════════════════════════════════════════════════════════
// CUDA Graph capture audit and validation
// ═══════════════════════════════════════════════════════════════════════════════

std::vector<cuda::CaptureAuditEntry> NativeDynamicShapePlan::getHostOnlyOps() const {
  std::vector<cuda::CaptureAuditEntry> result;
  for (const auto& entry : lastCaptureAudit_) {
    if (entry.isHostOnly()) {
      result.push_back(entry);
    }
  }
  return result;
}

void NativeDynamicShapePlan::printCaptureAudit() const {
  if (lastCaptureAudit_.empty()) {
    DSP_DIAG(SEGMENT, "NativeDynamicShapePlan: No capture audit data (no capture has occurred)");
    return;
  }

  DSP_DIAG(SEGMENT, "╔══════════════════════════════════════════════════════════════════════════╗");
  DSP_DIAG(SEGMENT, "║           CUDA GRAPH CAPTURE AUDIT (per-op node count)                 ║");
  DSP_DIAG(SEGMENT, "╠══════════════════════════════════════════════════════════════════════════╣");
  DSP_DIAG(SEGMENT, "║ Total ops in segment: %zu", lastCaptureAudit_.size());
  DSP_DIAG(SEGMENT, "╠══════════════════════════════════════════════════════════════════════════╣");

  int hostOnlyCount = 0;
  size_t totalNodes = 0;

  for (const auto& entry : lastCaptureAudit_) {
    totalNodes += entry.nodesContributed;
    if (entry.isHostOnly()) {
      hostOnlyCount++;
    }
  }

  DSP_DIAG(SEGMENT, "║ TOP-10 OPS BY NODE COUNT:");
  std::vector<size_t> indices(lastCaptureAudit_.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::sort(indices.begin(), indices.end(), [this](size_t a, size_t b) {
    return lastCaptureAudit_[a].nodesContributed > lastCaptureAudit_[b].nodesContributed;
  });
  int topN = std::min(static_cast<int>(indices.size()), 10);
  for (int i = 0; i < topN; i++) {
    const auto& entry = lastCaptureAudit_[indices[i]];
    DSP_DIAG(SEGMENT, "║  #%2d [slot %3d] %-25s  nodes: %3zu%s",
              i + 1, entry.slotIndex, entry.opName.c_str(), entry.nodesContributed,
              entry.isHostOnly() ? "  *** HOST-ONLY ***" : "");
  }

  DSP_DIAG(SEGMENT, "╠══════════════════════════════════════════════════════════════════════════╣");
  DSP_DIAG(SEGMENT, "║ Total CUDA graph nodes: %zu from %zu ops",
            totalNodes, lastCaptureAudit_.size());
  DSP_DIAG(SEGMENT, "║ Host-only ops: %d, Node-contributing ops: %zu",
            hostOnlyCount, lastCaptureAudit_.size() - hostOnlyCount);
  if (hostOnlyCount > 0) {
    DSP_DIAG(SEGMENT, "║ *** WARNING: %d HOST-ONLY ops detected! ***", hostOnlyCount);
    DSP_DIAG(SEGMENT, "║ Host-only ops do work during capture but NOT during replay.");
    DSP_DIAG(SEGMENT, "║ Their outputs will be STALE on the 2nd+ graph execution.");
  } else {
    DSP_DIAG(SEGMENT, "║ All ops contributed CUDA graph nodes. Graph is complete.");
  }
  DSP_DIAG(SEGMENT, "╚══════════════════════════════════════════════════════════════════════════╝");

  // Record summary into diagnostics
  DSP_DIAG(COMPILE, "capture audit: %zu nodes from %zu ops, %d host-only",
           totalNodes, lastCaptureAudit_.size(), hostOnlyCount);
  if (hostOnlyCount > 0) {
    DSP_THROW(COMPILE, "%d host-only ops in captured graph — outputs stale on replay",
              hostOnlyCount);
  }
}

bool NativeDynamicShapePlan::validateCapturedGraph(int segmentIndex) const {
  if (lastCaptureAudit_.empty()) return true;

  bool allOpsHaveNodes = true;

  for (const auto& entry : lastCaptureAudit_) {
    if (entry.isHostOnly()) {
      allOpsHaveNodes = false;
      DSP_DIAG_SLOT(COMPILE, entry.slotIndex, "CUDA GRAPH VALIDATION FAILURE: slot %d (%s) contributed 0 CUDA graph nodes. "
                   "This op does host-only work that will NOT be replayed",
                   entry.slotIndex, entry.opName.c_str());
    }
  }

  return allOpsHaveNodes;
}


// ═══════════════════════════════════════════════════════════════════════════════
// Additional platform dispatch (extracted from NativeDynamicShapePlan.cpp)
// ═══════════════════════════════════════════════════════════════════════════════

// Helper: log GPU memory state
static void logGpuMemState(const char* label) {
  size_t freeMem = 0, totalMem = 0;
  cudaMemGetInfo(&freeMem, &totalMem);
  size_t usedMem = totalMem - freeMem;

  cudaMemPool_t pool = nullptr;
  int deviceId = 0;
  cudaGetDevice(&deviceId);
  cudaDeviceGetDefaultMemPool(&pool, deviceId);

  uint64_t poolUsed = 0, poolReserved = 0;
  if (pool != nullptr) {
    cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemCurrent, &poolUsed);
    cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemCurrent, &poolReserved);
  }

  DSP_DIAG(MEMORY,
      "[GPU-MEM %s] dev%d: used=%zu MB, free=%zu MB, total=%zu MB | "
      "pool: used=%llu MB, reserved=%llu MB, reclaimable=%llu MB",
      label, deviceId,
      usedMem / (1024*1024), freeMem / (1024*1024), totalMem / (1024*1024),
      poolUsed / (1024ULL*1024), poolReserved / (1024ULL*1024),
      (poolReserved - poolUsed) / (1024ULL*1024));
}

void* NativeDynamicShapePlan::platformBeginExecution(void* stream, bool frozen, int execCount) {
  auto* ctx = new PlanExecutionContext();
  ctx->execCount = execCount;
  ctx->frozen = frozen;
  // Compute needsFullSync and isFrozenSteadyState early — used in this method's
  // sync decisions. The full populateDerivedState() call happens later in execute().
  ctx->needsFullSync = !frozen || execCount <= 1;
  ctx->isFrozenSteadyState = frozen && execCount > 1;

  // Create the per-execution cross-stream sync event.
  // This replaces the file-scope thread_local tl_crossStreamEvent so each
  // execute() call owns its event and there is no hidden per-thread state.
  cudaEventCreateWithFlags(&ctx->crossStreamEvent, cudaEventDisableTiming);

  // Resolve CUDA streams and set up DspStreamGuard RAII
  if (stream != nullptr) {
    ctx->dspStream = *static_cast<cudaStream_t*>(stream);
    ctx->streamGuard = new DspStreamGuard(ctx->dspStream);

    // Resolve LC default stream (a real async stream from ContextBuffers,
    // NOT CUDA stream 0). Post-execution ops (KvScatter, assign, mask updates)
    // run on this stream.
    auto* lcStreamPtr = LaunchContext::defaultContext()->getCudaStream();
    ctx->lcDefaultStream = (lcStreamPtr != nullptr) ? *lcStreamPtr : nullptr;
  }

  // Stream ordering: ensure all async CUDA operations from Java complete
  // before DSP execution begins.
  if (stream != nullptr) {
    // Check for prior CUDA errors before attempting sync
    auto priorErr = cudaGetLastError();
    if (priorErr != cudaSuccess) {
      DSP_DIAG(EXECUTE, "platformBeginExecution: PRIOR CUDA ERROR before sync: %s (%d)",
               cudaGetErrorString(priorErr), static_cast<int>(priorErr));
    }
    DSP_DIAG(EXECUTE, "platformBeginExecution: frozen=%d execCount=%d dspStream=%p lcDefault=%p",
             static_cast<int>(frozen), execCount,
             static_cast<void*>(ctx->dspStream), static_cast<void*>(ctx->lcDefaultStream));

    // Cross-stream sync: make the DSP stream wait for both the LC default
    // stream and CUDA stream 0 before starting execution.
    {
      cudaEvent_t evt = ctx->crossStreamEvent;
      // 1) LC default stream → DSP stream
      if (ctx->lcDefaultStream != nullptr && ctx->lcDefaultStream != ctx->dspStream) {
        cudaEventRecord(evt, ctx->lcDefaultStream);
        cudaStreamWaitEvent(ctx->dspStream, evt, 0);
      }
      // 2) CUDA stream 0 → DSP stream (cuBLAS default handle, misc)
      cudaEventRecord(evt, nullptr);
      cudaStreamWaitEvent(ctx->dspStream, evt, 0);
      ctx->recordEventSync();  // Track: cross-stream event ordering at entry
      DSP_DIAG(EXECUTE, "platformBeginExecution: cross-stream sync done");
    }
    if (ctx->needsFullSync) {
      // For early executions (warmup, capture), also sync the DSP stream
      // itself to ensure any prior DSP work is complete.
      auto syncErr = cudaStreamSynchronize(ctx->dspStream);
      ctx->recordStreamSync();  // Track: full stream sync at entry
      DSP_DIAG(EXECUTE, "platformBeginExecution: cudaStreamSynchronize returned %d (%s)",
               static_cast<int>(syncErr), cudaGetErrorString(syncErr));
    }
  }

  // ── Deterministic cuBLAS for SLOT_BY_SLOT and CUDA_GRAPHS ────────────
  // Three-pronged determinism strategy so captured CUDA graph kernels
  // produce bit-identical results to live (SLOT_BY_SLOT) execution:
  //
  // 1. CUBLAS_PEDANTIC_MATH — forces cuBLAS to select bitwise-reproducible
  //    algorithms. Without this, even CUBLAS_GEMM_DEFAULT can pick algorithms
  //    whose threadblock scheduling order varies between graph capture and
  //    graph replay, producing tiny FP differences that compound through
  //    GDN recurrent state until token divergence (~step 14).
  //
  // 2. No workspace — prevents split-K algorithms that accumulate partial
  //    sums in workspace with non-deterministic reduction order.
  //
  // 3. tl_cublasLtDisabled — blocks cublasLt (which has its own split-K)
  //    and forces CUBLAS_GEMM_DEFAULT instead of CUBLAS_GEMM_DEFAULT_TENSOR_OP.
  //
  // All three must be set for BOTH modes so they use identical cuBLAS state.
  // Modes requiring deterministic cuBLAS enforce PEDANTIC_MATH + no workspace + no Lt.
  // TRITON composite mode manages its own workspace/algorithm lifecycle.
  if (ModeContract::forMode(graphExecutionMode_).requiresDeterministicCublas) {
    auto* handlePtr = reinterpret_cast<cublasHandle_t*>(CublasHelper::getInstance().handle());
    if (handlePtr != nullptr) {
      // (1) Force bitwise-reproducible algorithms
      cublasSetMathMode(*handlePtr, CUBLAS_PEDANTIC_MATH);
      // (2) Clear workspace so cuBLAS cannot pick split-K
      cublasSetWorkspace(*handlePtr, nullptr, 0);
    }
    // Clear thread-locals so reapplyCublasWorkspace() is a no-op
    tl_cublasWorkspacePtr = nullptr;
    tl_cublasWorkspaceSize = 0;
    // (3) Block cublasLt and force CUBLAS_GEMM_DEFAULT
    tl_cublasLtDisabled = true;
    DSP_DIAG(EXECUTE, "platformBeginExecution: deterministic cuBLAS for mode=%d "
             "(PEDANTIC_MATH + no workspace + no Lt)",
             static_cast<int>(graphExecutionMode_));
  }

  return static_cast<void*>(ctx);
}

void NativeDynamicShapePlan::platformEndExecution(void* executionState, void* stream, bool frozen, int execCount) {
  auto* ctx = static_cast<PlanExecutionContext*>(executionState);

  // Cross-stream synchronization: make post-execution streams wait for DSP.
  if (stream != nullptr) {
    DSP_DIAG(EXECUTE, "platformEndExecution: frozen=%d execCount=%d syncLevel=%s "
             "dspStream=%p lcDefault=%p",
             static_cast<int>(ctx->frozen), ctx->execCount, ctx->syncLevelName(),
             static_cast<void*>(ctx->dspStream), static_cast<void*>(ctx->lcDefaultStream));

    if (ctx->isFrozenSteadyState) {
      // Lightweight event-based sync for steady-state frozen replay.
      if (executionCompleteEvent_ == nullptr) {
        cudaEvent_t evt;
        cudaEventCreateWithFlags(&evt, cudaEventDisableTiming);
        executionCompleteEvent_ = static_cast<void*>(new cudaEvent_t(evt));
      }
      cudaEvent_t evt = *static_cast<cudaEvent_t*>(executionCompleteEvent_);
      cudaEventRecord(evt, ctx->dspStream);
      // Make BOTH CUDA stream 0 AND the LC default stream wait for DSP.
      // Post-execution ops (KvScatter, assign, etc.) run on the LC default
      // stream. Without this ordering, they read outputs the DSP stream
      // hasn't finished writing yet.
      cudaStreamWaitEvent(nullptr, evt, 0);  // CUDA stream 0
      if (ctx->lcDefaultStream != nullptr && ctx->lcDefaultStream != ctx->dspStream) {
        cudaStreamWaitEvent(ctx->lcDefaultStream, evt, 0);
        DSP_DIAG(EXECUTE, "platformEndExecution: lcDefault=%p waiting on DSP=%p",
                 (void*)ctx->lcDefaultStream, (void*)ctx->dspStream);
      }
      ctx->recordEventSync();  // Track: event-based ordering at exit
    } else {
      // Full sync for early executions (warmup, capture).
      auto syncErr = cudaStreamSynchronize(ctx->dspStream);
      ctx->recordStreamSync();  // Track: full stream sync at exit
      DSP_DIAG(EXECUTE, "platformEndExecution: cudaStreamSynchronize returned %d (%s) syncLevel=%s",
               static_cast<int>(syncErr), cudaGetErrorString(syncErr), ctx->syncLevelName());
    }
  }

  // Destroy the per-execution cross-stream sync event.
  if (ctx->crossStreamEvent != nullptr) {
    cudaEventDestroy(ctx->crossStreamEvent);
    ctx->crossStreamEvent = nullptr;
  }

  // Restore cuBLAS state for modes that enforced deterministic cuBLAS.
  if (ModeContract::forMode(graphExecutionMode_).requiresDeterministicCublas) {
    if (tl_cublasWorkspacePtr != nullptr) {
      tl_cublasWorkspacePtr = nullptr;
      tl_cublasWorkspaceSize = 0;
    }
    tl_cublasLtDisabled = false;
    // Restore math mode to default (undo CUBLAS_PEDANTIC_MATH from begin)
    auto* handlePtr = reinterpret_cast<cublasHandle_t*>(CublasHelper::getInstance().handle());
    if (handlePtr != nullptr) {
      cublasSetMathMode(*handlePtr, CUBLAS_DEFAULT_MATH);
    }
  }

  // ── TLS STATE ASSERTIONS ─────────────────────────────────────────────────
  // Verify thread-local state consistency at execution boundary.
  // These catch state leaks: if any TLS was set during execution but not
  // properly restored, it poisons subsequent non-DSP operations.
  REQUIRE_TRUE(!tl_graphExecutionActive, 0,
               "TLS LEAK: tl_graphExecutionActive=true at platformEndExecution exit. "
               "A graph capture began but was not properly ended. "
               "mode=%d execCount=%d frozen=%d",
               static_cast<int>(graphExecutionMode_), execCount, static_cast<int>(frozen));
  // tl_cublasLtDisabled should be false by now (restored above for SBS/CG modes,
  // never set for TRITON/other modes).
  REQUIRE_TRUE(!tl_cublasLtDisabled, 0,
               "TLS LEAK: tl_cublasLtDisabled=true at platformEndExecution exit. "
               "cuBLAS determinism override was not restored. mode=%d",
               static_cast<int>(graphExecutionMode_));
  // Capture stream must be null — active capture would mean we're inside beginCapture
  // but exited execution without endCapture.
  REQUIRE_TRUE(tl_graphCaptureStream == nullptr, 0,
               "TLS LEAK: tl_graphCaptureStream=%p at platformEndExecution exit. "
               "A CUDA graph capture stream is still active — endCapture was not called.",
               (void*)tl_graphCaptureStream);

  // Explicitly delete the stream guard before the context.
  // DspStreamGuard restores tl_dspExecutionStream to its previous value.
  delete ctx->streamGuard;
  ctx->streamGuard = nullptr;
  delete ctx;
}

void NativeDynamicShapePlan::platformDumpExternalInputDiagnostics(NDArray** ext, int numExt, int execCount) {
  if (!DSP_DIAG_ENABLED(EXECUTE)) return;
  for (int dbgI = 0; dbgI < numExt; dbgI++) {
    NDArray* arr = ext[dbgI];
    if (arr == nullptr || arr->dataType() != FLOAT32 || arr->lengthOf() <= 0) continue;
    auto* db = arr->dataBuffer();
    int dumpN = std::min((int)arr->lengthOf(), 4);
    std::vector<float> hostBuf(dumpN, 0.0f);
    if (db && arr->specialBuffer() != nullptr) {
      cudaMemcpy(hostBuf.data(), arr->specialBuffer(), dumpN * 4, cudaMemcpyDeviceToHost);
    }
    const char* nm = (dbgI < (int)externalInputNames_.size()) ? externalInputNames_[dbgI].c_str() : "?";
    DSP_DIAG(EXECUTE, "EXT_ENTRY execCount=%d ext[%d]='%s' arr=%p sbuf=%p len=%lld "
             "pAct=%d sAct=%d dev=[%.4f %.4f %.4f %.4f]",
             execCount, dbgI, nm, (void*)arr, arr->specialBuffer(),
             (long long)arr->lengthOf(),
             db ? (db->isPrimaryActual() ? 1 : 0) : -1,
             db ? (db->isSpecialActual() ? 1 : 0) : -1,
             hostBuf[0], hostBuf[1], hostBuf[2], hostBuf[3]);
  }
}

void NativeDynamicShapePlan::platformDumpExtInputGpuValues(NDArray* arr, int extIdx, int execCount, void* stream) {
  if (arr == nullptr) return;
  if (arr->specialBuffer() != nullptr && arr->lengthOf() > 0 && arr->dataType() == FLOAT32) {
    int dumpCount = std::min((int)arr->lengthOf(), 8);
    std::vector<float> hostBuf(dumpCount);
    cudaDeviceSynchronize();
    cudaMemcpy(hostBuf.data(), arr->specialBuffer(), dumpCount * 4, cudaMemcpyDeviceToHost);
    std::string valStr;
    for (int v = 0; v < dumpCount; v++) {
      if (v > 0) valStr += ",";
      char buf[32]; snprintf(buf, sizeof(buf), "%.6f", hostBuf[v]); valStr += buf;
    }
    DSP_DIAG(VERIFY, "EXT_INPUT_START: exec=%d extIdx=%d GPU values: %s",
             execCount, extIdx, valStr.c_str());
  }
}

void NativeDynamicShapePlan::platformClearCastCache() {
  MmulHelper::clearCastCache();
}

void NativeDynamicShapePlan::platformPostSegmentPoolManagement(bool frozen, int execCount) {
  int activeDevice = 0;
  cudaGetDevice(&activeDevice);
  size_t poolUsedPostSegs = 0, poolReservedPostSegs = 0;
  sd::memory::CudaMemoryPool::getInstance().getStats(activeDevice, poolUsedPostSegs, poolReservedPostSegs);
  DSP_DIAG(MEMORY, "post-segments: pool used=%zuMB reserved=%zuMB",
           poolUsedPostSegs / (1024*1024), poolReservedPostSegs / (1024*1024));

  if (frozen) {
    int trimInterval = Environment::getInstance().dspTrimInterval();
    if (trimInterval > 0 && (execCount == 0 || (execCount % trimInterval) == 0)) {
      int trimDeviceId = 0;
      cudaGetDevice(&trimDeviceId);
      sd::memory::CudaMemoryPool::getInstance().trimPool(trimDeviceId);
      DSP_DIAG(MEMORY, "post-segments: trimmed pool on device %d (frozen exec=%d, interval=%d)",
               trimDeviceId, execCount, trimInterval);
    }
  }
}

void NativeDynamicShapePlan::platformDumpLogitsArgmax(int execCount, void* stream) {
  if (!DSP_DIAG_ENABLED(VERIFY) || execCount > 10) return;

  // Sync the DSP stream before reading GPU data to avoid racing in-flight writes.
  if (stream != nullptr) {
    cudaStream_t dspStr = *static_cast<cudaStream_t*>(stream);
    cudaStreamSynchronize(dspStr);
    DSP_DIAG(VERIFY, "LOGITS_ARGMAX: synced dspStream=%p before read (exec=%d)",
             static_cast<void*>(dspStr), execCount);
  }

  // Find the logits output: largest FLOAT32 requested output (vocab-sized).
  // KV cache outputs are typically rank 4 with small last dim (e.g., 4416),
  // while logits are rank 2-3 with large last dim (e.g., 49152).
  for (int i = 0; i < numRequestedOutputs_; i++) {
    int slotIdx = requestedOutputSlotIndices_[i];
    NDArray* arr = (slotIdx >= 0 && slotIdx < totalOutputSlots_) ? outputSlots_[slotIdx] : nullptr;
    if (arr == nullptr) continue;
    void* sbuf = arr->specialBuffer();
    // Logits: FLOAT32, length >= 10000 (any reasonable vocab), rank <= 3
    if (sbuf && arr->dataType() == FLOAT32 && arr->lengthOf() >= 10000 && arr->rankOf() <= 3) {
      auto len = arr->lengthOf();
      std::vector<float> fullBuf(len);
      cudaMemcpy(fullBuf.data(), sbuf, len * sizeof(float), cudaMemcpyDeviceToHost);
      float maxVal = -1e30f;
      int maxIdx = -1;
      bool allZero = true;
      for (int j = 0; j < (int)len; j++) {
        if (fullBuf[j] != 0.0f) allZero = false;
        if (fullBuf[j] > maxVal) { maxVal = fullBuf[j]; maxIdx = j; }
      }
      DSP_DIAG_SLOT(VERIFY, slotIdx,
          "LOGITS_ARGMAX exec=%d reqOut[%d] len=%lld maxIdx=%d maxVal=%.6f allZero=%d",
          execCount, i, (long long)len, maxIdx, maxVal, allZero ? 1 : 0);
    }
  }
}

void NativeDynamicShapePlan::platformDetectAndPrepareBatchedGemm(NDArray** ext, int numExt, void* stream) {
  if (!planLifecycle_.isSlotBySlot() && executeCount_ == 1 && batchedGemmGroups_.empty() &&
      Environment::getInstance().dspBatchedGemm()) {
    detectBatchedGemmGroups(ext, numExt);
    if (!batchedGemmGroups_.empty()) {
      cudaStream_t execStream = stream ? *static_cast<cudaStream_t*>(stream) : static_cast<cudaStream_t>(nullptr);
      prepareBatchedGemmDevice(execStream);
    }
  }
}

void NativeDynamicShapePlan::platformPreReplayPoolStats(size_t& poolUsedOut, size_t& poolReservedOut) {
  int activeDevice = 0;
  cudaGetDevice(&activeDevice);
  sd::memory::CudaMemoryPool::getInstance().getStats(activeDevice, poolUsedOut, poolReservedOut);
  DSP_DIAG(MEMORY, "pre-segments: pool used=%zuMB reserved=%zuMB",
           poolUsedOut / (1024*1024), poolReservedOut / (1024*1024));

  if (!planLifecycle_.isSlotBySlot() && executeCount_ > 0 &&
      cublasWorkspaceBuffer_ != nullptr && cublasWorkspaceSize_ > 0) {
    DSP_DIAG(MEMORY, "pre-segments: cuBLAS workspace PRESERVED (%zuMB) — plans stable",
             cublasWorkspaceSize_ / (1024*1024));
  }
}

void NativeDynamicShapePlan::platformPostReplayPoolManagement(size_t poolUsedPre, bool frozen, int execCount) {
  int activeDevice = 0;
  cudaGetDevice(&activeDevice);
  size_t poolUsedPostSegs = 0, poolReservedPostSegs = 0;
  sd::memory::CudaMemoryPool::getInstance().getStats(activeDevice, poolUsedPostSegs, poolReservedPostSegs);
  long long deltaMB = static_cast<long long>(poolUsedPostSegs - poolUsedPre) / (1024LL*1024);
  DSP_DIAG(MEMORY, "post-segments: pool used=%zuMB reserved=%zuMB (delta=%lldMB from pre-segs)",
           poolUsedPostSegs / (1024*1024), poolReservedPostSegs / (1024*1024), deltaMB);

  if (frozen) {
    int trimInterval = Environment::getInstance().dspTrimInterval();
    if (trimInterval > 0 && (execCount == 0 || (execCount % trimInterval) == 0)) {
      int trimDeviceId = 0;
      cudaGetDevice(&trimDeviceId);
      sd::memory::CudaMemoryPool::getInstance().trimPool(trimDeviceId);
      DSP_DIAG(MEMORY, "post-segments: trimmed pool on device %d (frozen exec=%d, interval=%d)",
               trimDeviceId, execCount, trimInterval);
    }
  }
}

void NativeDynamicShapePlan::platformTraceSlotValues(const GraphSegment& seg, void* stream, int execCount) {
  int traceSlot = sd::graph::DspDiagnostics::getInstance().traceSlot();
  if (traceSlot >= 0 && traceSlot < totalOutputSlots_ && !planLifecycle_.isSlotBySlot()) {
    auto* arr = outputSlots_[traceSlot];
    if (arr != nullptr) {
      auto* db = arr->dataBuffer();
      void* gpuPtr = arr->specialBuffer();
      float firstVals[4] = {0, 0, 0, 0};
      if (gpuPtr != nullptr && arr->lengthOf() > 0 && arr->dataType() == FLOAT32) {
        int n = std::min((int)arr->lengthOf(), 4);
        cudaStream_t execStr = (stream != nullptr) ? *static_cast<cudaStream_t*>(stream) : nullptr;
        if (execStr != nullptr) cudaStreamSynchronize(execStr);
        cudaMemcpy(firstVals, gpuPtr, n * sizeof(float), cudaMemcpyDeviceToHost);
      }
      bool allZero = (firstVals[0] == 0.0f && firstVals[1] == 0.0f &&
                     firstVals[2] == 0.0f && firstVals[3] == 0.0f);
      bool hasNaN = (std::isnan(firstVals[0]) || std::isnan(firstVals[1]) ||
                    std::isnan(firstVals[2]) || std::isnan(firstVals[3]));
      if (allZero || hasNaN || execCount > 0) {
        const char* tag = hasNaN ? "NaN" : (allZero ? "ZERO" : "OK");
        DSP_DIAG(VERIFY, "SLOT_TRACE %s after seg[%d-%d]: slot=%d "
                "arr=%p gpuPtr=%p db=%p closed=%d pAct=%d sAct=%d "
                "vals=[%.6f,%.6f,%.6f,%.6f] execCount=%d",
                tag,
                seg.def.startSlot, seg.def.endSlot, traceSlot,
                (void*)arr, gpuPtr, (void*)db,
                db ? db->isClosed() : -1,
                db ? (db->isPrimaryActual() ? 1 : 0) : -1,
                db ? (db->isSpecialActual() ? 1 : 0) : -1,
                firstVals[0], firstVals[1], firstVals[2], firstVals[3],
                execCount);
      }
    }
  }
}

SelectedBackend NativeDynamicShapePlan::platformResolveBackend(bool isGraphCapture) const {
  return isGraphCapture ? SelectedBackend::CUDA_GRAPHS : SelectedBackend::GPU_COMPILER;
}

bool NativeDynamicShapePlan::platformShouldBreakSegmentAtTraitBoundary(int currIdx, int prevIdx) const {
  return false;  // No trait-based segmentation on GPU
}

size_t NativeDynamicShapePlan::platformEstimateCaptureBudget() const {
  // Query actual GPU free memory and compute how much is available for
  // a single segment's intermediate buffers during CUDA graph capture.
  //
  // The budget accounts for:
  //   - capture workspace (512MB default, from DspConfig::captureWorkspaceMb)
  //   - cuBLAS workspace (from DspConfig::cublasWorkspaceMb)
  //   - graph metadata overhead (~20% of buffer footprint)
  //   - pinned host workspace for H2D nodes
  //   - safety margin for CUDA runtime allocations
  //
  // This adapts automatically to any GPU size (24GB, 48GB, 80GB) and any
  // model size (how much memory weights + KV cache consume).

  size_t gpuFree = 0, gpuTotal = 0;
  cudaMemGetInfo(&gpuFree, &gpuTotal);

  // Subtract fixed overhead that capture always needs
  size_t captureWsMb = static_cast<size_t>(sd::Environment::getInstance().dsp().captureWorkspaceMb());
  size_t cublasWsMb  = static_cast<size_t>(sd::Environment::getInstance().dsp().cublasWorkspaceMb());
  size_t fixedOverhead = (captureWsMb + cublasWsMb) * 1024ULL * 1024ULL;

  // Reserve 20% of remaining free memory as safety margin for graph metadata,
  // CUDA runtime internal allocations, and fragmentation.
  size_t safetyMargin = gpuFree / 5;

  size_t totalOverhead = fixedOverhead + safetyMargin;
  if (gpuFree <= totalOverhead) {
    // Almost no memory left — allow at most a small segment.
    // Return 64MB floor so we don't end up with 1-op segments.
    return 64ULL * 1024 * 1024;
  }

  size_t budget = gpuFree - totalOverhead;

  DSP_DIAG(MEMORY, "platformEstimateCaptureBudget: gpuFree=%zuMB gpuTotal=%zuMB "
           "fixedOverhead=%zuMB safetyMargin=%zuMB budget=%zuMB",
           gpuFree / (1024*1024), gpuTotal / (1024*1024),
           fixedOverhead / (1024*1024), safetyMargin / (1024*1024),
           budget / (1024*1024));

  return budget;
}

void NativeDynamicShapePlan::platformReleaseSegmentGpuResources() {
  logGpuMemState("STEP-0-ENTRY");
  for (auto& seg : segments_) {
    // Clean up monolithic replay handle
    if (seg.exec.replayHandle) {
      if (seg.exec.replayHandle->getWorkspacePtr() != nullptr) {
        seg.exec.replayHandle->releaseWorkspace(nullptr, seg.def.startSlot);
      }
      seg.exec.replayHandle->freeHostPointers();
      seg.exec.replayHandle->clearExternalAddresses();
      seg.exec.replayHandle.reset();
    }
    // Clean up merged replay handles (island-merged capture groups).
    // Must call releaseWorkspace explicitly for pool deregistration —
    // RAII destruction only calls cudaFree, skipping pool-aware cleanup.
    for (auto& h : seg.exec.compositeReplaySchedule.mergedReplayHandles) {
      if (h) {
        if (h->getWorkspacePtr() != nullptr) {
          h->releaseWorkspace(nullptr, seg.def.startSlot);
        }
        h->freeHostPointers();
        h->clearExternalAddresses();
        h.reset();
      }
    }
    seg.exec.compositeReplaySchedule.mergedReplayHandles.clear();
    for (auto& u : seg.exec.compositeReplaySchedule.units) {
      u.mergedGroupId = -1;
      u.isMergedLeader = false;
    }
    // Clean up composite (per-island) replay handles
    for (auto& h : seg.exec.compositeReplaySchedule.compositeReplayHandles) {
      if (h) {
        if (h->getWorkspacePtr() != nullptr) {
          h->releaseWorkspace(nullptr, seg.def.startSlot);
        }
        h->freeHostPointers();
        h->clearExternalAddresses();
        h.reset();
      }
    }
    seg.exec.gapOpsCapturedInGraph = false;
    seg.exec.bumpArgGeneration();
    seg.exec.argTableStable = false;
    seg.exec.addrKeyStableCount = 0;
    seg.exec.slotAddrStableCount = 0;
    seg.exec.capturedInputAddrKey = 0;
    seg.exec.compilationFailed = false;
    seg.exec.executionCount = 0;
    delete seg.exec.jitKernel;
    seg.exec.jitKernel = nullptr;
    seg.exec.cachedShapeKey = 0;
    seg.exec.capturedCreateValueKey = 0;
    seg.exec.captureOomRetries = 0;
    seg.exec.captureRetryAfterExec = 0;
    seg.exec.compiledByBackend.clear();
    seg.exec.segPhase.reset();  // PRIMARY: unified lifecycle
    seg.exec.lifecycleState = SegmentLifecycleState::NEEDS_WARMUP;  // legacy sync
    seg.exec.jitShapeKey = 0;
    seg.exec.jitCompileFailed = false;
    seg.def.shapeKeyState.reset();
  }
  logGpuMemState("STEP-1-AFTER-SEGMENTS");

  // Free cuBLAS workspace
  if (cublasWorkspaceBuffer_ != nullptr) {
    cudaFree(cublasWorkspaceBuffer_);
    cublasWorkspaceBuffer_ = nullptr;
    cublasWorkspaceSize_ = 0;
  }

  // Free batch-D2D and batched-GEMM device arrays
  freeBatchD2DResources();
  freeBatchedGemmResources();
  logGpuMemState("STEP-1-AFTER-BATCH-RESOURCES");
}

void NativeDynamicShapePlan::platformMigrateWeightsAndClearCaches() {
  DSP_DIAG(MEMORY, "releaseGpuIntermediates: freed intermediate NDArrays");
  logGpuMemState("STEP-2-AFTER-INTERMEDIATES");

  // Free untracked output cache
  if (untrackedOutputCache_) {
    for (int i = 0; i < untrackedOutputCacheSize_; i++) {
      delete untrackedOutputCache_[i];
      untrackedOutputCache_[i] = nullptr;
    }
  }

  // Clear MmulHelper cast cache
  MmulHelper::clearCastCache();
  logGpuMemState("STEP-4-AFTER-CAST-CACHE");

  // Migrate weight buffers from async pool to direct cudaMalloc
  {
    cudaError_t syncErr = cudaDeviceSynchronize();
    if (syncErr != cudaSuccess) {
      DSP_DIAG(MEMORY, "releaseGpuIntermediates: cudaDeviceSynchronize failed: %s",
               cudaGetErrorString(syncErr));
      cudaGetLastError();
    }
    int deviceId = 0;
    cudaGetDevice(&deviceId);

    int migratedCount = 0;
    int skippedDirect = 0;
    int skippedNonDevice = 0;
    int failedMigrations = 0;
    size_t migratedBytes = 0;
    size_t totalWeightBytes = 0;
    auto& pool = memory::CudaMemoryPool::getInstance();

    DSP_DIAG(MEMORY, "Weight migration: %zu protected weight buffers to check",
             protectedWeightBuffers_.size());

    for (auto* db : protectedWeightBuffers_) {
      if (db == nullptr || db->special() == nullptr) continue;

      size_t bufSize = db->getLenInBytes();
      totalWeightBytes += bufSize;

      if (pool.isDirectAllocation(db->special())) {
        skippedDirect++;
        continue;
      }

      cudaPointerAttributes ptrAttrs;
      cudaError_t attrErr = cudaPointerGetAttributes(&ptrAttrs, db->special());
      if (attrErr != cudaSuccess) {
        cudaGetLastError();
        continue;
      }
      if (ptrAttrs.type != cudaMemoryTypeDevice) {
        skippedNonDevice++;
        continue;
      }

      if (bufSize == 0) continue;

      void* directPtr = nullptr;
      cudaError_t allocErr = cudaMalloc(&directPtr, bufSize);
      if (allocErr != cudaSuccess || directPtr == nullptr) {
        cudaGetLastError();
        failedMigrations++;
        DSP_DIAG(MEMORY,
            "Weight migration FAILED for %zu bytes (%zu MB): %s",
            bufSize, bufSize / (1024*1024), cudaGetErrorString(allocErr));
        continue;
      }

      cudaError_t copyErr = cudaMemcpy(directPtr, db->special(), bufSize, cudaMemcpyDeviceToDevice);
      if (copyErr != cudaSuccess) {
        cudaFree(directPtr);
        cudaGetLastError();
        DSP_DIAG(MEMORY, "releaseGpuIntermediates: weight migration memcpy failed for %zu bytes: %s",
                 bufSize, cudaGetErrorString(copyErr));
        continue;
      }

      void* oldPtr = db->special();
      pool.free(oldPtr, deviceId);
      db->replaceSpecialBuffer(directPtr, true);
      pool.registerDirectAllocation(directPtr, bufSize);

      migratedCount++;
      migratedBytes += bufSize;
    }

    if (migratedCount > 0) {
      cudaDeviceSynchronize();
    }

    DSP_DIAG(MEMORY,
        "Weight migration summary: total=%zu MB, migrated=%d (%zu MB), "
        "skippedDirect=%d, skippedNonDevice=%d, failed=%d",
        totalWeightBytes / (1024*1024), migratedCount, migratedBytes / (1024*1024),
        skippedDirect, skippedNonDevice, failedMigrations);

    pool.trimPool(deviceId);
    logGpuMemState("STEP-4b-AFTER-MIGRATION-AND-TRIM");

    // Clear shape and TAD caches
    {
      auto shapeEntriesBefore = ConstantShapeHelper::getInstance().getCachedEntries();
      auto tadEntriesBefore = ConstantTadHelper::getInstance().getCachedEntries();

      ConstantShapeHelper::getInstance().clearCache();
      ConstantTadHelper::getInstance().clearCache();

      auto shapeEntriesAfter = ConstantShapeHelper::getInstance().getCachedEntries();
      auto tadEntriesAfter = ConstantTadHelper::getInstance().getCachedEntries();

      DSP_DIAG(MEMORY,
          "Shape/TAD cache clear: shapes %lld->%lld, TADs %lld->%lld",
          static_cast<long long>(shapeEntriesBefore), static_cast<long long>(shapeEntriesAfter),
          static_cast<long long>(tadEntriesBefore), static_cast<long long>(tadEntriesAfter));

      cudaDeviceSynchronize();
      pool.trimPool(deviceId);
      logGpuMemState("STEP-4c-AFTER-CACHE-CLEAR-AND-TRIM");
    }
  }

  // Invalidate Triton compiled kernel cache
#if HAVE_TRITON
  {
    std::vector<std::pair<int,int>> segRanges;
    segRanges.reserve(segments_.size());
    for (auto& seg : segments_) {
      segRanges.emplace_back(seg.def.startSlot, seg.def.endSlot);
    }
    if (!segRanges.empty()) {
      TritonGraphBackend::getInstance().invalidateCacheForSegments(segRanges);
    }
  }
#endif
}

}  // namespace graph
}  // namespace sd

#endif  // SD_CUDA

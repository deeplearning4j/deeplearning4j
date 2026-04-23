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
#include <memory/cuda/CudaMemoryPool.h>
#include <helpers/AttentionWorkspace.h>
#include <graph/gpu/NvrtcKernelBuilder.h>
#include <graph/gpu/NvrtcKernelCache.h>
#include <ops/declarable/helpers/kv_scatter.h>
#include <system/Environment.h>

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

namespace sd {
namespace graph {

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
      (graphExecutionMode_ == GraphExecutionMode::GEM_AUTO ||
       graphExecutionMode_ == GraphExecutionMode::GEM_CUDA_GRAPHS ||
       graphExecutionMode_ == GraphExecutionMode::GEM_TRITON);
  bool frozenFastPathInputStable = true;
  bool frozenFastPathSlotsStable = true;
  if (allowFrozenGraphFastPath && shapesFrozen_ && executeCount_ >= 1 && segments_.size() == 1) {
    auto& seg0 = segments_[0];
    // Per-address comparison: catches address changes that the hash may miss
    // (e.g. CUDA pool reuses an address for a different allocation, changing
    // only a subset of the hashed values in a way that produces a collision).
    if (pointersStable_) {
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
    if (pointersStable_) {
      frozenFastPathSlotsStable = true;
    } else if (seg0.exec.capturedSlotAddrHash != 0) {
      frozenFastPathSlotsStable =
          (computeSlotAddrHash(outputSlots_, seg0.def.startSlot, seg0.def.endSlot, totalOutputSlots_) ==
           seg0.exec.capturedSlotAddrHash);
    }
  }
  bool fastPathApplicable = allowFrozenGraphFastPath && shapesFrozen_ && executeCount_ >= 1 && segments_.size() == 1 &&
        frozenFastPathInputStable && frozenFastPathSlotsStable &&
        segments_[0].exec.replayHandle != nullptr &&
        segments_[0].exec.replayHandle->isReady();
  bool compositeFastPathApplicable = false;
  if (!fastPathApplicable && allowFrozenGraphFastPath && shapesFrozen_ && executeCount_ >= 1 && segments_.size() == 1 &&
      frozenFastPathInputStable && frozenFastPathSlotsStable &&
      !Environment::getInstance().tritonVerifyKernels()) {
    compositeFastPathApplicable = hasCompositeHandles(segments_[0]);
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

  // Clear stale CUDA errors
  sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(0);
  sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage("");
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

  // Lazily populate variableExternalInputIndices_ from externalInputIsVariable_
  // so the sync loop below iterates only variable inputs (2-3) not all (~1333).
  if (!variableIndicesCached_ && !externalInputIsVariable_.empty()) {
    variableExternalInputIndices_.clear();
    for (int i = 0; i < static_cast<int>(externalInputIsVariable_.size()); ++i) {
      if (externalInputIsVariable_[i]) {
        variableExternalInputIndices_.push_back(i);
      }
    }
    variableIndicesCached_ = true;
    DSP_DIAG(EXECUTE, "FROZEN_FAST_PATH: cached %d variable ext input indices out of %d total",
             static_cast<int>(variableExternalInputIndices_.size()),
             static_cast<int>(externalInputIsVariable_.size()));
  }

  if (fastPathApplicable) {
    bool ok = true;
  int syncedCount = 0, skippedCount = 0;
  // Steady-state: iterate only cached variable indices instead of all external inputs
  if (!variableExternalInputIndices_.empty()) {
    for (int idx = 0; idx < static_cast<int>(variableExternalInputIndices_.size()); ++idx) {
      int ei = variableExternalInputIndices_[idx];
      if (ei < 0 || ei >= numExternalInputs || externalInputs[ei] == nullptr) continue;
      auto* db = externalInputs[ei]->dataBuffer();
      if (db != nullptr && db->isPrimaryActual()) {
        db->syncToSpecial(true);  // Force H2D: host has newer data
      }
      syncedCount++;
    }
    skippedCount = numExternalInputs - static_cast<int>(variableExternalInputIndices_.size());
    if (skippedCount < 0) skippedCount = 0;
  } else {
    for (int ei = 0; ei < numExternalInputs; ei++) {
      if (externalInputs[ei] == nullptr) continue;
      // Same guard: only force H2D when host is authoritative
      auto* db = externalInputs[ei]->dataBuffer();
      if (db != nullptr && db->isPrimaryActual()) {
        db->syncToSpecial(true);
      }
      syncedCount++;
    }
  }
  DSP_DIAG(EXECUTE, "NativeDSP::frozenFastPath: synced=%d skipped=%d external inputs",
           syncedCount, skippedCount);

  if (ok) {
    auto tCopyDone = executionTimingEnabled_ ? Clock::now() : Clock::time_point{};

    // Cross-stream ordering: .assign() runs on the default stream while graph
    // replay launches on cudaStr. Ensure cudaStr waits on the default stream.
    {
      cudaStream_t defaultStream = nullptr;
      auto* defaultStreamPtr = LaunchContext::defaultContext()->getCudaStream();
      if (defaultStreamPtr != nullptr) {
        defaultStream = *defaultStreamPtr;
      }
      if (defaultStream != nullptr && defaultStream != cudaStr) {
        auto* execCtxFast = static_cast<PlanExecutionContext*>(activeExecutionContext());
        cudaEvent_t crossStreamEvt = (execCtxFast != nullptr) ? execCtxFast->crossStreamEvent : nullptr;
        if (crossStreamEvt != nullptr) {
          cudaEventRecord(crossStreamEvt, defaultStream);
          cudaStreamWaitEvent(cudaStr, crossStreamEvt, 0);
        }
        DSP_DIAG(EXECUTE, "CROSS_STREAM_SYNC: frozen fast path replay stream %p waiting on "
                 "default stream %p for seg[%d-%d]",
                 (void*)cudaStr, (void*)defaultStream, seg.def.startSlot, seg.def.endSlot);
      }
    }

#if HAVE_TRITON
    // Match compositeReplay: gate arg table refresh on !argTableStable
    if (!seg.exec.argTableStable) {
      auto* tritonBackend = dynamic_cast<TritonGraphBackend*>(getGpuGraphBackend());
      if (tritonBackend != nullptr) {
        tritonBackend->refreshArgTablesForReplay(seg, externalInputs, numExternalInputs,
                                                 outputSlots_, totalOutputSlots_,
                                                 stream);
        // Copy consolidated arg table to device after refresh (was missing)
        tritonBackend->copyConsolidatedArgTableToDevice(seg, stream);
      }
    } else {
      DSP_DIAG(EXECUTE, "FROZEN_FAST_PATH: argTableStable — skip refresh seg[%d-%d]",
               seg.def.startSlot, seg.def.endSlot);
    }
#endif

    // prezeroSegmentOutputs is NOT called on the replay hot path — output zeroing
    // is captured into the CUDA graph during the slot-by-slot capture phase and
    // replays automatically. Per-step cudaMemsetAsync here would be redundant work.

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

      if (seg.exec.lifecycleState == SegmentLifecycleState::CAPTURED) {
        DSP_TRACE_LIFECYCLE(trace_, static_cast<int8_t>(0),
                            static_cast<uint8_t>(seg.exec.lifecycleState),
                            static_cast<uint8_t>(SegmentLifecycleState::REPLAYING),
                            static_cast<uint32_t>(executeCount_));
        SegmentLifecycle::markReplaying(seg.exec);
      }

      // Tick actuality: CUDA graph replay writes device memory directly without
      // registerSpecialUse. Without this tick, syncToHost sees stale host data.
      for (int s = seg.def.startSlot; s <= seg.def.endSlot && s < totalOutputSlots_; s++) {
        NDArray* arr = outputSlots_[s];
        if (arr != nullptr && arr->dataBuffer() != nullptr && !arr->dataBuffer()->isClosed()) {
          arr->tickWriteDevice();
        }
      }

      auto tScatterDone = executionTimingEnabled_ ? Clock::now() : Clock::time_point{};
      if (cudaStr != nullptr) cudaStreamSynchronize(cudaStr);
      auto tSyncDone = executionTimingEnabled_ ? Clock::now() : Clock::time_point{};

      // ── REPLAY VERIFICATION (CUDA_GRAPHS path) ──────────────────────────
      if (Environment::getInstance().tritonVerifyKernels()) {
        performReplayVerify(seg, externalInputs, numExternalInputs, stream, "CUDA_GRAPHS");
      }

      if (executionTimingEnabled_) {
        auto copyUs = std::chrono::duration_cast<std::chrono::microseconds>(tCopyDone - t0).count();
        auto launchUs = std::chrono::duration_cast<std::chrono::microseconds>(tLaunchDone - tCopyDone).count();
        auto scatterUs = std::chrono::duration_cast<std::chrono::microseconds>(tScatterDone - tLaunchDone).count();
        auto syncUs = std::chrono::duration_cast<std::chrono::microseconds>(tSyncDone - tScatterDone).count();
        auto totalUs = std::chrono::duration_cast<std::chrono::microseconds>(tSyncDone - t0).count();
        DSP_DIAG(TIMING, "DSP timing: copy=%lldus launch=%lldus scatter=%lldus sync=%lldus total=%lldus "
                 "(copied=%d skipped=%d)",
                 copyUs, launchUs, scatterUs, syncUs, totalUs, syncedCount, 0);
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

      if (seg.exec.lifecycleState == SegmentLifecycleState::CAPTURED) {
        DSP_TRACE_LIFECYCLE(trace_, static_cast<int8_t>(0),
                            static_cast<uint8_t>(seg.exec.lifecycleState),
                            static_cast<uint8_t>(SegmentLifecycleState::REPLAYING),
                            static_cast<uint32_t>(executeCount_));
        SegmentLifecycle::markReplaying(seg.exec);
      }

      // compositeReplay already ticks all segment slots at its canonical end
      // (gpubackend.cu:1340-1345). No need to double-tick here.

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
  if (!shapesFrozen_ || executeCount_ == 0) {
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
  if (compilationDone_ || executeCount_ < 1 || graphExecutionMode_ == GraphExecutionMode::GEM_SLOT_BY_SLOT) {
    DSP_DIAG(COMPILE, "platformPrecompileSegments: skipped (compilationDone=%d execCount=%d mode=%d)",
             compilationDone_ ? 1 : 0, executeCount_, static_cast<int>(graphExecutionMode_));
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
    bool tryCapture = seg.def.isCapturable || (shapesFrozen_ && executeCount_ > 0);
    if (!tryCapture) continue;
    if (!gpuBackend->canFuseSegment(slots_, seg.def.startSlot, seg.def.endSlot)) continue;
    LongType segShapeKey = computeSegmentShapeKey(seg, externalInputs, numExternalInputs);
    int segTargetDevice = 0;
    if (seg.def.startSlot >= 0 && seg.def.startSlot < numSlots_) {
      segTargetDevice = slots_[seg.def.startSlot].targetDeviceId;
      if (segTargetDevice < 0) segTargetDevice = 0;
    }
    tasks.push_back({si, segShapeKey, segTargetDevice});
  }

  if (tasks.size() <= 1) return;

  // Determine thread count for parallel precompilation.
  // Inner sub-segment parallelism is handled by compileSegment (DEFAULT_MAX_PARALLEL_COMPILATIONS).
  // Outer segment-level parallelism is safe because:
  // - Each compilation creates its own MLIRContext (via getMlirContextMutex-protected factory)
  // - cuModuleLoadDataEx is serialized via loadModuleMtx
  // - LLVM init is done via std::once_flag
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

    if (sourceDevice < 0) sourceDevice = 0;  // External or auto — assume device 0
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
  if (graphExecutionMode_ == GraphExecutionMode::GEM_SLOT_BY_SLOT) {
    DSP_DIAG_SEG(EXECUTE, segment.def.startSlot, "platformShouldUseGraph: false (GEM_SLOT_BY_SLOT)");
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
  if (!shapesFrozen_) return false;

  bool tryCapture = (segment.def.isCapturable || (shapesFrozen_ && executeCount_ > 0))
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

      // GPU backend execution failed. Mark as failed and throw immediately.
      // Silent fallback to slot-by-slot masks the real bug.
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
        segment.exec.compilationFailed = true;
        DSP_THROW_SEG(COMPILE, segment.def.startSlot,
                      "NativeDSP::execute: CUDA graph capture FAILED for seg[%d-%d] status=%d. "
                      "Graph capture failed — fix the root cause.",
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

  // Optionally invalidate Triton singleton cache entries for this plan's segments.
  // Default OFF: compiled kernels are reused across plan lifetimes (the disk
  // cache key no longer includes slot numbers, so modules stay valid).
  // Set ND4J_TRITON_INVALIDATE_ON_PLAN_FREE=1 to enable aggressive cleanup
  // if GPU driver memory from accumulated CUmodule handles becomes a concern.
#if HAVE_TRITON
  if (Environment::getInstance().tritonInvalidateOnPlanFree()) {
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

  // Free replay workspaces and JIT kernels from all segments
  for (auto& seg : segments_) {
    if (seg.exec.replayHandle) {
      if (seg.exec.replayHandle->getWorkspacePtr() != nullptr) {
        seg.exec.replayHandle->releaseWorkspace(nullptr, seg.def.startSlot);
      }
      seg.exec.replayHandle->freeHostPointers();
      seg.exec.replayHandle->clearExternalAddresses();
      seg.exec.replayHandle.reset();
    }
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
  size_t poolUsedPostSegs = 0, poolReservedPostSegs = 0;
  sd::memory::CudaMemoryPool::getInstance().getStats(0, poolUsedPostSegs, poolReservedPostSegs);
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
  if (shapesFrozen_ && executeCount_ == 1 && batchedGemmGroups_.empty() &&
      Environment::getInstance().dspBatchedGemm()) {
    detectBatchedGemmGroups(ext, numExt);
    if (!batchedGemmGroups_.empty()) {
      cudaStream_t execStream = stream ? *static_cast<cudaStream_t*>(stream) : static_cast<cudaStream_t>(nullptr);
      prepareBatchedGemmDevice(execStream);
    }
  }
}

void NativeDynamicShapePlan::platformPreReplayPoolStats(size_t& poolUsedOut, size_t& poolReservedOut) {
  sd::memory::CudaMemoryPool::getInstance().getStats(0, poolUsedOut, poolReservedOut);
  DSP_DIAG(MEMORY, "pre-segments: pool used=%zuMB reserved=%zuMB",
           poolUsedOut / (1024*1024), poolReservedOut / (1024*1024));

  if (shapesFrozen_ && executeCount_ > 0 &&
      cublasWorkspaceBuffer_ != nullptr && cublasWorkspaceSize_ > 0) {
    DSP_DIAG(MEMORY, "pre-segments: cuBLAS workspace PRESERVED (%zuMB) — plans stable",
             cublasWorkspaceSize_ / (1024*1024));
  }
}

void NativeDynamicShapePlan::platformPostReplayPoolManagement(size_t poolUsedPre, bool frozen, int execCount) {
  size_t poolUsedPostSegs = 0, poolReservedPostSegs = 0;
  sd::memory::CudaMemoryPool::getInstance().getStats(0, poolUsedPostSegs, poolReservedPostSegs);
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
  if (traceSlot >= 0 && traceSlot < totalOutputSlots_ && shapesFrozen_) {
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

void NativeDynamicShapePlan::platformReleaseSegmentGpuResources() {
  logGpuMemState("STEP-0-ENTRY");
  for (auto& seg : segments_) {
    if (seg.exec.replayHandle) {
      if (seg.exec.replayHandle->getWorkspacePtr() != nullptr) {
        seg.exec.replayHandle->releaseWorkspace(nullptr, seg.def.startSlot);
      }
      seg.exec.replayHandle->freeHostPointers();
      seg.exec.replayHandle->clearExternalAddresses();
      seg.exec.replayHandle.reset();
    }
    seg.exec.gapOpsCapturedInGraph = false;
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
    seg.exec.lifecycleState = SegmentLifecycleState::NEEDS_WARMUP;
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
      cudaFreeAsync(oldPtr, nullptr);
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

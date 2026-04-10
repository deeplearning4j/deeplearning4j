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
#include <graph/DspHashUtils.h>
#include <graph/DspVerifyUtils.h>
#include <graph/cuda/CudaGraphReplayHandle.h>
#include <graph/DspStreamGuard.h>
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

namespace {

bool isStrictNoFallbackMode(GraphExecutionMode mode) {
  return mode == GraphExecutionMode::GEM_TRITON;
}

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
       graphExecutionMode_ == GraphExecutionMode::GEM_CUDA_GRAPHS);
  bool frozenFastPathInputStable = true;
  bool frozenFastPathSlotsStable = true;
  if (allowFrozenGraphFastPath && shapesFrozen_ && executeCount_ >= 1 && segments_.size() == 1) {
    auto& seg0 = segments_[0];
    // Per-address comparison: catches address changes that the hash may miss
    // (e.g. CUDA pool reuses an address for a different allocation, changing
    // only a subset of the hashed values in a way that produces a collision).
    if (seg0.exec.replayHandle && !seg0.exec.replayHandle->getCapturedExternalAddresses().empty()) {
      frozenFastPathInputStable = externalAddrsMatch(seg0, externalInputs, numExternalInputs);
    } else if (seg0.exec.capturedInputAddrKey != 0) {
      // Legacy fallback: hash-based check for graphs captured before the
      // per-address snapshot was introduced.
      frozenFastPathInputStable =
          (computeSegmentInputAddrKey(seg0, externalInputs, numExternalInputs) == seg0.exec.capturedInputAddrKey);
    }
    if (seg0.exec.capturedSlotAddrHash != 0) {
      frozenFastPathSlotsStable =
          (computeSlotAddrHash(outputSlots_, seg0.def.startSlot, seg0.def.endSlot, totalOutputSlots_) ==
           seg0.exec.capturedSlotAddrHash);
    }
  }
  if (!(allowFrozenGraphFastPath && shapesFrozen_ && executeCount_ >= 1 && segments_.size() == 1 &&
        frozenFastPathInputStable && frozenFastPathSlotsStable &&
        segments_[0].exec.replayHandle != nullptr &&
        segments_[0].exec.replayHandle->isReady())) {
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

  bool ok = true;
  int syncedCount = 0;
  for (int ei = 0; ei < numExternalInputs; ei++) {
    if (externalInputs[ei] == nullptr) continue;
    externalInputs[ei]->syncToDevice();
    syncedCount++;
  }
  DSP_DIAG(EXECUTE, "NativeDSP::frozenFastPath: synced=%d external inputs", syncedCount);
  hasPendingDecodeUpdate_ = false;

  if (ok) {
    auto tCopyDone = executionTimingEnabled_ ? Clock::now() : Clock::time_point{};

#if HAVE_TRITON
    auto* tritonBackend = dynamic_cast<TritonGraphBackend*>(getGpuGraphBackend());
    if (tritonBackend != nullptr) {
      tritonBackend->refreshArgTablesForReplay(seg, externalInputs, numExternalInputs,
                                               outputSlots_, totalOutputSlots_,
                                               stream);
    }
#endif

    if (seg.exec.replayHandle->replay(stream)) {
      auto tLaunchDone = executionTimingEnabled_ ? Clock::now() : Clock::time_point{};
      for (int i = 0; i < numRequestedOutputs_; i++) {
        int slotIdx = requestedOutputSlotIndices_[i];
        if (slotIdx >= 0 && slotIdx < totalOutputSlots_) {
          requestedOutputs[i] = slotArrayCache_[slotIdx];
        } else {
          requestedOutputs[i] = nullptr;
        }
      }
      totalGraphReplays_++;
      seg.exec.executionCount++;
      executeCount_++;

      if (kvCacheRetentionEnabled_) {
        scatterKvEntries(externalInputs, numExternalInputs, stream);
        kvCachePosition_++;
      }
      auto tScatterDone = executionTimingEnabled_ ? Clock::now() : Clock::time_point{};
      if (cudaStr != nullptr) cudaStreamSynchronize(cudaStr);
      auto tSyncDone = executionTimingEnabled_ ? Clock::now() : Clock::time_point{};

      // ── REPLAY VERIFICATION (CUDA_GRAPHS path) ──────────────────────────
      if (Environment::getInstance().tritonVerifyKernels()) {
        performReplayVerify(seg, externalInputs, numExternalInputs, stream, "CUDA_GRAPHS");
      }

      // pendingClose_ removed: arrays persist (one array per slot)

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

  DSP_DIAG(FALLBACK, "NativeDSP::execute: frozen fast path failed (ok=%d), falling back to full path",
           static_cast<int>(ok));
  return Status::MAYBE;  // Fall through to full execution path
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
  // Check ONLY for replay handle — NOT compilationFailed. The compilationFailed flag means
  // the Triton path failed, but the CUDA graph fallback may have succeeded and set
  // replayHandle. During cleanup between calls, compilationFailed is still true (Fix 10
  // clears it during the NEXT execution). If we also require !compilationFailed, cleanup
  // frees the segment's slots → graph replay D2D copies read freed memory → NaN.
  if (seg.exec.replayHandle != nullptr) return true;
  return false;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Platform dispatch: Parallel precompilation
// ═══════════════════════════════════════════════════════════════════════════════

void NativeDynamicShapePlan::platformPrecompileSegments(
    NDArray** externalInputs, int numExternalInputs) {
  using Clock = std::chrono::high_resolution_clock;

  if (executeCount_ != 1 || graphExecutionMode_ == GraphExecutionMode::GEM_SLOT_BY_SLOT) {
    DSP_DIAG(COMPILE, "platformPrecompileSegments: skipped (execCount=%d mode=%d)",
             executeCount_, static_cast<int>(graphExecutionMode_));
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
        DSP_DIAG(FALLBACK, "NativeDSP::precompile: cudaSetDevice(%d) failed for segment %d: %s",
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
        segments_[task.segIdx].def.shapeKey = task.shapeKey;
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

  // Non-frozen execution: use slot-by-slot to avoid memory leaks.
  // Graph capture/replay with tl_graphExecutionActive=true suppresses cudaFreeAsync
  // in deleteSpecial(), causing temporary NDArrays created by ops during capture
  // to leak their GPU memory (~260 MB/step for decoder models). With changing shapes
  // (KV cache grows each step), each step triggers a new capture, compounding the leak.
  // Slot-by-slot execution properly frees all temporaries.
  if (!shapesFrozen_) return false;

  bool tryCapture = (segment.def.isCapturable || (shapesFrozen_ && executeCount_ > 0))
                    && !segment.exec.compilationFailed;
  // Use selectedBackend to determine if graph capture is possible — no cascade check needed.
  bool hasGraphBackend = (segment.def.selectedBackend == SelectedBackend::GPU_COMPILER ||
                          segment.def.selectedBackend == SelectedBackend::CUDA_GRAPHS);
  return tryCapture && hasGraphBackend;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Platform dispatch: Switch-based backend dispatch (no cascade)
// ═══════════════════════════════════════════════════════════════════════════════

Status NativeDynamicShapePlan::platformExecuteSegmentWithBackends(
    GraphSegment& segment, NDArray** externalInputs, int numExternalInputs,
    void* stream, bool& usedGraph) {
  usedGraph = false;

  DSP_DIAG(EXECUTE, "NativeDSP::execute: seg[%d-%d] selectedBackend=%d isCapturable=%d executionCount=%d phase=%d",
           segment.def.startSlot, segment.def.endSlot,
           static_cast<int>(segment.def.selectedBackend), static_cast<int>(segment.def.isCapturable),
           segment.exec.executionCount, static_cast<int>(segment.exec.currentPhase));

  switch (segment.def.selectedBackend) {
    case SelectedBackend::GPU_COMPILER: {
      auto* gpuBackend = getGpuGraphBackend();
      if (gpuBackend) {
        auto status = executeSegmentWithGpuGraph(segment, externalInputs, numExternalInputs, stream);
        if (status == Status::OK) {
          usedGraph = true;
          if (segment.exec.executionCount <= 1) {
            segment.exec.currentPhase = ExecutionPhase::COMPILING;
          } else if (segment.exec.replayHandle && segment.exec.replayHandle->isReady()) {
            segment.exec.currentPhase = ExecutionPhase::REPLAYING;
          } else {
            segment.exec.currentPhase = ExecutionPhase::COMPILED;
          }
          return Status::OK;
        }
        // GPU backend failed — hard error. No cascade.
        DSP_DIAG(FALLBACK, "NativeDSP::execute: exec%d seg[%d-%d] gpuBackend=%s FAILED status=%d — hard error",
                 executeCount_, segment.def.startSlot, segment.def.endSlot, gpuBackend->name(),
                 static_cast<int>(status));
        return status;
      }
      // GEM_AUTO resolved to GPU_COMPILER but no backend available at runtime.
      // Fall through to CUDA graphs if enabled, otherwise slot-by-slot.
      if (gpuGraphCaptureEnabled_) {
        goto cuda_graphs;
      }
      goto slot_by_slot;
    }

    case SelectedBackend::CUDA_GRAPHS: {
cuda_graphs:
      auto status = executeSegmentWithGraph(segment, externalInputs, numExternalInputs, stream);
      if (status != Status::OK) {
        DSP_DIAG(COMPILE, "NativeDSP::execute: CUDA graph capture FAILED for seg[%d-%d] status=%d — hard error",
                 segment.def.startSlot, segment.def.endSlot, static_cast<int>(status));
        segment.exec.compilationFailed = true;
        return status;
      }
      usedGraph = (segment.exec.replayHandle != nullptr && segment.exec.replayHandle->isReady() && !segment.exec.compilationFailed);
      if (usedGraph) {
        segment.exec.currentPhase = ExecutionPhase::REPLAYING;
      } else if (segment.exec.executionCount <= 1) {
        segment.exec.currentPhase = ExecutionPhase::COMPILING;
      } else {
        segment.exec.currentPhase = ExecutionPhase::COMPILED;
      }
      return Status::OK;
    }

    case SelectedBackend::SLOT_BY_SLOT:
    default:
slot_by_slot:
      segment.exec.currentPhase = ExecutionPhase::SLOT_BY_SLOT;
      return executeSegmentSlotBySlot(segment, externalInputs, numExternalInputs, stream);

    case SelectedBackend::CPU_GRAPH:
      // CPU graph backend not applicable on CUDA build — treat as slot-by-slot
      segment.exec.currentPhase = ExecutionPhase::SLOT_BY_SLOT;
      return executeSegmentSlotBySlot(segment, externalInputs, numExternalInputs, stream);
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Platform dispatch: Post-segment error check
// ═══════════════════════════════════════════════════════════════════════════════

Status NativeDynamicShapePlan::platformCheckPostSegment(GraphSegment& segment) {
  auto lastErr = cudaGetLastError();
  if (lastErr != cudaSuccess) {
    char buf[512];
    snprintf(buf, sizeof(buf), "CUDA error after segment [%d-%d] (execCount=%d shapesFrozen=%d): %d (%s)",
             segment.def.startSlot, segment.def.endSlot,
             executeCount_, static_cast<int>(shapesFrozen_),
             static_cast<int>(lastErr), cudaGetErrorString(lastErr));
    DSP_DIAG(FALLBACK, "NativeDynamicShapePlan: %s", buf);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(static_cast<int>(lastErr));
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(buf);
    return Status::KERNEL_FAILURE;
  }
  return Status::OK;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Platform dispatch: KV scatter
// ═══════════════════════════════════════════════════════════════════════════════

void* NativeDynamicShapePlan::platformBeginKvScatter(void* stream) {
  auto* lc = LaunchContext::defaultContext();
  if (stream != nullptr) {
    cudaStream_t* saved = lc->getCudaStream();
    lc->setCudaStream(static_cast<cudaStream_t*>(stream));
    return saved;
  }
  return nullptr;
}

void NativeDynamicShapePlan::platformEndKvScatter(void* savedState) {
  if (savedState != nullptr) {
    LaunchContext::defaultContext()->setCudaStream(static_cast<cudaStream_t*>(savedState));
  }
}

void NativeDynamicShapePlan::platformScatterKvEntry(
    NDArray* presentKv, NDArray* staticBuf, int seqDim, int pos, void* stream) {
  auto* lc = LaunchContext::defaultContext();
  ops::helpers::kvScatter(presentKv, staticBuf, pos, lc);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Platform dispatch: KV replay annotation
// ═══════════════════════════════════════════════════════════════════════════════

void NativeDynamicShapePlan::platformMarkKvCaptureBuffersNeverSkip() {
  // Capture-buffer staging has been removed. KV cache inputs are replayed
  // directly from their canonical buffers, so there is nothing to annotate.
}

// ═══════════════════════════════════════════════════════════════════════════════
// Platform dispatch: Segment cleanup for rebuild
// ═══════════════════════════════════════════════════════════════════════════════

void NativeDynamicShapePlan::platformCleanupSegmentForRebuild(GraphSegment& seg) {
  DSP_DIAG_SEG(GRAPH_REPLAY, seg.def.startSlot,
               "platformCleanupSegmentForRebuild: seg[%d-%d] hasReplay=%d",
               seg.def.startSlot, seg.def.endSlot, seg.exec.replayHandle ? 1 : 0);
  if (seg.exec.replayHandle) {
    if (seg.exec.replayHandle->getWorkspacePtr() != nullptr) {
      seg.exec.replayHandle->releaseWorkspace(nullptr, seg.def.startSlot);
    }
    seg.exec.replayHandle->freeHostPointers();
    seg.exec.replayHandle->clearExternalAddresses();
    seg.exec.replayHandle.reset();
  }
  seg.exec.gapOpsCapturedInGraph = false;
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

  // Free batch-zero resources
  freeBatchZeroResources();

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
    if (seg.exec.replayHandle && seg.exec.replayHandle->isReady()) count++;
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
    DSP_DIAG(FALLBACK, "%d host-only ops in captured graph - outputs stale on replay",
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
  // Create DspStreamGuard (heap-allocated so it lives for full execute() scope)
  struct ExecutionState {
    std::unique_ptr<sd::graph::DspStreamGuard> streamGuard;
  };
  auto* state = new ExecutionState();
  if (stream != nullptr) {
    state->streamGuard = std::make_unique<sd::graph::DspStreamGuard>(
        *static_cast<cudaStream_t*>(stream));
  }

  // Stream ordering: ensure all async CUDA operations from Java complete before DSP execution.
  if (stream != nullptr) {
    cudaStream_t cudaStr = *static_cast<cudaStream_t*>(stream);
    // Check for prior CUDA errors before attempting sync
    auto priorErr = cudaGetLastError();
    if (priorErr != cudaSuccess) {
      DSP_DIAG(EXECUTE, "platformBeginExecution: PRIOR CUDA ERROR before sync: %s (%d)",
               cudaGetErrorString(priorErr), static_cast<int>(priorErr));
    }
    DSP_DIAG(EXECUTE, "platformBeginExecution: frozen=%d execCount=%d stream=%p syncing...",
             static_cast<int>(frozen), execCount, static_cast<void*>(cudaStr));
    if (!frozen || execCount <= 1) {
      auto syncErr = cudaStreamSynchronize(cudaStr);
      DSP_DIAG(EXECUTE, "platformBeginExecution: cudaStreamSynchronize returned %d (%s)",
               static_cast<int>(syncErr), cudaGetErrorString(syncErr));
    } else {
      cudaEvent_t defaultStreamEvent;
      cudaEventCreateWithFlags(&defaultStreamEvent, cudaEventDisableTiming);
      cudaEventRecord(defaultStreamEvent, nullptr);
      cudaStreamWaitEvent(cudaStr, defaultStreamEvent, 0);
      cudaEventDestroy(defaultStreamEvent);
    }
  }

  return static_cast<void*>(state);
}

void NativeDynamicShapePlan::platformEndExecution(void* executionState, void* stream, bool frozen, int execCount) {
  // Cross-stream synchronization
  if (stream != nullptr) {
    cudaStream_t cudaStr = *static_cast<cudaStream_t*>(stream);
    DSP_DIAG(EXECUTE, "platformEndExecution: frozen=%d execCount=%d stream=%p syncing...",
             static_cast<int>(frozen), execCount, static_cast<void*>(cudaStr));
    if (frozen && execCount > 1) {
      if (executionCompleteEvent_ == nullptr) {
        cudaEvent_t evt;
        cudaEventCreateWithFlags(&evt, cudaEventDisableTiming);
        executionCompleteEvent_ = static_cast<void*>(new cudaEvent_t(evt));
      }
      cudaEvent_t evt = *static_cast<cudaEvent_t*>(executionCompleteEvent_);
      cudaEventRecord(evt, cudaStr);
      cudaStreamWaitEvent(nullptr, evt, 0);
    } else {
      auto syncErr = cudaStreamSynchronize(cudaStr);
      DSP_DIAG(EXECUTE, "platformEndExecution: cudaStreamSynchronize returned %d (%s)",
               static_cast<int>(syncErr), cudaGetErrorString(syncErr));
    }
  }

  // Free the execution state (DspStreamGuard destroyed here via unique_ptr)
  struct ExecutionState {
    std::unique_ptr<sd::graph::DspStreamGuard> streamGuard;
  };
  delete static_cast<ExecutionState*>(executionState);
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
  if (!DSP_DIAG_ENABLED(VERIFY) || execCount > 4) return;
  for (int i = 0; i < numRequestedOutputs_; i++) {
    int slotIdx = requestedOutputSlotIndices_[i];
    NDArray* arr = (slotIdx >= 0 && slotIdx < totalOutputSlots_) ? outputSlots_[slotIdx] : nullptr;
    if (arr == nullptr) continue;
    void* sbuf = arr->specialBuffer();
    if (sbuf && arr->dataType() == FLOAT32 && arr->lengthOf() >= 49280) {
      auto len = arr->lengthOf();
      std::vector<float> fullBuf(len);
      cudaMemcpy(fullBuf.data(), sbuf, len * sizeof(float), cudaMemcpyDeviceToHost);
      float maxVal = -1e30f;
      int maxIdx = -1;
      for (int j = 0; j < (int)len; j++) {
        if (fullBuf[j] > maxVal) { maxVal = fullBuf[j]; maxIdx = j; }
      }
      DSP_DIAG_SLOT(VERIFY, slotIdx,
          "logits maxIdx=%d maxVal=%.4f v@44=%.4f v@15539=%.4f",
          maxIdx, maxVal, fullBuf[44], fullBuf[15539]);
    }
  }
}

void NativeDynamicShapePlan::platformDetectAndPrepareBatchedGemm(NDArray** ext, int numExt, void* stream) {
  if (shapesFrozen_ && executeCount_ == 1 && batchedGemmGroups_.empty() &&
      Environment::getInstance().dspBatchedGemm() &&
      !gpuGraphCaptureEnabled_) {
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

void NativeDynamicShapePlan::platformUpdateDecodeInputs(NDArray** ext, int numExt,
                                                         long long tokenId, int cachePos, void* stream) {
  cudaStream_t cudaStr = stream ? *static_cast<cudaStream_t*>(stream) : static_cast<cudaStream_t>(nullptr);

  DSP_DIAG(EXECUTE, "updateDecodeInputs: ENTER tokenId=%lld cachePos=%d numExt=%d idsIdx=%d posIdx=%d maskIdx=%d",
           tokenId, cachePos, numExt, decodeInputIdsExtIdx_, decodePositionIdsExtIdx_, decodeAttentionMaskExtIdx_);

  // input_ids[0] = tokenId
  if (decodeInputIdsExtIdx_ >= 0 && decodeInputIdsExtIdx_ < numExt) {
    NDArray* ids = ext[decodeInputIdsExtIdx_];
    DSP_DIAG(EXECUTE, "updateDecodeInputs: ids NDArray=%p specialBuf=%p len=%lld",
             ids, ids ? ids->specialBuffer() : nullptr, ids ? (long long)ids->lengthOf() : -1);
    if (ids != nullptr && ids->specialBuffer() != nullptr) {
      LongType val = static_cast<LongType>(tokenId);
      cudaMemcpyAsync(ids->specialBuffer(), &val, sizeof(LongType),
                      cudaMemcpyHostToDevice, cudaStr);
      ids->dataBuffer()->writeSpecial();
    }
  }

  // position_ids[0] = cachePos
  if (decodePositionIdsExtIdx_ >= 0 && decodePositionIdsExtIdx_ < numExt) {
    NDArray* pos = ext[decodePositionIdsExtIdx_];
    DSP_DIAG(EXECUTE, "updateDecodeInputs: pos NDArray=%p specialBuf=%p len=%lld",
             pos, pos ? pos->specialBuffer() : nullptr, pos ? (long long)pos->lengthOf() : -1);
    if (pos != nullptr && pos->specialBuffer() != nullptr) {
      LongType val = static_cast<LongType>(cachePos);
      cudaMemcpyAsync(pos->specialBuffer(), &val, sizeof(LongType),
                      cudaMemcpyHostToDevice, cudaStr);
      pos->dataBuffer()->writeSpecial();
    }
  }

  // attention_mask[cachePos - 1] = 1
  if (decodeAttentionMaskExtIdx_ >= 0 && decodeAttentionMaskExtIdx_ < numExt && cachePos > 0) {
    NDArray* mask = ext[decodeAttentionMaskExtIdx_];
    int writePos = cachePos - 1;
    DSP_DIAG(EXECUTE, "updateDecodeInputs: mask NDArray=%p specialBuf=%p len=%lld cachePos=%d writePos=%d",
             mask, mask ? mask->specialBuffer() : nullptr, mask ? (long long)mask->lengthOf() : -1,
             cachePos, writePos);
    if (mask != nullptr && mask->specialBuffer() != nullptr) {
      LongType one = 1;
      auto maskLen = mask->lengthOf();
      if (writePos < maskLen) {
        auto* dst = static_cast<LongType*>(mask->specialBuffer()) + writePos;
        DSP_DIAG(EXECUTE, "updateDecodeInputs: mask dst=%p (base=%p + %d * %d)",
                 dst, mask->specialBuffer(), writePos, (int)sizeof(LongType));
        cudaMemcpyAsync(dst, &one, sizeof(LongType),
                        cudaMemcpyHostToDevice, cudaStr);
        mask->dataBuffer()->writeSpecial();
      } else {
        DSP_DIAG(EXECUTE, "updateDecodeInputs: SKIP attn_mask write writePos=%d maskLen=%lld (OOB)",
                 writePos, (long long)maskLen);
      }
    }
  }

  DSP_DIAG(EXECUTE, "updateDecodeInputs: tokenId=%lld cachePos=%d", tokenId, cachePos);
}

void NativeDynamicShapePlan::platformPostKvScatterSync(int scattered, int pos, int numMappings) {
  cudaError_t scatterErr = cudaDeviceSynchronize();
  if (scatterErr != cudaSuccess) {
    DSP_DIAG(EXECUTE,
        "KV SCATTER CUDA ERROR: cudaDeviceSynchronize after kvScatterBatched "
        "returned error %d (%s). scattered=%d pos=%d numMappings=%d",
        static_cast<int>(scatterErr), cudaGetErrorString(scatterErr),
        scattered, pos, numMappings);
    cudaGetLastError();
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
    seg.exec.currentPhase = ExecutionPhase::WARMUP;
    seg.exec.jitShapeKey = 0;
    seg.exec.jitCompileFailed = false;
    seg.exec.segBatchZeroEntries.clear();
    seg.def.shapeKey = 0;
  }
  logGpuMemState("STEP-1-AFTER-SEGMENTS");

  // Free cuBLAS workspace
  if (cublasWorkspaceBuffer_ != nullptr) {
    cudaFree(cublasWorkspaceBuffer_);
    cublasWorkspaceBuffer_ = nullptr;
    cublasWorkspaceSize_ = 0;
  }

  // Free batch-zero, batch-D2D, and batched-GEMM device arrays
  freeBatchZeroResources();
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

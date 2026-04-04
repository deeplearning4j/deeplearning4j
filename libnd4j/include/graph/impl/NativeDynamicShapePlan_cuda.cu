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
#include <graph/DspVerifyUtils.h>
#include <graph/cuda/CudaGraphReplayHandle.h>
#include <helpers/ConstantShapeHelper.h>
#include <helpers/MmulHelper.h>
#include <memory/cuda/CudaMemoryPool.h>
#include <helpers/AttentionWorkspace.h>
#include <graph/gpu/NvrtcKernelBuilder.h>
#include <graph/gpu/NvrtcKernelCache.h>
#include <ops/declarable/helpers/kv_scatter.h>
#include <system/Environment.h>

#include <graph/gpu/CaptureBufferRegistry.h>

#if HAVE_TRITON
#include <graph/gpu/TritonGraphBackend.h>
#include <graph/gpu/OpCategoryTable.h>
#endif

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <future>
#include <numeric>
#include <unordered_set>

namespace sd {
namespace graph {

namespace {

bool isStrictNoFallbackMode(GraphExecutionMode mode) {
  return mode == GraphExecutionMode::GEM_TRITON;
}

bool bindSegmentCudaDevice(const GraphSegment& segment,
                           NativeSlot* slots,
                           int numSlots,
                           const char* phase) {
  int targetDevice = -1;
  if (segment.startSlot >= 0 && segment.startSlot < numSlots) {
    targetDevice = slots[segment.startSlot].targetDeviceId;
  }
  if (targetDevice < 0) return true;

  int deviceCount = 0;
  cudaError_t countErr = cudaGetDeviceCount(&deviceCount);
  if (countErr != cudaSuccess || deviceCount <= 0) {
    DSP_DIAG(BACKEND, "NativeDSP::execute: %s seg[%d-%d] targetDeviceId=%d but CUDA device query failed: %s",
             phase, segment.startSlot, segment.endSlot, targetDevice,
             cudaGetErrorString(countErr));
    cudaGetLastError();
    return false;
  }
  if (targetDevice >= deviceCount) {
    DSP_DIAG(BACKEND, "NativeDSP::execute: %s seg[%d-%d] invalid targetDeviceId=%d (deviceCount=%d)",
             phase, segment.startSlot, segment.endSlot, targetDevice, deviceCount);
    return false;
  }

  int currentDevice = -1;
  cudaError_t getErr = cudaGetDevice(&currentDevice);
  if (getErr != cudaSuccess) {
    DSP_DIAG(BACKEND, "NativeDSP::execute: %s seg[%d-%d] failed to query current CUDA device: %s",
             phase, segment.startSlot, segment.endSlot, cudaGetErrorString(getErr));
    cudaGetLastError();
    return false;
  }

  if (currentDevice != targetDevice) {
    cudaError_t setErr = cudaSetDevice(targetDevice);
    if (setErr != cudaSuccess) {
      DSP_DIAG(BACKEND, "NativeDSP::execute: %s seg[%d-%d] failed to switch CUDA device %d->%d: %s",
               phase, segment.startSlot, segment.endSlot,
               currentDevice, targetDevice, cudaGetErrorString(setErr));
      cudaGetLastError();
      return false;
    }
    DSP_DIAG(BACKEND, "NativeDSP::execute: %s seg[%d-%d] switched CUDA device %d->%d",
             phase, segment.startSlot, segment.endSlot, currentDevice, targetDevice);
  } else {
    DSP_DIAG(BACKEND, "NativeDSP::execute: %s seg[%d-%d] using CUDA device %d",
             phase, segment.startSlot, segment.endSlot, currentDevice);
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
  if (allowFrozenGraphFastPath && shapesFrozen_ && executeCount_ >= 1 && segments_.size() == 1) {
    auto& seg0 = segments_[0];
    if (!seg0.exec.replayHandle || seg0.exec.replayHandle->getCaptureBuffers().empty()) {
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
    }
  }
  if (!(allowFrozenGraphFastPath && shapesFrozen_ && executeCount_ >= 1 && segments_.size() == 1 &&
        frozenFastPathInputStable && segments_[0].exec.replayHandle != nullptr &&
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

  // Copy external inputs into fixed-address capture buffers.
  bool ok = true;
  int copiedCount = 0;
  int skippedCount = 0;
  for (auto& cb : seg.exec.replayHandle->getCaptureBuffers()) {
    if (cb.directReference) {
      skippedCount++;
      continue;
    }

    // Skip decode inputs (position_ids, attention_mask, input_ids) when C++ manages them.
    // The regular D2D/hostMirror copy would read stale HOST values (because Java doesn't
    // update host when nativeDecodeInputs=true). Instead, these are written directly to
    // capture buffers in the decode-input block below.
    if (hasPendingDecodeUpdate_ && isDecodeInputsConfigured()) {
      int ei = cb.externalInputIndex;
      if (ei >= 0 && (ei == decodeInputIdsExtIdx_ || ei == decodePositionIdsExtIdx_
                      || ei == decodeAttentionMaskExtIdx_)) {
        skippedCount++;
        continue;
      }
    }

    NDArray* src = nullptr;
    if (cb.externalInputIndex >= 0 && cb.externalInputIndex < numExternalInputs) {
      src = externalInputs[cb.externalInputIndex];
    } else if (cb.crossSegmentSlotIdx >= 0 && cb.crossSegmentSlotIdx < totalOutputSlots_) {
      src = slotArrayCache_[cb.crossSegmentSlotIdx];
    }
    if (src == nullptr || cb.buffer == nullptr) { ok = false; break; }

    size_t srcBytes = src->lengthOf() * src->sizeOfT();
    if (srcBytes != cb.capturedSize) { ok = false; break; }

    if (srcBytes > 0) {
      const void* currentPtr = src->specialBuffer();
      // NOTE: Never skip D2D copies based on pointer comparison.
      // GPU memory pools reuse addresses — a freed buffer's address can be
      // returned for a new allocation with completely different data.
      // Always copy to avoid alternating-stale-data bugs.

      auto dt = src->dataType();
      bool hostMirror = (dt == INT32 || dt == INT64 || dt == BOOL)
                        && src->lengthOf() > 0 && src->lengthOf() <= 32;
      if (hostMirror) {
        src->syncToHost();
        std::memcpy(cb.buffer->buffer(), src->buffer(), srcBytes);
        cb.buffer->tickWriteHost();
        cb.buffer->syncToDevice();
      } else {
        src->syncToDevice();
        auto copyErr = cudaMemcpyAsync(cb.buffer->specialBuffer(), src->specialBuffer(),
                                       srcBytes, cudaMemcpyDeviceToDevice, cudaStr);
        if (copyErr != cudaSuccess) { cudaGetLastError(); ok = false; break; }
        // Mark device as actual after D2D copy — prevents syncToSpecial()
        // from recording a stale H2D that overwrites fresh data on replay.
        cb.buffer->dataBuffer()->writeSpecial();
      }
      cb.lastSourcePtr = currentPtr;
      cb.initialCopyDone = true;
      copiedCount++;
    }
  }
  DSP_DIAG(EXECUTE, "NativeDSP::frozenFastPath: copied=%d skipped=%d total=%d",
           copiedCount, skippedCount, copiedCount + skippedCount);

  // Apply pending decode input updates directly to capture buffers.
  // updateDecodeInputs() writes to external input arrays, but the graph reads
  // from capture buffers (separate fixed-address copies). By writing to
  // capture buffers here (after D2D copies, before graph launch), the graph
  // sees the correct position_ids, attention_mask, and input_ids values.
  if (ok && hasPendingDecodeUpdate_ && isDecodeInputsConfigured()) {
    for (auto& cb : seg.exec.replayHandle->getCaptureBuffers()) {
      if (cb.directReference || cb.buffer == nullptr) continue;
      int ei = cb.externalInputIndex;
      if (ei < 0) continue;

      // input_ids capture buffer: write pendingTokenId_
      if (ei == decodeInputIdsExtIdx_ && cb.buffer->specialBuffer() != nullptr) {
        LongType val = static_cast<LongType>(pendingTokenId_);
        cudaMemcpyAsync(cb.buffer->specialBuffer(), &val, sizeof(LongType),
                        cudaMemcpyHostToDevice, cudaStr);
        DSP_DIAG(EXECUTE, "frozenFastPath: wrote input_ids=%lld to capture buffer (extIdx=%d)",
                 pendingTokenId_, ei);
      }
      // position_ids capture buffer: write pendingCachePos_
      else if (ei == decodePositionIdsExtIdx_ && cb.buffer->specialBuffer() != nullptr) {
        LongType val = static_cast<LongType>(pendingCachePos_);
        cudaMemcpyAsync(cb.buffer->specialBuffer(), &val, sizeof(LongType),
                        cudaMemcpyHostToDevice, cudaStr);
        DSP_DIAG(EXECUTE, "frozenFastPath: wrote position_ids=%d to capture buffer (extIdx=%d)",
                 pendingCachePos_, ei);
      }
      // attention_mask capture buffer: write 1 at cachePos - 1
      // cachePos is the NEXT write position — not yet filled. The position just filled
      // is cachePos - 1. This must match updateDecodeInputs() which writes at cachePos - 1.
      else if (ei == decodeAttentionMaskExtIdx_ && cb.buffer->specialBuffer() != nullptr) {
        int writePos = pendingCachePos_ - 1;
        auto maskLen = cb.buffer->lengthOf();
        if (writePos >= 0 && writePos < maskLen) {
          LongType one = 1;
          auto* dst = static_cast<LongType*>(cb.buffer->specialBuffer()) + writePos;
          cudaMemcpyAsync(dst, &one, sizeof(LongType),
                          cudaMemcpyHostToDevice, cudaStr);
          DSP_DIAG(EXECUTE, "frozenFastPath: wrote attention_mask[%d]=1 to capture buffer (extIdx=%d)",
                   writePos, ei);
        }
      }
    }
    hasPendingDecodeUpdate_ = false;
  }

  if (ok) {
    auto tCopyDone = executionTimingEnabled_ ? Clock::now() : Clock::time_point{};

#if HAVE_TRITON
    if (seg.exec.replayHandle->getCaptureBuffers().empty()) {
      for (int ei = 0; ei < numExternalInputs; ei++) {
        if (externalInputs[ei] != nullptr) {
          externalInputs[ei]->syncToDevice();
        }
      }
      auto* tritonBackend = dynamic_cast<TritonGraphBackend*>(getGpuGraphBackend());
      if (tritonBackend != nullptr) {
        tritonBackend->refreshArgTablesForReplay(seg, externalInputs, numExternalInputs,
                                                 outputSlots_, totalOutputSlots_,
                                                 stream);
      }
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
                 copyUs, launchUs, scatterUs, syncUs, totalUs, copiedCount, skippedCount);
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

  if (executeCount_ != 1 || graphExecutionMode_ == GraphExecutionMode::GEM_SLOT_BY_SLOT) return;

  auto* gpuBackend = getGpuGraphBackend();
  if (gpuBackend == nullptr) return;

  struct PrecompileTask {
    int segIdx;
    LongType shapeKey;
    int targetDevice;
  };
  std::vector<PrecompileTask> tasks;
  for (int si = 0; si < static_cast<int>(segments_.size()); si++) {
    auto& seg = segments_[si];
    if (seg.exec.compilationFailed) continue;
    bool tryCapture = seg.isCapturable || (shapesFrozen_ && executeCount_ > 0);
    if (!tryCapture) continue;
    if (!gpuBackend->canFuseSegment(slots_, seg.startSlot, seg.endSlot)) continue;
    LongType segShapeKey = computeSegmentShapeKey(seg, externalInputs, numExternalInputs);
    int segTargetDevice = 0;
    if (seg.startSlot >= 0 && seg.startSlot < numSlots_) {
      segTargetDevice = slots_[seg.startSlot].targetDeviceId;
      if (segTargetDevice < 0) segTargetDevice = 0;
    }
    tasks.push_back({si, segShapeKey, segTargetDevice});
  }

  if (tasks.size() <= 1) return;

  const int maxPrecompileThreads = std::min(
      static_cast<int>(tasks.size()),
      std::max(1, sd::Environment::getInstance().tritonBuildThreads()));
  DSP_DIAG(COMPILE, "NativeDSP::execute: parallel precompilation of %d segments "
           "using %d threads (executeCount=%d)",
           static_cast<int>(tasks.size()), maxPrecompileThreads, executeCount_);
  auto precompileStart = Clock::now();

  // Force-initialize static singleton tables on the main thread BEFORE
  // launching parallel workers.  getOpCategoryTable() is an inline function
  // in a header with a static local variable.  Although C++11 "magic statics"
  // guarantee thread-safe init, NVCC's handling of inline functions with
  // static locals across multiple .cu translation units can violate this.
  // Touching it here makes the race impossible.
  (void)sd::graph::getOpCategoryTable();

  std::vector<std::future<bool>> futures;
  futures.reserve(tasks.size());
  for (const auto& task : tasks) {
    futures.emplace_back(std::async(std::launch::async,
        [this, gpuBackend, externalInputs, numExternalInputs, task]() -> bool {
          cudaError_t setDevErr = cudaSetDevice(task.targetDevice);
          if (setDevErr != cudaSuccess) {
            DSP_DIAG(FALLBACK, "NativeDSP::precompile: cudaSetDevice(%d) failed for segment %d: %s",
                     task.targetDevice, task.segIdx, cudaGetErrorString(setDevErr));
            cudaGetLastError();
            return false;
          }
          auto& seg = segments_[task.segIdx];
          return gpuBackend->compileSegment(seg, slots_, externalInputs, numExternalInputs,
                                            outputSlots_, totalOutputSlots_, task.shapeKey,
                                            numSlots_);
        }));
  }

  int precompileOk = 0, precompileFail = 0;
  for (size_t i = 0; i < futures.size(); i++) {
    bool ok = futures[i].get();
    if (ok) {
      segments_[tasks[i].segIdx].shapeKey = tasks[i].shapeKey;
      precompileOk++;
    } else {
      precompileFail++;
    }
  }

  auto precompileMs = std::chrono::duration_cast<std::chrono::milliseconds>(
      Clock::now() - precompileStart).count();
  DSP_DIAG(COMPILE, "NativeDSP::execute: parallel precompilation done in %lld ms "
           "(ok=%d, failed=%d)",
           static_cast<long long>(precompileMs), precompileOk, precompileFail);

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
  if (seg.startSlot >= 0 && seg.startSlot < numSlots_) {
    targetDevice = slots_[seg.startSlot].targetDeviceId;
  }
  if (targetDevice < 0) return;  // Auto device — no migration needed

  migratedInputs_.clear();

  // Collect unique input slot indices that this segment reads from prior segments
  std::unordered_set<int> neededInputSlots;
  for (int s = seg.startSlot; s <= seg.endSlot && s < numSlots_; s++) {
    const NativeSlot& slot = slots_[s];
    for (int i = 0; i < slot.numInputs; i++) {
      int srcIdx = slot.inputSourceIndices[i];
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
      for (int o = 0; o < srcSlot.numOutputs; o++) {
        if (srcSlot.outputSlotIndices[o] == slotIdx) {
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
             migrated, targetDevice, seg.startSlot, seg.endSlot);
  }
}

void NativeDynamicShapePlan::platformCleanupMigratedInputs() {
  // Restore original arrays in outputSlots_ and delete migrated copies
  for (auto& mi : migratedInputs_) {
    if (mi.outputSlotIdx >= 0 && mi.outputSlotIdx < totalOutputSlots_) {
      outputSlots_[mi.outputSlotIdx] = mi.original;
    }
    delete mi.migrated;
  }
  migratedInputs_.clear();
}

// ═══════════════════════════════════════════════════════════════════════════════
// Platform dispatch: Graph eligibility check
// ═══════════════════════════════════════════════════════════════════════════════

bool NativeDynamicShapePlan::platformShouldUseGraph(const GraphSegment& segment) {
  if (graphExecutionMode_ == GraphExecutionMode::GEM_SLOT_BY_SLOT) return false;

  // Non-frozen execution: use slot-by-slot to avoid memory leaks.
  // Graph capture/replay with tl_graphExecutionActive=true suppresses cudaFreeAsync
  // in deleteSpecial(), causing temporary NDArrays created by ops during capture
  // to leak their GPU memory (~260 MB/step for decoder models). With changing shapes
  // (KV cache grows each step), each step triggers a new capture, compounding the leak.
  // Slot-by-slot execution properly frees all temporaries.
  if (!shapesFrozen_) return false;

  bool tryCapture = (segment.isCapturable || (shapesFrozen_ && executeCount_ > 0))
                    && !segment.exec.compilationFailed;
  // Use selectedBackend to determine if graph capture is possible — no cascade check needed.
  bool hasGraphBackend = (segment.selectedBackend == SelectedBackend::GPU_COMPILER ||
                          segment.selectedBackend == SelectedBackend::CUDA_GRAPHS);
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
           segment.startSlot, segment.endSlot,
           static_cast<int>(segment.selectedBackend), static_cast<int>(segment.isCapturable),
           segment.exec.executionCount, static_cast<int>(segment.exec.currentPhase));

  switch (segment.selectedBackend) {
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
                 executeCount_, segment.startSlot, segment.endSlot, gpuBackend->name(),
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
                 segment.startSlot, segment.endSlot, static_cast<int>(status));
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
             segment.startSlot, segment.endSlot,
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
// Platform dispatch: KV capture buffer annotation
// ═══════════════════════════════════════════════════════════════════════════════

void NativeDynamicShapePlan::platformMarkKvCaptureBuffersNeverSkip() {
  for (auto& seg : segments_) {
    if (!seg.exec.replayHandle) continue;
    for (auto& cb : seg.exec.replayHandle->getCaptureBuffers()) {
      for (int i = 0; i < kvCacheNumMappings_; i++) {
        if (cb.externalInputIndex == kvCacheMappings_[i].pastInputExternalIdx) {
          cb.neverSkipCopy = true;
          break;
        }
      }
    }
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Platform dispatch: Segment cleanup for rebuild
// ═══════════════════════════════════════════════════════════════════════════════

void NativeDynamicShapePlan::platformCleanupSegmentForRebuild(GraphSegment& seg) {
  if (seg.exec.replayHandle) {
    // Free capture buffer NDArrays before destroying the handle
    for (auto& cb : seg.exec.replayHandle->getCaptureBuffers()) {
      if (!cb.directReference) delete cb.buffer;
    }
    seg.exec.replayHandle->getCaptureBuffers().clear();
    // Free capture workspace
    {
      bool usePool = Environment::getInstance().dspCapturePoolEnabled() &&
                     captureBufferRegistry_ != nullptr;
      seg.exec.replayHandle->releaseWorkspace(
          usePool ? captureBufferRegistry_ : nullptr,
          seg.startSlot);
    }
    // Free pinned host pointers
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
      segRanges.emplace_back(seg.startSlot, seg.endSlot);
    }
    if (!segRanges.empty()) {
      TritonGraphBackend::getInstance().invalidateCacheForSegments(segRanges);
    }
  }
#endif

  bool usePool = Environment::getInstance().dspCapturePoolEnabled() &&
                 captureBufferRegistry_ != nullptr;

  // Free capture buffers, workspace, and JIT kernels from all segments
  for (auto& seg : segments_) {
    if (seg.exec.replayHandle) {
      for (auto& cb : seg.exec.replayHandle->getCaptureBuffers()) {
        if (!cb.directReference) delete cb.buffer;
      }
      seg.exec.replayHandle->getCaptureBuffers().clear();
      if (seg.exec.replayHandle->getWorkspacePtr() != nullptr) {
        seg.exec.replayHandle->releaseWorkspace(
            usePool ? captureBufferRegistry_ : nullptr,
            seg.startSlot);
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

  // Release all pool-managed capture buffers at once
  if (usePool) {
    auto* registry = static_cast<CaptureBufferRegistry*>(captureBufferRegistry_);
    registry->releaseAll();
    delete registry;
    captureBufferRegistry_ = nullptr;
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

}  // namespace graph
}  // namespace sd

#endif  // SD_CUDA

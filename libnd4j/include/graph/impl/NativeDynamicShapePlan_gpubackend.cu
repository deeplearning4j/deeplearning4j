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

// NativeDynamicShapePlan_gpubackend.cu
//
// CUDA-only implementation for DSP GPU backend execution:
//   - CUDA graph capture, replay, composite replay
//   - External input sync, cross-stream sync
//   - Segment dispatch: replay → capture → direct execution
//   - LRU graph eviction and proactive memory cleanup
//
// This translation unit is compiled only when SD_CUDA is defined.
// CPU-facing dispatch (getGpuGraphBackend, segDispatchWarmup, segDispatchCompile,
// hasCompositeHandles, cleanupSegmentForRebuild) remains in _gpubackend.cpp.

#ifdef SD_CUDA

#include <graph/NativeDynamicShapePlan.h>
#include <graph/PlanExecutionContext.h>
#include <graph/DspDiagnostics.h>
#include <graph/DspConstants.h>
#include <graph/DspHashUtils.h>
#include <graph/DspStreamGuard.h>
#include <graph/DspVerifyUtils.h>
#include <graph/DspAnalysisUtils.h>
#include <graph/DspSegmentLifecycle.h>
#include <graph/gpu/ViewRecipe.h>
#include <graph/gpu/OpCategoryTable.h>
#include <graph/gpu/NvrtcGraphBackend.h>
#include <graph/gpu/PtxGraphBackend.h>
#include <graph/cuda/CudaGraphReplayHandle.h>
#include <memory/cuda/CudaMemoryPool.h>
#include <helpers/ShapeBuilders.h>
#include <helpers/MmulHelper.h>
#include <ops/OpTraitTable.h>
#include <system/op_boilerplate.h>
#include <system/Environment.h>
#include <config.h>

#if HAVE_TRITON
#include <graph/gpu/TritonGraphBackend.h>
#endif

#include <algorithm>
#include <chrono>
#include <thread>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// DSP gap-stream override — defined in LaunchContext.cu (file scope, no namespace).
extern thread_local cudaStream_t tl_dspGapStream;
// cuBLAS workspace thread-locals — defined in MmulHelper.cu.
extern SD_TLS_EXPORT thread_local void*  tl_cublasWorkspacePtr;
extern SD_TLS_EXPORT thread_local size_t tl_cublasWorkspaceSize;

// Portable buffer accessor (CUDA form).
#define DSP_BUF(arr) ((arr)->specialBuffer())

// ── Isolation flags for debugging composite replay accuracy ──
static bool dsp_disable_view_fastpath() {
  static bool v = (std::getenv("ND4J_DSP_DISABLE_VIEW_FASTPATH") != nullptr);
  return v;
}
static bool dsp_disable_cast_hwm() {
  static bool v = (std::getenv("ND4J_DSP_DISABLE_CAST_HWM") != nullptr);
  return v;
}
static bool dsp_disable_workspace_skip() {
  static bool v = (std::getenv("ND4J_DSP_DISABLE_WS_SKIP") != nullptr);
  return v;
}

namespace sd {
namespace graph {

// File-level alias for the nested enum — avoids GraphSegmentExec:: prefix at call sites.
using SegmentLifecycleState = GraphSegmentExec::SegmentLifecycleState;

// SegmentLifecycle transition functions are defined as static inline in
// <graph/DspSegmentLifecycle.h>. Bring them into scope without the namespace prefix.
using namespace SegmentLifecycle;

// ── Cross-stream sync helper ──────────────────────────────────────────────
static void syncCrossStream(cudaStream_t dspStream, cudaEvent_t syncEvent,
                            const char* caller, int segStart, int segEnd, int execCount) {
  cudaStream_t defaultStream = nullptr;
  auto* defaultStreamPtr = LaunchContext::defaultContext()->getCudaStream();
  if (defaultStreamPtr != nullptr) defaultStream = *defaultStreamPtr;
  if (defaultStream != nullptr && defaultStream != dspStream && syncEvent != nullptr) {
    cudaEventRecord(syncEvent, defaultStream);
    cudaStreamWaitEvent(dspStream, syncEvent, 0);
    DSP_DIAG(STREAM_SYNC,
             "%s cross-stream sync: recordedOn=defaultStream=%p waitedOn=dspStream=%p "
             "seg=[%d-%d] execCount=%d",
             caller, (void*)defaultStream, (void*)dspStream,
             segStart, segEnd, execCount);
  }
}

// ── External input sync helper ────────────────────────────────────────────
static void syncExternalInputs(NDArray** externalArrays, int numExt,
                               const std::vector<bool>& externalInputIsVariable,
                               const std::vector<std::string>& externalInputNames,
                               bool shapesFrozen, bool tritonVerify,
                               const char* tag, int execCount,
                               int* outSynced = nullptr, int* outSkipped = nullptr) {
  int synced = 0, skipped = 0;
  bool useVariableFilter = shapesFrozen && !externalInputIsVariable.empty();

  // Assertion 2: External input classification audit — one-time summary at execCount==1.
  // Logs which inputs are variable vs weight when variable filtering is active.
  // Misclassification (e.g. a KV-cache placeholder marked as weight) is immediately
  // visible here: it will appear as "weight/SKIP" instead of "variable/SYNC".
  if (useVariableFilter && execCount == 1 &&
      DspDiagnostics::getInstance().isEnabled(DSP_DIAG_EXECUTE)) {
    DSP_DIAG(EXECUTE,
             "EXT_INPUT_CLASSIFICATION_AUDIT: tag=%s numExt=%d (shapesFrozen, useVariableFilter)",
             tag ? tag : "?", numExt);
    for (int ei = 0; ei < numExt; ei++) {
      if (externalArrays[ei] == nullptr) continue;
      bool isVar = (ei < static_cast<int>(externalInputIsVariable.size()))
                   ? externalInputIsVariable[ei] : false;
      const char* name = (ei < static_cast<int>(externalInputNames.size()))
                         ? externalInputNames[ei].c_str() : "?";
      auto* db = externalArrays[ei]->dataBuffer();
      DSP_DIAG(EXECUTE,
               "  EXT[%d] name='%s' class=%s action=%s pAct=%d sAct=%d len=%lld addr=%p",
               ei, name,
               isVar ? "variable" : "weight",
               isVar ? "SYNC" : "SKIP",
               db ? (db->isPrimaryActual() ? 1 : 0) : -1,
               db ? (db->isSpecialActual() ? 1 : 0) : -1,
               (long long)externalArrays[ei]->lengthOf(),
               DSP_BUF(externalArrays[ei]));
    }
  }

  for (int ei = 0; ei < numExt; ei++) {
    if (externalArrays[ei] == nullptr) continue;
    bool isVariable = useVariableFilter &&
                      ei < static_cast<int>(externalInputIsVariable.size()) &&
                      externalInputIsVariable[ei];
    bool isWeight = useVariableFilter &&
                    ei < static_cast<int>(externalInputIsVariable.size()) &&
                    !externalInputIsVariable[ei];

    if (tritonVerify) {
      auto* db = externalArrays[ei]->dataBuffer();
      DSP_DIAG(VERIFY, "SLOT_WRITE slot=%d tag=EXT_SYNC(%s) extIdx=%d pAct=%d sAct=%d "
                       "len=%lld addr=%p isVariable=%d",
               -(ei + 1), tag, ei,
               db ? (db->isPrimaryActual() ? 1 : 0) : -1,
               db ? (db->isSpecialActual() ? 1 : 0) : -1,
               (long long)externalArrays[ei]->lengthOf(),
               DSP_BUF(externalArrays[ei]),
               isVariable ? 1 : 0);
    }

    if (isWeight) {
      skipped++;
      continue;
    }
    if (isVariable) {
      // Force H2D only when host is authoritative (isPrimaryActual). When device
      // is authoritative (isSpecialActual=true, isPrimaryActual=false), forcing
      // H2D would overwrite valid device data with stale host data.
      auto* db = externalArrays[ei]->dataBuffer();
      if (db != nullptr && db->isPrimaryActual()) {
        db->syncToSpecial(true);  // Force H2D: host has newer data than device
      }
      // else: device is authoritative (after KvScatter or initial model load) — no H2D
    } else {
      externalArrays[ei]->syncToDevice();
    }
    synced++;
  }
  if (outSynced) *outSynced = synced;
  if (outSkipped) *outSkipped = skipped;
}

static void fingerprintVariableInputs(NDArray** externalArrays, int numExt,
                                      const std::vector<bool>& externalInputIsVariable,
                                      const std::vector<std::string>& externalInputNames,
                                      const char* label, int execCount) {
  for (int ei = 0; ei < numExt; ei++) {
    if (externalArrays[ei] == nullptr) continue;
    if (ei >= static_cast<int>(externalInputIsVariable.size()) ||
        !externalInputIsVariable[ei]) continue;
    const char* name = (ei < static_cast<int>(externalInputNames.size()))
                       ? externalInputNames[ei].c_str() : "?";
    DSP_DIAG_FINGERPRINT(label, ei, name, externalArrays[ei], execCount);
  }
}

static bool pushPrimaryCtxIfConfigured(int deviceId, CUcontext* outPrimary,
                                       CUcontext* outPrev) {
  if (!Environment::getInstance().tritonGraphCtxPush()) return false;
  CUdevice cuDev;
  cuDeviceGet(&cuDev, deviceId);
  cuDevicePrimaryCtxRetain(outPrimary, cuDev);
  cuCtxGetCurrent(outPrev);
  if (*outPrev != *outPrimary) {
    cuCtxPushCurrent(*outPrimary);
    DSP_DIAG(EXECUTE, "Triton capture pushed primary ctx %p (was %p) for device %d",
             (void*)*outPrimary, (void*)*outPrev, deviceId);
    return true;
  }
  return false;
}

static void popPrimaryCtxIfPushed(bool didPush, int deviceId) {
  if (!didPush) return;
  CUcontext dummy;
  cuCtxPopCurrent(&dummy);
  CUdevice cuDev;
  cuDeviceGet(&cuDev, deviceId);
  cuDevicePrimaryCtxRelease(cuDev);
}

// ── Capture TLS cleanup helper ──────────────────────────────────────────
static void cleanupCaptureTlsState(bool freeHostPtrs, cudaStream_t prevCaptureStream) {
  tl_graphExecutionActive = false;
  tl_captureWorkspace = nullptr;
  tl_captureWorkspaceSize = 0;
  tl_captureWorkspaceOffset = 0;
  if (freeHostPtrs) {
    for (auto* ptr : tl_capturedHostPtrs) {
      if (ptr != nullptr) cudaFreeHost(ptr);
    }
  }
  tl_captureHostWorkspace = nullptr;
  tl_captureHostWorkspaceSize = 0;
  tl_captureHostWorkspaceOffset = 0;
  tl_capturedHostPtrs.clear();
  tl_captureReplicateCache.clear();
  tl_graphCaptureStream = prevCaptureStream;
}

// Full abort: cleanup TLS + pop context + restore cuBLAS + restore slot state + destroy handle.
// This is the pattern repeated at every early-return from capture.
void NativeDynamicShapePlan::abortCapture(GraphSegment& seg,
                                          bool freeHostPtrs,
                                          bool didPushCtx, int captureDevice,
                                          cudaStream_t prevCaptureStream,
                                          const std::vector<NativeSlot::SlotState>& savedSlotState,
                                          void* stream) {
  cleanupCaptureTlsState(freeHostPtrs, prevCaptureStream);
  popPrimaryCtxIfPushed(didPushCtx, captureDevice);
  restoreCublasWorkspaceAfterCapture(stream);
  if (!savedSlotState.empty()) {
    for (int s = seg.def.startSlot; s <= seg.def.endSlot; s++) {
      slots_[s].state_ = savedSlotState[s - seg.def.startSlot];
    }
  }
  cleanupSegmentForRebuild(seg, "capture_abort");
}

// ── Slot state save/restore helpers ───────────────────────────────────────
// Demote FROZEN→SHAPE_CACHED for warmup/capture, preserving FROZEN_CONSTANT.
static void demoteFrozenSlotStates(NativeSlot* slots, int startSlot, int endSlot,
                                   std::vector<NativeSlot::SlotState>& savedState) {
  savedState.resize(endSlot - startSlot + 1);
  for (int s = startSlot; s <= endSlot; s++) {
    savedState[s - startSlot] = slots[s].state_;
    if (slots[s].state_ == NativeSlot::SlotState::FROZEN)
      slots[s].state_ = NativeSlot::SlotState::SHAPE_CACHED;
  }
}

static void restoreSlotStates(NativeSlot* slots, int startSlot, int endSlot,
                              const std::vector<NativeSlot::SlotState>& savedState) {
  for (int s = startSlot; s <= endSlot; s++) {
    slots[s].state_ = savedState[s - startSlot];
  }
}

// ── View-capable slot promotion helper ────────────────────────────────────
static int promoteViewCapableSlotsToFrozen(NativeSlot* slots, int startSlot, int endSlot) {
  int promoted = 0;
  for (int s = startSlot; s <= endSlot; s++) {
    auto& sl = slots[s];
    if (!sl.flags.isViewCapableOp || sl.state_ >= NativeSlot::SlotState::FROZEN) continue;
    sl.state_ = NativeSlot::SlotState::FROZEN;
    promoted++;
  }
  return promoted;
}

static void dumpSegFinalArgmax(const GraphSegment& seg,
                               NDArray** outputSlots, int totalOutputSlots,
                               int numSlots, NativeSlot* slots,
                               cudaStream_t cudaStr,
                               const char* label, int execCount) {
  if (!DSP_DIAG_ENABLED(EXECUTE)) return;
  int finalOutputSlot = -1;
  if (seg.def.endSlot < numSlots && slots[seg.def.endSlot].wiring.numOutputs > 0) {
    finalOutputSlot = slots[seg.def.endSlot].wiring.outputSlotIndices[0];
  }
  if (finalOutputSlot < 0 || finalOutputSlot >= totalOutputSlots)
    finalOutputSlot = seg.def.endSlot;
  if (finalOutputSlot < 0 || finalOutputSlot >= totalOutputSlots ||
      outputSlots[finalOutputSlot] == nullptr) return;
  auto* out = outputSlots[finalOutputSlot];
  if (out->dataType() != FLOAT32 || out->lengthOf() <= 0) return;
  cudaStreamSynchronize(cudaStr);
  int argmax = dspArgmax(DSP_BUF(out), out->dataType(), out->lengthOf());
  std::string vals = dspDumpSlotValues(DSP_BUF(out), out->dataType(), out->lengthOf(), 4);
  DSP_DIAG(EXECUTE, "%s seg[%d-%d] slot=%d argmax=%d len=%lld vals=%s execCount=%d",
           label, seg.def.startSlot, seg.def.endSlot, finalOutputSlot, argmax,
           (long long)out->lengthOf(), vals.c_str(), execCount);
}

static bool validateAndStoreMergedCapture(
    const char* diagPrefix,
    sd::cuda::CudaGraphHandle* nativeHandle,
    std::unique_ptr<GraphReplayHandle>& handle,
    ReplaySchedule& sched,
    int mergedGroupId, int startSlot, int endSlot,
    size_t nodeCount, void* stream, cudaStream_t cudaStr) {

  bool instOk = nativeHandle->instantiate();
  if (!instOk) {
    DSP_DIAG(EXECUTE, "%s: group=%d instantiate FAILED", diagPrefix, mergedGroupId);
    return false;
  }
  bool launchOk = handle->replay(stream);
  if (!launchOk) {
    DSP_DIAG(EXECUTE, "%s: group=%d validation replay FAILED", diagPrefix, mergedGroupId);
    return false;
  }
  cudaError_t syncErr = cudaStreamSynchronize(cudaStr);
  if (syncErr != cudaSuccess) {
    DSP_DIAG(EXECUTE, "%s: group=%d validation sync FAILED err=%d",
             diagPrefix, mergedGroupId, static_cast<int>(syncErr));
    cudaGetLastError();
    return false;
  }
  DSP_DIAG(EXECUTE, "%s: group=%d [%d-%d] captured+validated OK nodes=%zu",
           diagPrefix, mergedGroupId, startSlot, endSlot, nodeCount);
  sched.mergedReplayHandles.push_back(std::move(handle));
  return true;
}

// Default capture workspace sizes (configurable via env vars)
static size_t TRITON_CAPTURE_HOST_WORKSPACE_SIZE = []() -> size_t {
  size_t mb = static_cast<size_t>(Environment::getInstance().dsp().captureHostWorkspaceMb());
  return mb * 1024ULL * 1024ULL;
}();

// Default capture workspace size for Triton graph capture (128MB).
// Configurable via ND4J_DSP_CAPTURE_WORKSPACE_MB env var.
static size_t TRITON_CAPTURE_WORKSPACE_SIZE = []() -> size_t {
  size_t mb = static_cast<size_t>(Environment::getInstance().dsp().captureWorkspaceMb());
  return mb * 1024ULL * 1024ULL;
}();

// Status enum string helper — delegates to shared dsp::dspStatusName in DspPhaseUtils.h.
static inline const char* statusName_gpu(Status status) {
  return dsp::dspStatusName(status);
}

// Helper: extract specialBuffer() device addresses from NDArray** into void** for
// address snapshot diagnostics. Thread-local to avoid repeated allocation.
static void extractDeviceAddrs(NDArray** arrays, int count, std::vector<void*>& out) {
  out.resize(count);
  for (int i = 0; i < count; i++) {
    out[i] = (arrays != nullptr && arrays[i] != nullptr)
             ? DSP_BUF(arrays[i]) : nullptr;
  }
}

// ── Address snapshot helpers ──────────────────────────────────────────────
// Thread-local vectors for snapshot storage to avoid repeated allocation.
static thread_local std::vector<void*> tl_snapshotOutputAddrs;
static thread_local std::vector<void*> tl_snapshotExtAddrs;

static void snapshotAddrs(NDArray** outputSlots, int totalOutputSlots,
                          NDArray** externalArrays, int numExt,
                          const char* label) {
  if (!DSP_DIAG_ENABLED(MEMORY)) return;
  extractDeviceAddrs(outputSlots, totalOutputSlots, tl_snapshotOutputAddrs);
  extractDeviceAddrs(externalArrays, numExt, tl_snapshotExtAddrs);
  DSP_DIAG(MEMORY, "ADDR_SNAPSHOT(%s): %d output + %d ext addrs captured",
           label, totalOutputSlots, numExt);
}

/**
 * Compute FNV-1a hash of slot output specialBuffer() addresses for a segment.
 * Used to verify that output buffers haven't been reallocated between capture
 * and replay — stale addresses in a CUDA graph cause SIGSEGV or corruption.
 */
static LongType computeSlotAddrHash(NDArray** outputSlots, int startSlot, int endSlot, int totalSlots) {
  return dsp::computeSlotAddrHash(outputSlots, startSlot, endSlot, totalSlots,
                                  [](NDArray* a) -> void* { return DSP_BUF(a); });
}

static void dumpGpuContextState(int failedDeviceId, const char* errorType) {
  // 1. Check for pre-existing CUDA error (downstream error detection)
  cudaError_t preExisting = cudaPeekAtLastError();
  if (preExisting != cudaSuccess) {
    DSP_DIAG(MEMORY, "%s: PRE-EXISTING CUDA ERROR on device %d: %d (%s) — "
                     "this may be a downstream error from a previous operation",
             errorType, failedDeviceId,
             static_cast<int>(preExisting), cudaGetErrorString(preExisting));
  }

  // 2. Failed device: full detail
  {
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, failedDeviceId);
    size_t gpuFree = 0, gpuTotal = 0;
    cudaSetDevice(failedDeviceId);
    cudaMemGetInfo(&gpuFree, &gpuTotal);
    DSP_DIAG(MEMORY, "%s: device %d '%s' (cc=%d.%d): free=%zuMB total=%zuMB used=%zuMB "
                     "multiProcessorCount=%d",
             errorType, failedDeviceId, props.name,
             props.major, props.minor,
             gpuFree / (1024*1024), gpuTotal / (1024*1024),
             (gpuTotal - gpuFree) / (1024*1024),
             props.multiProcessorCount);
  }

  // 3. Other devices: one-line summary each
  int deviceCount = 0;
  cudaGetDeviceCount(&deviceCount);
  for (int d = 0; d < deviceCount; d++) {
    if (d == failedDeviceId) continue;
    cudaSetDevice(d);
    size_t otherFree = 0, otherTotal = 0;
    cudaMemGetInfo(&otherFree, &otherTotal);
    cudaError_t otherErr = cudaPeekAtLastError();
    DSP_DIAG(MEMORY, "%s: device %d: free=%zuMB total=%zuMB%s",
             errorType, d, otherFree / (1024*1024), otherTotal / (1024*1024),
             otherErr != cudaSuccess ?
             (std::string(" CUDA_ERROR=") + cudaGetErrorString(otherErr)).c_str() : "");
  }

  // 4. Restore original device
  cudaSetDevice(failedDeviceId);

  // 5. Report graph execution active state
  DSP_DIAG(MEMORY, "%s: tl_graphExecutionActive=%d",
           errorType, tl_graphExecutionActive ? 1 : 0);
}


static Status reportOomError(GraphSegment& seg, const char* phase,
                             size_t requestedBytes, int deviceId) {
  dumpGpuContextState(deviceId, "OOM");
  size_t gpuFree = 0, gpuTotal = 0;
  cudaMemGetInfo(&gpuFree, &gpuTotal);
  DSP_DIAG(MEMORY,
           "OOM ERROR in seg[%d-%d] during '%s' on device %d: "
           "requested=%zuMB gpuFree=%zuMB gpuTotal=%zuMB gpuUsed=%zuMB "
           "executionCount=%d phase=%d",
           seg.def.startSlot, seg.def.endSlot, phase, deviceId,
           requestedBytes / (1024*1024), gpuFree / (1024*1024),
           gpuTotal / (1024*1024), (gpuTotal - gpuFree) / (1024*1024),
           seg.exec.executionCount, seg.exec.displayPhaseName());
  SegmentLifecycle::markFailed(seg.exec, "oom");
  return Status::KERNEL_FAILURE;
}

static Status reportCaptureError(GraphSegment& seg, const char* step,
                                 cudaError_t cudaErr, int deviceId) {
  dumpGpuContextState(deviceId, "CAPTURE");
  size_t gpuFree = 0, gpuTotal = 0;
  cudaMemGetInfo(&gpuFree, &gpuTotal);
  DSP_DIAG(EXECUTE,
           "CAPTURE ERROR in seg[%d-%d] at step '%s' on device %d: "
           "cudaError=%d (%s) gpuFree=%zuMB gpuTotal=%zuMB "
           "executionCount=%d numOps=%d compiledBy='%s'",
           seg.def.startSlot, seg.def.endSlot, step, deviceId,
           static_cast<int>(cudaErr), cudaGetErrorString(cudaErr),
           gpuFree / (1024*1024), gpuTotal / (1024*1024),
           seg.exec.executionCount, seg.def.endSlot - seg.def.startSlot + 1,
           seg.exec.compiledByBackend.c_str());
  SegmentLifecycle::markFailed(seg.exec, step);
  cudaGetLastError(); // clear error state
  return Status::KERNEL_FAILURE;
}

static Status reportReplayError(GraphSegment& seg, const char* step,
                                cudaError_t cudaErr, int deviceId) {
  dumpGpuContextState(deviceId, "REPLAY");
  size_t gpuFree = 0, gpuTotal = 0;
  cudaMemGetInfo(&gpuFree, &gpuTotal);
  DSP_DIAG(EXECUTE,
           "REPLAY ERROR in seg[%d-%d] at step '%s' on device %d: "
           "cudaError=%d (%s) gpuFree=%zuMB gpuTotal=%zuMB "
           "executionCount=%d hasReplayHandle=%d",
           seg.def.startSlot, seg.def.endSlot, step, deviceId,
           static_cast<int>(cudaErr), cudaGetErrorString(cudaErr),
           gpuFree / (1024*1024), gpuTotal / (1024*1024),
           seg.exec.executionCount,
           seg.exec.replayHandle != nullptr ? 1 : 0);
  SegmentLifecycle::markFailed(seg.exec, step);
  cudaGetLastError(); // clear error state
  return Status::KERNEL_FAILURE;
}

#if HAVE_TRITON
static bool findUnsupportedTritonReplayGap(TritonGraphBackend* tritonBackend,
                                          const GraphSegment& seg,
                                          NativeSlot* slots,
                                          int* firstGapSlot,
                                          int* lastCoveredSlot,
                                          int* gapSlotCount) {
 if (firstGapSlot != nullptr) *firstGapSlot = -1;
 if (lastCoveredSlot != nullptr) *lastCoveredSlot = -1;
 if (gapSlotCount != nullptr) *gapSlotCount = 0;
 if (tritonBackend == nullptr) return false;

 auto gapSlots = tritonBackend->getGapSlots(seg, slots);
 if (gapSlotCount != nullptr) *gapSlotCount = static_cast<int>(gapSlots.size());
 if (gapSlots.empty()) return false;

 int maxCoveredSlot = -1;
 for (int slot = seg.def.startSlot; slot <= seg.def.endSlot; slot++) {
   if (gapSlots.find(slot) == gapSlots.end()) {
     maxCoveredSlot = slot;
   }
 }
 if (lastCoveredSlot != nullptr) *lastCoveredSlot = maxCoveredSlot;
 if (maxCoveredSlot < 0) return false;

 int earliestUnsupportedGap = -1;
 for (int slot = seg.def.startSlot; slot <= seg.def.endSlot; slot++) {
   if (gapSlots.find(slot) != gapSlots.end() && slot < maxCoveredSlot) {
     earliestUnsupportedGap = slot;
     break;
   }
 }
 if (firstGapSlot != nullptr) *firstGapSlot = earliestUnsupportedGap;
 return earliestUnsupportedGap >= 0;
}

/**
* Build an ordered replay schedule for a mixed Triton/gap segment.
*
* Given a segment like seg[200-399] with gaps at [298-312] and [347-369],
* this produces:
*   unit 0: TRITON_ISLAND [200-297]  islandIndex=0
*   unit 1: GAP            [298-312] islandIndex=-1
*   unit 2: TRITON_ISLAND [313-346]  islandIndex=1
*   unit 3: GAP            [347-369] islandIndex=-1
*   unit 4: TRITON_ISLAND [370-399]  islandIndex=2
*/
static ReplaySchedule buildCompositeReplaySchedule(const GraphSegment& seg,
                                                  NativeSlot* slots,
                                                  TritonGraphBackend* tritonBackend) {
 ReplaySchedule schedule;
 auto gap_slots = tritonBackend->getGapSlots(seg, slots);

 int islandIdx = 0;
 int rangeStart = seg.def.startSlot;
 bool inIsland = (gap_slots.find(seg.def.startSlot) == gap_slots.end());

 for (int slot = seg.def.startSlot; slot <= seg.def.endSlot + 1; slot++) {
   bool isGap = (slot <= seg.def.endSlot && gap_slots.find(slot) != gap_slots.end());
   bool atBoundary = (slot > seg.def.endSlot) || (inIsland && isGap) || (!inIsland && !isGap);

   if (atBoundary && slot > rangeStart) {
     if (inIsland) {
       DSP_DIAG_SEG(SEGMENT, rangeStart,
                    "compositeReplaySchedule UNIT kind=ISLAND islandBefore=%d startSlot=%d endSlot=%d segParent=[%d-%d]",
                    islandIdx, rangeStart, slot - 1, seg.def.startSlot, seg.def.endSlot);
       schedule.units.emplace_back(REPLAY_UNIT_TRITON_ISLAND, rangeStart, slot - 1, islandIdx++);
     } else {
       DSP_DIAG_SEG(SEGMENT, rangeStart,
                    "compositeReplaySchedule UNIT kind=GAP startSlot=%d endSlot=%d segParent=[%d-%d]",
                    rangeStart, slot - 1, seg.def.startSlot, seg.def.endSlot);
       schedule.units.emplace_back(REPLAY_UNIT_GAP, rangeStart, slot - 1, -1);
     }
     rangeStart = slot;
     inIsland = !isGap;
   }
   // If at seg.def.endSlot+1 boundary with pending range, the loop above handles it
   if (slot == seg.def.endSlot + 1 && rangeStart <= seg.def.endSlot) {
     if (inIsland) {
       DSP_DIAG_SEG(SEGMENT, rangeStart,
                    "compositeReplaySchedule UNIT kind=ISLAND-tail islandBefore=%d startSlot=%d endSlot=%d segParent=[%d-%d]",
                    islandIdx, rangeStart, seg.def.endSlot, seg.def.startSlot, seg.def.endSlot);
       schedule.units.emplace_back(REPLAY_UNIT_TRITON_ISLAND, rangeStart, seg.def.endSlot, islandIdx++);
     } else {
       DSP_DIAG_SEG(SEGMENT, rangeStart,
                    "compositeReplaySchedule UNIT kind=GAP-tail startSlot=%d endSlot=%d segParent=[%d-%d]",
                    rangeStart, seg.def.endSlot, seg.def.startSlot, seg.def.endSlot);
       schedule.units.emplace_back(REPLAY_UNIT_GAP, rangeStart, seg.def.endSlot, -1);
     }
   }
 }

 // ── Gap classification: mark capture-safe gaps ──────────────────────
 // A gap is capture-safe iff ALL its slots launch CUDA kernels and are
 // NOT pure-CPU metadata ops (views, identity) or frozen constants.
 // Capture-safe gaps can be recorded into the preceding island's CUDA
 // graph during merged capture, eliminating gap dispatch overhead.
 for (auto& unit : schedule.units) {
   if (unit.kind != REPLAY_UNIT_GAP) continue;
   unit.isCaptureSafe = true;
   for (int s = unit.startSlot; s <= unit.endSlot; s++) {
     if (slots[s].flags.isViewCapableOp ||
         slots[s].flags.isIdentityOp ||
         slots[s].frozenConstantSlot()) {
       unit.isCaptureSafe = false;
       break;
     }
   }
   DSP_DIAG_SEG(SEGMENT, unit.startSlot,
                "compositeReplaySchedule GAP [%d-%d] isCaptureSafe=%d",
                unit.startSlot, unit.endSlot, unit.isCaptureSafe ? 1 : 0);
 }

 // Pre-allocate replay handles for each island
 schedule.compositeReplayHandles.resize(schedule.units.size());
 return schedule;
}

Status NativeDynamicShapePlan::compositeReplay(
    GraphSegment& seg, ReplaySchedule& sched,
    NDArray** externalArrays, int numExt, void* stream) {

  // Phase assertion: compositeReplay MUST be called in SHAPES_FROZEN or later.
  // Calling during SLOT_BY_SLOT means shapes aren't stable and graph replay is unsafe.
  if (planPhase_ < PlanPhase::SHAPES_FROZEN) {
    DSP_DIAG(EXECUTE, "PHASE_VIOLATION: compositeReplay called in phase %s, "
                      "requires >= SHAPES_FROZEN. seg[%d-%d] execCount=%d",
             dsp::planPhaseName(planPhase_), seg.def.startSlot, seg.def.endSlot, executeCount_);
    REQUIRE_TRUE(false, 0,
                 "DSP phase contract violation: compositeReplay requires planPhase_ >= SHAPES_FROZEN "
                 "for seg[%d-%d].", seg.def.startSlot, seg.def.endSlot);
  }

  auto cudaStr = (stream != nullptr) ? *static_cast<cudaStream_t*>(stream) : nullptr;

  // Reset cast cache indices so gap-executed matmuls can reuse persistent
  // HALF buffers from the cast cache (populated during capture warmup).
  // Without this, each matmul allocates+frees a HALF buffer per call
  // (202 matmuls × 2 = 404 cudaMallocAsync/cudaFreeAsync per step).
  //
  // When merged CUDA graphs exist, those graphs bake in device pointers for
  // cast-cache slots [0, hwmA) / [0, hwmB).  Unmerged gap matmuls must start
  // from the high-water mark so they don't overwrite those baked buffers with
  // different data — which would corrupt the merged graph's FP16 matmul inputs.
  {
    auto& sched = seg.exec.compositeReplaySchedule;
    if (!dsp_disable_cast_hwm() &&
        !sched.mergedReplayHandles.empty() &&
        (sched.mergedCastHwmA > 0 || sched.mergedCastHwmB > 0)) {
      MmulHelper::resetCastCacheIndicesTo(sched.mergedCastHwmA, sched.mergedCastHwmB);
    } else {
      MmulHelper::resetCastCacheIndices();
    }
  }

  // ── cuBLAS workspace for replay gap ops ────────────────────────────────
  // During capture, cuBLAS had the DSP workspace set (via setCublasWorkspaceForCapture).
  // cuBLAS selects different algorithms based on available workspace size.  Without
  // setting the same workspace here, unmerged gap matmuls execute with workspace=nullptr
  // (restored by restoreCublasWorkspaceAfterCapture at end of capture phase), causing
  // cuBLAS to select a different algorithm → different numerical results.
  //
  // By setting the workspace here, we ensure ALL cuBLAS calls — whether baked into a
  // merged CUDA graph or executing live as unmerged gaps — use identical algorithm
  // selection, eliminating the accuracy regression.
  //
  // RAII guard restores previous workspace state on any exit path.
  //
  // CRITICAL: When merged CUDA graph replay handles exist, do NOT set the
  // workspace for live gap ops.  The merged graph's baked cublasLtMatmul nodes
  // store plan/descriptor data in cublasWorkspaceBuffer_.  If live gap ops
  // (batched GEMM or tryLtMatmul via executeSlot) use that same buffer as
  // scratch, they corrupt the plan data — causing the merged graph to produce
  // wrong results after enough replay iterations (typically 50-100 steps).
  //
  // Without an explicit workspace, live cuBLAS calls use internal allocation
  // (slightly less optimal algorithm, but correct).  The merged graph's nodes
  // are unaffected by TLS state — they replay the exact captured kernel args.
  bool hasMergedHandles = !sched.mergedReplayHandles.empty() && !dsp_disable_workspace_skip();
  struct CublasWorkspaceReplayGuard {
    void* prevPtr;
    size_t prevSize;
    CublasWorkspaceReplayGuard(NativeDynamicShapePlan* p, bool skipWorkspace)
        : prevPtr(tl_cublasWorkspacePtr), prevSize(tl_cublasWorkspaceSize) {
      if (!skipWorkspace && sd::Environment::getInstance().cublasCaptureWorkspace()) {
        p->setCublasWorkspaceForWarmup();  // Same workspace as capture
      }
      // When skipWorkspace is true, leave tl_cublasWorkspacePtr at its previous
      // value (typically nullptr after restoreCublasWorkspaceAfterCapture).
    }
    ~CublasWorkspaceReplayGuard() {
      tl_cublasWorkspacePtr = prevPtr;
      tl_cublasWorkspaceSize = prevSize;
    }
  } cublasWsGuard(this, hasMergedHandles);
#endif

  // Per-step deduplication: access the active PlanExecutionContext to avoid
  // repeating plan-level operations (ext input sync, cross-stream ordering,
  // input address hashing) once per segment when we have 14+ segments.
  // compositeReplay runs inside execute(), so the context MUST be set.
  auto* execCtx = static_cast<PlanExecutionContext*>(activeExecutionContext());
  assert(execCtx != nullptr && "compositeReplay called outside execute() — activeExecCtx_ is null");

  // Cross-stream sync: ensure Java .assign() writes on default stream are visible.
  // MUST happen BEFORE the gap-stream override below — syncCrossStream reads
  // the real default stream via getCudaStream(). With the override active,
  // getCudaStream() returns cudaStr, making the sync a no-op.
  if (!execCtx->crossStreamSynced) {
    syncCrossStream(cudaStr, execCtx->crossStreamEvent, "compositeReplay",
                    seg.def.startSlot, seg.def.endSlot, seg.exec.executionCount);
    execCtx->crossStreamSynced = true;
  }

  // ── Gap-stream unification ─────────────────────────────────────────────
  // Redirect LaunchContext::getCudaStream() to return cudaStr for the
  // duration of compositeReplay. This makes gap ops (matmul, reshape, etc.)
  // run on the same CUDA stream as Triton island graph replay, eliminating
  // all cross-stream event syncs (28 cudaEventRecord + cudaStreamWaitEvent
  // calls per segment per step with 14 islands). The cross-stream sync
  // guards below become no-ops because gapStream == cudaStr.
  // RAII: restored automatically on any exit path (early return, exception).
  struct GapStreamGuard {
    cudaStream_t prev;
    GapStreamGuard(cudaStream_t s) : prev(tl_dspGapStream) { tl_dspGapStream = s; }
    ~GapStreamGuard() { tl_dspGapStream = prev; }
  } gapStreamGuard(cudaStr);

  // Set DSP execution stream for async H2D copies
  sd::graph::DspStreamGuard dspStreamGuard(cudaStr);

  // Sync variable external inputs to device + fingerprint + KV stale check.
  // External inputs are plan-level (same arrays for all segments), so these
  // only need to run once per step. The first segment does the work; subsequent
  // segments skip via the execCtx dedup flag.
  if (!execCtx->extInputsSynced) {
    syncExternalInputs(externalArrays, numExt, externalInputIsVariable_,
                       externalInputNames_, shapesFrozen_, false, "replay", seg.exec.executionCount);

    // Fingerprint every variable external input at replay entry.
    if (shapesFrozen_ && !externalInputIsVariable_.empty()) {
      fingerprintVariableInputs(externalArrays, numExt, externalInputIsVariable_,
                                externalInputNames_, "replay-entry", seg.exec.executionCount);
    }

    // Assertion 1: KV-cache mutation detection.
    // After execCount > 2, at least SOME variable inputs should change each step (KV
    // caches grow/update, position IDs advance, etc.). If ALL variable inputs have
    // identical fingerprints to the previous call, that is the "KV caches never updated"
    // bug — nothing is actually writing new decode data.
    //
    // Implementation: compute FNV-1a of each variable input's host bytes (via syncToHost),
    // compare against the map from the previous call, and warn if every entry matches.
    // The map is thread-local and keyed by external-input index so multiple concurrent
    // plans on different threads don't interfere.
    if (shapesFrozen_ && !externalInputIsVariable_.empty() &&
        DspDiagnostics::getInstance().isEnabled(DSP_DIAG_EXECUTE)) {
      // Collect current fingerprints for variable inputs and compare to previous.
      std::unordered_map<int, uint64_t> currentFingerprints;
      bool anyVariable = false;
      bool allMatch = true;

      for (int ei = 0; ei < numExt; ei++) {
        if (externalArrays[ei] == nullptr) continue;
        if (ei >= static_cast<int>(externalInputIsVariable_.size()) ||
            !externalInputIsVariable_[ei]) continue;

        anyVariable = true;
        NDArray* arr = externalArrays[ei];
        arr->syncToHost();
        auto* db = arr->dataBuffer();
        uint64_t h = 0xcbf29ce484222325ULL;  // FNV-1a offset basis
        if (db != nullptr && db->primary() != nullptr) {
          size_t elemBytes = arr->sizeOfT();
          if (elemBytes == 0) elemBytes = 1;
          size_t totalBytes = static_cast<size_t>(arr->lengthOf()) * elemBytes;
          const uint8_t* base = static_cast<const uint8_t*>(db->primary())
                                + arr->offset() * elemBytes;
          for (size_t bi = 0; bi < totalBytes; bi++) {
            h ^= static_cast<uint64_t>(base[bi]);
            h *= 0x100000001b3ULL;  // FNV-1a prime
          }
        }
        currentFingerprints[ei] = h;

        auto prev = execCtx->prevVariableFingerprints.find(ei);
        if (prev != execCtx->prevVariableFingerprints.end() && prev->second != h) {
          allMatch = false;
        } else if (prev == execCtx->prevVariableFingerprints.end()) {
          // First time seeing this index — can't conclude all-match yet.
          allMatch = false;
        }
      }

      if (anyVariable && allMatch && !execCtx->prevVariableFingerprints.empty()) {
        // All variable inputs have identical content to the previous step.
        // This is the symptom of the "KV caches never updated" bug.
        DSP_DIAG(EXECUTE,
                 "KV_CACHE_STALE_WARNING: seg[%d-%d] execCount=%d — ALL variable external "
                 "inputs have identical fingerprints to the previous step. "
                 "KV caches may not be updating between decode steps. "
                 "numVariableInputs=%d",
                 seg.def.startSlot, seg.def.endSlot, seg.exec.executionCount,
                 static_cast<int>(currentFingerprints.size()));
        // Log each stuck input name for diagnosis.
        for (auto& kv : currentFingerprints) {
          const char* name = (kv.first < static_cast<int>(externalInputNames_.size()))
                             ? externalInputNames_[kv.first].c_str() : "?";
          DSP_DIAG(EXECUTE,
                   "  KV_CACHE_STALE: ext[%d] name='%s' fingerprint=0x%016llx (unchanged)",
                   kv.first, name, static_cast<unsigned long long>(kv.second));
        }
      }

      // Update previous fingerprints map for next call.
      execCtx->prevVariableFingerprints = std::move(currentFingerprints);
    }

    execCtx->extInputsSynced = true;
  }

  // ── Placeholder staging buffers ─────────────────────────────────────────
  // D2D copy variable external inputs into plan-owned stable device buffers.
  // This makes arg table pointers inherently stable regardless of Java-side
  // allocation patterns. Once per step (dedup via execCtx), all segments
  // share the same effective externals.
  //
  // effectiveExternals points to staging buffers for variable inputs and
  // original pointers for non-variable (weights). All downstream GPU
  // operations (arg table, address key, gap slots) use effectiveExternals.
  NDArray** effectiveExternals = externalArrays;
  if (!execCtx->stagingBuffersSynced) {
    effectiveExternals = ensureAndSyncStagingBuffers(externalArrays, numExt, stream);
    execCtx->stagingBuffersSynced = true;
  } else if (effectiveExternals_ != nullptr) {
    // Already synced this step — reuse the effective externals from first segment.
    // Non-variable entries may have changed (associateArrayWithVariable), refresh them.
    for (int i = 0; i < numExt; i++) {
      if (i >= static_cast<int>(externalInputIsVariable_.size()) ||
          !externalInputIsVariable_[i]) {
        effectiveExternals_[i] = externalArrays[i];
      }
    }
    effectiveExternals = effectiveExternals_;
  }

  // Diagnostic: re-validate external-input + cross-segment device addresses against
  // the captured arg table. Detects Java-side rebinds (e.g. associateArrayWithVariable)
  // that leave the argTable pointing at freed device memory. Only runs when
  // DSP_DIAG VERIFY is enabled — in production, refreshArgTablesForReplay handles
  // pointer updates directly.
  if (seg.exec.argTableStable && seg.exec.capturedInputAddrKey != 0 &&
      DspDiagnostics::getInstance().isEnabled(DSP_DIAG_VERIFY)) {
    LongType currentAddrKey = computeSegmentInputAddrKey(seg, effectiveExternals, numExt);
    if (currentAddrKey != seg.exec.capturedInputAddrKey) {
      DSP_DIAG(EXECUTE,
               "EXT_INPUT_REBIND_DETECTED: seg[%d-%d] current=%lld captured=%lld "
               "→ invalidating argTableStable (forcing refresh) execCount=%d",
               seg.def.startSlot, seg.def.endSlot,
               (long long)currentAddrKey, (long long)seg.exec.capturedInputAddrKey,
               seg.exec.executionCount);
      seg.exec.argTableStable = false;
      seg.exec.addrKeyStableCount = 0;
      seg.exec.slotAddrStableCount = 0;
    } else {
      seg.exec.addrKeyStableCount++;
      DSP_DIAG(EXECUTE,
               "EXT_ADDR_KEY_STABLE: seg[%d-%d] stableCount=%d execCount=%d",
               seg.def.startSlot, seg.def.endSlot,
               seg.exec.addrKeyStableCount, seg.exec.executionCount);
    }
  }

  // Diagnostic: re-validate INTERNAL output-slot device addresses against the
  // captured hash. Detects internal slot reallocation between replay steps.
  // Only runs when DSP_DIAG VERIFY is enabled.
  if (seg.exec.argTableStable && seg.exec.capturedSlotAddrHash != 0 &&
      DspDiagnostics::getInstance().isEnabled(DSP_DIAG_VERIFY)) {
    LongType currentSlotHash = computeSlotAddrHash(
        outputSlots_, seg.def.startSlot, seg.def.endSlot, totalOutputSlots_);
    if (currentSlotHash != seg.exec.capturedSlotAddrHash) {
      DSP_DIAG(EXECUTE,
               "SLOT_ADDR_DRIFT_DETECTED: seg[%d-%d] current=0x%llx captured=0x%llx "
               "→ invalidating argTableStable (forcing refresh) execCount=%d",
               seg.def.startSlot, seg.def.endSlot,
               (long long)currentSlotHash, (long long)seg.exec.capturedSlotAddrHash,
               seg.exec.executionCount);
      seg.exec.argTableStable = false;
      seg.exec.addrKeyStableCount = 0;
      seg.exec.slotAddrStableCount = 0;
    } else {
      seg.exec.slotAddrStableCount++;
      DSP_DIAG(EXECUTE,
               "SLOT_ADDR_STABLE: seg[%d-%d] hash=0x%llx stableCount=%d execCount=%d",
               seg.def.startSlot, seg.def.endSlot,
               (long long)currentSlotHash, seg.exec.slotAddrStableCount, seg.exec.executionCount);
    }
  }

  // Refresh arg tables + D2D copy (skip when stable — fast replay path)
  bool useFastReplay = seg.exec.argTableStable &&
                       !Environment::getInstance().tritonVerifyKernels();
#if HAVE_TRITON
  if (!useFastReplay) {
   auto* tritonBackend = dynamic_cast<TritonGraphBackend*>(getGpuGraphBackend());
   if (tritonBackend != nullptr) {
     auto refreshStatus = tritonBackend->refreshArgTablesForReplay(
         seg, effectiveExternals, numExt, outputSlots_, totalOutputSlots_, stream);
     if (refreshStatus != Status::OK) {
       DSP_DIAG(EXECUTE, "COMPOSITE_REPLAY: arg table refresh FAILED seg[%d-%d]",
                seg.def.startSlot, seg.def.endSlot);
       return refreshStatus;
     }
     tritonBackend->copyConsolidatedArgTableToDevice(seg, stream);

     // Update captured keys to current addresses so subsequent steps
     // see matching keys and argTableStable stays true (fast-replay path).
     // This is critical when staging buffers change ext-input addresses
     // from capture-time originals to plan-owned stable pointers.
     seg.exec.capturedInputAddrKey = computeSegmentInputAddrKey(seg, effectiveExternals, numExt);
     seg.exec.capturedSlotAddrHash = computeSlotAddrHash(
         outputSlots_, seg.def.startSlot, seg.def.endSlot, totalOutputSlots_);
   }
 }
#endif
  cudaGetLastError();  // Clear sticky errors

  DSP_DIAG_SEG(MEMORY, seg.def.startSlot,
               "compositeReplay invoking prezeroSegmentOutputs seg=[%d-%d] stream=%p execCount=%d",
               seg.def.startSlot, seg.def.endSlot, (void*)cudaStr, seg.exec.executionCount);
  prezeroSegmentOutputs(seg, stream);

  // ── Cross-stream sync: ensure prezero memsets on cudaStr are visible ───
  // prezeroSegmentOutputs issues cudaMemsetAsync on cudaStr. Gap ops that
  // follow run on the LaunchContext's default stream. Without sync, gap ops
  // on a different stream might start before prezero completes, seeing
  // stale (non-zero) data in output buffers instead of zeros — causing
  // accumulation errors for ops that read-modify-write their output.
  // Event-based: gap stream waits for prezero completion without CPU block.
  {
    auto* lcStream = LaunchContext::defaultContext()->getCudaStream();
    cudaStream_t gapStream = lcStream ? *lcStream : nullptr;
    if (gapStream != nullptr && gapStream != cudaStr) {
      cudaEvent_t evt = execCtx->crossStreamEvent;
      cudaEventRecord(evt, cudaStr);
      cudaStreamWaitEvent(gapStream, evt, 0);
    }
  }

  // Execute units in program order, with merged group support.
  // Units in a merged group are handled by a single merged CUDA graph:
  //   - Leader unit (isMergedLeader=true): launches the merged graph
  //   - Non-leader units (mergedGroupId>=0, isMergedLeader=false): skipped
  //   - Unmerged units (mergedGroupId<0): same as original per-island replay
  int mergedGroupCount = static_cast<int>(sched.mergedReplayHandles.size());
  int unmergedGapCount = 0;
  for (auto& u : sched.units) {
    if (u.kind == REPLAY_UNIT_GAP && u.mergedGroupId < 0) unmergedGapCount++;
  }

  DSP_DIAG(EXECUTE, "COMPOSITE_REPLAY: seg[%d-%d] %d units, %d merged groups, %d unmerged gaps, execCount=%d",
           seg.def.startSlot, seg.def.endSlot,
           static_cast<int>(sched.units.size()), mergedGroupCount, unmergedGapCount,
           seg.exec.executionCount);

  for (auto& unit : sched.units) {
    // ── Merged group: non-leader units are skipped (already executed by leader's graph) ──
    if (unit.mergedGroupId >= 0 && !unit.isMergedLeader) {
      DSP_DIAG(EXECUTE, "MERGED_REPLAY: skip unit [%d-%d] mergedGroup=%d (non-leader)",
               unit.startSlot, unit.endSlot, unit.mergedGroupId);
      // Still need to tick actuality for these slots — the merged graph wrote to them
      for (int s = unit.startSlot; s <= unit.endSlot && s < totalOutputSlots_; s++) {
        NDArray* arr = outputSlots_[s];
        if (arr != nullptr && arr->dataBuffer() != nullptr && !arr->dataBuffer()->isClosed()) {
          arr->tickWriteDevice();
        }
      }
      continue;
    }

    // ── Merged group leader: launch the merged graph ──
    if (unit.mergedGroupId >= 0 && unit.isMergedLeader) {
      int mgId = unit.mergedGroupId;
      if (mgId < 0 || mgId >= static_cast<int>(sched.mergedReplayHandles.size()) ||
          !sched.mergedReplayHandles[mgId] || !sched.mergedReplayHandles[mgId]->isReady()) {
        DSP_DIAG(EXECUTE, "MERGED_REPLAY: group %d handle not ready", mgId);
        return Status::KERNEL_FAILURE;
      }

      DSP_DIAG(EXECUTE, "MERGED_REPLAY: group %d leader [%d-%d] launching",
               mgId, unit.startSlot, unit.endSlot);
      bool launchOk = sched.mergedReplayHandles[mgId]->replay(stream);
      if (!launchOk) {
        DSP_DIAG(EXECUTE, "MERGED_REPLAY: group %d launch FAILED", mgId);
        return Status::KERNEL_FAILURE;
      }

      // Tick actuality for leader's slots
      for (int s = unit.startSlot; s <= unit.endSlot && s < totalOutputSlots_; s++) {
        NDArray* arr = outputSlots_[s];
        if (arr != nullptr && arr->dataBuffer() != nullptr && !arr->dataBuffer()->isClosed()) {
          arr->tickWriteDevice();
        }
      }
      continue;
    }

    // ── Unmerged units: original per-unit replay ──
    if (unit.kind == REPLAY_UNIT_GAP) {
      DSP_DIAG(EXECUTE, "COMPOSITE_REPLAY: gap [%d-%d]", unit.startSlot, unit.endSlot);
      // Gap slots execute real ops (including matmul for logits) via executeSlot.
      // With gap-stream unification (tl_dspGapStream), gap ops run on cudaStr —
      // the same stream as island graph replay. FIFO ordering guarantees gap ops
      // see island outputs without explicit sync.
      //
      // tl_dspReplayActive: suppress per-op cudaStreamSynchronize in cuDNN ops,
      // PointersManager::synchronize(), etc. All work is on the unified DSP stream
      // (tl_dspGapStream) — FIFO ordering makes per-op syncs redundant.
      bool prevReplayActive = tl_dspReplayActive;
      tl_dspReplayActive = true;
      for (int s = unit.startSlot; s <= unit.endSlot; s++) {
        // ── Batched GEMM dispatch in gap loop ──────────────────────────
        if (!batchedGemmGroups_.empty() && s < (int)slotToBatchedGemmGroup_.size()) {
          int bgIdx = slotToBatchedGemmGroup_[s];
          if (bgIdx >= 0 && bgIdx < (int)batchedGemmGroups_.size()) {
            auto& bgGroup = batchedGemmGroups_[bgIdx];
            if (s == bgGroup.triggerSlot) {
              cudaStream_t execStream = stream ? *static_cast<cudaStream_t*>(stream) : static_cast<cudaStream_t>(nullptr);
              Status batchStatus = executeBatchedGemmGroup(bgIdx, effectiveExternals, numExt, execStream);
              if (batchStatus != Status::OK) {
                tl_dspReplayActive = prevReplayActive;
                DSP_DIAG(EXECUTE, "COMPOSITE_REPLAY: batched GEMM group %d FAILED at slot %d status=%d",
                         bgIdx, s, static_cast<int>(batchStatus));
                return batchStatus;
              }
              continue;
            } else {
              continue;
            }
          }
        }

        // ── View-op fast path: skip executeSlot() when the backing buffer is unchanged ──
        // reshape_no_copy and other view-capable ops in FROZEN state just alias an input
        // buffer. If the output NDArray already points to the same DataBuffer as input0,
        // the view is still valid — no dispatch needed. tickWriteDevice() keeps device
        // actuality up to date. Falls through to executeSlot() when the check fails.
        if (!dsp_disable_view_fastpath() &&
            s < totalOutputSlots_ &&
            slots_[s].flags.isViewCapableOp &&
            slots_[s].state_ >= NativeSlot::SlotState::FROZEN &&
            slots_[s].wiring.numInputs >= 1 &&
            slots_[s].wiring.numOutputs >= 1) {
          int outSi = slots_[s].wiring.outputSlotIndices[0];
          if (outSi >= 0 && outSi < totalOutputSlots_) {
            NDArray* currentOut = outputSlots_[outSi];
            int inSrc = slots_[s].wiring.inputSourceIndices[0];
            NDArray* input0 = nullptr;
            if (inSrc >= 0) {
              if (inSrc < totalOutputSlots_) {
                input0 = outputSlots_[inSrc];
              }
            } else {
              int extIdx = -(inSrc + 1);
              if (extIdx >= 0 && extIdx < numExt) {
                input0 = effectiveExternals[extIdx];
              }
            }
            if (currentOut != nullptr && input0 != nullptr &&
                currentOut->dataBuffer() != nullptr &&
                currentOut->dataBuffer() == input0->dataBuffer()) {
              // View is still valid — same backing DataBuffer. Skip full dispatch.
              currentOut->tickWriteDevice();
              continue;
            }
          }
        }

        auto slotStatus = executeSlot(s, effectiveExternals, numExt, stream);
        if (slotStatus != Status::OK) {
          tl_dspReplayActive = prevReplayActive;
          DSP_DIAG(EXECUTE, "COMPOSITE_REPLAY: gap slot %d FAILED status=%d",
                   s, static_cast<int>(slotStatus));
          return slotStatus;
        }
      }
      tl_dspReplayActive = prevReplayActive;

      // Cross-stream sync after gap ops — no-op when gap stream == cudaStr
      {
        auto* lcStream = LaunchContext::defaultContext()->getCudaStream();
        cudaStream_t gapStream = lcStream ? *lcStream : nullptr;
        if (gapStream != nullptr && gapStream != cudaStr) {
          cudaEvent_t evt = execCtx->crossStreamEvent;
          cudaEventRecord(evt, gapStream);
          cudaStreamWaitEvent(cudaStr, evt, 0);
        }
      }

    } else {  // REPLAY_UNIT_TRITON_ISLAND (unmerged)
      int idx = unit.islandIndex;
      if (idx < 0 || idx >= static_cast<int>(sched.compositeReplayHandles.size()) ||
          !sched.compositeReplayHandles[idx] || !sched.compositeReplayHandles[idx]->isReady()) {
        DSP_DIAG(EXECUTE, "COMPOSITE_REPLAY: island %d handle not ready", idx);
        return Status::KERNEL_FAILURE;
      }

      DSP_DIAG(EXECUTE, "COMPOSITE_REPLAY: island %d [%d-%d] launching", idx, unit.startSlot, unit.endSlot);
      bool launchOk = sched.compositeReplayHandles[idx]->replay(stream);
      if (!launchOk) {
        DSP_DIAG(EXECUTE, "COMPOSITE_REPLAY: island %d launch FAILED", idx);
        return Status::KERNEL_FAILURE;
      }

      // Per-island actuality tick
      for (int s = unit.startSlot; s <= unit.endSlot && s < totalOutputSlots_; s++) {
        NDArray* arr = outputSlots_[s];
        if (arr != nullptr && arr->dataBuffer() != nullptr && !arr->dataBuffer()->isClosed()) {
          arr->tickWriteDevice();
        }
      }

      // Cross-stream sync after island replay
      {
        auto* lcStream = LaunchContext::defaultContext()->getCudaStream();
        cudaStream_t gapStream = lcStream ? *lcStream : nullptr;
        if (gapStream != nullptr && gapStream != cudaStr) {
          cudaEvent_t evt = execCtx->crossStreamEvent;
          cudaEventRecord(evt, cudaStr);
          cudaStreamWaitEvent(gapStream, evt, 0);
        }
      }
    }
  }

  // ── ACTUALITY TICK: mark device data as newer than host after replay ──
  //
  // Graph replay writes device memory directly without registerSpecialUse.
  // Without ticking _writeSpecial here, syncToHost sees equal host/device
  // counters and skips the D2H copy, returning stale host data.
  //
  // This is the canonical tick site — compositeReplay() is the convergence
  // point for all replay units (islands + gaps). Any future replay backend
  // (HIP, Metal, etc.) that bypasses registerSpecialUse must do the same.
  for (int s = seg.def.startSlot; s <= seg.def.endSlot && s < totalOutputSlots_; s++) {
    NDArray* arr = outputSlots_[s];
    if (arr != nullptr && arr->dataBuffer() != nullptr && !arr->dataBuffer()->isClosed()) {
      arr->tickWriteDevice();
    }
  }

  // Diagnostic: check final output after composite replay
  dumpSegFinalArgmax(seg, outputSlots_, totalOutputSlots_, numSlots_, slots_,
                     cudaStr, "POST_COMPOSITE_REPLAY_ARGMAX", seg.exec.executionCount);

  // Update replay tracking
  seg.exec.lastReplayExecCount = seg.exec.executionCount;

  // FORCE_RECAPTURE: invalidate after replay so next step re-captures
  if (Environment::getInstance().tritonForceRecapture()) {
    SegmentLifecycle::invalidateForRebuild(this, seg, "force_recapture_post_composite_replay");
    batchD2DCount_ = 0;
    DSP_DIAG(EXECUTE, "FORCE_RECAPTURE: invalidated after replay execCount=%d",
             seg.exec.executionCount);
  }

  return Status::OK;
}

// ── LRU GRAPH EVICTION ──────────────────────────────────────────────────────
// Evicts captured graphs to free GPU memory. Returns number of graphs evicted.
// When dspLruEviction is true, evicts least-recently-replayed graphs first.
// Otherwise evicts smallest (fewest nodes) first (legacy behavior).
int NativeDynamicShapePlan::evictLruGraphs(int segIdx, size_t neededBytes, void* stream) {
  auto cudaStr = (stream != nullptr) ? *static_cast<cudaStream_t*>(stream) : nullptr;
  bool lruMode = Environment::getInstance().dspLruEviction();
  int maxEvictions = Environment::getInstance().dspCaptureOomMaxRetries();
  int numEvicted = 0;

  for (int evictAttempt = 0; evictAttempt < maxEvictions; evictAttempt++) {
    // Check if we have enough memory already
    size_t gpuFree = 0, gpuTotal = 0;
    cudaMemGetInfo(&gpuFree, &gpuTotal);
    if (gpuFree >= neededBytes) {
      DSP_DIAG(MEMORY, "evictLruGraphs: have enough memory after %d evictions (%zuMB free >= %zuMB needed)",
               numEvicted, gpuFree / (1024*1024), neededBytes / (1024*1024));
      break;
    }

    // Find the best candidate to evict.
    // Composite-captured segments use a workspace-only sentinel as replayHandle
    // (state=EMPTY, isReady()=false) and store actual graphs in compositeReplayHandles[].
    // Ownership:
    //   - monolithic: seg.exec.replayHandle is READY → owns full cudaGraph + cudaGraphExec
    //   - composite:  seg.exec.replayHandle is EMPTY sentinel (workspace-only);
    //                 sched.compositeReplayHandles[i] own individual island cudaGraph+cudaGraphExec
    //   - cleanupSegmentForRebuild() → canonical cleanup of BOTH ownership paths
    int evictIdx = -1;
    if (lruMode) {
      // LRU: find segment with smallest lastReplayExecCount (least recently used)
      int lruExecCount = INT_MAX;
      for (size_t si = 0; si < segments_.size(); si++) {
        if (static_cast<int>(si) == segIdx) continue;
        auto& candidate = segments_[si];
        bool monolithicReady = candidate.exec.replayHandle && candidate.exec.replayHandle->isReady();
        bool compositeReady = hasCompositeHandles(candidate);
        if (!monolithicReady && !compositeReady) continue;
        if (candidate.exec.lastReplayExecCount < lruExecCount) {
          lruExecCount = candidate.exec.lastReplayExecCount;
          evictIdx = static_cast<int>(si);
        }
      }
    } else {
      // Smallest-first: find segment with fewest CUDA graph nodes.
      // For composite segments, sum node counts across all island handles.
      size_t smallestNodes = SIZE_MAX;
      for (size_t si = 0; si < segments_.size(); si++) {
        if (static_cast<int>(si) == segIdx) continue;
        auto& candidate = segments_[si];
        bool monolithicReady = candidate.exec.replayHandle && candidate.exec.replayHandle->isReady();
        bool compositeReady = hasCompositeHandles(candidate);
        if (!monolithicReady && !compositeReady) continue;
        size_t nodeCount = 0;
        if (monolithicReady) {
          auto* cudaReplay = dynamic_cast<CudaGraphReplayHandle*>(candidate.exec.replayHandle.get());
          nodeCount = cudaReplay ? cudaReplay->getNumNodes() : 1;
        } else {
          // Composite: sum node counts across merged + individual handles
          for (auto& h : candidate.exec.compositeReplaySchedule.mergedReplayHandles) {
            if (h && h->isReady()) {
              auto* cr = dynamic_cast<CudaGraphReplayHandle*>(h.get());
              nodeCount += cr ? cr->getNumNodes() : 1;
            }
          }
          for (auto& h : candidate.exec.compositeReplaySchedule.compositeReplayHandles) {
            if (h && h->isReady()) {
              auto* cr = dynamic_cast<CudaGraphReplayHandle*>(h.get());
              nodeCount += cr ? cr->getNumNodes() : 1;
            }
          }
        }
        if (nodeCount == 0) nodeCount = 1;
        if (nodeCount < smallestNodes) {
          smallestNodes = nodeCount;
          evictIdx = static_cast<int>(si);
        }
      }
    }

    if (evictIdx < 0) {
      DSP_DIAG(MEMORY, "evictLruGraphs: no more evictable segments (evicted %d)", numEvicted);
      break;
    }

    // Evict the selected segment via cleanupSegmentForRebuild(), which releases
    // both monolithic replayHandle and composite island handles.
    auto& evictSeg = segments_[evictIdx];
    bool evictIsComposite = hasCompositeHandles(evictSeg);
    DSP_DIAG(MEMORY, "evictLruGraphs: evicting seg[%d-%d] (lruExec=%d, captureMode=%s, algo=%s) "
                     "for seg idx=%d (attempt %d/%d)",
             evictSeg.def.startSlot, evictSeg.def.endSlot, evictSeg.exec.lastReplayExecCount,
             evictIsComposite ? "composite" : "monolithic",
             lruMode ? "LRU" : "smallest", segIdx, evictAttempt + 1, maxEvictions);

    SegmentLifecycle::invalidateForRebuild(this, evictSeg, "lru_eviction");

    numEvicted++;

    // Sync to ensure GPU memory is freed
    if (cudaStr != nullptr) {
      cudaStreamSynchronize(cudaStr);
    }
    cudaGetLastError();
  }

  // Final pool trim after evictions
  if (numEvicted > 0) {
    int deviceId = 0;
    cudaGetDevice(&deviceId);
    memory::CudaMemoryPool::getInstance().trimPool(deviceId);
    DSP_DIAG(MEMORY, "evictLruGraphs: evicted %d segments, trimmed pool on device %d", numEvicted, deviceId);
  }

  return numEvicted;
}


// ── PROACTIVE PRE-CAPTURE MEMORY CLEANUP ───────────────────────────────────
// Called before workspace allocation when about to capture a graph.
// Frees cached-but-unused GPU memory and evicts LRU graphs if needed.
void NativeDynamicShapePlan::proactivePreCaptureMemoryCleanup(GraphSegment& seg, int segIdx, void* stream) {
  auto cudaStr = (stream != nullptr) ? *static_cast<cudaStream_t*>(stream) : nullptr;
  int deviceId = 0;
  cudaGetDevice(&deviceId);

  // 1. Trim CUDA memory pool — cheap, can reclaim hundreds of MB
  DSP_DIAG(MEMORY, "proactive cleanup: trimming pool on device %d for seg[%d-%d]",
           deviceId, seg.def.startSlot, seg.def.endSlot);
  memory::CudaMemoryPool::getInstance().trimPool(deviceId);
  if (cudaStr != nullptr) {
    memory::CudaMemoryPool::getInstance().trimPoolOnStream(deviceId, cudaStr);
  }

  // 2. Check if we have enough memory
  size_t gpuFree = 0, gpuTotal = 0;
  cudaMemGetInfo(&gpuFree, &gpuTotal);

  // Estimate needed: capture workspace + cuBLAS workspace (if not allocated) + safety margin
  size_t neededBytes = 0;
  if (sharedCaptureWorkspace_ == nullptr) {
    neededBytes += TRITON_CAPTURE_WORKSPACE_SIZE;  // 128MB default
  }
  if (cublasWorkspaceBuffer_ == nullptr) {
    neededBytes += Environment::getInstance().dspCublasWorkspaceMb() * 1024ULL * 1024ULL;
  }
  neededBytes += Environment::getInstance().dspGraphMetadataSafetyMb() * 1024ULL * 1024ULL;

  DSP_DIAG(MEMORY, "proactive cleanup: gpuFree=%zuMB, needed=%zuMB (ws=%zuMB, cublas=%zuMB, safety=%dMB) for seg[%d-%d]",
           gpuFree / (1024*1024), neededBytes / (1024*1024),
           (sharedCaptureWorkspace_ == nullptr ? TRITON_CAPTURE_WORKSPACE_SIZE : 0) / (1024*1024),
           (cublasWorkspaceBuffer_ == nullptr ? (size_t)(Environment::getInstance().dspCublasWorkspaceMb()) : 0),
           Environment::getInstance().dspGraphMetadataSafetyMb(),
           seg.def.startSlot, seg.def.endSlot);

  if (gpuFree >= neededBytes) {
    DSP_DIAG(MEMORY, "proactive cleanup: sufficient memory (%zuMB >= %zuMB), no eviction needed",
             gpuFree / (1024*1024), neededBytes / (1024*1024));
    return;
  }

  // 3. LRU eviction
  DSP_DIAG(MEMORY, "proactive cleanup: insufficient memory (%zuMB < %zuMB), starting LRU eviction",
           gpuFree / (1024*1024), neededBytes / (1024*1024));
  int numEvicted = evictLruGraphs(segIdx, neededBytes, stream);

  // 4. Final trim after evictions
  if (numEvicted > 0) {
    memory::CudaMemoryPool::getInstance().trimPool(deviceId);
  }

  // Log final state
  cudaMemGetInfo(&gpuFree, &gpuTotal);
  DSP_DIAG(MEMORY, "proactive cleanup complete: evicted=%d, gpuFree=%zuMB/%zuMB for seg[%d-%d]",
           numEvicted, gpuFree / (1024*1024), gpuTotal / (1024*1024),
           seg.def.startSlot, seg.def.endSlot);
}

Status NativeDynamicShapePlan::segDispatchReplay(
    GraphSegment& seg, NDArray** externalArrays, int numExt, void* stream,
    bool allowTritonCudaGraphReplay,
    bool createValuesStable, bool extAddrsStable,
    LongType segShapeKey, const char* backendName) {

  bool hasComposite = hasCompositeHandles(seg);
  bool compositeReplayReady = hasComposite;

  DSP_DIAG(EXECUTE, "COMPOSITE_REPLAY_READY_CHECK: seg[%d-%d] compositeReplayReady=%d "
                    "hasCompositeHandles=%d compiledBy='%s' execCount=%d",
           seg.def.startSlot, seg.def.endSlot, compositeReplayReady ? 1 : 0,
           hasComposite ? 1 : 0,
           seg.exec.compiledByBackend.empty() ? "(empty)" : seg.exec.compiledByBackend.c_str(),
           seg.exec.executionCount);

  if (allowTritonCudaGraphReplay &&
      compositeReplayReady &&
      seg.exec.cachedShapeKey == segShapeKey &&
      createValuesStable &&
      extAddrsStable) {

    DSP_DIAG(EXECUTE, "COMPOSITE_REPLAY_ENTER: seg[%d-%d] → compositeReplay()",
             seg.def.startSlot, seg.def.endSlot);

    auto replayStatus = compositeReplay(seg, seg.exec.compositeReplaySchedule,
                                        externalArrays, numExt, stream);
    if (replayStatus == Status::OK) {
      seg.exec.executionCount++;
      totalGraphReplays_++;
      if (seg.exec.compiledByBackend.empty()) seg.exec.compiledByBackend = backendName;
      // Structured trace: lifecycle transition to REPLAYING.
      {
        int rlSegIdx = -1;
        for (int si = 0; si < static_cast<int>(segments_.size()); si++) {
          if (&segments_[si] == &seg) { rlSegIdx = si; break; }
        }
        DSP_TRACE_LIFECYCLE(trace_,
                            static_cast<int8_t>(rlSegIdx),
                            static_cast<uint8_t>(seg.exec.lifecycleState),
                            static_cast<uint8_t>(SegmentLifecycleState::REPLAYING),
                            static_cast<uint32_t>(executeCount_));
      }
      // Transition CAPTURED → REPLAYING on the first successful replay.
      // Subsequent replays stay in REPLAYING (many replays per one capture).
      if (seg.exec.lifecycleState == SegmentLifecycleState::CAPTURED) {
        SegmentLifecycle::markReplaying(seg.exec);
      }

      // Assertion 3: Post-replay output non-zero check.
      // After the first capture step (execCount > 1), the last output slot of this
      // segment should contain non-zero values if any real compute happened.
      // An all-zero last-slot at execCount > 1 often indicates:
      //   - Graph captured with zeroed buffers and never wrote real values
      //   - Prezero wiping outputs that were never re-populated
      //   - Stale actuality flags causing the graph to read wrong inputs
      // Gated on EXECUTE diagnostics to avoid overhead in production.
      if (seg.exec.executionCount > 1 &&
          DspDiagnostics::getInstance().isEnabled(DSP_DIAG_EXECUTE)) {
        // Find the last occupied output slot in this segment.
        int lastSlot = -1;
        for (int s = seg.def.endSlot; s >= seg.def.startSlot; s--) {
          if (s < totalOutputSlots_ && outputSlots_[s] != nullptr &&
              outputSlots_[s]->dataBuffer() != nullptr &&
              !outputSlots_[s]->dataBuffer()->isClosed() &&
              outputSlots_[s]->lengthOf() > 0) {
            lastSlot = s;
            break;
          }
        }
        if (lastSlot >= 0) {
          NDArray* checkArr = outputSlots_[lastSlot];
          // Sync to host to inspect values — outside stream capture so safe.
          checkArr->syncToHost();
          bool allZero = true;
          sd::LongType checkLen = sd::math::sd_min((sd::LongType)64, checkArr->lengthOf());
          for (sd::LongType ci = 0; ci < checkLen && allZero; ci++) {
            double v = checkArr->e<double>(ci);
            if (v != 0.0 && v == v) {  // != 0 and not NaN
              allZero = false;
            }
          }
          if (allZero) {
            DSP_DIAG(EXECUTE,
                     "POST_REPLAY_ZERO_OUTPUT_WARNING: seg[%d-%d] execCount=%d "
                     "lastSlot=%d len=%lld — last output slot is all-zero after replay. "
                     "Possible stale-capture or missed-prezero bug.",
                     seg.def.startSlot, seg.def.endSlot, seg.exec.executionCount,
                     lastSlot, (long long)checkArr->lengthOf());
          }
        }
      }
    }
    return replayStatus;
  }

  // Replay conditions not met — caller should fall through to capture/direct
  return Status::MAYBE;
}

// ═══════════════════════════════════════════════════════════════════════════
// SegCaptureCtx — bundles pre-computed state from the preamble that the
// capture dispatch method needs. Defined here (not in header) to avoid
// header bloat and rebuild cascades. The header forward-declares it.
// ═══════════════════════════════════════════════════════════════════════════
struct NativeDynamicShapePlan::SegCaptureCtx {
  // ── Segment identification ─────────────────────────────────────────────
  int segIdx = -1;

  // ── Shape / address keys ───────────────────────────────────────────────
  LongType segShapeKey = 0;
  LongType segInputAddrKey = 0;
  LongType createValueKey = 0;

  // ── Backend references ─────────────────────────────────────────────────
  const char* backendName = nullptr;
  GraphBackend* backend = nullptr;

  // ── CUDA stream ────────────────────────────────────────────────────────
  cudaStream_t cudaStr = nullptr;

  // ── Triton backend ─────────────────────────────────────────────────────
#if HAVE_TRITON
  TritonGraphBackend* tritonBackend = nullptr;
#else
  void* tritonBackend = nullptr;
#endif

  // ── Capture decision flags ─────────────────────────────────────────────
  int captureMinExec = 0;
  bool forceRecaptureEnabled = false;
  bool allowTritonCudaGraphReplay = false;
  bool requiresOrderedGapCapture = false;
  bool hasCudaStream = false;
};

// ═══════════════════════════════════════════════════════════════════════════
// segDispatchCaptureOrDirect — CUDA graph capture + direct (non-capture)
// Triton execution. Contains TritonOrderedRangeGuard RAII, capture decision,
// composite/monolithic capture, and direct fallback.
// ═══════════════════════════════════════════════════════════════════════════
Status NativeDynamicShapePlan::segDispatchCaptureOrDirect(
    GraphSegment& seg, NDArray** externalArrays, int numExt, void* stream,
    SegCaptureCtx& ctx) {

#if HAVE_TRITON
  struct TritonOrderedRangeGuard {
   bool active = false;
   ~TritonOrderedRangeGuard() {
     if (active) TritonGraphBackend::clearOrderedRangeExecutor();
   }
 } tritonOrderedRangeGuard;

 if (ctx.tritonBackend != nullptr) {
   TritonGraphBackend::setOrderedRangeExecutor(
       [this, &seg, externalArrays, numExt, stream](int startSlot, int endSlot) -> Status {
         if (startSlot > endSlot) return Status::OK;

         GraphSegment gapSeg;
         gapSeg.def.startSlot = startSlot;
         gapSeg.def.endSlot = endSlot;
         gapSeg.exec.executionCount = seg.exec.executionCount;
         gapSeg.exec.compilationFailed = seg.exec.compilationFailed;

         // Check if the Triton stream is currently being captured (CUDA graph recording).
         bool streamIsCapturing = false;
         if (stream != nullptr) {
           cudaStreamCaptureStatus capStat = cudaStreamCaptureStatusNone;
           cudaStreamIsCapturing(*static_cast<cudaStream_t*>(stream), &capStat);
           streamIsCapturing = (capStat != cudaStreamCaptureStatusNone);
         }

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

         if (streamIsCapturing) {
           // ── CAPTURE PATH: SKIP gap ops entirely ──
           //
           // During CUDA graph capture, Triton kernels are recorded into the graph.
           // Gap ops (matmul, attention, etc.) must NOT execute because:
           //
           //  1. Executing on the capturing stream bakes stale addresses into the
           //     graph — on replay, gap ops read/write wrong buffers, producing
           //     garbage that accumulates across 30 transformer layers.
           //
           //  2. Executing on a separate stream also fails because native ops
           //     internally use the legacy stream (stream 0) for D2H copies,
           //     allocations, and syncs — all of which are illegal during capture
           //     (error 900/224: "operation would make the legacy stream depend
           //     on a capturing blocking stream").
           //
           // Solution: SKIP gap ops during capture. Warmup already executed them
           // and populated outputSlots_ at the correct addresses. The Triton arg
           // table snapshot will reference these warmup addresses. On replay, the
           // composite replay schedule executes gaps FRESH before graph replay.
           //
           // This is correct because:
           //  - Shapes are frozen (gap output shapes don't change)
           //  - Output buffer addresses are stable (same outputSlots_ from warmup)
           //  - Triton kernels reference buffer addresses via arg tables, which
           //    are refreshed from outputSlots_ before each replay
           //  - The captured graph contains ONLY Triton kernels

           DSP_DIAG(EXECUTE, "GAP_SKIP_DURING_CAPTURE: gap[%d-%d] SKIPPED (warmup outputs "
                    "already at stable addresses) for seg[%d-%d]",
                    startSlot, endSlot, seg.def.startSlot, seg.def.endSlot);

           // gapOpsCapturedInGraph stays false — gaps are NOT in the graph
           return Status::OK;
         }

         // ── NON-CAPTURE PATH: normal gap execution with stream sync ──
         // Triton kernels and gap ops run on different streams. Synchronize
         // to ensure gap ops see completed Triton outputs and vice versa.
         if (!streamsMatch && stream != nullptr) {
           cudaStreamSynchronize(tritonStr);
         }
         bool savedGraphActive = tl_graphExecutionActive;
         tl_graphExecutionActive = false;
         auto gapStatus = executeSegmentSlotBySlot(gapSeg, externalArrays, numExt, stream);
         if (!streamsMatch && gapStr != nullptr) {
           cudaStreamSynchronize(gapStr);
         }
         tl_graphExecutionActive = savedGraphActive;
         return gapStatus;
       });
   tritonOrderedRangeGuard.active = true;
 }

  // ── CAPTURE + DIRECT EXECUTION ─────────────────────────────────────────────
  // This block handles CUDA graph capture and direct (non-capture) Triton
  // execution. It is ~1300 lines and deeply interleaved with preprocessor
  // conditionals, TLS state management, and a TritonOrderedRangeGuard RAII
  // object whose lifetime spans both capture and direct-exec paths.
  //
  // NOT extracted into segDispatchCapture() because:
  //   1. TritonOrderedRangeGuard RAII must span both capture and direct paths
  //   2. ~30 local variables from the preamble are referenced
  //   3. Complex #ifdef nesting makes function boundary extraction error-prone
  //   4. The guard's destructor interleaves with shared_ptr cleanup at scope exit
  //
  // The existing structure is: capture decision → capture body → direct fallback.
  // Each path has multiple early-return sites that deactivate the guard.

  Status status = Status::KERNEL_FAILURE;
  bool usedTritonGraphCapture = false;

  // Recompute shouldCaptureTritonGraph here (same logic as CAPTURE DECISION CHECK above).
  // This is the actual capture point - the diagnostic above just logs the decision.
  bool hasReplayHandleNow = (seg.exec.replayHandle != nullptr);
  bool replayHandleNullNow = (seg.exec.replayHandle == nullptr);
  bool execCountInWindowNow = (seg.exec.executionCount >= ctx.captureMinExec) &&
                              (ctx.forceRecaptureEnabled || seg.exec.executionCount <= (ctx.captureMinExec + 2));
  bool captureWindowSatisfiedNow = execCountInWindowNow || ctx.requiresOrderedGapCapture;
  bool shouldCaptureTritonGraphNow = ctx.allowTritonCudaGraphReplay &&
                                     !hasReplayHandleNow &&
                                     replayHandleNullNow &&
                                     !seg.exec.compilationFailed &&
                                     captureWindowSatisfiedNow &&
                                     ctx.hasCudaStream;
  // OOM retry deferred check: if a previous capture attempt failed with OOM and
  // we haven't reached the retry-after execution count, skip capture for this
  // execution and keep the warmup path active.
  if (seg.exec.captureOomRetries > 0 &&
      seg.exec.executionCount < seg.exec.captureRetryAfterExec) {
    DSP_DIAG_SEG(EXECUTE, seg.def.startSlot,
                 "OOM RETRY DEFERRED: seg[%d-%d] retries=%d execCount=%d retryAfter=%d — warmup path",
                 seg.def.startSlot, seg.def.endSlot, seg.exec.captureOomRetries,
                 seg.exec.executionCount, seg.exec.captureRetryAfterExec);
    shouldCaptureTritonGraphNow = false;
  }

  // Proactive memory cleanup before capture: trim pool, evict LRU graphs if needed.
  if (shouldCaptureTritonGraphNow && Environment::getInstance().dspProactiveEvictBeforeCapture()) {
    proactivePreCaptureMemoryCleanup(seg, ctx.segIdx, stream);
  }

  if (shouldCaptureTritonGraphNow) {
    DSP_DIAG_SEG(COMPILE, seg.def.startSlot,
                 "GRAPH CAPTURE BEGIN: seg[%d-%d] size=%d execCount=%d shapesFrozen=%d",
                 seg.def.startSlot, seg.def.endSlot, seg.def.endSlot - seg.def.startSlot + 1,
                 seg.exec.executionCount, shapesFrozen_ ? 1 : 0);
    seg.exec.gapOpsCapturedInGraph = false;

    // Set up capture workspace BEFORE beginCapture — cudaMalloc must be outside capture.
    // Native ordered range ops (matmul, attention, concat) need temporary buffers during execution.
    // With tl_graphExecutionActive=true, CudaMemoryPool allocates from this workspace
    // instead of calling cudaMallocAsync (which fails during capture).
    // TRITON_CAPTURE_WORKSPACE_SIZE is now at file scope (above).

    // Create the replayHandle BEFORE capture — it must exist to store workspace, host ptrs, etc.
    {
      int deviceId = 0;
      cudaGetDevice(&deviceId);
      seg.exec.replayHandle = GraphReplayFactory::create(deviceId);
    }

    if (seg.exec.replayHandle->getWorkspacePtr() == nullptr) {
      int deviceId = 0;
      cudaGetDevice(&deviceId);

      // Shared workspace: allocate once, reuse across all segments.
      // Segments execute sequentially (cudaGraphLaunch + cudaStreamSynchronize),
      // and workspace offset resets each capture, so sharing is safe.
      if (sharedCaptureWorkspace_ == nullptr) {
        // First segment — allocate the shared workspace
        cudaError_t err = cudaMalloc(&sharedCaptureWorkspace_, TRITON_CAPTURE_WORKSPACE_SIZE);
        if (err != cudaSuccess) {
          cudaGetLastError();
          sharedCaptureWorkspace_ = nullptr;
        }
        if (sharedCaptureWorkspace_ != nullptr) {
          sharedCaptureWorkspaceBytes_ = TRITON_CAPTURE_WORKSPACE_SIZE;
          sharedCaptureWorkspaceDevice_ = deviceId;
          memory::CudaMemoryPool::getInstance().registerCaptureWorkspace(
              sharedCaptureWorkspace_, sharedCaptureWorkspaceBytes_);
          DSP_DIAG_SEG(MEMORY, seg.def.startSlot,
                       "allocated SHARED capture workspace: %zuMB on device %d",
                       TRITON_CAPTURE_WORKSPACE_SIZE / (1024*1024), deviceId);
        } else {
          // Shared allocation failed — ABORT capture for this segment.
          SegmentLifecycle::invalidateForRebuild(this, seg, "oom_shared_workspace");
#if HAVE_TRITON
          // Guard destructor would call clearOrderedRangeExecutor(), which is safe,
          // but deactivate explicitly so the destructor is a no-op on this exit path.
          tritonOrderedRangeGuard.active = false;
          TritonGraphBackend::clearOrderedRangeExecutor();
#endif
          return reportOomError(seg, "shared_workspace_allocation",
                                TRITON_CAPTURE_WORKSPACE_SIZE, deviceId);
        }
      } else {
        DSP_DIAG_SEG(MEMORY, seg.def.startSlot,
                     "using shared workspace for seg[%d-%d]",
                     seg.def.startSlot, seg.def.endSlot);
      }

      // Point this segment's replay handle at the shared workspace
      seg.exec.replayHandle->useExternalWorkspace(
          sharedCaptureWorkspace_, sharedCaptureWorkspaceBytes_);
    }

    // Guard: if replay handle creation failed, crash immediately.
    // Silent fallthrough to slot-by-slot masks the real bug.
    if (seg.exec.replayHandle == nullptr) {
      int deviceId = 0;
      cudaGetDevice(&deviceId);
#if HAVE_TRITON
      tritonOrderedRangeGuard.active = false;
      TritonGraphBackend::clearOrderedRangeExecutor();
#endif
      DSP_THROW_SEG(COMPILE, seg.def.startSlot,
                    "NativeDSP: GraphReplayFactory::create returned nullptr for seg[%d-%d] on device %d. "
                    "Replay handle creation failed — fix the root cause.",
                    seg.def.startSlot, seg.def.endSlot, deviceId);
    } else {
      tl_captureWorkspace = seg.exec.replayHandle->getWorkspacePtr();
      tl_captureWorkspaceSize = seg.exec.replayHandle->getWorkspaceBytes();
      tl_captureWorkspaceOffset = 0;
      tl_capturedHostPtrs.clear();
      tl_captureReplicateCache.clear();

      // Allocate pinned host workspace for H2D source copies during capture.
      // During capture, DataBuffer::syncToSpecial and PointersManager need a persistent
      // pinned buffer as H2D memcpy source. Without this, they use _primaryBuffer directly.
      // Temporary arrays (axis/dimension params for gap ops) get freed after the op completes,
      // but the graph's H2D memcpy node bakes the source address — reading freed memory on
      // launch causes SIGSEGV. The pinned workspace persists for the graph's lifetime.
      void* captureHostWs = nullptr;
      {
        auto hostWsErr = cudaMallocHost(&captureHostWs, TRITON_CAPTURE_HOST_WORKSPACE_SIZE);
        if (hostWsErr != cudaSuccess) {
          cudaGetLastError();
          captureHostWs = nullptr;
          // Host workspace allocation failed — H2D copies during capture will use
          // non-pinned _primaryBuffer directly. When temporary arrays (axis/dimension
          // params for gap ops) are freed after the op completes, the graph's H2D
          // memcpy node still references the freed source address, causing SIGSEGV on
          // replay. This is a fatal error, not a degraded-but-correct path.
          int deviceId = 0;
          cudaGetDevice(&deviceId);
          // No TLS or context cleanup needed — capture hasn't started yet
          restoreCublasWorkspaceAfterCapture(stream);
          SegmentLifecycle::invalidateForRebuild(this, seg, "oom_host_workspace");
#if HAVE_TRITON
          tritonOrderedRangeGuard.active = false;
          TritonGraphBackend::clearOrderedRangeExecutor();
#endif
          DSP_THROW_SEG(COMPILE, seg.def.startSlot,
                        "NativeDSP: cudaMallocHost failed for capture host workspace (%zuMB) "
                        "seg[%d-%d] device %d cudaError=%d (%s). "
                        "Without pinned host workspace, graph replay will SIGSEGV on freed source addresses.",
                        TRITON_CAPTURE_HOST_WORKSPACE_SIZE / (1024*1024),
                        seg.def.startSlot, seg.def.endSlot, deviceId,
                        static_cast<int>(hostWsErr), cudaGetErrorString(hostWsErr));
        } else {
          DSP_DIAG(MEMORY, "allocated %zuMB pinned host workspace for Triton capture seg[%d-%d]",
                   TRITON_CAPTURE_HOST_WORKSPACE_SIZE / (1024*1024), seg.def.startSlot, seg.def.endSlot);
        }
      }
      tl_captureHostWorkspace = captureHostWs;
      tl_captureHostWorkspaceSize = (captureHostWs != nullptr) ? TRITON_CAPTURE_HOST_WORKSPACE_SIZE : 0;
      tl_captureHostWorkspaceOffset = 0;
      // Track the host workspace as a captured host pointer for lifetime management.
      // On successful capture, this moves to the replay handle (addCapturedHostPtr).
      // On failure, it's freed immediately (line below at tl_capturedHostPtrs cleanup).
      if (captureHostWs != nullptr) {
        tl_capturedHostPtrs.push_back(captureHostWs);
      }

      // Set capture stream so captureSafeStreamOrDefault() routes ops to the correct stream
      cudaStream_t prevCaptureStream = tl_graphCaptureStream;
      tl_graphCaptureStream = ctx.cudaStr;
      // Pre-allocate cuBLAS workspace to prevent internal cudaMalloc during capture.
      // cuBLAS internally allocates workspace on stream 0 for GEMM operations. During
      // graph capture on a named stream, this cross-stream allocation breaks capture,
      // producing invalid graph nodes that SIGSEGV on cudaGraphLaunch.
      const size_t CUBLAS_WORKSPACE_SIZE = Environment::getInstance().dspCublasWorkspaceMb() * 1024ULL * 1024ULL;
      ensureCublasWorkspace(CUBLAS_WORKSPACE_SIZE);
      // NOTE: setCublasWorkspaceForCapture is deferred to AFTER warmup (see below).
      // Calling it here sets cublasSetStream_v2 to the capture stream, which causes
      // cuBLAS matmuls in gap ops during warmup to run on tritonStr instead of gapStr.
      // This stream mismatch creates data races: cast ops on gapStr write matmul
      // inputs, but cuBLAS on tritonStr starts before gapStr completes.

      // Reset cast cache indices (NOT full clear) before warmup.
      // Previous segments' warmup entries may still be referenced by their captured
      // graphs — clearCastCache() would delete the NDArrays, causing cudaFreeAsync
      // on the GPU buffers. Those addresses are baked into the captured graph nodes
      // (assign + cuBLAS GEMM), so freeing them causes "illegal memory access" (700)
      // on replay. resetCastCacheIndices() preserves the buffers while letting this
      // segment's warmup reuse or append entries as needed.
      //
      // Note: Shape mismatches from speculative decode (draft vs target model) are
      // handled by the mid-execution clearCastCache() call inside MmulHelper::mmul
      // (lines ~761, 1058), which safely skips frees during capture.
      MmulHelper::resetCastCacheIndices();

      // ── Batch-zero preparation (OUTSIDE capture) ─────────────────────────
      // Collects the set of output buffers that need zeroing before each replay.
      // The pre-capture loop below consumes batchZeroEntries_ via cudaMemsetAsync.
      if (Environment::getInstance().dspBatchZero()) {
        std::unordered_set<int> gapSlots;
        if (Environment::getInstance().dspBatchZeroGapOnly()) {
#if HAVE_TRITON
          auto* tritonBE = dynamic_cast<TritonGraphBackend*>(ctx.backend);
         if (tritonBE != nullptr) {
           gapSlots = tritonBE->getGapSlots(seg, slots_);
         } else
#endif
          {
            for (int s = seg.def.startSlot; s <= seg.def.endSlot; s++) gapSlots.insert(s);
          }
        } else {
          for (int s = seg.def.startSlot; s <= seg.def.endSlot; s++) gapSlots.insert(s);
        }
        collectBatchZeroTargets(gapSlots);
      }

      // Sync external inputs to device before capture — variable inputs use forceSync=true.
      // Weight (non-variable) inputs are skipped: already device-authoritative.
      {
        int syncedCapture = 0, skippedCapture = 0;
        syncExternalInputs(externalArrays, numExt, externalInputIsVariable_,
                           externalInputNames_, shapesFrozen_,
                           Environment::getInstance().tritonVerifyKernels(),
                           "capture", seg.exec.executionCount,
                           &syncedCapture, &skippedCapture);
        DSP_DIAG(MEMORY, "pre-capture EXT_SYNC directReference: %d synced, %d weights skipped (frozen=%d, varFilter=%d)",
                 syncedCapture, skippedCapture, (int)shapesFrozen_,
                 (int)(shapesFrozen_ && !externalInputIsVariable_.empty()));
      }

      // Cross-stream ordering: Java-side assign() runs on the default stream or
      // a LaunchContext stream BEFORE DSP execution starts. syncToDevice() above
      // is a no-op when isSpecialActual()=true (set by tickDeviceWrite after
      // assign), so it doesn't establish ordering between the assign stream and
      // cudaStr. Without this, capture can bake in stale device data from the
      // previous step (assign kernel hasn't completed on its stream yet).
      {
        auto* execCtxCapture = static_cast<PlanExecutionContext*>(activeExecutionContext());
        syncCrossStream(ctx.cudaStr,
                        execCtxCapture ? execCtxCapture->crossStreamEvent : nullptr,
                        "pre-capture", seg.def.startSlot, seg.def.endSlot,
                        seg.exec.executionCount);
      }
      // Clear any sticky CUDA error before capture — stale errors from prior operations
      // (e.g., cudaFuncGetName on driver-API functions) contaminate capture and launch.
      cudaGetLastError();

      // Fingerprint variable inputs after sync (pre-capture).
      fingerprintVariableInputs(externalArrays, numExt, externalInputIsVariable_,
                                externalInputNames_, "capture-ext-sync",
                                seg.exec.executionCount);

      // Configurable: push primary CUDA context during capture.
      // Default OFF — the non-Triton path works without it. Pushing and then popping
      // after capture may cause SIGSEGV on replay (null pointer inside libcuda.so).
      // Enable via ND4J_TRITON_GRAPH_CTX_PUSH=1 for debugging.
      int tritonCaptureDevice = 0;
      cudaGetDevice(&tritonCaptureDevice);
      CUcontext primaryCtx = nullptr;
      CUcontext prevCtx = nullptr;
      bool didPushCtx = pushPrimaryCtxIfConfigured(tritonCaptureDevice, &primaryCtx, &prevCtx);

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
        DSP_DIAG_SEG(EXECUTE, seg.def.startSlot, "Triton pre-capture warmup for seg[%d-%d] execCount=%d",
                     seg.def.startSlot, seg.def.endSlot, seg.exec.executionCount);

        // Sync before warmup to ensure all prior segment work (capture validation
        // replays, gap op execution) is visible on the execution stream. Without
        // this, async GPU work from the previous segment's composite capture may
        // not yet be visible when the current segment's warmup reads outputSlots_
        // written by earlier ops — producing stale/zero inputs.
        if (ctx.hasCudaStream) {
          cudaStreamSynchronize(ctx.cudaStr);
        }

        // Set cuBLAS workspace during warmup too, so cuBLAS selects the same GEMM
        // algorithms as during capture. Without this, warmup may use different
        // algorithms than capture, causing shape/result divergence.
        setCublasWorkspaceForWarmup();

        // Disable frozen fast path for warmup — same rationale as capture below.
        std::vector<NativeSlot::SlotState> savedSlotStateWarmup;
        demoteFrozenSlotStates(slots_, seg.def.startSlot, seg.def.endSlot, savedSlotStateWarmup);

        // Use slot-by-slot for warmup — matches the REF path exactly.
        //
        // CRITICAL: Force host↔device sync during warmup.
        // executeSlot gates prepareSpecialUse/registerSpecialUse on:
        //   needsSync = !shapesFrozen_ || executeCount_ < 2
        // At exec=2, executeCount_=2 → needsSync=false → device coherency
        // calls are skipped entirely. The warmup NEEDS these calls because
        // prior segments' composite captures may have changed actuality flags
        // (validation replay writes device, ticks special-actual but not
        // primary-actual). Without prepareSpecialUse, ops read stale device
        // memory → zero outputs from seg[400+] onwards.
        // forceSync_ overrides the needsSync gate without changing executeCount_.
        Status warmupStatus;
        {
          forceSync_ = true;  // Override needsSync gate in executeSlot

          GraphSegment warmupSeg;
          warmupSeg.def.startSlot = seg.def.startSlot;
          warmupSeg.def.endSlot = seg.def.endSlot;
          warmupSeg.exec.executionCount = seg.exec.executionCount;
          warmupSeg.exec.compilationFailed = seg.exec.compilationFailed;
          warmupStatus = executeSegmentSlotBySlot(warmupSeg, externalArrays, numExt, stream);

          forceSync_ = false;
        }
        restoreSlotStates(slots_, seg.def.startSlot, seg.def.endSlot, savedSlotStateWarmup);

        if (warmupStatus != Status::OK) {
          DSP_DIAG(EXECUTE, "FATAL: Triton pre-capture warmup FAILED for seg[%d-%d] status=%d. "
                            "BLOCKING EXECUTION.",
                   seg.def.startSlot, seg.def.endSlot, static_cast<int>(warmupStatus));
          SegmentLifecycle::markFailed(seg.exec, "pre_capture_warmup_failed");
          // savedSlotStateTriton has not been populated yet (demote happens after warmup).
          // Pass an empty vector — no capture-phase slot demotion to undo here.
          const std::vector<NativeSlot::SlotState> emptySlotState;
          abortCapture(seg, true, didPushCtx, tritonCaptureDevice,
                      prevCaptureStream, emptySlotState, stream);
#if HAVE_TRITON
          // Deactivate guard: abortCapture cleaned up; destructor must not double-clear.
          tritonOrderedRangeGuard.active = false;
          TritonGraphBackend::clearOrderedRangeExecutor();
#endif
          return warmupStatus;
        }
        // Decrement executionCount — the warmup was an extra execution that should
        // not count toward the capture threshold.
        if (seg.exec.executionCount > 0) seg.exec.executionCount--;

        // Synchronize before capture to ensure warmup results are visible
        cudaStreamSynchronize(ctx.cudaStr);
        cudaGetLastError();

        // Reset cast cache indices after warmup so capture starts from index 0.
        // We intentionally preserve the warmup's cast cache entries (NOT full clear).
        // Capture reuses them via assign() — the graph records a cast kernel from the
        // real input to the cached buffer, then cuBLAS reads the cached buffer.
        // clearCastCache() would delete these entries, forcing capture to allocate
        // new ones from the capture workspace. Those workspace sub-allocations
        // cannot be individually freed by cudaFreeAsync (they're interior pointers
        // of the 32MB workspace block), so subsequent clearCastCache() calls
        // corrupt the CUDA memory pool → "illegal memory access" on replay.
        MmulHelper::resetCastCacheIndices();

        DSP_DIAG_SEG(EXECUTE, seg.def.startSlot, "Triton pre-capture warmup DONE for seg[%d-%d]",
                     seg.def.startSlot, seg.def.endSlot);

        // Diagnostic: dump warmup's final output argmax for comparison with replay
        dumpSegFinalArgmax(seg, outputSlots_, totalOutputSlots_, numSlots_, slots_,
                           ctx.cudaStr, "WARMUP_ARGMAX", seg.exec.executionCount);

        // ── RESTORE NULL OUTPUT SLOTS FROM CACHE ─────────────────────────────
        // The warmup execution may clear some outputSlots_ entries (e.g. control
        // flow CF_SWITCH dead outputs, or segment cleanup paths).  The values
        // were captured into outputSlots_ during execution, so restore any
      }

      // DIAGNOSTIC: warmup-only mode — skip capture, use warmup result directly.
      // Enables bisection: if warmup-only produces correct output but capture+replay
      // does not, the bug is in capture/replay.
      {
        static bool warmupOnly = Environment::getInstance().triton().warmupOnly();
        if (warmupOnly) {
          DSP_DIAG(EXECUTE, "WARMUP_ONLY: skipping capture for seg[%d-%d], using warmup result",
                   seg.def.startSlot, seg.def.endSlot);
          SegmentLifecycle::markFailed(seg.exec, "warmup_only_mode");
          // savedSlotStateTriton has not been populated yet (demote happens after warmup).
          // Pass an empty vector — no capture-phase slot demotion to undo here.
          const std::vector<NativeSlot::SlotState> emptySlotState;
          abortCapture(seg, true, didPushCtx, tritonCaptureDevice,
                      prevCaptureStream, emptySlotState, stream);
#if HAVE_TRITON
          // Deactivate guard: abortCapture cleaned up; destructor must not double-clear.
          tritonOrderedRangeGuard.active = false;
          TritonGraphBackend::clearOrderedRangeExecutor();
#endif
          return Status::OK;
        }
      }

      // NOW set cuBLAS handle to capture stream — AFTER warmup completed.
      // During warmup, gap ops must use their default stream (gapStr) for cuBLAS.
      // Only during actual capture do we switch cuBLAS to tritonStr so GEMM nodes
      // are recorded into the CUDA graph on the correct stream.
      setCublasWorkspaceForCapture(stream);

      // cuBLAS workspace preservation during capture.
      //
      //  Once shapes are frozen (shapesFrozen_ == true), NEVER zero the cuBLAS workspace.
      // During capture, cuBLAS stores plan/descriptor data in the workspace. Captured CUDA graphs
      // inherit these cached plans and omit H2D re-upload nodes. Zeroing the workspace destroys
      // cached plans, causing GEMM kernels to read zeros and hang on replay.
      //
      // The workspace content must be preserved across ALL captures and replays once frozen.
      // cuBLAS plans are stable for fixed shapes, so preservation is safe.
      //
      // Pre-frozen (shapes not yet frozen): zeroing is acceptable as no graphs are captured yet.
      if (shapesFrozen_ && cublasWorkspaceBuffer_ != nullptr && cublasWorkspaceSize_ > 0) {
        DSP_DIAG(MEMORY, "pre-capture: cuBLAS workspace PRESERVED (%zuMB) — shapes frozen, plans stable",
                 cublasWorkspaceSize_ / (1024*1024));
        // Do NOT zero — preserve cuBLAS plan data for captured graph replay
      }
      // Note: Pre-frozen zeroing removed entirely — not needed for correctness and
      // adds unnecessary overhead. cuBLAS handles uninitialized workspace correctly.

      // Disable frozen fast path during capture. Same rationale as non-Triton path:
      // capture may re-create views, and the frozen context has stale input/output pointers
      // from the prior non-capture execution. Using the full (non-frozen) path during capture
      // is a one-time cost — all context pointers are properly reconfigured with capture-time
      // arrays, including correct nullify() calls to zero output buffers.
      // Save and restore frozenContextReady after capture so replay uses frozen fast path.
      std::vector<NativeSlot::SlotState> savedSlotStateTriton;
      demoteFrozenSlotStates(slots_, seg.def.startSlot, seg.def.endSlot, savedSlotStateTriton);

      std::vector<std::pair<int, NDArray*>> savedExtForCapture;
      std::vector<std::pair<int, NDArray*>> savedSlotsForCapture;

      // Pre-capture promotion: view-capable slots should enter FROZEN state BEFORE
      // capture begins. The view installation path reuses the input's DataBuffer
      // directly, which is correct for BOTH constant and variable inputs:
      // - For constant inputs: the view sees the stable warmup value
      // - For variable inputs: the view shares the input's DataBuffer, so replay
      //   automatically sees updated values when the input is refreshed
      //
      // Without this promotion, view-capable slots go through normal execution
      // during capture: they allocate new output buffers and create H2D capture
      // nodes. Later, those H2D nodes can copy stale host data over the output,
      // corrupting the values downstream consumers depend on.
      //
      // isDataDependent is NOT a disqualifier: a reshape's output shape comes from
      // input values, but the view still shares the input's DataBuffer correctly.
      if (shapesFrozen_) {
        int promoted = promoteViewCapableSlotsToFrozen(slots_, seg.def.startSlot, seg.def.endSlot);
        if (promoted > 0) {
          DSP_DIAG(EXECUTE, "pre-capture view promotion: %d view-capable slots promoted to FROZEN for seg[%d-%d]",
                   promoted, seg.def.startSlot, seg.def.endSlot);
        }
      }

      // Pre-capture batch-zero: zero all registered buffers BEFORE beginCapture.
      // These cudaMemsetAsync calls execute normally on the stream (not captured).
      //
      // IMPORTANT: Only for MONOLITHIC capture. For COMPOSITE capture, skip batch-zero
      // here because composite capture re-executes gap ops between islands, and those
      // gap ops need valid intermediate results from the warmup as inputs. Batch-zero
      // would destroy those intermediate values (zeroing gap op input buffers), causing
      // gap ops to read zeros and produce wrong results that propagate through the
      // entire model. Composite replay handles zeroing correctly: pre-replay batch-zero
      // zeros outputs before each replay, and gap ops call nullify() on their own outputs.
      bool willUseCompositeCapture = false;
#if HAVE_TRITON
      {
       auto& schedCheck = seg.exec.compositeReplaySchedule;
       for (auto& u : schedCheck.units) {
         if (u.kind == REPLAY_UNIT_TRITON_ISLAND) { willUseCompositeCapture = true; break; }
       }
     }
#endif
      if (Environment::getInstance().dspBatchZero() && !batchZeroEntries_.empty() &&
          !willUseCompositeCapture) {
        for (auto& entry : batchZeroEntries_) {
          if (entry.ptr != nullptr && entry.bytes > 0) {
            cudaMemsetAsync(entry.ptr, 0, entry.bytes, ctx.cudaStr);
          }
        }
        DSP_DIAG(MEMORY, "pre-capture batch-zero: %d buffers zeroed via cudaMemsetAsync (fill engines, before beginCapture)",
                 static_cast<int>(batchZeroEntries_.size()));
      } else if (willUseCompositeCapture) {
        DSP_DIAG(MEMORY, "pre-capture batch-zero SKIPPED for composite capture — gap ops need valid warmup data as inputs");
      }

      // ── Save warmup output slot pointers BEFORE capture ─────────────────
      // Gap ops are skipped during capture (they return OK without executing).
      // outputSlots_[] retains warmup values throughout capture — no save/restore needed.
      // Downstream segments see valid warmup data. Triton sub-kernel arg tables
      // reference warmup addresses, which are stable.

      // POST-ALLOCATION MEMORY GATE: workspace + cuBLAS are allocated. Check that
      // enough headroom remains for CUDA driver graph metadata before starting capture.
      // This is tight and accurate — only graph metadata overhead remains.
      {
        size_t gpuFree = 0, gpuTotal = 0;
        cudaMemGetInfo(&gpuFree, &gpuTotal);
        size_t safetyBytes = Environment::getInstance().dspGraphMetadataSafetyMb() * 1024ULL * 1024ULL;
        if (gpuFree < safetyBytes) {
          int deviceId = 0;
          cudaGetDevice(&deviceId);
          DSP_DIAG_SEG(MEMORY, seg.def.startSlot,
                       "POST-ALLOC GATE FAILED: free=%zuMB < safety=%zuMB for seg[%d-%d]",
                       gpuFree / (1024*1024), safetyBytes / (1024*1024),
                       seg.def.startSlot, seg.def.endSlot);
          SegmentLifecycle::invalidateForRebuild(this, seg, "oom_post_alloc_gate");
#if HAVE_TRITON
          // Deactivate guard: cleanup done; destructor must not double-clear.
          tritonOrderedRangeGuard.active = false;
          TritonGraphBackend::clearOrderedRangeExecutor();
#endif
          return reportOomError(seg, "post_alloc_gate", safetyBytes, deviceId);
        }
      }

      // ── STAGING BUFFERS FOR COMPOSITE CAPTURE ───────────────────────────────
      // Ensure plan-owned stable staging buffers exist and are populated with the
      // current external inputs BEFORE capture begins. This ensures that when gap
      // ops are captured into the merged CUDA graph, the device addresses baked
      // into the graph are stable (plan-lifetime) staging addresses rather than
      // Java-side NDArray addresses (which can be reallocated between steps).
      //
      // At replay time, compositeReplay() calls ensureAndSyncStagingBuffers() which
      // D2D-copies fresh external input data into these same staging buffers.  The
      // merged CUDA graph therefore always reads up-to-date data from its baked-in
      // staging addresses — there is no stale-address risk.
      //
      // syncExternalInputs() above already synced external inputs to device.
      // ensureAndSyncStagingBuffers() here creates the staging NDArrays (one-time,
      // outside capture) and D2D-copies the synced device data into them (async on
      // the stream, also outside capture).  After this call, effectiveExternals_
      // is populated and effectiveExternalsForCapture is valid for the entire
      // composite capture block below.
      NDArray** effectiveExternalsForCapture = externalArrays;
#if HAVE_TRITON
      {
        NDArray** staged = ensureAndSyncStagingBuffers(externalArrays, numExt, stream);
        if (staged != nullptr) {
          effectiveExternalsForCapture = staged;
        }
        DSP_DIAG(MEMORY, "pre-composite-capture staging: effectiveExternalsForCapture=%p "
                 "(externalArrays=%p, staged=%p) seg[%d-%d]",
                 (void*)effectiveExternalsForCapture, (void*)externalArrays,
                 (void*)staged, seg.def.startSlot, seg.def.endSlot);
      }
#endif

      // ── MERGED COMPOSITE CAPTURE: island merging through capture-safe gaps ──
      // When a segment has interleaved gap ops between Triton islands, we merge
      // adjacent islands through capture-safe gaps (gaps where all ops launch CUDA
      // kernels). This produces fewer, larger merged CUDA graphs. View-only gaps
      // (reshape, expand_dims, etc.) break the merge — they're run natively.
      // Original per-island approach (for reference):
      //   1. Island A is captured → CudaGraphReplayHandle stored in compositeReplayHandles[0]
      //   2. Gap ops between A and B execute natively (fresh each replay)
      //   3. Island B is captured → compositeReplayHandles[1]
      //   ...repeat...
      // Replay then follows the schedule in program order:
      //   replay(island_A) → executeSlots(gap_B) → replay(island_C) → ...
      // This preserves data dependencies: gap_B reads fresh island_A output.
      // Declared before the #if block so it's visible to the monolithic path's guard.
      bool didCompositeCapture = false;
#if HAVE_TRITON
      {
       auto& sched = seg.exec.compositeReplaySchedule;
       bool hasIslandUnits = false;
       for (auto& u : sched.units) {
         if (u.kind == REPLAY_UNIT_TRITON_ISLAND) { hasIslandUnits = true; break; }
       }
       if (hasIslandUnits && !sched.units.empty()) {
         DSP_DIAG(EXECUTE, "COMPOSITE_CAPTURE_BEGIN: seg[%d-%d] — per-island capture for %d units",
                  seg.def.startSlot, seg.def.endSlot, static_cast<int>(sched.units.size()));
         // Resize compositeReplayHandles to accommodate all TRITON_ISLAND units.
         // islandIndex in TRITON_ISLAND units is the index into this vector.
         int maxIslandIdx = -1;
         for (auto& u : sched.units) {
           if (u.kind == REPLAY_UNIT_TRITON_ISLAND && u.islandIndex > maxIslandIdx) {
             maxIslandIdx = u.islandIndex;
           }
         }
         if (maxIslandIdx >= 0) {
           sched.compositeReplayHandles.resize(maxIslandIdx + 1);
         }

         bool allIslandsOk = true;
         int deviceId = 0;
         cudaGetDevice(&deviceId);

         // ── Merged island capture ────────────────────────────────────────
         // Instead of capturing each island individually, extend captures
         // through capture-safe gaps (gaps where ALL slots launch CUDA
         // kernels). This produces fewer, larger CUDA graphs — reducing
         // cudaGraphLaunch calls and eliminating gap dispatch overhead.
         //
         // Flow: begin capture at first island → if next gap is capture-safe,
         // keep capture active and run gap ops on capture stream → continue
         // to next island → ... → end capture when a non-capture-safe gap
         // or end-of-schedule is reached.
         //
         // Safety: capture-safe gaps have no view ops, identity ops, or
         // frozen constants. Their ops (matmul, cast, gather, etc.) launch
         // real CUDA kernels captured into the graph. The capture workspace
         // bump allocator ensures no buffer overlap. tl_dspGapStream
         // directs cuBLAS to the capture stream.

         bool captureActive = false;
         std::unique_ptr<GraphReplayHandle> mergedHandle;
         sd::cuda::CudaGraphHandle* mergedNativeHandle = nullptr;
         int mergedGroupId = -1;          // Current merged group index
         int mergedLeaderUnitIdx = -1;    // Unit index of the group leader
         int mergedStartSlot = INT_MAX;   // Slot range of the entire merged group
         int mergedEndSlot = INT_MIN;

         for (size_t unitIdx = 0; unitIdx < sched.units.size() && allIslandsOk; unitIdx++) {
           auto& unit = sched.units[unitIdx];

           if (unit.kind == REPLAY_UNIT_GAP) {
             if (captureActive && unit.isCaptureSafe) {
               // ── MERGED CAPTURE: gap ops recorded on capture stream ──────
               // tl_graphExecutionActive is already true from the preceding island.
               // tl_dspGapStream = ctx.cudaStr makes cuBLAS et al. use capture stream.
               // forceSync_ = true for prepareSpecialUse correctness at exec=2.
               DSP_DIAG(EXECUTE, "MERGED_CAPTURE: gap [%d-%d] mergedGroup=%d — recording on capture stream",
                        unit.startSlot, unit.endSlot, mergedGroupId);

               // Direct gap-stream to capture stream so cuBLAS records here
               cudaStream_t prevGapStream = tl_dspGapStream;
               tl_dspGapStream = ctx.cudaStr;
               forceSync_ = true;

               bool gapOk = true;
               for (int s = unit.startSlot; s <= unit.endSlot; s++) {
                 // Use effectiveExternalsForCapture (staging buffers) so the CUDA graph
                 // bakes in stable plan-owned device addresses, not Java-side pointers
                 // that may be reallocated between steps.
                 auto gapStatus = executeSlot(s, effectiveExternalsForCapture, numExt, stream);
                 if (gapStatus != Status::OK) {
                   DSP_DIAG(EXECUTE, "MERGED_CAPTURE: gap slot %d FAILED status=%d",
                            s, static_cast<int>(gapStatus));
                   gapOk = false;
                   break;
                 }
               }

               forceSync_ = false;
               tl_dspGapStream = prevGapStream;

               if (!gapOk) {
                 // Gap op failed during capture — abort merged capture
                 tl_graphExecutionActive = false;
                 tl_islandSlotMin = INT_MAX;
                 tl_islandSlotMax = INT_MIN;
                 if (mergedNativeHandle->isCapturing()) {
                   mergedNativeHandle->endCapture(ctx.cudaStr);
                 }
                 allIslandsOk = false;
                 break;
               }

               // Extend merged range to include this gap
               if (unit.endSlot > mergedEndSlot) mergedEndSlot = unit.endSlot;

               // Tag this unit as part of the merged group (non-leader, skipped in replay)
               unit.mergedGroupId = mergedGroupId;
               unit.isMergedLeader = false;

               continue;  // Stay in capture — check next unit
             }

             // Gap is NOT capture-safe or no capture active — run natively.
             // If capture was active, finalize the merged capture first.
             if (captureActive) {
               // End the merged capture
               tl_graphExecutionActive = false;
               tl_islandSlotMin = INT_MAX;
               tl_islandSlotMax = INT_MIN;

               bool endOk = mergedNativeHandle->endCapture(ctx.cudaStr);
               size_t nodeCount = endOk ? mergedNativeHandle->getNumNodes() : 0;
               DSP_DIAG(EXECUTE, "MERGED_CAPTURE_END: group=%d [%d-%d] endCapture=%d nodes=%zu",
                        mergedGroupId, mergedStartSlot, mergedEndSlot, endOk ? 1 : 0, nodeCount);

               if (endOk && nodeCount > 0) {
                 if (!validateAndStoreMergedCapture("MERGED_CAPTURE", mergedNativeHandle,
                         mergedHandle, sched, mergedGroupId, mergedStartSlot, mergedEndSlot,
                         nodeCount, stream, ctx.cudaStr)) {
                   allIslandsOk = false;
                 }
               } else {
                 DSP_DIAG(EXECUTE, "MERGED_CAPTURE: group=%d endCapture failed or 0 nodes", mergedGroupId);
                 allIslandsOk = false;
               }
               captureActive = false;
               mergedHandle.reset();
               mergedNativeHandle = nullptr;

               if (!allIslandsOk) break;
             }

             // Execute gap slots natively (not captured — non-capture-safe gap).
             // Use effectiveExternalsForCapture for consistency with the captured path:
             // all gap ops (captured or native) read from the same stable staging buffers.
             DSP_DIAG(EXECUTE, "COMPOSITE_CAPTURE: gap unit [%d-%d] — executing slots natively",
                      unit.startSlot, unit.endSlot);
             forceSync_ = true;
             for (int s = unit.startSlot; s <= unit.endSlot; s++) {
               auto gapStatus = executeSlot(s, effectiveExternalsForCapture, numExt, stream);
               if (gapStatus != Status::OK) {
                 forceSync_ = false;
                 DSP_DIAG(EXECUTE, "COMPOSITE_CAPTURE: gap slot %d FAILED status=%d",
                          s, static_cast<int>(gapStatus));
                 allIslandsOk = false;
                 break;
               }
             }
             forceSync_ = false;

             // Cross-stream sync after native gap ops
             {
               auto* lcStream = LaunchContext::defaultContext()->getCudaStream();
               cudaStream_t gapStream = lcStream ? *lcStream : nullptr;
               if (gapStream != nullptr && gapStream != ctx.cudaStr) {
                 auto* execCtxMergeCap = static_cast<PlanExecutionContext*>(activeExecutionContext());
                 cudaEvent_t evt = execCtxMergeCap ? execCtxMergeCap->crossStreamEvent : nullptr;
                 if (evt != nullptr) {
                   cudaEventRecord(evt, gapStream);
                   cudaStreamWaitEvent(ctx.cudaStr, evt, 0);
                 }
               }
             }
           } else {  // REPLAY_UNIT_TRITON_ISLAND
             int islandIdx = unit.islandIndex;

             if (!captureActive) {
               // ── Begin new merged capture ──────────────────────────────────
               DSP_DIAG(EXECUTE, "MERGED_CAPTURE_BEGIN: island %d [%d-%d] — starting new merged group",
                        islandIdx, unit.startSlot, unit.endSlot);

               auto newHandle = GraphReplayFactory::create(deviceId);
               if (!newHandle) {
                 DSP_DIAG(EXECUTE, "MERGED_CAPTURE: GraphReplayFactory::create failed for island %d", islandIdx);
                 allIslandsOk = false;
                 break;
               }
               newHandle->useExternalWorkspace(sharedCaptureWorkspace_, sharedCaptureWorkspaceBytes_);
               tl_captureWorkspace = newHandle->getWorkspacePtr();
               tl_captureWorkspaceSize = newHandle->getWorkspaceBytes();
               tl_captureWorkspaceOffset = 0;

               mergedHandle = std::move(newHandle);
               auto* cudaReplay = static_cast<CudaGraphReplayHandle*>(mergedHandle.get());
               mergedNativeHandle = cudaReplay->getNativeHandle();

               mergedGroupId = static_cast<int>(sched.mergedReplayHandles.size());
               mergedLeaderUnitIdx = static_cast<int>(unitIdx);
               mergedStartSlot = unit.startSlot;
               mergedEndSlot = unit.endSlot;

               // Tag this unit as the merged group leader
               unit.mergedGroupId = mergedGroupId;
               unit.isMergedLeader = true;

               // Also update compositeReplayHandles — leader's island still indexed
               // (but won't be used during replay because merged handle supersedes it)

               DSP_DIAG(MEMORY, "MERGED_CAPTURE: island %d — external inputs already device-actual "
                        "via syncToDevice+readSpecial (NO writeSpecial poisoning)", islandIdx);

               bool beginOk = mergedNativeHandle->beginCapture(ctx.cudaStr, cudaStreamCaptureModeRelaxed);
               if (!beginOk) {
                 DSP_DIAG(EXECUTE, "MERGED_CAPTURE: island %d beginCapture FAILED", islandIdx);
                 mergedHandle.reset();
                 mergedNativeHandle = nullptr;
                 allIslandsOk = false;
                 break;
               }
               tl_graphExecutionActive = true;
               captureActive = true;
             } else {
               // ── Extend existing merged capture to this island ──────────────
               DSP_DIAG(EXECUTE, "MERGED_CAPTURE: extending to island %d [%d-%d] mergedGroup=%d",
                        islandIdx, unit.startSlot, unit.endSlot, mergedGroupId);
               if (unit.endSlot > mergedEndSlot) mergedEndSlot = unit.endSlot;

               // Tag this unit as part of the merged group (non-leader)
               unit.mergedGroupId = mergedGroupId;
               unit.isMergedLeader = false;
             }

             // Set island filter range and execute this island's Triton sub-kernels.
             // Use effectiveExternalsForCapture so Triton arg tables are built with
             // staging buffer addresses at capture time.  At replay time,
             // refreshArgTablesForReplay() overwrites these with current staging
             // addresses (same pointers, updated content via D2D), keeping the
             // arg tables consistent with what the merged gap nodes read.
             tl_islandSlotMin = unit.startSlot;
             tl_islandSlotMax = unit.endSlot;

             auto captureStatus = ctx.backend->executeSegment(seg, slots_, effectiveExternalsForCapture, numExt,
                                                          outputSlots_, totalOutputSlots_, stream);

             // Check if next unit is a capture-safe gap — if so, keep capture active
             bool keepCaptureOpen = false;
             if (captureStatus == Status::OK && unitIdx + 1 < sched.units.size()) {
               auto& nextUnit = sched.units[unitIdx + 1];
               if (nextUnit.kind == REPLAY_UNIT_GAP && nextUnit.isCaptureSafe) {
                 // Next gap can be captured — don't end capture yet
                 keepCaptureOpen = true;
               }
               // If next unit is another island with no gap between, extend too
               if (nextUnit.kind == REPLAY_UNIT_TRITON_ISLAND) {
                 keepCaptureOpen = true;
               }
             }

             if (captureStatus == Status::OK && keepCaptureOpen) {
               // Keep capture active — tl_graphExecutionActive stays true
               // tl_islandSlotMin/Max will be updated when the next island is processed
               DSP_DIAG(EXECUTE, "MERGED_CAPTURE: island %d — keeping capture open for next unit",
                        islandIdx);
               continue;
             }

             // Either capture failed or this is the last unit / next gap is unsafe → end capture
             tl_graphExecutionActive = false;
             tl_islandSlotMin = INT_MAX;
             tl_islandSlotMax = INT_MIN;

             if (captureStatus != Status::OK) {
               if (mergedNativeHandle->isCapturing()) {
                 mergedNativeHandle->endCapture(ctx.cudaStr);
               }
               allIslandsOk = false;
               captureActive = false;
               mergedHandle.reset();
               mergedNativeHandle = nullptr;
               break;
             }

             // End merged capture, instantiate, validate
             bool endOk = mergedNativeHandle->endCapture(ctx.cudaStr);
             size_t nodeCount = endOk ? mergedNativeHandle->getNumNodes() : 0;
             DSP_DIAG(EXECUTE, "MERGED_CAPTURE_END: group=%d [%d-%d] endCapture=%d nodes=%zu",
                      mergedGroupId, mergedStartSlot, mergedEndSlot, endOk ? 1 : 0, nodeCount);

             if (endOk && nodeCount > 0) {
               if (!validateAndStoreMergedCapture("MERGED_CAPTURE", mergedNativeHandle,
                       mergedHandle, sched, mergedGroupId, mergedStartSlot, mergedEndSlot,
                       nodeCount, stream, ctx.cudaStr)) {
                 allIslandsOk = false;
               }
             } else {
               if (endOk && nodeCount == 0) {
                 DSP_DIAG(EXECUTE, "MERGED_CAPTURE: group=%d has 0 nodes — non-capturable", mergedGroupId);
               }
               allIslandsOk = false;
             }

             captureActive = false;
             mergedHandle.reset();
             mergedNativeHandle = nullptr;
           }
         }  // end for each unit

         // If capture still active at end of schedule, finalize it
         if (captureActive) {
           tl_graphExecutionActive = false;
           tl_islandSlotMin = INT_MAX;
           tl_islandSlotMax = INT_MIN;

           bool endOk = mergedNativeHandle->endCapture(ctx.cudaStr);
           size_t nodeCount = endOk ? mergedNativeHandle->getNumNodes() : 0;
           DSP_DIAG(EXECUTE, "MERGED_CAPTURE_END_TAIL: group=%d [%d-%d] endCapture=%d nodes=%zu",
                    mergedGroupId, mergedStartSlot, mergedEndSlot, endOk ? 1 : 0, nodeCount);

           if (endOk && nodeCount > 0) {
             if (!validateAndStoreMergedCapture("MERGED_CAPTURE_TAIL", mergedNativeHandle,
                     mergedHandle, sched, mergedGroupId, mergedStartSlot, mergedEndSlot,
                     nodeCount, stream, ctx.cudaStr)) {
               allIslandsOk = false;
             }
           } else {
             allIslandsOk = false;
           }
           captureActive = false;
           mergedHandle.reset();
           mergedNativeHandle = nullptr;
         }

         if (allIslandsOk) {
           // All merged groups captured successfully.
           // seg.exec.replayHandle is already created above (the sentinel).
           // Merged replay uses mergedReplayHandles indexed by mergedGroupId.

           // Record the cast-cache high-water mark.  During capture, merged
           // gap matmuls consumed tl_castIdxA / tl_castIdxB slots.  Those
           // slots contain device pointers baked into the merged CUDA graphs.
           // At replay time, unmerged gap matmuls must NOT reuse those slots —
           // they must start from the high-water mark instead of 0.
           {
             auto [hwmA, hwmB] = MmulHelper::getCastCacheHighWaterMark();
             sched.mergedCastHwmA = hwmA;
             sched.mergedCastHwmB = hwmB;
             DSP_DIAG(EXECUTE,
                      "MERGED_CAPTURE_COMPLETE: cast-cache HWM A=%zu B=%zu — "
                      "unmerged gap matmuls will start at these indices during replay",
                      hwmA, hwmB);
           }

           status = Status::OK;
           usedTritonGraphCapture = true;
           didCompositeCapture = true;
           seg.exec.cachedShapeKey = ctx.segShapeKey;
           snapshotExternalAddrs(seg, externalArrays, numExt);
           seg.exec.replayUnitCount = static_cast<int>(sched.units.size());
           SegmentLifecycle::markCaptured(seg.exec, ctx.segInputAddrKey, ctx.createValueKey,
               computeSlotAddrHash(outputSlots_, seg.def.startSlot, seg.def.endSlot, totalOutputSlots_),
               ctx.backendName);

           // Structured trace: record successful composite capture.
           DSP_TRACE_GRAPH_CAPTURED(trace_, static_cast<int8_t>(ctx.segIdx),
                                    static_cast<uint32_t>(executeCount_),
                                    seg.def.startSlot, seg.def.endSlot,
                                    static_cast<uint64_t>(sched.units.size()));

           int mergedGroupCount = static_cast<int>(sched.mergedReplayHandles.size());
           int unmergedUnitCount = 0;
           for (auto& u : sched.units) {
             if (u.mergedGroupId < 0) unmergedUnitCount++;
           }
           DSP_DIAG(EXECUTE, "MERGED_CAPTURE_COMPLETE: seg[%d-%d] %d merged groups, %d unmerged units, "
                    "%d total units",
                    seg.def.startSlot, seg.def.endSlot, mergedGroupCount, unmergedUnitCount,
                    static_cast<int>(sched.units.size()));

           // Diagnostic: check final output after composite capture
           dumpSegFinalArgmax(seg, outputSlots_, totalOutputSlots_, numSlots_, slots_,
                              ctx.cudaStr, "POST_COMPOSITE_CAPTURE_ARGMAX", seg.exec.executionCount);

           // No actuality reset needed — writeSpecial() is no longer called during
           // capture, so external input actuality flags remain in their natural bi-actual
           // state (isPrimaryActual()=true AND isSpecialActual()=true via readSpecial).
           DSP_DIAG(MEMORY, "MERGED_CAPTURE: no actuality reset needed — writeSpecial not called");

           // ── Post-merge slot dispatch reconciliation ────────────────
           // Slot-level dispatch tables (batchedGemmGroups_, slotToBatchedGemmGroup_)
           // are built before merging and can span units that are now in different
           // merged groups. Reconcile them with the final merge state so merged
           // slots aren't dispatched twice and unmerged slots aren't orphaned.
           reconcileSlotDispatchAfterMerge(sched);

           // ── Cleanup after successful merged capture ───────────────────
           // Captured host ptrs: merged graphs may contain H2D memcpy nodes whose
           // source addresses point into the pinned host workspace. Move ownership
           // to the first merged handle so pinned memory persists for graph lifetime.
           if (!tl_capturedHostPtrs.empty() && !sched.mergedReplayHandles.empty() &&
               sched.mergedReplayHandles[0] != nullptr) {
             for (auto* ptr : tl_capturedHostPtrs) {
               sched.mergedReplayHandles[0]->addCapturedHostPtr(ptr);
             }
             DSP_DIAG(MEMORY, "MERGED_CAPTURE: preserved %zu pinned host ptrs on mergedGroup[0]",
                      sched.mergedReplayHandles[0]->getCapturedHostPtrs().size());
           }
           cleanupCaptureTlsState(false, prevCaptureStream);  // false = do NOT free host ptrs
           popPrimaryCtxIfPushed(didPushCtx, tritonCaptureDevice);
           restoreCublasWorkspaceAfterCapture(stream);
           restoreSlotStates(slots_, seg.def.startSlot, seg.def.endSlot, savedSlotStateTriton);

           // FORCE_RECAPTURE: invalidate graph immediately after composite capture+launch
           // so the NEXT step also re-captures instead of replaying the just-captured graph.
           // Without this, composite captures persist and the next step enters compositeReplay()
           // instead of re-capturing — defeating the purpose of FORCE_RECAPTURE.
           if (Environment::getInstance().tritonForceRecapture()) {
             SegmentLifecycle::invalidateForRebuild(this, seg, "force_recapture_post_composite_capture");
             batchD2DCount_ = 0;
             DSP_DIAG(EXECUTE, "FORCE_RECAPTURE: invalidated after COMPOSITE capture+launch execCount=%d",
                      seg.exec.executionCount);
           }
           // Deactivate guard: composite capture done, executor no longer needed.
           // Prevents destructor from interleaving with unrelated code at function exit.
           tritonOrderedRangeGuard.active = false;
           TritonGraphBackend::clearOrderedRangeExecutor();
         } else {
           // Partial failure — free any successfully captured merged handles
           for (auto& h : sched.mergedReplayHandles) {
             h.reset();
           }
           sched.mergedReplayHandles.clear();
           // Clear merged group tags on units
           for (auto& u : sched.units) {
             u.mergedGroupId = -1;
             u.isMergedLeader = false;
           }
           // Mark segment as non-capturable to avoid repeated failed attempts
           SegmentLifecycle::markFailed(seg.exec, "merged_capture_failed");
           // freeHostPtrs=false: composite handles own the pinned host ptrs
           abortCapture(seg, false, didPushCtx, tritonCaptureDevice,
                       prevCaptureStream, savedSlotStateTriton, stream);
           tritonOrderedRangeGuard.active = false;
           TritonGraphBackend::clearOrderedRangeExecutor();
           DSP_DIAG(EXECUTE, "COMPOSITE_CAPTURE_FAILED: seg[%d-%d] — marking non-capturable",
                    seg.def.startSlot, seg.def.endSlot);
           // Fall through to slot-by-slot execution for this step
           status = Status::KERNEL_FAILURE;
         }
       }
     }
#endif  // HAVE_TRITON

  if (!didCompositeCapture) {
    // ── MONOLITHIC CAPTURE (non-composite segments only) ──
    // Note: didCompositeCapture is false when:
    //   - HAVE_TRITON && SD_CUDA is not defined (no Triton composite path)
    //   - No island units existed in the composite schedule
    //   - Composite capture failed (guard deactivated before fall-through)

      // Null check: replayHandle could have been reset by an earlier path
      // (e.g., FORCE_RECAPTURE composite cleanup, stale-gap invalidation).
      if (!seg.exec.replayHandle) {
        DSP_DIAG(EXECUTE, "MONOLITHIC_CAPTURE_SKIP: seg[%d-%d] replayHandle is null — "
                          "was cleaned up by composite or invalidation path",
                 seg.def.startSlot, seg.def.endSlot);
        popPrimaryCtxIfPushed(didPushCtx, tritonCaptureDevice);
        restoreCublasWorkspaceAfterCapture(stream);
        cleanupCaptureTlsState(true, prevCaptureStream);
        restoreSlotStates(slots_, seg.def.startSlot, seg.def.endSlot, savedSlotStateTriton);
        // Deactivate guard: no capture occurred, executor no longer needed.
        // Fall through — usedTritonGraphCapture stays false → direct exec path.
#if HAVE_TRITON
        if (tritonOrderedRangeGuard.active) {
          TritonGraphBackend::clearOrderedRangeExecutor();
          tritonOrderedRangeGuard.active = false;
        }
#endif
      } else {

      auto* cudaReplay = static_cast<CudaGraphReplayHandle*>(seg.exec.replayHandle.get());
      // Raw pointer — no refcount increment, no risk of touching freed control block.
      auto* handle = cudaReplay->getNativeHandle();
      bool captureOk = handle->beginCapture(ctx.cudaStr, cudaStreamCaptureModeRelaxed);
      if (captureOk) {
        DSP_DIAG_SEG(EXECUTE, seg.def.startSlot, "Triton graph capture started for seg[%d-%d] execCount=%d",
                     seg.def.startSlot, seg.def.endSlot, seg.exec.executionCount);
        tl_graphExecutionActive = true;

        // External inputs are already synced to device (syncToDevice before capture).
        // After syncToSpecial(), readSpecial() makes isSpecialActual()=true via
        // _readSpecial > _writePrimary. The capture-mode guard in DataBuffer::syncToSpecial
        // (if(isSpecialActual()) return;) prevents redundant H2D memcpy nodes.
        //
        // DO NOT call writeSpecial() here. It poisons isPrimaryActual()=false, causing
        // Java getFloat() to copy stale device zeros over valid host data across plans.
        // This was the root cause of the 20% VLM accuracy bug.
        //
        // Internal outputs keep their natural actuality state so nullify() records
        // memset nodes during capture for correct replay zeroing.
        DSP_DIAG(MEMORY, "capture: external inputs already device-actual via syncToDevice+readSpecial "
                         "(NO writeSpecial poisoning). Internal outputs NOT marked — nullify() records memset nodes");

        // Query node count mid-capture to verify operations are being recorded
        size_t midCaptureNodes = handle->getNumNodesDuringCapture(ctx.cudaStr);
        DSP_DIAG(EXECUTE, "Triton capture mid-check: %zu nodes recorded before executeSegment",
                 midCaptureNodes);

        // Snapshot all buffer addresses at capture entry — compare with replay to detect stale pointers
        {
          std::vector<void*> outAddrs, extAddrs;
          extractDeviceAddrs(outputSlots_, totalOutputSlots_, outAddrs);
          extractDeviceAddrs(externalArrays, numExt, extAddrs);
          DspDiagnostics::getInstance().clearAddressSnapshots();
          DSP_DIAG_SNAPSHOT_ADDRS("capture-entry", outAddrs.data(), totalOutputSlots_,
                                  extAddrs.data(), numExt);
        }

        auto captureStatus = ctx.backend->executeSegment(seg, slots_, externalArrays, numExt,
                                                     outputSlots_, totalOutputSlots_, stream);
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
                 tl_captureWorkspaceOffset, seg.exec.replayHandle->getWorkspaceBytes(),
                 seg.exec.replayHandle->getWorkspaceBytes() > 0 ? (100.0 * tl_captureWorkspaceOffset / seg.exec.replayHandle->getWorkspaceBytes()) : 0.0);
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
          auto capErr = cudaStreamGetCaptureInfo_v2(ctx.cudaStr, &capStat, &capId, &capGraph, nullptr, nullptr);
          if (capErr == cudaSuccess && capGraph != nullptr) {
            cudaGraphGetNodes(capGraph, nullptr, &postExecNodes);
          }
        }
        DSP_DIAG(EXECUTE, "Triton capture post-exec: %zu nodes, captureStatus=%d",
                 postExecNodes, static_cast<int>(captureStatus));

        bool endOk = false;
        if (captureStatus == Status::OK) {
          endOk = handle->endCapture(ctx.cudaStr);
        } else {
          DSP_DIAG(EXECUTE, "FATAL: Triton capture execution FAILED status=%d for seg[%d-%d]. "
                            "BLOCKING EXECUTION.",
                   static_cast<int>(captureStatus), seg.def.startSlot, seg.def.endSlot);
          fflush(stdout); fflush(stderr);
          if (handle->isCapturing()) {
            handle->endCapture(ctx.cudaStr);
          }
        }

        if (endOk) {
          size_t numGraphNodes = handle->getNumNodes();
          int segSize = seg.def.endSlot - seg.def.startSlot + 1;
          DSP_DIAG_SEG(COMPILE, seg.def.startSlot,
                       "GRAPH CAPTURE COMPLETE: seg[%d-%d] %zu nodes captured from %d slots (%.1f nodes/slot)",
                       seg.def.startSlot, seg.def.endSlot, numGraphNodes, segSize,
                       segSize > 0 ? (double)numGraphNodes / segSize : 0.0);
          DSP_DIAG(EXECUTE, "Triton capture endOk: graph has %zu nodes", numGraphNodes);

          // Near-empty graphs have almost no GPU work — replay would skip the vast
          // majority of ops, producing wrong results. A graph with < 5% of the segment's
          // ops as nodes means most ops were gap-skipped during capture and aren't in the
          // graph. Mark as non-capturable so future executions fall back to slot-by-slot.
          double nodeRatio = segSize > 0 ? (double)numGraphNodes / segSize : 0.0;
          if (numGraphNodes == 0 || (segSize > 10 && nodeRatio < 0.05)) {
            DSP_DIAG_SEG(COMPILE, seg.def.startSlot,
                         "near-empty Triton graph for seg[%d-%d] (%zu nodes from %d slots, "
                         "ratio=%.2f) — marking as non-capturable",
                         seg.def.startSlot, seg.def.endSlot, numGraphNodes, segSize, nodeRatio);
            SegmentLifecycle::markFailed(seg.exec, "near_empty_graph");
            abortCapture(seg, true, didPushCtx, tritonCaptureDevice,
                        prevCaptureStream, savedSlotStateTriton, stream);
            seg.exec.executionCount++;
#if HAVE_TRITON
            // Deactivate guard: abortCapture cleaned up; destructor must not double-clear.
            tritonOrderedRangeGuard.active = false;
            TritonGraphBackend::clearOrderedRangeExecutor();
#endif
            return Status::OK;
          }

          // Sample final output AFTER endCapture (stream no longer capturing, safe)
          if (seg.def.endSlot < totalOutputSlots_ && outputSlots_[seg.def.endSlot] != nullptr) {
            auto* finalOut = outputSlots_[seg.def.endSlot];
            if (finalOut->dataType() == FLOAT32) {
              DSP_DIAG_DUMP_SLOT("capture-post-endCapture", seg.def.endSlot,
                                 DSP_BUF(finalOut), finalOut->lengthOf());
            }
          }
          // Dump top logit from capture execution via DSP_DIAG
          // Use outputSlotIndices[0] to get the ACTUAL final output slot
          // (matches GRAPH_REPLAY logic for apples-to-apples comparison)
          {
            int captureOutputSlot = -1;
            if (seg.def.endSlot < numSlots_ && slots_[seg.def.endSlot].wiring.numOutputs > 0) {
              captureOutputSlot = slots_[seg.def.endSlot].wiring.outputSlotIndices[0];
            }
            if (captureOutputSlot < 0 || captureOutputSlot >= totalOutputSlots_) {
              captureOutputSlot = seg.def.endSlot;
            }
            if (captureOutputSlot >= 0 && captureOutputSlot < totalOutputSlots_ &&
                outputSlots_[captureOutputSlot] != nullptr) {
              auto* out = outputSlots_[captureOutputSlot];
              if (out->dataType() == FLOAT32 && out->lengthOf() > 0) {
                DSP_DIAG_DUMP_SEG_OUTPUT("CAPTURE_EXEC", captureOutputSlot, DSP_BUF(out),
                                         out->lengthOf(), seg.exec.executionCount, stream);
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

        if (!instantiateOk) {
          int deviceId = 0;
          cudaGetDevice(&deviceId);

          // Check if instantiation failed due to OOM — retry with eviction if possible.
          auto* cudaReplayForOom = seg.exec.replayHandle
                                   ? dynamic_cast<CudaGraphReplayHandle*>(seg.exec.replayHandle.get()) : nullptr;
          bool isOom = cudaReplayForOom && cudaReplayForOom->wasLastInstantiateOom();
          if (isOom && seg.exec.captureOomRetries < GraphSegment::maxOomRetries()) {
            int retryAfter = seg.exec.executionCount + GraphSegment::retryInterval();
            SegmentLifecycle::markOomDeferred(seg.exec, retryAfter);
            DSP_DIAG_SEG(MEMORY, seg.def.startSlot,
                         "INSTANTIATE OOM — retry %d/%d, evicting LRU graphs. retryAfterExec=%d",
                         seg.exec.captureOomRetries, GraphSegment::maxOomRetries(),
                         seg.exec.captureRetryAfterExec);

            // Evict LRU graphs to free memory for the next attempt
            evictLruGraphs(ctx.segIdx, TRITON_CAPTURE_WORKSPACE_SIZE, stream);

            // Cleanup this failed attempt but do NOT set compilationFailed
            abortCapture(seg, true, didPushCtx, tritonCaptureDevice,
                        prevCaptureStream, savedSlotStateTriton, stream);
            cudaGetLastError();  // Clear sticky error
            // OOM during graph instantiation — throw instead of silently falling back
            // to slot-by-slot. The eviction above freed memory; the next execution
            // attempt (deferred by captureRetryAfterExec) will retry capture.
            // Silently producing output via slot-by-slot masks the OOM and the caller
            // never knows the graph wasn't captured.
            {
#if HAVE_TRITON
              // Deactivate guard: abortCapture cleaned up; destructor must not double-clear.
              tritonOrderedRangeGuard.active = false;
              TritonGraphBackend::clearOrderedRangeExecutor();
#endif
              DSP_THROW_SEG(COMPILE, seg.def.startSlot,
                            "NativeDSP: graph instantiation OOM for seg[%d-%d] on device %d "
                            "(retry %d/%d, retryAfterExec=%d). Evicted LRU graphs. "
                            "Fix memory pressure — do NOT fall back to slot-by-slot.",
                            seg.def.startSlot, seg.def.endSlot, deviceId,
                            seg.exec.captureOomRetries, GraphSegment::maxOomRetries(),
                            seg.exec.captureRetryAfterExec);
            }
          }

          // Not OOM or retries exhausted — permanent failure
          abortCapture(seg, true, didPushCtx, tritonCaptureDevice,
                      prevCaptureStream, savedSlotStateTriton, stream);
#if HAVE_TRITON
          // Deactivate guard: abortCapture cleaned up; destructor must not double-clear.
          tritonOrderedRangeGuard.active = false;
          TritonGraphBackend::clearOrderedRangeExecutor();
#endif
          return reportCaptureError(seg, "instantiate", cudaGetLastError(), deviceId);
        }

        // POST-INSTANTIATE MEMORY CHECK removed: the validation launch immediately
        // below will reveal if the graph is usable. No speculative memory gate needed.

        // Graph instantiated — launch to validate the graph is not corrupted.
        // Warmup results are restored from savedWarmupOutputSlots below regardless.
        bool launchOk = false;
        {
          int deviceId = 0;
          cudaGetDevice(&deviceId);
          cudaGetLastError();
          bool replayResult = seg.exec.replayHandle->replay(stream);
          if (!replayResult) {
            abortCapture(seg, true, didPushCtx, tritonCaptureDevice,
                        prevCaptureStream, savedSlotStateTriton, stream);
#if HAVE_TRITON
            // Deactivate guard: abortCapture cleaned up; destructor must not double-clear.
            tritonOrderedRangeGuard.active = false;
            TritonGraphBackend::clearOrderedRangeExecutor();
#endif
            return reportReplayError(seg, "validation_launch", cudaGetLastError(), deviceId);
          }
          cudaError_t syncErr = cudaStreamSynchronize(ctx.cudaStr);
          if (syncErr != cudaSuccess) {
            abortCapture(seg, true, didPushCtx, tritonCaptureDevice,
                        prevCaptureStream, savedSlotStateTriton, stream);
#if HAVE_TRITON
            // Deactivate guard: abortCapture cleaned up; destructor must not double-clear.
            tritonOrderedRangeGuard.active = false;
            TritonGraphBackend::clearOrderedRangeExecutor();
#endif
            return reportReplayError(seg, "validation_sync", syncErr, deviceId);
          }
          DSP_DIAG(EXECUTE, "VALIDATION LAUNCH OK: seg[%d-%d] graph launched and synced successfully",
                   seg.def.startSlot, seg.def.endSlot);
          // LRU tracking: record when this segment was last replayed for eviction ordering
          seg.exec.lastReplayExecCount = executeCount_;
          launchOk = true;
        }

        if (launchOk) {
          if (seg.def.endSlot < totalOutputSlots_ && outputSlots_[seg.def.endSlot] != nullptr) {
            auto* finalOut = outputSlots_[seg.def.endSlot];
            if (finalOut->dataType() == FLOAT32) {
              DSP_DIAG_DUMP_SLOT("capture-post-launch", seg.def.endSlot,
                                 DSP_BUF(finalOut), finalOut->lengthOf());
            }
          }
          // Dump top logit from first replay (graph launch after capture) via DSP_DIAG
          if (seg.def.endSlot < totalOutputSlots_ && outputSlots_[seg.def.endSlot] != nullptr) {
            auto* out = outputSlots_[seg.def.endSlot];
            if (out->dataType() == FLOAT32 && out->lengthOf() > 0) {
              DSP_DIAG_DUMP_SEG_OUTPUT("REPLAY_LAUNCH", seg.def.endSlot, DSP_BUF(out),
                                       out->lengthOf(), seg.exec.executionCount, stream);
            }
          }
          // replayHandle already set (created before capture began)
          seg.exec.cachedShapeKey = ctx.segShapeKey;
          snapshotExternalAddrs(seg, externalArrays, numExt);
          SegmentLifecycle::markCaptured(seg.exec, ctx.segInputAddrKey, ctx.createValueKey,
              computeSlotAddrHash(outputSlots_, seg.def.startSlot, seg.def.endSlot, totalOutputSlots_),
              ctx.backendName);

          // Structured trace: record successful monolithic capture.
          DSP_TRACE_GRAPH_CAPTURED(trace_, static_cast<int8_t>(ctx.segIdx),
                                   static_cast<uint32_t>(executeCount_),
                                   seg.def.startSlot, seg.def.endSlot, 1u);

          // No actuality reset needed — writeSpecial() is no longer called during
          // capture, so external input actuality flags remain in their natural bi-actual
          // state (isPrimaryActual()=true AND isSpecialActual()=true via readSpecial).
          DSP_DIAG(MEMORY, "MONOLITHIC_CAPTURE: no actuality reset needed — writeSpecial not called");

          // Export graph stats and DOT file for diagnostics
          auto stats = handle->getStatistics();
          DSP_DIAG_SEG(EXECUTE, seg.def.startSlot, "Triton graph CAPTURED and launched for seg[%d-%d]: "
                                                   "%d kernels, %d memcpy, %d memset, %d memAlloc, %d memFree "
                                                   "(workspace=%zuMB, offset=%zu)",
                       seg.def.startSlot, seg.def.endSlot,
                       stats.numKernels, stats.numMemcpyH2D + stats.numMemcpyD2H + stats.numMemcpyD2D,
                       stats.numMemsets, stats.numMemAllocs, stats.numMemFrees,
                       seg.exec.replayHandle->getWorkspaceBytes() / (1024*1024), tl_captureWorkspaceOffset);
          // Dump H2D memcpy node details to identify the source of baked-in host addresses
          if (stats.numMemcpyH2D > 0 && DSP_DIAG_ENABLED(EXECUTE)) {
            size_t numGraphNodes = 0;
            cudaGraphGetNodes(handle->getGraph(), nullptr, &numGraphNodes);
            if (numGraphNodes > 0) {
              std::vector<cudaGraphNode_t> graphNodes(numGraphNodes);
              cudaGraphGetNodes(handle->getGraph(), graphNodes.data(), &numGraphNodes);
              for (size_t ni = 0; ni < numGraphNodes; ni++) {
                cudaGraphNodeType nodeType;
                cudaGraphNodeGetType(graphNodes[ni], &nodeType);
                if (nodeType == cudaGraphNodeTypeMemcpy) {
                  cudaMemcpy3DParms mcpyParams;
                  memset(&mcpyParams, 0, sizeof(mcpyParams));
                  if (cudaGraphMemcpyNodeGetParams(graphNodes[ni], &mcpyParams) == cudaSuccess) {
                    size_t bytes = mcpyParams.extent.width *
                                   std::max(mcpyParams.extent.height, (size_t)1) *
                                   std::max(mcpyParams.extent.depth, (size_t)1);
                    const char* kindStr = (mcpyParams.kind == cudaMemcpyHostToDevice) ? "H2D" :
                                          (mcpyParams.kind == cudaMemcpyDeviceToDevice) ? "D2D" :
                                          (mcpyParams.kind == cudaMemcpyDeviceToHost) ? "D2H" : "other";
                    DSP_DIAG(EXECUTE, "GRAPH_NODE[%zu] MEMCPY %s: %zu bytes src=%p dst=%p "
                                      "seg[%d-%d]",
                             ni, kindStr, bytes,
                             mcpyParams.srcPtr.ptr, mcpyParams.dstPtr.ptr,
                             seg.def.startSlot, seg.def.endSlot);
                  }
                }
              }
            }
          }
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
              fprintf(f, "segment=%d-%d\n", seg.def.startSlot, seg.def.endSlot);
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


          // FORCE_RECAPTURE: invalidate graph immediately after capture+launch
          // so the NEXT step also re-captures instead of replaying a stale graph.
          // This ensures every single step is a fresh capture+launch with zero replays.
          if (Environment::getInstance().tritonForceRecapture()) {
            SegmentLifecycle::invalidateForRebuild(this, seg, "force_recapture_post_monolithic_capture");
            batchD2DCount_ = 0;
            DSP_DIAG(EXECUTE, "FORCE_RECAPTURE: invalidated after capture+launch execCount=%d", seg.exec.executionCount);
          }
#if HAVE_TRITON
          // Deactivate guard: monolithic capture done (success path), executor no longer needed.
          // Prevents destructor from interleaving with unrelated code at function exit.
          tritonOrderedRangeGuard.active = false;
          TritonGraphBackend::clearOrderedRangeExecutor();
#endif
        } else {
          int deviceId = 0;
          cudaGetDevice(&deviceId);
          abortCapture(seg, true, didPushCtx, tritonCaptureDevice,
                      prevCaptureStream, savedSlotStateTriton, stream);
#if HAVE_TRITON
          // Deactivate guard: abortCapture cleaned up; destructor must not double-clear.
          tritonOrderedRangeGuard.active = false;
          TritonGraphBackend::clearOrderedRangeExecutor();
#endif
          return reportCaptureError(seg, "execute_during_capture", cudaGetLastError(), deviceId);
        }
      } else {
        int deviceId = 0;
        cudaGetDevice(&deviceId);

        // Check if beginCapture failed due to OOM — retry with eviction if possible.
        cudaError_t beginErr = cudaGetLastError();
        bool isOom = (beginErr == cudaErrorMemoryAllocation);
        if (isOom && seg.exec.captureOomRetries < GraphSegment::maxOomRetries()) {
          int retryAfter = seg.exec.executionCount + GraphSegment::retryInterval();
          SegmentLifecycle::markOomDeferred(seg.exec, retryAfter);
          DSP_DIAG_SEG(MEMORY, seg.def.startSlot,
                       "BEGIN_CAPTURE OOM — retry %d/%d, evicting LRU graphs. retryAfterExec=%d",
                       seg.exec.captureOomRetries, GraphSegment::maxOomRetries(),
                       seg.exec.captureRetryAfterExec);
          evictLruGraphs(ctx.segIdx, TRITON_CAPTURE_WORKSPACE_SIZE, stream);
          abortCapture(seg, true, didPushCtx, tritonCaptureDevice,
                      prevCaptureStream, savedSlotStateTriton, stream);
          // OOM during beginCapture — throw instead of silently falling back to slot-by-slot.
          {
#if HAVE_TRITON
            // Deactivate guard: abortCapture cleaned up; destructor must not double-clear.
            tritonOrderedRangeGuard.active = false;
            TritonGraphBackend::clearOrderedRangeExecutor();
#endif
            DSP_THROW_SEG(COMPILE, seg.def.startSlot,
                          "NativeDSP: beginCapture OOM for seg[%d-%d] on device %d "
                          "(retry %d/%d, retryAfterExec=%d). Evicted LRU graphs. "
                          "Fix memory pressure — do NOT fall back to slot-by-slot.",
                          seg.def.startSlot, seg.def.endSlot, deviceId,
                          seg.exec.captureOomRetries, GraphSegment::maxOomRetries(),
                          seg.exec.captureRetryAfterExec);
          }
        }

        abortCapture(seg, true, didPushCtx, tritonCaptureDevice,
                    prevCaptureStream, savedSlotStateTriton, stream);
#if HAVE_TRITON
        // Deactivate guard: abortCapture cleaned up; destructor must not double-clear.
        tritonOrderedRangeGuard.active = false;
        TritonGraphBackend::clearOrderedRangeExecutor();
#endif
        return reportCaptureError(seg, "beginCapture", beginErr, deviceId);
      }

      // ── Restore warmup output slots for downstream segment visibility ──
      // This MUST happen regardless of capture success/failure. During capture
      // execution, ops allocate outputs from capture workspace, overwriting
      // outputSlots_[] with workspace addresses. If capture fails (endCapture
      // error, instantiate error, or execution error), outputSlots_[] still
      // has the stale workspace addresses. Downstream segments reading from
      // these get garbage data → NaN propagation.
      //
      // No WARMUP_RESTORE needed: gap ops were skipped during capture, so
      // outputSlots_[] was never overwritten. Warmup data is intact.

      DSP_DIAG(EXECUTE, "CAPTURE_COMPLETE: seg[%d-%d] hasReplay=%d compilationFailed=%d "
                        "numCaptureBuffers=%d",
               seg.def.startSlot, seg.def.endSlot,
               seg.exec.replayHandle != nullptr,
               seg.exec.compilationFailed,
               0);

      // No external/cross-slot rewiring is needed now that replay uses the
      // canonical external and output buffers directly. The restore loops remain
      // harmless no-ops because the saved lists stay empty.
      for (auto& [extIdx, origArr] : savedExtForCapture) {
        externalArrays[extIdx] = origArr;
      }

      // Restore cross-segment output slots to warmup pointers.
      // The producing segment's replay writes fresh data to the warmup array's
      // GPU address (baked during capture). The D2D copy before the consuming
      // segment's replay reads from outputSlots_[] (warmup pointer, with fresh
      // GPU data from the producing segment's replay).
      for (auto& [slotIdx, origArr] : savedSlotsForCapture) {
        if (origArr != nullptr) {
          outputSlots_[slotIdx] = origArr;
        }
      }

      popPrimaryCtxIfPushed(didPushCtx, tritonCaptureDevice);
      restoreCublasWorkspaceAfterCapture(stream);

      // Pinned host ptrs: on success move to replay handle, on failure free.
      if (usedTritonGraphCapture && seg.exec.replayHandle) {
        for (auto* ptr : tl_capturedHostPtrs) {
          seg.exec.replayHandle->addCapturedHostPtr(ptr);
        }
        DSP_DIAG(MEMORY, "preserved %zu pinned host ptrs for Triton graph replay",
                 seg.exec.replayHandle->getCapturedHostPtrs().size());
        cleanupCaptureTlsState(false, prevCaptureStream);  // false = ptrs moved
      } else {
        cleanupCaptureTlsState(true, prevCaptureStream);   // true = free ptrs
      }

      restoreSlotStates(slots_, seg.def.startSlot, seg.def.endSlot, savedSlotStateTriton);
    }  // end else (replayHandle != nullptr — workspace allocation succeeded)

    }  // end else (replayHandle null check — monolithic capture body)

  }  // end if (!didCompositeCapture) — monolithic capture only for non-composite segments

  }  // end if (shouldCaptureTritonGraphNow)

  if (!usedTritonGraphCapture) {
    // Cross-stream ordering for direct execution (same issue as capture path).
    // Java-side assign() on the default stream needs to complete before Triton
    // reads device buffers. syncToDevice() below is a no-op when sAct=true.
    {
      auto* execCtxDirect = static_cast<PlanExecutionContext*>(activeExecutionContext());
      syncCrossStream(ctx.cudaStr,
                      execCtxDirect ? execCtxDirect->crossStreamEvent : nullptr,
                      "direct-exec", seg.def.startSlot, seg.def.endSlot,
                      seg.exec.executionCount);
    }

    // NOTE: We intentionally do NOT set cuBLAS workspace for direct/warmup execution.
    // Warmup runs with workspace=0, which causes cuBLAS to select algorithms that
    // don't require workspace. These algorithms are cached in tl_ltAlgoCache.
    // At capture time, setCublasWorkspaceForCapture() provides workspace for
    // capturability, but tryLtMatmul() hits the algo cache and reuses the warmup
    // algorithm — ensuring capture bakes in the SAME algorithm as warmup/live.
    // This makes merged CUDA graph replay numerically identical to live execution.

    // Sync external inputs to device before Triton segment execution.
    // Variable inputs use forceSync=true to bypass stale actuality flags.
    syncExternalInputs(externalArrays, numExt, externalInputIsVariable_,
                       externalInputNames_, shapesFrozen_,
                       Environment::getInstance().tritonVerifyKernels(),
                       "direct", seg.exec.executionCount);

    // NOTE: Do NOT set tl_graphExecutionActive=true here for non-capture Triton execution.
    // That flag suppresses syncToPrimary (D2H transfers), error checking, and
    // PointersManager sync -- behaviors only appropriate during CUDA graph capture.
    // The ordered range executor already handles capture detection independently:
    // it checks cudaStreamIsCapturing() and only sets tl_graphExecutionActive=true
    // when actually capturing. Setting it unconditionally here caused native ordered ops
    // (matmul, gather, etc.) to read stale host data, producing wrong output.

    // Disable frozen fast path for gap ops during Triton segment execution.
    // Same rationale as the capture path (lines 5325-5329): the pre-execution
    // slot restoration at lines 4955-5032 may replace NDArray objects in
    // outputSlots_[], making the frozen context's cached input/output pointers
    // stale. Without clearing frozenContextReady, gap ops write to old arrays
    // while downstream ops read from new arrays, producing wrong output.
    // Save and restore so subsequent executions still benefit from frozen fast path.
    // Demote FROZEN→SHAPE_CACHED so gap ops don't use stale frozen contexts.
    // FROZEN_CONSTANT slots preserved: prezeroSegmentOutputs relies on them.
    std::vector<NativeSlot::SlotState> savedSlotStateNonCapture;
    demoteFrozenSlotStates(slots_, seg.def.startSlot, seg.def.endSlot, savedSlotStateNonCapture);

    // Snapshot addresses for direct execution (baseline for comparison with capture/replay)
    snapshotAddrs(outputSlots_, totalOutputSlots_, externalArrays, numExt, "direct-entry");

    DSP_DIAG_SEG(MEMORY, seg.def.startSlot,
                 "direct-exec invoking prezeroSegmentOutputs seg=[%d-%d] stream=%p execCount=%d",
                 seg.def.startSlot, seg.def.endSlot, (void*)stream, seg.exec.executionCount);
    prezeroSegmentOutputs(seg, stream);

    try {
      status = ctx.backend->executeSegment(seg, slots_, externalArrays, numExt,
                                       outputSlots_, totalOutputSlots_, stream);
    } catch (...) {
      restoreSlotStates(slots_, seg.def.startSlot, seg.def.endSlot, savedSlotStateNonCapture);
      throw;
    }

    restoreSlotStates(slots_, seg.def.startSlot, seg.def.endSlot, savedSlotStateNonCapture);


    // Dump final output for direct Triton path (baseline comparison)
    if (status == Status::OK && seg.def.endSlot < totalOutputSlots_ &&
        outputSlots_[seg.def.endSlot] != nullptr) {
      auto* finalOut = outputSlots_[seg.def.endSlot];
      if (finalOut->dataType() == FLOAT32) {
        DSP_DIAG_DUMP_SLOT("direct", seg.def.endSlot,
                           DSP_BUF(finalOut), finalOut->lengthOf());
      }
    }
    // Always-on diagnostic: dump top logit for non-capture Triton execution
    if (!usedTritonGraphCapture && status == Status::OK &&
        seg.def.endSlot < totalOutputSlots_ && outputSlots_[seg.def.endSlot] != nullptr) {
      auto* out = outputSlots_[seg.def.endSlot];
      if (out->dataType() == FLOAT32 && out->lengthOf() > 0) {
        DSP_DIAG_DUMP_SEG_OUTPUT("DIRECT_TRITON", seg.def.endSlot, DSP_BUF(out),
                                 out->lengthOf(), seg.exec.executionCount, stream);
      }
    }
    // Diagnostic: segment exit argmax
    if (status == Status::OK) {
      dumpSegFinalArgmax(seg, outputSlots_, totalOutputSlots_, numSlots_, slots_,
                         ctx.cudaStr, "SEG_EXIT_ARGMAX", seg.exec.executionCount);
    }

    DSP_DIAG(EXECUTE, "executeSegmentWithGpuGraph: exec%d seg[%d-%d]: backend=%s %s status=%d(%s) "
                      "executionCount=%d compilationFailed=%d usedCapture=%d",
             seg.exec.executionCount, seg.def.startSlot, seg.def.endSlot,
             ctx.backendName, status == Status::OK ? "OK" : "FAILED",
             static_cast<int>(status), statusName_gpu(status),
             seg.exec.executionCount,
             seg.exec.compilationFailed ? 1 : 0, usedTritonGraphCapture ? 1 : 0);

    if (status == Status::OK) {
      seg.exec.executionCount++;
      totalGraphReplays_++;
      if (seg.exec.compiledByBackend.empty()) {
        seg.exec.compiledByBackend = ctx.backendName;
      }
    }

    if (Environment::getInstance().tritonVerifyKernels()) {
      DSP_DIAG(VERIFY, "SEG_EXIT seg[%d-%d] status=%s execCount=%d",
               seg.def.startSlot, seg.def.endSlot, statusName_gpu(status), seg.exec.executionCount);
    }

#if HAVE_TRITON
    // Deactivate guard before return on direct execution path.
    if (tritonOrderedRangeGuard.active) {
      TritonGraphBackend::clearOrderedRangeExecutor();
      tritonOrderedRangeGuard.active = false;
    }
#endif

    return status;

  }  // end if (!usedTritonGraphCapture)

  // Reached when usedTritonGraphCapture == true (capture succeeded and status was set).
  // Explicitly deactivate guard before return — prevents the destructor from interleaving
  // shared_ptr/register operations with clearOrderedRangeExecutor() at function exit.
  // All success paths above SHOULD have deactivated already, but this is a safety net.
#if HAVE_TRITON
  if (tritonOrderedRangeGuard.active) {
    TritonGraphBackend::clearOrderedRangeExecutor();
    tritonOrderedRangeGuard.active = false;
  }
#endif
  return status;

}  // segDispatchCaptureOrDirect

Status NativeDynamicShapePlan::executeSegmentWithGpuGraph(
    GraphSegment& seg, NDArray** externalArrays, int numExt, void* stream) {

  // Derive segIdx for proactive eviction and OOM retry.
  int segIdx = -1;
  for (int si = 0; si < static_cast<int>(segments_.size()); si++) {
    if (&segments_[si] == &seg) { segIdx = si; break; }
  }

  {
    const char* mode = SegmentLifecycle::stateName(seg.exec.lifecycleState);
    DSP_DIAG_SEG(SHAPE, seg.def.startSlot,
                 "executeSegmentWithGpuGraph: ENTER seg[%d-%d] lifecycle=%s execCount=%d capturable=%d",
                 seg.def.startSlot, seg.def.endSlot, mode, seg.exec.executionCount, seg.def.isCapturable ? 1 : 0);
  }

  // ── Segment lifecycle: SEG_ENTER ──────────────────────────────────────
  if (Environment::getInstance().tritonVerifyKernels()) {
    // Ensure VERIFY diagnostic category is enabled and output level is FULL
    // when tritonVerifyKernels is on (may be set at runtime via Java, after
    // DspDiagnostics constructor)
    if (!DSP_DIAG_ENABLED(VERIFY)) {
      sd::graph::DspDiagnostics::getInstance().enableCategories(sd::graph::DSP_DIAG_VERIFY);
      sd::graph::DspDiagnostics::getInstance().setLevel(sd::graph::DSP_LEVEL_FULL);
    }
    const char* lifecycle = SegmentLifecycle::stateName(seg.exec.lifecycleState);
    DSP_DIAG(VERIFY, "SEG_ENTER seg[%d-%d] execCount=%d lifecycle=%s",
             seg.def.startSlot, seg.def.endSlot, seg.exec.executionCount, lifecycle);
    // Dump external input actuality flags for first N inputs
    int detailLimit = sd::graph::DspDiagnostics::getInstance().diagDetailLimit();
    int dumpCount = std::min(numExt, detailLimit);
    for (int i = 0; i < dumpCount; i++) {
      if (externalArrays[i] != nullptr && externalArrays[i]->dataBuffer() != nullptr) {
        auto* db = externalArrays[i]->dataBuffer();
        DSP_DIAG(VERIFY, "  EXT_INPUT[%d] dtype=%s len=%lld pAct=%d sAct=%d addr=%p",
                 i, DataTypeUtils::asString(externalArrays[i]->dataType()).c_str(),
                 (long long)externalArrays[i]->lengthOf(),
                 db->isPrimaryActual() ? 1 : 0, db->isSpecialActual() ? 1 : 0,
                 DSP_BUF(externalArrays[i]));
      }
    }
    if (numExt > detailLimit) {
      DSP_DIAG(VERIFY, "  ... and %d more external inputs", numExt - detailLimit);
    }
  }

  // NOTE: Do NOT substitute effectiveExternals_ here. This function is called
  // from phaseReplay with original externalInputs. When compositeReplay is the
  // actual replay path (line ~1911), it receives externalArrays from here and
  // handles staging internally: syncExternalInputs on originals first, then
  // D2D copy to staging buffers. Substituting here would pass staging pointers
  // to compositeReplay, which would then skip H2D sync on the originals
  // (staging buffers are device-authoritative), leaving Java-side input data
  // stranded on the host.

  auto* backend = getGpuGraphBackend();
  if (backend == nullptr) {
    DSP_DIAG(BACKEND, "executeSegmentWithGpuGraph: no GPU backend selected for seg[%d-%d]",
             seg.def.startSlot, seg.def.endSlot);
    return Status::KERNEL_FAILURE;
  }
  const char* backendName = backend->name();
#if HAVE_TRITON
  auto* tritonBackend = dynamic_cast<TritonGraphBackend*>(backend);
#else
  void* tritonBackend = nullptr;
#endif

  // If compilation previously failed validation, never try again
  if (seg.exec.lifecycleState == SegmentLifecycleState::FAILED) {
    return Status::KERNEL_FAILURE;
  }

  // Check if this segment can be compiled by the selected GPU backend
  if (!backend->canFuseSegment(slots_, seg.def.startSlot, seg.def.endSlot)) {
    DSP_DIAG(BACKEND, "executeSegmentWithGpuGraph: backend=%s cannot fuse seg[%d-%d]",
             backendName, seg.def.startSlot, seg.def.endSlot);
    return Status::KERNEL_FAILURE;  // Caller will fall back to CUDA Graphs
  }

  // First execution: run slot-by-slot warmup BEFORE compilation.
  if (seg.exec.lifecycleState == SegmentLifecycleState::NEEDS_WARMUP) {
    // Shape key not yet computed at this point — record 0 (warmup has no cached key).
    DSP_TRACE_SEG_DISPATCH(trace_, static_cast<int8_t>(segIdx),
                           static_cast<uint8_t>(seg.def.selectedBackend),
                           seg.def.startSlot, seg.def.endSlot,
                           static_cast<uint32_t>(executeCount_), 0u);
    return segDispatchWarmup(seg, externalArrays, numExt, stream);
  }

  // Compute shape key for cache lookup.
  // When shapes are frozen and the key was already computed, reuse it — the shapes
  // cannot change so the hash is stable. Saves iterating all cross-segment inputs.
  // EXCEPTION: segments with value-dependent ops must ALWAYS recompute the shape key
  // because input VALUES (hashed by computeSegmentShapeKey for small inputs ≤32 elements)
  // can change even when shapes are frozen. Without this guard, the cached key would
  // miss value changes in reshape targets, broadcast dims, etc., causing CUDA graph
  // replay with stale output shapes.
  //
  // When shapes are frozen and the key is cached, shape key computation is skipped —
  // createValueKey/address stability checks handle value-dependent inputs separately.
  // ── Shape key: detect if segment needs recompilation ──
  // Frozen + cached key: reuse. Otherwise: compute once and cache.
  const bool hasInternalValueShapeInputs = segmentHasInternalValueShapeInputs(seg, slots_);
  LongType segShapeKey;
  if (shapesFrozen_ && seg.exec.cachedShapeKey != 0) {
    segShapeKey = seg.exec.cachedShapeKey;
  } else {
    segShapeKey = computeSegmentShapeKey(seg, externalArrays, numExt);
    if (shapesFrozen_) {
      seg.exec.cachedShapeKey = segShapeKey;
    }
  }

  // Diagnostic: scan all outputSlots_ entries for freed DataBuffers.
  // During warmup, this runs unconditionally (handles invalidation + rebuild).
  // During frozen replay, this only runs when DSP_DIAG VERIFY is enabled —
  // stale buffers in frozen replay indicate a bug, not a recoverable state.
  bool runStaleBufferScan = !shapesFrozen_ ||
                            DspDiagnostics::getInstance().isEnabled(DSP_DIAG_VERIFY);

  // View-producer slots that wrap a placeholder DataBuffer are legitimately
  // stale whenever SameDiff replaces the placeholder between calls (e.g.
  // EMULATED_REPLAY supplies a fresh external input every step). Refresh
  // those wrappers in place on EVERY frozen replay — the gate below only
  // controls the expensive stale-buffer scan, but view-wrapper refresh must
  // always run or the slot's DataBuffer will dangle into slot-by-slot exec,
  // where writeOutputSlot's frozen-phase guard rejects the replacement as a
  // lifecycle violation.
  if (shapesFrozen_ && slotIsViewProducer_ != nullptr) {
    int viewRefreshResult =
        refreshStaleViewWrappersInSegment(seg, externalArrays, numExt);
    if (viewRefreshResult > 0) {
      // Fresh wrappers expose new device addresses — force argTable refresh on
      // the next replay. Graph remains valid; no recapture needed.
      seg.exec.argTableStable = false;
      seg.exec.addrKeyStableCount = 0;
      seg.exec.slotAddrStableCount = 0;
      DSP_DIAG(MEMORY,
               "executeSegmentWithGpuGraph: refreshed %d stale view wrappers in seg[%d-%d]",
               viewRefreshResult, seg.def.startSlot, seg.def.endSlot);
    }
  }

  if (runStaleBufferScan) {
    int invalidCount = 0;
    for (int si = seg.def.startSlot; si <= seg.def.endSlot && si < totalOutputSlots_; si++) {
      NDArray* cached = outputSlots_[si];
      if (cached != nullptr && cached->hasValidShapeInfo() && !cached->isEmpty()) {
        auto* db = cached->dataBuffer();
        if (db == nullptr || !db->isValid()) {
          DSP_DIAG_SLOT(MEMORY, si, "STALE outputSlots_[%d] detected "
                                    "(arr=%p, db=%p, dbValid=%d, frozenConst=%d). Invalidating.",
                        si, (void*)cached, (void*)db, db ? (db->isValid() ? 1 : 0) : -1,
                        slots_[si].frozenConstantSlot() ? 1 : 0);
          outputSlots_[si] = nullptr;
          if (si < numSlots_ && slots_[si].state_ == NativeSlot::SlotState::FROZEN_CONSTANT) {
            slots_[si].state_ = NativeSlot::SlotState::FROZEN;
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
      if (shapesFrozen_ && seg.exec.executionCount > 1) {
        // After warmup with frozen shapes, stale buffers mean a bug in array lifecycle management
        REQUIRE_TRUE(false, 0, "Stale buffer detected after warmup (executionCount=%d, frozen=%d, "
                               "invalidCount=%d) in seg[%d-%d]. This indicates a bug in DSP array persistence.",
                     seg.exec.executionCount, (int)shapesFrozen_, invalidCount,
                     seg.def.startSlot, seg.def.endSlot);
      }
      // During warmup/transitions, invalidate and re-execute
      SegmentLifecycle::invalidateForRebuild(this, seg, "stale_view_wrappers");
      batchD2DCount_ = 0;
      DSP_THROW_SEG(COMPILE, seg.def.startSlot,
                    "graph invalidated for seg[%d-%d] due to %d stale view wrapper entries",
                    seg.def.startSlot, seg.def.endSlot, invalidCount);
    }
  }

  // Pre-execution: ensure all output slots in the segment have live arrays.
  // The Triton kernel's arg mapping references outputSlots_ for both inputs
  // (from prior ops) and outputs (to write results). Slot-by-slot warmup may
  // have released intermediate arrays via releaseAtStep_, leaving entries null.
  // First restore from outputSlots_, then allocate any remaining nulls
  // using cached shape info from warmup.
  //
  //  This MUST happen BEFORE compilation. The compiler resolves
  // arg mappings from outputSlots_ — if intermediate slots are null (released
  // after warmup), the compiler omits them from the arg table, producing
  // sub-kernels with missing inputs that read stale/garbage data on first
  // execution. By populating all slots before compilation, the compiler sees
  // all arrays and builds correct arg mappings.
  //
  // IMPORTANT: Java may close() output arrays between execution steps (e.g.,
  // prefill KV outputs via setCloseable(true)+close()). This frees the underlying
  // DataBuffer while outputSlots_ still holds the NDArray*. Validate the
  // DataBuffer before reusing — invalidate entries pointing to freed buffers.
  //
  //  If any output slot within the segment is allocated at a NEW address
  // (different from capture time), the cached CUDA graph becomes invalid. Triton
  // arg tables are refreshed with new addresses, but native ops (cuBLAS matmul)
  // have addresses baked into the graph. This address inconsistency causes the
  // graph to read stale data from old addresses while Triton writes to new ones.
  // Track any new allocations and invalidate the graph if needed.
  //
  // OPTIMIZATION: Skip when shapes are frozen and we have a valid replay handle.
  // In frozen replay, outputSlots_ are stable — no arrays are released or freed
  // between steps. Segments without a replay handle (pre-capture) MUST always
  // get pre-exec restoration — cleanup may have nulled cross-segment input slots.
  int preExecAllocCount = 0;
  if (!(shapesFrozen_ && seg.exec.replayHandle != nullptr)) {
    for (int stepIdx = seg.def.startSlot; stepIdx <= seg.def.endSlot; stepIdx++) {
      NativeSlot& slot = slots_[stepIdx];
      // Validate input DataBuffers — Java close() may have freed them.
      for (int i = 0; i < slot.wiring.numInputs; i++) {
        int srcIdx = slot.wiring.inputSourceIndices[i];
        if (srcIdx >= 0 && srcIdx < totalOutputSlots_ && outputSlots_[srcIdx] != nullptr) {
          auto* db = outputSlots_[srcIdx]->dataBuffer();
          if (db == nullptr || !db->isValid()) {
            outputSlots_[srcIdx] = nullptr;
            if (srcIdx < numSlots_ && slots_[srcIdx].state_ == NativeSlot::SlotState::FROZEN_CONSTANT) {
              slots_[srcIdx].state_ = NativeSlot::SlotState::FROZEN;
            }
          }
        }
      }
      // Validate or allocate output slot entries
      for (int i = 0; i < slot.wiring.numOutputs; i++) {
        int slotIdx = slot.wiring.outputSlotIndices[i];
        if (slotIdx < 0 || slotIdx >= totalOutputSlots_) continue;
        // DIAGNOSTIC: trace configured slot pre-exec validation (ND4J_DSP_TRACE_SLOT)
        if (DSP_DIAG_ENABLED(MEMORY)) {
          int ts = sd::graph::DspDiagnostics::getInstance().traceSlot();
          if (ts >= 0 && slotIdx == ts && shapesFrozen_) {
            auto* arr = outputSlots_[slotIdx];
            auto* db = arr != nullptr ? arr->dataBuffer() : nullptr;
            DSP_DIAG_SLOT(MEMORY, stepIdx,
                          "PRE_EXEC_VALIDATE: slot=%d arr=%p db=%p valid=%d exec=%d",
                          slotIdx, (void*)arr, (void*)db,
                          db != nullptr && db->isValid() ? 1 : 0,
                          seg.exec.executionCount);
          }
        }
        // Validate existing entry
        if (outputSlots_[slotIdx] != nullptr) {
          auto* db = outputSlots_[slotIdx]->dataBuffer();
          if (db == nullptr || !db->isValid()) {
            if (DSP_DIAG_ENABLED(MEMORY)) {
              int ts = sd::graph::DspDiagnostics::getInstance().traceSlot();
              if (ts >= 0 && slotIdx == ts) {
                DSP_DIAG_SLOT(MEMORY, stepIdx,
                              "PRE_EXEC_NULL: slot=%d db=%p was nullOrInvalid exec=%d",
                              slotIdx, (void*)db, seg.exec.executionCount);
              }
            }
            outputSlots_[slotIdx] = nullptr;
            if (stepIdx < numSlots_ && slots_[stepIdx].state_ == NativeSlot::SlotState::FROZEN_CONSTANT) {
              slots_[stepIdx].state_ = NativeSlot::SlotState::FROZEN;
            }
          }
        }
        if (outputSlots_[slotIdx] == nullptr) {
          // After warmup with frozen shapes, null output slots indicate a persistence bug.
          // Frozen constant slots are exempt (they never allocate output arrays).
          // Warn but continue — the allocation path below will recover.
          if (shapesFrozen_ && seg.exec.executionCount > 1 && !slot.frozenConstantSlot()) {
            DSP_DIAG_SLOT(VERIFY, slotIdx,
                          "BUG: Null output slot %d (%s) after warmup with frozen shapes — persistence bug. execCount=%d",
                          slotIdx, slot.ident.opName.c_str(), seg.exec.executionCount);
          }
          // Phase assertion: allocating a new NDArray during REPLAYING phase is a bug.
          // Output slots should already be populated from warmup/capture. New allocations
          // during replay mean the slot was freed or not persisted correctly.
          if (seg.exec.lifecycleState == SegmentLifecycleState::REPLAYING && !slot.frozenConstantSlot()) {
            DSP_DIAG(EXECUTE, "PHASE_VIOLATION: new NDArray allocation for slot %d (%s) during "
                              "REPLAYING phase — output should already exist from warmup. "
                              "seg[%d-%d] execCount=%d planPhase=%s",
                     slotIdx, slot.ident.opName.c_str(),
                     seg.def.startSlot, seg.def.endSlot, seg.exec.executionCount,
                     dsp::planPhaseName(planPhase_));
            REQUIRE_TRUE(false, 0,
                         "DSP phase contract violation: new NDArray allocation for slot %d (%s) "
                         "during REPLAYING phase for seg[%d-%d].",
                         slotIdx, slot.ident.opName.c_str(), seg.def.startSlot, seg.def.endSlot);
          }
          // Allocate from cached shape info (populated during warmup)
          const LongType* shapeInfo = nullptr;
          if (i < static_cast<int>(slot.shapeCache.cachedOutputShapes.size()) && slot.shapeCache.cachedOutputShapes[i]) {
            shapeInfo = slot.shapeCache.cachedOutputShapes[i];
          }
          // For identity/view-like ops that don't cache output shapes,
          // derive the shape from the first input source's existing array
          if (!shapeInfo && slot.wiring.numInputs > 0) {
            int srcIdx = slot.wiring.inputSourceIndices[0];
            NDArray* srcArr = nullptr;
            if (srcIdx < 0) {
              int extIdx = -(srcIdx + 1);
              if (extIdx < numExt) srcArr = externalArrays[extIdx];
            } else if (srcIdx < totalOutputSlots_) {
              srcArr = outputSlots_[srcIdx];
            }
            if (srcArr) shapeInfo = srcArr->shapeInfo();
          }
          if (shapeInfo) {
            auto dt = ArrayOptions::dataType(shapeInfo);
            // For cast ops, the output type must match the declared target type,
            // not the input type. When cachedOutputShapes is empty and the
            // using the input source's shape, the dtype would be wrong
            // (e.g., INT64 input for a cast-to-FLOAT op).
            if (slot.ident.op && slot.ident.op->getOpDescriptor() &&
                slot.ident.op->getOpDescriptor()->hasAnyTrait(sd::ops::OP_TRAIT_CAST) &&
                slot.args.numIArgs > 0 && slot.args.iArgs) {
              auto castDt = static_cast<DataType>(slot.args.iArgs[0]);
              if (castDt != dt) {
                DSP_DIAG(EXECUTE, "PRE_EXEC_ALLOC: cast dtype override slot=%d from %s to %s",
                         slotIdx, DataTypeUtils::asString(dt).c_str(),
                         DataTypeUtils::asString(castDt).c_str());
                dt = castDt;
              }
            }
            auto order = shape::order(shapeInfo);
            LongType rank = shape::rank(shapeInfo);
            std::vector<LongType> shapeVec(rank);
            for (int d = 0; d < rank; d++) shapeVec[d] = shapeInfo[d + 1];
            auto* arr = new NDArray(order, shapeVec, dt);
            outputSlots_[slotIdx] = arr;
            preExecAllocCount++;
            {
              int ts = sd::graph::DspDiagnostics::getInstance().traceSlot();
              if (ts >= 0 && slotIdx == ts) {
                auto* newDb = arr != nullptr ? arr->dataBuffer() : nullptr;
                DSP_DIAG_SLOT(MEMORY, stepIdx,
                              "PRE_EXEC_ALLOC: slot=%d arr=%p db=%p exec=%d",
                              slotIdx, (void*)arr, (void*)newDb, seg.exec.executionCount);
              }
            }
            if (Environment::getInstance().tritonVerifyKernels()) {
              DSP_DIAG(VERIFY, "SLOT_WRITE slot=%d tag=ALLOC dtype=%s len=%lld addr=%p",
                       slotIdx, DataTypeUtils::asString(dt).c_str(),
                       (long long)arr->lengthOf(), DSP_BUF(arr));
            }
          }
        }
      }
    }
  } // end if (!(shapesFrozen_ && replayHandle))

  // ── COMPILE DISPATCH ──────────────────────────────────────────────────────
  // segDispatchCompile handles: phase guard, shape-change mini-warmup,
  // backend->compileSegment(), markCompiled(), and first-compilation audit.
  // segShapeKey is passed by reference — shape-change recompile updates it.
  // Pre-exec output slot allocation above ensures all slots are populated
  // before the compiler resolves arg mappings.
  {
    if (seg.exec.lifecycleState == SegmentLifecycleState::NEEDS_COMPILE) {
      DSP_TRACE_SEG_DISPATCH(trace_, static_cast<int8_t>(segIdx),
                             static_cast<uint8_t>(seg.def.selectedBackend),
                             seg.def.startSlot, seg.def.endSlot,
                             static_cast<uint32_t>(executeCount_),
                             static_cast<uint64_t>(segShapeKey));
    }
    auto compileStatus = segDispatchCompile(seg, externalArrays, numExt, stream, segShapeKey);
    if (compileStatus != Status::OK) {
      DSP_TRACE_ERROR(trace_, static_cast<int8_t>(segIdx), seg.def.startSlot,
                      static_cast<uint32_t>(executeCount_),
                      static_cast<uint64_t>(compileStatus));
      return compileStatus;
    }
  }

  // Record shape key as compiled — future executions with same key skip recompile.
  seg.def.shapeKeyState.markCompiled(segShapeKey);
  DSP_SEG_EVENT(seg, SHAPE_KEY_STORED, "compilation/execution complete");

  cudaStream_t cudaStr = (stream != nullptr)
                         ? *static_cast<cudaStream_t*>(stream) : nullptr;

  // If any output slots were re-allocated at new addresses, the cached CUDA graph
  // is invalid — native ops (cuBLAS) have the old addresses baked in while Triton
  // arg tables were refreshed with new addresses. Invalidate and re-capture.
  // Check both monolithic (replayHandle non-null) and composite
  // (hasCompositeHandles) paths — composite sentinels have state=EMPTY.
  bool hasAnyCapturedGraph = (seg.exec.replayHandle != nullptr) || hasCompositeHandles(seg);
  if (preExecAllocCount > 0 && hasAnyCapturedGraph) {
    DSP_DIAG(EXECUTE, "GRAPH INVALIDATED: %d output slots re-allocated at new addresses "
                      "(cache entries freed by Java). seg[%d-%d] will re-capture.",
             preExecAllocCount, seg.def.startSlot, seg.def.endSlot);
    SegmentLifecycle::invalidateForRebuild(this, seg, "output_addr_change");
    batchD2DCount_ = 0;
  }

  // CUDA graph capture is BLOCKED when tritonSkipKernels=true. Without Triton
  // kernels, capture bakes syncToSpecial() H2D memcpy nodes into the graph.
  // On replay these overwrite freshly-synced device data with stale values.
  bool allowTritonCudaGraphReplay = Environment::getInstance().tritonGraphCapture() &&
                                    shapesFrozen_ &&
                                    !Environment::getInstance().tritonSkipKernels();

  int captureMinExec = Environment::getInstance().tritonCaptureMinExec();
  bool forceRecaptureEnabled = Environment::getInstance().tritonForceRecapture();
  // hasReplayHandle is true for both monolithic (READY handle) and composite
  // (EMPTY sentinel). In both cases it blocks new monolithic capture.
  bool hasReplayHandle = (seg.exec.replayHandle != nullptr);
  bool replayHandleNull = (seg.exec.replayHandle == nullptr);
  bool hasComposite = hasCompositeHandles(seg);  // true only for composite-captured segments
  bool notCaptureFailed = !seg.exec.compilationFailed;
  bool execCountInWindow = (seg.exec.executionCount >= captureMinExec) &&
                           (forceRecaptureEnabled || seg.exec.executionCount <= (captureMinExec + 2));
  bool hasCudaStream = (cudaStr != nullptr);
  bool requiresOrderedGapCapture = false;

  DSP_DIAG(EXECUTE, "=== CAPTURE DECISION CHECK seg[%d-%d] ===", seg.def.startSlot, seg.def.endSlot);
  DSP_DIAG(EXECUTE, "  tritonGraphCapture()=%d, shapesFrozen_=%d, tritonSkipKernels=%d => allowTritonCudaGraphReplay=%d",
           Environment::getInstance().tritonGraphCapture() ? 1 : 0,
           shapesFrozen_ ? 1 : 0,
           Environment::getInstance().tritonSkipKernels() ? 1 : 0,
           allowTritonCudaGraphReplay ? 1 : 0);
  DSP_DIAG(EXECUTE, "  seg.exec.executionCount=%d, captureMinExec=%d, window=[%d,%d], inWindow=%d",
           seg.exec.executionCount, captureMinExec, captureMinExec, captureMinExec + 2,
           execCountInWindow ? 1 : 0);
  // hasReplayHandle=1 + hasComposite=1 means composite-captured (sentinel + island handles).
  // hasReplayHandle=1 + hasComposite=0 means monolithic-captured (full graph in replayHandle).
  // hasReplayHandle=0 means no capture yet.
  DSP_DIAG(EXECUTE, "  hasReplayHandle=%d, replayHandleNull=%d, hasCompositeHandles=%d",
           hasReplayHandle ? 1 : 0, replayHandleNull ? 1 : 0, hasComposite ? 1 : 0);
  DSP_DIAG(EXECUTE, "  compilationFailed=%d, cudaStr!=nullptr=%d",
           seg.exec.compilationFailed ? 1 : 0, hasCudaStream ? 1 : 0);

  bool shouldCaptureTritonGraph = false;

  int tritonGapSlotCount = 0;
#if HAVE_TRITON
  // ── Unified composite replay: ALWAYS build a composite schedule ──
 // Every Triton-compiled segment uses composite replay. Segments with no gaps
 // get a single TRITON_ISLAND unit (functionally identical to monolithic replay).
 // Segments with gaps get interleaved TRITON_ISLAND + GAP units. This eliminates
 // the monolithic replay path and the broken view recipe system entirely.
 if (tritonBackend != nullptr) {
   if (seg.exec.compositeReplaySchedule.units.empty()) {
     seg.exec.compositeReplaySchedule = buildCompositeReplaySchedule(seg, slots_, tritonBackend);
     DSP_DIAG(SHAPE, "COMPOSITE_SCHEDULE_BUILT: seg[%d-%d] units=%d",
              seg.def.startSlot, seg.def.endSlot,
              static_cast<int>(seg.exec.compositeReplaySchedule.units.size()));
   }
   auto gapSlots = tritonBackend->getGapSlots(seg, slots_);
   tritonGapSlotCount = static_cast<int>(gapSlots.size());
   if (!gapSlots.empty()) {
     requiresOrderedGapCapture = true;
   }
 }
#else
  seg.exec.compositeReplaySchedule = ReplaySchedule();
#endif

  bool captureWindowSatisfied = execCountInWindow || requiresOrderedGapCapture;
  shouldCaptureTritonGraph = allowTritonCudaGraphReplay &&
                             !hasReplayHandle &&
                             replayHandleNull &&
                             notCaptureFailed &&
                             captureWindowSatisfied &&
                             hasCudaStream;

  if (requiresOrderedGapCapture) {
    DSP_DIAG(EXECUTE,
             "COMPOSITE_GAP_CAPTURE: seg[%d-%d] has %d gap slots. "
             "Gap ops will be EXCLUDED from CUDA graph; "
             "composite replay will execute gaps fresh before Triton-only graph replay.",
             seg.def.startSlot, seg.def.endSlot, tritonGapSlotCount);
  }

  DSP_DIAG(EXECUTE, "  => shouldCaptureTritonGraph=%d", shouldCaptureTritonGraph ? 1 : 0);
  if (!shouldCaptureTritonGraph) {
    if (!allowTritonCudaGraphReplay)
      DSP_DIAG(EXECUTE, "  BLOCKED: allowTritonCudaGraphReplay=false (tritonGraphCapture=%d OR shapesFrozen_=%d OR tritonSkipKernels=%d)",
               Environment::getInstance().tritonGraphCapture() ? 1 : 0, shapesFrozen_ ? 1 : 0,
               Environment::getInstance().tritonSkipKernels() ? 1 : 0);
    if (!replayHandleNull)
      DSP_DIAG(EXECUTE, "  BLOCKED: replayHandle already exists (%s capture already done or in progress)",
               hasComposite ? "composite" : "monolithic");
    if (seg.exec.compilationFailed)
      DSP_DIAG(EXECUTE, "  BLOCKED: compilationFailed=true (previous capture failed, warmup path only)");
    if (!captureWindowSatisfied)
      DSP_DIAG(EXECUTE, "  BLOCKED: executionCount=%d outside capture window [%d,%d]",
               seg.exec.executionCount, captureMinExec, captureMinExec + 2);
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
  bool canSkipReplayInvariantRecompute =
      seg.exec.argTableStable && allowTritonCudaGraphReplay &&
      !hasInternalValueShapeInputs;
  if (canSkipReplayInvariantRecompute) {
    // Fast path: arg table is stable, all addresses are known-good
    DSP_DIAG_SEG(EXECUTE, seg.def.startSlot,
                 "seg[%d-%d] argTableStable=true → FAST PATH (skip addr/createValue recompute)",
                 seg.def.startSlot, seg.def.endSlot);
    segInputAddrKey = seg.exec.capturedInputAddrKey;
    extAddrsStable = true;
    createValueKey = seg.exec.capturedCreateValueKey;
  } else {
    segInputAddrKey = computeSegmentInputAddrKey(seg, externalArrays, numExt);
    extAddrsStable = (seg.exec.replayHandle && !seg.exec.replayHandle->getCapturedExternalAddresses().empty())
                     ? externalAddrsMatch(seg, externalArrays, numExt)
                     : (seg.exec.capturedInputAddrKey != 0 && seg.exec.capturedInputAddrKey == segInputAddrKey);
    createValueKey = computeCreateOpValueKey(seg, externalArrays, numExt);
    DSP_DIAG(EXECUTE, "ADDR_CHECK_SLOW: seg[%d-%d] extAddrsStable=%d addrKey=%lld (cached=%lld)",
             seg.def.startSlot, seg.def.endSlot, extAddrsStable ? 1 : 0,
             (long long)segInputAddrKey, (long long)seg.exec.capturedInputAddrKey);
  }
  bool createValuesStable = (createValueKey == 0) ||  // no create ops
                            (seg.exec.capturedCreateValueKey == createValueKey);
  if (hasInternalValueShapeInputs) {
    const bool shapeKeyStable = (seg.exec.cachedShapeKey == 0) ||
                                (seg.exec.cachedShapeKey == segShapeKey);
    seg.exec.argTableStable = seg.exec.argTableStable &&
                              extAddrsStable &&
                              createValuesStable &&
                              shapeKeyStable;
    DSP_DIAG_SEG(EXECUTE, seg.def.startSlot,
                 "INTERNAL_VALUE_SHAPE_TRACKING: seg[%d-%d] argStable=%d shapeStable=%d "
                 "createStable=%d extAddrsStable=%d",
                 seg.def.startSlot, seg.def.endSlot,
                 seg.exec.argTableStable ? 1 : 0,
                 shapeKeyStable ? 1 : 0,
                 createValuesStable ? 1 : 0,
                 extAddrsStable ? 1 : 0);
  }
  if (!createValuesStable && seg.exec.replayHandle) {
    DSP_DIAG(EXECUTE, "CREATE_VALUE_KEY mismatch: captured=%lld current=%lld → invalidating graph seg[%d-%d]",
             (long long)seg.exec.capturedCreateValueKey, (long long)createValueKey, seg.def.startSlot, seg.def.endSlot);
    SegmentLifecycle::invalidateForRebuild(this, seg, "create_value_key_mismatch");
    batchD2DCount_ = 0;
    extAddrsStable = false;  // Force re-capture path
  }

  // Triton graph replay conditions:
  // 1. Shape key matches (frozen shapes)
  // 2. Create op input values stable (ConstantOfShape shapes unchanged)
  // 3. Input addresses are unchanged since capture
  //  Only enter the Triton replay path for segments actually compiled by Triton.
  // Segments captured by the raw CUDA graph path (NativeDynamicShapePlan_cudagraph.cu)
  // have replayHandles but NO Triton arg tables. The Triton replay path's D2D copy +
  // arg table refresh is incompatible with raw CUDA graphs — it can corrupt cross-segment
  // data, causing downstream segments to read zeros instead of valid output → NaN.
  // compiledByBackend is set to backendName ONLY after a successful Triton execution.
  // Raw CUDA captures leave it empty → excluded from this path → fall through to
  // executeSegmentWithGraph() in cudagraph.cu which handles replay correctly.
  bool isTritonCompiled = (!seg.exec.compiledByBackend.empty() && seg.exec.compiledByBackend == backendName);

  // Invalidate stale graphs that have gap ops baked in. Gap ops must NOT be
  // captured into CUDA graphs — their baked addresses go stale on replay.
  // New captures exclude gap ops; this catches legacy pre-fix graphs.
  if (allowTritonCudaGraphReplay &&
      seg.exec.replayHandle != nullptr &&
      seg.exec.replayHandle->isReady() &&
      isTritonCompiled &&
      seg.exec.gapOpsCapturedInGraph) {
    DSP_DIAG(EXECUTE,
             "STALE_GAP_GRAPH_INVALIDATE: invalidating seg[%d-%d] replay handle "
             "because gap ops were baked into the graph (stale addresses on replay).",
             seg.def.startSlot, seg.def.endSlot);
    SegmentLifecycle::invalidateForRebuild(this, seg, "stale_gap_graph");
    batchD2DCount_ = 0;
    seg.exec.gapOpsCapturedInGraph = false;
    seg.exec.executionCount = captureMinExec;  // Override: don't re-warmup, just re-capture
    hasReplayHandle = false;
    replayHandleNull = true;
    isTritonCompiled = false;
    extAddrsStable = false;
  }

  if (allowTritonCudaGraphReplay && seg.exec.replayHandle != nullptr &&
      seg.exec.replayHandle->isReady() && !isTritonCompiled) {
    DSP_DIAG(EXECUTE, "TRITON_REPLAY_SKIP: seg[%d-%d] has replayHandle but compiledBy='%s' (not %s) "
                      "→ falling through to raw CUDA graph replay path",
             seg.def.startSlot, seg.def.endSlot,
             seg.exec.compiledByBackend.empty() ? "(empty)" : seg.exec.compiledByBackend.c_str(),
             backendName);
  }

  // ── REPLAY DISPATCH ───────────────────────────────────────────────────────
  {
    auto replayResult = segDispatchReplay(seg, externalArrays, numExt, stream,
                                          allowTritonCudaGraphReplay,
                                          createValuesStable, extAddrsStable,
                                          segShapeKey, backendName);
    if (replayResult == Status::OK) {
      DSP_TRACE_GRAPH_REPLAYED(trace_, static_cast<int8_t>(segIdx),
                               static_cast<uint32_t>(executeCount_),
                               seg.def.startSlot, seg.def.endSlot);
    } else if (replayResult != Status::MAYBE) {
      DSP_TRACE_ERROR(trace_, static_cast<int8_t>(segIdx), seg.def.startSlot,
                      static_cast<uint32_t>(executeCount_),
                      static_cast<uint64_t>(replayResult));
    }
    if (replayResult != Status::MAYBE) return replayResult;
  }
  // Fall through to capture or direct execution

  // ── CAPTURE + DIRECT EXECUTION ─────────────────────────────────────────────
  {
    SegCaptureCtx ctx;
    ctx.segIdx = segIdx;
    ctx.segShapeKey = segShapeKey;
    ctx.segInputAddrKey = segInputAddrKey;
    ctx.createValueKey = createValueKey;
    ctx.backendName = backendName;
    ctx.backend = backend;
    ctx.cudaStr = cudaStr;
#if HAVE_TRITON
    ctx.tritonBackend = tritonBackend;
#endif
    ctx.captureMinExec = captureMinExec;
    ctx.forceRecaptureEnabled = forceRecaptureEnabled;
    ctx.allowTritonCudaGraphReplay = allowTritonCudaGraphReplay;
    ctx.requiresOrderedGapCapture = requiresOrderedGapCapture;
    ctx.hasCudaStream = hasCudaStream;
    return segDispatchCaptureOrDirect(seg, externalArrays, numExt, stream, ctx);
  }

}  // executeSegmentWithGpuGraph

}  // namespace graph
}  // namespace sd

#endif  // SD_CUDA

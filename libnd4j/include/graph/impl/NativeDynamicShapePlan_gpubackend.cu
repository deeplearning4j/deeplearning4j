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
#include <graph/DspThreadState.h>
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
#include <mutex>
#include <atomic>
#include <condition_variable>

#if HAVE_TRITON
#include <graph/gpu/TritonGraphBackend.h>
#endif

#include <cublas_v2.h>  // N6: cublasSetStream_v2 / cublasSetWorkspace for hoisted gap setup

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
// cuBLAS workspace thread-locals — defined in MmulHelper.cu / DataBuffer.cu.
extern SD_TLS_EXPORT thread_local void*  tl_cublasWorkspacePtr;
extern SD_TLS_EXPORT thread_local size_t tl_cublasWorkspaceSize;
// N6: defined in NativeDynamicShapePlan_batchgemm.cu — hoisted cuBLAS stream+workspace flag.
extern SD_TLS_EXPORT thread_local bool tl_cublasGapStreamReady;
// cuBLAS Lt disable flag — defined in DataBuffer.cu (inside namespace sd). Temporarily cleared
// during gap execution to enable cublasLt fast path for logits-projection matmul [1,K]×[K,N].
namespace sd { extern SD_TLS_EXPORT thread_local bool tl_cublasLtDisabled; }

// Portable buffer accessor (CUDA form).
#define DSP_BUF(arr) ((arr)->specialBuffer())

// ── Isolation flags for debugging composite replay accuracy ──
// -fno-threadsafe-statics: use std::call_once for thread-safe initialization.
static bool dsp_disable_view_fastpath() {
  static std::once_flag f; static bool v = false;
  std::call_once(f, []() { v = (std::getenv("ND4J_DSP_DISABLE_VIEW_FASTPATH") != nullptr); });
  return v;
}
static bool dsp_disable_cast_hwm() {
  static std::once_flag f; static bool v = false;
  std::call_once(f, []() { v = (std::getenv("ND4J_DSP_DISABLE_CAST_HWM") != nullptr); });
  return v;
}
static bool dsp_disable_workspace_skip() {
  static std::once_flag f; static bool v = false;
  std::call_once(f, []() { v = (std::getenv("ND4J_DSP_DISABLE_WS_SKIP") != nullptr); });
  return v;
}

namespace sd {
namespace graph {

// ── Global shared capture workspace ─────────────────────────────────────
// CUDA graph capture is serialized by DeviceCaptureGuard (only one plan
// captures at a time per device). So the capture workspace can be shared
// globally across all plan instances. This avoids the OOM that occurs when
// multiple concurrent plans each try to allocate their own 512MB workspace.
// Protected by DeviceCaptureGuard (no additional locking needed).
void* g_globalCaptureWorkspace = nullptr;
size_t g_globalCaptureWorkspaceBytes = 0;
int g_globalCaptureWorkspaceDevice = -1;

// ── Per-GPU CUDA graph capture/execution coordination ───────────────────
// Defined in _cuda.cu. Capture sets captureActive, waits for execCount==0,
// then proceeds. Execution waits while captureActive is true.
extern std::atomic<bool> g_captureActive[16];
extern std::mutex g_captureMtx[16];
extern std::condition_variable g_captureCV[16];
extern std::atomic<int> g_execCount[16];

static bool slotHasOnlyTransparentAliasOutputs(
    const NativeSlot& slot,
    const SlotBufferInfo* ownership,
    NDArray** outputSlots,
    NDArray** externalArrays,
    int numExt,
    int totalOutputSlots) {
  // Replay-stable host-only classes: shape metadata, constants, fused tails,
  // and aliasing views/identity below. Keep this in sync with CUDA graph
  // coverage validation.
  if (slot.frozenConstantSlot() ||
      slot.hasOpTrait(sd::ops::OP_TRAIT_SHAPE_ONLY_OUTPUT) ||
      slot.hasOpTrait(sd::ops::OP_TRAIT_CONSTANT_GENERATION) ||
      slot.fusedChain.isFusedChainTail) {
    return true;
  }

  if (!(slot.isViewCapableOp() || slot.isIdentityOp()) ||
      slot.wiring.numOutputs <= 0 ||
      ownership == nullptr) {
    return false;
  }

  for (int o = 0; o < slot.wiring.numOutputs; o++) {
    int outIdx = slot.wiring.outputSlotIndices[o];
    if (outIdx < 0 || outIdx >= totalOutputSlots) return false;

    BufferOwnership owner = ownership[outIdx].ownership;
    if (owner != BufferOwnership::VIEW_OF_SLOT &&
        owner != BufferOwnership::VIEW_OF_WEIGHT) {
      NDArray* out = (outputSlots != nullptr) ? outputSlots[outIdx] : nullptr;
      DataBuffer* outDb = (out != nullptr) ? out->dataBuffer() : nullptr;
      bool aliasesExternalInput = false;
      if (outDb != nullptr && externalArrays != nullptr) {
        for (int i = 0; i < slot.wiring.numInputs; i++) {
          int srcIdx = slot.wiring.inputSourceIndices[i];
          if (srcIdx >= 0) continue;
          int extIdx = -(srcIdx + 1);
          if (extIdx >= 0 && extIdx < numExt &&
              externalArrays[extIdx] != nullptr &&
              externalArrays[extIdx]->dataBuffer() == outDb) {
            aliasesExternalInput = true;
            break;
          }
        }
      }
      if (!aliasesExternalInput) return false;
    }
  }
  return true;
}

// RAII guard for CUDA graph capture: signals "capture active" on this device,
// waits for all OTHER concurrent executions to drain, then holds exclusive
// access until destruction.
//
// Capture is always initiated from within execute(), so the current thread
// has already incremented g_execCount. We temporarily decrement it so the
// wait condition (execCount == 0) only checks OTHER threads. The destructor
// re-increments it to restore the invariant.
struct DeviceCaptureGuard {
  int dev_;
  std::unique_lock<std::mutex> lock_;   // held for entire capture duration
  bool acquired_;                        // true if capture lock was obtained

  // Try to acquire the capture guard. If another thread is already capturing,
  // this constructor returns immediately with acquired_=false instead of
  // blocking. This prevents deadlock when multiple concurrent threads each
  // try to capture simultaneously — the non-winning threads skip capture
  // for this execution and retry on the next call.
  explicit DeviceCaptureGuard()
    : dev_(0), lock_(), acquired_(false) {
    cudaGetDevice(&dev_);
    if (dev_ < 0 || dev_ >= 16) dev_ = 0;

    // Try to acquire the mutex. If another thread is capturing, return
    // immediately — the caller checks acquired() and skips capture.
    lock_ = std::unique_lock<std::mutex>(g_captureMtx[dev_], std::try_to_lock);
    if (!lock_.owns_lock()) {
      return;  // Another thread is capturing — skip this attempt
    }

    // Temporarily remove this thread from the exec count
    g_execCount[dev_].fetch_sub(1, std::memory_order_acq_rel);
    // Signal that capture is starting — new executions will wait
    g_captureActive[dev_].store(true, std::memory_order_release);
    // Wait for all OTHER in-flight executions to finish.
    // Use a timed wait to prevent deadlock: if other threads are also
    // stuck trying to capture (they'll fail try_lock now), we only wait
    // for threads that are genuinely executing (not blocked on capture).
    bool waitResult = g_captureCV[dev_].wait_for(lock_, std::chrono::seconds(5),
        [this]{ return g_execCount[dev_].load(std::memory_order_acquire) == 0; });
    if (!waitResult) {
      // Timeout: other threads didn't finish in time. Abort capture.
      g_captureActive[dev_].store(false, std::memory_order_release);
      g_execCount[dev_].fetch_add(1, std::memory_order_acq_rel);
      lock_.unlock();
      g_captureCV[dev_].notify_all();
      return;  // acquired_ stays false
    }
    acquired_ = true;
    // lock_ stays held — no other thread can enter capture until destruction
  }
  ~DeviceCaptureGuard() {
    if (acquired_) {
      g_captureActive[dev_].store(false, std::memory_order_release);
      // Re-add this thread to the exec count (we're still executing)
      g_execCount[dev_].fetch_add(1, std::memory_order_acq_rel);
      // Release the mutex, then notify waiters
      lock_.unlock();
      g_captureCV[dev_].notify_all();
    }
  }
  bool acquired() const { return acquired_; }
  DeviceCaptureGuard(const DeviceCaptureGuard&) = delete;
  DeviceCaptureGuard& operator=(const DeviceCaptureGuard&) = delete;
};

// File-level alias for the nested enum — avoids GraphSegmentExec:: prefix at call sites.
using SegmentLifecycleState = GraphSegmentExec::SegmentLifecycleState;

// SegmentLifecycle transition functions are defined as static inline in
// <graph/DspSegmentLifecycle.h>. Bring them into scope without the namespace prefix.
using namespace SegmentLifecycle;

// REMOVED: syncCrossStream — all cross-stream sync now goes through
// performPreReplaySync (tracked via PreReplaySyncPhase).

// ── Gap-stream scope helper ───────────────────────────────────────────────
struct ScopedGapStreamOverride {
  cudaStream_t prev = nullptr;
  explicit ScopedGapStreamOverride(cudaStream_t stream) : prev(tl_dspGapStream) {
    tl_dspGapStream = stream;
  }
  ~ScopedGapStreamOverride() { tl_dspGapStream = prev; }
};

// REMOVED: syncExternalInputs — all H2D sync now goes through
// performPreReplaySync (tracked via PreReplaySyncPhase).

// REMOVED: platformPreSegmentExec, syncCrossStream, syncExternalInputs —
// all were parallel sync paths that competed with performPreReplaySync.
// ALL sync now goes through performPreReplaySync (called from dispatchSegment),
// tracked via PreReplaySyncPhase. ONE function, ONE state machine.

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

// ── RAII guard: ensures tl_graphExecutionActive is ALWAYS cleared ───────────
// when the enclosing capture scope exits (normal return, exception, break, etc.)
// activate() is explicit so the guard can live at outer scope for merged capture
// while waiting for the capture to actually begin.
struct CaptureLifecycleGuard {
  bool active_ = false;
  void activate() {
    DSP_DIAG(EXECUTE, "CaptureLifecycleGuard::activate — tl_graphExecutionActive: %d->1 "
             "tl_graphCaptureStream=%p", (int)tl_graphExecutionActive, (void*)tl_graphCaptureStream);
    tl_graphExecutionActive = true;
    active_ = true;
  }
  void deactivate() {
    if (active_) {
      DSP_DIAG(EXECUTE, "CaptureLifecycleGuard::deactivate — tl_graphExecutionActive: %d->0 "
               "tl_graphCaptureStream=%p->nullptr", (int)tl_graphExecutionActive, (void*)tl_graphCaptureStream);
      tl_graphExecutionActive = false;
      tl_graphCaptureStream = nullptr;
      active_ = false;
    }
  }
  ~CaptureLifecycleGuard() { deactivate(); }
};

// Full abort: cleanup TLS + pop context + restore cuBLAS + restore slot state + destroy handle.
// This is the pattern repeated at every early-return from capture.
void NativeDynamicShapePlan::abortCapture(GraphSegment& seg,
                                          bool freeHostPtrs,
                                          bool didPushCtx, int captureDevice,
                                          cudaStream_t prevCaptureStream,
                                          const std::vector<SlotPhase>& savedSlotPhases,
                                          void* stream) {
  DSP_DIAG(EXECUTE, "abortCapture: seg[%d-%d] freeHostPtrs=%d didPushCtx=%d captureDevice=%d "
           "tl_graphExecutionActive=%d tl_cublasWorkspacePtr=%p/%zu",
           seg.def.startSlot, seg.def.endSlot, (int)freeHostPtrs, (int)didPushCtx, captureDevice,
           (int)tl_graphExecutionActive,
           (void*)tl_cublasWorkspacePtr, tl_cublasWorkspaceSize);
  cleanupCaptureTlsState(freeHostPtrs, prevCaptureStream);
  popPrimaryCtxIfPushed(didPushCtx, captureDevice);
  restoreCublasWorkspaceAfterCapture(stream);
  if (!savedSlotPhases.empty()) {
    for (int s = seg.def.startSlot; s <= seg.def.endSlot; s++) {
      slots_[s].slotPhase = savedSlotPhases[s - seg.def.startSlot];  // PRIMARY restore
    }
  }
  cleanupSegmentForRebuild(seg, "capture_abort");
}

// ── Slot state save/restore helpers ───────────────────────────────────────
// Demote FROZEN→SHAPE_CACHED for warmup/capture, preserving FROZEN_CONSTANT.
static void demoteFrozenSlotStates(NativeSlot* slots, int startSlot, int endSlot,
                                   std::vector<SlotPhase>& savedPhases) {
  savedPhases.resize(endSlot - startSlot + 1);
  for (int s = startSlot; s <= endSlot; s++) {
    savedPhases[s - startSlot] = slots[s].slotPhase;  // PRIMARY save
    if (slots[s].slotPhase.isSealed() && !slots[s].slotPhase.isConstant) {
      slots[s].slotPhase.unseal();
      slots[s].slotPhase.shapeCacheValid = true;  // PRIMARY demote
    }
  }
}

static void restoreSlotStates(NativeSlot* slots, int startSlot, int endSlot,
                              const std::vector<SlotPhase>& savedPhases) {
  for (int s = startSlot; s <= endSlot; s++) {
    slots[s].slotPhase = savedPhases[s - startSlot];  // PRIMARY restore
  }
}

// ── View-capable slot promotion helper ────────────────────────────────────
static int promoteViewCapableSlotsToFrozen(NativeSlot* slots, int startSlot, int endSlot) {
  int promoted = 0;
  for (int s = startSlot; s <= endSlot; s++) {
    auto& sl = slots[s];
    if (!sl.isViewCapableOp() || sl.slotPhase.isSealed()) continue;
    sl.slotPhase.seal(false);  // PRIMARY: promote to SEALED (non-constant)
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
  DSP_DIAG(EXECUTE, "%s seg[%d-%d] slot=%d len=%lld sbuf=%p execCount=%d "
           "(async path: argmax/value dump skipped)",
           label, seg.def.startSlot, seg.def.endSlot, finalOutputSlot,
           (long long)out->lengthOf(), DSP_BUF(out), execCount);
}

static bool validateAndStoreMergedCapture(
    const char* diagPrefix,
    sd::cuda::CudaGraphHandle* nativeHandle,
    std::unique_ptr<GraphReplayHandle>& handle,
    ReplaySchedule& sched,
    int mergedGroupId, int startSlot, int endSlot,
    size_t nodeCount, void* stream, cudaStream_t cudaStr) {

  // ── Node type breakdown ──────────────────────────────────────────────
  // Log the full graph composition so crashes are self-diagnosing.
  auto stats = nativeHandle->getStatistics();
  DSP_DIAG(EXECUTE, "%s: group=%d [%d-%d] VALIDATE_BEGIN nodes=%zu "
           "[kernels=%d memcpyH2D=%d memcpyD2D=%d memcpyD2H=%d "
           "memsets=%d memAllocs=%d memFrees=%d hostCb=%d events=%d empty=%d] "
           "tl_graphExecutionActive=%d "
           "tl_captureWorkspace=%p tl_cublasWorkspacePtr=%p/%zu",
           diagPrefix, mergedGroupId, startSlot, endSlot, nodeCount,
           stats.numKernels, stats.numMemcpyH2D, stats.numMemcpyD2D, stats.numMemcpyD2H,
           stats.numMemsets, stats.numMemAllocs, stats.numMemFrees,
           stats.numHostCallbacks, stats.numEvents, stats.numEmpty,
           (int)tl_graphExecutionActive,
           (void*)tl_captureWorkspace, (void*)tl_cublasWorkspacePtr, tl_cublasWorkspaceSize);

  if (stats.numMemAllocs > 0 || stats.numMemFrees > 0) {
    DSP_DIAG(MEMORY, "%s: group=%d has %d MemAlloc + %d MemFree nodes "
             "(cuBLAS internal workspace — CUDA 12+ handles these on replay)",
             diagPrefix, mergedGroupId, stats.numMemAllocs, stats.numMemFrees);
  }
  if (stats.numHostCallbacks > 0) {
    DSP_DIAG(BACKEND, "%s: WARNING — group=%d has %d host callback nodes "
             "(these are NOT replay-safe and will cause SIGSEGV on cudaGraphLaunch)",
             diagPrefix, mergedGroupId, stats.numHostCallbacks);
  }

  // ── Pre-instantiate: verify stream is NOT still capturing ────────────
  {
    cudaStreamCaptureStatus capStat = cudaStreamCaptureStatusNone;
    cudaError_t capErr = cudaStreamGetCaptureInfo_v2(cudaStr, &capStat, nullptr, nullptr, nullptr, nullptr);
    if (capErr != cudaSuccess || capStat != cudaStreamCaptureStatusNone) {
      DSP_DIAG(EXECUTE, "%s: group=%d ABORT — stream %p still in capture mode "
               "(capStat=%d capErr=%d). Cannot instantiate/launch.",
               diagPrefix, mergedGroupId, (void*)cudaStr, (int)capStat, (int)capErr);
      if (capErr != cudaSuccess) cudaGetLastError();
      return false;
    }
  }

  // ── Clear any sticky CUDA error before instantiate ───────────────────
  {
    cudaError_t stickyErr = cudaGetLastError();
    if (stickyErr != cudaSuccess) {
      DSP_DIAG(EXECUTE, "%s: group=%d cleared sticky CUDA error %d (%s) before instantiate",
               diagPrefix, mergedGroupId, (int)stickyErr, cudaGetErrorString(stickyErr));
    }
  }

  // ── Pre-instantiation memory gate ──────────────────────────────────────
  // Each cudaGraphExec_t instantiation reserves GPU memory for graph metadata,
  // kernel arguments, and dependency tables. With many composite islands (e.g. 76
  // in the vision encoder), cumulative memory can exhaust the GPU, causing
  // error 700 during subsequent gap ops that need workspace allocations.
  // Check free memory BEFORE instantiation to bail out cleanly.
  {
    size_t gpuFree = 0, gpuTotal = 0;
    cudaMemGetInfo(&gpuFree, &gpuTotal);
    // Reserve enough headroom for gap op workspaces (xw_plus_b needs ~12MB each,
    // cuBLAS workspace is ~256MB). Use configurable safety threshold, default 384MB.
    size_t perIslandSafetyBytes = static_cast<size_t>(
        Environment::getInstance().dsp().graphMetadataSafetyMb()) * 1024ULL * 1024ULL;
    // Scale safety by node count — larger graphs need more instantiation memory
    size_t scaledSafety = perIslandSafetyBytes + (nodeCount * 4096ULL);
    if (gpuFree < scaledSafety) {
      DSP_DIAG(MEMORY, "%s: group=%d PRE-INSTANTIATE OOM GATE: gpuFree=%zuMB < safety=%zuMB "
               "(nodes=%zu, perIslandSafety=%zuMB). Bailing out cleanly to prevent error 700.",
               diagPrefix, mergedGroupId, gpuFree / (1024*1024), scaledSafety / (1024*1024),
               nodeCount, perIslandSafetyBytes / (1024*1024));
      return false;
    }
  }

  bool instOk = nativeHandle->instantiate();
  if (!instOk) {
    cudaError_t instErr = cudaGetLastError();
    DSP_DIAG(EXECUTE, "%s: group=%d instantiate FAILED — cudaGetLastError=%d (%s) "
             "wasOom=%d nodes=%zu",
             diagPrefix, mergedGroupId, (int)instErr, cudaGetErrorString(instErr),
             nativeHandle->wasLastInstantiateOom() ? 1 : 0, nodeCount);
    return false;
  }
  DSP_DIAG(EXECUTE, "%s: group=%d instantiate OK graphExec=%p — launching async validation replay "
           "on stream=%p cudaStr=%p device=%d",
           diagPrefix, mergedGroupId, (void*)nativeHandle->getGraphExec(),
           stream, (void*)cudaStr, nativeHandle->getDeviceId());

  bool launchOk = handle->replay(stream);
  if (!launchOk) {
    cudaError_t lastErr = cudaGetLastError();
    DSP_DIAG(EXECUTE, "%s: group=%d validation replay FAILED — cudaGraphLaunch error=%d (%s) "
             "graphExec=%p nodes=%zu stream=%p",
             diagPrefix, mergedGroupId, (int)lastErr, cudaGetErrorString(lastErr),
             (void*)nativeHandle->getGraphExec(), nodeCount, (void*)cudaStr);
    return false;
  }
  DSP_DIAG(EXECUTE, "%s: group=%d [%d-%d] VALIDATE_QUEUED nodes=%zu "
           "(async launch accepted)",
           diagPrefix, mergedGroupId, startSlot, endSlot, nodeCount);
  sched.mergedReplayHandles.push_back(std::move(handle));
  return true;
}

// Default capture workspace sizes (configurable via env vars)
static size_t TRITON_CAPTURE_HOST_WORKSPACE_SIZE = []() -> size_t {
  size_t mb = static_cast<size_t>(Environment::getInstance().dsp().captureHostWorkspaceMb());
  return mb * 1024ULL * 1024ULL;
}();

// Capture workspace size for Triton graph capture (default 512MB).
// Configurable via ND4J_DSP_CAPTURE_WORKSPACE_MB env var or Java property nd4j.dsp.captureWorkspaceMb.
// Read dynamically so Java-side property changes (e.g., in tests) take effect.
static inline size_t tritonCaptureWorkspaceSize() {
  size_t mb = static_cast<size_t>(Environment::getInstance().dsp().captureWorkspaceMb());
  return mb * 1024ULL * 1024ULL;
}

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

// ── Merged capture context flag ──────────────────────────────────────────
// When true, the tritonOrderedRangeGuard should NOT skip gap ops during
// capture. Merged capture records both islands AND their intervening gaps
// into a single CUDA graph. Gap ops within an island that are called via
// orderedRangeExecutor during merged capture MUST execute on the capture
// stream so their kernels are recorded. Without this flag, the guard
// unconditionally skips gaps when streamIsCapturing=true, which is only
// correct for per-island capture (where gaps are replayed fresh each step).
static thread_local bool tl_mergedCaptureActive = false;
// During merged capture, gap ops in the orderedRangeExecutor must use staging
// buffer externals (plan-owned stable addresses) instead of the original Java-
// side external arrays. This pointer is set before capture begins and cleared
// after. When non-null, the orderedRangeExecutor uses this instead of the
// captured externalArrays.
static thread_local NDArray** tl_mergedCaptureExternals = nullptr;

// Reset file-static merged-capture TLS — called from platformFreePlanResources()
// via extern linkage to ensure exception-escape leaks don't contaminate the next plan.
void resetMergedCaptureTLS() {
  tl_mergedCaptureActive = false;
  tl_mergedCaptureExternals = nullptr;
}

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

static void snapshotSlotOutputBuffers(const NativeSlot& slot, NDArray** outputSlots, int totalOutputSlots,
                                      void** outputBufs, int maxOutputs) {
  std::memset(outputBufs, 0, sizeof(void*) * maxOutputs);
  int numOutputs = slot.wiring.numOutputs < maxOutputs ? slot.wiring.numOutputs : maxOutputs;
  for (int i = 0; i < numOutputs; i++) {
    int outSi = slot.wiring.outputSlotIndices[i];
    if (outSi < 0 || outSi >= totalOutputSlots) continue;
    NDArray* outArr = outputSlots[outSi];
    outputBufs[i] = outArr != nullptr ? DSP_BUF(outArr) : nullptr;
  }
}

static bool slotOutputBuffersChanged(const NativeSlot& slot, NDArray** outputSlots, int totalOutputSlots,
                                     void** outputBufsBefore, int maxOutputs) {
  int numOutputs = slot.wiring.numOutputs < maxOutputs ? slot.wiring.numOutputs : maxOutputs;
  for (int i = 0; i < numOutputs; i++) {
    int outSi = slot.wiring.outputSlotIndices[i];
    if (outSi < 0 || outSi >= totalOutputSlots) continue;
    NDArray* outArr = outputSlots[outSi];
    void* currentBuf = outArr != nullptr ? DSP_BUF(outArr) : nullptr;
    if (currentBuf != outputBufsBefore[i]) {
      return true;
    }
  }
  return false;
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
  // Build a detailed error message with actionable fix info — never silently fail.
  char oomMsg[1024];
  std::snprintf(oomMsg, sizeof(oomMsg),
      "DSP OOM in seg[%d-%d] during '%s' on device %d: "
      "requested=%zuMB gpuFree=%zuMB gpuTotal=%zuMB gpuUsed=%zuMB "
      "captureWorkspace=%zuMB executionCount=%d. "
      "FIX: increase -Dnd4j.dsp.captureWorkspaceMb (current=%zuMB)",
      seg.def.startSlot, seg.def.endSlot, phase, deviceId,
      requestedBytes / (1024*1024), gpuFree / (1024*1024),
      gpuTotal / (1024*1024), (gpuTotal - gpuFree) / (1024*1024),
      tritonCaptureWorkspaceSize() / (1024*1024),
      seg.exec.executionCount,
      tritonCaptureWorkspaceSize() / (1024*1024));
  DSP_DIAG(MEMORY, "%s", oomMsg);
  SegmentLifecycle::markFailed(seg.exec, "oom", seg.def.startSlot, seg.def.endSlot);
  // Throw instead of silently returning KERNEL_FAILURE — OOM must produce a stack trace.
  THROW_EXCEPTION(oomMsg);
  return Status::KERNEL_FAILURE;  // unreachable, but satisfies return type
}

static Status reportCaptureError(NativeDynamicShapePlan* plan, GraphSegment& seg,
                                 const char* step,
                                 cudaError_t cudaErr, int deviceId) {
  dumpGpuContextState(deviceId, "CAPTURE");
  size_t gpuFree = 0, gpuTotal = 0;
  cudaMemGetInfo(&gpuFree, &gpuTotal);
  char captureMsg[1024];
  std::snprintf(captureMsg, sizeof(captureMsg),
      "CAPTURE ERROR in seg[%d-%d] at step '%s' on device %d: "
      "cudaError=%d (%s) gpuFree=%zuMB gpuTotal=%zuMB "
      "executionCount=%d numOps=%d compiledBy='%s'",
      seg.def.startSlot, seg.def.endSlot, step, deviceId,
      static_cast<int>(cudaErr), cudaGetErrorString(cudaErr),
      gpuFree / (1024*1024), gpuTotal / (1024*1024),
      seg.exec.executionCount, seg.def.endSlot - seg.def.startSlot + 1,
      seg.exec.compiledByBackend.c_str());
  DSP_DIAG(EXECUTE, "%s", captureMsg);
  // Capture errors are recoverable — reset segment for retry via lifecycle.
  // The CUDA context is NOT poisoned after endCapture + cudaGetLastError
  // clears the sticky error.
  cudaGetLastError(); // clear error state
  SegmentLifecycle::markCaptureErrorRetry(plan, seg, step);
  return Status::KERNEL_FAILURE;
}

static Status reportReplayError(GraphSegment& seg, const char* step,
                                cudaError_t cudaErr, int deviceId) {
  dumpGpuContextState(deviceId, "REPLAY");
  size_t gpuFree = 0, gpuTotal = 0;
  cudaMemGetInfo(&gpuFree, &gpuTotal);
  char replayMsg[1024];
  std::snprintf(replayMsg, sizeof(replayMsg),
      "REPLAY ERROR in seg[%d-%d] at step '%s' on device %d: "
      "cudaError=%d (%s) gpuFree=%zuMB gpuTotal=%zuMB "
      "executionCount=%d hasReplayHandle=%d",
      seg.def.startSlot, seg.def.endSlot, step, deviceId,
      static_cast<int>(cudaErr), cudaGetErrorString(cudaErr),
      gpuFree / (1024*1024), gpuTotal / (1024*1024),
      seg.exec.executionCount,
      seg.exec.replayHandle != nullptr ? 1 : 0);
  DSP_DIAG(EXECUTE, "%s", replayMsg);
  SegmentLifecycle::markFailed(seg.exec, step, seg.def.startSlot, seg.def.endSlot);
  cudaGetLastError(); // clear error state
  // Throw with full context — replay errors must never be silent.
  THROW_EXCEPTION(replayMsg);
  return Status::KERNEL_FAILURE;  // unreachable
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
// Query whether every slot in [startSlot, endSlot] can be inside a CUDA graph
// capture.  Delegates to NativeSlot::isCapturable() — the single source of
// truth for control-flow, data-dependent, and view/identity/frozen checks.
static bool isGapRangeCaptureSafe(NativeSlot* slots, int startSlot, int endSlot, bool mergeViews) {
  // A gap is safe to merge into a CUDA graph capture if ALL its slots either:
  //   (a) are zero-compute ops (views, identity, frozen constants), OR
  //   (b) are non-cuBLAS compute ops whose args are all plan-owned pointers.
  //
  // Category (b) is safe because:
  //   - syncToHost() is a no-op during capture (tl_graphExecutionActive guard)
  //   - effectiveExternalsForCapture provides stable staging pointers for ext inputs
  //   - outputSlots_ pointers are stable once pointersStable()=true
  //   - only cuBLAS ops have external state (handle/workspace) that goes stale
  //   - Gather indices change per step but pointer args don't (data changes, not addresses)
  //   - VALUE_DEPENDENT_SHAPE only affects shape function (not called during capture)
  //
  // We limit captured gap size to avoid lifecycle conflicts in large gaps.
  // Small gaps (≤8 slots) between islands are "glue" ops (equals, broadcast_to,
  // Where, gather) that create unnecessary island boundaries. Merging them
  // through reduces the number of graph launch/end transitions.
  // Larger gaps may contain view ops that trigger frozen DataBuffer conflicts
  // during capture-phase allocation.
  // Configurable via ND4J_DSP_MAX_CAPTURABLE_GAP_SLOTS env var (default 32).
  const int maxCapturableGapSlots = Environment::getInstance().dspMaxCapturableGapSlots();
  // Configurable via ND4J_DSP_GAP_CAPTURE_BLOCK_EXTERNAL_WORKSPACE env var (default true).
  // When true, ops declaring OP_TRAIT_EXTERNAL_WORKSPACE (cuBLAS matmul, etc.) are
  // excluded from gap capture. These ops use external library handles/workspaces that
  // may not replay correctly in CUDA graphs without explicit workspace pinning.
  const bool blockExtWorkspace = Environment::getInstance().dspGapCaptureBlockExternalWorkspace();

  int gapSize = endSlot - startSlot + 1;
  if (gapSize > maxCapturableGapSlots) {
    DSP_DIAG(SEGMENT,
             "isGapRangeCaptureSafe [%d-%d] UNSAFE: gapSize=%d > maxCapturableGapSlots=%d",
             startSlot, endSlot, gapSize, maxCapturableGapSlots);
    return false;
  }

  for (int s = startSlot; s <= endSlot; s++) {
    if (!slots[s].isCapturable(mergeViews)) {
      DSP_DIAG(SEGMENT,
               "isGapRangeCaptureSafe [%d-%d] UNSAFE: slot=%d op='%s' isCapturable=false",
               startSlot, endSlot, s, slots[s].ident.opName.c_str());
      return false;
    }
    // Zero-compute ops (view/identity/frozen) are always safe — no GPU kernel nodes.
    if (slots[s].aliasesInput() || slots[s].frozenConstantSlot()) continue;
    // Block ops that use external library workspaces (cuBLAS, etc.) — trait-driven, not op-specific.
    if (blockExtWorkspace && slots[s].hasOpTrait(sd::ops::OP_TRAIT_EXTERNAL_WORKSPACE)) {
      DSP_DIAG(SEGMENT,
               "isGapRangeCaptureSafe [%d-%d] UNSAFE: slot=%d op='%s' has OP_TRAIT_EXTERNAL_WORKSPACE",
               startSlot, endSlot, s, slots[s].ident.opName.c_str());
      return false;
    }
    // Block ops with dynamic output sizes (Where, unique) — they allocate during execution,
    // which poisons CUDA graph capture (cudaStreamCaptureStatusInvalidated).
    if (slots[s].hasOpTrait(sd::ops::OP_TRAIT_DYNAMIC_OUTPUT_SIZE)) {
      DSP_DIAG(SEGMENT,
               "isGapRangeCaptureSafe [%d-%d] UNSAFE: slot=%d op='%s' has OP_TRAIT_DYNAMIC_OUTPUT_SIZE",
               startSlot, endSlot, s, slots[s].ident.opName.c_str());
      return false;
    }
    // Block non-fully-writing ops (reduce, scatter) that need prezero.
    if (!slots[s].isFullyWriting()) {
      DSP_DIAG(SEGMENT,
               "isGapRangeCaptureSafe [%d-%d] UNSAFE: slot=%d op='%s' isFullyWriting=false",
               startSlot, endSlot, s, slots[s].ident.opName.c_str());
      return false;
    }
  }
  return true;
}

static ReplaySchedule buildCompositeReplaySchedule(const GraphSegment& seg,
                                                  NativeSlot* slots,
                                                  TritonGraphBackend* tritonBackend) {
 ReplaySchedule schedule;
 auto gap_slots = tritonBackend->getGapSlots(seg, slots);

 int totalSegSlots = seg.def.endSlot - seg.def.startSlot + 1;
 DSP_DIAG(SEGMENT,
          "buildCompositeReplaySchedule ENTER: seg[%d-%d] totalSlots=%d gapSlots=%d islandSlots=%d firstSlotIsGap=%d",
          seg.def.startSlot, seg.def.endSlot, totalSegSlots,
          static_cast<int>(gap_slots.size()),
          totalSegSlots - static_cast<int>(gap_slots.size()),
          (gap_slots.find(seg.def.startSlot) != gap_slots.end()) ? 1 : 0);

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
 // A gap is capture-safe iff ALL its slots are zero-copy metadata ops
 // (views, identity, frozen constants) — OR the gap contains compute ops
 // that launch CUDA kernels without allocating memory.
 //
 // When mergedCaptureThroughViews is enabled, gaps are classified as
 // Gap capturability is queried on-demand via isGapRangeCaptureSafe()
 // at the point of use — no cached isCaptureSafe flag to get out of sync.
 if (DSP_DIAG_ENABLED(SEGMENT)) {
   bool mv = Environment::getInstance().triton().mergedCaptureThroughViews();
   for (auto& unit : schedule.units) {
     if (unit.kind != REPLAY_UNIT_GAP) continue;
     DSP_DIAG_SEG(SEGMENT, unit.startSlot,
                  "compositeReplaySchedule GAP [%d-%d] captureSafe=%d",
                  unit.startSlot, unit.endSlot,
                  isGapRangeCaptureSafe(slots, unit.startSlot, unit.endSlot, mv) ? 1 : 0);
   }
 }

 // Pre-allocate replay handles for each island
 schedule.compositeReplayHandles.resize(schedule.units.size());

 // Exit summary: count islands vs gaps in the final schedule
 {
   int nIslands = 0, nGaps = 0;
   for (const auto& u : schedule.units) {
     if (u.kind == REPLAY_UNIT_TRITON_ISLAND) nIslands++;
     else nGaps++;
   }
   DSP_DIAG(SEGMENT,
            "buildCompositeReplaySchedule EXIT: seg[%d-%d] schedule units=%d (islands=%d gaps=%d) "
            "compositeHandlesAllocated=%d",
            seg.def.startSlot, seg.def.endSlot,
            static_cast<int>(schedule.units.size()), nIslands, nGaps,
            static_cast<int>(schedule.compositeReplayHandles.size()));
 }

 return schedule;
}
#endif  // HAVE_TRITON

Status NativeDynamicShapePlan::compositeReplay(
    GraphSegment& seg, ReplaySchedule& sched,
    NDArray** externalArrays, int numExt, void* stream) {

  using Clock = std::chrono::high_resolution_clock;
  auto t0 = executionTimingEnabled_ ? Clock::now() : Clock::time_point{};

  // Entry: log unit count and type breakdown so the replay trace is self-contained.
  {
    int nIslands = 0, nGaps = 0, nMergedLeaders = 0;
    for (const auto& u : sched.units) {
      if (u.kind == REPLAY_UNIT_TRITON_ISLAND) nIslands++;
      else nGaps++;
      if (u.isMergedLeader) nMergedLeaders++;
    }
    DSP_DIAG(EXECUTE,
             "COMPOSITE_REPLAY_ENTRY: seg[%d-%d] units=%d (islands=%d gaps=%d) "
             "mergedGroups=%d mergedLeaders=%d execCount=%d planPhase=%s",
             seg.def.startSlot, seg.def.endSlot,
             static_cast<int>(sched.units.size()), nIslands, nGaps,
             static_cast<int>(sched.mergedReplayHandles.size()), nMergedLeaders,
             seg.exec.executionCount, planLifecycle_.displayName());
  }

  // Phase assertion: compositeReplay MUST be called in SHAPES_FROZEN or later.
  // Calling during SLOT_BY_SLOT means shapes aren't stable and graph replay is unsafe.
  if (planLifecycle_.isSlotBySlot()) {
    DSP_DIAG(EXECUTE, "PHASE_VIOLATION: compositeReplay called in phase %s, "
                      "requires >= SHAPES_FROZEN. seg[%d-%d] execCount=%d",
             planLifecycle_.displayName(), seg.def.startSlot, seg.def.endSlot, executeCount_);
    REQUIRE_TRUE(false, 0,
                 "DSP phase contract violation: compositeReplay requires phase >= SHAPES_FROZEN "
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

  // Per-step deduplication: access the active PlanExecutionContext to avoid
  // repeating plan-level operations (ext input sync, cross-stream ordering,
  // input address hashing) once per segment when we have 14+ segments.
  // compositeReplay runs inside execute(), so the context MUST be set.
  auto* execCtx = static_cast<PlanExecutionContext*>(activeExecutionContext());
  assert(execCtx != nullptr && "compositeReplay called outside execute() — activeExecCtx_ is null");

  // ── Unified pre-replay sync ─────────────────────────────────────────────
  // Cross-stream ordering, H2D variable input sync, and staging buffer D2D
  // are all handled by performPreReplaySync(). MUST be called BEFORE the
  // gap-stream override below — cross-stream sync reads the real default
  // stream via getCudaStream(). With the override active, getCudaStream()
  // returns cudaStr, making the cross-stream sync a no-op.
  NDArray** effectiveExternals = performPreReplaySync(
      externalArrays, numExt, stream, "compositeReplay");

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

  // ── Ext input sync + staging ──────────────────────────────────────────
  // H2D variable input sync and staging D2D are handled by
  // performPreReplaySync() above (before gap stream guard). effectiveExternals
  // already points to staging buffers (if active) or raw externals.
  //
  // Diagnostic-only: fingerprint + KV-stale detection (gated behind DSP_DIAG).
  // FULL level only — syncToHost() + FNV hash of entire variable inputs is O(data_size)
  // and blocks the GPU pipeline. Adds ~10-20ms per decode step.
  if (!planLifecycle_.isSlotBySlot() && !externalInputIsVariable_.empty() &&
      DspDiagnostics::getInstance().isEnabled(DSP_DIAG_EXECUTE) &&
      DspDiagnostics::getInstance().getLevel() >= DSP_LEVEL_FULL) {
    // Fingerprint variable inputs at replay entry.
    fingerprintVariableInputs(externalArrays, numExt, externalInputIsVariable_,
                              externalInputNames_, "replay-entry", seg.exec.executionCount);

    // KV-cache mutation detection: if ALL variable inputs have identical
    // fingerprints to the previous step, KV caches may not be updating.
    std::unordered_map<int, uint64_t> currentFingerprints;
    bool anyVariable = false;
    bool allMatch = true;

    for (int ei = 0; ei < numExt; ei++) {
      if (externalArrays[ei] == nullptr) continue;
      if (ei >= static_cast<int>(externalInputIsVariable_.size()) ||
          !externalInputIsVariable_[ei]) continue;

      anyVariable = true;
      NDArray* arr = externalArrays[ei];
      auto* db = arr->dataBuffer();
      uint64_t h = 0xcbf29ce484222325ULL;  // FNV-1a offset basis
      if (db != nullptr) {
        uintptr_t specialAddr = reinterpret_cast<uintptr_t>(db->special());
        h ^= static_cast<uint64_t>(specialAddr);
        h *= 0x100000001b3ULL;
        h ^= static_cast<uint64_t>(arr->lengthOf());
        h *= 0x100000001b3ULL;
        h ^= static_cast<uint64_t>(db->isSpecialActual() ? 1 : 0);
        h *= 0x100000001b3ULL;
      }
      currentFingerprints[ei] = h;

      auto prev = execCtx->prevVariableFingerprints.find(ei);
      if (prev != execCtx->prevVariableFingerprints.end() && prev->second != h) {
        allMatch = false;
      } else if (prev == execCtx->prevVariableFingerprints.end()) {
        allMatch = false;
      }
    }

    if (anyVariable && allMatch && !execCtx->prevVariableFingerprints.empty()) {
      DSP_DIAG(EXECUTE,
               "KV_CACHE_STALE_WARNING: seg[%d-%d] execCount=%d — ALL variable external "
               "inputs have identical fingerprints to the previous step. "
               "KV caches may not be updating between decode steps. "
               "numVariableInputs=%d",
               seg.def.startSlot, seg.def.endSlot, seg.exec.executionCount,
               static_cast<int>(currentFingerprints.size()));
      for (auto& kv : currentFingerprints) {
        const char* name = (kv.first < static_cast<int>(externalInputNames_.size()))
                           ? externalInputNames_[kv.first].c_str() : "?";
        DSP_DIAG(EXECUTE,
                 "  KV_CACHE_STALE: ext[%d] name='%s' fingerprint=0x%016llx (unchanged)",
                 kv.first, name, static_cast<unsigned long long>(kv.second));
      }
    }

    execCtx->prevVariableFingerprints = std::move(currentFingerprints);
  }

  // Diagnostic: staging buffer metadata only. Value dumps would require host
  // visibility and would block the GPU pipeline.
  if (DSP_DIAG_ENABLED(VERIFY) &&
      DspDiagnostics::getInstance().getLevel() >= DSP_LEVEL_FULL &&
      !cachedVariableExtIndices_.empty() && effectiveExternals_ != nullptr) {
    for (int vi : cachedVariableExtIndices_) {
      if (vi >= numExt) continue;
      NDArray* staging = effectiveExternals_[vi];
      NDArray* original = externalArrays[vi];
      if (staging == nullptr || staging->isEmpty()) continue;
      void* stagingBuf = staging->dataBuffer() != nullptr ? staging->dataBuffer()->special() : nullptr;
      void* origBuf = (original != nullptr && original->dataBuffer() != nullptr)
                          ? original->dataBuffer()->special() : nullptr;
      const char* name = (vi < static_cast<int>(externalInputNames_.size()))
                         ? externalInputNames_[vi].c_str() : "?";
      DSP_DIAG(VERIFY, "STAGING_D2D_CHECK: ext[%d] name='%s' stagingBuf=%p origBuf=%p "
               "bytes=%zu execCount=%d (async path: value dump skipped)",
               vi, name, stagingBuf, origBuf,
               static_cast<size_t>(staging->lengthOf()) * staging->sizeOfT(),
               seg.exec.executionCount);
    }
  }

  // Defensive address-drift checks: validate that external input and slot output
  // addresses haven't changed since the arg table was last synced. If drift is
  // detected with merged CUDA graph handles present, this is a LIFECYCLE ERROR —
  // merged graphs have device addresses baked in and cannot be patched. The graph
  // must be invalidated and re-captured. Drift with only Triton arg-table replay
  // is recoverable via refresh.
  //
  // OPTIMIZATION: Once addresses have been confirmed stable for kAddrStableSkipThreshold
  // consecutive steps, skip the expensive O(N) hash computation (N = total slot inputs
  // in segment, typically 2500+). Only recheck every kAddrRecheckInterval steps as a
  // safety net. This is safe because:
  //   1. Staging buffers provide pointer stability after capture by design
  //   2. Shape freeze prevents reallocation
  //   3. Legitimate address changes go through invalidation paths that reset counters
  //   4. Merged CUDA graphs would crash on drift anyway (baked-in addresses)
  //   4. Merged CUDA graphs would crash on drift anyway (baked-in addresses)
  //
  // Staging buffers (ensureAndSyncStagingBuffers) isolate merged CUDA graphs
  // from external address drift: graphs were captured against staging buffer
  // addresses which are stable for plan lifetime. The skip optimization is
  // therefore safe even with merged graphs.
  static constexpr int kAddrStableSkipThreshold = 3;
  static constexpr int kAddrRecheckInterval = 64;
  bool driftDetected = false;
  if (!seg.exec.needsArgRefresh() && seg.exec.capturedInputAddrKey != 0) {
    // Skip the O(N) hash when addresses have been stable for multiple consecutive steps.
    // Periodic recheck (every kAddrRecheckInterval steps) catches pathological cases.
    bool skipAddrCheck = (seg.exec.addrKeyStableCount >= kAddrStableSkipThreshold) &&
                         ((seg.exec.executionCount % kAddrRecheckInterval) != 0);
    if (skipAddrCheck) {
      seg.exec.addrKeyStableCount++;
    } else {
    LongType currentAddrKey = computeSegmentInputAddrKey(seg, effectiveExternals, numExt);
    if (currentAddrKey != seg.exec.capturedInputAddrKey) {
      driftDetected = true;
      seg.exec.bumpArgGeneration();
      seg.exec.addrKeyStableCount = 0;
      seg.exec.slotAddrStableCount = 0;

      // Trace which specific external inputs drifted
      DSP_DIAG(EXECUTE,
               "EXT_INPUT_REBIND_DETECTED: seg[%d-%d] addrKey current=%lld captured=%lld execCount=%d",
               seg.def.startSlot, seg.def.endSlot,
               (long long)currentAddrKey, (long long)seg.exec.capturedInputAddrKey,
               seg.exec.executionCount);
      // Detailed per-ext-input trace (first 20 unique ext inputs only to avoid log flood)
      {
        int traceCount = 0;
        std::unordered_set<int> seen;
        for (int i = seg.def.startSlot; i <= seg.def.endSlot && traceCount < 20; i++) {
          if (i >= numSlots_) break;
          auto& slot = slots_[i];
          // External inputs are encoded as negative indices in inputSourceIndices: -(idx+1)
          for (int wi = 0; wi < slot.wiring.numInputs && traceCount < 20; wi++) {
            int srcIdx = slot.wiring.inputSourceIndices[wi];
            if (srcIdx < 0) {
              int extIdx = -(srcIdx + 1);
              if (extIdx >= 0 && extIdx < numExt && seen.find(extIdx) == seen.end()) {
                seen.insert(extIdx);
                void* buf = effectiveExternals[extIdx] ? effectiveExternals[extIdx]->specialBuffer() : nullptr;
                DSP_DIAG(EXECUTE,
                         "  DRIFT_TRACE: extIdx=%d name='%s' currentBuf=%p",
                         extIdx,
                         (extIdx < static_cast<int>(externalInputNames_.size())
                          ? externalInputNames_[extIdx].c_str() : "?"),
                         buf);
                traceCount++;
              }
            }
          }
        }
      }
    } else {
      seg.exec.addrKeyStableCount++;
    }
    } // end !skipAddrCheck
  }

  if (!driftDetected && !seg.exec.needsArgRefresh() && seg.exec.capturedSlotAddrHash != 0) {
    // Same skip optimization for slot address hash (O(slotRange) per step).
    // Staging buffers isolate merged graphs from external address drift, so
    // the skip threshold works correctly regardless of merged graph presence.
    bool skipSlotCheck = (seg.exec.slotAddrStableCount >= kAddrStableSkipThreshold) &&
                         ((seg.exec.executionCount % kAddrRecheckInterval) != 0);
    if (skipSlotCheck) {
      seg.exec.slotAddrStableCount++;
    } else {
    LongType currentSlotHash = computeSlotAddrHash(
        outputSlots_, seg.def.startSlot, seg.def.endSlot, totalOutputSlots_);
    if (currentSlotHash != seg.exec.capturedSlotAddrHash) {
      driftDetected = true;
      seg.exec.bumpArgGeneration();
      seg.exec.addrKeyStableCount = 0;
      seg.exec.slotAddrStableCount = 0;

      DSP_DIAG(EXECUTE,
               "SLOT_ADDR_DRIFT_DETECTED: seg[%d-%d] hash current=0x%llx captured=0x%llx execCount=%d",
               seg.def.startSlot, seg.def.endSlot,
               (long long)currentSlotHash, (long long)seg.exec.capturedSlotAddrHash,
               seg.exec.executionCount);
      // Trace first 20 non-null slots to identify the drifting output
      {
        int traceCount = 0;
        for (int s = seg.def.startSlot; s <= seg.def.endSlot && s < totalOutputSlots_ && traceCount < 20; s++) {
          if (outputSlots_[s] != nullptr) {
            void* buf = outputSlots_[s]->specialBuffer();
            DSP_DIAG(EXECUTE, "  SLOT_DRIFT_TRACE: slot=%d buf=%p len=%lld",
                     s, buf, (long long)outputSlots_[s]->lengthOf());
            traceCount++;
          }
        }
      }
    } else {
      seg.exec.slotAddrStableCount++;
    }
    } // end !skipSlotCheck
  }

  // LIFECYCLE ERROR: address drift with merged CUDA graph handles.
  // Merged graphs have device pointers baked into captured kernel nodes — they
  // cannot be updated via arg table refresh. Launching a merged graph with stale
  // addresses causes SIGSEGV in cudaGraphLaunch.
  //
  // This is NOT a recoverable soft-error. It indicates a bug upstream: something
  // reallocated a staging buffer or output slot between steps without going through
  // the proper invalidation path. Invalidate the graph and propagate the lifecycle
  // error so the caller can re-capture.
  if (driftDetected && !sched.mergedReplayHandles.empty()) {
    DSP_DIAG(EXECUTE,
             "MERGED_GRAPH_LIFECYCLE_ERROR: seg[%d-%d] address drift detected with %d "
             "merged CUDA graph handles. Merged graphs have baked-in device pointers "
             "that are now stale — launching would SIGSEGV. Invalidating. execCount=%d",
             seg.def.startSlot, seg.def.endSlot,
             static_cast<int>(sched.mergedReplayHandles.size()),
             seg.exec.executionCount);
    SegmentLifecycle::invalidateForRebuild(this, seg, "merged_graph_addr_drift");
    // Return KERNEL_FAILURE — caller (segDispatchReplay) handles re-capture fallthrough
    return Status::KERNEL_FAILURE;
  }

  // Refresh arg tables + D2D copy (skip when generation matches — fast replay path)
  // The generation counter is bumped by any address change; if capturedArgGeneration
  // matches argTableGeneration, no refresh is needed — correct by construction.
  bool useFastReplay = !seg.exec.needsArgRefresh() &&
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

     // Mark arg table as current — generation now matches, enabling fast-replay
     // on the next step without needing to recompute address hashes.
     seg.exec.markArgsCurrent();
     // Legacy keys kept for diagnostic VERIFY checks
     seg.exec.capturedInputAddrKey = computeSegmentInputAddrKey(seg, effectiveExternals, numExt);
     seg.exec.capturedSlotAddrHash = computeSlotAddrHash(
         outputSlots_, seg.def.startSlot, seg.def.endSlot, totalOutputSlots_);
   }
 }
#endif
  cudaGetLastError();  // Clear sticky errors

  // Composite replay mixes two zero-before-write mechanisms:
  //   1. Triton island outputs are zeroed by captured nullify/memset nodes inside
  //      the island graphs themselves.
  //   2. Live gap slots still need an explicit prezero pass because they execute
  //      outside those captured graphs on replay.
  //
  // The generic segment-wide prezero path zeros all qualifying outputs in the
  // segment, including Triton island outputs that the captured graphs already
  // handle. That duplicates zeroing work every step. For composite replay, rebuild
  // the current gap-slot target list from the schedule and zero only those outputs.
  {
    // Cache prezero targets after first compute — gap slot set and slot flags are
    // static once shapes are frozen. Only device pointers need refreshing, and those
    // are stable in replay mode (no reallocation after freeze).
    if (!gapPrezeroTargetsCached_) {
      std::unordered_set<int> gapSlots;
      for (const auto& unit : sched.units) {
        if (unit.kind != REPLAY_UNIT_GAP) continue;
        for (int s = unit.startSlot; s <= unit.endSlot; s++) {
          gapSlots.insert(s);
        }
      }

      collectBatchZeroTargets(gapSlots);
      cachedGapPrezeroCount_ = static_cast<int>(batchZeroEntries_.size());
      if (!planLifecycle_.isSlotBySlot()) {
        gapPrezeroTargetsCached_ = true;
        DSP_DIAG(MEMORY, "compositeReplay: cached %d prezero targets (will skip recompute on subsequent steps)",
                 cachedGapPrezeroCount_);
      }
    } else {
      // Fast path: refresh device pointers from cached output slot indices.
      // N19: Skip refresh entirely when pointers are stable — outputSlots_ addresses
      // don't change after pointer stabilization, so cached ptrs are still valid.
      if (!planLifecycle_.pointersStable()) {
        for (int i = 0; i < cachedGapPrezeroCount_; i++) {
          int outIdx = batchZeroEntries_[i].outputSlotIndex;
          NDArray* cached = outputSlots_[outIdx];
          if (cached != nullptr) {
            batchZeroEntries_[i].ptr = cached->specialBuffer();
            batchZeroEntries_[i].bytes = static_cast<int>(cached->dataBuffer()->getLenInBytes());
          }
        }
      }
    }

    int gapZeroCount = cachedGapPrezeroCount_;
    if (gapZeroCount > 0) {
      DSP_DIAG_SEG(MEMORY, seg.def.startSlot,
                   "compositeReplay prezero seg=[%d-%d] stream=%p execCount=%d zeroTargets=%d cached=%d",
                   seg.def.startSlot, seg.def.endSlot, (void*)cudaStr, seg.exec.executionCount,
                   gapZeroCount, (int)gapPrezeroTargetsCached_);

      if (gapZeroCount == 1) {
        auto& entry = batchZeroEntries_[0];
        if (entry.ptr != nullptr && entry.bytes > 0) {
          cudaMemsetAsync(entry.ptr, 0, entry.bytes, cudaStr);
        }
      } else {
        // Stack arrays avoid heap alloc/free per decode step (up to 128 entries).
        constexpr int kStackCap = 128;
        void* stackPtrs[kStackCap];
        size_t stackSizes[kStackCap];
        void** dstPtrs = (gapZeroCount <= kStackCap) ? stackPtrs : new void*[gapZeroCount];
        size_t* sizes = (gapZeroCount <= kStackCap) ? stackSizes : new size_t[gapZeroCount];
        for (int i = 0; i < gapZeroCount; i++) {
          dstPtrs[i] = batchZeroEntries_[i].ptr;
          sizes[i] = static_cast<size_t>(batchZeroEntries_[i].bytes);
        }
        launchBatchMemset(cudaStr, dstPtrs, sizes, gapZeroCount);
        if (gapZeroCount > kStackCap) { delete[] dstPtrs; delete[] sizes; }
        DSP_DIAG(MEMORY, "compositeReplay gap-only prezero: batched %d buffers in 1 kernel launch",
                 gapZeroCount);
      }
    } else {
      DSP_DIAG_SEG(MEMORY, seg.def.startSlot,
                   "compositeReplay gap-only prezero skipped seg=[%d-%d] zeroTargets=0",
                   seg.def.startSlot, seg.def.endSlot);
    }
  }

  auto tPrezero = executionTimingEnabled_ ? Clock::now() : Clock::time_point{};

  // NOTE: Single-stream unification was tried here (routing gap ops onto cudaStr
  // via tl_dspGapStream). Tested twice:
  //   1. Pre-bgemm (42.6→35.9 tok/s, -16%) — thought to eliminate beneficial overlap
  //   2. Post-bgemm (51.1→50.5 tok/s, neutral) — confirmed NO overlap exists
  //      (gap+merged ≈ total), but slight per-dispatch latency increase from
  //      sharing a single stream's command buffer. Not worth the change.

  // ── Cross-stream sync: ensure prezero memsets on cudaStr are visible ───
  // Gap-only prezero above issues cudaMemsetAsync / batchMemsetKernel on cudaStr. Gap ops that
  // follow run on the LaunchContext's default stream. Without sync, gap ops
  // on a different stream might start before prezero completes, seeing
  // stale (non-zero) data in output buffers instead of zeros — causing
  // accumulation errors for ops that read-modify-write their output.
  // Event-based: gap stream waits for prezero completion without CPU block.
  // Guard: skipped when gapStream==cudaStr (same stream, no sync needed).
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
  DSP_DIAG(EXECUTE, "COMPOSITE_REPLAY: seg[%d-%d] %d units, %d merged groups, execCount=%d",
           seg.def.startSlot, seg.def.endSlot,
           static_cast<int>(sched.units.size()), mergedGroupCount,
           seg.exec.executionCount);

  // ── Per-category timing accumulators (active when executionTimingEnabled_) ──
  long long tMergedLaunchUs = 0, tMergedDirtyUs = 0, tGapExecUs = 0;
  long long tIslandLaunchUs = 0, tIslandDirtyUs = 0, tArgRefreshUs = 0;
  int nMergedLaunches = 0, nGapUnits = 0, nIslandLaunches = 0, nArgRefreshes = 0;
  int nExecSlots = 0, nTickSlots = 0, nBatchedGemmSlots = 0;
  // Per-op-name EXECUTE slot counters for gap profiling (only first 250 decode steps)
  std::unordered_map<std::string, int> execOpCounts;
  bool collectExecOpNames = executionTimingEnabled_ && seg.exec.executionCount < 3;

  // ── N6: Hoist cuBLAS stream+workspace setup before gap loop ────────────────
  // All 60 bgemm groups use the same CUDA stream (cudaStr).  Per cuBLAS docs,
  // cublasSetStream resets the user workspace, so both calls are a coupled pair.
  // Calling once here and skipping 59 redundant pairs inside executeBatchedGemmGroup
  // eliminates 118 cuBLAS host-API calls per decode step.
  // RAII guard resets tl_cublasGapStreamReady on any exit path.
  struct CublasGapStreamGuard {
    CublasGapStreamGuard() { tl_cublasGapStreamReady = false; }
    ~CublasGapStreamGuard() { tl_cublasGapStreamReady = false; }
  } cublasGapStreamGuard;
  if (!batchedGemmGroups_.empty() && !tl_graphExecutionActive) {
    auto* ctx2 = LaunchContext::defaultContext();
    std::lock_guard<std::mutex> lock(*LaunchContext::deviceMutex());
    auto handle = reinterpret_cast<cublasHandle_t*>(ctx2->getCublasHandle());
    cublasSetStream_v2(*handle, cudaStr);
    if (tl_cublasWorkspacePtr != nullptr && tl_cublasWorkspaceSize > 0) {
      cublasSetWorkspace(*handle, tl_cublasWorkspacePtr, tl_cublasWorkspaceSize);
    }
    tl_cublasGapStreamReady = true;
  }

  // ── Tensor-core acceleration for steady-state gap matmuls (hoisted) ────────
  bool gapSlotsExecutedSinceArgCopy = false;
  for (auto& unit : sched.units) {
    // ── Merged group: non-leader units skip entirely ──
    // The leader does dirty-mark + tickWriteDevice for ALL slots in the group
    // using pre-computed mergedGroupSlotRanges, so non-leaders are pure no-ops.
    if (unit.mergedGroupId >= 0 && !unit.isMergedLeader) {
      continue;
    }

    // ── Merged group leader: launch the merged graph ──
    if (unit.mergedGroupId >= 0 && unit.isMergedLeader) {
      // Re-copy arg table if gap slots ran since last H2D copy (same fix as island path)
      if (gapSlotsExecutedSinceArgCopy) {
        auto tAR0 = executionTimingEnabled_ ? Clock::now() : Clock::time_point();
#if HAVE_TRITON
        auto* tritonBackend2 = dynamic_cast<TritonGraphBackend*>(getGpuGraphBackend());
        if (tritonBackend2 != nullptr) {
          tritonBackend2->refreshArgTablesForReplay(
              seg, effectiveExternals, numExt, outputSlots_, totalOutputSlots_, stream);
          tritonBackend2->copyConsolidatedArgTableToDevice(seg, stream);
        }
#endif
        if (executionTimingEnabled_) {
          tArgRefreshUs += std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - tAR0).count();
          nArgRefreshes++;
        }
        gapSlotsExecutedSinceArgCopy = false;
      }

      int mgId = unit.mergedGroupId;
      if (mgId < 0 || mgId >= static_cast<int>(sched.mergedReplayHandles.size()) ||
          !sched.mergedReplayHandles[mgId] || !sched.mergedReplayHandles[mgId]->isReady()) {
        DSP_DIAG(EXECUTE, "MERGED_REPLAY: group %d handle not ready — executing slots [%d-%d] slot-by-slot",
                 mgId, unit.startSlot, unit.endSlot);
        // Fall back to slot-by-slot for this group's slots instead of failing.
        // This handles the case where capture hasn't completed yet (e.g., after
        // shape-drift recompilation or first frozen warmup).
        for (int s = unit.startSlot; s <= unit.endSlot; s++) {
          if (s < 0 || s >= numSlots_) continue;
          auto slotStatus = executeSlot(s, effectiveExternals, numExt, stream);
          if (slotStatus != Status::OK) {
            DSP_DIAG(EXECUTE, "MERGED_REPLAY: fallback slot %d FAILED status=%d",
                     s, static_cast<int>(slotStatus));
            return slotStatus;
          }
        }
        gapSlotsExecutedSinceArgCopy = true;
        continue;
      }

      DSP_DIAG(EXECUTE, "MERGED_REPLAY: group %d leader [%d-%d] launching",
               mgId, unit.startSlot, unit.endSlot);
      // Diagnostic: staging buffer metadata before first merged group launch.
      // Keep this async; no host-visible value dump here.
      if (mgId == 0 && DSP_DIAG_ENABLED(VERIFY) &&
          DspDiagnostics::getInstance().getLevel() >= DSP_LEVEL_FULL &&
          effectiveExternals_ != nullptr && !cachedVariableExtIndices_.empty()) {
        for (int vi2 : cachedVariableExtIndices_) {
          if (vi2 >= numExt) continue;
          NDArray* staging = effectiveExternals_[vi2];
          if (staging == nullptr || staging->isEmpty()) continue;
          void* buf = staging->specialBuffer();
          const char* nm = (vi2 < static_cast<int>(externalInputNames_.size()))
                           ? externalInputNames_[vi2].c_str() : "?";
          DSP_DIAG(VERIFY, "PRE_MERGED_LAUNCH_STAGING: ext[%d] name='%s' buf=%p "
                   "bytes=%zu execCount=%d (async path: value dump skipped)",
                   vi2, nm, buf,
                   static_cast<size_t>(staging->lengthOf()) * staging->sizeOfT(),
                   seg.exec.executionCount);
        }
      }
      auto tML0 = executionTimingEnabled_ ? Clock::now() : Clock::time_point();
      bool launchOk = sched.mergedReplayHandles[mgId]->replay(stream);
      if (executionTimingEnabled_) {
        tMergedLaunchUs += std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - tML0).count();
        nMergedLaunches++;
      }
      if (!launchOk) {
        DSP_DIAG(EXECUTE, "MERGED_REPLAY: group %d launch FAILED", mgId);
        return Status::KERNEL_FAILURE;
      }

      // ── Post-merged-replay island output diagnostic ──────────────────────
      // Metadata only; value dumps would require a blocking host read.
      if (DSP_DIAG_ENABLED(EXECUTE) &&
          DspDiagnostics::getInstance().getLevel() >= DSP_LEVEL_FULL) {
        int postRangeMin, postRangeMax;
        if (mgId < static_cast<int>(sched.mergedGroupSlotRanges.size())) {
          postRangeMin = sched.mergedGroupSlotRanges[mgId].minSlot;
          postRangeMax = sched.mergedGroupSlotRanges[mgId].maxSlot;
        } else {
          postRangeMin = unit.startSlot;
          postRangeMax = unit.endSlot;
        }
        for (int diagSlot = postRangeMin; diagSlot <= postRangeMax; diagSlot++) {
          if (diagSlot < 0 || diagSlot >= numSlots_) continue;
          const NativeSlot& dSlot = slots_[diagSlot];
          for (int dO = 0; dO < dSlot.wiring.numOutputs; dO++) {
            int outSi = dSlot.wiring.outputSlotIndices[dO];
            if (outSi < 0 || outSi >= totalOutputSlots_) continue;
            NDArray* outArr = outputSlots_[outSi];
            if (outArr == nullptr || outArr->lengthOf() == 0) continue;
            void* outBuf = outArr->specialBuffer();
            if (outBuf == nullptr) continue;
            DSP_DIAG(EXECUTE,
                     "POST_MERGED_ISLAND_OUTPUT: mergedGroup=%d slot=%d outSlot=%d "
                     "op='%s' buf=%p len=%lld execCount=%d "
                     "(async path: value dump skipped)",
                     mgId, diagSlot, outSi, dSlot.ident.opName.c_str(),
                     outBuf, (long long)outArr->lengthOf(),
                     seg.exec.executionCount);
          }
        }
        // Also dump the effective external input values to verify D2D staging
        if (effectiveExternals_ != nullptr) {
          for (int ei = 0; ei < numExt; ei++) {
            NDArray* extArr = effectiveExternals_[ei];
            if (extArr == nullptr || extArr->lengthOf() == 0) continue;
            void* extBuf = extArr->specialBuffer();
            if (extBuf == nullptr) continue;
            const char* eName = (ei < static_cast<int>(externalInputNames_.size()))
                                ? externalInputNames_[ei].c_str() : "?";
            DSP_DIAG(EXECUTE,
                     "POST_MERGED_EXT_INPUT: ext[%d] name='%s' buf=%p dtype=%d "
                     "len=%lld execCount=%d (async path: value dump skipped)",
                     ei, eName, extBuf, static_cast<int>(extArr->dataType()),
                     (long long)extArr->lengthOf(), seg.exec.executionCount);
          }
        }
      }

      // Combined dirty-mark + tickWriteDevice using pre-computed group range.
      // Iterate through each slot's wiring.outputSlotIndices to correctly map
      // step indices to output slot indices (they differ for multi-output ops).
      auto tMD0 = executionTimingEnabled_ ? Clock::now() : Clock::time_point();
      int rangeMin, rangeMax;
      if (mgId < static_cast<int>(sched.mergedGroupSlotRanges.size())) {
        rangeMin = sched.mergedGroupSlotRanges[mgId].minSlot;
        rangeMax = sched.mergedGroupSlotRanges[mgId].maxSlot;
      } else {
        rangeMin = unit.startSlot;
        rangeMax = unit.endSlot;
      }
      for (int stepIdx = rangeMin; stepIdx <= rangeMax; stepIdx++) {
        if (stepIdx < 0 || stepIdx >= numSlots_) continue;
        const NativeSlot& slot = slots_[stepIdx];
        for (int o = 0; o < slot.wiring.numOutputs; o++) {
          int outIdx = slot.wiring.outputSlotIndices[o];
          if (outIdx < 0 || outIdx >= totalOutputSlots_) continue;
          dirtySlotGenerations_[outIdx] = currentDirtyGeneration_;
          NDArray* arr = outputSlots_[outIdx];
          if (arr != nullptr && arr->dataBuffer() != nullptr && !arr->dataBuffer()->isClosed()) {
            arr->tickWriteDevice();
          }
        }
      }
      if (executionTimingEnabled_) {
        tMergedDirtyUs += std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - tMD0).count();
      }
      continue;
    }

    // ── Unmerged units: original per-unit replay ──
    if (unit.kind == REPLAY_UNIT_GAP) {
      auto tGE0 = executionTimingEnabled_ ? Clock::now() : Clock::time_point();
      DSP_DIAG(EXECUTE, "COMPOSITE_REPLAY: gap [%d-%d]", unit.startSlot, unit.endSlot);

      // ── Pre-gap input diagnostic: metadata only, no host-blocking value dump.
      if (DSP_DIAG_ENABLED(EXECUTE) &&
          DspDiagnostics::getInstance().getLevel() >= DSP_LEVEL_FULL) {
        for (int gs = unit.startSlot; gs <= unit.endSlot && gs < numSlots_; gs++) {
          const NativeSlot& gSlot = slots_[gs];
          for (int gi = 0; gi < gSlot.wiring.numInputs; gi++) {
            int srcIdx = gSlot.wiring.inputSourceIndices[gi];
            NDArray* srcArr = nullptr;
            if (srcIdx < 0) {
              int extIdx = -(srcIdx + 1);
              if (extIdx < numExt) srcArr = effectiveExternals[extIdx];
            } else if (srcIdx < totalOutputSlots_) {
              srcArr = outputSlots_[srcIdx];
            }
            if (srcArr == nullptr || srcArr->lengthOf() == 0) continue;
            void* srcBuf = srcArr->specialBuffer();
            if (srcBuf == nullptr) continue;
            DSP_DIAG(EXECUTE,
                     "PRE_GAP_INPUT: gapSlot=%d input[%d] srcIdx=%d op='%s' "
                     "buf=%p len=%lld execCount=%d (async path: value dump skipped)",
                     gs, gi, srcIdx, gSlot.ident.opName.c_str(),
                     srcBuf, (long long)srcArr->lengthOf(),
                     seg.exec.executionCount);
          }
        }
      }
      sd::graph::DspReplayGuard replayGuard(true);
      bool gapOutputPointersChanged = false;
      bool skipPtrTracking = !seg.exec.needsArgRefresh() && planLifecycle_.pointersStable();

      // ── cuBLAS Lt for gap matmul ops (logits projection) ─────────────
      // Gap ops execute outside CUDA graph capture, so split-K non-determinism
      // from cublasLt doesn't affect graph replay. Temporarily enable cublasLt
      // for the gap execution window to unlock the fast path for large-N vocab
      // projections [1,K]×[K,N] where N≥16384 (e.g. SmolDocling logits [1,576]×[576,49280]).
      bool gapLtEnabled = Environment::getInstance().dsp().cublasLtGapEnabled() && tl_cublasLtDisabled;
      if (gapLtEnabled) {
        tl_cublasLtDisabled = false;
      }

      // ── Active gap slot cache: skip 97% of slot iterations in steady state ──
      // On the first frozen+steady pass, classify every slot and cache only those
      // that need work. On subsequent steps, iterate over the compact cached list.
      // NOTE: cache is per-gap-unit keyed by startSlot.
      bool useCachedActiveSlots = activeGapSlotsCachedSet_.count(unit.startSlot)
                                  && !planLifecycle_.isSlotBySlot() && executeCount_ >= 3;

      if (useCachedActiveSlots) {
        // ── FAST PATH: iterate only over pre-classified active slots ──
        auto& cachedSlots = cachedActiveGapSlotsMap_[unit.startSlot];
        for (auto& active : cachedSlots) {
          switch (active.action) {
            case ActiveSlotAction::IDENTITY_TICK: {
              int si = active.outputSlotIdx;
              if (si >= 0 && si < totalOutputSlots_ && outputSlots_[si] != nullptr) {
                outputSlots_[si]->tickWriteDevice();
              }
              if (executionTimingEnabled_) nTickSlots++;
              break;
            }
            case ActiveSlotAction::VIEW_TICK: {
              int si = active.outputSlotIdx;
              if (si >= 0 && si < totalOutputSlots_ && outputSlots_[si] != nullptr) {
                outputSlots_[si]->tickWriteDevice();
                dirtySlotGenerations_[si] = currentDirtyGeneration_;
              }
              if (executionTimingEnabled_) nTickSlots++;
              break;
            }
            case ActiveSlotAction::BATCHED_GEMM: {
              cudaStream_t execStream = stream ? *static_cast<cudaStream_t*>(stream) : static_cast<cudaStream_t>(nullptr);
              Status batchStatus = executeBatchedGemmGroup(active.batchedGemmGroupIdx, effectiveExternals, numExt, execStream);
              if (batchStatus != Status::OK) {
                return batchStatus;  // replayGuard restores tl_dspReplayActive
              }
              if (executionTimingEnabled_) nBatchedGemmSlots++;
              break;
            }
            case ActiveSlotAction::EXECUTE: {
              // ── N12: Reclassify EXECUTE → VIEW_TICK when view is now established ──
              // On cache build (executeCount_==2), reshape_no_copy may not yet share
              // dataBuffer with input, so it gets classified as EXECUTE. Once the view
              // is established in later steps, demote to VIEW_TICK to skip full op dispatch.
              if (executeCount_ >= 4 && skipPtrTracking) {
                auto& slot = slots_[active.slotIdx];
                if (slot.isViewCapableOp() && slot.wiring.numInputs >= 1 && slot.wiring.numOutputs >= 1) {
                  int outSi = slot.wiring.outputSlotIndices[0];
                  if (outSi >= 0 && outSi < totalOutputSlots_) {
                    NDArray* currentOut = outputSlots_[outSi];
                    int inSrc = slot.wiring.inputSourceIndices[0];
                    NDArray* input0 = nullptr;
                    if (inSrc >= 0 && inSrc < totalOutputSlots_) {
                      input0 = outputSlots_[inSrc];
                    } else if (inSrc < 0) {
                      int extIdx = -(inSrc + 1);
                      if (extIdx >= 0 && extIdx < numExt) input0 = effectiveExternals[extIdx];
                    }
                    if (currentOut != nullptr && input0 != nullptr &&
                        currentOut->dataBuffer() != nullptr &&
                        currentOut->dataBuffer() == input0->dataBuffer()) {
                      // View is established — demote to VIEW_TICK permanently
                      currentOut->tickWriteDevice();
                      dirtySlotGenerations_[outSi] = currentDirtyGeneration_;
                      active.action = ActiveSlotAction::VIEW_TICK;
                      active.outputSlotIdx = outSi;
                      if (executionTimingEnabled_) nTickSlots++;
                      break;
                    }
                  }
                }
              }
              // Use ultra-fast path in steady state (pointers stable, well past warmup)
              Status slotStatus;
              if (executeCount_ >= 5 && skipPtrTracking) {
                slotStatus = executeSlotGapFast(active.slotIdx, effectiveExternals, numExt);
              } else {
                slotStatus = executeSlot(active.slotIdx, effectiveExternals, numExt, stream);
              }
              if (slotStatus != Status::OK) {
                DSP_DIAG(EXECUTE, "COMPOSITE_REPLAY: gap slot %d FAILED status=%d",
                         active.slotIdx, static_cast<int>(slotStatus));
                return slotStatus;  // replayGuard restores tl_dspReplayActive
              }
              if (executionTimingEnabled_) nExecSlots++;
              if (collectExecOpNames && active.slotIdx >= 0 && active.slotIdx < numSlots_) {
                execOpCounts[slots_[active.slotIdx].ident.opName]++;
              }
              break;
            }
            case ActiveSlotAction::SKIP:
              break;
          }
        }
      } else {
        // ── CLASSIFICATION PATH: run full logic, optionally build cache ──
        bool buildingCache = !planLifecycle_.isSlotBySlot() && executeCount_ >= 2
                             && !activeGapSlotsCachedSet_.count(unit.startSlot);
        if (buildingCache) {
          cachedActiveGapSlotsMap_[unit.startSlot].clear();
          cachedActiveGapSlotsMap_[unit.startSlot].reserve(128);  // ~82 expected
        }

      for (int s = unit.startSlot; s <= unit.endSlot; s++) {
        // ── Frozen constant / identity / fused-tail early skip ────────────
        if (!planLifecycle_.isSlotBySlot() && executeCount_ >= 2) {
          auto& gapSlot = slots_[s];
          if (gapSlot.frozenConstantSlot()) {
            // No cache entry needed — frozen constants never do anything
            continue;
          }
          if (gapSlot.isIdentityOp()) {
            if (gapSlot.wiring.numOutputs >= 1) {
              int si = gapSlot.wiring.outputSlotIndices[0];
              if (si >= 0 && si < totalOutputSlots_ && outputSlots_[si] != nullptr) {
                outputSlots_[si]->tickWriteDevice();
              }
              if (buildingCache) {
                cachedActiveGapSlotsMap_[unit.startSlot].push_back({s, ActiveSlotAction::IDENTITY_TICK, -1, si});
              }
            }
            continue;
          }
          if (gapSlot.fusedChain.isFusedChainTail) continue;
        }

        // ── Batched GEMM dispatch in gap loop ──────────────────────────
        if (!batchedGemmGroups_.empty() && s < (int)slotToBatchedGemmGroup_.size()) {
          int bgIdx = slotToBatchedGemmGroup_[s];
          if (bgIdx >= 0 && bgIdx < (int)batchedGemmGroups_.size()) {
            auto& bgGroup = batchedGemmGroups_[bgIdx];
            if (s == bgGroup.triggerSlot) {
              cudaStream_t execStream = stream ? *static_cast<cudaStream_t*>(stream) : static_cast<cudaStream_t>(nullptr);
              Status batchStatus = executeBatchedGemmGroup(bgIdx, effectiveExternals, numExt, execStream);
              if (batchStatus != Status::OK) {
                DSP_DIAG(EXECUTE, "COMPOSITE_REPLAY: batched GEMM group %d FAILED at slot %d status=%d",
                         bgIdx, s, static_cast<int>(batchStatus));
                return batchStatus;  // replayGuard restores tl_dspReplayActive
              }
              if (!skipPtrTracking && !gapOutputPointersChanged) {
                void* outputBufsBefore[NativeDynamicShapePlan::MAX_OUTPUTS_PER_SLOT];
                snapshotSlotOutputBuffers(slots_[s], outputSlots_, totalOutputSlots_,
                                          outputBufsBefore, NativeDynamicShapePlan::MAX_OUTPUTS_PER_SLOT);
                if (slotOutputBuffersChanged(slots_[s], outputSlots_, totalOutputSlots_,
                                             outputBufsBefore, NativeDynamicShapePlan::MAX_OUTPUTS_PER_SLOT)) {
                  gapOutputPointersChanged = true;
                }
              }
              if (buildingCache) {
                cachedActiveGapSlotsMap_[unit.startSlot].push_back({s, ActiveSlotAction::BATCHED_GEMM, bgIdx, -1});
              }
              continue;
            } else {
              // Non-trigger slot in a batched group — skip
              continue;
            }
          }
        }

        // ── View-op fast path ──
        if (!dsp_disable_view_fastpath() &&
            s < totalOutputSlots_ &&
            slots_[s].isViewCapableOp() &&
            slots_[s].slotPhase.isSealed() &&
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
              currentOut->tickWriteDevice();
              dirtySlotGenerations_[outSi] = currentDirtyGeneration_;
              if (buildingCache) {
                cachedActiveGapSlotsMap_[unit.startSlot].push_back({s, ActiveSlotAction::VIEW_TICK, -1, outSi});
              }
              continue;
            }
          }
        }

        // ── Full executeSlot path ──
        void* outputBufsBefore[NativeDynamicShapePlan::MAX_OUTPUTS_PER_SLOT];
        if (!skipPtrTracking && !gapOutputPointersChanged) {
          snapshotSlotOutputBuffers(slots_[s], outputSlots_, totalOutputSlots_,
                                    outputBufsBefore, NativeDynamicShapePlan::MAX_OUTPUTS_PER_SLOT);
        }

        auto slotStatus = executeSlot(s, effectiveExternals, numExt, stream);
        if (slotStatus != Status::OK) {
          DSP_DIAG(EXECUTE, "COMPOSITE_REPLAY: gap slot %d FAILED status=%d",
                   s, static_cast<int>(slotStatus));
          return slotStatus;  // replayGuard restores tl_dspReplayActive
        }
        if (!skipPtrTracking && !gapOutputPointersChanged &&
            slotOutputBuffersChanged(slots_[s], outputSlots_, totalOutputSlots_,
                                     outputBufsBefore, NativeDynamicShapePlan::MAX_OUTPUTS_PER_SLOT)) {
          gapOutputPointersChanged = true;
        }
        if (buildingCache) {
          cachedActiveGapSlotsMap_[unit.startSlot].push_back({s, ActiveSlotAction::EXECUTE, -1, -1});
        }
        if (collectExecOpNames) {
          execOpCounts[slots_[s].ident.opName]++;
        }
      }

        if (buildingCache) {
          activeGapSlotsCachedSet_.insert(unit.startSlot);
          DSP_DIAG(EXECUTE, "COMPOSITE_REPLAY: gap[%d-%d] cached %d active gap slots out of %d total (%.1f%% skip rate)",
                   unit.startSlot, unit.endSlot,
                   static_cast<int>(cachedActiveGapSlotsMap_[unit.startSlot].size()),
                   unit.endSlot - unit.startSlot + 1,
                   100.0 * (1.0 - static_cast<double>(cachedActiveGapSlotsMap_[unit.startSlot].size()) /
                            std::max(1, unit.endSlot - unit.startSlot + 1)));
        }
      }  // end classification vs cached path

      // replayGuard destructor restores tl_dspReplayActive at scope end
      gapSlotsExecutedSinceArgCopy = gapOutputPointersChanged;
      DSP_DIAG(EXECUTE,
               "COMPOSITE_REPLAY: gap [%d-%d] outputPtrChanged=%d%s",
               unit.startSlot, unit.endSlot,
               gapOutputPointersChanged ? 1 : 0,
               gapOutputPointersChanged ? " -> refresh arg table before next Triton replay"
                                        : " -> skip arg table refresh");

      // ── Restore cuBLAS Lt disable for subsequent island replays ──
      if (gapLtEnabled) {
        tl_cublasLtDisabled = true;
      }

      // Cross-stream sync after gap ops
      {
        auto* lcStream = LaunchContext::defaultContext()->getCudaStream();
        cudaStream_t gapStream = lcStream ? *lcStream : nullptr;
        if (gapStream != nullptr && gapStream != cudaStr) {
          cudaEvent_t evt = execCtx->crossStreamEvent;
          cudaEventRecord(evt, gapStream);
          cudaStreamWaitEvent(cudaStr, evt, 0);
        }
      }
      if (executionTimingEnabled_) {
        tGapExecUs += std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - tGE0).count();
        nGapUnits++;
      }

    } else {  // REPLAY_UNIT_TRITON_ISLAND (unmerged)
      // ── Re-copy consolidated arg table if gap slots ran since the last H2D copy ──
      // Gap slots (e.g., reshape/view) update outputSlots_[] pointers on the host side.
      // The consolidated arg table H2D was done before gap execution, so the GPU copy
      // has stale pointers for gap slot outputs. Re-refresh + re-copy ensures the
      // island's Triton kernels read the updated pointers from the GPU arg table.
      if (gapSlotsExecutedSinceArgCopy) {
        auto tAR0 = executionTimingEnabled_ ? Clock::now() : Clock::time_point();
#if HAVE_TRITON
        auto* tritonBackend2 = dynamic_cast<TritonGraphBackend*>(getGpuGraphBackend());
        if (tritonBackend2 != nullptr) {
          tritonBackend2->refreshArgTablesForReplay(
              seg, effectiveExternals, numExt, outputSlots_, totalOutputSlots_, stream);
          tritonBackend2->copyConsolidatedArgTableToDevice(seg, stream);
          DSP_DIAG(EXECUTE, "COMPOSITE_REPLAY: re-copied consolidated arg table after gap slots for seg[%d-%d]",
                   seg.def.startSlot, seg.def.endSlot);
        }
#endif
        if (executionTimingEnabled_) {
          tArgRefreshUs += std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - tAR0).count();
          nArgRefreshes++;
        }
        gapSlotsExecutedSinceArgCopy = false;
      }

      int idx = unit.islandIndex;
      if (idx < 0 || idx >= static_cast<int>(sched.compositeReplayHandles.size()) ||
          !sched.compositeReplayHandles[idx] || !sched.compositeReplayHandles[idx]->isReady()) {
        DSP_DIAG(EXECUTE, "COMPOSITE_REPLAY: island %d handle not ready — executing slots [%d-%d] slot-by-slot",
                 idx, unit.startSlot, unit.endSlot);
        // Fall back to slot-by-slot for this island's slots instead of failing.
        // This handles the case where capture hasn't completed yet (e.g., after
        // shape-drift recompilation or first frozen warmup).
        for (int s = unit.startSlot; s <= unit.endSlot; s++) {
          if (s < 0 || s >= numSlots_) continue;
          auto slotStatus = executeSlot(s, effectiveExternals, numExt, stream);
          if (slotStatus != Status::OK) {
            DSP_DIAG(EXECUTE, "COMPOSITE_REPLAY: island fallback slot %d FAILED status=%d",
                     s, static_cast<int>(slotStatus));
            return slotStatus;
          }
        }
        gapSlotsExecutedSinceArgCopy = true;
        DSP_DIAG(EXECUTE,
                 "COMPOSITE_REPLAY: island %d [%d-%d] fallback slot-by-slot complete "
                 "— gapSlotsExecutedSinceArgCopy=true (arg table refresh required before next island)",
                 idx, unit.startSlot, unit.endSlot);
        continue;
      }

      DSP_DIAG(EXECUTE, "COMPOSITE_REPLAY: island %d [%d-%d] launching", idx, unit.startSlot, unit.endSlot);

      auto tIL0 = executionTimingEnabled_ ? Clock::now() : Clock::time_point();
      bool launchOk = sched.compositeReplayHandles[idx]->replay(stream);
      if (executionTimingEnabled_) {
        tIslandLaunchUs += std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - tIL0).count();
        nIslandLaunches++;
      }
      if (!launchOk) {
        DSP_DIAG(EXECUTE, "COMPOSITE_REPLAY: island %d launch FAILED", idx);
        return Status::KERNEL_FAILURE;
      }

      // Mark island output slots as dirty + tick actuality in one pass.
      // Use proper wiring.outputSlotIndices mapping (not step indices directly).
      auto tID0 = executionTimingEnabled_ ? Clock::now() : Clock::time_point();
      for (int stepIdx = unit.startSlot; stepIdx <= unit.endSlot; stepIdx++) {
        if (stepIdx < 0 || stepIdx >= numSlots_) continue;
        const NativeSlot& slot = slots_[stepIdx];
        for (int o = 0; o < slot.wiring.numOutputs; o++) {
          int outIdx = slot.wiring.outputSlotIndices[o];
          if (outIdx < 0 || outIdx >= totalOutputSlots_) continue;
          dirtySlotGenerations_[outIdx] = currentDirtyGeneration_;
          NDArray* arr = outputSlots_[outIdx];
          if (arr != nullptr && arr->dataBuffer() != nullptr && !arr->dataBuffer()->isClosed()) {
            arr->tickWriteDevice();
          }
        }
      }
      if (executionTimingEnabled_) {
        tIslandDirtyUs += std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - tID0).count();
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


  auto tActTick = executionTimingEnabled_ ? Clock::now() : Clock::time_point{};

  // Diagnostic: check final output after composite replay
  dumpSegFinalArgmax(seg, outputSlots_, totalOutputSlots_, numSlots_, slots_,
                     cudaStr, "POST_COMPOSITE_REPLAY_ARGMAX", seg.exec.executionCount);

  // ── CRITICAL-PATH TIMING (active when executionTimingEnabled_) ──────────────
  if (executionTimingEnabled_) {
    auto tDone = Clock::now();
    auto totalUs = std::chrono::duration_cast<std::chrono::microseconds>(tDone - t0).count();
    auto prezeroUs = std::chrono::duration_cast<std::chrono::microseconds>(tPrezero - t0).count();
    auto actTickUs = std::chrono::duration_cast<std::chrono::microseconds>(tActTick - t0).count();
    auto unitsUs = actTickUs - prezeroUs;
    DSP_DIAG(TIMING,
             "COMPOSITE_REPLAY_TIMING: total=%lldus prezero=%lldus units=%lldus "
             "execCount=%d mergedGroups=%d islands=%d "
             "BREAKDOWN: mergedLaunch=%lldus(%d) mergedDirty=%lldus gapExec=%lldus(%d) "
             "islandLaunch=%lldus(%d) islandDirty=%lldus argRefresh=%lldus(%d) "
             "GAP_SLOTS: exec=%d tick=%d bgemm=%d",
             totalUs, prezeroUs, unitsUs, seg.exec.executionCount,
             static_cast<int>(sched.mergedReplayHandles.size()),
             static_cast<int>(sched.compositeReplayHandles.size()),
             tMergedLaunchUs, nMergedLaunches, tMergedDirtyUs,
             tGapExecUs, nGapUnits,
             tIslandLaunchUs, nIslandLaunches, tIslandDirtyUs,
             tArgRefreshUs, nArgRefreshes,
             nExecSlots, nTickSlots, nBatchedGemmSlots);
    // Per-op-name breakdown of EXECUTE gap slots (first 3 steps only to avoid log spam)
    if (!execOpCounts.empty()) {
      std::string opBreakdown = "GAP_EXEC_OPS:";
      for (auto& kv : execOpCounts) {
        opBreakdown += " " + kv.first + "=" + std::to_string(kv.second);
      }
      DSP_DIAG(TIMING, "%s", opBreakdown.c_str());
    }
  }

  // Update replay tracking
  seg.exec.lastReplayExecCount = seg.exec.executionCount;

  // Exit summary: total units replayed, breakdown by kind, no failures
  DSP_DIAG(EXECUTE,
           "COMPOSITE_REPLAY_EXIT: seg[%d-%d] OK units=%d (merged=%d islands=%d gaps=%d) "
           "execCount=%d",
           seg.def.startSlot, seg.def.endSlot,
           static_cast<int>(sched.units.size()),
           nMergedLaunches, nIslandLaunches, nGapUnits,
           seg.exec.executionCount);

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

    DSP_DIAG(MEMORY, "evictLruGraphs: invalidated seg[%d-%d] without blocking stream sync",
             evictSeg.def.startSlot, evictSeg.def.endSlot);
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
  // Only need workspace allocation if neither per-plan nor global workspace exists
  if (sharedCaptureWorkspace_ == nullptr && g_globalCaptureWorkspace == nullptr) {
    neededBytes += tritonCaptureWorkspaceSize();
  }
  if (cublasWorkspaceBuffer_ == nullptr) {
    neededBytes += Environment::getInstance().dspCublasWorkspaceMb() * 1024ULL * 1024ULL;
  }
  neededBytes += Environment::getInstance().dspGraphMetadataSafetyMb() * 1024ULL * 1024ULL;

  DSP_DIAG(MEMORY, "proactive cleanup: gpuFree=%zuMB, needed=%zuMB (ws=%zuMB, cublas=%zuMB, safety=%dMB) for seg[%d-%d]",
           gpuFree / (1024*1024), neededBytes / (1024*1024),
           (sharedCaptureWorkspace_ == nullptr ? tritonCaptureWorkspaceSize() : 0) / (1024*1024),
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
    LongType segShapeKey, const char* backendName,
    bool& handled) {

  handled = false;

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
                            static_cast<uint8_t>(seg.exec.segPhase.toLegacyCode()),
                            static_cast<uint8_t>(SegmentLifecycleState::REPLAYING),
                            static_cast<uint32_t>(executeCount_));
      }
      // markCaptured() now transitions directly to REPLAYING — no separate
      // CAPTURED→REPLAYING step needed on first replay.

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
          DSP_DIAG(EXECUTE,
                   "POST_REPLAY_OUTPUT_METADATA: seg[%d-%d] execCount=%d "
                   "lastSlot=%d len=%lld special=%p asyncValues=true",
                   seg.def.startSlot, seg.def.endSlot, seg.exec.executionCount,
                   lastSlot, (long long)checkArr->lengthOf(),
                   checkArr->dataBuffer() != nullptr ? checkArr->dataBuffer()->special() : nullptr);
        }
      }
    }
    // KERNEL_FAILURE from compositeReplay with invalidated graph: fall through
    // to capture/direct path so the caller re-captures with current addresses.
    if (replayStatus == Status::KERNEL_FAILURE && seg.exec.replayHandle == nullptr) {
      DSP_DIAG(EXECUTE,
               "COMPOSITE_REPLAY_INVALIDATED: seg[%d-%d] graph invalidated during replay "
               "(likely merged_graph_addr_drift). Falling through to re-capture.",
               seg.def.startSlot, seg.def.endSlot);
      handled = false;
      return Status::OK;
    }
    handled = true;
    return replayStatus;
  } else if (compositeReplayReady) {
    // Composite handles exist but a condition blocked replay — log which one.
    DSP_DIAG(EXECUTE, "COMPOSITE_REPLAY_BLOCKED: seg[%d-%d] allowReplay=%d shapeKeyMatch=%d "
             "(cached=%lld current=%lld) createValuesStable=%d extAddrsStable=%d "
             "— falling through to monolithic/capture path",
             seg.def.startSlot, seg.def.endSlot,
             allowTritonCudaGraphReplay ? 1 : 0,
             (seg.exec.cachedShapeKey == segShapeKey) ? 1 : 0,
             (long long)seg.exec.cachedShapeKey, (long long)segShapeKey,
             createValuesStable ? 1 : 0, extAddrsStable ? 1 : 0);
  }

  // ── MONOLITHIC REPLAY ─────────────────────────────────────────────────────
  // If composite handles aren't available but a monolithic replayHandle exists
  // (captured via the monolithic capture path), replay it directly.
  // All-frozen-constant segments are skipped before reaching here (their
  // replayHandle is cleared by ZERO_NODE_REJECT in the capture path).
  //
  // Gap ops are included in the monolithic CUDA graph when nativeOnlyGraphCapture
  // was used (compiledByBackend == "CUDA", gapOpsCapturedInGraph == true).
  // No demotion is needed — the monolithic graph is complete.
  if (allowTritonCudaGraphReplay &&
      seg.exec.replayHandle != nullptr &&
      seg.exec.replayHandle->isReady() &&
      seg.exec.cachedShapeKey == segShapeKey &&
      createValuesStable &&
      extAddrsStable) {

    DSP_DIAG(EXECUTE, "MONOLITHIC_REPLAY_ENTER: seg[%d-%d] replayHandle=%p execCount=%d",
             seg.def.startSlot, seg.def.endSlot,
             (void*)seg.exec.replayHandle.get(), seg.exec.executionCount);

    // Pre-replay sync: H2D variable inputs + cross-stream ordering.
    // performPreReplaySync is idempotent (dedup via PlanExecutionContext flags).
    NDArray** effectiveExternals = performPreReplaySync(
        externalArrays, numExt, stream, "monolithicReplay");

    // Consolidated replay: arg refresh + prezero + cuBLAS zero + replay +
    // counters + post-fixup + verify — all in one call.
    auto replayStatus = replayMonolithicGraph(seg, effectiveExternals, numExt,
                                              stream, "gpubackend_monolithic");
    if (replayStatus == Status::OK) {
      if (seg.exec.compiledByBackend.empty()) seg.exec.compiledByBackend = backendName;
      handled = true;
      return Status::OK;
    } else {
      // Monolithic replay failed — likely stale addresses or corrupted graph.
      // Invalidate and fall through to re-capture.
      DSP_DIAG(EXECUTE, "MONOLITHIC_REPLAY_FAILED: seg[%d-%d] — invalidating for re-capture",
               seg.def.startSlot, seg.def.endSlot);
      SegmentLifecycle::invalidateForRebuild(this, seg, "monolithic_replay_failed");
      handled = false;
      return Status::OK;
    }
  }

  // Replay conditions not met — caller falls through to capture/direct
  DSP_DIAG(EXECUTE,
           "REPLAY_DISPATCH_PASS: seg[%d-%d] no replay path taken "
           "(hasComposite=%d hasMonolithic=%d allowReplay=%d) — falling through to capture/direct",
           seg.def.startSlot, seg.def.endSlot,
           hasCompositeHandles(seg) ? 1 : 0,
           (seg.exec.replayHandle != nullptr && seg.exec.replayHandle->isReady()) ? 1 : 0,
           allowTritonCudaGraphReplay ? 1 : 0);
  handled = false;
  return Status::OK;
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
  bool nativeOnlyGraphCapture = false;
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
         SegmentLifecycle::copyCompilationState(gapSeg.exec, seg.exec);

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
         // -fno-threadsafe-statics: use std::call_once for thread-safe initialization.
         static std::once_flag streamDiagFlag;
         std::call_once(streamDiagFlag, [&]() {
           DSP_DIAG(BACKEND, "stream diag: tritonStr=%p gapStr=%p match=%d capturing=%d",
                    (void*)tritonStr, (void*)gapStr, streamsMatch ? 1 : 0,
                    streamIsCapturing ? 1 : 0);
         });

         if (streamIsCapturing && !tl_mergedCaptureActive) {
           // ── PER-ISLAND CAPTURE PATH: SKIP gap ops ──
           //
           // During per-island CUDA graph capture, Triton kernels are recorded
           // into the graph. Gap ops (matmul, attention, etc.) must NOT execute:
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
           // Solution: SKIP gap ops during per-island capture. Warmup already
           // executed them and populated outputSlots_ at the correct addresses.
           // On replay, the composite replay schedule executes gaps FRESH before
           // each island's graph replay.
           //
           // NOTE: This skip does NOT apply to merged capture
           // (tl_mergedCaptureActive=true). In merged capture, gap ops MUST
           // execute on the capture stream so their kernels are recorded into
           // the merged CUDA graph. Without this, gaps become non-leader units
           // in the merged group and are never re-executed during replay —
           // causing stale outputs (e.g., gather reading frozen input_ids →
           // repeating tokens).

           DSP_DIAG(EXECUTE, "GAP_SKIP_DURING_CAPTURE: gap[%d-%d] SKIPPED (warmup outputs "
                    "already at stable addresses) for seg[%d-%d]",
                    startSlot, endSlot, seg.def.startSlot, seg.def.endSlot);

           // gapOpsCapturedInGraph stays false — gaps are NOT in the graph
           return Status::OK;
         }

         if (streamIsCapturing && tl_mergedCaptureActive) {
           // ── MERGED CAPTURE PATH: gap ops on capture stream ──
           // Only execute gap ops that fall within the current island's merged
           // range. The Triton backend fires the orderedRangeExecutor for ALL
           // gaps in the segment (including trailing gaps beyond the island),
           // but gaps outside the island range are handled by the outer
           // composite capture loop. Executing them here would bake their
           // CUDA kernels into the island's graph with no arg-table refresh
           // mechanism — producing stale output on replay.
           if (tl_islandSlotMin <= tl_islandSlotMax &&
               (startSlot > tl_islandSlotMax || endSlot < tl_islandSlotMin)) {
             // Gap is outside the current island range — skip it.
             // The outer composite capture loop will execute it natively.
             DSP_DIAG(EXECUTE, "MERGED_CAPTURE_GAP_SKIP: gap[%d-%d] OUTSIDE island "
                      "range [%d-%d] — skipping (outer loop handles it)",
                      startSlot, endSlot, tl_islandSlotMin, tl_islandSlotMax);
             return Status::OK;
           }

           DSP_DIAG(EXECUTE, "MERGED_CAPTURE_GAP_EXEC: gap[%d-%d] executing on capture stream "
                    "for seg[%d-%d]",
                    startSlot, endSlot, seg.def.startSlot, seg.def.endSlot);

           NDArray** effectiveExt = tl_mergedCaptureExternals ? tl_mergedCaptureExternals : externalArrays;
           SyncOverride syncGuard(*this, "merged_capture_gap");
           auto gapStatus = executeSegmentSlotBySlot(gapSeg, effectiveExt, numExt, stream);
           return gapStatus;
         }

         // ── NON-CAPTURE PATH: normal gap execution on the Triton stream ──
         // Gap ops are routed onto tritonStr below, so same-stream ordering
         // ensures they see completed Triton outputs without blocking the host.
         // SyncOverride: TRITON and AUTO modes have forcesSyncOnFrozen=false,
         // so needsSync() returns false once executeCount>=2 + shapes frozen.
         // Without this override, gap ops skip prepareSpecialUse/registerSpecialUse,
         // causing stale device reads — the root cause of stuck output in
         // island-gap-island graphs (test47, SV12A).
         SyncOverride gapSyncGuard(*this, "triton_direct_gap");
         sd::graph::DspGraphActiveGuard graphGuard(false);
         // Use effectiveExternals_ (staging-swapped) when available so gap ops
         // read from D2D-refreshed staging buffers instead of raw Java arrays.
         // After markExternalInputVariable, effectiveExternals_ is re-created
         // zero-initialized but ensureAndSyncStagingBuffers may not have run
         // yet for this execution. Populate null entries from externalArrays
         // so gap ops don't see null inputs.
         NDArray** gapExternals = externalArrays;
         if (effectiveExternals_ != nullptr &&
             !externalInputIsVariable_.empty() &&
             !planLifecycle_.isSlotBySlot()) {
           // Backfill null entries from externalArrays — ensureAndSyncStagingBuffers
           // does this via memcpy, but may not have run yet after markVariable.
           for (int ei = 0; ei < numExt; ei++) {
             if (effectiveExternals_[ei] == nullptr && externalArrays[ei] != nullptr) {
               effectiveExternals_[ei] = externalArrays[ei];
             }
           }
           gapExternals = effectiveExternals_;
         }
         // Route gap ops onto the Triton stream for this direct path. This removes
         // cross-stream launch ordering ambiguity between Triton sub-kernels and
         // native gap ops when capture is enabled but not yet active.
         ScopedGapStreamOverride gapStreamOverride(tritonStr);
         auto gapStatus = executeSegmentSlotBySlot(gapSeg, gapExternals, numExt, stream);
         return gapStatus;  // graphGuard + gapSyncGuard restored on any exit
       });
   tritonOrderedRangeGuard.active = true;
 }
#endif  // HAVE_TRITON (ordered-range guard + callback setup)

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
  bool execCountInWindowNow = (seg.exec.executionCount >= ctx.captureMinExec);
  bool captureWindowSatisfiedNow = execCountInWindowNow || ctx.requiresOrderedGapCapture;
  // compilationFailed prevents re-capture of permanently failed segments
  // (distinct from platformShouldUseGraph which gates graph USAGE).
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
                 seg.exec.executionCount, planLifecycle_.isShapesFrozen() ? 1 : 0);
    seg.exec.gapOpsCapturedInGraph = false;

    // Set up capture workspace BEFORE beginCapture — cudaMalloc must be outside capture.
    // Native ordered range ops (matmul, attention, concat) need temporary buffers during execution.
    // With tl_graphExecutionActive=true, CudaMemoryPool allocates from this workspace
    // instead of calling cudaMallocAsync (which fails during capture).
    // tritonCaptureWorkspaceSize() is now at file scope (above).

    // Create the replayHandle BEFORE capture — it must exist to store workspace, host ptrs, etc.
    {
      int deviceId = 0;
      cudaGetDevice(&deviceId);
      seg.exec.replayHandle = GraphReplayFactory::create(deviceId);
    }

    if (seg.exec.replayHandle->getWorkspacePtr() == nullptr) {
      int deviceId = 0;
      cudaGetDevice(&deviceId);

      // Global shared workspace: allocate once globally, reuse across ALL plans.
      // CUDA graph capture is serialized by DeviceCaptureGuard (only one plan
      // captures at a time per device), so the global workspace is safe to share.
      // This avoids OOM when multiple concurrent plans each try to allocate
      // their own 512MB workspace.
      if (sharedCaptureWorkspace_ == nullptr) {
        // Try to reuse the global workspace first (allocated by another plan)
        if (g_globalCaptureWorkspace != nullptr && g_globalCaptureWorkspaceDevice == deviceId) {
          sharedCaptureWorkspace_ = g_globalCaptureWorkspace;
          sharedCaptureWorkspaceBytes_ = g_globalCaptureWorkspaceBytes;
          sharedCaptureWorkspaceDevice_ = deviceId;
          DSP_DIAG_SEG(MEMORY, seg.def.startSlot,
                       "reusing GLOBAL capture workspace: %zuMB on device %d",
                       sharedCaptureWorkspaceBytes_ / (1024*1024), deviceId);
        } else {
          // First plan on this device — allocate the global workspace.
          // Trim pool to reclaim freed async memory before allocating.
          memory::CudaMemoryPool::getInstance().trimPool(deviceId);

          // Adaptive workspace sizing — scale down to fit available GPU memory.
          size_t gpuFree = 0, gpuTotal = 0;
          cudaMemGetInfo(&gpuFree, &gpuTotal);
          size_t headroom = 256ULL * 1024 * 1024;
          size_t workspaceSize = tritonCaptureWorkspaceSize();
          if (gpuFree > headroom) {
            size_t availableForWs = gpuFree - headroom;
            if (availableForWs < workspaceSize) {
              workspaceSize = availableForWs;
              DSP_DIAG_SEG(MEMORY, seg.def.startSlot,
                           "global workspace scaled down: gpuFree=%zuMB headroom=%zuMB -> workspace=%zuMB (max=%zuMB)",
                           gpuFree / (1024*1024), headroom / (1024*1024),
                           workspaceSize / (1024*1024), tritonCaptureWorkspaceSize() / (1024*1024));
            }
          } else {
            workspaceSize = 32ULL * 1024 * 1024;
            DSP_DIAG_SEG(MEMORY, seg.def.startSlot,
                         "global workspace minimal: gpuFree=%zuMB < headroom=%zuMB -> workspace=32MB",
                         gpuFree / (1024*1024), headroom / (1024*1024));
          }

          cudaError_t err = cudaMalloc(&sharedCaptureWorkspace_, workspaceSize);
          if (err != cudaSuccess) {
            cudaGetLastError();
            sharedCaptureWorkspace_ = nullptr;
          }
          if (sharedCaptureWorkspace_ != nullptr) {
            sharedCaptureWorkspaceBytes_ = workspaceSize;
            sharedCaptureWorkspaceDevice_ = deviceId;
            // Promote to global so other plans can reuse
            g_globalCaptureWorkspace = sharedCaptureWorkspace_;
            g_globalCaptureWorkspaceBytes = sharedCaptureWorkspaceBytes_;
            g_globalCaptureWorkspaceDevice = deviceId;
            memory::CudaMemoryPool::getInstance().registerCaptureWorkspace(
                sharedCaptureWorkspace_, sharedCaptureWorkspaceBytes_);
            DSP_DIAG_SEG(MEMORY, seg.def.startSlot,
                         "allocated GLOBAL capture workspace: %zuMB on device %d (max=%zuMB)",
                         workspaceSize / (1024*1024), deviceId,
                         tritonCaptureWorkspaceSize() / (1024*1024));
          } else {
            SegmentLifecycle::invalidateForRebuild(this, seg, "oom_shared_workspace");
#if HAVE_TRITON
            tritonOrderedRangeGuard.active = false;
            TritonGraphBackend::clearOrderedRangeExecutor();
#endif
            return reportOomError(seg, "shared_workspace_allocation",
                                  workspaceSize, deviceId);
          }
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

      // Set capture stream so captureSafeStreamOrDefault() routes ops to the correct stream.
      // Resolve null ctx.cudaStr to LaunchContext default to match the actual capture stream
      // (beginCapture passes ctx.cudaStr to cudaStreamBeginCapture; if null, CUDA uses the
      // default stream which is what LaunchContext::defaultContext()->getCudaStream() returns).
      cudaStream_t prevCaptureStream = tl_graphCaptureStream;
      {
        cudaStream_t resolvedCaptureStream = ctx.cudaStr;
        if (resolvedCaptureStream == nullptr) {
          auto* defaultStreamPtr = LaunchContext::defaultContext()->getCudaStream();
          if (defaultStreamPtr != nullptr) resolvedCaptureStream = *defaultStreamPtr;
        }
        tl_graphCaptureStream = resolvedCaptureStream;
      }
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

      // Cross-stream + H2D + staging sync: already done by performPreReplaySync
      // in dispatchSegment (tracked via PreReplaySyncPhase). No redundant sync here.
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

        // Prior work is ordered onto ctx.cudaStr by the execution context and
        // composite replay code. Run warmup on the same stream so it observes
        // preceding segment outputs without blocking the host.

        // Set cuBLAS workspace during warmup too, so cuBLAS selects the same GEMM
        // algorithms as during capture. Without this, warmup may use different
        // algorithms than capture, causing shape/result divergence.
        setCublasWorkspaceForWarmup();

        // Disable frozen fast path for warmup — same rationale as capture below.
        std::vector<SlotPhase> savedSlotPhasesWarmup;
        demoteFrozenSlotStates(slots_, seg.def.startSlot, seg.def.endSlot, savedSlotPhasesWarmup);

        // Use slot-by-slot for warmup — matches the REF path exactly.
        //
        // CRITICAL: Force host↔device sync during warmup.
        // executeSlot gates prepareSpecialUse/registerSpecialUse on:
        //   needsSync = !planLifecycle_.isShapesFrozen() || executeCount_ < 2
        // At exec=2, executeCount_=2 → needsSync=false → device coherency
        // calls are skipped entirely. The warmup NEEDS these calls because
        // prior segments' composite captures may have changed actuality flags
        // (validation replay writes device, ticks special-actual but not
        // primary-actual). Without prepareSpecialUse, ops read stale device
        // memory → zero outputs from seg[400+] onwards.
        // SyncOverride forces sync during pre-capture warmup.
        Status warmupStatus;
        {
          SyncOverride syncGuard(*this, "pre_capture_warmup");
          ScopedGapStreamOverride warmupStreamGuard(ctx.cudaStr);

          GraphSegment warmupSeg;
          warmupSeg.def.startSlot = seg.def.startSlot;
          warmupSeg.def.endSlot = seg.def.endSlot;
          warmupSeg.exec.executionCount = seg.exec.executionCount;
          SegmentLifecycle::copyCompilationState(warmupSeg.exec, seg.exec);
          warmupStatus = executeSegmentSlotBySlot(warmupSeg, externalArrays, numExt, stream);
        }
        restoreSlotStates(slots_, seg.def.startSlot, seg.def.endSlot, savedSlotPhasesWarmup);

        if (warmupStatus != Status::OK) {
          DSP_DIAG(EXECUTE, "FATAL: Triton pre-capture warmup FAILED for seg[%d-%d] status=%d. "
                            "BLOCKING EXECUTION.",
                   seg.def.startSlot, seg.def.endSlot, static_cast<int>(warmupStatus));
          SegmentLifecycle::markFailed(seg.exec, "pre_capture_warmup_failed", seg.def.startSlot, seg.def.endSlot);
          // savedSlotPhasesTriton has not been populated yet (demote happens after warmup).
          // Pass empty vector — no capture-phase slot demotion to undo here.
          const std::vector<SlotPhase> emptySlotPhases;
          abortCapture(seg, true, didPushCtx, tritonCaptureDevice,
                      prevCaptureStream, emptySlotPhases, stream);
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

        DSP_DIAG(EXECUTE, "Triton pre-capture warmup for seg[%d-%d] queued on capture stream=%p "
                          "(no blocking stream sync)",
                 seg.def.startSlot, seg.def.endSlot, (void*)ctx.cudaStr);
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
        // Read once per process — std::call_once protects against -fno-threadsafe-statics race
        static std::once_flag warmupOnceFlag;
        static bool warmupOnly = false;
        std::call_once(warmupOnceFlag, []() { warmupOnly = Environment::getInstance().triton().warmupOnly(); });
        if (warmupOnly) {
          DSP_DIAG(EXECUTE, "WARMUP_ONLY: skipping capture for seg[%d-%d], using warmup result",
                   seg.def.startSlot, seg.def.endSlot);
          SegmentLifecycle::markFailed(seg.exec, "warmup_only_mode", seg.def.startSlot, seg.def.endSlot);
          // savedSlotPhasesTriton has not been populated yet (demote happens after warmup).
          // Pass empty vector — no capture-phase slot demotion to undo here.
          const std::vector<SlotPhase> emptySlotPhases;
          abortCapture(seg, true, didPushCtx, tritonCaptureDevice,
                      prevCaptureStream, emptySlotPhases, stream);
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
      //  Once shapes are frozen (planLifecycle_.isShapesFrozen() == true), NEVER zero the cuBLAS workspace.
      // During capture, cuBLAS stores plan/descriptor data in the workspace. Captured CUDA graphs
      // inherit these cached plans and omit H2D re-upload nodes. Zeroing the workspace destroys
      // cached plans, causing GEMM kernels to read zeros and hang on replay.
      //
      // The workspace content must be preserved across ALL captures and replays once frozen.
      // cuBLAS plans are stable for fixed shapes, so preservation is safe.
      //
      // Pre-frozen (shapes not yet frozen): zeroing is acceptable as no graphs are captured yet.
      if (!planLifecycle_.isSlotBySlot() && cublasWorkspaceBuffer_ != nullptr && cublasWorkspaceSize_ > 0) {
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
      std::vector<SlotPhase> savedSlotPhasesTriton;
      demoteFrozenSlotStates(slots_, seg.def.startSlot, seg.def.endSlot, savedSlotPhasesTriton);

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
      if (!planLifecycle_.isSlotBySlot()) {
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
      //
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
      // performPreReplaySync in dispatchSegment already synced external inputs.
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
      // Skip capture entirely if the segment is in OOM_DEFERRED state (deferred by a
      // prior concurrent capture failure). This prevents multiple threads from
      // redundantly attempting capture on the same segment.
      if (seg.exec.captureOomRetries > 0 &&
          seg.exec.executionCount < seg.exec.captureRetryAfterExec) {
        didCompositeCapture = true;  // skip both composite and monolithic capture
        DSP_DIAG_SEG(EXECUTE, seg.def.startSlot,
                     "CAPTURE_SKIP: seg[%d-%d] is in OOM_DEFERRED state — "
                     "falling through to slot-by-slot (retryAfterExec=%d, current=%d)",
                     seg.def.startSlot, seg.def.endSlot,
                     seg.exec.captureRetryAfterExec, seg.exec.executionCount);
      }
#if HAVE_TRITON
      {
       auto& sched = seg.exec.compositeReplaySchedule;
       bool hasIslandUnits = false;
       for (auto& u : sched.units) {
         if (u.kind == REPLAY_UNIT_TRITON_ISLAND) { hasIslandUnits = true; break; }
       }
       if (hasIslandUnits && !sched.units.empty() && !didCompositeCapture) {
         // Serialize composite capture per GPU — hold lock for entire composite capture.
         DeviceCaptureGuard compositeCaptureGuard;
         if (!compositeCaptureGuard.acquired()) {
           // Another thread is capturing — skip capture this iteration.
           // didCompositeCapture stays false; falls through to monolithic path (which
           // will also fail try_lock) then to direct slot-by-slot execution.
           DSP_DIAG(COMPILE, "COMPOSITE_CAPTURE_DEFER: seg[%d-%d] another thread capturing, will retry next exec",
                    seg.def.startSlot, seg.def.endSlot);
         } else {
         DSP_DIAG(COMPILE, "COMPOSITE_CAPTURE_ENTER: seg[%d-%d] units=%d hasIsland=1 execCount=%d",
                  seg.def.startSlot, seg.def.endSlot, (int)sched.units.size(), executeCount_);
         DSP_DIAG(EXECUTE, "COMPOSITE_CAPTURE_BEGIN: seg[%d-%d] units=%d "
                  "segPhase=%s shapesFrozen=%d planPhase=%s executeCount=%d "
                  "tl_graphExecutionActive=%d tl_cublasWorkspacePtr=%p/%zu",
                  seg.def.startSlot, seg.def.endSlot, static_cast<int>(sched.units.size()),
                  seg.exec.segPhase.displayName(), (int)planLifecycle_.isShapesFrozen(),
                  planLifecycle_.displayName(), executeCount_,
                  (int)tl_graphExecutionActive, (void*)tl_cublasWorkspacePtr, tl_cublasWorkspaceSize);
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

         // Gap capturability is checked on-demand via isGapRangeCaptureSafe()
         // — no reclassification needed since there's no cached flag.
         bool mergeViewsNow = Environment::getInstance().triton().mergedCaptureThroughViews();

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
          CaptureLifecycleGuard mergedCapGuard;
          std::unique_ptr<GraphReplayHandle> mergedHandle;
         sd::cuda::CudaGraphHandle* mergedNativeHandle = nullptr;
         int mergedGroupId = -1;          // Current merged group index
         int mergedLeaderUnitIdx = -1;    // Unit index of the group leader
         int mergedStartSlot = INT_MAX;   // Slot range of the entire merged group
         int mergedEndSlot = INT_MIN;
         size_t nodesAtCaptureStart = 0;  // Incremental node tracking

         for (size_t unitIdx = 0; unitIdx < sched.units.size() && allIslandsOk; unitIdx++) {
           auto& unit = sched.units[unitIdx];

            if (unit.kind == REPLAY_UNIT_GAP) {
              if (captureActive && isGapRangeCaptureSafe(slots_, unit.startSlot, unit.endSlot, mergeViewsNow)) {
                // ── MERGED CAPTURE: gap ops recorded on capture stream ──────
                // tl_graphExecutionActive is already true from the preceding island.
                // tl_dspGapStream = ctx.cudaStr makes cuBLAS et al. use capture stream.
                // SyncOverride for prepareSpecialUse correctness at exec=2.
                 DSP_DIAG(EXECUTE, "MERGED_CAPTURE_GAP_BEGIN: gap [%d-%d] mergedGroup=%d "
                         "tl_dspGapStream=%p->%p tl_cublasWorkspacePtr=%p/%zu "
                         "tl_graphExecutionActive=%d ctx.cudaStr=%p",
                         unit.startSlot, unit.endSlot, mergedGroupId,
                         (void*)tl_dspGapStream, (void*)ctx.cudaStr,
                         (void*)tl_cublasWorkspacePtr, tl_cublasWorkspaceSize,
                         (int)tl_graphExecutionActive, (void*)ctx.cudaStr);

               // Expand the island filter to cover this capture-safe gap so the
               // gap handler in executeSegment also accepts gaps within this range.
               if (unit.startSlot < tl_islandSlotMin) tl_islandSlotMin = unit.startSlot;
               if (unit.endSlot > tl_islandSlotMax) tl_islandSlotMax = unit.endSlot;

               // Direct gap-stream to capture stream so cuBLAS records here.
               // RAII restore is required: executeSlot may throw.
               ScopedGapStreamOverride gapStreamOverride(ctx.cudaStr);
               SyncOverride gapSyncGuard(*this, "merged_capture_gap_begin");

               bool gapOk = true;
               for (int s = unit.startSlot; s <= unit.endSlot; s++) {
                 // Skip frozen constant slots — their outputs are populated from
                 // warmup and excluding them reduces the captured graph node count.
                 if (slots_[s].frozenConstantSlot()) {
                   bool allPop = true;
                   for (int o = 0; o < slots_[s].wiring.numOutputs; o++) {
                     int si = slots_[s].wiring.outputSlotIndices[o];
                     if (si >= 0 && si < totalOutputSlots_ && outputSlots_[si] == nullptr) {
                       allPop = false;
                       break;
                     }
                   }
                   if (allPop) continue;
                 }
                   // Use effectiveExternalsForCapture (staging buffers) so the CUDA graph
                   // bakes in stable plan-owned device addresses, not Java-side pointers
                   // that may be reallocated between steps.
                   {
                     bool _preInvalid = false;
                     DSP_CAPTURE_PROBE(ctx.cudaStr, s, "BEFORE_GAP_SLOT",
                                       slots_[s].ident.opName.c_str(), _preInvalid);
                     if (_preInvalid) { gapOk = false; break; }
                   }
                   auto gapStatus = executeSlot(s, effectiveExternalsForCapture, numExt, stream);
                   {
                     bool _postInvalid = false;
                     DSP_CAPTURE_PROBE(ctx.cudaStr, s, "AFTER_GAP_SLOT",
                                       slots_[s].ident.opName.c_str(), _postInvalid);
                     if (_postInvalid) { gapOk = false; break; }
                   }
                  if (gapStatus != Status::OK) {
                    DSP_DIAG(EXECUTE, "MERGED_CAPTURE: gap slot %d FAILED status=%d",
                             s, static_cast<int>(gapStatus));
                    gapOk = false;
                    break;
                  }
                }

                if (!gapOk) {
                  DSP_DIAG(COMPILE, "COMPOSITE_CAPTURE_FAIL: gap [%d-%d] invalidated capture (gapOk=false)",
                           unit.startSlot, unit.endSlot);
                  // Gap op invalidated capture — likely a pool allocation
                  // (cudaMallocAsync/cudaFreeAsync) that's capture-incompatible.
                  // After capture invalidation the GPU stream state is undefined —
                  // re-executing slots on this stream causes illegal memory access
                  // (error 700). Instead: abort the entire merged capture and let
                  // the existing failure path (allIslandsOk=false) mark the segment
                  // non-capturable and fall through to slot-by-slot execution,
                  // which re-executes everything from scratch on a clean stream.
                  DSP_DIAG(EXECUTE, "MERGED_CAPTURE_GAP_ABORT: gap [%d-%d] invalidated "
                           "capture — aborting group=%d, will fall through to slot-by-slot",
                           unit.startSlot, unit.endSlot, mergedGroupId);
                  mergedCapGuard.deactivate();
                  tl_mergedCaptureActive = false;
                  tl_mergedCaptureExternals = nullptr;
                  tl_islandSlotMin = INT_MAX;
                  tl_islandSlotMax = INT_MIN;
                  if (mergedNativeHandle->isCapturing()) {
                    mergedNativeHandle->endCapture(ctx.cudaStr);
                  }
                  // Clear any lingering CUDA error from the invalidated capture
                  cudaGetLastError();
                  // The fallback slot-by-slot path uses this same stream, so
                  // stream order preserves cleanup/execution ordering without
                  // blocking the host here.
                  cudaGetLastError();

                  captureActive = false;
                  mergedHandle.reset();
                  mergedNativeHandle = nullptr;
                  allIslandsOk = false;
                  break;
               }

               // ── Incremental node tracking: gap ──────────────────────────────
               if (mergedNativeHandle != nullptr) {
                 size_t nodesNow = mergedNativeHandle->getNumNodesDuringCapture(ctx.cudaStr);
                 size_t delta = (nodesNow > nodesAtCaptureStart) ? (nodesNow - nodesAtCaptureStart) : 0;
                 int gapSlotCount = unit.endSlot - unit.startSlot + 1;
                 DSP_DIAG(EXECUTE, "MERGED_CAPTURE_NODES: after gap [%d-%d] (%d slots) "
                          "totalNodes=%zu delta=+%zu (%.1f nodes/slot) group=%d",
                          unit.startSlot, unit.endSlot, gapSlotCount,
                          nodesNow, delta,
                          gapSlotCount > 0 ? (double)delta / gapSlotCount : 0.0,
                          mergedGroupId);
                 nodesAtCaptureStart = nodesNow;
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
                 mergedCapGuard.deactivate();
                 tl_mergedCaptureActive = false;
               tl_mergedCaptureExternals = nullptr;
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
             {
               SyncOverride gapSync(*this, "composite_gap_native");
               for (int s = unit.startSlot; s <= unit.endSlot; s++) {
                 auto gapStatus = executeSlot(s, effectiveExternalsForCapture, numExt, stream);
                 if (gapStatus != Status::OK) {
                   DSP_DIAG(EXECUTE, "COMPOSITE_CAPTURE: gap slot %d FAILED status=%d",
                            s, static_cast<int>(gapStatus));
                   allIslandsOk = false;
                   break;
                 }
               }
             }

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

               // Set cuBLAS handle to capture stream and provide explicit workspace.
               // Without this, cuBLAS gap ops (e.g. rms_norm_linear GEMM) during merged
               // capture either launch on the wrong stream or create internal MemAlloc/MemFree
               // graph nodes — both cause SIGSEGV on cudaGraphLaunch during validation replay.
               setCublasWorkspaceForCapture(stream);

               bool beginOk = mergedNativeHandle->beginCapture(ctx.cudaStr, cudaStreamCaptureModeThreadLocal);
               if (!beginOk) {
                 DSP_DIAG(EXECUTE, "MERGED_CAPTURE: island %d beginCapture FAILED", islandIdx);
                 mergedHandle.reset();
                 mergedNativeHandle = nullptr;
                 allIslandsOk = false;
                 break;
               }
                 nodesAtCaptureStart = 0;
                 tl_mergedCaptureActive = true;
                 tl_mergedCaptureExternals = effectiveExternalsForCapture;
                 captureActive = true;
                 mergedCapGuard.activate();
                 // Set capture stream TLS immediately after every activate() call.
                 // CaptureLifecycleGuard::deactivate() (called at end of each merged group)
                 // nulls tl_graphCaptureStream. When a subsequent island leader calls
                 // activate() again, tl_graphCaptureStream is still null — causing
                 // captureSafeStreamOrDefault() to return the default stream (which IS
                 // actively capturing after beginCapture above), triggering CUDA error 901.
                 {
                   cudaStream_t resolvedMergedCaptureStream = ctx.cudaStr;
                   if (resolvedMergedCaptureStream == nullptr) {
                     auto* defaultStreamPtr = LaunchContext::defaultContext()->getCudaStream();
                     if (defaultStreamPtr != nullptr) resolvedMergedCaptureStream = *defaultStreamPtr;
                   }
                   tl_graphCaptureStream = resolvedMergedCaptureStream;
                 }
                 DSP_DIAG(EXECUTE, "MERGED_CAPTURE_TLS_STATE: after beginCapture+activate island=%d "
                          "tl_graphExecutionActive=%d tl_mergedCaptureActive=%d "
                          "tl_graphCaptureStream=%p tl_captureWorkspace=%p/%zu "
                          "tl_cublasWorkspacePtr=%p/%zu tl_dspGapStream=%p",
                          islandIdx, (int)tl_graphExecutionActive, (int)tl_mergedCaptureActive,
                          (void*)tl_graphCaptureStream, (void*)tl_captureWorkspace, tl_captureWorkspaceSize,
                          (void*)tl_cublasWorkspacePtr, tl_cublasWorkspaceSize, (void*)tl_dspGapStream);
              } else {
               // ── Extend existing merged capture to this island ──────────────
               DSP_DIAG(EXECUTE, "MERGED_CAPTURE: extending to island %d [%d-%d] mergedGroup=%d",
                        islandIdx, unit.startSlot, unit.endSlot, mergedGroupId);
               if (unit.endSlot > mergedEndSlot) mergedEndSlot = unit.endSlot;
               // Expand the island filter so the gap handler accepts gaps within
               // the extended merged range (capture-safe gaps between islands).
               if (unit.startSlot < tl_islandSlotMin) tl_islandSlotMin = unit.startSlot;
               if (unit.endSlot > tl_islandSlotMax) tl_islandSlotMax = unit.endSlot;

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

               {
                 bool _islandInvalid = false;
                 DSP_CAPTURE_PROBE(ctx.cudaStr, unit.startSlot, "AFTER_TRITON_ISLAND",
                                   "triton_island", _islandInvalid);
                 if (_islandInvalid) captureStatus = Status::KERNEL_FAILURE;
               }

               // ── Incremental node tracking: island ────────────────────────────
               if (captureStatus == Status::OK && mergedNativeHandle != nullptr) {
                 size_t nodesNow = mergedNativeHandle->getNumNodesDuringCapture(ctx.cudaStr);
                 size_t delta = (nodesNow > nodesAtCaptureStart) ? (nodesNow - nodesAtCaptureStart) : 0;
                 DSP_DIAG(EXECUTE, "MERGED_CAPTURE_NODES: after island %d [%d-%d] "
                          "totalNodes=%zu delta=+%zu group=%d",
                          islandIdx, unit.startSlot, unit.endSlot,
                          nodesNow, delta, mergedGroupId);
                 nodesAtCaptureStart = nodesNow;
               }

             // Check if next unit is a capture-safe gap — if so, keep capture active
             bool keepCaptureOpen = false;
             if (captureStatus == Status::OK && unitIdx + 1 < sched.units.size()) {
               auto& nextUnit = sched.units[unitIdx + 1];
               if (nextUnit.kind == REPLAY_UNIT_GAP &&
                   isGapRangeCaptureSafe(slots_, nextUnit.startSlot, nextUnit.endSlot, mergeViewsNow)) {
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
              DSP_DIAG(EXECUTE, "MERGED_CAPTURE_END_MIDLOOP: island=%d group=%d [%d-%d] "
                       "captureStatus=%d keepCaptureOpen=%d — ending merged capture",
                       islandIdx, mergedGroupId, mergedStartSlot, mergedEndSlot,
                       (int)captureStatus, (int)keepCaptureOpen);
              mergedCapGuard.deactivate();
              tl_mergedCaptureActive = false;
             tl_mergedCaptureExternals = nullptr;
             tl_islandSlotMin = INT_MAX;
             tl_islandSlotMax = INT_MIN;

             if (captureStatus != Status::OK) {
               DSP_DIAG(COMPILE, "COMPOSITE_CAPTURE_FAIL: island captureStatus=%d at group=%d [%d-%d]",
                        (int)captureStatus, mergedGroupId, mergedStartSlot, mergedEndSlot);
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
                 DSP_DIAG(COMPILE, "COMPOSITE_CAPTURE_FAIL: validateAndStore FAILED group=%d [%d-%d] nodes=%zu",
                          mergedGroupId, mergedStartSlot, mergedEndSlot, nodeCount);
                 allIslandsOk = false;
               }
             } else {
               DSP_DIAG(COMPILE, "COMPOSITE_CAPTURE_FAIL: endCapture=%d nodeCount=%zu group=%d [%d-%d]",
                        endOk ? 1 : 0, nodeCount, mergedGroupId, mergedStartSlot, mergedEndSlot);
               allIslandsOk = false;
             }

             captureActive = false;
             tl_mergedCaptureActive = false;
             tl_mergedCaptureExternals = nullptr;
             mergedHandle.reset();
             mergedNativeHandle = nullptr;
           }
         }  // end for each unit

         // If capture still active at end of schedule, finalize it
          if (captureActive) {
            DSP_DIAG(EXECUTE, "MERGED_CAPTURE_TAIL_FINALIZE: group=%d [%d-%d] — "
                     "tl_graphExecutionActive=%d tl_mergedCaptureActive=%d "
                     "tl_cublasWorkspacePtr=%p/%zu tl_captureWorkspace=%p/%zu",
                     mergedGroupId, mergedStartSlot, mergedEndSlot,
                     (int)tl_graphExecutionActive, (int)tl_mergedCaptureActive,
                     (void*)tl_cublasWorkspacePtr, tl_cublasWorkspaceSize,
                     (void*)tl_captureWorkspace, tl_captureWorkspaceSize);
            mergedCapGuard.deactivate();
            tl_mergedCaptureActive = false;
           tl_mergedCaptureExternals = nullptr;
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
            // Guard: only transition lifecycle if the segment is not already SEALED.
            // Plan cache can return a plan whose segment was previously captured+sealed;
            // re-capturing updates handles but must not re-fire the lifecycle transition.
            // Use ctx.segInputAddrKey: variable inputs are skipped in computeSegmentInputAddrKey,
            // so the key is identical whether computed against raw or staged externals.
            if (!seg.exec.segPhase.isSealed()) {
              SegmentLifecycle::markCaptured(seg.exec, ctx.segInputAddrKey, ctx.createValueKey,
                  computeSlotAddrHash(outputSlots_, seg.def.startSlot, seg.def.endSlot, totalOutputSlots_),
                  ctx.backendName);
            } else {
              // Already sealed — just update the replay keys without lifecycle transition.
              seg.exec.sealCapture(ctx.segInputAddrKey, ctx.createValueKey,
                  computeSlotAddrHash(outputSlots_, seg.def.startSlot, seg.def.endSlot, totalOutputSlots_),
                  ctx.backendName, false);
              DSP_DIAG(EXECUTE, "COMPOSITE_CAPTURE: seg[%d-%d] already SEALED — updated replay keys without lifecycle transition",
                       seg.def.startSlot, seg.def.endSlot);
            }
            // Force arg-table refresh on the very first post-capture replay.
            // markCaptured() does NOT call markArgsCurrent(), so capturedArgGeneration
            // still equals argTableGeneration from the most-recent warmup execution.
            // When a new plan instance re-uses the same CompiledKernel singleton,
            // its cachedArgTableHostPinned may hold staging-buffer addresses from the
            // previous plan.  bumpArgGeneration() makes needsArgRefresh()=true,
            // guaranteeing refreshArgTablesForReplay() runs with this plan's own
            // effectiveExternals before the first CUDA graph launch.
            seg.exec.bumpArgGeneration();

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

           // ── Pre-compute per-merged-group slot ranges ─────────────────
           // Build [minSlot, maxSlot] for each merged group so the replay loop
           // can dirty-mark + tickWriteDevice in a single O(range) pass per
           // leader instead of scanning all units to find matching groups.
           {
             int numMergedGroups = static_cast<int>(sched.mergedReplayHandles.size());
             sched.mergedGroupSlotRanges.resize(numMergedGroups, {INT_MAX, INT_MIN});
             for (const auto& u : sched.units) {
               if (u.mergedGroupId >= 0 && u.mergedGroupId < numMergedGroups) {
                 auto& range = sched.mergedGroupSlotRanges[u.mergedGroupId];
                 if (u.startSlot < range.minSlot) range.minSlot = u.startSlot;
                 if (u.endSlot > range.maxSlot) range.maxSlot = u.endSlot;
               }
             }
           }

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
           restoreSlotStates(slots_, seg.def.startSlot, seg.def.endSlot, savedSlotPhasesTriton);

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
           // Partial failure — log which unit failed and why (COMPILE category
           // is always enabled for graph-capture configs, unlike EXECUTE).
           DSP_DIAG(COMPILE, "COMPOSITE_CAPTURE_FAILED_DETAIL: seg[%d-%d] allIslandsOk=false "
                    "units=%d mergedHandles=%d — scanning for first failure...",
                    seg.def.startSlot, seg.def.endSlot,
                    (int)sched.units.size(), (int)sched.mergedReplayHandles.size());
           for (size_t ui = 0; ui < sched.units.size(); ui++) {
             auto& u = sched.units[ui];
             DSP_DIAG(COMPILE, "  unit[%d] kind=%d [%d-%d] mergedGroupId=%d isMergedLeader=%d",
                      (int)ui, (int)u.kind, u.startSlot, u.endSlot, u.mergedGroupId,
                      u.isMergedLeader ? 1 : 0);
           }
           // Free any successfully captured merged handles
           size_t numCapturedIslands = sched.mergedReplayHandles.size();
           for (auto& h : sched.mergedReplayHandles) {
             h.reset();
           }
           sched.mergedReplayHandles.clear();
           // Clear merged group tags on units
           for (auto& u : sched.units) {
             u.mergedGroupId = -1;
             u.isMergedLeader = false;
           }
           // Composite capture failed. Check if memory pressure is the cause —
           // concurrent captures can exhaust GPU memory, causing cudaMallocAsync
           // to fail during capture, which poisons the capture stream (error 901).
           // In that case, defer and retry rather than permanently failing.
           abortCapture(seg, false, didPushCtx, tritonCaptureDevice,
                       prevCaptureStream, savedSlotPhasesTriton, stream);
           tritonOrderedRangeGuard.active = false;
           TritonGraphBackend::clearOrderedRangeExecutor();

           // ── Device health check after capture abort ──────────────────
           // If error 700 (illegal memory access) already corrupted the device,
           // we must NOT fall through to slot-by-slot — it would SIGSEGV.
           // Sync the stream to surface any pending async errors, then check.
           {
             cudaError_t syncErr = cudaStreamSynchronize(ctx.cudaStr);
             cudaError_t stickyErr = cudaGetLastError();
             if (syncErr == cudaErrorIllegalAddress || stickyErr == cudaErrorIllegalAddress) {
               DSP_DIAG_SEG(MEMORY, seg.def.startSlot,
                            "COMPOSITE_CAPTURE ABORT: device in error 700 state "
                            "(syncErr=%d stickyErr=%d). Cannot fall through to slot-by-slot.",
                            (int)syncErr, (int)stickyErr);
               SegmentLifecycle::markFailed(seg.exec, "device_error_700_after_capture_abort",
                                           seg.def.startSlot, seg.def.endSlot);
               DSP_THROW_SEG(EXECUTE, seg.def.startSlot,
                             "COMPOSITE_CAPTURE_DEVICE_ERROR: seg[%d-%d] device in unrecoverable "
                             "error 700 state after capture abort. GPU memory exhausted during "
                             "composite capture of %zu islands.",
                             seg.def.startSlot, seg.def.endSlot, numCapturedIslands);
             }
           }

           size_t gpuFreeAtFail = 0, gpuTotalAtFail = 0;
           cudaMemGetInfo(&gpuFreeAtFail, &gpuTotalAtFail);
           // Treat ALL capture failures as retryable for the first few attempts.
           // Capture can fail for transient reasons in concurrent scenarios:
           // - Cross-thread stream poisoning (error 906/901)
           // - Memory pressure during capture
           // - CUDA driver internal contention
           // Only permanently fail after exhausting retries.
           if (seg.exec.captureOomRetries < GraphSegment::maxOomRetries()) {
             int retryAfter = seg.exec.executionCount + GraphSegment::retryInterval();
             SegmentLifecycle::markOomDeferred(seg.exec, retryAfter);
             DSP_DIAG_SEG(MEMORY, seg.def.startSlot,
                          "COMPOSITE_CAPTURE FAILED — retry %d/%d, gpuFree=%zuMB. "
                          "retryAfterExec=%d. Falling through to slot-by-slot.",
                          seg.exec.captureOomRetries, GraphSegment::maxOomRetries(),
                          gpuFreeAtFail / (1024*1024), retryAfter);
             // Prevent fallthrough to monolithic capture — the segment state
             // was already cleaned up by abortCapture. Monolithic would try to
             // capture again with a stale replayHandle and fail.
             didCompositeCapture = true;
             // Don't throw — fall through to slot-by-slot execution for this step
           } else {
             SegmentLifecycle::markFailed(seg.exec, "composite_capture_failed", seg.def.startSlot, seg.def.endSlot);
             DSP_THROW_SEG(COMPILE, seg.def.startSlot,
                           "COMPOSITE_CAPTURE_FAILED: seg[%d-%d] merged capture failed after %d retries. "
                           "Fix the capture — do NOT reconfigure the segment.",
                           seg.def.startSlot, seg.def.endSlot, seg.exec.captureOomRetries);
           }
         }
         } // end else (compositeCaptureGuard.acquired())
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
        restoreSlotStates(slots_, seg.def.startSlot, seg.def.endSlot, savedSlotPhasesTriton);
        // NOTE: Do NOT clear orderedRangeExecutor_ here. The MONOLITHIC_CAPTURE_SKIP
        // path falls through to the direct exec path (line ~4184), which calls
        // ctx.backend->executeSegment(). That function needs the executor for gaps
        // between Triton sub-kernels. The RAII guard cleans up at function exit.
        // Fall through — usedTritonGraphCapture stays false → direct exec path.
      } else {

      auto* cudaReplay = static_cast<CudaGraphReplayHandle*>(seg.exec.replayHandle.get());
      // Raw pointer — no refcount increment, no risk of touching freed control block.
      auto* handle = cudaReplay->getNativeHandle();
      // Serialize capture per GPU — only one capture can be active per device at a time.
      DeviceCaptureGuard monolithicCaptureGuard;
      if (!monolithicCaptureGuard.acquired()) {
        // Another thread is capturing — skip monolithic capture this iteration.
        // Falls through to direct slot-by-slot execution below.
        DSP_DIAG(COMPILE, "MONOLITHIC_CAPTURE_DEFER: seg[%d-%d] another thread capturing, will retry next exec",
                 seg.def.startSlot, seg.def.endSlot);
        popPrimaryCtxIfPushed(didPushCtx, tritonCaptureDevice);
        restoreCublasWorkspaceAfterCapture(stream);
        cleanupCaptureTlsState(true, prevCaptureStream);
        restoreSlotStates(slots_, seg.def.startSlot, seg.def.endSlot, savedSlotPhasesTriton);
      } else {
      // Allocate pinned host workspace for H2D source copies during capture.
      // MUST be done BEFORE beginCapture — cudaMallocHost is NOT capture-safe
      // and will invalidate the capture context if called during stream capture.
      void* monolithicCaptureHostWs = nullptr;
      {
        auto hostWsErr = cudaMallocHost(&monolithicCaptureHostWs, TRITON_CAPTURE_HOST_WORKSPACE_SIZE);
        if (hostWsErr != cudaSuccess) {
          cudaGetLastError();
          monolithicCaptureHostWs = nullptr;
          DSP_DIAG(MEMORY, "MONOLITHIC: cudaMallocHost for capture host workspace failed (%zuMB) — "
                   "H2D copies during capture will use non-pinned buffers",
                   TRITON_CAPTURE_HOST_WORKSPACE_SIZE / (1024*1024));
        }
      }

      bool nativeOnlyCapture = ctx.nativeOnlyGraphCapture;
      const bool originallyNativeOnly = nativeOnlyCapture;  // track if we started as native-only
      bool captureOk = nativeOnlyCapture ||
                       handle->beginCapture(ctx.cudaStr, cudaStreamCaptureModeThreadLocal);
      if (captureOk) {
        CaptureLifecycleGuard capGuard;
        if (nativeOnlyCapture) {
          DSP_DIAG_SEG(EXECUTE, seg.def.startSlot,
                       "NATIVE_ONLY_CAPTURE: seg[%d-%d] has no Triton islands — "
                       "skipping empty Triton capture and capturing native ops directly",
                       seg.def.startSlot, seg.def.endSlot);
          // Activate capture lifecycle: sets tl_graphExecutionActive=true so that
          // PointersManager uses the persistent pinned host workspace for H2D copies.
          // Without this, gather/concat ops copy from temp stack buffers that are freed
          // after the op completes. The CUDA graph's H2D memcpy nodes bake the source
          // address — reading freed memory on replay causes error 700.
          capGuard.activate();
          tl_graphCaptureStream = ctx.cudaStr;
        } else {
          DSP_DIAG_SEG(EXECUTE, seg.def.startSlot, "Triton graph capture started for seg[%d-%d] execCount=%d",
                       seg.def.startSlot, seg.def.endSlot, seg.exec.executionCount);
          capGuard.activate();
          tl_graphCaptureStream = ctx.cudaStr;
        }

        // ── Set up capture workspace for monolithic path ─────────────────────
        // The composite capture path (line ~2891) sets these TLS variables so that
        // DataBuffer::allocateSpecial() and CudaMemoryPool::allocate() use the
        // pre-allocated capture workspace instead of cudaMallocAsync (which creates
        // MemAlloc graph nodes that corrupt the capture). The monolithic path was
        // missing this setup, causing CudaMemoryPool::allocate to throw
        // "called during CUDA graph capture but NO capture workspace is set".
        tl_captureWorkspace = seg.exec.replayHandle->getWorkspacePtr();
        tl_captureWorkspaceSize = seg.exec.replayHandle->getWorkspaceBytes();
        tl_captureWorkspaceOffset = 0;
        tl_capturedHostPtrs.clear();
        tl_captureReplicateCache.clear();

        tl_captureHostWorkspace = monolithicCaptureHostWs;
        tl_captureHostWorkspaceSize = (monolithicCaptureHostWs != nullptr) ? TRITON_CAPTURE_HOST_WORKSPACE_SIZE : 0;
        tl_captureHostWorkspaceOffset = 0;
        if (monolithicCaptureHostWs != nullptr) {
          tl_capturedHostPtrs.push_back(monolithicCaptureHostWs);
        }

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

        // Query node count mid-capture to verify operations are being recorded.
        if (!nativeOnlyCapture) {
          size_t midCaptureNodes = handle->getNumNodesDuringCapture(ctx.cudaStr);
          DSP_DIAG(EXECUTE, "Triton capture mid-check: %zu nodes recorded before executeSegment",
                   midCaptureNodes);
        }

        // Snapshot all buffer addresses at capture entry — compare with replay to detect stale pointers
        {
          std::vector<void*> outAddrs, extAddrs;
          extractDeviceAddrs(outputSlots_, totalOutputSlots_, outAddrs);
          extractDeviceAddrs(externalArrays, numExt, extAddrs);
          DspDiagnostics::getInstance().clearAddressSnapshots();
          DSP_DIAG_SNAPSHOT_ADDRS("capture-entry", outAddrs.data(), totalOutputSlots_,
                                  extAddrs.data(), numExt);
        }

        auto captureStatus = nativeOnlyCapture
            ? Status::KERNEL_FAILURE
            : ctx.backend->executeSegment(seg, slots_, effectiveExternalsForCapture, numExt,
                                          outputSlots_, totalOutputSlots_, stream);

        if (!nativeOnlyCapture) {
          DSP_DIAG(EXECUTE, "MONOLITHIC_CAPTURE: executeSegment returned status=%d for seg[%d-%d]",
                   static_cast<int>(captureStatus), seg.def.startSlot, seg.def.endSlot);
        }

        // ── Zero-node detection: Triton backend returned OK but all ops were ──
        // gap-skipped during capture, producing 0 CUDA graph nodes. This happens
        // when compileSegment classified all ops as native-handled (gap ops) and
        // created a cache entry with no Triton kernels. During capture,
        // orderedRangeExecutor skips gap ops (streamIsCapturing && !mergedCapture),
        // so executeSegment returns OK without recording any work. Treat this the
        // same as KERNEL_FAILURE to fall through to native-only capture, which
        // executes ops via executeSlot() on the capture stream to actually record
        // cuBLAS/cuDNN/elementwise CUDA kernels into the graph.
        if (!nativeOnlyCapture && captureStatus == Status::OK) {
          size_t midNodes = 0;
          {
            cudaStreamCaptureStatus midStat = cudaStreamCaptureStatusNone;
            cudaGraph_t midGraph = nullptr;
            unsigned long long midId = 0;
            auto midErr = cudaStreamGetCaptureInfo_v2(ctx.cudaStr, &midStat, &midId, &midGraph, nullptr, nullptr);
            if (midErr == cudaSuccess && midGraph != nullptr) {
              cudaGraphGetNodes(midGraph, nullptr, &midNodes);
            }
          }
          if (midNodes == 0) {
            DSP_DIAG(EXECUTE, "MONOLITHIC_CAPTURE: executeSegment returned OK but 0 CUDA graph nodes "
                     "for seg[%d-%d] — all ops were gap-skipped during capture. "
                     "Falling through to native-only capture.",
                     seg.def.startSlot, seg.def.endSlot);
            captureStatus = Status::KERNEL_FAILURE;
            // Mark as native-only so the post-capture path at line ~4781 sets
            // compiledByBackend="CUDA" (not "Triton GPU") and clears the composite
            // schedule. Without this, the frozen fast path sees compiledByBackend=
            // "Triton GPU" and routes to composite replay which re-executes all
            // 2419 slots slot-by-slot instead of using cudaGraphLaunch.
            nativeOnlyCapture = true;
          }
        }

        // ── NATIVE-ONLY GRAPH CAPTURE ─────────────────────────────────────────
        // When executeSegment returns KERNEL_FAILURE (no compiled Triton segment —
        // all ops are native/gap), capture native GPU ops directly via executeSlot
        // with the gap stream redirected to the capture stream.  This records
        // cuBLAS matmuls, cuDNN attention, elementwise CUDA kernels, etc. into the
        // CUDA graph.  The same mechanism is used by merged gap capture in the
        // composite path above (tl_dspGapStream + SyncOverride + executeSlot).
        //
        // After successful capture, the monolithic replay path at
        // platformTryFrozenFastPath uses a single cudaGraphLaunch() per step
        // instead of hundreds of individual kernel launches via executeSlot().
        if (nativeOnlyCapture || captureStatus == Status::KERNEL_FAILURE) {
          DSP_DIAG(EXECUTE, "NATIVE_ONLY_CAPTURE: seg[%d-%d] (%d slots) — no compiled Triton segment, "
                   "capturing native ops via executeSlot on capture stream",
                   seg.def.startSlot, seg.def.endSlot,
                   seg.def.endSlot - seg.def.startSlot + 1);

          // ── End the failed Triton capture before native-only re-capture ──
          // executeSegment returned KERNEL_FAILURE (no Triton-compilable ops),
          // which means the current capture is invalid/empty.  End it so we can
          // execute view/identity ops outside any capture scope, then start a
          // fresh capture for just the compute ops.
          if (!originallyNativeOnly) {
            capGuard.deactivate();
            tl_graphCaptureStream = nullptr;
            if (handle->isCapturing()) {
              handle->endCapture(ctx.cudaStr);  // discard the invalid/empty graph
              DSP_DIAG(EXECUTE, "NATIVE_ONLY_CAPTURE: ended %s capture before re-capture",
                       (captureStatus == Status::OK) ? "zero-node" : "failed Triton");
            }
            cudaGetLastError();  // clear any sticky error from the failed capture
          }

          // ── Pre-capture: execute view/identity ops OUTSIDE capture scope ──
          // View-capable and identity ops (reshape, permute, squeeze, etc.)
          // only manipulate NDArray metadata (shape pointers, offsets, strides)
          // and never launch GPU kernels, so they record 0 graph nodes.  Their
          // view-install path (writeOutputSlot → DataBuffer swaps → potential
          // syncToSpecial) makes CUDA API calls that are NOT capture-safe and
          // will poison the capture stream (cudaStreamCaptureStatusInvalidated).
          //
          // Execute them here before the fresh capture so downstream compute ops
          // see correct output slot pointers during capture.
          int viewPreExec = 0;
          for (int s = seg.def.startSlot; s <= seg.def.endSlot; s++) {
            if (slotHasOnlyTransparentAliasOutputs(
                    slots_[s], slotOwnership_, outputSlots_, effectiveExternalsForCapture,
                    numExt, totalOutputSlots_) &&
                slots_[s].wiring.numInputs >= 1) {
              auto viewStatus = executeSlot(s, effectiveExternalsForCapture, numExt, stream);
              if (viewStatus != Status::OK) {
                DSP_DIAG(EXECUTE, "NATIVE_ONLY_CAPTURE: pre-capture view slot %d (%s) FAILED status=%d",
                         s, slots_[s].ident.opName.c_str(), static_cast<int>(viewStatus));
              }
              viewPreExec++;
            }
          }
          if (viewPreExec > 0) {
            DSP_DIAG(EXECUTE, "NATIVE_ONLY_CAPTURE: pre-executed %d view/identity slots outside capture scope",
                     viewPreExec);
          }

          // ── Start fresh capture for compute-only ops ──
          // Reset capture workspace offset for the fresh capture — the first
          // (failed Triton) capture may have bumped it, and the new capture needs
          // to start from the beginning of the workspace.
          tl_captureWorkspaceOffset = 0;
          // Free host ptrs from the failed first capture before clearing
          for (void* hp : tl_capturedHostPtrs) {
            if (hp != nullptr) cudaFreeHost(hp);
          }
          tl_capturedHostPtrs.clear();
          tl_captureReplicateCache.clear();

          // Allocate fresh pinned host workspace for the re-capture
          {
            void* recaptureHostWs = nullptr;
            auto hostWsErr = cudaMallocHost(&recaptureHostWs, TRITON_CAPTURE_HOST_WORKSPACE_SIZE);
            if (hostWsErr != cudaSuccess) {
              cudaGetLastError();
              recaptureHostWs = nullptr;
            }
            tl_captureHostWorkspace = recaptureHostWs;
            tl_captureHostWorkspaceSize = (recaptureHostWs != nullptr) ? TRITON_CAPTURE_HOST_WORKSPACE_SIZE : 0;
            tl_captureHostWorkspaceOffset = 0;
            if (recaptureHostWs != nullptr) {
              tl_capturedHostPtrs.push_back(recaptureHostWs);
            }
          }

          int slotsExecuted = 0;
          int frozenSkipped = 0;
          int viewSkipped = 0;
          bool nativeRecaptureOk = handle->beginCapture(ctx.cudaStr, cudaStreamCaptureModeThreadLocal);
          if (!nativeRecaptureOk) {
            DSP_DIAG(EXECUTE, "NATIVE_ONLY_CAPTURE: beginCapture for native re-capture FAILED");
            captureStatus = Status::KERNEL_FAILURE;
          } else {
            capGuard.activate();
            tl_graphCaptureStream = ctx.cudaStr;

          // Exception-safe gap-stream routing during native-only capture slots.
          ScopedGapStreamOverride gapStreamOverride(ctx.cudaStr);

          captureStatus = Status::OK;
          {
            SyncOverride capSync(*this, "native_only_capture");
            for (int s = seg.def.startSlot; s <= seg.def.endSlot; s++) {
              if (slots_[s].frozenConstantSlot()) {
                bool allOutputsPopulated = true;
                for (int o = 0; o < slots_[s].wiring.numOutputs; o++) {
                  int si = slots_[s].wiring.outputSlotIndices[o];
                  if (si >= 0 && si < totalOutputSlots_ && outputSlots_[si] == nullptr) {
                    allOutputsPopulated = false;
                    break;
                  }
                }
                if (allOutputsPopulated) {
                  frozenSkipped++;
                  continue;
                }
              }
              // Skip view/identity ops — already executed pre-capture above.
              // These ops produce 0 GPU graph nodes and their view-install path
              // poisons CUDA graph capture.
              if (slotHasOnlyTransparentAliasOutputs(
                      slots_[s], slotOwnership_, outputSlots_, effectiveExternalsForCapture,
                      numExt, totalOutputSlots_) &&
                  slots_[s].wiring.numInputs >= 1) {
                viewSkipped++;
                continue;
              }
              auto slotStatus = executeSlot(s, effectiveExternalsForCapture, numExt, stream);
              slotsExecuted++;
              if (slotStatus != Status::OK) {
                DSP_DIAG(EXECUTE, "NATIVE_ONLY_CAPTURE: slot %d (%s) FAILED status=%d after %d slots",
                         s, slots_[s].ident.opName.c_str(), static_cast<int>(slotStatus), slotsExecuted);
                captureStatus = slotStatus;
                break;
              }
              {
                bool _slotInvalid = false;
                DSP_CAPTURE_PROBE(ctx.cudaStr, s, "AFTER_NATIVE_SLOT",
                                  slots_[s].ident.opName.c_str(), _slotInvalid);
                if (_slotInvalid) {
                  captureStatus = Status::KERNEL_FAILURE;
                  break;
                }
              }
            }
          }
          if (frozenSkipped > 0 || viewSkipped > 0) {
            DSP_DIAG(EXECUTE, "NATIVE_ONLY_CAPTURE: skipped %d frozen constant + %d view/identity slots (of %d total)",
                     frozenSkipped, viewSkipped, seg.def.endSlot - seg.def.startSlot + 1);
          }
          } // end nativeRecaptureOk

          // Check how many nodes are in the graph now
          size_t nativeNodes = 0;
          {
            cudaStreamCaptureStatus ncStat = cudaStreamCaptureStatusNone;
            cudaGraph_t ncGraph = nullptr;
            unsigned long long ncId = 0;
            auto ncErr = cudaStreamGetCaptureInfo_v2(ctx.cudaStr, &ncStat, &ncId, &ncGraph, nullptr, nullptr);
            if (ncErr == cudaSuccess && ncGraph != nullptr) {
              cudaGraphGetNodes(ncGraph, nullptr, &nativeNodes);
            }
          }

          DSP_DIAG(EXECUTE, "NATIVE_ONLY_CAPTURE: seg[%d-%d] done — %d slots executed, %zu graph nodes, status=%d",
                   seg.def.startSlot, seg.def.endSlot, slotsExecuted, nativeNodes, static_cast<int>(captureStatus));
        }

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
          if (handle->isCapturing()) {
            handle->endCapture(ctx.cudaStr);
          }
        }

        DSP_DIAG(EXECUTE, "MONOLITHIC_CAPTURE: endCapture endOk=%d for seg[%d-%d]",
                 endOk ? 1 : 0, seg.def.startSlot, seg.def.endSlot);

        if (endOk) {
          size_t numGraphNodes = handle->getNumNodes();
          int segSize = seg.def.endSlot - seg.def.startSlot + 1;

          DSP_DIAG_SEG(COMPILE, seg.def.startSlot,
                       "GRAPH CAPTURE COMPLETE: seg[%d-%d] %zu nodes captured from %d slots (%.1f nodes/slot)",
                       seg.def.startSlot, seg.def.endSlot, numGraphNodes, segSize,
                       segSize > 0 ? (double)numGraphNodes / segSize : 0.0);
          DSP_DIAG(EXECUTE, "Triton capture endOk: graph has %zu nodes", numGraphNodes);

          // 0-node graphs mean every slot was a frozen constant (skipped during
          // capture). These segments should have been caught by allFrozenConstants
          // in buildSegments() and skipped before reaching capture. If we still end
          // up here, reject the capture — replaying a 0-node graph is a no-op that
          // wastes time and blocks real execution.
          if (numGraphNodes == 0) {
            DSP_DIAG_SEG(COMPILE, seg.def.startSlot,
                         "ZERO_NODE_REJECT: seg[%d-%d] captured 0 nodes from %d slots — "
                         "rejecting (will execute slot-by-slot, no re-capture)",
                         seg.def.startSlot, seg.def.endSlot, segSize);
            // Discard the 0-node graph and mark the segment so we don't re-attempt
            // capture every step. These ops produce no GPU kernels — slot-by-slot
            // execution is correct and avoids the begin/end capture overhead.
            seg.exec.replayHandle.reset();
            SegmentLifecycle::markZeroKernel(seg.exec, "zero_node_graph", seg.def.startSlot, seg.def.endSlot);
            popPrimaryCtxIfPushed(didPushCtx, tritonCaptureDevice);
            restoreCublasWorkspaceAfterCapture(stream);
            cleanupCaptureTlsState(true, prevCaptureStream);
            restoreSlotStates(slots_, seg.def.startSlot, seg.def.endSlot, savedSlotPhasesTriton);
#if HAVE_TRITON
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
          std::string graphDebugPath;
          {
            const std::string& dumpDir = Environment::getInstance().tritonDumpDir();
            if (!dumpDir.empty()) {
              graphDebugPath = dumpDir;
              if (graphDebugPath.back() != '/' && graphDebugPath.back() != '\\') graphDebugPath += '/';
            } else {
              const char* tmpEnv = std::getenv("TMPDIR");
              if (!tmpEnv) tmpEnv = std::getenv("TMP");
              if (!tmpEnv) tmpEnv = std::getenv("TEMP");
              graphDebugPath = tmpEnv ? std::string(tmpEnv) + "/" : "/tmp/";
            }
            graphDebugPath += "triton_graph_debug.dot";
          }
          cudaGraphDebugDotPrint(handle->getGraph(), graphDebugPath.c_str(), 0);
          DSP_DIAG(EXECUTE, "Triton graph dumped to %s", graphDebugPath.c_str());
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
          // Also treat any instantiation error under memory pressure as retryable:
          // concurrent graph captures can starve each other, producing non-OOM errors
          // (e.g. cudaErrorInvalidValue) that resolve once memory is freed.
          auto* cudaReplayForOom = seg.exec.replayHandle
                                   ? dynamic_cast<CudaGraphReplayHandle*>(seg.exec.replayHandle.get()) : nullptr;
          bool isOom = cudaReplayForOom && cudaReplayForOom->wasLastInstantiateOom();
          // If not a classic OOM error but memory is critically low, treat as retryable
          if (!isOom && cudaReplayForOom && !endOk) {
            // endCapture failed — not an instantiation error, skip OOM heuristic
          } else if (!isOom && cudaReplayForOom) {
            size_t gpuFreeCheck = 0, gpuTotalCheck = 0;
            cudaMemGetInfo(&gpuFreeCheck, &gpuTotalCheck);
            // Under 512MB free: any instantiation failure is likely memory-pressure-related
            if (gpuFreeCheck < 512ULL * 1024 * 1024) {
              int actualErr = cudaReplayForOom->getLastInstantiateError();
              DSP_DIAG_SEG(MEMORY, seg.def.startSlot,
                           "INSTANTIATE FAIL err=%d under memory pressure (gpuFree=%zuMB) — "
                           "treating as retryable OOM",
                           actualErr, gpuFreeCheck / (1024*1024));
              isOom = true;
            }
          }
          if (isOom && seg.exec.captureOomRetries < GraphSegment::maxOomRetries()) {
            int retryAfter = seg.exec.executionCount + GraphSegment::retryInterval();
            SegmentLifecycle::markOomDeferred(seg.exec, retryAfter);
            DSP_DIAG_SEG(MEMORY, seg.def.startSlot,
                         "INSTANTIATE OOM — retry %d/%d, evicting LRU graphs. retryAfterExec=%d",
                         seg.exec.captureOomRetries, GraphSegment::maxOomRetries(),
                         seg.exec.captureRetryAfterExec);

            // Evict LRU graphs to free memory for the next attempt
            evictLruGraphs(ctx.segIdx, tritonCaptureWorkspaceSize(), stream);

            // Cleanup this failed attempt but do NOT set compilationFailed
            abortCapture(seg, true, didPushCtx, tritonCaptureDevice,
                        prevCaptureStream, savedSlotPhasesTriton, stream);
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
              DSP_DIAG_SEG(COMPILE, seg.def.startSlot,
                            "NativeDSP: graph instantiation OOM for seg[%d-%d] on device %d "
                            "(retry %d/%d, retryAfterExec=%d). Evicted LRU graphs. "
                            "Returning KERNEL_FAILURE to caller — memory-budget segmentation "
                            "should prevent this.",
                            seg.def.startSlot, seg.def.endSlot, deviceId,
                            seg.exec.captureOomRetries, GraphSegment::maxOomRetries(),
                            seg.exec.captureRetryAfterExec);
              return Status::KERNEL_FAILURE;
            }
          }

          // Not OOM or retries exhausted — permanent failure
          // Use the actual error stored by the handle, not cudaGetLastError() which
          // was already cleared at line 4342 before instantiation.
          int actualInstErr = cudaReplayForOom ? cudaReplayForOom->getLastInstantiateError() : 0;
          cudaError_t reportErr = actualInstErr != 0
                                  ? static_cast<cudaError_t>(actualInstErr)
                                  : cudaGetLastError();
          abortCapture(seg, true, didPushCtx, tritonCaptureDevice,
                      prevCaptureStream, savedSlotPhasesTriton, stream);
#if HAVE_TRITON
          // Deactivate guard: abortCapture cleaned up; destructor must not double-clear.
          tritonOrderedRangeGuard.active = false;
          TritonGraphBackend::clearOrderedRangeExecutor();
#endif
          return reportCaptureError(this, seg, "instantiate", reportErr, deviceId);
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
                        prevCaptureStream, savedSlotPhasesTriton, stream);
#if HAVE_TRITON
            // Deactivate guard: abortCapture cleaned up; destructor must not double-clear.
            tritonOrderedRangeGuard.active = false;
            TritonGraphBackend::clearOrderedRangeExecutor();
#endif
            return reportReplayError(seg, "validation_launch", cudaGetLastError(), deviceId);
          }
	          DSP_DIAG(EXECUTE, "VALIDATION LAUNCH QUEUED: seg[%d-%d] async graph launch accepted",
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
          // Guard: only transition lifecycle if the segment is not already SEALED.
          // Plan cache can return a plan whose segment was previously captured+sealed;
          // re-capturing updates handles but must not re-fire the lifecycle transition.
          if (!seg.exec.segPhase.isSealed()) {
            const char* captureBackendName = nativeOnlyCapture ? "CUDA" : ctx.backendName;
            SegmentLifecycle::markCaptured(seg.exec, ctx.segInputAddrKey, ctx.createValueKey,
                computeSlotAddrHash(outputSlots_, seg.def.startSlot, seg.def.endSlot, totalOutputSlots_),
                captureBackendName, nativeOnlyCapture);
          } else {
            // Already sealed — just update the replay keys without lifecycle transition.
            seg.exec.sealCapture(ctx.segInputAddrKey, ctx.createValueKey,
                computeSlotAddrHash(outputSlots_, seg.def.startSlot, seg.def.endSlot, totalOutputSlots_),
                nativeOnlyCapture ? "CUDA" : ctx.backendName, nativeOnlyCapture);
            DSP_DIAG(EXECUTE, "MONOLITHIC_CAPTURE: seg[%d-%d] already SEALED — updated replay keys without lifecycle transition",
                     seg.def.startSlot, seg.def.endSlot);
          }
          // Force arg-table refresh on the first post-capture replay.
          // Same rationale as the composite-capture path above: markCaptured() does
          // not call markArgsCurrent(), but the shared CompiledKernel singleton may
          // already have cachedArgTableHostPinned populated from a prior plan instance.
          // bumpArgGeneration() guarantees needsArgRefresh()=true so
          // refreshArgTablesForReplay() always runs once with this plan's own
          // effectiveExternals before the first CUDA graph launch.
          seg.exec.bumpArgGeneration();

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
            std::string dotDir;
            {
              const std::string& dd = Environment::getInstance().tritonDumpDir();
              if (!dd.empty()) {
                dotDir = dd;
                if (dotDir.back() != '/' && dotDir.back() != '\\') dotDir += '/';
              } else {
                const char* tmpEnv = std::getenv("TMPDIR");
                if (!tmpEnv) tmpEnv = std::getenv("TMP");
                if (!tmpEnv) tmpEnv = std::getenv("TEMP");
                dotDir = tmpEnv ? std::string(tmpEnv) + "/" : "/tmp/";
              }
            }
            std::string dotPath = dotDir + "triton_graph_captured.dot";
            unsigned int dotFlags = Environment::getInstance().tritonGraphDotVerbose()
                                    ? cudaGraphDebugDotFlagsVerbose : 0;
            auto dotErr = cudaGraphDebugDotPrint(handle->getGraph(), dotPath.c_str(), dotFlags);
            if (dotErr == cudaSuccess) {
              DSP_DIAG(EXECUTE, "Exported Triton graph DOT to %s (verbose=%d)",
                       dotPath.c_str(), dotFlags != 0);
            }
            cudaGetLastError(); // Clear any error from dot print
          }
          // Write stats to a file for diagnostic inspection
          if (sd::Environment::getInstance().isVerbose()) {
            std::string statsDir;
            {
              const std::string& dd = Environment::getInstance().tritonDumpDir();
              if (!dd.empty()) {
                statsDir = dd;
                if (statsDir.back() != '/' && statsDir.back() != '\\') statsDir += '/';
              } else {
                const char* tmpEnv = std::getenv("TMPDIR");
                if (!tmpEnv) tmpEnv = std::getenv("TMP");
                if (!tmpEnv) tmpEnv = std::getenv("TEMP");
                statsDir = tmpEnv ? std::string(tmpEnv) + "/" : "/tmp/";
              }
            }
            std::string statsPath = statsDir + "triton_graph_stats.txt";
            FILE* f = fopen(statsPath.c_str(), "w");
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

          // Native-only captures contain all gap/native work in the monolithic
          // CUDA graph. Do not tag them as Triton or build a gap-only composite
          // schedule; that would route replay back through live slot execution.
          if (nativeOnlyCapture) {
            seg.exec.compiledByBackend = "CUDA";
            seg.exec.compositeReplaySchedule = ReplaySchedule();
            DSP_DIAG(SHAPE, "NATIVE_ONLY_CAPTURE: seg[%d-%d] kept as CUDA monolithic graph "
                     "(no Triton composite schedule)",
                     seg.def.startSlot, seg.def.endSlot);
          } else {
            // CRITICAL: Mark this segment as Triton-compiled so subsequent replays
            // use the Triton replay path (arg table refresh, D2D copies) instead of
            // raw CUDA graph replay. Without this, isTritonCompiled=false on replay
            // and the code falls through to raw CUDA graph replay which doesn't invoke
            // Triton kernels -> tritonKernelLaunchCount stays at 0.
            seg.exec.compiledByBackend = ctx.backendName;
            // Build composite schedule now that Triton gap data is accurate.
#if HAVE_TRITON
            if (seg.exec.compositeReplaySchedule.units.empty() && ctx.tritonBackend != nullptr) {
              seg.exec.compositeReplaySchedule = buildCompositeReplaySchedule(seg, slots_, ctx.tritonBackend);
              DSP_DIAG(SHAPE, "COMPOSITE_SCHEDULE_BUILT: seg[%d-%d] units=%d compiledBy=%s (capture path)",
                       seg.def.startSlot, seg.def.endSlot,
                       static_cast<int>(seg.exec.compositeReplaySchedule.units.size()),
                       seg.exec.compiledByBackend.c_str());
            }
#endif
          }

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
                      prevCaptureStream, savedSlotPhasesTriton, stream);
#if HAVE_TRITON
          // Deactivate guard: abortCapture cleaned up; destructor must not double-clear.
          tritonOrderedRangeGuard.active = false;
          TritonGraphBackend::clearOrderedRangeExecutor();
#endif
          return reportCaptureError(this, seg, "execute_during_capture", cudaGetLastError(), deviceId);
        }
      } else {
        // beginCapture failed — free the pre-allocated host workspace
        if (monolithicCaptureHostWs != nullptr) {
          cudaFreeHost(monolithicCaptureHostWs);
          monolithicCaptureHostWs = nullptr;
        }
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
          evictLruGraphs(ctx.segIdx, tritonCaptureWorkspaceSize(), stream);
          abortCapture(seg, true, didPushCtx, tritonCaptureDevice,
                      prevCaptureStream, savedSlotPhasesTriton, stream);
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
                    prevCaptureStream, savedSlotPhasesTriton, stream);
#if HAVE_TRITON
        // Deactivate guard: abortCapture cleaned up; destructor must not double-clear.
        tritonOrderedRangeGuard.active = false;
        TritonGraphBackend::clearOrderedRangeExecutor();
#endif
        return reportCaptureError(this, seg, "beginCapture", beginErr, deviceId);
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

      restoreSlotStates(slots_, seg.def.startSlot, seg.def.endSlot, savedSlotPhasesTriton);
    }  // end else (monolithicCaptureGuard.acquired())

    }  // end else (replayHandle != nullptr — workspace allocation succeeded)

    }  // end else (replayHandle null check — monolithic capture body)

  }  // end if (!didCompositeCapture) — monolithic capture only for non-composite segments

  }  // end if (shouldCaptureTritonGraphNow)

  if (!usedTritonGraphCapture) {
    // Cross-stream + H2D + staging sync: already done by performPreReplaySync
    // in dispatchSegment (tracked via PreReplaySyncPhase). No redundant sync here.

    // NOTE: We intentionally do NOT set cuBLAS workspace for direct/warmup execution.
    // Warmup runs with workspace=0, which causes cuBLAS to select algorithms that
    // don't require workspace. These algorithms are cached in tl_ltAlgoCache.
    // At capture time, setCublasWorkspaceForCapture() provides workspace for
    // capturability, but tryLtMatmul() hits the algo cache and reuses the warmup
    // algorithm — ensuring capture bakes in the SAME algorithm as warmup/live.
    // This makes merged CUDA graph replay numerically identical to live execution.

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
    std::vector<SlotPhase> savedSlotPhasesNonCapture;
    demoteFrozenSlotStates(slots_, seg.def.startSlot, seg.def.endSlot, savedSlotPhasesNonCapture);

    // Snapshot addresses for direct execution (baseline for comparison with capture/replay)
    snapshotAddrs(outputSlots_, totalOutputSlots_, externalArrays, numExt, "direct-entry");

    // Pre-execution sync: H2D variable external inputs + cross-stream ordering.
    // The frozen fast-path (forcesSyncOnFrozen=false for TRITON/AUTO) skips
    // prepareSpecialUse after execCount>=2, so variable placeholder inputs like
    // "x" have stale device buffers. performPreReplaySync forces H2D for inputs
    // classified as variable, regardless of frozen state. Without this, Triton
    // direct execution reads stale capture-time device data every step.
    NDArray** directExternals = performPreReplaySync(
        externalArrays, numExt, stream, "tritonDirectExec");

    DSP_DIAG_SEG(MEMORY, seg.def.startSlot,
                 "direct-exec invoking prezeroSegmentOutputs seg=[%d-%d] stream=%p execCount=%d",
                 seg.def.startSlot, seg.def.endSlot, (void*)stream, seg.exec.executionCount);
    prezeroSegmentOutputs(seg, stream);

    try {
      status = ctx.backend->executeSegment(seg, slots_, directExternals, numExt,
                                       outputSlots_, totalOutputSlots_, stream);
    } catch (...) {
      restoreSlotStates(slots_, seg.def.startSlot, seg.def.endSlot, savedSlotPhasesNonCapture);
      throw;
    }

    // When the Triton backend returns KERNEL_FAILURE (no compiled segment —
    // e.g. vision encoder with ops Triton can't handle, or after composite
    // capture failure cleared the schedule), fall through to native slot-by-slot
    // execution.  This is the non-capture equivalent of the NATIVE_ONLY_CAPTURE
    // path at lines ~3519-3585.
    if (status == Status::KERNEL_FAILURE) {
      DSP_DIAG(EXECUTE, "NATIVE_DIRECT_EXEC: seg[%d-%d] (%d slots) — Triton backend returned "
               "KERNEL_FAILURE, executing all slots natively via executeSlot",
               seg.def.startSlot, seg.def.endSlot,
               seg.def.endSlot - seg.def.startSlot + 1);

      // Exception-safe gap-stream routing during native direct fallback slots.
      ScopedGapStreamOverride gapStreamOverride(ctx.cudaStr);

      status = Status::OK;
      {
        SyncOverride directSync(*this, "native_direct_exec");
        for (int s = seg.def.startSlot; s <= seg.def.endSlot; s++) {
          auto slotStatus = executeSlot(s, directExternals, numExt, stream);
          if (slotStatus != Status::OK) {
            DSP_DIAG(EXECUTE, "NATIVE_DIRECT_EXEC: slot %d (%s) FAILED status=%d",
                     s, slots_[s].ident.opName.c_str(), static_cast<int>(slotStatus));
            status = slotStatus;
            break;
          }
        }
      }
    }

    restoreSlotStates(slots_, seg.def.startSlot, seg.def.endSlot, savedSlotPhasesNonCapture);

    // Mark output slot device buffers as current after Triton kernel writes.
    // The Triton kernel writes directly to device memory via specialBuffer(),
    // but does not update NDArray actuality flags. Without tickWriteDevice(),
    // subsequent Java-side .dup() reads from stale host data (isPrimaryActual
    // was true from prior H2D sync), producing identical outputs every step.
    if (status == Status::OK) {
      for (int si = seg.def.startSlot; si <= seg.def.endSlot && si < totalOutputSlots_; si++) {
        if (outputSlots_[si] != nullptr && outputSlots_[si]->dataBuffer() != nullptr) {
          outputSlots_[si]->tickWriteDevice();
        }
      }
    }

    // DSP_DIAG-gated diagnostics for Triton direct execution path.
    // All output verification is behind DSP_DIAG_ENABLED checks to avoid
    // unconditional host-side value dumps on every step.
    if (DSP_DIAG_ENABLED(EXECUTE)) {
      // Dump final output for direct Triton path (baseline comparison)
      if (status == Status::OK && seg.def.endSlot < totalOutputSlots_ &&
          outputSlots_[seg.def.endSlot] != nullptr) {
        auto* finalOut = outputSlots_[seg.def.endSlot];
        if (finalOut->dataType() == FLOAT32) {
          DSP_DIAG_DUMP_SLOT("direct", seg.def.endSlot,
                             DSP_BUF(finalOut), finalOut->lengthOf());
        }
      }
      // Segment exit argmax
      if (status == Status::OK) {
        dumpSegFinalArgmax(seg, outputSlots_, totalOutputSlots_, numSlots_, slots_,
                           ctx.cudaStr, "SEG_EXIT_ARGMAX", seg.exec.executionCount);
      }
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

#if HAVE_TRITON
      // segDispatchCompile can set compiledByBackend before the first direct
      // execution. Build the composite schedule whenever it is missing, not only
      // on the transition from an empty compiledByBackend. Without the schedule,
      // composite capture has no island units and the segment can remain
      // BUILDING:CAPTURING with no replay handles.
      if (ctx.tritonBackend != nullptr &&
          seg.exec.compiledByBackend == ctx.backendName &&
          seg.exec.compositeReplaySchedule.units.empty()) {
        seg.exec.compositeReplaySchedule = buildCompositeReplaySchedule(seg, slots_, ctx.tritonBackend);
        DSP_DIAG(SHAPE, "COMPOSITE_SCHEDULE_BUILT: seg[%d-%d] units=%d compiledBy=%s (direct path)",
                 seg.def.startSlot, seg.def.endSlot,
                 static_cast<int>(seg.exec.compositeReplaySchedule.units.size()),
                 seg.exec.compiledByBackend.c_str());
      }
#endif
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

  // All-frozen-constant segments: outputs already populated from warmup.
  // Should have been caught earlier but defend here too.
  if (seg.def.allFrozenConstants) {
    seg.exec.executionCount++;
    DSP_DIAG_SEG(EXECUTE, seg.def.startSlot,
                 "FROZEN_CONST_SKIP_GPU: seg[%d-%d] all frozen constants — skipping graph path",
                 seg.def.startSlot, seg.def.endSlot);
    return Status::OK;
  }

  // Derive segIdx for proactive eviction and OOM retry.
  int segIdx = -1;
  for (int si = 0; si < static_cast<int>(segments_.size()); si++) {
    if (&segments_[si] == &seg) { segIdx = si; break; }
  }

  {
    const char* mode = seg.exec.segPhase.displayName();
    DSP_DIAG_SEG(SHAPE, seg.def.startSlot,
                 "executeSegmentWithGpuGraph: ENTER seg[%d-%d] phase=%s execCount=%d capturable=%d",
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
    const char* lifecycle = seg.exec.segPhase.displayName();
    DSP_DIAG(VERIFY, "SEG_ENTER seg[%d-%d] execCount=%d phase=%s",
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
  // handles staging internally: performPreReplaySync does H2D on originals first,
  // then D2D copy to staging buffers. Substituting here would pass staging pointers
  // which would skip H2D sync on the originals (staging buffers are device-
  // authoritative), leaving Java-side input data stranded on the host.

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
  if (seg.exec.segPhase.isFailed()) {
    DSP_DIAG(FALLBACK, "EXEC_SEG_FAILED_GATE: seg[%d-%d] phase=%s — returning KERNEL_FAILURE",
              seg.def.startSlot, seg.def.endSlot, seg.exec.segPhase.displayName());
    return Status::KERNEL_FAILURE;
  }

  // Safety check: caller (platformExecuteSegmentWithBackends) already gates on
  // canFuseSegment() before invoking this function. This secondary check defends
  // against direct callers (phaseCompile, precompile) that may bypass the outer gate.
  // Returns KERNEL_FAILURE so those callers can skip the segment without marking it
  // as permanently compilationFailed.
  if (!backend->canFuseSegment(slots_, seg.def.startSlot, seg.def.endSlot)) {
    DSP_DIAG(BACKEND, "executeSegmentWithGpuGraph: backend=%s cannot fuse seg[%d-%d] "
             "(should have been pre-checked by caller — reaching here is unexpected)",
             backendName, seg.def.startSlot, seg.def.endSlot);
    return Status::KERNEL_FAILURE;
  }

  // First execution: run slot-by-slot warmup BEFORE compilation.
  if (seg.exec.segPhase.needsWarmup()) {
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
  const bool hasInternalValueShapeInputs = dsp::segmentHasInternalValueShapeInputs(seg, slots_);
  LongType segShapeKey;
  if (!planLifecycle_.isSlotBySlot() && seg.exec.cachedShapeKey != 0) {
    segShapeKey = seg.exec.cachedShapeKey;
  } else {
    segShapeKey = computeSegmentShapeKey(seg, externalArrays, numExt);
    if (!planLifecycle_.isSlotBySlot()) {
      seg.exec.cachedShapeKey = segShapeKey;
    }
  }

  // Diagnostic: scan all outputSlots_ entries for freed DataBuffers.
  // During warmup, this runs unconditionally (handles invalidation + rebuild).
  // During frozen replay, this only runs when DSP_DIAG VERIFY is enabled —
  // stale buffers in frozen replay indicate a bug, not a recoverable state.
  bool runStaleBufferScan = planLifecycle_.isSlotBySlot() ||
                            DspDiagnostics::getInstance().isEnabled(DSP_DIAG_VERIFY);

  // View-producer slots that wrap a placeholder DataBuffer are legitimately
  // stale whenever SameDiff replaces the placeholder between calls (e.g.
  // EMULATED_REPLAY supplies a fresh external input every step). Refresh
  // those wrappers in place on EVERY frozen replay — the gate below only
  // controls the expensive stale-buffer scan, but view-wrapper refresh must
  // always run or the slot's DataBuffer will dangle into slot-by-slot exec,
  // where writeOutputSlot's frozen-phase guard rejects the replacement as a
  // lifecycle violation.
  if (!planLifecycle_.isSlotBySlot()) {
    int viewRefreshResult =
        refreshStaleViewWrappersInSegment(seg, externalArrays, numExt);
    if (viewRefreshResult > 0) {
      // Fresh wrappers expose new device addresses — force argTable refresh on
      // the next replay. Graph remains valid; no recapture needed.
      seg.exec.bumpArgGeneration();
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
      // Never invalidate frozen constant outputs — their buffers were validated
      // at classification time and hold frozen refs. If metadata appears corrupted,
      // that's heap corruption from an adjacent allocation; nulling the slot causes
      // worse failures (Triton compile failure, KERNEL_FAILURE on execution).
      if (si < numSlots_ && slots_[si].frozenConstantSlot()) continue;
      NDArray* cached = outputSlots_[si];
      if (cached != nullptr && cached->hasValidShapeInfo() && !cached->isEmpty()) {
        auto* db = cached->dataBuffer();
        if (db == nullptr || !db->isValid()) {
          DSP_DIAG_SLOT(MEMORY, si, "STALE outputSlots_[%d] detected "
                                    "(arr=%p, db=%p, dbValid=%d, frozenConst=%d). Invalidating.",
                        si, (void*)cached, (void*)db, db ? (db->isValid() ? 1 : 0) : -1,
                        slots_[si].frozenConstantSlot() ? 1 : 0);
          outputSlots_[si] = nullptr;
          if (si < numSlots_ && slots_[si].slotPhase.isSealed() && slots_[si].slotPhase.isConstant) {
            slots_[si].slotPhase.isConstant = false;  // PRIMARY: demote constant → sealed
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
      if (!planLifecycle_.isSlotBySlot() && seg.exec.executionCount > 1) {
        // After warmup with frozen shapes, stale buffers mean a bug in array lifecycle management
        REQUIRE_TRUE(false, 0, "Stale buffer detected after warmup (executionCount=%d, frozen=%d, "
                               "invalidCount=%d) in seg[%d-%d]. This indicates a bug in DSP array persistence.",
                     seg.exec.executionCount, (int)planLifecycle_.isShapesFrozen(), invalidCount,
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
  if (!(!planLifecycle_.isSlotBySlot() && seg.exec.replayHandle != nullptr)) {
    for (int stepIdx = seg.def.startSlot; stepIdx <= seg.def.endSlot; stepIdx++) {
      NativeSlot& slot = slots_[stepIdx];
      // Validate input DataBuffers — Java close() may have freed them.
      for (int i = 0; i < slot.wiring.numInputs; i++) {
        int srcIdx = slot.wiring.inputSourceIndices[i];
        if (srcIdx >= 0 && srcIdx < totalOutputSlots_ && outputSlots_[srcIdx] != nullptr) {
          // Skip frozen constants — their buffers are protected by frozen refs
          if (srcIdx < numSlots_ && slots_[srcIdx].frozenConstantSlot()) continue;
          auto* db = outputSlots_[srcIdx]->dataBuffer();
          if (db == nullptr || !db->isValid()) {
            outputSlots_[srcIdx] = nullptr;
            if (srcIdx < numSlots_ && slots_[srcIdx].slotPhase.isSealed() && slots_[srcIdx].slotPhase.isConstant) {
              slots_[srcIdx].slotPhase.isConstant = false;  // PRIMARY: demote constant → sealed
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
          if (ts >= 0 && slotIdx == ts && !planLifecycle_.isSlotBySlot()) {
            auto* arr = outputSlots_[slotIdx];
            auto* db = arr != nullptr ? arr->dataBuffer() : nullptr;
            DSP_DIAG_SLOT(MEMORY, stepIdx,
                          "PRE_EXEC_VALIDATE: slot=%d arr=%p db=%p valid=%d exec=%d",
                          slotIdx, (void*)arr, (void*)db,
                          db != nullptr && db->isValid() ? 1 : 0,
                          seg.exec.executionCount);
          }
        }
        // Validate existing entry — skip frozen constants (protected by frozen refs)
        if (outputSlots_[slotIdx] != nullptr && !(slotIdx < numSlots_ && slots_[slotIdx].frozenConstantSlot())) {
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
            if (stepIdx < numSlots_ && slots_[stepIdx].slotPhase.isSealed() && slots_[stepIdx].slotPhase.isConstant) {
              slots_[stepIdx].slotPhase.isConstant = false;  // PRIMARY: demote constant → sealed
            }
          }
        }
        if (outputSlots_[slotIdx] == nullptr) {
          // After warmup with frozen shapes, null output slots indicate a persistence bug.
          // Frozen constant slots are exempt (they never allocate output arrays).
          // Warn but continue — the allocation path below will recover.
          if (!planLifecycle_.isSlotBySlot() && seg.exec.executionCount > 1 && !slot.frozenConstantSlot()) {
            DSP_DIAG_SLOT(VERIFY, slotIdx,
                          "BUG: Null output slot %d (%s) after warmup with frozen shapes — persistence bug. execCount=%d",
                          slotIdx, slot.ident.opName.c_str(), seg.exec.executionCount);
          }
          // Phase assertion: allocating a new NDArray during REPLAYING phase is a bug.
          // Output slots should already be populated from warmup/capture. New allocations
          // during replay mean the slot was freed or not persisted correctly.
          if (seg.exec.segPhase.isSealed() && !slot.frozenConstantSlot()) {
            DSP_DIAG(EXECUTE, "PHASE_VIOLATION: new NDArray allocation for slot %d (%s) during "
                              "SEALED phase — output should already exist from warmup. "
                              "seg[%d-%d] execCount=%d planPhase=%s",
                     slotIdx, slot.ident.opName.c_str(),
                     seg.def.startSlot, seg.def.endSlot, seg.exec.executionCount,
                     planLifecycle_.displayName());
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
  } // end if (!(planLifecycle_.isShapesFrozen() && replayHandle))

  // ── COMPILE DISPATCH ──────────────────────────────────────────────────────
  // segDispatchCompile handles: phase guard, shape-change mini-warmup,
  // backend->compileSegment(), markCompiled(), and first-compilation audit.
  // segShapeKey is passed by reference — shape-change recompile updates it.
  // Pre-exec output slot allocation above ensures all slots are populated
  // before the compiler resolves arg mappings.
  {
    if (seg.exec.segPhase.needsCompile()) {
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
    // NOTE: shapeKeyState.markCompiled(segShapeKey) is now called inside
    // segDispatchCompile, only when compilation actually occurs. Previously
    // it was unconditional here, which on exec2 with changed shapes (KV cache
    // growth) overwrote compiledShapeKey with a value that had no compiled
    // kernel in the Triton cache — causing KERNEL_FAILURE on lookup.
  }

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
  // Debug mode disables graph capture: debug tracing adds syncToHost calls
  // that cannot be captured and cause capture workspace OOM.
  bool allowTritonCudaGraphReplay = Environment::getInstance().tritonGraphCapture() &&
                                    !planLifecycle_.isSlotBySlot() &&
                                    !Environment::getInstance().tritonSkipKernels();

  int captureMinExec = Environment::getInstance().tritonCaptureMinExec();
  bool forceRecaptureEnabled = Environment::getInstance().tritonForceRecapture();
  // hasReplayHandle is true for both monolithic (READY handle) and composite
  // (EMPTY sentinel). In both cases it blocks new monolithic capture.
  bool hasReplayHandle = (seg.exec.replayHandle != nullptr);
  bool replayHandleNull = (seg.exec.replayHandle == nullptr);
  bool hasComposite = hasCompositeHandles(seg);  // true only for composite-captured segments
  bool notCaptureFailed = !seg.exec.compilationFailed;
  bool execCountInWindow = (seg.exec.executionCount >= captureMinExec);
  bool hasCudaStream = (cudaStr != nullptr);
  bool requiresOrderedGapCapture = false;

  DSP_DIAG(EXECUTE, "=== CAPTURE DECISION CHECK seg[%d-%d] ===", seg.def.startSlot, seg.def.endSlot);
  DSP_DIAG(EXECUTE, "  tritonGraphCapture()=%d, planFrozen=%d, tritonSkipKernels=%d => allowTritonCudaGraphReplay=%d",
           Environment::getInstance().tritonGraphCapture() ? 1 : 0,
           planLifecycle_.isShapesFrozen() ? 1 : 0,
           Environment::getInstance().tritonSkipKernels() ? 1 : 0,
           allowTritonCudaGraphReplay ? 1 : 0);
  DSP_DIAG(EXECUTE, "  seg.exec.executionCount=%d, captureMinExec=%d, window=[%d,inf), inWindow=%d",
           seg.exec.executionCount, captureMinExec, captureMinExec,
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
  bool allSlotsAreGaps = false;
#if HAVE_TRITON
  // Composite schedule is built at compiledByBackend-set-time (capture and
  // direct paths in segDispatchCaptureOrDirect).  All-gap detection is handled
  // by platformShouldUseGraph() — the single eligibility gate.
  if (tritonBackend != nullptr) {
   auto gapSlots = tritonBackend->getGapSlots(seg, slots_);
   tritonGapSlotCount = static_cast<int>(gapSlots.size());
   int totalSegSlots = seg.def.endSlot - seg.def.startSlot + 1;
   allSlotsAreGaps = (tritonGapSlotCount == totalSegSlots);
   if (!gapSlots.empty() && !allSlotsAreGaps) {
     requiresOrderedGapCapture = true;
   }
 }
#else
  seg.exec.compositeReplaySchedule = ReplaySchedule();
#endif

  // When ANY gap slots exist, use native-only monolithic capture. This captures
  // ALL ops (Triton islands + cuBLAS gaps) into a single CUDA graph via
  // executeSlot() on the capture stream. Phases stay linear: no composite
  // schedule, no demotion, no re-capture. The monolithic graph is complete.
  //
  // Previously, mixed segments used Triton capture (which skips gap ops) then
  // tried composite replay (islands from sub-graphs + gaps live). This required
  // demotion from monolithic→composite and caused error 700 crashes. Native-only
  // capture avoids all of this by recording everything in one graph.
  bool forceNativeCapture = false;
  if (tritonGapSlotCount > 0) {
    forceNativeCapture = true;
    DSP_DIAG(EXECUTE, "NATIVE_CAPTURE_FORCED: seg[%d-%d] has %d gap slots out of %d total — "
                      "capturing ALL ops natively (monolithic graph includes gaps)",
             seg.def.startSlot, seg.def.endSlot, tritonGapSlotCount,
             seg.def.endSlot - seg.def.startSlot + 1);
  }

  bool captureWindowSatisfied = execCountInWindow || requiresOrderedGapCapture;
  shouldCaptureTritonGraph = allowTritonCudaGraphReplay &&
                             !hasReplayHandle &&
                             replayHandleNull &&
                             notCaptureFailed &&
                             captureWindowSatisfied &&
                             hasCudaStream;

  if (requiresOrderedGapCapture) {
    DSP_DIAG(EXECUTE,
             "GAP_CAPTURE_MODE: seg[%d-%d] has %d gap slots — %s",
             seg.def.startSlot, seg.def.endSlot, tritonGapSlotCount,
             forceNativeCapture
               ? "native-only monolithic capture (gaps INCLUDED in CUDA graph)"
               : "ordered gap capture (gaps excluded, composite replay)");
  }

  DSP_DIAG(EXECUTE, "  => shouldCaptureTritonGraph=%d", shouldCaptureTritonGraph ? 1 : 0);
  if (!shouldCaptureTritonGraph) {
    if (!allowTritonCudaGraphReplay)
      DSP_DIAG(EXECUTE, "  BLOCKED: allowTritonCudaGraphReplay=false (tritonGraphCapture=%d OR planFrozen=%d OR tritonSkipKernels=%d)",
               Environment::getInstance().tritonGraphCapture() ? 1 : 0, planLifecycle_.isShapesFrozen() ? 1 : 0,
               Environment::getInstance().tritonSkipKernels() ? 1 : 0);
    if (!replayHandleNull)
      DSP_DIAG(EXECUTE, "  BLOCKED: replayHandle already exists (%s capture already done or in progress)",
               hasComposite ? "composite" : "monolithic");
    if (seg.exec.compilationFailed)
      DSP_DIAG(EXECUTE, "  BLOCKED: compilationFailed=true (previous capture failed, warmup path only)");
    if (!captureWindowSatisfied)
      DSP_DIAG(EXECUTE, "  BLOCKED: executionCount=%d below captureMinExec=%d",
               seg.exec.executionCount, captureMinExec);
    if (!hasCudaStream)
      DSP_DIAG(EXECUTE, "  BLOCKED: cudaStr=nullptr (no CUDA stream available)");
  } else {
    DSP_DIAG(EXECUTE, "  >>> CAPTURE WILL BE ATTEMPTED <<<");
  }
  DSP_DIAG(EXECUTE, "=== END CAPTURE DECISION CHECK ===");

  // NOTE: shouldCaptureTritonGraph is ONLY checked when we don't have a captured graph.
  // Once captured, we use useFastReplay based on generation counter, not executionCount.
  // The executionCount window check prevents repeated capture attempts after success.

  // OPTIMIZATION: When generation matches (no addr drift), skip the expensive
  // hash/comparison loops over all external inputs.
  LongType segInputAddrKey;
  bool extAddrsStable;
  LongType createValueKey;
  bool canSkipReplayInvariantRecompute =
      !seg.exec.needsArgRefresh() && allowTritonCudaGraphReplay &&
      !hasInternalValueShapeInputs;
  if (canSkipReplayInvariantRecompute) {
    // Fast path: arg table is stable, all addresses are known-good
    DSP_DIAG_SEG(EXECUTE, seg.def.startSlot,
                 "seg[%d-%d] needsArgRefresh()=false → FAST PATH (skip addr/createValue recompute)",
                 seg.def.startSlot, seg.def.endSlot);
    segInputAddrKey = seg.exec.capturedInputAddrKey;
    extAddrsStable = true;
    createValueKey = seg.exec.capturedCreateValueKey;
  } else {
    segInputAddrKey = computeSegmentInputAddrKey(seg, externalArrays, numExt);
    // Prefer the filtered/staged address key whenever it exists. Triton decode
    // intentionally stabilizes variable placeholder inputs through
    // ensureAndSyncStagingBuffers(); the raw external snapshot still sees the
    // fresh Java-side placeholder wrappers and will churn every step even
    // though the replayed graph is using stable staged pointers.
    extAddrsStable = (seg.exec.capturedInputAddrKey != 0)
                     ? (seg.exec.capturedInputAddrKey == segInputAddrKey)
                     : ((seg.exec.replayHandle && !seg.exec.replayHandle->getCapturedExternalAddresses().empty())
                        ? externalAddrsMatch(seg, externalArrays, numExt)
                        : false);
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
    // For composite captures, create-value stability doesn't affect arg table
    // because create ops execute live as gap slots (not baked into CUDA graphs).
    // Don't let create-value churn poison the arg table stability flag.
    const bool effectiveCreateValuesStable = createValuesStable || hasComposite;
    const bool wasStable = !seg.exec.needsArgRefresh();
    const bool nowStable = wasStable && extAddrsStable && effectiveCreateValuesStable && shapeKeyStable;
    // Keep generation counter in sync: bump if transitioning from stable to unstable
    if (wasStable && !nowStable) {
      seg.exec.bumpArgGeneration();
    } else if (!wasStable && nowStable) {
      seg.exec.markArgsCurrent();
    }
    DSP_DIAG_SEG(EXECUTE, seg.def.startSlot,
                 "INTERNAL_VALUE_SHAPE_TRACKING: seg[%d-%d] argStable=%d shapeStable=%d "
                 "createStable=%d extAddrsStable=%d",
                 seg.def.startSlot, seg.def.endSlot,
                 !seg.exec.needsArgRefresh() ? 1 : 0,
                 shapeKeyStable ? 1 : 0,
                 createValuesStable ? 1 : 0,
                 extAddrsStable ? 1 : 0);
  }
  // Create-value-key invalidation: only for MONOLITHIC captures where create ops
  // (ConstantOfShape, etc.) may be baked into the CUDA graph with specific memset sizes.
  // Composite captures execute create ops as live gap slots — they are NOT baked into
  // CUDA graphs — so value changes are handled naturally during replay execution.
  // Invalidating composite handles here destroys the entire capture and triggers
  // COMPOSITE_CAPTURE_FAILED on re-capture (the capture window has passed).
  if (!createValuesStable && seg.exec.replayHandle && !hasComposite) {
    DSP_DIAG(EXECUTE, "CREATE_VALUE_KEY mismatch: captured=%lld current=%lld → invalidating MONOLITHIC graph seg[%d-%d]",
             (long long)seg.exec.capturedCreateValueKey, (long long)createValueKey, seg.def.startSlot, seg.def.endSlot);
    SegmentLifecycle::invalidateForRebuild(this, seg, "create_value_key_mismatch");
    batchD2DCount_ = 0;
    extAddrsStable = false;  // Force re-capture path
  } else if (!createValuesStable && hasComposite) {
    DSP_DIAG(EXECUTE, "CREATE_VALUE_KEY mismatch: captured=%lld current=%lld — COMPOSITE seg[%d-%d] "
             "skipping invalidation (create ops execute live as gap slots)",
             (long long)seg.exec.capturedCreateValueKey, (long long)createValueKey,
             seg.def.startSlot, seg.def.endSlot);
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
    bool replayHandled = false;
    auto replayResult = segDispatchReplay(seg, externalArrays, numExt, stream,
                                          allowTritonCudaGraphReplay,
                                          createValuesStable, extAddrsStable,
                                          segShapeKey, backendName,
                                          replayHandled);
    if (replayHandled) {
      if (replayResult == Status::OK) {
        DSP_TRACE_GRAPH_REPLAYED(trace_, static_cast<int8_t>(segIdx),
                                 static_cast<uint32_t>(executeCount_),
                                 seg.def.startSlot, seg.def.endSlot);
      } else {
        DSP_TRACE_ERROR(trace_, static_cast<int8_t>(segIdx), seg.def.startSlot,
                        static_cast<uint32_t>(executeCount_),
                        static_cast<uint64_t>(replayResult));
      }
      return replayResult;
    }
  }
  // Fall through to capture or direct execution

  // Segments with terminal outcomes execute slots directly — no GPU graph.
  // This covers ZERO_KERNEL_SBS (0 GPU nodes), NOT_FUSIBLE, COMPILE_FAILED.
  if (isTerminalOutcome(seg.exec.outcome)) {
    DSP_DIAG(BACKEND,
             "TERMINAL_SLOT_BY_SLOT: seg[%d-%d] outcome=%s — "
             "executing %d slots directly (no GPU graph) execCount=%d",
             seg.def.startSlot, seg.def.endSlot,
             segmentExecOutcomeName(seg.exec.outcome),
             seg.def.endSlot - seg.def.startSlot + 1, seg.exec.executionCount);
    // Cross-stream + H2D + staging sync handled by dispatchSegment ->
    // performPreReplaySync (ONE function, ONE state machine).
    // SyncOverride forces prepareSpecialUse/registerSpecialUse in frozen mode.
    SyncOverride zeroNodeSyncGuard(*this, "terminal_sbs");
    return executeSegmentSlotBySlot(seg, externalArrays, numExt, stream);
  }

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
    ctx.nativeOnlyGraphCapture = allSlotsAreGaps || forceNativeCapture;
    ctx.hasCudaStream = hasCudaStream;
    return segDispatchCaptureOrDirect(seg, externalArrays, numExt, stream, ctx);
  }

}  // executeSegmentWithGpuGraph

}  // namespace graph
}  // namespace sd

#endif  // SD_CUDA

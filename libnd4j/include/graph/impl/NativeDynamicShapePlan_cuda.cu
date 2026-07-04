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

// Win32 threading API (HANDLE/LPVOID/DWORD/CreateThread) is used in the _WIN32
// branch below to launch precompile workers with a 64 MB stack.
// MUST define these guards BEFORE including <windows.h>:
//   NOGDI  — suppresses wingdi.h's `#define ERROR 0`, which otherwise clobbers the
//            `ERROR` enumerator in execution/cuda/CudaGraphScheduler.h's
//            `enum class GraphState { ... ERROR }` (this TU includes that header),
//            producing nvcc "error: expected an identifier". We use no GDI here.
//   NOMINMAX — keeps windows.h from defining min()/max() macros that break the
//            templated CUDA/std code pulled in transitively.
//   WIN32_LEAN_AND_MEAN — trims rarely-used Win32 headers (also speeds the parse).
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef NOGDI
#define NOGDI
#endif
#include <windows.h>
#endif

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
#include <graph/gpu/DspCudaDispatch.h>
#include <graph/PlanExecutionContext.h>
#include <helpers/MmulHelper.h>
#include <helpers/cublasHelper.h>
#include <cublas_v2.h>
#include <memory/cuda/CudaMemoryPool.h>
#include <helpers/AttentionWorkspace.h>
#include <graph/gpu/NvrtcKernelBuilder.h>
#include <graph/gpu/NvrtcKernelCache.h>
#include <ops/declarable/helpers/kv_scatter.h>
#include <system/Environment.h>
#ifndef _WIN32
#include <pthread.h>
#endif
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
#include <condition_variable>
#include <mutex>
#include <numeric>
#include <unordered_set>

extern thread_local cudaStream_t tl_dspGapStream;

namespace sd {

void dspPublishThreadCompletionEvent(void* streamPtr);

namespace graph {

// ── Per-GPU CUDA graph capture/execution coordination ───────────────────
// Shared across _cuda.cu, _cudagraph.cu, and _gpubackend.cu (extern there).
//
// During CUDA graph capture, ANY concurrent CUDA operation on the same device
// that touches the legacy/default stream triggers cudaError 906. This includes
// cudaMemcpyAsync, kernel launches, and stream synchronization from other
// threads executing plans concurrently.
//
// Coordination strategy:
//   - captureActive[dev]: atomic flag, set when a thread is capturing on device
//   - captureCV[dev] + captureMtx[dev]: condition variable to wake blocked threads
//   - Before capture: set flag, wait for all concurrent executions to finish
//   - Before execution: wait if capture is active on this device
//
// The executing thread holds no long-lived lock — it just checks the flag at
// entry. The capture thread sets the flag and waits for a short drain period.
std::atomic<bool> g_captureActive[16] = {};
std::mutex g_captureMtx[16];
std::condition_variable g_captureCV[16];
// Count of threads currently executing on each device.
std::atomic<int> g_execCount[16] = {};
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

  // Soft preconditions — return MAYBE so the caller falls through to normal execution.
  if (ModeContract::forMode(graphExecutionMode_).isSlotBySlot || planLifecycle_.isSlotBySlot()) {
    return Status::MAYBE;
  }
  if (executeCount_ < 1) {
    return Status::MAYBE;
  }
  // The frozen fast path requires shapes to be frozen and all segments to have
  // ready replay handles (monolithic or composite). Without this check, early
  // executions fall through to compositeReplay with empty schedules → KERNEL_FAILURE.
  if (!planLifecycle_.isInFrozenOrReplayState() || !allSegmentsReplayReady()) {
    return Status::MAYBE;
  }
  // After markVariable(), invalidateSegmentCaptures resets segment executionCount
  // to 0 and clears replay handles. However, the plan-level planLifecycle_.isShapesFrozen() stays
  // true and executeCount_ is NOT reset. If allSegmentsReplayReady() still returns
  // true (e.g., the handle wasn't fully cleared due to a code path bug), the
  // frozen fast path would replay a stale CUDA graph with baked-in addresses from
  // before markVariable — causing stuck/repeating outputs.
  //
  // Defensive gate: if ANY capturable segment has executionCount==0, it was
  // recently invalidated and needs warmup+recapture. Skip the frozen fast path
  // and let phaseReplay handle it through executeSegmentWithGraph (which does
  // slot-by-slot warmup when executionCount < CAPTURE_MIN_WARMUPS).
  for (auto& seg : segments_) {
    if (seg.def.allFrozenConstants) continue;
    if (isTerminalOutcome(seg.exec.outcome)) continue;
    if (!seg.def.isCapturable) continue;
    if (seg.exec.executionCount == 0) {
      DSP_DIAG(EXECUTE, "FROZEN_FAST_PATH: SKIP — seg[%d-%d] executionCount=0 "
               "(recently invalidated by markVariable), needs warmup+recapture",
               seg.def.startSlot, seg.def.endSlot);
      return Status::MAYBE;
    }
  }
  // Mode contract: some modes explicitly disable the frozen fast path.
  if (!ModeContract::forMode(graphExecutionMode_).allowsFrozenFastPath) {
    return Status::MAYBE;
  }

  // Ensure VERIFY diagnostics are enabled and at FULL level when tritonVerifyKernels is on.
  if (Environment::getInstance().tritonVerifyKernels()) {
    if (!DSP_DIAG_ENABLED(VERIFY)) {
      sd::graph::DspDiagnostics::getInstance().enableCategories(sd::graph::DSP_DIAG_VERIFY);
      sd::graph::DspDiagnostics::getInstance().setLevel(sd::graph::DSP_LEVEL_FULL);
    }
  }

  using Clock = std::chrono::high_resolution_clock;
  auto t0 = executionTimingEnabled_ ? Clock::now() : Clock::time_point{};

  cudaGetLastError();  // Clear stale CUDA error

  cudaStream_t cudaStr = (stream != nullptr)
      ? *static_cast<cudaStream_t*>(stream) : nullptr;
  sd::graph::DspStreamGuard dspStreamGuard(cudaStr);

  // Unified pre-replay sync for all segments: cross-stream ordering + H2D
  // variable inputs + D2D staging. Idempotent (PlanExecutionContext dedup flags).
  // Set GRAPH_REPLAY target: frozen fast path always replays captured graphs.
  {
    auto* execCtx = static_cast<PlanExecutionContext*>(activeExecCtx_);
    if (execCtx != nullptr) {
      execCtx->execTarget = ExecTarget::GRAPH_REPLAY;
    }
  }
  externalInputs = performPreReplaySync(externalInputs, numExternalInputs, stream, "frozen_fast_path");

  // ── NO gap-stream guard at this layer (by design — do not re-add) ─────────
  // The frozen fast path must NOT install a GapStreamGuard. Gap-stream ownership
  // belongs to whoever actually runs live inter-island gap ops, and only one
  // of the three per-segment branches below does:
  //   • Composite segments  → compositeReplay() installs its OWN GapStreamGuard
  //                           AFTER its own performPreReplaySync (gpubackend.cu
  //                           ~1099). It alone runs the gap matmuls; this path
  //                           delegates to it, exactly like the execute() path.
  //   • Monolithic segments → replayMonolithicGraph() replays a single captured
  //                           graph with every op baked in — no live gap ops.
  //   • Terminal/non-capturable → executeSegmentSlotBySlot() runs kernel-free
  //                           reshape/view/identity ops on the LaunchContext
  //                           stream — no cross-stream ordering requirement.
  //
  // A guard HERE is not merely redundant, it is doubly harmful:
  //   1. getCudaStream() overrides on tl_dspGapStream ONLY (LaunchContext.cu:224).
  //      With the guard active, performPreReplaySync (line 263) would observe
  //      getCudaStream()==cudaStr, so its cross-stream fence self-identifies
  //      (`defaultStream == cudaStr`) and is SILENTLY DROPPED → the replayed
  //      Triton island reads stale capture-time attention masks / position_ids
  //      → FROZEN token (the 27136-stuck decode). The retained DspStreamGuard
  //      above sets tl_dspExecutionStream (a different TL) and does NOT feed
  //      getCudaStream(), so the fence above fires correctly on the real stream.
  //   2. Holding the guard across the compositeReplay call double-routes the
  //      cuBLAS gap-stream setup → null gap-matmul arg → SIGSEGV in
  //      batchedGemmCastFloat2Half.
  // df8cee5d5f added a guard here (before the sync); the correct layer is inside
  // compositeReplay, not here.

  // ── Refresh stale view wrappers before replay ───────────────────────────
  // View ops (reshape, permute) create NDArray wrappers that alias their
  // input's DataBuffer. During CUDA graph capture, downstream compute kernels
  // (mmul, softmax) are recorded with the capture-time device addresses from
  // these view wrappers. If external input arrays are swapped between steps
  // (new NDArray objects with different specialBuffer() addresses), the view
  // wrappers become stale — but the captured CUDA graph still holds the old
  // addresses, causing error 700 on replay.
  //
  // The normal execute() path refreshes at NativeDynamicShapePlan.cpp line 2485,
  // but only when isShapesFrozen(). Once the plan enters REPLAYING, that guard
  // is false and the frozen fast path (this method) handles all execution —
  // so the refresh must happen here as well.
  for (size_t ri = 0; ri < segments_.size(); ri++) {
    refreshStaleViewWrappersInSegment(segments_[ri], externalInputs, numExternalInputs);
  }

  // ── Slot address drift detection ──────────────────────────────────────────
  // The monolithic CUDA graph has native op (cuBLAS) pointer arguments baked
  // into graph nodes at capture time. If any output slot's specialBuffer()
  // address changed since capture (e.g., view wrapper refresh created new
  // NDArray objects backed by different DataBuffers), replaying the graph
  // would dereference stale device pointers → CUDA error 700.
  //
  // The normal (non-frozen) path in executeSegmentWithCudaGraph has this check
  // and triggers recapture on drift. The frozen fast path must also check.
  // On drift, return MAYBE to fall back to the normal path which handles
  // invalidation and recapture properly.
  for (size_t segIdx = 0; segIdx < segments_.size(); segIdx++) {
    GraphSegment& seg = segments_[segIdx];
    if (seg.def.allFrozenConstants) continue;
    if (isTerminalOutcome(seg.exec.outcome) || !seg.def.isCapturable) continue;
    if (seg.exec.capturedSlotAddrHash == 0) continue;

    LongType currentAddrHash = computeSlotAddrHash(
        outputSlots_, seg.def.startSlot, seg.def.endSlot, totalOutputSlots_);
    if (seg.exec.slotAddrDrifted(currentAddrHash)) {
      DSP_DIAG(EXECUTE,
               "FROZEN_FAST_PATH: SLOT_ADDR_DRIFT for seg[%d-%d] "
               "captured=0x%llx current=0x%llx — falling back to normal path for recapture",
               seg.def.startSlot, seg.def.endSlot,
               (long long)seg.exec.capturedSlotAddrHash, (long long)currentAddrHash);
      return Status::MAYBE;
    }
  }

  // ── Per-segment replay iteration ─────────────────────────────────────────
  // Iterate all segments and replay each one. Every segment must have a replay
  // handle (monolithic or composite) — allSegmentsReplayReady() was checked
  // by the caller before entry.
  DSP_DIAG(EXECUTE, "FROZEN_FAST_PATH: replaying %d segments (execCount=%d)",
           (int)segments_.size(), (int)executeCount_);

  for (size_t segIdx = 0; segIdx < segments_.size(); segIdx++) {
    GraphSegment& seg = segments_[segIdx];

    // All-frozen-constant segments: outputs are already populated from warmup.
    // No capture, no replay, no execution needed.
    if (seg.def.allFrozenConstants) {
      seg.exec.executionCount++;
      continue;
    }

    // Segments with terminal outcomes or non-capturable — no replay handles.
    // Execute slot-by-slot (reshape/view/identity ops with no kernels).
    if (isTerminalOutcome(seg.exec.outcome) || !seg.def.isCapturable) {
      if (!bindSegmentCudaDevice(seg, slots_, numSlots_, "frozenFastPath_sbs")) {
        return Status::KERNEL_FAILURE;
      }
      // SyncOverride: frozen fast path has needsSync()=false (frozen steady
      // state, no contract override). Without this, executeSlot skips
      // registerSpecialUse — output actuality flags stay stale from the
      // previous step, causing the NEXT step to read stale device data.
      // This mirrors the TERMINAL_SLOT_BY_SLOT guard in
      // executeSegmentWithGpuGraph (NativeDynamicShapePlan_gpubackend.cu).
      SyncOverride frozenSbsSync(*this, "frozenFastPath_terminal_sbs");
      auto sbsStatus = executeSegmentSlotBySlot(seg, externalInputs, numExternalInputs, stream);
      if (sbsStatus != Status::OK) {
        DSP_DIAG(EXECUTE, "FROZEN_FAST_PATH: slot-by-slot FAILED seg[%d-%d] status=%d "
                 "(outcome=%d isCapturable=%d)",
                 seg.def.startSlot, seg.def.endSlot, (int)sbsStatus,
                 (int)seg.exec.outcome, (int)seg.def.isCapturable);
        return sbsStatus;
      }
      seg.exec.executionCount++;
      continue;
    }

    if (!bindSegmentCudaDevice(seg, slots_, numSlots_, "frozenFastPath")) {
      return Status::KERNEL_FAILURE;
    }

    bool hasMonolithicReplay = (seg.exec.replayHandle != nullptr && seg.exec.replayHandle->isReady());
    bool hasCompositeSchedule = !seg.exec.compositeReplaySchedule.units.empty();

    if (hasMonolithicReplay) {
      // ── Monolithic graph replay (consolidated) ─────────────────────────
      // All pre-zero, arg refresh, replay, counters, fixup, and verify steps
      // are handled by the unified replayMonolithicGraph() method.

      // ── View-of-ext-input staleness guard ─────────────────────────────
      // If any view slot's DataBuffer changed since capture (VIEW-BUF-CHANGE
      // in _slotexec.cpp bumped argTableGeneration), the monolithic CUDA graph
      // has stale baked device addresses for that view's output slot.
      // Replaying the graph with a stale address reads freed GPU memory → err700.
      //
      // When needsArgRefresh() is true here (bumped by VIEW-BUF-CHANGE), fall
      // back to the normal execute() path which calls refreshStaleViewWrappers
      // + SLOT_ADDR_DRIFT detection + recapture.  The arg generation is NOT
      // cleared here — the normal path's segment dispatch will call
      // markArgsCurrent() after the recapture is committed.
      if (seg.exec.needsArgRefresh()) {
        DSP_DIAG(EXECUTE,
                 "FROZEN_FAST_PATH: needsArgRefresh seg[%d-%d] "
                 "(view-of-ext-input addr changed, stale baked addr) — returning MAYBE",
                 seg.def.startSlot, seg.def.endSlot);
        return Status::MAYBE;
      }

      DSP_DIAG(STREAM_SYNC,
               "FROZEN_FAST_PATH pre-replay: seg[%d-%d] execCount=%d "
               "cublasWsPtr=%p cublasWsSize=%zu deterministicCublas=%d cublasLtDisabled=%d",
               seg.def.startSlot, seg.def.endSlot, (int)executeCount_,
               cublasWorkspaceBuffer_, cublasWorkspaceSize_,
               (int)ModeContract::forMode(graphExecutionMode_).requiresDeterministicCublas,
               (int)tl_cublasLtDisabled);

      auto replayStatus = replayMonolithicGraph(seg, externalInputs, numExternalInputs,
                                                stream, "frozen_fast_path");
      if (replayStatus == Status::KERNEL_FAILURE) {
        // MONOLITHIC_SLOT_ADDR_DRIFT: the monolithic CUDA graph has stale cuBLAS
        // args (view-slot buffer addresses changed since capture).
        // invalidateForRebuild() was called inside replayMonolithicGraph.
        // Return MAYBE so the caller falls back to full execute() which will
        // re-warmup and re-capture with the correct device addresses.
        DSP_DIAG(EXECUTE,
                 "FROZEN_FAST_PATH: replayMonolithicGraph KERNEL_FAILURE seg[%d-%d] "
                 "(monolithic_slot_addr_drift) — returning MAYBE for recapture",
                 seg.def.startSlot, seg.def.endSlot);
        return Status::MAYBE;
      }
      if (replayStatus != Status::OK) return replayStatus;

    } else if (!seg.exec.compositeReplaySchedule.units.empty() && !seg.exec.hasGapsInGraph()) {
      // ── Composite replay (schedule has units — merged or island handles) ──
      // Guard: hasGapsInGraph()=true when monolithic (native-only) capture baked the
      // gap ops into the CUDA graph. In that case replayHandle IS set; arriving here
      // means replayHandle is somehow null — fall through to the BUG branch rather
      // than trying composite replay with null handles (→ cudaLaunchKernel SIGSEGV).
      //
      // ── needsArgRefresh() guard (mirrors monolithic branch above) ─────────
      // compositeReplay's internal SLOT_ADDR_DRIFT checks are only exercised
      // DURING replay (inside compositeReplay).  When needsArgRefresh()=true
      // (bumped each post-capture replay by bumpArgGeneration, or by a
      // VIEW-BUF-CHANGE event), the merged CUDA-graph nodes may have stale
      // baked cuBLAS device addresses that compositeReplay's drift checks will
      // NOT catch on this path because refreshArgTablesForReplay updates only
      // the Triton arg table.  Replaying with stale addresses reads freed GPU
      // memory → err700 (observed in testBufferAliasVaryingInput).
      //
      // Return MAYBE to fall back to the normal execute() path which calls
      // refreshStaleViewWrappers + SLOT_ADDR_DRIFT detection + recapture.
      // The arg generation is NOT cleared here — the normal path's segment
      // dispatch will call markArgsCurrent() after recapture is committed.
      if (seg.exec.needsArgRefresh()) {
        DSP_DIAG(EXECUTE,
                 "FROZEN_FAST_PATH: needsArgRefresh seg[%d-%d] "
                 "(composite: stale baked cuBLAS addr, drift-check skipped) — returning MAYBE",
                 seg.def.startSlot, seg.def.endSlot);
        return Status::MAYBE;
      }

      auto replayStatus = compositeReplay(seg, seg.exec.compositeReplaySchedule,
                                          externalInputs, numExternalInputs, stream);
      if (replayStatus != Status::OK) {
        DSP_DIAG(EXECUTE, "FROZEN_FAST_PATH: composite replay FAILED seg[%d-%d] status=%d",
                 seg.def.startSlot, seg.def.endSlot, (int)replayStatus);
        // KERNEL_FAILURE from compositeReplay due to merged graph addr drift:
        // invalidateForRebuild() was called inside compositeReplay, clearing all
        // handles and resetting state for re-capture. Return MAYBE so the caller
        // falls back to full execute() which handles re-warmup and re-capture.
        // Without this, KERNEL_FAILURE propagates to autoregressive_decode which
        // treats it as a fatal non-recoverable error.
        if (replayStatus == Status::KERNEL_FAILURE &&
            seg.exec.replayHandle == nullptr &&
            seg.exec.compositeReplaySchedule.units.empty()) {
          DSP_DIAG(EXECUTE,
                   "FROZEN_FAST_PATH: composite replay invalidated seg[%d-%d] "
                   "(merged_graph_addr_drift) — returning MAYBE for full execute() fallback",
                   seg.def.startSlot, seg.def.endSlot);
          return Status::MAYBE;
        }
        return replayStatus;
      }
      totalGraphReplays_++;
      seg.exec.executionCount++;
    } else {
      // ── No replay handles — this is a BUG, not a fallback ──
      // If we reached the frozen fast path, allSegmentsReplayReady() was
      // supposed to be true. A capturable segment with no replay handles
      // means capture silently failed or was never attempted. Return MAYBE
      // so the caller falls back to full execute() which handles re-warmup
      // and re-capture. NEVER silently fall back to slot-by-slot here —
      // that hides bugs and makes it impossible to tell which path ran.
      DSP_DIAG(EXECUTE, "FROZEN_FAST_PATH: BUG — no replay handles for seg[%d-%d] "
               "(capturable=%d outcome=%d compileFailed=%d execCount=%d compiledBy=%s) "
               "— returning MAYBE for full execute() re-warmup",
               seg.def.startSlot, seg.def.endSlot,
               (int)seg.def.isCapturable, (int)seg.exec.outcome,
               (int)seg.exec.compilationFailed, seg.exec.executionCount,
               seg.exec.compiledByBackend.c_str());
      return Status::MAYBE;
    }
  }

  // All segments replayed successfully.
  // Plan-output boundary: materialize any VIEW in a requested-output slot before
  // returning to Java. Same reasoning as the normal-path version in NativeDynamicShapePlan.cpp:
  // a view's DataBuffer is shared with its parent slot; the next replay will overwrite
  // that parent → Java's previously-returned pointer would read stale/zero data.
  for (int i = 0; i < numRequestedOutputs_; i++) {
    int slotIdx = requestedOutputSlotIndices_[i];
    if (slotIdx >= 0 && slotIdx < totalOutputSlots_) {
      NDArray* slotArr = outputSlots_[slotIdx];
      if (slotArr != nullptr && slotArr->isView()) {
        materializeViewSlot(slotIdx, "plan-output-view-boundary-frozen");
      }
    }
  }

  // Populate requested outputs from (potentially-materialized) slots
  for (int i = 0; i < numRequestedOutputs_; i++) {
    int slotIdx = requestedOutputSlotIndices_[i];
    if (slotIdx >= 0 && slotIdx < totalOutputSlots_) {
      requestedOutputs[i] = outputSlots_[slotIdx];
    } else {
      requestedOutputs[i] = nullptr;
    }
  }
  incrementExecuteCount("native_replay");

  if (executionTimingEnabled_) {
    auto tDone = Clock::now();
    auto totalUs = std::chrono::duration_cast<std::chrono::microseconds>(tDone - t0).count();
    DSP_DIAG(TIMING, "DSP timing: frozen_fast_path total=%lldus segs=%d",
             totalUs, (int)segments_.size());
  }

  // ── Frozen fast path output diagnostics ─────────────────────────────────
  // Mirror the normal execute() path's POST_EXEC and LOGITS_ARGMAX diagnostics.
  // Without this, the frozen fast path is a diagnostic black hole — no output
  // value logging, no argmax, no way to trace divergence without re-running
  // through the normal path.
  if (DSP_DIAG_ENABLED(VERIFY)) {
    // POST_EXEC slot/segment summary
    int nullSlots = 0, liveSlots = 0;
    int replaySegs = 0, slotBySlotSegsCount = 0;
    for (int i = 0; i < totalOutputSlots_; i++) {
      if (outputSlots_[i] == nullptr) { nullSlots++; } else { liveSlots++; }
    }
    for (const auto& seg : segments_) {
      bool hasComposite = false;
      for (auto& h : seg.exec.compositeReplaySchedule.mergedReplayHandles) {
        if (h != nullptr && h->isReady()) { hasComposite = true; break; }
      }
      if (!hasComposite) {
        for (auto& u : seg.exec.compositeReplaySchedule.units) {
          if (u.kind == REPLAY_UNIT_TRITON_ISLAND && u.mergedGroupId < 0) {
            int idx = u.islandIndex;
            if (idx >= 0 && idx < static_cast<int>(seg.exec.compositeReplaySchedule.compositeReplayHandles.size()) &&
                seg.exec.compositeReplaySchedule.compositeReplayHandles[idx] != nullptr &&
                seg.exec.compositeReplaySchedule.compositeReplayHandles[idx]->isReady()) {
              hasComposite = true; break;
            }
          }
        }
      }
      if ((seg.exec.replayHandle && seg.exec.replayHandle->isReady()) || hasComposite) replaySegs++;
      else slotBySlotSegsCount++;
    }
    DSP_DIAG(VERIFY, "POST_EXEC_FROZEN exec=%d: slots(live=%d null=%d/%d) "
             "segs(replay=%d sbs=%d/%d) graphReplays=%d",
             executeCount_, liveSlots, nullSlots, totalOutputSlots_,
             replaySegs, slotBySlotSegsCount, (int)segments_.size(),
             (int)totalGraphReplays_);

    // Per-requested-output metadata. Host value dumps are intentionally omitted
    // here: frozen replay stays fully async and must not drain the DSP stream.
    for (int i = 0; i < numRequestedOutputs_; i++) {
      int slotIdx = requestedOutputSlotIndices_[i];
      NDArray* arr = (slotIdx >= 0 && slotIdx < totalOutputSlots_)
                     ? outputSlots_[slotIdx] : nullptr;
      if (arr != nullptr && arr->specialBuffer() != nullptr && arr->lengthOf() > 0) {
        DSP_DIAG_SLOT(VERIFY, slotIdx,
            "FROZEN_FAST_PATH reqOut[%d] exec=%d len=%lld dtype=%d sbuf=%p "
            "(async path: value dump skipped)",
            i, executeCount_, (long long)arr->lengthOf(),
            static_cast<int>(arr->dataType()), arr->specialBuffer());
      } else if (arr == nullptr) {
        DSP_DIAG_SLOT(VERIFY, slotIdx, "FROZEN_FAST_PATH reqOut[%d] exec=%d nullptr",
                      i, executeCount_);
      }
    }

    // LOGITS_ARGMAX — reuse the standard diagnostic function
    platformDumpLogitsArgmax(executeCount_, stream);
  }

  // Diagnostic: metadata only. Do not block the stream to dump values here.
  if (Environment::getInstance().isDebug() && executeCount_ >= 10 && executeCount_ <= 22) {
    if (!segments_.empty()) {
      auto& lastSeg = segments_.back();
      // Use the last step's actual output slot index (not the step index)
      int lastStepIdx = lastSeg.def.endSlot;
      int lastOutSlot = -1;
      if (lastStepIdx >= 0 && lastStepIdx < numSlots_ && slots_[lastStepIdx].wiring.numOutputs > 0) {
        lastOutSlot = slots_[lastStepIdx].wiring.outputSlotIndices[0];
      }
      if (lastOutSlot >= 0 && lastOutSlot < totalOutputSlots_ && outputSlots_[lastOutSlot] != nullptr) {
        NDArray* logitsArr = outputSlots_[lastOutSlot];
        if (logitsArr->lengthOf() > 0 && logitsArr->specialBuffer() != nullptr) {
          DSP_DIAG(VERIFY, "REPLAY DEBUG: exec=%d slot=%d len=%lld dtype=%d sbuf=%p "
                           "(async path: argmax dump skipped)",
                   executeCount_, lastOutSlot, (long long)logitsArr->lengthOf(),
                   static_cast<int>(logitsArr->dataType()), logitsArr->specialBuffer());
        }
      }
    }
  }

  return Status::OK;
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

  // Launch worker threads with 64 MB stack — compileSegment() calls into
  // Triton MLIR passes whose recursive SSA traversal can exceed the default
  // 8 MB std::thread stack.  Inner compile workers already use this size
  // (TRITON_COMPILE_WORKER_STACK_SIZE in TritonGraphBackend_compile.cu);
  // outer workers need it too since they call compileSegment() directly.
  static constexpr size_t PRECOMPILE_WORKER_STACK_SIZE = 64 * 1024 * 1024;
#ifdef _WIN32
  {
    struct WinWorkerArg { std::function<void()>* fn; };
    auto fnObj = std::function<void()>(workerFn);

    std::vector<HANDLE> threadHandles(numThreads);
    std::vector<WinWorkerArg> args(numThreads);
    std::vector<std::thread> fallbackThreads;

    for (int t = 0; t < numThreads; t++) {
      args[t].fn = &fnObj;
      threadHandles[t] = CreateThread(
          nullptr,
          PRECOMPILE_WORKER_STACK_SIZE,
          [](LPVOID arg) -> DWORD {
            auto* wa = static_cast<WinWorkerArg*>(arg);
            (*wa->fn)();
            return 0;
          }, &args[t], STACK_SIZE_PARAM_IS_A_RESERVATION, nullptr);
      if (threadHandles[t] == nullptr) {
        fallbackThreads.emplace_back(workerFn);
      }
    }

    for (int t = 0; t < numThreads; t++) {
      if (threadHandles[t] != nullptr) {
        WaitForSingleObject(threadHandles[t], INFINITE);
        CloseHandle(threadHandles[t]);
      }
    }
    for (auto& ft : fallbackThreads) {
      ft.join();
    }
  }
#else
  {
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, PRECOMPILE_WORKER_STACK_SIZE);

    struct WorkerArg { std::function<void()>* fn; };
    auto fnObj = std::function<void()>(workerFn);

    std::vector<pthread_t> threads(numThreads, 0);
    std::vector<WorkerArg> args(numThreads);
    int inlineFallbackCount = 0;  // number of workers that must run on calling thread

    for (int t = 0; t < numThreads; t++) {
      args[t].fn = &fnObj;
      int rc = pthread_create(&threads[t], &attr,
          [](void* arg) -> void* {
            auto* wa = static_cast<WorkerArg*>(arg);
            (*wa->fn)();
            return nullptr;
          }, &args[t]);
      if (rc != 0) {
        threads[t] = 0;
        inlineFallbackCount++;
        DSP_DIAG(COMPILE,
                 "NativeDSP::precompile: pthread_create failed for outer worker %d (rc=%d), "
                 "will run inline on calling thread", t, rc);
      }
    }
    pthread_attr_destroy(&attr);

    for (int t = 0; t < numThreads; t++) {
      if (threads[t] != 0) pthread_join(threads[t], nullptr);
    }
    // Run inline fallback iterations on the calling thread instead of spawning
    // std::thread (which would also throw std::system_error when resources are
    // exhausted, crashing the process).  workerFn is a work-stealing loop that
    // drains the shared task queue, so calling it once per failed pthread
    // consumes all remaining tasks.
    for (int i = 0; i < inlineFallbackCount; i++) {
      workerFn();
    }
  }
#endif

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

    // Migrate device-to-device without a host-side synchronization.
    int savedDevice = -1;
    cudaGetDevice(&savedDevice);

    cudaSetDevice(sourceDevice);
    std::vector<NDArray*> reads{arr};
    NDArray::prepareSpecialUse({}, reads);
    void* srcDev = db->special();

    cudaSetDevice(targetDevice);

    // Create new array on target device with same shape and data type
    std::vector<LongType> shapeVec(*arr->getShapeAsVector());
    auto* copy = new NDArray(arr->ordering(), shapeVec, arr->dataType(),
                             LaunchContext::defaultContext());

    std::vector<NDArray*> writes{copy};
    NDArray::prepareSpecialUse(writes, {});
    void* dstDev = copy->dataBuffer() != nullptr ? copy->dataBuffer()->special() : nullptr;
    auto srcLen = arr->lengthOf() * DataTypeUtils::sizeOf(arr->dataType());
    if (srcLen > 0 && srcDev != nullptr && dstDev != nullptr) {
      auto* streamPtr = LaunchContext::defaultContext()->getCudaStream();
      cudaStream_t cudaStr = (streamPtr != nullptr) ? *streamPtr : nullptr;
      auto copyErr = cudaMemcpyPeerAsync(dstDev, targetDevice, srcDev, sourceDevice,
                                         static_cast<size_t>(srcLen), cudaStr);
      if (copyErr != cudaSuccess) {
        DSP_DIAG(MULTI_DEVICE,
                 "migrateSlotInputsToTargetDevice: cudaMemcpyPeerAsync failed srcDev=%d dstDev=%d bytes=%lld err=%s",
                 sourceDevice, targetDevice, (long long)srcLen, cudaGetErrorString(copyErr));
        cudaGetLastError();
        delete copy;
        if (savedDevice >= 0) cudaSetDevice(savedDevice);
        continue;
      }
    }
    NDArray::registerSpecialUse(writes, reads);

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
// Pre-segment sync is handled by performPreReplaySync (in _prereplay.cu),
// called from dispatchSegment. All sync tracked via PreReplaySyncPhase.

// ═══════════════════════════════════════════════════════════════════════════════

bool NativeDynamicShapePlan::platformShouldUseGraph(const GraphSegment& segment) {
  auto mode = ModeContract::forMode(graphExecutionMode_);

  // ── Structural exemptions: these legitimately skip graph execution ──
  if (mode.isSlotBySlot) return false;
  if (planLifecycle_.isSlotBySlot()) return false;
  if (!segment.def.isCapturable) return false;  // data-dependent / control-flow ops

  if (Environment::getInstance().tritonSkipKernels()) {
    DSP_DIAG_SEG(EXECUTE, segment.def.startSlot,
                 "platformShouldUseGraph: false (tritonSkipKernels)");
    return false;
  }

  // ── Capturable segment, post-freeze, graph mode — should use graph ──
  // No Triton-island check here: CUDA graph capture records ALL GPU operations
  // (cuBLAS, element-wise, Triton-compiled, etc.). A segment with 0 Triton
  // sub-kernels is still graph-capturable via monolithic capture. Triton is just
  // another kernel type — segments do NOT need it to be replayable.
  bool hasBackend = (segment.def.selectedBackend == SelectedBackend::GPU_COMPILER ||
                     segment.def.selectedBackend == SelectedBackend::CUDA_GRAPHS);
  bool canCapture = !segment.exec.compilationFailed &&
                    !isTerminalOutcome(segment.exec.outcome) && hasBackend;

  if (!canCapture && planLifecycle_.isInFrozenOrReplayState() && !mode.allowsFallback) {
    // Terminal outcomes are normal — the segment had no GPU work or can't be fused.
    // It permanently executes slot-by-slot. This is NOT a mode violation.
    if (!isTerminalOutcome(segment.exec.outcome)) {
      REQUIRE_TRUE(false, 0,
                   "DSP MODE VIOLATION: seg[%d-%d] capturable but cannot graph-execute. "
                   "mode=%d compilFailed=%d backend=%d outcome=%d. "
                   "Graph mode requires capture/replay — silent fallback is banned.",
                   segment.def.startSlot, segment.def.endSlot,
                   static_cast<int>(graphExecutionMode_),
                   static_cast<int>(segment.exec.compilationFailed),
                   static_cast<int>(segment.def.selectedBackend),
                   static_cast<int>(segment.exec.outcome));
    }
  }

  if (!canCapture) {
    DSP_DIAG_SEG(EXECUTE, segment.def.startSlot,
                 "platformShouldUseGraph: false (compilFailed=%d backend=%d outcome=%d)",
                 segment.exec.compilationFailed ? 1 : 0,
                 static_cast<int>(segment.def.selectedBackend),
                 static_cast<int>(segment.exec.outcome));
  }
  return canCapture;
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

      // compilationFailed is checked by platformShouldUseGraph() — the single
      // gate for graph eligibility.  If we reach here, it returned true, which
      // implies compilationFailed == false.

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
        SegmentLifecycle::markNotFusible(segment.exec, "gpu_compiler_not_fusible", segment.def.startSlot, segment.def.endSlot);
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

      // GPU backend (Triton) capture/execution failed. POLICY: NEVER fall back to
      // slot-by-slot — a fallback masks the real capture failure (and silently drops to
      // ~8 tok/s). Mark permanently failed and throw so the root cause is fixed (capture
      // must actually engage), exactly like the CUDA_GRAPHS path below.
      SegmentLifecycle::markFailed(segment.exec, "gpu_backend_exec_failed", segment.def.startSlot, segment.def.endSlot);
      DSP_THROW_SEG(COMPILE, segment.def.startSlot,
                    "NativeDSP::execute: exec%d seg[%d-%d] gpuBackend=%s FAILED status=%d. "
                    "GPU compilation/capture failed — fix the root cause.",
                    executeCount_, segment.def.startSlot, segment.def.endSlot, gpuBackend->name(),
                    static_cast<int>(status));
    }

    case SelectedBackend::CUDA_GRAPHS: {
      auto status = executeSegmentWithGraph(segment, externalInputs, numExternalInputs, stream);
      if (status != Status::OK) {
        // Graph capture failed — mark permanently failed and throw.
        // Do NOT fall back to slot-by-slot; fix the root cause.
        SegmentLifecycle::markFailed(segment.exec, "cuda_graph_capture_failed", segment.def.startSlot, segment.def.endSlot);
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
                   executeCount_, static_cast<int>(planLifecycle_.isShapesFrozen()),
                   static_cast<int>(lastErr));
  }
  return Status::OK;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Platform dispatch: Graph-baked address pin/unpin
// ═══════════════════════════════════════════════════════════════════════════════

void NativeDynamicShapePlan::platformPinGraphBakedAddress(void* ptr, int deviceId) {
  memory::CudaMemoryPool::getInstance().pinGraphBakedAddress(ptr, deviceId);
}

void NativeDynamicShapePlan::platformFlushGraphBakedPins(void* streamVoid) {
  if (graphPinnedAddrs_.empty()) return;
  cudaStream_t freeStream = (ownedStream_ != nullptr) ? *ownedStream_
                            : (streamVoid != nullptr ? static_cast<cudaStream_t>(streamVoid) : nullptr);
  DSP_DIAG(MEMORY, "platformFlushGraphBakedPins: flushing %d graph-baked pins stream=%p",
           (int)graphPinnedAddrs_.size(), (void*)freeStream);
  for (auto& pa : graphPinnedAddrs_) {
    memory::CudaMemoryPool::getInstance().unpinGraphBakedAddress(pa.ptr, pa.deviceId, freeStream);
  }
  graphPinnedAddrs_.clear();
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
    SegmentLifecycle::resetForResourceRelease(seg.exec);
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
  // Clear merged group tags on schedule units, then discard the units entirely so
  // that buildCompositeReplaySchedule() is forced to rebuild them after the next
  // recompile.  Without this, the lazy guard
  //   if (seg.exec.compositeReplaySchedule.units.empty() && ctx.tritonBackend != nullptr)
  // never fires when a plan is invalidated+recompiled with different Triton settings,
  // leaving stale island/gap boundaries that cause COMPOSITE_CAPTURE_FAILED.
  for (auto& u : seg.exec.compositeReplaySchedule.units) {
    u.mergedGroupId = -1;
    u.isMergedLeader = false;
  }
  seg.exec.compositeReplaySchedule.units.clear();
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
  seg.exec.compositeReplaySchedule.compositeReplayHandles.clear();
  seg.exec.gapOpsCapturedInGraph = false;
  seg.exec.markArgsStale();
  seg.resolvedCpuBackend = nullptr;

  // Release graph-baked address pins for the segment being invalidated — but ONLY the
  // plan-owned intermediates (externalOwned=false). Their now-dead graph will not read them,
  // so unpinning (and any deferred free) here is safe. KEEP externally-owned pins (a
  // SOURCE_VARIABLE weight or a view over one): a later exec in the SAME execute() can still
  // read that buffer after a weight rebind, and if a user close() already set freeRequested,
  // unpinning here would issue cudaFreeAsync MID-execute and dangle the matmul's weight input
  // → err700 illegal access. Externally-owned pins are flushed at plan teardown
  // (releaseGpuIntermediates → platformFlushGraphBakedPins), after the owned stream is synced
  // and no exec can read the buffer. graphPinnedAddrs_ entries carry segStartSlot + externalOwned.
  if (!graphPinnedAddrs_.empty()) {
    cudaStream_t freeStream = (ownedStream_ != nullptr) ? *ownedStream_ : nullptr;
    int toFlush = 0, deferredExt = 0;
    std::vector<GraphPinnedAddr> remaining;
    for (auto& pa : graphPinnedAddrs_) {
      if (pa.segStartSlot == seg.def.startSlot && !pa.externalOwned) {
        memory::CudaMemoryPool::getInstance().unpinGraphBakedAddress(pa.ptr, pa.deviceId, freeStream);
        toFlush++;
      } else {
        if (pa.segStartSlot == seg.def.startSlot) deferredExt++;
        remaining.push_back(pa);
      }
    }
    graphPinnedAddrs_ = std::move(remaining);
    DSP_DIAG(MEMORY, "platformCleanupSegmentForRebuild: released %d intermediate pins for seg[%d-%d], "
             "deferred %d external-owned to teardown, %d total remaining",
             toFlush, seg.def.startSlot, seg.def.endSlot, deferredExt, (int)graphPinnedAddrs_.size());
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Platform dispatch: Plan resource cleanup (destructor)
// ═══════════════════════════════════════════════════════════════════════════════

void NativeDynamicShapePlan::platformFreePlanResources() {
  DSP_DIAG(MEMORY, "platformFreePlanResources: segments=%d slots=%d outputs=%d",
           (int)segments_.size(), numSlots_, totalOutputSlots_);

  // This teardown frees the plan's constant DataBuffers; their heap addresses
  // will be reused. Invalidate every thread's cast-cache skip-assign guards
  // (pointer-identity keyed) so a successor plan's same-shape constant at a
  // recycled address cannot silently reuse THIS plan's cast weight values.
  MmulHelper::bumpCastCacheEpoch();

  // Drain any pending GPU work on the plan's owned stream BEFORE destroying
  // any resources.  Without this, in-flight kernels or CUDA graph replays
  // may still reference buffers, streams, or graph handles that are about
  // to be freed — producing SIGSEGV (use-after-free) or "invalid resource
  // handle" errors.  This is the root fix for Pattern B crashes where
  // executeDynamicShapePlan faults on a stale cudaStream_t.
  if (ownedStream_ != nullptr) {
    cudaError_t syncErr = cudaStreamSynchronize(*ownedStream_);
    if (syncErr != cudaSuccess) {
      // Stream sync can fail if the CUDA context is already destroyed
      // (e.g., during JVM shutdown) or if a prior async kernel error is
      // pending.  Clear the sticky error and continue with teardown —
      // we still need to null out pointers to prevent double-free.
      // Without cudaGetLastError(), the sticky error propagates to the
      // next plan's cudaStreamBeginCapture → cascade failure → SIGABRT.
      cudaGetLastError();
      DSP_DIAG(MEMORY, "platformFreePlanResources: cudaStreamSynchronize failed: %s (cleared, continuing teardown)",
               cudaGetErrorString(syncErr));
    }
  }

  // Free plan-owned CUDA stream
  if (ownedStream_ != nullptr) {
    DSP_DIAG(MEMORY, "platformFreePlanResources: destroying plan-owned stream=%p",
             static_cast<void*>(*ownedStream_));
    cudaStreamDestroy(*ownedStream_);
    delete ownedStream_;
    ownedStream_ = nullptr;
  }

  // Free sync-free buffer fingerprint ring (BUF_FP_RING instrumentation).
  if (d_fpRing_ != nullptr) { cudaFree(d_fpRing_); d_fpRing_ = nullptr; }
  if (h_fpRing_ != nullptr) { delete[] h_fpRing_; h_fpRing_ = nullptr; }
  fpRingEnabled_ = false; fpRingDrained_ = false;

  // Clear AttentionWorkspace — holds named GPU buffers (attention scratch, softmax
  // intermediate) that persist across plan lifetimes. Without this, the next plan's
  // CUDA graph capture records addresses of the old workspace buffers.
  AttentionWorkspace::getInstance()->clear();

  // Free CUDA event used for cross-stream sync (on the device it was created on).
  // Handle-value representation: executionCompleteEvent_ holds the cudaEvent_t handle directly
  // (matches dspCreateEvent and the steady-state path), NOT a pointer to a heap-allocated handle.
  // Previously this deref'd the void* as a cudaEvent_t* and delete'd it; when the steady-state
  // path created the event via dspCreateEvent (handle-value) that deref read garbage and
  // cudaEventDestroy segfaulted (TestNativeDecodeLoopRegression crash).
  if (executionCompleteEvent_ != nullptr) {
    cudaEvent_t evt = reinterpret_cast<cudaEvent_t>(executionCompleteEvent_);
    if (executionCompleteEventDeviceId_ >= 0) {
      int savedDev;
      cudaGetDevice(&savedDev);
      if (savedDev != executionCompleteEventDeviceId_) {
        cudaSetDevice(executionCompleteEventDeviceId_);
        cudaEventDestroy(evt);
        cudaSetDevice(savedDev);
      } else {
        cudaEventDestroy(evt);
      }
    } else {
      cudaEventDestroy(evt);
    }
    executionCompleteEvent_ = nullptr;
    executionCompleteEventDeviceId_ = -1;
  }

  // Free cached steady-state cross-stream event (created via dspCreateEvent -> handle-value).
  if (steadyStateCrossStreamEvent_ != nullptr) {
    sd::graph::dspDestroyEvent(steadyStateCrossStreamEvent_);
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

  // Flush all graph-baked address pins BEFORE destroying segment GPU resources.
  // Addresses pinned by writeOutputSlot are safe to release now because the
  // CUDA graphs that baked them are about to be destroyed below. Flushing here
  // (before replayHandle.reset()) ensures no graph can replay against the freed
  // addresses after unpinning.
  // NOTE: ownedStream_ was already destroyed above (cudaStreamDestroy + nullptr).
  // Use stream 0 (nullptr) for the deferred cudaFreeAsync — the pool will sync
  // it if needed during the next trimPool call.
  if (!graphPinnedAddrs_.empty()) {
    DSP_DIAG(MEMORY, "platformFreePlanResources: flushing %d graph-baked pins (stream 0)",
             (int)graphPinnedAddrs_.size());
    for (auto& pa : graphPinnedAddrs_) {
      memory::CudaMemoryPool::getInstance().unpinGraphBakedAddress(pa.ptr, pa.deviceId, nullptr);
    }
    graphPinnedAddrs_.clear();
  }

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
      SegmentLifecycle::resetForResourceRelease(seg.exec);
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
    seg.exec.markArgsStale();
    seg.exec.gapOpsCapturedInGraph = false;
    seg.resolvedCpuBackend = nullptr;
    delete seg.exec.jitKernel;
    seg.exec.jitKernel = nullptr;
  }

  // Release capture workspace reference. If this is the global shared workspace
  // (used by multiple plans), we don't free it — it persists for other plans.
  // Only free if it's a plan-private workspace (legacy or non-global).
  if (sharedCaptureWorkspace_ != nullptr) {
    extern void* g_globalCaptureWorkspace;
    if (sharedCaptureWorkspace_ == g_globalCaptureWorkspace) {
      DSP_DIAG(MEMORY, "platformFreePlanResources: releasing reference to GLOBAL capture workspace %zuMB (NOT freeing)",
               sharedCaptureWorkspaceBytes_ / (1024*1024));
    } else {
      memory::CudaMemoryPool::getInstance().unregisterCaptureWorkspace(sharedCaptureWorkspace_);
      memory::CudaMemoryPool::getInstance().free(sharedCaptureWorkspace_, sharedCaptureWorkspaceDevice_);
      DSP_DIAG(MEMORY, "platformFreePlanResources: pool.free private capture workspace %zuMB on device %d",
               sharedCaptureWorkspaceBytes_ / (1024*1024), sharedCaptureWorkspaceDevice_);
    }
    sharedCaptureWorkspace_ = nullptr;
    sharedCaptureWorkspaceBytes_ = 0;
    sharedCaptureWorkspaceDevice_ = -1;
  }
  // Free pre-allocated cuBLAS workspace
  if (cublasWorkspaceBuffer_ != nullptr && cublasWorkspaceDevice_ >= 0) {
    memory::CudaMemoryPool::getInstance().free(cublasWorkspaceBuffer_, cublasWorkspaceDevice_);
    cublasWorkspaceBuffer_ = nullptr;
    cublasWorkspaceSize_ = 0;
    cublasWorkspaceDevice_ = -1;
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

  // Clear any sticky CUDA errors accumulated during teardown.
  // Without this, the next plan on this thread inherits the error —
  // cudaStreamBeginCapture fails immediately with the stale error,
  // the capture abort path can't recover, and the process gets SIGABRT.
  cudaGetLastError();
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

void NativeDynamicShapePlan::platformFreeCaptureWorkspace() {
  if (sharedCaptureWorkspace_ != nullptr) {
    if (sd::graph::dspIsGlobalCaptureWorkspace(sharedCaptureWorkspace_)) {
      DSP_DIAG(MEMORY, "~NativeDynamicShapePlan: releasing reference to GLOBAL capture workspace %zuMB (NOT freeing)",
               sharedCaptureWorkspaceBytes_ / (1024*1024));
    } else {
      sd::graph::dspFreeWorkspaceOnPool(sharedCaptureWorkspace_);
      DSP_DIAG(MEMORY, "~NativeDynamicShapePlan: freed SHARED capture workspace %zuMB on device %d",
               sharedCaptureWorkspaceBytes_ / (1024*1024), sharedCaptureWorkspaceDevice_);
    }
    sharedCaptureWorkspace_ = nullptr;
    sharedCaptureWorkspaceBytes_ = 0;
    sharedCaptureWorkspaceDevice_ = -1;
  }
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
  // Clear any sticky CUDA errors from previous plan teardown BEFORE creating
  // the owned stream.  Without this, cudaStreamCreateWithFlags fails with the
  // stale error, ownedStream_ stays null, ctx->dspStream inherits the Java
  // caller's stale stream pointer (from the destroyed previous plan), and
  // cudaEventRecord in platformEndExecution hits error 400 (invalid handle).
  cudaGetLastError();

  // A top-level DSP execute must not preserve a stale tl_dspExecutionStream
  // from an earlier plan. DspStreamGuard restores the previous TLS value on
  // destruction; if that previous value points at a destroyed ownedStream_,
  // subsequent Java-side putScalar()/syncToDevice() calls can enqueue work on
  // the dead stream and the next first-execute warmup reads zero/stale inputs.
  // Active capture/replay scopes legitimately own the TLS, so only clear the
  // idle leak state before installing this plan's guard.
  if (tl_dspExecutionStream != nullptr && !tl_graphExecutionActive &&
      !tl_dspReplayActive && tl_graphCaptureStream == nullptr) {
    DSP_DIAG(EXECUTE,
             "platformBeginExecution: clearing stale tl_dspExecutionStream=%p before installing plan guard",
             tl_dspExecutionStream);
    tl_dspExecutionStream = nullptr;
  }

  // ── Lazy-create plan-owned CUDA stream ─────────────────────────────────
  // Each plan gets its own stream so that CUDA graph captures (which
  // happen on this stream) don't conflict with Java-side syncToDevice()
  // calls on the shared default stream from other threads. The stream
  // is created once and reused for all subsequent executions.
  if (ownedStream_ == nullptr) {
    ownedStream_ = new cudaStream_t();
    auto err = cudaStreamCreateWithFlags(ownedStream_, cudaStreamNonBlocking);
    if (err != cudaSuccess) {
      DSP_DIAG(EXECUTE, "platformBeginExecution: failed to create plan-owned stream: %s (%d)",
               cudaGetErrorString(err), static_cast<int>(err));
      cudaGetLastError();  // clear this error too
      delete ownedStream_;
      ownedStream_ = nullptr;
    } else {
      DSP_DIAG(EXECUTE, "platformBeginExecution: created plan-owned stream=%p",
               static_cast<void*>(*ownedStream_));
    }
  }
  // If caller passed the old default stream but we have our own, override it.
  // The Java executor caches the stream from getExecutionStream() — on the
  // first call it gets the default (ownedStream_ not yet created), but on
  // subsequent calls getExecutionStream() returns ownedStream_. We must also
  // override here for the first execution where the caller still holds
  // the old default stream pointer.
  if (ownedStream_ != nullptr) {
    stream = reinterpret_cast<void*>(ownedStream_);
  } else {
    // ownedStream_ creation failed above (e.g. a stale CUDA error tripped
    // cudaStreamCreateWithFlags). Do NOT fall through to dereferencing the
    // caller-passed `stream`: the Java executor CACHES that pointer across
    // executions (DynamicShapePlanExecutor.cachedExecStream). It can dangle —
    // pointing at a previously-destroyed ownedStream_ (platformFreePlanResources
    // does `delete ownedStream_`) or into a thread-local ContextBuffers that was
    // since released. Dereferencing it yields a dead stream, and every pool
    // alloc/free + kernel stream-sync then fails with CUDA 201
    // (cudaErrorDeviceUninitialized). Resolve the LIVE current-thread stream
    // instead — getCudaStream() always returns a stream valid for this thread.
    auto* liveStreamPtr = LaunchContext::defaultContext()->getCudaStream();
    stream = (liveStreamPtr != nullptr) ? reinterpret_cast<void*>(liveStreamPtr) : nullptr;
    DSP_DIAG(EXECUTE,
             "platformBeginExecution: ownedStream_ unavailable — using live thread-local "
             "stream=%p instead of caller-cached pointer (avoids stale-stream CUDA 201)",
             stream);
  }

  // Wait if another thread is capturing a CUDA graph on this device.
  // During capture, ANY CUDA operation on this device from another thread
  // (including cudaMemcpyAsync on the legacy stream) triggers error 906.
  //
  // TOCTOU fix: hold the mutex across the check-and-increment so no thread
  // can slip past the gate between reading g_captureActive and incrementing
  // g_execCount.  Without the mutex, a capturing thread could set
  // g_captureActive *after* we read it as false but *before* we increment
  // g_execCount — the capturing thread would then see g_execCount==0
  // and start capture while this thread is still executing.
  {
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= 16) dev = 0;
    {
      std::unique_lock<std::mutex> lk(g_captureMtx[dev]);
      g_captureCV[dev].wait(lk, [dev]{ return !g_captureActive[dev].load(std::memory_order_acquire); });
      g_execCount[dev].fetch_add(1, std::memory_order_acq_rel);
    }
  }

  // Safety reset: clear stale TLS from any prior crashed execution that
  // didn't reach platformEndExecution. Without this, a crash in config A
  // leaves tl_cublasLtDisabled=true, poisoning every subsequent config.
  if (tl_cublasLtDisabled) {
    tl_cublasLtDisabled = false;
    auto* handlePtr = reinterpret_cast<cublasHandle_t*>(CublasHelper::getInstance().handle());
    if (handlePtr != nullptr) {
      cublasSetMathMode(*handlePtr, CUBLAS_DEFAULT_MATH);
    }
    // Leaked flag = a prior begin never reached its end; balance the
    // deterministic window as well (clamped at zero if already closed).
    CublasHelper::exitDeterministicWindow();
  }

  auto* ctx = new PlanExecutionContext();
  ctx->execCount = execCount;
  ctx->frozen = frozen;

  // Capture the CUDA device for this execution. All events and streams
  // created during this execute() must be on this device. The DspStreamGuard
  // pins the device via cudaSetDevice and restores on destruction, so even
  // if an op temporarily switches devices, the guard's destructor restores.
  cudaGetDevice(&ctx->deviceId);

  // Sync decisions use anySegmentNeedsWarmup() — the SINGLE source of truth
  // for whether segments need warmup after invalidateSegmentCaptures.
  bool segWarmup = anySegmentNeedsWarmup();
  ctx->needsFullSync = !frozen || execCount <= 1 || segWarmup;
  ctx->isFrozenSteadyState = frozen && execCount > 1 && !segWarmup;

  // Create the per-execution cross-stream sync event.
  // This replaces the file-scope thread_local tl_crossStreamEvent so each
  // execute() call owns its event and there is no hidden per-thread state.
  {
    cudaEvent_t tmpEvt = nullptr;
    cudaEventCreateWithFlags(&tmpEvt, cudaEventDisableTiming);
    ctx->crossStreamEvent = static_cast<void*>(tmpEvt);
  }

  // Resolve CUDA streams and set up DspStreamGuard RAII
  if (stream != nullptr) {
    cudaStream_t cudaStr = *static_cast<cudaStream_t*>(stream);
    ctx->dspStream = static_cast<void*>(cudaStr);
    // Pass deviceId to DspStreamGuard so it pins cudaSetDevice for the
    // duration of execution and restores the previous device on destruction.
    ctx->streamGuard = static_cast<void*>(new DspStreamGuard(cudaStr, ctx->deviceId));

    // Resolve LC default stream (a real async stream from ContextBuffers,
    // NOT CUDA stream 0). Post-execution ops (KvScatter, assign, mask updates)
    // run on this stream.
    auto* lcStreamPtr = LaunchContext::defaultContext()->getCudaStream();
    ctx->lcDefaultStream = static_cast<void*>((lcStreamPtr != nullptr) ? *lcStreamPtr : nullptr);
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
             ctx->dspStream, ctx->lcDefaultStream);

    // Cross-stream sync: make the DSP stream wait for both the LC default
    // stream and CUDA stream 0 before starting execution.
    // Advances PreReplaySyncPhase to CROSS_STREAM_DONE so performPreReplaySync
    // (called later from dispatchSegment) skips the redundant cross-stream sync.
    {
      cudaEvent_t evt = reinterpret_cast<cudaEvent_t>(ctx->crossStreamEvent);
      cudaStream_t dspStr = reinterpret_cast<cudaStream_t>(ctx->dspStream);
      cudaStream_t lcStr  = reinterpret_cast<cudaStream_t>(ctx->lcDefaultStream);
      // 1) LC default stream → DSP stream
      if (lcStr != nullptr && lcStr != dspStr) {
        cudaEventRecord(evt, lcStr);
        cudaStreamWaitEvent(dspStr, evt, 0);
      }
      // 2) CUDA stream 0 → DSP stream (cuBLAS default handle, misc)
      cudaEventRecord(evt, nullptr);
      cudaStreamWaitEvent(dspStr, evt, 0);
      ctx->recordEventSync();  // Track: cross-stream event ordering at entry
      ctx->markCrossStreamSynced();  // Advance sync phase — single source of truth
      DSP_DIAG(EXECUTE, "platformBeginExecution: cross-stream sync done (syncPhase=%s)",
               ctx->syncPhaseName());
    }
    if (ctx->needsFullSync) {
      // Early executions used to drain the DSP stream here. Stream-ordering is
      // sufficient: prior DSP work on the same stream naturally precedes this
      // step, and cross-stream producers were ordered above with events.
      DSP_DIAG(EXECUTE,
               "platformBeginExecution: early execution uses async event ordering "
               "(no blocking stream sync)");
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
  // Modes requiring deterministic cuBLAS enforce PEDANTIC_MATH + workspace + no Lt.
  // A workspace MUST be provided: CUBLAS_PEDANTIC_MATH with no workspace causes
  // CUBLAS_GEMM_DEFAULT to produce all-zeros for FP16 inputs on some GPUs because
  // the only PEDANTIC-compatible algorithm for that precision needs workspace.
  // TRITON composite mode manages its own workspace/algorithm lifecycle.
  if (ModeContract::forMode(graphExecutionMode_).requiresDeterministicCublas) {
    auto* handlePtr = reinterpret_cast<cublasHandle_t*>(CublasHelper::getInstance().handle());
    if (handlePtr != nullptr) {
      // (1) Force bitwise-reproducible algorithms
      cublasSetMathMode(*handlePtr, CUBLAS_PEDANTIC_MATH);
      // (2) Provide workspace — required for PEDANTIC + FP16 algorithm selection.
      // ensureCublasWorkspace is idempotent (allocates once).
      ensureCublasWorkspace(sd::Environment::getInstance().dspCublasWorkspaceMb() * 1024ULL * 1024ULL);
      if (cublasWorkspaceBuffer_ != nullptr) {
        cublasSetWorkspace(*handlePtr, cublasWorkspaceBuffer_, cublasWorkspaceSize_);
        tl_cublasWorkspacePtr = cublasWorkspaceBuffer_;
        tl_cublasWorkspaceSize = cublasWorkspaceSize_;
      } else {
        cublasSetWorkspace(*handlePtr, nullptr, 0);
        tl_cublasWorkspacePtr = nullptr;
        tl_cublasWorkspaceSize = 0;
      }
    }
    // (3) Block cublasLt and force CUBLAS_GEMM_DEFAULT
    tl_cublasLtDisabled = true;
    // (4) Open the process-global deterministic window: cuBLAS handles are
    // THREAD-LOCAL, so the PEDANTIC set above only covers THIS thread. Gap
    // GEMMs dispatched from executor/pool threads acquire their own handles —
    // CublasHelper::handle() applies PEDANTIC to any handle acquired while
    // the window is open (task #55: off-thread GEMMs ran DEFAULT/TF32 against
    // the PEDANTIC reference — bit-identical batch-only drift).
    CublasHelper::enterDeterministicWindow();
    DSP_DIAG(EXECUTE, "platformBeginExecution: deterministic cuBLAS for mode=%d "
             "(PEDANTIC_MATH + workspace=%p size=%zuMB + no Lt + global window)",
             static_cast<int>(graphExecutionMode_),
             cublasWorkspaceBuffer_, cublasWorkspaceSize_ / (1024*1024));
  }

  // Reset FP16 cast-cache indices at plan execution boundary.
  // Two interleaved plans share the same thread-local cast cache
  // (tl_castA/tl_castB). Without resetting, plan2 inherits
  // plan1's stale index and reads wrong HALF-cast buffers, causing
  // maxDiff=83+ in mixed-precision FP16 matmuls.
  MmulHelper::resetCastCacheIndices();

  return static_cast<void*>(ctx);
}

void NativeDynamicShapePlan::platformEndExecution(void* executionState, void* stream, bool frozen, int execCount) {
  auto* ctx = static_cast<PlanExecutionContext*>(executionState);

  // Cross-stream synchronization: make post-execution streams wait for DSP.
  if (stream != nullptr) {
    DSP_DIAG(EXECUTE, "platformEndExecution: frozen=%d execCount=%d syncLevel=%s "
             "dspStream=%p lcDefault=%p deviceId=%d",
             static_cast<int>(ctx->frozen), ctx->execCount, ctx->syncLevelName(),
             ctx->dspStream, ctx->lcDefaultStream,
             ctx->deviceId);

    // Ensure we're on the correct device before creating/recording events.
    // An op during slot execution may have temporarily switched devices
    // (e.g., cross-device memory failover). Events and streams must match.
    cudaSetDevice(ctx->deviceId);

    // Clear any sticky CUDA error from a failed slot execution (e.g., OOM
    // causing error 700 — illegal memory access). Without this, every CUDA
    // call below (cudaEventCreateWithFlags, cudaEventRecord, etc.) would
    // inherit the sticky error and crash the process.
    auto stickyErr = cudaGetLastError();
    bool cudaContextHealthy = (stickyErr == cudaSuccess);
    if (!cudaContextHealthy) {
      DSP_DIAG(EXECUTE, "platformEndExecution: cleared sticky CUDA error: %s — skipping event sync",
               cudaGetErrorString(stickyErr));
    }

    if (cudaContextHealthy) {
      // Re-create executionCompleteEvent_ if it was created on a different device.
      if (executionCompleteEvent_ != nullptr && executionCompleteEventDeviceId_ != ctx->deviceId) {
        cudaEvent_t oldEvt = reinterpret_cast<cudaEvent_t>(executionCompleteEvent_);
        // Destroy on the device it was created on
        int savedDev;
        cudaGetDevice(&savedDev);
        cudaSetDevice(executionCompleteEventDeviceId_);
        cudaEventDestroy(oldEvt);
        cudaSetDevice(savedDev);
        executionCompleteEvent_ = nullptr;
        executionCompleteEventDeviceId_ = -1;
      }

      if (executionCompleteEvent_ == nullptr) {
        cudaEvent_t evt;
        auto createErr = cudaEventCreateWithFlags(&evt, cudaEventDisableTiming);
        if (createErr != cudaSuccess) {
          cudaGetLastError();  // clear
          DSP_DIAG(EXECUTE, "platformEndExecution: cudaEventCreateWithFlags failed: %s — skipping event sync",
                   cudaGetErrorString(createErr));
          cudaContextHealthy = false;
        } else {
          executionCompleteEvent_ = reinterpret_cast<void*>(evt);  // handle-value (consistent with dspCreateEvent)
          executionCompleteEventDeviceId_ = ctx->deviceId;
        }
      }
    }

    if (cudaContextHealthy && executionCompleteEvent_ != nullptr) {
      cudaEvent_t evt = reinterpret_cast<cudaEvent_t>(executionCompleteEvent_);
      cudaStream_t dspStr = reinterpret_cast<cudaStream_t>(ctx->dspStream);
      cudaStream_t lcStr  = reinterpret_cast<cudaStream_t>(ctx->lcDefaultStream);
      cudaEventRecord(evt, dspStr);
      sd::dspPublishThreadCompletionEvent(ctx->dspStream);
      // Make BOTH CUDA stream 0 AND the LC default stream wait for DSP.
      // Post-execution ops (KvScatter, assign, etc.) run on the LC default
      // stream. Without this ordering, they read outputs the DSP stream
      // hasn't finished writing yet. This is async and applies to warmup,
      // capture, and replay uniformly.
      cudaStreamWaitEvent(nullptr, evt, 0);  // CUDA stream 0
      if (lcStr != nullptr && lcStr != dspStr) {
        cudaStreamWaitEvent(lcStr, evt, 0);
        DSP_DIAG(EXECUTE, "platformEndExecution: lcDefault=%p waiting on DSP=%p",
                 ctx->lcDefaultStream, ctx->dspStream);
      }
      ctx->recordEventSync();  // Track: event-based ordering at exit
    }
  }

  // Destroy the per-execution cross-stream sync event.
  if (ctx->crossStreamEvent != nullptr) {
    cudaEventDestroy(reinterpret_cast<cudaEvent_t>(ctx->crossStreamEvent));
    ctx->crossStreamEvent = nullptr;
  }

  // Restore cuBLAS state for modes that enforced deterministic cuBLAS.
  if (ModeContract::forMode(graphExecutionMode_).requiresDeterministicCublas) {
    // Clear workspace from handle and TLS (workspace buffer itself is kept for reuse)
    auto* restoreHandle = reinterpret_cast<cublasHandle_t*>(CublasHelper::getInstance().handle());
    if (restoreHandle != nullptr && tl_cublasWorkspacePtr != nullptr) {
      cublasSetWorkspace(*restoreHandle, nullptr, 0);
    }
    tl_cublasWorkspacePtr = nullptr;
    tl_cublasWorkspaceSize = 0;
    // Get handle while tl_cublasLtDisabled is still true — this suppresses
    // the lazy-TF32 logic in CublasHelper::handle() so it doesn't overwrite
    // our restore below with a stale TF32/DEFAULT mode.
    auto* handlePtr = reinterpret_cast<cublasHandle_t*>(CublasHelper::getInstance().handle());
    if (handlePtr != nullptr) {
      cublasSetMathMode(*handlePtr, CUBLAS_DEFAULT_MATH);
    }
    // Clear AFTER math mode restore — the next CublasHelper::handle() call
    // from non-DSP code will see tl_cublasLtDisabled=false and correctly
    // lazy-apply TF32 if wanted.
    tl_cublasLtDisabled = false;
    // Close the deterministic window opened by platformBeginExecution.
    // Other threads' handles converge back to TF32/DEFAULT on their next
    // acquisition (lazy, per-thread).
    CublasHelper::exitDeterministicWindow();
  }

  // ── TLS STATE CLEANUP + ASSERTIONS ─────────────────────────────────────
  // Verify thread-local state consistency at execution boundary.
  // These catch state leaks: if any TLS was set during execution but not
  // properly restored, it poisons subsequent non-DSP operations.

  // tl_dspReplayActive MUST be false when execute() returns to Java.
  // The DspReplayGuard RAII in compositeReplay() should restore it, but if
  // any code path leaks (exception, longjmp, signal), the guard may not run.
  // A leaked tl_dspReplayActive=true causes syncToPrimary() to skip D2H
  // transfers for ALL subsequent DataBuffer reads on this thread — including
  // the output copy that Java does immediately after execute() returns.
  // Result: Java reads uninitialized zeros from the host buffer.
  if (tl_dspReplayActive) {
    DSP_DIAG(EXECUTE, "TLS_CLEANUP: tl_dspReplayActive=true at platformEndExecution — "
             "force-resetting (mode=%d execCount=%d). "
             "DspReplayGuard failed to restore — investigate the leak.",
             static_cast<int>(graphExecutionMode_), execCount);
    tl_dspReplayActive = false;
  }

  REQUIRE_TRUE(!tl_graphExecutionActive, 0,
               "TLS LEAK: tl_graphExecutionActive=true at platformEndExecution exit. "
               "A graph capture began but was not properly ended. "
               "mode=%d execCount=%d frozen=%d",
               static_cast<int>(graphExecutionMode_), execCount, static_cast<int>(frozen));
  // tl_cublasLtDisabled should be false by now (restored above for SBS/CG modes,
  // never set for TRITON/other modes). If a prior execution crashed before
  // platformEndExecution, this TLS may be stale. Force-reset it to prevent
  // cascading failures into subsequent configs.
  if (tl_cublasLtDisabled) {
    DSP_DIAG(EXECUTE, "TLS_CLEANUP: tl_cublasLtDisabled=true at platformEndExecution — "
             "force-resetting (mode=%d). Likely leaked from a prior crashed execution.",
             static_cast<int>(graphExecutionMode_));
    tl_cublasLtDisabled = false;
    auto* handlePtr = reinterpret_cast<cublasHandle_t*>(CublasHelper::getInstance().handle());
    if (handlePtr != nullptr) {
      cublasSetMathMode(*handlePtr, CUBLAS_DEFAULT_MATH);
    }
    // The leaked flag implies a begin that never reached its end — balance
    // the deterministic window too (exit clamps at zero if already closed).
    CublasHelper::exitDeterministicWindow();
  }
  // Capture stream must be null — active capture would mean we're inside beginCapture
  // but exited execution without endCapture.
  // Defensive cleanup: clear instead of throwing. The capture lifecycle guards should
  // have cleared this already, but edge cases in OOM_DEFERRED/retry paths or the
  // NATIVE_ONLY_CAPTURE re-capture path may leave it set. Throwing here blocks all
  // subsequent execution on this plan even though the actual execution succeeded.
  if (tl_graphCaptureStream != nullptr) {
    DSP_DIAG(EXECUTE, "TLS CLEANUP: tl_graphCaptureStream=%p was non-null at "
             "platformEndExecution exit (execCount=%d). Clearing defensively.",
             (void*)tl_graphCaptureStream, executeCount_);
    tl_graphCaptureStream = nullptr;
  }

  // Explicitly delete the stream guard before the context.
  // DspStreamGuard restores tl_dspExecutionStream to its previous value.
  delete static_cast<DspStreamGuard*>(ctx->streamGuard);
  ctx->streamGuard = nullptr;
  delete ctx;

  // Decrement per-device execution counter and notify any waiting capture thread.
  {
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= 16) dev = 0;
    int prev = g_execCount[dev].fetch_sub(1, std::memory_order_acq_rel);
    if (prev <= 1) {
      // Last executor on this device — wake the capture thread if waiting
      g_captureCV[dev].notify_all();
    }
  }
}

void NativeDynamicShapePlan::platformSetDeterministicCublas(bool enable) {
  if (enable) {
    // Reset stale cublasLt state from prior non-DSP ops
    if (tl_cublasLtDisabled) {
      tl_cublasLtDisabled = false;
      auto* handlePtr = reinterpret_cast<cublasHandle_t*>(CublasHelper::getInstance().handle());
      if (handlePtr != nullptr) {
        cublasSetMathMode(*handlePtr, CUBLAS_DEFAULT_MATH);
      }
    }
    // Set deterministic cuBLAS: PEDANTIC_MATH + no workspace + no cublasLt
    auto* handlePtr = reinterpret_cast<cublasHandle_t*>(CublasHelper::getInstance().handle());
    if (handlePtr != nullptr) {
      cublasSetMathMode(*handlePtr, CUBLAS_PEDANTIC_MATH);
      cublasSetWorkspace(*handlePtr, nullptr, 0);
    }
    tl_cublasWorkspacePtr = nullptr;
    tl_cublasWorkspaceSize = 0;
    tl_cublasLtDisabled = true;
  } else {
    // Restore cuBLAS to default state
    tl_cublasWorkspacePtr = nullptr;
    tl_cublasWorkspaceSize = 0;
    tl_cublasLtDisabled = false;
    auto* handlePtr = reinterpret_cast<cublasHandle_t*>(CublasHelper::getInstance().handle());
    if (handlePtr != nullptr) {
      cublasSetMathMode(*handlePtr, CUBLAS_DEFAULT_MATH);
    }
  }
}

void NativeDynamicShapePlan::platformSetupSteadyStateCuda(void* execCtxVoid, void* stream) {
  auto* execCtx = static_cast<PlanExecutionContext*>(execCtxVoid);

  // Capture the CUDA device for this execution — all events created during
  // execute() must be on this device to avoid cross-device handle errors.
  execCtx->deviceId = sd::graph::dspGetCurrentDevice();

  // Reuse cross-stream event (avoid cudaEventCreate/Destroy per step)
  if (steadyStateCrossStreamEvent_ == nullptr) {
    steadyStateCrossStreamEvent_ = sd::graph::dspCreateEvent();
  }
  execCtx->crossStreamEvent = steadyStateCrossStreamEvent_;

  // Resolve CUDA streams and set DSP execution stream. `stream` is a STREAM-POINTER
  // (cudaStream_t* — autoregressive_decode and the JNI both pass a pointer); the
  // PlanExecutionContext (ctx->dspStream), tl_dspExecutionStream, and the dspXxx helpers
  // below all consume a STREAM-VALUE (see DspCudaDispatch.h). Convert once. Storing the raw
  // pointer here previously made every steady-state event sync (dspEventRecord/WaitEvent on
  // ctx->dspStream) operate on a host address instead of the real stream.
  if (stream != nullptr) {
    void* streamVal = sd::graph::dspStreamPtrToValue(stream);
    execCtx->dspStream = streamVal;
    sd::graph::dspSetExecutionStream(streamVal);

    execCtx->lcDefaultStream = sd::graph::dspGetLcDefaultStream();

    // Event-based cross-stream ordering in steady state.
    void* evt = steadyStateCrossStreamEvent_;
    if (execCtx->lcDefaultStream != nullptr && execCtx->lcDefaultStream != execCtx->dspStream) {
      sd::graph::dspEventRecord(evt, execCtx->lcDefaultStream);
      sd::graph::dspStreamWaitEvent(streamVal, evt);
    }
    sd::graph::dspEventRecord(evt, nullptr);  // CUDA stream 0
    sd::graph::dspStreamWaitEvent(streamVal, evt);
    // Advance sync phase so platformTryFrozenFastPath and compositeReplay skip the duplicate
    execCtx->markCrossStreamSynced();
  }

  // Deterministic cuBLAS for modes that require it (CUDA_GRAPHS, AUTO).
  if (ModeContract::forMode(graphExecutionMode_).requiresDeterministicCublas) {
    platformSetDeterministicCublas(true);
  }
}

void NativeDynamicShapePlan::platformTeardownSteadyStateCuda(void* execCtxVoid, void* stream, void* prevDspStream) {
  auto* execCtx = static_cast<PlanExecutionContext*>(execCtxVoid);

  // Event-based completion signal (reuses executionCompleteEvent_)
  if (stream != nullptr) {
    // Ensure correct device before event creation/recording (an op may
    // have temporarily switched devices during slot execution).
    sd::graph::dspSetCurrentDevice(execCtx->deviceId);

    // Clear any sticky CUDA error before event operations.
    int stickyErr = sd::graph::dspClearLastCudaError();
    bool cudaContextHealthy = (stickyErr == 0);

    if (cudaContextHealthy) {
      // Re-create executionCompleteEvent_ if it was created on a different device.
      if (executionCompleteEvent_ != nullptr && executionCompleteEventDeviceId_ != execCtx->deviceId) {
        int savedDev = sd::graph::dspGetCurrentDevice();
        sd::graph::dspSetCurrentDevice(executionCompleteEventDeviceId_);
        sd::graph::dspDestroyEvent(executionCompleteEvent_);
        sd::graph::dspSetCurrentDevice(savedDev);
        executionCompleteEvent_ = nullptr;
        executionCompleteEventDeviceId_ = -1;
      }

      if (executionCompleteEvent_ == nullptr) {
        void* newEvt = sd::graph::dspCreateEvent();
        if (newEvt == nullptr) {
          sd::graph::dspClearLastCudaError();
          cudaContextHealthy = false;
        } else {
          executionCompleteEvent_ = newEvt;
          executionCompleteEventDeviceId_ = execCtx->deviceId;
        }
      }
    }

    if (cudaContextHealthy && executionCompleteEvent_ != nullptr) {
      void* evtVoid = executionCompleteEvent_;
      sd::graph::dspEventRecord(evtVoid, execCtx->dspStream);
      sd::graph::dspPublishThreadCompletionEvent(execCtx->dspStream);
      sd::graph::dspStreamWaitEvent(nullptr, evtVoid);  // CUDA stream 0
      if (execCtx->lcDefaultStream != nullptr && execCtx->lcDefaultStream != execCtx->dspStream) {
        sd::graph::dspStreamWaitEvent(execCtx->lcDefaultStream, evtVoid);
      }
    }

    // Restore DspStreamGuard
    sd::graph::dspSetExecutionStream(prevDspStream);
  }

  // Restore cuBLAS state for modes that enforced deterministic cuBLAS.
  if (ModeContract::forMode(graphExecutionMode_).requiresDeterministicCublas) {
    platformSetDeterministicCublas(false);
  }
}

void NativeDynamicShapePlan::platformResetGapCaches() {
  gapPrezeroTargetsCached_ = false;
  cachedGapPrezeroCount_ = 0;
  activeGapSlotsCachedSet_.clear();
  cachedActiveGapSlotsMap_.clear();
}

void NativeDynamicShapePlan::platformResetBatchD2D() {
  batchD2DCount_ = 0;
}

void NativeDynamicShapePlan::platformDumpExternalInputDiagnostics(NDArray** ext, int numExt, int execCount) {
  if (!DSP_DIAG_ENABLED(EXECUTE)) return;
  for (int dbgI = 0; dbgI < numExt; dbgI++) {
    NDArray* arr = ext[dbgI];
    if (arr == nullptr || arr->dataType() != FLOAT32 || arr->lengthOf() <= 0) continue;
    auto* db = arr->dataBuffer();
    const char* nm = (dbgI < (int)externalInputNames_.size()) ? externalInputNames_[dbgI].c_str() : "?";
    DSP_DIAG(EXECUTE, "EXT_ENTRY execCount=%d ext[%d]='%s' arr=%p sbuf=%p len=%lld "
             "pAct=%d sAct=%d",
             execCount, dbgI, nm, (void*)arr, arr->specialBuffer(),
             (long long)arr->lengthOf(),
             db ? (db->isPrimaryActual() ? 1 : 0) : -1,
             db ? (db->isSpecialActual() ? 1 : 0) : -1);
  }
}

void NativeDynamicShapePlan::platformDumpExtInputGpuValues(NDArray* arr, int extIdx, int execCount, void* stream) {
  if (arr == nullptr) return;
  if (arr->specialBuffer() != nullptr && arr->lengthOf() > 0 && arr->dataType() == FLOAT32) {
    DSP_DIAG(VERIFY, "EXT_INPUT_START: exec=%d extIdx=%d len=%lld sbuf=%p "
                     "(async path: value dump skipped)",
             execCount, extIdx, (long long)arr->lengthOf(), arr->specialBuffer());
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

  // Find the logits output metadata without blocking the DSP stream.
  for (int i = 0; i < numRequestedOutputs_; i++) {
    int slotIdx = requestedOutputSlotIndices_[i];
    NDArray* arr = (slotIdx >= 0 && slotIdx < totalOutputSlots_) ? outputSlots_[slotIdx] : nullptr;
    if (arr == nullptr) continue;
    void* sbuf = arr->specialBuffer();
    // Logits: FLOAT32, length >= 10000 (any reasonable vocab), rank <= 3
    if (sbuf && arr->dataType() == FLOAT32 && arr->lengthOf() >= 10000 && arr->rankOf() <= 3) {
      DSP_DIAG_SLOT(VERIFY, slotIdx,
          "LOGITS_ARGMAX exec=%d reqOut[%d] len=%lld sbuf=%p "
          "(async path: argmax dump skipped)",
          execCount, i, (long long)arr->lengthOf(), sbuf);
    }
  }
}

void NativeDynamicShapePlan::platformDetectAndPrepareBatchedGemm(NDArray** ext, int numExt, void* stream) {
  if (!planLifecycle_.isSlotBySlot() && executeCount_ == 1 && batchedGemmGroups_.empty() &&
      Environment::getInstance().dspBatchedGemm()) {
    detectBatchedGemmGroups(ext, numExt);
    if (!batchedGemmGroups_.empty()) {
      // prepareBatchedGemmDevice reinterpret_casts its void* to cudaStream_t (stream VALUE),
      // so convert the JNI pointer param to a value first (matches executeBatchedGemmGroup).
      prepareBatchedGemmDevice(sd::graph::dspStreamPtrToValue(stream));
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
      if (gpuPtr != nullptr && arr->lengthOf() > 0 && arr->dataType() == FLOAT32) {
        DSP_DIAG(VERIFY, "SLOT_TRACE after seg[%d-%d]: slot=%d "
                "arr=%p gpuPtr=%p db=%p closed=%d pAct=%d sAct=%d "
                "len=%lld execCount=%d (async path: value dump skipped)",
                seg.def.startSlot, seg.def.endSlot, traceSlot,
                (void*)arr, gpuPtr, (void*)db,
                db ? db->isClosed() : -1,
                db ? (db->isPrimaryActual() ? 1 : 0) : -1,
                db ? (db->isSpecialActual() ? 1 : 0) : -1,
                (long long)arr->lengthOf(),
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

size_t NativeDynamicShapePlan::platformEstimateSegmentCaptureBytes(int startSlot, int endSlot) const {
  // During cudaStreamCapture every intermediate output a segment's slots produce is
  // allocated FROM the capture workspace (cudaMalloc is illegal mid-capture) and persists
  // (DSP keeps one array per slot), so the workspace must hold their aligned total. Add a
  // margin for transient native-op temporaries (matmul tiles, concat/attention staging).
  size_t total = 0;
  for (int s = startSlot; s <= endSlot && s < totalOutputSlots_; s++) {
    const NativeSlot& slot = slots_[s];
    for (int o = 0; o < slot.wiring.numOutputs; o++) {
      int outIdx = slot.wiring.outputSlotIndices[o];
      if (outIdx >= 0 && outIdx < totalOutputSlots_ && outputSlots_[outIdx] != nullptr) {
        size_t bytes = static_cast<size_t>(outputSlots_[outIdx]->lengthOf()) *
                       static_cast<size_t>(outputSlots_[outIdx]->sizeOfT());
        total += (bytes + 255ULL) & ~static_cast<size_t>(255ULL);  // 256B-aligned bump allocator
      }
    }
  }
  size_t withMargin = total + total / 2;  // +50% headroom for transient capture temporaries
  DSP_DIAG(MEMORY, "platformEstimateSegmentCaptureBytes: seg[%d-%d] slotOutputs=%zuMB withMargin=%zuMB",
           startSlot, endSlot, total / (1024*1024), withMargin / (1024*1024));
  return withMargin;
}

void NativeDynamicShapePlan::platformReleaseSegmentGpuResources() {
  if (segments_.empty()) {
    DSP_DIAG(MEMORY, "platformReleaseSegmentGpuResources: no segments — nothing to release");
    return;
  }
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
    seg.exec.markArgsStale();
    seg.exec.resetCaptureKeys();
    SegmentLifecycle::resetForResourceRelease(seg.exec);
    seg.exec.executionCount = 0;
    delete seg.exec.jitKernel;
    seg.exec.jitKernel = nullptr;
    seg.exec.captureOomRetries = 0;
    seg.exec.captureRetryAfterExec = 0;
    seg.exec.compiledByBackend.clear();
    SegmentLifecycle::initSegmentPhase(seg.exec, seg.def.startSlot, seg.def.endSlot);
    seg.exec.jitShapeKey = 0;
    seg.exec.jitCompileFailed = false;
    seg.def.shapeKeyState.reset();
  }
  logGpuMemState("STEP-1-AFTER-SEGMENTS");

  // Free cuBLAS workspace
  if (cublasWorkspaceBuffer_ != nullptr && cublasWorkspaceDevice_ >= 0) {
    memory::CudaMemoryPool::getInstance().free(cublasWorkspaceBuffer_, cublasWorkspaceDevice_);
    cublasWorkspaceBuffer_ = nullptr;
    cublasWorkspaceSize_ = 0;
    cublasWorkspaceDevice_ = -1;
  }

  // Free batch-D2D and batched-GEMM device arrays
  freeBatchD2DResources();
  freeBatchedGemmResources();
  logGpuMemState("STEP-1-AFTER-BATCH-RESOURCES");
}

void NativeDynamicShapePlan::platformMigrateWeightsAndClearCaches() {
  DSP_DIAG(MEMORY, "releaseGpuIntermediates: freed intermediate NDArrays");
  logGpuMemState("STEP-2-AFTER-INTERMEDIATES");

  // NOTE: untrackedOutputCache_ cleanup is handled by step 4e in
  // releaseGpuIntermediates() (NativeDynamicShapePlan.cpp) which uses the
  // safe pattern (deleteBuffers + setShapeInfo(nullptr) before delete).
  // Raw delete here corrupts the heap: ~NDArray() tries to free() GPU
  // device pointers and ConstantShapeBuffer* interiors that were never
  // standalone malloc'd blocks.

  // Reset MmulHelper cast-cache INDICES only (do NOT free the NDArray buffers).
  // platformMigrateWeightsAndClearCaches() runs during releaseGpuIntermediates()
  // while the plan's CUDA graphs are still live.  Those graphs have tl_castB /
  // tl_castA device addresses BAKED as kernel arguments — freeing the NDArrays
  // here causes a stale-pointer read (NaN) on the next replay of ANY plan that
  // shares this thread-local cache.  Index-only reset is the same safe pattern
  // used at phaseFreeze / phaseWarmup boundaries (MmulHelper.cu ~231).
  // The actual NDArray memory is freed in the destructor (platformFreePlanResources
  // line ~1544) after all CUDA graphs for this plan have been torn down.
  MmulHelper::resetCastCacheIndices();
  logGpuMemState("STEP-4-AFTER-CAST-CACHE");

  // Migrate weight buffers from async pool to direct cudaMalloc
  {
    cudaGetLastError();
    int deviceId = 0;
    cudaGetDevice(&deviceId);

    int migratedCount = 0;
    int skippedDirect = 0;
    int skippedStillFrozen = 0;
    int skippedNonDevice = 0;
    int failedMigrations = 0;
	    size_t migratedBytes = 0;
	    size_t totalWeightBytes = 0;
	    auto& pool = memory::CudaMemoryPool::getInstance();
	    auto* lcStreamPtr = LaunchContext::defaultContext()->getCudaStream();
	    cudaStream_t migrationStream = (lcStreamPtr != nullptr) ? *lcStreamPtr : nullptr;

    DSP_DIAG(MEMORY, "Weight migration: %zu protected weight buffers to check",
             protectedWeightBuffers_.size());

    for (auto* db : protectedWeightBuffers_) {
      if (db == nullptr || db->special() == nullptr) continue;

      size_t bufSize = db->getLenInBytes();
      totalWeightBytes += bufSize;

      // Multiple cached plans can share immutable constant/weight DataBuffers.
      // After this plan drops its own frozen refs, a non-zero count means another
      // plan still has CUDA graph/slot contexts baked against the current pointer.
      // Teardown migration must defer until the last frozen owner releases it.
      if (db->isFrozenPlanRegistered()) {
        skippedStillFrozen++;
        DSP_DIAG(MEMORY,
            "Weight migration: deferring DataBuffer %p (%zu bytes) because "
            "another frozen plan still owns its special pointer",
            static_cast<void*>(db), bufSize);
        continue;
      }

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

      // Capture-safe persistent allocation via the pool (cudaMallocAsync on a dedicated
      // non-capturing stream) — no raw cudaMalloc. allocateDirect() tracks the pointer
      // for cudaFreeAsync routing, so no registerDirectAllocation is needed below.
      // NOTE(perf): this comes from the shared default mempool; for full weight/pool
      // trim-separation a dedicated mempool for allocateDirect is a follow-up.
      void* directPtr = pool.allocateDirect(bufSize, deviceId);
      if (directPtr == nullptr) {
        failedMigrations++;
        DSP_DIAG(MEMORY,
            "Weight migration FAILED for %zu bytes (%zu MB): allocateDirect returned null",
            bufSize, bufSize / (1024*1024));
        continue;
      }

	      cudaError_t copyErr = cudaMemcpyAsync(directPtr, db->special(), bufSize,
	                                            cudaMemcpyDeviceToDevice, migrationStream);
	      if (copyErr != cudaSuccess) {
	        pool.free(directPtr, deviceId, migrationStream);
	        cudaGetLastError();
        DSP_DIAG(MEMORY, "releaseGpuIntermediates: weight migration memcpy failed for %zu bytes: %s",
                 bufSize, cudaGetErrorString(copyErr));
        continue;
	      }

	      void* oldPtr = db->special();
	      pool.free(oldPtr, deviceId, migrationStream);
      db->replaceSpecialBuffer(directPtr, true);
      // allocateDirect() already tracks directPtr for capture-safe cudaFreeAsync routing;
      // no separate registerDirectAllocation is needed here.

      migratedCount++;
      migratedBytes += bufSize;
    }

    DSP_DIAG(MEMORY,
        "Weight migration summary: total=%zu MB, migrated=%d (%zu MB), "
        "skippedDirect=%d, skippedStillFrozen=%d, skippedNonDevice=%d, failed=%d",
        totalWeightBytes / (1024*1024), migratedCount, migratedBytes / (1024*1024),
        skippedDirect, skippedStillFrozen, skippedNonDevice, failedMigrations);

    pool.trimPool(deviceId);
    logGpuMemState("STEP-4b-AFTER-MIGRATION-AND-TRIM");

    // Shape/TAD helper caches are process-wide metadata caches. They can back
    // shape-info pointers on Java-owned arrays that are still alive while a
    // native plan is being torn down, so a per-plan cleanup path must not clear
    // them. Trimming the CUDA pool after weight migration is still safe.
    pool.trimPool(deviceId);
    logGpuMemState("STEP-4c-AFTER-WEIGHT-MIGRATION-TRIM");
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

int NativeDynamicShapePlan::copyStagingToBuffer(int extIdx, sd::DataBuffer* dstDataBuffer) {
  if (placeholderStagingBuffers_ == nullptr || extIdx < 0 || extIdx >= numExternalInputs_)
    return -1;
  NDArray* staging = placeholderStagingBuffers_[extIdx];
  if (staging == nullptr) return -1;

  auto* srcDb = staging->dataBuffer();
  if (srcDb == nullptr || srcDb->isClosed()) return -2;
  if (dstDataBuffer == nullptr) return -3;

  // Just-in-time staging sync: during warmup after markExternalInputVariable,
  // the staging buffer is pre-allocated but zero-filled (ensureAndSyncStagingBuffers
  // only runs during capture/replay, not during warmup slot-by-slot). Sync from the
  // last external input to the staging buffer now so the caller reads fresh data.
  NDArray* lastExt = getLastExternalInput(extIdx);
  if (lastExt != nullptr && !lastExt->isEmpty() && lastExt->lengthOf() > 0) {
    std::vector<NDArray*> writes{staging};
    std::vector<NDArray*> reads{lastExt};
    NDArray::prepareSpecialUse(writes, reads);
    void* stagingDev = staging->dataBuffer() != nullptr ? staging->dataBuffer()->special() : nullptr;
    void* extDev = lastExt->dataBuffer() != nullptr ? lastExt->dataBuffer()->special() : nullptr;
    if (stagingDev != nullptr && extDev != nullptr && stagingDev != extDev) {
      size_t bytes = static_cast<size_t>(lastExt->lengthOf()) * lastExt->sizeOfT();
      if (bytes > 0) {
        auto* streamPtr = LaunchContext::defaultContext()->getCudaStream();
        cudaStream_t cudaStr = (streamPtr != nullptr) ? *streamPtr : nullptr;
        cudaMemcpyAsync(stagingDev, extDev, bytes, cudaMemcpyDeviceToDevice, cudaStr);
      }
    }
    NDArray::registerSpecialUse(writes, reads);
  }

  // DataBuffer::memcpy does async D2D via cudaMemcpyAsync on captureSafeStreamOrDefault().
  sd::DataBuffer::memcpy(dstDataBuffer, srcDb, 0, 0, staging->lengthOf());

  // The async copy is on LaunchContext::defaultContext()->getCudaStream().
  // Order stream-0 consumers after it without blocking the host.
  auto* streamPtr = LaunchContext::defaultContext()->getCudaStream();
  if (streamPtr != nullptr) {
    cudaEvent_t copyDone = nullptr;
    cudaEventCreateWithFlags(&copyDone, cudaEventDisableTiming);
    cudaEventRecord(copyDone, *streamPtr);
    cudaStreamWaitEvent(nullptr, copyDone, 0);
    cudaEventDestroy(copyDone);
  }
  return 0;
}

}  // namespace graph
}  // namespace sd

#endif  // SD_CUDA

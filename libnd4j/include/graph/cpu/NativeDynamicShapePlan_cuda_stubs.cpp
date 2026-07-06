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
 * NativeDynamicShapePlan — CPU Platform Dispatch Stubs
 *
 * Provides CPU-only fallback implementations for the platform dispatch
 * methods declared in NativeDynamicShapePlan.h. On GPU builds, the real
 * implementations in the platform-specific .cu file are linked instead.
 *
 * For most GPU-only features (frozen graph fast path, GPU error checking,
 * capture buffers), these are no-ops that return safe defaults.
 * For features with real CPU fallbacks (KV scatter, segment execution),
 * real CPU implementations are provided.
 */

#include <graph/NativeDynamicShapePlan.h>
#include <graph/ModeContract.h>
#include <graph/PlanExecutionContext.h>
#include <graph/GraphBackend.h>
#include <graph/GraphReplayHandle.h>
#include <graph/cpu/FunctionalReplayHandle.h>
#include <graph/DspPhaseUtils.h>
#include <graph/DspSegmentLifecycle.h>
#include <config.h>

#include <cstdint>
#include <cstring>
#include <ops/declarable/OpRegistrator.h>

#if HAVE_ONEDNN
#include <graph/cpu/OneDnnGraphBackend.h>
#endif
#if HAVE_OPENVINO
#include <graph/cpu/OpenVinoGraphBackend.h>
#endif

namespace sd {
namespace graph {

using SegmentLifecycleState = GraphSegmentExec::SegmentLifecycleState;

// ── Frozen fast path: skip segment loop overhead on CPU ─────────────────────
//
// When shapes are frozen and all segments have resolved CPU backends,
// skip the full phaseReplay() segment iteration + lifecycle checks and
// go directly to backend execution. This eliminates per-step overhead
// from segment dispatch, phase guards, and diagnostic logging.
//
// On GPU this fast path launches a CUDA graph. On CPU it re-executes
// the compiled backend (OneDNN graph, OpenVINO model, MLIR JIT, etc.)
// directly — the compilation artifact is reused, only the compute runs.

Status NativeDynamicShapePlan::platformTryFrozenFastPath(
    NDArray** externalInputs, int numExternalInputs,
    NDArray** requestedOutputs, int numRequestedOutputs, void* stream) {

  // Soft preconditions — return MAYBE so the caller falls through to normal execution.
  if (ModeContract::forMode(graphExecutionMode_).isSlotBySlot || planLifecycle_.isSlotBySlot()) {
    return Status::MAYBE;
  }
  if (executeCount_ < 2) {
    return Status::MAYBE;
  }
  // The frozen fast path requires shapes to be frozen and all segments to have
  // ready replay handles. Without this, early executions with no handles fail.
  if (!planLifecycle_.isInFrozenOrReplayState() || !allSegmentsReplayReady()) {
    return Status::MAYBE;
  }
  if (!ModeContract::forMode(graphExecutionMode_).allowsFrozenFastPath) {
    return Status::MAYBE;
  }

  DSP_DIAG(EXECUTE, "CPU_FROZEN_FAST_PATH: segments=%d executeCount=%d",
           (int)segments_.size(), executeCount_);

  // Execute all segments using their resolved backends (OneDNN/OpenVINO)
  // when available, falling back to slot-by-slot for unresolved segments.
  // Graph backends fuse multiple ops into optimized subgraphs, dramatically
  // reducing per-op dispatch overhead (1761 individual calls → ~dozens of fused calls).
  for (auto& seg : segments_) {
    // All-frozen-constant segments: outputs already populated from warmup.
    if (seg.def.allFrozenConstants) {
      seg.exec.executionCount++;
      continue;
    }
    Status status;
    if (seg.resolvedCpuBackend != nullptr) {
      // Fast path: backend already compiled — execute via executeSegmentWithSpecificBackend
      // which installs the NativeSlotExecutor for native-deferred ops (rope, attention).
      // Calling backend->executeSegment() directly would skip NativeSlotExecutor setup
      // and cause native-deferred ops to fail with KERNEL_FAILURE.
      status = executeSegmentWithSpecificBackend(
          seg, seg.resolvedCpuBackend, externalInputs, numExternalInputs, stream);
      if (status != Status::OK) {
        // Backend that was working now failed — this is a real bug.
        // Mark it failed so we fall to slot-by-slot on subsequent calls.
        SegmentLifecycle::markFailed(seg.exec, "cpu_frozen_backend_failed",
                                     seg.def.startSlot, seg.def.endSlot);
        DSP_DIAG(EXECUTE, "CPU_FROZEN: backend failed seg[%d-%d], marked FAILED via lifecycle, "
                 "falling back to slot-by-slot",
                 seg.def.startSlot, seg.def.endSlot);
        status = executeSegmentSlotBySlot(seg, externalInputs, numExternalInputs, stream);
      }
    } else if (seg.def.selectedBackend == SelectedBackend::CPU_GRAPH &&
               !seg.exec.compilationFailed && !seg.exec.noFusibleOps) {
      // Lazy compile path: backend not yet resolved. Call the full cascade
      // (warmup → compile → execute) which populates resolvedCpuBackend.
      // This happens when platformPrecompileSegments skips segments with
      // executionCount==0 after freeze resets them.
      status = executeSegmentWithCpuGraph(seg, externalInputs, numExternalInputs, stream);
      if (status != Status::OK) {
        SegmentLifecycle::markFailed(seg.exec, "cpu_frozen_cascade_failed",
                                     seg.def.startSlot, seg.def.endSlot);
        DSP_DIAG(EXECUTE, "CPU_FROZEN: cascade failed seg[%d-%d], marked FAILED via lifecycle, "
                 "falling back to slot-by-slot",
                 seg.def.startSlot, seg.def.endSlot);
        status = executeSegmentSlotBySlot(seg, externalInputs, numExternalInputs, stream);
      }
    } else {
      status = executeSegmentSlotBySlot(seg, externalInputs, numExternalInputs, stream);
    }
    if (status != Status::OK) {
      return status;
    }
  }

  // Populate requestedOutputs from outputSlots_ — mirrors the CUDA fast path
  // (NativeDynamicShapePlan_cuda.cu lines 361-368). Without this, the caller
  // receives all-nullptr outputs and Java sees all-zero arrays.
  for (int i = 0; i < numRequestedOutputs; i++) {
    int slotIdx = requestedOutputSlotIndices_[i];
    if (slotIdx >= 0 && slotIdx < totalOutputSlots_) {
      requestedOutputs[i] = outputSlots_[slotIdx];
    } else {
      requestedOutputs[i] = nullptr;
    }
  }

  executeCount_++;
  return Status::OK;
}

// ── Pre-execute setup: no GPU-specific work on CPU ──────────────────────────

void NativeDynamicShapePlan::platformPreExecuteSetup(
    NDArray** externalInputs, int numExternalInputs, void* stream) {
  // No GPU-specific work on CPU. Arrays persist (one array per slot).
}

// ── Segment cache retention: check capturability on CPU ─────────────────────

bool NativeDynamicShapePlan::platformShouldKeepSegmentCache(const GraphSegment& seg) const {
  if (seg.exec.replayHandle && seg.exec.replayHandle->isReady()) return true;
  if (seg.def.isCapturable && !seg.exec.compilationFailed) return true;
  return false;
}

// ── Eager precompilation: compile all CPU graph backend segments ─────────────
//
// On GPU, this triggers parallel Triton/NVRTC compilation. On CPU, we eagerly
// resolve and compile CPU graph backends (OneDNN, OpenVINO, MLIR, etc.) for
// all segments during phaseCompile() instead of lazily on second execution.
// This moves compilation latency out of the hot path.

void NativeDynamicShapePlan::platformPrecompileSegments(
    NDArray** externalInputs, int numExternalInputs) {
  DSP_REQUIRE_PLAN_PHASE_AT_MOST(PlanPhase::SHAPES_FROZEN, "platformPrecompileSegments");

  const auto& chain = getCpuGraphBackendChain();
  if (chain.empty()) {
    DSP_DIAG(COMPILE, "platformPrecompileSegments: no CPU graph backends available, skipping");
    return;
  }

  int compiled = 0, skipped = 0, failed = 0;
  for (auto& seg : segments_) {
    // Skip segments that already have a resolved backend
    if (seg.resolvedCpuBackend != nullptr) {
      skipped++;
      continue;
    }
    // Skip non-capturable or already-failed segments
    if (!seg.def.isCapturable || seg.exec.compilationFailed) {
      skipped++;
      continue;
    }
    // Skip segments that haven't been warmed up (no shape info yet)
    if (seg.exec.executionCount == 0) {
      skipped++;
      continue;
    }

    // Try each backend in priority order
    bool resolved = false;
    for (size_t i = 0; i < chain.size(); i++) {
      GraphBackend* backend = chain[i];
      if (!backend->canFuseSegment(slots_, seg.def.startSlot, seg.def.endSlot)) {
        continue;
      }

      // Compute shape key for compilation
      LongType segShapeKey = computeSegmentShapeKey(seg, externalInputs, numExternalInputs);
      seg.def.shapeKeyState.recordComputed(segShapeKey);

      if (backend->compileSegment(seg, slots_, externalInputs, numExternalInputs,
                                   outputSlots_, totalOutputSlots_, segShapeKey,
                                   numSlots_)) {
        seg.resolvedCpuBackend = backend;
        seg.exec.cachedShapeKey = segShapeKey;
        seg.def.shapeKeyState.markCompiled(segShapeKey);
        compiled++;
        resolved = true;
        DSP_DIAG(COMPILE, "platformPrecompileSegments: seg[%d-%d] compiled by %s",
                 seg.def.startSlot, seg.def.endSlot, backend->name());
        break;
      }
    }
    if (!resolved) {
      failed++;
      DSP_DIAG(COMPILE, "platformPrecompileSegments: seg[%d-%d] no backend accepted",
               seg.def.startSlot, seg.def.endSlot);
    }
  }

  DSP_DIAG(COMPILE, "platformPrecompileSegments: compiled=%d skipped=%d failed=%d total=%d",
           compiled, skipped, failed, (int)segments_.size());
}

// ── Segment device binding: always succeeds on CPU ──────────────────────────

bool NativeDynamicShapePlan::platformBindSegmentDevice(const GraphSegment& segment) {
  return true;
}

void NativeDynamicShapePlan::platformRestoreSegmentDevice() {}

// ── Cross-device migration: no-op on CPU ────────────────────────────────────

void NativeDynamicShapePlan::platformMigrateSegmentInputs(
    const GraphSegment& seg, NDArray** externalInputs, int numExternalInputs) {
  // No cross-device migration needed on CPU
}

void NativeDynamicShapePlan::platformCleanupMigratedInputs() {
  // No-op on CPU
}

// CPU stub — all memory is host-accessible; no migration required.
NDArray* NativeDynamicShapePlan::platformGetOutputForDevice0(NDArray* arr, int /*slotIdx*/, int /*outputIdx*/) {
  return arr;
}

// ── performPreReplaySync: passthrough on CPU ──────────────────────────────────
// On CPU there are no CUDA streams, no cross-stream ordering, no D2D staging.
// Return externalArrays unchanged.

NDArray** NativeDynamicShapePlan::performPreReplaySync(
    NDArray** externalArrays, int numExt, void* stream, const char* diagTag) {
  (void)numExt; (void)stream; (void)diagTag;
  return externalArrays;
}

void NativeDynamicShapePlan::verifyStagingNotStale(
    NDArray** externalArrays, NDArray** effectiveArrays,
    int numExt, void* stream, const char* diagTag) {
  // No-op on CPU
  (void)externalArrays; (void)effectiveArrays; (void)numExt; (void)stream; (void)diagTag;
}

// ── Graph eligibility: check CPU/GPU graph backends ─────────────────────────

bool NativeDynamicShapePlan::platformShouldUseGraph(const GraphSegment& segment) {
  if (!segment.def.isCapturable) return false;
  return (segment.def.selectedBackend == SelectedBackend::CPU_GRAPH);
}

// ── Segment execution: Cascading CPU dispatch ───────────────────────────────

Status NativeDynamicShapePlan::platformExecuteSegmentWithBackends(
    GraphSegment& segment, NDArray** externalInputs, int numExternalInputs,
    void* stream, bool& usedGraph) {
  usedGraph = false;

  DSP_DIAG(EXECUTE, "NativeDSP::execute: seg[%d-%d] selectedBackend=%d isCapturable=%d executionCount=%d",
           segment.def.startSlot, segment.def.endSlot,
           static_cast<int>(segment.def.selectedBackend), static_cast<int>(segment.def.isCapturable),
           segment.exec.executionCount);

  switch (segment.def.selectedBackend) {
    case SelectedBackend::CPU_GRAPH: {
      // Use the full CPU graph backend cascade: warmup → compile → execute.
      // executeSegmentWithCpuGraph() handles the complete lifecycle:
      //   - First call: runs slot-by-slot warmup to establish output shapes
      //   - Second call: tries each backend (OneDNN, OpenVINO, ...) in priority order
      //   - Subsequent calls: reuses the resolved backend directly
      // This is the ONLY correct way to dispatch CPU_GRAPH — never bypass it
      // with direct resolvedCpuBackend calls (the backend may not be compiled yet).
      if (!segment.exec.compilationFailed && !segment.exec.noFusibleOps) {
        Status st = executeSegmentWithCpuGraph(segment, externalInputs, numExternalInputs, stream);
        if (st == Status::OK) {
          usedGraph = (segment.resolvedCpuBackend != nullptr);
          return Status::OK;
        }
        // Cascade returned KERNEL_FAILURE = no backend could fuse this segment.
        // Mark failed via lifecycle so we don't retry the cascade on every call.
        SegmentLifecycle::markFailed(segment.exec, "cpu_graph_cascade_failed",
                                     segment.def.startSlot, segment.def.endSlot);
        DSP_DIAG(BACKEND, "NativeDSP::execute: CPU_GRAPH cascade failed seg[%d-%d], "
                 "marked FAILED via lifecycle, falling back to slot-by-slot",
                 segment.def.startSlot, segment.def.endSlot);
      }
      [[fallthrough]];
    }

    case SelectedBackend::GPU_COMPILER:
    case SelectedBackend::CUDA_GRAPHS:
      // GPU backends not applicable on CPU build — slot-by-slot
      // (fall through)

    case SelectedBackend::EMULATED_REPLAY:
      // Emulated replay uses the same slot-by-slot + FunctionalReplayHandle path
      // (fall through)

    case SelectedBackend::SLOT_BY_SLOT:
    default:
      // Slot-by-slot with FunctionalReplayHandle for caching
      if (!segment.exec.replayHandle && segment.def.isCapturable) {
        segment.exec.replayHandle = GraphReplayFactory::create(0);
        segment.exec.replayHandle->beginCapture(nullptr);
      }

      {
        auto status = executeSegmentSlotBySlot(segment, externalInputs, numExternalInputs, stream);

        if (segment.exec.replayHandle && segment.exec.replayHandle->getState() == ReplayState::CAPTURING) {
          if (status == Status::OK) {
            segment.exec.replayHandle->endCapture(nullptr);
            segment.exec.replayHandle->finalize();
            DSP_SET_SEG_PHASE(segment, ExecutionPhase::COMPILED, "functional_capture_end_ok");
          } else {
            SegmentLifecycle::markFunctionalCaptureFailure(segment.exec,
                segment.def.startSlot, segment.def.endSlot);
            DSP_SET_SEG_PHASE(segment, ExecutionPhase::SLOT_BY_SLOT, "functional_capture_failed");
          }
        } else if (segment.exec.replayHandle && segment.exec.replayHandle->isReady()) {
          segment.exec.replayHandle->replay(nullptr);
          DSP_SET_SEG_PHASE(segment, ExecutionPhase::REPLAYING, "functional_replay_ready");
        } else {
          DSP_SET_SEG_PHASE(segment, ExecutionPhase::SLOT_BY_SLOT, "non_capturable_slot_by_slot");
        }

        segment.exec.compiledByBackend = "slot-by-slot";
        return status;
      }
  }
}

// ── Post-segment check: no GPU errors on CPU ───────────────────────────────

Status NativeDynamicShapePlan::platformCheckPostSegment(GraphSegment& segment) {
  return Status::OK;
}

// ── Post-graph-replay fixup: no-op on CPU (no CUDA graphs) ──────────────────

Status NativeDynamicShapePlan::postGraphReplayFixup(
    GraphSegment& seg, NDArray** externalArrays, int numExt,
    void* stream, const char* diagTag) {
  return Status::OK;
}

// ── Monolithic graph replay: no-op on CPU (no CUDA graphs) ──────────────────

Status NativeDynamicShapePlan::replayMonolithicGraph(
    GraphSegment& seg, NDArray** externalArrays, int numExt,
    void* stream, const char* diagTag) {
  return Status::OK;
}

// ── Segment cleanup for rebuild: reset replayHandle on CPU ──────────────────

void NativeDynamicShapePlan::platformCleanupSegmentForRebuild(GraphSegment& seg) {
  seg.exec.replayHandle.reset();
  SegmentLifecycle::resetForResourceRelease(seg.exec);
  // Also reset composite replay handles
  seg.exec.compositeReplaySchedule.mergedReplayHandles.clear();
  seg.exec.compositeReplaySchedule.compositeReplayHandles.clear();
  seg.exec.compositeReplaySchedule.units.clear();
  seg.exec.markArgsStale();
  seg.exec.gapOpsCapturedInGraph = false;
  seg.resolvedCpuBackend = nullptr;
}

// ── Plan resource cleanup: reset replayHandles on CPU ───────────────────────

void NativeDynamicShapePlan::platformFreePlanResources() {
  for (auto& seg : segments_) {
    seg.exec.replayHandle.reset();
    SegmentLifecycle::resetForResourceRelease(seg.exec);
    // Also reset composite replay handles
    seg.exec.compositeReplaySchedule.mergedReplayHandles.clear();
    seg.exec.compositeReplaySchedule.compositeReplayHandles.clear();
    seg.exec.compositeReplaySchedule.units.clear();
    seg.exec.markArgsStale();
    seg.exec.gapOpsCapturedInGraph = false;
    seg.resolvedCpuBackend = nullptr;
  }
  // Free cached steady-state execution context (CPU analog of GPU's context reuse)
  if (steadyStateExecCtx_ != nullptr) {
    delete static_cast<PlanExecutionContext*>(steadyStateExecCtx_);
    steadyStateExecCtx_ = nullptr;
  }
}

void NativeDynamicShapePlan::platformFreeCaptureWorkspace() {
  // No capture workspace on CPU — nothing to free.
}

// ── Statistics: count segments with ready replayHandles ─────────────────────

int NativeDynamicShapePlan::platformCountCapturedGraphSegments() const {
  int count = 0;
  for (const auto& seg : segments_) {
    if ((seg.exec.replayHandle && seg.exec.replayHandle->isReady()) ||
        hasCompositeHandles(seg)) {
      count++;
    }
  }
  return count;
}

// ── Adaptive splitting: no-op on CPU (no GPU graphs to benefit from) ────────

void NativeDynamicShapePlan::platformMaybeSplitIfEnabled() {
  // No-op: adaptive splitting only benefits GPU graph capture
}

// ── Additional platform dispatch stubs (extracted from NativeDynamicShapePlan.cpp) ──

void* NativeDynamicShapePlan::platformBeginExecution(void* stream, bool frozen, int execCount) {
  // Reuse cached context in steady state to avoid heap alloc/free per step.
  // Same optimization as GPU's steadyStateExecCtx_ in executeSteadyState().
  PlanExecutionContext* ctx;
  if (frozen && execCount > 2 && steadyStateExecCtx_ != nullptr) {
    ctx = static_cast<PlanExecutionContext*>(steadyStateExecCtx_);
    // Reset per-step sync state machine
    ctx->resetSyncPhase();
    ctx->flowEventCount = 0;
    ctx->streamSyncCount = 0;
    ctx->eventSyncCount = 0;
  } else {
    ctx = new PlanExecutionContext();
    if (frozen && execCount > 1 && steadyStateExecCtx_ == nullptr) {
      steadyStateExecCtx_ = static_cast<void*>(ctx);
    }
  }
  ctx->execCount = execCount;
  ctx->frozen = frozen;
  // Sync decisions use anySegmentNeedsWarmup() — the SINGLE source of truth.
  bool segWarmup = anySegmentNeedsWarmup();
  ctx->needsFullSync = !frozen || execCount <= 1 || segWarmup;
  ctx->isFrozenSteadyState = frozen && execCount > 1 && !segWarmup;
  return static_cast<void*>(ctx);
}

void NativeDynamicShapePlan::platformEndExecution(void* executionState, void* stream, bool frozen, int execCount) {
  // Don't delete if it's the reused steady-state context
  if (executionState != steadyStateExecCtx_) {
    delete static_cast<PlanExecutionContext*>(executionState);
  }
}

void NativeDynamicShapePlan::platformDumpExternalInputDiagnostics(NDArray** ext, int numExt, int execCount) {
  // GPU diagnostic only
}

void NativeDynamicShapePlan::platformDumpExtInputGpuValues(NDArray* arr, int extIdx, int execCount, void* stream) {
  // GPU diagnostic only
}

void NativeDynamicShapePlan::platformClearCastCache() {
  // MmulHelper cast cache is CUDA-only
}

void NativeDynamicShapePlan::platformSetDeterministicCublas(bool enable) {
  // cuBLAS is CUDA-only — no-op on CPU
}

void NativeDynamicShapePlan::platformSetupSteadyStateCuda(void* execCtxVoid, void* stream) {
  // CUDA cross-stream setup — no-op on CPU
}

void NativeDynamicShapePlan::platformTeardownSteadyStateCuda(void* execCtxVoid, void* stream, void* prevDspStream) {
  // CUDA cross-stream teardown — no-op on CPU
}

void NativeDynamicShapePlan::platformResetGapCaches() {
  // Gap caches are CUDA-only — no-op on CPU
}

void NativeDynamicShapePlan::platformResetBatchD2D() {
  // Batch D2D transfer cache is CUDA-only — no-op on CPU
}

void NativeDynamicShapePlan::platformPostSegmentPoolManagement(bool frozen, int execCount) {
  // No GPU memory pool on CPU
}

void NativeDynamicShapePlan::platformDumpLogitsArgmax(int execCount, void* stream) {
  // GPU diagnostic only
}

void NativeDynamicShapePlan::platformDetectAndPrepareBatchedGemm(NDArray** ext, int numExt, void* stream) {
  // Batched GEMM is GPU-only
}

// ── Batched GEMM — CPU stubs (no-ops; vectors stay empty on CPU) ─────────────

const LongType* NativeDynamicShapePlan::resolveInputShapeInfo(
    int /*srcIdx*/, NDArray** /*externalArrays*/, int /*numExt*/) const {
  return nullptr;
}

void NativeDynamicShapePlan::detectBatchedGemmGroups(NDArray** /*externalArrays*/, int /*numExt*/) {}

void NativeDynamicShapePlan::reconcileSlotDispatchAfterMerge(const ReplaySchedule& /*sched*/) {}

void NativeDynamicShapePlan::prepareBatchedGemmDevice(void* /*stream*/) {}

Status NativeDynamicShapePlan::executeBatchedGemmGroup(
    int /*groupIdx*/, NDArray** /*externalArrays*/, int /*numExt*/, void* /*stream*/) {
  // Should never be called on CPU — batchedGemmGroups_ stays empty.
  return Status::KERNEL_FAILURE;
}

void NativeDynamicShapePlan::freeBatchedGemmResources() {}

void NativeDynamicShapePlan::platformPreReplayPoolStats(size_t& poolUsedOut, size_t& poolReservedOut) {
  poolUsedOut = 0;
  poolReservedOut = 0;
}

void NativeDynamicShapePlan::platformPostReplayPoolManagement(size_t poolUsedPre, bool frozen, int execCount) {
  // No GPU memory pool on CPU
}

void NativeDynamicShapePlan::platformTraceSlotValues(const GraphSegment& seg, void* stream, int execCount) {
  // GPU diagnostic only
}

SelectedBackend NativeDynamicShapePlan::platformResolveBackend(bool isGraphCapture) const {
  // On CPU builds, graph capture (CUDA/HIP graphs) is not available natively.
  // However, when a CPU graph backend is available (OneDNN, OpenVINO, etc.),
  // honor the user's intent: "use the best graph backend" maps to CPU_GRAPH.
  // This ensures CUDA_GRAPHS mode works meaningfully on CPU by routing to OneDNN
  // instead of silently falling back to SLOT_BY_SLOT.
#if HAVE_ONEDNN || HAVE_OPENVINO || HAVE_ARMCOMPUTE || HAVE_MLIR || HAVE_NNAPI || HAVE_MLX
  return SelectedBackend::CPU_GRAPH;
#else
  // No CPU graph backends available — use emulated replay for ALL non-SLOT_BY_SLOT
  // modes. EMULATED_REPLAY correctly handles the replay lifecycle on CPU
  // (shape freezing, pointer stability tracking, segment timing) without
  // requiring actual hardware graph capture. Previously, isGraphCapture=true
  // returned SLOT_BY_SLOT, but the plan-level mode (GEM_CUDA_GRAPHS) still
  // dispatched to phaseReplay, which expected EMULATED_REPLAY segments.
  // The backend mismatch caused phaseReplay to hit the fallback path with
  // wrong lifecycle state, corrupting execution.
  return SelectedBackend::EMULATED_REPLAY;
#endif
}

size_t NativeDynamicShapePlan::platformEstimateCaptureBudget() const {
  // CPU: no GPU memory constraint — return max so segment splitting is never
  // triggered by memory budget (only by op-count limits or other criteria).
  return SIZE_MAX;
}

bool NativeDynamicShapePlan::platformShouldBreakSegmentAtTraitBoundary(int /*currIdx*/, int /*prevIdx*/) const {
  // No trait-based segmentation on CPU. OpenVINO and OneDNN backends handle
  // mixed segments via the NativeSlotExecutor callback — unmappable ops within
  // a segment are executed natively while the backend handles the mappable ones.
  // Breaking at every trait boundary produced ~187 tiny segments for Qwen 0.8B
  // instead of ~20 larger ones, causing massive per-segment overhead (mutex,
  // OV model dispatch, f16↔f32 copy per boundary). Same as GPU: return false.
  return false;
}

void NativeDynamicShapePlan::platformReleaseSegmentGpuResources() {
  // CPU path: reset segment execution state (no CUDA graphs or GPU resources)
  for (auto& seg : segments_) {
    seg.exec.reset();
    seg.def.shapeKeyState.reset();
  }
}

void NativeDynamicShapePlan::platformMigrateWeightsAndClearCaches() {
  // No GPU memory pool migration on CPU
}

// ── Slot execution platform dispatch stubs ────────────────────────────────────

void NativeDynamicShapePlan::platformPrezeroSegmentOutputs(const GraphSegment& seg, void* stream) {
  // CPU path: individual nullify for qualifying output buffers
  for (int s = seg.def.startSlot; s <= seg.def.endSlot; s++) {
    if (s < 0 || s >= numSlots_) continue;
    NativeSlot& slot = slots_[s];

    if (!slot.needsPrezero()) continue;

    bool didZero = false;
    for (int o = 0; o < slot.wiring.numOutputs; o++) {
      int outIdx = slot.wiring.outputSlotIndices[o];
      if (outIdx < 0 || outIdx >= totalOutputSlots_) continue;
      if (slots_[outIdx].slotPhase.isViewProducer) continue;
      NDArray* arr = outputSlots_[outIdx];
      if (arr == nullptr) continue;
      if (arr->isView()) continue;
      auto* db = arr->dataBuffer();
      if (db == nullptr) continue;
      size_t bytes = db->getLenInBytes();
      if (bytes == 0) continue;
      arr->nullify();
      didZero = true;
    }
    if (didZero) slot.bumpGeneration();
  }
}

void NativeDynamicShapePlan::platformReconcileOutputActuality(
    const char* stage, int stepIdx, const NativeSlot& slot, NDArray* output) {
  // No device actuality tracking on CPU — no-op
  (void)stage;
  (void)stepIdx;
  (void)slot;
  (void)output;
}

bool NativeDynamicShapePlan::platformValidateSlotInputBuffer(
    int stepIdx, const NativeSlot& slot, int inputIdx, NDArray* input) {
  if (input == nullptr || input->isEmpty()) return true;
  auto* db = input->dataBuffer();
  if (db == nullptr) return true;
  // On CPU, validate the primary (host) buffer
  if (db->primary() == nullptr) {
    return false;
  }
  return true;
}

bool NativeDynamicShapePlan::platformValidateReusableSlotBuffer(NDArray* cached) {
  if (cached == nullptr) return true;
  auto* db = cached->dataBuffer();
  if (db == nullptr) return true;
  // On CPU, a non-empty array with null primary buffer is invalid
  if (db->primary() == nullptr && !cached->isEmpty()) {
    return false;
  }
  return true;
}

void NativeDynamicShapePlan::platformSetLtEpilogue(const NativeSlot& slot, NDArray* biasArray) {
  // cublasLt epilogue not available on CPU — no-op
  (void)slot;
  (void)biasArray;
}

void NativeDynamicShapePlan::platformClearLtEpilogue() {
  // cublasLt epilogue not available on CPU — no-op
}

void NativeDynamicShapePlan::platformLogSlotOutput(
    int stepIdx, const char* opName, const char* tag,
    const int* outputSlotIndices, int numOutputs) {
  // Triton verify logging not available on CPU — no-op
  (void)stepIdx;
  (void)opName;
  (void)tag;
  (void)outputSlotIndices;
  (void)numOutputs;
}

int NativeDynamicShapePlan::copyStagingToBuffer(int extIdx, sd::DataBuffer* dstDataBuffer) {
  if (placeholderStagingBuffers_ == nullptr || extIdx < 0 || extIdx >= numExternalInputs_)
    return -1;
  NDArray* staging = placeholderStagingBuffers_[extIdx];
  if (staging == nullptr) return -1;

  auto* srcDb = staging->dataBuffer();
  if (srcDb == nullptr || srcDb->isClosed()) return -2;
  if (dstDataBuffer == nullptr) return -3;

  // CPU: DataBuffer::memcpy does synchronous H2H copy.
  sd::DataBuffer::memcpy(dstDataBuffer, srcDb, 0, 0, staging->lengthOf());
  return 0;
}

// ── Graph-baked address pinning — CPU stubs (CudaMemoryPool is CUDA-only) ─────
// On GPU, writeOutputSlot pins a sealed segment's plan-owned buffer addresses so
// CudaMemoryPool::free() cannot re-hand them while a live CUDA graph still
// references them (real implementations in NativeDynamicShapePlan_cuda.cu). CPU
// has no CudaMemoryPool and no CUDA graph capture, so these are no-ops.

void NativeDynamicShapePlan::platformPinGraphBakedAddress(void* /*ptr*/, int /*deviceId*/) {
  // No-op on CPU — CudaMemoryPool is CUDA-only.
}

void NativeDynamicShapePlan::platformFlushGraphBakedPins(void* /*stream*/) {
  // No-op on CPU — CudaMemoryPool is CUDA-only. Clear graphPinnedAddrs_ defensively
  // (writeOutputSlot only pins for sealed CUDA-graph segments, so it is normally
  // empty here) to avoid stale entries across plan lifetimes.
  graphPinnedAddrs_.clear();
}

// ── Fingerprint ring: CPU stubs (no-op) ─────────────────────────────────────

void NativeDynamicShapePlan::drainFingerprintRingPublic() {
  // CPU build: no device ring; nothing to drain.
}

const char* NativeDynamicShapePlan::getFingerprintJson() {
  return "null";  // CPU build: fingerprints are CUDA-only
}

}  // namespace graph
}  // namespace sd

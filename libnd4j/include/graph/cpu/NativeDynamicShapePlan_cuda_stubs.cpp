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
#include <graph/GraphBackendResolver.h>
#include <graph/GraphReplayHandle.h>
#include <graph/cpu/FunctionalReplayHandle.h>
#include <graph/DspPhaseUtils.h>
#include <graph/DspSegmentLifecycle.h>
#include <graph/DspSegmentHelpers.h>
#include <graph/DspHashUtils.h>
#include <config.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <ops/declarable/OpRegistrator.h>

#if HAVE_ONEDNN
#include <graph/cpu/OneDnnGraphBackend.h>
#endif
#if HAVE_OPENVINO
#include <graph/cpu/OpenVinoGraphBackend.h>
#endif

namespace sd {
namespace graph {

namespace {

Status cpuPlanFailure(const std::string& detail) {
  const std::string message =
      detail + " [DSP status=KERNEL_FAILURE (50)]";
  auto* errorReference = LaunchContext::defaultContext()->errorReference();
  errorReference->setErrorCode(static_cast<int>(Status::KERNEL_FAILURE));
  errorReference->setErrorMessage(message);
  return Status::KERNEL_FAILURE;
}

}  // namespace

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
  // CPU/mobile steady state accepts direct compiled artifacts and real replay
  // handles, but never terminal or implicit slot-by-slot segments.
  if (!planLifecycle_.isInFrozenOrReplayState() ||
      !allFrozenDispatchUnitsReady()) {
    return Status::MAYBE;
  }
  if (hasDynamicSegmentBoundaries_) {
    return Status::MAYBE;
  }
  if (!ModeContract::forMode(graphExecutionMode_).allowsFrozenFastPath) {
    return Status::MAYBE;
  }

  DSP_DIAG(EXECUTE, "CPU_FROZEN_FAST_PATH: segments=%d executeCount=%d",
           (int)segments_.size(), executeCount_);

  // Execute all segments using their resolved backends (OneDNN/OpenVINO)
  // when available. Functional replay segments must keep using their recorded
  // program here; routing them through the generic slot loop would make the
  // frozen fast path silently bypass the recorder.
  // Graph backends fuse multiple ops into optimized subgraphs, dramatically
  // reducing per-op dispatch overhead (1761 individual calls → ~dozens of fused calls).
  for (auto& seg : segments_) {
    // All-frozen-constant segments: outputs already populated from warmup.
    if (seg.def.allFrozenConstants) {
      seg.exec.executionCount++;
      continue;
    }
    Status status;
    if (seg.def.selectedBackend == SelectedBackend::EMULATED_REPLAY) {
      status = executeSegmentEmulatedReplay(
          seg, externalInputs, numExternalInputs, stream);
    } else if (seg.resolvedGraphBackend != nullptr) {
      // Fast path: backend already compiled — execute via executeSegmentWithSpecificBackend
      // which installs the NativeSlotExecutor for native-deferred ops (rope, attention).
      // Calling backend->executeSegment() directly would skip NativeSlotExecutor setup
      // and cause native-deferred ops to fail with KERNEL_FAILURE.
      DspExecutionResult backendResult = executeSegmentWithSpecificBackend(
          seg, seg.resolvedGraphBackend, externalInputs, numExternalInputs,
          stream);
      status = backendResult.status;
      if (backendResult.preExecutionRejection()) {
        DSP_DIAG(EXECUTE,
                 "CPU_FROZEN: backend rejected seg[%d-%d] before execution; "
                 "returning MAYBE for ordered backend re-resolution",
                 seg.def.startSlot, seg.def.endSlot);
        return Status::MAYBE;
      }
      if (status != Status::OK) {
        DSP_DIAG(EXECUTE,
                 "CPU_FROZEN: resolved graph backend failed seg[%d-%d]; "
                 "post-start failover is forbidden",
                 seg.def.startSlot, seg.def.endSlot);
        return status;
      }
    } else {
      DSP_DIAG(EXECUTE,
               "CPU_FROZEN: seg[%d-%d] has no explicit steady-state artifact",
               seg.def.startSlot, seg.def.endSlot);
      return cpuPlanFailure(
          "CPU frozen DSP path found no explicit compiled/replay artifact for "
          "segment [" + std::to_string(seg.def.startSlot) + "-" +
          std::to_string(seg.def.endSlot) + "]");
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

// ── Eager precompilation: compile all resolved graph-backend segments ────────
//
// On GPU, this triggers parallel Triton/NVRTC compilation. On CPU, we eagerly
// resolve and compile graph backends (oneDNN, OpenVINO, NNAPI, ACL, etc.) for
// all segments during phaseCompile() instead of lazily on second execution.
// This moves compilation latency out of the hot path.

void NativeDynamicShapePlan::platformPrecompileSegments(
    NDArray** externalInputs, int numExternalInputs) {
  DSP_REQUIRE_PLAN_PHASE_AT_MOST(PlanPhase::SHAPES_FROZEN, "platformPrecompileSegments");

  const auto& chain = getGraphBackendCandidates();
  const GraphBackendRequest request = makeGraphBackendRequest();
  if (chain.empty()) {
    DSP_DIAG(COMPILE, "platformPrecompileSegments: no graph backends resolved, skipping");
    return;
  }

  const auto planningPolicy =
      GraphBackendResolver::aggregatePlanningPolicy(request, chain);
  if (planningPolicy.requiresPrecommitFunctionalWarmup) {
    // Calibration backends cannot all be compiled from one earlier functional
    // snapshot: compiled upstream quantization changes the values observed by
    // downstream boundaries. Dispatch one complete frozen pass in graph order.
    // Each graph segment compiles immediately before its first device execution,
    // while explicit-replay/native ranges materialize the intervening values.
    // phaseCompile applies the atomic plan seal only after this pass succeeds.
    DSP_DIAG(COMPILE,
             "PROGRESSIVE_CALIBRATION_PASS_BEGIN segments=%d",
             static_cast<int>(segments_.size()));
    int dispatched = 0;
    for (auto& seg : segments_) {
      bool usedGraph = false;
      const Status status = dispatchSegment(
          seg, externalInputs, numExternalInputs, nullptr, usedGraph);
      if (status != Status::OK) {
        DSP_THROW(
            COMPILE,
            "PROGRESSIVE_CALIBRATION_PASS_FAILED seg[%d-%d] status=%s (%d)",
            seg.def.startSlot, seg.def.endSlot, dsp::dspStatusName(status),
            static_cast<int>(status));
      }
      ++dispatched;
    }
    DSP_DIAG(COMPILE,
             "PROGRESSIVE_CALIBRATION_PASS_DONE dispatched=%d",
             dispatched);
    return;
  }

  int compiled = 0, skipped = 0, replayFallback = 0, failed = 0;
  const bool allowsReplayFallback =
      ModeContract::forMode(graphExecutionMode_).allowsFallback;
  for (auto& seg : segments_) {
    // CPU precompilation is only valid for segments explicitly resolved to
    // GRAPH_BACKEND.
    if (seg.def.selectedBackend != SelectedBackend::GRAPH_BACKEND) {
      skipped++;
      continue;
    }
    // Skip segments that already have a resolved backend
    if (seg.resolvedGraphBackend != nullptr) {
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

    const LongType segShapeKey =
        computeSegmentShapeKey(seg, externalInputs, numExternalInputs);
    seg.def.shapeKeyState.recordComputed(segShapeKey);

    const auto lowering = GraphBackendResolver::lowerSegment(
        request, chain, seg.resolvedGraphBackend, seg, slots_,
        seg.def.startSlot, seg.def.endSlot, externalInputs, numExternalInputs,
        outputSlots_, totalOutputSlots_, segShapeKey, numSlots_,
        requestedOutputSlotIndices_, numRequestedOutputs_);
    if (lowering.succeeded()) {
      seg.setResolvedGraphBackend(lowering.backend, request);
      seg.exec.cachedShapeKey = segShapeKey;
      seg.def.shapeKeyState.markCompiled(segShapeKey);
      if (seg.resolvedGraphBackendPolicy.artifactKind ==
              GraphBackendArtifactKind::DIRECT_COMPILED &&
          seg.compiledGraphBackendArtifactOwner == lowering.backend &&
          seg.compiledGraphBackendArtifactShapeKey == segShapeKey &&
          seg.compiledGraphBackendArtifact && !seg.exec.segPhase.isSealed()) {
        SegmentLifecycle::markDirectGraphBackendCompiled(
            seg.exec, lowering.backend->name(), segShapeKey,
            seg.def.startSlot, seg.def.endSlot);
      } else if (seg.resolvedGraphBackendPolicy.artifactKind ==
                     GraphBackendArtifactKind::BACKEND_REPLAY_HANDLE &&
                 seg.exec.replayHandle && seg.exec.replayHandle->isReady() &&
                 !seg.exec.segPhase.isSealed()) {
        SegmentLifecycle::markBackendReplayHandleSealed(
            seg.exec, lowering.backend->name(), segShapeKey,
            seg.def.startSlot, seg.def.endSlot);
      }
      compiled++;
      DSP_DIAG(COMPILE,
               "platformPrecompileSegments: seg[%d-%d] compiled by %s",
               seg.def.startSlot, seg.def.endSlot, lowering.backend->name());
    } else if (lowering.prerequisiteBlocked()) {
      failed++;
      const std::string message =
          std::string("graph backend compilation prerequisite missing for ") +
          lowering.prerequisiteBlockedBackend->name() + " seg[" +
          std::to_string(seg.def.startSlot) + "-" +
          std::to_string(seg.def.endSlot) + "]: " +
          lowering.prerequisiteFailureReason;
      sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(
          message.c_str());
      DSP_THROW(COMPILE, "%s", message.c_str());
    } else if (allowsReplayFallback) {
      // Keep eager precompile behavior identical to the runtime backend cascade.
      // ARM hybrid is an ordered NNAPI -> ACL -> explicit-replay policy: a
      // concrete device capability rejection (for example, EdgeTPU rejecting an
      // otherwise valid NNAPI model) is not a compiler defect and must transfer
      // ownership to the functional recorder before the compilation seal. Strict
      // compiler modes leave allowsFallback=false and continue to fail closed.
      seg.resetGraphBackend();
      seg.def.selectedBackend = SelectedBackend::EMULATED_REPLAY;
      SegmentLifecycle::prepareFunctionalReplayHandoff(
          seg.exec, seg.def.startSlot, seg.def.endSlot);
      replayFallback++;
      DSP_DIAG(COMPILE,
               "platformPrecompileSegments: seg[%d-%d] no backend lowered; "
               "ownership -> explicit replay before compilation seal",
               seg.def.startSlot, seg.def.endSlot);
    } else {
      failed++;
      DSP_DIAG(COMPILE,
               "platformPrecompileSegments: seg[%d-%d] no backend accepted",
               seg.def.startSlot, seg.def.endSlot);
    }
  }

  DSP_DIAG(COMPILE,
           "platformPrecompileSegments: compiled=%d skipped=%d explicitReplay=%d "
           "failed=%d total=%d",
           compiled, skipped, replayFallback, failed, (int)segments_.size());
}

// ── Segment device binding: thread-local device bracketing ──────────────────

bool NativeDynamicShapePlan::platformBindSegmentDevice(const GraphSegment& segment) {
  return true;
}

void NativeDynamicShapePlan::platformRestoreSegmentDevice() {
}

// ── Cross-device migration: no-op on CPU ────────────────────────────────────

Status NativeDynamicShapePlan::platformMigrateSegmentInputs(
    const GraphSegment& seg, NDArray** externalInputs, int numExternalInputs) {
  // No cross-device migration needed on CPU
  return Status::OK;
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

DspStagingSyncResult NativeDynamicShapePlan::performPreReplaySync(
    NDArray** externalArrays, int numExt, void* stream, const char* diagTag) {
  (void)numExt; (void)stream; (void)diagTag;
  return {externalArrays, DspStagingSyncStatus::NOT_REQUIRED, 0, false};
}

void NativeDynamicShapePlan::verifyStagingNotStale(
    NDArray** externalArrays, NDArray** effectiveArrays,
    int numExt, void* stream, const char* diagTag) {
  // No-op on CPU
  (void)externalArrays; (void)effectiveArrays; (void)numExt; (void)stream; (void)diagTag;
}

// ── Graph eligibility: check the resolved backend cascade ───────────────────

bool NativeDynamicShapePlan::platformShouldUseGraph(const GraphSegment& segment) {
  if (!segment.def.isCapturable) return false;
  return segment.def.selectedBackend == SelectedBackend::GRAPH_BACKEND;
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
    case SelectedBackend::GRAPH_BACKEND: {
      // Use the resolved graph-backend cascade: warmup → compile → execute.
      // executeSegmentWithGraphBackend() handles the complete lifecycle:
      //   - First call: runs slot-by-slot warmup to establish output shapes
      //   - Second call: tries each backend (OneDNN, OpenVINO, ...) in priority order
      //   - Subsequent calls: reuses the resolved backend directly
      // This is the ONLY correct way to dispatch GRAPH_BACKEND — never bypass it
      // with direct resolvedGraphBackend calls (the backend may not be compiled yet).
      if (!segment.exec.compilationFailed && !segment.exec.noFusibleOps) {
        DspExecutionResult result = executeSegmentWithGraphBackend(
            segment, externalInputs, numExternalInputs, stream);
        if (result.ok()) {
          usedGraph = (segment.resolvedGraphBackend != nullptr);
          return Status::OK;
        }
        DSP_DIAG(BACKEND,
                 "NativeDSP::execute: GRAPH_BACKEND failed seg[%d-%d]; "
                 "post-start failover is forbidden",
                 segment.def.startSlot, segment.def.endSlot);
        return result.status;
      }
      return cpuPlanFailure(
          "CPU graph backend dispatch rejected segment [" +
          std::to_string(segment.def.startSlot) + "-" +
          std::to_string(segment.def.endSlot) + "]: compilationFailed=" +
          std::to_string(segment.exec.compilationFailed ? 1 : 0) +
          ", noFusibleOps=" +
          std::to_string(segment.exec.noFusibleOps ? 1 : 0) +
          ", phase=" + segment.exec.displayPhaseName());
    }

    case SelectedBackend::EMULATED_REPLAY:
      segment.exec.compiledByBackend = "explicit-replay";
      return executeSegmentEmulatedReplay(
          segment, externalInputs, numExternalInputs, stream);

    case SelectedBackend::DEVICE_REPLAY:
    case SelectedBackend::SLOT_BY_SLOT:
    default:
      segment.exec.compiledByBackend = "slot-by-slot";
      return executeSegmentSlotBySlot(
          segment, externalInputs, numExternalInputs, stream);
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
  seg.resetGraphBackend();
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
    seg.resetGraphBackend();
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
  // CPU: no GPU memory pool.
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
  // CPU: no GPU memory pool.
}

void NativeDynamicShapePlan::platformTraceSlotValues(const GraphSegment& seg, void* stream, int execCount) {
  // GPU diagnostic only
}

SelectedBackend NativeDynamicShapePlan::platformResolveBackend(
    bool isGraphCapture) const {
  (void)isGraphCapture;
  auto* mutablePlan = const_cast<NativeDynamicShapePlan*>(this);
  return mutablePlan->getGraphBackendCandidates().empty()
             ? SelectedBackend::EMULATED_REPLAY
             : SelectedBackend::GRAPH_BACKEND;
}

SelectedBackend NativeDynamicShapePlan::platformResolvePortableReplayBackend() const {
  // PORTABLE_REPLAY is the backend-neutral recorder contract. CPU has no
  // platform graph handle, so selecting a compiler candidate here would make
  // the result depend on optional OneDNN/OpenVINO availability and can leave
  // an unfused range with no executable owner. Use the functional recorder
  // deterministically; compiler-backed modes resolve their own candidates.
  return SelectedBackend::EMULATED_REPLAY;
}

size_t NativeDynamicShapePlan::platformEstimateCaptureBudget() const {
  // CPU has no GPU memory constraint.
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
  // CPU/Android direct graph artifacts are plan-owned resources too. Releasing
  // a session or passivating a cached plan must drop NNAPI/ACL/oneDNN/OpenVINO
  // compilations and their constants before the plan is returned to the cache.
  for (auto& seg : segments_) {
    seg.exec.reset();
    seg.resetGraphBackend();
  }
}

void NativeDynamicShapePlan::platformMigrateWeightsAndClearCaches() {
  // No GPU memory pool migration on CPU
}

// ── Slot execution platform dispatch stubs ────────────────────────────────────

Status NativeDynamicShapePlan::platformExecuteSlot(const NativeSlot& slot,
                                                   Context& context) {
  return slot.ident.op->execute(&context);
}

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
      if (slot.slotPhase.isViewProducer) continue;
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

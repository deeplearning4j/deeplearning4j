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

#ifndef SD_CUDA

#include <graph/NativeDynamicShapePlan.h>
#include <graph/GraphBackend.h>
#include <graph/GraphReplayHandle.h>
#include <graph/cpu/FunctionalReplayHandle.h>
#include <graph/DspPhaseUtils.h>
#include <config.h>

#include <cstring>
#include <ops/declarable/OpRegistrator.h>

namespace sd {
namespace graph {

// ── Frozen graph fast path: not available on CPU ────────────────────────────

Status NativeDynamicShapePlan::platformTryFrozenFastPath(
    NDArray** externalInputs, int numExternalInputs,
    NDArray** requestedOutputs, int numRequestedOutputs, void* stream) {
  return Status::MAYBE;  // Not available — fall through to normal path
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

// ── Parallel precompilation: no-op on CPU ───────────────────────────────────

void NativeDynamicShapePlan::platformPrecompileSegments(
    NDArray** externalInputs, int numExternalInputs) {
  DSP_REQUIRE_PLAN_PHASE_AT_MOST(PlanPhase::SHAPES_FROZEN, "platformPrecompileSegments");
  // No GPU compilation on CPU builds
}

// ── Segment device binding: always succeeds on CPU ──────────────────────────

bool NativeDynamicShapePlan::platformBindSegmentDevice(const GraphSegment& segment) {
  return true;
}

// ── Cross-device migration: no-op on CPU ────────────────────────────────────

void NativeDynamicShapePlan::platformMigrateSegmentInputs(
    const GraphSegment& seg, NDArray** externalInputs, int numExternalInputs) {
  // No cross-device migration needed on CPU
}

void NativeDynamicShapePlan::platformCleanupMigratedInputs() {
  // No-op on CPU
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
      // Use cascading backend selection — executeSegmentWithCpuGraph iterates the
      // backend chain and caches the resolved backend per-segment.
      auto status = executeSegmentWithCpuGraph(segment, externalInputs, numExternalInputs, stream);
      if (status == Status::OK) {
        usedGraph = true;
        if (segment.resolvedCpuBackend) {
          segment.exec.compiledByBackend = segment.resolvedCpuBackend->name();
        } else {
          segment.exec.compiledByBackend = "CPU";
        }
        if (segment.exec.executionCount <= 1) {
          DSP_SET_SEG_PHASE(segment, ExecutionPhase::COMPILING, "cpu_graph_first_exec");
        } else if (segment.exec.replayHandle && segment.exec.replayHandle->isReady()) {
          DSP_SET_SEG_PHASE(segment, ExecutionPhase::REPLAYING, "cpu_graph_replay_ready");
        } else {
          DSP_SET_SEG_PHASE(segment, ExecutionPhase::COMPILED, "cpu_graph_compiled_no_replay");
        }
        return Status::OK;
      }
      // All backends in the cascade failed — fall through to slot-by-slot
      DSP_DIAG(FALLBACK, "NativeDSP::execute: all CPU backends failed for seg[%d-%d] — falling back to slot-by-slot",
               segment.def.startSlot, segment.def.endSlot);
      [[fallthrough]];
    }

    case SelectedBackend::GPU_COMPILER:
    case SelectedBackend::CUDA_GRAPHS:
      // GPU backends not applicable on CPU build — slot-by-slot
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
            segment.exec.replayHandle.reset();
            segment.exec.argTableStable = false;  // Invalidate fast-replay on capture failure
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

// ── Segment cleanup for rebuild: reset replayHandle on CPU ──────────────────

void NativeDynamicShapePlan::platformCleanupSegmentForRebuild(GraphSegment& seg) {
  seg.exec.replayHandle.reset();
  seg.exec.argTableStable = false;  // Invalidate fast-replay when handles are cleared
  seg.exec.gapOpsCapturedInGraph = false;
  seg.resolvedCpuBackend = nullptr;
}

// ── Plan resource cleanup: reset replayHandles on CPU ───────────────────────

void NativeDynamicShapePlan::platformFreePlanResources() {
  for (auto& seg : segments_) {
    seg.exec.replayHandle.reset();
    seg.exec.argTableStable = false;  // Invalidate fast-replay on plan teardown
    seg.exec.gapOpsCapturedInGraph = false;
    seg.resolvedCpuBackend = nullptr;
  }
}

// ── Statistics: count segments with ready replayHandles ─────────────────────

int NativeDynamicShapePlan::platformCountCapturedGraphSegments() const {
  int count = 0;
  for (const auto& seg : segments_) {
    if (seg.exec.replayHandle && seg.exec.replayHandle->isReady()) count++;
  }
  return count;
}

// ── Adaptive splitting: no-op on CPU (no GPU graphs to benefit from) ────────

void NativeDynamicShapePlan::platformMaybeSplitIfEnabled() {
  // No-op: adaptive splitting only benefits GPU graph capture
}

// ── Additional platform dispatch stubs (extracted from NativeDynamicShapePlan.cpp) ──

void* NativeDynamicShapePlan::platformBeginExecution(void* stream, bool frozen, int execCount) {
  return nullptr;  // No stream management on CPU
}

void NativeDynamicShapePlan::platformEndExecution(void* executionState, void* stream, bool frozen, int execCount) {
  // No cross-stream sync on CPU
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

void NativeDynamicShapePlan::platformPostSegmentPoolManagement(bool frozen, int execCount) {
  // No GPU memory pool on CPU
}

void NativeDynamicShapePlan::platformDumpLogitsArgmax(int execCount, void* stream) {
  // GPU diagnostic only
}

void NativeDynamicShapePlan::platformDetectAndPrepareBatchedGemm(NDArray** ext, int numExt, void* stream) {
  // Batched GEMM is GPU-only
}

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
  // Graph capture (CUDA graphs on CPU) is not available on CPU builds.
  if (isGraphCapture) return SelectedBackend::SLOT_BY_SLOT;

  // CPU_GRAPH requires at least one optional CPU graph backend compiled in
  // (OneDNN, OpenVINO, ACL, MLIR, NNAPI, or MLX). If none are available,
  // fall back to SLOT_BY_SLOT so GEM_AUTO doesn't trigger executeSegmentWithCpuGraph
  // on a plain CPU build with no optional backends registered.
#if HAVE_ONEDNN || HAVE_OPENVINO || HAVE_ARMCOMPUTE || HAVE_MLIR || HAVE_NNAPI || HAVE_MLX
  return SelectedBackend::CPU_GRAPH;
#else
  return SelectedBackend::SLOT_BY_SLOT;
#endif
}

bool NativeDynamicShapePlan::platformShouldBreakSegmentAtTraitBoundary(int currIdx, int prevIdx) const {
  // On CPU, break segments at ops with no traits registered in OpTraitTable.
  // Ops with traits (elementwise, reduction, matmul, normalization, etc.) are
  // compilable by OneDNN/OpenVINO. Ops without traits (custom/unknown ops) are
  // isolated into 1-slot segments for slot-by-slot execution.
  auto& registrator = sd::ops::OpRegistrator::getInstance();
  auto hasTraits = [&](int idx) -> bool {
    auto* op = registrator.getOperation(slots_[idx].ident.opName.c_str());
    if (op == nullptr) return false;
    auto* desc = op->getOpDescriptor();
    if (desc == nullptr) return false;
    return desc->getTraits() != 0;
  };
  bool currHasTraits = hasTraits(currIdx);
  bool prevHasTraits = hasTraits(prevIdx);
  return currHasTraits != prevHasTraits;
}

void NativeDynamicShapePlan::platformReleaseSegmentGpuResources() {
  // CPU path: reset segment execution state (no CUDA graphs or GPU resources)
  for (auto& seg : segments_) {
    if (seg.exec.replayHandle) {
      seg.exec.replayHandle.reset();
    }
    seg.exec.gapOpsCapturedInGraph = false;
    seg.exec.argTableStable = false;
    seg.exec.capturedInputAddrKey = 0;
    seg.exec.compilationFailed = false;
    seg.exec.executionCount = 0;
    seg.exec.cachedShapeKey = 0;
    seg.exec.capturedCreateValueKey = 0;
    seg.exec.captureOomRetries = 0;
    seg.exec.captureRetryAfterExec = 0;
    seg.exec.compiledByBackend.clear();
    seg.exec.currentPhase = ExecutionPhase::WARMUP;
    seg.def.shapeKey = 0;
  }
}

void NativeDynamicShapePlan::platformMigrateWeightsAndClearCaches() {
  // No GPU memory pool migration on CPU
}

}  // namespace graph
}  // namespace sd

#endif  // !SD_CUDA

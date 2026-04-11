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

#include <cstring>
#include <ops/declarable/OpRegistrator.h>

namespace sd {
namespace graph {

// ── Batch-zero stubs (GPU-only feature) ─────────────────────────────────────

bool NativeDynamicShapePlan::isBatchZeroActive() { return false; }
bool NativeDynamicShapePlan::isBatchZeroRegistering() { return false; }
void NativeDynamicShapePlan::registerBatchZeroBuffer(void*, size_t, int) {}

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
          segment.exec.currentPhase = ExecutionPhase::COMPILING;
        } else if (segment.exec.replayHandle && segment.exec.replayHandle->isReady()) {
          segment.exec.currentPhase = ExecutionPhase::REPLAYING;
        } else {
          segment.exec.currentPhase = ExecutionPhase::COMPILED;
        }
        return Status::OK;
      }
      // All backends in the cascade failed — fall through to slot-by-slot
      DSP_DIAG(FALLBACK, "NativeDSP::execute: all CPU backends failed for seg[%d-%d] — falling back to slot-by-slot",
               segment.def.startSlot, segment.def.endSlot);
      goto slot_by_slot;
    }

    case SelectedBackend::GPU_COMPILER:
    case SelectedBackend::CUDA_GRAPHS:
      // GPU backends not applicable on CPU build — slot-by-slot
      // (fall through)

    case SelectedBackend::SLOT_BY_SLOT:
    default:
slot_by_slot:
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
            segment.exec.currentPhase = ExecutionPhase::COMPILED;
          } else {
            segment.exec.replayHandle.reset();
            segment.exec.currentPhase = ExecutionPhase::SLOT_BY_SLOT;
          }
        } else if (segment.exec.replayHandle && segment.exec.replayHandle->isReady()) {
          segment.exec.replayHandle->replay(nullptr);
          segment.exec.currentPhase = ExecutionPhase::REPLAYING;
        } else {
          segment.exec.currentPhase = ExecutionPhase::SLOT_BY_SLOT;
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

// ── KV scatter: CPU fallback using operator() + assign() ────────────────────

void* NativeDynamicShapePlan::platformBeginKvScatter(void* stream) {
  return nullptr;  // No stream management on CPU
}

void NativeDynamicShapePlan::platformEndKvScatter(void* savedState) {
  // No-op on CPU
}

void NativeDynamicShapePlan::platformScatterKvEntry(
    NDArray* presentKv, NDArray* staticBuf, int seqDim, int pos, void* stream) {
  int rank = presentKv->rankOf();
  LongType lastPos = presentKv->sizeAt(seqDim) - 1;
  std::vector<LongType> srcIdx(rank * 2), dstIdx(rank * 2);
  for (int d = 0; d < rank; d++) {
    if (d == seqDim) {
      srcIdx[d*2] = lastPos; srcIdx[d*2+1] = lastPos + 1;
      dstIdx[d*2] = pos; dstIdx[d*2+1] = pos + 1;
    } else {
      srcIdx[d*2] = 0; srcIdx[d*2+1] = 0;
      dstIdx[d*2] = 0; dstIdx[d*2+1] = 0;
    }
  }
  NDArray* srcSlice = (*presentKv)(srcIdx, true);
  NDArray* dstSlice = (*staticBuf)(dstIdx, true);
  dstSlice->assign(srcSlice);
  delete srcSlice;
  delete dstSlice;
}

// ── KV capture buffer annotation: no-op on CPU ─────────────────────────────

void NativeDynamicShapePlan::platformMarkKvCaptureBuffersNeverSkip() {
  // No capture buffers on CPU
}

// ── Segment cleanup for rebuild: reset replayHandle on CPU ──────────────────

void NativeDynamicShapePlan::platformCleanupSegmentForRebuild(GraphSegment& seg) {
  seg.exec.replayHandle.reset();
  seg.exec.gapOpsCapturedInGraph = false;
  seg.resolvedCpuBackend = nullptr;
}

// ── Plan resource cleanup: reset replayHandles on CPU ───────────────────────

void NativeDynamicShapePlan::platformFreePlanResources() {
  for (auto& seg : segments_) {
    seg.exec.replayHandle.reset();
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

void NativeDynamicShapePlan::platformUpdateDecodeInputs(NDArray** ext, int numExt,
                                                         long long tokenId, int cachePos, void* stream) {
  // Decode input device writes are GPU-only (Java handles CPU updates)
}

void NativeDynamicShapePlan::platformPostKvScatterSync(int scattered, int pos, int numMappings) {
  // No CUDA sync needed on CPU
}

SelectedBackend NativeDynamicShapePlan::platformResolveBackend(bool isGraphCapture) const {
  return isGraphCapture ? SelectedBackend::SLOT_BY_SLOT : SelectedBackend::CPU_GRAPH;
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
    seg.exec.segBatchZeroEntries.clear();
    seg.def.shapeKey = 0;
  }
}

void NativeDynamicShapePlan::platformMigrateWeightsAndClearCaches() {
  // No GPU memory pool migration on CPU
}

}  // namespace graph
}  // namespace sd

#endif  // !SD_CUDA

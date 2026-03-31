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

#include <cstring>

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
  if (seg.isCapturable && !seg.exec.compilationFailed) return true;
  return false;
}

// ── Parallel precompilation: no-op on CPU ───────────────────────────────────

void NativeDynamicShapePlan::platformPrecompileSegments(
    NDArray** externalInputs, int numExternalInputs) {
  // No GPU compilation on CPU builds
}

// ── Segment device binding: always succeeds on CPU ──────────────────────────

bool NativeDynamicShapePlan::platformBindSegmentDevice(const GraphSegment& segment) {
  return true;
}

// ── Graph eligibility: check CPU/GPU graph backends ─────────────────────────

bool NativeDynamicShapePlan::platformShouldUseGraph(const GraphSegment& segment) {
  if (!segment.isCapturable || segment.exec.compilationFailed) return false;
  return (segment.selectedBackend == SelectedBackend::CPU_GRAPH);
}

// ── Segment execution: Switch-based CPU dispatch (no cascade) ───────────────

Status NativeDynamicShapePlan::platformExecuteSegmentWithBackends(
    GraphSegment& segment, NDArray** externalInputs, int numExternalInputs,
    void* stream, bool& usedGraph) {
  usedGraph = false;

  DSP_DIAG(EXECUTE, "NativeDSP::execute: seg[%d-%d] selectedBackend=%d isCapturable=%d executionCount=%d",
           segment.startSlot, segment.endSlot,
           static_cast<int>(segment.selectedBackend), static_cast<int>(segment.isCapturable),
           segment.exec.executionCount);

  switch (segment.selectedBackend) {
    case SelectedBackend::CPU_GRAPH: {
      auto* cpuBackend = getCpuGraphBackend();
      if (cpuBackend) {
        auto status = executeSegmentWithCpuGraph(segment, externalInputs, numExternalInputs, stream);
        if (status == Status::OK) {
          usedGraph = true;
          segment.exec.compiledByBackend = "CPU";
          if (segment.exec.executionCount <= 1) {
            segment.exec.currentPhase = ExecutionPhase::COMPILING;
          } else if (segment.exec.replayHandle && segment.exec.replayHandle->isReady()) {
            segment.exec.currentPhase = ExecutionPhase::REPLAYING;
          } else {
            segment.exec.currentPhase = ExecutionPhase::COMPILED;
          }
          return Status::OK;
        }
        // CPU graph backend failed — hard error. No cascade.
        DSP_DIAG(FALLBACK, "NativeDSP::execute: cpuBackend FAILED for seg[%d-%d] status=%d — hard error",
                 segment.startSlot, segment.endSlot, static_cast<int>(status));
        return status;
      }
      // Backend resolved at build time but not available at runtime — slot-by-slot
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
      if (!segment.exec.replayHandle && segment.isCapturable && !segment.exec.compilationFailed) {
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
}

// ── Plan resource cleanup: reset replayHandles on CPU ───────────────────────

void NativeDynamicShapePlan::platformFreePlanResources() {
  for (auto& seg : segments_) {
    seg.exec.replayHandle.reset();
    seg.exec.gapOpsCapturedInGraph = false;
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

}  // namespace graph
}  // namespace sd

#endif  // !SD_CUDA

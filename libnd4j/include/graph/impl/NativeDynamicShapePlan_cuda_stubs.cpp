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

#include <cstring>

namespace sd {
namespace graph {

// ── Batch-zero stubs (GPU-only feature) ─────────────────────────────────────

bool NativeDynamicShapePlan::isBatchZeroActive() { return false; }
bool NativeDynamicShapePlan::isBatchZeroRegistering() { return false; }
void NativeDynamicShapePlan::registerBatchZeroBuffer(void*, size_t) {}

// ── Frozen graph fast path: not available on CPU ────────────────────────────

Status NativeDynamicShapePlan::platformTryFrozenFastPath(
    NDArray** externalInputs, int numExternalInputs,
    NDArray** requestedOutputs, int numRequestedOutputs, void* stream) {
  return Status::MAYBE;  // Not available — fall through to normal path
}

// ── Pre-execute setup: no GPU-specific work on CPU ──────────────────────────

void NativeDynamicShapePlan::platformPreExecuteSetup(
    NDArray** externalInputs, int numExternalInputs, void* stream) {
  // No GPU errors to clear, no attention workspace, no stale graph invalidation.
  // Just flush pending close for memory management.
  flushPendingClose(stream);
}

// ── Segment cache retention: check capturability on CPU ─────────────────────

bool NativeDynamicShapePlan::platformShouldKeepSegmentCache(const GraphSegment& seg) const {
  if (seg.isCapturable && !seg.captureFailed) return true;
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
  if (!segment.isCapturable || segment.captureFailed) return false;
  // Use graph if either CPU or GPU graph backend is available
  return (getCpuGraphBackend() != nullptr || getGpuGraphBackend() != nullptr);
}

// ── Segment execution: CPU backend cascade ──────────────────────────────────

Status NativeDynamicShapePlan::platformExecuteSegmentWithBackends(
    GraphSegment& segment, NDArray** externalInputs, int numExternalInputs,
    void* stream, bool& usedGraph) {
  usedGraph = false;

  // Try Triton GPU compiler first (for native HIP/Level Zero GPU builds)
  auto* gpuBackend = getGpuGraphBackend();
  if (gpuBackend) {
    auto status = executeSegmentWithGpuGraph(segment, externalInputs, numExternalInputs, stream);
    if (status == Status::OK) {
      usedGraph = true;
      return Status::OK;
    }
  }

  // Fall back to CPU graph backend (oneDNN/ACL)
  auto status = executeSegmentWithCpuGraph(segment, externalInputs, numExternalInputs, stream);
  if (status == Status::OK) {
    usedGraph = true;
    return Status::OK;
  }

  // Fall back to slot-by-slot
  return executeSegmentSlotBySlot(segment, externalInputs, numExternalInputs, stream);
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

// ── Segment cleanup for rebuild: no-op on CPU ───────────────────────────────

void NativeDynamicShapePlan::platformCleanupSegmentForRebuild(GraphSegment& seg) {
  // No GPU graphs, capture buffers, or workspace to free
}

// ── Plan resource cleanup: no-op on CPU ─────────────────────────────────────

void NativeDynamicShapePlan::platformFreePlanResources() {
  // No GPU-specific resources to free
}

// ── Statistics: no captured graphs on CPU ────────────────────────────────────

int NativeDynamicShapePlan::platformCountCapturedGraphSegments() const {
  return 0;
}

// ── Adaptive splitting: no-op on CPU (no GPU graphs to benefit from) ────────

void NativeDynamicShapePlan::platformMaybeSplitIfEnabled() {
  // No-op: adaptive splitting only benefits GPU graph capture
}

}  // namespace graph
}  // namespace sd

#endif  // !SD_CUDA

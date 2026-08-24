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

//
// TritonGraphBackend core: singleton, configuration, availability, fusibility.
//
// Method implementations are split across focused compilation units:
//   TritonGraphBackend_cache.cpp   — disk cache (read/write PTX + metadata)
//   TritonGraphBackend_compile.cpp — segment compilation (splitting, parallel work-stealing)
//   TritonGraphBackend_kernel.cpp  — single kernel launch + arg table + replay
//   TritonGraphBackend_execute.cpp — segment execution (sub-kernel loop, gaps, diagnostics)
//   TritonGraphBackend_binary.cpp  — GPU binary compilation (IR → PTX → module)
//
// Shared internal helpers live in TritonGraphBackend_internal.h.
//

#include <config.h>

#if HAVE_TRITON

#include <graph/gpu/TritonGraphBackend.h>
#include <graph/gpu/TritonIRBuilder.h>
#include <graph/DspDiagnostics.h>
#include <graph/DspHashUtils.h>
#include <helpers/logger.h>
#include <system/Environment.h>

#include <mutex>

namespace sd {
namespace graph {

// ─── Static member initialization ───────────────────────────────────────────

int TritonGraphBackend::maxParallelCompilations_ = DEFAULT_MAX_PARALLEL_COMPILATIONS;
std::mutex TritonGraphBackend::configMtx_;
thread_local TritonGraphBackend::OrderedRangeExecutor TritonGraphBackend::orderedRangeExecutor_ = nullptr;

// ─── Parallel compilation configuration ─────────────────────────────────────

int TritonGraphBackend::getMaxParallelCompilations() {
  std::lock_guard<std::mutex> lock(configMtx_);

  int configuredThreads = sd::Environment::getInstance().tritonBuildThreads();
  if (configuredThreads > 0 && configuredThreads <= 16) {
    maxParallelCompilations_ = configuredThreads;
  } else {
    maxParallelCompilations_ = DEFAULT_MAX_PARALLEL_COMPILATIONS;
  }

  static int lastReported = -1;
  if (lastReported != maxParallelCompilations_) {
    DSP_DIAG(COMPILE, "TritonGraphBackend: Using %d parallel compilation threads (Environment)",
             maxParallelCompilations_);
    lastReported = maxParallelCompilations_;
  }

  return maxParallelCompilations_;
}

void TritonGraphBackend::setMaxParallelCompilations(int maxThreads) {
  std::lock_guard<std::mutex> lock(configMtx_);
  if (maxThreads > 0 && maxThreads <= 16) {
    maxParallelCompilations_ = maxThreads;
    DSP_DIAG(COMPILE, "TritonGraphBackend: Set max parallel compilations to %d", maxThreads);
  } else {
    DSP_DIAG(COMPILE, "TritonGraphBackend: Invalid maxThreads=%d (must be 1-16), keeping %d",
             maxThreads, maxParallelCompilations_);
  }
}

// ─── Singleton ──────────────────────────────────────────────────────────────

TritonGraphBackend& TritonGraphBackend::getInstance() {
  static TritonGraphBackend* instance = nullptr;
  static std::once_flag initFlag;
  std::call_once(initFlag, []() {
    instance = new TritonGraphBackend();
  });
  return *instance;
}

void TritonGraphBackend::setOrderedRangeExecutor(OrderedRangeExecutor executor) {
  orderedRangeExecutor_ = std::move(executor);
}

void TritonGraphBackend::clearOrderedRangeExecutor() {
  orderedRangeExecutor_ = nullptr;
}

TritonGraphBackend::TritonGraphBackend() = default;

TritonGraphBackend::~TritonGraphBackend() {
  invalidateCache();
}

// ─── Availability ───────────────────────────────────────────────────────────

bool TritonGraphBackend::isAvailable() const {
  return TritonTargetDispatch::isReady();
}

bool TritonGraphBackend::isResolvable(
    const GraphBackendRequest& request) const {
  return request.executionMode == GraphExecutionMode::GEM_TRITON ||
         request.executionMode == GraphExecutionMode::GEM_AUTO;
}

int TritonGraphBackend::resolutionPriority(
    const GraphBackendRequest& request) const {
  return request.executionMode == GraphExecutionMode::GEM_TRITON ? 1000 : 700;
}

GraphBackendPlanningPolicy TritonGraphBackend::planningPolicy(
    const GraphBackendRequest& request) const {
  auto policy = GraphBackend::planningPolicy(request);
  policy.requiresPlatformReplayHandle = true;
  policy.artifactKind = GraphBackendArtifactKind::PLATFORM_REPLAY_REQUIRED;
  policy.separateMatrixMultiplySegments =
      Environment::getInstance().dspMatmulSegmentation();
  return policy;
}

GraphBackendExecutionPolicy TritonGraphBackend::executionPolicy(
    const GraphBackendRequest& request) const {
  (void)request;
  auto& environment = Environment::getInstance();
  GraphBackendExecutionPolicy policy;
  policy.bypassCompiledExecution = environment.tritonSkipKernels();
  policy.allowPlatformGraphReplay = environment.tritonGraphCapture();
  policy.verifyCompiledExecution = environment.tritonVerifyKernels();
  return policy;
}

// ─── Check if all ops in a range are Triton-mappable ────────────────────────

bool TritonGraphBackend::areAllOpsMappable(NativeSlot* slots, int start, int end) {
  for (int i = start; i <= end; i++) {
    if (!TritonIRBuilder::isTritonMappable(slots[i].ident.opName)) {
      return false;
    }
  }
  return true;
}

// ─── Segment fusibility check ───────────────────────────────────────────────
//
// A segment is fusible if it contains at least one Triton-mappable op.
// Non-mappable ops (matmul, gather, etc.) become native ordered ranges inside
// compileSegment() — they run in program order while the mappable
// chains get compiled into Triton kernels.  The previous version required
// ALL ops to be mappable, which caused the post-freeze mega-segment
// (1 segment, 3407 ops) to always fail canFuseSegment → launches=0.

bool TritonGraphBackend::canFuseSegment(NativeSlot* slots, int start, int end) {
  if (!isAvailable()) return false;

  int totalOps = end - start + 1;
  if (totalOps < MIN_MAPPABLE_OPS) return false;

  // Check if at least one op is Triton-mappable.
  // compileSegment handles mixed segments via isFallbackSection.
  for (int i = start; i <= end; i++) {
    if (TritonIRBuilder::isTritonMappable(slots[i].ident.opName)) {
      return true;
    }
  }

  return false;  // No mappable ops at all
}

// ─── Segment-internal dtype hash ────────────────────────────────────────────
//
// Hash the DataType of every output slot owned by the segment [startSlot,
// endSlot].  These are segment-INTERNAL arrays produced by ops within the
// segment.  computeSegmentShapeKey() in NativeDynamicShapePlan_segments.cpp
// hashes dtype only for CROSS-SEGMENT (external) inputs; it skips internal
// outputs because those are not visible at the shapeKey-computation callsites.
//
// Problem this fixes: if the same slots compile once with FLOAT32 K/V and once
// with INT8 K/V (e.g. INT8-V2 KV-cache path), both runs compute an identical
// segShapeKey (shapes match, only internal intermediates differ in dtype).
// They collide in the singleton TritonGraphBackend cache_, causing the second
// compile to get a cache HIT and reuse the FLOAT32 kernel — reading INT8 bytes
// as float, producing garbage outputs.
//
// The hash is FNV-1a over the sequence:
//   for each slot s in [startSlot, endSlot]:
//     for each output o of slot s:
//       mix(outputSlots[wiring.outputSlotIndices[o]]->dataType())
//
// Slots whose output slot index is out of range or whose array is null are
// skipped (contributes nothing — safe for partially-materialised plans).
// The hash value 0 (empty/all-null segment) is a valid sentinel meaning "no
// internal dtype information"; lookup sites that cannot compute the hash (e.g.
// copyConsolidatedArgTableToDevice which lacks outputSlots) pass 0 and will
// get a cache miss — which is benign (they return early without error).

/*static*/
size_t TritonGraphBackend::computeSegInternalDtypeHash(NativeSlot* slots,
                                                        int startSlot, int endSlot,
                                                        NDArray** externalInputs,
                                                        int numExternalInputs,
                                                        NDArray** outputSlots,
                                                        int totalOutputSlots) {
  if (slots == nullptr) return 0;
  uint64_t hash = dsp::FNV1A64_OFFSET_BASIS;

  // Output slot indices produced INSIDE the segment — consumers of these are
  // segment-internal, not cross-segment inputs.
  std::unordered_set<int> segOutputSlots;
  for (int s = startSlot; s <= endSlot; s++) {
    const NativeSlot& slot = slots[s];
    for (int o = 0; o < slot.wiring.numOutputs; o++) {
      segOutputSlots.insert(slot.wiring.outputSlotIndices[o]);
    }
  }

  // Mix the dtype of every cross-segment input: external input arrays (srcIdx < 0)
  // and slot-outputs produced outside the segment. Mirrors computeSegmentShapeKey()'s
  // input traversal so an INT8 vs FLOAT32 KV cache of identical shape yields distinct
  // cache entries.
  bool sawInt8 = false;
  for (int s = startSlot; s <= endSlot; s++) {
    const NativeSlot& slot = slots[s];
    for (int i = 0; i < slot.wiring.numInputs; i++) {
      int srcIdx = slot.wiring.inputSourceIndices[i];
      NDArray* arr = nullptr;
      if (srcIdx < 0) {
        int extIdx = -(srcIdx + 1);
        if (externalInputs != nullptr && extIdx >= 0 && extIdx < numExternalInputs)
          arr = externalInputs[extIdx];
      } else if (segOutputSlots.find(srcIdx) == segOutputSlots.end()) {
        if (outputSlots != nullptr && srcIdx >= 0 && srcIdx < totalOutputSlots)
          arr = outputSlots[srcIdx];
      }
      if (arr != nullptr) {
        DataType dt = arr->dataType();
        if (dt == DataType::INT8) sawInt8 = true;
        dsp::fnv1aMixValue(hash, static_cast<uint64_t>(dt));
      }
    }
  }

  if (sawInt8) {
    DSP_DIAG(JIT, "computeSegInternalDtypeHash: seg[%d-%d] INT8 cross-input -> dtypeHash=%zu",
             startSlot, endSlot, static_cast<size_t>(hash));
  }
  return static_cast<size_t>(hash);
}

}  // namespace graph
}  // namespace sd

#endif  // HAVE_TRITON

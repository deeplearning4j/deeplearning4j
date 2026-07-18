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

// Wired into NativeDynamicShapePlan::getGpuGraphBackend()
// (graph/impl/NativeDynamicShapePlan_gpubackend.cpp), mirroring the
// SD_TPU / HAVE_HEXAGON_MLIR branches: explicit GEM_HIP_GRAPHS selects it,
// GEM_AUTO tries it after the JIT backends when isAvailable() (libamdhip64.so
// dlopen succeeds — false on non-AMD hosts, so AUTO on NVIDIA is unaffected).
//
// Gate: native HIP builds (SD_HIP) and ZLUDA+AMD builds (ZLUDA_TARGET_AMD /
// HAVE_MIOPEN — ROCm is installed there, and ZLUDA streams are hipStream_t
// underneath, so hipStreamBeginCapture on the plan stream records both
// ZLUDA-translated launches and directly-launched HIP Triton kernels).

#ifndef LIBND4J_HIP_GRAPH_BACKEND_H
#define LIBND4J_HIP_GRAPH_BACKEND_H

#if defined(SD_HIP) || defined(ZLUDA_TARGET_AMD) || defined(HAVE_MIOPEN)

#include <graph/GraphBackend.h>
#include <graph/IslandCapturePolicy.h>

#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace sd {
namespace graph {

// Forward declarations — avoid circular include with NativeDynamicShapePlan.h
struct NativeSlot;
struct GraphSegment;

/**
 * Per-island compiled handle stored inside a GraphSegment.
 *
 * A single segment may produce multiple islands (each captured as one
 * hipGraph_t) plus zero or more eager gaps between them.  The handles
 * are stored in order and replayed in order during executeSegment().
 */
struct HipIslandHandle {
  void* graph    = nullptr;  // hipGraph_t        (opaque)
  void* graphExec = nullptr; // hipGraphExec_t     (opaque, from hipGraphInstantiate)
  int   beginSlot = -1;      // inclusive slot index in the parent segment
  int   endSlot   = -1;      // exclusive slot index in the parent segment
};

/**
 * Opaque per-segment state bag stored on the segment's replayHandle or in
 * the backend's internal cache.  Contains all compiled island handles for
 * one segment so executeSegment() can replay them in order without repeating
 * the policy partition.
 */
struct HipSegmentCapture {
  // Ordered island handles (capture=true ranges only).
  std::vector<HipIslandHandle> islands;

  // Ordered gap ranges (capture=false) for the caller to execute eagerly.
  // begin/end are slot indices in the parent segment.
  struct GapRange { int begin; int end; };
  std::vector<GapRange> gaps;

  // Ordered replay plan: alternating islands and gaps as they appear in the
  // segment, so executeSegment() can drive them in program order.
  // Each entry is either an island index (>=0) or a gap index (encoded as
  // -(gapIdx+1)).  Positive = island index, negative = -(gapIdx+1).
  std::vector<int> replayOrder;

  // The shape key this capture was built for.  Used to detect stale captures.
  LongType compiledShapeKey = 0;
};

/**
 * HipGraphBackend — GraphBackend implementation for AMD ROCm hipGraph replay.
 *
 * Architecture:
 *   1. canFuseSegment()  — delegates to IslandCapturePolicy::partition(forRocm())
 *      to check whether any op in the range is capturable.
 *   2. compileSegment()  — partitions the segment into ROCm-safe islands and
 *      eager gaps via IslandCapturePolicy(forRocm()); for each island:
 *        hipStreamBeginCapture → (caller's kernel launches replay here) →
 *        hipStreamEndCapture → hipGraphInstantiate
 *      Gap slot ranges are recorded for eager re-execution.
 *   3. executeSegment()  — replays each island via hipGraphLaunch(exec,stream);
 *      for gap ranges, signals the caller via the audit/Status return so they
 *      can dispatch those slots eagerly.
 *   4. invalidateCache() — destroys all hipGraph_t and hipGraphExec_t handles
 *      and clears the per-segment capture map.
 *
 * Hardware-dependent paths:
 *   All HIP API calls (hipStreamBeginCapture, hipGraphLaunch, etc.) are
 *   made through HipRuntimeManager which dlopens libamdhip64.so at runtime.
 *   On non-AMD hosts, isAvailable() returns false and the backend is a no-op.
 *   Paths that require an AMD GPU are commented "Requires ROCm/HIP runtime".
 *
 * Singleton: consistent with TpuGraphBackend and other graph backends.
 */
class SD_LIB_EXPORT HipGraphBackend : public GraphBackend {
 public:
  /**
   * Get the singleton instance.
   */
  static HipGraphBackend& getInstance();

  // ── GraphBackend interface ────────────────────────────────────────────────

  /**
   * Returns true if HipRuntimeManager::isAvailable() — i.e. libamdhip64.so
   * loaded and all required symbols resolved.
   * Always false on non-AMD / non-ROCm hosts.
   */
  bool isAvailable() const override;

  /**
   * Returns true if the range [start, end) contains at least one slot that
   * IslandCapturePolicy would place in a capture=true island under the ROCm
   * profile.  A range with only non-capturable (gap) slots still returns false.
   */
  bool canFuseSegment(NativeSlot* slots, int start, int end) override;

  /**
   * Compile a segment into per-island hipGraph_t handles.
   *
   * For each capture=true island from IslandCapturePolicy(forRocm()):
   *   1. hipStreamBeginCapture on the caller-provided stream (cast from void*).
   *   2. Execute the island's slot range via slot-by-slot dispatch INTO the
   *      capture stream so the kernel launches are recorded.
   *   3. hipStreamEndCapture → hipGraphInstantiate → store HipIslandHandle.
   * Gap ranges are recorded as GapRange entries in HipSegmentCapture.
   *
   * Requires ROCm/HIP runtime — on non-AMD hosts returns false immediately.
   *
   * The compiled HipSegmentCapture is stored in the cache keyed by
   * seg.def.startSlot so executeSegment() can retrieve it.
   */
  bool compileSegment(GraphSegment& seg, NativeSlot* slots,
                      NDArray** externalInputs, int numExternalInputs,
                      NDArray** outputSlots, int totalOutputSlots,
                      LongType shapeKey,
                      int totalSlots = 0,
                      int* requestedOutputSlotIndices = nullptr,
                      int numRequestedOutputs = 0) override;

  /**
   * Execute a previously compiled segment.
   *
   * For each entry in the compiled HipSegmentCapture in program order:
   *   - Island entry: hipGraphLaunch(exec, stream).
   *   - Gap entry: returns a non-OK Status signalling the caller to execute
   *     those slot indices eagerly.  The current implementation executes gap
   *     slots eagerly inline (slot-by-slot dispatch into the stream).
   *
   * Requires ROCm/HIP runtime.
   */
  Status executeSegment(GraphSegment& seg, NativeSlot* slots,
                        NDArray** externalInputs, int numExternalInputs,
                        NDArray** outputSlots, int totalOutputSlots,
                        void* stream) override;

  /**
   * Destroy all cached hipGraph_t / hipGraphExec_t handles.
   * Call on shape change, plan reset, or OOM recovery.
   */
  void invalidateCache() override;

  /**
   * Backend name for diagnostics.
   */
  const char* name() const override { return "HIP (ROCm)"; }

  /**
   * Return the compilation audit from the most recent compileSegment() call.
   */
  std::vector<CompilationAuditEntry> getLastCompilationAudit() const override;

 private:
  HipGraphBackend();
  ~HipGraphBackend() override;

  // Non-copyable
  HipGraphBackend(const HipGraphBackend&) = delete;
  HipGraphBackend& operator=(const HipGraphBackend&) = delete;

  /**
   * Destroy one HipIslandHandle's HIP resources.
   * Requires ROCm/HIP runtime.
   */
  void destroyHandle(HipIslandHandle& handle);

  /**
   * Destroy all handles in one segment capture and remove it from the cache.
   * Requires ROCm/HIP runtime.
   */
  void destroyCapture(HipSegmentCapture& cap);

  // ── Internal state ────────────────────────────────────────────────────────

  // Per-segment compiled capture, keyed by seg.def.startSlot.
  // Access under mutex_.
  std::unordered_map<int, HipSegmentCapture> captureCache_;

  // Compilation audit for the most recent compileSegment() call.
  std::vector<CompilationAuditEntry> lastAudit_;

  // Mutex protecting captureCache_ and lastAudit_.
  mutable std::mutex mutex_;
};

}  // namespace graph
}  // namespace sd

#endif  // SD_HIP
#endif  // LIBND4J_HIP_GRAPH_BACKEND_H

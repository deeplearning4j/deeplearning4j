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

#ifndef LIBND4J_GRAPH_BACKEND_H
#define LIBND4J_GRAPH_BACKEND_H

#include <array/NDArray.h>
#include <system/common.h>

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace sd {
namespace graph {

// Forward declarations — avoid circular include with NativeDynamicShapePlan.h
struct NativeSlot;
struct GraphSegment;
enum class GraphExecutionMode : int;

/**
 * Backend-neutral artifact policy shared by graph compilers and recorders.
 *
 * This value is passed explicitly through lowering. It must not be recovered
 * from thread-local or platform-global state because the same DSP lifecycle is
 * used by desktop, mobile, compiler, and command-buffer backends.
 */
struct GraphCompilationPolicy {
  bool runtimeCompilationAllowed = true;
  std::string runtimeArtifactDirectory;

  bool requiresPrecompiledArtifact() const {
    return !runtimeCompilationAllowed;
  }
};

/**
 * Backend-neutral request used by the DSP resolver.
 *
 * A backend owns the policy for deciding whether it can satisfy a requested
 * execution mode. NativeDynamicShapePlan only supplies the request and a
 * catalog of compiled backend implementations.
 */
struct GraphBackendRequest {
  GraphExecutionMode executionMode;
  bool runtimeCompilationAllowed = true;
  std::string runtimeArtifactDirectory;
  std::string deviceCompilationCacheDirectory;
  std::string deviceCompilationCacheModelKey;

  GraphCompilationPolicy compilationPolicy() const {
    return {runtimeCompilationAllowed, runtimeArtifactDirectory};
  }
};

/**
 * Backend-owned hints consumed by the generic plan lifecycle.
 *
 * These are optimization and preparation requirements, not backend selection.
 * NativeDynamicShapePlan applies the same mechanics regardless of which backend
 * supplied the policy.
 */
enum class GraphBackendArtifactKind : uint8_t {
  UNSPECIFIED = 0,
  DIRECT_COMPILED = 1,
  BACKEND_REPLAY_HANDLE = 2,
  PLATFORM_REPLAY_REQUIRED = 3,
};

struct GraphBackendPlanningPolicy {
  bool requiresShapePrePass = false;
  bool requiresSuccessfulShapePrePass = false;
  bool precompileBeforeFirstExecution = false;
  bool allowsShapeOnlyWarmup = false;
  // Split generic DSP segments whenever the ordered set of backends capable of
  // lowering an individual slot changes. Backends own the slot capability
  // predicate; the plan owns the partitioning mechanics.
  bool requiresCapabilityPartitioning = false;
  // The request must never escape to top-level slot execution when no backend
  // admits a numeric range. Mixed native ranges remain backend-owned; this gate
  // rejects only unresolved plan segments.
  bool requiresCompleteLowering = false;
  // The lowered artifact must be wrapped in a platform replay handle before the
  // plan may enter replay steady state. This is a backend property, not a CPU/GPU
  // distinction; direct graph runtimes leave it false.
  bool requiresPlatformReplayHandle = false;
  GraphBackendArtifactKind artifactKind =
      GraphBackendArtifactKind::UNSPECIFIED;
  // True when successful backend execution leaves every framework output slot
  // in the lowered range materialized and valid. Backends that intentionally
  // keep island-internal values private validate only their published boundary
  // tensors and set this false.
  bool materializesAllFrameworkSlots = true;
  // Split matrix-multiply/attention ranges from neighboring elementwise ranges.
  // The backend requests the topology; the plan owns the generic split mechanics.
  bool separateMatrixMultiplySegments = false;
  int preferredMaxSegmentOps = 0;  // zero means no backend-specific bound
};

/**
 * Backend-owned controls consumed by the generic execution lifecycle.
 *
 * The plan applies these mechanics without knowing which backend requested
 * them. Backend-specific environment switches and diagnostics must be
 * translated into this contract by the backend implementation.
 */
struct GraphBackendExecutionPolicy {
  // Execute the plan through the portable slot path without dispatching
  // compiled backend artifacts. Intended for backend verification/debug modes.
  bool bypassCompiledExecution = false;
  // Permit the platform replay layer to wrap and replay lowered artifacts.
  bool allowPlatformGraphReplay = false;
  // Route through the full execution path so compiled results can be verified.
  bool verifyCompiledExecution = false;
};

/**
 * Backend-agnostic compilation audit entry.
 * Tracks whether each op in a segment was compiled by the graph backend
 * or silently skipped. Skipped ops produce stale outputs on graph replay.
 */
struct CompilationAuditEntry {
  int slotIndex = -1;
  std::string opName;
  bool wasCompiled = false;    // true if backend compiled this op via its graph API
  bool isNativeHandled = false; // true if backend will execute this op natively (not an error)
  std::string reason;          // why it was skipped (e.g., "unmappable op kind")
};

/**
 * Abstract graph backend interface for hardware-specific graph APIs.
 *
 * Graph compiler and recorder backends follow the same pattern:
 * 1. Detect fusible segments in the plan
 * 2. Compile the segment into a hardware-specific graph
 * 3. Execute the compiled graph
 *
 * Backend resolution is data-driven:
 * - isResolvable() declares whether this implementation accepts a request
 * - canResolveSegment() validates the concrete slot range
 * - compileSegment() lowers that accepted range into backend-specific form
 * - executeSegment() dispatches the lowered artifact
 *
 * NativeDynamicShapePlan owns only the generic filtering/cascade lifecycle.
 */
class SD_LIB_EXPORT GraphBackend {
 public:
  virtual ~GraphBackend() = default;

  using NativeSlotExecutor = std::function<Status(int startSlot, int endSlot)>;

  /**
   * Check if this backend is available at runtime.
   */
  virtual bool isAvailable() const = 0;

  /**
   * Request-level capability gate. Implementations opt into execution modes;
   * the shared resolver contains no backend-specific mode switch.
   */
  virtual bool isResolvable(const GraphBackendRequest& request) const {
    (void)request;
    return false;
  }

  /**
   * Higher values are attempted first when more than one backend resolves the
   * same request. Equal priorities retain catalog registration order.
   */
  virtual int resolutionPriority(const GraphBackendRequest& request) const {
    (void)request;
    return 0;
  }

  /**
   * Return backend-owned lifecycle hints for this request. The plan applies
   * these through one backend-neutral preparation and segmentation path.
   */
  virtual GraphBackendPlanningPolicy planningPolicy(
      const GraphBackendRequest& request) const {
    (void)request;
    return {};
  }

  /**
   * Return backend-owned runtime controls for this request. The shared plan
   * never reads backend-specific environment flags directly.
   */
  virtual GraphBackendExecutionPolicy executionPolicy(
      const GraphBackendRequest& request) const {
    (void)request;
    return {};
  }

  /**
   * Check if a contiguous range of slots can be lowered by this backend.
   * Existing implementations express this through canFuseSegment(); the
   * resolver-facing name keeps selection semantics independent of graph API.
   */
  virtual bool canResolveSegment(NativeSlot* slots, int start, int end) {
    return canFuseSegment(slots, start, end);
  }

  virtual bool canResolveSegment(const GraphBackendRequest& request,
                                 NativeSlot* slots, int start, int end) {
    (void)request;
    return canResolveSegment(slots, start, end);
  }

  /**
   * Backend-owned operation capability used by the shared segment partitioner.
   * The default reuses single-slot segment admission. Backends whose segment
   * admission also includes profitability constraints override this method so
   * operation validity remains distinct from segment profitability.
   */
  virtual bool canResolveSlot(const GraphBackendRequest& request,
                              NativeSlot* slots, int slotIndex) {
    return canResolveSegment(request, slots, slotIndex, slotIndex);
  }

  virtual bool canFuseSegment(NativeSlot* slots, int start, int end) = 0;

  /**
   * Optional native-range executor used by mixed lowered/native segments.
   * Backends that do not need it inherit the no-op implementation.
   */
  virtual void setNativeSlotExecutor(NativeSlotExecutor executor) {
    (void)executor;
  }
  virtual void clearNativeSlotExecutor() {}

  /**
   * Compile a graph segment into backend-specific graph representation.
   * Returns true on success.
   *
   * @param seg The segment to compile
   * @param slots All slots in the plan
   * @param externalInputs External input arrays (constants, variables, placeholders)
   * @param numExternalInputs Count of external inputs
   * @param outputSlots The plan's output slot array (intermediate results)
   * @param totalOutputSlots Total number of output slots
   * @param shapeKey Shape signature for cache invalidation
   */
  virtual bool compileSegment(GraphSegment& seg, NativeSlot* slots,
                              NDArray** externalInputs, int numExternalInputs,
                              NDArray** outputSlots, int totalOutputSlots,
                              LongType shapeKey,
                              int totalSlots = 0,
                              int* requestedOutputSlotIndices = nullptr,
                              int numRequestedOutputs = 0) = 0;

  virtual bool compileSegment(const GraphBackendRequest& request,
                              GraphSegment& seg, NativeSlot* slots,
                              NDArray** externalInputs, int numExternalInputs,
                              NDArray** outputSlots, int totalOutputSlots,
                              LongType shapeKey, int totalSlots,
                              int* requestedOutputSlotIndices,
                              int numRequestedOutputs) {
    (void)request;
    return compileSegment(seg, slots, externalInputs, numExternalInputs,
                          outputSlots, totalOutputSlots, shapeKey, totalSlots,
                          requestedOutputSlotIndices, numRequestedOutputs);
  }

  /**
   * Execute a previously compiled segment.
   *
   * @param seg The compiled segment
   * @param slots All slots in the plan
   * @param externalInputs External input arrays
   * @param numExternalInputs Count of external inputs
   * @param outputSlots The plan's output slot array (written by execution)
   * @param totalOutputSlots Total number of output slots
   * @param stream Backend-specific execution stream (nullptr for CPU)
   *
   * Status contract: BAD_GRAPH is reserved for rejection before any backend
   * work starts. Once execution begins, failures must use another status so the
   * resolver cannot retry a different backend after partial execution.
   */
  virtual Status executeSegment(GraphSegment& seg, NativeSlot* slots,
                                NDArray** externalInputs, int numExternalInputs,
                                NDArray** outputSlots, int totalOutputSlots,
                                void* stream) = 0;

  virtual Status executeSegment(const GraphBackendRequest& request,
                                GraphSegment& seg, NativeSlot* slots,
                                NDArray** externalInputs, int numExternalInputs,
                                NDArray** outputSlots, int totalOutputSlots,
                                void* stream) {
    (void)request;
    return executeSegment(seg, slots, externalInputs, numExternalInputs,
                          outputSlots, totalOutputSlots, stream);
  }

  /**
   * Invalidate all cached compiled graphs (e.g., on shape change).
   */
  virtual void invalidateCache() = 0;

  /**
   * Get the backend name for diagnostics.
   */
  virtual const char* name() const = 0;

  /**
   * Get the compilation audit for the most recent compileSegment() call.
   * Each entry shows whether a slot was compiled by the backend or skipped.
   * Skipped ops will produce stale outputs on graph replay.
   */
  virtual std::vector<CompilationAuditEntry> getLastCompilationAudit() const = 0;
};

}  // namespace graph
}  // namespace sd

#endif  // LIBND4J_GRAPH_BACKEND_H

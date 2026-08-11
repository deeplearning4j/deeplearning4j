/* ******************************************************************************
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#ifndef LIBND4J_GRAPH_BACKEND_RESOLVER_H
#define LIBND4J_GRAPH_BACKEND_RESOLVER_H

#include <graph/GraphBackend.h>

#include <algorithm>
#include <cstddef>
#include <vector>

namespace sd {
namespace graph {

/**
 * Shared resolver for every GraphBackend implementation.
 *
 * The catalog is the build-composition boundary: it lists implementations that
 * were compiled into the binary. All runtime capability and execution-mode
 * policy remains on GraphBackend itself.
 */
class GraphBackendResolver {
 public:
  static std::vector<GraphBackend*> resolve(
      const GraphBackendRequest& request,
      const std::vector<GraphBackend*>& catalog) {
    struct Candidate {
      GraphBackend* backend;
      int priority;
    };

    std::vector<Candidate> candidates;
    candidates.reserve(catalog.size());
    for (size_t i = 0; i < catalog.size(); ++i) {
      GraphBackend* backend = catalog[i];
      if (backend == nullptr || !backend->isAvailable() ||
          !backend->isResolvable(request)) {
        continue;
      }
      candidates.push_back(
          Candidate{backend, backend->resolutionPriority(request)});
    }

    std::stable_sort(
        candidates.begin(), candidates.end(),
        [](const Candidate& lhs, const Candidate& rhs) {
          return lhs.priority > rhs.priority;
        });

    std::vector<GraphBackend*> resolved;
    resolved.reserve(candidates.size());
    for (const auto& candidate : candidates) {
      resolved.push_back(candidate.backend);
    }
    return resolved;
  }

  /**
   * Apply the backend-neutral segment admission contract to a request-level
   * candidate chain. A previously selected backend is tried first when it is
   * still part of the chain; every remaining backend retains resolver order.
   */
  static std::vector<GraphBackend*> resolveSegment(
      const GraphBackendRequest& request,
      const std::vector<GraphBackend*>& candidates, NativeSlot* slots,
      int start, int end, GraphBackend* preferred = nullptr) {
    std::vector<GraphBackend*> admitted;
    admitted.reserve(candidates.size());

    auto admit = [&](GraphBackend* backend) {
      if (backend == nullptr ||
          std::find(candidates.begin(), candidates.end(), backend) ==
              candidates.end() ||
          std::find(admitted.begin(), admitted.end(), backend) !=
              admitted.end() ||
          !backend->isAvailable() || !backend->isResolvable(request) ||
          !backend->canResolveSegment(request, slots, start, end)) {
        return;
      }
      admitted.push_back(backend);
    };

    admit(preferred);
    for (GraphBackend* backend : candidates) {
      admit(backend);
    }
    return admitted;
  }

  /**
   * Return the resolver-ordered capability set for one operation. This is the
   * backend-neutral primitive used to partition a generic DSP slot stream into
   * ranges with coherent lowering candidates.
   */
  static std::vector<GraphBackend*> resolveSlot(
      const GraphBackendRequest& request,
      const std::vector<GraphBackend*>& candidates, NativeSlot* slots,
      int slotIndex) {
    std::vector<GraphBackend*> admitted;
    admitted.reserve(candidates.size());
    for (GraphBackend* backend : candidates) {
      if (backend == nullptr || !backend->isAvailable() ||
          !backend->isResolvable(request) ||
          !backend->canResolveSlot(request, slots, slotIndex)) {
        continue;
      }
      admitted.push_back(backend);
    }
    return admitted;
  }

  struct LoweringAttempt {
    GraphBackend* backend = nullptr;
    bool succeeded = false;
    std::vector<CompilationAuditEntry> audit;
  };

  struct LoweringResult {
    GraphBackend* backend = nullptr;
    std::vector<LoweringAttempt> attempts;

    bool anyBackendAdmitted() const { return !attempts.empty(); }
    bool succeeded() const { return backend != nullptr; }
  };

  /**
   * Run the single generic admission/lowering cascade used by every platform.
   * Backend implementations own capability checks and lowering; the resolver
   * owns ordering and fallback mechanics.
   */
  static LoweringResult lowerSegment(
      const GraphBackendRequest& request,
      const std::vector<GraphBackend*>& candidates, GraphBackend* preferred,
      GraphSegment& segment, NativeSlot* slots, int start, int end,
      NDArray** externalInputs, int numExternalInputs, NDArray** outputSlots,
      int totalOutputSlots, LongType shapeKey, int totalSlots,
      int* requestedOutputSlotIndices, int numRequestedOutputs) {
    LoweringResult result;
    const auto admitted =
        resolveSegment(request, candidates, slots, start, end, preferred);
    result.attempts.reserve(admitted.size());
    for (GraphBackend* backend : admitted) {
      const bool succeeded = backend->compileSegment(
          request, segment, slots, externalInputs, numExternalInputs,
          outputSlots, totalOutputSlots, shapeKey, totalSlots,
          requestedOutputSlotIndices, numRequestedOutputs);
      result.attempts.push_back(
          LoweringAttempt{backend, succeeded, backend->getLastCompilationAudit()});
      if (succeeded) {
        result.backend = backend;
        break;
      }
    }
    return result;
  }

  /**
   * Conservatively combine lifecycle requirements for an ordered fallback
   * chain. The plan must satisfy every prerequisite that may be required by a
   * later candidate, because admission and lowering are segment-specific.
   */
  static GraphBackendPlanningPolicy aggregatePlanningPolicy(
      const GraphBackendRequest& request,
      const std::vector<GraphBackend*>& candidates) {
    GraphBackendPlanningPolicy combined;
    for (GraphBackend* backend : candidates) {
      if (backend == nullptr) continue;
      const auto policy = backend->planningPolicy(request);
      combined.requiresShapePrePass |= policy.requiresShapePrePass;
      combined.requiresSuccessfulShapePrePass |=
          policy.requiresSuccessfulShapePrePass;
      combined.precompileBeforeFirstExecution |=
          policy.precompileBeforeFirstExecution;
      combined.allowsShapeOnlyWarmup |= policy.allowsShapeOnlyWarmup;
      combined.requiresCapabilityPartitioning |=
          policy.requiresCapabilityPartitioning;
      combined.requiresCompleteLowering |= policy.requiresCompleteLowering;
      combined.requiresPlatformReplayHandle |=
          policy.requiresPlatformReplayHandle;
      combined.separateMatrixMultiplySegments |=
          policy.separateMatrixMultiplySegments;
      if (policy.preferredMaxSegmentOps > 0 &&
          (combined.preferredMaxSegmentOps == 0 ||
           policy.preferredMaxSegmentOps <
               combined.preferredMaxSegmentOps)) {
        combined.preferredMaxSegmentOps = policy.preferredMaxSegmentOps;
      }
    }
    return combined;
  }

  /**
   * Conservatively combine runtime controls for an ordered fallback chain.
   * Any candidate may become the selected backend for a concrete segment, so
   * the generic lifecycle honors every requested safety/debug control.
   */
  static GraphBackendExecutionPolicy aggregateExecutionPolicy(
      const GraphBackendRequest& request,
      const std::vector<GraphBackend*>& candidates) {
    GraphBackendExecutionPolicy combined;
    for (GraphBackend* backend : candidates) {
      if (backend == nullptr) continue;
      const auto policy = backend->executionPolicy(request);
      combined.bypassCompiledExecution |= policy.bypassCompiledExecution;
      combined.allowPlatformGraphReplay |= policy.allowPlatformGraphReplay;
      combined.verifyCompiledExecution |= policy.verifyCompiledExecution;
    }
    return combined;
  }
};

}  // namespace graph
}  // namespace sd

#endif  // LIBND4J_GRAPH_BACKEND_RESOLVER_H

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

// NativeDynamicShapePlan_gpubackend.cpp — Platform-agnostic GPU backend dispatch.
//
// Contains:
//   - segDispatchWarmup(): slot-by-slot warmup before compilation
//   - segDispatchCompile(): backend compilation and audit
//   - cleanupSegmentForRebuild(): segment cleanup with diagnostics
//   - hasCompositeHandles(): composite replay readiness check
//
// CUDA-specific execution (sync, capture, replay, eviction) is in
// NativeDynamicShapePlan_gpubackend.cu, compiled only by NVCC.

#include <graph/NativeDynamicShapePlan.h>
#include <graph/GraphBackendResolver.h>
#include <graph/ModeContract.h>
#include <graph/DspDiagnostics.h>
#include <graph/DspSegmentLifecycle.h>
#include <graph/gpu/DspCudaDispatch.h>
#include <system/op_boilerplate.h>
#include <system/Environment.h>
#include <config.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

namespace sd {
namespace graph {

// File-level alias for the nested enum.
using SegmentLifecycleState = GraphSegmentExec::SegmentLifecycleState;

// ── cleanupSegmentForRebuild wrapper ─────────────────────────────────────────
// Wraps platformCleanupSegmentForRebuild() with DSP_DIAG tracing and
// phase-safety warning. Called from all cleanup sites.
void NativeDynamicShapePlan::cleanupSegmentForRebuild(GraphSegment& seg,
                                                      const char* reason) {
  bool isMonolithicReplaying = (seg.exec.segPhase.isSealed() &&
                                seg.exec.replayHandle != nullptr &&
                                seg.exec.replayHandle->isReady());
  bool isCompositeReplaying  = (seg.exec.segPhase.isSealed() &&
                                hasCompositeHandles(seg));
  if (isMonolithicReplaying || isCompositeReplaying) {
    DSP_DIAG(EXECUTE,
             "cleanupSegmentForRebuild[%s]: WARNING seg[%d-%d] is in REPLAYING state "
             "(%s capture) — cleanup during active replay is dangerous",
             reason, seg.def.startSlot, seg.def.endSlot,
             isCompositeReplaying ? "composite" : "monolithic");
  }
  DSP_DIAG(EXECUTE,
           "cleanupSegmentForRebuild[%s]: seg[%d-%d] lifecycle=%s hasReplay=%d "
           "compositeHandles=%d mergedGroups=%d execCount=%d",
           reason, seg.def.startSlot, seg.def.endSlot,
           seg.exec.displayPhaseName(),
           seg.exec.replayHandle ? 1 : 0,
           static_cast<int>(seg.exec.compositeReplaySchedule.compositeReplayHandles.size()),
           static_cast<int>(seg.exec.compositeReplaySchedule.mergedReplayHandles.size()),
           seg.exec.executionCount);
  platformCleanupSegmentForRebuild(seg);
}

// ── dumpSegmentGraphState ───────────────────────────────────────────────────
// Dumps full graph state for all segments as structured JSON to DspDiagnostics.
void NativeDynamicShapePlan::dumpSegmentGraphState(const char* tag) const {
  std::string json = "{\"tag\":\"";
  json += (tag ? tag : "unknown");
  json += "\",\"planExecCount\":";
  json += std::to_string(executeCount_);
  json += ",\"planPhase\":\"";
  json += planLifecycle_.displayName();
  json += "\",\"segments\":[";

  for (size_t i = 0; i < segments_.size(); i++) {
    const auto& seg = segments_[i];
    if (i > 0) json += ",";
    json += "{\"idx\":";
    json += std::to_string(i);
    json += ",\"slots\":[";
    json += std::to_string(seg.def.startSlot);
    json += ",";
    json += std::to_string(seg.def.endSlot);
    json += "],\"phase\":\"";
    json += seg.exec.displayPhaseName();
    json += "\",\"outcome\":\"";
    json += segmentExecOutcomeName(seg.exec.outcome);
    json += "\",\"execCount\":";
    json += std::to_string(seg.exec.executionCount);
    json += ",\"hasHandle\":";
    json += (seg.exec.replayHandle ? "true" : "false");
    json += ",\"handleReady\":";
    json += (seg.exec.replayHandle && seg.exec.replayHandle->isReady() ? "true" : "false");
    json += ",\"capturedInputAddrKey\":";
    json += std::to_string(seg.exec.capturedInputAddrKey);
    json += ",\"capturedSlotAddrHash\":";
    json += std::to_string(seg.exec.capturedSlotAddrHash);
    json += ",\"tracker\":";
    json += seg.exec.handleTracker.toJsonSummary(seg.def.startSlot, seg.def.endSlot);
    json += "}";
  }

  json += "]}";

  DspDiagnostics::getInstance().recordGraphStateDump(tag, json.c_str());
  // Also print to stderr for immediate visibility when debugging
  sd_debug("=== GRAPH STATE DUMP [%s] ===\n%s\n", tag, json.c_str());
}

// ═══════════════════════════════════════════════════════════════════════════════
// hasCompositeHandles — check if a composite schedule is ready for replay
// ═══════════════════════════════════════════════════════════════════════════════
bool NativeDynamicShapePlan::hasCompositeHandles(const GraphSegment& seg) const {
  auto& sched = seg.exec.compositeReplaySchedule;
  bool hasIslandUnits = false;
  // Check merged replay handles — at least one merged group must be ready
  for (auto& h : sched.mergedReplayHandles) {
    if (h != nullptr && h->isReady()) return true;
  }
  // Fallback: check individual composite handles (backward compat)
  for (auto& u : sched.units) {
    if (u.kind == REPLAY_UNIT_TRITON_ISLAND && u.mergedGroupId < 0) {
      hasIslandUnits = true;
      int idx = u.islandIndex;
      if (idx >= 0 && idx < static_cast<int>(sched.compositeReplayHandles.size()) &&
          sched.compositeReplayHandles[idx] != nullptr &&
          sched.compositeReplayHandles[idx]->isReady()) {
        return true;
      }
    } else if (u.kind == REPLAY_UNIT_TRITON_ISLAND) {
      hasIslandUnits = true;
    }
  }

  // A sealed schedule with no islands is an intentional live-gap-only
  // composite plan. It has no CUDA replay handles by construction: every unit
  // executes live in program order. Treat it as replay-ready so it cannot fall
  // through to monolithic capture and bake value-dependent range/create data.
  if (!sched.units.empty() && !hasIslandUnits && seg.exec.segPhase.isSealed() &&
      seg.exec.replayUnitCount == static_cast<int>(sched.units.size())) {
    return true;
  }
  return false;
}

// ─── DSP Verify Helpers ────────────────────────────────────────────────────

// Source type name for diagnostics
static const char* sourceTypeName(int8_t st) {
  switch (static_cast<NativeSourceType>(st)) {
    case SOURCE_CONSTANT: return "CONSTANT";
    case SOURCE_VARIABLE: return "VARIABLE";
    case SOURCE_PLACEHOLDER: return "PLACEHOLDER";
    case SOURCE_OP_OUTPUT: return "OP_OUTPUT";
    default: return "UNKNOWN";
  }
}

void NativeDynamicShapePlan::clearGpuBackendFailedCache() {
  dspTritonClearFailedCache();
}

// ═══════════════════════════════════════════════════════════════════════════
// segDispatchWarmup — NEEDS_WARMUP state handler
// Runs slot-by-slot to populate shape caches and output slots.
// On success, transitions to NEEDS_COMPILE (or skips compile if already compiled).
// ═══════════════════════════════════════════════════════════════════════════
Status NativeDynamicShapePlan::segDispatchWarmup(
    GraphSegment& seg, NDArray** externalArrays, int numExt, void* stream) {
  // ── Plan structure dump (one-time, on first segment execution) ─────────
  if (Environment::getInstance().tritonVerifyKernels()) {
    DSP_DIAG(VERIFY, "=== PLAN STRUCTURE ===");
    DSP_DIAG(VERIFY, "Plan: %d steps, %d output slots, %d external inputs, %d segments",
             numSlots_, totalOutputSlots_, numExternalInputs_, (int)segments_.size());
    for (int si = 0; si < (int)segments_.size(); si++) {
      auto& s = segments_[si];
      DSP_DIAG(VERIFY, "Segment %d: slots [%d..%d] (%d ops) %s",
               si, s.def.startSlot, s.def.endSlot, s.def.endSlot - s.def.startSlot + 1,
               s.def.isCapturable ? "capturable" : "non-capturable");
    }
    // Per-step wiring
    std::unordered_map<std::string, int> opHistogram;
    for (int s = 0; s < numSlots_; s++) {
      auto& sl = slots_[s];
      opHistogram[sl.ident.opName]++;
      // Build input description
      std::string inputsStr;
      for (int i = 0; i < sl.wiring.numInputs; i++) {
        if (i > 0) inputsStr += ", ";
        int srcIdx = sl.wiring.inputSourceIndices[i];
        if (srcIdx >= 0) {
          inputsStr += "slot#" + std::to_string(srcIdx);
        } else {
          int extIdx = -(srcIdx + 1);
          inputsStr += "ext#" + std::to_string(extIdx);
          if (extIdx < (int)externalInputNames_.size() && !externalInputNames_[extIdx].empty()) {
            inputsStr += ":\"" + externalInputNames_[extIdx] + "\"";
          }
          if (sl.wiring.inputSourceTypes != nullptr) {
            inputsStr += ":";
            inputsStr += sourceTypeName(sl.wiring.inputSourceTypes[i]);
          }
        }
      }
      std::string outputsStr;
      for (int i = 0; i < sl.wiring.numOutputs; i++) {
        if (i > 0) outputsStr += ",";
        outputsStr += std::to_string(sl.wiring.outputSlotIndices[i]);
      }
      DSP_DIAG(VERIFY, "STEP %4d: %-20s inputs:[%s] -> outputs:[%s]%s%s%s",
               s, sl.ident.opName.c_str(), inputsStr.c_str(), outputsStr.c_str(),
               sl.isIdentityOp() ? " [IDENTITY]" : "",
               sl.frozenConstantSlot() ? " [FROZEN]" : "",
               sl.fusedChain.isFusedChainTail ? " [FUSED_TAIL]" : "");
    }
    // Op histogram
    std::string histStr;
    std::vector<std::pair<std::string, int>> sorted(opHistogram.begin(), opHistogram.end());
    std::sort(sorted.begin(), sorted.end(),
              [](const auto& a, const auto& b) { return b.second < a.second; });
    for (auto& p : sorted) {
      if (!histStr.empty()) histStr += ", ";
      histStr += p.first + "=" + std::to_string(p.second);
    }
    DSP_DIAG(VERIFY, "Op histogram: %s", histStr.c_str());
    DSP_DIAG(VERIFY, "=== END PLAN STRUCTURE ===");
  }

  // Demote FROZEN slots to WARMUP before warmup execution.  After
  // invalidateForRebuild or restoreSlotStates, slots may be FROZEN with stale
  // cachedOutputShapes (e.g. FLOAT32 from a prior matmul dtype promotion).
  // If we execute with FROZEN state, the frozen context path fires and uses
  // stale cached shapes — potentially "correcting" a correct HALF output to
  // FLOAT32.  Demoting to WARMUP forces the normal execution path which runs
  // fresh calculateOutputShape with current input dtypes.
  //
  // NOTE: the old code PROMOTED view-capable slots to FROZEN here, which made
  // the problem worse — those slots would take the frozen context path with
  // stale dtypes.  Warmup must always use the normal path to re-derive shapes.
  if (!planLifecycle_.isSlotBySlot()) {
    int demoted = 0;
    for (int s = seg.def.startSlot; s <= seg.def.endSlot; s++) {
      auto& sl = slots_[s];
      if (sl.slotPhase.shapeCacheValid || sl.slotPhase.isSealed()) {
        sl.slotPhase.reset();  // PRIMARY: demote to BUILDING
        demoted++;
      }
    }
    if (demoted > 0) {
      DSP_DIAG(EXECUTE, "pre-warmup demotion: %d slots demoted to WARMUP for seg[%d-%d]",
               demoted, seg.def.startSlot, seg.def.endSlot);
    }
  }

  DSP_SEG_EVENT(seg, WARMUP_START, "phase=%s", planLifecycle_.displayName());
  auto warmupStatus = executeSegmentSlotBySlot(seg, externalArrays, numExt, stream);

  if (!planLifecycle_.isSlotBySlot() && warmupStatus == Status::OK && seg.exec.executionCount == 1
      && !Environment::getInstance().dspFreezeRecompile()) {
    if (!seg.def.shapeKeyState.neverCompiled()) {
      seg.exec.executionCount = 2;
      seg.exec.cachedShapeKey = seg.def.shapeKeyState.compiledShapeKey;
      DSP_SEG_EVENT(seg, SHAPE_KEY_MATCHED, "Post-freeze warmup: skipping recompile "
                    "(already compiled, bumped executionCount to 2)");
    } else {
      DSP_SEG_EVENT(seg, WARMUP_DONE, "Post-freeze warmup: NOT skipping compile "
                    "(never compiled, executionCount stays at 1)");
    }
  }
  if (warmupStatus == Status::OK) {
    int wuSegIdx = -1;
    for (int si = 0; si < static_cast<int>(segments_.size()); si++) {
      if (&segments_[si] == &seg) { wuSegIdx = si; break; }
    }
    DSP_TRACE_LIFECYCLE(trace_,
                        static_cast<int8_t>(wuSegIdx),
                        static_cast<uint8_t>(seg.exec.segPhase.toLegacyCode()),
                        static_cast<uint8_t>(SegmentLifecycleState::NEEDS_COMPILE),
                        static_cast<uint32_t>(executeCount_));
    SegmentLifecycle::markWarmupDone(seg.exec);
  }
  return warmupStatus;
}

// ═══════════════════════════════════════════════════════════════════════════
// segDispatchCompile — NEEDS_COMPILE state handler
// ═══════════════════════════════════════════════════════════════════════════
Status NativeDynamicShapePlan::segDispatchCompile(
    GraphSegment& seg, NDArray** externalArrays, int numExt, void* stream,
    LongType& segShapeKey, bool& invocationSatisfiedByWarmup) {
  invocationSatisfiedByWarmup = false;
  const GraphBackendRequest request = makeGraphBackendRequest();
  const auto& resolvedCandidates = getGraphBackendCandidates();
  GraphBackend* backend = seg.resolvedGraphBackend;
  if (backend == nullptr && !resolvedCandidates.empty()) {
    backend = resolvedCandidates.front();
  }
  if (backend == nullptr) {
    return Status::KERNEL_FAILURE;
  }
  const char* backendName = backend->name();
  bool shapeChangeWarmupCompleted = false;

  auto collectInputSignatures = [&]() {
    std::vector<std::string> signatures;
    std::unordered_set<int> segmentOutputs;
    std::unordered_set<int64_t> seenSources;
    for (int s = seg.def.startSlot; s <= seg.def.endSlot; s++) {
      const NativeSlot& slot = slots_[s];
      for (int i = 0; i < slot.wiring.numOutputs; i++) {
        segmentOutputs.insert(slot.wiring.outputSlotIndices[i]);
      }
    }
    for (int s = seg.def.startSlot; s <= seg.def.endSlot; s++) {
      const NativeSlot& slot = slots_[s];
      for (int i = 0; i < slot.wiring.numInputs; i++) {
        const int srcIdx = slot.wiring.inputSourceIndices[i];
        const bool isExternal = srcIdx < 0;
        const int resolvedIdx = isExternal ? -(srcIdx + 1) : srcIdx;
        if (!isExternal && segmentOutputs.find(resolvedIdx) != segmentOutputs.end()) {
          continue;
        }
        const int64_t sourceId = isExternal
            ? -(static_cast<int64_t>(resolvedIdx) + 1)
            : static_cast<int64_t>(resolvedIdx) + 1;
        if (!seenSources.insert(sourceId).second) {
          continue;
        }
        NDArray* arr = nullptr;
        std::string label;
        if (isExternal) {
          label = "ext[" + std::to_string(resolvedIdx) + "]";
          if (resolvedIdx >= 0 && resolvedIdx < static_cast<int>(externalInputNames_.size()) &&
              !externalInputNames_[resolvedIdx].empty()) {
            label += ":" + externalInputNames_[resolvedIdx];
          }
          if (resolvedIdx >= 0 && resolvedIdx < numExt && externalArrays != nullptr) {
            arr = externalArrays[resolvedIdx];
          }
        } else {
          label = "slot[" + std::to_string(resolvedIdx) + "]";
          if (resolvedIdx >= 0 && resolvedIdx < totalOutputSlots_ && outputSlots_ != nullptr) {
            arr = outputSlots_[resolvedIdx];
          }
        }
        std::string signature = label + "=";
        if (arr == nullptr || !arr->hasValidShapeInfo()) {
          signature += "<null-or-invalid>";
        } else {
          signature += "[";
          const LongType* shapeInfo = arr->shapeInfo();
          const int rank = shape::rank(shapeInfo);
          for (int d = 0; d < rank; d++) {
            if (d > 0) signature += ",";
            signature += std::to_string(static_cast<long long>(shapeInfo[d + 1]));
          }
          signature += "]";
          signature += ";dtype=" + std::to_string(static_cast<int>(arr->dataType()));
          signature += ";len=" + std::to_string(static_cast<long long>(arr->lengthOf()));
        }
        signatures.emplace_back(std::move(signature));
      }
    }
    return signatures;
  };

  seg.def.shapeKeyState.recordComputed(segShapeKey);
  bool needsCompile = seg.exec.segPhase.needsCompile() ||
                      seg.def.shapeKeyState.hasDrifted();
  if (seg.resolvedGraphBackend == nullptr) {
    needsCompile = true;
  }

  const bool isRecompileDueToShapeChange =
      seg.def.shapeKeyState.hasDrifted();

  // A compiled artifact may not be replaced during REPLAYING unless a freshly
  // computed boundary key proves that the current artifact is invalid. That
  // transition uses the normal segment invalidation lifecycle below.
  if (needsCompile && planLifecycle_.isReplaying() && !isRecompileDueToShapeChange) {
    DSP_DIAG(COMPILE,
             "ERROR: compilation triggered during REPLAYING phase for seg[%d-%d] "
             "(executionCount=%d, shapeKey compiled=%lld current=%lld, phase=%s). "
             "Compilation must only happen during warmup/capture phases.",
             seg.def.startSlot, seg.def.endSlot, seg.exec.executionCount,
             (long long)seg.def.shapeKeyState.compiledShapeKey, (long long)segShapeKey,
             planLifecycle_.displayName());
    REQUIRE_TRUE(false, 0,
                 "DSP phase contract violation: compilation triggered during REPLAYING phase "
                 "for seg[%d-%d] (executionCount=%d). Fix the phase management bug.",
                 seg.def.startSlot, seg.def.endSlot, seg.exec.executionCount);
    demotePlanPhase(PlanPhase::SHAPES_FROZEN,
                    "compilation triggered during REPLAYING phase");
  }

  if (needsCompile) {
    if (isRecompileDueToShapeChange) {
      if (DSP_DIAG_ENABLED(LIFECYCLE)) {
        const auto currentSignatures = collectInputSignatures();
        const auto& compiledSignatures = seg.def.shapeKeyState.compiledInputSignatures;
        DSP_DIAG(LIFECYCLE,
                 "SHAPE_KEY_DRIFT_INPUTS: seg[%d-%d] compiledKey=%lld currentKey=%lld "
                 "compiledInputs=%zu currentInputs=%zu",
                 seg.def.startSlot, seg.def.endSlot,
                 (long long)seg.def.shapeKeyState.compiledShapeKey, (long long)segShapeKey,
                 compiledSignatures.size(), currentSignatures.size());
        const size_t common = std::min(compiledSignatures.size(), currentSignatures.size());
        for (size_t i = 0; i < common; i++) {
          if (compiledSignatures[i] != currentSignatures[i]) {
            DSP_DIAG(LIFECYCLE, "SHAPE_KEY_DRIFT_INPUT[%zu]: compiled={%s} current={%s}",
                     i, compiledSignatures[i].c_str(), currentSignatures[i].c_str());
          }
        }
        for (size_t i = common; i < compiledSignatures.size(); i++) {
          DSP_DIAG(LIFECYCLE, "SHAPE_KEY_DRIFT_INPUT[%zu]: compiled={%s} current={<missing>}",
                   i, compiledSignatures[i].c_str());
        }
        for (size_t i = common; i < currentSignatures.size(); i++) {
          DSP_DIAG(LIFECYCLE, "SHAPE_KEY_DRIFT_INPUT[%zu]: compiled={<missing>} current={%s}",
                   i, currentSignatures[i].c_str());
        }
      }
      char reasonBuf[128];
      std::snprintf(reasonBuf, sizeof(reasonBuf),
                    "shape-change recompile (shapeKey %lld->%lld, executionCount=%d)",
                    (long long)seg.def.shapeKeyState.compiledShapeKey, (long long)segShapeKey,
                    seg.exec.executionCount);
      recordMidExecutionCompile(seg.def.startSlot, seg.def.endSlot, reasonBuf);
      DSP_SEG_EVENT(seg, RECOMPILE_TRIGGERED,
                    "shape change detected. Running slot-by-slot warmup to "
                    "refresh outputSlots_ before recompilation.");
      // Invalidate only this segment. Resetting the plan-wide execute counter
      // would destructively re-warm unrelated captured segments.
      SegmentLifecycle::invalidateSegmentCaptures(this, seg, "shape_change");
      platformResetGapCaches();
      platformResetBatchD2D();
      Status warmupStatus;
      {
        ShapeChangeWarmupGuard warmupGuard(*this, seg.def.startSlot, seg.def.endSlot);
        warmupStatus = executeSegmentSlotBySlot(seg, externalArrays, numExt, stream);
      }
      if (warmupStatus != Status::OK) {
        DSP_DIAG(COMPILE, "segDispatchCompile: shape-change warmup FAILED for seg[%d-%d] status=%d",
                 seg.def.startSlot, seg.def.endSlot, static_cast<int>(warmupStatus));
        return warmupStatus;
      }
      // Transition NEEDS_WARMUP -> NEEDS_COMPILE after successful warmup.
      // invalidateSegmentCaptures set state to NEEDS_WARMUP; the slot-by-slot
      // execution above completed the warmup, so advance the state machine
      // before calling markCompiled (which asserts NEEDS_COMPILE).
      SegmentLifecycle::markWarmupDone(seg.exec);
      shapeChangeWarmupCompleted = true;
      segShapeKey = computeSegmentShapeKey(seg, externalArrays, numExt);
      DSP_DIAG(COMPILE, "segDispatchCompile: shape-change warmup OK for seg[%d-%d], "
                        "recomputed shapeKey=%lld", seg.def.startSlot, seg.def.endSlot, segShapeKey);
    }

    DSP_SEG_EVENT(seg, COMPILE_START,
                  "resolver-cascade candidates=%d preferred=%s",
                  static_cast<int>(resolvedCandidates.size()),
                  seg.resolvedGraphBackend != nullptr
                      ? seg.resolvedGraphBackend->name()
                      : "<none>");
    const auto lowering = GraphBackendResolver::lowerSegment(
        request, resolvedCandidates, seg.resolvedGraphBackend, seg, slots_,
        seg.def.startSlot, seg.def.endSlot, externalArrays, numExt,
        outputSlots_, totalOutputSlots_, segShapeKey, numSlots_,
        requestedOutputSlotIndices_, numRequestedOutputs_);

    std::string cascadeFailures;
    for (const auto& attempt : lowering.attempts) {
      lastCompilationAudit_ = attempt.audit;
      if (attempt.succeeded) {
        continue;
      }
      std::string failedOps;
      for (const auto& entry : attempt.audit) {
        if (!entry.wasCompiled && !entry.isNativeHandled) {
          if (!failedOps.empty()) failedOps += ", ";
          failedOps += "slot " + std::to_string(entry.slotIndex) + " (" +
                       entry.opName + "): " + entry.reason;
        }
      }
      if (failedOps.empty()) {
        for (int s = seg.def.startSlot; s <= seg.def.endSlot; s++) {
          if (!failedOps.empty()) failedOps += ", ";
          failedOps += slots_[s].ident.opName;
        }
        failedOps =
            "segment ops: " + failedOps + " (no per-op audit available)";
      }
      if (!cascadeFailures.empty()) cascadeFailures += " | ";
      cascadeFailures +=
          std::string(attempt.backend->name()) + ": " + failedOps;
      DSP_SEG_EVENT(seg, COMPILE_FAILED, "backend=%s failedOps=[%s]",
                    attempt.backend->name(), failedOps.c_str());
    }

    if (!lowering.succeeded()) {
      lastCompileFailureDetail_ = cascadeFailures.empty()
          ? "no resolver candidate admitted the segment"
          : cascadeFailures;
      return Status::KERNEL_FAILURE;
    }
    backend = lowering.backend;
    seg.setResolvedGraphBackend(backend, request);
    backendName = backend->name();
    lastCompilationAudit_ = lowering.attempts.back().audit;
    DSP_SEG_EVENT(seg, COMPILE_DONE, "backend=%s", backendName);
    {
      int lcSegIdx = -1;
      for (int si = 0; si < static_cast<int>(segments_.size()); si++) {
        if (&segments_[si] == &seg) { lcSegIdx = si; break; }
      }
      DSP_TRACE_LIFECYCLE(trace_,
                          static_cast<int8_t>(lcSegIdx),
                          static_cast<uint8_t>(seg.exec.segPhase.toLegacyCode()),
                          static_cast<uint8_t>(SegmentLifecycleState::CAPTURE_PENDING),
                          static_cast<uint32_t>(executeCount_));
    }
  }

  // Validate compilation coverage on every compilation (not just the first).
  // Post-freeze recompiles at execCount=2+ can have different failure modes
  // (e.g., TritonIRBuilder missing SSA values for frozen slots) that must be
  // detected here rather than producing KERNEL_FAILURE during execution.
  if (needsCompile) {
    auto audit = backend->getLastCompilationAudit();
    lastCompilationAudit_ = audit;
    int compiledCount = 0;
    int nativeHandledCount = 0;
    int failedCount = 0;
    for (const auto& entry : audit) {
      if (entry.wasCompiled) {
        compiledCount++;
      } else if (entry.isNativeHandled) {
        nativeHandledCount++;
        DSP_DIAG_SLOT(COMPILE, entry.slotIndex, "%s VALIDATION: slot %d (%s) native-handled: %s",
                      backendName, entry.slotIndex, entry.opName.c_str(), entry.reason.c_str());
      } else {
        failedCount++;
        DSP_DIAG_SLOT(COMPILE, entry.slotIndex, "%s VALIDATION: slot %d (%s) was NOT compiled: %s",
                      backendName, entry.slotIndex, entry.opName.c_str(), entry.reason.c_str());
      }
    }
    if (compiledCount == 0 && nativeHandledCount == 0 && failedCount > 0) {
      DSP_DIAG(COMPILE, "%s COMPILE ERROR: segment [%d-%d] has zero compiled ops "
                        "(failed=%d). Compilation failures are errors, not fallbacks.",
               backendName, seg.def.startSlot, seg.def.endSlot, failedCount);
      DSP_TRACE_ERROR(trace_, -1, seg.def.startSlot,
                      static_cast<uint32_t>(executeCount_),
                      static_cast<uint64_t>(Status::KERNEL_FAILURE));
      SegmentLifecycle::markFailed(seg.exec, "zero_compiled_ops", seg.def.startSlot, seg.def.endSlot);
      return Status::KERNEL_FAILURE;
    }
    if (compiledCount == 0 && failedCount == 0) {
      DSP_DIAG(COMPILE, "%s: segment [%d-%d] has only native ordered sections (no Triton kernels needed, "
                        "nativeHandled=%d). Segment remains eligible for CUDA graph capture.",
               backendName, seg.def.startSlot, seg.def.endSlot, nativeHandledCount);
    }
    if (failedCount > 0) {
      DSP_DIAG(COMPILE, "%s COMPILE ERROR: segment [%d-%d] partial compile FAILED "
                        "(compiled=%d nativeHandled=%d failed=%d). Compilation failures are errors, not fallbacks.",
               backendName, seg.def.startSlot, seg.def.endSlot, compiledCount, nativeHandledCount, failedCount);
      DSP_TRACE_ERROR(trace_, -1, seg.def.startSlot,
                      static_cast<uint32_t>(executeCount_),
                      static_cast<uint64_t>(Status::KERNEL_FAILURE));
      SegmentLifecycle::markFailed(seg.exec, "partial_compile_failure", seg.def.startSlot, seg.def.endSlot);
      return Status::KERNEL_FAILURE;
    }
    if (nativeHandledCount > 0) {
      DSP_DIAG(COMPILE, "%s: segment [%d-%d] mixed compile OK (compiled=%d nativeHandled=%d). "
                        "Native-handled ops will execute via slot-by-slot within the segment.",
               backendName, seg.def.startSlot, seg.def.endSlot, compiledCount, nativeHandledCount);
    }
  }

  // Publish lifecycle and cache identity only after lowering coverage has been
  // validated. A rejected partial compile must never become replay-visible.
  if (needsCompile) {
    if (seg.exec.segPhase.needsCompile()) {
      SegmentLifecycle::markCompiled(seg.exec, backendName, segShapeKey);
    } else if (!seg.exec.segPhase.isSealed()) {
      DSP_THROW_SEG(COMPILE, seg.def.startSlot,
                    "segDispatchCompile: successful compile for seg[%d-%d] ended in invalid phase %s",
                    seg.def.startSlot, seg.def.endSlot,
                    seg.exec.segPhase.displayName());
    }
    seg.def.shapeKeyState.markCompiled(segShapeKey);
    if (DSP_DIAG_ENABLED(LIFECYCLE)) {
      seg.def.shapeKeyState.compiledInputSignatures = collectInputSignatures();
    } else {
      seg.def.shapeKeyState.compiledInputSignatures.clear();
    }
    DSP_SEG_EVENT(seg, SHAPE_KEY_STORED, "validated compilation complete");
    if (shapeChangeWarmupCompleted) {
      // The bounded functional warmup produced this invocation's outputs.
      // Publish capture readiness only after the replacement artifact and its
      // coverage audit are valid. Capture/direct execution must wait until the
      // next call or stateful/in-place operations would execute twice.
      seg.exec.markShapeChangeWarmupCaptureReady();
      invocationSatisfiedByWarmup = true;
    }
  }

  return Status::OK;
}


}  // namespace graph
}  // namespace sd

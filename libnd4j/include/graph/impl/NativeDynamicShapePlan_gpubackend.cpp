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
//   - getGpuGraphBackend(): selects the best available GPU compiler backend
//   - segDispatchWarmup(): slot-by-slot warmup before compilation
//   - segDispatchCompile(): backend compilation and audit
//   - cleanupSegmentForRebuild(): segment cleanup with diagnostics
//   - hasCompositeHandles(): composite replay readiness check
//
// CUDA-specific execution (sync, capture, replay, eviction) is in
// NativeDynamicShapePlan_gpubackend.cu, compiled only by NVCC.

#include <graph/NativeDynamicShapePlan.h>
#include <graph/ModeContract.h>
#include <graph/DspDiagnostics.h>
#include <graph/DspSegmentLifecycle.h>
#include <system/op_boilerplate.h>
#include <system/Environment.h>
#include <config.h>

#if HAVE_TRITON && defined(SD_CUDA)
#include <graph/gpu/TritonGraphBackend.h>
#endif
#ifdef SD_CUDA
#include <graph/gpu/NvrtcGraphBackend.h>
#include <graph/gpu/PtxGraphBackend.h>
#endif
#ifdef SD_TPU
#include <graph/tpu/TpuGraphBackend.h>
#endif
#ifdef HAVE_HEXAGON_MLIR
#include <graph/hexagon/HexagonGraphBackend.h>
#endif

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
// hasCompositeHandles — check if merged captures are ready for replay
// ═══════════════════════════════════════════════════════════════════════════════
bool NativeDynamicShapePlan::hasCompositeHandles(const GraphSegment& seg) const {
  auto& sched = seg.exec.compositeReplaySchedule;
  // Check merged replay handles — at least one merged group must be ready
  for (auto& h : sched.mergedReplayHandles) {
    if (h != nullptr && h->isReady()) return true;
  }
  // Fallback: check individual composite handles (backward compat)
  for (auto& u : sched.units) {
    if (u.kind == REPLAY_UNIT_TRITON_ISLAND && u.mergedGroupId < 0) {
      int idx = u.islandIndex;
      if (idx >= 0 && idx < static_cast<int>(sched.compositeReplayHandles.size()) &&
          sched.compositeReplayHandles[idx] != nullptr &&
          sched.compositeReplayHandles[idx]->isReady()) {
        return true;
      }
    }
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
#if HAVE_TRITON && defined(SD_CUDA)
  TritonGraphBackend::getInstance().clearFailedSegmentCache();
#endif
}

GraphBackend* NativeDynamicShapePlan::getGpuGraphBackend() {
  if (gpuGraphBackendChecked_) return gpuGraphBackend_;
  gpuGraphBackendChecked_ = true;

  // Modes that don't need a JIT compilation backend (pure replay or slot-by-slot).
  if (!ModeContract::forMode(graphExecutionMode_).needsJitBackend) {
    gpuGraphBackend_ = nullptr;
    return nullptr;
  }

#if HAVE_TRITON && defined(SD_CUDA)
  if (graphExecutionMode_ == GraphExecutionMode::GEM_TRITON ||
     graphExecutionMode_ == GraphExecutionMode::GEM_AUTO) {
   auto& triton = TritonGraphBackend::getInstance();
   if (triton.isAvailable()) {
     gpuGraphBackend_ = &triton;
     DSP_DIAG(BACKEND, "using Triton GPU compiler backend");
     return gpuGraphBackend_;
   }
   if (graphExecutionMode_ == GraphExecutionMode::GEM_TRITON) {
     DSP_DIAG(BACKEND, "Triton backend requested but not available");
     gpuGraphBackend_ = nullptr;
     return nullptr;
   }
   DSP_DIAG(BACKEND, "Triton unavailable in AUTO mode, trying NVRTC/PTX backends");
 }
#else
  if (graphExecutionMode_ == GraphExecutionMode::GEM_TRITON) {
    DSP_DIAG(BACKEND, "Triton backend requested but not compiled (HAVE_TRITON=0)");
    gpuGraphBackend_ = nullptr;
    return nullptr;
  }
  if (graphExecutionMode_ == GraphExecutionMode::GEM_AUTO) {
    DSP_DIAG(BACKEND, "Triton not compiled (HAVE_TRITON=0); AUTO mode will try NVRTC/PTX/CUDA graphs");
  }
#endif

#ifdef SD_CUDA
  if (graphExecutionMode_ == GraphExecutionMode::GEM_NVRTC_JIT ||
      graphExecutionMode_ == GraphExecutionMode::GEM_AUTO) {
    auto& nvrtc = NvrtcGraphBackend::getInstance();
    if (nvrtc.isAvailable()) {
      gpuGraphBackend_ = &nvrtc;
      DSP_DIAG(BACKEND, "using NVRTC GPU compiler backend");
      return gpuGraphBackend_;
    }
    if (graphExecutionMode_ == GraphExecutionMode::GEM_NVRTC_JIT) {
      DSP_DIAG(BACKEND, "NVRTC backend requested but not available");
      gpuGraphBackend_ = nullptr;
      return nullptr;
    }
  }

  if (graphExecutionMode_ == GraphExecutionMode::GEM_PTX_JIT ||
      graphExecutionMode_ == GraphExecutionMode::GEM_AUTO) {
    auto& ptx = PtxGraphBackend::getInstance();
    if (ptx.isAvailable()) {
      gpuGraphBackend_ = &ptx;
      DSP_DIAG(BACKEND, "using PTX template GPU compiler backend");
      return gpuGraphBackend_;
    }
    if (graphExecutionMode_ == GraphExecutionMode::GEM_PTX_JIT) {
      DSP_DIAG(BACKEND, "PTX backend requested but not available");
      gpuGraphBackend_ = nullptr;
      return nullptr;
    }
  }
#endif

#ifdef SD_TPU
  if (graphExecutionMode_ == GraphExecutionMode::GEM_TPU ||
     graphExecutionMode_ == GraphExecutionMode::GEM_AUTO) {
   auto& tpu = TpuGraphBackend::getInstance();
   if (tpu.isAvailable()) {
     gpuGraphBackend_ = &tpu;
     DSP_DIAG(BACKEND, "using TPU HLO compiler backend");
     return gpuGraphBackend_;
   }
   if (graphExecutionMode_ == GraphExecutionMode::GEM_TPU) {
     DSP_DIAG(BACKEND, "TPU backend requested but not available");
     gpuGraphBackend_ = nullptr;
     return nullptr;
   }
 }
#else
  if (graphExecutionMode_ == GraphExecutionMode::GEM_TPU) {
    DSP_DIAG(BACKEND, "TPU backend requested but not compiled (SD_TPU=0)");
    gpuGraphBackend_ = nullptr;
    return nullptr;
  }
#endif

#ifdef HAVE_HEXAGON_MLIR
  if (graphExecutionMode_ == GraphExecutionMode::GEM_HEXAGON ||
     graphExecutionMode_ == GraphExecutionMode::GEM_AUTO) {
   auto& hexagon = HexagonGraphBackend::getInstance();
   if (hexagon.isAvailable()) {
     gpuGraphBackend_ = &hexagon;
     DSP_DIAG(BACKEND, "using Hexagon MLIR NPU compiler backend");
     return gpuGraphBackend_;
   }
   if (graphExecutionMode_ == GraphExecutionMode::GEM_HEXAGON) {
     DSP_DIAG(BACKEND, "Hexagon backend requested but not available");
     gpuGraphBackend_ = nullptr;
     return nullptr;
   }
 }
#else
  if (graphExecutionMode_ == GraphExecutionMode::GEM_HEXAGON) {
    DSP_DIAG(BACKEND, "Hexagon backend requested but not compiled (HAVE_HEXAGON_MLIR=0)");
    gpuGraphBackend_ = nullptr;
    return nullptr;
  }
#endif

  gpuGraphBackend_ = nullptr;
  return nullptr;
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
    LongType& segShapeKey) {
  auto* backend = getGpuGraphBackend();
  if (backend == nullptr) {
    return Status::KERNEL_FAILURE;
  }
  const char* backendName = backend->name();

  seg.def.shapeKeyState.recordComputed(segShapeKey);
  bool needsCompile = seg.exec.segPhase.needsCompile() ||
                      seg.def.shapeKeyState.hasDrifted();

  // Phase guard: compilation must not happen during REPLAYING
  if (needsCompile && planLifecycle_.isReplaying()) {
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
    bool isRecompileDueToShapeChange = (seg.exec.executionCount > 1) && seg.def.shapeKeyState.hasDrifted();
    if (isRecompileDueToShapeChange) {
      char reasonBuf[128];
      std::snprintf(reasonBuf, sizeof(reasonBuf),
                    "shape-change recompile (shapeKey %lld->%lld, executionCount=%d)",
                    (long long)seg.def.shapeKeyState.compiledShapeKey, (long long)segShapeKey,
                    seg.exec.executionCount);
      recordMidExecutionCompile(seg.def.startSlot, seg.def.endSlot, reasonBuf);
      DSP_SEG_EVENT(seg, RECOMPILE_TRIGGERED,
                    "shape change detected. Running slot-by-slot warmup to "
                    "refresh outputSlots_ before recompilation.");
      // Invalidate cached graph
      SegmentLifecycle::invalidateForRebuild(this, seg, "shape_change");
#ifdef SD_CUDA
      batchD2DCount_ = 0;
#endif
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
      // invalidateForRebuild set state to NEEDS_WARMUP; the slot-by-slot
      // execution above completed the warmup, so advance the state machine
      // before calling markCompiled (which asserts NEEDS_COMPILE).
      SegmentLifecycle::markWarmupDone(seg.exec);
      segShapeKey = computeSegmentShapeKey(seg, externalArrays, numExt);
      DSP_DIAG(COMPILE, "segDispatchCompile: shape-change warmup OK for seg[%d-%d], "
                        "recomputed shapeKey=%lld", seg.def.startSlot, seg.def.endSlot, segShapeKey);
    }

    DSP_SEG_EVENT(seg, COMPILE_START, "backend=%s", backendName);
    if (!backend->compileSegment(seg, slots_, externalArrays, numExt,
                                 outputSlots_, totalOutputSlots_, segShapeKey,
                                 numSlots_)) {
      // Fetch audit even on failure — it contains per-op detail about what failed
      auto failAudit = backend->getLastCompilationAudit();
      lastCompilationAudit_ = failAudit;
      std::string failedOps;
      int failCount = 0;
      for (const auto& entry : failAudit) {
        if (!entry.wasCompiled && !entry.isNativeHandled) {
          if (!failedOps.empty()) failedOps += ", ";
          failedOps += "slot " + std::to_string(entry.slotIndex) + " (" + entry.opName + "): " + entry.reason;
          failCount++;
        }
      }
      if (failedOps.empty()) {
        // No audit entries — list the ops in the segment for context
        for (int s = seg.def.startSlot; s <= seg.def.endSlot; s++) {
          if (!failedOps.empty()) failedOps += ", ";
          failedOps += slots_[s].ident.opName;
        }
        failedOps = "segment ops: " + failedOps + " (no per-op audit available)";
      }
      DSP_SEG_EVENT(seg, COMPILE_FAILED, "backend=%s failedOps=[%s]", backendName, failedOps.c_str());
      // Store the failure detail so DSP_THROW_SEG can include it
      lastCompileFailureDetail_ = failedOps;
      return Status::KERNEL_FAILURE;
    }
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
    // Guard: only transition lifecycle if the segment is in the expected
    // NEEDS_COMPILE state. Plan cache can return a plan whose segment was
    // previously captured+sealed; re-compiling (e.g. after plan-swap refreeze)
    // must not fire a lifecycle transition on an already-sealed segment.
    if (seg.exec.segPhase.needsCompile()) {
      SegmentLifecycle::markCompiled(seg.exec, backendName, segShapeKey);
    } else if (seg.exec.segPhase.isSealed()) {
      // Already sealed from a prior run on this plan — skip lifecycle transition
      // but still update shape key state so the Triton cache lookup matches.
      DSP_DIAG(COMPILE, "segDispatchCompile: seg[%d-%d] already SEALED — "
               "skipping markCompiled lifecycle transition (plan cache re-hit)",
               seg.def.startSlot, seg.def.endSlot);
    }
    // Store the compiled shape key so the Triton cache lookup key matches.
    // This MUST only happen when compilation actually occurred — the caller
    // previously did this unconditionally, which overwrote compiledShapeKey
    // on non-compilation execs with a new shape key (e.g. KV cache grew),
    // causing KERNEL_FAILURE on the Triton cache lookup.
    seg.def.shapeKeyState.markCompiled(segShapeKey);
    DSP_SEG_EVENT(seg, SHAPE_KEY_STORED, "compilation complete");
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

  return Status::OK;
}


}  // namespace graph
}  // namespace sd

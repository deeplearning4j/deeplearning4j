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
  bool isMonolithicReplaying = (seg.exec.lifecycleState == SegmentLifecycleState::REPLAYING &&
                                seg.exec.replayHandle != nullptr &&
                                seg.exec.replayHandle->isReady());
  bool isCompositeReplaying  = (seg.exec.lifecycleState == SegmentLifecycleState::REPLAYING &&
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

  // If a specific backend is forced via setGraphExecutionMode(), use only that one.
  if (graphExecutionMode_ == GraphExecutionMode::GEM_SLOT_BY_SLOT ||
      graphExecutionMode_ == GraphExecutionMode::GEM_CUDA_GRAPHS ||
      graphExecutionMode_ == GraphExecutionMode::GEM_HIP_GRAPHS ||
      graphExecutionMode_ == GraphExecutionMode::GEM_LEVELZERO ||
      graphExecutionMode_ == GraphExecutionMode::GEM_VULKAN ||
      graphExecutionMode_ == GraphExecutionMode::GEM_METAL) {
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
               sl.flags.isIdentityOp ? " [IDENTITY]" : "",
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

  // Pre-warmup promotion: view-capable slots enter FROZEN state BEFORE
  // warmup execution when shapes are already frozen.
  if (shapesFrozen_) {
    int promoted = 0;
    for (int s = seg.def.startSlot; s <= seg.def.endSlot; s++) {
      auto& sl = slots_[s];
      if (!sl.flags.isViewCapableOp || sl.state_ >= NativeSlot::SlotState::FROZEN) continue;
      sl.state_ = NativeSlot::SlotState::FROZEN;
      promoted++;
    }
    if (promoted > 0) {
      DSP_DIAG(EXECUTE, "pre-warmup view promotion: %d view-capable slots promoted to FROZEN for seg[%d-%d]",
               promoted, seg.def.startSlot, seg.def.endSlot);
    }
  }

  DSP_SEG_EVENT(seg, WARMUP_START, "shapesFrozen=%d", shapesFrozen_ ? 1 : 0);
  auto warmupStatus = executeSegmentSlotBySlot(seg, externalArrays, numExt, stream);

  if (shapesFrozen_ && warmupStatus == Status::OK && seg.exec.executionCount == 1
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
                        static_cast<uint8_t>(seg.exec.lifecycleState),
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
  bool needsCompile = (seg.exec.lifecycleState == SegmentLifecycleState::NEEDS_COMPILE) ||
                      seg.def.shapeKeyState.hasDrifted();

  // Phase guard: compilation must not happen during REPLAYING
  if (needsCompile && planPhase_ >= PlanPhase::REPLAYING) {
    DSP_DIAG(COMPILE,
             "ERROR: compilation triggered during REPLAYING phase for seg[%d-%d] "
             "(executionCount=%d, shapeKey compiled=%lld current=%lld, planPhase=%d). "
             "Compilation must only happen during warmup/capture phases.",
             seg.def.startSlot, seg.def.endSlot, seg.exec.executionCount,
             (long long)seg.def.shapeKeyState.compiledShapeKey, (long long)segShapeKey,
             static_cast<int>(planPhase_));
    REQUIRE_TRUE(false, 0,
                 "DSP phase contract violation: compilation triggered during REPLAYING phase "
                 "for seg[%d-%d] (executionCount=%d). Fix the phase management bug.",
                 seg.def.startSlot, seg.def.endSlot, seg.exec.executionCount);
    demotePlanPhase(PlanPhase::POINTERS_STABLE,
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
      batchD2DCount_ = 0;
      auto warmupStatus = executeSegmentSlotBySlot(seg, externalArrays, numExt, stream);
      if (warmupStatus != Status::OK) {
        DSP_DIAG(COMPILE, "segDispatchCompile: shape-change warmup FAILED for seg[%d-%d] status=%d",
                 seg.def.startSlot, seg.def.endSlot, static_cast<int>(warmupStatus));
        return warmupStatus;
      }
      segShapeKey = computeSegmentShapeKey(seg, externalArrays, numExt);
      DSP_DIAG(COMPILE, "segDispatchCompile: shape-change warmup OK for seg[%d-%d], "
                        "recomputed shapeKey=%lld", seg.def.startSlot, seg.def.endSlot, segShapeKey);
    }

    DSP_SEG_EVENT(seg, COMPILE_START, "backend=%s", backendName);
    if (!backend->compileSegment(seg, slots_, externalArrays, numExt,
                                 outputSlots_, totalOutputSlots_, segShapeKey,
                                 numSlots_)) {
      DSP_SEG_EVENT(seg, COMPILE_FAILED, "backend=%s", backendName);
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
                          static_cast<uint8_t>(seg.exec.lifecycleState),
                          static_cast<uint8_t>(SegmentLifecycleState::CAPTURE_PENDING),
                          static_cast<uint32_t>(executeCount_));
    }
    SegmentLifecycle::markCompiled(seg.exec, backendName, segShapeKey);
  }

  // On first compilation, validate coverage
  if (seg.exec.executionCount == 1) {
    auto audit = backend->getLastCompilationAudit();
    lastCompilationAudit_ = audit;
    int compiledCount = 0;
    int failedCount = 0;
    for (const auto& entry : audit) {
      if (entry.wasCompiled) {
        compiledCount++;
      } else {
        failedCount++;
        DSP_DIAG_SLOT(COMPILE, entry.slotIndex, "%s VALIDATION: slot %d (%s) was NOT compiled: %s",
                      backendName, entry.slotIndex, entry.opName.c_str(), entry.reason.c_str());
      }
    }
    if (compiledCount == 0 && failedCount > 0) {
      DSP_DIAG(COMPILE, "%s COMPILE ERROR: segment [%d-%d] has zero compiled ops "
                        "(failed=%d). Compilation failures are errors, not fallbacks.",
               backendName, seg.def.startSlot, seg.def.endSlot, failedCount);
      DSP_TRACE_ERROR(trace_, -1, seg.def.startSlot,
                      static_cast<uint32_t>(executeCount_),
                      static_cast<uint64_t>(Status::KERNEL_FAILURE));
      SegmentLifecycle::markFailed(seg.exec, "zero_compiled_ops");
      return Status::KERNEL_FAILURE;
    }
    if (compiledCount == 0 && failedCount == 0) {
      DSP_DIAG(COMPILE, "%s: segment [%d-%d] has only native ordered sections (no Triton kernels needed). "
                        "Segment remains eligible for CUDA graph capture.",
               backendName, seg.def.startSlot, seg.def.endSlot);
    }
    if (failedCount > 0) {
      DSP_DIAG(COMPILE, "%s COMPILE ERROR: segment [%d-%d] partial compile FAILED "
                        "(compiled=%d failed=%d). Compilation failures are errors, not fallbacks.",
               backendName, seg.def.startSlot, seg.def.endSlot, compiledCount, failedCount);
      DSP_TRACE_ERROR(trace_, -1, seg.def.startSlot,
                      static_cast<uint32_t>(executeCount_),
                      static_cast<uint64_t>(Status::KERNEL_FAILURE));
      SegmentLifecycle::markFailed(seg.exec, "partial_compile_failure");
      return Status::KERNEL_FAILURE;
    }
  }

  return Status::OK;
}


}  // namespace graph
}  // namespace sd

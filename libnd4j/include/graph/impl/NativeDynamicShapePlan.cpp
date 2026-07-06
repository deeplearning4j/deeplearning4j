/* ******************************************************************************
 *
 * Copyright (c) 2024-2026 Contributors
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

#include <graph/NativeDynamicShapePlan.h>
#include <graph/ModeContract.h>
#include <graph/DspSegmentLifecycle.h>
#include <graph/DspExecutionTrace.h>
#include <graph/PlanExecutionContext.h>
#include <graph/NativePlanCompiler.h>
#include <system/op_boilerplate.h>
#include <graph/DspStreamGuard.h>
#include <graph/DspAnalysisUtils.h>
#include <graph/DspPhaseUtils.h>
#include <sstream>
#include <graph/gpu/SymbolicShapeRanges.h>
#include <graph/DspDiagnostics.h>
#include <graph/LegacyOpTypeCodes.h>
#include <graph/gpu/DspCudaDispatch.h>
#include <graph/FusionPass.h>
#include <ops/declarable/helpers/fusedElementwiseChain.h>
#include <graph/GraphBackend.h>
#include <array/DataBuffer.h>
#include <helpers/ConstantShapeHelper.h>
#include <helpers/ConstantTadHelper.h>
#include <helpers/MmulHelper.h>
#include <helpers/helper_hash.h>
#include <ops/OpTraitTable.h>
#include <ops/declarable/OpRegistrator.h>
#include <ops/declarable/LegacyTransformSameOp.h>
#include <ops/declarable/LegacyTransformStrictOp.h>
#include <ops/declarable/LegacyTransformFloatOp.h>
#include <ops/declarable/LegacyTransformBoolOp.h>
#include <ops/declarable/LegacyScalarOp.h>
#include <ops/declarable/LegacyScalarBoolOp.h>
#include <ops/declarable/LegacyPairwiseTransformOp.h>
#include <ops/declarable/LegacyPairwiseTransformBoolOp.h>
#include <ops/declarable/LegacyReduceFloatOp.h>
#include <ops/declarable/LegacyReduceSameOp.h>
#include <ops/declarable/LegacyReduceBoolOp.h>
#include <ops/declarable/LegacyReduceLongOp.h>
#include <ops/declarable/LegacyReduce3Op.h>
#include <ops/declarable/LegacyStatsOp.h>
#include <ops/declarable/LegacyIndexReduceOp.h>
#include <ops/declarable/LegacyBroadcastOp.h>
#include <ops/declarable/LegacyBroadcastBoolOp.h>
#include <ops/declarable/LegacyRandomOp.h>
#include <ops/declarable/helpers/kv_scatter.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <future>
#include <memory>
#include <numeric>
#include <climits>
#include <cstring>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <system/Environment.h>


// Include CPU graph backends conditionally
#include <config.h>
#if HAVE_ONEDNN
#include <graph/cpu/OneDnnGraphBackend.h>
#endif
#if HAVE_ARMCOMPUTE
#include <graph/cpu/AclGraphBackend.h>
#endif
#if HAVE_MLIR
#include <graph/cpu/MlirCpuGraphBackend.h>
#if defined(__ANDROID__) || (defined(__linux__) && defined(__aarch64__))
#include <graph/cpu/ArmHybridGraphBackend.h>
#endif
#endif
#if HAVE_NNAPI
#include <graph/cpu/NnapiGraphBackend.h>
#endif
#if HAVE_MLX
#include <graph/cpu/MlxGraphBackend.h>
#endif
// GPU graph backends are included only in the files that use them
// (_gpubackend.cpp, platform dispatch files). This file is platform-neutral.

namespace sd {
namespace graph {

// ── Frozen-pin liveness registry ─────────────────────────────────────────────
// frozenProtectedRefBuffers_/frozenOutputRefBuffers_ hold RAW DataBuffer*, and
// addFrozenRef()/removeFrozenRef() WRITE an atomic counter inside the object.
// Nothing native owns those objects: Java teardown can delete them at any time
// (freeModelArrays before close), after which removeFrozenRef() is a write to
// freed memory — heap corruption that crashed teardown's local unordered_set
// lookups (hs_err_pid1286526 / hs_err_pid927393, SEGV in _M_find_before_node).
// This registry records every pinned buffer; ~DataBuffer erases itself, so the
// release paths can tell live pins from dead pointers WITHOUT dereferencing.
// The map counts pins per buffer (a buffer may be pinned by multiple plans).
namespace {
std::mutex g_frozenPinMtx;
std::unordered_map<DataBuffer*, int> g_frozenPinCounts;

void trackFrozenPin(DataBuffer* db) {
  std::lock_guard<std::mutex> lk(g_frozenPinMtx);
  g_frozenPinCounts[db]++;
}

// Returns true when the buffer object is still alive (pin found) — only then
// may the caller touch it. Dead/never-tracked pointers return false.
bool untrackFrozenPin(DataBuffer* db) {
  std::lock_guard<std::mutex> lk(g_frozenPinMtx);
  auto it = g_frozenPinCounts.find(db);
  if (it == g_frozenPinCounts.end()) return false;
  if (--it->second <= 0) g_frozenPinCounts.erase(it);
  return true;
}
}  // namespace

}  // namespace graph

// Called from DataBuffer::~DataBuffer() (fast-gated on isFrozenPlanRegistered)
// so a buffer destroyed while pinned drops out of the registry and teardown
// skips it instead of writing through the dangling pointer. Lives in sd:: (not
// sd::graph::) so the DataBuffer.cpp seam declaration stays one line.
SD_LIB_EXPORT void notifyFrozenPinTrackerOfDestruction(DataBuffer* db) {
  std::lock_guard<std::mutex> lk(sd::graph::g_frozenPinMtx);
  sd::graph::g_frozenPinCounts.erase(db);
}

namespace graph {

static void releasePlanFrozenRefsForTeardown(
    const char* owner,
    bool shouldRelease,
    std::vector<DataBuffer*>& frozenProtectedRefBuffers,
    std::vector<DataBuffer*>& frozenOutputRefBuffers) {
  if (!shouldRelease) return;

  int protectedRemoved = 0;
  int protectedDead = 0;
  for (auto* db : frozenProtectedRefBuffers) {
    if (db != nullptr) {
      if (untrackFrozenPin(db)) {
        db->removeFrozenRef();
        protectedRemoved++;
      } else {
        protectedDead++;  // destroyed externally while pinned — must not touch
      }
    }
  }
  frozenProtectedRefBuffers.clear();

  int outputRemoved = 0;
  int outputDead = 0;
  for (auto* db : frozenOutputRefBuffers) {
    if (db != nullptr) {
      if (untrackFrozenPin(db)) {
        db->removeFrozenRef();
        outputRemoved++;
      } else {
        outputDead++;
      }
    }
  }
  frozenOutputRefBuffers.clear();
  if (protectedDead > 0 || outputDead > 0) {
    DSP_DIAG(MEMORY,
             "%s: skipped frozen-ref release for %d protected + %d output buffers "
             "destroyed externally while pinned (Java freed model arrays before close)",
             owner, protectedDead, outputDead);
  }

  DSP_DIAG(MEMORY,
           "%s: removed tracked frozen refs before identity teardown — protectedRefs=%d outputSlotRefs=%d",
           owner, protectedRemoved, outputRemoved);
}

static bool hasTrackedPlanFrozenRefs(const std::vector<DataBuffer*>& frozenProtectedRefBuffers,
                                     const std::vector<DataBuffer*>& frozenOutputRefBuffers) {
  return !frozenProtectedRefBuffers.empty() || !frozenOutputRefBuffers.empty();
}

static void replacePlanFrozenRefsForCurrentState(
    const char* owner,
    const std::unordered_set<DataBuffer*>& protectedWeightBuffers,
    NDArray** outputSlots,
    int totalOutputSlots,
    std::vector<DataBuffer*>& frozenProtectedRefBuffers,
    std::vector<DataBuffer*>& frozenOutputRefBuffers) {
  releasePlanFrozenRefsForTeardown(
      owner, hasTrackedPlanFrozenRefs(frozenProtectedRefBuffers, frozenOutputRefBuffers),
      frozenProtectedRefBuffers, frozenOutputRefBuffers);

  int protectedAdded = 0;
  for (auto* db : protectedWeightBuffers) {
    if (db != nullptr) {
      db->addFrozenRef();
      trackFrozenPin(db);
      frozenProtectedRefBuffers.push_back(db);
      protectedAdded++;
    }
  }

  int outputAdded = 0;
  if (outputSlots != nullptr) {
    for (int i = 0; i < totalOutputSlots; i++) {
      if (outputSlots[i] != nullptr && outputSlots[i]->dataBuffer() != nullptr) {
        DataBuffer* db = outputSlots[i]->dataBuffer();
        db->addFrozenRef();
        trackFrozenPin(db);
        frozenOutputRefBuffers.push_back(db);
        outputAdded++;
      }
    }
  }

  DSP_DIAG(MEMORY,
           "%s: added tracked frozen refs for current state — protectedRefs=%d "
           "outputSlotRefs=%d totalOutputSlots=%d",
           owner, protectedAdded, outputAdded, totalOutputSlots);
}

// ── Per-device warmup serialization ──────────────────────────────────────
// During warmup and CUDA graph capture, legacy host-blocking CUDA API calls
// on the legacy stream poison any active
// capture on the same device (error 906 → cascade 901).
//
// The existing DeviceCaptureGuard (in _gpubackend.cu) only blocks during
// the actual capture sub-phase. But other threads can still execute
// warmup ops concurrently, making sync calls that poison the capture.
//
// This mutex serializes ALL plan execution on a device while ANY plan is
// in a non-steady-state (warmup or capture). Once a plan reaches REPLAYING
// (SEALED phase), it no longer acquires this mutex — replay is lock-free
// and uses only async APIs. CPU uses a single global mutex (index 0).
//
// Timeline with 8 concurrent threads:
//   Thread A: warmup → capture → REPLAYING (mutex held)
//   Thread B: (waits) → warmup → capture → REPLAYING (mutex held)
//   ...
//   After all reach REPLAYING: full parallel execution, no mutex.
static constexpr int kMaxDevices = 16;
static std::mutex g_warmupSerializationMtx[kMaxDevices];

static void scanAllSlotsForCorruption(
    NDArray** outputSlots, int totalOutputSlots,
    const char* checkpoint, int execCount) {
  for (int i = 0; i < totalOutputSlots; i++) {
    if (outputSlots[i] == nullptr) continue;
    auto* sib = outputSlots[i]->shapeInfoConstBuffer();
    if (sib == nullptr) continue;
    uintptr_t sibAddr = reinterpret_cast<uintptr_t>(sib);
    if (sibAddr % alignof(ConstantShapeBuffer) != 0) {
      DSP_DIAG(MEMORY,
               "CORRUPTION_SCAN_HIT: checkpoint=%s slot=%d arr=%p "
               "_shapeInfoBuffer=%p alignOffset=%zu execCount=%d",
               checkpoint, i, (void*)outputSlots[i], (void*)sib,
               static_cast<size_t>(sibAddr % alignof(ConstantShapeBuffer)),
               execCount);
      return;
    }
  }
}

// ─── Deferred slot-array deletion ───────────────────────────────────────────
// writeOutputSlot() replaces plan-owned NDArray pointers inline during slot
// execution.  Calling `delete old` immediately can corrupt heap metadata when
// the allocator tries to update its internal free-list while the plan is still
// iterating over adjacent allocations (the exact failure mode seen in the
// Workspace::allocateBytes SIGSEGV where `this` is garbage string data).
//
// The fix: push the old pointer into a thread-local vector and delete it only
// once execution of the current plan step is fully complete (just before
// platformEndExecution is called from execute()).  By that point no more slot
// iteration is in progress and the heap is in a consistent state.
static thread_local std::vector<NDArray*> tl_deferredSlotDeletes;

static void flushDeferredSlotDeletes() {
  if (tl_deferredSlotDeletes.empty()) return;
  // Swap into a local vector so that any re-entrant call during deletion
  // (e.g., a DataBuffer destructor that triggers another writeOutputSlot)
  // accumulates into a fresh tl_deferredSlotDeletes rather than invalidating
  // the iterator we are currently walking.
  std::vector<NDArray*> pending;
  pending.swap(tl_deferredSlotDeletes);
  for (NDArray* arr : pending) {
    delete arr;
  }
}

namespace {
std::string normalizeOpName(const std::string& opName) {
  std::string normalized = opName;
  std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return normalized;
}

bool segmentBlocksPlanPhase(const GraphSegment& seg) {
  return seg.def.isCapturable && !seg.exec.compilationFailed;
}

// Check if a segment has at least one ready composite replay handle.
// Composite replay handles are per-island CUDA graph handles used by Triton
// GPU segments. Unlike the monolithic seg.exec.replayHandle, these are stored
// in the ReplaySchedule and represent a captured composite (island+gap) replay.
// This is the standalone equivalent of NativeDynamicShapePlan::hasCompositeHandles().
bool segmentHasReadyCompositeHandles(const GraphSegment& seg) {
  auto& sched = seg.exec.compositeReplaySchedule;
  // Check merged replay handles first
  for (auto& h : sched.mergedReplayHandles) {
    if (h != nullptr && h->isReady()) return true;
  }
  // Fallback: check individual composite handles (unmerged islands)
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

}  // anonymous namespace

bool NativeDynamicShapePlan::anySegmentNeedsWarmup() const {
  for (const auto& seg : segments_) {
    if (seg.exec.executionCount < 2) return true;
  }
  return false;
}

bool NativeDynamicShapePlan::allSegmentsReplayReady() const {
  bool hasReplayableSegment = false;
  int segIdx = 0;
  for (auto& seg : segments_) {
    // All-frozen-constant segments need no replay — outputs are already populated
    if (seg.def.allFrozenConstants) {
      DSP_DIAG(GRAPH_REPLAY, "allSegmentsReplayReady: seg[%d-%d] idx=%d SKIP (allFrozenConstants)",
               seg.def.startSlot, seg.def.endSlot, segIdx);
      segIdx++;
      continue;
    }
    // Terminal-outcome segments (ZERO_KERNEL_SBS, NOT_FUSIBLE, COMPILE_FAILED)
    // and non-capturable segments execute slot-by-slot — no replay needed
    if (isTerminalOutcome(seg.exec.outcome) || !seg.def.isCapturable) {
      DSP_DIAG(GRAPH_REPLAY, "allSegmentsReplayReady: seg[%d-%d] idx=%d SKIP (terminal=%d capturable=%d outcome=%d)",
               seg.def.startSlot, seg.def.endSlot, segIdx,
               isTerminalOutcome(seg.exec.outcome) ? 1 : 0, seg.def.isCapturable ? 1 : 0,
               static_cast<int>(seg.exec.outcome));
      segIdx++;
      continue;
    }
    // This is a capturable segment — it must have a ready replay handle
    // Monolithic replay handle
    if (seg.exec.replayHandle != nullptr && seg.exec.replayHandle->isReady()) {
      DSP_DIAG(GRAPH_REPLAY, "allSegmentsReplayReady: seg[%d-%d] idx=%d READY (monolithic handle)",
               seg.def.startSlot, seg.def.endSlot, segIdx);
      hasReplayableSegment = true;
      segIdx++;
      continue;
    }
    // Composite replay handles
    if (segmentHasReadyCompositeHandles(seg)) {
      DSP_DIAG(GRAPH_REPLAY, "allSegmentsReplayReady: seg[%d-%d] idx=%d READY (composite handles)",
               seg.def.startSlot, seg.def.endSlot, segIdx);
      hasReplayableSegment = true;
      segIdx++;
      continue;
    }
    // This segment has no replay handle — fast path cannot be used
    DSP_DIAG(GRAPH_REPLAY, "allSegmentsReplayReady: seg[%d-%d] idx=%d NOT READY "
             "(capturable but no handle: phase=%s replayHandle=%p)",
             seg.def.startSlot, seg.def.endSlot, segIdx,
             seg.exec.segPhase.displayName(), (void*)seg.exec.replayHandle.get());
    return false;
  }
  // At least one segment must have an actual replay handle for the fast path
  // to be meaningful. If all segments were skipped (frozen/non-capturable/0-node),
  // there's nothing to replay — fall through to normal execution.
  DSP_DIAG(GRAPH_REPLAY, "allSegmentsReplayReady: result=%d (segCount=%d)",
           (int)hasReplayableSegment, segIdx);
  return hasReplayableSegment;
}

namespace {

bool segmentIsCompiledSteadyState(const GraphSegment& seg, int minExecutionCountExclusive) {
  if (!seg.exec.segPhase.needsCapture()) return false;
  if (seg.exec.executionCount <= minExecutionCountExclusive) return false;

  switch (seg.def.selectedBackend) {
    case SelectedBackend::CPU_GRAPH:
      return seg.resolvedCpuBackend != nullptr;
    case SelectedBackend::GPU_COMPILER:
      return !seg.exec.compiledByBackend.empty();
    default:
      return false;
  }
}

// Delegate to shared utilities in DspAnalysisUtils.h
uint32_t resolvePlanPhaseTraits(const NativeSlot& slot) {
  return dsp::resolveSlotTraits(slot);
}

int findProducerStepInSegment(const GraphSegment& seg, NativeSlot* slots, int outputSlotIdx) {
  return dsp::findProducerStepInSegment(seg, slots, outputSlotIdx);
}

bool segmentHasInternalValueShapeInputs(const GraphSegment& seg, NativeSlot* slots) {
  return dsp::segmentHasInternalValueShapeInputs(seg, slots);
}

bool isSmallIntegralControlArray(NDArray* arr) {
  if (arr == nullptr) return false;
  const auto dt = arr->dataType();
  if (dt != INT32 && dt != INT64 && dt != BOOL) return false;
  const auto len = arr->lengthOf();
  return len > 0 && len <= 32;
}

bool segmentHasStablePointersForPlanPhase(const GraphSegment& seg, NativeSlot* slots) {
  // Single debuggable decision (mirrors segmentIsFullyReplayingForPlanPhase): compute
  // pointer stability for the backend, log it, return it. One exit point so an
  // unstable segment is traceable in DSP_DIAG. Logic preserved from the prior
  // per-case early returns.
  bool stable;
  const char* why;

  if (!segmentBlocksPlanPhase(seg)) {
    stable = true;  why = "non_blocking";
  } else if (isTerminalOutcome(seg.exec.outcome)) {
    // Terminal (ZERO_KERNEL_SBS / NOT_FUSIBLE / COMPILE_FAILED): permanently
    // slot-by-slot, never participates in graph replay → don't block stability.
    stable = true;  why = "terminal";
  } else {
    switch (seg.def.selectedBackend) {
      case SelectedBackend::EMULATED_REPLAY:
        stable = !seg.exec.needsArgRefresh();  why = "emulated";
        break;
      case SelectedBackend::CPU_GRAPH:
        stable = seg.exec.segPhase.isSealed() || segmentIsCompiledSteadyState(seg, 1);
        why = "cpu_graph";
        break;
      case SelectedBackend::GPU_COMPILER: {
        const bool argStable =
            !segmentHasInternalValueShapeInputs(seg, slots) || !seg.exec.needsArgRefresh();
        const bool hasReadyReplay =
            (seg.exec.replayHandle && seg.exec.replayHandle->isReady()) ||
            segmentHasReadyCompositeHandles(seg);
        if (seg.exec.segPhase.isSealed()) {
          // Sealed: stable if it never expected a replay graph, else once the graph
          // is ready (and args are stable).
          const bool expectsReplay =
              hasReadyReplay || seg.exec.compiledByBackend == "Triton GPU";
          stable = !expectsReplay || (hasReadyReplay && argStable);
          why = "gpu_sealed";
        } else if (hasReadyReplay) {
          stable = argStable;  why = "gpu_handle_ready";
        } else {
          stable = segmentIsCompiledSteadyState(seg, 1) && argStable;  why = "gpu_compiled";
        }
        break;
      }
      case SelectedBackend::CUDA_GRAPHS:
      case SelectedBackend::SLOT_BY_SLOT:
        stable = seg.exec.replayHandle && seg.exec.replayHandle->isReady();
        why = "cuda_handle";
        break;
      default:
        stable = false;  why = "unknown_backend";
        break;
    }
  }

  DSP_DIAG(EXECUTE,
           "segmentHasStablePointersForPlanPhase: seg[%d-%d] backend=%d phase=%s outcome=%d "
           "-> %s (%s)",
           seg.def.startSlot, seg.def.endSlot, (int)seg.def.selectedBackend,
           seg.exec.displayPhaseName(), (int)seg.exec.outcome,
           stable ? "STABLE" : "UNSTABLE", why);
  return stable;
}

bool segmentIsFullyReplayingForPlanPhase(const GraphSegment& seg) {
  // Single debuggable decision: compute "is this segment in steady-state replay" for
  // its backend, then log it and return it. One exit point so a stuck segment is
  // visible in DSP_DIAG (backend, phase/outcome/execCount, and the verdict + reason)
  // — never a scattered set of early returns that can't be traced.
  bool replaying;
  const char* why;

  if (!segmentBlocksPlanPhase(seg)) {
    replaying = true;  why = "non_blocking";
  } else if (isTerminalOutcome(seg.exec.outcome)) {
    // Terminal (ZERO_KERNEL_SBS / FAILED): permanently in its final state.
    replaying = true;  why = "terminal";
  } else {
    const bool sealed = seg.exec.segPhase.isSealed();
    // A ready replay graph: monolithic handle OR ready composite (Triton island+gap)
    // handles. Composite is a no-op for non-composite backends.
    const bool graphReady =
        (seg.exec.replayHandle && seg.exec.replayHandle->isReady()) ||
        segmentHasReadyCompositeHandles(seg);
    switch (seg.def.selectedBackend) {
      case SelectedBackend::EMULATED_REPLAY:
        // Re-executes every slot fresh each step (no baked graph); sealed == replaying.
        replaying = sealed;  why = "emulated_sealed";
        break;
      case SelectedBackend::CPU_GRAPH:
        replaying = sealed || segmentIsCompiledSteadyState(seg, 2);  why = "cpu_graph";
        break;
      case SelectedBackend::GPU_COMPILER:
      case SelectedBackend::CUDA_GRAPHS:
      case SelectedBackend::SLOT_BY_SLOT:
        // Replaying ⟺ SEALED AND a ready graph. A merely-"compiled steady state"
        // segment (still CAPTURING, no graph) is NOT replaying — that desync let the
        // seal-gate block mid-capture and froze executeCount, forcing slot-by-slot.
        replaying = sealed && graphReady;  why = "gpu_graph_ready";
        break;
      default:
        replaying = false;  why = "unknown_backend";
        break;
    }
  }

  DSP_DIAG(EXECUTE,
           "segmentIsFullyReplayingForPlanPhase: seg[%d-%d] backend=%d phase=%s outcome=%d "
           "execCount=%d -> %s (%s)",
           seg.def.startSlot, seg.def.endSlot, (int)seg.def.selectedBackend,
           seg.exec.displayPhaseName(), (int)seg.exec.outcome, seg.exec.executionCount,
           replaying ? "REPLAYING" : "NOT_REPLAYING", why);
  return replaying;
}
/**
 * Returns the number of "structural" iArgs for an op — these are control parameters
 * (masks, mode flags, axis indices) that are always passed via iArgs regardless of
 * whether data parameters come from input tensors or from iArgs.
 * Returns -1 if all iArgs are structural (the default for most ops).
 */
static int getStructuralIArgCount(const std::string& opName) {
    static const std::unordered_map<std::string, int> STRUCTURAL_IARGS = {
        {"strided_slice", 5},   // 5 mask bits (begin/end/shrink/new_axis/ellipsis)
        {"concat", 1},          // axis
        {"split", 1},           // num_splits
        {"split_v", 1},         // axis
        {"one_hot", 2},         // axis, depth
        {"top_k", 1},           // k
    };
    auto it = STRUCTURAL_IARGS.find(opName);
    return (it != STRUCTURAL_IARGS.end()) ? it->second : -1;
}

}  // namespace

// NativeSlot move operations removed: sub-structs manage their own memory.
// NativeSlot is now non-movable (deleted in header).

// ─── Broad pre-replay sync flag (weight rebind on cached-plan reuse) ─────────
// All transitions of needsBroadPreReplaySync_ go through these DSP_DIAG-logged
// accessors — never poke the field directly (same consolidation contract as the
// GraphSegmentExec state methods).

void NativeDynamicShapePlan::markWeightRebindNeedsBroadSync(const char* reason) {
  needsBroadPreReplaySync_ = true;
  DSP_DIAG(EXECUTE,
           "BROAD_SYNC_MARK: %s — weight DataBuffer rebind detected; next ext-input "
           "H2D prepare will cover ALL inputs (phase=%s execCount=%d)",
           reason, planLifecycle_.displayName(), executeCount_);
}

bool NativeDynamicShapePlan::consumeBroadPreReplaySync(const char* site) {
  if (!needsBroadPreReplaySync_) return false;
  needsBroadPreReplaySync_ = false;
  DSP_DIAG(EXECUTE,
           "BROAD_SYNC_CONSUME: %s — broad ext-input H2D prepare for weight rebind; "
           "flag cleared (phase=%s execCount=%d)",
           site, planLifecycle_.displayName(), executeCount_);
  return true;
}

// ─── NativeDynamicShapePlan ─────────────────────────────────────────────────

NativeDynamicShapePlan::NativeDynamicShapePlan()
    : slots_(nullptr), numSlots_(0), totalOutputSlots_(0), numExternalInputs_(0),
      releaseAtStep_(nullptr), releaseAtStepCounts_(nullptr),
      requestedOutputSlotIndices_(nullptr), numRequestedOutputs_(0),
      outputSlots_(nullptr),
      contextPool_(nullptr), viewProducerDetectionDone_(false), frozenConstantDetectionDone_(false),
      gpuGraphCaptureEnabled_(false), totalGraphReplays_(0), jitMode_(JitMode::GRAPH_ONLY), graphExecutionMode_(GraphExecutionMode::GEM_AUTO),
      executeCount_(0), syncOverrideDepth_(0), shapePrePassDone_(true), executionTimingEnabled_(false), traceEnabled_(false),
      cpuGraphBackend_(nullptr), cpuGraphBackendChecked_(false),
      gpuGraphBackend_(nullptr), gpuGraphBackendChecked_(false),
      untrackedOutputCache_(nullptr), untrackedOutputCacheSize_(0),
      hasControlFlow_(false), loopRegions_(nullptr), numLoopRegions_(0),
      cfLoopBackStep_(-1),
      slotIsDead_(nullptr), slotIsDeadSize_(0),
      slotOwnership_(nullptr),
      dirtySlotGenerations_(),
      currentDirtyGeneration_(1) {
  trace_ = DspDiagnostics::getInstance().isEnabled(DSP_DIAG_ALL)
             ? new DspExecutionTrace()
             : nullptr;
}

void NativeDynamicShapePlan::pinSegmentGraphBakedSlots(GraphSegment& seg, NDArray** externalArrays,
                                                       int numExt, bool pinOwnedOutputs) {
  // CONTRACT: every device buffer a sealed segment will re-read on a later replay/re-exec MUST
  // outlive that replay. A buffer dangles (→ err700 illegal access, surfacing downstream e.g. at
  // add_scalar's stream sync) when a caller close()/rebind/pool-reuse frees it while the segment
  // still references its baked/cached device address. Pin each such address at SEAL so
  // CudaMemoryPool::free() DEFERS it (isGraphBakedPinned); released as deferred cudaFreeAsync at
  // teardown (platformFlushGraphBakedPins). Two distinct hazard classes, both covered here:
  //   • VIEW slot outputs + SOURCE_VARIABLE inputs — their device buffer is owned EXTERNALLY (a
  //     weight/variable or a prior slot the user may close()/rebind). Pinned in EVERY mode,
  //     including slot-by-slot (NOT_FUSIBLE), where this was previously uncovered.
  //   • OWNED (non-view) intermediate outputs — pinned ONLY when a captured graph baked their
  //     raw address (pinOwnedOutputs). In slot-by-slot they are recomputed each exec and MUST
  //     stay freeable, so pinOwnedOutputs=false skips them (no transient-buffer leak).
  // Idempotent (dedup by address, plan-wide). GENERALIZES the prior lazy WRITE_SLOT-only pin.
  const int safeNumExt = (externalArrays != nullptr) ? numExt : 0;  // guard null external table
  auto pinOne = [&](NDArray* slotArr, int slotForDiag, bool externalOwned) {
    if (slotArr == nullptr) return;
    DataBuffer* db = slotArr->dataBuffer();
    if (db == nullptr || !db->isValid() || db->special() == nullptr) return;  // closed/garbage → skip
    void* addr = db->special();                          // for a VIEW this is its base buffer's addr
    for (const auto& pa : graphPinnedAddrs_) { if (pa.ptr == addr) return; }  // idempotent
    int dbDev = db->deviceId();
    platformPinGraphBakedAddress(addr, dbDev);
    graphPinnedAddrs_.push_back({addr, dbDev, seg.def.startSlot, externalOwned});
    DSP_DIAG(MEMORY, "SEAL_PIN: seg[%d-%d] slot=%d pinned baked addr=%p dev=%d extOwned=%d",
             seg.def.startSlot, seg.def.endSlot, slotForDiag, addr, dbDev, externalOwned ? 1 : 0);
  };
  // (1) Slot outputs. A VIEW shares an externally-owned base buffer (a weight/variable or prior
  // slot) — pin it ALWAYS (this is the close-weight dangling buffer: a reshape view over a
  // weight, reached as outputSlots_[slot]). An OWNED intermediate is pinned only when its raw
  // address was baked into the captured graph (pinOwnedOutputs).
  for (int ps = seg.def.startSlot; ps <= seg.def.endSlot && ps < totalOutputSlots_; ps++) {
    NDArray* out = outputSlots_[ps];
    if (out == nullptr) continue;
    // A VIEW shares an externally-owned base (externalOwned=true → defer free to teardown). An
    // OWNED graph-baked intermediate (pinOwnedOutputs) is plan-owned (externalOwned=false).
    if (out->isView() || pinOwnedOutputs) pinOne(out, ps, /*externalOwned=*/out->isView());
  }
  // (2) SOURCE_VARIABLE input buffers (trainable weights). The graph bakes their RAW device
  // address (vs staged addresses for SOURCE_EXTERNAL placeholders). A weight is reached through
  // inputSourceIndices that may be a prior-slot ref (>=0) OR the external encoding -(extIdx+1)
  // (<0) — the SAME resolution the slot executor uses (resolveInputSourceArray). The old code
  // only handled >=0 and indexed outputSlots_, silently dropping EVERY external-encoded weight;
  // resolve canonically so a direct (non-view) weight input is pinned too.
  for (int s = seg.def.startSlot; s <= seg.def.endSlot && s < numSlots_; s++) {
    const SlotWiring& w = slots_[s].wiring;
    if (w.inputSourceTypes == nullptr || w.inputSourceIndices == nullptr) continue;
    for (int i = 0; i < w.numInputs; i++) {
      if (w.inputSourceTypes[i] != SOURCE_VARIABLE) continue;     // only weights (raw-baked); skip externals/prior-slots
      NDArray* srcArr = dsp::resolveInputSourceArray(
          w.inputSourceIndices[i], outputSlots_, totalOutputSlots_, externalArrays, safeNumExt);
      pinOne(srcArr, w.inputSourceIndices[i], /*externalOwned=*/true);  // weight — defer free to teardown
    }
  }
}

void NativeDynamicShapePlan::writeOutputSlot(int slotIdx, NDArray* value, const char* tag) {
  if (slotIdx < 0 || slotIdx >= totalOutputSlots_) {
    DSP_THROW(EXECUTE, "writeOutputSlot: index %d out of range [0, %d)", slotIdx, totalOutputSlots_);
  }

  // Lifecycle check: catch stale/freed NDArray pointers BEFORE storing them
  if (value != nullptr) {
    uintptr_t addr = reinterpret_cast<uintptr_t>(value);
    if (addr < 0x10000) {
      char msg[512];
      snprintf(msg, sizeof(msg),
               "DSP LIFECYCLE ERROR: writeOutputSlot(%d, tag=%s) — value pointer %p "
               "is a stale/freed NDArray. This indicates a kernel or allocation "
               "returned a corrupted pointer. execCount=%d planPhase=%s",
               slotIdx, tag, (void*)value, executeCount_, planLifecycle_.displayName());
      THROW_EXCEPTION(msg);
    }
  }

  NDArray* old = outputSlots_[slotIdx];

  // DIAGNOSTIC: trace writes to the configured trace slot (ND4J_DSP_TRACE_SLOT)
  // across all phases, not just frozen execution. The warmup path is where
  // shared-session vision failures currently manifest, so restricting this to
  // frozen phases hides the relevant slot lineage.
  if (DSP_DIAG_ENABLED(MEMORY)) {
    int ts = sd::graph::DspDiagnostics::getInstance().traceSlot();
    if (ts >= 0 && slotIdx == ts) {
      auto* oldDb = old != nullptr ? old->dataBuffer() : nullptr;
      auto* newDb = value != nullptr ? value->dataBuffer() : nullptr;
      DSP_DIAG(MEMORY,
               "WOS_%d: tag=%s old=%p new=%p oldDb=%p newDb=%p oldShape=%s newShape=%s exec=%d phase=%s",
               slotIdx, tag, (void*)old, (void*)value, (void*)oldDb, (void*)newDb,
               old != nullptr ? ShapeUtils::shapeAsString(old).c_str() : "null",
               value != nullptr ? ShapeUtils::shapeAsString(value).c_str() : "null",
               executeCount_, planLifecycle_.displayName());
    }
  }

  if (planLifecycle_.isShapesFrozen() || planLifecycle_.isReplaying()) {
    if (old != nullptr && value != nullptr && old != value) {
      // A pure wrapper swap — same underlying DataBuffer, identical
      // offset/shape/stride/dtype — is NOT a lifecycle violation.  View ops
      // like reshape/permute/slice routinely mint a fresh NDArray wrapper over
      // the same input buffer on every execution; the slot pointer differs but
      // no memory actually changes hands.  Only reject swaps that would
      // install a different DataBuffer (or a differently-shaped view of the
      // same buffer), which is a real lifecycle violation.
      const bool sameBuffer =
          old->dataBuffer() != nullptr &&
          value->dataBuffer() != nullptr &&
          old->dataBuffer() == value->dataBuffer();
      // Old view may have been destructed (use-after-free from stale view
      // refresh) — _shapeInfo is nullptr after destructor.
      // Check alignment first to catch +1 byte corruption before hasValidShapeInfo
      // tries to dereference the corrupted pointer.
      auto* oldSib = old->shapeInfoConstBuffer();
      bool oldShapeValid = false;
      if (oldSib != nullptr) {
        uintptr_t oldSibAddr = reinterpret_cast<uintptr_t>(oldSib);
        if (oldSibAddr % alignof(ConstantShapeBuffer) != 0) {
          DSP_DIAG(MEMORY,
                   "WRITE_SLOT_CORRUPTION: slot=%d tag=%s old=%p has misaligned "
                   "_shapeInfoBuffer=%p (alignOffset=%zu) — heap corruption detected",
                   slotIdx, tag, (void*)old, (void*)oldSib,
                   static_cast<size_t>(oldSibAddr % alignof(ConstantShapeBuffer)));
          // Treat as invalid — don't try to read shape info
        } else {
          oldShapeValid = old->hasValidShapeInfo();
        }
      } else {
        // _shapeInfoBuffer is null — check raw _shapeInfo
        oldShapeValid = old->hasValidShapeInfo();
      }
      const bool sameView = sameBuffer && oldShapeValid &&
          old->offset() == value->offset() &&
          old->dataType() == value->dataType() &&
          shape::shapeEquals(old->shapeInfo(), value->shapeInfo()) &&
          shape::strideEquals(old->shapeInfo(), value->shapeInfo());
      const bool isViewProducerSlot =
          slots_[slotIdx].slotPhase.isViewProducer;
      const bool isViewOpTag = tag != nullptr &&
          (strcmp(tag, "view-op-install") == 0 ||
           strcmp(tag, "view-op-reuse") == 0 ||
           strcmp(tag, "ff-view-install") == 0 ||
           strcmp(tag, "view-install") == 0);
      const bool newDbValid = value->dataBuffer() != nullptr &&
          !value->dataBuffer()->isClosed();
      // A view-op tag with the SAME underlying DataBuffer is always a legitimate
      // wrapper swap, regardless of whether the slot was pre-flagged as a view
      // producer. View ops deterministically regenerate their output NDArray
      // wrapper each call (fresh metadata descriptor over the same memory); the
      // new wrapper may differ in offset/shape/stride details but no memory
      // changes hands, so this is safe.
      const bool allowSameBufferViewSwap = sameBuffer && isViewOpTag && newDbValid;
      // First-level view producers over external inputs (placeholders) legitimately
      // have a NEW DataBuffer every call: the upstream placeholder is fresh, and
      // the view wrapper is minted over that fresh input buffer.  Allow such swaps
      // when the tag proves this write comes from the view-op execution code
      // (view-op-install / view-op-reuse / ff-view-install / view-install).
      // The tag is authoritative — refreshStaleViewWrappersInSegment may have
      // demoted isViewProducer during pre-execution cleanup (e.g., placeholder
      // closed between calls), but the view-op code is about to re-set it.
      const bool allowViewProducerDbSwap = isViewOpTag && newDbValid;
      // Control flow ops (merge, switch, enter, exit, etc.) legitimately
      // produce different output pointers on each execution depending on
      // which branch is active. CF slots are already non-capturable (never
      // included in CUDA/Triton graph capture) and always execute slot-by-slot,
      // so pointer replacement is safe.
      const bool isCfTag = tag != nullptr &&
          (strcmp(tag, "cf-merge") == 0 ||
           strcmp(tag, "cf-switch-live") == 0 ||
           strcmp(tag, "cf-switch-dead") == 0 ||
           strcmp(tag, "cf-enter") == 0 ||
           strcmp(tag, "cf-exit") == 0 ||
           strcmp(tag, "cf-loop-cond") == 0 ||
           strcmp(tag, "cf-next-iter") == 0);
      if (!sameView && !allowSameBufferViewSwap && !allowViewProducerDbSwap && !isCfTag) {
        DSP_THROW(EXECUTE,
                 "LIFECYCLE VIOLATION: NDArray pointer replacement at slot %d (tag=%s) "
                 "during frozen phase (phase=%s execCount=%d). old=%p new=%p "
                 "oldDb=%p newDb=%p",
                 slotIdx, tag, planLifecycle_.displayName(), executeCount_,
                 (void*)old, (void*)value,
                 (void*)(old->dataBuffer()), (void*)(value->dataBuffer()));
      }
      if (!sameView && allowViewProducerDbSwap) {
        DSP_DIAG(MEMORY,
                 "WRITE_SLOT_VIEW_DB_SWAP: slot=%d tag=%s oldDb=%p newDb=%p "
                 "(view-producer wrapper updated for fresh external input)",
                 slotIdx, tag, (void*)(old->dataBuffer()), (void*)(value->dataBuffer()));
      }
      // Wrapper-swap accepted — fall through to normal slot update logic,
      // which will free the old wrapper and install the new one.  The
      // underlying DataBuffer is refcounted, so neither wrapper owns it
      // exclusively.
      DSP_DIAG(MEMORY,
               "WRITE_SLOT_WRAPPER_SWAP: slot=%d tag=%s same DataBuffer %p "
               "(old=%p new=%p execCount=%d)",
               slotIdx, tag, (void*)(old->dataBuffer()),
               (void*)old, (void*)value, executeCount_);
    }
  }

  // Phase assertion: during REPLAYING, output slots that belong to a captured
  // Triton island should only be written by the graph itself (via kernel output).
  // Writing from the host indicates a phase violation (e.g., a slot-by-slot op
  // overwriting a graph-owned buffer). Gated behind DSP_DIAG_ENABLED to avoid
  // O(segments * units) iteration overhead per write in the hot decode loop.
  if (DSP_DIAG_ENABLED(EXECUTE) &&
      planLifecycle_.isReplaying() && executeCount_ > 2 &&
      old != nullptr && value != nullptr && old != value) {
    // Check if this slot belongs to any segment currently in REPLAYING phase
    for (const auto& seg : segments_) {
      if (seg.exec.segPhase.isSealed() &&
          slotIdx >= seg.def.startSlot && slotIdx <= seg.def.endSlot) {
        // Check if this is a Triton island slot (not a gap)
        bool isIslandSlot = false;
        for (const auto& unit : seg.exec.compositeReplaySchedule.units) {
          if (unit.kind == REPLAY_UNIT_TRITON_ISLAND &&
              slotIdx >= unit.startSlot && slotIdx <= unit.endSlot) {
            isIslandSlot = true;
            break;
          }
        }
        if (isIslandSlot) {
          DSP_DIAG(EXECUTE, "PHASE_VIOLATION: writeOutputSlot(%d, tag=%s) writes to "
                   "Triton island slot during REPLAYING phase — graph owns this buffer. "
                   "seg[%d-%d] execCount=%d",
                   slotIdx, tag, seg.def.startSlot, seg.def.endSlot, executeCount_);
          REQUIRE_TRUE(false, 0,
                       "DSP phase contract violation: writeOutputSlot(%d) to Triton island slot "
                       "during REPLAYING phase for seg[%d-%d].",
                       slotIdx, seg.def.startSlot, seg.def.endSlot);
        }
        break;
      }
    }
  }

  // Array validity check: catch closed, destroyed, corrupt, or null-GPU arrays
  // at the moment they enter the slot system — not later when a kernel crashes.
  if (value != nullptr && value->hasValidShapeInfo() && !value->isEmpty()) {
    ArrayInvalidReason reason = validateArrayForExecution(value);
    if (reason != ArrayInvalidReason::VALID) {
      DataBuffer* db = value->dataBuffer();
      DSP_THROW(MEMORY,
               "ARRAY_INVALID in writeOutputSlot: slot=%d tag=%s reason=%s "
               "DataBuffer=%p exec=%d phase=%s — "
               "invalid array being installed into slot system",
               slotIdx, tag, arrayInvalidReasonName(reason),
               (void*)db, executeCount_, planLifecycle_.displayName());
    }
  }

  if (value != nullptr && value != old &&
      planOwnedArrays_.count(value) == 0 &&
      value->dataBuffer() != nullptr &&
      protectedWeightBuffers_.count(value->dataBuffer()) == 0) {
    planOwnedArrays_.insert(value);
  }

  DSP_DIAG(MEMORY, "WRITE_SLOT: slot=%d tag=%s phase=%s execCount=%d plan=%p value=%p",
           slotIdx, tag, planLifecycle_.displayName(), executeCount_,
           (void*)this, (void*)value);

  // Capture old buffer address BEFORE any deletion — specialBuffer() calls
  // shapeInfo() internally (via sizeOfT→dataType), so calling DSP_BUF_SAFE
  // after delete is use-after-free on the destroyed NDArray's _shapeInfo.
  //
  // MUST be a raw special() peek behind an isValid() gate, NOT
  // dspBufferSafe→NDArray::specialBuffer(): specialBuffer() SELF-HEALS a
  // device-null buffer by calling allocateSpecial(). When this plan came from
  // the shape-keyed plan cache and the previous borrower's test closed its
  // inputs, `old` is a stale view over a closed/freed DataBuffer — the
  // self-heal then runs allocateSpecial on the dead object and reads its
  // freed _workspace field (Workspace::allocateBytes SIGSEGV with poisoned
  // `this`; MALLOC_PERTURB_ shows ws=0xaaaa...). Same isValid() contract as
  // the delete-guard below: closed, destroyed, and reused-garbage buffers
  // all fail the magic check and are never touched.
  uint64_t oldBufAddr = 0;
  if (old != nullptr) {
    auto* oldDbPeek = old->dataBuffer();
    if (oldDbPeek != nullptr && oldDbPeek->isValid()) {
      oldBufAddr = reinterpret_cast<uint64_t>(oldDbPeek->special());
    }
  }

  // Free the OLD plan-owned array when it's being replaced, unless its
  // DataBuffer is a protected weight or shared with another slot.
  if (old != nullptr && old != value) {
    bool isPlanOwned = planOwnedArrays_.count(old) > 0;
    if (isPlanOwned) {
      auto* oldDb = old->dataBuffer();
      bool isProtected = oldDb != nullptr && protectedWeightBuffers_.count(oldDb) > 0;
      if (!isProtected) {
        // Check that no OTHER slot still references this exact NDArray pointer.
        // View ops can share the same NDArray across slots.
        bool referencedElsewhere = false;
        for (int i = 0; i < totalOutputSlots_; i++) {
          if (i != slotIdx && outputSlots_[i] == old) {
            referencedElsewhere = true;
            break;
          }
        }
        if (!referencedElsewhere) {
          planOwnedArrays_.erase(old);
          outputSlots_[slotIdx] = nullptr;  // Null slot BEFORE delete to prevent dangling pointer window
          // Guard: if the DataBuffer was closed from the Java side (e.g. a placeholder closed
          // between DSP calls), the C++ DataBuffer object still exists (JavaCPP doesn't immediately
          // free it) but its memory has been released. Deleting a view NDArray that references
          // a closed DataBuffer would access freed memory → heap corruption → DataType::UNKNOWN.
          //
          // A slot that VIEWS an external input can also have its DataBuffer C++ object fully
          // DESTROYED (magic → MAGIC_DESTROYED) by JavaCPP GC after the test closes + re-supplies
          // that input — not merely closed. isClosed() alone reads the `closed` field out of the
          // freed object (garbage) and lets us go on to read special()/deviceId() (the ~DataBuffer
          // poison) → pin/free a bogus address (CUDA err700), then delete the NDArray whose
          // ~NDArray poison is later reused by a cached op-context as a Workspace `this` →
          // Workspace::allocateBytes SIGSEGV. isValid() (magic == MAGIC_NUMBER && !closed) rejects
          // closed, destroyed, AND reused-garbage buffers in one check, so treat any non-valid
          // DataBuffer as unsafe-to-touch: skip the pin/deviceId reads AND the delete (leaking the
          // small view NDArray), exactly as the closed-db path already does.
          bool oldDbUnsafe = (oldDb != nullptr && !oldDb->isValid());
          if (!oldDbUnsafe) {
            // Defer deletion until after the full execute() step completes.
            // Deleting inline while plan execution is still iterating slots can
            // corrupt heap metadata: the allocator's free-list update overwrites
            // adjacent allocations (seen as Workspace::allocateBytes SIGSEGV with
            // a garbage `this` pointer full of ASCII string data).
            // flushDeferredSlotDeletes() is called at every platformEndExecution
            // site in execute(), by which point slot iteration is finished and the
            // heap is in a consistent state.
            //
            // Graph-baked address protection: if any segment is SEALED (has a live
            // captured CUDA graph with this slot's GPU address baked in), pin that
            // address in CudaMemoryPool to prevent pool reuse. The pool would
            // otherwise re-hand this address to a subsequent allocation (e.g., a
            // ref-SD weight during testBufferAliasVaryingInput), and the CUDA graph
            // replay would then read the wrong memory → CUDA err700 (illegal access).
            // The pin is released by platformCleanupSegmentForRebuild /
            // platformFreePlanResources / releaseGpuIntermediates when the graph is
            // destroyed — at which point the deferred cudaFreeAsync is issued.
            // Only pin if the NDArray owns its GPU buffer (~NDArray calls deleteSpecial
            // only when !isView). Pinning a view's pointer would cause a spurious
            // cudaFreeAsync when we unpin — the view never issued a cudaFreeAsync via
            // CudaMemoryPool::free() in the first place.
            void* oldSpecial = (oldDb != nullptr && !old->isView() && oldDb->special() != nullptr)
                                ? oldDb->special() : nullptr;
            // Find the sealed segment that covers this slot — only that segment's
            // CUDA graph has this address baked in. segStartSlot=-1 means none found.
            int sealedSegStart = -1;
            if (oldSpecial != nullptr) {
              for (const auto& seg : segments_) {
                if (seg.exec.segPhase.isSealed() &&
                    slotIdx >= seg.def.startSlot && slotIdx <= seg.def.endSlot) {
                  sealedSegStart = seg.def.startSlot;
                  break;
                }
              }
            }
            if (sealedSegStart >= 0 && oldSpecial != nullptr) {
              int dbDev = oldDb->deviceId();
              platformPinGraphBakedAddress(oldSpecial, dbDev);
              graphPinnedAddrs_.push_back({oldSpecial, dbDev, sealedSegStart, /*externalOwned=*/false});  // overwritten non-view intermediate
              DSP_DIAG(MEMORY, "WRITE_SLOT_FREE_DEFERRED: slot=%d pinned graph-baked addr=%p dev=%d seg[start=%d] old=%p (db=%p)",
                       slotIdx, oldSpecial, dbDev, sealedSegStart, (void*)old, (void*)oldDb);
            } else {
              DSP_DIAG(MEMORY, "WRITE_SLOT_FREE_DEFERRED: slot=%d deferred delete of old=%p (db=%p)",
                       slotIdx, (void*)old, (void*)oldDb);
            }
            tl_deferredSlotDeletes.push_back(old);
          } else {
            DSP_DIAG(MEMORY,
                     "WRITE_SLOT_SKIP_FREE_UNSAFE: slot=%d old array %p has invalid (closed or "
                     "GC-destroyed) db=%p — skipping delete to avoid use-after-free (view of freed input)",
                     slotIdx, (void*)old, (void*)oldDb);
          }
          old = nullptr;  // Prevent any further access to freed memory
        }
      } else {
        DSP_DIAG(MEMORY, "WRITE_SLOT_SKIP_FREE: slot=%d old array %p has protected weight db=%p — not freed",
                 slotIdx, (void*)old, (void*)oldDb);
      }
    } else {
      // NOT plan-owned but being replaced — this is a potential leak.
      long long leakedBytes = old->dataBuffer() ? (long long)old->dataBuffer()->getLenInBytes() : 0;
      DSP_DIAG(MEMORY, "WRITE_SLOT_LEAK: slot=%d tag=%s old=%p NOT plan-owned, bytes=%lld planOwned=%d",
               slotIdx, tag, (void*)old, leakedBytes, (int)planOwnedArrays_.size());
    }
  }

  // Structured trace: record every slot write with buffer identity.
  // When the buffer address changes (replacement), emit BUFFER_REPLACED so
  // post-mortem analysis can distinguish genuine replacements from same-buffer
  // re-writes (e.g. in-place ops that reuse the same allocation).
  // Uses pre-captured oldBufAddr — old NDArray may have been queued for deferred deletion above.
  // Short-circuit when trace_ is null (the common case in production) to avoid
  // computing newAddr and the two DSP_BUF_SAFE() calls on every write.
  if (trace_ != nullptr) {
    uint64_t newAddr = (value != nullptr && value->dataBuffer() != nullptr)
        ? reinterpret_cast<uint64_t>(sd::graph::dspBufferSafe(value)) : 0;
    if (oldBufAddr != 0 && oldBufAddr != newAddr) {
      // Buffer REPLACED — record both old and new addresses.
      trace_->recordBufferReplaced(-1, slotIdx,
                                   static_cast<uint32_t>(executeCount_),
                                   oldBufAddr, newAddr);
    } else {
      DSP_TRACE_SLOT_WRITTEN(trace_, -1, slotIdx,
                             static_cast<uint32_t>(executeCount_), newAddr);
    }
  }

  outputSlots_[slotIdx] = value;
}

// ═══════════════════════════════════════════════════════════════════════════════
// clearOutputSlot — the ONLY way to null an output slot
//
// Self-contained: inspects the array currently at slotIdx and decides how to
// clean it up (deferred delete, ownership removal, etc.) based on its state.
// The caller just names the slot and the reason — no external boolean flags.
// ═══════════════════════════════════════════════════════════════════════════════
void NativeDynamicShapePlan::clearOutputSlot(int slotIdx, const char* tag, bool deferDelete) {
  if (slotIdx < 0 || slotIdx >= totalOutputSlots_) return;

  NDArray* old = outputSlots_[slotIdx];
  if (old == nullptr) return;  // Already null — no-op

  // Inspect the array's current state for the diagnostic record
  DataBuffer* db = old->dataBuffer();
  bool dbNull = (db == nullptr);
  bool dbClosed = (!dbNull && db->isClosed());
  bool dbInvalid = (!dbNull && !db->isValid());
  bool isPlanOwned = planOwnedArrays_.count(old) > 0;
  bool hasValidShape = (old->shapeInfo() != nullptr && old->hasValidShapeInfo());

  DSP_DIAG(MEMORY, "CLEAR_SLOT: slot=%d tag=%s old=%p "
           "planOwned=%d dbNull=%d dbClosed=%d dbInvalid=%d hasValidShape=%d "
           "deferDelete=%d phase=%s exec=%d",
           slotIdx, tag, (void*)old,
           (int)isPlanOwned, (int)dbNull, (int)dbClosed, (int)dbInvalid,
           (int)hasValidShape, (int)deferDelete,
           planLifecycle_.displayName(), executeCount_);

  if (isPlanOwned) {
    planOwnedArrays_.erase(old);
    // Only defer-delete if the DataBuffer is still live — deleting an array
    // whose DataBuffer was closed or GC-destroyed from Java causes use-after-free.
    // isValid() checks both the magic number (MAGIC_NUMBER, not MAGIC_DESTROYED)
    // AND the closed flag in one call, covering the case where a DataBuffer has
    // been fully freed (magic overwritten) but `closed` reads as false from the
    // freed memory (UB/garbage). Prior guard checked only !dbClosed && !dbNull,
    // missing the MAGIC_DESTROYED-but-closed-reads-false path that produced
    // Workspace::allocateBytes SIGSEGV with this=0xDEADBEEFCAFEBABE.
    if (deferDelete && !dbNull && db->isValid()) {
      tl_deferredSlotDeletes.push_back(old);
    }
  }

  if (slotOwnership_ != nullptr) {
    slotOwnership_[slotIdx].reset();
  }

  outputSlots_[slotIdx] = nullptr;
}

// ═══════════════════════════════════════════════════════════════════════════════
// markViewProducer — set the structural view-producer flag on a slot
//
// This is a STRUCTURAL property of the op (permute/reshape always produce
// views). Distinct from phase transitions — persists across reset()/unseal().
// ═══════════════════════════════════════════════════════════════════════════════
void NativeDynamicShapePlan::markViewProducer(int slotIdx, const char* tag) {
  if (slotIdx < 0 || slotIdx >= numSlots_) return;
  bool was = slots_[slotIdx].slotPhase.isViewProducer;
  slots_[slotIdx].slotPhase.isViewProducer = true;
  if (!was) {
    DSP_DIAG(LIFECYCLE, "MARK_VIEW_PRODUCER: slot=%d tag=%s phase=%s exec=%d",
             slotIdx, tag, planLifecycle_.displayName(), executeCount_);
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// demoteViewProducer — clear the view-producer flag, inspect the slot, clean up
//
// Self-contained: examines the current output array at this slot and decides
// whether to null it out based on the array's DataBuffer validity. The caller
// just names the slot and the reason — the method handles all the inspection.
//
// Decision logic (logged in DSP_DIAG so every code path is visible):
//   - If no output array: just clear the flag.
//   - If output array has no valid shape info: clear flag + null slot
//     (array is corrupt, can't safely keep it).
//   - If output array's DataBuffer is null/closed/invalid: clear flag + null slot
//     (view wraps freed memory, keeping it would crash on access).
//   - If output array has valid DataBuffer: clear flag only
//     (array data is fine, just stop treating it as a view).
// ═══════════════════════════════════════════════════════════════════════════════
void NativeDynamicShapePlan::demoteViewProducer(int slotIdx, const char* tag, bool /*unused*/) {
  if (slotIdx < 0 || slotIdx >= numSlots_) return;

  bool wasViewProducer = slots_[slotIdx].slotPhase.isViewProducer;
  slots_[slotIdx].slotPhase.isViewProducer = false;

  // Inspect the output slot to decide whether it needs to be nulled
  int outSi = slotIdx;  // For view-producer slots, outSi == slotIdx (the output slot index)
  NDArray* cached = (outSi >= 0 && outSi < totalOutputSlots_) ? outputSlots_[outSi] : nullptr;

  if (cached == nullptr) {
    DSP_DIAG(LIFECYCLE, "DEMOTE_VIEW_PRODUCER: slot=%d tag=%s wasView=%d "
             "output=null (flag cleared, no cleanup needed) phase=%s exec=%d",
             slotIdx, tag, (int)wasViewProducer,
             planLifecycle_.displayName(), executeCount_);
    return;
  }

  bool hasValidShape = (cached->shapeInfo() != nullptr && cached->hasValidShapeInfo());
  DataBuffer* db = hasValidShape ? cached->dataBuffer() : nullptr;
  bool dbValid = (db != nullptr && db->isValid() && !db->isClosed());

  bool needsClear = !hasValidShape || !dbValid;

  DSP_DIAG(LIFECYCLE, "DEMOTE_VIEW_PRODUCER: slot=%d tag=%s wasView=%d "
           "cached=%p hasValidShape=%d dbValid=%d needsClear=%d phase=%s exec=%d",
           slotIdx, tag, (int)wasViewProducer,
           (void*)cached, (int)hasValidShape, (int)dbValid, (int)needsClear,
           planLifecycle_.displayName(), executeCount_);

  if (needsClear) {
    clearOutputSlot(outSi, tag, /*deferDelete=*/hasValidShape);
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// materializeViewSlot — replace a view with an independent deep copy
//
// Self-contained: inspects the slot, verifies it holds a valid view, creates
// an independent copy, and swaps ownership. No external conditions needed.
// ═══════════════════════════════════════════════════════════════════════════════
void NativeDynamicShapePlan::materializeViewSlot(int slotIdx, const char* tag) {
  if (slotIdx < 0 || slotIdx >= totalOutputSlots_) return;

  NDArray* viewArr = outputSlots_[slotIdx];
  if (viewArr == nullptr) return;

  DataBuffer* viewDb = viewArr->dataBuffer();
  if (viewDb == nullptr) {
    DSP_DIAG(LIFECYCLE, "MATERIALIZE_VIEW: slot=%d tag=%s SKIP — null DataBuffer",
             slotIdx, tag);
    return;
  }
  if (!viewDb->isValid() || viewDb->isClosed()) {
    DSP_DIAG(LIFECYCLE, "MATERIALIZE_VIEW: slot=%d tag=%s SKIP — DataBuffer invalid/closed "
             "(valid=%d closed=%d)",
             slotIdx, tag, (int)viewDb->isValid(), (int)viewDb->isClosed());
    return;
  }

  // Multi-GPU: a view output produced on a secondary device MUST be materialized ON that
  // device. dup() allocates the copy on the current device and reads the view's (device-N)
  // buffer; running it on the primary device (the current device at output extraction) would
  // copy across the non-peer boundary and silently produce garbage. Use the PRODUCING slot's
  // targetDeviceId — the DataBuffer's deviceId() metadata can be stale (0) when a device-0
  // pre-pass array had its data migrated to a secondary device. Single-GPU: producer <= 0.
  int viewDev = -1;
  for (int s = 0; s < numSlots_ && viewDev < 0; s++) {
    for (int o = 0; o < slots_[s].wiring.numOutputs; o++) {
      if (slots_[s].wiring.outputSlotIndices[o] == slotIdx) { viewDev = slots_[s].targetDeviceId; break; }
    }
  }
  int savedDev = -1;
  bool switchedDev = false;
  if (viewDev > 0) {
    savedDev = sd::graph::dspGetCurrentDevice();
    if (savedDev != viewDev) {
      sd::graph::dspSetCurrentDevice(viewDev);
      switchedDev = true;
    }
  }

  // Create an independent deep copy (on the view's device — see above). dup() already returns
  // a heap-allocated NDArray*; wrapping it in `new NDArray(...)` copies it and leaks the dup()
  // result — use the pointer directly.
  NDArray* dup = viewArr->dup(viewArr->ordering());

  if (switchedDev) sd::graph::dspSetCurrentDevice(savedDev);

  DSP_DIAG(LIFECYCLE, "MATERIALIZE_VIEW: slot=%d tag=%s "
           "oldArr=%p oldDb=%p newArr=%p newDb=%p shape=%s phase=%s exec=%d",
           slotIdx, tag,
           (void*)viewArr, (void*)viewDb,
           (void*)dup, (void*)dup->dataBuffer(),
           ShapeUtils::shapeAsString(dup).c_str(),
           planLifecycle_.displayName(), executeCount_);

  // Swap: remove old, install new
  planOwnedArrays_.erase(viewArr);
  planOwnedArrays_.insert(dup);
  if (slotOwnership_ != nullptr) {
    slotOwnership_[slotIdx].dataBuffer = dup->dataBuffer();
  }
  outputSlots_[slotIdx] = dup;
  delete viewArr;
}

void NativeDynamicShapePlan::setGraphExecutionMode(GraphExecutionMode mode) {
  // ── Platform-aware mode remapping ──────────────────────────────────────
  // On CPU builds without graph backends, modes that require hardware graph
  // capture (CUDA_GRAPHS, HIP_GRAPHS, etc.) cannot be honored. Rather than
  // letting the plan-level mode stay as GEM_CUDA_GRAPHS (which has
  // isSlotBySlot=false in ModeContract, driving freeze→compile→phaseReplay),
  // remap to GEM_EMULATED_REPLAY at the plan level. EMULATED_REPLAY has
  // isSlotBySlot=true, so populateDerivedState() correctly stays on the
  // slot-by-slot execution path while still tracking replay lifecycle
  // (shape freezing, pointer stability, segment timing).
#if !defined(HAVE_ONEDNN) && !defined(HAVE_OPENVINO) && \
    !defined(HAVE_ARMCOMPUTE) && !defined(HAVE_MLIR) && !defined(HAVE_NNAPI) && !defined(HAVE_MLX)
  if (!sd::graph::dspIsCudaBuild() && ModeContract::forMode(mode).usesGraphCapture) {
    DSP_DIAG(EXECUTE, "setGraphExecutionMode: remapping %d -> GEM_EMULATED_REPLAY (no graph backend on this platform)",
             static_cast<int>(mode));
    mode = GraphExecutionMode::GEM_EMULATED_REPLAY;
  }
#endif

  if (graphExecutionMode_ == mode) return;  // idempotent: no-op if unchanged
  // Mode is set once at plan creation (part of cache key). Phase guard ensures
  // this is never called after the plan has advanced — one flow, no reclassification.
  DSP_REQUIRE_PLAN_PHASE_AT_MOST(PlanPhase::SLOT_BY_SLOT, "setGraphExecutionMode");
  DSP_DIAG(EXECUTE, "setGraphExecutionMode: %d -> %d", static_cast<int>(graphExecutionMode_), static_cast<int>(mode));
  graphExecutionMode_ = mode;
  // Reset cached backends so buildSegments() uses the correct mode.
  gpuGraphBackendChecked_ = false;
  gpuGraphBackend_ = nullptr;
  cpuGraphBackendChecked_ = false;
  cpuGraphBackend_ = nullptr;
  cpuGraphBackendChainBuilt_ = false;
  cpuGraphBackendChain_.clear();
  // Enable GPU graph capture for modes that use hardware graph capture.
  // EMULATED_REPLAY and SLOT_BY_SLOT do not use graph capture, so leave
  // gpuGraphCaptureEnabled_ false for those.
  if (ModeContract::forMode(mode).usesGraphCapture) {
    gpuGraphCaptureEnabled_ = true;
  }
  // Clear GPU backend failed-compilation cache so segments that failed with
  // incomplete shapes (e.g., attention with seqK=0 before KV setup)
  // can retry when called again with correct external input shapes.
  clearGpuBackendFailedCache();
}

NativeDynamicShapePlan::~NativeDynamicShapePlan() {
  DSP_DIAG(MEMORY, "~NativeDynamicShapePlan: START plan=%p numSlots=%d totalOutputSlots=%d planOwned=%zu",
           this, numSlots_, totalOutputSlots_, planOwnedArrays_.size());
  const bool hadFrozenRefsOnEntry =
      planLifecycle_.isInFrozenOrReplayState() ||
      hasTrackedPlanFrozenRefs(frozenProtectedRefBuffers_, frozenOutputRefBuffers_);

  // ── Phase 1: Free GPU resources FIRST ─────────────────────────────────
  // Platform GPU resources (replay handles, JIT kernels, cuBLAS workspace,
  // batch-zero) may hold direct references into outputSlots_. Clean them
  // BEFORE freeing slot arrays to avoid dangling pointer access during teardown.
  DSP_DIAG(MEMORY, "~NativeDynamicShapePlan: freeing platform GPU resources");
  platformFreePlanResources();

  // Release KV scatter config (clears entry list, nulls position pointer).
  // kvPositionDevice_ is NOT freed here — it's owned by the Java caller.
  releaseKvScatterResources();

  // Remove frozen reference counts before deleting or nulling slot arrays.
  // Frozen sealing adds one output-slot ref per non-null slot, so release exactly
  // the recorded list; do not dedupe shared DataBuffers.
  releasePlanFrozenRefsForTeardown("~NativeDynamicShapePlan", hadFrozenRefsOnEntry,
                                   frozenProtectedRefBuffers_, frozenOutputRefBuffers_);

  // Free symbolic shape range profiles from all segments
  for (auto& seg : segments_) {
    if (seg.exec.symbolicRangeData != nullptr) {
      freeSegmentShapeProfile(static_cast<SegmentShapeProfile*>(seg.exec.symbolicRangeData));
      seg.exec.symbolicRangeData = nullptr;
    }
  }

  // ── Phase 2: Free slot data ───────────────────────────────────────────
  // Free slots metadata
  if (slots_) {
    delete[] slots_;
  }

  // Free release schedule
  if (releaseAtStep_) {
    for (int i = 0; i < numSlots_; i++) {
      delete[] releaseAtStep_[i];
    }
    delete[] releaseAtStep_;
  }
  delete[] releaseAtStepCounts_;

  // Free slot liveness data
  delete slotLiveness_;
  slotLiveness_ = nullptr;

  // Free requested output mapping
  delete[] requestedOutputSlotIndices_;

  // Dedup set to prevent double-free (identity ops can share pointers across slots)
  std::unordered_set<NDArray*> deleted;

  // Free slot arrays. Only delete arrays that the plan created (in planOwnedArrays_).
  // Arrays from external inputs or model variables are NOT plan-owned and must survive.
  DSP_DIAG(MEMORY, "~NativeDynamicShapePlan: freeing outputSlots_ (%d slots, %zu plan-owned)",
           totalOutputSlots_, planOwnedArrays_.size());
  if (outputSlots_) {
    int freedOwned = 0, skippedExternal = 0;
    for (int i = 0; i < totalOutputSlots_; i++) {
      if (outputSlots_[i] == nullptr) continue;
      if (!deleted.insert(outputSlots_[i]).second) continue;

      if (planOwnedArrays_.count(outputSlots_[i]) > 0) {
        // Pre-clean the DataBuffer before deleting the NDArray to avoid crash
        // in ~NDArray if _shapeInfo points to freed ConstantShapeHelper memory.
        // isValid() checks MAGIC_NUMBER + !closed in one call. Using only isClosed()
        // is insufficient: a GC-destroyed DataBuffer has MAGIC_DESTROYED magic, so
        // isClosed() reads garbage from freed memory and may return false, causing
        // deleteBuffers() on a stale pointer → Workspace::allocateBytes SIGSEGV
        // with this=0xDEADBEEFCAFEBABE (same root as the 3-site guarded fix in
        // clearOutputSlot/writeOutputSlot/refreshFrozenViews).
        auto* db = outputSlots_[i]->dataBuffer();
        bool dbSafe = (db != nullptr && db->isValid() && !db->isClosed());
        if (dbSafe) {
          db->deleteBuffers();
        }
        outputSlots_[i]->setShapeInfo((sd::LongType*)nullptr);
        freedOwned++;
        delete outputSlots_[i];
      } else {
        skippedExternal++;
      }
    }
    DSP_DIAG(MEMORY, "~NativeDynamicShapePlan: freed %d plan-owned, skipped %d external from outputSlots_",
             freedOwned, skippedExternal);
    DSP_DIAG(MEMORY, "~NativeDynamicShapePlan: about to delete[] outputSlots_ array (%p)", (void*)outputSlots_);
    delete[] outputSlots_;
    DSP_DIAG(MEMORY, "~NativeDynamicShapePlan: delete[] outputSlots_ done");
  }
  // outputSlots_ owns the NDArray* array — do NOT delete[] separately

  // Free placeholder staging buffers (plan-owned stable device buffers for variable inputs)
  DSP_DIAG(MEMORY, "~NativeDynamicShapePlan: stagingBuffers=%p numExtInputs=%d",
           (void*)placeholderStagingBuffers_, numExternalInputs_);
  if (placeholderStagingBuffers_ != nullptr) {
    int freedStaging = 0;
    for (int i = 0; i < numExternalInputs_; i++) {
      DSP_DIAG(MEMORY, "~NativeDynamicShapePlan: staging[%d] ptr=%p", i, (void*)placeholderStagingBuffers_[i]);
      if (placeholderStagingBuffers_[i] != nullptr) {
        auto* db = placeholderStagingBuffers_[i]->dataBuffer();
        // isValid() checks MAGIC_NUMBER + !closed in one call. Using only isClosed()
        // is insufficient: a GC-destroyed DataBuffer has MAGIC_DESTROYED magic, so
        // isClosed() reads garbage from freed memory and may return false, causing
        // deleteBuffers() on a stale pointer → heap corruption.
        bool dbSafe = (db != nullptr && db->isValid() && !db->isClosed());
        DSP_DIAG(MEMORY, "~NativeDynamicShapePlan: staging[%d] db=%p dbSafe=%d",
                 i, (void*)db, (int)dbSafe);
        // Pre-clean GPU memory AND null _shapeInfo before deleting the NDArray.
        // The NDArray destructor reads _shapeInfo to check isView — if _shapeInfo
        // is a dangling pointer (ConstantShapeHelper cache entry freed during an
        // earlier teardown phase), the destructor crashes. Nulling _shapeInfo makes
        // the destructor skip the isView check (defaults to false = non-view).
        // Pre-cleaning the DataBuffer ensures ~DataBuffer sees closed=true and skips
        // GPU free, avoiding double-free with the pool.
        if (dbSafe) {
          db->deleteBuffers();
        }
        placeholderStagingBuffers_[i]->setShapeInfo((sd::LongType*)nullptr);
        delete placeholderStagingBuffers_[i];
        placeholderStagingBuffers_[i] = nullptr;
        freedStaging++;
      }
    }
    delete[] placeholderStagingBuffers_;
    placeholderStagingBuffers_ = nullptr;
  }
  DSP_DIAG(MEMORY, "~NativeDynamicShapePlan: staging done, freeing effectiveExternals");
  delete[] effectiveExternals_;
  effectiveExternals_ = nullptr;
  cachedVariableExtIndices_.clear();

  // View producer flags are now stored in slots_[].slotPhase.isViewProducer — no separate array to free.

  // Free context pool
  DSP_DIAG(MEMORY, "~NativeDynamicShapePlan: freeing contextPool (%p, numSlots=%d)", (void*)contextPool_, numSlots_);
  if (contextPool_) {
    for (int i = 0; i < numSlots_; i++) {
      delete contextPool_[i];
    }
    delete[] contextPool_;
  }
  DSP_DIAG(MEMORY, "~NativeDynamicShapePlan: contextPool done");

  // Free owned legacy ops (created during deserialization for ops
  // not registered in OpRegistrator, like exp, log, abs, etc.)
  DSP_DIAG(MEMORY, "~NativeDynamicShapePlan: freeing %zu legacy ops", ownedLegacyOps_.size());
  for (auto* legacyOp : ownedLegacyOps_) {
    delete legacyOp;
  }
  DSP_DIAG(MEMORY, "~NativeDynamicShapePlan: legacy ops done");

  // Free untracked output cache
  DSP_DIAG(MEMORY, "~NativeDynamicShapePlan: untrackedCache=%p size=%d",
           (void*)untrackedOutputCache_, untrackedOutputCacheSize_);
  if (untrackedOutputCache_) {
    for (int i = 0; i < untrackedOutputCacheSize_; i++) {
      if (untrackedOutputCache_[i] != nullptr) {
        auto* udb = untrackedOutputCache_[i]->dataBuffer();
        // isValid() checks MAGIC_NUMBER + !closed in one call (same fix as outputSlots_
        // and stagingBuffers above — isClosed() alone reads garbage from freed memory).
        bool udbSafe = (udb != nullptr && udb->isValid() && !udb->isClosed());
        DSP_DIAG(MEMORY, "~NativeDynamicShapePlan: untracked[%d]=%p db=%p dbSafe=%d",
                 i, (void*)untrackedOutputCache_[i], (void*)udb, (int)udbSafe);
        if (udbSafe) {
          udb->deleteBuffers();
        }
        untrackedOutputCache_[i]->setShapeInfo((sd::LongType*)nullptr);
        delete untrackedOutputCache_[i];
      }
    }
    delete[] untrackedOutputCache_;
  }
  DSP_DIAG(MEMORY, "~NativeDynamicShapePlan: untracked cache done");

  // ── Deferred workspace free (CUDA only) ───────────────────────────────────
  // The capture workspace is freed HERE — after all plan-owned DataBuffers have
  // been destroyed — so that DataBuffer::deleteSpecial() can use
  // CudaMemoryPool::isInCaptureWorkspace() to skip invalid cudaFreeAsync calls
  // on workspace-interior pointers. Handled in platform-specific destructor code.
  platformFreeCaptureWorkspace();

  // Free control flow structures
  delete[] loopRegions_;
  delete[] slotIsDead_;

  // Free slot buffer ownership metadata
  delete[] slotOwnership_;

  // Clear protected weight buffer set so stale DataBuffer pointers don't
  // linger. These are external (caller-owned) — we never freed them, but
  // holding stale pointers after plan destruction is a hazard.
  protectedWeightBuffers_.clear();
  frozenProtectedRefBuffers_.clear();
  frozenOutputRefBuffers_.clear();

  // Free Phase 3/4 structures
  if (planDef_ != nullptr) {
    planDef_->release();
    planDef_ = nullptr;
  }
  delete execState_;
  execState_ = nullptr;

  // Free the structured execution trace ring buffer.
  // Do this last so trace recording is valid throughout destruction.
  delete trace_;
  trace_ = nullptr;

  DSP_DIAG(MEMORY, "~NativeDynamicShapePlan: DONE plan=%p", this);

  // Finalize diagnostics AFTER all cleanup so destructor logging is captured
  DspDiagnostics::getInstance().endPlanExecution();
  DspDiagnostics::getInstance().printPlanReport();
  DspDiagnostics::getInstance().flushJsonReport();
}

// ─── Deserialization from binary plan ─────────────────────────────────────────

static const uint32_t DSP_MAGIC = 0x44535031;  // "DSP1"
static const int32_t DSP_VERSION_MAX = 5;  // Max supported version

/**
 * Helper to read typed values from a byte stream.
 */
class BinaryReader {
 public:
  BinaryReader(const uint8_t* data, LongType size)
      : data_(data), size_(size), pos_(0) {}

  template <typename T>
  T read() {
    if (pos_ + sizeof(T) > static_cast<size_t>(size_)) {
      THROW_EXCEPTION("BinaryReader: read past end of buffer");
    }
    T val;
    std::memcpy(&val, data_ + pos_, sizeof(T));
    pos_ += sizeof(T);
    return val;
  }

  template <typename T>
  void readArray(T* dest, int count) {
    size_t bytes = count * sizeof(T);
    if (pos_ + bytes > static_cast<size_t>(size_)) {
      THROW_EXCEPTION("BinaryReader: readArray past end of buffer");
    }
    std::memcpy(dest, data_ + pos_, bytes);
    pos_ += bytes;
  }

  std::string readString() {
    size_t lenPos = pos_;
    int32_t len = read<int32_t>();
    if (len < 0 || pos_ + len > static_cast<size_t>(size_)) {
      DSP_THROW(COMPILE,
                "BinaryReader: invalid string length %d at pos %zu (bufSize=%zu, lenFieldPos=%zu)",
                static_cast<int>(len), pos_, static_cast<size_t>(size_), lenPos);
    }
    std::string s(reinterpret_cast<const char*>(data_ + pos_), len);
    pos_ += len;
    return s;
  }

  size_t remaining() const { return size_ - pos_; }

 private:
  const uint8_t* data_;
  LongType size_;
  size_t pos_;
};

NativeDynamicShapePlan* NativeDynamicShapePlan::fromSerializedPlan(
    const void* data, LongType size, GraphExecutionMode mode) {
  // Ensure op traits are populated before the fusion pass runs.
  // Without this, the fusion pass sees no OP_TRAIT_REDUCTION/NORMALIZATION traits
  // on the first deserialization (before NativePlanCompiler::compile is called),
  // leading to non-deterministic fusion results across plan instances.
  sd::ops::initOpTraits();

  BinaryReader reader(static_cast<const uint8_t*>(data), size);

  // Read header
  uint32_t magic = reader.read<uint32_t>();
  if (magic != DSP_MAGIC) {
    DSP_DIAG(COMPILE, "NativeDynamicShapePlan: invalid magic 0x%08x (expected 0x%08x)", magic, DSP_MAGIC);
    char errbuf[128];
    snprintf(errbuf, sizeof(errbuf),
             "DSP fromSerializedPlan: invalid magic 0x%08x (expected 0x%08x, planSize=%lld)",
             magic, DSP_MAGIC, (long long)size);
    THROW_EXCEPTION(errbuf);
  }

  int32_t version = reader.read<int32_t>();
  if (version < 1 || version > DSP_VERSION_MAX) {
    DSP_DIAG(COMPILE, "NativeDynamicShapePlan: unsupported version %d (expected 1-%d)", version, DSP_VERSION_MAX);
    char errbuf[128];
    snprintf(errbuf, sizeof(errbuf),
             "DSP fromSerializedPlan: unsupported version %d (accepted 1-%d, planSize=%lld)",
             version, DSP_VERSION_MAX, (long long)size);
    THROW_EXCEPTION(errbuf);
  }

  auto* plan = new NativeDynamicShapePlan();
  plan->numSlots_ = reader.read<int32_t>();
  plan->totalOutputSlots_ = reader.read<int32_t>();
  plan->dirtySlotGenerations_.resize(plan->totalOutputSlots_, 0);
  plan->numExternalInputs_ = reader.read<int32_t>();
  plan->numRequestedOutputs_ = reader.read<int32_t>();

  REQUIRE_TRUE(plan->numSlots_ > 0 && plan->numSlots_ < 100000, 0,
               "NativeDynamicShapePlan::fromSerializedPlan: invalid numSlots %d", plan->numSlots_);
  REQUIRE_TRUE(plan->totalOutputSlots_ >= plan->numSlots_ && plan->totalOutputSlots_ < 500000, 0,
               "NativeDynamicShapePlan::fromSerializedPlan: invalid totalOutputSlots %d (numSlots=%d)",
               plan->totalOutputSlots_, plan->numSlots_);
  REQUIRE_TRUE(plan->numExternalInputs_ >= 0 && plan->numExternalInputs_ < 100000, 0,
               "NativeDynamicShapePlan::fromSerializedPlan: invalid numExternalInputs %d", plan->numExternalInputs_);
  // numRequestedOutputs_ can exceed totalOutputSlots_ because requested outputs
  // may reference external inputs (constants/variables/placeholders) not produced by any slot.
  // Those entries have slotIdx = -1 in requestedOutputSlotIndices_ and are handled downstream.
  REQUIRE_TRUE(plan->numRequestedOutputs_ >= 0 && plan->numRequestedOutputs_ < 500000, 0,
               "NativeDynamicShapePlan::fromSerializedPlan: invalid numRequestedOutputs %d (totalOutputSlots=%d)",
               plan->numRequestedOutputs_, plan->totalOutputSlots_);

  // Allocate slots
  plan->slots_ = new NativeSlot[plan->numSlots_];

  // Read per-slot data
  for (int s = 0; s < plan->numSlots_; s++) {
    NativeSlot& slot = plan->slots_[s];
    slot.ident.opHash = reader.read<int64_t>();
    slot.ident.opName = reader.readString();
    slot.wiring.numInputs = reader.read<int32_t>();
    slot.wiring.numOutputs = reader.read<int32_t>();

    REQUIRE_TRUE(slot.wiring.numInputs >= 0 && slot.wiring.numInputs < 10000, 0,
                 "NativeDynamicShapePlan::fromSerializedPlan: slot %d has invalid numInputs %d", s, slot.wiring.numInputs);
    REQUIRE_TRUE(slot.wiring.numOutputs >= 0 && slot.wiring.numOutputs < 10000, 0,
                 "NativeDynamicShapePlan::fromSerializedPlan: slot %d has invalid numOutputs %d", s, slot.wiring.numOutputs);

    // Input wiring
    slot.wiring.inputSourceIndices = new int[slot.wiring.numInputs];
    reader.readArray(slot.wiring.inputSourceIndices, slot.wiring.numInputs);

    // Validate each inputSourceIndex is in valid range
    for (int i = 0; i < slot.wiring.numInputs; i++) {
      REQUIRE_TRUE(slot.wiring.inputSourceIndices[i] >= -(plan->numExternalInputs_ + 1) &&
                   slot.wiring.inputSourceIndices[i] < plan->totalOutputSlots_, 0,
                   "NativeDynamicShapePlan::fromSerializedPlan: slot %d inputSourceIndices[%d]=%d out of range [%d, %d)",
                   s, i, slot.wiring.inputSourceIndices[i], -(plan->numExternalInputs_ + 1), plan->totalOutputSlots_);
    }

    slot.wiring.inputSourceTypes = new int8_t[slot.wiring.numInputs];
    reader.readArray(slot.wiring.inputSourceTypes, slot.wiring.numInputs);

    // Output wiring
    slot.wiring.outputSlotIndices = new int[slot.wiring.numOutputs];
    reader.readArray(slot.wiring.outputSlotIndices, slot.wiring.numOutputs);

    // iArgs
    slot.args.numIArgs = reader.read<int32_t>();
    if (slot.args.numIArgs > 0) {
      slot.args.iArgs = new LongType[slot.args.numIArgs];
      reader.readArray(slot.args.iArgs, slot.args.numIArgs);
    }

    // tArgs
    slot.args.numTArgs = reader.read<int32_t>();
    if (slot.args.numTArgs > 0) {
      slot.args.tArgs = new double[slot.args.numTArgs];
      reader.readArray(slot.args.tArgs, slot.args.numTArgs);
    }

    // bArgs
    slot.args.numBArgs = reader.read<int32_t>();
    if (slot.args.numBArgs > 0) {
      slot.args.bArgs = new bool[slot.args.numBArgs];
      reader.readArray(slot.args.bArgs, slot.args.numBArgs);
    }

    // dArgs
    slot.args.numDArgs = reader.read<int32_t>();
    if (slot.args.numDArgs > 0) {
      slot.args.dArgs = new DataType[slot.args.numDArgs];
      // dArgs are serialized as int32
      for (int i = 0; i < slot.args.numDArgs; i++) {
        slot.args.dArgs[i] = static_cast<DataType>(reader.read<int32_t>());
      }
    }

    slot.args.numSArgs = 0;
    if (version >= 5) {
      slot.args.numSArgs = reader.read<int32_t>();
      if (slot.args.numSArgs > 0) {
        slot.args.sArgs = new std::string[slot.args.numSArgs];
        for (int i = 0; i < slot.args.numSArgs; i++) {
          slot.args.sArgs[i] = reader.readString();
        }
      }
    }

    // Flags — read serialized bytes for format compatibility (Java writes placeholders
    // for needsZeroedOutput and isDataDependent; only outputShapeDependsOnInputValues
    // carries a real value). Trait-derived flags come from opTraits_ set below.
    reader.read<uint8_t>();  // needsZeroedOutput placeholder (derived from opTraits_)
    reader.read<uint8_t>();  // isDataDependent placeholder (derived from opTraits_)
    slot.flags.outputShapeDependsOnInputValues = reader.read<uint8_t>() != 0;
    slot.flags.needsIntLongSync = reader.read<uint8_t>() != 0;
    slot.flags.isCustomOp = reader.read<uint8_t>() != 0;
    slot.targetDeviceId = reader.read<int32_t>();

    // V2: legacy op type and opNum for ops not registered as DeclarableOp
    slot.legacy.legacyOpType = 0;
    slot.legacy.legacyOpNum = -1;
    if (version >= 2) {
      slot.legacy.legacyOpType = reader.read<int32_t>();
      slot.legacy.legacyOpNum = reader.read<int32_t>();
    }

    // V3: control flow metadata
    slot.cf.controlFlowType = CF_NONE;
    slot.cf.loopBackTarget = -1;
    slot.cf.loopRegionIndex = -1;
    if (version >= 3) {
      slot.cf.controlFlowType = static_cast<ControlFlowType>(reader.read<uint8_t>());
      slot.cf.loopBackTarget = reader.read<int32_t>();
      slot.cf.loopRegionIndex = reader.read<int32_t>();
    }

    // Resolve op: First try name lookup in OpRegistrator. If found, use it — unless
    // it's a synonym clash (e.g., DECLARE_SYN(dot, matmul) shadows legacy reduce3 "dot").
    // Detect synonym clashes by comparing the returned op's actual name against slot name.
    slot.ident.op = sd::ops::OpRegistrator::getInstance().getOperation(slot.ident.opName);
    if (slot.ident.op && slot.legacy.legacyOpType > 0 && slot.legacy.legacyOpNum >= 0) {
      // Check for synonym clash: if the registered op's actual name differs from
      // our requested name, C++ has mapped our name to a different op via DECLARE_SYN.
      // In that case, discard the synonym match and use the legacy wrapper instead.
      auto* registeredName = slot.ident.op->getOpName();
      if (registeredName && slot.ident.opName != std::string(registeredName->c_str())) {
        sd_debug("NativeDynamicShapePlan: synonym clash for '%s' -> registered as '%s', using legacy wrapper\n",
                 slot.ident.opName.c_str(), registeredName->c_str());
        slot.ident.op = nullptr;  // clear synonym match, will fall through to legacy
      }
    }
    if (!slot.ident.op && slot.legacy.legacyOpType > 0 && slot.legacy.legacyOpNum >= 0) {
      // Create a legacy op wrapper for ops not in the OpRegistrator
      // (e.g., exp, log, abs, neg, sqrt, sin, cos, etc.)
      sd::ops::DeclarableOp* legacyOp = nullptr;
      switch (slot.legacy.legacyOpType) {
        case LEGACY_TRANSFORM_SAME:
          legacyOp = new sd::ops::LegacyTransformSameOp(slot.legacy.legacyOpNum);
          break;
        case LEGACY_TRANSFORM_STRICT:
          legacyOp = new sd::ops::LegacyTransformStrictOp(slot.legacy.legacyOpNum);
          break;
        case LEGACY_TRANSFORM_FLOAT:
          legacyOp = new sd::ops::LegacyTransformFloatOp(slot.legacy.legacyOpNum);
          break;
        case LEGACY_TRANSFORM_BOOL:
          legacyOp = new sd::ops::LegacyTransformBoolOp(slot.legacy.legacyOpNum);
          break;
        case LEGACY_SCALAR:
          legacyOp = new sd::ops::LegacyScalarOp(slot.legacy.legacyOpNum);
          break;
        case LEGACY_PAIRWISE_TRANSFORM:
          legacyOp = new sd::ops::LegacyPairwiseTransformOp(slot.legacy.legacyOpNum);
          break;
        case LEGACY_SCALAR_BOOL:
          legacyOp = new sd::ops::LegacyScalarBoolOp(slot.legacy.legacyOpNum);
          break;
        case LEGACY_REDUCE_FLOAT:
          legacyOp = new sd::ops::LegacyReduceFloatOp(slot.legacy.legacyOpNum);
          break;
        case LEGACY_REDUCE_SAME:
          legacyOp = new sd::ops::LegacyReduceSameOp(slot.legacy.legacyOpNum);
          break;
        case LEGACY_REDUCE_BOOL:
          legacyOp = new sd::ops::LegacyReduceBoolOp(slot.legacy.legacyOpNum);
          break;
        case LEGACY_REDUCE_LONG:
          legacyOp = new sd::ops::LegacyReduceLongOp(slot.legacy.legacyOpNum);
          break;
        case LEGACY_REDUCE3:
          legacyOp = new sd::ops::LegacyReduce3Op(slot.legacy.legacyOpNum);
          break;
        case LEGACY_STATS:
          legacyOp = new sd::ops::LegacyStatsOp(slot.legacy.legacyOpNum);
          break;
        case LEGACY_INDEX_REDUCE:
          legacyOp = new sd::ops::LegacyIndexReduceOp(slot.legacy.legacyOpNum);
          break;
        case LEGACY_BROADCAST:
          legacyOp = new sd::ops::LegacyBroadcastOp(slot.legacy.legacyOpNum);
          break;
        case LEGACY_BROADCAST_BOOL:
          legacyOp = new sd::ops::LegacyBroadcastBoolOp(slot.legacy.legacyOpNum);
          break;
        case LEGACY_RANDOM:
          legacyOp = new sd::ops::LegacyRandomOp(slot.legacy.legacyOpNum);
          break;
        case LEGACY_PAIRWISE_BOOL:
          legacyOp = new sd::ops::LegacyPairwiseTransformBoolOp(slot.legacy.legacyOpNum);
          break;
        default:
          DSP_DIAG(COMPILE, "unknown legacy op type %d for '%s'",
                    slot.legacy.legacyOpType, slot.ident.opName.c_str());
          break;
      }
      if (legacyOp) {
        plan->ownedLegacyOps_.push_back(legacyOp);
        slot.ident.op = legacyOp;
        sd_debug("NativeDynamicShapePlan: created legacy op type=%d num=%d for '%s'\n",
                 slot.legacy.legacyOpType, slot.legacy.legacyOpNum, slot.ident.opName.c_str());
      }
    }
    if (!slot.ident.op && slot.cf.controlFlowType != CF_NONE) {
      // Control flow ops dont need a DeclarableOp — dispatched by CF engine
      sd_debug("NativeDynamicShapePlan: CF op '%s' (type=%d) — no DeclarableOp needed\n",
               slot.ident.opName.c_str(), static_cast<int>(slot.cf.controlFlowType));
    } else if (!slot.ident.op) {
      DSP_DIAG(COMPILE, "NativeDynamicShapePlan: op not found for name '%s' (serialized hash: %lld, legacyType: %d, legacyNum: %d)",
                slot.ident.opName.c_str(), slot.ident.opHash, slot.legacy.legacyOpType, slot.legacy.legacyOpNum);
      char errbuf[512];
      snprintf(errbuf, sizeof(errbuf),
               "DSP fromSerializedPlan: op not found for name '%s' (slot %d/%d, "
               "serializedHash=%lld, legacyType=%d, legacyNum=%d). "
               "Ensure the op is compiled with SD_ALL_OPS and registered via CUSTOM_OP_IMPL.",
               slot.ident.opName.c_str(), s, plan->numSlots_,
               (long long)slot.ident.opHash,
               slot.legacy.legacyOpType, slot.legacy.legacyOpNum);
      delete plan;
      THROW_EXCEPTION(errbuf);
    }


    // Use the C++ hash for internal computations (shape key, etc.)
    slot.ident.opHash = sd::ops::HashHelper::getInstance().getLongHash(slot.ident.opName);
    // Set opTraits_ bitmask — single source of truth for trait-derived queries.
    // Query methods on NativeSlot (isDataDependent, isIdentityOp, isViewCapableOp,
    // isFullyWriting, needsZeroedOutput, aliasesInput) all derive from this mask.
    //
    // IMPORTANT: ALWAYS merge BOTH sources (op descriptor AND OpTraitTable).
    // The op descriptor (DECLARE_TYPES addTraits) encodes traits known at op
    // compile time, but OP_TRAIT_DYNAMIC_OUTPUT_SIZE and similar DSP-specific
    // traits are only in OpTraitTable — they are never added via addTraits() in
    // op .cpp files.  Using the table as a fallback-only (opTraits_==0 guard)
    // loses all table traits for ops that also have descriptor traits set.
    // Example: 1-arg where has DATA_DEPENDENT from descriptor → opTraits_ != 0
    // → fallback never fires → DYNAMIC_OUTPUT_SIZE never set → pre-pass runs
    // where with zero-init arrays → wrong [0,1] shape → rank<2 for reduce_sum.
    if (slot.ident.op != nullptr && slot.ident.op->getOpDescriptor() != nullptr) {
      slot.opTraits_ = slot.ident.op->getOpDescriptor()->getTraits();
    }
    // Always OR in the table traits regardless of descriptor (they are
    // complementary, not alternatives).
    if (!slot.ident.opName.empty()) {
      slot.opTraits_ |= sd::ops::getOpTraitsByName(slot.ident.opName);
    }
    // A ternary-elementwise op invoked with exactly 3 inputs (e.g. select cond?x:y) has a
    // fixed broadcast output shape — it is NOT data-dependent or dynamic-output-size. The
    // same op may be table-marked DATA_DEPENDENT|DYNAMIC_OUTPUT_SIZE for a lower-arity
    // variant (e.g. 1-input coordinate extraction / NonZero). Resolve by TRAIT + arity —
    // NEVER by hardcoded op name (trait handling must stay general across all such ops).
    if (slot.hasOpTrait(sd::ops::OP_TRAIT_TERNARY_ELEMENTWISE) && slot.wiring.numInputs == 3) {
      slot.clearOpTrait(sd::ops::OP_TRAIT_DATA_DEPENDENT);
      slot.clearOpTrait(sd::ops::OP_TRAIT_DYNAMIC_OUTPUT_SIZE);
    }

    // Set structural iArg count from table (consistent with NativePlanCompiler)
    slot.flags.structuralIArgCount = getStructuralIArgCount(normalizeOpName(slot.ident.opName));

    // Initialize fusion fields (will be set by FusionPass::applyFusions later)
    slot.disableInPlaceFusion();
    slot.fusedChain.isFusedChainHead = false;
    slot.fusedChain.fusedChainLength = 0;
    slot.fusedChain.isFusedChainTail = false;
    std::memset(slot.fusedChain.fusedChainOpCodes, 0, sizeof(slot.fusedChain.fusedChainOpCodes));
    std::memset(slot.fusedChain.fusedChainSlots, 0, sizeof(slot.fusedChain.fusedChainSlots));
    std::fill(std::begin(slot.fusedChain.fusedChainSecondaryInputSources), std::end(slot.fusedChain.fusedChainSecondaryInputSources), INT32_MIN);
  }

  // ═══════════════════════════════════════════════════════════════════════
  // Structural integrity validation — catch corruption BEFORE execution.
  // These checks are O(slots × max_outputs) and run once at deserialization.
  // ═══════════════════════════════════════════════════════════════════════
  {
    // Track which step produces each output slot (for ordering + uniqueness checks).
    std::vector<int> outputSlotProducer(plan->totalOutputSlots_, -1);
    for (int s = 0; s < plan->numSlots_; s++) {
      auto& slot = plan->slots_[s];
      for (int o = 0; o < slot.wiring.numOutputs; o++) {
        int si = slot.wiring.outputSlotIndices[o];
        if (si < 0 || si >= plan->totalOutputSlots_) continue;
        if (outputSlotProducer[si] != -1 && outputSlotProducer[si] != s) {
          DSP_DIAG(COMPILE,
                   "DSP VALIDATION ERROR: output slot %d produced by BOTH step %d (%s) "
                   "and step %d (%s) — duplicate output slot assignment",
                   si, outputSlotProducer[si],
                   plan->slots_[outputSlotProducer[si]].ident.opName.c_str(),
                   s, slot.ident.opName.c_str());
          REQUIRE_TRUE(false, 0,
                       "NativeDynamicShapePlan::fromSerializedPlan: output slot %d assigned "
                       "to multiple steps (%d and %d) — plan wiring is corrupt", si,
                       outputSlotProducer[si], s);
        }
        outputSlotProducer[si] = s;
      }
    }

    // Check for self-references and dependency ordering violations.
    int selfRefCount = 0;
    int forwardRefCount = 0;
    for (int s = 0; s < plan->numSlots_; s++) {
      auto& slot = plan->slots_[s];
      // Collect this step's output slots for self-reference detection
      for (int i = 0; i < slot.wiring.numInputs; i++) {
        int srcIdx = slot.wiring.inputSourceIndices[i];
        if (srcIdx < 0) continue;  // external input — always valid
        if (srcIdx >= plan->totalOutputSlots_) continue;  // range-checked above

        // Self-reference: input reads from a slot produced by THIS step
        bool isSelfRef = false;
        for (int o = 0; o < slot.wiring.numOutputs; o++) {
          if (slot.wiring.outputSlotIndices[o] == srcIdx) {
            isSelfRef = true;
            break;
          }
        }
        if (isSelfRef) {
          selfRefCount++;
          DSP_DIAG(COMPILE,
                   "DSP VALIDATION ERROR: step %d (%s) input[%d] srcIdx=%d is a "
                   "SELF-REFERENCE (reads its own output slot) — plan wiring corrupt",
                   s, slot.ident.opName.c_str(), i, srcIdx);
        }

        // Forward reference: input reads from a slot produced by a LATER step.
        // (Control flow ops with loop-back are exempt — they legitimately read
        // from later steps via NextIteration→Merge cycles.)
        int producerStep = outputSlotProducer[srcIdx];
        if (producerStep > s && slot.cf.controlFlowType == CF_NONE) {
          forwardRefCount++;
          DSP_DIAG(COMPILE,
                   "DSP VALIDATION WARNING: step %d (%s) input[%d] srcIdx=%d is produced "
                   "by LATER step %d (%s) — forward reference (may indicate wiring bug)",
                   s, slot.ident.opName.c_str(), i, srcIdx,
                   producerStep, plan->slots_[producerStep].ident.opName.c_str());
        }
      }
    }

    if (selfRefCount > 0) {
      REQUIRE_TRUE(false, 0,
                   "NativeDynamicShapePlan::fromSerializedPlan: %d self-referencing input(s) "
                   "detected — plan wiring is corrupt. An op cannot read its own output slot "
                   "as input. This typically indicates a bug in the Java DynamicShapePlanCompiler.",
                   selfRefCount);
    }

    if (forwardRefCount > 0) {
      DSP_DIAG(COMPILE,
               "NativeDynamicShapePlan: %d forward reference(s) in non-CF ops — "
               "execution ordering may cause NULL inputs", forwardRefCount);
    }
  }

  // Read release schedule
  plan->releaseAtStep_ = new int*[plan->numSlots_];
  plan->releaseAtStepCounts_ = new int[plan->numSlots_];
  for (int s = 0; s < plan->numSlots_; s++) {
    int count = reader.read<int32_t>();
    plan->releaseAtStepCounts_[s] = count;
    if (count > 0) {
      plan->releaseAtStep_[s] = new int[count];
      reader.readArray(plan->releaseAtStep_[s], count);
    } else {
      plan->releaseAtStep_[s] = nullptr;
    }
  }

  // V3: Read loop regions
  plan->loopRegions_ = nullptr;
  plan->numLoopRegions_ = 0;
  plan->hasControlFlow_ = false;
  plan->cfLoopBackStep_ = -1;
  if (version >= 3) {
    plan->numLoopRegions_ = reader.read<int32_t>();
    if (plan->numLoopRegions_ > 0) {
      plan->loopRegions_ = new LoopRegion[plan->numLoopRegions_];
      for (int i = 0; i < plan->numLoopRegions_; i++) {
        plan->loopRegions_[i].mergeSlot = reader.read<int32_t>();
        plan->loopRegions_[i].switchSlot = reader.read<int32_t>();
        plan->loopRegions_[i].nextIterSlot = reader.read<int32_t>();
        plan->loopRegions_[i].exitSlot = reader.read<int32_t>();
        plan->loopRegions_[i].bodyStartSlot = reader.read<int32_t>();
        plan->loopRegions_[i].bodyEndSlot = reader.read<int32_t>();
      }
    }
    // Check if any slot has control flow
    for (int s = 0; s < plan->numSlots_; s++) {
      if (plan->slots_[s].cf.controlFlowType != CF_NONE) {
        plan->hasControlFlow_ = true;
        break;
      }
    }
    if (plan->hasControlFlow_) {
      DSP_DIAG(COMPILE, "control flow detected (%d loop regions)",
               plan->numLoopRegions_);
    }
  }

  // Allocate dead-slot tracking for control flow
  plan->slotIsDeadSize_ = plan->totalOutputSlots_;
  plan->slotIsDead_ = new bool[plan->slotIsDeadSize_];
  std::memset(plan->slotIsDead_, 0, sizeof(bool) * plan->slotIsDeadSize_);

  // Read requested output slot indices
  plan->requestedOutputSlotIndices_ = new int[plan->numRequestedOutputs_];
  reader.readArray(plan->requestedOutputSlotIndices_, plan->numRequestedOutputs_);

  // Read external input names (v4+)
  if (version >= 4) {
    plan->externalInputNames_.resize(plan->numExternalInputs_);
    for (int i = 0; i < plan->numExternalInputs_; i++) {
      plan->externalInputNames_[i] = reader.readString();
    }
  }

  // Build external input classification by scanning slot input source types.
  // externalInputIsVariable_ is the replay/staging class: PLACEHOLDER inputs
  // need value refresh before CUDA graph replay. externalInputIsPlaceholder_
  // is the lifecycle class: placeholders are caller-owned feeds and must not
  // be tracked as protected model weights.
  plan->externalInputIsVariable_.resize(plan->numExternalInputs_, false);
  plan->externalInputIsPlaceholder_.resize(plan->numExternalInputs_, false);
  for (int s = 0; s < plan->numSlots_; s++) {
    auto& slot = plan->slots_[s];
    for (int i = 0; i < slot.wiring.numInputs; i++) {
      int srcIdx = slot.wiring.inputSourceIndices[i];
      if (srcIdx < 0) {
        int extIdx = -(srcIdx + 1);
        if (extIdx < plan->numExternalInputs_) {
          if (slot.wiring.inputSourceTypes[i] == SOURCE_PLACEHOLDER) {
            plan->externalInputIsVariable_[extIdx] = true;
            plan->externalInputIsPlaceholder_[extIdx] = true;
          }
          // NOTE: SOURCE_VARIABLE inputs (trainable weights) are NOT marked
          // variable here. During inference, weights are constants — they never
          // change between decode steps. Marking weights as variable prevents
          // detectFrozenConstants() from freezing any weight-dependent slot and
          // adds unnecessary D2D staging copies per step. For training, the
          // Java side calls markPlanExternalInputVariable() selectively.
          // See NativePlanCompiler.cpp lines 630-641 for the canonical rationale.
        }
      }
    }
  }

  // Compute transitive variable dependency for the frozen fast-path gate.
  plan->computeSlotVariableDependency();

  // Allocate execution state
  plan->outputSlots_ = new NDArray*[plan->totalOutputSlots_];
  std::memset(plan->outputSlots_, 0, sizeof(NDArray*) * plan->totalOutputSlots_);

  // outputSlots_ owns all slot arrays

  // slotIsViewProducer_ replaced by slots_[i].slotPhase.isViewProducer (value-initialized to false with slots_)

  // Allocate slot buffer ownership metadata (value-initialized to UNSET)
  plan->slotOwnership_ = new SlotBufferInfo[plan->totalOutputSlots_]();

  // Allocate untracked output cache (for outputs with outputSlotIndices[i] < 0).
  // These are temporary buffers needed by ops but not referenced downstream.
  // Cached here so they can be reused during GPU graph capture (where allocs fail).
  plan->untrackedOutputCacheSize_ = plan->numSlots_ * MAX_OUTPUTS_PER_SLOT;
  plan->untrackedOutputCache_ = new NDArray*[plan->untrackedOutputCacheSize_];
  std::memset(plan->untrackedOutputCache_, 0, sizeof(NDArray*) * plan->untrackedOutputCacheSize_);

  // Pre-allocate context pool
  plan->contextPool_ = new Context*[plan->numSlots_];
  for (int i = 0; i < plan->numSlots_; i++) {
    plan->contextPool_[i] = new Context(1);
  }

  // ── Shape static analysis: classify each slot as shape-static or shape-dynamic ──
  // A slot is shape-dynamic if it transitively depends on any placeholder input
  // or is data-dependent. Everything else is shape-static (constants/variables
  // never change shape between executions).
  // Slots are in topological order, so predecessors are already classified.
  {
    // Build reverse mapping: outputSlotIndex -> stepIndex (which slot produced it)
    std::vector<int> outputSlotToStepIndex(plan->totalOutputSlots_, -1);
    for (int s = 0; s < plan->numSlots_; s++) {
      NativeSlot& slot = plan->slots_[s];
      for (int i = 0; i < slot.wiring.numOutputs; i++) {
        int si = slot.wiring.outputSlotIndices[i];
        if (si >= 0 && si < plan->totalOutputSlots_) {
          outputSlotToStepIndex[si] = s;
        }
      }
    }

    int staticCount = 0, dynamicCount = 0;
    for (int s = 0; s < plan->numSlots_; s++) {
      NativeSlot& slot = plan->slots_[s];
      slot.shapeCache.shapeStatic = true;  // assume static

      // Value-dependent shape ops are dynamic (output shape depends on runtime values)
      if (slot.hasValueDependentShape()) {
        slot.shapeCache.shapeStatic = false;
        dynamicCount++;
        continue;
      }

      for (int i = 0; i < slot.wiring.numInputs; i++) {
        int srcIdx = slot.wiring.inputSourceIndices[i];
        if (srcIdx < 0) {
          // External input: placeholders are dynamic, constants/variables are static
          if (slot.wiring.inputSourceTypes[i] == SOURCE_PLACEHOLDER) {
            slot.shapeCache.shapeStatic = false;
            break;
          }
        } else {
          // From prior slot output — check if producer is dynamic
          if (srcIdx < plan->totalOutputSlots_) {
            int producerStep = outputSlotToStepIndex[srcIdx];
            if (producerStep >= 0 && !plan->slots_[producerStep].shapeCache.shapeStatic) {
              slot.shapeCache.shapeStatic = false;
              break;
            }
          }
        }
      }

      if (slot.shapeCache.shapeStatic) staticCount++;
      else dynamicCount++;
    }

    DSP_DIAG(SHAPE, "shape analysis: %d static, %d dynamic out of %d slots",
             staticCount, dynamicCount, plan->numSlots_);

    // Count identity ops for diagnostics
    int identityCount = 0;
    for (int i = 0; i < plan->numSlots_; i++) {
      if (plan->slots_[i].isIdentityOp()) identityCount++;
    }
    if (identityCount > 0) {
      DSP_DIAG(SHAPE, "%d identity ops (will use fast-path)", identityCount);
    }
  }

  // Set mode BEFORE buildSegments() so segments get the correct selectedBackend.
  // buildSegments() calls resolveBackendForSegment() which reads graphExecutionMode_.
  if (mode != GraphExecutionMode::GEM_AUTO) {
    plan->setGraphExecutionMode(mode);
  }

  // Build graph segments for GPU graph capture.
  plan->buildSegments();

  // Detect and apply fusion candidates
  if (plan->numSlots_ > 1) {
    auto fusions = FusionPass::detectFusions(plan->slots_, plan->numSlots_);
    if (!fusions.empty()) {
      DSP_DIAG(FUSION, "detected %d fusion candidates",
               static_cast<int>(fusions.size()));
      for (auto& f : fusions) {
        DSP_DIAG_SLOT(FUSION, f.startSlot, "fusion: slots %d-%d, type=%d, chain=%d",
                      f.startSlot, f.endSlot, static_cast<int>(f.type), f.chainLength);
      }

      int applied = FusionPass::applyFusions(plan->slots_, plan->numSlots_, fusions);
      DSP_DIAG(FUSION, "applied %d of %d fusion candidates (in-place execution)",
               applied, static_cast<int>(fusions.size()));

      // Post-fusion guard: disable in-place when the source slot is a requested output.
      // FusionPass doesn't know about requested outputs, so we must check here.
      if (plan->requestedOutputSlotIndices_ != nullptr && plan->numRequestedOutputs_ > 0) {
        std::unordered_set<int> reqOutSet;
        for (int ri = 0; ri < plan->numRequestedOutputs_; ri++) {
          int si = plan->requestedOutputSlotIndices_[ri];
          if (si >= 0) reqOutSet.insert(si);
        }
        int disabledForReqOutput = 0;
        for (int s = 0; s < plan->numSlots_; s++) {
          auto& sl = plan->slots_[s];
          int srcSlot = sl.inPlaceSourceSlot();
          if (srcSlot >= 0 && reqOutSet.count(srcSlot)) {
            sl.disableInPlaceFusion();
            disabledForReqOutput++;
          }
        }
        if (disabledForReqOutput > 0) {
          DSP_DIAG(FUSION, "post-fusion: disabled %d in-place ops (source is requested output)",
                   disabledForReqOutput);
        }
      }
    }
  }

  // ── Build shared immutable PlanDefinition ───────────────────────────────
  {
    auto builder = PlanDefinition::Builder();
    builder.setNumSlots(plan->numSlots_)
           .setTotalOutputSlots(plan->totalOutputSlots_)
           .setNumExternalInputs(plan->numExternalInputs_)
           .setNumRequestedOutputs(plan->numRequestedOutputs_)
           .setRequestedOutputSlotIndices(plan->requestedOutputSlotIndices_,
                                          plan->numRequestedOutputs_)
           .setExternalInputNames(plan->externalInputNames_)
           .setExternalInputIsVariable(plan->externalInputIsVariable_)
           .setHasControlFlow(plan->hasControlFlow_)
           .setNumLoopRegions(plan->numLoopRegions_)
           .setBackendPriority(plan->backendPriority_);
    plan->planDef_ = builder.build();
  }

  // ── Create per-instance ExecutionState ──────────────────────────────────
  plan->execState_ = new ExecutionState(plan->totalOutputSlots_);

  // Notify diagnostics that a plan was compiled
  DspDiagnostics::getInstance().beginPlanExecution(
      plan->numSlots_, static_cast<int>(plan->segments_.size()));

  // Compute plan identity fingerprint: FNV-1a hash over (numSlots, all opNames in order).
  // Logged at creation and on every cache hit to detect plan-swap mismatches where
  // different plans have the same cache key.
  {
    uint64_t fp = 14695981039346656037ULL;  // FNV offset basis
    auto mixByte = [&](uint8_t b) { fp ^= b; fp *= 1099511628211ULL; };
    auto mixInt = [&](int v) {
      for (int shift = 0; shift < 32; shift += 8) mixByte(static_cast<uint8_t>((v >> shift) & 0xFF));
    };
    mixInt(plan->numSlots_);
    for (int s = 0; s < plan->numSlots_; s++) {
      const auto& name = plan->slots_[s].ident.opName;
      for (char c : name) mixByte(static_cast<uint8_t>(c));
      mixByte(0);  // NUL separator
      // Also mix output slot indices to catch wiring differences
      for (int o = 0; o < plan->slots_[s].wiring.numOutputs; o++) {
        mixInt(plan->slots_[s].wiring.outputSlotIndices[o]);
      }
    }
    plan->identityFingerprint_ = fp;
    DSP_DIAG(COMPILE, "plan identity fingerprint: 0x%016llx (addr=%p slots=%d segs=%d)",
             (unsigned long long)fp, (void*)plan, plan->numSlots_,
             static_cast<int>(plan->segments_.size()));
  }

  DSP_DIAG(COMPILE, "plan compiled: %d slots, %d segments, planDef refCount=%d",
           plan->numSlots_, static_cast<int>(plan->segments_.size()),
           plan->planDef_ ? plan->planDef_->refCount() : -1);

  return plan;
}

// ─── Execution ──────────────────────────────────────────────────────────────

Status NativeDynamicShapePlan::execute(
    NDArray** externalInputs, int numExternalInputs,
    NDArray** requestedOutputs, int numRequestedOutputs,
    void* stream) {

  DSP_DIAG(EXECUTE, "NativeDynamicShapePlan::execute ENTER addr=%p fingerprint=0x%016llx "
           "slots=%d extIn=%d/%d execCount=%d frozen=%d",
           (void*)this, (unsigned long long)identityFingerprint_,
           numSlots_, numExternalInputs, numExternalInputs_,
           executeCount_, planLifecycle_.isShapesFrozen() ? 1 : 0);

  // Clear any sticky CUDA error that accumulated from a previous plan execution
  // (e.g., Triton compilation capture errors, ContextBuffers init errors, or
  // cross-plan errors from a shared thread). Without this, error 906/901 from
  // a previous capture attempt causes all cudaMemsetAsync / kernel launches in
  // THIS execution to fail with 901, even though this plan is NOT capturing.
  sd::graph::dspClearLastCudaError();

  // Additionally verify the default LaunchContext stream and the execution stream
  // aren't stuck in capture mode. During Triton compilation, beginCapture may capture
  // on the plan's execution stream; if capture is aborted but endCapture wasn't called,
  // all subsequent kernel launches on that stream fail with 901.
  {
    void* defaultStream = sd::graph::dspGetLcDefaultStream();
    if (defaultStream != nullptr) {
      sd::graph::dspEndStaleCapture(defaultStream, "default");
    }
  }
  if (stream != nullptr) {
    // `stream` is a STREAM-POINTER (cudaStream_t*); dspEndStaleCapture consumes a
    // STREAM-VALUE (see the convention block in DspCudaDispatch.h) — convert first, else
    // cudaStreamGetCaptureInfo() is handed a host address and SIGSEGVs inside libcuda.
    sd::graph::dspEndStaleCapture(sd::graph::dspStreamPtrToValue(stream), "execution");
  }
  sd::graph::dspEndStaleCapture(sd::graph::dspGetGraphCaptureStream(), "tl_graphCaptureStream");
  // Clear graph execution TLS state so that downstream slot-by-slot operations
  // do not mistakenly use the capture stream. The stale capture has already been
  // ended above, but the TLS flag persists and causes DataBuffer::setToZeroBuffers
  // to issue cudaMemsetAsync on pool-allocated memory with the capture stream,
  // producing error 901.
  if (tl_graphExecutionActive) {
    tl_graphExecutionActive = false;
    sd::graph::dspSetGraphCaptureStream(nullptr);
  }

  // ── Warmup serialization gate ──────────────────────────────────────────
  // Serialize execute() calls across threads while any plan on this device
  // is in warmup or capture phase. This prevents concurrent synchronous
  // CUDA calls from poisoning an active capture on another thread.
  //
  // Once a plan reaches REPLAYING (SEALED phase), it skips the mutex
  // entirely — replay uses only async APIs and is fully thread-safe.
  // This means the mutex is only held during the first ~7 steps per
  // thread (warmup + capture), then execution is lock-free.
  struct WarmupSerializationGuard {
    std::mutex* mtx;
    WarmupSerializationGuard(std::mutex* m) : mtx(m) { if (mtx) mtx->lock(); }
    ~WarmupSerializationGuard() { if (mtx) mtx->unlock(); }
    WarmupSerializationGuard(const WarmupSerializationGuard&) = delete;
    WarmupSerializationGuard& operator=(const WarmupSerializationGuard&) = delete;
  };
  int warmupDeviceIdx = sd::graph::dspGetCurrentDevice();
  if (warmupDeviceIdx < 0 || warmupDeviceIdx >= kMaxDevices) warmupDeviceIdx = 0;
  bool needsWarmupLock = !planLifecycle_.isReplaying();
  WarmupSerializationGuard warmupGuard(needsWarmupLock ? &g_warmupSerializationMtx[warmupDeviceIdx] : nullptr);

  if (numExternalInputs != numExternalInputs_) {
    DSP_DIAG(EXECUTE, "NativeDynamicShapePlan::execute: expected %d external inputs, got %d",
              numExternalInputs_, numExternalInputs);
    return Status::BAD_ARGUMENTS;
  }

  if (numRequestedOutputs != numRequestedOutputs_) {
    DSP_DIAG(EXECUTE, "NativeDynamicShapePlan::execute: expected %d requested outputs, got %d",
              numRequestedOutputs_, numRequestedOutputs);
    return Status::BAD_ARGUMENTS;
  }

  // Clear dirty bitmap.
  std::fill(dirtySlotGenerations_.begin(), dirtySlotGenerations_.end(), 0);

  // Store a persistent copy of external input pointers so they remain valid
  // after execute() returns. The NDArray* pointers themselves are Java-owned
  // and live beyond the call, but the array-of-pointers may be stack-allocated.
  lastExternalInputsCopy_.assign(externalInputs, externalInputs + numExternalInputs);
  lastExternalInputs_ = lastExternalInputsCopy_.data();
  lastNumExternalInputs_ = numExternalInputs;
  // Record the ext-input buffer addresses NOW, while the Java-owned NDArrays are guaranteed
  // live (the duration of this JNI call). The JNI address query (getLastExternalInputAddress)
  // reads these recorded values — it must NOT dereference the stored NDArray* later, because
  // callers pass fresh inputs each step and free the old ones (query-time deref = UAF:
  // garbage magic number → DataBuffer integrity-check throw). Raw db pointers only — no
  // specialBuffer() self-heal side effects at record time.
  lastExternalInputAddrs_.resize(numExternalInputs);
  for (int _e = 0; _e < numExternalInputs; _e++) {
    NDArray* _a = externalInputs[_e];
    void* _p = nullptr;
    if (_a != nullptr && _a->dataBuffer() != nullptr) {
      _p = _a->dataBuffer()->special();
      if (_p == nullptr) _p = _a->dataBuffer()->primary();
    }
    lastExternalInputAddrs_[_e] = reinterpret_cast<long long>(_p);
  }

  // Capture external input ranks on first call — used by FusionPass pass 5
  // to distinguish 1D bias vectors from N-D residual operands.
  if (externalInputRanks_.empty() && numExternalInputs > 0) {
    externalInputRanks_.resize(numExternalInputs, -1);
    for (int i = 0; i < numExternalInputs; i++) {
      if (externalInputs[i] != nullptr)
        externalInputRanks_[i] = externalInputs[i]->rankOf();
    }
  }

  // ── PlanExecutionContext: consolidates all per-execute() state ─────────
  // Created by platformBeginExecution (CUDA: stream guard + cross-stream sync,
  // CPU: minimal struct). Destroyed by platformEndExecution at end of execute().
  // Cast from void* to typed pointer — header keeps void* to avoid rebuild cascade.
  const bool frozenOrReplayAtEntry = planLifecycle_.isInFrozenOrReplayState();
  void* executionStatePtr = platformBeginExecution(stream, frozenOrReplayAtEntry, executeCount_);
  auto* execCtx = static_cast<PlanExecutionContext*>(executionStatePtr);
  activeExecCtx_ = executionStatePtr;  // Expose to _gpubackend.cpp methods
  // Per-exec reset: staging is only "current" if this exec's pre-replay sync
  // refreshes it (ensureAndSyncStagingBuffers sets it back to true). Direct-SBS
  // execs skip staging, so views must alias RAW externals, not stale staging.
  stagingMaintainedThisExec_ = false;

  // RAII guard to ensure platformEndExecution is called even if an exception
  // is thrown (e.g., from reportCaptureError or DSP_THROW_SEG). Without this,
  // a thrown exception leaks g_execCount — the next capture attempt on the same
  // device waits forever for the phantom execution to drain (deadlock).
  struct PlatformEndGuard {
    NativeDynamicShapePlan* plan;
    void*& statePtr;
    void* stream;
    bool frozen;
    int execCount;
    bool dismissed;
    PlatformEndGuard(NativeDynamicShapePlan* p, void*& sp, void* s, bool f, int e)
      : plan(p), statePtr(sp), stream(s), frozen(f), execCount(e), dismissed(false) {}
    ~PlatformEndGuard() {
      if (!dismissed && statePtr != nullptr) {
        plan->activeExecCtx_ = nullptr;
        plan->platformEndExecution(statePtr, stream, frozen, execCount);
        statePtr = nullptr;
      }
    }
    void dismiss() { dismissed = true; }
    PlatformEndGuard(const PlatformEndGuard&) = delete;
    PlatformEndGuard& operator=(const PlatformEndGuard&) = delete;
  };
  PlatformEndGuard platformEndGuard(this, executionStatePtr, stream, frozenOrReplayAtEntry, executeCount_);

  // When tritonSkipKernels is active, force the plan to behave exactly like
  // GEM_SLOT_BY_SLOT. The plan was compiled with GEM_TRITON (segments have
  // selectedBackend=GPU_COMPILER, gpuGraphCaptureEnabled_=true, etc.) but
  // all compiled kernels are skipped. Force GEM_SLOT_BY_SLOT so that:
  //   1. phaseCompile is skipped (no pointless Triton compilation)
  //   2. phaseSlotBySlot is used (no phaseReplay differences)
  //   3. AUTO_SEAL is skipped (no premature shape freezing)
  //   4. Segments behave as SLOT_BY_SLOT (no graph capture state changes)
  bool tritonSkip = Environment::getInstance().tritonSkipKernels();
  // Apply tritonSkip transiently via execCtx — do NOT mutate plan-level state.
  // Permanently overwriting graphExecutionMode_/gpuGraphCaptureEnabled_ corrupts
  // the plan if tritonSkipKernels is later cleared (no restore path existed).
  auto effectiveMode = graphExecutionMode_;
  auto effectiveGpuCapture = gpuGraphCaptureEnabled_;
  if (tritonSkip && effectiveMode != GraphExecutionMode::GEM_SLOT_BY_SLOT) {
    effectiveMode = GraphExecutionMode::GEM_SLOT_BY_SLOT;
    effectiveGpuCapture = false;
  }

  // Populate all derived state once — every method reads from the context,
  // not from scattered plan fields. This eliminates re-derivation of the same
  // conditions across execute(), platform methods, and segment dispatch.
  // Use isInFrozenOrReplayState() — not isShapesFrozen() — because the plan
  // may have already transitioned to REPLAYING (SEALED). isShapesFrozen() returns
  // false for REPLAYING, which would incorrectly route to phaseSlotBySlot.
  const bool frozenOrReplay = planLifecycle_.isInFrozenOrReplayState();
  execCtx->populateDerivedState(
      frozenOrReplay, executeCount_,
      static_cast<int>(effectiveMode),
      tritonSkip,
      Environment::getInstance().tritonGraphCapture(),
      Environment::getInstance().tritonVerifyKernels(),
      !externalInputIsVariable_.empty(),
      executionTimingEnabled_,
      anySegmentNeedsWarmup());
  execCtx->segmentsTotal = static_cast<int>(segments_.size());
  execCtx->recordFlow(PlanExecutionContext::FlowEventType::EXECUTE_ENTRY,
                       executeCount_, frozenOrReplay ? 1 : 0);

  // Begin DSP diagnostic step tracking (endDiag called at end of execute,
  // including early return paths via the context's lifecycle tracking).
  execCtx->beginDiag(executeCount_);

  DSP_DIAG(EXECUTE, "step %d: mode=%s frozen=%d segs=%d graphCapture=%d ext=%d "
           "steadyState=%d fullSync=%d graphReplay=%d varFilter=%d",
           executeCount_, execCtx->dispatchModeName(),
           static_cast<int>(planLifecycle_.isShapesFrozen()),
           static_cast<int>(segments_.size()),
           static_cast<int>(effectiveGpuCapture), numExternalInputs,
           execCtx->isFrozenSteadyState ? 1 : 0,
           execCtx->needsFullSync ? 1 : 0,
           execCtx->allowGraphCaptureReplay ? 1 : 0,
           execCtx->useVariableFilter ? 1 : 0);

  auto isProtectedExternalInput = [&](int extIdx) -> bool {
    if (extIdx < 0 || extIdx >= numExternalInputs) return false;
    bool isPlaceholder = extIdx < static_cast<int>(externalInputIsPlaceholder_.size()) &&
                         externalInputIsPlaceholder_[extIdx];
    return !isPlaceholder;
  };

  auto buildCurrentProtectedWeightBuffers = [&]() {
    std::unordered_set<DataBuffer*> current;
    current.reserve(static_cast<size_t>(numExternalInputs));
    for (int i = 0; i < numExternalInputs; i++) {
      NDArray* arr = externalInputs[i];
      if (arr == nullptr || !isProtectedExternalInput(i)) continue;
      DataBuffer* db = arr->dataBuffer();
      if (db != nullptr && !db->isClosed() && db->isValid()) {
        current.insert(db);
      }
    }
    return current;
  };

  auto refreshProtectedWeightBuffers = [&]() {
    std::unordered_set<DataBuffer*> current = buildCurrentProtectedWeightBuffers();

    bool changed = current.size() != protectedWeightBuffers_.size();
    if (!changed) {
      for (auto* db : current) {
        if (protectedWeightBuffers_.count(db) == 0) {
          changed = true;
          break;
        }
      }
    }

    if (!changed) return;

    int added = 0;
    int removed = 0;
    int changedExternalIndices = 0;
    for (auto* db : current) {
      if (protectedWeightBuffers_.count(db) == 0) added++;
    }
    for (auto* db : protectedWeightBuffers_) {
      if (current.count(db) == 0) removed++;
    }
    for (int i = 0; i < numExternalInputs; i++) {
      NDArray* arr = externalInputs[i];
      if (arr == nullptr || !isProtectedExternalInput(i)) continue;
      DataBuffer* db = arr->dataBuffer();
      if (db != nullptr && !db->isClosed() && protectedWeightBuffers_.count(db) == 0) {
        changedExternalIndices++;
      }
    }

    const bool establishedFrozenState =
        planLifecycle_.isInFrozenOrReplayState() && executeCount_ > 0;
    if (establishedFrozenState) {
      for (auto* db : protectedWeightBuffers_) {
        if (db != nullptr && current.count(db) == 0) {
          auto it = std::find(frozenProtectedRefBuffers_.begin(),
                              frozenProtectedRefBuffers_.end(), db);
          if (it != frozenProtectedRefBuffers_.end()) {
            // Liveness-gated: a replaced external may already have been
            // destroyed by Java (weight close between replays).
            if (untrackFrozenPin(db)) {
              db->removeFrozenRef();
            }
            frozenProtectedRefBuffers_.erase(it);
          } else {
            DSP_DIAG(MEMORY,
                     "PROTECTED_EXT_REFRESH: old protected db=%p had no tracked "
                     "frozen ref during removal",
                     (void*)db);
          }
        }
      }
      for (auto* db : current) {
        if (db != nullptr && protectedWeightBuffers_.count(db) == 0) {
          db->addFrozenRef();
          trackFrozenPin(db);
          frozenProtectedRefBuffers_.push_back(db);
        }
      }
    }

    protectedWeightBuffers_.swap(current);

    DSP_DIAG(MEMORY,
             "PROTECTED_EXT_REFRESH: protected=%d added=%d removed=%d "
             "changedExtIdx=%d establishedFrozen=%d phase=%s execCount=%d",
             static_cast<int>(protectedWeightBuffers_.size()), added, removed,
             changedExternalIndices, establishedFrozenState ? 1 : 0,
             planLifecycle_.displayName(), executeCount_);

    if (!establishedFrozenState || (added == 0 && removed == 0)) {
      return;
    }

    if (planLifecycle_.isReplaying()) {
      planLifecycle_.unseal();
    } else {
      planLifecycle_.recordPointersUnstable();
    }
    frozenSnapshot_.clear();
    frozenConstantDetectionDone_ = false;
    planLifecycle_.compilationDone = false;
    // Weight DataBuffers were rebound (new executor borrowed this cached plan):
    // the next ext-input H2D prepare must cover ALL inputs — the executeCount_>0
    // fast path would otherwise skip the new weight buffers (batch-only wrong
    // results). Consumed by performPreReplaySync's broad branch.
    markWeightRebindNeedsBroadSync("protected_external_rebind");

    int invalidatedSegments = 0;
    for (auto& seg : segments_) {
      SegmentLifecycle::invalidateSegmentCaptures(this, seg, "protected_external_rebind");
      invalidatedSegments++;
    }
    clearGpuBackendFailedCache();
    platformClearCastCache();

    execCtx->populateDerivedState(
        !planLifecycle_.isSlotBySlot(), executeCount_,
        static_cast<int>(effectiveMode),
        tritonSkip,
        Environment::getInstance().tritonGraphCapture(),
        Environment::getInstance().tritonVerifyKernels(),
        !externalInputIsVariable_.empty(),
        executionTimingEnabled_,
        anySegmentNeedsWarmup());
    execCtx->recordFlow(PlanExecutionContext::FlowEventType::DERIVED_STATE_REFRESH,
                         executeCount_, invalidatedSegments);

    DSP_DIAG(EXECUTE,
             "PROTECTED_EXT_REBIND: invalidated %d segment captures and cleared "
             "frozen snapshot for recapture (phase=%s execCount=%d)",
             invalidatedSegments, planLifecycle_.displayName(), executeCount_);
  };

  refreshProtectedWeightBuffers();

  // Lifecycle validation must only track protected weight/constant external
  // inputs — placeholders are supplied fresh per call and legitimately get a
  // new DataBuffer every execution. Always build the filtered view from the
  // refreshed protected set so rebound variables replace stale closed buffers.
  std::vector<NDArray*> lifecycleExternalInputs;
  NDArray** lifecycleExternalInputPtrs = externalInputs;
  auto refreshLifecycleExternalInputs = [&]() {
    if (numExternalInputs <= 0) return;
    lifecycleExternalInputs.assign(externalInputs, externalInputs + numExternalInputs);
    for (int i = 0; i < numExternalInputs; i++) {
      NDArray* arr = externalInputs[i];
      DataBuffer* db = arr != nullptr ? arr->dataBuffer() : nullptr;
      bool trackForLifecycle = db != nullptr && protectedWeightBuffers_.count(db) > 0;
      if (!trackForLifecycle) {
        lifecycleExternalInputs[i] = nullptr;
      }
    }
    lifecycleExternalInputPtrs = lifecycleExternalInputs.data();
  };
  refreshLifecycleExternalInputs();

  platformDumpExternalInputDiagnostics(externalInputs, numExternalInputs, executeCount_);

  // External inputs cross the Java/native boundary as already-bound NDArrays.
  // Java only binds wrappers; data readiness is owned here through the normal
  // prepare/register contract on the DSP execution stream.
  struct ExternalInputSpecialUseGuard {
    std::vector<NDArray*> reads;
    bool active;

    explicit ExternalInputSpecialUseGuard(std::vector<NDArray*>&& readList)
        : reads(std::move(readList)), active(!reads.empty()) {
      if (active) {
        NDArray::prepareSpecialUse({}, reads);
      }
    }

    ~ExternalInputSpecialUseGuard() {
      if (active) {
        NDArray::registerSpecialUse({}, reads);
      }
    }

    ExternalInputSpecialUseGuard(const ExternalInputSpecialUseGuard&) = delete;
    ExternalInputSpecialUseGuard& operator=(const ExternalInputSpecialUseGuard&) = delete;
  };

  auto buildExternalReadList = [&]() {
    std::vector<NDArray*> reads;
    reads.reserve(static_cast<size_t>(numExternalInputs));
    int skippedDevicePending = 0;
    int skippedAlreadyDeviceActual = 0;
    int skippedEmpty = 0;
    const bool hasVariableInputs = !cachedVariableExtIndices_.empty();
    // weightRebindBroadSyncPending: read-only peek — a cached-plan reuse rebound
    // the weight DataBuffers, so this prepare must be broad even in steady state.
    // Consumption (flag clear) is owned by performPreReplaySync's broad branch.
    const bool broadPrepare = !frozenOrReplayAtEntry || executeCount_ <= 1
                              || anySegmentNeedsWarmup() || hasVariableInputs
                              || weightRebindBroadSyncPending();

    for (int i = 0; i < numExternalInputs; i++) {
      NDArray* arr = externalInputs[i];
      if (arr == nullptr || arr->isEmpty() || arr->lengthOf() == 0) {
        skippedEmpty++;
        continue;
      }
      if (i < static_cast<int>(deviceWritePending_.size()) && deviceWritePending_[i]) {
        skippedDevicePending++;
        continue;
      }

      DataBuffer* db = arr->dataBuffer();
      if (db == nullptr || db->isClosed() || !db->isValid()) {
        continue;
      }

      if (broadPrepare || !db->isSpecialActual()) {
        reads.push_back(arr);
      } else {
        skippedAlreadyDeviceActual++;
      }
    }

    DSP_DIAG(STREAM_SYNC,
             "EXT_INPUT_PREPARE: prepared=%d skippedDevicePending=%d "
             "skippedDeviceActual=%d skippedEmpty=%d broad=%d execTarget=%s",
             static_cast<int>(reads.size()), skippedDevicePending,
             skippedAlreadyDeviceActual, skippedEmpty, broadPrepare ? 1 : 0,
             execCtx->execTargetName());
    return reads;
  };

  ExternalInputSpecialUseGuard externalInputUseGuard(buildExternalReadList());
  // On CPU there are no CUDA streams, so cross-stream sync is trivially done.
  // Mark it here so the phase machine reaches CROSS_STREAM_DONE before we
  // advance to EXT_INPUTS_DONE.
  if (!execCtx->isCrossStreamSynced()) {
    execCtx->markCrossStreamSynced();
  }
  execCtx->markExtInputsSynced();

  // Debug: dump external input at a configured index (ND4J_DSP_TRACE_EXT_INPUT)
  // useful for diagnosing forced-H2D-sync issues where device-authoritative buffers get overwritten
  {
    int traceExt = sd::graph::DspDiagnostics::getInstance().traceExtInput();
    if (traceExt >= 0 && traceExt < numExternalInputs && DSP_DIAG_ENABLED(VERIFY)) {
      NDArray* extArr = externalInputs[traceExt];
      if (extArr != nullptr) {
        DSP_DIAG(VERIFY, "EXT_INPUT_START: exec=%d extIdx=%d dtype=%d shape=[%lld] len=%lld "
                 "specialBuf=%p primaryBuf=%p dbPtr=%p pAct=%d sAct=%d",
                 executeCount_, traceExt, (int)extArr->dataType(),
                 (long long)(extArr->rankOf() > 0 ? extArr->sizeAt(0) : 0),
                 (long long)extArr->lengthOf(),
                 sd::graph::dspBuffer(extArr), extArr->buffer(),
                 static_cast<void*>(extArr->dataBuffer()),
                 extArr->dataBuffer() ? (extArr->dataBuffer()->isPrimaryActual() ? 1 : 0) : -1,
                 extArr->dataBuffer() ? (extArr->dataBuffer()->isSpecialActual() ? 1 : 0) : -1);
        platformDumpExtInputGpuValues(extArr, traceExt, executeCount_, stream);
      }
      // Check if the traced external input shares a buffer with any output slot in the cache
      if (extArr != nullptr && sd::graph::dspBuffer(extArr) != nullptr && outputSlots_ != nullptr) {
        void* extAddr = sd::graph::dspBuffer(extArr);
        int aliasCount = 0;
        for (int si = 0; si < totalOutputSlots_; si++) {
          if (outputSlots_[si] != nullptr && sd::graph::dspBuffer(outputSlots_[si]) == extAddr) {
            DSP_DIAG(VERIFY, "EXT_INPUT_ALIAS: extIdx=%d addr=%p == slotArrayCache[%d] (len=%lld)",
                     traceExt, extAddr, si, (long long)outputSlots_[si]->lengthOf());
            aliasCount++;
          }
        }
        if (aliasCount == 0) {
          DSP_DIAG(VERIFY, "EXT_INPUT_ALIAS: extIdx=%d addr=%p NO alias found in %d output slots",
                   traceExt, extAddr, totalOutputSlots_);
        }
      }
    }
  }

  // Frozen graph fast path: if shapes are frozen and a single captured GPU graph
  // covers the entire plan, skip all per-slot/per-segment abstractions.
  // Returns OK if fast path handled execution, MAYBE to fall through.
  auto fastPathResult = platformTryFrozenFastPath(
      externalInputs, numExternalInputs, requestedOutputs, numRequestedOutputs, stream);
  if (fastPathResult != Status::MAYBE) {
    if (fastPathResult == Status::OK && planLifecycle_.isInFrozenOrReplayState()) {
      // Keep phase accounting in sync with the normal execute() path. Without this,
      // a successful frozen fast-path return can bypass advancePlanPhase(), leaving
      // the plan stuck at SHAPES_FROZEN even while replay is already stable.
      if (planLifecycle_.isShapesFrozen()) {
        planLifecycle_.recordPostFreezeExec();
        advancePlanPhase();
      }
    }
    execCtx->endDiag(executeCount_);
    activeExecCtx_ = nullptr;
    flushDeferredSlotDeletes();
    platformEndGuard.dismiss();
    platformEndExecution(executionStatePtr, stream, planLifecycle_.isInFrozenOrReplayState(), executeCount_);
    return fastPathResult;
  }

  // ── Phase-aware lifecycle validation ─────────────────────────────────────
  // Hard errors (not logs) when buffer lifecycle is violated during frozen execution.
  // This catches: freed buffers, pointer drift, stale ownership, dangling views.
  // When freezeMergeSegments is active, merged segments contain value-dependent ops
  // (reshape, gather, broadcast_to) whose output DataBuffers are re-created on each
  // execution by initializeOutputs. This is correct behavior — shapes are frozen but
  // the allocation path creates fresh arrays. The lifecycle validation (designed for
  // the non-merged case where each segment's slots have stable buffers) incorrectly
  // rejects these buffer replacements as "stale ownership".
  if ((planLifecycle_.isShapesFrozen() || planLifecycle_.isReplaying()) && executeCount_ > 0) {
    // Refresh stale view wrappers BEFORE lifecycle validation.
    // View-producer slots (squeeze, reshape, expand_dims, permute) share their
    // input's DataBuffer. When the input array is replaced between calls (e.g.,
    // a new placeholder), the view wrapper becomes stale — its DataBuffer pointer
    // no longer matches slotOwnership_[].dataBuffer. Refreshing here ensures the
    // lifecycle validator sees consistent ownership for view slots.
    if (planLifecycle_.isInFrozenOrReplayState()) {
      if (sd::Environment::getInstance().isDebug()) {
        scanAllSlotsForCorruption(outputSlots_, totalOutputSlots_,
                                  "BEFORE_refreshStaleViewWrappers", executeCount_);
      }
      for (int segIdx = 0; segIdx < (int)segments_.size(); segIdx++) {
        auto& seg = segments_[segIdx];
        refreshStaleViewWrappersInSegment(seg, externalInputs, numExternalInputs);
      }
      if (sd::Environment::getInstance().isDebug()) {
        scanAllSlotsForCorruption(outputSlots_, totalOutputSlots_,
                                  "AFTER_refreshStaleViewWrappers", executeCount_);
      }
    }

    // Reconcile slotOwnership_ with actual outputSlots_ before validation.
    // Dynamic-shape slots and view ops replace output arrays via direct
    // outputSlots_[] assignment (not writeOutputSlot), leaving stale
    // DataBuffer pointers in slotOwnership_. Sync here so the validator
    // sees consistent state.
    if (slotOwnership_ != nullptr) {
      for (int i = 0; i < totalOutputSlots_; i++) {
        auto& info = slotOwnership_[i];
        if (info.ownership == BufferOwnership::UNSET) continue;
        NDArray* arr = outputSlots_[i];
        if (arr == nullptr) {
          info.dataBuffer = nullptr;
          continue;
        }
        DataBuffer* actualDb = arr->dataBuffer();
        if (actualDb != info.dataBuffer) {
          info.dataBuffer = actualDb;
        }
      }
    }

    char errMsg[512] = {};
    bool lifecycleOk = validateLifecycleForPhase(
        planLifecycle_.toLegacyCode(),
        slotOwnership_, totalOutputSlots_,
        outputSlots_,
        lifecycleExternalInputPtrs, numExternalInputs,
        protectedWeightBuffers_,
        frozenSnapshot_.valid ? &frozenSnapshot_ : nullptr,
        errMsg, sizeof(errMsg));
    if (!lifecycleOk) {
      execCtx->endDiag(executeCount_);
      activeExecCtx_ = nullptr;
      flushDeferredSlotDeletes();
      platformEndGuard.dismiss();
      platformEndExecution(executionStatePtr, stream, planLifecycle_.isInFrozenOrReplayState(), executeCount_);
      DSP_THROW(VERIFY, "LIFECYCLE_VALIDATION_FAILED: %s", errMsg);
    }

    if (frozenSnapshot_.valid) {
      frozenSnapshot_.detectStaleActualityTransitions(
          outputSlots_, totalOutputSlots_,
          lifecycleExternalInputPtrs, numExternalInputs,
          planLifecycle_.toLegacyCode());
    }

    // ── Buffer coloring validation (debug mode only) ────────────────────
    if (colorMap_.isApplied() && sd::env_isDebug()) {
      try {
        colorMap_.validate(outputSlots_, slotOwnership_, totalOutputSlots_);
      } catch (const std::exception& e) {
        DSP_DIAG(MEMORY, "COLORING_EJECTED reason=RUNTIME_DRIFT: %s", e.what());
        auto& pool = DspBufferPool::forCurrentDevice();
        colorMap_.eject(outputSlots_, slotOwnership_, planOwnedArrays_, pool);
        colorMap_.reset();
      }
    }
  }

  if (planLifecycle_.isReplaying()) {
    // In REPLAYING phase, every replay-eligible segment must still be in
    // backend-specific steady state. If any segment drops out, throw.
    for (size_t si = 0; si < segments_.size(); si++) {
      auto& seg = segments_[si];
      if (!segmentIsFullyReplayingForPlanPhase(seg)) {
        // demotePlanPhase throws — include segment details in the reason
        char reason[512];
        snprintf(reason, sizeof(reason),
                 "segment no longer satisfies replay steady state: seg[%d-%d] "
                 "backend=%d execPhase=%s segExecCount=%d handleReady=%d "
                 "compositeReady=%d argStable=%d execCount=%d",
                 seg.def.startSlot, seg.def.endSlot,
                 static_cast<int>(seg.def.selectedBackend),
                 seg.exec.displayPhaseName(),
                 seg.exec.executionCount,
                 seg.exec.replayHandle && seg.exec.replayHandle->isReady() ? 1 : 0,
                 segmentHasReadyCompositeHandles(seg) ? 1 : 0,
                 !seg.exec.needsArgRefresh() ? 1 : 0,
                 executeCount_);
        demotePlanPhase(PlanPhase::SHAPES_FROZEN, reason);
      }
    }
  }

  // Pre-execute setup: clear stale errors, manage attention workspace,
  // flush pending close, invalidate stale cached graphs.
  sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(0);
  sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage("");
  platformPreExecuteSetup(externalInputs, numExternalInputs, stream);

  // Step 1: Initialize output slots
  // When shapes are frozen (after warmup), pre-populate from outputSlots_ so
  // downstream ops can read inputs without each slot individually setting outputSlots_.
  // View-producer slots will be overwritten during execution.
  //
  // Non-capturable (and permanently capture-failed) segments execute slot-by-slot
  // across decode steps. Their shape-driving scalar tensors often keep the same
  // shape while values change (KV length growth), so cross-execution shape cache
  // reuse can become stale and cause later broadcast mismatches. Invalidate these
  // segment-local caches each execute; capturable graph-replay segments keep caches.

  // replayManagedSlots REMOVED: arrays persist (one array per slot), no protection needed.

  // ── Always-on pre-execution array validity scan ─────────────────────────
  // Catch closed/destroyed/corrupt DataBuffers BEFORE any op runs.
  // NOT gated behind diagnostics — these are cheap field reads and catching
  // invalid state here prevents mysterious crashes deep in kernel execution.
  if (slotOwnership_ != nullptr) {
    int closed = detectClosedBuffers(slotOwnership_, totalOutputSlots_,
                                     outputSlots_, externalInputs, numExternalInputs,
                                     protectedWeightBuffers_);
    if (closed > 0) {
      DSP_THROW(MEMORY,
               "LIFECYCLE_ERROR: %d closed DataBuffer(s) found at execute() entry "
               "(execCount=%d phase=%s) — use-after-free imminent. "
               "Enable DSP_DIAG_MEMORY for per-slot details.",
               closed, executeCount_, planLifecycle_.displayName());
    }
  }

  // Validate ALL external inputs — these cross the JNI boundary and can go
  // stale between Java and C++ execution (GC, session cleanup, etc.).
  {
    char extErr[512] = {};
    int extInvalid = 0;
    for (int i = 0; i < numExternalInputs; i++) {
      if (externalInputs[i] == nullptr) continue;
      ArrayInvalidReason reason = validateArrayForExecution(externalInputs[i]);
      if (reason != ArrayInvalidReason::VALID) {
        extInvalid++;
        if (extInvalid == 1) {
          DataBuffer* db = (reason != ArrayInvalidReason::NULL_ARRAY &&
                            reason != ArrayInvalidReason::NULL_SHAPE_INFO)
              ? externalInputs[i]->dataBuffer() : nullptr;
          snprintf(extErr, sizeof(extErr),
                   "ARRAY_INVALID external input %d: reason=%s DataBuffer=%p "
                   "exec=%d phase=%s",
                   i, arrayInvalidReasonName(reason), (void*)db,
                   executeCount_, planLifecycle_.displayName());
        }
        DSP_DIAG(MEMORY,
                 "ARRAY_INVALID_EXT_INPUT: idx=%d reason=%s exec=%d phase=%s",
                 i, arrayInvalidReasonName(reason),
                 executeCount_, planLifecycle_.displayName());
      }
    }
    if (extInvalid > 0) {
      DSP_THROW(MEMORY,
               "LIFECYCLE_ERROR: %d invalid external input(s) at execute() entry: %s",
               extInvalid, extErr);
    }
  }

  // External inputs may have had device actuality updated by prepare/register.
  // Rebuild the filtered lifecycle view from the already-refreshed protected set.
  refreshLifecycleExternalInputs();

  // Arrays allocated on first execution, reused for all subsequent executions.
  // View ops create lightweight wrappers deleted inline when replaced.
  DSP_DIAG(MEMORY, "execute: arrays persist (exec=%d, frozen=%d, slots=%d)",
           executeCount_, planLifecycle_.isShapesFrozen() ? 1 : 0, totalOutputSlots_);

  // Non-frozen first execution only: reset segment state for warmup.
  // compilationFailed is managed by lifecycle (markFailed/reset) — not set here.
  if (executeCount_ == 0 && planLifecycle_.isSlotBySlot()) {
    for (auto& segment : segments_) {
      segment.exec.resetForWarmup();
      if (segment.exec.replayHandle) {
        platformCleanupSegmentForRebuild(segment);
      }
    }
  }

  if (planLifecycle_.isSlotBySlot()) {
    // Reset cast-cache INDEX only — do NOT call clearCastCache() here.
    //
    // clearCastCache() deletes the cached FP32-upcast NDArray objects (tl_castB[0],
    // tl_castA[0] etc.) from the thread-local cast cache. This is unsafe when another
    // plan's CUDA graph is live on the same thread: that graph has cuBLAS kernel nodes
    // with device pointers BAKED AT CAPTURE TIME pointing to these same cast buffers.
    // Deleting the buffers leaves the baked pointers dangling → cuBLAS reads freed
    // GPU memory → NaN on the very first replay call (frozen multi-plan switch return).
    //
    // Concrete case: {{seqLen=1, 5 steps}, {seqLen=16, 3 steps}} —
    //   Phase 1: seqLen=16 plan warms up, castWithPersistentCache creates cache[0]
    //            (FP32 upcast of HALF proj_weight), CUDA graph captures cache[0]->specialBuffer().
    //   {1,5}: seqLen=1 runs SBS. Old clearCastCache() deleted cache[0]. GPU buffer freed.
    //   {16,3}: seqLen=16 CUDA graph replays — cuBLAS reads freed cache[0] ptr → NaN.
    //
    // resetCastCacheIndices() resets the index to 0 so castWithPersistentCache starts
    // fresh each SBS execution, while keeping the underlying NDArray buffers alive so
    // any sibling plan's captured CUDA graph pointers remain valid.
    // The same safe pattern is used at phaseFreeze/phaseWarmup boundaries (lines 4020, 4102).
    MmulHelper::resetCastCacheIndices();
  }

  // Reset dead-slot flags once per plan execution (not per segment).
  // Dead flags from Switch in one segment must persist to affect ops in later segments.
  if (hasControlFlow_ && slotIsDead_ != nullptr) {
    std::memset(slotIsDead_, 0, sizeof(bool) * slotIsDeadSize_);
  }

  // Timing instrumentation — record start via execution context
  execCtx->t0 = execCtx->now();
  PhaseExecutionStats phaseStats;

  // ── Automatic shape pre-pass ──────────────────────────────────────────
  // On the first execution, run a lean shape-inference-only pass to
  // pre-populate all slot shape caches and allocate output arrays BEFORE
  // any op kernels execute. This eliminates calculateOutputShape calls
  // from the hot execution path — the subsequent slot-by-slot execution
  // will hit the shape cache for every slot and skip shape inference.
  //
  // Skip when: shapes are already frozen (pre-pass is pointless), or when
  // the explicit SHAPE_INFERENCE_ONLY mode is active (it handles its own
  // return path below), or when the pre-pass already ran for this plan.
  auto modeContract = ModeContract::forMode(graphExecutionMode_);
  if (!shapePrePassDone_ && planLifecycle_.isSlotBySlot() && !modeContract.isShapeInferenceOnly) {
    DSP_DIAG(SHAPE, "AUTO_SHAPE_PREPASS: running shape pre-pass before first execution "
             "(slots=%d extInputs=%d)", numSlots_, numExternalInputs);
    Status prePassStatus = phaseShapeInferenceOnly(externalInputs, numExternalInputs, stream);
    shapePrePassDone_ = true;
    if (prePassStatus != Status::OK) {
      // Shape pre-pass failure is non-fatal — log and fall through to normal
      // execution which will re-derive shapes as needed.
      DSP_DIAG(SHAPE, "AUTO_SHAPE_PREPASS: failed (status=%d), falling through to normal execution",
               static_cast<int>(prePassStatus));
    } else {
      DSP_DIAG(SHAPE, "AUTO_SHAPE_PREPASS: completed successfully, shape caches populated");
    }
  }

  // ── SHAPE_INFERENCE_ONLY: lean shape-only path ──────────────────────────
  // When mode is GEM_SHAPE_INFERENCE_ONLY, skip compilation, graph capture,
  // phase advancement, and all post-execution lifecycle management. Just
  // propagate shapes through the graph and return output arrays with correct
  // shapes (buffers allocated but not computed).
  if (modeContract.isShapeInferenceOnly) {
    DSP_DIAG(SHAPE, "SHAPE_INFERENCE_ONLY: entering shape-only path (slots=%d)", numSlots_);
    execCtx->recordFlow(PlanExecutionContext::FlowEventType::PHASE_DISPATCH,
                         static_cast<int>(GraphExecutionMode::GEM_SHAPE_INFERENCE_ONLY));
    Status siStatus = phaseShapeInferenceOnly(externalInputs, numExternalInputs, stream);
    if (siStatus != Status::OK) {
      DSP_DIAG(SHAPE, "SHAPE_INFERENCE_ONLY: phase failed status=%d", static_cast<int>(siStatus));
      execCtx->endDiag(executeCount_);
      activeExecCtx_ = nullptr;
      flushDeferredSlotDeletes();
      platformEndGuard.dismiss();
      platformEndExecution(executionStatePtr, stream, planLifecycle_.isInFrozenOrReplayState(), executeCount_);
      return siStatus;
    }
    // Extract outputs — shape-only arrays are in outputSlots_
    for (int i = 0; i < numRequestedOutputs_; i++) {
      int slotIdx = requestedOutputSlotIndices_[i];
      if (slotIdx >= 0 && slotIdx < totalOutputSlots_ && outputSlots_[slotIdx] != nullptr) {
        requestedOutputs[i] = outputSlots_[slotIdx];
      }
    }
    DSP_DIAG(SHAPE, "SHAPE_INFERENCE_ONLY: done, %d outputs populated", numRequestedOutputs_);
    execCtx->endDiag(executeCount_);
    activeExecCtx_ = nullptr;
    flushDeferredSlotDeletes();
    platformEndGuard.dismiss();
    platformEndExecution(executionStatePtr, stream, planLifecycle_.isInFrozenOrReplayState(), executeCount_);
    return Status::OK;
  }

  // Step 1b: Parallel precompilation of all GPU-compilable segments.
  // Skip the first frozen warmup because shapes are still being populated.
  // phaseCompile() itself checks planLifecycle_.compilationDone and returns immediately if already done.
  if (!planLifecycle_.compilationDone && !execCtx->isFirstFrozenWarmup &&
      !execCtx->forcedSlotBySlot && !ModeContract::forMode(execCtx->graphExecutionMode).isSlotBySlot) {
    phaseCompile(externalInputs, numExternalInputs);
  }

  // Pre-dispatch corruption scan — if corruption exists before any segment
  // runs, it was introduced by external input sync, shape pre-pass, or
  // refreshStaleViewWrappers.
  if (sd::Environment::getInstance().isDebug()) {
    scanAllSlotsForCorruption(outputSlots_, totalOutputSlots_,
        "BEFORE_PHASE_DISPATCH", executeCount_);
  }

  // Phase dispatch — resolved ONCE by PlanExecutionContext::resolveExecPhase(), the
  // single source of truth shared with dispatchModeName(). No scattered booleans or
  // ternaries here. WARMUP persists for the whole capture window (isFirstFrozenWarmup ==
  // shapesFrozen && anySegmentNeedsWarmup()), REPLAY once every segment has captured,
  // SLOT_BY_SLOT otherwise (forced/explicit slot-by-slot mode, or not yet frozen).
  Status phaseStatus = Status::OK;
  const PlanExecutionContext::ExecPhase execPhase = execCtx->resolveExecPhase();
  execCtx->recordFlow(PlanExecutionContext::FlowEventType::PHASE_DISPATCH,
                       execCtx->graphExecutionMode);
  DSP_DIAG(EXECUTE,
           "PHASE_RESOLVE: plan=%p %s (mode=%d inWarmup=%d replay=%d execCount=%d "
           "anySegWarmup=%d frozen=%d)",
           (void*)this, PlanExecutionContext::execPhaseName(execPhase),
           execCtx->graphExecutionMode, (int)execCtx->isFirstFrozenWarmup,
           (int)execCtx->isReplay, executeCount_, (int)anySegmentNeedsWarmup(),
           (int)planLifecycle_.isShapesFrozen());
  switch (execPhase) {
    case PlanExecutionContext::ExecPhase::WARMUP:
      phaseStatus = phaseWarmup(externalInputs, numExternalInputs, stream, &phaseStats);
      break;
    case PlanExecutionContext::ExecPhase::REPLAY:
      phaseStatus = phaseReplay(externalInputs, numExternalInputs, stream, &phaseStats);
      break;
    case PlanExecutionContext::ExecPhase::SLOT_BY_SLOT:
      phaseStatus = phaseSlotBySlot(externalInputs, numExternalInputs, stream, &phaseStats);
      break;
  }
  DSP_DIAG(EXECUTE, "PHASE_DISPATCH: phase returned status=%d", static_cast<int>(phaseStatus));
  if (phaseStatus != Status::OK) {
    // Structured trace: record the error and dump the last 128 events for diagnostics.
    DSP_TRACE_ERROR(trace_, -1, -1,
                    static_cast<uint32_t>(executeCount_),
                    static_cast<uint64_t>(phaseStatus));
    dumpTrace(stderr, 128);
    execCtx->endDiag(executeCount_);
    activeExecCtx_ = nullptr;
    flushDeferredSlotDeletes();
    platformEndGuard.dismiss();
    platformEndExecution(executionStatePtr, stream, planLifecycle_.isInFrozenOrReplayState(), executeCount_);
    return phaseStatus;
  }

  // ── Native KV scatter post-execution ─────────────────────────────────────
  // When the plan manages KV cache updates natively (configureKvScatter was called),
  // run the batched scatter here — after all segment execution, before outputs are
  // collected. This eliminates the Java-side scatterNewEntries() round-trip and
  // makes the scatter part of the same CUDA stream as the main graph execution.
  if (kvScatterConfigured_) {
    executeKvScatterPostExec(stream);
  }

  execCtx->tSegsDone = execCtx->now();

  platformPostSegmentPoolManagement(execCtx->frozen, execCtx->execCount);

  // ── Consistency assertions: verify slot reuse and replay integrity ───
  // These checks run after every execution to catch lifecycle bugs early.
  if (execCtx->diagVerifyEnabled) {
    int nullSlots = 0, liveSlots = 0, viewSlots = 0;
    int replaySegs = 0, slotBySlotSegsCount = 0, compilationFailedSegs = 0;
    for (int i = 0; i < totalOutputSlots_; i++) {
      if (outputSlots_[i] == nullptr) { nullSlots++; }
      else {
        liveSlots++;
        auto* db = outputSlots_[i]->dataBuffer();
        if (db != nullptr && protectedWeightBuffers_.count(db) > 0) viewSlots++;
      }
    }
    for (const auto& seg : segments_) {
      if ((seg.exec.replayHandle && seg.exec.replayHandle->isReady()) ||
          segmentHasReadyCompositeHandles(seg)) replaySegs++;
      else if (seg.exec.compilationFailed) compilationFailedSegs++;
      else slotBySlotSegsCount++;
    }
    DSP_DIAG(VERIFY, "POST_EXEC exec=%d frozen=%d: slots(live=%d null=%d weightView=%d/%d) "
             "segs(replay=%d sbs=%d capFail=%d/%d) graphReplays=%d slotBySlot=%d",
             executeCount_, planLifecycle_.isShapesFrozen() ? 1 : 0,
             liveSlots, nullSlots, viewSlots, totalOutputSlots_,
             replaySegs, slotBySlotSegsCount, compilationFailedSegs, (int)segments_.size(),
             phaseStats.graphReplaySegs, phaseStats.slotBySlotSegs);
  }

  // Plan-output boundary: materialize any VIEW that lives in a requested-output slot.
  // A view shares its DataBuffer with its parent slot inside the plan. On the NEXT
  // execute() call, refreshStaleViewWrappersInSegment can demote or replace the parent,
  // invalidating the view's DataBuffer from Java's perspective (reads as zeros).
  // Materializing produces an independent copy with its own DataBuffer that survives
  // plan re-execution. Performance: only fires for slots where isView()==true (rare —
  // contiguous materialized outputs and frozen-constant slots are never views).
  for (int i = 0; i < numRequestedOutputs_; i++) {
    int slotIdx = requestedOutputSlotIndices_[i];
    if (slotIdx >= 0 && slotIdx < totalOutputSlots_) {
      NDArray* slotArr = outputSlots_[slotIdx];
      if (slotArr != nullptr && slotArr->isView()) {
        materializeViewSlot(slotIdx, "plan-output-view-boundary");
      }
    }
  }

  for (int i = 0; i < numRequestedOutputs_; i++) {
    int slotIdx = requestedOutputSlotIndices_[i];
    if (slotIdx >= 0 && slotIdx < totalOutputSlots_) {
      NDArray* outArr = outputSlots_[slotIdx];
      // Lifecycle validation: catch corrupted output slots before returning to Java
      if (outArr != nullptr) {
        uintptr_t addr = reinterpret_cast<uintptr_t>(outArr);
        if (addr < 0x10000) {
          char msg[512];
          snprintf(msg, sizeof(msg),
                   "DSP LIFECYCLE ERROR: requestedOutput[%d] from outputSlots_[%d] "
                   "is a stale/freed pointer %p. execCount=%d planPhase=%s",
                   i, slotIdx, (void*)outArr, executeCount_, planLifecycle_.displayName());
          THROW_EXCEPTION(msg);
        }
        // Validate the array's shape info is not corrupted
        if (!outArr->hasValidShapeInfo()) {
          char msg[512];
          snprintf(msg, sizeof(msg),
                   "DSP LIFECYCLE ERROR: requestedOutput[%d] from outputSlots_[%d] "
                   "has invalid shapeInfo (use-after-free or corruption). "
                   "NDArray=%p execCount=%d planPhase=%s",
                   i, slotIdx, (void*)outArr, executeCount_, planLifecycle_.displayName());
          THROW_EXCEPTION(msg);
        }
      }
      // Multi-GPU shard: if this output was produced on a secondary device, migrate
      // it asynchronously to device-0 before returning to Java.  The plan keeps the
      // original device-N buffer in outputSlots_[slotIdx] for the next execution step;
      // only the copy returned here is handed to Java (Java owns it).
      // On CPU or single-GPU this is a no-op returning outArr unchanged.
      requestedOutputs[i] = platformGetOutputForDevice0(outArr, slotIdx, i);
    } else {
      requestedOutputs[i] = nullptr;
    }
  }

  // Diagnostic: dump requested output slot info and argmax for logits comparison.
  // Uses execCtx->diagVerifyEnabled (precomputed at entry) instead of re-checking
  // DSP_DIAG_ENABLED(VERIFY) && withinExecLimit() at every diagnostic site.
  if (execCtx->diagVerifyEnabled) {
    for (int i = 0; i < numRequestedOutputs_; i++) {
      int slotIdx = requestedOutputSlotIndices_[i];
      if (requestedOutputs[i] != nullptr) {
        auto* arr = requestedOutputs[i];
        DSP_DIAG_SLOT(VERIFY, slotIdx,
            "reqOut[%d] len=%lld dt=%d rank=%d",
            i, (long long)arr->lengthOf(), (int)arr->dataType(), arr->rankOf());
        // Only inspect an existing host mirror; VERIFY diagnostics must not
        // materialize primary storage on frozen replay buffers.
        if (arr->dataType() == FLOAT32 && arr->lengthOf() > 0) {
          auto* db = arr->dataBuffer();
          auto* primary = db != nullptr ? db->primary() : nullptr;
          if (primary != nullptr) {
            auto* buf = reinterpret_cast<float*>(primary);
            auto len = arr->lengthOf();
            float vMin = buf[0];
            for (sd::LongType vi = 1; vi < len; vi++) { if (buf[vi] < vMin) vMin = buf[vi]; }
            DSP_DIAG_SLOT(VERIFY, slotIdx,
                "reqOut[%d] VALUES min=%.6f first4=[%.4f,%.4f,%.4f,%.4f]",
                i, vMin,
                len > 0 ? buf[0] : 0.f, len > 1 ? buf[1] : 0.f,
                len > 2 ? buf[2] : 0.f, len > 3 ? buf[3] : 0.f);
          } else {
            DSP_DIAG_SLOT(VERIFY, slotIdx,
                "reqOut[%d] VALUES skipped (no host primary; special=%p frozen=%d)",
                i,
                db != nullptr ? db->special() : nullptr,
                db != nullptr && db->isFrozenPlanRegistered() ? 1 : 0);
          }
        }
        platformDumpLogitsArgmax(executeCount_, stream);
      } else {
        DSP_DIAG_SLOT(VERIFY, slotIdx, "reqOut[%d] nullptr", i);
      }
    }
  }

  // Log execution summary with segment dispatch breakdown and sync state
  execCtx->logExecutionSummary(executeCount_);

  execCtx->tOutputsDone = execCtx->now();

  // Step 4: No flush needed — arrays persist (one array per slot)
  execCtx->tFlushDone = execCtx->now();

  // Log plan-owned array count and total GPU allocation
  DSP_DIAG(MEMORY, "DSP_EXEC_END execCount=%d: planOwnedArrays=%d totalSlots=%d",
            executeCount_, (int)planOwnedArrays_.size(), totalOutputSlots_);

  // Track execution count for shapes-frozen optimization.
  //
  // Assertion 5: Execution count monotonicity.
  // executeCount_ must only increase during active execution. A value that is
  // not strictly greater than the pre-increment value indicates corruption —
  // either a race (two threads executing the same plan), an integer overflow,
  // or a bug that resets the count mid-execution without going through the
  // intentional reset path (releaseGpuIntermediates teardown path).
  //
  // NOTE: intentional resets (executeCount_ = 0) happen in
  // releaseGpuIntermediates() (teardown) BEFORE execute() is called, so this
  // assertion NEVER fires at that site. It only fires if the count goes backward
  // DURING a call to execute(), which is always a bug.
  if (planLifecycle_.isShapesFrozen() || planLifecycle_.isReplaying()) {
    int prevCount = executeCount_;
    incrementExecuteCount("execute");
    // Post-increment sanity: new value must be exactly prevCount+1.
    // If not, something corrupted the field during this execute() call.
    if (executeCount_ != prevCount + 1) {
      DSP_DIAG(EXECUTE,
               "EXEC_COUNT_MONOTONICITY_VIOLATION: expected %d got %d "
               "(shapesFrozen=%d planPhase=%s) — possible concurrent execution or "
               "mid-execute reset",
               prevCount + 1, executeCount_,
               planLifecycle_.isShapesFrozen() ? 1 : 0, planLifecycle_.displayName());
      REQUIRE_TRUE(false, 0,
                   "EXEC_COUNT_MONOTONICITY: executeCount_ expected %d but got %d.",
                   prevCount + 1, executeCount_);
    }
    DSP_DIAG(EXECUTE, "EXEC_COUNT_INCREMENT: %d -> %d shapesFrozen=%d",
             prevCount, executeCount_, planLifecycle_.isShapesFrozen() ? 1 : 0);
    execCtx->recordFlow(PlanExecutionContext::FlowEventType::EXEC_COUNT_INC,
                         prevCount, executeCount_);
  }

  // Re-classify slot ownership after the first frozen execution (capture step).
  // Capture execution may replace slot arrays, invalidating compile-time
  // ownership classification. Re-classify so lifecycle validation passes.
  if ((planLifecycle_.isShapesFrozen() || planLifecycle_.isReplaying()) && executeCount_ == 2 && slotOwnership_ != nullptr && outputSlots_ != nullptr) {
    for (int i = 0; i < totalOutputSlots_; i++) {
      if (outputSlots_[i] == nullptr) {
        slotOwnership_[i].reset();
        continue;
      }
      classifyAndUpdateOwnership(
          slotOwnership_[i], outputSlots_[i], i,
          externalInputs, numExternalInputs,
          outputSlots_, totalOutputSlots_,
          slotOwnership_);
    }
  }

  auto captureFrozenSnapshotIfReady = [&]() {
    if (!planLifecycle_.isReplaying()) return;
    if (frozenSnapshot_.valid) return;

    frozenSnapshot_.capture(outputSlots_, totalOutputSlots_,
                             lifecycleExternalInputPtrs, numExternalInputs);
    // Null out snapshot entries for transient VIEW_OF_WEIGHT slots whose
    // underlying DataBuffer is a placeholder (not in protectedWeightBuffers_).
    // These views are refreshed each call by slot-exec, so the captured
    // pointers/identity are not authoritative and must be skipped in validate().
    // Pass the RAW externalInputs (not lifecycleExternalInputPtrs) because
    // lifecycleExternalInputPtrs has placeholder entries nulled out by
    // refreshLifecycleExternalInputs(). pruneTransientViewSlots needs to
    // see placeholder DataBuffers to detect slots that alias them.
    frozenSnapshot_.pruneTransientViewSlots(slotOwnership_, protectedWeightBuffers_,
                                              externalInputs, numExternalInputs);

    // Dynamic output slots are intentionally mutable during frozen execution:
    // their buffers may be replaced by plan-internal control/value-shape
    // updates, and they are excluded from addFrozenRef()/frozen fast-path
    // assumptions. Skip them in the frozen snapshot so lifecycle validation
    // continues to enforce pointer stability only for replay-stable outputs.
    if (frozenSnapshot_.slotDataBuffers != nullptr && outputSlots_ != nullptr) {
      std::vector<bool> dynOutputSlot(totalOutputSlots_, false);
      for (int s = 0; s < numSlots_; s++) {
        // FROZEN_CONSTANT slots are never dynamic for snapshot purposes —
        // their buffers must remain stable since they're skipped during execution.
        if (!slots_[s].flags.isDynamicShape || slots_[s].frozenConstantSlot()) continue;
        for (int o = 0; o < slots_[s].wiring.numOutputs; o++) {
          const int oi = slots_[s].wiring.outputSlotIndices[o];
          if (oi >= 0 && oi < totalOutputSlots_) {
            dynOutputSlot[oi] = true;
          }
        }
      }

      // Fused elementwise chains install the LAST chain slot's output buffer
      // into every chain-member output slot (see fused-chain-member writes in
      // slotexec). Those member output slots are logical aliases, not stable
      // storage. Snapshotting them as independent frozen outputs creates false
      // positives when warmup/unfused aliases are replaced by the fused chain's
      // shared output buffer on later executions. Keep lifecycle validation on
      // the canonical tail output slot and prune the member aliases.
      std::vector<bool> fusedChainAliasOutputSlot(totalOutputSlots_, false);
      for (int s = 0; s < numSlots_; s++) {
        const auto& slot = slots_[s];
        if (slot.fusedChain.fusedChainLength <= 1) continue;

        const int lastChainIdx = slot.fusedChain.fusedChainLength - 1;
        const int lastSlotIdx = slot.fusedChain.fusedChainSlots[lastChainIdx];
        if (lastSlotIdx < 0 || lastSlotIdx >= numSlots_) continue;
        const int lastOutputSlotIdx = slots_[lastSlotIdx].wiring.numOutputs > 0
                                          ? slots_[lastSlotIdx].wiring.outputSlotIndices[0]
                                          : -1;

        for (int ci = 0; ci < lastChainIdx; ci++) {
          const int memberSlotIdx = slot.fusedChain.fusedChainSlots[ci];
          if (memberSlotIdx < 0 || memberSlotIdx >= numSlots_) continue;
          if (slots_[memberSlotIdx].wiring.numOutputs <= 0) continue;
          const int oi = slots_[memberSlotIdx].wiring.outputSlotIndices[0];
          if (oi >= 0 && oi < totalOutputSlots_ && oi != lastOutputSlotIdx) {
            fusedChainAliasOutputSlot[oi] = true;
          }
        }
      }

      auto clearSnapshotSlot = [&](int i) {
        if (frozenSnapshot_.slotGpuAddresses != nullptr) frozenSnapshot_.slotGpuAddresses[i] = nullptr;
        frozenSnapshot_.slotDataBuffers[i] = nullptr;
        if (frozenSnapshot_.slotDeviceIds != nullptr) frozenSnapshot_.slotDeviceIds[i] = -1;
        if (frozenSnapshot_.slotPrimaryAddresses != nullptr) frozenSnapshot_.slotPrimaryAddresses[i] = nullptr;
        if (frozenSnapshot_.slotShapeInfoAddresses != nullptr) frozenSnapshot_.slotShapeInfoAddresses[i] = nullptr;
        if (frozenSnapshot_.slotNDArrayIdentity != nullptr) frozenSnapshot_.slotNDArrayIdentity[i] = nullptr;
        if (frozenSnapshot_.slotBufferOffsets != nullptr) frozenSnapshot_.slotBufferOffsets[i] = 0;
        if (frozenSnapshot_.slotLengths != nullptr) frozenSnapshot_.slotLengths[i] = 0;
        if (frozenSnapshot_.slotActualityFlags != nullptr) frozenSnapshot_.slotActualityFlags[i] = 0;
        if (frozenSnapshot_.slotOrderings != nullptr) frozenSnapshot_.slotOrderings[i] = 0;
      };

      int prunedDynamicSlots = 0;
      for (int i = 0; i < totalOutputSlots_; i++) {
        if (!dynOutputSlot[i] || frozenSnapshot_.slotDataBuffers[i] == nullptr) continue;
        clearSnapshotSlot(i);
        prunedDynamicSlots++;
      }

      if (prunedDynamicSlots > 0) {
        DSP_DIAG(MEMORY,
            "LIFECYCLE: pruned %d dynamic output slot(s) from frozen snapshot",
            prunedDynamicSlots);
      }

      int prunedFusedAliasSlots = 0;
      for (int i = 0; i < totalOutputSlots_; i++) {
        if (!fusedChainAliasOutputSlot[i] || frozenSnapshot_.slotDataBuffers[i] == nullptr) continue;
        clearSnapshotSlot(i);
        prunedFusedAliasSlots++;
      }

      if (prunedFusedAliasSlots > 0) {
        DSP_DIAG(MEMORY,
            "LIFECYCLE: pruned %d fused-chain alias output slot(s) from frozen snapshot",
            prunedFusedAliasSlots);
      }
    }

    DSP_DIAG(EXECUTE, "LIFECYCLE: captured buffer pointer snapshot (%d slots, %d extInputs)",
             totalOutputSlots_, numExternalInputs);
    {
      int ts = sd::graph::DspDiagnostics::getInstance().traceSlot();
      if (ts >= 0 && ts < totalOutputSlots_ && outputSlots_[ts] != nullptr) {
        DSP_DIAG(MEMORY, "SNAPSHOT_SLOT_%d: arr=%p db=%p special=%p len=%lld",
                 ts, (void*)outputSlots_[ts], (void*)outputSlots_[ts]->dataBuffer(),
                 (void*)outputSlots_[ts]->specialBuffer(),
                 (long long)outputSlots_[ts]->lengthOf());
      }
    }
  };

  // ── Plan-level phase advancement ───────────────────────────────────────────
  // Phase transitions are automatic based on observed stability.
  // postFreezeExecCount increments here and advancePlanPhase() handles the
  // transitions. Snapshot capture is deferred until the END of the execution,
  // after post-warmup compile/fusion has finished mutating slot buffers.
  if (planLifecycle_.isShapesFrozen() || planLifecycle_.isReplaying()) {
    planLifecycle_.recordPostFreezeExec();
    advancePlanPhase();
  }

  // Auto-seal: after a successful slot-by-slot pass with no explicit freeze,
  // transition the plan in-place to SHAPES_FROZEN. No re-warmup — the slot-by-slot
  // pass we just finished populated slot shape caches and segment execution counts,
  // so platformPrecompileSegments can run directly against the current state. This
  // seals compilation exactly once per plan lifetime so the mid-execution compile
  // counter reflects real post-seal Triton compiles going forward. The eager
  // precompile gate below then calls phaseCompile.
  //
  // Applies to every mode EXCEPT GEM_SLOT_BY_SLOT and GEM_SHAPE_INFERENCE_ONLY.
  // For explicit modes like GEM_TRITON / GEM_NVRTC_JIT / GEM_CUDA_GRAPHS, the Java
  // side does NOT propagate shapesFrozen (see DynamicShapePlanExecutor.applySettingsIfNewHandle
  // comment at line 1248+) — the C++ plan owns its own frozen-state transition. Without
  // this seal, planLifecycle_.isShapesFrozen() stays false, executeCount_ never increments (guarded at
  // line 1476), phaseCompile defers (requires executeCount_>=1), and
  // platformShouldUseGraph returns false — forcing every segment to run slot-by-slot
  // for the life of the plan.
  //
  // GEM_EMULATED_REPLAY (CPU without graph backends) has isSlotBySlot=true in its
  // ModeContract for execution dispatch, but MUST still participate in auto-seal so
  // the plan lifecycle advances to SHAPES_FROZEN. The old guard (!explicitSlotBySlot)
  // conflated "slot-by-slot execution" with "never advance lifecycle" — the check
  // below distinguishes them by testing the actual mode, not the contract property.
  const bool neverAutoSeal = execCtx->forcedSlotBySlot ||
      graphExecutionMode_ == GraphExecutionMode::GEM_SLOT_BY_SLOT ||
      graphExecutionMode_ == GraphExecutionMode::GEM_SHAPE_INFERENCE_ONLY;
  if (!planLifecycle_.compilationDone && !planLifecycle_.isShapesFrozen() &&
      !planLifecycle_.isReplaying() && !neverAutoSeal &&
      planLifecycle_.isSlotBySlot()) {
    int oldExecCount = executeCount_;
    int oldSegCount = static_cast<int>(segments_.size());
    DSP_DIAG(COMPILE,
             "AUTO_SEAL: in-place transition SLOT_BY_SLOT -> SHAPES_FROZEN "
             "(mode=%d segs=%d extInputs=%d executeCount=%d)",
             static_cast<int>(graphExecutionMode_),
             oldSegCount, numExternalInputs, executeCount_);
    execCtx->recordFlow(PlanExecutionContext::FlowEventType::PHASE_TRANSITION,
                         planLifecycle_.toLegacyCode(), static_cast<int>(PlanPhase::SHAPES_FROZEN));
    // legacy sync
    resegmentForFreeze();
    int newSegCount = static_cast<int>(segments_.size());
    execCtx->recordFlow(PlanExecutionContext::FlowEventType::RESEGMENT, oldSegCount, newSegCount);
    planLifecycle_.freezeShapes();
    if (executeCount_ < 1) executeCount_ = 1;
    replacePlanFrozenRefsForCurrentState(
        "AUTO_SEAL", protectedWeightBuffers_, outputSlots_, totalOutputSlots_,
        frozenProtectedRefBuffers_, frozenOutputRefBuffers_);
    execCtx->recordFlow(PlanExecutionContext::FlowEventType::AUTO_SEAL_FIRED,
                         oldExecCount, executeCount_);

    // Auto-seal mutated planLifecycle_ and executeCount_ — the derived state
    // snapshot taken at execute() entry is now stale. Re-derive so downstream
    // phase dispatch, sync gates, and diagnostic checks see the sealed state.
    execCtx->populateDerivedState(
        !planLifecycle_.isSlotBySlot(), executeCount_,
        static_cast<int>(graphExecutionMode_),
        Environment::getInstance().tritonSkipKernels(),
        Environment::getInstance().tritonGraphCapture(),
        Environment::getInstance().tritonVerifyKernels(),
        !externalInputIsVariable_.empty(),
        executionTimingEnabled_,
        anySegmentNeedsWarmup());
    execCtx->recordFlow(PlanExecutionContext::FlowEventType::DERIVED_STATE_REFRESH,
                         executeCount_, planLifecycle_.isShapesFrozen() ? 1 : 0);
  }

  // Frozen constant detection MUST run AFTER auto-seal (which sets planLifecycle_.isShapesFrozen()
  // and executeCount_) but BEFORE Triton precompilation. detectFrozenConstants()
  // gates on executeCount_ >= 1 (at least one warmup has populated slot outputs) and
  // !frozenConstantDetectionDone_ (so it runs exactly once per markExternalInputVariable epoch).
  // It marks shape_of and other constant-producing slots as FROZEN_CONSTANT. The
  // Triton IR builder checks frozenConstantSlot() and skips these slots —
  // preventing the compiled kernel from overwriting frozen constant device buffers
  // during graph replay. If precompilation runs first, the Triton kernel includes
  // frozen constant ops, and replay corrupts their device data.
  detectFrozenConstants();

  // Update allFrozenConstants flag on each segment now that frozen constant
  // detection has run. Segments where EVERY slot is a frozen constant need no
  // capture or execution — their outputs are already populated from warmup.
  {
    int frozenConstSegCount = 0;
    for (auto& seg : segments_) {
      seg.def.allFrozenConstants = true;
      for (int s = seg.def.startSlot; s <= seg.def.endSlot; s++) {
        if (!slots_[s].frozenConstantSlot()) {
          seg.def.allFrozenConstants = false;
          break;
        }
      }
      if (seg.def.allFrozenConstants) frozenConstSegCount++;
    }
    if (frozenConstSegCount > 0) {
      DSP_DIAG(SEGMENT, "Post-freeze: %d of %d segments are all-frozen-constants "
               "(will skip capture and execution)",
               frozenConstSegCount, static_cast<int>(segments_.size()));
    }
  }

  // Eager precompilation: after warmup (executeCount_ just became 1), all shapes
  // are populated in outputSlots_. Compile all Triton modules now so the 2nd
  // execute() goes straight to replay instead of blocking on compilation.
  // planLifecycle_.compilationDone gate ensures this only happens once per plan lifecycle.
  //
  // GEM_EMULATED_REPLAY has isSlotBySlot=true in ModeContract (for execution dispatch)
  // but must still receive a compilation seal so isCompilationSealed() returns true.
  // The old guard (!explicitSlotBySlot) conflated "slot-by-slot execution" with
  // "never seal" — use !neverAutoSeal instead, which only excludes true GEM_SLOT_BY_SLOT
  // and GEM_SHAPE_INFERENCE_ONLY.
  if (!planLifecycle_.compilationDone && !planLifecycle_.isSlotBySlot() && executeCount_ == 1 && !neverAutoSeal) {
    execCtx->recordFlow(PlanExecutionContext::FlowEventType::PHASE_COMPILE,
                         static_cast<int>(segments_.size()));
    phaseCompile(externalInputs, numExternalInputs);

    // Triton compilation internally captures CUDA graphs on DSP-managed
    // streams (tl_dspExecutionStream, tl_dspGapStream). If the capture was
    // aborted or endCapture was not called, the stream remains in capture
    // mode. Subsequent operations on OTHER streams that access pool-allocated
    // memory from the capturing stream's pool will fail with
    // cudaErrorStreamCaptureImplicit (901). Check ALL DSP streams and end
    // any stale capture before entering the replay phase.
    {
      // Check execution stream (passed by Java). `stream` is a STREAM-POINTER
      // (cudaStream_t*); dspEndStaleCapture consumes a STREAM-VALUE — convert first.
      if (stream != nullptr) sd::graph::dspEndStaleCapture(sd::graph::dspStreamPtrToValue(stream), "execution");
      // Check default LaunchContext stream
      {
        void* defaultStream = sd::graph::dspGetLcDefaultStream();
        if (defaultStream != nullptr) sd::graph::dspEndStaleCapture(defaultStream, "default");
      }
      // Check DSP-managed streams (Triton compilation targets)
      sd::graph::dspEndStaleCapture(sd::graph::dspGetExecutionStream(), "tl_dspExecutionStream");
      sd::graph::dspEndStaleCapture(sd::graph::dspGetGapStream(), "tl_dspGapStream");
      sd::graph::dspEndStaleCapture(sd::graph::dspGetGraphCaptureStream(), "tl_graphCaptureStream");
    }
  }

  platformDetectAndPrepareBatchedGemm(externalInputs, numExternalInputs, stream);

  // Adaptive segment splitting (GPU only): if a segment's shape key
  // changes for consecutive executions, split it at the midpoint.
  platformMaybeSplitIfEnabled();

  // Capture the frozen snapshot only after ALL post-warmup mutation has
  // completed AND the plan has reached REPLAYING. SHAPES_FROZEN still
  // allows legitimate pointer churn while segments converge on steady-state
  // allocations, so the baseline must be taken only once pointer stability is
  // actually observed.
  captureFrozenSnapshotIfReady();

  // Print timing breakdown via execution context timing points
  if (execCtx->timingEnabled) {
    auto segMs = std::chrono::duration_cast<std::chrono::microseconds>(execCtx->tSegsDone - execCtx->t0).count();
    auto outMs = std::chrono::duration_cast<std::chrono::microseconds>(execCtx->tOutputsDone - execCtx->tSegsDone).count();
    auto flushMs = std::chrono::duration_cast<std::chrono::microseconds>(execCtx->tFlushDone - execCtx->tOutputsDone).count();
    auto totalMs = std::chrono::duration_cast<std::chrono::microseconds>(execCtx->tFlushDone - execCtx->t0).count();
    DSP_DIAG(TIMING, "segments=%lldus outputs=%lldus flush=%lldus total=%lldus (%d segs, %d slots) | graph=%lldus(%d segs/%d slots) sbs=%lldus(%d segs/%d slots)",
             segMs, outMs, flushMs, totalMs,
             static_cast<int>(segments_.size()), numSlots_,
             phaseStats.graphReplayUs, phaseStats.graphReplaySegs, phaseStats.graphReplaySlots,
             phaseStats.slotBySlotUs, phaseStats.slotBySlotSegs, phaseStats.slotBySlotSlots);
  }

  // End DSP diagnostic step + cross-stream sync + DspStreamGuard cleanup.
  // execCtx->endDiag() is safe to call even if already ended by an early return path.
  execCtx->endDiag(executeCount_);

  // Snapshot per-execution stats from PlanExecutionContext before clearing it.
  snapshotExecStats(execCtx);

  activeExecCtx_ = nullptr;
  flushDeferredSlotDeletes();
  platformEndGuard.dismiss();  // Normal exit — call manually, don't double-call from destructor
  platformEndExecution(executionStatePtr, stream, planLifecycle_.isInFrozenOrReplayState(), executeCount_);
  executionStatePtr = nullptr;

  return Status::OK;
}

// ─── Steady-State Fast Path ─────────────────────────────────────────────────
//
// executeSteadyState() is the hot-path replacement for execute() during
// autoregressive decode. It eliminates ~200ms of per-step CPU overhead by
// skipping ALL validation, lifecycle checks, and diagnostic instrumentation.
//
// The full execute() performs per step:
//   - detectClosedBuffers (scans all 2700+ slots)
//   - External input validation (per-array DataBuffer checks)
//   - Lifecycle external inputs filtering (vector alloc + scan)
//   - refreshStaleViewWrappersInSegment (all segments x slots)
//   - validateLifecycleForPhase (scans all slots)
//   - frozenSnapshot transitions (scans all slots)
//   - REPLAYING segment validation (checks all segments)
//   - Post-exec consistency assertions (scans all slots)
//   - Ownership reclassification (only useful at execCount==2)
//   - Phase advancement (only useful during transitions)
//
// executeSteadyState() skips ALL of the above and goes directly to:
//   platformBeginExecution -> phaseReplay -> outputs -> executeCount_++ -> platformEndExecution
//
// SAFETY: The caller (autoregressive_decode.cu) ensures the plan has been
// through full execute() for the first few steps, validating correctness.
// By step 4+, if the plan had any lifecycle issues, they would have been
// caught. The steady-state path trusts that the plan is healthy.

Status NativeDynamicShapePlan::executeSteadyState(
    NDArray** externalInputs, int numExternalInputs,
    NDArray** requestedOutputs, int numRequestedOutputs,
    void* stream) {

  // One-shot consolidated state for the native-decode hot path. Tagged with plan=%p
  // so it correlates with the Java-side redispatchForCurrentShapes multi-plan switch
  // (which identifies plans by handle). Without this the native decode is invisible in
  // the log and indistinguishable from the slot-by-slot Java reference path.
  dumpPlanPhaseState("executeSteadyState");

  if (ModeContract::forMode(graphExecutionMode_).isSlotBySlot ||
      Environment::getInstance().tritonSkipKernels()) {
    DSP_DIAG(EXECUTE, "[DSP_GATE] executeSteadyState delegates to execute() for slot-by-slot mode");
    return execute(externalInputs, numExternalInputs,
                   requestedOutputs, numRequestedOutputs, stream);
  }

  // Precondition check: fall back to full execute() if not in steady state
  if (!planLifecycle_.isReplaying() || executeCount_ < 4 ||
      Environment::getInstance().tritonVerifyKernels()) {
    DSP_DIAG(EXECUTE, "[DSP_GATE] FALLBACK execute() — shapesFrozen=%d executeCount=%d planPhase=%s verifyKernels=%d",
             (int)planLifecycle_.isShapesFrozen(), (int)executeCount_, planLifecycle_.displayName(),
             (int)Environment::getInstance().tritonVerifyKernels());
    return execute(externalInputs, numExternalInputs,
                   requestedOutputs, numRequestedOutputs, stream);
  }
  DSP_DIAG(EXECUTE, "[DSP_GATE] FAST executeSteadyState() — executeCount=%d planPhase=%s",
           (int)executeCount_, planLifecycle_.displayName());

  // Quick argument count validation (cheap)
  if (numExternalInputs != numExternalInputs_) {
    return Status::BAD_ARGUMENTS;
  }
  if (numRequestedOutputs != numRequestedOutputs_) {
    return Status::BAD_ARGUMENTS;
  }

  // Advance dirty generation instead of clearing the entire bitmap.
  // The compositeReplay loop marks active slots with currentDirtyGeneration_,
  // and tickWriteDevice skips slots where generation != current.
  // This replaces O(totalOutputSlots_) memset with O(1) increment.
  // Wrap around at UINT32_MAX-1 to avoid collision with the 0 sentinel;
  // on wrap, do a full reset so stale generation values can't alias.
  if (currentDirtyGeneration_ < UINT32_MAX - 1) {
    currentDirtyGeneration_++;
  } else {
    currentDirtyGeneration_ = 1;
    std::fill(dirtySlotGenerations_.begin(), dirtySlotGenerations_.end(), 0);
  }

  // Store a persistent copy of external input pointers (see execute() comment)
  lastExternalInputsCopy_.assign(externalInputs, externalInputs + numExternalInputs);
  lastExternalInputs_ = lastExternalInputsCopy_.data();
  lastNumExternalInputs_ = numExternalInputs;
  // Record the ext-input buffer addresses NOW, while the Java-owned NDArrays are guaranteed
  // live (the duration of this JNI call). The JNI address query (getLastExternalInputAddress)
  // reads these recorded values — it must NOT dereference the stored NDArray* later, because
  // callers pass fresh inputs each step and free the old ones (query-time deref = UAF:
  // garbage magic number → DataBuffer integrity-check throw). Raw db pointers only — no
  // specialBuffer() self-heal side effects at record time.
  lastExternalInputAddrs_.resize(numExternalInputs);
  for (int _e = 0; _e < numExternalInputs; _e++) {
    NDArray* _a = externalInputs[_e];
    void* _p = nullptr;
    if (_a != nullptr && _a->dataBuffer() != nullptr) {
      _p = _a->dataBuffer()->special();
      if (_p == nullptr) _p = _a->dataBuffer()->primary();
    }
    lastExternalInputAddrs_[_e] = reinterpret_cast<long long>(_p);
  }

  // Reuse cached PlanExecutionContext — avoid heap alloc/free per step.
  // On first call, create the context and a reusable cross-stream event.
  // On subsequent calls, just reset the dedup flags.
  PlanExecutionContext* execCtx;
  if (steadyStateExecCtx_ != nullptr) {
    execCtx = static_cast<PlanExecutionContext*>(steadyStateExecCtx_);
    // Reset per-step sync state machine (fresh contexts start at UNSYNCED)
    execCtx->resetSyncPhase();
    execCtx->flowEventCount = 0;
    execCtx->streamSyncCount = 0;
    execCtx->eventSyncCount = 0;
    DSP_LIFECYCLE_CLEAR();
  } else {
    execCtx = new PlanExecutionContext();
    steadyStateExecCtx_ = static_cast<void*>(execCtx);
  }

  // Save previous DSP stream for RAII-style restore at all exit points.
  // On CPU builds dspGetExecutionStream() returns nullptr — no-op on all DSP calls below.
  void* prevDspStream = sd::graph::dspGetExecutionStream();

  // CUDA-only: device capture, cross-stream events, deterministic cuBLAS.
  // No-op on CPU builds.
  platformSetupSteadyStateCuda(static_cast<void*>(execCtx), stream);

  // Populate derived state for steady state.
  // Use anySegmentNeedsWarmup() — the SINGLE source of truth for warmup state.
  bool segWarmup = anySegmentNeedsWarmup();
  execCtx->execCount = executeCount_;
  execCtx->frozen = true;
  execCtx->isFrozenSteadyState = !segWarmup;
  execCtx->isFirstFrozenWarmup = false;
  execCtx->needsFullSync = segWarmup;
  execCtx->forcedSlotBySlot = false;
  execCtx->graphExecutionMode = static_cast<int>(graphExecutionMode_);
  execCtx->allowGraphCaptureReplay = Environment::getInstance().tritonGraphCapture();
  execCtx->useVariableFilter = !externalInputIsVariable_.empty();
  execCtx->tritonVerifyEnabled = false;
  execCtx->diagAnyEnabled = false;
  execCtx->diagVerifyEnabled = false;
  execCtx->timingEnabled = executionTimingEnabled_;
  execCtx->isReplay = true;
  execCtx->segmentsTotal = static_cast<int>(segments_.size());

  activeExecCtx_ = static_cast<void*>(execCtx);

  // Clear stale CUDA errors — only reset message when an error was actually set
  // (avoids heap delete+new of std::string on every decode step)
  auto* errRef = sd::LaunchContext::defaultContext()->errorReference();
  if (errRef->errorCode() != 0) {
    errRef->setErrorCode(0);
    errRef->setErrorMessage("");
  }

  // Re-run frozen constant detection if markExternalInputVariable() cleared the done
  // flag since the last detection pass. This covers the autoregressive_decode pattern:
  //   1. Java warms up the plan (execute() runs detectFrozenConstants() at executeCount_==1,
  //      but KV inputs are SOURCE_VARIABLE → externalInputIsVariable_[kvIdx]=false → any
  //      slot reading only from KV with no placeholder-chain is incorrectly FROZEN_CONSTANT)
  //   2. C++ autoregressive_decode calls markExternalInputVariable(kvIdx) before the loop,
  //      which: (a) sets externalInputIsVariable_[kvIdx]=true, (b) calls
  //      invalidateSegmentCaptures which calls resetSlotStatesForSegment, which calls
  //      slotPhase.reset() for all slots — clearing the incorrect FROZEN_CONSTANT flags,
  //      (c) sets frozenConstantDetectionDone_=false to request re-detection.
  //   3. The decode loop calls executeSteadyState() (NOT execute()), so detectFrozenConstants()
  //      would never be re-called without this explicit call here.
  //   4. On the first executeSteadyState() call after markExternalInputVariable(), this
  //      re-runs detectFrozenConstants() with all KV inputs correctly marked as variable,
  //      producing correct FROZEN_CONSTANT classifications before any CUDA graph capture.
  //
  // The guard in detectFrozenConstants() ensures it only runs when:
  //   - Not slot-by-slot mode
  //   - executeCount_ >= 1 (at least one warmup has run, slot outputs are populated)
  //   - frozenConstantDetectionDone_ is false (cleared by markExternalInputVariable)
  // So this call is a fast no-op (single branch check) when re-detection is not needed.
  if (!frozenConstantDetectionDone_ && !planLifecycle_.isSlotBySlot() && executeCount_ >= 1) {
    detectFrozenConstants();
    // After re-detection, update allFrozenConstants on each segment to reflect
    // the corrected slot classifications.
    for (auto& seg : segments_) {
      seg.def.allFrozenConstants = true;
      for (int s = seg.def.startSlot; s <= seg.def.endSlot; s++) {
        if (!slots_[s].frozenConstantSlot()) {
          seg.def.allFrozenConstants = false;
          break;
        }
      }
    }
  }

  // Use platformTryFrozenFastPath — this is the SAME fast path that execute()
  // uses in steady state. It handles ext input H2D sync, cross-stream ordering,
  // arg table refresh, and single-graph replay in one tight function.
  // phaseReplay is heavier (segment iteration, lifecycle checks) and was causing
  // accuracy issues because the ext input sync flow differs from the frozen fast path.
  bool usedFrozenFastPath = false;
  auto result = platformTryFrozenFastPath(
      externalInputs, numExternalInputs, requestedOutputs, numRequestedOutputs, stream);

  if (result == Status::MAYBE) {
    // Frozen fast path not applicable — fall back to full phaseReplay.
    // This shouldn't happen in steady state, but handle gracefully.
    PhaseExecutionStats phaseStats;
    result = phaseReplay(externalInputs, numExternalInputs,
                         stream, &phaseStats);
  } else if (result == Status::OK) {
    usedFrozenFastPath = true;
  }

  if (result != Status::OK) {
    activeExecCtx_ = nullptr;
    if (sd::graph::dspIsCudaBuild()) {
      if (stream != nullptr) {
        sd::graph::dspSetExecutionStream(prevDspStream);
      }
      // Restore cuBLAS on early exit too
      if (ModeContract::forMode(graphExecutionMode_).requiresDeterministicCublas) {
        platformSetDeterministicCublas(false);
      }
    }
    return result;
  }

  // Native KV scatter (if configured on the plan)
  if (kvScatterConfigured_) {
    executeKvScatterPostExec(stream);
  }

  // platformTryFrozenFastPath increments executeCount_ internally when OK.
  // Only increment here if phaseReplay was used instead.
  if (!usedFrozenFastPath) {
    // Dump lifecycle trail at end of every execute (before incrementing count)
    DSP_LIFECYCLE_DUMP("EXEC_COMPLETE");
    incrementExecuteCount("exec_complete");
  }

  // CUDA-only: completion event, stream restore, cuBLAS restore.
  // No-op on CPU builds.
  platformTeardownSteadyStateCuda(static_cast<void*>(execCtx), stream, prevDspStream);

  activeExecCtx_ = nullptr;
  return Status::OK;
}

// ─── Statistics ─────────────────────────────────────────────────────────────

int NativeDynamicShapePlan::getNumCapturedGraphSegments() const {
  return platformCountCapturedGraphSegments();
}

int NativeDynamicShapePlan::getTotalGraphReplays() const {
  return totalGraphReplays_;
}

std::string NativeDynamicShapePlan::getSegmentCompilationAudit(int segIdx) const {
  if (segIdx < 0 || segIdx >= static_cast<int>(segments_.size())) return "{}";
  auto& seg = segments_[segIdx];
  std::ostringstream ss;
  ss << "{\"segmentIdx\":" << segIdx
     << ",\"startSlot\":" << seg.def.startSlot
     << ",\"endSlot\":" << seg.def.endSlot
     << ",\"compiledByBackend\":\"" << seg.exec.compiledByBackend << "\""
     << ",\"capturable\":" << (seg.def.isCapturable ? "true" : "false")
     << ",\"compilationFailed\":" << (seg.exec.compilationFailed ? "true" : "false")
     << ",\"executionCount\":" << seg.exec.executionCount
     << "}";
  return ss.str();
}

void NativeDynamicShapePlan::setBackendPriority(const std::vector<std::string>& priority) {
  backendPriority_ = priority;
  // Reset cached backends so new priority takes effect
  gpuGraphBackendChecked_ = false;
  gpuGraphBackend_ = nullptr;
  cpuGraphBackendChecked_ = false;
  cpuGraphBackend_ = nullptr;
  cpuGraphBackendChainBuilt_ = false;
  cpuGraphBackendChain_.clear();
}

// ─── Memory management ─────────────────────────────────────────────────────

// View wrappers deleted inline in slotexec. No batched/deferred close needed.

void NativeDynamicShapePlan::markExternalInputVariable(int extIdx) {
  DSP_DIAG(EXECUTE, "markExternalInputVariable: CALLED extIdx=%d numExt=%d",
           extIdx, numExternalInputs_);
  if (extIdx < 0 || extIdx >= numExternalInputs_) return;

  // Resize if needed (covers plans loaded from binary that didn't populate this vector).
  if (externalInputIsVariable_.empty()) {
    externalInputIsVariable_.resize(numExternalInputs_, false);
  }
  if (extIdx >= static_cast<int>(externalInputIsVariable_.size())) {
    externalInputIsVariable_.resize(numExternalInputs_, false);
  }

  bool wasAlreadyVariable = externalInputIsVariable_[extIdx];
  DSP_DIAG(EXECUTE, "markExternalInputVariable: ext[%d] %s", extIdx,
           wasAlreadyVariable ? "ALREADY variable" : "marking as variable NOW");

  externalInputIsVariable_[extIdx] = true;

  if (!wasAlreadyVariable) {
    // ── Update variable index caches ──────────────────────────────────────────
    // Rebuild cachedVariableExtIndices_ from the authoritative
    // externalInputIsVariable_ vector so sync paths and introspection APIs
    // reflect the new variable set immediately.
    cachedVariableExtIndices_.clear();
    variableExternalInputIndices_.clear();
    variableIndicesCached_ = false;
    for (int i = 0; i < static_cast<int>(externalInputIsVariable_.size()); i++) {
      if (externalInputIsVariable_[i]) {
        cachedVariableExtIndices_.push_back(i);
      }
    }

    // Recompute transitive variable dependency since the variable set changed.
    computeSlotVariableDependency();
  }

  // Only invalidate captures when the variable status actually changed.
  // Redundant marks (ext already variable) must NOT invalidate — each
  // invalidation resets segment execution counts and forces re-warmup +
  // re-capture of CUDA graphs, which in a 60-KV-cache model means 60+
  // redundant invalidations that prevent the graph from ever stabilizing.
  if (!wasAlreadyVariable && !planLifecycle_.isSlotBySlot()) {
    DSP_DIAG(EXECUTE, "markExternalInputVariable: ext[%d] invalidating captures "
             "after explicit variable mark (segments=%d phase=%s)",
             extIdx, (int)segments_.size(), planLifecycle_.displayName());
    if (planLifecycle_.isReplaying()) {
      planLifecycle_.unseal();
    } else if (planLifecycle_.isInFrozenOrReplayState()) {
      planLifecycle_.recordPointersUnstable();
    }
    frozenSnapshot_.clear();
    frozenConstantDetectionDone_ = false;
    planLifecycle_.compilationDone = false;
    for (auto& seg : segments_) {
      SegmentLifecycle::invalidateSegmentCaptures(this, seg, "mark_external_input_variable");
    }
    clearGpuBackendFailedCache();
    platformClearCastCache();
  } else if (wasAlreadyVariable) {
    DSP_DIAG(EXECUTE, "markExternalInputVariable: ext[%d] name='%s' ALREADY variable — "
             "skipping invalidation (captures remain valid)",
             extIdx,
             (extIdx < static_cast<int>(externalInputNames_.size()))
                 ? externalInputNames_[extIdx].c_str() : "?");
  }

  // ── Staging buffer cleanup (only if staging existed before) ────────────────
  // If staging buffers exist, the CUDA graph was captured reading from staging
  // addresses. The staging buffer for the newly-variable input may not exist
  // yet (it was a weight before). Pre-allocate it so ensureAndSyncStagingBuffers
  // can D2D-copy into it before the next replay.
  if (effectiveExternals_ != nullptr || placeholderStagingBuffers_ != nullptr) {
    // Clear stale staging address records so frozenFastPath CHECK 3 doesn't
    // compare new staging addresses against old (potentially freed) ones.
    prevStagingAddresses_.clear();
  }

  // Pre-allocate staging buffer for the marked input so getStagingBufferAddress()
  // returns non-zero immediately after markVariable (before the plan re-enters
  // composite replay where ensureAndSyncStagingBuffers normally allocates).
  NDArray* lastExt = getLastExternalInput(extIdx);
  if (lastExt != nullptr && !lastExt->isEmpty()) {
    if (placeholderStagingBuffers_ == nullptr) {
      placeholderStagingBuffers_ = new NDArray*[numExternalInputs_]();
      effectiveExternals_ = new NDArray*[numExternalInputs_]();
    }
    if (placeholderStagingBuffers_[extIdx] == nullptr) {
      placeholderStagingBuffers_[extIdx] = new NDArray(
          lastExt->ordering(), *lastExt->getShapeAsVector(),
          lastExt->dataType(), LaunchContext::defaultContext());
    }
  }

  // Re-detect frozen constants since the variable set changed — ops that
  // were frozen because their transitive inputs appeared constant may now
  // depend on a variable external.
  frozenConstantDetectionDone_ = false;

  const char* name = (extIdx < static_cast<int>(externalInputNames_.size()))
                     ? externalInputNames_[extIdx].c_str() : "?";
  DSP_DIAG(EXECUTE, "markExternalInputVariable: ext[%d] name='%s' now variable. "
           "wasAlready=%d segments=%d",
           extIdx, name, wasAlreadyVariable ? 1 : 0, (int)segments_.size());
}

void NativeDynamicShapePlan::markExternalInputPlaceholder(int extIdx) {
  if (extIdx < 0 || extIdx >= numExternalInputs_) return;
  if (externalInputIsPlaceholder_.empty()) {
    externalInputIsPlaceholder_.resize(numExternalInputs_, false);
  }
  externalInputIsPlaceholder_[extIdx] = true;
  // Also mark as variable (placeholders are a subset of variables)
  markExternalInputVariable(extIdx);
}

void NativeDynamicShapePlan::setShapesFrozen(bool frozen) {
  // ── Lifecycle enforcement: phases are LINEAR and IMMUTABLE ───────────────
  // Once a plan is frozen it stays frozen forever. Backwards transitions are
  // architectural errors — the plan cache manages separate entries for each
  // distinct shape key, so there is NEVER a valid reason to unfreeze an
  // existing plan. If shapes change, the caller must destroy this plan and let
  // the cache produce a fresh entry for the new shape.
  if (!frozen) {
    DSP_THROW(EXECUTE,
              "LIFECYCLE VIOLATION: setShapesFrozen(false) called on plan %p (phase=%s). "
              "Phases are strictly linear: SLOT_BY_SLOT → SHAPES_FROZEN → REPLAYING. "
              "Backwards transitions are illegal. Destroy this plan and let the plan cache create "
              "a fresh entry for the new shape.",
              this, planLifecycle_.displayName());
  }

  bool wasFrozen = !planLifecycle_.isSlotBySlot();
  // Idempotent: if already frozen and caller wants to freeze again, no-op.
  // This handles the case where the plan auto-advanced to SHAPES_FROZEN
  // during execute() and Java tries to freeze afterward.
  if (wasFrozen) {
    return;
  }

  DSP_REQUIRE_PLAN_PHASE_AT_MOST(PlanPhase::SLOT_BY_SLOT, "setShapesFrozen(true)");

  // ── LIFECYCLE VIOLATION: freezing after execute ────────────────────────
  // If executeCount_ > 0, the plan has already executed N times unfrozen.
  // After freeze, isFirstFrozenWarmup = (executeCount == 0) will be FALSE,
  // meaning the warmup/capture phase is skipped entirely. The plan jumps
  // straight to replay without ever capturing CUDA graphs or establishing
  // baseline output buffers. This causes stale buffer reuse and
  // non-deterministic results.
  //
  // Correct lifecycle: compile → freeze → warmup execute → replay.
  // Freezing after execute violates this ordering.
  REQUIRE_TRUE(executeCount_ == 0, 0,
               "LIFECYCLE VIOLATION in setShapesFrozen(true): executeCount=%d > 0. "
               "The plan has already executed %d times without being frozen. "
               "Freezing now means isFirstFrozenWarmup will never be true — "
               "the warmup/capture phase will be skipped, causing stale buffers "
               "and non-deterministic replay. "
               "Fix: freeze the plan BEFORE the first execute(), or let auto-seal "
               "handle the transition (auto-seal fires during execute).",
               executeCount_, executeCount_);

  auto status = phaseFreeze();
  if (status != Status::OK) {
    DSP_THROW(COMPILE,
             "setShapesFrozen(true): phaseFreeze failed with status %d — "
             "plan NOT frozen (would leave partially-frozen inconsistent state)",
             static_cast<int>(status));
  }
}

// ─── Phase lifecycle methods ──────────────────────────────────────────────
// Each method encapsulates ALL work for its phase. No scattered logic.

// One-shot DSP state dump: the COMPLETE plan + per-segment picture in a single
// DSP_DIAG block. Each per-segment line carries the RAW inputs that drive every
// plan-phase decision (phase/outcome/execCount/sealed/handleReady/composite/
// compiledBy) so an odd/stuck state (frozen-not-advancing, never-captures,
// premature-block) is diagnosable from ONE place — the replay verdict
// (sealed && (handleReady||composite)) and the seal-gate outcome are both readable.
void NativeDynamicShapePlan::dumpPlanPhaseState(const char* context) const {
  DSP_DIAG(EXECUTE,
           "DSP_STATE[%s]: plan=%p mode=%d phase=%s execCount=%d postFreezeExec=%d pointersStable=%d segs=%zu",
           context, (void*)this, (int)graphExecutionMode_, planLifecycle_.displayName(), executeCount_,
           planLifecycle_.postFreezeExecCount, (int)planLifecycle_.pointersStable(),
           segments_.size());
  for (size_t i = 0; i < segments_.size(); i++) {
    const auto& s = segments_[i];
    DSP_DIAG(EXECUTE,
             "  DSP_STATE seg[%zu] %d-%d backend=%d phase=%s outcome=%d execCount=%d blocks=%d "
             "capturable=%d sealed=%d handleReady=%d composite=%d compiledBy=%s",
             i, s.def.startSlot, s.def.endSlot, (int)s.def.selectedBackend,
             s.exec.displayPhaseName(), (int)s.exec.outcome, s.exec.executionCount,
             (int)segmentBlocksPlanPhase(s), (int)s.def.isCapturable,
             (int)s.exec.segPhase.isSealed(),
             (int)(s.exec.replayHandle != nullptr && s.exec.replayHandle->isReady()),
             (int)segmentHasReadyCompositeHandles(s),
             s.exec.compiledByBackend.empty() ? "-" : s.exec.compiledByBackend.c_str());
  }
}

void NativeDynamicShapePlan::advancePlanPhase() {
  // ── Plan-level phase advancement ───────────────────────────────────────────
  // Phase transitions are automatic based on observed stability:
  //   SHAPES_FROZEN → REPLAYING: after 2+ frozen executions with every
  //                              replay-eligible segment pointer-stable AND
  //                              in backend-specific replay steady state.
  //
  // Caller is responsible for: calling planLifecycle_.recordPostFreezeExecution(),
  // capturing frozenSnapshot_ on first frozen execution.
  if (!planLifecycle_.isShapesFrozen()) return;

  // One-shot state snapshot at every frozen-plan advancement (DSP_DIAG-gated).
  dumpPlanPhaseState("advancePlanPhase");

  // Check pointer stability across all replay-eligible segments.
  // Requires stability to be observed on 2 CONSECUTIVE steps before confirming.
  // This replaces the old POINTERS_STABLE intermediate phase — without it, the plan
  // could advance to REPLAYING on the same step as capture, causing the frozen fast
  // path to skip address verification (pointersStable_=true skips hash checks) while
  // placeholder input addresses haven't yet been validated across steps.
  if (planLifecycle_.isShapesFrozen() && planLifecycle_.postFreezeExecCount >= 2) {
    bool allStable = true;
    for (auto& seg : segments_) {
      if (!segmentHasStablePointersForPlanPhase(seg, slots_)) {
        allStable = false;
        break;
      }
    }
    if (allStable) {
      planLifecycle_.recordPointersStable();
    } else {
      planLifecycle_.recordPointersUnstable();
    }
  }

  // Promote to REPLAYING only once pointers are stable AND every replay-eligible
  // segment has reached steady-state replay.
  if (planLifecycle_.pointersStable() && planLifecycle_.isShapesFrozen()) {
    bool hasReplayEligibleSegment = false;
    bool allReplaying = true;
    for (size_t si = 0; si < segments_.size(); si++) {
      auto& seg = segments_[si];
      if (!segmentBlocksPlanPhase(seg)) continue;
      hasReplayEligibleSegment = true;
      if (!segmentIsFullyReplayingForPlanPhase(seg)) {
        allReplaying = false;
        DSP_DIAG(EXECUTE, "[PHASE_BLOCK] seg[%d] (%d-%d) blocks REPLAYING: backend=%d lifecycle=%s "
                 "execCount=%d handleReady=%d compositeReady=%d argStable=%d frozenExec=%d",
                 (int)si, seg.def.startSlot, seg.def.endSlot,
                 (int)seg.def.selectedBackend, seg.exec.displayPhaseName(),
                 seg.exec.executionCount,
                 (int)(seg.exec.replayHandle != nullptr && seg.exec.replayHandle->isReady()),
                 (int)segmentHasReadyCompositeHandles(seg),
                 (int)!seg.exec.needsArgRefresh(), planLifecycle_.postFreezeExecCount);

        // ── HARD ERROR: mode contract violation after sufficient executions ──
        // If CUDA_GRAPHS or TRITON mode hasn't reached replay after 10 frozen
        // executions, something is fundamentally broken. Throw instead of
        // silently running slot-by-slot forever.
        if (planLifecycle_.postFreezeExecCount >= 10 &&
            !ModeContract::forMode(graphExecutionMode_).allowsPhaseStall) {
          dumpPlanPhaseState("phase_stall_10x");
          REQUIRE_TRUE(false, 0,
                       "DSP MODE VIOLATION: seg[%d-%d] (backend=%d) still not replaying after "
                       "%d frozen executions in mode=%d. handleReady=%d compilationFailed=%d "
                       "lifecycle=%s. The execution mode requires ALL capturable segments to "
                       "reach replay state. Silent slot-by-slot fallback is banned.",
                       seg.def.startSlot, seg.def.endSlot,
                       static_cast<int>(seg.def.selectedBackend),
                       planLifecycle_.postFreezeExecCount,
                       static_cast<int>(graphExecutionMode_),
                       static_cast<int>(seg.exec.replayHandle != nullptr && seg.exec.replayHandle->isReady()),
                       static_cast<int>(seg.exec.compilationFailed),
                       seg.exec.displayPhaseName());
        }
        break;
      }
    }
    if (hasReplayEligibleSegment && allReplaying && !planLifecycle_.isReplaying()) {
      // Count segments with GRAPH_REPLAY outcome — if zero, the plan will run
      // slot-by-slot forever and must NOT be marked REPLAYING.
      //
      // EMULATED_REPLAY segments (CPU without graph backends) are sealed with
      // ZERO_KERNEL_SBS outcome by markEmulatedSealed() — they re-execute ops
      // slot-by-slot on each step rather than replaying a baked CUDA graph.
      // Despite the ZERO_KERNEL_SBS label, sealed EMULATED_REPLAY segments ARE
      // performing their intended replay: the lifecycle (WARMUP → CAPTURING →
      // SEALED) completed successfully and totalGraphReplays_ is incremented
      // per step.  Count them as replay-capable so the plan can advance to
      // REPLAYING (SEALED) and assertFrozenExecCountAtLeast / assertPointersStable
      // work correctly on CPU.
      int graphReplaySegCount = 0;
      for (auto& s : segments_) {
        if (s.exec.outcome == SegmentExecOutcome::GRAPH_REPLAY) {
          graphReplaySegCount++;
        } else if (s.def.selectedBackend == SelectedBackend::EMULATED_REPLAY &&
                   s.exec.segPhase.isSealed()) {
          // Sealed EMULATED_REPLAY = CPU replay steady state: counts as replay.
          graphReplaySegCount++;
        }
      }
      const char* oldPhase = planLifecycle_.displayName();
      if (graphReplaySegCount == 0) {
        // All eligible segments reached steady state but none has a real CUDA graph
        // (genuinely graphless: terminal / 0-kernel). This can no longer fire
        // mid-capture: segmentIsFullyReplayingForPlanPhase only reports a segment
        // "fully replaying" once it actually has a ready graph, so a still-capturing
        // segment keeps allReplaying=false and this gate is not reached until every
        // segment has truly resolved. Block instead of falsely sealing as REPLAYING.
        DSP_DIAG(EXECUTE, "[PHASE_TRANSITION] plan %s -> REPLAY_BLOCKED reason=zero_graph_replay_segments frozenExec=%d",
                 oldPhase, planLifecycle_.postFreezeExecCount);
        planLifecycle_.blockReplay("zero_graph_replay_segments");
      } else {
        DSP_TRACE_PHASE(trace_, -1,
                        static_cast<uint8_t>(getPlanPhase()),
                        static_cast<uint8_t>(PlanPhase::REPLAYING),
                        static_cast<uint32_t>(executeCount_));
        planLifecycle_.seal();
        DSP_DIAG(EXECUTE, "[PHASE_TRANSITION] plan %s -> REPLAYING reason=all_segments_fully_replaying frozenExec=%d",
                 oldPhase, planLifecycle_.postFreezeExecCount);

        // ── Compilation summary at warmup→replay transition ─────────────────
        // Emitted exactly once when all segments reach steady-state replay.
        // This marks the end of compilation/capture overhead.
        {
          int compiledSegs = 0, capturedSegs = 0, failedSegs = 0, slotBySlotSegs = 0;
          for (auto& s : segments_) {
            if (s.exec.compilationFailed) {
              failedSegs++;
            } else if (!s.exec.compiledByBackend.empty()) {
              compiledSegs++;
            }
            if (s.exec.replayHandle != nullptr && s.exec.replayHandle->isReady()) {
              capturedSegs++;
            }
            if (!s.def.isCapturable) {
              slotBySlotSegs++;
            }
          }
          int totalIslandHandles = 0;
          int totalMergedGroups = 0;
          for (auto& s : segments_) {
            totalIslandHandles += static_cast<int>(s.exec.compositeReplaySchedule.compositeReplayHandles.size());
            totalMergedGroups += static_cast<int>(s.exec.compositeReplaySchedule.mergedReplayHandles.size());
          }
          DSP_DIAG_BANNER(COMPILE, "WARMUP COMPLETE",
                   "segs=%d compiled=%d captured=%d failed=%d sbs=%d "
                   "islands=%d mergedGroups=%d replays=%d warmupExec=%d",
                   static_cast<int>(segments_.size()), compiledSegs, capturedSegs,
                   failedSegs, slotBySlotSegs,
                   totalIslandHandles, totalMergedGroups,
                   totalGraphReplays_, planLifecycle_.postFreezeExecCount);
        }
      }  // end else (graphReplaySegCount > 0)
    }
  }
}

void NativeDynamicShapePlan::demotePlanPhase(PlanPhase targetPhase, const char* reason) {
  DSP_THROW(FALLBACK,
           "[PHASE_TRANSITION] plan %s -> %s reason=%s frozenExec=%d kind=demotion — "
           "phase demotion is an error, not a fallback",
           planLifecycle_.displayName(), dsp::planPhaseName(targetPhase),
           reason, planLifecycle_.postFreezeExecCount);
}

Status NativeDynamicShapePlan::phaseFreeze() {
  DSP_REQUIRE_PLAN_PHASE_EXACT(PlanPhase::SLOT_BY_SLOT, "phaseFreeze");
  auto& env = Environment::getInstance();
  bool mergeSegments = env.dspFreezeMergeSegments();
  frozenSnapshot_.clear();

  // ── Fusion pass (slot-by-slot → freeze transition) ──────────────────
  if (numSlots_ > 1) {
    auto fusions = FusionPass::detectFusions(slots_, numSlots_, externalInputRanks_);
    if (!fusions.empty()) {
      DSP_DIAG(FUSION, "detected %d fusion candidates (post-warmup)",
               (int)fusions.size());
      int applied = FusionPass::applyFusions(slots_, numSlots_, fusions);
      DSP_DIAG(FUSION, "applied %d of %d fusion candidates",
               applied, (int)fusions.size());

      // Post-fusion guard: disable in-place when the source slot is a requested output.
      if (requestedOutputSlotIndices_ != nullptr && numRequestedOutputs_ > 0) {
        std::unordered_set<int> reqOutSet;
        for (int ri = 0; ri < numRequestedOutputs_; ri++) {
          int si = requestedOutputSlotIndices_[ri];
          if (si >= 0) reqOutSet.insert(si);
        }
        int disabledForReqOutput = 0;
        for (int s = 0; s < numSlots_; s++) {
          auto& sl = slots_[s];
          int srcSlot = sl.inPlaceSourceSlot();
          if (srcSlot >= 0 && reqOutSet.count(srcSlot)) {
            sl.disableInPlaceFusion();
            disabledForReqOutput++;
          }
        }
        if (disabledForReqOutput > 0) {
          DSP_DIAG(FUSION, "post-fusion: disabled %d in-place ops (source is requested output)",
                   disabledForReqOutput);
        }
      }
    }
  }

  DSP_DIAG(SEGMENT, "SEGMENT_MAP_BEFORE_FREEZE: %d segments", (int)segments_.size());
  for (int i = 0; i < (int)segments_.size(); i++) {
    auto& s = segments_[i];
    DSP_DIAG(SEGMENT, "  seg[%d]: slots[%d-%d] capturable=%d hasReplay=%d "
             "compilationFailed=%d execCount=%d",
             i, s.def.startSlot, s.def.endSlot, s.def.isCapturable,
             s.exec.replayHandle != nullptr, s.exec.compilationFailed, s.exec.executionCount);
  }

  // Resegment: merge data-dependent ops into capturable segments now that
  // shapes are frozen. This collapses hundreds of fragments into a few large
  // segments, enabling monolithic graph capture/replay.
  resegmentForFreeze();

  // Reset segment execution state for freeze.
  for (auto& seg : segments_) {
    seg.exec.resetForWarmup();
    seg.exec.markArgsStale();
    // compilationFailed is managed by lifecycle (markFailed/reset) — not reset here
  }

  planLifecycle_.freezeShapes();

  // Validate view reference count consistency at freeze boundary.
  // Mismatched refcounts indicate missed addViewRef/removeViewRef calls
  // during warmup execution — catch these before they cause leaked buffers.
  if (slotOwnership_ != nullptr && outputSlots_ != nullptr) {
    int staleCount = detectStaleOwnership(slotOwnership_, totalOutputSlots_, outputSlots_);
    if (staleCount > 0) {
      DSP_DIAG(MEMORY, "phaseFreeze: WARNING: %d stale ownership entries detected — "
               "re-classifying before freeze", staleCount);
    }
    if (!validateViewRefCounts(slotOwnership_, totalOutputSlots_, outputSlots_)) {
      DSP_DIAG(MEMORY, "phaseFreeze: WARNING: viewRefCount inconsistency detected — "
               "view reference tracking may be corrupted");
    }
  }

  DSP_DIAG(EXECUTE, "[PHASE_TRANSITION] plan SLOT_BY_SLOT -> SHAPES_FROZEN reason=phaseFreeze "
            "segments=%d slots=%d extInputs=%d mergeSegments=%d recompile=%d",
            (int)segments_.size(), numSlots_, numExternalInputs_,
            mergeSegments ? 1 : 0, env.dspFreezeRecompile() ? 1 : 0);

  // Reset cast-cache indices instead of clearing. clearCastCache() deletes
  // the thread-local FP32-upcast NDArrays whose device addresses are baked
  // into other plans' CUDA graph nodes (captured during compositeReplay).
  // When phaseFreeze() is invoked externally via setPlanShapesFrozen(true)
  // on a brand-new plan (from the Java frozen multi-plan switch, line 1669),
  // clearing destroys entries that the ORIGINAL plan's CUDA graph depends on.
  // Replaying that graph then reads freed GPU memory → NaN in rms_norm_linear.
  //
  // resetCastCacheIndices() achieves the same goal (new plan warmup starts
  // from slot 0) without freeing the NDArrays. This is the same pattern used
  // in compositeReplay pre-capture warmup (NativeDynamicShapePlan_gpubackend.cu
  // line 3079) and is safe for multi-plan CUDA graph scenarios.
  MmulHelper::resetCastCacheIndices();

  resetExecuteCount("phase_freeze");
  planLifecycle_.compilationDone = false;
  shapePrePassDone_ = false;

  // ── Buffer coloring: compute color assignments ──────────────────────────
  // Compute at freeze time when ownership is classified and shapes are stable.
  // apply() happens lazily on the first post-freeze execution (phaseWarmup).
  if (slotLiveness_ != nullptr && slotOwnership_ != nullptr && outputSlots_ != nullptr) {
    // Build requested output set
    std::unordered_set<int> requestedOutputSet;
    if (requestedOutputSlotIndices_ != nullptr) {
      for (int i = 0; i < numRequestedOutputs_; i++) {
        requestedOutputSet.insert(requestedOutputSlotIndices_[i]);
      }
    }

    try {
      colorMap_.compute(*slotLiveness_, outputSlots_, totalOutputSlots_,
                        slotOwnership_, requestedOutputSet);
      if (colorMap_.numColoredSlots() > 0) {
        DSP_DIAG(MEMORY, "phaseFreeze: buffer coloring computed: %d slots -> %d colors, "
                 "estimated saving %zuMB",
                 colorMap_.numColoredSlots(), colorMap_.numColors(),
                 colorMap_.estimatedBytesSaved() / (1024 * 1024));
      }
    } catch (const std::exception& e) {
      DSP_DIAG(MEMORY, "phaseFreeze: buffer coloring failed: %s", e.what());
      colorMap_.reset();
    }
  }

  // NOTE: Neither protectedWeightBuffers_ nor outputSlots_ DataBuffers are
  // frozen here. ALL freezing happens at the end of phaseWarmup() after warmup
  // execution completes. Freezing before warmup prevents warmup from executing
  // correctly: reshape ops create zero-copy views of weight buffers and call
  // z->buffer() which triggers allocatePrimary on device-only buffers. The
  // frozen refs protect against mutation during graph capture/replay — warmup
  // is internal execution that needs full write access to everything.

  return Status::OK;
}

Status NativeDynamicShapePlan::phaseWarmup(NDArray** externalInputs, int numExternalInputs,
                                           void* stream, PhaseExecutionStats* stats) {
  DSP_DIAG(EXECUTE, "phaseWarmup: BEGIN segments=%d extInputs=%d", (int)segments_.size(), numExternalInputs);

  long long slotBySlotUs = 0;
  int slotBySlotSegs = 0;
  int slotBySlotSlots = 0;
  using Clock = std::chrono::high_resolution_clock;

  // Reset segment state for warmup.
  // compilationFailed is managed by lifecycle (markFailed/reset) — not reset here.
  for (auto& segment : segments_) {
    segment.exec.resetForWarmup();
    if (segment.exec.replayHandle) {
      platformCleanupSegmentForRebuild(segment);
    }
  }

  // Reset cast-cache indices (NOT full clear) before warmup.
  // phaseWarmup() is called exclusively for externally-frozen plans
  // (setPlanShapesFrozen/frozen multi-plan switch). The thread-local cast
  // cache may contain FP32-upcast NDArrays whose device addresses are already
  // baked into another plan's CUDA graph nodes. clearCastCache() would delete
  // those NDArrays, causing the OTHER plan's graph replay to read freed GPU
  // memory → NaN in rms_norm_linear (the VLM lm_logits NaN after plan swap).
  //
  // resetCastCacheIndices() resets the slot counter to 0 so this warmup
  // builds its cast entries from slot 0 — either reusing matching existing
  // entries or appending new ones — without freeing any NDArray that another
  // plan's CUDA graph may depend on. This mirrors the pre-capture warmup
  // pattern in NativeDynamicShapePlan_gpubackend.cu (line 3079) and
  // compositeReplay (line 2945) which deliberately avoid clearCastCache().
  MmulHelper::resetCastCacheIndices();

  // Reset ALL slot states to WARMUP and clear shape caches. The unfrozen
  // pass left slots in various states (SHAPE_CACHED, FROZEN, FROZEN_CONSTANT)
  // with cached shapes from prefill execution. Warmup re-executes with
  // decode-time inputs so shapes may differ. Without this reset, slots with
  // frozenContextReady()=true would use stale cached output shapes/types
  // from the unfrozen pass.
  for (int i = 0; i < numSlots_; i++) {
    if (slots_[i].slotPhase.shapeCacheValid || slots_[i].slotPhase.isSealed()) {
      slots_[i].slotPhase.reset();  // PRIMARY
      slots_[i].shapeCache.cachedOutputShapes.clear();
    }
  }

  // Execute all segments slot-by-slot to populate shapes
  int segIdx = 0;
  for (auto& segment : segments_) {
    DSP_DIAG(EXECUTE, "phaseWarmup: seg[%d] slots=[%d-%d] capturable=%d starting...",
             segIdx, segment.def.startSlot, segment.def.endSlot,
             static_cast<int>(segment.def.isCapturable));
    if (!platformBindSegmentDevice(segment)) {
      return Status::KERNEL_FAILURE;
    }
    platformMigrateSegmentInputs(segment, externalInputs, numExternalInputs);

    SegmentLifecycle::initSegmentPhase(segment.exec, segment.def.startSlot, segment.def.endSlot);

    auto tSegStart = executionTimingEnabled_ ? Clock::now() : Clock::time_point{};
    auto status = executeSegmentSlotBySlot(segment, externalInputs, numExternalInputs, stream);
    DSP_DIAG(EXECUTE, "phaseWarmup: seg[%d] slots=[%d-%d] completed status=%d",
             segIdx, segment.def.startSlot, segment.def.endSlot, static_cast<int>(status));
    segIdx++;
    if (status != Status::OK) return status;

    // Increment executionCount so that executeSegmentWithGraph sees exec >= 1
    // and proceeds to graph capture instead of repeating warmup.
    segment.exec.executionCount = 1;

    // Capture baseline shape and address keys for EMULATED_REPLAY segments.
    // Without these baselines, the first frozen replay computes keys and compares
    // against zeros — keys never match, needsArgRefresh() stays true, and the plan
    // is permanently stuck at SHAPES_FROZEN (postFreezeExecCount >= 2 but
    // segmentHasStablePointersForPlanPhase always returns false).
    // Note: computeSegmentShapeKey hashes small input values (<=32 elements) which
    // requires D2H sync, but this only happens once during warmup.
    if (segment.def.selectedBackend == SelectedBackend::EMULATED_REPLAY) {
      segment.exec.recordReplayBaselineKeys(
          computeSegmentShapeKey(segment, externalInputs, numExternalInputs),
          computeSegmentInputAddrKeyPortable(segment, externalInputs, numExternalInputs),
          "phase_warmup_emulated_replay");
    }

    // Note: We intentionally do NOT call computeSegmentShapeKey for non-EMULATED_REPLAY
    // segments here. It is extremely expensive (per-element D2H sync for small inputs)
    // and unnecessary during warmup. executeSegmentWithGraph will compute the
    // key when it actually needs it for graph capture.

    if (executionTimingEnabled_) {
      auto segUs = std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - tSegStart).count();
      slotBySlotUs += segUs;
      slotBySlotSegs++;
      slotBySlotSlots += segment.def.endSlot - segment.def.startSlot + 1;
    }

    platformCleanupMigratedInputs();
    auto postStatus = platformCheckPostSegment(segment);
    // Multi-GPU sharding: restore the plan-primary device + execution TLS after a secondary
    // segment (no-op for single-GPU / primary segments). Must run before the next segment binds.
    platformRestoreSegmentDevice();
    if (postStatus != Status::OK) return postStatus;
  }

  if (stats != nullptr) {
    stats->slotBySlotUs = slotBySlotUs;
    stats->slotBySlotSegs = slotBySlotSegs;
    stats->slotBySlotSlots = slotBySlotSlots;
  }

  // ── Dynamic-shape transitive propagation ─────────────────────────────────
  // After warmup execution, some slots may have been marked isDynamicShape=true
  // by the step3-warmup-reassign or fused-chain-warmup-reassign paths (shape
  // drift detected during this warmup execution vs. the prior SLOT_BY_SLOT pass).
  //
  // Propagate transitively: if slot S is dynamic, any downstream slot T whose
  // input comes from one of S's output slot indices is also dynamic. This handles
  // cases like slot 347 being a view of slot 346 (dynamic KV cache slice).
  //
  // Algorithm: build a set of "dynamic output slot indices" from direct markings,
  // then iterate slots in topological order (they are already topologically sorted
  // by construction) and propagate. One forward pass is sufficient.
  if (numSlots_ > 0) {
    // Build set of output slot indices produced by already-dynamic slots.
    // Using a flat bool array indexed by output slot index for O(1) lookup.
    std::vector<bool> isDynamicOutputSlot(totalOutputSlots_, false);
    for (int s = 0; s < numSlots_; s++) {
      if (slots_[s].flags.isDynamicShape) {
        for (int o = 0; o < slots_[s].wiring.numOutputs; o++) {
          int oi = slots_[s].wiring.outputSlotIndices[o];
          if (oi >= 0 && oi < totalOutputSlots_) {
            isDynamicOutputSlot[oi] = true;
          }
        }
      }
    }

    // Forward pass: mark downstream slots dynamic if any of their inputs
    // comes from a dynamic output slot index.
    int propagatedCount = 0;
    for (int s = 0; s < numSlots_; s++) {
      auto& sl = slots_[s];
      if (sl.flags.isDynamicShape) {
        // Already dynamic — just keep its output slots flagged.
        for (int o = 0; o < sl.wiring.numOutputs; o++) {
          int oi = sl.wiring.outputSlotIndices[o];
          if (oi >= 0 && oi < totalOutputSlots_) {
            isDynamicOutputSlot[oi] = true;
          }
        }
        continue;
      }
      // Check if any input comes from a dynamic output slot.
      bool inputIsDynamic = false;
      for (int i = 0; i < sl.wiring.numInputs && !inputIsDynamic; i++) {
        int src = sl.wiring.inputSourceIndices[i];
        if (src >= 0 && src < totalOutputSlots_ && isDynamicOutputSlot[src]) {
          inputIsDynamic = true;
        }
      }
      if (inputIsDynamic) {
        sl.flags.isDynamicShape = true;
        propagatedCount++;
        DSP_DIAG(SHAPE,
            "phaseWarmup: slot %d (%s) marked isDynamicShape=true (transitive propagation)",
            s, sl.ident.opName.c_str());
        // Mark this slot's outputs as dynamic too for downstream propagation.
        for (int o = 0; o < sl.wiring.numOutputs; o++) {
          int oi = sl.wiring.outputSlotIndices[o];
          if (oi >= 0 && oi < totalOutputSlots_) {
            isDynamicOutputSlot[oi] = true;
          }
        }
      }
    }

    // Some value-dependent shape ops are driven by a small plan-internal
    // control chain (for example equals/cast/concat feeding Where). Those
    // upstream control producers can legitimately rotate their buffers across
    // frozen executions even when the large downstream payload tensors remain
    // shape-stable. Mark the upstream control chain dynamic AFTER the forward
    // pass so we do not flood the dynamic flag into the large downstream data
    // path.
    int controlAncestorCount = 0;
    std::vector<uint8_t> visited(numSlots_, 0);
    std::vector<int> worklist;
    for (int s = 0; s < numSlots_; s++) {
      if (slots_[s].flags.outputShapeDependsOnInputValues) {
        worklist.push_back(s);
        visited[s] = 1;
      }
    }

    while (!worklist.empty()) {
      const int consumerStep = worklist.back();
      worklist.pop_back();
      const auto& consumer = slots_[consumerStep];

      for (int i = 0; i < consumer.wiring.numInputs; i++) {
        const int srcIdx = consumer.wiring.inputSourceIndices[i];
        NDArray* srcArr = dsp::resolveInputSourceArray(srcIdx, outputSlots_, totalOutputSlots_,
                                                       externalInputs, numExternalInputs);
        if (!isSmallIntegralControlArray(srcArr)) continue;
        if (srcIdx < 0) continue;  // external control tensors are tracked separately

        const int producerStep = dsp::findProducingStepForOutputSlot(slots_, numSlots_, srcIdx);
        if (producerStep < 0 || producerStep >= numSlots_) continue;

        if (!slots_[producerStep].flags.isDynamicShape) {
          slots_[producerStep].flags.isDynamicShape = true;
          controlAncestorCount++;
          DSP_DIAG(SHAPE,
              "phaseWarmup: slot %d (%s) marked isDynamicShape=true "
              "(plan-internal control ancestor of value-dependent slot %d (%s))",
              producerStep, slots_[producerStep].ident.opName.c_str(),
              consumerStep, consumer.ident.opName.c_str());
        }

        if (!visited[producerStep]) {
          visited[producerStep] = 1;
          worklist.push_back(producerStep);
        }
      }
    }

    int dynamicCount = 0;
    for (int s = 0; s < numSlots_; s++) {
      if (slots_[s].flags.isDynamicShape) dynamicCount++;
    }
    DSP_DIAG(SHAPE,
        "phaseWarmup: dynamic-shape classification done — %d slots total "
        "(%d direct + %d propagated + %d upstream-control) out of %d slots",
        dynamicCount,
        dynamicCount - propagatedCount - controlAncestorCount,
        propagatedCount, controlAncestorCount, numSlots_);
  }

  // ── Post-warmup dtype consistency validation ────────────────────────────
  // Verify that output slot array dtypes match their cached shape dtypes.
  // A mismatch here means shape inference produced one dtype but the actual
  // execution allocated/reused an array with a different dtype — typically
  // FLOAT32 placeholder contamination surviving through warmup, or a stale
  // output array from a prior lifecycle being reused via a dtype-blind check.
  if (DSP_DIAG_ENABLED(SHAPE)) {
    int dtypeMismatches = 0;
    for (int s = 0; s < numSlots_; s++) {
      auto& sl = slots_[s];
      if (sl.shapeCache.cachedOutputShapes.empty()) continue;
      for (int o = 0; o < sl.wiring.numOutputs && o < static_cast<int>(sl.shapeCache.cachedOutputShapes.size()); o++) {
        int oi = sl.wiring.outputSlotIndices[o];
        if (oi < 0 || oi >= totalOutputSlots_) continue;
        NDArray* arr = outputSlots_[oi];
        const LongType* cachedShape = sl.shapeCache.cachedOutputShapes[o];
        if (arr == nullptr || cachedShape == nullptr) continue;
        auto arrDt = arr->dataType();
        auto cachedDt = ArrayOptions::dataType(cachedShape);
        if (arrDt != cachedDt) {
          DSP_DIAG(SHAPE,
              "WARMUP_DTYPE_MISMATCH: slot %d (%s) output[%d] slotIdx=%d "
              "array dtype=%s != cached shape dtype=%s — output will have wrong type",
              s, sl.ident.opName.c_str(), o, oi,
              DataTypeUtils::asString(arrDt).c_str(),
              DataTypeUtils::asString(cachedDt).c_str());
          dtypeMismatches++;
        }
      }
    }
    if (dtypeMismatches > 0) {
      DSP_DIAG(SHAPE, "WARMUP_DTYPE_AUDIT: %d dtype mismatches detected between "
               "output arrays and cached shapes — these will cause StrictOps failures",
               dtypeMismatches);
    } else {
      DSP_DIAG(SHAPE, "WARMUP_DTYPE_AUDIT: all output dtypes consistent with cached shapes");
    }
  }

  // Ensure DataBuffers have the allocations they need BEFORE freezing.
  // Once frozen, allocatePrimary/allocateSpecial will throw.
  //
  // Weight buffers: need both primary+special (host-side access patterns).
  // Requested output slots: need both (Java reads primary via syncToPrimary).
  // Intermediate output slots: ONLY need special (device) — they are never
  //   read on host during graph replay. Skipping primary allocation for these
  //   saves ~50% of warmup GPU memory overhead (host mirrors waste address space
  //   and prevent the pool from reclaiming device memory).
  if (sd::graph::dspIsCudaBuild()) {
    std::vector<NDArray*> allocationPrepareReads;

    auto ensureFullAllocation = [&](NDArray* arr) {
      if (arr == nullptr) return;
      DataBuffer* db = arr->dataBuffer();
      if (db == nullptr || !db->isValid()) return;
      if (db->special() != nullptr && db->primary() == nullptr) {
        db->allocatePrimary();
      }
      if (db->primary() != nullptr && db->special() == nullptr) {
        allocationPrepareReads.push_back(arr);
      }
    };

    // Device-only: ensure special exists but do NOT allocate primary.
    auto ensureDeviceOnly = [&](NDArray* arr) {
      if (arr == nullptr) return;
      DataBuffer* db = arr->dataBuffer();
      if (db == nullptr || !db->isValid()) return;
      if (db->primary() != nullptr && db->special() == nullptr) {
        allocationPrepareReads.push_back(arr);
      }
      // If neither exists, nothing to do — slot wasn't used in warmup.
    };

    int weightEnsuredCount = 0;
    for (auto* arr : lastExternalInputsCopy_) {
      DataBuffer* db = arr != nullptr ? arr->dataBuffer() : nullptr;
      if (db != nullptr && protectedWeightBuffers_.count(db) > 0) {
        ensureFullAllocation(arr);
        weightEnsuredCount++;
      }
    }

    // Build set of requested output slot indices for O(1) lookup.
    std::unordered_set<int> requestedOutputSet;
    if (requestedOutputSlotIndices_ != nullptr) {
      for (int i = 0; i < numRequestedOutputs_; i++) {
        int si = requestedOutputSlotIndices_[i];
        if (si >= 0 && si < totalOutputSlots_) {
          requestedOutputSet.insert(si);
        }
      }
    }

    int outputEnsuredFull = 0;
    int outputEnsuredDeviceOnly = 0;
    if (outputSlots_ != nullptr) {
      for (int i = 0; i < totalOutputSlots_; i++) {
        if (outputSlots_[i] != nullptr && outputSlots_[i]->dataBuffer() != nullptr) {
          if (requestedOutputSet.count(i) > 0) {
            // Requested output — Java will syncToPrimary, need both allocations.
            ensureFullAllocation(outputSlots_[i]);
            outputEnsuredFull++;
          } else {
            // Intermediate — only device buffer needed for graph replay.
            ensureDeviceOnly(outputSlots_[i]);
            outputEnsuredDeviceOnly++;
          }
        }
      }
    }
    if (!allocationPrepareReads.empty()) {
      NDArray::prepareSpecialUse({}, allocationPrepareReads);
      NDArray::registerSpecialUse({}, allocationPrepareReads);
    }
    DSP_DIAG(MEMORY,
        "phaseWarmup: ensured allocations before freeze — "
        "weights=%d outputSlots(full)=%d outputSlots(deviceOnly)=%d",
        weightEnsuredCount, outputEnsuredFull, outputEnsuredDeviceOnly);
  }

  // Freeze all DataBuffers now that warmup execution has fully allocated them.
  // Track exactly the refs owned by this plan so teardown/rebind never guesses
  // from current slot state after identities and views have been nulled.
  replacePlanFrozenRefsForCurrentState(
      "phaseWarmup", protectedWeightBuffers_, outputSlots_, totalOutputSlots_,
      frozenProtectedRefBuffers_, frozenOutputRefBuffers_);

  // ── Buffer coloring: apply color assignments ────────────────────────────
  // At this point all slot buffers are allocated and frozen. Apply coloring
  // to replace per-slot buffers with shared color buffers.
  if (colorMap_.isComputed() && !colorMap_.isApplied() && colorMap_.numColoredSlots() > 0) {
    try {
      auto& pool = DspBufferPool::forCurrentDevice();
      int consolidated = colorMap_.apply(outputSlots_, slotOwnership_,
                                          planOwnedArrays_, pool);
      DSP_DIAG(MEMORY, "phaseWarmup: buffer coloring applied: consolidated=%d buffers, "
               "saved %zuMB", consolidated,
               colorMap_.estimatedBytesSaved() / (1024 * 1024));
    } catch (const std::exception& e) {
      DSP_DIAG(MEMORY, "phaseWarmup: buffer coloring apply failed: %s — ejecting", e.what());
      auto& pool = DspBufferPool::forCurrentDevice();
      colorMap_.eject(outputSlots_, slotOwnership_, planOwnedArrays_, pool);
      colorMap_.reset();
    }
  }

  return Status::OK;
}

void NativeDynamicShapePlan::phaseCompile(NDArray** externalInputs, int numExternalInputs) {
  DSP_REQUIRE_PLAN_PHASE_AT_MOST(PlanPhase::SHAPES_FROZEN, "phaseCompile");
  if (planLifecycle_.compilationDone) return;
  // Require at least one warmup execution so slot shape caches are populated.
  // Without shapes, Triton IR builds fail on cross-segment inputs.
  if (executeCount_ < 1) {
    DSP_DIAG(COMPILE, "phaseCompile: deferred (executeCount=%d, shapes not yet populated)", executeCount_);
    return;
  }
  DSP_DIAG(COMPILE, "phaseCompile: BEGIN segments=%d extInputs=%d", (int)segments_.size(), numExternalInputs);
  platformPrecompileSegments(externalInputs, numExternalInputs);
  planLifecycle_.compilationDone = true;
  DSP_DIAG(COMPILE, "phaseCompile: END (compilationDone=true sealed=1)");
}

void NativeDynamicShapePlan::recordMidExecutionCompile(int startSlot, int endSlot, const char* reason) {
  const int64_t count = midExecutionCompileCount_.fetch_add(1, std::memory_order_relaxed) + 1;
  DSP_DIAG(COMPILE,
           "COMPILE_VIOLATION seg[%d-%d]: %s "
           "(totalMidExecCompiles=%lld, executionCount=%d)",
           startSlot, endSlot, reason ? reason : "(no reason)",
           (long long)count, executeCount_);
}

Status NativeDynamicShapePlan::precompilePlan(NDArray** externalInputs, int numExternalInputs,
                                              void* stream) {
  DSP_DIAG(COMPILE,
           "precompilePlan: BEGIN segments=%d extInputs=%d frozen=%d executeCount=%d sealed=%d",
           (int)segments_.size(), numExternalInputs,
           planLifecycle_.isShapesFrozen() ? 1 : 0, executeCount_, planLifecycle_.compilationDone ? 1 : 0);

  if (planLifecycle_.compilationDone) {
    // Already sealed — the caller's contract is that precompilePlan resets the
    // mid-execution compile counter so the next measurement window starts at 0.
    resetMidExecutionCompileCount();
    DSP_DIAG(COMPILE,
             "precompilePlan: already sealed (compilationDone=1) — counter reset");
    return Status::OK;
  }

  // Populate external input ranks before phaseFreeze() — needed by FusionPass pass 5
  if (externalInputRanks_.empty() && numExternalInputs > 0) {
    externalInputRanks_.resize(numExternalInputs, -1);
    for (int i = 0; i < numExternalInputs; i++) {
      if (externalInputs[i] != nullptr)
        externalInputRanks_[i] = externalInputs[i]->rankOf();
    }
  }

  if (planLifecycle_.isSlotBySlot()) {
    DSP_DIAG(COMPILE, "precompilePlan: auto-freezing plan (phaseFreeze)");
    auto freezeStatus = phaseFreeze();
    if (freezeStatus != Status::OK) {
      DSP_DIAG(COMPILE, "precompilePlan: phaseFreeze FAILED status=%d", (int)freezeStatus);
      return freezeStatus;
    }
  }

  if (executeCount_ < 1) {
    DSP_DIAG(COMPILE, "precompilePlan: running phaseWarmup to populate shape caches");
    auto warmupStatus = phaseWarmup(externalInputs, numExternalInputs, stream, nullptr);
    if (warmupStatus != Status::OK) {
      DSP_DIAG(COMPILE, "precompilePlan: phaseWarmup FAILED status=%d", (int)warmupStatus);
      return warmupStatus;
    }
    incrementExecuteCount("warmup_done");  // phaseCompile's guard passes once counted.
  }

  phaseCompile(externalInputs, numExternalInputs);
  if (!planLifecycle_.compilationDone) {
    DSP_DIAG(COMPILE,
             "precompilePlan: phaseCompile did not reach sealed state "
             "(segments=%d executeCount=%d)",
             (int)segments_.size(), executeCount_);
    return Status::KERNEL_FAILURE;
  }

  // Reset the violation counter so callers get a clean baseline for the
  // measured window that follows precompile. Any mid-execution compile from
  // here on will increment it and fire a loud [COMPILE_VIOLATION] log.
  resetMidExecutionCompileCount();

  DSP_DIAG(COMPILE,
           "precompilePlan: END (sealed=1 segments=%d midExecCompiles=0 reset)",
           (int)segments_.size());
  return Status::OK;
}

// phaseSlotBySlot is now a thin wrapper around phaseReplay — the segment loop
// is shared. platformShouldUseGraph returns false for slot-by-slot modes, so
// phaseReplay routes all segments through executeSegmentSlotBySlot identically.
Status NativeDynamicShapePlan::phaseSlotBySlot(NDArray** externalInputs, int numExternalInputs,
                                               void* stream, PhaseExecutionStats* stats) {
  DSP_DIAG(EXECUTE, "phaseSlotBySlot: delegating to phaseReplay (unified segment loop)");
  return phaseReplay(externalInputs, numExternalInputs, stream, stats);
}

// ─── Shape inference only phase ───────────────────────────────────────────────
// Propagates shapes through the graph without executing any op kernels.
// For each slot: gather inputs → calculateOutputShape → allocate outputs → cache shapes.
// Skips: op execution, host/device sync, frozen detection, phase advancement,
// context pool management, view sharing, fused chain handling, KV scatter.

Status NativeDynamicShapePlan::phaseShapeInferenceOnly(
    NDArray** externalInputs, int numExternalInputs, void* stream) {
  DSP_DIAG(SHAPE, "phaseShapeInferenceOnly: BEGIN numSlots=%d extInputs=%d",
           numSlots_, numExternalInputs);

  // Thread-local scratch vectors to avoid per-slot heap allocation.
  static thread_local std::vector<NDArray*> siInputs;
  static thread_local std::vector<const LongType*> siOutputShapes;

  for (int stepIdx = 0; stepIdx < numSlots_; stepIdx++) {
    NativeSlot& slot = slots_[stepIdx];

    // ── Fused chain tails produce no independent output — skip ───────────
    if (slot.fusedChain.isFusedChainTail && !slot.fusedChain.isFusedChainHead) {
      continue;
    }

    // ── Data-dependent ops: skip — their shape functions read actual tensor
    // values (e.g., Where counts true elements, NonZero etc.). In the pre-pass
    // all internal arrays are zero-initialised, so these functions return wrong
    // shapes (e.g. numOfTrue=0) that corrupt every downstream slot's shape.
    // Leave outputSlots_ null for these slots; downstream ops will propagate
    // the skip via the null-input check below.
    if (slot.hasValueDependentShape()) {
      DSP_DIAG(SHAPE, "SHAPE_INFER_ONLY: slot %d (%s) has value-dependent shape — skipping shape pre-pass",
               stepIdx, slot.ident.opName.c_str());
      continue;
    }

    // ── Step 1: Gather inputs ────────────────────────────────────────────
    siInputs.resize(slot.wiring.numInputs);
    for (int i = 0; i < slot.wiring.numInputs; i++) {
      int srcIdx = slot.wiring.inputSourceIndices[i];
      if (srcIdx < 0) {
        int extIdx = -(srcIdx + 1);
        siInputs[i] = (extIdx >= 0 && extIdx < numExternalInputs)
                           ? externalInputs[extIdx] : nullptr;
      } else if (srcIdx >= 0 && srcIdx < totalOutputSlots_) {
        siInputs[i] = outputSlots_[srcIdx];
      } else {
        siInputs[i] = nullptr;
      }
    }

    // Validate inputs.  Three cases cause a skip:
    // 1. All inputs are null (op has no usable input at all).
    // 2. Any input sourced from an internal slot is null — that upstream slot
    //    was either data-dependent (skipped above) or itself downstream of one.
    //    Feeding a null input to calculateOutputShape would either crash (out-of-
    //    bounds on ShapeList) or produce a wrong shape.  Skip the whole slot so
    //    the null propagates cleanly to every consumer.
    // 3. Any input sourced from an external array is null — the caller did not
    //    provide the required input.  CRITICAL: phaseShapeInferenceOnly null-pads
    //    _fastpath_in to slot.wiring.numInputs so that block.width() reflects the
    //    full wired count.  If an EXTERNAL input[i] is null, _fastpath_in[i] is
    //    nullptr after padding.  An op's DECLARE_SHAPE_FN may then do:
    //      if (block.width() > i) INPUT_VARIABLE(i)->someMethod()
    //    which dereferences nullptr → NDArray::shapeInfo() with this=0x0 → SIGSEGV
    //    (si_addr=0x10 = offsetof(NDArray, _shapeInfoBuffer)).  Skip the slot when
    //    any external input is null to prevent reaching DECLARE_SHAPE_FN with a
    //    null fastpath entry.  Affected ops: reduce_mean_bp (INPUT_VARIABLE(2) →
    //    adjustAxis), gather (INPUT_VARIABLE(2)->e<LongType>(0)), and any op that
    //    checks block.width() > i before reading an optional tensor input.
    if (slot.wiring.numInputs > 0) {
      bool hasAnyInput = false;
      bool hasNullInternalInput = false;
      bool hasNullExternalInput = false;
      for (int i = 0; i < slot.wiring.numInputs; i++) {
        if (siInputs[i] != nullptr) {
          hasAnyInput = true;
        } else {
          int srcIdx = slot.wiring.inputSourceIndices[i];
          if (srcIdx >= 0) {
            // A null from an internal slot source means an upstream data-dependent
            // (or otherwise un-inferrable) op was skipped — propagate the skip.
            hasNullInternalInput = true;
          } else {
            // A null from an external source means the caller did not provide this
            // input.  The null-padded _fastpath_in[i] would make INPUT_VARIABLE(i)
            // return nullptr, and any op that dereferences it crashes (see case 3).
            hasNullExternalInput = true;
          }
        }
      }
      if (!hasAnyInput) {
        DSP_DIAG(SHAPE, "SHAPE_INFER_ONLY: slot %d (%s) has %d inputs but all are null — skipping",
                 stepIdx, slot.ident.opName.c_str(), slot.wiring.numInputs);
        continue;
      }
      if (hasNullInternalInput) {
        DSP_DIAG(SHAPE, "SHAPE_INFER_ONLY: slot %d (%s) has null internal input (upstream data-dep skipped) — skipping",
                 stepIdx, slot.ident.opName.c_str());
        continue;
      }
      if (hasNullExternalInput) {
        DSP_DIAG(SHAPE, "SHAPE_INFER_ONLY: slot %d (%s) has null external input (not provided by caller) — skipping"
                 " to prevent null INPUT_VARIABLE dereference in DECLARE_SHAPE_FN",
                 stepIdx, slot.ident.opName.c_str());
        continue;
      }
    }

    // ── Identity ops: output shape = input[0] shape ─────────────────────
    if (slot.isIdentityOp() && slot.wiring.numInputs >= 1 && siInputs[0] != nullptr) {
      DSP_DIAG(SHAPE, "SHAPE_PRE_PASS: slot %d (%s) IDENTITY — input[0] dtype=%s shape=%s",
               stepIdx, slot.ident.opName.c_str(),
               DataTypeUtils::asString(siInputs[0]->dataType()).c_str(),
               ShapeUtils::shapeAsString(siInputs[0]).c_str());
      for (int i = 0; i < slot.wiring.numOutputs; i++) {
        int slotIdx = slot.wiring.outputSlotIndices[i];
        if (slotIdx >= 0 && slotIdx < totalOutputSlots_) {
          outputSlots_[slotIdx] = siInputs[0];
        }
      }
      // Cache the identity shape + shape key for executeSlot cache hits
      slot.shapeCache.cachedShapeKey = computeShapeKey(slot, siInputs.data(), slot.wiring.numInputs);
      slot.shapeCache.cachedOutputShapes.resize(1);
      slot.shapeCache.cachedOutputShapes[0] = siInputs[0]->shapeInfo();
      if (!slot.slotPhase.shapeCacheValid) {
        slot.slotPhase.markShapeCached();  // PRIMARY
      }
      continue;
    }

    // ── Diagnostic: dump input dtypes for this slot ────────────────────
    DSP_DIAG(SHAPE, "SHAPE_PRE_PASS: slot %d (%s) numInputs=%d",
             stepIdx, slot.ident.opName.c_str(), slot.wiring.numInputs);
    for (int i = 0; i < slot.wiring.numInputs; i++) {
      if (siInputs[i] != nullptr) {
        int srcIdx = slot.wiring.inputSourceIndices[i];
        DSP_DIAG(SHAPE, "  input[%d] srcIdx=%d dtype=%s shape=%s",
                 i, srcIdx,
                 DataTypeUtils::asString(siInputs[i]->dataType()).c_str(),
                 ShapeUtils::shapeAsString(siInputs[i]).c_str());
      } else {
        DSP_DIAG(SHAPE, "  input[%d] srcIdx=%d NULL", i, slot.wiring.inputSourceIndices[i]);
      }
    }

    // ── Step 2: Build context with op arguments (for calculateOutputShape) ─
    Context ctx(stepIdx);
    for (int i = 0; i < slot.wiring.numInputs; i++) {
      if (siInputs[i] != nullptr) ctx.setInputArray(i, siInputs[i]);
    }
    // Ensure block.width() returns the full input count even when some inputs
    // are null (optional inputs like KV cache). Ops use block.width() to detect
    // optional inputs, so the count must match the op's declared input count.
    if (ctx.fastpath_in().size() < static_cast<size_t>(slot.wiring.numInputs)) {
      ctx.fastpath_in().resize(slot.wiring.numInputs, nullptr);
    }
    if (slot.args.numIArgs > 0) ctx.setIArguments(slot.args.iArgs, slot.args.numIArgs);
    if (slot.args.numTArgs > 0) ctx.setTArguments(slot.args.tArgs, slot.args.numTArgs);
    if (slot.args.numBArgs > 0) ctx.setBArguments(slot.args.bArgs, slot.args.numBArgs);
    if (slot.args.numDArgs > 0) ctx.setDArguments(slot.args.dArgs, slot.args.numDArgs);
    ctx.getSArguments()->clear();
    if (slot.args.numSArgs > 0) {
      ctx.getSArguments()->insert(ctx.getSArguments()->end(),
                                   slot.args.sArgs, slot.args.sArgs + slot.args.numSArgs);
    }
    // Legacy reduce/broadcast ops read reduction dims from block.getAxis().
    if (legacyOpReadsAxisFromIArgs(slot.legacy.legacyOpType)) {
      ctx.getAxis()->clear();
      for (int i = 0; i < slot.args.numIArgs; i++) {
        ctx.getAxis()->emplace_back(static_cast<sd::LongType>(slot.args.iArgs[i]));
      }
    }

    // ── Step 3: Calculate output shapes ──────────────────────────────────
    // Maintain positional alignment: every input index must have an entry so
    // that inputShape->at(i) in DECLARE_SHAPE_FN maps to the correct input.
    // Null inputs (optional, e.g. KV cache) get an empty shape info placeholder
    // whose dtype matches the first non-null input.
    DataType placeholderDtype = DataType::FLOAT32;
    for (int i = 0; i < slot.wiring.numInputs; i++) {
      if (siInputs[i] != nullptr) {
        placeholderDtype = siInputs[i]->dataType();
        break;
      }
    }
    ShapeList inputShapes;
    for (int i = 0; i < slot.wiring.numInputs; i++) {
      if (siInputs[i] != nullptr) {
        inputShapes.push_back(siInputs[i]->shapeInfo());
      } else {
        inputShapes.push_back(
            ConstantShapeHelper::getInstance().emptyShapeInfo(placeholderDtype));
      }
    }

    ShapeList* shapeList = nullptr;
    try {
      shapeList = slot.ident.op->calculateOutputShape(&inputShapes, ctx);
    } catch (const std::exception& e) {
      DSP_DIAG(SHAPE, "SHAPE_INFER_ONLY: slot %d (%s) calculateOutputShape EXCEPTION: %s",
               stepIdx, slot.ident.opName.c_str(), e.what());
      std::string errMsg = "shape inference failed at slot " + std::to_string(stepIdx) +
          " (" + slot.ident.opName + "): " + e.what();
      sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(errMsg.c_str());
      return Status::KERNEL_FAILURE;
    }
    if (shapeList == nullptr || shapeList->size() == 0) {
      DSP_DIAG(SHAPE, "SHAPE_INFER_ONLY: slot %d (%s) returned null/empty shape list",
               stepIdx, slot.ident.opName.c_str());
      std::string errMsg = "shape inference returned null at slot " + std::to_string(stepIdx) +
          " (" + slot.ident.opName + ")";
      sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(errMsg.c_str());
      return Status::KERNEL_FAILURE;
    }

    // ── Dtype contamination detection ────────────────────────────────────
    // If all real (non-null) inputs share a common dtype but the output dtype
    // differs, the shape function likely promoted via a FLOAT32 placeholder
    // for a null optional input.  Record the correct dtype so Step 4 below
    // can substitute corrected shape info when caching.
    //
    // Exempt ops that intentionally produce a different output dtype:
    //   - bool-output ops (equals, greater, less, etc.)
    //   - cast / reduce_long / argmax / shape_of / size / where etc.
    //   - ops with explicit DArg dtype override
    DataType commonInputDtype = DataType::UNKNOWN;
    bool allSameDtype = true;
    for (int i = 0; i < slot.wiring.numInputs; i++) {
      if (siInputs[i] == nullptr) continue;
      DataType dt = siInputs[i]->dataType();
      if (commonInputDtype == DataType::UNKNOWN) {
        commonInputDtype = dt;
      } else if (dt != commonInputDtype) {
        allSameDtype = false;
        break;
      }
    }
    // When all real inputs agree on dtype, no DArgs override, and the output
    // is a promoted floating type — flag it for correction in the cache step.
    bool needsDtypeCorrection = allSameDtype && commonInputDtype != DataType::UNKNOWN &&
                                 slot.args.numDArgs == 0;

    // ── Diagnostic: dump computed output shapes + dtypes ────────────────
    for (int i = 0; i < static_cast<int>(shapeList->size()); i++) {
      const LongType* outShape = shapeList->at(i);
      auto outDt = ArrayOptions::dataType(outShape);
      int outSlotIdx = (i < slot.wiring.numOutputs) ? slot.wiring.outputSlotIndices[i] : -1;
      DSP_DIAG(SHAPE, "  output[%d] outSlotIdx=%d dtype=%s rank=%d dims=[%s]",
               i, outSlotIdx,
               DataTypeUtils::asString(outDt).c_str(),
               shape::rank(outShape),
               ShapeUtils::shapeAsString(outShape).c_str());
    }

    // ── Step 4: Cache shapes + shape key ───────────────────────────────
    // Compute and cache the shape key so that executeSlot's cache-hit check
    // (cachedShapeKey == computeShapeKey) succeeds on the first real execution,
    // allowing it to skip redundant calculateOutputShape calls.
    slot.shapeCache.cachedShapeKey = computeShapeKey(slot, siInputs.data(), slot.wiring.numInputs);
    int numShapeOutputs = static_cast<int>(shapeList->size());
    siOutputShapes.resize(numShapeOutputs);
    slot.shapeCache.cachedOutputShapes.resize(numShapeOutputs);
    for (int i = 0; i < numShapeOutputs; i++) {
      const LongType* rawShape = shapeList->at(i);
      auto outDt = ArrayOptions::dataType(rawShape);

      // Correct placeholder-contaminated dtypes: if all real inputs are e.g.
      // HALF but the output is FLOAT32 (promoted via a FLOAT32 placeholder for
      // a null optional input), substitute a shape info with the correct dtype.
      const LongType* correctedShape = rawShape;
      if (needsDtypeCorrection && outDt != commonInputDtype &&
          DataTypeUtils::isR(outDt) && DataTypeUtils::isR(commonInputDtype)) {
        DSP_DIAG(SHAPE, "SHAPE_PREPASS_DTYPE_FIX: slot %d (%s) output[%d] dtype %s != "
                 "common input dtype %s — correcting (likely placeholder contamination)",
                 stepIdx, slot.ident.opName.c_str(), i,
                 DataTypeUtils::asString(outDt).c_str(),
                 DataTypeUtils::asString(commonInputDtype).c_str());
        correctedShape = ConstantShapeHelper::getInstance().createShapeInfo(
            commonInputDtype, shape::order(rawShape),
            shape::rank(rawShape), shape::shapeOf(const_cast<LongType*>(rawShape)));
      }

      auto cached = ConstantShapeHelper::getInstance().createFromExisting(
          const_cast<LongType*>(correctedShape));
      siOutputShapes[i] = cached;
      slot.shapeCache.cachedOutputShapes[i] = cached;
    }
    if (!slot.slotPhase.shapeCacheValid) {
      slot.slotPhase.markShapeCached();  // PRIMARY
    }
    delete shapeList;

    // ── Step 5: Allocate output arrays (shape + buffer, no compute) ──────
    int numWiredOutputs = slot.wiring.numOutputs;
    for (int i = 0; i < numShapeOutputs; i++) {
      int slotIdx = (i < numWiredOutputs) ? slot.wiring.outputSlotIndices[i] : -1;
      if (slotIdx < 0 || slotIdx >= totalOutputSlots_) continue;

      // Reuse existing array if shape already matches
      NDArray* existing = outputSlots_[slotIdx];
      if (existing != nullptr && existing->hasValidShapeInfo()) {
        if (shape::equalsSoft(existing->shapeInfo(), siOutputShapes[i]) &&
            ArrayOptions::dataType(existing->shapeInfo()) ==
                ArrayOptions::dataType(siOutputShapes[i])) {
          continue;  // Shape matches — no reallocation needed
        }
        // Shape changed — delete old array if plan-owned
        if (planOwnedArrays_.count(existing) > 0) {
          planOwnedArrays_.erase(existing);
          delete existing;
          if (sd::Environment::getInstance().isDebug()) {
            char siLabel[256];
            snprintf(siLabel, sizeof(siLabel),
                     "AFTER_shapeInference_delete_slot%d", slotIdx);
            scanAllSlotsForCorruption(outputSlots_, totalOutputSlots_,
                siLabel, executeCount_);
          }
        }
        outputSlots_[slotIdx] = nullptr;
      }

      // Allocate new array with the inferred shape
      NDArray* outArr = new NDArray(const_cast<LongType*>(siOutputShapes[i]), true);
      outputSlots_[slotIdx] = outArr;
      planOwnedArrays_.insert(outArr);
      DSP_DIAG(SHAPE, "SHAPE_PRE_PASS_ALLOC: slot %d (%s) outSlotIdx=%d allocated dtype=%s shape=%s",
               stepIdx, slot.ident.opName.c_str(), slotIdx,
               DataTypeUtils::asString(outArr->dataType()).c_str(),
               ShapeUtils::shapeAsString(outArr).c_str());
    }
  }

  DSP_DIAG(SHAPE, "phaseShapeInferenceOnly: END numSlots=%d", numSlots_);
  return Status::OK;
}

// ═══════════════════════════════════════════════════════════════════════════════
// dispatchSegment — consolidated segment execution dispatch
// ═══════════════════════════════════════════════════════════════════════════════
//
// Single entry point for ALL segment execution. Uses three state dimensions:
//   1. selectedBackend  — HOW the segment was classified at build time
//   2. segPhase         — WHERE in the lifecycle (BUILDING/SEALED/FAILED)
//   3. outcome          — WHY it executes this way (result of build process)
//
// Decision tree:
//   frozen constants → skip
//   terminal outcome (ZERO_KERNEL_SBS, NOT_FUSIBLE, COMPILE_FAILED) → slot-by-slot
//   GRAPH_REPLAY + SEALED → graph replay via backend
//   OOM_DEFERRED → retry check or slot-by-slot
//   PENDING → forward to backend-specific build/capture path

Status NativeDynamicShapePlan::dispatchSegment(
    GraphSegment& seg, NDArray** externalArrays, int numExt,
    void* stream, bool& usedGraph) {
  usedGraph = false;

  DSP_DIAG(EXECUTE,
           "dispatchSegment: seg[%d-%d] backend=%d phase=%s outcome=%s execCount=%d",
           seg.def.startSlot, seg.def.endSlot,
           static_cast<int>(seg.def.selectedBackend),
           seg.exec.segPhase.displayName(),
           segmentExecOutcomeName(seg.exec.outcome),
           seg.exec.executionCount);

  // ── Loud sync-override diagnostic ────────────────────────────────────────
  // If syncOverrideDepth_ > 0 while shapes are frozen or replaying, this
  // segment is forced slot-by-slot and CUDA graph replay cannot happen.
  if ((planLifecycle_.isShapesFrozen() || planLifecycle_.isReplaying()) && syncOverrideDepth_ > 0) {
    DSP_DIAG(EXECUTE, "DSP ERROR: syncOverrideDepth=%d during %s phase — segment [%d-%d] forced "
             "slot-by-slot, blocking CUDA graph replay",
             syncOverrideDepth_, planLifecycle_.displayName(),
             seg.def.startSlot, seg.def.endSlot);
  }

  // ── VALIDATION: detect invalid state combinations ──────────────────────
  // These throw hard errors — invalid states are bugs, not edge cases.

  // V1: SEALED + GRAPH_REPLAY but no replay handle = broken lifecycle
  if (seg.exec.outcome == SegmentExecOutcome::GRAPH_REPLAY &&
      seg.exec.segPhase.isSealed() &&
      !seg.exec.replayHandle &&
      !hasCompositeHandles(seg)) {
    DSP_THROW_SEG(EXECUTE, seg.def.startSlot,
                  "DISPATCH VALIDATION: seg[%d-%d] outcome=GRAPH_REPLAY and phase=SEALED "
                  "but no replay handle (monolithic or composite). Broken lifecycle — "
                  "markCaptured was called but handle was destroyed without invalidation.",
                  seg.def.startSlot, seg.def.endSlot);
  }

  // V2: SLOT_BY_SLOT backend should never have GRAPH_REPLAY outcome
  if (seg.def.selectedBackend == SelectedBackend::SLOT_BY_SLOT &&
      seg.exec.outcome == SegmentExecOutcome::GRAPH_REPLAY) {
    DSP_THROW_SEG(EXECUTE, seg.def.startSlot,
                  "DISPATCH VALIDATION: seg[%d-%d] backend=SLOT_BY_SLOT but outcome=GRAPH_REPLAY. "
                  "SLOT_BY_SLOT segments never capture graphs — outcome is inconsistent.",
                  seg.def.startSlot, seg.def.endSlot);
  }

  // V3: Mode contract violation — requires compilation but none was done
  {
    auto contract = ModeContract::forMode(graphExecutionMode_);
    auto* execCtx = static_cast<PlanExecutionContext*>(activeExecCtx_);
    bool dispatchedAsSlotBySlot = execCtx && (execCtx->forcedSlotBySlot || !execCtx->isReplay);
    if (!planLifecycle_.compilationDone && !contract.isSlotBySlot &&
        !dispatchedAsSlotBySlot && contract.requiresCompilation) {
      REQUIRE_TRUE(false, 0,
                   "DSP MODE VIOLATION: dispatchSegment entered with compilationDone=false "
                   "and mode=%d which requires compilation. The compile phase was never run.",
                   static_cast<int>(graphExecutionMode_));
    }
  }

  // ── 1. Frozen constants — skip entirely ─────────────────────────────────
  if (seg.def.allFrozenConstants && !planLifecycle_.isSlotBySlot()) {
    seg.exec.executionCount++;
    DSP_DIAG_SEG(EXECUTE, seg.def.startSlot,
                 "FROZEN_CONST_SKIP: seg[%d-%d] all %d slots are frozen constants — skipping",
                 seg.def.startSlot, seg.def.endSlot,
                 seg.def.endSlot - seg.def.startSlot + 1);
    return Status::OK;
  }

  // ── 1b. EMULATED_REPLAY backend — dispatch BEFORE terminal outcome check.
  // EMULATED_REPLAY manages its own lifecycle (WARMUP → CAPTURING → SEALED)
  // and sets outcome=ZERO_KERNEL_SBS when sealing. If the terminal outcome
  // check fires first, it bypasses executeSegmentEmulatedReplay entirely,
  // routing through executeSegmentSlotBySlot with SBS_ON_LC_STREAM (no staging,
  // raw ext arrays). This causes view-producer slots to wrap the raw
  // placeholder DataBuffer instead of the staging buffer. When the placeholder
  // is closed between executions, the view's DataBuffer becomes invalid and
  // getSlotOutput returns null.
  if (seg.def.selectedBackend == SelectedBackend::EMULATED_REPLAY) {
    auto status = executeSegmentEmulatedReplay(seg, externalArrays, numExt, stream);
    usedGraph = (status == Status::OK);
    return status;
  }

  // ── 2. Terminal outcomes — permanent slot-by-slot ───────────────────────
  // V4: ZERO_KERNEL_SBS must only appear on SEALED segments — lifecycle methods
  // enforce this invariant. If we see it on a non-sealed segment, the lifecycle
  // was bypassed somewhere.
  if (seg.exec.outcome == SegmentExecOutcome::ZERO_KERNEL_SBS && !seg.exec.segPhase.isSealed()) {
    DSP_THROW_SEG(EXECUTE, seg.def.startSlot,
        "DISPATCH VALIDATION: seg[%d-%d] outcome=ZERO_KERNEL_SBS but phase=%s",
        seg.def.startSlot, seg.def.endSlot, seg.exec.segPhase.displayName());
  }

  // Terminal outcomes (ZERO_KERNEL_SBS, NOT_FUSIBLE, COMPILE_FAILED) will
  // NEVER do graph replay. Route through performPreReplaySync with
  // SBS_ON_LC_STREAM — H2D only, no staging, no cross-stream.
  if (isTerminalOutcome(seg.exec.outcome)) {
    DSP_DIAG(EXECUTE,
             "dispatchSegment: seg[%d-%d] terminal outcome=%s — direct SBS (no staging)",
             seg.def.startSlot, seg.def.endSlot,
             segmentExecOutcomeName(seg.exec.outcome));

    auto* execCtx = static_cast<PlanExecutionContext*>(activeExecCtx_);
    if (execCtx != nullptr) {
      execCtx->execTarget = ExecTarget::SBS_ON_LC_STREAM;
    }
    externalArrays = performPreReplaySync(externalArrays, numExt, stream, "terminal_sbs");

    SyncOverride terminalSync(*this, "terminal_outcome_sbs");
    return executeSegmentSlotBySlot(seg, externalArrays, numExt, stream);
  }

  // ── 0. Unified sync + staging — for graph-capable paths only ───────────
  // Set ExecTarget to GRAPH_REPLAY for all graph-capable segments. The
  // specific target may be refined downstream (GRAPH_CAPTURE for pending
  // segments), but GRAPH_REPLAY is the correct default for sealed segments
  // entering graph replay, and for PENDING segments the staging is still
  // needed in case capture fires this step.
  {
    auto* execCtx = static_cast<PlanExecutionContext*>(activeExecCtx_);
    if (execCtx != nullptr) {
      execCtx->execTarget = ExecTarget::GRAPH_REPLAY;
    }
  }

  if (!externalInputIsVariable_.empty()) {
    for (int i : cachedVariableExtIndices_) {
      if (i >= 0 && i < numExt && externalArrays[i] != nullptr) {
        DSP_LIFECYCLE_EVENT(executeCount_, seg.def.startSlot, "DISPATCH_EXT_PRE_SYNC", externalArrays[i]);
      }
    }
  }

  externalArrays = performPreReplaySync(externalArrays, numExt, stream, "dispatchSegment");

  if (!externalInputIsVariable_.empty()) {
    for (int i : cachedVariableExtIndices_) {
      if (i >= 0 && i < numExt && externalArrays[i] != nullptr) {
        DSP_LIFECYCLE_EVENT(executeCount_, seg.def.startSlot, "DISPATCH_EXT_POST_SYNC", externalArrays[i]);
      }
    }
  }

  // ── 3. Sealed with GRAPH_REPLAY — steady-state replay ──────────────────
  if (seg.exec.outcome == SegmentExecOutcome::GRAPH_REPLAY &&
      seg.exec.segPhase.isSealed()) {
    usedGraph = true;
    return platformExecuteSegmentWithBackends(
        seg, externalArrays, numExt, stream, usedGraph);
  }

  // ── 4. OOM deferred — check retry window ───────────────────────────────
  if (seg.exec.outcome == SegmentExecOutcome::OOM_DEFERRED) {
    if (seg.exec.segPhase.oomRetryPending &&
        seg.exec.executionCount >= seg.exec.segPhase.oomRetryAfterExec) {
      SegmentLifecycle::markOomRetryFiring(seg.exec, seg.def.startSlot, seg.def.endSlot);
      // Fall through to PENDING handling below
    } else {
      DSP_DIAG(EXECUTE,
               "dispatchSegment: seg[%d-%d] OOM deferred, waiting for retry (exec=%d retryAfter=%d)",
               seg.def.startSlot, seg.def.endSlot,
               seg.exec.executionCount, seg.exec.segPhase.oomRetryAfterExec);
      SyncOverride oomSync(*this, "oom_deferred_sbs");
      return executeSegmentSlotBySlot(seg, externalArrays, numExt, stream);
    }
  }

  // ── 5. PENDING — still building (warmup/compile/capture) ───────────────
  // Note: EMULATED_REPLAY is handled above in step 1b, before terminal outcomes.

  if (platformShouldUseGraph(seg)) {
    return platformExecuteSegmentWithBackends(
        seg, externalArrays, numExt, stream, usedGraph);
  }

  // Slot-by-slot fallback — SyncOverride for warmup sync
  SegmentLifecycle::initSegmentPhase(seg.exec, seg.def.startSlot, seg.def.endSlot);
  {
    SyncOverride warmupSync(*this, "sbs_warmup");
    return executeSegmentSlotBySlot(seg, externalArrays, numExt, stream);
  }
}

Status NativeDynamicShapePlan::phaseReplay(NDArray** externalInputs, int numExternalInputs,
                                           void* stream, PhaseExecutionStats* stats) {
  DSP_DIAG(EXECUTE, "phaseReplay: BEGIN segments=%d extInputs=%d frozen=%d phase=%s execCount=%d",
           (int)segments_.size(), numExternalInputs, planLifecycle_.isShapesFrozen() ? 1 : 0,
           planLifecycle_.displayName(), executeCount_);

  // Sync policy is contract-driven via needsSync() — no mutable state to set.
  // ModeContract.forcesSyncOnFrozen is checked by needsSync() at each slot.
  {
    auto contract = ModeContract::forMode(graphExecutionMode_);
    DSP_DIAG(STREAM_SYNC,
             "phaseReplay sync policy: needsSync=%s reason=%s "
             "contract[forcesSyncOnFrozen=%d forceSyncDuringCapture=%d] "
             "mode=%s frozen=%d overrideDepth=%d execCount=%d",
             needsSync() ? "YES" : "NO", syncReason(),
             (int)contract.forcesSyncOnFrozen, (int)contract.forceSyncDuringCapture,
             ModeContract::modeName(static_cast<int>(graphExecutionMode_)), (int)planLifecycle_.isShapesFrozen(),
             syncOverrideDepth_, executeCount_);
  }

  // Mode contract validation moved to dispatchSegment() — checked per-segment.

  long long graphReplayUs = 0, slotBySlotUs = 0;
  int graphReplaySegs = 0, slotBySlotSegs = 0, graphReplaySlots = 0, slotBySlotSlots = 0;
  int cfLoopIterations = 0;
  cfLoopBackStep_ = -1;  // Reset at start of execution

  using Clock = std::chrono::high_resolution_clock;

  for (size_t segIdx = 0; segIdx < segments_.size(); segIdx++) {
    auto& segment = segments_[segIdx];
    if (!platformBindSegmentDevice(segment)) {
      return Status::KERNEL_FAILURE;
    }
    platformMigrateSegmentInputs(segment, externalInputs, numExternalInputs);

    auto tSegStart = executionTimingEnabled_ ? Clock::now() : Clock::time_point{};
    bool segUsedGraph = false;
    int segSlots = segment.def.endSlot - segment.def.startSlot + 1;

    // Consolidated dispatch — single entry point for all segment execution.
    {
      auto status = dispatchSegment(segment, externalInputs, numExternalInputs,
                                    stream, segUsedGraph);
      if (status != Status::OK) return status;
    }

    if (executionTimingEnabled_) {
      auto segUs = std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - tSegStart).count();
      if (segUsedGraph) {
        graphReplayUs += segUs;
        graphReplaySegs++;
        graphReplaySlots += segSlots;
      } else {
        slotBySlotUs += segUs;
        slotBySlotSegs++;
        slotBySlotSlots += segSlots;
      }
    }

    platformCleanupMigratedInputs();
    auto postStatus = platformCheckPostSegment(segment);
    // Multi-GPU sharding: restore the plan-primary device + execution TLS after a secondary
    // segment (no-op for single-GPU / primary segments). Must run before the next segment binds.
    platformRestoreSegmentDevice();
    if (postStatus != Status::OK) return postStatus;

    // ── Control flow loop-back across segments ──────────────────────────
    // NextIteration sets cfLoopBackStep_ to the target Merge step. After
    // the last segment containing a NextIteration for this loop executes,
    // we jump back to the segment containing that Merge.
    if (cfLoopBackStep_ >= 0) {
      // Check if there are more NextIteration segments ahead that belong
      // to the same loop (they target Merges near cfLoopBackStep_). We
      // must let ALL NextIterations execute before jumping back.
      bool moreNextItersAhead = false;
      for (size_t ahead = segIdx + 1; ahead < segments_.size(); ahead++) {
        auto& aheadSeg = segments_[ahead];
        for (int s = aheadSeg.def.startSlot; s <= aheadSeg.def.endSlot; s++) {
          if (slots_[s].cf.controlFlowType == CF_NEXT_ITERATION
              && slots_[s].cf.loopBackTarget >= 0) {
            moreNextItersAhead = true;
            break;
          }
        }
        if (moreNextItersAhead) break;
      }

      if (!moreNextItersAhead) {
        // All NextIterations have fired. Handle loop-back.
        cfLoopIterations++;
        if (cfLoopIterations >= MAX_LOOP_ITERATIONS) {
          DSP_DIAG(EXECUTE, "loop iteration limit (%d) reached at cfLoopBackStep_=%d",
                   MAX_LOOP_ITERATIONS, cfLoopBackStep_);
          return Status::VALIDATION;
        }

        int earliestMerge = cfLoopBackStep_;
        // Find the last NextIteration step to determine loop body range
        int lastNextIter = segment.def.endSlot;
        for (int s = numSlots_ - 1; s >= earliestMerge; s--) {
          if (slots_[s].cf.controlFlowType == CF_NEXT_ITERATION
              && slots_[s].cf.loopBackTarget >= 0) {
            lastNextIter = s;
            break;
          }
        }

        // Clear dead flags for the full loop body range so body ops re-execute
        if (slotIsDead_) {
          for (int s = earliestMerge; s <= lastNextIter && s < numSlots_; s++) {
            NativeSlot& bodySlot = slots_[s];
            for (int oi = 0; oi < bodySlot.wiring.numOutputs; oi++) {
              int si = bodySlot.wiring.outputSlotIndices[oi];
              if (si >= 0 && si < slotIsDeadSize_) slotIsDead_[si] = false;
            }
          }

          // Mark Enter outputs dead for ALL Merges in this loop so each
          // Merge picks the NextIteration value instead of the Enter value.
          for (int s = earliestMerge; s <= lastNextIter; s++) {
            if (slots_[s].cf.controlFlowType == CF_MERGE && slots_[s].wiring.numInputs >= 2) {
              int enterSrcIdx = slots_[s].wiring.inputSourceIndices[0];
              if (enterSrcIdx >= 0 && enterSrcIdx < slotIsDeadSize_) {
                slotIsDead_[enterSrcIdx] = true;
              }
            }
          }
        }

        // Find the segment containing the earliest Merge and jump back
        cfLoopBackStep_ = -1;
        for (size_t si = 0; si < segments_.size(); si++) {
          if (earliestMerge >= segments_[si].def.startSlot
              && earliestMerge <= segments_[si].def.endSlot) {
            segIdx = si - 1; // will be incremented by for-loop
            break;
          }
        }
        continue; // restart from target segment
      }
    }

    // Trace slot reporting (GPU only)
    platformTraceSlotValues(segment, stream, executeCount_);

    // NaN detection (gated behind tritonVerifyKernels)
    if ((planLifecycle_.isShapesFrozen() || planLifecycle_.isReplaying()) && Environment::getInstance().tritonVerifyKernels()) {
      bool nanDetected = false;
      for (int stepIdx = segment.def.startSlot; stepIdx <= segment.def.endSlot && !nanDetected; stepIdx++) {
        auto& slot = slots_[stepIdx];
        for (int o = 0; o < slot.wiring.numOutputs && !nanDetected; o++) {
          int si = slot.wiring.outputSlotIndices[o];
          if (si < 0 || si >= totalOutputSlots_ || outputSlots_[si] == nullptr) continue;
          auto* arr = outputSlots_[si];
          auto* db = arr->dataBuffer();
          if (db == nullptr || sd::graph::dspBuffer(arr) == nullptr || arr->lengthOf() == 0) continue;
          bool dbClosed = db->isClosed();
          if (dbClosed) {
            DSP_DIAG_SLOT(VERIFY, stepIdx, slot.ident.opName.c_str(),
                    "NaN_CLOSED_DB seg[%d-%d] outSlot=%d DataBuffer CLOSED! "
                    "frozenConst=%d shapeStatic=%d execCount=%d",
                    segment.def.startSlot, segment.def.endSlot, si,
                    slot.frozenConstantSlot() ? 1 : 0, slot.shapeCache.shapeStatic ? 1 : 0, executeCount_);
            continue;
          }
          DSP_DIAG_SLOT(VERIFY, stepIdx, slot.ident.opName.c_str(),
                  "NaN_CHECK_METADATA seg[%d-%d] output[%d]=%d usedGraph=%d "
                  "execCount=%d len=%lld hasReplay=%d frozenConst=%d shapeStatic=%d "
                  "asyncValues=true",
                  segment.def.startSlot, segment.def.endSlot, o, si,
                  segUsedGraph ? 1 : 0, executeCount_, (long long)arr->lengthOf(),
                  segment.exec.replayHandle != nullptr ? 1 : 0,
                  slot.frozenConstantSlot() ? 1 : 0, slot.shapeCache.shapeStatic ? 1 : 0);
        }
      }
    }
  }

  // Pool management and output copying are handled by execute() — no duplication here.

  // Timing breakdown
  if (executionTimingEnabled_) {
    DSP_DIAG(TIMING, "replay: graph=%lldus(%d segs/%d slots) sbs=%lldus(%d segs/%d slots)",
             graphReplayUs, graphReplaySegs, graphReplaySlots,
             slotBySlotUs, slotBySlotSegs, slotBySlotSlots);
  }

  if (stats != nullptr) {
    stats->graphReplayUs = graphReplayUs;
    stats->slotBySlotUs = slotBySlotUs;
    stats->graphReplaySegs = graphReplaySegs;
    stats->slotBySlotSegs = slotBySlotSegs;
    stats->graphReplaySlots = graphReplaySlots;
    stats->slotBySlotSlots = slotBySlotSlots;
  }

  // No mutable sync state to reset — needsSync() is computed from contract + state.

  return Status::OK;
}

void NativeDynamicShapePlan::clearShapeCaches() {
  // When shapes are frozen, skip clearing entirely after first execution.
  // All cached shapes remain valid since external input shapes are constant.
  if (!planLifecycle_.isSlotBySlot() && executeCount_ > 0) return;

  for (int i = 0; i < numSlots_; i++) {
    if (!slots_[i].shapeCache.shapeStatic) {
      slots_[i].shapeCache.cachedShapeKey = 0;
      slots_[i].shapeCache.cachedOutputShapes.clear();
      // Demote to WARMUP if currently beyond warmup (non-static slots need re-inference)
      if (slots_[i].slotPhase.shapeCacheValid || slots_[i].slotPhase.isSealed()) {
        slots_[i].slotPhase.reset();  // PRIMARY
      }
    }
  }
}

void NativeDynamicShapePlan::clearAllShapeCachesForce() {
  DSP_REQUIRE_PLAN_PHASE_AT_MOST(PlanPhase::SHAPES_FROZEN, "clearAllShapeCachesForce");
  for (int i = 0; i < numSlots_; i++) {
    slots_[i].shapeCache.cachedShapeKey = 0;
    slots_[i].shapeCache.cachedOutputShapes.clear();
    // Force demote all slots to WARMUP
    if (slots_[i].slotPhase.shapeCacheValid || slots_[i].slotPhase.isSealed()) {
      slots_[i].slotPhase.reset();  // PRIMARY
    }
  }
}

// ─── Reset segment execution state ──────────────────────────────────────────

void NativeDynamicShapePlan::resetSegmentExecutionState() {
  // Phase demotion is an architectural violation — destroy the plan and create
  // a fresh one instead. This method exists only as a hard error sentinel.
  DSP_THROW(FALLBACK,
           "[PHASE_VIOLATION] resetSegmentExecutionState called — "
           "phase demotion is prohibited. Destroy the plan and recreate.");
}

// ─── Passivation ────────────────────────────────────────────────────────────

size_t NativeDynamicShapePlan::passivate() {
  if (passivated_) return 0;
  size_t bytesBefore = estimatedOwnedBytes();
  DSP_DIAG(MEMORY, "CACHE_PASSIVATE plan=%p freeing %zuMB to pool", this,
           bytesBefore / (1024 * 1024));
  releaseGpuIntermediates();
  passivated_ = true;
  return bytesBefore;
}

void NativeDynamicShapePlan::reactivate() {
  if (!passivated_) return;
  DSP_DIAG(MEMORY, "CACHE_REACTIVATE plan=%p", this);
  passivated_ = false;
}

void NativeDynamicShapePlan::invalidateExternalViewSlotsOnReacquire() {
  if (slots_ == nullptr || outputSlots_ == nullptr) return;
  int cleared = 0;
  for (int s = 0; s < numSlots_; s++) {
    NativeSlot& slot = slots_[s];
    if (!slot.isViewCapableOp()) continue;
    if (slot.wiring.numInputs < 1 || slot.wiring.numOutputs < 1) continue;
    // Only views whose PRIMARY input is an external array (src < 0): those
    // wrap borrower-owned memory. Views over slot outputs wrap plan-owned
    // buffers and stay valid across borrowers.
    if (slot.wiring.inputSourceIndices[0] >= 0) continue;
    int si = slot.wiring.outputSlotIndices[0];
    if (si < 0 || si >= totalOutputSlots_) continue;
    if (outputSlots_[si] == nullptr) continue;
    // Null value passes the frozen-phase guard and runs the guarded old-array
    // disposal (isValid()-gated delete, graph-baked address pinning).
    writeOutputSlot(si, nullptr, "new-borrower-ext-view-invalidate");
    cleared++;
    // The cleared slot's segment must pass through warmup again so the view
    // re-mints BEFORE any REPLAYING-phase all-slots-populated validation runs
    // (otherwise: "null output slots in REPLAYING phase" on the first exec
    // after a borrower switch of a frozen plan).
    for (auto& seg : segments_) {
      if (s >= seg.def.startSlot && s <= seg.def.endSlot) {
        // resetForWarmup+markArgsStale alone re-warms CUDA-graph/JIT modes but
        // NOT the emulated replay handle — EMULATED_REPLAY kept "replaying"
        // past the freshly-nulled slot without re-minting the view
        // (longViewChain/EMULATED_REPLAY frozen_replay_5 null-slot residue
        // after the NVRTC/PTX siblings were fixed). Use the same full
        // segment-capture invalidation the weight-rebind path uses
        // (refreshProtectedWeightBuffers -> invalidateSegmentCaptures): it
        // covers CUDA-graph, JIT and emulated handles in one call.
        SegmentLifecycle::invalidateSegmentCaptures(this, seg, "new_borrower_ext_view");
        seg.exec.resetForWarmup();
        seg.exec.markArgsStale();
        break;
      }
    }
  }
  if (cleared > 0) {
    // A frozen/replaying plan with freshly-nulled slots must re-warm before
    // replay validation; segment resets above make anySegmentNeedsWarmup()
    // true, which the execute path already honors.
    DSP_DIAG(EXECUTE,
             "NEW_BORROWER: invalidated %d external-fed view slots plan=%p (segments reset to warmup)",
             cleared, (void*)this);
  }
}

// ─── Release GPU intermediates ───────────────────────────────────────────────


int NativeDynamicShapePlan::releaseGpuIntermediates() {
  DSP_DIAG(MEMORY, "releaseGpuIntermediates: START plan=%p numSlots=%d totalOutputSlots=%d",
           this, numSlots_, totalOutputSlots_);
  // Capture this before the teardown lifecycle reset below. Frozen refs are an
  // ownership fact established by the previous frozen/replay execution; after
  // reset(), planLifecycle_ can no longer answer whether refs were added.
  const bool hadFrozenRefsOnEntry =
      planLifecycle_.isInFrozenOrReplayState() ||
      hasTrackedPlanFrozenRefs(frozenProtectedRefBuffers_, frozenOutputRefBuffers_);

  // ── Buffer coloring: eject before teardown ─────────────────────────────
  // Undo coloring so each slot gets its own buffer before the deletion loop.
  // This prevents the deletion loop from double-freeing shared buffers.
  if (colorMap_.isApplied()) {
    auto& pool = DspBufferPool::forCurrentDevice();
    int restored = colorMap_.eject(outputSlots_, slotOwnership_, planOwnedArrays_, pool);
    DSP_DIAG(MEMORY, "releaseGpuIntermediates: ejected coloring, restored %d slots", restored);
    colorMap_.reset();
  }

  // ── Flush deferred slot deletes BEFORE any direct deletion ────────────────
  // writeOutputSlot() defers old-array deletes into tl_deferredSlotDeletes to
  // prevent heap corruption during active slot iteration. If the session is
  // torn down (destroySession → releaseGpuIntermediates) before the next
  // execute() flushes them, the deferred list still holds live pointers.
  // releaseGpuIntermediates then deletes those same arrays from outputSlots_,
  // and any later flush of the deferred list double-frees them (SIGSEGV on
  // 0xdeadbeefcafebabe poison value in ~NDArray).
  // Flush here so releaseGpuIntermediates is the sole owner of all deletions.
  flushDeferredSlotDeletes();

  // ── Flush graph-baked address pins ────────────────────────────────────────
  // Addresses pinned by writeOutputSlot to prevent pool reuse while baked into
  // live CUDA graphs must be released now that all segment GPU resources are
  // about to be destroyed. Use nullptr stream (stream 0) for the deferred free
  // since the plan-owned stream may already be scheduled for teardown.
  platformFlushGraphBakedPins(nullptr);

  // ── Phase demotion: demote to SLOT_BY_SLOT BEFORE freeing any arrays ──────
  // This ensures no code path can observe REPLAYING phase
  // while buffers are being freed, which would violate the phase contract
  // (those phases guarantee stable buffer pointers).
  {
    const char* oldPhaseName = planLifecycle_.displayName();
    static const char* reasonNames[] = {"NORMAL_CLOSE", "SESSION_RESET", "OOM_RECOVERY",
                                         "DEVICE_SWITCH", "CAPTURE_FAILURE", "SHAPE_CHANGE", "ERROR_RECOVERY"};
    const char* reasonName = reasonNames[static_cast<int>(destructionReason_)];
    if (!planLifecycle_.isSlotBySlot()) {
      DSP_DIAG(MEMORY, "[PHASE_TRANSITION] plan %s -> SLOT_BY_SLOT reason=releaseGpuIntermediates "
               "kind=teardown destructionReason=%s", oldPhaseName, reasonName);
      planLifecycle_.reset();
      frozenSnapshot_.clear();
    }
  }

  // ── Step 1: Free per-segment GPU resources (CUDA graphs, capture workspaces,
  //            pinned host pointers) ──────────────────────────────────────────
  // This is the same cleanup as the destructor's platformFreePlanResources(),
  // but we keep the segment metadata (slot ranges, op definitions) intact.
  platformReleaseSegmentGpuResources();

  // ── Step 2: Free non-weight NDArrays from outputSlots_ ─────────────────
  // Only free SLOT_OWNED buffers. Views and weights are externally owned.
  //
  //  Re-classify ownership before freeing. After CUDA graph capture,
  // outputSlots_[] is restored to warmup arrays (line 2716 in gpubackend.cpp)
  // but slotOwnership_[] still reflects capture-time classification. The warmup
  // arrays may have different ownership than the capture arrays:
  //   - Warmup array has unique buffer → should be SLOT_OWNED → must be freed
  //   - Capture array shared buffer with weight → was VIEW_OF_WEIGHT
  // Without re-classification, warmup arrays with unique buffers are skipped
  // (classified as VIEW_OF_WEIGHT from capture), leaking ~1.7 GB per page cycle.
  int freedCount = 0;
  std::unordered_set<NDArray*> deleted;
  bool frozenRefsReleasedForTeardown = false;
  auto releaseFrozenRefsForTeardown = [&]() {
    if (frozenRefsReleasedForTeardown) return;
    frozenRefsReleasedForTeardown = true;
    releasePlanFrozenRefsForTeardown("releaseGpuIntermediates", hadFrozenRefsOnEntry,
                                     frozenProtectedRefBuffers_, frozenOutputRefBuffers_);
  };

  if (outputSlots_) {
    // ── Build requested output protection set ────────────────────────────
    // Requested output slots (logits, etc.) must NEVER be freed during
    // releaseGpuIntermediates — Java may still hold references to their
    // DataBuffers via zeroCopyOutputCache or direct pointers. Also protect
    // any slot whose DataBuffer is shared with a requested output (views).
    std::unordered_set<int> requestedOutputSlotSet;
    std::unordered_set<DataBuffer*> requestedOutputDataBuffers;
    if (requestedOutputSlotIndices_ != nullptr) {
      for (int i = 0; i < numRequestedOutputs_; i++) {
        int si = requestedOutputSlotIndices_[i];
        if (si >= 0 && si < totalOutputSlots_) {
          requestedOutputSlotSet.insert(si);
          // Only dereference arrays whose object lifetime the plan controls.
          // Slots can alias externally-owned NDArrays (pass-through ops store
          // siInputs[0] directly); Java teardown may already have deleted those
          // objects, so ->dataBuffer() on them reads freed memory (SEGV_MAPERR
          // when the freed pages were returned to the OS — hs_err_pid1286526).
          // Non-plan-owned slots are protected by index alone.
          if (outputSlots_[si] != nullptr &&
              planOwnedArrays_.count(outputSlots_[si]) > 0 &&
              outputSlots_[si]->dataBuffer() != nullptr) {
            requestedOutputDataBuffers.insert(outputSlots_[si]->dataBuffer());
          }
        }
      }
      DSP_DIAG(MEMORY, "releaseGpuIntermediates: protecting %zu requested output slots "
               "(%zu unique DataBuffers)",
               requestedOutputSlotSet.size(), requestedOutputDataBuffers.size());
    }

    // Re-classify ownership for ALL slots based on the CURRENT outputSlots_[] arrays.
    // protectedWeightBuffers_ contains DataBuffers from ALL external inputs (built
    // during execute()). Any slot whose DataBuffer matches an external input is
    // BORROWED (model-owned, never freed by the plan). Everything else is an
    // intermediate (plan-owned, freed here).
    if (slotOwnership_) {
      for (int i = 0; i < totalOutputSlots_; i++) {
        slotOwnership_[i].reset();
        if (outputSlots_[i] == nullptr) continue;
        // Externally-owned arrays (not created by this plan) may already have been
        // deleted by the Java session teardown — dereferencing them is a UAF read.
        // Classify as VIEW_OF_WEIGHT (borrowed, never freed) WITHOUT touching the
        // object; pass 3 nulls the pointer. This is the same doctrine as pass 3's
        // "do NOT delete externally-owned wrappers" but applied to READS as well.
        if (planOwnedArrays_.count(outputSlots_[i]) == 0) {
          slotOwnership_[i].ownership = BufferOwnership::VIEW_OF_WEIGHT;
          continue;
        }
        auto* db = outputSlots_[i]->dataBuffer();
        if (db == nullptr) {
          slotOwnership_[i].ownership = BufferOwnership::UNSET;
          continue;
        }
        // Requested output DataBuffers are protected just like weight buffers.
        if (protectedWeightBuffers_.count(db) > 0 ||
            requestedOutputDataBuffers.count(db) > 0) {
          slotOwnership_[i].ownership = BufferOwnership::VIEW_OF_WEIGHT;
          slotOwnership_[i].dataBuffer = db;
          continue;
        }
        bool isViewOfSlot = false;
        for (int j = 0; j < i; j++) {
          // Same non-deref rule for the comparison target: borrowed slot arrays
          // may already be deleted externally — only read plan-owned ones.
          if (outputSlots_[j] != nullptr &&
              planOwnedArrays_.count(outputSlots_[j]) > 0 &&
              outputSlots_[j]->dataBuffer() == db) {
            slotOwnership_[i].ownership = BufferOwnership::VIEW_OF_SLOT;
            slotOwnership_[i].parentSlotIdx = j;
            slotOwnership_[i].dataBuffer = db;
            isViewOfSlot = true;
            break;
          }
        }
        if (!isViewOfSlot) {
          slotOwnership_[i].ownership = BufferOwnership::SLOT_OWNED;
          slotOwnership_[i].dataBuffer = db;
        }
      }
      DSP_DIAG(MEMORY, "releaseGpuIntermediates: re-classified ownership for %d slots "
               "(%zu protected external buffers, %zu protected output buffers)",
               totalOutputSlots_, protectedWeightBuffers_.size(),
               requestedOutputDataBuffers.size());
    }

    // Segment GPU resources are gone, so baked graph/slot addresses are no
    // longer live. Drop frozen refs before any identity-mutating cleanup below
    // (slot deletion, staging deletion, weight migration).
    releaseFrozenRefsForTeardown();

    if (slotOwnership_) {
      // First pass: null out all VIEW_OF_SLOT entries (they'll be invalidated
      // when their parent SLOT_OWNED buffer is freed).
      // SKIP requested output slots — their DataBuffer must persist.
      for (int i = 0; i < totalOutputSlots_; i++) {
        if (outputSlots_[i] != nullptr &&
            slotOwnership_[i].ownership == BufferOwnership::VIEW_OF_SLOT) {
          // Protect requested output slots and any slot sharing their DataBuffer.
          if (requestedOutputSlotSet.count(i) > 0 ||
              requestedOutputDataBuffers.count(outputSlots_[i]->dataBuffer()) > 0) {
            DSP_DIAG(MEMORY,
                     "releaseGpuIntermediates: PROTECTING VIEW_OF_SLOT slot %d "
                     "(requested output or shares DataBuffer with requested output)",
                     i);
            continue;
          }
          // Don't delete — the parent owns the buffer. Just null out.
          outputSlots_[i] = nullptr;
          slotOwnership_[i].reset();
        }
      }
      // Second pass: free SLOT_OWNED buffers that are plan-owned.
      // Only delete arrays the plan created (in planOwnedArrays_).
      // SKIP requested output slots — their DataBuffer must persist for Java.
      for (int i = 0; i < totalOutputSlots_; i++) {
        if (outputSlots_[i] != nullptr &&
            slotOwnership_[i].ownership == BufferOwnership::SLOT_OWNED) {
          // Protect requested output slots
          if (requestedOutputSlotSet.count(i) > 0) {
            DSP_DIAG(MEMORY,
                     "releaseGpuIntermediates: PROTECTING SLOT_OWNED slot %d "
                     "(requested output — DataBuffer %p must persist for Java)",
                     i, (void*)outputSlots_[i]->dataBuffer());
            continue;
          }
          // Also protect any slot whose DataBuffer is shared with a requested output
          if (requestedOutputDataBuffers.count(outputSlots_[i]->dataBuffer()) > 0) {
            DSP_DIAG(MEMORY,
                     "releaseGpuIntermediates: PROTECTING SLOT_OWNED slot %d "
                     "(DataBuffer %p shared with requested output)",
                     i, (void*)outputSlots_[i]->dataBuffer());
            continue;
          }
          slotOwnership_[i].viewRefCount = 0;
          if (planOwnedArrays_.count(outputSlots_[i]) > 0 &&
              deleted.insert(outputSlots_[i]).second) {
            planOwnedArrays_.erase(outputSlots_[i]);
            // Pre-clean the DataBuffer (mirror the destructor guard at ~1226-1233)
            // so a GC-freed buffer (MAGIC_DESTROYED) doesn't reach deleteBuffers()
            // → Workspace::allocateBytes SIGSEGV (this=0xDEADBEEFCAFEBABE).
            auto* db = outputSlots_[i]->dataBuffer();
            if (db != nullptr && db->isValid() && !db->isClosed()) {
              db->deleteBuffers();
            }
            outputSlots_[i]->setShapeInfo((sd::LongType*)nullptr);
            delete outputSlots_[i];
            freedCount++;
          }
          outputSlots_[i] = nullptr;
          slotOwnership_[i].reset();
        }
      }
    } else {
      // Fallback: no ownership info — only free plan-owned arrays.
      // Still protect requested output slots.
      for (int i = 0; i < totalOutputSlots_; i++) {
        if (outputSlots_[i] != nullptr && planOwnedArrays_.count(outputSlots_[i]) > 0) {
          // Protect requested output slots
          if (requestedOutputSlotSet.count(i) > 0) {
            DSP_DIAG(MEMORY,
                     "releaseGpuIntermediates: PROTECTING slot %d (requested output, fallback path)",
                     i);
            continue;
          }
          if (requestedOutputDataBuffers.count(outputSlots_[i]->dataBuffer()) > 0) {
            continue;
          }
          if (deleted.insert(outputSlots_[i]).second) {
            planOwnedArrays_.erase(outputSlots_[i]);
            // Pre-clean the DataBuffer (mirror the destructor guard at ~1226-1233)
            // so a GC-freed buffer (MAGIC_DESTROYED) doesn't reach deleteBuffers()
            // → Workspace::allocateBytes SIGSEGV (this=0xDEADBEEFCAFEBABE).
            auto* db = outputSlots_[i]->dataBuffer();
            if (db != nullptr && db->isValid() && !db->isClosed()) {
              db->deleteBuffers();
            }
            outputSlots_[i]->setShapeInfo((sd::LongType*)nullptr);
            delete outputSlots_[i];
            freedCount++;
          }
          outputSlots_[i] = nullptr;
        }
      }
    }

    // ── Pass 3: Null all remaining non-null outputSlots_ entries ──────────
    // After passes 1 and 2, any remaining non-null entries are VIEW_OF_WEIGHT
    // or UNSET-ownership arrays that reference external DataBuffers (model
    // weights, constants). These NDArray wrappers are valid NOW but become
    // dangling when the Java session destroys its weight arrays and the
    // plan is reused from cache for a new session. Null them here so the
    // next session starts with a clean slate — warmup will recreate all
    // arrays from scratch via op execution.
    // We do NOT delete these arrays: their DataBuffers are externally owned,
    // and the NDArray destructor would decrement a refcount that may belong
    // to a DataBuffer being freed concurrently by the Java session teardown.
    // The NDArray wrapper leak is negligible (~200 bytes each).
    {
      int nulledRemaining = 0;
      int firstNulled = -1;
      for (int i = 0; i < totalOutputSlots_; i++) {
        if (outputSlots_[i] != nullptr) {
          outputSlots_[i] = nullptr;
          if (firstNulled < 0) firstNulled = i;
          nulledRemaining++;
          if (slotOwnership_) {
            slotOwnership_[i].reset();
          }
        }
      }
      if (nulledRemaining > 0) {
        DSP_DIAG(MEMORY, "releaseGpuIntermediates: pass 3 nulled %d remaining "
                 "slots (VIEW_OF_WEIGHT / residual, first=%d plan=%p execCount=%d) "
                 "to prevent dangling pointers on plan cache reuse",
                 nulledRemaining, firstNulled, (void*)this, executeCount_);
      }
    }

    // Clear planOwnedArrays_ — all plan-created arrays are either freed or
    // orphaned (VIEW_OF_WEIGHT wrappers). Either way, the next session's
    // warmup will populate fresh entries.
    planOwnedArrays_.clear();
  }
  // If there were no output slots, protected external refs still need to be
  // dropped before platformMigrateWeightsAndClearCaches() replaces pointers.
  releaseFrozenRefsForTeardown();
  platformMigrateWeightsAndClearCaches();

  // ── Step 4b: Clear context pool output pointers ─────────────────────────
  // The context pool stores NDArray* in _fastpath_out that reference the
  // outputSlots_ arrays freed above. Clear them so no re-use of this plan
  // (from cache or otherwise) can dereference stale pointers via the
  // frozen fast-path.
  if (contextPool_ != nullptr) {
    int clearedCtxOutputs = 0;
    for (int si = 0; si < numSlots_; si++) {
      if (contextPool_[si] != nullptr && !contextPool_[si]->fastpath_out().empty()) {
        contextPool_[si]->fastpath_out().clear();
        clearedCtxOutputs++;
      }
    }
    if (clearedCtxOutputs > 0) {
      DSP_DIAG(MEMORY, "releaseGpuIntermediates: cleared context pool outputs for %d/%d slots",
               clearedCtxOutputs, numSlots_);
    }
  }

  // ── Step 4c: Clear ext input pointer caches ─────────────────────────────
  // The NDArray* pointers target Java-owned arrays that become invalid once
  // the Java session resets. They are rebuilt on the next execute() call.
  lastExternalInputsCopy_.clear();
  lastExternalInputs_ = nullptr;
  lastNumExternalInputs_ = 0;
  lastExternalInputAddrs_.clear();
  externalInputRanks_.clear();

  // ── Step 4d: Free placeholder staging buffers ──────────────────────────
  // These are plan-owned stable device buffers for variable external inputs,
  // allocated by ensureAndSyncStagingBuffers(). They survive across decode
  // steps but must be freed on session reset to reclaim GPU memory. They are
  // re-allocated lazily on the next executeSteadyState() / execute() call.
  DSP_DIAG(MEMORY, "releaseGpuIntermediates: Step 4d staging buffers=%p numExtInputs=%d",
           (void*)placeholderStagingBuffers_, numExternalInputs_);
  if (placeholderStagingBuffers_ != nullptr) {
    int freedStaging = 0;
    for (int i = 0; i < numExternalInputs_; i++) {
      DSP_DIAG(MEMORY, "releaseGpuIntermediates: staging[%d] ptr=%p",
               i, (void*)placeholderStagingBuffers_[i]);
      if (placeholderStagingBuffers_[i] != nullptr) {
        auto* db = placeholderStagingBuffers_[i]->dataBuffer();
        bool dbClosed = (db != nullptr && db->isClosed());
        DSP_DIAG(MEMORY, "releaseGpuIntermediates: staging[%d] db=%p closed=%d",
                 i, (void*)db, dbClosed);

        // Pre-clean GPU memory AND null _shapeInfo before deleting the NDArray.
        // Same rationale as the destructor staging cleanup: _shapeInfo may be dangling.
        if (db != nullptr && db->isValid() && !db->isClosed()) {
          db->deleteBuffers();
        }
        placeholderStagingBuffers_[i]->setShapeInfo((sd::LongType*)nullptr);
        delete placeholderStagingBuffers_[i];
        placeholderStagingBuffers_[i] = nullptr;
        freedStaging++;
      }
    }
    delete[] placeholderStagingBuffers_;
    placeholderStagingBuffers_ = nullptr;
    DSP_DIAG(MEMORY, "releaseGpuIntermediates: freed %d placeholder staging buffers", freedStaging);
  }
  if (effectiveExternals_ != nullptr) {
    delete[] effectiveExternals_;
    effectiveExternals_ = nullptr;
  }
  cachedVariableExtIndices_.clear();
  variableExternalInputIndices_.clear();
  variableIndicesCached_ = false;

  // ── Step 4e: Free untracked output cache ────────────────────────────────
  // The untrackedOutputCache_ holds NDArrays created during slot execution
  // that are NOT tracked in planOwnedArrays_ (e.g., multi-output ops).
  // Without cleaning them here, the destructor (called later from
  // NativePlanCache::clear) will try to delete NDArrays whose DataBuffers
  // may have already been closed by Java's destroySession() → SIGSEGV.
  if (untrackedOutputCache_ != nullptr) {
    int freedUntracked = 0;
    for (int i = 0; i < untrackedOutputCacheSize_; i++) {
      if (untrackedOutputCache_[i] != nullptr) {
        auto* db = untrackedOutputCache_[i]->dataBuffer();
        if (db != nullptr && db->isValid() && !db->isClosed()) {
          db->deleteBuffers();
        }
        untrackedOutputCache_[i]->setShapeInfo((sd::LongType*)nullptr);
        delete untrackedOutputCache_[i];
        untrackedOutputCache_[i] = nullptr;
        freedUntracked++;
      }
    }
    delete[] untrackedOutputCache_;
    untrackedOutputCache_ = nullptr;
    untrackedOutputCacheSize_ = 0;
    DSP_DIAG(MEMORY, "releaseGpuIntermediates: freed %d untracked output cache entries", freedUntracked);
  }

  // ── Step 5: Reset execution state so plan re-warms on next execute() ────
  viewProducerDetectionDone_ = false;
  frozenConstantDetectionDone_ = false;
  resetExecuteCount("release_gpu_intermediates");
  planLifecycle_.compilationDone = false;
  shapePrePassDone_ = false;
  // Frozen refs were released before identity-mutating teardown. Keep this call
  // as an idempotent guard for future releaseGpuIntermediates() edits.
  releaseFrozenRefsForTeardown();
  // ── Teardown-only lifecycle reset ──────────────────────���─────────────────
  // This is the ONLY legitimate place that resets planLifecycle_ to its initial state.
  // It is NOT an "unfreeze" — it is a cold reset after full GPU resource teardown.
  // The public setShapesFrozen(false) API is BANNED (throws LIFECYCLE VIOLATION).
  // After this reset, the plan is cold and can be re-warmed by the plan cache for
  // the same shape. Without this, stale segment state (shapeKey, cachedShapeKey)
  // causes Triton recompilation to be skipped, and the plan tries to replay
  // CUDA graphs that were destroyed, leading to error 700.
  planLifecycle_.reset();
  // Reset CUDA-only gap caches (no-op on CPU)
  platformResetGapCaches();

  // Clear protected weight buffers so they're rebuilt from the next session's
  // external inputs. Stale DataBuffer pointers from the old session would cause
  // incorrect ownership classification and lifecycle filtering on reuse.
  protectedWeightBuffers_.clear();
  frozenProtectedRefBuffers_.clear();
  frozenOutputRefBuffers_.clear();

  // Clear shape caches so shapes are re-inferred
  clearAllShapeCachesForce();

  // Clear GPU backend failed-segment cache
  clearGpuBackendFailedCache();

  DSP_DIAG(MEMORY, "releaseGpuIntermediates: DONE plan=%p, freed %d arrays. "
           "Plan is now cold — next execute() will re-warm.", this, freedCount);

  return freedCount;
}

void NativeDynamicShapePlan::setOutputSlotMaxSizes(const int* slotIndices, const LongType* maxSizes, int numSlots) {
  if (slotIndices == nullptr || maxSizes == nullptr || numSlots <= 0) return;

  outputSlotMaxSizes_.clear();
  maxAllocatedSlots_.clear();

  int configuredCount = 0;
  int expandedCount = 0;
  for (int i = 0; i < numSlots; i++) {
    if (slotIndices[i] >= 0 && slotIndices[i] < totalOutputSlots_ && maxSizes[i] > 0) {
      int slotIdx = slotIndices[i];
      LongType maxElements = maxSizes[i];
      outputSlotMaxSizes_[slotIdx] = maxElements;
      configuredCount++;

      NDArray* existing = outputSlots_ != nullptr ? outputSlots_[slotIdx] : nullptr;
      if (existing != nullptr && existing->dataBuffer() != nullptr) {
        LongType currentElements = existing->lengthOf();
        LongType effectiveMaxElements = std::max(maxElements, currentElements);
        outputSlotMaxSizes_[slotIdx] = effectiveMaxElements;

        size_t maxBytes = static_cast<size_t>(effectiveMaxElements) * existing->sizeOfT();
        auto* db = existing->dataBuffer();
        size_t beforeBytes = db->getLenInBytes();
        if (maxBytes > beforeBytes) {
          db->expand(maxBytes);
          expandedCount++;
        }
        maxAllocatedSlots_.insert(slotIdx);

        DSP_DIAG(MEMORY,
                 "setOutputSlotMaxSizes: slot=%d currentElements=%lld maxElements=%lld "
                 "bytesBefore=%zu bytesAfter=%zu",
                 slotIdx,
                 static_cast<long long>(currentElements),
                 static_cast<long long>(effectiveMaxElements),
                 beforeBytes,
                 db->getLenInBytes());
      }
    }
  }
  DSP_DIAG(MEMORY,
           "setOutputSlotMaxSizes: configured=%d expandedExisting=%d totalOutputSlots=%d",
           configuredCount, expandedCount, totalOutputSlots_);
}

// ─── Native KV scatter post-execution ───────────────────────────────────────

void NativeDynamicShapePlan::configureKvScatter(const int* presentSlotIndices,
                                                 NDArray** staticKvBuffers,
                                                 int numPairs,
                                                 DataType dtype,
                                                 LongType heads,
                                                 LongType srcSeqLen,
                                                 LongType dstSeqLen,
                                                 LongType dim,
                                                 LongType* kvPositionDevice) {
  if (presentSlotIndices == nullptr || staticKvBuffers == nullptr ||
      numPairs <= 0 || kvPositionDevice == nullptr) {
    DSP_DIAG(KV_CACHE, "NativeDynamicShapePlan::configureKvScatter: invalid arguments");
    return;
  }

  kvScatterEntries_.clear();
  kvScatterEntries_.reserve(numPairs);

  for (int i = 0; i < numPairs; i++) {
    int slotIdx = presentSlotIndices[i];
    if (slotIdx < 0 || slotIdx >= totalOutputSlots_) {
      DSP_DIAG(KV_CACHE,
               "NativeDynamicShapePlan::configureKvScatter: slot index %d out of range [0, %d)",
               slotIdx, totalOutputSlots_);
      continue;
    }
    if (staticKvBuffers[i] == nullptr) {
      DSP_DIAG(KV_CACHE, "NativeDynamicShapePlan::configureKvScatter: staticKvBuffers[%d] is null", i);
      continue;
    }

    NativeKvScatterEntry entry;
    entry.presentSlotIdx = slotIdx;
    entry.staticBuf = staticKvBuffers[i];
    entry.heads = heads;
    entry.srcSeqLen = srcSeqLen;
    entry.dstSeqLen = dstSeqLen;
    entry.dim = dim;
    kvScatterEntries_.push_back(entry);
  }

  kvScatterDtype_ = dtype;
  kvPositionDevice_ = kvPositionDevice;
  kvScatterConfigured_ = !kvScatterEntries_.empty();

  DSP_DIAG(EXECUTE, "configureKvScatter: %d entries configured dtype=%d heads=%lld srcSeq=%lld dstSeq=%lld dim=%lld",
           (int)kvScatterEntries_.size(), (int)dtype,
           (long long)heads, (long long)srcSeqLen, (long long)dstSeqLen, (long long)dim);
}

void NativeDynamicShapePlan::resetKvCachePosition(LongType position) {
  if (!kvScatterConfigured_ || kvPositionDevice_ == nullptr) return;
  *kvPositionDevice_ = position;
}

LongType NativeDynamicShapePlan::getKvCachePosition() const {
  if (!kvScatterConfigured_ || kvPositionDevice_ == nullptr) return -1LL;
  return *kvPositionDevice_;
}

void NativeDynamicShapePlan::executeKvScatterPostExec(void* stream) {
  if (!kvScatterConfigured_ || kvScatterEntries_.empty() || kvPositionDevice_ == nullptr) return;

  // Build dynamic scatter entries from current output slot state
  std::vector<sd::ops::helpers::KvScatterDynEntry> dynEntries;
  dynEntries.reserve(kvScatterEntries_.size());

  LongType currentPos = *kvPositionDevice_;

  for (auto& entry : kvScatterEntries_) {
    NDArray* present = outputSlots_[entry.presentSlotIdx];
    if (present == nullptr) {
      DSP_DIAG(EXECUTE, "executeKvScatterPostExec: present slot %d is null — skipping",
               entry.presentSlotIdx);
      continue;
    }

    sd::ops::helpers::KvScatterDynEntry dynEntry;
    dynEntry.srcPtr = sd::graph::dspBuffer(present);
    dynEntry.dstPtr = sd::graph::dspBuffer(entry.staticBuf);
    dynEntry.kvPosPtr = kvPositionDevice_;
    dynEntry.heads = entry.heads;
    // Use actual present tensor's seqLen (may differ from configured srcSeqLen in edge cases)
    dynEntry.srcSeqLen = present->rankOf() >= 3 ? present->sizeAt(2) : entry.srcSeqLen;
    dynEntry.dstSeqLen = entry.dstSeqLen;
    dynEntry.dim = entry.dim;
    dynEntry.lastPos = dynEntry.srcSeqLen - 1;

    dynEntries.push_back(dynEntry);
  }

  if (dynEntries.empty()) return;

  // Validate position is in range
  LongType maxPos = kvScatterEntries_[0].dstSeqLen;
  if (currentPos < 0 || currentPos >= maxPos) {
    DSP_DIAG(EXECUTE, "executeKvScatterPostExec: cachePos=%lld out of range [0, %lld) — skipping",
             (long long)currentPos, (long long)maxPos);
    return;
  }

  DSP_DIAG(EXECUTE, "executeKvScatterPostExec: scatter %d pairs at cachePos=%lld",
           (int)dynEntries.size(), (long long)currentPos);

  auto* ctx = sd::LaunchContext::defaultContext();
  sd::ops::helpers::kvScatterDynBatched(dynEntries.data(), static_cast<int>(dynEntries.size()),
                                         kvScatterDtype_, ctx);

  // Tick actuality on static KV buffers — the scatter kernel wrote to device memory
  // directly without registerSpecialUse. Without tickWriteDevice, the DataBuffer's
  // isSpecialActual() stays false, and subsequent syncToSpecial calls would no-op
  // (or worse, overwrite valid device data with stale host zeros).
  for (auto& entry : kvScatterEntries_) {
    if (entry.staticBuf != nullptr && entry.staticBuf->dataBuffer() != nullptr) {
      entry.staticBuf->tickWriteDevice();
    }
  }

  // Advance the position counter by 1
  (*kvPositionDevice_)++;
}

void NativeDynamicShapePlan::releaseKvScatterResources() {
  // Note: kvPositionDevice_ is owned by the caller (Java side), not by the plan.
  // The plan does NOT free it — it's managed by the Java UnifiedKvCacheManager
  // or whoever called configureKvScatter. We just clear the pointer.
  kvScatterEntries_.clear();
  kvPositionDevice_ = nullptr;
  kvScatterConfigured_ = false;
}

// ─── Backend resolution (one-time, at segment build) ────────────────────────

SelectedBackend NativeDynamicShapePlan::resolveBackendForSegment(bool isCapturable) const {
  if (!isCapturable) {
    // Non-capturable segments (control flow only) must use slot-by-slot.
    // For modes that ban fallback, this is a compile-time error — the plan
    // should not contain non-capturable segments without control flow.
    return SelectedBackend::SLOT_BY_SLOT;
  }

  switch (graphExecutionMode_) {
    case GraphExecutionMode::GEM_SLOT_BY_SLOT:
      return SelectedBackend::SLOT_BY_SLOT;

    case GraphExecutionMode::GEM_CUDA_GRAPHS:
    case GraphExecutionMode::GEM_HIP_GRAPHS:
    case GraphExecutionMode::GEM_LEVELZERO:
    case GraphExecutionMode::GEM_VULKAN:
    case GraphExecutionMode::GEM_METAL:
      return platformResolveBackend(true);

    case GraphExecutionMode::GEM_TRITON:
    case GraphExecutionMode::GEM_NVRTC_JIT:
    case GraphExecutionMode::GEM_PTX_JIT:
    case GraphExecutionMode::GEM_TPU:
    case GraphExecutionMode::GEM_HEXAGON:
      return platformResolveBackend(false);

    case GraphExecutionMode::GEM_MLX:
    case GraphExecutionMode::GEM_ARM_HYBRID:
    case GraphExecutionMode::GEM_NNAPI:
      return SelectedBackend::CPU_GRAPH;

    case GraphExecutionMode::GEM_EMULATED_REPLAY:
      return SelectedBackend::EMULATED_REPLAY;

    case GraphExecutionMode::GEM_AUTO: {
      return platformResolveBackend(false);
    }

    default:
      return SelectedBackend::SLOT_BY_SLOT;
  }
}

// ─── Freeze-time resegmentation ──────────────────────────────────────────────
//
// After shapes freeze, existing CUDA graph handles from warmup captures are
// invalid (pointer addresses may have changed). This cleans up GPU resources
// and rebuilds segments so fresh captures can occur with frozen shapes.

void NativeDynamicShapePlan::resegmentForFreeze() {
  if (!Environment::getInstance().dspFreezeMergeSegments()) return;
  int oldSegCount = static_cast<int>(segments_.size());
  if (oldSegCount <= 1) return;

  // Cleanup GPU resources (CUDA graph handles etc.) from existing segments
  for (auto& seg : segments_) {
    if (seg.exec.replayHandle) {
      platformCleanupSegmentForRebuild(seg);
    }
  }
  segments_.clear();
  nativeRangeSegments_.clear();

  buildSegments();

  DSP_DIAG(SEGMENT, "RESEGMENT: %d -> %d segments (shapes frozen)",
           oldSegCount, static_cast<int>(segments_.size()));
}

// ─── Graph segmentation for GPU graph capture ───────────────────────────────

void NativeDynamicShapePlan::buildSegments() {
  if (numSlots_ == 0) {
    DSP_DIAG(SEGMENT, "buildSegments: skipped (numSlots=0)");
    return;
  }
  DSP_DIAG(SEGMENT, "buildSegments: BEGIN numSlots=%d matmulSeg=%d",
           numSlots_, Environment::getInstance().dspMatmulSegmentation() ? 1 : 0);

  // Segmentation policy:
  //
  // Merge as many consecutive slots as possible into each capturable segment.
  // Each contiguous capturable run (with the same device) becomes ONE segment.
  // At runtime, if a segment's shapes are stable it gets captured once and
  // replayed every step. If shapes change, the segment recompiles via the
  // shape key cache — no physical splitting needed.
  //
  // Capturability: a slot is capturable iff:
  //   1. It is NOT data-dependent (where/unique/nms produce variable-length output)
  //   2. Value-dep-shape ops (reshape/concat/gather whose output SHAPE depends on
  //      runtime VALUES) are now capturable — computeSegmentShapeKey hashes actual
  //      data values of small inputs (≤32 elements), so value changes are detected.
  //      Segments containing these ops have hasValueDepOps=true, which forces shape
  //      key recomputation even when shapes are frozen.

  // Most ops are capturable. The shapeKey system handles dynamic shapes:
  // - computeSegmentShapeKey hashes input values for small arrays
  // - hasValueDepOps forces recomputation even when frozen
  // - cache miss triggers recompilation with correct shapes
  // EXCEPTION: Data-dependent ops (Where/1-input, unique, nms) require host
  // sync during execution to count/find variable-length elements. This host
  // sync invalidates CUDA graph capture (error 901). These ops must be in
  // their own non-capturable segment, executed slot-by-slot.
  // Segment topology is built ONCE and never changes. Only truly uncapturable
  // ops (data-dependent, control flow) create segment boundaries.
  // Matmul segmentation: break segments at matmul/attention op boundaries.
  // This isolates element-wise chains for Triton fusion while matmuls run via cuBLAS.
  const bool matmulSegmentation = Environment::getInstance().dspMatmulSegmentation();

  // Segments are built once — use a large cap. Memory-budget splitting
  // (below) handles CUDA_GRAPHS monolithic capture constraints.
  const int MAX_SEGMENT_SIZE = 100000;

  // ── Memory-budget-aware segment splitting for CUDA_GRAPHS ──────────────
  //
  // CUDA_GRAPHS mode captures the ENTIRE segment into a single CUDA graph.
  // All intermediate buffers must exist simultaneously during capture. For
  // large models (Qwen3.5 1.7B: 28 layers × ~50 ops = ~1400 slots), the
  // combined buffer footprint can exceed GPU free memory.
  //
  // Instead of a hard op-count limit, query actual GPU free memory and
  // track cumulative output buffer sizes per segment. Split when the
  // cumulative size exceeds the available budget. This adapts to:
  //   - Different GPU sizes (24GB, 48GB, 80GB)
  //   - Different model sizes (more/fewer weights → more/less free memory)
  //   - Runtime memory pressure (other processes consuming GPU memory)
  //
  // For TRITON/AUTO modes, composite capture handles islands internally
  // and doesn't require all buffers simultaneously, so memory-budget
  // splitting is not needed — the 100,000 op-count cap suffices.
  // Memory budget splitting is needed only for modes that use graph capture
  // without JIT compilation (CUDA_GRAPHS). JIT modes handle islands internally.
  const bool useMemoryBudget =
      !planLifecycle_.isSlotBySlot() &&
      ModeContract::forMode(graphExecutionMode_).usesGraphCapture &&
      !ModeContract::forMode(graphExecutionMode_).requiresCompilation;

  const size_t captureBudget = useMemoryBudget
      ? platformEstimateCaptureBudget()
      : SIZE_MAX;

  // Helper: estimate the output buffer footprint of a single slot.
  // Uses the current outputSlots_ which are populated after warmup.
  auto estimateSlotOutputBytes = [this](int slotIdx) -> size_t {
    size_t bytes = 0;
    const NativeSlot& slot = slots_[slotIdx];
    for (int o = 0; o < slot.wiring.numOutputs; o++) {
      int outIdx = slot.wiring.outputSlotIndices[o];
      if (outIdx >= 0 && outIdx < totalOutputSlots_ && outputSlots_[outIdx] != nullptr) {
        bytes += static_cast<size_t>(outputSlots_[outIdx]->lengthOf()) *
                 outputSlots_[outIdx]->sizeOfT();
      }
    }
    return bytes;
  };

  // Running accumulator for the current segment's estimated buffer footprint.
  size_t currentSegmentBytes = 0;

  auto isMatmulOrAttention = [this](int idx) -> bool {
    auto* op = slots_[idx].ident.op;
    if (!op || !op->getOpDescriptor()) return false;
    return op->getOpDescriptor()->hasAnyTrait(
        sd::ops::OP_TRAIT_MATMUL | sd::ops::OP_TRAIT_ATTENTION);
  };

  // Control flow ops and dynamic-output-size ops break segments.
  // Dynamic-output-size ops (e.g. single-arg Where, Unique, NMS) require
  // host synchronization during execution, which invalidates CUDA graph
  // capture. Delegates to NativeSlot::isCapturable() for the full check.
  auto isSlotCapturable = [](const NativeSlot& slot) -> bool {
    return slot.isCapturable();
  };

  GraphSegment current;
  current.def.startSlot = 0;
  current.def.isCapturable = isSlotCapturable(slots_[0]);
  currentSegmentBytes = estimateSlotOutputBytes(0);

  for (int i = 1; i < numSlots_; i++) {
    bool thisCapturable = isSlotCapturable(slots_[i]);
    bool deviceChange = (slots_[i].targetDeviceId != slots_[i - 1].targetDeviceId);
    int currentSize = i - current.def.startSlot;
    bool sizeLimit = (current.def.isCapturable && currentSize >= MAX_SEGMENT_SIZE);

    // Memory-budget check: would adding this slot push the segment over budget?
    size_t slotBytes = estimateSlotOutputBytes(i);
    bool memoryBudgetExceeded = false;
    if (useMemoryBudget && current.def.isCapturable && currentSize >= 2) {
      // Only split if we already have at least 2 ops in the segment
      // (avoid degenerate 1-op segments) and the budget would be exceeded.
      if (currentSegmentBytes + slotBytes > captureBudget) {
        memoryBudgetExceeded = true;
        DSP_DIAG_SEG(SEGMENT, current.def.startSlot,
                     "memoryBudget SPLIT at slot %d: segment[%d-%d] accumBytes=%zuMB + nextSlot=%zuKB > budget=%zuMB",
                     i, current.def.startSlot, i - 1,
                     currentSegmentBytes / (1024*1024),
                     slotBytes / 1024,
                     captureBudget / (1024*1024));
      }
    }

    // Break at matmul/attention boundaries for Triton fusion
    bool matmulBreak = false;
    if (matmulSegmentation) {
      bool thisIsMatmul = isMatmulOrAttention(i);
      bool prevIsMatmul = isMatmulOrAttention(i - 1);
      if (thisIsMatmul != prevIsMatmul) {
        // Transition detected. Only break if:
        // 1. Going from elementwise→matmul AND the elementwise range has
        //    outputs consumed by slots AFTER the upcoming matmul range, OR
        // 2. Going from matmul→elementwise (always break — matmul is a natural
        //    compilation unit boundary, and the elementwise tail should be
        //    isolated for Triton fusion)
        if (prevIsMatmul && !thisIsMatmul) {
          // matmul→elementwise: always break (isolate elementwise for Triton)
          matmulBreak = true;
          DSP_DIAG_SEG(SEGMENT, current.def.startSlot,
                       "matmulBoundary BREAK prev=matmul->cur=elementwise startSlot=%d curSlot=%d",
                       current.def.startSlot, i);
        } else {
          // elementwise→matmul: check if any slot in the current elementwise
          // range [current.def.startSlot .. i-1] has outputs consumed by
          // slots beyond i (outside this matmul). If yes, break. If all
          // outputs feed only slot i (the matmul), defer the break.
          bool hasExternalConsumers = false;
          for (int s = current.def.startSlot; s < i; s++) {
            for (int o = 0; o < slots_[s].wiring.numOutputs; o++) {
              int outIdx = slots_[s].wiring.outputSlotIndices[o];
              // Check if any consumer of this output is beyond the matmul
              for (int c = i + 1; c < numSlots_; c++) {
                for (int ci = 0; ci < slots_[c].wiring.numInputs; ci++) {
                  if (slots_[c].wiring.inputSourceIndices[ci] == outIdx) {
                    hasExternalConsumers = true;
                    break;
                  }
                }
                if (hasExternalConsumers) break;
              }
              if (hasExternalConsumers) break;
            }
            if (hasExternalConsumers) break;
          }
          matmulBreak = hasExternalConsumers;
          DSP_DIAG_SEG(SEGMENT, current.def.startSlot,
                       "matmulBoundary %s prev=elementwise->cur=matmul startSlot=%d curSlot=%d hasExternalConsumers=%d",
                       matmulBreak ? "BREAK" : "DEFER",
                       current.def.startSlot, i, hasExternalConsumers ? 1 : 0);
        }
      }
    }

    bool cpuTraitBreak = platformShouldBreakSegmentAtTraitBoundary(i, i - 1);

    if (thisCapturable != current.def.isCapturable || deviceChange || sizeLimit
        || matmulBreak || cpuTraitBreak || memoryBudgetExceeded) {
      // End current segment
      current.def.endSlot = i - 1;
      segments_.push_back(std::move(current));

      // Start new segment with this slot's bytes
      current = GraphSegment();
      current.def.startSlot = i;
      current.def.isCapturable = thisCapturable;
      currentSegmentBytes = slotBytes;
    } else {
      // Accumulate this slot's output bytes into the current segment
      currentSegmentBytes += slotBytes;
    }
  }

  // Finalize last segment
  current.def.endSlot = numSlots_ - 1;
  segments_.push_back(std::move(current));

  // Log segment structure
  int capturableCount = 0, totalCapturable = 0;
  int staticCapturableCount = 0, dynamicCapturableCount = 0;
  for (auto& seg : segments_) {
    if (seg.def.isCapturable) {
      capturableCount++;
      int sz = seg.def.endSlot - seg.def.startSlot + 1;
      totalCapturable += sz;
      // A segment is "static" if all its slots have stable shapes
      bool allStatic = true;
      for (int s = seg.def.startSlot; s <= seg.def.endSlot && allStatic; s++)
        allStatic = slots_[s].shapeCache.shapeStatic;
      if (allStatic) staticCapturableCount++;
      else dynamicCapturableCount++;
    }
  }
  DSP_DIAG(SEGMENT, "%d segments (%d capturable: %d static, %d dynamic; covering %d/%d slots)",
           (int)segments_.size(), capturableCount,
           staticCapturableCount, dynamicCapturableCount,
           totalCapturable, numSlots_);

  int maxLoggedSegments = 8;
  int logged = std::min(static_cast<int>(segments_.size()), maxLoggedSegments);
  for (int i = 0; i < logged; i++) {
    const auto& seg = segments_[i];
    int targetDevice = -1;
    if (seg.def.startSlot >= 0 && seg.def.startSlot < numSlots_) {
      targetDevice = slots_[seg.def.startSlot].targetDeviceId;
    }
    DSP_DIAG_SEG(SEGMENT, i, "segment[%d] [%d-%d] capturable=%d targetDeviceId=%d",
                 i, seg.def.startSlot, seg.def.endSlot, static_cast<int>(seg.def.isCapturable), targetDevice);
  }
  if ((int)segments_.size() > maxLoggedSegments) {
    DSP_DIAG(SEGMENT, "... %d additional segments not shown in device map",
             static_cast<int>(segments_.size()) - maxLoggedSegments);
  }

  // Propagate outputSlots_, resolve backend, and detect value-dep ops for all segments.
  for (auto& seg : segments_) {
    seg.slotArrayCache = outputSlots_;
    seg.def.selectedBackend = resolveBackendForSegment(seg.def.isCapturable);
    // Scan slots for value-dependent ops — these require shape key recomputation
    // even when shapes are frozen, because input VALUES (not just shapes) affect output shape.
    seg.def.hasValueDepOps = false;
    for (int s = seg.def.startSlot; s <= seg.def.endSlot; s++) {
      if (slots_[s].flags.outputShapeDependsOnInputValues) {
        seg.def.hasValueDepOps = true;
        break;
      }
    }

    // allFrozenConstants is set later by the post-freeze scan (after
    // detectFrozenConstants runs). Default false here — safe conservative.
    seg.def.allFrozenConstants = false;

    DSP_DIAG_SEG(SEGMENT, seg.def.startSlot, "segment[%d-%d] selectedBackend=%d hasValueDepOps=%d",
                 seg.def.startSlot, seg.def.endSlot, static_cast<int>(seg.def.selectedBackend),
                 seg.def.hasValueDepOps ? 1 : 0);
  }

  // Initialize symbolic shape ranges if enabled
  if (Environment::getInstance().dspSymbolicShapes()) {
    int warmup = Environment::getInstance().dspSymbolicShapeWarmup();
    for (auto& seg : segments_) {
      seg.exec.symbolicShapeEnabled = true;
      seg.exec.symbolicWarmupRemaining = warmup;
      seg.exec.symbolicRangeData = createSegmentShapeProfile(warmup);
    }
  }

  // ── Post-pass: merge unprofitable small segments ──────────────────────────
  // Segments below MIN_PROFITABLE_SIZE that consist entirely of transparent ops
  // (views, shapes, identity, constants) are merged into the preceding segment.
  // This mirrors XLA's DeclusterNodes which removes trivially small clusters.
  static constexpr int MIN_PROFITABLE_SIZE = 4;

  if (segments_.size() > 1) {
    std::vector<GraphSegment> merged;
    merged.reserve(segments_.size());
    merged.push_back(std::move(segments_[0]));

    for (size_t i = 1; i < segments_.size(); i++) {
      auto& seg = segments_[i];
      int sz = seg.def.endSlot - seg.def.startSlot + 1;

      // Check if segment is small AND all ops are transparent (non-materializing)
      bool isSmallTransparent = false;
      if (sz < MIN_PROFITABLE_SIZE && seg.def.isCapturable) {
        isSmallTransparent = true;
        for (int s = seg.def.startSlot; s <= seg.def.endSlot; s++) {
          // A slot is "transparent" if it's a view, identity, shape-only, or constant.
          // Uses the slot's opTraits_ bitmask directly — no separate trait lookup needed.
          bool isTransparent = slots_[s].aliasesInput() ||
                               slots_[s].hasOpTrait(sd::ops::OP_TRAIT_SHAPE_ONLY_OUTPUT) ||
                               slots_[s].hasOpTrait(sd::ops::OP_TRAIT_CONSTANT_GENERATION);
          if (!isTransparent) {
            isSmallTransparent = false;
            break;
          }
        }
      }

      if (isSmallTransparent && !merged.empty()) {
        // Absorb into preceding segment
        auto& prev = merged.back();
        DSP_DIAG(FUSION,
                 "segmentMerge VERDICT=merged segA=[%d-%d] segB=[%d-%d] size=%d reason=small-transparent",
                 prev.def.startSlot, prev.def.endSlot,
                 seg.def.startSlot, seg.def.endSlot, sz);
        prev.def.endSlot = seg.def.endSlot;
        // Preserve hasValueDepOps
        if (seg.def.hasValueDepOps) prev.def.hasValueDepOps = true;
        DSP_DIAG(SEGMENT, "Merged small transparent segment [%d-%d] (%d slots) into [%d-%d]",
                 seg.def.startSlot, seg.def.endSlot, sz,
                 prev.def.startSlot, prev.def.endSlot);
      } else {
        const char* rejectReason =
            (sz < MIN_PROFITABLE_SIZE && !seg.def.isCapturable) ? "non-capturable" :
            (sz >= MIN_PROFITABLE_SIZE) ? "above-min-profitable-size" :
            "materializing-op";
        DSP_DIAG(FUSION,
                 "segmentMerge VERDICT=rejected segB=[%d-%d] size=%d reason=%s",
                 seg.def.startSlot, seg.def.endSlot, sz, rejectReason);
        merged.push_back(std::move(seg));
      }
    }

    if (merged.size() < segments_.size()) {
      DSP_DIAG(SEGMENT, "Profitability post-pass: %d -> %d segments",
               (int)segments_.size(), (int)merged.size());
      segments_ = std::move(merged);
    }
  }
}

// ─── fromFlatGraph (delegates to NativePlanCompiler) ─────────────────────────

NativeDynamicShapePlan* NativeDynamicShapePlan::fromFlatGraph(
    const ::graph::FlatGraph* graph,
    const std::unordered_map<std::string, NDArray*>& variables,
    const std::vector<std::string>& requestedOutputs) {
  return NativePlanCompiler::compile(graph, variables, requestedOutputs);
}

// ─── GPU graph capture audit and validation ─────────────────────────────────
// Moved to platform dispatch (getHostOnlyOps, printCaptureAudit,
// validateCapturedGraph). These methods are GPU-only and defined in the .cu file.

// ─── CPU Graph compilation audit and validation ─────────────────────────────

bool NativeDynamicShapePlan::validateCompiledCpuGraph(int segmentIndex) const {
  if (lastCompilationAudit_.empty()) return true;  // No audit data = no validation

  bool allOpsCompiled = true;

  for (const auto& entry : lastCompilationAudit_) {
    if (!entry.wasCompiled) {
      allOpsCompiled = false;
      const char* backendName = cpuGraphBackend_ ? cpuGraphBackend_->name() : "unknown";
      DSP_DIAG(COMPILE, "CPU GRAPH VALIDATION FAILURE: slot %d (%s) was NOT compiled by %s backend: %s",
                entry.slotIndex, entry.opName.c_str(), backendName, entry.reason.c_str());
    }
  }

  return allOpsCompiled;
}

// ─── JNI introspection implementations ──────────────────────────────────────

void NativeDynamicShapePlan::snapshotExecStats(void* execCtxPtr) {
  if (execCtxPtr == nullptr) return;
  auto* ctx = static_cast<PlanExecutionContext*>(execCtxPtr);
  lastExecStats_.segmentsWarmup = ctx->segmentsWarmup;
  lastExecStats_.segmentsCaptured = ctx->segmentsCaptured;
  lastExecStats_.segmentsReplayed = ctx->segmentsReplayed;
  lastExecStats_.segmentsSlotBySlot = ctx->segmentsSlotBySlot;
  lastExecStats_.segmentsFailed = ctx->segmentsFailed;
  lastExecStats_.segmentsTotal = ctx->segmentsTotal;
  lastExecStats_.syncLevel = static_cast<int>(ctx->currentSyncLevel);
  lastExecStats_.streamSyncCount = ctx->streamSyncCount;
  // consecutiveUnchangedCount from prevStepFingerprints_ sentinel key -1
  auto it = prevStepFingerprints_.find(-1);
  lastExecStats_.consecutiveUnchangedCount = (it != prevStepFingerprints_.end())
      ? static_cast<int>(it->second) : 0;
  lastExecStats_.valid = true;
}

int NativeDynamicShapePlan::writeDeviceBufferOnDefaultStream(int extIdx, void* srcHost, long long numBytes) {
  if (extIdx < 0 || extIdx >= numExternalInputs_) return -1;
  // Lazily allocate staging buffers if ensureAndSyncStagingBuffers hasn't run yet
  // (plan may still be in warmup when JNI write is called after initial output())
  if (placeholderStagingBuffers_ == nullptr) {
    NDArray* lastExt = getLastExternalInput(extIdx);
    if (lastExt == nullptr || lastExt->isEmpty()) return -1;
    placeholderStagingBuffers_ = new NDArray*[numExternalInputs_]();
    effectiveExternals_ = new NDArray*[numExternalInputs_]();
  }
  if (placeholderStagingBuffers_[extIdx] == nullptr) {
    NDArray* lastExt = getLastExternalInput(extIdx);
    if (lastExt == nullptr || lastExt->isEmpty()) return -2;
    placeholderStagingBuffers_[extIdx] = new NDArray(
        lastExt->ordering(), *lastExt->getShapeAsVector(),
        lastExt->dataType(), LaunchContext::defaultContext());
  }
  NDArray* staging = placeholderStagingBuffers_[extIdx];
  if (staging == nullptr || sd::graph::dspBuffer(staging) == nullptr) return -2;
  int err = sd::graph::dspMemcpyH2DAsync(sd::graph::dspBuffer(staging), srcHost,
                                         static_cast<size_t>(numBytes), nullptr);
  if (err != 0) return -3;
  // Also write to the external array's device buffer so warmup execution
  // (which reads from externalArrays directly, not staging) sees the data.
  NDArray* ext = getLastExternalInput(extIdx);
  if (ext != nullptr && sd::graph::dspBuffer(ext) != nullptr) {
    sd::graph::dspMemcpyH2DAsync(sd::graph::dspBuffer(ext), srcHost,
                                 static_cast<size_t>(numBytes), nullptr);
    // Mark device as authoritative so performPreReplaySync H2D doesn't
    // overwrite our write with stale host data.
    ext->dataBuffer()->writeSpecial();
  }
  // Mark staging as JNI-written so ensureAndSyncStagingBuffers skips D2D overwrite
  if (static_cast<int>(deviceWritePending_.size()) <= extIdx)
    deviceWritePending_.resize(numExternalInputs_, false);
  deviceWritePending_[extIdx] = true;
  return 0;
}

int NativeDynamicShapePlan::writeDeviceBufferOnExplicitStream(int extIdx, void* srcHost, long long numBytes, void* stream) {
  if (extIdx < 0 || extIdx >= numExternalInputs_) return -1;
  // Lazily allocate staging buffers if ensureAndSyncStagingBuffers hasn't run yet
  if (placeholderStagingBuffers_ == nullptr) {
    NDArray* lastExt = getLastExternalInput(extIdx);
    if (lastExt == nullptr || lastExt->isEmpty()) return -1;
    placeholderStagingBuffers_ = new NDArray*[numExternalInputs_]();
    effectiveExternals_ = new NDArray*[numExternalInputs_]();
  }
  if (placeholderStagingBuffers_[extIdx] == nullptr) {
    NDArray* lastExt = getLastExternalInput(extIdx);
    if (lastExt == nullptr || lastExt->isEmpty()) return -2;
    placeholderStagingBuffers_[extIdx] = new NDArray(
        lastExt->ordering(), *lastExt->getShapeAsVector(),
        lastExt->dataType(), LaunchContext::defaultContext());
  }
  NDArray* staging = placeholderStagingBuffers_[extIdx];
  if (staging == nullptr || sd::graph::dspBuffer(staging) == nullptr) return -2;
  int err = sd::graph::dspMemcpyH2DAsync(sd::graph::dspBuffer(staging), srcHost,
                                         static_cast<size_t>(numBytes), stream);
  if (err != 0) return -3;
  // Also write to external array so warmup execution sees the data.
  NDArray* ext = getLastExternalInput(extIdx);
  if (ext != nullptr && sd::graph::dspBuffer(ext) != nullptr) {
    sd::graph::dspMemcpyH2DAsync(sd::graph::dspBuffer(ext), srcHost,
                                 static_cast<size_t>(numBytes), stream);
    ext->dataBuffer()->writeSpecial();
  }
  if (static_cast<int>(deviceWritePending_.size()) <= extIdx)
    deviceWritePending_.resize(numExternalInputs_, false);
  deviceWritePending_[extIdx] = true;
  return 0;
}

std::string NativeDynamicShapePlan::getSegmentsSummaryJson() const {
  std::string json = "[";
  for (int i = 0; i < static_cast<int>(segments_.size()); i++) {
    if (i > 0) json += ",";
    const auto& seg = segments_[i];
    json += "{\"idx\":" + std::to_string(i)
         + ",\"start\":" + std::to_string(seg.def.startSlot)
         + ",\"end\":" + std::to_string(seg.def.endSlot)
         + ",\"phase\":\"" + std::string(seg.exec.displayPhaseName()) + "\""
         + ",\"capturable\":" + (seg.def.isCapturable ? "true" : "false")
         + ",\"argGen\":" + std::to_string(seg.exec.argTableGeneration)
         + ",\"capArgGen\":" + std::to_string(seg.exec.capturedArgGeneration)
         + ",\"needsRefresh\":" + (seg.exec.needsArgRefresh() ? "true" : "false")
         + ",\"backend\":\"" + seg.exec.compiledByBackend + "\""
         + "}";
  }
  json += "]";
  return json;
}

// copyStagingToBuffer is platform-dispatched:
//   CUDA: NativeDynamicShapePlan_cuda.cu  (D2D + stream sync)
//   CPU:  NativeDynamicShapePlan_cuda_stubs.cpp  (H2H memcpy)

// drainFingerprintRingPublic and getFingerprintJson are platform-dispatched:
//   CUDA: NativeDynamicShapePlan_cudagraph.cu
//   CPU:  NativeDynamicShapePlan_cuda_stubs.cpp

}  // namespace graph
}  // namespace sd

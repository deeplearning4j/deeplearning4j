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
 * NativeDynamicShapePlan — Segment Management
 *
 * Contains computeSegmentShapeKey(),
 * executeSegmentWithCpuGraph(), and executeSegmentSlotBySlot().
 */

#include <graph/NativeDynamicShapePlan.h>
#include <graph/gpu/SymbolicShapeRanges.h>
#include <graph/DspDiagnostics.h>
#include <graph/DspPhaseUtils.h>
#include <graph/DspHashUtils.h>
#include <graph/PlanExecutionContext.h>

// Portable buffer accessor: specialBuffer() on CUDA, buffer() on CPU.
#ifdef SD_CUDA
#define DSP_BUF(arr) ((arr)->specialBuffer())
#else
#define DSP_BUF(arr) ((arr)->buffer())
#endif
#include <graph/cpu/FunctionalReplayHandle.h>
#include <helpers/MmulHelper.h>
#include <helpers/ShapeUtils.h>
#include <system/Environment.h>

#include <algorithm>
#include <unordered_set>

// GraphSegment static methods (moved from header to avoid Environment.h in NativeDynamicShapePlan.h)
int GraphSegment::maxOomRetries() { return sd::Environment::getInstance().dspCaptureOomMaxRetries(); }
int GraphSegment::retryInterval() { return sd::Environment::getInstance().dspCaptureOomRetryInterval(); }

#ifdef SD_CUDA
#include <cuda_runtime.h>
#endif

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
#if HAVE_OPENVINO
#include <graph/cpu/OpenVinoGraphBackend.h>
#endif
namespace sd {
namespace graph {

namespace {
// Status enum string helper — delegates to shared dsp::dspStatusName in DspPhaseUtils.h.
const char* statusName_seg(Status status) {
  return dsp::dspStatusName(status);
}

}  // namespace

// ─── Segment shape key computation ──────────────────────────────────────────

LongType NativeDynamicShapePlan::computeSegmentShapeKey(
    GraphSegment& seg, NDArray** externalInputs, int numExt) {

  // ── Frozen fast path: reuse cached key if shapes can't change ──
  // This is the AUTHORITATIVE cache check — applies to ALL callers
  // (phaseCompile, executeSegmentWithGpuGraph, executeSegmentWithSpecificBackend, etc.)
  if (shapesFrozen_ && seg.exec.cachedShapeKey != 0) {
    return seg.exec.cachedShapeKey;
  }

  // ── Symbolic shape range path ──────────────────────────────────────────
  // When enabled, collect cross-segment inputs, feed them to the shape
  // profiler, and (after warmup) use range-based hashing that ignores
  // dynamic dimensions.
  if (seg.exec.symbolicShapeEnabled && seg.exec.symbolicRangeData != nullptr) {
    auto* profile = static_cast<SegmentShapeProfile*>(seg.exec.symbolicRangeData);

    // Collect cross-segment input arrays (same logic as standard path below)
    std::unordered_set<int> segOutputSlots;
    for (int s = seg.def.startSlot; s <= seg.def.endSlot; s++) {
      NativeSlot& slot = slots_[s];
      for (int i = 0; i < slot.wiring.numOutputs; i++) {
        segOutputSlots.insert(slot.wiring.outputSlotIndices[i]);
      }
    }

    std::vector<NDArray*> crossInputs;
    for (int s = seg.def.startSlot; s <= seg.def.endSlot; s++) {
      NativeSlot& slot = slots_[s];
      for (int i = 0; i < slot.wiring.numInputs; i++) {
        int srcIdx = slot.wiring.inputSourceIndices[i];
        if (srcIdx < 0) {
          int extIdx = -(srcIdx + 1);
          if (extIdx < numExt && externalInputs[extIdx] != nullptr) {
            crossInputs.push_back(externalInputs[extIdx]);
          }
        } else if (srcIdx >= 0 && segOutputSlots.find(srcIdx) == segOutputSlots.end()) {
          if (srcIdx < totalOutputSlots_ && outputSlots_[srcIdx] != nullptr) {
            crossInputs.push_back(outputSlots_[srcIdx]);
          }
        }
      }
    }

    // Record observations during warmup.
    // When shapesFrozen_, shapes are constant — force warmup complete immediately
    // by recording one observation (sufficient for frozen shapes) and skip waiting
    // for the normal 2-observation warmup cycle.
    if (!isWarmupComplete(profile)) {
      recordObservedShapes(profile, crossInputs.data(),
                           static_cast<int>(crossInputs.size()));
      if (shapesFrozen_ && !isWarmupComplete(profile)) {
        // Frozen shapes won't change, so one observation is enough.
        // Record again to satisfy the 2-step warmup requirement immediately.
        recordObservedShapes(profile, crossInputs.data(),
                             static_cast<int>(crossInputs.size()));
        DSP_DIAG(COMPILE, "SymbolicShapes: seg[%d-%d] fast-completed warmup (shapesFrozen)",
                 seg.def.startSlot, seg.def.endSlot);
      } else {
        DSP_DIAG(COMPILE, "SymbolicShapes: seg[%d-%d] observation %d/%d",
                 seg.def.startSlot, seg.def.endSlot,
                 getObservationCount(profile), getWarmupSteps(profile));
      }
    }

    // After warmup, use range-based key
    if (isWarmupComplete(profile)) {
      LongType rangeKey = computeRangeBasedShapeKey(
          profile, crossInputs.data(), static_cast<int>(crossInputs.size()),
          seg.def.startSlot, seg.def.endSlot);

      // Mix op names, iArgs, and tArgs into the range-based key so different
      // plans with the same input shapes but different ops produce unique keys
      // in singleton backend caches (OpenVINO, OneDNN Graph). v2-cache-fix.
      uint64_t rangeKeyU64 = static_cast<uint64_t>(rangeKey);
      auto mixRange = [&rangeKeyU64](LongType val) {
        dsp::fnv1aMixValue(rangeKeyU64, static_cast<uint64_t>(val));
      };
      for (int s = seg.def.startSlot; s <= seg.def.endSlot; s++) {
        NativeSlot& slot = slots_[s];
        if (!slot.ident.opName.empty()) {
          for (const char* p = slot.ident.opName.c_str(); *p != '\0'; p++) {
            mixRange(static_cast<LongType>(*p));
          }
        }
        mixRange(static_cast<LongType>(slot.args.numIArgs));
        for (int a = 0; a < slot.args.numIArgs; a++) {
          mixRange(static_cast<LongType>(slot.args.iArgs[a]));
        }
        mixRange(static_cast<LongType>(slot.args.numTArgs));
      }

      DSP_DIAG(COMPILE, "SymbolicShapes: seg[%d-%d] using range-based key=%lld (with-op-mix)",
               seg.def.startSlot, seg.def.endSlot, static_cast<long long>(rangeKeyU64));
      // NOTE: Do NOT set seg.exec.cachedShapeKey here. cachedShapeKey must only be
      // written after a successful compile+execute in executeSegmentWithSpecificBackend().
      // Writing it here causes the cascade to skip compilation for fallback backends
      // when the first backend fails (cachedShapeKey != 0 → needsCompile = false).
      return static_cast<LongType>(rangeKeyU64);
    }
    // Fall through to standard path during warmup
  }

  // ── Standard FNV-1a path ───────────────────────────────────────────────
  uint64_t key = dsp::FNV1A64_OFFSET_BASIS;
  auto mix = [&key](LongType val) {
    dsp::fnv1aMixValue(key, static_cast<uint64_t>(val));
  };

  // Hash array shape signature: rank + dims + length + dtype.
  auto mixArraySignature = [&](NDArray* arr) {
    if (arr == nullptr) return;

    const LongType* si = arr->shapeInfo();
    LongType rank = shape::rank(si);
    mix(rank);
    for (int d = 0; d < rank; d++) {
      mix(si[d + 1]);
    }
    mix(static_cast<LongType>(arr->lengthOf()));
    mix(static_cast<LongType>(arr->dataType()));
  };

  mix(seg.def.startSlot);
  mix(seg.def.endSlot);

  // Mix op names so different plans with same slot indices + shapes don't collide
  // in singleton backend caches (e.g. OpenVINO, OneDNN Graph)
  for (int s = seg.def.startSlot; s <= seg.def.endSlot; s++) {
    NativeSlot& slot = slots_[s];
    if (!slot.ident.opName.empty()) {
      for (const char* p = slot.ident.opName.c_str(); *p != '\0'; p++) {
        mix(static_cast<LongType>(*p));
      }
    }
    mix(static_cast<LongType>(slot.wiring.numInputs));
    mix(static_cast<LongType>(slot.wiring.numOutputs));
    mix(static_cast<LongType>(slot.args.numIArgs));
    // Mix actual iArg values (e.g. reshape target shape, axis indices)
    for (int a = 0; a < slot.args.numIArgs; a++) {
      mix(static_cast<LongType>(slot.args.iArgs[a]));
    }
    // Mix tArg count (float args like epsilon, scale)
    mix(static_cast<LongType>(slot.args.numTArgs));
  }

  std::unordered_set<int> segOutputSlots;
  for (int s = seg.def.startSlot; s <= seg.def.endSlot; s++) {
    NativeSlot& slot = slots_[s];
    for (int i = 0; i < slot.wiring.numOutputs; i++) {
      segOutputSlots.insert(slot.wiring.outputSlotIndices[i]);
    }
  }

  for (int s = seg.def.startSlot; s <= seg.def.endSlot; s++) {
    NativeSlot& slot = slots_[s];
    for (int i = 0; i < slot.wiring.numInputs; i++) {
      int srcIdx = slot.wiring.inputSourceIndices[i];
      if (srcIdx < 0) {
        int extIdx = -(srcIdx + 1);
        if (extIdx < numExt && externalInputs[extIdx] != nullptr) {
          mixArraySignature(externalInputs[extIdx]);
        }
      } else if (srcIdx >= 0 && segOutputSlots.find(srcIdx) == segOutputSlots.end()) {
        if (srcIdx < totalOutputSlots_ && outputSlots_[srcIdx] != nullptr) {
          mixArraySignature(outputSlots_[srcIdx]);
        }
      }
    }
  }

  // NOTE: Value-dependent-shape ops (reshape, broadcast_to, etc.) do NOT
  // need data value hashing at the segment level. Reasons:
  //   1. During warmup (execCount==0), segments always run slot-by-slot.
  //   2. The per-slot computeShapeKey() in _slotexec.cpp hashes values
  //      gated on outputShapeDependsOnInputValues — handles correctness.
  //   3. After shapes freeze, the frozen fast-path returns the cached key.
  //   4. iArgs (already hashed above) encode the same shape info for most ops.
  // Removing syncToHost here eliminates GPU→CPU sync during key computation.

  // NOTE: Do NOT set seg.exec.cachedShapeKey here. It must only be written after
  // a successful compile+execute in executeSegmentWithSpecificBackend() (line ~681).
  // Writing it here causes the cascade to skip compilation for fallback backends.
  return key;
}

// ─── CPU Graph backend selection ────────────────────────────────────────────

// ─── CPU Graph backend chain (prioritized list of all available backends) ────

const std::vector<GraphBackend*>& NativeDynamicShapePlan::getCpuGraphBackendChain() {
  if (cpuGraphBackendChainBuilt_) return cpuGraphBackendChain_;
  cpuGraphBackendChainBuilt_ = true;
  cpuGraphBackendChain_.clear();

  const auto mode = graphExecutionMode_;

  // If mode is explicitly non-CPU-graph, return empty chain
  // On CPU builds (no SD_CUDA), TRITON/NVRTC/PTX/HIP/etc. have no GPU backends,
  // so fall through to the CPU backend chain (oneDNN, OpenVINO, etc.) instead of
  // returning empty and forcing slot-by-slot.
  if (mode == GraphExecutionMode::GEM_SLOT_BY_SLOT) {
    return cpuGraphBackendChain_;
  }
#ifdef SD_CUDA
  if (mode == GraphExecutionMode::GEM_TRITON ||
      mode == GraphExecutionMode::GEM_NVRTC_JIT ||
      mode == GraphExecutionMode::GEM_PTX_JIT ||
      mode == GraphExecutionMode::GEM_HIP_GRAPHS ||
      mode == GraphExecutionMode::GEM_LEVELZERO ||
      mode == GraphExecutionMode::GEM_VULKAN ||
      mode == GraphExecutionMode::GEM_METAL ||
      mode == GraphExecutionMode::GEM_TPU ||
      mode == GraphExecutionMode::GEM_HEXAGON) {
    return cpuGraphBackendChain_;
  }
#endif

#ifdef SD_CUDA
  const bool autoLikeMode = (mode == GraphExecutionMode::GEM_AUTO ||
                             mode == GraphExecutionMode::GEM_CUDA_GRAPHS);
#else
  const bool autoLikeMode = (mode == GraphExecutionMode::GEM_AUTO ||
                             mode == GraphExecutionMode::GEM_CUDA_GRAPHS ||
                             mode == GraphExecutionMode::GEM_TRITON ||
                             mode == GraphExecutionMode::GEM_NVRTC_JIT ||
                             mode == GraphExecutionMode::GEM_PTX_JIT ||
                             mode == GraphExecutionMode::GEM_HIP_GRAPHS ||
                             mode == GraphExecutionMode::GEM_LEVELZERO ||
                             mode == GraphExecutionMode::GEM_VULKAN ||
                             mode == GraphExecutionMode::GEM_METAL ||
                             mode == GraphExecutionMode::GEM_TPU ||
                             mode == GraphExecutionMode::GEM_HEXAGON);
#endif

  // If a specific backend is forced, only return that one
  bool forcedMode = !autoLikeMode;

#if HAVE_MLX
  if (mode == GraphExecutionMode::GEM_MLX || autoLikeMode) {
    auto& mlx = MlxGraphBackend::getInstance();
    if (mlx.isAvailable()) {
      cpuGraphBackendChain_.push_back(&mlx);
      if (forcedMode) return cpuGraphBackendChain_;
    }
  }
#endif

#if HAVE_OPENVINO
  // OpenVINO has the broadest op coverage (~200 ops, including rms_norm, reshape,
  // permute, silu, cast, gather, etc.) and supports native-deferred execution for
  // complex ops (rope, attention). For transformer models like Qwen, it can fuse
  // nearly the entire decoder layer. Try it BEFORE OneDNN which has narrower
  // coverage (~40 ops, no rms_norm/reshape/permute/silu).
  if (mode == GraphExecutionMode::GEM_OPENVINO || autoLikeMode) {
    auto& ov = OpenVinoGraphBackend::getInstance();
    if (ov.isAvailable()) {
      cpuGraphBackendChain_.push_back(&ov);
      if (forcedMode) return cpuGraphBackendChain_;
    }
  }
#endif

#if HAVE_ONEDNN
  // OneDNN as fallback: narrower op coverage but handles mixed segments well.
  // Segments that OpenVINO rejects (due to ALL-or-nothing op requirement)
  // may still be partially fused by OneDNN's island-based approach.
  if (autoLikeMode) {
    auto& onednn = OneDnnGraphBackend::getInstance();
    if (onednn.isAvailable()) {
      cpuGraphBackendChain_.push_back(&onednn);
    }
  }
#endif

#if HAVE_ARMCOMPUTE
  if (autoLikeMode) {
    auto& acl = AclGraphBackend::getInstance();
    if (acl.isAvailable()) {
      cpuGraphBackendChain_.push_back(&acl);
    }
  }
#endif

#if HAVE_NNAPI
  if (mode == GraphExecutionMode::GEM_NNAPI || autoLikeMode) {
    auto& nnapi = NnapiGraphBackend::getInstance();
    if (nnapi.isAvailable()) {
      cpuGraphBackendChain_.push_back(&nnapi);
      if (forcedMode) return cpuGraphBackendChain_;
    }
  }
#endif

#if HAVE_MLIR
#if defined(__ANDROID__) || (defined(__linux__) && defined(__aarch64__))
  if (mode == GraphExecutionMode::GEM_ARM_HYBRID || autoLikeMode) {
    auto& armHybrid = ArmHybridGraphBackend::getInstance();
    if (armHybrid.isAvailable()) {
      cpuGraphBackendChain_.push_back(&armHybrid);
      if (forcedMode) return cpuGraphBackendChain_;
    }
  }
#endif

  if (autoLikeMode) {
    auto& mlirBackend = MlirCpuGraphBackend::getInstance();
    if (mlirBackend.isAvailable()) {
      cpuGraphBackendChain_.push_back(&mlirBackend);
    }
  }
#endif

  if (!cpuGraphBackendChain_.empty()) {
    DSP_DIAG(BACKEND, "CPU backend chain built: %d backends available", (int)cpuGraphBackendChain_.size());
    for (size_t i = 0; i < cpuGraphBackendChain_.size(); i++) {
      DSP_DIAG(BACKEND, "  chain[%d] = %s", (int)i, cpuGraphBackendChain_[i]->name());
    }
  }

  return cpuGraphBackendChain_;
}

// ─── Segment execution: CPU graph backend (with per-segment cascade) ────────

Status NativeDynamicShapePlan::executeSegmentWithCpuGraph(
    GraphSegment& seg, NDArray** externalArrays, int numExt, void* stream) {
  DSP_REQUIRE_PLAN_PHASE_AT_LEAST(PlanPhase::SLOT_BY_SLOT, "executeSegmentWithCpuGraph");

  // If all backends have been exhausted for this segment, hard fail.
  // Falling back to slot-by-slot is BANNED — fix the compilation failure.
  if (seg.exec.compilationFailed) {
    DSP_THROW_SEG(COMPILE, seg.def.startSlot,
                  "executeSegmentWithCpuGraph: seg[%d-%d] permanently failed — all backends exhausted. "
                  "Fix the compilation failure instead of falling back to slot-by-slot.",
                  seg.def.startSlot, seg.def.endSlot);
  }

  // If we already resolved a backend for this segment, use it directly
  if (seg.resolvedCpuBackend != nullptr) {
    return executeSegmentWithSpecificBackend(seg, seg.resolvedCpuBackend, externalArrays, numExt, stream);
  }

  // Cascade through the backend chain to find one that works
  const auto& chain = getCpuGraphBackendChain();
  if (chain.empty()) {
    DSP_THROW_SEG(COMPILE, seg.def.startSlot,
                  "executeSegmentWithCpuGraph: no CPU graph backends available for seg[%d-%d]. "
                  "Cannot execute — a backend must be configured.",
                  seg.def.startSlot, seg.def.endSlot);
  }

  // Warmup must happen before any backend tries to compile (needs output shapes)
  if (seg.exec.executionCount == 0) {
    auto warmupStatus = executeSegmentSlotBySlot(seg, externalArrays, numExt, stream);
    DSP_DIAG(EXECUTE, "executeSegmentWithCpuGraph: warmup %s for seg[%d-%d], executionCount→%d",
             warmupStatus == Status::OK ? "OK" : "FAILED",
             seg.def.startSlot, seg.def.endSlot, seg.exec.executionCount);
    if (warmupStatus != Status::OK) {
      return warmupStatus;
    }
  }

  // Try each backend in priority order.
  // Track whether ANY backend attempted compilation (canFuseSegment=true)
  // vs none could fuse it at all (segment has no fusible ops).
  bool anyBackendAttemptedCompile = false;

  for (size_t i = 0; i < chain.size(); i++) {
    GraphBackend* backend = chain[i];
    const char* backendName = backend->name();

    if (!backend->canFuseSegment(slots_, seg.def.startSlot, seg.def.endSlot)) {
      DSP_DIAG(BACKEND, "cascade: backend=%s cannot fuse seg[%d-%d], trying next",
                backendName, seg.def.startSlot, seg.def.endSlot);
      continue;
    }

    anyBackendAttemptedCompile = true;

    // Attempt compile + validate + execute with this backend
    auto status = executeSegmentWithSpecificBackend(seg, backend, externalArrays, numExt, stream);
    if (status == Status::OK) {
      // Cache the resolved backend for future executions
      seg.resolvedCpuBackend = backend;
      DSP_DIAG(BACKEND, "cascade: seg[%d-%d] resolved to backend=%s (chain position %d/%d)",
                seg.def.startSlot, seg.def.endSlot, backendName, (int)i + 1, (int)chain.size());
      return Status::OK;
    }

    DSP_DIAG(BACKEND, "cascade: backend=%s failed for seg[%d-%d] (status=%d), trying next",
              backendName, seg.def.startSlot, seg.def.endSlot, static_cast<int>(status));
    // Reset compilationFailed so next backend gets a fresh try
    seg.exec.compilationFailed = false;
  }

  if (!anyBackendAttemptedCompile) {
    // No backend could fuse this segment — all returned canFuseSegment=false.
    // This means the segment has no fusible ops (e.g., all permutes/reshapes/identity).
    // Mark compilationFailed so the frozen fast path doesn't re-attempt the cascade
    // every step for this permanently-unfusible segment.
    seg.exec.compilationFailed = true;
    DSP_DIAG(BACKEND, "cascade: NO backend can fuse seg[%d-%d] (no fusible ops) — "
              "demoting to slot-by-slot native execution",
              seg.def.startSlot, seg.def.endSlot);
    return Status::KERNEL_FAILURE;
  }

  // At least one backend tried to compile but ALL failed.
  // Demote to slot-by-slot native execution — same as the "no fusible ops" path.
  // This can happen when shapes/types change between executions (e.g., BF16 on a CPU
  // without AVX-512 BF16) or when OV model shapes don't match cached buffers.
  seg.exec.compilationFailed = true;
  DSP_DIAG(COMPILE, "cascade: ALL %d backends failed to compile seg[%d-%d] — "
            "demoting to slot-by-slot native execution",
            (int)chain.size(), seg.def.startSlot, seg.def.endSlot);
  return Status::KERNEL_FAILURE;
}

// ─── Execute segment with a specific backend (shared logic) ─────────────────

Status NativeDynamicShapePlan::executeSegmentWithSpecificBackend(
    GraphSegment& seg, GraphBackend* backend, NDArray** externalArrays, int numExt, void* stream) {
  DSP_REQUIRE_PLAN_PHASE_AT_LEAST(PlanPhase::SLOT_BY_SLOT, "executeSegmentWithSpecificBackend");

  const char* backendName = backend->name();

  // ── Shape key: detect if segment needs recompilation ──
  // Frozen + cached key: reuse (shapes can't change). Otherwise: compute and cache.
  // NOTE: cachedShapeKey is only set AFTER successful compilation (below), not here.
  // Setting it before compile would cause the cascade to skip compilation for the
  // next backend when the first backend fails (the key would be non-zero but no
  // compiled segment exists in the next backend's cache).
  LongType segShapeKey;
  bool needsCompile;
  if (shapesFrozen_ && seg.exec.cachedShapeKey != 0) {
    segShapeKey = seg.exec.cachedShapeKey;
    needsCompile = false;
  } else {
    segShapeKey = computeSegmentShapeKey(seg, externalArrays, numExt);
    seg.def.shapeKeyState.recordComputed(segShapeKey);
    needsCompile = (seg.exec.executionCount == 1) || seg.def.shapeKeyState.hasDrifted();
  }

  // ── Phase guard: compilation must not happen during REPLAYING ────────────
  if (needsCompile && planPhase_ >= PlanPhase::REPLAYING) {
    DSP_DIAG(COMPILE,
             "ERROR: CPU backend compilation triggered during REPLAYING phase for seg[%d-%d] "
             "(executionCount=%d, planPhase=%d). Demoting plan phase.",
             seg.def.startSlot, seg.def.endSlot, seg.exec.executionCount,
             static_cast<int>(planPhase_));
    REQUIRE_TRUE(false, 0,
                 "DSP phase contract violation: CPU compilation during REPLAYING phase "
                 "for seg[%d-%d].", seg.def.startSlot, seg.def.endSlot);
    demotePlanPhase(PlanPhase::POINTERS_STABLE,
                    "CPU compilation triggered during REPLAYING phase");
  }

  if (needsCompile) {
    DSP_DIAG_SEG(COMPILE, seg.def.startSlot,
                 "seg[%d-%d] needs compile: %s (execCount=%d shapeKey=%lld->%lld backend=%s)",
                 seg.def.startSlot, seg.def.endSlot,
                 seg.exec.executionCount == 1 ? "first-compile" : "shape-key-changed",
                 seg.exec.executionCount,
                 static_cast<long long>(seg.def.shapeKeyState.compiledShapeKey),
                 static_cast<long long>(segShapeKey),
                 backendName);
  }
  if (needsCompile) {
    if (!backend->compileSegment(seg, slots_, externalArrays, numExt,
                                 outputSlots_, totalOutputSlots_, segShapeKey,
                                 numSlots_)) {
      DSP_DIAG(COMPILE, "executeSegmentWithSpecificBackend: backend=%s compile failed for seg[%d-%d]",
                backendName, seg.def.startSlot, seg.def.endSlot);
      return Status::KERNEL_FAILURE;
    }
  }

  if (needsCompile && seg.exec.executionCount == 1) {
    auto audit = backend->getLastCompilationAudit();
    lastCompilationAudit_ = audit;
    bool allCovered = true;
    int compiledCount = 0, nativeHandledCount = 0, uncoveredCount = 0;
    for (const auto& entry : audit) {
      if (entry.wasCompiled) {
        compiledCount++;
      } else if (entry.isNativeHandled) {
        // Backend owns native execution for this op (e.g., mixed-segment interleaving).
        // This is NOT an error — the backend guarantees correct execution.
        nativeHandledCount++;
        DSP_DIAG(COMPILE, "%s VALIDATION: slot %d (%s) natively handled by backend: %s",
                  backendName, entry.slotIndex, entry.opName.c_str(), entry.reason.c_str());
      } else {
        allCovered = false;
        uncoveredCount++;
        DSP_DIAG(COMPILE, "%s VALIDATION: slot %d (%s) was NOT compiled: %s",
                  backendName, entry.slotIndex, entry.opName.c_str(), entry.reason.c_str());
      }
    }
    if (!allCovered) {
      seg.exec.compilationFailed = true;
      DSP_THROW_SEG(COMPILE, seg.def.startSlot,
                    "%s VALIDATION FAILURE: segment [%d-%d] has %d ops not covered by backend. "
                    "Fix the backend to compile or natively handle all ops — silent fallback is not permitted.",
                    backendName, seg.def.startSlot, seg.def.endSlot, uncoveredCount);
    } else {
      DSP_DIAG_SEG(COMPILE, seg.def.startSlot,
                   "%s VALIDATION OK: seg[%d-%d] all %d ops covered "
                   "(compiled=%d nativeHandled=%d)",
                   backendName, seg.def.startSlot, seg.def.endSlot, (int)audit.size(),
                   compiledCount, nativeHandledCount);
    }
  }

  // Only update compiledShapeKey when compilation actually occurred.
  // Without this guard, a no-compile reuse call overwrites compiledShapeKey with
  // the current segShapeKey even though the backend's compiled artifacts were
  // produced for a DIFFERENT key. In the CPU cascade (OneDNN → OpenVINO), the
  // second backend reads compiledShapeKey via neverCompiled() / hasDrifted() —
  // an unconditional write here masks compile failures from the first backend.
  if (needsCompile) {
    seg.def.shapeKeyState.markCompiled(segShapeKey);
  }
  // tl_graphExecutionActive must NOT be set here — it is a CUDA-graph-capture
  // guard that suppresses frees and skips syncs. This function drives non-capture
  // paths (CPU backends, Triton warmup). Capture manages the flag internally.
  DSP_DIAG(EXECUTE, "PRE-EXECUTE: seg[%d-%d] backend=%s shapeKey=%lld",
           seg.def.startSlot, seg.def.endSlot, backendName, (long long)segShapeKey);

#if HAVE_ONEDNN
  // For OneDNN mixed segments: install a NativeSlotExecutor so the backend can
  // call back into the plan's slot-by-slot path for unmappable op ranges.
  // The executor is thread-local (mirrors the Triton orderedRangeExecutor_ model)
  // and must be cleared after executeSegment returns.
  bool installedOneDnnNativeExecutor = false;
  auto* onednnBackend = dynamic_cast<OneDnnGraphBackend*>(backend);
  if (onednnBackend != nullptr) {
    onednnBackend->setNativeSlotExecutor(
        [this, &externalArrays, numExt, &stream](int nativeStart, int nativeEnd) -> Status {
          // Build a temporary segment spanning [nativeStart, nativeEnd] and
          // execute it slot-by-slot. We reuse the existing segment infrastructure
          // so that shape caches, control flow, and all other slot logic work correctly.
          // We construct a minimal GraphSegment on the stack to avoid heap allocation.
          GraphSegment nativeSeg;
          nativeSeg.def.startSlot = nativeStart;
          nativeSeg.def.endSlot   = nativeEnd;
          nativeSeg.def.isCapturable = false;
          // Initialize exec state so executeSegmentSlotBySlot doesn't fail phase checks
          nativeSeg.exec.executionCount = 1;  // Past warmup: skip warmup logic inside slotexec
          DSP_DIAG(FALLBACK, "OneDNN NativeSlotExecutor: executing native range [%d-%d] via slot-by-slot",
                   nativeStart, nativeEnd);
          return executeSegmentSlotBySlot(nativeSeg, externalArrays, numExt, stream);
        });
    installedOneDnnNativeExecutor = true;
    DSP_DIAG(EXECUTE, "OneDNN NativeSlotExecutor installed for seg[%d-%d]",
             seg.def.startSlot, seg.def.endSlot);
  }
#endif

#if HAVE_OPENVINO
  bool installedOpenVinoNativeExecutor = false;
  auto* openvinoBackend = dynamic_cast<OpenVinoGraphBackend*>(backend);
  if (openvinoBackend != nullptr) {
    openvinoBackend->setNativeSlotExecutor(
        [this, &externalArrays, numExt, &stream](int nativeStart, int nativeEnd) -> Status {
          GraphSegment nativeSeg;
          nativeSeg.def.startSlot = nativeStart;
          nativeSeg.def.endSlot   = nativeEnd;
          nativeSeg.def.isCapturable = false;
          nativeSeg.exec.executionCount = 1;
          DSP_DIAG(FALLBACK, "OpenVINO NativeSlotExecutor: executing native range [%d-%d] via slot-by-slot",
                   nativeStart, nativeEnd);
          return executeSegmentSlotBySlot(nativeSeg, externalArrays, numExt, stream);
        });
    installedOpenVinoNativeExecutor = true;
    DSP_DIAG(EXECUTE, "OpenVINO NativeSlotExecutor installed for seg[%d-%d]",
             seg.def.startSlot, seg.def.endSlot);
  }
#endif

  auto status = backend->executeSegment(seg, slots_, externalArrays, numExt,
                                         outputSlots_, totalOutputSlots_, stream);

#if HAVE_ONEDNN
  if (installedOneDnnNativeExecutor && onednnBackend != nullptr) {
    onednnBackend->clearNativeSlotExecutor();
    DSP_DIAG(EXECUTE, "OneDNN NativeSlotExecutor cleared for seg[%d-%d]",
             seg.def.startSlot, seg.def.endSlot);
  }
#endif

#if HAVE_OPENVINO
  if (installedOpenVinoNativeExecutor && openvinoBackend != nullptr) {
    openvinoBackend->clearNativeSlotExecutor();
    DSP_DIAG(EXECUTE, "OpenVINO NativeSlotExecutor cleared for seg[%d-%d]",
             seg.def.startSlot, seg.def.endSlot);
  }
#endif

  DSP_DIAG(EXECUTE, "POST-EXECUTE: seg[%d-%d] backend=%s status=%d",
           seg.def.startSlot, seg.def.endSlot, backendName, (int)status);

  DSP_DIAG(EXECUTE, "executeSegmentWithSpecificBackend: exec%d seg[%d-%d]: backend=%s status=%d(%s)",
            seg.exec.executionCount, seg.def.startSlot, seg.def.endSlot, backendName,
            static_cast<int>(status), statusName_seg(status));

  if (status == Status::OK) {
    // Cache the shape key only after successful compile+execute so the cascade
    // doesn't skip compilation for the next backend when the current one fails.
    if (shapesFrozen_) {
      seg.exec.cachedShapeKey = segShapeKey;
    }
    seg.exec.executionCount++;
    totalGraphReplays_++;

    // ── Segment boundary validation (warmup only, backend path) ──────────
    if (executeCount_ < 4) {
      char segErr[512] = {};
      int segInvalid = validateSlotRange(
          seg.def.startSlot, seg.def.endSlot,
          outputSlots_, totalOutputSlots_,
          executeCount_, static_cast<int>(planPhase_),
          segErr, sizeof(segErr));
      if (segInvalid > 0) {
        DSP_THROW(MEMORY,
                 "SEGMENT_BOUNDARY_INVALID: %d invalid array(s) at end of seg[%d-%d] "
                 "(backend=%s, segExecCount=%d): %s",
                 segInvalid, seg.def.startSlot, seg.def.endSlot,
                 backendName, seg.exec.executionCount, segErr);
      }
    }
  }

  return status;
}

// ─── Segment execution: slot-by-slot ─────────────────────────────────────────

// ─── Control flow helpers ────────────────────────────────────────────────────

namespace {

// Resolve an input for a control flow slot
inline NDArray* resolveCfInput(NativeSlot& slot, int inputIdx,
                               NDArray** outputSlots, int totalOutputSlots,
                               NDArray** externalInputs, int numExt) {
  if (inputIdx < 0 || inputIdx >= slot.wiring.numInputs) return nullptr;
  int srcIdx = slot.wiring.inputSourceIndices[inputIdx];
  if (srcIdx >= 0) {
    return (srcIdx < totalOutputSlots) ? outputSlots[srcIdx] : nullptr;
  } else {
    int extIdx = -(srcIdx + 1);
    return (extIdx < numExt) ? externalInputs[extIdx] : nullptr;
  }
}

// Check if any input from an output slot is dead.
// In TF-style control flow, if ANY input comes from a dead Switch branch,
// the op is on that dead branch and must be skipped entirely.
// External inputs (srcIdx < 0) don't participate in dead propagation.
inline bool anyInputDead(NativeSlot& slot, bool* slotIsDead, int slotIsDeadSize) {
  for (int i = 0; i < slot.wiring.numInputs; i++) {
    int srcIdx = slot.wiring.inputSourceIndices[i];
    if (srcIdx >= 0 && srcIdx < slotIsDeadSize && slotIsDead[srcIdx]) {
      return true;
    }
  }
  return false;
}

// Mark all outputs of a slot as dead
inline void markOutputsDead(NativeSlot& slot, bool* slotIsDead, int slotIsDeadSize) {
  for (int i = 0; i < slot.wiring.numOutputs; i++) {
    int si = slot.wiring.outputSlotIndices[i];
    if (si >= 0 && si < slotIsDeadSize) slotIsDead[si] = true;
  }
}

// Forward input[0] to all outputs (identity operation for Enter/Exit/LoopCond/NextIteration)
inline void forwardInput(NativeDynamicShapePlan* plan, NativeSlot& slot, NDArray** outputSlots,
                         int totalOutputSlots, NDArray** externalInputs, int numExt,
                         const char* tag) {
  NDArray* input = resolveCfInput(slot, 0, outputSlots, totalOutputSlots, externalInputs, numExt);
  for (int i = 0; i < slot.wiring.numOutputs; i++) {
    int si = slot.wiring.outputSlotIndices[i];
    if (si >= 0 && si < totalOutputSlots) {
      plan->writeOutputSlot(si, input, tag);
    }
  }
}

#ifdef SD_CUDA
// Verify helper: log control flow slot output mutations
inline void verifyCfSlotWrite(int stepIdx, const char* cfType, const char* opName,
                               NDArray** outputSlots, int* outputSlotIndices,
                               int numOutputs, int totalOutputSlots) {
  if (!Environment::getInstance().tritonVerifyKernels()) return;
  for (int i = 0; i < numOutputs; i++) {
    int si = outputSlotIndices[i];
    if (si < 0 || si >= totalOutputSlots) continue;
    NDArray* out = outputSlots[si];
    if (out == nullptr) {
      DSP_DIAG(VERIFY, "SLOT_WRITE slot=%d tag=CF_FORWARD cf=%s op=%s (nullptr/dead)", si, cfType, opName);
    } else {
      DSP_DIAG(VERIFY, "SLOT_WRITE slot=%d tag=CF_FORWARD cf=%s op=%s dtype=%s len=%lld addr=%p",
                si, cfType, opName,
                DataTypeUtils::asString(out->dataType()).c_str(),
                (long long)out->lengthOf(), DSP_BUF(out));
    }
  }
}
#endif

}  // namespace

Status NativeDynamicShapePlan::executeSegmentSlotBySlot(
    GraphSegment& seg, NDArray** externalArrays, int numExt, void* stream) {
  DSP_REQUIRE_PLAN_PHASE_AT_LEAST(PlanPhase::SLOT_BY_SLOT, "executeSegmentSlotBySlot");
  DSP_DIAG_SEG(EXECUTE, seg.def.startSlot,
               "executeSegmentSlotBySlot: ENTER seg[%d-%d] size=%d execCount=%d capturable=%d compilationFailed=%d",
               seg.def.startSlot, seg.def.endSlot, seg.def.endSlot - seg.def.startSlot + 1,
               seg.exec.executionCount, seg.def.isCapturable ? 1 : 0, seg.exec.compilationFailed ? 1 : 0);

  // Skip prezero in frozen steady-state: output buffers are reused and ops
  // fully overwrite them. prezero iterates ALL slots in the segment doing
  // memset on each qualifying output — pure overhead for decode.
  if (!(shapesFrozen_ && executeCount_ >= 2)) {
    prezeroSegmentOutputs(seg, stream);
  }

  // Reset per-segment allocation counters
  tl_dspAllocBytes = 0; tl_dspFreeBytes = 0;
  tl_dspAllocCount = 0; tl_dspFreeCount = 0; tl_dspFreeSkipCount = 0;
  bool streamIsCapturing = false;
#ifdef SD_CUDA
  if (stream != nullptr) {
    cudaStreamCaptureStatus capStat = cudaStreamCaptureStatusNone;
    cudaStreamIsCapturing(*static_cast<cudaStream_t*>(stream), &capStat);
    streamIsCapturing = (capStat != cudaStreamCaptureStatusNone);
  }
#endif

  // Dead-slot flags are reset once per plan execution (in the main execute loop),
  // NOT per segment — dead flags from Switch in seg N must persist to affect
  // ops in seg N+1.

  int stepIdx = seg.def.startSlot;
  int loopIterations = 0;

  // ── CPU Frozen Replay Fast Path ────────────────────────────────────────
  // When shapes are frozen, past warmup, no control flow in this segment,
  // and a FunctionalReplayHandle is ready with recorded executable slots,
  // iterate ONLY the recorded slots instead of the full slot range.
  // This skips: control flow dispatch, dead propagation, batched GEMM checks,
  // diagnostic logging, and all non-executable slots (frozen constants,
  // identity ops, fused chain tails) at the outer loop level.
  if (shapesFrozen_ && executeCount_ >= 2 && !hasControlFlow_ &&
      seg.exec.replayHandle && seg.exec.replayHandle->isReady()) {
    auto* funcHandle = dynamic_cast<FunctionalReplayHandle*>(seg.exec.replayHandle.get());
    if (funcHandle && funcHandle->hasExecutableSlotIndices()) {
      const auto& execSlots = funcHandle->getExecutableSlotIndices();
      DSP_DIAG(EXECUTE, "CPU_FROZEN_REPLAY: seg[%d-%d] iterating %d/%d executable slots",
               seg.def.startSlot, seg.def.endSlot,
               (int)execSlots.size(), seg.def.endSlot - seg.def.startSlot + 1);

      for (int idx : execSlots) {
        auto status = executeSlot(idx, externalArrays, numExt, stream);
        if (status != Status::OK) {
          DSP_DIAG(EXECUTE, "CPU_FROZEN_REPLAY: slot %d (%s) failed status=%d",
                   idx, slots_[idx].ident.opName.c_str(), (int)status);
          return status;
        }
      }

      seg.exec.executionCount++;
      funcHandle->replay(nullptr);  // statistics tracking
      return Status::OK;
    }
  }

  while (stepIdx <= seg.def.endSlot) {
    NativeSlot& slot = slots_[stepIdx];

    // STEP_ENTER trace: only during warmup to avoid per-slot string formatting
    // overhead (std::to_string heap alloc) in frozen steady-state decode.
    if (executeCount_ < 2) {
      DSP_DIAG_SLOT(EXECUTE, stepIdx,
                    "STEP_ENTER step=%d op=%s cf=%d numIn=%d numOut=%d outSlots=[%d%s%s] "
                    "identity=%d fusedHead=%d fusedTail=%d fusedLen=%d frozenConst=%d",
                    stepIdx, slot.ident.opName.c_str(),
                    (int)slot.cf.controlFlowType,
                    slot.wiring.numInputs, slot.wiring.numOutputs,
                    slot.wiring.numOutputs > 0 ? slot.wiring.outputSlotIndices[0] : -1,
                    slot.wiring.numOutputs > 1 ? "," : "",
                    slot.wiring.numOutputs > 1 ? std::to_string(slot.wiring.outputSlotIndices[1]).c_str() : "",
                    slot.flags.isIdentityOp ? 1 : 0,
                    slot.fusedChain.isFusedChainHead ? 1 : 0,
                    slot.fusedChain.isFusedChainTail ? 1 : 0,
                    slot.fusedChain.fusedChainLength,
                    slot.frozenConstantSlot() ? 1 : 0);
    }

    // ── Control flow dispatch ────────────────────────────────────────
    if (slot.cf.controlFlowType != CF_NONE) {
      // Dead propagation: if all inputs are dead and this is not a Merge, propagate dead
      if (slot.cf.controlFlowType != CF_MERGE && hasControlFlow_ && slotIsDead_ != nullptr) {
        if (anyInputDead(slot, slotIsDead_, slotIsDeadSize_)) {
          DSP_DIAG_SLOT(EXECUTE, stepIdx,
                        "slot %d (%s) DEAD: propagated from dead input (cf=%d)",
                        stepIdx, slot.ident.opName.c_str(), (int)slot.cf.controlFlowType);
          markOutputsDead(slot, slotIsDead_, slotIsDeadSize_);
          stepIdx++;
          continue;
        }
      }

      switch (slot.cf.controlFlowType) {
        case CF_SWITCH: {
          // Switch: input[0] = data, input[1] = predicate
          // If predicate is true: output[1] = data, output[0] is dead
          // If predicate is false: output[0] = data, output[1] is dead
          NDArray* data = resolveCfInput(slot, 0, outputSlots_, totalOutputSlots_, externalArrays, numExt);
          NDArray* pred = resolveCfInput(slot, 1, outputSlots_, totalOutputSlots_, externalArrays, numExt);
          bool predValue = false;
          if (pred != nullptr && !pred->isEmpty()) {
            predValue = pred->e<bool>(0);
          }
          int liveIdx = predValue ? 1 : 0;
          int deadIdx = predValue ? 0 : 1;
          for (int i = 0; i < slot.wiring.numOutputs; i++) {
            int si = slot.wiring.outputSlotIndices[i];
            if (si >= 0 && si < totalOutputSlots_) {
              if (i == liveIdx) {
                writeOutputSlot(si, data, "cf-switch-live");
                if (slotIsDead_) slotIsDead_[si] = false;
              } else {
                writeOutputSlot(si, nullptr, "cf-switch-dead");
                if (slotIsDead_) slotIsDead_[si] = true;
              }
            }
          }
#ifdef SD_CUDA
          verifyCfSlotWrite(stepIdx, "SWITCH", slot.ident.opName.c_str(),
                            outputSlots_, slot.wiring.outputSlotIndices, slot.wiring.numOutputs, totalOutputSlots_);
#endif
          break;
        }

        case CF_MERGE: {
          // Merge: select first non-dead, non-null input
          NDArray* selected = nullptr;
          for (int i = 0; i < slot.wiring.numInputs; i++) {
            int srcIdx = slot.wiring.inputSourceIndices[i];
            bool isDead = (srcIdx >= 0 && srcIdx < slotIsDeadSize_ && slotIsDead_ && slotIsDead_[srcIdx]);
            if (!isDead) {
              NDArray* inp = resolveCfInput(slot, i, outputSlots_, totalOutputSlots_, externalArrays, numExt);
              if (inp != nullptr) {
                selected = inp;
                break;
              }
            }
          }
          for (int i = 0; i < slot.wiring.numOutputs; i++) {
            int si = slot.wiring.outputSlotIndices[i];
            if (si >= 0 && si < totalOutputSlots_) {
              writeOutputSlot(si, selected, "cf-merge");
              if (slotIsDead_) slotIsDead_[si] = (selected == nullptr);
            }
          }
#ifdef SD_CUDA
          verifyCfSlotWrite(stepIdx, "MERGE", slot.ident.opName.c_str(),
                            outputSlots_, slot.wiring.outputSlotIndices, slot.wiring.numOutputs, totalOutputSlots_);
#endif
          break;
        }

        case CF_ENTER:
        case CF_EXIT:
        case CF_LOOP_COND:
          // Identity: forward input[0] to output[0]
          forwardInput(this, slot, outputSlots_, totalOutputSlots_, externalArrays, numExt,
                       slot.cf.controlFlowType == CF_ENTER ? "cf-enter"
                       : slot.cf.controlFlowType == CF_EXIT ? "cf-exit"
                       : "cf-loop-cond");
#ifdef SD_CUDA
          {
            const char* cfName = (slot.cf.controlFlowType == CF_ENTER) ? "ENTER" :
                                  (slot.cf.controlFlowType == CF_EXIT) ? "EXIT" : "LOOP_COND";
            verifyCfSlotWrite(stepIdx, cfName, slot.ident.opName.c_str(),
                              outputSlots_, slot.wiring.outputSlotIndices, slot.wiring.numOutputs, totalOutputSlots_);
          }
#endif
          break;

        case CF_NEXT_ITERATION: {
          // Forward input[0] to output[0], then jump back to Merge
          forwardInput(this, slot, outputSlots_, totalOutputSlots_, externalArrays, numExt,
                       "cf-next-iter");
#ifdef SD_CUDA
          verifyCfSlotWrite(stepIdx, "NEXT_ITER", slot.ident.opName.c_str(),
                            outputSlots_, slot.wiring.outputSlotIndices, slot.wiring.numOutputs, totalOutputSlots_);
#endif

          if (slot.cf.loopBackTarget >= 0 && slot.cf.loopBackTarget >= seg.def.startSlot) {
            loopIterations++;
            if (loopIterations >= MAX_LOOP_ITERATIONS) {
              DSP_THROW_SEG(EXECUTE, seg.def.startSlot,
                            "loop iteration limit (%d) reached at slot %d (%s) in seg[%d-%d]. "
                            "Possible infinite loop in control flow.",
                            MAX_LOOP_ITERATIONS, stepIdx, slots_[stepIdx].ident.opName.c_str(),
                            seg.def.startSlot, seg.def.endSlot);
            }
            // Clear dead flags for loop body range
            if (slotIsDead_ && slot.cf.loopRegionIndex >= 0 && slot.cf.loopRegionIndex < numLoopRegions_) {
              LoopRegion& lr = loopRegions_[slot.cf.loopRegionIndex];
              for (int s = lr.mergeSlot; s <= lr.bodyEndSlot && s < numSlots_; s++) {
                NativeSlot& bodySlot = slots_[s];
                for (int oi = 0; oi < bodySlot.wiring.numOutputs; oi++) {
                  int si = bodySlot.wiring.outputSlotIndices[oi];
                  if (si >= 0 && si < slotIsDeadSize_) slotIsDead_[si] = false;
                }
              }
            }
            stepIdx = slot.cf.loopBackTarget;
            continue; // jump back to Merge, don't increment stepIdx
          }
          break;
        }

        default:
          break;
      }

      // Release schedule removed: arrays persist (one array per slot, never nullified)

      stepIdx++;
      continue;
    }

    // ── Dead propagation for regular ops in CF graphs ────────────────
    if (hasControlFlow_ && slotIsDead_ != nullptr) {
      if (anyInputDead(slot, slotIsDead_, slotIsDeadSize_)) {
        markOutputsDead(slot, slotIsDead_, slotIsDeadSize_);
        stepIdx++;
        continue;
      }
    }

    // ── Batched GEMM dispatch ─────────────────────────────────────────
    // Strategy: the FIRST member in each group is the trigger.
    // When reached, it executes the entire batch and populates outputs for
    // ALL members. Non-first members are skipped (output already computed).
    // This ensures downstream ops between members see valid outputs.
#ifdef SD_CUDA
    if (!batchedGemmGroups_.empty() && stepIdx < (int)slotToBatchedGemmGroup_.size()) {
      int bgIdx = slotToBatchedGemmGroup_[stepIdx];
      if (bgIdx >= 0 && bgIdx < (int)batchedGemmGroups_.size()) {
        auto& bgGroup = batchedGemmGroups_[bgIdx];
        if (stepIdx == bgGroup.triggerSlot) {
          // This is the trigger (FIRST slot in group) — execute entire batch.
          // All members' inputs are guaranteed available (checked at detection time).
          cudaStream_t execStream = stream ? *static_cast<cudaStream_t*>(stream) : static_cast<cudaStream_t>(nullptr);
          Status batchStatus = executeBatchedGemmGroup(bgIdx, externalArrays, numExt, execStream);

          if (batchStatus == Status::OK) {
            // Release schedule removed: arrays persist (one array per slot)
            stepIdx++;
            continue;
          }
          // Batched GEMM failure is a hard error — do not silently fall back to individual execution
          DSP_THROW(EXECUTE,
                    "batched GEMM group %d failed (status=%d) at slot %d (%s). "
                    "Fix the batched GEMM execution — silent fallback to individual execution is not permitted.",
                    bgIdx, (int)batchStatus, stepIdx, slots_[stepIdx].ident.opName.c_str());
        } else {
          // Non-first member: output already computed by the trigger's batch call.
          // Release schedule removed: arrays persist (one array per slot)
          stepIdx++;
          continue;
        }
      }
    }
#endif

    // ── Outer-level fast skips (frozen steady-state) ─────────────────
    // When shapes are frozen and past warmup, skip trivially no-op slots
    // at the loop level, avoiding executeSlot() function call overhead.
    // This mirrors the GPU compositeReplay gap loop optimization.
    if (shapesFrozen_ && executeCount_ >= 2) {
      // Frozen constant: output never changes, skip entirely
      if (slot.frozenConstantSlot()) {
        bool allPopulated = true;
        for (int o = 0; o < slot.wiring.numOutputs; o++) {
          int si = slot.wiring.outputSlotIndices[o];
          if (si >= 0 && si < totalOutputSlots_ && outputSlots_[si] == nullptr) {
            allPopulated = false;
            break;
          }
        }
        if (allPopulated) {
          stepIdx++;
          continue;
        }
      }

      // Fused chain tail: head already executed the entire chain
      if (slot.fusedChain.isFusedChainTail) {
        stepIdx++;
        continue;
      }

      // Identity op: just alias input → output
      if (slot.flags.isIdentityOp && slot.wiring.numInputs == 1 && slot.wiring.numOutputs >= 1) {
        int srcIdx = slot.wiring.inputSourceIndices[0];
        NDArray* input = nullptr;
        if (srcIdx >= 0 && srcIdx < totalOutputSlots_) {
          input = outputSlots_[srcIdx];
        } else if (srcIdx < 0) {
          int extIdx = -(srcIdx + 1);
          if (extIdx < numExt) input = externalArrays[extIdx];
        }
        if (input != nullptr) {
          for (int o = 0; o < slot.wiring.numOutputs; o++) {
            int si = slot.wiring.outputSlotIndices[o];
            if (si >= 0 && si < totalOutputSlots_) {
              writeOutputSlot(si, input, "outer-identity");
            }
          }
          slot.bumpGeneration();
          stepIdx++;
          continue;
        }
      }
    }

    // ── Normal op execution ──────────────────────────────────────────
    Status status;
    bool retriedAfterTrim = false;
    bool shouldRetry = false;
    do {
      shouldRetry = false;
      try {
        status = executeSlot(stepIdx, externalArrays, numExt, stream);
      } catch (const std::exception& e) {
        std::string msg = e.what();
#ifdef SD_CUDA
        if (!streamIsCapturing &&
            !retriedAfterTrim && (msg.find("cannot allocate") != std::string::npos ||
                                   msg.find("out of memory") != std::string::npos ||
                                   msg.find("Error code: [2]") != std::string::npos)) {
          retriedAfterTrim = true;
          shouldRetry = true;
          DSP_DIAG_SLOT(MEMORY, stepIdx, "slot %d (%s) OOM, trimming pool and retrying...",
                    stepIdx, slots_[stepIdx].ident.opName.c_str());
          cudaGetLastError();
          if (stream) {
            cudaStream_t execStr = *static_cast<cudaStream_t*>(stream);
            cudaStreamSynchronize(execStr);
          }
          cudaStreamSynchronize(static_cast<cudaStream_t>(nullptr));
          {
            cudaMemPool_t pool = nullptr;
            int dev = 0;
            cudaGetDevice(&dev);
            if (cudaDeviceGetMemPool(&pool, dev) == cudaSuccess && pool != nullptr) {
              cudaMemPoolTrimTo(pool, 0);
              DSP_DIAG(MEMORY, "trimmed memory pool on device %d", dev);
            }
          }
          continue;  // retry the slot execution after trimming
        }
#endif
        DSP_THROW(EXECUTE, "slot %d (%s) threw exception: %s",
                  stepIdx, slots_[stepIdx].ident.opName.c_str(), e.what());
      } catch (...) {
        DSP_THROW(EXECUTE, "slot %d (%s) threw unknown exception",
                  stepIdx, slots_[stepIdx].ident.opName.c_str());
      }
    } while (shouldRetry);

    // ── Post-slot output validation ─────────────────────────────────────
    // After each slot executes, verify its outputs are valid before the next
    // slot reads them as inputs. Catches the exact slot that produces an
    // invalid array, rather than discovering it N slots later.
    // Skip in frozen steady-state (executeCount >= 3): all slot outputs have
    // been validated on prior executions. This eliminates ~1883 validateSlotOutputs
    // calls per decode step.
    if (status == Status::OK && executeCount_ < 3) {
      auto& doneSlot = slots_[stepIdx];
      char postSlotErr[512] = {};
      int badOutputs = validateSlotOutputs(
          stepIdx, doneSlot.ident.opName.c_str(),
          outputSlots_, totalOutputSlots_,
          doneSlot.wiring.outputSlotIndices, doneSlot.wiring.numOutputs,
          executeCount_, static_cast<int>(planPhase_),
          postSlotErr, sizeof(postSlotErr));
      if (badOutputs > 0) {
        DSP_THROW(MEMORY,
                 "SLOT_OUTPUT_INVALID: %d invalid output(s) detected AFTER slot %d (%s) "
                 "execution: %s",
                 badOutputs, stepIdx, doneSlot.ident.opName.c_str(), postSlotErr);
      }
    }

    // ── Diagnostic: per-slot CUDA error check on warmup execution ──────
    // On the first execution of each segment (warmup), synchronize the device
    // after every slot to catch latent CUDA kernel errors (error 700) at the
    // exact slot that caused them, rather than discovering them hundreds of
    // slots later during an unrelated cudaMallocAsync call.
    // This is expensive (blocks GPU pipeline) but essential for diagnosing
    // stale-pointer bugs in restored cached plan handles.
    //
    // SKIP when tl_graphCaptureStream is set: indicates another stream on this
    // thread is being captured. cudaDeviceSynchronize is device-wide and fails
    // with error 900 if ANY stream is capturing. Gap ops execute on a dedicated
    // non-capturing stream during Triton graph capture, but cudaDeviceSynchronize
    // would still try to sync the capturing stream.
#ifdef SD_CUDA
    if (DSP_DIAG_ENABLED(EXECUTE) && status == Status::OK && seg.exec.executionCount == 0
        && !streamIsCapturing && tl_graphCaptureStream == nullptr) {
      cudaError_t syncErr = cudaDeviceSynchronize();
      if (syncErr != cudaSuccess) {
        // Log all inputs to the failing slot
        auto& faultSlot = slots_[stepIdx];
        for (int i = 0; i < faultSlot.wiring.numInputs; i++) {
          int srcIdx = faultSlot.wiring.inputSourceIndices[i];
          NDArray* inp = nullptr;
          if (srcIdx >= 0 && srcIdx < totalOutputSlots_) {
            inp = outputSlots_[srcIdx];
          } else if (srcIdx < 0) {
            int extIdx = -(srcIdx + 1);
            if (extIdx >= 0 && extIdx < numExt) inp = externalArrays[extIdx];
          }
          if (inp != nullptr && inp->dataBuffer() != nullptr) {
            DSP_DIAG(EXECUTE,
                "  FAULT INPUT[%d] srcIdx=%d: shape=%s special=%p primary=%p "
                "len=%lld db=%p closed=%d devId=%d",
                i, srcIdx, ShapeUtils::shapeAsString(inp).c_str(),
                inp->dataBuffer()->special(), inp->dataBuffer()->primary(),
                (long long)inp->lengthOf(), (void*)inp->dataBuffer(),
                inp->dataBuffer()->isClosed() ? 1 : 0,
                inp->dataBuffer()->deviceId());
          } else {
            DSP_DIAG(EXECUTE, "  FAULT INPUT[%d] srcIdx=%d: %s",
                     i, srcIdx, inp ? "db=null" : "null");
          }
        }
        // Log outputs of the failing slot
        for (int i = 0; i < faultSlot.wiring.numOutputs; i++) {
          int si = faultSlot.wiring.outputSlotIndices[i];
          NDArray* out = (si >= 0 && si < totalOutputSlots_) ? outputSlots_[si] : nullptr;
          if (out != nullptr && out->dataBuffer() != nullptr) {
            DSP_DIAG(EXECUTE,
                "  FAULT OUTPUT[%d] slotIdx=%d: shape=%s special=%p len=%lld",
                i, si, ShapeUtils::shapeAsString(out).c_str(),
                out->dataBuffer()->special(), (long long)out->lengthOf());
          } else {
            DSP_DIAG(EXECUTE, "  FAULT OUTPUT[%d] slotIdx=%d: %s",
                     i, si, out ? "db=null" : "null");
          }
        }
        DSP_THROW_CUDA(EXECUTE, syncErr,
                       "CUDA ERROR 700 DIAGNOSTIC: cudaDeviceSynchronize after slot %d (%s) "
                       "returned error %d. This kernel accessed invalid GPU memory. "
                       "seg=[%d-%d] execCount=%d shapesFrozen=%d",
                       stepIdx, slots_[stepIdx].ident.opName.c_str(),
                       static_cast<int>(syncErr),
                       seg.def.startSlot, seg.def.endSlot, executeCount_, static_cast<int>(shapesFrozen_));
      }
    }
#endif

    if (status != Status::OK) {
      // Record the failure in the execution flow log FIRST so it's
      // available even if the exception below is caught by a caller.
      auto* execCtx = static_cast<PlanExecutionContext*>(activeExecutionContext());
      if (execCtx != nullptr) {
        execCtx->recordFlow(PlanExecutionContext::FlowEventType::SLOT_EXEC_FAIL,
                             stepIdx, static_cast<int>(status));
        // Dump entire flow log on failure so we can see exactly what
        // happened this execution (auto-seal, frozen constants, etc.)
        execCtx->dumpFlowLog(executeCount_);
      }

      char buf[1024];
      const char* existingMsg =
          sd::LaunchContext::defaultContext()->errorReference()->errorMessage();
      if (existingMsg != nullptr && existingMsg[0] != '\0') {
        snprintf(buf, sizeof(buf), "slot %d (%s) failed with status %d: %s",
                 stepIdx, slots_[stepIdx].ident.opName.c_str(),
                 static_cast<int>(status), existingMsg);
      } else {
        snprintf(buf, sizeof(buf), "slot %d (%s) failed with status %d",
                 stepIdx, slots_[stepIdx].ident.opName.c_str(), static_cast<int>(status));
      }

      // Log full input details for the failing slot
      auto& failedSlot = slots_[stepIdx];
      for (int i = 0; i < failedSlot.wiring.numInputs; i++) {
        int srcIdx = failedSlot.wiring.inputSourceIndices[i];
        if (srcIdx >= 0) {
          NDArray* inp = (srcIdx < totalOutputSlots_ ? outputSlots_[srcIdx] : nullptr);
          if (inp != nullptr) {
            // Protect rankOf() call — if shapeInfo is null, rankOf() would throw
            // and propagate out of this catch handler, causing cascading failures.
            try {
              // Check if the source slot is frozen — if so, that's critical diagnostic info
              bool srcIsFrozen = false;
              for (int fs = 0; fs < numSlots_; fs++) {
                for (int fo = 0; fo < slots_[fs].wiring.numOutputs; fo++) {
                  if (slots_[fs].wiring.outputSlotIndices[fo] == srcIdx) {
                    srcIsFrozen = slots_[fs].frozenConstantSlot();
                    break;
                  }
                }
                if (srcIsFrozen) break;
              }
              DSP_DIAG(EXECUTE, "  input[%d] from outputSlot[%d]: rank=%lld shape=%s "
                        "shapeInfo=%p db=%p frozenSrc=%d",
                        i, srcIdx, (long long)inp->rankOf(),
                        ShapeUtils::shapeAsString(inp).c_str(),
                        (void*)inp->shapeInfo(), (void*)inp->dataBuffer(),
                        srcIsFrozen ? 1 : 0);
            } catch (...) {
              DSP_DIAG(EXECUTE, "  input[%d] from outputSlot[%d]: ptr=%p (shapeInfo INVALID)",
                        i, srcIdx, (void*)inp);
            }
          } else {
            DSP_DIAG(EXECUTE, "  input[%d] from outputSlot[%d]: null", i, srcIdx);
          }
        } else {
          DSP_DIAG(EXECUTE, "  input[%d] from external[%d]", i, -(srcIdx + 1));
        }
      }

      // Log output slot info for the failing slot
      for (int o = 0; o < failedSlot.wiring.numOutputs; o++) {
        int outIdx = failedSlot.wiring.outputSlotIndices[o];
        NDArray* out = (outIdx >= 0 && outIdx < totalOutputSlots_) ? outputSlots_[outIdx] : nullptr;
        if (out != nullptr) {
          try {
            DSP_DIAG(EXECUTE, "  output[%d] outputSlot[%d]: rank=%lld shape=%s db=%p",
                      o, outIdx, (long long)out->rankOf(),
                      ShapeUtils::shapeAsString(out).c_str(),
                      (void*)out->dataBuffer());
          } catch (...) {
            DSP_DIAG(EXECUTE, "  output[%d] outputSlot[%d]: ptr=%p (shapeInfo INVALID)",
                      o, outIdx, (void*)out);
          }
        } else {
          DSP_DIAG(EXECUTE, "  output[%d] outputSlot[%d]: %s", o, outIdx,
                    outIdx >= 0 ? "null" : "invalid-idx");
        }
      }

#ifdef SD_CUDA
      cudaGetLastError();
#endif
      DSP_THROW(EXECUTE, "%s", buf);
    }

    // Classify ownership for all outputs produced by this slot.
    // Skip in frozen steady-state: ownership is stable after warmup.
    if (slotOwnership_ != nullptr && executeCount_ < 2) {
      for (int o = 0; o < slot.wiring.numOutputs; o++) {
        int si = slot.wiring.outputSlotIndices[o];
        if (si >= 0 && si < totalOutputSlots_ && outputSlots_[si] != nullptr) {
          classifyAndUpdateOwnership(
              slotOwnership_[si], outputSlots_[si], si,
              externalArrays, numExt,
              outputSlots_, totalOutputSlots_,
              slotOwnership_);
        }
      }
    }

    // Record op for FunctionalReplayHandle capture — skip slots that will be
    // handled by outer-level fast skips in frozen steady-state (executeCount >= 2).
    // Frozen constants, identity ops, and fused chain tails never need kernel
    // execution after warmup, so excluding them from executableSlotIndices lets
    // the CPU_FROZEN_REPLAY path iterate a smaller set.
    if (seg.exec.replayHandle && seg.exec.replayHandle->getState() == ReplayState::CAPTURING) {
      bool skipRecord = slot.frozenConstantSlot()
                     || slot.fusedChain.isFusedChainTail
                     || (slot.flags.isIdentityOp && slot.wiring.numInputs == 1);
      if (!skipRecord) {
        auto* funcHandle = dynamic_cast<FunctionalReplayHandle*>(seg.exec.replayHandle.get());
        if (funcHandle) funcHandle->recordOp(slot.ident.op, stepIdx);
      }
    }

    // Release schedule removed: arrays persist (one array per slot, never nullified).
    // Same plan = same shapes. Arrays allocated on first execution, reused forever.

    stepIdx++;
  }

  if (!viewProducerDetectionDone_) {
    viewProducerDetectionDone_ = true;
    int viewCount = 0;
    for (int i = 0; i < totalOutputSlots_; i++) {
      if (slotIsViewProducer_[i]) viewCount++;
    }
    DSP_DIAG(SHAPE, "view producer detection done: %d/%d output slots are view producers",
              viewCount, totalOutputSlots_);
  }

  // Log per-segment allocation/free summary
  DSP_DIAG(MEMORY, "SEG-MEM: seg[%d-%d] exec=%d alloc=%lldMB(%d) free=%lldMB(%d) freeSkip=%d net=%lldMB",
           seg.def.startSlot, seg.def.endSlot, seg.exec.executionCount,
           tl_dspAllocBytes / (1024*1024), tl_dspAllocCount,
           tl_dspFreeBytes / (1024*1024), tl_dspFreeCount, tl_dspFreeSkipCount,
           (tl_dspAllocBytes - tl_dspFreeBytes) / (1024*1024));

  seg.exec.executionCount++;

  // ── Segment boundary validation (warmup only) ──────────────────────────
  // After a slot-by-slot segment completes, verify all outputs in the segment
  // range are valid before downstream segments read them as inputs.
  // Gated to first 4 executions only — in frozen steady-state, shapes and
  // buffers are stable and this O(totalSlots) scan is pure overhead.
  if (executeCount_ < 4) {
    char segErr[512] = {};
    int segInvalid = validateSlotRange(
        seg.def.startSlot, seg.def.endSlot,
        outputSlots_, totalOutputSlots_,
        executeCount_, static_cast<int>(planPhase_),
        segErr, sizeof(segErr));
    if (segInvalid > 0) {
      DSP_THROW(MEMORY,
               "SEGMENT_BOUNDARY_INVALID: %d invalid array(s) at end of seg[%d-%d] "
               "(slot-by-slot, segExecCount=%d): %s",
               segInvalid, seg.def.startSlot, seg.def.endSlot,
               seg.exec.executionCount, segErr);
    }
  }

  return Status::OK;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Emulated Graph Replay
// ═══════════════════════════════════════════════════════════════════════════════
//
// Executes ops slot-by-slot but emulates the full graph replay lifecycle:
//   executionCount == 0 (WARMUP): Record baseline shape key + address snapshot
//   executionCount == 1 (EMULATED_CAPTURE): Verify shape/address stability
//   executionCount >= 2 (EMULATED_REPLAY): Steady-state with stability tracking
//
// Emits DSP_DIAG_EMULATED_REPLAY diagnostics at every phase, reporting what a
// real CUDA graph replay backend would see. This lets users diagnose graph
// replay failures without needing actual CUDA graph capture.
// ═══════════════════════════════════════════════════════════════════════════════

LongType NativeDynamicShapePlan::computeSegmentInputAddrKeyPortable(
    GraphSegment& seg, NDArray** externalInputs, int numExt) {
  // FNV-1a hash of buffer addresses for all CROSS-SEGMENT inputs plus all
  // non-PLACEHOLDER external inputs (weights / constants). On CUDA, uses
  // specialBuffer(); on CPU, uses primaryBuffer(). Address changes between
  // executions indicate the graph would have stale pointers.
  //
  // PLACEHOLDER externals (position_ids, attention_mask, …) are excluded —
  // Java allocates fresh arrays for them every decode step, so their
  // pointers always change; hashing them would pin the plan below
  // POINTERS_STABLE forever. Weights / constants are device-authoritative
  // and supposed to be stable; a user-visible close+associateArrayWithVariable
  // rebind on one of them should invalidate the cached replay rather than
  // silently replay against freed device memory.
  uint64_t hash = dsp::FNV1A64_OFFSET_BASIS;
  auto mix = [&hash](uintptr_t val) {
    dsp::fnv1aMixValue(hash, static_cast<uint64_t>(val));
  };

  // externalInputIsVariable_ is populated at plan-load time, not at
  // shape-freeze time — in explicit graphExecutionMode (CUDA_GRAPHS,
  // TRITON, NVRTC_JIT, PTX_JIT) the plan never goes through AUTO_SEAL so
  // shapesFrozen_ stays false even once graph replay is active. Gate on
  // the vector being non-empty only.
  const bool canClassifyExternals = !externalInputIsVariable_.empty();

  for (int s = seg.def.startSlot; s <= seg.def.endSlot; s++) {
    NativeSlot& slot = slots_[s];
    for (int i = 0; i < slot.wiring.numInputs; i++) {
      int srcIdx = slot.wiring.inputSourceIndices[i];
      if (srcIdx < 0) {
        if (!canClassifyExternals) continue;
        int extIdx = -(srcIdx + 1);
        if (extIdx < 0 || extIdx >= numExt) continue;
        if (extIdx >= static_cast<int>(externalInputIsVariable_.size())) continue;
        if (externalInputIsVariable_[extIdx]) continue;  // skip placeholders
        NDArray* extArr = externalInputs[extIdx];
        if (extArr == nullptr) continue;
#if defined(SD_CUDA)
        mix(reinterpret_cast<uintptr_t>(extArr->specialBuffer()));
#else
        mix(reinterpret_cast<uintptr_t>(extArr->buffer()));
#endif
        continue;
      }
      NDArray* arr = nullptr;
      if (srcIdx < totalOutputSlots_) {
        arr = outputSlots_[srcIdx];
      }
      if (arr != nullptr) {
#if defined(SD_CUDA)
        mix(reinterpret_cast<uintptr_t>(arr->specialBuffer()));
#else
        mix(reinterpret_cast<uintptr_t>(arr->buffer()));
#endif
      } else {
        mix(0);  // cross-segment nullptr sentinel
      }
    }
  }
  return hash;
}

Status NativeDynamicShapePlan::executeSegmentEmulatedReplay(
    GraphSegment& seg, NDArray** externalArrays, int numExt, void* stream) {
  DSP_REQUIRE_PLAN_PHASE_AT_LEAST(PlanPhase::SLOT_BY_SLOT, "executeSegmentEmulatedReplay");

  int segSize = seg.def.endSlot - seg.def.startSlot + 1;
  int execCount = seg.exec.executionCount;

  // ── Phase determination ─────────────────────────────────────────────────
  const char* phaseName;
  if (execCount == 0) {
    DSP_SET_SEG_PHASE(seg, ExecutionPhase::WARMUP, "emulated_replay_exec0");
    phaseName = "WARMUP";
  } else if (execCount == 1) {
    DSP_SET_SEG_PHASE(seg, ExecutionPhase::COMPILING, "emulated_replay_capture");  // "capture" equivalent
    phaseName = "EMULATED_CAPTURE";
  } else {
    DSP_SET_SEG_PHASE(seg, ExecutionPhase::REPLAYING, "emulated_replay_steady");
    phaseName = "EMULATED_REPLAY";
  }

  DSP_DIAG(EMULATED_REPLAY,
           "EMULATED seg[%d-%d] phase=%s execCount=%d slots=%d capturable=%d frozen=%d",
           seg.def.startSlot, seg.def.endSlot, phaseName, execCount, segSize,
           seg.def.isCapturable ? 1 : 0, shapesFrozen_ ? 1 : 0);

  // ── Gap 1: Fast path — skip key recomputation when stable ──────────────
  // When argTableStable was set on the previous execution (both shape and addr
  // keys matched), skip the expensive hash computations and go straight to
  // slot-by-slot execution. This eliminates shape key overhead (~5-10us per
  // segment) that real graph replay also avoids.
  bool fastPath = false;
  if (execCount >= 2 && seg.exec.argTableStable) {
    fastPath = true;
    DSP_DIAG(EMULATED_REPLAY,
             "  FAST PATH: argTableStable=true from previous step, skipping key recomputation");
  }

  LongType currentShapeKey = 0;
  LongType currentAddrKey = 0;

  if (!fastPath) {
    // ── Compute shape key ──────────────────────────────────────────────────
    currentShapeKey = computeSegmentShapeKey(seg, externalArrays, numExt);
    currentAddrKey = computeSegmentInputAddrKeyPortable(seg, externalArrays, numExt);
  }

  if (execCount == 0) {
    // ══════════════════════════════════════════════════════════════════════
    // WARMUP: baseline keys + fusion analysis + capture buffer sizing + DOT
    // ══════════════════════════════════════════════════════════════════════
    seg.exec.cachedShapeKey = currentShapeKey;
    seg.exec.capturedInputAddrKey = currentAddrKey;
    seg.exec.argTableStable = false;
    seg.exec.addrKeyStableCount = 0;
    seg.exec.slotAddrStableCount = 0;

    DSP_DIAG(EMULATED_REPLAY,
             "  WARMUP baseline: shapeKey=0x%llx addrKey=0x%llx",
             (long long)currentShapeKey, (long long)currentAddrKey);

    // ── Gap 3: Capture buffer sizing (byte-level) ────────────────────────
    int numPlaceholders = 0, numConstants = 0, numVariables = 0;
    size_t placeholderBytes = 0, constantBytes = 0, variableBytes = 0;
    // Track unique external indices to avoid double-counting
    std::unordered_set<int> seenExt;

    for (int s = seg.def.startSlot; s <= seg.def.endSlot; s++) {
      NativeSlot& slot = slots_[s];
      for (int i = 0; i < slot.wiring.numInputs; i++) {
        int srcIdx = slot.wiring.inputSourceIndices[i];
        if (srcIdx < 0) {
          int extIdx = -(srcIdx + 1);
          if (seenExt.count(extIdx)) continue;
          seenExt.insert(extIdx);

          int8_t srcType = slot.wiring.inputSourceTypes[i];
          size_t bytes = 0;
          if (extIdx < numExt && externalArrays[extIdx] != nullptr) {
            bytes = externalArrays[extIdx]->lengthOf() * externalArrays[extIdx]->sizeOfT();
          }

          if (srcType == SOURCE_PLACEHOLDER) {
            numPlaceholders++;
            placeholderBytes += bytes;
          } else if (srcType == SOURCE_CONSTANT) {
            numConstants++;
            constantBytes += bytes;
          } else if (srcType == SOURCE_VARIABLE) {
            numVariables++;
            variableBytes += bytes;
          }
        }
      }
    }

    DSP_DIAG(EMULATED_REPLAY,
             "  capture buffers: %d placeholders (%zuKB staging needed), "
             "%d constants (%zuKB direct ref), %d variables (%zuKB direct ref if frozen)",
             numPlaceholders, placeholderBytes / 1024,
             numConstants, constantBytes / 1024,
             numVariables, variableBytes / 1024);

    // Per-placeholder detail for large inputs
    if (DSP_DIAG_ENABLED(EMULATED_REPLAY)) {
      seenExt.clear();
      for (int s = seg.def.startSlot; s <= seg.def.endSlot; s++) {
        NativeSlot& slot = slots_[s];
        for (int i = 0; i < slot.wiring.numInputs; i++) {
          int srcIdx = slot.wiring.inputSourceIndices[i];
          if (srcIdx < 0) {
            int extIdx = -(srcIdx + 1);
            if (seenExt.count(extIdx)) continue;
            seenExt.insert(extIdx);
            if (slot.wiring.inputSourceTypes[i] == SOURCE_PLACEHOLDER &&
                extIdx < numExt && externalArrays[extIdx] != nullptr) {
              auto* arr = externalArrays[extIdx];
              size_t bytes = arr->lengthOf() * arr->sizeOfT();
              DSP_DIAG(EMULATED_REPLAY,
                       "    ext[%d] PLACEHOLDER shape=[%s] dtype=%d bytes=%zu",
                       extIdx, ShapeUtils::shapeAsString(arr).c_str(),
                       (int)arr->dataType(), bytes);
            }
          }
        }
      }
    }

    // ── Gap 2: Kernel fusion analysis ────────────────────────────────────
    int numIdentity = 0, numViewOps = 0, numFusedChains = 0, numFusedTails = 0;
    int numInPlaceFused = 0, numDataDependent = 0, numControlFlow = 0;
    int numMatmul = 0, numElementwise = 0, numOther = 0;
    int totalFusedChainOps = 0;

    for (int s = seg.def.startSlot; s <= seg.def.endSlot; s++) {
      NativeSlot& slot = slots_[s];
      if (slot.flags.isIdentityOp)     numIdentity++;
      if (slot.flags.isViewCapableOp)  numViewOps++;
      if (slot.fusedChain.isFusedChainHead) { numFusedChains++; totalFusedChainOps += slot.fusedChain.fusedChainLength; }
      if (slot.fusedChain.isFusedChainTail) numFusedTails++;
      if (slot.flags.inPlaceFused)     numInPlaceFused++;
      if (slot.flags.isDataDependent)  numDataDependent++;
      if (slot.cf.controlFlowType != CF_NONE) numControlFlow++;

      // Classify by op name heuristic
      const auto& name = slot.ident.opName;
      if (name.find("matmul") != std::string::npos || name.find("mmul") != std::string::npos ||
          name.find("gemm") != std::string::npos || name.find("batched_gemm") != std::string::npos) {
        numMatmul++;
      } else if (slot.flags.isIdentityOp || slot.flags.isViewCapableOp || slot.fusedChain.isFusedChainTail) {
        // Already counted above — these are "free" ops
      } else {
        // Heuristic: ops with no iArgs, 1-2 inputs, and no data dependency are likely elementwise
        if (!slot.flags.isDataDependent && slot.wiring.numInputs <= 2 && slot.wiring.numOutputs == 1) {
          numElementwise++;
        } else {
          numOther++;
        }
      }
    }

    int eliminatedOps = numIdentity + numFusedTails;
    int effectiveOps = segSize - eliminatedOps;

    DSP_DIAG(EMULATED_REPLAY,
             "  fusion analysis: %d total ops, %d effective (-%d identity, -%d fused tails)",
             segSize, effectiveOps, numIdentity, numFusedTails);
    DSP_DIAG(EMULATED_REPLAY,
             "    matmul=%d elementwise=%d view=%d inPlaceFused=%d dataDep=%d controlFlow=%d other=%d",
             numMatmul, numElementwise, numViewOps, numInPlaceFused,
             numDataDependent, numControlFlow, numOther);
    if (numFusedChains > 0) {
      DSP_DIAG(EMULATED_REPLAY,
               "    fused chains: %d chains covering %d ops (avg %.1f ops/chain)",
               numFusedChains, totalFusedChainOps,
               numFusedChains > 0 ? (float)totalFusedChainOps / numFusedChains : 0.0f);
    }

    // Segment pattern classification
    const char* pattern = "MIXED";
    if (numDataDependent > 0)           pattern = "DATA_DEPENDENT (non-capturable)";
    else if (numMatmul > 0 && numElementwise > 0) pattern = "MATMUL_EPILOGUE (best for graph capture)";
    else if (numMatmul > 0)             pattern = "PURE_MATMUL (cuBLAS graph capture)";
    else if (numElementwise == effectiveOps) pattern = "PURE_ELEMENTWISE (best for kernel fusion)";
    else if (numViewOps == segSize)      pattern = "PURE_VIEW (zero compute, identity graph)";

    DSP_DIAG(EMULATED_REPLAY, "    segment pattern: %s", pattern);

    // ── Gap 4: DOT graph topology ────────────────────────────────────────
    if (DSP_DIAG_ENABLED(EMULATED_REPLAY)) {
      DSP_DIAG(EMULATED_REPLAY, "  DOT_BEGIN seg[%d-%d]", seg.def.startSlot, seg.def.endSlot);
      DSP_DIAG(EMULATED_REPLAY, "  digraph segment_%d_%d {", seg.def.startSlot, seg.def.endSlot);
      DSP_DIAG(EMULATED_REPLAY, "    rankdir=TB;");
      DSP_DIAG(EMULATED_REPLAY, "    node [shape=box, fontsize=10];");

      // External input nodes
      std::unordered_set<int> emittedExt;
      for (int s = seg.def.startSlot; s <= seg.def.endSlot; s++) {
        NativeSlot& slot = slots_[s];
        for (int i = 0; i < slot.wiring.numInputs; i++) {
          int srcIdx = slot.wiring.inputSourceIndices[i];
          if (srcIdx < 0) {
            int extIdx = -(srcIdx + 1);
            if (emittedExt.count(extIdx)) continue;
            emittedExt.insert(extIdx);
            const char* srcLabel = "EXT";
            if (slot.wiring.inputSourceTypes[i] == SOURCE_PLACEHOLDER) srcLabel = "PH";
            else if (slot.wiring.inputSourceTypes[i] == SOURCE_CONSTANT) srcLabel = "CONST";
            else if (slot.wiring.inputSourceTypes[i] == SOURCE_VARIABLE) srcLabel = "VAR";
            DSP_DIAG(EMULATED_REPLAY,
                     "    ext_%d [label=\"%s[%d]\", shape=ellipse, style=filled, fillcolor=lightblue];",
                     extIdx, srcLabel, extIdx);
          }
        }
      }

      // Op nodes
      for (int s = seg.def.startSlot; s <= seg.def.endSlot; s++) {
        NativeSlot& slot = slots_[s];
        const char* color = "white";
        if (slot.flags.isIdentityOp)         color = "gray90";
        else if (slot.fusedChain.isFusedChainHead) color = "lightyellow";
        else if (slot.fusedChain.isFusedChainTail) color = "lightyellow";
        else if (slot.flags.isViewCapableOp)  color = "honeydew";
        else if (slot.flags.isDataDependent)  color = "mistyrose";

        DSP_DIAG(EMULATED_REPLAY,
                 "    slot_%d [label=\"[%d] %s\", style=filled, fillcolor=%s];",
                 s, s, slot.ident.opName.empty() ? "?" : slot.ident.opName.c_str(), color);
      }

      // Edges
      for (int s = seg.def.startSlot; s <= seg.def.endSlot; s++) {
        NativeSlot& slot = slots_[s];
        for (int i = 0; i < slot.wiring.numInputs; i++) {
          int srcIdx = slot.wiring.inputSourceIndices[i];
          if (srcIdx < 0) {
            int extIdx = -(srcIdx + 1);
            DSP_DIAG(EMULATED_REPLAY, "    ext_%d -> slot_%d;", extIdx, s);
          } else if (srcIdx >= 0) {
            // Find which slot produces this output
            bool foundProducer = false;
            for (int ps = seg.def.startSlot; ps < s && !foundProducer; ps++) {
              NativeSlot& pslot = slots_[ps];
              for (int o = 0; o < pslot.wiring.numOutputs; o++) {
                if (pslot.wiring.outputSlotIndices[o] == srcIdx) {
                  DSP_DIAG(EMULATED_REPLAY, "    slot_%d -> slot_%d;", ps, s);
                  foundProducer = true;
                  break;
                }
              }
            }
            if (!foundProducer) {
              // Cross-segment input
              DSP_DIAG(EMULATED_REPLAY, "    cross_%d [label=\"slot[%d]\", shape=diamond];", srcIdx, srcIdx);
              DSP_DIAG(EMULATED_REPLAY, "    cross_%d -> slot_%d;", srcIdx, s);
            }
          }
        }
      }

      DSP_DIAG(EMULATED_REPLAY, "  }");
      DSP_DIAG(EMULATED_REPLAY, "  DOT_END seg[%d-%d]", seg.def.startSlot, seg.def.endSlot);
    }

  } else if (!fastPath) {
    // ══════════════════════════════════════════════════════════════════════
    // POST-WARMUP: stability checks (not on fast path)
    // ══════════════════════════════════════════════════════════════════════
    bool shapeStable = (currentShapeKey == seg.exec.cachedShapeKey);
    bool addrStable = (currentAddrKey == seg.exec.capturedInputAddrKey);

    const char* shapeVerdict = shapeStable ? "STABLE" : "CHANGED";
    const char* addrVerdict = addrStable ? "STABLE" : "CHANGED";

    DSP_DIAG(EMULATED_REPLAY,
             "  stability: shape=%s (0x%llx vs cached 0x%llx) addr=%s (0x%llx vs cached 0x%llx)",
             shapeVerdict,
             (long long)currentShapeKey, (long long)seg.exec.cachedShapeKey,
             addrVerdict,
             (long long)currentAddrKey, (long long)seg.exec.capturedInputAddrKey);

    if (!shapeStable) {
      DSP_DIAG(EMULATED_REPLAY,
               "  ** SHAPE KEY CHANGED: CUDA graph would INVALIDATE and re-capture. "
               "Identify which input shapes changed between executions.");

      // Detailed: find which external inputs changed shape
      for (int s = seg.def.startSlot; s <= seg.def.endSlot; s++) {
        NativeSlot& slot = slots_[s];
        for (int i = 0; i < slot.wiring.numInputs; i++) {
          int srcIdx = slot.wiring.inputSourceIndices[i];
          if (srcIdx < 0) {
            int extIdx = -(srcIdx + 1);
            if (extIdx < numExt && externalArrays[extIdx] != nullptr) {
              auto* arr = externalArrays[extIdx];
              DSP_DIAG(EMULATED_REPLAY,
                       "    ext[%d] type=%d shape=[%s] dtype=%d",
                       extIdx, (int)slot.wiring.inputSourceTypes[i],
                       ShapeUtils::shapeAsString(arr).c_str(),
                       (int)arr->dataType());
            }
          }
        }
      }

      seg.exec.cachedShapeKey = currentShapeKey;
    }

    if (!addrStable) {
      DSP_DIAG(EMULATED_REPLAY,
               "  ** ADDRESS KEY CHANGED: capture buffer D2D copies needed. "
               "Placeholders with new addresses require staging buffer updates.");
      seg.exec.capturedInputAddrKey = currentAddrKey;
    }

    // Replay readiness assessment
    if (shapeStable && addrStable) {
      seg.exec.argTableStable = true;  // Enable fast path on next step
      DSP_DIAG(EMULATED_REPLAY,
               "  REPLAY READY: shapes and addresses stable — "
               "CUDA graph replay would succeed without re-capture. (fast path enabled)");
    } else {
      seg.exec.argTableStable = false;  // Disable fast path
      seg.exec.addrKeyStableCount = 0;
      seg.exec.slotAddrStableCount = 0;
      if (shapeStable && !addrStable) {
        DSP_DIAG(EMULATED_REPLAY,
                 "  REPLAY with D2D: shapes stable but addresses changed — "
                 "CUDA graph would replay after capture buffer D2D copies.");
      } else {
        DSP_DIAG(EMULATED_REPLAY,
                 "  RE-CAPTURE needed: shape change requires full graph re-capture.");
      }
    }
  }
  // else: fast path — no key computation, no stability check, just execute

  // ── Execute ops slot-by-slot ────────────────────────────────────────────
  auto tSlotStart = std::chrono::high_resolution_clock::now();

  auto status = executeSegmentSlotBySlot(seg, externalArrays, numExt, stream);

  auto tSlotEnd = std::chrono::high_resolution_clock::now();
  auto slotUs = std::chrono::duration_cast<std::chrono::microseconds>(tSlotEnd - tSlotStart).count();

  // Dispatch overhead estimate: ~15us per effective op (shape inference + dispatch)
  // Identity/fused-tail ops are skipped by executeSlot, so don't count them.
  int estimatedSkippedOps = 0;
  for (int s = seg.def.startSlot; s <= seg.def.endSlot; s++) {
    if (slots_[s].flags.isIdentityOp || slots_[s].fusedChain.isFusedChainTail) estimatedSkippedOps++;
  }
  int effectiveDispatchOps = segSize - estimatedSkippedOps;
  long long estimatedDispatchUs = effectiveDispatchOps * 15LL;

  DSP_DIAG(EMULATED_REPLAY,
           "  execution: %lldus total (%d ops, %d dispatched, %d skipped)%s",
           (long long)slotUs, segSize, effectiveDispatchOps, estimatedSkippedOps,
           fastPath ? " [FAST PATH]" : "");
  DSP_DIAG(EMULATED_REPLAY,
           "  overhead estimate: ~%lldus dispatch + ~%lldus compute = %lldus. "
           "Graph replay would save ~%lldus (%.0f%%)",
           estimatedDispatchUs,
           (long long)slotUs - estimatedDispatchUs,
           (long long)slotUs,
           estimatedDispatchUs,
           slotUs > 0 ? (100.0 * estimatedDispatchUs / slotUs) : 0.0);

  if (status != Status::OK) {
    seg.exec.argTableStable = false;  // Force stability re-check on next step
    seg.exec.addrKeyStableCount = 0;
    seg.exec.slotAddrStableCount = 0;
    DSP_DIAG(EMULATED_REPLAY,
             "  ** EXECUTION FAILED: status=%d — graph capture would also fail here",
             (int)status);
  }

  // Note: executeSegmentSlotBySlot already increments seg.exec.executionCount
  return status;
}

}  // namespace graph
}  // namespace sd

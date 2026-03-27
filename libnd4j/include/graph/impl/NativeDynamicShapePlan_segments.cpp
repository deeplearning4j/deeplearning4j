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
 * Contains computeSegmentShapeKey(), maybeSplitUnstableSegments(),
 * executeSegmentWithCpuGraph(), and executeSegmentSlotBySlot().
 */

#include <graph/NativeDynamicShapePlan.h>
#include <graph/gpu/SymbolicShapeRanges.h>
#include <graph/DspDiagnostics.h>
#include <graph/cpu/FunctionalReplayHandle.h>
#include <helpers/MmulHelper.h>
#include <system/Environment.h>

#include <algorithm>
#include <cstring>
#include <unordered_set>

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

namespace sd {
namespace graph {

namespace {
const char* statusName_seg(Status status) {
  switch (status) {
    case Status::OK: return "OK";
    case Status::BAD_INPUT: return "BAD_INPUT";
    case Status::BAD_SHAPE: return "BAD_SHAPE";
    case Status::BAD_RANK: return "BAD_RANK";
    case Status::BAD_PARAMS: return "BAD_PARAMS";
    case Status::BAD_OUTPUT: return "BAD_OUTPUT";
    case Status::KERNEL_FAILURE: return "KERNEL_FAILURE";
    default: return "UNKNOWN";
  }
}
}  // namespace

// ─── Segment shape key computation ──────────────────────────────────────────

LongType NativeDynamicShapePlan::computeSegmentShapeKey(
    GraphSegment& seg, NDArray** externalInputs, int numExt) {

  // ── Symbolic shape range path ──────────────────────────────────────────
  // When enabled, collect cross-segment inputs, feed them to the shape
  // profiler, and (after warmup) use range-based hashing that ignores
  // dynamic dimensions.
  if (seg.symbolicShapeEnabled && seg.symbolicRangeData != nullptr) {
    auto* profile = static_cast<SegmentShapeProfile*>(seg.symbolicRangeData);

    // Collect cross-segment input arrays (same logic as standard path below)
    std::unordered_set<int> segOutputSlots;
    for (int s = seg.startSlot; s <= seg.endSlot; s++) {
      NativeSlot& slot = slots_[s];
      for (int i = 0; i < slot.numOutputs; i++) {
        segOutputSlots.insert(slot.outputSlotIndices[i]);
      }
    }

    std::vector<NDArray*> crossInputs;
    for (int s = seg.startSlot; s <= seg.endSlot; s++) {
      NativeSlot& slot = slots_[s];
      for (int i = 0; i < slot.numInputs; i++) {
        int srcIdx = slot.inputSourceIndices[i];
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

    // Record observations during warmup
    if (!isWarmupComplete(profile)) {
      recordObservedShapes(profile, crossInputs.data(),
                           static_cast<int>(crossInputs.size()));
      DSP_DIAG(COMPILE, "SymbolicShapes: seg[%d-%d] observation %d/%d",
               seg.startSlot, seg.endSlot,
               getObservationCount(profile), getWarmupSteps(profile));
    }

    // After warmup, use range-based key
    if (isWarmupComplete(profile)) {
      LongType rangeKey = computeRangeBasedShapeKey(
          profile, crossInputs.data(), static_cast<int>(crossInputs.size()),
          seg.startSlot, seg.endSlot);
      DSP_DIAG(COMPILE, "SymbolicShapes: seg[%d-%d] using range-based key=%lld",
               seg.startSlot, seg.endSlot, rangeKey);
      // Cache the key for subsequent calls (when shapesFrozen_ is enabled)
      seg.cachedShapeKey = rangeKey;
      return rangeKey;
    }
    // Fall through to standard path during warmup
  }

  // ── Standard FNV-1a path ───────────────────────────────────────────────
  LongType key = 0xcbf29ce484222325ULL;
  auto mix = [&key](LongType val) {
    key ^= val;
    key *= 0x100000001b3ULL;
  };

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

  mix(seg.startSlot);
  mix(seg.endSlot);

  std::unordered_set<int> segOutputSlots;
  for (int s = seg.startSlot; s <= seg.endSlot; s++) {
    NativeSlot& slot = slots_[s];
    for (int i = 0; i < slot.numOutputs; i++) {
      segOutputSlots.insert(slot.outputSlotIndices[i]);
    }
  }

  for (int s = seg.startSlot; s <= seg.endSlot; s++) {
    NativeSlot& slot = slots_[s];
    for (int i = 0; i < slot.numInputs; i++) {
      int srcIdx = slot.inputSourceIndices[i];
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

  return key;
}

// ─── Adaptive segment splitting ─────────────────────────────────────────────

void NativeDynamicShapePlan::maybeSplitUnstableSegments() {
  bool anySplit = false;
  for (auto& seg : segments_) {
    if (seg.needsSplit) { anySplit = true; break; }
  }
  if (!anySplit) return;

  std::vector<GraphSegment> result;
  result.reserve(segments_.size() + 4);

  for (auto& seg : segments_) {
    if (!seg.needsSplit) {
      result.push_back(std::move(seg));
      continue;
    }

    int segSize = seg.endSlot - seg.startSlot + 1;
    if (segSize <= GraphSegment::MIN_SPLIT_SIZE) {
      seg.needsSplit = false;
      seg.captureFailed = true;
      seg.consecutiveShapeChanges = 0;
      result.push_back(std::move(seg));
      continue;
    }

    {
      int mid = seg.startSlot + segSize / 2;

      auto makeSubSeg = [&](int start, int end) {
        if (start > end) return;
        GraphSegment sub;
        sub.startSlot = start;
        sub.endSlot = end;
        sub.isCapturable = seg.isCapturable;
        sub.executionCount = 0;
        sub.consecutiveShapeChanges = 0;
        sub.needsSplit = false;
        sub.slotArrayCache = slotArrayCache_;
#ifdef SD_CUDA
        sub.cachedShapeKey = 0;
#endif
        for (int s = start; s <= end; s++) {
          slots_[s].shapeCacheValid = false;
          slots_[s].cachedShapeKey = 0;
          slots_[s].cachedOutputShapes.clear();
          slots_[s].frozenContextReady = false;
          slots_[s].frozenConstantSlot = false;
        }
        result.push_back(std::move(sub));
      };

      makeSubSeg(seg.startSlot, mid - 1);
      makeSubSeg(mid, seg.endSlot);

      DSP_DIAG(SEGMENT, "binary-splitting unstable segment [%d-%d] (%d ops) "
                "at midpoint %d into 2 sub-segments",
                seg.startSlot, seg.endSlot, segSize, mid);
    }
  }

  segments_ = std::move(result);
}

// ─── CPU Graph backend selection ────────────────────────────────────────────

GraphBackend* NativeDynamicShapePlan::getCpuGraphBackend() {
  if (cpuGraphBackendChecked_) return cpuGraphBackend_;
  cpuGraphBackendChecked_ = true;
  const auto mode = graphExecutionMode_;

  if (mode == GraphExecutionMode::GEM_SLOT_BY_SLOT ||
      mode == GraphExecutionMode::GEM_TRITON ||
      mode == GraphExecutionMode::GEM_NVRTC_JIT ||
      mode == GraphExecutionMode::GEM_PTX_JIT ||
      mode == GraphExecutionMode::GEM_HIP_GRAPHS ||
      mode == GraphExecutionMode::GEM_LEVELZERO ||
      mode == GraphExecutionMode::GEM_VULKAN ||
      mode == GraphExecutionMode::GEM_METAL) {
    cpuGraphBackend_ = nullptr;
    return nullptr;
  }

  const bool autoLikeMode = (mode == GraphExecutionMode::GEM_AUTO ||
                             mode == GraphExecutionMode::GEM_CUDA_GRAPHS);

#if HAVE_MLX
  if (mode == GraphExecutionMode::GEM_MLX || autoLikeMode) {
    auto& mlx = MlxGraphBackend::getInstance();
    if (mlx.isAvailable()) {
      cpuGraphBackend_ = &mlx;
      if (mode == GraphExecutionMode::GEM_MLX) {
        DSP_DIAG(BACKEND, "using MLX Apple Silicon backend (forced)");
      } else {
        DSP_DIAG(BACKEND, "using MLX Apple Silicon backend");
      }
      return cpuGraphBackend_;
    }
    if (mode == GraphExecutionMode::GEM_MLX) {
      DSP_DIAG(BACKEND, "GEM_MLX requested but MLX not available");
      cpuGraphBackend_ = nullptr;
      return nullptr;
    }
  }
#else
  if (mode == GraphExecutionMode::GEM_MLX) {
    DSP_DIAG(BACKEND, "GEM_MLX requested but HAVE_MLX=0");
    cpuGraphBackend_ = nullptr;
    return nullptr;
  }
#endif

#if HAVE_ONEDNN
  if (autoLikeMode) {
    auto& onednn = OneDnnGraphBackend::getInstance();
    if (onednn.isAvailable()) {
      cpuGraphBackend_ = &onednn;
      DSP_DIAG(BACKEND, "using oneDNN Graph backend");
      return cpuGraphBackend_;
    }
  }
#endif

#if HAVE_ARMCOMPUTE
  if (autoLikeMode) {
    auto& acl = AclGraphBackend::getInstance();
    if (acl.isAvailable()) {
      cpuGraphBackend_ = &acl;
      DSP_DIAG(BACKEND, "using ARM ACL backend");
      return cpuGraphBackend_;
    }
  }
#endif

#if HAVE_NNAPI
  if (mode == GraphExecutionMode::GEM_NNAPI || autoLikeMode) {
    auto& nnapi = NnapiGraphBackend::getInstance();
    if (nnapi.isAvailable()) {
      cpuGraphBackend_ = &nnapi;
      if (mode == GraphExecutionMode::GEM_NNAPI) {
        DSP_DIAG(BACKEND, "using Android NNAPI backend (forced)");
      } else {
        DSP_DIAG(BACKEND, "using Android NNAPI backend");
      }
      return cpuGraphBackend_;
    }
    if (mode == GraphExecutionMode::GEM_NNAPI) {
      DSP_DIAG(BACKEND, "GEM_NNAPI requested but NNAPI not available");
      cpuGraphBackend_ = nullptr;
      return nullptr;
    }
  }
#else
  if (mode == GraphExecutionMode::GEM_NNAPI) {
    DSP_DIAG(BACKEND, "GEM_NNAPI requested but HAVE_NNAPI=0");
    cpuGraphBackend_ = nullptr;
    return nullptr;
  }
#endif

#if HAVE_MLIR
#if defined(__ANDROID__) || (defined(__linux__) && defined(__aarch64__))
  if (mode == GraphExecutionMode::GEM_ARM_HYBRID || autoLikeMode) {
    auto& armHybrid = ArmHybridGraphBackend::getInstance();
    if (armHybrid.isAvailable()) {
      cpuGraphBackend_ = &armHybrid;
      if (mode == GraphExecutionMode::GEM_ARM_HYBRID) {
        DSP_DIAG(BACKEND, "using ARM Hybrid (MLIR CPU + Vulkan) backend (forced)");
      } else {
        DSP_DIAG(BACKEND, "using ARM Hybrid (MLIR CPU + Vulkan) backend");
      }
      return cpuGraphBackend_;
    }
    if (mode == GraphExecutionMode::GEM_ARM_HYBRID) {
      DSP_DIAG(BACKEND, "GEM_ARM_HYBRID requested but backend not available");
      cpuGraphBackend_ = nullptr;
      return nullptr;
    }
  }
#else
  if (mode == GraphExecutionMode::GEM_ARM_HYBRID) {
    DSP_DIAG(BACKEND, "GEM_ARM_HYBRID requested but this platform is not ARM Android/Linux");
    cpuGraphBackend_ = nullptr;
    return nullptr;
  }
#endif

  if (autoLikeMode) {
    auto& mlirBackend = MlirCpuGraphBackend::getInstance();
    if (mlirBackend.isAvailable()) {
      cpuGraphBackend_ = &mlirBackend;
      DSP_DIAG(BACKEND, "using MLIR CPU JIT backend");
      return cpuGraphBackend_;
    }
  }
#else
  if (mode == GraphExecutionMode::GEM_ARM_HYBRID) {
    DSP_DIAG(BACKEND, "GEM_ARM_HYBRID requested but HAVE_MLIR=0");
    cpuGraphBackend_ = nullptr;
    return nullptr;
  }
#endif

  cpuGraphBackend_ = nullptr;
  return nullptr;
}

// ─── Segment execution: CPU graph backend ───────────────────────────────────

Status NativeDynamicShapePlan::executeSegmentWithCpuGraph(
    GraphSegment& seg, NDArray** externalArrays, int numExt, void* stream) {

  auto* backend = getCpuGraphBackend();
  if (backend == nullptr) {
    DSP_DIAG_SEG(BACKEND, seg.startSlot,
                 "executeSegmentWithCpuGraph: no CPU graph backend available for seg[%d-%d]",
                 seg.startSlot, seg.endSlot);
    return Status::KERNEL_FAILURE;
  }
  const char* backendName = backend->name();

  if (seg.captureFailed) {
    DSP_DIAG_SEG(FALLBACK, seg.startSlot,
                 "executeSegmentWithCpuGraph: seg[%d-%d] skipped (captureFailed=true, backend=%s)",
                 seg.startSlot, seg.endSlot, backendName);
    return Status::KERNEL_FAILURE;
  }

  if (!backend->canFuseSegment(slots_, seg.startSlot, seg.endSlot)) {
    DSP_DIAG(BACKEND, "executeSegmentWithCpuGraph: backend=%s cannot fuse seg[%d-%d]",
              backendName, seg.startSlot, seg.endSlot);
    return Status::KERNEL_FAILURE;
  }

  if (seg.executionCount == 0) {
    return executeSegmentSlotBySlot(seg, externalArrays, numExt, stream);
  }

  LongType segShapeKey = computeSegmentShapeKey(seg, externalArrays, numExt);

  bool needsCompile = (seg.executionCount == 1) || (seg.shapeKey != segShapeKey);
  if (needsCompile) {
    DSP_DIAG_SEG(COMPILE, seg.startSlot,
                 "seg[%d-%d] needs compile: %s (execCount=%d shapeKey=%lld->%lld backend=%s)",
                 seg.startSlot, seg.endSlot,
                 seg.executionCount == 1 ? "first-compile" : "shape-key-changed",
                 seg.executionCount,
                 static_cast<long long>(seg.shapeKey),
                 static_cast<long long>(segShapeKey),
                 backendName);
  } else {
    DSP_DIAG_SEG(COMPILE, seg.startSlot,
                 "seg[%d-%d] shape cache HIT (shapeKey=%lld execCount=%d backend=%s)",
                 seg.startSlot, seg.endSlot,
                 static_cast<long long>(segShapeKey),
                 seg.executionCount, backendName);
  }
  if (needsCompile) {
    // Restore outputSlots_ from slotArrayCache_ for the compilation range.
    // When shapes aren't frozen, outputSlots_ was zeroed at the start of execute().
    // The compiler needs access to warmup arrays for shape resolution.
    if (slotArrayCache_ != nullptr) {
      for (int si = seg.startSlot; si <= seg.endSlot && si < totalOutputSlots_; si++) {
        if (outputSlots_[si] == nullptr && slotArrayCache_[si] != nullptr) {
          auto* db = slotArrayCache_[si]->dataBuffer();
          if (db != nullptr && db->isValid()) {
            outputSlots_[si] = slotArrayCache_[si];
          }
        }
      }
    }
    if (!backend->compileSegment(seg, slots_, externalArrays, numExt,
                                 outputSlots_, totalOutputSlots_, segShapeKey,
                                 numSlots_)) {
      DSP_DIAG(COMPILE, "executeSegmentWithCpuGraph: backend=%s compile failed for seg[%d-%d]",
                backendName, seg.startSlot, seg.endSlot);
      return Status::KERNEL_FAILURE;
    }
  }

  if (seg.executionCount == 1) {
    auto audit = backend->getLastCompilationAudit();
    lastCompilationAudit_ = audit;
    bool allCompiled = true;
    for (const auto& entry : audit) {
      if (!entry.wasCompiled) {
        allCompiled = false;
        DSP_DIAG(COMPILE, "%s VALIDATION: slot %d (%s) was NOT compiled: %s",
                  backendName, entry.slotIndex, entry.opName.c_str(), entry.reason.c_str());
      }
    }
    if (!allCompiled) {
      DSP_DIAG(FALLBACK, "%s VALIDATION FAILURE: segment [%d-%d] has ops not covered by backend. "
                "Falling back to slot-by-slot.",
                backendName, seg.startSlot, seg.endSlot);
      seg.captureFailed = true;
      return Status::KERNEL_FAILURE;
    } else {
      DSP_DIAG_SEG(COMPILE, seg.startSlot,
                   "%s VALIDATION OK: seg[%d-%d] all %d ops compiled successfully",
                   backendName, seg.startSlot, seg.endSlot, (int)audit.size());
    }
  }

  seg.shapeKey = segShapeKey;
  tl_graphExecutionActive = true;
  auto status = backend->executeSegment(seg, slots_, externalArrays, numExt,
                                         outputSlots_, totalOutputSlots_, stream);
  tl_graphExecutionActive = false;

  DSP_DIAG(EXECUTE, "executeSegmentWithCpuGraph: exec%d seg[%d-%d]: backend=%s status=%d(%s)",
            seg.executionCount, seg.startSlot, seg.endSlot, backendName,
            static_cast<int>(status), statusName_seg(status));

  if (status == Status::OK) {
    seg.executionCount++;
    totalGraphReplays_++;
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
  if (inputIdx < 0 || inputIdx >= slot.numInputs) return nullptr;
  int srcIdx = slot.inputSourceIndices[inputIdx];
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
  for (int i = 0; i < slot.numInputs; i++) {
    int srcIdx = slot.inputSourceIndices[i];
    if (srcIdx >= 0 && srcIdx < slotIsDeadSize && slotIsDead[srcIdx]) {
      return true;
    }
  }
  return false;
}

// Mark all outputs of a slot as dead
inline void markOutputsDead(NativeSlot& slot, bool* slotIsDead, int slotIsDeadSize) {
  for (int i = 0; i < slot.numOutputs; i++) {
    int si = slot.outputSlotIndices[i];
    if (si >= 0 && si < slotIsDeadSize) slotIsDead[si] = true;
  }
}

// Forward input[0] to all outputs (identity operation for Enter/Exit/LoopCond/NextIteration)
inline void forwardInput(NativeSlot& slot, NDArray** outputSlots, int totalOutputSlots,
                         NDArray** externalInputs, int numExt) {
  NDArray* input = resolveCfInput(slot, 0, outputSlots, totalOutputSlots, externalInputs, numExt);
  for (int i = 0; i < slot.numOutputs; i++) {
    int si = slot.outputSlotIndices[i];
    if (si >= 0 && si < totalOutputSlots) {
      outputSlots[si] = input;
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
                (long long)out->lengthOf(), out->specialBuffer());
    }
  }
}
#endif

}  // namespace

Status NativeDynamicShapePlan::executeSegmentSlotBySlot(
    GraphSegment& seg, NDArray** externalArrays, int numExt, void* stream) {
  DSP_DIAG_SEG(EXECUTE, seg.startSlot,
               "executeSegmentSlotBySlot: ENTER seg[%d-%d] size=%d execCount=%d capturable=%d captureFailed=%d",
               seg.startSlot, seg.endSlot, seg.endSlot - seg.startSlot + 1,
               seg.executionCount, seg.isCapturable ? 1 : 0, seg.captureFailed ? 1 : 0);
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

  int stepIdx = seg.startSlot;
  int loopIterations = 0;

  while (stepIdx <= seg.endSlot) {
    NativeSlot& slot = slots_[stepIdx];

    // ── Control flow dispatch ────────────────────────────────────────
    if (slot.controlFlowType != CF_NONE) {
      // Dead propagation: if all inputs are dead and this is not a Merge, propagate dead
      if (slot.controlFlowType != CF_MERGE && hasControlFlow_ && slotIsDead_ != nullptr) {
        if (anyInputDead(slot, slotIsDead_, slotIsDeadSize_)) {
          DSP_DIAG_SLOT(EXECUTE, stepIdx,
                        "slot %d (%s) DEAD: propagated from dead input (cf=%d)",
                        stepIdx, slot.opName.c_str(), (int)slot.controlFlowType);
          markOutputsDead(slot, slotIsDead_, slotIsDeadSize_);
          stepIdx++;
          continue;
        }
      }

      switch (slot.controlFlowType) {
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
          for (int i = 0; i < slot.numOutputs; i++) {
            int si = slot.outputSlotIndices[i];
            if (si >= 0 && si < totalOutputSlots_) {
              if (i == liveIdx) {
                outputSlots_[si] = data;
                if (slotIsDead_) slotIsDead_[si] = false;
              } else {
                outputSlots_[si] = nullptr;
                if (slotIsDead_) slotIsDead_[si] = true;
              }
            }
          }
#ifdef SD_CUDA
          verifyCfSlotWrite(stepIdx, "SWITCH", slot.opName.c_str(),
                            outputSlots_, slot.outputSlotIndices, slot.numOutputs, totalOutputSlots_);
#endif
          break;
        }

        case CF_MERGE: {
          // Merge: select first non-dead, non-null input
          NDArray* selected = nullptr;
          for (int i = 0; i < slot.numInputs; i++) {
            int srcIdx = slot.inputSourceIndices[i];
            bool isDead = (srcIdx >= 0 && srcIdx < slotIsDeadSize_ && slotIsDead_ && slotIsDead_[srcIdx]);
            if (!isDead) {
              NDArray* inp = resolveCfInput(slot, i, outputSlots_, totalOutputSlots_, externalArrays, numExt);
              if (inp != nullptr) {
                selected = inp;
                break;
              }
            }
          }
          for (int i = 0; i < slot.numOutputs; i++) {
            int si = slot.outputSlotIndices[i];
            if (si >= 0 && si < totalOutputSlots_) {
              outputSlots_[si] = selected;
              if (slotIsDead_) slotIsDead_[si] = (selected == nullptr);
            }
          }
#ifdef SD_CUDA
          verifyCfSlotWrite(stepIdx, "MERGE", slot.opName.c_str(),
                            outputSlots_, slot.outputSlotIndices, slot.numOutputs, totalOutputSlots_);
#endif
          break;
        }

        case CF_ENTER:
        case CF_EXIT:
        case CF_LOOP_COND:
          // Identity: forward input[0] to output[0]
          forwardInput(slot, outputSlots_, totalOutputSlots_, externalArrays, numExt);
#ifdef SD_CUDA
          {
            const char* cfName = (slot.controlFlowType == CF_ENTER) ? "ENTER" :
                                  (slot.controlFlowType == CF_EXIT) ? "EXIT" : "LOOP_COND";
            verifyCfSlotWrite(stepIdx, cfName, slot.opName.c_str(),
                              outputSlots_, slot.outputSlotIndices, slot.numOutputs, totalOutputSlots_);
          }
#endif
          break;

        case CF_NEXT_ITERATION: {
          // Forward input[0] to output[0], then jump back to Merge
          forwardInput(slot, outputSlots_, totalOutputSlots_, externalArrays, numExt);
#ifdef SD_CUDA
          verifyCfSlotWrite(stepIdx, "NEXT_ITER", slot.opName.c_str(),
                            outputSlots_, slot.outputSlotIndices, slot.numOutputs, totalOutputSlots_);
#endif

          if (slot.loopBackTarget >= 0 && slot.loopBackTarget >= seg.startSlot) {
            loopIterations++;
            if (loopIterations >= MAX_LOOP_ITERATIONS) {
              DSP_DIAG(EXECUTE, "loop iteration limit (%d) reached at slot %d",
                        MAX_LOOP_ITERATIONS, stepIdx);
              return Status::KERNEL_FAILURE;
            }
            // Clear dead flags for loop body range
            if (slotIsDead_ && slot.loopRegionIndex >= 0 && slot.loopRegionIndex < numLoopRegions_) {
              LoopRegion& lr = loopRegions_[slot.loopRegionIndex];
              for (int s = lr.mergeSlot; s <= lr.bodyEndSlot && s < numSlots_; s++) {
                NativeSlot& bodySlot = slots_[s];
                for (int oi = 0; oi < bodySlot.numOutputs; oi++) {
                  int si = bodySlot.outputSlotIndices[oi];
                  if (si >= 0 && si < slotIsDeadSize_) slotIsDead_[si] = false;
                }
              }
            }
            stepIdx = slot.loopBackTarget;
            continue; // jump back to Merge, don't increment stepIdx
          }
          break;
        }

        default:
          break;
      }

      // Release schedule for CF slots
      if (stepIdx < numSlots_) {
        int releaseCount = releaseAtStepCounts_[stepIdx];
        if (releaseCount > 0) {
          for (int r = 0; r < releaseCount; r++) {
            int slotIdx = releaseAtStep_[stepIdx][r];
            outputSlots_[slotIdx] = nullptr;
          }
        }
      }

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
            // Process release schedule for the trigger slot
            int releaseCount = releaseAtStepCounts_[stepIdx];
            if (releaseCount > 0) {
              for (int r = 0; r < releaseCount; r++) {
                int slotIdx2 = releaseAtStep_[stepIdx][r];
                outputSlots_[slotIdx2] = nullptr;
              }
            }
            stepIdx++;
            continue;
          }
          // On failure, fall through to individual execution of this slot
          DSP_DIAG(FALLBACK, "batched GEMM group %d failed (status=%d), falling back to individual execution",
                    bgIdx, (int)batchStatus);
        } else {
          // Non-first member: output already computed by the trigger's batch call.
          // Skip execution but still process the release schedule.
          int releaseCount = releaseAtStepCounts_[stepIdx];
          if (releaseCount > 0) {
            for (int r = 0; r < releaseCount; r++) {
              int slotIdx2 = releaseAtStep_[stepIdx][r];
              outputSlots_[slotIdx2] = nullptr;
            }
          }
          stepIdx++;
          continue;
        }
      }
    }
#endif

    // ── Normal op execution ──────────────────────────────────────────
    Status status;
    bool retriedAfterTrim = false;
executeSlot_retry:
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
        DSP_DIAG_SLOT(MEMORY, stepIdx, "slot %d (%s) OOM, flushing pending frees and retrying...",
                  stepIdx, slots_[stepIdx].opName.c_str());
        cudaGetLastError();
        if (stream) {
          cudaStream_t execStr = *static_cast<cudaStream_t*>(stream);
          cudaStreamSynchronize(execStr);
        }
        flushPendingClose(stream);
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
        goto executeSlot_retry;
      }
#endif
      char buf[512];
      snprintf(buf, sizeof(buf), "slot %d (%s) threw exception: %s",
               stepIdx, slots_[stepIdx].opName.c_str(), e.what());
      DSP_DIAG(FALLBACK, "%s", buf);
      sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
      sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(buf);
      status = Status::KERNEL_FAILURE;
    } catch (...) {
      char buf[512];
      snprintf(buf, sizeof(buf), "slot %d (%s) threw unknown exception",
               stepIdx, slots_[stepIdx].opName.c_str());
      DSP_DIAG(FALLBACK, "%s", buf);
      sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(1);
      sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(buf);
      status = Status::KERNEL_FAILURE;
    }
    if (status != Status::OK) {
      char buf[512];
      snprintf(buf, sizeof(buf), "slot %d (%s) failed with status %d",
               stepIdx, slots_[stepIdx].opName.c_str(), static_cast<int>(status));
      DSP_DIAG(FALLBACK, "%s", buf);

      auto& failedSlot = slots_[stepIdx];
      for (int i = 0; i < failedSlot.numInputs; i++) {
        int srcIdx = failedSlot.inputSourceIndices[i];
        if (srcIdx >= 0) {
          NDArray* inp = (srcIdx < totalOutputSlots_ ? outputSlots_[srcIdx] : nullptr);
          if (inp != nullptr) {
            DSP_DIAG(FALLBACK, "  input[%d] from outputSlot[%d]: rank=%lld",
                      i, srcIdx, (long long)inp->rankOf());
          } else {
            DSP_DIAG(FALLBACK, "  input[%d] from outputSlot[%d]: null", i, srcIdx);
          }
        } else {
          DSP_DIAG(FALLBACK, "  input[%d] from external[%d]", i, -(srcIdx + 1));
        }
      }

      sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(static_cast<int>(status));
      sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(buf);

#ifdef SD_CUDA
      cudaGetLastError();
#endif
      return status;
    }

    // Record op for FunctionalReplayHandle capture
    if (seg.replayHandle && seg.replayHandle->getState() == ReplayState::CAPTURING) {
      auto* funcHandle = dynamic_cast<FunctionalReplayHandle*>(seg.replayHandle.get());
      if (funcHandle) funcHandle->recordOp(slot.op, stepIdx);
    }

    int releaseCount = releaseAtStepCounts_[stepIdx];
    if (releaseCount > 0) {
      for (int r = 0; r < releaseCount; r++) {
        int slotIdx = releaseAtStep_[stepIdx][r];
        outputSlots_[slotIdx] = nullptr;
      }
    }

    if (!streamIsCapturing &&
        ((stepIdx % 100 == 99) || pendingCloseBytes_ > 256ULL * 1024 * 1024)) {
      flushPendingClose(stream);
    }

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

  seg.executionCount++;
  return Status::OK;
}

}  // namespace graph
}  // namespace sd

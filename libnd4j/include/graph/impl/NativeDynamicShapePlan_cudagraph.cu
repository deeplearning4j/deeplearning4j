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
 * NativeDynamicShapePlan — CUDA Graph Capture/Replay
 *
 * Contains executeSegmentWithGraph() (CUDA graph warmup/capture/replay
 * state machine), computeSegmentInputAddrKey() (GPU address hashing for
 * graph invalidation), and executeSegmentWithJit() (NVRTC JIT compilation).
 *
 * This file is compiled as .cu (CUDA source). All code is CUDA-only.
 */

#ifdef SD_CUDA

#include <graph/NativeDynamicShapePlan.h>
#include <graph/NativePlanCompiler.h>
#include <graph/CaptureStateGuard.h>
#include <graph/DspDiagnostics.h>
#include <graph/DspHashUtils.h>
#include <graph/DspVerifyUtils.h>
#include <graph/DspSegmentLifecycle.h>
#include <ops/OpTraitTable.h>
#include <graph/cuda/CudaGraphReplayHandle.h>
#include <helpers/ConstantShapeHelper.h>
#include <helpers/MmulHelper.h>
#include <memory/cuda/CudaMemoryPool.h>
#include <helpers/AttentionWorkspace.h>
#include <graph/gpu/NvrtcKernelBuilder.h>
#include <graph/gpu/NvrtcKernelCache.h>
#if HAVE_TRITON
#include <graph/gpu/TritonGraphBackend.h>
#endif

#include <system/Environment.h>
#include <system/env_functions.h>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <unordered_set>

namespace sd {

// Thread-local cuBLAS deterministic mode flag — gates workspace usage during capture.
// Defined in DataBuffer.cu inside namespace sd.
extern SD_TLS_EXPORT thread_local bool tl_cublasLtDisabled;

namespace graph {

// Cross-stream sync is now handled by performPreReplaySync() using
// PlanExecutionContext::crossStreamEvent. The TLS event that was here
// was a divergent copy that bypassed the PlanExecutionContext dedup flags.

// Default capture host workspace size (32MB, configurable via ND4J_DSP_CAPTURE_HOST_WORKSPACE_MB)
static size_t CAPTURE_HOST_WORKSPACE_SIZE = []() -> size_t {
  size_t mb = static_cast<size_t>(sd::env_dspCaptureHostWorkspaceMb());
  return mb * 1024ULL * 1024ULL;
}();

// ─── Slot output address fingerprinting ─────────────────────────────────────
// FNV-1a hash of slot output specialBuffer() addresses for a segment.
// Verified before replay — mismatch means output buffers were reallocated
// and the CUDA graph has stale baked-in addresses (would SIGSEGV or corrupt).
static LongType computeSlotAddrHash(NDArray** outputSlots, int startSlot, int endSlot, int totalSlots) {
  return dsp::computeSlotAddrHash(outputSlots, startSlot, endSlot, totalSlots,
      [](NDArray* a) -> void* { return a->specialBuffer(); });
}

// ─── Segment input address key computation ──────────────────────────────────

LongType NativeDynamicShapePlan::computeSegmentInputAddrKey(
    GraphSegment& seg, NDArray** externalInputs, int numExt) {
  uint64_t key = dsp::FNV1A64_OFFSET_BASIS;
  auto mix = [&key](LongType val) {
    dsp::fnv1aMixValue(key, static_cast<uint64_t>(val));
  };

  // Use a flat bitvector instead of unordered_set for O(1) lookup.
  // Thread-local to avoid per-call allocation in the hot replay path.
  // IMPORTANT: Must clear the ENTIRE vector — not just the current segment range —
  // because this TLS vector persists across plan instances. A prior plan's segment
  // may have set entries in different ranges that contaminate the address key hash.
  static thread_local std::vector<bool> isSegOutput;
  if (static_cast<int>(isSegOutput.size()) < totalOutputSlots_) {
    isSegOutput.resize(totalOutputSlots_, false);
  }
  std::fill(isSegOutput.begin(), isSegOutput.end(), false);
  for (int s = seg.def.startSlot; s <= seg.def.endSlot; s++) {
    NativeSlot& slot = slots_[s];
    for (int i = 0; i < slot.wiring.numOutputs; i++) {
      int oi = slot.wiring.outputSlotIndices[i];
      if (oi >= 0 && oi < totalOutputSlots_) isSegOutput[oi] = true;
    }
  }

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
        if (externalInputIsVariable_[extIdx]) continue;
        NDArray* extArr = externalInputs[extIdx];
        if (extArr == nullptr) continue;
        mix(reinterpret_cast<LongType>(extArr->specialBuffer()));
      } else if (srcIdx < totalOutputSlots_ && !isSegOutput[srcIdx]) {
        if (outputSlots_[srcIdx] != nullptr) {
          mix(reinterpret_cast<LongType>(outputSlots_[srcIdx]->specialBuffer()));
        }
      }
    }
  }

  return key;
}

// ─── Create (ConstantOfShape) op value key ──────────────────────────────────
// Hashes the input DATA values of all 'create' ops in a segment, PLUS the data
// values of all VARIABLE external inputs (those marked isVariable).
//
// Create ops have value-dependent output shapes: their single input is a shape
// tensor whose *values* determine the output dimensions.  If these values change
// between capture and replay, the baked-in CUDA memset produces wrong-sized output.
//
// Variable external inputs (e.g., ConstantOfShape outputs computed by Java SameDiff)
// may also contain data that changes between steps.  Gap ops within the captured
// graph read from these external addresses — if the data changes but the graph
// isn't re-captured, replay produces stale results.  Hashing their data values
// detects these changes and forces re-capture.
//
// Returns 0 only if the segment has no create ops AND no variable external inputs.

namespace {
uint32_t resolveCreateValueKeyTraits(const NativeSlot& slot) {
  uint32_t traits = 0;
  if (slot.ident.op != nullptr && slot.ident.op->getOpDescriptor() != nullptr) {
    traits |= slot.ident.op->getOpDescriptor()->getTraits();
  }
  // Fallback: look up traits by op name from the trait table.
  if (traits == 0 && !slot.ident.opName.empty()) {
    traits |= sd::ops::getOpTraitsByName(slot.ident.opName);
  }
  if (slot.flags.outputShapeDependsOnInputValues) traits |= sd::ops::OP_TRAIT_VALUE_DEPENDENT_SHAPE;
  if (slot.flags.isDataDependent) traits |= sd::ops::OP_TRAIT_DATA_DEPENDENT;
  if (slot.flags.isViewCapableOp) traits |= sd::ops::OP_TRAIT_VIEW_PRODUCING;
  if (slot.flags.isIdentityOp) traits |= sd::ops::OP_TRAIT_IDENTITY;
  return traits;
}

bool slotUsesValueTrackedConstantGeneration(const NativeSlot& slot) {
  const uint32_t traits = resolveCreateValueKeyTraits(slot);
  return (traits & sd::ops::OP_TRAIT_CONSTANT_GENERATION) != 0 &&
         (traits & sd::ops::OP_TRAIT_VALUE_DEPENDENT_SHAPE) != 0;
}
}  // namespace

LongType NativeDynamicShapePlan::computeCreateOpValueKey(
    GraphSegment& seg, NDArray** externalInputs, int numExt) {
  uint64_t key = 0;
  auto mix = [&key](LongType val) {
    if (key == 0) key = dsp::FNV1A64_OFFSET_BASIS;
    dsp::fnv1aMixValue(key, static_cast<uint64_t>(val));
  };

  // Part 1: Hash create op inputs (original logic)
  for (int s = seg.def.startSlot; s <= seg.def.endSlot; s++) {
    NativeSlot& slot = slots_[s];
    if (!slotUsesValueTrackedConstantGeneration(slot)) continue;

    // Track the inputs of any value-tracked constant-generation op.
    // This keeps replay invalidation trait-driven instead of relying on op names.
    for (int i = 0; i < slot.wiring.numInputs; i++) {
      int srcIdx = slot.wiring.inputSourceIndices[i];
      NDArray* inputArr = nullptr;
      if (srcIdx < 0) {
        int extIdx = -(srcIdx + 1);
        if (extIdx < numExt) inputArr = externalInputs[extIdx];
      } else if (srcIdx >= 0 && srcIdx < totalOutputSlots_) {
        inputArr = outputSlots_[srcIdx];
      }
      if (inputArr == nullptr || inputArr->lengthOf() == 0) continue;

      // Read values from device (shape tensors are small, typically 4-5 elements)
      int n = (int)inputArr->lengthOf();
      if (n > 16) n = 16;  // Cap for safety
      int elemSize = DataTypeUtils::sizeOf(inputArr->dataType());
      std::vector<uint8_t> buf(n * elemSize);
      if (inputArr->specialBuffer()) {
        cudaMemcpy(buf.data(), inputArr->specialBuffer(), n * elemSize, cudaMemcpyDeviceToHost);
      } else if (inputArr->buffer()) {
        std::memcpy(buf.data(), inputArr->buffer(), n * elemSize);
      }
      // Hash each element as LongType
      for (int j = 0; j < n; j++) {
        LongType val = 0;
        if (elemSize == 8) {
          std::memcpy(&val, buf.data() + j * 8, 8);
        } else if (elemSize == 4) {
          int32_t v32; std::memcpy(&v32, buf.data() + j * 4, 4);
          val = (LongType)v32;
        }
        mix(val);
      }
    }
  }


  return key;
}

// ─── External address snapshot/compare ─────────────────────────────────────

void NativeDynamicShapePlan::snapshotExternalAddrs(
    GraphSegment& seg, NDArray** externalInputs, int numExt) {
  if (seg.exec.replayHandle) {
    seg.exec.replayHandle->snapshotExternalAddresses(externalInputs, numExt);
  }
}

bool NativeDynamicShapePlan::externalAddrsMatch(
    const GraphSegment& seg, NDArray** externalInputs, int numExt) const {
  if (!seg.exec.replayHandle) return false;
  return seg.exec.replayHandle->externalAddressesMatch(externalInputs, numExt);
}


Status NativeDynamicShapePlan::executeSegmentWithGraph(
    GraphSegment& seg, NDArray** externalArrays, int numExt, void* stream) {

  int segIdx = 0;
  for (size_t i = 0; i < segments_.size(); ++i) {
    if (&segments_[i] == &seg) { segIdx = static_cast<int>(i); break; }
  }

  // ── Segment shape key: detect whether recompilation/recapture is needed ──
  // Frozen + cached key: reuse (shapes can't change). Otherwise: compute once and cache.
  // computeSegmentShapeKey is expensive (per-element D2H sync) so only call when needed.
  LongType segShapeKey;
  if (!planLifecycle_.isSlotBySlot() && seg.exec.cachedShapeKey != 0) {
    segShapeKey = seg.exec.cachedShapeKey;
  } else {
    segShapeKey = computeSegmentShapeKey(seg, externalArrays, numExt);
    if (!planLifecycle_.isSlotBySlot()) {
      seg.exec.cachedShapeKey = segShapeKey;
    }
  }

  {
    bool hasGraph = (seg.exec.replayHandle != nullptr);
    bool shapeMatch = hasGraph && (seg.exec.cachedShapeKey == segShapeKey);
    DSP_DIAG_SEG(EXECUTE, segIdx, "seg[%d-%d] execCount=%d hasGraph=%d shapeMatch=%d compilationFailed=%d",
                 seg.def.startSlot, seg.def.endSlot, executeCount_,
                 static_cast<int>(hasGraph), static_cast<int>(shapeMatch),
                 static_cast<int>(seg.exec.compilationFailed));
  }

  auto invalidateSegmentShapeState = [&](GraphSegment& segRef) {
    for (int stepIdx = segRef.def.startSlot; stepIdx <= segRef.def.endSlot; stepIdx++) {
      auto& slot = slots_[stepIdx];
      slot.slotPhase.reset();  // PRIMARY
      slot.state_ = NativeSlot::SlotState::WARMUP;  // legacy sync
      slot.shapeCache.cachedShapeKey = 0;
      slot.shapeCache.cachedOutputShapes.clear();
    }
  };

  auto clearGraphStreamError = [&](cudaStream_t cudaStrm) {
    cudaGetLastError();
    if (cudaStrm != nullptr) {
      cudaStreamSynchronize(cudaStrm);
      cudaGetLastError();
    }
  };

  // ── REPLAY: cached graph with matching shapes ──
  if (seg.exec.replayHandle && seg.exec.cachedShapeKey == segShapeKey &&
      seg.exec.replayHandle->isReady()) {

    cudaStream_t cudaStr = (stream != nullptr)
        ? *static_cast<cudaStream_t*>(stream) : nullptr;
    bool replayInputsStable = true;
    if (seg.exec.capturedInputAddrKey != 0) {
      replayInputsStable =
          computeSegmentInputAddrKey(seg, externalArrays, numExt) == seg.exec.capturedInputAddrKey;
    } else if (!seg.exec.replayHandle->getCapturedExternalAddresses().empty()) {
      replayInputsStable = externalAddrsMatch(seg, externalArrays, numExt);
    }

    if (replayInputsStable) {
      // Unified pre-replay sync: cross-stream + H2D + staging D2D.
      // Replaces three separate inline blocks. Idempotent via PlanExecutionContext.
      externalArrays = performPreReplaySync(externalArrays, numExt, stream, "cudagraph_replay");

#if HAVE_TRITON
      // Gate arg table refresh on generation counter (replaces argTableStable boolean)
      if (seg.exec.needsArgRefresh()) {
        auto* tritonBackend = dynamic_cast<TritonGraphBackend*>(getGpuGraphBackend());
        if (tritonBackend != nullptr) {
          tritonBackend->refreshArgTablesForReplay(seg, externalArrays, numExt,
                                                   outputSlots_, totalOutputSlots_,
                                                   stream);
          tritonBackend->copyConsolidatedArgTableToDevice(seg, stream);
        }
        seg.exec.markArgsCurrent();
      } else {
        DSP_DIAG(EXECUTE, "CUDAGRAPH_REPLAY: args current — skip refresh seg[%d-%d]",
                 seg.def.startSlot, seg.def.endSlot);
      }
#endif

      // Slot address drift check: graph bakes slot output device pointers.
      // If any slot was reallocated, the graph has stale addresses and must rebuild.
      if (seg.exec.capturedSlotAddrHash != 0) {
        LongType currentAddrHash = computeSlotAddrHash(
            outputSlots_, seg.def.startSlot, seg.def.endSlot, totalOutputSlots_);
        if (currentAddrHash != seg.exec.capturedSlotAddrHash) {
          DSP_DIAG(MEMORY, "SLOT ADDRESS DRIFT for seg[%d-%d]: "
                   "captured=0x%llx current=0x%llx — invalidating replay handle",
                   seg.def.startSlot, seg.def.endSlot,
                   (long long)seg.exec.capturedSlotAddrHash, (long long)currentAddrHash);
          clearGraphStreamError(cudaStr);
          platformCleanupSegmentForRebuild(seg);
          replayInputsStable = false;
        }
      }

      // Zero slot outputs that require it before each graph replay (see
      // platformTryFrozenFastPath comment for the full rationale).
      prezeroSegmentOutputs(seg, stream);

      // Zero cuBLAS workspace before graph replay to match live cuBLAS behavior.
      if (cublasWorkspaceBuffer_ != nullptr && cublasWorkspaceSize_ > 0) {
        cudaMemsetAsync(cublasWorkspaceBuffer_, 0, cublasWorkspaceSize_, cudaStr);
      }

      if (replayInputsStable && seg.exec.replayHandle->replay(stream)) {
        seg.exec.lastReplayExecCount = executeCount_;
        totalGraphReplays_++;
        seg.exec.executionCount++;
        return Status::OK;
      }

      if (replayInputsStable) {
        clearGraphStreamError(cudaStr);
        platformCleanupSegmentForRebuild(seg);
        DSP_THROW_SEG(EXECUTE, seg.def.startSlot,
                      "CUDA graph replay failed for seg[%d-%d]",
                      seg.def.startSlot, seg.def.endSlot);
      }
    } else {
      clearGraphStreamError(cudaStr);
      platformCleanupSegmentForRebuild(seg);
      DSP_THROW_SEG(EXECUTE, seg.def.startSlot,
                    "CUDA graph replay invalidated for seg[%d-%d]: input addresses drifted since capture",
                    seg.def.startSlot, seg.def.endSlot);
    }
  }

  // ── WARM-UP ──
  // Capture contract: require captureMinWarmups (2) slot-by-slot executions before
  // CUDA graph capture is allowed. This ensures:
  //   1. All output NDArrays are allocated and shape-cached
  //   2. All intermediate device buffers have valid content
  //   3. cuBLAS algorithm selection is stable (same algo warmup→capture→replay)
  //   4. No pre-capture hacks needed (hasValueDependentShapeOps save/restore eliminated)
  //
  // This matches the TRITON path which uses Environment::tritonCaptureMinExec() = 2.
  // Using the same threshold for ALL capture modes eliminates mode-specific special cases.
  static constexpr int CAPTURE_MIN_WARMUPS = 2;

  bool shapeChanged = (seg.exec.cachedShapeKey != segShapeKey);

  if (seg.exec.executionCount < CAPTURE_MIN_WARMUPS || (shapeChanged && !seg.exec.compilationFailed)) {
    if (shapeChanged && seg.exec.replayHandle) {
      platformCleanupSegmentForRebuild(seg);
    }
    seg.exec.cachedShapeKey = segShapeKey;
    if (shapeChanged) inShapeChangeWarmup_ = true;
    auto warmupResult = executeSegmentSlotBySlot(seg, externalArrays, numExt, stream);
    if (shapeChanged) inShapeChangeWarmup_ = false;
    if (warmupResult == Status::OK) {
      seg.exec.executionCount++;
    }
    return warmupResult;
  }

  // Lifecycle: warmup complete → advance segPhase to CAPTURING.
  // CUDA_GRAPHS mode skips the compile step (no JIT), so we go directly
  // from WARMUP → CAPTURING via skipCompileToCapturing(). The post-capture
  // markWarmupDone→markCompiled→markCaptured sequence (below, ~line 1166)
  // handles the legacy lifecycleState field transitions via SegmentLifecycle functions.
  if (seg.exec.segPhase.needsWarmup()) {
    seg.exec.segPhase.skipCompileToCapturing();
  }

  // ── CAPTURE ──
  // PRE-CAPTURE LIFECYCLE ASSERTION: segment must be in BUILDING:CAPTURING sub-phase.
  // If it's not, something skipped a lifecycle step — fail loudly instead of silently
  // capturing with incorrect state.
  if (seg.exec.segPhase.isSealed()) {
    DSP_THROW_SEG(EXECUTE, seg.def.startSlot,
                  "LIFECYCLE VIOLATION: attempting CUDA graph capture on SEALED segment "
                  "seg[%d-%d]. Sealed segments must only REPLAY, not re-capture. "
                  "Call invalidateSegmentCaptures() before re-capture.",
                  seg.def.startSlot, seg.def.endSlot);
  }
  if (seg.exec.segPhase.isFailed()) {
    DSP_THROW_SEG(EXECUTE, seg.def.startSlot,
                  "LIFECYCLE VIOLATION: attempting CUDA graph capture on FAILED segment "
                  "seg[%d-%d]. Failed segments must slot-by-slot only.",
                  seg.def.startSlot, seg.def.endSlot);
  }

  // Phase enforcement: CUDA graph capture during REPLAYING means a phase management bug.
  // The plan should have been demoted before re-capture is needed.
  if (planLifecycle_.isReplaying()) {
    DSP_DIAG(COMPILE,
             "ERROR: CUDA graph capture triggered during REPLAYING phase for seg[%d-%d] "
             "(executionCount=%d, phase=%s). Capture must only happen during "
             "warmup/compile/capture phases. Demoting plan phase.",
             seg.def.startSlot, seg.def.endSlot, seg.exec.executionCount,
             planLifecycle_.displayName());
    REQUIRE_TRUE(false, 0,
                 "DSP phase contract violation: CUDA graph capture during REPLAYING phase "
                 "for seg[%d-%d]. Fix the phase management bug.",
                 seg.def.startSlot, seg.def.endSlot);
    demotePlanPhase(PlanPhase::SHAPES_FROZEN,
                    "CUDA graph capture triggered during REPLAYING phase");
  }
  // Phase enforcement: log if graph capture starts before pointers are stable.
  // This is informational — capture may still succeed on first try, but unstable
  // pointers mean the captured graph may need re-capture on the next execution.
  if (!planLifecycle_.isReplaying()) {
    DSP_DIAG(SEGMENT, "PHASE_INFO: graph capture starting for seg[%d-%d] at planPhase=%s "
              "(before SEALED). External/input addresses may still drift.",
              seg.def.startSlot, seg.def.endSlot, planLifecycle_.displayName());
  }

  if (seg.exec.replayHandle && seg.exec.cachedShapeKey != segShapeKey) {
    platformCleanupSegmentForRebuild(seg);
  }

  if (seg.exec.captureOomRetries > 0 && seg.exec.executionCount < seg.exec.captureRetryAfterExec) {
    // HARD ERROR: OOM during capture is a bug — fix memory management.
    // Silent fallback to slot-by-slot violates the execution mode contract.
    DSP_THROW_SEG(MEMORY, seg.def.startSlot,
                  "CUDA graph capture OOM for seg[%d-%d] (retry %d/%d, retryAfterExec=%d, "
                  "currentExecCount=%d). Fix memory management — do NOT fall back to slot-by-slot.",
                  seg.def.startSlot, seg.def.endSlot,
                  seg.exec.captureOomRetries, GraphSegment::maxOomRetries(),
                  seg.exec.captureRetryAfterExec, seg.exec.executionCount);
  }

  // NOTE: The hasValueDependentShapeOps pre-capture warmup hack was REMOVED.
  // It existed to work around capturing after only 1 warmup step — it ran the model
  // a second time, saved/restored outputSlots_, decremented executionCount. This
  // created subtle state corruption: device memory contained the extra warmup's
  // results while pointers were restored to pre-warmup state.
  //
  // With CAPTURE_MIN_WARMUPS=2 (above), the segment naturally executes slot-by-slot
  // twice before capture, achieving the same goal without save/restore tricks.
  // The 2nd warmup naturally populates all output buffers with valid content that
  // the capture step can then correctly overwrite.

  cudaStream_t cudaStr = (stream != nullptr)
      ? *static_cast<cudaStream_t*>(stream) : nullptr;

  auto& scheduler = cuda::CudaGraphScheduler::getInstance();

  int currentDevice = 0;
  cudaError_t currentDeviceErr = cudaGetDevice(&currentDevice);
  if (currentDeviceErr != cudaSuccess) {
    DSP_THROW_CUDA(COMPILE, currentDeviceErr,
                   "cudaGetDevice failed during graph capture setup for seg[%d-%d]",
                   seg.def.startSlot, seg.def.endSlot);
  }
  if (!scheduler.deviceSupportsGraphs(currentDevice)) {
    DSP_THROW_SEG(COMPILE, seg.def.startSlot,
                  "device %d does not support CUDA graphs for seg[%d-%d]",
                  currentDevice, seg.def.startSlot, seg.def.endSlot);
  }

  // ── PRE-CAPTURE MEMORY CHECK ──
  bool isOomRetry = (seg.exec.captureOomRetries > 0);
  if (!isOomRetry) {
    size_t estimatedCaptureBytes = 0;
    for (int stepIdx = seg.def.startSlot; stepIdx <= seg.def.endSlot; stepIdx++) {
      NativeSlot& slot = slots_[stepIdx];
      for (int i = 0; i < slot.wiring.numOutputs; i++) {
        int slotIdx = slot.wiring.outputSlotIndices[i];
        if (slotIdx >= 0 && slotIdx < totalOutputSlots_ && outputSlots_[slotIdx] != nullptr) {
          estimatedCaptureBytes += outputSlots_[slotIdx]->lengthOf() *
                                   outputSlots_[slotIdx]->sizeOfT();
        }
      }
    }

    size_t gpuFree = 0, gpuTotal = 0;
    cudaMemGetInfo(&gpuFree, &gpuTotal);

    // CUDA graph capture does NOT duplicate output buffers — it records kernel
    // launches with their existing device pointers. The capture overhead is only
    // the graph structure itself (typically a few MB) plus any temporary allocations
    // that kernels make internally during the capture pass. A 20% safety margin
    // over the working set covers runtime overhead without over-estimating.
    size_t captureOverhead = estimatedCaptureBytes / 5;  // 20% margin
    size_t requiredFree = captureOverhead;
    if (requiredFree > gpuFree) {
      DSP_DIAG_SEG(MEMORY, 0, "insufficient GPU memory for graph capture seg[%d-%d] (%d ops): "
                    "estimated overhead %zuMB (20%% of %zuMB working set) > free %zuMB (total %zuMB) "
                    "— returning KERNEL_FAILURE (memory-budget segmentation should prevent this)",
                    seg.def.startSlot, seg.def.endSlot, seg.def.endSlot - seg.def.startSlot + 1,
                    requiredFree / (1024 * 1024),
                    estimatedCaptureBytes / (1024 * 1024),
                    gpuFree / (1024 * 1024),
                    gpuTotal / (1024 * 1024));
      return Status::KERNEL_FAILURE;
    }
  }

  seg.exec.replayHandle = GraphReplayFactory::create(currentDevice);
  auto* cudaReplay = static_cast<CudaGraphReplayHandle*>(seg.exec.replayHandle.get());
  auto* handle = cudaReplay->getNativeHandle();

  cudaGetLastError();
  if (cudaStr != nullptr) {
    auto syncErr = cudaStreamSynchronize(cudaStr);
    if (syncErr != cudaSuccess) {
      DSP_THROW_CUDA(COMPILE, syncErr,
                     "cudaStreamSynchronize failed before graph capture for seg[%d-%d]",
                     seg.def.startSlot, seg.def.endSlot);
    }
  }

  // In deterministic mode (tl_cublasLtDisabled), setCublasWorkspaceForCapture
  // will clear the workspace, so skip the allocation entirely to save GPU memory.
  if (!tl_cublasLtDisabled) {
    const size_t CUBLAS_WORKSPACE_SIZE = sd::env_dspCublasWorkspaceMb() * 1024ULL * 1024ULL;
    ensureCublasWorkspace(CUBLAS_WORKSPACE_SIZE);
  }
  setCublasWorkspaceForCapture(stream);

  MmulHelper::resetCastCacheIndices();

  std::vector<std::pair<int, NDArray*>> savedExternalInputs;
  std::vector<std::pair<int, NDArray*>> savedOutputSlots;
  std::vector<NDArray*> preCapOutputSlots(outputSlots_, outputSlots_ + totalOutputSlots_);

  std::vector<SlotPhase> savedSlotPhases(seg.def.endSlot - seg.def.startSlot + 1);
  std::vector<NativeSlot::SlotState> savedFrozenContextReady(seg.def.endSlot - seg.def.startSlot + 1);
  for (int s = seg.def.startSlot; s <= seg.def.endSlot; s++) {
    savedSlotPhases[s - seg.def.startSlot] = slots_[s].slotPhase;
    savedFrozenContextReady[s - seg.def.startSlot] = slots_[s].state_;
    if (slots_[s].slotPhase.isSealed() && !slots_[s].slotPhase.isConstant) {
      slots_[s].slotPhase.unseal(); slots_[s].slotPhase.shapeCacheValid = true;  // PRIMARY: demote for capture
      slots_[s].state_ = NativeSlot::SlotState::SHAPE_CACHED;  // legacy sync
    }
  }

  if (cudaStr != nullptr) {
    cudaStreamSynchronize(cudaStr);
  }

  cudaStream_t resolvedCaptureStream = cudaStr;
  if (resolvedCaptureStream == nullptr) {
    auto* defaultStreamPtr = LaunchContext::defaultContext()->getCudaStream();
    if (defaultStreamPtr != nullptr) {
      resolvedCaptureStream = *defaultStreamPtr;
    }
  }

  // Allocate capture workspace — adaptive to available GPU memory.
  // The configured size (default 512MB) is the MAXIMUM. On memory-constrained
  // GPUs (e.g. 24GB with a 14GB model), we scale down to fit. Before allocating,
  // trim the memory pool to reclaim cached-but-unused buffers.
  static size_t MAX_CAPTURE_WORKSPACE = []() -> size_t {
    size_t mb = static_cast<size_t>(sd::env_dspCaptureWorkspaceMb());
    return mb * 1024ULL * 1024ULL;
  }();
  DSP_DIAG_SEG(MEMORY, segIdx, "capture workspace check seg[%d-%d]: ptr=%p bytes=%zu",
               seg.def.startSlot, seg.def.endSlot, seg.exec.replayHandle->getWorkspacePtr(), seg.exec.replayHandle->getWorkspaceBytes());
  if (seg.exec.replayHandle->getWorkspacePtr() == nullptr) {
    int deviceId = 0;
    cudaGetDevice(&deviceId);

    // Trim pool first to reclaim cached memory before workspace allocation.
    // The pool reports reserved memory as "used" to cudaMemGetInfo, but trim
    // releases it back to the driver, making it available for cudaMalloc.
    memory::CudaMemoryPool::getInstance().trimPool(deviceId);

    // Query actual free memory and scale workspace to fit.
    // Reserve at least 256MB headroom for kernel temporaries + cuBLAS workspace.
    size_t gpuFree = 0, gpuTotal = 0;
    cudaMemGetInfo(&gpuFree, &gpuTotal);
    size_t headroom = 256ULL * 1024 * 1024;
    size_t workspaceSize = MAX_CAPTURE_WORKSPACE;
    if (gpuFree > headroom) {
      size_t availableForWs = gpuFree - headroom;
      if (availableForWs < workspaceSize) {
        workspaceSize = availableForWs;
        DSP_DIAG_SEG(MEMORY, segIdx,
                     "capture workspace scaled down: gpuFree=%zuMB headroom=%zuMB → workspace=%zuMB (max=%zuMB)",
                     gpuFree / (1024*1024), headroom / (1024*1024),
                     workspaceSize / (1024*1024), MAX_CAPTURE_WORKSPACE / (1024*1024));
      }
    } else {
      // Barely any free memory — use minimum viable workspace (32MB)
      workspaceSize = 32ULL * 1024 * 1024;
      DSP_DIAG_SEG(MEMORY, segIdx,
                   "capture workspace minimal: gpuFree=%zuMB < headroom=%zuMB → workspace=32MB",
                   gpuFree / (1024*1024), headroom / (1024*1024));
    }

    if (!seg.exec.replayHandle->allocateWorkspace(workspaceSize, deviceId, nullptr, seg.def.startSlot)) {
      DSP_THROW_SEG(COMPILE, seg.def.startSlot,
                    "capture workspace allocation failed for seg[%d-%d]: gpuFree=%zuMB, "
                    "requested=%zuMB. Graph would contain cudaMallocAsync nodes which "
                    "cannot be replayed safely",
                    seg.def.startSlot, seg.def.endSlot,
                    gpuFree / (1024*1024), workspaceSize / (1024*1024));
    }
  }

  // Allocate pinned host workspace for H2D source copies during capture.
  // This eliminates cudaMallocHost calls during capture — all host data for
  // graph H2D memcpy nodes is bump-allocated from this pre-allocated buffer.
  void* captureHostWs = nullptr;
  auto hostWsErr = cudaMallocHost(&captureHostWs, CAPTURE_HOST_WORKSPACE_SIZE);
  if (hostWsErr != cudaSuccess) {
    captureHostWs = nullptr;
    DSP_THROW_CUDA(MEMORY, hostWsErr,
                   "capture host workspace allocation failed (%zu bytes) for seg[%d-%d]",
                   CAPTURE_HOST_WORKSPACE_SIZE, seg.def.startSlot, seg.def.endSlot);
  }

  // RAII guard manages all capture TLS. On scope exit:
  // - restores previous tl_graphCaptureStream
  // - clears workspace/host workspace TLS
  // - frees captured host ptrs if commit() was not called
  CaptureStateGuard captureGuard(resolvedCaptureStream,
      seg.exec.replayHandle->getWorkspacePtr(),
      seg.exec.replayHandle->getWorkspaceBytes(),
      captureHostWs,
      (captureHostWs != nullptr) ? CAPTURE_HOST_WORKSPACE_SIZE : 0);

  if (captureHostWs != nullptr) {
    captureGuard.trackHostPtr(captureHostWs);
  }

  // Ensure stable plan-owned staging buffers for variable (placeholder) inputs
  // BEFORE capture begins. The graph will bake in the staging buffer device pointers,
  // which remain stable for the plan's lifetime. On replay, platformTryFrozenFastPath
  // D2D-copies new token data into these same staging buffers before replay.
  // Without this, capture bakes in Java-side NDArray addresses that may change between
  // decode steps, causing replay to read stale/zero data → "!!!!!" output.
  NDArray** captureExternals = externalArrays;
  if (!planLifecycle_.isSlotBySlot() && !externalInputIsVariable_.empty()) {
    // Sync variable inputs H2D first (they need device data for the capture step)
    for (int ei = 0; ei < numExt; ei++) {
      if (ei >= static_cast<int>(externalInputIsVariable_.size()) ||
          !externalInputIsVariable_[ei]) continue;
      if (externalArrays[ei] == nullptr) continue;
      externalArrays[ei]->syncToDevice();
    }
    NDArray** staged = ensureAndSyncStagingBuffers(externalArrays, numExt, stream);
    if (staged != nullptr) {
      captureExternals = staged;
      DSP_DIAG_SEG(COMPILE, segIdx, "capture using staging buffers for variable inputs "
                   "— graph will bake in stable device pointers");
    }
    // Sync stream so D2D copies into staging are complete before capture begins
    if (cudaStr != nullptr) cudaStreamSynchronize(cudaStr);
  }

  DSP_DIAG_SEG(MEMORY, segIdx, "tl_captureWorkspace=%p size=%zu for capture",
               tl_captureWorkspace, tl_captureWorkspaceSize);

  if (!handle->beginCapture(cudaStr, cudaStreamCaptureModeRelaxed)) {
    restoreCublasWorkspaceAfterCapture(stream);
    clearGraphStreamError(cudaStr);
    platformCleanupSegmentForRebuild(seg);
    for (auto& [extIdx, origPtr] : savedExternalInputs) {
      externalArrays[extIdx] = origPtr;
    }
    for (auto& [slotIdx, origPtr] : savedOutputSlots) {
      outputSlots_[slotIdx] = origPtr;
    }
    invalidateSegmentShapeState(seg);
    std::memcpy(outputSlots_, preCapOutputSlots.data(), sizeof(NDArray*) * totalOutputSlots_);
    for (int s = seg.def.startSlot; s <= seg.def.endSlot; s++) {
      slots_[s].slotPhase = savedSlotPhases[s - seg.def.startSlot];  // PRIMARY restore
      slots_[s].state_ = savedFrozenContextReady[s - seg.def.startSlot];  // legacy sync
    }
    DSP_THROW_SEG(COMPILE, seg.def.startSlot,
                  "CUDA graph capture beginCapture failed for seg[%d-%d]",
                  seg.def.startSlot, seg.def.endSlot);
  }

  bool captureOk = true;
  bool captureOomFailure = false;
  int lastCaptureSlot = seg.def.startSlot;
  lastCaptureAudit_.clear();

  try {
    for (int stepIdx = seg.def.startSlot; stepIdx <= seg.def.endSlot; stepIdx++) {
      lastCaptureSlot = stepIdx;
      {
        cudaStreamCaptureStatus capStatus;
        cudaError_t capErr = cudaStreamGetCaptureInfo(cudaStr, &capStatus, nullptr);
        if (capErr != cudaSuccess || capStatus != cudaStreamCaptureStatusActive) {
          DSP_DIAG_SLOT(COMPILE, stepIdx, "CAPTURE BROKEN before slot %d (%s): "
                       "capErr=%d capStatus=%d",
                        stepIdx, slots_[stepIdx].ident.opName.c_str(),
                       static_cast<int>(capErr), static_cast<int>(capStatus));
          captureOk = false;
          break;
        }
      }

      size_t nodesBefore = handle->getNumNodesDuringCapture(cudaStr);

      auto status = executeSlot(stepIdx, captureExternals, numExt, stream);
      if (status != Status::OK) {
        DSP_DIAG_SLOT(COMPILE, stepIdx, "op execution during capture failed at slot %d", stepIdx);
        captureOk = false;
        captureOomFailure = true;
        break;
      }

      {
        cudaStreamCaptureStatus capStatus;
        cudaError_t capErr = cudaStreamGetCaptureInfo(cudaStr, &capStatus, nullptr);
        if (capErr != cudaSuccess || capStatus != cudaStreamCaptureStatusActive) {
          DSP_DIAG_SLOT(COMPILE, stepIdx, "CAPTURE INVALIDATED by slot %d (%s)! "
                       "capErr=%d capStatus=%d",
                        stepIdx, slots_[stepIdx].ident.opName.c_str(),
                       static_cast<int>(capErr), static_cast<int>(capStatus));
          captureOk = false;
          break;
        }
      }

      size_t nodesAfter = handle->getNumNodesDuringCapture(cudaStr);
      {
        cuda::CaptureAuditEntry entry;
        entry.slotIndex = stepIdx;
          entry.opName = slots_[stepIdx].ident.opName;
        entry.nodesBefore = nodesBefore;
        entry.nodesAfter = nodesAfter;
        entry.nodesContributed = (nodesAfter > nodesBefore) ? (nodesAfter - nodesBefore) : 0;
        lastCaptureAudit_.push_back(std::move(entry));
      }

    }
  } catch (const std::exception& e) {
    // Guard destructor handles TLS cleanup + host ptr freeing on rethrow.
    tl_graphExecutionActive = false;  // deactivate before endCapture
    handle->endCapture(cudaStr);
    clearGraphStreamError(cudaStr);
    restoreCublasWorkspaceAfterCapture(stream);
    for (auto& [extIdx, origPtr] : savedExternalInputs) {
      externalArrays[extIdx] = origPtr;
    }
    for (auto& [slotIdx, origPtr] : savedOutputSlots) {
      outputSlots_[slotIdx] = origPtr;
    }
    std::memcpy(outputSlots_, preCapOutputSlots.data(), sizeof(NDArray*) * totalOutputSlots_);
    for (int s = seg.def.startSlot; s <= seg.def.endSlot; s++) {
      slots_[s].slotPhase = savedSlotPhases[s - seg.def.startSlot];  // PRIMARY restore
      slots_[s].state_ = savedFrozenContextReady[s - seg.def.startSlot];  // legacy sync
    }
    platformCleanupSegmentForRebuild(seg);
    throw;  // rethrow — guard destructor frees host ptrs + restores remaining TLS
  } catch (...) {
    tl_graphExecutionActive = false;
    handle->endCapture(cudaStr);
    clearGraphStreamError(cudaStr);
    restoreCublasWorkspaceAfterCapture(stream);
    for (auto& [extIdx, origPtr] : savedExternalInputs) {
      externalArrays[extIdx] = origPtr;
    }
    for (auto& [slotIdx, origPtr] : savedOutputSlots) {
      outputSlots_[slotIdx] = origPtr;
    }
    std::memcpy(outputSlots_, preCapOutputSlots.data(), sizeof(NDArray*) * totalOutputSlots_);
    for (int s = seg.def.startSlot; s <= seg.def.endSlot; s++) {
      slots_[s].slotPhase = savedSlotPhases[s - seg.def.startSlot];  // PRIMARY restore
      slots_[s].state_ = savedFrozenContextReady[s - seg.def.startSlot];  // legacy sync
    }
    platformCleanupSegmentForRebuild(seg);
    THROW_EXCEPTION("Unknown exception during CUDA graph capture");
  }

  // Capture phase complete — save workspace used before guard clears TLS.
  // The guard destructor will run at function scope exit and restore all
  // capture TLS. We save the offset now while it is still valid.
  size_t captureWorkspaceUsed = captureGuard.workspaceUsed();
  tl_graphExecutionActive = false;  // deactivate before endCapture (guard restores remaining TLS at scope exit)
  restoreCublasWorkspaceAfterCapture(stream);

  for (auto& [extIdx, origPtr] : savedExternalInputs) {
    externalArrays[extIdx] = origPtr;
  }
  for (auto& [slotIdx, origPtr] : savedOutputSlots) {
    outputSlots_[slotIdx] = origPtr;
  }

  if (!captureOk) {
    handle->endCapture(cudaStr);

    cudaGetLastError();

    if (cudaStr != nullptr) {
      cudaStreamSynchronize(cudaStr);
      cudaGetLastError();
    }

    // Guard destructor will free captured host ptrs (commit() not called yet).

    if (captureOomFailure && seg.exec.captureOomRetries < GraphSegment::maxOomRetries()) {
      seg.exec.captureOomRetries++;
      seg.exec.captureRetryAfterExec = seg.exec.executionCount + GraphSegment::retryInterval();
      DSP_DIAG_SEG(MEMORY, segIdx, "graph capture OOM for seg[%d-%d], retry %d/%d after exec %d",
                   seg.def.startSlot, seg.def.endSlot,
                   seg.exec.captureOomRetries, GraphSegment::maxOomRetries(),
                   seg.exec.captureRetryAfterExec);
    } else {
      seg.exec.compilationFailed = true;
    }

    // Arrays persist across capture failures — only slot state needs restoring.

    platformCleanupSegmentForRebuild(seg);

    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(0);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage("");
    std::memcpy(outputSlots_, preCapOutputSlots.data(), sizeof(NDArray*) * totalOutputSlots_);
    for (int s = seg.def.startSlot; s <= seg.def.endSlot; s++) {
      slots_[s].slotPhase = savedSlotPhases[s - seg.def.startSlot];  // PRIMARY restore
      slots_[s].state_ = savedFrozenContextReady[s - seg.def.startSlot];  // legacy sync
    }
    invalidateSegmentShapeState(seg);

    // Return KERNEL_FAILURE — the caller in _cuda.cu propagates this via
    // DSP_THROW_SEG so the error surfaces to the user. With memory-budget
    // segment splitting, this should not happen for well-sized segments.
    DSP_DIAG_SEG(COMPILE, 0, "CUDA graph capture failed for seg[%d-%d] (oom=%s, retries=%d) "
                 "— returning KERNEL_FAILURE to caller",
                 seg.def.startSlot, seg.def.endSlot,
                 captureOomFailure ? "true" : "false",
                 seg.exec.captureOomRetries);
    return Status::KERNEL_FAILURE;
  }

  // Helper lambda to restore slot state on capture failure.
  auto cleanupCaptureBuffersOnFailure = [&seg, &savedSlotPhases, &savedFrozenContextReady, this]() {
    // Arrays persist across capture failures — only slot state needs restoring.
    for (int s = seg.def.startSlot; s <= seg.def.endSlot; s++) {
      slots_[s].slotPhase = savedSlotPhases[s - seg.def.startSlot];  // PRIMARY restore
      slots_[s].state_ = savedFrozenContextReady[s - seg.def.startSlot];  // legacy sync
    }
  };

  if (!handle->endCapture(cudaStr)) {
    cudaGetLastError();
    cudaStreamSynchronize(cudaStr);
    cudaGetLastError();
    // Guard destructor frees host ptrs (commit() not called).
    cleanupCaptureBuffersOnFailure();
    platformCleanupSegmentForRebuild(seg);
    std::memcpy(outputSlots_, preCapOutputSlots.data(), sizeof(NDArray*) * totalOutputSlots_);
    invalidateSegmentShapeState(seg);
    DSP_DIAG_SEG(COMPILE, 0, "CUDA graph endCapture failed for seg[%d-%d] "
                  "— returning KERNEL_FAILURE to caller",
                  seg.def.startSlot, seg.def.endSlot);
    return Status::KERNEL_FAILURE;
  }

  if (!handle->instantiate()) {
    // ── OOM eviction: free smallest captured graphs to reclaim GPU memory ──
    // instantiate() destroys _graph on failure, so we defer re-capture to the
    // next execution via captureRetryAfterExec.
    int numEvicted = 0;
    if (handle->wasLastInstantiateOom()) {
      DSP_DIAG(MEMORY, "graph instantiate OOM for seg[%d-%d], attempting eviction (up to %d segments)",
               seg.def.startSlot, seg.def.endSlot, GraphSegment::maxOomRetries());

      for (int evictAttempt = 0; evictAttempt < GraphSegment::maxOomRetries(); evictAttempt++) {
        // Find the segment with the smallest captured graph (fewest nodes) to evict.
        // Skip the current segment being instantiated.
        int evictIdx = -1;
        size_t smallestNodes = SIZE_MAX;
        for (size_t si = 0; si < segments_.size(); si++) {
          if (static_cast<int>(si) == segIdx) continue;
          auto& candidate = segments_[si];
          if (!candidate.exec.replayHandle || !candidate.exec.replayHandle->isReady()) continue;
          // Get node count from the CUDA replay handle
          auto* candidateCudaReplay = dynamic_cast<CudaGraphReplayHandle*>(candidate.exec.replayHandle.get());
          size_t nodeCount = candidateCudaReplay ? candidateCudaReplay->getNumNodes() : 0;
          if (nodeCount == 0) nodeCount = 1;  // Treat unknown as minimal
          if (nodeCount < smallestNodes) {
            smallestNodes = nodeCount;
            evictIdx = static_cast<int>(si);
          }
        }

        if (evictIdx < 0) {
          DSP_DIAG(MEMORY, "no more evictable graph segments found for OOM recovery (evicted %d so far)",
                   numEvicted);
          break;
        }

        // Evict the selected segment's graph with full cleanup
        auto& evictSeg = segments_[evictIdx];
        DSP_DIAG(MEMORY, "evicting graph for seg[%d-%d] (%zu nodes) to free memory for seg[%d-%d] (attempt %d/%d)",
                 evictSeg.def.startSlot, evictSeg.def.endSlot, smallestNodes,
                 seg.def.startSlot, seg.def.endSlot, evictAttempt + 1, GraphSegment::maxOomRetries());

        evictSeg.exec.replayHandle->releaseWorkspace(nullptr, evictSeg.def.startSlot);

        // Free pinned host pointers allocated during capture
        evictSeg.exec.replayHandle->freeHostPointers();
        evictSeg.exec.replayHandle->clearExternalAddresses();

        // Destroy the replay handle (frees cudaGraphExec + cudaGraph via
        // CudaGraphHandle::cleanup())
        evictSeg.exec.replayHandle.reset();

        // Reset the evicted segment so it can re-capture on a future execution
        evictSeg.exec.cachedShapeKey = 0;
        evictSeg.exec.capturedInputAddrKey = 0;
        evictSeg.exec.capturedCreateValueKey = 0;
        evictSeg.exec.compilationFailed = false;
        evictSeg.exec.gapOpsCapturedInGraph = false;
        evictSeg.exec.argTableStable = false;
        evictSeg.exec.addrKeyStableCount = 0;
        evictSeg.exec.slotAddrStableCount = 0;
        evictSeg.exec.compiledByBackend.clear();
        // Reset execution count so evicted segment goes through warmup -> capture again
        evictSeg.exec.executionCount = 0;

        numEvicted++;

        // Synchronize to ensure GPU memory is actually freed before trying more
        if (cudaStr != nullptr) {
          cudaStreamSynchronize(cudaStr);
        }
        cudaGetLastError();

        DSP_DIAG(MEMORY, "evicted seg[%d-%d] (%zu nodes), total evicted: %d",
                 evictSeg.def.startSlot, evictSeg.def.endSlot, smallestNodes, numEvicted);
      }
    }

    // The current segment's _graph was destroyed by instantiate() on failure,
    // so we cannot retry instantiation. Fall through to OOM retry or permanent
    // failure.
    cudaGetLastError();

    if (handle->wasLastInstantiateOom() && seg.exec.captureOomRetries < GraphSegment::maxOomRetries()) {
      // Use the OOM retry mechanism: defer re-capture to a future execution.
      // If we evicted segments above, retry on the very next execution (interval=1)
      // since the freed memory should be immediately available. Otherwise use
      // the standard retry interval to wait for memory pressure to decrease.
      seg.exec.captureOomRetries++;
      int retryInterval = (numEvicted > 0) ? 1 : GraphSegment::retryInterval();
      seg.exec.captureRetryAfterExec = seg.exec.executionCount + retryInterval;
      DSP_DIAG(MEMORY, "graph instantiate OOM for seg[%d-%d], will retry %d/%d after exec %d (evicted %d segments)",
               seg.def.startSlot, seg.def.endSlot,
               seg.exec.captureOomRetries, GraphSegment::maxOomRetries(),
               seg.exec.captureRetryAfterExec, numEvicted);
    } else {
      seg.exec.compilationFailed = true;
    }

    // Guard destructor frees host ptrs (commit() not called).
    clearGraphStreamError(cudaStr);
    cleanupCaptureBuffersOnFailure();
    platformCleanupSegmentForRebuild(seg);
    std::memcpy(outputSlots_, preCapOutputSlots.data(), sizeof(NDArray*) * totalOutputSlots_);
    invalidateSegmentShapeState(seg);
    DSP_DIAG_SEG(COMPILE, 0, "CUDA graph instantiate failed for seg[%d-%d] (oom=%s, retries=%d, evicted=%d) "
                  "— returning KERNEL_FAILURE to caller",
                  seg.def.startSlot, seg.def.endSlot,
                  handle->wasLastInstantiateOom() ? "true" : "false",
                  seg.exec.captureOomRetries, numEvicted);
    return Status::KERNEL_FAILURE;
  }

  cudaGetLastError();

  {
    auto stats = handle->getStatistics();
    DSP_DIAG_SEG(COMPILE, segIdx, "graph captured for seg[%d-%d]: "
                 "%zu nodes, %zu edges, %d kernels, %d memcpys, %d memsets, "
                 "%d memAllocs, %d memFrees, %d hostCallbacks, %d events, %d empty",
                 seg.def.startSlot, seg.def.endSlot,
                 handle->getNumNodes(), handle->getNumEdges(),
                 stats.numKernels, stats.numMemcpyH2D, stats.numMemsets,
                 stats.numMemAllocs, stats.numMemFrees,
                 stats.numHostCallbacks, stats.numEvents, stats.numEmpty);
    if (stats.numMemAllocs != stats.numMemFrees) {
      DSP_DIAG_SEG(COMPILE, segIdx, "WARNING: Unbalanced memory nodes: %d allocs vs %d frees. "
                   "This WILL cause graph launch failure",
                   stats.numMemAllocs, stats.numMemFrees);
    }
    // Empty graphs (0 kernels, 0 memcpys, 0 memsets) have no GPU work.
    // Don't replay them — the slot-by-slot warmup already produced the correct
    // output, and replaying an empty graph just wastes launch overhead + causes
    // spurious fingerprint mismatches when slot addresses change.
    if (stats.numKernels == 0 && stats.numMemcpyH2D == 0 && stats.numMemsets == 0) {
      DSP_DIAG_SEG(COMPILE, segIdx, "empty graph for seg[%d-%d] (0 kernels) — skipping replay, "
                   "marking segment as non-capturable",
                   seg.def.startSlot, seg.def.endSlot);
      seg.exec.compilationFailed = true;
      // tl_capturedHostPtrs is already clear at this point (guard cleared it on construction,
      // and the empty graph path produces no host allocations). The guard destructor will
      // iterate an empty vector — no double-free.
      tl_captureReplicateCache.clear();
      platformCleanupSegmentForRebuild(seg);
      seg.exec.executionCount++;
      return Status::OK;
    }
  }

  if (!handle->launchAsync(cudaStr)) {
    cudaGetLastError();
    // Guard destructor frees host ptrs (commit() not called).
    clearGraphStreamError(cudaStr);
    seg.exec.compilationFailed = true;
    cleanupCaptureBuffersOnFailure();
    platformCleanupSegmentForRebuild(seg);
    std::memcpy(outputSlots_, preCapOutputSlots.data(), sizeof(NDArray*) * totalOutputSlots_);
    invalidateSegmentShapeState(seg);
    DSP_THROW_SEG(COMPILE, seg.def.startSlot,
                  "CUDA graph launchAsync failed for seg[%d-%d]",
                  seg.def.startSlot, seg.def.endSlot);
  }

  // Transfer host pointer ownership to the replay handle, then commit the guard
  // so its destructor does NOT free them (they now belong to the replay handle).
  for (auto* ptr : tl_capturedHostPtrs) {
    seg.exec.replayHandle->addCapturedHostPtr(ptr);
  }
  tl_capturedHostPtrs.clear();
  tl_captureReplicateCache.clear();
  captureGuard.commit();  // ownership transferred — guard must NOT free host ptrs

  // replayHandle is already set (created before capture began)
  seg.exec.cachedShapeKey = segShapeKey;
  // Use captureExternals for addr key + snapshot: the graph was captured against
  // staging buffer addresses for variable inputs, and raw addresses for weights.
  // computeSegmentInputAddrKey skips variable inputs, so using captureExternals
  // gives the same result as externalArrays for the weight-only hash.
  seg.exec.capturedInputAddrKey = computeSegmentInputAddrKey(seg, captureExternals, numExt);
  seg.exec.capturedCreateValueKey = computeCreateOpValueKey(seg, captureExternals, numExt);
  seg.exec.capturedSlotAddrHash = computeSlotAddrHash(
      outputSlots_, seg.def.startSlot, seg.def.endSlot, totalOutputSlots_);
  snapshotExternalAddrs(seg, captureExternals, numExt);
  seg.exec.executionCount++;
  totalGraphReplays_++;

  // Mark as captured by raw CUDA graph path (not Triton).
  // This prevents the Triton replay path in NativeDynamicShapePlan_gpubackend.cpp
  // from incorrectly handling this segment — Triton replay has incompatible
  // D2D copy and arg table logic that can corrupt cross-segment data.
  if (seg.exec.compiledByBackend.empty()) {
    seg.exec.compiledByBackend = "CUDA";
  }

  // Clear compilationFailed — the CUDA graph path succeeded even if the Triton path
  // failed earlier. Without this, cleanup treats this segment as non-graph-managed
  // and frees its output/cross-segment slots, causing stale data on replay.
  if (seg.exec.compilationFailed) {
    DSP_DIAG(COMPILE, "clearing compilationFailed for seg[%d-%d] after successful CUDA graph capture",
             seg.def.startSlot, seg.def.endSlot);
    seg.exec.compilationFailed = false;
  }

  // ── LIFECYCLE STATE TRANSITION ──
  // The CUDA graph path goes NEEDS_WARMUP → captured+replaying in one shot
  // (there's no separate compile step — capture IS compilation for CUDA graphs).
  // Transition through all required states so segmentIsFullyReplayingForPlanPhase()
  // recognizes this segment as SEALED (was REPLAYING).
  if (seg.exec.segPhase.needsWarmup()) {
    SegmentLifecycle::markWarmupDone(seg.exec);
    SegmentLifecycle::markCompiled(seg.exec, "CUDA", segShapeKey);
    SegmentLifecycle::markCaptured(seg.exec, seg.exec.capturedInputAddrKey,
                                   seg.exec.capturedCreateValueKey,
                                   seg.exec.capturedSlotAddrHash, "CUDA");
  } else if (seg.exec.segPhase.needsCapture()) {
    // Re-capture after invalidation — already compiled, just needs capture transition
    SegmentLifecycle::markCaptured(seg.exec, seg.exec.capturedInputAddrKey,
                                   seg.exec.capturedCreateValueKey,
                                   seg.exec.capturedSlotAddrHash, "CUDA");
  } else if (seg.exec.segPhase.needsCompile()) {
    SegmentLifecycle::markCompiled(seg.exec, "CUDA", segShapeKey);
    SegmentLifecycle::markCaptured(seg.exec, seg.exec.capturedInputAddrKey,
                                   seg.exec.capturedCreateValueKey,
                                   seg.exec.capturedSlotAddrHash, "CUDA");
  }


  if (seg.exec.captureOomRetries > 0) {
    DSP_DIAG_SEG(MEMORY, segIdx, "graph capture SUCCEEDED on OOM retry %d for seg[%d-%d]",
                 seg.exec.captureOomRetries, seg.def.startSlot, seg.def.endSlot);
    seg.exec.captureOomRetries = 0;
    seg.exec.captureRetryAfterExec = 0;
  }

  for (int s = seg.def.startSlot; s <= seg.def.endSlot; s++) {
    slots_[s].slotPhase = savedSlotPhases[s - seg.def.startSlot];  // PRIMARY restore
    slots_[s].state_ = savedFrozenContextReady[s - seg.def.startSlot];  // legacy sync
  }

  if (executionTimingEnabled_) {
    auto stats = handle->getStatistics();
    double wsUtilPct = seg.exec.replayHandle->getWorkspaceBytes() > 0
        ? (100.0 * captureWorkspaceUsed / seg.exec.replayHandle->getWorkspaceBytes()) : 0.0;
    DSP_DIAG_SEG(TIMING, segIdx, "captured CUDA graph seg[%d-%d] (%zu nodes, %zu edges) "
                 "[%d kern, %d memcpy, %d memset, %d alloc, %d free] ws=%zuKB/%zuKB (%.1f%%)",
                 seg.def.startSlot, seg.def.endSlot,
                 handle->getNumNodes(), handle->getNumEdges(),
                 stats.numKernels, stats.numMemcpyH2D, stats.numMemsets,
                 stats.numMemAllocs, stats.numMemFrees,
                 seg.exec.replayHandle->getWorkspacePtr() ? (captureWorkspaceUsed / 1024) : 0,
                 seg.exec.replayHandle->getWorkspaceBytes() / 1024, wsUtilPct);

    if (!lastCaptureAudit_.empty()) {
      printCaptureAudit();
    }
  }

  return Status::OK;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Replay Verification — reusable by all paths (Triton, CUDA_GRAPHS, etc.)
// ═══════════════════════════════════════════════════════════════════════════════

void NativeDynamicShapePlan::performReplayVerify(
    GraphSegment& seg, NDArray** externalArrays, int numExt,
    void* stream, const char* pathLabel) {
  DSP_DIAG(VERIFY, "performReplayVerify ENTERED path=%s execCount=%d",
           pathLabel, seg.exec.executionCount);
  fflush(stderr);

  // Ensure VERIFY diagnostics are enabled (may have been set after DspDiagnostics construction)
  DspDiagnostics::getInstance().enableCategories(DSP_DIAG_VERIFY);
  DspDiagnostics::getInstance().setLevel(DSP_LEVEL_FULL);

  cudaStream_t cudaStr = stream ? *static_cast<cudaStream_t*>(stream) : nullptr;

  // Find final output slot for argmax
  int finalOutputSlot = -1;
  if (seg.def.endSlot >= 0 && seg.def.endSlot < numSlots_ && slots_[seg.def.endSlot].wiring.numOutputs > 0) {
    finalOutputSlot = slots_[seg.def.endSlot].wiring.outputSlotIndices[0];
  }
  if (finalOutputSlot < 0 || finalOutputSlot >= totalOutputSlots_) {
    finalOutputSlot = seg.def.endSlot;
  }

  // 1. Compute argmax from REPLAY output
  int replayArgmax = -1;
  if (finalOutputSlot >= 0 && finalOutputSlot < totalOutputSlots_ &&
      outputSlots_[finalOutputSlot] != nullptr) {
    auto* replayFinal = outputSlots_[finalOutputSlot];
    if (replayFinal->lengthOf() > 0 && replayFinal->specialBuffer() != nullptr) {
      replayArgmax = dspArgmax(replayFinal->specialBuffer(), replayFinal->dataType(),
                                replayFinal->lengthOf());
      DSP_DIAG(VERIFY, "REPLAY_VERIFY ARGMAX(replay): slot=%d argmax=%d (of %lld elements) path=%s",
                finalOutputSlot, replayArgmax, (long long)replayFinal->lengthOf(), pathLabel);
    }
  }

  // 2. Snapshot all output slots from replay
  struct SlotSnap {
    int slotIdx, stepIdx;
    DataType dtype;
    LongType length;
    void* bufAddr;
    std::vector<uint8_t> data;
  };
  std::vector<SlotSnap> snaps;

  std::unordered_map<int, int> slotToStep;
  for (int s = seg.def.startSlot; s <= seg.def.endSlot && s < numSlots_; s++) {
    for (int oi = 0; oi < slots_[s].wiring.numOutputs; oi++) {
      int si = slots_[s].wiring.outputSlotIndices[oi];
      if (si >= 0 && si < totalOutputSlots_) slotToStep[si] = s;
    }
  }

  for (int si = 0; si < totalOutputSlots_; si++) {
    NDArray* arr = outputSlots_[si];
    if (!arr || arr->lengthOf() <= 0 || !arr->specialBuffer()) continue;
    if (slotToStep.find(si) == slotToStep.end()) continue;
    DataType dt = arr->dataType();
    int elemSize = DataTypeUtils::sizeOf(dt);
    if (elemSize <= 0) continue;
    int snapCount = std::min(static_cast<int>(arr->lengthOf()), 16);
    SlotSnap snap;
    snap.slotIdx = si;
    snap.stepIdx = slotToStep[si];
    snap.dtype = dt;
    snap.length = arr->lengthOf();
    snap.bufAddr = arr->specialBuffer();
    snap.data.resize(snapCount * elemSize);
    cudaMemcpy(snap.data.data(), arr->specialBuffer(), snapCount * elemSize, cudaMemcpyDeviceToHost);
    snaps.push_back(std::move(snap));
  }
  DSP_DIAG(VERIFY, "REPLAY_VERIFY: saved %zu snapshots from replay (%s path)", snaps.size(), pathLabel);

  // 3. Re-execute slot-by-slot for ground truth
  // Save segment state
  int savedSegExecCount = seg.exec.executionCount;
  bool savedCaptureFailed = seg.exec.compilationFailed;
  seg.exec.compilationFailed = true;
  seg.exec.executionCount = 999;

  // Disable releaseAtStep (prevents nullifying outputs before comparison)
  int** savedReleaseAtStep = releaseAtStep_;
  int* savedReleaseAtStepCounts = releaseAtStepCounts_;
  int* zeroedCounts = new int[numSlots_]();
  int** dummyRelease = new int*[numSlots_]();
  releaseAtStep_ = dummyRelease;
  releaseAtStepCounts_ = zeroedCounts;

  // Reset frozenContextReady to force normal execution path.
  // The frozen path only refreshes external/view-producer inputs — it does NOT
  // refresh regular slot-to-slot inputs, so downstream ops would read stale
  // warmup-era data instead of freshly computed outputs.
  std::vector<SlotPhase> savedVerifySlotPhases(seg.def.endSlot - seg.def.startSlot + 1);
  std::vector<NativeSlot::SlotState> savedFrozenCtx(seg.def.endSlot - seg.def.startSlot + 1);
  for (int s = seg.def.startSlot; s <= seg.def.endSlot; s++) {
    savedVerifySlotPhases[s - seg.def.startSlot] = slots_[s].slotPhase;
    savedFrozenCtx[s - seg.def.startSlot] = slots_[s].state_;
    if (slots_[s].slotPhase.isSealed() && !slots_[s].slotPhase.isConstant) {
      slots_[s].slotPhase.unseal();
      slots_[s].slotPhase.shapeCacheValid = true;
      slots_[s].state_ = NativeSlot::SlotState::SHAPE_CACHED;  // legacy sync
    }
  }
  // Set executeCount_ to 0 so shape inference runs fresh
  int savedExecCountGlobal = executeCount_;
  executeCount_ = 0;

  // Dump VARIABLE externals before fresh re-execution
  dspDumpVariableExternals(externalArrays, numExt, externalInputIsVariable_,
                           externalInputNames_, "before-fresh");


  auto freshStatus = executeSegmentSlotBySlot(seg, externalArrays, numExt, stream);

  // Restore all state
  releaseAtStep_ = savedReleaseAtStep;
  releaseAtStepCounts_ = savedReleaseAtStepCounts;
  delete[] zeroedCounts;
  delete[] dummyRelease;
  for (int s = seg.def.startSlot; s <= seg.def.endSlot; s++) {
    slots_[s].slotPhase = savedVerifySlotPhases[s - seg.def.startSlot];  // PRIMARY restore
    slots_[s].state_ = savedFrozenCtx[s - seg.def.startSlot];  // legacy sync
  }
  executeCount_ = savedExecCountGlobal;
  seg.exec.executionCount = savedSegExecCount;
  seg.exec.compilationFailed = savedCaptureFailed;

  if (freshStatus != Status::OK) {
    DSP_DIAG(VERIFY, "REPLAY_VERIFY: slot-by-slot re-execution FAILED (%s path)", pathLabel);
    return;
  }

  cudaStreamSynchronize(cudaStr);

  // 4. Compute argmax from FRESH execution
  int freshArgmax = -1;
  if (finalOutputSlot >= 0 && finalOutputSlot < totalOutputSlots_ &&
      outputSlots_[finalOutputSlot] != nullptr) {
    auto* freshFinal = outputSlots_[finalOutputSlot];
    if (freshFinal->lengthOf() > 0 && freshFinal->specialBuffer() != nullptr) {
      freshArgmax = dspArgmax(freshFinal->specialBuffer(), freshFinal->dataType(),
                               freshFinal->lengthOf());
      DSP_DIAG(VERIFY, "REPLAY_VERIFY ARGMAX(fresh): slot=%d argmax=%d (of %lld elements)",
                finalOutputSlot, freshArgmax, (long long)freshFinal->lengthOf());
    }
  }

  // 5. Compare snapshots vs fresh
  int mismatchCount = 0;
  int firstMismatchSlot = -1;
  float worstMaxDiff = 0.0f;
  for (auto& snap : snaps) {
    NDArray* fresh = outputSlots_[snap.slotIdx];
    if (!fresh || !fresh->specialBuffer()) continue;
    int elemSize = DataTypeUtils::sizeOf(snap.dtype);
    if (elemSize <= 0) continue;
    int compareCount = std::min(static_cast<int>(snap.data.size()) / elemSize,
                                 std::min(static_cast<int>(fresh->lengthOf()), 16));
    std::vector<uint8_t> freshData(compareCount * elemSize);
    cudaMemcpy(freshData.data(), fresh->specialBuffer(), compareCount * elemSize, cudaMemcpyDeviceToHost);
    float maxDiff = dspMaxDiff(snap.data.data(), freshData.data(), snap.dtype, compareCount);
    if (maxDiff > worstMaxDiff) worstMaxDiff = maxDiff;
    if (maxDiff > 1e-3f) {
      mismatchCount++;
      if (firstMismatchSlot < 0) firstMismatchSlot = snap.slotIdx;
       const char* opName = (snap.stepIdx < numSlots_) ? slots_[snap.stepIdx].ident.opName.c_str() : "?";
      int nShow = std::min(compareCount, 4);
      float rv[4]={0}, fv[4]={0};
      dspBytesToFloat(snap.data.data(), snap.dtype, rv, nShow);
      dspBytesToFloat(freshData.data(), snap.dtype, fv, nShow);
      // Build input info
      std::string inputInfo;
      if (snap.stepIdx < numSlots_) {
        auto& slot = slots_[snap.stepIdx];
        for (int ii = 0; ii < slot.wiring.numInputs; ii++) {
          if (ii > 0) inputInfo += " ";
          int srcIdx = slot.wiring.inputSourceIndices[ii];
          if (srcIdx >= 0) {
            inputInfo += "slot#" + std::to_string(srcIdx);
            if (srcIdx < totalOutputSlots_ && outputSlots_[srcIdx])
              inputInfo += "(len=" + std::to_string(outputSlots_[srcIdx]->lengthOf()) + ")";
          } else {
            int extIdx = -(srcIdx+1);
            inputInfo += "ext#" + std::to_string(extIdx);
            if (extIdx < (int)externalInputNames_.size())
              inputInfo += ":\"" + externalInputNames_[extIdx] + "\"";
          }
        }
      }
      DSP_DIAG(VERIFY, "REPLAY_VERIFY MISMATCH slot=%d step=%d op=%s maxDiff=%.6f "
                "replay=[%.4f,%.4f,%.4f,%.4f] fresh=[%.4f,%.4f,%.4f,%.4f] inputs=[%s]",
                snap.slotIdx, snap.stepIdx, opName, maxDiff,
                rv[0], rv[1], rv[2], rv[3], fv[0], fv[1], fv[2], fv[3],
                inputInfo.c_str());
    }
  }

  if (mismatchCount > 0) {
    DSP_DIAG(VERIFY, "REPLAY_VERIFY SUMMARY: %d/%zu slots exceed 1e-3 tolerance "
              "(first mismatch slot=%d, worst maxDiff=%.6g) path=%s execCount=%d",
              mismatchCount, snaps.size(), firstMismatchSlot, worstMaxDiff,
              pathLabel, executeCount_);
    if (replayArgmax == freshArgmax) {
      DSP_DIAG(VERIFY, "REPLAY_VERIFY ARGMAX: MATCH (replay=%d fresh=%d)", replayArgmax, freshArgmax);
    } else {
      DSP_DIAG(VERIFY, "REPLAY_VERIFY ARGMAX: *** MISMATCH *** (replay=%d fresh=%d)", replayArgmax, freshArgmax);
    }
  } else {
    DSP_DIAG(VERIFY, "REPLAY_VERIFY SUMMARY: ALL MATCH (%zu slots, maxDiff=%.6g) path=%s execCount=%d",
              snaps.size(), worstMaxDiff, pathLabel, executeCount_);
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// ensureAndSyncStagingBuffers — plan-owned stable device buffers for
// variable (placeholder) external inputs. Ensures specialBuffer() pointers
// in CUDA graph arg tables remain valid for the plan's lifetime regardless
// of how Java allocates its input arrays.
// ═══════════════════════════════════════════════════════════════════════════
NDArray** NativeDynamicShapePlan::ensureAndSyncStagingBuffers(
    NDArray** externalArrays, int numExt, void* stream) {
  if (planLifecycle_.isSlotBySlot() || externalInputIsVariable_.empty() || numExt <= 0) {
    return externalArrays;
  }

  cudaStream_t cudaStr = (stream != nullptr)
      ? *static_cast<cudaStream_t*>(stream) : nullptr;

  // One-time allocation — fixed size, never resized.
  if (placeholderStagingBuffers_ == nullptr) {
    placeholderStagingBuffers_ = new NDArray*[numExt]();  // zero-initialized
    effectiveExternals_ = new NDArray*[numExt]();
  }

  // Fast path: after first call, only iterate variable input indices
  // instead of all 1000+ entries. Non-variable (weight) pointers are stable.
  if (!cachedVariableExtIndices_.empty()) {
    for (int i : cachedVariableExtIndices_) {
      NDArray* ext = externalArrays[i];
      effectiveExternals_[i] = externalArrays[i];  // default passthrough
      if (ext == nullptr || ext->isEmpty()) continue;

      NDArray* staging = placeholderStagingBuffers_[i];
      if (staging == nullptr) continue;  // KV buffer or uninitialized

      void* dstBuf = staging->specialBuffer();
      void* srcBuf = ext->specialBuffer();
      if (dstBuf != nullptr && srcBuf != nullptr) {
        size_t bytes = static_cast<size_t>(ext->lengthOf()) * ext->sizeOfT();
        if (bytes > 0) {
          cudaMemcpyAsync(dstBuf, srcBuf, bytes, cudaMemcpyDeviceToDevice, cudaStr);
        }
      }

      // Diagnostic: dump first few values of staging buffer after D2D copy
      // for variable inputs around the divergence point.
      // Gate behind isDebug() to avoid perf impact in production.
      if (Environment::getInstance().isDebug() && executeCount_ >= 10 && executeCount_ <= 20) {
        // Sync to ensure D2D is complete before reading
        if (cudaStr != nullptr) cudaStreamSynchronize(cudaStr);
        int numDump = std::min(16, static_cast<int>(ext->lengthOf()));
        if (ext->dataType() == DataType::FLOAT32 && numDump > 0) {
          std::vector<float> vals(numDump);
          cudaMemcpy(vals.data(), dstBuf, numDump * sizeof(float), cudaMemcpyDeviceToHost);
          std::string name = (i < static_cast<int>(externalInputNames_.size()))
                             ? externalInputNames_[i] : "?";
          sd_printf("STAGING_DIAG[exec=%d] ext[%d]='%s' staging first %d vals:",
                    executeCount_, i, name.c_str(), numDump);
          for (int v = 0; v < numDump; v++) {
            sd_printf(" %.4f", vals[v]);
          }
          sd_printf("\n");
          // Also dump source buffer for comparison
          cudaMemcpy(vals.data(), srcBuf, numDump * sizeof(float), cudaMemcpyDeviceToHost);
          sd_printf("STAGING_DIAG[exec=%d] ext[%d]='%s' source  first %d vals:",
                    executeCount_, i, name.c_str(), numDump);
          for (int v = 0; v < numDump; v++) {
            sd_printf(" %.4f", vals[v]);
          }
          sd_printf("\n");
        } else if (ext->dataType() == DataType::INT64 && numDump > 0) {
          std::vector<LongType> vals(numDump);
          cudaMemcpy(vals.data(), dstBuf, numDump * sizeof(LongType), cudaMemcpyDeviceToHost);
          std::string name = (i < static_cast<int>(externalInputNames_.size()))
                             ? externalInputNames_[i] : "?";
          sd_printf("STAGING_DIAG[exec=%d] ext[%d]='%s' staging first %d vals:",
                    executeCount_, i, name.c_str(), numDump);
          for (int v = 0; v < numDump; v++) {
            sd_printf(" %lld", (long long)vals[v]);
          }
          sd_printf("\n");
        } else if (ext->dataType() == DataType::HALF && numDump > 0) {
          // Read as raw bytes, convert from fp16
          std::vector<uint16_t> raw(numDump);
          cudaMemcpy(raw.data(), dstBuf, numDump * sizeof(uint16_t), cudaMemcpyDeviceToHost);
          std::string name = (i < static_cast<int>(externalInputNames_.size()))
                             ? externalInputNames_[i] : "?";
          sd_printf("STAGING_DIAG[exec=%d] ext[%d]='%s' staging(fp16) first %d raw:",
                    executeCount_, i, name.c_str(), numDump);
          for (int v = 0; v < numDump; v++) {
            sd_printf(" 0x%04x", raw[v]);
          }
          sd_printf("\n");
        }
      }

      staging->dataBuffer()->writeSpecial();
      effectiveExternals_[i] = staging;
    }
    return effectiveExternals_;
  }

  for (int i = 0; i < numExt; i++) {
    // Non-variable inputs (model weights) — pass through directly.
    // Their specialBuffer() is already stable (same DataBuffer for plan lifetime).
    if (i >= static_cast<int>(externalInputIsVariable_.size()) ||
        !externalInputIsVariable_[i]) {
      effectiveExternals_[i] = externalArrays[i];
      continue;
    }

    NDArray* ext = externalArrays[i];
    if (ext == nullptr || ext->isEmpty()) {
      effectiveExternals_[i] = ext;
      continue;
    }

    // Skip staging for KV cache buffers. KV caches are written by GPU kernels
    // (KV scatter) during execution — their device buffer is the source of truth,
    // and their pointers are already stable (static buffers owned by the caller).
    // D2D-copying them into staging would capture stale pre-scatter data.
    //
    // Detection: match the external input's device buffer address against the
    // static KV buffers registered via configureKvScatter(). NDArray pointers
    // differ (Java creates fresh wrappers), so compare specialBuffer() addresses.
    if (kvScatterConfigured_ && ext->specialBuffer() != nullptr) {
      bool isKvBuffer = false;
      void* extDevAddr = ext->specialBuffer();
      for (auto& entry : kvScatterEntries_) {
        if (entry.staticBuf != nullptr &&
            entry.staticBuf->specialBuffer() == extDevAddr) {
          isKvBuffer = true;
          break;
        }
      }
      if (isKvBuffer) {
        effectiveExternals_[i] = externalArrays[i];
        continue;
      }
    }

    // Record this as a variable index for the fast path on subsequent calls
    cachedVariableExtIndices_.push_back(i);

    NDArray* staging = placeholderStagingBuffers_[i];

    // Allocate staging buffer on first use. Shapes are frozen — this
    // allocation happens exactly once per variable input. The staging
    // NDArray and its device buffer persist for the plan's lifetime,
    // giving all arg tables a stable specialBuffer() pointer.
    if (staging == nullptr) {
      staging = new NDArray(ext->ordering(), *ext->getShapeAsVector(),
                            ext->dataType(), LaunchContext::defaultContext());
      placeholderStagingBuffers_[i] = staging;
    }

    // D2D copy: external input (already H2D-synced) → plan-owned staging buffer.
    // Async on the execution stream — no CPU sync needed.
    void* dstBuf = staging->specialBuffer();
    void* srcBuf = ext->specialBuffer();
    if (dstBuf != nullptr && srcBuf != nullptr) {
      size_t bytes = static_cast<size_t>(ext->lengthOf()) * ext->sizeOfT();
      if (bytes > 0) {
        cudaMemcpyAsync(dstBuf, srcBuf, bytes, cudaMemcpyDeviceToDevice, cudaStr);
      }
    }
    staging->dataBuffer()->writeSpecial();

    effectiveExternals_[i] = staging;
  }

  return effectiveExternals_;
}

}  // namespace graph
}  // namespace sd

#endif  // SD_CUDA

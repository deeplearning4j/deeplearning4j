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
#include <graph/PlanExecutionContext.h>
#include <graph/NativePlanCompiler.h>
#include <graph/ModeContract.h>
#include <graph/CaptureStateGuard.h>
#include <graph/DspDiagnostics.h>
#include <graph/DspHashUtils.h>
#include <graph/DspThreadState.h>
#include <graph/DspVerifyUtils.h>
#include <graph/DspSegmentLifecycle.h>
#include <graph/DspSegmentHelpers.h>
#include <helpers/DebugHelper.h>
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
#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

namespace sd {

namespace graph {

// Per-GPU capture/execution coordination (defined in _cuda.cu, namespace sd::graph)
extern std::atomic<bool> g_captureActive[16];
extern std::mutex g_captureMtx[16];
extern std::condition_variable g_captureCV[16];
extern std::atomic<int> g_execCount[16];

struct ScopedDspGapStream {
  cudaStream_t prev;
  explicit ScopedDspGapStream(cudaStream_t stream) : prev(::tl_dspGapStream) {
    ::tl_dspGapStream = stream;
  }
  ~ScopedDspGapStream() {
    ::tl_dspGapStream = prev;
  }
};

// Cross-stream sync is now handled by performPreReplaySync() using
// PlanExecutionContext::crossStreamEvent. The TLS event that was here
// was a divergent copy that bypassed the PlanExecutionContext dedup flags.

// Default capture host workspace size (32MB, configurable via ND4J_DSP_CAPTURE_HOST_WORKSPACE_MB)
static size_t CAPTURE_HOST_WORKSPACE_SIZE = []() -> size_t {
  size_t mb = static_cast<size_t>(sd::env_dspCaptureHostWorkspaceMb());
  return mb * 1024ULL * 1024ULL;
}();

// ─── Sync-free buffer fingerprint ring — kernel + API ────────────────────────
//
// Runs entirely on the capture/replay stream; no host sync.
// One uint64 per (step, trackIdx): XOR of all 64-bit words in the buffer.
// Two-pass: all threads XOR into a per-warp partial (shared mem), then
// atomicXOR into the ring slot.  Works on any buffer size including odd ones.
//
__global__ void bufXorFingerprintKernel(const uint64_t* __restrict__ src,
                                        int numWords,
                                        uint64_t* __restrict__ dst) {
  __shared__ uint64_t smem[32];  // one per warp (max 1024 threads / 32)
  int tid  = threadIdx.x;
  int nWarp = (blockDim.x + 31) >> 5;
  if (tid < nWarp) smem[tid] = 0;
  __syncthreads();

  uint64_t acc = 0;
  for (int i = blockIdx.x * blockDim.x + tid; i < numWords;
       i += gridDim.x * blockDim.x) {
    acc ^= src[i];
  }
  // warp reduce
  for (int mask = 16; mask > 0; mask >>= 1)
    acc ^= __shfl_xor_sync(0xFFFFFFFFu, acc, mask);
  // one thread per warp writes to smem
  if ((tid & 31) == 0)
    smem[tid >> 5] ^= acc;
  __syncthreads();
  // single thread reduces smem → dst via atomicXOR
  if (tid == 0) {
    uint64_t total = 0;
    for (int w = 0; w < nWarp; w++) total ^= smem[w];
    atomicXor(reinterpret_cast<unsigned long long*>(dst), (unsigned long long)total);
  }
}

// ── NativeDynamicShapePlan fingerprint ring implementation ──────────────────

void NativeDynamicShapePlan::initFingerprintRing() {
  if (d_fpRing_ != nullptr) return;
  size_t ringBytes = sizeof(uint64_t) * BUF_FP_MAX_STEPS * BUF_FP_MAX_TRACKED;
  cudaError_t e = cudaMalloc(&d_fpRing_, ringBytes);
  if (e != cudaSuccess) { d_fpRing_ = nullptr; fpRingEnabled_ = false; return; }
  cudaMemset(d_fpRing_, 0, ringBytes);
  h_fpRing_ = new uint64_t[BUF_FP_MAX_STEPS * BUF_FP_MAX_TRACKED]();
}

void NativeDynamicShapePlan::maybeInitFingerprintRing() {
  if (fpRingEnabled_) return;
  const char* env = std::getenv("BUF_FP_RING");
  if (env == nullptr || env[0] != '1') return;
  fpRingEnabled_ = true;
  initFingerprintRing();
}

void NativeDynamicShapePlan::recordBufFingerprintPublic(cudaStream_t stream, int step,
                                                        int trackIdx,
                                                        const void* devPtr, size_t numBytes) {
  if (!fpRingEnabled_ || d_fpRing_ == nullptr || devPtr == nullptr) return;
  if (trackIdx < 0 || trackIdx >= BUF_FP_MAX_TRACKED) return;
  if (numBytes == 0) return;
  // Skip fingerprinting during CUDA graph capture — the kernel+memset would be
  // baked into the graph and would corrupt the ring on every replay.
  // We only want steady-state values (step >= 4), never capture-time values.
  if (DebugHelper::inGraphCapture(stream != nullptr ? &stream : nullptr)) return;
  // Only track steady-state steps (4+); warmup/capture steps don't show the freeze.
  if (step < 4) return;
  int clampedStep = (step >= BUF_FP_MAX_STEPS ? BUF_FP_MAX_STEPS - 1 : step);
  // Zero the ring slot first (atomicXOR accumulates; we need a fresh start per step)
  uint64_t* ringSlot = d_fpRing_ + clampedStep * BUF_FP_MAX_TRACKED + trackIdx;
  cudaMemsetAsync(ringSlot, 0, sizeof(uint64_t), stream);
  int numWords = static_cast<int>((numBytes + 7) / 8);
  int threads  = 256;
  int blocks   = std::min(256, (numWords + threads - 1) / threads);
  bufXorFingerprintKernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const uint64_t*>(devPtr), numWords, ringSlot);
}

// ── Platform dispatch: CUDA implementations ──────────────────────────────────

void NativeDynamicShapePlan::drainFingerprintRingPublic() {
  if (!fpRingEnabled_ || d_fpRing_ == nullptr || fpRingDrained_) return;
  cudaDeviceSynchronize();
  size_t ringBytes = sizeof(uint64_t) * BUF_FP_MAX_STEPS * BUF_FP_MAX_TRACKED;
  cudaMemcpy(h_fpRing_, d_fpRing_, ringBytes, cudaMemcpyDeviceToHost);
  fpRingDrained_ = true;
}

const char* NativeDynamicShapePlan::getFingerprintJson() {
  if (!fpRingEnabled_ || !fpRingDrained_ || h_fpRing_ == nullptr) return "null";
  // Collect all active track indices: staging [0..fpRingStagingCount_-1]
  // + gemm [BUF_FP_MAX_STAGING..BUF_FP_MAX_STAGING+fpRingGemmCount_*2-1].
  // Build two sorted lists so the JSON is ordered and trackIdx is explicit.
  std::vector<int> activeTrackIdxs;
  for (int t = 0; t < fpRingStagingCount_; t++) activeTrackIdxs.push_back(t);
  for (int t = BUF_FP_MAX_STAGING; t < BUF_FP_MAX_STAGING + fpRingGemmCount_; t++)
    activeTrackIdxs.push_back(t);

  fpJsonBuffer_.clear();
  fpJsonBuffer_ += "{\"labels\":[";
  bool firstLabel = true;
  for (int t : activeTrackIdxs) {
    const auto& lb = fpLabels_[t];
    if (lb.tag[0] == '\0') continue;
    char tmp[128];
    snprintf(tmp, sizeof(tmp),
             "{\"trackIdx\":%d,\"tag\":\"%s\",\"extIdx\":%d,\"groupIdx\":%d,\"whichAB\":%d}",
             t, lb.tag, lb.extIdx, lb.groupIdx, lb.whichAB);
    if (!firstLabel) fpJsonBuffer_ += ',';
    firstLabel = false;
    fpJsonBuffer_ += tmp;
  }
  fpJsonBuffer_ += "],\"steps\":[";
  bool firstStep = true;
  for (int s = 0; s < BUF_FP_MAX_STEPS; s++) {
    bool hasData = false;
    for (int t : activeTrackIdxs) {
      if (h_fpRing_[s * BUF_FP_MAX_TRACKED + t] != 0) { hasData = true; break; }
    }
    if (!hasData) continue;
    if (!firstStep) fpJsonBuffer_ += ',';
    firstStep = false;
    char stmp[32];
    snprintf(stmp, sizeof(stmp), "{\"step\":%d,\"fps\":{", s);
    fpJsonBuffer_ += stmp;
    bool firstFp = true;
    for (int t : activeTrackIdxs) {
      const auto& lb = fpLabels_[t];
      if (lb.tag[0] == '\0') continue;
      char ftmp[96];
      snprintf(ftmp, sizeof(ftmp), "\"%s\":\"0x%016llx\"",
               lb.tag, (unsigned long long)h_fpRing_[s * BUF_FP_MAX_TRACKED + t]);
      if (!firstFp) fpJsonBuffer_ += ',';
      firstFp = false;
      fpJsonBuffer_ += ftmp;
    }
    fpJsonBuffer_ += "}}";
  }
  fpJsonBuffer_ += "]}";
  return fpJsonBuffer_.c_str();
}

// ─── Slot output address fingerprinting ─────────────────────────────────────
// FNV-1a hash of slot output specialBuffer() addresses for a segment.
// Verified before replay — mismatch means output buffers were reallocated
// and the CUDA graph has stale baked-in addresses (would SIGSEGV or corrupt).
static LongType computeSlotAddrHash(NDArray** outputSlots, int startSlot, int endSlot, int totalSlots) {
  return dsp::computeSlotAddrHash(outputSlots, startSlot, endSlot, totalSlots,
      [](NDArray* a) -> void* { return a->specialBuffer(); });
}

static bool slotIsTransparentHostOnlyForGraphCoverage(
    const NativeSlot& slot,
    const SlotBufferInfo* ownership,
    NDArray** outputSlots,
    NDArray** externalArrays,
    int numExt,
    int totalOutputSlots) {
  // Replay-stable host-only classes: shape metadata, constants, fused tails,
  // and aliasing views/identity below. Constant-generation outputs are
  // covered by the trait-driven value key when their values affect replay.
  //
  // CONST_GEN restricted to VALUE_DEPENDENT_SHAPE: fill ops (zeros_as, ones_as)
  // are CUDA-graph-safe and must run inside the graph (see _gpubackend.cu comment).
  // Only CONST_GEN+VALDEP ops (range, linspace, create) need pre-capture / host-sync.
  if (slot.frozenConstantSlot() ||
      slot.hasOpTrait(sd::ops::OP_TRAIT_SHAPE_ONLY_OUTPUT) ||
      (slot.hasOpTrait(sd::ops::OP_TRAIT_CONSTANT_GENERATION) &&
       slot.hasOpTrait(sd::ops::OP_TRAIT_VALUE_DEPENDENT_SHAPE)) ||
      slot.fusedChain.isFusedChainTail) {
    return true;
  }

  if (!(slot.isViewCapableOp() || slot.isIdentityOp()) ||
      slot.wiring.numOutputs <= 0 ||
      ownership == nullptr) {
    return false;
  }

  for (int o = 0; o < slot.wiring.numOutputs; o++) {
    int outIdx = slot.wiring.outputSlotIndices[o];
    if (outIdx < 0 || outIdx >= totalOutputSlots) return false;

    BufferOwnership owner = ownership[outIdx].ownership;
    if (owner != BufferOwnership::VIEW_OF_SLOT &&
        owner != BufferOwnership::VIEW_OF_WEIGHT) {
      NDArray* out = (outputSlots != nullptr) ? outputSlots[outIdx] : nullptr;
      DataBuffer* outDb = (out != nullptr) ? out->dataBuffer() : nullptr;
      bool aliasesExternalInput = false;
      if (outDb != nullptr && externalArrays != nullptr) {
        for (int i = 0; i < slot.wiring.numInputs; i++) {
          int srcIdx = slot.wiring.inputSourceIndices[i];
          if (srcIdx >= 0) continue;
          int extIdx = -(srcIdx + 1);
          if (extIdx >= 0 && extIdx < numExt &&
              externalArrays[extIdx] != nullptr &&
              externalArrays[extIdx]->dataBuffer() == outDb) {
            aliasesExternalInput = true;
            break;
          }
        }
      }
      if (!aliasesExternalInput) return false;
    }
  }
  return true;
}

static bool slotSkipsPostReplayFixup(const NativeSlot& slot) {
  return slot.frozenConstantSlot() || slot.fusedChain.isFusedChainTail;
}

static const char* postReplayFixupSkipReason(const NativeSlot& slot) {
  if (slot.frozenConstantSlot()) return "frozen constant output is stable";
  if (slot.fusedChain.isFusedChainTail) return "fused-chain tail output is produced by the chain head";
  return "output is stable by construction";
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
        // Skip ALL variable inputs: their addresses churn every step as Java
        // recreates NDArray wrappers, causing the addr key to be perpetually
        // unstable and forcing expensive O(N) recomputation. Variable inputs
        // are handled by performPreReplaySync (H2D sync every step) and staging
        // buffers (D2D into stable plan-owned addresses for graph replay).
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
  return slot.opTraits();
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

      // Hash host-visible values when available. Async-only DSP execution must
      // not block on D2H reads here, so device-only values fall back to metadata.
      int n = (int)inputArr->lengthOf();
      if (n > 16) n = 16;  // Cap for safety
      int elemSize = DataTypeUtils::sizeOf(inputArr->dataType());
      std::vector<uint8_t> buf(n * elemSize);
      if (inputArr->buffer() && inputArr->dataBuffer() && inputArr->dataBuffer()->isPrimaryActual()) {
        std::memcpy(buf.data(), inputArr->buffer(), n * elemSize);
      } else {
        mix(reinterpret_cast<LongType>(inputArr->specialBuffer()));
        mix(static_cast<LongType>(inputArr->lengthOf()));
        mix(static_cast<LongType>(static_cast<int>(inputArr->dataType())));
        continue;
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
      slot.shapeCache.cachedShapeKey = 0;
      slot.shapeCache.cachedOutputShapes.clear();
    }
  };

  auto clearGraphStreamError = [&](cudaStream_t cudaStrm) {
    (void)cudaStrm;
    cudaGetLastError();
  };

  // ── REPLAY: cached graph with matching shapes ──
  if (seg.exec.replayHandle && seg.exec.cachedShapeKey == segShapeKey &&
      seg.exec.replayHandle->isReady()) {

    cudaStream_t cudaStr = (stream != nullptr)
        ? *static_cast<cudaStream_t*>(stream) : nullptr;
    bool replayInputsStable = true;
    bool usedAddrKey = false;
    bool usedExtAddrMatch = false;
    if (seg.exec.capturedInputAddrKey != 0) {
      LongType currentAddrKey = computeSegmentInputAddrKey(seg, externalArrays, numExt);
      replayInputsStable = (currentAddrKey == seg.exec.capturedInputAddrKey);
      usedAddrKey = true;
      DSP_DIAG(GRAPH_REPLAY,
               "REPLAY_READINESS: seg[%d-%d] execCount=%d addrKeyCheck: "
               "captured=0x%llx current=0x%llx stable=%s",
               seg.def.startSlot, seg.def.endSlot, executeCount_,
               (long long)seg.exec.capturedInputAddrKey, (long long)currentAddrKey,
               replayInputsStable ? "YES" : "NO — will invalidate");
    } else if (!seg.exec.replayHandle->getCapturedExternalAddresses().empty()) {
      replayInputsStable = externalAddrsMatch(seg, externalArrays, numExt);
      usedExtAddrMatch = true;
      DSP_DIAG(GRAPH_REPLAY,
               "REPLAY_READINESS: seg[%d-%d] execCount=%d extAddrMatch: stable=%s",
               seg.def.startSlot, seg.def.endSlot, executeCount_,
               replayInputsStable ? "YES" : "NO — will invalidate");
    } else {
      DSP_DIAG(GRAPH_REPLAY,
               "REPLAY_READINESS: seg[%d-%d] execCount=%d no addr key or ext snapshot — "
               "assuming stable (capturedInputAddrKey=0, noExtAddrs=%s)",
               seg.def.startSlot, seg.def.endSlot, executeCount_,
               seg.exec.replayHandle->getCapturedExternalAddresses().empty() ? "true" : "false");
    }
    (void)usedAddrKey; (void)usedExtAddrMatch;

    if (replayInputsStable) {
      // Unified pre-replay sync: cross-stream + H2D + staging D2D.
      // Set GRAPH_REPLAY target for proper sync behavior.
      {
        auto* execCtx = static_cast<PlanExecutionContext*>(activeExecCtx_);
        if (execCtx != nullptr) {
          execCtx->execTarget = ExecTarget::GRAPH_REPLAY;
        }
      }
      externalArrays = performPreReplaySync(externalArrays, numExt, stream, "cudagraph_replay");

      // Slot address drift check: graph bakes slot output device pointers.
      // If any slot was reallocated, the graph has stale addresses and must rebuild.
      if (seg.exec.capturedSlotAddrHash != 0) {
        LongType currentAddrHash = computeSlotAddrHash(
            outputSlots_, seg.def.startSlot, seg.def.endSlot, totalOutputSlots_);
        if (seg.exec.slotAddrDrifted(currentAddrHash)) {
          seg.exec.handleTracker.record(ReplayHandleEvent::Kind::ADDRESS_DRIFT,
                                        seg.exec.executionCount, 0, 0, "slot_addr_drift");
          DSP_DIAG(MEMORY, "SLOT ADDRESS DRIFT for seg[%d-%d]: "
                   "captured=0x%llx current=0x%llx — invalidating replay handle",
                   seg.def.startSlot, seg.def.endSlot,
                   (long long)seg.exec.capturedSlotAddrHash, (long long)currentAddrHash);
          clearGraphStreamError(cudaStr);
          platformCleanupSegmentForRebuild(seg);
          replayInputsStable = false;
        }
      }

      if (replayInputsStable) {
        auto replayStatus = replayMonolithicGraph(seg, externalArrays, numExt,
                                                  stream, "cudagraph_replay");
        if (replayStatus != Status::OK) {
          clearGraphStreamError(cudaStr);
          platformCleanupSegmentForRebuild(seg);
          DSP_THROW_SEG(EXECUTE, seg.def.startSlot,
                        "CUDA graph replay failed for seg[%d-%d]",
                        seg.def.startSlot, seg.def.endSlot);
        }
        return Status::OK;
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
    std::unique_ptr<ShapeChangeWarmupGuard> warmupGuard;
    if (shapeChanged) {
      warmupGuard.reset(new ShapeChangeWarmupGuard(*this, seg.def.startSlot, seg.def.endSlot));
    }
    // After markVariable, plan-level executeCount_ stays high but segment
    // executionCount resets to 0. needsSync() checks executeCount_ (plan-level)
    // and returns false — but warmup after invalidation needs sync to pick up
    // fresh host-written external inputs (Java .assign() writes host then syncs
    // on default stream; without prepareSpecialUse, device data stays stale).
    // Force sync during segment-level warmup when plan is already past initial warmup.
    std::unique_ptr<SyncOverride> warmupSyncGuard;
    bool isPostMarkWarmup = (executeCount_ >= 2 && seg.exec.executionCount < CAPTURE_MIN_WARMUPS);
    if (isPostMarkWarmup) {
      warmupSyncGuard.reset(new SyncOverride(*this, "postMarkVariable_warmup"));
      // Diagnostic: log slot phase states during post-markVariable warmup.
      // This is the path where stale SEALED slots caused the step 0=step 1 bug.
      if (DSP_DIAG_ENABLED(EXECUTE)) {
        int sealedCount = 0, buildingCount = 0;
        for (int si = seg.def.startSlot; si <= seg.def.endSlot && si < numSlots_; si++) {
          if (slots_[si].slotPhase.isSealed()) sealedCount++;
          else buildingCount++;
        }
        DSP_DIAG(EXECUTE, "POST_MARK_WARMUP: seg[%d-%d] segExecCount=%d planExecCount=%d "
                 "slotPhase(sealed=%d building=%d) syncOverride=YES shapeChanged=%d",
                 seg.def.startSlot, seg.def.endSlot, seg.exec.executionCount,
                 executeCount_, sealedCount, buildingCount, shapeChanged ? 1 : 0);
      }
    }
    auto warmupResult = executeSegmentSlotBySlot(seg, externalArrays, numExt, stream);
    if (warmupResult == Status::OK) {
      dspSegIncrementExecCount(seg, "cuda-graph-warmup");

      // Staging buffer sync during warmup: handled by performPreReplaySync
      // in dispatchSegment (cross-stream + H2D + D2D staging, all tracked
      // via PreReplaySyncPhase). No inline sync needed here.
    }
    return warmupResult;
  }

  // Lifecycle: warmup complete → advance segPhase to CAPTURING.
  // CUDA_GRAPHS mode skips the compile step (no JIT), so we go directly
  // from WARMUP → CAPTURING via skipCompileToCapturing(). The post-capture
  // markWarmupDone→markCompiled→markCaptured sequence (below, ~line 1166)
  // handles the legacy lifecycleState field transitions via SegmentLifecycle functions.
  if (seg.exec.segPhase.needsWarmup()) {
    SegmentLifecycle::skipToCapturing(seg.exec, "cuda_graphs",
                                     seg.def.startSlot, seg.def.endSlot);
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

  bool isOomRetry = (seg.exec.captureOomRetries > 0);
  if (!isOomRetry) {
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
  for (int s = seg.def.startSlot; s <= seg.def.endSlot; s++) {
    savedSlotPhases[s - seg.def.startSlot] = slots_[s].slotPhase;
    if (slots_[s].slotPhase.isSealed() && !slots_[s].slotPhase.isConstant) {
      slots_[s].slotPhase.unseal(); slots_[s].slotPhase.shapeCacheValid = true;  // PRIMARY: demote for capture
    }
  }

  cudaStream_t resolvedCaptureStream = cudaStr;
  if (resolvedCaptureStream == nullptr) {
    auto* defaultStreamPtr = LaunchContext::defaultContext()->getCudaStream();
    if (defaultStreamPtr != nullptr) {
      resolvedCaptureStream = *defaultStreamPtr;
    }
  }

  // Allocate capture workspace. The configured size is the baseline; large
  // segments can need more temporary capture storage than the fixed default.
  // Bound growth by the segment working set and then scale down to available
  // GPU memory if necessary. Before allocating, trim the memory pool to reclaim
  // cached-but-unused buffers.
  // Dynamic — read config each time so tests can override via system property.
  size_t CONFIGURED_CAPTURE_WORKSPACE = static_cast<size_t>(sd::env_dspCaptureWorkspaceMb()) * 1024ULL * 1024ULL;
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
    size_t workspaceSize = CONFIGURED_CAPTURE_WORKSPACE;
    if (estimatedCaptureBytes > 0 && CONFIGURED_CAPTURE_WORKSPACE > 0) {
      size_t adaptiveCeiling = CONFIGURED_CAPTURE_WORKSPACE;
      if (CONFIGURED_CAPTURE_WORKSPACE <= ((size_t)-1) / 4) {
        adaptiveCeiling = CONFIGURED_CAPTURE_WORKSPACE * 4;
      }
      size_t workingSetWorkspace = estimatedCaptureBytes / 4;
      if (workingSetWorkspace > adaptiveCeiling) {
        workingSetWorkspace = adaptiveCeiling;
      }
      if (workingSetWorkspace > workspaceSize) {
        DSP_DIAG_SEG(MEMORY, segIdx,
                     "capture workspace grown from segment working set: configured=%zuMB "
                     "workingSet=%zuMB requested=%zuMB ceiling=%zuMB",
                     CONFIGURED_CAPTURE_WORKSPACE / (1024*1024),
                     estimatedCaptureBytes / (1024*1024),
                     workingSetWorkspace / (1024*1024),
                     adaptiveCeiling / (1024*1024));
        workspaceSize = workingSetWorkspace;
      }
    }
    if (isOomRetry && CONFIGURED_CAPTURE_WORKSPACE > 0) {
      size_t retryWorkspace = CONFIGURED_CAPTURE_WORKSPACE;
      for (int r = 0; r < seg.exec.captureOomRetries && retryWorkspace <= ((size_t)-1) / 2; r++) {
        retryWorkspace *= 2;
      }
      if (retryWorkspace > workspaceSize) {
        DSP_DIAG_SEG(MEMORY, segIdx,
                     "capture workspace grown for retry %d/%d: %zuMB -> %zuMB",
                     seg.exec.captureOomRetries, GraphSegment::maxOomRetries(),
                     workspaceSize / (1024*1024), retryWorkspace / (1024*1024));
        workspaceSize = retryWorkspace;
      }
    }
    if (gpuFree > headroom) {
      size_t availableForWs = gpuFree - headroom;
      if (availableForWs < workspaceSize) {
        workspaceSize = availableForWs;
        DSP_DIAG_SEG(MEMORY, segIdx,
                     "capture workspace scaled down: gpuFree=%zuMB headroom=%zuMB → workspace=%zuMB (max=%zuMB)",
                     gpuFree / (1024*1024), headroom / (1024*1024),
                     workspaceSize / (1024*1024), CONFIGURED_CAPTURE_WORKSPACE / (1024*1024));
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
  //
  // Route through performPreReplaySync with GRAPH_CAPTURE target, which handles:
  //   1. H2D sync (variable inputs to device)
  //   2. Cross-stream ordering (default stream → DSP stream)
  //   3. D2D staging (copy to plan-owned stable buffers on the capture stream)
  NDArray** captureExternals = externalArrays;
  {
    auto* execCtx = static_cast<PlanExecutionContext*>(activeExecCtx_);
    if (execCtx != nullptr) {
      execCtx->execTarget = ExecTarget::GRAPH_CAPTURE;
      // Reset sync phase for the capture context — earlier sync in the same
      // execute() call was for the dispatchSegment GRAPH_REPLAY target, but
      // capture needs its own staging pass with stream synchronization.
      execCtx->resetSyncPhase();
    }
    captureExternals = performPreReplaySync(externalArrays, numExt, stream, "cudagraph_capture");
  }

  DSP_DIAG_SEG(MEMORY, segIdx, "tl_captureWorkspace=%p size=%zu for capture",
               tl_captureWorkspace, tl_captureWorkspaceSize);

  // RAII guard: serialize capture with concurrent executions on this device.
  // See DeviceCaptureGuard in _gpubackend.cu for the full explanation.
  // Uses try_to_lock to prevent deadlock when multiple threads try to capture
  // simultaneously — non-winning threads skip capture and execute slot-by-slot.
  struct CudaGraphCaptureGuard {
    int dev_;
    std::unique_lock<std::mutex> lock_;
    bool acquired_;
    CudaGraphCaptureGuard() : dev_(0), lock_(), acquired_(false) {
      cudaGetDevice(&dev_);
      if (dev_ < 0 || dev_ >= 16) dev_ = 0;
      lock_ = std::unique_lock<std::mutex>(g_captureMtx[dev_], std::try_to_lock);
      if (!lock_.owns_lock()) {
        return;  // Another thread is capturing — skip
      }
      g_execCount[dev_].fetch_sub(1, std::memory_order_acq_rel);
      g_captureActive[dev_].store(true, std::memory_order_release);
      bool waitResult = g_captureCV[dev_].wait_for(lock_, std::chrono::seconds(5),
          [this]{ return g_execCount[dev_].load(std::memory_order_acquire) == 0; });
      if (!waitResult) {
        g_captureActive[dev_].store(false, std::memory_order_release);
        g_execCount[dev_].fetch_add(1, std::memory_order_acq_rel);
        lock_.unlock();
        g_captureCV[dev_].notify_all();
        return;  // acquired_ stays false
      }
      acquired_ = true;
    }
    ~CudaGraphCaptureGuard() {
      if (acquired_) {
        g_captureActive[dev_].store(false, std::memory_order_release);
        g_execCount[dev_].fetch_add(1, std::memory_order_acq_rel);
        lock_.unlock();
        g_captureCV[dev_].notify_all();
      }
    }
    bool acquired() const { return acquired_; }
  } cudaGraphCaptureGuard;

  // If another thread is already capturing, skip capture for this iteration.
  // Execute slot-by-slot instead — capture will be attempted next execution.
  if (!cudaGraphCaptureGuard.acquired()) {
    DSP_DIAG(COMPILE, "CUDA_GRAPH_CAPTURE_DEFER: seg[%d-%d] another thread capturing, "
             "executing slot-by-slot this iteration",
             seg.def.startSlot, seg.def.endSlot);
    restoreCublasWorkspaceAfterCapture(stream);
    // Restore slot states saved before capture
    for (int s = seg.def.startSlot; s <= seg.def.endSlot; s++) {
      slots_[s].slotPhase = savedSlotPhases[s - seg.def.startSlot];
    }
    SegmentLifecycle::markCaptureBailout(seg.exec, seg.def.startSlot, seg.def.endSlot);
    return executeSegmentSlotBySlot(seg, externalArrays, numExt, stream);
  }

  // Raw CUDA graph capture must route every LaunchContext-backed op onto the
  // same stream passed to cudaStreamBeginCapture. Composite Triton capture
  // already uses this override for gap ops; monolithic CUDA graph capture needs
  // the same rule or generic elementwise/broadcast ops run on the LC default
  // stream and appear as false "host-only" holes in the capture audit.
  ScopedDspGapStream gapStreamCaptureGuard(cudaStr);

  // PRE-MATERIALIZE output device shape buffers BEFORE beginCapture. Allocation is illegal
  // once the thread is capturing (cudaMallocAsync → err900 cudaErrorStreamCaptureUnsupported,
  // even on a dedicated stream). An op that looks up a not-yet-cached output shapeInfo during
  // capture (e.g. slice's [4,8] ews=-1 view) hits ConstantShapeHelper cache-miss →
  // CudaShapeBufferCreator::create → replicatePointer → allocateDirect err900 → the op throws →
  // seg KERNEL_FAILURE/status=50. Touch each slot output's shapeInfo here, while alloc is still
  // legal, so every capture-time lookup is a cache HIT (these are process-lifetime shape
  // constants; warming them is off the replay hot path and idempotent).
  //
  // Also materialize the per-device capture-constant ARENA here, before beginCapture. Ops that
  // replicate a not-yet-cached shape/TAD constant DURING capture (e.g. slice's [4,8] ews=-1
  // internal view) bump-allocate it from this arena — cudaMallocAsync is illegal mid-capture
  // (err900). No capture is active yet (CaptureStateGuard set the flag, but beginCapture is
  // below), so the 64MB backing allocates legally; capture-time replicatePointer then only bumps.
  {
    int arenaDev = 0;
    cudaGetDevice(&arenaDev);
    memory::CudaMemoryPool::getInstance().ensureCaptureArena(arenaDev);
  }

  for (int preS = seg.def.startSlot; preS <= seg.def.endSlot; preS++) {
    for (int preO = 0; preO < slots_[preS].wiring.numOutputs; preO++) {
      int preSi = slots_[preS].wiring.outputSlotIndices[preO];
      if (preSi >= 0 && preSi < totalOutputSlots_ && outputSlots_[preSi] != nullptr) {
        LongType* preOutShape = const_cast<LongType*>(outputSlots_[preSi]->shapeInfo());
        if (preOutShape != nullptr) {
          ConstantShapeHelper::getInstance().bufferForShapeInfo(preOutShape);
        }
      }
    }
  }

  seg.exec.handleTracker.record(ReplayHandleEvent::Kind::CAPTURE_BEGIN,
                                seg.exec.executionCount, 0, 0, "monolithic");
  if (!handle->beginCapture(cudaStr, cudaStreamCaptureModeThreadLocal)) {
    seg.exec.handleTracker.record(ReplayHandleEvent::Kind::EXEC_ERROR,
                                  seg.exec.executionCount, 0, 0, "beginCapture_failed");
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
    }
    DSP_THROW_SEG(COMPILE, seg.def.startSlot,
                  "CUDA graph capture beginCapture failed for seg[%d-%d]",
                  seg.def.startSlot, seg.def.endSlot);
  }

  bool captureOk = true;
  bool captureOomFailure = false;
  int lastCaptureSlot = seg.def.startSlot;
  int frozenConstSkipped = 0;
  lastCaptureAudit_.clear();

  // Contract-driven capture behavior: the mode contract declares whether
  // forceSync and frozen-const-skip are appropriate for this capture style.
  // Monolithic capture (CUDA_GRAPHS): forceSync=false, skipFrozenConsts=false
  // Composite capture (TRITON etc.): forceSync=true, skipFrozenConsts=true
  auto contract = ModeContract::forMode(graphExecutionMode_);
  DSP_DIAG(GRAPH_REPLAY,
           "captureSegment: contract[forceSyncDuringCapture=%d skipFrozenConsts=%d "
           "forcesSyncOnFrozen=%d deterministicCublas=%d] mode=%s seg[%d-%d]",
           (int)contract.forceSyncDuringCapture, (int)contract.skipFrozenConstsDuringCapture,
           (int)contract.forcesSyncOnFrozen, (int)contract.requiresDeterministicCublas,
           ModeContract::modeName(static_cast<int>(graphExecutionMode_)), seg.def.startSlot, seg.def.endSlot);
  // Scoped sync override: if the contract requires sync during capture,
  // push an override so needsSync() returns true for each captured slot.
  // Destroyed at function exit — no manual restore needed.
  std::unique_ptr<SyncOverride> captureSync;
  if (contract.forceSyncDuringCapture) {
    captureSync.reset(new SyncOverride(*this, "cuda_graph_capture"));
  }

  try {
    for (int stepIdx = seg.def.startSlot; stepIdx <= seg.def.endSlot; stepIdx++) {
      lastCaptureSlot = stepIdx;

      // Skip frozen constant slots if the mode contract says to.
      if (contract.skipFrozenConstsDuringCapture && slots_[stepIdx].frozenConstantSlot()) {
        bool allOutputsPopulated = true;
        for (int o = 0; o < slots_[stepIdx].wiring.numOutputs; o++) {
          int si = slots_[stepIdx].wiring.outputSlotIndices[o];
          if (si >= 0 && si < totalOutputSlots_ && outputSlots_[si] == nullptr) {
            allOutputsPopulated = false;
            break;
          }
        }
        if (allOutputsPopulated) {
          frozenConstSkipped++;
          continue;
        }
      }

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
      size_t nodesContributed = (nodesAfter > nodesBefore) ? (nodesAfter - nodesBefore) : 0;
      bool isHostOnlyOp = (nodesContributed == 0);
      {
        cuda::CaptureAuditEntry entry;
        entry.slotIndex = stepIdx;
        entry.opName = slots_[stepIdx].ident.opName;
        entry.nodesBefore = nodesBefore;
        entry.nodesAfter = nodesAfter;
        entry.nodesContributed = nodesContributed;

        // Populate per-op node type breakdown by querying the in-progress capture graph.
        // This lets postGraphReplayFixup distinguish ops with only memcpy/memset nodes
        // (no compute kernel) from ops with actual kernel launches.
        if (nodesContributed > 0) {
          cudaStreamCaptureStatus auditCapStatus;
          cudaGraph_t auditCapGraph = nullptr;
          unsigned long long auditCapId = 0;
          cudaError_t auditErr = cudaStreamGetCaptureInfo_v2(
              cudaStr, &auditCapStatus, &auditCapId, &auditCapGraph, nullptr, nullptr);
          if (auditErr == cudaSuccess && auditCapStatus == cudaStreamCaptureStatusActive
              && auditCapGraph != nullptr) {
            size_t totalNodes = 0;
            auditErr = cudaGraphGetNodes(auditCapGraph, nullptr, &totalNodes);
            if (auditErr == cudaSuccess && totalNodes >= nodesAfter) {
              std::vector<cudaGraphNode_t> allNodes(totalNodes);
              auditErr = cudaGraphGetNodes(auditCapGraph, allNodes.data(), &totalNodes);
              if (auditErr == cudaSuccess) {
                for (size_t ni = nodesBefore; ni < nodesAfter && ni < totalNodes; ni++) {
                  cudaGraphNodeType nodeType;
                  cudaGraphNodeGetType(allNodes[ni], &nodeType);
                  switch (nodeType) {
                    case cudaGraphNodeTypeKernel:  entry.kernels++;  break;
                    case cudaGraphNodeTypeMemcpy:  entry.memcpys++;  break;
                    case cudaGraphNodeTypeMemset:  entry.memsets++;  break;
                    case cudaGraphNodeTypeMemAlloc: entry.memAllocs++; break;
                    case cudaGraphNodeTypeMemFree:  entry.memFrees++;  break;
                    default: break;
                  }
                }
              }
            }
          }
          if (auditErr != cudaSuccess) cudaGetLastError();
        }

        lastCaptureAudit_.push_back(std::move(entry));
      }

      // Per-slot capture diagnostic: shows exactly which ops contribute GPU nodes
      // and which are host-only (contribute 0 nodes). Host-only ops inside capture
      // are valid (e.g. shape ops) but are a source of confusion when replaying.
      // Also shows kernel/memcpy/memset breakdown for ops that contribute nodes
      // but have no compute kernel (e.g. relu alpha setup).
      DSP_DIAG(GRAPH_REPLAY,
               "CAPTURE_SLOT[%d/%d] op='%s' nodesBefore=%zu nodesAfter=%zu "
               "nodesContributed=%zu kernels=%d memcpys=%d memsets=%d hostOnly=%s totalSoFar=%zu",
               stepIdx, seg.def.endSlot,
               slots_[stepIdx].ident.opName.c_str(),
               nodesBefore, nodesAfter, nodesContributed,
               lastCaptureAudit_.back().kernels,
               lastCaptureAudit_.back().memcpys,
               lastCaptureAudit_.back().memsets,
               isHostOnlyOp ? "YES" : "no",
               nodesAfter);

    }
    // captureSync (SyncOverride) destroyed automatically at scope exit.
    // Log total graph node count after all slots have been captured.
    {
      size_t totalNodeCount = handle->getNumNodesDuringCapture(cudaStr);
      DSP_DIAG(GRAPH_REPLAY,
               "CAPTURE_COMPLETE: seg[%d-%d] totalSlots=%d frozenConstSkipped=%d "
               "totalGraphNodes=%zu captureOk=%s",
               seg.def.startSlot, seg.def.endSlot,
               seg.def.endSlot - seg.def.startSlot + 1,
               frozenConstSkipped, totalNodeCount,
               captureOk ? "YES" : "FAILED");
    }
  } catch (const std::exception& e) {
    // Guard destructor handles TLS cleanup + host ptr freeing on rethrow.
    // captureSync (SyncOverride) destroyed automatically by stack unwinding.
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
    }
    platformCleanupSegmentForRebuild(seg);
    THROW_EXCEPTION("Unknown exception during CUDA graph capture");
  }

  if (frozenConstSkipped > 0) {
    DSP_DIAG_SEG(COMPILE, segIdx, "capture skipped %d frozen constant slots (of %d total)",
                 frozenConstSkipped, seg.def.endSlot - seg.def.startSlot + 1);
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
      cudaGetLastError();
    }

    // Guard destructor will free captured host ptrs (commit() not called yet).

    if (captureOomFailure && seg.exec.captureOomRetries < GraphSegment::maxOomRetries()) {
      dspSegScheduleOomRetry(seg);
      DSP_DIAG_SEG(MEMORY, segIdx, "graph capture OOM for seg[%d-%d], retry %d/%d after exec %d",
                   seg.def.startSlot, seg.def.endSlot,
                   seg.exec.captureOomRetries, GraphSegment::maxOomRetries(),
                   seg.exec.captureRetryAfterExec);
    } else {
      SegmentLifecycle::markFailed(seg.exec, "cuda_graph_capture_failed",
                                   seg.def.startSlot, seg.def.endSlot);
    }

    // Arrays persist across capture failures — only slot state needs restoring.

    platformCleanupSegmentForRebuild(seg);

    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(0);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage("");
    std::memcpy(outputSlots_, preCapOutputSlots.data(), sizeof(NDArray*) * totalOutputSlots_);
    for (int s = seg.def.startSlot; s <= seg.def.endSlot; s++) {
      slots_[s].slotPhase = savedSlotPhases[s - seg.def.startSlot];  // PRIMARY restore
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
  auto cleanupCaptureBuffersOnFailure = [&seg, &savedSlotPhases, this]() {
    // Arrays persist across capture failures — only slot state needs restoring.
    for (int s = seg.def.startSlot; s <= seg.def.endSlot; s++) {
      slots_[s].slotPhase = savedSlotPhases[s - seg.def.startSlot];  // PRIMARY restore
    }
  };

  if (!handle->endCapture(cudaStr)) {
    seg.exec.handleTracker.record(ReplayHandleEvent::Kind::EXEC_ERROR,
                                  seg.exec.executionCount, 0, 0, "endCapture_failed");
    cudaGetLastError();
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

  seg.exec.handleTracker.record(ReplayHandleEvent::Kind::CAPTURE_END,
                                seg.exec.executionCount,
                                static_cast<int>(handle->getNumNodes()), 0, "monolithic");

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

        SegmentLifecycle::evictSegmentCapture(evictSeg.exec,
                                              evictSeg.def.startSlot, evictSeg.def.endSlot);

        numEvicted++;

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
      // dspSegScheduleOomRetry uses GraphSegment::retryInterval() for the standard case.
      // When segments were evicted, freed memory is immediately available so retry sooner.
      dspSegScheduleOomRetry(seg);
      if (numEvicted > 0) {
        seg.exec.captureRetryAfterExec = seg.exec.executionCount + 1;
      }
      int retryInterval = seg.exec.captureRetryAfterExec - seg.exec.executionCount;
      DSP_DIAG(MEMORY, "graph instantiate OOM for seg[%d-%d], will retry %d/%d after exec %d (evicted %d segments, interval=%d)",
               seg.def.startSlot, seg.def.endSlot,
               seg.exec.captureOomRetries, GraphSegment::maxOomRetries(),
               seg.exec.captureRetryAfterExec, numEvicted, retryInterval);
    } else {
      SegmentLifecycle::markFailed(seg.exec, "cuda_graph_instantiate_failed",
                                   seg.def.startSlot, seg.def.endSlot);
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
                   "marking segment as zero-kernel slot-by-slot",
                   seg.def.startSlot, seg.def.endSlot);
      SegmentLifecycle::markZeroKernel(seg.exec, "empty_cuda_graph", seg.def.startSlot, seg.def.endSlot);
      tl_captureReplicateCache.clear();
      platformCleanupSegmentForRebuild(seg);
      dspSegIncrementExecCount(seg, "cuda-graph-zero-kernel");
      return Status::OK;
    }

    // Interleaved non-transparent host-only check: after the gap-stream capture
    // override above, ordinary GPU-capable ops must contribute graph nodes. If a
    // materializing op still contributes 0 nodes before a downstream GPU op, the
    // segment topology is invalid for monolithic capture and must be fixed at
    // segmentation level rather than hidden behind permanent slot-by-slot.
    if (!lastCaptureAudit_.empty()) {
      bool hasInterleavedHostOnly = false;
      for (size_t ai = 0; ai < lastCaptureAudit_.size(); ai++) {
        if (lastCaptureAudit_[ai].nodesContributed == 0) {
          int hostSlotIdx = lastCaptureAudit_[ai].slotIndex;
          bool transparent =
              hostSlotIdx >= 0 && hostSlotIdx < numSlots_ &&
              slotIsTransparentHostOnlyForGraphCoverage(
                  slots_[hostSlotIdx], slotOwnership_, outputSlots_, captureExternals,
                  numExt, totalOutputSlots_);
          if (transparent) continue;
          for (size_t aj = ai + 1; aj < lastCaptureAudit_.size(); aj++) {
            if (lastCaptureAudit_[aj].nodesContributed > 0) {
              DSP_DIAG_SEG(COMPILE, segIdx,
                           "INTERLEAVED_HOST_ONLY: seg[%d-%d] slot %d (%s) is host-only (0 nodes) "
                           "but downstream slot %d (%s) has %zu GPU nodes after capture-stream "
                           "unification. This is a segmentation/capture coverage bug, not a "
                           "valid terminal slot-by-slot outcome.",
                           seg.def.startSlot, seg.def.endSlot,
                           lastCaptureAudit_[ai].slotIndex,
                           lastCaptureAudit_[ai].opName.c_str(),
                           lastCaptureAudit_[aj].slotIndex,
                           lastCaptureAudit_[aj].opName.c_str(),
                           lastCaptureAudit_[aj].nodesContributed);
              hasInterleavedHostOnly = true;
              break;
            }
          }
          if (hasInterleavedHostOnly) break;
        }
      }
      if (hasInterleavedHostOnly) {
        tl_captureReplicateCache.clear();
        clearGraphStreamError(cudaStr);
        platformCleanupSegmentForRebuild(seg);
        clearGraphStreamError(cudaStr);
        std::memcpy(outputSlots_, preCapOutputSlots.data(), sizeof(NDArray*) * totalOutputSlots_);
        for (int s = seg.def.startSlot; s <= seg.def.endSlot; s++) {
          slots_[s].slotPhase = savedSlotPhases[s - seg.def.startSlot];
        }
        invalidateSegmentShapeState(seg);
        sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(0);
        sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage("");
        return Status::KERNEL_FAILURE;
      }
    }
  }

  if (!handle->launchAsync(cudaStr)) {
    cudaGetLastError();
    // Guard destructor frees host ptrs (commit() not called).
    clearGraphStreamError(cudaStr);
    SegmentLifecycle::markFailed(seg.exec, "cuda_graph_launch_failed",
                                 seg.def.startSlot, seg.def.endSlot);
    cleanupCaptureBuffersOnFailure();
    platformCleanupSegmentForRebuild(seg);
    std::memcpy(outputSlots_, preCapOutputSlots.data(), sizeof(NDArray*) * totalOutputSlots_);
    invalidateSegmentShapeState(seg);
    DSP_THROW_SEG(COMPILE, seg.def.startSlot,
                  "CUDA graph launchAsync failed for seg[%d-%d]",
                  seg.def.startSlot, seg.def.endSlot);
  }

  // Post-replay fixup: tick device actuality + re-execute host-only ops.
  // During capture, host-only ops (0 graph nodes) executed with stale inputs
  // because their GPU-dependent inputs were only RECORDED, not executed.
  // The launchAsync above produced fresh GPU outputs — re-execute host-only
  // ops now to pick up the correct values.
  {
    auto fixupStatus = postGraphReplayFixup(seg, externalArrays, numExt,
                                            stream, "post_capture");
    if (fixupStatus != Status::OK) return fixupStatus;
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
  seg.exec.sealCapture(
      computeSegmentInputAddrKey(seg, captureExternals, numExt),
      computeCreateOpValueKey(seg, captureExternals, numExt),
      computeSlotAddrHash(outputSlots_, seg.def.startSlot, seg.def.endSlot, totalOutputSlots_),
      "CUDA",
      /*gapsCaptured=*/false);
  snapshotExternalAddrs(seg, captureExternals, numExt);

  // Pin every device address this captured graph baked in (weights/intermediates/outputs),
  // at seal — so no later close()/rebind/pool-reuse can dangle a buffer a live replay reads.
  // pinOwnedOutputs=true: a captured graph bakes raw intermediate addresses, which must outlive it.
  pinSegmentGraphBakedSlots(seg, captureExternals, numExt, /*pinOwnedOutputs=*/true);

  dspSegIncrementExecCount(seg, "cuda-graph-capture-complete");
  totalGraphReplays_++;

  // Mark as captured by raw CUDA graph path (not Triton).
  // This prevents the Triton replay path in NativeDynamicShapePlan_gpubackend.cpp
  // from incorrectly handling this segment — Triton replay has incompatible
  // D2D copy and arg table logic that can corrupt cross-segment data.
  // Note: compiledByBackend is already set by sealCapture() above.

  // Clear compilationFailed — the CUDA graph path succeeded even if the Triton path
  // failed earlier. Without this, cleanup treats this segment as non-graph-managed
  // and frees its output/cross-segment slots, causing stale data on replay.
  SegmentLifecycle::clearCompilationFailedOnRecovery(seg.exec,
      "cuda_graph_capture_succeeded", seg.def.startSlot, seg.def.endSlot);

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
    snap.data.clear();
    snaps.push_back(std::move(snap));
  }
  DSP_DIAG(VERIFY, "REPLAY_VERIFY: saved %zu snapshots from replay (%s path)", snaps.size(), pathLabel);

  // 3. Re-execute slot-by-slot for ground truth
  // RAII guard overrides compilationFailed+executionCount to force slot-by-slot,
  // then restores original state on scope exit (even if an exception occurs).
  ReplayVerifyStateGuard verifyGuard(seg.exec);

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
  for (int s = seg.def.startSlot; s <= seg.def.endSlot; s++) {
    savedVerifySlotPhases[s - seg.def.startSlot] = slots_[s].slotPhase;
    if (slots_[s].slotPhase.isSealed() && !slots_[s].slotPhase.isConstant) {
      slots_[s].slotPhase.unseal();
      slots_[s].slotPhase.shapeCacheValid = true;
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
  }
  executeCount_ = savedExecCountGlobal;
  // seg.exec.executionCount and seg.exec.compilationFailed are restored
  // by ReplayVerifyStateGuard destructor at scope exit.

  if (freshStatus != Status::OK) {
    DSP_DIAG(VERIFY, "REPLAY_VERIFY: slot-by-slot re-execution FAILED (%s path)", pathLabel);
    return;
  }

  DSP_DIAG(VERIFY, "REPLAY_VERIFY: fresh slot-by-slot execution queued on stream=%p "
                   "(no blocking stream sync)", (void*)cudaStr);

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
    if (snap.data.empty()) continue;
    std::vector<uint8_t> freshData(compareCount * elemSize);
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
// postGraphReplayFixup — unified post-replay fixup for all graph replay paths.
// 1. Ticks device actuality on all slot outputs (graph replay writes device
//    memory directly without registerSpecialUse — without this tick,
//    syncToHost sees stale host data).
// 2. Re-executes ops that need live execution after graph replay:
//    a) Host-only ops (0 CUDA graph nodes, e.g. shape, identity)
//    b) Setup-only ops (contributed graph nodes but 0 compute kernels,
//       e.g. relu whose only nodes are memcpy for alpha + memset — no kernel
//       that reads the input and writes the output). On replay, these ops'
//       memcpy/memset nodes run with baked-in capture-time addresses, so
//       the output stays frozen at capture-time values.
// ═══════════════════════════════════════════════════════════════════════════
Status NativeDynamicShapePlan::postGraphReplayFixup(
    GraphSegment& seg, NDArray** externalArrays, int numExt,
    void* stream, const char* diagTag) {

  // Step 1: tick device actuality on all slot outputs in this segment.
  for (int stepIdx = seg.def.startSlot; stepIdx <= seg.def.endSlot; stepIdx++) {
    if (stepIdx < 0 || stepIdx >= numSlots_) continue;
    const NativeSlot& slot = slots_[stepIdx];
    for (int o = 0; o < slot.wiring.numOutputs; o++) {
      int outIdx = slot.wiring.outputSlotIndices[o];
      if (outIdx < 0 || outIdx >= totalOutputSlots_) continue;
      NDArray* arr = outputSlots_[outIdx];
      if (arr != nullptr && arr->dataBuffer() != nullptr && !arr->dataBuffer()->isClosed()) {
        arr->tickWriteDevice();
      }
    }
  }

  // Step 2: re-execute ops that have no compute kernel in the captured graph.
  // An op needs re-execution if:
  //   - nodesContributed == 0: pure host-only (shape ops, identity)
  //   - nodesContributed > 0 && kernels == 0: setup-only (e.g. relu with
  //     memcpy for alpha param + memset, but no compute kernel to process input)
  // CRITICAL: keep replay fixups on the same DSP stream as the graph launch.
  // LaunchContext::getCudaStream() honors tl_dspGapStream, so executeSlot()
  // below records any GPU work after the graph replay without CPU synchronization
  // or a cross-stream event hop.
  if (!lastCaptureAudit_.empty()) {
    ScopedDspGapStream gapStreamFixupGuard(
        stream != nullptr ? *static_cast<cudaStream_t*>(stream) : nullptr);
    for (const auto& entry : lastCaptureAudit_) {
      if (entry.slotIndex >= seg.def.startSlot &&
          entry.slotIndex <= seg.def.endSlot &&
          (entry.nodesContributed == 0 || entry.kernels == 0)) {
        if (entry.slotIndex >= 0 && entry.slotIndex < numSlots_ &&
            slotSkipsPostReplayFixup(slots_[entry.slotIndex])) {
          DSP_DIAG(EXECUTE, "%s: post-replay fixup skip slot %d (%s) — %s",
                   diagTag, entry.slotIndex, entry.opName.c_str(),
                   postReplayFixupSkipReason(slots_[entry.slotIndex]));
          continue;
        }
        const char* reason = (entry.nodesContributed == 0) ? "host-only (0 nodes)"
                             : "setup-only (0 kernels)";
        DSP_DIAG(EXECUTE, "%s: post-replay re-exec slot %d (%s) — %s "
                 "(nodes=%zu kernels=%d memcpys=%d memsets=%d)",
                 diagTag, entry.slotIndex, entry.opName.c_str(), reason,
                 entry.nodesContributed, entry.kernels, entry.memcpys, entry.memsets);
        SyncOverride reExecSync(*this, diagTag);
        auto slotStatus = executeSlot(entry.slotIndex, externalArrays, numExt, stream);
        if (slotStatus != Status::OK) {
          DSP_DIAG(EXECUTE, "%s: re-exec slot %d FAILED status=%d",
                   diagTag, entry.slotIndex, (int)slotStatus);
          return slotStatus;
        }
      }
    }
  }

  return Status::OK;
}

// ═══════════════════════════════════════════════════════════════════════════
// replayMonolithicGraph — consolidated monolithic CUDA graph replay sequence.
// Replaces duplicated inline replay blocks in executeSegmentWithGraph (regular
// replay) and platformTryFrozenFastPath (monolithic branch).
//
// Sequence:
//   1. Triton arg table refresh (generation-counter gated)
//   2. prezeroSegmentOutputs (zero accumulator slots)
//   3. cuBLAS workspace zero
//   4. replayHandle->replay(stream)
//   5. Counter increments (totalGraphReplays_, executionCount, lastReplayExecCount)
//   6. postGraphReplayFixup (tick device actuality + host-only re-execution)
//   7. performReplayVerify (if tritonVerifyKernels enabled)
//
// Caller is responsible for:
//   - performPreReplaySync (H2D + staging D2D) — done once per execute, not per segment
//   - Slot address drift check (cudagraph.cu caller invalidates before calling this)
//   - Binding CUDA device (bindSegmentCudaDevice)
//   - Checking replayHandle->isReady()
// ═══════════════════════════════════════════════════════════════════════════
Status NativeDynamicShapePlan::replayMonolithicGraph(
    GraphSegment& seg, NDArray** externalArrays, int numExt,
    void* stream, const char* diagTag) {

  cudaStream_t cudaStr = (stream != nullptr)
      ? *static_cast<cudaStream_t*>(stream) : nullptr;

  // ── Step 1: Triton arg table refresh (generation-counter gated) ──
  // needsArgRefresh() is true when:
  //   (a) Triton arg tables need refreshing for a new input shape/address, OR
  //   (b) VIEW-BUF-CHANGE fired (view-of-ext-input address changed) and bumped
  //       argTableGeneration to prevent the frozen fast path from replaying the
  //       segment with stale baked device addresses (→ err700).
  // markArgsCurrent() is called unconditionally so case (b) is always cleared
  // after the segment is successfully replayed in the normal path.
  if (seg.exec.needsArgRefresh()) {
#if HAVE_TRITON
    auto* tritonBackend = dynamic_cast<TritonGraphBackend*>(getGpuGraphBackend());
    if (tritonBackend != nullptr) {
      tritonBackend->refreshArgTablesForReplay(seg, externalArrays, numExt,
                                               outputSlots_, totalOutputSlots_,
                                               stream);
      tritonBackend->copyConsolidatedArgTableToDevice(seg, stream);
    }
#endif

    seg.exec.markArgsCurrent();
  } else {
    DSP_DIAG(EXECUTE, "%s: args current — skip refresh seg[%d-%d]",
             diagTag, seg.def.startSlot, seg.def.endSlot);
  }

  // ── Step 1b: Slot address drift guard for monolithic CUDA graph ───────────
  // The monolithic CUDA graph has cuBLAS (and other native op) kernel nodes with
  // device pointers baked in at capture time. Unlike Triton kernels (whose arg
  // tables are refreshed above), cuBLAS args are immutable after capture.
  //
  // If any output slot's specialBuffer() changed since capture — e.g., because
  // refreshStaleViewWrappersInSegment installed a new view NDArray with a
  // different offset into a different DataBuffer — the cuBLAS nodes have stale
  // pointers. Launching the graph would read freed / garbage GPU memory → err700.
  //
  // Detection: re-compute the FNV1a hash of all slot specialBuffer() addresses
  // in this segment and compare against capturedSlotAddrHash (set at sealCapture).
  // Mismatch → invalidate + return KERNEL_FAILURE so the caller triggers recapture.
  //
  // This mirrors the SLOT_ADDR_DRIFT check in executeSegmentWithCudaGraph (the
  // composite-replay path) but is placed here for the monolithic-replay path.
  if (seg.exec.capturedSlotAddrHash != 0) {
    LongType currentSlotHash = computeSlotAddrHash(
        outputSlots_, seg.def.startSlot, seg.def.endSlot, totalOutputSlots_);
    if (seg.exec.slotAddrDrifted(currentSlotHash)) {
      DSP_DIAG(EXECUTE,
               "%s: MONOLITHIC_SLOT_ADDR_DRIFT seg[%d-%d] "
               "captured=0x%llx current=0x%llx — cuBLAS args are stale, "
               "invalidating for recapture",
               diagTag, seg.def.startSlot, seg.def.endSlot,
               (long long)seg.exec.capturedSlotAddrHash,
               (long long)currentSlotHash);
      SegmentLifecycle::invalidateForRebuild(this, seg, "monolithic_slot_addr_drift");
      return Status::KERNEL_FAILURE;
    }
    DSP_DIAG(EXECUTE, "%s: slot addr hash stable seg[%d-%d] hash=0x%llx",
             diagTag, seg.def.startSlot, seg.def.endSlot,
             (long long)currentSlotHash);
  }

  // ── Step 1.5: Live execution of value-shape create ops ──
  // When create ops (range, create, lin_space — CONSTANT_GENERATION +
  // VALUE_DEPENDENT_SHAPE) were excluded from the CUDA graph during native-only
  // capture (createOpsExcludedFromGraph=true), their output slots contain stable
  // device buffers that the graph reads.  But the VALUES must be updated each step
  // because their inputs change (position IDs, step count, sequence length).
  // Execute them live here — outside the CUDA graph — before launching the graph,
  // so the graph reads up-to-date values from the already-allocated slot buffers.
  if (seg.exec.createOpsExcludedFromGraph) {
    int liveCreateCount = 0;
    SyncOverride liveCreateSync(*this, "live_create_ops");
    for (int s = seg.def.startSlot; s <= seg.def.endSlot; s++) {
      const uint32_t slotTraits = slots_[s].opTraits();
      if ((slotTraits & sd::ops::OP_TRAIT_CONSTANT_GENERATION) != 0 &&
          (slotTraits & sd::ops::OP_TRAIT_VALUE_DEPENDENT_SHAPE) != 0) {
        auto liveStatus = executeSlot(s, externalArrays, numExt, stream);
        if (liveStatus != Status::OK) {
          DSP_DIAG(EXECUTE, "%s: live create-op slot %d (%s) FAILED status=%d",
                   diagTag, s, slots_[s].ident.opName.c_str(), static_cast<int>(liveStatus));
          // Non-fatal: the graph will use stale values from a prior step (still
          // better than invalidating the entire graph).  Do NOT return failure —
          // the graph replay may still produce usable output.
        }
        liveCreateCount++;
      }
    }
    DSP_DIAG(EXECUTE, "%s: executed %d value-shape create ops live before graph launch seg[%d-%d]",
             diagTag, liveCreateCount, seg.def.startSlot, seg.def.endSlot);
  }

  // ── Step 2: Prezero segment outputs ──
  // Slots that accumulate (e.g. scatter-add, reduce) need their output buffers
  // zeroed before each replay to prevent stale value accumulation and FP drift.
  prezeroSegmentOutputs(seg, stream);

  // ── Step 3: cuBLAS workspace zero ──
  // Zero cuBLAS workspace before replay to match live cuBLAS behavior.
  if (cublasWorkspaceBuffer_ != nullptr && cublasWorkspaceSize_ > 0) {
    cudaMemsetAsync(cublasWorkspaceBuffer_, 0, cublasWorkspaceSize_, cudaStr);
  }

  // ── Step 4: Replay ──
  DSP_DIAG(GRAPH_REPLAY,
           "%s: REPLAY seg[%d-%d] execCount=%d segExecCount=%d",
           diagTag, seg.def.startSlot, seg.def.endSlot,
           executeCount_, seg.exec.executionCount);

  seg.exec.handleTracker.record(ReplayHandleEvent::Kind::REPLAY,
                                seg.exec.executionCount, 0, 0, "monolithic");
  if (!seg.exec.replayHandle->replay(stream)) {
    seg.exec.handleTracker.record(ReplayHandleEvent::Kind::EXEC_ERROR,
                                  seg.exec.executionCount, 0, 0, "monolithic_replay_failed");
    DSP_DIAG(EXECUTE, "%s: monolithic replay FAILED seg[%d-%d]",
             diagTag, seg.def.startSlot, seg.def.endSlot);
    return Status::KERNEL_FAILURE;
  }

  // ── Step 5: Counter increments ──
  seg.exec.lastReplayExecCount = executeCount_;
  totalGraphReplays_++;
  dspSegIncrementExecCount(seg, "monolithic-replay");

  DSP_DIAG(GRAPH_REPLAY,
           "%s: REPLAY_SUCCESS seg[%d-%d] execCount=%d segExecCount=%d totalReplays=%d",
           diagTag, seg.def.startSlot, seg.def.endSlot,
           executeCount_, seg.exec.executionCount, totalGraphReplays_);

  // ── Step 6: Post-replay fixup ──
  // Tick device actuality on all slot outputs + re-execute host-only ops.
  auto fixupStatus = postGraphReplayFixup(seg, externalArrays, numExt,
                                          stream, diagTag);
  if (fixupStatus != Status::OK) return fixupStatus;

  // ── Step 7: Optional replay verify ──
  if (Environment::getInstance().tritonVerifyKernels()) {
    performReplayVerify(seg, externalArrays, numExt, stream, diagTag);
  }

  return Status::OK;
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

  // One-time activation of fingerprint ring (checks env BUF_FP_RING=1)
  maybeInitFingerprintRing();

  cudaStream_t cudaStr = (stream != nullptr)
      ? *static_cast<cudaStream_t*>(stream) : nullptr;

  // One-time allocation — fixed size, never resized.
  if (placeholderStagingBuffers_ == nullptr) {
    placeholderStagingBuffers_ = new NDArray*[numExt]();  // zero-initialized
    effectiveExternals_ = new NDArray*[numExt]();
  }

  // Fast path: after first call, only iterate variable input indices
  // instead of all 1000+ entries. Non-variable (weight) pointers are stable.
  // Guard: fast path requires all variable staging buffers to be allocated.
  // After markExternalInputVariable, cachedVariableExtIndices_ is populated
  // immediately but staging buffers haven't been allocated yet — fall through
  // to the slow path which handles allocation.
  bool allStagingAllocated = !cachedVariableExtIndices_.empty();
  if (allStagingAllocated) {
    for (int i : cachedVariableExtIndices_) {
      if (placeholderStagingBuffers_[i] == nullptr) {
        allStagingAllocated = false;
        break;
      }
    }
  }

  if (allStagingAllocated) {
    // Populate ALL entries from externalArrays first — the loop below only
    // overwrites variable entries with staging buffers. Without this, non-variable
    // entries (weights/constants) stay null from the zero-initialized allocation,
    // causing NULL input errors during CUDA graph capture.
    std::memcpy(effectiveExternals_, externalArrays, sizeof(NDArray*) * numExt);

    // Weight rebinding (associateArrayWithVariable) is handled on the Java side:
    // DynamicShapePlanExecutor detects identity changes and passes the updated
    // INDArray via setGraphContextInputArray before calling into C++. No need
    // to scan all ~1333 non-variable inputs for isPrimaryActual() here.

    int copiedCount = 0, skippedNull = 0, skippedEmpty = 0, skippedNullBuf = 0, skippedJniWrite = 0;
    for (int i : cachedVariableExtIndices_) {
      NDArray* ext = externalArrays[i];
      effectiveExternals_[i] = externalArrays[i];  // default passthrough
      if (ext == nullptr || ext->isEmpty()) {
        skippedEmpty++;
        continue;
      }

      NDArray* staging = placeholderStagingBuffers_[i];

      // If JNI wrote directly to staging via writeDeviceBuffer*, skip D2D overwrite
      // — the staging buffer already has the fresh data from the JNI write.
      bool jniWritten = (i < static_cast<int>(deviceWritePending_.size()) && deviceWritePending_[i]);
      if (jniWritten) {
        deviceWritePending_[i] = false;
        skippedJniWrite++;
        staging->dataBuffer()->writeSpecial();
        effectiveExternals_[i] = staging;
        DSP_DIAG(MEMORY,
                 "STAGING_D2D[%d]: SKIPPED — JNI direct write pending (deviceWritePending), "
                 "staging=%p already has fresh device data",
                 i, staging->specialBuffer());
        continue;
      }

      void* dstBuf = staging->specialBuffer();
      void* srcBuf = ext->specialBuffer();
      DSP_LIFECYCLE_EVENT(executeCount_, i, "STAGING_D2D_SRC", ext);
      DSP_LIFECYCLE_EVENT(executeCount_, i, "STAGING_D2D_DST_BEFORE", staging);
      if (dstBuf != nullptr && srcBuf != nullptr) {
        size_t bytes = static_cast<size_t>(ext->lengthOf()) * ext->sizeOfT();
        if (bytes > 0) {
          // A non-contiguous VIEW source (e.g. a KV-cache slice past_key = staticKey[:,:,0:N,:])
          // MUST NOT be staged with a raw contiguous cudaMemcpyAsync: that copies bytes by linear
          // offset, pulling in data OUTSIDE the view (a scatter to staticKey[:,:,N,:]) and skipping
          // the strided gaps — corrupting the staging buffer the captured CUDA graph replays from
          // (this is the testViewAliasing MaxDiff bug). Only a contiguous source is safe for the
          // raw byte copy. For a view, use a stride-aware assign that gathers by the source's
          // actual strides, routed onto cudaStr (the staging/replay stream, which getCudaStream()
          // honors via tl_dspGapStream) so it stays ordered with the graph replay.
          if (shape::strideDescendingCAscendingF(ext->shapeInfo())) {
            cudaMemcpyAsync(dstBuf, srcBuf, bytes, cudaMemcpyDeviceToDevice, cudaStr);
          } else {
            ScopedDspGapStream gapToCudaStr(cudaStr);
            staging->assign(ext);
          }
          DSP_LIFECYCLE_RAW(executeCount_, i, "STAGING_D2D_ISSUED",
                            (void*)ext, (void*)ext->dataBuffer(), srcBuf, ext->buffer(),
                            ext->dataBuffer()->isPrimaryActual(),
                            ext->dataBuffer()->isSpecialActual(),
                            (int64_t)bytes);
          copiedCount++;
        }
      } else {
        skippedNullBuf++;
      }

      staging->dataBuffer()->writeSpecial();
      DSP_LIFECYCLE_EVENT(executeCount_, i, "STAGING_D2D_DST_AFTER", staging);

      // ── Shape corruption detector ─────────────────────────────────
      // Validates that the D2D copy did not overwrite the staging
      // NDArray's shapeInfo memory. The shapeInfo is a CPU pointer into
      // ConstantShapeHelper's trie; if a buffer overrun or address
      // aliasing causes the D2D copy to land on CPU shape memory, the
      // rank field will contain data values (e.g., float32 200.0 =
      // 0x43480000) instead of a small integer.
      if (staging->shapeInfo() != nullptr) {
        LongType sRank = staging->shapeInfo()[0];
        if (sRank < 0 || sRank > SD_MAX_RANK) {
          DSP_DIAG(MEMORY,
                   "STAGING_SHAPE_CORRUPTION_DETECTED: ext[%d] shapeInfo[0]=%lld (0x%llx) "
                   "after D2D copy. staging=%p shapeInfo=%p specialBuffer=%p "
                   "primaryBuffer=%p lenInBytes=%lld",
                   i, (long long)sRank, (unsigned long long)sRank,
                   (void*)staging, (void*)staging->shapeInfo(),
                   staging->specialBuffer(), staging->buffer(),
                   (long long)staging->dataBuffer()->getLenInBytes());
        }
      }

      // ── Sync-free fingerprint (BUF_FP_RING=1) ─────────────────────
      // Record XOR fingerprint of the staging buffer AFTER D2D, on the same
      // stream — no host sync, ordered with the copy above.
      if (fpRingEnabled_) {
        // Assign a stable track slot: use position of i in cachedVariableExtIndices_
        // (determined once; order is fixed after first entry into the fast path).
        int trackSlot = fpRingStagingCount_;
        // Walk cachedVariableExtIndices_ to find the slot for i
        for (int ci = 0; ci < static_cast<int>(cachedVariableExtIndices_.size()); ci++) {
          if (cachedVariableExtIndices_[ci] == i) { trackSlot = ci; break; }
        }
        if (trackSlot < BUF_FP_MAX_STAGING) {
          // Register label on first step (executeCount_ == stable first step)
          if (fpLabels_[trackSlot].tag[0] == '\0') {
            snprintf(fpLabels_[trackSlot].tag, sizeof(fpLabels_[trackSlot].tag), "stg[%d]", i);
            fpLabels_[trackSlot].extIdx   = i;
            fpLabels_[trackSlot].groupIdx = -1;
            fpLabels_[trackSlot].whichAB  = -1;
            if (trackSlot + 1 > fpRingStagingCount_) fpRingStagingCount_ = trackSlot + 1;
          }
          size_t fpBytes = static_cast<size_t>(staging->dataBuffer()->getLenInBytes());
          recordBufFingerprintPublic(cudaStr, executeCount_, trackSlot,
                                     staging->specialBuffer(), fpBytes);
        }
      }

      effectiveExternals_[i] = staging;
    }

    // Detect silent D2D skip conditions — O(1) counter checks, zero perf impact.
    if (copiedCount == 0 && static_cast<int>(cachedVariableExtIndices_.size()) > 0) {
      DSP_DIAG(EXECUTE,
               "STAGING_D2D_WARNING: ALL %d variable inputs skipped D2D copy! "
               "Breakdown: empty=%d nullBuf=%d. "
               "CUDA graph replay will use STALE staging data.",
               static_cast<int>(cachedVariableExtIndices_.size()),
               skippedEmpty, skippedNullBuf);
    }
    DSP_DIAG(EXECUTE, "STAGING_D2D: copied=%d skippedEmpty=%d "
             "skippedNullBuf=%d total=%d",
             copiedCount, skippedEmpty, skippedNullBuf,
             static_cast<int>(cachedVariableExtIndices_.size()));

    return effectiveExternals_;
  }

  for (int i = 0; i < numExt; i++) {
    // Non-variable inputs (model weights) — pass through directly.
    // Their specialBuffer() is stable (same DataBuffer for plan lifetime).
    // Weight rebinding is handled on the Java side via identity detection.
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
        DSP_DIAG(MEMORY,
                 "STAGING_D2D_SLOW[%d]: SKIPPED — KV cache buffer detected "
                 "(devAddr=%p matches kvScatterEntry), passing through directly",
                 i, ext->specialBuffer());
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
      DSP_DIAG(MEMORY,
               "STAGING_ALLOC: ext[%d] name='%s' dtype=%d len=%lld bytes=%lld "
               "srcBuf=%p dstBuf=%p — first-time staging buffer allocated",
               i,
               (i < static_cast<int>(externalInputNames_.size()))
                   ? externalInputNames_[i].c_str() : "?",
               static_cast<int>(ext->dataType()),
               (long long)ext->lengthOf(),
               (long long)(ext->lengthOf() * ext->sizeOfT()),
               ext->specialBuffer(), staging->specialBuffer());

      // Verify shape immediately after allocation
      if (staging->shapeInfo() != nullptr) {
        LongType sRank = staging->shapeInfo()[0];
        if (sRank < 0 || sRank > SD_MAX_RANK) {
          DSP_DIAG(MEMORY,
                   "STAGING_SHAPE_CORRUPT_AT_ALLOC: ext[%d] rank=%lld (0x%llx) "
                   "shapeInfo=%p specialBuffer=%p primaryBuffer=%p",
                   i, (long long)sRank, (unsigned long long)sRank,
                   (void*)staging->shapeInfo(), staging->specialBuffer(),
                   staging->buffer());
        }
      }
    }

    // If JNI wrote directly to staging, skip D2D overwrite (slow path)
    bool jniWritten = (i < static_cast<int>(deviceWritePending_.size()) && deviceWritePending_[i]);
    if (jniWritten) {
      deviceWritePending_[i] = false;
      DSP_DIAG(MEMORY,
               "STAGING_D2D_SLOW[%d]: SKIPPED — JNI direct write pending (deviceWritePending), "
               "staging already has fresh device data. srcBuf=%p dstBuf=%p",
               i, ext->specialBuffer(), staging->specialBuffer());
    } else {
      // D2D copy: external input (already H2D-synced) → plan-owned staging buffer.
      // Async on the execution stream — no CPU sync needed.
      void* dstBuf = staging->specialBuffer();
      void* srcBuf = ext->specialBuffer();
      if (dstBuf != nullptr && srcBuf != nullptr) {
        size_t bytes = static_cast<size_t>(ext->lengthOf()) * ext->sizeOfT();
        if (bytes > 0) {
          // Stride-aware staging: a non-contiguous VIEW source (e.g. a KV-cache slice
          // past_key = staticKey[:,:,0:N,:]) MUST NOT be raw-copied contiguously — that reads
          // linear bytes OUTSIDE the view (a scatter to staticKey[:,:,N,:]) and skips strided
          // gaps, corrupting the staging buffer the CUDA graph replays from (testViewAliasing
          // MaxDiff bug). Contiguous → raw byte copy (fast); view → stride-aware assign routed
          // onto cudaStr (getCudaStream honors tl_dspGapStream) so it is ordered with replay.
          if (shape::strideDescendingCAscendingF(ext->shapeInfo())) {
            cudaMemcpyAsync(dstBuf, srcBuf, bytes, cudaMemcpyDeviceToDevice, cudaStr);
          } else {
            ScopedDspGapStream gapToCudaStr(cudaStr);
            staging->assign(ext);
          }
          DSP_DIAG(MEMORY,
                   "STAGING_D2D_SLOW[%d]: name='%s' srcBuf=%p dstBuf=%p bytes=%zu "
                   "stream=%p — D2D copy issued",
                   i,
                   (i < static_cast<int>(externalInputNames_.size()))
                       ? externalInputNames_[i].c_str() : "?",
                   srcBuf, dstBuf, bytes, (void*)cudaStr);
        } else {
          DSP_DIAG(MEMORY,
                   "STAGING_D2D_SLOW[%d]: SKIPPED — empty array (bytes=0) srcBuf=%p dstBuf=%p",
                   i, srcBuf, dstBuf);
        }
      } else {
        DSP_DIAG(MEMORY,
                 "STAGING_D2D_SLOW[%d]: SKIPPED — null buffer: srcBuf=%p dstBuf=%p",
                 i, srcBuf, dstBuf);
      }
    }
    staging->dataBuffer()->writeSpecial();

    // ── Shape corruption detector (slow path) ──────────────────────
    if (staging->shapeInfo() != nullptr) {
      LongType sRank = staging->shapeInfo()[0];
      if (sRank < 0 || sRank > SD_MAX_RANK) {
        DSP_DIAG(MEMORY,
                 "STAGING_SHAPE_CORRUPT_AFTER_D2D (slow): ext[%d] rank=%lld (0x%llx) "
                 "shapeInfo=%p specialBuffer=%p primaryBuffer=%p",
                 i, (long long)sRank, (unsigned long long)sRank,
                 (void*)staging->shapeInfo(), staging->specialBuffer(),
                 staging->buffer());
      }
    }

    effectiveExternals_[i] = staging;
  }

  return effectiveExternals_;
}

}  // namespace graph
}  // namespace sd

#endif  // SD_CUDA

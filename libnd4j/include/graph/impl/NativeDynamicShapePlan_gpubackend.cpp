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

// GPU graph backend (Triton/NVRTC/PTX) execution methods.
//
// Contains getGpuGraphBackend() which selects the best available GPU compiler
// backend (Triton > NVRTC > PTX) based on the configured GraphExecutionMode,
// and executeSegmentWithGpuGraph() which drives segment compilation, CUDA graph
// capture/replay for Triton fused kernels, and fallback orchestration.

#include <graph/NativeDynamicShapePlan.h>
#include <graph/DspDiagnostics.h>
#include <graph/DspStreamGuard.h>
#include <graph/DspVerifyUtils.h>
#include <helpers/MmulHelper.h>
#include <system/op_boilerplate.h>
#include <system/Environment.h>
#include <config.h>

// Portable buffer accessor: specialBuffer() on CUDA, buffer() on CPU.
#ifdef SD_CUDA
#define DSP_BUF(arr) ((arr)->specialBuffer())
#else
#define DSP_BUF(arr) ((arr)->buffer())
#endif

#include <algorithm>
#include <chrono>
#include <thread>
#include <cmath>
#include <cstring>
#include <unordered_map>
#include <unordered_set>

// GPU graph backends (conditional)
#if HAVE_TRITON && defined(SD_CUDA)
#include <graph/gpu/TritonGraphBackend.h>
#endif
#ifdef SD_CUDA
#include <graph/gpu/NvrtcGraphBackend.h>
#include <graph/gpu/PtxGraphBackend.h>
#include <graph/gpu/CaptureBufferRegistry.h>
#include <graph/cuda/CudaGraphReplayHandle.h>
#include <memory/cuda/CudaMemoryPool.h>
#endif
#ifdef SD_TPU
#include <graph/tpu/TpuGraphBackend.h>
#endif
#ifdef HAVE_HEXAGON_MLIR
#include <graph/hexagon/HexagonGraphBackend.h>
#endif

namespace sd {
namespace graph {

// Default capture host workspace size for Triton path (32MB, same as non-Triton path).
// Configurable via ND4J_DSP_CAPTURE_HOST_WORKSPACE_MB env var.
#ifdef SD_CUDA
static size_t TRITON_CAPTURE_HOST_WORKSPACE_SIZE = []() -> size_t {
  const char* envVal = std::getenv("ND4J_DSP_CAPTURE_HOST_WORKSPACE_MB");
  size_t mb = 32;
  if (envVal != nullptr) {
    int parsed = std::atoi(envVal);
    if (parsed > 0 && parsed <= 1024) mb = static_cast<size_t>(parsed);
  }
  return mb * 1024ULL * 1024ULL;
}();

// Default capture workspace size for Triton capture buffers (128MB).
// Configurable via ND4J_DSP_CAPTURE_WORKSPACE_MB env var.
static size_t TRITON_CAPTURE_WORKSPACE_SIZE = []() -> size_t {
  const char* envVal = std::getenv("ND4J_DSP_CAPTURE_WORKSPACE_MB");
  size_t mb = 128;
  if (envVal != nullptr) {
    int parsed = std::atoi(envVal);
    if (parsed > 0 && parsed <= 4096) mb = static_cast<size_t>(parsed);
  }
  return mb * 1024ULL * 1024ULL;
}();
#endif

// Local helper: convert Status enum to human-readable string for diagnostics.
static const char* statusName_gpu(Status status) {
  switch (status) {
    case Status::OK: return "OK";
    case Status::BAD_INPUT: return "BAD_INPUT";
    case Status::BAD_SHAPE: return "BAD_SHAPE";
    case Status::BAD_RANK: return "BAD_RANK";
    case Status::BAD_PARAMS: return "BAD_PARAMS";
    case Status::BAD_OUTPUT: return "BAD_OUTPUT";
    case Status::BAD_RNG: return "BAD_RNG";
    case Status::BAD_EPSILON: return "BAD_EPSILON";
    case Status::BAD_GRADIENTS: return "BAD_GRADIENTS";
    case Status::BAD_BIAS: return "BAD_BIAS";
    case Status::VALIDATION: return "VALIDATION";
    case Status::BAD_GRAPH: return "BAD_GRAPH";
    case Status::BAD_LENGTH: return "BAD_LENGTH";
    case Status::BAD_DIMENSIONS: return "BAD_DIMENSIONS";
    case Status::BAD_ORDER: return "BAD_ORDER";
    case Status::BAD_ARGUMENTS: return "BAD_ARGUMENTS";
    case Status::DOUBLE_WRITE: return "DOUBLE_WRITE";
    case Status::DOUBLE_READ: return "DOUBLE_READ";
    case Status::KERNEL_FAILURE: return "KERNEL_FAILURE";
    case Status::EQ_TRUE: return "EQ_TRUE";
    case Status::EQ_FALSE: return "EQ_FALSE";
    case Status::MAYBE: return "MAYBE";
    default: return "UNKNOWN";
  }
}

// Helper: extract specialBuffer() device addresses from NDArray** into void** for
// address snapshot diagnostics. Thread-local to avoid repeated allocation.
static void extractDeviceAddrs(NDArray** arrays, int count, std::vector<void*>& out) {
  out.resize(count);
  for (int i = 0; i < count; i++) {
    out[i] = (arrays != nullptr && arrays[i] != nullptr)
             ? DSP_BUF(arrays[i]) : nullptr;
  }
}

/**
 * Compute FNV-1a hash of slot output specialBuffer() addresses for a segment.
 * Used to verify that output buffers haven't been reallocated between capture
 * and replay — stale addresses in a CUDA graph cause SIGSEGV or corruption.
 */
static LongType computeSlotAddrHash(NDArray** outputSlots, int startSlot, int endSlot, int totalSlots) {
  LongType hash = 0xcbf29ce484222325ULL;  // FNV-1a offset basis
  for (int si = startSlot; si <= endSlot && si < totalSlots; si++) {
    void* addr = (outputSlots[si] != nullptr) ? DSP_BUF(outputSlots[si]) : nullptr;
    LongType bits = reinterpret_cast<uintptr_t>(addr);
    hash ^= bits;
    hash *= 0x100000001b3ULL;  // FNV-1a prime
  }
  return hash;
}

#ifdef SD_CUDA
static bool isCurrentDevicePointer(void* ptr, int currentDeviceId) {
  if (ptr == nullptr) return false;

  cudaPointerAttributes attrs;
  auto res = cudaPointerGetAttributes(&attrs, ptr);
  if (res != cudaSuccess) {
    cudaGetLastError();
    return false;
  }

  return attrs.type == cudaMemoryTypeDevice && attrs.device == currentDeviceId;
}
#else
static bool isCurrentDevicePointer(void* /*ptr*/, int /*currentDeviceId*/) {
  return false;
}
#endif

// ── GPU CONTEXT PROBE ──────────────────────────────────────────────────────
// Shared helper that dumps multi-device memory state + CUDA context health.
// Called by all error handlers to detect downstream/pre-existing errors.
#ifdef SD_CUDA
static void dumpGpuContextState(int failedDeviceId, const char* errorType) {
  // 1. Check for pre-existing CUDA error (downstream error detection)
  cudaError_t preExisting = cudaPeekAtLastError();
  if (preExisting != cudaSuccess) {
    DSP_DIAG(MEMORY, "%s: PRE-EXISTING CUDA ERROR on device %d: %d (%s) — "
             "this may be a downstream error from a previous operation",
             errorType, failedDeviceId,
             static_cast<int>(preExisting), cudaGetErrorString(preExisting));
  }

  // 2. Failed device: full detail
  {
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, failedDeviceId);
    size_t gpuFree = 0, gpuTotal = 0;
    cudaSetDevice(failedDeviceId);
    cudaMemGetInfo(&gpuFree, &gpuTotal);
    DSP_DIAG(MEMORY, "%s: device %d '%s' (cc=%d.%d): free=%zuMB total=%zuMB used=%zuMB "
             "multiProcessorCount=%d",
             errorType, failedDeviceId, props.name,
             props.major, props.minor,
             gpuFree / (1024*1024), gpuTotal / (1024*1024),
             (gpuTotal - gpuFree) / (1024*1024),
             props.multiProcessorCount);
  }

  // 3. Other devices: one-line summary each
  int deviceCount = 0;
  cudaGetDeviceCount(&deviceCount);
  for (int d = 0; d < deviceCount; d++) {
    if (d == failedDeviceId) continue;
    cudaSetDevice(d);
    size_t otherFree = 0, otherTotal = 0;
    cudaMemGetInfo(&otherFree, &otherTotal);
    cudaError_t otherErr = cudaPeekAtLastError();
    DSP_DIAG(MEMORY, "%s: device %d: free=%zuMB total=%zuMB%s",
             errorType, d, otherFree / (1024*1024), otherTotal / (1024*1024),
             otherErr != cudaSuccess ?
               (std::string(" CUDA_ERROR=") + cudaGetErrorString(otherErr)).c_str() : "");
  }

  // 4. Restore original device
  cudaSetDevice(failedDeviceId);

  // 5. Report graph execution active state
  DSP_DIAG(MEMORY, "%s: tl_graphExecutionActive=%d",
           errorType, tl_graphExecutionActive ? 1 : 0);
}

// ── DISTINCT ERROR HANDLERS ────────────────────────────────────────────────

static Status reportOomError(GraphSegment& seg, const char* phase,
                             size_t requestedBytes, int deviceId) {
  dumpGpuContextState(deviceId, "OOM");
  size_t gpuFree = 0, gpuTotal = 0;
  cudaMemGetInfo(&gpuFree, &gpuTotal);
  DSP_DIAG(MEMORY,
    "OOM ERROR in seg[%d-%d] during '%s' on device %d: "
    "requested=%zuMB gpuFree=%zuMB gpuTotal=%zuMB gpuUsed=%zuMB "
    "executionCount=%d phase=%d",
    seg.startSlot, seg.endSlot, phase, deviceId,
    requestedBytes / (1024*1024), gpuFree / (1024*1024),
    gpuTotal / (1024*1024), (gpuTotal - gpuFree) / (1024*1024),
    seg.exec.executionCount, static_cast<int>(seg.exec.currentPhase));
  seg.exec.compilationFailed = true;
  return Status::KERNEL_FAILURE;
}

static Status reportCaptureError(GraphSegment& seg, const char* step,
                                 cudaError_t cudaErr, int deviceId) {
  dumpGpuContextState(deviceId, "CAPTURE");
  size_t gpuFree = 0, gpuTotal = 0;
  cudaMemGetInfo(&gpuFree, &gpuTotal);
  DSP_DIAG(EXECUTE,
    "CAPTURE ERROR in seg[%d-%d] at step '%s' on device %d: "
    "cudaError=%d (%s) gpuFree=%zuMB gpuTotal=%zuMB "
    "executionCount=%d numOps=%d compiledBy='%s'",
    seg.startSlot, seg.endSlot, step, deviceId,
    static_cast<int>(cudaErr), cudaGetErrorString(cudaErr),
    gpuFree / (1024*1024), gpuTotal / (1024*1024),
    seg.exec.executionCount, seg.endSlot - seg.startSlot + 1,
    seg.exec.compiledByBackend.c_str());
  seg.exec.compilationFailed = true;
  cudaGetLastError(); // clear error state
  return Status::KERNEL_FAILURE;
}

static Status reportReplayError(GraphSegment& seg, const char* step,
                                cudaError_t cudaErr, int deviceId) {
  dumpGpuContextState(deviceId, "REPLAY");
  size_t gpuFree = 0, gpuTotal = 0;
  cudaMemGetInfo(&gpuFree, &gpuTotal);
  DSP_DIAG(EXECUTE,
    "REPLAY ERROR in seg[%d-%d] at step '%s' on device %d: "
    "cudaError=%d (%s) gpuFree=%zuMB gpuTotal=%zuMB "
    "executionCount=%d hasReplayHandle=%d",
    seg.startSlot, seg.endSlot, step, deviceId,
    static_cast<int>(cudaErr), cudaGetErrorString(cudaErr),
    gpuFree / (1024*1024), gpuTotal / (1024*1024),
    seg.exec.executionCount,
    seg.exec.replayHandle != nullptr ? 1 : 0);
  seg.exec.compilationFailed = true;
  cudaGetLastError(); // clear error state
  return Status::KERNEL_FAILURE;
}

// ── LRU GRAPH EVICTION ──────────────────────────────────────────────────────
// Evicts captured graphs to free GPU memory. Returns number of graphs evicted.
// When dspLruEviction is true, evicts least-recently-replayed graphs first.
// Otherwise evicts smallest (fewest nodes) first (legacy behavior).
int NativeDynamicShapePlan::evictLruGraphs(int segIdx, size_t neededBytes, void* stream) {
  auto cudaStr = (stream != nullptr) ? *static_cast<cudaStream_t*>(stream) : nullptr;
  bool usePool = Environment::getInstance().dspCapturePoolEnabled() &&
                 captureBufferRegistry_ != nullptr;
  bool lruMode = Environment::getInstance().dspLruEviction();
  int maxEvictions = Environment::getInstance().dspCaptureOomMaxRetries();
  int numEvicted = 0;

  for (int evictAttempt = 0; evictAttempt < maxEvictions; evictAttempt++) {
    // Check if we have enough memory already
    size_t gpuFree = 0, gpuTotal = 0;
    cudaMemGetInfo(&gpuFree, &gpuTotal);
    if (gpuFree >= neededBytes) {
      DSP_DIAG(MEMORY, "evictLruGraphs: have enough memory after %d evictions (%zuMB free >= %zuMB needed)",
               numEvicted, gpuFree / (1024*1024), neededBytes / (1024*1024));
      break;
    }

    // Find the best candidate to evict
    int evictIdx = -1;
    if (lruMode) {
      // LRU: find segment with smallest lastReplayExecCount (least recently used)
      int lruExecCount = INT_MAX;
      for (size_t si = 0; si < segments_.size(); si++) {
        if (static_cast<int>(si) == segIdx) continue;
        auto& candidate = segments_[si];
        if (!candidate.exec.replayHandle || !candidate.exec.replayHandle->isReady()) continue;
        if (candidate.exec.lastReplayExecCount < lruExecCount) {
          lruExecCount = candidate.exec.lastReplayExecCount;
          evictIdx = static_cast<int>(si);
        }
      }
    } else {
      // Smallest-first: find segment with fewest CUDA graph nodes
      size_t smallestNodes = SIZE_MAX;
      for (size_t si = 0; si < segments_.size(); si++) {
        if (static_cast<int>(si) == segIdx) continue;
        auto& candidate = segments_[si];
        if (!candidate.exec.replayHandle || !candidate.exec.replayHandle->isReady()) continue;
        auto* cudaReplay = dynamic_cast<CudaGraphReplayHandle*>(candidate.exec.replayHandle.get());
        size_t nodeCount = cudaReplay ? cudaReplay->getNumNodes() : 1;
        if (nodeCount == 0) nodeCount = 1;
        if (nodeCount < smallestNodes) {
          smallestNodes = nodeCount;
          evictIdx = static_cast<int>(si);
        }
      }
    }

    if (evictIdx < 0) {
      DSP_DIAG(MEMORY, "evictLruGraphs: no more evictable segments (evicted %d)", numEvicted);
      break;
    }

    // Evict the selected segment
    auto& evictSeg = segments_[evictIdx];
    DSP_DIAG(MEMORY, "evictLruGraphs: evicting seg[%d-%d] (lruExec=%d, mode=%s) for seg idx=%d (attempt %d/%d)",
             evictSeg.startSlot, evictSeg.endSlot, evictSeg.exec.lastReplayExecCount,
             lruMode ? "LRU" : "smallest", segIdx, evictAttempt + 1, maxEvictions);

    // Free capture buffer NDArrays
    for (auto& cb : evictSeg.exec.replayHandle->getCaptureBuffers()) {
      if (!cb.directReference) delete cb.buffer;
    }
    evictSeg.exec.replayHandle->getCaptureBuffers().clear();

    // Release capture workspace
    evictSeg.exec.replayHandle->releaseWorkspace(
        usePool ? captureBufferRegistry_ : nullptr,
        evictSeg.startSlot);

    // Free pinned host pointers
    evictSeg.exec.replayHandle->freeHostPointers();
    evictSeg.exec.replayHandle->clearExternalAddresses();

    // Destroy replay handle (frees cudaGraphExec + cudaGraph)
    evictSeg.exec.replayHandle.reset();

    // Reset evicted segment for future re-capture
    evictSeg.exec.cachedShapeKey = 0;
    evictSeg.exec.capturedInputAddrKey = 0;
    evictSeg.exec.capturedCreateValueKey = 0;
    evictSeg.exec.compilationFailed = false;
    evictSeg.exec.gapOpsCapturedInGraph = false;
    evictSeg.exec.argTableStable = false;
    evictSeg.exec.compiledByBackend.clear();
    evictSeg.exec.executionCount = 0;
    evictSeg.exec.lastReplayExecCount = 0;

    numEvicted++;

    // Sync to ensure GPU memory is freed
    if (cudaStr != nullptr) {
      cudaStreamSynchronize(cudaStr);
    }
    cudaGetLastError();
  }

  // Final pool trim after evictions
  if (numEvicted > 0) {
    int deviceId = 0;
    cudaGetDevice(&deviceId);
    memory::CudaMemoryPool::getInstance().trimPool(deviceId);
    DSP_DIAG(MEMORY, "evictLruGraphs: evicted %d segments, trimmed pool on device %d", numEvicted, deviceId);
  }

  return numEvicted;
}

// ── PROACTIVE PRE-CAPTURE MEMORY CLEANUP ───────────────────────────────────
// Called before workspace allocation when about to capture a graph.
// Frees cached-but-unused GPU memory and evicts LRU graphs if needed.
void NativeDynamicShapePlan::proactivePreCaptureMemoryCleanup(GraphSegment& seg, int segIdx, void* stream) {
  auto cudaStr = (stream != nullptr) ? *static_cast<cudaStream_t*>(stream) : nullptr;
  int deviceId = 0;
  cudaGetDevice(&deviceId);

  // 1. Trim CUDA memory pool — cheap, can reclaim hundreds of MB
  DSP_DIAG(MEMORY, "proactive cleanup: trimming pool on device %d for seg[%d-%d]",
           deviceId, seg.startSlot, seg.endSlot);
  memory::CudaMemoryPool::getInstance().trimPool(deviceId);
  if (cudaStr != nullptr) {
    memory::CudaMemoryPool::getInstance().trimPoolOnStream(deviceId, cudaStr);
  }

  // 2. Check if we have enough memory
  size_t gpuFree = 0, gpuTotal = 0;
  cudaMemGetInfo(&gpuFree, &gpuTotal);

  // Estimate needed: capture workspace + cuBLAS workspace (if not allocated) + safety margin
  size_t neededBytes = 0;
  if (sharedCaptureWorkspace_ == nullptr) {
    neededBytes += TRITON_CAPTURE_WORKSPACE_SIZE;  // 128MB default
  }
  if (cublasWorkspaceBuffer_ == nullptr) {
    neededBytes += Environment::getInstance().dspCublasWorkspaceMb() * 1024ULL * 1024ULL;
  }
  neededBytes += Environment::getInstance().dspGraphMetadataSafetyMb() * 1024ULL * 1024ULL;

  DSP_DIAG(MEMORY, "proactive cleanup: gpuFree=%zuMB, needed=%zuMB (ws=%zuMB, cublas=%zuMB, safety=%dMB) for seg[%d-%d]",
           gpuFree / (1024*1024), neededBytes / (1024*1024),
           (sharedCaptureWorkspace_ == nullptr ? TRITON_CAPTURE_WORKSPACE_SIZE : 0) / (1024*1024),
           (cublasWorkspaceBuffer_ == nullptr ? (size_t)(Environment::getInstance().dspCublasWorkspaceMb()) : 0),
           Environment::getInstance().dspGraphMetadataSafetyMb(),
           seg.startSlot, seg.endSlot);

  if (gpuFree >= neededBytes) {
    DSP_DIAG(MEMORY, "proactive cleanup: sufficient memory (%zuMB >= %zuMB), no eviction needed",
             gpuFree / (1024*1024), neededBytes / (1024*1024));
    return;
  }

  // 3. LRU eviction
  DSP_DIAG(MEMORY, "proactive cleanup: insufficient memory (%zuMB < %zuMB), starting LRU eviction",
           gpuFree / (1024*1024), neededBytes / (1024*1024));
  int numEvicted = evictLruGraphs(segIdx, neededBytes, stream);

  // 4. Final trim after evictions
  if (numEvicted > 0) {
    memory::CudaMemoryPool::getInstance().trimPool(deviceId);
  }

  // Log final state
  cudaMemGetInfo(&gpuFree, &gpuTotal);
  DSP_DIAG(MEMORY, "proactive cleanup complete: evicted=%d, gpuFree=%zuMB/%zuMB for seg[%d-%d]",
           numEvicted, gpuFree / (1024*1024), gpuTotal / (1024*1024),
           seg.startSlot, seg.endSlot);
}

#endif  // SD_CUDA

// Capture buffers are dense fixed-address staging arrays. We only use a raw D2D
// memcpy for the narrow case where both source and destination already resolve to
// device-local pointers on the current GPU. Anything view-backed, host-mirrored,
// or cross-device falls back to assign() so the normal prepare/register/migration
// path preserves tensor semantics and multi-device handling.
#ifdef SD_CUDA
static void copyIntoCaptureBuffer(NDArray* dst, NDArray* src, cudaStream_t cudaStr,
                                  bool mirrorHost, const char* kind, int index,
                                  int segStart, int segEnd) {
  if (dst == nullptr || src == nullptr) return;

  const size_t srcBytes = src->lengthOf() * src->sizeOfT();
  if (srcBytes == 0) return;

  void* srcSpecial = DSP_BUF(src);
  void* dstSpecial = DSP_BUF(dst);

  // REPLAY OPTIMIZATION: Fast path for contiguous arrays (including views with EWS=1).
  // In frozen replay, most sources are views (reshape/slice output) with EWS=1.
  // The original code called isCurrentDevicePointer() (cudaPointerGetAttributes, ~5-10us)
  // on both src and dst, and fell through to assign() + synchronize() (BLOCKING SYNC)
  // for views. For contiguous views, raw cudaMemcpyAsync is safe and avoids both
  // the expensive CUDA API query AND the blocking synchronization.
  //
  // Condition: src has EWS=1 (contiguous in memory), not mirrorHost, and both
  // device pointers are non-null. This covers 90%+ of replay D2D copies.
  if (!mirrorHost && srcSpecial != nullptr && dstSpecial != nullptr && src->ews() == 1) {
    cudaMemcpyAsync(dstSpecial, srcSpecial, srcBytes, cudaMemcpyDeviceToDevice, cudaStr);
    dst->dataBuffer()->writeSpecial();
    return;
  }

  const int currentDeviceId = AffinityManager::currentDeviceId();
  const bool canUseRawDeviceCopy = !mirrorHost && !src->isView() &&
                                   isCurrentDevicePointer(srcSpecial, currentDeviceId) &&
                                   isCurrentDevicePointer(dstSpecial, currentDeviceId);

  if (!canUseRawDeviceCopy) {
    DSP_DIAG(EXECUTE,
             "CAPTURE_BUFFER_LOGICAL_COPY: %s#%d seg[%d-%d] len=%lld ews=%lld order=%c mirrorHost=%d rawSafe=%d",
             kind, index, segStart, segEnd,
             static_cast<long long>(src->lengthOf()),
             static_cast<long long>(src->ews()),
             src->ordering(),
             mirrorHost ? 1 : 0,
             canUseRawDeviceCopy ? 1 : 0);

    dst->assign(src);
    // assign() dispatches on the source array's LaunchContext stream.
    src->synchronize("NativeDynamicShapePlan capture buffer logical copy");

    if (mirrorHost) {
      dst->syncToHost();
    }
    return;
  }

  if (srcSpecial != nullptr && dstSpecial != nullptr) {
    cudaMemcpyAsync(dstSpecial, srcSpecial, srcBytes, cudaMemcpyDeviceToDevice, cudaStr);
    // Keep device actuality in sync with the fresh D2D copy so no later code
    // bakes a stale H2D sync into the graph.
    dst->dataBuffer()->writeSpecial();
  }
}
#else
static void copyIntoCaptureBuffer(NDArray* dst, NDArray* src, void* /*cudaStr*/,
                                  bool mirrorHost, const char* /*kind*/, int /*index*/,
                                  int /*segStart*/, int /*segEnd*/) {
  if (dst == nullptr || src == nullptr) return;
  dst->assign(src);
  if (mirrorHost) {
    dst->syncToHost();
  }
}
#endif

// Strict mode: fail fast instead of silently degrading to slot-by-slot.
static bool isStrictNoFallbackMode_gpu(GraphExecutionMode mode) {
  return mode == GraphExecutionMode::GEM_TRITON;
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

#ifdef SD_CUDA
// Templated helpers in DspVerifyUtils.h (dspVerifyCopyValues, dspMaxDiff, dspFormatValues, etc.)
#endif  // SD_CUDA

void NativeDynamicShapePlan::clearGpuBackendFailedCache() {
#if HAVE_TRITON && defined(SD_CUDA)
  TritonGraphBackend::getInstance().clearFailedSegmentCache();
#endif
}

GraphBackend* NativeDynamicShapePlan::getGpuGraphBackend() {
  if (gpuGraphBackendChecked_) return gpuGraphBackend_;
  gpuGraphBackendChecked_ = true;

  // If a specific backend is forced via setGraphExecutionMode(), use only that one.
  // SLOT_BY_SLOT and graph-replay-only modes don't use a GPU compiler backend —
  // they rely on the GraphReplayHandle (CUDA/HIP/L0/Vulkan/Metal) for capture/replay.
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

Status NativeDynamicShapePlan::executeSegmentWithGpuGraph(
    GraphSegment& seg, NDArray** externalArrays, int numExt, void* stream) {

  // Derive segIdx for proactive eviction and OOM retry.
  int segIdx = -1;
  for (int si = 0; si < static_cast<int>(segments_.size()); si++) {
    if (&segments_[si] == &seg) { segIdx = si; break; }
  }

  {
    const char* mode = "unknown";
    if (seg.exec.executionCount == 0) mode = "warmup";
    else if (seg.exec.replayHandle != nullptr && seg.exec.replayHandle->isReady()) mode = "replay";
    else if (seg.exec.compilationFailed) mode = "slot-by-slot(failed)";
    else if (seg.exec.executionCount >= 1) mode = "capture-candidate";
    DSP_DIAG_SEG(EXECUTE, seg.startSlot,
                 "executeSegmentWithGpuGraph: ENTER seg[%d-%d] mode=%s execCount=%d capturable=%d",
                 seg.startSlot, seg.endSlot, mode, seg.exec.executionCount, seg.isCapturable ? 1 : 0);
  }

#ifdef SD_CUDA
  // ── Segment lifecycle: SEG_ENTER ──────────────────────────────────────
  if (Environment::getInstance().tritonVerifyKernels()) {
    // Ensure VERIFY diagnostic category is enabled and output level is FULL
    // when tritonVerifyKernels is on (may be set at runtime via Java, after
    // DspDiagnostics constructor)
    if (!DSP_DIAG_ENABLED(VERIFY)) {
      sd::graph::DspDiagnostics::getInstance().enableCategories(sd::graph::DSP_DIAG_VERIFY);
      sd::graph::DspDiagnostics::getInstance().setLevel(sd::graph::DSP_LEVEL_FULL);
    }
    const char* mode = "unknown";
    if (seg.exec.executionCount == 0) mode = "warmup";
    else if (seg.exec.executionCount == 1) mode = "compile";
    else if (seg.exec.replayHandle != nullptr) mode = "replay";
    else if (seg.exec.compilationFailed) mode = "slot-by-slot";
    else mode = "capture";
    DSP_DIAG(VERIFY, "SEG_ENTER seg[%d-%d] execCount=%d mode=%s",
              seg.startSlot, seg.endSlot, seg.exec.executionCount, mode);
    // Dump external input actuality flags for first N inputs
    int dumpCount = std::min(numExt, 8);
    for (int i = 0; i < dumpCount; i++) {
      if (externalArrays[i] != nullptr && externalArrays[i]->dataBuffer() != nullptr) {
        auto* db = externalArrays[i]->dataBuffer();
        DSP_DIAG(VERIFY, "  EXT_INPUT[%d] dtype=%s len=%lld pAct=%d sAct=%d addr=%p",
                  i, DataTypeUtils::asString(externalArrays[i]->dataType()).c_str(),
                  (long long)externalArrays[i]->lengthOf(),
                  db->isPrimaryActual() ? 1 : 0, db->isSpecialActual() ? 1 : 0,
                  DSP_BUF(externalArrays[i]));
      }
    }
    if (numExt > 8) {
      DSP_DIAG(VERIFY, "  ... and %d more external inputs", numExt - 8);
    }
  }
#endif

  auto* backend = getGpuGraphBackend();
  if (backend == nullptr) {
    DSP_DIAG(BACKEND, "executeSegmentWithGpuGraph: no GPU backend selected for seg[%d-%d]",
              seg.startSlot, seg.endSlot);
    return Status::KERNEL_FAILURE;
  }
  const char* backendName = backend->name();

  // If compilation previously failed validation, never try again
  if (seg.exec.compilationFailed) {
    return Status::KERNEL_FAILURE;
  }

  // Check if this segment can be compiled by the selected GPU backend
  if (!backend->canFuseSegment(slots_, seg.startSlot, seg.endSlot)) {
    DSP_DIAG(BACKEND, "executeSegmentWithGpuGraph: backend=%s cannot fuse seg[%d-%d]",
              backendName, seg.startSlot, seg.endSlot);
    return Status::KERNEL_FAILURE;  // Caller will fall back to CUDA Graphs
  }

  // First execution: run slot-by-slot warmup BEFORE compilation.
  if (seg.exec.executionCount == 0) {
#ifdef SD_CUDA
    // ── Plan structure dump (one-time, on first segment execution) ─────────
    if (Environment::getInstance().tritonVerifyKernels()) {
      DSP_DIAG(VERIFY, "=== PLAN STRUCTURE ===");
      DSP_DIAG(VERIFY, "Plan: %d steps, %d output slots, %d external inputs, %d segments",
                numSlots_, totalOutputSlots_, numExternalInputs_, (int)segments_.size());
      for (int si = 0; si < (int)segments_.size(); si++) {
        auto& s = segments_[si];
        DSP_DIAG(VERIFY, "Segment %d: slots [%d..%d] (%d ops) %s",
                  si, s.startSlot, s.endSlot, s.endSlot - s.startSlot + 1,
                  s.isCapturable ? "capturable" : "non-capturable");
      }
      // Per-step wiring
      std::unordered_map<std::string, int> opHistogram;
      for (int s = 0; s < numSlots_; s++) {
        auto& sl = slots_[s];
        opHistogram[sl.opName]++;
        // Build input description
        std::string inputsStr;
        for (int i = 0; i < sl.numInputs; i++) {
          if (i > 0) inputsStr += ", ";
          int srcIdx = sl.inputSourceIndices[i];
          if (srcIdx >= 0) {
            inputsStr += "slot#" + std::to_string(srcIdx);
          } else {
            int extIdx = -(srcIdx + 1);
            inputsStr += "ext#" + std::to_string(extIdx);
            if (extIdx < (int)externalInputNames_.size() && !externalInputNames_[extIdx].empty()) {
              inputsStr += ":\"" + externalInputNames_[extIdx] + "\"";
            }
            if (sl.inputSourceTypes != nullptr) {
              inputsStr += ":";
              inputsStr += sourceTypeName(sl.inputSourceTypes[i]);
            }
          }
        }
        // Build output description
        std::string outputsStr;
        for (int i = 0; i < sl.numOutputs; i++) {
          if (i > 0) outputsStr += ",";
          outputsStr += std::to_string(sl.outputSlotIndices[i]);
        }
        DSP_DIAG(VERIFY, "STEP %4d: %-20s inputs:[%s] -> outputs:[%s]%s%s%s",
                  s, sl.opName.c_str(), inputsStr.c_str(), outputsStr.c_str(),
                  sl.isIdentityOp ? " [IDENTITY]" : "",
                  sl.frozenConstantSlot() ? " [FROZEN]" : "",
                  sl.isFusedChainTail ? " [FUSED_TAIL]" : "");
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
#endif

    auto warmupStatus = executeSegmentSlotBySlot(seg, externalArrays, numExt, stream);
    // NOTE: executeSegmentSlotBySlot already increments seg.exec.executionCount on OK
    // (NativeDynamicShapePlan_segments.cpp line 930). Do NOT increment again here
    // — double-increment causes executionCount to skip the capture window [0,2],
    // preventing CUDA graph capture entirely and causing OOM from leaked per-step
    // allocations.

    // When shapes are frozen and executionCount is 1, the next call would
    // trigger compilation (needsCompile = executionCount==1). If compilation
    // already succeeded during unfrozen execution, skip recompilation by
    // bumping executionCount to 2. But DON'T skip if compilation hasn't
    // happened yet — let it trigger on the next call so cross-segment
    // shapes (now backfilled) are available for the Triton IR builder.
    if (shapesFrozen_ && warmupStatus == Status::OK && seg.exec.executionCount == 1
        && !Environment::getInstance().dspFreezeRecompile()) {
      // Only skip recompilation if segment already has compiled kernels.
      // seg.shapeKey != 0 means compilation ran previously and cached the key.
      if (seg.shapeKey != 0) {
        seg.exec.executionCount = 2;
        seg.exec.cachedShapeKey = seg.shapeKey;
        DSP_DIAG(COMPILE, "Post-freeze warmup: skipping recompile for seg[%d-%d] "
                  "(already compiled, shapeKey=%lld, bumped executionCount to 2)",
                  seg.startSlot, seg.endSlot, seg.shapeKey);
      } else {
        // Segment was never compiled — let executionCount stay at 1 so the
        // next call triggers compilation with backfilled cross-segment shapes.
        // DO NOT set seg.shapeKey here — it must stay 0 so the next call's
        // shapeKey check correctly identifies this as "never compiled".
        DSP_DIAG(COMPILE, "Post-freeze warmup: NOT skipping compile for seg[%d-%d] "
                  "(never compiled, executionCount stays at 1)",
                  seg.startSlot, seg.endSlot);
      }
    }
    return warmupStatus;
  }

  // Compute shape key for cache lookup.
  // When shapes are frozen and the key was already computed, reuse it — the shapes
  // cannot change so the hash is stable. Saves iterating all cross-segment inputs.
  // EXCEPTION: segments with value-dependent ops must ALWAYS recompute the shape key
  // because input VALUES (hashed by computeSegmentShapeKey for small inputs ≤32 elements)
  // can change even when shapes are frozen. Without this guard, the cached key would
  // miss value changes in reshape targets, broadcast dims, etc., causing CUDA graph
  // replay with stale output shapes.
  //
  // REPLAY OPTIMIZATION: During stable replay (executionCount >= 3 with a valid replay
  // handle), skip shape key computation entirely — even for hasValueDepOps segments.
  // The shape key was validated at capture time. Value-dependent inputs (reshape targets,
  // broadcast dims) are handled by capture buffer D2D refresh, not by shape key changes.
  // If a value change truly requires graph invalidation, the createValueKey mechanism
  // catches it. Skipping shape key here eliminates N syncToHost calls per step
  // (one per small INT/INT64 cross-segment input array).
  LongType segShapeKey;
  bool isStableReplay = shapesFrozen_ && seg.exec.executionCount >= 3 &&
                         seg.exec.replayHandle != nullptr && seg.exec.replayHandle->isReady() &&
                         seg.exec.cachedShapeKey != 0;
  if (isStableReplay) {
    segShapeKey = seg.exec.cachedShapeKey;
  } else if (shapesFrozen_ && seg.exec.cachedShapeKey != 0 && !seg.hasValueDepOps) {
    segShapeKey = seg.exec.cachedShapeKey;
  } else {
    segShapeKey = computeSegmentShapeKey(seg, externalArrays, numExt);
  }

  // Diagnostic: scan all slotArrayCache_ entries for freed DataBuffers.
  // Java may have closed DSP output arrays between steps (e.g., prefill KV outputs via
  // setCloseable(true)+close()), deleting the C++ NDArray and leaving dangling pointers.
  //
  // Always run this scan: during warmup/transitions it handles invalidation gracefully;
  // after warmup with frozen shapes, stale buffers indicate a bug (hard error via REQUIRE_TRUE).
  //
  // REPLAY OPTIMIZATION: Skip during stable replay (executionCount >= 4). In frozen
  // replay, arrays persist and are never closed by Java. The scan iterates all slots
  // in the segment range + all external inputs (~1333). For 278 captured segments,
  // this is significant host-side iteration. After the first few replays validate
  // no stale entries exist, skip the scan.
  if (seg.exec.executionCount < 4 || !isStableReplay) {
    int invalidCount = 0;
    for (int si = seg.startSlot; si <= seg.endSlot && si < totalOutputSlots_; si++) {
      NDArray* cached = slotArrayCache_[si];
      if (cached != nullptr && !cached->isEmpty()) {
        auto* db = cached->dataBuffer();
        if (db == nullptr || !db->isValid()) {
          DSP_DIAG_SLOT(MEMORY, si, "STALE slotArrayCache_[%d] detected "
                    "(arr=%p, db=%p, dbValid=%d, frozenConst=%d). Invalidating.",
                    si, (void*)cached, (void*)db, db ? (db->isValid() ? 1 : 0) : -1,
                    slots_[si].frozenConstantSlot() ? 1 : 0);
          slotArrayCache_[si] = nullptr;
          if (outputSlots_[si] == cached) outputSlots_[si] = nullptr;
          if (si < numSlots_ && slots_[si].state_ == NativeSlot::SlotState::FROZEN_CONSTANT) {
            slots_[si].state_ = NativeSlot::SlotState::FROZEN;
          }
          invalidCount++;
        }
      }
    }
    for (int ei = 0; ei < numExt; ei++) {
      NDArray* ext = externalArrays[ei];
      if (ext != nullptr && !ext->isEmpty()) {
        auto* db = ext->dataBuffer();
        if (db == nullptr || !db->isValid()) {
          DSP_DIAG(MEMORY, "STALE externalInput[%d] detected "
                    "(arr=%p, db=%p, dbValid=%d)",
                    ei, (void*)ext, (void*)db, db ? (db->isValid() ? 1 : 0) : -1);
          invalidCount++;
        }
      }
    }
    if (invalidCount > 0) {
      DSP_DIAG(MEMORY, "executeSegmentWithGpuGraph: found %d stale entries in slot/external arrays",
                invalidCount);
      if (shapesFrozen_ && seg.exec.executionCount > 1) {
        // After warmup with frozen shapes, stale buffers mean a bug in array lifecycle management
        REQUIRE_TRUE(false, 0, "Stale buffer detected after warmup (executionCount=%d, frozen=%d, "
                     "invalidCount=%d) in seg[%d-%d]. This indicates a bug in DSP array persistence.",
                     seg.exec.executionCount, (int)shapesFrozen_, invalidCount,
                     seg.startSlot, seg.endSlot);
      }
      // During warmup/transitions, invalidate and re-execute
#ifdef SD_CUDA
      platformCleanupSegmentForRebuild(seg);
      seg.exec.argTableStable = false;
      batchD2DCount_ = 0;
      seg.exec.cachedShapeKey = 0;
#endif
      seg.exec.compilationFailed = false;
      DSP_DIAG(FALLBACK, "invalidated graph for seg[%d-%d] "
                "due to %d stale entries - executing slot-by-slot this step",
                seg.startSlot, seg.endSlot, invalidCount);
      return executeSegmentSlotBySlot(seg, externalArrays, numExt, stream);
    }
  }

  // Pre-execution: ensure all output slots in the segment have live arrays.
  // The Triton kernel's arg mapping references outputSlots_ for both inputs
  // (from prior ops) and outputs (to write results). Slot-by-slot warmup may
  // have released intermediate arrays via releaseAtStep_, leaving entries null.
  // First restore from slotArrayCache_, then allocate any remaining nulls
  // using cached shape info from warmup.
  //
  //  This MUST happen BEFORE compilation. The compiler resolves
  // arg mappings from outputSlots_ — if intermediate slots are null (released
  // after warmup), the compiler omits them from the arg table, producing
  // sub-kernels with missing inputs that read stale/garbage data on first
  // execution. By populating all slots before compilation, the compiler sees
  // all arrays and builds correct arg mappings.
  //
  // IMPORTANT: Java may close() output arrays between execution steps (e.g.,
  // prefill KV outputs via setCloseable(true)+close()). This frees the underlying
  // DataBuffer while slotArrayCache_ still holds the NDArray*. Validate the
  // DataBuffer before reusing — invalidate entries pointing to freed buffers.
  //
  //  If any output slot within the segment is allocated at a NEW address
  // (different from capture time), the cached CUDA graph becomes invalid. Triton
  // arg tables are refreshed with new addresses, but native ops (cuBLAS matmul)
  // have addresses baked into the graph. This address inconsistency causes the
  // graph to read stale data from old addresses while Triton writes to new ones.
  // Track any new allocations and invalidate the graph if needed.
  //
  // OPTIMIZATION: Skip when shapes are frozen and we've already done this
  // restoration at least once (executionCount > 2). In steady-state decode,
  // outputSlots_ are stable — no arrays are released or freed between steps.
  // EXCEPTION: segments entering capture for the first time (no replay handle)
  // MUST always get pre-exec restoration — cleanup may have nulled cross-segment
  // input slots that the capture path needs for capture buffer initialization.
  int preExecAllocCount = 0;
  if (!(shapesFrozen_ && seg.exec.executionCount > 2 && seg.exec.replayHandle != nullptr)) {
  for (int stepIdx = seg.startSlot; stepIdx <= seg.endSlot; stepIdx++) {
    NativeSlot& slot = slots_[stepIdx];
    // Phase 2: slotArrayCache_ == outputSlots_ (unified). No restore needed.
    // Validate input DataBuffers — Java close() may have freed them.
    for (int i = 0; i < slot.numInputs; i++) {
      int srcIdx = slot.inputSourceIndices[i];
      if (srcIdx >= 0 && srcIdx < totalOutputSlots_ && outputSlots_[srcIdx] != nullptr) {
        auto* db = outputSlots_[srcIdx]->dataBuffer();
        if (db == nullptr || !db->isValid()) {
          outputSlots_[srcIdx] = nullptr;
          if (srcIdx < numSlots_ && slots_[srcIdx].state_ == NativeSlot::SlotState::FROZEN_CONSTANT) {
            slots_[srcIdx].state_ = NativeSlot::SlotState::FROZEN;
          }
        }
      }
    }
    // Validate or allocate output slot entries
    for (int i = 0; i < slot.numOutputs; i++) {
      int slotIdx = slot.outputSlotIndices[i];
      if (slotIdx < 0 || slotIdx >= totalOutputSlots_) continue;
      // Validate existing entry
      if (outputSlots_[slotIdx] != nullptr) {
        auto* db = outputSlots_[slotIdx]->dataBuffer();
        if (db == nullptr || !db->isValid()) {
          outputSlots_[slotIdx] = nullptr;
          if (stepIdx < numSlots_ && slots_[stepIdx].state_ == NativeSlot::SlotState::FROZEN_CONSTANT) {
            slots_[stepIdx].state_ = NativeSlot::SlotState::FROZEN;
          }
        }
      }
      if (outputSlots_[slotIdx] == nullptr) {
        // After warmup with frozen shapes, null output slots indicate a persistence bug.
        // Frozen constant slots are exempt (they never allocate output arrays).
        // Warn but continue — the allocation path below will recover.
        if (shapesFrozen_ && seg.exec.executionCount > 1 && !slot.frozenConstantSlot()) {
          sd_printf("DSP BUG: Null output slot %d (%s) after warmup with frozen shapes — persistence bug. execCount=%d\n",
                    slotIdx, slot.opName.c_str(), seg.exec.executionCount);
        }
        // Allocate from cached shape info (populated during warmup)
        const LongType* shapeInfo = nullptr;
        if (i < static_cast<int>(slot.cachedOutputShapes.size()) && slot.cachedOutputShapes[i]) {
          shapeInfo = slot.cachedOutputShapes[i];
        }
        // Fallback: for identity/view-like ops that don't cache output shapes,
        // derive the shape from the first input source's existing array
        if (!shapeInfo && slot.numInputs > 0) {
          int srcIdx = slot.inputSourceIndices[0];
          NDArray* srcArr = nullptr;
          if (srcIdx < 0) {
            int extIdx = -(srcIdx + 1);
            if (extIdx < numExt) srcArr = externalArrays[extIdx];
          } else if (srcIdx < totalOutputSlots_) {
            srcArr = outputSlots_[srcIdx];
            // Phase 2: slotArrayCache_ == outputSlots_ (unified), no separate restore
          }
          if (srcArr) shapeInfo = srcArr->shapeInfo();
        }
        if (shapeInfo) {
          auto dt = ArrayOptions::dataType(shapeInfo);
          // For cast ops, the output type must match the declared target type,
          // not the input type. When cachedOutputShapes is empty and the
          // fallback uses the input source's shape, the dtype would be wrong
          // (e.g., INT64 input for a cast-to-FLOAT op).
          if ((slot.opName == "cast" || slot.opName == "Cast") &&
              slot.numIArgs > 0 && slot.iArgs) {
            auto castDt = static_cast<DataType>(slot.iArgs[0]);
            if (castDt != dt) {
              DSP_DIAG(EXECUTE, "PRE_EXEC_ALLOC: cast dtype override slot=%d from %s to %s",
                       slotIdx, DataTypeUtils::asString(dt).c_str(),
                       DataTypeUtils::asString(castDt).c_str());
              dt = castDt;
            }
          }
          auto order = shape::order(shapeInfo);
          LongType rank = shape::rank(shapeInfo);
          std::vector<LongType> shapeVec(rank);
          for (int d = 0; d < rank; d++) shapeVec[d] = shapeInfo[d + 1];
          auto* arr = new NDArray(order, shapeVec, dt);
          outputSlots_[slotIdx] = arr;
          // Phase 2: slotArrayCache_ == outputSlots_ (unified), no separate assignment needed
          preExecAllocCount++;
          if (Environment::getInstance().tritonVerifyKernels()) {
            DSP_DIAG(VERIFY, "SLOT_WRITE slot=%d tag=ALLOC dtype=%s len=%lld addr=%p",
                      slotIdx, DataTypeUtils::asString(dt).c_str(),
                      (long long)arr->lengthOf(), DSP_BUF(arr));
          }
        }
      }
    }
  }
  } // end if (!(shapesFrozen_ && executionCount > 2))

  // Compile once per stable shape; skip cache probe on steady-state replay.
  // This keeps the hot path focused on dispatch instead of repeated compile checks.
  // NOTE: Pre-exec output slot allocation above ensures all slots are populated
  // before the compiler resolves arg mappings. Without this ordering, intermediate
  // slots released after warmup are null and get omitted from the arg table,
  // causing sub-kernels to read stale data on their first execution.
  bool needsCompile = (seg.exec.executionCount == 1) || (seg.shapeKey != segShapeKey);
  if (needsCompile) {
    // When recompiling due to shape change (not the first compile), outputSlots_
    // has stale shapes from the previous execution. The compiler reads these shapes
    // to derive kernel parameters (e.g., seqQ/seqK for FUSED_ATTENTION). Run a
    // slot-by-slot pass first to populate outputSlots_ with current shapes before
    // compiling. This is like a mini-warmup for the new shape configuration.
    bool isRecompileDueToShapeChange = (seg.exec.executionCount > 1) && (seg.shapeKey != segShapeKey);
    if (isRecompileDueToShapeChange) {
      DSP_DIAG(COMPILE, "executeSegmentWithGpuGraph: shape change detected for seg[%d-%d] "
                "(shapeKey %lld->%lld, executionCount=%d). Running slot-by-slot warmup to "
                "refresh outputSlots_ before recompilation.",
                seg.startSlot, seg.endSlot, seg.shapeKey, segShapeKey, seg.exec.executionCount);
      // Invalidate cached graph — addresses and shapes changed
#ifdef SD_CUDA
      platformCleanupSegmentForRebuild(seg);
      seg.exec.argTableStable = false;
      batchD2DCount_ = 0;
#endif
      auto warmupStatus = executeSegmentSlotBySlot(seg, externalArrays, numExt, stream);
      if (warmupStatus != Status::OK) {
        DSP_DIAG(COMPILE, "executeSegmentWithGpuGraph: shape-change warmup FAILED for seg[%d-%d] status=%d",
                  seg.startSlot, seg.endSlot, static_cast<int>(warmupStatus));
        return warmupStatus;
      }
      // Recompute shape key after warmup — outputSlots_ now has correct shapes
      segShapeKey = computeSegmentShapeKey(seg, externalArrays, numExt);
      DSP_DIAG(COMPILE, "executeSegmentWithGpuGraph: shape-change warmup OK for seg[%d-%d], "
                "recomputed shapeKey=%lld", seg.startSlot, seg.endSlot, segShapeKey);
    }

    if (!backend->compileSegment(seg, slots_, externalArrays, numExt,
                                 outputSlots_, totalOutputSlots_, segShapeKey,
                                 numSlots_)) {
      DSP_DIAG(COMPILE, "executeSegmentWithGpuGraph: backend=%s compile failed for seg[%d-%d]",
                backendName, seg.startSlot, seg.endSlot);
      return Status::KERNEL_FAILURE;
    }
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
      // All ops FAILED compilation — hard error.
      DSP_DIAG(COMPILE, "%s COMPILE ERROR: segment [%d-%d] has zero compiled ops "
                "(failed=%d). Compilation failures are errors, not fallbacks.",
                backendName, seg.startSlot, seg.endSlot, failedCount);
      seg.exec.compilationFailed = true;
      return Status::KERNEL_FAILURE;
    }
    if (compiledCount == 0 && failedCount == 0) {
      // All sections are intentional fallback (e.g., all non-elementwise/matmul).
      // The compiled segment has 0 sub-kernels; executeSegment will run
      // everything via fallbackRangeExecutor_.
      // DO NOT set compilationFailed — allow these segments to be captured as
      // CUDA graphs. During Triton graph capture, gap ops (cuBLAS matmuls) are
      // recorded into the graph via the fallback lambda, enabling single-launch
      // replay instead of per-op kernel dispatch overhead.
      DSP_DIAG(COMPILE, "%s: segment [%d-%d] has only fallback sections (no compilation needed). "
                "Segment eligible for CUDA graph capture via fallback path.",
                backendName, seg.startSlot, seg.endSlot);
    }
    if (failedCount > 0) {
      // Partial compilation failure — hard error. Fix the kernel.
      DSP_DIAG(COMPILE, "%s COMPILE ERROR: segment [%d-%d] partial compile FAILED "
                "(compiled=%d failed=%d). Compilation failures are errors, not fallbacks.",
                backendName, seg.startSlot, seg.endSlot, compiledCount, failedCount);
      seg.exec.compilationFailed = true;
      return Status::KERNEL_FAILURE;
    }
  }

  // Execute via selected GPU backend
  seg.shapeKey = segShapeKey;

#ifdef SD_CUDA
  cudaStream_t cudaStr = (stream != nullptr)
      ? *static_cast<cudaStream_t*>(stream) : nullptr;

  // If any output slots were re-allocated at new addresses, the cached CUDA graph
  // is invalid — native ops (cuBLAS) have the old addresses baked in while Triton
  // arg tables were refreshed with new addresses. Invalidate and re-capture.
  if (preExecAllocCount > 0 && seg.exec.replayHandle != nullptr) {
    DSP_DIAG(EXECUTE, "GRAPH INVALIDATED: %d output slots re-allocated at new addresses "
              "(cache entries freed by Java). seg[%d-%d] will re-capture.",
              preExecAllocCount, seg.startSlot, seg.endSlot);
    platformCleanupSegmentForRebuild(seg);
    seg.exec.argTableStable = false;
    batchD2DCount_ = 0;
    seg.exec.capturedInputAddrKey = 0;
    // Reset execution count to trigger warmup→capture flow
    seg.exec.executionCount = 0;
    seg.exec.compilationFailed = false;
  }

  bool allowTritonCudaGraphReplay = Environment::getInstance().tritonGraphCapture() &&
                                    shapesFrozen_;

  // BLATANT DIAGNOSTIC: Log the capture decision factors
  int captureMinExec = Environment::getInstance().tritonCaptureMinExec();
  bool forceRecaptureEnabled = Environment::getInstance().tritonForceRecapture();
  bool hasReplayHandle = (seg.exec.replayHandle != nullptr);
  bool hasCaptureBuffers = hasReplayHandle && !seg.exec.replayHandle->getCaptureBuffers().empty();
  bool replayHandleNull = (seg.exec.replayHandle == nullptr);
  bool notCaptureFailed = !seg.exec.compilationFailed;
  bool execCountInWindow = (seg.exec.executionCount >= captureMinExec) && 
                           (forceRecaptureEnabled || seg.exec.executionCount <= (captureMinExec + 2));
  bool hasCudaStream = (cudaStr != nullptr);
  
  DSP_DIAG(EXECUTE, "=== CAPTURE DECISION CHECK seg[%d-%d] ===", seg.startSlot, seg.endSlot);
  DSP_DIAG(EXECUTE, "  tritonGraphCapture()=%d, shapesFrozen_=%d => allowTritonCudaGraphReplay=%d",
           Environment::getInstance().tritonGraphCapture() ? 1 : 0,
           shapesFrozen_ ? 1 : 0, allowTritonCudaGraphReplay ? 1 : 0);
  DSP_DIAG(EXECUTE, "  seg.exec.executionCount=%d, captureMinExec=%d, window=[%d,%d], inWindow=%d",
           seg.exec.executionCount, captureMinExec, captureMinExec, captureMinExec + 2,
           execCountInWindow ? 1 : 0);
  DSP_DIAG(EXECUTE, "  hasReplayHandle=%d, hasCaptureBuffers=%d, replayHandleNull=%d",
           hasReplayHandle ? 1 : 0, hasCaptureBuffers ? 1 : 0, replayHandleNull ? 1 : 0);
  DSP_DIAG(EXECUTE, "  compilationFailed=%d, cudaStr!=nullptr=%d",
           seg.exec.compilationFailed ? 1 : 0, hasCudaStream ? 1 : 0);
  
  bool shouldCaptureTritonGraph = allowTritonCudaGraphReplay &&
                                  (!hasReplayHandle || !hasCaptureBuffers) &&
                                  replayHandleNull &&
                                  notCaptureFailed &&
                                  execCountInWindow &&
                                  hasCudaStream;
  
  DSP_DIAG(EXECUTE, "  => shouldCaptureTritonGraph=%d", shouldCaptureTritonGraph ? 1 : 0);
  if (!shouldCaptureTritonGraph) {
    if (!allowTritonCudaGraphReplay) 
      DSP_DIAG(EXECUTE, "  BLOCKED: allowTritonCudaGraphReplay=false (tritonGraphCapture=%d OR shapesFrozen_=%d)",
               Environment::getInstance().tritonGraphCapture() ? 1 : 0, shapesFrozen_ ? 1 : 0);
    if (!replayHandleNull)
      DSP_DIAG(EXECUTE, "  BLOCKED: replayHandle already exists (capture already done or in progress)");
    if (seg.exec.compilationFailed)
      DSP_DIAG(EXECUTE, "  BLOCKED: compilationFailed=true (previous capture failed, falling back to slot-by-slot)");
    if (!execCountInWindow)
      DSP_DIAG(EXECUTE, "  BLOCKED: executionCount=%d outside capture window [%d,%d]",
               seg.exec.executionCount, captureMinExec, captureMinExec + 2);
    if (!hasCudaStream)
      DSP_DIAG(EXECUTE, "  BLOCKED: cudaStr=nullptr (no CUDA stream available)");
  } else {
    DSP_DIAG(EXECUTE, "  >>> CAPTURE WILL BE ATTEMPTED <<<");
  }
  DSP_DIAG(EXECUTE, "=== END CAPTURE DECISION CHECK ===");
  
  // NOTE: shouldCaptureTritonGraph is ONLY checked when we don't have a captured graph.
  // Once captured, we use useFastReplay based on argTableStable, not executionCount.
  // The executionCount window check prevents repeated capture attempts after success.
  
  // OPTIMIZATION: When argTableStable, addresses and create-op values haven't changed
  // since last refresh — skip the expensive hash/comparison loops over all external inputs.
  LongType segInputAddrKey;
  bool extAddrsStable;
  LongType createValueKey;
  if (seg.exec.argTableStable && allowTritonCudaGraphReplay) {
    // Fast path: arg table is stable, all addresses are known-good
    DSP_DIAG_SEG(EXECUTE, seg.startSlot,
                 "seg[%d-%d] argTableStable=true → FAST PATH (skip addr/createValue recompute)",
                 seg.startSlot, seg.endSlot);
    segInputAddrKey = seg.exec.capturedInputAddrKey;
    extAddrsStable = true;
    createValueKey = seg.exec.capturedCreateValueKey;
  } else {
    segInputAddrKey = computeSegmentInputAddrKey(seg, externalArrays, numExt);
    extAddrsStable = (seg.exec.replayHandle && !seg.exec.replayHandle->getCapturedExternalAddresses().empty())
        ? externalAddrsMatch(seg, externalArrays, numExt)
        : (seg.exec.capturedInputAddrKey != 0 && seg.exec.capturedInputAddrKey == segInputAddrKey);
    createValueKey = computeCreateOpValueKey(seg, externalArrays, numExt);
    DSP_DIAG(EXECUTE, "ADDR_CHECK_SLOW: seg[%d-%d] extAddrsStable=%d addrKey=%lld (cached=%lld)",
             seg.startSlot, seg.endSlot, extAddrsStable ? 1 : 0,
             (long long)segInputAddrKey, (long long)seg.exec.capturedInputAddrKey);
  }
  bool createValuesStable = (createValueKey == 0) ||  // no create ops
                            (seg.exec.capturedCreateValueKey == createValueKey);
  if (!createValuesStable && seg.exec.replayHandle) {
    DSP_DIAG(EXECUTE, "CREATE_VALUE_KEY mismatch: captured=%lld current=%lld → invalidating graph seg[%d-%d]",
             (long long)seg.exec.capturedCreateValueKey, (long long)createValueKey, seg.startSlot, seg.endSlot);
    platformCleanupSegmentForRebuild(seg);
    seg.exec.argTableStable = false;
    batchD2DCount_ = 0;
    seg.exec.capturedInputAddrKey = 0;
    seg.exec.capturedCreateValueKey = 0;
    seg.exec.executionCount = 0;
    seg.exec.compilationFailed = false;
    extAddrsStable = false;  // Force re-capture path
  }

  // Triton graph replay conditions:
  // 1. Shape key matches (frozen shapes)
  // 2. Create op input values stable (ConstantOfShape shapes unchanged)
  // 3. Either: addresses stable OR capture buffers handle data freshness
  //
  // With capture buffers (for PLACEHOLDER inputs), we D2D copy fresh data
  // before replay. The graph reads from capture buffer addresses (baked in)
  // and gets current placeholder values (position_ids, attention_mask, etc.).
  bool hasTritonCaptureBuffers = seg.exec.replayHandle != nullptr &&
                                  !seg.exec.replayHandle->getCaptureBuffers().empty();

  // CRITICAL: Only enter the Triton replay path for segments actually compiled by Triton.
  // Segments captured by the raw CUDA graph path (NativeDynamicShapePlan_cudagraph.cu)
  // have replayHandles but NO Triton arg tables. The Triton replay path's D2D copy +
  // arg table refresh is incompatible with raw CUDA graphs — it can corrupt cross-segment
  // data, causing downstream segments to read zeros instead of valid output → NaN.
  // compiledByBackend is set to backendName ONLY after a successful Triton execution.
  // Raw CUDA captures leave it empty → excluded from this path → fall through to
  // executeSegmentWithGraph() in cudagraph.cu which handles replay correctly.
  bool isTritonCompiled = (!seg.exec.compiledByBackend.empty() && seg.exec.compiledByBackend == backendName);

  if (allowTritonCudaGraphReplay && seg.exec.replayHandle != nullptr &&
      seg.exec.replayHandle->isReady() && !isTritonCompiled) {
    DSP_DIAG(EXECUTE, "TRITON_REPLAY_SKIP: seg[%d-%d] has replayHandle but compiledBy='%s' (not %s) "
             "→ falling through to raw CUDA graph replay path",
             seg.startSlot, seg.endSlot,
             seg.exec.compiledByBackend.empty() ? "(empty)" : seg.exec.compiledByBackend.c_str(),
             backendName);
  }

  if (allowTritonCudaGraphReplay &&
      seg.exec.replayHandle != nullptr &&
      seg.exec.replayHandle->isReady() &&
      isTritonCompiled &&
      seg.exec.cachedShapeKey == segShapeKey &&
      createValuesStable &&
      (hasTritonCaptureBuffers || extAddrsStable)) {

    DSP_DIAG(EXECUTE, "TRITON_REPLAY_ENTER: seg[%d-%d] extAddrsStable=%d argTableStable=%d "
             "hasTritonCaptureBuffers=%d captureBufferCount=%d",
             seg.startSlot, seg.endSlot, extAddrsStable ? 1 : 0,
             seg.exec.argTableStable ? 1 : 0, hasTritonCaptureBuffers ? 1 : 0,
             hasTritonCaptureBuffers ? (int)seg.exec.replayHandle->getCaptureBuffers().size() : 0);

    // ── Lineage validation: verify directReference addresses haven't drifted ──
    // DirectReference entries (weights, KV cache) assume the graph reads from
    // the original buffer address. If the address changed (freed/reallocated),
    // the graph reads garbage. Detect and invalidate.
    // OPTIMIZATION: Skip when argTableStable is true AND external addresses are stable.
    // When extAddrsStable is false, external input buffers may have been reallocated,
    // so directReference entries could point to freed memory. MUST check lineage.
    bool lineageInvalidated = false;
    if (hasTritonCaptureBuffers && (!(seg.exec.argTableStable && allowTritonCudaGraphReplay) || !extAddrsStable)) {
      DSP_DIAG(EXECUTE, "LINEAGE_CHECK_ENTER: seg[%d-%d] reason=%s (argTableStable=%d extAddrsStable=%d)",
               seg.startSlot, seg.endSlot,
               !extAddrsStable ? "extAddrsUnstable" : "argTableNotStable",
               seg.exec.argTableStable ? 1 : 0, extAddrsStable ? 1 : 0);
      bool addressDrift = false;
      for (auto& cb : seg.exec.replayHandle->getCaptureBuffers()) {
        if (!cb.directReference) continue;
        const void* currentPtr = nullptr;
        if (cb.externalInputIndex >= 0 && cb.externalInputIndex < numExt) {
          NDArray* current = externalArrays[cb.externalInputIndex];
          currentPtr = current ? DSP_BUF(current) : nullptr;
        } else if (cb.crossSegmentSlotIdx >= 0 && cb.crossSegmentSlotIdx < totalOutputSlots_) {
          NDArray* current = outputSlots_[cb.crossSegmentSlotIdx];
          currentPtr = current ? DSP_BUF(current) : nullptr;
        }
        if (currentPtr != cb.lastSourcePtr) {
          DSP_DIAG(EXECUTE, "LINEAGE_DRIFT: %s#%d addr changed %p → %p → invalidate seg[%d-%d]",
                   cb.externalInputIndex >= 0 ? "ext" : "slot",
                   cb.externalInputIndex >= 0 ? cb.externalInputIndex : cb.crossSegmentSlotIdx,
                   cb.lastSourcePtr, currentPtr, seg.startSlot, seg.endSlot);
          addressDrift = true;
          break;
        }
      }
      if (addressDrift) {
        if (planPhase_ >= PlanPhase::REPLAYING) {
          // HARD ERROR during REPLAYING: address drift means the captured graph
          // would read/write stale memory. This is a correctness violation.
          DSP_DIAG(FALLBACK, "LINEAGE_DRIFT_HARD_ERROR: seg[%d-%d] address drift during REPLAYING phase "
                   "— captured graph would read stale memory. planPhase=%d execCount=%d",
                   seg.startSlot, seg.endSlot, static_cast<int>(planPhase_), executeCount_);
          char errMsg[256];
          snprintf(errMsg, sizeof(errMsg),
                   "DSP phase contract violation: address drift in REPLAYING phase for seg[%d-%d]. "
                   "Buffer pointer changed after graph was captured — graph replay would read stale memory.",
                   seg.startSlot, seg.endSlot);
          sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(
              static_cast<int>(Status::KERNEL_FAILURE));
          sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(errMsg);
          return Status::KERNEL_FAILURE;
        }
        DSP_DIAG(EXECUTE, "LINEAGE_DRIFT_INVALIDATE: seg[%d-%d] → cleaning up for re-capture",
                 seg.startSlot, seg.endSlot);
        platformCleanupSegmentForRebuild(seg);
        seg.exec.argTableStable = false;
        batchD2DCount_ = 0;
        seg.exec.capturedInputAddrKey = 0;
        seg.exec.executionCount = 0;
        seg.exec.compilationFailed = false;
        hasTritonCaptureBuffers = false;
        lineageInvalidated = true;
      } else {
        DSP_DIAG(EXECUTE, "LINEAGE_CHECK_PASS: seg[%d-%d] all directReference addresses match",
                 seg.startSlot, seg.endSlot);
      }
    } else {
      DSP_DIAG(EXECUTE, "LINEAGE_CHECK_SKIP: seg[%d-%d] argTableStable=%d extAddrsStable=%d "
               "→ skipped (both stable, addresses known-good)",
               seg.startSlot, seg.endSlot,
               seg.exec.argTableStable ? 1 : 0, extAddrsStable ? 1 : 0);
    }

    // Fast-replay: when arg table pointers are stable (all unchanged since last
    // refresh), skip the arg table refresh loop and EXT_INPUT_SYNC entirely.
    // Only D2D capture buffer copies + graph launch needed.
    bool useFastReplay = hasTritonCaptureBuffers && seg.exec.argTableStable
                         && !Environment::getInstance().tritonVerifyKernels();
    DSP_DIAG(EXECUTE, "REPLAY_PATH: seg[%d-%d] useFastReplay=%d (hasCapBufs=%d argStable=%d verify=%d) "
             "lineageInvalidated=%d",
             seg.startSlot, seg.endSlot, useFastReplay ? 1 : 0,
             hasTritonCaptureBuffers ? 1 : 0, seg.exec.argTableStable ? 1 : 0,
             Environment::getInstance().tritonVerifyKernels() ? 1 : 0,
             lineageInvalidated ? 1 : 0);

    // ── cuBLAS workspace invariant assertion during REPLAYING ──────────────
    // During REPLAYING, the cuBLAS workspace address and size must not change.
    // cuBLAS plans captured in the graph reference specific workspace addresses.
    // If the workspace moved or was resized, graph replay reads stale pointers.
    if (planPhase_ >= PlanPhase::REPLAYING && cublasWorkspaceBuffer_ != nullptr) {
      static thread_local void* lastCublasWorkspaceAddr = nullptr;
      static thread_local size_t lastCublasWorkspaceSize = 0;
      if (lastCublasWorkspaceAddr == nullptr) {
        // First check — record current state
        lastCublasWorkspaceAddr = cublasWorkspaceBuffer_;
        lastCublasWorkspaceSize = cublasWorkspaceSize_;
      } else {
        if (cublasWorkspaceBuffer_ != lastCublasWorkspaceAddr ||
            cublasWorkspaceSize_ != lastCublasWorkspaceSize) {
          char errMsg[256];
          snprintf(errMsg, sizeof(errMsg),
                   "DSP phase contract violation: cuBLAS workspace changed during REPLAYING phase. "
                   "addr %p → %p, size %zu → %zu. Captured graphs have stale workspace pointers.",
                   lastCublasWorkspaceAddr, cublasWorkspaceBuffer_,
                   lastCublasWorkspaceSize, cublasWorkspaceSize_);
          DSP_DIAG(FALLBACK, "CUBLAS_WORKSPACE_DRIFT: %s", errMsg);
          sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(
              static_cast<int>(Status::KERNEL_FAILURE));
          sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(errMsg);
          return Status::KERNEL_FAILURE;
        }
      }
    }

    // CRITICAL FIX: Set tl_dspExecutionStream for ALL Triton executions, not just capture replay.
    // Without this, syncToSpecial() calls fall back to stream 0 and do full cudaStreamSynchronize,
    // causing 657k sync calls per decode step. Setting tl_dspExecutionStream allows async H2D
    // copies on the same stream as compute, with stream ordering guaranteeing correctness.
    // RAII guard: restores previous tl_dspExecutionStream value when this function exits.
    sd::graph::DspStreamGuard dspStreamGuard(cudaStr);

    // Update capture buffers with fresh data (D2D copy).
    // Handles BOTH placeholder external inputs AND cross-segment output slots.
    // Use tl_dspExecutionStream for any syncToDevice calls inside the loop.
    bool crossSegSizeMismatch = false;
    if (hasTritonCaptureBuffers) {
      auto& captureBuffers = seg.exec.replayHandle->getCaptureBuffers();

      // REPLAY OPTIMIZATION: In stable replay (executionCount >= 4 with frozen shapes),
      // skip size mismatch checks — shapes are frozen and can never change. This avoids
      // per-buffer size computation (lengthOf * sizeOfT) and conditional branches.
      // Also skip syncToDevice for external inputs when useFastReplay is true — the
      // fast replay ext sync loop already synced variable inputs, and non-variable
      // inputs are already on device.
      bool skipSizeCheck = shapesFrozen_ && seg.exec.executionCount >= 4;

      // D2D copies for ALL capture buffers (placeholders + cross-segment).
      {
        int cbExtUpdated = 0, cbSlotUpdated = 0;
        for (auto& cb : captureBuffers) {
          if (cb.directReference) continue;
          if (cb.buffer == nullptr) continue;

          NDArray* src = nullptr;
          if (cb.externalInputIndex >= 0 && cb.externalInputIndex < numExt) {
            // Placeholder external input
            src = externalArrays[cb.externalInputIndex];
          } else if (cb.crossSegmentSlotIdx >= 0 && cb.crossSegmentSlotIdx < totalOutputSlots_) {
            // Cross-segment output slot
            src = outputSlots_[cb.crossSegmentSlotIdx];
          }
          if (src == nullptr) continue;

          if (skipSizeCheck) {
            // Stable replay fast path: skip size validation and conditional sync.
            // Cross-segment outputs are already on device (written by prior segment).
            // External variable inputs were synced by the fast replay ext sync loop.
            const void* srcPtr = DSP_BUF(src);
            if (!srcPtr || !DSP_BUF(cb.buffer)) continue;

            copyIntoCaptureBuffer(cb.buffer, src, cudaStr, false,
                                  cb.externalInputIndex >= 0 ? "ext" : "slot",
                                  cb.externalInputIndex >= 0 ? cb.externalInputIndex
                                                             : cb.crossSegmentSlotIdx,
                                  seg.startSlot, seg.endSlot);
            cb.lastSourcePtr = srcPtr;
            if (cb.externalInputIndex >= 0) cbExtUpdated++;
            else cbSlotUpdated++;
          } else {
            // Normal path with full validation
            size_t srcBytes = src->lengthOf() * src->sizeOfT();
            if (srcBytes == 0) continue;
            if (srcBytes != cb.capturedSize) {
              // Size mismatch — shape changed since capture. The CUDA graph has
              // baked-in tensor dimensions, so we must invalidate and re-capture.
              // But first: reallocate the staging buffer to the new size so the
              // re-capture will use a correctly-sized buffer (no churn on next exec).
              if (cb.crossSegmentSlotIdx >= 0) {
                DSP_DIAG(EXECUTE, "CROSS_SEG_SIZE_MISMATCH: slot#%d captured=%zu current=%zu → invalidate",
                         cb.crossSegmentSlotIdx, cb.capturedSize, srcBytes);
              } else {
                // Reallocate staging buffer to match new input size.
                // This is the "patching path" — ideally callers pad inputs to max size
                // so this never fires. But when it does, one realloc stabilizes the buffer.
                auto srcShapeVec = *src->getShapeAsVector();
                NDArray* newBuf = new NDArray(src->ordering(), srcShapeVec, src->dataType(),
                                              sd::LaunchContext::defaultContext());
                DSP_DIAG(EXECUTE, "EXT_STAGING_REALLOC: ext#%d captured=%zu current=%zu → "
                         "reallocated staging buffer for seg[%d-%d] (ideally pad inputs to max size)",
                         cb.externalInputIndex, cb.capturedSize, srcBytes,
                         seg.startSlot, seg.endSlot);
                // Free old buffer and swap in new one
                if (cb.buffer != nullptr) {
                  delete cb.buffer;
                }
                cb.buffer = newBuf;
                cb.capturedSize = srcBytes;
              }
              crossSegSizeMismatch = true;
              break;
            }

            // Cross-segment outputs were just written on cudaStr by the previous
            // segment's slot-by-slot execution — already on device, no sync needed.
            // Only external (placeholder) inputs need syncToDevice() for H2D.
            if (cb.externalInputIndex >= 0) {
              src->syncToDevice();
            }
            const void* srcPtr = DSP_BUF(src);
            if (!srcPtr || !DSP_BUF(cb.buffer)) continue;

            // Always refresh — GPU memory pools reuse addresses, so pointer comparison
            // cannot detect stale data. Dense sources use raw D2D; view-backed sources
            // use a logical tensor copy to preserve layout semantics.
            copyIntoCaptureBuffer(cb.buffer, src, cudaStr, false,
                                  cb.externalInputIndex >= 0 ? "ext" : "slot",
                                  cb.externalInputIndex >= 0 ? cb.externalInputIndex
                                                             : cb.crossSegmentSlotIdx,
                                  seg.startSlot, seg.endSlot);
            cb.lastSourcePtr = srcPtr;
            cb.initialCopyDone = true;
            if (cb.externalInputIndex >= 0) cbExtUpdated++;
            else cbSlotUpdated++;
          }
        }
        DSP_DIAG(EXECUTE, "CAPTURE_BUFFER_UPDATE: ext=%d slot=%d "
                 "fastReplay=%d execCount=%d", cbExtUpdated, cbSlotUpdated,
                 useFastReplay ? 1 : 0, seg.exec.executionCount);
      }
    }
    // NOTE: tl_dspExecutionStream is managed by DspStreamGuard in this function and in execute().
    // It remains set during replay for syncToSpecial() to use the correct stream.

    // Explicit decode input propagation during replay: directly write decode
    // input values (input_ids, position_ids, attention_mask) to their capture
    // buffers via cudaMemcpyAsync H2D. This supplements the D2D copy above
    // which copies from external arrays — the D2D path goes through
    // copyIntoCaptureBuffer which may use a logical copy path for views or
    // non-dense arrays. The direct H2D write here is guaranteed correct and
    // ensures decode inputs are always fresh in capture buffers, even on the
    // first replay after capture when propagateDecodeInputsToCaptureBuffers
    // in execute() may not have found handles yet.
    if (hasTritonCaptureBuffers && !crossSegSizeMismatch && isDecodeInputsConfigured()) {
      auto& captureBuffers = seg.exec.replayHandle->getCaptureBuffers();
      for (auto& cb : captureBuffers) {
        if (cb.directReference || cb.buffer == nullptr) continue;
        int ei = cb.externalInputIndex;
        if (ei < 0) continue;
        void* dstBuf = DSP_BUF(cb.buffer);
        if (dstBuf == nullptr) continue;

        if (ei == decodeInputIdsExtIdx_) {
          LongType val = static_cast<LongType>(pendingTokenId_);
          cudaMemcpyAsync(dstBuf, &val, sizeof(LongType),
                          cudaMemcpyHostToDevice, cudaStr);
        } else if (ei == decodePositionIdsExtIdx_) {
          LongType val = static_cast<LongType>(pendingCachePos_);
          cudaMemcpyAsync(dstBuf, &val, sizeof(LongType),
                          cudaMemcpyHostToDevice, cudaStr);
        } else if (ei == decodeAttentionMaskExtIdx_) {
          int writePos = pendingCachePos_ - 1;
          auto maskLen = cb.buffer->lengthOf();
          if (writePos >= 0 && writePos < static_cast<int>(maskLen)) {
            LongType one = 1;
            auto* dst = static_cast<LongType*>(dstBuf) + writePos;
            cudaMemcpyAsync(dst, &one, sizeof(LongType),
                            cudaMemcpyHostToDevice, cudaStr);
          }
        }
      }
      DSP_DIAG(EXECUTE, "TRITON_REPLAY_DECODE_PROPAGATION: seg[%d-%d] wrote decode inputs "
               "to capture buffers (tokenId=%lld cachePos=%d)",
               seg.startSlot, seg.endSlot, pendingTokenId_, pendingCachePos_);
    }

    // Capture buffer size mismatch: invalidate graph and fall through to re-capture.
    // For external inputs, the staging buffer was already reallocated above.
    // For cross-segment, the slot data shape changed (data-dependent).
    if (crossSegSizeMismatch && seg.exec.replayHandle) {
      DSP_DIAG(EXECUTE, "GRAPH INVALIDATED: capture buffer size mismatch for seg[%d-%d] → re-capture",
               seg.startSlot, seg.endSlot);
      platformCleanupSegmentForRebuild(seg);
      seg.exec.argTableStable = false;
      batchD2DCount_ = 0;
      seg.exec.capturedInputAddrKey = 0;
      seg.exec.executionCount = 0;
      seg.exec.compilationFailed = false;
    } else

    if (useFastReplay) {
      // Fast path: arg table pointers are stable so skip refresh.
      // Only sync VARIABLE (PLACEHOLDER) external inputs — model weights and
      // constants are already on device and never change. This reduces the
      // sync loop from ~1333 inputs to ~3 (input_ids, attention_mask, position_ids).
      //
      // REPLAY OPTIMIZATION: Use cached variable input indices to avoid iterating
      // all 1333 external inputs. Populates variableExternalInputIndices_ once on
      // first use, then iterates only the ~3 variable indices per segment.
      cudaGetLastError();
      // tl_dspExecutionStream managed by DspStreamGuard in execute() and this function
      if (!variableIndicesCached_) {
        variableExternalInputIndices_.clear();
        for (int ei = 0; ei < numExt; ei++) {
          if (ei < static_cast<int>(externalInputIsVariable_.size()) &&
              externalInputIsVariable_[ei]) {
            variableExternalInputIndices_.push_back(ei);
          }
        }
        variableIndicesCached_ = true;
        DSP_DIAG(EXECUTE, "FAST_REPLAY_CACHED_VARIABLE_INDICES: %d variable inputs out of %d total",
                 static_cast<int>(variableExternalInputIndices_.size()), numExt);
      }
      int fastSynced = 0;
      for (int ei : variableExternalInputIndices_) {
        if (ei >= numExt || externalArrays[ei] == nullptr) continue;
        externalArrays[ei]->syncToDevice();
        fastSynced++;
      }
      DSP_DIAG(EXECUTE, "FAST_REPLAY_EXT_SYNC: %d H2D (of %d variable) execCount=%d",
               fastSynced, static_cast<int>(variableExternalInputIndices_.size()),
               seg.exec.executionCount);
    } else {
    // Standard replay: sync ext inputs, refresh arg tables, diagnostics.
    // tl_dspExecutionStream managed by DspStreamGuard in execute() and this function
    //
    // REPLAY OPTIMIZATION: When shapes are frozen (stable replay), only sync
    // variable (PLACEHOLDER) inputs. Non-variable inputs (weights, constants)
    // are already on device and never change. This reduces the sync loop from
    // ~1333 inputs to ~3. On non-frozen path, sync all inputs for safety.
    {
      DspDiagnostics::ExtInputSyncResult syncResult = {0, 0, 0};
      DSP_DIAG_DUMP_EXT_INPUTS(externalArrays, numExt, seg.exec.executionCount, syncResult);
      int synced = 0, skipped = 0;
      if (shapesFrozen_ && !externalInputIsVariable_.empty()) {
        // Frozen replay: only sync variable inputs
        for (int ei = 0; ei < numExt; ei++) {
          if (externalArrays[ei] == nullptr) continue;
          if (ei < static_cast<int>(externalInputIsVariable_.size()) &&
              !externalInputIsVariable_[ei]) {
            skipped++;
            continue;
          }
          auto* db = externalArrays[ei]->dataBuffer();
          bool pAct = db ? db->isPrimaryActual() : false;
          bool sAct = db ? db->isSpecialActual() : false;
          if (pAct && !sAct) synced++;
          else skipped++;
          externalArrays[ei]->syncToDevice();
        }
      } else {
        // Non-frozen or no variable info: sync all
        for (int ei = 0; ei < numExt; ei++) {
          if (externalArrays[ei] != nullptr) {
            auto* db = externalArrays[ei]->dataBuffer();
            bool pAct = db ? db->isPrimaryActual() : false;
            bool sAct = db ? db->isSpecialActual() : false;
            if (pAct && !sAct) synced++;
            else skipped++;
            externalArrays[ei]->syncToDevice();
          }
        }
      }
      DSP_DIAG(EXECUTE, "EXT_INPUT_SYNC replay: %d H2D, %d skip (device up-to-date) execCount=%d",
               synced, skipped, seg.exec.executionCount);

      // Dump SMALL variable external inputs (verify mode only)
      if (Environment::getInstance().tritonVerifyKernels()) {
      cudaDeviceSynchronize();
      for (int ei = 0; ei < numExt; ei++) {
        if (externalArrays[ei] == nullptr) continue;
        auto* arr = externalArrays[ei];
        bool isSmall = arr->lengthOf() <= 16;
        std::string name = (ei < (int)externalInputNames_.size()) ? externalInputNames_[ei] : "?";
        std::string vals = "?";
        if (isSmall && DSP_BUF(arr)) {
          int n = std::min((int)arr->lengthOf(), 4);
          int elemSize = DataTypeUtils::sizeOf(arr->dataType());
          std::vector<uint8_t> devBytes(n * elemSize);
          cudaMemcpy(devBytes.data(), DSP_BUF(arr), n * elemSize, cudaMemcpyDeviceToHost);
          vals = "";
          for (int j = 0; j < n; j++) {
            if (j > 0) vals += ",";
            if (arr->dataType() == INT64 || arr->dataType() == DataType::INT64) {
              int64_t v; std::memcpy(&v, devBytes.data() + j * 8, 8);
              vals += std::to_string(v);
            } else if (arr->dataType() == INT32) {
              int32_t v; std::memcpy(&v, devBytes.data() + j * 4, 4);
              vals += std::to_string(v);
            } else if (arr->dataType() == FLOAT32) {
              float v; std::memcpy(&v, devBytes.data() + j * 4, 4);
              vals += std::to_string(v);
            } else {
              vals += "?";
            }
          }
        }
        if (!isSmall || name.find("input") != std::string::npos ||
            name.find("position") != std::string::npos ||
            name.find("attention") != std::string::npos ||
            name.find("embed") != std::string::npos ||
            name.find("past") != std::string::npos) {
          DSP_DIAG(EXECUTE, "EXT_DATA[%d]:\"%s\" type=%d rank=%d len=%lld addr=%p vals=[%s] execCount=%d",
                   ei, name.c_str(), (int)arr->dataType(), (int)arr->rankOf(),
                   (long long)arr->lengthOf(),
                   DSP_BUF(arr), vals.c_str(), seg.exec.executionCount);
        }
      }
      } // end tritonVerifyKernels() EXT_DATA dump
    }
    // Snapshot buffer addresses BEFORE replay for comparison with capture-time addresses.
    // REPLAY OPTIMIZATION: Only compute address snapshots during first few replays
    // or when diagnostics are enabled. In stable replay (executionCount >= 4),
    // addresses don't change. Skipping saves 2 vector allocations + iteration
    // over all output slots + external inputs (~3000+ arrays) per segment.
    if (seg.exec.executionCount < 4 || DSP_DIAG_ENABLED(EXECUTE)) {
      std::vector<void*> outAddrs, extAddrs;
      extractDeviceAddrs(outputSlots_, totalOutputSlots_, outAddrs);
      extractDeviceAddrs(externalArrays, numExt, extAddrs);
      DSP_DIAG_SNAPSHOT_ADDRS("replay-entry", outAddrs.data(), totalOutputSlots_,
                               extAddrs.data(), numExt);
      int mismatches = DSP_DIAG_COMPARE_ADDRS("capture-entry", "replay-entry");
      if (mismatches > 0) {
        DSP_DIAG(EXECUTE, "WARNING: %d address mismatches between capture and replay!", mismatches);
      }
    }

    // Refresh Triton arg table pinned buffers before replay.
    // When capture buffers exist, temporarily swap externalArrays AND outputSlots_
    // to capture buffer addresses so the arg table gets the addresses baked into
    // the graph. This covers both placeholder external inputs AND cross-segment
    // output slots.
#if HAVE_TRITON && defined(SD_CUDA)
    {
      std::vector<std::pair<int, NDArray*>> savedForArgRefresh;
      std::vector<std::pair<int, NDArray*>> savedSlotsForArgRefresh;
      if (hasTritonCaptureBuffers) {
        for (auto& cb : seg.exec.replayHandle->getCaptureBuffers()) {
          if (cb.externalInputIndex >= 0 && cb.externalInputIndex < numExt && cb.buffer) {
            savedForArgRefresh.push_back({cb.externalInputIndex, externalArrays[cb.externalInputIndex]});
            externalArrays[cb.externalInputIndex] = cb.buffer;
          }
          if (cb.crossSegmentSlotIdx >= 0 && cb.crossSegmentSlotIdx < totalOutputSlots_ && cb.buffer) {
            savedSlotsForArgRefresh.push_back({cb.crossSegmentSlotIdx, outputSlots_[cb.crossSegmentSlotIdx]});
            outputSlots_[cb.crossSegmentSlotIdx] = cb.buffer;
          }
        }
      }
      auto* tritonBackend = dynamic_cast<TritonGraphBackend*>(backend);
      if (tritonBackend != nullptr) {
        tritonBackend->refreshArgTablesForReplay(seg, externalArrays, numExt,
                                                 outputSlots_, totalOutputSlots_,
                                                 stream);
      }
      
      // CRITICAL FIX: After refreshing arg tables on host, copy to device BEFORE graph launch.
      // The captured graph has arg table addresses baked in - we just need to update the content.
      // This consolidated copy replaces ~N per-kernel cudaMemcpyAsync calls with ONE copy.
      if (tritonBackend != nullptr) {
        tritonBackend->copyConsolidatedArgTableToDevice(seg, stream);
      }
      
      for (auto& [extIdx, origArr] : savedForArgRefresh) {
        externalArrays[extIdx] = origArr;
      }
      for (auto& [slotIdx, origArr] : savedSlotsForArgRefresh) {
        outputSlots_[slotIdx] = origArr;
      }
    }
#endif
    // All H2D copies (ext input sync) and D2D copies (capture buffers) are on
    // cudaStr. Graph launch on cudaStr is ordered after them — no explicit sync needed.
    cudaGetLastError();  // Clear any sticky errors
    } // end standard replay path (else branch of useFastReplay)

    // DIAGNOSTIC: Zero capture workspace before replay to test stale-data hypothesis.
    // If zeroing the workspace fixes divergence, stale workspace data is the root cause.
    // This is gated on tritonVerifyKernels to avoid performance impact in production.
    if (Environment::getInstance().tritonVerifyKernels() &&
        seg.exec.replayHandle && seg.exec.replayHandle->getWorkspacePtr() != nullptr &&
        seg.exec.replayHandle->getWorkspaceBytes() > 0) {
      cudaMemsetAsync(seg.exec.replayHandle->getWorkspacePtr(), 0,
                      seg.exec.replayHandle->getWorkspaceBytes(), cudaStr);
      cudaStreamSynchronize(cudaStr);
      DSP_DIAG(VERIFY, "REPLAY_DIAG: zeroed capture workspace (%zuMB) before replay execCount=%d",
               seg.exec.replayHandle->getWorkspaceBytes() / (1024*1024), seg.exec.executionCount);
    }

    // DIAGNOSTIC: Dump specific VARIABLE external inputs before replay to trace stale data.
    if (Environment::getInstance().tritonVerifyKernels()) {
      cudaDeviceSynchronize();
      for (int ei = 0; ei < numExt; ei++) {
        if (ei < (int)externalInputIsVariable_.size() && externalInputIsVariable_[ei] &&
            externalArrays[ei] != nullptr && externalArrays[ei]->lengthOf() <= 8) {
          auto* arr = externalArrays[ei];
          auto* db = arr->dataBuffer();
          int n = std::min((int)arr->lengthOf(), 8);
          int elemSize = DataTypeUtils::sizeOf(arr->dataType());
          std::vector<uint8_t> hostBytes(n * elemSize), devBytes(n * elemSize);
          if (db && db->primary()) std::memcpy(hostBytes.data(), static_cast<char*>(arr->buffer()), n * elemSize);
          if (DSP_BUF(arr)) cudaMemcpy(devBytes.data(), DSP_BUF(arr), n * elemSize, cudaMemcpyDeviceToHost);
          float hv[8]={0}, dv[8]={0};
          dspBytesToFloat(hostBytes.data(), arr->dataType(), hv, n);
          dspBytesToFloat(devBytes.data(), arr->dataType(), dv, n);
          std::string name = (ei < (int)externalInputNames_.size()) ? externalInputNames_[ei] : "?";
          DSP_DIAG(VERIFY, "PRE_REPLAY ext#%d:\"%s\" len=%d pAct=%d sAct=%d host=[%.0f,%.0f,%.0f,%.0f] dev=[%.0f,%.0f,%.0f,%.0f]",
                    ei, name.c_str(), n,
                    db ? (db->isPrimaryActual()?1:0) : -1,
                    db ? (db->isSpecialActual()?1:0) : -1,
                    hv[0],hv[1],hv[2],hv[3], dv[0],dv[1],dv[2],dv[3]);
        }
      }
    }

    // Pre-replay batch-zero: zero all output buffers OUTSIDE the graph.
    // Individual cudaMemsetAsync calls use dedicated fill engines (not SMs),
    // pipeline efficiently, and add 0 graph nodes (they run before cudaGraphLaunch).
    // Stream ordering guarantees all zeroing completes before graph launch.
    // NOTE: Do NOT use batchZeroKernel here — it runs on SMs (competition with
    // compute kernels) and has alignment requirements that cause accuracy issues.
    // Use per-segment batch-zero entries (saved during capture) instead of the
    // shared batchZeroEntries_ which only contains the LAST captured segment's data.
    auto& segBZ = seg.exec.segBatchZeroEntries;
    if (Environment::getInstance().dspBatchZero() && !segBZ.empty()) {
      // Refresh batch-zero pointers from current slotArrayCache_ entries.
      // During frozen replay, the pre-exec restoration may be skipped (optimization),
      // but slotArrayCache_ entries persist with stable shapes. Re-derive the GPU
      // pointer from the authoritative source to avoid stale pointers that cause
      // CUDA error 700 (illegal memory access) during cudaMemsetAsync.
      for (auto& entry : segBZ) {
        if (entry.outputSlotIndex >= 0 && entry.outputSlotIndex < totalOutputSlots_) {
          NDArray* cached = slotArrayCache_[entry.outputSlotIndex];
          if (cached != nullptr && DSP_BUF(cached) != nullptr) {
            entry.ptr = DSP_BUF(cached);
            entry.bytes = static_cast<int>(cached->dataBuffer()->getLenInBytes());
          }
        }
      }
      for (auto& entry : segBZ) {
        if (entry.ptr != nullptr && entry.bytes > 0) {
          cudaMemsetAsync(entry.ptr, 0, entry.bytes, cudaStr);
        }
      }
      DSP_DIAG(MEMORY, "pre-replay batch-zero: %d buffers zeroed via cudaMemsetAsync (fill engines, outside graph) seg[%d-%d]",
                static_cast<int>(segBZ.size()), seg.startSlot, seg.endSlot);
    }

    // Replay strategy: configurable via ND4J_TRITON_GRAPH_REINSTANTIATE.
    // Default (OFF): direct replay of existing graphExec.
    // ON: destroy and re-instantiate graphExec from graph template before each replay.
    // Skip entirely if lineage validation or cross-segment size mismatch invalidated the graph.
    {
      // NOTE: Replay preserves the shared cuBLAS workspace. Capture zeroes it only
      // for the first captured segment in a fresh session; once graphs exist, later
      // captures and all replays preserve the accumulated plan/descriptor state.
      // Per-segment replay zeroing was destroying cuBLAS state that later segments
      // depend on but do not re-upload via explicit H2D nodes.

      // Pre-launch CUDA error check: detect accumulated errors from prior segments
      // that would manifest as hangs during this segment's cudaStreamSynchronize.
      {
        cudaError_t preLaunchErr = cudaPeekAtLastError();
        if (preLaunchErr != cudaSuccess) {
          DSP_DIAG(EXECUTE, "PRE_REPLAY_ERROR: seg[%d-%d] cudaPeekAtLastError=%d (%s) — clearing",
                   seg.startSlot, seg.endSlot, (int)preLaunchErr,
                   cudaGetErrorString(preLaunchErr));
          cudaGetLastError();  // clear it
        }
      }

      // ── TRIPWIRE: validate all pointers before graph launch ──────────────
      // NULL dereference in libcuda.so during cudaGraphLaunch means a kernel
      // arg or memcpy source/dest is NULL. Check everything we can reach.
      //
      // REPLAY OPTIMIZATION: Skip tripwire on stable replay (executionCount >= 4).
      // After 3+ successful replays, pointers are stable. Running the full
      // tripwire (~64 capture buffers + ~10 output slots + external inputs per
      // segment × 278 segments = ~20,000+ pointer checks) adds unnecessary
      // host-side overhead to every step. Only run during first few replays
      // and when verify mode is on.
      if (seg.exec.replayHandle && hasTritonCaptureBuffers &&
          (seg.exec.executionCount < 4 || Environment::getInstance().tritonVerifyKernels())) {
        int nullCapBufs = 0, nullCapBufDevPtrs = 0, nullDirectRefPtrs = 0;
        int totalCapBufs = 0;
        auto& tripCBs = seg.exec.replayHandle->getCaptureBuffers();
        for (size_t cbi = 0; cbi < tripCBs.size(); cbi++) {
          auto& cb = tripCBs[cbi];
          totalCapBufs++;
          if (cb.buffer == nullptr) {
            nullCapBufs++;
            DSP_DIAG(EXECUTE, "TRIPWIRE_NULL_CAPBUF: seg[%d-%d] cb[%zu] buffer=NULL "
                     "extIdx=%d crossSlot=%d directRef=%d",
                     seg.startSlot, seg.endSlot, cbi,
                     cb.externalInputIndex, cb.crossSegmentSlotIdx,
                     cb.directReference ? 1 : 0);
          } else if (DSP_BUF(cb.buffer) == nullptr) {
            nullCapBufDevPtrs++;
            DSP_DIAG(EXECUTE, "TRIPWIRE_NULL_CAPBUF_DEVPTR: seg[%d-%d] cb[%zu] "
                     "buffer=%p specialBuffer=NULL extIdx=%d crossSlot=%d "
                     "directRef=%d len=%lld dtype=%d",
                     seg.startSlot, seg.endSlot, cbi, (void*)cb.buffer,
                     cb.externalInputIndex, cb.crossSegmentSlotIdx,
                     cb.directReference ? 1 : 0,
                     (long long)cb.buffer->lengthOf(),
                     (int)cb.buffer->dataType());
          }
          if (cb.directReference && cb.buffer != nullptr &&
              DSP_BUF(cb.buffer) == nullptr) {
            nullDirectRefPtrs++;
            // This is the most dangerous case: directReference means the graph
            // reads from this buffer's device address directly
            DSP_DIAG(EXECUTE, "TRIPWIRE_NULL_DIRECTREF: seg[%d-%d] cb[%zu] "
                     "extIdx=%d — graph will dereference NULL!",
                     seg.startSlot, seg.endSlot, cbi, cb.externalInputIndex);
          }
        }
        // Check output slot device pointers for this segment's range
        int nullSlots = 0;
        for (int si = seg.startSlot; si <= seg.endSlot && si < numSlots_; si++) {
          for (int oi = 0; oi < slots_[si].numOutputs; oi++) {
            int slotIdx = slots_[si].outputSlotIndices[oi];
            if (slotIdx >= 0 && slotIdx < totalOutputSlots_) {
              NDArray* slotArr = slotArrayCache_[slotIdx];
              if (slotArr == nullptr || DSP_BUF(slotArr) == nullptr) {
                nullSlots++;
                DSP_DIAG(EXECUTE, "TRIPWIRE_NULL_SLOT: seg[%d-%d] step=%d "
                         "outputSlot=%d arr=%p devPtr=%p",
                         seg.startSlot, seg.endSlot, si, slotIdx,
                         (void*)slotArr,
                         slotArr ? DSP_BUF(slotArr) : nullptr);
              }
            }
          }
        }
        // Check workspace pointer
        void* wsPtr = seg.exec.replayHandle->getWorkspacePtr();
        size_t wsBytes = seg.exec.replayHandle->getWorkspaceBytes();
        if (wsBytes > 0 && wsPtr == nullptr) {
          DSP_DIAG(EXECUTE, "TRIPWIRE_NULL_WORKSPACE: seg[%d-%d] wsBytes=%zu "
                   "wsPtr=NULL — graph H2D nodes will crash!",
                   seg.startSlot, seg.endSlot, wsBytes);
        }
        // Check captured host pointers
        auto& hostPtrs = seg.exec.replayHandle->getCapturedHostPtrs();
        int nullHostPtrs = 0;
        for (size_t hi = 0; hi < hostPtrs.size(); hi++) {
          if (hostPtrs[hi] == nullptr) {
            nullHostPtrs++;
            DSP_DIAG(EXECUTE, "TRIPWIRE_NULL_HOSTPTR: seg[%d-%d] hostPtr[%zu]=NULL",
                     seg.startSlot, seg.endSlot, hi);
          }
        }
        // Check key external inputs that the segment uses
        int nullExtInputs = 0;
        for (int ei = 0; ei < numExt; ei++) {
          if (externalArrays[ei] != nullptr &&
              DSP_BUF(externalArrays[ei]) == nullptr) {
            nullExtInputs++;
            if (nullExtInputs <= 5) {
              std::string name = (ei < (int)externalInputNames_.size())
                                 ? externalInputNames_[ei] : "?";
              DSP_DIAG(EXECUTE, "TRIPWIRE_NULL_EXT_DEVPTR: seg[%d-%d] ext[%d]=\"%s\" "
                       "len=%lld dtype=%d — device pointer is NULL",
                       seg.startSlot, seg.endSlot, ei, name.c_str(),
                       (long long)externalArrays[ei]->lengthOf(),
                       (int)externalArrays[ei]->dataType());
            }
          }
        }
        // Summary
        if (nullCapBufs > 0 || nullCapBufDevPtrs > 0 || nullDirectRefPtrs > 0 ||
            nullSlots > 0 || nullHostPtrs > 0 || nullExtInputs > 0 ||
            (wsBytes > 0 && wsPtr == nullptr)) {
          DSP_DIAG(EXECUTE, "TRIPWIRE_SUMMARY: seg[%d-%d] DANGER — "
                   "nullCapBufs=%d nullCapBufDevPtrs=%d nullDirectRef=%d "
                   "nullSlots=%d nullHostPtrs=%d nullExtDevPtrs=%d "
                   "wsPtr=%p wsBytes=%zu totalCapBufs=%d",
                   seg.startSlot, seg.endSlot,
                   nullCapBufs, nullCapBufDevPtrs, nullDirectRefPtrs,
                   nullSlots, nullHostPtrs, nullExtInputs,
                   wsPtr, wsBytes, totalCapBufs);
        } else {
          DSP_DIAG(EXECUTE, "TRIPWIRE_OK: seg[%d-%d] all %d capBufs, "
                   "%d hostPtrs, ws=%p/%zuMB — no NULL pointers detected",
                   seg.startSlot, seg.endSlot, totalCapBufs,
                   (int)hostPtrs.size(), wsPtr, wsBytes / (1024*1024));
        }
      }
      // ── END TRIPWIRE ─────────────────────────────────────────────────────

      // Address fingerprinting: detect slot output GPU address changes.
      // Slot addresses CAN change between capture and replay (e.g., when
      // releaseGpuIntermediates frees warmup arrays and the pool recycles
      // addresses). This is handled by the capture buffer D2D refresh and
      // refreshArgTablesForReplay which update the consolidated arg table
      // with current addresses before each replay. The fingerprint is
      // logged for diagnostics but does NOT invalidate the graph.
      //
      // REPLAY OPTIMIZATION: Skip fingerprinting during stable replay
      // (executionCount >= 4 with argTableStable). In frozen replay, buffer
      // addresses are persistent — they never get freed/reallocated. The hash
      // computation iterates all output slots in the segment range, adding
      // host-side overhead per segment per step.
      if (seg.exec.capturedSlotAddrHash != 0 &&
          (seg.exec.executionCount < 4 || !seg.exec.argTableStable)) {
        LongType currentAddrHash = computeSlotAddrHash(
            outputSlots_, seg.startSlot, seg.endSlot, totalOutputSlots_);
        if (currentAddrHash != seg.exec.capturedSlotAddrHash) {
          DSP_DIAG(MEMORY, "SLOT ADDRESS DRIFT for seg[%d-%d]: "
                   "captured=0x%llx current=0x%llx — arg table refresh will handle",
                   seg.startSlot, seg.endSlot,
                   (long long)seg.exec.capturedSlotAddrHash, (long long)currentAddrHash);
          seg.exec.capturedSlotAddrHash = currentAddrHash;
        }
      }

      bool replayOk = false;
      if (lineageInvalidated || crossSegSizeMismatch || !seg.exec.replayHandle) {
        // Graph was invalidated — skip replay, fall through to re-capture/slot-by-slot
        DSP_DIAG(EXECUTE, "REPLAY_SKIPPED: lineage=%d sizeMismatch=%d handle=%p seg[%d-%d]",
                 lineageInvalidated ? 1 : 0, crossSegSizeMismatch ? 1 : 0,
                 (void*)seg.exec.replayHandle.get(), seg.startSlot, seg.endSlot);
      } else if (Environment::getInstance().tritonGraphReinstantiate()) {
        DSP_DIAG_SEG(EXECUTE, seg.startSlot,
                     "seg[%d-%d] REPLAY via reInstantiate path (execCount=%d replays=%d)",
                     seg.startSlot, seg.endSlot, seg.exec.executionCount,
                     seg.exec.replayHandle->getStatistics().replayCount);
        auto* cudaReplay = static_cast<CudaGraphReplayHandle*>(seg.exec.replayHandle.get());
        if (!cudaReplay->getNativeHandle()->reInstantiate()) {
          DSP_DIAG_SEG(EXECUTE, seg.startSlot, "Triton graph reInstantiate FAILED for seg[%d-%d]",
                    seg.startSlot, seg.endSlot);
        } else {
          replayOk = seg.exec.replayHandle->replay(stream);
          DSP_DIAG_SEG(EXECUTE, seg.startSlot,
                       "seg[%d-%d] reInstantiate replay %s",
                       seg.startSlot, seg.endSlot, replayOk ? "OK" : "FAILED");
        }
      } else {
        DSP_DIAG_SEG(EXECUTE, seg.startSlot,
                     "seg[%d-%d] REPLAY via direct path (execCount=%d replays=%d)",
                     seg.startSlot, seg.endSlot, seg.exec.executionCount,
                     seg.exec.replayHandle->getStatistics().replayCount);
        replayOk = seg.exec.replayHandle->replay(stream);
        DSP_DIAG_SEG(EXECUTE, seg.startSlot,
                     "seg[%d-%d] direct replay %s",
                     seg.startSlot, seg.endSlot, replayOk ? "OK" : "FAILED");
      }
      if (replayOk) {
        // LRU tracking: record when this segment was last replayed for eviction ordering
        seg.exec.lastReplayExecCount = executeCount_;

        // Find the ACTUAL final output slot index (not the step index)
        int finalOutputSlot = -1;
        if (seg.endSlot < numSlots_ && slots_[seg.endSlot].numOutputs > 0) {
          finalOutputSlot = slots_[seg.endSlot].outputSlotIndices[0];
        }
        // Fallback to seg.endSlot if output slot lookup fails
        if (finalOutputSlot < 0 || finalOutputSlot >= totalOutputSlots_) {
          finalOutputSlot = seg.endSlot;
        }

        // ── Timed sync with 30s timeout — BEFORE any D2H diagnostic copies ──
        // DSP_DIAG_DUMP_SLOT and DSP_DIAG_DUMP_SEG_OUTPUT internally call
        // cudaStreamSynchronize via safeDtoH(). If the GPU is hung, those calls
        // block forever. By syncing here first with a timeout, we can detect
        // and report the hang instead of blocking.
        bool replaySyncOk = true;
        if (DSP_DIAG_ENABLED(EXECUTE)) {
          // Graph node stats for the replayed graph
          auto* cudaReplayForDiag = dynamic_cast<CudaGraphReplayHandle*>(seg.exec.replayHandle.get());
          if (cudaReplayForDiag && cudaReplayForDiag->getNativeHandle()) {
            auto stats = cudaReplayForDiag->getNativeHandle()->getStatistics();
            DSP_DIAG(EXECUTE, "REPLAY_GRAPH_STATS: seg[%d-%d] kernels=%d memcpyH2D=%d memsets=%d "
                     "memAllocs=%d memFrees=%d hostCbs=%d childGraphs=%d totalNodes=%zu",
                     seg.startSlot, seg.endSlot,
                     stats.numKernels, stats.numMemcpyH2D, stats.numMemsets,
                     stats.numMemAllocs, stats.numMemFrees, stats.numHostCallbacks,
                     stats.numChildGraphs, cudaReplayForDiag->getNativeHandle()->getNumNodes());
          }
          fflush(stdout); fflush(stderr);

          // Timed sync: use event polling with 30s timeout
          cudaEvent_t syncEvt;
          cudaEventCreateWithFlags(&syncEvt, cudaEventDisableTiming);
          cudaEventRecord(syncEvt, cudaStr);

          auto syncStart = std::chrono::steady_clock::now();
          const int timeoutSec = 30;
          while (true) {
            cudaError_t evtErr = cudaEventQuery(syncEvt);
            if (evtErr == cudaSuccess) {
              break;
            } else if (evtErr == cudaErrorNotReady) {
              auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                  std::chrono::steady_clock::now() - syncStart).count();
              if (elapsed >= timeoutSec) {
                replaySyncOk = false;
                DSP_DIAG(EXECUTE, "GPU_HANG_DETECTED: seg[%d-%d] cudaEventQuery not ready after %ds "
                         "— graph replay stuck! execCount=%d replays=%d",
                         seg.startSlot, seg.endSlot, timeoutSec,
                         seg.exec.executionCount, seg.exec.replayHandle->getStatistics().replayCount);
                // Check for CUDA errors that might explain the hang
                cudaError_t hangErr = cudaPeekAtLastError();
                if (hangErr != cudaSuccess) {
                  DSP_DIAG(EXECUTE, "GPU_HANG_CUDA_ERROR: %d (%s)", (int)hangErr,
                           cudaGetErrorString(hangErr));
                }
                // Log GPU memory state
                size_t freeMem = 0, totalMem = 0;
                cudaMemGetInfo(&freeMem, &totalMem);
                DSP_DIAG(EXECUTE, "GPU_HANG_MEM: free=%zuMB total=%zuMB used=%zuMB",
                         freeMem/(1024*1024), totalMem/(1024*1024),
                         (totalMem-freeMem)/(1024*1024));
                fflush(stdout); fflush(stderr);
                // Fatal: graph replay hang means GPU is stuck. Continuing
                // produces garbage and may cascade into further hangs.
                {
                  std::string msg = "CUDA graph replay hung for seg[" +
                      std::to_string(seg.startSlot) + "-" +
                      std::to_string(seg.endSlot) + "] after " +
                      std::to_string(timeoutSec) + "s — aborting execution";
                  THROW_EXCEPTION(msg.c_str());
                }
              }
              std::this_thread::sleep_for(std::chrono::milliseconds(1));
            } else {
              DSP_DIAG(EXECUTE, "REPLAY_SYNC_ERROR: seg[%d-%d] cudaEventQuery returned %d (%s)",
                       seg.startSlot, seg.endSlot, (int)evtErr, cudaGetErrorString(evtErr));
              // Fatal: CUDA error during graph replay means the graph is corrupt
              {
                std::string msg = "CUDA graph replay error for seg[" +
                    std::to_string(seg.startSlot) + "-" +
                    std::to_string(seg.endSlot) + "]: cudaEventQuery returned " +
                    std::to_string((int)evtErr) + " (" +
                    cudaGetErrorString(evtErr) + ")";
                THROW_EXCEPTION(msg.c_str());
              }
            }
          }
          cudaEventDestroy(syncEvt);
        }

        // Only do D2H diagnostic copies if sync succeeded (GPU not hung)
        if (replaySyncOk && finalOutputSlot >= 0 && finalOutputSlot < totalOutputSlots_ &&
            outputSlots_[finalOutputSlot] != nullptr) {
          auto* finalOut = outputSlots_[finalOutputSlot];
          if (finalOut->dataType() == FLOAT32) {
            DSP_DIAG_DUMP_SLOT("replay", finalOutputSlot,
                               DSP_BUF(finalOut), finalOut->lengthOf());
          }
          if (finalOut->dataType() == FLOAT32 && finalOut->lengthOf() > 0) {
            DSP_DIAG_DUMP_SEG_OUTPUT("GRAPH_REPLAY", finalOutputSlot, DSP_BUF(finalOut),
                                     finalOut->lengthOf(), seg.exec.executionCount, stream);
          }
          if (DSP_DIAG_ENABLED(EXECUTE)) {
            int replayArgmax = dspArgmax(DSP_BUF(finalOut), finalOut->dataType(),
                                         finalOut->lengthOf());
            std::string firstVals = dspDumpSlotValues(DSP_BUF(finalOut), finalOut->dataType(),
                                                       finalOut->lengthOf(), 4);
            DSP_DIAG(EXECUTE, "GRAPH_REPLAY ARGMAX: slot=%d argmax=%d len=%lld vals=%s execCount=%d",
                     finalOutputSlot, replayArgmax, (long long)finalOut->lengthOf(),
                     firstVals.c_str(), seg.exec.executionCount);
          }
        }

        seg.exec.executionCount++;
        totalGraphReplays_++;

        // ── REPLAY VERIFICATION ─────────────────────────────────────────
        if (replaySyncOk && Environment::getInstance().tritonVerifyKernels()) {
          cudaStreamSynchronize(cudaStr);
          performReplayVerify(seg, externalArrays, numExt, stream, "TRITON");
        }

        // Force re-capture every step (diagnostic mode).
        // Invalidates the cached graph after each replay so the next step
        // re-captures with fresh data.  Correct but slow.
        if (Environment::getInstance().tritonForceRecapture()) {
          platformCleanupSegmentForRebuild(seg);
          seg.exec.argTableStable = false;
          batchD2DCount_ = 0;
          seg.exec.capturedInputAddrKey = 0;
          DSP_DIAG(EXECUTE, "FORCE_RECAPTURE: invalidated after replay execCount=%d", seg.exec.executionCount);
        }

        // ── Re-execute GAP ops after graph replay only when they were not captured ──
        // Triton fallback ranges can run either outside CUDA graph capture
        // (re-execute them here) or while capture is active (already part of
        // the graph, so running them again is incorrect).
        if (!replaySyncOk) {
          DSP_DIAG(EXECUTE, "SKIP_GAP_REEXEC: seg[%d-%d] replay sync failed, skipping gap ops",
                   seg.startSlot, seg.endSlot);
        }
        if (replaySyncOk) {
#if HAVE_TRITON && defined(SD_CUDA)
          auto* tritonBE = dynamic_cast<TritonGraphBackend*>(backend);
          if (tritonBE != nullptr) {
            if (seg.exec.gapOpsCapturedInGraph) {
              DSP_DIAG(EXECUTE, "POST_REPLAY_GAP: seg[%d-%d] skipped because fallback gap ops were captured in the graph",
                       seg.startSlot, seg.endSlot);
            } else {
              auto gapSlots = tritonBE->getGapSlots(seg, slots_);
              if (!gapSlots.empty()) {
                // Sync Triton stream before gap ops (they run on default stream)
                cudaStreamSynchronize(cudaStr);

                // Build contiguous ranges from gap slots for efficient execution
                std::vector<int> sortedGaps(gapSlots.begin(), gapSlots.end());
                std::sort(sortedGaps.begin(), sortedGaps.end());
                int rangeStart = sortedGaps[0];
                int rangeEnd = sortedGaps[0];
                auto executeGapRange = [&](int start, int end) -> Status {
                  GraphSegment gapSeg;
                  gapSeg.startSlot = start;
                  gapSeg.endSlot = end;
                  gapSeg.exec.executionCount = seg.exec.executionCount;
                  gapSeg.exec.compilationFailed = true;  // Never capture gap ops themselves
                  return executeSegmentSlotBySlot(gapSeg, externalArrays, numExt, stream);
                };
                Status gapStatus = Status::OK;
                for (size_t i = 1; i < sortedGaps.size() && gapStatus == Status::OK; i++) {
                  if (sortedGaps[i] == rangeEnd + 1) {
                    rangeEnd = sortedGaps[i];
                  } else {
                    gapStatus = executeGapRange(rangeStart, rangeEnd);
                    rangeStart = sortedGaps[i];
                    rangeEnd = sortedGaps[i];
                  }
                }
                if (gapStatus == Status::OK) {
                  gapStatus = executeGapRange(rangeStart, rangeEnd);
                }
                DSP_DIAG(EXECUTE, "POST_REPLAY_GAP: seg[%d-%d] executed %d gap slots (%d ranges) status=%d",
                         seg.startSlot, seg.endSlot, static_cast<int>(gapSlots.size()),
                         static_cast<int>(sortedGaps.size()), static_cast<int>(gapStatus));
                if (gapStatus != Status::OK) {
                  DSP_DIAG(FALLBACK, "POST_REPLAY_GAP: gap ops failed for seg[%d-%d] status=%d",
                           seg.startSlot, seg.endSlot, static_cast<int>(gapStatus));
                  return gapStatus;
                }
              }
            }
          }
#endif  // HAVE_TRITON
        }

        // Phase 2: slotArrayCache_ == outputSlots_ (unified).
        // Post-replay restoration is a no-op — arrays are already in place.

        if (Environment::getInstance().tritonVerifyKernels()) {
          DSP_DIAG(VERIFY, "SEG_EXIT seg[%d-%d] status=OK(replay) execCount=%d",
                    seg.startSlot, seg.endSlot, seg.exec.executionCount);
        }
        return Status::OK;
      }
      // Launch failed — this is a fatal error. Graph replay failure means
      // the captured graph is corrupt or the CUDA runtime is in a bad state.
      {
        int deviceId = 0;
        cudaGetDevice(&deviceId);
        platformCleanupSegmentForRebuild(seg);
        return reportReplayError(seg, "graph_replay", cudaGetLastError(), deviceId);
      }
    }
  }

  if (allowTritonCudaGraphReplay &&
      (!seg.exec.replayHandle || seg.exec.replayHandle->getCaptureBuffers().empty()) &&
      seg.exec.replayHandle != nullptr &&
      seg.exec.cachedShapeKey == segShapeKey &&
      !extAddrsStable) {
    platformCleanupSegmentForRebuild(seg);
    seg.exec.argTableStable = false;
    batchD2DCount_ = 0;
    seg.exec.capturedInputAddrKey = 0;
  }

  if (allowTritonCudaGraphReplay &&
      (!seg.exec.replayHandle || seg.exec.replayHandle->getCaptureBuffers().empty()) &&
      seg.exec.replayHandle != nullptr &&
      seg.exec.cachedShapeKey != segShapeKey) {
    platformCleanupSegmentForRebuild(seg);
    seg.exec.argTableStable = false;
    batchD2DCount_ = 0;
    seg.exec.capturedInputAddrKey = 0;
  }
#endif

#if HAVE_TRITON && defined(SD_CUDA)
  struct TritonFallbackGuard {
    bool active = false;
    ~TritonFallbackGuard() {
      if (active) TritonGraphBackend::clearFallbackRangeExecutor();
    }
  } tritonFallbackGuard;

  auto* tritonBackend = dynamic_cast<TritonGraphBackend*>(backend);
  if (tritonBackend != nullptr) {
    TritonGraphBackend::setFallbackRangeExecutor(
        [this, &seg, externalArrays, numExt, stream](int startSlot, int endSlot) -> Status {
          if (startSlot > endSlot) return Status::OK;

          GraphSegment gapSeg;
          gapSeg.startSlot = startSlot;
          gapSeg.endSlot = endSlot;
          gapSeg.exec.executionCount = seg.exec.executionCount;
          gapSeg.exec.compilationFailed = seg.exec.compilationFailed;

          // Check if the stream is currently being captured (CUDA graph recording).
          // During capture: keep tl_graphExecutionActive=true so fallback ops use the
          // pre-allocated capture workspace for any allocations. The workspace must be
          // set up before beginCapture (see shouldCaptureTritonGraph block below).
          // Outside capture: set tl_graphExecutionActive=false so fallback ops use
          // normal allocation paths (cudaMallocAsync) and sync guards work normally.
          bool streamIsCapturing = false;
#ifdef SD_CUDA
          if (stream != nullptr) {
            cudaStreamCaptureStatus capStat = cudaStreamCaptureStatusNone;
            cudaStreamIsCapturing(*static_cast<cudaStream_t*>(stream), &capStat);
            streamIsCapturing = (capStat != cudaStreamCaptureStatusNone);
          }

          // Synchronize between the Triton execution stream and the gap ops' stream.
          // Triton kernels use the explicit stream parameter; native fallback ops use
          // the thread-local LaunchContext stream (a different CUDA stream). Without
          // synchronization, gap ops can read stale data from before the preceding
          // Triton kernel completes, and subsequent Triton kernels can read stale
          // gap op outputs.
          //
          // Outside capture: use cudaStreamSynchronize (simple, no overhead concern
          // since gap ops are already the bottleneck).
          // During capture: use CUDA events to create graph dependency edges between
          // the capture stream and the gap ops' stream. cudaStreamSynchronize cannot
          // be used during capture.
          cudaStream_t tritonStr = *static_cast<cudaStream_t*>(stream);
          auto* lcStream = LaunchContext::defaultContext()->getCudaStream();
          cudaStream_t gapStr = lcStream ? *lcStream : nullptr;
          bool streamsMatch = (tritonStr == gapStr);

          // One-time diagnostic: log whether streams match
          static bool streamDiagDone = false;
          if (!streamDiagDone) {
            DSP_DIAG(BACKEND, "stream diag: tritonStr=%p gapStr=%p match=%d capturing=%d",
                     (void*)tritonStr, (void*)gapStr, streamsMatch ? 1 : 0,
                     streamIsCapturing ? 1 : 0);
            streamDiagDone = true;
          }

          if (!streamIsCapturing && stream != nullptr) {
            if (!streamsMatch) {
              cudaStreamSynchronize(tritonStr);
            }
          } else if (streamIsCapturing && !streamsMatch && gapStr != nullptr) {
            // During capture: record event on Triton stream, make gap stream wait.
            // This creates a dependency edge in the CUDA graph.
            cudaEvent_t syncEvent;
            cudaEventCreateWithFlags(&syncEvent, cudaEventDisableTiming);
            cudaEventRecord(syncEvent, tritonStr);
            cudaStreamWaitEvent(gapStr, syncEvent, 0);
            cudaEventDestroy(syncEvent);
          }
#endif
          if (streamIsCapturing) {
            // SKIP gap execution during capture. Gap ops (shape_of, reshape, etc.)
            // are not part of the Triton kernel — they must NOT execute on the
            // capture stream. Their writes to outputSlots_[] would get baked into
            // the CUDA graph, and on replay those baked writes would produce stale
            // data. Instead, gap ops re-execute BEFORE each replay (the existing
            // !gapOpsCapturedInGraph replay path handles this).
            // outputSlots_[] retains the warmup values — the arg table already
            // references these addresses.
            seg.exec.gapOpsCapturedInGraph = false;
            return Status::OK;
          }
          bool savedGraphActive = tl_graphExecutionActive;
          tl_graphExecutionActive = false;
          auto gapStatus = executeSegmentSlotBySlot(gapSeg, externalArrays, numExt, stream);
#ifdef SD_CUDA
          if (!streamsMatch && gapStr != nullptr) {
            cudaStreamSynchronize(gapStr);
          }
#endif
          tl_graphExecutionActive = savedGraphActive;
          return gapStatus;
        });
    tritonFallbackGuard.active = true;
  }
#endif

  Status status = Status::KERNEL_FAILURE;
  bool usedTritonGraphCapture = false;

#ifdef SD_CUDA
  // Recompute shouldCaptureTritonGraph here (same logic as CAPTURE DECISION CHECK above)
  // This is the actual capture point - the diagnostic above just logs the decision.
  bool hasReplayHandleNow = (seg.exec.replayHandle != nullptr);
  bool hasCaptureBuffersNow = hasReplayHandleNow && !seg.exec.replayHandle->getCaptureBuffers().empty();
  bool replayHandleNullNow = (seg.exec.replayHandle == nullptr);
  bool execCountInWindowNow = (seg.exec.executionCount >= captureMinExec) &&
                              (forceRecaptureEnabled || seg.exec.executionCount <= (captureMinExec + 2));
  bool shouldCaptureTritonGraphNow = allowTritonCudaGraphReplay &&
                                     (!hasReplayHandleNow || !hasCaptureBuffersNow) &&
                                     replayHandleNullNow &&
                                     !seg.exec.compilationFailed &&
                                     execCountInWindowNow &&
                                     hasCudaStream;
  // OOM retry deferred check: if a previous capture attempt failed with OOM and
  // we haven't reached the retry-after execution count, skip capture and fall
  // through to slot-by-slot execution (same pattern as cudagraph.cu).
  if (seg.exec.captureOomRetries > 0 &&
      seg.exec.executionCount < seg.exec.captureRetryAfterExec) {
    DSP_DIAG_SEG(EXECUTE, seg.startSlot,
                 "OOM RETRY DEFERRED: seg[%d-%d] retries=%d execCount=%d retryAfter=%d — slot-by-slot",
                 seg.startSlot, seg.endSlot, seg.exec.captureOomRetries,
                 seg.exec.executionCount, seg.exec.captureRetryAfterExec);
    shouldCaptureTritonGraphNow = false;
  }

  // Proactive memory cleanup before capture: trim pool, evict LRU graphs if needed.
  if (shouldCaptureTritonGraphNow && Environment::getInstance().dspProactiveEvictBeforeCapture()) {
    proactivePreCaptureMemoryCleanup(seg, segIdx, stream);
  }

  if (shouldCaptureTritonGraphNow) {
    DSP_DIAG_SEG(COMPILE, seg.startSlot,
                 "GRAPH CAPTURE BEGIN: seg[%d-%d] size=%d execCount=%d shapesFrozen=%d",
                 seg.startSlot, seg.endSlot, seg.endSlot - seg.startSlot + 1,
                 seg.exec.executionCount, shapesFrozen_ ? 1 : 0);
    seg.exec.gapOpsCapturedInGraph = false;

    // Set up capture workspace BEFORE beginCapture — cudaMalloc must be outside capture.
    // Fallback ops (matmul, attention, concat) need temporary buffers during execution.
    // With tl_graphExecutionActive=true, CudaMemoryPool allocates from this workspace
    // instead of calling cudaMallocAsync (which fails during capture).
    // TRITON_CAPTURE_WORKSPACE_SIZE is now at file scope (above).

    // Create the replayHandle BEFORE capture — it must exist to store workspace, host ptrs, etc.
    {
      int deviceId = 0;
      cudaGetDevice(&deviceId);
      seg.exec.replayHandle = GraphReplayFactory::create(deviceId);
    }

    if (seg.exec.replayHandle->getWorkspacePtr() == nullptr) {
      int deviceId = 0;
      cudaGetDevice(&deviceId);

      // Shared workspace: allocate once, reuse across all segments.
      // Segments execute sequentially (cudaGraphLaunch + cudaStreamSynchronize),
      // and workspace offset resets each capture, so sharing is safe.
      if (sharedCaptureWorkspace_ == nullptr) {
        // First segment — allocate the shared workspace
        if (captureBufferRegistry_ != nullptr) {
          auto* registry = static_cast<CaptureBufferRegistry*>(captureBufferRegistry_);
          sharedCaptureWorkspace_ = registry->allocate(-1, TRITON_CAPTURE_WORKSPACE_SIZE, deviceId);
        }
        if (sharedCaptureWorkspace_ == nullptr) {
          // Fallback to raw cudaMalloc
          cudaError_t err = cudaMalloc(&sharedCaptureWorkspace_, TRITON_CAPTURE_WORKSPACE_SIZE);
          if (err != cudaSuccess) {
            cudaGetLastError();
            sharedCaptureWorkspace_ = nullptr;
          }
        }
        if (sharedCaptureWorkspace_ != nullptr) {
          sharedCaptureWorkspaceBytes_ = TRITON_CAPTURE_WORKSPACE_SIZE;
          sharedCaptureWorkspaceDevice_ = deviceId;
          memory::CudaMemoryPool::getInstance().registerCaptureWorkspace(
              sharedCaptureWorkspace_, sharedCaptureWorkspaceBytes_);
          DSP_DIAG_SEG(MEMORY, seg.startSlot,
                    "allocated SHARED capture workspace: %zuMB on device %d",
                    TRITON_CAPTURE_WORKSPACE_SIZE / (1024*1024), deviceId);
        } else {
          // Shared allocation failed — ABORT capture for this segment.
          platformCleanupSegmentForRebuild(seg);
          return reportOomError(seg, "shared_workspace_allocation",
                                TRITON_CAPTURE_WORKSPACE_SIZE, deviceId);
        }
      } else {
        DSP_DIAG_SEG(MEMORY, seg.startSlot,
                  "using shared workspace for seg[%d-%d]",
                  seg.startSlot, seg.endSlot);
      }

      // Point this segment's replay handle at the shared workspace
      seg.exec.replayHandle->useExternalWorkspace(
          sharedCaptureWorkspace_, sharedCaptureWorkspaceBytes_);
    }

    // Guard: if workspace allocation failed, replayHandle is now nullptr.
    // Skip ALL capture setup and execution — fall through to slot-by-slot path.
    if (seg.exec.replayHandle == nullptr) {
      // No capture — ensure usedTritonGraphCapture stays false
      // and skip the rest of the capture block.
    } else {
    tl_captureWorkspace = seg.exec.replayHandle->getWorkspacePtr();
    tl_captureWorkspaceSize = seg.exec.replayHandle->getWorkspaceBytes();
    tl_captureWorkspaceOffset = 0;
    tl_capturedHostPtrs.clear();
    tl_captureReplicateCache.clear();

    // Allocate pinned host workspace for H2D source copies during capture.
    // During capture, DataBuffer::syncToSpecial and PointersManager need a persistent
    // pinned buffer as H2D memcpy source. Without this, they use _primaryBuffer directly.
    // Temporary arrays (axis/dimension params for gap ops) get freed after the op completes,
    // but the graph's H2D memcpy node bakes the source address — reading freed memory on
    // launch causes SIGSEGV. The pinned workspace persists for the graph's lifetime.
    void* captureHostWs = nullptr;
    {
      auto hostWsErr = cudaMallocHost(&captureHostWs, TRITON_CAPTURE_HOST_WORKSPACE_SIZE);
      if (hostWsErr != cudaSuccess) {
        cudaGetLastError();
        captureHostWs = nullptr;
        DSP_DIAG(FALLBACK, "Triton capture host workspace alloc failed (%zuMB), "
                  "H2D copies may use non-pinned sources",
                  TRITON_CAPTURE_HOST_WORKSPACE_SIZE / (1024*1024));
      } else {
        DSP_DIAG(MEMORY, "allocated %zuMB pinned host workspace for Triton capture seg[%d-%d]",
                  TRITON_CAPTURE_HOST_WORKSPACE_SIZE / (1024*1024), seg.startSlot, seg.endSlot);
      }
    }
    tl_captureHostWorkspace = captureHostWs;
    tl_captureHostWorkspaceSize = (captureHostWs != nullptr) ? TRITON_CAPTURE_HOST_WORKSPACE_SIZE : 0;
    tl_captureHostWorkspaceOffset = 0;
    // Track the host workspace as a captured host pointer for lifetime management.
    // On successful capture, this moves to the replay handle (addCapturedHostPtr).
    // On failure, it's freed immediately (line below at tl_capturedHostPtrs cleanup).
    if (captureHostWs != nullptr) {
      tl_capturedHostPtrs.push_back(captureHostWs);
    }

    // Set capture stream so captureSafeStreamOrDefault() routes ops to the correct stream
    cudaStream_t prevCaptureStream = tl_graphCaptureStream;
    tl_graphCaptureStream = cudaStr;

    // Pre-allocate cuBLAS workspace to prevent internal cudaMalloc during capture.
    // cuBLAS internally allocates workspace on stream 0 for GEMM operations. During
    // graph capture on a named stream, this cross-stream allocation breaks capture,
    // producing invalid graph nodes that SIGSEGV on cudaGraphLaunch.
    const size_t CUBLAS_WORKSPACE_SIZE = Environment::getInstance().dspCublasWorkspaceMb() * 1024ULL * 1024ULL;
    ensureCublasWorkspace(CUBLAS_WORKSPACE_SIZE);
    // NOTE: setCublasWorkspaceForCapture is deferred to AFTER warmup (see below).
    // Calling it here sets cublasSetStream_v2 to the capture stream, which causes
    // cuBLAS matmuls in gap ops during warmup to run on tritonStr instead of gapStr.
    // This stream mismatch creates data races: cast ops on gapStr write matmul
    // inputs, but cuBLAS on tritonStr starts before gapStr completes.

    // Reset cast cache indices (NOT full clear) before warmup.
    // Previous segments' warmup entries may still be referenced by their captured
    // graphs — clearCastCache() would delete the NDArrays, causing cudaFreeAsync
    // on the GPU buffers. Those addresses are baked into the captured graph nodes
    // (assign + cuBLAS GEMM), so freeing them causes "illegal memory access" (700)
    // on replay. resetCastCacheIndices() preserves the buffers while letting this
    // segment's warmup reuse or append entries as needed.
    //
    // Note: Shape mismatches from speculative decode (draft vs target model) are
    // handled by the mid-execution clearCastCache() call inside MmulHelper::mmul
    // (lines ~761, 1058), which safely skips frees during capture.
    MmulHelper::resetCastCacheIndices();

    // ── Batch-zero preparation (OUTSIDE capture) ─────────────────────────
    // Use the registration-based approach: batchZeroEntries_ was populated
    // by finishBatchZeroRegistration() during the warmup execution (execCount==1).
    // This contains ONLY the buffers that were actually nullified during warmup,
    // avoiding the ~143 extra buffers that collectBatchZeroTargets() would include
    // for slots that don't actually execute (identity ops, fused chains, etc.).
    //
    // If registration didn't happen (e.g., capture retry), fall back to
    // collectBatchZeroTargets for the pre-scan approach.
    if (Environment::getInstance().dspBatchZero()) {
      if (!batchZeroEntries_.empty()) {
        // Registration-based: entries already populated from warmup
        DSP_DIAG(MEMORY, "batch-zero using %d REGISTERED buffers (from warmup observation)",
                  static_cast<int>(batchZeroEntries_.size()));
      } else {
        // Fallback: pre-scan approach (may include extra buffers)
        DSP_DIAG(MEMORY, "batch-zero registration empty, falling back to collectBatchZeroTargets");
        std::unordered_set<int> gapSlots;
        if (Environment::getInstance().dspBatchZeroGapOnly()) {
#if HAVE_TRITON && defined(SD_CUDA)
          auto* tritonBE = dynamic_cast<TritonGraphBackend*>(backend);
          if (tritonBE != nullptr) {
            gapSlots = tritonBE->getGapSlots(seg, slots_);
          } else
#endif
          {
            for (int s = seg.startSlot; s <= seg.endSlot; s++) gapSlots.insert(s);
          }
        } else {
          for (int s = seg.startSlot; s <= seg.endSlot; s++) gapSlots.insert(s);
        }
        collectBatchZeroTargets(gapSlots);
      }
      prepareBatchZeroDevice(cudaStr);

      // Save per-segment batch-zero entries so replay uses THIS segment's
      // entries instead of the shared batchZeroEntries_ (which gets overwritten
      // by subsequent segments' warmup/capture cycles).
      seg.exec.segBatchZeroEntries.clear();
      seg.exec.segBatchZeroEntries.reserve(batchZeroEntries_.size());
      for (auto& e : batchZeroEntries_) {
        seg.exec.segBatchZeroEntries.push_back({e.ptr, e.bytes, e.outputSlotIndex});
      }
      DSP_DIAG(MEMORY, "saved %d batch-zero entries to seg[%d-%d]",
                static_cast<int>(seg.exec.segBatchZeroEntries.size()),
                seg.startSlot, seg.endSlot);
    }

    // Sync external inputs to device before capture — same rationale as non-capture path.
    // Java may have modified host buffers (putScalar + tagLocation(HOST)) between steps.
    // specialBuffer() in arg table population doesn't check for stale device data.
    for (int ei = 0; ei < numExt; ei++) {
      if (externalArrays[ei] != nullptr) {
        if (Environment::getInstance().tritonVerifyKernels()) {
          auto* db = externalArrays[ei]->dataBuffer();
          DSP_DIAG(VERIFY, "SLOT_WRITE slot=%d tag=EXT_SYNC(capture) extIdx=%d pAct=%d sAct=%d len=%lld addr=%p",
                    -(ei + 1), ei,
                    db ? (db->isPrimaryActual() ? 1 : 0) : -1,
                    db ? (db->isSpecialActual() ? 1 : 0) : -1,
                    (long long)externalArrays[ei]->lengthOf(),
                    DSP_BUF(externalArrays[ei]));
        }
        externalArrays[ei]->syncToDevice();
      }
    }

    // Synchronize before capture to ensure all prior async work is complete
    cudaStreamSynchronize(cudaStr);
    // Clear any sticky CUDA error before capture — stale errors from prior operations
    // (e.g., cudaFuncGetName on driver-API functions) contaminate capture and launch.
    cudaGetLastError();

    // Configurable: push primary CUDA context during capture.
    // Default OFF — the non-Triton path works without it. Pushing and then popping
    // after capture may cause SIGSEGV on replay (null pointer inside libcuda.so).
    // Enable via ND4J_TRITON_GRAPH_CTX_PUSH=1 for debugging.
    int tritonCaptureDevice = 0;
    cudaGetDevice(&tritonCaptureDevice);
    CUcontext primaryCtx = nullptr;
    CUcontext prevCtx = nullptr;
    bool didPushCtx = false;
    if (Environment::getInstance().tritonGraphCtxPush()) {
      CUdevice cuDev;
      cuDeviceGet(&cuDev, tritonCaptureDevice);
      cuDevicePrimaryCtxRetain(&primaryCtx, cuDev);
      cuCtxGetCurrent(&prevCtx);
      if (prevCtx != primaryCtx) {
        cuCtxPushCurrent(primaryCtx);
        didPushCtx = true;
        DSP_DIAG(EXECUTE, "Triton capture pushed primary ctx %p (was %p) for device %d",
                  (void*)primaryCtx, (void*)prevCtx, tritonCaptureDevice);
      }
    }

    // ── PRE-CAPTURE WARMUP EXECUTION ────────────────────────────────────────
    // During CUDA graph capture, GPU operations are NOT executed — they are only
    // recorded into the graph.  The capture step's output buffers retain whatever
    // values they had BEFORE capture started.  Without a warmup, those values are
    // from the PREVIOUS step's execution, producing a stale/wrong token that
    // corrupts the entire decode sequence.
    //
    // Fix: run a non-capture execution BEFORE capture to produce correct output
    // for this step.  The capture then records the same operations (for replay),
    // but the output buffers already have the correct values from the warmup.
    // This matches the non-Triton CUDA graph path (NativeDynamicShapePlan_cudagraph.cu
    // line 488-490) which runs executeSegmentSlotBySlot() before capture.
    {
      DSP_DIAG_SEG(EXECUTE, seg.startSlot, "Triton pre-capture warmup for seg[%d-%d] execCount=%d",
                seg.startSlot, seg.endSlot, seg.exec.executionCount);

      // Set cuBLAS workspace during warmup too, so cuBLAS selects the same GEMM
      // algorithms as during capture. Without this, warmup may use different
      // algorithms than capture, causing shape/result divergence.
      setCublasWorkspaceForWarmup();

      // Disable frozen fast path for warmup — same rationale as capture below.
      std::vector<NativeSlot::SlotState> savedSlotStateWarmup(seg.endSlot - seg.startSlot + 1);
      for (int s = seg.startSlot; s <= seg.endSlot; s++) {
        savedSlotStateWarmup[s - seg.startSlot] = slots_[s].state_;
        if (slots_[s].state_ >= NativeSlot::SlotState::FROZEN)
          slots_[s].state_ = NativeSlot::SlotState::SHAPE_CACHED;
      }

      auto warmupStatus = backend->executeSegment(seg, slots_, externalArrays, numExt,
                                                   outputSlots_, totalOutputSlots_, stream);
      // Restore frozen state after warmup
      for (int s = seg.startSlot; s <= seg.endSlot; s++) {
        slots_[s].state_ = savedSlotStateWarmup[s - seg.startSlot];
      }

      if (warmupStatus != Status::OK) {
        DSP_DIAG(EXECUTE, "FATAL: Triton pre-capture warmup FAILED for seg[%d-%d] status=%d. "
                  "BLOCKING EXECUTION.",
                  seg.startSlot, seg.endSlot, static_cast<int>(warmupStatus));
        seg.exec.compilationFailed = true;
        // Destroy the replay handle created before warmup — it holds workspace
        // memory even though capture never started.
        platformCleanupSegmentForRebuild(seg);
        return warmupStatus;
      }
      // Decrement executionCount — the warmup was an extra execution that should
      // not count toward the capture threshold.
      if (seg.exec.executionCount > 0) seg.exec.executionCount--;

      // Synchronize before capture to ensure warmup results are visible
      cudaStreamSynchronize(cudaStr);
      cudaGetLastError();

      // Reset cast cache indices after warmup so capture starts from index 0.
      // We intentionally preserve the warmup's cast cache entries (NOT full clear).
      // Capture reuses them via assign() — the graph records a cast kernel from the
      // real input to the cached buffer, then cuBLAS reads the cached buffer.
      // clearCastCache() would delete these entries, forcing capture to allocate
      // new ones from the capture workspace. Those workspace sub-allocations
      // cannot be individually freed by cudaFreeAsync (they're interior pointers
      // of the 32MB workspace block), so subsequent clearCastCache() calls
      // corrupt the CUDA memory pool → "illegal memory access" on replay.
      MmulHelper::resetCastCacheIndices();

      DSP_DIAG_SEG(EXECUTE, seg.startSlot, "Triton pre-capture warmup DONE for seg[%d-%d]",
                seg.startSlot, seg.endSlot);

      // DIAGNOSTIC: dump warmup's final output argmax for comparison with replay
      {
        int finalOutputSlot = -1;
        if (seg.endSlot < numSlots_ && slots_[seg.endSlot].numOutputs > 0) {
          finalOutputSlot = slots_[seg.endSlot].outputSlotIndices[0];
        }
        if (finalOutputSlot < 0 || finalOutputSlot >= totalOutputSlots_)
          finalOutputSlot = seg.endSlot;
        if (finalOutputSlot >= 0 && finalOutputSlot < totalOutputSlots_ &&
            outputSlots_[finalOutputSlot] != nullptr) {
          auto* warmupOut = outputSlots_[finalOutputSlot];
          if (warmupOut->dataType() == FLOAT32 && warmupOut->lengthOf() > 0) {
            int warmupArgmax = dspArgmax(DSP_BUF(warmupOut), warmupOut->dataType(),
                                          warmupOut->lengthOf());
            std::string warmupVals = dspDumpSlotValues(DSP_BUF(warmupOut), warmupOut->dataType(),
                                                        warmupOut->lengthOf(), 4);
            DSP_DIAG(EXECUTE, "WARMUP ARGMAX: slot=%d argmax=%d len=%lld vals=%s execCount=%d",
                     finalOutputSlot, warmupArgmax, (long long)warmupOut->lengthOf(),
                     warmupVals.c_str(), seg.exec.executionCount);
          }
        }
      }

      // ── RESTORE NULL OUTPUT SLOTS FROM CACHE ─────────────────────────────
      // The warmup execution may clear some outputSlots_ entries (e.g. control
      // flow CF_SWITCH dead outputs, or segment cleanup paths).  The values
      // were captured into slotArrayCache_ during execution, so restore any
      // Phase 2: slotArrayCache_ == outputSlots_ (unified).
      // Post-warmup restoration is a no-op — arrays produced during warmup
      // are already in outputSlots_ (which IS slotArrayCache_).
    }

    // DIAGNOSTIC: warmup-only mode — skip capture, use warmup result directly.
    // Enables bisection: if warmup-only produces correct output but capture+replay
    // does not, the bug is in capture/replay. Set ND4J_TRITON_WARMUP_ONLY=1.
    {
      static bool warmupOnly = (std::getenv("ND4J_TRITON_WARMUP_ONLY") != nullptr &&
                                 std::string(std::getenv("ND4J_TRITON_WARMUP_ONLY")) == "1");
      if (warmupOnly) {
        DSP_DIAG(EXECUTE, "WARMUP_ONLY: skipping capture for seg[%d-%d], using warmup result",
                  seg.startSlot, seg.endSlot);
        // Clean up thread-local state
        tl_captureWorkspace = nullptr;
        tl_captureWorkspaceSize = 0;
        tl_captureWorkspaceOffset = 0;
        // Free host workspace — no graph captured, no replay to reference it
        if (tl_captureHostWorkspace != nullptr) {
          cudaFreeHost(tl_captureHostWorkspace);
        }
        tl_captureHostWorkspace = nullptr;
        tl_captureHostWorkspaceSize = 0;
        tl_captureHostWorkspaceOffset = 0;
        tl_capturedHostPtrs.clear();
        tl_graphCaptureStream = prevCaptureStream;
        // Don't need the replay handle — fall through to non-capture path next time.
        // Destroy it to free the workspace memory allocated at line 1756.
        seg.exec.compilationFailed = true;
        platformCleanupSegmentForRebuild(seg);
        if (didPushCtx) {
          CUcontext dummy;
          cuCtxPopCurrent(&dummy);
          CUdevice cuDev;
          cuDeviceGet(&cuDev, tritonCaptureDevice);
          cuDevicePrimaryCtxRelease(cuDev);
        }
        restoreCublasWorkspaceAfterCapture(stream);
        return Status::OK;
      }
    }

    // NOW set cuBLAS handle to capture stream — AFTER warmup completed.
    // During warmup, gap ops must use their default stream (gapStr) for cuBLAS.
    // Only during actual capture do we switch cuBLAS to tritonStr so GEMM nodes
    // are recorded into the CUDA graph on the correct stream.
    setCublasWorkspaceForCapture(stream);

    // cuBLAS workspace preservation during capture.
    //
    // CRITICAL: Once shapes are frozen (shapesFrozen_ == true), NEVER zero the cuBLAS workspace.
    // During capture, cuBLAS stores plan/descriptor data in the workspace. Captured CUDA graphs
    // inherit these cached plans and omit H2D re-upload nodes. Zeroing the workspace destroys
    // cached plans, causing GEMM kernels to read zeros and hang on replay.
    //
    // The workspace content must be preserved across ALL captures and replays once frozen.
    // cuBLAS plans are stable for fixed shapes, so preservation is safe.
    //
    // Pre-frozen (shapes not yet frozen): zeroing is acceptable as no graphs are captured yet.
    if (shapesFrozen_ && cublasWorkspaceBuffer_ != nullptr && cublasWorkspaceSize_ > 0) {
      DSP_DIAG(MEMORY, "pre-capture: cuBLAS workspace PRESERVED (%zuMB) — shapes frozen, plans stable",
               cublasWorkspaceSize_ / (1024*1024));
      // Do NOT zero — preserve cuBLAS plan data for captured graph replay
    }
    // Note: Pre-frozen zeroing removed entirely — not needed for correctness and
    // adds unnecessary overhead. cuBLAS handles uninitialized workspace correctly.

    // Disable frozen fast path during capture. Same rationale as non-Triton path:
    // capture may re-create views, and the frozen context has stale input/output pointers
    // from the prior non-capture execution. Using the full (non-frozen) path during capture
    // is a one-time cost — all context pointers are properly reconfigured with capture-time
    // arrays, including correct nullify() calls to zero output buffers.
    // Save and restore frozenContextReady after capture so replay uses frozen fast path.
    std::vector<NativeSlot::SlotState> savedSlotStateTriton(seg.endSlot - seg.startSlot + 1);
    for (int s = seg.startSlot; s <= seg.endSlot; s++) {
      savedSlotStateTriton[s - seg.startSlot] = slots_[s].state_;
      if (slots_[s].state_ >= NativeSlot::SlotState::FROZEN)
        slots_[s].state_ = NativeSlot::SlotState::SHAPE_CACHED;
    }

    // ── Create capture buffers for PLACEHOLDER external inputs ─────────────
    // Only buffer the dynamic inputs (position_ids, attention_mask, input_ids,
    // inputs_embeds) that Java updates between decode steps. Model weights and
    // ConstantOfShape intermediates are NOT buffered — their data doesn't change
    // or is handled by the createValueKey mechanism.
    //
    // During capture, the graph bakes in capture buffer addresses. Before each
    // replay, we D2D copy fresh data from Java's arrays to capture buffers.
    std::vector<std::pair<int, NDArray*>> savedExtForCapture;
    {
      std::unordered_set<int> capturedExtIndices;
      for (int ei = 0; ei < numExt; ei++) {
        if (ei >= static_cast<int>(externalInputIsVariable_.size())) break;
        if (!externalInputIsVariable_[ei]) continue;  // Only PLACEHOLDER inputs
        if (capturedExtIndices.count(ei)) continue;
        NDArray* src = externalArrays[ei];
        if (src == nullptr || src->lengthOf() == 0) continue;

        capturedExtIndices.insert(ei);
        src->syncToDevice();
        size_t srcBytes = src->lengthOf() * src->sizeOfT();

        // Check if this is a KV cache input — these use directReference
        // (the graph reads/writes the original buffer, no copy needed)
        bool isKvCacheInput = false;
        if (kvCacheRetentionEnabled_) {
          for (int km = 0; km < kvCacheNumMappings_; km++) {
            if (kvCacheMappings_[km].pastInputExternalIdx == ei) {
              isKvCacheInput = true;
              break;
            }
          }
        }

        if (isKvCacheInput) {
          // KV cache: graph uses the actual buffer — no copy needed on replay
          ReplayCaptureBuffer cb;
          cb.buffer = src;
          cb.externalInputIndex = ei;
          cb.crossSegmentSlotIdx = -1;
          cb.capturedSize = srcBytes;
          cb.directReference = true;
          cb.initialCopyDone = true;
          cb.lastSourcePtr = DSP_BUF(src);
          seg.exec.replayHandle->addCaptureBuffer(std::move(cb));
          // Do NOT save/replace externalArrays — graph uses src directly
        } else {
          // Detect weight tensors: large inputs (> 1MB) that are NOT the dynamic
          // decode inputs (position_ids, attention_mask, input_ids, inputs_embeds).
          // Weights never change between decode steps — using directReference
          // avoids duplicating ~10GB of model weights in capture buffers.
          constexpr size_t WEIGHT_THRESHOLD = 1 * 1024 * 1024;  // 1MB
          
          // During capture, hasPendingDecodeUpdate_ is false. Check external input
          // indices directly to identify dynamic decode inputs.
          // Dynamic decode inputs are small (scalars or 1D/2D tensors).
          // Weights are typically large (>= 1MB) and rank 2 or 4.
          bool isDynamicDecodeInput = false;
          if (isDecodeInputsConfigured()) {
            isDynamicDecodeInput = (ei == decodeInputIdsExtIdx_ ||
                                    ei == decodePositionIdsExtIdx_ ||
                                    ei == decodeAttentionMaskExtIdx_);
          }
          
          // inputs_embeds is typically a rank-3 tensor with shape [batch, seq, hidden]
          // where seq is the prompt length (varies per inference). Check shape to distinguish.
          bool isInputsEmbeds = false;
          if (src->rankOf() == 3) {
            auto* shape = src->shapeOf();
            // inputs_embeds has shape [batch, seq, hidden] where seq >= 1
            // Weight tensors with rank 3 are rare (usually rank 2 or 4)
            // If seq dim (shape[1]) is large (> 100), likely inputs_embeds
            if (shape[1] > 100) {
              isInputsEmbeds = true;
            }
          }

          // KV cache tensors are rank-4 with shape [batch, num_heads, seq, head_dim].
          // They're large but dynamic - NOT weights. Exclude them from directReference.
          bool isKvCacheTensor = false;
          if (src->rankOf() == 4) {
            auto* shape = src->shapeOf();
            // KV cache: [batch=1, num_heads, seq, head_dim]
            // num_heads is typically 8-32, head_dim is typically 64-128
            // If shape[1] is small (<= 64) and shape[3] is small (<= 256), likely KV cache
            if (shape[1] <= 64 && shape[3] <= 256) {
              isKvCacheTensor = true;
            }
          }

          bool isWeight = !isDynamicDecodeInput && !isInputsEmbeds && !isKvCacheTensor && srcBytes >= WEIGHT_THRESHOLD;

          if (isWeight) {
            // Weight tensor: use directReference to avoid duplicating GPU memory.
            // The graph reads directly from the original weight buffer, which
            // never moves (protected by frozen ref count).
            ReplayCaptureBuffer cb;
            cb.buffer = src;
            cb.externalInputIndex = ei;
            cb.crossSegmentSlotIdx = -1;
            cb.capturedSize = srcBytes;
            cb.directReference = true;
            cb.initialCopyDone = true;
            cb.lastSourcePtr = DSP_BUF(src);
            seg.exec.replayHandle->addCaptureBuffer(std::move(cb));
            DSP_DIAG(MEMORY, "CAPTURE: extIdx=%d is weight (%zu MB), using directReference (no copy)",
                     ei, srcBytes / (1024 * 1024));
          } else {
            // Regular placeholder (dynamic decode input or small tensor):
            // create a fixed-address capture buffer
            auto srcShapeVec = *src->getShapeAsVector();
            auto* capBuf = new NDArray(src->ordering(), srcShapeVec, src->dataType(),
                                       sd::LaunchContext::defaultContext());
            copyIntoCaptureBuffer(capBuf, src, cudaStr, true, "ext", ei,
                                  seg.startSlot, seg.endSlot);

            ReplayCaptureBuffer cb;
            cb.buffer = capBuf;
            cb.externalInputIndex = ei;
            cb.crossSegmentSlotIdx = -1;
            cb.capturedSize = srcBytes;
            cb.neverSkipCopy = true;
            seg.exec.replayHandle->addCaptureBuffer(std::move(cb));

            savedExtForCapture.push_back({ei, externalArrays[ei]});
            externalArrays[ei] = capBuf;
          }
        }
      }
      if (!capturedExtIndices.empty()) {
        cudaStreamSynchronize(cudaStr);
        DSP_DIAG(EXECUTE, "CAPTURE_BUFFERS: created %zu buffers for PLACEHOLDER ext inputs",
                 capturedExtIndices.size());
      }
    }

    // ── Create capture buffers for CROSS-SEGMENT output slot inputs ─────────
    // When a non-capturable segment (data-dependent ops like Where, NonZero)
    // precedes this capturable segment, its output slots feed into this segment
    // as inputs. The graph bakes in capture-time addresses. If the non-capturable
    // segment reallocates output arrays (data-dependent shape changes), the graph
    // reads stale data from the old address. Capture buffers provide fixed-address
    // staging areas, with D2D copies of fresh data before each replay.
    std::vector<std::pair<int, NDArray*>> savedSlotsForCapture;
    {
      // Build precise set of output slots PRODUCED by steps within this segment.
      // Using range-based check (srcIdx < startSlot || srcIdx > endSlot) is wrong
      // because output slot indices don't necessarily equal step indices.
      // A step at index N may produce output at slot M != N. So a slot index
      // that falls within [startSlot, endSlot] may NOT be produced by any step
      // in this segment — it's actually a cross-segment dependency.
      std::unordered_set<int> producedBySegment;
      for (int s = seg.startSlot; s <= seg.endSlot; s++) {
        for (int o = 0; o < slots_[s].numOutputs; o++) {
          int si = slots_[s].outputSlotIndices[o];
          if (si >= 0 && si < totalOutputSlots_) {
            producedBySegment.insert(si);
          }
        }
      }
      std::unordered_set<int> crossSegSlots;
      for (int s = seg.startSlot; s <= seg.endSlot; s++) {
        for (int i = 0; i < slots_[s].numInputs; i++) {
          int srcIdx = slots_[s].inputSourceIndices[i];
          if (srcIdx >= 0 && !producedBySegment.count(srcIdx)) {
            crossSegSlots.insert(srcIdx);
          }
        }
      }
      int crossSegCreated = 0;
      for (int slotIdx : crossSegSlots) {
        if (slotIdx >= totalOutputSlots_) continue;
        // Phase 2: slotArrayCache_ == outputSlots_ (unified), no separate restore needed
        if (outputSlots_[slotIdx] == nullptr) {
          DSP_DIAG(EXECUTE, "CAPTURE_BUF_INIT: skipping cross-seg slot %d (null outputSlot)", slotIdx);
          continue;
        }
        NDArray* src = outputSlots_[slotIdx];
        src->syncToDevice();
        size_t srcBytes = src->lengthOf() * src->sizeOfT();
        if (srcBytes == 0) continue;

        // Create fixed-address capture buffer for this cross-segment input
        auto srcShapeVec = *src->getShapeAsVector();
        auto* capBuf = new NDArray(src->ordering(), srcShapeVec, src->dataType(),
                                   sd::LaunchContext::defaultContext());
        copyIntoCaptureBuffer(capBuf, src, cudaStr, false, "slot", slotIdx,
                              seg.startSlot, seg.endSlot);

        ReplayCaptureBuffer cb;
        cb.buffer = capBuf;
        cb.externalInputIndex = -1;
        cb.crossSegmentSlotIdx = slotIdx;
        cb.capturedSize = srcBytes;
        cb.neverSkipCopy = true;
        seg.exec.replayHandle->addCaptureBuffer(std::move(cb));

        // Swap outputSlots_ so graph captures with capture buffer addresses
        DSP_DIAG(EXECUTE, "SAVE_SLOT_FOR_CAPTURE: slot %d saved=%p (db=%p valid=%d), replacing with capBuf=%p",
                 slotIdx, (void*)outputSlots_[slotIdx],
                 outputSlots_[slotIdx] ? (void*)outputSlots_[slotIdx]->dataBuffer() : nullptr,
                 (outputSlots_[slotIdx] && outputSlots_[slotIdx]->dataBuffer()) ?
                   outputSlots_[slotIdx]->dataBuffer()->isValid() : -1,
                 (void*)capBuf);
        savedSlotsForCapture.push_back({slotIdx, outputSlots_[slotIdx]});
        outputSlots_[slotIdx] = capBuf;
        crossSegCreated++;
      }
      if (crossSegCreated > 0) {
        cudaStreamSynchronize(cudaStr);
        DSP_DIAG(EXECUTE, "CAPTURE_BUFFERS: created %d buffers for CROSS-SEGMENT slot inputs",
                 crossSegCreated);
      }
    }

    // Pre-capture batch-zero: zero all registered buffers BEFORE beginCapture.
    // These cudaMemsetAsync calls execute normally on the stream (not captured).
    // This ensures ops get zeroed outputs during the capture run for correct results.
    // During capture, individual nullify() calls are suppressed (no memset graph nodes).
    // On replay, the same zeroing happens via pre-replay batch-zero above.
    if (Environment::getInstance().dspBatchZero() && !batchZeroEntries_.empty()) {
      for (auto& entry : batchZeroEntries_) {
        if (entry.ptr != nullptr && entry.bytes > 0) {
          cudaMemsetAsync(entry.ptr, 0, entry.bytes, cudaStr);
        }
      }
      DSP_DIAG(MEMORY, "pre-capture batch-zero: %d buffers zeroed via cudaMemsetAsync (fill engines, before beginCapture)",
                static_cast<int>(batchZeroEntries_.size()));
    }

    // ── Save warmup output slot pointers BEFORE capture ─────────────────
    // Gap ops are skipped during capture (they return OK without executing).
    // outputSlots_[] retains warmup values throughout capture — no save/restore needed.
    // Downstream segments see valid warmup data. Triton sub-kernel arg tables
    // reference warmup addresses, which are stable.

    // POST-ALLOCATION MEMORY GATE: workspace + cuBLAS are allocated. Check that
    // enough headroom remains for CUDA driver graph metadata before starting capture.
    // This is tight and accurate — only graph metadata overhead remains.
    {
      size_t gpuFree = 0, gpuTotal = 0;
      cudaMemGetInfo(&gpuFree, &gpuTotal);
      size_t safetyBytes = Environment::getInstance().dspGraphMetadataSafetyMb() * 1024ULL * 1024ULL;
      if (gpuFree < safetyBytes) {
        int deviceId = 0;
        cudaGetDevice(&deviceId);
        DSP_DIAG_SEG(MEMORY, seg.startSlot,
                     "POST-ALLOC GATE FAILED: free=%zuMB < safety=%zuMB for seg[%d-%d]",
                     gpuFree / (1024*1024), safetyBytes / (1024*1024),
                     seg.startSlot, seg.endSlot);
        platformCleanupSegmentForRebuild(seg);
        return reportOomError(seg, "post_alloc_gate", safetyBytes, deviceId);
      }
    }

    auto* cudaReplay = static_cast<CudaGraphReplayHandle*>(seg.exec.replayHandle.get());
    auto handle = cudaReplay->getNativeHandle();
    bool captureOk = handle->beginCapture(cudaStr, cudaStreamCaptureModeRelaxed);
    if (captureOk) {
      DSP_DIAG_SEG(EXECUTE, seg.startSlot, "Triton graph capture started for seg[%d-%d] execCount=%d",
                seg.startSlot, seg.endSlot, seg.exec.executionCount);
      tl_graphExecutionActive = true;

      // Batch-zero during capture: DON'T launch inside the graph — instead,
      // suppress individual nullify() calls so no memset nodes get captured.
      // The actual zeroing happens OUTSIDE the graph before each replay() call
      // using cudaMemsetAsync (fill engines, no SM competition).
      // This removes ~700 memset graph nodes while keeping fill-engine efficiency.
      if (Environment::getInstance().dspBatchZero() && !batchZeroEntries_.empty()) {
        setBatchZeroActive(true);
        DSP_DIAG(MEMORY, "batch-zero CAPTURE-SKIP: suppressing %d individual nullify() calls, "
                  "zeroing will happen outside graph before replay",
                  static_cast<int>(batchZeroEntries_.size()));

        //  Mark ALL output slot DataBuffers as device-actual (sAct=1)
        // after batch-zero.  Batch-zero zeroes device memory directly via a GPU
        // kernel, bypassing NDArray's actuality tracking.  Without this,
        // DataBuffer::syncToSpecial() inside native gap ops sees sAct=0 (stale
        // from a previous step) and generates an H2D memcpy that gets RECORDED
        // in the CUDA graph.  On replay, that H2D copies STALE host data
        // (from capture time) over the freshly batch-zeroed device buffer,
        // corrupting inputs to downstream ops.
        //
        // By marking sAct=1 here, syncToSpecial() during capture becomes a
        // no-op for internal buffers (device is already "actual" — it has
        // zeros, which is the correct initial state).  This matches the
        // standard CUDA graph path which uses capture buffers with correct
        // actuality.
        int markedCount = 0;
        for (int si = seg.startSlot; si <= seg.endSlot; si++) {
          for (int o = 0; o < slots_[si].numOutputs; o++) {
            int outIdx = slots_[si].outputSlotIndices[o];
            if (outIdx >= 0 && outIdx < totalOutputSlots_ && outputSlots_[outIdx]) {
              auto* db = outputSlots_[outIdx]->dataBuffer();
              if (db) {
                db->writeSpecial();
                markedCount++;
              }
            }
          }
        }
        DSP_DIAG(MEMORY, "batch-zero actuality: marked %d output DataBuffers as device-actual",
                  markedCount);
        if (Environment::getInstance().tritonVerifyKernels()) {
          DSP_DIAG(VERIFY, "SLOT_WRITE tag=BATCH_ZERO seg[%d-%d] %d buffers suppressed (nullify skipped), %d marked sAct=1",
                    seg.startSlot, seg.endSlot, static_cast<int>(batchZeroEntries_.size()), markedCount);
        }
      } else {
        DSP_DIAG(MEMORY, "batch-zero DISABLED (dspBatchZero=%d, entries=%d)",
                  (int)Environment::getInstance().dspBatchZero(), static_cast<int>(batchZeroEntries_.size()));
      }

      // Query node count mid-capture to verify operations are being recorded
      size_t midCaptureNodes = handle->getNumNodesDuringCapture(cudaStr);
      DSP_DIAG(EXECUTE, "Triton capture mid-check: %zu nodes recorded before executeSegment (batchZero=%d entries, outside-graph)",
                midCaptureNodes, static_cast<int>(batchZeroEntries_.size()));

      // Snapshot all buffer addresses at capture entry — compare with replay to detect stale pointers
      {
        std::vector<void*> outAddrs, extAddrs;
        extractDeviceAddrs(outputSlots_, totalOutputSlots_, outAddrs);
        extractDeviceAddrs(externalArrays, numExt, extAddrs);
        DspDiagnostics::getInstance().clearAddressSnapshots();
        DSP_DIAG_SNAPSHOT_ADDRS("capture-entry", outAddrs.data(), totalOutputSlots_,
                                 extAddrs.data(), numExt);
      }

      auto captureStatus = backend->executeSegment(seg, slots_, externalArrays, numExt,
                                                   outputSlots_, totalOutputSlots_, stream);
      setBatchZeroActive(false);
      tl_graphExecutionActive = false;

      // Snapshot addresses AFTER capture execution to detect pointer changes during capture
      {
        std::vector<void*> outAddrs, extAddrs;
        extractDeviceAddrs(outputSlots_, totalOutputSlots_, outAddrs);
        extractDeviceAddrs(externalArrays, numExt, extAddrs);
        DSP_DIAG_SNAPSHOT_ADDRS("capture-exit", outAddrs.data(), totalOutputSlots_,
                                 extAddrs.data(), numExt);
        int changed = DSP_DIAG_COMPARE_ADDRS("capture-entry", "capture-exit");
        if (changed > 0) {
          DSP_DIAG(EXECUTE, "WARNING: %d buffer addresses CHANGED during capture execution!", changed);
        }
      }

      // Diagnostic: capture workspace usage
      DSP_DIAG(MEMORY, "capture workspace used: %zu / %zu bytes (%.1f%%)",
               tl_captureWorkspaceOffset, seg.exec.replayHandle->getWorkspaceBytes(),
               seg.exec.replayHandle->getWorkspaceBytes() > 0 ? (100.0 * tl_captureWorkspaceOffset / seg.exec.replayHandle->getWorkspaceBytes()) : 0.0);
      // Check for CUDA errors generated during capture — these become invalid graph nodes.
      // Don't use cudaGetLastError (which clears) — peek first for diagnostics.
      {
        cudaError_t capPhaseErr = cudaPeekAtLastError();
        if (capPhaseErr != cudaSuccess) {
          DSP_DIAG(BACKEND, "WARNING - CUDA error during Triton capture phase: %s (%d)",
                    cudaGetErrorString(capPhaseErr), (int)capPhaseErr);
          // Clear it so endCapture can proceed (the graph may still be partially valid)
          cudaGetLastError();
        }
      }

      // Query node count after execution to see how many ops were captured
      size_t postExecNodes = 0;
      {
        cudaStreamCaptureStatus capStat = cudaStreamCaptureStatusNone;
        cudaGraph_t capGraph = nullptr;
        unsigned long long capId = 0;
        auto capErr = cudaStreamGetCaptureInfo_v2(cudaStr, &capStat, &capId, &capGraph, nullptr, nullptr);
        if (capErr == cudaSuccess && capGraph != nullptr) {
          cudaGraphGetNodes(capGraph, nullptr, &postExecNodes);
        }
      }
      DSP_DIAG(EXECUTE, "Triton capture post-exec: %zu nodes, captureStatus=%d",
                postExecNodes, static_cast<int>(captureStatus));
      fflush(stdout); fflush(stderr);

      bool endOk = false;
      if (captureStatus == Status::OK) {
        endOk = handle->endCapture(cudaStr);
      } else {
        DSP_DIAG(EXECUTE, "FATAL: Triton capture execution FAILED status=%d for seg[%d-%d]. "
                  "BLOCKING EXECUTION.",
                  static_cast<int>(captureStatus), seg.startSlot, seg.endSlot);
        fflush(stdout); fflush(stderr);
        if (handle->isCapturing()) {
          handle->endCapture(cudaStr);
        }
      }

      if (endOk) {
        size_t numGraphNodes = handle->getNumNodes();
        int segSize = seg.endSlot - seg.startSlot + 1;
        DSP_DIAG_SEG(COMPILE, seg.startSlot,
                     "GRAPH CAPTURE COMPLETE: seg[%d-%d] %zu nodes captured from %d slots (%.1f nodes/slot)",
                     seg.startSlot, seg.endSlot, numGraphNodes, segSize,
                     segSize > 0 ? (double)numGraphNodes / segSize : 0.0);
        DSP_DIAG(EXECUTE, "Triton capture endOk: graph has %zu nodes", numGraphNodes);

        // Empty graphs (0 nodes) have no GPU work — skip replay to avoid
        // spurious fingerprint mismatches when slot addresses change.
        if (numGraphNodes == 0) {
          DSP_DIAG_SEG(COMPILE, seg.startSlot,
                       "empty Triton graph for seg[%d-%d] (0 nodes) — marking as non-capturable",
                       seg.startSlot, seg.endSlot);
          seg.exec.compilationFailed = true;
          seg.exec.replayHandle.reset();
          seg.exec.executionCount++;
          return Status::OK;
        }

        // Sample final output AFTER endCapture (stream no longer capturing, safe)
        if (seg.endSlot < totalOutputSlots_ && outputSlots_[seg.endSlot] != nullptr) {
          auto* finalOut = outputSlots_[seg.endSlot];
          if (finalOut->dataType() == FLOAT32) {
            DSP_DIAG_DUMP_SLOT("capture-post-endCapture", seg.endSlot,
                               DSP_BUF(finalOut), finalOut->lengthOf());
          }
        }
        // Dump top logit from capture execution via DSP_DIAG
        // Use outputSlotIndices[0] to get the ACTUAL final output slot
        // (matches GRAPH_REPLAY logic for apples-to-apples comparison)
        {
          int captureOutputSlot = -1;
          if (seg.endSlot < numSlots_ && slots_[seg.endSlot].numOutputs > 0) {
            captureOutputSlot = slots_[seg.endSlot].outputSlotIndices[0];
          }
          if (captureOutputSlot < 0 || captureOutputSlot >= totalOutputSlots_) {
            captureOutputSlot = seg.endSlot;
          }
          if (captureOutputSlot >= 0 && captureOutputSlot < totalOutputSlots_ &&
              outputSlots_[captureOutputSlot] != nullptr) {
            auto* out = outputSlots_[captureOutputSlot];
            if (out->dataType() == FLOAT32 && out->lengthOf() > 0) {
              DSP_DIAG_DUMP_SEG_OUTPUT("CAPTURE_EXEC", captureOutputSlot, DSP_BUF(out),
                                       out->lengthOf(), seg.exec.executionCount, stream);
            }
          }
        }
      }

      if (endOk) {
        auto stats = handle->getStatistics();
        DSP_DIAG(EXECUTE, "Triton graph stats: %d kernels, %d memcpys, %d memsets, "
                  "%d memAllocs, %d memFrees, %d hostCallbacks, %d events, %d empty",
                  stats.numKernels, stats.numMemcpyH2D, stats.numMemsets,
                  stats.numMemAllocs, stats.numMemFrees,
                  stats.numHostCallbacks, stats.numEvents, stats.numEmpty);
        fflush(stdout); fflush(stderr);
        if (stats.numMemAllocs > 0 || stats.numMemFrees > 0) {
          DSP_DIAG(EXECUTE, "Triton graph has %d MemAlloc + %d MemFree nodes "
                    "(paired alloc/free from cuBLAS internal workspace - CUDA 12+ handles these on replay).",
                    stats.numMemAllocs, stats.numMemFrees);
        }
        if (stats.numHostCallbacks > 0) {
          DSP_DIAG(BACKEND, "WARNING - Graph has %d host callback nodes!",
                    stats.numHostCallbacks);
        }
      }

      // Skip DOT dump by default for Triton graphs — cudaGraphDebugDotPrint with verbose
      // flags may also call cudaGraphKernelNodeGetParams internally, causing the same
      // cudaErrorInvalidDeviceFunction poisoning as getDetailedNodeInfo().
      if (endOk && Environment::getInstance().tritonDumpGraphDot()) {
        cudaGraphDebugDotPrint(handle->getGraph(), "/tmp/triton_graph_debug.dot", 0);
        DSP_DIAG(EXECUTE, "Triton graph dumped to /tmp/triton_graph_debug.dot");
        fflush(stdout); fflush(stderr);
      }

      // Skip getDetailedNodeInfo() for Triton graphs — it calls cudaFuncGetName on each
      // kernel node, which returns cudaErrorInvalidDeviceFunction (error 98) for Triton
      // kernels loaded via cuModuleLoadDataEx (driver API). The 658+ consecutive errors
      // poison the CUDA error state and cause cudaGraphLaunch to SIGSEGV.
      // Use getNumNodes() for basic stats instead (no per-node introspection).
      bool allKernelsValid = true;
      if (endOk) {
#ifdef SD_CUDA
        size_t totalNodes = handle->getNumNodes();
        DSP_DIAG(EXECUTE, "Triton graph has %zu nodes (skipping per-node inspection to avoid error-98 poisoning)",
                  totalNodes);
        fflush(stdout); fflush(stderr);
        // Ensure no sticky errors before instantiation
        cudaGetLastError();
#endif
      }

      bool instantiateOk = endOk && allKernelsValid && handle->instantiate();
      if (instantiateOk) {
        DSP_DIAG(EXECUTE, "Triton graph instantiated OK (graphExec=%p), about to launch...",
                  handle->getGraphExec());
        fflush(stdout); fflush(stderr);
      }

      if (!instantiateOk) {
        int deviceId = 0;
        cudaGetDevice(&deviceId);

        // Check if instantiation failed due to OOM — retry with eviction if possible.
        auto* cudaReplayForOom = seg.exec.replayHandle
            ? dynamic_cast<CudaGraphReplayHandle*>(seg.exec.replayHandle.get()) : nullptr;
        bool isOom = cudaReplayForOom && cudaReplayForOom->wasLastInstantiateOom();
        if (isOom && seg.exec.captureOomRetries < GraphSegment::maxOomRetries()) {
          seg.exec.captureOomRetries++;
          seg.exec.captureRetryAfterExec = seg.exec.executionCount + GraphSegment::retryInterval();
          DSP_DIAG_SEG(MEMORY, seg.startSlot,
                       "INSTANTIATE OOM — retry %d/%d, evicting LRU graphs. retryAfterExec=%d",
                       seg.exec.captureOomRetries, GraphSegment::maxOomRetries(),
                       seg.exec.captureRetryAfterExec);

          // Evict LRU graphs to free memory for the next attempt
          evictLruGraphs(segIdx, TRITON_CAPTURE_WORKSPACE_SIZE, stream);

          // Cleanup this failed attempt but do NOT set compilationFailed
          platformCleanupSegmentForRebuild(seg);
          cudaGetLastError();  // Clear sticky error
          // Fall through to slot-by-slot for this execution
          return executeSegmentSlotBySlot(seg, externalArrays, numExt, stream);
        }

        // Not OOM or retries exhausted — permanent failure
        platformCleanupSegmentForRebuild(seg);
        return reportCaptureError(seg, "instantiate", cudaGetLastError(), deviceId);
      }

      // POST-INSTANTIATE MEMORY CHECK removed: the validation launch immediately
      // below will reveal if the graph is usable. No speculative memory gate needed.

      // Graph instantiated — launch to validate the graph is not corrupted.
      // Warmup results are restored from savedWarmupOutputSlots below regardless.
      bool launchOk = false;
      {
        int deviceId = 0;
        cudaGetDevice(&deviceId);
        cudaGetLastError();
        bool replayResult = seg.exec.replayHandle->replay(stream);
        if (!replayResult) {
          platformCleanupSegmentForRebuild(seg);
          return reportReplayError(seg, "validation_launch", cudaGetLastError(), deviceId);
        }
        cudaError_t syncErr = cudaStreamSynchronize(cudaStr);
        if (syncErr != cudaSuccess) {
          platformCleanupSegmentForRebuild(seg);
          return reportReplayError(seg, "validation_sync", syncErr, deviceId);
        }
        DSP_DIAG(EXECUTE, "VALIDATION LAUNCH OK: seg[%d-%d] graph launched and synced successfully",
                 seg.startSlot, seg.endSlot);
        // LRU tracking: record when this segment was last replayed for eviction ordering
        seg.exec.lastReplayExecCount = executeCount_;
        launchOk = true;
      }

      if (launchOk) {
        if (seg.endSlot < totalOutputSlots_ && outputSlots_[seg.endSlot] != nullptr) {
          auto* finalOut = outputSlots_[seg.endSlot];
          if (finalOut->dataType() == FLOAT32) {
            DSP_DIAG_DUMP_SLOT("capture-post-launch", seg.endSlot,
                               DSP_BUF(finalOut), finalOut->lengthOf());
          }
        }
        // Dump top logit from first replay (graph launch after capture) via DSP_DIAG
        if (seg.endSlot < totalOutputSlots_ && outputSlots_[seg.endSlot] != nullptr) {
          auto* out = outputSlots_[seg.endSlot];
          if (out->dataType() == FLOAT32 && out->lengthOf() > 0) {
            DSP_DIAG_DUMP_SEG_OUTPUT("REPLAY_LAUNCH", seg.endSlot, DSP_BUF(out),
                                     out->lengthOf(), seg.exec.executionCount, stream);
          }
        }
        // replayHandle already set (created before capture began)
        seg.exec.cachedShapeKey = segShapeKey;
        seg.exec.capturedInputAddrKey = segInputAddrKey;
        seg.exec.capturedCreateValueKey = createValueKey;
        seg.exec.capturedSlotAddrHash = computeSlotAddrHash(
            outputSlots_, seg.startSlot, seg.endSlot, totalOutputSlots_);
        snapshotExternalAddrs(seg, externalArrays, numExt);

        // Export graph stats and DOT file for diagnostics
        auto stats = handle->getStatistics();
        DSP_DIAG_SEG(EXECUTE, seg.startSlot, "Triton graph CAPTURED and launched for seg[%d-%d]: "
                  "%d kernels, %d memcpy, %d memset, %d memAlloc, %d memFree "
                  "(workspace=%zuMB, offset=%zu)",
                  seg.startSlot, seg.endSlot,
                  stats.numKernels, stats.numMemcpyH2D + stats.numMemcpyD2H + stats.numMemcpyD2D,
                  stats.numMemsets, stats.numMemAllocs, stats.numMemFrees,
                  seg.exec.replayHandle->getWorkspaceBytes() / (1024*1024), tl_captureWorkspaceOffset);
        // Dump H2D memcpy node details to identify the source of baked-in host addresses
        if (stats.numMemcpyH2D > 0 && DSP_DIAG_ENABLED(EXECUTE)) {
          size_t numGraphNodes = 0;
          cudaGraphGetNodes(handle->getGraph(), nullptr, &numGraphNodes);
          if (numGraphNodes > 0) {
            std::vector<cudaGraphNode_t> graphNodes(numGraphNodes);
            cudaGraphGetNodes(handle->getGraph(), graphNodes.data(), &numGraphNodes);
            for (size_t ni = 0; ni < numGraphNodes; ni++) {
              cudaGraphNodeType nodeType;
              cudaGraphNodeGetType(graphNodes[ni], &nodeType);
              if (nodeType == cudaGraphNodeTypeMemcpy) {
                cudaMemcpy3DParms mcpyParams;
                memset(&mcpyParams, 0, sizeof(mcpyParams));
                if (cudaGraphMemcpyNodeGetParams(graphNodes[ni], &mcpyParams) == cudaSuccess) {
                  size_t bytes = mcpyParams.extent.width *
                                 std::max(mcpyParams.extent.height, (size_t)1) *
                                 std::max(mcpyParams.extent.depth, (size_t)1);
                  const char* kindStr = (mcpyParams.kind == cudaMemcpyHostToDevice) ? "H2D" :
                                        (mcpyParams.kind == cudaMemcpyDeviceToDevice) ? "D2D" :
                                        (mcpyParams.kind == cudaMemcpyDeviceToHost) ? "D2H" : "other";
                  DSP_DIAG(EXECUTE, "GRAPH_NODE[%zu] MEMCPY %s: %zu bytes src=%p dst=%p "
                           "seg[%d-%d]",
                           ni, kindStr, bytes,
                           mcpyParams.srcPtr.ptr, mcpyParams.dstPtr.ptr,
                           seg.startSlot, seg.endSlot);
                }
              }
            }
          }
        }
        // Write DOT file for offline analysis.
        // Default: non-verbose (flag 0). Verbose queries kernel node params via
        // cudaFuncGetName, which returns cudaErrorInvalidDeviceFunction for
        // Triton CUfunction handles and may poison driver state.
        // Enable via ND4J_TRITON_GRAPH_DOT_VERBOSE=1 for debugging.
        {
          std::string dotPath = "/tmp/triton_graph_captured.dot";
          unsigned int dotFlags = Environment::getInstance().tritonGraphDotVerbose()
              ? cudaGraphDebugDotFlagsVerbose : 0;
          auto dotErr = cudaGraphDebugDotPrint(handle->getGraph(), dotPath.c_str(), dotFlags);
          if (dotErr == cudaSuccess) {
            DSP_DIAG(EXECUTE, "Exported Triton graph DOT to %s (verbose=%d)",
                      dotPath.c_str(), dotFlags != 0);
          }
          cudaGetLastError(); // Clear any error from dot print
        }
        // Write stats to a file the test can read
        {
          FILE* f = fopen("/tmp/triton_graph_stats.txt", "w");
          if (f) {
            fprintf(f, "segment=%d-%d\n", seg.startSlot, seg.endSlot);
            fprintf(f, "kernels=%d\n", stats.numKernels);
            fprintf(f, "memcpyH2D=%d\n", stats.numMemcpyH2D);
            fprintf(f, "memcpyD2H=%d\n", stats.numMemcpyD2H);
            fprintf(f, "memcpyD2D=%d\n", stats.numMemcpyD2D);
            fprintf(f, "memsets=%d\n", stats.numMemsets);
            fprintf(f, "memAllocs=%d\n", stats.numMemAllocs);
            fprintf(f, "memFrees=%d\n", stats.numMemFrees);
            fprintf(f, "hostCallbacks=%d\n", stats.numHostCallbacks);
            fprintf(f, "events=%d\n", stats.numEvents);
            fprintf(f, "childGraphs=%d\n", stats.numChildGraphs);
            fprintf(f, "totalNodes=%zu\n", handle->getNumNodes());
            fclose(f);
          }
        }
        status = Status::OK;
        usedTritonGraphCapture = true;

        // Phase 2: slotArrayCache_ == outputSlots_ (unified).
        // No need to sync cache ← output — they are the same pointer.

        // FORCE_RECAPTURE: invalidate graph immediately after capture+launch
        // so the NEXT step also re-captures instead of replaying a stale graph.
        // This ensures every single step is a fresh capture+launch with zero replays.
        if (Environment::getInstance().tritonForceRecapture()) {
          platformCleanupSegmentForRebuild(seg);
          seg.exec.argTableStable = false;
          batchD2DCount_ = 0;
          seg.exec.capturedInputAddrKey = 0;
          DSP_DIAG(EXECUTE, "FORCE_RECAPTURE: invalidated after capture+launch execCount=%d", seg.exec.executionCount);
        }
      } else {
        int deviceId = 0;
        cudaGetDevice(&deviceId);
        platformCleanupSegmentForRebuild(seg);
        return reportCaptureError(seg, "execute_during_capture", cudaGetLastError(), deviceId);
      }
    } else {
      int deviceId = 0;
      cudaGetDevice(&deviceId);

      // Check if beginCapture failed due to OOM — retry with eviction if possible.
      cudaError_t beginErr = cudaGetLastError();
      bool isOom = (beginErr == cudaErrorMemoryAllocation);
      if (isOom && seg.exec.captureOomRetries < GraphSegment::maxOomRetries()) {
        seg.exec.captureOomRetries++;
        seg.exec.captureRetryAfterExec = seg.exec.executionCount + GraphSegment::retryInterval();
        DSP_DIAG_SEG(MEMORY, seg.startSlot,
                     "BEGIN_CAPTURE OOM — retry %d/%d, evicting LRU graphs. retryAfterExec=%d",
                     seg.exec.captureOomRetries, GraphSegment::maxOomRetries(),
                     seg.exec.captureRetryAfterExec);
        evictLruGraphs(segIdx, TRITON_CAPTURE_WORKSPACE_SIZE, stream);
        platformCleanupSegmentForRebuild(seg);
        return executeSegmentSlotBySlot(seg, externalArrays, numExt, stream);
      }

      platformCleanupSegmentForRebuild(seg);
      return reportCaptureError(seg, "beginCapture", beginErr, deviceId);
    }

    // ── Restore warmup output slots for downstream segment visibility ──
    // This MUST happen regardless of capture success/failure. During capture
    // execution, ops allocate outputs from capture workspace, overwriting
    // outputSlots_[] with workspace addresses. If capture fails (endCapture
    // error, instantiate error, or execution error), outputSlots_[] still
    // has the stale workspace addresses. Downstream segments reading from
    // these get garbage data → NaN propagation.
    //
    // No WARMUP_RESTORE needed: gap ops were skipped during capture, so
    // outputSlots_[] was never overwritten. Warmup data is intact.

    DSP_DIAG(EXECUTE, "CAPTURE_COMPLETE: seg[%d-%d] hasReplay=%d compilationFailed=%d "
             "numCaptureBuffers=%d",
             seg.startSlot, seg.endSlot,
             seg.exec.replayHandle != nullptr,
             seg.exec.compilationFailed,
             seg.exec.replayHandle ? (int)seg.exec.replayHandle->getCaptureBuffers().size() : 0);

    // Restore original external arrays after capture (undo capture buffer wiring)
    for (auto& [extIdx, origArr] : savedExtForCapture) {
      externalArrays[extIdx] = origArr;
    }

    // Restore saved output slots after capture (direct, no pendingClose_ needed)
    for (auto& [slotIdx, origArr] : savedSlotsForCapture) {
      if (origArr != nullptr) {
        outputSlots_[slotIdx] = origArr;
      }
    }

    // Restore primary CUDA context if we pushed it
    if (didPushCtx) {
      CUcontext dummy;
      cuCtxPopCurrent(&dummy);
      CUdevice cuDev;
      cuDeviceGet(&cuDev, tritonCaptureDevice);
      cuDevicePrimaryCtxRelease(cuDev);
    }

    // Restore cuBLAS workspace to default (undo setCublasWorkspaceForCapture)
    restoreCublasWorkspaceAfterCapture(stream);

    // Reset thread-local state after capture attempt
    tl_captureWorkspace = nullptr;
    tl_captureWorkspaceSize = 0;
    tl_captureWorkspaceOffset = 0;
    // Reset host workspace thread-locals (ownership moves to tl_capturedHostPtrs → replay handle)
    tl_captureHostWorkspace = nullptr;
    tl_captureHostWorkspaceSize = 0;
    tl_captureHostWorkspaceOffset = 0;
    tl_graphCaptureStream = prevCaptureStream;
    // Pinned host ptrs: graph's H2D memcpy nodes reference these on replay.
    // On success: move to segment so they persist for graph lifetime.
    // On failure: free immediately (no graph to replay).
    if (usedTritonGraphCapture && seg.exec.replayHandle) {
      for (auto* ptr : tl_capturedHostPtrs) {
        seg.exec.replayHandle->addCapturedHostPtr(ptr);
      }
      DSP_DIAG(MEMORY, "preserved %zu pinned host ptrs for Triton graph replay",
                seg.exec.replayHandle->getCapturedHostPtrs().size());
    } else {
      // No graph captured — free pinned host ptrs immediately
      for (auto* ptr : tl_capturedHostPtrs) {
        cudaFreeHost(ptr);
      }
    }
    tl_capturedHostPtrs.clear();
    tl_captureReplicateCache.clear();

    // Arrays persist — no pendingClose_ flush needed after capture

    // Restore frozen context state so subsequent executions (including graph replay
    // steps that fall through to direct execution) use the frozen fast path.
    for (int s = seg.startSlot; s <= seg.endSlot; s++) {
      slots_[s].state_ = savedSlotStateTriton[s - seg.startSlot];
    }
    }  // end else (replayHandle != nullptr — workspace allocation succeeded)
  }
#endif

  if (!usedTritonGraphCapture) {
    // ── Batch-zero registration: learn which buffers actually get nullified ──
    // On the execution right before capture (executionCount == 1 → next call is
    // executionCount == 2 which triggers capture), enable registration mode.
    // Each nullify() site calls registerBatchZeroBuffer() when registering,
    // building the exact set of buffers that need zeroing.
    // This replaces the pre-scan approach (collectBatchZeroTargets) which
    // collected ~143 EXTRA buffers for slots that don't actually execute,
    // including buffers whose GPU addresses alias external KV cache inputs.
    bool batchZeroRegistrationActive = false;
#ifdef SD_CUDA
    {
      // Check the same conditions as shouldCaptureTritonGraph but for executionCount==1
      // (the warmup step right BEFORE capture). We register which buffers get nullified
      // so the batch-zero kernel during capture zeros EXACTLY the right set.
      // Registration doesn't require shapesFrozen_ — shapes may freeze after
      // this execution but before capture. We just need to be the pre-capture
      // warmup step (executionCount == 1) with no existing graph.
      bool wouldCaptureNextStep =
          Environment::getInstance().tritonGraphCapture() &&
          (!seg.exec.replayHandle || seg.exec.replayHandle->getCaptureBuffers().empty()) &&
          seg.exec.replayHandle == nullptr &&
          !seg.exec.compilationFailed &&
          seg.exec.executionCount == 1;
      if (Environment::getInstance().dspBatchZero() && wouldCaptureNextStep) {
        startBatchZeroRegistration();
        batchZeroRegistrationActive = true;
        DSP_DIAG_SEG(MEMORY, seg.startSlot, "batch-zero registration enabled for warmup execution (seg[%d-%d] execCount=%d)",
                  seg.startSlot, seg.endSlot, seg.exec.executionCount);
      }
    }
#endif

    // ── Sync external inputs to device BEFORE setting tl_graphExecutionActive ──
    // Triton's arg table population uses specialBuffer() to resolve GPU pointers.
    // specialBuffer() only calls syncToDevice() when the device buffer is nullptr
    // or on the wrong device — it does NOT check if the device data is stale.
    // Java modifies external inputs (attention_mask, position_ids, input_ids) on the
    // host via putScalar() + tagLocation(HOST), making the device data stale.
    // Native ops handle this via prepareSpecialUse() which calls syncToDevice()
    // unconditionally, but Triton bypasses native ops and reads device buffers directly.
    // We must sync BEFORE setting tl_graphExecutionActive because that flag changes
    // syncToSpecial() to use an async path that skips cudaStreamSynchronize.
    for (int ei = 0; ei < numExt; ei++) {
      if (externalArrays[ei] != nullptr) {
        if (Environment::getInstance().tritonVerifyKernels()) {
          auto* db = externalArrays[ei]->dataBuffer();
          DSP_DIAG(VERIFY, "SLOT_WRITE slot=%d tag=EXT_SYNC(direct) extIdx=%d pAct=%d sAct=%d len=%lld addr=%p",
                    -(ei + 1), ei,
                    db ? (db->isPrimaryActual() ? 1 : 0) : -1,
                    db ? (db->isSpecialActual() ? 1 : 0) : -1,
                    (long long)externalArrays[ei]->lengthOf(),
                    DSP_BUF(externalArrays[ei]));
        }
        externalArrays[ei]->syncToDevice();
      }
    }

    // NOTE: Do NOT set tl_graphExecutionActive=true here for non-capture Triton execution.
    // That flag suppresses syncToPrimary (D2H transfers), error checking, and
    // PointersManager sync -- behaviors only appropriate during CUDA graph capture.
    // The fallback lambda (gap ops) already handles capture detection independently:
    // it checks cudaStreamIsCapturing() and only sets tl_graphExecutionActive=true
    // when actually capturing.  Setting it unconditionally here caused gap ops
    // (matmul, gather, etc.) to read stale host data, producing wrong output.

    // Disable frozen fast path for gap ops during Triton segment execution.
    // Same rationale as the capture path (lines 5325-5329): the pre-execution
    // slot restoration at lines 4955-5032 may replace NDArray objects in
    // outputSlots_[], making the frozen context's cached input/output pointers
    // stale. Without clearing frozenContextReady, gap ops write to old arrays
    // while downstream ops read from new arrays, producing wrong output.
    // Save and restore so subsequent executions still benefit from frozen fast path.
    std::vector<NativeSlot::SlotState> savedSlotStateNonCapture(seg.endSlot - seg.startSlot + 1);
    for (int s = seg.startSlot; s <= seg.endSlot; s++) {
      savedSlotStateNonCapture[s - seg.startSlot] = slots_[s].state_;
      if (slots_[s].state_ >= NativeSlot::SlotState::FROZEN)
        slots_[s].state_ = NativeSlot::SlotState::SHAPE_CACHED;
    }

    // Snapshot addresses for direct execution (baseline for comparison with capture/replay)
    {
      std::vector<void*> outAddrs, extAddrs;
      extractDeviceAddrs(outputSlots_, totalOutputSlots_, outAddrs);
      extractDeviceAddrs(externalArrays, numExt, extAddrs);
      DSP_DIAG_SNAPSHOT_ADDRS("direct-entry", outAddrs.data(), totalOutputSlots_,
                               extAddrs.data(), numExt);
    }

    try {
      status = backend->executeSegment(seg, slots_, externalArrays, numExt,
                                       outputSlots_, totalOutputSlots_, stream);
    } catch (...) {
      // Restore frozenContextReady on exception
      for (int s = seg.startSlot; s <= seg.endSlot; s++) {
        slots_[s].state_ = savedSlotStateNonCapture[s - seg.startSlot];
      }
#ifdef SD_CUDA
      if (batchZeroRegistrationActive) {
        finishBatchZeroRegistration();
      }
#endif
      throw;  // Re-throw after cleanup
    }

    // Restore frozen context state so subsequent calls use the frozen fast path
    // once context pointers are re-established by the normal path above.
    for (int s = seg.startSlot; s <= seg.endSlot; s++) {
      slots_[s].state_ = savedSlotStateNonCapture[s - seg.startSlot];
    }

#ifdef SD_CUDA
    if (batchZeroRegistrationActive) {
      finishBatchZeroRegistration();
    }
#endif
  }

  // Dump final output for direct Triton path (baseline comparison)
  if (status == Status::OK && seg.endSlot < totalOutputSlots_ &&
      outputSlots_[seg.endSlot] != nullptr) {
    auto* finalOut = outputSlots_[seg.endSlot];
    if (finalOut->dataType() == FLOAT32) {
      DSP_DIAG_DUMP_SLOT("direct", seg.endSlot,
                         DSP_BUF(finalOut), finalOut->lengthOf());
    }
  }
  // Always-on diagnostic: dump top logit for non-capture Triton execution
  if (!usedTritonGraphCapture && status == Status::OK &&
      seg.endSlot < totalOutputSlots_ && outputSlots_[seg.endSlot] != nullptr) {
    auto* out = outputSlots_[seg.endSlot];
    if (out->dataType() == FLOAT32 && out->lengthOf() > 0) {
      DSP_DIAG_DUMP_SEG_OUTPUT("DIRECT_TRITON", seg.endSlot, DSP_BUF(out),
                               out->lengthOf(), seg.exec.executionCount, stream);
    }
  }

  DSP_DIAG(EXECUTE, "executeSegmentWithGpuGraph: exec%d seg[%d-%d]: backend=%s %s status=%d(%s) "
            "executionCount=%d compilationFailed=%d usedCapture=%d",
            seg.exec.executionCount, seg.startSlot, seg.endSlot,
            backendName, status == Status::OK ? "OK" : "FAILED",
            static_cast<int>(status), statusName_gpu(status),
            seg.exec.executionCount,
            seg.exec.compilationFailed ? 1 : 0, usedTritonGraphCapture ? 1 : 0);

  if (status == Status::OK) {
    seg.exec.executionCount++;
    totalGraphReplays_++;
    if (seg.exec.compiledByBackend.empty()) {
      seg.exec.compiledByBackend = backendName;
    }
  }

#ifdef SD_CUDA
  if (Environment::getInstance().tritonVerifyKernels()) {
    DSP_DIAG(VERIFY, "SEG_EXIT seg[%d-%d] status=%s execCount=%d",
              seg.startSlot, seg.endSlot, statusName_gpu(status), seg.exec.executionCount);
  }
#endif

  return status;
}

}  // namespace graph
}  // namespace sd

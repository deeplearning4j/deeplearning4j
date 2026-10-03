/*
*  ******************************************************************************
*  *
*  *
*  * This program and the accompanying materials are made available under the
*  * terms of the Apache License, Version 2.0 which is available at
*  * https://www.apache.org/licenses/LICENSE-2.0.
*  *
*  * See the NOTICE file distributed with this work for additional
*  * information regarding copyright ownership.
*  * Unless required by applicable law or agreed to in writing, software
*  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
*  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
*  * License for the specific language governing permissions and limitations
*  * under the License.
*  *
*  * SPDX-License-Identifier: Apache-2.0
*  *****************************************************************************
*/

//
// @author raver119@gmail.com
// @author Yurii Shyrma (iuriish@yahoo.com)
//
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_fp16.h>
#include <helpers/PointersManager.h>
#include <string>
#include <helpers/ShapeUtils.h>
#include <ops/specials_cuda.h>
#include <ops/op_types.h>
#include <ops/declarable/helpers/matmul.h>
#include <ops/declarable/helpers/cuda/device_primitives.cuh>

#include <algorithm>
#include <atomic>
#include <graph/DspDiagnostics.h>
#include <helpers/DebugHelper.h>
#include <memory>
#include <mutex>
#include <numeric>
#include <unordered_map>
#include <utility>

#include "../MmulHelper.h"
#include "../cublasHelper.h"
#include "execution/cuda/LaunchDims.h"
#include <system/env_functions.h>
#include <system/Environment.h>
#include <config.h>
#if HAVE_CUTLASS
#include <helpers/CutlassGemmHelper.h>
#include <helpers/CutlassHelper.h>
#endif

// Declared in NativeDynamicShapePlan_batchgemm.cu — true when cuBLAS stream+workspace
// already configured for DSP gap loop. When true, skip cublasSetStream +
// reapplyCublasWorkspace + prepareSpecialUse/registerSpecialUse (arrays device-resident).
extern SD_TLS_EXPORT thread_local bool tl_cublasGapStreamReady;

namespace sd { namespace graph {
int recordActiveMmulFingerprintTriplet(const void* aPtr, size_t aBytes,
                                       const void* bPtr, size_t bBytes,
                                       const void* cPtr, size_t cBytes,
                                       cudaStream_t intendedStream,
                                       cudaStream_t handleStream,
                                       int mathMode, int pointerMode, int atomicsMode,
                                       bool deterministicWindow, bool ltDisabled,
                                       const void* workspacePtr, size_t workspaceBytes);
void recordActiveMmulOutputFingerprint(int ordinal, const void* cPtr, size_t cBytes,
                                       cudaStream_t handleStream);
} }

namespace sd {

template <typename X, typename Y, typename Z>
static SD_KERNEL void serialGemmKernel(const X* x, const Y* y, Z* z,
    const LongType* xs, const LongType* ys, const LongType* zs,
    LongType length, bool tx, bool ty, double alpha, double beta) {
  for (LongType i = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < length; i += static_cast<LongType>(gridDim.x) * blockDim.x)
    ops::helpers::matmulSerialElement(i, x, y, z, xs, ys, zs, tx, ty, alpha, beta);
}

// Fixed algorithm geometry, not runtime tuning knobs. Padding avoids shared-bank
// conflicts both for contiguous-K staging and output-parallel consumption.
static constexpr int serialTileK = 32;
static constexpr int serialTileN = 128;
static constexpr int serialTileRows = 4;
static constexpr int serialTileSharedElements =
    serialTileN * (serialTileK + 1) + serialTileRows * serialTileK;

struct SerialTileLayout {
  LongType m, n, k, aRow, cRow, cCol;
};

static bool serialTileLayout(NDArray* x, NDArray* y, NDArray* z, bool tx, bool ty,
                             SerialTileLayout& layout) {
  if (z->dataType() != DataType::FLOAT32) return false;
  const int xr = x->rankOf(), yr = y->rankOf(), zr = z->rankOf();
  if ((xr != 2 && xr != 3) || (yr != 2 && yr != 3) || zr != std::max(xr, yr)) return false;
  if ((xr == 3 && x->sizeAt(0) != 1) || (yr == 3 && y->sizeAt(0) != 1) ||
      (zr == 3 && z->sizeAt(0) != 1)) return false;
  const int xm = xr - (tx ? 1 : 2), xk = xr - (tx ? 2 : 1);
  const int yk = yr - (ty ? 1 : 2), yn = yr - (ty ? 2 : 1);
  layout = {x->sizeAt(xm), y->sizeAt(yn), x->sizeAt(xk),
            shape::stride(x->shapeInfo())[xm], shape::stride(z->shapeInfo())[zr - 2],
            shape::stride(z->shapeInfo())[zr - 1]};
  if (layout.m <= 0 || layout.n <= 0 || layout.k <= 0 || y->sizeAt(yk) != layout.k ||
      z->sizeAt(zr - 2) != layout.m || z->sizeAt(zr - 1) != layout.n) return false;
  // Effective A rows have contiguous K; B is [K,N] with strides [1,K].
  // Shifted view bases are supported, including padded A rows and output rows/columns.
  if (shape::stride(x->shapeInfo())[xk] != 1 || layout.aRow < layout.k ||
      shape::stride(y->shapeInfo())[yk] != 1 || shape::stride(y->shapeInfo())[yn] != layout.k)
    return false;
  // Explicitly prove disjoint output rows or columns. Other output views retain
  // the general coordinate/stride kernel, rather than assuming contiguity.
  return (layout.cCol == 1 && layout.cRow >= layout.n) ||
         (layout.cRow == 1 && layout.cCol >= layout.m);
}

template <typename X, typename Y, typename Z, int Rows>
static SD_KERNEL void serialGemmTiledKernel(const X* x, const Y* y, Z* z,
                                           SerialTileLayout layout, double alpha, double beta) {
  using AccT = typename simdOps::AggregateType<Z>::type;
  extern __shared__ unsigned char sharedStorage[];
  auto* staged = reinterpret_cast<AccT*>(sharedStorage);
  AccT* aTile = staged + serialTileN * (serialTileK + 1);
  const LongType nTiles = (layout.n - 1) / serialTileN + 1;
  const LongType tiles = nTiles * ((layout.m - 1) / Rows + 1);
  const int lane = threadIdx.x;
  for (LongType tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
    const LongType firstRow = (tile / nTiles) * Rows;
    const LongType firstCol = (tile % nTiles) * serialTileN;
    const LongType col = firstCol + lane;
    // A single persistent accumulator per output: tiles stage operands only.
    AccT sums[Rows] = {};
    for (LongType firstK = 0; firstK < layout.k;) {
      const int activeK = static_cast<int>(layout.k - firstK < serialTileK ?
                                           layout.k - firstK : serialTileK);
      // Adjacent lanes load adjacent K, not widely separated output columns.
      for (int i = lane; i < serialTileN * serialTileK; i += blockDim.x) {
        const int n = i / serialTileK, k = i % serialTileK;
        if (firstCol + n < layout.n && k < activeK)
          staged[n * (serialTileK + 1) + k] = static_cast<AccT>(y[(firstCol + n) * layout.k + firstK + k]);
      }
      for (int i = lane; i < Rows * serialTileK; i += blockDim.x) {
        const int r = i / serialTileK, k = i % serialTileK;
        if (firstRow + r < layout.m && k < activeK)
          aTile[i] = static_cast<AccT>(x[(firstRow + r) * layout.aRow + firstK + k]);
      }
      __syncthreads();
      if (col < layout.n) {
        for (int k = 0; k < activeK; ++k) {
          const AccT weight = staged[lane * (serialTileK + 1) + k];
          for (int r = 0; r < Rows; ++r)
            if (firstRow + r < layout.m)
              sums[r] = ops::helpers::matmulFma(aTile[r * serialTileK + k], weight, sums[r]);
        }
      }
      // Tail output lanes participate too; no thread can overwrite the next
      // tile before every consumer finishes. No padded K values enter an FMA.
      __syncthreads();
      firstK += activeK;
    }
    if (col < layout.n) {
      for (int r = 0; r < Rows; ++r) {
        if (firstRow + r < layout.m) {
          const LongType offset = (firstRow + r) * layout.cRow + col * layout.cCol;
          AccT result = ops::helpers::matmulMultiply(static_cast<AccT>(alpha), sums[r]);
          if (beta != 0.0)
            result = ops::helpers::matmulFma(static_cast<AccT>(beta), static_cast<AccT>(z[offset]), result);
          z[offset] = static_cast<Z>(result);
        }
      }
    }
  }
}

template <typename X, typename Y, typename Z>
static void launchSerialGemmTiled(dim3 dims, cudaStream_t* stream, NDArray* x, NDArray* y, NDArray* z,
                                  SerialTileLayout layout, double alpha, double beta) {
  using AccT = typename simdOps::AggregateType<Z>::type;
  if (dims.z != serialTileSharedElements * sizeof(AccT))
    THROW_EXCEPTION("MATMUL SERIAL_FMA: shared memory does not match accumulator dtype");
  const auto* a = static_cast<const X*>(x->specialBuffer());
  const auto* b = static_cast<const Y*>(y->specialBuffer());
  auto* c = static_cast<Z*>(z->specialBuffer());
  const int rows = layout.m == 1 ? 1 : serialTileRows;
  const LongType tiles = ((layout.n - 1) / serialTileN + 1) * ((layout.m - 1) / rows + 1);
  const unsigned int blocks = static_cast<unsigned int>(std::min<LongType>(dims.x, tiles));
  if (rows == 1)
    serialGemmTiledKernel<X, Y, Z, 1><<<blocks, dims.y, dims.z, *stream>>>(a, b, c, layout, alpha, beta);
  else
    serialGemmTiledKernel<X, Y, Z, serialTileRows><<<blocks, dims.y, dims.z, *stream>>>(a, b, c, layout, alpha, beta);
}

template <typename X, typename Y = X, typename Z = X>
static void launchSerialGemm(dim3 dims, cudaStream_t* stream, NDArray* x, NDArray* y, NDArray* z,
                            bool tx, bool ty, double alpha, double beta) {
  const auto* xs = x->specialShapeInfo();
  const auto* ys = y->specialShapeInfo();
  const auto* zs = z->specialShapeInfo();
  serialGemmKernel<X, Y, Z><<<dims.x, dims.y, 0, *stream>>>(
      static_cast<const X*>(x->specialBuffer()), static_cast<const Y*>(y->specialBuffer()),
      static_cast<Z*>(z->specialBuffer()), xs, ys, zs, z->lengthOf(), tx, ty, alpha, beta);
}

void MmulHelper::matmulSerial(LaunchContext* context, NDArray* x, NDArray* y, NDArray* z,
                            bool tx, bool ty, double alpha, double beta) {
  if (!ops::helpers::matmulSerialStorageSupported(x->dataType(), y->dataType(), z->dataType()))
    THROW_EXCEPTION("MATMUL SERIAL_FMA: unsupported storage dtype combination");
  if (z->isEmpty()) return;
  if (x->getDataBuffer() == z->getDataBuffer() || y->getDataBuffer() == z->getDataBuffer())
    THROW_EXCEPTION("MATMUL SERIAL_FMA: output must not alias an input");
  SerialTileLayout layout{};
  const bool tiled = serialTileLayout(x, y, z, tx, ty, layout);
  const auto dims = getLaunchDims(tiled ? "matmul_serial_fma_tiled" : "matmul_serial_fma");
  // Both paths use only existing device buffers and the supplied stream.
  cudaDeviceProp properties;
  if (cudaGetDeviceProperties(&properties, context->getDeviceID()) != cudaSuccess)
    THROW_EXCEPTION("MATMUL SERIAL_FMA: unable to query launch limits");
  if (dims.x == 0 || dims.x > static_cast<unsigned int>(properties.maxGridSize[0]) ||
      dims.y == 0 || dims.y > static_cast<unsigned int>(properties.maxThreadsPerBlock) ||
      dims.y > static_cast<unsigned int>(properties.maxThreadsDim[0]))
    THROW_EXCEPTION("MATMUL SERIAL_FMA: invalid named launch dimensions");
  if (tiled) {
    if (dims.y != serialTileN || dims.z != serialTileSharedElements * z->sizeOfT() ||
        dims.z > properties.sharedMemPerBlock || properties.warpSize != serialTileK)
      THROW_EXCEPTION("MATMUL SERIAL_FMA: tiled launch requires 128 threads, 17408 shared bytes and 32-lane warps");
  } else if (dims.z != 0) {
    THROW_EXCEPTION("MATMUL SERIAL_FMA: general launch requires zero shared bytes");
  }
  if (beta != 0.0) NDArray::prepareSpecialUse({z}, {x, y, z});
  else NDArray::prepareSpecialUse({z}, {x, y});
  auto* stream = context->getCudaStream();
  if (tiled) {
    BUILD_TRIPLE_SELECTOR(x->dataType(), y->dataType(), z->dataType(), launchSerialGemmTiled,
                          (dims, stream, x, y, z, layout, alpha, beta),
                          SD_FLOAT_TYPES, SD_FLOAT_TYPES, SD_FLOAT_TYPES);
  } else {
    BUILD_TRIPLE_SELECTOR(x->dataType(), y->dataType(), z->dataType(), launchSerialGemm,
                          (dims, stream, x, y, z, tx, ty, alpha, beta),
                          SD_FLOAT_TYPES, SD_FLOAT_TYPES, SD_FLOAT_TYPES);
  }
  NDArray::registerSpecialUse({z}, {x, y});
  if (!DebugHelper::inGraphCapture(stream))
    DebugHelper::checkGlobalErrorCode("MATMUL SERIAL_FMA launch failed");
}

// Thread-local cublasLt epilogue state — set by DSP executor before matmul dispatch
struct LtEpilogueState {
  int type = 0;
  const void* biasPtr = nullptr;
  int64_t biasSize = 0;

  void set(int t, const void* bp, int64_t bs) { type = t; biasPtr = bp; biasSize = bs; }
  void clear() { type = 0; biasPtr = nullptr; biasSize = 0; }
};

static thread_local LtEpilogueState tl_ltEpilogue;

void MmulHelper::setLtEpilogue(int type, const void* biasPtr, int64_t biasSize) {
  tl_ltEpilogue.set(type, biasPtr, biasSize);
}

void MmulHelper::clearLtEpilogue() {
  tl_ltEpilogue.clear();
}

// Re-apply cuBLAS workspace after cublasSetStream whenever DSP configured one.
// Skip during CUDA graph capture: workspace was pre-set by setCublasWorkspaceForCapture
// before cudaStreamBeginCapture.  Calling cublasSetWorkspace on a capturing stream may
// inject an internal host-callback node into the graph, serialising every replay on CPU.
static inline void reapplyCublasWorkspace(cublasHandle_t handle) {
  if (!DebugHelper::inGraphCapture(nullptr) && tl_cublasWorkspacePtr != nullptr && tl_cublasWorkspaceSize > 0) {
    // The DSP cuBLAS workspace is a SINGLE buffer allocated on the primary device (0). Applying it
    // to a cuBLAS handle on a SECONDARY device (multi-GPU op-segment sharding) makes the gemm fail
    // with CUBLAS_STATUS_EXECUTION_FAILED — the workspace must live on the handle's device. On a
    // secondary device, verify the workspace actually resides here; if not, skip it and let cuBLAS
    // use its own default per-handle workspace on the correct device. The primary-device hot path
    // (curDev == 0) is unchanged — no extra query, no behavioural difference.
    int curDev = 0;
    cudaGetDevice(&curDev);
    if (curDev != 0) {
      cudaPointerAttributes wa;
      if (cudaPointerGetAttributes(&wa, tl_cublasWorkspacePtr) == cudaSuccess &&
          wa.type == cudaMemoryTypeDevice && wa.device != curDev) {
        cudaGetLastError();
        // Actively RESET the handle's workspace to cuBLAS's own default (not just skip): the
        // device-1 handle may already carry a stale device-0 workspace from an earlier
        // setCublasWorkspaceForWarmup/Capture at curDev==1. Leaving it set makes the gemm fail.
        cublasSetWorkspace(handle, nullptr, 0);
        return;
      }
      cudaGetLastError();
    }
    cublasSetWorkspace(handle, tl_cublasWorkspacePtr, tl_cublasWorkspaceSize);
  }
}

// Map sd::DataType to the corresponding cudaDataType constant used by cublasGemmEx /
// cublasLtMatmulDescCreate.  Returns true when the type is supported, false otherwise.
// BFLOAT16 requires CUDA 11+ (CUDA_R_16BF).
static inline bool sdToCudaDataType(DataType dt, cudaDataType& out) {
  switch (dt) {
    case FLOAT32:   out = CUDA_R_32F;  return true;
    case DOUBLE:    out = CUDA_R_64F;  return true;
    case HALF:      out = CUDA_R_16F;  return true;
    case BFLOAT16:  out = CUDA_R_16BF; return true;
    case INT8:      out = CUDA_R_8I;   return true;
    case FLOAT8:      out = CUDA_R_8F_E4M3; return true;
    case FLOAT8_E5M2: out = CUDA_R_8F_E5M2; return true;
    default:        return false;
  }
}

// Per-operand cast cache state for CUDA graph capture.
// During non-capture execution, cast results are cached here.
// During capture, cached buffers are reused via assign() to avoid
// capture workspace allocations that may not replay correctly.
// Consolidates what was 14 separate thread-locals into two struct instances.
struct CastCacheSide {
  std::vector<NDArray*> cache;
  size_t index = 0;
  std::unordered_map<const NDArray*, NDArray*> captureCastReuse;
  const NDArray* lastCaptureCastSource = nullptr;
  NDArray* lastCaptureCastArray = nullptr;
  // Per-slot source buffer tracking for skip-assign optimization.
  // When the source NDArray's DataBuffer pointer hasn't changed since the last
  // assign to a given cache slot, we skip the copy — constant weights (frozen
  // in DSP) are the same every decode step.
  // HAZARD: the pointer is only a valid identity while the DataBuffer object
  // lives. Plan teardown frees constants and the heap reuses their addresses;
  // a new plan's same-shape constant can then alias a retired entry and skip
  // the refresh. bumpCastCacheEpoch() (called at plan destruction) invalidates
  // these guards lazily on every thread via the epoch check below.
  std::vector<const void*> sourcePtrs;
  uint64_t epoch = 0;

  void resetIndices() {
    index = 0;
    captureCastReuse.clear();
    lastCaptureCastSource = nullptr;
    lastCaptureCastArray = nullptr;
  }

  void resetIndicesTo(size_t hwm) {
    // hwm is the first cache slot NOT baked into a captured CUDA graph. If it
    // equals cache.size(), start appending; wrapping to zero would let live gap
    // matmuls overwrite captured graph inputs.
    index = std::min(hwm, cache.size());
    captureCastReuse.clear();
    lastCaptureCastSource = nullptr;
    lastCaptureCastArray = nullptr;
  }

  // Arrays a mismatching cast replaced while a captured graph could still read
  // them. They are freed with the side, once no graph that baked them can replay.
  std::vector<NDArray*> retired;

  void clear() {
    for (auto* p : cache) delete p;
    cache.clear();
    for (auto* p : retired) delete p;
    retired.clear();
    index = 0;
    captureCastReuse.clear();
    lastCaptureCastSource = nullptr;
    lastCaptureCastArray = nullptr;
    sourcePtrs.clear();
  }
};

// The A and B slots of one execution unit. A captured CUDA graph bakes the
// device addresses of the slots it used, so each DSP segment casts into a scope
// of its own (MmulHelper::enterCastCacheScope). One thread-wide slot list was
// shared by every segment of every plan: a segment on another device, or with
// other shapes, re-cast (retiring the old array for good) or migrated a slot that
// another segment's graph still read, on every decode step. Work outside a
// segment uses the calling thread's default scope.
struct CastCacheScope {
  CastCacheSide a;
  CastCacheSide b;

  void resetIndices() {
    a.resetIndices();
    b.resetIndices();
  }

  void clear() {
    a.clear();
    b.clear();
  }
};

struct CastCacheScopeKey {
  const void* owner;
  LongType unit;

  bool operator==(const CastCacheScopeKey& other) const {
    return owner == other.owner && unit == other.unit;
  }
};

struct CastCacheScopeKeyHash {
  std::size_t operator()(const CastCacheScopeKey& key) const {
    std::size_t h = std::hash<const void*>{}(key.owner);
    h ^= std::hash<LongType>{}(key.unit) + 0x9e3779b9 + (h << 6) + (h >> 2);
    return h;
  }
};

// Segment scopes belong to their owner (the plan), not to a thread: a plan can be
// torn down on another thread than the one that executed it.
struct CastCacheScopeRegistry {
  std::mutex mutex;
  std::unordered_map<CastCacheScopeKey, std::unique_ptr<CastCacheScope>, CastCacheScopeKeyHash> scopes;
};

// Never destroyed: plans can be torn down during process exit, after static
// destructors have run.
static CastCacheScopeRegistry& castScopeRegistry() {
  static auto* registry = new CastCacheScopeRegistry();
  return *registry;
}

static thread_local CastCacheScope tl_defaultCastScope;
static thread_local CastCacheScope* tl_activeCastScope = nullptr;

static CastCacheScope& activeCastScope() {
  return tl_activeCastScope != nullptr ? *tl_activeCastScope : tl_defaultCastScope;
}

static CastCacheSide& castSideA() { return activeCastScope().a; }
static CastCacheSide& castSideB() { return activeCastScope().b; }

static size_t castArrayBytes(const std::vector<NDArray*>& arrays) {
  size_t bytes = 0;
  for (auto* array : arrays) {
    if (array != nullptr) {
      bytes += static_cast<size_t>(array->lengthOf()) * array->sizeOfT();
    }
  }
  return bytes;
}

// Persistent PINNED-HOST cuBLAS scalar parameters for CUDA graph capture.
// cuBLAS internally enqueues H2D memcpy from the alpha/beta host addresses.
// During graph capture, these H2D copies are baked into the graph with the
// original host source pointer.  On replay, the graph re-reads from the SAME
// host address.  Requirements:
//   1. Address must be stable across capture and all replays (not stack-local)
//   2. Memory must be PAGE-LOCKED (pinned) for async H2D in graph replay mode
//      (unpinned host memory causes graph replay to hang/stall)
// We allocate a small pinned host buffer per thread containing all scalar types.
struct CublasScalarsPinned {
  float  alphaF;
  float  betaF;
  double alphaD;
  double betaD;
  __half alphaH;
  __half betaH;
};

static CublasScalarsPinned* getCublasScalars() {
  static thread_local CublasScalarsPinned* ptr = nullptr;
  if (ptr == nullptr) {
    cudaError_t err = cudaHostAlloc(&ptr, sizeof(CublasScalarsPinned), cudaHostAllocDefault);
    if (err != cudaSuccess) {
      // Fallback: use regular heap (will work for non-graph execution)
      ptr = new CublasScalarsPinned();
    }
    ptr->alphaF = 1.0f;
    ptr->betaF  = 0.0f;
    ptr->alphaD = 1.0;
    ptr->betaD  = 0.0;
    ptr->alphaH = __float2half(1.0f);
    ptr->betaH  = __float2half(0.0f);
  }
  return ptr;
}

// cuBLAS Lt algorithm cache key: uniquely identifies a GEMM configuration
struct LtMatmulCacheKey {
  int deviceId;
  int M, N, K;
  cudaDataType aType, bType, cType;
  cublasOperation_t transA, transB;
  int epilogueType;  // 0=none, 1=bias, 2=bias+relu, 3=bias+gelu
  size_t workspaceSizeHint;
  int graphCaptureEnabled;
  int scaledOperands = 0;  // 1 when A/B carry cuBLASLt scalar scale pointers
  int deterministic = 0;   // 1 when selection excluded split-K reductions

  bool operator==(const LtMatmulCacheKey& other) const {
    return deviceId == other.deviceId && M == other.M && N == other.N && K == other.K &&
           aType == other.aType && bType == other.bType && cType == other.cType &&
           transA == other.transA && transB == other.transB &&
           epilogueType == other.epilogueType &&
           workspaceSizeHint == other.workspaceSizeHint &&
           graphCaptureEnabled == other.graphCaptureEnabled &&
           scaledOperands == other.scaledOperands && deterministic == other.deterministic;
  }
};

// Hash function for LtMatmulCacheKey
struct LtMatmulCacheKeyHash {
  std::size_t operator()(const LtMatmulCacheKey& key) const {
    std::size_t h = 0;
    h ^= std::hash<int>{}(key.deviceId) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<int>{}(key.M) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<int>{}(key.N) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<int>{}(key.K) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<int>{}(key.aType) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<int>{}(key.bType) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<int>{}(key.cType) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<int>{}(key.transA) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<int>{}(key.transB) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<int>{}(key.epilogueType) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<size_t>{}(key.workspaceSizeHint) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<int>{}(key.graphCaptureEnabled) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<int>{}(key.scaledOperands) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<int>{}(key.deterministic) + 0x9e3779b9 + (h << 6) + (h >> 2);
    return h;
  }
};

// Cached cuBLAS Lt algorithm descriptor
struct LtMatmulAlgoCacheEntry {
  cublasLtMatmulAlgo_t algo;
  size_t workspaceSize;
};

// Thread-local cuBLAS Lt algorithm cache
static thread_local std::unordered_map<LtMatmulCacheKey, LtMatmulAlgoCacheEntry, LtMatmulCacheKeyHash> tl_ltAlgoCache;

void MmulHelper::resetCastCacheIndices() {
  activeCastScope().resetIndices();
  // NOTE: tl_ltAlgoCache is intentionally NOT cleared here.
  // The algo cache is keyed by {M, N, K, types, flags} — different shapes have
  // separate entries, so there is no stale-key risk between steps.  Preserving
  // the cache ensures that capture and replay use the SAME algorithm as warmup
  // (selected with workspace=0).  Clearing it between warmup and capture forces
  // re-selection with a different workspace size, causing algorithm divergence
  // and numerical accuracy regression in merged CUDA graph replay.
}

void MmulHelper::resetCastCacheIndicesTo(size_t hwmA, size_t hwmB) {
  // Clamp to actual cache sizes — the high-water mark must not exceed them.
  // Clear per-call reuse maps: they were populated during capture and
  // are no longer valid for the current replay step's unmerged gap matmuls.
  auto& scope = activeCastScope();
  scope.a.resetIndicesTo(hwmA);
  scope.b.resetIndicesTo(hwmB);
  // NOTE: tl_ltAlgoCache is intentionally NOT cleared here — see resetCastCacheIndices().
}

std::pair<size_t, size_t> MmulHelper::getCastCacheHighWaterMark() {
  auto& scope = activeCastScope();
  return {scope.a.index, scope.b.index};
}

void MmulHelper::clearCastCache() {
  auto& scope = tl_defaultCastScope;
  DSP_DIAG(MEMORY,
           "CAST_CACHE_CLEAR arraysA=%zu arraysB=%zu retired=%zu bytesA=%zu bytesB=%zu retiredBytes=%zu",
           scope.a.cache.size(), scope.b.cache.size(),
           scope.a.retired.size() + scope.b.retired.size(),
           castArrayBytes(scope.a.cache), castArrayBytes(scope.b.cache),
           castArrayBytes(scope.a.retired) + castArrayBytes(scope.b.retired));
  scope.clear();
  tl_ltAlgoCache.clear();
}

void* MmulHelper::enterCastCacheScope(const void* owner, LongType unit) {
  CastCacheScope* previous = tl_activeCastScope;
  CastCacheScope* scope = nullptr;
  if (owner != nullptr) {
    auto& registry = castScopeRegistry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    auto& entry = registry.scopes[CastCacheScopeKey{owner, unit}];
    if (entry == nullptr) entry = std::make_unique<CastCacheScope>();
    scope = entry.get();
  }
  // Re-entering the active scope keeps its position: a nested path of the same
  // unit must not rewind slots the enclosing pass already handed out.
  if (scope != previous) {
    tl_activeCastScope = scope;
    activeCastScope().resetIndices();
  }
  return previous;
}

void MmulHelper::restoreCastCacheScope(void* previous) {
  tl_activeCastScope = static_cast<CastCacheScope*>(previous);
}

void MmulHelper::releaseCastCacheScopes(const void* owner) {
  if (owner == nullptr) return;
  std::vector<std::unique_ptr<CastCacheScope>> released;
  {
    auto& registry = castScopeRegistry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    for (auto it = registry.scopes.begin(); it != registry.scopes.end();) {
      if (it->first.owner == owner) {
        released.push_back(std::move(it->second));
        it = registry.scopes.erase(it);
      } else {
        ++it;
      }
    }
  }
  size_t arrays = 0, retiredArrays = 0, bytes = 0, retiredBytes = 0;
  for (auto& scope : released) {
    if (tl_activeCastScope == scope.get()) tl_activeCastScope = nullptr;
    for (CastCacheSide* side : {&scope->a, &scope->b}) {
      arrays += side->cache.size();
      bytes += castArrayBytes(side->cache);
      retiredArrays += side->retired.size();
      retiredBytes += castArrayBytes(side->retired);
    }
    scope->clear();
  }
  DSP_DIAG(MEMORY,
           "CAST_CACHE_RELEASE owner=%p scopes=%zu arrays=%zu retired=%zu bytes=%zu retiredBytes=%zu",
           owner, released.size(), arrays, retiredArrays, bytes, retiredBytes);
}

// Constant-free epoch: bumped whenever constant DataBuffers may have been
// freed (plan destruction). Each thread's cast cache lazily invalidates its
// skip-assign guards when it observes a newer epoch — pointer identity is
// not trustworthy across frees (heap/pool address reuse, ABA).
static std::atomic<uint64_t> g_castCacheEpoch{1};

void MmulHelper::bumpCastCacheEpoch() {
  uint64_t prev = g_castCacheEpoch.fetch_add(1, std::memory_order_relaxed);
  DSP_DIAG(MEMORY, "CAST_CACHE_EPOCH_BUMP: %llu -> %llu (constant DataBuffers freed — "
           "all threads' skip-assign guards invalidate at next cast)",
           (unsigned long long)prev, (unsigned long long)(prev + 1));
}

// Extern-linkage hook for array/cuda/DataBuffer.cu (same pattern as
// resetMergedCaptureTLS): a CONSTANT DataBuffer is being freed OUTSIDE plan
// teardown (e.g. Java-side SameDiff.close() closing weight arrays after the
// plan handle is already gone). Its object address can be heap-reused by the
// next graph's constant, so the pointer-keyed skip-assign guards must drop.
// Plan-teardown frees also route here — double bumps are harmless.
void notifyConstantBufferFreedForCastCache() {
  MmulHelper::bumpCastCacheEpoch();
}

static NDArray* castWithPersistentCache(CastCacheSide& side, NDArray* source, DataType targetType) {
  auto& cache = side.cache;
  auto& index = side.index;
  auto& srcPtrs = side.sourcePtrs;

  // Epoch check: after any plan teardown, stop trusting recorded source
  // pointers (freed DataBuffer addresses get reused). Cached cast arrays
  // stay — their CONTENT refreshes via assign() once the guard is dropped.
  {
    uint64_t nowEpoch = g_castCacheEpoch.load(std::memory_order_relaxed);
    if (side.epoch != nowEpoch) {
      DSP_DIAG(MEMORY, "CAST_CACHE_EPOCH_SYNC: thread cast cache guards invalidated "
               "(epoch %llu -> %llu, %zu skip-assign guards dropped, %zu cached arrays kept)",
               (unsigned long long)side.epoch, (unsigned long long)nowEpoch,
               srcPtrs.size(), cache.size());
      std::fill(srcPtrs.begin(), srcPtrs.end(), nullptr);
      side.captureCastReuse.clear();
      side.lastCaptureCastSource = nullptr;
      side.lastCaptureCastArray = nullptr;
      side.epoch = nowEpoch;
    }
  }

  if (index < cache.size()) {
    NDArray* cached = cache[index];
    if (cached != nullptr && cached->dataType() == targetType && cached->isSameShape(source)) {
      // Only immutable buffers may skip the refresh. Decode activations keep
      // stable DataBuffer addresses while their values change every step; using
      // pointer stability alone causes CUDA graph capture to omit the cast node
      // and replay stale warmup values from the cached buffer.
      auto* sourceDb = source->dataBuffer();
      const void* srcBufPtr = sourceDb != nullptr ? static_cast<const void*>(sourceDb) : nullptr;
      const bool sourceIsConstant = sourceDb != nullptr && sourceDb->isConstant;
      bool sourceChanged = (index >= srcPtrs.size()) || (srcPtrs[index] != srcBufPtr);
      if (!sourceIsConstant || sourceChanged) {
        cached->assign(source);
        // Track the source pointer for next time
        if (index >= srcPtrs.size()) srcPtrs.resize(index + 1, nullptr);
        srcPtrs[index] = srcBufPtr;
      }
      index++;
      return cached;
    }

    if (cached != nullptr) {
      if (tl_graphExecutionActive || tl_dspReplayActive) {
        side.retired.push_back(cached);
      } else {
        delete cached;
      }
    }
    cached = source->cast(targetType);
    cache[index] = cached;
    // Track source pointer
    const void* srcBufPtr = source->dataBuffer() != nullptr ? static_cast<const void*>(source->dataBuffer()) : nullptr;
    if (index >= srcPtrs.size()) srcPtrs.resize(index + 1, nullptr);
    srcPtrs[index] = srcBufPtr;
    index++;
    return cached;
  }

  NDArray* cached = source->cast(targetType);
  cache.push_back(cached);
  // Track source pointer
  const void* srcBufPtr = source->dataBuffer() != nullptr ? static_cast<const void*>(source->dataBuffer()) : nullptr;
  srcPtrs.push_back(srcBufPtr);
  index++;
  return cached;
}

// cuBLAS Lt matmul for decoder logits projection: [1,K] x [K,N] -> [1,N]
// ─── Shared cuBLASLt machinery ───────────────────────────────────────────────

// Multi-GPU sharding: the DSP cuBLAS workspace (tl_cublasWorkspacePtr) is a SINGLE buffer on the
// PRIMARY device (0). On a secondary device it is unusable — and cublasLt fails hard if an algo
// chosen FOR a workspace is then executed WITHOUT one (CUBLAS_STATUS_EXECUTION_FAILED). So resolve
// the EFFECTIVE workspace size ONCE and thread it through the algo-cache key, the heuristic
// preference, and execution: on a device that doesn't own the workspace it is 0, so the heuristic
// picks a no-workspace algo, the cache is partitioned per regime, and execution passes no
// workspace — all three stay consistent. Primary device is unchanged.
static size_t ltUsableWorkspaceSize() {
  size_t usable = 0;
  if (tl_cublasWorkspacePtr != nullptr && tl_cublasWorkspaceSize > 0) {
    usable = tl_cublasWorkspaceSize;
    int currentDevice = 0;
    cudaGetDevice(&currentDevice);
    cudaPointerAttributes attributes;
    if (cudaPointerGetAttributes(&attributes, tl_cublasWorkspacePtr) == cudaSuccess &&
        attributes.type == cudaMemoryTypeDevice && attributes.device != currentDevice) {
      usable = 0;  // workspace lives on another device — unusable here
    }
    cudaGetLastError();
  }
  return usable;
}

// One cuBLASLt problem description: the operation descriptor plus the A, B and
// C/D layouts. Destroys whatever it created; `valid` is set by the builder once
// every descriptor and attribute was created successfully.
struct LtMatmulDescriptors {
  cublasLtMatmulDesc_t operation = nullptr;
  cublasLtMatrixLayout_t a = nullptr;
  cublasLtMatrixLayout_t b = nullptr;
  cublasLtMatrixLayout_t c = nullptr;
  bool valid = false;

  ~LtMatmulDescriptors() {
    if (c != nullptr) cublasLtMatrixLayoutDestroy(c);
    if (b != nullptr) cublasLtMatrixLayoutDestroy(b);
    if (a != nullptr) cublasLtMatrixLayoutDestroy(a);
    if (operation != nullptr) cublasLtMatmulDescDestroy(operation);
  }
};

// Returns the algorithm cached for `key`, running the heuristic on a miss.
// `deterministic` restricts selection to algorithms without a split-K
// reduction: the chosen kernel is then bit-reproducible from run to run, which
// is what CUDA graph capture/replay and the DSP deterministic window require.
static bool ltSelectAlgorithm(cublasLtHandle_t handle, const LtMatmulCacheKey& key,
                              const LtMatmulDescriptors& descriptors, size_t maxWorkspace,
                              bool deterministic, LtMatmulAlgoCacheEntry& entry) {
  auto cached = tl_ltAlgoCache.find(key);
  if (cached != tl_ltAlgoCache.end()) {
    entry = cached->second;
    return true;
  }

  cublasLtMatmulPreference_t preference = nullptr;
  if (cublasLtMatmulPreferenceCreate(&preference) != CUBLAS_STATUS_SUCCESS) return false;
  cublasStatus_t status = cublasLtMatmulPreferenceSetAttribute(
      preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &maxWorkspace, sizeof(maxWorkspace));
  if (status == CUBLAS_STATUS_SUCCESS && deterministic) {
    const uint32_t reductionMask = CUBLASLT_REDUCTION_SCHEME_NONE;
    status = cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK,
                                                  &reductionMask, sizeof(reductionMask));
  }
  cublasLtMatmulHeuristicResult_t result;
  int returned = 0;
  if (status == CUBLAS_STATUS_SUCCESS) {
    status = cublasLtMatmulAlgoGetHeuristic(handle, descriptors.operation, descriptors.a, descriptors.b,
                                            descriptors.c, descriptors.c, preference, 1, &result, &returned);
  }
  cublasLtMatmulPreferenceDestroy(preference);
  if (status != CUBLAS_STATUS_SUCCESS || returned == 0) return false;

  entry = {result.algo, result.workspaceSize};
  tl_ltAlgoCache[key] = entry;
  return true;
}

// Launches D = alpha * op(A) * op(B) + beta * D. The scalars live in the
// persistent thread-local pinned block (not stack locals) so the host addresses
// baked into a captured CUDA graph stay valid on replay.
static bool ltLaunch(cublasLtHandle_t handle, const LtMatmulDescriptors& descriptors,
                     const LtMatmulAlgoCacheEntry& entry, double alpha, double beta,
                     const void* a, const void* b, void* d, cudaStream_t stream) {
  getCublasScalars()->alphaF = static_cast<float>(alpha);
  getCublasScalars()->betaF = static_cast<float>(beta);
  // The algorithm was selected against ltUsableWorkspaceSize(), so its size is
  // 0 on a device that does not own tl_cublasWorkspacePtr.
  void* workspace = nullptr;
  size_t workspaceSize = 0;
  if (entry.workspaceSize > 0 && tl_cublasWorkspacePtr != nullptr && entry.workspaceSize <= tl_cublasWorkspaceSize) {
    workspace = tl_cublasWorkspacePtr;
    workspaceSize = entry.workspaceSize;
  }
  return cublasLtMatmul(handle, descriptors.operation, &getCublasScalars()->alphaF, a, descriptors.a, b,
                        descriptors.b, &getCublasScalars()->betaF, d, descriptors.c, d, descriptors.c, &entry.algo,
                        workspace, workspaceSize, stream) == CUBLAS_STATUS_SUCCESS;
}

// Row-major C[M,N] = A[M,K] x B[K,N] expressed column-major as C^T = B^T x A^T:
// B[K,N] row-major is [N,K] column-major (ld N) and A[1,K] row-major is [K,1]
// column-major (ld K), so both Lt operands stay non-transposed and B is passed
// first. Optionally fuses a bias(+activation) epilogue.
static void buildRowMajorLtDescriptors(LtMatmulDescriptors& descriptors, int M, int N, int K,
                                       cudaDataType aType, cudaDataType bType, cudaDataType cType,
                                       int epilogueType, const void* biasPtr) {
  cublasStatus_t status = cublasLtMatmulDescCreate(&descriptors.operation, CUBLAS_COMPUTE_32F, CUDA_R_32F);
  const cublasOperation_t noTranspose = CUBLAS_OP_N;
  if (status == CUBLAS_STATUS_SUCCESS)
    status = cublasLtMatmulDescSetAttribute(descriptors.operation, CUBLASLT_MATMUL_DESC_TRANSA, &noTranspose,
                                            sizeof(noTranspose));
  if (status == CUBLAS_STATUS_SUCCESS)
    status = cublasLtMatmulDescSetAttribute(descriptors.operation, CUBLASLT_MATMUL_DESC_TRANSB, &noTranspose,
                                            sizeof(noTranspose));
  if (status == CUBLAS_STATUS_SUCCESS && epilogueType > 0 && biasPtr != nullptr) {
    cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_DEFAULT;
    if (epilogueType == 1) epilogue = CUBLASLT_EPILOGUE_BIAS;
    else if (epilogueType == 2) epilogue = CUBLASLT_EPILOGUE_RELU_BIAS;
    else if (epilogueType == 3) epilogue = CUBLASLT_EPILOGUE_GELU_BIAS;
    if (epilogue != CUBLASLT_EPILOGUE_DEFAULT) {
      status = cublasLtMatmulDescSetAttribute(descriptors.operation, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue,
                                              sizeof(epilogue));
      if (status == CUBLAS_STATUS_SUCCESS)
        status = cublasLtMatmulDescSetAttribute(descriptors.operation, CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                                &biasPtr, sizeof(biasPtr));
    }
  }
  if (status == CUBLAS_STATUS_SUCCESS) status = cublasLtMatrixLayoutCreate(&descriptors.a, bType, N, K, N);
  if (status == CUBLAS_STATUS_SUCCESS) status = cublasLtMatrixLayoutCreate(&descriptors.b, aType, K, M, K);
  if (status == CUBLAS_STATUS_SUCCESS) status = cublasLtMatrixLayoutCreate(&descriptors.c, cType, N, M, N);
  descriptors.valid = status == CUBLAS_STATUS_SUCCESS;
}

// Narrow fast path targeting large N (vocab projection) with mixed precision.
// Returns true if Lt matmul was executed, false to fall back to standard cuBLAS.
static bool tryLtMatmul(NDArray* A, NDArray* B, NDArray* C, double alpha, double beta,
                       NDArray* pA, NDArray* pB, NDArray* pC,
                       int M, int N, int K,
                       bool transA, bool transB,
                       cudaDataType aType, cudaDataType bType, cudaDataType cType,
                       int epilogueType = 0, const void* biasPtr = nullptr, int64_t biasSize = 0) {
  // Skip cublasLt when tl_cublasLtDisabled is set (CUDA_GRAPHS, AUTO capture).
  // This path selects algorithms without restricting split-K, whose internal
  // reductions are non-deterministic when replayed via CUDA graph (threadblock
  // scheduling order varies between replay iterations).
  if (tl_cublasLtDisabled) return false;

  // When epilogue fusion is requested, relax the gating — cublasLt handles general matmul.
  // Without epilogue, keep tight gating for the decoder logits projection fast path.
  if (epilogueType == 0) {
    if (M != 1) return false;
    if (cType != CUDA_R_32F) return false;
    if (bType != CUDA_R_16F) return false;
    if (aType != CUDA_R_32F && aType != CUDA_R_16F) return false;
    if (N < 16384) return false;  // Only for large vocab projections
    if (transA || !transB) return false;  // Expect row-major [1,K] x [K,N]
  }

  auto ltHandlePtr = CublasHelper::getInstance().ltHandle();
  if (ltHandlePtr == nullptr) return false;
  auto ltHandle = reinterpret_cast<cublasLtHandle_t*>(ltHandlePtr);
  auto stream = A->getContext()->getCudaStream();
  const size_t usableWorkspaceSize = ltUsableWorkspaceSize();

  LtMatmulCacheKey key;
  key.deviceId = AffinityManager::currentDeviceId();
  key.M = M;
  key.N = N;
  key.K = K;
  key.aType = aType;
  key.bType = bType;
  key.cType = cType;
  key.transA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
  key.transB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
  key.epilogueType = epilogueType;
  // Partitioned by effective workspace regime (usableWorkspaceSize, so a secondary device's
  // no-workspace regime is cached separately) and by graph-capture regime: tests toggle it
  // between methods and reusing Lt heuristics across regimes can select unstable algos.
  key.workspaceSizeHint = usableWorkspaceSize;
  key.graphCaptureEnabled = Environment::getInstance().tritonGraphCapture() ? 1 : 0;

  // During capture only an algorithm cached at warmup may be used; otherwise
  // fall back to standard cuBLAS.
  if (tl_graphExecutionActive && tl_ltAlgoCache.find(key) == tl_ltAlgoCache.end()) return false;

  LtMatmulDescriptors descriptors;
  buildRowMajorLtDescriptors(descriptors, M, N, K, aType, bType, cType, epilogueType, biasPtr);
  if (!descriptors.valid) return false;

  LtMatmulAlgoCacheEntry entry;
  if (!ltSelectAlgorithm(*ltHandle, key, descriptors, usableWorkspaceSize, false, entry)) return false;
  const bool launched = ltLaunch(*ltHandle, descriptors, entry, alpha, beta, pB->specialBuffer(),
                                 pA->specialBuffer(), pC->specialBuffer(), *stream);
  DSP_DIAG(BACKEND, "MmulHelper: cuBLASLt row-major matmul M=%d N=%d K=%d epilogue=%d launched=%d", M, N, K,
           epilogueType, launched ? 1 : 0);
  return launched;
}

// Largest row count of the decode class. Every call in the class runs the
// algorithm selected for exactly this width, so a row's reduction order — and
// therefore its bits — do not depend on how many rows the call carries.
// Speculative decoding relies on that: the same token position is computed in
// calls of different widths and must agree bit for bit with greedy decoding.
static constexpr LongType kLtDecodeClassRows = 16;

// Row-major z[rows, columns] = x[rows, depth] . w[columns, depth]^T is, column-major,
// z^T(columns x rows) = op_T(w: depth x columns, ld depth) * (x: depth x rows, ld depth)
// — the TN form low-precision cuBLASLt kernels require.
static void buildScaledLtDescriptors(LtMatmulDescriptors& descriptors, LongType rows, LongType columns,
                                     LongType depth, cudaDataType operandType, cudaDataType outputType,
                                     const float* xScale, const float* wScale) {
  cublasStatus_t status = cublasLtMatmulDescCreate(&descriptors.operation, CUBLAS_COMPUTE_32F, CUDA_R_32F);
  const cublasOperation_t transposeW = CUBLAS_OP_T;
  const cublasOperation_t noTranspose = CUBLAS_OP_N;
  if (status == CUBLAS_STATUS_SUCCESS)
    status = cublasLtMatmulDescSetAttribute(descriptors.operation, CUBLASLT_MATMUL_DESC_TRANSA, &transposeW,
                                            sizeof(transposeW));
  if (status == CUBLAS_STATUS_SUCCESS)
    status = cublasLtMatmulDescSetAttribute(descriptors.operation, CUBLASLT_MATMUL_DESC_TRANSB, &noTranspose,
                                            sizeof(noTranspose));
  if (status == CUBLAS_STATUS_SUCCESS && wScale != nullptr)
    status = cublasLtMatmulDescSetAttribute(descriptors.operation, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &wScale,
                                            sizeof(wScale));
  if (status == CUBLAS_STATUS_SUCCESS && xScale != nullptr)
    status = cublasLtMatmulDescSetAttribute(descriptors.operation, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &xScale,
                                            sizeof(xScale));
  if (status == CUBLAS_STATUS_SUCCESS)
    status = cublasLtMatrixLayoutCreate(&descriptors.a, operandType, depth, columns, depth);
  if (status == CUBLAS_STATUS_SUCCESS)
    status = cublasLtMatrixLayoutCreate(&descriptors.b, operandType, depth, rows, depth);
  if (status == CUBLAS_STATUS_SUCCESS)
    status = cublasLtMatrixLayoutCreate(&descriptors.c, outputType, columns, rows, columns);
  descriptors.valid = status == CUBLAS_STATUS_SUCCESS;
}

bool MmulHelper::ltMatmulScaled(LaunchContext* context, const void* x, const void* w, void* z,
                                LongType rows, LongType columns, LongType depth, DataType operandType,
                                DataType outputType, const float* xScale, const float* wScale) {
  cudaDataType operandCudaType, outputCudaType;
  if (!sdToCudaDataType(operandType, operandCudaType) || !sdToCudaDataType(outputType, outputCudaType))
    return false;
  auto ltHandlePtr = CublasHelper::getInstance().ltHandle();
  if (ltHandlePtr == nullptr) return false;
  auto ltHandle = reinterpret_cast<cublasLtHandle_t*>(ltHandlePtr);

  LtMatmulDescriptors launch;
  buildScaledLtDescriptors(launch, rows, columns, depth, operandCudaType, outputCudaType, xScale, wScale);
  if (!launch.valid) return false;

  // The algorithm is a pure function of the problem class. Selection uses no
  // workspace and the key carries no execution-regime state, because workspace
  // availability and capture mode change between warmup, capture and replay
  // and between plans; an algorithm that followed them would change the
  // reduction order, and the bits, of the same computation.
  const LongType selectionRows = rows <= kLtDecodeClassRows ? kLtDecodeClassRows : rows;
  LtMatmulCacheKey key;
  key.deviceId = AffinityManager::currentDeviceId();
  key.M = static_cast<int>(selectionRows);
  key.N = static_cast<int>(columns);
  key.K = static_cast<int>(depth);
  key.aType = operandCudaType;
  key.bType = operandCudaType;
  key.cType = outputCudaType;
  key.transA = CUBLAS_OP_T;
  key.transB = CUBLAS_OP_N;
  key.epilogueType = 0;
  key.workspaceSizeHint = 0;
  key.graphCaptureEnabled = 0;
  key.scaledOperands = 1;
  key.deterministic = 1;

  LtMatmulAlgoCacheEntry entry;
  if (selectionRows == rows) {
    if (!ltSelectAlgorithm(*ltHandle, key, launch, 0, true, entry)) return false;
  } else {
    LtMatmulDescriptors selection;
    buildScaledLtDescriptors(selection, selectionRows, columns, depth, operandCudaType, outputCudaType, xScale,
                             wScale);
    if (!selection.valid || !ltSelectAlgorithm(*ltHandle, key, selection, 0, true, entry)) return false;
    cublasLtMatmulHeuristicResult_t check;
    if (cublasLtMatmulAlgoCheck(*ltHandle, launch.operation, launch.a, launch.b, launch.c, launch.c, &entry.algo,
                                &check) != CUBLAS_STATUS_SUCCESS)
      THROW_EXCEPTION(("MmulHelper::ltMatmulScaled: the decode-class algorithm selected for " +
                       std::to_string(kLtDecodeClassRows) + " rows rejects " + std::to_string(rows) +
                       " rows; running it on another path would break bit-parity across decode widths")
                          .c_str());
  }
  return ltLaunch(*ltHandle, launch, entry, 1.0, 0.0, w, x, z, *context->getCudaStream());
}

//////////////////////////////////////////////////////////////////////////////
// MXK x KxN = MxN              -> actual sequence of axes doesn't matter
template <typename T1, typename T2, typename T3>
static SD_KERNEL void usualCudaGemm(const void* vA, const LongType* aShapeInfo, const void* vB,
                                   const LongType* bShapeInfo, void* vC, const LongType* cShapeInfo,
                                   const int aMaxis, const int aKaxis, const int bKaxis, const int bNaxis,
                                   const int cMaxis, const int cNaxis, const double alpha, const double beta) {
 using AccT = typename simdOps::AggregateType<T3>::type;
 // Cache shape information in shared memory
 __shared__ LongType K;
 __shared__ LongType cLen;
 __shared__ LongType totalThreads;
 __shared__ bool betaPresent;
 __shared__ AccT alphaZ;
 __shared__ AccT betaZ;
 __shared__ const LongType* aShape;
 __shared__ const LongType* bShape;
 __shared__ const LongType* cShape;
 __shared__ const LongType* aStride;
 __shared__ const LongType* bStride;
 __shared__ const LongType* cStride;
 __shared__ LongType aRank;
 __shared__ LongType bRank;
 __shared__ LongType cRank;
 __shared__ LongType* coords;

 if (threadIdx.x == 0) {
   extern __shared__ unsigned char shmem[];
   coords = reinterpret_cast<LongType*>(shmem);

   // Cache all shape information at start
   aRank = shape::rank(aShapeInfo);
   bRank = shape::rank(bShapeInfo);
   cRank = shape::rank(cShapeInfo);

   aShape = shape::shapeOf(aShapeInfo);
   bShape = shape::shapeOf(bShapeInfo);
   cShape = shape::shapeOf(cShapeInfo);

   aStride = shape::stride(aShapeInfo);
   bStride = shape::stride(bShapeInfo);
   cStride = shape::stride(cShapeInfo);

   cLen = shape::length(cShapeInfo);
   K = aShape[aKaxis];
   betaPresent = beta != 0;
   totalThreads = gridDim.x * blockDim.x;
   alphaZ = alpha;
   betaZ = beta;
 }
 __syncthreads();

 const T1* A = reinterpret_cast<const T1*>(vA);
 const T2* B = reinterpret_cast<const T2*>(vB);
 T3* C = reinterpret_cast<T3*>(vC);

 auto aCoords = coords + threadIdx.x * 6;  // 6 = (aRank + bRank + cRank)
 auto bCoords = aCoords + 2;
 auto cCoords = bCoords + 2;

 const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

 for (LongType i = tid; i < cLen; i += totalThreads) {
   // evaluate C coordinates
   INDEX2COORDS(i, cRank, cShape, cCoords);

   // evaluate A coordinates
   aCoords[aMaxis] = cCoords[cMaxis];
   aCoords[aKaxis] = 0;

   // evaluate B coordinates
   bCoords[bKaxis] = 0;
   bCoords[bNaxis] = cCoords[cNaxis];

   LongType aOffset, bOffset, cOffset;
   COORDS2INDEX(aRank, aStride, aCoords, aOffset);
   COORDS2INDEX(bRank, bStride, bCoords, bOffset);

   AccT val = static_cast<AccT>(A[aOffset]) * static_cast<AccT>(B[bOffset]);  // first iteration

   for (LongType j = 1; j < K; ++j) {  // rest iterations
     aOffset += aStride[aKaxis];
     bOffset += bStride[bKaxis];
     val = val + static_cast<AccT>(A[aOffset]) * static_cast<AccT>(B[bOffset]);
   }

   COORDS2INDEX(cRank, cStride, cCoords, cOffset);

   if (betaPresent)
     C[cOffset] = alphaZ * val + betaZ * static_cast<AccT>(C[cOffset]);
   else
     C[cOffset] = alphaZ * val;
 }
}

////////////////////////////////////////////////////////////////////////
template <typename T1, typename T2, typename T3>
SD_HOST static void usualGemm(const int blocksPerGrid, const int threadsPerBlock, const int sharedMem,
                             cudaStream_t* stream, const void* vA, const LongType* aShapeInfo, const void* vB,
                             const LongType* bShapeInfo, void* vC, const LongType* cShapeInfo,
                             const int aMaxis, const int aKaxis, const int bKaxis, const int bNaxis, const int cMaxis,
                             const int cNaxis, const double alpha, const double beta) {
 usualCudaGemm<T1, T2, T3><<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(
     vA, aShapeInfo, vB, bShapeInfo, vC, cShapeInfo, aMaxis, aKaxis, bKaxis, bNaxis, cMaxis, cNaxis, alpha, beta);
 DebugHelper::checkGlobalErrorCode("MMUL cuda gemv case failed(...) failed");
}

////////////////////////////////////////////////////////////////////////
// MXN x N = M  -> actual sequence of {M,N} axes doesn't matter
template <typename T1, typename T2, typename T3>
static SD_KERNEL void usualCudaGemv(const void* vA, const LongType* aShapeInfo, const void* vX,
                                   const LongType* xShapeInfo, void* vY, const LongType* yShapeInfo,
                                   const int incx, const int incy, const int aMaxis, const double alpha,
                                   const double beta) {

 using AccT = typename simdOps::AggregateType<T3>::type;
 // Cache shape information in shared memory
 __shared__ LongType M;
 __shared__ LongType N;
 __shared__ bool betaPresent;
 __shared__ LongType totalThreads;
 __shared__ LongType aNstride;
 __shared__ LongType aMstride;
 __shared__ AccT alphaZ;
 __shared__ AccT betaZ;
 __shared__ const LongType* aShape;
 __shared__ const LongType* aStride;
 __shared__ LongType aRank;

 if (threadIdx.x == 0) {
   N = shape::length(xShapeInfo);
   M = shape::length(yShapeInfo);
   aRank = shape::rank(aShapeInfo);

   aShape = shape::shapeOf(aShapeInfo);
   aStride = shape::stride(aShapeInfo);

   aMstride = aStride[aMaxis];
   aNstride = aStride[aMaxis == 0 ? 1 : 0];

   totalThreads = gridDim.x * blockDim.x;
   betaPresent = beta != 0;
   alphaZ = alpha;
   betaZ = beta;
 }
 __syncthreads();

 const T1* A = reinterpret_cast<const T1*>(vA);
 const T2* X = reinterpret_cast<const T2*>(vX);
 T3* Y = reinterpret_cast<T3*>(vY);

 const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

 for (LongType i = tid; i < M; i += totalThreads) {
   // evaluate offsets
   auto aOffset = i * aMstride;
   auto xOffset = 0;

   AccT val = static_cast<AccT>(A[aOffset]) * static_cast<AccT>(X[xOffset]);  // first iteration

   for (LongType j = 1; j < N; ++j) {  // rest iterations
     aOffset += aNstride;
     xOffset += incx;
     val = val + static_cast<AccT>(A[aOffset]) * static_cast<AccT>(X[xOffset]);
   }

   auto yOffset = i * incy;

   if (betaPresent)
     Y[yOffset] = alphaZ * val + betaZ * static_cast<AccT>(Y[yOffset]);
   else
     Y[yOffset] = alphaZ * val;
 }
}

////////////////////////////////////////////////////////////////////////
template <typename T1, typename T2, typename T3>
SD_HOST static void usualGemv(const int blocksPerGrid, const int threadsPerBlock, cudaStream_t* stream, const void* vA,
                             const LongType* aShapeInfo, const void* vX, const LongType* xShapeInfo, void* vY,
                             const LongType* yShapeInfo, const int incx, const int incy, const int aMaxis,
                             const double alpha, const double beta) {
 usualCudaGemv<T1, T2, T3><<<blocksPerGrid, threadsPerBlock, 512, *stream>>>(
     vA, aShapeInfo, vX, xShapeInfo, vY, yShapeInfo, incx, incy, aMaxis, alpha, beta);
 DebugHelper::checkGlobalErrorCode("MMUL cuda gemv case failed(...) failed");
}

//////////////////////////////////////////////////////////////////////////////
template <typename T1, typename T2, typename T3>
static SD_KERNEL void usualCudaDot(const LongType length, const double alpha, const void* vX,
                                  const LongType incx, const void* vY, const LongType incy, const double beta,
                                  void* vZ) {
 // Widen low-precision storage only; integer and double contracts stay unchanged.
 using AccT = typename simdOps::AggregateType<T3>::type;
 extern __shared__ unsigned char shmem[];
 AccT* partials = reinterpret_cast<AccT*>(shmem);

 const T1* X = reinterpret_cast<const T1*>(vX);
 const T2* Y = reinterpret_cast<const T2*>(vY);
 T3* Z = reinterpret_cast<T3*>(vZ);

 // The launch is a single block, so its threads cover every product and one shared-memory
 // reduction adds them all.
 AccT sum = static_cast<AccT>(0);
 for (LongType i = threadIdx.x; i < length; i += blockDim.x)
   sum = sum + static_cast<AccT>(X[i * incx]) * static_cast<AccT>(Y[i * incy]);
 partials[threadIdx.x] = sum;
 __syncthreads();

 // Halving tree; the upper half of an odd count folds into the lower half.
 for (unsigned int active = blockDim.x; active > 1;) {
   const unsigned int half = (active + 1) / 2;
   if (threadIdx.x < active - half) partials[threadIdx.x] = partials[threadIdx.x] + partials[threadIdx.x + half];
   __syncthreads();
   active = half;
 }

 if (threadIdx.x == 0) {
   const AccT result = static_cast<AccT>(alpha) * partials[0];
   if (beta != 0.0)
     *Z = static_cast<T3>(result + static_cast<AccT>(beta) * static_cast<AccT>(*Z));
   else
     *Z = static_cast<T3>(result);
 }
}

////////////////////////////////////////////////////////////////////////
template <typename T1, typename T2, typename T3>
SD_HOST static void usualDot(const dim3& launchDims, cudaStream_t* stream,
                            const LongType length, const double alpha, const void* vX, const LongType incx,
                            const void* vY, const LongType incy, const double beta, void* vZ) {
 using AccT = typename simdOps::AggregateType<T3>::type;
 // One block reduces the whole vector; its shared memory holds one partial sum per thread.
 usualCudaDot<T1, T2, T3><<<1, launchDims.y, launchDims.y * sizeof(AccT), *stream>>>(
     length, alpha, vX, incx, vY, incy, beta, vZ);
 DebugHelper::checkGlobalErrorCode("concat dot failed(...) failed");
}

////////////////////////////////////////////////////////////////////////
// [bS,M,K] x [bS,K,N] = [bS,M,N]
// [bS,M,K] x    [K,N] = [bS,M,N]
//    [M,K] x [bS,K,N] = [bS,M,N]
// bS could stand for several axes
template <typename T1, typename T2, typename T3>
static SD_KERNEL void batchedCudaGemm(const void* vA, const LongType* aShapeInfo, const void* vB,
                                     const LongType* bShapeInfo, void* vC, const LongType* cShapeInfo,
                                     const LongType* aBatchDims, const LongType* bBatchDims,
                                     const LongType* cBatchDims, const LongType aMaxis, const LongType aKaxis,
                                     const LongType bKaxis, const LongType bNaxis, const LongType cMaxis,
                                     const LongType cNaxis, const double alpha, const double beta) {

 using AccT = typename simdOps::AggregateType<T3>::type;
 // Cache shape information in shared memory
 __shared__ struct {
   bool betaPresent;
   LongType aRank, bRank, cRank, K;
   LongType cLen, totalThreads;
   AccT alphaZ, betaZ;
   const LongType* aShape;
   const LongType* bShape;
   const LongType* cShape;
   const LongType* aStride;
   const LongType* bStride;
   const LongType* cStride;
   LongType* coords;
 } shared;

 if (threadIdx.x == 0) {
   extern __shared__ unsigned char shmem[];
   shared.coords = reinterpret_cast<LongType*>(shmem);
   shared.cLen = shape::length(cShapeInfo);

   shared.aRank = shape::rank(aShapeInfo);
   shared.bRank = shape::rank(bShapeInfo);
   shared.cRank = shape::rank(cShapeInfo);

   shared.aShape = shape::shapeOf(aShapeInfo);
   shared.bShape = shape::shapeOf(bShapeInfo);
   shared.cShape = shape::shapeOf(cShapeInfo);

   shared.aStride = shape::stride(aShapeInfo);
   shared.bStride = shape::stride(bShapeInfo);
   shared.cStride = shape::stride(cShapeInfo);

   shared.K = shared.aShape[aKaxis];
   shared.betaPresent = beta != 0;
   shared.totalThreads = gridDim.x * blockDim.x;
   shared.alphaZ = alpha;
   shared.betaZ = beta;
 }
 __syncthreads();

 const T1* A = reinterpret_cast<const T1*>(vA);
 const T2* B = reinterpret_cast<const T2*>(vB);
 T3* C = reinterpret_cast<T3*>(vC);

 auto aCoords = shared.coords + threadIdx.x * (shared.aRank + shared.bRank + shared.cRank);
 auto bCoords = aCoords + shared.aRank;
 auto cCoords = bCoords + shared.bRank;

 const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

 for (LongType i = tid; i < shared.cLen; i += shared.totalThreads) {
   // evaluate C coordinates
   INDEX2COORDS(i, shared.cRank, shared.cShape, cCoords);

   // Copy batch coordinates directly from C to A and B.
   // Batch dimensions have matching shapes (validated in mmulNxN).
   // The old code computed a "batch index" using C's strides, then decomposed
   // it using A/B's shapes. That is wrong because stride-based offset != linear index
   // for non-contiguous arrays or when batch strides differ from shape products.
   if (aBatchDims != nullptr) {
     for (int d = 0; d < shared.aRank - 2; ++d) {
       aCoords[d] = cCoords[d];
     }
   }
   aCoords[aMaxis] = cCoords[cMaxis];
   aCoords[aKaxis] = 0;

   if (bBatchDims != nullptr) {
     for (int d = 0; d < shared.bRank - 2; ++d) {
       bCoords[d] = cCoords[d];
     }
   }
   bCoords[bKaxis] = 0;
   bCoords[bNaxis] = cCoords[cNaxis];

   LongType aOffset, bOffset, cOffset;
   COORDS2INDEX(shared.aRank, shared.aStride, aCoords, aOffset);
   COORDS2INDEX(shared.bRank, shared.bStride, bCoords, bOffset);

   AccT val = static_cast<AccT>(A[aOffset]) * static_cast<AccT>(B[bOffset]);  // first iteration

   for (LongType j = 1; j < shared.K; ++j) {  // rest iterations
     aOffset += shared.aStride[aKaxis];
     bOffset += shared.bStride[bKaxis];
     val = val + static_cast<AccT>(A[aOffset]) * static_cast<AccT>(B[bOffset]);
   }

   COORDS2INDEX(shared.cRank, shared.cStride, cCoords, cOffset);

   if (shared.betaPresent)
     C[cOffset] = shared.alphaZ * val + shared.betaZ * static_cast<AccT>(C[cOffset]);
   else
     C[cOffset] = shared.alphaZ * val;
 }
}

////////////////////////////////////////////////////////////////////////
template <typename T1, typename T2, typename T3>
SD_HOST static void batchedGemm(const int blocksPerGrid, const int threadsPerBlock, const int sharedMem,
                               cudaStream_t* stream, const void* vA, const LongType* aShapeInfo, const void* vB,
                               const LongType* bShapeInfo, void* vC, const LongType* cShapeInfo,
                               const LongType* aBatchDims, const LongType* bBatchDims, const LongType* cBatchDims,
                               const LongType aMaxis, const LongType aKaxis, const LongType bKaxis,
                               const LongType bNaxis, const LongType cMaxis, const LongType cNaxis, const double alpha,
                               const double beta) {
 batchedCudaGemm<T1, T2, T3><<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(
     vA, aShapeInfo, vB, bShapeInfo, vC, cShapeInfo, aBatchDims, bBatchDims, cBatchDims, aMaxis, aKaxis, bKaxis,
     bNaxis, cMaxis, cNaxis, alpha, beta);
 DebugHelper::checkGlobalErrorCode("batch gemm failed(...) failed");
}

// Use the persistent cast cache while DSP owns graph/replay state or the cuBLAS workspace.
// Warmup fills the cache, capture records stable cache buffer addresses, and replay reuses
// the same buffers after resetCastCacheIndices().
static bool matmulUsesCastCache() {
 return tl_graphExecutionActive || tl_dspReplayActive || tl_cublasGapStreamReady ||
        tl_cublasWorkspacePtr != nullptr;
}

// source in computeType: source itself, a persistent cache buffer, or a new array that the
// caller owns through owned. A cast of C carries C's values, so beta keeps its meaning.
static NDArray* castForMixedGemm(CastCacheSide& side, NDArray* source, DataType computeType,
                                 std::vector<NDArray*>& owned) {
 if (source->dataType() == computeType) return source;
 if (matmulUsesCastCache()) return castWithPersistentCache(side, source, computeType);
 owned.push_back(source->cast(computeType));
 return owned.back();
}

//////////////////////////////////////////////////////////////////////////////
// Mixed GEMV (ops::helpers::MixedGemvLayout): both kernels load W, x and y in their storage
// types and sum in the accumulator type, so no operand is copied.

// The products a thread loads before it adds them, in its summation order: enough loads in
// flight that a decode-size GEMV (a few hundred to a few thousand rows) keeps memory busy.
static constexpr int mixedGemvStagedLoads = 8;

template <typename AccT, typename Y>
static SD_DEVICE SD_INLINE void storeMixedGemvRow(Y* output, AccT sum, double alpha, double beta) {
 AccT result = static_cast<AccT>(alpha) * sum;
 if (beta != 0.0) result += static_cast<AccT>(beta) * static_cast<AccT>(*output);
 *output = static_cast<Y>(result);
}

// Depth-major W: one warp per row. The lanes stride over the row's depth, so a warp loads 32
// adjacent weights, and the warp adds the lanes' sums.
template <typename W, typename X, typename Y>
static SD_KERNEL void mixedGemvRowKernel(const W* w, const X* x, Y* y, ops::helpers::MixedGemvLayout layout,
                                        double alpha, double beta) {
 using AccT = ops::helpers::ProductAccumulator<W, X, Y>;
 const int lane = threadIdx.x % sd::device::WARP_SIZE;
 const LongType warpsPerBlock = blockDim.x / sd::device::WARP_SIZE;
 // A row belongs to a whole warp, so all 32 lanes reach the shuffles of warpReduceSum.
 for (LongType row = blockIdx.x * warpsPerBlock + threadIdx.x / sd::device::WARP_SIZE; row < layout.rows;
      row += static_cast<LongType>(gridDim.x) * warpsPerBlock) {
   const W* weights = w + row * layout.rowStride;
   AccT sum = static_cast<AccT>(0);
   for (LongType first = lane; first < layout.depth; first += sd::device::WARP_SIZE * mixedGemvStagedLoads) {
     AccT products[mixedGemvStagedLoads];
#pragma unroll
     for (int j = 0; j < mixedGemvStagedLoads; j++) {
       const LongType k = first + j * sd::device::WARP_SIZE;
       products[j] = k < layout.depth
                         ? static_cast<AccT>(weights[k * layout.depthStride]) * static_cast<AccT>(x[k * layout.xStride])
                         : static_cast<AccT>(0);
     }
#pragma unroll
     for (int j = 0; j < mixedGemvStagedLoads; j++) sum += products[j];
   }
   sum = sd::device::warpReduceSum(sum);
   if (lane == 0) storeMixedGemvRow(y + row * layout.yStride, sum, alpha, beta);
 }
}

// Row-major W: lane i of each warp takes row 32 * tile + i, so a warp loads 32 adjacent weights,
// and the block's warps (threadIdx.y) split the depth. Thread row y == 0 adds the warps' sums in
// order.
template <typename W, typename X, typename Y>
static SD_KERNEL void mixedGemvColumnKernel(const W* w, const X* x, Y* y, ops::helpers::MixedGemvLayout layout,
                                           double alpha, double beta) {
 using AccT = ops::helpers::ProductAccumulator<W, X, Y>;
 extern __shared__ double mixedGemvShared[];
 AccT* partials = reinterpret_cast<AccT*>(mixedGemvShared);
 const LongType tiles = (layout.rows + sd::device::WARP_SIZE - 1) / sd::device::WARP_SIZE;
 // A tile belongs to the whole block, so every thread reaches both barriers.
 for (LongType tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
   const LongType row = tile * sd::device::WARP_SIZE + threadIdx.x;
   AccT sum = static_cast<AccT>(0);
   if (row < layout.rows) {
     const W* weights = w + row * layout.rowStride;
     const LongType step = blockDim.y;
     for (LongType first = threadIdx.y; first < layout.depth; first += step * mixedGemvStagedLoads) {
       AccT products[mixedGemvStagedLoads];
#pragma unroll
       for (int j = 0; j < mixedGemvStagedLoads; j++) {
         const LongType k = first + j * step;
         products[j] = k < layout.depth
                           ? static_cast<AccT>(weights[k * layout.depthStride]) * static_cast<AccT>(x[k * layout.xStride])
                           : static_cast<AccT>(0);
       }
#pragma unroll
       for (int j = 0; j < mixedGemvStagedLoads; j++) sum += products[j];
     }
   }
   partials[threadIdx.y * sd::device::WARP_SIZE + threadIdx.x] = sum;
   __syncthreads();
   if (threadIdx.y == 0 && row < layout.rows) {
     AccT total = static_cast<AccT>(0);
     for (unsigned int part = 0; part < blockDim.y; ++part) total += partials[part * sd::device::WARP_SIZE + threadIdx.x];
     storeMixedGemvRow(y + row * layout.yStride, total, alpha, beta);
   }
   __syncthreads();
 }
}

template <typename W, typename X, typename Y>
static void launchMixedGemv(dim3 dims, cudaStream_t* stream, const void* w, const void* x, void* y,
                           ops::helpers::MixedGemvLayout layout, double alpha, double beta) {
 using AccT = ops::helpers::ProductAccumulator<W, X, Y>;
 const auto* weights = static_cast<const W*>(w);
 const auto* vector = static_cast<const X*>(x);
 auto* output = static_cast<Y*>(y);
 const unsigned int warps = dims.y / sd::device::WARP_SIZE;
 if (layout.depthMajor()) {
   const auto blocks = static_cast<unsigned int>(std::min<LongType>(dims.x, (layout.rows + warps - 1) / warps));
   mixedGemvRowKernel<W, X, Y><<<blocks, dims.y, 0, *stream>>>(weights, vector, output, layout, alpha, beta);
 } else {
   const LongType tiles = (layout.rows + sd::device::WARP_SIZE - 1) / sd::device::WARP_SIZE;
   const auto blocks = static_cast<unsigned int>(std::min<LongType>(dims.x, tiles));
   const size_t shared = std::max<size_t>(dims.z, static_cast<size_t>(dims.y) * sizeof(AccT));
   mixedGemvColumnKernel<W, X, Y><<<blocks, dim3(sd::device::WARP_SIZE, warps), shared, *stream>>>(
       weights, vector, output, layout, alpha, beta);
 }
}

// y = alpha * w x + beta * y through layout on the context's stream; the caller admits the types
// with ops::helpers::mixedGemvApplies.
static void mixedGemv(LaunchContext* context, NDArray* w, NDArray* x, NDArray* y,
                      const ops::helpers::MixedGemvLayout& layout, double alpha, double beta) {
 const dim3 dims = getLaunchDims(layout.depthMajor() ? "mixed_gemv_rows" : "mixed_gemv_columns");
 if (dims.x == 0 || dims.y == 0 || dims.y % sd::device::WARP_SIZE != 0 || dims.y > SD_MAX_NUM_THREADS)
   THROW_EXCEPTION("MmulHelper mixed GEMV: the named launch needs a grid and a block of whole warps");
 // In a DSP gap loop the operands are already device-resident (tl_cublasGapStreamReady).
 if (!tl_cublasGapStreamReady) NDArray::prepareSpecialUse({y}, {w, x, beta != 0.0 ? y : nullptr});
 auto* stream = context->getCudaStream();
 BUILD_TRIPLE_SELECTOR(w->dataType(), x->dataType(), y->dataType(), launchMixedGemv,
                       (dims, stream, w->specialBuffer(), x->specialBuffer(), y->specialBuffer(), layout, alpha, beta),
                       SD_FLOAT_TYPES, SD_FLOAT_TYPES, SD_FLOAT_TYPES);
 if (!tl_cublasGapStreamReady) NDArray::registerSpecialUse({y}, {w, x});
 if (!DebugHelper::inGraphCapture(stream)) DebugHelper::checkGlobalErrorCode("MmulHelper mixed GEMV failed");
}

// No pending cuBLASLt epilogue: the mixed GEMV writes the plain product.
static bool mixedGemvAdmitted(DataType matrix, DataType vector, DataType output) {
 return tl_ltEpilogue.type == 0 && ops::helpers::mixedGemvApplies(matrix, vector, output);
}

//////////////////////////////////////////////////////////////////////////////
// MXK x KxN = MxN
NDArray* MmulHelper::mmulMxM(NDArray* A, NDArray* B, NDArray* C, double alpha, double beta,
                            const char outOrder) {
 if (A->rankOf() != 2) THROW_EXCEPTION("MmulHelper::mmulMxM cuda: rank of A array is not equal 2 !");
 if (B->rankOf() != 2) THROW_EXCEPTION("MmulHelper::mmulMxM cuda: rank of B array is not equal 2 !");

 const auto M = A->sizeAt(0);
 const auto K = A->sizeAt(1);
 const auto N = B->sizeAt(1);

 if (C != nullptr && C->rankOf() != 2)
   THROW_EXCEPTION("MmulHelper::mmulMxM cuda: rank of C array is not equal 2 !");
 if (B->sizeAt(0) != K) THROW_EXCEPTION("MmulHelper::mmulMxM cuda: B array has wrong number of rows !");
 if (C != nullptr && C->sizeAt(0) != M)
   THROW_EXCEPTION("MmulHelper::mmulMxM cuda: C array has wrong number of rows !");
 if (C != nullptr && C->sizeAt(1) != N)
   THROW_EXCEPTION("MmulHelper::mmulMxM cuda: C array has wrong number of columns !");

 std::vector<LongType> cShape = {M, N};
 if (C == nullptr)
   C = new NDArray(outOrder, cShape, ops::helpers::matmulOutputType(A->dataType(), B->dataType()),
                   A->getContext());

 if (C->isEmpty()) return C;

 const int major = sd::env_deviceCapabilityMajor(AffinityManager::currentDeviceId());

#if HAVE_CUTLASS
 // Try CUTLASS dispatch before cuBLAS path.
 // CUTLASS is preferred for: M > 1 with matching types on SM80+,
 // and FP8 inputs on SM89+ (Ada Lovelace).
 {
   int smVersion = CutlassHelper::getSmVersion(AffinityManager::currentDeviceId());
   if (CutlassGemmHelper::shouldUseCutlass(M, N, K, A->dataType(), B->dataType(), smVersion)) {
     if (CutlassGemmHelper::gemm(A, B, C, alpha, beta)) {
       return C;
     }
     // CUTLASS failed (e.g., unsupported config) — fall through to cuBLAS
   }
 }
#endif

 // One output row or column against a matrix whose storage type is not the vector's: the mixed
 // GEMV reads the matrix in place instead of widening it.
 if (M == 1 || N == 1) {
   NDArray* matrix = M == 1 ? B : A;
   NDArray* vector = M == 1 ? A : B;
   if (mixedGemvAdmitted(matrix->dataType(), vector->dataType(), C->dataType())) {
     mixedGemv(A->getContext(), matrix, vector, C, ops::helpers::mixedGemvLayoutOfGemm(A, B, C), alpha, beta);
     return C;
   }
 }

 const auto aType = A->dataType();
 const auto bType = B->dataType();
 const auto cType = C->dataType();

 // Mixed-precision handling: when one input is FLOAT32 and the other is HALF,
 // Mixed-type handling: when one operand is HALF and the other is FLOAT32
 // (common with FP16 weight pre-casting via GraphOptimizer), upcast the HALF
 // operand to FLOAT32 so both match. This preserves activation precision
 // (no downcast of FLOAT32→HALF which loses bits and compounds across layers)
 // while cuBLAS gets same-type inputs it requires.
 //
 // NOTE: cuBLAS does NOT support mixed A/B types (CUDA_R_16F × CUDA_R_32F) —
 // it returns CUBLAS_STATUS_NOT_SUPPORTED. Both must match.
 //
 NDArray* castA = nullptr;
 NDArray* castB = nullptr;
 NDArray* effA = const_cast<NDArray*>(A);
 NDArray* effB = const_cast<NDArray*>(B);

 bool useCastCache = matmulUsesCastCache();

 // NOTE: FP16 autocast for FP32×FP32 matmul REMOVED.
 // Casting FP32 inputs to HALF loses precision on the input data itself,
 // causing incorrect results for models that require FP32 fidelity (e.g.,
 // gdn_qkv output was 7x attenuated vs CPU reference). FP32 accumulation
 // only helps during the dot product — it cannot recover precision lost
 // at the cast step. Use cublasSgemm (pure FP32) for FP32 model weights.

 // Mixed HALF×FLOAT32: upcast the HALF operand to FLOAT32 (preserves precision).
 // NEVER downcast FLOAT32→HALF — that loses activation precision across 30+ layers.
 // The upcast is always done here so the cublasGemmEx fallback path works correctly.
 // The cublasLt path (tryLtMatmul) uses origAType/origBType for descriptors but
 // the upcast data is harmless since Lt will handle the type mismatch via its API.
 if (cType == FLOAT32 && A->dataType() != B->dataType() && major >= 6) {
   if (A->dataType() == HALF && B->dataType() == FLOAT32) {
     // Weight is HALF, activation is FLOAT32 → upcast weight to FLOAT32
     if (useCastCache) {
       effA = castWithPersistentCache(castSideA(), effA, FLOAT32);
     } else {
       castA = effA->cast(FLOAT32);
       effA = castA;
     }
   } else if (A->dataType() == FLOAT32 && B->dataType() == HALF) {
     // Activation is FLOAT32, weight is HALF → upcast weight to FLOAT32
     if (useCastCache) {
       effB = castWithPersistentCache(castSideB(), effB, FLOAT32);
     } else {
       castB = effB->cast(FLOAT32);
       effB = castB;
     }
   }
 }

 // cuBLAS GEMM paths require the buffers passed for A, B and C to match the
 // descriptor types.  The generic fallback below is type-homogeneous too
 // (BUILD_SINGLE_SELECTOR_THRICE), so mixed FLOAT32/DOUBLE inputs with DOUBLE
 // output must be normalized before dispatch.  Otherwise the fallback can
 // reinterpret a FLOAT32 weight buffer as DOUBLE and produce NaNs.
 if (cType == DOUBLE && (effA->dataType() != DOUBLE || effB->dataType() != DOUBLE)) {
   const auto currentAType = effA->dataType();
   const auto currentBType = effB->dataType();
   const bool supportedA = currentAType == DOUBLE || currentAType == FLOAT32 || currentAType == HALF;
   const bool supportedB = currentBType == DOUBLE || currentBType == FLOAT32 || currentBType == HALF;
   if (supportedA && supportedB) {
     if (currentAType != DOUBLE) {
       if (useCastCache) {
         effA = castWithPersistentCache(castSideA(), effA, DOUBLE);
       } else {
         castA = effA->cast(DOUBLE);
         effA = castA;
       }
     }
     if (currentBType != DOUBLE) {
       if (useCastCache) {
         effB = castWithPersistentCache(castSideB(), effB, DOUBLE);
       } else {
         castB = effB->cast(DOUBLE);
         effB = castB;
       }
     }
   }
 }

 const auto effAType = effA->dataType();
 const auto effBType = effB->dataType();

 const bool AB(effAType == effBType), AC(effAType == cType), ABC(AB && AC);

 const bool typeDouble = ABC && effAType == DOUBLE;
 const bool typeFloat = ABC && effAType == FLOAT32;
 const bool typeHalf = ABC && effAType == HALF && major >= 6;
 const bool typeBfloat = ABC && effAType == BFLOAT16 && major >= 8;
 const bool typeIntFloat = AB && effAType == INT8 && cType == FLOAT32 && major >= 6;
 const bool typeHalfFloat = AB && effAType == HALF && cType == FLOAT32 && major >= 6;

 // The generic kernel below is instantiated on A's type for all three operands, so storage
 // no typed cuBLAS path covers (e.g. FLOAT32 x BFLOAT16 -> BFLOAT16) would be read and
 // written with A's element size. Compute it in one type instead and assign into C.
 // This runs before the device lock, which the same-type call takes itself.
 if (!ABC && !typeIntFloat && !typeHalfFloat) {
   const DataType computeType = ops::helpers::mixedGemmComputeType(effAType, effBType, cType);
   std::vector<NDArray*> owned;
   NDArray* computeA = castForMixedGemm(castSideA(), effA, computeType, owned);
   NDArray* computeB = castForMixedGemm(castSideB(), effB, computeType, owned);
   NDArray* computeC = castForMixedGemm(castSideB(), C, computeType, owned);
   mmulMxM(computeA, computeB, computeC, alpha, beta, outOrder);
   if (computeC != C) C->assign(computeC);
   for (auto* array : owned) deleteTemporary(array);
   deleteTemporary(castA);
   deleteTemporary(castB);
   return C;
 }

 std::lock_guard<std::mutex> lock(*LaunchContext::deviceMutex());

 auto handle = reinterpret_cast<cublasHandle_t*>(A->getContext()->getCublasHandle());
 auto stream = A->getContext()->getCudaStream();


  // N107: Skip cublasSetStream + workspace when DSP gap stream already configured.
  // The gap loop's CublasGapStreamGuard sets the cuBLAS handle stream+workspace once
  // at gap-loop start. Skipping here eliminates 2 cuBLAS host-API calls × 91 matmuls/step.
  // Also preserve the preconfigured handle throughout the outer composite-capture
  // scope. Native gaps can run before the first cudaStreamBeginCapture; resetting
  // the handle there sends GEMMs to the array context stream while the following
  // gap kernels use the composite stream, creating an intra-gap data race.
  cublasStatus_t status = CUBLAS_STATUS_SUCCESS;
  if (!tl_cublasGapStreamReady && !DebugHelper::inGraphCapture(nullptr)) {
    status = cublasSetStream_v2(*handle, *stream);
    if (status != CUBLAS_STATUS_SUCCESS) {
      std::string msg = "MmulHelper::mmulMxM cuda failed [cublasSetStream]; Error code: [" + std::to_string(status) + "]";
      THROW_EXCEPTION(msg.c_str());
    }
    reapplyCublasWorkspace(*handle);
  }

 if (!typeDouble && !typeFloat && !typeHalf && !typeBfloat && !typeIntFloat && !typeHalfFloat) {
   dim3 dims = getMMulDims(C->lengthOf(),DataTypeUtils::sizeOf(cType));
   // beta reads C, so C's current values must reach the device first.
   if (!tl_cublasGapStreamReady) NDArray::prepareSpecialUse({C}, {effA, effB, beta != 0.0 ? C : nullptr});
   BUILD_SINGLE_SELECTOR_THRICE(aType, usualGemm,
                                (dims.x, dims.y, dims.z, stream, effA->specialBuffer(),
                                 effA->specialShapeInfo(), effB->specialBuffer(), effB->specialShapeInfo(), C->specialBuffer(),
                                 C->specialShapeInfo(), 0, 1, 0, 1, 0, 1, alpha, beta),
                                SD_NUMERIC_TYPES)
   if (!tl_cublasGapStreamReady) NDArray::registerSpecialUse({C}, {effA, effB});

 } else {
   std::vector<NDArray*> toDelete;

   NDArray *pA(const_cast<NDArray*>(effA)), *pB(const_cast<NDArray*>(effB)), *pC(const_cast<NDArray*>(C));

   bool aMcont = M == 1 || effA->strideAt(0) == 1;
   bool aKcont = K == 1 || effA->strideAt(1) == 1;
   bool bKcont = K == 1 || effB->strideAt(0) == 1;
   bool bNcont = N == 1 || effB->strideAt(1) == 1;
   bool cMcont = M == 1 || C->strideAt(0) == 1;
   bool cNcont = N == 1 || C->strideAt(1) == 1;

   if (!aMcont && !aKcont) {
     pA = effA->dup('f');
     toDelete.push_back(pA);
     aMcont = true;
   }
   if (!bKcont && !bNcont) {
     pB = effB->dup('f');
     toDelete.push_back(pB);
     bKcont = true;
   }
   if (!cMcont) {
     pC = C->dup('f');
     toDelete.push_back(pC);
     cMcont = true;
   }

   const bool transA = !aMcont;
   const bool transB = !bKcont;

   const int lda = (aMcont && aKcont) ? M : transA ? pA->strideAt(0) : pA->strideAt(1);
   const int ldb = (bKcont && bNcont) ? K : transB ? pB->strideAt(0) : pB->strideAt(1);
   const int ldc = (cMcont && cNcont) ? M : pC->strideAt(1);

   const cublasOperation_t transAblas = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
   const cublasOperation_t transBblas = transB ? CUBLAS_OP_T : CUBLAS_OP_N;

   if (!tl_cublasGapStreamReady) NDArray::prepareSpecialUse({pC}, {pA, pB, beta != 0.0 ? pC : nullptr});

   // cuBLAS Lt fast path for decoder logits projection [1,K] x [K,N] -> [1,N]
   // Use ORIGINAL types (before upcast) for Lt descriptors — cublasLt natively
   // supports mixed A/B types via separate layout descriptors. After upcast,
   // pA/pB data is FLOAT32 but Lt is told the type is CUDA_R_16F for the HALF
   // operand — WRONG: Lt would read FLOAT32 data as HALF. So use effAType/effBType
   // (post-upcast) which correctly reflect the actual data in pA/pB.
   cudaDataType ltAType = CUDA_R_32F, ltBType = CUDA_R_32F, ltCType = CUDA_R_32F;
   sdToCudaDataType(effAType, ltAType);
   sdToCudaDataType(effBType, ltBType);
   sdToCudaDataType(cType,    ltCType);

   if (tryLtMatmul(effA, effB, C, alpha, beta, pA, pB, pC, M, N, K, transA, transB, ltAType, ltBType, ltCType,
                   tl_ltEpilogue.type, tl_ltEpilogue.biasPtr, tl_ltEpilogue.biasSize)) {
     if (!tl_cublasGapStreamReady) NDArray::registerSpecialUse({pC}, {pA, pB});
     if (C != pC) C->assign(pC);
     for (int i = toDelete.size() - 1; i >= 0; --i) deleteTemporary(toDelete[i]);
     deleteTemporary(castA);
     deleteTemporary(castB);
     return C;
   }

   // Decode-phase projections are dominated by row-vector GEMMs [1,K] x [K,N].
   // Map the row-major problem to cuBLAS's column-major view as [N,K] x [K,1] = [N,1]
   // so the library can use a better algorithm than the generic m=1, n=N path.
   // Use stride-based contiguity check instead of ews() which is deprecated/unreliable.
   // ews() returns 0 for arrays created by DSP pre-allocation even when strides are contiguous.
   const bool aContiguous = shape::strideDescendingCAscendingF(pA->shapeInfo());
   const bool cContiguous = shape::strideDescendingCAscendingF(pC->shapeInfo());
   const bool rowVectorFastPath =
       M == 1 && !transA && transB &&
       aContiguous && pB->strideAt(1) == 1 && cContiguous;

   // When tl_cublasLtDisabled is true (deterministic cuBLAS for DSP modes),
   // use CUBLAS_GEMM_DEFAULT — under CUBLAS_PEDANTIC_MATH (set by
   // platformBeginExecution), DEFAULT selects deterministic non-tensor-core
   // algorithms that produce identical results inside/outside CUDA graph
   // capture. ALGO0 silently produces zeros for FP16 input with no
   // workspace on some GPUs. DEFAULT + PEDANTIC_MATH is the supported
   // combination for deterministic results across all precisions.
   // When tl_cublasLtDisabled is false, use CUBLAS_GEMM_DEFAULT_TENSOR_OP
   // for performance (standard execution outside DSP).
   const cublasGemmAlgo_t gemmAlgo = tl_cublasLtDisabled
       ? CUBLAS_GEMM_DEFAULT : CUBLAS_GEMM_DEFAULT_TENSOR_OP;

   if (rowVectorFastPath && (typeDouble || typeFloat || typeHalf || typeHalfFloat)) {
     const int ldaFast = static_cast<int>(pB->strideAt(0));
     const int ldbFast = K;
     const int ldcFast = N;

     if (typeDouble) {
       getCublasScalars()->alphaD = alpha;
       getCublasScalars()->betaD  = beta;
       status = cublasGemmEx(*handle, CUBLAS_OP_N, CUBLAS_OP_N, N, 1, K, &getCublasScalars()->alphaD,
                            pB->specialBuffer(), CUDA_R_64F, ldaFast,
                            pA->specialBuffer(), CUDA_R_64F, ldbFast,
                            &getCublasScalars()->betaD, pC->specialBuffer(), CUDA_R_64F, ldcFast,
                            CUBLAS_COMPUTE_64F, gemmAlgo);
     } else if (typeFloat) {
       getCublasScalars()->alphaF = static_cast<float>(alpha);
       getCublasScalars()->betaF  = static_cast<float>(beta);
       status = cublasGemmEx(*handle, CUBLAS_OP_N, CUBLAS_OP_N, N, 1, K, &getCublasScalars()->alphaF,
                            pB->specialBuffer(), CUDA_R_32F, ldaFast,
                            pA->specialBuffer(), CUDA_R_32F, ldbFast,
                            &getCublasScalars()->betaF, pC->specialBuffer(), CUDA_R_32F, ldcFast,
                            CUBLAS_COMPUTE_32F, gemmAlgo);
     } else if (typeHalf) {
       // Use GemmEx with FP32 accumulation for row-vector path (decode phase).
       // cublasHgemm accumulates in FP16 — catastrophic for K=1024 transformer matmuls.
       getCublasScalars()->alphaF = static_cast<float>(alpha);
       getCublasScalars()->betaF  = static_cast<float>(beta);
       status = cublasGemmEx(*handle, CUBLAS_OP_N, CUBLAS_OP_N, N, 1, K, &getCublasScalars()->alphaF,
                             pB->specialBuffer(), CUDA_R_16F, ldaFast,
                             pA->specialBuffer(), CUDA_R_16F, ldbFast,
                             &getCublasScalars()->betaF,
                             pC->specialBuffer(), CUDA_R_16F, ldcFast,
                             CUBLAS_COMPUTE_32F, gemmAlgo);
     } else if (typeHalfFloat) {
       // HALF inputs with FP32 output, row-vector fast path
       getCublasScalars()->alphaF = static_cast<float>(alpha);
       getCublasScalars()->betaF  = static_cast<float>(beta);
       status = cublasGemmEx(*handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             N, 1, K, &getCublasScalars()->alphaF,
                             pB->specialBuffer(), CUDA_R_16F, ldaFast,
                             pA->specialBuffer(), CUDA_R_16F, ldbFast,
                             &getCublasScalars()->betaF,
                             pC->specialBuffer(), CUDA_R_32F, ldcFast,
                             CUBLAS_COMPUTE_32F, gemmAlgo);
     }
   } else if (typeDouble) {
     getCublasScalars()->alphaD = alpha;
     getCublasScalars()->betaD  = beta;
     status = cublasGemmEx(*handle, transAblas, transBblas, M, N, K, &getCublasScalars()->alphaD,
                          pA->specialBuffer(), CUDA_R_64F, lda,
                          pB->specialBuffer(), CUDA_R_64F, ldb,
                          &getCublasScalars()->betaD, pC->specialBuffer(), CUDA_R_64F, ldc,
                          CUBLAS_COMPUTE_64F, gemmAlgo);
   } else if (typeFloat) {
     getCublasScalars()->alphaF = static_cast<float>(alpha);
     getCublasScalars()->betaF  = static_cast<float>(beta);
     status = cublasGemmEx(*handle, transAblas, transBblas, M, N, K, &getCublasScalars()->alphaF,
                          pA->specialBuffer(), CUDA_R_32F, lda,
                          pB->specialBuffer(), CUDA_R_32F, ldb,
                          &getCublasScalars()->betaF, pC->specialBuffer(), CUDA_R_32F, ldc,
                          CUBLAS_COMPUTE_32F, gemmAlgo);
   } else if (typeHalf) {
     // Use GemmEx with FP32 accumulation instead of cublasHgemm (FP16 accumulation).
     // cublasHgemm accumulates dot products in FP16, causing catastrophic precision loss
     // for large K (e.g., K=1024 in transformer matmuls). FP32 accumulation matches CPU.
     getCublasScalars()->alphaF = static_cast<float>(alpha);
     getCublasScalars()->betaF  = static_cast<float>(beta);
     status = cublasGemmEx(*handle, transAblas, transBblas, M, N, K, &getCublasScalars()->alphaF,
                           pA->specialBuffer(), CUDA_R_16F, lda,
                           pB->specialBuffer(), CUDA_R_16F, ldb,
                           &getCublasScalars()->betaF,
                           pC->specialBuffer(), CUDA_R_16F, ldc,
                           CUBLAS_COMPUTE_32F, gemmAlgo);
   } else if (typeBfloat) {
     getCublasScalars()->alphaF = static_cast<float>(alpha);
     getCublasScalars()->betaF = static_cast<float>(beta);
     status = cublasGemmEx(*handle, transAblas, transBblas, M, N, K, &getCublasScalars()->alphaF,
                          pA->specialBuffer(), CUDA_R_16BF, lda,
                          pB->specialBuffer(), CUDA_R_16BF, ldb,
                          &getCublasScalars()->betaF, pC->specialBuffer(), CUDA_R_16BF, ldc,
                          CUBLAS_COMPUTE_32F, gemmAlgo);
   } else if (typeIntFloat) {
     getCublasScalars()->alphaF = static_cast<float>(alpha);
     getCublasScalars()->betaF  = static_cast<float>(beta);
     status = cublasGemmEx(*handle, transAblas, transBblas, M, N, K, &getCublasScalars()->alphaF,
                           pA->specialBuffer(), CUDA_R_8I, lda,
                           pB->specialBuffer(), CUDA_R_8I, ldb,
                           &getCublasScalars()->betaF, pC->specialBuffer(), CUDA_R_32F, ldc,
                           CUBLAS_COMPUTE_32F, gemmAlgo);
   } else if (typeHalfFloat) {
     getCublasScalars()->alphaF = static_cast<float>(alpha);
     getCublasScalars()->betaF  = static_cast<float>(beta);
     status = cublasGemmEx(*handle, transAblas, transBblas, M, N, K, &getCublasScalars()->alphaF,
                           pA->specialBuffer(), CUDA_R_16F, lda,
                           pB->specialBuffer(), CUDA_R_16F, ldb,
                           &getCublasScalars()->betaF, pC->specialBuffer(), CUDA_R_32F, ldc,
                           CUBLAS_COMPUTE_32F, gemmAlgo);
   }

   if (status != CUBLAS_STATUS_SUCCESS) {
     std::string msg = "MmulHelper::mmulMxM cuda failed [cublasGemmEx-general]; Error code: [" + std::to_string(status) + "]";
     THROW_EXCEPTION(msg.c_str());
   }

   if (!tl_cublasGapStreamReady) NDArray::registerSpecialUse({pC}, {pA, pB});

   if (C != pC) C->assign(pC);

   for (int i = toDelete.size() - 1; i >= 0; --i) deleteTemporary(toDelete[i]);
 }

 // CUDA kernels and cuBLAS consume cast operands asynchronously.
 deleteTemporary(castA);
 deleteTemporary(castB);

 return C;
}////////////////////////////////////////////////////////////////////////////
// MXN x N = M
NDArray* MmulHelper::mmulMxV(NDArray* A, NDArray* X, NDArray* Y, const double alpha, const double beta,
                            const char outOrder) {
 LongType xLenDim, yLenDim(0);

 if (A->rankOf() != 2) THROW_EXCEPTION("MmulHelper::mmulMxV cuda: rank of A array is not equal 2 !");
 if (!shape::isCommonVector(X->shapeInfo(), xLenDim))
   THROW_EXCEPTION("MmulHelper::mmulMxV cuda: X array must be vector !");

 const auto M = A->sizeAt(0);
 const auto N = A->sizeAt(1);

 if (Y != nullptr && !shape::isCommonVector(Y->shapeInfo(), yLenDim))
   THROW_EXCEPTION("MmulHelper::mmulMxV cuda: Y array must be vector !");
 if (X->lengthOf() != N) THROW_EXCEPTION("MmulHelper::mmulMxV cuda: X vector has wrong length !");
 if (Y != nullptr && Y->lengthOf() != M)
   THROW_EXCEPTION("MmulHelper::mmulMxV cuda: Y array has wrong length !");

 std::vector<LongType> yShape = {M};
 if (Y == nullptr)
   Y = new NDArray(outOrder, yShape, ops::helpers::matmulOutputType(A->dataType(), X->dataType()),
                   A->getContext());

 if (Y->isEmpty()) return Y;

 // A matrix whose storage type is not the vector's: the mixed GEMV reads it in place instead of
 // widening it.
 if (ops::helpers::mixedGemvApplies(A->dataType(), X->dataType(), Y->dataType())) {
   mixedGemv(A->getContext(), A, X, Y,
             {M, N, A->strideAt(0), A->strideAt(1), X->strideAt(xLenDim), Y->strideAt(yLenDim)}, alpha, beta);
   return Y;
 }

 const int incy = Y->strideAt(yLenDim);

 const int major = sd::env_deviceCapabilityMajor(AffinityManager::currentDeviceId());

 // Mixed float storage took the mixed GEMV above, so A and X share a type here or one of them
 // is an integer type.
 const auto aType = A->dataType();
 const auto xType = X->dataType();
 const auto yType = Y->dataType();
 const int incx = X->strideAt(xLenDim);

 const bool AX(aType == xType), AY(aType == yType), AXY(AX && AY);

 const bool typeDouble = AXY && aType == DOUBLE;
 const bool typeFloat = AXY && aType == FLOAT32;
 // cublasGemmEx takes X and Y as [N,1] and [M,1] matrices, so both must be contiguous.
 const bool typeHalfFloat = AX && aType == HALF && yType == FLOAT32 && major >= 6 && incx == 1 && incy == 1;

 // usualGemv reads and writes A, X and Y through one element type, so a mix that no cuBLAS
 // path covers is computed in one type and assigned into Y, as in mmulMxM. This runs
 // before the device lock, which the same-type call takes itself.
 if (!AXY && !typeHalfFloat) {
   const DataType computeType = ops::helpers::mixedGemmComputeType(aType, xType, yType);
   std::vector<NDArray*> owned;
   NDArray* computeA = castForMixedGemm(castSideA(), A, computeType, owned);
   NDArray* computeX = castForMixedGemm(castSideB(), X, computeType, owned);
   NDArray* computeY = castForMixedGemm(castSideB(), Y, computeType, owned);
   mmulMxV(computeA, computeX, computeY, alpha, beta, outOrder);
   if (computeY != Y) Y->assign(computeY);
   for (auto* array : owned) deleteTemporary(array);
   return Y;
 }

 std::lock_guard<std::mutex> lock(*LaunchContext::deviceMutex());

 auto handle = reinterpret_cast<cublasHandle_t*>(A->getContext()->getCublasHandle());
 auto stream = A->getContext()->getCudaStream();

  // N107: Skip cublasSetStream + workspace when DSP gap stream already configured (same as mmulMxM).
  cublasStatus_t status = CUBLAS_STATUS_SUCCESS;
  if (!tl_cublasGapStreamReady && !DebugHelper::inGraphCapture(nullptr)) {
    status = cublasSetStream_v2(*handle, *stream);
    if (status != CUBLAS_STATUS_SUCCESS) {
      std::string msg = "MmulHelper::mmulMxV cuda failed !; Error code: [" + std::to_string(status) + "]";
      THROW_EXCEPTION(msg.c_str());
    }
    reapplyCublasWorkspace(*handle);
  }

 if (!typeDouble && !typeFloat && !typeHalfFloat) {
   dim3 dims = getGemVDims(M);
   // beta reads Y, so Y's current values must reach the device first.
   if (!tl_cublasGapStreamReady) NDArray::prepareSpecialUse({Y}, {A, X, beta != 0.0 ? Y : nullptr});

   const int blocksPerGrid = dims.x;
   const int threadsPerBlock = dims.y;
   BUILD_SINGLE_SELECTOR_THRICE(
       xType, usualGemv,
       (blocksPerGrid,threadsPerBlock,stream, A->specialBuffer(), A->specialShapeInfo(), X->specialBuffer(),
        X->specialShapeInfo(), Y->specialBuffer(), Y->specialShapeInfo(), incx, incy, 0, alpha, beta),
       SD_NUMERIC_TYPES)
   if (!tl_cublasGapStreamReady) NDArray::registerSpecialUse({Y}, {A, X});

 } else {
   NDArray* pA(A);

   bool aMcont = M == 1 || A->strideAt(0) == 1;
   bool aNcont = N == 1 || A->strideAt(1) == 1;

   if (!aMcont && !aNcont) {
     pA = A->dup('f');
     aMcont = true;
   }

   const bool transA = !aMcont;

   const int lda = (aMcont && aNcont) ? M : transA ? pA->strideAt(0) : pA->strideAt(1);

   const cublasOperation_t transAblas = transA ? CUBLAS_OP_T : CUBLAS_OP_N;

   if (!tl_cublasGapStreamReady) NDArray::prepareSpecialUse({Y}, {pA, X, beta != 0.0 ? Y : nullptr});

   if (typeDouble) {
     getCublasScalars()->alphaD = alpha;
     getCublasScalars()->betaD  = beta;
     status = cublasDgemv(*handle, transAblas, transA ? N : M, transA ? M : N, &getCublasScalars()->alphaD, (double*)pA->specialBuffer(),
                          lda, (double*)X->specialBuffer(), incx, &getCublasScalars()->betaD, (double*)Y->specialBuffer(), incy);
   } else if (typeFloat) {
     getCublasScalars()->alphaF = static_cast<float>(alpha);
     getCublasScalars()->betaF  = static_cast<float>(beta);
     status = cublasSgemv(*handle, transAblas, transA ? N : M, transA ? M : N, &getCublasScalars()->alphaF, (float*)pA->specialBuffer(),
                          lda, (float*)X->specialBuffer(), incx, &getCublasScalars()->betaF, (float*)Y->specialBuffer(), incy);
   } else if (typeHalfFloat) {
     // FP16 GEMV via cublasGemmEx: treat vector X as [N,1] matrix → GEMM [M,N] × [N,1] = [M,1]
     // HALF inputs with FP32 output and FP32 accumulation for precision.
     getCublasScalars()->alphaF = static_cast<float>(alpha);
     getCublasScalars()->betaF  = static_cast<float>(beta);
     // op(A) is the logical [M,N] matrix whichever way A is stored; only gemv takes the
     // stored matrix's dimensions.
     status = cublasGemmEx(*handle, transAblas, CUBLAS_OP_N,
                           M,               // m (rows of op(A))
                           1,               // n = 1 (vector)
                           N,               // k
                           &getCublasScalars()->alphaF,
                           pA->specialBuffer(), CUDA_R_16F, lda,
                           X->specialBuffer(), CUDA_R_16F, N,  // ldb = N (contiguous vector)
                           &getCublasScalars()->betaF,
                           Y->specialBuffer(), CUDA_R_32F, M,    // ldc = M
                           CUBLAS_COMPUTE_32F,
                           tl_cublasLtDisabled ? CUBLAS_GEMM_DEFAULT : CUBLAS_GEMM_DEFAULT_TENSOR_OP);
   }

   if (status != CUBLAS_STATUS_SUCCESS) {
     std::string msg = "MmulHelper::mmulMxV cuda failed !; Error code: [" + std::to_string(status) + "]";
     THROW_EXCEPTION(msg.c_str());
   }

   if (!tl_cublasGapStreamReady) NDArray::registerSpecialUse({Y}, {pA, X});

   if (pA != A) deleteTemporary(pA);
 }

 return Y;
}////////////////////////////////////////////////////////////////////////////
// (X * Y) = Z[0]
NDArray* MmulHelper::dot(NDArray* X, NDArray* Y, NDArray* Z, const double alpha, const double beta) {
 LongType xLenDim(0), yLenDim(0);

 if (!shape::isCommonVector(X->shapeInfo(), xLenDim))
   THROW_EXCEPTION("MmulHelper::dot cuda: X array must be vector !");
 if (!shape::isCommonVector(Y->shapeInfo(), yLenDim))
   THROW_EXCEPTION("MmulHelper::dot cuda: Y array must be vector !");
 if (Z != nullptr && Z->lengthOf() > 1) {
   THROW_EXCEPTION("MmulHelper::dot: Z array must be scalar !");
 }

 const auto length = X->lengthOf();

 if (Y->lengthOf() != length)
   THROW_EXCEPTION("MmulHelper::dot cuda: lengths of input vectors are different !");

 if (Z == nullptr)
   Z = new NDArray(ops::helpers::matmulOutputType(X->dataType(), Y->dataType()), X->getContext());

 // usualDot reads and writes X, Y and Z through X's element type, so mixed storage is
 // computed in one type and assigned into Z, as in mmulMxM.
 if (Y->dataType() != X->dataType() || Z->dataType() != X->dataType()) {
   const DataType computeType = ops::helpers::mixedGemmComputeType(X->dataType(), Y->dataType(), Z->dataType());
   std::vector<NDArray*> owned;
   NDArray* computeX = castForMixedGemm(castSideA(), X, computeType, owned);
   NDArray* computeY = castForMixedGemm(castSideB(), Y, computeType, owned);
   NDArray* computeZ = castForMixedGemm(castSideB(), Z, computeType, owned);
   dot(computeX, computeY, computeZ, alpha, beta);
   if (computeZ != Z) Z->assign(computeZ);
   for (auto* array : owned) deleteTemporary(array);
   return Z;
 }

 const LongType incx = X->strideAt(xLenDim);
 const LongType incy = Y->strideAt(yLenDim);

 const auto xType = X->dataType();
 const auto yType = Y->dataType();
 const auto zType = Z->dataType();

 if (!X->isActualOnDeviceSide()) X->syncToDevice();
 if (!Y->isActualOnDeviceSide()) Y->syncToDevice();
 if (!Z->isActualOnDeviceSide()) Z->syncToDevice();

 cudaStream_t* stream = X->getContext()->getCudaStream();

 dim3 dims = getMMulDims(length,DataTypeUtils::sizeOf(zType));

 NDArray::prepareSpecialUse({Z}, {X, Y});


 BUILD_SINGLE_SELECTOR_THRICE(xType, usualDot,
                              (dims, stream, length, alpha, X->specialBuffer(), incx,
                               Y->specialBuffer(), incy, beta, Z->specialBuffer()),
                              SD_NUMERIC_TYPES);

 NDArray::registerSpecialUse({Z}, {X, Y});
 // Don't sync - let CUDA operations run asynchronously

 return Z;
}
///////////////////////////////////////////////////////////////////
NDArray* MmulHelper::mmulNxN(NDArray* A, NDArray* B, NDArray* C, double alpha, double beta,
                            const char outOrder) {
 const LongType aRank = A->rankOf();
 const LongType bRank = B->rankOf();

 // input ranks validation
 if (aRank > bRank && bRank != 2) {
   THROW_EXCEPTION("MmulHelper::mmulNxN: rank of B array should be equal 2 !");
 }
 else if (bRank > aRank && aRank != 2) {
   THROW_EXCEPTION("MmulHelper::mmulNxN: rank of A array should be equal 2 !");
 }
 else if (aRank == bRank) {
   for (int i = 0; i < aRank - 2; ++i)
     if (A->sizeAt(i) != B->sizeAt(i))
       THROW_EXCEPTION(
           "MmulHelper::mmulNxN: shapes of A and B arrays are not suitable for matrix multiplication !");
 }

 if (A->sizeAt(-1) != B->sizeAt(-2)) {
   THROW_EXCEPTION("MmulHelper::mmulNxN: shapes of A and B arrays are not suitable for matrix multiplication !");
 }
 // validation of C array
 auto* cExpectedShapePtr = aRank > bRank ? A->getShapeAsVector() : B->getShapeAsVector();
 std::vector<LongType> cExpectedShape = *cExpectedShapePtr;
 delete cExpectedShapePtr;
 cExpectedShape[cExpectedShape.size() - 2] = A->sizeAt(-2);
 cExpectedShape[cExpectedShape.size() - 1] = B->sizeAt(-1);

 if (C != nullptr) {
   if (!C->isSameShape(cExpectedShape))
     THROW_EXCEPTION("MmulHelper::mmulNxN: shape of C array is not suitable for AxB matrix multiplication !");
 } else
   C = new NDArray(outOrder, cExpectedShape, ops::helpers::matmulOutputType(A->dataType(), B->dataType()),
                   A->getContext());

 if (C->isEmpty()) return C;

 // Try cuBLAS strided batch GEMM first for 3D tensors - much faster than custom kernel
 if (aRank == 3 && bRank == 3) {
   if (tryBlasStridedBatched(A, B, C, alpha, beta)) {
     return C;
   }
 }

 // Mixed-precision path: when A and B have different types, the batchedGemm
 // custom kernel can't handle it (BUILD_SINGLE_SELECTOR_THRICE assumes all
 // types match). Reshape higher-rank A to 2D and delegate to mmulMxM, which
 // normalizes mixed FLOAT32/HALF and FLOAT32/DOUBLE inputs before cuBLAS.
 // Folding A's batch into rows needs one B shared by every batch, so B must be 2D.
 if (bRank == 2 && A->dataType() != B->dataType()) {
   const auto aType = A->dataType();
   const auto bType = B->dataType();
   const bool isMixedHalfFloat = ((aType == FLOAT32 && bType == HALF) || (aType == HALF && bType == FLOAT32));
   const bool isMixedDoubleFloat = ((aType == DOUBLE && bType == FLOAT32) || (aType == FLOAT32 && bType == DOUBLE));
   const bool isMixedDoubleHalf = ((aType == DOUBLE && bType == HALF) || (aType == HALF && bType == DOUBLE));

   if (isMixedHalfFloat || isMixedDoubleFloat || isMixedDoubleHalf) {
     // For [B0, B1, ..., M, K] × [K, N], reshape A to [B0*B1*...*M, K] (2D),
     // call mmulMxM, then result is [B0*B1*...*M, N] which we reshape to C's shape.
     // This works because each row of A is independently multiplied by B.
     // A and C must fold their rows in the same order, so both are reshaped in logical 'c'
     // order; reshape returns a view when the strides allow one and a copy otherwise.
     const LongType K = A->sizeAt(-1);
     const LongType N = B->sizeAt(-1);

     // Compute total rows = product of all dims except last
     LongType totalRows = 1;
     for (int i = 0; i < aRank - 1; ++i) {
       totalRows *= A->sizeAt(i);
     }

     // Reshape A to 2D [totalRows, K]
     std::vector<LongType> a2dShape = {totalRows, K};
     NDArray* a2d = A->reshape('c', a2dShape, false);
     NDArray* effA2d = a2d;
     NDArray* b2d = B;
     NDArray* effB2d = b2d;

     // During graph capture, decoder projections frequently reuse the same
     // activation tensor across multiple matmuls (for example q/k/v and gate/up).
     // Reuse the first casted HALF buffer for repeated sources, but still
     // advance the cache index so warmup and capture consume buffers in the
     // same order.
     if (tl_graphExecutionActive) {
       if (aType == FLOAT32 && bType == HALF) {
         CastCacheSide& sideA = castSideA();
         auto reuseIt = sideA.captureCastReuse.find(A);
         if (reuseIt != sideA.captureCastReuse.end()) {
           if (sideA.index < sideA.cache.size()) sideA.index++;
           effA2d = reuseIt->second;
         } else if (sideA.index < sideA.cache.size()
                    && sideA.cache[sideA.index]->dataType() == HALF
                    && sideA.cache[sideA.index]->isSameShape(a2d)) {
           NDArray* cached = sideA.cache[sideA.index++];
           cached->assign(a2d);
           sideA.captureCastReuse.emplace(A, cached);
           effA2d = cached;
         }
       } else if (aType == HALF && bType == FLOAT32) {
         CastCacheSide& sideB = castSideB();
         auto reuseIt = sideB.captureCastReuse.find(B);
         if (reuseIt != sideB.captureCastReuse.end()) {
           if (sideB.index < sideB.cache.size()) sideB.index++;
           effB2d = reuseIt->second;
         } else if (sideB.index < sideB.cache.size()
                    && sideB.cache[sideB.index]->dataType() == HALF
                    && sideB.cache[sideB.index]->isSameShape(b2d)) {
           NDArray* cached = sideB.cache[sideB.index++];
           cached->assign(b2d);
           sideB.captureCastReuse.emplace(B, cached);
           effB2d = cached;
         }
       }
     }

     // Reshape C to 2D [totalRows, N]
     std::vector<LongType> c2dShape = {totalRows, N};
     NDArray* c2d = C->reshape('c', c2dShape, false);

     mmulMxM(effA2d, effB2d, c2d, alpha, beta, c2d->ordering());

     const bool cCopied = c2d->getDataBuffer() != C->getDataBuffer();
     if (cCopied) {
       // C has no 'c' view of its folded rows, so the product went to a copy. Write it back
       // through a view of the copy in C's own shape, which keeps the row order.
       std::vector<LongType>* cShape = C->getShapeAsVector();
       NDArray* unfolded = c2d->reshape('c', *cShape, false);
       delete cShape;
       C->assign(unfolded);
       delete unfolded;
     }
     // Copies are retired behind the stream that still reads them; views only drop their wrapper.
     if (a2d->getDataBuffer() != A->getDataBuffer())
       deleteTemporary(a2d);
     else
       delete a2d;
     if (cCopied)
       deleteTemporary(c2d);
     else
       delete c2d;

     return C;
   }
 }

 // Every other mixed storage (BFLOAT16, a batched B, or a C of another type) would be
 // read and written by batchedGemm with A's element size. Compute it in one type, as
 // mmulMxM does, and assign the result into C.
 if (B->dataType() != A->dataType() || C->dataType() != A->dataType()) {
   const DataType computeType = ops::helpers::mixedGemmComputeType(A->dataType(), B->dataType(), C->dataType());
   std::vector<NDArray*> owned;
   NDArray* computeA = castForMixedGemm(castSideA(), A, computeType, owned);
   NDArray* computeB = castForMixedGemm(castSideB(), B, computeType, owned);
   NDArray* computeC = castForMixedGemm(castSideB(), C, computeType, owned);
   mmulNxN(computeA, computeB, computeC, alpha, beta, outOrder);
   if (computeC != C) C->assign(computeC);
   for (auto* array : owned) deleteTemporary(array);
   return C;
 }

 const LongType cRank = C->rankOf();

 const LongType aMaxis(aRank - 2), aKaxis(aRank - 1), bKaxis(bRank - 2), bNaxis(bRank - 1), cMaxis(cRank - 2),
     cNaxis(cRank - 1);

 const int threadsPerBlock = SD_MAX_NUM_THREADS / 8;
 const int blocksPerGrid = (C->lengthOf() + threadsPerBlock - 1) / threadsPerBlock;
 const int sharedMem = threadsPerBlock * sizeof(LongType) * (aRank + bRank + cRank) + 128;

 PointersManager manager(A->getContext(), "MmulHelper::mmulNxN");

 const LongType *aBatchDims(nullptr), *bBatchDims(nullptr), *cBatchDims(nullptr);

 std::vector<LongType> aDimsVec = {aMaxis,aKaxis};
 std::vector<LongType> *aDims = ShapeUtils::evalDimsToExclude(aRank, 2,aDimsVec.data());

 std::vector<LongType> bDimsVec = {bKaxis, bNaxis};
 std::vector<LongType> *bDims =  ShapeUtils::evalDimsToExclude(bRank,2, bDimsVec.data());


 std::vector<LongType> cDimsVec = {cMaxis, cNaxis};
 std::vector<LongType> *cDims = ShapeUtils::evalDimsToExclude(cRank, cDimsVec.size(),cDimsVec.data());
 if (aRank > 2)
   aBatchDims = reinterpret_cast<LongType*>(manager.replicatePointer(
       aDims->data(), (aRank - 2) * sizeof(LongType)));
 if (bRank > 2)
   bBatchDims = reinterpret_cast<LongType*>(manager.replicatePointer(
       bDims->data(), (bRank - 2) * sizeof(LongType)));
 if (cRank > 2)
   cBatchDims = reinterpret_cast<LongType*>(manager.replicatePointer(
       cDims->data(), (cRank - 2) * sizeof(LongType)));

 // beta reads C, so C's current values must reach the device first.
 NDArray::prepareSpecialUse({C}, {A, B, beta != 0.0 ? C : nullptr});
 BUILD_SINGLE_SELECTOR_THRICE(
     A->dataType(), batchedGemm,
     (blocksPerGrid, threadsPerBlock, sharedMem, A->getContext()->getCudaStream(), A->specialBuffer(),
      A->specialShapeInfo(), B->specialBuffer(), B->specialShapeInfo(), C->specialBuffer(), C->specialShapeInfo(),
      aBatchDims, bBatchDims, cBatchDims, aMaxis, aKaxis, bKaxis, bNaxis, cMaxis, cNaxis, alpha, beta),
     SD_NUMERIC_TYPES)
 NDArray::registerSpecialUse({C}, {A, B});
 // Don't sync explicitly - manager destructor handles it if needed

 delete aDims;
 delete bDims;
 delete cDims;

 return C;
}

//////////////////////////////////////////////////////////////////////////
// cuBLAS Strided Batch GEMM - most efficient for contiguous batched data on GPU
// Uses cublasSgemmStridedBatched/cublasDgemmStridedBatched/cublasHgemmStridedBatched
// This avoids kernel launch overhead for each batch element
// Supports 2D [M, K], 3D [batch, M, K], and
// 4D [batch0, batch1, M, K] tensors.
//////////////////////////////////////////////////////////////////////////
bool MmulHelper::tryBlasStridedBatched(NDArray* A, NDArray* B, NDArray* C,
                                        double alpha, double beta,
                                        bool transA, bool transB) {
  // Map row-major transpose requests directly onto the swapped column-major
  // cuBLAS operands. This avoids materializing asynchronous permute+dup temporaries
  // whose storage could otherwise be recycled while GEMM is still reading it.

  const int aRank = A->rankOf();
  const int bRank = B->rankOf();
  const int cRank = C->rankOf();

  // Handle 2D, 3D, and 4D tensors with matching ranks. Rank-2 is the
  // batch-count-1 case and is especially important for transpose requests:
  // keeping the transpose in cuBLAS avoids an asynchronous permute+dup
  // temporary between a producer kernel and GEMM.
  if (aRank != bRank || bRank != cRank ||
      (aRank != 2 && aRank != 3 && aRank != 4)) {
    return false;
  }

  const auto xType = A->dataType();
  const auto yType = B->dataType();
  const auto zType = C->dataType();

  // Types must match for cuBLAS strided batched
  if (xType != yType || yType != zType) {
    return false;
  }

  // Only float, double, and half supported
  if (xType != DataType::FLOAT32 && xType != DataType::DOUBLE && xType != DataType::HALF) {
    return false;
  }

  // Validate buffers
  if (A->specialBuffer() == nullptr || B->specialBuffer() == nullptr || C->specialBuffer() == nullptr) {
    return false;
  }

  // For 4D tensors, flatten the first two batch dimensions. The stored
  // row/column sizes remain distinct from logical M/K/N when a transpose is requested.
  const LongType aRows = A->sizeAt(-2);
  const LongType aCols = A->sizeAt(-1);
  const LongType bRows = B->sizeAt(-2);
  const LongType bCols = B->sizeAt(-1);
  const LongType M = transA ? aCols : aRows;
  const LongType K = transA ? aRows : aCols;
  const LongType kFromB = transB ? bCols : bRows;
  const LongType N = transB ? bRows : bCols;
  LongType batchSize;

  if (aRank == 2) {
    batchSize = 1;
  } else if (aRank == 3) {
    if (A->sizeAt(0) != B->sizeAt(0) || A->sizeAt(0) != C->sizeAt(0)) {
      return false;
    }
    batchSize = A->sizeAt(0);
  } else {  // aRank == 4
    if (A->sizeAt(0) != B->sizeAt(0) || A->sizeAt(1) != B->sizeAt(1) ||
        A->sizeAt(0) != C->sizeAt(0) || A->sizeAt(1) != C->sizeAt(1)) {
      return false;
    }
    batchSize = A->sizeAt(0) * A->sizeAt(1);
  }

  if (M <= 0 || K <= 0 || N <= 0 || batchSize <= 0 || K != kFromB) {
    return false;
  }

  // Check C dimensions match expected output
  if (C->sizeAt(-2) != M || C->sizeAt(-1) != N) {
    return false;
  }

  // cuBLAS is column-major, so we need to check for compatible memory layouts
  // For row-major [..., M, K]: stride[-1]=1, stride[-2]=K
  // We'll use the trick: C = A*B in row-major is equivalent to C^T = B^T * A^T in col-major
  // So we swap A and B and compute B * A instead

  // Check the stored row-major layout, independent of logical transposition.
  const bool aRowMajor = (A->strideAt(-1) == 1) && (A->strideAt(-2) == aCols);
  const bool bRowMajor = (B->strideAt(-1) == 1) && (B->strideAt(-2) == bCols);
  const bool cRowMajor = (C->strideAt(-1) == 1) && (C->strideAt(-2) == N);

  if (!aRowMajor || !bRowMajor || !cRowMajor) {
    return false;
  }

  // Calculate strides for batched operation
  // For 3D: stride between batches is stride[0]
  // For 4D: stride between batches is stride[1] (stride within b0), and we need contiguous b0*b1
  long long strideA, strideB, strideC;

  if (aRank == 2) {
    // Strides are ignored by cuBLAS when batchSize == 1, but valid matrix
    // extents keep the call well-defined and make this path equivalent to GEMM.
    strideA = aRows * aCols;
    strideB = bRows * bCols;
    strideC = M * N;
  } else if (aRank == 3) {
    const LongType expectedStrideA = aRows * aCols;
    const LongType expectedStrideB = bRows * bCols;
    const LongType expectedStrideC = M * N;

    if (A->strideAt(0) < expectedStrideA || B->strideAt(0) < expectedStrideB || C->strideAt(0) < expectedStrideC) {
      return false;
    }

    strideA = A->strideAt(0);
    strideB = B->strideAt(0);
    strideC = C->strideAt(0);
  } else {  // aRank == 4
    // For 4D, we need the stride between individual batch elements
    // Each batch element is M*K for A, K*N for B, M*N for C
    const LongType expectedStrideA = aRows * aCols;
    const LongType expectedStrideB = bRows * bCols;
    const LongType expectedStrideC = M * N;

    // Check that the 4D tensor is laid out as contiguous batches
    // stride[1] should be M*K (stride between b1 elements)
    // stride[0] should be b1*M*K (stride between b0 elements)
    if (A->strideAt(-3) < expectedStrideA || B->strideAt(-3) < expectedStrideB || C->strideAt(-3) < expectedStrideC) {
      return false;
    }

    // Also verify outer batch dimension is contiguous
    LongType b1 = A->sizeAt(1);
    if (A->strideAt(0) < b1 * expectedStrideA || B->strideAt(0) < b1 * expectedStrideB || C->strideAt(0) < b1 * expectedStrideC) {
      return false;
    }

    // Use the inner batch stride (stride[1] for 4D = stride[-3])
    strideA = A->strideAt(-3);
    strideB = B->strideAt(-3);
    strideC = C->strideAt(-3);
  }

  // Get cuBLAS handle
  auto handle = reinterpret_cast<cublasHandle_t*>(A->getContext()->getCublasHandle());
  auto stream = A->getContext()->getCudaStream();
  // Skip cublasSetStream during both an active graph capture and its outer
  // composite scope (see mmulMxM comment above).
  if (!DebugHelper::inGraphCapture(nullptr)) {
    cublasSetStream(*handle, *stream);
  }
  reapplyCublasWorkspace(*handle);

  // beta reads C, so C's current values must reach the device first.
  NDArray::prepareSpecialUse({C}, {A, B, beta != 0.0 ? C : nullptr});

  cudaStream_t intendedStream = stream != nullptr ? *stream : nullptr;
  cudaStream_t handleStream = intendedStream;
  cublasMath_t handleMathMode = CUBLAS_DEFAULT_MATH;
  cublasPointerMode_t handlePointerMode = CUBLAS_POINTER_MODE_HOST;
  cublasAtomicsMode_t handleAtomicsMode = CUBLAS_ATOMICS_NOT_ALLOWED;
  cublasStatus_t status;
  const cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
  const cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
  const int ldB = static_cast<int>(bCols);
  const int ldA = static_cast<int>(aCols);

  // cuBLAS sees each row-major operand as its column-major transpose.
  // Compute C^T = op(B)^T * op(A)^T by swapping operands and mirroring
  // each requested row-major transpose into the corresponding cuBLAS op.

  int activeMmulFpOrdinal = -1;
  // The handle-state reads below are diagnostic probes, not execution
  // requirements. Keep them completely off the production path: live DSP gaps
  // execute 90 matmuls per SmolDocling token, so querying four cuBLAS handle
  // properties for every GEMM materially reduces steady-state throughput.
  if (!tl_graphExecutionActive && DSP_DIAG_ENABLED(MEMORY)) {
    if (cublasGetStream(*handle, &handleStream) != CUBLAS_STATUS_SUCCESS) {
      handleStream = intendedStream;
    }
    cublasGetMathMode(*handle, &handleMathMode);
    cublasGetPointerMode(*handle, &handlePointerMode);
    cublasGetAtomicsMode(*handle, &handleAtomicsMode);
    activeMmulFpOrdinal = graph::recordActiveMmulFingerprintTriplet(
        A->specialBuffer(), static_cast<size_t>(A->lengthOf()) * A->sizeOfT(),
        B->specialBuffer(), static_cast<size_t>(B->lengthOf()) * B->sizeOfT(),
        C->specialBuffer(), static_cast<size_t>(C->lengthOf()) * C->sizeOfT(),
        intendedStream, handleStream,
        static_cast<int>(handleMathMode), static_cast<int>(handlePointerMode),
        static_cast<int>(handleAtomicsMode), CublasHelper::inDeterministicWindow(),
        tl_cublasLtDisabled, tl_cublasWorkspacePtr, tl_cublasWorkspaceSize);
    if (activeMmulFpOrdinal == 1 || activeMmulFpOrdinal == 3 ||
        activeMmulFpOrdinal == 5 || activeMmulFpOrdinal == 7) {
      const void* alphaPtr = xType == DataType::DOUBLE
          ? static_cast<const void*>(&getCublasScalars()->alphaD)
          : static_cast<const void*>(&getCublasScalars()->alphaF);
      const void* betaPtr = xType == DataType::DOUBLE
          ? static_cast<const void*>(&getCublasScalars()->betaD)
          : static_cast<const void*>(&getCublasScalars()->betaF);
      DSP_DIAG(MEMORY,
               "BUF_FP_MMUL_ARGS ordinal=%d cublasHandle=%p rank=%d M=%lld N=%lld K=%lld batch=%lld strideA=%lld strideB=%lld strideC=%lld transA=%d transB=%d opA=%d opB=%d ldA=%d ldB=%d alpha=%.17g beta=%.17g alphaPtr=%p betaPtr=%p",
               activeMmulFpOrdinal, (void*)*handle, aRank,
               static_cast<long long>(M), static_cast<long long>(N), static_cast<long long>(K),
               static_cast<long long>(batchSize), strideA, strideB, strideC,
               static_cast<int>(transA), static_cast<int>(transB),
               static_cast<int>(opA), static_cast<int>(opB), ldA, ldB,
               alpha, beta, alphaPtr, betaPtr);
    }
  }

  if (xType == DataType::DOUBLE) {
    getCublasScalars()->alphaD = alpha;
    getCublasScalars()->betaD  = beta;
    status = cublasDgemmStridedBatched(
        *handle,
        opB, opA,
        N, M, K,  // Swapped M,N for row-major
        &getCublasScalars()->alphaD,
        reinterpret_cast<const double*>(B->specialBuffer()), ldB, strideB,
        reinterpret_cast<const double*>(A->specialBuffer()), ldA, strideA,
        &getCublasScalars()->betaD,
        reinterpret_cast<double*>(C->specialBuffer()), N, strideC,
        batchSize);
  } else if (xType == DataType::FLOAT32) {
    getCublasScalars()->alphaF = static_cast<float>(alpha);
    getCublasScalars()->betaF  = static_cast<float>(beta);
    status = cublasSgemmStridedBatched(
        *handle,
        opB, opA,
        N, M, K,  // Swapped M,N for row-major
        &getCublasScalars()->alphaF,
        reinterpret_cast<const float*>(B->specialBuffer()), ldB, strideB,
        reinterpret_cast<const float*>(A->specialBuffer()), ldA, strideA,
        &getCublasScalars()->betaF,
        reinterpret_cast<float*>(C->specialBuffer()), N, strideC,
        batchSize);
  } else if (xType == DataType::HALF) {
    // Use GemmStridedBatchedEx with FP32 accumulation instead of cublasHgemmStridedBatched.
    // FP16 accumulation is catastrophic for K=64-1024 in multi-head attention matmuls.
    getCublasScalars()->alphaF = static_cast<float>(alpha);
    getCublasScalars()->betaF  = static_cast<float>(beta);
    status = cublasGemmStridedBatchedEx(
        *handle,
        opB, opA,
        N, M, K,
        &getCublasScalars()->alphaF,
        B->specialBuffer(), CUDA_R_16F, ldB, strideB,
        A->specialBuffer(), CUDA_R_16F, ldA, strideA,
        &getCublasScalars()->betaF,
        C->specialBuffer(), CUDA_R_16F, N, strideC,
        batchSize,
        CUBLAS_COMPUTE_32F,
        tl_cublasLtDisabled ? CUBLAS_GEMM_DEFAULT : CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  } else {
    NDArray::registerSpecialUse({C}, {A, B});
    return false;
  }

  if (status != CUBLAS_STATUS_SUCCESS) {
    NDArray::registerSpecialUse({C}, {A, B});
    return false;
  }

  if (!tl_graphExecutionActive) {
    graph::recordActiveMmulOutputFingerprint(
        activeMmulFpOrdinal, C->specialBuffer(),
        static_cast<size_t>(C->lengthOf()) * C->sizeOfT(), handleStream);
  }

  NDArray::registerSpecialUse({C}, {A, B});
  // Don't sync - let CUDA operations run asynchronously

  return true;
}

} // namespace sd

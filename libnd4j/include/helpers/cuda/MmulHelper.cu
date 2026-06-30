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

#include <algorithm>
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

namespace sd {

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
  if (!tl_graphExecutionActive && tl_cublasWorkspacePtr != nullptr && tl_cublasWorkspaceSize > 0) {
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
  std::vector<const void*> sourcePtrs;

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

  void clear(std::vector<NDArray*>& retiredCache) {
    for (auto* p : cache) delete p;
    cache.clear();
    index = 0;
    captureCastReuse.clear();
    lastCaptureCastSource = nullptr;
    lastCaptureCastArray = nullptr;
    sourcePtrs.clear();
  }
};

static thread_local CastCacheSide tl_castA;
static thread_local CastCacheSide tl_castB;
static thread_local std::vector<NDArray*> tl_retiredCastCache;

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

  bool operator==(const LtMatmulCacheKey& other) const {
    return deviceId == other.deviceId && M == other.M && N == other.N && K == other.K &&
           aType == other.aType && bType == other.bType && cType == other.cType &&
           transA == other.transA && transB == other.transB &&
           epilogueType == other.epilogueType &&
           workspaceSizeHint == other.workspaceSizeHint &&
           graphCaptureEnabled == other.graphCaptureEnabled;
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
  tl_castA.resetIndices();
  tl_castB.resetIndices();
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
  tl_castA.resetIndicesTo(hwmA);
  tl_castB.resetIndicesTo(hwmB);
  // NOTE: tl_ltAlgoCache is intentionally NOT cleared here — see resetCastCacheIndices().
}

std::pair<size_t, size_t> MmulHelper::getCastCacheHighWaterMark() {
  return {tl_castA.index, tl_castB.index};
}

void MmulHelper::clearCastCache() {
  tl_castA.clear(tl_retiredCastCache);
  tl_castB.clear(tl_retiredCastCache);
  for (auto* p : tl_retiredCastCache) delete p;
  tl_retiredCastCache.clear();
  tl_ltAlgoCache.clear();
}

static NDArray* castWithPersistentCache(CastCacheSide& side, NDArray* source, DataType targetType) {
  auto& cache = side.cache;
  auto& index = side.index;
  auto& srcPtrs = side.sourcePtrs;

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
        tl_retiredCastCache.push_back(cached);
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
// Narrow fast path targeting large N (vocab projection) with mixed precision.
// Returns true if Lt matmul was executed, false to fall back to standard cuBLAS.
static bool tryLtMatmul(NDArray* A, NDArray* B, NDArray* C, double alpha, double beta,
                       NDArray* pA, NDArray* pB, NDArray* pC,
                       int M, int N, int K,
                       bool transA, bool transB,
                       cudaDataType aType, cudaDataType bType, cudaDataType cType,
                       int epilogueType = 0, const void* biasPtr = nullptr, int64_t biasSize = 0) {
  // Skip cublasLt when tl_cublasLtDisabled is set (CUDA_GRAPHS, AUTO capture).
  // cublasLt may select split-K algorithms whose internal reductions are
  // non-deterministic when replayed via CUDA graph (threadblock scheduling
  // order varies between replay iterations).
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

  // Get cuBLAS Lt handle
  auto ltHandlePtr = CublasHelper::getInstance().ltHandle();
  if (ltHandlePtr == nullptr) return false;
  auto ltHandle = reinterpret_cast<cublasLtHandle_t*>(ltHandlePtr);
  auto stream = A->getContext()->getCudaStream();

  // Build cache key
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
  // Keep algo cache partitioned by effective workspace regime. This prevents
  // cross-run reuse of heuristics selected under different workspace settings.
  key.workspaceSizeHint = tl_cublasWorkspaceSize;
  // Partition by graph-capture execution regime: tests toggle this between
  // methods and reusing Lt heuristics across regimes can select unstable algos.
  key.graphCaptureEnabled = Environment::getInstance().tritonGraphCapture() ? 1 : 0;

  // Look up cached algorithm
  auto cacheIt = tl_ltAlgoCache.find(key);
  cublasLtMatmulAlgo_t algo;
  size_t workspaceSize = 0;

  if (cacheIt != tl_ltAlgoCache.end()) {
    // Use cached algorithm
    algo = cacheIt->second.algo;
    workspaceSize = cacheIt->second.workspaceSize;
  } else if (tl_graphExecutionActive) {
    // During capture without cached algo: fall back to standard cuBLAS
    return false;
  } else {
    // Warmup: get heuristic algorithm
    cublasLtMatmulDesc_t operationDesc = nullptr;
    cublasLtMatmulPreference_t preference = nullptr;
    cublasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr;
    int resultsReturned = 0;
    cublasLtMatmulHeuristicResult_t heuristicResults[1];

    // For row-major C = A * B, use cuBLAS Lt column-major with:
    // C^T = B^T * A^T, so we swap operands
    cublasStatus_t status = cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    if (status == CUBLAS_STATUS_SUCCESS) {
      // We interpret row-major buffers as column-major transposes:
      //   B[K,N] row-major -> [N,K] column-major = B^T
      //   A[1,K] row-major -> [K,1] column-major = A^T
      // Then C^T = B^T * A^T, so both Lt operands stay non-transposed.
      cublasOperation_t opTransA = CUBLAS_OP_N;
      cublasOperation_t opTransB = CUBLAS_OP_N;
      status = cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opTransA, sizeof(opTransA));
      if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opTransB, sizeof(opTransB));
      }

      // Set cublasLt epilogue for fused bias+activation
      if (status == CUBLAS_STATUS_SUCCESS && epilogueType > 0 && biasPtr != nullptr) {
        cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_DEFAULT;
        if (epilogueType == 1) epilogue = CUBLASLT_EPILOGUE_BIAS;
        else if (epilogueType == 2) epilogue = CUBLASLT_EPILOGUE_RELU_BIAS;
        else if (epilogueType == 3) epilogue = CUBLASLT_EPILOGUE_GELU_BIAS;

        if (epilogue != CUBLASLT_EPILOGUE_DEFAULT) {
          status = cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE,
                                                   &epilogue, sizeof(epilogue));
          if (status == CUBLAS_STATUS_SUCCESS) {
            status = cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                                     &biasPtr, sizeof(biasPtr));
          }
        }
      }

      // Layouts for column-major interpretation of row-major data:
      // B is [K,N] row-major -> treat as [N,K] column-major with ld=N
      // A is [1,K] row-major -> treat as [K,1] column-major with ld=K
      // C is [1,N] row-major -> treat as [N,1] column-major with ld=N
      if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasLtMatrixLayoutCreate(&Bdesc, bType, N, K, (int64_t)N);
      }
      if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasLtMatrixLayoutCreate(&Adesc, aType, K, M, (int64_t)K);
        if (status == CUBLAS_STATUS_SUCCESS) {
          status = cublasLtMatrixLayoutCreate(&Cdesc, cType, N, M, (int64_t)N);
          if (status == CUBLAS_STATUS_SUCCESS) {
            status = cublasLtMatmulPreferenceCreate(&preference);
            if (status == CUBLAS_STATUS_SUCCESS) {
              size_t maxWorkspace = tl_cublasWorkspaceSize;
              status = cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                            &maxWorkspace, sizeof(maxWorkspace));

              if (status == CUBLAS_STATUS_SUCCESS) {
                // Note: B is first operand (as A in col-major), A is second operand (as B in col-major)
                status = cublasLtMatmulAlgoGetHeuristic(*ltHandle, operationDesc, Bdesc, Adesc, Cdesc, Cdesc,
                                                        preference, 1, heuristicResults, &resultsReturned);
              }
              cublasLtMatmulPreferenceDestroy(preference);
            }
            cublasLtMatrixLayoutDestroy(Cdesc);
          }
          cublasLtMatrixLayoutDestroy(Adesc);
        }
        cublasLtMatrixLayoutDestroy(Bdesc);
      }
      cublasLtMatmulDescDestroy(operationDesc);
    }

    if (status != CUBLAS_STATUS_SUCCESS || resultsReturned == 0) {
      // Heuristic failed: fall back to standard cuBLAS
      return false;
    }

    // Cache the best algorithm
    algo = heuristicResults[0].algo;
    workspaceSize = heuristicResults[0].workspaceSize;
    tl_ltAlgoCache[key] = {algo, workspaceSize};
  }

  // Execute Lt matmul
  cublasLtMatmulDesc_t operationDesc = nullptr;
  cublasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr;
  bool success = false;

  cublasStatus_t status = cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
  if (status == CUBLAS_STATUS_SUCCESS) {
    cublasOperation_t opTransA = CUBLAS_OP_N;
    cublasOperation_t opTransB = CUBLAS_OP_N;
    status = cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opTransA, sizeof(opTransA));
    if (status == CUBLAS_STATUS_SUCCESS) {
      status = cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opTransB, sizeof(opTransB));
    }

    // Set cublasLt epilogue for fused bias+activation (execution path)
    if (status == CUBLAS_STATUS_SUCCESS && epilogueType > 0 && biasPtr != nullptr) {
      cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_DEFAULT;
      if (epilogueType == 1) epilogue = CUBLASLT_EPILOGUE_BIAS;
      else if (epilogueType == 2) epilogue = CUBLASLT_EPILOGUE_RELU_BIAS;
      else if (epilogueType == 3) epilogue = CUBLASLT_EPILOGUE_GELU_BIAS;

      if (epilogue != CUBLASLT_EPILOGUE_DEFAULT) {
        status = cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE,
                                                 &epilogue, sizeof(epilogue));
        if (status == CUBLAS_STATUS_SUCCESS) {
          status = cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                                   &biasPtr, sizeof(biasPtr));
        }
      }
    }

    if (status == CUBLAS_STATUS_SUCCESS) {
      status = cublasLtMatrixLayoutCreate(&Bdesc, bType, N, K, (int64_t)N);
    }
    if (status == CUBLAS_STATUS_SUCCESS) {
      status = cublasLtMatrixLayoutCreate(&Adesc, aType, K, M, (int64_t)K);
      if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasLtMatrixLayoutCreate(&Cdesc, cType, N, M, (int64_t)N);
        if (status == CUBLAS_STATUS_SUCCESS) {
          // Use persistent thread-local scalars (not stack locals) so that
          // the host addresses baked into captured CUDA graphs remain valid on replay.
          getCublasScalars()->alphaF = static_cast<float>(alpha);
          getCublasScalars()->betaF  = static_cast<float>(beta);

          void* workspace = nullptr;
          size_t actualWorkspaceSize = 0;
          if (workspaceSize > 0 && tl_cublasWorkspacePtr != nullptr && workspaceSize <= tl_cublasWorkspaceSize) {
            workspace = tl_cublasWorkspacePtr;
            actualWorkspaceSize = workspaceSize;
          }

          // B first (as A in col-major), A second (as B in col-major)
          status = cublasLtMatmul(*ltHandle, operationDesc, &getCublasScalars()->alphaF,
                                  pB->specialBuffer(), Bdesc,
                                  pA->specialBuffer(), Adesc,
                                  &getCublasScalars()->betaF,
                                  pC->specialBuffer(), Cdesc,
                                  pC->specialBuffer(), Cdesc,
                                  &algo,
                                  workspace, actualWorkspaceSize,
                                  *stream);

          if (status == CUBLAS_STATUS_SUCCESS) {
            success = true;
          }
          cublasLtMatrixLayoutDestroy(Cdesc);
        }
        cublasLtMatrixLayoutDestroy(Adesc);
      }
      cublasLtMatrixLayoutDestroy(Bdesc);
    }
    cublasLtMatmulDescDestroy(operationDesc);
  }

  return success;
}

//////////////////////////////////////////////////////////////////////////////
// MXK x KxN = MxN              -> actual sequence of axes doesn't matter
template <typename T1, typename T2, typename T3>
static SD_KERNEL void usualCudaGemm(const void* vA, const LongType* aShapeInfo, const void* vB,
                                   const LongType* bShapeInfo, void* vC, const LongType* cShapeInfo,
                                   const int aMaxis, const int aKaxis, const int bKaxis, const int bNaxis,
                                   const int cMaxis, const int cNaxis, const double alpha, const double beta) {
 // Cache shape information in shared memory
 __shared__ LongType K;
 __shared__ LongType cLen;
 __shared__ LongType totalThreads;
 __shared__ bool betaPresent;
 __shared__ T3 alphaZ;
 __shared__ T3 betaZ;
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

   T3 val = A[aOffset] * B[bOffset];  // first iteration

   for (LongType j = 1; j < K; ++j) {  // rest iterations
     aOffset += aStride[aKaxis];
     bOffset += bStride[bKaxis];
     val = val + A[aOffset] * B[bOffset];
   }

   COORDS2INDEX(cRank, cStride, cCoords, cOffset);

   if (betaPresent)
     C[cOffset] = alphaZ * val + betaZ * C[cOffset];
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

 // Cache shape information in shared memory
 __shared__ LongType M;
 __shared__ LongType N;
 __shared__ bool betaPresent;
 __shared__ LongType totalThreads;
 __shared__ LongType aNstride;
 __shared__ LongType aMstride;
 __shared__ T3 alphaZ;
 __shared__ T3 betaZ;
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

   T3 val = A[aOffset] * X[xOffset];  // first iteration

   for (LongType j = 1; j < N; ++j) {  // rest iterations
     aOffset += aNstride;
     xOffset += incx;
     val = val + A[aOffset] * X[xOffset];
   }

   auto yOffset = i * incy;

   if (betaPresent)
     Y[yOffset] = alphaZ * val + betaZ * Y[yOffset];
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
 // Cache values in shared memory
 __shared__ T3* pairwiseMul;
 __shared__ T3 alphaZ;
 __shared__ T3 betaZ;
 __shared__ bool betaPresent;

 if (threadIdx.x == 0) {
   extern __shared__ unsigned char shmem[];
   pairwiseMul = reinterpret_cast<T3*>(shmem);
   alphaZ = alpha;
   betaZ = beta;
   betaPresent = beta != 0;
 }
 __syncthreads();

 T1* X = reinterpret_cast<T1*>(const_cast<void*>(vX));
 T2* Y = reinterpret_cast<T2*>(const_cast<void*>(vY));
 T3* Z = reinterpret_cast<T3*>(vZ);

 const int tid = blockIdx.x * blockDim.x + threadIdx.x;
 if (tid < length) {
   pairwiseMul[tid] = X[tid * incx] * Y[tid * incy];
 }

 __syncthreads();

 if (tid == 0) {
   T3 sum = static_cast<T3>(0);
   for (LongType i = 0; i < length; ++i) {
     sum = sum + pairwiseMul[i];
   }

   if (betaPresent)
     *Z = alphaZ * sum + betaZ * *Z;
   else
     *Z = alphaZ * sum;
 }
}

////////////////////////////////////////////////////////////////////////
template <typename T1, typename T2, typename T3>
SD_HOST static void usualDot(const dim3& launchDims, cudaStream_t* stream,
                            const LongType length, const double alpha, const void* vX, const LongType incx,
                            const void* vY, const LongType incy, const double beta, void* vZ) {
 usualCudaDot<T1, T2, T3><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(
     length, alpha, vX, incx, vY, incy, beta, vZ);
 DebugHelper::checkGlobalErrorCode("concat dot failed(...) failed");
}

//////////////////////////////////////////////////////////////////////////////
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

 // Cache shape information in shared memory
 __shared__ struct {
   bool betaPresent;
   LongType aRank, bRank, cRank, K;
   LongType cLen, totalThreads;
   T3 alphaZ, betaZ;
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

   T3 val = A[aOffset] * B[bOffset];  // first iteration

   for (LongType j = 1; j < shared.K; ++j) {  // rest iterations
     aOffset += shared.aStride[aKaxis];
     bOffset += shared.bStride[bKaxis];
     val = val + A[aOffset] * B[bOffset];
   }

   COORDS2INDEX(shared.cRank, shared.cStride, cCoords, cOffset);

   if (shared.betaPresent)
     C[cOffset] = shared.alphaZ * val + shared.betaZ * C[cOffset];
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
   C = new NDArray(outOrder, cShape, DataTypeUtils::pickPairwiseResultType(A->dataType(), B->dataType()),
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

 // Use cast cache while DSP owns graph/replay state or the cuBLAS workspace.
 // Warmup fills the cache, capture records stable cache buffer addresses, and
 // replay reuses the same buffers after resetCastCacheIndices().
 bool useCastCache = tl_graphExecutionActive || tl_dspReplayActive ||
                     tl_cublasGapStreamReady || tl_cublasWorkspacePtr != nullptr;

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
       effA = castWithPersistentCache(tl_castA, effA, FLOAT32);
     } else {
       castA = effA->cast(FLOAT32);
       effA = castA;
     }
   } else if (A->dataType() == FLOAT32 && B->dataType() == HALF) {
     // Activation is FLOAT32, weight is HALF → upcast weight to FLOAT32
     if (useCastCache) {
       effB = castWithPersistentCache(tl_castB, effB, FLOAT32);
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
         effA = castWithPersistentCache(tl_castA, effA, DOUBLE);
       } else {
         castA = effA->cast(DOUBLE);
         effA = castA;
       }
     }
     if (currentBType != DOUBLE) {
       if (useCastCache) {
         effB = castWithPersistentCache(tl_castB, effB, DOUBLE);
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
 const bool typeIntFloat = AB && effAType == INT8 && cType == FLOAT32 && major >= 6;
 const bool typeHalfFloat = AB && effAType == HALF && cType == FLOAT32 && major >= 6;

 std::lock_guard<std::mutex> lock(*LaunchContext::deviceMutex());

 auto handle = reinterpret_cast<cublasHandle_t*>(A->getContext()->getCublasHandle());
 auto stream = A->getContext()->getCudaStream();

  // N107: Skip cublasSetStream + workspace when DSP gap stream already configured.
  // The gap loop's CublasGapStreamGuard sets the cuBLAS handle stream+workspace once
  // at gap-loop start. Skipping here eliminates 2 cuBLAS host-API calls × 91 matmuls/step.
  // Also skip during CUDA graph replay (existing optimization).
  cublasStatus_t status = CUBLAS_STATUS_SUCCESS;
  if (!tl_cublasGapStreamReady && !tl_graphExecutionActive) {
    status = cublasSetStream_v2(*handle, *stream);
    if (status != CUBLAS_STATUS_SUCCESS) {
      std::string msg = "MmulHelper::mmulMxM cuda failed !; Error code: [" + std::to_string(status) + "]";
      THROW_EXCEPTION(msg.c_str());
    }
    reapplyCublasWorkspace(*handle);
  }

 if (!typeDouble && !typeFloat && !typeHalf && !typeIntFloat && !typeHalfFloat) {
   dim3 dims = getMMulDims(C->lengthOf(),DataTypeUtils::sizeOf(cType));
   if (!tl_cublasGapStreamReady) NDArray::prepareSpecialUse({C}, {effA, effB});
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

   if (!tl_cublasGapStreamReady) NDArray::prepareSpecialUse({pC}, {pA, pB});

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
     for (int i = toDelete.size() - 1; i >= 0; --i) delete toDelete[i];
     delete castA;
     delete castB;
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
     std::string msg = "MmulHelper::mmulMxM cuda failed !; Error code: [" + std::to_string(status) + "]";
     THROW_EXCEPTION(msg.c_str());
   }

   if (!tl_cublasGapStreamReady) NDArray::registerSpecialUse({pC}, {pA, pB});

   if (C != pC) C->assign(pC);

   for (int i = toDelete.size() - 1; i >= 0; --i) delete toDelete[i];
 }

 // Cleanup cast arrays
 delete castA;
 delete castB;

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
   Y = new NDArray(outOrder, yShape, DataTypeUtils::pickPairwiseResultType(A->dataType(), X->dataType()),
                   A->getContext());

 if (Y->isEmpty()) return Y;

 const int incx = X->strideAt(xLenDim);
 const int incy = Y->strideAt(yLenDim);

 const int major = sd::env_deviceCapabilityMajor(AffinityManager::currentDeviceId());

 const auto aType = A->dataType();
 const auto xType = X->dataType();
 const auto yType = Y->dataType();

 // NOTE: FP16 GEMV autocast for BOTH-FP32 inputs REMOVED (don't force FP32→HALF).
 // However, mixed-type handling is REQUIRED: when GraphOptimizer pre-casts weights
 // to HALF but activation is still FLOAT32 (or vice versa), we must cast the
 // mismatched operand so both have the same type before dispatch. Without this,
 // usualGemv interprets HALF memory as FLOAT32 bytes → garbage output.
 NDArray* castA = nullptr;
 NDArray* castX = nullptr;
 NDArray* effA = const_cast<NDArray*>(A);
 NDArray* effX = const_cast<NDArray*>(X);

 // Mixed-type handling for GEMV: cublasSgemv/cublasGemmEx require same type for A and X.
 // When one is HALF and the other FLOAT32, upcast the HALF operand to FLOAT32.
 // NEVER downcast FLOAT32→HALF — that loses precision across transformer layers.
 if (A->dataType() != X->dataType() && (yType == FLOAT32 || yType == HALF) && major >= 6) {
   if (A->dataType() == HALF && X->dataType() == FLOAT32) {
     // Weight is HALF, vector is FLOAT32 → upcast weight to FLOAT32
     castA = A->cast(FLOAT32);
     effA = castA;
   } else if (A->dataType() == FLOAT32 && X->dataType() == HALF) {
     // Weight is FLOAT32, vector is HALF → upcast vector to FLOAT32
     castX = X->cast(FLOAT32);
     effX = castX;
   } else if (A->dataType() == BFLOAT16 && X->dataType() == FLOAT32) {
     castA = A->cast(FLOAT32);
     effA = castA;
   } else if (A->dataType() == FLOAT32 && X->dataType() == BFLOAT16) {
     castX = X->cast(FLOAT32);
     effX = castX;
   }
 }

 const auto effAType = effA->dataType();
 const auto effXType = effX->dataType();

 const bool AX(effAType == effXType), AY(effAType == yType), AXY(AX && AY);

 const bool typeDouble = AXY && effAType == DOUBLE;
 const bool typeFloat = AXY && effAType == FLOAT32;
 const bool typeHalfFloat = AX && effAType == HALF && yType == FLOAT32 && major >= 6;

 std::lock_guard<std::mutex> lock(*LaunchContext::deviceMutex());

 auto handle = reinterpret_cast<cublasHandle_t*>(A->getContext()->getCublasHandle());
 auto stream = A->getContext()->getCudaStream();

  // N107: Skip cublasSetStream + workspace when DSP gap stream already configured (same as mmulMxM).
  cublasStatus_t status = CUBLAS_STATUS_SUCCESS;
  if (!tl_cublasGapStreamReady && !tl_graphExecutionActive) {
    status = cublasSetStream_v2(*handle, *stream);
    if (status != CUBLAS_STATUS_SUCCESS) {
      std::string msg = "MmulHelper::mmulMxV cuda failed !; Error code: [" + std::to_string(status) + "]";
      THROW_EXCEPTION(msg.c_str());
    }
    reapplyCublasWorkspace(*handle);
  }

 if (!typeDouble && !typeFloat && !typeHalfFloat) {
   dim3 dims = getGemVDims(M);
   if (!tl_cublasGapStreamReady) NDArray::prepareSpecialUse({Y}, {A, X});

   const int blocksPerGrid = dims.x;
   const int threadsPerBlock = dims.y;
   BUILD_SINGLE_SELECTOR_THRICE(
       xType, usualGemv,
       (blocksPerGrid,threadsPerBlock,stream, A->specialBuffer(), A->specialShapeInfo(), X->specialBuffer(),
        X->specialShapeInfo(), Y->specialBuffer(), Y->specialShapeInfo(), incx, incy, 0, alpha, beta),
       SD_NUMERIC_TYPES)
   if (!tl_cublasGapStreamReady) NDArray::registerSpecialUse({Y}, {A, X});

 } else {
   NDArray* pA(const_cast<NDArray*>(effA));

   bool aMcont = M == 1 || effA->strideAt(0) == 1;
   bool aNcont = N == 1 || effA->strideAt(1) == 1;

   if (!aMcont && !aNcont) {
     pA = effA->dup('f');
     aMcont = true;
   }

   const bool transA = !aMcont;

   const int lda = (aMcont && aNcont) ? M : transA ? pA->strideAt(0) : pA->strideAt(1);

   const cublasOperation_t transAblas = transA ? CUBLAS_OP_T : CUBLAS_OP_N;

   if (!tl_cublasGapStreamReady) NDArray::prepareSpecialUse({Y}, {pA, effX});

   if (typeDouble) {
     getCublasScalars()->alphaD = alpha;
     getCublasScalars()->betaD  = beta;
     status = cublasDgemv(*handle, transAblas, transA ? N : M, transA ? M : N, &getCublasScalars()->alphaD, (double*)pA->specialBuffer(),
                          lda, (double*)effX->specialBuffer(), incx, &getCublasScalars()->betaD, (double*)Y->specialBuffer(), incy);
   } else if (typeFloat) {
     getCublasScalars()->alphaF = static_cast<float>(alpha);
     getCublasScalars()->betaF  = static_cast<float>(beta);
     status = cublasSgemv(*handle, transAblas, transA ? N : M, transA ? M : N, &getCublasScalars()->alphaF, (float*)pA->specialBuffer(),
                          lda, (float*)effX->specialBuffer(), incx, &getCublasScalars()->betaF, (float*)Y->specialBuffer(), incy);
   } else if (typeHalfFloat) {
     // FP16 GEMV via cublasGemmEx: treat vector X as [N,1] matrix → GEMM [M,N] × [N,1] = [M,1]
     // HALF inputs with FP32 output and FP32 accumulation for precision.
     getCublasScalars()->alphaF = static_cast<float>(alpha);
     getCublasScalars()->betaF  = static_cast<float>(beta);
     status = cublasGemmEx(*handle, transAblas, CUBLAS_OP_N,
                           transA ? N : M,  // m (rows of op(A))
                           1,               // n = 1 (vector)
                           transA ? M : N,  // k
                           &getCublasScalars()->alphaF,
                           pA->specialBuffer(), CUDA_R_16F, lda,
                           effX->specialBuffer(), CUDA_R_16F, N,  // ldb = N (contiguous vector)
                           &getCublasScalars()->betaF,
                           Y->specialBuffer(), CUDA_R_32F, M,    // ldc = M
                           CUBLAS_COMPUTE_32F,
                           tl_cublasLtDisabled ? CUBLAS_GEMM_DEFAULT : CUBLAS_GEMM_DEFAULT_TENSOR_OP);
   }

   if (status != CUBLAS_STATUS_SUCCESS) {
     std::string msg = "MmulHelper::mmulMxV cuda failed !; Error code: [" + std::to_string(status) + "]";
     THROW_EXCEPTION(msg.c_str());
   }

   if (!tl_cublasGapStreamReady) NDArray::registerSpecialUse({Y}, {pA, effX});

   if (pA != effA) delete pA;
 }

 delete castA;
 delete castX;

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
   Z = new NDArray(DataTypeUtils::pickPairwiseResultType(X->dataType(), Y->dataType()), X->getContext());

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
   C = new NDArray(outOrder, cExpectedShape, DataTypeUtils::pickPairwiseResultType(A->dataType(), B->dataType()),
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
 if (A->dataType() != B->dataType()) {
   const auto aType = A->dataType();
   const auto bType = B->dataType();
   const bool isMixedHalfFloat = ((aType == FLOAT32 && bType == HALF) || (aType == HALF && bType == FLOAT32));
   const bool isMixedDoubleFloat = ((aType == DOUBLE && bType == FLOAT32) || (aType == FLOAT32 && bType == DOUBLE));
   const bool isMixedDoubleHalf = ((aType == DOUBLE && bType == HALF) || (aType == HALF && bType == DOUBLE));

   if (isMixedHalfFloat || isMixedDoubleFloat || isMixedDoubleHalf) {
     // For [B0, B1, ..., M, K] × [K, N], reshape A to [B0*B1*...*M, K] (2D),
     // call mmulMxM, then result is [B0*B1*...*M, N] which we reshape to C's shape.
     // This works because each row of A is independently multiplied by B.
     const LongType M = A->sizeAt(-2);
     const LongType K = A->sizeAt(-1);
     const LongType N = B->sizeAt(-1);

     // Compute total rows = product of all dims except last
     LongType totalRows = 1;
     for (int i = 0; i < aRank - 1; ++i) {
       totalRows *= A->sizeAt(i);
     }

     // Reshape A to 2D [totalRows, K]
     std::vector<LongType> a2dShape = {totalRows, K};
     NDArray* a2d = A->reshape(A->ordering(), a2dShape, false);
     NDArray* effA2d = a2d;

     // Reshape B to 2D if needed (should already be 2D for the common case)
     NDArray* b2d = const_cast<NDArray*>(B);
     bool bReshaped = false;
     if (bRank > 2) {
       std::vector<LongType> b2dShape = {K, N};
       b2d = B->reshape(B->ordering(), b2dShape, false);
       bReshaped = true;
     }
     NDArray* effB2d = b2d;

     // During graph capture, decoder projections frequently reuse the same
     // activation tensor across multiple matmuls (for example q/k/v and gate/up).
     // Reuse the first casted HALF buffer for repeated sources, but still
     // advance the cache index so warmup and capture consume buffers in the
     // same order.
     if (tl_graphExecutionActive) {
       if (aType == FLOAT32 && bType == HALF) {
         auto reuseIt = tl_castA.captureCastReuse.find(A);
         if (reuseIt != tl_castA.captureCastReuse.end()) {
           if (tl_castA.index < tl_castA.cache.size()) tl_castA.index++;
           effA2d = reuseIt->second;
         } else if (tl_castA.index < tl_castA.cache.size()
                    && tl_castA.cache[tl_castA.index]->dataType() == HALF
                    && tl_castA.cache[tl_castA.index]->isSameShape(a2d)) {
           NDArray* cached = tl_castA.cache[tl_castA.index++];
           cached->assign(a2d);
           tl_castA.captureCastReuse.emplace(A, cached);
           effA2d = cached;
         }
       } else if (aType == HALF && bType == FLOAT32) {
         auto reuseIt = tl_castB.captureCastReuse.find(B);
         if (reuseIt != tl_castB.captureCastReuse.end()) {
           if (tl_castB.index < tl_castB.cache.size()) tl_castB.index++;
           effB2d = reuseIt->second;
         } else if (tl_castB.index < tl_castB.cache.size()
                    && tl_castB.cache[tl_castB.index]->dataType() == HALF
                    && tl_castB.cache[tl_castB.index]->isSameShape(b2d)) {
           NDArray* cached = tl_castB.cache[tl_castB.index++];
           cached->assign(b2d);
           tl_castB.captureCastReuse.emplace(B, cached);
           effB2d = cached;
         }
       }
     }

     // Reshape C to 2D [totalRows, N]
     std::vector<LongType> c2dShape = {totalRows, N};
     NDArray* c2d = C->reshape(C->ordering(), c2dShape, false);

     mmulMxM(effA2d, effB2d, c2d, alpha, beta, c2d->ordering());

     delete a2d;
     if (bReshaped) delete b2d;
     delete c2d;

     return C;
   }
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


 std::vector<LongType> cDimsVec = {cMaxis,2, cNaxis};
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

 NDArray::prepareSpecialUse({C}, {A, B});
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
// Supports both 3D [batch, M, K] and 4D [batch0, batch1, M, K] tensors
//////////////////////////////////////////////////////////////////////////
bool MmulHelper::tryBlasStridedBatched(NDArray* A, NDArray* B, NDArray* C,
                                        double alpha, double beta,
                                        bool transA, bool transB) {
  // transA/transB native fast path (cuBLAS CUBLAS_OP_T on the un-permuted operands) is not
  // reinstated in this routine yet. When a transpose is requested, decline so the caller
  // (MmulHelper.cpp ~1201) takes its DOCUMENTED alternative: re-enter on the already-permuted
  // xT/yT with transA=transB=false (the `else if (!tryBlasStridedBatched(xT, yT, zT, ...))`
  // branch). That alternative is mathematically identical — it differs only in floating-point
  // accumulation order — so this is path-selection, not a silent degrade. The 7-arg signature
  // is required because MmulHelper.h declares it and MmulHelper.cpp calls it.
  // TODO(restore-native-transpose): reinstate the CUBLAS_OP_T strided-batched path for
  // bit-for-bit parity with the per-slice 2D accumulation order.
  if (transA || transB) return false;

  const int aRank = A->rankOf();
  const int bRank = B->rankOf();
  const int cRank = C->rankOf();

  // Handle 3D and 4D tensors with matching ranks
  if (aRank != bRank || bRank != cRank || (aRank != 3 && aRank != 4)) {
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

  // For 4D tensors, flatten first two batch dimensions
  // 3D: [batch, M, K] @ [batch, K, N] -> [batch, M, N]
  // 4D: [b0, b1, M, K] @ [b0, b1, K, N] -> [b0, b1, M, N] (treat as [b0*b1, M, K] @ [b0*b1, K, N])
  LongType batchSize;
  LongType M, K, N;

  if (aRank == 3) {
    batchSize = A->sizeAt(0);
    M = A->sizeAt(1);
    K = A->sizeAt(2);
    N = B->sizeAt(2);
  } else {  // aRank == 4
    // Flatten first two batch dimensions
    batchSize = A->sizeAt(0) * A->sizeAt(1);
    M = A->sizeAt(2);
    K = A->sizeAt(3);
    N = B->sizeAt(3);

    // Verify batch dimensions match
    if (A->sizeAt(0) != B->sizeAt(0) || A->sizeAt(1) != B->sizeAt(1) ||
        A->sizeAt(0) != C->sizeAt(0) || A->sizeAt(1) != C->sizeAt(1)) {
      return false;
    }
  }

  if (M <= 0 || K <= 0 || N <= 0 || batchSize <= 0) {
    return false;
  }

  // Check inner dimensions match for matmul
  const int kAxisA = aRank - 1;  // Last axis of A
  const int kAxisB = aRank - 2;  // Second-to-last axis of B
  if (A->sizeAt(kAxisA) != B->sizeAt(kAxisB)) {
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

  // Check for contiguous row-major layout (innermost dimensions)
  const bool aRowMajor = (A->strideAt(-1) == 1) && (A->strideAt(-2) == K);
  const bool bRowMajor = (B->strideAt(-1) == 1) && (B->strideAt(-2) == N);
  const bool cRowMajor = (C->strideAt(-1) == 1) && (C->strideAt(-2) == N);

  if (!aRowMajor || !bRowMajor || !cRowMajor) {
    return false;
  }

  // Calculate strides for batched operation
  // For 3D: stride between batches is stride[0]
  // For 4D: stride between batches is stride[1] (stride within b0), and we need contiguous b0*b1
  long long strideA, strideB, strideC;

  if (aRank == 3) {
    const LongType expectedStrideA = M * K;
    const LongType expectedStrideB = K * N;
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
    const LongType expectedStrideA = M * K;
    const LongType expectedStrideB = K * N;
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
  // Skip cublasSetStream during CUDA graph capture (see mmulMxM comment above).
  if (!tl_graphExecutionActive) {
    cublasSetStream(*handle, *stream);
  }
  reapplyCublasWorkspace(*handle);

  NDArray::prepareSpecialUse({C}, {A, B});

  cublasStatus_t status;

  // cuBLAS uses column-major, so we compute C^T = B^T * A^T
  // In terms of our row-major arrays: C = A * B becomes computing with swapped order
  // cublas: C_col = B_col * A_col where our row-major A is cublas's A^T
  // So we call cublas with (B, A) to get A*B in row-major

  if (xType == DataType::DOUBLE) {
    getCublasScalars()->alphaD = alpha;
    getCublasScalars()->betaD  = beta;
    status = cublasDgemmStridedBatched(
        *handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,  // Swapped M,N for row-major
        &getCublasScalars()->alphaD,
        reinterpret_cast<const double*>(B->specialBuffer()), N, strideB,
        reinterpret_cast<const double*>(A->specialBuffer()), K, strideA,
        &getCublasScalars()->betaD,
        reinterpret_cast<double*>(C->specialBuffer()), N, strideC,
        batchSize);
  } else if (xType == DataType::FLOAT32) {
    getCublasScalars()->alphaF = static_cast<float>(alpha);
    getCublasScalars()->betaF  = static_cast<float>(beta);
    status = cublasSgemmStridedBatched(
        *handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,  // Swapped M,N for row-major
        &getCublasScalars()->alphaF,
        reinterpret_cast<const float*>(B->specialBuffer()), N, strideB,
        reinterpret_cast<const float*>(A->specialBuffer()), K, strideA,
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
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &getCublasScalars()->alphaF,
        B->specialBuffer(), CUDA_R_16F, N, strideB,
        A->specialBuffer(), CUDA_R_16F, K, strideA,
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

  NDArray::registerSpecialUse({C}, {A, B});
  // Don't sync - let CUDA operations run asynchronously

  return true;
}

} // namespace sd

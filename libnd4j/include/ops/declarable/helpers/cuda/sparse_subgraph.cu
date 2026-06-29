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

//
// csr_subgraph_extract / csr_subgraph_extract_bp — CUDA implementation.
//
// DESIGN:
//
// FORWARD — 5 deterministic steps:
//   Step 1 (markKeptKernel): one thread per original edge e in [0, nnz).
//     Linear scan over K selected rows to find e's source-row rank s.
//     Binary search colIdx[e] in nodeIdx for the destination rank.
//     keptFlag[e]=1 if both found; remappedCol[e] = destination rank.
//
//   Step 2 (rowCountKernel): one thread per selected row s.
//     Sum keptFlag over [rowPtr[nodeIdx[s]], rowPtr[nodeIdx[s]+1]) → counts[s].
//
//   Step 3 (subgraphRowPrefixKernel): single-block exclusive prefix-scan on
//     counts[0..K-1] → newRowPtr[0..K].  K is small (SAGPool); shared-memory
//     sequential scan in thread 0 is correct and sufficient.
//
//   Step 4 (computeGlobalPosKernel): one thread per edge e.
//     For each kept edge e in selected row s:
//       localRank = sum of keptFlag in [rowPtr[nodeIdx[s]], e)
//       globalPos[e] = newRowPtr[s] + localRank     (DETERMINISTIC — no atomics)
//     Dropped edges: globalPos[e] = -1.
//
//   Step 5 (scatterKernel): one thread per edge e.
//     If keptFlag[e]: write newValues[globalPos[e]], newColIdx[globalPos[e]].
//
// BACKWARD — same steps 1-4 to reconstruct globalPos, then:
//   Step 5 (bpScatterKernel): for each kept edge e,
//     dValues[e] = dNewValues[globalPos[e]].  dValues pre-zeroed.
//
// Because globalPos is computed DETERMINISTICALLY (same formula, same data),
// the forward and backward always produce the identical e→e' mapping, which
// is required for correct gradients.
//
// No raw cudaMalloc/cudaFree (temporaries via NDArrayFactory / pool).
// No thrust; hand-written sequential prefix-scan in shared memory.
// prepareSpecialUse / registerSpecialUse bracket all kernel launches.
//

#include <cuda_runtime.h>
#include <math/templatemath.h>
#include <ops/declarable/helpers/sparse_subgraph.h>
#include <system/op_boilerplate.h>
#include <types/bfloat16.h>
#include <types/float16.h>
#include <array/NDArrayFactory.h>

#include <algorithm>
#include <memory>
#include <vector>

namespace sd {
namespace ops {
namespace helpers {

// ────────────────────────────────────────────────────────────────────────────
// Device: binary search of key in sorted buf[0..K-1] → rank or -1.
// ────────────────────────────────────────────────────────────────────────────
template <typename I>
static __device__ __forceinline__ LongType devBsearch(const I* buf, LongType K, I key) {
  LongType lo = 0, hi = K - 1;
  while (lo <= hi) {
    LongType mid = (lo + hi) >> 1;
    I mv = buf[mid];
    if      (mv == key) return mid;
    else if (mv < key)  lo = mid + 1;
    else                hi = mid - 1;
  }
  return static_cast<LongType>(-1);
}

// ────────────────────────────────────────────────────────────────────────────
// Device: find which selected-row s owns original edge e.
// Linear scan over K (K is small in the SAGPool use-case).
// ────────────────────────────────────────────────────────────────────────────
template <typename I>
static __device__ __forceinline__
LongType findSelectedRowRank(const I* niPtr, const I* rpPtr, LongType K, LongType e) {
  for (LongType s = 0; s < K; ++s) {
    LongType r      = static_cast<LongType>(niPtr[s]);
    LongType eStart = static_cast<LongType>(rpPtr[r]);
    LongType eEnd   = static_cast<LongType>(rpPtr[r + 1]);
    if (e >= eStart && e < eEnd) return s;
  }
  return static_cast<LongType>(-1);
}

// ────────────────────────────────────────────────────────────────────────────
// Step 1 — markKeptKernel
// ────────────────────────────────────────────────────────────────────────────
template <typename I>
static SD_KERNEL void markKeptKernel(const I* ciPtr, const I* rpPtr, const I* niPtr,
                                      int* keptFlag, int* remappedCol,
                                      LongType nnz, LongType K) {
  const LongType e = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (e >= nnz) return;

  LongType srcRank = findSelectedRowRank(niPtr, rpPtr, K, e);
  if (srcRank < 0) { keptFlag[e] = 0; return; }

  I col = ciPtr[e];
  LongType dstRank = devBsearch(niPtr, K, col);
  if (dstRank < 0) { keptFlag[e] = 0; return; }

  keptFlag[e]    = 1;
  remappedCol[e] = static_cast<int>(dstRank);
}

// ────────────────────────────────────────────────────────────────────────────
// Step 2 — rowCountKernel
// ────────────────────────────────────────────────────────────────────────────
template <typename I>
static SD_KERNEL void rowCountKernel(const I* rpPtr, const I* niPtr,
                                      const int* keptFlag, int* counts, LongType K) {
  const LongType s = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (s >= K) return;

  const LongType r      = static_cast<LongType>(niPtr[s]);
  const LongType eStart = static_cast<LongType>(rpPtr[r]);
  const LongType eEnd   = static_cast<LongType>(rpPtr[r + 1]);

  int cnt = 0;
  for (LongType e = eStart; e < eEnd; ++e) cnt += keptFlag[e];
  counts[s] = cnt;
}

// ────────────────────────────────────────────────────────────────────────────
// Step 3 — single-block exclusive prefix-scan: counts[0..K-1] → newRowPtr[0..K].
// Sequential scan in thread 0 using shared memory; K is small.
// Mirrors csrPrefixSumKernel pattern from sparse_csr.cu.
// ────────────────────────────────────────────────────────────────────────────
static SD_KERNEL void subgraphRowPrefixKernel(const int* counts, int* newRowPtr, LongType K) {
  extern __shared__ int smem[];  // (K+1) ints

  for (LongType i = threadIdx.x; i < K; i += blockDim.x) smem[i + 1] = counts[i];
  __syncthreads();

  if (threadIdx.x == 0) {
    smem[0] = 0;
    for (LongType i = 1; i <= K; ++i) smem[i] += smem[i - 1];
  }
  __syncthreads();

  for (LongType i = threadIdx.x; i <= K; i += blockDim.x) newRowPtr[i] = smem[i];
}

// ────────────────────────────────────────────────────────────────────────────
// Step 4 — computeGlobalPosKernel (DETERMINISTIC — no atomics).
// ────────────────────────────────────────────────────────────────────────────
template <typename I>
static SD_KERNEL void computeGlobalPosKernel(const I* rpPtr, const I* niPtr,
                                              const int* keptFlag, const int* newRowPtr,
                                              int* globalPos, LongType nnz, LongType K) {
  const LongType e = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (e >= nnz) return;

  if (!keptFlag[e]) { globalPos[e] = -1; return; }

  LongType s = findSelectedRowRank(niPtr, rpPtr, K, e);
  if (s < 0) { globalPos[e] = -1; return; }

  const LongType eStart = static_cast<LongType>(rpPtr[static_cast<LongType>(niPtr[s])]);
  int localRank = 0;
  for (LongType ep = eStart; ep < e; ++ep) localRank += keptFlag[ep];

  globalPos[e] = newRowPtr[s] + localRank;
}

// ────────────────────────────────────────────────────────────────────────────
// Step 5 (forward) — scatter values and remapped column indices
// ────────────────────────────────────────────────────────────────────────────
template <typename X>
static SD_KERNEL void scatterKernel(const X* valPtr, const int* remappedCol,
                                     const int* globalPos, X* nvPtr, int* nciPtr, LongType nnz) {
  const LongType e = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (e >= nnz) return;
  int gp = globalPos[e];
  if (gp < 0) return;
  nvPtr[gp]  = valPtr[e];
  nciPtr[gp] = remappedCol[e];
}

// ────────────────────────────────────────────────────────────────────────────
// Step 5 (backward) — scatter dNewValues[globalPos[e]] → dValues[e]
// ────────────────────────────────────────────────────────────────────────────
template <typename X>
static SD_KERNEL void bpScatterKernel(const int* globalPos, const X* dnvPtr,
                                       X* dvPtr, LongType nnz) {
  const LongType e = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (e >= nnz) return;
  int gp = globalPos[e];
  if (gp < 0) return;
  dvPtr[e] = dnvPtr[gp];
}

// ────────────────────────────────────────────────────────────────────────────
// Shared helper: run steps 1-4 to fill keptFlag, remappedCol, newRowPtr,
// globalPos on the device.
//
// Caller owns the temporary NDArray* allocations; they must remain live until
// the kernels complete.  Since all kernels run on the same stream, the caller
// can destroy them after stream synchronisation (or just let unique_ptr handle
// it after the helper returns, which is safe because the helper is synchronous
// from the perspective of kernel ordering on the same stream).
// ────────────────────────────────────────────────────────────────────────────
template <typename I>
static void computeSubgraphStructure(
    cudaStream_t stream,
    const I* ciPtr, const I* rpPtr, const I* niPtr,
    int* keptFlag, int* remappedCol, int* counts, int* nrpPtr, int* globalPos,
    LongType nnz, LongType K)
{
  const int BS = 256;

  if (nnz > 0) {
    int grid = static_cast<int>((nnz + BS - 1) / BS);
    markKeptKernel<I><<<grid, BS, 0, stream>>>(ciPtr, rpPtr, niPtr, keptFlag, remappedCol, nnz, K);
  }

  if (K > 0) {
    int grid = static_cast<int>((K + BS - 1) / BS);
    rowCountKernel<I><<<grid, BS, 0, stream>>>(rpPtr, niPtr, keptFlag, counts, K);

    int smemBytes = static_cast<int>((K + 1) * sizeof(int));
    int blkScan   = static_cast<int>(std::min(static_cast<LongType>(K + 1),
                                              static_cast<LongType>(1024)));
    subgraphRowPrefixKernel<<<1, blkScan, smemBytes, stream>>>(counts, nrpPtr, K);
  } else {
    // K==0: newRowPtr has one element; set it to 0
    if (nrpPtr) cudaMemsetAsync(nrpPtr, 0, sizeof(int), stream);
  }

  if (nnz > 0) {
    int grid = static_cast<int>((nnz + BS - 1) / BS);
    computeGlobalPosKernel<I><<<grid, BS, 0, stream>>>(rpPtr, niPtr, keptFlag, nrpPtr, globalPos, nnz, K);
  }
}

// ────────────────────────────────────────────────────────────────────────────
// Typed dispatch — forward
// ────────────────────────────────────────────────────────────────────────────
template <typename X, typename I>
static void csrSubgraphExtractCuda_(LaunchContext* ctx,
                                     NDArray& values,    NDArray& colIdx,    NDArray& rowPtr,
                                     NDArray& nodeIdx,   NDArray& newValues, NDArray& newColIdx,
                                     NDArray& newRowPtr, sd::LongType N,     sd::LongType K) {
  auto* stream = ctx->getCudaStream();

  const LongType nnz  = values.lengthOf();
  const LongType nnzP = newValues.lengthOf();

  const X* valPtr = reinterpret_cast<const X*>(values.specialBuffer());
  const I* ciPtr  = reinterpret_cast<const I*>(colIdx.specialBuffer());
  const I* rpPtr  = reinterpret_cast<const I*>(rowPtr.specialBuffer());
  const I* niPtr  = reinterpret_cast<const I*>(nodeIdx.specialBuffer());
  X*       nvPtr  = reinterpret_cast<X*>(newValues.specialBuffer());
  int*     nciPtr = reinterpret_cast<int*>(newColIdx.specialBuffer());
  int*     nrpPtr = reinterpret_cast<int*>(newRowPtr.specialBuffer());

  // Pool-backed temporaries (released on scope exit via unique_ptr)
  const LongType safeNnz = std::max(nnz, static_cast<LongType>(1));
  const LongType safeK   = std::max(K,   static_cast<LongType>(1));

  std::unique_ptr<NDArray> keptFlagArr   (NDArrayFactory::create('c', std::vector<LongType>{safeNnz}, DataType::INT32, ctx));
  std::unique_ptr<NDArray> remappedColArr(NDArrayFactory::create('c', std::vector<LongType>{safeNnz}, DataType::INT32, ctx));
  std::unique_ptr<NDArray> countsArr     (NDArrayFactory::create('c', std::vector<LongType>{safeK},   DataType::INT32, ctx));
  std::unique_ptr<NDArray> globalPosArr  (NDArrayFactory::create('c', std::vector<LongType>{safeNnz}, DataType::INT32, ctx));

  int* keptFlag    = reinterpret_cast<int*>(keptFlagArr->specialBuffer());
  int* remappedCol = reinterpret_cast<int*>(remappedColArr->specialBuffer());
  int* counts      = reinterpret_cast<int*>(countsArr->specialBuffer());
  int* globalPos   = reinterpret_cast<int*>(globalPosArr->specialBuffer());

  if (nnz > 0) {
    cudaMemsetAsync(keptFlag,    0, sizeof(int) * nnz, *stream);
    cudaMemsetAsync(remappedCol, 0, sizeof(int) * nnz, *stream);
    cudaMemsetAsync(globalPos,   0, sizeof(int) * nnz, *stream);
  }
  if (K > 0) {
    cudaMemsetAsync(counts, 0, sizeof(int) * K, *stream);
  }

  // Steps 1-4
  computeSubgraphStructure<I>(*stream, ciPtr, rpPtr, niPtr,
                               keptFlag, remappedCol, counts, nrpPtr, globalPos, nnz, K);

  // Step 5: scatter
  if (nnz > 0 && nnzP > 0) {
    int grid = static_cast<int>((nnz + 255) / 256);
    scatterKernel<X><<<grid, 256, 0, *stream>>>(valPtr, remappedCol, globalPos, nvPtr, nciPtr, nnz);
  }
}

void csr_subgraph_extract(LaunchContext* ctx,
                           NDArray& values,    NDArray& colIdx,    NDArray& rowPtr,
                           NDArray& nodeIdx,   NDArray& newValues, NDArray& newColIdx,
                           NDArray& newRowPtr, sd::LongType N,     sd::LongType K) {
  NDArray::prepareSpecialUse({&newValues, &newColIdx, &newRowPtr},
                             {&values, &colIdx, &rowPtr, &nodeIdx});

  BUILD_DOUBLE_SELECTOR(values.dataType(), colIdx.dataType(), csrSubgraphExtractCuda_,
                        (ctx, values, colIdx, rowPtr, nodeIdx, newValues, newColIdx, newRowPtr, N, K),
                        SD_FLOAT_TYPES, SD_INDEXING_TYPES);

  NDArray::registerSpecialUse({&newValues, &newColIdx, &newRowPtr},
                              {&values, &colIdx, &rowPtr, &nodeIdx});
}

// ────────────────────────────────────────────────────────────────────────────
// Typed dispatch — backward
// ────────────────────────────────────────────────────────────────────────────
template <typename X, typename I>
static void csrSubgraphExtractBpCuda_(LaunchContext* ctx,
                                       NDArray& values,     NDArray& colIdx,
                                       NDArray& rowPtr,     NDArray& nodeIdx,
                                       NDArray& dNewValues, NDArray& dValues,
                                       sd::LongType N,      sd::LongType K) {
  auto* stream = ctx->getCudaStream();

  const LongType nnz  = values.lengthOf();
  const LongType nnzP = dNewValues.lengthOf();

  const I* ciPtr  = reinterpret_cast<const I*>(colIdx.specialBuffer());
  const I* rpPtr  = reinterpret_cast<const I*>(rowPtr.specialBuffer());
  const I* niPtr  = reinterpret_cast<const I*>(nodeIdx.specialBuffer());
  const X* dnvPtr = reinterpret_cast<const X*>(dNewValues.specialBuffer());
  X*       dvPtr  = reinterpret_cast<X*>(dValues.specialBuffer());

  // Zero dValues on the context stream
  if (dValues.lengthOf() > 0)
    cudaMemsetAsync(dvPtr, 0, sizeof(X) * static_cast<size_t>(dValues.lengthOf()), *stream);

  if (nnz == 0 || nnzP == 0) return;

  const LongType safeNnz = std::max(nnz, static_cast<LongType>(1));
  const LongType safeK   = std::max(K,   static_cast<LongType>(1));

  std::unique_ptr<NDArray> keptFlagArr   (NDArrayFactory::create('c', std::vector<LongType>{safeNnz}, DataType::INT32, ctx));
  std::unique_ptr<NDArray> remappedColArr(NDArrayFactory::create('c', std::vector<LongType>{safeNnz}, DataType::INT32, ctx));
  std::unique_ptr<NDArray> countsArr     (NDArrayFactory::create('c', std::vector<LongType>{safeK},   DataType::INT32, ctx));
  std::unique_ptr<NDArray> newRowPtrArr  (NDArrayFactory::create('c', std::vector<LongType>{K + 1},   DataType::INT32, ctx));
  std::unique_ptr<NDArray> globalPosArr  (NDArrayFactory::create('c', std::vector<LongType>{safeNnz}, DataType::INT32, ctx));

  int* keptFlag  = reinterpret_cast<int*>(keptFlagArr->specialBuffer());
  int* remapped  = reinterpret_cast<int*>(remappedColArr->specialBuffer());
  int* counts    = reinterpret_cast<int*>(countsArr->specialBuffer());
  int* nrpPtr    = reinterpret_cast<int*>(newRowPtrArr->specialBuffer());
  int* globalPos = reinterpret_cast<int*>(globalPosArr->specialBuffer());

  if (nnz > 0) {
    cudaMemsetAsync(keptFlag, 0, sizeof(int) * nnz, *stream);
    cudaMemsetAsync(remapped, 0, sizeof(int) * nnz, *stream);
    cudaMemsetAsync(globalPos,0, sizeof(int) * nnz, *stream);
  }
  if (K > 0) cudaMemsetAsync(counts, 0, sizeof(int) * K, *stream);

  // Re-run the same deterministic structure to recover globalPos
  computeSubgraphStructure<I>(*stream, ciPtr, rpPtr, niPtr,
                               keptFlag, remapped, counts, nrpPtr, globalPos, nnz, K);

  // Backward scatter
  int grid = static_cast<int>((nnz + 255) / 256);
  bpScatterKernel<X><<<grid, 256, 0, *stream>>>(globalPos, dnvPtr, dvPtr, nnz);
}

void csr_subgraph_extract_bp(LaunchContext* ctx,
                               NDArray& values,     NDArray& colIdx,
                               NDArray& rowPtr,     NDArray& nodeIdx,
                               NDArray& dNewValues, NDArray& dValues,
                               sd::LongType N,      sd::LongType K) {
  NDArray::prepareSpecialUse({&dValues},
                             {&values, &colIdx, &rowPtr, &nodeIdx, &dNewValues});

  BUILD_DOUBLE_SELECTOR(values.dataType(), colIdx.dataType(), csrSubgraphExtractBpCuda_,
                        (ctx, values, colIdx, rowPtr, nodeIdx, dNewValues, dValues, N, K),
                        SD_FLOAT_TYPES, SD_INDEXING_TYPES);

  NDArray::registerSpecialUse({&dValues},
                              {&values, &colIdx, &rowPtr, &nodeIdx, &dNewValues});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd

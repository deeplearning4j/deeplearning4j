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

#include <cuda_runtime.h>
#include <math/templatemath.h>
#include <memory/cuda/CudaMemoryPool.h>
#include <ops/declarable/helpers/sparse_coo.h>
#include <system/op_boilerplate.h>

// ---------------------------------------------------------------------------
// NOTE: Thrust is deliberately NOT used in this file.  libnd4j avoids Thrust
// because it is not guaranteed to be present in all CUDA distributions.
// All prefix-sum, sort, and reduce-by-key operations are replaced with
// portable hand-written CUDA kernels below.
// ---------------------------------------------------------------------------

namespace sd {
namespace ops {
namespace helpers {

// ===========================================================================
// Portable chunked inclusive prefix-sum (int32_t specialisation)
//
// Correct for ARBITRARY n — no upper-bound on array size.
//
// Algorithm (single-block, chunked carry):
//   One block loops over the input in chunks of up to SCAN_BLOCK elements:
//     1. Load chunk from d_in into shared memory (zero-pad short tail).
//     2. Hillis-Steele inclusive scan within shared memory.
//     3. Write chunk results to d_out, each element offset by `carry`
//        (= inclusive sum of all preceding chunks).
//     4. Advance carry by the chunk total (smem[chunk_sz - 1]).
//
// This mirrors csrPrefixSumKernel in sparse_csr.cu but is out-of-place
// (reads d_in, writes d_out).  d_blockSums is accepted for API compatibility
// with existing call sites but is NOT used — the chunked single-block
// algorithm needs no inter-block scratch buffer.
//
// All phases run on the caller's LaunchContext stream.
// ===========================================================================

static constexpr int SCAN_BLOCK = 512;

// Chunked out-of-place inclusive scan: d_out[i] = sum(d_in[0..i]) for all i.
// Single block loops over chunks of `blockDim.x` — correct for any n.
__global__ void cooInclusiveScanKernel(const int32_t* __restrict__ d_in,
                                       int32_t*       __restrict__ d_out,
                                       int n) {
  extern __shared__ int32_t smem[];
  const int bsz   = blockDim.x;
  const int tid   = threadIdx.x;
  int32_t   carry = 0;

  for (int base = 0; base < n; base += bsz) {
    const int rem      = n - base;
    const int chunk_sz = rem < bsz ? rem : bsz;

    // Load chunk; zero-pad tail so Hillis-Steele arithmetic is always clean.
    smem[tid] = (tid < chunk_sz) ? d_in[base + tid] : 0;
    __syncthreads();

    // Hillis-Steele inclusive scan over the shared-memory window.
    for (int stride = 1; stride < bsz; stride <<= 1) {
      const int32_t val = (tid >= stride) ? smem[tid - stride] : 0;
      __syncthreads();
      smem[tid] += val;
      __syncthreads();
    }

    // Capture chunk total before write-back (smem still holds scan results).
    const int32_t chunkSum = smem[chunk_sz - 1];

    // Write: element i gets carry (sum of all prior chunks) + its local prefix.
    if (tid < chunk_sz) d_out[base + tid] = carry + smem[tid];
    __syncthreads();  // ensure writes complete before next chunk load

    carry += chunkSum;  // uniform across all threads
  }
}

// Host-callable launcher: inclusive scan of d_in[n] → d_out[n].
// d_blockSums is accepted for API compatibility with existing call sites but
// is not used — the chunked single-block algorithm needs no scratch buffer.
static void portableInclusiveScan(const int32_t* d_in, int32_t* d_out,
                                  int32_t* /*d_blockSums*/, int n,
                                  cudaStream_t stream) {
  if (n <= 0) return;
  // Select block size: smallest power-of-two >= min(n, SCAN_BLOCK).
  // For n > SCAN_BLOCK the kernel simply loops over multiple chunks.
  int bsz = 1;
  while (bsz < n && bsz < SCAN_BLOCK) bsz <<= 1;
  const size_t smem = static_cast<size_t>(bsz) * sizeof(int32_t);
  cooInclusiveScanKernel<<<1, bsz, smem, stream>>>(d_in, d_out, n);
}

// ===========================================================================
// denseToCoo — per-row design, no large flag array, no Thrust
//
// Algorithm:
//   Phase 1 (dtcCountKernel)  – 1 thread per row: count |val|>threshold
//                                columns → rowCount[r].
//   Phase 2                   – portableInclusiveScan(rowCount) → rowOffsetIncl.
//                                rowOffsetIncl[r] = total nonzeros in rows [0..r].
//   Phase 3 (dtcScatterKernel)– 1 thread per row: walk columns ascending;
//                                for each |val|>threshold write outIndices and
//                                outValues at position (r>0 ? rowOffsetIncl[r-1]:0)
//                                + local counter (register, no atomics).
//
// Row-major column walk ⇒ output sorted by (row, col) without any sort pass.
// Nonzero test |x|>threshold matches the DECLARE_SHAPE_FN count.
// Dense is read via specialBuffer() with strides passed as kernel args —
// fully stride-aware; no contiguity assumption.
// ===========================================================================

// Phase 1: count nonzero elements per row.
template <typename X>
__global__ void dtcCountKernel(const X* __restrict__ dense,
                               int32_t* __restrict__ rowCount,
                               sd::LongType rows, sd::LongType cols,
                               sd::LongType stride0, sd::LongType stride1,
                               X threshold) {
  sd::LongType r = static_cast<sd::LongType>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (r >= rows) return;
  int32_t cnt = 0;
  for (sd::LongType c = 0; c < cols; ++c) {
    X v = dense[r * stride0 + c * stride1];
    X a = v < static_cast<X>(0) ? -v : v;
    if (a > threshold) ++cnt;
  }
  rowCount[r] = cnt;
}

// Phase 3: scatter each nonzero into outIndices / outValues.
// rowOffsetIncl[r] = inclusive prefix sum of rowCount → exclusive start of row r
// is (r > 0 ? rowOffsetIncl[r-1] : 0).
template <typename X>
__global__ void dtcScatterKernel(const X*          __restrict__ dense,
                                 const int32_t*    __restrict__ rowOffsetIncl,
                                 sd::LongType*     __restrict__ outIndices,
                                 X*                __restrict__ outValues,
                                 sd::LongType rows, sd::LongType cols,
                                 sd::LongType stride0, sd::LongType stride1,
                                 X threshold) {
  sd::LongType r = static_cast<sd::LongType>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (r >= rows) return;
  int32_t pos = (r == 0) ? 0 : rowOffsetIncl[r - 1];
  for (sd::LongType c = 0; c < cols; ++c) {
    X v = dense[r * stride0 + c * stride1];
    X a = v < static_cast<X>(0) ? -v : v;
    if (a > threshold) {
      outIndices[pos * 2 + 0] = static_cast<sd::LongType>(r);
      outIndices[pos * 2 + 1] = static_cast<sd::LongType>(c);
      outValues[pos]          = v;
      ++pos;
    }
  }
}

template <typename X>
static void denseToCooCuda_(sd::LaunchContext* context,
                            NDArray* dense, NDArray* indices, NDArray* values,
                            double threshold) {
  const sd::LongType rows   = dense->sizeAt(0);
  const sd::LongType cols   = dense->sizeAt(1);
  const X            thr    = static_cast<X>(threshold);
  const int          devId  = context->getDeviceID();
  const cudaStream_t stream = *context->getCudaStream();

  const sd::LongType stride0 = dense->stridesOf()[0];
  const sd::LongType stride1 = dense->stridesOf()[1];

  const X*      dDense = reinterpret_cast<const X*>(dense->specialBuffer());
  sd::LongType* dIdx   = reinterpret_cast<sd::LongType*>(indices->specialBuffer());
  X*            dVals  = reinterpret_cast<X*>(values->specialBuffer());

  if (rows == 0 || cols == 0 || values->lengthOf() == 0) return;

  auto& pool = sd::memory::CudaMemoryPool::getInstance();

  // Temporaries (all via CudaMemoryPool — no raw cudaMalloc):
  //   rowCount[rows]   – per-row nonzero count (Phase 1 output)
  //   rowOffsetIncl[rows] – inclusive prefix sum (Phase 2 output)
  //   blockSums[nBlocks]  – scratch for portableInclusiveScan
  const int    nScanBlocks = (int)((rows + SCAN_BLOCK - 1) / SCAN_BLOCK);
  const size_t rowBytes    = static_cast<size_t>(rows) * sizeof(int32_t);
  const size_t bsBytes     = static_cast<size_t>(nScanBlocks) * sizeof(int32_t);

  void* rcRaw = pool.allocate(rowBytes, devId, stream);
  void* roRaw = pool.allocate(rowBytes, devId, stream);
  void* bsRaw = pool.allocate(bsBytes,  devId, stream);

  int32_t* dRowCount  = reinterpret_cast<int32_t*>(rcRaw);
  int32_t* dRowOffset = reinterpret_cast<int32_t*>(roRaw);
  int32_t* dBlockSums = reinterpret_cast<int32_t*>(bsRaw);

  // Phase 1 — count nonzeros per row.
  {
    const int threads = 128;
    const int blocks  = (int)((rows + threads - 1) / threads);
    dtcCountKernel<X><<<blocks, threads, 0, stream>>>(
        dDense, dRowCount, rows, cols, stride0, stride1, thr);
  }

  // Phase 2 — inclusive prefix sum: rowOffsetIncl[r] = # nonzeros in rows [0..r].
  portableInclusiveScan(dRowCount, dRowOffset, dBlockSums, (int)rows, stream);

  // Phase 3 — scatter nonzeros into COO output, sorted by (row, col).
  {
    const int threads = 128;
    const int blocks  = (int)((rows + threads - 1) / threads);
    dtcScatterKernel<X><<<blocks, threads, 0, stream>>>(
        dDense, dRowOffset, dIdx, dVals, rows, cols, stride0, stride1, thr);
  }

  pool.free(rcRaw, devId, stream);
  pool.free(roRaw, devId, stream);
  pool.free(bsRaw, devId, stream);
}

void denseToCoo(sd::LaunchContext* context, NDArray* dense, NDArray* indices, NDArray* values,
                double threshold) {
  NDArray::prepareSpecialUse({indices, values}, {dense});

  BUILD_SINGLE_SELECTOR(dense->dataType(), denseToCooCuda_,
                        (context, dense, indices, values, threshold),
                        SD_FLOAT_TYPES);

  NDArray::registerSpecialUse({indices, values}, {dense});
}

// ===========================================================================
// cooToCsr — portable coalescing, no global sort, no Thrust
//
// Input:  cooIndices[nnz, 2] = (row, col), arbitrary order, possible duplicate
//         (row, col) pairs.  cooValues[nnz].
// Output: CSR with per-row ascending-col UNIQUE columns; duplicate (row,col)
//         values are SUMMED.  csrValues, colIdx, rowPtr sized by DECLARE_SHAPE_FN.
//
// Six-phase algorithm (no global sort):
//   Phase 1 (ccsCountByRowKernel)   – atomicAdd per entry into rowCount[row].
//   Phase 2 (portableInclusiveScan) – rowCount → exclusive row starts in
//                                     rowStart[0..rows] (rowStart[0]=0).
//   Phase 3 (ccsGroupByRowKernel)   – bin each entry into grouped temp arrays
//                                     (cols/vals) keyed by row, unsorted within
//                                     row, duplicates present.
//   Phase 4 (ccsPerRowCoalesceKernel)–1 thread/row: insertion-sort slice by col,
//                                     then merge equal cols summing values in
//                                     place; record unique count in coalCount[r].
//   Phase 5 (portableInclusiveScan) – coalCount → csrRowPtr[1..rows]
//                                     (csrRowPtr[0]=0 from earlier memset).
//   Phase 6 (ccsWriteCsrKernel)     – 1 thread/row: copy deduped prefix from
//                                     grouped temp into csrColIdx / csrValues.
//
// All temporaries via CudaMemoryPool.  No sync-to-host.  No raw cudaMalloc.
// prepareSpecialUse / registerSpecialUse used for coherence (no manual ticks).
// ===========================================================================

// Phase 1: count raw (possibly duplicate) entries per row.
template <typename I>
__global__ void ccsCountByRowKernel(const I*  __restrict__ cooIndices,
                                    int32_t*  __restrict__ rowCount,
                                    sd::LongType nnz) {
  sd::LongType e = static_cast<sd::LongType>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (e >= nnz) return;
  int32_t r = static_cast<int32_t>(cooIndices[e * 2 + 0]);
  atomicAdd(&rowCount[r], 1);
}

// Phase 3: bin each COO entry into the grouped temp arrays by row.
// rowFill[r] is an atomic fill counter initialised to 0.
template <typename X, typename I>
__global__ void ccsGroupByRowKernel(const I*       __restrict__ cooIndices,
                                    const X*       __restrict__ cooValues,
                                    const int32_t* __restrict__ rowStart,
                                    int32_t*       __restrict__ rowFill,
                                    int32_t*       __restrict__ groupedCol,
                                    X*             __restrict__ groupedVal,
                                    sd::LongType nnz) {
  sd::LongType e = static_cast<sd::LongType>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (e >= nnz) return;
  int32_t r   = static_cast<int32_t>(cooIndices[e * 2 + 0]);
  int32_t c   = static_cast<int32_t>(cooIndices[e * 2 + 1]);
  X       v   = cooValues[e];
  int32_t pos = rowStart[r] + atomicAdd(&rowFill[r], 1);
  groupedCol[pos] = c;
  groupedVal[pos] = v;
}

// Phase 4: per-row insertion-sort by col + in-place dedup (sum equal cols).
// 1 thread per row.  Row degree is assumed small (sparse graph/matrix use case).
// Operates on the slice [rowStart[r], rowStart[r+1]) of groupedCol/groupedVal.
// Writes the unique count into coalCount[r].
template <typename X>
__global__ void ccsPerRowCoalesceKernel(int32_t*       __restrict__ groupedCol,
                                        X*             __restrict__ groupedVal,
                                        const int32_t* __restrict__ rowStart,
                                        int32_t*       __restrict__ coalCount,
                                        sd::LongType rows) {
  sd::LongType r = static_cast<sd::LongType>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (r >= rows) return;

  int32_t lo  = rowStart[r];
  int32_t hi  = rowStart[r + 1];  // rowStart has size rows+1
  int32_t deg = hi - lo;

  if (deg == 0) {
    coalCount[r] = 0;
    return;
  }

  // Insertion sort by col within [lo, hi).
  for (int32_t i = lo + 1; i < hi; ++i) {
    int32_t kc = groupedCol[i];
    X       kv = groupedVal[i];
    int32_t j  = i - 1;
    while (j >= lo && groupedCol[j] > kc) {
      groupedCol[j + 1] = groupedCol[j];
      groupedVal[j + 1] = groupedVal[j];
      --j;
    }
    groupedCol[j + 1] = kc;
    groupedVal[j + 1] = kv;
  }

  // In-place dedup: sum values for equal consecutive cols, compact forward.
  int32_t out = lo;
  for (int32_t i = lo + 1; i < hi; ++i) {
    if (groupedCol[i] == groupedCol[out]) {
      groupedVal[out] += groupedVal[i];
    } else {
      ++out;
      groupedCol[out] = groupedCol[i];
      groupedVal[out] = groupedVal[i];
    }
  }
  coalCount[r] = out - lo + 1;
}

// Phase 6: copy the deduped per-row slice from the grouped temp into the final
// CSR column-index and values arrays.
// csrRowPtr[r] = destination offset for row r (from Phase 5).
// groupedCol/groupedVal[rowStart[r]..rowStart[r]+coalCount[r]) = source.
template <typename X>
__global__ void ccsWriteCsrKernel(const int32_t* __restrict__ groupedCol,
                                  const X*       __restrict__ groupedVal,
                                  const int32_t* __restrict__ rowStart,
                                  const int32_t* __restrict__ csrRowPtr,
                                  int32_t*       __restrict__ csrColIdx,
                                  X*             __restrict__ csrValues,
                                  sd::LongType rows) {
  sd::LongType r = static_cast<sd::LongType>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (r >= rows) return;
  int32_t src = rowStart[r];
  int32_t dst = csrRowPtr[r];
  int32_t cnt = csrRowPtr[r + 1] - dst;
  for (int32_t i = 0; i < cnt; ++i) {
    csrColIdx[dst + i] = groupedCol[src + i];
    csrValues[dst + i] = groupedVal[src + i];
  }
}

template <typename X, typename I>
static void cooToCsrCuda_(sd::LaunchContext* context,
                          NDArray* cooIndices, NDArray* cooValues,
                          NDArray* csrValues, NDArray* colIdx, NDArray* rowPtr,
                          sd::LongType rows) {
  const sd::LongType nnz          = cooValues->lengthOf();
  const sd::LongType coalescedNnz = csrValues->lengthOf();
  const int          devId        = context->getDeviceID();
  const cudaStream_t stream       = *context->getCudaStream();

  // Zero rowPtr (rows+1 int32 elements) so rowPtr[0]=0 and Phase 5's
  // portableInclusiveScan can write into rowPtr[1..rows].
  cudaMemsetAsync(rowPtr->specialBuffer(), 0,
                  static_cast<size_t>(rows + 1) * sizeof(int32_t), stream);

  if (nnz == 0) return;

  auto& pool = sd::memory::CudaMemoryPool::getInstance();

  const int nScanBlocks = (int)((rows + SCAN_BLOCK - 1) / SCAN_BLOCK);

  // All temporaries via CudaMemoryPool (no raw cudaMalloc):
  //   rowCount[rows]     – Phase 1: raw entry count per row (incl. dups)
  //   rowFill[rows]      – Phase 3: atomic fill counter per row
  //   rowStart[rows+1]   – Phase 2 output: exclusive start offset per row
  //                        (rowStart[0]=0, rowStart[r+1]=sum of rowCount[0..r])
  //   groupedCol[nnz]    – Phase 3/4 temp: col indices grouped by row
  //   groupedVal[nnz]    – Phase 3/4 temp: values grouped by row
  //   coalCount[rows]    – Phase 4 output: unique col count per row after dedup
  //   blockSums[nBlocks] – scratch for portableInclusiveScan (reused for
  //                        both Phase-2 and Phase-5 scans)
  const size_t rowBytes  = static_cast<size_t>(rows)       * sizeof(int32_t);
  const size_t rsBytes   = static_cast<size_t>(rows + 1)   * sizeof(int32_t);
  const size_t nnzIBytes = static_cast<size_t>(nnz)        * sizeof(int32_t);
  const size_t nnzVBytes = static_cast<size_t>(nnz)        * sizeof(X);
  const size_t bsBytes   = static_cast<size_t>(nScanBlocks)* sizeof(int32_t);

  void* rcRaw = pool.allocate(rowBytes,  devId, stream);
  void* rfRaw = pool.allocate(rowBytes,  devId, stream);
  void* rsRaw = pool.allocate(rsBytes,   devId, stream);
  void* gcRaw = pool.allocate(nnzIBytes, devId, stream);
  void* gvRaw = pool.allocate(nnzVBytes, devId, stream);
  void* ccRaw = pool.allocate(rowBytes,  devId, stream);
  void* bsRaw = pool.allocate(bsBytes,   devId, stream);

  int32_t* dRowCount  = reinterpret_cast<int32_t*>(rcRaw);
  int32_t* dRowFill   = reinterpret_cast<int32_t*>(rfRaw);
  int32_t* dRowStart  = reinterpret_cast<int32_t*>(rsRaw);
  int32_t* dGroupedCol= reinterpret_cast<int32_t*>(gcRaw);
  X*       dGroupedVal= reinterpret_cast<X*>(gvRaw);
  int32_t* dCoalCount = reinterpret_cast<int32_t*>(ccRaw);
  int32_t* dBlockSums = reinterpret_cast<int32_t*>(bsRaw);

  // Zero the counters and the rowStart[0] sentinel (entire rowStart zeroed for
  // safety; rowStart[0]=0 must hold for the exclusive-scan trick).
  cudaMemsetAsync(dRowCount, 0, rowBytes, stream);
  cudaMemsetAsync(dRowFill,  0, rowBytes, stream);
  cudaMemsetAsync(dRowStart, 0, rsBytes,  stream);

  const I* dIdxIn  = reinterpret_cast<const I*>(cooIndices->specialBuffer());
  const X* dValIn  = reinterpret_cast<const X*>(cooValues->specialBuffer());
  int32_t* dColIdx = reinterpret_cast<int32_t*>(colIdx->specialBuffer());
  int32_t* dRowPtr = reinterpret_cast<int32_t*>(rowPtr->specialBuffer());
  X*       dCsrVal = reinterpret_cast<X*>(csrValues->specialBuffer());

  // Phase 1 — count entries per row (raw, including duplicates).
  {
    const int threads = 256;
    const int blocks  = (int)((nnz + threads - 1) / threads);
    ccsCountByRowKernel<I><<<blocks, threads, 0, stream>>>(dIdxIn, dRowCount, nnz);
  }

  // Phase 2 — build exclusive row-start offsets.
  // We write the inclusive scan of rowCount into dRowStart[1..rows]; dRowStart[0]
  // stays 0 (from the memset above), giving us the exclusive-prefix-sum layout:
  //   dRowStart[r]   = sum of rowCount[0..r-1]  (start of row r in grouped buf)
  //   dRowStart[r+1] = sum of rowCount[0..r]    (one-past-end of row r)
  portableInclusiveScan(dRowCount, dRowStart + 1, dBlockSums, (int)rows, stream);

  // Phase 3 — bin each COO entry into grouped temp arrays by row.
  {
    const int threads = 256;
    const int blocks  = (int)((nnz + threads - 1) / threads);
    ccsGroupByRowKernel<X, I><<<blocks, threads, 0, stream>>>(
        dIdxIn, dValIn, dRowStart, dRowFill, dGroupedCol, dGroupedVal, nnz);
  }

  // Phase 4 — per-row insertion-sort + in-place dedup.
  // dRowStart[r] and dRowStart[r+1] delimit the row-r slice.
  {
    const int threads = 128;
    const int blocks  = (int)((rows + threads - 1) / threads);
    ccsPerRowCoalesceKernel<X><<<blocks, threads, 0, stream>>>(
        dGroupedCol, dGroupedVal, dRowStart, dCoalCount, rows);
  }

  // Phase 5 — inclusive scan of coalCount → csrRowPtr[1..rows].
  // csrRowPtr[0] is already 0 (from the cudaMemsetAsync at the top).
  // dBlockSums reuse is safe — same size as Phase 2 (both scans over 'rows' elems).
  portableInclusiveScan(dCoalCount, dRowPtr + 1, dBlockSums, (int)rows, stream);
  // After this: csrRowPtr[rows] == coalescedNnz (asserted by shape-fn sizing).

  // Phase 6 — write the deduped, sorted per-row data into the final CSR output.
  {
    const int threads = 128;
    const int blocks  = (int)((rows + threads - 1) / threads);
    ccsWriteCsrKernel<X><<<blocks, threads, 0, stream>>>(
        dGroupedCol, dGroupedVal, dRowStart, dRowPtr, dColIdx, dCsrVal, rows);
  }

  // Return all temporaries to the pool (async-safe on the same stream).
  pool.free(rcRaw, devId, stream);
  pool.free(rfRaw, devId, stream);
  pool.free(rsRaw, devId, stream);
  pool.free(gcRaw, devId, stream);
  pool.free(gvRaw, devId, stream);
  pool.free(ccRaw, devId, stream);
  pool.free(bsRaw, devId, stream);
}

// ---------------------------------------------------------------------------
// cooToCsr — public CUDA entry point
// ---------------------------------------------------------------------------
void cooToCsr(sd::LaunchContext* context, NDArray* cooIndices, NDArray* cooValues,
              NDArray* csrValues, NDArray* colIdx, NDArray* rowPtr, sd::LongType rows) {
  NDArray::prepareSpecialUse({csrValues, colIdx, rowPtr}, {cooIndices, cooValues});

  BUILD_DOUBLE_SELECTOR(cooValues->dataType(), cooIndices->dataType(), cooToCsrCuda_,
                        (context, cooIndices, cooValues, csrValues, colIdx, rowPtr, rows),
                        SD_FLOAT_TYPES, SD_INDEXING_TYPES);

  NDArray::registerSpecialUse({csrValues, colIdx, rowPtr}, {cooIndices, cooValues});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd

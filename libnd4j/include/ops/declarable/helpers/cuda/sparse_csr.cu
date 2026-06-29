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
#include <ops/declarable/helpers/sparse_csr.h>
#include <system/op_boilerplate.h>

namespace sd {
namespace ops {
namespace helpers {

// ---------------------------------------------------------------------------
// csr_to_dense — CUDA
//
// Algorithm (all on device, zero host compute):
//   1. cudaMemsetAsync zeros the dense output buffer.
//   2. SCATTER kernel: one thread per nonzero entry e in [0, nnz).
//      - Binary-search rowPtr[0..rows] to find row i (rowPtr[i] <= e < rowPtr[i+1]).
//      - Write: output[i*stride0 + colIdx[e]*stride1] = values[e].
//   Strides are passed as kernel args; output need not be contiguous.
//   All reads from specialBuffer() (device); no host loop, no sync/tick.
// ---------------------------------------------------------------------------

template <typename X, typename I>
static SD_KERNEL void csrToDenseKernel(const X* __restrict__ values,
                                       const I* __restrict__ colIdx,
                                       const I* __restrict__ rowPtr,
                                       X* __restrict__ output,
                                       LongType rows, LongType nnz,
                                       LongType stride0, LongType stride1) {
  const LongType e = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (e >= nnz) return;

  // Binary search: find largest i in [0, rows-1] s.t. rowPtr[i] <= e.
  // Invariant: rowPtr[0]=0 <= e < nnz = rowPtr[rows], so answer always exists.
  LongType lo = 0, hi = rows - 1;
  while (lo < hi) {
    const LongType mid = (lo + hi + 1) >> 1;  // upper-mid avoids infinite loop
    if (static_cast<LongType>(rowPtr[mid]) <= e)
      lo = mid;
    else
      hi = mid - 1;
  }
  // lo is the row that owns nonzero e.
  const LongType c = static_cast<LongType>(colIdx[e]);
  output[lo * stride0 + c * stride1] = values[e];
}

template <typename X, typename I>
static void csrToDenseCuda_(NDArray& values, NDArray& colIdx, NDArray& rowPtr, NDArray& output) {
  const LongType rows = output.sizeAt(0);
  const LongType nnz  = values.lengthOf();

  const X* vBuf  = reinterpret_cast<const X*>(values.specialBuffer());
  const I* ciBuf = reinterpret_cast<const I*>(colIdx.specialBuffer());
  const I* rpBuf = reinterpret_cast<const I*>(rowPtr.specialBuffer());
  X*       oBuf  = reinterpret_cast<X*>(output.specialBuffer());

  const auto*    strides = output.stridesOf();
  const LongType stride0 = strides[0];
  const LongType stride1 = strides[1];

  const cudaStream_t stream = *output.getContext()->getCudaStream();

  // Zero the entire dense output buffer on device (async mem op, not an allocation).
  cudaMemsetAsync(oBuf, 0, output.getDataBuffer()->getLenInBytes(), stream);

  if (nnz == 0) return;

  const int blockSize = 256;
  const int gridSize  = static_cast<int>((nnz + blockSize - 1) / blockSize);
  csrToDenseKernel<X, I><<<gridSize, blockSize, 0, stream>>>(
      vBuf, ciBuf, rpBuf, oBuf, rows, nnz, stride0, stride1);
}

void csr_to_dense(NDArray& values, NDArray& colIdx, NDArray& rowPtr, NDArray& output) {
  NDArray::prepareSpecialUse({&output}, {&values, &colIdx, &rowPtr});

  BUILD_DOUBLE_SELECTOR(values.dataType(), colIdx.dataType(), csrToDenseCuda_,
                        (values, colIdx, rowPtr, output),
                        SD_FLOAT_TYPES, SD_INDEXING_TYPES);

  NDArray::registerSpecialUse({&output}, {&values, &colIdx, &rowPtr});
}

// ---------------------------------------------------------------------------
// dense_to_csr — CUDA (all compute on device)
//
// Algorithm (4 phases, all on context stream, no host loops, no temp allocs):
//
//   1. cudaMemsetAsync zeros rowPtr[0..rows] on device.
//
//   2. COUNT kernel (one GPU thread per row r):
//      Walk columns 0..cols-1 in ascending order; for each entry where
//      |dense[r*s0 + c*s1]| > threshold, increment a register-local counter.
//      Write counter to rowPtr[r+1] directly (no race: each thread owns a
//      unique r+1 slot).
//
//   3. csrPrefixSumKernel prefix-sums rowPtr[1..rows] in place.
//      After this rowPtr[r] is the start offset of row r's entries.
//
//   4. SCATTER kernel (one GPU thread per row r):
//      Walk columns 0..cols-1 in ascending order; for each qualifying entry,
//      write values[rowPtr[r] + localCount] = val and
//            colIdx[rowPtr[r] + localCount] = c,
//      then increment the register-local counter.
//      colIdx is naturally column-sorted per row (ascending walk order).
//      No temp buffer or atomic needed: one thread owns each row's write range.
//
// Nonzero condition: |x| > threshold — matches DECLARE_SHAPE_FN's
// CountNonZero (threshold==0) / MatchCondition mode=7 (threshold>0).
// All reads from specialBuffer() (device). No host loop, no sync/tick,
// no raw cudaMalloc/cudaFree, no std::vector.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// csrPrefixSumKernel — portable in-place inclusive prefix sum (no Thrust).
//
// Computes an inclusive prefix sum over data[0..n-1] in place.
// Called as: csrPrefixSumKernel<I><<<1, bsz, bsz*sizeof(I), stream>>>(ptr, n)
//
// Algorithm: Hillis-Steele within shared memory, chunked so that any n is
// handled correctly regardless of how large it is.  Each iteration of the
// outer for-loop processes a contiguous chunk of `bsz` elements:
//   1. Load chunk into shared memory (pad tail chunk with zeros).
//   2. Run Hillis-Steele inclusive scan in shared memory.
//   3. Write results back to global memory, offset by the running `carry`
//      (= prefix sum of all preceding chunks).
//   4. All threads update their identical copy of `carry` by adding
//      the chunk-local total (sdata[chunk_sz-1]), which they read before
//      the write-back sync.
//
// Size assumptions:
//   - Must be launched with gridDim.x == 1 (single block).
//   - Dynamic shared memory must be blockDim.x * sizeof(I) bytes.
//   - blockDim.x must be a power of two (Hillis-Steele stride sequence).
//   - Handles any n; per-chunk shared mem ≤ 1024 * 8 = 8 KB (well within 48 KB).
// ---------------------------------------------------------------------------
template <typename I>
static SD_KERNEL void csrPrefixSumKernel(I* __restrict__ data, LongType n) {
  extern __shared__ char smem[];
  I*        sdata = reinterpret_cast<I*>(smem);
  const int bsz   = blockDim.x;
  const int tid   = threadIdx.x;
  I         carry = static_cast<I>(0);  // identical across all threads each iteration

  for (LongType base = 0; base < n; base += bsz) {
    const LongType rem      = n - base;
    const int      chunk_sz = static_cast<int>(rem < static_cast<LongType>(bsz) ? rem : static_cast<LongType>(bsz));

    // Load chunk; pad short tail to zero so scan arithmetic is always clean.
    sdata[tid] = (tid < chunk_sz) ? data[base + tid] : static_cast<I>(0);
    __syncthreads();

    // Hillis-Steele inclusive scan over sdata[0..bsz-1].
    for (int stride = 1; stride < bsz; stride <<= 1) {
      const I val = (tid >= stride) ? sdata[tid - stride] : static_cast<I>(0);
      __syncthreads();
      sdata[tid] += val;
      __syncthreads();
    }

    // Read chunk total before write-back (sdata still contains scan result).
    const I chunkSum = sdata[chunk_sz - 1];

    // Write back: data[base+tid] = carry + chunk-local prefix sum.
    if (tid < chunk_sz) data[base + tid] = carry + sdata[tid];
    __syncthreads();  // ensure global writes complete before next chunk load

    // Advance carry uniformly (every thread computes the same value).
    carry += chunkSum;
  }
}

template <typename X, typename I>
static SD_KERNEL void csrCountKernel(const X* __restrict__ denseData,
                                     I* __restrict__ rowPtr,
                                     LongType rows, LongType cols,
                                     LongType stride0, LongType stride1,
                                     X thr) {
  const LongType r = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (r >= rows) return;

  I cnt = static_cast<I>(0);
  for (LongType c = 0; c < cols; ++c) {
    const X val    = denseData[r * stride0 + c * stride1];
    const X absVal = val < static_cast<X>(0) ? -val : val;
    if (absVal > thr) ++cnt;
  }
  // Direct write — no race: each thread r writes to its own rowPtr[r+1] slot.
  rowPtr[r + 1] = cnt;
}

template <typename X, typename I>
static SD_KERNEL void csrScatterKernel(const X* __restrict__ denseData,
                                       X* __restrict__ outValues,
                                       I* __restrict__ outColIdx,
                                       const I* __restrict__ rowPtr,
                                       LongType rows, LongType cols,
                                       LongType stride0, LongType stride1,
                                       X thr) {
  const LongType r = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (r >= rows) return;

  // rowPtr[r] (after prefix sum) is this row's start offset in outputs.
  LongType offset = static_cast<LongType>(rowPtr[r]);
  for (LongType c = 0; c < cols; ++c) {
    const X val    = denseData[r * stride0 + c * stride1];
    const X absVal = val < static_cast<X>(0) ? -val : val;
    if (absVal > thr) {
      outValues[offset] = val;
      outColIdx[offset] = static_cast<I>(c);
      ++offset;
    }
  }
}

template <typename X, typename I>
static void denseToCsrCuda_(NDArray& input, NDArray& values, NDArray& colIdx, NDArray& rowPtr,
                             double threshold) {
  const LongType rows = input.sizeAt(0);
  const LongType cols = input.sizeAt(1);
  const X        thr  = static_cast<X>(threshold);

  const X* dInput  = reinterpret_cast<const X*>(input.specialBuffer());
  X*       dValues = reinterpret_cast<X*>(values.specialBuffer());
  I*       dColIdx = reinterpret_cast<I*>(colIdx.specialBuffer());
  I*       dRowPtr = reinterpret_cast<I*>(rowPtr.specialBuffer());

  const auto*    strides = input.stridesOf();
  const LongType stride0 = strides[0];
  const LongType stride1 = strides[1];

  const cudaStream_t stream = *input.getContext()->getCudaStream();

  // Phase 1: zero rowPtr[0..rows] on device.
  cudaMemsetAsync(dRowPtr, 0, static_cast<size_t>(rows + 1) * sizeof(I), stream);

  const int blockSize = 256;
  const int gridSize  = static_cast<int>((rows + blockSize - 1) / blockSize);

  // Phase 2: count nonzeros per row into rowPtr[r+1].
  csrCountKernel<X, I><<<gridSize, blockSize, 0, stream>>>(
      dInput, dRowPtr, rows, cols, stride0, stride1, thr);

  // Phase 3: portable in-place inclusive prefix-sum over rowPtr[1..rows].
  // rowPtr[0] stays 0 (from the cudaMemsetAsync in Phase 1); the kernel
  // receives (dRowPtr+1, rows) so it scans exactly the per-row count slots
  // and writes rowPtr[r] = number of nonzeros in rows 0..r-1 — the standard
  // CSR row-pointer layout consumed by csrScatterKernel.
  //
  // Launch constraints: single block (gridDim.x == 1), blockDim.x is the
  // smallest power-of-two >= min(rows, 1024); shared mem = blockDim.x * sizeof(I).
  if (rows > 0) {
    // Round up to next power of two, capped at 1024.
    int scanBlock = 1;
    while (scanBlock < static_cast<int>(rows) && scanBlock < 1024) scanBlock <<= 1;
    const size_t smemScan = static_cast<size_t>(scanBlock) * sizeof(I);
    csrPrefixSumKernel<I><<<1, scanBlock, smemScan, stream>>>(dRowPtr + 1, rows);
  }

  // Phase 4: scatter values and colIdx (skip if no nonzeros).
  if (values.lengthOf() > 0) {
    csrScatterKernel<X, I><<<gridSize, blockSize, 0, stream>>>(
        dInput, dValues, dColIdx, dRowPtr, rows, cols, stride0, stride1, thr);
  }
}

void dense_to_csr(NDArray& input, NDArray& values, NDArray& colIdx, NDArray& rowPtr, double threshold) {
  NDArray::prepareSpecialUse({&values, &colIdx, &rowPtr}, {&input});

  BUILD_DOUBLE_SELECTOR(input.dataType(), colIdx.dataType(), denseToCsrCuda_,
                        (input, values, colIdx, rowPtr, threshold),
                        SD_FLOAT_TYPES, SD_INDEXING_TYPES);

  NDArray::registerSpecialUse({&values, &colIdx, &rowPtr}, {&input});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd

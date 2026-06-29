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
// CUDA implementations for CSC (compressed-sparse-column) ops.
//
// csr_to_csc:  uses cusparseCsr2cscEx2 (cuSPARSE required; throws without it).
//              half/bfloat16 values are up-cast to float32, converted, then
//              cast back — no host round-trip for the compute path.
//              No raw cudaMalloc; workspace comes from CudaMemoryPool.
//
// csc_to_dense: zeros output via cudaMemsetAsync, then a __global__ scatter
//               kernel (one thread per nonzero) writes stride-aware into dense.
//
// dense_to_csc: fully on-GPU construction —
//               Phase 1: count-per-column kernel → colCounts[cols]
//               Phase 2: portable single-block Hillis-Steele prefix-sum kernel
//                        → cscColPtr[1..cols], set [0]=0
//               Phase 3: scatter kernel (one thread per column, sequential row
//                        scan) writes cscValues + cscRowIdx stride-aware.
//               Temporaries from CudaMemoryPool only; no host buffers.
//

#include <cuda_runtime.h>
#include <memory/cuda/CudaMemoryPool.h>
#include <ops/declarable/helpers/sparse_csc.h>
#include <system/op_boilerplate.h>
#include <types/bfloat16.h>
#include <types/float16.h>

#include <memory>
#include <type_traits>
#include <vector>

#if defined(HAVE_CUSPARSE)
#include <cusparse_v2.h>
#endif

namespace sd {
namespace ops {
namespace helpers {

// ═══════════════════════════════════════════════════════════════════════════
// Section A: csr_to_csc — cuSPARSE cusparseCsr2cscEx2
// ═══════════════════════════════════════════════════════════════════════════

#if defined(HAVE_CUSPARSE)

// ── A.1  Error helper ────────────────────────────────────────────────────────

static SD_INLINE void checkCusparseCsc(cusparseStatus_t st, const char* where) {
  if (st != CUSPARSE_STATUS_SUCCESS) {
    std::string msg = std::string("csr_to_csc cuSPARSE error at [") + where +
                      "] status=" + std::to_string(static_cast<int>(st));
    THROW_EXCEPTION(msg.c_str());
  }
}

// ── A.2  Value-type classification ───────────────────────────────────────────

// Types that cuSPARSE Csr2csc supports natively (float and double).
template <typename X> struct Csc2cNative { static constexpr bool value = false; };
template <> struct Csc2cNative<float>    { static constexpr bool value = true;  };
template <> struct Csc2cNative<double>   { static constexpr bool value = true;  };

// cudaDataType for float / double native path.
template <typename T> static SD_INLINE cudaDataType cscCudaValType();
template <> SD_INLINE cudaDataType cscCudaValType<float>()  { return CUDA_R_32F; }
template <> SD_INLINE cudaDataType cscCudaValType<double>() { return CUDA_R_64F; }

// ── A.3  Core cusparseCsr2cscEx2 execution ──────────────────────────────────
//
// Preconditions:
//   • handle stream already set.
//   • All index buffers are int32_t device pointers (cuSPARSE requirement).
//   • cscValPtr / cscColOff32 / cscRowInd32 are the final output device buffers.

static void runCsr2csc(
    cusparseHandle_t  handle,
    cudaStream_t      stream,
    int               devId,
    // CSR inputs (device, int32)
    const void*       csrValPtr,
    const int32_t*    csrRowOff32,
    const int32_t*    csrColInd32,
    int               nnz,
    int               rows,
    int               cols,
    cudaDataType      valType,
    // CSC outputs (device, int32)
    void*             cscValPtr,
    int32_t*          cscColOff32,
    int32_t*          cscRowInd32) {

  // ── 1. Query workspace size ─────────────────────────────────────────────
  size_t bufSz = 0;
  checkCusparseCsc(
      cusparseCsr2cscEx2_bufferSize(
          handle, rows, cols, nnz,
          csrValPtr, csrRowOff32, csrColInd32,
          cscValPtr, cscColOff32, cscRowInd32,
          valType,
          CUSPARSE_ACTION_NUMERIC,
          CUSPARSE_INDEX_BASE_ZERO,
          CUSPARSE_CSR2CSC_ALG_DEFAULT,
          &bufSz),
      "Csr2cscEx2_bufferSize");

  // ── 2. Allocate workspace from pool (no raw cudaMalloc) ─────────────────
  void* buf = nullptr;
  if (bufSz > 0) {
    buf = memory::CudaMemoryPool::getInstance().allocate(bufSz, devId, stream);
    if (!buf) THROW_EXCEPTION("csr_to_csc: CudaMemoryPool failed to allocate workspace");
  }

  // ── 3. Execute ──────────────────────────────────────────────────────────
  checkCusparseCsc(
      cusparseCsr2cscEx2(
          handle, rows, cols, nnz,
          csrValPtr, csrRowOff32, csrColInd32,
          cscValPtr, cscColOff32, cscRowInd32,
          valType,
          CUSPARSE_ACTION_NUMERIC,
          CUSPARSE_INDEX_BASE_ZERO,
          CUSPARSE_CSR2CSC_ALG_DEFAULT,
          buf),
      "Csr2cscEx2");

  // ── 4. Release workspace ─────────────────────────────────────────────────
  if (buf) memory::CudaMemoryPool::getInstance().free(buf, devId, stream);
}

// ── A.4  Template dispatch: X (value type), I (CSR input integer type) ───────

template <typename X, typename I>
static void csrToCscCuda_(NDArray& csrValues, NDArray& csrColIdx, NDArray& csrRowPtr,
                           NDArray& cscValues, NDArray& cscRowIdx, NDArray& cscColPtr,
                           LongType rows, LongType cols) {
  auto* ctx    = cscValues.getContext();
  auto  stream = ctx->getCudaStream();   // cudaStream_t*
  auto* handle = reinterpret_cast<cusparseHandle_t*>(ctx->getCusparseHandle());

  checkCusparseCsc(cusparseSetStream(*handle, *stream), "cusparseSetStream");

  const int devId = ctx->getDeviceID();
  const int nnz   = static_cast<int>(csrValues.lengthOf());

  // ── Step 1: Resolve INT32 CSR index pointers ─────────────────────────────
  //
  // cusparseCsr2cscEx2 requires 32-bit index arrays.  If I == INT64 we cast
  // the input index arrays on-device; no host round-trip.

  std::unique_ptr<NDArray> csrColIdxI32, csrRowPtrI32;
  const int32_t*           csrColInd32;
  const int32_t*           csrRowOff32;

  if constexpr (sizeof(I) == 4) {
    csrColInd32 = reinterpret_cast<const int32_t*>(csrColIdx.specialBuffer());
    csrRowOff32 = reinterpret_cast<const int32_t*>(csrRowPtr.specialBuffer());
  } else {
    // INT64 → INT32 on-device; include in prepareSpecialUse read-set so the
    // device buffer is current before cuSPARSE reads specialBuffer().
    csrColIdxI32.reset(csrColIdx.cast(sd::DataType::INT32));
    csrRowPtrI32.reset(csrRowPtr.cast(sd::DataType::INT32));
    NDArray::prepareSpecialUse({}, {csrColIdxI32.get(), csrRowPtrI32.get()});
    csrColInd32 = reinterpret_cast<const int32_t*>(csrColIdxI32->specialBuffer());
    csrRowOff32 = reinterpret_cast<const int32_t*>(csrRowPtrI32->specialBuffer());
  }

  // Output index arrays are always INT32 per op contract.
  int32_t* cscColOff32 = reinterpret_cast<int32_t*>(cscColPtr.specialBuffer());
  int32_t* cscRowInd32 = reinterpret_cast<int32_t*>(cscRowIdx.specialBuffer());

  // ── Step 2: Native float/double or up-cast half/bf16 path ────────────────

  if constexpr (Csc2cNative<X>::value) {
    // ── 2a. Native path (float or double) ─────────────────────────────────
    runCsr2csc(*handle, *stream, devId,
               csrValues.specialBuffer(), csrRowOff32, csrColInd32,
               nnz, static_cast<int>(rows), static_cast<int>(cols),
               cscCudaValType<X>(),
               cscValues.specialBuffer(), cscColOff32, cscRowInd32);

  } else {
    // ── 2b. Up-cast path: float16 or bfloat16 ─────────────────────────────
    //
    // cusparseCsr2cscEx2 does not support half or bfloat16 (as of CUDA 12.x).
    // Strategy:
    //   1. Cast csrValues (X) → FLOAT32 on-device (CUDA elementwise op).
    //   2. Allocate a temporary FLOAT32 csc-values buffer.
    //   3. Run cusparseCsr2cscEx2 entirely in FLOAT32.
    //   4. Assign the FLOAT32 result back to X into cscValues.
    // No host round-trip at any step.

    std::unique_ptr<NDArray> csrValF32(csrValues.cast(sd::DataType::FLOAT32));

    std::unique_ptr<NDArray> cscValF32(
        NDArrayFactory::create('c', std::vector<sd::LongType>{static_cast<sd::LongType>(nnz)},
                               sd::DataType::FLOAT32, ctx));

    // csrValF32 is a cast output (device-written); cscValF32 will be written
    // by runCsr2csc.  prepareSpecialUse marks csrValF32 device-current for
    // reading and prepares cscValF32 to receive a device write.
    NDArray::prepareSpecialUse({cscValF32.get()}, {csrValF32.get()});

    runCsr2csc(*handle, *stream, devId,
               csrValF32->specialBuffer(), csrRowOff32, csrColInd32,
               nnz, static_cast<int>(rows), static_cast<int>(cols),
               CUDA_R_32F,
               cscValF32->specialBuffer(), cscColOff32, cscRowInd32);

    NDArray::registerSpecialUse({cscValF32.get()}, {csrValF32.get()});

    // Assign FLOAT32 csc temp back into the X-typed output.
    cscValues.assign(cscValF32.get());
  }
}

#endif  // HAVE_CUSPARSE

// ── A.5  Public csr_to_csc entry point ──────────────────────────────────────

void csr_to_csc(NDArray& csrValues, NDArray& csrColIdx, NDArray& csrRowPtr,
                NDArray& cscValues, NDArray& cscRowIdx, NDArray& cscColPtr,
                LongType rows, LongType cols) {
  NDArray::prepareSpecialUse({&cscValues, &cscRowIdx, &cscColPtr},
                             {&csrValues, &csrColIdx, &csrRowPtr});

#if defined(HAVE_CUSPARSE)
  BUILD_DOUBLE_SELECTOR(csrValues.dataType(), csrColIdx.dataType(), csrToCscCuda_,
                        (csrValues, csrColIdx, csrRowPtr,
                         cscValues, cscRowIdx, cscColPtr, rows, cols),
                        SD_FLOAT_TYPES, SD_INDEXING_TYPES);
#else
  THROW_EXCEPTION(
      "csr_to_csc CUDA: cuSPARSE is required but was not compiled in "
      "(HAVE_CUSPARSE not defined). Rebuild with -Dlibnd4j.cusparse=ON "
      "or use the CPU backend.");
#endif

  NDArray::registerSpecialUse({&cscValues, &cscRowIdx, &cscColPtr},
                              {&csrValues, &csrColIdx, &csrRowPtr});
}


// ═══════════════════════════════════════════════════════════════════════════
// Section B: csc_to_dense — CUDA kernel
//
// Steps:
//   1. Zero the output via cudaMemsetAsync (output is contiguous, freshly
//      allocated by the op).
//   2. Launch cscToDenseKernel: one thread per nonzero element.  Each thread
//      binary-searches cscColPtr to find its column j, reads its row i from
//      cscRowIdx, then writes stride-aware: output[i*outS0 + j*outS1].
//
// Thread mapping: one thread per nonzero (nnz total), avoids per-column
// serialisation for wide-but-sparse matrices.
// ═══════════════════════════════════════════════════════════════════════════

template <typename X, typename I>
static SD_KERNEL void cscToDenseKernel(const X* cscValues, const I* cscRowIdx,
                                        const I* cscColPtr, X* output,
                                        LongType rows, LongType cols,
                                        LongType outS0, LongType outS1,
                                        LongType nnz) {
  // One thread per nonzero entry.
  const LongType e = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (e >= nnz) return;

  // Binary-search cscColPtr[0..cols] to find column j such that
  // cscColPtr[j] <= e < cscColPtr[j+1].
  I lo = 0, hi = static_cast<I>(cols);
  while (lo < hi) {
    I mid = lo + (hi - lo) / 2;
    if (cscColPtr[mid + 1] <= static_cast<I>(e))
      lo = mid + 1;
    else
      hi = mid;
  }
  const LongType j = static_cast<LongType>(lo);
  const LongType i = static_cast<LongType>(cscRowIdx[e]);

  // Stride-aware write into dense output.
  output[i * outS0 + j * outS1] = cscValues[e];
}

template <typename X, typename I>
static void cscToDenseCuda_(NDArray& cscValues, NDArray& cscRowIdx, NDArray& cscColPtr,
                             NDArray& output) {
  const LongType rows = output.sizeAt(0);
  const LongType cols = output.sizeAt(1);
  const LongType nnz  = cscValues.lengthOf();

  const auto* outStrides = output.stridesOf();
  const LongType outS0   = outStrides[0];
  const LongType outS1   = outStrides[1];

  auto  stream = *output.getContext()->getCudaStream();

  // Zero the output device buffer before scattering nonzeros.
  cudaMemsetAsync(output.specialBuffer(), 0,
                  static_cast<size_t>(output.lengthOf()) * sizeof(X), stream);

  if (nnz == 0) return;

  const X* vBuf  = reinterpret_cast<const X*>(cscValues.specialBuffer());
  const I* riBuf = reinterpret_cast<const I*>(cscRowIdx.specialBuffer());
  const I* cpBuf = reinterpret_cast<const I*>(cscColPtr.specialBuffer());
  X*       oBuf  = reinterpret_cast<X*>(output.specialBuffer());

  const int blockSize = 256;
  const int gridSize  = static_cast<int>((nnz + blockSize - 1) / blockSize);

  cscToDenseKernel<X, I><<<gridSize, blockSize, 0, stream>>>(
      vBuf, riBuf, cpBuf, oBuf, rows, cols, outS0, outS1, nnz);
}

void csc_to_dense(NDArray& cscValues, NDArray& cscRowIdx, NDArray& cscColPtr,
                  NDArray& output) {
  NDArray::prepareSpecialUse({&output}, {&cscValues, &cscRowIdx, &cscColPtr});

  BUILD_DOUBLE_SELECTOR(cscValues.dataType(), cscRowIdx.dataType(), cscToDenseCuda_,
                        (cscValues, cscRowIdx, cscColPtr, output),
                        SD_FLOAT_TYPES, SD_INDEXING_TYPES);

  NDArray::registerSpecialUse({&output}, {&cscValues, &cscRowIdx, &cscColPtr});
}


// ═══════════════════════════════════════════════════════════════════════════
// Section C: dense_to_csc — fully GPU construction
//
// Three-phase pipeline, all on the CUDA stream; no host compute:
//
//   Phase 1 – countPerColKernel:
//     One thread per column j.  Scans rows 0..rows-1 stride-aware via
//     input.stridesOf().  Counts |val| > threshold → colCounts[j].
//     Temporary colCounts allocated via CudaMemoryPool.
//
//   Phase 2 – prefix sum:
//     cscColPtr[0] = 0 via cudaMemsetAsync.
//     inclusivePrefixSumKernel(colCounts[0..cols-1]) → cscColPtr[1..cols].
//     Portable single-block Hillis-Steele scan; processes cols in chunks of
//     kScanBlock with a shared carry for arbitrarily large col counts.
//     This gives cscColPtr[c] = start offset of column c for c in [0,cols].
//
//   Phase 3 – scatterCscKernel:
//     One thread per column j.  Re-scans rows 0..rows-1 stride-aware;
//     starts writing at cscColPtr[j] and increments a local counter —
//     no atomics, no contention (each column is owned by exactly one thread).
// ═══════════════════════════════════════════════════════════════════════════

// Phase 1: count nonzeros per column.
template <typename X, typename I>
static SD_KERNEL void countPerColKernel(const X* input,
                                         LongType rows, LongType cols,
                                         LongType s0, LongType s1,
                                         X threshold, I* colCounts) {
  const LongType c = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (c >= cols) return;
  I cnt = 0;
  for (LongType r = 0; r < rows; ++r) {
    X val = input[r * s0 + c * s1];
    if (val < static_cast<X>(0)) val = -val;
    if (val > threshold) ++cnt;
  }
  colCounts[c] = cnt;
}

// Phase 3: scatter values and row indices column by column.
template <typename X, typename I>
static SD_KERNEL void scatterCscKernel(const X* input,
                                        LongType rows, LongType cols,
                                        LongType s0, LongType s1,
                                        X threshold,
                                        const I* cscColPtr,
                                        X* cscValues, I* cscRowIdx) {
  const LongType c = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (c >= cols) return;
  I pos = cscColPtr[c];  // start of this column's slot in cscValues/cscRowIdx
  for (LongType r = 0; r < rows; ++r) {
    X val    = input[r * s0 + c * s1];
    X absVal = val < static_cast<X>(0) ? -val : val;
    if (absVal > threshold) {
      cscValues[pos] = val;
      cscRowIdx[pos] = static_cast<I>(r);
      ++pos;
    }
  }
}

// ── Phase 2 helper: portable inclusive prefix-sum (no Thrust) ────────────────
//
// Single-block Hillis-Steele scan, shared-memory width kScanBlock.
// Handles arbitrarily large n by processing it in chunks of kScanBlock
// with an inter-chunk carry cell in shared memory.
//
// Launch as: <<<1, kScanBlock, (kScanBlock+1)*sizeof(I), stream>>>
//
// Semantics: dst[i] = src[0] + src[1] + ... + src[i]  for i in [0, n).
// Used as: src=colCounts[0..cols-1], dst=cscColPtr[1..cols]
// so that cscColPtr[c] (=dst[c-1]) = start offset for column c.  ✓
// cscColPtr[0] = 0 is set separately by cudaMemsetAsync before this kernel.

static constexpr int kScanBlock = 512;

template <typename I>
static SD_KERNEL void inclusivePrefixSumKernel(const I* __restrict__ src,
                                                I* __restrict__ dst,
                                                int n) {
  // smem layout: [0 .. B-1] scan workspace, [B] inter-chunk carry.
  extern __shared__ char smem_raw[];
  I* smem  = reinterpret_cast<I*>(smem_raw);
  I* carry = smem + blockDim.x;  // single carry cell

  const int B = static_cast<int>(blockDim.x);

  if (threadIdx.x == 0) carry[0] = static_cast<I>(0);
  __syncthreads();

  for (int base = 0; base < n; base += B) {
    const int tid = threadIdx.x;
    const int gid = base + tid;

    // Load one chunk; zero-pad threads past the end of src.
    smem[tid] = (gid < n) ? src[gid] : static_cast<I>(0);
    __syncthreads();

    // Hillis-Steele inclusive scan over smem[0..B-1].
    for (int stride = 1; stride < B; stride <<= 1) {
      I val = (tid >= stride) ? smem[tid - stride] : static_cast<I>(0);
      __syncthreads();
      smem[tid] += val;
      __syncthreads();
    }

    // Write chunk result, adjusted by the inter-chunk carry.
    if (gid < n) dst[gid] = smem[tid] + carry[0];
    __syncthreads();

    // Update carry: add this chunk's inclusive total.
    // smem[last_valid] holds the local inclusive sum of the last element
    // in this chunk (without the carry), so carry += smem[last_valid].
    if (threadIdx.x == 0) {
      const int last = (base + B <= n) ? (B - 1) : (n - base - 1);
      carry[0] += smem[last];
    }
    __syncthreads();  // ensure carry[0] is visible to all threads in next iter
  }
}

template <typename X, typename I>
static void denseToCscCuda_(NDArray& input, NDArray& cscValues, NDArray& cscRowIdx,
                             NDArray& cscColPtr, double threshold) {
  const LongType rows = input.sizeAt(0);
  const LongType cols = input.sizeAt(1);
  const X        thr  = static_cast<X>(threshold);

  const auto* strides = input.stridesOf();
  const LongType s0   = strides[0];
  const LongType s1   = strides[1];

  auto* ctx    = input.getContext();
  auto  stream = *ctx->getCudaStream();
  const int devId = ctx->getDeviceID();

  const X* inBuf     = reinterpret_cast<const X*>(input.specialBuffer());
  X*       valBuf    = reinterpret_cast<X*>(cscValues.specialBuffer());
  I*       rowBuf    = reinterpret_cast<I*>(cscRowIdx.specialBuffer());
  I*       colPtrBuf = reinterpret_cast<I*>(cscColPtr.specialBuffer());

  // Allocate temp colCounts[cols] from pool — no raw cudaMalloc.
  const size_t colCountsBytes = static_cast<size_t>(cols) * sizeof(I);
  void* colCountsMem = memory::CudaMemoryPool::getInstance().allocate(colCountsBytes, devId, stream);
  if (!colCountsMem) THROW_EXCEPTION("dense_to_csc: CudaMemoryPool failed to allocate colCounts");
  I* colCounts = reinterpret_cast<I*>(colCountsMem);
  cudaMemsetAsync(colCounts, 0, colCountsBytes, stream);

  const int blockSize = 256;
  const int gridSize  = static_cast<int>((cols + blockSize - 1) / blockSize);

  // ── Phase 1: count nonzeros per column ────────────────────────────────────
  countPerColKernel<X, I><<<gridSize, blockSize, 0, stream>>>(
      inBuf, rows, cols, s0, s1, thr, colCounts);

  // ── Phase 2: build cscColPtr via portable inclusive prefix sum ───────────────
  //   cscColPtr[0]       = 0  (set by cudaMemsetAsync)
  //   cscColPtr[1..cols] = inclusive_scan(colCounts[0..cols-1])
  //   → cscColPtr[c] is the start offset for column c (c in 0..cols-1),
  //     and cscColPtr[cols] is the total nnz.
  //   Uses single-block Hillis-Steele with kScanBlock=512; for cols > 512 the
  //   kernel iterates over chunks and carries the running total via shared mem.
  cudaMemsetAsync(colPtrBuf, 0, sizeof(I), stream);
  {
    const size_t scanSmem = (static_cast<size_t>(kScanBlock) + 1) * sizeof(I);
    inclusivePrefixSumKernel<I><<<1, kScanBlock, scanSmem, stream>>>(
        colCounts, colPtrBuf + 1, static_cast<int>(cols));
  }

  // ── Phase 3: scatter values and row indices ────────────────────────────────
  scatterCscKernel<X, I><<<gridSize, blockSize, 0, stream>>>(
      inBuf, rows, cols, s0, s1, thr,
      colPtrBuf, valBuf, rowBuf);

  memory::CudaMemoryPool::getInstance().free(colCountsMem, devId, stream);
}

void dense_to_csc(NDArray& input, NDArray& cscValues, NDArray& cscRowIdx,
                  NDArray& cscColPtr, double threshold) {
  NDArray::prepareSpecialUse({&cscValues, &cscRowIdx, &cscColPtr}, {&input});

  BUILD_DOUBLE_SELECTOR(input.dataType(), cscRowIdx.dataType(), denseToCscCuda_,
                        (input, cscValues, cscRowIdx, cscColPtr, threshold),
                        SD_FLOAT_TYPES, SD_INDEXING_TYPES);

  NDArray::registerSpecialUse({&cscValues, &cscRowIdx, &cscColPtr}, {&input});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd

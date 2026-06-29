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
#include <memory/cuda/CudaMemoryPool.h>
#include <ops/declarable/helpers/sparse_blas.h>
#include <system/op_boilerplate.h>
#include <types/bfloat16.h>
#include <types/float16.h>
#include <math/templatemath.h>

#if defined(HAVE_CUSPARSE)
#include <cusparse_v2.h>
#endif

namespace sd {
namespace ops {
namespace helpers {

// ===========================================================================
// Type-mapping helpers (compile-time, host-callable)
// ===========================================================================

#if defined(HAVE_CUSPARSE)

// Map C++ floating type → CUDA data type constant used by cuSPARSE generic API
template <typename T>
static cudaDataType cusparseValueType();

template <> SD_INLINE cudaDataType cusparseValueType<float>()    { return CUDA_R_32F; }
template <> SD_INLINE cudaDataType cusparseValueType<double>()   { return CUDA_R_64F; }
template <> SD_INLINE cudaDataType cusparseValueType<float16>()  { return CUDA_R_16F; }
template <> SD_INLINE cudaDataType cusparseValueType<bfloat16>() { return CUDA_R_16BF; }

// Map C++ integer type → cuSPARSE index type (by size)
template <typename I>
static cusparseIndexType_t cusparseIdxType() {
  return (sizeof(I) == 4) ? CUSPARSE_INDEX_32I : CUSPARSE_INDEX_64I;
}

// Per-type cuSPARSE compute traits.
// float and double have a natural cuSPARSE compute type.
// float16/bfloat16 require CUDA_R_32F compute with float scalars — but cuSPARSE
// support for those type combinations varies across versions.  Mark them as
// unsupported here so the code falls through to the hand-written kernels.
template <typename X>
struct CusparseComputeTraits {
  static constexpr bool    supported   = false;
  static constexpr cudaDataType computeType = CUDA_R_32F;  // placeholder
  using scalar_t = float;
  static scalar_t one()  { return 1.0f; }
  static scalar_t zero() { return 0.0f; }
};
template <>
struct CusparseComputeTraits<float> {
  static constexpr bool    supported   = true;
  static constexpr cudaDataType computeType = CUDA_R_32F;
  using scalar_t = float;
  static scalar_t one()  { return 1.0f; }
  static scalar_t zero() { return 0.0f; }
};
template <>
struct CusparseComputeTraits<double> {
  static constexpr bool    supported   = true;
  static constexpr cudaDataType computeType = CUDA_R_64F;
  using scalar_t = double;
  static scalar_t one()  { return 1.0; }
  static scalar_t zero() { return 0.0; }
};

#endif  // HAVE_CUSPARSE

// ===========================================================================
// Fallback CUDA kernels (used when cuSPARSE is unavailable or for unsupported
// type combinations)
// ===========================================================================

// ---- SpMV kernels ----------------------------------------------------------

template <typename X, typename I>
static SD_KERNEL void csrSpMVKernel(const X* values, const I* colIdx, const I* rowPtr,
                                     const X* x, X* y,
                                     LongType rows, LongType xStride0, LongType yStride0) {
  const LongType r = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (r >= rows) return;
  const I start = rowPtr[r];
  const I end   = rowPtr[r + 1];
  X acc = static_cast<X>(0);
  for (I k = start; k < end; ++k) {
    acc += values[k] * x[static_cast<LongType>(colIdx[k]) * xStride0];
  }
  y[r * yStride0] = acc;
}

// Transpose SpMV: y[colIdx[k]] += values[k]*x[i]  — requires atomicAdd.
// Generic atomicAdd for double (CUDA < 6.0 lacked it; we define it defensively).
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
SD_DEVICE static double atomicAddDouble(double* addr, double val) {
  unsigned long long int* ptr = (unsigned long long int*)addr;
  unsigned long long int old  = *ptr, assumed;
  do {
    assumed = old;
    old = atomicCAS(ptr, assumed,
                    __double_as_longlong(val + __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
}
#endif

template <typename X, typename I>
static SD_KERNEL void csrSpMVTransposeKernel(const X* values, const I* colIdx, const I* rowPtr,
                                              const X* x, X* y,
                                              LongType rows, LongType xStride0, LongType yStride0) {
  const LongType r = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (r >= rows) return;
  const I start = rowPtr[r];
  const I end   = rowPtr[r + 1];
  const X xi    = x[r * xStride0];
  for (I k = start; k < end; ++k) {
    const LongType ci = static_cast<LongType>(colIdx[k]);
    sd::math::atomics::sd_atomicAdd(&y[ci * yStride0], static_cast<X>(values[k] * xi));
  }
}

// ---- SpMM kernels ----------------------------------------------------------

template <typename X, typename I>
static SD_KERNEL void csrSpMMKernel(const X* values, const I* colIdx, const I* rowPtr,
                                     const X* B, X* C,
                                     LongType rows, LongType n,
                                     LongType bStride0, LongType bStride1,
                                     LongType cStride0, LongType cStride1) {
  const LongType r = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x;
  const LongType j = static_cast<LongType>(blockIdx.y) * blockDim.y + threadIdx.y;
  if (r >= rows || j >= n) return;

  const I start = rowPtr[r];
  const I end   = rowPtr[r + 1];
  X acc = static_cast<X>(0);
  for (I k = start; k < end; ++k) {
    acc += values[k] * B[static_cast<LongType>(colIdx[k]) * bStride0 + j * bStride1];
  }
  C[r * cStride0 + j * cStride1] = acc;
}

template <typename X, typename I>
static SD_KERNEL void csrSpMMTransposeKernel(const X* values, const I* colIdx, const I* rowPtr,
                                              const X* B, X* C,
                                              LongType rows, LongType n,
                                              LongType bStride0, LongType bStride1,
                                              LongType cStride0, LongType cStride1) {
  const LongType r = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (r >= rows) return;
  const I start = rowPtr[r];
  const I end   = rowPtr[r + 1];
  for (I k = start; k < end; ++k) {
    const LongType ci = static_cast<LongType>(colIdx[k]);
    for (LongType j = 0; j < n; ++j) {
      sd::math::atomics::sd_atomicAdd(&C[ci * cStride0 + j * cStride1],
                static_cast<X>(values[k] * B[r * bStride0 + j * bStride1]));
    }
  }
}

// ---- SDDMM kernel ----------------------------------------------------------

template <typename X, typename I>
static SD_KERNEL void sddmmKernel(const I* rowPtr, const I* colIdx,
                                   const X* D1, const X* D2, X* outValues,
                                   LongType rows, LongType nnz, LongType p,
                                   LongType d1Stride0, LongType d1Stride1,
                                   LongType d2Stride0, LongType d2Stride1) {
  // One thread per CSR nonzero k
  const LongType k = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (k >= nnz) return;

  // Binary search for row i such that rowPtr[i] <= k < rowPtr[i+1]
  LongType lo = 0, hi = rows - 1, row = 0;
  while (lo <= hi) {
    LongType mid = (lo + hi) / 2;
    if (static_cast<LongType>(rowPtr[mid]) <= k && k < static_cast<LongType>(rowPtr[mid + 1])) {
      row = mid;
      break;
    } else if (static_cast<LongType>(rowPtr[mid]) > k) {
      hi = mid - 1;
    } else {
      lo = mid + 1;
    }
  }

  const LongType j = static_cast<LongType>(colIdx[k]);
  X acc = static_cast<X>(0);
  for (LongType l = 0; l < p; ++l) {
    acc += D1[row * d1Stride0 + l * d1Stride1] * D2[j * d2Stride0 + l * d2Stride1];
  }
  outValues[k] = acc;
}

// ===========================================================================
// csr_spmv  — CUDA dispatch
// ===========================================================================

template <typename X, typename I>
static void csrSpMVCuda_(NDArray& values, NDArray& colIdx, NDArray& rowPtr,
                          NDArray& x, NDArray& y,
                          sd::LongType rows, sd::LongType cols, int transposeA) {
  auto stream   = y.getContext()->getCudaStream();
  const int deviceId = y.getContext()->getDeviceID();

  const X* vBuf  = reinterpret_cast<const X*>(values.specialBuffer());
  const I* ciBuf = reinterpret_cast<const I*>(colIdx.specialBuffer());
  const I* rpBuf = reinterpret_cast<const I*>(rowPtr.specialBuffer());
  const X* xBuf  = reinterpret_cast<const X*>(x.specialBuffer());
  X*       yBuf  = reinterpret_cast<X*>(y.specialBuffer());

  const LongType xStride0 = x.stridesOf()[0];
  const LongType yStride0 = y.stridesOf()[0];

#if defined(HAVE_CUSPARSE)
  if constexpr (CusparseComputeTraits<X>::supported) {
    auto* handle = reinterpret_cast<cusparseHandle_t*>(y.getContext()->getCusparseHandle());
    cusparseSetStream(*handle, *stream);

    const LongType nnz     = values.lengthOf();
    const cudaDataType valType  = cusparseValueType<X>();
    const cusparseIndexType_t idxType = cusparseIdxType<I>();
    using CT = CusparseComputeTraits<X>;
    typename CT::scalar_t alpha = CT::one(), beta = CT::zero();

    // Create sparse CSR descriptor
    cusparseSpMatDescr_t spA;
    cusparseCreateCsr(&spA,
                      static_cast<int64_t>(rows), static_cast<int64_t>(cols),
                      static_cast<int64_t>(nnz),
                      const_cast<void*>(reinterpret_cast<const void*>(rpBuf)),
                      const_cast<void*>(reinterpret_cast<const void*>(ciBuf)),
                      const_cast<void*>(reinterpret_cast<const void*>(vBuf)),
                      idxType,   // rowPtr index type (same as colIdx — enforced by op)
                      idxType,   // colIdx index type
                      CUSPARSE_INDEX_BASE_ZERO,
                      valType);

    const LongType xLen = x.lengthOf();
    const LongType yLen = y.lengthOf();

    cusparseDnVecDescr_t dnX, dnY;
    cusparseCreateDnVec(&dnX, static_cast<int64_t>(xLen),
                        const_cast<void*>(reinterpret_cast<const void*>(xBuf)), valType);
    cusparseCreateDnVec(&dnY, static_cast<int64_t>(yLen),
                        reinterpret_cast<void*>(yBuf), valType);

    const cusparseOperation_t op = transposeA ? CUSPARSE_OPERATION_TRANSPOSE
                                              : CUSPARSE_OPERATION_NON_TRANSPOSE;

    size_t bufSize = 0;
    cusparseSpMV_bufferSize(*handle, op, &alpha, spA, dnX, &beta, dnY,
                            CT::computeType, CUSPARSE_SPMV_ALG_DEFAULT, &bufSize);

    void* workBuf = nullptr;
    if (bufSize > 0) {
      workBuf = memory::CudaMemoryPool::getInstance().allocate(bufSize, deviceId, *stream);
    }

    cusparseSpMV(*handle, op, &alpha, spA, dnX, &beta, dnY,
                 CT::computeType, CUSPARSE_SPMV_ALG_DEFAULT, workBuf);

    if (workBuf) memory::CudaMemoryPool::getInstance().free(workBuf, deviceId, *stream);
    cusparseDestroyDnVec(dnX);
    cusparseDestroyDnVec(dnY);
    cusparseDestroySpMat(spA);
    return;
  }
#endif

  // ---- Fallback: hand-written CUDA kernel ----
  {
    const int blockSize = 256;
    const int gridSize  = static_cast<int>((rows + blockSize - 1) / blockSize);

    if (transposeA == 0) {
      csrSpMVKernel<X, I><<<gridSize, blockSize, 0, *stream>>>(
          vBuf, ciBuf, rpBuf, xBuf, yBuf, rows, xStride0, yStride0);
    } else {
      csrSpMVTransposeKernel<X, I><<<gridSize, blockSize, 0, *stream>>>(
          vBuf, ciBuf, rpBuf, xBuf, yBuf, rows, xStride0, yStride0);
    }
  }
}

void csr_spmv(NDArray& values, NDArray& colIdx, NDArray& rowPtr,
              NDArray& x, NDArray& y,
              sd::LongType rows, sd::LongType cols, int transposeA) {
  NDArray::prepareSpecialUse({&y}, {&values, &colIdx, &rowPtr, &x});

  BUILD_DOUBLE_SELECTOR(values.dataType(), colIdx.dataType(), csrSpMVCuda_,
                        (values, colIdx, rowPtr, x, y, rows, cols, transposeA),
                        SD_FLOAT_TYPES, SD_INDEXING_TYPES);

  NDArray::registerSpecialUse({&y}, {&values, &colIdx, &rowPtr, &x});
}

// ===========================================================================
// csr_spmm  — CUDA dispatch
// ===========================================================================

template <typename X, typename I>
static void csrSpMMCuda_(NDArray& values, NDArray& colIdx, NDArray& rowPtr,
                          NDArray& B, NDArray& C,
                          sd::LongType rows, sd::LongType cols, int transposeA) {
  auto stream    = C.getContext()->getCudaStream();
  const int deviceId = C.getContext()->getDeviceID();

  const X* vBuf  = reinterpret_cast<const X*>(values.specialBuffer());
  const I* ciBuf = reinterpret_cast<const I*>(colIdx.specialBuffer());
  const I* rpBuf = reinterpret_cast<const I*>(rowPtr.specialBuffer());
  const X* bBuf  = reinterpret_cast<const X*>(B.specialBuffer());
  X*       cBuf  = reinterpret_cast<X*>(C.specialBuffer());

  const LongType n       = B.sizeAt(1);
  const LongType bStride0 = B.stridesOf()[0];
  const LongType bStride1 = B.stridesOf()[1];
  const LongType cStride0 = C.stridesOf()[0];
  const LongType cStride1 = C.stridesOf()[1];

#if defined(HAVE_CUSPARSE)
  if constexpr (CusparseComputeTraits<X>::supported) {
    auto* handle = reinterpret_cast<cusparseHandle_t*>(C.getContext()->getCusparseHandle());
    cusparseSetStream(*handle, *stream);

    const LongType nnz = values.lengthOf();
    const cudaDataType valType = cusparseValueType<X>();
    const cusparseIndexType_t idxType = cusparseIdxType<I>();
    using CT = CusparseComputeTraits<X>;
    typename CT::scalar_t alpha = CT::one(), beta = CT::zero();

    // B and C must be contiguous row-major for cuSPARSE DnMat.
    // If strides are non-standard, fall through to manual kernel.
    const bool bContiguous = (bStride1 == 1 && bStride0 == n);
    const bool cContiguous = (cStride1 == 1);

    if (bContiguous && cContiguous) {
      // Sparse A
      cusparseSpMatDescr_t spA;
      cusparseCreateCsr(&spA,
                        static_cast<int64_t>(rows), static_cast<int64_t>(cols),
                        static_cast<int64_t>(nnz),
                        const_cast<void*>(reinterpret_cast<const void*>(rpBuf)),
                        const_cast<void*>(reinterpret_cast<const void*>(ciBuf)),
                        const_cast<void*>(reinterpret_cast<const void*>(vBuf)),
                        idxType,   // rowPtr — same integer type as colIdx
                        idxType,   // colIdx
                        CUSPARSE_INDEX_BASE_ZERO,
                        valType);

      // Dense B: [Brows, n]
      const LongType Brows = B.sizeAt(0);
      cusparseDnMatDescr_t dnB, dnC;
      cusparseCreateDnMat(&dnB,
                          static_cast<int64_t>(Brows), static_cast<int64_t>(n),
                          static_cast<int64_t>(n),  // leading dim (row-major)
                          const_cast<void*>(reinterpret_cast<const void*>(bBuf)),
                          valType, CUSPARSE_ORDER_ROW);

      // Dense C: [Crows, n], leading dim = cStride0
      const LongType Crows = C.sizeAt(0);
      cusparseCreateDnMat(&dnC,
                          static_cast<int64_t>(Crows), static_cast<int64_t>(n),
                          static_cast<int64_t>(cStride0),
                          reinterpret_cast<void*>(cBuf),
                          valType, CUSPARSE_ORDER_ROW);

      const cusparseOperation_t op = transposeA ? CUSPARSE_OPERATION_TRANSPOSE
                                                : CUSPARSE_OPERATION_NON_TRANSPOSE;

      size_t bufSize = 0;
      cusparseSpMM_bufferSize(*handle, op, CUSPARSE_OPERATION_NON_TRANSPOSE,
                              &alpha, spA, dnB, &beta, dnC,
                              CT::computeType, CUSPARSE_SPMM_ALG_DEFAULT, &bufSize);

      void* workBuf = nullptr;
      if (bufSize > 0) {
        workBuf = memory::CudaMemoryPool::getInstance().allocate(bufSize, deviceId, *stream);
      }

      cusparseSpMM(*handle, op, CUSPARSE_OPERATION_NON_TRANSPOSE,
                   &alpha, spA, dnB, &beta, dnC,
                   CT::computeType, CUSPARSE_SPMM_ALG_DEFAULT, workBuf);

      if (workBuf) memory::CudaMemoryPool::getInstance().free(workBuf, deviceId, *stream);
      cusparseDestroyDnMat(dnB);
      cusparseDestroyDnMat(dnC);
      cusparseDestroySpMat(spA);
      return;
    }
    // fall through to manual kernel if strides are non-standard
  }
#endif

  // ---- Fallback: hand-written CUDA kernels ----
  if (transposeA == 0) {
    const dim3 block(16, 16);
    const dim3 grid(static_cast<unsigned>((rows + block.x - 1) / block.x),
                    static_cast<unsigned>((n    + block.y - 1) / block.y));
    csrSpMMKernel<X, I><<<grid, block, 0, *stream>>>(
        vBuf, ciBuf, rpBuf, bBuf, cBuf,
        rows, n, bStride0, bStride1, cStride0, cStride1);
  } else {
    const int blockSize = 256;
    const int gridSize  = static_cast<int>((rows + blockSize - 1) / blockSize);
    csrSpMMTransposeKernel<X, I><<<gridSize, blockSize, 0, *stream>>>(
        vBuf, ciBuf, rpBuf, bBuf, cBuf,
        rows, n, bStride0, bStride1, cStride0, cStride1);
  }
}

void csr_spmm(NDArray& values, NDArray& colIdx, NDArray& rowPtr,
              NDArray& B, NDArray& C,
              sd::LongType rows, sd::LongType cols, int transposeA) {
  NDArray::prepareSpecialUse({&C}, {&values, &colIdx, &rowPtr, &B});

  BUILD_DOUBLE_SELECTOR(values.dataType(), colIdx.dataType(), csrSpMMCuda_,
                        (values, colIdx, rowPtr, B, C, rows, cols, transposeA),
                        SD_FLOAT_TYPES, SD_INDEXING_TYPES);

  NDArray::registerSpecialUse({&C}, {&values, &colIdx, &rowPtr, &B});
}

// ===========================================================================
// sddmm  — CUDA dispatch
// ===========================================================================

template <typename X, typename I>
static void sddmmCuda_(NDArray& rowPtr, NDArray& colIdx,
                        NDArray& D1, NDArray& D2, NDArray& outValues,
                        sd::LongType rows, sd::LongType cols) {
  auto stream    = outValues.getContext()->getCudaStream();
  const int deviceId = outValues.getContext()->getDeviceID();

  const I* rpBuf  = reinterpret_cast<const I*>(rowPtr.specialBuffer());
  const I* ciBuf  = reinterpret_cast<const I*>(colIdx.specialBuffer());
  const X* d1Buf  = reinterpret_cast<const X*>(D1.specialBuffer());
  const X* d2Buf  = reinterpret_cast<const X*>(D2.specialBuffer());
  X*       oBuf   = reinterpret_cast<X*>(outValues.specialBuffer());

  const LongType p  = D1.sizeAt(1);
  const LongType nnz = outValues.lengthOf();

  const LongType d1Stride0 = D1.stridesOf()[0];
  const LongType d1Stride1 = D1.stridesOf()[1];
  const LongType d2Stride0 = D2.stridesOf()[0];
  const LongType d2Stride1 = D2.stridesOf()[1];

#if defined(HAVE_CUSPARSE)
  if constexpr (CusparseComputeTraits<X>::supported) {
    auto* handle = reinterpret_cast<cusparseHandle_t*>(outValues.getContext()->getCusparseHandle());
    cusparseSetStream(*handle, *stream);

    const cudaDataType valType = cusparseValueType<X>();
    const cusparseIndexType_t idxType = cusparseIdxType<I>();
    using CT = CusparseComputeTraits<X>;
    typename CT::scalar_t alpha = CT::one(), beta = CT::zero();

    // cusparseSDDMM: C = alpha * (A * B) ⊙ spy(C) + beta * C
    // A = D1 [rows, p] (dense), opA = NON_TRANSPOSE
    // B = D2 [cols, p] (dense), opB = TRANSPOSE  → effective B = [p, cols]
    // C = sparse [rows, cols] (our output)
    //
    // Only valid when D1/D2 are contiguous row-major.
    const bool d1Contiguous = (d1Stride1 == 1 && d1Stride0 == p);
    const bool d2Contiguous = (d2Stride1 == 1 && d2Stride0 == p);

    if (d1Contiguous && d2Contiguous) {
      // Sparse output C descriptor
      cusparseSpMatDescr_t spC;
      cusparseCreateCsr(&spC,
                        static_cast<int64_t>(rows), static_cast<int64_t>(cols),
                        static_cast<int64_t>(nnz),
                        const_cast<void*>(reinterpret_cast<const void*>(rpBuf)),
                        const_cast<void*>(reinterpret_cast<const void*>(ciBuf)),
                        reinterpret_cast<void*>(oBuf),
                        idxType,   // rowPtr — same integer type as colIdx
                        idxType,   // colIdx
                        CUSPARSE_INDEX_BASE_ZERO,
                        valType);

      // Dense A = D1 [rows, p]
      cusparseDnMatDescr_t dnA, dnB;
      cusparseCreateDnMat(&dnA,
                          static_cast<int64_t>(rows), static_cast<int64_t>(p),
                          static_cast<int64_t>(p),
                          const_cast<void*>(reinterpret_cast<const void*>(d1Buf)),
                          valType, CUSPARSE_ORDER_ROW);

      // Dense B = D2 [cols, p] (transposed in the call → effective [p, cols])
      cusparseCreateDnMat(&dnB,
                          static_cast<int64_t>(cols), static_cast<int64_t>(p),
                          static_cast<int64_t>(p),
                          const_cast<void*>(reinterpret_cast<const void*>(d2Buf)),
                          valType, CUSPARSE_ORDER_ROW);

      size_t bufSize = 0;
      cusparseSDDMM_bufferSize(*handle,
                               CUSPARSE_OPERATION_NON_TRANSPOSE,  // opA: D1
                               CUSPARSE_OPERATION_TRANSPOSE,      // opB: D2^T
                               &alpha, dnA, dnB, &beta, spC,
                               CT::computeType, CUSPARSE_SDDMM_ALG_DEFAULT, &bufSize);

      void* workBuf = nullptr;
      if (bufSize > 0) {
        workBuf = memory::CudaMemoryPool::getInstance().allocate(bufSize, deviceId, *stream);
      }

      // Preprocess (required by cuSPARSE SDDMM before the compute call)
      cusparseSDDMM_preprocess(*handle,
                               CUSPARSE_OPERATION_NON_TRANSPOSE,
                               CUSPARSE_OPERATION_TRANSPOSE,
                               &alpha, dnA, dnB, &beta, spC,
                               CT::computeType, CUSPARSE_SDDMM_ALG_DEFAULT, workBuf);

      cusparseSDDMM(*handle,
                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                    CUSPARSE_OPERATION_TRANSPOSE,
                    &alpha, dnA, dnB, &beta, spC,
                    CT::computeType, CUSPARSE_SDDMM_ALG_DEFAULT, workBuf);

      if (workBuf) memory::CudaMemoryPool::getInstance().free(workBuf, deviceId, *stream);
      cusparseDestroyDnMat(dnA);
      cusparseDestroyDnMat(dnB);
      cusparseDestroySpMat(spC);
      return;
    }
    // fall through to manual kernel if strides are non-standard
  }
#endif

  // ---- Fallback: hand-written CUDA kernel (one thread per NNZ) ----
  const int blockSize = 256;
  const int gridSize  = static_cast<int>((nnz + blockSize - 1) / blockSize);
  sddmmKernel<X, I><<<gridSize, blockSize, 0, *stream>>>(
      rpBuf, ciBuf, d1Buf, d2Buf, oBuf,
      rows, nnz, p,
      d1Stride0, d1Stride1, d2Stride0, d2Stride1);
}

void sddmm(NDArray& rowPtr, NDArray& colIdx,
           NDArray& D1, NDArray& D2, NDArray& outValues,
           sd::LongType rows, sd::LongType cols) {
  NDArray::prepareSpecialUse({&outValues}, {&rowPtr, &colIdx, &D1, &D2});

  BUILD_DOUBLE_SELECTOR(D1.dataType(), colIdx.dataType(), sddmmCuda_,
                        (rowPtr, colIdx, D1, D2, outValues, rows, cols),
                        SD_FLOAT_TYPES, SD_INDEXING_TYPES);

  NDArray::registerSpecialUse({&outValues}, {&rowPtr, &colIdx, &D1, &D2});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd

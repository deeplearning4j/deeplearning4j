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
// BSR helpers — CUDA implementations.
//
// Three operations:
//
//   csr_to_bsr  — uses cusparseXcsr2bsrNnz (fills bsrRowPtr, returns nnzb host-side)
//                 + cusparseScsr2bsr / cusparseDcsr2bsr (DIRECTION_ROW, BASE_ZERO).
//                 INT64 → INT32 cast on-device via NDArray::cast().
//                 half/bf16 → float32 cast-up on-device; result cast back.
//                 Asserts cuSPARSE nnzb == DECLARE_SHAPE_FN count; throws on mismatch.
//
//   bsr_to_dense — custom CUDA kernel (gridDim.x=nnzb, blockDim.x=bd).
//                  One thread per row within each BSR block; writes bd values per thread.
//                  bi found via binary search on bsrRowPtr (device-side, shared per block).
//
//   bsr_spmm    — BSR→CSR conversion (cusparseSbsr2csr / cusparseDbsr2csr) using pool
//                 workspace, then cusparseSpMM generic API with CUSPARSE_ORDER_ROW.
//                 This correctly handles row-major B and C (avoids col-major mismatch
//                 of the legacy cusparseSbsrmm with row-major inputs).
//                 half/bf16: cast to float32 for both conversion and SpMM steps.
//
// Rules (matching all other sparse CUDA helpers):
//   • No raw cudaMalloc — all temporary device memory from memory::CudaMemoryPool.
//   • cusparseSetStream before every cuSPARSE call sequence.
//   • #if defined(HAVE_CUSPARSE) guards throughout; throw if absent.
//   • INT64 → INT32 cast via NDArray::cast() + prepareSpecialUse read-set, no host round-trip.
//   • All NDArrayFactory::create calls wrapped in std::unique_ptr<NDArray>.
//   • No fallback to slot-by-slot or host compute on failure; always throw.
//

#include <array/NDArrayFactory.h>
#include <cuda_runtime.h>
#include <memory/cuda/CudaMemoryPool.h>
#include <ops/declarable/helpers/sparse_bsr.h>
#include <system/op_boilerplate.h>
#include <types/bfloat16.h>
#include <types/float16.h>

#include <memory>
#include <type_traits>

#if defined(HAVE_CUSPARSE)
#include <cusparse_v2.h>
#endif

namespace sd {
namespace ops {
namespace helpers {

// ══════════════════════════════════════════════════════════════════════════════
// Error helper (local to this TU)
// ══════════════════════════════════════════════════════════════════════════════

#if defined(HAVE_CUSPARSE)

static SD_INLINE void checkBsr(cusparseStatus_t st, const char* where) {
  if (st != CUSPARSE_STATUS_SUCCESS) {
    std::string msg = std::string("BSR cuSPARSE error at [") + where +
                      "] status=" + std::to_string(static_cast<int>(st));
    THROW_EXCEPTION(msg.c_str());
  }
}

// Types natively supported by the legacy cuSPARSE BSR API (float and double).
template <typename X>
struct BsrNative {
  static constexpr bool value = false;
};
template <>
struct BsrNative<float> {
  static constexpr bool value = true;
};
template <>
struct BsrNative<double> {
  static constexpr bool value = true;
};

// ── Helper: create and configure a cuSPARSE general matrix descriptor ─────────
static cusparseMatDescr_t makeBsrDescr(const char* tag) {
  cusparseMatDescr_t d;
  checkBsr(cusparseCreateMatDescr(&d), tag);
  cusparseSetMatIndexBase(d, CUSPARSE_INDEX_BASE_ZERO);
  cusparseSetMatType(d, CUSPARSE_MATRIX_TYPE_GENERAL);
  return d;
}

#endif  // HAVE_CUSPARSE

// ══════════════════════════════════════════════════════════════════════════════
// Section A: csr_to_bsr — cuSPARSE legacy MatDescr API
//
//   Phase 1: cusparseXcsr2bsrNnz → fills bsrRowPtr on device, returns nnzb host-side.
//   Phase 2: cusparseScsr2bsr / cusparseDcsr2bsr → fills bsrValues and bsrColIdx.
// ══════════════════════════════════════════════════════════════════════════════

#if defined(HAVE_CUSPARSE)

// Count non-zero blocks and fill bsrRowPtr.
static int runXcsr2bsrNnz(cusparseHandle_t handle, int rows, int cols, int bd,
                           const int32_t* csrRowPtrDev, const int32_t* csrColIdxDev,
                           int32_t* bsrRowPtrDev) {
  cusparseMatDescr_t dA = makeBsrDescr("Xcsr2bsrNnz:A");
  cusparseMatDescr_t dC = makeBsrDescr("Xcsr2bsrNnz:C");
  int nnzb = 0;
  checkBsr(cusparseXcsr2bsrNnz(handle, CUSPARSE_DIRECTION_ROW, rows, cols, dA, csrRowPtrDev,
                                csrColIdxDev, bd, dC, bsrRowPtrDev, &nnzb),
           "cusparseXcsr2bsrNnz");
  cusparseDestroyMatDescr(dA);
  cusparseDestroyMatDescr(dC);
  return nnzb;
}

// Fill bsrValues and bsrColIdx for float32.
static void runScsr2bsr(cusparseHandle_t handle, int rows, int cols, int bd,
                        const float* csrVal, const int32_t* csrRowPtr, const int32_t* csrColIdx,
                        float* bsrVal, int32_t* bsrRowPtr, int32_t* bsrColIdx) {
  cusparseMatDescr_t dA = makeBsrDescr("Scsr2bsr:A");
  cusparseMatDescr_t dC = makeBsrDescr("Scsr2bsr:C");
  checkBsr(cusparseScsr2bsr(handle, CUSPARSE_DIRECTION_ROW, rows, cols, dA, csrVal, csrRowPtr,
                             csrColIdx, bd, dC, bsrVal, bsrRowPtr, bsrColIdx),
           "cusparseScsr2bsr");
  cusparseDestroyMatDescr(dA);
  cusparseDestroyMatDescr(dC);
}

// Fill bsrValues and bsrColIdx for float64.
static void runDcsr2bsr(cusparseHandle_t handle, int rows, int cols, int bd,
                        const double* csrVal, const int32_t* csrRowPtr, const int32_t* csrColIdx,
                        double* bsrVal, int32_t* bsrRowPtr, int32_t* bsrColIdx) {
  cusparseMatDescr_t dA = makeBsrDescr("Dcsr2bsr:A");
  cusparseMatDescr_t dC = makeBsrDescr("Dcsr2bsr:C");
  checkBsr(cusparseDcsr2bsr(handle, CUSPARSE_DIRECTION_ROW, rows, cols, dA, csrVal, csrRowPtr,
                             csrColIdx, bd, dC, bsrVal, bsrRowPtr, bsrColIdx),
           "cusparseDcsr2bsr");
  cusparseDestroyMatDescr(dA);
  cusparseDestroyMatDescr(dC);
}

template <typename X, typename I>
static void csrToBsrCuda_(NDArray& csrValues, NDArray& csrColIdx, NDArray& csrRowPtr,
                           NDArray& bsrValues, NDArray& bsrColIdx, NDArray& bsrRowPtr,
                           LongType rows, LongType cols, LongType blockDim) {
  auto* ctx    = bsrValues.getContext();
  auto  stream = ctx->getCudaStream();
  auto* handle = reinterpret_cast<cusparseHandle_t*>(ctx->getCusparseHandle());
  checkBsr(cusparseSetStream(*handle, *stream), "csr_to_bsr: cusparseSetStream");

  const int iRows = static_cast<int>(rows);
  const int iCols = static_cast<int>(cols);
  const int bd    = static_cast<int>(blockDim);
  // nnzb pre-computed by DECLARE_SHAPE_FN symbolic pass.
  const int expectedNnzb = static_cast<int>(bsrColIdx.lengthOf());

  // ── INT64 → INT32 on-device if needed (legacy API requires int32) ──────────
  std::unique_ptr<NDArray> csrColIdxI32, csrRowPtrI32;
  const int32_t* csrColInd32;
  const int32_t* csrRowOff32;

  if constexpr (sizeof(I) == 4) {
    csrColInd32 = reinterpret_cast<const int32_t*>(csrColIdx.specialBuffer());
    csrRowOff32 = reinterpret_cast<const int32_t*>(csrRowPtr.specialBuffer());
  } else {
    csrColIdxI32.reset(csrColIdx.cast(sd::DataType::INT32));
    csrRowPtrI32.reset(csrRowPtr.cast(sd::DataType::INT32));
    NDArray::prepareSpecialUse({}, {csrColIdxI32.get(), csrRowPtrI32.get()});
    csrColInd32 = reinterpret_cast<const int32_t*>(csrColIdxI32->specialBuffer());
    csrRowOff32 = reinterpret_cast<const int32_t*>(csrRowPtrI32->specialBuffer());
  }

  int32_t* bsrRowPtrDev = reinterpret_cast<int32_t*>(bsrRowPtr.specialBuffer());
  int32_t* bsrColIdxDev = reinterpret_cast<int32_t*>(bsrColIdx.specialBuffer());

  // ── Dispatch by value type ─────────────────────────────────────────────────
  if constexpr (BsrNative<X>::value) {
    // Phase 1: fill bsrRowPtr and get nnzb; verify against DECLARE_SHAPE_FN.
    const int nnzb = runXcsr2bsrNnz(*handle, iRows, iCols, bd, csrRowOff32, csrColInd32,
                                     bsrRowPtrDev);
    if (nnzb != expectedNnzb) {
      THROW_EXCEPTION(
          "csr_to_bsr CUDA: cuSPARSE nnzb mismatch with DECLARE_SHAPE_FN — "
          "graph-construction and runtime CSR structure differ");
    }
    if (nnzb == 0) return;  // empty BSR

    // Phase 2: fill bsrValues and bsrColIdx.
    if constexpr (std::is_same_v<X, float>) {
      runScsr2bsr(*handle, iRows, iCols, bd,
                  reinterpret_cast<const float*>(csrValues.specialBuffer()),
                  csrRowOff32, csrColInd32,
                  reinterpret_cast<float*>(bsrValues.specialBuffer()),
                  bsrRowPtrDev, bsrColIdxDev);
    } else {
      runDcsr2bsr(*handle, iRows, iCols, bd,
                  reinterpret_cast<const double*>(csrValues.specialBuffer()),
                  csrRowOff32, csrColInd32,
                  reinterpret_cast<double*>(bsrValues.specialBuffer()),
                  bsrRowPtrDev, bsrColIdxDev);
    }
  } else {
    // half / bfloat16: up-cast to float32 on-device, run Scsr2bsr, cast back.
    std::unique_ptr<NDArray> csrValF32(csrValues.cast(sd::DataType::FLOAT32));
    NDArray::prepareSpecialUse({}, {csrValF32.get()});

    const LongType bsrValLen = static_cast<LongType>(expectedNnzb) * blockDim * blockDim;
    std::unique_ptr<NDArray> bsrValF32(
        NDArrayFactory::create('c', std::vector<sd::LongType>{bsrValLen},
                               sd::DataType::FLOAT32, ctx));

    // Phase 1
    const int nnzb = runXcsr2bsrNnz(*handle, iRows, iCols, bd, csrRowOff32, csrColInd32,
                                     bsrRowPtrDev);
    if (nnzb != expectedNnzb) {
      THROW_EXCEPTION("csr_to_bsr CUDA (half path): cuSPARSE nnzb mismatch with DECLARE_SHAPE_FN");
    }
    if (nnzb == 0) return;

    // Phase 2
    runScsr2bsr(*handle, iRows, iCols, bd,
                reinterpret_cast<const float*>(csrValF32->specialBuffer()),
                csrRowOff32, csrColInd32,
                reinterpret_cast<float*>(bsrValF32->specialBuffer()),
                bsrRowPtrDev, bsrColIdxDev);

    NDArray::prepareSpecialUse({}, {bsrValF32.get()});
    bsrValues.assign(bsrValF32.get());
  }
}

#endif  // HAVE_CUSPARSE

void csr_to_bsr(NDArray& csrValues, NDArray& csrColIdx, NDArray& csrRowPtr,
                NDArray& bsrValues, NDArray& bsrColIdx, NDArray& bsrRowPtr,
                LongType rows, LongType cols, LongType blockDim) {
  NDArray::prepareSpecialUse({&bsrValues, &bsrColIdx, &bsrRowPtr},
                             {&csrValues, &csrColIdx, &csrRowPtr});
#if defined(HAVE_CUSPARSE)
  BUILD_DOUBLE_SELECTOR(
      csrValues.dataType(), csrColIdx.dataType(), csrToBsrCuda_,
      (csrValues, csrColIdx, csrRowPtr, bsrValues, bsrColIdx, bsrRowPtr, rows, cols, blockDim),
      SD_FLOAT_TYPES, SD_INDEXING_TYPES);
#else
  THROW_EXCEPTION(
      "csr_to_bsr CUDA: cuSPARSE required (HAVE_CUSPARSE not defined). "
      "Rebuild with -Dlibnd4j.cusparse=ON or use CPU backend.");
#endif
  NDArray::registerSpecialUse({&bsrValues, &bsrColIdx, &bsrRowPtr},
                              {&csrValues, &csrColIdx, &csrRowPtr});
}

// ══════════════════════════════════════════════════════════════════════════════
// Section B: bsr_to_dense — custom CUDA kernel
//
// gridDim.x = nnzb  (one CUDA thread-block per BSR block)
// blockDim.x = bd   (one thread per row within the BSR block, bd ≤ 1024)
//
// Each thread writes one full row of the BSR block (bd values).
// bi is found via binary search on bsrRowPtr, shared across the CUDA block
// via one read by thread 0 broadcast via shared memory.
// ══════════════════════════════════════════════════════════════════════════════

template <typename X, typename I>
static SD_KERNEL void bsrToDenseKernel(const X* bsrValues, const I* bsrColIdx,
                                        const I* bsrRowPtr, X* output, LongType mb,
                                        LongType bd, LongType oStride0, LongType oStride1) {
  // One CUDA block per BSR block, one thread per in-block row.
  const LongType blk = static_cast<LongType>(blockIdx.x);
  const LongType r   = static_cast<LongType>(threadIdx.x);
  if (r >= bd) return;

  // Shared bi: thread 0 binary-searches bsrRowPtr; all threads read it.
  __shared__ LongType shBi;
  if (r == 0) {
    LongType lo = 0, hi = mb - 1, bi = 0;
    while (lo <= hi) {
      const LongType mid = (lo + hi) / 2;
      const LongType rL  = static_cast<LongType>(bsrRowPtr[mid]);
      const LongType rR  = static_cast<LongType>(bsrRowPtr[mid + 1]);
      if (rL <= blk && blk < rR) { bi = mid; break; }
      else if (rL > blk)         { if (mid == 0) break; hi = mid - 1; }
      else                       { lo = mid + 1; }
    }
    shBi = bi;
  }
  __syncthreads();

  const LongType bi      = shBi;
  const LongType bj      = static_cast<LongType>(bsrColIdx[blk]);
  const LongType blkBase = blk * bd * bd;
  const LongType rowOut  = bi * bd + r;

  for (LongType c = 0; c < bd; ++c) {
    output[rowOut * oStride0 + (bj * bd + c) * oStride1] = bsrValues[blkBase + r * bd + c];
  }
}

template <typename X, typename I>
static void bsrToDenseCuda_(NDArray& bsrValues, NDArray& bsrColIdx, NDArray& bsrRowPtr,
                             NDArray& output, LongType rows, LongType /*cols*/, LongType blockDim) {
  const LongType mb   = rows / blockDim;
  const LongType bd   = blockDim;
  const LongType nnzb = bsrColIdx.lengthOf();
  if (nnzb == 0) return;  // already zeroed by OUTPUT_NULLIFIED

  const X* bvBuf = reinterpret_cast<const X*>(bsrValues.specialBuffer());
  const I* bciB  = reinterpret_cast<const I*>(bsrColIdx.specialBuffer());
  const I* brpB  = reinterpret_cast<const I*>(bsrRowPtr.specialBuffer());
  X*       oBuf  = reinterpret_cast<X*>(output.specialBuffer());

  auto stream = output.getContext()->getCudaStream();

  bsrToDenseKernel<X, I><<<static_cast<int>(nnzb), static_cast<int>(bd), 0, *stream>>>(
      bvBuf, bciB, brpB, oBuf, mb, bd,
      output.stridesOf()[0], output.stridesOf()[1]);
}

void bsr_to_dense(NDArray& bsrValues, NDArray& bsrColIdx, NDArray& bsrRowPtr,
                  NDArray& output, LongType rows, LongType cols, LongType blockDim) {
  NDArray::prepareSpecialUse({&output}, {&bsrValues, &bsrColIdx, &bsrRowPtr});

  BUILD_DOUBLE_SELECTOR(bsrValues.dataType(), bsrColIdx.dataType(), bsrToDenseCuda_,
                        (bsrValues, bsrColIdx, bsrRowPtr, output, rows, cols, blockDim),
                        SD_FLOAT_TYPES, SD_INDEXING_TYPES);

  NDArray::registerSpecialUse({&output}, {&bsrValues, &bsrColIdx, &bsrRowPtr});
}

// ══════════════════════════════════════════════════════════════════════════════
// Section C: bsr_spmm — BSR→CSR + cusparseSpMM (generic, row-major)
//
// Row-major convention:
//   cusparseSbsrmm / cusparseDbsrmm expect column-major B and C, so they cannot
//   directly accept our row-major inputs without producing wrong results when
//   rows ≠ n.  Instead:
//
//   1. Convert BSR → CSR on-device (cusparseSbsr2csr / cusparseDbsr2csr).
//      Workspace buffers from CudaMemoryPool; freed before return.
//   2. Run cusparseSpMM generic API with CUSPARSE_ORDER_ROW (correctly handles
//      row-major B [cols, n] and C [rows, n]).
//
//   This is the same pattern as csr_spmm in sparse_blas.cu.
//   half/bf16: cast bsrValues and B to float32, run, cast C back.
// ══════════════════════════════════════════════════════════════════════════════

#if defined(HAVE_CUSPARSE)

// Run BSR→CSR using pool-allocated CSR buffers, then run generic cusparseSpMM.
// FloatT is float or double.
template <typename FloatT>
static void runBsrSpmm(
    cusparseHandle_t handle, cudaStream_t stream, int deviceId,
    const FloatT*  bsrVal, const int32_t* bsrRowPtrDev, const int32_t* bsrColIdxDev,
    int mb, int nb, int bd, int nnzb,
    LongType rows, LongType cols,
    const FloatT*  bBuf, LongType n, LongType bLd,
    FloatT*        cBuf, LongType cLd)
{
  const cudaDataType valType = std::is_same<FloatT, float>::value ? CUDA_R_32F : CUDA_R_64F;

  const LongType csrNnz     = static_cast<LongType>(nnzb) * bd * bd;
  const size_t   csrValSz   = csrNnz * sizeof(FloatT);
  const size_t   csrColSz   = csrNnz * sizeof(int32_t);
  const size_t   csrRowSz   = (rows + 1) * sizeof(int32_t);

  // ── Allocate temporary CSR buffers from pool ────────────────────────────
  auto* csrVal    = reinterpret_cast<FloatT*>(
      memory::CudaMemoryPool::getInstance().allocate(csrValSz, deviceId, stream));
  auto* csrColIdx = reinterpret_cast<int32_t*>(
      memory::CudaMemoryPool::getInstance().allocate(csrColSz, deviceId, stream));
  auto* csrRowPtr = reinterpret_cast<int32_t*>(
      memory::CudaMemoryPool::getInstance().allocate(csrRowSz, deviceId, stream));

  if (!csrVal || !csrColIdx || !csrRowPtr) {
    if (csrVal)    memory::CudaMemoryPool::getInstance().free(csrVal,    deviceId, stream);
    if (csrColIdx) memory::CudaMemoryPool::getInstance().free(csrColIdx, deviceId, stream);
    if (csrRowPtr) memory::CudaMemoryPool::getInstance().free(csrRowPtr, deviceId, stream);
    THROW_EXCEPTION("bsr_spmm: CudaMemoryPool allocation failed for BSR→CSR workspace");
  }

  // ── BSR → CSR conversion ────────────────────────────────────────────────
  cusparseMatDescr_t dA = makeBsrDescr("bsr_spmm:bsr2csr:A");
  cusparseMatDescr_t dC = makeBsrDescr("bsr_spmm:bsr2csr:C");

  cusparseStatus_t st;
  if constexpr (std::is_same_v<FloatT, float>) {
    st = cusparseSbsr2csr(handle, CUSPARSE_DIRECTION_ROW, mb, nb, dA,
                           reinterpret_cast<const float*>(bsrVal), bsrRowPtrDev, bsrColIdxDev, bd,
                           dC, reinterpret_cast<float*>(csrVal), csrRowPtr, csrColIdx);
  } else {
    st = cusparseDbsr2csr(handle, CUSPARSE_DIRECTION_ROW, mb, nb, dA,
                           reinterpret_cast<const double*>(bsrVal), bsrRowPtrDev, bsrColIdxDev, bd,
                           dC, reinterpret_cast<double*>(csrVal), csrRowPtr, csrColIdx);
  }
  cusparseDestroyMatDescr(dA);
  cusparseDestroyMatDescr(dC);
  if (st != CUSPARSE_STATUS_SUCCESS) {
    memory::CudaMemoryPool::getInstance().free(csrVal,    deviceId, stream);
    memory::CudaMemoryPool::getInstance().free(csrColIdx, deviceId, stream);
    memory::CudaMemoryPool::getInstance().free(csrRowPtr, deviceId, stream);
    checkBsr(st, "bsr_spmm: cusparseSbsr2csr/cusparseDbsr2csr");
  }

  // ── Generic cusparseSpMM with CUSPARSE_ORDER_ROW ────────────────────────
  using scalar_t = FloatT;
  scalar_t alpha = static_cast<scalar_t>(1), beta = static_cast<scalar_t>(0);

  cusparseSpMatDescr_t spA;
  checkBsr(cusparseCreateCsr(&spA,
                              static_cast<int64_t>(rows), static_cast<int64_t>(cols),
                              static_cast<int64_t>(csrNnz),
                              static_cast<void*>(csrRowPtr),
                              static_cast<void*>(csrColIdx),
                              static_cast<void*>(csrVal),
                              CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                              CUSPARSE_INDEX_BASE_ZERO, valType),
           "bsr_spmm: cusparseCreateCsr");

  // B [cols, n] row-major, leading dim = bLd (= n for contiguous)
  cusparseDnMatDescr_t dnB;
  checkBsr(cusparseCreateDnMat(&dnB,
                                static_cast<int64_t>(cols), static_cast<int64_t>(n),
                                static_cast<int64_t>(bLd),
                                const_cast<void*>(reinterpret_cast<const void*>(bBuf)),
                                valType, CUSPARSE_ORDER_ROW),
           "bsr_spmm: cusparseCreateDnMat(B)");

  // C [rows, n] row-major, leading dim = cLd (= n for contiguous)
  cusparseDnMatDescr_t dnC;
  checkBsr(cusparseCreateDnMat(&dnC,
                                static_cast<int64_t>(rows), static_cast<int64_t>(n),
                                static_cast<int64_t>(cLd),
                                reinterpret_cast<void*>(cBuf),
                                valType, CUSPARSE_ORDER_ROW),
           "bsr_spmm: cusparseCreateDnMat(C)");

  const cudaDataType computeType = std::is_same<FloatT, float>::value ? CUDA_R_32F : CUDA_R_64F;

  size_t bufSize = 0;
  checkBsr(cusparseSpMM_bufferSize(handle,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha, spA, dnB, &beta, dnC,
                                    computeType, CUSPARSE_SPMM_ALG_DEFAULT, &bufSize),
           "bsr_spmm: cusparseSpMM_bufferSize");

  void* workBuf = nullptr;
  if (bufSize > 0) {
    workBuf = memory::CudaMemoryPool::getInstance().allocate(bufSize, deviceId, stream);
    if (!workBuf) {
      cusparseDestroyDnMat(dnB);
      cusparseDestroyDnMat(dnC);
      cusparseDestroySpMat(spA);
      memory::CudaMemoryPool::getInstance().free(csrVal,    deviceId, stream);
      memory::CudaMemoryPool::getInstance().free(csrColIdx, deviceId, stream);
      memory::CudaMemoryPool::getInstance().free(csrRowPtr, deviceId, stream);
      THROW_EXCEPTION("bsr_spmm: CudaMemoryPool failed to allocate SpMM workspace");
    }
  }

  checkBsr(cusparseSpMM(handle,
                         CUSPARSE_OPERATION_NON_TRANSPOSE,
                         CUSPARSE_OPERATION_NON_TRANSPOSE,
                         &alpha, spA, dnB, &beta, dnC,
                         computeType, CUSPARSE_SPMM_ALG_DEFAULT, workBuf),
           "bsr_spmm: cusparseSpMM");

  // ── Cleanup ───────────────────────────────────────────────────────────────
  if (workBuf) memory::CudaMemoryPool::getInstance().free(workBuf,   deviceId, stream);
  cusparseDestroyDnMat(dnB);
  cusparseDestroyDnMat(dnC);
  cusparseDestroySpMat(spA);
  memory::CudaMemoryPool::getInstance().free(csrVal,    deviceId, stream);
  memory::CudaMemoryPool::getInstance().free(csrColIdx, deviceId, stream);
  memory::CudaMemoryPool::getInstance().free(csrRowPtr, deviceId, stream);
}

template <typename X, typename I>
static void bsrSpmmCuda_(NDArray& bsrValues, NDArray& bsrColIdx, NDArray& bsrRowPtr,
                          NDArray& B, NDArray& C,
                          LongType rows, LongType cols, LongType blockDim) {
  auto* ctx    = C.getContext();
  auto  stream = ctx->getCudaStream();
  auto* handle = reinterpret_cast<cusparseHandle_t*>(ctx->getCusparseHandle());
  checkBsr(cusparseSetStream(*handle, *stream), "bsr_spmm: cusparseSetStream");

  const int mb   = static_cast<int>(rows / blockDim);
  const int nb   = static_cast<int>(cols / blockDim);
  const int bd   = static_cast<int>(blockDim);
  const int nnzb = static_cast<int>(bsrColIdx.lengthOf());
  if (nnzb == 0) return;  // C is already OUTPUT_NULLIFIED = zeroed

  const int deviceId = ctx->getDeviceID();

  const LongType n    = B.sizeAt(1);
  const LongType bLd  = B.stridesOf()[0];    // row stride of B (row-major leading dim)
  const LongType cLd  = C.stridesOf()[0];    // row stride of C

  // BSR index arrays are always INT32 per op contract.
  const int32_t* bsrColIdxDev = reinterpret_cast<const int32_t*>(bsrColIdx.specialBuffer());
  const int32_t* bsrRowPtrDev = reinterpret_cast<const int32_t*>(bsrRowPtr.specialBuffer());

  if constexpr (BsrNative<X>::value) {
    runBsrSpmm<X>(*handle, *stream, deviceId,
                   reinterpret_cast<const X*>(bsrValues.specialBuffer()),
                   bsrRowPtrDev, bsrColIdxDev,
                   mb, nb, bd, nnzb, rows, cols,
                   reinterpret_cast<const X*>(B.specialBuffer()), n, bLd,
                   reinterpret_cast<X*>(C.specialBuffer()), cLd);
  } else {
    // half / bfloat16: cast bsrValues and B to float32, run, cast C back.
    std::unique_ptr<NDArray> bsrValF32(bsrValues.cast(sd::DataType::FLOAT32));
    std::unique_ptr<NDArray> bF32(B.cast(sd::DataType::FLOAT32));
    NDArray::prepareSpecialUse({}, {bsrValF32.get(), bF32.get()});

    std::unique_ptr<NDArray> cF32(
        NDArrayFactory::create('c', std::vector<sd::LongType>{rows, n},
                               sd::DataType::FLOAT32, ctx));

    runBsrSpmm<float>(*handle, *stream, deviceId,
                       reinterpret_cast<const float*>(bsrValF32->specialBuffer()),
                       bsrRowPtrDev, bsrColIdxDev,
                       mb, nb, bd, nnzb, rows, cols,
                       reinterpret_cast<const float*>(bF32->specialBuffer()),
                       n, bF32->stridesOf()[0],
                       reinterpret_cast<float*>(cF32->specialBuffer()),
                       cF32->stridesOf()[0]);

    NDArray::prepareSpecialUse({}, {cF32.get()});
    C.assign(cF32.get());
  }
}

#endif  // HAVE_CUSPARSE

void bsr_spmm(NDArray& bsrValues, NDArray& bsrColIdx, NDArray& bsrRowPtr,
              NDArray& B, NDArray& C,
              LongType rows, LongType cols, LongType blockDim) {
  NDArray::prepareSpecialUse({&C}, {&bsrValues, &bsrColIdx, &bsrRowPtr, &B});

#if defined(HAVE_CUSPARSE)
  BUILD_DOUBLE_SELECTOR(bsrValues.dataType(), bsrColIdx.dataType(), bsrSpmmCuda_,
                        (bsrValues, bsrColIdx, bsrRowPtr, B, C, rows, cols, blockDim),
                        SD_FLOAT_TYPES, SD_INDEXING_TYPES);
#else
  THROW_EXCEPTION(
      "bsr_spmm CUDA: cuSPARSE required (HAVE_CUSPARSE not defined). "
      "Rebuild with -Dlibnd4j.cusparse=ON or use CPU backend.");
#endif

  NDArray::registerSpecialUse({&C}, {&bsrValues, &bsrColIdx, &bsrRowPtr, &B});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd

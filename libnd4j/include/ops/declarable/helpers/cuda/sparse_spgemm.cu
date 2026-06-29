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
// CSR SpGEMM — CUDA implementation using the cuSPARSE generic SpGEMM API.
//
// C = A * B  (alpha=1, beta=0, both NON_TRANSPOSE)
//
// Design notes
// ────────────
// 1. OUTPUT PRE-ALLOCATION: The op's DECLARE_SHAPE_FN runs a host-side
//    Gustavson symbolic pass and pre-allocates cValues[cnnz], cColIdx[cnnz]
//    (INT32), cRowPtr[m+1] (INT32) with the exact final nnz.  cuSPARSE runs its
//    own symbolic pass; its nnz is asserted equal to cnnz.  On mismatch we
//    THROW (no silent truncation, no fallback).
//
// 2. INDEX TYPE: cuSPARSE SpGEMM requires CUSPARSE_INDEX_32I for all CSR index
//    buffers.  If the template index type I is 64-bit, the input index arrays
//    (aColIdx, aRowPtr, bColIdx, bRowPtr) are cast to INT32 on-device via
//    NDArray::cast() with no host round-trip.  The output index arrays
//    (cColIdx, cRowPtr) are always INT32 per the op contract.
//
// 3. VALUE TYPE: float and double are natively supported.  float16 / bfloat16
//    are NOT supported by cuSPARSE SpGEMM; they are cast to float on-device,
//    the multiplication is executed in float, and the result is cast back to X
//    on-device.  No host fallback at any step.
//
// 4. WORKSPACE: all temporary GPU memory is allocated from CudaMemoryPool
//    (no raw cudaMalloc).
//
// 5. NO FALLBACK: any cuSPARSE error or nnz mismatch throws immediately.
//
// REFERENCE PATTERN: libnd4j/include/ops/declarable/helpers/cuda/sparse_blas.cu
//

#include <cuda_runtime.h>
#include <memory/cuda/CudaMemoryPool.h>
#include <ops/declarable/helpers/sparse_spgemm.h>
#include <system/op_boilerplate.h>
#include <types/bfloat16.h>
#include <types/float16.h>

#if defined(HAVE_CUSPARSE)
#include <cusparse_v2.h>
#endif

namespace sd {
namespace ops {
namespace helpers {

// ═══════════════════════════════════════════════════════════════════════════════
// Section A: cuSPARSE-dependent implementation
// ═══════════════════════════════════════════════════════════════════════════════

#if defined(HAVE_CUSPARSE)

// ── A.1  Type-mapping helpers (mirrors sparse_blas.cu) ─────────────────────────

// Map C++ floating type → CUDA compute/value type (float and double only here;
// float16/bfloat16 are promoted to CUDA_R_32F before reaching cuSPARSE).
template <typename T>
static SD_INLINE cudaDataType spgemmCudaValType();
template <> SD_INLINE cudaDataType spgemmCudaValType<float>()  { return CUDA_R_32F; }
template <> SD_INLINE cudaDataType spgemmCudaValType<double>() { return CUDA_R_64F; }

// Compute scalar (alpha/beta) and compute type for cuSPARSE.
template <typename X> struct SpgemmCompute {
  // Default: promote everything to float.
  using scalar_t = float;
  static cudaDataType computeType() { return CUDA_R_32F; }
  static scalar_t one()  { return 1.0f; }
  static scalar_t zero() { return 0.0f; }
};
template <> struct SpgemmCompute<double> {
  using scalar_t = double;
  static cudaDataType computeType() { return CUDA_R_64F; }
  static scalar_t one()  { return 1.0; }
  static scalar_t zero() { return 0.0; }
};

// True iff X can be passed directly to cuSPARSE SpGEMM without up-casting.
template <typename X> struct SpgemmNative { static constexpr bool value = false; };
template <> struct SpgemmNative<float>    { static constexpr bool value = true;  };
template <> struct SpgemmNative<double>   { static constexpr bool value = true;  };

// ── A.2  Status-checking helper ────────────────────────────────────────────────

static SD_INLINE void checkCusparse(cusparseStatus_t st, const char* where) {
  if (st != CUSPARSE_STATUS_SUCCESS) {
    std::string msg = std::string("csr_spgemm cuSPARSE error at [") + where +
                      "] status=" + std::to_string(static_cast<int>(st));
    THROW_EXCEPTION(msg.c_str());
  }
}

// ── A.3  Core SpGEMM routine (float or double on-device) ──────────────────────
//
// Preconditions:
//   • handle has its stream already set.
//   • All index buffers are INT32 device pointers (CUSPARSE_INDEX_32I).
//   • cRowPtr32 is already bound as the pre-allocated output rowPtr buffer.
//   • cColIdx32 and cValBuf are the pre-allocated output colIdx/values buffers.
//   • expectedCnnz == cValues.lengthOf() from the caller; we assert it equals
//     the nnz that cuSPARSE computes symbolically.
//
template <typename FloatT>
static void runSpGEMMDevice(
    cusparseHandle_t handle,
    cudaStream_t     stream,
    int              deviceId,
    // A CSR (inputs)
    const void*      aVal,
    const int32_t*   aColIdx32,
    const int32_t*   aRowPtr32,
    sd::LongType     annz,
    // B CSR (inputs)
    const void*      bVal,
    const int32_t*   bColIdx32,
    const int32_t*   bRowPtr32,
    sd::LongType     bnnz,
    // C CSR (pre-allocated outputs)
    void*            cVal,
    int32_t*         cColIdx32,
    int32_t*         cRowPtr32,
    // Matrix dimensions
    sd::LongType     m,
    sd::LongType     k,
    sd::LongType     n,
    // Expected output nnz (from DECLARE_SHAPE_FN)
    sd::LongType     expectedCnnz)
{
  using SC = SpgemmCompute<FloatT>;
  const cudaDataType valType = spgemmCudaValType<FloatT>();
  typename SC::scalar_t alpha = SC::one(), beta = SC::zero();

  // ── 1. Create sparse CSR descriptors ─────────────────────────────────────
  cusparseSpMatDescr_t matA, matB, matC;

  checkCusparse(
    cusparseCreateCsr(
        &matA,
        static_cast<int64_t>(m), static_cast<int64_t>(k),
        static_cast<int64_t>(annz),
        const_cast<int32_t*>(aRowPtr32),   // rowOffsets
        const_cast<int32_t*>(aColIdx32),   // colIndices
        const_cast<void*>(aVal),            // values
        CUSPARSE_INDEX_32I,                 // rowOffsets type
        CUSPARSE_INDEX_32I,                 // colIndices type
        CUSPARSE_INDEX_BASE_ZERO,
        valType),
    "cusparseCreateCsr(A)");

  checkCusparse(
    cusparseCreateCsr(
        &matB,
        static_cast<int64_t>(k), static_cast<int64_t>(n),
        static_cast<int64_t>(bnnz),
        const_cast<int32_t*>(bRowPtr32),
        const_cast<int32_t*>(bColIdx32),
        const_cast<void*>(bVal),
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO,
        valType),
    "cusparseCreateCsr(B)");

  // C starts with 0 nnz; colIdx and values pointers are NULL until after
  // compute (cuSPARSE fills rowPtr during compute, colIdx/values during copy).
  checkCusparse(
    cusparseCreateCsr(
        &matC,
        static_cast<int64_t>(m), static_cast<int64_t>(n),
        0LL,                                // nnz = 0 initially
        static_cast<void*>(cRowPtr32),       // pre-allocated rowPtr buffer
        nullptr,                             // colIdx: set after compute
        nullptr,                             // values: set after compute
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO,
        valType),
    "cusparseCreateCsr(C)");

  // ── 2. Create SpGEMM descriptor ───────────────────────────────────────────
  cusparseSpGEMMDescr_t desc;
  checkCusparse(cusparseSpGEMM_createDescr(&desc), "SpGEMM_createDescr");

  // ── 3. Work estimation — phase 1 (query buffer size) ─────────────────────
  size_t bufSz1 = 0;
  checkCusparse(
    cusparseSpGEMM_workEstimation(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, matB, &beta, matC,
        valType, CUSPARSE_SPGEMM_DEFAULT,
        desc, &bufSz1, nullptr),
    "SpGEMM_workEstimation(query)");

  void* buf1 = nullptr;
  if (bufSz1 > 0) {
    buf1 = memory::CudaMemoryPool::getInstance().allocate(bufSz1, deviceId, stream);
    if (!buf1) {
      cusparseSpGEMM_destroyDescr(desc);
      cusparseDestroySpMat(matA); cusparseDestroySpMat(matB); cusparseDestroySpMat(matC);
      THROW_EXCEPTION("csr_spgemm: CudaMemoryPool failed to allocate SpGEMM workspace-1");
    }
  }

  // ── 3b. Work estimation — phase 2 (fill buffer) ───────────────────────────
  checkCusparse(
    cusparseSpGEMM_workEstimation(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, matB, &beta, matC,
        valType, CUSPARSE_SPGEMM_DEFAULT,
        desc, &bufSz1, buf1),
    "SpGEMM_workEstimation(fill)");

  // ── 4. Compute — phase 1 (query buffer size) ─────────────────────────────
  size_t bufSz2 = 0;
  checkCusparse(
    cusparseSpGEMM_compute(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, matB, &beta, matC,
        valType, CUSPARSE_SPGEMM_DEFAULT,
        desc, &bufSz2, nullptr),
    "SpGEMM_compute(query)");

  void* buf2 = nullptr;
  if (bufSz2 > 0) {
    buf2 = memory::CudaMemoryPool::getInstance().allocate(bufSz2, deviceId, stream);
    if (!buf2) {
      if (buf1) memory::CudaMemoryPool::getInstance().free(buf1, deviceId, stream);
      cusparseSpGEMM_destroyDescr(desc);
      cusparseDestroySpMat(matA); cusparseDestroySpMat(matB); cusparseDestroySpMat(matC);
      THROW_EXCEPTION("csr_spgemm: CudaMemoryPool failed to allocate SpGEMM workspace-2");
    }
  }

  // ── 4b. Compute — phase 2 (symbolic + numeric, fills rowPtr, computes nnz) ─
  checkCusparse(
    cusparseSpGEMM_compute(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, matB, &beta, matC,
        valType, CUSPARSE_SPGEMM_DEFAULT,
        desc, &bufSz2, buf2),
    "SpGEMM_compute(fill)");

  // ── 5. Retrieve C size; assert nnz matches the pre-allocated output ────────
  int64_t cRows = 0, cCols = 0, cNnz = 0;
  checkCusparse(cusparseSpMatGetSize(matC, &cRows, &cCols, &cNnz), "SpMatGetSize");

  if (cNnz != static_cast<int64_t>(expectedCnnz)) {
    // Clean up before throwing.
    if (buf1) memory::CudaMemoryPool::getInstance().free(buf1, deviceId, stream);
    if (buf2) memory::CudaMemoryPool::getInstance().free(buf2, deviceId, stream);
    cusparseSpGEMM_destroyDescr(desc);
    cusparseDestroySpMat(matA); cusparseDestroySpMat(matB); cusparseDestroySpMat(matC);
    std::string msg =
        "csr_spgemm: nnz mismatch — DECLARE_SHAPE_FN (Gustavson) expected " +
        std::to_string(expectedCnnz) + " but cuSPARSE computed " + std::to_string(cNnz) +
        ". Both algorithms are deterministic; this is an implementation bug.";
    THROW_EXCEPTION(msg.c_str());
  }

  // ── 6. Bind pre-allocated output buffers (rowPtr already bound at creation) ─
  //    After this call matC carries pointers to the three output arrays.
  checkCusparse(
    cusparseCsrSetPointers(
        matC,
        static_cast<void*>(cRowPtr32),
        static_cast<void*>(cColIdx32),
        cVal),
    "cusparseCsrSetPointers");

  // ── 7. Copy — write numeric values and column indices into output arrays ───
  checkCusparse(
    cusparseSpGEMM_copy(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, matB, &beta, matC,
        valType, CUSPARSE_SPGEMM_DEFAULT,
        desc),
    "SpGEMM_copy");

  // ── 8. Cleanup ─────────────────────────────────────────────────────────────
  cusparseSpGEMM_destroyDescr(desc);
  cusparseDestroySpMat(matA);
  cusparseDestroySpMat(matB);
  cusparseDestroySpMat(matC);
  if (buf1) memory::CudaMemoryPool::getInstance().free(buf1, deviceId, stream);
  if (buf2) memory::CudaMemoryPool::getInstance().free(buf2, deviceId, stream);
}

// ── A.4  Main dispatch template ─────────────────────────────────────────────────
//
// Dispatched via BUILD_DOUBLE_SELECTOR over SD_FLOAT_TYPES × SD_INDEXING_TYPES.
// Called AFTER prepareSpecialUse has synced input device buffers.

template <typename X, typename I>
static void csrSpGEMMCuda_(NDArray& aValues, NDArray& aColIdx, NDArray& aRowPtr,
                            NDArray& bValues, NDArray& bColIdx, NDArray& bRowPtr,
                            NDArray& cValues, NDArray& cColIdx, NDArray& cRowPtr,
                            sd::LongType m, sd::LongType k, sd::LongType n) {
  auto* ctx    = cValues.getContext();
  auto  stream = ctx->getCudaStream();
  auto* handle = reinterpret_cast<cusparseHandle_t*>(ctx->getCusparseHandle());

  // Set the cuSPARSE stream once here; runSpGEMMDevice skips the redundant set.
  checkCusparse(cusparseSetStream(*handle, *stream), "cusparseSetStream");

  const int        devId = ctx->getDeviceID();
  const sd::LongType annz = aValues.lengthOf();
  const sd::LongType bnnz = bValues.lengthOf();
  const sd::LongType cnnz = cValues.lengthOf();  // pre-allocated by DECLARE_SHAPE_FN

  // ── Step 1: Resolve INT32 index device pointers ───────────────────────────
  //
  // cuSPARSE SpGEMM only supports CUSPARSE_INDEX_32I (requires CUDA ≥ 11.0;
  // INT64 support was not added as of CUDA 12.x for SpGEMM).
  //
  // If I is INT64 we cast the input index arrays to INT32 on-device using
  // NDArray::cast(), which runs a CUDA elementwise op.  The result NDArrays
  // own their device memory; they are destroyed at the end of this scope.
  //
  // The output index arrays (cColIdx, cRowPtr) are always INT32 per the op
  // contract (DECLARE_SHAPE_FN emits them as INT32), so no cast is needed there.

  std::unique_ptr<NDArray> aColIdxI32, aRowPtrI32, bColIdxI32, bRowPtrI32;

  const int32_t* aColIdx32;
  const int32_t* aRowPtr32;
  const int32_t* bColIdx32;
  const int32_t* bRowPtr32;

  if constexpr (sizeof(I) == 4) {
    // Already INT32 — use device buffers in-place; zero extra allocation.
    aColIdx32 = reinterpret_cast<const int32_t*>(aColIdx.specialBuffer());
    aRowPtr32 = reinterpret_cast<const int32_t*>(aRowPtr.specialBuffer());
    bColIdx32 = reinterpret_cast<const int32_t*>(bColIdx.specialBuffer());
    bRowPtr32 = reinterpret_cast<const int32_t*>(bRowPtr.specialBuffer());
  } else {
    // INT64 → INT32 on-device (no host round-trip).
    // Values fit in INT32 for matrices the op can legally receive (row/col
    // counts < 2^31 is an implicit prerequisite of cuSPARSE SpGEMM itself).
    aColIdxI32.reset(aColIdx.cast(sd::DataType::INT32));
    aRowPtrI32.reset(aRowPtr.cast(sd::DataType::INT32));
    bColIdxI32.reset(bColIdx.cast(sd::DataType::INT32));
    bRowPtrI32.reset(bRowPtr.cast(sd::DataType::INT32));

    // Declare cast temporaries as inputs so the coherence layer marks their
    // device buffers as ready before cuSPARSE reads them.
    NDArray::prepareSpecialUse({}, {aColIdxI32.get(), aRowPtrI32.get(),
                                     bColIdxI32.get(), bRowPtrI32.get()});

    aColIdx32 = reinterpret_cast<const int32_t*>(aColIdxI32->specialBuffer());
    aRowPtr32 = reinterpret_cast<const int32_t*>(aRowPtrI32->specialBuffer());
    bColIdx32 = reinterpret_cast<const int32_t*>(bColIdxI32->specialBuffer());
    bRowPtr32 = reinterpret_cast<const int32_t*>(bRowPtrI32->specialBuffer());
  }

  // Output index arrays are always INT32 (see op contract).
  int32_t* cColIdx32 = reinterpret_cast<int32_t*>(cColIdx.specialBuffer());
  int32_t* cRowPtr32 = reinterpret_cast<int32_t*>(cRowPtr.specialBuffer());

  // ── Step 2: Dispatch based on value type ─────────────────────────────────
  if constexpr (SpgemmNative<X>::value) {
    // ── 2a. Native path: float or double ─────────────────────────────────
    runSpGEMMDevice<X>(
        *handle, *stream, devId,
        aValues.specialBuffer(), aColIdx32, aRowPtr32, annz,
        bValues.specialBuffer(), bColIdx32, bRowPtr32, bnnz,
        cValues.specialBuffer(), cColIdx32, cRowPtr32,
        m, k, n, cnnz);

  } else {
    // ── 2b. Up-cast path: float16 or bfloat16 ────────────────────────────
    //
    // cuSPARSE SpGEMM does not support half or bfloat16 value types (as of
    // CUDA 12.x).  Strategy:
    //   1. Cast aValues and bValues to FLOAT32 on-device (CUDA elementwise op).
    //   2. Run cuSPARSE SpGEMM entirely in FLOAT32 into a temporary NDArray.
    //      cColIdx and cRowPtr are INT32 and written directly to the output.
    //   3. Cast the FLOAT32 temporary back to X and assign into cValues.
    // No host round-trip at any step.

    // Cast input values to float32 on-device.
    std::unique_ptr<NDArray> aValF32(aValues.cast(sd::DataType::FLOAT32));
    std::unique_ptr<NDArray> bValF32(bValues.cast(sd::DataType::FLOAT32));

    // Declare cast input temps as inputs so the coherence layer marks their
    // device buffers as ready before cuSPARSE reads them.
    NDArray::prepareSpecialUse({}, {aValF32.get(), bValF32.get()});

    // Allocate a temporary FLOAT32 output buffer for cValues.
    // NDArrayFactory::create allocates device memory on the same LaunchContext.
    std::unique_ptr<NDArray> cValF32(
        NDArrayFactory::create('c', std::vector<sd::LongType>{cnnz},
                               sd::DataType::FLOAT32, ctx));

    runSpGEMMDevice<float>(
        *handle, *stream, devId,
        aValF32->specialBuffer(), aColIdx32, aRowPtr32, annz,
        bValF32->specialBuffer(), bColIdx32, bRowPtr32, bnnz,
        cValF32->specialBuffer(), cColIdx32, cRowPtr32,
        m, k, n, cnnz);

    // Inform the coherence layer that cuSPARSE has written cValF32's device
    // buffer, making it authoritative for the subsequent assign into cValues.
    NDArray::registerSpecialUse({cValF32.get()}, {});

    // Cast float32 → X on-device and assign into the pre-allocated cValues.
    // NDArray::assign internally routes through a CUDA cast kernel when dtypes
    // differ, with no host involvement.
    cValues.assign(cValF32.get());
  }
}

#endif  // HAVE_CUSPARSE

// ═══════════════════════════════════════════════════════════════════════════════
// Section B: Public entry point
// ═══════════════════════════════════════════════════════════════════════════════

void csr_spgemm(NDArray& aValues, NDArray& aColIdx, NDArray& aRowPtr,
                NDArray& bValues, NDArray& bColIdx, NDArray& bRowPtr,
                NDArray& cValues, NDArray& cColIdx, NDArray& cRowPtr,
                sd::LongType m, sd::LongType k, sd::LongType n) {
  NDArray::prepareSpecialUse({&cValues, &cColIdx, &cRowPtr},
                             {&aValues, &aColIdx, &aRowPtr,
                              &bValues, &bColIdx, &bRowPtr});

#if defined(HAVE_CUSPARSE)
  BUILD_DOUBLE_SELECTOR(aValues.dataType(), aColIdx.dataType(), csrSpGEMMCuda_,
                        (aValues, aColIdx, aRowPtr,
                         bValues, bColIdx, bRowPtr,
                         cValues, cColIdx, cRowPtr,
                         m, k, n),
                        SD_FLOAT_TYPES, SD_INDEXING_TYPES);
#else
  // Without cuSPARSE there is no on-GPU implementation.  The host-side
  // placeholder was removed deliberately; the CPU helper in
  // cpu/sparse_spgemm.cpp handles the CPU backend.
  THROW_EXCEPTION(
      "csr_spgemm CUDA: cuSPARSE is required but was not compiled in "
      "(HAVE_CUSPARSE not defined). Rebuild with -Dlibnd4j.cusparse=ON "
      "or use the CPU backend.");
#endif

  NDArray::registerSpecialUse({&cValues, &cColIdx, &cRowPtr},
                              {&aValues, &aColIdx, &aRowPtr,
                               &bValues, &bColIdx, &bRowPtr});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd

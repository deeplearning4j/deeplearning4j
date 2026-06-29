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
// CSR sparse-sparse elementwise addition — CUDA implementation.
//
// C = A + B, A and B both CSR [m, n] with the SAME logical shape.
//
// Design notes
// ────────────
// 1. OUTPUT PRE-ALLOCATION: DECLARE_SHAPE_FN runs a host-side symbolic union
//    pass and pre-allocates cValues[cnnz], cColIdx[cnnz] (INT32), cRowPtr[m+1]
//    (INT32) with the exact final nnz.  cuSPARSE's cusparseXcsrgeam2Nnz runs its
//    own symbolic pass; its nnz is asserted equal to cnnz.  On mismatch we THROW
//    (no silent truncation, no fallback).
//
// 2. INDEX TYPE: The legacy csrgeam2 API requires int (32-bit) index arrays.
//    If the template index type I is 64-bit, the input index arrays are cast to
//    INT32 on-device via NDArray::cast() with no host round-trip.  Output index
//    arrays (cColIdx, cRowPtr) are always INT32 per the op contract.
//
// 3. VALUE TYPE: float and double are natively supported by cusparseScsrgeam2
//    and cusparseDcsrgeam2 respectively.  float16 / bfloat16 are NOT supported
//    by csrgeam2; they are cast to float on-device, the addition is executed in
//    float into a temporary cValues buffer, and the result is cast back to X
//    on-device.  No host round-trip at any step.
//
// 4. WORKSPACE: all temporary GPU memory is allocated from CudaMemoryPool
//    (no raw cudaMalloc).
//
// 5. NO FALLBACK: any cuSPARSE error or nnz mismatch throws immediately.
//
// cuSPARSE GEAM2 call sequence (for float; double replaces S→D):
//   cusparseScsrgeam2_bufferSizeExt  → bufSize
//   CudaMemoryPool::allocate(bufSize) → workspace
//   cusparseXcsrgeam2Nnz             → fills cRowPtr[m+1], nnzC (host int)
//   ASSERT nnzC == expectedCnnz
//   cusparseScsrgeam2                → fills cColIdx[cnnz], cValues[cnnz]
//   CudaMemoryPool::free(workspace)
//

#include <cuda_runtime.h>
#include <memory/cuda/CudaMemoryPool.h>
#include <ops/declarable/helpers/sparse_add.h>
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

namespace {  // anonymous — these helpers are local to this translation unit

// ── A.1  Status-checking helper ────────────────────────────────────────────────

SD_INLINE void checkGeam(cusparseStatus_t st, const char* where) {
  if (st != CUSPARSE_STATUS_SUCCESS) {
    std::string msg = std::string("csr_add cuSPARSE error at [") + where +
                      "] status=" + std::to_string(static_cast<int>(st));
    THROW_EXCEPTION(msg.c_str());
  }
}

// ── A.2  Native-type predicate ─────────────────────────────────────────────────
// True iff FloatT maps directly to a csrgeam2 variant (S or D).
template <typename T> struct GeamNative { static constexpr bool value = false; };
template <> struct GeamNative<float>  { static constexpr bool value = true;  };
template <> struct GeamNative<double> { static constexpr bool value = true;  };

// ── A.3  Core geam2 routine for float ─────────────────────────────────────────
//
// Runs the full cuSPARSE geam2 flow for float data:
//   bufferSizeExt → allocate workspace → Xcsrgeam2Nnz (symbolic, fills cRowPtr)
//   → assert nnz → Scsrgeam2 (numeric, fills cColIdx + cValues)
//
static void runGeam2Float(
    cusparseHandle_t     handle,
    cusparseMatDescr_t   descrA,
    cusparseMatDescr_t   descrB,
    cusparseMatDescr_t   descrC,
    int                  m,
    int                  n,
    int                  nnzA,
    const float*         aVal,
    const int*           aRowPtr32,
    const int*           aColIdx32,
    int                  nnzB,
    const float*         bVal,
    const int*           bRowPtr32,
    const int*           bColIdx32,
    float*               cVal,
    int*                 cRowPtr32,    // pre-allocated [m+1]
    int*                 cColIdx32,    // pre-allocated [cnnz]
    sd::LongType         expectedCnnz,
    int                  deviceId,
    cudaStream_t         stream)
{
  static const float alpha = 1.0f, beta = 1.0f;

  // ── 1. Query workspace size ───────────────────────────────────────────────
  size_t bufSize = 0;
  checkGeam(
    cusparseScsrgeam2_bufferSizeExt(
        handle, m, n,
        &alpha, descrA, nnzA, aVal, aRowPtr32, aColIdx32,
        &beta,  descrB, nnzB, bVal, bRowPtr32, bColIdx32,
        descrC, cVal, cRowPtr32, cColIdx32,
        &bufSize),
    "Scsrgeam2_bufferSizeExt");

  // ── 2. Allocate workspace from pool ──────────────────────────────────────
  // Allocate at least 1 byte so the pointer is non-null (some cuSPARSE
  // versions segfault on a null workspace even when bufSize==0).
  const size_t allocSz = (bufSize > 0) ? bufSize : 1;
  void* workspace = memory::CudaMemoryPool::getInstance().allocate(
      allocSz, deviceId, stream);
  if (!workspace) {
    THROW_EXCEPTION("csr_add: CudaMemoryPool failed to allocate geam2 workspace");
  }

  // ── 3. Symbolic pass: fill cRowPtr and get nnzC ──────────────────────────
  int nnzC = 0;
  checkGeam(
    cusparseXcsrgeam2Nnz(
        handle, m, n,
        descrA, nnzA, aRowPtr32, aColIdx32,
        descrB, nnzB, bRowPtr32, bColIdx32,
        descrC, cRowPtr32, &nnzC,
        workspace),
    "Xcsrgeam2Nnz");

  // ── 4. Assert nnzC matches the pre-allocated output ───────────────────────
  if (static_cast<sd::LongType>(nnzC) != expectedCnnz) {
    memory::CudaMemoryPool::getInstance().free(workspace, deviceId, stream);
    std::string msg =
        "csr_add: nnz mismatch — DECLARE_SHAPE_FN (symbolic union) expected " +
        std::to_string(expectedCnnz) +
        " but cuSPARSE Xcsrgeam2Nnz computed " + std::to_string(nnzC) +
        ". Both algorithms are deterministic; this is an implementation bug.";
    THROW_EXCEPTION(msg.c_str());
  }

  // ── 5. Numeric pass: write cColIdx and cValues ───────────────────────────
  checkGeam(
    cusparseScsrgeam2(
        handle, m, n,
        &alpha, descrA, nnzA, aVal, aRowPtr32, aColIdx32,
        &beta,  descrB, nnzB, bVal, bRowPtr32, bColIdx32,
        descrC, cVal, cRowPtr32, cColIdx32,
        workspace),
    "Scsrgeam2");

  // ── 6. Release workspace ─────────────────────────────────────────────────
  memory::CudaMemoryPool::getInstance().free(workspace, deviceId, stream);
}

// ── A.4  Core geam2 routine for double ────────────────────────────────────────
// Mirrors runGeam2Float exactly, substituting D for S.

static void runGeam2Double(
    cusparseHandle_t     handle,
    cusparseMatDescr_t   descrA,
    cusparseMatDescr_t   descrB,
    cusparseMatDescr_t   descrC,
    int                  m,
    int                  n,
    int                  nnzA,
    const double*        aVal,
    const int*           aRowPtr32,
    const int*           aColIdx32,
    int                  nnzB,
    const double*        bVal,
    const int*           bRowPtr32,
    const int*           bColIdx32,
    double*              cVal,
    int*                 cRowPtr32,
    int*                 cColIdx32,
    sd::LongType         expectedCnnz,
    int                  deviceId,
    cudaStream_t         stream)
{
  static const double alpha = 1.0, beta = 1.0;

  size_t bufSize = 0;
  checkGeam(
    cusparseDcsrgeam2_bufferSizeExt(
        handle, m, n,
        &alpha, descrA, nnzA, aVal, aRowPtr32, aColIdx32,
        &beta,  descrB, nnzB, bVal, bRowPtr32, bColIdx32,
        descrC, cVal, cRowPtr32, cColIdx32,
        &bufSize),
    "Dcsrgeam2_bufferSizeExt");

  const size_t allocSz = (bufSize > 0) ? bufSize : 1;
  void* workspace = memory::CudaMemoryPool::getInstance().allocate(
      allocSz, deviceId, stream);
  if (!workspace) {
    THROW_EXCEPTION("csr_add: CudaMemoryPool failed to allocate geam2 workspace");
  }

  int nnzC = 0;
  checkGeam(
    cusparseXcsrgeam2Nnz(
        handle, m, n,
        descrA, nnzA, aRowPtr32, aColIdx32,
        descrB, nnzB, bRowPtr32, bColIdx32,
        descrC, cRowPtr32, &nnzC,
        workspace),
    "Xcsrgeam2Nnz");

  if (static_cast<sd::LongType>(nnzC) != expectedCnnz) {
    memory::CudaMemoryPool::getInstance().free(workspace, deviceId, stream);
    std::string msg =
        "csr_add: nnz mismatch — DECLARE_SHAPE_FN (symbolic union) expected " +
        std::to_string(expectedCnnz) +
        " but cuSPARSE Xcsrgeam2Nnz computed " + std::to_string(nnzC) +
        ". Both algorithms are deterministic; this is an implementation bug.";
    THROW_EXCEPTION(msg.c_str());
  }

  checkGeam(
    cusparseDcsrgeam2(
        handle, m, n,
        &alpha, descrA, nnzA, aVal, aRowPtr32, aColIdx32,
        &beta,  descrB, nnzB, bVal, bRowPtr32, bColIdx32,
        descrC, cVal, cRowPtr32, cColIdx32,
        workspace),
    "Dcsrgeam2");

  memory::CudaMemoryPool::getInstance().free(workspace, deviceId, stream);
}

}  // anonymous namespace

// ── A.5  Main dispatch template ──────────────────────────────────────────────
//
// Dispatched via BUILD_DOUBLE_SELECTOR over SD_FLOAT_TYPES × SD_INDEXING_TYPES.
// Called AFTER prepareSpecialUse has synced input device buffers.

template <typename X, typename I>
static void csrAddCuda_(NDArray& aValues, NDArray& aColIdx, NDArray& aRowPtr,
                        NDArray& bValues, NDArray& bColIdx, NDArray& bRowPtr,
                        NDArray& cValues, NDArray& cColIdx, NDArray& cRowPtr,
                        sd::LongType m, sd::LongType n) {
  auto* ctx    = cValues.getContext();
  auto  stream = ctx->getCudaStream();
  auto* handle = reinterpret_cast<cusparseHandle_t*>(ctx->getCusparseHandle());

  const sd::LongType aNnz = aValues.lengthOf();
  const sd::LongType bNnz = bValues.lengthOf();
  const sd::LongType cnnz = cValues.lengthOf();  // from DECLARE_SHAPE_FN

  // ── Special case: 0-nnz operand(s) ─────────────────────────────────────────
  // cusparseXcsrgeam2Nnz / csrgeam2 do not tolerate length-0 arrays whose
  // specialBuffer() may be null.  Handle all zero-nnz combinations here, before
  // any cuSPARSE call, using pure D2D cudaMemcpyAsync (no host round-trip, no
  // raw cudaMalloc).
  if (aNnz == 0 || bNnz == 0) {
    if (aNnz > 0 || bNnz > 0) {
      // Exactly one operand is non-empty.  C = that operand.
      NDArray& srcValues = (aNnz == 0) ? bValues : aValues;
      NDArray& srcColIdx = (aNnz == 0) ? bColIdx : aColIdx;
      NDArray& srcRowPtr = (aNnz == 0) ? bRowPtr : aRowPtr;
      const sd::LongType srcNnz = (aNnz == 0) ? bNnz : aNnz;

      // Values: both dtype X, direct D2D copy.
      cudaMemcpyAsync(cValues.specialBuffer(), srcValues.specialBuffer(),
                      static_cast<size_t>(srcNnz) * sizeof(X),
                      cudaMemcpyDeviceToDevice, *stream);

      // Index arrays: src dtype is I, output is always INT32 per op contract.
      if constexpr (sizeof(I) == 4) {
        // I == INT32: direct D2D copy for both colIdx and rowPtr.
        cudaMemcpyAsync(cColIdx.specialBuffer(), srcColIdx.specialBuffer(),
                        static_cast<size_t>(srcNnz) * sizeof(int),
                        cudaMemcpyDeviceToDevice, *stream);
        cudaMemcpyAsync(cRowPtr.specialBuffer(), srcRowPtr.specialBuffer(),
                        static_cast<size_t>(m + 1) * sizeof(int),
                        cudaMemcpyDeviceToDevice, *stream);
      } else {
        // I == INT64: cast on-device to INT32, then D2D copy (no host round-trip).
        std::unique_ptr<NDArray> srcColIdxI32(srcColIdx.cast(sd::DataType::INT32));
        std::unique_ptr<NDArray> srcRowPtrI32(srcRowPtr.cast(sd::DataType::INT32));
        // cast() ran a CUDA kernel; make the device buffers visible as reads.
        NDArray::prepareSpecialUse({}, {srcColIdxI32.get(), srcRowPtrI32.get()});
        cudaMemcpyAsync(cColIdx.specialBuffer(), srcColIdxI32->specialBuffer(),
                        static_cast<size_t>(srcNnz) * sizeof(int),
                        cudaMemcpyDeviceToDevice, *stream);
        cudaMemcpyAsync(cRowPtr.specialBuffer(), srcRowPtrI32->specialBuffer(),
                        static_cast<size_t>(m + 1) * sizeof(int),
                        cudaMemcpyDeviceToDevice, *stream);
      }
    }
    // Both-empty: cRowPtr/cColIdx/cValues are pre-zeroed/length-0; nothing to do.
    return;
  }

  // Use host-side pointer mode so alpha/beta and nnzC are plain host values.
  checkGeam(cusparseSetPointerMode(*handle, CUSPARSE_POINTER_MODE_HOST),
            "cusparseSetPointerMode(HOST)");
  checkGeam(cusparseSetStream(*handle, *stream), "cusparseSetStream");

  const int          devId    = ctx->getDeviceID();
  const int          m32      = static_cast<int>(m);
  const int          n32      = static_cast<int>(n);
  const int          annz32   = static_cast<int>(aNnz);
  const int          bnnz32   = static_cast<int>(bNnz);

  // ── Step 1: Resolve INT32 index device pointers ───────────────────────────
  // cusparseXcsrgeam2Nnz / cusparseScsrgeam2 require int (32-bit) arrays.
  // If I == INT64, cast on-device; no host round-trip.

  std::unique_ptr<NDArray> aColIdxI32, aRowPtrI32, bColIdxI32, bRowPtrI32;

  const int* aColIdx32;
  const int* aRowPtr32;
  const int* bColIdx32;
  const int* bRowPtr32;

  if constexpr (sizeof(I) == 4) {
    aColIdx32 = reinterpret_cast<const int*>(aColIdx.specialBuffer());
    aRowPtr32 = reinterpret_cast<const int*>(aRowPtr.specialBuffer());
    bColIdx32 = reinterpret_cast<const int*>(bColIdx.specialBuffer());
    bRowPtr32 = reinterpret_cast<const int*>(bRowPtr.specialBuffer());
  } else {
    // INT64 → INT32 on-device (no host round-trip).
    aColIdxI32.reset(aColIdx.cast(sd::DataType::INT32));
    aRowPtrI32.reset(aRowPtr.cast(sd::DataType::INT32));
    bColIdxI32.reset(bColIdx.cast(sd::DataType::INT32));
    bRowPtrI32.reset(bRowPtr.cast(sd::DataType::INT32));

    // cast() ran CUDA kernels; make the device buffers visible as reads
    // before any cuSPARSE call consumes them.
    NDArray::prepareSpecialUse({}, {aColIdxI32.get(), aRowPtrI32.get(),
                                     bColIdxI32.get(), bRowPtrI32.get()});

    aColIdx32 = reinterpret_cast<const int*>(aColIdxI32->specialBuffer());
    aRowPtr32 = reinterpret_cast<const int*>(aRowPtrI32->specialBuffer());
    bColIdx32 = reinterpret_cast<const int*>(bColIdxI32->specialBuffer());
    bRowPtr32 = reinterpret_cast<const int*>(bRowPtrI32->specialBuffer());
  }

  // Output index arrays are always INT32 per op contract.
  int* cColIdx32 = reinterpret_cast<int*>(cColIdx.specialBuffer());
  int* cRowPtr32 = reinterpret_cast<int*>(cRowPtr.specialBuffer());

  // ── Step 2: Create cuSPARSE matrix descriptors ────────────────────────────
  cusparseMatDescr_t descrA = nullptr, descrB = nullptr, descrC = nullptr;
  checkGeam(cusparseCreateMatDescr(&descrA), "cusparseCreateMatDescr(A)");
  checkGeam(cusparseCreateMatDescr(&descrB), "cusparseCreateMatDescr(B)");
  checkGeam(cusparseCreateMatDescr(&descrC), "cusparseCreateMatDescr(C)");

  // All three matrices: general storage, zero base index.
  cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
  cusparseSetMatType(descrB, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descrB, CUSPARSE_INDEX_BASE_ZERO);
  cusparseSetMatType(descrC, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descrC, CUSPARSE_INDEX_BASE_ZERO);

  // ── Step 3: Dispatch based on value type ─────────────────────────────────
  if constexpr (GeamNative<X>::value) {
    // ── 3a. Native path: float or double ─────────────────────────────────

    if constexpr (std::is_same<X, float>::value) {
      runGeam2Float(
          *handle, descrA, descrB, descrC, m32, n32,
          annz32,
          reinterpret_cast<const float*>(aValues.specialBuffer()),
          aRowPtr32, aColIdx32,
          bnnz32,
          reinterpret_cast<const float*>(bValues.specialBuffer()),
          bRowPtr32, bColIdx32,
          reinterpret_cast<float*>(cValues.specialBuffer()),
          cRowPtr32, cColIdx32,
          cnnz, devId, *stream);
    } else {
      // double
      runGeam2Double(
          *handle, descrA, descrB, descrC, m32, n32,
          annz32,
          reinterpret_cast<const double*>(aValues.specialBuffer()),
          aRowPtr32, aColIdx32,
          bnnz32,
          reinterpret_cast<const double*>(bValues.specialBuffer()),
          bRowPtr32, bColIdx32,
          reinterpret_cast<double*>(cValues.specialBuffer()),
          cRowPtr32, cColIdx32,
          cnnz, devId, *stream);
    }

  } else {
    // ── 3b. Up-cast path: float16 or bfloat16 ────────────────────────────
    //
    // csrgeam2 has no half/bfloat16 variant.  Strategy:
    //   1. Cast aValues and bValues to FLOAT32 on-device.
    //   2. Run Scsrgeam2 into a temporary FLOAT32 cValues NDArray.
    //      cColIdx and cRowPtr are INT32 and written directly to the output.
    //   3. Cast the FLOAT32 result back to X and assign into cValues.
    // No host round-trip at any step.

    std::unique_ptr<NDArray> aValF32(aValues.cast(sd::DataType::FLOAT32));
    std::unique_ptr<NDArray> bValF32(bValues.cast(sd::DataType::FLOAT32));

    // Temporary FLOAT32 output buffer for cValues.
    std::unique_ptr<NDArray> cValF32(
        NDArrayFactory::create('c', std::vector<sd::LongType>{cnnz},
                               sd::DataType::FLOAT32, ctx));

    // cast() ran CUDA kernels; make device buffers visible as inputs.
    // Also registers cValF32 as the device-side output for cuSPARSE.
    NDArray::prepareSpecialUse({cValF32.get()}, {aValF32.get(), bValF32.get()});

    runGeam2Float(
        *handle, descrA, descrB, descrC, m32, n32,
        annz32,
        reinterpret_cast<const float*>(aValF32->specialBuffer()),
        aRowPtr32, aColIdx32,
        bnnz32,
        reinterpret_cast<const float*>(bValF32->specialBuffer()),
        bRowPtr32, bColIdx32,
        reinterpret_cast<float*>(cValF32->specialBuffer()),
        cRowPtr32, cColIdx32,
        cnnz, devId, *stream);

    // cuSPARSE wrote cValF32 on-device; tell the framework the device buffer
    // is authoritative so that assign() below can read it correctly.
    NDArray::registerSpecialUse({cValF32.get()}, {aValF32.get(), bValF32.get()});

    // Cast float32 → X on-device and write into the pre-allocated cValues.
    cValues.assign(cValF32.get());
  }

  // ── Step 4: Cleanup descriptors ───────────────────────────────────────────
  cusparseDestroyMatDescr(descrA);
  cusparseDestroyMatDescr(descrB);
  cusparseDestroyMatDescr(descrC);
}

#endif  // HAVE_CUSPARSE

// ═══════════════════════════════════════════════════════════════════════════════
// Section B: Public entry point
// ═══════════════════════════════════════════════════════════════════════════════

void csr_add(NDArray& aValues, NDArray& aColIdx, NDArray& aRowPtr,
             NDArray& bValues, NDArray& bColIdx, NDArray& bRowPtr,
             NDArray& cValues, NDArray& cColIdx, NDArray& cRowPtr,
             sd::LongType m, sd::LongType n) {
  NDArray::prepareSpecialUse({&cValues, &cColIdx, &cRowPtr},
                             {&aValues, &aColIdx, &aRowPtr,
                              &bValues, &bColIdx, &bRowPtr});

#if defined(HAVE_CUSPARSE)
  BUILD_DOUBLE_SELECTOR(aValues.dataType(), aColIdx.dataType(), csrAddCuda_,
                        (aValues, aColIdx, aRowPtr,
                         bValues, bColIdx, bRowPtr,
                         cValues, cColIdx, cRowPtr,
                         m, n),
                        SD_FLOAT_TYPES, SD_INDEXING_TYPES);
#else
  THROW_EXCEPTION(
      "csr_add CUDA: cuSPARSE is required but was not compiled in "
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

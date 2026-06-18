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
// @author raver119@gmail.com
// @author Yurii Shyrma (iuriish@yahoo.com)
//
#include "../MmulHelper.h"

#include <array/NDArrayFactory.h>

#include <execution/Threads.h>
#include <helpers/BlasHelper.h>
#include <helpers/ShapeUtils.h>
#include <helpers/MklBlasHelper.h>
#include <system/Environment.h>
#include <system/env_functions.h>
#include <system/openmp_pragmas.h>
namespace sd {

// CPU stubs for cast cache methods (only used on CUDA)
void MmulHelper::clearCastCache() {}
void MmulHelper::resetCastCacheIndices() {}


//////////////////////////////////////////////////////////////////////////////
// MXK x KxN = MxN              -> actual sequence of axes doesn't matter
template <typename T1, typename T2, typename T3>
static void usualGemm(NDArray* vA, NDArray* vB, NDArray* vC, const int aMaxis, const int aKaxis,
                      const int bKaxis, const int bNaxis, const int cMaxis, const int cNaxis, const double alpha,
                      const double beta) {
  // Validate type sizes match template types to prevent buffer overrun from type-punned pointer arithmetic.
  // usualGemm reinterprets buffer memory as T1/T2/T3 pointers; if the actual element size differs,
  // pointer arithmetic advances by sizeof(T) bytes but elements are sizeof(actual) bytes apart,
  // producing out-of-bounds accesses (SEGV_ACCERR on guard page).
  if (sizeof(T1) != DataTypeUtils::sizeOf(vA->dataType()))
    THROW_EXCEPTION("usualGemm: type size mismatch for A — pointer arithmetic would overrun buffer");
  if (sizeof(T2) != DataTypeUtils::sizeOf(vB->dataType()))
    THROW_EXCEPTION("usualGemm: type size mismatch for B — pointer arithmetic would overrun buffer");
  if (sizeof(T3) != DataTypeUtils::sizeOf(vC->dataType()))
    THROW_EXCEPTION("usualGemm: type size mismatch for C — pointer arithmetic would overrun buffer");

  T1* A = vA->bufferAsT<T1>();
  T2* B = vB->bufferAsT<T2>();
  T3* C = vC->bufferAsT<T3>();
  if (A == nullptr) {
    THROW_EXCEPTION("usualGemm: A is nullptr");
  }
  if (B == nullptr) {
    THROW_EXCEPTION("usualGemm: B is nullptr");
  }
  if (C == nullptr) {
    THROW_EXCEPTION("usualGemm: C is nullptr");
  }

  const T3 alphaZ = static_cast<T3>(alpha);
  const T3 betaZ = static_cast<T3>(beta);

  const bool betaPresent = beta;

  const sd::LongType* aShapeInfo = vA->shapeInfo();
  const sd::LongType* bShapeInfo = vB->shapeInfo();
  const sd::LongType* cShapeInfo = vC->shapeInfo();

  const int aRank = vA->rankOf();
  const int bRank = vB->rankOf();
  const int cRank = vC->rankOf();
  const int M = vC->sizeAt(cMaxis);
  const int N = vC->sizeAt(cNaxis);
  const int K = vA->sizeAt(aKaxis);

  sd::LongType *aStride = shape::stride(aShapeInfo);
  sd::LongType *bStride = shape::stride(bShapeInfo);
  sd::LongType *cStride = shape::stride(cShapeInfo);

  // Check array layouts for fast paths
  // Row-major: stride[last]=1, stride[first]=num_cols
  // Column-major: stride[first]=1, stride[last]=num_rows
  const bool aRowMajor = (aStride[aKaxis] == 1) && (aStride[aMaxis] == K);
  const bool bRowMajor = (bStride[bNaxis] == 1) && (bStride[bKaxis] == N);
  const bool cRowMajor = (cStride[cNaxis] == 1) && (cStride[cMaxis] == N);
  const bool cColMajor = (cStride[cMaxis] == 1) && (cStride[cNaxis] == M);

  if (aRowMajor && bRowMajor && cRowMajor) {
    // Fast path: all row-major - cache-friendly loop order
    auto func = PRAGMA_THREADS_FOR {
      for (auto m = start; m < stop; ++m) {
        T1* aRow = A + m * K;
        T3* cRow = C + m * N;

        // Initialize output row
        if (betaPresent) {
          PRAGMA_OMP_SIMD
          for (int n = 0; n < N; ++n) {
            cRow[n] *= betaZ;
          }
        } else {
          PRAGMA_OMP_SIMD
          for (int n = 0; n < N; ++n) {
            cRow[n] = static_cast<T3>(0);
          }
        }

        // Cache-friendly loop order: k outer, n inner
        for (int k = 0; k < K; ++k) {
          const T3 aVal = alphaZ * static_cast<T3>(aRow[k]);
          const T2* bRow = B + k * N;
          PRAGMA_OMP_SIMD
          for (int n = 0; n < N; ++n) {
            cRow[n] += aVal * static_cast<T3>(bRow[n]);
          }
        }
      }
    };
    samediff::Threads::parallel_tad(func, 0, M);
  } else if (aRowMajor && bRowMajor && cColMajor) {
    // Fast path: A,B row-major, C column-major (common from tensorDot)
    // C column-major means C[m,n] is at offset m + n*M
    auto func = PRAGMA_THREADS_FOR {
      for (auto m = start; m < stop; ++m) {
        T1* aRow = A + m * K;

        // Initialize output column elements for this row
        if (betaPresent) {
          PRAGMA_OMP_SIMD
          for (int n = 0; n < N; ++n) {
            C[m + n * M] *= betaZ;
          }
        } else {
          PRAGMA_OMP_SIMD
          for (int n = 0; n < N; ++n) {
            C[m + n * M] = static_cast<T3>(0);
          }
        }

        // k outer for A cache locality, n inner for B cache locality
        for (int k = 0; k < K; ++k) {
          const T3 aVal = alphaZ * static_cast<T3>(aRow[k]);
          const T2* bRow = B + k * N;
          PRAGMA_OMP_SIMD
          for (int n = 0; n < N; ++n) {
            C[m + n * M] += aVal * static_cast<T3>(bRow[n]);
          }
        }
      }
    };
    samediff::Threads::parallel_tad(func, 0, M);
  } else {
    // General strided path - original implementation
    const sd::LongType cLen = vC->lengthOf();

    auto func = PRAGMA_THREADS_FOR {
      std::vector<sd::LongType> aCoords(aRank), bCoords(bRank), cCoords(cRank);

      for (auto i = start; i < stop; i++) {
        // evaluate C coordinates
        INDEX2COORDS(i, cRank, shape::shapeOf(cShapeInfo), cCoords.data());

        // evaluate A coordinates
        aCoords[aMaxis] = cCoords[cMaxis];
        aCoords[aKaxis] = 0;

        // evaluate B coordinates
        bCoords[bKaxis] = 0;
        bCoords[bNaxis] = cCoords[cNaxis];

        sd::LongType aOffset, bOffset, cOffset;
        COORDS2INDEX(aRank, aStride, aCoords.data(), aOffset);
        COORDS2INDEX(bRank, bStride, bCoords.data(), bOffset);

        T3 val = A[aOffset] * B[bOffset];  // first iteration

        for (int j = 1; j < K; j++) {  // rest iterations
          aOffset += aStride[aKaxis];
          bOffset += bStride[bKaxis];
          val += A[aOffset] * B[bOffset];
        }

        COORDS2INDEX(cRank, cStride, cCoords.data(), cOffset);
        if (betaPresent) {
          C[cOffset] = alphaZ * val + betaZ * C[cOffset];
        } else {
          C[cOffset] = alphaZ * val;
        }
      }
    };
    samediff::Threads::parallel_tad(func, 0, cLen);
  }
}

//////////////////////////////////////////////////////////////////////////////
// MXN x N = M  -> actual sequence of {M,N} axes doesn't matter
template <typename T1, typename T2, typename T3>
static void usualGemv( NDArray* vA, NDArray* vX, NDArray* vY, const int incx, const int incy,
                       const int aMaxis, const double alpha, const double beta) {
  T1* A = vA->bufferAsT<T1>();
  T2* X = vX->bufferAsT<T2>();
  T3* Y = vY->bufferAsT<T3>();

  const T3 alphaZ = static_cast<T3>(alpha);
  const T3 betaZ = static_cast<T3>(beta);

  const bool betaPersent = beta;

  const sd::LongType* aShapeInfo = vA->shapeInfo();
  const sd::LongType* xShapeInfo = vX->shapeInfo();
  const sd::LongType* yShapeInfo = vY->shapeInfo();

  const int N = vX->lengthOf();
  const int M = vY->lengthOf();

  const auto aMstride = vA->strideAt(aMaxis);
  const auto aNstride = vA->strideAt(aMaxis == 0 ? 1 : 0);

  auto func = PRAGMA_THREADS_FOR {
    for (auto i = start; i < stop; ++i) {
      // evaluate offsets
      auto aOffset = i * aMstride;
      auto xOffset = 0;

      T3 val = A[aOffset] * X[xOffset];  // first iteration

      for (int j = 1; j < N; ++j) {  // rest iterations
        aOffset += aNstride;
        xOffset += incx;
        val = val + A[aOffset] * X[xOffset];
      }

      auto yOffset = i * incy;

      if (betaPersent)
        Y[yOffset] = alphaZ * val + betaZ * Y[yOffset];
      else
        Y[yOffset] = alphaZ * val;
    }
  };

  samediff::Threads::parallel_tad(func, 0, M);
}

//////////////////////////////////////////////////////////////////////////////
// (X*Y) = Z[0]
template <typename T1, typename T2, typename T3>
static void usualDot(const sd::LongType length, const double alpha, const void* vX, const sd::LongType incx,
                     const void* vY, const sd::LongType incy, const double beta, void* vZ) {
  T1* X = reinterpret_cast<T1*>(const_cast<void*>(vX));
  T2* Y = reinterpret_cast<T2*>(const_cast<void*>(vY));
  T3* Z = reinterpret_cast<T3*>(vZ);
  T3 alphaZ(alpha), betaZ(beta);

  const bool betaPersent = beta;

  T3 sum = static_cast<T3>(0);

  auto func = PRAGMA_THREADS_FOR {
    for (sd::LongType i = start; i < stop; ++i) {
      sum += X[i * incx] * Y[i * incy];
    }
  };

  samediff::Threads::parallel_for(func, 0, length);
  if (betaPersent)
    *Z = alphaZ * sum + betaZ * *Z;
  else
    *Z = alphaZ * sum;
}

//////////////////////////////////////////////////////////////////////////////
// MXK x KxN = MxN
NDArray* MmulHelper::mmulMxM( NDArray* A,  NDArray* B, NDArray* C, const double alpha, const double beta,
                              const char outOrder) {

  // Cache dimensions to avoid repeated virtual calls
  const auto M = A->sizeAt(0);
  const auto K = A->sizeAt(1);
  const auto N = B->sizeAt(1);

  // Validate inputs (lazy error message construction)
  if (C != nullptr && C->rankOf() != 2) {
    THROW_EXCEPTION("MmulHelper::mmulMxM: rank of C array should be equal to 2");
  }
  if (B->sizeAt(0) != K) {
    THROW_EXCEPTION("MmulHelper::mmulMxM: B rows must equal A columns");
  }
  if (C != nullptr && (C->sizeAt(0) != M || C->sizeAt(1) != N)) {
    THROW_EXCEPTION("MmulHelper::mmulMxM: C shape must match [M, N]");
  }

  if (C == nullptr) {
    std::vector<sd::LongType> shape = {M, N};
    C = new NDArray(outOrder, shape, DataTypeUtils::pickPairwiseResultType(A->dataType(), B->dataType()),
                    A->getContext());
  }
  if (C->isEmpty()) return C;

  const auto aType = A->dataType();
  const auto bType = B->dataType();
  const auto cType = C->dataType();

  const bool ABC = (aType == bType) && (aType == cType);

  // Fast path: check for row-major contiguous arrays (most common case in BERT)
  // Row-major means stride(-1)=1 and stride(-2)=ncols
  const bool aRowMajor = (A->strideAt(1) == 1) && (A->strideAt(0) == K);
  const bool bRowMajor = (B->strideAt(1) == 1) && (B->strideAt(0) == N);
  const bool cRowMajor = (C->strideAt(1) == 1) && (C->strideAt(0) == N);

  if (ABC && aRowMajor && bRowMajor && cRowMajor && sd::env_isEnableBlas()) {
    // Validate dimensions before BLAS call to prevent crashes
    if (M <= 0 || N <= 0 || K <= 0) {
      std::string errorMessage = "MmulHelper::mmul (fast path): Invalid matrix dimensions. ";
      errorMessage += "M=" + std::to_string(M) + ", N=" + std::to_string(N) + ", K=" + std::to_string(K);
      errorMessage += ". Shapes: A=" + ShapeUtils::shapeAsString(A) + ", B=" + ShapeUtils::shapeAsString(B) + ", C=" + ShapeUtils::shapeAsString(C);
      THROW_EXCEPTION(errorMessage.c_str());
    }

    // Optimized fast path for contiguous row-major arrays
    auto& blasHelper = BlasHelper::getInstance();
    auto blasLock = blasHelper.lockBlas();

    if (aType == DataType::FLOAT32 && blasHelper.hasGEMM(aType)) {
      blasHelper.sgemm()(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, (float)alpha,
                         A->bufferAsT<float>(), K, B->bufferAsT<float>(), N, (float)beta,
                         C->bufferAsT<float>(), N);
      return C;
    } else if (aType == DataType::DOUBLE && blasHelper.hasGEMM(aType)) {
      blasHelper.dgemm()(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha,
                         A->bufferAsT<double>(), K, B->bufferAsT<double>(), N, beta,
                         C->bufferAsT<double>(), N);
      return C;
    } else if ((aType == DataType::HALF || aType == DataType::BFLOAT16) && blasHelper.hasGEMM(DataType::FLOAT32)) {
      // FP16/BF16 pre-cast path: cast to FP32, run sgemm, cast result back.
      // Avoids per-element half→float→half conversion in usualGemm inner loop.
      auto *aF32 = A->cast(DataType::FLOAT32);
      auto *bF32 = B->cast(DataType::FLOAT32);
      std::vector<LongType> cShape = {M, N};
      NDArray cF32('c', cShape, DataType::FLOAT32, A->getContext());
      blasHelper.sgemm()(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, (float)alpha,
                         aF32->bufferAsT<float>(), K, bF32->bufferAsT<float>(), N, (float)beta,
                         cF32.bufferAsT<float>(), N);
      C->assign(&cF32);
      delete aF32;
      delete bF32;
      return C;
    }
  }

  // General path for non-contiguous or non-standard layouts
  const bool hasGemm = BlasHelper::getInstance().hasGEMM(aType);
  const bool typeDouble = hasGemm && ABC && aType == DataType::DOUBLE;
  const bool typeFloat = hasGemm && ABC && aType == DataType::FLOAT32;

  // FP16/BF16 pre-cast for non-contiguous layouts — cast to FP32 and use sgemm
  if ((!typeFloat && !typeDouble) && (aType == DataType::HALF || aType == DataType::BFLOAT16)
      && ABC && BlasHelper::getInstance().hasGEMM(DataType::FLOAT32) && sd::env_isEnableBlas()) {
    auto *aF32 = A->cast(DataType::FLOAT32);
    auto *bF32 = B->cast(DataType::FLOAT32);
    std::vector<LongType> cShape = {M, N};
    NDArray cF32('c', cShape, DataType::FLOAT32, A->getContext());
    auto blasLock2 = BlasHelper::getInstance().lockBlas();
    BlasHelper::getInstance().sgemm()(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, (float)alpha,
                       aF32->bufferAsT<float>(), K, bF32->bufferAsT<float>(), N, (float)beta,
                       cF32.bufferAsT<float>(), N);
    C->assign(&cF32);
    delete aF32;
    delete bF32;
    return C;
  }

  // Mixed-type safe path: when A is FP32/FP64 but types don't all match (ABC=false),
  // usualGemm would reinterpret B and C buffers using float pointer arithmetic — this
  // produces wrong element offsets for non-FP32 B (e.g., FLOAT16: 2 bytes/element but
  // accessed as 4 bytes/element → buffer overrun → SEGV_ACCERR on guard page).
  // Fix: cast all inputs to A's type, use a tight-packed temp C, then assign back.
  if (!ABC && sd::env_isEnableBlas()) {
    if (aType == DataType::FLOAT32 && BlasHelper::getInstance().hasGEMM(DataType::FLOAT32)) {
      // Cast A and B to contiguous FP32; use a tight FP32 temp for C.
      NDArray* aF32 = (aType == DataType::FLOAT32 && A->strideAt(1) == 1 && A->strideAt(0) == K)
                          ? nullptr : A->cast(DataType::FLOAT32);
      NDArray* bF32 = (bType == DataType::FLOAT32 && B->strideAt(1) == 1 && B->strideAt(0) == N)
                          ? nullptr : B->cast(DataType::FLOAT32);
      NDArray* aEff = (aF32 != nullptr) ? aF32 : const_cast<NDArray*>(A);
      NDArray* bEff = (bF32 != nullptr) ? bF32 : const_cast<NDArray*>(B);
      std::vector<LongType> cShape = {M, N};
      NDArray cF32('c', cShape, DataType::FLOAT32, A->getContext());
      auto blasLock3 = BlasHelper::getInstance().lockBlas();
      BlasHelper::getInstance().sgemm()(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, (float)alpha,
                         aEff->bufferAsT<float>(), K, bEff->bufferAsT<float>(), N, (float)beta,
                         cF32.bufferAsT<float>(), N);
      C->assign(&cF32);
      delete aF32;
      delete bF32;
      return C;
    } else if (aType == DataType::DOUBLE && BlasHelper::getInstance().hasGEMM(DataType::DOUBLE)) {
      NDArray* aF64 = (aType == DataType::DOUBLE && A->strideAt(1) == 1 && A->strideAt(0) == K)
                          ? nullptr : A->cast(DataType::DOUBLE);
      NDArray* bF64 = (bType == DataType::DOUBLE && B->strideAt(1) == 1 && B->strideAt(0) == N)
                          ? nullptr : B->cast(DataType::DOUBLE);
      NDArray* aEff = (aF64 != nullptr) ? aF64 : const_cast<NDArray*>(A);
      NDArray* bEff = (bF64 != nullptr) ? bF64 : const_cast<NDArray*>(B);
      std::vector<LongType> cShape = {M, N};
      NDArray cF64('c', cShape, DataType::DOUBLE, A->getContext());
      auto blasLock4 = BlasHelper::getInstance().lockBlas();
      BlasHelper::getInstance().dgemm()(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha,
                         aEff->bufferAsT<double>(), K, bEff->bufferAsT<double>(), N, beta,
                         cF64.bufferAsT<double>(), N);
      C->assign(&cF64);
      delete aF64;
      delete bF64;
      return C;
    }
  }

  if ((!typeFloat && !typeDouble) || !sd::env_isEnableBlas()) {
    // When all three types match, usualGemm is safe as-is.
    // When types differ (ABC=false), we must cast to a common type first —
    // BUILD_SINGLE_SELECTOR_THRICE uses aType for all template params,
    // which would reinterpret B/C buffers with wrong element size.
    if (!ABC) {
      // Promote to A's type — cast B and C if they differ.
      NDArray* castB = (bType != aType) ? B->cast(aType) : nullptr;
      std::vector<LongType> cShape = {M, N};
      NDArray* castC = (cType != aType) ? new NDArray(NDArray('c', cShape, aType, C->getContext())) : nullptr;
      NDArray* effB = (castB != nullptr) ? castB : const_cast<NDArray*>(B);
      NDArray* effC = (castC != nullptr) ? castC : const_cast<NDArray*>(C);
      BUILD_SINGLE_SELECTOR_THRICE(aType, usualGemm, (A, effB, effC, 0, 1, 0, 1, 0, 1, alpha, beta), SD_NUMERIC_TYPES);
      if (castC != nullptr) {
        C->assign(effC);
      }
      delete castB;
      delete castC;
    } else {
      BUILD_SINGLE_SELECTOR_THRICE(aType, usualGemm, (A, B, C, 0, 1, 0, 1, 0, 1, alpha, beta), SD_NUMERIC_TYPES);
    }
  } else {
    NDArray *pA = const_cast<NDArray*>(A);
    NDArray *pB = const_cast<NDArray*>(B);
    NDArray *pC = const_cast<NDArray*>(C);
    NDArray *tempA = nullptr, *tempB = nullptr, *tempC = nullptr;

    bool aMcont = M == 1 || A->strideAt(0) == 1;
    bool aKcont = K == 1 || A->strideAt(1) == 1;
    bool bKcont = K == 1 || B->strideAt(0) == 1;
    bool bNcont = N == 1 || B->strideAt(1) == 1;
    bool cMcont = M == 1 || C->strideAt(0) == 1;
    bool cNcont = N == 1 || C->strideAt(1) == 1;

    if (!aMcont && !aKcont) {
      tempA = pA = A->dup('f', false);
      aMcont = true;
    }
    if (!bKcont && !bNcont) {
      tempB = pB = B->dup('f', false);
      bKcont = true;
    }
    if (!cMcont && !cNcont) {
      tempC = pC = C->dup('f', false);
      cMcont = true;
    }

    const CBLAS_ORDER blasOrder = cMcont ? CblasColMajor : CblasRowMajor;

    const bool transA = (!aMcont && cMcont) || (aMcont && !cMcont);
    const bool transB = (!bKcont && cMcont) || (bKcont && !cMcont);

    const CBLAS_TRANSPOSE transAblas = transA ? CblasTrans : CblasNoTrans;
    const CBLAS_TRANSPOSE transBblas = transB ? CblasTrans : CblasNoTrans;

    const int lda = (aMcont && aKcont) ? M : !aMcont ? pA->strideAt(0) : pA->strideAt(1);
    const int ldb = (bKcont && bNcont) ? K : !bKcont ? pB->strideAt(0) : pB->strideAt(1);
    const int ldc = (cMcont && cNcont) ? M : !cMcont ? pC->strideAt(0) : pC->strideAt(1);

    // Validate BLAS parameters before calling to prevent crashes
    // For row-major: lda >= K (or M if transposed), ldb >= N (or K if transposed), ldc >= N (or M if transposed)
    // For col-major: lda >= M (or K if transposed), ldb >= K (or N if transposed), ldc >= M (or N if transposed)
    if (lda <= 0 || ldb <= 0 || ldc <= 0) {
      std::string errorMessage = "MmulHelper::mmul: Invalid leading dimension(s). ";
      errorMessage += "lda=" + std::to_string(lda) + ", ldb=" + std::to_string(ldb) + ", ldc=" + std::to_string(ldc);
      errorMessage += ". Shapes: A=" + ShapeUtils::shapeAsString(pA) + ", B=" + ShapeUtils::shapeAsString(pB) + ", C=" + ShapeUtils::shapeAsString(pC);
      THROW_EXCEPTION(errorMessage.c_str());
    }

    if (M <= 0 || N <= 0 || K <= 0) {
      std::string errorMessage = "MmulHelper::mmul: Invalid matrix dimensions. ";
      errorMessage += "M=" + std::to_string(M) + ", N=" + std::to_string(N) + ", K=" + std::to_string(K);
      THROW_EXCEPTION(errorMessage.c_str());
    }

    // Acquire BLAS lock to prevent OpenBLAS TLS corruption and race conditions
    auto blasLock = BlasHelper::getInstance().lockBlas();

    if (typeFloat) {
      BlasHelper::getInstance().sgemm()(blasOrder, transAblas, transBblas, M, N, K, (float)alpha,
                                        pA->bufferAsT<float>(), lda, pB->bufferAsT<float>(), ldb, (float)beta,
                                        pC->bufferAsT<float>(), ldc);
    } else if (typeDouble) {
      BlasHelper::getInstance().dgemm()(blasOrder, transAblas, transBblas, M, N, K, (double)alpha,
                                        pA->bufferAsT<double>(), lda, pB->bufferAsT<double>(), ldb, (double)beta,
                                        pC->bufferAsT<double>(), ldc);
    }

    if (pC != C) {
      C->assign(pC);
    }

    delete tempA;
    delete tempB;
    delete tempC;
  }

  return C;
}

////////////////////////////////////////////////////////////////////////////
// MXN x N = M
NDArray* MmulHelper::mmulMxV( NDArray* A, NDArray* X, sd::NDArray* Y, const double alpha, const double beta,
                              const char outOrder) {
  if (X->dataType() != A->dataType()) {
    std::string errorMessage;
    errorMessage = "mmulMxV expects all data types to be the same";
    errorMessage += "A: " + DataTypeUtils::asString(A->dataType());
    errorMessage += "X: " + DataTypeUtils::asString(X->dataType());
    THROW_EXCEPTION(errorMessage.c_str());
  }
  if (Y != nullptr && X->dataType() != Y->dataType()) {
    std::string errorMessage;
    errorMessage = "mmulMxV expects all data types to be the same";
    errorMessage += "X: " + DataTypeUtils::asString(X->dataType());
    errorMessage += "Y: " + DataTypeUtils::asString(Y->dataType());
    THROW_EXCEPTION(errorMessage.c_str());
  }
  sd::LongType xLenDim, yLenDim(0);

  if (A->rankOf() != 2) THROW_EXCEPTION("MmulHelper::mmulMxV: rank of A array is not equal 2 !");
  if (!shape::isCommonVector(X->shapeInfo(), xLenDim))
    THROW_EXCEPTION("MmulHelper::mmulMxV: X array must be vector !");

  const auto M = A->sizeAt(0);
  const auto N = A->sizeAt(1);

  if (Y != nullptr && !shape::isCommonVector(Y->shapeInfo(), yLenDim))
    THROW_EXCEPTION("MmulHelper::mmulMxV: Y array must be vector !");
  if (X->lengthOf() != N) THROW_EXCEPTION("MmulHelper::mmulMxV: X vector has wrong length !");
  if (Y != nullptr && Y->lengthOf() != M) THROW_EXCEPTION("MmulHelper::mmulMxV: Y array has wrong length !");

  if (Y == nullptr) {
    std::vector<sd::LongType> shape = {M};
    Y = new NDArray(outOrder,shape, DataTypeUtils::pickPairwiseResultType(A->dataType(), X->dataType()),
                    A->getContext());
  }
  if (Y->isEmpty()) return Y;

  const int incx = X->stridesOf()[xLenDim];
  const int incy = Y->stridesOf()[yLenDim];

  const auto aType = A->dataType();
  const auto xType = X->dataType();
  const auto yType = Y->dataType();

  const bool AX(aType == xType), AY(aType == yType), AXY(AX && AY);
  const bool hasGemv = BlasHelper::getInstance().hasGEMV(aType);

  const bool typeDouble = hasGemv && AXY && aType == DataType::DOUBLE;
  const bool typeFloat = hasGemv && AXY && aType == DataType::FLOAT32;

  if ((!typeDouble && !typeFloat) || !sd::env_isEnableBlas()) {
    BUILD_SINGLE_SELECTOR_THRICE(aType, usualGemv, (A, X, Y, incx, incy, 0, alpha, beta), SD_NUMERIC_TYPES);
  } else {
    NDArray* pA(const_cast<NDArray*>(A));

    bool aMcont = M == 1 || A->strideAt(0) == 1;
    bool aNcont = N == 1 || A->strideAt(1) == 1;

    if (!aMcont && !aNcont) {
      pA = A->dup('f', false);  // dup() already returns NDArray*, no need for new
      aMcont = true;
    }
    const CBLAS_ORDER blasOrder = aMcont ? CblasColMajor : CblasRowMajor;

    const int lda = (aMcont && aNcont) ? M : !aMcont ? pA->strideAt(0) : pA->strideAt(1);

    // Acquire BLAS lock to prevent OpenBLAS TLS corruption and race conditions
    auto blasLock = BlasHelper::getInstance().lockBlas();

    // choose appropriate cuda gemm api depending on data types
    if (typeDouble) {
      BlasHelper::getInstance().dgemv()(blasOrder, CblasNoTrans, M, N, alpha, (double*)pA->buffer(), lda,
                                        (double*)X->buffer(), incx, beta, (double*)Y->buffer(), incy);
    } else if (typeFloat) {
      BlasHelper::getInstance().sgemv()(blasOrder, CblasNoTrans, M, N, (float)alpha, (float*)pA->buffer(), lda,
                                        (float*)X->buffer(), incx, (float)beta, (float*)Y->buffer(), incy);
    }

    // Clean up duplicated array
    if (pA != A) {
      delete pA;
    }

  }

  return Y;
}

////////////////////////////////////////////////////////////////////////////
// (X * Y) = Z[0]
NDArray* MmulHelper::dot(NDArray* X, NDArray* Y, sd::NDArray* Z, const double alpha, const double beta) {
  if (X->dataType() != Y->dataType()) {
    std::string errorMessage = "Dot expects all data types to be the same. ";
    errorMessage += "X datatype: " + DataTypeUtils::asString(X->dataType()) + ", ";
    errorMessage += "Y datatype: " + DataTypeUtils::asString(Y->dataType());
    THROW_EXCEPTION(errorMessage.c_str());
  }
  if (Z != nullptr && X->dataType() != Z->dataType()) {
    std::string errorMessage = "Dot expects all data types to be the same. ";
    errorMessage += "X datatype: " + DataTypeUtils::asString(X->dataType()) + ", ";
    errorMessage += "Z datatype: " + DataTypeUtils::asString(Z->dataType());
    THROW_EXCEPTION(errorMessage.c_str());
  }
  sd::LongType xLenDim(0), yLenDim(0);

  if (!shape::isCommonVector(X->shapeInfo(), xLenDim)) {
    std::string errorMessage = "MmulHelper::dot: X array must be a vector, but its shape is: ";
    for (int i = 0; i < X->rankOf(); ++i) {
      errorMessage += std::to_string(X->sizeAt(i));
      if (i < X->rankOf() - 1) errorMessage += "x";
    }
    THROW_EXCEPTION(errorMessage.c_str());
  }
  if (!shape::isCommonVector(Y->shapeInfo(), yLenDim)) {
    std::string errorMessage = "MmulHelper::dot: Y array must be a vector, but its shape is: ";
    for (int i = 0; i < Y->rankOf(); ++i) {
      errorMessage += std::to_string(Y->sizeAt(i));
      if (i < Y->rankOf() - 1) errorMessage += "x";
    }
    THROW_EXCEPTION(errorMessage.c_str());
  }
  if (Z != nullptr && Z->lengthOf() > 1) {
    std::string errorMessage = "MmulHelper::dot: Z array must be a scalar, but it has length " + std::to_string(Z->lengthOf());
    THROW_EXCEPTION(errorMessage.c_str());
  }

  const auto length = X->lengthOf();

  if (Y->lengthOf() != length) {
    std::string errorMessage = "MmulHelper::dot: lengths of input vectors are different! ";
    errorMessage += "X length: " + std::to_string(X->lengthOf()) + ", ";
    errorMessage += "Y length: " + std::to_string(Y->lengthOf());
    THROW_EXCEPTION(errorMessage.c_str());
  }

  if (Z == nullptr)
    Z = new NDArray(DataTypeUtils::pickPairwiseResultType(X->dataType(), Y->dataType()), X->getContext());

  const sd::LongType incx = X->stridesOf()[xLenDim];
  const sd::LongType incy = Y->stridesOf()[yLenDim];

  const auto xType = X->dataType();
  const auto yType = Y->dataType();
  const auto zType = Z->dataType();

  BUILD_SINGLE_SELECTOR_THRICE(
      xType, usualDot, (length, alpha, X->buffer(), incx, Y->buffer(), incy, beta, Z->buffer()), SD_NUMERIC_TYPES);

  return Z;
}

//////////////////////////////////////////////////////////////////////////////
// [bS,M,K] x [bS,K,N] = [bS,M,N]
// [bS,M,K] x    [K,N] = [bS,M,N]
//    [M,K] x [bS,K,N] = [bS,M,N]
// bS could stand for several axes
template <typename T1, typename T2, typename T3>
static void batchedGemm(NDArray* vA, NDArray* vB, NDArray* vC, const sd::LongType* aBatchDims,
                        const sd::LongType* bBatchDims, const sd::LongType* cBatchDims, sd::LongType aMaxis, sd::LongType aKaxis,
                        sd::LongType bKaxis, sd::LongType bNaxis, sd::LongType cMaxis, sd::LongType cNaxis, const double alpha, const double beta) {
  T1* A = vA->bufferAsT<T1>();
  T2* B = vB->bufferAsT<T2>();
  T3* C = vC->bufferAsT<T3>();

  const T3 alphaZ = static_cast<T3>(alpha);
  const T3 betaZ = static_cast<T3>(beta);

  const bool betaPresent = beta;

  const sd::LongType* aShapeInfo = vA->shapeInfo();
  const sd::LongType* bShapeInfo = vB->shapeInfo();
  const sd::LongType* cShapeInfo = vC->shapeInfo();

  const sd::LongType aRank = vA->rankOf();
  const sd::LongType bRank = vB->rankOf();
  const sd::LongType cRank = vC->rankOf();

  const sd::LongType M = vC->sizeAt(cMaxis);
  const sd::LongType N = vC->sizeAt(cNaxis);
  const sd::LongType K = vA->sizeAt(aKaxis);

  sd::LongType *aStride = shape::stride(aShapeInfo);
  sd::LongType *bStride = shape::stride(bShapeInfo);
  sd::LongType *cStride = shape::stride(cShapeInfo);

  // Check for contiguous 3D case: [batch, M, K] x [batch, K, N] = [batch, M, N]
  // All arrays must be rank 3 with row-major layout (last dim stride=1, second-to-last=N or K)
  const bool is3D = (aRank == 3) && (bRank == 3) && (cRank == 3);
  const bool aRowMajor = (aStride[aKaxis] == 1) && (aStride[aMaxis] == K);
  const bool bRowMajor = (bStride[bNaxis] == 1) && (bStride[bKaxis] == N);
  const bool cRowMajor = (cStride[cNaxis] == 1) && (cStride[cMaxis] == N);
  const bool allRowMajor = aRowMajor && bRowMajor && cRowMajor;

  // Check batch strides for contiguity
  const bool aBatchContiguous = is3D && (aStride[0] == M * K);
  const bool bBatchContiguous = is3D && (bStride[0] == K * N);
  const bool cBatchContiguous = is3D && (cStride[0] == M * N);
  const bool allBatchContiguous = aBatchContiguous && bBatchContiguous && cBatchContiguous;

  if (is3D && allRowMajor && allBatchContiguous) {
    // Fast path for contiguous 3D: parallelize over (batch * M) rows
    const sd::LongType batchSize = vA->sizeAt(0);
    const sd::LongType totalRows = batchSize * M;
    const sd::LongType strideA = M * K;
    const sd::LongType strideB = K * N;
    const sd::LongType strideC = M * N;

    auto func = PRAGMA_THREADS_FOR {
      for (auto rowIdx = start; rowIdx < stop; ++rowIdx) {
        const sd::LongType b = rowIdx / M;
        const sd::LongType m = rowIdx % M;
        T1* aRow = A + b * strideA + m * K;
        T2* bBase = B + b * strideB;
        T3* cRow = C + b * strideC + m * N;

        // Initialize output row
        if (betaPresent) {
          PRAGMA_OMP_SIMD
          for (sd::LongType n = 0; n < N; ++n) {
            cRow[n] *= betaZ;
          }
        } else {
          PRAGMA_OMP_SIMD
          for (sd::LongType n = 0; n < N; ++n) {
            cRow[n] = static_cast<T3>(0);
          }
        }

        // Cache-friendly loop order: k outer, n inner
        for (sd::LongType k = 0; k < K; ++k) {
          const T3 aVal = alphaZ * static_cast<T3>(aRow[k]);
          const T2* bRow = bBase + k * N;
          PRAGMA_OMP_SIMD
          for (sd::LongType n = 0; n < N; ++n) {
            cRow[n] += aVal * static_cast<T3>(bRow[n]);
          }
        }
      }
    };
    samediff::Threads::parallel_tad(func, 0, totalRows);
  } else {
    // General strided path - original implementation
    const sd::LongType cLen = vC->lengthOf();

    sd::LongType *cShape = shape::shapeOf(cShapeInfo);

    auto func = PRAGMA_THREADS_FOR {
      std::vector<sd::LongType> aCoords(aRank), bCoords(bRank), cCoords(cRank);

      for (sd::LongType i = start; i < stop; ++i) {
        // evaluate C coordinates
        INDEX2COORDS(i, cRank, cShape, cCoords.data());

        // evaluate A coordinates: copy batch dims directly from C coords.
        // Batch dims are always the leading dims: [0..rank-3] for each array.
        // The previous approach (batchInd via COORDS2INDEX + INDEX2COORDS roundtrip)
        // was incorrect for ranks > 3 because cCoords includes the M/N matrix dims,
        // making batchInd exceed A's element count and producing wrong batch coordinates.
        if (aRank > 2) {
          // Copy shared batch dimensions from C to A (batch dims are 0..aRank-3)
          for (sd::LongType d = 0; d < aRank - 2; ++d) aCoords[d] = cCoords[d];
        }
        aCoords[aMaxis] = cCoords[cMaxis];
        aCoords[aKaxis] = 0;

        // evaluate B coordinates: same direct-copy approach for batch dims
        if (bRank > 2) {
          for (sd::LongType d = 0; d < bRank - 2; ++d) bCoords[d] = cCoords[d];
        }
        bCoords[bKaxis] = 0;
        bCoords[bNaxis] = cCoords[cNaxis];

        sd::LongType aOffset, bOffset, cOffset;
        COORDS2INDEX(aRank, aStride, aCoords.data(), aOffset);
        COORDS2INDEX(bRank, bStride, bCoords.data(), bOffset);

        T3 val = A[aOffset] * B[bOffset];  // first iteration

        for (sd::LongType j = 1; j < K; ++j) {  // rest iterations
          aOffset += aStride[aKaxis];
          bOffset += bStride[bKaxis];
          val = val + A[aOffset] * B[bOffset];
        }

        COORDS2INDEX(cRank, cStride, cCoords.data(), cOffset);

        if (betaPresent)
          C[cOffset] = alphaZ * val + betaZ * C[cOffset];
        else
          C[cOffset] = alphaZ * val;
      }
    };

    samediff::Threads::parallel_tad(func, 0, cLen);
  }
}
//////////////////////////////////////////////////////////////////////////
NDArray* MmulHelper::mmulNxN( NDArray* A,  NDArray* B, NDArray* C, const double alpha, const double beta,
                              const char outOrder) {
  const sd::LongType aRank = A->rankOf();
  const sd::LongType bRank = B->rankOf();

  auto shapeToString = []( NDArray* arr) {
    std::string shape = "[";
    for (int i = 0; i < arr->rankOf(); ++i) {
      shape += std::to_string(arr->sizeAt(i));
      if (i < arr->rankOf() - 1) shape += ",";
    }
    shape += "]";
    return shape;
  };

  // input ranks validation
  if (aRank > bRank && bRank != 2) {
    std::string errorMessage = "MmulHelper::mmulNxN: rank of B array should be equal 2, but got " + std::to_string(bRank) +
                               "! A shape: " + shapeToString(A) + ", B shape: " + shapeToString(B);
    THROW_EXCEPTION(errorMessage.c_str());
  } else if (bRank > aRank && aRank != 2) {
    std::string errorMessage = "MmulHelper::mmulNxN: rank of A array should be equal 2, but got " + std::to_string(aRank) +
                               "! A shape: " + shapeToString(A) + ", B shape: " + shapeToString(B);
    THROW_EXCEPTION(errorMessage.c_str());
  } else if (aRank == bRank) {
    for (int i = 0; i < aRank - 2; ++i)
      if (A->sizeAt(i) != B->sizeAt(i)) {
        std::string errorMessage = "MmulHelper::mmulNxN: shapes of A and B arrays are not suitable for matrix multiplication! "
                                   "Mismatch at dimension " + std::to_string(i) + ": A[" + std::to_string(i) + "] = " +
                                   std::to_string(A->sizeAt(i)) + ", B[" + std::to_string(i) + "] = " + std::to_string(B->sizeAt(i)) +
                                   ". Full shapes: A " + shapeToString(A) + ", B " + shapeToString(B);
        THROW_EXCEPTION(errorMessage.c_str());
      }
  }

  if (A->sizeAt(-1) != B->sizeAt(-2)) {
    std::string errorMessage = "MmulHelper::mmulNxN: shapes of A and B arrays are not suitable for matrix multiplication! "
                               "A's last dimension (" + std::to_string(A->sizeAt(-1)) + ") must match B's second-to-last dimension (" +
                               std::to_string(B->sizeAt(-2)) + "). Full shapes: A " + shapeToString(A) + ", B " + shapeToString(B);
    THROW_EXCEPTION(errorMessage.c_str());
  }
  // validation of C array - build expected shape without allocation
  const sd::LongType maxRank = aRank > bRank ? aRank : bRank;
  std::vector<sd::LongType> cExpectedShape(maxRank);
  NDArray* sourceArr = aRank > bRank ? A : B;
  for (sd::LongType i = 0; i < maxRank; ++i) {
    cExpectedShape[i] = sourceArr->sizeAt(i);
  }
  cExpectedShape[maxRank - 2] = A->sizeAt(-2);
  cExpectedShape[maxRank - 1] = B->sizeAt(-1);

  if (C != nullptr) {
    if (!C->isSameShape(cExpectedShape)) {
      std::string errorMessage = "MmulHelper::mmulNxN: shape of C array is not suitable for AxB matrix multiplication! "
                                 "Expected shape: [";
      for (size_t i = 0; i < cExpectedShape.size(); ++i) {
        errorMessage += std::to_string(cExpectedShape[i]);
        if (i < cExpectedShape.size() - 1) errorMessage += ",";
      }
      errorMessage += "], but got: " + shapeToString(C) + ". A shape: " + shapeToString(A) + ", B shape: " + shapeToString(B);
      THROW_EXCEPTION(errorMessage.c_str());
    }
  } else {
    C = new NDArray(outOrder, cExpectedShape, B->dataType());
  }

  if (C->isEmpty()) return C;

  const sd::LongType cRank = C->rankOf();

  const sd::LongType aMaxis(aRank - 2), aKaxis(aRank - 1), bKaxis(bRank - 2), bNaxis(bRank - 1), cMaxis(cRank - 2),
      cNaxis(cRank - 1);

  std::vector<sd::LongType> *aBatchDims, *bBatchDims, *cBatchDims;
  if (aRank > 2) {
    sd::LongType aaxes[2];
    aaxes[0] = aMaxis;
    aaxes[1] = aKaxis;
    aBatchDims = ShapeUtils::evalDimsToExclude(aRank,2,aaxes);
  } else {
    aBatchDims = new std::vector<sd::LongType>();
  }
  if (bRank > 2) {
    sd::LongType baxes[2];
    baxes[0] = bKaxis;
    baxes[1] = bNaxis;
    bBatchDims = ShapeUtils::evalDimsToExclude(bRank, 2,baxes);
  } else {
    bBatchDims = new std::vector<sd::LongType>();
  }

  if (cRank > 2) {
    sd::LongType caxes[2];
    caxes[0] = cMaxis;
    caxes[1] = cNaxis;
    cBatchDims = ShapeUtils::evalDimsToExclude(cRank, 2,caxes);
  } else {
    cBatchDims = new std::vector<sd::LongType>();
  }

  BUILD_SINGLE_SELECTOR_THRICE(A->dataType(), batchedGemm,
                               (A, B, C, aBatchDims->data(), bBatchDims->data(), cBatchDims->data(), aMaxis, aKaxis,
                                   bKaxis, bNaxis, cMaxis
                                , cNaxis, alpha, beta),
                               SD_NUMERIC_TYPES);

  if(aBatchDims != nullptr)
    delete aBatchDims;
  if(bBatchDims != nullptr)
    delete bBatchDims;
  if(cBatchDims != nullptr)
    delete cBatchDims;

  return C;
}

//////////////////////////////////////////////////////////////////////////
// Strided batch GEMM - most efficient for contiguous batched data
// Uses cblas_sgemm_batch_strided/cblas_dgemm_batch_strided (MKL on CPU)
// Only handles simple row-major contiguous case for reliability
// Non-standard layouts fall through to tryBlasBatched/tryBlasPerBatch
//////////////////////////////////////////////////////////////////////////
bool MmulHelper::tryBlasStridedBatched(NDArray* A, NDArray* B, NDArray* C,
                                        double alpha, double beta) {
#if defined(HAVE_MKL)
  if (!Environment::getInstance().isEnableBlas()) {
    return false;
  }

  // Check if strided batch GEMM is available
  if (!BlasHelper::getInstance().hasStridedBatchGEMM()) {
    return false;
  }

  const int aRank = A->rankOf();
  const int bRank = B->rankOf();
  const int cRank = C->rankOf();

  // Only handle 3D tensors for simplicity (most common case)
  if (aRank != 3 || bRank != 3 || cRank != 3) {
    return false;
  }

  const auto xType = A->dataType();
  const auto yType = B->dataType();
  const auto zType = C->dataType();

  // Types must match
  if (xType != yType || yType != zType) {
    return false;
  }

  // Only float and double supported for strided batch
  if (xType != DataType::FLOAT32 && xType != DataType::DOUBLE) {
    return false;
  }

  // Validate buffers
  if (A->buffer() == nullptr || B->buffer() == nullptr || C->buffer() == nullptr) {
    return false;
  }

  // Get dimensions: A[bS,M,K] @ B[bS,K,N] = C[bS,M,N]
  const int batchSize = static_cast<int>(A->sizeAt(0));
  const int M = static_cast<int>(A->sizeAt(1));
  const int K = static_cast<int>(A->sizeAt(2));
  const int K2 = static_cast<int>(B->sizeAt(1));
  const int N = static_cast<int>(B->sizeAt(2));

  // Validate dimensions
  if (M <= 0 || K <= 0 || N <= 0 || batchSize <= 0 || K != K2) {
    return false;
  }

  // Require strict row-major contiguous layout for all matrices
  const LongType expectedStrideA = static_cast<LongType>(M) * K;
  const LongType expectedStrideB = static_cast<LongType>(K) * N;
  const LongType expectedStrideC = static_cast<LongType>(M) * N;

  const bool aValid = (A->strideAt(2) == 1) && (A->strideAt(1) == K) && (A->strideAt(0) == expectedStrideA);
  const bool bValid = (B->strideAt(2) == 1) && (B->strideAt(1) == N) && (B->strideAt(0) == expectedStrideB);
  const bool cValid = (C->strideAt(2) == 1) && (C->strideAt(1) == N) && (C->strideAt(0) == expectedStrideC);

  if (!aValid || !bValid || !cValid) {
    return false;
  }

  // Use MKL's strided batch GEMM
  auto blasLock = BlasHelper::getInstance().lockBlas();

  if (xType == DataType::FLOAT32) {
    sd::mkl::sgemmBatchStrided(
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        M, N, K,
        static_cast<float>(alpha),
        A->bufferAsT<float>(), K, static_cast<MKL_INT>(expectedStrideA),
        B->bufferAsT<float>(), N, static_cast<MKL_INT>(expectedStrideB),
        static_cast<float>(beta),
        C->bufferAsT<float>(), N, static_cast<MKL_INT>(expectedStrideC),
        batchSize);
  } else {
    sd::mkl::dgemmBatchStrided(
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        M, N, K,
        alpha,
        A->bufferAsT<double>(), K, static_cast<MKL_INT>(expectedStrideA),
        B->bufferAsT<double>(), N, static_cast<MKL_INT>(expectedStrideB),
        beta,
        C->bufferAsT<double>(), N, static_cast<MKL_INT>(expectedStrideC),
        batchSize);
  }

  return true;
#else
  // MKL not available - strided batch not supported here
  return false;
#endif
}

}  // namespace sd

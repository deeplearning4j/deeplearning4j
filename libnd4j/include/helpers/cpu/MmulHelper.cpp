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
#include <exceptions/datatype_exception.h>
#include <execution/Threads.h>
#include <helpers/BlasHelper.h>
#include <helpers/ShapeUtils.h>

namespace sd {

//////////////////////////////////////////////////////////////////////////////
// MXK x KxN = MxN              -> actual sequence of axes doesn't matter
template <typename T1, typename T2, typename T3>
static void usualGemm(NDArray* vA, NDArray* vB, NDArray* vC, const int aMaxis, const int aKaxis,
                      const int bKaxis, const int bNaxis, const int cMaxis, const int cNaxis, const double alpha,
                      const double beta) {
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

  const T3 alphaZ = alpha;
  const T3 betaZ = beta;

  const bool betaPresent = beta;

  const sd::LongType* aShapeInfo = vA->shapeInfo();
  const sd::LongType* bShapeInfo = vB->shapeInfo();
  const sd::LongType* cShapeInfo = vC->shapeInfo();

  const int aRank = vA->rankOf();
  const int bRank = vB->rankOf();
  const int cRank = vC->rankOf();
  const sd::LongType cLen = vC->lengthOf();
  const int K = vA->sizeAt(aKaxis);

  sd::LongType *cShape = shape::shapeOf(cShapeInfo);
  sd::LongType *aShape = shape::shapeOf(aShapeInfo);
  sd::LongType *bShape = shape::shapeOf(bShapeInfo);
  sd::LongType *aStride = shape::stride(aShapeInfo);
  sd::LongType *bStride = shape::stride(bShapeInfo);
  sd::LongType *cStride = shape::stride(cShapeInfo);

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

//////////////////////////////////////////////////////////////////////////////
// MXN x N = M  -> actual sequence of {M,N} axes doesn't matter
template <typename T1, typename T2, typename T3>
static void usualGemv( NDArray* vA, NDArray* vX, NDArray* vY, const int incx, const int incy,
                       const int aMaxis, const double alpha, const double beta) {
  T1* A = vA->bufferAsT<T1>();
  T2* X = vX->bufferAsT<T2>();
  T3* Y = vY->bufferAsT<T3>();

  const T3 alphaZ = alpha;
  const T3 betaZ = beta;

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

  T3 sum = 0;

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

  auto M = A->sizeAt(0);
  auto K = A->sizeAt(1);
  auto N = B->sizeAt(1);

  if (C != nullptr && C->rankOf() != 2) {
    std::string errorMessage = "MmulHelper::mmulMxM: rank of C array should be equal to 2, but got " +
                               std::to_string(C->rankOf()) + ". ";
    errorMessage += "C datatype: " + DataTypeUtils::asString(C->dataType());
    THROW_EXCEPTION(errorMessage.c_str());
  }
  if (B->sizeAt(0) != K) {
    std::string errorMessage = "MmulHelper::mmulMxM: B array should have the same number of rows as A has columns. ";
    errorMessage += "A columns: " + std::to_string(K) + ", ";
    errorMessage += "B rows: " + std::to_string(B->sizeAt(0));
    THROW_EXCEPTION(errorMessage.c_str());
  }
  if (C != nullptr && C->sizeAt(0) != M) {
    std::string errorMessage = "MmulHelper::mmulMxM: C array should have the same number of rows as A. ";
    errorMessage += "A rows: " + std::to_string(M) + ", ";
    errorMessage += "C rows: " + std::to_string(C->sizeAt(0));
    THROW_EXCEPTION(errorMessage.c_str());}

  if (C != nullptr && C->sizeAt(1) != N) {
    std::string errorMessage = "MmulHelper::mmulMxM: C array should have the same number of columns as B. ";
    errorMessage += "B columns: " + std::to_string(N) + ", ";
    errorMessage += "C columns: " + std::to_string(C->sizeAt(1));
    THROW_EXCEPTION(errorMessage.c_str());
  }

  if (C == nullptr) {
    std::vector<sd::LongType> shape = {M,N};
    C = new NDArray(outOrder, shape, DataTypeUtils::pickPairwiseResultType(A->dataType(), B->dataType()),
                    A->getContext());
  }
  if (C->isEmpty()) return C;

  const auto aType = A->dataType();
  const auto bType = B->dataType();
  const auto cType = C->dataType();

  const bool AB(aType == bType), AC(aType == cType), ABC(AB && AC);
  const bool hasGemm = BlasHelper::getInstance().hasGEMM(aType);

  const bool typeDouble = hasGemm && ABC && aType == DataType::DOUBLE;
  const bool typeFloat = hasGemm && ABC && aType == DataType::FLOAT32;

  if ((!typeFloat && !typeDouble) || !Environment::getInstance().isEnableBlas()) {
    BUILD_SINGLE_SELECTOR_THRICE(aType, usualGemm, (A, B, C, 0, 1, 0, 1, 0, 1, alpha, beta), SD_NUMERIC_TYPES);
  } else {
    std::vector<NDArray*> toDelete;

    NDArray *pA(const_cast<NDArray*>(A)), *pB(const_cast<NDArray*>(B)), *pC(const_cast<NDArray*>(C));

    bool aMcont = M == 1 || A->strideAt(0) == 1;
    bool aKcont = K == 1 || A->strideAt(1) == 1;
    bool bKcont = K == 1 || B->strideAt(0) == 1;
    bool bNcont = N == 1 || B->strideAt(1) == 1;
    bool cMcont = M == 1 || C->strideAt(0) == 1;
    bool cNcont = N == 1 || C->strideAt(1) == 1;

    if (!aMcont && !aKcont) {
      pA = new NDArray(A->dup('f', false));
      toDelete.push_back(pA);
      aMcont = true;
    }
    if (!bKcont && !bNcont) {
      pB = new NDArray(B->dup('f', false));
      toDelete.push_back(pB);
      bKcont = true;
    }
    if (!cMcont && !cNcont) {
      pC = new NDArray(C->dup('f', false));
      toDelete.push_back(pC);
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

    for (auto* arr : toDelete) {
      delete arr;
    }
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

  if ((!typeDouble && !typeFloat) || !Environment::getInstance().isEnableBlas()) {
    BUILD_SINGLE_SELECTOR_THRICE(aType, usualGemv, (A, X, Y, incx, incy, 0, alpha, beta), SD_NUMERIC_TYPES);
  } else {
    NDArray* pA(const_cast<NDArray*>(A));

    bool aMcont = M == 1 || A->strideAt(0) == 1;
    bool aNcont = N == 1 || A->strideAt(1) == 1;

    if (!aMcont && !aNcont) {
      pA = new NDArray(A->dup('f', false));
      aMcont = true;
    }
    const CBLAS_ORDER blasOrder = aMcont ? CblasColMajor : CblasRowMajor;

    const int lda = (aMcont && aNcont) ? M : !aMcont ? pA->strideAt(0) : pA->strideAt(1);

    // choose appropriate cuda gemm api depending on data types
    if (typeDouble) {
      BlasHelper::getInstance().dgemv()(blasOrder, CblasNoTrans, M, N, alpha, (double*)pA->buffer(), lda,
                                        (double*)X->buffer(), incx, beta, (double*)Y->buffer(), incy);
    } else if (typeFloat) {
      BlasHelper::getInstance().sgemv()(blasOrder, CblasNoTrans, M, N, (float)alpha, (float*)pA->buffer(), lda,
                                        (float*)X->buffer(), incx, (float)beta, (float*)Y->buffer(), incy);
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
static void batchedGemm(NDArray* vA, NDArray* vB, NDArray* vC, LongType* aBatchDims,
                        const LongType* bBatchDims, const LongType* cBatchDims, LongType aMaxis, LongType aKaxis,
                        LongType bKaxis, LongType bNaxis, LongType cMaxis, LongType cNaxis, const double alpha, const double beta) {
  T1* A = vA->bufferAsT<T1>();
  T2* B = vB->bufferAsT<T2>();
  T3* C = vC->bufferAsT<T3>();

  const T3 alphaZ = alpha;
  const T3 betaZ = beta;

  const bool betaPersent = beta;

  const sd::LongType* aShapeInfo = vA->shapeInfo();
  const sd::LongType* bShapeInfo = vB->shapeInfo();
  const sd::LongType* cShapeInfo = vC->shapeInfo();

  const sd::LongType aRank = vA->rankOf();
  const sd::LongType bRank = vB->rankOf();
  const sd::LongType cRank = vC->rankOf();

  const sd::LongType cLen = vC->lengthOf();

  const sd::LongType K = vA->sizeAt(aKaxis);

  sd::LongType *cShape = shape::shapeOf(cShapeInfo);
  sd::LongType *aShape = shape::shapeOf(aShapeInfo);
  sd::LongType *bShape = shape::shapeOf(bShapeInfo);
  sd::LongType *aStride = shape::stride(aShapeInfo);
  sd::LongType *bStride = shape::stride(bShapeInfo);
  sd::LongType *cStride = shape::stride(cShapeInfo);


  auto func = PRAGMA_THREADS_FOR {
    std::vector<sd::LongType> aCoords(aRank), bCoords(bRank), cCoords(cRank);

    for (sd::LongType i = start; i < stop; ++i) {
      // evaluate C coordinates
      INDEX2COORDS(i, cRank,cShape, cCoords.data());

      // calculate index of current batch
      sd::LongType batchInd;
      if (cRank > 2) COORDS2INDEX(cRank, cStride, cCoords.data(), batchInd);

      // evaluate A coordinates
      if (aRank > 2) INDEX2COORDS(batchInd, aRank, aShape, aCoords.data());
      aCoords[aMaxis] = cCoords[cMaxis];
      aCoords[aKaxis] = 0;

      // evaluate B coordinates
      if (bRank > 2) INDEX2COORDS(batchInd, bRank, bShape, bCoords.data());
      bCoords[bKaxis] = 0;
      bCoords[bNaxis] = cCoords[cNaxis];

      sd::LongType aOffset, bOffset, cOffset;
      COORDS2INDEX(aRank, aShape, aCoords.data(), aOffset);
      COORDS2INDEX(bRank, bShape, bCoords.data(), bOffset);

      T3 val = A[aOffset] * B[bOffset];  // first iteration

      for (int j = 1; j < K; ++j) {  // rest iterations
        aOffset += aStride[aKaxis];
        bOffset += bStride[bKaxis];
        val = val + A[aOffset] * B[bOffset];
      }

      COORDS2INDEX(cRank,cShape, cCoords.data(), cOffset);

      if (betaPersent)
        C[cOffset] = alphaZ * val + betaZ * C[cOffset];
      else
        C[cOffset] = alphaZ * val;
    }
  };

  samediff::Threads::parallel_tad(func, 0, cLen);
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
  // validation of C array
  std::vector<sd::LongType> cExpectedShape = aRank > bRank ? A->getShapeAsVector() : B->getShapeAsVector();
  cExpectedShape[cExpectedShape.size() - 2] = A->sizeAt(-2);
  cExpectedShape[cExpectedShape.size() - 1] = B->sizeAt(-1);

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
                                   bKaxis, bNaxis, cMaxis, cNaxis, alpha, beta),
                               SD_NUMERIC_TYPES);

  if(aBatchDims != nullptr)
    delete aBatchDims;
  if(bBatchDims != nullptr)
    delete bBatchDims;
  if(cBatchDims != nullptr)
    delete cBatchDims;

  return C;
}




}  // namespace sd

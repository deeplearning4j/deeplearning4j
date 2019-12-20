/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
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
#include <NDArrayFactory.h>
#include <helpers/BlasHelper.h>
#include <helpers/ShapeUtils.h>
#include <exceptions/datatype_exception.h>
#include <execution/Threads.h>


namespace nd4j {

//////////////////////////////////////////////////////////////////////////////
// MXK x KxN = MxN              -> actual sequence of axes doesn't matter
template <typename T1, typename T2, typename T3>
static  void usualGemm(const NDArray* vA, const NDArray* vB, NDArray* vC,
                                 const int aMaxis, const int aKaxis, const int bKaxis, const int bNaxis, const int cMaxis, const int cNaxis,
                                 const double alpha, const double beta) {

    const T1* A = vA->bufferAsT<T1>();
    const T2* B = vB->bufferAsT<T2>();
          T3* C = vC->bufferAsT<T3>();

    const T3 alphaZ = alpha;
    const T3 betaZ  = beta;

    const bool betaPersent = beta;

    const Nd4jLong* aShapeInfo = vA->getShapeInfo();
    const Nd4jLong* bShapeInfo = vB->getShapeInfo();
    const Nd4jLong* cShapeInfo = vC->getShapeInfo();

    const int aRank = vA->rankOf();
    const int bRank = vB->rankOf();
    const int cRank = vC->rankOf();

    const Nd4jLong cLen = vC->lengthOf();

    const int K = vA->sizeAt(aKaxis);

    auto func = PRAGMA_THREADS_FOR {

        std::vector<Nd4jLong> aCoords(2), bCoords(2), cCoords(2);

        for (auto i = start; i < stop; ++i) {

            // evaluate C coordinates
            shape::index2coords(i, cShapeInfo, cCoords.data());

            // evaluate A coordinates
            aCoords[aMaxis] = cCoords[cMaxis];
            aCoords[aKaxis] = 0;

            // evaluate B coordinates
            bCoords[bKaxis] = 0;
            bCoords[bNaxis] = cCoords[cNaxis];

            auto aOffset = shape::getOffset(aShapeInfo, aCoords.data());
            auto bOffset = shape::getOffset(bShapeInfo, bCoords.data());

            T3 val = A[aOffset] * B[bOffset];                       // first iteration

            for (uint j = 1; j < K; ++j) {                          // rest iterations
                aOffset += shape::stride(aShapeInfo)[aKaxis];
                bOffset += shape::stride(bShapeInfo)[bKaxis];
                val = val + A[aOffset] * B[bOffset];
            }

            auto cOffset = shape::getOffset(cShapeInfo, cCoords.data());

            if(betaPersent)
                C[cOffset] = alphaZ * val + betaZ * C[cOffset];
            else
                C[cOffset] = alphaZ * val;
        }
    };

    samediff::Threads::parallel_tad(func, 0, cLen);
}


//////////////////////////////////////////////////////////////////////////////
// MXN x N = M  -> actual sequence of {M,N} axes doesn't matter
template <typename T1, typename T2, typename T3>
static  void usualGemv(const NDArray* vA, const NDArray* vX, NDArray* vY, const int incx, const int incy, const int aMaxis, const double alpha, const double beta) {

    const T1* A = vA->bufferAsT<T1>();
    const T2* X = vX->bufferAsT<T2>();
          T3* Y = vY->bufferAsT<T3>();

    const T3 alphaZ = alpha;
    const T3 betaZ  = beta;

    const bool betaPersent = beta;

    const Nd4jLong* aShapeInfo = vA->getShapeInfo();
    const Nd4jLong* xShapeInfo = vX->getShapeInfo();
    const Nd4jLong* yShapeInfo = vY->getShapeInfo();

    const int N = vX->lengthOf();
    const int M = vY->lengthOf();

    const auto aMstride = vA->strideAt(aMaxis);
    const auto aNstride = vA->strideAt(aMaxis == 0 ? 1 : 0);

    auto func = PRAGMA_THREADS_FOR {

        for (auto i = start; i < stop; ++i) {

            // evaluate offsets
            auto aOffset = i * aMstride;
            auto xOffset = 0;

            T3 val = A[aOffset] * X[xOffset];                       // first iteration

            for (uint j = 1; j < N; ++j) {                          // rest iterations
                aOffset += aNstride;
                xOffset += incx;
                val = val + A[aOffset] * X[xOffset];
            }

            auto yOffset = i * incy;

            if(betaPersent)
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
static void usualDot(const Nd4jLong length, const double alpha, const void* vX, const Nd4jLong incx, const void* vY, const Nd4jLong incy, const double beta, void* vZ) {

    T1* X = reinterpret_cast<T1*>(const_cast<void*>(vX));
    T2* Y = reinterpret_cast<T2*>(const_cast<void*>(vY));
    T3* Z = reinterpret_cast<T3*>(vZ);
    T3 alphaZ(alpha), betaZ(beta);

    const bool betaPersent = beta;

    T3 sum = 0;
    PRAGMA_OMP_PARALLEL_FOR_ARGS(OMP_IF(length > Environment::getInstance()->elementwiseThreshold()) schedule(guided) reduction(OMP_SUMT:sum))
    for(int i = 0; i < length; ++i)
            sum += X[i * incx] * Y[i * incy];

    if(betaPersent)
        *Z = alphaZ * sum + betaZ * *Z;
    else
        *Z = alphaZ * sum;
}

//////////////////////////////////////////////////////////////////////////////
// MXK x KxN = MxN
NDArray* MmulHelper::mmulMxM(const NDArray* A, const NDArray* B, NDArray* C, const double alpha, const double beta, const char outOrder) {
    if (A->dataType() != B->dataType())
        throw datatype_exception::build("mmulMxM expects all data types to be the same", A->dataType(), B->dataType());

    if (C != nullptr && A->dataType() != C->dataType())
        throw datatype_exception::build("mmulMxM expects all data types to be the same", A->dataType(), C->dataType());

    if(A->rankOf() != 2)
        throw std::runtime_error("MmulHelper::mmulMxM: rank of A array is not equal 2 !");
    if(B->rankOf() != 2)
        throw std::runtime_error("MmulHelper::mmulMxM: rank of B array is not equal 2 !");

    const auto M = A->sizeAt(0);
    const auto K = A->sizeAt(1);
    const auto N = B->sizeAt(1);

    if(C != nullptr && C->rankOf() != 2)
        throw std::runtime_error("MmulHelper::mmulMxM: rank of C array is not equal 2 !");
    if(B->sizeAt(0) != K)
        throw std::runtime_error("MmulHelper::mmulMxM: B array has wrong number of rows !");
    if(C != nullptr && C->sizeAt(0) != M)
        throw std::runtime_error("MmulHelper::mmulMxM: C array has wrong number of rows !");
    if(C != nullptr && C->sizeAt(1) != N)
        throw std::runtime_error("MmulHelper::mmulMxM: C array has wrong number of columns !");

    if(C == nullptr)
        C = new NDArray(outOrder, {M,N}, DataTypeUtils::pickPairwiseResultType(A->dataType(), B->dataType()), A->getContext());

    const auto aType = A->dataType();
    const auto bType = B->dataType();
    const auto cType = C->dataType();

    const bool AB(aType == bType), AC(aType == cType), ABC(AB && AC);
    const bool hasGemm = BlasHelper::getInstance()->hasGEMM(aType);

    const bool typeDouble = hasGemm && ABC &&  aType == DataType::DOUBLE;
    const bool typeFloat  = hasGemm && ABC &&  aType == DataType::FLOAT32;

    if(!typeFloat && !typeDouble) {
        BUILD_SINGLE_SELECTOR_THRICE(aType, usualGemm, (A, B, C, 0, 1, 0, 1, 0, 1, alpha, beta), NUMERIC_TYPES);
        // BUILD_TRIPLE_SELECTOR(aType, bType, cType, usualGemm, (A, B, C, 0, 1, 0, 1, 0, 1, alpha, beta), LIBND4J_TYPES, FLOAT_TYPES, FLOAT_TYPES);
    }
    else {

        std::vector<NDArray*> toDelete;

        NDArray *pA(const_cast<NDArray*>(A)), *pB(const_cast<NDArray*>(B)), *pC(const_cast<NDArray*>(C));

        bool aMcont = M == 1 || A->strideAt(0) == 1;
        bool aKcont = K == 1 || A->strideAt(1) == 1;
        bool bKcont = K == 1 || B->strideAt(0) == 1;
        bool bNcont = N == 1 || B->strideAt(1) == 1;
        bool cMcont = M == 1 || C->strideAt(0) == 1;
        bool cNcont = N == 1 || C->strideAt(1) == 1;

        if(!aMcont && !aKcont) {
            pA = new NDArray(A->dup('f'));
            toDelete.push_back(pA);
            aMcont = true;
        }
        if(!bKcont && !bNcont) {
            pB = new NDArray(B->dup('f'));
            toDelete.push_back(pB);
            bKcont = true;
        }
        if(!cMcont && !cNcont) {
            pC = new NDArray(C->dup('f'));
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

        if(typeFloat) {
            BlasHelper::getInstance()->sgemm()(blasOrder, transAblas, transBblas, M, N, K, (float) alpha, pA->bufferAsT<float>(), lda, pB->bufferAsT<float>(), ldb, (float) beta, pC->bufferAsT<float>(), ldc);
        }
        else if(typeDouble) {
            BlasHelper::getInstance()->dgemm()(blasOrder, transAblas, transBblas, M, N, K, (double) alpha, pA->bufferAsT<double>(), lda, pB->bufferAsT<double>(), ldb, (double) beta, pC->bufferAsT<double>(), ldc);
        }

        if(pC != C) {
            C->assign(pC);
            delete pC;
        }
        if(pA != A)
            delete pA;
        if(pB != B)
            delete pB;
    }

    return C;
}

////////////////////////////////////////////////////////////////////////////
// MXN x N = M
NDArray* MmulHelper::mmulMxV(const NDArray* A, const NDArray* X, nd4j::NDArray* Y, const double alpha, const double beta, const char outOrder) {

    if (X->dataType() != A->dataType())
        throw datatype_exception::build("mmulMxV expects all data types to be the same", A->dataType(), X->dataType());

    if (Y != nullptr && X->dataType() != Y->dataType())
        throw datatype_exception::build("mmulMxV expects all data types to be the same", A->dataType(), Y->dataType());

    int xLenDim, yLenDim(0);

    if(A->rankOf() != 2)
        throw std::runtime_error("MmulHelper::mmulMxV: rank of A array is not equal 2 !");
    if(!shape::isCommonVector(X->getShapeInfo(), xLenDim))
        throw std::runtime_error("MmulHelper::mmulMxV: X array must be vector !");

    const auto M = A->sizeAt(0);
    const auto N = A->sizeAt(1);

    if(Y != nullptr && !shape::isCommonVector(Y->getShapeInfo(), yLenDim))
        throw std::runtime_error("MmulHelper::mmulMxV: Y array must be vector !");
    if(X->lengthOf() != N)
        throw std::runtime_error("MmulHelper::mmulMxV: X vector has wrong length !");
    if(Y != nullptr && Y->lengthOf() != M)
        throw std::runtime_error("MmulHelper::mmulMxV: Y array has wrong length !");

    if(Y == nullptr)
        Y = new NDArray(outOrder, {M}, DataTypeUtils::pickPairwiseResultType(A->dataType(), X->dataType()), A->getContext());

    const int incx = X->stridesOf()[xLenDim];
    const int incy = Y->stridesOf()[yLenDim];

    const auto aType = A->dataType();
    const auto xType = X->dataType();
    const auto yType = Y->dataType();

    const bool AX(aType == xType), AY(aType == yType), AXY(AX && AY);
    const bool hasGemv = BlasHelper::getInstance()->hasGEMV(aType);

    const bool typeDouble = hasGemv && AXY && aType == DataType::DOUBLE;
    const bool typeFloat  = hasGemv && AXY && aType == DataType::FLOAT32;

    if(!typeDouble && !typeFloat) {
        BUILD_SINGLE_SELECTOR_THRICE(aType, usualGemv, (A, X, Y, incx, incy, 0, alpha, beta), NUMERIC_TYPES);
        // BUILD_TRIPLE_SELECTOR(aType, xType, yType, usualGemv, (A, X, Y, incx, incy, 0, alpha, beta), LIBND4J_TYPES, FLOAT_TYPES, FLOAT_TYPES);
    }
    else {

        NDArray *pA(const_cast<NDArray*>(A));

        bool aMcont = M == 1 || A->strideAt(0) == 1;
        bool aNcont = N == 1 || A->strideAt(1) == 1;

        if(!aMcont && !aNcont) {
            pA = new NDArray(A->dup('f'));
            aMcont = true;
        }
        const CBLAS_ORDER blasOrder = aMcont ? CblasColMajor : CblasRowMajor;

        const int lda = (aMcont && aNcont) ? M : !aMcont ? pA->strideAt(0) : pA->strideAt(1);

        // choose appropriate cuda gemm api depending on data types
        if(typeDouble) {
            BlasHelper::getInstance()->dgemv()(blasOrder, CblasNoTrans, M, N, alpha, (double*)pA->getBuffer(), lda, (double*)X->getBuffer(), incx, beta, (double*)Y->getBuffer(), incy);
        }
        else if(typeFloat) {
            BlasHelper::getInstance()->sgemv()(blasOrder, CblasNoTrans, M, N, (float)alpha, (float*)pA->getBuffer(), lda, (float*)X->getBuffer(), incx, (float)beta, (float*)Y->getBuffer(), incy);
        }

        if(pA != A)
            delete pA;
    }

    return Y;
}

////////////////////////////////////////////////////////////////////////////
// (X * Y) = Z[0]
NDArray* MmulHelper::dot(const NDArray* X, const NDArray* Y, nd4j::NDArray* Z, const double alpha, const double beta) {
    if (X->dataType() != Y->dataType())
        throw datatype_exception::build("Dot expects all data types to be the same", X->dataType(), Y->dataType());

    if (Z != nullptr && X->dataType() != Z->dataType())
        throw datatype_exception::build("Dot expects all data types to be the same", X->dataType(), Z->dataType());

    int xLenDim(0), yLenDim(0);

    if(!shape::isCommonVector(X->getShapeInfo(), xLenDim))
        throw std::runtime_error("MmulHelper::dot cuda: X array must be vector !");
    if(!shape::isCommonVector(Y->getShapeInfo(), yLenDim))
        throw std::runtime_error("MmulHelper::dot cuda: Y array must be vector !");
    if(Z != nullptr && !Z->isScalar())
        throw std::runtime_error("MmulHelper::dot cuda: Z array must be scalar !");

    const auto length = X->lengthOf();

    if(Y->lengthOf() != length)
        throw std::runtime_error("MmulHelper::dot cuda: lengths of input vectors are different !");

    if(Z == nullptr)
        Z = new NDArray(DataTypeUtils::pickPairwiseResultType(X->dataType(), Y->dataType()), X->getContext());

    const Nd4jLong incx = X->stridesOf()[xLenDim];
    const Nd4jLong incy = Y->stridesOf()[yLenDim];

    const auto xType = X->dataType();
    const auto yType = Y->dataType();
    const auto zType = Z->dataType();

    BUILD_SINGLE_SELECTOR_THRICE(xType, usualDot, (length, alpha, X->getBuffer(), incx, Y->getBuffer(), incy, beta, Z->getBuffer()), NUMERIC_TYPES);
        //BUILD_TRIPLE_SELECTOR(xType, yType, zType, usualDot, (length, alpha, X->getBuffer(), incx, Y->getBuffer(), incy, beta, Z->getBuffer()), LIBND4J_TYPES, FLOAT_TYPES, FLOAT_TYPES);

    return Z;
}

//////////////////////////////////////////////////////////////////////////////
// [bS,M,K] x [bS,K,N] = [bS,M,N]
// [bS,M,K] x    [K,N] = [bS,M,N]
//    [M,K] x [bS,K,N] = [bS,M,N]
// bS could stand for several axes
template <typename T1, typename T2, typename T3>
static void batchedGemm(const NDArray* vA, const NDArray* vB,  NDArray* vC,
                        const int* aBatchDims, const int* bBatchDims, const int* cBatchDims,
                        const int aMaxis, const int aKaxis, const int bKaxis, const int bNaxis, const int cMaxis, const int cNaxis,
                        const double alpha, const double beta) {

    const T1* A = vA->bufferAsT<T1>();
    const T2* B = vB->bufferAsT<T2>();
          T3* C = vC->bufferAsT<T3>();

    const T3 alphaZ = alpha;
    const T3 betaZ  = beta;

    const bool betaPersent = beta;

    const Nd4jLong* aShapeInfo = vA->getShapeInfo();
    const Nd4jLong* bShapeInfo = vB->getShapeInfo();
    const Nd4jLong* cShapeInfo = vC->getShapeInfo();

    const int aRank = vA->rankOf();
    const int bRank = vB->rankOf();
    const int cRank = vC->rankOf();

    const Nd4jLong cLen = vC->lengthOf();

    const int K = vA->sizeAt(aKaxis);

    auto func = PRAGMA_THREADS_FOR {

        std::vector<Nd4jLong> aCoords(aRank), bCoords(bRank), cCoords(cRank);

        for (auto i = start; i < stop; ++i) {

            // evaluate C coordinates
            shape::index2coords(i, cShapeInfo, cCoords.data());

            // calculate index of current batch
            Nd4jLong batchInd;
            if(cRank > 2)
                batchInd = shape::coords2index(cShapeInfo, cCoords.data(), cRank - 2, cBatchDims);

            // evaluate A coordinates
            if(aRank > 2)
                shape::index2coords(batchInd, aShapeInfo, aCoords.data(), aRank - 2, aBatchDims);
            aCoords[aMaxis] = cCoords[cMaxis];
            aCoords[aKaxis] = 0;

            // evaluate B coordinates
            if(bRank > 2)
                shape::index2coords(batchInd, bShapeInfo, bCoords.data(), bRank - 2, bBatchDims);
            bCoords[bKaxis] = 0;
            bCoords[bNaxis] = cCoords[cNaxis];

            auto aOffset = shape::getOffset(aShapeInfo, aCoords.data());
            auto bOffset = shape::getOffset(bShapeInfo, bCoords.data());

            T3 val = A[aOffset] * B[bOffset];                       // first iteration

            for (uint j = 1; j < K; ++j) {                          // rest iterations
                aOffset += shape::stride(aShapeInfo)[aKaxis];
                bOffset += shape::stride(bShapeInfo)[bKaxis];
                val = val + A[aOffset] * B[bOffset];
            }

            auto cOffset = shape::getOffset(cShapeInfo, cCoords.data());

            if(betaPersent)
                C[cOffset] = alphaZ * val + betaZ * C[cOffset];
            else
                C[cOffset] = alphaZ * val;
        }
    };

    samediff::Threads::parallel_tad(func, 0, cLen);
}

//////////////////////////////////////////////////////////////////////////
// [bS,M,K] x [bS,K,N] = [bS,M,N]
// [bS,M,K] x    [K,N] = [bS,M,N]
//    [M,K] x [bS,K,N] = [bS,M,N]
// bS could stand for several axes
NDArray* MmulHelper::mmulNxN(const NDArray* A, const NDArray* B, NDArray* C, const double alpha, const double beta, const char outOrder) {

    const int aRank = A->rankOf();
    const int bRank = B->rankOf();

    // input ranks validation
    if(aRank > bRank && bRank != 2)
        throw std::runtime_error("MmulHelper::mmulNxN: rank of B array should be equal 2 !");
    else if(bRank > aRank && aRank != 2)
        throw std::runtime_error("MmulHelper::mmulNxN: rank of A array should be equal 2 !");
    else if (aRank == bRank ) {
        for(int i = 0; i < aRank - 2; ++i)
            if(A->sizeAt(i) != B->sizeAt(i))
                throw std::runtime_error("MmulHelper::mmulNxN: shapes of A and B arrays are not suitable for matrix multiplication !");
    }

    if(A->sizeAt(-1) != B->sizeAt(-2))
        throw std::runtime_error("MmulHelper::mmulNxN: shapes of A and B arrays are not suitable for matrix multiplication !");

    // validation of C array
    std::vector<Nd4jLong> cExpectedShape = aRank > bRank ? A->getShapeAsVector() : B->getShapeAsVector();
    cExpectedShape[cExpectedShape.size() - 2] = A->sizeAt(-2);
    cExpectedShape[cExpectedShape.size() - 1] = B->sizeAt(-1);

    if(C != nullptr ) {
        if(!C->isSameShape(cExpectedShape))
            throw std::runtime_error("MmulHelper::mmulNxN: shape of C array is not suitable for AxB matrix multiplication !");
    }
    else {
        C = new NDArray(outOrder, cExpectedShape, B->dataType());
    }

    const int cRank = C->rankOf();

    const int aMaxis(aRank-2), aKaxis(aRank-1), bKaxis(bRank-2), bNaxis(bRank-1), cMaxis(cRank-2), cNaxis(cRank-1);

    std::vector<int> aBatchDims, bBatchDims, cBatchDims;

    if(aRank > 2)
        aBatchDims = ShapeUtils::evalDimsToExclude(aRank, {aMaxis, aKaxis});
    if(bRank > 2)
        bBatchDims = ShapeUtils::evalDimsToExclude(bRank, {bKaxis, bNaxis});
    if(cRank > 2)
        cBatchDims = ShapeUtils::evalDimsToExclude(cRank, {cMaxis, cNaxis});

    // BUILD_TRIPLE_SELECTOR(A->dataType(), B->dataType(), C->dataType(), batchedGemm, (A, B, C, aBatchDims.data(), bBatchDims.data(), cBatchDims.data(), aMaxis, aKaxis, bKaxis, bNaxis, cMaxis, cNaxis, alpha, beta), LIBND4J_TYPES, FLOAT_TYPES, FLOAT_TYPES);
    BUILD_SINGLE_SELECTOR_THRICE(A->dataType(), batchedGemm, (A, B, C, aBatchDims.data(), bBatchDims.data(), cBatchDims.data(), aMaxis, aKaxis, bKaxis, bNaxis, cMaxis, cNaxis, alpha, beta), NUMERIC_TYPES);

    return C;
}

/*
//////////////////////////////////////////////////////////////////////////
NDArray* MmulHelper::mmulNxN(const NDArray* A, const NDArray* B, NDArray* C, const double alpha, const double beta, const char outOrder) {

    const int aRank = A->rankOf();
    const int bRank = B->rankOf();

    // input ranks validation
    if(aRank > bRank && bRank != 2)
        throw std::runtime_error("MmulHelper::mmulNxN: rank of B array should be equal 2 !");
    else if(bRank > aRank && aRank != 2)
        throw std::runtime_error("MmulHelper::mmulNxN: rank of A array should be equal 2 !");
    else if (aRank == bRank ) {
        for(int i = 0; i < aRank - 2; ++i)
            if(A->sizeAt(i) != B->sizeAt(i))
                throw std::runtime_error("MmulHelper::mmulNxN: shapes of A and B arrays are not suitable for matrix multiplication !");
    }

    if(A->sizeAt(-1) != B->sizeAt(-2))
        throw std::runtime_error("MmulHelper::mmulNxN: shapes of A and B arrays are not suitable for matrix multiplication !");

    // validation of C array
    std::vector<Nd4jLong> cExpectedShape = aRank > bRank ? A->getShapeAsVector() : B->getShapeAsVector();
    cExpectedShape[cExpectedShape.size() - 2] = A->sizeAt(-2);
    cExpectedShape[cExpectedShape.size() - 1] = B->sizeAt(-1);

    if(C != nullptr ) {
        if(!C->isSameShape(cExpectedShape))
            throw std::runtime_error("MmulHelper::mmulNxN: shape of C array is not suitable for AxB matrix multiplication !");
    }
    else {
        C = new NDArray(outOrder, cExpectedShape, B->dataType());
    }


    // multiplication
    const std::vector<int> dimsToExclude = ShapeUtils::evalDimsToExclude(C->rankOf(), {-2, -1});
    const Nd4jLong numOfSubArrs = ShapeUtils::getNumOfSubArrs(C->getShapeInfo(), dimsToExclude);
    std::vector<Nd4jLong> idxRanges(2 * C->rankOf());

// #pragma omp parallel for schedule(guided) firstprivate(idxRanges)
        for(Nd4jLong i = 0; i < numOfSubArrs; ++i) {

            ShapeUtils::evalIdxRangesForSubArr(i, C->getShapeInfo(), dimsToExclude, idxRanges.data());
            NDArray cSubArr = (*C)(idxRanges);

            if(aRank > bRank) {
                NDArray aSubArr = (*A)(idxRanges);
                mmulMxM(&aSubArr, B, &cSubArr, 1., 0., outOrder);
            }
            else if(bRank > aRank) {
                NDArray bSubArr = (*B)(idxRanges);
                mmulMxM(A, &bSubArr, &cSubArr, 1., 0, outOrder);
            }
            else {
                NDArray aSubArr = (*A)(idxRanges);
                NDArray bSubArr = (*B)(idxRanges);
                mmulMxM(&aSubArr, &bSubArr, &cSubArr, 1., 0., outOrder);
            }
        }

    return C;
}

//////////////////////////////////////////////////////////////////////////////
// MXK x KxN = MxN
template <typename T1, typename T2, typename T3>
static void usualGemm(const char cOrder, const bool transA, const bool transB, const int M, const int N, const int K, const double alpha, const void* vA, const int lda, const void* vB, const int ldb, const double beta, void* vC, const int ldc) {

    T1* A = reinterpret_cast<T1*>(const_cast<void*>(vA));
    T2* B = reinterpret_cast<T2*>(const_cast<void*>(vB));
    T3* C = reinterpret_cast<T3*>(vC);
    T3 alphaZ(alpha), betaZ(beta);

    const bool flagC = cOrder == 'f';
    const bool flagA = (flagC && transA) || (!flagC && !transA);
    const bool flagB = (flagC && transB) || (!flagC && !transB);

    // PRAGMA_OMP_PARALLEL_FOR_ARGS(OMP_IF(M*N > Environment::getInstance()->elementwiseThreshold()) schedule(guided))
    // for(uint row = 0; row < M; ++row) {

    //     T3* c = flagC ? (C + row) : (C + row * ldc);

    //     for(uint col = 0; col < N; ++col)
    //         c[flagC ? col * ldc : col] = 0;

    //     for(uint i = 0; i < K; ++i) {

    //         T3* b = flagB ? (B + i * ldb) : (B + i);
    //         T3* a = flagA ? (A + row * lda + i) : (A + row + i * lda);

    //         if(flagC) {
    //             for(uint col = 0; col < N; ++col) {
    //                 if(betaZ)
    //                     c[col * ldc] += a * b[flagB ? col : col * ldb] + betaZ * c[col * ldc];
    //                 else
    //                     c[col * ldc] += a * b[flagB ? col : col * ldb];
    //             }
    //         }
    //         else {
    //             for(uint col = 0; col < N; ++col) {
    //                 if(betaZ)
    //                     c[col] += a * b[flagB ? col : col * ldb] + betaZ * c[col];
    //                 else
    //                     c[col] += a * b[flagB ? col : col * ldb];
    //             }
    //         }
    //     }
    // }

    auto func = PRAGMA_THREADS_FOR_2D { ;
        for (auto row = start_x; row < stop_x; row += inc_x) {
            for (auto col = start_y; col < stop_y; col += inc_y) {
                T3 *c = flagC ? (C + row + col * ldc) : (C + row * ldc + col);
                T3 val = 0;

                for (uint i = 0; i < K; ++i) {
                    T3 a = flagA ? *(A + row * lda + i) : *(A + row + i * lda);
                    T3 b = flagB ? *(B + col + i * ldb) : *(B + col * ldb + i);
                    val += alphaZ * a * b;
                }

                if (betaZ)
                    *c = val + betaZ * *c;
                else
                    *c = val;
            }
        }
    };

    samediff::Threads::parallel_tad(func, 0, M, 1, 0, N, 1);
}

//////////////////////////////////////////////////////////////////////////////
// MXN x N = M
template <typename T1, typename T2, typename T3>
static void usualGemv(const char aOrder, const int M, const int N, const double alpha, const void* vA, const int lda, const void* vX, const int incx, const double beta, void* vY, const int incy) {

    T1* A = reinterpret_cast<T1*>(const_cast<void*>(vA));
    T2* X = reinterpret_cast<T2*>(const_cast<void*>(vX));
    T3* Y = reinterpret_cast<T3*>(vY);
    T3 alphaZ(alpha), betaZ(beta);

    const bool flagA = aOrder == 'f';

    auto func = PRAGMA_THREADS_FOR {
        for (auto row = start; row < stop; row += increment) {

            T3 *y = Y + row * incy;
            T3 val = 0;

            for (int i = 0; i < N; ++i) {
                T3 a = flagA ? *(A + row + i * lda) : *(A + row * lda + i);
                T3 x = *(X + i * incx);
                val += alphaZ * a * x;
            }

            if (betaZ)
                *y = val + betaZ * *y;
            else
                *y = val;
        }
    };

        samediff::Threads::parallel_tad(func, 0, M);
}
*/

//BUILD_TRIPLE_TEMPLATE(template void usualGemm, (const char cOrder, const bool transA, const bool transB, const int M, const int N, const int K, const double alpha, const void* A, const int lda, const void* B, const int ldb, const double beta, void* C, const int ldc), LIBND4J_TYPES, FLOAT_TYPES, FLOAT_TYPES);
//BUILD_TRIPLE_TEMPLATE(template void usualGemv, (const char aOrder, const int M, const int N, const double alpha, const void* A, const int lda, const void* B, const int incx, const double beta, void* C, const int incy), LIBND4J_TYPES, FLOAT_TYPES, FLOAT_TYPES);
//BUILD_TRIPLE_TEMPLATE(template void usualDot,  (const Nd4jLong length, const double alpha, const void* vX, const Nd4jLong incx, const void* vY, const Nd4jLong incy, const double beta, void* vZ), LIBND4J_TYPES, FLOAT_TYPES, FLOAT_TYPES);

}

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


namespace nd4j {

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

    // PRAGMA_OMP_PARALLEL_FOR_ARGS(if(M*N > Environment::getInstance()->elementwiseThreshold()) schedule(guided))
    // for(uint row = 0; row < M; ++row) {

    //     T3* c = flagC ? (C + row) : (C + row * ldc);

    //     for(uint col = 0; col < N; ++col)
    //         c[flagC ? col * ldc : col] = 0;

    //     for(uint i = 0; i < K; ++i) {
            
    //         T3* b = flagB ? (B + i * ldb) : (B + i);
    //         T3* a = flagA ? (A + row * lda + i) : (A + row + i * lda);

    //         if(flagC) {
    //             PRAGMA_OMP_SIMD
    //             for(uint col = 0; col < N; ++col) {
    //                 if(betaZ)
    //                     c[col * ldc] += a * b[flagB ? col : col * ldb] + betaZ * c[col * ldc];
    //                 else
    //                     c[col * ldc] += a * b[flagB ? col : col * ldb];
    //             }
    //         }
    //         else {
    //             PRAGMA_OMP_SIMD
    //             for(uint col = 0; col < N; ++col) {
    //                 if(betaZ)
    //                     c[col] += a * b[flagB ? col : col * ldb] + betaZ * c[col];
    //                 else
    //                     c[col] += a * b[flagB ? col : col * ldb];
    //             }
    //         }
    //     }
    // }   

    PRAGMA_OMP_PARALLEL_FOR_ARGS(if(M*N > Environment::getInstance()->elementwiseThreshold()) schedule(guided) collapse(2))
    for(uint row = 0; row < M; ++row) {
       for(uint col = 0; col < N; ++col) {
            
            T3* c = flagC ? (C + row + col * ldc) : (C + row * ldc + col);
            T3 val = 0;  

            PRAGMA_OMP_SIMD
            for(uint i = 0; i < K; ++i) {
                T3 a = flagA ? *(A + row * lda + i) : *(A + row + i * lda);
                T3 b = flagB ? *(B + col + i * ldb) : *(B + col * ldb + i);             
                val += alphaZ * a * b;
            }
            
            if(betaZ)
                *c = val + betaZ * *c;
            else
                *c = val;
       }
    }
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

    PRAGMA_OMP_PARALLEL_FOR_ARGS(if(M > Environment::getInstance()->elementwiseThreshold()) schedule(guided))
    for(int row = 0; row < M; ++row) {
                        
        T3* y = Y + row * incy;
        T3 val = 0;

        PRAGMA_OMP_SIMD
        for(int i = 0; i < N; ++i) {
            T3 a = flagA ? *(A + row + i * lda) : *(A + row * lda + i);
            T3 x = *(X + i * incx);
            val += alphaZ * a * x;
        }
        
        if(betaZ)
            *y = val + betaZ * *y;
        else
            *y = val;
    }
}

//////////////////////////////////////////////////////////////////////////////
// (X*Y) = Z[0]
template <typename T1, typename T2, typename T3>
static void usualDot(const Nd4jLong length, const double alpha, const void* vX, const Nd4jLong incx, const void* vY, const Nd4jLong incy, const double beta, void* vZ) {

    T1* X = reinterpret_cast<T1*>(const_cast<void*>(vX));
    T2* Y = reinterpret_cast<T2*>(const_cast<void*>(vY));
    T3* Z = reinterpret_cast<T3*>(vZ);
    T3 alphaZ(alpha), betaZ(beta);

    T3 sum = 0;
    PRAGMA_OMP_PARALLEL_FOR_ARGS(if(length > Environment::getInstance()->elementwiseThreshold()) schedule(guided) reduction(OMP_SUMT:sum))
    for(int i = 0; i < length; ++i)
            sum = sum + X[i * incx] * Y[i * incy];        
    
    *Z = alphaZ * sum + betaZ * *Z;
}

//////////////////////////////////////////////////////////////////////////////
// MXK x KxN = MxN
NDArray* MmulHelper::mmulMxM(const NDArray* A, const NDArray* B, NDArray* C, const double alpha, const double beta, const char outOrder) {    

    if(A->rankOf() != 2)
        throw std::runtime_error("MmulHelper::mmulMxM: rank of A array is not equal 2 !");
    if(B->rankOf() != 2)
        throw std::runtime_error("MmulHelper::mmulMxM: rank of B array is not equal 2 !");    

    const auto M     = A->sizeAt(0);
    const auto K     = A->sizeAt(1);
    const auto N     = B->sizeAt(1);
    const auto bRows = B->sizeAt(0);
    
    if(C != nullptr && C->rankOf() != 2)
        throw std::runtime_error("MmulHelper::mmulMxM: rank of C array is not equal 2 !");
    if(bRows != K)
        throw std::runtime_error("MmulHelper::mmulMxM: B array has wrong number of rows !");
    if(C != nullptr && C->sizeAt(0) != M)
        throw std::runtime_error("MmulHelper::mmulMxM: C array has wrong number of rows !");
    if(C != nullptr && C->sizeAt(1) != N)
        throw std::runtime_error("MmulHelper::mmulMxM: C array has wrong number of columns !");

    if(C == nullptr)
        C = new NDArray(outOrder, {M,N}, DataTypeUtils::pickPairwiseResultType(A->dataType(), B->dataType()), A->getContext());       

    NDArray *pA(const_cast<NDArray*>(A)), *pB(const_cast<NDArray*>(B)), *pC(const_cast<NDArray*>(C));    

    const auto cOrder = C->ordering();

    if(A->ews() != 1)
        pA = pA->dup(cOrder);
    if(B->ews() != 1)
        pB = pB->dup(cOrder);
    if(C->ews() != 1)
        pC = pC->dup(cOrder);

    const auto aOrder = pA->ordering();
    const auto bOrder = pB->ordering();    

    const bool transA = aOrder != cOrder;
    const bool transB = bOrder != cOrder;
    
    const CBLAS_ORDER blasOrder  = cOrder == 'f' ? CblasColMajor : CblasRowMajor;    
    const CBLAS_TRANSPOSE transAblas = transA ? CblasTrans : CblasNoTrans;
    const CBLAS_TRANSPOSE transBblas = transB ? CblasTrans : CblasNoTrans;

    const int lda = aOrder == 'f' ? M : K;
    const int ldb = bOrder == 'f' ? K : N;
    const int ldc = cOrder == 'f' ? M : N;    

    const auto aType = pA->dataType();
    const auto bType = pB->dataType();
    const auto cType = pC->dataType();

    const bool AB(aType == bType), AC(aType == cType), ABC(AB && AC);
    const bool hasGemm = BlasHelper::getInstance()->hasGEMM(aType);
    
    // we'll use platform-specific gemm here eventually. maybe tomorrow.
    // TODO: put proper _gemm here
    if (ABC && hasGemm && aType == DataType::FLOAT32) {
        BlasHelper::getInstance()->sgemm()(blasOrder, transAblas, transBblas, M, N, K, (float) alpha, reinterpret_cast<float *>(pA->getBuffer()), lda, reinterpret_cast<float *>(pB->getBuffer()), ldb, (float) beta, reinterpret_cast<float *>(pC->getBuffer()), ldc);
    }
    else if (ABC && hasGemm && aType == DataType::DOUBLE) {
        BlasHelper::getInstance()->dgemm()(blasOrder, transAblas, transBblas, M, N, K, (double) alpha, reinterpret_cast<double *>(pA->getBuffer()), lda, reinterpret_cast<double *>(pB->getBuffer()), ldb, (double) beta, reinterpret_cast<double *>(pC->getBuffer()), ldc);
    }
    else {
        BUILD_TRIPLE_SELECTOR(aType, bType, cType, usualGemm, (cOrder, transA, transB, M, N, K, alpha, pA->getBuffer(), lda, pB->getBuffer(), ldb, beta, pC->getBuffer(), ldc), LIBND4J_TYPES, FLOAT_TYPES, FLOAT_TYPES);
    }    

    if(pC != C) {
        C->assign(pC);
        delete pC;
    }
    if(pA != A)
        delete pA;
    if(pB != B)
        delete pB;

    return C;
}

////////////////////////////////////////////////////////////////////////////
// MXN x N = M
NDArray* MmulHelper::mmulMxV(const NDArray* A, const NDArray* X, nd4j::NDArray* Y, const double alpha, const double beta, const char outOrder) {

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
    
    NDArray *pA(const_cast<NDArray*>(A));

    if(A->ews() != 1)
        pA = pA->dup();
    
    CBLAS_ORDER blasOrder;
    int lda;
    if (pA->ordering() == 'f')  {blasOrder = CblasColMajor; lda = M; }
    else                        {blasOrder = CblasRowMajor; lda = N; }
         
    const int incx = X->stridesOf()[xLenDim];
    const int incy = Y->stridesOf()[yLenDim];

    const auto aType = pA->dataType();
    const auto xType = X->dataType();
    const auto yType = Y->dataType();

    const bool AX(aType == xType), AY(aType == yType), AXY(AX && AY);
    const bool hasGemv = BlasHelper::getInstance()->hasGEMV(aType);
    
    // choose appropriate cuda gemm api depending on data types    
    if(AXY && hasGemv && aType == DataType::DOUBLE) {
        BlasHelper::getInstance()->dgemv()(blasOrder, CblasNoTrans, M, N, alpha,       (double*)pA->getBuffer(), lda, (double*)X->getBuffer(), incx, beta,        (double*)Y->getBuffer(), incy);
    }
    else if(AXY && hasGemv && aType == DataType::FLOAT32) {                
        BlasHelper::getInstance()->sgemv()(blasOrder, CblasNoTrans, M, N, (float)alpha, (float*)pA->getBuffer(), lda, (float*)X->getBuffer(),  incx, (float)beta, (float*)Y->getBuffer(),  incy);
    }
    else {
        BUILD_TRIPLE_SELECTOR(aType, xType, yType, usualGemv, (pA->ordering(), M, N, alpha, pA->getBuffer(), lda, X->getBuffer(), incx, beta, Y->getBuffer(), incy), LIBND4J_TYPES, FLOAT_TYPES, FLOAT_TYPES);        
    }

    if(pA != A)
        delete pA;
    
    return Y;
}

////////////////////////////////////////////////////////////////////////////
// (X * Y) = Z[0]
NDArray* MmulHelper::dot(const NDArray* X, const NDArray* Y, nd4j::NDArray* Z, const double alpha, const double beta) {

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
    
    BUILD_TRIPLE_SELECTOR(xType, yType, zType, usualDot, (length, alpha, X->getBuffer(), incx, Y->getBuffer(), incy, beta, Z->getBuffer()), LIBND4J_TYPES, FLOAT_TYPES, FLOAT_TYPES);        

    return Z;
}

BUILD_TRIPLE_TEMPLATE(template void usualGemm, (const char cOrder, const bool transA, const bool transB, const int M, const int N, const int K, const double alpha, const void* A, const int lda, const void* B, const int ldb, const double beta, void* C, const int ldc), LIBND4J_TYPES, FLOAT_TYPES, FLOAT_TYPES);
BUILD_TRIPLE_TEMPLATE(template void usualGemv, (const char aOrder, const int M, const int N, const double alpha, const void* A, const int lda, const void* B, const int incx, const double beta, void* C, const int incy), LIBND4J_TYPES, FLOAT_TYPES, FLOAT_TYPES);
BUILD_TRIPLE_TEMPLATE(template void usualDot,  (const Nd4jLong length, const double alpha, const void* vX, const Nd4jLong incx, const void* vY, const Nd4jLong incy, const double beta, void* vZ), LIBND4J_TYPES, FLOAT_TYPES, FLOAT_TYPES);

}

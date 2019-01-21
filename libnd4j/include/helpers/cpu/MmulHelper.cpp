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
template <typename X, typename Y, typename Z>
static void usualGemm(const char cOrder, const bool transA, const bool transB, const int M, const int N, const int K, const double alpha, const void* vA, const int lda, const void* vB, const int ldb, const double beta, void* vC, const int ldc) {

    X* A = reinterpret_cast<X*>(const_cast<void*>(vA));
    Y* B = reinterpret_cast<Y*>(const_cast<void*>(vB));
    Z* C = reinterpret_cast<Z*>(vC);
    Z alphaZ(alpha), betaZ(beta);
    
    Nd4jLong strideArow, strideAcol, strideBrow, strideBcol, strideCrow, strideCcol;

    if(cOrder == 'f') {        
        strideCrow = 1; 
        strideCcol = ldc;

        if(transA) { strideArow = lda; strideAcol = 1; } else { strideArow = 1; strideAcol = lda; }
        if(transB) { strideBrow = ldb; strideBcol = 1; } else { strideBrow = 1; strideBcol = ldb; }
    }
    else {
        strideCrow = ldc; 
        strideCcol = 1;

        if(transA) { strideArow = 1; strideAcol = lda; } else { strideArow = lda; strideAcol = 1; }
        if(transB) { strideBrow = 1; strideBcol = ldb; } else { strideBrow = ldb; strideBcol = 1; }
    }

    #pragma omp parallel for if(M*N > Environment::getInstance()->elementwiseThreshold()) schedule(guided) collapse(2)        
    for(int row = 0; row < M; ++row) {
       for(int col = 0; col < N; ++col) {            
            X* a = A + row * strideArow;
            Y* b = B + col * strideBcol;            
            Z* c = C + row * strideCrow + col * strideCcol;
            Z val = 0;            
            for(int i = 0; i < K; ++i)
                val = val + a[i*strideAcol] * b[i*strideBrow];            
            *c = alphaZ * val + betaZ * *c;
       }
    }
}


//////////////////////////////////////////////////////////////////////////////
// MXK x KxN = MxN
NDArray* MmulHelper::mmulMxM(const NDArray* A, const NDArray* B, NDArray* C, const double alpha, const double beta, const char outOrder) {

    if(A->rankOf() != 2)
        throw std::runtime_error("MmulHelper::mmulMxM: rank of A array is not equal 2 !");
    if(B->rankOf() != 2)
        throw std::runtime_error("MmulHelper::mmulMxM: rank of B array is not equal 2 !");
    if(C != nullptr && C->rankOf() != 2)
        throw std::runtime_error("MmulHelper::mmulMxM: rank of C array is not equal 2 !");

    const auto M = A->sizeAt(0);
    const auto K = A->sizeAt(1);
    const auto N = B->sizeAt(1);

    if(B->sizeAt(0) != K)
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
// static
template <typename X, typename Y, typename Z>
nd4j::NDArray* MmulHelper::mmulMxV(const NDArray* A, const NDArray* B, NDArray* C , const double alpha, const double beta, const char outOrder) {
    
    nd4j::NDArray* result = C;
        // gemv
        if (A->columns() != B->lengthOf())
            throw std::runtime_error("A columns != B length");
        if (result == nullptr)
            result = new NDArray('f', {A->rows(),1}, DataTypeUtils::fromT<Z>(), A->getContext());

        auto xType = A->dataType();
        auto yType = B->dataType();
        auto zType = result->dataType();

        // TODO: strides!!!
        if (xType == yType && xType == zType && BlasHelper::getInstance()->hasGEMV<X>()) {
            nd4j_debug("Using provided GEMV pointer\n","");
            auto layout = A->ordering() == 'f' ? CblasColMajor : CblasRowMajor;
            if (std::is_same<X, float>::value)
                BlasHelper::getInstance()->sgemv()(layout, CblasNoTrans, A->rows(), A->columns(), (float) alpha, reinterpret_cast<float *>(A->getBuffer()), layout == CblasColMajor ? A->rows() : A->columns(), reinterpret_cast<float *>(B->getBuffer()), 1, (float) beta, reinterpret_cast<float *>(result->getBuffer()), 1);
            else if (std::is_same<X, double>::value)
                BlasHelper::getInstance()->dgemv()(layout, CblasNoTrans, A->rows(), A->columns(), (double) alpha, reinterpret_cast<double *>(A->getBuffer()), layout == CblasColMajor ? A->rows() : A->columns(), reinterpret_cast<double *>(B->getBuffer()), 1, (double) beta, reinterpret_cast<double *>(result->getBuffer()), 1);
            else
                nd4j::blas::GEMV<X, Y, Z>::op(A->ordering() == 'f' ? CblasTrans : 0, A->rows(), A->columns(), alpha, A->getBuffer(), B->lengthOf(), B->getBuffer(), 1, beta, result->getBuffer(), 1);
        } else {
            nd4j_debug("Using fallback GEMV impl\n","");
            nd4j::blas::GEMV<X, Y, Z>::op(A->ordering() == 'f' ? CblasTrans : 0, A->rows(), A->columns(), alpha, A->getBuffer(), B->lengthOf(), B->getBuffer(), 1, beta, result->getBuffer(), 1);
        }
    return result;
}


BUILD_TRIPLE_TEMPLATE(template void usualGemm, (const char cOrder, const bool transA, const bool transB, const int M, const int N, const int K, const double alpha, const void* A, const int lda, const void* B, const int ldb, const double beta, void* C, const int ldc), LIBND4J_TYPES, FLOAT_TYPES, FLOAT_TYPES);
BUILD_TRIPLE_TEMPLATE(template NDArray* MmulHelper::mmulMxV, (const NDArray* A, const NDArray* B, NDArray* C, const double alpha, const double beta, const char outOrder), LIBND4J_TYPES, FLOAT_TYPES, FLOAT_TYPES);
}

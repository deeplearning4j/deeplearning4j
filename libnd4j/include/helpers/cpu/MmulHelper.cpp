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
// static
template<typename X, typename Y, typename Z>
NDArray* MmulHelper::mmulMxM(const NDArray* A, const NDArray* B, NDArray* C , double alpha, double beta) {
    nd4j::NDArray* result = C;
    bool needAllocA = false;
    bool needAllocB = false;
    if (A->isView()) {
        needAllocA = true;
    }
    if (B->isView()) {
        needAllocB = true;
    }
    if (result == nullptr) {
        nd4j_verbose("mmulMxM: Creating new array: [%i x %i]\n", A->rows(), B->columns());
        result = NDArrayFactory::create_<Z>('f', {A->rows(), B->columns()}, A->getContext());
    }
        
    auto aShape = A->shapeOf();
    auto bShape = B->shapeOf();
    auto cShape = result->shapeOf();
    char rOrder;
    int M, N, K, lda, ldb, ldc;
    CBLAS_TRANSPOSE transA = CblasNoTrans, 
                    transB = CblasNoTrans;
    M = cShape[0]; // c.rows
    N = cShape[1]; // c.columns
    K = aShape[1]; // a.columns
    rOrder = 'f'; //aOrder;
    nd4j::NDArray* pA = nullptr;
    nd4j::NDArray* pB = nullptr;
    nd4j::NDArray* pC = nullptr;;
    nd4j::NDArray* tA;
    nd4j::NDArray* tB;
    nd4j::NDArray* tC = result;
    
    if (needAllocA) {
        tA = new nd4j::NDArray(A->getBuffer(), A->getShapeInfo(), A->getContext());
        nd4j_verbose("Matrix A was recreated from view.\n", "");
    }
    else 
        tA = const_cast<NDArray*>(A); 
    if (needAllocB) {
        tB = new nd4j::NDArray(B->getBuffer(), B->getShapeInfo(), B->getContext());
        nd4j_verbose("Matrix B was recreated from view.\n", "");
    }
    else 
        tB = const_cast<NDArray*>(B); 
    char aOrder = tA->ordering();
    char bOrder = tB->ordering();
    char cOrder = tC->ordering();
    if (cOrder != rOrder) {
        pC = tC->dup('f');
    } else {
        pC = tC;
    }

// the lines in gemm.cpp for reference
//        bool transAFlag = TransA == CblasTrans;
//        bool transBFlag = TransB == CblasTrans;
    if (tB->ews() == -1) {
        pB = tB->dup('f');
        transB = CblasNoTrans;
    }
    else 
        pB = tB; //->dup('f');
    if (tA->ews() == -1) {
        pA = tA->dup('c');
        transA = CblasNoTrans;
    }
    else 
        pA = tA; //->dup('c');
    
    lda = pA->ordering() == 'f' ? pA->rows() : pA->columns();
    ldb = pB->ordering() == 'f' ? pB->rows() : pB->columns();
    ldc = pC->rows();
    transA = (pA->ordering() == 'c'? CblasTrans:CblasNoTrans);
    transB = (pB->ordering() == 'c' ? CblasTrans:CblasNoTrans);

    auto xType = A->dataType();
    auto yType = B->dataType();
    auto zType = result->dataType();

    // we'll use platform-specific gemm here eventually. maybe tomorrow.
    // TODO: put proper _gemm here
    if (xType == yType && yType == zType && BlasHelper::getInstance()->template hasGEMM<X>()) {
        nd4j_debug("Using provided GEMM pointer\n","");
        if (xType == FLOAT32)
            BlasHelper::getInstance()->sgemm()(CblasColMajor, transA, transB, M, N, K, (float) alpha, reinterpret_cast<float *>(pA->getBuffer()), lda, reinterpret_cast<float *>(pB->getBuffer()), ldb, (float) beta, reinterpret_cast<float *>(pC->getBuffer()), ldc);
        else if (xType == DOUBLE)
            BlasHelper::getInstance()->dgemm()(CblasColMajor, transA, transB, M, N, K, (double) alpha, reinterpret_cast<double *>(pA->getBuffer()), lda, reinterpret_cast<double *>(pB->getBuffer()), ldb, (double) beta, reinterpret_cast<double *>(pC->getBuffer()), ldc);
        else {
            BUILD_TRIPLE_SELECTOR(xType, yType, zType, nd4j::blas::GEMM, ::op(rOrder, transA, transB, M, N, K, alpha, pA->getBuffer(), lda, pB->getBuffer(), ldb, beta, pC->getBuffer(), ldc), LIBND4J_TYPES, FLOAT_TYPES, FLOAT_TYPES);
        }
    } else {
        nd4j_debug("mmulMxM: Using fallback GEMM impl\n","");
       
        nd4j::blas::GEMM<X, Y, Z>::op(rOrder, transA, transB, M, N, K, alpha, pA->getBuffer(), lda, pB->getBuffer(), ldb, beta, pC->getBuffer(), ldc);
    }
    if (tC != pC) {
        tC->assign(pC);
    }
    if (tA != pA)
        delete pA;
    if (tB != pB)
        delete pB;
    if (tC != pC)
        delete pC;
    if (tA != A)
        delete tA;
    if (tB != B)
        delete tB;
    return result;
}

    BUILD_TRIPLE_TEMPLATE(template nd4j::NDArray* MmulHelper::mmulMxM, (const NDArray* A, const NDArray* B, nd4j::NDArray* C, double alpha, double beta), LIBND4J_TYPES, FLOAT_TYPES, FLOAT_TYPES);

}
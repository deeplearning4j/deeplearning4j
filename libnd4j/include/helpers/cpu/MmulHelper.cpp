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
// all arrays must to be in f order and have continuous buffer
void MmulHelper::basicGemm(const NDArray* A, const NDArray* B, NDArray* C, double alpha, double beta) {

    const int M = A->sizeAt(0);
    const int K = A->sizeAt(1);
    const int N = B->sizeAt(1);

    const auto aType = A->dataType();
    const auto bType = B->dataType();
    const auto cType = C->dataType();

    const bool AB(aType == bType), AC(aType == cType), ABC(AB && AC);
    const bool hasGemm = BlasHelper::getInstance()->hasGEMM(aType);

    // we'll use platform-specific gemm here eventually. maybe tomorrow.
    // TODO: put proper _gemm here
    if (ABC && hasGemm && aType == DataType::FLOAT32) {
        BlasHelper::getInstance()->sgemm()(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, (float) alpha, reinterpret_cast<float *>(A->getBuffer()), M, reinterpret_cast<float *>(B->getBuffer()), K, (float) beta, reinterpret_cast<float *>(C->getBuffer()), M);
    }
    else if (ABC && hasGemm && aType == DataType::DOUBLE) {
        BlasHelper::getInstance()->dgemm()(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, (double) alpha, reinterpret_cast<double *>(A->getBuffer()), M, reinterpret_cast<double *>(B->getBuffer()), K, (double) beta, reinterpret_cast<double *>(C->getBuffer()), M);
    }
    else {
        BUILD_TRIPLE_SELECTOR(aType, bType, cType, nd4j::blas::GEMM, ::op('f', CblasNoTrans, CblasNoTrans, M, N, K, alpha, A->getBuffer(), M, B->getBuffer(), K, beta, C->getBuffer(), M), LIBND4J_TYPES, FLOAT_TYPES, FLOAT_TYPES);
    }    
}


//////////////////////////////////////////////////////////////////////////////
// MXK x KxN = MxN
NDArray* MmulHelper::mmulMxM(const NDArray* A, const NDArray* B, NDArray* C, double alpha, double beta) {

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
        C = new NDArray('f', {M,N}, DataTypeUtils::pickPairwiseResultType(A->dataType(), B->dataType()), A->getContext());

    NDArray *pA(const_cast<NDArray*>(A)), *pB(const_cast<NDArray*>(B)), *pC(const_cast<NDArray*>(C));

    if(A->ews() != 1 || A->ordering() == 'c')
        pA = pA->dup('f');
    if(B->ews() != 1 || B->ordering() == 'c')
        pB = pB->dup('f');
    if(C->ews() != 1 || C->ordering() == 'c')
        pC = pC->dup('f');

    MmulHelper::basicGemm(pA, pB, pC, alpha, beta);

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


}
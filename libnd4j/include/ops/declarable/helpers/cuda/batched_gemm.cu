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
//  @author raver119@gmail.com
//  @author Yurii Shyrma (iuriish@yahoo.com)
//

#include <exceptions/cuda_exception.h>
#include <cublas_v2.h>
#include <specials_cuda.h>
#include <op_boilerplate.h>
#include <types/float16.h>
#include <ops/declarable/helpers/batched_gemm.h>
#include <PointersManager.h>


namespace nd4j {
namespace ops {
namespace helpers {

//////////////////////////////////////////////////////////////////////////////
// bsxMXK x bSxKxN = bSxMxN
void bgemm(const std::vector<NDArray*>& vA, const std::vector<NDArray*>& vB, std::vector<NDArray*>& vC, const NDArray* alphas, const NDArray* betas, int transA, int transB, int M, int N, int K, const int lda, const int ldb, const int ldc) {

    const auto bS = vA.size();      // batch size

    std::vector<NDArray*> pA(bS), pB(bS), pC(bS);

    std::vector<NDArray*> toDelete;

    for(int i = 0; i < bS; ++i) {

        if(vA[i]->ews() != 1) {
            pA[i] = vA[i]->dup('f');
            toDelete.emplace_back(pA[i]);
        }
        else
            pA[i] = vA[i];

        if(vB[i]->ews() != 1) {
            pB[i] = vB[i]->dup('f');
            toDelete.emplace_back(pB[i]);
        }
        else
            pB[i] = vB[i];

        if(vC[i]->ews() != 1) {
            pC[i] = vC[i]->dup('f');
            toDelete.emplace_back(pC[i]);
        }
        else
            pC[i] = vC[i];

        if(pC[i]->ordering() != 'f') {
            auto temp = pA[i];
            pA[i] = pB[i]->permute({1,0});
            pB[i] = temp ->permute({1,0});
            pC[i] = pC[i]->permute({1,0});
            toDelete.push_back(pA[i]);
            toDelete.push_back(pB[i]);
            toDelete.push_back(pC[i]);
            M = pA[i]->sizeAt(0);
            K = pA[i]->sizeAt(1);
            N = pB[i]->sizeAt(1);
        }

        NDArray::prepareSpecialUse ({pC[i]}, {pA[i], pB[i]});
        NDArray::registerSpecialUse({pC[i]}, {pA[i], pB[i]});
    }

    NDArray::prepareSpecialUse ({}, {alphas, betas});
    NDArray::registerSpecialUse({}, {alphas, betas});

    std::vector<void*> pAbuffs(bS), pBbuffs(bS), pCbuffs(bS);
    for(int i = 0; i < bS; ++i) {
        pAbuffs[i] = pA[i]->getSpecialBuffer();
        pBbuffs[i] = pB[i]->getSpecialBuffer();
        pCbuffs[i] = pC[i]->getSpecialBuffer();
    }

    nd4j::LaunchContext* context = vA[0]->getContext();
    PointersManager manager(context, "helpers::bgemm cuda");

    const void** aBuffers = reinterpret_cast<const void**>(manager.replicatePointer(pAbuffs.data(), bS * sizeof(void*)));
    const void** bBuffers = reinterpret_cast<const void**>(manager.replicatePointer(pBbuffs.data(), bS * sizeof(void*)));
          void** cBuffers = reinterpret_cast<void**>(manager.replicatePointer(pCbuffs.data(), bS * sizeof(void*)));

    // const auto aOrder = pA->ordering();
    // const auto bOrder = pB->ordering();

    // const bool transA = aOrder != 'f';
    // const bool transB = bOrder != 'f';

    const cublasOperation_t transAblas = transA == 112 ? CUBLAS_OP_T : CUBLAS_OP_N;
    const cublasOperation_t transBblas = transB == 112 ? CUBLAS_OP_T : CUBLAS_OP_N;

    // const int lda = aOrder == 'f' ? M : K;
    // const int ldb = bOrder == 'f' ? K : N;
    // const int ldc = M; // cOrder == 'f' ? M : N;

    const auto aType = pA[0]->dataType();
    const auto bType = pB[0]->dataType();
    const auto cType = pC[0]->dataType();

    auto handle = reinterpret_cast<cublasHandle_t*>(context->getCublasHandle());
    auto stream = context->getCudaStream();

    auto status = cublasSetStream_v2(*handle, *stream);

    if (status != CUBLAS_STATUS_SUCCESS)
        throw cuda_exception::build("MmulHelper::mmulMxM cuda failed !", status);

    const bool AB(aType == bType), AC(aType == cType), ABC(AB && AC);

    // choose appropriate cuda gemm api depending on data types
    if(ABC && aType == DataType::DOUBLE) {
        double alpha = alphas->e<double>(0);
        double beta  = betas->e<double>(0);
        status = cublasDgemmBatched(*handle, transAblas, transBblas, M, N, K, &alpha, (const double**)aBuffers, lda, (const double**)bBuffers, ldb, &beta, (double**)cBuffers, ldc, bS);
    }
    else if(ABC && aType == DataType::FLOAT32) {
        float alpha = alphas->e<float>(0);
        float beta  = betas->e<float>(0);
        status = cublasSgemmBatched(*handle, transAblas, transBblas, M, N, K, &alpha, (const float**)aBuffers, lda, (const float**)bBuffers, ldb, &beta, (float**)cBuffers, ldc, bS);
    }
    else if(ABC && aType == DataType::HALF) {
        __half alpha = alphas->e<float>(0);
        __half beta  = betas->e<float>(0);
        status = cublasHgemmBatched(*handle, transAblas, transBblas, M, N, K, &alpha, (const __half**)aBuffers, lda, (const __half**)bBuffers, ldb, &beta, (__half**)cBuffers, ldc, bS);
    }
    else if(AB && aType == DataType::INT8 && cType == DataType::FLOAT32) {
        float alpha = alphas->e<float>(0);
        float beta  = betas->e<float>(0);
        status = cublasGemmBatchedEx(*handle, transAblas, transBblas, M, N, K, &alpha, aBuffers, CUDA_R_8I, lda, bBuffers, CUDA_R_8I, ldb, &beta, cBuffers, CUDA_R_32F, ldc, bS, CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
    }
    else if(AB && aType == DataType::HALF && cType == DataType::FLOAT32) {
        float alpha = alphas->e<float>(0);
        float beta  = betas->e<float>(0);
        status = cublasGemmBatchedEx(*handle, transAblas, transBblas, M, N, K, &alpha, aBuffers, CUDA_R_16F, lda, bBuffers, CUDA_R_16F, ldb, &beta, cBuffers, CUDA_R_32F, ldc, bS, CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
    }
    else
        throw std::runtime_error("batched gemm cuda: this mode is not implemented yet !");


    if (status != CUBLAS_STATUS_SUCCESS)
        throw cuda_exception::build("MmulHelper::mmulMxM cuda failed !", status);

    auto cudaResult = cudaStreamSynchronize(*stream);
    if (cudaResult != 0)
        throw cuda_exception::build("MmulHelper::mmulMxM cuda failed !", cudaResult);

    for(int i = 0; i < bS; ++i)
    if(vC[i]->ews() != 1)
        vC[i]->assign(pC[i]);

    for(int i = toDelete.size() - 1; i >= 0; --i)
        delete toDelete[i];
}

}
}
}
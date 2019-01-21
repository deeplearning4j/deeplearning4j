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

#include <exceptions/cuda_exception.h>
#include <cublas_v2.h>
#include "../MmulHelper.h"
#include <specials_cuda.h>


namespace nd4j { 


//////////////////////////////////////////////////////////////////////////////
// MXK x KxN = MxN
// C array must be in f order
template <typename X, typename Y, typename Z>
static __global__ void usualCudaGemm(const bool transA, const bool transB, const int M, const int N, const int K, const double alpha, const void* vA, const int lda, const void* vB, const int ldb, const double beta, void* vC, const int ldc) {

    X* A = reinterpret_cast<X*>(const_cast<void*>(vA));
    Y* B = reinterpret_cast<Y*>(const_cast<void*>(vB));
    Z* C = reinterpret_cast<Z*>(vC);     

    __shared__ Z alphaZ, betaZ;
    __shared__ Nd4jLong strideArow, strideAcol, strideBrow, strideBcol;

    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row == 0 && col == 0) {        
        
        alphaZ = alpha;
        betaZ  = beta;

        if(transA) { strideArow = lda; strideAcol = 1; } else { strideArow = 1; strideAcol = lda; }
        if(transB) { strideBrow = ldb; strideBcol = 1; } else { strideBrow = 1; strideBcol = ldb; }
    }

    __syncthreads();

    Z val = 0;
    if (row < M && col < N)         
        for (int i = 0; i < K; i++)             
            val = val + A[row * strideArow + i * strideAcol] * B[i * strideBrow + col * strideBcol];
            
    C[row + col * ldc] = alphaZ * val + betaZ * C[row + col * ldc];    
}

////////////////////////////////////////////////////////////////////////
template <typename X, typename Y, typename Z>    
__host__ static void usualGemm(const dim3 &blocksPerGrid, const dim3 &threadsPerBlock, cudaStream_t *stream, const bool transA, const bool transB, const int M, const int N, const int K, const double alpha, const void* vA, const int lda, const void* vB, const int ldb, const double beta, void* vC, const int ldc) {
    
    usualCudaGemm<X,Y,Z><<<blocksPerGrid, threadsPerBlock, 1024, *stream>>>(transA, transB, M, N, K, alpha, vA, lda, vB, ldb, beta, vC, ldc);
}

//////////////////////////////////////////////////////////////////////////////
// MXK x KxN = MxN
NDArray* MmulHelper::mmulMxM(const NDArray* A, const NDArray* B, NDArray* C, double alpha, double beta, const char outOrder) {

	if(A->rankOf() != 2)
		throw std::runtime_error("MmulHelper::mmulMxM cuda: rank of A array is not equal 2 !");
	if(B->rankOf() != 2)
		throw std::runtime_error("MmulHelper::mmulMxM cuda: rank of B array is not equal 2 !");
	if(C != nullptr && C->rankOf() != 2)
		throw std::runtime_error("MmulHelper::mmulMxM cuda: rank of C array is not equal 2 !");

	const auto M = A->sizeAt(0);
	const auto K = A->sizeAt(1);
	const auto N = B->sizeAt(1);

	if(B->sizeAt(0) != K)
		throw std::runtime_error("MmulHelper::mmulMxM cuda: B array has wrong number of rows !");
	if(C != nullptr && C->sizeAt(0) != M)
		throw std::runtime_error("MmulHelper::mmulMxM cuda: C array has wrong number of rows !");
	if(C != nullptr && C->sizeAt(1) != N)
		throw std::runtime_error("MmulHelper::mmulMxM cuda: C array has wrong number of columns !");

	if(C == nullptr) 		
        C = new NDArray(outOrder, {M,N}, DataTypeUtils::pickPairwiseResultType(A->dataType(), B->dataType()), A->getContext());

	NDArray *pA(const_cast<NDArray*>(A)), *pB(const_cast<NDArray*>(B)), *pC(const_cast<NDArray*>(C));        

    if(A->ews() != 1)
        pA = pA->dup('f');
    if(B->ews() != 1)
        pB = pB->dup('f');
    if(C->ews() != 1 || C->ordering() != 'f')
        pC = pC->dup('f');

    const auto aOrder = pA->ordering();
    const auto bOrder = pB->ordering();    

    const bool transA = aOrder != 'f';
    const bool transB = bOrder != 'f';
    
    const cublasOperation_t transAblas = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
    const cublasOperation_t transBblas = transB ? CUBLAS_OP_T : CUBLAS_OP_N;

    const int lda = aOrder == 'f' ? M : K;
    const int ldb = bOrder == 'f' ? K : N;
    const int ldc = M; // cOrder == 'f' ? M : N;    

    const auto aType = pA->dataType();
    const auto bType = pB->dataType();
    const auto cType = pC->dataType();

    if(!pA->isActualOnDeviceSide())
        pA->syncToDevice();
    if(!pB->isActualOnDeviceSide())
        pB->syncToDevice();
    if(!pC->isActualOnDeviceSide())
        pC->syncToDevice();    

    cublasStatus_t status;
    cublasHandle_t handle;

    cudaStream_t* stream = A->getContext()->getCudaStream();

    status = cublasCreate_v2(&handle); // initialize CUBLAS context
    if (status != CUBLAS_STATUS_SUCCESS) throw cuda_exception::build("MmulHelper::mmulMxM cuda failed !", status);

    status = cublasSetStream_v2(handle, *stream);
    if (status != CUBLAS_STATUS_SUCCESS) throw cuda_exception::build("MmulHelper::mmulMxM cuda failed !", status);

    const bool AB(aType == bType), AC(aType == cType), ABC(AB && AC);

    // choose appropriate cuda gemm api depending on data types    
    if(ABC && aType == DataType::DOUBLE) {
        status = cublasDgemm(handle, transAblas, transBblas, M, N, K, &alpha, (double*)pA->getSpecialBuffer(), lda, (double*)pB->getSpecialBuffer(), ldb, &beta, (double*)pC->getSpecialBuffer(), ldc);
    }
    else if(ABC && aType == DataType::FLOAT32) {        
        float alphaF(alpha), betaF(beta);
        status = cublasSgemm(handle, transAblas, transBblas, M, N, K, &alphaF, (float*)pA->getSpecialBuffer(), lda, (float*)pB->getSpecialBuffer(), ldb, &betaF, (float*)pC->getSpecialBuffer(), ldc);
    }
    else if(ABC && aType == DataType::HALF) {
        float16 alphaH(alpha), betaH(beta);
        status = cublasHgemm(handle, transAblas, transBblas, M, N, K, &alphaH.data, (__half*)pA->getSpecialBuffer(), lda, (__half*)pB->getSpecialBuffer(), ldb, &betaH.data, (__half*)pC->getSpecialBuffer(), ldc);
    }       
    else if(AB && aType == DataType::INT8 && cType == DataType::FLOAT32) {            
           float alphaF(alpha), betaF(beta);
           status = cublasSgemmEx(handle, transAblas, transBblas, M, N, K, &alphaF, pA->getSpecialBuffer(), CUDA_R_8I, lda, pB->getSpecialBuffer(), CUDA_R_8I, ldb, &betaF, pC->getSpecialBuffer(), CUDA_R_32F, ldc);
    }
    else if(AB && aType == DataType::HALF && cType == DataType::FLOAT32) {
        float alphaF(alpha), betaF(beta);
        status = cublasSgemmEx(handle, transAblas, transBblas, M, N, K, &alphaF, pA->getSpecialBuffer(), CUDA_R_16F, lda, pB->getSpecialBuffer(), CUDA_R_16F, ldb, &betaF, pC->getSpecialBuffer(), CUDA_R_32F, ldc);
    }    
    else {        
        dim3 threadsPerBlock(N, M);
        dim3 blocksPerGrid(1, 1);
        if (M*N > 512){
            threadsPerBlock.x = threadsPerBlock.y = 512;             
            blocksPerGrid.x = math::nd4j_ceil<double, int>(static_cast<double>(N) / threadsPerBlock.x);    // cols
            blocksPerGrid.y = math::nd4j_ceil<double, int>(static_cast<double>(M) / threadsPerBlock.y);    // rows
        }

        BUILD_TRIPLE_SELECTOR(aType, bType, cType, usualGemm, (blocksPerGrid, threadsPerBlock, stream, transA, transB, M, N, K, alpha, pA->getSpecialBuffer(), lda, pB->getSpecialBuffer(), ldb, beta, pC->getSpecialBuffer(), ldc), LIBND4J_TYPES, FLOAT_TYPES, FLOAT_TYPES);
    }   
   
    if (status != CUBLAS_STATUS_SUCCESS) throw cuda_exception::build("MmulHelper::mmulMxM cuda failed !", status);

    auto cudaResult = cudaStreamSynchronize(*stream);
    if (cudaResult != 0) throw cuda_exception::build("MmulHelper::mmulMxM cuda failed !", cudaResult);
   
    cublasDestroy(handle);    

    pA->tickReadDevice();
    pB->tickReadDevice();
    pC->tickWriteDevice();

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
// MXN x N = M
template <typename X, typename Y, typename Z>
NDArray* MmulHelper::mmulMxV(const NDArray* A, const NDArray* B, nd4j::NDArray* C, const double alpha, const double beta, const char outOrder) {

    int bLenDim, cLenDim;

    if(A->rankOf() != 2)
        throw std::runtime_error("MmulHelper::mmulMxV cuda: rank of A array is not equal 2 !");
    if(!B->isVector() && !shape::isCommonVector(B->getShapeInfo(), bLenDim))
        throw std::runtime_error("MmulHelper::mmulMxV cuda: B array must be vector !");
    if(C != nullptr && !C->isVector() && !shape::isCommonVector(C->getShapeInfo(), cLenDim))
        throw std::runtime_error("MmulHelper::mmulMxV cuda: C array must be vector !");

    const auto M = A->sizeAt(0);    
    const auto N = A->sizeAt(1);

    if(B->lengthOf() != N)
        throw std::runtime_error("MmulHelper::mmulMxV cuda: B vector has wrong length !");
    if(C != nullptr && C->lengthOf() != M)
        throw std::runtime_error("MmulHelper::mmulMxV cuda: C array has wrong length !");    

    if(C == nullptr)        
        C = new NDArray(outOrder, {M}, DataTypeUtils::pickPairwiseResultType(A->dataType(), B->dataType()), A->getContext());
    
    NDArray *pA(const_cast<NDArray*>(A));

    if(A->ews() != 1 || A->ordering() != 'f')
        pA = pA->dup('f');

    // const auto aOrder = pA->ordering();
    // const auto bOrder = pB->ordering();    

    // const bool transA = aOrder != 'f';
    // const bool transB = bOrder != 'f';
    
    // const cublasOperation_t transAblas = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
    // const cublasOperation_t transBblas = transB ? CUBLAS_OP_T : CUBLAS_OP_N;

    // const int lda = aOrder == 'f' ? M : K;
    // const int ldb = bOrder == 'f' ? K : N;
    // const int ldc = M; // cOrder == 'f' ? M : N;    

    // const auto aType = pA->dataType();
    // const auto bType = pB->dataType();
    // const auto cType = pC->dataType();


    //     auto xType = A->dataType();
    //     auto yType = B->dataType();
    //     auto zType = result->dataType();

        // // TODO: strides!!!
        // if (xType == yType && xType == zType && BlasHelper::getInstance()->hasGEMV<X>()) {
        //     nd4j_debug("Using provided GEMV pointer\n","");
        //     auto layout = A->ordering() == 'f' ? CblasColMajor : CblasRowMajor;
        //     if (std::is_same<X, float>::value)
        //         BlasHelper::getInstance()->sgemv()(layout, CblasNoTrans, A->rows(), A->columns(), (float) alpha, reinterpret_cast<float *>(A->getBuffer()), layout == CblasColMajor ? A->rows() : A->columns(), reinterpret_cast<float *>(B->getBuffer()), 1, (float) beta, reinterpret_cast<float *>(result->getBuffer()), 1);
        //     else if (std::is_same<X, double>::value)
        //         BlasHelper::getInstance()->dgemv()(layout, CblasNoTrans, A->rows(), A->columns(), (double) alpha, reinterpret_cast<double *>(A->getBuffer()), layout == CblasColMajor ? A->rows() : A->columns(), reinterpret_cast<double *>(B->getBuffer()), 1, (double) beta, reinterpret_cast<double *>(result->getBuffer()), 1);
        //     else
        //         nd4j::blas::GEMV<X, Y, Z>::op(A->ordering() == 'f' ? CblasTrans : 0, A->rows(), A->columns(), alpha, A->getBuffer(), B->lengthOf(), B->getBuffer(), 1, beta, result->getBuffer(), 1);
        // } else {
        //     nd4j_debug("Using fallback GEMV impl\n","");
        //     nd4j::blas::GEMV<X, Y, Z>::op(A->ordering() == 'f' ? CblasTrans : 0, A->rows(), A->columns(), alpha, A->getBuffer(), B->lengthOf(), B->getBuffer(), 1, beta, result->getBuffer(), 1);
        // }
    return C;
}


BUILD_TRIPLE_TEMPLATE(template void usualGemm, (const dim3 &blocksPerGrid, const dim3 &threadsPerBlock, cudaStream_t *stream, const bool transA, const bool transB, const int M, const int N, const int K, const double alpha, const void* vA, const int lda, const void* vB, const int ldb, const double beta, void* vC, const int ldc), LIBND4J_TYPES, FLOAT_TYPES, FLOAT_TYPES);
BUILD_TRIPLE_TEMPLATE(template NDArray* MmulHelper::mmulMxV, (const NDArray* A, const NDArray* B, NDArray* C, const double alpha, const double beta, const char outOrder), LIBND4J_TYPES, FLOAT_TYPES, FLOAT_TYPES);
}

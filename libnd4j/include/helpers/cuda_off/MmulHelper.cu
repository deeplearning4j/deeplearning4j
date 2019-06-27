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
#include <helpers/PointersManager.h>

namespace nd4j {


//////////////////////////////////////////////////////////////////////////////
// MXK x KxN = MxN
// C array must be in f order
template <typename T1, typename T2, typename T3>
static __global__ void usualCudaGemm(const bool transA, const bool transB, const int M, const int N, const int K, const double alpha, const void* vA, const int lda, const void* vB, const int ldb, const double beta, void* vC, const int ldc) {

    T1* A = reinterpret_cast<T1*>(const_cast<void*>(vA));
    T2* B = reinterpret_cast<T2*>(const_cast<void*>(vB));
    T3* C = reinterpret_cast<T3*>(vC);

    __shared__ T3 alphaZ, betaZ;
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

    T3 val = 0;
    if (row < M && col < N)
        for (int i = 0; i < K; i++)
            val = val + A[row * strideArow + i * strideAcol] * B[i * strideBrow + col * strideBcol];

    C[row + col * ldc] = alphaZ * val + betaZ * C[row + col * ldc];
}

////////////////////////////////////////////////////////////////////////
template <typename T1, typename T2, typename T3>
__host__ static void usualGemm(const dim3 &blocksPerGrid, const dim3 &threadsPerBlock, cudaStream_t *stream, const bool transA, const bool transB, const int M, const int N, const int K, const double alpha, const void* vA, const int lda, const void* vB, const int ldb, const double beta, void* vC, const int ldc) {

    usualCudaGemm<T1,T2,T3><<<blocksPerGrid, threadsPerBlock, 1024, *stream>>>(transA, transB, M, N, K, alpha, vA, lda, vB, ldb, beta, vC, ldc);
}

//////////////////////////////////////////////////////////////////////////////
// MXN x N = M
template <typename T1, typename T2, typename T3>
static __global__ void usualCudaGemv(const bool transA, const int M, const int N, const double alpha, const void* vA, const int lda, const void* vX, const int incx, const double beta, void* vY, const int incy) {

    T1* A = reinterpret_cast<T1*>(const_cast<void*>(vA));
    T2* X = reinterpret_cast<T2*>(const_cast<void*>(vX));
    T3* Y = reinterpret_cast<T3*>(vY);

    __shared__ T3 alphaZ, betaZ;
    __shared__ Nd4jLong strideArow, strideAcol;

    const int row = blockIdx.x * blockDim.x + threadIdx.x;

    if(row == 0) {

        alphaZ = alpha;
        betaZ  = beta;

        if(transA) { strideArow = lda; strideAcol = 1; } else { strideArow = 1; strideAcol = lda; }
    }

    __syncthreads();

    T3 val = 0;
    if (row < M)
        for (int i = 0; i < N; i++) {
            val = val + A[row * strideArow + i * strideAcol] * X[i * incx];
        }

    Y[row * incy] = alphaZ * val + betaZ * Y[row * incy];
}

////////////////////////////////////////////////////////////////////////
template <typename T1, typename T2, typename T3>
__host__ static void usualGemv(const dim3 &blocksPerGrid, const dim3 &threadsPerBlock, cudaStream_t *stream, const bool transA, const int M, const int N, const double alpha, const void* vA, const int lda, const void* vX, const int incx, const double beta, void* vY, const int incy) {

    usualCudaGemv<T1,T2,T3><<<blocksPerGrid, threadsPerBlock, 1024, *stream>>>(transA, M, N, alpha, vA, lda, vX, incx, beta, vY, incy);
}

//////////////////////////////////////////////////////////////////////////////
template <typename T1, typename T2, typename T3>
static __global__ void usualCudaDot(const Nd4jLong length, const double alpha, const void* vX, const Nd4jLong incx, const void* vY, const Nd4jLong incy, const double beta, void* vZ) {

    T1* X = reinterpret_cast<T1*>(const_cast<void*>(vX));
    T2* Y = reinterpret_cast<T2*>(const_cast<void*>(vY));
    T3* Z = reinterpret_cast<T3*>(vZ);

    extern __shared__ char shmem[];
    auto pairwiseMul = reinterpret_cast<T3*>(shmem);

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < length)
        pairwiseMul[tid] = X[tid * incx] * Y[tid * incy];

    __syncthreads();

    if(tid == 0) {
        T3 sum = 0;
        for(Nd4jLong i = 0; i < length; ++i)
            sum = sum + pairwiseMul[i];
        *Z = (T3)alpha * sum + (T3)beta * *Z;
    }
}

////////////////////////////////////////////////////////////////////////
template <typename T1, typename T2, typename T3>
__host__ static void usualDot(const dim3 &blocksPerGrid, const dim3 &threadsPerBlock, cudaStream_t *stream, const Nd4jLong length, const double alpha, const void* vX, const Nd4jLong incx, const void* vY, const Nd4jLong incy, const double beta, void* vZ) {

    usualCudaDot<T1,T2,T3><<<blocksPerGrid, threadsPerBlock, length*sizeof(T3) + 128, *stream>>>(length, alpha, vX, incx, vY, incy, beta, vZ);
}

//////////////////////////////////////////////////////////////////////////////
// MXK x KxN = MxN
NDArray* MmulHelper::mmulMxM(const NDArray* A, const NDArray* B, NDArray* C, double alpha, double beta, const char outOrder) {

    if(A->rankOf() != 2)
        throw std::runtime_error("MmulHelper::mmulMxM cuda: rank of A array is not equal 2 !");
    if(B->rankOf() != 2)
        throw std::runtime_error("MmulHelper::mmulMxM cuda: rank of B array is not equal 2 !");

    auto M = A->sizeAt(0);
    auto K = A->sizeAt(1);
    auto N = B->sizeAt(1);

    if(C != nullptr && C->rankOf() != 2)
        throw std::runtime_error("MmulHelper::mmulMxM cuda: rank of C array is not equal 2 !");
    if(B->sizeAt(0) != K)
        throw std::runtime_error("MmulHelper::mmulMxM cuda: B array has wrong number of rows !");
    if(C != nullptr && C->sizeAt(0) != M)
        throw std::runtime_error("MmulHelper::mmulMxM cuda: C array has wrong number of rows !");
    if(C != nullptr && C->sizeAt(1) != N)
        throw std::runtime_error("MmulHelper::mmulMxM cuda: C array has wrong number of columns !");

    if(C == nullptr)
        C = new NDArray(outOrder, {M,N}, DataTypeUtils::pickPairwiseResultType(A->dataType(), B->dataType()), A->getContext());

    NDArray *pA(const_cast<NDArray*>(A)), *pB(const_cast<NDArray*>(B)), *pC(const_cast<NDArray*>(C));
    std::vector<NDArray*> toDelete;

    if(A->ews() != 1) {
        pA = pA->dup('f');
        toDelete.push_back(pA);
    }
    if(B->ews() != 1) {
        pB = pB->dup('f');
        toDelete.push_back(pB);
    }
    if(C->ews() != 1) {
        pC = pC->dup('f');
        toDelete.push_back(pC);
    }

    if(pC->ordering() != 'f') {
        auto temp = pA;
        pA = new NDArray(pB  ->permute({1,0}));
        pB = new NDArray(temp->permute({1,0}));
        pC = new NDArray(pC  ->permute({1,0}));
        toDelete.push_back(pA);
        toDelete.push_back(pB);
        toDelete.push_back(pC);
        M = pA->sizeAt(0);
        K = pA->sizeAt(1);
        N = pB->sizeAt(1);
    }

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

    auto handle = reinterpret_cast<cublasHandle_t *>(A->getContext()->getCublasHandle());
    auto stream = A->getContext()->getCudaStream();

    auto status = cublasSetStream_v2(*handle, *stream);
    if (status != CUBLAS_STATUS_SUCCESS) throw cuda_exception::build("MmulHelper::mmulMxM cuda failed !", status);

    const bool AB(aType == bType), AC(aType == cType), ABC(AB && AC);

    NDArray::prepareSpecialUse({pC}, {pA, pB});

    // choose appropriate cuda gemm api depending on data types
    if(ABC && aType == DataType::DOUBLE) {
        status = cublasDgemm(*handle, transAblas, transBblas, M, N, K, &alpha, (double*)pA->getSpecialBuffer(), lda, (double*)pB->getSpecialBuffer(), ldb, &beta, (double*)pC->getSpecialBuffer(), ldc);
    }
    else if(ABC && aType == DataType::FLOAT32) {
        float alphaF(alpha), betaF(beta);
        status = cublasSgemm(*handle, transAblas, transBblas, M, N, K, &alphaF, (float*)pA->getSpecialBuffer(), lda, (float*)pB->getSpecialBuffer(), ldb, &betaF, (float*)pC->getSpecialBuffer(), ldc);
    }
    else if(ABC && aType == DataType::HALF) {
        float16 alphaH(alpha), betaH(beta);
        status = cublasHgemm(*handle, transAblas, transBblas, M, N, K, &alphaH.data, (__half*)pA->getSpecialBuffer(), lda, (__half*)pB->getSpecialBuffer(), ldb, &betaH.data, (__half*)pC->getSpecialBuffer(), ldc);
    }
    else if(AB && aType == DataType::INT8 && cType == DataType::FLOAT32) {
           float alphaF(alpha), betaF(beta);
           status = cublasSgemmEx(*handle, transAblas, transBblas, M, N, K, &alphaF, pA->getSpecialBuffer(), CUDA_R_8I, lda, pB->getSpecialBuffer(), CUDA_R_8I, ldb, &betaF, pC->getSpecialBuffer(), CUDA_R_32F, ldc);
    }
    else if(AB && aType == DataType::HALF && cType == DataType::FLOAT32) {
        float alphaF(alpha), betaF(beta);
        status = cublasSgemmEx(*handle, transAblas, transBblas, M, N, K, &alphaF, pA->getSpecialBuffer(), CUDA_R_16F, lda, pB->getSpecialBuffer(), CUDA_R_16F, ldb, &betaF, pC->getSpecialBuffer(), CUDA_R_32F, ldc);
    }
    else {
        dim3 threadsPerBlock(N, M);
        dim3 blocksPerGrid(1, 1);
        if (M*N > 512){
            threadsPerBlock.x = threadsPerBlock.y = 512;
            blocksPerGrid.x = math::nd4j_ceil<double, int>(static_cast<double>(N) / threadsPerBlock.x);    // cols
            blocksPerGrid.y = math::nd4j_ceil<double, int>(static_cast<double>(M) / threadsPerBlock.y);    // rows
        }

        //BUILD_TRIPLE_SELECTOR(aType, bType, cType, usualGemm, (blocksPerGrid, threadsPerBlock, stream, transA, transB, M, N, K, alpha, pA->getSpecialBuffer(), lda, pB->getSpecialBuffer(), ldb, beta, pC->getSpecialBuffer(), ldc), LIBND4J_TYPES, FLOAT_TYPES, FLOAT_TYPES);
        BUILD_SINGLE_SELECTOR_THRICE(aType, usualGemm, (blocksPerGrid, threadsPerBlock, stream, transA, transB, M, N, K, alpha, pA->getSpecialBuffer(), lda, pB->getSpecialBuffer(), ldb, beta, pC->getSpecialBuffer(), ldc), LIBND4J_TYPES)
    }

    if (status != CUBLAS_STATUS_SUCCESS) throw cuda_exception::build("MmulHelper::mmulMxM cuda failed !", status);

    auto cudaResult = cudaStreamSynchronize(*stream);
    if (cudaResult != 0) throw cuda_exception::build("MmulHelper::mmulMxM cuda failed !", cudaResult);

    NDArray::registerSpecialUse({pC}, {pA, pB});

    if(C->ews() != 1)
        C->assign(pC);

    for(int i = toDelete.size() - 1; i >= 0; --i)
        delete toDelete[i];

    return C;
}

////////////////////////////////////////////////////////////////////////////
// MXN x N = M
NDArray* MmulHelper::mmulMxV(const NDArray* A, const NDArray* X, nd4j::NDArray* Y, const double alpha, const double beta, const char outOrder) {

    int xLenDim, yLenDim(0);

    if(A->rankOf() != 2)
        throw std::runtime_error("MmulHelper::mmulMxV cuda: rank of A array is not equal 2 !");
    if(!shape::isCommonVector(X->getShapeInfo(), xLenDim))
        throw std::runtime_error("MmulHelper::mmulMxV cuda: X array must be vector !");

    const auto M = A->sizeAt(0);
    const auto N = A->sizeAt(1);

    if(Y != nullptr && !shape::isCommonVector(Y->getShapeInfo(), yLenDim))
        throw std::runtime_error("MmulHelper::mmulMxV cuda: Y array must be vector !");
    if(X->lengthOf() != N)
        throw std::runtime_error("MmulHelper::mmulMxV cuda: X vector has wrong length !");
    if(Y != nullptr && Y->lengthOf() != M)
        throw std::runtime_error("MmulHelper::mmulMxV cuda: Y array has wrong length !");

    if(Y == nullptr)
        Y = new NDArray(outOrder, {M}, DataTypeUtils::pickPairwiseResultType(A->dataType(), X->dataType()), A->getContext());

    NDArray *pA(const_cast<NDArray*>(A));

    if(A->ews() != 1)
        pA = pA->dup('f');

    const bool transA = pA->ordering() == 'c';

    const cublasOperation_t transAblas = transA ? CUBLAS_OP_T : CUBLAS_OP_N;

    int lda, lta;
    if(transA) { lda = N; lta = M; }
    else       { lda = M; lta = N; }

    const int incx = X->stridesOf()[xLenDim];
    const int incy = Y->stridesOf()[yLenDim];

    const auto aType = pA->dataType();
    const auto xType = X->dataType();
    const auto yType = Y->dataType();

    auto handle = reinterpret_cast<cublasHandle_t *>(A->getContext()->getCublasHandle());
    auto stream = A->getContext()->getCudaStream();

    auto status = cublasSetStream_v2(*handle, *stream);
    if (status != CUBLAS_STATUS_SUCCESS) throw cuda_exception::build("MmulHelper::mmulMxV cuda failed !", status);

    const bool AX(aType == xType), AY(aType == yType), AXY(AX && AY);

    NDArray::prepareSpecialUse({Y}, {pA, X});

    // choose appropriate cuda gemm api depending on data types
    if(AXY && aType == DataType::DOUBLE) {
        status = cublasDgemv(*handle, transAblas, lda, lta, &alpha, (double*)pA->getSpecialBuffer(), lda, (double*)X->getSpecialBuffer(), incx, &beta, (double*)Y->getSpecialBuffer(), incy);
    }
    else if(AXY && aType == DataType::FLOAT32) {
        float alphaF(alpha), betaF(beta);
        status = cublasSgemv(*handle, transAblas, lda, lta, &alphaF, (float*)pA->getSpecialBuffer(), lda, (float*)X->getSpecialBuffer(), incx, &betaF, (float*)Y->getSpecialBuffer(), incy);
    }
    else {
        dim3 threadsPerBlock(M);
        dim3 blocksPerGrid(1);
        if (M > 512){
            threadsPerBlock.x = 512;
            blocksPerGrid.x = math::nd4j_ceil<double, int>(static_cast<double>(M) / threadsPerBlock.x);    // rows
        }
        //BUILD_TRIPLE_SELECTOR(aType, xType, yType, usualGemv, (blocksPerGrid, threadsPerBlock, stream, transA, M, N, alpha, pA->getSpecialBuffer(), lda, X->getSpecialBuffer(), incx, beta, Y->getSpecialBuffer(), incy), LIBND4J_TYPES, FLOAT_TYPES, FLOAT_TYPES);
        BUILD_SINGLE_SELECTOR_THRICE(xType, usualGemv, (blocksPerGrid, threadsPerBlock, stream, transA, M, N, alpha, pA->getSpecialBuffer(), lda, X->getSpecialBuffer(), incx, beta, Y->getSpecialBuffer(), incy), LIBND4J_TYPES)
    }

    if (status != CUBLAS_STATUS_SUCCESS) throw cuda_exception::build("MmulHelper::mmulMxV cuda failed !", status);

    auto cudaResult = cudaStreamSynchronize(*stream);
    if (cudaResult != 0) throw cuda_exception::build("MmulHelper::mmulMxV cuda failed !", cudaResult);

    NDArray::registerSpecialUse({Y}, {pA, X});

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

    if(!X->isActualOnDeviceSide())  X->syncToDevice();
    if(!Y->isActualOnDeviceSide())  Y->syncToDevice();
    if(!Z->isActualOnDeviceSide())  Z->syncToDevice();

    cudaStream_t* stream = X->getContext()->getCudaStream();

    dim3 threadsPerBlock(512);
    dim3 blocksPerGrid(1);
    if (length > 512)
        threadsPerBlock.x = math::nd4j_ceil<double, int>(static_cast<double>(length) / 512);

    NDArray::prepareSpecialUse({Z}, {X, Y});

    //BUILD_TRIPLE_SELECTOR(xType, yType, zType, usualDot, (blocksPerGrid, threadsPerBlock, stream, length, alpha, X->getSpecialBuffer(), incx, Y->getSpecialBuffer(), incy, beta, Z->getSpecialBuffer()), LIBND4J_TYPES, FLOAT_TYPES, FLOAT_TYPES);
    BUILD_SINGLE_SELECTOR_THRICE(xType, usualDot, (blocksPerGrid, threadsPerBlock, stream, length, alpha, X->getSpecialBuffer(), incx, Y->getSpecialBuffer(), incy, beta, Z->getSpecialBuffer()), LIBND4J_TYPES)

    auto cudaResult = cudaStreamSynchronize(*stream);
    if (cudaResult != 0) throw cuda_exception::build("MmulHelper::dot cuda failed !", cudaResult);

    NDArray::registerSpecialUse({Z}, {X, Y});

    return Z;
}

//BUILD_TRIPLE_TEMPLATE(template void usualGemm, (const dim3 &blocksPerGrid, const dim3 &threadsPerBlock, cudaStream_t *stream, const bool transA, const bool transB, const int M, const int N, const int K, const double alpha, const void* vA, const int lda, const void* vB, const int ldb, const double beta, void* vC, const int ldc), LIBND4J_TYPES, FLOAT_TYPES, FLOAT_TYPES);
//BUILD_TRIPLE_TEMPLATE(template void usualGemv, (const dim3 &blocksPerGrid, const dim3 &threadsPerBlock, cudaStream_t *stream, const bool transA, const int M, const int N, const double alpha, const void* vA, const int lda, const void* vB, const int incx, const double beta, void* vC, const int incy), LIBND4J_TYPES, FLOAT_TYPES, FLOAT_TYPES);
//BUILD_TRIPLE_TEMPLATE(template void usualDot,  (const dim3 &blocksPerGrid, const dim3 &threadsPerBlock, cudaStream_t *stream, const Nd4jLong length, const double alpha, const void* vX, const Nd4jLong incx, const void* vY, const Nd4jLong incy, const double beta, void* vZ), LIBND4J_TYPES, FLOAT_TYPES, FLOAT_TYPES);

}
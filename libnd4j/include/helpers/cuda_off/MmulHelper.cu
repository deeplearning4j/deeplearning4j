/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 * Copyright (c) 2019 Konduit K.K.
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
#include <ops/specials_cuda.h>
#include <helpers/ShapeUtils.h>
#include <helpers/PointersManager.h>
#include <numeric>

namespace sd {

//////////////////////////////////////////////////////////////////////////////
// MXK x KxN = MxN              -> actual sequence of axes doesn't matter
template <typename T1, typename T2, typename T3>
static __global__ void usualCudaGemm(const void* vA, const Nd4jLong* aShapeInfo, const void* vB, const Nd4jLong* bShapeInfo, void* vC, const Nd4jLong* cShapeInfo,
                                     const int aMaxis, const int aKaxis, const int bKaxis, const int bNaxis, const int cMaxis, const int cNaxis,
                                     const double alpha, const double beta) {

    const T1* A = reinterpret_cast<const T1*>(vA);
    const T2* B = reinterpret_cast<const T2*>(vB);
          T3* C = reinterpret_cast<      T3*>(vC);

    __shared__ int K, *coords;
    __shared__ bool betaPresent;
    __shared__ Nd4jLong cLen, totalThreads;
    __shared__ T3 alphaZ, betaZ;

    if (threadIdx.x == 0) {

        extern __shared__ unsigned char shmem[];
        coords = reinterpret_cast<int*>(shmem);
        cLen = shape::length(cShapeInfo);

        K = shape::shapeOf(const_cast<Nd4jLong*>(aShapeInfo))[aKaxis];

        betaPresent = beta;

        totalThreads = gridDim.x * blockDim.x;

        alphaZ = alpha;
        betaZ  = beta;
    }
    __syncthreads();

    auto aCoords = coords + threadIdx.x * 6;    // 6 = (aRank + bRank + cRank)
    auto bCoords = aCoords + 2;
    auto cCoords = bCoords + 2;

    const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (Nd4jLong i = tid; i < cLen; i += totalThreads) {

        // evaluate C coordinates
        shape::index2coords(i, cShapeInfo, cCoords);

        // evaluate A coordinates
        aCoords[aMaxis] = cCoords[cMaxis];
        aCoords[aKaxis] = 0;

        // evaluate B coordinates
        bCoords[bKaxis] = 0;
        bCoords[bNaxis] = cCoords[cNaxis];

        auto aOffset = shape::getOffset(aShapeInfo, aCoords);
        auto bOffset = shape::getOffset(bShapeInfo, bCoords);

        T3 val = A[aOffset] * B[bOffset];                       // first iteration

        for (uint j = 1; j < K; ++j) {                          // rest iterations
            aOffset += shape::stride(aShapeInfo)[aKaxis];
            bOffset += shape::stride(bShapeInfo)[bKaxis];
            val = val + A[aOffset] * B[bOffset];
        }

        auto cOffset = shape::getOffset(cShapeInfo, cCoords);

        if(betaPresent)
            C[cOffset] = alphaZ * val + betaZ * C[cOffset];
        else
            C[cOffset] = alphaZ * val;
    }
}

////////////////////////////////////////////////////////////////////////
template <typename T1, typename T2, typename T3>
__host__ static void usualGemm(const int blocksPerGrid, const int threadsPerBlock, const int sharedMem, cudaStream_t *stream, const void* vA, const Nd4jLong* aShapeInfo, const void* vB, const Nd4jLong* bShapeInfo, void* vC, const Nd4jLong* cShapeInfo, const int aMaxis, const int aKaxis, const int bKaxis, const int bNaxis, const int cMaxis, const int cNaxis, const double alpha, const double beta) {

    usualCudaGemm<T1,T2,T3><<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(vA, aShapeInfo, vB, bShapeInfo, vC, cShapeInfo, aMaxis, aKaxis, bKaxis, bNaxis, cMaxis, cNaxis, alpha, beta);
}

////////////////////////////////////////////////////////////////////////
// MXN x N = M  -> actual sequence of {M,N} axes doesn't matter
template <typename T1, typename T2, typename T3>
static __global__ void usualCudaGemv(const void* vA, const Nd4jLong* aShapeInfo, const void* vX, const Nd4jLong* xShapeInfo, void* vY, const Nd4jLong* yShapeInfo,
                                     const int incx, const int incy, const int aMaxis, const double alpha, const double beta) {

    const T1* A = reinterpret_cast<const T1*>(vA);
    const T2* X = reinterpret_cast<const T2*>(vX);
          T3* Y = reinterpret_cast<      T3*>(vY);

    __shared__ int M, N;
    __shared__ bool betaPresent;
    __shared__ Nd4jLong cLen, totalThreads, aNstride, aMstride;
    __shared__ T3 alphaZ, betaZ;

    if (threadIdx.x == 0) {

        N = shape::length(xShapeInfo);
        M = shape::length(yShapeInfo);

        aMstride = shape::stride(aShapeInfo)[aMaxis];
        aNstride = shape::stride(aShapeInfo)[aMaxis == 0 ? 1 : 0];

        totalThreads = gridDim.x * blockDim.x;

        betaPresent = beta;

        alphaZ = alpha;
        betaZ  = beta;
    }
    __syncthreads();


    const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (Nd4jLong i = tid; i < M; i += totalThreads) {

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

        if(betaPresent)
            Y[yOffset] = alphaZ * val + betaZ * Y[yOffset];
        else
            Y[yOffset] = alphaZ * val;
    }
}

////////////////////////////////////////////////////////////////////////
template <typename T1, typename T2, typename T3>
__host__ static void usualGemv(const int blocksPerGrid, const int threadsPerBlock, cudaStream_t *stream, const void* vA, const Nd4jLong* aShapeInfo, const void* vX, const Nd4jLong* xShapeInfo, void* vY, const Nd4jLong* yShapeInfo, const int incx, const int incy, const int aMaxis, const double alpha, const double beta) {

    usualCudaGemv<T1,T2,T3><<<blocksPerGrid, threadsPerBlock, 512, *stream>>>(vA, aShapeInfo, vX, xShapeInfo, vY, yShapeInfo, incx, incy, aMaxis, alpha, beta);
}


//////////////////////////////////////////////////////////////////////////////
template <typename T1, typename T2, typename T3>
static __global__ void usualCudaDot(const Nd4jLong length, const double alpha, const void* vX, const Nd4jLong incx, const void* vY, const Nd4jLong incy, const double beta, void* vZ) {

    T1* X = reinterpret_cast<T1*>(const_cast<void*>(vX));
    T2* Y = reinterpret_cast<T2*>(const_cast<void*>(vY));
    T3* Z = reinterpret_cast<T3*>(vZ);

    extern __shared__ unsigned char shmem[];
    auto pairwiseMul = reinterpret_cast<T3*>(shmem);

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < length)
        pairwiseMul[tid] = X[tid * incx] * Y[tid * incy];

    __syncthreads();

    if(tid == 0) {
        T3 sum = 0;
        for(Nd4jLong i = 0; i < length; ++i)
            sum = sum + pairwiseMul[i];

        if(beta)
            *Z = (T3)alpha * sum + (T3)beta * *Z;
        else
            *Z = (T3)alpha * sum;
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

    const auto M = A->sizeAt(0);
    const auto K = A->sizeAt(1);
    const auto N = B->sizeAt(1);

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

    if (C->isEmpty())
        return C;

    const int major = Environment::getInstance()->capabilities()[AffinityManager::currentDeviceId()].first();

    const auto aType = A->dataType();
    const auto bType = B->dataType();
    const auto cType = C->dataType();

    const bool AB(aType == bType), AC(aType == cType), ABC(AB && AC);

    const bool typeDouble    = ABC && aType == DataType::DOUBLE;
    const bool typeFloat     = ABC && aType == DataType::FLOAT32;
    const bool typeHalf      = ABC && aType == DataType::HALF && major >= 6;
    const bool typeIntFloat  = AB  && aType == DataType::INT8 && cType == DataType::FLOAT32 && major >= 6;
    const bool typeHalfFloat = AB  && aType == DataType::HALF && cType == DataType::FLOAT32  && major >= 6;

    auto handle = reinterpret_cast<cublasHandle_t *>(A->getContext()->getCublasHandle());
    auto stream = A->getContext()->getCudaStream();

    auto status = cublasSetStream_v2(*handle, *stream);
    if (status != CUBLAS_STATUS_SUCCESS)
        throw cuda_exception::build("MmulHelper::mmulMxM cuda failed !", status);

    if(!typeDouble && !typeFloat && !typeHalf && !typeIntFloat && !typeHalfFloat) {

        const int threadsPerBlock = MAX_NUM_THREADS / 2;
        const int blocksPerGrid = (C->lengthOf() + threadsPerBlock - 1) / threadsPerBlock;
        const int sharedMem = threadsPerBlock * sizeof(int) * 6 + 128;                             // 6 = aRank + bRank + cRank

        NDArray::prepareSpecialUse({C}, {A, B});
        // BUILD_TRIPLE_SELECTOR(aType, bType, cType, usualGemm, (blocksPerGrid, threadsPerBlock, sharedMem, stream, A->getSpecialBuffer(), A->getSpecialShapeInfo(), B->getSpecialBuffer(), B->getSpecialShapeInfo(), C->getSpecialBuffer(), C->getSpecialShapeInfo(), 0, 1, 0, 1, 0, 1, alpha, beta), NUMERIC_TYPES, NUMERIC_TYPES, FLOAT_TYPES);
        BUILD_SINGLE_SELECTOR_THRICE(aType, usualGemm, (blocksPerGrid, threadsPerBlock, sharedMem, stream, A->getSpecialBuffer(), A->getSpecialShapeInfo(), B->getSpecialBuffer(), B->getSpecialShapeInfo(), C->getSpecialBuffer(), C->getSpecialShapeInfo(), 0, 1, 0, 1, 0, 1, alpha, beta), NUMERIC_TYPES)
        NDArray::registerSpecialUse({C}, {A, B});

        auto cudaResult = cudaStreamSynchronize(*stream);
        if (cudaResult != 0)
            throw cuda_exception::build("MmulHelper::mmulMxM cuda failed !", cudaResult);
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
        if(!cMcont) {
            pC = new NDArray(C->dup('f'));
            toDelete.push_back(pC);
            cMcont = true;
        }

        const bool transA = !aMcont;
        const bool transB = !bKcont;

        const int lda = (aMcont && aKcont) ? M : transA ? pA->strideAt(0) : pA->strideAt(1);
        const int ldb = (bKcont && bNcont) ? K : transB ? pB->strideAt(0) : pB->strideAt(1);
        const int ldc = (cMcont && cNcont) ? M : pC->strideAt(1);

        const cublasOperation_t transAblas = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
        const cublasOperation_t transBblas = transB ? CUBLAS_OP_T : CUBLAS_OP_N;

        NDArray::prepareSpecialUse({pC}, {pA, pB});

        // choose appropriate cuda gemm api depending on data types
        if(typeDouble) {
            status = cublasDgemm(*handle, transAblas, transBblas, M, N, K, &alpha, (double*)pA->getSpecialBuffer(), lda, (double*)pB->getSpecialBuffer(), ldb, &beta, (double*)pC->getSpecialBuffer(), ldc);
        }
        else if(typeFloat) {
            float alphaF(alpha), betaF(beta);
            status = cublasSgemm(*handle, transAblas, transBblas, M, N, K, &alphaF, (float*)pA->getSpecialBuffer(), lda, (float*)pB->getSpecialBuffer(), ldb, &betaF, (float*)pC->getSpecialBuffer(), ldc);
        }
        else if(typeHalf) {
            float16 alphaH(alpha), betaH(beta);
            status = cublasHgemm(*handle, transAblas, transBblas, M, N, K, &alphaH.data, (__half*)pA->getSpecialBuffer(), lda, (__half*)pB->getSpecialBuffer(), ldb, &betaH.data, (__half*)pC->getSpecialBuffer(), ldc);
        }
        else if(typeIntFloat) {
               float alphaF(alpha), betaF(beta);
               status = cublasSgemmEx(*handle, transAblas, transBblas, M, N, K, &alphaF, pA->getSpecialBuffer(), CUDA_R_8I, lda, pB->getSpecialBuffer(), CUDA_R_8I, ldb, &betaF, pC->getSpecialBuffer(), CUDA_R_32F, ldc);
        }
        else if(typeHalfFloat) {
            float alphaF(alpha), betaF(beta);
            status = cublasSgemmEx(*handle, transAblas, transBblas, M, N, K, &alphaF, pA->getSpecialBuffer(), CUDA_R_16F, lda, pB->getSpecialBuffer(), CUDA_R_16F, ldb, &betaF, pC->getSpecialBuffer(), CUDA_R_32F, ldc);
        }

        if (status != CUBLAS_STATUS_SUCCESS)
            throw cuda_exception::build("MmulHelper::mmulMxM cuda failed !", status);

        NDArray::registerSpecialUse({pC}, {pA, pB});

        auto cudaResult = cudaStreamSynchronize(*stream);
        if (cudaResult != 0)
            throw cuda_exception::build("MmulHelper::mmulMxM cuda failed !", cudaResult);

        if(C != pC)
            C->assign(pC);

        for(int i = toDelete.size() - 1; i >= 0; --i)
            delete toDelete[i];
    }

    return C;
}

////////////////////////////////////////////////////////////////////////////
// MXN x N = M
NDArray* MmulHelper::mmulMxV(const NDArray* A, const NDArray* X, sd::NDArray* Y, const double alpha, const double beta, const char outOrder) {

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

    if (Y->isEmpty())
        return Y;

    const int incx = X->strideAt(xLenDim);
    const int incy = Y->strideAt(yLenDim);

    const auto aType = A->dataType();
    const auto xType = X->dataType();
    const auto yType = Y->dataType();

    const bool AX(aType == xType), AY(aType == yType), AXY(AX && AY);

    const bool typeDouble = AXY && aType == DataType::DOUBLE;
    const bool typeFloat  = AXY && aType == DataType::FLOAT32;

    auto handle = reinterpret_cast<cublasHandle_t *>(A->getContext()->getCublasHandle());
    auto stream = A->getContext()->getCudaStream();

    auto status = cublasSetStream_v2(*handle, *stream);
    if (status != CUBLAS_STATUS_SUCCESS)
        throw cuda_exception::build("MmulHelper::mmulMxV cuda failed !", status);

    if(!typeDouble && !typeFloat) {

        const int threadsPerBlock = MAX_NUM_THREADS;
        const int blocksPerGrid = (M + threadsPerBlock - 1) / threadsPerBlock;

        NDArray::prepareSpecialUse({Y}, {A, X});
        // BUILD_TRIPLE_SELECTOR(aType, xType, yType, usualGemv, (blocksPerGrid, threadsPerBlock, stream, A->getSpecialBuffer(), A->getSpecialShapeInfo(), X->getSpecialBuffer(), X->getSpecialShapeInfo(), Y->getSpecialBuffer(), Y->getSpecialShapeInfo(), incx, incy, 0, alpha, beta), NUMERIC_TYPES, NUMERIC_TYPES, FLOAT_TYPES);
        BUILD_SINGLE_SELECTOR_THRICE(xType, usualGemv, (blocksPerGrid, threadsPerBlock, stream, A->getSpecialBuffer(), A->getSpecialShapeInfo(), X->getSpecialBuffer(), X->getSpecialShapeInfo(), Y->getSpecialBuffer(), Y->getSpecialShapeInfo(), incx, incy, 0, alpha, beta), NUMERIC_TYPES)
        NDArray::registerSpecialUse({Y}, {A, X});

        auto cudaResult = cudaStreamSynchronize(*stream);
        if (cudaResult != 0)
            throw cuda_exception::build("MmulHelper::mmulMxV cuda failed !", cudaResult);

    }
    else {

        NDArray *pA(const_cast<NDArray*>(A));

        bool aMcont = M == 1 || A->strideAt(0) == 1;
        bool aNcont = N == 1 || A->strideAt(1) == 1;

        if(!aMcont && !aNcont) {
            pA = new NDArray(A->dup('f'));
            aMcont = true;
        }

        const bool transA = !aMcont;

        const int lda = (aMcont && aNcont) ? M : transA ? pA->strideAt(0) : pA->strideAt(1);

        const cublasOperation_t transAblas = transA ? CUBLAS_OP_T : CUBLAS_OP_N;

        NDArray::prepareSpecialUse({Y}, {pA, X});

        // choose appropriate cuda gemm api depending on data types
        if(typeDouble) {
            status = cublasDgemv(*handle, transAblas, transA ? N : M, transA ? M : N, &alpha, (double*)pA->getSpecialBuffer(), lda, (double*)X->getSpecialBuffer(), incx, &beta, (double*)Y->getSpecialBuffer(), incy);
        }
        else if(typeFloat) {
            float alphaF(alpha), betaF(beta);
            status = cublasSgemv(*handle, transAblas, transA ? N : M, transA ? M : N, &alphaF, (float*)pA->getSpecialBuffer(), lda, (float*)X->getSpecialBuffer(), incx, &betaF, (float*)Y->getSpecialBuffer(), incy);
        }

        if (status != CUBLAS_STATUS_SUCCESS)
            throw cuda_exception::build("MmulHelper::mmulMxV cuda failed !", status);

        auto cudaResult = cudaStreamSynchronize(*stream);
        if (cudaResult != 0)
            throw cuda_exception::build("MmulHelper::mmulMxV cuda failed !", cudaResult);

        NDArray::registerSpecialUse({Y}, {pA, X});

        if(pA != A)
            delete pA;
    }

    return Y;
}

////////////////////////////////////////////////////////////////////////////
// (X * Y) = Z[0]
NDArray* MmulHelper::dot(const NDArray* X, const NDArray* Y, sd::NDArray* Z, const double alpha, const double beta) {

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

    const Nd4jLong incx = X->strideAt(xLenDim);
    const Nd4jLong incy = Y->strideAt(yLenDim);

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

    //BUILD_TRIPLE_SELECTOR(xType, yType, zType, usualDot, (blocksPerGrid, threadsPerBlock, stream, length, alpha, X->getSpecialBuffer(), incx, Y->getSpecialBuffer(), incy, beta, Z->getSpecialBuffer()), NUMERIC_TYPES, NUMERIC_TYPES, FLOAT_TYPES);
    BUILD_SINGLE_SELECTOR_THRICE(xType, usualDot, (blocksPerGrid, threadsPerBlock, stream, length, alpha, X->getSpecialBuffer(), incx, Y->getSpecialBuffer(), incy, beta, Z->getSpecialBuffer()), NUMERIC_TYPES)

    auto cudaResult = cudaStreamSynchronize(*stream);
    if (cudaResult != 0) throw cuda_exception::build("MmulHelper::dot cuda failed !", cudaResult);

    NDArray::registerSpecialUse({Z}, {X, Y});

    return Z;
}

//////////////////////////////////////////////////////////////////////////////
// [bS,M,K] x [bS,K,N] = [bS,M,N]
// [bS,M,K] x    [K,N] = [bS,M,N]
//    [M,K] x [bS,K,N] = [bS,M,N]
// bS could stand for several axes
template <typename T1, typename T2, typename T3>
static __global__ void batchedCudaGemm(const void* vA, const Nd4jLong* aShapeInfo, const void* vB, const Nd4jLong* bShapeInfo, void* vC, const Nd4jLong* cShapeInfo,
                                       const int* aBatchDims, const int* bBatchDims, const int* cBatchDims,
                                       const int aMaxis, const int aKaxis, const int bKaxis, const int bNaxis, const int cMaxis, const int cNaxis,
                                       const double alpha, const double beta) {

    const T1* A = reinterpret_cast<const T1*>(vA);
    const T2* B = reinterpret_cast<const T2*>(vB);
          T3* C = reinterpret_cast<      T3*>(vC);

    __shared__ bool betaPresent;
    __shared__ int aRank, bRank, cRank, K, *coords;
    __shared__ Nd4jLong cLen, totalThreads;
    __shared__ T3 alphaZ, betaZ;

    if (threadIdx.x == 0) {

        extern __shared__ unsigned char shmem[];
        coords = reinterpret_cast<int*>(shmem);
        cLen = shape::length(cShapeInfo);

        K = shape::shapeOf(const_cast<Nd4jLong*>(aShapeInfo))[aKaxis];

        totalThreads = gridDim.x * blockDim.x;
        aRank = shape::rank(aShapeInfo);
        bRank = shape::rank(bShapeInfo);
        cRank = shape::rank(cShapeInfo);

        betaPresent = beta;

        alphaZ = alpha;
        betaZ  = beta;
    }
    __syncthreads();

    auto aCoords = coords + threadIdx.x * (aRank + bRank + cRank);
    auto bCoords = aCoords + aRank;
    auto cCoords = bCoords + bRank;

    const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (Nd4jLong i = tid; i < cLen; i += totalThreads) {

        // evaluate C coordinates
        shape::index2coords(i, cShapeInfo, cCoords);

        // calculate index of current batch
        Nd4jLong batchInd;
        if(cBatchDims != nullptr)
            batchInd = shape::coords2index(cShapeInfo, cCoords, cRank - 2, cBatchDims);

        // evaluate A coordinates
        if(aBatchDims != nullptr)
            shape::index2coords(batchInd, aShapeInfo, aCoords, aRank - 2, aBatchDims);
        aCoords[aMaxis] = cCoords[cMaxis];
        aCoords[aKaxis] = 0;

        // evaluate B coordinates
        if(bBatchDims != nullptr)
            shape::index2coords(batchInd, bShapeInfo, bCoords, bRank - 2, bBatchDims);
        bCoords[bKaxis] = 0;
        bCoords[bNaxis] = cCoords[cNaxis];

        auto aOffset = shape::getOffset(aShapeInfo, aCoords);
        auto bOffset = shape::getOffset(bShapeInfo, bCoords);

        T3 val = A[aOffset] * B[bOffset];                       // first iteration

        for (uint j = 1; j < K; ++j) {                          // rest iterations
            aOffset += shape::stride(aShapeInfo)[aKaxis];
            bOffset += shape::stride(bShapeInfo)[bKaxis];
            val = val + A[aOffset] * B[bOffset];
        }

        auto cOffset = shape::getOffset(cShapeInfo, cCoords);

        if(betaPresent)
            C[cOffset] = alphaZ * val + betaZ * C[cOffset];
        else
            C[cOffset] = alphaZ * val;
    }
}

////////////////////////////////////////////////////////////////////////
template <typename T1, typename T2, typename T3>
__host__ static void batchedGemm(const int blocksPerGrid, const int threadsPerBlock, const int sharedMem, cudaStream_t *stream, const void* vA, const Nd4jLong* aShapeInfo, const void* vB, const Nd4jLong* bShapeInfo, void* vC, const Nd4jLong* cShapeInfo, const int* aBatchDims, const int* bBatchDims, const int* cBatchDims, const int aMaxis, const int aKaxis, const int bKaxis, const int bNaxis, const int cMaxis, const int cNaxis, const double alpha, const double beta) {

    batchedCudaGemm<T1,T2,T3><<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(vA, aShapeInfo, vB, bShapeInfo, vC, cShapeInfo, aBatchDims, bBatchDims, cBatchDims, aMaxis, aKaxis, bKaxis, bNaxis, cMaxis, cNaxis, alpha, beta);
}

///////////////////////////////////////////////////////////////////
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
    else
        C = new NDArray(outOrder, cExpectedShape, DataTypeUtils::pickPairwiseResultType(A->dataType(), B->dataType()), A->getContext());

    if (C->isEmpty())
        return C;

    const int cRank = C->rankOf();

    const int aMaxis(aRank-2), aKaxis(aRank-1), bKaxis(bRank-2), bNaxis(bRank-1), cMaxis(cRank-2), cNaxis(cRank-1);

    const int threadsPerBlock = MAX_NUM_THREADS / 8;
    const int blocksPerGrid = (C->lengthOf() + threadsPerBlock - 1) / threadsPerBlock;
    const int sharedMem = threadsPerBlock * sizeof(int) * (aRank + bRank + cRank) + 128;

    PointersManager manager(A->getContext(), "MmulHelper::mmulNxN");

    const int *aBatchDims(nullptr), *bBatchDims(nullptr), *cBatchDims(nullptr);

    if(aRank > 2)
        aBatchDims = reinterpret_cast<int*>(manager.replicatePointer(ShapeUtils::evalDimsToExclude(aRank, {aMaxis, aKaxis}).data(), (aRank - 2) * sizeof(int)));
    if(bRank > 2)
        bBatchDims = reinterpret_cast<int*>(manager.replicatePointer(ShapeUtils::evalDimsToExclude(bRank, {bKaxis, bNaxis}).data(), (bRank - 2) * sizeof(int)));
    if(cRank > 2)
        cBatchDims = reinterpret_cast<int*>(manager.replicatePointer(ShapeUtils::evalDimsToExclude(cRank, {cMaxis, cNaxis}).data(), (cRank - 2) * sizeof(int)));

    NDArray::prepareSpecialUse({C}, {A, B});
    // BUILD_TRIPLE_SELECTOR(A->dataType(), b->dataType(), C->dataType(), batchedGemm, (blocksPerGrid, threadsPerBlock, A->getContext()->getCudaStream(), A->getSpecialBuffer(), A->getSpecialShapeInfo(), B->getSpecialBuffer(), B->getSpecialShapeInfo(), C->getSpecialBuffer(), C->getSpecialShapeInfo(), aMaxis, aKaxis, bKaxis, bNaxis, cMaxis, cNaxis, alpha, beta), NUMERIC_TYPES, NUMERIC_TYPES, FLOAT_TYPES);
    BUILD_SINGLE_SELECTOR_THRICE(A->dataType(), batchedGemm, (blocksPerGrid, threadsPerBlock, sharedMem, A->getContext()->getCudaStream(), A->getSpecialBuffer(), A->getSpecialShapeInfo(), B->getSpecialBuffer(), B->getSpecialShapeInfo(), C->getSpecialBuffer(), C->getSpecialShapeInfo(), aBatchDims, bBatchDims, cBatchDims, aMaxis, aKaxis, bKaxis, bNaxis, cMaxis, cNaxis, alpha, beta), NUMERIC_TYPES)
    NDArray::registerSpecialUse({C}, {A, B});

    manager.synchronize();

    return C;
}


/*
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
        for (int i = 0; i < N; i++)
            val = val + A[row * strideArow + i * strideAcol] * X[i * incx];

    Y[row * incy] = alphaZ * val + betaZ * Y[row * incy];
}

////////////////////////////////////////////////////////////////////////
template <typename T1, typename T2, typename T3>
__host__ static void usualGemv(const dim3 &blocksPerGrid, const dim3 &threadsPerBlock, cudaStream_t *stream, const bool transA, const int M, const int N, const double alpha, const void* vA, const int lda, const void* vX, const int incx, const double beta, void* vY, const int incy) {

    usualCudaGemv<T1,T2,T3><<<blocksPerGrid, threadsPerBlock, 1024, *stream>>>(transA, M, N, alpha, vA, lda, vX, incx, beta, vY, incy);
}
*/
/*
//////////////////////////////////////////////////////////////////////////////
MXK x KxN = MxN
C array must be in f order
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

//////////////////////////////////////////////////////////////////////////////
template <typename T1, typename T2, typename T3>
__host__ static void usualGemm(const dim3 &blocksPerGrid, const dim3 &threadsPerBlock, cudaStream_t *stream, const bool transA, const bool transB, const int M, const int N, const int K, const double alpha, const void* vA, const int lda, const void* vB, const int ldb, const double beta, void* vC, const int ldc) {

    usualCudaGemm<T1,T2,T3><<<blocksPerGrid, threadsPerBlock, 1024, *stream>>>(transA, transB, M, N, K, alpha, vA, lda, vB, ldb, beta, vC, ldc);
}
*/
//////////////////////////////////////////////////////////////////////////
/*
NDArray* MmulHelper::mmulNxNold1(const NDArray* A, const NDArray* B, NDArray* C, const double alpha, const double beta, const char outOrder) {

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
*/

//////////////////////////////////////////////////////////////////////////
// [bS,M,K] x [bS,K,N] = [bS,M,N]
// [bS,M,K] x    [K,N] = [bS,M,N]
//    [M,K] x [bS,K,N] = [bS,M,N]
// bS could stand for several axes
/*
NDArray* MmulHelper::mmulNxNold2(const NDArray* A, const NDArray* B, NDArray* C, const double alpha, const double beta, const char outOrder) {

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
    else
        C = new NDArray(outOrder, cExpectedShape, B->dataType());

    const int cRank = C->rankOf();

    const auto M = A->sizeAt(-2);
    const auto K = A->sizeAt(-1);
    const auto N = B->sizeAt(-1);

    NDArray *pA(const_cast<NDArray*>(A)), *pB(const_cast<NDArray*>(B)), *pC(const_cast<NDArray*>(C));
    std::vector<NDArray*> toDelete;

    bool aMcont = M == 1 || A->strideAt(-2) == 1;
    bool aKcont = K == 1 || A->strideAt(-1) == 1;
    bool bKcont = K == 1 || B->strideAt(-2) == 1;
    bool bNcont = N == 1 || B->strideAt(-1) == 1;
    bool cMcont = M == 1 || C->strideAt(-2) == 1;
    bool cNcont = N == 1 || C->strideAt(-1) == 1;

    if(!aMcont && !aKcont) {
        pA = new NDArray(A->dup('c'));
        toDelete.push_back(pA);
        aKcont = true;
    }
    if(!bKcont && !bNcont) {
        pB = new NDArray(B->dup('c'));
        toDelete.push_back(pB);
        bNcont = true;
    }
    std::vector<int> permut(cRank);
    if(!cMcont) {
        std::iota(permut.begin(), permut.end(), 0);
        permut[cRank - 2] = cRank - 1;
        permut[cRank - 1] = cRank - 2;  // swap two last dimensions [..., M,N] -> [..., N,M]
        auto Cpermut = C->permute(permut);
        pC = new NDArray('c', Cpermut.getShapeAsVector(), Cpermut.dataType(), A->getContext());
        pC->assign(Cpermut);
        toDelete.push_back(pC);
        cMcont = true;
    }


    const auto aType = pA->dataType();
    const auto bType = pB->dataType();
    const auto cType = pC->dataType();

    const bool AB(aType == bType), AC(aType == cType), ABC(AB && AC);

    bool badTypes = false;
    cudaDataType_t cudaType, cudaAType, cudaBType, cudaCType;

    if(ABC && aType == DataType::HALF) {
        cudaType = cudaAType = cudaBType = cudaCType = CUDA_R_16F;
    }
    else if(ABC && aType == DataType::FLOAT32) {
        cudaType = cudaAType = cudaBType = cudaCType = CUDA_R_32F;
    }
    else if(ABC && aType == DataType::DOUBLE) {
        cudaType = cudaAType = cudaBType = cudaCType = CUDA_R_64F;
    }
    else if(AB && cType == DataType::FLOAT32 && aType == DataType::INT8) {
        cudaType = cudaCType = CUDA_R_32F;
        cudaAType = cudaBType = CUDA_R_8I;
    }
    else if(AB && cType == DataType::FLOAT32 && aType == DataType::HALF) {
        cudaType = cudaCType = CUDA_R_32F;
        cudaAType = cudaBType = CUDA_R_16F;
    }
    else
        badTypes = true;

    const int bS = pC->lengthOf() / (M*N);

    const std::vector<int> dimsToExclude = ShapeUtils::evalDimsToExclude(cRank, {-2, -1});

    NDArray::prepareSpecialUse({pC}, {pA, pB});

    if(!badTypes) {

        std::vector<Nd4jLong> subArrOffsets(bS);
        std::vector<Nd4jLong> subArrShapeInfo(shape::shapeInfoLength(2));                         // all sub-arrays have rank = 2

        std::vector<void*> aSubArrs(bS), bSubArrs(bS), cSubArrs(bS);

        if(aRank > 2)
            shape::calcSubArrsShapeInfoAndOffsets(pA->getShapeInfo(), bS, dimsToExclude.size(), dimsToExclude.data(), subArrShapeInfo.data(), subArrOffsets.data());
        for (int i = 0; i < bS; ++i)
            aSubArrs[i] = aRank == 2 ? pA->getSpecialBuffer() : pA->getSpecialBuffer() + subArrOffsets[i] * pA->sizeOfT();

        if(bRank > 2)
            shape::calcSubArrsShapeInfoAndOffsets(pB->getShapeInfo(), bS, dimsToExclude.size(), dimsToExclude.data(), subArrShapeInfo.data(), subArrOffsets.data());
        for (int i = 0; i < bS; ++i)
            bSubArrs[i] = bRank == 2 ? pB->getSpecialBuffer() : pB->getSpecialBuffer() + subArrOffsets[i] * pB->sizeOfT();

        shape::calcSubArrsShapeInfoAndOffsets(pC->getShapeInfo(), bS, dimsToExclude.size(), dimsToExclude.data(), subArrShapeInfo.data(), subArrOffsets.data());
        for (int i = 0; i < bS; ++i)
            cSubArrs[i] = pC->getSpecialBuffer() + subArrOffsets[i] * pC->sizeOfT();

        PointersManager manager(A->getContext(), "mmulNxN");

        const void** aSubArrsCuda = reinterpret_cast<const void **>(manager.replicatePointer(aSubArrs.data(),  aSubArrs.size() * sizeof(void*)));
        const void** bSubArrsCuda = reinterpret_cast<const void **>(manager.replicatePointer(bSubArrs.data(),  bSubArrs.size() * sizeof(void*)));
              void** cSubArrsCuda = reinterpret_cast<      void **>(manager.replicatePointer(cSubArrs.data(),  cSubArrs.size() * sizeof(void*)));

        const bool transA = !aMcont;
        const bool transB = !bKcont;

        const int lda = (aMcont && aKcont) ? M : transA  ? pA->strideAt(-2) : pA->strideAt(-1);
        const int ldb = (bKcont && bNcont) ? K : transB  ? pB->strideAt(-2) : pB->strideAt(-1);
        const int ldc = (cMcont && cNcont) ? M : C != pC ? pC->strideAt(-2) : pC->strideAt(-1);

        const cublasOperation_t transAblas = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
        const cublasOperation_t transBblas = transB ? CUBLAS_OP_T : CUBLAS_OP_N;

        union Coeff {__half _h; float _f; double _d; };
        Coeff uAlpha, uBeta;

        if(cudaType == CUDA_R_16F) {
            uAlpha._h = alpha;
            uBeta._h  = beta;
        }
        else if(cudaType == CUDA_R_32F) {
            uAlpha._f = alpha;
            uBeta._f  = beta;
        }
        else if(cudaType == CUDA_R_64F) {
            uAlpha._d = alpha;
            uBeta._d  = beta;
        }

        auto handle = reinterpret_cast<cublasHandle_t *>(A->getContext()->getCublasHandle());
        auto stream = A->getContext()->getCudaStream();

        auto status = cublasSetStream_v2(*handle, *stream);
        if (status != CUBLAS_STATUS_SUCCESS)
            throw cuda_exception::build("MmulHelper::mmulNxN cuda failed !", status);

        status = cublasGemmBatchedEx(*handle, transAblas, transBblas, M, N, K, &uAlpha, aSubArrsCuda, cudaAType, lda, bSubArrsCuda, cudaBType, ldb, &uBeta, cSubArrsCuda, cudaCType, ldc, bS, cudaType, CUBLAS_GEMM_DEFAULT);

        if (status != CUBLAS_STATUS_SUCCESS)
            throw cuda_exception::build("MmulHelper::mmulNxN cuda failed !", status);

        auto cudaResult = cudaStreamSynchronize(*stream);
        if (cudaResult != 0)
            throw cuda_exception::build("MmulHelper::mmulNxN cuda failed !", cudaResult);
    }
    else {

        std::vector<Nd4jLong> idxRanges(2 * pC->rankOf());

        for(Nd4jLong i = 0; i < bS; ++i) {

            ShapeUtils::evalIdxRangesForSubArr(i, pC->getShapeInfo(), dimsToExclude, idxRanges.data());
            NDArray cSubArr = (*pC)(idxRanges);

            if(aRank > bRank) {
                NDArray aSubArr = (*pA)(idxRanges);
                mmulMxM(&aSubArr, pB, &cSubArr, 1., 0., pC->ordering());
            }
            else if(bRank > aRank) {
                NDArray bSubArr = (*pB)(idxRanges);
                mmulMxM(pA, &bSubArr, &cSubArr, 1., 0, pC->ordering());
            }
            else {
                NDArray aSubArr = (*pA)(idxRanges);
                NDArray bSubArr = (*pB)(idxRanges);
                mmulMxM(&aSubArr, &bSubArr, &cSubArr, 1., 0., pC->ordering());
            }
        }
    }

    NDArray::registerSpecialUse({pC}, {pA, pB});

    if(C != pC)
        C->assign(pC->permute(permut));

    for(int i = toDelete.size() - 1; i >= 0; --i)
        delete toDelete[i];

    return C;
}
*/

//BUILD_TRIPLE_TEMPLATE(template void usualGemm, (const dim3 &blocksPerGrid, const dim3 &threadsPerBlock, cudaStream_t *stream, const bool transA, const bool transB, const int M, const int N, const int K, const double alpha, const void* vA, const int lda, const void* vB, const int ldb, const double beta, void* vC, const int ldc), NUMERIC_TYPES, NUMERIC_TYPES, FLOAT_TYPES);
//BUILD_TRIPLE_TEMPLATE(template void usualGemv, (const dim3 &blocksPerGrid, const dim3 &threadsPerBlock, cudaStream_t *stream, const bool transA, const int M, const int N, const double alpha, const void* vA, const int lda, const void* vB, const int incx, const double beta, void* vC, const int incy), NUMERIC_TYPES, NUMERIC_TYPES, FLOAT_TYPES);
//BUILD_TRIPLE_TEMPLATE(template void usualDot,  (const dim3 &blocksPerGrid, const dim3 &threadsPerBlock, cudaStream_t *stream, const Nd4jLong length, const double alpha, const void* vX, const Nd4jLong incx, const void* vY, const Nd4jLong incy, const double beta, void* vZ), NUMERIC_TYPES, NUMERIC_TYPES, FLOAT_TYPES);

}
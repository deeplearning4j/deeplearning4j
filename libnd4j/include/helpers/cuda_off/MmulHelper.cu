/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

//
// @author raver119@gmail.com
// @author Yurii Shyrma (iuriish@yahoo.com)
//
#include <cublas_v2.h>
#include <exceptions/cuda_exception.h>
#include <helpers/PointersManager.h>
#include <helpers/ShapeUtils.h>
#include <ops/specials_cuda.h>

#include <numeric>

#include "../MmulHelper.h"
#include "execution/cuda/LaunchDims.h"

namespace sd {

//////////////////////////////////////////////////////////////////////////////
// MXK x KxN = MxN              -> actual sequence of axes doesn't matter
template <typename T1, typename T2, typename T3>
static SD_KERNEL void usualCudaGemm(const void* vA, const LongType* aShapeInfo, const void* vB,
                                    const LongType* bShapeInfo, void* vC, const LongType* cShapeInfo,
                                    const int aMaxis, const int aKaxis, const int bKaxis, const int bNaxis,
                                    const int cMaxis, const int cNaxis, const double alpha, const double beta) {
  const T1* A = reinterpret_cast<const T1*>(vA);
  const T2* B = reinterpret_cast<const T2*>(vB);
  T3* C = reinterpret_cast<T3*>(vC);

  __shared__ LongType K, *coords;
  __shared__ bool betaPresent;
  __shared__ LongType cLen, totalThreads;
  __shared__ T3 alphaZ, betaZ;

  if (threadIdx.x == 0) {
    extern __shared__ unsigned char shmem[];
    coords = reinterpret_cast<LongType*>(shmem);
    cLen = shape::length(cShapeInfo);

    K = shape::shapeOf(const_cast<LongType*>(aShapeInfo))[aKaxis];

    betaPresent = beta;

    totalThreads = gridDim.x * blockDim.x;

    alphaZ = alpha;
    betaZ = beta;
  }
  __syncthreads();

  auto aCoords = coords + threadIdx.x * 6;  // 6 = (aRank + bRank + cRank)
  auto bCoords = aCoords + 2;
  auto cCoords = bCoords + 2;

  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

  for (LongType i = tid; i < cLen; i += totalThreads) {
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

    T3 val = A[aOffset] * B[bOffset];  // first iteration

    for (LongType j = 1; j < K; ++j) {  // rest iterations
      aOffset += shape::stride(aShapeInfo)[aKaxis];
      bOffset += shape::stride(bShapeInfo)[bKaxis];
      val = val + A[aOffset] * B[bOffset];
    }

    auto cOffset = shape::getOffset(cShapeInfo, cCoords);

    if (betaPresent)
      C[cOffset] = alphaZ * val + betaZ * C[cOffset];
    else
      C[cOffset] = alphaZ * val;
  }
}

////////////////////////////////////////////////////////////////////////
template <typename T1, typename T2, typename T3>
SD_HOST static void usualGemm(const int blocksPerGrid, const int threadsPerBlock, const int sharedMem,
                              cudaStream_t* stream, const void* vA, const LongType* aShapeInfo, const void* vB,
                              const LongType* bShapeInfo, void* vC, const LongType* cShapeInfo,
                              const int aMaxis, const int aKaxis, const int bKaxis, const int bNaxis, const int cMaxis,
                              const int cNaxis, const double alpha, const double beta) {
  usualCudaGemm<T1, T2, T3><<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(
      vA, aShapeInfo, vB, bShapeInfo, vC, cShapeInfo, aMaxis, aKaxis, bKaxis, bNaxis, cMaxis, cNaxis, alpha, beta);
  DebugHelper::checkGlobalErrorCode("MMUL cuda gemm failed(...) failed");

}

////////////////////////////////////////////////////////////////////////
// MXN x N = M  -> actual sequence of {M,N} axes doesn't matter
template <typename T1, typename T2, typename T3>
static SD_KERNEL void usualCudaGemv(const void* vA, const LongType* aShapeInfo, const void* vX,
                                    const LongType* xShapeInfo, void* vY, const LongType* yShapeInfo,
                                    const int incx, const int incy, const int aMaxis, const double alpha,
                                    const double beta) {
  const T1* A = reinterpret_cast<const T1*>(vA);
  const T2* X = reinterpret_cast<const T2*>(vX);
  T3* Y = reinterpret_cast<T3*>(vY);

  __shared__ int M, N;
  __shared__ bool betaPresent;
  __shared__ LongType cLen, totalThreads, aNstride, aMstride;
  __shared__ T3 alphaZ, betaZ;

  if (threadIdx.x == 0) {
    N = shape::length(xShapeInfo);
    M = shape::length(yShapeInfo);

    aMstride = shape::stride(aShapeInfo)[aMaxis];
    aNstride = shape::stride(aShapeInfo)[aMaxis == 0 ? 1 : 0];

    totalThreads = gridDim.x * blockDim.x;

    betaPresent = beta;

    alphaZ = alpha;
    betaZ = beta;
  }
  __syncthreads();

  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

  for (LongType i = tid; i < M; i += totalThreads) {
    // evaluate offsets
    auto aOffset = i * aMstride;
    auto xOffset = 0;

    T3 val = A[aOffset] * X[xOffset];  // first iteration

    for (LongType j = 1; j < N; ++j) {  // rest iterations
      aOffset += aNstride;
      xOffset += incx;
      val = val + A[aOffset] * X[xOffset];
    }

    auto yOffset = i * incy;

    if (betaPresent)
      Y[yOffset] = alphaZ * val + betaZ * Y[yOffset];
    else
      Y[yOffset] = alphaZ * val;
  }
}

////////////////////////////////////////////////////////////////////////
template <typename T1, typename T2, typename T3>
SD_HOST static void usualGemv(const int blocksPerGrid, const int threadsPerBlock, cudaStream_t* stream, const void* vA,
                              const LongType* aShapeInfo, const void* vX, const LongType* xShapeInfo, void* vY,
                              const LongType* yShapeInfo, const int incx, const int incy, const int aMaxis,
                              const double alpha, const double beta) {
  usualCudaGemv<T1, T2, T3><<<blocksPerGrid, threadsPerBlock, 512, *stream>>>(
      vA, aShapeInfo, vX, xShapeInfo, vY, yShapeInfo, incx, incy, aMaxis, alpha, beta);
  DebugHelper::checkGlobalErrorCode("MMUL cuda gemv case failed(...) failed");

}

//////////////////////////////////////////////////////////////////////////////
template <typename T1, typename T2, typename T3>
static SD_KERNEL void usualCudaDot(const LongType length, const double alpha, const void* vX,
                                   const LongType incx, const void* vY, const LongType incy, const double beta,
                                   void* vZ) {
  T1* X = reinterpret_cast<T1*>(const_cast<void*>(vX));
  T2* Y = reinterpret_cast<T2*>(const_cast<void*>(vY));
  T3* Z = reinterpret_cast<T3*>(vZ);

  extern __shared__ unsigned char shmem[];
  auto pairwiseMul = reinterpret_cast<T3*>(shmem);

  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < length) pairwiseMul[tid] = X[tid * incx] * Y[tid * incy];

  __syncthreads();

  if (tid == 0) {
    T3 sum = 0;
    for (LongType i = 0; i < length; ++i) sum = sum + pairwiseMul[i];

    if (beta)
      *Z = (T3)alpha * sum + (T3)beta * *Z;
    else
      *Z = (T3)alpha * sum;
  }
}

////////////////////////////////////////////////////////////////////////
template <typename T1, typename T2, typename T3>
SD_HOST static void usualDot(const dim3& launchDims, cudaStream_t* stream,
                             const LongType length, const double alpha, const void* vX, const LongType incx,
                             const void* vY, const LongType incy, const double beta, void* vZ) {
  usualCudaDot<T1, T2, T3><<<launchDims.x, launchDims.y,launchDims.z, *stream>>>(
      length, alpha, vX, incx, vY, incy, beta, vZ);
  DebugHelper::checkGlobalErrorCode("concat dot failed(...) failed");

}

//////////////////////////////////////////////////////////////////////////////
// MXK x KxN = MxN
NDArray* MmulHelper::mmulMxM(const NDArray* A, const NDArray* B, NDArray* C, double alpha, double beta,
                             const char outOrder) {
  if (A->rankOf() != 2) THROW_EXCEPTION("MmulHelper::mmulMxM cuda: rank of A array is not equal 2 !");
  if (B->rankOf() != 2) THROW_EXCEPTION("MmulHelper::mmulMxM cuda: rank of B array is not equal 2 !");

  const auto M = A->sizeAt(0);
  const auto K = A->sizeAt(1);
  const auto N = B->sizeAt(1);

  if (C != nullptr && C->rankOf() != 2)
    THROW_EXCEPTION("MmulHelper::mmulMxM cuda: rank of C array is not equal 2 !");
  if (B->sizeAt(0) != K) THROW_EXCEPTION("MmulHelper::mmulMxM cuda: B array has wrong number of rows !");
  if (C != nullptr && C->sizeAt(0) != M)
    THROW_EXCEPTION("MmulHelper::mmulMxM cuda: C array has wrong number of rows !");
  if (C != nullptr && C->sizeAt(1) != N)
    THROW_EXCEPTION("MmulHelper::mmulMxM cuda: C array has wrong number of columns !");

  std::vector<LongType> cShape = {M, N};
  if (C == nullptr)
    C = new NDArray(outOrder, cShape, DataTypeUtils::pickPairwiseResultType(A->dataType(), B->dataType()),
                    A->getContext());

  if (C->isEmpty()) return C;

  const int major = Environment::getInstance().capabilities()[AffinityManager::currentDeviceId()].first();

  const auto aType = A->dataType();
  const auto bType = B->dataType();
  const auto cType = C->dataType();

  const bool AB(aType == bType), AC(aType == cType), ABC(AB && AC);

  const bool typeDouble = ABC && aType == DOUBLE;
  const bool typeFloat = ABC && aType == FLOAT32;
  const bool typeHalf = ABC && aType == HALF && major >= 6;
  const bool typeIntFloat = AB && aType == INT8 && cType == FLOAT32 && major >= 6;
  const bool typeHalfFloat = AB && aType == HALF && cType == FLOAT32 && major >= 6;

  std::lock_guard<std::mutex> lock(*LaunchContext::deviceMutex());

  auto handle = reinterpret_cast<cublasHandle_t*>(A->getContext()->getCublasHandle());
  auto stream = A->getContext()->getCudaStream();

  auto status = cublasSetStream_v2(*handle, *stream);
  if (status != CUBLAS_STATUS_SUCCESS) throw cuda_exception::build("MmulHelper::mmulMxM cuda failed !", status);

  if (!typeDouble && !typeFloat && !typeHalf && !typeIntFloat && !typeHalfFloat) {
    dim3 dims = getMMulDims(C->lengthOf(),DataTypeUtils::sizeOf(cType));
    NDArray::prepareSpecialUse({C}, {A, B});
    BUILD_SINGLE_SELECTOR_THRICE(aType, usualGemm,
                                 (dims.y, dims.x, dims.z, stream, A->specialBuffer(),
                                     A->specialShapeInfo(), B->specialBuffer(), B->specialShapeInfo(), C->specialBuffer(),
                                     C->specialShapeInfo(), 0, 1, 0, 1, 0, 1, alpha, beta),
                                 SD_NUMERIC_TYPES)
    NDArray::registerSpecialUse({C}, {A, B});

    auto cudaResult = cudaStreamSynchronize(*stream);
    if (cudaResult != 0) throw cuda_exception::build("MmulHelper::mmulMxM cuda failed !", cudaResult);
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
      pA = new NDArray(A->dup('f'));
      toDelete.push_back(pA);
      aMcont = true;
    }
    if (!bKcont && !bNcont) {
      pB = new NDArray(B->dup('f'));
      toDelete.push_back(pB);
      bKcont = true;
    }
    if (!cMcont) {
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
    if (typeDouble) {
      status = cublasDgemm(*handle, transAblas, transBblas, M, N, K, &alpha, (double*)pA->specialBuffer(), lda,
                           (double*)pB->specialBuffer(), ldb, &beta, (double*)pC->specialBuffer(), ldc);
    } else if (typeFloat) {
      float alphaF(alpha), betaF(beta);
      status = cublasSgemm(*handle, transAblas, transBblas, M, N, K, &alphaF, (float*)pA->specialBuffer(), lda,
                           (float*)pB->specialBuffer(), ldb, &betaF, (float*)pC->specialBuffer(), ldc);
    } else if (typeHalf) {
      float16 alphaH(alpha), betaH(beta);
      status = cublasHgemm(*handle, transAblas, transBblas, M, N, K, &alphaH.data, (__half*)pA->specialBuffer(), lda,
                           (__half*)pB->specialBuffer(), ldb, &betaH.data, (__half*)pC->specialBuffer(), ldc);
    } else if (typeIntFloat) {
      float alphaF(alpha), betaF(beta);
      status = cublasSgemmEx(*handle, transAblas, transBblas, M, N, K, &alphaF, pA->specialBuffer(), CUDA_R_8I, lda,
                             pB->specialBuffer(), CUDA_R_8I, ldb, &betaF, pC->specialBuffer(), CUDA_R_32F, ldc);
    } else if (typeHalfFloat) {
      float alphaF(alpha), betaF(beta);
      status = cublasSgemmEx(*handle, transAblas, transBblas, M, N, K, &alphaF, pA->specialBuffer(), CUDA_R_16F, lda,
                             pB->specialBuffer(), CUDA_R_16F, ldb, &betaF, pC->specialBuffer(), CUDA_R_32F, ldc);
    }

    if (status != CUBLAS_STATUS_SUCCESS) throw cuda_exception::build("MmulHelper::mmulMxM cuda failed !", status);

    NDArray::registerSpecialUse({pC}, {pA, pB});

    auto cudaResult = cudaStreamSynchronize(*stream);
    if (cudaResult != 0) throw cuda_exception::build("MmulHelper::mmulMxM cuda failed !", cudaResult);

    if (C != pC) C->assign(pC);

    for (int i = toDelete.size() - 1; i >= 0; --i) delete toDelete[i];
  }

  return C;
}

////////////////////////////////////////////////////////////////////////////
// MXN x N = M
NDArray* MmulHelper::mmulMxV(const NDArray* A, const NDArray* X, NDArray* Y, const double alpha, const double beta,
                             const char outOrder) {
  LongType xLenDim, yLenDim(0);

  if (A->rankOf() != 2) THROW_EXCEPTION("MmulHelper::mmulMxV cuda: rank of A array is not equal 2 !");
  if (!shape::isCommonVector(X->shapeInfo(), xLenDim))
    THROW_EXCEPTION("MmulHelper::mmulMxV cuda: X array must be vector !");

  const auto M = A->sizeAt(0);
  const auto N = A->sizeAt(1);

  if (Y != nullptr && !shape::isCommonVector(Y->shapeInfo(), yLenDim))
    THROW_EXCEPTION("MmulHelper::mmulMxV cuda: Y array must be vector !");
  if (X->lengthOf() != N) THROW_EXCEPTION("MmulHelper::mmulMxV cuda: X vector has wrong length !");
  if (Y != nullptr && Y->lengthOf() != M)
    THROW_EXCEPTION("MmulHelper::mmulMxV cuda: Y array has wrong length !");

  std::vector<LongType> yShape = {M};
  if (Y == nullptr)
    Y = new NDArray(outOrder, yShape, DataTypeUtils::pickPairwiseResultType(A->dataType(), X->dataType()),
                    A->getContext());

  if (Y->isEmpty()) return Y;

  const int incx = X->strideAt(xLenDim);
  const int incy = Y->strideAt(yLenDim);

  const auto aType = A->dataType();
  const auto xType = X->dataType();
  const auto yType = Y->dataType();

  const bool AX(aType == xType), AY(aType == yType), AXY(AX && AY);

  const bool typeDouble = AXY && aType == DOUBLE;
  const bool typeFloat = AXY && aType == FLOAT32;

  std::lock_guard<std::mutex> lock(*LaunchContext::deviceMutex());

  auto handle = reinterpret_cast<cublasHandle_t*>(A->getContext()->getCublasHandle());
  auto stream = A->getContext()->getCudaStream();

  auto status = cublasSetStream_v2(*handle, *stream);
  if (status != CUBLAS_STATUS_SUCCESS) throw cuda_exception::build("MmulHelper::mmulMxV cuda failed !", status);

  if (!typeDouble && !typeFloat) {
    dim3 dims = getGemVDims(M);
    NDArray::prepareSpecialUse({Y}, {A, X});

    const int blocksPerGrid = dims.x;
    const int threadsPerBlock = dims.y;
    BUILD_SINGLE_SELECTOR_THRICE(
        xType, usualGemv,
        (blocksPerGrid,threadsPerBlock,stream, A->specialBuffer(), A->specialShapeInfo(), X->specialBuffer(),
            X->specialShapeInfo(), Y->specialBuffer(), Y->specialShapeInfo(), incx, incy, 0, alpha, beta),
        SD_NUMERIC_TYPES)
    NDArray::registerSpecialUse({Y}, {A, X});

    auto cudaResult = cudaStreamSynchronize(*stream);
    if (cudaResult != 0) throw cuda_exception::build("MmulHelper::mmulMxV cuda failed !", cudaResult);

  } else {
    NDArray* pA(const_cast<NDArray*>(A));

    bool aMcont = M == 1 || A->strideAt(0) == 1;
    bool aNcont = N == 1 || A->strideAt(1) == 1;

    if (!aMcont && !aNcont) {
      pA = new NDArray(A->dup('f'));
      aMcont = true;
    }

    const bool transA = !aMcont;

    const int lda = (aMcont && aNcont) ? M : transA ? pA->strideAt(0) : pA->strideAt(1);

    const cublasOperation_t transAblas = transA ? CUBLAS_OP_T : CUBLAS_OP_N;

    NDArray::prepareSpecialUse({Y}, {pA, X});

    // choose appropriate cuda gemm api depending on data types
    if (typeDouble) {
      status = cublasDgemv(*handle, transAblas, transA ? N : M, transA ? M : N, &alpha, (double*)pA->specialBuffer(),
                           lda, (double*)X->specialBuffer(), incx, &beta, (double*)Y->specialBuffer(), incy);
    } else if (typeFloat) {
      float alphaF(alpha), betaF(beta);
      status = cublasSgemv(*handle, transAblas, transA ? N : M, transA ? M : N, &alphaF, (float*)pA->specialBuffer(),
                           lda, (float*)X->specialBuffer(), incx, &betaF, (float*)Y->specialBuffer(), incy);
    }

    if (status != CUBLAS_STATUS_SUCCESS) throw cuda_exception::build("MmulHelper::mmulMxV cuda failed !", status);

    auto cudaResult = cudaStreamSynchronize(*stream);
    if (cudaResult != 0) throw cuda_exception::build("MmulHelper::mmulMxV cuda failed !", cudaResult);

    NDArray::registerSpecialUse({Y}, {pA, X});

    if (pA != A) delete pA;
  }

  return Y;
}

////////////////////////////////////////////////////////////////////////////
// (X * Y) = Z[0]
NDArray* MmulHelper::dot(const NDArray* X, const NDArray* Y, NDArray* Z, const double alpha, const double beta) {
  LongType xLenDim(0), yLenDim(0);

  if (!shape::isCommonVector(X->shapeInfo(), xLenDim))
    THROW_EXCEPTION("MmulHelper::dot cuda: X array must be vector !");
  if (!shape::isCommonVector(Y->shapeInfo(), yLenDim))
    THROW_EXCEPTION("MmulHelper::dot cuda: Y array must be vector !");
  if (Z != nullptr && Z->lengthOf() > 1) {
    THROW_EXCEPTION("MmulHelper::dot: Z array must be scalar !");
  }

  const auto length = X->lengthOf();

  if (Y->lengthOf() != length)
    THROW_EXCEPTION("MmulHelper::dot cuda: lengths of input vectors are different !");

  if (Z == nullptr)
    Z = new NDArray(DataTypeUtils::pickPairwiseResultType(X->dataType(), Y->dataType()), X->getContext());

  const LongType incx = X->strideAt(xLenDim);
  const LongType incy = Y->strideAt(yLenDim);

  const auto xType = X->dataType();
  const auto yType = Y->dataType();
  const auto zType = Z->dataType();

  if (!X->isActualOnDeviceSide()) X->syncToDevice();
  if (!Y->isActualOnDeviceSide()) Y->syncToDevice();
  if (!Z->isActualOnDeviceSide()) Z->syncToDevice();

  cudaStream_t* stream = X->getContext()->getCudaStream();

  dim3 dims = getMMulDims(length,DataTypeUtils::sizeOf(zType));

  NDArray::prepareSpecialUse({Z}, {X, Y});


  BUILD_SINGLE_SELECTOR_THRICE(xType, usualDot,
                               (dims, stream, length, alpha, X->specialBuffer(), incx,
                                   Y->specialBuffer(), incy, beta, Z->specialBuffer()),
                               SD_NUMERIC_TYPES)

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
static SD_KERNEL void batchedCudaGemm(const void* vA, const LongType* aShapeInfo, const void* vB,
                                      const LongType* bShapeInfo, void* vC, const LongType* cShapeInfo,
                                      const LongType* aBatchDims, const LongType* bBatchDims,
                                      const LongType* cBatchDims, const LongType aMaxis, const LongType aKaxis,
                                      const LongType bKaxis, const LongType bNaxis, const LongType cMaxis,
                                      const LongType cNaxis, const double alpha, const double beta) {
  const T1* A = reinterpret_cast<const T1*>(vA);
  const T2* B = reinterpret_cast<const T2*>(vB);
  T3* C = reinterpret_cast<T3*>(vC);

  __shared__ bool betaPresent;
  __shared__ LongType aRank, bRank, cRank, K, *coords;
  __shared__ LongType cLen, totalThreads;
  __shared__ T3 alphaZ, betaZ;

  if (threadIdx.x == 0) {
    extern __shared__ unsigned char shmem[];
    coords = reinterpret_cast<LongType*>(shmem);
    cLen = shape::length(cShapeInfo);

    K = shape::shapeOf(const_cast<LongType*>(aShapeInfo))[aKaxis];

    totalThreads = gridDim.x * blockDim.x;
    aRank = shape::rank(aShapeInfo);
    bRank = shape::rank(bShapeInfo);
    cRank = shape::rank(cShapeInfo);

    betaPresent = beta;

    alphaZ = alpha;
    betaZ = beta;
  }
  __syncthreads();

  auto aCoords = coords + threadIdx.x * (aRank + bRank + cRank);
  auto bCoords = aCoords + aRank;
  auto cCoords = bCoords + bRank;

  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

  for (LongType i = tid; i < cLen; i += totalThreads) {
    // evaluate C coordinates
    shape::index2coords(i, cShapeInfo, cCoords);

    // calculate index of current batch
    LongType batchInd;
    if (cBatchDims != nullptr) batchInd = shape::coords2index(cShapeInfo, cBatchDims, cRank - 2, cCoords);

    // evaluate A coordinates
    if (aBatchDims != nullptr) shape::index2coords(batchInd, aShapeInfo, aBatchDims, aRank - 2, aCoords);
    aCoords[aMaxis] = cCoords[cMaxis];
    aCoords[aKaxis] = 0;

    // evaluate B coordinates
    if (bBatchDims != nullptr) shape::index2coords(batchInd, bShapeInfo, bBatchDims, bRank - 2, bCoords);
    bCoords[bKaxis] = 0;
    bCoords[bNaxis] = cCoords[cNaxis];

    auto aOffset = shape::getOffset(aShapeInfo, aCoords);
    auto bOffset = shape::getOffset(bShapeInfo, bCoords);

    T3 val = A[aOffset] * B[bOffset];  // first iteration

    for (LongType j = 1; j < K; ++j) {  // rest iterations
      aOffset += shape::stride(aShapeInfo)[aKaxis];
      bOffset += shape::stride(bShapeInfo)[bKaxis];
      val = val + A[aOffset] * B[bOffset];
    }

    auto cOffset = shape::getOffset(cShapeInfo, cCoords);

    if (betaPresent)
      C[cOffset] = alphaZ * val + betaZ * C[cOffset];
    else
      C[cOffset] = alphaZ * val;
  }
}

////////////////////////////////////////////////////////////////////////
template <typename T1, typename T2, typename T3>
SD_HOST static void batchedGemm(const int blocksPerGrid, const int threadsPerBlock, const int sharedMem,
                                cudaStream_t* stream, const void* vA, const LongType* aShapeInfo, const void* vB,
                                const LongType* bShapeInfo, void* vC, const LongType* cShapeInfo,
                                const LongType* aBatchDims, const LongType* bBatchDims, const LongType* cBatchDims,
                                const LongType aMaxis, const LongType aKaxis, const LongType bKaxis,
                                const LongType bNaxis, const LongType cMaxis, const LongType cNaxis, const double alpha, const double beta) {
  batchedCudaGemm<T1, T2, T3><<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(
      vA, aShapeInfo, vB, bShapeInfo, vC, cShapeInfo, aBatchDims, bBatchDims, cBatchDims, aMaxis, aKaxis, bKaxis,
      bNaxis, cMaxis, cNaxis, alpha, beta);
  DebugHelper::checkGlobalErrorCode("batch gemm failed(...) failed");

}

///////////////////////////////////////////////////////////////////
NDArray* MmulHelper::mmulNxN(NDArray* A, NDArray* B, NDArray* C, const double alpha, const double beta,
                             const char outOrder) {
  const LongType aRank = A->rankOf();
  const LongType bRank = B->rankOf();

  // input ranks validation
  if (aRank > bRank && bRank != 2) {
    THROW_EXCEPTION("MmulHelper::mmulNxN: rank of B array should be equal 2 !");
  }
  else if (bRank > aRank && aRank != 2) {
    THROW_EXCEPTION("MmulHelper::mmulNxN: rank of A array should be equal 2 !");
  }
  else if (aRank == bRank) {
    for (int i = 0; i < aRank - 2; ++i)
      if (A->sizeAt(i) != B->sizeAt(i))
        THROW_EXCEPTION(
            "MmulHelper::mmulNxN: shapes of A and B arrays are not suitable for matrix multiplication !");
  }

  if (A->sizeAt(-1) != B->sizeAt(-2)) {
    THROW_EXCEPTION("MmulHelper::mmulNxN: shapes of A and B arrays are not suitable for matrix multiplication !");
  }
  // validation of C array
  std::vector<LongType> cExpectedShape = aRank > bRank ? A->getShapeAsVector() : B->getShapeAsVector();
  cExpectedShape[cExpectedShape.size() - 2] = A->sizeAt(-2);
  cExpectedShape[cExpectedShape.size() - 1] = B->sizeAt(-1);

  if (C != nullptr) {
    if (!C->isSameShape(cExpectedShape))
      THROW_EXCEPTION("MmulHelper::mmulNxN: shape of C array is not suitable for AxB matrix multiplication !");
  } else
    C = new NDArray(outOrder, cExpectedShape, DataTypeUtils::pickPairwiseResultType(A->dataType(), B->dataType()),
                    A->getContext());

  if (C->isEmpty()) return C;

  const LongType cRank = C->rankOf();

  const LongType aMaxis(aRank - 2), aKaxis(aRank - 1), bKaxis(bRank - 2), bNaxis(bRank - 1), cMaxis(cRank - 2),
      cNaxis(cRank - 1);

  const int threadsPerBlock = SD_MAX_NUM_THREADS / 8;
  const int blocksPerGrid = (C->lengthOf() + threadsPerBlock - 1) / threadsPerBlock;
  const int sharedMem = threadsPerBlock * sizeof(LongType) * (aRank + bRank + cRank) + 128;

  PointersManager manager(A->getContext(), "MmulHelper::mmulNxN");

  const LongType *aBatchDims(nullptr), *bBatchDims(nullptr), *cBatchDims(nullptr);

  std::vector<LongType> aDimsVec = {aMaxis,aKaxis};
  std::vector<LongType> *aDims = ShapeUtils::evalDimsToExclude(aRank, 2,aDimsVec.data());

  std::vector<LongType> bDimsVec = {bKaxis, bNaxis};
  std::vector<LongType> *bDims =  ShapeUtils::evalDimsToExclude(bRank,2, bDimsVec.data());


  std::vector<LongType> cDimsVec = {cMaxis,2, cNaxis};
  std::vector<LongType> *cDims = ShapeUtils::evalDimsToExclude(cRank, cDimsVec.size(),cDimsVec.data());
  if (aRank > 2)
    aBatchDims = reinterpret_cast<LongType*>(manager.replicatePointer(
        aDims->data(), (aRank - 2) * sizeof(LongType)));
  if (bRank > 2)
    bBatchDims = reinterpret_cast<LongType*>(manager.replicatePointer(
        bDims->data(), (bRank - 2) * sizeof(LongType)));
  if (cRank > 2)
    cBatchDims = reinterpret_cast<LongType*>(manager.replicatePointer(
        cDims->data(), (cRank - 2) * sizeof(LongType)));

  NDArray::prepareSpecialUse({C}, {A, B});
  BUILD_SINGLE_SELECTOR_THRICE(
      A->dataType(), batchedGemm,
      (blocksPerGrid, threadsPerBlock, sharedMem, A->getContext()->getCudaStream(), A->specialBuffer(),
          A->specialShapeInfo(), B->specialBuffer(), B->specialShapeInfo(), C->specialBuffer(), C->specialShapeInfo(),
          aBatchDims, bBatchDims, cBatchDims, aMaxis, aKaxis, bKaxis, bNaxis, cMaxis, cNaxis, alpha, beta),
      SD_NUMERIC_TYPES)
  NDArray::registerSpecialUse({C}, {A, B});

  manager.synchronize();

  delete aDims;
  delete bDims;
  delete cDims;

  return C;
}


}  // namespace sd

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
#include <cuda_fp16.h>
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
 // Cache shape information in shared memory
 __shared__ LongType K;
 __shared__ LongType cLen;
 __shared__ LongType totalThreads;
 __shared__ bool betaPresent;
 __shared__ T3 alphaZ;
 __shared__ T3 betaZ;
 __shared__ const LongType* aShape;
 __shared__ const LongType* bShape;
 __shared__ const LongType* cShape;
 __shared__ const LongType* aStride;
 __shared__ const LongType* bStride;
 __shared__ const LongType* cStride;
 __shared__ LongType aRank;
 __shared__ LongType bRank;
 __shared__ LongType cRank;
 __shared__ LongType* coords;

 if (threadIdx.x == 0) {
   extern __shared__ unsigned char shmem[];
   coords = reinterpret_cast<LongType*>(shmem);

   // Cache all shape information at start
   aRank = shape::rank(aShapeInfo);
   bRank = shape::rank(bShapeInfo);
   cRank = shape::rank(cShapeInfo);

   aShape = shape::shapeOf(aShapeInfo);
   bShape = shape::shapeOf(bShapeInfo);
   cShape = shape::shapeOf(cShapeInfo);

   aStride = shape::stride(aShapeInfo);
   bStride = shape::stride(bShapeInfo);
   cStride = shape::stride(cShapeInfo);

   cLen = shape::length(cShapeInfo);
   K = aShape[aKaxis];
   betaPresent = beta != 0;
   totalThreads = gridDim.x * blockDim.x;
   alphaZ = alpha;
   betaZ = beta;
 }
 __syncthreads();

 const T1* A = reinterpret_cast<const T1*>(vA);
 const T2* B = reinterpret_cast<const T2*>(vB);
 T3* C = reinterpret_cast<T3*>(vC);

 auto aCoords = coords + threadIdx.x * 6;  // 6 = (aRank + bRank + cRank)
 auto bCoords = aCoords + 2;
 auto cCoords = bCoords + 2;

 const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

 for (LongType i = tid; i < cLen; i += totalThreads) {
   // evaluate C coordinates
   INDEX2COORDS(i, cRank, cShape, cCoords);

   // evaluate A coordinates
   aCoords[aMaxis] = cCoords[cMaxis];
   aCoords[aKaxis] = 0;

   // evaluate B coordinates
   bCoords[bKaxis] = 0;
   bCoords[bNaxis] = cCoords[cNaxis];

   LongType aOffset, bOffset, cOffset;
   COORDS2INDEX(aRank, aStride, aCoords, aOffset);
   COORDS2INDEX(bRank, bStride, bCoords, bOffset);

   T3 val = A[aOffset] * B[bOffset];  // first iteration

   for (LongType j = 1; j < K; ++j) {  // rest iterations
     aOffset += aStride[aKaxis];
     bOffset += bStride[bKaxis];
     val = val + A[aOffset] * B[bOffset];
   }

   COORDS2INDEX(cRank, cStride, cCoords, cOffset);

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
 DebugHelper::checkGlobalErrorCode("MMUL cuda gemv case failed(...) failed");
}

////////////////////////////////////////////////////////////////////////
// MXN x N = M  -> actual sequence of {M,N} axes doesn't matter
template <typename T1, typename T2, typename T3>
static SD_KERNEL void usualCudaGemv(const void* vA, const LongType* aShapeInfo, const void* vX,
                                   const LongType* xShapeInfo, void* vY, const LongType* yShapeInfo,
                                   const int incx, const int incy, const int aMaxis, const double alpha,
                                   const double beta) {

 // Cache shape information in shared memory
 __shared__ LongType M;
 __shared__ LongType N;
 __shared__ bool betaPresent;
 __shared__ LongType totalThreads;
 __shared__ LongType aNstride;
 __shared__ LongType aMstride;
 __shared__ T3 alphaZ;
 __shared__ T3 betaZ;
 __shared__ const LongType* aShape;
 __shared__ const LongType* aStride;
 __shared__ LongType aRank;

 if (threadIdx.x == 0) {
   N = shape::length(xShapeInfo);
   M = shape::length(yShapeInfo);
   aRank = shape::rank(aShapeInfo);

   aShape = shape::shapeOf(aShapeInfo);
   aStride = shape::stride(aShapeInfo);

   aMstride = aStride[aMaxis];
   aNstride = aStride[aMaxis == 0 ? 1 : 0];

   totalThreads = gridDim.x * blockDim.x;
   betaPresent = beta != 0;
   alphaZ = alpha;
   betaZ = beta;
 }
 __syncthreads();

 const T1* A = reinterpret_cast<const T1*>(vA);
 const T2* X = reinterpret_cast<const T2*>(vX);
 T3* Y = reinterpret_cast<T3*>(vY);

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
 // Cache values in shared memory
 __shared__ T3* pairwiseMul;
 __shared__ T3 alphaZ;
 __shared__ T3 betaZ;
 __shared__ bool betaPresent;

 if (threadIdx.x == 0) {
   extern __shared__ unsigned char shmem[];
   pairwiseMul = reinterpret_cast<T3*>(shmem);
   alphaZ = alpha;
   betaZ = beta;
   betaPresent = beta != 0;
 }
 __syncthreads();

 T1* X = reinterpret_cast<T1*>(const_cast<void*>(vX));
 T2* Y = reinterpret_cast<T2*>(const_cast<void*>(vY));
 T3* Z = reinterpret_cast<T3*>(vZ);

 const int tid = blockIdx.x * blockDim.x + threadIdx.x;
 if (tid < length) {
   pairwiseMul[tid] = X[tid * incx] * Y[tid * incy];
 }

 __syncthreads();

 if (tid == 0) {
   T3 sum = static_cast<T3>(0);
   for (LongType i = 0; i < length; ++i) {
     sum = sum + pairwiseMul[i];
   }

   if (betaPresent)
     *Z = alphaZ * sum + betaZ * *Z;
   else
     *Z = alphaZ * sum;
 }
}

////////////////////////////////////////////////////////////////////////
template <typename T1, typename T2, typename T3>
SD_HOST static void usualDot(const dim3& launchDims, cudaStream_t* stream,
                            const LongType length, const double alpha, const void* vX, const LongType incx,
                            const void* vY, const LongType incy, const double beta, void* vZ) {
 usualCudaDot<T1, T2, T3><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(
     length, alpha, vX, incx, vY, incy, beta, vZ);
 DebugHelper::checkGlobalErrorCode("concat dot failed(...) failed");
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

 // Cache shape information in shared memory
 __shared__ struct {
   bool betaPresent;
   LongType aRank, bRank, cRank, K;
   LongType cLen, totalThreads;
   T3 alphaZ, betaZ;
   const LongType* aShape;
   const LongType* bShape;
   const LongType* cShape;
   const LongType* aStride;
   const LongType* bStride;
   const LongType* cStride;
   LongType* coords;
 } shared;

 if (threadIdx.x == 0) {
   extern __shared__ unsigned char shmem[];
   shared.coords = reinterpret_cast<LongType*>(shmem);
   shared.cLen = shape::length(cShapeInfo);

   shared.aRank = shape::rank(aShapeInfo);
   shared.bRank = shape::rank(bShapeInfo);
   shared.cRank = shape::rank(cShapeInfo);

   shared.aShape = shape::shapeOf(aShapeInfo);
   shared.bShape = shape::shapeOf(bShapeInfo);
   shared.cShape = shape::shapeOf(cShapeInfo);

   shared.aStride = shape::stride(aShapeInfo);
   shared.bStride = shape::stride(bShapeInfo);
   shared.cStride = shape::stride(cShapeInfo);

   shared.K = shared.aShape[aKaxis];
   shared.betaPresent = beta != 0;
   shared.totalThreads = gridDim.x * blockDim.x;
   shared.alphaZ = alpha;
   shared.betaZ = beta;
 }
 __syncthreads();

 const T1* A = reinterpret_cast<const T1*>(vA);
 const T2* B = reinterpret_cast<const T2*>(vB);
 T3* C = reinterpret_cast<T3*>(vC);

 auto aCoords = shared.coords + threadIdx.x * (shared.aRank + shared.bRank + shared.cRank);
 auto bCoords = aCoords + shared.aRank;
 auto cCoords = bCoords + shared.bRank;

 const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

 for (LongType i = tid; i < shared.cLen; i += shared.totalThreads) {
   // evaluate C coordinates
   INDEX2COORDS(i, shared.cRank, shared.cShape, cCoords);

   // calculate index of current batch
   LongType batchInd;
   if (cBatchDims != nullptr) {
     COORDS2INDEX(shared.cRank - 2, shared.cStride, cCoords, batchInd);
   }

   // evaluate A coordinates
   if (aBatchDims != nullptr) {
     INDEX2COORDS(batchInd, shared.aRank - 2, shared.aShape, aCoords);
   }
   aCoords[aMaxis] = cCoords[cMaxis];
   aCoords[aKaxis] = 0;

   // evaluate B coordinates
   if (bBatchDims != nullptr) {
     INDEX2COORDS(batchInd, shared.bRank - 2, shared.bShape, bCoords);
   }
   bCoords[bKaxis] = 0;
   bCoords[bNaxis] = cCoords[cNaxis];

   LongType aOffset, bOffset, cOffset;
   COORDS2INDEX(shared.aRank, shared.aStride, aCoords, aOffset);
   COORDS2INDEX(shared.bRank, shared.bStride, bCoords, bOffset);

   T3 val = A[aOffset] * B[bOffset];  // first iteration

   for (LongType j = 1; j < shared.K; ++j) {  // rest iterations
     aOffset += shared.aStride[aKaxis];
     bOffset += shared.bStride[bKaxis];
     val = val + A[aOffset] * B[bOffset];
   }

   COORDS2INDEX(shared.cRank, shared.cStride, cCoords, cOffset);

   if (shared.betaPresent)
     C[cOffset] = shared.alphaZ * val + shared.betaZ * C[cOffset];
   else
     C[cOffset] = shared.alphaZ * val;
 }
}

////////////////////////////////////////////////////////////////////////
template <typename T1, typename T2, typename T3>
SD_HOST static void batchedGemm(const int blocksPerGrid, const int threadsPerBlock, const int sharedMem,
                               cudaStream_t* stream, const void* vA, const LongType* aShapeInfo, const void* vB,
                               const LongType* bShapeInfo, void* vC, const LongType* cShapeInfo,
                               const LongType* aBatchDims, const LongType* bBatchDims, const LongType* cBatchDims,
                               const LongType aMaxis, const LongType aKaxis, const LongType bKaxis,
                               const LongType bNaxis, const LongType cMaxis, const LongType cNaxis, const double alpha,
                               const double beta) {
 batchedCudaGemm<T1, T2, T3><<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(
     vA, aShapeInfo, vB, bShapeInfo, vC, cShapeInfo, aBatchDims, bBatchDims, cBatchDims, aMaxis, aKaxis, bKaxis,
     bNaxis, cMaxis, cNaxis, alpha, beta);
 DebugHelper::checkGlobalErrorCode("batch gemm failed(...) failed");
}
//////////////////////////////////////////////////////////////////////////////
// MXK x KxN = MxN
NDArray* MmulHelper::mmulMxM(NDArray* A, NDArray* B, NDArray* C, double alpha, double beta,
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
   // Don't sync - let CUDA operations run asynchronously

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
     pA = A->dup('f');
     toDelete.push_back(pA);
     aMcont = true;
   }
   if (!bKcont && !bNcont) {
     pB = B->dup('f');
     toDelete.push_back(pB);
     bKcont = true;
   }
   if (!cMcont) {
     pC = C->dup('f');
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
   // Don't sync - let CUDA operations run asynchronously

   if (C != pC) C->assign(pC);

   for (int i = toDelete.size() - 1; i >= 0; --i) delete toDelete[i];
 }

 return C;
}////////////////////////////////////////////////////////////////////////////
// MXN x N = M
NDArray* MmulHelper::mmulMxV(NDArray* A, NDArray* X, NDArray* Y, const double alpha, const double beta,
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
   // Don't sync - let CUDA operations run asynchronously

 } else {
   NDArray* pA(const_cast<NDArray*>(A));

   bool aMcont = M == 1 || A->strideAt(0) == 1;
   bool aNcont = N == 1 || A->strideAt(1) == 1;

   if (!aMcont && !aNcont) {
     pA = A->dup('f');  // dup() already returns NDArray*, no need for new
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

   NDArray::registerSpecialUse({Y}, {pA, X});
   // Don't sync - let CUDA operations run asynchronously

   if (pA != A) delete pA;
 }

 return Y;
}////////////////////////////////////////////////////////////////////////////
// (X * Y) = Z[0]
NDArray* MmulHelper::dot(NDArray* X, NDArray* Y, NDArray* Z, const double alpha, const double beta) {
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
                              SD_NUMERIC_TYPES);

 NDArray::registerSpecialUse({Z}, {X, Y});
 // Don't sync - let CUDA operations run asynchronously

 return Z;
}
///////////////////////////////////////////////////////////////////
NDArray* MmulHelper::mmulNxN(NDArray* A, NDArray* B, NDArray* C, double alpha, double beta,
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
 auto* cExpectedShapePtr = aRank > bRank ? A->getShapeAsVector() : B->getShapeAsVector();
 std::vector<LongType> cExpectedShape = *cExpectedShapePtr;
 delete cExpectedShapePtr;
 cExpectedShape[cExpectedShape.size() - 2] = A->sizeAt(-2);
 cExpectedShape[cExpectedShape.size() - 1] = B->sizeAt(-1);

 if (C != nullptr) {
   if (!C->isSameShape(cExpectedShape))
     THROW_EXCEPTION("MmulHelper::mmulNxN: shape of C array is not suitable for AxB matrix multiplication !");
 } else
   C = new NDArray(outOrder, cExpectedShape, DataTypeUtils::pickPairwiseResultType(A->dataType(), B->dataType()),
                   A->getContext());

 if (C->isEmpty()) return C;

 // Try cuBLAS strided batch GEMM first for 3D tensors - much faster than custom kernel
 if (aRank == 3 && bRank == 3) {
   if (tryBlasStridedBatched(A, B, C, alpha, beta)) {
     return C;
   }
 }

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
 // Don't sync explicitly - manager destructor handles it if needed

 delete aDims;
 delete bDims;
 delete cDims;

 return C;
}

//////////////////////////////////////////////////////////////////////////
// cuBLAS Strided Batch GEMM - most efficient for contiguous batched data on GPU
// Uses cublasSgemmStridedBatched/cublasDgemmStridedBatched/cublasHgemmStridedBatched
// This avoids kernel launch overhead for each batch element
// Supports both 3D [batch, M, K] and 4D [batch0, batch1, M, K] tensors
//////////////////////////////////////////////////////////////////////////
bool MmulHelper::tryBlasStridedBatched(NDArray* A, NDArray* B, NDArray* C,
                                        double alpha, double beta) {
  const int aRank = A->rankOf();
  const int bRank = B->rankOf();
  const int cRank = C->rankOf();

  // Handle 3D and 4D tensors with matching ranks
  if (aRank != bRank || bRank != cRank || (aRank != 3 && aRank != 4)) {
    return false;
  }

  const auto xType = A->dataType();
  const auto yType = B->dataType();
  const auto zType = C->dataType();

  // Types must match for cuBLAS strided batched
  if (xType != yType || yType != zType) {
    return false;
  }

  // Only float, double, and half supported
  if (xType != DataType::FLOAT32 && xType != DataType::DOUBLE && xType != DataType::HALF) {
    return false;
  }

  // Validate buffers
  if (A->specialBuffer() == nullptr || B->specialBuffer() == nullptr || C->specialBuffer() == nullptr) {
    return false;
  }

  // For 4D tensors, flatten first two batch dimensions
  // 3D: [batch, M, K] @ [batch, K, N] -> [batch, M, N]
  // 4D: [b0, b1, M, K] @ [b0, b1, K, N] -> [b0, b1, M, N] (treat as [b0*b1, M, K] @ [b0*b1, K, N])
  LongType batchSize;
  LongType M, K, N;

  if (aRank == 3) {
    batchSize = A->sizeAt(0);
    M = A->sizeAt(1);
    K = A->sizeAt(2);
    N = B->sizeAt(2);
  } else {  // aRank == 4
    // Flatten first two batch dimensions
    batchSize = A->sizeAt(0) * A->sizeAt(1);
    M = A->sizeAt(2);
    K = A->sizeAt(3);
    N = B->sizeAt(3);

    // Verify batch dimensions match
    if (A->sizeAt(0) != B->sizeAt(0) || A->sizeAt(1) != B->sizeAt(1) ||
        A->sizeAt(0) != C->sizeAt(0) || A->sizeAt(1) != C->sizeAt(1)) {
      return false;
    }
  }

  if (M <= 0 || K <= 0 || N <= 0 || batchSize <= 0) {
    return false;
  }

  // Check inner dimensions match for matmul
  const int kAxisA = aRank - 1;  // Last axis of A
  const int kAxisB = aRank - 2;  // Second-to-last axis of B
  if (A->sizeAt(kAxisA) != B->sizeAt(kAxisB)) {
    return false;
  }

  // Check C dimensions match expected output
  if (C->sizeAt(-2) != M || C->sizeAt(-1) != N) {
    return false;
  }

  // cuBLAS is column-major, so we need to check for compatible memory layouts
  // For row-major [..., M, K]: stride[-1]=1, stride[-2]=K
  // We'll use the trick: C = A*B in row-major is equivalent to C^T = B^T * A^T in col-major
  // So we swap A and B and compute B * A instead

  // Check for contiguous row-major layout (innermost dimensions)
  const bool aRowMajor = (A->strideAt(-1) == 1) && (A->strideAt(-2) == K);
  const bool bRowMajor = (B->strideAt(-1) == 1) && (B->strideAt(-2) == N);
  const bool cRowMajor = (C->strideAt(-1) == 1) && (C->strideAt(-2) == N);

  if (!aRowMajor || !bRowMajor || !cRowMajor) {
    return false;
  }

  // Calculate strides for batched operation
  // For 3D: stride between batches is stride[0]
  // For 4D: stride between batches is stride[1] (stride within b0), and we need contiguous b0*b1
  long long strideA, strideB, strideC;

  if (aRank == 3) {
    const LongType expectedStrideA = M * K;
    const LongType expectedStrideB = K * N;
    const LongType expectedStrideC = M * N;

    if (A->strideAt(0) < expectedStrideA || B->strideAt(0) < expectedStrideB || C->strideAt(0) < expectedStrideC) {
      return false;
    }

    strideA = A->strideAt(0);
    strideB = B->strideAt(0);
    strideC = C->strideAt(0);
  } else {  // aRank == 4
    // For 4D, we need the stride between individual batch elements
    // Each batch element is M*K for A, K*N for B, M*N for C
    const LongType expectedStrideA = M * K;
    const LongType expectedStrideB = K * N;
    const LongType expectedStrideC = M * N;

    // Check that the 4D tensor is laid out as contiguous batches
    // stride[1] should be M*K (stride between b1 elements)
    // stride[0] should be b1*M*K (stride between b0 elements)
    if (A->strideAt(-3) < expectedStrideA || B->strideAt(-3) < expectedStrideB || C->strideAt(-3) < expectedStrideC) {
      return false;
    }

    // Also verify outer batch dimension is contiguous
    LongType b1 = A->sizeAt(1);
    if (A->strideAt(0) < b1 * expectedStrideA || B->strideAt(0) < b1 * expectedStrideB || C->strideAt(0) < b1 * expectedStrideC) {
      return false;
    }

    // Use the inner batch stride (stride[1] for 4D = stride[-3])
    strideA = A->strideAt(-3);
    strideB = B->strideAt(-3);
    strideC = C->strideAt(-3);
  }

  // Get cuBLAS handle
  auto handle = reinterpret_cast<cublasHandle_t*>(A->getContext()->getCublasHandle());
  auto stream = A->getContext()->getCudaStream();
  cublasSetStream(*handle, *stream);

  NDArray::prepareSpecialUse({C}, {A, B});

  cublasStatus_t status;

  // cuBLAS uses column-major, so we compute C^T = B^T * A^T
  // In terms of our row-major arrays: C = A * B becomes computing with swapped order
  // cublas: C_col = B_col * A_col where our row-major A is cublas's A^T
  // So we call cublas with (B, A) to get A*B in row-major

  if (xType == DataType::DOUBLE) {
    status = cublasDgemmStridedBatched(
        *handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,  // Swapped M,N for row-major
        &alpha,
        reinterpret_cast<const double*>(B->specialBuffer()), N, strideB,
        reinterpret_cast<const double*>(A->specialBuffer()), K, strideA,
        &beta,
        reinterpret_cast<double*>(C->specialBuffer()), N, strideC,
        batchSize);
  } else if (xType == DataType::FLOAT32) {
    float alphaF = static_cast<float>(alpha);
    float betaF = static_cast<float>(beta);
    status = cublasSgemmStridedBatched(
        *handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,  // Swapped M,N for row-major
        &alphaF,
        reinterpret_cast<const float*>(B->specialBuffer()), N, strideB,
        reinterpret_cast<const float*>(A->specialBuffer()), K, strideA,
        &betaF,
        reinterpret_cast<float*>(C->specialBuffer()), N, strideC,
        batchSize);
  } else if (xType == DataType::HALF) {
    __half alphaH = __float2half(static_cast<float>(alpha));
    __half betaH = __float2half(static_cast<float>(beta));
    status = cublasHgemmStridedBatched(
        *handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,  // Swapped M,N for row-major
        &alphaH,
        reinterpret_cast<const __half*>(B->specialBuffer()), N, strideB,
        reinterpret_cast<const __half*>(A->specialBuffer()), K, strideA,
        &betaH,
        reinterpret_cast<__half*>(C->specialBuffer()), N, strideC,
        batchSize);
  } else {
    NDArray::registerSpecialUse({C}, {A, B});
    return false;
  }

  if (status != CUBLAS_STATUS_SUCCESS) {
    NDArray::registerSpecialUse({C}, {A, B});
    return false;
  }

  NDArray::registerSpecialUse({C}, {A, B});
  // Don't sync - let CUDA operations run asynchronously

  return true;
}

} // namespace sd

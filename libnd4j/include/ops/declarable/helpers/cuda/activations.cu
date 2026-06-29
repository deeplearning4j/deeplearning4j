/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
// @author Yurii Shyrma (iuriish@yahoo.com), created on 19.04.2018
// @author raver119@gmail.com
//
#include <helpers/ConstantTadHelper.h>
#include <helpers/PointersManager.h>
#include <helpers/ShapeUtils.h>
#include <ops/declarable/helpers/activations.h>
#include <system/op_boilerplate.h>

#include <numeric>
#include <type_traits>

#include "execution/cuda/LaunchDims.h"


namespace sd {
namespace ops {
namespace helpers {

///////////////////////////////////////////////////////////////////
template <typename X, typename Y>
void SD_KERNEL __launch_bounds__(256, 2) preluCuda(const void *vx, const LongType *xShapeInfo, const void *vy, const LongType *yShapeInfo,
                         void *vz) {
  const auto x = reinterpret_cast<const X *>(vx);
  const auto y = reinterpret_cast<const Y *>(vy);
  auto z = reinterpret_cast<X *>(vz);

  __shared__ LongType xzLen;
  __shared__ int xzRank, yRank;
  __shared__ const LongType *xzShape;
  __shared__ const LongType *xzStride;
  __shared__ const LongType *yShape;
  __shared__ const LongType *yStride;

  if (threadIdx.x == 0) {
    xzLen = shape::length(xShapeInfo);
    xzRank = shape::rank(xShapeInfo);
    yRank = shape::rank(yShapeInfo);
    xzShape = shape::shapeOf(xShapeInfo);
    xzStride = shape::stride(xShapeInfo);
    yShape = shape::shapeOf(yShapeInfo);
    yStride = shape::stride(yShapeInfo);
  }
  __syncthreads();

  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  LongType coords[SD_MAX_RANK];

  for (int i = tid; i < xzLen; i += blockDim.x * gridDim.x) {
    INDEX2COORDS(i, xzRank, xzShape, coords);

    LongType xzOffset;
    COORDS2INDEX(xzRank, xzStride, coords, xzOffset);
    const auto xVal = x[xzOffset];

    if (xVal < 0) {
      for (LongType j = 0; j < yRank; ++j)
        if (yShapeInfo[j + 1] == 1) coords[j + 1] = 0;

      LongType yOffset;
      COORDS2INDEX(yRank, yStride, coords + 1, yOffset);
      z[xzOffset] = xVal * y[yOffset];
    } else {
      z[xzOffset] = xVal;
    }
  }
}
///////////////////////////////////////////////////////////////////
template <typename X, typename Y>
void preluCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const int sharedMem,
                       const cudaStream_t *stream, const void *vx, const LongType *xShapeInfo, const void *vy,
                       const LongType *yShapeInfo, void *vz) {
  preluCuda<X, Y><<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(vx, xShapeInfo, vy, yShapeInfo, vz);
  sd::DebugHelper::checkGlobalErrorCode("prelu  failed");

}

///////////////////////////////////////////////////////////////////
void prelu(LaunchContext *context, NDArray *input, NDArray *alpha, NDArray *output) {
  dim3 launchDims = getLaunchDims("prelu");
  // Cap at 256: preluCuda kernel uses __launch_bounds__(256, 2)
  if (launchDims.y > 256) launchDims.y = 256;

  const auto xType = input->dataType();
  const auto yType = alpha->dataType();

  NDArray::prepareSpecialUse({output}, {&input, &alpha});
  BUILD_SINGLE_SELECTOR_TWICE(
      xType, preluCudaLauncher,
      (launchDims.x, launchDims.y, launchDims.z, context->getCudaStream(), input->specialBuffer(),
          input->specialShapeInfo(), alpha->specialBuffer(), alpha->specialShapeInfo(), output->specialBuffer()),
      SD_FLOAT_TYPES);
  NDArray::registerSpecialUse({output}, {&input, &alpha});
  // Don't sync - let CUDA operations run asynchronously
}

///////////////////////////////////////////////////////////////////
template <typename X, typename Y>
void SD_KERNEL __launch_bounds__(256, 2) preluBPCuda(const void *vIn, const LongType *inShapeInfo, const void *vAlpha,
                           const LongType *alphaShapeInfo, const void *vdLdO, const LongType *dLdOShapeInfo,
                           void *vdLdI, const LongType *dLdIShapeInfo, void *vdLdA,
                           const LongType *dLdAShapeInfo) {
  const auto in = reinterpret_cast<const X *>(vIn);
  const auto alpha = reinterpret_cast<const Y *>(vAlpha);
  const auto dLdO = reinterpret_cast<const Y *>(vdLdO);
  auto dLdI = reinterpret_cast<Y *>(vdLdI);
  auto dLdA = reinterpret_cast<Y *>(vdLdA);

  __shared__ LongType inLen, totalThreads;
  __shared__ int inRank, alphaRank;
  __shared__ const LongType *inShape;
  __shared__ const LongType *inStride;
  __shared__ const LongType *dLdOStride;
  __shared__ const LongType *dLdIStride;
  __shared__ const LongType *alphaStride;
  __shared__ const LongType *dLdAStride;

  if (threadIdx.x == 0) {
    inLen = shape::length(inShapeInfo);
    totalThreads = gridDim.x * blockDim.x;

    inRank = shape::rank(inShapeInfo);
    alphaRank = shape::rank(alphaShapeInfo);

    // Cache shapes and strides
    inShape = shape::shapeOf(inShapeInfo);
    inStride = shape::stride(inShapeInfo);
    dLdOStride = shape::stride(dLdOShapeInfo);
    dLdIStride = shape::stride(dLdIShapeInfo);
    alphaStride = shape::stride(alphaShapeInfo);
    dLdAStride = shape::stride(dLdAShapeInfo);
  }
  __syncthreads();

  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  LongType coords[SD_MAX_RANK];

  for (int i = tid; i < inLen; i += totalThreads) {
    INDEX2COORDS(i, inRank, inShape, coords);

    LongType inOffset, dLdOOffset, dLdIOffset;
    COORDS2INDEX(inRank, inStride, coords, inOffset);
    COORDS2INDEX(inRank, dLdOStride, coords, dLdOOffset);
    COORDS2INDEX(inRank, dLdIStride, coords, dLdIOffset);

    const auto xVal = in[inOffset];
    const auto grO = dLdO[dLdOOffset];

    if (xVal < 0) {
      for (LongType j = 0; j < alphaRank; ++j)
        if (alphaShapeInfo[j + 1] == 1) coords[j + 1] = 0;

      LongType alphaOffset, dLdAOffset;
      COORDS2INDEX(alphaRank, alphaStride, coords + 1, alphaOffset);
      COORDS2INDEX(alphaRank, dLdAStride, coords + 1, dLdAOffset);

      dLdI[dLdIOffset] = grO * alpha[alphaOffset];

      math::atomics::sd_atomicAdd<Y>(&dLdA[dLdAOffset], static_cast<Y>(grO * xVal));
    } else {
      dLdI[dLdIOffset] = grO;
    }
  }
}

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Y>
void SD_HOST preluBPCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const int sharedMem,
                                 const cudaStream_t *stream, const void *vIn, const LongType *inShapeInfo,
                                 const void *vAlpha, const LongType *alphaShapeInfo, const void *vdLdO,
                                 const LongType *dLdOShapeInfo, void *vdLdI, const LongType *dLdIShapeInfo,
                                 void *vdLdA, const LongType *dLdAShapeInfo) {
  preluBPCuda<X, Y><<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(
      vIn, inShapeInfo, vAlpha, alphaShapeInfo, vdLdO, dLdOShapeInfo, vdLdI, dLdIShapeInfo, vdLdA, dLdAShapeInfo);
  sd::DebugHelper::checkGlobalErrorCode("prelu bp failed");

}

//////////////////////////////////////////////////////////////////////////
void preluBP(LaunchContext *context, NDArray *input, NDArray *alpha, NDArray *dLdO, NDArray *dLdI,
             NDArray *dLdA) {
  dim3 launchDims = getLaunchDims("prelu");
  // Cap at 256: preluBPCuda kernel uses __launch_bounds__(256, 2)
  if (launchDims.y > 256) launchDims.y = 256;

  const auto xType = input->dataType();
  const auto zType = alpha->dataType();

  // prepareSpecialUse must come before nullify() to allocate the device buffer first;
  // dLdA uses atomicAdd accumulation so it must be zero-initialized on device.
  // nullify() is a no-op on device when special()==nullptr (from Java/JNI side).
  NDArray::prepareSpecialUse({dLdI, dLdA}, {input, alpha, dLdO});
  dLdA->nullify();
  BUILD_SINGLE_SELECTOR_TWICE(
      xType, preluBPCudaLauncher,
      (launchDims.x, launchDims.y, launchDims.z, context->getCudaStream(), input->specialBuffer(),
          input->specialShapeInfo(), alpha->specialBuffer(), alpha->specialShapeInfo(), dLdO->specialBuffer(),
          dLdO->specialShapeInfo(), dLdI->specialBuffer(), dLdI->specialShapeInfo(), dLdA->specialBuffer(),
          dLdA->specialShapeInfo()),
      SD_FLOAT_TYPES);
  NDArray::registerSpecialUse({dLdI, dLdA}, {input, alpha, dLdO});
  // Don't sync - let CUDA operations run asynchronously
}

///////////////////////////////////////////////////////////////////
// Tree reduction for max (matches aggregatePartials pattern)
template <typename T>
SD_DEVICE void reduceMaxHybrid(T* sPartials, LongType tid, LongType numItems) {
  for (LongType s = numItems / 2; s > 0; s >>= 1) {
    if (tid < s && (tid + s) < numItems) {
      sPartials[tid] = math::sd_max<T>(sPartials[tid], sPartials[tid + s]);
    }
    __syncthreads();
  }
}

// Tree reduction for sum (matches aggregatePartials pattern)
template <typename T>
SD_DEVICE void reduceSumHybrid(T* sPartials, LongType tid, LongType numItems) {
  for (LongType s = numItems / 2; s > 0; s >>= 1) {
    if (tid < s && (tid + s) < numItems) {
      sPartials[tid] += sPartials[tid + s];
    }
    __syncthreads();
  }
}

///////////////////////////////////////////////////////////////////
// Parallel softmax using tree reduction (same pattern as reduce ops)
template <typename T>
SD_DEVICE void softMaxForVectorCuda(const void *vx, const LongType *xShapeInfo, void *vz,
                                    const LongType *zShapeInfo) {
  auto inBuff = reinterpret_cast<const T *>(vx);
  auto outBuff = reinterpret_cast<T *>(vz);

  // Shared memory for reductions
  __shared__ T sPartials[SD_CUDA_BLOCK_SIZE];
  __shared__ T globalMax;
  __shared__ T globalSum;
  __shared__ LongType tadLen;
  __shared__ int xRank;
  __shared__ LongType xStride0;
  __shared__ LongType zStride0;

  if (threadIdx.x == 0) {
    tadLen = shape::length(xShapeInfo);
    xRank = shape::rank(xShapeInfo);
    xStride0 = shape::stride(xShapeInfo)[0];
    zStride0 = shape::stride(zShapeInfo)[0];
  }
  __syncthreads();

  // Use blockDim.x (power of 2) for reductions — NOT min(tadLen, blockDim.x).
  // Non-power-of-2 numItems causes the tree reduction to skip elements
  // (e.g., numItems=6: s=3,1 skips sPartials[2]).
  // All threads initialize sPartials to identity values, so reducing the
  // full blockDim.x is safe and eliminates this class of bug.
  const LongType numItems = static_cast<LongType>(blockDim.x);

  // Fast path for rank 1 TADs (most common for attention softmax)
  if (xRank == 1) {
    // Phase 1: Each thread finds local max
    T threadMax = -DataTypeUtils::max<T>();
    for (LongType j = threadIdx.x; j < tadLen; j += blockDim.x) {
      threadMax = math::sd_max<T>(threadMax, inBuff[j * xStride0]);
    }
    sPartials[threadIdx.x] = threadMax;
    __syncthreads();

    reduceMaxHybrid<T>(sPartials, threadIdx.x, numItems);
    if (threadIdx.x == 0) globalMax = sPartials[0];
    __syncthreads();

    // Phase 2: Compute exp and local sum
    T threadSum = static_cast<T>(0);
    for (LongType j = threadIdx.x; j < tadLen; j += blockDim.x) {
      T temp = math::sd_exp<T, T>(inBuff[j * xStride0] - globalMax);
      outBuff[j * zStride0] = temp;
      threadSum += temp;
    }
    sPartials[threadIdx.x] = threadSum;
    __syncthreads();

    reduceSumHybrid<T>(sPartials, threadIdx.x, numItems);
    if (threadIdx.x == 0) globalSum = sPartials[0];
    __syncthreads();

    // Phase 3: Normalize
    for (LongType j = threadIdx.x; j < tadLen; j += blockDim.x) {
      outBuff[j * zStride0] /= globalSum;
    }
  } else {
    // General path for higher ranks
    __shared__ int zRank;
    __shared__ const LongType *xShape;
    __shared__ const LongType *xStride;
    __shared__ const LongType *zShape;
    __shared__ const LongType *zStride;

    if (threadIdx.x == 0) {
      zRank = shape::rank(zShapeInfo);
      xShape = shape::shapeOf(xShapeInfo);
      xStride = shape::stride(xShapeInfo);
      zShape = shape::shapeOf(zShapeInfo);
      zStride = shape::stride(zShapeInfo);
    }
    __syncthreads();

    // Phase 1: Each thread finds local max
    T threadMax = -DataTypeUtils::max<T>();
    for (LongType j = threadIdx.x; j < tadLen; j += blockDim.x) {
      LongType xCoords[SD_MAX_RANK];
      LongType xOffset;
      INDEX2COORDS(j, xRank, xShape, xCoords);
      COORDS2INDEX(xRank, xStride, xCoords, xOffset);
      T val = inBuff[xOffset];
      if (val > threadMax) threadMax = val;
    }
    sPartials[threadIdx.x] = threadMax;
    __syncthreads();

    reduceMaxHybrid<T>(sPartials, threadIdx.x, numItems);
    if (threadIdx.x == 0) globalMax = sPartials[0];
    __syncthreads();

    // Phase 2: Compute exp and local sum
    T threadSum = static_cast<T>(0);
    for (LongType j = threadIdx.x; j < tadLen; j += blockDim.x) {
      LongType xCoords[SD_MAX_RANK];
      LongType zCoords[SD_MAX_RANK];
      LongType xOffset, zOffset;
      INDEX2COORDS(j, xRank, xShape, xCoords);
      COORDS2INDEX(xRank, xStride, xCoords, xOffset);
      INDEX2COORDS(j, zRank, zShape, zCoords);
      COORDS2INDEX(zRank, zStride, zCoords, zOffset);

      T temp = math::sd_exp<T, T>(inBuff[xOffset] - globalMax);
      outBuff[zOffset] = temp;
      threadSum += temp;
    }
    sPartials[threadIdx.x] = threadSum;
    __syncthreads();

    reduceSumHybrid<T>(sPartials, threadIdx.x, numItems);
    if (threadIdx.x == 0) globalSum = sPartials[0];
    __syncthreads();

    // Phase 3: Normalize
    for (LongType j = threadIdx.x; j < tadLen; j += blockDim.x) {
      LongType zCoords[SD_MAX_RANK];
      LongType zOffset;
      INDEX2COORDS(j, zRank, zShape, zCoords);
      COORDS2INDEX(zRank, zStride, zCoords, zOffset);
      outBuff[zOffset] /= globalSum;
    }
  }
}

template <typename T>
void SD_KERNEL __launch_bounds__(256, 2) softMaxForVectorCudaGlobal(const void *vx, const LongType *xShapeInfo, void *vz,
                                          const LongType *zShapeInfo, LongType numOfSubArrs) {
  softMaxForVectorCuda<T>(vx, xShapeInfo, vz, zShapeInfo);
}

///////////////////////////////////////////////////////////////////
template <typename T>
void softMaxForVectorCudaLauncher(const cudaStream_t *stream, const void *vx, const LongType *xShapeInfo, void *vz,
                                  const LongType *zShapeInfo, LongType numTads) {
  softMaxForVectorCudaGlobal<T><<<1, SD_CUDA_BLOCK_SIZE, 0, *stream>>>(vx, xShapeInfo, vz, zShapeInfo, numTads);
  sd::DebugHelper::checkGlobalErrorCode("softmax  failed");

}

///////////////////////////////////////////////////////////////////
// Warp-level max reduction using shuffle
template <typename T>
SD_DEVICE SD_INLINE T warpReduceMax(T val) {
  for (int offset = 16; offset > 0; offset /= 2) {
    val = math::sd_max<T>(val, __shfl_down_sync(0xffffffff, val, offset));
  }
  return val;
}

// Warp-level sum reduction using shuffle
template <typename T>
SD_DEVICE SD_INLINE T warpReduceSum(T val) {
  for (int offset = 16; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

///////////////////////////////////////////////////////////////////
// Warp-per-TAD kernel with vectorized float4 loads for contiguous data
// Each warp processes one TAD, using float4 for 4x memory bandwidth
template <typename T>
SD_KERNEL __launch_bounds__(256, 2) static void softMaxCudaWarpPerTadVec4(const void *vx, const LongType *xOffsets,
                                                 void *vz, const LongType *zOffsets,
                                                 LongType numTads, LongType tadLen) {
  const auto x = reinterpret_cast<const float *>(vx);
  auto z = reinterpret_cast<float *>(vz);

  const int warpId = threadIdx.x / 32;
  const int laneId = threadIdx.x % 32;
  const int numWarpsPerBlock = blockDim.x / 32;

  // Each warp handles one TAD
  for (LongType tadIdx = blockIdx.x * numWarpsPerBlock + warpId; tadIdx < numTads; tadIdx += gridDim.x * numWarpsPerBlock) {
    const float* inBuff = x + xOffsets[tadIdx];
    float* outBuff = z + zOffsets[tadIdx];

    // Process 4 elements at a time with float4
    const LongType vec4Len = tadLen / 4;
    const LongType remainder = tadLen % 4;

    // Phase 1: Find max using float4
    float threadMax = -3.4028235E38f;
    for (LongType j = laneId; j < vec4Len; j += 32) {
      float4 val = reinterpret_cast<const float4*>(inBuff)[j];
      threadMax = fmaxf(threadMax, fmaxf(fmaxf(val.x, val.y), fmaxf(val.z, val.w)));
    }
    // Handle remainder
    for (LongType j = vec4Len * 4 + laneId; j < tadLen; j += 32) {
      threadMax = fmaxf(threadMax, inBuff[j]);
    }
    float maxVal = warpReduceMax<float>(threadMax);
    maxVal = __shfl_sync(0xffffffff, maxVal, 0);

    // Phase 2: Compute exp and sum using float4
    float threadSum = 0.0f;
    for (LongType j = laneId; j < vec4Len; j += 32) {
      float4 val = reinterpret_cast<const float4*>(inBuff)[j];
      float4 expVal;
      expVal.x = __expf(val.x - maxVal);
      expVal.y = __expf(val.y - maxVal);
      expVal.z = __expf(val.z - maxVal);
      expVal.w = __expf(val.w - maxVal);
      reinterpret_cast<float4*>(outBuff)[j] = expVal;
      threadSum += expVal.x + expVal.y + expVal.z + expVal.w;
    }
    // Handle remainder
    for (LongType j = vec4Len * 4 + laneId; j < tadLen; j += 32) {
      float temp = __expf(inBuff[j] - maxVal);
      outBuff[j] = temp;
      threadSum += temp;
    }
    float sumVal = warpReduceSum<float>(threadSum);
    sumVal = __shfl_sync(0xffffffff, sumVal, 0);

    // Phase 3: Normalize using float4
    float invSum = 1.0f / sumVal;
    for (LongType j = laneId; j < vec4Len; j += 32) {
      float4 val = reinterpret_cast<float4*>(outBuff)[j];
      val.x *= invSum;
      val.y *= invSum;
      val.z *= invSum;
      val.w *= invSum;
      reinterpret_cast<float4*>(outBuff)[j] = val;
    }
    // Handle remainder
    for (LongType j = vec4Len * 4 + laneId; j < tadLen; j += 32) {
      outBuff[j] *= invSum;
    }
  }
}

///////////////////////////////////////////////////////////////////
// Warp-per-TAD kernel: each warp processes one TAD independently
// Optimized for contiguous short TADs (typical in attention: 128-512 elements)
// Multiple warps per block = multiple TADs processed in parallel
template <typename T>
SD_KERNEL __launch_bounds__(256, 2) static void softMaxCudaWarpPerTad(const void *vx, const LongType *xTadShapeInfo, const LongType *xOffsets,
                                            void *vz, const LongType *zTadShapeInfo, const LongType *zOffsets,
                                            LongType numTads, LongType tadLen, LongType xStride0, LongType zStride0) {
  const auto x = reinterpret_cast<const T *>(vx);
  auto z = reinterpret_cast<T *>(vz);

  const int warpId = threadIdx.x / 32;
  const int laneId = threadIdx.x % 32;
  const int numWarpsPerBlock = blockDim.x / 32;

  // Each warp handles one TAD, grid-stride over TADs
  for (LongType tadIdx = blockIdx.x * numWarpsPerBlock + warpId; tadIdx < numTads; tadIdx += gridDim.x * numWarpsPerBlock) {
    const T* inBuff = x + xOffsets[tadIdx];
    T* outBuff = z + zOffsets[tadIdx];

    // Phase 1: Find max (each lane handles multiple elements) - contiguous access
    T threadMax = -DataTypeUtils::max<T>();
    for (LongType j = laneId; j < tadLen; j += 32) {
      threadMax = math::sd_max<T>(threadMax, inBuff[j]);
    }
    T maxVal = warpReduceMax<T>(threadMax);
    maxVal = __shfl_sync(0xffffffff, maxVal, 0);  // Broadcast max to all lanes

    // Phase 2: Compute exp and sum - contiguous access
    T threadSum = static_cast<T>(0);
    for (LongType j = laneId; j < tadLen; j += 32) {
      T temp = math::sd_exp<T, T>(inBuff[j] - maxVal);
      outBuff[j] = temp;
      threadSum += temp;
    }
    T sumVal = warpReduceSum<T>(threadSum);
    sumVal = __shfl_sync(0xffffffff, sumVal, 0);  // Broadcast sum to all lanes

    // Phase 3: Normalize - contiguous access
    T invSum = static_cast<T>(1) / sumVal;
    for (LongType j = laneId; j < tadLen; j += 32) {
      outBuff[j] *= invSum;
    }
  }
}

///////////////////////////////////////////////////////////////////
// Multi-TAD kernel: each block processes multiple TADs using grid-stride loop
// Uses warp shuffle for fast reductions - for longer TADs
template <typename T>
SD_KERNEL __launch_bounds__(256, 2) static void softMaxCuda(const void *vx, const LongType *xTadShapeInfo, const LongType *xOffsets,
                                  void *vz, const LongType *zTadShapeInfo, const LongType *zOffsets, LongType numTads) {
  const auto x = reinterpret_cast<const T *>(vx);
  auto z = reinterpret_cast<T *>(vz);

  // Shared memory for inter-warp reduction (max 32 warps per block)
  __shared__ T warpPartials[32];
  __shared__ T globalMax;
  __shared__ T globalSum;
  __shared__ LongType tadLen;
  __shared__ int xRank;
  __shared__ LongType xStride0;
  __shared__ LongType zStride0;

  const int warpId = threadIdx.x / 32;
  const int laneId = threadIdx.x % 32;
  const int numWarps = (blockDim.x + 31) / 32;

  // Cache TAD shape info once (same for all TADs)
  if (threadIdx.x == 0) {
    tadLen = shape::length(xTadShapeInfo);
    xRank = shape::rank(xTadShapeInfo);
    xStride0 = shape::stride(xTadShapeInfo)[0];
    zStride0 = shape::stride(zTadShapeInfo)[0];
  }
  __syncthreads();

  // Grid-stride loop: each block processes multiple TADs
  for (LongType tadIdx = blockIdx.x; tadIdx < numTads; tadIdx += gridDim.x) {
    const T* inBuff = x + xOffsets[tadIdx];
    T* outBuff = z + zOffsets[tadIdx];

    // Fast path for rank 1 TADs
    if (xRank == 1) {
      // Phase 1: Find max using warp shuffles
      T threadMax = -DataTypeUtils::max<T>();
      for (LongType j = threadIdx.x; j < tadLen; j += blockDim.x) {
        threadMax = math::sd_max<T>(threadMax, inBuff[j * xStride0]);
      }

      // Warp-level reduction
      threadMax = warpReduceMax<T>(threadMax);

      // Store warp results
      if (laneId == 0) warpPartials[warpId] = threadMax;
      __syncthreads();

      // Final reduction by first warp
      if (warpId == 0) {
        T val = (laneId < numWarps) ? warpPartials[laneId] : -DataTypeUtils::max<T>();
        val = warpReduceMax<T>(val);
        if (laneId == 0) globalMax = val;
      }
      __syncthreads();

      // Phase 2: Compute exp and sum
      T threadSum = static_cast<T>(0);
      for (LongType j = threadIdx.x; j < tadLen; j += blockDim.x) {
        T temp = math::sd_exp<T, T>(inBuff[j * xStride0] - globalMax);
        outBuff[j * zStride0] = temp;
        threadSum += temp;
      }

      // Warp-level sum reduction
      threadSum = warpReduceSum<T>(threadSum);

      // Store warp results
      if (laneId == 0) warpPartials[warpId] = threadSum;
      __syncthreads();

      // Final reduction by first warp
      if (warpId == 0) {
        T val = (laneId < numWarps) ? warpPartials[laneId] : static_cast<T>(0);
        val = warpReduceSum<T>(val);
        if (laneId == 0) globalSum = val;
      }
      __syncthreads();

      // Phase 3: Normalize
      for (LongType j = threadIdx.x; j < tadLen; j += blockDim.x) {
        outBuff[j * zStride0] /= globalSum;
      }
    } else {
      // General path - delegate to device function
      softMaxForVectorCuda<T>(inBuff, xTadShapeInfo, outBuff, zTadShapeInfo);
    }
    __syncthreads();  // Ensure all threads done before next TAD
  }
}

///////////////////////////////////////////////////////////////////
template <typename T>
static void softMaxWarpPerTadLauncher(const int blocksPerGrid, const int threadsPerBlock,
                                      const cudaStream_t *stream, const void *vx, const LongType *xTadShapeInfo,
                                      const LongType *xOffsets, void *vz, const LongType *zTadShapeInfo,
                                      const LongType *zOffsets, LongType numTads, LongType tadLen,
                                      LongType xStride0, LongType zStride0) {
  softMaxCudaWarpPerTad<T><<<blocksPerGrid, threadsPerBlock, 0, *stream>>>(
      vx, xTadShapeInfo, xOffsets, vz, zTadShapeInfo, zOffsets, numTads, tadLen, xStride0, zStride0);
  sd::DebugHelper::checkGlobalErrorCode("softmax warp-per-tad failed");
}

///////////////////////////////////////////////////////////////////
template <typename T>
static void softMaxCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const int sharedMem,
                                const cudaStream_t *stream, const void *vx, const LongType *xTadShapeInfo,
                                const LongType *xOffsets, void *vz, const LongType *zTadShapeInfo,
                                const LongType *zOffsets, LongType numTads) {
  softMaxCuda<T><<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(vx, xTadShapeInfo, xOffsets, vz, zTadShapeInfo,
                                                                         zOffsets, numTads);
  sd::DebugHelper::checkGlobalErrorCode("softmax failed");
}

//////////////////////////////////////////////////////////////////////////
void softmax(LaunchContext *context, NDArray *input, NDArray *output, const int dimension) {
  const int rank = input->rankOf();
  // Normalize negative dimension before passing to tadForDimensions.
  // tadForDimensions treats -1 as a sentinel meaning "entire array", not "last dimension".
  const int dim = (dimension < 0) ? (rank + dimension) : dimension;

  if (input->isVector()) {
    if (rank == 1 || input->sizeAt(dim) != 1) {
      NDArray::prepareSpecialUse({output}, {input});
      BUILD_SINGLE_SELECTOR(input->dataType(), softMaxForVectorCudaLauncher,
                            (context->getCudaStream(), input->specialBuffer(), input->specialShapeInfo(),
                                output->specialBuffer(), output->specialShapeInfo(), 1),
                            SD_FLOAT_TYPES);
      NDArray::registerSpecialUse({output}, {input});
    } else
      *output = 1.;
  } else {
    auto packX = ConstantTadHelper::getInstance().tadForDimensions(input->shapeInfo(), {(LongType)dim});
    auto packZ = ConstantTadHelper::getInstance().tadForDimensions(output->shapeInfo(), {(LongType)dim});

    LongType numTads = packX->numberOfTads();
    LongType tadLen = shape::length(packX->primaryShapeInfo());
    int xRank = shape::rank(packX->primaryShapeInfo());
    LongType xStride0 = shape::stride(packX->primaryShapeInfo())[0];
    LongType zStride0 = shape::stride(packZ->primaryShapeInfo())[0];

    NDArray::prepareSpecialUse({output}, {input});

    // Get optimized launch dimensions from centralized config
    dim3 softmaxDims = getSoftmaxDims(numTads, tadLen);

    BUILD_SINGLE_SELECTOR(input->dataType(), softMaxCudaLauncher,
                          (softmaxDims.x, softmaxDims.y,
                              softmaxDims.z,
                              context->getCudaStream(),
                              input->specialBuffer(),
                              packX->specialShapeInfo(),
                              packX->specialOffsets(), output->specialBuffer(),
                              packZ->specialShapeInfo(),
                              packZ->specialOffsets(), numTads),
                          SD_FLOAT_TYPES);
    NDArray::registerSpecialUse({output}, {input});
  }

}

///////////////////////////////////////////////////////////////////
template <typename T>
void SD_KERNEL __launch_bounds__(256, 2) logSoftMaxForVectorCuda(const void *vx, const LongType *xzShapeInfo, void *vz) {
  // logic of this kernel is based on assumption gridDim = 1

  const auto x = reinterpret_cast<const T *>(vx);
  auto z = reinterpret_cast<T *>(vz);

  __shared__ LongType len;
  __shared__ int numOfIters;
  __shared__ int xzRank;
  __shared__ const LongType *xzShape;
  __shared__ const LongType *xzStride;
  __shared__ T shmem[SD_CUDA_BLOCK_SIZE];

  if (threadIdx.x == 0) {
    len = shape::length(xzShapeInfo);
    numOfIters = (len + blockDim.x - 1) / blockDim.x;  // ceil (len / blockDim.x)

    // Cache rank, shape and stride information
    xzRank = shape::rank(xzShapeInfo);
    xzShape = shape::shapeOf(xzShapeInfo);
    xzStride = shape::stride(xzShapeInfo);
  }
  __syncthreads();

  T temp = -DataTypeUtils::max<T>();

  // ************ evaluate max element in input array x ************ //
  for (int i = 0; i < numOfIters; ++i) {
    const LongType elemIdx = i * blockDim.x + threadIdx.x;
    if (elemIdx < len) {
      LongType offset;
      sd::LongType coords[SD_MAX_RANK];
      INDEX2COORDS(elemIdx, xzRank, xzShape, coords);
      COORDS2INDEX(xzRank, xzStride, coords, offset);
      shmem[threadIdx.x] = (threadIdx.x != 0) ? x[offset] : math::sd_max<T>(x[offset], temp);  // take into account max element evaluated on previous iteration and stored in temp
    } else {
      shmem[threadIdx.x] = -DataTypeUtils::max<T>();
    }

    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s /= 2) {
      if (threadIdx.x < s) shmem[threadIdx.x] = math::sd_max<T>(shmem[threadIdx.x], shmem[threadIdx.x + s]);
      __syncthreads();
    }

    temp = shmem[0];  // save max value calculated at current iteration
  }

  const T max = temp;
  temp = 0;

  // ************ evaluate value of exp(x[offset] - max) per each element, store it to shared memory shmem ************
  // at the same time evaluate sum of exponents, sum will be stored in shmem[0]
  for (int i = 0; i < numOfIters; ++i) {
    const LongType elemIdx = i * blockDim.x + threadIdx.x;
    if (elemIdx < len) {
      LongType offset;
      sd::LongType coords[SD_MAX_RANK];
      INDEX2COORDS(elemIdx, xzRank, xzShape, coords);
      COORDS2INDEX(xzRank, xzStride, coords, offset);
      z[offset] = math::sd_exp<T, T>(x[offset] - max);
      shmem[threadIdx.x] = (threadIdx.x != 0) ? z[offset] : (z[offset] + temp);  // take into account sum element evaluated on previous iteration and stored in temp
    } else {
      shmem[threadIdx.x] = 0;
    }

    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s /= 2) {
      if (threadIdx.x < s) shmem[threadIdx.x] += shmem[threadIdx.x + s];
      __syncthreads();
    }

    temp = shmem[0];  // save sum calculated at current iteration
  }

  // ************ evaluate log(z[offset] / sum)  ************ //
  for (int i = 0; i < numOfIters; ++i) {
    const LongType elemIdx = i * blockDim.x + threadIdx.x;
    if (elemIdx < len) {  // Added bounds check that was missing in original
      LongType offset;
      sd::LongType coords[SD_MAX_RANK];
      INDEX2COORDS(elemIdx, xzRank, xzShape, coords);
      COORDS2INDEX(xzRank, xzStride, coords, offset);
      z[offset] = math::sd_log<T, T>(z[offset] / shmem[0]);
    }
  }
}
///////////////////////////////////////////////////////////////////
template <typename T>
void logSoftMaxForVectorCudaLauncher(const cudaStream_t *stream, const void *vx, const LongType *xzShapeInfo,
                                     void *vz) {
  dim3 launchDims = getLaunchDims("softmax");
  logSoftMaxForVectorCuda<T><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(vx, xzShapeInfo, vz);
  sd::DebugHelper::checkGlobalErrorCode("logsoftmax  failed");

}

//////////////////////////////////////////////////////////////////////////
void logSoftmax(LaunchContext *context, NDArray *input, NDArray *output, const int dimension) {
  const int rank = input->rankOf();

  if (input->isVector()) {
    if (rank == 1 || input->sizeAt(dimension) != 1) {
      NDArray::prepareSpecialUse({output}, {input});
      BUILD_SINGLE_SELECTOR(
          input->dataType(), logSoftMaxForVectorCudaLauncher,
          (context->getCudaStream(), input->specialBuffer(), input->specialShapeInfo(), output->specialBuffer()),
          SD_FLOAT_TYPES);
      NDArray::registerSpecialUse({output}, {input});
    } else
      *output = 0.;
  } else {
    // log(softmax(x)) = x - max(x) - log(sum(exp(x - max(x))))
    // All ops below are high-level NDArray operations that manage their own coherence.
    std::vector<LongType> dim = {static_cast<LongType>(dimension)};
    auto maxAlongDim = const_cast<NDArray *>(input)->reduceAlongDimension(reduce::Max, &dim, true);
    auto inputMinusMax = *input - *maxAlongDim;
    // Compute exp(x - max) into a temp array
    NDArray expTemp(output->shapeInfo(), false, const_cast<LaunchContext*>(context), true);
    inputMinusMax->applyTransform(transform::Exp, &expTemp);
    auto sumExp = expTemp.reduceAlongDimension(reduce::Sum, &dim, true);
    sumExp->applyTransform(transform::Log, sumExp);
    // output = (x - max) - log(sumExp)
    auto* result = (*inputMinusMax) - (*sumExp);
    output->assign(result);
    delete result;
    delete maxAlongDim;
    delete inputMinusMax;
    delete sumExp;
  }
}

///////////////////////////////////////////////////////////////////
template <typename T>
void SD_KERNEL __launch_bounds__(256, 2) softMaxDerivForVectorCuda(const void *vx, const LongType *xzShapeInfo, void *vz) {
  // logic of this kernel is based on assumption gridDim = 1

  const auto x = reinterpret_cast<const T *>(vx);
  auto z = reinterpret_cast<T *>(vz);

  __shared__ LongType len;
  __shared__ int numOfIters;
  __shared__ int xzRank;
  __shared__ const LongType *xzShape;
  __shared__ const LongType *xzStride;
  __shared__ T shmem[SD_CUDA_BLOCK_SIZE];

  if (threadIdx.x == 0) {
    len = shape::length(xzShapeInfo);
    numOfIters = (len + blockDim.x - 1) / blockDim.x;  // ceil (len / blockDim.x)

    // Cache rank, shape and stride information
    xzRank = shape::rank(xzShapeInfo);
    xzShape = shape::shapeOf(xzShapeInfo);
    xzStride = shape::stride(xzShapeInfo);
  }
  __syncthreads();

  T temp = -DataTypeUtils::max<T>();

  // ************ evaluate max element in input array x ************ //
  for (int i = 0; i < numOfIters; ++i) {
    const LongType elemIdx = i * blockDim.x + threadIdx.x;
    if (elemIdx < len) {
      LongType offset;
      sd::LongType coords[SD_MAX_RANK];
      INDEX2COORDS(elemIdx, xzRank, xzShape, coords);
      COORDS2INDEX(xzRank, xzStride, coords, offset);
      shmem[threadIdx.x] = (threadIdx.x != 0) ? x[offset] : math::sd_max<T>(x[offset], temp);  // take into account max element evaluated on previous iteration and stored in temp
    } else {
      shmem[threadIdx.x] = -DataTypeUtils::max<T>();
    }

    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s /= 2) {
      if (threadIdx.x < s) shmem[threadIdx.x] = math::sd_max<T>(shmem[threadIdx.x], shmem[threadIdx.x + s]);
      __syncthreads();
    }

    temp = shmem[0];  // save max value calculated at current iteration
  }

  const T max = temp;
  temp = 0;

  // ************ evaluate value of exp(x[offset] - max) per each element, store it to shared memory shmem ************
  // at the same evaluate sum of exponents, sum will be stored in shmem[0]
  for (int i = 0; i < numOfIters; ++i) {
    const LongType elemIdx = i * blockDim.x + threadIdx.x;
    if (elemIdx < len) {
      LongType offset;
      sd::LongType coords[SD_MAX_RANK];
      INDEX2COORDS(elemIdx, xzRank, xzShape, coords);
      COORDS2INDEX(xzRank, xzStride, coords, offset);
      z[offset] = math::sd_exp<T, T>(x[offset] - max);
      shmem[threadIdx.x] = (threadIdx.x != 0) ? z[offset] : (z[offset] + temp);  // take into account sum element evaluated on previous iteration and stored in temp
    } else {
      shmem[threadIdx.x] = 0;
    }

    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s /= 2) {
      if (threadIdx.x < s) shmem[threadIdx.x] += shmem[threadIdx.x + s];
      __syncthreads();
    }

    temp = shmem[0];  // save sum calculated at current iteration
  }

  // ************ evaluate (z[offset] / sum) and derivative z[offset] = z[offset] * (1 - z[offset]) ************ //
  for (int i = 0; i < numOfIters; ++i) {
    const LongType elemIdx = i * blockDim.x + threadIdx.x;
    if (elemIdx >= len) continue;

    LongType offset;
    sd::LongType coords[SD_MAX_RANK];
    INDEX2COORDS(elemIdx, xzRank, xzShape, coords);
    COORDS2INDEX(xzRank, xzStride, coords, offset);
    z[offset] /= shmem[0];
    z[offset] *= (1.f - z[offset]);  // derivative
  }
}
///////////////////////////////////////////////////////////////////
template <typename T>
void softMaxDerivForVectorCudaLauncher(const cudaStream_t *stream, const void *vx, const LongType *xzShapeInfo,
                                       void *vz) {
  dim3 launchDims = getLaunchDims("softmax");

  softMaxDerivForVectorCuda<T><<<launchDims.x,launchDims.y, launchDims.z, *stream>>>(vx, xzShapeInfo, vz);
  sd::DebugHelper::checkGlobalErrorCode("softmax derivative  failed");

}

///////////////////////////////////////////////////////////////////
void softmaxDerivative(LaunchContext *context, NDArray *input, NDArray *output, const int dimension) {
  const int rank = input->rankOf();
  LongType temp;

  if (shape::isCommonVector(input->shapeInfo(), temp)) {
    NDArray::prepareSpecialUse({output}, {input});
    BUILD_SINGLE_SELECTOR(
        input->dataType(), softMaxDerivForVectorCudaLauncher,
        (context->getCudaStream(), input->specialBuffer(), input->specialShapeInfo(), output->specialBuffer()),
        SD_FLOAT_TYPES);
    NDArray::registerSpecialUse({output}, {input});
  } else {
    // All ops below are high-level NDArray operations that manage their own coherence.
    std::vector<LongType> dim = {static_cast<LongType>(dimension)};
    auto maxAlongDim = const_cast<NDArray *>(input)->reduceAlongDimension(reduce::Max, &dim, true);
    auto inputMinusMax = *input - *maxAlongDim;
    inputMinusMax->applyTransform(transform::Exp, output);  // output contains exponents temporarily
    auto sumAlongDim = output->reduceAlongDimension(reduce::Sum, &dim, true);
    *output /= *sumAlongDim;
    auto oneMinusOutput = 1.f - *output;
    *output *= *oneMinusOutput;  // derivative
    delete maxAlongDim;
    delete inputMinusMax;
    delete sumAlongDim;
    delete oneMinusOutput;
  }
}

template <typename T>
void thresholdRelu_(NDArray  *input, double threshold, NDArray *output) {
  auto routine = LAMBDA_T(_x, threshold) { return _x > (T)threshold ? _x : (T)0.f; });
  input->applyLambda(routine, output);
}

void thresholdRelu(LaunchContext *context, NDArray *input, double threshold, NDArray *output) {
  BUILD_SINGLE_SELECTOR(input->dataType(), thresholdRelu_, (input, threshold, output), SD_FLOAT_TYPES);
}

template <typename T>
void thresholdReluDerivative_(NDArray *input, double theta, NDArray *dLdO, NDArray *output) {
  // applyPairwiseLambda is a no-op stub on CUDA (std::function cannot run device-side).
  // scalar::Step(x, theta) → 1 where x > theta, 0 elsewhere; then multiply by dLdO.
  input->applyScalar(scalar::Step, static_cast<T>(theta), output);
  output->applyPairwiseTransform(pairwise::Multiply, dLdO, output);
}

void thresholdReluDerivative(LaunchContext *context, NDArray *input, double threshold, NDArray *dLdO,
                             NDArray *output) {
  // applyPairwiseLambda requires all arrays to share the same type; cast dLdO if needed
  NDArray* dLdOToUse = dLdO;
  NDArray* dLdOCast = nullptr;
  if (dLdO->dataType() != input->dataType()) {
    dLdOCast = dLdO->cast(input->dataType());
    dLdOToUse = dLdOCast;
  }
  BUILD_SINGLE_SELECTOR(input->dataType(), thresholdReluDerivative_, (input, threshold, dLdOToUse, output), SD_FLOAT_TYPES);
  if (dLdOCast != nullptr) delete dLdOCast;
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd

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

#include "execution/cuda/LaunchDims.h"


namespace sd {
namespace ops {
namespace helpers {

///////////////////////////////////////////////////////////////////
template <typename X, typename Y>
void SD_KERNEL preluCuda(const void *vx, const LongType *xShapeInfo, const void *vy, const LongType *yShapeInfo,
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
  PointersManager manager(context, "prelu");

  dim3 launchDims = getLaunchDims("prelu");

  const auto xType = input->dataType();
  const auto yType = alpha->dataType();

  NDArray::prepareSpecialUse({output}, {&input, &alpha});
  BUILD_SINGLE_SELECTOR_TWICE(
      xType, preluCudaLauncher,
      (launchDims.x, launchDims.y, launchDims.z, context->getCudaStream(), input->specialBuffer(),
          input->specialShapeInfo(), alpha->specialBuffer(), alpha->specialShapeInfo(), output->specialBuffer()),
      SD_FLOAT_TYPES);
  NDArray::registerSpecialUse({output}, {&input, &alpha});

  manager.synchronize();
}

///////////////////////////////////////////////////////////////////
template <typename X, typename Y>
void SD_KERNEL preluBPCuda(const void *vIn, const LongType *inShapeInfo, const void *vAlpha,
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
  dLdA->nullify();

  PointersManager manager(context, "preluBP");

  dim3 launchDims = getLaunchDims("prelu");

  const auto xType = input->dataType();
  const auto zType = alpha->dataType();

  NDArray::prepareSpecialUse({dLdI, dLdA}, {input, alpha, dLdO});
  BUILD_SINGLE_SELECTOR_TWICE(
      xType, preluBPCudaLauncher,
      (launchDims.x, launchDims.y, launchDims.z, context->getCudaStream(), input->specialBuffer(),
          input->specialShapeInfo(), alpha->specialBuffer(), alpha->specialShapeInfo(), dLdO->specialBuffer(),
          dLdO->specialShapeInfo(), dLdI->specialBuffer(), dLdI->specialShapeInfo(), dLdA->specialBuffer(),
          dLdA->specialShapeInfo()),
      SD_FLOAT_TYPES);
  NDArray::registerSpecialUse({&dLdI, &dLdA}, {input, alpha, dLdO});

  manager.synchronize();
}

///////////////////////////////////////////////////////////////////
template <typename T>
SD_DEVICE void softMaxForVectorCuda(const void *vx, const LongType *xShapeInfo, void *vz,
                                    const LongType *zShapeInfo) {
  auto inBuff = reinterpret_cast<const T *>(vx);
  auto outBuff = reinterpret_cast<T *>(vz);

  __shared__ T shmemMax;
  __shared__ T shmemSum;
  __shared__ LongType tadLen;
  __shared__ int xRank;
  __shared__ int zRank;
  __shared__ const LongType *xShape;
  __shared__ const LongType *xStride;
  __shared__ const LongType *zShape;
  __shared__ const LongType *zStride;

  if (threadIdx.x == 0) {
    tadLen = shape::length(xShapeInfo);
    shmemMax = -DataTypeUtils::max<T>();
    shmemSum = 0.f;

    // Cache ranks
    xRank = shape::rank(xShapeInfo);
    zRank = shape::rank(zShapeInfo);

    // Cache shapes and strides
    xShape = shape::shapeOf(xShapeInfo);
    xStride = shape::stride(xShapeInfo);
    zShape = shape::shapeOf(zShapeInfo);
    zStride = shape::stride(zShapeInfo);
  }
  __syncthreads();

  T max = -DataTypeUtils::max<T>();
  T sum = 0.f;

  LongType xCoords[SD_MAX_RANK];
  LongType xOffset;

  // Calculate max using cached values
  for (LongType j = 0; j < tadLen; ++j) {
    INDEX2COORDS(j, xRank, xShape, xCoords);
    COORDS2INDEX(xRank, xStride, xCoords, xOffset);
    max = math::sd_max<T>(max, inBuff[xOffset]);
  }

  LongType zCoords[SD_MAX_RANK];
  LongType zOffset;

  // Calculate exp(x - max) and sum using cached values
  for (LongType j = 0; j < tadLen; ++j) {
    INDEX2COORDS(j, xRank, xShape, xCoords);
    COORDS2INDEX(xRank, xStride, xCoords, xOffset);
    T temp = math::sd_exp<T, T>(inBuff[xOffset] - max);
    INDEX2COORDS(j, zRank, zShape, zCoords);
    COORDS2INDEX(zRank, zStride, zCoords, zOffset);
    outBuff[zOffset] = temp;
    sum += temp;
  }

  // Final division step using cached values
  for (LongType j = 0; j < tadLen; ++j) {
    INDEX2COORDS(j, zRank, zShape, zCoords);
    COORDS2INDEX(zRank, zStride, zCoords, zOffset);
    outBuff[zOffset] /= sum;
  }
}

template <typename T>
void SD_KERNEL softMaxForVectorCudaGlobal(const void *vx, const LongType *xShapeInfo, void *vz,
                                          const LongType *zShapeInfo, LongType numOfSubArrs) {
  softMaxForVectorCuda<T>(vx, xShapeInfo, vz, zShapeInfo);
}

///////////////////////////////////////////////////////////////////
template <typename T>
void softMaxForVectorCudaLauncher(const cudaStream_t *stream, const void *vx, const LongType *xShapeInfo, void *vz,
                                  const LongType *zShapeInfo, LongType numTads) {

  softMaxForVectorCudaGlobal<T><<<1, SD_CUDA_BLOCK_SIZE, 1024, *stream>>>(vx, xShapeInfo, vz, zShapeInfo, numTads);
  sd::DebugHelper::checkGlobalErrorCode("softmax  failed");

}

///////////////////////////////////////////////////////////////////

template <typename T>
SD_KERNEL void softmaxEws1Kernel(const T *input, const LongType *inputOffsets, T *output,
                                 const LongType *outputOffsets,
                                 LongType numOfSubArrs, LongType tadLen) {
  int i = blockIdx.x;  // Each block handles one TAD

  if (i >= numOfSubArrs) return;  // Out-of-bounds check for TADs

  auto inBuff = input + inputOffsets[i];
  auto outBuff = output + outputOffsets[i];

  __shared__ T shmemMax;
  __shared__ T shmemSum;

  if (threadIdx.x == 0) {
    shmemMax = -DataTypeUtils::max<T>();
    shmemSum = 0.f;
  }
  __syncthreads();


  // Calculate max
  for (LongType j = threadIdx.x; j < tadLen; j+= gridDim.x) {
    math::atomics::sd_atomicMax(&shmemMax, inBuff[j]);
  }
  __syncthreads();

  // Calculate exp(x - max) and sum
  for (LongType j = threadIdx.x; j < tadLen; j += gridDim.x) {
    T temp = math::sd_exp<T, T>(inBuff[j] - shmemMax);
    outBuff[j] = temp;
    math::atomics::sd_atomicAdd(&shmemSum, temp);
  }
  __syncthreads();

  // Final division step
  for (LongType j = threadIdx.x; j < tadLen; j += blockDim.x) {
    outBuff[j] /= shmemSum;
  }


}
template <typename T>
SD_KERNEL static void softMaxCuda(const void *vx, const LongType *xTadShapeInfo, const LongType *xOffsets,
                                  void *vz, const LongType *zTadShapeInfo, const LongType *zOffsets, LongType numTads) {
  int i = blockIdx.x;
  if(i >= numTads) return;

  const auto x = reinterpret_cast<const T *>(vx);
  auto z = reinterpret_cast<T *>(vz);

  const auto *xTad = x + xOffsets[blockIdx.x];
  auto *zTad = z + zOffsets[blockIdx.x];
  softMaxForVectorCuda<T>(xTad, xTadShapeInfo, zTad, zTadShapeInfo);
}

///////////////////////////////////////////////////////////////////

template <typename T>
static void softMaxEws1CudaLauncher(const int blocksPerGrid,
                                    const int threadsPerBlock,
                                    const int sharedMem,
                                    const cudaStream_t *stream,
                                    const void *vx, const LongType *xOffsets, void *vz,
                                    const LongType *zOffsets, LongType numTads, LongType tadLength) {



  auto reCastInputs = reinterpret_cast<const T *>(vx);
  auto reCastOutputs = reinterpret_cast<T *>(vz);
  softmaxEws1Kernel<T>
  <<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(reCastInputs,
                                                           xOffsets,
                                                           reCastOutputs,
                                                           zOffsets,
                                                           numTads,
                                                           tadLength);
  sd::DebugHelper::checkGlobalErrorCode("softmaxews  failed");

}

template <typename T>
static void softMaxCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const int sharedMem,
                                const cudaStream_t *stream, const void *vx, const LongType *xTadShapeInfo,
                                const LongType *xOffsets, void *vz, const LongType *zTadShapeInfo,
                                const LongType *zOffsets, LongType numTads) {


  softMaxCuda<T><<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(vx, xTadShapeInfo, xOffsets, vz, zTadShapeInfo,
                                                                         zOffsets ,numTads);
  sd::DebugHelper::checkGlobalErrorCode("softmax  failed");

}

//////////////////////////////////////////////////////////////////////////
void softmax(LaunchContext *context, NDArray *input, NDArray *output, const int dimension) {
  const int rank = input->rankOf();

  PointersManager manager(context, "helpers::softmax");

  if (input->isVector()) {
    if (rank == 1 || input->sizeAt(dimension) != 1) {
      NDArray::prepareSpecialUse({output}, {input});
      BUILD_SINGLE_SELECTOR(input->dataType(), softMaxForVectorCudaLauncher,
                            (context->getCudaStream(), input->specialBuffer(), input->specialShapeInfo(),
                                output->specialBuffer(), output->specialShapeInfo(),1),
                            SD_FLOAT_TYPES);
      NDArray::registerSpecialUse({output}, {input});
    } else
      *output = 1.;
  } else {
    auto packX = ConstantTadHelper::getInstance().tadForDimensions(input->shapeInfo(), {dimension});
    auto packZ = ConstantTadHelper::getInstance().tadForDimensions(output->shapeInfo(), {dimension});

    dim3 softmaxDims = getSoftmaxDims(packZ->numberOfTads());


    NDArray::prepareSpecialUse({output}, {input});
    BUILD_SINGLE_SELECTOR(input->dataType(), softMaxCudaLauncher,
                          (softmaxDims.x, softmaxDims.y,
                              softmaxDims.z,
                              context->getCudaStream(),
                              input->specialBuffer(),
                              packX->specialShapeInfo(),
                              packX->specialOffsets(), output->specialBuffer(),
                              packZ->specialShapeInfo(),
                              packZ->specialOffsets(),packX->numberOfTads()),
                          SD_FLOAT_TYPES);
    NDArray::registerSpecialUse({output}, {input});

  }

  manager.synchronize();

  output->tickWriteDevice();
}

///////////////////////////////////////////////////////////////////
template <typename T>
void SD_KERNEL logSoftMaxForVectorCuda(const void *vx, const LongType *xzShapeInfo, void *vz) {
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

  T temp = -DataTypeUtils::max<T>();  // set start value to compare with at first iteration, FIXME: what if T is unsigned ??

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
      shmem[threadIdx.x] = -DataTypeUtils::max<T>();  // FIXME: what if T is unsigned ??
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
  if (!input->isActualOnDeviceSide()) input->syncToDevice();
  const int rank = input->rankOf();

  if (input->isVector()) {
    if (rank == 1 || input->sizeAt(dimension) != 1) {
      BUILD_SINGLE_SELECTOR(
          input->dataType(), logSoftMaxForVectorCudaLauncher,
          (context->getCudaStream(), input->specialBuffer(), input->specialShapeInfo(), output->specialBuffer()),
          SD_FLOAT_TYPES);
      input->tickReadDevice();
    } else
      *output = 0.;
  } else {
    std::vector<LongType> dim = {static_cast<LongType>(dimension)};
    auto maxAlongDim = const_cast<NDArray *>(input)->reduceAlongDimension(reduce::Max, &dim, true);
    auto inputMinusMax = *input - maxAlongDim;
    inputMinusMax.applyTransform(transform::Exp, output);  // output contains exponents temporarily
    auto sumAlongDim = output->reduceAlongDimension(reduce::Sum, &dim, true);
    *output /= sumAlongDim;
    output->applyTransform(transform::Log, output);
    input->tickReadDevice();
  }

  PointersManager manager(context, "helpers::logSoftmax");
  manager.synchronize();

  output->tickWriteDevice();
}

///////////////////////////////////////////////////////////////////
template <typename T>
void SD_KERNEL softMaxDerivForVectorCuda(const void *vx, const LongType *xzShapeInfo, void *vz) {
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

  T temp = -DataTypeUtils::max<T>();  // set start value to compare with at first iteration, FIXME: what if T is unsigned ??

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
      shmem[threadIdx.x] = -DataTypeUtils::max<T>();  // FIXME: what if T is unsigned ??
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
  if (!input->isActualOnDeviceSide()) input->syncToDevice();
  const int rank = input->rankOf();
  LongType temp;

  if (shape::isCommonVector(input->shapeInfo(), temp)) {
    BUILD_SINGLE_SELECTOR(
        input->dataType(), softMaxDerivForVectorCudaLauncher,
        (context->getCudaStream(), input->specialBuffer(), input->specialShapeInfo(), output->specialBuffer()),
        SD_FLOAT_TYPES);
    input->tickReadDevice();
  } else {
    std::vector<LongType> dim = {static_cast<LongType>(dimension)};
    auto maxAlongDim = const_cast<NDArray *>(input)->reduceAlongDimension(reduce::Max, &dim, true);
    auto inputMinusMax = *input - maxAlongDim;
    inputMinusMax.applyTransform(transform::Exp, output);  // output contains exponents temporarily
    auto sumAlongDim = output->reduceAlongDimension(reduce::Sum, &dim, true);
    *output /= sumAlongDim;
    *output *= (1.f - *output);  // derivative
    input->tickReadDevice();
  }

  PointersManager manager(context, "helpers::softmaxDerivative");
  manager.synchronize();

  output->tickWriteDevice();
}

template <typename T>
void thresholdRelu_(NDArray const *input, double threshold, NDArray *output) {
  auto routine = LAMBDA_T(_x, threshold) { return _x > (T)threshold ? _x : (T)0.f; };
  const_cast<NDArray *>(input)->applyLambda(routine, output);
}

void thresholdRelu(LaunchContext *context, NDArray *input, double threshold, NDArray *output) {
  BUILD_SINGLE_SELECTOR(input->dataType(), thresholdRelu_, (input, threshold, output), SD_FLOAT_TYPES);
}

template <typename T>
void thresholdReluDerivative_(NDArray *input, double theta, NDArray *dLdO, NDArray *output) {
  auto derivative = LAMBDA_TT(_x, grO, theta) {
    if (_x > theta)
      return grO;
    else
      return static_cast<T>(0);
  };

  input->applyPairwiseLambda(dLdO, derivative, output);
}

void thresholdReluDerivative(LaunchContext *context, NDArray *input, double threshold, NDArray *dLdO,
                             NDArray *output) {
  BUILD_SINGLE_SELECTOR(input->dataType(), thresholdReluDerivative_, (input, threshold, dLdO, output), SD_FLOAT_TYPES);
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd

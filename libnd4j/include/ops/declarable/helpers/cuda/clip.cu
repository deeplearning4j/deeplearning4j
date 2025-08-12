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
// @author Yurii Shyrma (iuriish@yahoo.com)
// @author sgazeos@gmail.com
// @author raver119@gmail.com
//

#include <helpers/ConstantTadHelper.h>
#include <helpers/PointersManager.h>
#include <helpers/ShapeUtils.h>
#include <ops/declarable/helpers/transforms.h>

#include "execution/cuda/LaunchDims.h"


namespace sd {
namespace ops {
namespace helpers {

//////////////////////////////////////////////////////////////////////////
template <typename T>
SD_KERNEL static void clipByNormCuda(const void* vClipNorm, const void* vNorm, const LongType* normShapeInfo,
                                     void* vz, const LongType* zShapeInfo, const LongType* dimensions,
                                     const LongType dimsLen,
                                     const bool useAverage) {
  const T clipNorm = *reinterpret_cast<const T*>(vClipNorm);
  const T* norm = reinterpret_cast<const T*>(vNorm);
  T* z = reinterpret_cast<T*>(vz);

  __shared__ LongType zLen, tadLen, totalThreads;
  __shared__ int zRank, normRank;
  __shared__ const LongType *zShape;
  __shared__ const LongType *zStride;
  __shared__ const LongType *normStride;

  if (threadIdx.x == 0) {
    zLen = shape::length(zShapeInfo);
    tadLen = zLen / shape::length(normShapeInfo);
    totalThreads = gridDim.x * blockDim.x;

    // Cache ranks
    zRank = shape::rank(zShapeInfo);
    normRank = shape::rank(normShapeInfo);

    // Cache shapes and strides
    zShape = shape::shapeOf(zShapeInfo);
    zStride = shape::stride(zShapeInfo);
    normStride = shape::stride(normShapeInfo);
  }

  __syncthreads();

  LongType zCoords[SD_MAX_RANK], normCoords[SD_MAX_RANK];

  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

  for (LongType i = tid; i < zLen; i += totalThreads) {
    INDEX2COORDS(i, zRank, zShape, zCoords);

    // deduce norm coords
    for (int j = 0; j < dimsLen; ++j) normCoords[j] = zCoords[dimensions[j]];

    LongType normOffset, zOffset;
    COORDS2INDEX(normRank, normStride, normCoords, normOffset);
    COORDS2INDEX(zRank, zStride, zCoords, zOffset);

    const T actualNorm = useAverage ? static_cast<T>(norm[normOffset]) / static_cast<T>(tadLen) : static_cast<T>(norm[normOffset]);

    if (actualNorm > clipNorm) z[zOffset] *= static_cast<T>(clipNorm) / static_cast<T>(actualNorm);
  }
}
//////////////////////////////////////////////////////////////////////////
template <typename T>
SD_HOST static void clipByNormCudaLauncher(const int blocksPerGrid, const int threadsPerBlock,
                                           const cudaStream_t* stream, const void* vClipNorm, const void* vNorm,
                                           const LongType* normShapeInfo, void* vz, const LongType* zShapeInfo,
                                           const LongType* dimensions, const LongType dimsLen, const bool useAverage) {
  clipByNormCuda<T><<<blocksPerGrid, threadsPerBlock, 512, *stream>>>(vClipNorm, vNorm, normShapeInfo, vz, zShapeInfo,
                                                                      dimensions, dimsLen, useAverage);
  sd::DebugHelper::checkGlobalErrorCode("clipByNorm  failed");

}

//////////////////////////////////////////////////////////////////////////
void clipByNorm(LaunchContext* context, NDArray* input, NDArray* output, const std::vector<LongType>& dims,
                NDArray* clipNorm, const bool isInplace, const bool useAverage) {
  NDArray* z = nullptr;

  if (isInplace) {
    z = input;
  } else {
    output->assign(input);
    z = output;
  }

  if (dims.empty()) {
    std::vector<LongType> empty;
    NDArray actualNorm = useAverage ? z->reduceAlongDimension(reduce::Norm2, &empty) / z->lengthOf()
                                    : z->reduceAlongDimension(reduce::Norm2, &empty);

    if (actualNorm.e<float>(0) > clipNorm->e<float>(0)) *z *= *clipNorm / actualNorm;
  } else {
    NDArray actualNorms = z->reduceAlongDimension(reduce::Norm2, &dims);

    std::vector<LongType> *dimsToExclude = ShapeUtils::evalDimsToExclude(z->rankOf(), dims.size(),dims.data());

    const int threadsPerBlock = SD_MAX_NUM_THREADS / 2;
    const int blocksPerGrid = (z->lengthOf() + threadsPerBlock - 1) / threadsPerBlock;

    PointersManager manager(context, "clipByNorm");

    const LongType* dimensions = reinterpret_cast<const LongType*>(
        manager.replicatePointer(dimsToExclude->data(), dimsToExclude->size() * sizeof(LongType)));

    NDArray::prepareSpecialUse({z}, {z, &actualNorms, clipNorm});


    BUILD_SINGLE_SELECTOR(z->dataType(), clipByNormCudaLauncher,
                          (blocksPerGrid,
                              threadsPerBlock,
                              context->getCudaStream(),
                              clipNorm->specialBuffer(),
                              actualNorms.specialBuffer(),
                              actualNorms.specialShapeInfo(),
                              z->specialBuffer(),
                              z->specialShapeInfo(),
                              dimensions,
                              dimsToExclude->size(),
                              useAverage),
                          SD_FLOAT_TYPES);
    NDArray::registerSpecialUse({z}, {z, &actualNorms, clipNorm});

    manager.synchronize();
    delete dimsToExclude;
  }
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
SD_KERNEL static void clipByNormBpCuda(const void* vClipNorm, const void* vx, const LongType* xShapeInfo,  // input
                                       const void* vy, const LongType* yShapeInfo,                         // gradO
                                       const void* vNorm, const LongType* normShapeInfo, const void* vSum,
                                       const LongType* sumShapeInfo, void* vz,
                                       const LongType* zShapeInfo,  // gradI
                                       const LongType* dimensions, const LongType dimsLen, const bool useAverage) {
  const T clipNorm = *reinterpret_cast<const T*>(vClipNorm);
  const T* norm = reinterpret_cast<const T*>(vNorm);
  const T* sum = reinterpret_cast<const T*>(vSum);
  const T* x = reinterpret_cast<const T*>(vx);
  const T* y = reinterpret_cast<const T*>(vy);
  T* z = reinterpret_cast<T*>(vz);

  __shared__ LongType zLen, tadLen, totalThreads;
  __shared__ bool sameOffsets;
  __shared__ int zRank, yRank, normRank, sumRank, xRank;
  __shared__ const LongType *zShape;
  __shared__ const LongType *zStride;
  __shared__ const LongType *yStride;
  __shared__ const LongType *normStride;
  __shared__ const LongType *sumStride;
  __shared__ const LongType *xStride;

  if (threadIdx.x == 0) {
    zLen = shape::length(zShapeInfo);
    tadLen = zLen / shape::length(normShapeInfo);
    totalThreads = gridDim.x * blockDim.x;

    sameOffsets = shape::haveSameShapeAndStrides(xShapeInfo, yShapeInfo, zShapeInfo);

    // Cache ranks
    zRank = shape::rank(zShapeInfo);
    yRank = shape::rank(yShapeInfo);
    normRank = shape::rank(normShapeInfo);
    sumRank = shape::rank(sumShapeInfo);
    xRank = shape::rank(xShapeInfo);

    // Cache shapes and strides
    zShape = shape::shapeOf(zShapeInfo);
    zStride = shape::stride(zShapeInfo);
    yStride = shape::stride(yShapeInfo);
    normStride = shape::stride(normShapeInfo);
    sumStride = shape::stride(sumShapeInfo);
    xStride = shape::stride(xShapeInfo);
  }

  __syncthreads();

  LongType zCoords[SD_MAX_RANK], normCoords[SD_MAX_RANK];

  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

  for (LongType i = tid; i < zLen; i += totalThreads) {
    INDEX2COORDS(i, zRank, zShape, zCoords);

    LongType zOffset, yOffset;
    COORDS2INDEX(zRank, zStride, zCoords, zOffset);
    if(sameOffsets) {
      yOffset = zOffset;
    } else {
      COORDS2INDEX(yRank, yStride, zCoords, yOffset);
    }

    // deduce norm coords
    for (int j = 0; j < dimsLen; ++j) normCoords[j] = zCoords[dimensions[j]];

    LongType normOffset;
    COORDS2INDEX(normRank, normStride, normCoords, normOffset);

    const T actualNorm = useAverage ? norm[normOffset] / tadLen : norm[normOffset];

    if (actualNorm > clipNorm) {
      LongType sumOffset, xOffset;
      COORDS2INDEX(sumRank, sumStride, normCoords, sumOffset);
      if(sameOffsets) {
        xOffset = zOffset;
      } else {
        COORDS2INDEX(xRank, xStride, zCoords, xOffset);
      }

      const T sumVal = sum[sumOffset];
      z[zOffset] = (clipNorm / actualNorm) * y[yOffset] *
                   (static_cast<T>(1.f) - (x[xOffset] * sumVal) / (actualNorm * actualNorm));
    } else {
      z[zOffset] = y[yOffset];
    }
  }
}
//////////////////////////////////////////////////////////////////////////
template <typename T>
void clipByNormBp_(LaunchContext* context, NDArray* input, NDArray* gradO, NDArray* gradI,
                   const std::vector<LongType>& dims, NDArray* clipNorm, const bool useAverage) {
  const int rank = input->rankOf();

  auto actualNorms = input->reduceAlongDimension(reduce::Norm2, &dims);

  if (actualNorms.lengthOf() == 1) {
    const T norm = useAverage ? actualNorms.e<T>(0) / static_cast<T>(input->lengthOf()) : actualNorms.e<T>(0);

    auto clipVal = clipNorm->e<T>(0);

    if (norm > clipVal) {
      const T sum = input->reduceNumber(reduce::Sum).e<T>(0);  // reduce to scalar
      const T factor1 = clipVal / norm;
      const T factor2 = static_cast<T>(1.f) / (norm * norm);  // 1 / (norm*norm*norm)

      auto lambda = LAMBDA_TT(x, y, sum, factor1, factor2) {
        return factor1 * y * (static_cast<T>(1.f) - factor2 * x * sum);
      });

      input->applyPairwiseLambda(gradO, lambda, gradI);
      const_cast<NDArray*>(input)->applyPairwiseLambda(const_cast<NDArray*>(gradO), lambda, gradI);
    } else
      gradI->assign(gradO);
  } else {
    NDArray actualNorms = input->reduceAlongDimension(reduce::Norm2, &dims);
    NDArray sums = input->reduceAlongDimension(reduce::Sum, &dims);

    std::vector<LongType> *dimsToExclude = ShapeUtils::evalDimsToExclude(gradI->rankOf(), dims.size(),dims.data());


    dim3 launchDims = clipDims(gradI->lengthOf());
    PointersManager manager(context, "clipByNormBp");

    const LongType* dimensions = reinterpret_cast<const LongType*>(
        manager.replicatePointer(dimsToExclude->data(), dimsToExclude->size() * sizeof(LongType)));

    NDArray::prepareSpecialUse({gradI}, {&actualNorms, &sums, clipNorm, input, gradO});
    clipByNormBpCuda<T><<<launchDims.y, launchDims.x,launchDims.z, *context->getCudaStream()>>>(
        clipNorm->specialBuffer(), input->specialBuffer(), input->specialShapeInfo(), gradO->specialBuffer(),
        gradO->specialShapeInfo(), actualNorms.specialBuffer(), actualNorms.specialShapeInfo(), sums.specialBuffer(),
        sums.specialShapeInfo(), gradI->specialBuffer(), gradI->specialShapeInfo(), dimensions, (LongType)dimsToExclude->size(),
        useAverage);
    sd::DebugHelper::checkGlobalErrorCode("clipByNorm  failed");

    NDArray::registerSpecialUse({gradI}, {&actualNorms, &sums, clipNorm, input, gradO});

    manager.synchronize();
    delete dimsToExclude;
  }
}
BUILD_SINGLE_TEMPLATE( void clipByNormBp_,
                      (sd::LaunchContext * context, NDArray* input, NDArray* gradO, NDArray* gradI,
                          const std::vector<sd::LongType>& dimensions, NDArray* clipNorm, const bool useAverage),
                      SD_FLOAT_TYPES);

//////////////////////////////////////////////////////////////////////////
void clipByNormBp(LaunchContext* context, NDArray* input, NDArray* gradO, NDArray* gradI,
                  const std::vector<LongType>& dimensions, NDArray* clipNorm, const bool useAverage) {
  NDArray casted = clipNorm->cast(input->dataType());
  BUILD_SINGLE_SELECTOR(gradI->dataType(), clipByNormBp_,
                        (context, &casted, gradO, gradI, dimensions, clipNorm, useAverage), SD_FLOAT_TYPES);
}

template <typename T>
void clipByGlobalNorm_(LaunchContext* context, std::vector<NDArray*>& inputs, double clipNorm,
                       memory::Workspace* workspace, std::vector<NDArray*>& outputs, bool isInplace) {
  T globalNorm = static_cast<T>(0.f);

  for (auto i = 0; i < inputs.size(); i++) {
    auto input = inputs[i];
    auto l2norm = input->reduceNumber(reduce::Norm2);
    globalNorm += l2norm.e<T>(0) * l2norm.e<T>(0);
  }

  globalNorm = math::sd_sqrt<T,T>(globalNorm);
  outputs[inputs.size()]->p(0, globalNorm);
  const T factor = static_cast<T>(clipNorm) / globalNorm;

  for (size_t e = 0; e < inputs.size(); e++) {
    // all-reduce
    auto input = inputs[e];
    auto output = outputs[e];

    if (static_cast<double>(globalNorm) <= clipNorm) {
      output->assign(input);
    } else {
      auto lambda = LAMBDA_T(_x, factor) { return _x * factor; });
      input->applyLambda(lambda, output);
    }
  }
}

void clipByGlobalNorm(LaunchContext* context, std::vector<NDArray*>& inputs, double clipNorm,
                      memory::Workspace* workspace, std::vector<NDArray*>& outputs, bool isInplace) {
  BUILD_SINGLE_SELECTOR(outputs[0]->dataType(), clipByGlobalNorm_,
                        (context, inputs, clipNorm, workspace, outputs, isInplace), SD_FLOAT_TYPES);
}

BUILD_SINGLE_TEMPLATE( void clipByGlobalNorm_,
                      (sd::LaunchContext * context, std::vector<NDArray*> & inputs, double clipNorm,
                          sd::memory::Workspace* workspace, std::vector<NDArray*>& outputs, bool isInplace),
                      SD_FLOAT_TYPES);

template <typename T>
static void SD_KERNEL clipByValueKernel(void* input, const LongType* inputShape, void* output,
                                        const LongType* outputShape, double leftBound, double rightBound) {
  __shared__ T* outputBuf;
  __shared__ T* inputBuf;
  __shared__ LongType length;
  __shared__ LongType inputRank;
  __shared__ LongType outputRank;
  __shared__ LongType* inputShapePtr;
  __shared__ LongType* outputShapePtr;
  __shared__ LongType* inputStridePtr;
  __shared__ LongType* outputStridePtr;

  if (threadIdx.x == 0) {
    outputBuf = reinterpret_cast<T*>(output);
    inputBuf = reinterpret_cast<T*>(input);
    length = shape::length(inputShape);

    // Cache shape information
    inputRank = shape::rank(inputShape);
    outputRank = shape::rank(outputShape);
    inputShapePtr = shape::shapeOf(inputShape);
    outputShapePtr = shape::shapeOf(outputShape);
    inputStridePtr = shape::stride(inputShape);
    outputStridePtr = shape::stride(outputShape);
  }
  __syncthreads();

  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  const auto step = gridDim.x * blockDim.x;

  for (LongType e = tid; e < length; e += step) {
    LongType inputCoords[SD_MAX_RANK];
    LongType outputCoords[SD_MAX_RANK];
    LongType inputOffset;
    LongType outputOffset;

    INDEX2COORDS(e, inputRank, inputShapePtr, inputCoords);
    COORDS2INDEX(inputRank, inputStridePtr, inputCoords, inputOffset);
    INDEX2COORDS(e, outputRank, outputShapePtr, outputCoords);
    COORDS2INDEX(outputRank, outputStridePtr, outputCoords, outputOffset);

    if (inputBuf[inputOffset] > rightBound)
      outputBuf[outputOffset] = (T)rightBound;
    else if (inputBuf[inputOffset] < leftBound)
      outputBuf[outputOffset] = (T)leftBound;
    else
      outputBuf[outputOffset] = inputBuf[inputOffset];
  }
}

template <typename T>
static void clipByValue_(LaunchContext* context, NDArray* input, double leftBound, double rightBound,
                         NDArray* output) {
  auto stream = context->getCudaStream();
  if (!input->isActualOnDeviceSide()) input->syncToDevice();
  NDArray::prepareSpecialUse({output}, {input});
  dim3 launchDims = getLaunchDims("clip");
  clipByValueKernel<T><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(input->specialBuffer(), input->specialShapeInfo(),
                                                                              output->specialBuffer(), output->specialShapeInfo(), leftBound,
                                                                              rightBound);
  sd::DebugHelper::checkGlobalErrorCode("clipByValue failed");

  NDArray::registerSpecialUse({output}, {input});
}

void clipByValue(LaunchContext* context, NDArray* input, double leftBound, double rightBound, NDArray* output) {
  BUILD_SINGLE_SELECTOR(input->dataType(), clipByValue_, (context, input, leftBound, rightBound, output),
                        SD_FLOAT_TYPES);
}

BUILD_SINGLE_TEMPLATE( void clipByValue_, (sd::LaunchContext * context, NDArray* input, double leftBound,
    double rightBound, NDArray* output);
, SD_FLOAT_TYPES);

}  // namespace helpers
}  // namespace ops
}  // namespace sd

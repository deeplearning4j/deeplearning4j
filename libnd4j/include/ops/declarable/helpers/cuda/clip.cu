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
  if (z->lengthOf() == 0) return;

  // The norm of every tensor along dims and its comparison with the clip value stay on the device; with no dims (or
  // dims covering every axis) the whole array is one tensor, whose norm is a scalar the kernel reads at offset 0. A
  // host read of the norm forced a synchronization and could not be captured in a CUDA graph. The kernel reads the
  // clip value as z's type: clipbyavgnorm's from a floating point argument is DOUBLE, and an input clip value may be
  // any type.
  NDArray* clipCast = clipNorm->dataType() == z->dataType() ? nullptr : clipNorm->cast(z->dataType());
  NDArray* clip = clipCast != nullptr ? clipCast : clipNorm;
  NDArray* actualNorms = z->reduceAlongDimension(reduce::Norm2, &dims);

  std::vector<LongType>* dimsToExclude = ShapeUtils::evalDimsToExclude(z->rankOf(), dims.size(), dims.data());

  // clipDims gives (blocks, threads, shared memory)
  dim3 launchDims = clipDims(z->lengthOf());
  PointersManager manager(context, "clipByNorm");

  const LongType* dimensions = reinterpret_cast<const LongType*>(
      manager.replicatePointer(dimsToExclude->data(), dimsToExclude->size() * sizeof(LongType)));

  NDArray::prepareSpecialUse({z}, {z, actualNorms, clip});

  BUILD_SINGLE_SELECTOR(z->dataType(), clipByNormCudaLauncher,
                        (launchDims.x, launchDims.y, context->getCudaStream(), clip->specialBuffer(),
                         actualNorms->specialBuffer(), actualNorms->specialShapeInfo(), z->specialBuffer(),
                         z->specialShapeInfo(), dimensions, dimsToExclude->size(), useAverage),
                        SD_FLOAT_TYPES);
  NDArray::registerSpecialUse({z}, {z, actualNorms, clip});

  manager.synchronize();
  delete dimsToExclude;
  delete actualNorms;
  delete clipCast;
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

    const T plainNorm = norm[normOffset];
    const T actualNorm = useAverage ? plainNorm / tadLen : plainNorm;

    if (actualNorm > clipNorm) {
      LongType sumOffset, xOffset;
      COORDS2INDEX(sumRank, sumStride, normCoords, sumOffset);
      if(sameOffsets) {
        xOffset = zOffset;
      } else {
        COORDS2INDEX(xRank, xStride, zCoords, xOffset);
      }

      // dL/dx = (clip / a) * (gradO - x * dot(gradO, x) / |x|^2), the dot product taken over the whole tensor and
      // a the norm compared with clip (|x|, or |x| / n for the average norm). `sum` holds that dot product per tensor.
      const T dotVal = sum[sumOffset];
      z[zOffset] = (clipNorm / actualNorm) * (y[yOffset] - (x[xOffset] * dotVal) / (plainNorm * plainNorm));
    } else {
      z[zOffset] = y[yOffset];
    }
  }
}
//////////////////////////////////////////////////////////////////////////
template <typename T>
void clipByNormBp_(LaunchContext* context, NDArray* input, NDArray* gradO, NDArray* gradI,
                   const std::vector<LongType>& dims, NDArray* clipNorm, const bool useAverage) {
  if (gradI->lengthOf() == 0) return;
  // The norm and dot(gradO, input) of every tensor, the comparison with clipNorm and the gradient all stay on the
  // device; the whole array is one tensor when the dimensions cover every axis (its norm and dot product are scalars,
  // which the kernel reads at offset 0). A host read of the norm here forced a synchronization and could not be
  // captured in a CUDA graph.
  NDArray* norms = input->reduceAlongDimension(reduce::Norm2, &dims);
  // dot(gradO, input) of every tensor: the gradient's component along the input. A sum of the input alone only
  // equals it where gradO is the same everywhere in the tensor.
  NDArray* weighted = (*input) * (*gradO);
  NDArray* sums = weighted->reduceAlongDimension(reduce::Sum, &dims);

  std::vector<LongType>* dimsToExclude = ShapeUtils::evalDimsToExclude(gradI->rankOf(), dims.size(), dims.data());

  // clipDims gives (blocks, threads, shared memory); launching it as (threads, blocks) ran ceil(length / 512) threads
  // per block, past the 1024 limit for arrays of more than 512 * 1024 elements.
  dim3 launchDims = clipDims(gradI->lengthOf());
  PointersManager manager(context, "clipByNormBp");

  const LongType* dimensions = reinterpret_cast<const LongType*>(
      manager.replicatePointer(dimsToExclude->data(), dimsToExclude->size() * sizeof(LongType)));

  NDArray::prepareSpecialUse({gradI}, {norms, sums, clipNorm, input, gradO});
  clipByNormBpCuda<T><<<launchDims.x, launchDims.y, launchDims.z, *context->getCudaStream()>>>(
      clipNorm->specialBuffer(), input->specialBuffer(), input->specialShapeInfo(), gradO->specialBuffer(),
      gradO->specialShapeInfo(), norms->specialBuffer(), norms->specialShapeInfo(), sums->specialBuffer(),
      sums->specialShapeInfo(), gradI->specialBuffer(), gradI->specialShapeInfo(), dimensions,
      (LongType)dimsToExclude->size(), useAverage);
  sd::DebugHelper::checkGlobalErrorCode("clipByNorm  failed");

  NDArray::registerSpecialUse({gradI}, {norms, sums, clipNorm, input, gradO});

  manager.synchronize();
  delete dimsToExclude;
  delete norms;
  delete weighted;
  delete sums;
}
BUILD_SINGLE_TEMPLATE( void clipByNormBp_,
                      (sd::LaunchContext * context, NDArray* input, NDArray* gradO, NDArray* gradI,
                          const std::vector<sd::LongType>& dimensions, NDArray* clipNorm, const bool useAverage),
                      SD_FLOAT_TYPES);

//////////////////////////////////////////////////////////////////////////
void clipByNormBp(LaunchContext* context, NDArray* input, NDArray* gradO, NDArray* gradI,
                  const std::vector<LongType>& dimensions, NDArray* clipNorm, const bool useAverage) {
  NDArray* casted = clipNorm->cast(input->dataType());
  BUILD_SINGLE_SELECTOR(gradI->dataType(), clipByNormBp_,
                        (context, input, gradO, gradI, dimensions, casted, useAverage), SD_FLOAT_TYPES);
  delete casted;
}

template <typename T>
void clipByGlobalNorm_(LaunchContext* context, std::vector<NDArray*>& inputs, double clipNorm,
                       memory::Workspace* workspace, std::vector<NDArray*>& outputs, bool isInplace) {
  T globalNorm = static_cast<T>(0.f);

  for (auto i = 0; i < inputs.size(); i++) {
    auto input = inputs[i];
    auto l2norm = input->reduceNumber(reduce::Norm2);
    T normVal = l2norm->e<T>(0);
    globalNorm += normVal * normVal;
    delete l2norm;
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
      // applyLambda is a no-op CUDA stub; use native scalar multiply instead
      output->assign(input);
      output->applyScalar(scalar::Multiply, factor, output);
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
                        SD_COMMON_TYPES);
}

BUILD_SINGLE_TEMPLATE( void clipByValue_, (sd::LaunchContext * context, NDArray* input, double leftBound,
    double rightBound, NDArray* output);
, SD_COMMON_TYPES);

}  // namespace helpers
}  // namespace ops
}  // namespace sd

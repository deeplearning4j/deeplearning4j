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
SD_KERNEL static void clipByNormCuda(const void* vClipNorm, const void* vNorm, const sd::LongType* normShapeInfo,
                                     void* vz, const sd::LongType* zShapeInfo, const LongType* dimensions,
                                     const LongType dimsLen,
                                     const bool useAverage) {
  const T clipNorm = *reinterpret_cast<const T*>(vClipNorm);
  const T* norm = reinterpret_cast<const T*>(vNorm);
  T* z = reinterpret_cast<T*>(vz);

  __shared__ sd::LongType zLen, tadLen, totalThreads;

  if (threadIdx.x == 0) {
    zLen = shape::length(zShapeInfo);
    tadLen = zLen / shape::length(normShapeInfo);
    totalThreads = gridDim.x * blockDim.x;
  }

  __syncthreads();

  sd::LongType zCoords[SD_MAX_RANK], normCoords[SD_MAX_RANK];

  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

  for (sd::LongType i = tid; i < zLen; i += totalThreads) {
    shape::index2coords(i, zShapeInfo, zCoords);

    // deduce norm coords
    for (int j = 0; j < dimsLen; ++j) normCoords[j] = zCoords[dimensions[j]];

    const T actualNorm = useAverage ? norm[shape::getOffset(normShapeInfo, normCoords)] / tadLen
                                    : norm[shape::getOffset(normShapeInfo, normCoords)];

    if (actualNorm > clipNorm) z[shape::getOffset(zShapeInfo, zCoords)] *= clipNorm / actualNorm;
  }
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
SD_HOST static void clipByNormCudaLauncher(const int blocksPerGrid, const int threadsPerBlock,
                                           const cudaStream_t* stream, const void* vClipNorm, const void* vNorm,
                                           const sd::LongType* normShapeInfo, void* vz, const sd::LongType* zShapeInfo,
                                           const LongType* dimensions, const LongType dimsLen, const bool useAverage) {
  clipByNormCuda<T><<<blocksPerGrid, threadsPerBlock, 512, *stream>>>(vClipNorm, vNorm, normShapeInfo, vz, zShapeInfo,
                                                                      dimensions, dimsLen, useAverage);
}

//////////////////////////////////////////////////////////////////////////
void clipByNorm(sd::LaunchContext* context, NDArray& input, NDArray& output, const std::vector<sd::LongType>& dims,
                const NDArray& clipNorm, const bool isInplace, const bool useAverage) {
  NDArray* z = nullptr;

  if (isInplace) {
    z = &input;
  } else {
    output.assign(input);
    z = &output;
  }

  if (dims.empty()) {
    std::vector<sd::LongType> empty;
    const NDArray actualNorm = useAverage ? z->reduceAlongDimension(reduce::Norm2, &empty) / z->lengthOf()
                                          : z->reduceAlongDimension(reduce::Norm2, &empty);

    if (actualNorm.e<float>(0) > clipNorm.e<float>(0)) *z *= clipNorm / actualNorm;
  } else {
    const NDArray actualNorms = z->reduceAlongDimension(reduce::Norm2, &dims);

    std::vector<sd::LongType> *dimsToExclude = ShapeUtils::evalDimsToExclude(z->rankOf(), dims.size(),dims.data());

    const int threadsPerBlock = SD_MAX_NUM_THREADS / 2;
    const int blocksPerGrid = (z->lengthOf() + threadsPerBlock - 1) / threadsPerBlock;

    PointersManager manager(context, "clipByNorm");

    const sd::LongType * dimensions = reinterpret_cast<const sd::LongType *>(
        manager.replicatePointer(dimsToExclude->data(), dimsToExclude->size() * sizeof(sd::LongType)));

    NDArray::prepareSpecialUse({z}, {z, &actualNorms, &clipNorm});


    BUILD_SINGLE_SELECTOR(z->dataType(), clipByNormCudaLauncher,
                          (blocksPerGrid,
                              threadsPerBlock,
                              context->getCudaStream(),
                              clipNorm.specialBuffer(),
                              actualNorms.specialBuffer(),
                              actualNorms.specialShapeInfo(),
                              z->specialBuffer(),
                              z->specialShapeInfo(),
                              dimensions,
                              dimsToExclude->size(),
                              useAverage),
                          SD_FLOAT_TYPES);
    NDArray::registerSpecialUse({z}, {z, &actualNorms, &clipNorm});

    manager.synchronize();
    delete dimsToExclude;
  }
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
SD_KERNEL static void clipByNormBpCuda(const void* vClipNorm, const void* vx, const sd::LongType* xShapeInfo,  // input
                                       const void* vy, const sd::LongType* yShapeInfo,                         // gradO
                                       const void* vNorm, const sd::LongType* normShapeInfo, const void* vSum,
                                       const sd::LongType* sumShapeInfo, void* vz,
                                       const sd::LongType* zShapeInfo,  // gradI
                                       const sd::LongType* dimensions, const sd::LongType dimsLen, const bool useAverage) {
  const T clipNorm = *reinterpret_cast<const T*>(vClipNorm);
  const T* norm = reinterpret_cast<const T*>(vNorm);
  const T* sum = reinterpret_cast<const T*>(vSum);
  const T* x = reinterpret_cast<const T*>(vx);
  const T* y = reinterpret_cast<const T*>(vy);
  T* z = reinterpret_cast<T*>(vz);

  __shared__ sd::LongType zLen, tadLen, totalThreads;
  __shared__ bool sameOffsets;

  if (threadIdx.x == 0) {
    zLen = shape::length(zShapeInfo);
    tadLen = zLen / shape::length(normShapeInfo);
    totalThreads = gridDim.x * blockDim.x;

    sameOffsets = shape::haveSameShapeAndStrides(xShapeInfo, yShapeInfo, zShapeInfo);
  }

  __syncthreads();

  sd::LongType zCoords[SD_MAX_RANK], normCoords[SD_MAX_RANK];

  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

  for (sd::LongType i = tid; i < zLen; i += totalThreads) {
    shape::index2coords(i, zShapeInfo, zCoords);

    const auto zOffset = shape::getOffset(zShapeInfo, zCoords);
    const auto yOffset = sameOffsets ? zOffset : shape::getOffset(yShapeInfo, zCoords);

    // deduce norm coords
    for (int j = 0; j < dimsLen; ++j) normCoords[j] = zCoords[dimensions[j]];

    const T actualNorm = useAverage ? norm[shape::getOffset(normShapeInfo, normCoords)] / tadLen
                                    : norm[shape::getOffset(normShapeInfo, normCoords)];

    if (actualNorm > clipNorm) {
      const T sumVal = sum[shape::getOffset(sumShapeInfo, normCoords)];
      const auto xOffset = sameOffsets ? zOffset : shape::getOffset(xShapeInfo, zCoords);

      z[zOffset] = (clipNorm / actualNorm) * y[yOffset] *
                   (static_cast<T>(1.f) - (x[xOffset] * sumVal) / (actualNorm * actualNorm));
    } else
      z[zOffset] = y[yOffset];
  }
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void clipByNormBp_(sd::LaunchContext* context, const NDArray& input, const NDArray& gradO, NDArray& gradI,
                   const std::vector<sd::LongType>& dims, const NDArray& clipNorm, const bool useAverage) {
  const int rank = input.rankOf();

  auto actualNorms = input.reduceAlongDimension(reduce::Norm2, &dims);

  if (actualNorms.lengthOf() == 1) {
    const T norm = useAverage ? actualNorms.e<T>(0) / static_cast<T>(input.lengthOf()) : actualNorms.e<T>(0);

    auto clipVal = clipNorm.e<T>(0);

    if (norm > clipVal) {
      const T sum = input.reduceNumber(reduce::Sum).e<T>(0);  // reduce to scalar
      const T factor1 = clipVal / norm;
      const T factor2 = static_cast<T>(1.f) / (norm * norm);  // 1 / (norm*norm*norm)

      auto lambda = LAMBDA_TT(x, y, sum, factor1, factor2) {
        return factor1 * y * (static_cast<T>(1.f) - factor2 * x * sum);
      };

      const_cast<NDArray&>(input).applyPairwiseLambda(const_cast<NDArray&>(gradO), lambda, gradI);
    } else
      gradI.assign(gradO);
  } else {
    const NDArray actualNorms = input.reduceAlongDimension(reduce::Norm2, &dims);
    const NDArray sums = input.reduceAlongDimension(reduce::Sum, &dims);

    std::vector<sd::LongType> *dimsToExclude = ShapeUtils::evalDimsToExclude(gradI.rankOf(), dims.size(),dims.data());


    dim3 launchDims = clipDims(gradI.lengthOf());
    PointersManager manager(context, "clipByNormBp");

    const sd::LongType* dimensions = reinterpret_cast<const sd::LongType*>(
        manager.replicatePointer(dimsToExclude->data(), dimsToExclude->size() * sizeof(sd::LongType)));

    NDArray::prepareSpecialUse({&gradI}, {&actualNorms, &sums, &clipNorm, &input, &gradO});
    clipByNormBpCuda<T><<<launchDims.y, launchDims.x,launchDims.z, *context->getCudaStream()>>>(
        clipNorm.specialBuffer(), input.specialBuffer(), input.specialShapeInfo(), gradO.specialBuffer(),
        gradO.specialShapeInfo(), actualNorms.specialBuffer(), actualNorms.specialShapeInfo(), sums.specialBuffer(),
        sums.specialShapeInfo(), gradI.specialBuffer(), gradI.specialShapeInfo(), dimensions, (sd::LongType)dimsToExclude->size(),
        useAverage);
    NDArray::registerSpecialUse({&gradI}, {&actualNorms, &sums, &clipNorm, &input, &gradO});

    manager.synchronize();
    delete dimsToExclude;
  }
}
BUILD_SINGLE_TEMPLATE(template void clipByNormBp_,
                      (sd::LaunchContext * context, const NDArray& input, const NDArray& gradO, NDArray& gradI,
                          const std::vector<sd::LongType>& dimensions, const NDArray& clipNorm, const bool useAverage),
                      SD_FLOAT_TYPES);

//////////////////////////////////////////////////////////////////////////
void clipByNormBp(sd::LaunchContext* context, const NDArray& input, const NDArray& gradO, NDArray& gradI,
                  const std::vector<sd::LongType>& dimensions, const NDArray& clipNorm, const bool useAverage) {
  const NDArray& castedInput = gradI.dataType() == input.dataType() ? input : input.cast(gradI.dataType());
  BUILD_SINGLE_SELECTOR(gradI.dataType(), clipByNormBp_,
                        (context, castedInput, gradO, gradI, dimensions, clipNorm, useAverage), SD_FLOAT_TYPES);
}

template <typename T>
void clipByGlobalNorm_(sd::LaunchContext* context, std::vector<NDArray*> const& inputs, double clipNorm,
                       sd::memory::Workspace* workspace, std::vector<NDArray*>& outputs, bool isInplace) {
  NDArray globalNorm =
      NDArrayFactory::create<T>(0, inputs[0]->getContext());  // sqrt(sum([l2norm(t)**2 for t in t_list]))

  for (auto i = 0; i < inputs.size(); i++) {
    auto input = inputs[i];
    auto l2norm = input->reduceNumber(reduce::Norm2);
    globalNorm += l2norm * l2norm;
  }

  globalNorm.applyTransform(transform::Sqrt, globalNorm);
  outputs[inputs.size()]->p(0, globalNorm);
  globalNorm.syncToHost();
  const T factor = static_cast<T>(clipNorm) / globalNorm.e<T>(0);

  for (size_t e = 0; e < inputs.size(); e++) {
    // all-reduce
    auto input = inputs[e];
    auto output = outputs[e];

    if (globalNorm.e<double>(0) <= clipNorm) {
      output->assign(input);
    } else {
      auto lambda = LAMBDA_T(_x, factor) { return _x * factor; };
      input->applyLambda(lambda, *output);
    }
  }
}

void clipByGlobalNorm(sd::LaunchContext* context, std::vector<NDArray*> const& inputs, double clipNorm,
                      sd::memory::Workspace* workspace, std::vector<NDArray*>& outputs, bool isInplace) {
  BUILD_SINGLE_SELECTOR(outputs[0]->dataType(), clipByGlobalNorm_,
                        (context, inputs, clipNorm, workspace, outputs, isInplace), SD_FLOAT_TYPES);
}

BUILD_SINGLE_TEMPLATE(template void clipByGlobalNorm_,
                      (sd::LaunchContext * context, std::vector<NDArray*> const& inputs, double clipNorm,
                          sd::memory::Workspace* workspace, std::vector<NDArray*>& outputs, bool isInplace),
                      SD_FLOAT_TYPES);

template <typename T>
static void SD_KERNEL clipByValueKernel(void* input, const sd::LongType* inputShape, void* output,
                                        const sd::LongType* outputShape, double leftBound, double rightBound) {
  __shared__ T* outputBuf;
  __shared__ T* inputBuf;
  __shared__ sd::LongType length;
  __shared__ bool linearBuffers;
  if (threadIdx.x == 0) {
    outputBuf = reinterpret_cast<T*>(output);
    inputBuf = reinterpret_cast<T*>(input);
    length = shape::length(inputShape);
    linearBuffers = shape::elementWiseStride(inputShape) == shape::elementWiseStride(outputShape) &&
                    shape::elementWiseStride(inputShape) == 1;
  }
  __syncthreads();
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  const auto step = gridDim.x * blockDim.x;

  for (sd::LongType e = tid; e < length; e += step) {
    if (linearBuffers) {
      if (inputBuf[e] > rightBound)
        outputBuf[e] = (T)rightBound;
      else if (inputBuf[e] < leftBound)
        outputBuf[e] = (T)leftBound;
      else
        outputBuf[e] = inputBuf[e];
    } else {
      auto inputOffset = shape::getIndexOffset(e, inputShape);
      auto outputOffset = shape::getIndexOffset(e, outputShape);
      if (inputBuf[inputOffset] > rightBound)
        outputBuf[outputOffset] = (T)rightBound;
      else if (inputBuf[inputOffset] < leftBound)
        outputBuf[outputOffset] = (T)leftBound;
      else
        outputBuf[outputOffset] = inputBuf[outputOffset];
    }
  }
}

template <typename T>
static void clipByValue_(sd::LaunchContext* context, NDArray& input, double leftBound, double rightBound,
                         NDArray& output) {
  auto stream = context->getCudaStream();
  if (!input.isActualOnDeviceSide()) input.syncToDevice();
  NDArray::prepareSpecialUse({&output}, {&input});
  dim3 launchDims = getLaunchDims("clip");
  clipByValueKernel<T><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(input.specialBuffer(), input.specialShapeInfo(),
                                                                              output.specialBuffer(), output.specialShapeInfo(), leftBound,
                                                                              rightBound);
  NDArray::registerSpecialUse({&output}, {&input});
}

void clipByValue(sd::LaunchContext* context, NDArray& input, double leftBound, double rightBound, NDArray& output) {
  BUILD_SINGLE_SELECTOR(input.dataType(), clipByValue_, (context, input, leftBound, rightBound, output),
                        SD_FLOAT_TYPES);
}

BUILD_SINGLE_TEMPLATE(template void clipByValue_, (sd::LaunchContext * context, NDArray& input, double leftBound,
    double rightBound, NDArray& output);
, SD_FLOAT_TYPES);

}  // namespace helpers
}  // namespace ops
}  // namespace sd

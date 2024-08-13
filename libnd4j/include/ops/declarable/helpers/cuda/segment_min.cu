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
//  @author GS <sgazeos@gmail.com>
//
#include <array/NDArrayFactory.h>
#include <exceptions/cuda_exception.h>
#include <execution/cuda/LaunchDims.h>
#include <helpers/ConstantTadHelper.h>
#include <helpers/PointersManager.h>
#include <helpers/ShapeUtils.h>
#include <helpers/TAD.h>
#include <ops/declarable/helpers/segment.h>
#include <ops/declarable/helpers/segment_common.h>

#include "helpers/DebugHelper.h"


namespace sd {
namespace ops {
namespace helpers {
// -------------------------------------------------------------------------------------------------------------- //
// Segment ops linear kernels
// -------------------------------------------------------------------------------------------------------------- //

template <typename T, typename I>
static SD_KERNEL void segmentMinLinearKernel(const void* input, const LongType* inputShape, LongType* starts,
                                             LongType* lengths, LongType numOfClasses, void* output,
                                             const LongType* outputShape) {
  __shared__ T* val;
  __shared__ LongType xLen, zLen, zIndex;
  __shared__ const T* x;
  __shared__ T* z;
  __shared__ LongType threadsPerSegment, start, finish;

  auto segment = blockIdx.x;
  if(blockIdx.x >= numOfClasses)
    return;
  if (threadIdx.x == 0) {
    x = reinterpret_cast<const T*>(input);
    z = reinterpret_cast<T*>(output);
    extern __shared__ unsigned char shmem[];
    val = reinterpret_cast<T*>(shmem);
    xLen = shape::length(inputShape);
    zLen = shape::length(outputShape);

    if (segment < numOfClasses) {
      zIndex = shape::getIndexOffset(segment, outputShape);
      if(zIndex >= zLen)
        return;
      start = starts[segment];
      finish = start + lengths[segment];
      z[zIndex] = x[shape::getIndexOffset(start, inputShape)];
      val[segment] = z[zIndex];
    }
  }
  __syncthreads();

  for (auto e = start + threadIdx.x + 1; e < finish; e += blockDim.x) {
    auto xIndex = shape::getIndexOffset(e, inputShape);
    if (xIndex >= xLen) return;
    math::atomics::sd_atomicMin(&z[zIndex], x[xIndex]);
  }
}
// -------------------------------------------------------------------------------------------------------------- //

template <typename T, typename I>
static SD_KERNEL void unsortedSegmentMinLinearKernel(const void* input, const LongType* inputShape,
                                                     const void* indices, const LongType* indicesShape, LongType* starts, LongType* lengths,
                                                     LongType numOfClasses, void* output,
                                                     const LongType* outputShape) {
  __shared__ T* val;
  __shared__ LongType xLen, zLen, segment, zIndex;
  __shared__ const T* x;
  __shared__ T* z;
  __shared__ const I* y;  // int threadsPerSegment, start, finish;

  if (threadIdx.x == 0) {
    segment = blockIdx.x;
    x = reinterpret_cast<const T*>(input);
    z = reinterpret_cast<T*>(output);
    y = reinterpret_cast<const I*>(indices);
    xLen = shape::length(inputShape);
    zLen = shape::length(outputShape);

    zIndex = shape::getIndexOffset(segment, outputShape);
    if (lengths[segment] > 0)
      z[zIndex] = x[shape::getIndexOffset(starts[segment], inputShape)];
    else
      z[zIndex] = DataTypeUtils::max<T>();
  }
  __syncthreads();

  if (lengths[segment] > 0)
    for (auto e = threadIdx.x + 1; e < xLen; e += blockDim.x) {
      auto xIndex = shape::getIndexOffset(e, inputShape);
      auto yIndex = shape::getIndexOffset(e, indicesShape);
      if (y[yIndex] == segment) {
        math::atomics::sd_atomicMin(&z[zIndex], x[xIndex]);
      }
    }
}
// -------------------------------------------------------------------------------------------------------------- //
// SegmentMin kernel
template <typename T, typename I>
static SD_KERNEL void segmentMinTadKernel(const void* inputBuf, const LongType* inputShape,
                                          const LongType* inputTads, const LongType* inputTadOffsets,
                                          I* indices, LongType* starts,
                                          LongType* lengths, LongType numOfClasses, void* outputBuf, const LongType* outputShape,
                                          const LongType* outputTads, const LongType* outputTadOffsets, LongType indicesLen) {
  __shared__ T* val;
  __shared__ LongType len, zIndex, total;
  __shared__ T* z;
  __shared__ int threadsPerSegment, start, finish;
  if(blockIdx.x >= indicesLen)
    return;
  auto segment = indices[blockIdx.x];  // / threadsPerSegment;



  if (threadIdx.x == 0) {
    z = reinterpret_cast<T*>(outputBuf) + outputTadOffsets[segment];
    len = shape::length(inputTads);
    start = starts[segment];
    finish = start + lengths[segment];
    total = shape::sizeAt(inputShape, 0);
  }
  __syncthreads();

  auto idx = blockIdx.x;
  if (blockIdx.x <= total) {
    auto x = reinterpret_cast<const T*>(inputBuf) + inputTadOffsets[idx];
    if (blockIdx.x == start) {
      for (auto e = threadIdx.x; e < len; e += blockDim.x) {
        auto xIndex = shape::getIndexOffset(e, inputTads);
        auto zIndex = shape::getIndexOffset(e, outputTads);
        math::atomics::sd_atomicMin(&z[zIndex], x[xIndex]);
      }
    } else {
      for (auto e = threadIdx.x; e < len; e += blockDim.x) {
        auto xIndex = shape::getIndexOffset(e, inputTads);
        auto zIndex = shape::getIndexOffset(e, outputTads);
        math::atomics::sd_atomicMin(&z[zIndex], x[xIndex]);
      }
    }
  }
}
// -------------------------------------------------------------------------------------------------------------- //
// segmen min
template <typename T, typename I>
static void segmentMinFunctor_(LaunchContext* context, NDArray* input, NDArray* indices, NDArray* output) {
  auto stream = context->getCudaStream();
  LongType numClasses = indices->e<LongType>(indices->lengthOf() - 1) + 1;
  auto classesRangesLens = NDArrayFactory::create<LongType>('c', {numClasses}, context);
  auto classesRangesBegs = NDArrayFactory::create<LongType>('c', {numClasses}, context);
  output->assign(DataTypeUtils::infOrMax<T>());
  classesRangesBegs.assign(indices->lengthOf());
  classesRangesLens.assign(0);

  fillUpSegments(indices, numClasses, classesRangesBegs, classesRangesLens);
  NDArray::prepareSpecialUse({output}, {input, indices, &classesRangesBegs, &classesRangesLens});
  LongType* begins = reinterpret_cast<LongType*>(classesRangesBegs.specialBuffer());
  LongType* lengths = reinterpret_cast<LongType*>(classesRangesLens.specialBuffer());
  if (input->isVector()  || input->isScalar()) {
    dim3 launchDims = segmentDims(numClasses,input->lengthOf());
    segmentMinLinearKernel<T, I><<<launchDims.y,launchDims.x, launchDims.z, *stream>>>(
        input->specialBuffer(), input->specialShapeInfo(), begins, lengths, numClasses, output->specialBuffer(),
        output->specialShapeInfo());
    sd::DebugHelper::checkErrorCode(stream, "segmentMinLinearKernel failed");

  } else {
    LongType zero = 0;
    std::vector<LongType> *dimensions = ShapeUtils::evalDimsToExclude(input->rankOf(),1,&zero);
    auto packX = ConstantTadHelper::getInstance().tadForDimensions(input->shapeInfo(), dimensions);
    auto packZ = ConstantTadHelper::getInstance().tadForDimensions(output->shapeInfo(), dimensions);
    auto inputTads = packX->specialShapeInfo();
    auto inputTadOffsets = packX->specialOffsets();
    auto outputTads = packZ->specialShapeInfo();
    auto outputTadOffsets = packZ->specialOffsets();
    dim3 launchDims = segmentTad(input->sizeAt(0));
    segmentMinTadKernel<T, I><<<launchDims.y, launchDims.x, launchDims.z, *stream>>>(
        input->specialBuffer(), input->specialShapeInfo(), inputTads, inputTadOffsets,
        reinterpret_cast<I*>(indices->specialBuffer()), begins, lengths, numClasses, output->specialBuffer(),
        output->specialShapeInfo(), outputTads, outputTadOffsets, indices->lengthOf());
    sd::DebugHelper::checkErrorCode(stream, "segmentMinTadKernel failed");

    delete dimensions;
  }
  NDArray::registerSpecialUse({output}, {input, indices, &classesRangesBegs, &classesRangesLens});
}
// -------------------------------------------------------------------------------------------------------------- //
void segmentMinFunctor(LaunchContext* context, NDArray* input, NDArray* indices, NDArray* output) {
  NDArray::prepareSpecialUse({output}, {input, indices});
  output->nullify();
  BUILD_DOUBLE_SELECTOR(input->dataType(), indices->dataType(), segmentMinFunctor_, (context, input, indices, output),
                        SD_NUMERIC_TYPES, SD_INDEXING_TYPES);
  NDArray::registerSpecialUse({output}, {input, indices});
}

// -------------------------------------------------------------------------------------------------------------- //

template <typename T, typename I>
static void unsortedSegmentMinFunctor_(LaunchContext* context, NDArray* input, NDArray* indices, LongType numOfClasses, NDArray* output) {
  auto stream = context->getCudaStream();
  NDArray classesRangesBegs = NDArrayFactory::create<LongType>('c', {numOfClasses}, context);
  NDArray classesRangesLens = NDArrayFactory::create<LongType>('c', {numOfClasses}, context);
  output->assign(DataTypeUtils::infOrMax<T>());
  classesRangesBegs.assign(indices->lengthOf());
  classesRangesLens.assign(0);
  dim3 dims = getFillUpSegmentsDims(numOfClasses, indices->lengthOf());
  fillUpSegments(indices, numOfClasses, classesRangesBegs, classesRangesLens);
  LongType* begins = reinterpret_cast<LongType*>(classesRangesBegs.specialBuffer());
  LongType* lengths = reinterpret_cast<LongType*>(classesRangesLens.specialBuffer());
  NDArray::prepareSpecialUse({output}, {input, indices});
  if (input->isVector()  || input->isScalar()) {
    unsortedSegmentMinLinearKernel<T, I><<<dims.x, dims.y, dims.z, *stream>>>(
        input->specialBuffer(), input->specialShapeInfo(), indices->specialBuffer(), indices->specialShapeInfo(),
        begins, lengths, numOfClasses, output->specialBuffer(), output->specialShapeInfo());
    sd::DebugHelper::checkErrorCode(stream, "unsortedSegmentMinLinearKernel failed");

  } else {
    output->assign(DataTypeUtils::max<T>());
    LongType zero = 0;
    std::vector<LongType> *dimensions = ShapeUtils::evalDimsToExclude(input->rankOf(),1,&zero);
    auto packX = ConstantTadHelper::getInstance().tadForDimensions(input->shapeInfo(), dimensions);
    auto packZ = ConstantTadHelper::getInstance().tadForDimensions(output->shapeInfo(), dimensions);
    auto inputTads = packX->specialShapeInfo();
    auto inputTadOffsets = packX->specialOffsets();
    auto outputTads = packZ->specialShapeInfo();
    auto outputTadOffsets = packZ->specialOffsets();
    dims.x = input->sizeAt(0);
    segmentMinTadKernel<T, I><<<dims.x, dims.y, dims.z, *stream>>>(
        input->specialBuffer(), input->specialShapeInfo(), inputTads, inputTadOffsets,
        reinterpret_cast<I*>(indices->specialBuffer()), begins, lengths, numOfClasses, output->specialBuffer(),
        output->specialShapeInfo(), outputTads, outputTadOffsets, indices->lengthOf());
    sd::DebugHelper::checkErrorCode(stream, "segmentMinTadKernel failed");

    delete dimensions;
  }
  NDArray::registerSpecialUse({output}, {input, indices});
}
// -------------------------------------------------------------------------------------------------------------- //
void unsortedSegmentMinFunctor(LaunchContext* context, NDArray* input, NDArray* indices, LongType numOfClasses,
                               NDArray* output) {
  NDArray::prepareSpecialUse({output}, {input, indices});
  output->nullify();
  BUILD_DOUBLE_SELECTOR(input->dataType(), indices->dataType(), unsortedSegmentMinFunctor_,
                        (context, input, indices, numOfClasses, output), SD_NUMERIC_TYPES, SD_INDEXING_TYPES);
  NDArray::registerSpecialUse({output}, {input, indices});
}

template <typename T, typename I>
static SD_KERNEL void segmentMinBPLinearKernel(const void* inputBuf, const LongType* inputShape,
                                               void* forwardOutput, const LongType* forwardShape, void* eps,
                                               const LongType* epsShape, const void* indicesBuf,
                                               const LongType* indicesShape, void* outputBuf,
                                               const LongType* outputShape) {
  __shared__ const T* x;
  __shared__ T* gradIn;
  __shared__ T* gradOut;
  __shared__ const I* y;
  __shared__ T* z;
  __shared__ LongType xLen, gradLen;

  if (threadIdx.x == 0) {
    xLen = shape::length(inputShape);
    x = reinterpret_cast<const T*>(inputBuf);
    y = reinterpret_cast<const I*>(indicesBuf);
    z = reinterpret_cast<T*>(outputBuf);
    gradIn = reinterpret_cast<T*>(forwardOutput);
    gradOut = reinterpret_cast<T*>(eps);
    gradLen = shape::length(epsShape);
  }
  __syncthreads();

  auto start = blockIdx.x * blockDim.x + threadIdx.x;
  auto step = gridDim.x * blockDim.x;

  for (auto e = start; e < xLen; e += step) {
    auto zOffset = shape::getIndexOffset(e, outputShape);
    auto xOffset = shape::getIndexOffset(e, inputShape);
    auto yOffset = shape::getIndexOffset(e, indicesShape);
    auto classIndex = y[yOffset];
    auto gradOffsetI = shape::getIndexOffset(classIndex, forwardShape);
    auto gradOffsetO = shape::getIndexOffset(classIndex, epsShape);

    if (math::sd_abs(gradIn[gradOffsetI] - x[xOffset]) <= T(1.e-6)) {
      z[zOffset] = gradOut[gradOffsetO];
    }
  }
}

// -------------------------------------------------------------------------------------------------------------- //
template <typename T, typename I>
static SD_KERNEL void segmentMinBPTadKernel(const void* inputBuf, const LongType* inputShape, void* forwardOutput,
                                            const LongType* forwardShape, void* eps, const LongType* epsShape,
                                            const void* indicesBuf, const LongType* indicesShape, void* outputBuf,
                                            const LongType* outputShape, const LongType* inputTad,
                                            const LongType* inputOffsets, const LongType* gradInTad,
                                            const LongType* gradInOffsets, const LongType* gradOutTad,
                                            const LongType* gradOutOffsets, const LongType* outTad,
                                            const LongType* outOffsets) {
  __shared__ const T* x;
  __shared__ T* gradIn;
  __shared__ T* gradOut;
  __shared__ const I* y;
  __shared__ T* z;
  __shared__ LongType xLen, yLen, gradLen, currentLen;

  if (threadIdx.x == 0) {
    xLen = shape::length(inputShape);
    x = reinterpret_cast<const T*>(inputBuf);
    y = reinterpret_cast<const I*>(indicesBuf);
    z = reinterpret_cast<T*>(outputBuf);
    yLen = shape::length(indicesShape);
    gradOut = reinterpret_cast<T*>(eps);
    gradIn = reinterpret_cast<T*>(forwardOutput);
    gradLen = shape::length(epsShape);
    currentLen = shape::length(outTad);
  }
  __syncthreads();

  for (auto i = blockIdx.x; i < yLen; i += gridDim.x) {
    auto yIndex = shape::getIndexOffset(i, indicesShape);
    auto segment = y[yIndex];
    auto current = x + inputOffsets[i];
    auto currentOut = z + outOffsets[i];
    auto in = gradIn + gradInOffsets[segment];
    auto outGrad = gradOut + gradOutOffsets[segment];

    for (auto e = threadIdx.x; e < currentLen; e += blockDim.x) {
      if (math::sd_abs(in[e] - current[e]) <= T(1.e-6)) currentOut[e] = outGrad[e];
    }
  }
}

// -------------------------------------------------------------------------------------------------------------- //
template <typename T, typename I>
Status segmentMinFunctorBP_(LaunchContext* context, NDArray* input, NDArray* indices, NDArray* gradOut,
                                NDArray* output) {

  // if input is a vector: (as if in doc sample)
  auto stream = context->getCudaStream();
  auto outShape = gradOut->getShapeAsVector();
  NDArray tempRes(gradOut->ordering(), outShape, DataTypeUtils::fromT<T>(),
                  context);
  segmentMinFunctor_<T, I>(context, input, indices, &tempRes);
  NDArray::prepareSpecialUse({output}, {input, indices, gradOut, &tempRes});
  if (input->isVector()  || input->isScalar()) {
    LongType loop_size = input->lengthOf();
    auto numOfClasses = gradOut->lengthOf();

    segmentMinBPLinearKernel<T, I><<<gradOut->lengthOf(), input->lengthOf(), 256, *stream>>>(
        input->specialBuffer(), input->specialShapeInfo(), tempRes.specialBuffer(), tempRes.specialShapeInfo(),
        gradOut->specialBuffer(), gradOut->specialShapeInfo(), indices->specialBuffer(), indices->specialShapeInfo(),
        output->specialBuffer(), output->specialShapeInfo());
    sd::DebugHelper::checkErrorCode(stream, "segmentMinBPLinearKernel failed");


  } else {
    LongType zero = 0;
    std::vector<LongType> *dimensions = ShapeUtils::evalDimsToExclude(input->rankOf(),1,&zero);
    auto packX = ConstantTadHelper::getInstance().tadForDimensions(input->shapeInfo(), dimensions);
    auto packZ = ConstantTadHelper::getInstance().tadForDimensions(output->shapeInfo(), dimensions);
    auto packGradIn = ConstantTadHelper::getInstance().tadForDimensions(tempRes.shapeInfo(), dimensions);
    auto packGradOut = ConstantTadHelper::getInstance().tadForDimensions(gradOut->shapeInfo(), dimensions);
    auto inputTads = packX->specialShapeInfo();
    auto inputTadOffsets = packX->specialOffsets();
    auto outputTads = packZ->specialShapeInfo();
    auto outputTadOffsets = packZ->specialOffsets();
    auto gradInTads = packGradIn->specialShapeInfo();
    auto gradInTadOffsets = packGradIn->specialOffsets();
    auto gradOutTads = packGradOut->specialShapeInfo();
    auto gradOutTadOffsets = packGradOut->specialOffsets();

    segmentMinBPTadKernel<T, I><<<gradOut->lengthOf(), input->lengthOf(), 256, *stream>>>(
        input->specialBuffer(), input->specialShapeInfo(), tempRes.specialBuffer(), tempRes.specialShapeInfo(),
        gradOut->specialBuffer(), gradOut->specialShapeInfo(), indices->specialBuffer(), indices->specialShapeInfo(),
        output->specialBuffer(), output->specialShapeInfo(), inputTads, inputTadOffsets, gradInTads, gradInTadOffsets,
        gradOutTads, gradOutTadOffsets, outputTads, outputTadOffsets);
    sd::DebugHelper::checkErrorCode(stream, "segmentMinBPTadKernel failed");

  }
  NDArray::registerSpecialUse({output}, {input, indices, gradOut, &tempRes});
  return Status::OK;
}
// -------------------------------------------------------------------------------------------------------------- //
// segmen min
Status segmentMinFunctorBP(LaunchContext* context, NDArray* input, NDArray* indices, NDArray* gradOut,
                               NDArray* output) {
  NDArray::prepareSpecialUse({output}, {input, indices, gradOut});
  BUILD_DOUBLE_SELECTOR(output->dataType(), indices->dataType(), return segmentMinFunctorBP_,
                        (context, input, indices, gradOut, output), SD_FLOAT_TYPES, SD_INDEXING_TYPES);
  NDArray::registerSpecialUse({output}, {input, indices, gradOut});
}

template <typename T, typename I>
static Status unsortedSegmentMinFunctorBP_(LaunchContext* context, NDArray* input, NDArray* indices,
                                               NDArray* gradOut,
                                           LongType numOfClasses, NDArray* output) {
  // if input is a vector: (as if in doc sample)
  auto stream = context->getCudaStream();
  auto outShape = gradOut->getShapeAsVector();

  NDArray tempRes(gradOut->ordering(), outShape, DataTypeUtils::fromT<T>(),
                  context);
  unsortedSegmentMinFunctor_<T, I>(context, input, indices, numOfClasses, &tempRes);
  NDArray::prepareSpecialUse({output}, {input, indices, gradOut, &tempRes});
  if (input->isVector()  || input->isScalar()) {
    LongType loop_size = input->lengthOf();
    auto numOfClasses = gradOut->lengthOf();
    segmentMinBPLinearKernel<T, I><<<gradOut->lengthOf(), input->lengthOf(), 256, *stream>>>(
        input->specialBuffer(), input->specialShapeInfo(), tempRes.specialBuffer(), tempRes.specialShapeInfo(),
        gradOut->specialBuffer(), gradOut->specialShapeInfo(), indices->specialBuffer(), indices->specialShapeInfo(),
        output->specialBuffer(), output->specialShapeInfo());
    sd::DebugHelper::checkErrorCode(stream, "segmentMinBPLinearKernel failed");

  } else {
    LongType zero = 0;

    std::vector<LongType> *dimensions = ShapeUtils::evalDimsToExclude(input->rankOf(), 1,&zero);
    auto packX = ConstantTadHelper::getInstance().tadForDimensions(input->shapeInfo(), dimensions);
    auto packZ = ConstantTadHelper::getInstance().tadForDimensions(output->shapeInfo(), dimensions);
    auto packGradIn = ConstantTadHelper::getInstance().tadForDimensions(tempRes.shapeInfo(), dimensions);
    auto packGradOut = ConstantTadHelper::getInstance().tadForDimensions(gradOut->shapeInfo(), dimensions);
    auto inputTads = packX->specialShapeInfo();
    auto inputTadOffsets = packX->specialOffsets();
    auto outputTads = packZ->specialShapeInfo();
    auto outputTadOffsets = packZ->specialOffsets();
    auto gradInTads = packGradIn->specialShapeInfo();
    auto gradInTadOffsets = packGradIn->specialOffsets();
    auto gradOutTads = packGradOut->specialShapeInfo();
    auto gradOutTadOffsets = packGradOut->specialOffsets();

    segmentMinBPTadKernel<T, I><<<gradOut->lengthOf(), input->lengthOf(), 256, *stream>>>(
        input->specialBuffer(), input->specialShapeInfo(), tempRes.specialBuffer(), tempRes.specialShapeInfo(),
        gradOut->specialBuffer(), gradOut->specialShapeInfo(), indices->specialBuffer(), indices->specialShapeInfo(),
        output->specialBuffer(), output->specialShapeInfo(), inputTads, inputTadOffsets, gradInTads, gradInTadOffsets,
        gradOutTads, gradOutTadOffsets, outputTads, outputTadOffsets);
    sd::DebugHelper::checkErrorCode(stream, "segmentMinBPTadKernel failed");

    delete dimensions;
  }
  NDArray::registerSpecialUse({output}, {input, indices, gradOut, &tempRes});
  return Status::OK;
}
// -------------------------------------------------------------------------------------------------------------- //
Status unsortedSegmentMinFunctorBP(LaunchContext* context, NDArray* input, NDArray* indices, NDArray* gradOut,
                                   LongType numOfClasses, NDArray* output) {
  NDArray::prepareSpecialUse({output}, {input, indices, gradOut});
  BUILD_DOUBLE_SELECTOR(output->dataType(), indices->dataType(), return unsortedSegmentMinFunctorBP_,
                        (context, input, indices, gradOut, numOfClasses, output), SD_FLOAT_TYPES, SD_INDEXING_TYPES);
  NDArray::registerSpecialUse({output}, {input, indices, gradOut});
}
}  // namespace helpers
}  // namespace ops
}  // namespace sd

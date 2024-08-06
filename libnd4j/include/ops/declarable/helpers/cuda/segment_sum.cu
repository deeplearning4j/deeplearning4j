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
static SD_KERNEL void segmentSumLinearKernel(const void* input, const LongType* inputShape, LongType* starts,
                                             LongType* lengths, LongType numOfClasses, void* output,
                                             const LongType* outputShape) {
  __shared__ T* val;
  __shared__ LongType xLen, zLen, segment, zIndex;
  __shared__ const T* x;
  __shared__ T* z;
  __shared__ int threadsPerSegment, start, finish;

  if (threadIdx.x == 0) {
    threadsPerSegment = (gridDim.x + numOfClasses - 1) / numOfClasses;
    segment = blockIdx.x / threadsPerSegment;
    x = reinterpret_cast<const T*>(input);
    z = reinterpret_cast<T*>(output);

    xLen = shape::length(inputShape);
    zLen = shape::length(outputShape);

    if (segment < numOfClasses) {
      zIndex = shape::getIndexOffset(segment, outputShape);
      if(zIndex >= zLen)
        return;
      start = starts[segment];
      finish = start + lengths[segment];
      z[zIndex] = x[shape::getIndexOffset(start, inputShape)];
    }
  }
  __syncthreads();

  for (auto e = start + threadIdx.x + 1; e < finish; e += blockDim.x) {
    auto xIndex = shape::getIndexOffset(e, inputShape);
    if (xIndex >= xLen) return;
    math::atomics::sd_atomicAdd(&z[zIndex], x[xIndex]);
  }
}
// -------------------------------------------------------------------------------------------------------------- //

template <typename T, typename I>
static SD_KERNEL void unsortedSegmentSumLinearKernel(const void* input, const LongType* inputShape,
                                                     const void* indices, const LongType* indicesShape, LongType* starts, LongType* lengths,
                                                     LongType numOfClasses, void* output,
                                                     const LongType* outputShape) {
  __shared__ T* val;
  __shared__ LongType xLen, zLen, segment, zIndex;
  __shared__ const T* x;
  __shared__ T* z;
  __shared__ const I* y;

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
      z[zIndex] = 0;
  }
  __syncthreads();

  if (lengths[segment] > 0)
    for (auto e = threadIdx.x; e < xLen; e += blockDim.x) {
      auto xIndex = shape::getIndexOffset(e, inputShape);
      auto yIndex = shape::getIndexOffset(e, indicesShape);
      if (y[yIndex] == segment && e != starts[segment]) {
        math::atomics::sd_atomicAdd(&z[zIndex], x[xIndex]);
      }
    }
}
// -------------------------------------------------------------------------------------------------------------- //
// SegmentSum kernel
template <typename T, typename I>
static SD_KERNEL void segmentSumTadKernel(void* inputBuf, const LongType* inputShape,
                                          const LongType* inputTads, const LongType* inputTadOffsets,
                                          const I* indices, LongType* starts,
                                          LongType* lengths, LongType numOfClasses, void* outputBuf, const LongType* outputShape,
                                          const LongType* outputTads, const LongType* outputTadOffsets, LongType numIndices) {


   __shared__ LongType len, total;

   if (threadIdx.x == 0) {
     total = shape::sizeAt(inputShape, 0);
     len = shape::length(inputTads);
   }
   __syncthreads();

   for (auto idx = blockIdx.x; idx < total; idx += gridDim.x) {
     auto x = reinterpret_cast<T*>(inputBuf) + inputTadOffsets[idx];
     auto segment = indices[idx];
     auto z = reinterpret_cast<T*>(outputBuf) + outputTadOffsets[segment];
     auto start = starts[segment];
     auto finish = start + lengths[segment];
     if (lengths[segment] == 0) continue;
     for (auto e = threadIdx.x; e < len; e += blockDim.x) {
       auto xIndex = shape::getIndexOffset(e, inputTads);
       auto zIndex = shape::getIndexOffset(e, outputTads);
      math::atomics::sd_atomicAdd(&z[zIndex], x[xIndex]);
     }
   }
}
// -------------------------------------------------------------------------------------------------------------- //

template <typename T, typename I>
static void segmentSumFunctor_(LaunchContext* context, NDArray* input, NDArray* indices, NDArray* output) {
  auto stream = context->getCudaStream();
  LongType numClasses = indices->e<LongType>(indices->lengthOf() - 1) + 1;
  NDArray classesRangesLens = NDArrayFactory::create<LongType>('c', {numClasses}, context);
  NDArray classesRangesBegs = NDArrayFactory::create<LongType>('c', {numClasses}, context);

  classesRangesBegs.assign(indices->lengthOf());
  classesRangesLens.assign(0);

  fillUpSegments(indices, numClasses, classesRangesBegs, classesRangesLens);
  LongType* begins = reinterpret_cast<LongType*>(classesRangesBegs.specialBuffer());
  LongType* lengths = reinterpret_cast<LongType*>(classesRangesLens.specialBuffer());

  if (input->isVector() || input->isScalar()) {
    segmentSumLinearKernel<T, I><<<numClasses, input->lengthOf(), numClasses * 32 + 32, *stream>>>(
        input->specialBuffer(), input->specialShapeInfo(), begins, lengths, numClasses, output->specialBuffer(),
        output->specialShapeInfo());
    sd::DebugHelper::checkErrorCode(stream, "segmentSumLinearKernel failed");

  } else {
    LongType zero = 0;
    std::vector<LongType> *dimensions = ShapeUtils::evalDimsToExclude(input->rankOf(), 1,&zero);
    auto packX = ConstantTadHelper::getInstance().tadForDimensions(input->shapeInfo(), dimensions);
    auto packZ = ConstantTadHelper::getInstance().tadForDimensions(output->shapeInfo(), dimensions);
    auto inputTads = packX->specialShapeInfo();
    auto inputTadOffsets = packX->specialOffsets();
    auto outputTads = packZ->specialShapeInfo();
    auto outputTadOffsets = packZ->specialOffsets();
    dim3 segmentTadDims = segmentTad(input->sizeAt(0));
    segmentSumTadKernel<T, I><<<segmentTadDims.y,segmentTadDims.x,segmentTadDims.z, *stream>>>(
        input->specialBuffer(), input->specialShapeInfo(), inputTads, inputTadOffsets,
        reinterpret_cast<I*>(indices->specialBuffer()), begins, lengths, numClasses, output->specialBuffer(),
        output->specialShapeInfo(), outputTads, outputTadOffsets, indices->lengthOf());
    sd::DebugHelper::checkErrorCode(stream, "segmentSumTadKernel failed");

    delete dimensions;
  }
}
// -------------------------------------------------------------------------------------------------------------- //
void segmentSumFunctor(LaunchContext* context, NDArray* input, NDArray* indices, NDArray* output) {
  NDArray::prepareSpecialUse({output}, {input, indices});
  output->nullify();
  BUILD_DOUBLE_SELECTOR(input->dataType(), indices->dataType(), segmentSumFunctor_, (context, input, indices, output),
                        SD_NUMERIC_TYPES, SD_INDEXING_TYPES);
  NDArray::registerSpecialUse({output}, {input, indices});
}

// -------------------------------------------------------------------------------------------------------------- //
template <typename T, typename I>
static void unsortedSegmentSumFunctor_(LaunchContext* context, NDArray* input, NDArray* indices, LongType numOfClasses, NDArray* output) {
  auto stream = context->getCudaStream();
  NDArray classesRangesBegs = NDArrayFactory::create<LongType>('c', {numOfClasses}, context);
  NDArray classesRangesLens = NDArrayFactory::create<LongType>('c', {numOfClasses}, context);
  classesRangesBegs.assign(indices->lengthOf());
  classesRangesLens.assign(0);
  dim3 dims = getSegmentSumDims(numOfClasses,indices->lengthOf());
  fillUpSegments(indices, numOfClasses, classesRangesBegs, classesRangesLens);
  LongType* begins = reinterpret_cast<LongType*>(classesRangesBegs.specialBuffer());
  LongType* lengths = reinterpret_cast<LongType*>(classesRangesLens.specialBuffer());

  if (input->isVector() || input->isScalar()) {
    unsortedSegmentSumLinearKernel<T, I><<<dims.x, dims.y, dims.z, *stream>>>(
        input->specialBuffer(), input->specialShapeInfo(), indices->specialBuffer(), indices->specialShapeInfo(),
        begins, lengths, numOfClasses, output->specialBuffer(), output->specialShapeInfo());
        sd::DebugHelper::checkErrorCode(stream, "unsortedSegmentSumLinearKernel failed");

  } else {

    output->assign(0);
    LongType zero = 0;
    std::vector<LongType> *dimensions = ShapeUtils::evalDimsToExclude(input->rankOf(),1,&zero);
    auto packX = ConstantTadHelper::getInstance().tadForDimensions(input->shapeInfo(), dimensions);
    auto packZ = ConstantTadHelper::getInstance().tadForDimensions(output->shapeInfo(), dimensions);
    auto inputTads = packX->specialShapeInfo();
    auto inputTadOffsets = packX->specialOffsets();
    auto outputTads = packZ->specialShapeInfo();
    auto outputTadOffsets = packZ->specialOffsets();
    dim3 dims = segmentTad(input->sizeAt(0));
    segmentSumTadKernel<T, I><<<dims.x, dims.y, dims.z, *stream>>>(
        input->specialBuffer(), input->specialShapeInfo(), inputTads, inputTadOffsets,
        reinterpret_cast<I*>(indices->specialBuffer()), begins, lengths, numOfClasses, output->specialBuffer(),
        output->specialShapeInfo(), outputTads, outputTadOffsets, indices->lengthOf());
    sd::DebugHelper::checkErrorCode(stream, "segmentSumTadKernel failed");

    delete dimensions;
    dimensions = nullptr;
  }
}
// -------------------------------------------------------------------------------------------------------------- //
void unsortedSegmentSumFunctor(LaunchContext* context, NDArray* input, NDArray* indices, LongType numOfClasses,
                               NDArray* output) {
  NDArray::prepareSpecialUse({output}, {input, indices});
  output->nullify();
  BUILD_DOUBLE_SELECTOR(input->dataType(), indices->dataType(), unsortedSegmentSumFunctor_,
                        (context, input, indices, numOfClasses, output), SD_NUMERIC_TYPES, SD_INDEXING_TYPES);
  NDArray::registerSpecialUse({output}, {input, indices});
}

// -------------------------------------------------------------------------------------------------------------- //
// Backpropagate ops
// -------------------------------------------------------------------------------------------------------------- //
// Sorted sum backpropagate
template <typename T, typename I>
static SD_KERNEL void segmentSumBPLinearKernel(const void* inputBuf, const LongType* inputShape, const void* eps,
                                               const LongType* epsShape, const void* indicesBuf,
                                               const LongType* indicesShape, void* outputBuf,
                                               const LongType* outputShape) {
  auto x = reinterpret_cast<const T*>(inputBuf);
  auto y = reinterpret_cast<const I*>(indicesBuf);
  auto z = reinterpret_cast<T*>(outputBuf);
  auto gradOut = reinterpret_cast<const T*>(eps);
  __shared__ LongType xLen, gradLen;

  if (threadIdx.x == 0) {
    xLen = shape::length(inputShape);
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
    auto gradOffsetO = shape::getIndexOffset(classIndex, epsShape);

    z[zOffset] = gradOut[gradOffsetO];
  }
}
// -------------------------------------------------------------------------------------------------------------- //
template <typename T, typename I>
static SD_KERNEL void segmentSumBPTadKernel(const void* inputBuf, const LongType* inputShape, const void* eps,
                                            const LongType* epsShape, const void* indicesBuf,
                                            const LongType* indicesShape, void* outputBuf,
                                            const LongType* outputShape, const LongType* inputTad,
                                            const LongType* inputOffsets, const LongType* gradOutTad,
                                            const LongType* gradOutOffsets, const LongType* outTad,
                                            const LongType* outOffsets) {
  __shared__ const T* x;
  __shared__ const T* gradOut;
  __shared__ const I* y;
  __shared__ T* z;
  __shared__ LongType xLen, yLen, gradLen, currentLen;

  if (threadIdx.x == 0) {
    xLen = shape::length(inputShape);
    x = reinterpret_cast<const T*>(inputBuf);
    y = reinterpret_cast<const I*>(indicesBuf);
    z = reinterpret_cast<T*>(outputBuf);
    yLen = shape::length(indicesShape);
    gradOut = reinterpret_cast<const T*>(eps);
    gradLen = shape::length(epsShape);
    currentLen = shape::length(outTad);
  }
  __syncthreads();

  for (auto i = blockIdx.x; i < yLen; i += gridDim.x) {
    auto yIndex = shape::getIndexOffset(i, indicesShape);
    auto segment = y[yIndex];
    auto currentOut = z + outOffsets[i];
    auto outGrad = gradOut + gradOutOffsets[segment];

    for (auto e = threadIdx.x; e < currentLen; e += blockDim.x) {
      currentOut[e] = outGrad[e];
    }
  }
}
// -------------------------------------------------------------------------------------------------------------- //
template <typename T, typename I>
Status segmentSumFunctorBP_(LaunchContext* context, NDArray* input, NDArray* indices, NDArray* gradOut,
                                NDArray* output) {
  auto stream = context->getCudaStream();
  NDArray::prepareSpecialUse({output}, {input, indices, gradOut});
  if (input->isVector()  || input->isScalar()) {
    LongType loop_size = input->lengthOf();
    auto numOfClasses = gradOut->lengthOf();
    segmentSumBPLinearKernel<T, I><<<gradOut->lengthOf(), input->lengthOf(), 256, *stream>>>(
        input->specialBuffer(), input->specialShapeInfo(), gradOut->specialBuffer(), gradOut->specialShapeInfo(),
        indices->specialBuffer(), indices->specialShapeInfo(), output->specialBuffer(), output->specialShapeInfo());
    sd::DebugHelper::checkErrorCode(stream, "segmentSumBPLinearKernel failed");

  } else {
    LongType zero = 0;
    std::vector<LongType> *dimensions = ShapeUtils::evalDimsToExclude(input->rankOf(), 1,&zero);
    auto packX = ConstantTadHelper::getInstance().tadForDimensions(input->shapeInfo(), dimensions);
    auto packZ = ConstantTadHelper::getInstance().tadForDimensions(output->shapeInfo(), dimensions);
    auto packGradOut = ConstantTadHelper::getInstance().tadForDimensions(gradOut->shapeInfo(), dimensions);
    auto inputTads = packX->specialShapeInfo();
    auto inputTadOffsets = packX->specialOffsets();
    auto outputTads = packZ->specialShapeInfo();
    auto outputTadOffsets = packZ->specialOffsets();
    auto gradOutTads = packGradOut->specialShapeInfo();
    auto gradOutTadOffsets = packGradOut->specialOffsets();

    segmentSumBPTadKernel<T, I><<<gradOut->lengthOf(), input->lengthOf(), 256, *stream>>>(
        input->specialBuffer(), input->specialShapeInfo(), gradOut->specialBuffer(), gradOut->specialShapeInfo(),
        indices->specialBuffer(), indices->specialShapeInfo(), output->specialBuffer(), output->specialShapeInfo(),
        inputTads, inputTadOffsets, gradOutTads, gradOutTadOffsets, outputTads, outputTadOffsets);
    sd::DebugHelper::checkErrorCode(stream, "segmentSumBPTadKernel failed");

    delete dimensions;
  }
  NDArray::registerSpecialUse({output}, {input, indices, gradOut});
  return Status::OK;
}
// -------------------------------------------------------------------------------------------------------------- //

Status segmentSumFunctorBP(LaunchContext* context, NDArray* input, NDArray* indices, NDArray* gradOut,
                               NDArray* output) {
  NDArray::prepareSpecialUse({output}, {input, indices, gradOut});
  BUILD_DOUBLE_SELECTOR(output->dataType(), indices->dataType(), return segmentSumFunctorBP_,
                        (context, input, indices, gradOut, output), SD_FLOAT_TYPES, SD_INDEXING_TYPES);
  NDArray::registerSpecialUse({output}, {input, indices, gradOut});
}

template <typename T, typename I>
static Status unsortedSegmentSumFunctorBP_(LaunchContext* context, NDArray* input, NDArray* indices,
                                               NDArray* gradOut,
                                           LongType numOfClasses, NDArray* output) {
  auto stream = context->getCudaStream();
  NDArray::prepareSpecialUse({output}, {input, indices, gradOut});
  if (input->isVector()  || input->isScalar()) {
    LongType loop_size = input->lengthOf();
    auto numOfClasses = gradOut->lengthOf();
    segmentSumBPLinearKernel<T, I><<<gradOut->lengthOf(), input->lengthOf(), 256, *stream>>>(
        input->specialBuffer(), input->specialShapeInfo(), gradOut->specialBuffer(), gradOut->specialShapeInfo(),
        indices->specialBuffer(), indices->specialShapeInfo(), output->specialBuffer(), output->specialShapeInfo());
    sd::DebugHelper::checkErrorCode(stream, "segmentSumBPLinearKernel failed");

  } else {
    LongType zero = 0;
    std::vector<LongType> *dimensions = ShapeUtils::evalDimsToExclude(input->rankOf(), 1,&zero);
    auto packX = ConstantTadHelper::getInstance().tadForDimensions(input->shapeInfo(), dimensions);
    auto packZ = ConstantTadHelper::getInstance().tadForDimensions(output->shapeInfo(), dimensions);
    auto packGradOut = ConstantTadHelper::getInstance().tadForDimensions(gradOut->shapeInfo(), dimensions);
    auto inputTads = packX->specialShapeInfo();
    auto inputTadOffsets = packX->specialOffsets();
    auto outputTads = packZ->specialShapeInfo();
    auto outputTadOffsets = packZ->specialOffsets();
    auto gradOutTads = packGradOut->specialShapeInfo();
    auto gradOutTadOffsets = packGradOut->specialOffsets();

    segmentSumBPTadKernel<T, I><<<gradOut->lengthOf(), input->lengthOf(), 256, *stream>>>(
        input->specialBuffer(), input->specialShapeInfo(), gradOut->specialBuffer(), gradOut->specialShapeInfo(),
        indices->specialBuffer(), indices->specialShapeInfo(), output->specialBuffer(), output->specialShapeInfo(),
        inputTads, inputTadOffsets, gradOutTads, gradOutTadOffsets, outputTads, outputTadOffsets);
    sd::DebugHelper::checkErrorCode(stream, "segmentSumBPTadKernel failed");

    delete dimensions;
  }
  NDArray::registerSpecialUse({output}, {input, indices, gradOut});
  return Status::OK;
}
// -------------------------------------------------------------------------------------------------------------- //
Status unsortedSegmentSumFunctorBP(LaunchContext* context, NDArray* input, NDArray* indices, NDArray* gradOut,
                                   LongType numOfClasses, NDArray* output) {
  NDArray::prepareSpecialUse({output}, {input, indices, gradOut});
  BUILD_DOUBLE_SELECTOR(output->dataType(), indices->dataType(), return unsortedSegmentSumFunctorBP_,
                        (context, input, indices, gradOut, numOfClasses, output), SD_FLOAT_TYPES, SD_INDEXING_TYPES);
  NDArray::registerSpecialUse({output}, {input, indices, gradOut});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd

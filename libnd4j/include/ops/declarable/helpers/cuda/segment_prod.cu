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
// Segment Prod ops linear kernels
// -------------------------------------------------------------------------------------------------------------- //

template <typename T, typename I>
static SD_KERNEL void segmentProdLinearKernel(void* input, LongType const* inputShape, LongType* starts,
                                              LongType* lengths, LongType numOfClasses, void* output,
                                              LongType const* outputShape) {
  __shared__ LongType xLen, zLen;
  __shared__ T* x;
  __shared__ T* z;

  if (threadIdx.x == 0) {
    x = reinterpret_cast<T*>(input);
    z = reinterpret_cast<T*>(output);
    xLen = shape::length(inputShape);
    zLen = shape::length(outputShape);
  }
  __syncthreads();

  LongType inputCoords[SD_MAX_RANK];
  LongType outputCoords[SD_MAX_RANK];
  LongType xIndex;
  LongType zIndex;

  for (auto segment = blockIdx.x; segment < numOfClasses; segment += gridDim.x) {
    INDEX2COORDS(segment, shape::rank(outputShape), shape::shapeOf(outputShape), outputCoords);
    COORDS2INDEX(shape::rank(outputShape), shape::stride(outputShape), outputCoords, zIndex);
    if(zIndex >= zLen)
      continue;
    auto start = starts[segment];
    auto finish = start + lengths[segment];
    if (lengths[segment] == 0) {
      continue;
    }
    for (auto e = start + threadIdx.x; e < finish; e += blockDim.x) {
      INDEX2COORDS(e, shape::rank(inputShape), shape::shapeOf(inputShape), inputCoords);
      COORDS2INDEX(shape::rank(inputShape), shape::stride(inputShape), inputCoords, xIndex);
      if (xIndex >= xLen) return;
      math::atomics::sd_atomicMul(&z[zIndex], x[xIndex]);
    }
  }
}
// -------------------------------------------------------------------------------------------------------------- //
template <typename T, typename I>
static SD_KERNEL void unsortedSegmentProdLinearKernel(T* input, LongType const* inputShape, I* indices,
                                                      LongType const* indicesShape, LongType* starts, LongType* lengths,
                                                      LongType numOfClasses, T* output, LongType const* outputShape) {
  __shared__ LongType xLen, zLen;

  if (threadIdx.x == 0) {
    xLen = shape::length(inputShape);
    zLen = shape::length(outputShape);
  }
  __syncthreads();
  auto start = threadIdx.x + blockIdx.x * blockDim.x;
  auto step = blockDim.x * gridDim.x;

  LongType xCoords[SD_MAX_RANK];
  LongType yCoords[SD_MAX_RANK];
  LongType zCoords[SD_MAX_RANK];
  LongType xIndex;
  LongType yIndex;
  LongType zIndex;

  for (auto idx = start; idx < xLen; idx += step) {
    INDEX2COORDS(idx, shape::rank(inputShape), shape::shapeOf(inputShape), xCoords);
    COORDS2INDEX(shape::rank(inputShape), shape::stride(inputShape), xCoords, xIndex);

    INDEX2COORDS(idx, shape::rank(indicesShape), shape::shapeOf(indicesShape), yCoords);
    COORDS2INDEX(shape::rank(indicesShape), shape::stride(indicesShape), yCoords, yIndex);

    auto segment = indices[yIndex];

    INDEX2COORDS(segment, shape::rank(outputShape), shape::shapeOf(outputShape), zCoords);
    COORDS2INDEX(shape::rank(outputShape), shape::stride(outputShape), zCoords, zIndex);

    if (lengths[segment] == 0) {
      continue;
    }
    math::atomics::sd_atomicMul(&output[zIndex], input[xIndex]);
  }
}
// -------------------------------------------------------------------------------------------------------------- //
// SegmentProd kernel
template <typename T, typename I>
static SD_KERNEL void segmentProdTadKernel(void* inputBuf, LongType const* inputShape, LongType const* inputTads,
                                           LongType const* inputTadOffsets,
                                           I* indices, LongType* starts,
                                           LongType* lengths, LongType numOfClasses, void* outputBuf,
                                           LongType const* outputShape, LongType const* outputTads,
                                           LongType const* outputTadOffsets, LongType indicesLen) {

  if(blockIdx.x >= indicesLen)
    return;
  __shared__ LongType len, total;

  if (threadIdx.x == 0) {
    total = shape::sizeAt(inputShape, 0);
    len = shape::length(inputTads);
  }
  __syncthreads();

  LongType inputCoords[SD_MAX_RANK];
  LongType outputCoords[SD_MAX_RANK];
  LongType xIndex;
  LongType zIndex;

  for (auto idx = blockIdx.x; idx < total; idx += gridDim.x) {
    auto x = reinterpret_cast<T*>(inputBuf) + inputTadOffsets[idx];
    auto segment = indices[idx];
    auto z = reinterpret_cast<T*>(outputBuf) + outputTadOffsets[segment];
    auto start = starts[segment];
    auto finish = start + lengths[segment];
    if (lengths[segment] == 0) continue;
    for (auto e = threadIdx.x; e < len; e += blockDim.x) {
      INDEX2COORDS(e, shape::rank(inputTads), shape::shapeOf(inputTads), inputCoords);
      COORDS2INDEX(shape::rank(inputTads), shape::stride(inputTads), inputCoords, xIndex);
      INDEX2COORDS(e, shape::rank(outputTads), shape::shapeOf(outputTads), outputCoords);
      COORDS2INDEX(shape::rank(outputTads), shape::stride(outputTads), outputCoords, zIndex);
      math::atomics::sd_atomicMul(&z[zIndex], x[xIndex]);
    }
  }
}
// -------------------------------------------------------------------------------------------------------------- //

template <typename T, typename I>
static void segmentProdFunctor_(LaunchContext* context, NDArray* input, NDArray* indices, NDArray* output) {
  auto stream = context->getCudaStream();
  LongType numClasses = indices->e<LongType>(indices->lengthOf() - 1) + 1;
  NDArray classesRangesLens = NDArrayFactory::create<LongType>('c', {numClasses}, context);
  NDArray classesRangesBegs = NDArrayFactory::create<LongType>('c', {numClasses}, context);
  sd::LongType zero = 0;
  sd::LongType  one = 1;
  sd::LongType  len = indices->lengthOf();
  output->assign(one);
  classesRangesBegs.assign(len);
  classesRangesLens.assign(zero);

  fillUpSegments(indices, numClasses, classesRangesBegs, classesRangesLens);
  LongType* begins = reinterpret_cast<LongType*>(classesRangesBegs.specialBuffer());
  LongType* lengths = reinterpret_cast<LongType*>(classesRangesLens.specialBuffer());

  if (input->isVector()  || input->isScalar()) {
    dim3 launchDims = segmentDims(indices->lengthOf(),input->lengthOf());
    segmentProdLinearKernel<T, I><<<launchDims.y, launchDims.x, launchDims.z, *stream>>>(input->specialBuffer(), input->specialShapeInfo(), begins,
                                                                                         lengths, numClasses, output->specialBuffer(),
                                                                                         output->specialShapeInfo());
    sd::DebugHelper::checkErrorCode(stream, "segmentProdLinearKernel failed");

  } else {
    LongType zero = 0;
    std::vector<LongType> *dimensions = ShapeUtils::evalDimsToExclude(input->rankOf(), 1,&zero);
    auto packX = ConstantTadHelper::getInstance().tadForDimensions(input->shapeInfo(), dimensions);
    auto packZ = ConstantTadHelper::getInstance().tadForDimensions(output->shapeInfo(), dimensions);
    auto inputTads = packX->specialShapeInfo();
    auto inputTadOffsets = packX->specialOffsets();
    auto outputTads = packZ->specialShapeInfo();
    auto outputTadOffsets = packZ->specialOffsets();
    dim3 launchDims = segmentTad(input->lengthOf());
    segmentProdTadKernel<T, I><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(
        input->specialBuffer(), input->specialShapeInfo(), inputTads, inputTadOffsets,
        reinterpret_cast<I*>(indices->specialBuffer()), begins, lengths, numClasses, output->specialBuffer(),
        output->specialShapeInfo(), outputTads, outputTadOffsets, indices->lengthOf());
    sd::DebugHelper::checkErrorCode(stream, "segmentProdTadKernel failed");

    delete dimensions;
  }
}
// -------------------------------------------------------------------------------------------------------------- //
void segmentProdFunctor(LaunchContext* context, NDArray* input, NDArray* indices, NDArray* output) {
  NDArray::prepareSpecialUse({output}, {input, indices});
  BUILD_DOUBLE_SELECTOR(output->dataType(), indices->dataType(), segmentProdFunctor_, (context, input, indices, output),
                        SD_NUMERIC_TYPES, SD_INDEXING_TYPES);
  NDArray::registerSpecialUse({output}, {input, indices});
}

// -------------------------------------------------------------------------------------------------------------- //
template <typename T, typename I>
static void unsortedSegmentProdFunctor_(LaunchContext* context, NDArray* input, NDArray* indices, LongType numOfClasses, NDArray* output) {
  auto stream = context->getCudaStream();
  NDArray classesRangesBegs = NDArrayFactory::create<LongType>('c', {numOfClasses}, context);
  NDArray classesRangesLens = NDArrayFactory::create<LongType>('c', {numOfClasses}, context);
  sd::LongType zero = 0;
  sd::LongType  one = 1;
  sd::LongType  len = indices->lengthOf();
  classesRangesBegs.assign(len);
  classesRangesLens.assign(zero);
  dim3 dims = getFillUpSegmentsDims(numOfClasses,indices->lengthOf());
  fillUpSegments(indices, numOfClasses, classesRangesBegs, classesRangesLens);
  LongType* begins = reinterpret_cast<LongType*>(classesRangesBegs.specialBuffer());
  LongType* lengths = reinterpret_cast<LongType*>(classesRangesLens.specialBuffer());
  output->assign(one);

  dim3 launchDims = getLaunchDims("unsorted_segment_prod_2");
  if (input->isVector()) {
    unsortedSegmentProdLinearKernel<T, I><<<launchDims.y, launchDims.x, launchDims.z, *stream>>>(
        input->dataBuffer()->specialAsT<T>(), input->specialShapeInfo(), indices->dataBuffer()->specialAsT<I>(),
        indices->specialShapeInfo(), begins, lengths, numOfClasses, output->dataBuffer()->specialAsT<T>(),
        output->specialShapeInfo());
    sd::DebugHelper::checkErrorCode(stream, "unsortedSegmentProdLinearKernel failed");

  } else {
    LongType zero = 0;
    std::vector<LongType> *dimensions = ShapeUtils::evalDimsToExclude(input->rankOf(), 1,&zero);
    auto packX = ConstantTadHelper::getInstance().tadForDimensions(input->shapeInfo(), dimensions);
    auto packZ = ConstantTadHelper::getInstance().tadForDimensions(output->shapeInfo(), dimensions);
    auto inputTads = packX->specialShapeInfo();
    auto inputTadOffsets = packX->specialOffsets();
    auto outputTads = packZ->specialShapeInfo();
    auto outputTadOffsets = packZ->specialOffsets();
    dims.x = input->sizeAt(0);
    segmentProdTadKernel<T, I><<<launchDims.y, launchDims.x, launchDims.z, *stream>>>(
        input->specialBuffer(), input->specialShapeInfo(), inputTads, inputTadOffsets,
        reinterpret_cast<I*>(indices->specialBuffer()), begins, lengths, numOfClasses, output->specialBuffer(),
        output->specialShapeInfo(), outputTads, outputTadOffsets, indices->lengthOf());
    sd::DebugHelper::checkErrorCode(stream, "segmentProdTadKernel failed");

    delete dimensions;
  }
}
// -------------------------------------------------------------------------------------------------------------- //
void unsortedSegmentProdFunctor(LaunchContext* context, NDArray* input, NDArray* indices, LongType numOfClasses,
                                NDArray* output) {
  NDArray::prepareSpecialUse({output}, {input, indices});
  BUILD_DOUBLE_SELECTOR(input->dataType(), indices->dataType(), unsortedSegmentProdFunctor_,
                        (context, input, indices, numOfClasses, output), SD_NUMERIC_TYPES, SD_INDEXING_TYPES);
  NDArray::registerSpecialUse({output}, {input, indices});
}

// -------------------------------------------------------------------------------------------------------------- //
template <typename T, typename I>
static SD_KERNEL void segmentProdBPLinearKernel(void* inputBuf, LongType const* inputShape, void* forwardOutput,
                                                LongType const* forwardShape, void* eps, LongType const* epsShape, void* indicesBuf, LongType const* indicesShape, void* outputBuf,
                                                LongType const* outputShape) {
  __shared__ T* x;
  __shared__ T* gradIn;
  __shared__ T* gradOut;
  __shared__ I* y;
  __shared__ T* z;
  __shared__ LongType xLen, gradLen;

  if (threadIdx.x == 0) {
    xLen = shape::length(inputShape);
    x = reinterpret_cast<T*>(inputBuf);
    y = reinterpret_cast<I*>(indicesBuf);
    z = reinterpret_cast<T*>(outputBuf);
    gradIn = reinterpret_cast<T*>(forwardOutput);
    gradOut = reinterpret_cast<T*>(eps);
    gradLen = shape::length(epsShape);
  }
  __syncthreads();

  auto start = blockIdx.x * blockDim.x + threadIdx.x;
  auto step = gridDim.x * blockDim.x;

  LongType xCoords[SD_MAX_RANK];
  LongType yCoords[SD_MAX_RANK];
  LongType zCoords[SD_MAX_RANK];
  LongType gradICoords[SD_MAX_RANK];
  LongType gradOCoords[SD_MAX_RANK];
  LongType xOffset;
  LongType yOffset;
  LongType zOffset;
  LongType gradOffsetI;
  LongType gradOffsetO;

  for (auto e = start; e < xLen; e += step) {
    INDEX2COORDS(e, shape::rank(inputShape), shape::shapeOf(inputShape), xCoords);
    COORDS2INDEX(shape::rank(inputShape), shape::stride(inputShape), xCoords, xOffset);

    INDEX2COORDS(e, shape::rank(indicesShape), shape::shapeOf(indicesShape), yCoords);
    COORDS2INDEX(shape::rank(indicesShape), shape::stride(indicesShape), yCoords, yOffset);

    auto classIndex = y[yOffset];

    INDEX2COORDS(classIndex, shape::rank(forwardShape), shape::shapeOf(forwardShape), gradICoords);
    COORDS2INDEX(shape::rank(forwardShape), shape::stride(forwardShape), gradICoords, gradOffsetI);

    INDEX2COORDS(classIndex, shape::rank(epsShape), shape::shapeOf(epsShape), gradOCoords);
    COORDS2INDEX(shape::rank(epsShape), shape::stride(epsShape), gradOCoords, gradOffsetO);

    INDEX2COORDS(e, shape::rank(outputShape), shape::shapeOf(outputShape), zCoords);
    COORDS2INDEX(shape::rank(outputShape), shape::stride(outputShape), zCoords, zOffset);

    z[zOffset] = gradOut[gradOffsetO] * gradIn[gradOffsetI] / x[xOffset];
  }
}
// -------------------------------------------------------------------------------------------------------------- //
template <typename T, typename I>
static SD_KERNEL void segmentProdBPTadKernel(void* inputBuf, LongType const* inputShape, void* forwardOutput,
                                             LongType const* forwardShape, void* eps, LongType const* epsShape,
                                             void* indicesBuf, LongType const* indicesShape, void* outputBuf,
                                             LongType const* outputShape, LongType const* inputTad,
                                             LongType const* inputOffsets, LongType const* gradInTad,
                                             LongType const* gradInOffsets, LongType const* gradOutTad,
                                             LongType const* gradOutOffsets, LongType const* outTad,
                                             LongType const* outOffsets) {
  __shared__ T* x;
  __shared__ T* gradIn;
  __shared__ T* gradOut;
  __shared__ I* y;
  __shared__ T* z;
  __shared__ LongType xLen, yLen, gradLen, currentLen;

  if (threadIdx.x == 0) {
    xLen = shape::length(inputShape);
    x = reinterpret_cast<T*>(inputBuf);
    y = reinterpret_cast<I*>(indicesBuf);
    z = reinterpret_cast<T*>(outputBuf);
    yLen = shape::length(indicesShape);
    gradOut = reinterpret_cast<T*>(eps);
    gradIn = reinterpret_cast<T*>(forwardOutput);
    gradLen = shape::length(epsShape);
    currentLen = shape::length(outTad);
  }
  __syncthreads();

  for (auto i = blockIdx.x; i < yLen; i += gridDim.x) {
    LongType yCoords[SD_MAX_RANK];
    LongType yIndex;
    INDEX2COORDS(i, shape::rank(indicesShape), shape::shapeOf(indicesShape), yCoords);
    COORDS2INDEX(shape::rank(indicesShape), shape::shapeOf(indicesShape), yCoords, yIndex);
    auto segment = y[yIndex];
    T* current = x + inputOffsets[i];
    T* currentOut = z + outOffsets[i];
    T* in = gradIn + gradInOffsets[segment];
    T* outGrad = gradOut + gradOutOffsets[segment];

    for (auto e = threadIdx.x; e < currentLen; e += blockDim.x) {
      currentOut[e] = outGrad[e] * in[e] / current[e];
    }
  }
}

// -------------------------------------------------------------------------------------------------------------- //
template <typename T, typename I>
Status segmentProdFunctorBP_(LaunchContext* context, NDArray* input, NDArray* indices, NDArray* gradOut,
                                 NDArray* output) {
  auto stream = context->getCudaStream();
  auto outShape = gradOut->getShapeAsVector();

  NDArray tempRes(gradOut->ordering(), outShape, DataTypeUtils::fromT<T>(),
                  context);  //->shapeInfo(), context);
  segmentProdFunctor_<T, I>(context, input, indices, &tempRes);
  NDArray::prepareSpecialUse({output}, {input, indices, gradOut});
  if (input->isVector()) {
    LongType loopSize = input->lengthOf();
    auto numOfClasses = gradOut->lengthOf();  // indices->e<sd::LongType>(loop_size - 1);
    segmentProdBPLinearKernel<T, I><<<gradOut->lengthOf(), loopSize, 256, *stream>>>(
        input->specialBuffer(), input->specialShapeInfo(), tempRes.specialBuffer(), tempRes.specialShapeInfo(),
        gradOut->specialBuffer(), gradOut->specialShapeInfo(), indices->specialBuffer(), indices->specialShapeInfo(),
        output->specialBuffer(), output->specialShapeInfo());
    sd::DebugHelper::checkErrorCode(stream, "segmentProdBPLinearKernel failed");

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
    dim3 segmentBpTad2 = segmentBpTad(gradOut->lengthOf(),input->lengthOf());

    segmentProdBPTadKernel<T, I><<<segmentBpTad2.y,segmentBpTad2.x, segmentBpTad2.z, *stream>>>(
        input->specialBuffer(), input->specialShapeInfo(), tempRes.specialBuffer(), tempRes.specialShapeInfo(),
        gradOut->specialBuffer(), gradOut->specialShapeInfo(), indices->specialBuffer(), indices->specialShapeInfo(),
        output->specialBuffer(), output->specialShapeInfo(), inputTads, inputTadOffsets, gradInTads, gradInTadOffsets,
        gradOutTads, gradOutTadOffsets, outputTads, outputTadOffsets);
    sd::DebugHelper::checkErrorCode(stream, "segmentProdBPTadKernel failed");

    delete dimensions;
  }
  NDArray::registerSpecialUse({output}, {input, indices, gradOut});
  return Status::OK;
}

// -------------------------------------------------------------------------------------------------------------- //

Status segmentProdFunctorBP(LaunchContext* context, NDArray* input, NDArray* indices, NDArray* gradOut,
                                NDArray* output) {
  NDArray::prepareSpecialUse({output}, {input, indices, gradOut});
  BUILD_DOUBLE_SELECTOR(output->dataType(), indices->dataType(), return segmentProdFunctorBP_,
                        (context, input, indices, gradOut, output), SD_FLOAT_TYPES, SD_INDEXING_TYPES);
  NDArray::registerSpecialUse({output}, {input, indices, gradOut});
}

// -------------------------------------------------------------------------------------------------------------- //

template <typename T, typename I>
static Status unsortedSegmentProdFunctorBP_(LaunchContext* context, NDArray* input, NDArray* indices,
                                                NDArray* gradOut,
                                            LongType numOfClasses, NDArray* output) {
  auto stream = context->getCudaStream();
  auto outShape = gradOut->getShapeAsVector();
  NDArray tempRes(gradOut->ordering(),outShape, DataTypeUtils::fromT<T>(),
                  context);
  unsortedSegmentProdFunctor_<T, I>(context, input, indices, numOfClasses, &tempRes);
  NDArray::prepareSpecialUse({output}, {input, indices, gradOut});
  if (input->isVector()) {
    LongType loopSize = input->lengthOf();
    auto numOfClasses = gradOut->lengthOf();
    dim3 segmentBpTad2 = segmentBpDims(gradOut->lengthOf(),input->lengthOf());
    segmentProdBPLinearKernel<T, I><<<segmentBpTad2.y, segmentBpTad2.x,segmentBpTad2.z, *stream>>>(
        input->specialBuffer(), input->specialShapeInfo(), tempRes.specialBuffer(), tempRes.specialShapeInfo(),
        gradOut->specialBuffer(), gradOut->specialShapeInfo(), indices->specialBuffer(), indices->specialShapeInfo(),
        output->specialBuffer(), output->specialShapeInfo());
    sd::DebugHelper::checkErrorCode(stream, "segmentProdBPLinearKernel failed");

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
    dim3 segmentBpTad2 = segmentBpTad(gradOut->lengthOf(),input->lengthOf());
    segmentProdBPTadKernel<T, I><<<indices->lengthOf(), input->lengthOf(), 256, *stream>>>(
        input->specialBuffer(), input->specialShapeInfo(), tempRes.specialBuffer(), tempRes.specialShapeInfo(),
        gradOut->specialBuffer(), gradOut->specialShapeInfo(), indices->specialBuffer(), indices->specialShapeInfo(),
        output->specialBuffer(), output->specialShapeInfo(), inputTads, inputTadOffsets, gradInTads, gradInTadOffsets,
        gradOutTads, gradOutTadOffsets, outputTads, outputTadOffsets);
    sd::DebugHelper::checkErrorCode(stream, "segmentProdBPTadKernel failed");

    delete dimensions;
  }
  NDArray::registerSpecialUse({output}, {input, indices, gradOut});
  return Status::OK;
}

// -------------------------------------------------------------------------------------------------------------- //
Status unsortedSegmentProdFunctorBP(LaunchContext* context, NDArray* input, NDArray* indices, NDArray* gradOut,
                                    LongType numOfClasses, NDArray* output) {
  NDArray::prepareSpecialUse({output}, {input, indices, gradOut});
  BUILD_DOUBLE_SELECTOR(output->dataType(), indices->dataType(), return unsortedSegmentProdFunctorBP_,
                        (context, input, indices, gradOut, numOfClasses, output), SD_FLOAT_TYPES, SD_INDEXING_TYPES);
  NDArray::registerSpecialUse({output}, {input, indices, gradOut});
}

// -------------------------------------------------------------------------------------------------------------- //

}  // namespace helpers
}  // namespace ops
}  // namespace sd

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
static SD_KERNEL void segmentMeanLinearKernel(void* input, LongType const* inputShape, LongType* indices,
                                              LongType* lengths, LongType numOfClasses, void* output,
                                              LongType const* outputShape) {
  __shared__ T* val;
  __shared__ LongType xLen, zLen, zIndex;
  __shared__ T* x;
  __shared__ T* z;
  __shared__ LongType threadsPerSegment, start, finish;

  auto segment = blockIdx.x;
  if (threadIdx.x == 0) {
    x = reinterpret_cast<T*>(input);
    z = reinterpret_cast<T*>(output);
    extern __shared__ unsigned char shmem[];
    val = reinterpret_cast<T*>(shmem);
    xLen = shape::length(inputShape);
    zLen = shape::length(outputShape);

    if (segment < numOfClasses) {
      LongType outputCoords[SD_MAX_RANK];
      LongType inputCoords[SD_MAX_RANK];
      LongType xOffset;
      LongType zOffset;

      INDEX2COORDS(segment, shape::rank(outputShape), shape::shapeOf(outputShape), outputCoords);
      COORDS2INDEX(shape::rank(outputShape), shape::stride(outputShape), outputCoords, zIndex);
      start = indices[segment];
      finish = start + lengths[segment];
      INDEX2COORDS(start, shape::rank(inputShape), shape::shapeOf(inputShape), inputCoords);
      COORDS2INDEX(shape::rank(inputShape), shape::stride(inputShape), inputCoords, xOffset);
      if (lengths[segment] > 0)
        z[zIndex] = T(x[xOffset] / T(lengths[segment]));
      else
        z[zIndex] = 0;
    }
    val[segment] = z[zIndex];
  }
  __syncthreads();

  for (auto e = start + threadIdx.x + 1; e < finish; e += blockDim.x) {
    LongType inputCoords[SD_MAX_RANK];
    LongType xOffset;
    INDEX2COORDS(e, shape::rank(inputShape), shape::shapeOf(inputShape), inputCoords);
    COORDS2INDEX(shape::rank(inputShape), shape::stride(inputShape), inputCoords, xOffset);
    math::atomics::sd_atomicAdd(&z[zIndex], T(x[xOffset] / static_cast<T>(lengths[segment])));
  }
}
// -------------------------------------------------------------------------------------------------------------- //
template <typename T, typename I>
static SD_KERNEL void unsortedSegmentMeanLinearKernel(void* input, LongType const* inputShape, void* indices,
                                                      LongType const* indicesShape, LongType* starts, LongType* lengths,
                                                      LongType numOfClasses, void* output,
                                                      LongType const* outputShape) {
  __shared__ LongType xLen, zLen, zIndex;
  __shared__ T* x;
  __shared__ T* z;
  __shared__ I* y;
  auto segment = blockIdx.x;
  if (threadIdx.x == 0) {
    x = reinterpret_cast<T*>(input);
    z = reinterpret_cast<T*>(output);
    y = reinterpret_cast<I*>(indices);
    xLen = shape::length(inputShape);
    zLen = shape::length(outputShape);

    LongType outputCoords[SD_MAX_RANK];
    LongType inputCoords[SD_MAX_RANK];
    LongType xOffset;
    LongType zOffset;

    INDEX2COORDS(segment, shape::rank(outputShape), shape::shapeOf(outputShape), outputCoords);
    COORDS2INDEX(shape::rank(outputShape), shape::stride(outputShape), outputCoords, zIndex);
    INDEX2COORDS(starts[segment], shape::rank(inputShape), shape::shapeOf(inputShape), inputCoords);
    COORDS2INDEX(shape::rank(inputShape), shape::stride(inputShape), inputCoords, xOffset);

    if (lengths[segment] > 0)
      z[zIndex] = T(x[xOffset] / T(lengths[segment]));
    else
      z[zIndex] = 0;
  }
  __syncthreads();
  if (lengths[segment] > 0)
    for (auto e = threadIdx.x; e < xLen; e += blockDim.x) {
      LongType inputCoords[SD_MAX_RANK];
      LongType xOffset;
      LongType yIndex;

      INDEX2COORDS(e, shape::rank(inputShape), shape::shapeOf(inputShape), inputCoords);
      COORDS2INDEX(shape::rank(inputShape), shape::stride(inputShape), inputCoords, xOffset);
      INDEX2COORDS(e, shape::rank(indicesShape), shape::shapeOf(indicesShape), inputCoords);
      COORDS2INDEX(shape::rank(indicesShape), shape::stride(indicesShape), inputCoords, yIndex);

      if (y[yIndex] == segment && e != starts[segment]) {
        math::atomics::sd_atomicAdd(&z[zIndex], T(x[xOffset] / T(lengths[segment])));
      }
    }
}
// -------------------------------------------------------------------------------------------------------------- //
// SegmentMean kernel
template <typename T, typename I>
static SD_KERNEL void segmentMeanTadKernel(void* inputBuf, LongType const* inputShape, LongType const* inputTads,
                                           LongType const* inputTadOffsets,
                                           I* indices, LongType* starts,
                                           LongType* lengths, LongType numOfClasses, void* outputBuf,
                                           LongType const* outputShape, LongType const* outputTads,
                                           LongType const* outputTadOffsets, LongType indicesLen) {
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
    auto x = reinterpret_cast<T*>(inputBuf) + inputTadOffsets[idx];
    if (blockIdx.x == start) {
      for (auto e = threadIdx.x; e < len; e += blockDim.x) {
        LongType xCoords[SD_MAX_RANK];
        LongType zCoords[SD_MAX_RANK];
        LongType xIndex;
        LongType zIndex;

        INDEX2COORDS(e, shape::rank(inputTads), shape::shapeOf(inputTads), xCoords);
        COORDS2INDEX(shape::rank(inputTads), shape::stride(inputTads), xCoords, xIndex);
        INDEX2COORDS(e, shape::rank(outputTads), shape::shapeOf(outputTads), zCoords);
        COORDS2INDEX(shape::rank(outputTads), shape::stride(outputTads), zCoords, zIndex);

        math::atomics::sd_atomicAdd(&z[zIndex], T(x[xIndex] / lengths[segment]));
      }
    } else {
      for (auto e = threadIdx.x; e < len; e += blockDim.x) {
        LongType xCoords[SD_MAX_RANK];
        LongType zCoords[SD_MAX_RANK];
        LongType xIndex;
        LongType zIndex;

        INDEX2COORDS(e, shape::rank(inputTads), shape::shapeOf(inputTads), xCoords);
        COORDS2INDEX(shape::rank(inputTads), shape::stride(inputTads), xCoords, xIndex);
        INDEX2COORDS(e, shape::rank(outputTads), shape::shapeOf(outputTads), zCoords);
        COORDS2INDEX(shape::rank(outputTads), shape::stride(outputTads), zCoords, zIndex);

        if (lengths[segment]) math::atomics::sd_atomicAdd(&z[zIndex], T(x[xIndex] / lengths[segment]));
      }
    }
  }
}
// -------------------------------------------------------------------------------------------------------------- //
// segment mean
template <typename T, typename I>
static void segmentMeanFunctor_(LaunchContext* context, NDArray* input, NDArray* indices, NDArray* output) {
  auto stream = context->getCudaStream();
  LongType numClasses = indices->e<LongType>(indices->lengthOf() - 1) + 1;
  NDArray classesRangesLens = NDArrayFactory::create<LongType>('c', {numClasses}, context);
  NDArray classesRangesBegs = NDArrayFactory::create<LongType>('c', {numClasses}, context);

  int zero2 = 0;
  sd::LongType len = indices->lengthOf();
  classesRangesBegs.assign(len);
  classesRangesLens.assign(zero2);
  NDArray::prepareSpecialUse({output}, {input, indices});
  LongType* begins = reinterpret_cast<LongType*>(classesRangesBegs.specialBuffer());
  LongType* lengths = reinterpret_cast<LongType*>(classesRangesLens.specialBuffer());
  fillUpSegments(indices, numClasses, classesRangesBegs, classesRangesLens);

  if (input->isVector()  || input->isScalar()) {
    dim3 launchDims = segmentDims(numClasses,input->lengthOf());
    segmentMeanLinearKernel<T, I><<<launchDims.y, launchDims.x, launchDims.z, *stream>>>(
        input->specialBuffer(), input->specialShapeInfo(), begins, lengths, numClasses, output->specialBuffer(),
        output->specialShapeInfo());
    sd::DebugHelper::checkErrorCode(stream, "segmentMeanLinearKernel failed");

  } else {
    LongType zero = 0;
    std::vector<LongType> *dimensions = ShapeUtils::evalDimsToExclude(input->rankOf(), 1,&zero);
    auto packX = ConstantTadHelper::getInstance().tadForDimensions(input->shapeInfo(), dimensions);
    auto packZ = ConstantTadHelper::getInstance().tadForDimensions(output->shapeInfo(), dimensions);
    auto inputTads = packX->specialShapeInfo();
    auto inputTadOffsets = packX->specialOffsets();
    auto outputTads = packZ->specialShapeInfo();
    auto outputTadOffsets = packZ->specialOffsets();
    dim3 launchDims = segmentTad(input->sizeAt(0));
    segmentMeanTadKernel<T, I><<<launchDims.y, launchDims.x, launchDims.z, *stream>>>(
        input->specialBuffer(), input->specialShapeInfo(), inputTads, inputTadOffsets,
        reinterpret_cast<I*>(indices->specialBuffer()), begins, lengths, numClasses, output->specialBuffer(),
        output->specialShapeInfo(), outputTads, outputTadOffsets,indices->lengthOf());
    sd::DebugHelper::checkErrorCode(stream, "segmentMeanTadKernel failed");

    delete dimensions;
  }
  NDArray::registerSpecialUse({output}, {input, indices});
}
// -------------------------------------------------------------------------------------------------------------- //
void segmentMeanFunctor(LaunchContext* context, NDArray* input, NDArray* indices, NDArray* output) {
  NDArray::prepareSpecialUse({output}, {input, indices});
  BUILD_DOUBLE_SELECTOR(output->dataType(), indices->dataType(), segmentMeanFunctor_, (context, input, indices, output),
                        SD_NUMERIC_TYPES, SD_INDEXING_TYPES);
  NDArray::registerSpecialUse({output}, {input, indices});
}

// -------------------------------------------------------------------------------------------------------------- //
template <typename T, typename I>
static void unsortedSegmentMeanFunctor_(LaunchContext* context, NDArray* input, NDArray* indices, LongType numOfClasses, NDArray* output) {
  auto stream = context->getCudaStream();

  NDArray classesRangesBegs = NDArrayFactory::create<LongType>('c', {numOfClasses}, context);
  NDArray classesRangesLens = NDArrayFactory::create<LongType>('c', {numOfClasses}, context);

  int zero2 = 0;
  sd::LongType len = indices->lengthOf();
  classesRangesBegs.assign(len);
  classesRangesLens.assign(zero2);
  dim3 dims = getFillUpSegmentsDims(numOfClasses, indices->lengthOf());
  fillUpSegments(indices, numOfClasses, classesRangesBegs, classesRangesLens);
  LongType* begins = reinterpret_cast<LongType*>(classesRangesBegs.specialBuffer());
  LongType* lengths = reinterpret_cast<LongType*>(classesRangesLens.specialBuffer());

  if (input->isVector()  || input->isScalar()) {
    unsortedSegmentMeanLinearKernel<T, I><<<dims.x, dims.y, dims.z, *stream>>>(
        input->specialBuffer(), input->specialShapeInfo(), indices->specialBuffer(), indices->specialShapeInfo(),
        begins, lengths, numOfClasses, output->specialBuffer(), output->specialShapeInfo());
    sd::DebugHelper::checkErrorCode(stream, "unsortedSegmentMeanLinearKernel failed");

  } else {
    LongType zero = 0;
    output->assign(zero);
    std::vector<LongType> *dimensions = ShapeUtils::evalDimsToExclude(input->rankOf(), 1,&zero);
    auto packX = ConstantTadHelper::getInstance().tadForDimensions(input->shapeInfo(), dimensions);
    auto packZ = ConstantTadHelper::getInstance().tadForDimensions(output->shapeInfo(), dimensions);
    LongType const* inputTads = packX->specialShapeInfo();
    LongType const* inputTadOffsets = packX->specialOffsets();
    LongType const* outputTads = packZ->specialShapeInfo();
    LongType const* outputTadOffsets = packZ->specialOffsets();
    dims.x = input->sizeAt(0);
    segmentMeanTadKernel<T, I><<<dims.x, dims.y, dims.z, *stream>>>(
        input->specialBuffer(), input->specialShapeInfo(), inputTads, inputTadOffsets,
        reinterpret_cast<I*>(indices->specialBuffer()), begins, lengths, numOfClasses, output->specialBuffer(),
        output->specialShapeInfo(), outputTads, outputTadOffsets, indices->lengthOf());
    sd::DebugHelper::checkErrorCode(stream, "segmentMeanTadKernel failed");

    delete dimensions;
  }
}
// -------------------------------------------------------------------------------------------------------------- //
void unsortedSegmentMeanFunctor(LaunchContext* context, NDArray* input, NDArray* indices, LongType numOfClasses,
                                NDArray* output) {
  NDArray::prepareSpecialUse({output}, {input, indices});
  BUILD_DOUBLE_SELECTOR(input->dataType(), indices->dataType(), unsortedSegmentMeanFunctor_,
                        (context, input, indices, numOfClasses, output), SD_NUMERIC_TYPES, SD_INDEXING_TYPES);
  NDArray::registerSpecialUse({output}, {input, indices});
}

// -------------------------------------------------------------------------------------------------------------- //
template <typename T, typename I>
static SD_KERNEL void segmentMeanBPLinearKernel(void* inputBuf, LongType const* inputShape, void* eps,
                                                LongType const* epsShape, void* indicesBuf,
                                                LongType const* indicesShape, LongType* lengths, void* outputBuf,
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
    gradOut = reinterpret_cast<T*>(eps);
    gradLen = shape::length(epsShape);
  }
  __syncthreads();

  auto start = blockIdx.x * blockDim.x + threadIdx.x;
  auto step = gridDim.x * blockDim.x;

  for (auto e = start; e < xLen; e += step) {
    LongType zOffset, xOffset, yOffset, gradOffsetO;
    sd::LongType zCoords[SD_MAX_RANK], xCoords[SD_MAX_RANK], yCoords[SD_MAX_RANK], gradCoords[SD_MAX_RANK];

    INDEX2COORDS(e, shape::rank(outputShape), shape::shapeOf(outputShape), zCoords);
    COORDS2INDEX(shape::rank(outputShape), shape::stride(outputShape), zCoords, zOffset);

    INDEX2COORDS(e, shape::rank(inputShape), shape::shapeOf(inputShape), xCoords);
    COORDS2INDEX(shape::rank(inputShape), shape::stride(inputShape), xCoords, xOffset);

    INDEX2COORDS(e, shape::rank(indicesShape), shape::shapeOf(indicesShape), yCoords);
    COORDS2INDEX(shape::rank(indicesShape), shape::stride(indicesShape), yCoords, yOffset);

    auto classIndex = y[yOffset];

    INDEX2COORDS(classIndex, shape::rank(epsShape), shape::shapeOf(epsShape), gradCoords);
    COORDS2INDEX(shape::rank(epsShape), shape::stride(epsShape), gradCoords, gradOffsetO);

    z[zOffset] = T(gradOut[gradOffsetO] / float(lengths[classIndex]));
  }
}
// -------------------------------------------------------------------------------------------------------------- //
template <typename T, typename I>
static SD_KERNEL void segmentMeanBPTadKernel(void* inputBuf, LongType const* inputShape, void* eps,
                                             LongType const* epsShape, void* indicesBuf, LongType const* indicesShape,
                                             LongType* lengths, void* outputBuf, LongType const* outputShape,
                                             LongType const* inputTad, LongType const* inputOffsets,
                                             LongType const* gradOutTad, LongType const* gradOutOffsets,
                                             LongType const* outTad, LongType const* outOffsets) {
  __shared__ T* x;
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
    gradLen = shape::length(epsShape);
    currentLen = shape::length(outTad);
  }
  __syncthreads();

  for (auto i = blockIdx.x; i < yLen; i += gridDim.x) {
    auto segment = y[i];
    T* currentOut = z + outOffsets[i];
    T* outGrad = gradOut + gradOutOffsets[segment];

    for (auto e = threadIdx.x; e < currentLen; e += blockDim.x) {
      sd::LongType zCoords[SD_MAX_RANK];
      sd::LongType gradCoords[SD_MAX_RANK];
      sd::LongType zIndex;
      sd::LongType gradIndex;

      INDEX2COORDS(e, shape::rank(outTad), shape::shapeOf(outTad), zCoords);
      COORDS2INDEX(shape::rank(outTad), shape::stride(outTad), zCoords, zIndex);
      INDEX2COORDS(e, shape::rank(gradOutTad), shape::shapeOf(gradOutTad), gradCoords);
      COORDS2INDEX(shape::rank(gradOutTad), shape::stride(gradOutTad), gradCoords, gradIndex);

      if (lengths[segment] > 0) currentOut[zIndex] = T(outGrad[gradIndex] / float(lengths[segment]));
    }
  }
}
// -------------------------------------------------------------------------------------------------------------- //
// backrop for mean
template <typename T, typename I>
Status segmentMeanFunctorBP_(LaunchContext* context, NDArray* input, NDArray* indices, NDArray* gradOut,
                                 NDArray* output) {
  auto stream = context->getCudaStream();
  NDArray::prepareSpecialUse({output}, {input, indices, gradOut});
  auto numClasses = indices->e<LongType>(indices->lengthOf() - 1) + 1;
  NDArray classesRangesLens = NDArrayFactory::create<LongType>('c', {numClasses}, context);
  NDArray classesRangesBegs = NDArrayFactory::create<LongType>('c', {numClasses}, context);
  sd::LongType zero2 = 0;
  sd::LongType len = indices->lengthOf();
  classesRangesBegs.assign(zero2);
  classesRangesLens.assign(len);
  fillUpSegments(indices, numClasses, classesRangesBegs, classesRangesLens);
  LongType* begins = reinterpret_cast<LongType*>(classesRangesBegs.specialBuffer());
  LongType* lengths = reinterpret_cast<LongType*>(classesRangesLens.specialBuffer());

  if (input->isVector()  || input->isScalar()) {
    LongType loop_size = input->lengthOf();
    auto numOfClasses = gradOut->lengthOf();  // indices->e<sd::LongType>(loop_size - 1);
    dim3 segmentBpDims2 = segmentBpDims(gradOut->lengthOf(),input->lengthOf());
    segmentMeanBPLinearKernel<T, I><<<segmentBpDims2.y, segmentBpDims2.x, segmentBpDims2.z, *stream>>>(
        input->specialBuffer(), input->specialShapeInfo(), gradOut->specialBuffer(), gradOut->specialShapeInfo(),
        indices->specialBuffer(), indices->specialShapeInfo(), lengths, output->specialBuffer(),
        output->specialShapeInfo());
        sd::DebugHelper::checkErrorCode(stream, "segmentMeanBPLinearKernel failed");

  } else {
    LongType zero = 0;
    std::vector<LongType> *dimensions = ShapeUtils::evalDimsToExclude(input->rankOf(), 1,&zero);
    auto packX = ConstantTadHelper::getInstance().tadForDimensions(input->shapeInfo(), dimensions);
    auto packZ = ConstantTadHelper::getInstance().tadForDimensions(output->shapeInfo(), dimensions);
    auto packGradOut = ConstantTadHelper::getInstance().tadForDimensions(gradOut->shapeInfo(), dimensions);
    LongType const* inputTads = packX->specialShapeInfo();
    LongType const* inputTadOffsets = packX->specialOffsets();
    LongType const* outputTads = packZ->specialShapeInfo();
    LongType const* outputTadOffsets = packZ->specialOffsets();
    LongType const* gradOutTads = packGradOut->specialShapeInfo();
    LongType const* gradOutTadOffsets = packGradOut->specialOffsets();
    dim3 segmentBpTad2 = segmentBpTad(indices->lengthOf(),input->lengthOf());

    segmentMeanBPTadKernel<T, I><<<segmentBpTad2.y, segmentBpTad2.x, segmentBpTad2.z, *stream>>>(
        input->specialBuffer(), input->specialShapeInfo(), gradOut->specialBuffer(), gradOut->specialShapeInfo(),
        indices->specialBuffer(), indices->specialShapeInfo(), lengths, output->specialBuffer(),
        output->specialShapeInfo(), inputTads, inputTadOffsets, gradOutTads, gradOutTadOffsets, outputTads,
        outputTadOffsets);
    sd::DebugHelper::checkErrorCode(stream, "segmentMeanBPTadKernel failed");

    delete dimensions;
  }
  NDArray::registerSpecialUse({output}, {input, indices, gradOut});
  return Status::OK;
}
// -------------------------------------------------------------------------------------------------------------- //
// segmen mean bp main
Status segmentMeanFunctorBP(LaunchContext* context, NDArray* input, NDArray* indices, NDArray* gradOut,
                                NDArray* output) {
  NDArray::prepareSpecialUse({output}, {input, indices, gradOut});
  BUILD_DOUBLE_SELECTOR(output->dataType(), indices->dataType(), return segmentMeanFunctorBP_,
                        (context, input, indices, gradOut, output), SD_FLOAT_TYPES, SD_INDEXING_TYPES);
  NDArray::registerSpecialUse({output}, {input, indices, gradOut});
}
// -------------------------------------------------------------------------------------------------------------- //

template <typename T, typename I>
static Status unsortedSegmentMeanFunctorBP_(LaunchContext* context, NDArray* input, NDArray* indices,
                                                NDArray* gradOut,
                                            LongType numOfClasses, NDArray* output) {
  auto stream = context->getCudaStream();
  NDArray::prepareSpecialUse({output}, {input, indices, gradOut});
  auto numClasses = indices->e<LongType>(indices->lengthOf() - 1) + 1;
  NDArray classesRangesLens = NDArrayFactory::create<LongType>('c', {numClasses}, context);
  NDArray classesRangesBegs = NDArrayFactory::create<LongType>('c', {numClasses}, context);

  sd::LongType zero2 = 0;
  sd::LongType len = indices->lengthOf();
  classesRangesBegs.assign(zero2);
  classesRangesLens.assign(len);
  fillUpSegments(indices, numClasses, classesRangesBegs, classesRangesLens);
  LongType* begins = reinterpret_cast<LongType*>(classesRangesBegs.specialBuffer());
  LongType* lengths = reinterpret_cast<LongType*>(classesRangesLens.specialBuffer());

  if (input->isVector()  || input->isScalar()) {
    LongType loop_size = input->lengthOf();
    auto numOfClasses = gradOut->lengthOf();
    dim3 segmentBpDims2 = segmentBpDims(gradOut->lengthOf(),input->lengthOf());
    segmentMeanBPLinearKernel<T, I><<<segmentBpDims2.y,segmentBpDims2.x,segmentBpDims2.z, *stream>>>(
        input->specialBuffer(), input->specialShapeInfo(), gradOut->specialBuffer(), gradOut->specialShapeInfo(),
        indices->specialBuffer(), indices->specialShapeInfo(), lengths, output->specialBuffer(),
        output->specialShapeInfo());
    sd::DebugHelper::checkErrorCode(stream, "segmentMeanBPLinearKernel failed");


  } else {
    LongType zero = 0;
    std::vector<LongType> *dimensions = ShapeUtils::evalDimsToExclude(input->rankOf(),1, &zero);
    auto packX = ConstantTadHelper::getInstance().tadForDimensions(input->shapeInfo(), dimensions);
    auto packZ = ConstantTadHelper::getInstance().tadForDimensions(output->shapeInfo(), dimensions);

    auto packGradOut = ConstantTadHelper::getInstance().tadForDimensions(gradOut->shapeInfo(), dimensions);
    LongType const* inputTads = packX->specialShapeInfo();
    LongType const* inputTadOffsets = packX->specialOffsets();
    LongType const* outputTads = packZ->specialShapeInfo();
    LongType const* outputTadOffsets = packZ->specialOffsets();
    LongType const* gradOutTads = packGradOut->specialShapeInfo();
    LongType const* gradOutTadOffsets = packGradOut->specialOffsets();
    dim3 segmentBpTad2 = segmentBpTad(indices->lengthOf(),input->lengthOf());

    segmentMeanBPTadKernel<T, I><<<segmentBpTad2.y,segmentBpTad2.x, segmentBpTad2.z, *stream>>>(
        input->specialBuffer(), input->specialShapeInfo(), gradOut->specialBuffer(), gradOut->specialShapeInfo(),
        indices->specialBuffer(), indices->specialShapeInfo(), lengths, output->specialBuffer(),
        output->specialShapeInfo(), inputTads, inputTadOffsets, gradOutTads, gradOutTadOffsets, outputTads,
        outputTadOffsets);
    sd::DebugHelper::checkErrorCode(stream, "segmentMeanBPTadKernel failed");

    delete dimensions;
  }
  NDArray::registerSpecialUse({output}, {input, indices, gradOut});
  return Status::OK;
}
// -------------------------------------------------------------------------------------------------------------- //
Status unsortedSegmentMeanFunctorBP(LaunchContext* context, NDArray* input, NDArray* indices, NDArray* gradOut,
                                    LongType numOfClasses, NDArray* output) {
  NDArray::prepareSpecialUse({output}, {input, indices, gradOut});
  BUILD_DOUBLE_SELECTOR(output->dataType(), indices->dataType(), return unsortedSegmentMeanFunctorBP_,
                        (context, input, indices, gradOut, numOfClasses, output), SD_FLOAT_TYPES, SD_INDEXING_TYPES);
  NDArray::registerSpecialUse({output}, {input, indices, gradOut});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd

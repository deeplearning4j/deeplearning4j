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

  // Shared memory for caching shape, stride, and rank information
  __shared__ LongType inputRank;
  __shared__ const LongType* inputShapePtr;
  __shared__ const LongType* inputStridePtr;

  __shared__ LongType outputRank;
  __shared__ const LongType* outputShapePtr;
  __shared__ const LongType* outputStridePtr;

  // Shared memory for pointers and lengths initialized by thread 0
  __shared__ T* x;
  __shared__ T* z;
  __shared__ LongType xLen;
  __shared__ LongType zLen;

  if (threadIdx.x == 0) {
    // Cache rank, shape, and stride for inputShape
    inputRank = shape::rank(inputShape);
    inputShapePtr = shape::shapeOf(inputShape);
    inputStridePtr = shape::stride(inputShape);

    // Cache rank, shape, and stride for outputShape
    outputRank = shape::rank(outputShape);
    outputShapePtr = shape::shapeOf(outputShape);
    outputStridePtr = shape::stride(outputShape);

    // Cache lengths
    xLen = shape::length(inputShape);
    zLen = shape::length(outputShape);

    // Initialize pointers
    x = reinterpret_cast<T*>(input);
    z = reinterpret_cast<T*>(output);
  }
  __syncthreads();

  // Calculate global thread index and step size
  LongType startIdx = threadIdx.x + blockIdx.x * blockDim.x;
  LongType step = blockDim.x * gridDim.x;

  // Coordinate arrays
  LongType inputCoords[SD_MAX_RANK];
  LongType outputCoords[SD_MAX_RANK];

  // Offset variables
  LongType xIndex;
  LongType zIndex;

  // Iterate over each class segment assigned to this block
  for (LongType segment = blockIdx.x; segment < numOfClasses; segment += gridDim.x) {
    // Convert segment index to coordinates for outputShape
    INDEX2COORDS(segment, outputRank, outputShapePtr, outputCoords);
    // Convert coordinates back to linear index for outputShape
    COORDS2INDEX(outputRank, outputStridePtr, outputCoords, zIndex);

    // Skip processing if zIndex is out of bounds
    if (zIndex >= zLen)
      continue;

    // Retrieve start and finish indices for the current segment
    auto start = starts[segment];
    auto finish = start + lengths[segment];

    // Skip processing if the length for the segment is zero
    if (lengths[segment] == 0) {
      continue;
    }

    // Iterate over elements within the segment, distributing work among threads
    for (LongType e = startIdx; e < finish; e += step) {
      // Convert linear index to coordinates for inputShape
      INDEX2COORDS(e, inputRank, inputShapePtr, inputCoords);
      // Convert coordinates back to linear index for inputShape
      COORDS2INDEX(inputRank, inputStridePtr, inputCoords, xIndex);

      // Skip processing if xIndex is out of bounds
      if (xIndex >= xLen)
        continue;

      // Perform atomic multiplication on the output buffer
      math::atomics::sd_atomicMul(&z[zIndex], x[xIndex]);
    }
  }
}

// -------------------------------------------------------------------------------------------------------------- //
template <typename T, typename I>
static SD_KERNEL void unsortedSegmentProdLinearKernel(T* input, LongType const* inputShape, I* indices,
                                                      LongType const* indicesShape, LongType* starts, LongType* lengths,
                                                      LongType numOfClasses, T* output, LongType const* outputShape) {

  // Shared memory for caching shape, stride, and rank information
  __shared__ LongType inputRank;
  __shared__ const LongType* inputShapePtr;
  __shared__ const LongType* inputStridePtr;

  __shared__ LongType indicesRank;
  __shared__ const LongType* indicesShapePtr;
  __shared__ const LongType* indicesStridePtr;

  __shared__ LongType outputRank;
  __shared__ const LongType* outputShapePtr;
  __shared__ const LongType* outputStridePtr;

  // Shared memory for pointers and lengths initialized by thread 0
  __shared__ T* x;
  __shared__ I* y;
  __shared__ T* z;
  __shared__ LongType xLen;
  __shared__ LongType zLen;

  if (threadIdx.x == 0) {
    // Cache rank, shape, and stride for inputShape
    inputRank = shape::rank(inputShape);
    inputShapePtr = shape::shapeOf(inputShape);
    inputStridePtr = shape::stride(inputShape);

    // Cache rank, shape, and stride for indicesShape
    indicesRank = shape::rank(indicesShape);
    indicesShapePtr = shape::shapeOf(indicesShape);
    indicesStridePtr = shape::stride(indicesShape);

    // Cache rank, shape, and stride for outputShape
    outputRank = shape::rank(outputShape);
    outputShapePtr = shape::shapeOf(outputShape);
    outputStridePtr = shape::stride(outputShape);

    // Cache lengths
    xLen = shape::length(inputShape);
    zLen = shape::length(outputShape);

    // Initialize pointers
    x = input;
    y = indices;
    z = output;
  }
  __syncthreads();

  // Calculate global thread index and step size
  LongType startIdx = threadIdx.x + blockIdx.x * blockDim.x;
  LongType step = blockDim.x * gridDim.x;

  // Coordinate arrays
  LongType xCoords[SD_MAX_RANK];
  LongType yCoords[SD_MAX_RANK];
  LongType zCoords[SD_MAX_RANK];

  // Offset variables
  LongType xIndex;
  LongType yIndex;
  LongType zIndex;

  for (LongType idx = startIdx; idx < xLen; idx += step) {
    // Convert linear index to coordinates for inputShape
    INDEX2COORDS(idx, inputRank, inputShapePtr, xCoords);
    // Convert coordinates back to linear index for inputShape
    COORDS2INDEX(inputRank, inputStridePtr, xCoords, xIndex);

    // Convert linear index to coordinates for indicesShape
    INDEX2COORDS(idx, indicesRank, indicesShapePtr, yCoords);
    // Convert coordinates back to linear index for indicesShape
    COORDS2INDEX(indicesRank, indicesStridePtr, yCoords, yIndex);

    // Retrieve the segment index from indices
    auto segment = y[yIndex];

    // Convert segment index to coordinates for outputShape
    INDEX2COORDS(segment, outputRank, outputShapePtr, zCoords);
    // Convert coordinates back to linear index for outputShape
    COORDS2INDEX(outputRank, outputStridePtr, zCoords, zIndex);

    // Skip processing if the length for the segment is zero
    if (lengths[segment] == 0) {
      continue;
    }

    // Perform atomic multiplication on the output buffer
    math::atomics::sd_atomicMul(&z[zIndex], x[xIndex]);
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

  // Early exit if block index is out of range
  if (blockIdx.x >= indicesLen)
    return;

  // Shared memory for caching shape, stride, and rank information
  __shared__ LongType inputTadRank;
  __shared__ const LongType* inputTadShapePtr;
  __shared__ const LongType* inputTadStridePtr;

  __shared__ LongType outputTadRank;
  __shared__ const LongType* outputTadShapePtr;
  __shared__ const LongType* outputTadStridePtr;

  __shared__ LongType inputRank;
  __shared__ const LongType* inputShapePtr;
  __shared__ const LongType* inputStridePtr;

  __shared__ LongType outputRank;
  __shared__ const LongType* outputShapePtr;
  __shared__ const LongType* outputStridePtr;

  // Shared memory for pointers and lengths initialized by thread 0
  __shared__ T* x;
  __shared__ T* z;
  __shared__ LongType len;
  __shared__ LongType total;

  if (threadIdx.x == 0) {
    // Cache rank, shape, and stride for inputTads
    inputTadRank = shape::rank(inputTads);
    inputTadShapePtr = shape::shapeOf(inputTads);
    inputTadStridePtr = shape::stride(inputTads);

    // Cache rank, shape, and stride for outputTads
    outputTadRank = shape::rank(outputTads);
    outputTadShapePtr = shape::shapeOf(outputTads);
    outputTadStridePtr = shape::stride(outputTads);

    // Cache rank, shape, and stride for inputShape
    inputRank = shape::rank(inputShape);
    inputShapePtr = shape::shapeOf(inputShape);
    inputStridePtr = shape::stride(inputShape);

    // Cache rank, shape, and stride for outputShape
    outputRank = shape::rank(outputShape);
    outputShapePtr = shape::shapeOf(outputShape);
    outputStridePtr = shape::stride(outputShape);

    // Cache lengths and total size
    total = shape::sizeAt(inputShape, 0);
    len = shape::length(inputTads);

    // Initialize pointers
    x = reinterpret_cast<T*>(inputBuf);
    z = reinterpret_cast<T*>(outputBuf);
  }
  __syncthreads();

  // Calculate global thread index and step size
  LongType startIdx = blockIdx.x;
  LongType step = gridDim.x;

  // Coordinate arrays
  LongType inputCoords[SD_MAX_RANK];
  LongType outputCoords[SD_MAX_RANK];

  // Offset variables
  LongType xIndex;
  LongType zIndex;

  for (auto idx = startIdx; idx < total; idx += step) {
    // Retrieve the segment index from indices
    auto segment = indices[idx];

    // Pointers to the current input and output TADs
    T* current = x + inputTadOffsets[idx];
    T* currentOut = z + outputTadOffsets[segment];

    // Retrieve start and finish indices for the current segment
    LongType start = starts[segment];
    LongType finish = start + lengths[segment];

    // Skip processing if the length for the segment is zero
    if (lengths[segment] == 0) continue;

    // Iterate over elements within the TAD
    for (auto e = threadIdx.x; e < len; e += blockDim.x) {
      // Convert linear index to coordinates for inputTads
      INDEX2COORDS(e, inputTadRank, inputTadShapePtr, inputCoords);
      // Convert coordinates back to linear index for inputTads
      COORDS2INDEX(inputTadRank, inputTadStridePtr, inputCoords, xIndex);

      // Convert linear index to coordinates for outputTads
      INDEX2COORDS(e, outputTadRank, outputTadShapePtr, outputCoords);
      // Convert coordinates back to linear index for outputTads
      COORDS2INDEX(outputTadRank, outputTadStridePtr, outputCoords, zIndex);

      // Perform atomic multiplication on the output buffer
      math::atomics::sd_atomicMul(&currentOut[zIndex], current[xIndex]);
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
                                                LongType const* forwardShape, void* eps, LongType const* epsShape,
                                                void* indicesBuf, LongType const* indicesShape, void* outputBuf,
                                                LongType const* outputShape) {

  // Shared memory for caching shape, stride, and rank information
  __shared__ LongType inputRank;
  __shared__ const LongType* inputShapePtr;
  __shared__ const LongType* inputStridePtr;

  __shared__ LongType forwardRank;
  __shared__ const LongType* forwardShapePtr;
  __shared__ const LongType* forwardStridePtr;

  __shared__ LongType epsRank;
  __shared__ const LongType* epsShapePtr;
  __shared__ const LongType* epsStridePtr;

  __shared__ LongType indicesRank;
  __shared__ const LongType* indicesShapePtr;
  __shared__ const LongType* indicesStridePtr;

  __shared__ LongType outputRank;
  __shared__ const LongType* outputShapePtr;
  __shared__ const LongType* outputStridePtr;

  // Shared memory for pointers and lengths initialized by thread 0
  __shared__ T* x;
  __shared__ T* gradIn;
  __shared__ T* gradOut;
  __shared__ I* y;
  __shared__ T* z;
  __shared__ LongType xLen;
  __shared__ LongType gradLen;
  __shared__ LongType currentLen;

  if (threadIdx.x == 0) {
    // Cache rank, shape, and stride for inputShape
    inputRank = shape::rank(inputShape);
    inputShapePtr = shape::shapeOf(inputShape);
    inputStridePtr = shape::stride(inputShape);

    // Cache rank, shape, and stride for forwardShape
    forwardRank = shape::rank(forwardShape);
    forwardShapePtr = shape::shapeOf(forwardShape);
    forwardStridePtr = shape::stride(forwardShape);

    // Cache rank, shape, and stride for epsShape
    epsRank = shape::rank(epsShape);
    epsShapePtr = shape::shapeOf(epsShape);
    epsStridePtr = shape::stride(epsShape);

    // Cache rank, shape, and stride for indicesShape
    indicesRank = shape::rank(indicesShape);
    indicesShapePtr = shape::shapeOf(indicesShape);
    indicesStridePtr = shape::stride(indicesShape);

    // Cache rank, shape, and stride for outputShape
    outputRank = shape::rank(outputShape);
    outputShapePtr = shape::shapeOf(outputShape);
    outputStridePtr = shape::stride(outputShape);

    // Initialize pointers and lengths
    xLen = shape::length(inputShape);
    gradLen = shape::length(epsShape);
    currentLen = shape::length(outputShape); // Assuming 'currentLen' corresponds to outputShape

    x = reinterpret_cast<T*>(inputBuf);
    y = reinterpret_cast<I*>(indicesBuf);
    z = reinterpret_cast<T*>(outputBuf);
    gradIn = reinterpret_cast<T*>(forwardOutput);
    gradOut = reinterpret_cast<T*>(eps);
  }
  __syncthreads();

  // Calculate global thread index and step size
  LongType start = blockIdx.x * blockDim.x + threadIdx.x;
  LongType step = gridDim.x * blockDim.x;

  // Coordinate arrays
  LongType xCoords[SD_MAX_RANK];
  LongType yCoords[SD_MAX_RANK];
  LongType zCoords[SD_MAX_RANK];
  LongType gradICoords[SD_MAX_RANK];
  LongType gradOCoords[SD_MAX_RANK];

  // Offset variables
  LongType xOffset;
  LongType yOffset;
  LongType zOffset;
  LongType gradOffsetI;
  LongType gradOffsetO;

  for (LongType e = start; e < xLen; e += step) {
    // Convert linear index to coordinates for inputShape
    INDEX2COORDS(e, inputRank, inputShapePtr, xCoords);
    // Convert coordinates back to linear index for inputShape
    COORDS2INDEX(inputRank, inputStridePtr, xCoords, xOffset);

    // Convert linear index to coordinates for indicesShape
    INDEX2COORDS(e, indicesRank, indicesShapePtr, yCoords);
    // Convert coordinates back to linear index for indicesShape
    COORDS2INDEX(indicesRank, indicesStridePtr, yCoords, yOffset);

    // Retrieve the class index from indices
    auto classIndex = y[yOffset];

    // Convert class index to coordinates for forwardShape
    INDEX2COORDS(classIndex, forwardRank, forwardShapePtr, gradICoords);
    // Convert coordinates back to linear index for forwardShape
    COORDS2INDEX(forwardRank, forwardStridePtr, gradICoords, gradOffsetI);

    // Convert class index to coordinates for epsShape
    INDEX2COORDS(classIndex, epsRank, epsShapePtr, gradOCoords);
    // Convert coordinates back to linear index for epsShape
    COORDS2INDEX(epsRank, epsStridePtr, gradOCoords, gradOffsetO);

    // Convert linear index to coordinates for outputShape
    INDEX2COORDS(e, outputRank, outputShapePtr, zCoords);
    // Convert coordinates back to linear index for outputShape
    COORDS2INDEX(outputRank, outputStridePtr, zCoords, zOffset);

    // Perform the computation: z[zOffset] = gradOut[gradOffsetO] * gradIn[gradOffsetI] / x[xOffset];
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

  // Shared memory for caching shape, stride, and rank information
  __shared__ LongType inputRank;
  __shared__ const LongType* inputShapePtr;
  __shared__ const LongType* inputStridePtr;

  __shared__ LongType forwardRank;
  __shared__ const LongType* forwardShapePtr;
  __shared__ const LongType* forwardStridePtr;

  __shared__ LongType epsRank;
  __shared__ const LongType* epsShapePtr;
  __shared__ const LongType* epsStridePtr;

  __shared__ LongType indicesRank;
  __shared__ const LongType* indicesShapePtr;
  __shared__ const LongType* indicesStridePtr;

  __shared__ LongType outputRank;
  __shared__ const LongType* outputShapePtr;
  __shared__ const LongType* outputStridePtr;

  __shared__ LongType inputTadRank;
  __shared__ const LongType* inputTadShapePtr;
  __shared__ const LongType* inputTadStridePtr;

  __shared__ LongType gradInTadRank;
  __shared__ const LongType* gradInTadShapePtr;
  __shared__ const LongType* gradInTadStridePtr;

  __shared__ LongType gradOutTadRank;
  __shared__ const LongType* gradOutTadShapePtr;
  __shared__ const LongType* gradOutTadStridePtr;

  __shared__ LongType outTadRank;
  __shared__ const LongType* outTadShapePtr;
  __shared__ const LongType* outTadStridePtr;

  // Shared memory for pointers and lengths initialized by thread 0
  __shared__ T* x;
  __shared__ T* gradIn;
  __shared__ T* gradOut;
  __shared__ I* y;
  __shared__ T* z;
  __shared__ LongType xLen;
  __shared__ LongType yLen;
  __shared__ LongType gradLen;
  __shared__ LongType currentLen;

  if (threadIdx.x == 0) {
    // Cache rank, shape, and stride for inputShape
    inputRank = shape::rank(inputShape);
    inputShapePtr = shape::shapeOf(inputShape);
    inputStridePtr = shape::stride(inputShape);

    // Cache rank, shape, and stride for forwardShape
    forwardRank = shape::rank(forwardShape);
    forwardShapePtr = shape::shapeOf(forwardShape);
    forwardStridePtr = shape::stride(forwardShape);

    // Cache rank, shape, and stride for epsShape
    epsRank = shape::rank(epsShape);
    epsShapePtr = shape::shapeOf(epsShape);
    epsStridePtr = shape::stride(epsShape);

    // Cache rank, shape, and stride for indicesShape
    indicesRank = shape::rank(indicesShape);
    indicesShapePtr = shape::shapeOf(indicesShape);
    indicesStridePtr = shape::stride(indicesShape);

    // Cache rank, shape, and stride for outputShape
    outputRank = shape::rank(outputShape);
    outputShapePtr = shape::shapeOf(outputShape);
    outputStridePtr = shape::stride(outputShape);

    // Cache rank, shape, and stride for inputTad
    inputTadRank = shape::rank(inputTad);
    inputTadShapePtr = shape::shapeOf(inputTad);
    inputTadStridePtr = shape::stride(inputTad);

    // Cache rank, shape, and stride for gradInTad
    gradInTadRank = shape::rank(gradInTad);
    gradInTadShapePtr = shape::shapeOf(gradInTad);
    gradInTadStridePtr = shape::stride(gradInTad);

    // Cache rank, shape, and stride for gradOutTad
    gradOutTadRank = shape::rank(gradOutTad);
    gradOutTadShapePtr = shape::shapeOf(gradOutTad);
    gradOutTadStridePtr = shape::stride(gradOutTad);

    // Cache rank, shape, and stride for outTad
    outTadRank = shape::rank(outTad);
    outTadShapePtr = shape::shapeOf(outTad);
    outTadStridePtr = shape::stride(outTad);

    // Initialize pointers and lengths
    xLen = shape::length(inputShape);
    yLen = shape::length(indicesShape);
    gradLen = shape::length(epsShape);
    currentLen = shape::length(outTad);

    x = reinterpret_cast<T*>(inputBuf);
    y = reinterpret_cast<I*>(indicesBuf);
    z = reinterpret_cast<T*>(outputBuf);
    gradOut = reinterpret_cast<T*>(eps);
    gradIn = reinterpret_cast<T*>(forwardOutput);
  }
  __syncthreads();

  // Calculate global thread index and step size
  LongType startIdx = blockIdx.x;
  LongType step = gridDim.x;

  // Coordinate arrays
  LongType yCoords[SD_MAX_RANK];
  LongType yIndex;

  // Iterate over all relevant indices
  for (auto i = startIdx; i < yLen; i += step) {
    // Convert linear index to coordinates for indicesShape
    INDEX2COORDS(i, indicesRank, indicesShapePtr, yCoords);
    // Convert coordinates back to linear index for indicesShape
    COORDS2INDEX(indicesRank, indicesStridePtr, yCoords, yIndex);

    // Retrieve the segment index from indices
    auto segment = y[yIndex];

    // Pointers to the current input and output TADs
    T* current = x + inputOffsets[i];
    T* currentOut = z + outOffsets[i];

    // Pointers to the corresponding gradIn and gradOut TADs
    T* in = gradIn + gradInOffsets[segment];
    T* outGrad = gradOut + gradOutOffsets[segment];

    // Perform element-wise computation within the current TAD
    for (auto e = threadIdx.x; e < currentLen; e += blockDim.x) {
      // Compute output: currentOut[e] = outGrad[e] * in[e] / current[e];
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

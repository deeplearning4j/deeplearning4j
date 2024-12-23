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

  // Early exit if block index is out of range
  if (blockIdx.x >= numOfClasses)
    return;

  // Shared memory for caching shape, stride, and rank information
  __shared__ LongType inputShapeRank;
  __shared__ const LongType* inputShapePtr;
  __shared__ const LongType* inputStridePtr;

  __shared__ LongType outputShapeRank;
  __shared__ const LongType* outputShapePtr;
  __shared__ const LongType* outputStridePtr;

  // Shared memory for pointers and lengths initialized by thread 0
  __shared__ const T* x;
  __shared__ T* z;
  __shared__ LongType xLen;
  __shared__ LongType zLen;
  __shared__ LongType zIndex;
  __shared__ LongType startIdx;
  __shared__ LongType finishIdx;

  if (threadIdx.x == 0) {
    // Cache rank, shape, and stride for inputShape
    inputShapeRank = shape::rank(inputShape);
    inputShapePtr = shape::shapeOf(inputShape);
    inputStridePtr = shape::stride(inputShape);

    // Cache rank, shape, and stride for outputShape
    outputShapeRank = shape::rank(outputShape);
    outputShapePtr = shape::shapeOf(outputShape);
    outputStridePtr = shape::stride(outputShape);

    // Cache lengths
    xLen = shape::length(inputShape);
    zLen = shape::length(outputShape);

    // Initialize pointers
    x = reinterpret_cast<const T*>(input);
    z = reinterpret_cast<T*>(output);

    // Compute zIndex based on the current segment (blockIdx.x)
    LongType segment = blockIdx.x;

    LongType outputCoords[SD_MAX_RANK];
    INDEX2COORDS(segment, outputShapeRank, outputShapePtr, outputCoords);
    COORDS2INDEX(outputShapeRank, outputStridePtr, outputCoords, zIndex);

    if (zIndex < zLen) {
      LongType start = indices[segment];
      LongType finish = start + lengths[segment];

      if (lengths[segment] > 0) {
        LongType startCoords[SD_MAX_RANK];
        LongType startIndex;
        INDEX2COORDS(start, inputShapeRank, inputShapePtr, startCoords);
        COORDS2INDEX(inputShapeRank, inputStridePtr, startCoords, startIndex);
        z[zIndex] = x[startIndex] / static_cast<T>(lengths[segment]);
      } else {
        z[zIndex] = 0;
      }

      // Store start and finish indices for the segment
      startIdx = start;
      finishIdx = finish;
    } else {
      // If zIndex is out of bounds, set start and finish to invalid values
      startIdx = 0;
      finishIdx = 0;
    }
  }
  __syncthreads();

  // Skip processing if zIndex is out of bounds or segment length is zero
  if (zIndex >= zLen || lengths[blockIdx.x] == 0)
    return;

  // Calculate global thread index and step size
  LongType threadIdxGlobal = threadIdx.x;
  LongType step = blockDim.x;

  // Iterate over elements within the segment, distributing work among threads
  for (LongType e = startIdx + threadIdxGlobal + 1; e < finishIdx; e += step) {
    // Convert linear index to coordinates for inputShape
    LongType inputCoords[SD_MAX_RANK];
    INDEX2COORDS(e, inputShapeRank, inputShapePtr, inputCoords);
    // Convert coordinates back to linear index for inputShape
    LongType xOffset;
    COORDS2INDEX(inputShapeRank, inputStridePtr, inputCoords, xOffset);

    // Boundary check for input index
    if (xOffset >= xLen)
      continue;

    // Perform atomic addition on the output buffer with mean computation
    math::atomics::sd_atomicAdd(&z[zIndex], x[xOffset] / static_cast<T>(lengths[blockIdx.x]));
  }
}

// -------------------------------------------------------------------------------------------------------------- //
template <typename T, typename I>
static SD_KERNEL void unsortedSegmentMeanLinearKernel(void* input, LongType const* inputShape, void* indices,
                                                      LongType const* indicesShape, LongType* starts, LongType* lengths,
                                                      LongType numOfClasses, void* output,
                                                      LongType const* outputShape) {

  // Early exit if block index is out of range
  if (blockIdx.x >= numOfClasses)
    return;

  // Shared memory for caching shape, stride, and rank information
  __shared__ LongType inputShapeRank;
  __shared__ const LongType* inputShapePtr;
  __shared__ const LongType* inputStridePtr;

  __shared__ LongType indicesShapeRank;
  __shared__ const LongType* indicesShapePtr;
  __shared__ const LongType* indicesStridePtr;

  __shared__ LongType outputShapeRank;
  __shared__ const LongType* outputShapePtr;
  __shared__ const LongType* outputStridePtr;

  // Shared memory for pointers and lengths initialized by thread 0
  __shared__ const T* x;
  __shared__ I* y;
  __shared__ T* z;
  __shared__ LongType xLen;
  __shared__ LongType zLen;
  __shared__ LongType zIndex;
  __shared__ LongType startIdx;
  __shared__ LongType finishIdx;

  if (threadIdx.x == 0) {
    // Cache rank, shape, and stride for inputShape
    inputShapeRank = shape::rank(inputShape);
    inputShapePtr = shape::shapeOf(inputShape);
    inputStridePtr = shape::stride(inputShape);

    // Cache rank, shape, and stride for indicesShape
    indicesShapeRank = shape::rank(indicesShape);
    indicesShapePtr = shape::shapeOf(indicesShape);
    indicesStridePtr = shape::stride(indicesShape);

    // Cache rank, shape, and stride for outputShape
    outputShapeRank = shape::rank(outputShape);
    outputShapePtr = shape::shapeOf(outputShape);
    outputStridePtr = shape::stride(outputShape);

    // Cache lengths
    xLen = shape::length(inputShape);
    zLen = shape::length(outputShape);

    // Initialize pointers
    x = reinterpret_cast<const T*>(input);
    y = reinterpret_cast<I*>(indices);
    z = reinterpret_cast<T*>(output);

    // Compute zIndex based on the current segment (blockIdx.x)
    LongType segment = blockIdx.x;

    LongType zCoords[SD_MAX_RANK];
    INDEX2COORDS(segment, outputShapeRank, outputShapePtr, zCoords);
    COORDS2INDEX(outputShapeRank, outputStridePtr, zCoords, zIndex);

    if (zIndex < zLen) {
      LongType start = starts[segment];
      LongType finish = start + lengths[segment];

      if (lengths[segment] > 0) {
        LongType startCoords[SD_MAX_RANK];
        LongType startIndex;
        INDEX2COORDS(start, inputShapeRank, inputShapePtr, startCoords);
        COORDS2INDEX(inputShapeRank, inputStridePtr, startCoords, startIndex);
        z[zIndex] = x[startIndex] / static_cast<T>(lengths[segment]);
      } else {
        z[zIndex] = 0;
      }

      // Store start and finish indices for the segment
      startIdx = start;
      finishIdx = finish;
    } else {
      // If zIndex is out of bounds, set start and finish to invalid values
      startIdx = 0;
      finishIdx = 0;
    }
  }
  __syncthreads();

  // Skip processing if zIndex is out of bounds or segment length is zero
  if (zIndex >= zLen || lengths[blockIdx.x] == 0)
    return;

  // Calculate global thread index and step size
  LongType threadIdxGlobal = threadIdx.x;
  LongType step = blockDim.x;

  // Iterate over elements within the segment, distributing work among threads
  for (LongType e = startIdx + threadIdxGlobal + 1; e < finishIdx; e += step) {
    // Convert linear index to coordinates for inputShape
    LongType inputCoords[SD_MAX_RANK];
    INDEX2COORDS(e, inputShapeRank, inputShapePtr, inputCoords);
    // Convert coordinates back to linear index for inputShape
    LongType xOffset;
    COORDS2INDEX(inputShapeRank, inputStridePtr, inputCoords, xOffset);

    // Convert linear index to coordinates for indicesShape
    LongType indicesCoords[SD_MAX_RANK];
    INDEX2COORDS(e, indicesShapeRank, indicesShapePtr, indicesCoords);
    // Convert coordinates back to linear index for indicesShape
    LongType yIndex;
    COORDS2INDEX(indicesShapeRank, indicesStridePtr, indicesCoords, yIndex);

    // Check if the current element belongs to the current segment
    if (y[yIndex] == blockIdx.x && e != startIdx) {
      // Perform atomic addition on the output buffer with mean computation
      math::atomics::sd_atomicAdd(&z[zIndex], x[xOffset] / static_cast<T>(lengths[blockIdx.x]));
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

  // Early exit if block index is out of range
  if (blockIdx.x >= indicesLen)
    return;

  // Shared memory for caching shape, stride, and rank information
  __shared__ LongType inputShapeRank;
  __shared__ const LongType* inputShapePtr;
  __shared__ const LongType* inputStridePtr;

  __shared__ LongType outputShapeRank;
  __shared__ const LongType* outputShapePtr;
  __shared__ const LongType* outputStridePtr;

  __shared__ LongType inputTadRank;
  __shared__ const LongType* inputTadShapePtr;
  __shared__ const LongType* inputTadStridePtr;

  __shared__ LongType outputTadRank;
  __shared__ const LongType* outputTadShapePtr;
  __shared__ const LongType* outputTadStridePtr;

  // Shared memory for pointers and lengths initialized by thread 0
  __shared__ const T* x;
  __shared__ T* z;
  __shared__ LongType len;
  __shared__ LongType total;
  __shared__ LongType segment;
  __shared__ LongType startIdx;
  __shared__ LongType finishIdx;

  if (threadIdx.x == 0) {
    // Cache rank, shape, and stride for inputShape
    inputShapeRank = shape::rank(inputShape);
    inputShapePtr = shape::shapeOf(inputShape);
    inputStridePtr = shape::stride(inputShape);

    // Cache rank, shape, and stride for outputShape
    outputShapeRank = shape::rank(outputShape);
    outputShapePtr = shape::shapeOf(outputShape);
    outputStridePtr = shape::stride(outputShape);

    // Cache rank, shape, and stride for inputTads
    inputTadRank = shape::rank(inputTads);
    inputTadShapePtr = shape::shapeOf(inputTads);
    inputTadStridePtr = shape::stride(inputTads);

    // Cache rank, shape, and stride for outputTads
    outputTadRank = shape::rank(outputTads);
    outputTadShapePtr = shape::shapeOf(outputTads);
    outputTadStridePtr = shape::stride(outputTads);

    // Cache lengths and total size
    len = shape::length(inputTads);
    total = shape::sizeAt(inputShape, 0);

    // Initialize pointers
    x = reinterpret_cast<const T*>(inputBuf);
    z = reinterpret_cast<T*>(outputBuf);

    // Retrieve the current segment index from indices
    segment = indices[blockIdx.x];
  }
  __syncthreads();

  // Pointers to the current output TAD
  T* currentOut = z + outputTadOffsets[segment];

  // Retrieve start and finish indices for the current segment
  if (threadIdx.x == 0) {
    startIdx = starts[segment];
    finishIdx = startIdx + lengths[segment];
  }
  __syncthreads();

  // Skip processing if the length for the segment is zero
  if (lengths[segment] == 0)
    return;

  // Iterate over input TADs assigned to this block
  for (LongType idx = blockIdx.x; idx < total; idx += gridDim.x) {
    const T* currentInput = x + inputTadOffsets[idx];

    // Iterate over elements within the input TAD, distributing work among threads
    for (LongType e = threadIdx.x; e < len; e += blockDim.x) {
      // Convert linear index to coordinates for inputTads
      LongType inputCoords[SD_MAX_RANK];
      INDEX2COORDS(e, inputTadRank, inputTadShapePtr, inputCoords);

      // Convert coordinates back to linear index for inputTads
      LongType xOffset;
      COORDS2INDEX(inputTadRank, inputTadStridePtr, inputCoords, xOffset);

      // Convert linear index to coordinates for outputTads
      LongType outputCoords[SD_MAX_RANK];
      INDEX2COORDS(e, outputTadRank, outputTadShapePtr, outputCoords);

      // Convert coordinates back to linear index for outputTads
      LongType zOffset;
      COORDS2INDEX(outputTadRank, outputTadStridePtr, outputCoords, zOffset);

      // Perform atomic addition on the output buffer with mean computation
      math::atomics::sd_atomicAdd(&currentOut[zOffset], T(currentInput[xOffset] / static_cast<T>(lengths[segment])));
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

  // Early exit if block index is out of range
  if (blockIdx.x >= shape::length(indicesShape))
    return;

  // Shared memory for caching shape, stride, and rank information
  __shared__ LongType inputShapeRank;
  __shared__ const LongType* inputShapePtr;
  __shared__ const LongType* inputStridePtr;

  __shared__ LongType indicesShapeRank;
  __shared__ const LongType* indicesShapePtr;
  __shared__ const LongType* indicesStridePtr;

  __shared__ LongType outputShapeRank;
  __shared__ const LongType* outputShapePtr;
  __shared__ const LongType* outputStridePtr;

  __shared__ LongType epsShapeRank;
  __shared__ const LongType* epsShapePtr;
  __shared__ const LongType* epsStridePtr;

  // Shared memory for pointers and lengths initialized by thread 0
  __shared__ const T* x;
  __shared__ I* y;
  __shared__ T* z;
  __shared__ T* gradOut;
  __shared__ LongType xLen;
  __shared__ LongType yLen;
  __shared__ LongType gradLen;
  __shared__ LongType currentLen;

  if (threadIdx.x == 0) {
    // Cache rank, shape, and stride for inputShape
    inputShapeRank = shape::rank(inputShape);
    inputShapePtr = shape::shapeOf(inputShape);
    inputStridePtr = shape::stride(inputShape);

    // Cache rank, shape, and stride for indicesShape
    indicesShapeRank = shape::rank(indicesShape);
    indicesShapePtr = shape::shapeOf(indicesShape);
    indicesStridePtr = shape::stride(indicesShape);

    // Cache rank, shape, and stride for outputShape
    outputShapeRank = shape::rank(outputShape);
    outputShapePtr = shape::shapeOf(outputShape);
    outputStridePtr = shape::stride(outputShape);

    // Cache rank, shape, and stride for epsShape
    epsShapeRank = shape::rank(epsShape);
    epsShapePtr = shape::shapeOf(epsShape);
    epsStridePtr = shape::stride(epsShape);

    // Cache lengths
    xLen = shape::length(inputShape);
    yLen = shape::length(indicesShape);
    gradLen = shape::length(epsShape);
    currentLen = shape::length(outTad);

    // Initialize pointers
    x = reinterpret_cast<const T*>(inputBuf);
    y = reinterpret_cast<I*>(indicesBuf);
    z = reinterpret_cast<T*>(outputBuf);
    gradOut = reinterpret_cast<T*>(eps);
  }
  __syncthreads();

  // Retrieve the current segment index from indices
  LongType i = blockIdx.x;
  if (i >= yLen)
    return;

  LongType segment = y[i];

  // Calculate pointers to the current output and gradOut TADs
  T* currentOut = z + outOffsets[i];
  T* outGrad = gradOut + gradOutOffsets[segment];

  // Retrieve the length for the current segment
  LongType segmentLength = lengths[segment];
  if (segmentLength == 0)
    return;

  // Calculate the number of elements to process
  // Assuming 'currentLen' corresponds to the number of elements per TAD
  for (auto e = threadIdx.x; e < currentLen; e += blockDim.x) {
    // Convert linear index to coordinates for outTad
    LongType zCoords[SD_MAX_RANK];
    INDEX2COORDS(e, shape::rank(outTad), shape::shapeOf(outTad), zCoords);
    // Convert coordinates back to linear index for outTad
    LongType zIndex;
    COORDS2INDEX(shape::rank(outTad), shape::stride(outTad), zCoords, zIndex);

    // Convert linear index to coordinates for gradOutTad
    LongType gradCoords[SD_MAX_RANK];
    INDEX2COORDS(e, shape::rank(gradOutTad), shape::shapeOf(gradOutTad), gradCoords);
    // Convert coordinates back to linear index for gradOutTad
    LongType gradIndex;
    COORDS2INDEX(shape::rank(gradOutTad), shape::stride(gradOutTad), gradCoords, gradIndex);

    // Compute the mean gradient
    T meanGrad = outGrad[gradIndex] / static_cast<T>(segmentLength);

    // Perform the assignment
    currentOut[zIndex] = meanGrad;
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

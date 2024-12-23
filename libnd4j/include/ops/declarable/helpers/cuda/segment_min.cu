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
      LongType zCoords[SD_MAX_RANK];
      INDEX2COORDS(segment, shape::rank(outputShape), shape::shapeOf(outputShape), zCoords);
      COORDS2INDEX(shape::rank(outputShape), shape::stride(outputShape), zCoords, zIndex);
      if(zIndex >= zLen)
        return;
      start = starts[segment];
      finish = start + lengths[segment];
      LongType startCoords[SD_MAX_RANK];
      LongType startIndex;
      INDEX2COORDS(start, shape::rank(inputShape), shape::shapeOf(inputShape), startCoords);
      COORDS2INDEX(shape::rank(inputShape), shape::stride(inputShape), startCoords, startIndex);
      z[zIndex] = x[startIndex];
      val[segment] = z[zIndex];
    }
  }
  __syncthreads();

  for (auto e = start + threadIdx.x + 1; e < finish; e += blockDim.x) {
    LongType eCoords[SD_MAX_RANK];
    LongType eIndex;
    INDEX2COORDS(e, shape::rank(inputShape), shape::shapeOf(inputShape), eCoords);
    COORDS2INDEX(shape::rank(inputShape), shape::stride(inputShape), eCoords, eIndex);
    if (eIndex >= xLen) return;
    math::atomics::sd_atomicMin(&z[zIndex], x[eIndex]);
  }
}
// -------------------------------------------------------------------------------------------------------------- //

template <typename T, typename I>
static SD_KERNEL void unsortedSegmentMinLinearKernel(const void* input, const LongType* inputShape,
                                                     const void* indices, const LongType* indicesShape,
                                                     LongType* starts, LongType* lengths,
                                                     LongType numOfClasses, void* output,
                                                     const LongType* outputShape) {

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
  __shared__ const I* y;
  __shared__ T* z;
  __shared__ LongType xLen;
  __shared__ LongType zLen;

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
    y = reinterpret_cast<const I*>(indices);
    z = reinterpret_cast<T*>(output);
  }
  __syncthreads();

  // Coordinate arrays
  LongType zCoords[SD_MAX_RANK];
  LongType inputCoords[SD_MAX_RANK];
  LongType yCoords[SD_MAX_RANK];

  // Offset variables
  LongType zIndex;
  LongType xIndex;
  LongType yIndex;

  // Calculate global thread index and step size
  LongType startIdx = threadIdx.x + blockIdx.x * blockDim.x;
  LongType step = blockDim.x * gridDim.x;

  // Iterate over each element assigned to this thread
  for (LongType e = startIdx; e < xLen; e += step) {
    // Convert linear index to coordinates for inputShape
    INDEX2COORDS(e, inputShapeRank, inputShapePtr, inputCoords);
    // Convert coordinates back to linear index for inputShape
    COORDS2INDEX(inputShapeRank, inputStridePtr, inputCoords, xIndex);

    // Convert linear index to coordinates for indicesShape
    INDEX2COORDS(e, indicesShapeRank, indicesShapePtr, yCoords);
    // Convert coordinates back to linear index for indicesShape
    COORDS2INDEX(indicesShapeRank, indicesStridePtr, yCoords, yIndex);

    // Retrieve the segment index from indices
    auto segment = y[yIndex];

    // Convert segment index to coordinates for outputShape
    INDEX2COORDS(segment, outputShapeRank, outputShapePtr, zCoords);
    // Convert coordinates back to linear index for outputShape
    COORDS2INDEX(outputShapeRank, outputStridePtr, zCoords, zIndex);

    // Boundary check for output index
    if (zIndex >= zLen)
      continue;

    // Check if the length for the segment is zero
    if (lengths[segment] == 0) {
      continue;
    }

    // Perform atomic multiplication on the output buffer
    math::atomics::sd_atomicMul(&z[zIndex], x[xIndex]);
  }
}

// -------------------------------------------------------------------------------------------------------------- //
// SegmentMin kernel
template <typename T, typename I>
static SD_KERNEL void segmentMinTadKernel(const void* inputBuf, const LongType* inputShape,
                                          const LongType* inputTads, const LongType* inputTadOffsets,
                                          I* indices, LongType* starts,
                                          LongType* lengths, LongType numOfClasses, void* outputBuf,
                                          const LongType* outputShape,
                                          const LongType* outputTads, const LongType* outputTadOffsets,
                                          LongType indicesLen) {

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

  __shared__ LongType inputShapeRank;
  __shared__ const LongType* inputShapePtr;
  __shared__ const LongType* inputStridePtr;

  __shared__ LongType outputShapeRank;
  __shared__ const LongType* outputShapePtr;
  __shared__ const LongType* outputStridePtr;

  // Shared memory for pointers and lengths initialized by thread 0
  __shared__ const T* x;
  __shared__ T* z;
  __shared__ LongType len;
  __shared__ LongType total;
  __shared__ LongType segment;
  __shared__ LongType startIdx;
  __shared__ LongType finishIdx;

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
    inputShapeRank = shape::rank(inputShape);
    inputShapePtr = shape::shapeOf(inputShape);
    inputStridePtr = shape::stride(inputShape);

    // Cache rank, shape, and stride for outputShape
    outputShapeRank = shape::rank(outputShape);
    outputShapePtr = shape::shapeOf(outputShape);
    outputStridePtr = shape::stride(outputShape);

    // Cache lengths and total size
    len = shape::length(inputTads);
    total = shape::sizeAt(inputShape, 0);

    // Initialize pointers
    x = reinterpret_cast<const T*>(inputBuf);
    z = reinterpret_cast<T*>(outputBuf);

    // Retrieve the current segment index from indices
    segment = y[blockIdx.x]; // Assuming y is properly initialized
  }
  __syncthreads();

  // After synchronization, all threads can access cached values

  // Pointers to the current input and output TADs
  const T* currentInput = x + inputTadOffsets[blockIdx.x];
  T* currentOutput = z + outputTadOffsets[segment];

  // Retrieve start and finish indices for the current segment
  if (threadIdx.x == 0) {
    startIdx = starts[segment];
    finishIdx = startIdx + lengths[segment];
  }
  __syncthreads();

  // Skip processing if the length for the segment is zero
  if (lengths[segment] == 0)
    return;

  // Iterate over elements within the current TAD, distributing work among threads
  for (auto e = threadIdx.x; e < len; e += blockDim.x) {
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

    // Perform atomic minimum on the output buffer
    math::atomics::sd_atomicMin(&currentOutput[zOffset], currentInput[xOffset]);
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
  T val = DataTypeUtils::infOrMax<T>();
  output->assign(val);
  sd::LongType zero2 = 0;
  sd::LongType len = indices->lengthOf();
  classesRangesBegs.assign(zero2);
  classesRangesLens.assign(len);
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
  T val = DataTypeUtils::infOrMax<T>();
  sd::LongType  len = indices->lengthOf();
  output->assign(val);
  sd::LongType  zero = 0;
  classesRangesBegs.assign(len);
  classesRangesLens.assign(zero);
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
    T val = DataTypeUtils::max<T>();
    output->assign(val);
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

  // Shared memory for caching shape, stride, and rank information
  __shared__ LongType inputShapeRank;
  __shared__ const LongType* inputShapePtr;
  __shared__ const LongType* inputStridePtr;

  __shared__ LongType forwardShapeRank;
  __shared__ const LongType* forwardShapePtr;
  __shared__ const LongType* forwardStridePtr;

  __shared__ LongType epsShapeRank;
  __shared__ const LongType* epsShapePtr;
  __shared__ const LongType* epsStridePtr;

  __shared__ LongType indicesShapeRank;
  __shared__ const LongType* indicesShapePtr;
  __shared__ const LongType* indicesStridePtr;

  __shared__ LongType outputShapeRank;
  __shared__ const LongType* outputShapePtr;
  __shared__ const LongType* outputStridePtr;

  // Shared memory for pointers and lengths initialized by thread 0
  __shared__ const T* x;
  __shared__ T* gradIn;
  __shared__ T* gradOut;
  __shared__ const I* y;
  __shared__ T* z;
  __shared__ LongType xLen;
  __shared__ LongType yLen;
  __shared__ LongType gradLen;
  __shared__ LongType currentLen;

  if (threadIdx.x == 0) {
    // Cache rank, shape, and stride for inputShape
    inputShapeRank = shape::rank(inputShape);
    inputShapePtr = shape::shapeOf(inputShape);
    inputStridePtr = shape::stride(inputShape);

    // Cache rank, shape, and stride for forwardShape
    forwardShapeRank = shape::rank(forwardShape);
    forwardShapePtr = shape::shapeOf(forwardShape);
    forwardStridePtr = shape::stride(forwardShape);

    // Cache rank, shape, and stride for epsShape
    epsShapeRank = shape::rank(epsShape);
    epsShapePtr = shape::shapeOf(epsShape);
    epsStridePtr = shape::stride(epsShape);

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
    yLen = shape::length(indicesShape);
    gradLen = shape::length(epsShape);
    currentLen = shape::length(outputShape); // Assuming 'currentLen' corresponds to the length of outputTad

    // Initialize pointers
    x = reinterpret_cast<const T*>(inputBuf);
    y = reinterpret_cast<const I*>(indicesBuf);
    z = reinterpret_cast<T*>(outputBuf);
    gradIn = reinterpret_cast<T*>(forwardOutput);
    gradOut = reinterpret_cast<T*>(eps);
  }
  __syncthreads();

  // Calculate global thread index and step size
  LongType startIdx = blockIdx.x * blockDim.x + threadIdx.x;
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

  for (LongType e = startIdx; e < xLen; e += step) {
    // Convert linear index to coordinates for outputShape
    INDEX2COORDS(e, outputShapeRank, outputShapePtr, zCoords);
    // Convert coordinates back to linear index for outputShape
    COORDS2INDEX(outputShapeRank, outputStridePtr, zCoords, zOffset);

    // Convert linear index to coordinates for inputShape
    INDEX2COORDS(e, inputShapeRank, inputShapePtr, xCoords);
    // Convert coordinates back to linear index for inputShape
    COORDS2INDEX(inputShapeRank, inputStridePtr, xCoords, xOffset);

    // Convert linear index to coordinates for indicesShape
    INDEX2COORDS(e, indicesShapeRank, indicesShapePtr, yCoords);
    // Convert coordinates back to linear index for indicesShape
    COORDS2INDEX(indicesShapeRank, indicesStridePtr, yCoords, yOffset);

    // Retrieve the segment index from indices
    auto segment = y[yOffset];

    // Convert segment index to coordinates for forwardShape
    INDEX2COORDS(segment, forwardShapeRank, forwardShapePtr, gradICoords);
    // Convert coordinates back to linear index for forwardShape
    COORDS2INDEX(forwardShapeRank, forwardStridePtr, gradICoords, gradOffsetI);

    // Convert segment index to coordinates for epsShape
    INDEX2COORDS(segment, epsShapeRank, epsShapePtr, gradOCoords);
    // Convert coordinates back to linear index for epsShape
    COORDS2INDEX(epsShapeRank, epsStridePtr, gradOCoords, gradOffsetO);

    // Compute the absolute difference
    T diff = math::sd_abs<T, T>(gradIn[gradOffsetI] - x[xOffset]);

    // Check if the difference is within the tolerance
    if (diff <= static_cast<T>(1.e-6)) {
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
  __shared__ T* gradIn;
  __shared__ T* gradOut;
  __shared__ const I* y;
  __shared__ T* z;
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

    // Cache lengths
    xLen = shape::length(inputShape);
    yLen = shape::length(indicesShape);
    gradLen = shape::length(epsShape);
    currentLen = shape::length(outTad);

    // Initialize pointers
    x = reinterpret_cast<const T*>(inputBuf);
    y = reinterpret_cast<const I*>(indicesBuf);
    z = reinterpret_cast<T*>(outputBuf);
    gradIn = reinterpret_cast<T*>(forwardOutput);
    gradOut = reinterpret_cast<T*>(eps);
  }
  __syncthreads();

  // Iterate over each index assigned to this block
  for (auto i = blockIdx.x; i < yLen; i += gridDim.x) {
    // Convert linear index to coordinates for indicesShape
    LongType yCoords[SD_MAX_RANK];
    INDEX2COORDS(i, indicesShapeRank, indicesShapePtr, yCoords);

    // Convert coordinates back to linear index for indicesShape
    LongType yIndex;
    COORDS2INDEX(indicesShapeRank, indicesStridePtr, yCoords, yIndex);

    // Retrieve the segment index from indices
    auto segment = y[yIndex];

    // Pointers to the current input and output TADs
    const T* current = x + inputOffsets[i];
    T* currentOut = z + outOffsets[i];

    // Pointers to the corresponding gradIn and gradOut TADs
    T* in = gradIn + gradInOffsets[segment];
    T* outGrad = gradOut + gradOutOffsets[segment];

    // Iterate over elements within the current TAD, distributing work among threads
    for (auto e = threadIdx.x; e < currentLen; e += blockDim.x) {
      // Compute the absolute difference
      T diff = math::sd_abs<T, T>(in[e] - current[e]);

      // Check if the difference is within the tolerance
      if (diff <= static_cast<T>(1.e-6)) {
        currentOut[e] = outGrad[e];
      }
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

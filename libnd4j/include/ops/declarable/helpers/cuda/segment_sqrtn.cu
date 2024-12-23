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
template <typename T, typename I>
static SD_KERNEL void unsortedSegmentSqrtNLinearKernel(T* input, LongType const* inputShape, I* indices,
                                                       LongType const* indicesShape, LongType* starts,
                                                       LongType* lengths, LongType numOfClasses, T* output,
                                                       LongType const* outputShape) {

  // Shared memory for caching shape, stride, and rank
  __shared__ LongType inputRank;
  __shared__ const LongType* inputShapePtr;
  __shared__ const LongType* inputStridePtr;

  __shared__ LongType indicesRank;
  __shared__ const LongType* indicesShapePtr;
  __shared__ const LongType* indicesStridePtr;

  __shared__ LongType outputRank;
  __shared__ const LongType* outputShapePtr;
  __shared__ const LongType* outputStridePtr;

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

    // Cache lengths for input and output
    xLen = shape::length(inputShape);
    zLen = shape::length(outputShape);
  }
  __syncthreads();

  // Calculate global thread index and step size
  auto startIdx = threadIdx.x + blockIdx.x * blockDim.x;
  auto step = blockDim.x * gridDim.x;

  // Coordinate arrays
  LongType yCoords[SD_MAX_RANK];
  LongType zCoords[SD_MAX_RANK];
  LongType xCoords[SD_MAX_RANK];

  // Offset variables
  LongType yIndex;
  LongType zIndex;
  LongType xIndex;

  for (auto idx = startIdx; idx < xLen; idx += step) {
    // Convert linear index to coordinates for indicesShape
    INDEX2COORDS(idx, indicesRank, indicesShapePtr, yCoords);
    // Convert coordinates back to linear index for indicesShape
    COORDS2INDEX(indicesRank, indicesStridePtr, yCoords, yIndex);

    // Retrieve the segment index from indices
    auto segment = indices[yIndex];

    // Convert segment index to coordinates for outputShape
    INDEX2COORDS(segment, outputRank, outputShapePtr, zCoords);
    // Convert coordinates back to linear index for outputShape
    COORDS2INDEX(outputRank, outputStridePtr, zCoords, zIndex);

    // Skip if the length for the segment is zero
    if (lengths[segment] == 0) continue;

    // Convert linear index to coordinates for inputShape
    INDEX2COORDS(idx, inputRank, inputShapePtr, xCoords);
    // Convert coordinates back to linear index for inputShape
    COORDS2INDEX(inputRank, inputStridePtr, xCoords, xIndex);

    // Skip if xIndex is out of bounds
    if (xIndex >= xLen) continue;

    // Perform atomic addition to the output buffer
    math::atomics::sd_atomicAdd(&output[zIndex], input[xIndex] /
                                                     math::sd_sqrt<LongType, T>(lengths[segment]));
  }
}

// -------------------------------------------------------------------------------------------------------------- //
// SegmentSqrtN kernel
template <typename T, typename I>
static SD_KERNEL void segmentSqrtNTadKernel(T* inputBuf, LongType const* inputShape, LongType const* inputTads,
                                            LongType const* inputTadOffsets, I* indices, LongType* starts,
                                            LongType* lengths, LongType numOfClasses, void* outputBuf,
                                            LongType const* outputShape, LongType const* outputTads,
                                            LongType const* outputTadOffsets, LongType numIndices) {

  if(blockIdx.x >= numIndices)
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
    auto segment = indices[idx];
    auto x = inputBuf + inputTadOffsets[idx];
    auto z = reinterpret_cast<T*>(outputBuf) + outputTadOffsets[segment];
    auto start = starts[segment];
    auto finish = start + lengths[segment];

    for (auto e = threadIdx.x; e < len; e += blockDim.x) {
      INDEX2COORDS(e, shape::rank(inputTads), shape::shapeOf(inputTads), inputCoords);
      COORDS2INDEX(shape::rank(inputTads), shape::stride(inputTads), inputCoords, xIndex);
      INDEX2COORDS(e, shape::rank(outputTads), shape::shapeOf(outputTads), outputCoords);
      COORDS2INDEX(shape::rank(outputTads), shape::stride(outputTads), outputCoords, zIndex);
      math::atomics::sd_atomicAdd(&z[zIndex], x[xIndex] / math::sd_sqrt<LongType, T>(lengths[segment]));
    }
  }
}
// -------------------------------------------------------------------------------------------------------------- //
template <typename T, typename I>
static void unsortedSegmentSqrtNFunctor_(LaunchContext* context, NDArray* input, NDArray* indices,
                                         LongType numOfClasses, NDArray* output) {
  auto stream = context->getCudaStream();
  NDArray classesRangesBegs = NDArrayFactory::create<LongType>('c', {numOfClasses}, context);
  NDArray classesRangesLens = NDArrayFactory::create<LongType>('c', {numOfClasses}, context);
  sd::LongType zero = 0;
  sd::LongType  one = 1;
  sd::LongType  len = indices->lengthOf();
  classesRangesBegs.assign(len);
  classesRangesLens.assign(zero);
  dim3 dims= getLaunchDims("segmentSqrtN");
  fillUpSegments(indices, numOfClasses, classesRangesBegs, classesRangesLens);
  LongType* begins = reinterpret_cast<LongType*>(classesRangesBegs.specialBuffer());
  LongType* lengths = reinterpret_cast<LongType*>(classesRangesLens.specialBuffer());
  output->nullify();
  if (input->isVector()  || input->isScalar()) {
    unsortedSegmentSqrtNLinearKernel<T, I><<<dims.x, dims.y, dims.z, *stream>>>(
        input->dataBuffer()->specialAsT<T>(), input->specialShapeInfo(), indices->dataBuffer()->specialAsT<I>(),
        indices->specialShapeInfo(), begins, lengths, numOfClasses, output->dataBuffer()->specialAsT<T>(),
        output->specialShapeInfo());
    sd::DebugHelper::checkErrorCode(stream, "unsortedSegmentSqrtNLinearKernel failed");

  } else {
    output->nullify();
    LongType zero = 0;
    std::vector<LongType> *dimensions = ShapeUtils::evalDimsToExclude(input->rankOf(), 1,&zero);
    auto packX = ConstantTadHelper::getInstance().tadForDimensions(input->shapeInfo(), dimensions);
    auto packZ = ConstantTadHelper::getInstance().tadForDimensions(output->shapeInfo(), dimensions);
    auto inputTads = packX->specialShapeInfo();
    auto inputTadOffsets = packX->specialOffsets();
    auto outputTads = packZ->specialShapeInfo();
    auto outputTadOffsets = packZ->specialOffsets();
    dims.x = input->sizeAt(0);
    segmentSqrtNTadKernel<T, I><<<dims.x, dims.y, dims.z, *stream>>>(
        input->dataBuffer()->specialAsT<T>(), input->specialShapeInfo(), inputTads, inputTadOffsets,
        indices->dataBuffer()->specialAsT<I>(), begins, lengths, numOfClasses, output->specialBuffer(),
        output->specialShapeInfo(), outputTads, outputTadOffsets, indices->lengthOf());
    sd::DebugHelper::checkErrorCode(stream, "segmentSqrtNTadKernel failed");

    delete dimensions;
  }
}
// -------------------------------------------------------------------------------------------------------------- //
void unsortedSegmentSqrtNFunctor(LaunchContext* context, NDArray* input, NDArray* indices, LongType numOfClasses, NDArray* output) {
  NDArray::prepareSpecialUse({output}, {input, indices});
  BUILD_DOUBLE_SELECTOR(input->dataType(), indices->dataType(), unsortedSegmentSqrtNFunctor_,
                        (context, input, indices, numOfClasses, output), SD_FLOAT_TYPES, SD_INDEXING_TYPES);
  NDArray::registerSpecialUse({output}, {input, indices});
}
// -------------------------------------------------------------------------------------------------------------- //
template <typename T, typename I>
static SD_KERNEL void segmentSqrtNBPLinearKernel(void* inputBuf, LongType const* inputShape, void* eps,
                                                 LongType const* epsShape, void* indicesBuf,
                                                 LongType const* indicesShape, LongType* lengths, void* outputBuf,
                                                 LongType const* outputShape) {
  // Shared memory for caching shape, stride, and rank
  __shared__ LongType inputRank;
  __shared__ const LongType* inputShapePtr;
  __shared__ const LongType* inputStridePtr;

  __shared__ LongType epsRank;
  __shared__ const LongType* epsShapePtr;
  __shared__ const LongType* epsStridePtr;

  __shared__ LongType indicesRank;
  __shared__ const LongType* indicesShapePtr;
  __shared__ const LongType* indicesStridePtr;

  __shared__ LongType outputRank;
  __shared__ const LongType* outputShapePtr;
  __shared__ const LongType* outputStridePtr;

  __shared__ LongType classRank;
  __shared__ const LongType* classShapePtr;
  __shared__ const LongType* classStridePtr;

  // Shared memory for variables initialized by thread 0
  __shared__ T* x;
  __shared__ T* gradIn;
  __shared__ T* gradOut;
  __shared__ I* y;
  __shared__ T* z;
  __shared__ LongType xLen;
  __shared__ LongType gradLen;

  if (threadIdx.x == 0) {
    // Initialize cached shape, stride, and rank for input
    inputRank = shape::rank(inputShape);
    inputShapePtr = shape::shapeOf(inputShape);
    inputStridePtr = shape::stride(inputShape);

    // Initialize cached shape, stride, and rank for eps
    epsRank = shape::rank(epsShape);
    epsShapePtr = shape::shapeOf(epsShape);
    epsStridePtr = shape::stride(epsShape);

    // Initialize cached shape, stride, and rank for indices
    indicesRank = shape::rank(indicesShape);
    indicesShapePtr = shape::shapeOf(indicesShape);
    indicesStridePtr = shape::stride(indicesShape);

    // Initialize cached shape, stride, and rank for output
    outputRank = shape::rank(outputShape);
    outputShapePtr = shape::shapeOf(outputShape);
    outputStridePtr = shape::stride(outputShape);

    // Initialize cached shape, stride, and rank for lengths (epsShape)
    classRank = epsRank;
    classShapePtr = epsShapePtr;
    classStridePtr = epsStridePtr;

    // Initialize pointers and lengths
    xLen = shape::length(inputShape);
    gradLen = shape::length(epsShape);
    x = reinterpret_cast<T*>(inputBuf);
    y = reinterpret_cast<I*>(indicesBuf);
    z = reinterpret_cast<T*>(outputBuf);
    gradOut = reinterpret_cast<T*>(eps);
  }
  __syncthreads();

  // Calculate global thread index and step size
  auto start = blockIdx.x * blockDim.x + threadIdx.x;
  auto step = gridDim.x * blockDim.x;

  // Coordinate arrays
  LongType zCoords[SD_MAX_RANK];
  LongType xCoords[SD_MAX_RANK];
  LongType yCoords[SD_MAX_RANK];
  LongType gradZCoords[SD_MAX_RANK];

  // Offset variables
  LongType zOffset;
  LongType xOffset;
  LongType yOffset;
  LongType gradOffsetO;

  for (auto e = start; e < xLen; e += step) {
    // Convert linear index to coordinates for output
    INDEX2COORDS(e, outputRank, outputShapePtr, zCoords);
    // Convert coordinates back to linear index for output
    COORDS2INDEX(outputRank, outputStridePtr, zCoords, zOffset);

    // Convert linear index to coordinates for input
    INDEX2COORDS(e, inputRank, inputShapePtr, xCoords);
    // Convert coordinates back to linear index for input
    COORDS2INDEX(inputRank, inputStridePtr, xCoords, xOffset);

    // Convert linear index to coordinates for indices
    INDEX2COORDS(e, indicesRank, indicesShapePtr, yCoords);
    // Convert coordinates back to linear index for indices
    COORDS2INDEX(indicesRank, indicesStridePtr, yCoords, yOffset);

    // Retrieve the class index from indices
    auto classIndex = y[yOffset];

    // Convert class index to coordinates for gradOut (eps)
    INDEX2COORDS(classIndex, classRank, classShapePtr, gradZCoords);
    // Convert coordinates back to linear index for gradOut
    COORDS2INDEX(classRank, classStridePtr, gradZCoords, gradOffsetO);

    // Perform the computation and store the result
    z[zOffset] = static_cast<T>(gradOut[gradOffsetO] /
                                math::sd_sqrt<LongType, float>(lengths[classIndex]));
  }
}

// -------------------------------------------------------------------------------------------------------------- //

// SegmentSqrtNBPTad Kernel Refactored
template <typename T, typename I>
static SD_KERNEL void segmentSqrtNBPTadKernel(void* inputBuf, LongType const* inputShape, void* epsBuf,
                                              LongType const* epsShape, void* indicesBuf, LongType const* indicesShape,
                                              LongType* lengths, void* outputBuf, LongType const* outputShape,
                                              LongType const* inputTad, LongType const* inputOffsets,
                                              LongType const* gradOutTad, LongType const* gradOutOffsets,
                                              LongType const* outTad, LongType const* outOffsets) {
  // Reinterpret input and output buffers
  const T* x = reinterpret_cast<const T*>(inputBuf);
  const I* y = reinterpret_cast<const I*>(indicesBuf);
  T* z = reinterpret_cast<T*>(outputBuf);
  const T* gradOut = reinterpret_cast<const T*>(epsBuf);

  // Shared memory for caching shape information and related variables
  extern __shared__ unsigned char shmem[];
  // Pointers within shared memory
  LongType* sharedMem = reinterpret_cast<LongType*>(shmem);

  // Shared variables
  __shared__ LongType shared_xLen;
  __shared__ LongType shared_gradLen;
  __shared__ LongType shared_currentLen;
  __shared__ LongType shared_numOfClasses;

  // Cached shape and stride pointers
  __shared__ const LongType* shared_inputShapePtr;
  __shared__ const LongType* shared_inputStridePtr;
  __shared__ const LongType* shared_epsShapePtr;
  __shared__ const LongType* shared_epsStridePtr;
  __shared__ const LongType* shared_outputShapePtr;
  __shared__ const LongType* shared_outputStridePtr;
  __shared__ const LongType* shared_inputTadPtr;
  __shared__ const LongType* shared_inputTadOffsetsPtr;
  __shared__ const LongType* shared_gradOutTadPtr;
  __shared__ const LongType* shared_gradOutOffsetsPtr;
  __shared__ const LongType* shared_outTadPtr;
  __shared__ const LongType* shared_outOffsetsPtr;

  if (threadIdx.x == 0) {
    // Cache lengths
    shared_xLen = shape::length(inputShape);
    shared_gradLen = shape::length(epsShape);
    shared_currentLen = shape::length(outTad);
    shared_numOfClasses = shape::sizeAt(inputShape, 0); // Assuming the first dimension represents classes

    // Cache shape and stride pointers
    shared_inputShapePtr = shape::shapeOf(inputShape);
    shared_inputStridePtr = shape::stride(inputShape);
    shared_epsShapePtr = shape::shapeOf(epsShape);
    shared_epsStridePtr = shape::stride(epsShape);
    shared_outputShapePtr = shape::shapeOf(outputShape);
    shared_outputStridePtr = shape::stride(outputShape);

    // Cache TAD information
    shared_inputTadPtr = inputTad;
    shared_inputTadOffsetsPtr = inputOffsets;
    shared_gradOutTadPtr = gradOutTad;
    shared_gradOutOffsetsPtr = gradOutOffsets;
    shared_outTadPtr = outTad;
    shared_outOffsetsPtr = outOffsets;
  }

  // Ensure all threads have access to the cached values
  __syncthreads();

  // Calculate the global thread ID
  const LongType tid = blockIdx.x * blockDim.x + threadIdx.x;

  // Loop over each index in yLen assigned to this thread
  for (LongType i = blockIdx.x; i < shared_numOfClasses; i += gridDim.x) {
    // Retrieve the segment (class) for this index
    I segment = y[i];

    // Calculate pointers to the current output and corresponding gradOut
    T* currentOut = z + shared_outOffsetsPtr[i];
    T* currentGradOut = gradOut + shared_gradOutOffsetsPtr[segment];

    // Check if the segment has any elements to process
    if (lengths[segment] == 0) continue;

    // Each thread processes multiple elements based on stride
    for (LongType e = tid; e < shared_currentLen; e += blockDim.x) {
      // Convert linear index 'e' to multi-dimensional coordinates for output TAD
      LongType zCoords[SD_MAX_RANK];
      INDEX2COORDS(e, shape::rank(outTad), shared_outTadPtr, zCoords);
      LongType zIndex;
      COORDS2INDEX(shape::rank(outTad), shape::stride(outTad), zCoords, zIndex);

      // Convert linear index 'e' to multi-dimensional coordinates for gradOut TAD
      LongType gradCoords[SD_MAX_RANK];
      INDEX2COORDS(e, shape::rank(gradOutTad), shared_gradOutTadPtr, gradCoords);
      LongType gradIndex;
      COORDS2INDEX(shape::rank(gradOutTad), shape::stride(gradOutTad), gradCoords, gradIndex);

      // Compute the square root of the segment length
      T sqrtLen = math::sd_sqrt<LongType, float>(lengths[segment]);

      // Ensure sqrtLen is not zero to avoid division by zero
      if (sqrtLen == static_cast<T>(0)) {
        sqrtLen = static_cast<T>(1);
      }

      // Perform the backward pass computation
      currentOut[zIndex] = currentGradOut[gradIndex] / sqrtLen;
    }
  }
}

// -------------------------------------------------------------------------------------------------------------- //

template <typename T, typename I>
static Status unsortedSegmentSqrtNFunctorBP_(LaunchContext* context, NDArray* input, NDArray* indices,
                                                 NDArray* gradOut,
                                             LongType numOfClasses, NDArray* output) {
  auto stream = context->getCudaStream();
  NDArray::prepareSpecialUse({output}, {input, indices, gradOut});
  auto numClasses = indices->e<LongType>(indices->lengthOf() - 1) + 1;
  NDArray classesRangesLens = NDArrayFactory::create<LongType>('c', {numClasses}, context);
  NDArray classesRangesBegs = NDArrayFactory::create<LongType>('c', {numClasses}, context);
  sd::LongType zero = 0;
  sd::LongType  one = 1;
  sd::LongType  len = indices->lengthOf();
  classesRangesBegs.assign(len);
  classesRangesLens.assign(zero);
  fillUpSegments(indices, numClasses, classesRangesBegs, classesRangesLens);
  LongType* begins = reinterpret_cast<LongType*>(classesRangesBegs.specialBuffer());
  LongType* lengths = reinterpret_cast<LongType*>(classesRangesLens.specialBuffer());

  if (input->isVector()  || input->isScalar()) {
    LongType loop_size = input->lengthOf();
    auto numOfClasses = gradOut->lengthOf();
    segmentSqrtNBPLinearKernel<T, I><<<gradOut->lengthOf(), input->lengthOf(), 256, *stream>>>(
        input->specialBuffer(), input->specialShapeInfo(), gradOut->specialBuffer(), gradOut->specialShapeInfo(),
        indices->specialBuffer(), indices->specialShapeInfo(), lengths, output->specialBuffer(),
        output->specialShapeInfo());
    sd::DebugHelper::checkErrorCode(stream, "segmentSqrtNBPLinearKernel failed");

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
    dim3 segmentBpTad2 = segmentBpTad(indices->lengthOf(),input->lengthOf());

    segmentSqrtNBPTadKernel<T, I><<<segmentBpTad2.y, segmentBpTad2.x, segmentBpTad2.z, *stream>>>(
        input->specialBuffer(), input->specialShapeInfo(), gradOut->specialBuffer(), gradOut->specialShapeInfo(),
        indices->specialBuffer(), indices->specialShapeInfo(), lengths, output->specialBuffer(),
        output->specialShapeInfo(), inputTads, inputTadOffsets, gradOutTads, gradOutTadOffsets, outputTads,
        outputTadOffsets);
    sd::DebugHelper::checkErrorCode(stream, "segmentSqrtNBPTadKernel failed");

    delete dimensions;
  }
  NDArray::registerSpecialUse({output}, {input, indices, gradOut});

  return Status::OK;
}
// -------------------------------------------------------------------------------------------------------------- //
Status unsortedSegmentSqrtNFunctorBP(LaunchContext* context, NDArray* input, NDArray* indices, NDArray* gradOut,
                                     LongType numOfClasses, NDArray* output) {
  NDArray::prepareSpecialUse({output}, {input, indices, gradOut});
  BUILD_DOUBLE_SELECTOR(output->dataType(), indices->dataType(), return unsortedSegmentSqrtNFunctorBP_,
                        (context, input, indices, gradOut, numOfClasses, output), SD_FLOAT_TYPES, SD_INDEXING_TYPES);
  NDArray::registerSpecialUse({output}, {input, indices, gradOut});
}
}  // namespace helpers
}  // namespace ops
}  // namespace sd

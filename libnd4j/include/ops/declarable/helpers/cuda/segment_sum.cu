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
__global__ static void segmentSumLinearKernel(const void* inputBuf, const LongType* inputShape, LongType* starts,
                                              LongType* lengths, LongType numOfClasses, void* outputBuf,
                                              const LongType* outputShape) {
  // Reinterpret input and output buffers
  const T* input = reinterpret_cast<const T*>(inputBuf);
  T* output = reinterpret_cast<T*>(outputBuf);

  // Shared memory for caching shape information and related variables
  extern __shared__ unsigned char shmem[];
  // Pointers within shared memory
  LongType* sharedMem = reinterpret_cast<LongType*>(shmem);

  // Shared variables
  __shared__ LongType shared_inputLen;
  __shared__ LongType shared_outputLen;
  __shared__ int shared_inputRank;
  __shared__ int shared_outputRank;
  __shared__ LongType shared_zDim;
  __shared__ LongType shared_totalThreads;
  __shared__ int shared_threadsPerSegment;

  // Cached shape and stride pointers
  __shared__ const LongType* shared_inputShape;
  __shared__ const LongType* shared_inputStride;
  __shared__ const LongType* shared_outputShape;
  __shared__ const LongType* shared_outputStride;

  if (threadIdx.x == 0) {
    // Cache input tensor shape and stride
    shared_inputRank = shape::rank(inputShape);
    shared_inputShape = shape::shapeOf(inputShape);
    shared_inputStride = shape::stride(inputShape);

    // Cache output tensor shape and stride
    shared_outputRank = shape::rank(outputShape);
    shared_outputShape = shape::shapeOf(outputShape);
    shared_outputStride = shape::stride(outputShape);

    // Cache lengths
    shared_inputLen = shape::length(inputShape);
    shared_outputLen = shape::length(outputShape);

    // Calculate threads per segment
    shared_threadsPerSegment = (gridDim.x + numOfClasses - 1) / numOfClasses;

    // Calculate total number of threads across all blocks
    shared_totalThreads = gridDim.x * blockDim.x;
  }

  // Ensure all threads have access to the cached values
  __syncthreads();

  // Calculate the global thread ID
  const LongType tid = blockIdx.x * blockDim.x + threadIdx.x;

  // Allocate space in shared memory for coordinates (if needed)
  // Not used in this kernel as per the original implementation
  // LongType* coords = sharedMem + threadIdx.x * shared_inputRank;

  // Determine the segment this block is responsible for
  LongType segment = blockIdx.x / shared_threadsPerSegment;

  // Each block handles one segment
  if (threadIdx.x == 0 && segment < numOfClasses) {
    // Calculate zCoords based on the current segment
    LongType zCoords[SD_MAX_RANK];
    INDEX2COORDS(segment, shared_outputRank, shared_outputShape, zCoords);

    // Convert zCoords to linear index
    LongType zIndex;
    COORDS2INDEX(shared_outputRank, shared_outputStride, zCoords, zIndex);

    // Boundary check for zIndex
    if (zIndex >= shared_outputLen) {
      // Invalid zIndex; exit early
      return;
    }

    // Get the start and finish indices for this segment
    LongType start = starts[segment];
    LongType finish = start + lengths[segment];

    // Convert start index to coordinates
    LongType xCoords[SD_MAX_RANK];
    INDEX2COORDS(start, shared_inputRank, shared_inputShape, xCoords);

    // Convert xCoords to linear index
    LongType xOffset;
    COORDS2INDEX(shared_inputRank, shared_inputStride, xCoords, xOffset);

    // Initialize the output at zIndex with the first value from the segment
    if (xOffset < shared_inputLen) {
      output[zIndex] = input[xOffset];
    } else {
      // Handle out-of-bounds access if necessary
      output[zIndex] = static_cast<T>(0);
    }
  }

  // Ensure the initialization is complete before proceeding
  __syncthreads();

  // Now, each thread will handle adding elements to the output[zIndex]
  // We need to ensure that only threads handling valid segments proceed
  if (segment >= numOfClasses) return;

  // Get the start and finish indices for this segment
  LongType start = starts[segment];
  LongType finish = start + lengths[segment];

  // Calculate zCoords and zIndex again for use in the loop
  LongType zCoords[SD_MAX_RANK];
  INDEX2COORDS(segment, shared_outputRank, shared_outputShape, zCoords);
  LongType zIndex;
  COORDS2INDEX(shared_outputRank, shared_outputStride, zCoords, zIndex);

  if (zIndex >= shared_outputLen) return;

  // Loop over the elements in the segment assigned to this thread
  for (LongType k = tid; k < finish; k += shared_totalThreads) {
    if (k >= shared_inputLen) continue;

    // Convert linear index 'k' to multi-dimensional coordinates
    LongType xCoords[SD_MAX_RANK];
    INDEX2COORDS(k, shared_inputRank, shared_inputShape, xCoords);

    // Convert coordinates to linear index for input tensor
    LongType xOffset;
    COORDS2INDEX(shared_inputRank, shared_inputStride, xCoords, xOffset);

    if (xOffset >= shared_inputLen) continue;

    // Atomic add to ensure correct summation across threads
    math::atomics::sd_atomicAdd(&output[zIndex], input[xOffset]);
  }
}

// -------------------------------------------------------------------------------------------------------------- //
template <typename T, typename I>
__global__ static void unsortedSegmentSumLinearKernel(const void* inputBuf, const LongType* inputShape,
                                                      const void* indicesBuf, const LongType* indicesShape,
                                                      LongType* starts, LongType* lengths,
                                                      LongType numOfClasses, void* outputBuf,
                                                      const LongType* outputShape) {
  // Reinterpret input and output buffers
  const T* input = reinterpret_cast<const T*>(inputBuf);
  const I* indices = reinterpret_cast<const I*>(indicesBuf);
  T* output = reinterpret_cast<T*>(outputBuf);

  // Shared memory for caching shape information and related variables
  extern __shared__ unsigned char shmem[];
  // Pointers within shared memory
  LongType* sharedMem = reinterpret_cast<LongType*>(shmem);

  // Shared variables
  __shared__ LongType shared_inputLen;
  __shared__ LongType shared_outputLen;
  __shared__ int shared_inputRank;
  __shared__ int shared_outputRank;
  __shared__ int shared_indicesRank;
  __shared__ LongType shared_totalThreads;

  // Cached shape and stride pointers
  __shared__ const LongType* shared_inputShape;
  __shared__ const LongType* shared_inputStride;
  __shared__ const LongType* shared_outputShape;
  __shared__ const LongType* shared_outputStride;
  __shared__ const LongType* shared_indicesShape;
  __shared__ const LongType* shared_indicesStride;

  if (threadIdx.x == 0) {
    // Cache input tensor shape and stride
    shared_inputRank = shape::rank(inputShape);
    shared_inputShape = shape::shapeOf(inputShape);
    shared_inputStride = shape::stride(inputShape);

    // Cache indices tensor shape and stride
    shared_indicesRank = shape::rank(indicesShape);
    shared_indicesShape = shape::shapeOf(indicesShape);
    shared_indicesStride = shape::stride(indicesShape);

    // Cache output tensor shape and stride
    shared_outputRank = shape::rank(outputShape);
    shared_outputShape = shape::shapeOf(outputShape);
    shared_outputStride = shape::stride(outputShape);

    // Cache lengths
    shared_inputLen = shape::length(inputShape);
    shared_outputLen = shape::length(outputShape);

    // Calculate total number of threads across all blocks
    shared_totalThreads = gridDim.x * blockDim.x;
  }

  // Ensure all threads have access to the cached values
  __syncthreads();

  // Calculate the global thread ID
  const LongType tid = blockIdx.x * blockDim.x + threadIdx.x;

  // Allocate space in shared memory for coordinates
  // Assuming maximum rank is defined and consistent
  LongType* coords = sharedMem + threadIdx.x * shared_inputRank;

  // Each block is responsible for a specific class (segment)
  const LongType segment = blockIdx.x;

  // Boundary check for segment
  if (segment >= numOfClasses) return;

  // Calculate zCoords based on the current segment
  LongType zCoords[SD_MAX_RANK];
  INDEX2COORDS(segment, shared_outputRank, shared_outputShape, zCoords);

  // Convert zCoords to linear index
  LongType zIndex;
  COORDS2INDEX(shared_outputRank, shared_outputStride, zCoords, zIndex);

  // Boundary check for zIndex
  if (zIndex >= shared_outputLen) return;

  // Initialize the output at zIndex with the first value from the segment if length > 0, else 0
  if (threadIdx.x == 0) {
    if (lengths[segment] > 0) {
      LongType start = starts[segment];
      LongType xCoords[SD_MAX_RANK];
      INDEX2COORDS(start, shared_inputRank, shared_inputShape, xCoords);
      LongType xOffset;
      COORDS2INDEX(shared_inputRank, shared_inputStride, xCoords, xOffset);
      if (xOffset < shared_inputLen) {
        output[zIndex] = input[xOffset];
      } else {
        // Handle out-of-bounds access if necessary
        output[zIndex] = static_cast<T>(0);
      }
    } else {
      output[zIndex] = static_cast<T>(0);
    }
  }

  // Ensure the initialization is complete before proceeding
  __syncthreads();

  // Proceed only if the segment has elements
  if (lengths[segment] > 0) {
    // Each thread handles multiple elements based on stride
    for (LongType e = tid; e < shared_inputLen; e += shared_totalThreads) {
      // Convert linear index 'e' to multi-dimensional coordinates for input
      INDEX2COORDS(e, shared_inputRank, shared_inputShape, coords);

      // Convert coordinates to linear index for input tensor
      LongType xOffset;
      COORDS2INDEX(shared_inputRank, shared_inputStride, coords, xOffset);

      // Boundary check for xOffset
      if (xOffset >= shared_inputLen) continue;

      // Convert linear index 'e' to multi-dimensional coordinates for indices
      LongType yCoords[SD_MAX_RANK];
      INDEX2COORDS(e, shared_indicesRank, shared_indicesShape, yCoords);

      // Convert coordinates to linear index for indices tensor
      LongType yIndex;
      COORDS2INDEX(shared_indicesRank, shared_indicesStride, yCoords, yIndex);

      // Boundary check for yIndex
      if (yIndex >= shape::length(indicesShape)) continue;

      // Check if the index corresponds to the current segment and is not the start
      if (indices[yIndex] == static_cast<I>(segment) && e != starts[segment]) {
        // Atomic add to ensure correct summation across threads
        math::atomics::sd_atomicAdd(&output[zIndex], input[xOffset]);
      }
    }
  }
}

// -------------------------------------------------------------------------------------------------------------- //
// SegmentSumTad Kernel Refactored
template <typename T, typename I>
__global__ static void segmentSumTadKernel(void* inputBuf, const LongType* inputShape,
                                           const LongType* inputTads, const LongType* inputTadOffsets,
                                           const I* indices, LongType* starts,
                                           LongType* lengths, LongType numOfClasses, void* outputBuf,
                                           const LongType* outputShape,
                                           const LongType* outputTads, const LongType* outputTadOffsets,
                                           LongType numIndices) {
  // Reinterpret input and output buffers
  const T* input = reinterpret_cast<const T*>(inputBuf);
  T* output = reinterpret_cast<T*>(outputBuf);
  const I* idx = reinterpret_cast<const I*>(indices);

  // Shared memory for caching shape information and related variables
  extern __shared__ unsigned char shmem[];
  // Pointers within shared memory
  LongType* sharedMem = reinterpret_cast<LongType*>(shmem);

  // Shared variables
  __shared__ LongType shared_len;
  __shared__ LongType shared_total;
  __shared__ int shared_inputRank;
  __shared__ int shared_outputRank;
  __shared__ LongType shared_numOfClasses;
  __shared__ LongType shared_numIndices;

  // Cached shape and stride pointers
  __shared__ const LongType* shared_inputShape;
  __shared__ const LongType* shared_inputStride;
  __shared__ const LongType* shared_outputShape;
  __shared__ const LongType* shared_outputStride;
  __shared__ const LongType* shared_inputTads;
  __shared__ const LongType* shared_inputTadOffsets;
  __shared__ const LongType* shared_outputTads;
  __shared__ const LongType* shared_outputTadOffsets;

  if (threadIdx.x == 0) {
    // Cache ranks
    shared_inputRank = shape::rank(inputShape);
    shared_outputRank = shape::rank(outputShape);

    // Cache shape and stride pointers
    shared_inputShape = shape::shapeOf(inputShape);
    shared_inputStride = shape::stride(inputShape);
    shared_outputShape = shape::shapeOf(outputShape);
    shared_outputStride = shape::stride(outputShape);

    // Cache TAD information
    shared_inputTads = inputTads;
    shared_inputTadOffsets = inputTadOffsets;
    shared_outputTads = outputTads;
    shared_outputTadOffsets = outputTadOffsets;

    // Cache lengths
    shared_len = shape::length(inputTads); // Assuming shape::length(inputTads) is valid
    shared_total = shape::sizeAt(inputShape, 0); // e.g., number of segments or batches

    // Cache additional parameters
    shared_numOfClasses = numOfClasses;
    shared_numIndices = numIndices;
  }

  // Ensure all threads have access to the cached values
  __syncthreads();

  // Calculate the global thread ID
  const LongType tid = blockIdx.x * blockDim.x + threadIdx.x;

  // Each block handles one segment (class)
  for (LongType segment = blockIdx.x; segment < shared_numOfClasses; segment += gridDim.x) {
    // Retrieve the input and output TAD offsets
    LongType inputTadOffset = shared_inputTadOffsets[segment];
    LongType outputTadOffset = shared_outputTadOffsets[segment];

    // Pointer to the input and output TAD
    const T* x = input + inputTadOffset;
    T* z = output + outputTadOffset;

    // Retrieve the start and finish indices for this segment
    LongType start = starts[segment];
    LongType finish = start + lengths[segment];

    // Initialize the output for this segment
    if (threadIdx.x == 0) {
      if (lengths[segment] > 0) {
        LongType xOffset = start;
        if (xOffset < shared_len) {
          z[0] = x[xOffset];
        } else {
          // Handle out-of-bounds access if necessary
          z[0] = static_cast<T>(0);
        }
      } else {
        z[0] = static_cast<T>(0);
      }
    }

    // Ensure the initialization is complete before proceeding
    __syncthreads();

    // Proceed only if the segment has elements
    if (lengths[segment] > 0) {
      // Each thread handles multiple elements based on stride
      for (LongType e = tid; e < finish; e += shared_totalThreads) {
        if (e >= shared_len) continue;

        // Retrieve the corresponding index for this element
        I class_idx = idx[e];

        // Check if the current element belongs to this segment
        if (class_idx == static_cast<I>(segment) && e != start) {
          // Convert linear index 'e' to multi-dimensional coordinates for input TAD
          LongType inputCoords[SD_MAX_RANK];
          INDEX2COORDS(e, shared_inputRank, shared_inputShape, inputCoords);

          // Convert coordinates to linear index for input TAD
          LongType xIndex;
          COORDS2INDEX(shared_inputRank, shared_inputStride, inputCoords, xIndex);

          // Convert linear index to multi-dimensional coordinates for output TAD
          LongType outputCoords[SD_MAX_RANK];
          INDEX2COORDS(e - start, shared_outputRank, shared_outputShape, outputCoords);

          // Convert coordinates to linear index for output TAD
          LongType zIndex;
          COORDS2INDEX(shared_outputRank, shared_outputStride, outputCoords, zIndex);

          // Atomic add to accumulate the sum
          math::atomics::sd_atomicAdd(&z[zIndex], x[xIndex]);
        }
      }
    }

    // Optional: Synchronize threads if necessary (depends on specific requirements)
    __syncthreads();
  }
}

// -------------------------------------------------------------------------------------------------------------- //

template <typename T, typename I>
static void segmentSumFunctor_(LaunchContext* context, NDArray* input, NDArray* indices, NDArray* output) {
  auto stream = context->getCudaStream();
  LongType numClasses = indices->e<LongType>(indices->lengthOf() - 1) + 1;
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
  sd::LongType zero = 0;
  sd::LongType  one = 1;
  sd::LongType  len = indices->lengthOf();
  classesRangesBegs.assign(len);
  classesRangesLens.assign(zero);
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
    output->assign(zero);
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
// SegmentSumTad Kernel Refactored
template <typename T, typename I>
static SD_KERNEL void segmentSumTadKernel(void* inputBuf, const LongType* inputShape,
                                          const LongType* inputTads, const LongType* inputTadOffsets,
                                          const I* indices, LongType* starts,
                                          LongType* lengths, LongType numOfClasses, void* outputBuf,
                                          const LongType* outputShape,
                                          const LongType* outputTads, const LongType* outputTadOffsets,
                                          LongType numIndices) {
  // Reinterpret input and output buffers
  const T* input = reinterpret_cast<const T*>(inputBuf);
  T* output = reinterpret_cast<T*>(outputBuf);
  const I* idx = reinterpret_cast<const I*>(indices);

  // Shared memory for caching shape information and related variables
  extern __shared__ unsigned char shmem[];
  // Pointers within shared memory
  LongType* sharedMem = reinterpret_cast<LongType*>(shmem);

  // Shared variables
  __shared__ LongType shared_len;
  __shared__ LongType shared_total;
  __shared__ int shared_inputRank;
  __shared__ int shared_outputRank;

  // Cached shape and stride pointers
  __shared__ const LongType* shared_inputShapePtr;
  __shared__ const LongType* shared_inputStridePtr;
  __shared__ const LongType* shared_outputShapePtr;
  __shared__ const LongType* shared_outputStridePtr;
  __shared__ const LongType* shared_inputTadsPtr;
  __shared__ const LongType* shared_inputTadOffsetsPtr;
  __shared__ const LongType* shared_outputTadsPtr;
  __shared__ const LongType* shared_outputTadOffsetsPtr;

  if (threadIdx.x == 0) {
    // Cache ranks
    shared_inputRank = shape::rank(inputShape);
    shared_outputRank = shape::rank(outputShape);

    // Cache shape and stride pointers
    shared_inputShapePtr = shape::shapeOf(inputShape);
    shared_inputStridePtr = shape::stride(inputShape);
    shared_outputShapePtr = shape::shapeOf(outputShape);
    shared_outputStridePtr = shape::stride(outputShape);

    // Cache TAD information
    shared_inputTadsPtr = inputTads;
    shared_inputTadOffsetsPtr = inputTadOffsets;
    shared_outputTadsPtr = outputTads;
    shared_outputTadOffsetsPtr = outputTadOffsets;

    // Cache lengths
    shared_len = shape::length(inputTads); // Assuming shape::length(inputTads) is valid
    shared_total = shape::sizeAt(inputShape, 0); // e.g., number of segments or batches
  }

  // Ensure all threads have access to the cached values
  __syncthreads();

  // Calculate the global thread ID
  const LongType tid = blockIdx.x * blockDim.x + threadIdx.x;

  // Each block handles one or multiple segments (classes)
  for (LongType idxSegment = blockIdx.x; idxSegment < numOfClasses; idxSegment += gridDim.x) {
    // Boundary check for segment
    if (idxSegment >= numOfClasses) continue;

    // Retrieve the input and output TAD offsets
    LongType inputTadOffset = shared_inputTadOffsetsPtr[idxSegment];
    LongType outputTadOffset = shared_outputTadOffsetsPtr[idxSegment];

    // Pointer to the input and output TAD
    const T* x = input + inputTadOffset;
    T* z = output + outputTadOffset;

    // Retrieve the start and finish indices for this segment
    LongType start = starts[idxSegment];
    LongType finish = start + lengths[idxSegment];

    // Initialize the output for this segment if lengths > 0
    if (threadIdx.x == 0) {
      if (lengths[idxSegment] > 0) {
        LongType firstElementOffset = start;
        if (firstElementOffset < shared_len) {
          z[0] = x[firstElementOffset];
        } else {
          // Handle out-of-bounds access if necessary
          z[0] = static_cast<T>(0);
        }
      } else {
        z[0] = static_cast<T>(0);
      }
    }

    // Ensure the initialization is complete before proceeding
    __syncthreads();

    // Proceed only if the segment has elements
    if (lengths[idxSegment] > 0) {
      // Each thread handles multiple elements based on stride
      for (LongType e = tid; e < finish; e += shared_total) {
        if (e >= shared_len) continue;

        // Retrieve the corresponding index for this element
        I class_idx = idx[e];

        // Check if the current element belongs to this segment
        if (class_idx == static_cast<I>(idxSegment) && e != start) {
          // Convert linear index 'e' to multi-dimensional coordinates for input TAD
          LongType xCoords[SD_MAX_RANK];
          INDEX2COORDS(e, shared_inputRank, shared_inputShapePtr, xCoords);

          // Convert coordinates to linear index for input TAD
          LongType xIndex;
          COORDS2INDEX(shared_inputRank, shared_inputStridePtr, xCoords, xIndex);

          // Convert linear index to multi-dimensional coordinates for output TAD
          LongType zCoords[SD_MAX_RANK];
          INDEX2COORDS(e - start, shared_outputRank, shared_outputShapePtr, zCoords);

          // Convert coordinates to linear index for output TAD
          LongType zIndex;
          COORDS2INDEX(shared_outputRank, shared_outputStridePtr, zCoords, zIndex);

          // Atomic add to accumulate the sum
          math::atomics::sd_atomicAdd(&z[zIndex], x[xIndex]);
        }
      }
    }

    // Optional: Synchronize threads if necessary (depends on specific requirements)
    __syncthreads();
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
    LongType yCoords[SD_MAX_RANK];
    LongType yIndex;
    INDEX2COORDS(i, shape::rank(indicesShape), shape::shapeOf(indicesShape), yCoords);
    COORDS2INDEX(shape::rank(indicesShape), shape::stride(indicesShape), yCoords, yIndex);
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

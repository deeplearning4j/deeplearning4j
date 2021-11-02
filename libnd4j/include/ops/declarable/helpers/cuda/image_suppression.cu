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
//  @author sgazeos@gmail.com
//
#include <array/NDArrayFactory.h>
#include <exceptions/cuda_exception.h>
#include <legacy/NativeOps.h>
#include <ops/declarable/helpers/image_suppression.h>

#include <queue>

namespace sd {
namespace ops {
namespace helpers {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// needToSuppressWithThreshold - predicate for suppression
//      boxes - boxes tensor buffer
//      boxesShape boxes tensor shape
//      previousIndex - index for current pos value
//      nextIndex - index for neighbor pos value
//      threshold - threashold value to suppress
//
//      return value: true, if threshold is overcome, false otherwise
//
template <typename T>
static SD_DEVICE bool needToSuppressWithThreshold(T* boxes, sd::LongType const* boxesShape, int previousIndex,
                                                  int nextIndex, T threshold) {
  sd::LongType previous0[] = {previousIndex, 0};
  sd::LongType previous1[] = {previousIndex, 1};
  sd::LongType previous2[] = {previousIndex, 2};
  sd::LongType previous3[] = {previousIndex, 3};
  sd::LongType next0[] = {nextIndex, 0};
  sd::LongType next1[] = {nextIndex, 1};
  sd::LongType next2[] = {nextIndex, 2};
  sd::LongType next3[] = {nextIndex, 3};

  // we have rectangle with given max values. Compute vexes of rectangle first

  T minYPrev =
      sd::math::sd_min(boxes[shape::getOffset(boxesShape, previous0)], boxes[shape::getOffset(boxesShape, previous2)]);
  T minXPrev =
      sd::math::sd_min(boxes[shape::getOffset(boxesShape, previous1)], boxes[shape::getOffset(boxesShape, previous3)]);
  T maxYPrev =
      sd::math::sd_max(boxes[shape::getOffset(boxesShape, previous0)], boxes[shape::getOffset(boxesShape, previous2)]);
  T maxXPrev =
      sd::math::sd_max(boxes[shape::getOffset(boxesShape, previous1)], boxes[shape::getOffset(boxesShape, previous3)]);
  T minYNext = sd::math::sd_min(boxes[shape::getOffset(boxesShape, next0)], boxes[shape::getOffset(boxesShape, next2)]);
  T minXNext = sd::math::sd_min(boxes[shape::getOffset(boxesShape, next1)], boxes[shape::getOffset(boxesShape, next3)]);
  T maxYNext = sd::math::sd_max(boxes[shape::getOffset(boxesShape, next0)], boxes[shape::getOffset(boxesShape, next2)]);
  T maxXNext = sd::math::sd_max(boxes[shape::getOffset(boxesShape, next1)], boxes[shape::getOffset(boxesShape, next3)]);

  // compute areas for comparation
  T areaPrev = (maxYPrev - minYPrev) * (maxXPrev - minXPrev);
  T areaNext = (maxYNext - minYNext) * (maxXNext - minXNext);

  // of course, areas should be positive
  if (areaNext <= T(0.f) || areaPrev <= T(0.f)) return false;

  // compute intersection of rectangles
  T minIntersectionY = sd::math::sd_max(minYPrev, minYNext);
  T minIntersectionX = sd::math::sd_max(minXPrev, minXNext);
  T maxIntersectionY = sd::math::sd_min(maxYPrev, maxYNext);
  T maxIntersectionX = sd::math::sd_min(maxXPrev, maxXNext);
  T intersectionArea = sd::math::sd_max(T(maxIntersectionY - minIntersectionY), T(0.0f)) *
                       sd::math::sd_max(T(maxIntersectionX - minIntersectionX), T(0.0f));
  T intersectionValue = intersectionArea / (areaPrev + areaNext - intersectionArea);
  // final check
  return intersectionValue > threshold;
}

template <typename T>
static SD_DEVICE T similirityV3(T* boxes, sd::LongType const* boxesShape, int previousIndex, int nextIndex) {
  sd::LongType previous0[] = {previousIndex, 0};
  sd::LongType previous1[] = {previousIndex, 1};
  sd::LongType previous2[] = {previousIndex, 2};
  sd::LongType previous3[] = {previousIndex, 3};
  sd::LongType next0[] = {nextIndex, 0};
  sd::LongType next1[] = {nextIndex, 1};
  sd::LongType next2[] = {nextIndex, 2};
  sd::LongType next3[] = {nextIndex, 3};

  // we have rectangle with given max values. Compute vexes of rectangle first

  T minYPrev =
      sd::math::sd_min(boxes[shape::getOffset(boxesShape, previous0)], boxes[shape::getOffset(boxesShape, previous2)]);
  T minXPrev =
      sd::math::sd_min(boxes[shape::getOffset(boxesShape, previous1)], boxes[shape::getOffset(boxesShape, previous3)]);
  T maxYPrev =
      sd::math::sd_max(boxes[shape::getOffset(boxesShape, previous0)], boxes[shape::getOffset(boxesShape, previous2)]);
  T maxXPrev =
      sd::math::sd_max(boxes[shape::getOffset(boxesShape, previous1)], boxes[shape::getOffset(boxesShape, previous3)]);
  T minYNext = sd::math::sd_min(boxes[shape::getOffset(boxesShape, next0)], boxes[shape::getOffset(boxesShape, next2)]);
  T minXNext = sd::math::sd_min(boxes[shape::getOffset(boxesShape, next1)], boxes[shape::getOffset(boxesShape, next3)]);
  T maxYNext = sd::math::sd_max(boxes[shape::getOffset(boxesShape, next0)], boxes[shape::getOffset(boxesShape, next2)]);
  T maxXNext = sd::math::sd_max(boxes[shape::getOffset(boxesShape, next1)], boxes[shape::getOffset(boxesShape, next3)]);

  // compute areas for comparation
  T areaPrev = (maxYPrev - minYPrev) * (maxXPrev - minXPrev);
  T areaNext = (maxYNext - minYNext) * (maxXNext - minXNext);

  // of course, areas should be positive
  if (areaNext <= T(0.f) || areaPrev <= T(0.f)) return false;

  // compute intersection of rectangles
  T minIntersectionY = sd::math::sd_max(minYPrev, minYNext);
  T minIntersectionX = sd::math::sd_max(minXPrev, minXNext);
  T maxIntersectionY = sd::math::sd_min(maxYPrev, maxYNext);
  T maxIntersectionX = sd::math::sd_min(maxXPrev, maxXNext);
  T intersectionArea = sd::math::sd_max(T(maxIntersectionY - minIntersectionY), T(0.0f)) *
                       sd::math::sd_max(T(maxIntersectionX - minIntersectionX), T(0.0f));
  T intersectionValue = intersectionArea / (areaPrev + areaNext - intersectionArea);
  // final check
  return intersectionValue;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// shouldSelectKernel - compute status for all selected rectangles (boxes)
//
// we compute boolean flag as shared uint32 and return it on final only for the first thread
//
template <typename T, typename I>
static SD_KERNEL void shouldSelectKernel(T* boxesBuf, sd::LongType const* boxesShape, I* indexBuf,
                                         I* selectedIndicesData, double threshold, int numSelected, int i,
                                         bool* shouldSelect) {
  auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  auto step = gridDim.x * blockDim.x;
  __shared__ unsigned int shouldSelectShared;
  if (threadIdx.x == 0) {
    shouldSelectShared = (unsigned int)shouldSelect[0];
  }
  __syncthreads();
  for (int j = numSelected - 1 - tid; j >= 0; j -= step) {
    if (shouldSelectShared) {
      if (needToSuppressWithThreshold(boxesBuf, boxesShape, indexBuf[i], indexBuf[selectedIndicesData[j]],
                                      T(threshold)))
        atomicCAS(&shouldSelectShared, 1, 0);  // exchange only when need to suppress
    }
  }
  __syncthreads();

  // final move: collect result
  if (threadIdx.x == 0) {
    *shouldSelect = shouldSelectShared > 0;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// indices - type depended, indicesLong - type defined (only 64bit integers)
//
template <typename I>
static SD_KERNEL void copyIndices(void* indices, void* indicesLong, sd::LongType len) {
  I* indexBuf = reinterpret_cast<I*>(indices);
  sd::LongType* srcBuf = reinterpret_cast<sd::LongType*>(indicesLong);
  ;

  auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  auto step = blockDim.x * gridDim.x;

  for (auto i = tid; i < len; i += step) indexBuf[i] = (I)srcBuf[i];
}

template <typename T, typename I>
static SD_KERNEL void suppressScores(T* scores, I* indices, sd::LongType length, T scoreThreshold) {
  auto start = blockIdx.x * blockDim.x;
  auto step = gridDim.x * blockDim.x;

  for (auto e = start + threadIdx.x; e < (int)length; e += step) {
    if (scores[e] < scoreThreshold) {
      scores[e] = scoreThreshold;
      indices[e] = -1;
    } else {
      indices[e] = I(e);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// nonMaxSuppressionV2 algorithm - given from TF NonMaxSuppressionV2 implementation
//
template <typename T, typename I>
static void nonMaxSuppressionV2_(sd::LaunchContext* context, NDArray* boxes, NDArray* scales, int maxSize,
                                 double threshold, double scoreThreshold, NDArray* output) {
  auto stream = context->getCudaStream();
  NDArray::prepareSpecialUse({output}, {boxes, scales});
  std::unique_ptr<NDArray> indices(NDArrayFactory::create_<I>(
      'c', {scales->lengthOf()}, context));  // - 1, scales->lengthOf()); //, scales->getContext());

  NDArray scores(*scales);
  sd::Pointer extras[2] = {nullptr, stream};
  auto indexBuf = indices->dataBuffer()->specialAsT<I>();  /// reinterpret_cast<I*>(indices->specialBuffer());
  auto scoreBuf = scores.dataBuffer()->specialAsT<T>();
  suppressScores<T, I><<<128, 128, 128, *stream>>>(scoreBuf, indexBuf, scores.lengthOf(), T(scoreThreshold));
  indices->tickWriteDevice();
  sortByValue(extras, indices->buffer(), indices->shapeInfo(), indices->specialBuffer(), indices->specialShapeInfo(),
              scores.buffer(), scores.shapeInfo(), scores.specialBuffer(), scores.specialShapeInfo(), true);
  indices->tickWriteDevice();
  NDArray selectedIndices = NDArrayFactory::create<I>('c', {output->lengthOf()}, context);
  int numSelected = 0;
  int numBoxes = boxes->sizeAt(0);
  auto boxesBuf = reinterpret_cast<T*>(boxes->specialBuffer());

  auto selectedIndicesData = reinterpret_cast<I*>(selectedIndices.specialBuffer());
  auto outputBuf = reinterpret_cast<I*>(output->specialBuffer());

  bool* shouldSelectD;
  auto err = cudaMalloc(&shouldSelectD, sizeof(bool));
  if (err) {
    throw cuda_exception::build("helpers::nonMaxSuppressionV2: Cannot allocate memory for bool flag", err);
  }
  for (I i = 0; i < boxes->sizeAt(0); ++i) {
    bool shouldSelect = numSelected < output->lengthOf();
    if (shouldSelect) {
      err = cudaMemcpy(shouldSelectD, &shouldSelect, sizeof(bool), cudaMemcpyHostToDevice);
      if (err) {
        throw cuda_exception::build("helpers::nonMaxSuppressionV2: Cannot set up bool flag to device", err);
      }

      shouldSelectKernel<T, I><<<128, 256, 1024, *stream>>>(
          boxesBuf, boxes->specialShapeInfo(), indexBuf, selectedIndicesData, threshold, numSelected, i, shouldSelectD);
      err = cudaMemcpy(&shouldSelect, shouldSelectD, sizeof(bool), cudaMemcpyDeviceToHost);
      if (err) {
        throw cuda_exception::build("helpers::nonMaxSuppressionV2: Cannot set up bool flag to host", err);
      }
    }

    if (shouldSelect) {
      cudaMemcpy(reinterpret_cast<I*>(output->specialBuffer()) + numSelected, indexBuf + i, sizeof(I),
                 cudaMemcpyDeviceToDevice);
      cudaMemcpy(selectedIndicesData + numSelected, &i, sizeof(I), cudaMemcpyHostToDevice);
      numSelected++;
    }
  }

  err = cudaFree(shouldSelectD);
  if (err) {
    throw cuda_exception::build("helpers::nonMaxSuppressionV2: Cannot deallocate memory for bool flag", err);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, typename I>
static SD_DEVICE bool checkOverlapBoxes(T* boxes, sd::LongType const* shape, T* scores, I* indices, I* selectedIndices,
                                        I* startIndices, I selectedSize, I nextCandidateIndex, T overlapThreshold,
                                        T scoreThreshold, bool simple) {
  bool shouldHardSuppress = false;
  T& nextCandidateScore = scores[nextCandidateIndex];
  I selectedIndex = indices[nextCandidateIndex];
  I finish = startIndices[nextCandidateIndex];

  for (int j = selectedSize; j > finish; --j) {
    T boxVal;
    if (simple) {
      sd::LongType xPos[] = {selectedIndex, selectedIndices[j - 1]};
      auto xShift = shape::getOffset(shape, xPos, 0);
      boxVal = boxes[xShift];
    } else {
      boxVal = similirityV3(boxes, shape, selectedIndex, selectedIndices[j - 1]);
    }
    if (boxVal > static_cast<T>(overlapThreshold)) nextCandidateScore = static_cast<T>(0.f);

    // First decide whether to perform hard suppression
    if (boxVal >= overlapThreshold) {
      shouldHardSuppress = true;
      break;
    }

    // If nextCandidate survives hard suppression, apply soft suppression
    if (nextCandidateScore <= static_cast<T>(scoreThreshold)) break;
  }

  return shouldHardSuppress;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, typename I>
static SD_KERNEL void suppressNonMaxOverlapKernel(T* boxes, sd::LongType const* boxesShape, T* scoresData, I* indices,
                                                  I* startIndices, sd::LongType length, I maxOutputLen,
                                                  T overlapThreshold, T scoreThreshold, I* output,
                                                  sd::LongType const* outputShape, I* outputLength, bool simple) {
  __shared__ I selectedSize;
  __shared__ I* tempOutput;

  if (threadIdx.x == 0) {
    selectedSize = outputLength ? *outputLength : maxOutputLen;
    extern __shared__ unsigned char shmem[];
    tempOutput = (I*)shmem;
  }
  __syncthreads();

  auto start = blockIdx.x * blockDim.x;
  auto step = blockDim.x * gridDim.x;

  for (I nextCandidateIndex = start + threadIdx.x; selectedSize < maxOutputLen && nextCandidateIndex < (I)length;) {
    auto originalScore = scoresData[nextCandidateIndex];  // nextCandidate._score;
    I nextCandidateBoxIndex = indices[nextCandidateIndex];
    auto selectedSizeMark = selectedSize;

    // skip for cases when index is less than 0 (under score threshold)
    if (nextCandidateBoxIndex < 0) {
      nextCandidateIndex += step;
      continue;
    }
    // check for overlaps
    bool shouldHardSuppress =
        checkOverlapBoxes(boxes, boxesShape, scoresData, indices, tempOutput, startIndices, selectedSize,
                          nextCandidateIndex, overlapThreshold, scoreThreshold, simple);  // false;
    T nextCandidateScore = scoresData[nextCandidateIndex];

    startIndices[nextCandidateIndex] = selectedSize;
    if (!shouldHardSuppress) {
      if (nextCandidateScore == originalScore) {
        // Suppression has not occurred, so select nextCandidate
        if (output) output[selectedSize] = nextCandidateBoxIndex;
        tempOutput[selectedSize] = nextCandidateBoxIndex;
        math::atomics::sd_atomicAdd(&selectedSize, (I)1);
      }

      if (nextCandidateScore > scoreThreshold) {
        // Soft suppression has occurred and current score is still greater than
        // scoreThreshold; add nextCandidate back onto priority queue.
        continue;  // in some cases, this index not 0
      }
    }
    nextCandidateIndex += step;
  }

  if (threadIdx.x == 0) {
    if (outputLength) *outputLength = selectedSize;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, typename I>
static sd::LongType nonMaxSuppressionGeneric_(sd::LaunchContext* context, NDArray* boxes, NDArray* scores,
                                              int outputSize, double overlapThreshold, double scoreThreshold,
                                              NDArray* output, bool simple) {
  auto stream = context->getCudaStream();
  if (output)
    NDArray::prepareSpecialUse({output}, {boxes, scores});
  else {
    if (!boxes->isActualOnDeviceSide()) boxes->syncToDevice();
    if (!scores->isActualOnDeviceSide()) scores->syncToDevice();
  }

  NDArray indices = NDArrayFactory::create<I>('c', {scores->lengthOf()},
                                              context);  // - 1, scales->lengthOf()); //, scales->getContext());
  NDArray startPositions = NDArrayFactory::create<I>('c', {scores->lengthOf()}, context);
  NDArray selectedScores(*scores);
  sd::Pointer extras[2] = {nullptr, stream};
  auto indexBuf = indices.dataBuffer()->specialAsT<I>();  /// reinterpret_cast<I*>(indices->specialBuffer());

  suppressScores<<<128, 128, 128, *stream>>>(selectedScores.dataBuffer()->specialAsT<T>(), indexBuf,
                                             selectedScores.lengthOf(), T(scoreThreshold));

  sortByValue(extras, indices.buffer(), indices.shapeInfo(), indices.specialBuffer(), indices.specialShapeInfo(),
              selectedScores.buffer(), selectedScores.shapeInfo(), selectedScores.specialBuffer(),
              selectedScores.specialShapeInfo(), true);
  indices.tickWriteDevice();
  selectedScores.tickWriteDevice();

  auto scoresData = selectedScores.dataBuffer()->specialAsT<T>();  //, numBoxes, scoresData.begin());

  auto startIndices = startPositions.dataBuffer()->specialAsT<I>();
  I selectedSize = 0;
  sd::LongType res = 0;
  if (output) {  // this part used when output shape already calculated to fill up values on output
    DataBuffer selectedSizeBuf(&selectedSize, sizeof(I), DataTypeUtils::fromT<I>());
    suppressNonMaxOverlapKernel<<<1, 1, 1024, *stream>>>(
        boxes->dataBuffer()->specialAsT<T>(), boxes->specialShapeInfo(), scoresData, indexBuf, startIndices,
        scores->lengthOf(), (I)outputSize, T(overlapThreshold), T(scoreThreshold),
        output->dataBuffer()->specialAsT<I>(), output->specialShapeInfo(), selectedSizeBuf.specialAsT<I>(), simple);
  } else {  // this case used on calculation of output shape. Output and output shape shoulde be nullptr.
    DataBuffer selectedSizeBuf(&selectedSize, sizeof(I), DataTypeUtils::fromT<I>());
    suppressNonMaxOverlapKernel<<<1, 1, 1024, *stream>>>(
        boxes->dataBuffer()->specialAsT<T>(), boxes->specialShapeInfo(), scoresData, indexBuf, startIndices,
        scores->lengthOf(), (I)outputSize, T(overlapThreshold), T(scoreThreshold), (I*)nullptr, (sd::LongType*)nullptr,
        selectedSizeBuf.specialAsT<I>(), simple);
    selectedSizeBuf.syncToPrimary(context, true);
    res = *selectedSizeBuf.primaryAsT<I>();
  }

  if (output) NDArray::registerSpecialUse({output}, {boxes, scores});

  return res;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void nonMaxSuppression(sd::LaunchContext* context, NDArray* boxes, NDArray* scales, int maxSize, double threshold,
                       double scoreThreshold, NDArray* output) {
  BUILD_DOUBLE_SELECTOR(boxes->dataType(), output->dataType(), nonMaxSuppressionV2_,
                        (context, boxes, scales, maxSize, threshold, scoreThreshold, output), SD_FLOAT_TYPES,
                        SD_INDEXING_TYPES);
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

sd::LongType nonMaxSuppressionGeneric(sd::LaunchContext* context, NDArray* boxes, NDArray* scales, int maxSize,
                                      double threshold, double scoreThreshold, NDArray* output) {
  BUILD_DOUBLE_SELECTOR(
      boxes->dataType(), output ? output->dataType() : DataType::INT32, return nonMaxSuppressionGeneric_,
      (context, boxes, scales, maxSize, threshold, scoreThreshold, output, true), SD_FLOAT_TYPES, SD_INDEXING_TYPES);
  return boxes->sizeAt(0);
}

sd::LongType nonMaxSuppressionV3(sd::LaunchContext* context, NDArray* boxes, NDArray* scores, int maxSize,
                                 double overlapThreshold, double scoreThreshold, NDArray* output) {
  BUILD_DOUBLE_SELECTOR(boxes->dataType(), output ? output->dataType() : DataType::INT32,
                        return nonMaxSuppressionGeneric_,
                        (context, boxes, scores, maxSize, overlapThreshold, scoreThreshold, output, false),
                        SD_FLOAT_TYPES, SD_INDEXING_TYPES);
  return boxes->sizeAt(0);
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd

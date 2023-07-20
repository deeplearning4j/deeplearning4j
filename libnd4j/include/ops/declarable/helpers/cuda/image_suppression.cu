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

#include "execution/cuda/LaunchDims.h"

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
static  inline T similirityV3_(NDArray const& boxes, sd::LongType i, sd::LongType j) {
  const T zero = static_cast<T>(0.f);
  const T yminI = math::sd_min(boxes.t<T>(i, 0), boxes.t<T>(i, 2));
  const T xminI = math::sd_min(boxes.t<T>(i, 1), boxes.t<T>(i, 3));
  const T ymaxI = math::sd_max(boxes.t<T>(i, 0), boxes.t<T>(i, 2));
  const T xmaxI = math::sd_max(boxes.t<T>(i, 1), boxes.t<T>(i, 3));
  const T yminJ = math::sd_min(boxes.t<T>(j, 0), boxes.t<T>(j, 2));
  const T xminJ = math::sd_min(boxes.t<T>(j, 1), boxes.t<T>(j, 3));
  const T ymaxJ = math::sd_max(boxes.t<T>(j, 0), boxes.t<T>(j, 2));
  const T xmaxJ = math::sd_max(boxes.t<T>(j, 1), boxes.t<T>(j, 3));
  const T areaI = (ymaxI - yminI) * (xmaxI - xminI);
  const T areaJ = (ymaxJ - yminJ) * (xmaxJ - xminJ);
  if (areaI <= zero || areaJ <= zero) {
    return zero;
  }
  const T intersectionYmin = math::sd_max(yminI, yminJ);
  const T intersectionXmin = math::sd_max(xminI, xminJ);
  const T intersectionYmax = math::sd_min(ymaxI, ymaxJ);
  const T intersectionXmax = math::sd_min(xmaxI, xmaxJ);
  const T intersectionY = intersectionYmax - intersectionYmin;
  const T intersectionX = intersectionXmax - intersectionXmin;
  const T intersectionArea = math::sd_max(intersectionY, zero) * math::sd_max(intersectionX, zero);
  return intersectionArea / (areaI + areaJ - intersectionArea);
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

  // compute areas for comparator
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
  auto indexBuf = indices->dataBuffer()->specialAsT<I>();
  auto scoreBuf = scores.dataBuffer()->specialAsT<T>();
  dim3 launchDims = getLaunchDims("image_suppress_scores");
  suppressScores<T, I><<<launchDims.x, launchDims.y,launchDims.z, *stream>>>(scoreBuf, indexBuf, scores.lengthOf(), T(scoreThreshold));
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

      dim3 selectDims = getLaunchDims("image_suppress_select");
      shouldSelectKernel<T, I><<<selectDims.y,selectDims.x,selectDims.z, *stream>>>(
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
    auto originalScore = scoresData[nextCandidateIndex];
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
        I currSize = math::atomics::sd_atomicAdd(&selectedSize, (I)1);
        if (output) {
          printf("Setting output currSize: %i, nextCandidateBoxIndex: %i\n", currSize, nextCandidateBoxIndex);
          output[currSize] = nextCandidateBoxIndex;
        }
        tempOutput[currSize] = nextCandidateBoxIndex;
        printf(" tempOutput: currSize: %i, nextCandidateBoxIndex: %i\n", currSize, nextCandidateBoxIndex);

      }

      if ((float) nextCandidateScore > (float) scoreThreshold) {
        // Soft suppression has occurred and current score is still greater than
        // scoreThreshold; add nextCandidate back onto priority queue.
        continue;  // in some cases, this index not 0
      }
    }
    nextCandidateIndex += step;
  }

  __syncthreads();


  if (threadIdx.x == 0) {
    printf("selectedSize: %i\n", selectedSize);
    if (outputLength) *outputLength = selectedSize;
  }
}


typedef NDArray (*SimilarityFunc)(NDArray const& boxes, sd::LongType i, sd::LongType j);
template <typename T>
static inline T similarityOverlaps_(NDArray const& boxes, sd::LongType i, sd::LongType j) {
  return boxes.t<T>(i, j);
}

static NDArray similiratyOverlaps(NDArray const& boxes, sd::LongType i, sd::LongType j) {
  NDArray res(boxes.dataType(), boxes.getContext());  // = NDArrayFactory::create(0.);
  BUILD_SINGLE_SELECTOR(boxes.dataType(), res = similarityOverlaps_, (boxes, i, j), SD_FLOAT_TYPES);
  return res;
}

static NDArray similarityV3(NDArray const& boxes, sd::LongType i, sd::LongType j) {
  NDArray res(boxes.dataType(), boxes.getContext());  // = NDArrayFactory::create(0.);
  BUILD_SINGLE_SELECTOR(boxes.dataType(), res = similirityV3_, (boxes, i, j), SD_FLOAT_TYPES);
  return res;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, typename I>
static sd::LongType nonMaxSuppressionGeneric_(sd::LaunchContext* context, NDArray* boxes, NDArray* scores,
                                              int outputSize, float overlapThreshold, float scoreThreshold,
                                              NDArray* output, SimilarityFunc f) {
  auto stream = context->getCudaStream();
  if (output)
    NDArray::preparePrimaryUse({output}, {boxes, scores});
  else {
    if (!boxes->isActualOnHostSide()) boxes->syncToHost();
    if (!scores->isActualOnHostSide()) scores->syncToHost();
  }

  auto numBoxes = boxes->sizeAt(0);
  T* scoresData = scores->dataBuffer()->primaryAsT<T>();

  // Data structure for a selection candidate in NMS.
  struct Candidate {
    int _boxIndex;
    T _score;
    int _suppressBeginIndex;
  };

  auto cmp = [](const Candidate& bsI, const Candidate& bsJ) -> bool {
    return ((bsI._score == bsJ._score) && (bsI._boxIndex > bsJ._boxIndex)) || (bsI._score < bsJ._score);
  };

  std::priority_queue<Candidate, std::deque<Candidate>, decltype(cmp)> candidatePriorityQueue(cmp);
  for (auto i = 0; i < scores->lengthOf(); ++i) {
    if ((float)scoresData[i] > (float)scoreThreshold) {
      candidatePriorityQueue.emplace(Candidate({i, scoresData[i], 0}));
    }
  }

  std::vector<I> selected;
  T similarity, originalScore;
  Candidate nextCandidate;

  while (selected.size() < outputSize && !candidatePriorityQueue.empty()) {
    nextCandidate = candidatePriorityQueue.top();
    originalScore = nextCandidate._score;
    candidatePriorityQueue.pop();

    // Overlapping boxes are likely to have similar scores, therefore we
    // iterate through the previously selected boxes backwards in order to
    // see if `nextCandidate` should be suppressed. We also enforce a property
    // that a candidate can be suppressed by another candidate no more than
    // once via `suppress_begin_index` which tracks which previously selected
    // boxes have already been compared against next_candidate prior to a given
    // iteration.  These previous selected boxes are then skipped over in the
    // following loop.
    bool shouldHardSuppress = false;
    for (int j = static_cast<int>(selected.size()) - 1; j >= nextCandidate._suppressBeginIndex; --j) {
      auto similarityA =
          f(*boxes, nextCandidate._boxIndex, selected[j]);  // boxes->t<T>(nextCandidate._boxIndex, selected[j]);
      similarity = similarityA.template t<T>(0);
      nextCandidate._score *= T(similarity <= overlapThreshold ? 1.0 : 0.);  // suppressWeightFunc(similarity);

      // First decide whether to perform hard suppression
      if ((float)similarity >= static_cast<float>(overlapThreshold)) {
        shouldHardSuppress = true;
        break;
      }

      // If next_candidate survives hard suppression, apply soft suppression
      if ((float)nextCandidate._score <= (float)scoreThreshold) break;
    }
    // If `nextCandidate._score` has not dropped below `scoreThreshold`
    // by this point, then we know that we went through all of the previous
    // selections and can safely update `suppress_begin_index` to
    // `selected.size()`. If on the other hand `next_candidate.score`
    // *has* dropped below the score threshold, then since `suppressWeight`
    // always returns values in [0, 1], further suppression by items that were
    // not covered in the above for loop would not have caused the algorithm
    // to select this item. We thus do the same update to
    // `suppressBeginIndex`, but really, this element will not be added back
    // into the priority queue in the following.
    nextCandidate._suppressBeginIndex = selected.size();

    if (!shouldHardSuppress) {
      if (nextCandidate._score == originalScore) {
        // Suppression has not occurred, so select next_candidate
        selected.push_back(nextCandidate._boxIndex);
      }
      if ((float)nextCandidate._score > (float)scoreThreshold) {
        // Soft suppression has occurred and current score is still greater than
        // score_threshold; add next_candidate back onto priority queue.
        candidatePriorityQueue.push(nextCandidate);
      }
    }
  }

  if (output) {
    DataBuffer buf(selected.data(), selected.size() * sizeof(I), DataTypeUtils::fromT<I>());
    output->dataBuffer()->copyBufferFrom(buf, buf.getLenInBytes());
  }

  return (sd::LongType)selected.size();
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
      (context, boxes, scales, maxSize, threshold, scoreThreshold, output, similiratyOverlaps), SD_FLOAT_TYPES, SD_INDEXING_TYPES);
  return boxes->sizeAt(0);
}

sd::LongType nonMaxSuppressionV3(sd::LaunchContext* context, NDArray* boxes, NDArray* scores, int maxSize,
                                 double overlapThreshold, double scoreThreshold, NDArray* output) {
  BUILD_DOUBLE_SELECTOR(boxes->dataType(), output ? output->dataType() : DataType::INT32,
                        return nonMaxSuppressionGeneric_,
                        (context, boxes, scores, maxSize, overlapThreshold, scoreThreshold, output, similarityV3),
                        SD_FLOAT_TYPES, SD_INDEXING_TYPES);
  return boxes->sizeAt(0);
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd

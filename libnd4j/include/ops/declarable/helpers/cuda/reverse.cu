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
// @author Yurii Shyrma, created on 16.04.2018
//
#include <array/ResultSet.h>
#include <helpers/ConstantTadHelper.h>
#include <helpers/PointersManager.h>
#include <helpers/ShapeUtils.h>
#include <helpers/TAD.h>
#include <ops/declarable/helpers/reverse.h>

#include "execution/cuda/LaunchDims.h"


namespace sd {
namespace ops {
namespace helpers {

template <typename T>
static SD_KERNEL void reverseTadKernel(const void* vinput, const LongType* inputShape, void* voutput,
                                       const LongType* outputShape, const LongType* inputTadShape,
                                       const LongType* inputTadOffsets, const LongType* outputTadShape,
                                       const LongType* outputTadOffsets, uint64_t limit,
                                       uint64_t numOfElemsToReverse, uint64_t numTads) {
  auto input = reinterpret_cast<const T*>(vinput);
  auto output = reinterpret_cast<T*>(voutput);
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  const auto step = gridDim.x * blockDim.x;

  // this means that we'll have additional cycle, to move middle element
  auto div = numOfElemsToReverse / 2;
  auto odd = numOfElemsToReverse % 2 != 0;
  auto rlimit = odd ? limit / 2 + 1 : limit / 2;

  // all threads operate in the same input/output space
  for (uint64_t e = tid; e < rlimit; e += step) {
    // finding out the TAD we're going to process
    auto tadId = e / div;

    if (tadId >= numTads) continue;

    // now finding out element within tad
    auto idx = e % div;

    auto tadInput = input + inputTadOffsets[tadId];
    auto tadOutput = output + outputTadOffsets[tadId];

    LongType fCoords[SD_MAX_RANK];
    LongType lCoords[SD_MAX_RANK];
    LongType fOffset;
    LongType lOffset;

    INDEX2COORDS(idx, shape::rank(inputTadShape), shape::shapeOf(inputTadShape), fCoords);
    COORDS2INDEX(shape::rank(inputTadShape), shape::stride(inputTadShape), fCoords, fOffset);
    INDEX2COORDS(numOfElemsToReverse - idx - 1, shape::rank(inputTadShape), shape::shapeOf(inputTadShape), lCoords);
    COORDS2INDEX(shape::rank(inputTadShape), shape::stride(inputTadShape),lCoords, lOffset);

    // now we're storing input values
    auto v1 = tadInput[fOffset];
    auto v2 = tadInput[lOffset];

    LongType zfCoords[SD_MAX_RANK];
    LongType zlCoords[SD_MAX_RANK];
    LongType zfOffset;
    LongType zlOffset;

    INDEX2COORDS(idx, shape::rank(outputTadShape), shape::shapeOf(outputTadShape), zfCoords);
    COORDS2INDEX(shape::rank(outputTadShape), shape::stride(outputTadShape), zfCoords, zfOffset);
    INDEX2COORDS(numOfElemsToReverse - idx - 1, shape::rank(outputTadShape), shape::shapeOf(outputTadShape), zlCoords);
    COORDS2INDEX(shape::rank(outputTadShape), shape::stride(outputTadShape), zlCoords, zlOffset);

    // and saving values to output arrays
    tadOutput[zfOffset] = v2;
    tadOutput[zlOffset] = v1;
  }

  // moving odd element in blocks
  if (odd && threadIdx.x == 0) {
    for (uint64_t e = blockIdx.x; e < numTads; e += gridDim.x) {
      auto tadInput = input + inputTadOffsets[e];
      auto tadOutput = output + outputTadOffsets[e];

      LongType xCoords[SD_MAX_RANK];
      LongType zCoords[SD_MAX_RANK];
      LongType xOffset;
      LongType zOffset;

      INDEX2COORDS(numOfElemsToReverse / 2, shape::rank(inputTadShape), shape::shapeOf(inputTadShape), xCoords);
      COORDS2INDEX(shape::rank(inputTadShape), shape::stride(inputTadShape), xCoords, xOffset);
      INDEX2COORDS(numOfElemsToReverse / 2, shape::rank(outputTadShape), shape::shapeOf(outputTadShape), zCoords);
      COORDS2INDEX(shape::rank(outputTadShape), shape::stride(outputTadShape), zCoords, zOffset);

      tadOutput[zOffset] = tadInput[xOffset];
    }
  }
}

template <typename T>
static SD_KERNEL void reverseArrayKernel(const void* input, const LongType* inputShape, void* output,
                                         const LongType* outputShape, LongType numOfElemsToReverse) {
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  const auto step = gridDim.x * blockDim.x;
  __shared__ const T* inputArr;
  __shared__ T* outputArr;
  __shared__ char inputOrder, outputOrder;

  if (threadIdx.x == 0) {
    char inputOrder = shape::order(inputShape);
    char outputOrder = shape::order(outputShape);
    inputArr = reinterpret_cast<const T*>(input);
    outputArr = reinterpret_cast<T*>(output);
  }
  __syncthreads();

  auto odd = numOfElemsToReverse % 2 != 0;
  auto limit = numOfElemsToReverse / 2;

  for (uint64_t e = tid; e < limit; e += step) {
    LongType fCoords[SD_MAX_RANK];
    LongType lCoords[SD_MAX_RANK];
    LongType fOffset;
    LongType lOffset;

    INDEX2COORDS(e, shape::rank(inputShape), shape::shapeOf(inputShape), fCoords);
    COORDS2INDEX(shape::rank(inputShape), shape::stride(inputShape), fCoords, fOffset);
    INDEX2COORDS(numOfElemsToReverse - e - 1, shape::rank(inputShape), shape::shapeOf(inputShape), lCoords);
    COORDS2INDEX(shape::rank(inputShape), shape::stride(inputShape), lCoords, lOffset);

    auto v1 = inputArr[fOffset];
    auto v2 = inputArr[lOffset];

    LongType zfCoords[SD_MAX_RANK];
    LongType zlCoords[SD_MAX_RANK];
    LongType zfOffset;
    LongType zlOffset;

    INDEX2COORDS(e, shape::rank(outputShape), shape::shapeOf(shape::shapeOf(zShapeInfo)outputShape), zfCoords);
    COORDS2INDEX(shape::rank(outputShape), shape::stride(outputShape), zfCoords, zfOffset);
    INDEX2COORDS(numOfElemsToReverse - e - 1, shape::rank(outputShape), shape::shapeOf(outputShape), zlCoords);
    COORDS2INDEX(shape::rank(outputShape), shape::stride(outputShape), zlCoords, zlOffset);

    outputArr[zfOffset] = v2;
    outputArr[zlOffset] = v1;
  }

  if (odd && tid == 0) {
    LongType xCoords[SD_MAX_RANK];
    LongType zCoords[SD_MAX_RANK];
    LongType xOffset;
    LongType zOffset;

    INDEX2COORDS(limit, shape::rank(inputShape), shape::shapeOf(inputShape), xCoords);
    COORDS2INDEX(shape::rank(inputShape), shape::stride(inputShape), xCoords, xOffset);
    INDEX2COORDS(limit, shape::rank(outputShape), shape::shapeOf(outputShape), zCoords);
    COORDS2INDEX(shape::rank(outputShape), shape::stride(outputShape), zCoords, zOffset);

    outputArr[zOffset] = inputArr[xOffset];
  }
}
template <typename T>
static void reverseTad(LaunchContext* context, NDArray* input, NDArray* output,
                       const LongType* inputTadShape, const LongType* inputTadOffsets,
                       const LongType* outputTadShape, const LongType* outputTadOffsets, uint64_t tadLength) {
  auto stream = context->getCudaStream();
  dim3 launchDims = getLaunchDims("reverse");

  reverseTadKernel<T><<<launchDims.y, launchDims.x, launchDims.z, *stream>>>(input->specialBuffer(), input->specialShapeInfo(),
                                                   output->specialBuffer(), output->specialShapeInfo(), inputTadShape,
                                                   inputTadOffsets, outputTadShape, outputTadOffsets, input->lengthOf(),
                                                   tadLength, input->lengthOf() / tadLength);
  sd::DebugHelper::checkErrorCode(stream, "reverseTadKernel failed");

}

template <typename T>
static void reverseArray(LaunchContext* context, NDArray* input, NDArray* output, LongType numOfElemsToReverse) {
  auto stream = context->getCudaStream();
  LongType numOfReverse = numOfElemsToReverse;
  if (numOfElemsToReverse == 0) numOfReverse = input->lengthOf();
  dim3 launchDims = getLaunchDims("reverse");

  reverseArrayKernel<T><<<launchDims.y,launchDims.x, launchDims.z, *stream>>>(input->specialBuffer(), input->specialShapeInfo(),
                                                     output->specialBuffer(), output->specialShapeInfo(), numOfReverse);
  sd::DebugHelper::checkErrorCode(stream, "reverseArrayKernel failed");

}

///////////////////////////////////////////////////////////////////
template <typename T>
static void reverseSequence_(LaunchContext* context, NDArray* input, NDArray* seqLengths,
                             NDArray* output, int seqDim, const int batchDim) {
  int posOfNonUnityDim = -1;
  seqLengths->syncToHost();
  auto stream = context->getCudaStream();
  dim3 launchDims = getLaunchDims("reverse");
  if (input->isVector() || shape::isLikeVector(input->shapeInfo(), posOfNonUnityDim) || seqLengths->lengthOf() == 1) {
    LongType numOfElemsToReverse = seqLengths->e<LongType>(0);
    if ((seqDim == 0 && input->sizeAt(0) == 1) || (batchDim == posOfNonUnityDim))
      output->assign(*input);
    else
      reverseArrayKernel<T><<<launchDims.y, launchDims.x, launchDims.z, *stream>>>(
          input->specialBuffer(), input->specialShapeInfo(), output->specialBuffer(), output->specialShapeInfo(),
          numOfElemsToReverse);
    sd::DebugHelper::checkErrorCode(stream, "reverseArrayKernel failed");

  } else {
    if (seqDim > batchDim) --seqDim;

    std::vector<LongType> dim = {batchDim};
    std::vector<LongType> *dimensions = ShapeUtils::evalDimsToExclude(input->rankOf(), 1,dim.data());

    auto inSubArrsSet = input->allTensorsAlongDimension(*dimensions);
    auto outSubArrsSet = output->allTensorsAlongDimension(*dimensions);

    for (int i = 0; i < inSubArrsSet.size(); ++i) {
      LongType numOfElemsToReverse = seqLengths->e<LongType>(i);

      if (numOfElemsToReverse == 0 || numOfElemsToReverse == 1) {
        outSubArrsSet.at(i)->assign(*inSubArrsSet.at(i));
      } else {
        auto inInnerSet = inSubArrsSet.at(i)->allTensorsAlongDimension({seqDim});
        auto outInnerSet = outSubArrsSet.at(i)->allTensorsAlongDimension({seqDim});
        for (int j = 0; j < inInnerSet.size(); ++j)
          reverseArray<T>(context, inInnerSet.at(j), outInnerSet.at(j), numOfElemsToReverse);
      }
    }

    delete dimensions;
  }
}

void reverseSequence(LaunchContext* context, NDArray* input, NDArray* seqLengths, NDArray* output,
                     int seqDim, const int batchDim) {
  NDArray::prepareSpecialUse({output}, {input, seqLengths});

  // if op isn't inplace - copy original data into output array
  if (output->specialBuffer() != input->specialBuffer()) output->assign(*input);

  BUILD_SINGLE_SELECTOR(input->dataType(), reverseSequence_, (context, input, seqLengths, output, seqDim, batchDim),
                        SD_COMMON_TYPES);
  NDArray::registerSpecialUse({output}, {input, seqLengths});
}

//////////////////////////////////////////////////////////////////////////
void reverse(LaunchContext* context, NDArray* input, NDArray* output, const std::vector<LongType>* intArgs) {
  auto packX = ConstantTadHelper::getInstance().tadForDimensions(input->shapeInfo(), intArgs);
  auto packZ = ConstantTadHelper::getInstance().tadForDimensions(output->shapeInfo(), intArgs);

  NDArray::prepareSpecialUse({output}, {input});

  if (packX->numberOfTads() == 1) {
    BUILD_SINGLE_SELECTOR(input->dataType(), reverseArray, (context, input, output, 0), SD_COMMON_TYPES);
  } else {
    BUILD_SINGLE_SELECTOR(
        input->dataType(), reverseTad,
        (context, input, output, packX->platformShapeInfo(), packX->platformOffsets(), packZ->platformShapeInfo(),
            packZ->platformOffsets(), (uint64_t)(input->lengthOf() / packX->numberOfTads())),
        SD_COMMON_TYPES);
  }

  NDArray::registerSpecialUse({output}, {input});
}
}  // namespace helpers
}  // namespace ops
}  // namespace sd

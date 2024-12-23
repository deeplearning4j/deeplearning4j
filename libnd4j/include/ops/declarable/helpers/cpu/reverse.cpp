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
#include <execution/Threads.h>
#include <helpers/ShapeUtils.h>
#include <ops/declarable/helpers/reverse.h>
#if NOT_EXCLUDED(OP_reverse)
namespace sd {
namespace ops {
namespace helpers {

template <typename T>
inline void swap(T* arr, sd::LongType from, sd::LongType to) {
  T tmp = arr[from];
  arr[from] = arr[to];
  arr[to] = tmp;
}

/////////////////////////////////////////////////////////////////////////////////////
// this legacy op is written by raver119@gmail.com

template <typename T>
static void reverseArray(sd::LaunchContext* context, void const* vinArr, sd::LongType const* inShapeBuffer,
                         void* voutArr, sd::LongType const* outShapeBuffer, int numOfElemsToReverse = 0) {
  auto inArr = reinterpret_cast<T const*>(vinArr);
  auto outArr = reinterpret_cast<T*>(voutArr);

  // Cache shape information
  const auto inRank = shape::rank(inShapeBuffer);
  const auto outRank = shape::rank(outShapeBuffer);
  const auto* inShape = shape::shapeOf(inShapeBuffer);
  const auto* outShape = shape::shapeOf(outShapeBuffer);
  const auto* inStride = shape::stride(inShapeBuffer);
  const auto* outStride = shape::stride(outShapeBuffer);

  sd::LongType inLength = shape::length(inShapeBuffer);
  sd::LongType outLength = shape::length(outShapeBuffer);
  if (numOfElemsToReverse == 0) numOfElemsToReverse = inLength;
  sd::LongType sLength = numOfElemsToReverse - 1;

  LongType inCoords[SD_MAX_RANK];
  LongType outCoords[SD_MAX_RANK];
  LongType inOffset;
  LongType outOffset;

  // two step phase here
  if (inArr == outArr) {
    auto func = PRAGMA_THREADS_FOR {
      for (sd::LongType e = start; e < stop; e++) {
        INDEX2COORDS(e, inRank, inShape, inCoords);
        COORDS2INDEX(inRank, inStride, inCoords, inOffset);
        INDEX2COORDS(sLength - e, inRank, inShape, outCoords);
        COORDS2INDEX(inRank, inStride, outCoords, outOffset);
        swap(const_cast<T*>(inArr), inOffset, outOffset);
      }
    };
    samediff::Threads::parallel_for(func, 0, numOfElemsToReverse / 2);
  } else {
    // single step phase here
    auto func = PRAGMA_THREADS_FOR {
      for (sd::LongType e = start; e < stop; e++) {
        INDEX2COORDS(e, inRank, inShape, inCoords);
        COORDS2INDEX(inRank, inStride, inCoords, inOffset);
        INDEX2COORDS(sLength - e, outRank, outShape, outCoords);
        COORDS2INDEX(outRank, outStride, outCoords, outOffset);
        outArr[outOffset] = inArr[inOffset];
      }
    };
    samediff::Threads::parallel_for(func, 0, numOfElemsToReverse);

    if (inLength != numOfElemsToReverse) {
      auto f2 = PRAGMA_THREADS_FOR {
        for (sd::LongType e = start; e < stop; e++) {
          INDEX2COORDS(e, inRank, inShape, inCoords);
          COORDS2INDEX(inRank, inStride, inCoords, inOffset);
          INDEX2COORDS(e, outRank, outShape, outCoords);
          COORDS2INDEX(outRank, outStride, outCoords, outOffset);
          outArr[outOffset] = inArr[inOffset];
        }
      };
      samediff::Threads::parallel_for(f2, numOfElemsToReverse, inLength);
    }
  }
}

///////////////////////////////////////////////////////////////////
template <typename T>
static void reverseSequence_(sd::LaunchContext* context, NDArray* input, NDArray* seqLengths,
                             NDArray* output, int seqDim, const int batchDim) {
  int posOfNonUnityDim = -1;
  if (input->isVector() || shape::isLikeVector(input->shapeInfo(), posOfNonUnityDim)) {
    if ((seqDim == 0 && input->sizeAt(0) == 1) || (batchDim == posOfNonUnityDim))
      output->assign(*input);
    else
      helpers::reverseArray<T>(context, const_cast<NDArray*>(input)->buffer(), const_cast<NDArray*>(input)->shapeInfo(),
                               output->buffer(), output->shapeInfo(), seqLengths->e<int>(0));
  } else {
    if (seqDim > batchDim) --seqDim;

    std::vector<sd::LongType> batchDimVec = {batchDim};
    std::vector<sd::LongType> *dimensions = ShapeUtils::evalDimsToExclude(input->rankOf(), 1,batchDimVec.data());

    auto inSubArrsSet = input->allTensorsAlongDimension(*dimensions);
    auto outSubArrsSet = output->allTensorsAlongDimension(*dimensions);
    delete dimensions;
    
    for (int i = 0; i < inSubArrsSet.size(); ++i) {
      sd::LongType numOfElemsToReverse = seqLengths->e<sd::LongType>(i);

      if (numOfElemsToReverse == 0 || numOfElemsToReverse == 1) {
        outSubArrsSet.at(i)->assign(*inSubArrsSet.at(i));
      } else {
        auto inInnerSet = inSubArrsSet.at(i)->allTensorsAlongDimension({seqDim});
        auto outInnerSet = outSubArrsSet.at(i)->allTensorsAlongDimension({seqDim});
        for (int j = 0; j < inInnerSet.size(); ++j)
          helpers::reverseArray<T>(context, inInnerSet.at(j)->buffer(), inInnerSet.at(j)->shapeInfo(),
                                   outInnerSet.at(j)->buffer(), outInnerSet.at(j)->shapeInfo(), numOfElemsToReverse);
      }
    }
  }
}

void reverseSequence(sd::LaunchContext* context, NDArray* input, NDArray* seqLengths, NDArray* output,
                     int seqDim, const int batchDim) {
  BUILD_SINGLE_SELECTOR(input->dataType(), reverseSequence_, (context, input, seqLengths, output, seqDim, batchDim),
                        SD_COMMON_TYPES);
}

//////////////////////////////////////////////////////////////////////////
void reverse(sd::LaunchContext* context, NDArray* input, NDArray* output, const std::vector<LongType>* intArgs) {
  auto listOut = output->allTensorsAlongDimension(*intArgs);
  auto listIn = input->allTensorsAlongDimension(*intArgs);

  NDArray *subArrIn, *subArrOut;

  for (int i = 0; i < listIn.size(); ++i) {  // listIn.size() = listOut.size()
    subArrIn = listIn.at(i);
    subArrOut = listOut.at(i);
    BUILD_SINGLE_SELECTOR(
        input->dataType(), helpers::reverseArray,
        (context, subArrIn->buffer(), subArrIn->shapeInfo(), subArrOut->buffer(), subArrOut->shapeInfo()),
        SD_COMMON_TYPES);
  }
}

BUILD_SINGLE_TEMPLATE(template void reverseSequence_,
                      (sd::LaunchContext * context, NDArray* input, NDArray* seqLengths, NDArray* output,
                          int seqDim, const int batchDim),
                      SD_COMMON_TYPES);
BUILD_SINGLE_TEMPLATE(template void reverseArray,
                      (sd::LaunchContext * context, void const* inArr, sd::LongType const* inShapeBuffer, void* outArr,
                          sd::LongType const* outShapeBuffer, int numOfElemsToReverse),
                      SD_COMMON_TYPES);

}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif
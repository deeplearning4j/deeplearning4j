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
//  @author raver119@gmail.com
//
#include <array/NDArrayFactory.h>
#include <execution/Threads.h>
#include <ops/declarable/headers/parity_ops.h>
#include <ops/declarable/helpers/top_k.h>

#include "ops/specials.h"
#if NOT_EXCLUDED(OP_top_k)
namespace sd {
namespace ops {
namespace helpers {

template <typename T>
static sd::Status topKFunctor_(NDArray* input, NDArray* values, NDArray* indices, const sd::LongType k,
                               bool needSort) {
  sd::LongType width = input->sizeAt(-1);
  sd::LongType lastDim = input->rankOf() - 1;
  std::vector<sd::LongType> dimsToExclude(input->rankOf() - 1);
  for (size_t d = 0; d < dimsToExclude.size(); ++d) dimsToExclude[d] = d;

  const sd::LongType numOfSubArrs = ShapeUtils::getNumOfSubArrs(input->shapeInfo(), dimsToExclude);

  if (k == 1) {
    for (sd::LongType e = 0; e < numOfSubArrs; ++e) {
      auto trial = (*input)(e, dimsToExclude);
      // Sync host buffer once, then access directly via pointer to avoid the
      // double-offset bug in trial->e<T>(i): buffer() already applies _offset,
      // and getOffset(i) adds _offset a second time for TAD views with offset>0.
      trial->syncToHost();
      const T* trialBuf = trial->bufferAsT<T>();
      const sd::LongType trialStride = trial->stridesOf()[0];
      sd::LongType maxPos = 0;
      T maxVal = trialBuf[0];
      for (sd::LongType pos = 1; pos < trial->lengthOf(); pos++)
        if (maxVal < trialBuf[pos * trialStride]) {
          maxPos = pos;
          maxVal = trialBuf[pos * trialStride];
        }
      if (indices) indices->p(e, maxPos);  // topIndex;
      if (values) values->p(e, maxVal);
      delete trial;
    }
  } else {
    int nextPos = 0;

    for (sd::LongType e = 0; e < numOfSubArrs; ++e) {
      auto trial = (*input)(e, dimsToExclude);
      // Sync host buffer once, then use direct pointer access.
      // trial->t<T>(i) / trial->e<T>(i) both call bufferWithOffset(getOffset(i))
      // which expands to buffer()[_offset + stride*i], but buffer() already
      // starts at _offset — so they double-count _offset for any TAD whose
      // base is not at element 0.  Using bufferAsT<T>() (== buffer() already
      // positioned at the TAD start) and indexing by [i * stride] is correct.
      trial->syncToHost();
      const T* trialBuf = trial->bufferAsT<T>();
      const sd::LongType trialStride = trial->stridesOf()[0];

      // fill up the first k elements
      NDArray *topValues = NDArrayFactory::create<T>('c', {k}, input->getContext());
      NDArray *sortedVals = NDArrayFactory::create<T>('c', {k}, input->getContext());
      NDArray *topIndices = NDArrayFactory::create<sd::LongType>('c', {k}, input->getContext());
      for (sd::LongType pos = 0; pos < k; ++pos) {
        topIndices->r<sd::LongType>(pos) = pos;
        topValues->r<T>(pos) = trialBuf[pos * trialStride];
      }
      sortedVals->assign(topValues);
      SpecialMethods<T>::sortGeneric(sortedVals, false);
      for (sd::LongType i = static_cast<sd::LongType>(k); i < width; ++i) {
        T val = trialBuf[i * trialStride];
        T minTopVal = sortedVals->t<T>(0);
        if (minTopVal < val) {  // value should be inserted to top k
          // only if it is not contained in
          T* begin = reinterpret_cast<T*>(sortedVals->buffer());
          T* end = begin + k;
          bool exists = std::binary_search(begin, end, val);
          if (!exists) {
            // exchangePos - a distance between begin and minimal existed to be suppressed by val
            T* topBegin = reinterpret_cast<T*>(topValues->buffer());
            T* topEnd = topBegin + k;
            auto exchangePos = std::distance(topBegin, std::find(topBegin, topEnd, sortedVals->t<T>(0)));
            topValues->r<T>(exchangePos) = val;  //*exchangeIt = val;
            topIndices->r<sd::LongType>(exchangePos) = i;
            sortedVals->r<T>(0) = val;  // suppress in sorted
            SpecialMethods<T>::sortGeneric(sortedVals, false);

          }
        }

      }
      if (needSort) {
        SpecialMethods<T>::sortGeneric(topValues,true);

        for (sd::LongType j = 0; j < width; j++)
          for (sd::LongType pos = 0; pos < k; ++pos)
            if (topValues->t<T>(pos) == trialBuf[j * trialStride]) topIndices->r<sd::LongType>(pos) = j;
      } else {  // else sort by indices
        std::map<sd::LongType, T> sortValsMap;
        for (sd::LongType e = 0; e < topValues->lengthOf(); ++e) {
          sortValsMap[topIndices->t<sd::LongType>(e)] = topValues->t<T>(e);
        }

        sd::LongType e = 0;
        for (auto it = sortValsMap.begin(); it != sortValsMap.end(); ++it, e++) {
          topIndices->r<sd::LongType>(e) = it->first;
          topValues->r<T>(e) = it->second;
        }
      }
      if (values) {
        auto valuesView = (*values)(e, dimsToExclude);
        valuesView->assign(topValues);
        delete valuesView;
      }
      if (indices) {
        auto indicesView = (*indices)(e, dimsToExclude);
        indicesView->assign(topIndices);
        delete indicesView;
      }


      delete sortedVals;
      delete topValues;
      delete topIndices;

      delete trial;
    }
  }

  return sd::Status::OK;
}
// ----------------------------------------------------------------------------------------------- //

template <typename T>
static sd::Status inTopKFunctor_(sd::LaunchContext* context, NDArray* input, NDArray* target,
                                 NDArray* result, const sd::LongType k) {
  result->nullify();

  sd::LongType numRows = input->sizeAt(0);
  sd::LongType width = input->sizeAt(1);

  auto func = PRAGMA_THREADS_FOR {
    for (auto e = start; e < stop; e++) {
      sd::LongType targetIdx = target->e<sd::LongType>(e);
      T targetVal = input->e<T>(e, targetIdx);

      // Count values strictly greater than targetVal in this row
      sd::LongType countGreater = 0;
      for (sd::LongType j = 0; j < width; j++) {
        if (input->e<T>(e, j) > targetVal) {
          countGreater++;
        }
      }

      // Target is in top-k if fewer than k values are strictly greater
      if (countGreater < k) {
        result->p<bool>(e, true);
      }
    }
  };

  samediff::Threads::parallel_tad(func, 0, numRows);
  return sd::Status::OK;
}

sd::Status topKFunctor(sd::LaunchContext* context, NDArray* input, NDArray* values, NDArray* indices,
                       const sd::LongType k, bool needSort) {
  BUILD_SINGLE_SELECTOR(input->dataType(), return topKFunctor_, (input, values, indices, k, needSort),
                        SD_NUMERIC_TYPES);


}

sd::Status inTopKFunctor(sd::LaunchContext* context, NDArray* input, NDArray* target, NDArray* result,
                         const sd::LongType k) {
  BUILD_SINGLE_SELECTOR(input->dataType(), return inTopKFunctor_, (context, input, target, result, k),
                        SD_NUMERIC_TYPES);
}

BUILD_SINGLE_TEMPLATE( sd::Status topKFunctor_,
                       (NDArray* input, NDArray* values, NDArray* indices, const sd::LongType k, bool needSort),
                       SD_NUMERIC_TYPES);
BUILD_SINGLE_TEMPLATE( sd::Status inTopKFunctor_,
                       (sd::LaunchContext * context, NDArray* input, NDArray* target, NDArray* result,
                           const sd::LongType k),
                       SD_NUMERIC_TYPES);
}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif
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
#if NOT_EXCLUDED(OP_top_k)
namespace sd {
namespace ops {
namespace helpers {

template <typename T>
static sd::Status topKFunctor_(const NDArray* input, NDArray* values, NDArray* indices, const sd::LongType k,
                               bool needSort) {
  sd::LongType width = input->sizeAt(-1);
  sd::LongType lastDim = input->rankOf() - 1;
  std::vector<sd::LongType> dimsToExclude(input->rankOf() - 1);
  for (size_t d = 0; d < dimsToExclude.size(); ++d) dimsToExclude[d] = d;

  const sd::LongType numOfSubArrs = ShapeUtils::getNumOfSubArrs(input->shapeInfo(), dimsToExclude);

  if (k == 1) {
    for (sd::LongType e = 0; e < numOfSubArrs; ++e) {
      auto trial = (*input)(e, dimsToExclude);
      sd::LongType maxPos = 0;
      T maxVal = trial.e<T>(0);
      for (sd::LongType pos = 1; pos < trial.lengthOf(); pos++)
        if (maxVal < trial.e<T>(pos)) {
          maxPos = pos;
          maxVal = trial.e<T>(pos);
        }
      if (indices) indices->p(e, maxPos);  // topIndex;
      if (values) values->p(e, maxVal);
    }
  } else {
    int nextPos = 0;

    for (sd::LongType e = 0; e < numOfSubArrs; ++e) {
      auto trial = (*input)(e, dimsToExclude);

      // fill up the first k elements
      NDArray topValues = NDArrayFactory::create<T>('c', {k}, input->getContext());
      NDArray sortedVals = NDArrayFactory::create<T>('c', {k}, input->getContext());
      NDArray topIndices = NDArrayFactory::create<sd::LongType>('c', {k}, input->getContext());
      for (sd::LongType pos = 0; pos < k; ++pos) {
        topIndices.r<sd::LongType>(pos) = pos;
        topValues.r<T>(pos) = trial.t<T>(pos);
      }
      // std::vector<T> sortedVals(topValues);
      sortedVals.assign(topValues);  // = NDArrayFactory::create<T>('c', {k});
      // std::sort(sortedVals.begin(), sortedVals.end()); // sorted in ascending order
      SpecialMethods<T>::sortGeneric(sortedVals.buffer(), sortedVals.shapeInfo(), false);
      for (sd::LongType i = static_cast<sd::LongType>(k); i < width; ++i) {
        T val = trial.e<T>(i);
        T minTopVal = sortedVals.t<T>(0);
        if (minTopVal < val) {  // value should be inserted to top k
          // only if it is not contained in
          T* begin = reinterpret_cast<T*>(sortedVals.buffer());
          T* end = begin + k;
          bool exists = std::binary_search(begin, end, val);
          if (!exists) {
            // exchangePos - a distance between begin and minimal existed to be suppressed by val
            T* topBegin = reinterpret_cast<T*>(topValues.buffer());
            T* topEnd = topBegin + k;
            auto exchangePos = std::distance(topBegin, std::find(topBegin, topEnd, sortedVals.t<T>(0)));
            topValues.r<T>(exchangePos) = val;  //*exchangeIt = val;
            topIndices.r<sd::LongType>(exchangePos) = i;
            sortedVals.r<T>(0) = val;  // suppress in sorted
            // std::sort(sortedVals.begin(), sortedVals.end()); // sorted in ascending order
            SpecialMethods<T>::sortGeneric(sortedVals.buffer(), sortedVals.shapeInfo(), false);
          }
        }
      }
      if (needSort) {
        SpecialMethods<T>::sortGeneric(topValues.buffer(), topValues.shapeInfo(), true);

        for (sd::LongType j = 0; j < width; j++)
          for (sd::LongType pos = 0; pos < k; ++pos)
            if (topValues.t<T>(pos) == trial.t<T>(j)) topIndices.r<sd::LongType>(pos) = j;
      } else {  // else sort by indices
        std::map<sd::LongType, T> sortValsMap;
        for (sd::LongType e = 0; e < topValues.lengthOf(); ++e) {
          sortValsMap[topIndices.t<sd::LongType>(e)] = topValues.t<T>(e);
        }

        sd::LongType e = 0;
        for (auto it = sortValsMap.begin(); it != sortValsMap.end(); ++it, e++) {
          topIndices.r<sd::LongType>(e) = it->first;
          topValues.r<T>(e) = it->second;
        }
      }
      if (values) (*values)(e, dimsToExclude).assign(topValues);
      if (indices) (*indices)(e, dimsToExclude).assign(topIndices);
    }
  }
  return sd::Status::OK;
}
// ----------------------------------------------------------------------------------------------- //

template <typename T>
static sd::Status inTopKFunctor_(sd::LaunchContext* context, const NDArray* input, const NDArray* target,
                                 NDArray* result, const sd::LongType k) {
  std::vector<sd::LongType> shapeI(input->rankOf());
  for (int i = 0; i < input->rankOf() - 1; i++) shapeI[i] = input->sizeAt(i);
  shapeI[input->rankOf() - 1] = k;
  std::unique_ptr<NDArray> indices(NDArrayFactory::create_<sd::LongType>(input->ordering(), shapeI, context));
  NDArray* values = nullptr;
  sd::Status status = topKFunctor(context, input, values, indices.get(), k, true);
  result->assign(0);
  if (status == sd::Status::OK) {
    auto func = PRAGMA_THREADS_FOR {
      for (auto e = start; e < stop; e++) {
        bool found = false;
        for (sd::LongType j = 0; j < k; j++) {
          if (target->e<sd::LongType>(e) == indices->e<sd::LongType>(e * k + j)) {
            found = true;
            break;
          }
        }
        if (found) result->p<bool>(e, true);
      }
    };

    samediff::Threads::parallel_tad(func, 0, target->lengthOf());
  }
  return status;
}

sd::Status topKFunctor(sd::LaunchContext* context, const NDArray* input, NDArray* values, NDArray* indices,
                       const sd::LongType k, bool needSort) {
  BUILD_SINGLE_SELECTOR(input->dataType(), return topKFunctor_, (input, values, indices, k, needSort),
                        SD_NUMERIC_TYPES);
}

sd::Status inTopKFunctor(sd::LaunchContext* context, const NDArray* input, const NDArray* target, NDArray* result,
                         const sd::LongType k) {
  BUILD_SINGLE_SELECTOR(input->dataType(), return inTopKFunctor_, (context, input, target, result, k),
                        SD_NUMERIC_TYPES);
}

BUILD_SINGLE_TEMPLATE(template sd::Status topKFunctor_,
                      (const NDArray* input, NDArray* values, NDArray* indices, const sd::LongType k, bool needSort),
                      SD_NUMERIC_TYPES);
BUILD_SINGLE_TEMPLATE(template sd::Status inTopKFunctor_,
                      (sd::LaunchContext * context, const NDArray* input, const NDArray* target, NDArray* result,
                       const sd::LongType k),
                      SD_NUMERIC_TYPES);
}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif
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
#include <execution/Threads.h>
#include <graph/Variable.h>
#include <ops/declarable/helpers/unique.h>

namespace sd {
namespace ops {
namespace helpers {

template <typename T>
static sd::LongType uniqueCount_(NDArray* input) {
  sd::LongType count = 0;

  std::vector<T> values;

  for (sd::LongType e = 0; e < input->lengthOf(); e++) {
    T v = input->e<T>(e);
    if (std::find(values.begin(), values.end(), v) == values.end()) {
      values.push_back(v);
      count++;
    }
  }
  return count;
}

sd::LongType uniqueCount(sd::LaunchContext* context, NDArray* input) {
  BUILD_SINGLE_SELECTOR(input->dataType(), return uniqueCount_, (input), SD_COMMON_TYPES);
}

BUILD_SINGLE_TEMPLATE(template sd::LongType uniqueCount_, (NDArray * input), SD_COMMON_TYPES);

template <typename T>
static sd::Status uniqueFunctor_(NDArray* input, NDArray* values, NDArray* indices, NDArray* counts) {
  std::vector<T> valuesVector;
  SD_MAP_IMPL<T, int> indicesMap;
  SD_MAP_IMPL<T, int> countsMap;

  for (sd::LongType e = 0; e < input->lengthOf(); e++) {
    T v = input->e<T>(e);
    if (std::find(valuesVector.begin(), valuesVector.end(), v) == valuesVector.end()) {
      valuesVector.push_back(v);
      indicesMap[v] = e;
      countsMap[v] = 1;
    } else {
      countsMap[v]++;
    }
  }

  auto func = PRAGMA_THREADS_FOR {
    for (auto e = start; e < stop; e++) {
      values->p(e, static_cast<T>(valuesVector[e]));
      if (counts != nullptr) counts->p(e, countsMap[valuesVector[e]]);
    }
  };
  samediff::Threads::parallel_for(func, 0, values->lengthOf());

  for (sd::LongType e = 0; e < indices->lengthOf(); e++) {
    auto posI = std::find(valuesVector.begin(), valuesVector.end(), input->e<T>(e));
    auto dist = std::distance(valuesVector.begin(), posI);
    indices->p(e, sd::LongType(dist));  // indicesMap[(*input)(e)];
  }

  return sd::Status::OK;
}

sd::Status uniqueFunctor(sd::LaunchContext* context, NDArray* input, NDArray* values, NDArray* indices,
                         NDArray* counts) {
  input->syncToHost();
  values->syncToHost();
  indices->syncToHost();

  if (counts != nullptr) counts->syncToHost();

  BUILD_SINGLE_SELECTOR(input->dataType(), return uniqueFunctor_, (input, values, indices, counts), SD_COMMON_TYPES);

  input->syncToDevice();
  values->syncToDevice();
  indices->syncToDevice();

  if (counts != nullptr) counts->syncToDevice();
}

BUILD_SINGLE_TEMPLATE(template sd::Status uniqueFunctor_,
                      (NDArray * input, NDArray* values, NDArray* indices, NDArray* counts), SD_COMMON_TYPES);
}  // namespace helpers
}  // namespace ops
}  // namespace sd

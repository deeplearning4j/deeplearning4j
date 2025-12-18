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
#include <execution/Threads.h>
#include <ops/declarable/helpers/compare_elem.h>

namespace sd {
namespace ops {
namespace helpers {
template <typename T>
static void _compare_elem(NDArray* input, bool isStrictlyIncreasing, bool& output) {
  auto length = shape::length(input->shapeInfo());

  int elementsPerThread = length / ELEMENT_THRESHOLD;
  int num_threads = sd::math::sd_max<int>(1, elementsPerThread);
  num_threads = sd::math::sd_min<int>(num_threads, omp_get_max_threads());
  sd::LongType sumt = 0;

  if (isStrictlyIncreasing) {
    auto func = PRAGMA_REDUCE_LONG {
      sd::LongType sum = 0;
      for (auto i = start; i < stop; i++) {
        auto val0 = input->t<T>(i);
        auto val1 = input->t<T>(i + 1);
        sum += val0 >= val1 ? -1 : 0;
      }
      return sum;
    };
    sumt = samediff::Threads::parallel_long(func, LAMBDA_SUML, 0, length - 1);
  } else {
    auto func = PRAGMA_REDUCE_LONG {
      sd::LongType sum = 0;
      for (auto i = start; i < stop; i++) {
        auto val0 = input->t<T>(i);
        auto val1 = input->t<T>(i + 1);
        sum += val0 > val1 ? -1 : 0;
      }

      return sum;
    };
    sumt = samediff::Threads::parallel_long(func, LAMBDA_SUML, 0, length - 1);
  }

  output = (sumt > -1);
}

void compare_elem(sd::LaunchContext* context, NDArray* input, bool isStrictlyIncreasing, bool& output) {
  auto xType = input->dataType();

  BUILD_SINGLE_SELECTOR(xType, _compare_elem, (input, isStrictlyIncreasing, output), SD_COMMON_TYPES);
}

BUILD_SINGLE_TEMPLATE( void _compare_elem, (NDArray * A, bool isStrictlyIncreasing, bool& output);
                      , SD_COMMON_TYPES);
}  // namespace helpers
}  // namespace ops
}  // namespace sd

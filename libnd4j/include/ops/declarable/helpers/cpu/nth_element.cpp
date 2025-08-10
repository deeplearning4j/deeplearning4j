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
#include <helpers/ConstantTadHelper.h>
#include <helpers/ShapeUtils.h>

#include <ops/declarable/helpers/nth_element.h>
#include <system/selective_rendering.h>
#include "ops/specials.h"
#if NOT_EXCLUDED(OP_nth_element)
namespace sd {
namespace ops {
namespace helpers {

template <typename T>
void nthElementFunctor_(NDArray* input, sd::LongType n, NDArray* output, bool reverse) {
  NDArray sortedVals(*input);
  if (input->isVector()) {
    SpecialMethods<T>::sortGeneric(input, reverse);
    output->p(0, input->e<T>(n));
  } else {  // rank greater than 1
    std::vector<sd::LongType> lastDims(
        {input->rankOf() - 1});
    SpecialMethods<T>::sortTadGeneric(&sortedVals, lastDims.data(), lastDims.size(),
                                      reverse);

    ResultSet rows = sortedVals.allTensorsAlongDimension(lastDims);
    sd::LongType oL = output->lengthOf();

    auto func = PRAGMA_THREADS_FOR {
      for (auto e = start; e < stop; e++) {
        auto row = rows.at(e);
        output->p(e, row->e<T>(n));
      }
    };

    samediff::Threads::parallel_for(func, 0, oL);
  }
}

void nthElementFunctor(sd::LaunchContext* launchContext, NDArray* input, sd::LongType n, NDArray* output,
                       bool reverse) {

  auto inputDType = input->dataType();
  BUILD_SINGLE_SELECTOR(input->dataType(), nthElementFunctor_, (input, n, output, reverse), SD_NUMERIC_TYPES);
}
BUILD_SINGLE_TEMPLATE(template void nthElementFunctor_,
                      (NDArray * input, sd::LongType n, NDArray* output, bool reverse), SD_NUMERIC_TYPES);

}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif
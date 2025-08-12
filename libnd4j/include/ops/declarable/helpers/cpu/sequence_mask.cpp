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
#include <execution/Threads.h>
#include <ops/declarable/helpers/sequence_mask.h>
#if NOT_EXCLUDED(OP_sequence_mask)
namespace sd {
namespace ops {
namespace helpers {

template <typename I, typename B>
static void sequenceMask_(NDArray* input, NDArray* output, int maxIndex) {
  auto func = PRAGMA_THREADS_FOR_2D {
    for (sd::LongType i = start_x; i < stop_x; i += inc_x)
      for (auto k = start_y; k < stop_y; k += inc_y)
        if (i < static_cast<sd::LongType>(input->t<I>(k))) output->r<B>(k * maxIndex + i) = B(true);  //,  T(1.0f));
  };

  samediff::Threads::parallel_for(func, 0, maxIndex, 1, 0, input->lengthOf(), 1);
}

void sequenceMask(sd::LaunchContext* context, NDArray* input, NDArray* output, int maxIndex) {
  BUILD_DOUBLE_SELECTOR(input->dataType(), output->dataType(), sequenceMask_, (input, output, maxIndex),
                        SD_INTEGER_TYPES, SD_COMMON_TYPES_EXTENDED);
}

BUILD_DOUBLE_TEMPLATE( void sequenceMask_, (NDArray * input, NDArray* output, int maxIndex), SD_INTEGER_TYPES,
                      SD_COMMON_TYPES_EXTENDED);
}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif
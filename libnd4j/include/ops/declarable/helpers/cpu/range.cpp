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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 27.08.2018
//

#include <execution/Threads.h>
#include <ops/declarable/helpers/range.h>
#if NOT_EXCLUDED(OP_range)
namespace sd {
namespace ops {
namespace helpers {

//////////////////////////////////////////////////////////////////////////
// be careful: outVector must have c-order and ews = 1 !!!
template <typename T>
static void _range(NDArray& start, NDArray& delta, NDArray& outVector) {
  const sd::LongType len = outVector.lengthOf();

  auto buff = outVector.bufferAsT<T>();
  auto s = start.e<T>(0);
  auto d = delta.e<T>(0);
  auto func = PRAGMA_THREADS_FOR {
    for (auto i = start; i < stop; i++) {
      buff[i] = s + i * d;
    }
  };
  samediff::Threads::parallel_for(func, 0, len);
}

void range(sd::LaunchContext* context, NDArray& start, NDArray& delta, NDArray& outVector) {
  BUILD_SINGLE_SELECTOR(outVector.dataType(), _range, (start, delta, outVector), SD_COMMON_TYPES);
}

BUILD_SINGLE_TEMPLATE( void _range, (NDArray& start, NDArray& delta, NDArray& outVector),
                      SD_COMMON_TYPES);

}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif
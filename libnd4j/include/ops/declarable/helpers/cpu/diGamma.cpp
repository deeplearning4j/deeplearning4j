/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

//
// @author Yurii Shyrma (iuriish@yahoo.com)
//
#include <execution/Threads.h>
#include <ops/declarable/helpers/gammaMathFunc.h>
#if NOT_EXCLUDED(OP_digamma)
namespace sd {
namespace ops {
namespace helpers {

//////////////////////////////////////////////////////////////////////////
// calculate digamma function for array elements
template <typename T>
static void diGamma_(NDArray& x, NDArray& z) {
  auto func = PRAGMA_THREADS_FOR {
    for (auto i = start; i < stop; i++) z.p(i, diGammaScalar<T>(x.e<T>(i)));
  };
  samediff::Threads::parallel_for(func, 0, x.lengthOf());
}

void diGamma(sd::LaunchContext* context, NDArray& x, NDArray& z) {
  BUILD_SINGLE_SELECTOR(x.dataType(), diGamma_, (x, z), SD_FLOAT_TYPES);
}

BUILD_SINGLE_TEMPLATE( void diGamma_, (NDArray& x, NDArray& z), SD_FLOAT_TYPES);

}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif
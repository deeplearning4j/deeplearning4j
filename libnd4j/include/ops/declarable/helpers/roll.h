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
#include <ops/declarable/helpers/helpers.h>
#ifndef __HELPERS__ROLL__H__
#define __HELPERS__ROLL__H__
namespace sd {
namespace ops {
namespace helpers {

SD_LIB_HIDDEN void rollFunctorLinear(sd::LaunchContext* context, NDArray* input, NDArray* output, int shift,
                                     bool inplace = false);

SD_LIB_HIDDEN void rollFunctorFull(sd::LaunchContext* context, NDArray* input, NDArray* output,
                                   std::vector<int> const& shifts, std::vector<int> const& axes, bool inplace = false);

}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif

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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 20.04.2018
//

#include <ops/declarable/helpers/transforms.h>
#include <ops/specials.h>
#include <system/selective_rendering.h>
#if NOT_EXCLUDED(OP_concat)
namespace sd {
namespace ops {
namespace helpers {
//////////////////////////////////////////////////////////////////////////
template <typename T>
static void concat_(const std::vector<NDArray*>& inArrs, NDArray& output, const int axis) {
  sd::SpecialMethods<T>::concatCpuGeneric(inArrs, output, axis);
}

void concat(sd::LaunchContext* context, const std::vector<NDArray*>& inArrs, NDArray& output, const int axis) {
auto outputTYpe = output.dataType();
  BUILD_SINGLE_SELECTOR(output.dataType(), concat_, (inArrs, output, axis), SD_COMMON_TYPES);
}

BUILD_SINGLE_TEMPLATE( void concat_,
                      (const std::vector<NDArray*>& inArrs, NDArray& output, const int axis), SD_COMMON_TYPES);
}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif
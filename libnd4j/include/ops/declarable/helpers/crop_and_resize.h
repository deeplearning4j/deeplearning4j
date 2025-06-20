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

#ifndef SD_CROP_AND_RESIZE_H
#define SD_CROP_AND_RESIZE_H
#include <array/NDArray.h>
#include <system/op_boilerplate.h>

namespace sd {
namespace ops {
namespace helpers {
template <typename T, typename F, typename I>
 void cropAndResizeFunctor_(NDArray * images, NDArray * boxes, NDArray * indices,
                                         NDArray * cropSize, int method, double extrapolationVal, NDArray* crops);

 void cropAndResizeFunctor(LaunchContext* context, NDArray * images, NDArray * boxes,
                                        NDArray * indices, NDArray * cropSize, int method,
                                        double extrapolationVal, NDArray* crops);
}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // SD_CROP_AND_RESIZE_H

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
// Created by GS <sgazeos@gmail.com> on 05.04.18.
//

#ifndef __DYNAMIC_H_HELPERS__
#define __DYNAMIC_H_HELPERS__
#include <array/NDArray.h>
#include <system/op_boilerplate.h>

namespace sd {
namespace ops {
namespace helpers {

/**
 * dynamic_partition: the slices of the input (the input has the shape of the indices followed by the dimensions of a
 * slice) go to the output of the partition their index names, in the order of the slices. Slices of an index outside
 * [0, outputList.size()) are dropped.
 */
SD_LIB_HIDDEN void dynamicPartitionFunctor(LaunchContext* context, NDArray * input, NDArray * indices,
                                           std::vector<NDArray*>& outputList);

/**
 * dynamic_stitch: the slices of the inputs go to the row of the output their index names. When an index appears more
 * than once, the slice of the last input that has it wins.
 */
SD_LIB_HIDDEN Status dynamicStitchFunctor(LaunchContext* context, std::vector<NDArray*> const& inputs,
                                              std::vector<NDArray*> const& indices, NDArray* output);

/**
 * gradient of dynamic_partition: outputList[0] (the shape of the input) takes, for every slice, the slice of the
 * gradient of its partition (gradientInputList[partition]) at the position of the slice in the partition; slices that
 * no partition holds get zeros. The gradient of dynamic_stitch is a gather of the output's gradient along the first
 * dimension (SameDiff builds it with the gather op), so it has no helper.
 */
SD_LIB_HIDDEN void dynamicPartitionFunctorBP(LaunchContext* context, NDArray * input, NDArray * indices,
                                             std::vector<NDArray*> const& gradientInputList,
                                             std::vector<NDArray*>& outputList);
}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif

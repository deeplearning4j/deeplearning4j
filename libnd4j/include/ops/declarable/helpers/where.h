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
// Created by raver119 on 24/09/18.
//

#ifndef DEV_TESTS_WHERE_H
#define DEV_TESTS_WHERE_H
#include <ops/declarable/helpers/helpers.h>

namespace sd {
namespace ops {
namespace helpers {

// Single input where: returns coordinates of true elements
// Output shape is [numTrue, conditionRank]
SD_LIB_HIDDEN void _where(LaunchContext *context, NDArray &condition, NDArray &output,
                          memory::Workspace *workspace);

// Three input where: condition ? x : y (element-wise)
SD_LIB_HIDDEN void _whereElementWise(LaunchContext *context, NDArray &condition, NDArray &x,
                                      NDArray &y, NDArray &output);

// TAD-based where: select entire TADs based on 1D condition
SD_LIB_HIDDEN void _whereTad(LaunchContext *context, NDArray &condition, NDArray &x,
                              NDArray &y, NDArray &output, const std::vector<LongType> &axis);

// Count true elements in condition (for shape function, GPU-accelerated)
SD_LIB_HIDDEN LongType countTrue(LaunchContext *context, NDArray &condition);

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // DEV_TESTS_WHERE_H

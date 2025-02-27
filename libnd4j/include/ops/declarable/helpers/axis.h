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
#ifndef __AXIS_H_HELPERS__
#define __AXIS_H_HELPERS__
#include <array/NDArray.h>
#include <system/op_boilerplate.h>

namespace sd {
namespace ops {
namespace helpers {

/*
 * adjustAxis routines: adjust data with output to non-negative values.
 * */
SD_LIB_HIDDEN void adjustAxis(LongType rank, NDArray* axisVector, std::vector<LongType>& output);
SD_LIB_HIDDEN void adjustAxis(LongType rank, std::vector<LongType>& output);

}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif

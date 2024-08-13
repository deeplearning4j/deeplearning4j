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
// Created by raver119 on 19.01.18.
//

#ifndef LIBND4J_S_T_B_H
#define LIBND4J_S_T_B_H
#include <ops/declarable/helpers/helpers.h>

namespace sd {
namespace ops {
namespace helpers {

SD_LIB_HIDDEN void batchToSpace(sd::LaunchContext* context, NDArray input, NDArray& output,
                                const sd::LongType cropBottom, const sd::LongType cropTop, const sd::LongType cropLeft,
                                const sd::LongType cropRight, const sd::LongType blockSize);

SD_LIB_HIDDEN void spaceToBatch(LaunchContext* context, const NDArray& input, NDArray& output,
                                const LongType padBottom, const LongType padTop, const LongType padLeft,
                                const LongType padRight, const LongType blockSize);

SD_LIB_HIDDEN void spaceToBatchND(LaunchContext* context, const NDArray& input, const NDArray& blockShape,
                                  const NDArray& padding, NDArray& output);

SD_LIB_HIDDEN void batchToSpaceND(sd::LaunchContext* context, NDArray& input, const NDArray& blockShape, const NDArray& crop,
                                  NDArray& output);

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // LIBND4J_S_T_B_H

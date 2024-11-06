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
// Created by raver119 on 30.11.17.
//

#ifndef LIBND4J_HELPERS_H
#define LIBND4J_HELPERS_H
#include <ops/declarable/helpers/helpers.h>

namespace sd {
namespace ops {
namespace helpers {

SD_LIB_HIDDEN void im2col(sd::LaunchContext& context, NDArray& im, NDArray& col, const LongType kH,
                          const LongType kW, const LongType sH, const LongType sW, const LongType pH, const LongType pW,
                          const LongType dH, const LongType dW, NDArray& arrZeroPadVal);
}
}  // namespace ops
}  // namespace sd

#endif  // LIBND4J_HELPERS_H

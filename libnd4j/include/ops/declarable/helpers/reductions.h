
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
// @author AbdelRauf    (rauf@konduit.ai)
//

#ifndef LIBND4J_HELPERS_REDUCTIONS_H
#define LIBND4J_HELPERS_REDUCTIONS_H
#include <array/NDArray.h>
#include <math/templatemath.h>
#include <system/op_boilerplate.h>

namespace sd {
namespace ops {
namespace helpers {

SD_LIB_HIDDEN void argMax(NDArray& input, NDArray& output, const std::vector<LongType>& dimensions);
SD_LIB_HIDDEN void argAbsMax(NDArray& input, NDArray& output, const std::vector<LongType>& dimensions);
SD_LIB_HIDDEN void argMin(NDArray& input, NDArray& output, const std::vector<LongType>& dimensions);
SD_LIB_HIDDEN void argAbsMin(NDArray& input, NDArray& output, const std::vector<LongType>& dimensions);
SD_LIB_HIDDEN void variance(NDArray& input, NDArray& output, const std::vector<LongType>& dimensions,
                            bool biasCorrected);
SD_LIB_HIDDEN void standardDeviation(NDArray& input, NDArray& output, const std::vector<LongType>& dimensions,
                                     bool biasCorrected);

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif

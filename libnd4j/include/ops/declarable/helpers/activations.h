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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 19.04.2018
//

#ifndef LIBND4J_ACTIVATIONS_H
#define LIBND4J_ACTIVATIONS_H
#include <ops/declarable/helpers/helpers.h>

namespace sd {
namespace ops {
namespace helpers {

SD_LIB_HIDDEN void softMaxForVector(sd::LaunchContext *context, const NDArray &input, NDArray &output);

SD_LIB_HIDDEN void logSoftMaxForVector(sd::LaunchContext *context, const NDArray &input, NDArray &output);

SD_LIB_HIDDEN void softmax(sd::LaunchContext *context, const NDArray &input, NDArray &output, const int dimension);

SD_LIB_HIDDEN void logSoftmax(sd::LaunchContext *context, const NDArray &input, NDArray &output, const int dimension);

SD_LIB_HIDDEN void softmaxDerivative(sd::LaunchContext *context, const NDArray &input, NDArray &output,
                                     const int dimension);

SD_LIB_HIDDEN void prelu(sd::LaunchContext *context, const NDArray &input, const NDArray &alpha, NDArray &output);

SD_LIB_HIDDEN void preluBP(sd::LaunchContext *context, const NDArray &input, const NDArray &alpha, const NDArray &dLdO,
                           NDArray &dLdI, NDArray &dLdA);

SD_LIB_HIDDEN void thresholdRelu(sd::LaunchContext *context, const NDArray &input, double threshold, NDArray &output);

SD_LIB_HIDDEN void thresholdReluDerivative(sd::LaunchContext *context, NDArray *input, double threshold, NDArray *dLdO,
                                           NDArray *output);
}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // LIBND4J_ACTIVATIONS_H

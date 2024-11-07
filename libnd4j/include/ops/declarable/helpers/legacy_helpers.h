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
#ifndef __H_LEGACY_HELPERS__
#define __H_LEGACY_HELPERS__
#include <array/NDArray.h>

namespace sd {
namespace ops {
namespace helpers {

SD_LIB_HIDDEN void reluDerivative(LaunchContext* context, NDArray* theFirst, NDArray* theSecond);
SD_LIB_HIDDEN void reluDerivative(LaunchContext* context, NDArray* theFirst, NDArray* theSecond,
                                  NDArray* theOutput);
SD_LIB_HIDDEN void relu6Derivative(LaunchContext* context, NDArray* theFirst, NDArray* theSecond,
                                   NDArray* theOutput);
SD_LIB_HIDDEN void leakyReluDerivative(LaunchContext* context, NDArray* theFirst, NDArray* theSecond,
                                       NDArray* theOutput, const float alpha);
SD_LIB_HIDDEN void eluDerivative(LaunchContext* context, NDArray* theFirst, NDArray* theSecond, NDArray* theOutput,
                                 const float alpha);
SD_LIB_HIDDEN void seluDerivative(LaunchContext* context, NDArray* theFirst, NDArray* theSecond,
                                  NDArray* theOutput);
SD_LIB_HIDDEN void cubeDerivative(LaunchContext* context, NDArray* theFirst, NDArray* theSecond,
                                  NDArray* theOutput);
SD_LIB_HIDDEN void reduceNorm1(LaunchContext* context, NDArray* theFirst, NDArray* theSecond, NDArray* theOutput);
SD_LIB_HIDDEN void sigmCrossEntropy(LaunchContext* context, NDArray* logits, NDArray* lablels, NDArray* theOutput);
SD_LIB_HIDDEN void sigmCrossEntropyGrad(LaunchContext* context, NDArray* logits, NDArray* lablels,
                                        NDArray* theOutput);
SD_LIB_HIDDEN void tanhDerivative(LaunchContext* context, NDArray* theFirst, NDArray* theSecond,
                                  NDArray* theOutput);
SD_LIB_HIDDEN void hardTanhDerivative(LaunchContext* context, NDArray* theFirst, NDArray* theSecond,
                                      NDArray* theOutput);
SD_LIB_HIDDEN void rationalTanhDerivative(LaunchContext* context, NDArray* theFirst, NDArray* theSecond,
                                          NDArray* theOutput);
SD_LIB_HIDDEN void rectifiedTanhDerivative(LaunchContext* context, NDArray* theFirst, NDArray* theSecond,
                                           NDArray* theOutput);
SD_LIB_HIDDEN void softSignDerivative(LaunchContext* context, NDArray* theFirst, NDArray* theSecond,
                                      NDArray* theOutput);
SD_LIB_HIDDEN void softPlusDerivative(LaunchContext* context, NDArray* theFirst, NDArray* theSecond,
                                      NDArray* theOutput);
SD_LIB_HIDDEN void sigmoidDerivative(LaunchContext* context, NDArray* theFirst, NDArray* theSecond,
                                     NDArray* theOutput);
SD_LIB_HIDDEN void hardSigmoidDerivative(LaunchContext* context, NDArray* theFirst, NDArray* theSecond,
                                         NDArray* theOutput);
SD_LIB_HIDDEN void logSumExp(LaunchContext* context, NDArray* input, NDArray* axis, NDArray* output);
SD_LIB_HIDDEN void logSumExp(LaunchContext* context, NDArray* input, NDArray* subtrah, NDArray* axis,
                             NDArray* output);
SD_LIB_HIDDEN void weightedCrossEntropyWithLogitsFunctor(LaunchContext* context, NDArray* targets,
                                                         NDArray* input, NDArray* weights, NDArray* output);
}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif

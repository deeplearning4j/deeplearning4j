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
/*
    SD_INLINE void reluDerivative(NDArray* theFirst, NDArray const* theSecond);
    SD_INLINE void reluDerivative(NDArray* theFirst, NDArray* theSecond, NDArray* theOutput);
    SD_INLINE void relu6Derivative(NDArray* theFirst, NDArray* theSecond, NDArray* theOutput);
    SD_INLINE void leakyReluDerivative(NDArray* theFirst, NDArray* theSecond, NDArray* theOutput);
    SD_INLINE void eluDerivative(NDArray* theFirst, NDArray* theSecond, NDArray* theOutput);
    SD_INLINE void seluDerivative(NDArray* theFirst, NDArray* theSecond, NDArray* theOutput);
    SD_INLINE void cubeDerivative(NDArray* theFirst, NDArray* theSecond, NDArray* theOutput);
    SD_INLINE void reduceNorm1(NDArray* theFirst, NDArray* theSecond, NDArray* theOutput);
    SD_INLINE void sxeLossWithLogits(NDArray* theFirst, NDArray* theSecond, NDArray* theOutput);
    SD_INLINE void tanhDerivative(NDArray* theFirst, NDArray* theSecond, NDArray* theOutput);
    SD_INLINE void hardTanhDerivative(NDArray* theFirst, NDArray* theSecond, NDArray* theOutput);
    SD_INLINE void rationalTanhDerivative(NDArray* theFirst, NDArray* theSecond, NDArray* theOutput);
    SD_INLINE void rectifiedTanhDerivative(NDArray* theFirst, NDArray* theSecond, NDArray* theOutput);
    SD_INLINE void softSignDerivative(NDArray* theFirst, NDArray* theSecond, NDArray* theOutput);
    SD_INLINE void softPlusDerivative(NDArray* theFirst, NDArray* theSecond, NDArray* theOutput);
    SD_INLINE void sigmoidDerivative(NDArray* theFirst, NDArray* theSecond, NDArray* theOutput);
    SD_INLINE void hardSigmoidDerivative(NDArray* theFirst, NDArray* theSecond, NDArray* theOutput);
*/
SD_LIB_HIDDEN void reluDerivative(sd::LaunchContext* context, NDArray* theFirst, NDArray* theSecond);
SD_LIB_HIDDEN void reluDerivative(sd::LaunchContext* context, NDArray* theFirst, NDArray* theSecond,
                                  NDArray* theOutput);
SD_LIB_HIDDEN void relu6Derivative(sd::LaunchContext* context, NDArray* theFirst, NDArray* theSecond,
                                   NDArray* theOutput);
SD_LIB_HIDDEN void leakyReluDerivative(sd::LaunchContext* context, NDArray* theFirst, NDArray* theSecond,
                                       NDArray* theOutput, const float alpha);
SD_LIB_HIDDEN void eluDerivative(sd::LaunchContext* context, NDArray* theFirst, NDArray* theSecond, NDArray* theOutput,
                                 const float alpha);
SD_LIB_HIDDEN void seluDerivative(sd::LaunchContext* context, NDArray* theFirst, NDArray* theSecond,
                                  NDArray* theOutput);
SD_LIB_HIDDEN void cubeDerivative(sd::LaunchContext* context, NDArray* theFirst, NDArray* theSecond,
                                  NDArray* theOutput);
SD_LIB_HIDDEN void reduceNorm1(sd::LaunchContext* context, NDArray* theFirst, NDArray* theSecond, NDArray* theOutput);
SD_LIB_HIDDEN void sigmCrossEntropy(sd::LaunchContext* context, NDArray* logits, NDArray* lablels, NDArray* theOutput);
SD_LIB_HIDDEN void sigmCrossEntropyGrad(sd::LaunchContext* context, NDArray* logits, NDArray* lablels,
                                        NDArray* theOutput);
SD_LIB_HIDDEN void tanhDerivative(sd::LaunchContext* context, NDArray* theFirst, NDArray* theSecond,
                                  NDArray* theOutput);
SD_LIB_HIDDEN void hardTanhDerivative(sd::LaunchContext* context, NDArray* theFirst, NDArray* theSecond,
                                      NDArray* theOutput);
SD_LIB_HIDDEN void rationalTanhDerivative(sd::LaunchContext* context, NDArray* theFirst, NDArray* theSecond,
                                          NDArray* theOutput);
SD_LIB_HIDDEN void rectifiedTanhDerivative(sd::LaunchContext* context, NDArray* theFirst, NDArray* theSecond,
                                           NDArray* theOutput);
SD_LIB_HIDDEN void softSignDerivative(sd::LaunchContext* context, NDArray* theFirst, NDArray* theSecond,
                                      NDArray* theOutput);
SD_LIB_HIDDEN void softPlusDerivative(sd::LaunchContext* context, NDArray* theFirst, NDArray* theSecond,
                                      NDArray* theOutput);
SD_LIB_HIDDEN void sigmoidDerivative(sd::LaunchContext* context, NDArray* theFirst, NDArray* theSecond,
                                     NDArray* theOutput);
SD_LIB_HIDDEN void hardSigmoidDerivative(sd::LaunchContext* context, NDArray* theFirst, NDArray* theSecond,
                                         NDArray* theOutput);
SD_LIB_HIDDEN void logSumExp(sd::LaunchContext* context, NDArray* input, NDArray* axis, NDArray* output);
SD_LIB_HIDDEN void logSumExp(sd::LaunchContext* context, NDArray* input, NDArray* subtrah, NDArray* axis,
                             NDArray* output);
SD_LIB_HIDDEN void weightedCrossEntropyWithLogitsFunctor(sd::LaunchContext* context, NDArray const* targets,
                                                         NDArray const* input, NDArray const* weights, NDArray* output);
}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif

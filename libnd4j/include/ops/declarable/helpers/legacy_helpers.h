/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
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
    FORCEINLINE void reluDerivative(NDArray* theFirst, NDArray const* theSecond);
    FORCEINLINE void reluDerivative(NDArray* theFirst, NDArray* theSecond, NDArray* theOutput);
    FORCEINLINE void relu6Derivative(NDArray* theFirst, NDArray* theSecond, NDArray* theOutput);
    FORCEINLINE void leakyReluDerivative(NDArray* theFirst, NDArray* theSecond, NDArray* theOutput);
    FORCEINLINE void eluDerivative(NDArray* theFirst, NDArray* theSecond, NDArray* theOutput);
    FORCEINLINE void seluDerivative(NDArray* theFirst, NDArray* theSecond, NDArray* theOutput);
    FORCEINLINE void cubeDerivative(NDArray* theFirst, NDArray* theSecond, NDArray* theOutput);
    FORCEINLINE void reduceNorm1(NDArray* theFirst, NDArray* theSecond, NDArray* theOutput);
    FORCEINLINE void sxeLossWithLogits(NDArray* theFirst, NDArray* theSecond, NDArray* theOutput);
    FORCEINLINE void tanhDerivative(NDArray* theFirst, NDArray* theSecond, NDArray* theOutput);
    FORCEINLINE void hardTanhDerivative(NDArray* theFirst, NDArray* theSecond, NDArray* theOutput);
    FORCEINLINE void rationalTanhDerivative(NDArray* theFirst, NDArray* theSecond, NDArray* theOutput);
    FORCEINLINE void rectifiedTanhDerivative(NDArray* theFirst, NDArray* theSecond, NDArray* theOutput);
    FORCEINLINE void softSignDerivative(NDArray* theFirst, NDArray* theSecond, NDArray* theOutput);
    FORCEINLINE void softPlusDerivative(NDArray* theFirst, NDArray* theSecond, NDArray* theOutput);
    FORCEINLINE void sigmoidDerivative(NDArray* theFirst, NDArray* theSecond, NDArray* theOutput);
    FORCEINLINE void hardSigmoidDerivative(NDArray* theFirst, NDArray* theSecond, NDArray* theOutput);
*/
    void reluDerivative(sd::LaunchContext * context, NDArray* theFirst, NDArray* theSecond);
    void reluDerivative(sd::LaunchContext * context, NDArray* theFirst, NDArray* theSecond, NDArray* theOutput);
    void relu6Derivative(sd::LaunchContext * context, NDArray* theFirst, NDArray* theSecond, NDArray* theOutput);
    void leakyReluDerivative(sd::LaunchContext * context, NDArray* theFirst, NDArray* theSecond, NDArray* theOutput, const float alpha);
    void eluDerivative(sd::LaunchContext * context, NDArray* theFirst, NDArray* theSecond, NDArray* theOutput, const float alpha);
    void seluDerivative(sd::LaunchContext * context, NDArray* theFirst, NDArray* theSecond, NDArray* theOutput);
    void cubeDerivative(sd::LaunchContext * context, NDArray* theFirst, NDArray* theSecond, NDArray* theOutput);
    void reduceNorm1(sd::LaunchContext * context, NDArray* theFirst, NDArray* theSecond, NDArray* theOutput);
    void sigmCrossEntropy(sd::LaunchContext * context, NDArray* logits, NDArray* lablels, NDArray* theOutput);
    void sigmCrossEntropyGrad(sd::LaunchContext * context, NDArray* logits, NDArray* lablels, NDArray* theOutput);
    void tanhDerivative(sd::LaunchContext * context, NDArray* theFirst, NDArray* theSecond, NDArray* theOutput);
    void hardTanhDerivative(sd::LaunchContext * context, NDArray* theFirst, NDArray* theSecond, NDArray* theOutput);
    void rationalTanhDerivative(sd::LaunchContext * context, NDArray* theFirst, NDArray* theSecond, NDArray* theOutput);
    void rectifiedTanhDerivative(sd::LaunchContext * context, NDArray* theFirst, NDArray* theSecond, NDArray* theOutput);
    void softSignDerivative(sd::LaunchContext * context, NDArray* theFirst, NDArray* theSecond, NDArray* theOutput);
    void softPlusDerivative(sd::LaunchContext * context, NDArray* theFirst, NDArray* theSecond, NDArray* theOutput);
    void sigmoidDerivative(sd::LaunchContext * context, NDArray* theFirst, NDArray* theSecond, NDArray* theOutput);
    void hardSigmoidDerivative(sd::LaunchContext * context, NDArray* theFirst, NDArray* theSecond, NDArray* theOutput);
    void logSumExp(sd::LaunchContext * context, NDArray* input, NDArray* axis, NDArray* output);
    void logSumExp(sd::LaunchContext * context, NDArray* input, NDArray* subtrah, NDArray* axis, NDArray* output);
    void weightedCrossEntropyWithLogitsFunctor(sd::LaunchContext * context, NDArray const* targets, NDArray const* input, NDArray const* weights, NDArray* output);
}
}
}
#endif

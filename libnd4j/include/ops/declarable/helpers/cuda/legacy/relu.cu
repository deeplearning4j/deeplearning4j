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
//  @author GS <sgazeos@gmail.com>
//

#include <ops/declarable/helpers/legacy_helpers.h>
#include <NDArrayFactory.h>
#include <op_boilerplate.h>

namespace nd4j {
    namespace ops {
        namespace helpers {

            template <typename T>
            linkage void reluDerivative__(NDArray* theFirst, NDArray* theSecond) {
                auto functor = LAMBDA_TT(x, y){
                    return x > (T) 0.f ? y : T(0.f);
                };

                theFirst->applyPairwiseLambda(theSecond, functor, nullptr);
            }

            void reluDerivative(nd4j::LaunchContext * context, NDArray* theFirst, NDArray* theSecond) {
                BUILD_SINGLE_SELECTOR(theFirst->dataType(), reluDerivative__, (theFirst, theSecond), FLOAT_TYPES);
            }

            template <typename T>
            linkage void reluDerivative_(NDArray* input, NDArray* epsilon, NDArray* output) {
                auto functor = LAMBDA_TT(x, y){
                    return x > (T)0.f ? y : T(0.f);
                };

                input->applyPairwiseLambda(epsilon, functor, output);
            }

            void reluDerivative(nd4j::LaunchContext * context, NDArray* theFirst, NDArray* theSecond, NDArray* theOutput) {
                BUILD_SINGLE_SELECTOR(theFirst->dataType(), reluDerivative_, (theFirst, theSecond, theOutput), FLOAT_TYPES);
            }

            template <typename T>
            linkage void relu6Derivative_(NDArray* input, NDArray* epsilon, NDArray* output) {
                auto functor = LAMBDA_TT(x, y){
                    return x > (T)0.f && x < (T)6.f? y : T(0.f);
                };

                input->applyPairwiseLambda(epsilon, functor, output);
            }

            void relu6Derivative(nd4j::LaunchContext * context, NDArray* theFirst, NDArray* theSecond, NDArray* theOutput) {
                BUILD_SINGLE_SELECTOR(theFirst->dataType(), relu6Derivative_, (theFirst, theSecond, theOutput), FLOAT_TYPES);
            }

            template <typename T>
            linkage void leakyReluDerivative_(NDArray* input, NDArray* epsilon, NDArray* output, const float alpha) {

                const T alphaT = static_cast<T>(alpha);

                auto functor = LAMBDA_TT(x, y, alphaT) {
                    return x < 0 ? alphaT * y : y;
                };

                input->applyPairwiseLambda(epsilon, functor, output);
            }

            void leakyReluDerivative(nd4j::LaunchContext * context, NDArray* theFirst, NDArray* theSecond, NDArray* theOutput, const float alpha) {
                BUILD_SINGLE_SELECTOR(theFirst->dataType(), leakyReluDerivative_, (theFirst, theSecond, theOutput, alpha), FLOAT_TYPES);
            }

            template <typename T>
            linkage void eluDerivative_(NDArray* input, NDArray* epsilon, NDArray* output, const float alpha) {

                const T alphaT = static_cast<T>(alpha);

                auto functor = LAMBDA_TT(x, y, alphaT){
                    return y * nd4j::math::nd4j_eluderivative<T,T>(x, alphaT);
                };

                input->applyPairwiseLambda(epsilon, functor, output);
            }

            void eluDerivative(nd4j::LaunchContext * context, NDArray* theFirst, NDArray* theSecond, NDArray* theOutput, const float alpha) {
                BUILD_SINGLE_SELECTOR(theFirst->dataType(), eluDerivative_, (theFirst, theSecond, theOutput, alpha), FLOAT_TYPES);
            }

            template <typename T>
            linkage void seluDerivative_(NDArray* input, NDArray* epsilon, NDArray* output) {
                auto functor = LAMBDA_TT(x, y){
                    return y * simdOps::SELUDerivative<T>::op(x, nullptr);
                };

                input->applyPairwiseLambda(epsilon, functor, output);
            }

            void seluDerivative(nd4j::LaunchContext * context, NDArray* theFirst, NDArray* theSecond, NDArray* theOutput) {
                BUILD_SINGLE_SELECTOR(theFirst->dataType(), seluDerivative_, (theFirst, theSecond, theOutput), FLOAT_TYPES);
            }
        }
    }
}
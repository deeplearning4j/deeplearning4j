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
            ////////////////////////////////////////////////////////////////////////
            template <typename T>
            linkage void tanhDerivative_(NDArray* input, NDArray* epsilon, NDArray* output) {
                auto functor = LAMBDA_TT(x, y){
                    T th = nd4j::math::nd4j_tanh<T,T>(x);
                    return y * ((T)1.0f - (th * th));
                };

                input->applyPairwiseLambda(epsilon, functor, output);
            }

            void tanhDerivative(nd4j::LaunchContext * context, NDArray* theFirst, NDArray* theSecond, NDArray* theOutput) {
                BUILD_SINGLE_SELECTOR(theFirst->dataType(), tanhDerivative_, (theFirst, theSecond, theOutput), FLOAT_TYPES);
            }

            // return static_cast<X>(d2) * simdOps::HardTanhDerivative<X>::op(d1, nullptr);
            template <typename T>
            linkage void hardTanhDerivative_(NDArray* input, NDArray* epsilon, NDArray* output) {
                auto functor = LAMBDA_TT(x, y){
                    T th = nd4j::math::nd4j_tanh<T,T>(x);
                    return y * simdOps::HardTanhDerivative<T>::op(x, nullptr);
                };

                input->applyPairwiseLambda(epsilon, functor, output);
            }

            void hardTanhDerivative(nd4j::LaunchContext * context, NDArray* theFirst, NDArray* theSecond, NDArray* theOutput) {
                BUILD_SINGLE_SELECTOR(theFirst->dataType(), hardTanhDerivative_, (theFirst, theSecond, theOutput), FLOAT_TYPES);
            }

            template <typename T>
            linkage void rationalTanhDerivative_(NDArray* input, NDArray* epsilon, NDArray* output) {
                auto functor = LAMBDA_TT(x, y){
                    return y * simdOps::RationalTanhDerivative<T>::op(x, nullptr);
                };

                input->applyPairwiseLambda(epsilon, functor, output);
            }

            void rationalTanhDerivative(nd4j::LaunchContext * context, NDArray* theFirst, NDArray* theSecond, NDArray* theOutput) {
                BUILD_SINGLE_SELECTOR(theFirst->dataType(), rationalTanhDerivative_, (theFirst, theSecond, theOutput), FLOAT_TYPES);
            }

            template <typename T>
            linkage void rectifiedTanhDerivative_(NDArray* input, NDArray* epsilon, NDArray* output) {
                auto functor = LAMBDA_TT(x, y){
                    return x > (T) 0.0f ? y * (nd4j::math::nd4j_tanhderivative<T,T>(x)) : (T) 0.0f;
                };

                input->applyPairwiseLambda(epsilon, functor, output);
            }

            void rectifiedTanhDerivative(nd4j::LaunchContext * context, NDArray* theFirst, NDArray* theSecond, NDArray* theOutput) {
                BUILD_SINGLE_SELECTOR(theFirst->dataType(), rectifiedTanhDerivative_, (theFirst, theSecond, theOutput), FLOAT_TYPES);
            }
        }
    }
}
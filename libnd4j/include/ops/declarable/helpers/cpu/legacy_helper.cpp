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

namespace nd4j {
namespace ops {
namespace helpers {
    template <typename T>
    static void reluDerivative__(NDArray* theFirst, NDArray* theSecond) {
        auto functor = LAMBDA_TT(x, y){
            return x > (T) 0.f ? y : T(0.f);
        };

        theFirst->applyPairwiseLambda<T>(theSecond, functor, nullptr);
    }
    BUILD_SINGLE_TEMPLATE(template void reluDerivative__, (NDArray* input, NDArray* epsilon), FLOAT_TYPES);

    void reluDerivative(NDArray* theFirst, NDArray* theSecond) {
        BUILD_SINGLE_SELECTOR(theFirst->dataType(), reluDerivative__, (theFirst, theSecond), FLOAT_TYPES);
    }

    template <typename T>
    static void reluDerivative_(NDArray* input, NDArray* epsilon, NDArray* output) {
        auto functor = LAMBDA_TT(x, y){
            return x > (T)0.f ? y : T(0.f);
        };

        input->applyPairwiseLambda<T>(epsilon, functor, output);
    }
    BUILD_SINGLE_TEMPLATE(template void reluDerivative_, (NDArray* input, NDArray* epsilon, NDArray*output);, FLOAT_TYPES);

    void reluDerivative(NDArray* theFirst, NDArray* theSecond, NDArray* theOutput) {
        BUILD_SINGLE_SELECTOR(theFirst->dataType(), reluDerivative_, (theFirst, theSecond, theOutput), FLOAT_TYPES);
    }


    void relu6Derivative(NDArray* theFirst, NDArray* theSecond, NDArray* theOutput) {}
    void leakyReluDerivative(NDArray* theFirst, NDArray* theSecond, NDArray* theOutput) {}
    void eluDerivative(NDArray* theFirst, NDArray* theSecond, NDArray* theOutput) {}
    void seluDerivative(NDArray* theFirst, NDArray* theSecond, NDArray* theOutput) {}
    void cubeDerivative(NDArray* theFirst, NDArray* theSecond, NDArray* theOutput) {}
    void reduceNorm1(NDArray* theFirst, NDArray* theSecond, NDArray* theOutput) {}
    void sxeLossWithLogits(NDArray* theFirst, NDArray* theSecond, NDArray* theOutput) {}
    void tanhDerivative(NDArray* theFirst, NDArray* theSecond, NDArray* theOutput) {}
    void hardTanhDerivative(NDArray* theFirst, NDArray* theSecond, NDArray* theOutput) {}
    void rationalTanhDerivative(NDArray* theFirst, NDArray* theSecond, NDArray* theOutput) {}
    void rectifiedTanhDerivative(NDArray* theFirst, NDArray* theSecond, NDArray* theOutput) {}
    void softSignDerivative(NDArray* theFirst, NDArray* theSecond, NDArray* theOutput) {}
    void softPlusDerivative(NDArray* theFirst, NDArray* theSecond, NDArray* theOutput) {}
    void sigmoidDerivative(NDArray* theFirst, NDArray* theSecond, NDArray* theOutput) {}
    void hardSigmoidDerivative(NDArray* theFirst, NDArray* theSecond, NDArray* theOutput) {}


}
}
}
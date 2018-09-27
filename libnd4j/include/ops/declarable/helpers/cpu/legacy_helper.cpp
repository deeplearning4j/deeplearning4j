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
            return (double(x) > 0.)?y:T(0.);
        };

        theFirst->applyPairwiseLambda(theSecond, functor, nullptr);
    }

    void reluDerivative(NDArray* theFirst, NDArray const* theSecond) {
        BUILD_SINGLE_SELECTOR(theFirst->dataType(), reluDerivative__, (theFirst, theSecond), NUMERIC_TYPES);
    }
    BUILD_SINGLE_TEMPLATE(template void reluDerivative__, (NDArray* input, NDArray* epsilon);, NUMERIC_TYPES);

    template <typename T>
    static void reluDerivative_(NDArray* input, NDArray* epsilon, NDArray* output) {
        auto functor = [](T x, T y) -> T{
            return (double(x) > 0.)?y:T(0.);
        };

        input->applyPairwiseLambda(epsilon, functor, output);
    }

    void reluDerivative(NDArray* theFirst, NDArray* theSecond, NDArray* theOutput) {
        BUILD_SINGLE_SELECTOR(theFirst->dataType(), reluDerivative_, (theFirst, theSecond, theOutput), LIBND4J_TYPES);
    }
    BUILD_SINGLE_TEMPLATE(template void reluDerivative_, (NDArray* input, NDArray* epsilon, NDArray *output);, LIBND4J_TYPES);

    void relu6Derivative(NDArray const* theFirst, NDArray const* theSecond, NDArray* theOutput) {}
    void leakyReluDerivative(NDArray const* theFirst, NDArray const* theSecond, NDArray* theOutput) {}
    void eluDerivative(NDArray const* theFirst, NDArray const* theSecond, NDArray* theOutput) {}
    void seluDerivative(NDArray const* theFirst, NDArray const* theSecond, NDArray* theOutput) {}
    void cubeDerivative(NDArray const* theFirst, NDArray const* theSecond, NDArray* theOutput) {}
    void reduceNorm1(NDArray const* theFirst, NDArray const* theSecond, NDArray* theOutput) {}
    void sxeLossWithLogits(NDArray const* theFirst, NDArray const* theSecond, NDArray* theOutput) {}
    void tanhDerivative(NDArray const* theFirst, NDArray const* theSecond, NDArray* theOutput) {}
    void hardTanhDerivative(NDArray const* theFirst, NDArray const* theSecond, NDArray* theOutput) {}
    void rationalTanhDerivative(NDArray const* theFirst, NDArray const* theSecond, NDArray* theOutput) {}
    void rectifiedTanhDerivative(NDArray const* theFirst, NDArray const* theSecond, NDArray* theOutput) {}
    void softSignDerivative(NDArray const* theFirst, NDArray const* theSecond, NDArray* theOutput) {}
    void softPlusDerivative(NDArray const* theFirst, NDArray const* theSecond, NDArray* theOutput) {}
    void sigmoidDerivative(NDArray const* theFirst, NDArray const* theSecond, NDArray* theOutput) {}
    void hardSigmoidDerivative(NDArray const* theFirst, NDArray const* theSecond, NDArray* theOutput) {}


}
}
}
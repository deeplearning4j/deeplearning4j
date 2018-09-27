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

    template <typename T>
    static void relu6Derivative_(NDArray* input, NDArray* epsilon, NDArray* output) {
        auto functor = LAMBDA_TT(x, y){
            return x > (T)0.f && x < (T)6.f? y : T(0.f);
        };

        input->applyPairwiseLambda<T>(epsilon, functor, output);
    }

    BUILD_SINGLE_TEMPLATE(template void relu6Derivative_, (NDArray* input, NDArray* epsilon, NDArray*output);, FLOAT_TYPES);

    void relu6Derivative(NDArray* theFirst, NDArray* theSecond, NDArray* theOutput) {
        BUILD_SINGLE_SELECTOR(theFirst->dataType(), relu6Derivative_, (theFirst, theSecond, theOutput), FLOAT_TYPES);
    }

    template <typename T>
    static void leakyReluDerivative_(NDArray* input, NDArray* epsilon, NDArray* output) {
        auto functor = LAMBDA_TT(x, y){
            return x >= (T)0.f? T(1.f) : T(0.f);
        };

        input->applyPairwiseLambda<T>(epsilon, functor, output);
    }

    BUILD_SINGLE_TEMPLATE(template void leakyReluDerivative_, (NDArray* input, NDArray* epsilon, NDArray*output);, FLOAT_TYPES);

    void leakyReluDerivative(NDArray* theFirst, NDArray* theSecond, NDArray* theOutput) {
        BUILD_SINGLE_SELECTOR(theFirst->dataType(), leakyReluDerivative_, (theFirst, theSecond, theOutput), FLOAT_TYPES);
    }

    template <typename T>
    static void eluDerivative_(NDArray* input, NDArray* epsilon, NDArray* output) {
        auto functor = LAMBDA_TT(x, y){
            return y * nd4j::math::nd4j_eluderivative(x);
        };

        input->applyPairwiseLambda<T>(epsilon, functor, output);
    }

    BUILD_SINGLE_TEMPLATE(template void eluDerivative_, (NDArray* input, NDArray* epsilon, NDArray*output);, FLOAT_TYPES);

    void eluDerivative(NDArray* theFirst, NDArray* theSecond, NDArray* theOutput) {
        BUILD_SINGLE_SELECTOR(theFirst->dataType(), eluDerivative_, (theFirst, theSecond, theOutput), FLOAT_TYPES);
    }

    template <typename T>
    static void seluDerivative_(NDArray* input, NDArray* epsilon, NDArray* output) {
        auto functor = LAMBDA_TT(x, y){
            return y * simdOps::SELUDerivative<T>::op(x, nullptr);
        };

        input->applyPairwiseLambda<T>(epsilon, functor, output);
    }

    BUILD_SINGLE_TEMPLATE(template void seluDerivative_, (NDArray* input, NDArray* epsilon, NDArray*output);, FLOAT_TYPES);

    void seluDerivative(NDArray* theFirst, NDArray* theSecond, NDArray* theOutput) {
        BUILD_SINGLE_SELECTOR(theFirst->dataType(), seluDerivative_, (theFirst, theSecond, theOutput), FLOAT_TYPES);
    }

    template <typename T>
    static void cubeDerivative_(NDArray* input, NDArray* epsilon, NDArray* output) {
        auto functor = LAMBDA_TT(x, y){
            return y * (3 * x * x);
        };

        input->applyPairwiseLambda<T>(epsilon, functor, output);
    }

    BUILD_SINGLE_TEMPLATE(template void cubeDerivative_, (NDArray* input, NDArray* epsilon, NDArray*output);, FLOAT_TYPES);

    void cubeDerivative(NDArray* theFirst, NDArray* theSecond, NDArray* theOutput) {
        BUILD_SINGLE_SELECTOR(theFirst->dataType(), cubeDerivative_, (theFirst, theSecond, theOutput), FLOAT_TYPES);
    }

    //return (x >= X(0.f) ? y: -y);
    template <typename T>
    static void reduceNorm1_(NDArray* input, NDArray* epsilon, NDArray* output) {
        auto functor = LAMBDA_TT(x, y){
            return x > T(0.f)? y : -y;
        };

        input->applyPairwiseLambda<T>(epsilon, functor, output);
    }

    BUILD_SINGLE_TEMPLATE(template void reduceNorm1_, (NDArray* input, NDArray* epsilon, NDArray*output);, FLOAT_TYPES);

    void reduceNorm1(NDArray* theFirst, NDArray* theSecond, NDArray* theOutput) {
        BUILD_SINGLE_SELECTOR(theFirst->dataType(), reduceNorm1_, (theFirst, theSecond, theOutput), FLOAT_TYPES);
    }
    template <typename T>
    static void sxeLossWithLogits_(NDArray* input, NDArray* epsilon, NDArray* output) {
        auto functor = LAMBDA_TT(x, y){
            return nd4j::math::nd4j_max<T>(x, (T)0.f) - x * y + nd4j::math::nd4j_log<T,T>((T)1.f + nd4j::math::nd4j_exp<T,T>(-nd4j::math::nd4j_abs(x)));
        };

        input->applyPairwiseLambda<T>(epsilon, functor, output);
    }

    BUILD_SINGLE_TEMPLATE(template void sxeLossWithLogits_, (NDArray* input, NDArray* epsilon, NDArray*output);, FLOAT_TYPES);

    void sxeLossWithLogits(NDArray* theFirst, NDArray* theSecond, NDArray* theOutput) {
        BUILD_SINGLE_SELECTOR(theFirst->dataType(), sxeLossWithLogits_, (theFirst, theSecond, theOutput), FLOAT_TYPES);
    }

    template <typename T>
    static void tanhDerivative_(NDArray* input, NDArray* epsilon, NDArray* output) {
        auto functor = LAMBDA_TT(x, y){
            T th = nd4j::math::nd4j_tanh<T,T>(x);
            return y * ((T)1.0f - (th * th));
        };

        input->applyPairwiseLambda<T>(epsilon, functor, output);
    }

    BUILD_SINGLE_TEMPLATE(template void tanhDerivative_, (NDArray* input, NDArray* epsilon, NDArray*output);, FLOAT_TYPES);

    void tanhDerivative(NDArray* theFirst, NDArray* theSecond, NDArray* theOutput) {
        BUILD_SINGLE_SELECTOR(theFirst->dataType(), tanhDerivative_, (theFirst, theSecond, theOutput), FLOAT_TYPES);
    }

    // return static_cast<X>(d2) * simdOps::HardTanhDerivative<X>::op(d1, nullptr);
    template <typename T>
    static void hardTanhDerivative_(NDArray* input, NDArray* epsilon, NDArray* output) {
        auto functor = LAMBDA_TT(x, y){
            T th = nd4j::math::nd4j_tanh<T,T>(x);
            return y * simdOps::HardTanhDerivative<T>::op(x, nullptr);
        };

        input->applyPairwiseLambda<T>(epsilon, functor, output);
    }

    BUILD_SINGLE_TEMPLATE(template void hardTanhDerivative_, (NDArray* input, NDArray* epsilon, NDArray*output);, FLOAT_TYPES);

    void hardTanhDerivative(NDArray* theFirst, NDArray* theSecond, NDArray* theOutput) {
        BUILD_SINGLE_SELECTOR(theFirst->dataType(), hardTanhDerivative_, (theFirst, theSecond, theOutput), FLOAT_TYPES);
    }

    template <typename T>
    static void rationalTanhDerivative_(NDArray* input, NDArray* epsilon, NDArray* output) {
        auto functor = LAMBDA_TT(x, y){
            return y * simdOps::RationalTanhDerivative<T>::op(x, nullptr);
        };

        input->applyPairwiseLambda<T>(epsilon, functor, output);
    }

    BUILD_SINGLE_TEMPLATE(template void rationalTanhDerivative_, (NDArray* input, NDArray* epsilon, NDArray*output);, FLOAT_TYPES);

    void rationalTanhDerivative(NDArray* theFirst, NDArray* theSecond, NDArray* theOutput) {
        BUILD_SINGLE_SELECTOR(theFirst->dataType(), rationalTanhDerivative_, (theFirst, theSecond, theOutput), FLOAT_TYPES);
    }

    template <typename T>
    static void rectifiedTanhDerivative_(NDArray* input, NDArray* epsilon, NDArray* output) {
        auto functor = LAMBDA_TT(x, y){
            return x > (T) 0.0f ? y * (nd4j::math::nd4j_tanhderivative<T>(x)) : (T) 0.0f;
        };

        input->applyPairwiseLambda<T>(epsilon, functor, output);
    }

    BUILD_SINGLE_TEMPLATE(template void rectifiedTanhDerivative_, (NDArray* input, NDArray* epsilon, NDArray*output);, FLOAT_TYPES);

    void rectifiedTanhDerivative(NDArray* theFirst, NDArray* theSecond, NDArray* theOutput) {
        BUILD_SINGLE_SELECTOR(theFirst->dataType(), rectifiedTanhDerivative_, (theFirst, theSecond, theOutput), FLOAT_TYPES);
    }

    //            X f = (X) 1.0f + nd4j::math::nd4j_abs<X>(d1);
    //            return (X) d2 * ((X) 1.0f / (f * f));

    template <typename T>
    static void softSignDerivative_(NDArray* input, NDArray* epsilon, NDArray* output) {
        auto functor = LAMBDA_TT(x, y){
            T ss = (T)1.f + nd4j::math::nd4j_abs<T>(x);
            return y * ((T) 1.0f  / (ss * ss));
        };

        input->applyPairwiseLambda<T>(epsilon, functor, output);
    }

    BUILD_SINGLE_TEMPLATE(template void softSignDerivative_, (NDArray* input, NDArray* epsilon, NDArray*output);, FLOAT_TYPES);

    void softSignDerivative(NDArray* theFirst, NDArray* theSecond, NDArray* theOutput) {
        BUILD_SINGLE_SELECTOR(theFirst->dataType(), softSignDerivative_, (theFirst, theSecond, theOutput), FLOAT_TYPES);
    }

    template <typename T>
    static void softPlusDerivative_(NDArray* input, NDArray* epsilon, NDArray* output) {
        auto functor = LAMBDA_TT(x, y){
            T p = nd4j::math::nd4j_pow<T, T, T>(static_cast<T>(M_E), x);
            return y * (p / (p + 1.));
        };

        input->applyPairwiseLambda<T>(epsilon, functor, output);
    }

    BUILD_SINGLE_TEMPLATE(template void softPlusDerivative_, (NDArray* input, NDArray* epsilon, NDArray*output);, FLOAT_TYPES);

    void softPlusDerivative(NDArray* theFirst, NDArray* theSecond, NDArray* theOutput) {
        BUILD_SINGLE_SELECTOR(theFirst->dataType(), softPlusDerivative_, (theFirst, theSecond, theOutput), FLOAT_TYPES);
    }
///
/// \param theFirst
/// \param theSecond
/// \param theOutput
    template <typename T>
    static void sigmoidDerivative_(NDArray* input, NDArray* epsilon, NDArray* output) {
        auto functor = LAMBDA_TT(x, y){
            T s = nd4j::math::nd4j_sigmoid<T>(x);
            return y * (s * ((T) 1.0f - s));
        };

        input->applyPairwiseLambda<T>(epsilon, functor, output);
    }

    BUILD_SINGLE_TEMPLATE(template void sigmoidDerivative_, (NDArray* input, NDArray* epsilon, NDArray*output);, FLOAT_TYPES);

    void sigmoidDerivative(NDArray* theFirst, NDArray* theSecond, NDArray* theOutput) {
        BUILD_SINGLE_SELECTOR(theFirst->dataType(), sigmoidDerivative_, (theFirst, theSecond, theOutput), FLOAT_TYPES);
    }

    template <typename T>
    static void hardSigmoidDerivative_(NDArray* input, NDArray* epsilon, NDArray* output) {
        auto functor = LAMBDA_TT(x, y){
            return y * simdOps::HardSigmoidDerivative<T>::op(x, nullptr);
        };

        input->applyPairwiseLambda<T>(epsilon, functor, output);
    }

    BUILD_SINGLE_TEMPLATE(template void hardSigmoidDerivative_, (NDArray* input, NDArray* epsilon, NDArray*output);, FLOAT_TYPES);

    void hardSigmoidDerivative(NDArray* theFirst, NDArray* theSecond, NDArray* theOutput) {
        BUILD_SINGLE_SELECTOR(theFirst->dataType(), hardSigmoidDerivative_, (theFirst, theSecond, theOutput), FLOAT_TYPES);
    }


}
}
}
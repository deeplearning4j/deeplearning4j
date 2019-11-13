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
#include <ops/ops.h>

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

    void reluDerivative(nd4j::LaunchContext * context, NDArray* theFirst, NDArray* theSecond) {
        BUILD_SINGLE_SELECTOR(theFirst->dataType(), reluDerivative__, (theFirst, theSecond), FLOAT_TYPES);
    }

    template <typename T>
    static void reluDerivative_(NDArray* input, NDArray* epsilon, NDArray* output) {

        T zero = (T) 0.f;
        auto functor = LAMBDA_TT(x, y, zero){
            return x > zero ? y : zero;
        };

        input->applyPairwiseLambda<T>(epsilon, functor, output);

        /*
        auto x =  input->bufferAsT<T>();
        auto y =  epsilon->bufferAsT<T>();
        auto z =  output->bufferAsT<T>();

        int length = input->lengthOf();

        T zero = (T) 0.f;

        PRAGMA_OMP_PARALLEL_FOR
        for (int e = 0; e < length; e++) {
            z[e] = x[e] > zero ? y[e] : zero;
        }
        */
    }

    void reluDerivative(nd4j::LaunchContext * context, NDArray* theFirst, NDArray* theSecond, NDArray* theOutput) {
        BUILD_SINGLE_SELECTOR(theFirst->dataType(), reluDerivative_, (theFirst, theSecond, theOutput), FLOAT_TYPES);
    }

    template <typename T>
    static void relu6Derivative_(NDArray* input, NDArray* epsilon, NDArray* output) {
        auto functor = LAMBDA_TT(x, y){
            return x > (T)0.f && x < (T)6.f? y : T(0.f);
        };

        input->applyPairwiseLambda<T>(epsilon, functor, output);
    }

    void relu6Derivative(nd4j::LaunchContext * context, NDArray* theFirst, NDArray* theSecond, NDArray* theOutput) {
        BUILD_SINGLE_SELECTOR(theFirst->dataType(), relu6Derivative_, (theFirst, theSecond, theOutput), FLOAT_TYPES);
    }

    template <typename T>
    static void leakyReluDerivative_(NDArray* input, NDArray* epsilon, NDArray* output, const float alpha) {

        const T alphaT = static_cast<T>(alpha);

        auto functor = LAMBDA_TT(x, y, alphaT) {
            return x < 0 ? alphaT * y : y;
        };

        input->applyPairwiseLambda<T>(epsilon, functor, output);
    }

    void leakyReluDerivative(nd4j::LaunchContext * context, NDArray* theFirst, NDArray* theSecond, NDArray* theOutput, const float alpha) {
        BUILD_SINGLE_SELECTOR(theFirst->dataType(), leakyReluDerivative_, (theFirst, theSecond, theOutput, alpha), FLOAT_TYPES);
    }

    template <typename T>
    static void eluDerivative_(NDArray* input, NDArray* epsilon, NDArray* output, const float alpha) {

        const T alphaT = static_cast<T>(alpha);

        auto functor = LAMBDA_TT(x, y, alphaT){
            return y * nd4j::math::nd4j_eluderivative<T,T>(x, alphaT);
        };

        input->applyPairwiseLambda<T>(epsilon, functor, output);
    }

    void eluDerivative(nd4j::LaunchContext * context, NDArray* theFirst, NDArray* theSecond, NDArray* theOutput, const float alpha) {
        BUILD_SINGLE_SELECTOR(theFirst->dataType(), eluDerivative_, (theFirst, theSecond, theOutput, alpha), FLOAT_TYPES);
    }

    template <typename T>
    static void seluDerivative_(NDArray* input, NDArray* epsilon, NDArray* output) {
        auto functor = LAMBDA_TT(x, y){
            return y * simdOps::SELUDerivative<T>::op(x, nullptr);
        };

        input->applyPairwiseLambda<T>(epsilon, functor, output);
    }

    void seluDerivative(nd4j::LaunchContext * context, NDArray* theFirst, NDArray* theSecond, NDArray* theOutput) {
        BUILD_SINGLE_SELECTOR(theFirst->dataType(), seluDerivative_, (theFirst, theSecond, theOutput), FLOAT_TYPES);
    }

    template <typename T>
    static void cubeDerivative_(NDArray* input, NDArray* epsilon, NDArray* output) {
        auto functor = LAMBDA_TT(x, y){
            return y * (3 * x * x);
        };

        input->applyPairwiseLambda<T>(epsilon, functor, output);
    }

    void cubeDerivative(nd4j::LaunchContext * context, NDArray* theFirst, NDArray* theSecond, NDArray* theOutput) {
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

    void reduceNorm1(nd4j::LaunchContext * context, NDArray* theFirst, NDArray* theSecond, NDArray* theOutput) {
        BUILD_SINGLE_SELECTOR(theFirst->dataType(), reduceNorm1_, (theFirst, theSecond, theOutput), FLOAT_TYPES);
    }

    ////////////////////////////////////////////////////////////////////////
    template <typename T>
    static void sigmCrossEntropy_(NDArray* logits, NDArray* labels, NDArray* output) {
        auto functor = LAMBDA_TT(x, y){
            return nd4j::math::nd4j_max<T>(x, (T)0.f) - x * y + nd4j::math::nd4j_log<T,T>((T)1.f + nd4j::math::nd4j_exp<T,T>(-nd4j::math::nd4j_abs(x)));
        };

        logits->applyPairwiseLambda<T>(labels, functor, output);
    }

    void sigmCrossEntropy(nd4j::LaunchContext * context, NDArray* logits, NDArray* labels, NDArray* output) {
        BUILD_SINGLE_SELECTOR(logits->dataType(), sigmCrossEntropy_, (logits, labels, output), FLOAT_TYPES);
    }

    ////////////////////////////////////////////////////////////////////////
    template <typename T>
    static void sigmCrossEntropyGrad_(NDArray* logits, NDArray* labels, NDArray* output) {
        // 1 - labels - 1 / (1 + exp(logits))
        auto functor = LAMBDA_TT(x, y) {
            if(x <= 0)
                return static_cast<T>(1.) - y - static_cast<T>(1.) / (static_cast<T>(1.) + nd4j::math::nd4j_exp<T,T>(x));
            auto e = nd4j::math::nd4j_exp<T,T>(-x);
            return static_cast<T>(1.) - y - e / (static_cast<T>(1.) + e);
        };

        logits->applyPairwiseLambda<T>(labels, functor, output);
    }

    void sigmCrossEntropyGrad(nd4j::LaunchContext * context, NDArray* logits, NDArray* labels, NDArray* output) {
        BUILD_SINGLE_SELECTOR(logits->dataType(), sigmCrossEntropyGrad_, (logits, labels, output), FLOAT_TYPES);
    }

    ////////////////////////////////////////////////////////////////////////
    template <typename T>
    static void tanhDerivative_(NDArray* input, NDArray* epsilon, NDArray* output) {
        auto functor = LAMBDA_TT(x, y){
            T th = nd4j::math::nd4j_tanh<T,T>(x);
            return y * ((T)1.0f - (th * th));
        };

        input->applyPairwiseLambda<T>(epsilon, functor, output);
    }

    void tanhDerivative(nd4j::LaunchContext * context, NDArray* theFirst, NDArray* theSecond, NDArray* theOutput) {
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

    void hardTanhDerivative(nd4j::LaunchContext * context, NDArray* theFirst, NDArray* theSecond, NDArray* theOutput) {
        BUILD_SINGLE_SELECTOR(theFirst->dataType(), hardTanhDerivative_, (theFirst, theSecond, theOutput), FLOAT_TYPES);
    }

    template <typename T>
    static void rationalTanhDerivative_(NDArray* input, NDArray* epsilon, NDArray* output) {
        auto functor = LAMBDA_TT(x, y){
            return y * simdOps::RationalTanhDerivative<T>::op(x, nullptr);
        };

        input->applyPairwiseLambda<T>(epsilon, functor, output);
    }

    void rationalTanhDerivative(nd4j::LaunchContext * context, NDArray* theFirst, NDArray* theSecond, NDArray* theOutput) {
        BUILD_SINGLE_SELECTOR(theFirst->dataType(), rationalTanhDerivative_, (theFirst, theSecond, theOutput), FLOAT_TYPES);
    }

    template <typename T>
    static void rectifiedTanhDerivative_(NDArray* input, NDArray* epsilon, NDArray* output) {
        auto functor = LAMBDA_TT(x, y){
            return x > (T) 0.0f ? y * (nd4j::math::nd4j_tanhderivative<T,T>(x)) : (T) 0.0f;
        };

        input->applyPairwiseLambda<T>(epsilon, functor, output);
    }

    void rectifiedTanhDerivative(nd4j::LaunchContext * context, NDArray* theFirst, NDArray* theSecond, NDArray* theOutput) {
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

    void softSignDerivative(nd4j::LaunchContext * context, NDArray* theFirst, NDArray* theSecond, NDArray* theOutput) {
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

    void softPlusDerivative(nd4j::LaunchContext * context, NDArray* theFirst, NDArray* theSecond, NDArray* theOutput) {
        BUILD_SINGLE_SELECTOR(theFirst->dataType(), softPlusDerivative_, (theFirst, theSecond, theOutput), FLOAT_TYPES);
    }
///
/// \param theFirst
/// \param theSecond
/// \param theOutput
    template <typename T>
    static void sigmoidDerivative_(NDArray* input, NDArray* epsilon, NDArray* output) {
        auto functor = LAMBDA_TT(x, y){
            T s = nd4j::math::nd4j_sigmoid<T,T>(x);
            return y * (s * ((T) 1.0f - s));
        };

        input->applyPairwiseLambda<T>(epsilon, functor, output);
    }

    void sigmoidDerivative(nd4j::LaunchContext * context, NDArray* theFirst, NDArray* theSecond, NDArray* theOutput) {
        BUILD_SINGLE_SELECTOR(theFirst->dataType(), sigmoidDerivative_, (theFirst, theSecond, theOutput), FLOAT_TYPES);
    }

    template <typename T>
    static void hardSigmoidDerivative_(NDArray* input, NDArray* epsilon, NDArray* output) {
        auto functor = LAMBDA_TT(x, y){
            return y * simdOps::HardSigmoidDerivative<T>::op(x, nullptr);
        };

        input->applyPairwiseLambda<T>(epsilon, functor, output);
    }

    void hardSigmoidDerivative(nd4j::LaunchContext * context, NDArray* theFirst, NDArray* theSecond, NDArray* theOutput) {
        BUILD_SINGLE_SELECTOR(theFirst->dataType(), hardSigmoidDerivative_, (theFirst, theSecond, theOutput), FLOAT_TYPES);
    }

    template <typename T>
    static void logSumExp_(NDArray* input, NDArray* axis, NDArray* output) {
        // reduce along axis with
        std::unique_ptr<NDArray> tempInput(input->dup());
        input->applyTransform(transform::Exp, tempInput.get());
        std::vector<int> axisVector;
        if (axis != nullptr) {
            axisVector.resize(axis->lengthOf());
            for (size_t i = 0; i < axisVector.size(); ++i)
                axisVector[i] = axis->e<int>(i);
        }
        tempInput->reduceAlongDimension(reduce::Sum, output, axisVector);
        output->applyTransform(transform::Log, nullptr, nullptr);
    }

    template <typename T>
    static void logSumExp_(NDArray* input, NDArray* subtrah, NDArray* axis, NDArray* output) {
        // reduce along axis with
        std::unique_ptr<NDArray> tempInput(input->dup());
        input->applyPairwiseTransform(pairwise::Subtract, subtrah, tempInput.get(), nullptr);
        tempInput->applyTransform(transform::Exp, nullptr, nullptr);

        std::vector<int> axisVector;
        if (axis != nullptr) {
            axisVector.resize(axis->lengthOf());
            for (size_t i = 0; i < axisVector.size(); ++i)
                axisVector[i] = axis->e<int>(i);
        }
        tempInput->reduceAlongDimension(reduce::Sum, output, axisVector);
        output->applyTransform(transform::Log, nullptr, nullptr);
    }

    void logSumExp(nd4j::LaunchContext * context, NDArray* input, NDArray* axis, NDArray* output) {
        BUILD_SINGLE_SELECTOR(input->dataType(), logSumExp_, (input, axis, output), FLOAT_TYPES);
    }

    void logSumExp(nd4j::LaunchContext * context, NDArray* input, NDArray* subtrah, NDArray* axis, NDArray* output) {
        BUILD_SINGLE_SELECTOR(input->dataType(), logSumExp_, (input, subtrah, axis, output), FLOAT_TYPES);
    }

//////////////////////////////////////////////////////////////////////////
template <typename T>
static void weightedCrossEntropyWithLogitsFunctor_(NDArray const* targets, NDArray const* input, NDArray const* weights, NDArray* output) {

    T posWeight = weights->e<T>(0);

    auto mainRoutineT1 = LAMBDA_TT(_x, _z, posWeight) {
        T targetWeight = (1. + (posWeight - (T)1.f) * _z);
        return (1. - _z) * _x +
               targetWeight * (nd4j::math::nd4j_log<T,T>((T)1.f + nd4j::math::nd4j_exp<T,T>(-nd4j::math::nd4j_abs(_x))) +
                               nd4j::math::nd4j_max(-_x, T(0.f))
               );
    };

    auto mainRoutineT2 = LAMBDA_TTT(_x, _z, _w) {
        return (((T)1.0 - _z) * _x) +
               _w * (nd4j::math::nd4j_log<T,T>(T(1.) + nd4j::math::nd4j_exp<T,T>(-nd4j::math::nd4j_abs(_x))) +
                     nd4j::math::nd4j_max(-_x, T(0.f)));
    };


    if (weights->isScalar()) {
        const_cast<NDArray*>(input)->applyPairwiseLambda<T>(const_cast<NDArray*>(targets), mainRoutineT1, output);
    }
    else
    {
        std::unique_ptr<NDArray> targetVector(new NDArray(*weights));
        targetVector->applyScalar(scalar::Add, -1.f);

        std::unique_ptr<NDArray> targetTensor(new NDArray(*targets));
        *targetTensor = (*targetVector * *targetTensor) + T(1.f);
        const_cast<NDArray*>(input)->applyTriplewiseLambda<T>(const_cast<NDArray*>(targets), targetTensor.get(), mainRoutineT2, output);
    }
}

void weightedCrossEntropyWithLogitsFunctor(nd4j::LaunchContext * context, NDArray const* targets, NDArray const* input, NDArray const* weights, NDArray* output) {
    BUILD_SINGLE_SELECTOR(targets->dataType(), weightedCrossEntropyWithLogitsFunctor_, (targets, input, weights, output), FLOAT_TYPES);
}

}
}
}
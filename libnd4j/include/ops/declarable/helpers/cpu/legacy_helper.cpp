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
//  @author GS <sgazeos@gmail.com>
//

#include <array/NDArrayFactory.h>
#include <ops/declarable/helpers/legacy_helpers.h>
#include <ops/ops.h>

namespace sd {
namespace ops {
namespace helpers {
template <typename T>
static void reluDerivative__(NDArray* theFirst, NDArray* theSecond) {
  auto functor = LAMBDA_TT(x, y) { return x > (T)0.f ? y : T(0.f); });

  theFirst->applyPairwiseLambda<T>(theSecond, functor, theFirst);
}

void reluDerivative(sd::LaunchContext* context, NDArray* theFirst, NDArray* theSecond) {
  BUILD_SINGLE_SELECTOR(theFirst->dataType(), reluDerivative__, (theFirst, theSecond), SD_FLOAT_TYPES);
}

template <typename T>
static void reluDerivative_(NDArray* input, NDArray* epsilon, NDArray* output) {
  T zero = (T)0.f;
  auto functor = LAMBDA_TT(x, y, zero) { return x > zero ? y : zero; });

  input->applyPairwiseLambda<T>(epsilon, functor, output);

}

void reluDerivative(sd::LaunchContext* context, NDArray* theFirst, NDArray* theSecond, NDArray* theOutput) {
  BUILD_SINGLE_SELECTOR(theFirst->dataType(), reluDerivative_, (theFirst, theSecond, theOutput), SD_FLOAT_TYPES);
}

template <typename T>
static void relu6Derivative_(NDArray* input, NDArray* epsilon, NDArray* output) {
  auto functor = LAMBDA_TT(x, y) { return x > (T)0.f && x < (T)6.f ? y : T(0.f); });

  input->applyPairwiseLambda<T>(epsilon, functor, output);
}

void relu6Derivative(sd::LaunchContext* context, NDArray* theFirst, NDArray* theSecond, NDArray* theOutput) {
  BUILD_SINGLE_SELECTOR(theFirst->dataType(), relu6Derivative_, (theFirst, theSecond, theOutput), SD_FLOAT_TYPES);
}

template <typename T>
static void leakyReluDerivative_(NDArray* input, NDArray* epsilon, NDArray* output, const float alpha) {
  const T alphaT = static_cast<T>(alpha);

  auto functor = LAMBDA_TT(x, y, alphaT) { return x < 0 ? alphaT * y : y; });

  input->applyPairwiseLambda<T>(epsilon, functor, output);
}

void leakyReluDerivative(sd::LaunchContext* context, NDArray* theFirst, NDArray* theSecond, NDArray* theOutput,
                         const float alpha) {
  BUILD_SINGLE_SELECTOR(theFirst->dataType(), leakyReluDerivative_, (theFirst, theSecond, theOutput, alpha),
                        SD_FLOAT_TYPES);
}

template <typename T>
static void eluDerivative_(NDArray* input, NDArray* epsilon, NDArray* output, const float alpha) {
  const T alphaT = static_cast<T>(alpha);

  auto functor = LAMBDA_TT(x, y, alphaT) { return y * sd::math::sd_eluderivative<T, T>(x, alphaT); });

  input->applyPairwiseLambda<T>(epsilon, functor, output);
}

void eluDerivative(sd::LaunchContext* context, NDArray* theFirst, NDArray* theSecond, NDArray* theOutput,
                   const float alpha) {
  BUILD_SINGLE_SELECTOR(theFirst->dataType(), eluDerivative_, (theFirst, theSecond, theOutput, alpha), SD_FLOAT_TYPES);
}

template <typename T>
static void seluDerivative_(NDArray* input, NDArray* epsilon, NDArray* output) {
  auto functor = LAMBDA_TT(x, y) { return y * simdOps::SELUDerivative<T>::op(x, nullptr); });

  input->applyPairwiseLambda<T>(epsilon, functor, output);
}

void seluDerivative(sd::LaunchContext* context, NDArray* theFirst, NDArray* theSecond, NDArray* theOutput) {
  BUILD_SINGLE_SELECTOR(theFirst->dataType(), seluDerivative_, (theFirst, theSecond, theOutput), SD_FLOAT_TYPES);
}

template <typename T>
static void cubeDerivative_(NDArray* input, NDArray* epsilon, NDArray* output) {
  auto functor = LAMBDA_TT(x, y) { return y * (3 * x * x); });

  input->applyPairwiseLambda<T>(epsilon, functor, output);
}

void cubeDerivative(sd::LaunchContext* context, NDArray* theFirst, NDArray* theSecond, NDArray* theOutput) {
  BUILD_SINGLE_SELECTOR(theFirst->dataType(), cubeDerivative_, (theFirst, theSecond, theOutput), SD_FLOAT_TYPES);
}

// return (x >= X(0.f) ? y: -y);
template <typename T>
static void reduceNorm1_(NDArray* input, NDArray* epsilon, NDArray* output) {
  auto functor = LAMBDA_TT(x, y) { return x > T(0.f) ? y : -y; });

  input->applyPairwiseLambda<T>(epsilon, functor, output);
}

void reduceNorm1(sd::LaunchContext* context, NDArray* theFirst, NDArray* theSecond, NDArray* theOutput) {
  BUILD_SINGLE_SELECTOR(theFirst->dataType(), reduceNorm1_, (theFirst, theSecond, theOutput), SD_FLOAT_TYPES);
}

////////////////////////////////////////////////////////////////////////
template <typename T>
static void sigmCrossEntropy_(NDArray* logits, NDArray* labels, NDArray* output) {
  auto functor = LAMBDA_TT(x, y) {
    return sd::math::sd_max<T>(x, (T)0.f) - x * y +
           sd::math::sd_log<T, T>((T)1.f + sd::math::sd_exp<T, T>(-sd::math::sd_abs<T,T>(x)));
  });

  logits->applyPairwiseLambda<T>(labels, functor, output);
}

void sigmCrossEntropy(sd::LaunchContext* context, NDArray* logits, NDArray* labels, NDArray* output) {
  BUILD_SINGLE_SELECTOR(logits->dataType(), sigmCrossEntropy_, (logits, labels, output), SD_FLOAT_TYPES);
}

////////////////////////////////////////////////////////////////////////
template <typename T>
static void sigmCrossEntropyGrad_(NDArray* logits, NDArray* labels, NDArray* output) {
  // 1 - labels - 1 / (1 + exp(logits))
  auto functor = LAMBDA_TT(x, y) {
    if (x <= 0) return static_cast<T>(1.) - y - static_cast<T>(1.) / (static_cast<T>(1.) + sd::math::sd_exp<T, T>(x));
    auto e = sd::math::sd_exp<T, T>(-x);
    return static_cast<T>(1.) - y - e / (static_cast<T>(1.) + e);
  });

  logits->applyPairwiseLambda<T>(labels, functor, output);
}

void sigmCrossEntropyGrad(sd::LaunchContext* context, NDArray* logits, NDArray* labels, NDArray* output) {
  BUILD_SINGLE_SELECTOR(logits->dataType(), sigmCrossEntropyGrad_, (logits, labels, output), SD_FLOAT_TYPES);
}

////////////////////////////////////////////////////////////////////////
template <typename T>
static void tanhDerivative_(NDArray* input, NDArray* epsilon, NDArray* output) {
  auto functor = LAMBDA_TT(x, y) {
    T th = sd::math::sd_tanh<T, T>(x);
    return y * ((T)1.0f - (th * th));
  });

  input->applyPairwiseLambda<T>(epsilon, functor, output);
}

void tanhDerivative(sd::LaunchContext* context, NDArray* theFirst, NDArray* theSecond, NDArray* theOutput) {
  BUILD_SINGLE_SELECTOR(theFirst->dataType(), tanhDerivative_, (theFirst, theSecond, theOutput), SD_FLOAT_TYPES);
}

template <typename T>
static void hardTanhDerivative_(NDArray* input, NDArray* epsilon, NDArray* output) {
  auto functor = LAMBDA_TT(x, y) {
    return y * simdOps::HardTanhDerivative<T>::op(x, nullptr);
  });

  input->applyPairwiseLambda<T>(epsilon, functor,output);
}

void hardTanhDerivative(sd::LaunchContext* context, NDArray* theFirst, NDArray* theSecond, NDArray* theOutput) {
  BUILD_SINGLE_SELECTOR(theFirst->dataType(), hardTanhDerivative_, (theFirst, theSecond, theOutput), SD_FLOAT_TYPES);
}

template <typename T>
static void rationalTanhDerivative_(NDArray* input, NDArray* epsilon, NDArray* output) {
  auto functor = LAMBDA_TT(x, y) { return y * simdOps::RationalTanhDerivative<T>::op(x, nullptr); });

  input->applyPairwiseLambda<T>(epsilon, functor, output);
}

void rationalTanhDerivative(sd::LaunchContext* context, NDArray* theFirst, NDArray* theSecond, NDArray* theOutput) {
  BUILD_SINGLE_SELECTOR(theFirst->dataType(), rationalTanhDerivative_, (theFirst, theSecond, theOutput),
                        SD_FLOAT_TYPES);
}

template <typename T>
static void rectifiedTanhDerivative_(NDArray* input, NDArray* epsilon, NDArray* output) {
  auto functor = LAMBDA_TT(x, y) { return x > (T)0.0f ? y * (sd::math::sd_tanhderivative<T, T>(x)) : (T)0.0f; });

  input->applyPairwiseLambda<T>(epsilon, functor, output);
}

void rectifiedTanhDerivative(sd::LaunchContext* context, NDArray* theFirst, NDArray* theSecond, NDArray* theOutput) {
  BUILD_SINGLE_SELECTOR(theFirst->dataType(), rectifiedTanhDerivative_, (theFirst, theSecond, theOutput),
                        SD_FLOAT_TYPES);
}



template <typename T>
static void softSignDerivative_(NDArray* input, NDArray* epsilon, NDArray* output) {
  auto functor = LAMBDA_TT(x, y) {
    T ss = (T)1.f + sd::math::sd_abs<T,T>(x);
    return y * ((T)1.0f / (ss * ss));
  });

  input->applyPairwiseLambda<T>(epsilon, functor, output);
}

void softSignDerivative(sd::LaunchContext* context, NDArray* theFirst, NDArray* theSecond, NDArray* theOutput) {
  BUILD_SINGLE_SELECTOR(theFirst->dataType(), softSignDerivative_, (theFirst, theSecond, theOutput), SD_FLOAT_TYPES);
}

template <typename T>
static void softPlusDerivative_(NDArray* input, NDArray* epsilon, NDArray* output) {
  auto functor = LAMBDA_TT(x, y) {
    T p = sd::math::sd_pow<T, T, T>(static_cast<T>(M_E), x);
    return y * (p / (p + 1.));
  });

  input->applyPairwiseLambda<T>(epsilon, functor, output);
}

void softPlusDerivative(sd::LaunchContext* context, NDArray* theFirst, NDArray* theSecond, NDArray* theOutput) {
  BUILD_SINGLE_SELECTOR(theFirst->dataType(), softPlusDerivative_, (theFirst, theSecond, theOutput), SD_FLOAT_TYPES);
}
///
/// \param theFirst
/// \param theSecond
/// \param theOutput
template <typename T>
static void sigmoidDerivative_(NDArray* input, NDArray* epsilon, NDArray* output) {
  auto functor = LAMBDA_TT(x, y) {
    T s = sd::math::sd_sigmoid<T, T>(x);
    return y * (s * ((T)1.0f - s));
  });

  input->applyPairwiseLambda<T>(epsilon, functor, output);
}

void sigmoidDerivative(sd::LaunchContext* context, NDArray* theFirst, NDArray* theSecond, NDArray* theOutput) {
  BUILD_SINGLE_SELECTOR(theFirst->dataType(), sigmoidDerivative_, (theFirst, theSecond, theOutput), SD_FLOAT_TYPES);
}

template <typename T>
static void hardSigmoidDerivative_(NDArray* input, NDArray* epsilon, NDArray* output) {
  auto functor = LAMBDA_TT(x, y) { return y * simdOps::HardSigmoidDerivative<T>::op(x, nullptr); });

  input->applyPairwiseLambda<T>(epsilon, functor, output);
}

void hardSigmoidDerivative(sd::LaunchContext* context, NDArray* theFirst, NDArray* theSecond, NDArray* theOutput) {
  BUILD_SINGLE_SELECTOR(theFirst->dataType(), hardSigmoidDerivative_, (theFirst, theSecond, theOutput), SD_FLOAT_TYPES);
}

template <typename T>
static void logSumExp_(NDArray* input, NDArray* axis, NDArray* output) {
  // reduce along axis with
  NDArray tempInput = input->dup();
  input->applyTransform(transform::Exp, &tempInput);
  std::vector<sd::LongType> axisVector;
  if (axis != nullptr) {
    axisVector.resize(axis->lengthOf());
    for (size_t i = 0; i < axisVector.size(); ++i) axisVector[i] = axis->e<sd::LongType>(i);
  }
  tempInput.reduceAlongDimension(reduce::Sum, output, &axisVector);
  output->applyTransform(transform::Log, output);
}

template <typename T>
static void logSumExp_(NDArray* input, NDArray* subtrah, NDArray* axis, NDArray* output) {
  // reduce along axis with
  NDArray tempInput = input->dup();
  input->applyPairwiseTransform(pairwise::Subtract, subtrah, &tempInput);
  tempInput.applyTransform(transform::Exp, &tempInput);

  std::vector<sd::LongType> axisVector;
  if (axis != nullptr) {
    axisVector.resize(axis->lengthOf());
    for (size_t i = 0; i < axisVector.size(); ++i) axisVector[i] = axis->e<sd::LongType>(i);
  }
  tempInput.reduceAlongDimension(reduce::Sum, output, &axisVector);
  output->applyTransform(transform::Log, output);
}

void logSumExp(sd::LaunchContext* context, NDArray* input, NDArray* axis, NDArray* output) {
  BUILD_SINGLE_SELECTOR(input->dataType(), logSumExp_, (input, axis, output), SD_FLOAT_TYPES);
}

void logSumExp(sd::LaunchContext* context, NDArray* input, NDArray* subtrah, NDArray* axis, NDArray* output) {
  BUILD_SINGLE_SELECTOR(input->dataType(), logSumExp_, (input, subtrah, axis, output), SD_FLOAT_TYPES);
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
static void weightedCrossEntropyWithLogitsFunctor_(NDArray * targets, NDArray * input, NDArray * weights,
                                                   NDArray* output) {
  T posWeight = weights->e<T>(0);

  auto mainRoutineT1 = LAMBDA_TT(_x, _z, posWeight) {
    T targetWeight = (1. + (posWeight - (T)1.f) * _z);
    return (1. - _z) * _x +
           targetWeight * (sd::math::sd_log<T, T>((T)1.f + sd::math::sd_exp<T, T>(-sd::math::sd_abs<T,T>(_x))) +
                           sd::math::sd_max(-_x, T(0.f)));
  });

  auto mainRoutineT2 = LAMBDA_TTT(_x, _z, _w) {
    return (((T)1.0 - _z) * _x) + _w * (sd::math::sd_log<T, T>(T(1.) + sd::math::sd_exp<T, T>(-sd::math::sd_abs<T,T>(_x))) +
                                        sd::math::sd_max(-_x, T(0.f)));
  });

  if (weights->isScalar()) {
    input->applyPairwiseLambda<T>(targets, mainRoutineT1, output);
  } else {
    std::unique_ptr<NDArray> targetVector(new NDArray(*weights));
    targetVector->applyScalar(scalar::Add, -1.f, targetVector.get());

    *targets = (*targets * *targets) + T(1.f);
    input->applyTriplewiseLambda<T>(targets, targets,mainRoutineT2, output);
  }
}

void weightedCrossEntropyWithLogitsFunctor(sd::LaunchContext* context, NDArray * targets, NDArray * input,
                                           NDArray * weights, NDArray* output) {
  BUILD_SINGLE_SELECTOR(targets->dataType(), weightedCrossEntropyWithLogitsFunctor_, (targets, input, weights, output),
                        SD_FLOAT_TYPES);
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd

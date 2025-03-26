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
#include <system/op_boilerplate.h>


namespace sd {
namespace ops {
namespace helpers {

template <typename T>
void reluDerivative__(NDArray* theFirst, NDArray* theSecond) {
  auto functor = LAMBDA_TT(x, y) { return x > (T)0.f ? y : T(0.f); });

  theFirst->applyPairwiseLambda(theSecond, functor,theFirst);
}

void reluDerivative(LaunchContext* context, NDArray* theFirst, NDArray* theSecond) {
  BUILD_SINGLE_SELECTOR(theFirst->dataType(), reluDerivative__, (theFirst, theSecond), SD_FLOAT_TYPES);
}

template <typename T>
void reluDerivative_(NDArray* input, NDArray* epsilon, NDArray* output) {
  auto functor = LAMBDA_TT(x, y) { return x > (T)0.f ? y : T(0.f); });

  input->applyPairwiseLambda(epsilon, functor, output);
}

void reluDerivative(LaunchContext* context, NDArray* theFirst, NDArray* theSecond, NDArray* theOutput) {
  BUILD_SINGLE_SELECTOR(theFirst->dataType(), reluDerivative_, (theFirst, theSecond, theOutput), SD_FLOAT_TYPES);
}

template <typename T>
void relu6Derivative_(NDArray* input, NDArray* epsilon, NDArray* output) {
  auto functor = LAMBDA_TT(x, y) { return x > (T)0.f && x < (T)6.f ? y : T(0.f); });

  input->applyPairwiseLambda(epsilon, functor, output);
}

void relu6Derivative(LaunchContext* context, NDArray* theFirst, NDArray* theSecond, NDArray* theOutput) {
  BUILD_SINGLE_SELECTOR(theFirst->dataType(), relu6Derivative_, (theFirst, theSecond, theOutput), SD_FLOAT_TYPES);
}

template <typename T>
void leakyReluDerivative_(NDArray* input, NDArray* epsilon, NDArray* output, const float alpha) {
  const T alphaT = static_cast<T>(alpha);

  auto functor = LAMBDA_TT(x, y, alphaT) { return x < 0 ? alphaT * y : y; });

  input->applyPairwiseLambda(epsilon, functor, output);
}

void leakyReluDerivative(LaunchContext* context, NDArray* theFirst, NDArray* theSecond, NDArray* theOutput,
                         const float alpha) {
  BUILD_SINGLE_SELECTOR(theFirst->dataType(), leakyReluDerivative_, (theFirst, theSecond, theOutput, alpha),
                        SD_FLOAT_TYPES);
}

template <typename T>
void eluDerivative_(NDArray* input, NDArray* epsilon, NDArray* output, const float alpha) {
  const T alphaT = static_cast<T>(alpha);

  auto functor = LAMBDA_TT(x, y, alphaT) { return y * math::sd_eluderivative<T, T>(x, alphaT); });

  input->applyPairwiseLambda(epsilon, functor, output);
}

void eluDerivative(LaunchContext* context, NDArray* theFirst, NDArray* theSecond, NDArray* theOutput,
                   const float alpha) {
  BUILD_SINGLE_SELECTOR(theFirst->dataType(), eluDerivative_, (theFirst, theSecond, theOutput, alpha), SD_FLOAT_TYPES);
}

template <typename T>
void seluDerivative_(NDArray* input, NDArray* epsilon, NDArray* output) {
  auto functor = LAMBDA_TT(x, y) { return y * simdOps::SELUDerivative<T>::op(x, nullptr); });

  input->applyPairwiseLambda(epsilon, functor, output);
}

void seluDerivative(LaunchContext* context, NDArray* theFirst, NDArray* theSecond, NDArray* theOutput) {
  BUILD_SINGLE_SELECTOR(theFirst->dataType(), seluDerivative_, (theFirst, theSecond, theOutput), SD_FLOAT_TYPES);
}
}  // namespace helpers
}  // namespace ops
}  // namespace sd

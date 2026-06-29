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
  // applyPairwiseLambda is a no-op stub on CUDA (std::function cannot run device-side).
  // Use scalar::RELUDerivative(x, 0) → 1 where x>0, 0 elsewhere; then multiply by epsilon.
  // reluDerivative__ is the in-place overload: output == theFirst, epsilon == theSecond.
  theFirst->applyScalar(scalar::RELUDerivative, (T)0.f, theFirst);
  theFirst->applyPairwiseTransform(pairwise::Multiply, theSecond, theFirst);
}

void reluDerivative(LaunchContext* context, NDArray* theFirst, NDArray* theSecond) {
  BUILD_SINGLE_SELECTOR(theFirst->dataType(), reluDerivative__, (theFirst, theSecond), SD_FLOAT_TYPES);
}

template <typename T>
void reluDerivative_(NDArray* input, NDArray* epsilon, NDArray* output) {
  // applyPairwiseLambda is a no-op stub on CUDA.
  // scalar::RELUDerivative(x, 0) → 1 where x>0, 0 elsewhere; then multiply by epsilon.
  input->applyScalar(scalar::RELUDerivative, (T)0.f, output);
  output->applyPairwiseTransform(pairwise::Multiply, epsilon, output);
}

void reluDerivative(LaunchContext* context, NDArray* theFirst, NDArray* theSecond, NDArray* theOutput) {
  BUILD_SINGLE_SELECTOR(theFirst->dataType(), reluDerivative_, (theFirst, theSecond, theOutput), SD_FLOAT_TYPES);
}

template <typename T>
void relu6Derivative_(NDArray* input, NDArray* epsilon, NDArray* output) {
  // applyPairwiseLambda is a no-op stub on CUDA.
  // Derivative is 1 if 0 < x < 6, else 0.
  // Step 1: output = (x > 0) ? 1 : 0  via scalar::RELUDerivative(x, 0)
  // Step 2: temp   = (x > 6) ? 1 : 0  via scalar::Step(x, 6)
  // Step 3: output = output - temp     → 1 iff 0 < x <= 6  (x == 6 is measure-zero)
  // Step 4: output *= epsilon
  input->applyScalar(scalar::RELUDerivative, (T)0.f, output);
  NDArray* temp = input->dup();
  temp->applyScalar(scalar::Step, (T)6.f, temp);
  output->applyPairwiseTransform(pairwise::Subtract, temp, output);
  delete temp;
  output->applyPairwiseTransform(pairwise::Multiply, epsilon, output);
}

void relu6Derivative(LaunchContext* context, NDArray* theFirst, NDArray* theSecond, NDArray* theOutput) {
  BUILD_SINGLE_SELECTOR(theFirst->dataType(), relu6Derivative_, (theFirst, theSecond, theOutput), SD_FLOAT_TYPES);
}

template <typename T>
void leakyReluDerivative_(NDArray* input, NDArray* epsilon, NDArray* output, const float alpha) {
  // applyPairwiseLambda is a no-op stub on CUDA.
  // scalar::LeakyRELUDerivative(x, alpha) → 1 if x>=0, alpha if x<0; then multiply by epsilon.
  input->applyScalar(scalar::LeakyRELUDerivative, static_cast<T>(alpha), output);
  output->applyPairwiseTransform(pairwise::Multiply, epsilon, output);
}

void leakyReluDerivative(LaunchContext* context, NDArray* theFirst, NDArray* theSecond, NDArray* theOutput,
                         const float alpha) {
  BUILD_SINGLE_SELECTOR(theFirst->dataType(), leakyReluDerivative_, (theFirst, theSecond, theOutput, alpha),
                        SD_FLOAT_TYPES);
}

template <typename T>
void eluDerivative_(NDArray* input, NDArray* epsilon, NDArray* output, const float alpha) {
  // applyPairwiseLambda is a no-op stub on CUDA.
  // scalar::ELUDerivative(x, alpha) → sd_eluderivative(x, alpha); then multiply by epsilon.
  input->applyScalar(scalar::ELUDerivative, static_cast<T>(alpha), output);
  output->applyPairwiseTransform(pairwise::Multiply, epsilon, output);
}

void eluDerivative(LaunchContext* context, NDArray* theFirst, NDArray* theSecond, NDArray* theOutput,
                   const float alpha) {
  BUILD_SINGLE_SELECTOR(theFirst->dataType(), eluDerivative_, (theFirst, theSecond, theOutput, alpha), SD_FLOAT_TYPES);
}

template <typename T>
void seluDerivative_(NDArray* input, NDArray* epsilon, NDArray* output) {
  // applyPairwiseLambda is a no-op stub on CUDA.
  // transform::SELUDerivative: x>0 → SELU_LAMBDA; x<=0 → SELU_ALPHA*SELU_LAMBDA*exp(x).
  input->applyTransform(transform::SELUDerivative, output);
  output->applyPairwiseTransform(pairwise::Multiply, epsilon, output);
}

void seluDerivative(LaunchContext* context, NDArray* theFirst, NDArray* theSecond, NDArray* theOutput) {
  BUILD_SINGLE_SELECTOR(theFirst->dataType(), seluDerivative_, (theFirst, theSecond, theOutput), SD_FLOAT_TYPES);
}
}  // namespace helpers
}  // namespace ops
}  // namespace sd

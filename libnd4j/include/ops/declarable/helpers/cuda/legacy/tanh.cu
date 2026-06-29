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
////////////////////////////////////////////////////////////////////////
template <typename T>
void tanhDerivative_(NDArray* input, NDArray* epsilon, NDArray* output) {
  // Compute output = 1 - tanh²(input) using the native CUDA transform op (device-side),
  // then multiply by epsilon.  applyPairwiseLambda with std::function cannot invoke the
  // lambda inside a CUDA kernel (the kernel is a no-op placeholder), so we use the
  // correct native-op path that works on both CPU and CUDA.
  input->applyTransform(transform::TanhDerivative, output);
  output->applyPairwiseTransform(pairwise::Multiply, epsilon, output);
}

void tanhDerivative(LaunchContext* context, NDArray* theFirst, NDArray* theSecond, NDArray* theOutput) {
  BUILD_SINGLE_SELECTOR(theFirst->dataType(), tanhDerivative_, (theFirst, theSecond, theOutput), SD_FLOAT_TYPES);
}

template <typename T>
void hardTanhDerivative_(NDArray* input, NDArray* epsilon, NDArray* output) {
  // Native CUDA transform + pairwise multiply (applyPairwiseLambda is broken on CUDA).
  input->applyTransform(transform::HardTanhDerivative, output);
  output->applyPairwiseTransform(pairwise::Multiply, epsilon, output);
}

void hardTanhDerivative(LaunchContext* context, NDArray* theFirst, NDArray* theSecond, NDArray* theOutput) {
  BUILD_SINGLE_SELECTOR(theFirst->dataType(), hardTanhDerivative_, (theFirst, theSecond, theOutput), SD_FLOAT_TYPES);
}

template <typename T>
void rationalTanhDerivative_(NDArray* input, NDArray* epsilon, NDArray* output) {
  // Native CUDA transform + pairwise multiply (applyPairwiseLambda is broken on CUDA).
  input->applyTransform(transform::RationalTanhDerivative, output);
  output->applyPairwiseTransform(pairwise::Multiply, epsilon, output);
}

void rationalTanhDerivative(LaunchContext* context, NDArray* theFirst, NDArray* theSecond, NDArray* theOutput) {
  BUILD_SINGLE_SELECTOR(theFirst->dataType(), rationalTanhDerivative_, (theFirst, theSecond, theOutput),
                        SD_FLOAT_TYPES);
}

template <typename T>
void rectifiedTanhDerivative_(NDArray* input, NDArray* epsilon, NDArray* output) {
  // Native CUDA transform + pairwise multiply (applyPairwiseLambda is broken on CUDA).
  // transform::RectifiedTanhDerivative computes: x > 0 ? (1 - tanh²(x)) : 0
  input->applyTransform(transform::RectifiedTanhDerivative, output);
  output->applyPairwiseTransform(pairwise::Multiply, epsilon, output);
}

void rectifiedTanhDerivative(LaunchContext* context, NDArray* theFirst, NDArray* theSecond, NDArray* theOutput) {
  BUILD_SINGLE_SELECTOR(theFirst->dataType(), rectifiedTanhDerivative_, (theFirst, theSecond, theOutput),
                        SD_FLOAT_TYPES);
}
}  // namespace helpers
}  // namespace ops
}  // namespace sd

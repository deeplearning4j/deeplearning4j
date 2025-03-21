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
// @author raver119@gmail.com
//
#include <helpers/BitwiseUtils.h>
#include <ops/declarable/helpers/toggle_bits.h>


namespace sd {
namespace ops {
namespace helpers {
template <typename T>
void toggle_bits__(NDArray *in, NDArray *out) {
  auto lambda = LAMBDA_T(_x) {
    return ~_x;
  };

  in->applyLambda(lambda, out);
}
BUILD_SINGLE_TEMPLATE(template void toggle_bits__, (NDArray * in, NDArray *out), SD_INTEGER_TYPES);

void __toggle_bits(LaunchContext *context, NDArray *in, NDArray *out) {
  BUILD_SINGLE_SELECTOR(in->dataType(), toggle_bits__, (in, out), SD_INTEGER_TYPES);
}
}  // namespace helpers
}  // namespace ops
}  // namespace sd

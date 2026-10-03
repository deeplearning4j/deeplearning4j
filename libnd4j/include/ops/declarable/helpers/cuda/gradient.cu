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
//  @author sgazeos@gmail.com
//
#include <ops/declarable/helpers/axis.h>
#include <system/op_boilerplate.h>


namespace sd {
namespace ops {
namespace helpers {
// output = input - step * weight, with array operations on the device (a lambda here runs on the host, where a CUDA
// graph capturing the op never sees it). The step is scaled into a temporary: the op may run in place, its output the
// input array.
void applyGradientDescent(LaunchContext* context, NDArray* input, NDArray* step, double weight, NDArray* output) {
  NDArray scaled(step->shapeInfo(), step->dataType(), false, context, false);
  step->applyScalar(scalar::Multiply, weight, &scaled);
  input->applyPairwiseTransform(pairwise::Subtract, &scaled, output);
}
}  // namespace helpers
}  // namespace ops
}  // namespace sd

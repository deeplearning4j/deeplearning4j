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
//  @author raver119@gmail.com
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_sigmoid)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/legacy_helpers.h>
namespace sd {
namespace ops {
CONFIGURABLE_OP_IMPL(sigmoid, 1, 1, true, 0, 0) {
  auto first = INPUT_VARIABLE(0);
  auto z = OUTPUT_VARIABLE(0);

  first->applyTransform(transform::Sigmoid, z);

  STORE_RESULT(*z);

  return Status::OK;
}

DECLARE_TYPES(sigmoid) {
  getOpDescriptor()->setAllowedInputTypes(0, ANY)->setAllowedOutputTypes(0, {ALL_FLOATS});
}

CONFIGURABLE_OP_IMPL(sigmoid_bp, 2, 1, true, 0, 0) {
  auto input = INPUT_VARIABLE(0);
  auto epsilon = INPUT_VARIABLE(1);

  auto z = OUTPUT_VARIABLE(0);

  helpers::sigmoidDerivative(block.launchContext(), input, epsilon, z);
  return Status::OK;
}

DECLARE_TYPES(sigmoid_bp) {
  getOpDescriptor()
      ->setAllowedInputTypes(0, ANY)
      ->setAllowedInputTypes(1, {FLOAT32, DOUBLE, HALF})
      ->setAllowedOutputTypes(0, {FLOAT32, DOUBLE, HALF});
}
}  // namespace ops
}  // namespace sd

#endif

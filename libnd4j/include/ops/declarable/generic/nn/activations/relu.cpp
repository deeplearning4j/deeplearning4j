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
#if NOT_EXCLUDED(OP_relu)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/legacy_helpers.h>

namespace sd {
namespace ops {
CONFIGURABLE_OP_IMPL(relu, 1, 1, true, 1, 0) {
  auto first = INPUT_VARIABLE(0);
  auto z = OUTPUT_VARIABLE(0);

  auto scalar = block.numT() > 0 ? block.getTArguments()->at(0) : 0.0;

  first->applyScalar(scalar::RELU, scalar, z);

  STORE_RESULT(*z);

  return Status::OK;
}

DECLARE_TYPES(relu) { getOpDescriptor()->setAllowedInputTypes(0, ANY)->setSameMode(true); }

CONFIGURABLE_OP_IMPL(relu_bp, 2, 1, true, 0, 0) {
  auto input = INPUT_VARIABLE(0);
  auto epsilon = INPUT_VARIABLE(1);

  auto z = OUTPUT_VARIABLE(0);
  helpers::reluDerivative(block.launchContext(), input, epsilon, z);
  return Status::OK;
}
DECLARE_SYN(ReluGrad, relu_bp);

DECLARE_TYPES(relu_bp) {
  getOpDescriptor()
      ->setAllowedInputTypes(0, ANY)
      ->setAllowedInputTypes(1, {FLOAT32, DOUBLE, HALF})
      ->setAllowedOutputTypes(0, {FLOAT32, DOUBLE, HALF});
}
}  // namespace ops
}  // namespace sd

#endif

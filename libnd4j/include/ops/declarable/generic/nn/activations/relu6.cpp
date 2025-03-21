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
// @author Yurii Shyrma, created on 16.02.2018
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_relu6)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/legacy_helpers.h>

namespace sd {
namespace ops {

////////////////////////////////////////////////////////////////////////
CONFIGURABLE_OP_IMPL(relu6, 1, 1, true, 1, 0) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  input->applyScalar(scalar::RELU6, T_ARG(0), output);

  return Status::OK;
}

DECLARE_TYPES(relu6) { getOpDescriptor()->setAllowedInputTypes(0, ANY)->setSameMode(true); }

////////////////////////////////////////////////////////////////////////
CONFIGURABLE_OP_IMPL(relu6_bp, 2, 1, true, 0, 0) {
  auto input = INPUT_VARIABLE(0);
  auto gradO = INPUT_VARIABLE(1);
  auto gradI = OUTPUT_VARIABLE(0);

  helpers::relu6Derivative(block.launchContext(), input, gradO, gradI);
  return Status::OK;
}

DECLARE_TYPES(relu6_bp) {
  getOpDescriptor()
      ->setAllowedInputTypes(0, ANY)
      ->setAllowedInputTypes(1, {FLOAT32, DOUBLE, HALF})
      ->setAllowedOutputTypes(0, {FLOAT32, DOUBLE, HALF});
}

}  // namespace ops
}  // namespace sd

#endif

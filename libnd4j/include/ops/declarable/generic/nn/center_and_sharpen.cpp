/*
*  ******************************************************************************
*  *
*  *
*  * This program and the accompanying materials are made available under the
*  * terms of the Apache License, Version 2.0 which is available at
*  * https://www.apache.org/licenses/LICENSE-2.0.
*  *
*  * See the NOTICE file distributed with this work for additional
*  * information regarding copyright ownership.
*  * Unless required by applicable law or agreed to in writing, software
*  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
*  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
*  * License for the specific language governing permissions and limitations
*  * under the License.
*  *
*  * SPDX-License-Identifier: Apache-2.0
*  *****************************************************************************
*/

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_center_and_sharpen)

#include <ops/declarable/headers/nn.h>
#include <ops/declarable/helpers/center_and_sharpen.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(center_and_sharpen, 2, 1, false, 0, 0) {
  auto input = INPUT_VARIABLE(0);
  auto center = INPUT_VARIABLE(1);
  auto output = OUTPUT_VARIABLE(0);

  double temperature = block.getTArguments()->size() > 0 ? T_ARG(0) : 0.07;

  helpers::centerAndSharpen(input, center, output, temperature, block.launchContext());

  return sd::Status::OK;
}

DECLARE_TYPES(center_and_sharpen) {
  getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS});
  getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
  getOpDescriptor()->addTraits(OP_TRAIT_UNARY_ELEMENTWISE | OP_TRAIT_FULLY_WRITING);
}

DECLARE_SHAPE_FN(center_and_sharpen) {
  return SHAPELIST(CONSTANT(inputShape->at(0)));
}

//////////////////////////////////////////////////////////////////////////

CUSTOM_OP_IMPL(center_and_sharpen_bp, 3, 2, false, 0, 0) {
  auto input = INPUT_VARIABLE(0);
  auto center = INPUT_VARIABLE(1);
  auto gradOutput = INPUT_VARIABLE(2);

  auto dLdInput = OUTPUT_VARIABLE(0);
  auto dLdCenter = OUTPUT_VARIABLE(1);

  double temperature = block.getTArguments()->size() > 0 ? T_ARG(0) : 0.07;

  helpers::centerAndSharpenBp(input, center, gradOutput, dLdInput, dLdCenter, temperature, block.launchContext());

  return sd::Status::OK;
}

DECLARE_TYPES(center_and_sharpen_bp) {
  getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS});
  getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
  getOpDescriptor()->addTraits(OP_TRAIT_UNARY_ELEMENTWISE | OP_TRAIT_FULLY_WRITING | OP_TRAIT_BACKWARD);
}

DECLARE_SHAPE_FN(center_and_sharpen_bp) {
  return SHAPELIST(CONSTANT(inputShape->at(0)), CONSTANT(inputShape->at(1)));
}

}  // namespace ops
}  // namespace sd

#endif

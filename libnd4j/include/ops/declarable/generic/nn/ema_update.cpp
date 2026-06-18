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
#if NOT_EXCLUDED(OP_ema_update)

#include <ops/declarable/headers/nn.h>
#include <ops/declarable/helpers/ema_update.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(ema_update, 2, 1, false, 0, 0) {
  auto model = INPUT_VARIABLE(0);
  auto shadow = INPUT_VARIABLE(1);
  auto output = OUTPUT_VARIABLE(0);

  double decay = block.getTArguments()->size() > 0 ? T_ARG(0) : 0.999;

  helpers::emaUpdate(model, shadow, output, decay, block.launchContext());

  return sd::Status::OK;
}

DECLARE_TYPES(ema_update) {
  getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS});
  getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
  getOpDescriptor()->addTraits(OP_TRAIT_DATA_MOVEMENT | OP_TRAIT_FULLY_WRITING);
}

DECLARE_SHAPE_FN(ema_update) {
  return SHAPELIST(CONSTANT(inputShape->at(0)));
}

//////////////////////////////////////////////////////////////////////////

CUSTOM_OP_IMPL(ema_update_bp, 3, 2, false, 0, 0) {
  auto model = INPUT_VARIABLE(0);
  auto shadow = INPUT_VARIABLE(1);
  auto gradOutput = INPUT_VARIABLE(2);

  auto dLdModel = OUTPUT_VARIABLE(0);
  auto dLdShadow = OUTPUT_VARIABLE(1);

  double decay = block.getTArguments()->size() > 0 ? T_ARG(0) : 0.999;

  helpers::emaUpdateBp(model, shadow, gradOutput, dLdModel, dLdShadow, decay, block.launchContext());

  return sd::Status::OK;
}

DECLARE_TYPES(ema_update_bp) {
  getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS});
  getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
  getOpDescriptor()->addTraits(OP_TRAIT_DATA_MOVEMENT | OP_TRAIT_FULLY_WRITING | OP_TRAIT_BACKWARD);
}

DECLARE_SHAPE_FN(ema_update_bp) {
  return SHAPELIST(CONSTANT(inputShape->at(0)), CONSTANT(inputShape->at(1)));
}

}  // namespace ops
}  // namespace sd

#endif

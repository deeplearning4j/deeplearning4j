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
#if NOT_EXCLUDED(OP_contrastive_loss)

#include <ops/declarable/headers/loss.h>
#include <ops/declarable/helpers/contrastive_loss.h>
#include <helpers/ConstantShapeHelper.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(contrastive_loss, 2, 1, false, 0, 0) {
  auto imageEmbeddings = INPUT_VARIABLE(0);
  auto textEmbeddings = INPUT_VARIABLE(1);
  auto output = OUTPUT_VARIABLE(0);

  double temperature = block.getTArguments()->size() > 0 ? T_ARG(0) : 1.0;

  helpers::contrastiveLoss(imageEmbeddings, textEmbeddings, output, temperature, block.launchContext());

  return sd::Status::OK;
}

DECLARE_TYPES(contrastive_loss) {
  getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS});
  getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
}

DECLARE_SHAPE_FN(contrastive_loss) {
  // Output is a scalar loss value matching input type
  return SHAPELIST(ConstantShapeHelper::getInstance().scalarShapeInfo(
      sd::ArrayOptions::dataType(inputShape->at(0))));
}

//////////////////////////////////////////////////////////////////////////

CUSTOM_OP_IMPL(contrastive_loss_grad, 2, 2, false, 0, 0) {
  auto imageEmbeddings = INPUT_VARIABLE(0);
  auto textEmbeddings = INPUT_VARIABLE(1);

  auto dLdImage = OUTPUT_VARIABLE(0);
  auto dLdText = OUTPUT_VARIABLE(1);

  double temperature = block.getTArguments()->size() > 0 ? T_ARG(0) : 1.0;

  helpers::contrastiveLossBp(imageEmbeddings, textEmbeddings, dLdImage, dLdText, temperature, block.launchContext());

  return sd::Status::OK;
}

DECLARE_TYPES(contrastive_loss_grad) {
  getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS});
  getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
}

DECLARE_SHAPE_FN(contrastive_loss_grad) {
  return SHAPELIST(CONSTANT(inputShape->at(0)), CONSTANT(inputShape->at(1)));
}

}  // namespace ops
}  // namespace sd

#endif

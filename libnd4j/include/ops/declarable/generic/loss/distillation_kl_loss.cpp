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
#if NOT_EXCLUDED(OP_distillation_kl_loss)
#include <ops/declarable/headers/loss.h>
#include <ops/declarable/helpers/distillation_kl_loss.h>
#include <helpers/ConstantShapeHelper.h>
namespace sd {
namespace ops {
CUSTOM_OP_IMPL(distillation_kl_loss, 2, 1, false, 0, 0) {
  auto studentLogits = INPUT_VARIABLE(0);  // [batch, classes]
  auto teacherLogits = INPUT_VARIABLE(1);  // [batch, classes]
  NDArray* hardLabels = block.width() > 2 ? INPUT_VARIABLE(2) : nullptr;  // optional [batch]
  auto output = OUTPUT_VARIABLE(0);
  double temperature = block.getTArguments()->size() > 0 ? T_ARG(0) : 4.0;
  double alpha = block.getTArguments()->size() > 1 ? T_ARG(1) : 0.5;
  helpers::distillationKLLoss(studentLogits, teacherLogits, hardLabels, output,
                               temperature, alpha, block.launchContext());
  return sd::Status::OK;
}
DECLARE_TYPES(distillation_kl_loss) {
  getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS});
  getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
}
DECLARE_SHAPE_FN(distillation_kl_loss) {
  return SHAPELIST(ConstantShapeHelper::getInstance().scalarShapeInfo(
      sd::ArrayOptions::dataType(inputShape->at(0))));
}
//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(distillation_kl_loss_grad, 2, 2, false, 0, 0) {
  auto studentLogits = INPUT_VARIABLE(0);
  auto teacherLogits = INPUT_VARIABLE(1);
  NDArray* hardLabels = block.width() > 2 ? INPUT_VARIABLE(2) : nullptr;
  auto dLdStudent = OUTPUT_VARIABLE(0);
  auto dLdTeacher = OUTPUT_VARIABLE(1);
  double temperature = block.getTArguments()->size() > 0 ? T_ARG(0) : 4.0;
  double alpha = block.getTArguments()->size() > 1 ? T_ARG(1) : 0.5;
  helpers::distillationKLLossBp(studentLogits, teacherLogits, hardLabels,
                                  dLdStudent, dLdTeacher, temperature, alpha,
                                  block.launchContext());
  return sd::Status::OK;
}
DECLARE_TYPES(distillation_kl_loss_grad) {
  getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS});
  getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
}
DECLARE_SHAPE_FN(distillation_kl_loss_grad) {
  return SHAPELIST(CONSTANT(inputShape->at(0)), CONSTANT(inputShape->at(1)));
}
}  // namespace ops
}  // namespace sd
#endif

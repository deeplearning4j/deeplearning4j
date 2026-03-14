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
#if NOT_EXCLUDED(OP_attention_distillation_loss)
#include <ops/declarable/headers/loss.h>
#include <ops/declarable/helpers/attention_distillation_loss.h>
#include <helpers/ConstantShapeHelper.h>
namespace sd {
namespace ops {
CUSTOM_OP_IMPL(attention_distillation_loss, 2, 1, false, 0, 0) {
  auto studentAttn = INPUT_VARIABLE(0);  // [batch, s_heads, seq, seq]
  auto teacherAttn = INPUT_VARIABLE(1);  // [batch, t_heads, seq, seq]
  auto output = OUTPUT_VARIABLE(0);
  helpers::attentionDistillationLoss(studentAttn, teacherAttn, output, block.launchContext());
  return sd::Status::OK;
}
DECLARE_TYPES(attention_distillation_loss) {
  getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS});
  getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
}
DECLARE_SHAPE_FN(attention_distillation_loss) {
  return SHAPELIST(ConstantShapeHelper::getInstance().scalarShapeInfo(
      sd::ArrayOptions::dataType(inputShape->at(0))));
}
//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(attention_distillation_loss_grad, 2, 2, false, 0, 0) {
  auto studentAttn = INPUT_VARIABLE(0);
  auto teacherAttn = INPUT_VARIABLE(1);
  auto dLdStudent = OUTPUT_VARIABLE(0);
  auto dLdTeacher = OUTPUT_VARIABLE(1);
  helpers::attentionDistillationLossBp(studentAttn, teacherAttn, dLdStudent, dLdTeacher,
                                        block.launchContext());
  return sd::Status::OK;
}
DECLARE_TYPES(attention_distillation_loss_grad) {
  getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS});
  getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
}
DECLARE_SHAPE_FN(attention_distillation_loss_grad) {
  return SHAPELIST(CONSTANT(inputShape->at(0)), CONSTANT(inputShape->at(1)));
}
}  // namespace ops
}  // namespace sd
#endif

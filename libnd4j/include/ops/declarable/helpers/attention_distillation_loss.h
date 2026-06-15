/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

#ifndef LIBND4J_HELPERS_ATTENTION_DISTILLATION_LOSS_H
#define LIBND4J_HELPERS_ATTENTION_DISTILLATION_LOSS_H

#include <system/op_boilerplate.h>
#include <array/NDArray.h>

namespace sd {
namespace ops {
namespace helpers {

/**
 * Attention Distillation Loss for matching student/teacher attention maps.
 *
 * L = MSE(student_attn, teacher_attn)
 *
 * If head counts differ, teacher attention heads are averaged to match student head count.
 *
 * @param studentAttn  [batch, s_heads, seq, seq]
 * @param teacherAttn  [batch, t_heads, seq, seq]
 * @param output       scalar loss
 * @param context      launch context
 */
SD_LIB_HIDDEN void attentionDistillationLoss(NDArray* studentAttn, NDArray* teacherAttn,
                                               NDArray* output, LaunchContext* context);

SD_LIB_HIDDEN void attentionDistillationLossBp(NDArray* studentAttn, NDArray* teacherAttn,
                                                 NDArray* dLdStudent, NDArray* dLdTeacher,
                                                 LaunchContext* context);

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif

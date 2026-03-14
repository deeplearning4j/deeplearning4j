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

#ifndef LIBND4J_HELPERS_FEATURE_DISTILLATION_LOSS_H
#define LIBND4J_HELPERS_FEATURE_DISTILLATION_LOSS_H

#include <system/op_boilerplate.h>
#include <array/NDArray.h>

namespace sd {
namespace ops {
namespace helpers {

/**
 * Feature Distillation Loss for intermediate layer matching (TinyBERT/MiniLM style).
 *
 * L = MSE(projection(student_hidden), teacher_hidden)
 *
 * If projection_weight is provided: L = MSE(student @ projection, teacher)
 * Otherwise: L = MSE(student, teacher) (dimensions must match)
 *
 * @param studentFeatures   [batch, d_student]
 * @param teacherFeatures   [batch, d_teacher]
 * @param projectionWeight  [d_student, d_teacher] (optional, may be nullptr)
 * @param output            scalar loss
 * @param context           launch context
 */
SD_LIB_HIDDEN void featureDistillationLoss(NDArray* studentFeatures, NDArray* teacherFeatures,
                                             NDArray* projectionWeight, NDArray* output,
                                             LaunchContext* context);

SD_LIB_HIDDEN void featureDistillationLossBp(NDArray* studentFeatures, NDArray* teacherFeatures,
                                               NDArray* projectionWeight,
                                               NDArray* dLdStudent, NDArray* dLdTeacher,
                                               NDArray* dLdProjection,
                                               LaunchContext* context);

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif

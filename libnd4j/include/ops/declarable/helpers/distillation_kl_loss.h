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

#ifndef LIBND4J_HELPERS_DISTILLATION_KL_LOSS_H
#define LIBND4J_HELPERS_DISTILLATION_KL_LOSS_H

#include <system/op_boilerplate.h>
#include <array/NDArray.h>

namespace sd {
namespace ops {
namespace helpers {

/**
 * Knowledge Distillation KL Divergence Loss.
 *
 * L = alpha * T^2 * KL(softmax(student/T) || softmax(teacher/T))
 *     + (1-alpha) * CE(student, hardLabels)
 *
 * @param studentLogits [batch, classes]
 * @param teacherLogits [batch, classes]
 * @param hardLabels    [batch] (optional, may be nullptr if alpha=1.0)
 * @param output        scalar loss
 * @param temperature   softmax temperature (default 4.0)
 * @param alpha         interpolation weight (default 0.5)
 * @param context       launch context
 */
SD_LIB_HIDDEN void distillationKLLoss(NDArray* studentLogits, NDArray* teacherLogits,
                                        NDArray* hardLabels, NDArray* output,
                                        double temperature, double alpha,
                                        LaunchContext* context);

SD_LIB_HIDDEN void distillationKLLossBp(NDArray* studentLogits, NDArray* teacherLogits,
                                          NDArray* hardLabels,
                                          NDArray* dLdStudent, NDArray* dLdTeacher,
                                          double temperature, double alpha,
                                          LaunchContext* context);

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif

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

#include <ops/declarable/helpers/distillation_kl_loss.h>
#include <math/templatemath.h>
#include <system/openmp_pragmas.h>
#include <cmath>
#include <algorithm>
#include <vector>

namespace sd {
namespace ops {
namespace helpers {

// Row-wise log-softmax with temperature scaling
static void logSoftmaxWithTemp(const double* logits, double* output, sd::LongType len, double temperature) {
    double maxVal = logits[0] / temperature;
    for (sd::LongType i = 1; i < len; i++) {
        double v = logits[i] / temperature;
        if (v > maxVal) maxVal = v;
    }
    double logSum = 0.0;
    for (sd::LongType i = 0; i < len; i++) {
        output[i] = logits[i] / temperature - maxVal;
        logSum += sd::math::sd_exp<double>(output[i]);
    }
    logSum = sd::math::sd_log<double>(logSum);
    for (sd::LongType i = 0; i < len; i++) {
        output[i] -= logSum;
    }
}

// Row-wise softmax with temperature
static void softmaxWithTemp(const double* logits, double* output, sd::LongType len, double temperature) {
    logSoftmaxWithTemp(logits, output, len, temperature);
    for (sd::LongType i = 0; i < len; i++) {
        output[i] = sd::math::sd_exp<double>(output[i]);
    }
}

void distillationKLLoss(NDArray* studentLogits, NDArray* teacherLogits,
                         NDArray* hardLabels, NDArray* output,
                         double temperature, double alpha,
                         LaunchContext* context) {
    auto batch = studentLogits->sizeAt(0);
    auto classes = studentLogits->sizeAt(1);

    double klLoss = 0.0;
    double ceLoss = 0.0;

    PRAGMA_OMP_PARALLEL_FOR_REDUCTION(+:klLoss, +:ceLoss)
    for (sd::LongType b = 0; b < batch; b++) {
        // Extract logits for this sample
        std::vector<double> sLogits(classes), tLogits(classes);
        for (sd::LongType c = 0; c < classes; c++) {
            sLogits[c] = studentLogits->e<double>(b, c);
            tLogits[c] = teacherLogits->e<double>(b, c);
        }

        // KL divergence: KL(P_teacher || P_student) = sum(P_t * (log P_t - log P_s))
        std::vector<double> teacherSoftmax(classes);
        std::vector<double> teacherLogSoftmax(classes);
        softmaxWithTemp(tLogits.data(), teacherSoftmax.data(), classes, temperature);
        logSoftmaxWithTemp(tLogits.data(), teacherLogSoftmax.data(), classes, temperature);

        std::vector<double> studentLogSoftmax(classes);
        logSoftmaxWithTemp(sLogits.data(), studentLogSoftmax.data(), classes, temperature);

        for (sd::LongType c = 0; c < classes; c++) {
            klLoss += teacherSoftmax[c] * (teacherLogSoftmax[c] - studentLogSoftmax[c]);
        }

        // Hard-label CE if alpha < 1 and hardLabels provided
        if (alpha < 1.0 && hardLabels != nullptr) {
            // log-softmax of student at temperature 1
            std::vector<double> studentLogSm1(classes);
            logSoftmaxWithTemp(sLogits.data(), studentLogSm1.data(), classes, 1.0);

            auto label = hardLabels->e<sd::LongType>(b);
            if (label >= 0 && label < classes) {
                ceLoss += -studentLogSm1[label];
            }
        }
    }

    // Average over batch
    klLoss /= batch;
    ceLoss /= batch;

    // Combined loss: alpha * T^2 * KL + (1 - alpha) * CE
    double totalLoss = alpha * temperature * temperature * klLoss + (1.0 - alpha) * ceLoss;
    output->p(0, totalLoss);
}

void distillationKLLossBp(NDArray* studentLogits, NDArray* teacherLogits,
                            NDArray* hardLabels,
                            NDArray* dLdStudent, NDArray* dLdTeacher,
                            double temperature, double alpha,
                            LaunchContext* context) {
    auto batch = studentLogits->sizeAt(0);
    auto classes = studentLogits->sizeAt(1);

    double scale = 1.0 / batch;

    PRAGMA_OMP_PARALLEL_FOR
    for (sd::LongType b = 0; b < batch; b++) {
        std::vector<double> sLogits(classes), tLogits(classes);
        for (sd::LongType c = 0; c < classes; c++) {
            sLogits[c] = studentLogits->e<double>(b, c);
            tLogits[c] = teacherLogits->e<double>(b, c);
        }

        // Softmax of student and teacher at temperature T
        std::vector<double> sSoftmax(classes), tSoftmax(classes);
        softmaxWithTemp(sLogits.data(), sSoftmax.data(), classes, temperature);
        softmaxWithTemp(tLogits.data(), tSoftmax.data(), classes, temperature);

        for (sd::LongType c = 0; c < classes; c++) {
            // dL/d(student_logits) from KL part:
            // d/d(s_c) [ alpha * T^2 * sum_j p_t_j * (log p_t_j - log p_s_j) ]
            // = alpha * T^2 * (-1/T) * (p_t_c / p_s_c * p_s_c - ... ) simplified:
            // = alpha * T * (p_s_c - p_t_c)
            double dKL_student = alpha * temperature * (sSoftmax[c] - tSoftmax[c]);

            // Hard-label CE gradient for student
            double dCE_student = 0.0;
            if (alpha < 1.0 && hardLabels != nullptr) {
                std::vector<double> sSm1(classes);
                softmaxWithTemp(sLogits.data(), sSm1.data(), classes, 1.0);
                auto label = hardLabels->e<sd::LongType>(b);
                double target = (c == label) ? 1.0 : 0.0;
                dCE_student = (1.0 - alpha) * (sSm1[c] - target);
            }

            dLdStudent->p(b, c, (dKL_student + dCE_student) * scale);

            // dL/d(teacher_logits) from KL part:
            // Gradient through teacher softmax
            // = alpha * T * (p_t_c * (log p_t_c - log p_s_c + 1) - sum_j p_t_j * (log p_t_j - log p_s_j + 1) * p_t_c)
            // Simplified: alpha * T * p_t_c * (1 + log(p_t_c/p_s_c) - KL - 1)
            // For practical purposes, teacher gradient is small since teacher is typically frozen
            std::vector<double> tLogSm(classes), sLogSm(classes);
            logSoftmaxWithTemp(tLogits.data(), tLogSm.data(), classes, temperature);
            logSoftmaxWithTemp(sLogits.data(), sLogSm.data(), classes, temperature);

            double klSum = 0.0;
            for (sd::LongType j = 0; j < classes; j++) {
                klSum += tSoftmax[j] * (tLogSm[j] - sLogSm[j]);
            }
            double dKL_teacher = alpha * temperature * tSoftmax[c] * ((tLogSm[c] - sLogSm[c]) - klSum);
            dLdTeacher->p(b, c, dKL_teacher * scale);
        }
    }
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd

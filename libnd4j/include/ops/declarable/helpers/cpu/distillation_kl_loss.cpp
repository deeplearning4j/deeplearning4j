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
        logSum += sd::math::sd_exp<double, double>(output[i]);
    }
    logSum = sd::math::sd_log<double, double>(logSum);
    for (sd::LongType i = 0; i < len; i++) {
        output[i] -= logSum;
    }
}

// Row-wise softmax with temperature
static void softmaxWithTemp(const double* logits, double* output, sd::LongType len, double temperature) {
    logSoftmaxWithTemp(logits, output, len, temperature);
    for (sd::LongType i = 0; i < len; i++) {
        output[i] = sd::math::sd_exp<double, double>(output[i]);
    }
}

void distillationKLLoss(NDArray* studentLogits, NDArray* teacherLogits,
                         NDArray* hardLabels, NDArray* output,
                         double temperature, double alpha,
                         LaunchContext* context) {
    // Support rank-2 [batch, classes] and rank-3 [batch, seq, classes].
    // For rank-3 inputs we flatten the two leading dims into a single batch dimension.
    const int rank = studentLogits->rankOf();
    if (rank < 2 || rank > 3) {
        throw std::runtime_error("distillationKLLoss: input rank must be 2 or 3, got " +
                                 std::to_string(rank));
    }

    sd::LongType batch, classes;
    if (rank == 3) {
        batch = studentLogits->sizeAt(0) * studentLogits->sizeAt(1);
        classes = studentLogits->sizeAt(2);
    } else {
        batch = studentLogits->sizeAt(0);
        classes = studentLogits->sizeAt(1);
    }

    // Validate class dimension matches before entering OMP (throws from OMP → terminate)
    if (classes <= 0) {
        throw std::runtime_error("distillationKLLoss: class dimension must be > 0");
    }

    // For rank-3, use flattened linear index: flat_idx = b0*sizeAt(1) + b1
    // We access elements via raw buffer offset to avoid rank-specific e() calls.
    // Strides for a c-order rank-3 [B, S, V]: stride0=S*V, stride1=V, stride2=1
    // For rank-2 [B, V]: stride0=V, stride1=1
    const sd::LongType stride0 = (rank == 3) ? studentLogits->sizeAt(1) * classes : classes;
    // stride for flat batch (either sizeAt(1)*classes or classes depending on rank)
    // flat batch index b in [0, batch): raw offset = b * classes (already encoded above in stride0 choice)
    // We just need: for flat b, elements b*classes .. b*classes+classes-1
    // Because c-order and we treat the leading dims as one flat batch, this is correct.

    double klLoss = 0.0;
    double ceLoss = 0.0;

    PRAGMA_OMP_PARALLEL_FOR_REDUCTION(+:klLoss, ceLoss)
    for (sd::LongType b = 0; b < batch; b++) {
        // Extract logits for this sample
        std::vector<double> sLogits(classes), tLogits(classes);
        const sd::LongType offset = b * classes;
        for (sd::LongType c = 0; c < classes; c++) {
            sLogits[c] = studentLogits->e<double>(offset + c);
            tLogits[c] = teacherLogits->e<double>(offset + c);
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
    // Support rank-2 [batch, classes] and rank-3 [batch, seq, classes].
    const int rank = studentLogits->rankOf();
    if (rank < 2 || rank > 3) {
        throw std::runtime_error("distillationKLLossBp: input rank must be 2 or 3, got " +
                                 std::to_string(rank));
    }

    sd::LongType batch, classes;
    if (rank == 3) {
        batch = studentLogits->sizeAt(0) * studentLogits->sizeAt(1);
        classes = studentLogits->sizeAt(2);
    } else {
        batch = studentLogits->sizeAt(0);
        classes = studentLogits->sizeAt(1);
    }

    if (classes <= 0) {
        throw std::runtime_error("distillationKLLossBp: class dimension must be > 0");
    }

    double scale = 1.0 / batch;

    PRAGMA_OMP_PARALLEL_FOR
    for (sd::LongType b = 0; b < batch; b++) {
        std::vector<double> sLogits(classes), tLogits(classes);
        const sd::LongType offset = b * classes;
        for (sd::LongType c = 0; c < classes; c++) {
            sLogits[c] = studentLogits->e<double>(offset + c);
            tLogits[c] = teacherLogits->e<double>(offset + c);
        }

        // Softmax of student and teacher at temperature T
        std::vector<double> sSoftmax(classes), tSoftmax(classes);
        softmaxWithTemp(sLogits.data(), sSoftmax.data(), classes, temperature);
        softmaxWithTemp(tLogits.data(), tSoftmax.data(), classes, temperature);

        // Compute log-softmaxes once per sample (not once per class)
        std::vector<double> tLogSm(classes), sLogSm(classes);
        logSoftmaxWithTemp(tLogits.data(), tLogSm.data(), classes, temperature);
        logSoftmaxWithTemp(sLogits.data(), sLogSm.data(), classes, temperature);

        // Pre-compute KL sum for teacher gradient (avoids O(classes^2) recomputation)
        double klSum = 0.0;
        for (sd::LongType j = 0; j < classes; j++) {
            klSum += tSoftmax[j] * (tLogSm[j] - sLogSm[j]);
        }

        // Hard-label CE softmax at T=1 (computed once per sample, not per class)
        std::vector<double> sSm1(classes);
        sd::LongType hardLabel = -1;
        if (alpha < 1.0 && hardLabels != nullptr) {
            softmaxWithTemp(sLogits.data(), sSm1.data(), classes, 1.0);
            hardLabel = hardLabels->e<sd::LongType>(b);
        }

        for (sd::LongType c = 0; c < classes; c++) {
            // dL/d(student_logits) from KL part:
            // = alpha * T * (p_s_c - p_t_c)
            double dKL_student = alpha * temperature * (sSoftmax[c] - tSoftmax[c]);

            // Hard-label CE gradient for student
            double dCE_student = 0.0;
            if (alpha < 1.0 && hardLabels != nullptr && hardLabel >= 0 && hardLabel < classes) {
                double target = (c == hardLabel) ? 1.0 : 0.0;
                dCE_student = (1.0 - alpha) * (sSm1[c] - target);
            }

            dLdStudent->p(offset + c, (dKL_student + dCE_student) * scale);

            // dL/d(teacher_logits) from KL part via teacher softmax gradient
            double dKL_teacher = alpha * temperature * tSoftmax[c] * ((tLogSm[c] - sLogSm[c]) - klSum);
            dLdTeacher->p(offset + c, dKL_teacher * scale);
        }
    }
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd

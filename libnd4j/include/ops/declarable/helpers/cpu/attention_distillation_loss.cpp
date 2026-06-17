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

#include <ops/declarable/helpers/attention_distillation_loss.h>
#include <system/openmp_pragmas.h>

namespace sd {
namespace ops {
namespace helpers {

void attentionDistillationLoss(NDArray* studentAttn, NDArray* teacherAttn,
                                NDArray* output, LaunchContext* context) {
    auto batch = studentAttn->sizeAt(0);
    auto sHeads = studentAttn->sizeAt(1);
    auto tHeads = teacherAttn->sizeAt(1);
    auto seqLen = studentAttn->sizeAt(2);

    double totalLoss = 0.0;
    double totalElements = 0.0;

    if (sHeads == tHeads) {
        // Direct MSE between matching heads
        auto diff = *studentAttn - *teacherAttn;
        auto diffSq = *diff * *diff;
        totalElements = static_cast<double>(diffSq->lengthOf());
        totalLoss = diffSq->sumNumber().e<double>(0) / totalElements;
        delete diff;
        delete diffSq;
    } else {
        // Average teacher heads to match student head count
        // Teacher heads per student head
        sd::LongType ratio = tHeads / sHeads;

        PRAGMA_OMP_PARALLEL_FOR_REDUCTION(+:totalLoss, totalElements)
        for (sd::LongType b = 0; b < batch; b++) {
            for (sd::LongType sh = 0; sh < sHeads; sh++) {
                for (sd::LongType i = 0; i < seqLen; i++) {
                    for (sd::LongType j = 0; j < seqLen; j++) {
                        double sVal = studentAttn->e<double>(b, sh, i, j);

                        // Average corresponding teacher heads
                        double tAvg = 0.0;
                        sd::LongType startHead = sh * ratio;
                        sd::LongType endHead = std::min(startHead + ratio, tHeads);
                        for (sd::LongType th = startHead; th < endHead; th++) {
                            tAvg += teacherAttn->e<double>(b, th, i, j);
                        }
                        tAvg /= (endHead - startHead);

                        double diff = sVal - tAvg;
                        totalLoss += diff * diff;
                        totalElements += 1.0;
                    }
                }
            }
        }
        totalLoss /= totalElements;
    }

    output->p(0, totalLoss);
}

void attentionDistillationLossBp(NDArray* studentAttn, NDArray* teacherAttn,
                                   NDArray* dLdStudent, NDArray* dLdTeacher,
                                   LaunchContext* context) {
    auto batch = studentAttn->sizeAt(0);
    auto sHeads = studentAttn->sizeAt(1);
    auto tHeads = teacherAttn->sizeAt(1);
    auto seqLen = studentAttn->sizeAt(2);

    double zero = 0.0;
    dLdStudent->assign(zero);
    dLdTeacher->assign(zero);

    if (sHeads == tHeads) {
        double totalElements = static_cast<double>(studentAttn->lengthOf());
        double scale = 2.0 / totalElements;
        // dL/dStudent = 2 * (student - teacher) / N
        auto rawDiff = *studentAttn - *teacherAttn;
        auto diff = *rawDiff * scale;
        dLdStudent->assign(diff);
        auto negDiff = *diff * -1.0;
        dLdTeacher->assign(negDiff);
        delete rawDiff;
        delete diff;
        delete negDiff;
    } else {
        sd::LongType ratio = tHeads / sHeads;
        double totalElements = static_cast<double>(batch * sHeads * seqLen * seqLen);
        double scale = 2.0 / totalElements;

        PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(2)
        for (sd::LongType b = 0; b < batch; b++) {
            for (sd::LongType sh = 0; sh < sHeads; sh++) {
                sd::LongType startHead = sh * ratio;
                sd::LongType endHead = std::min(startHead + ratio, tHeads);
                sd::LongType numHeads = endHead - startHead;

                for (sd::LongType i = 0; i < seqLen; i++) {
                    for (sd::LongType j = 0; j < seqLen; j++) {
                        double sVal = studentAttn->e<double>(b, sh, i, j);

                        double tAvg = 0.0;
                        for (sd::LongType th = startHead; th < endHead; th++) {
                            tAvg += teacherAttn->e<double>(b, th, i, j);
                        }
                        tAvg /= numHeads;

                        double diff = (sVal - tAvg) * scale;

                        // dL/dStudent
                        dLdStudent->p(b, sh, i, j, diff);

                        // dL/dTeacher: spread gradient to averaged heads
                        double teacherGrad = -diff / numHeads;
                        for (sd::LongType th = startHead; th < endHead; th++) {
                            double existing = dLdTeacher->e<double>(b, th, i, j);
                            dLdTeacher->p(b, th, i, j, existing + teacherGrad);
                        }
                    }
                }
            }
        }
    }
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd

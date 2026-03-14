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

#include <ops/declarable/helpers/feature_distillation_loss.h>
#include <helpers/MmulHelper.h>

namespace sd {
namespace ops {
namespace helpers {

void featureDistillationLoss(NDArray* studentFeatures, NDArray* teacherFeatures,
                              NDArray* projectionWeight, NDArray* output,
                              LaunchContext* context) {
    auto batch = studentFeatures->sizeAt(0);

    std::vector<sd::LongType> projShape = {batch, teacherFeatures->sizeAt(1)};
    NDArray projected('c', projShape, studentFeatures->dataType());
    NDArray* effectiveStudent;

    if (projectionWeight != nullptr) {
        // projected = student @ projection  [batch, d_teacher]
        MmulHelper::mmul(studentFeatures, projectionWeight, &projected);
        effectiveStudent = &projected;
    } else {
        effectiveStudent = studentFeatures;
    }

    // MSE = mean((student - teacher)^2)
    auto diff = *effectiveStudent - *teacherFeatures;
    auto diffSq = *diff * *diff;

    // Sum all elements and divide by total count
    double totalElements = static_cast<double>(diffSq->lengthOf());
    double mse = diffSq->sumNumber().e<double>(0) / totalElements;

    output->p(0, mse);
    delete diff;
    delete diffSq;
}

void featureDistillationLossBp(NDArray* studentFeatures, NDArray* teacherFeatures,
                                NDArray* projectionWeight,
                                NDArray* dLdStudent, NDArray* dLdTeacher,
                                NDArray* dLdProjection,
                                LaunchContext* context) {
    auto batch = studentFeatures->sizeAt(0);

    std::vector<sd::LongType> projShape2 = {batch, teacherFeatures->sizeAt(1)};
    NDArray projected('c', projShape2, studentFeatures->dataType());
    NDArray* effectiveStudent;

    if (projectionWeight != nullptr) {
        MmulHelper::mmul(studentFeatures, projectionWeight, &projected);
        effectiveStudent = &projected;
    } else {
        effectiveStudent = studentFeatures;
    }

    // dMSE/d(eff_student) = 2 * (eff_student - teacher) / N
    double totalElements = static_cast<double>(effectiveStudent->lengthOf());
    double scale = 2.0 / totalElements;
    auto rawDiff = *effectiveStudent - *teacherFeatures;
    auto diff = *rawDiff * scale;

    // dL/dTeacher = -2 * (eff_student - teacher) / N
    auto negDiff = *diff * -1.0;
    dLdTeacher->assign(negDiff);
    delete negDiff;

    if (projectionWeight != nullptr) {
        // dL/dStudent = dL/dProjected @ projection^T
        NDArray* projT = projectionWeight->transpose();
        MmulHelper::mmul(diff, projT, dLdStudent);
        delete projT;

        // dL/dProjection = student^T @ dL/dProjected
        if (dLdProjection != nullptr) {
            NDArray* studentT = studentFeatures->transpose();
            MmulHelper::mmul(studentT, diff, dLdProjection);
            delete studentT;
        }
    } else {
        dLdStudent->assign(diff);
    }
    delete rawDiff;
    delete diff;
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd

/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include <ops/declarable/helpers/feature_distillation_loss.h>
#include <helpers/MmulHelper.h>
#include <array/NDArray.h>
#include <array/NDArrayFactory.h>

namespace sd {
namespace ops {
namespace helpers {

void featureDistillationLoss(NDArray* studentFeatures, NDArray* teacherFeatures,
                              NDArray* projectionWeight, NDArray* output,
                              LaunchContext* context) {
    auto batch = studentFeatures->sizeAt(0);

    NDArray::prepareSpecialUse({output}, {studentFeatures, teacherFeatures});

    NDArray* effectiveStudent;
    NDArray* projected = nullptr;

    if (projectionWeight != nullptr) {
        NDArray::prepareSpecialUse({}, {projectionWeight});
        std::vector<sd::LongType> projShape = {batch, teacherFeatures->sizeAt(1)};
        projected = NDArrayFactory::create_('c', projShape, studentFeatures->dataType(), context);
        MmulHelper::mmul(studentFeatures, projectionWeight, projected);
        effectiveStudent = projected;
    } else {
        effectiveStudent = studentFeatures;
    }

    // MSE = mean((student - teacher)^2)
    NDArray diff(effectiveStudent->shapeInfo(), false, context);
    effectiveStudent->applyPairwiseTransform(pairwise::Subtract, teacherFeatures, &diff);
    NDArray diffSq(diff.shapeInfo(), false, context);
    diff.applyPairwiseTransform(pairwise::Multiply, &diff, &diffSq);
    double totalElements = static_cast<double>(diffSq.lengthOf());
    NDArray* sum = diffSq.reduceNumber(reduce::SameOps::Sum);
    double mse = sum->e<double>(0) / totalElements;
    delete sum;
    output->p(0, mse);

    NDArray::registerSpecialUse({output}, {studentFeatures, teacherFeatures});
    if (projectionWeight != nullptr) {
        NDArray::registerSpecialUse({}, {projectionWeight});
    }

    delete projected;
}

void featureDistillationLossBp(NDArray* studentFeatures, NDArray* teacherFeatures,
                                NDArray* projectionWeight,
                                NDArray* dLdStudent, NDArray* dLdTeacher,
                                NDArray* dLdProjection,
                                LaunchContext* context) {
    auto batch = studentFeatures->sizeAt(0);

    NDArray::prepareSpecialUse({dLdStudent, dLdTeacher}, {studentFeatures, teacherFeatures});

    NDArray* effectiveStudent;
    NDArray* projected = nullptr;

    if (projectionWeight != nullptr) {
        NDArray::prepareSpecialUse({}, {projectionWeight});
        std::vector<sd::LongType> projShape = {batch, teacherFeatures->sizeAt(1)};
        projected = NDArrayFactory::create_('c', projShape, studentFeatures->dataType(), context);
        MmulHelper::mmul(studentFeatures, projectionWeight, projected);
        effectiveStudent = projected;
    } else {
        effectiveStudent = studentFeatures;
    }

    double totalElements = static_cast<double>(effectiveStudent->lengthOf());
    double scale = 2.0 / totalElements;

    // diff = (effectiveStudent - teacherFeatures) * scale
    NDArray diff(effectiveStudent->shapeInfo(), false, context);
    effectiveStudent->applyPairwiseTransform(pairwise::Subtract, teacherFeatures, &diff);
    NDArray scaledDiff(diff.shapeInfo(), false, context);
    diff.applyScalar<double>(scalar::Multiply, scale, &scaledDiff);

    // dLdTeacher = -scaledDiff
    scaledDiff.applyScalar<double>(scalar::Multiply, -1.0, dLdTeacher);

    if (projectionWeight != nullptr) {
        NDArray* projT = projectionWeight->transpose();
        MmulHelper::mmul(&scaledDiff, projT, dLdStudent);
        delete projT;

        if (dLdProjection != nullptr) {
            NDArray::prepareSpecialUse({dLdProjection}, {});
            NDArray* studentT = studentFeatures->transpose();
            MmulHelper::mmul(studentT, &scaledDiff, dLdProjection);
            delete studentT;
            NDArray::registerSpecialUse({dLdProjection}, {});
        }
    } else {
        dLdStudent->assign(&scaledDiff);
    }

    NDArray::registerSpecialUse({dLdStudent, dLdTeacher}, {studentFeatures, teacherFeatures});
    if (projectionWeight != nullptr) {
        NDArray::registerSpecialUse({}, {projectionWeight});
    }

    delete projected;
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd

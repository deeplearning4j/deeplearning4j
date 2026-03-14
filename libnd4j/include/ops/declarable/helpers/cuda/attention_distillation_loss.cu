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

#include <ops/declarable/helpers/attention_distillation_loss.h>
#include <array/NDArray.h>
#include <array/NDArrayFactory.h>

namespace sd {
namespace ops {
namespace helpers {

void attentionDistillationLoss(NDArray* studentAttn, NDArray* teacherAttn,
                                NDArray* output, LaunchContext* context) {
    auto sHeads = studentAttn->sizeAt(1);
    auto tHeads = teacherAttn->sizeAt(1);

    NDArray::prepareSpecialUse({output}, {studentAttn, teacherAttn});

    if (sHeads == tHeads) {
        NDArray diff(studentAttn->shapeInfo(), false, context);
        studentAttn->applyPairwiseTransform(pairwise::Subtract, teacherAttn, &diff);
        NDArray diffSq(diff.shapeInfo(), false, context);
        diff.applyPairwiseTransform(pairwise::Multiply, &diff, &diffSq);
        double totalElements = static_cast<double>(diffSq.lengthOf());
        NDArray* sum = diffSq.reduceNumber(reduce::SameOps::Sum);
        double mse = sum->e<double>(0) / totalElements;
        delete sum;
        output->p(0, mse);
    } else {
        auto batch = studentAttn->sizeAt(0);
        auto seqLen = studentAttn->sizeAt(2);
        sd::LongType ratio = tHeads / sHeads;

        std::vector<sd::LongType> reshapeShape = {batch, sHeads, ratio, seqLen, seqLen};
        NDArray* teacherReshaped = teacherAttn->reshape('c', reshapeShape);
        std::vector<sd::LongType> axes = {2};
        NDArray* teacherAvg = teacherReshaped->reduceAlongDimension(reduce::FloatOps::Mean, &axes);

        NDArray diff(studentAttn->shapeInfo(), false, context);
        studentAttn->applyPairwiseTransform(pairwise::Subtract, teacherAvg, &diff);
        NDArray diffSq(diff.shapeInfo(), false, context);
        diff.applyPairwiseTransform(pairwise::Multiply, &diff, &diffSq);
        double totalElements = static_cast<double>(diffSq.lengthOf());
        NDArray* sum = diffSq.reduceNumber(reduce::SameOps::Sum);
        double mse = sum->e<double>(0) / totalElements;
        delete sum;
        delete teacherAvg;
        delete teacherReshaped;
        output->p(0, mse);
    }

    NDArray::registerSpecialUse({output}, {studentAttn, teacherAttn});
}

void attentionDistillationLossBp(NDArray* studentAttn, NDArray* teacherAttn,
                                   NDArray* dLdStudent, NDArray* dLdTeacher,
                                   LaunchContext* context) {
    auto sHeads = studentAttn->sizeAt(1);
    auto tHeads = teacherAttn->sizeAt(1);

    NDArray::prepareSpecialUse({dLdStudent, dLdTeacher}, {studentAttn, teacherAttn});

    if (sHeads == tHeads) {
        double totalElements = static_cast<double>(studentAttn->lengthOf());
        double scale = 2.0 / totalElements;

        NDArray diff(studentAttn->shapeInfo(), false, context);
        studentAttn->applyPairwiseTransform(pairwise::Subtract, teacherAttn, &diff);
        diff.applyScalar<double>(scalar::Multiply, scale, dLdStudent);
        dLdStudent->applyScalar<double>(scalar::Multiply, -1.0, dLdTeacher);
    } else {
        auto batch = studentAttn->sizeAt(0);
        auto seqLen = studentAttn->sizeAt(2);
        sd::LongType ratio = tHeads / sHeads;

        std::vector<sd::LongType> reshapeShape = {batch, sHeads, ratio, seqLen, seqLen};
        NDArray* teacherReshaped = teacherAttn->reshape('c', reshapeShape);
        std::vector<sd::LongType> axes = {2};
        NDArray* teacherAvg = teacherReshaped->reduceAlongDimension(reduce::FloatOps::Mean, &axes);

        double totalElements = static_cast<double>(studentAttn->lengthOf());
        double scale = 2.0 / totalElements;

        NDArray diff(studentAttn->shapeInfo(), false, context);
        studentAttn->applyPairwiseTransform(pairwise::Subtract, teacherAvg, &diff);
        diff.applyScalar<double>(scalar::Multiply, scale, dLdStudent);

        // Spread gradient back to teacher heads
        double negScale = -1.0 / static_cast<double>(ratio);
        NDArray teacherGrad(diff.shapeInfo(), false, context);
        diff.applyScalar<double>(scalar::Multiply, negScale, &teacherGrad);

        NDArray* zeroArr = NDArrayFactory::create_<float>(0.0f, context);
        dLdTeacher->assign(zeroArr);
        delete zeroArr;
        for (sd::LongType sh = 0; sh < sHeads; sh++) {
            for (sd::LongType r = 0; r < ratio; r++) {
                sd::LongType th = sh * ratio + r;
                for (sd::LongType b = 0; b < batch; b++) {
                    for (sd::LongType i = 0; i < seqLen; i++) {
                        for (sd::LongType j = 0; j < seqLen; j++) {
                            dLdTeacher->p(b, th, i, j, teacherGrad.e<double>(b, sh, i, j));
                        }
                    }
                }
            }
        }
        delete teacherAvg;
        delete teacherReshaped;
    }

    NDArray::registerSpecialUse({dLdStudent, dLdTeacher}, {studentAttn, teacherAttn});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd

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

package org.nd4j.linalg.api.ops.impl.loss;

import lombok.NoArgsConstructor;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.loss.bp.FeatureDistillationLossBp;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * Feature Distillation Loss for intermediate layer matching.
 *
 * L = MSE(projection(student_hidden), teacher_hidden)
 *
 * Inputs:
 *   0: studentFeatures  [batch, d_student]
 *   1: teacherFeatures  [batch, d_teacher]
 *   2: projectionWeight [d_student, d_teacher] (optional)
 *
 * Output:
 *   0: scalar loss value
 *
 * @author Eclipse Deeplearning4j Contributors
 */
@NoArgsConstructor
public class FeatureDistillationLoss extends DynamicCustomOp {

    public FeatureDistillationLoss(INDArray studentFeatures, INDArray teacherFeatures) {
        super(new INDArray[]{studentFeatures, teacherFeatures}, null);
    }

    public FeatureDistillationLoss(INDArray studentFeatures, INDArray teacherFeatures, INDArray projectionWeight) {
        super(new INDArray[]{studentFeatures, teacherFeatures, projectionWeight}, null);
    }

    public FeatureDistillationLoss(SameDiff sameDiff, SDVariable studentFeatures, SDVariable teacherFeatures) {
        super(null, sameDiff, new SDVariable[]{studentFeatures, teacherFeatures}, false);
    }

    public FeatureDistillationLoss(SameDiff sameDiff, SDVariable studentFeatures, SDVariable teacherFeatures,
                                    SDVariable projectionWeight) {
        super(null, sameDiff, new SDVariable[]{studentFeatures, teacherFeatures, projectionWeight}, false);
    }

    @Override
    public String opName() {
        return "feature_distillation_loss";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> grad) {
        if (args().length > 2) {
            return Arrays.asList(new FeatureDistillationLossBp(sameDiff, arg(0), arg(1), arg(2)).outputVariables());
        }
        return Arrays.asList(new FeatureDistillationLossBp(sameDiff, arg(0), arg(1)).outputVariables());
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes) {
        return Collections.singletonList(inputDataTypes.get(0));
    }

    @Override
    public int getNumOutputs() {
        return 1;
    }
}

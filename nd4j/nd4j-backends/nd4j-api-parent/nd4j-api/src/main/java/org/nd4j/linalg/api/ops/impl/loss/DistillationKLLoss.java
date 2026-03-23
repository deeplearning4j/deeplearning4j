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

import lombok.Getter;
import lombok.NoArgsConstructor;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.loss.bp.DistillationKLLossBp;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * Knowledge Distillation KL Divergence Loss.
 *
 * Computes the combined distillation loss:
 *   L = alpha * T^2 * KL(softmax(student/T) || softmax(teacher/T))
 *       + (1-alpha) * CE(student, hardLabels)
 *
 * Inputs:
 *   0: studentLogits [batch, classes]
 *   1: teacherLogits [batch, classes]
 *   2: hardLabels    [batch] (optional)
 *
 * Float args:
 *   0: temperature (default: 4.0)
 *   1: alpha (default: 0.5)
 *
 * Output:
 *   0: scalar loss value
 *
 * Adam Gibson
 */
@NoArgsConstructor
public class DistillationKLLoss extends DynamicCustomOp {

    @Getter private double temperature = 4.0;
    @Getter private double alpha = 0.5;

    public DistillationKLLoss(INDArray studentLogits, INDArray teacherLogits) {
        this(studentLogits, teacherLogits, 4.0, 0.5);
    }

    public DistillationKLLoss(INDArray studentLogits, INDArray teacherLogits, double temperature, double alpha) {
        super(new INDArray[]{studentLogits, teacherLogits}, null);
        this.temperature = temperature;
        this.alpha = alpha;
        addTArgument(temperature, alpha);
    }

    public DistillationKLLoss(INDArray studentLogits, INDArray teacherLogits, INDArray hardLabels,
                               double temperature, double alpha) {
        super(new INDArray[]{studentLogits, teacherLogits, hardLabels}, null);
        this.temperature = temperature;
        this.alpha = alpha;
        addTArgument(temperature, alpha);
    }

    public DistillationKLLoss(SameDiff sameDiff, SDVariable studentLogits, SDVariable teacherLogits) {
        this(sameDiff, studentLogits, teacherLogits, 4.0, 0.5);
    }

    public DistillationKLLoss(SameDiff sameDiff, SDVariable studentLogits, SDVariable teacherLogits,
                               double temperature, double alpha) {
        super(null, sameDiff, new SDVariable[]{studentLogits, teacherLogits}, false);
        this.temperature = temperature;
        this.alpha = alpha;
        addTArgument(temperature, alpha);
    }

    public DistillationKLLoss(SameDiff sameDiff, SDVariable studentLogits, SDVariable teacherLogits,
                               SDVariable hardLabels, double temperature, double alpha) {
        super(null, sameDiff, new SDVariable[]{studentLogits, teacherLogits, hardLabels}, false);
        this.temperature = temperature;
        this.alpha = alpha;
        addTArgument(temperature, alpha);
    }

    @Override
    public void configureFromArguments() {
        super.configureFromArguments();
        if (tArguments.size() > 0) this.temperature = tArguments.get(0);
        if (tArguments.size() > 1) this.alpha = tArguments.get(1);
    }

    @Override
    public String opName() {
        return "distillation_kl_loss";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> grad) {
        if (args().length > 2) {
            return Arrays.asList(new DistillationKLLossBp(sameDiff, arg(0), arg(1), arg(2),
                temperature, alpha).outputVariables());
        }
        return Arrays.asList(new DistillationKLLossBp(sameDiff, arg(0), arg(1),
            temperature, alpha).outputVariables());
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

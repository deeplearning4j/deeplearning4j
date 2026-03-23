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

package org.nd4j.linalg.api.ops.impl.loss.bp;

import lombok.NoArgsConstructor;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.Arrays;
import java.util.List;

/**
 * Backward pass for Knowledge Distillation KL Divergence Loss.
 *
 * Adam Gibson
 */
@NoArgsConstructor
public class DistillationKLLossBp extends DynamicCustomOp {

    private double temperature = 4.0;
    private double alpha = 0.5;

    public DistillationKLLossBp(SameDiff sameDiff, SDVariable studentLogits, SDVariable teacherLogits,
                                  double temperature, double alpha) {
        super(null, sameDiff, new SDVariable[]{studentLogits, teacherLogits}, false);
        this.temperature = temperature;
        this.alpha = alpha;
        addTArgument(temperature, alpha);
    }

    public DistillationKLLossBp(SameDiff sameDiff, SDVariable studentLogits, SDVariable teacherLogits,
                                  SDVariable hardLabels, double temperature, double alpha) {
        super(null, sameDiff, new SDVariable[]{studentLogits, teacherLogits, hardLabels}, false);
        this.temperature = temperature;
        this.alpha = alpha;
        addTArgument(temperature, alpha);
    }

    @Override
    public String opName() {
        return "distillation_kl_loss_grad";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> grad) {
        throw new UnsupportedOperationException("Differentiation of " + getClass().getName() + " not supported");
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes) {
        return Arrays.asList(inputDataTypes.get(0), inputDataTypes.get(1));
    }

    @Override
    public int getNumOutputs() {
        return 2;
    }
}

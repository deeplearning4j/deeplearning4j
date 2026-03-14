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
import org.nd4j.linalg.api.ops.impl.loss.bp.AttentionDistillationLossBp;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * Attention Distillation Loss for matching student/teacher attention maps.
 *
 * L = MSE(student_attn, teacher_attn)
 * If head counts differ, teacher heads are averaged to match student.
 *
 * Inputs:
 *   0: studentAttn [batch, s_heads, seq, seq]
 *   1: teacherAttn [batch, t_heads, seq, seq]
 *
 * Output:
 *   0: scalar loss value
 *
 * @author Eclipse Deeplearning4j Contributors
 */
@NoArgsConstructor
public class AttentionDistillationLoss extends DynamicCustomOp {

    public AttentionDistillationLoss(INDArray studentAttn, INDArray teacherAttn) {
        super(new INDArray[]{studentAttn, teacherAttn}, null);
    }

    public AttentionDistillationLoss(SameDiff sameDiff, SDVariable studentAttn, SDVariable teacherAttn) {
        super(null, sameDiff, new SDVariable[]{studentAttn, teacherAttn}, false);
    }

    @Override
    public String opName() {
        return "attention_distillation_loss";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> grad) {
        return Arrays.asList(new AttentionDistillationLossBp(sameDiff, arg(0), arg(1)).outputVariables());
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

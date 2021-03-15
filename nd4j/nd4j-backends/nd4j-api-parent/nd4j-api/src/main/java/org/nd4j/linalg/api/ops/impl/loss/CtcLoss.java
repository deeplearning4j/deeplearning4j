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

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.loss.bp.CtcLossBp;

import java.util.List;

public class CtcLoss extends DynamicCustomOp {


    public CtcLoss(INDArray targetLabels, INDArray logitInputs, INDArray targetLabelLengths, INDArray logitInputLengths) {
        super(new INDArray[]{targetLabels,logitInputs,targetLabelLengths,logitInputLengths},null);
    }


    public CtcLoss(SameDiff sameDiff, SDVariable targetLabels,SDVariable logitInputs,SDVariable targetLabelLengths,SDVariable logitInputLengths){
        super(sameDiff,new SDVariable[]{targetLabels,logitInputs,targetLabelLengths,logitInputLengths});
    }


    public CtcLoss(){ }

    @Override
    public String opName() {
        return "ctc_loss";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> grad) {
        //No external gradient
        //Args are: predictions, weights, label
        return new CtcLossBp(sameDiff,  arg(0), arg(1), arg(2),arg(3)).outputs();
    }
}

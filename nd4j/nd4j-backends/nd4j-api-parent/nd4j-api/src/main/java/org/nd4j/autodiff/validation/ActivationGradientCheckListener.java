/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.autodiff.validation;

import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import org.nd4j.autodiff.listeners.At;
import org.nd4j.autodiff.listeners.BaseListener;
import org.nd4j.autodiff.listeners.Operation;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.List;

import org.nd4j.linalg.api.ops.OpContext;
import org.nd4j.linalg.dataset.api.MultiDataSet;

/**
 * A listener used for debugging and testing purposes - specifically for gradient checking activations internally in
 * {@link GradCheckUtil}. It probably isn't useful for anything outside of this.
 *
 * @author Alex Black
 */
@NoArgsConstructor
public class ActivationGradientCheckListener extends BaseListener {

    @Getter @Setter
    private String variableName;
    @Getter @Setter
    private long[] idx;
    @Getter @Setter
    private double eps;

    @Override
    public boolean isActive(Operation operation) {
        return true;
    }

    @Override
    public void opExecution(SameDiff sd, At at, MultiDataSet batch, SameDiffOp op, OpContext opContext, INDArray[] outputs) {
        Preconditions.checkState(variableName != null, "No variable name has been set yet. Variable name must be set before using this listener");
        Preconditions.checkState(eps != 0.0, "Epsilon has not been set");


        List<String> outs = op.getOutputsOfOp();
        int i = 0;
        for(String s : outs){
            if(variableName.equals(s)){
                Preconditions.checkState(idx != null || outputs[i].isScalar(),
                        "No index to modify has been set yet. Index must be set before using this listener");

                double orig = outputs[i].getDouble(idx);
                outputs[i].putScalar(idx, orig + eps);
                return;
            }
            i++;
        }
    }

}

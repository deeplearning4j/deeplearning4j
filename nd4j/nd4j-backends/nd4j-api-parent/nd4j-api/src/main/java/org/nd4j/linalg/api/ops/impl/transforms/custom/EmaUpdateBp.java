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

package org.nd4j.linalg.api.ops.impl.transforms.custom;

import lombok.NoArgsConstructor;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.Arrays;
import java.util.List;

/**
 * Backward pass for EMA Update.
 *
 * Given forward: output = decay * shadow + (1-decay) * model
 *   dL/dModel  = (1 - decay) * gradOutput
 *   dL/dShadow = decay * gradOutput
 *
 * Inputs:
 *   0: model
 *   1: shadow
 *   2: gradOutput (upstream gradient)
 *
 * Float args:
 *   0: decay
 *
 * Outputs:
 *   0: dL/dModel
 *   1: dL/dShadow
 *
 * @author Eclipse Deeplearning4j Contributors
 */
@NoArgsConstructor
public class EmaUpdateBp extends DynamicCustomOp {

    private double decay = 0.999;

    public EmaUpdateBp(SameDiff sameDiff, SDVariable model, SDVariable shadow,
                        SDVariable gradOutput, double decay) {
        super(null, sameDiff, new SDVariable[]{model, shadow, gradOutput}, false);
        this.decay = decay;
        addTArgument(decay);
    }

    @Override
    public String opName() {
        return "ema_update_bp";
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

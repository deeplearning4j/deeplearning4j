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

import lombok.Getter;
import lombok.NoArgsConstructor;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * Exponential Moving Average parameter update for DINOv2 teacher networks.
 *
 * output = decay * shadow + (1 - decay) * model
 *
 * Inputs:
 *   0: model parameters (student)
 *   1: shadow parameters (teacher/EMA)
 *
 * Float args:
 *   0: decay factor (default: 0.999)
 *
 * Output:
 *   0: updated shadow parameters
 *
 * Adam Gibson
 */
@NoArgsConstructor
public class EmaUpdate extends DynamicCustomOp {

    @Getter private double decay = 0.999;

    public EmaUpdate(INDArray model, INDArray shadow) {
        this(model, shadow, 0.999);
    }

    public EmaUpdate(INDArray model, INDArray shadow, double decay) {
        super(new INDArray[]{model, shadow}, null);
        this.decay = decay;
        addTArgument(decay);
    }

    public EmaUpdate(SameDiff sameDiff, SDVariable model, SDVariable shadow) {
        this(sameDiff, model, shadow, 0.999);
    }

    public EmaUpdate(SameDiff sameDiff, SDVariable model, SDVariable shadow, double decay) {
        super(null, sameDiff, new SDVariable[]{model, shadow}, false);
        this.decay = decay;
        addTArgument(decay);
    }

    @Override
    public void configureFromArguments() {
        super.configureFromArguments();
        if (tArguments.size() > 0) this.decay = tArguments.get(0);
    }

    @Override
    public String opName() {
        return "ema_update";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> grad) {
        return Arrays.asList(new EmaUpdateBp(sameDiff, arg(0), arg(1), grad.get(0), decay).outputVariables());
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

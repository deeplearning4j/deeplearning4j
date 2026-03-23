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
 * DINOv2 Centering and Sharpening operation.
 *
 * Prevents mode collapse in self-supervised learning:
 *   output = softmax((input - center) / temperature)
 *
 * Inputs:
 *   0: input logits [batch, features] - teacher output
 *   1: center vector [features] - running mean of teacher outputs
 *
 * Float args:
 *   0: temperature (default: 0.07)
 *
 * Output:
 *   0: sharpened probabilities [batch, features]
 *
 * Adam Gibson
 */
@NoArgsConstructor
public class CenterAndSharpen extends DynamicCustomOp {

    @Getter private double temperature = 0.07;

    public CenterAndSharpen(INDArray input, INDArray center) {
        this(input, center, 0.07);
    }

    public CenterAndSharpen(INDArray input, INDArray center, double temperature) {
        super(new INDArray[]{input, center}, null);
        this.temperature = temperature;
        addTArgument(temperature);
    }

    public CenterAndSharpen(SameDiff sameDiff, SDVariable input, SDVariable center) {
        this(sameDiff, input, center, 0.07);
    }

    public CenterAndSharpen(SameDiff sameDiff, SDVariable input, SDVariable center, double temperature) {
        super(null, sameDiff, new SDVariable[]{input, center}, false);
        this.temperature = temperature;
        addTArgument(temperature);
    }

    @Override
    public void configureFromArguments() {
        super.configureFromArguments();
        if (tArguments.size() > 0) this.temperature = tArguments.get(0);
    }

    @Override
    public String opName() {
        return "center_and_sharpen";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> grad) {
        return Arrays.asList(new CenterAndSharpenBp(sameDiff, arg(0), arg(1), grad.get(0), temperature).outputVariables());
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

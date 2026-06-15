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
 * Backward pass for DINOv2 Center and Sharpen.
 *
 * Given forward: y = softmax((x - c) / t)
 *   dL/dx = softmax_jacobian(y) * gradOutput / t
 *   dL/dc = -sum(dL/dx, axis=0)
 *
 * Inputs:
 *   0: input [batch, features]
 *   1: center [features]
 *   2: gradOutput (upstream gradient) [batch, features]
 *
 * Float args:
 *   0: temperature
 *
 * Outputs:
 *   0: dL/dInput [batch, features]
 *   1: dL/dCenter [features]
 *
 * Adam Gibson
 */
@NoArgsConstructor
public class CenterAndSharpenBp extends DynamicCustomOp {

    private double temperature = 0.07;

    public CenterAndSharpenBp(SameDiff sameDiff, SDVariable input, SDVariable center,
                               SDVariable gradOutput, double temperature) {
        super(null, sameDiff, new SDVariable[]{input, center, gradOutput}, false);
        this.temperature = temperature;
        addTArgument(temperature);
    }

    @Override
    public String opName() {
        return "center_and_sharpen_bp";
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

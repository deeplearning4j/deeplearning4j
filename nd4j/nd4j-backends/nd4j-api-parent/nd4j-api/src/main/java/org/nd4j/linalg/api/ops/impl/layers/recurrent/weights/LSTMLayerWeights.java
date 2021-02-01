/*
 *  ******************************************************************************
 *  *
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
package org.nd4j.linalg.api.ops.impl.layers.recurrent.weights;


import lombok.Builder;
import lombok.Data;
import lombok.EqualsAndHashCode;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.LSTMLayer;
import org.nd4j.common.util.ArrayUtil;

/**
 * The weight configuration of a LSTMLayer.  For {@link LSTMLayer}
 * @author Alex Black
 */
@EqualsAndHashCode(callSuper = true)
@Data
@Builder
public class LSTMLayerWeights extends RNNWeights {

    /**
     * Input to hidden weights with a shape of [inSize, 4*numUnits].
     *
     * Input to hidden and hidden to hidden are concatenated in dimension 0,
     * so the input to hidden weights are [:inSize, :] and the hidden to hidden weights are [inSize:, :].
     */
    private SDVariable weights;
    private INDArray iWeights;

    /**
     * hidden to hidden weights (aka "recurrent weights", with a shape of [numUnits, 4*numUnits].
     *
     */
    private SDVariable rWeights;
    private INDArray irWeights;

    /**
     * Peephole weights, with a shape of [3*numUnits].
     */
    private SDVariable peepholeWeights;
    private INDArray iPeepholeWeights;

    /**
     * Input to hidden and hidden to hidden biases, with shape [4*numUnits].
     */
    private SDVariable bias;
    private INDArray iBias;

    @Override
    public SDVariable[] args() {
        return filterNonNull(weights, rWeights, peepholeWeights, bias);
    }

    @Override
    public INDArray[] arrayArgs() {
        return filterNonNull(iWeights, irWeights, iPeepholeWeights, iBias);
    }

    @Override
    public SDVariable[] argsWithInputs(SDVariable... inputs){
        Preconditions.checkArgument(inputs.length == 4, "Expected 4 inputs, got %s", inputs.length);   //Order: x, seqLen, yLast, cLast
        //lstmLayer c++ op expects: x, Wx, Wr, Wp, b, seqLen, yLast, cLast
        return ArrayUtil.filterNull(inputs[0], weights, rWeights, bias, inputs[1], inputs[2], inputs[3], peepholeWeights);
    }

    @Override
    public INDArray[] argsWithInputs(INDArray... inputs) {
        Preconditions.checkArgument(inputs.length == 4, "Expected 4 inputs, got %s", inputs.length);   //Order: x, seqLen, yLast, cLast
        //lstmLayer c++ op expects: x, Wx, Wr, Wp, b, seqLen, yLast, cLast
        return ArrayUtil.filterNull(inputs[0], iWeights, irWeights, iBias, inputs[1], inputs[2], inputs[3], iPeepholeWeights);
    }


    public boolean hasBias() {
        return (bias!=null||iBias!=null);
    }

    public boolean hasPH() {
        return (peepholeWeights!=null||iPeepholeWeights!=null);
    }

}

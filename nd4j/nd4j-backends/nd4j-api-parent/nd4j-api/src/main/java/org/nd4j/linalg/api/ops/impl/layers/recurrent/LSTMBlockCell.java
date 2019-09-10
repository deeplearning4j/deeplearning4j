/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.nd4j.linalg.api.ops.impl.layers.recurrent;

import lombok.Getter;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.config.LSTMConfiguration;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.weights.LSTMWeights;
import org.nd4j.linalg.primitives.Pair;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.Arrays;
import java.util.List;
import java.util.Map;

/**
 * LSTM Block cell - represents forward pass for a single time step of an LSTM RNN.<br>
 * Same operation used internally in op/layer {@link LSTMLayer}.<br>
 * Implementation of operation for LSTM layer with optional peep hole connections.<br>
 * S. Hochreiter and J. Schmidhuber. "Long Short-Term Memory". Neural Computation and <a href="https://research.google.com/pubs/archive/43905.pdf">https://research.google.com/pubs/archive/43905.pdf</a><br>
 * Hasim Sak, Andrew Senior, and Francoise Beaufays. "Long short-term memory recurrent neural network architectures for large scale acoustic modeling." INTERSPEECH, 2014.<br>
 * See also: <a href="https://arxiv.org/pdf/1503.04069.pdf">https://arxiv.org/pdf/1503.04069.pdf</a><br>
 * <br>
 * See also {@link LSTMLayer} for 'full time series" op - this lstmBlockCell op is used internally at C++ level in LSTMLayer.<br>
 * Input arrays: <br>
 * 0: input [bS, inSize] at time t<br>
 * 1: previous cell state  [bS, numUnits], time t-1<br>
 * 2: previous output [bS, numUnits], time t-1<br>
 * 3: Weights - concatenated (input-to-hidden, hidden-to-hidden weights)  weights, [(inSize+numUnits), 4*numUnits]<br>
 * 4: weights - cell peephole (t-1) connections to input modulation gate, [numUnits]<br>
 * 5: weights - cell peephole (t-1) connections to forget gate, [numUnits]<br>
 * 6: weights - cell peephole (t) connections to output gate, [numUnits]<br>
 * 7: biases, shape [4*numUnits]<br>
 * <br>
 * Weights are set via {@link LSTMWeights}.<br>
 * <br>
 * Input integer arguments: set via {@link LSTMConfiguration}<br>
 * 0: if not zero, provide peephole connections<br>
 * <br>
 * Input float arguments: set via {@link LSTMConfiguration}<br>
 * 0: the bias added to forget gates in order to reduce the scale of forgetting in the beginning of the training<br>
 * 1: clipping value for cell state, if it is not equal to zero, then cell state is clipped<br>
 * <br>
 * Output arrays:<br>
 * 0: i      - Input modulation gate activations [bS, numUnits]<br>
 * 1: c (cs) - Cell state (pre tanh) [bs, numUnits] (cs)<br>
 * 2: f      - Output - forget gate activations [bs, numUnits]<br>
 * 3: o      - Output - output gate activations [bs, numUnits]<br>
 * 4: z (ci) - Output - block input [bs, numUnits]<br>
 * 5: h (co) - Cell state, post tanh [bs, numUnits]<br>
 * 6: y (h)  - Current cell output [bS, numUnits], time t<br>
 *
 * @author Alex Black
 */
public class LSTMBlockCell extends DynamicCustomOp {

    private LSTMConfiguration configuration;

    @Getter
    private LSTMWeights weights;

    public LSTMBlockCell() {
    }

    public LSTMBlockCell(SameDiff sameDiff, SDVariable x, SDVariable cLast, SDVariable yLast, LSTMWeights weights, LSTMConfiguration configuration) {
        super(null, sameDiff, weights.argsWithInputs(x, cLast, yLast));
        this.configuration = configuration;
        this.weights = weights;
        addIArgument(configuration.iArgs(false));
        addTArgument(configuration.tArgs());
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes) {
        Preconditions.checkState(inputDataTypes != null && inputDataTypes.size() == 8, "Expected exactly 8 inputs to LSTMBlockCell, got %s", inputDataTypes);
        //7 outputs, all of same type as input
        DataType dt = inputDataTypes.get(0);
        Preconditions.checkState(dt.isFPType(), "Input type 0 must be a floating point type, got %s", dt);
        return Arrays.asList(dt, dt, dt, dt, dt, dt, dt);
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> grads) {
        throw new UnsupportedOperationException("Not yet implemented");
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        configuration = LSTMConfiguration.builder()
                .forgetBias(attributesForNode.get("forget_bias").getF())
                .clippingCellValue(attributesForNode.get("cell_clip").getF())
                .peepHole(attributesForNode.get("use_peephole").getB())
                .build();
        addIArgument(configuration.iArgs(false));
        addTArgument(configuration.tArgs());
    }

    @Override
    public String opName() {
        return "lstmBlockCell";
    }

    @Override
    public Map<String, Object> propertiesForFunction() {
        return configuration.toProperties(false);
    }

    @Override
    public String tensorflowName() {
        return "LSTMBlockCell";
    }

}

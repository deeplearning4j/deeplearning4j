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
import lombok.NonNull;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.config.LSTMConfiguration;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.config.RnnDataFormat;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.weights.LSTMWeights;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.Arrays;
import java.util.List;
import java.util.Map;

/**
 * LSTM layer implemented as a single operation.
 * Implementation of operation for LSTM layer with optional peep hole connections.<br>
 * S. Hochreiter and J. Schmidhuber. "Long Short-Term Memory". Neural Computation and <a href="https://research.google.com/pubs/archive/43905.pdf">https://research.google.com/pubs/archive/43905.pdf</a><br>
 * Hasim Sak, Andrew Senior, and Francoise Beaufays. "Long short-term memory recurrent neural network architectures for large scale acoustic modeling." INTERSPEECH, 2014.<br>
 * See also: <a href="https://arxiv.org/pdf/1503.04069.pdf">https://arxiv.org/pdf/1503.04069.pdf</a><br>
 * <p>
 * See also {@link LSTMBlockCell} - lstmBlockCell op is used internally at C++ level for computation.<br>
 * <br>
 * Input arrays:<br>
 * 0: max sequence length; long/int64 scalar<br>
 * 1: input [seqLength, bS, inSize] at time t<br>
 * 2: previous/initial cell state  [bS, numUnits]<br>
 * 3: previous/initial output [bS, numUnits]<br>
 * 4: Weights - concatenated (input-to-hidden, hidden-to-hidden weights)  weights, [(inSize+numUnits), 4*numUnits]<br>
 * 5: weights - cell peephole (t-1) connections to input modulation gate, [numUnits]<br>
 * 6: weights - cell peephole (t-1) connections to forget gate, [numUnits]<br>
 * 7: weights - cell peephole (t) connections to output gate, [numUnits]<br>
 * 8: biases, shape [4*numUnits]<br>
 * <br>
 * Input integer arguments: set via {@link LSTMConfiguration}<br>
 * 0: if not zero, provide peephole connections<br>
 * 1: Data format - 0=TNS=[seqLen,mb,size]; 1=NST=[mb,size,seqLen]; 2=NTS=[mb,seqLen,size]<br>
 * <br>
 * Input float arguments: set via {@link LSTMConfiguration}<br>
 * 0: the bias added to forget gates in order to reduce the scale of forgetting in the beginning of the training<br>
 * 1: clipping value for cell state, if it is not equal to zero, then cell state is clipped<br>
 * <p>
 * Output arrays:<br>
 * 0: i      - Input modulation gate activations, rank 3, shape as per dataFormat<br>
 * 1: c (cs) - Cell state (pre tanh), rank 3, shape as per dataFormat<br>
 * 2: f      - Output - forget gate activations, rank 3, shape as per dataFormat<br>
 * 3: o      - Output - output gate activations, rank 3, shape as per dataFormat<br>
 * 4: z (ci) - Output - block input, rank 3, shape as per dataFormat<br>
 * 5: h (co) - Cell state, post tanh, rank 3, shape as per dataFormat<br>
 * 6: y (h)  - Current cell output, rank 3, shape as per dataFormat<br>
 *
 * @author Alex Black
 */
public class LSTMLayer extends DynamicCustomOp {

    private LSTMConfiguration configuration;

    @Getter
    private LSTMWeights weights;

    public LSTMLayer() {
    }

    public LSTMLayer(@NonNull SameDiff sameDiff, SDVariable maxTSLength, SDVariable x, SDVariable cLast, SDVariable yLast, LSTMWeights weights, LSTMConfiguration configuration) {
        super(null, sameDiff, weights.argsWithInputs(maxTSLength, x, cLast, yLast));
        this.configuration = configuration;
        this.weights = weights;
        addIArgument(configuration.iArgs(true));
        addTArgument(configuration.tArgs());
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes) {
        Preconditions.checkState(inputDataTypes != null && inputDataTypes.size() == 9, "Expected exactly 9 inputs to LSTMLayer, got %s", inputDataTypes);
        //7 outputs, all of same type as input. Note that input 0 is max sequence length (int64), input 1 is actual input
        DataType dt = inputDataTypes.get(1);
        Preconditions.checkState(dt.isFPType(), "Input type 1 must be a floating point type, got %s", dt);
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
                .dataFormat(RnnDataFormat.TNS)  //Always time major for TF BlockLSTM
                .build();
        addIArgument(configuration.iArgs(true));
        addTArgument(configuration.tArgs());
    }

    @Override
    public String opName() {
        return "lstmBlock";
    }

    @Override
    public Map<String, Object> propertiesForFunction() {
        return configuration.toProperties(true);
    }

    @Override
    public String tensorflowName() {
        return "BlockLSTM";
    }

}

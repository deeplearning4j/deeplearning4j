/* ******************************************************************************
 * Copyright (c) 2020 Konduit K.K.
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
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.config.LSTMLayerConfig;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.config.LSTMLayerConfig;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.weights.LSTMLayerWeights;
import sun.security.util.ArrayUtil;

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
 * Input integer arguments: set via {@link LSTMLayerConfig}<br>
 * 0: if not zero, provide peephole connections<br>
 * 1: Data format - 0=TNS=[seqLen,mb,size]; 1=NST=[mb,size,seqLen]; 2=NTS=[mb,seqLen,size]<br>
 * <br>
 * Input float arguments: set via {@link LSTMLayerConfig}<br>
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

    private LSTMLayerConfig configuration;

    @Getter
    private LSTMLayerWeights weights;


    public LSTMLayer() {
    }

    public LSTMLayer(@NonNull SameDiff sameDiff, SDVariable maxTSLength, SDVariable x, SDVariable cLast, SDVariable yLast, LSTMLayerWeights weights, LSTMLayerConfig configuration) {
        super(null, sameDiff, weights.argsWithInputs(x, maxTSLength, cLast, yLast));
        this.configuration = configuration;
        this.weights = weights;
        addIArgument(iArgs());
        addTArgument(tArgs(configuration));
        addBArgument(bArgs(weights, maxTSLength, yLast, cLast, configuration));


    }

    public LSTMLayer(INDArray x, INDArray cLast, INDArray yLast, INDArray maxTSLength, LSTMLayerWeights lstmWeights, LSTMLayerConfig LSTMLayerConfig) {
        super(null, null, lstmWeights.argsWithInputs(maxTSLength, x, cLast, yLast));
        this.configuration = LSTMLayerConfig;
        this.weights = lstmWeights;
        addIArgument(iArgs());
        addTArgument(tArgs(configuration));

        addBArgument(bArgs(weights, maxTSLength, yLast, cLast, configuration));
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes) {
        Preconditions.checkState(inputDataTypes != null && inputDataTypes.size() == 8, "Expected exactly 8 inputs to LSTMLayer, got %s", inputDataTypes);
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
    public String opName() {
        return "lstmLayer";
    }

    @Override
    public Map<String, Object> propertiesForFunction() {
        return configuration.toProperties(true, true);
    }


    public long[] iArgs(LSTMLayerConfig configuration) {
        return new long[]{
                (long) configuration.toProperties(true, true).get("LSTMDataFormat"),  // INT_ARG(0)
                (long) configuration.toProperties(true, true).get("LSTMDirectionMode") , // INT_ARG(1)
                (long) configuration.toProperties(true, true).get("gateAct"),  // INT_ARG(2)
                (long) configuration.toProperties(true, true).get("outAct") , // INT_ARG(3)
                (long) configuration.toProperties(true, true).get("cellAct")  // INT_ARG(4)

        };
    }

    public double[] tArgs(LSTMLayerConfig configuration) {
        return new double[]{(double) configuration.toProperties(true, true).get("cellClip")}; // T_ARG(0)
    }


    public boolean[] bArgs(LSTMLayerWeights weights, INDArray maxTSLength, INDArray yLast, INDArray cLast, LSTMLayerConfig configuration) {
        return new boolean[]{
                weights.hasBias(),         // hasBiases: B_ARG(0)
                maxTSLength != null,         // hasSeqLen: B_ARG(1)
                yLast != null,               // hasInitH: B_ARG(2)
                cLast != null,              // hasInitC: B_ARG(3)
                weights.hasPH(),          // hasPH: B_ARG(4)
                (boolean) configuration.toProperties(true, true).get("retFullSequence"), //retFullSequence: B_ARG(5)
                (boolean) configuration.toProperties(true, true).get("retLastH"),  //  retLastH: B_ARG(6)
                (boolean) configuration.toProperties(true, true).get("retLastC")   // retLastC: B_ARG(7)
        };

    }

    public boolean[] bArgs(LSTMLayerWeights weights, SDVariable maxTSLength, SDVariable yLast, SDVariable cLast, LSTMLayerConfig configuration) {
        return new boolean[]{
                weights.hasBias(),         // hasBiases: B_ARG(0)
                maxTSLength != null,         // hasSeqLen: B_ARG(1)
                yLast != null,               // hasInitH: B_ARG(2)
                cLast != null,              // hasInitC: B_ARG(3)
                weights.hasPH(),          // hasPH: B_ARG(4)
                (boolean) configuration.toProperties(true, true).get("retFullSequence"), //retFullSequence: B_ARG(5)
                (boolean) configuration.toProperties(true, true).get("retLastH"),  //  retLastH: B_ARG(6)
                (boolean) configuration.toProperties(true, true).get("retLastC")   // retLastC: B_ARG(7)
        };

    }


}



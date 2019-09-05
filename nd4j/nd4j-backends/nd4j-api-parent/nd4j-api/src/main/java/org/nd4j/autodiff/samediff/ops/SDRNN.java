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

package org.nd4j.autodiff.samediff.ops;

import lombok.NonNull;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.*;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.config.*;

import java.util.Arrays;
import java.util.List;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.outputs.GRUCellOutputs;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.outputs.LSTMCellOutputs;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.outputs.LSTMLayerOutputs;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.outputs.SRUCellOutputs;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.outputs.SRULayerOutputs;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.weights.GRUWeights;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.weights.LSTMWeights;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.weights.SRUWeights;
import org.nd4j.linalg.primitives.Pair;

/**
 * SameDiff Recurrent Neural Network operations<br>
 * Accessible via {@link SameDiff#rnn()}<br>
 * See also {@link SDNN} (accessible via {@link SameDiff#nn()} for general neural network ops.<br>
 * See also {@link SDCNN} (accessible via {@link SameDiff#cnn()} for convolutional neural network ops.<br>
 *
 * @author Alex Black
 */
public class SDRNN extends SDOps {
    public SDRNN(SameDiff sameDiff) {
        super(sameDiff);
    }


    /**
     * See {@link #gru(String, SDVariable, SDVariable, GRUWeights)}.
     */
    public GRUCellOutputs gru(@NonNull SDVariable x, @NonNull SDVariable hLast, @NonNull GRUWeights weights) {
        GRUCell c = new GRUCell(sd, x, hLast, weights);
        return new GRUCellOutputs(c.outputVariables());
    }

    /**
     * The GRU cell.  Does a single time step operation.
     *
     * @param baseName  The base name for the gru cell
     * @param x         Input, with shape [batchSize, inSize]
     * @param hLast     Output of the previous cell/time step, with shape [batchSize, numUnits]
     * @param weights   The cell's weights.
     * @return          The cell's outputs.
     */
    public GRUCellOutputs gru(String baseName, @NonNull SDVariable x, @NonNull SDVariable hLast, @NonNull GRUWeights weights) {
        GRUCell c = new GRUCell(sd, x, hLast, weights);
        return new GRUCellOutputs(c.outputVariables(baseName));
    }

    /**
     * See {@link #lstmCell(String, SDVariable, SDVariable, SDVariable, LSTMWeights, LSTMConfiguration)}.
     */
    public LSTMCellOutputs lstmCell(@NonNull SDVariable x, @NonNull SDVariable cLast, @NonNull SDVariable yLast,
            LSTMWeights weights, LSTMConfiguration config){
        LSTMBlockCell c = new LSTMBlockCell(sd, x, cLast, yLast, weights, config);
        return new LSTMCellOutputs(c.outputVariables());
    }

    /**
     * The LSTM cell.  Does a single time step operation.
     *
     * @param baseName  The base name for the lstm cell
     * @param x         Input, with shape [batchSize, inSize]
     * @param cLast     Previous cell state, with shape [batchSize, numUnits]
     * @param yLast     Previous cell output, with shape [batchSize, numUnits]
     * @param weights   The cell's weights.
     * @param config    The cell's config.
     * @return          The cell's outputs.
     */
    public LSTMCellOutputs lstmCell(String baseName, @NonNull SDVariable x, @NonNull SDVariable cLast, @NonNull SDVariable yLast,
            @NonNull LSTMWeights weights, @NonNull LSTMConfiguration config){
        LSTMBlockCell c = new LSTMBlockCell(sd, x, cLast, yLast, weights, config);
        return new LSTMCellOutputs(c.outputVariables(baseName));
    }

    /**
     * See {@link #lstmLayer(String, SDVariable, SDVariable, SDVariable, SDVariable, LSTMWeights, LSTMConfiguration)}
     */
    public LSTMLayerOutputs lstmLayer(@NonNull SDVariable maxTSLength,
            @NonNull SDVariable x, @NonNull SDVariable cLast, @NonNull SDVariable yLast,
            @NonNull  LSTMWeights weights, @NonNull LSTMConfiguration config){
        LSTMLayer c = new LSTMLayer(sd, maxTSLength, x, cLast, yLast, weights, config);
        return new LSTMLayerOutputs(c.outputVariables(), config.getDataFormat());
    }

    /**
     * See {@link #lstmLayer(String, SDVariable, SDVariable, SDVariable, SDVariable, LSTMWeights, LSTMConfiguration)}
     */
    public LSTMLayerOutputs lstmLayer(int maxTSLength, @NonNull SDVariable x, @NonNull SDVariable cLast, @NonNull SDVariable yLast,
            @NonNull LSTMWeights weights, @NonNull LSTMConfiguration config){
        return lstmLayer(
                sd.scalar("lstm_max_ts_length", maxTSLength),
                x, cLast, yLast, weights, config);
    }

    /**
     * See {@link #lstmLayer(String, SDVariable, SDVariable, SDVariable, SDVariable, LSTMWeights, LSTMConfiguration)}
     */
    public LSTMLayerOutputs lstmLayer(String baseName, int maxTSLength, @NonNull SDVariable x, @NonNull SDVariable cLast, @NonNull SDVariable yLast,
            @NonNull LSTMWeights weights, @NonNull LSTMConfiguration config){
        if(baseName != null) {
            return lstmLayer(baseName,
                    sd.scalar(sd.generateDistinctCustomVariableName(baseName + "_max_ts_length"), maxTSLength),
                    x, cLast, yLast, weights, config);
        } else {
            return lstmLayer(maxTSLength, x, cLast, yLast, weights, config);
        }
    }

    /**
     * The LSTM layer.  Does multiple time steps.
     *
     * Input shape depends on data format (in config):<br>
     * TNS -> [timeSteps, batchSize, inSize]<br>
     * NST -> [batchSize, inSize, timeSteps]<br>
     * NTS -> [batchSize, timeSteps, inSize]<br>
     *
     * @param baseName  The base name for the lstm layer
     * @param x         Input, with shape dependent on the data format (in config).
     * @param cLast     Previous/initial cell state, with shape [batchSize, numUnits]
     * @param yLast     Previous/initial cell output, with shape [batchSize, numUnits]
     * @param weights   The layer's weights.
     * @param config    The layer's config.
     * @return          The layer's outputs.
     */
    public LSTMLayerOutputs lstmLayer(String baseName, @NonNull SDVariable maxTSLength,
            @NonNull SDVariable x, @NonNull SDVariable cLast, @NonNull SDVariable yLast,
            @NonNull LSTMWeights weights, @NonNull LSTMConfiguration config){
        LSTMLayer c = new LSTMLayer(sd, maxTSLength, x, cLast, yLast, weights, config);
        return new LSTMLayerOutputs(c.outputVariables(baseName), config.getDataFormat());
    }

    /**
     * See {@link #sruCell(String, SDVariable, SDVariable, SRUWeights)}.
     */
    public SRUCellOutputs sruCell(@NonNull SDVariable x, @NonNull SDVariable cLast, @NonNull SRUWeights weights) {
        return new SRUCellOutputs(new SRUCell(sd, x, cLast, weights).outputVariables());
    }

    /**
     * The SRU cell.  Does a single time step operation.
     *
     * @param baseName  The base name for the sru cell
     * @param x         Input, with shape [batchSize, inSize]
     * @param cLast     Previous cell state, with shape [batchSize, inSize]
     * @param weights   The cell's weights.
     * @return          The cell's outputs.
     */
    public SRUCellOutputs sruCell(String baseName, @NonNull SDVariable x, @NonNull SDVariable cLast, @NonNull SRUWeights weights) {
        return new SRUCellOutputs(new SRUCell(sd, x, cLast, weights).outputVariables(baseName));
    }

    /**
     * See {@link #sru(String, SDVariable, SDVariable, SDVariable, SRUWeights)}
     */
    public SRULayerOutputs sru(@NonNull SDVariable x, @NonNull SDVariable initialC, @NonNull SRUWeights weights) {
        return sru(x, initialC, null, weights);
    }

    /**
     * See {@link #sru(String, SDVariable, SDVariable, SDVariable, SRUWeights)}
     */
    public SRULayerOutputs sru(String baseName, @NonNull SDVariable x, @NonNull SDVariable initialC, @NonNull SRUWeights weights) {
        return sru(baseName, x, initialC, null, weights);
    }

    /**
     * See {@link #sru(String, SDVariable, SDVariable, SDVariable, SRUWeights)}
     */
    public SRULayerOutputs sru(@NonNull SDVariable x, @NonNull SDVariable initialC, SDVariable mask, @NonNull SRUWeights weights) {
        return new SRULayerOutputs(new SRU(sd, x, initialC, mask, weights).outputVariables());
    }

    /**
     * The SRU layer.  Does a single time step operation.
     *
     * @param baseName  The base name for the sru layer
     * @param x         Input, with shape [batchSize, inSize, timeSeriesLength]
     * @param initialC  Initial cell state, with shape [batchSize, inSize]
     * @param mask      An optional dropout mask, with shape [batchSize, inSize]
     * @param weights   The layer's weights.
     * @return          The layer's outputs.
     */
    public SRULayerOutputs sru(String baseName, @NonNull SDVariable x, @NonNull SDVariable initialC, SDVariable mask, @NonNull SRUWeights weights) {
        return new SRULayerOutputs(new SRU(sd, x, initialC, mask, weights).outputVariables(baseName));
    }

}

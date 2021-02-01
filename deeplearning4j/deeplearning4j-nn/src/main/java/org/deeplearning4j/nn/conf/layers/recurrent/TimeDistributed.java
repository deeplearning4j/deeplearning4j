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

package org.deeplearning4j.nn.conf.layers.recurrent;

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NonNull;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.RNNFormat;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.layers.wrapper.BaseWrapperLayer;
import org.deeplearning4j.nn.layers.recurrent.TimeDistributedLayer;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.util.Collection;

/**
 * TimeDistributed wrapper layer.<br>
 * Note: only the "Feed forward layer time distributed in an RNN" is currently supported.
 * For example, a time distributed dense layer.<br>
 * Usage: {@code .layer(new TimeDistributed(new DenseLayer.Builder()....build(), timeAxis))}<br>
 * Note that for DL4J RNNs, time axis is always 2 - i.e., RNN activations have shape [minibatch, size, sequenceLength]
 *
 * @author Alex Black
 */
@Data
@EqualsAndHashCode(callSuper = true)
public class TimeDistributed extends BaseWrapperLayer {

    private RNNFormat rnnDataFormat = RNNFormat.NCW;

    /**
     * @param underlying Underlying (internal) layer - should be a feed forward type such as DenseLayer
     */
    public TimeDistributed(@JsonProperty("underlying") @NonNull Layer underlying, @JsonProperty("rnnDataFormat") RNNFormat rnnDataFormat) {
        super(underlying);
        this.rnnDataFormat = rnnDataFormat;
    }

    public TimeDistributed(Layer underlying){
        super(underlying);
    }

    @Override
    public org.deeplearning4j.nn.api.Layer instantiate(NeuralNetConfiguration conf, Collection<TrainingListener> trainingListeners,
                                                       int layerIndex, INDArray layerParamsView, boolean initializeParams, DataType networkDataType) {
        NeuralNetConfiguration conf2 = conf.clone();
        conf2.setLayer(((TimeDistributed) conf2.getLayer()).getUnderlying());
        return new TimeDistributedLayer(underlying.instantiate(conf2, trainingListeners, layerIndex, layerParamsView,
                initializeParams, networkDataType), rnnDataFormat);
    }

    @Override
    public InputType getOutputType(int layerIndex, InputType inputType) {
        if (inputType.getType() != InputType.Type.RNN) {
            throw new IllegalStateException("Only RNN input type is supported as input to TimeDistributed layer (layer #" + layerIndex + ")");
        }

        InputType.InputTypeRecurrent rnn = (InputType.InputTypeRecurrent) inputType;
        InputType ff = InputType.feedForward(rnn.getSize());
        InputType ffOut = underlying.getOutputType(layerIndex, ff);
        return InputType.recurrent(ffOut.arrayElementsPerExample(), rnn.getTimeSeriesLength(), rnnDataFormat);
    }

    @Override
    public void setNIn(InputType inputType, boolean override) {
        if (inputType.getType() != InputType.Type.RNN) {
            throw new IllegalStateException("Only RNN input type is supported as input to TimeDistributed layer");
        }

        InputType.InputTypeRecurrent rnn = (InputType.InputTypeRecurrent) inputType;
        InputType ff = InputType.feedForward(rnn.getSize());
        this.rnnDataFormat = rnn.getFormat();
        underlying.setNIn(ff, override);
    }

    @Override
    public InputPreProcessor getPreProcessorForInputType(InputType inputType) {
        //No preprocessor - the wrapper layer operates as the preprocessor
        return null;
    }
}

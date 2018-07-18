/*
 * Copyright (c) 2015-2018 Skymind, Inc.
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

package org.deeplearning4j.nn.conf.layers;

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.val;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.samediff.SDLayerParams;
import org.deeplearning4j.nn.conf.layers.samediff.SameDiffLayer;
import org.deeplearning4j.nn.conf.layers.samediff.SameDiffLayerUtils;
import org.deeplearning4j.nn.params.ConvolutionParamInitializer;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.nn.weights.WeightInitUtil;
import org.deeplearning4j.util.ConvolutionUtils;
import org.nd4j.autodiff.samediff.SDIndex;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.shade.jackson.annotation.JsonIgnoreProperties;

import java.util.*;

@Data
@EqualsAndHashCode(callSuper = true)
@JsonIgnoreProperties({"paramShapes"})
/**
 * Gated recursive unit (GRU) layer
 *
 * @author Max Pumperla
 */
public class GRU extends SameDiffLayer {

    public final static String RECURRENT_WEIGHT_KEY = "RW";
    public final static String WEIGHT_KEY = "W";
    public final static String BIAS_KEY = "b";



    private static final List<String> WEIGHT_KEYS = Arrays.asList(WEIGHT_KEY, RECURRENT_WEIGHT_KEY);
    private static final List<String> BIAS_KEYS = Collections.singletonList(BIAS_KEY);
    private static final List<String> PARAM_KEYS = Arrays.asList(
                WEIGHT_KEY, RECURRENT_WEIGHT_KEY, BIAS_KEY
            );

    private long nIn;
    private long nOut;
    private Activation activation;

    protected GRU(Builder builder) {
        super(builder);
        this.nIn = builder.nIn;
        this.nOut = builder.nOut;
        this.activation = builder.activation;
    }

    private GRU(){
        //No arg constructor for Jackson/JSON serialization
    }


    @Override
    public InputType getOutputType(int layerIndex, InputType inputType) {
        if (inputType == null || inputType.getType() != InputType.Type.RNN) {
            throw new IllegalStateException("Invalid input for RNN layer (layer index = " + layerIndex
                    + ", layer name = \"" + getLayerName() + "\"): expect RNN input type with size > 0. Got: "
                    + inputType);
        }

        InputType.InputTypeRecurrent itr = (InputType.InputTypeRecurrent) inputType;

        return InputType.recurrent(nOut, itr.getTimeSeriesLength());
    }

    @Override
    public void setNIn(InputType inputType, boolean override) {
        if (inputType == null || inputType.getType() != InputType.Type.RNN) {
            throw new IllegalStateException("Invalid input for RNN layer (layer name = \"" + getLayerName()
                    + "\"): expect RNN input type with size > 0. Got: " + inputType);
        }

        if (nIn <= 0 || override) {
            InputType.InputTypeRecurrent r = (InputType.InputTypeRecurrent) inputType;
            this.nIn = r.getSize();
        }
    }

    @Override
    public InputPreProcessor getPreProcessorForInputType(InputType inputType) {
        return InputTypeUtil.getPreprocessorForInputTypeRnnLayers(inputType, getLayerName());
    }

    @Override
    public void defineParameters(SDLayerParams params) {
        params.clear();
        long[] weightsShape = new long[]{nIn, 3* nOut};
        params.addWeightParam(WEIGHT_KEY, weightsShape);
        long[] recurrentWeightsShape = new long[]{nOut, 3 * nOut};
        params.addWeightParam(RECURRENT_WEIGHT_KEY, recurrentWeightsShape);
        long[] biasShape = new long[]{1, nOut};
        params.addBiasParam(BIAS_KEY, biasShape);
    }

    @Override
    public void initializeParameters(Map<String, INDArray> params) {
        // TODO
    }

    @Override
    public SDVariable defineLayer(SameDiff sameDiff, SDVariable layerInput, Map<String, SDVariable> paramTable) {

        SDVariable w = paramTable.get(WEIGHT_KEY);
        SDVariable rW = paramTable.get(RECURRENT_WEIGHT_KEY);
        SDVariable b = paramTable.get(BIAS_KEY);

        long[] inputShape = layerInput.getShape();
        long miniBatch = inputShape[0];

        SDVariable initState = sameDiff.var(miniBatch, nOut);

        SDVariable gruOut = sameDiff.gru("gru", layerInput, initState, w, rW, b);

        return activation.asSameDiff("out", sameDiff, gruOut);

    }

    @Override
    public void applyGlobalConfigToLayer(NeuralNetConfiguration.Builder globalConfig) {
        if (activation == null) {
            activation = SameDiffLayerUtils.fromIActivation(globalConfig.getActivationFn());
        }
    }

    public static class Builder extends SameDiffLayer.Builder<Builder> {

        private int nIn;
        private int nOut;
        private Activation activation = Activation.TANH;


        public Builder nIn(int nIn) {
            this.nIn = nIn;
            return this;
        }

        public Builder nOut(int nOut) {
            this.nOut = nOut;
            return this;
        }


        public Builder activation(Activation activation) {
            this.activation = activation;
            return this;
        }


        @Override
        @SuppressWarnings("unchecked")
        public GRU build() {
            return new GRU(this);
        }
    }
}

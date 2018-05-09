/*-
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.nn.conf.layers;

import lombok.*;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.api.layers.LayerConstraint;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.nn.layers.recurrent.LSTMHelpers;
import org.deeplearning4j.nn.params.GravesBidirectionalLSTMParamInitializer;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.ActivationSigmoid;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.*;

/**
 * Bidirectional LSTM recurrent net, based on Graves: Supervised Sequence Labelling with Recurrent Neural Networks
 * http://www.cs.toronto.edu/~graves/phd.pdf
 *
 * @deprecated use {@link org.deeplearning4j.nn.conf.layers.recurrent.Bidirectional} instead. With the
 * Bidirectional layer wrapper you can make any recurrent layer bidirectional, in particular GravesLSTM.
 * Note that this layer adds the output of both directions, which translates into "ADD" mode in Bidirectional.
 *
 * Usage: {@code .layer(new Bidirectional(Bidirectional.Mode.ADD, new GravesLSTM.Builder()....build()))}
 */
@Data
@NoArgsConstructor
@ToString(callSuper = true)
@EqualsAndHashCode(callSuper = true)
@Deprecated
public class GravesBidirectionalLSTM extends BaseRecurrentLayer {

    private double forgetGateBiasInit;
    private IActivation gateActivationFn = new ActivationSigmoid();

    private GravesBidirectionalLSTM(Builder builder) {
        super(builder);
        this.forgetGateBiasInit = builder.forgetGateBiasInit;
        this.gateActivationFn = builder.gateActivationFn;

        initializeConstraints(builder);
    }

    @Override
    protected void initializeConstraints(org.deeplearning4j.nn.conf.layers.Layer.Builder<?> builder){
        super.initializeConstraints(builder);
        if(((Builder)builder).recurrentConstraints != null){
            if(constraints == null){
                constraints = new ArrayList<>();
            }
            for (LayerConstraint c : ((Builder) builder).recurrentConstraints) {
                LayerConstraint c2 = c.clone();
                Set<String> s = new HashSet<>();
                s.add(GravesBidirectionalLSTMParamInitializer.RECURRENT_WEIGHT_KEY_FORWARDS);
                s.add(GravesBidirectionalLSTMParamInitializer.RECURRENT_WEIGHT_KEY_BACKWARDS);
                c2.setParams(s);
                constraints.add(c2);
            }
        }
    }

    @Override
    public Layer instantiate(NeuralNetConfiguration conf, Collection<TrainingListener> trainingListeners,
                    int layerIndex, INDArray layerParamsView, boolean initializeParams) {
        org.deeplearning4j.nn.layers.recurrent.GravesBidirectionalLSTM ret =
                        new org.deeplearning4j.nn.layers.recurrent.GravesBidirectionalLSTM(conf);
        ret.setListeners(trainingListeners);
        ret.setIndex(layerIndex);
        ret.setParamsViewArray(layerParamsView);
        Map<String, INDArray> paramTable = initializer().init(conf, layerParamsView, initializeParams);
        ret.setParamTable(paramTable);
        ret.setConf(conf);
        return ret;
    }

    @Override
    public ParamInitializer initializer() {
        return GravesBidirectionalLSTMParamInitializer.getInstance();
    }

    @Override
    public double getL1ByParam(String paramName) {
        switch (paramName) {
            case GravesBidirectionalLSTMParamInitializer.INPUT_WEIGHT_KEY_FORWARDS:
            case GravesBidirectionalLSTMParamInitializer.RECURRENT_WEIGHT_KEY_FORWARDS:
            case GravesBidirectionalLSTMParamInitializer.INPUT_WEIGHT_KEY_BACKWARDS:
            case GravesBidirectionalLSTMParamInitializer.RECURRENT_WEIGHT_KEY_BACKWARDS:
                return l1;
            case GravesBidirectionalLSTMParamInitializer.BIAS_KEY_FORWARDS:
            case GravesBidirectionalLSTMParamInitializer.BIAS_KEY_BACKWARDS:
                return l1Bias;
            default:
                throw new IllegalArgumentException("Unknown parameter name: \"" + paramName + "\"");
        }
    }

    @Override
    public double getL2ByParam(String paramName) {
        switch (paramName) {
            case GravesBidirectionalLSTMParamInitializer.INPUT_WEIGHT_KEY_FORWARDS:
            case GravesBidirectionalLSTMParamInitializer.RECURRENT_WEIGHT_KEY_FORWARDS:
            case GravesBidirectionalLSTMParamInitializer.INPUT_WEIGHT_KEY_BACKWARDS:
            case GravesBidirectionalLSTMParamInitializer.RECURRENT_WEIGHT_KEY_BACKWARDS:
                return l2;
            case GravesBidirectionalLSTMParamInitializer.BIAS_KEY_FORWARDS:
            case GravesBidirectionalLSTMParamInitializer.BIAS_KEY_BACKWARDS:
                return l2Bias;
            default:
                throw new IllegalArgumentException("Unknown parameter name: \"" + paramName + "\"");
        }
    }

    @Override
    public LayerMemoryReport getMemoryReport(InputType inputType) {
        return LSTMHelpers.getMemoryReport(this, inputType);
    }

    @AllArgsConstructor
    @NoArgsConstructor
    public static class Builder extends BaseRecurrentLayer.Builder<Builder> {

        private double forgetGateBiasInit = 1.0;
        private IActivation gateActivationFn = new ActivationSigmoid();

        /** Set forget gate bias initalizations. Values in range 1-5 can potentially
         * help with learning or longer-term dependencies.
         */
        public Builder forgetGateBiasInit(double biasInit) {
            this.forgetGateBiasInit = biasInit;
            return this;
        }

        /**
         * Activation function for the LSTM gates.
         * Note: This should be bounded to range 0-1: sigmoid or hard sigmoid, for example
         *
         * @param gateActivationFn Activation function for the LSTM gates
         */
        public Builder gateActivationFunction(String gateActivationFn) {
            return gateActivationFunction(Activation.fromString(gateActivationFn));
        }

        /**
         * Activation function for the LSTM gates.
         * Note: This should be bounded to range 0-1: sigmoid or hard sigmoid, for example
         *
         * @param gateActivationFn Activation function for the LSTM gates
         */
        public Builder gateActivationFunction(Activation gateActivationFn) {
            return gateActivationFunction(gateActivationFn.getActivationFunction());
        }

        /**
         * Activation function for the LSTM gates.
         * Note: This should be bounded to range 0-1: sigmoid or hard sigmoid, for example
         *
         * @param gateActivationFn Activation function for the LSTM gates
         */
        public Builder gateActivationFunction(IActivation gateActivationFn) {
            this.gateActivationFn = gateActivationFn;
            return this;
        }

        @SuppressWarnings("unchecked")
        public GravesBidirectionalLSTM build() {
            return new GravesBidirectionalLSTM(this);
        }
    }

}

/*
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
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.params.GravesBidirectionalLSTMParamInitializer;
import org.deeplearning4j.optimize.api.IterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Collection;
import java.util.Map;

/**
 * LSTM recurrent net, based on Graves: Supervised Sequence Labelling with Recurrent Neural Networks
 * http://www.cs.toronto.edu/~graves/phd.pdf
 */
@Data @NoArgsConstructor
@ToString(callSuper = true)
@EqualsAndHashCode(callSuper = true)
public class GravesBidirectionalLSTM extends BaseRecurrentLayer {

    private double forgetGateBiasInit;

    private GravesBidirectionalLSTM(Builder builder) {
    	super(builder);
        this.forgetGateBiasInit = builder.forgetGateBiasInit;
    }

    @Override
    public Layer instantiate(NeuralNetConfiguration conf, Collection<IterationListener> iterationListeners, int layerIndex, INDArray layerParamsView, boolean initializeParams) {
        org.deeplearning4j.nn.layers.recurrent.GravesBidirectionalLSTM ret
                = new org.deeplearning4j.nn.layers.recurrent.GravesBidirectionalLSTM(conf);
        ret.setListeners(iterationListeners);
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
        switch(paramName){
            case GravesBidirectionalLSTMParamInitializer.INPUT_WEIGHT_KEY_FORWARDS:
            case GravesBidirectionalLSTMParamInitializer.RECURRENT_WEIGHT_KEY_FORWARDS:
            case GravesBidirectionalLSTMParamInitializer.INPUT_WEIGHT_KEY_BACKWARDS:
            case GravesBidirectionalLSTMParamInitializer.RECURRENT_WEIGHT_KEY_BACKWARDS:
                return l1;
            case GravesBidirectionalLSTMParamInitializer.BIAS_KEY_FORWARDS:
            case GravesBidirectionalLSTMParamInitializer.BIAS_KEY_BACKWARDS:
                return 0.0;
            default:
                throw new IllegalArgumentException("Unknown parameter name: \"" + paramName + "\"");
        }
    }

    @Override
    public double getL2ByParam(String paramName) {
        switch(paramName){
            case GravesBidirectionalLSTMParamInitializer.INPUT_WEIGHT_KEY_FORWARDS:
            case GravesBidirectionalLSTMParamInitializer.RECURRENT_WEIGHT_KEY_FORWARDS:
            case GravesBidirectionalLSTMParamInitializer.INPUT_WEIGHT_KEY_BACKWARDS:
            case GravesBidirectionalLSTMParamInitializer.RECURRENT_WEIGHT_KEY_BACKWARDS:
                return l2;
            case GravesBidirectionalLSTMParamInitializer.BIAS_KEY_FORWARDS:
            case GravesBidirectionalLSTMParamInitializer.BIAS_KEY_BACKWARDS:
                return 0.0;
            default:
                throw new IllegalArgumentException("Unknown parameter name: \"" + paramName + "\"");
        }
    }

    @Override
    public double getLearningRateByParam(String paramName) {
        switch(paramName){
            case GravesBidirectionalLSTMParamInitializer.INPUT_WEIGHT_KEY_FORWARDS:
            case GravesBidirectionalLSTMParamInitializer.RECURRENT_WEIGHT_KEY_FORWARDS:
            case GravesBidirectionalLSTMParamInitializer.INPUT_WEIGHT_KEY_BACKWARDS:
            case GravesBidirectionalLSTMParamInitializer.RECURRENT_WEIGHT_KEY_BACKWARDS:
                return learningRate;
            case GravesBidirectionalLSTMParamInitializer.BIAS_KEY_FORWARDS:
            case GravesBidirectionalLSTMParamInitializer.BIAS_KEY_BACKWARDS:
                if(!Double.isNaN(biasLearningRate)){
                    //Bias learning rate has been explicitly set
                    return biasLearningRate;
                } else {
                    return learningRate;
                }
            default:
                throw new IllegalArgumentException("Unknown parameter name: \"" + paramName + "\"");
        }
    }

    @AllArgsConstructor @NoArgsConstructor
    public static class Builder extends BaseRecurrentLayer.Builder<Builder> {

        private double forgetGateBiasInit = 1.0;

        /** Set forget gate bias initalizations. Values in range 1-5 can potentially
         * help with learning or longer-term dependencies.
         */
        public Builder forgetGateBiasInit(double biasInit){
            this.forgetGateBiasInit = biasInit;
            return this;
        }

        @SuppressWarnings("unchecked")
        public GravesBidirectionalLSTM build() {
            return new GravesBidirectionalLSTM(this);
        }
    }

}

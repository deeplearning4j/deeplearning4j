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

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import lombok.ToString;
import org.deeplearning4j.nn.params.PretrainParamInitializer;
import org.nd4j.linalg.lossfunctions.LossFunctions;

@Data @NoArgsConstructor
@ToString(callSuper = true)
@EqualsAndHashCode(callSuper = true)
public abstract class BasePretrainNetwork extends FeedForwardLayer {

    protected LossFunctions.LossFunction lossFunction;
    @Deprecated
    protected String customLossFunction;
    protected double visibleBiasInit;
    private int preTrainIterations;

    public BasePretrainNetwork(Builder builder){
    	super(builder);
        this.lossFunction = builder.lossFunction;
        this.customLossFunction = builder.customLossFunction;
        this.visibleBiasInit = builder.visibleBiasInit;
        this.preTrainIterations = builder.preTrainIterations;

    }

    @Override
    public double getL1ByParam(String paramName) {
        switch (paramName){
            case PretrainParamInitializer.WEIGHT_KEY:
                return l1;
            case PretrainParamInitializer.BIAS_KEY:
                return 0.0;
            case PretrainParamInitializer.VISIBLE_BIAS_KEY:
                return 0.0;
            default:
                throw new IllegalArgumentException("Unknown parameter name: \"" + paramName + "\"");
        }
    }

    @Override
    public double getL2ByParam(String paramName) {
        switch (paramName){
            case PretrainParamInitializer.WEIGHT_KEY:
                return l2;
            case PretrainParamInitializer.BIAS_KEY:
                return 0.0;
            case PretrainParamInitializer.VISIBLE_BIAS_KEY:
                return 0.0;
            default:
                throw new IllegalArgumentException("Unknown parameter name: \"" + paramName + "\"");
        }
    }

    @Override
    public double getLearningRateByParam(String paramName) {
        switch (paramName){
            case PretrainParamInitializer.WEIGHT_KEY:
                return learningRate;
            case PretrainParamInitializer.BIAS_KEY:
                if(!Double.isNaN(biasLearningRate)){
                    //Bias learning rate has been explicitly set
                    return biasLearningRate;
                } else {
                    return learningRate;
                }
            case PretrainParamInitializer.VISIBLE_BIAS_KEY:
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

    public static abstract class Builder<T extends Builder<T>> extends FeedForwardLayer.Builder<T> {
        protected LossFunctions.LossFunction lossFunction = LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY;
        protected String customLossFunction = null;
        protected double visibleBiasInit = 0.0;
        protected int preTrainIterations = 1;

        public Builder() {}

        public T lossFunction(LossFunctions.LossFunction lossFunction) {
            this.lossFunction = lossFunction;
            return (T) this;
        }

        @Deprecated
        public T customLossFunction(String customLossFunction) {
            this.customLossFunction = customLossFunction;
            return (T) this;
        }

        public T visibleBiasInit(double visibleBiasInit){
            this.visibleBiasInit = visibleBiasInit;
            return (T) this;
        }

        public T preTrainIterations(int preTrainIterations){
            this.preTrainIterations = preTrainIterations;
            return (T) this;
        }

    }
}

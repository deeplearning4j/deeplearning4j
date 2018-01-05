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

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import lombok.ToString;
import org.deeplearning4j.nn.params.PretrainParamInitializer;
import org.nd4j.linalg.lossfunctions.LossFunctions;

@Data
@NoArgsConstructor
@ToString(callSuper = true)
@EqualsAndHashCode(callSuper = true)
public abstract class BasePretrainNetwork extends FeedForwardLayer {

    protected LossFunctions.LossFunction lossFunction;
    protected double visibleBiasInit;

    public BasePretrainNetwork(Builder builder) {
        super(builder);
        this.lossFunction = builder.lossFunction;
        this.visibleBiasInit = builder.visibleBiasInit;

    }

    @Override
    public double getL1ByParam(String paramName) {
        switch (paramName) {
            case PretrainParamInitializer.WEIGHT_KEY:
                return l1;
            case PretrainParamInitializer.BIAS_KEY:
                return l1Bias;
            case PretrainParamInitializer.VISIBLE_BIAS_KEY:
                return l1Bias;
            default:
                throw new IllegalArgumentException("Unknown parameter name: \"" + paramName + "\"");
        }
    }

    @Override
    public double getL2ByParam(String paramName) {
        switch (paramName) {
            case PretrainParamInitializer.WEIGHT_KEY:
                return l2;
            case PretrainParamInitializer.BIAS_KEY:
                return l2Bias;
            case PretrainParamInitializer.VISIBLE_BIAS_KEY:
                return l2Bias;
            default:
                throw new IllegalArgumentException("Unknown parameter name: \"" + paramName + "\"");
        }
    }

    @Override
    public boolean isPretrainParam(String paramName) {
        return PretrainParamInitializer.VISIBLE_BIAS_KEY.equals(paramName);
    }

    public static abstract class Builder<T extends Builder<T>> extends FeedForwardLayer.Builder<T> {
        protected LossFunctions.LossFunction lossFunction = LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY;
        protected double visibleBiasInit = 0.0;

        public Builder() {}

        public T lossFunction(LossFunctions.LossFunction lossFunction) {
            this.lossFunction = lossFunction;
            return (T) this;
        }

        public T visibleBiasInit(double visibleBiasInit) {
            this.visibleBiasInit = visibleBiasInit;
            return (T) this;
        }

    }
}

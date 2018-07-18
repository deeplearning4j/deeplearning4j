/*******************************************************************************
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
import lombok.NoArgsConstructor;
import lombok.ToString;
import org.deeplearning4j.nn.params.PretrainParamInitializer;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.shade.jackson.annotation.JsonIgnoreProperties;

@Data
@NoArgsConstructor
@ToString(callSuper = true, exclude = {"pretrain"})
@EqualsAndHashCode(callSuper = true, exclude = {"pretrain"})
@JsonIgnoreProperties("pretrain")
public abstract class BasePretrainNetwork extends FeedForwardLayer {

    protected LossFunctions.LossFunction lossFunction;
    protected double visibleBiasInit;
    protected boolean pretrain;

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

    @Override
    public boolean isPretrain(){
        return pretrain;
    }

    @Override
    public void setPretrain(boolean pretrain){
        this.pretrain = pretrain;
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

/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
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

package org.deeplearning4j.nn.conf.layers;

import lombok.*;
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

    public BasePretrainNetwork(Builder builder) {
        super(builder);
        this.lossFunction = builder.lossFunction;
        this.visibleBiasInit = builder.visibleBiasInit;

    }

    @Override
    public boolean isPretrainParam(String paramName) {
        return PretrainParamInitializer.VISIBLE_BIAS_KEY.equals(paramName);
    }

    @Getter
    @Setter
    public static abstract class Builder<T extends Builder<T>> extends FeedForwardLayer.Builder<T> {

        protected LossFunctions.LossFunction lossFunction = LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY;

        protected double visibleBiasInit = 0.0;

        public Builder() {}

        public T lossFunction(LossFunctions.LossFunction lossFunction) {
            this.setLossFunction(lossFunction);
            return (T) this;
        }

        public T visibleBiasInit(double visibleBiasInit) {
            this.setVisibleBiasInit(visibleBiasInit);
            return (T) this;
        }

    }
}

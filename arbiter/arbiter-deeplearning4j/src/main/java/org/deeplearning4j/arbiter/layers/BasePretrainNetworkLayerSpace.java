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

package org.deeplearning4j.arbiter.layers;

import lombok.AccessLevel;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import org.deeplearning4j.arbiter.optimize.api.ParameterSpace;
import org.deeplearning4j.arbiter.optimize.parameter.FixedValue;
import org.deeplearning4j.nn.conf.layers.BasePretrainNetwork;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.nd4j.shade.jackson.annotation.JsonProperty;


@Data
@EqualsAndHashCode(callSuper = true)
@NoArgsConstructor(access = AccessLevel.PROTECTED) //For Jackson JSON/YAML deserialization
public abstract class BasePretrainNetworkLayerSpace<L extends BasePretrainNetwork> extends FeedForwardLayerSpace<L> {
    @JsonProperty
    protected ParameterSpace<LossFunction> lossFunction;

    protected BasePretrainNetworkLayerSpace(Builder builder) {
        super(builder);
        this.lossFunction = builder.lossFunction;
    }


    public static abstract class Builder<T> extends FeedForwardLayerSpace.Builder<T> {
        protected ParameterSpace<LossFunction> lossFunction;

        public T lossFunction(LossFunction lossFunction) {
            return lossFunction(new FixedValue<LossFunction>(lossFunction));
        }

        public T lossFunction(ParameterSpace<LossFunction> lossFunction) {
            this.lossFunction = lossFunction;
            return (T) this;
        }

    }

}

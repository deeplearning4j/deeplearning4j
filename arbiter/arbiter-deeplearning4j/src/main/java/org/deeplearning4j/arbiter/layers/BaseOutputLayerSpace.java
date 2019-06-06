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
import org.deeplearning4j.arbiter.adapter.LossFunctionParameterSpaceAdapter;
import org.deeplearning4j.arbiter.optimize.api.ParameterSpace;
import org.deeplearning4j.arbiter.optimize.parameter.FixedValue;
import org.deeplearning4j.nn.conf.layers.BaseOutputLayer;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

/**
 * @param <L>    Type of the (concrete) output layer
 */
@Data
@EqualsAndHashCode(callSuper = true)
@NoArgsConstructor(access = AccessLevel.PROTECTED) //For Jackson JSON/YAML deserialization
public abstract class BaseOutputLayerSpace<L extends BaseOutputLayer> extends FeedForwardLayerSpace<L> {

    protected ParameterSpace<ILossFunction> lossFunction;
    protected ParameterSpace<Boolean> hasBias;

    protected BaseOutputLayerSpace(Builder builder) {
        super(builder);
        this.lossFunction = builder.lossFunction;
        this.hasBias = builder.hasBias;
    }

    protected void setLayerOptionsBuilder(BaseOutputLayer.Builder builder, double[] values) {
        super.setLayerOptionsBuilder(builder, values);
        if (lossFunction != null)
            builder.lossFunction(lossFunction.getValue(values));
        if (hasBias != null)
            builder.hasBias(hasBias.getValue(values));
    }

    @SuppressWarnings("unchecked")
    public static abstract class Builder<T> extends FeedForwardLayerSpace.Builder<T> {

        protected ParameterSpace<ILossFunction> lossFunction;
        protected ParameterSpace<Boolean> hasBias;

        public T lossFunction(LossFunction lossFunction) {
            return lossFunction(new FixedValue<>(lossFunction));
        }

        public T lossFunction(ParameterSpace<LossFunction> lossFunction) {
            return iLossFunction(new LossFunctionParameterSpaceAdapter(lossFunction));
        }

        public T iLossFunction(ILossFunction lossFunction) {
            return iLossFunction(new FixedValue<>(lossFunction));
        }

        public T iLossFunction(ParameterSpace<ILossFunction> lossFunction) {
            this.lossFunction = lossFunction;
            return (T) this;
        }

        public T hasBias(boolean hasBias){
            return hasBias(new FixedValue<>(hasBias));
        }

        public T hasBias(ParameterSpace<Boolean> hasBias){
            this.hasBias = hasBias;
            return (T)this;
        }
    }
}

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
import org.deeplearning4j.arbiter.adapter.ActivationParameterSpaceAdapter;
import org.deeplearning4j.arbiter.optimize.api.ParameterSpace;
import org.deeplearning4j.arbiter.optimize.parameter.FixedValue;
import org.deeplearning4j.arbiter.util.LeafUtils;
import org.deeplearning4j.nn.conf.layers.AbstractLSTM;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.IActivation;

/**
 * Layer space for LSTM layers
 *
 * @author Alex Black
 */
@Data
@EqualsAndHashCode(callSuper = true)
@NoArgsConstructor(access = AccessLevel.PROTECTED) //For Jackson JSON/YAML deserialization
public abstract class AbstractLSTMLayerSpace<T extends AbstractLSTM> extends FeedForwardLayerSpace<T> {

    protected ParameterSpace<Double> forgetGateBiasInit;
    protected ParameterSpace<IActivation> gateActivationFn;

    protected AbstractLSTMLayerSpace(Builder builder) {
        super(builder);
        this.forgetGateBiasInit = builder.forgetGateBiasInit;
        this.gateActivationFn = builder.gateActivationFn;

        this.numParameters = LeafUtils.countUniqueParameters(collectLeaves());
    }

    protected void setLayerOptionsBuilder(AbstractLSTM.Builder builder, double[] values) {
        super.setLayerOptionsBuilder(builder, values);
        if (forgetGateBiasInit != null)
            builder.forgetGateBiasInit(forgetGateBiasInit.getValue(values));
        if(gateActivationFn != null)
            builder.gateActivationFunction(gateActivationFn.getValue(values));
    }

    @Override
    public String toString() {
        return toString(", ");
    }

    @Override
    public String toString(String delim) {
        StringBuilder sb = new StringBuilder(); //"AbstractLSTMLayerSpace(");
        if (forgetGateBiasInit != null)
            sb.append("forgetGateBiasInit: ").append(forgetGateBiasInit).append(delim);
        if (gateActivationFn != null)
            sb.append("gateActivationFn: ").append(gateActivationFn).append(delim);
        sb.append(super.toString(delim));
        return sb.toString();
    }

    public static abstract class Builder<T> extends FeedForwardLayerSpace.Builder<T> {

        private ParameterSpace<Double> forgetGateBiasInit;
        private ParameterSpace<IActivation> gateActivationFn;

        public T forgetGateBiasInit(double forgetGateBiasInit) {
            return forgetGateBiasInit(new FixedValue<>(forgetGateBiasInit));
        }

        public T forgetGateBiasInit(ParameterSpace<Double> forgetGateBiasInit) {
            this.forgetGateBiasInit = forgetGateBiasInit;
            return (T)this;
        }

        public T gateActivationFn(Activation activation){
            return gateActivationFn(activation.getActivationFunction());
        }

        public T gateActivation(ParameterSpace<Activation> gateActivationFn){
            return gateActivationFn(new ActivationParameterSpaceAdapter(gateActivationFn));
        }

        public T gateActivationFn(IActivation gateActivationFn){
            return gateActivationFn(new FixedValue<>(gateActivationFn));
        }

        public T gateActivationFn(ParameterSpace<IActivation> gateActivationFn){
            this.gateActivationFn = gateActivationFn;
            return (T)this;
        }
    }
}

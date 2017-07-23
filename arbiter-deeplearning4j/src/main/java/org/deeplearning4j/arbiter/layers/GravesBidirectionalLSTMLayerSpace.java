/*-
 *
 *  * Copyright 2016 Skymind,Inc.
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
package org.deeplearning4j.arbiter.layers;

import lombok.AccessLevel;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import org.deeplearning4j.arbiter.optimize.api.ParameterSpace;
import org.deeplearning4j.arbiter.optimize.parameter.FixedValue;
import org.deeplearning4j.arbiter.util.LeafUtils;
import org.deeplearning4j.nn.conf.layers.GravesBidirectionalLSTM;

import java.util.List;

/**
 * Layer space for Bidirectional LSTM layers
 *
 * @author Alex Black
 */
@Data
@EqualsAndHashCode(callSuper = true)
@NoArgsConstructor(access = AccessLevel.PRIVATE) //For Jackson JSON/YAML deserialization
public class GravesBidirectionalLSTMLayerSpace extends FeedForwardLayerSpace<GravesBidirectionalLSTM> {

    private ParameterSpace<Double> forgetGateBiasInit;

    private GravesBidirectionalLSTMLayerSpace(Builder builder) {
        super(builder);
        this.forgetGateBiasInit = builder.forgetGateBiasInit;

        List<ParameterSpace> l = collectLeaves();
        this.numParameters = LeafUtils.countUniqueParameters(collectLeaves());
    }


    @Override
    public GravesBidirectionalLSTM getValue(double[] values) {
        GravesBidirectionalLSTM.Builder b = new GravesBidirectionalLSTM.Builder();
        setLayerOptionsBuilder(b, values);
        return b.build();
    }

    protected void setLayerOptionsBuilder(GravesBidirectionalLSTM.Builder builder, double[] values) {
        super.setLayerOptionsBuilder(builder, values);
        if (forgetGateBiasInit != null)
            builder.forgetGateBiasInit(forgetGateBiasInit.getValue(values));
    }

    @Override
    public String toString() {
        return toString(", ");
    }

    @Override
    public String toString(String delim) {
        StringBuilder sb = new StringBuilder("GravesBidirectionalLSTMLayerSpace(");
        if (forgetGateBiasInit != null)
            sb.append("forgetGateBiasInit: ").append(forgetGateBiasInit).append(delim);
        sb.append(super.toString(delim)).append(")");
        return sb.toString();
    }

    public static class Builder extends FeedForwardLayerSpace.Builder<Builder> {

        private ParameterSpace<Double> forgetGateBiasInit;

        public Builder forgetGateBiasInit(double forgetGateBiasInit) {
            return forgetGateBiasInit(new FixedValue<>(forgetGateBiasInit));
        }

        public Builder forgetGateBiasInit(ParameterSpace<Double> forgetGateBiasInit) {
            this.forgetGateBiasInit = forgetGateBiasInit;
            return this;
        }

        @Override
        @SuppressWarnings("unchecked")
        public GravesBidirectionalLSTMLayerSpace build() {
            return new GravesBidirectionalLSTMLayerSpace(this);
        }
    }
}

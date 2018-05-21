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
import org.deeplearning4j.nn.conf.layers.AutoEncoder;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * Layer space for autoencoder layers
 */
@Data
@EqualsAndHashCode(callSuper = true)
@NoArgsConstructor(access = AccessLevel.PRIVATE) //For Jackson JSON/YAML deserialization
public class AutoEncoderLayerSpace extends BasePretrainNetworkLayerSpace<AutoEncoder> {
    @JsonProperty
    private ParameterSpace<Double> corruptionLevel;
    @JsonProperty
    private ParameterSpace<Double> sparsity;

    private AutoEncoderLayerSpace(Builder builder) {
        super(builder);
        this.corruptionLevel = builder.corruptionLevel;
        this.sparsity = builder.sparsity;
        this.numParameters = LeafUtils.countUniqueParameters(collectLeaves());
    }

    @Override
    public AutoEncoder getValue(double[] values) {
        AutoEncoder.Builder b = new AutoEncoder.Builder();
        setLayerOptionsBuilder(b, values);
        return b.build();
    }

    protected void setLayerOptionsBuilder(AutoEncoder.Builder builder, double[] values) {
        super.setLayerOptionsBuilder(builder, values);
        if (corruptionLevel != null)
            builder.corruptionLevel(corruptionLevel.getValue(values));
        if (sparsity != null)
            builder.sparsity(sparsity.getValue(values));
    }

    @Override
    public String toString() {
        return toString(", ");
    }

    @Override
    public String toString(String delim) {
        StringBuilder sb = new StringBuilder("AutoEncoderLayerSpace(");
        if (corruptionLevel != null)
            sb.append("corruptionLevel: ").append(corruptionLevel).append(delim);
        if (sparsity != null)
            sb.append("sparsity: ").append(sparsity).append(delim);
        sb.append(super.toString(delim)).append(")");
        return sb.toString();
    }

    public static class Builder extends BasePretrainNetworkLayerSpace.Builder<Builder> {

        private ParameterSpace<Double> corruptionLevel;
        private ParameterSpace<Double> sparsity;

        public Builder corruptionLevel(double corruptionLevel) {
            return corruptionLevel(new FixedValue<>(corruptionLevel));
        }

        public Builder corruptionLevel(ParameterSpace<Double> corruptionLevel) {
            this.corruptionLevel = corruptionLevel;
            return this;
        }

        public Builder sparsity(double sparsity) {
            return sparsity(new FixedValue<>(sparsity));
        }

        public Builder sparsity(ParameterSpace<Double> sparsity) {
            this.sparsity = sparsity;
            return this;
        }

        public AutoEncoderLayerSpace build() {
            return new AutoEncoderLayerSpace(this);
        }

    }
}

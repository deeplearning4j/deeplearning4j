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

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import org.deeplearning4j.arbiter.optimize.api.ParameterSpace;
import org.deeplearning4j.arbiter.optimize.parameter.FixedValue;
import org.deeplearning4j.arbiter.util.LeafUtils;
import org.deeplearning4j.nn.conf.layers.DenseLayer;

/**
 * layer hyperparameter configuration space for dense layers (i.e., multi-layer perceptron layers)
 *
 * @author Alex Black
 */
@Data
@EqualsAndHashCode(callSuper = true)
@NoArgsConstructor //For Jackson JSON/YAML deserialization
public class DenseLayerSpace extends FeedForwardLayerSpace<DenseLayer> {

    protected ParameterSpace<Boolean> hasBias;

    private DenseLayerSpace(Builder builder) {
        super(builder);

        this.hasBias = builder.hasBias;
        this.numParameters = LeafUtils.countUniqueParameters(collectLeaves());
    }

    @Override
    public DenseLayer getValue(double[] values) {
        //Using the builder here, to get default options
        DenseLayer.Builder b = new DenseLayer.Builder();
        setLayerOptionsBuilder(b, values);
        return b.build();
    }

    protected void setLayerOptionsBuilder(DenseLayer.Builder builder, double[] values) {
        super.setLayerOptionsBuilder(builder, values);
        if(hasBias != null)
            builder.hasBias(hasBias.getValue(values));
    }

    public static class Builder extends FeedForwardLayerSpace.Builder<Builder> {

        protected ParameterSpace<Boolean> hasBias;

        public Builder hasBias(boolean hasBias){
            return hasBias(new FixedValue<>(hasBias));
        }

        public Builder hasBias(ParameterSpace<Boolean> hasBias){
            this.hasBias = hasBias;
            return this;
        }

        @Override
        @SuppressWarnings("unchecked")
        public DenseLayerSpace build() {
            return new DenseLayerSpace(this);
        }
    }

    @Override
    public String toString() {
        return toString(", ");
    }

    @Override
    public String toString(String delim) {
        return "DenseLayerSpace(" + super.toString(delim) + ")";
    }

}

/*
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
package org.arbiter.deeplearning4j.layers;

import org.arbiter.optimize.parameter.FixedValue;
import org.arbiter.optimize.api.ParameterSpace;
import org.deeplearning4j.nn.conf.layers.AutoEncoder;

public class AutoEncoderLayerSpace extends BasePretrainNetworkLayerSpace<AutoEncoder> {

    private ParameterSpace<Double> corruptionLevel;
    private ParameterSpace<Double> sparsity;

    private AutoEncoderLayerSpace(Builder builder){
        super(builder);
        this.corruptionLevel = builder.corruptionLevel;
        this.sparsity = builder.sparsity;
    }

    @Override
    public AutoEncoder randomLayer() {
        AutoEncoder.Builder b = new AutoEncoder.Builder();
        setLayerOptionsBuilder(b);
        return b.build();
    }

    protected void setLayerOptionsBuilder(AutoEncoder.Builder builder){
        super.setLayerOptionsBuilder(builder);
        if(corruptionLevel != null) builder.corruptionLevel(corruptionLevel.randomValue());
        if(sparsity != null) builder.sparsity(sparsity.randomValue());
    }

    @Override
    public String  toString(){
        return toString(", ");
    }

    @Override
    public String toString(String delim){
        StringBuilder sb = new StringBuilder("AutoEncoderLayerSpace(");
        if(corruptionLevel != null) sb.append("corruptionLevel: ").append(corruptionLevel).append(delim);
        if(sparsity != null) sb.append("sparsity: ").append(sparsity).append(delim);
        sb.append(super.toString(delim)).append(")");
        return sb.toString();
    }

    public class Builder extends BasePretrainNetworkLayerSpace.Builder{

        private ParameterSpace<Double> corruptionLevel;
        private ParameterSpace<Double> sparsity;

        public Builder corruptionLevel(double corruptionLevel){
            return corruptionLevel(new FixedValue<>(corruptionLevel));
        }

        public Builder corruptionLevel(ParameterSpace<Double> corruptionLevel){
            this.corruptionLevel = corruptionLevel;
            return this;
        }

        public Builder sparsity(double sparsity){
            return sparsity(new FixedValue<>(sparsity));
        }

        public Builder sparsity(ParameterSpace<Double> sparsity){
            this.sparsity = sparsity;
            return this;
        }

        public AutoEncoderLayerSpace build(){
            return new AutoEncoderLayerSpace(this);
        }

    }
}

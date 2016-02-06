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
import org.deeplearning4j.nn.conf.layers.LocalResponseNormalization;

import java.util.List;

public class LocalResponseNormalizationLayerSpace extends LayerSpace<LocalResponseNormalization> {

    private ParameterSpace<Double> n;
    private ParameterSpace<Double> k;
    private ParameterSpace<Double> alpha;
    private ParameterSpace<Double> beta;


    private LocalResponseNormalizationLayerSpace(Builder builder){
        super(builder);
        this.n = builder.n;
        this.k = builder.k;
        this.alpha = builder.alpha;
        this.beta = builder.beta;
    }
    
    @Override
    public List<ParameterSpace> collectLeaves(){
        List<ParameterSpace> list = super.collectLeaves();
        if(n != null) list.addAll(n.collectLeaves());
        if(k != null) list.addAll(k.collectLeaves());
        if(alpha != null) list.addAll(alpha.collectLeaves());
        if(beta != null) list.addAll(beta.collectLeaves());
        return list;
    }

    @Override
    public LocalResponseNormalization getValue(double[] values) {
        LocalResponseNormalization.Builder b = new LocalResponseNormalization.Builder();
        setLayerOptionsBuilder(b,values);
        return b.build();
    }

    protected void setLayerOptionsBuilder(LocalResponseNormalization.Builder builder, double[] values){
        super.setLayerOptionsBuilder(builder,values);
        if(n != null) builder.n(n.getValue(values));
        if(k != null) builder.k(k.getValue(values));
        if(alpha != null) builder.alpha(alpha.getValue(values));
        if(beta != null) builder.beta(beta.getValue(values));
    }


    public class Builder extends LayerSpace.Builder {

        private ParameterSpace<Double> n;
        private ParameterSpace<Double> k;
        private ParameterSpace<Double> alpha;
        private ParameterSpace<Double> beta;


        public Builder n(double n){
            return n(new FixedValue<>(n));
        }

        public Builder n(ParameterSpace<Double> n){
            this.n = n;
            return this;
        }

        public Builder k(double k){
            return k(new FixedValue<>(k));
        }

        public Builder k(ParameterSpace<Double> k){
            this.k = k;
            return this;
        }

        public Builder alpha(double alpha){
            return alpha(new FixedValue<>(alpha));
        }

        public Builder alpha(ParameterSpace<Double> alpha){
            this.alpha = alpha;
            return this;
        }

        public Builder beta(double beta){
            return beta(new FixedValue<>(beta));
        }

        public Builder beta(ParameterSpace<Double> beta){
            this.beta = beta;
            return this;
        }

        public LocalResponseNormalizationLayerSpace build(){
            return new LocalResponseNormalizationLayerSpace(this);
        }

    }

}

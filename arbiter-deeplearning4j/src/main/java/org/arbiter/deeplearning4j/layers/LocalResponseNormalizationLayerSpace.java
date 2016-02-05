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
    public LocalResponseNormalization randomLayer() {
        LocalResponseNormalization.Builder b = new LocalResponseNormalization.Builder();
        setLayerOptionsBuilder(b);
        return b.build();
    }

    protected void setLayerOptionsBuilder(LocalResponseNormalization.Builder builder){
        super.setLayerOptionsBuilder(builder);
        if(n != null) builder.n(n.randomValue());
        if(k != null) builder.k(k.randomValue());
        if(alpha != null) builder.alpha(alpha.randomValue());
        if(beta != null) builder.beta(beta.randomValue());
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

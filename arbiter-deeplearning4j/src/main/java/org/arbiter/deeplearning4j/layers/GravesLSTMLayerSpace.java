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
import org.deeplearning4j.nn.conf.layers.GravesLSTM;

public class GravesLSTMLayerSpace extends FeedForwardLayerSpace<GravesLSTM> {

    private ParameterSpace<Double> forgetGateBiasInit;

    private GravesLSTMLayerSpace(Builder builder){
        super(builder);
        this.forgetGateBiasInit = builder.forgetGateBiasInit;
    }


    @Override
    public GravesLSTM randomLayer() {
        GravesLSTM.Builder b = new GravesLSTM.Builder();
        setLayerOptionsBuilder(b);
        return b.build();
    }

    protected void setLayerOptionsBuilder(GravesLSTM.Builder builder){
        super.setLayerOptionsBuilder(builder);
        if(forgetGateBiasInit != null) builder.forgetGateBiasInit(forgetGateBiasInit.randomValue());
    }

    @Override
    public String toString(){
        return toString(", ");
    }

    @Override
    public String toString(String delim){
        StringBuilder sb = new StringBuilder("GravesLSTMLayerSpace(");
        if(forgetGateBiasInit != null) sb.append("forgetGateBiasInit: ").append(forgetGateBiasInit).append(delim);
        sb.append(super.toString(delim)).append(")");
        return sb.toString();
    }

    public static class Builder extends FeedForwardLayerSpace.Builder<Builder>{

        private ParameterSpace<Double> forgetGateBiasInit;

        public Builder forgetGateBiasInit(double forgetGateBiasInit){
            return forgetGateBiasInit(new FixedValue<>(forgetGateBiasInit));
        }

        public Builder forgetGateBiasInit( ParameterSpace<Double> forgetGateBiasInit){
            this.forgetGateBiasInit = forgetGateBiasInit;
            return this;
        }

        @Override
        @SuppressWarnings("unchecked")
        public GravesLSTMLayerSpace build(){
            return new GravesLSTMLayerSpace(this);
        }
    }
}

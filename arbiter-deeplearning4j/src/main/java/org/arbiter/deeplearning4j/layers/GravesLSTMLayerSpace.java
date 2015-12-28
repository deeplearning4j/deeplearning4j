package org.arbiter.deeplearning4j.layers;

import org.arbiter.optimize.parameter.FixedValue;
import org.arbiter.optimize.parameter.ParameterSpace;
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
            return forgetGateBiasInit(new FixedValue<Double>(forgetGateBiasInit));
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

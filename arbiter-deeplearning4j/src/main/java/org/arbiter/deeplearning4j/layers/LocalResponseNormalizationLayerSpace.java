package org.arbiter.deeplearning4j.layers;

import org.arbiter.optimize.parameter.FixedValue;
import org.arbiter.optimize.parameter.ParameterSpace;
import org.deeplearning4j.nn.conf.layers.LocalResponseNormalization;

/**
 * Created by Alex on 28/12/2015.
 */
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
            return n(new FixedValue<Double>(n));
        }

        public Builder n(ParameterSpace<Double> n){
            this.n = n;
            return this;
        }

        public Builder k(double k){
            return k(new FixedValue<Double>(k));
        }

        public Builder k(ParameterSpace<Double> k){
            this.k = k;
            return this;
        }

        public Builder alpha(double alpha){
            return alpha(new FixedValue<Double>(alpha));
        }

        public Builder alpha(ParameterSpace<Double> alpha){
            this.alpha = alpha;
            return this;
        }

        public Builder beta(double beta){
            return beta(new FixedValue<Double>(beta));
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

package org.arbiter.deeplearning4j.layers;

import org.arbiter.optimize.parameter.FixedValue;
import org.arbiter.optimize.parameter.ParameterSpace;
import org.deeplearning4j.nn.conf.layers.RBM;

public class RBMLayerSpace extends BasePretrainNetworkLayerSpace<RBM> {

    private ParameterSpace<RBM.HiddenUnit> hiddenUnit;
    private ParameterSpace<RBM.VisibleUnit> visibleUnit;
    private ParameterSpace<Integer> k;
    private ParameterSpace<Double> sparsity;

    private RBMLayerSpace(Builder builder){
        super(builder);
        this.hiddenUnit = builder.hiddenUnit;
        this.visibleUnit = builder.visibleUnit;
        this.k = builder.k;
        this.sparsity = builder.sparsity;
    }

    @Override
    public RBM randomLayer() {
        RBM.Builder b = new RBM.Builder();
        setLayerOptionsBuilder(b);
        return b.build();
    }

    protected void setLayerOptionsBuilder(RBM.Builder builder){
        super.setLayerOptionsBuilder(builder);
        if(hiddenUnit != null) builder.hiddenUnit(hiddenUnit.randomValue());
        if(visibleUnit != null) builder.visibleUnit(visibleUnit.randomValue());
        if(k != null) builder.k(k.randomValue());
        if(sparsity != null) builder.sparsity(sparsity.randomValue());
    }

    @Override
    public String toString(){
        return toString(", ");
    }

    @Override
    public String toString(String delim){
        StringBuilder sb = new StringBuilder("RBMLayerSpace(");
        if(hiddenUnit != null) sb.append("hiddenUnit: ").append(hiddenUnit).append(delim);
        if(visibleUnit != null) sb.append("visibleUnit: ").append(visibleUnit).append(delim);
        if(k != null) sb.append("k: ").append(k).append(delim);
        if(sparsity != null) sb.append("sparsity: ").append(sparsity).append(delim);
        sb.append(super.toString(delim)).append(")");
        return sb.toString();
    }


    public class Builder extends BasePretrainNetworkLayerSpace.Builder{
        private ParameterSpace<RBM.HiddenUnit> hiddenUnit;
        private ParameterSpace<RBM.VisibleUnit> visibleUnit;
        private ParameterSpace<Integer> k;
        private ParameterSpace<Double> sparsity;

        public Builder hiddenUnit(RBM.HiddenUnit hiddenUnit){
            return hiddenUnit(new FixedValue<>(hiddenUnit));
        }

        public Builder hiddenUnit(ParameterSpace<RBM.HiddenUnit> hiddenUnit){
            this.hiddenUnit = hiddenUnit;
            return this;
        }

        public Builder visibleUnit(RBM.VisibleUnit visibleUnit){
            return visibleUnit(new FixedValue<>(visibleUnit));
        }

        public Builder visibleUnit(ParameterSpace<RBM.VisibleUnit> visibleUnit){
            this.visibleUnit = visibleUnit;
            return this;
        }

        public Builder k( int k ){
            return k(new FixedValue<>(k));
        }

        public Builder k(ParameterSpace<Integer> k){
            this.k = k;
            return this;
        }

        public Builder sparsity(double sparsity){
            return sparsity(new FixedValue<>(sparsity));
        }

        public Builder sparsity(ParameterSpace<Double> sparsity){
            this.sparsity = sparsity;
            return this;
        }

        public RBMLayerSpace build(){
            return new RBMLayerSpace(this);
        }
    }

}

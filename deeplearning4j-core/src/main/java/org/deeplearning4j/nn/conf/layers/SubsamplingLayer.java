package org.deeplearning4j.nn.conf.layers;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import org.deeplearning4j.nn.conf.distribution.Distribution;
import org.deeplearning4j.nn.weights.WeightInit;

/**
 * @author Adam Gibson
 */
@Data @NoArgsConstructor
public class SubsamplingLayer extends Layer {

    protected PoolingType poolingType;
    protected int[] kernelSize; // Same as filter size from the last conv layer
    protected int[] stride; // Default is 2. Down-sample by a factor of 2

    public enum PoolingType {
        MAX, AVG, SUM, NONE
    }

    private SubsamplingLayer(Builder builder) {
    	super(builder);
        this.poolingType = builder.poolingType;
        this.stride = builder.stride;
        this.kernelSize = kernelSize;
    }

    @AllArgsConstructor
    public static class Builder extends Layer.Builder {
        private PoolingType poolingType = PoolingType.MAX;;
        private int[] kernelSize = new int[] {2, 2}; // Same as filter size from the last conv layer
        private int[] stride = new int[] {2, 2}; // Default is 2. Down-sample by a factor of 2

        public Builder(PoolingType poolingType, int[] stride) {
            this.poolingType = poolingType;
            this.stride = stride;
        }

        @Override
        public Builder activation(String activationFunction) {
            this.activationFunction = activationFunction;
            return this;
        }
        @Override
        public Builder weightInit(WeightInit weightInit) {
            this.weightInit = weightInit;
            return this;
        }
        @Override
        public Builder dist(Distribution dist){
            this.dist = dist;
            return this;
        }
        @Override
        public Builder dropOut(double dropOut) {
            this.dropOut = dropOut;
            return this;
        }
        @Override
        @SuppressWarnings("unchecked")
        public SubsamplingLayer build() {
            return new SubsamplingLayer(this);
        }
    }

}

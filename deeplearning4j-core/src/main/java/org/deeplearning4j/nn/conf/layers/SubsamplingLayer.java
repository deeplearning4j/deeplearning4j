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

    private static final long serialVersionUID = -7095644470333017030L;
    protected poolingType poolingType;
    protected int[] kernelSize; // Same as filter size from the last conv layer
    protected int[] stride; // Default is 2. Down-sample by a factor of 2

    public enum poolingType {
        MAX, AVG, SUM, NONE
    }

    private SubsamplingLayer(Builder builder) {
        super(builder);
        this.poolingType = builder.poolingType;
        this.kernelSize = builder.kernelSize;
        this.stride = builder.stride;
    }

    @AllArgsConstructor
    @NoArgsConstructor
    public static class Builder extends Layer.Builder {
        private poolingType poolingType;
        private int[] kernelSize; // Same as filter size from the last conv layer
        private int[] stride; // Default is 2. Down-sample by a factor of 2

        public Builder(poolingType poolingType, int[] stride) {
            this.poolingType = poolingType;
            this.stride = stride;
        }

        public Builder poolingType(poolingType poolingType) {
            this.poolingType = poolingType;
            return this;
        }
        public Builder kernelSize(int[] kernelSize) {
            this.kernelSize = kernelSize;
            return this;
        }

        public Builder stride(int[] stride) {
            this.stride = stride;
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

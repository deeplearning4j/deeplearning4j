package org.deeplearning4j.nn.conf.layers;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.deeplearning4j.nn.weights.WeightInit;

/**
 * @author Adam Gibson
 */
@Data
@NoArgsConstructor
public class SubsamplingLayer extends Layer {

    private static final long serialVersionUID = -7095644470333017030L;
    protected PoolingType poolingType;
    protected int[] filterSize; // Same as filter size from the last conv layer
    protected int[] stride; // Default is 2. Down-sample by a factor of 2

    public enum PoolingType {
        MAX, AVG, SUM, NONE
    }

    private SubsamplingLayer(Builder builder) {
        this.poolingType = builder.poolingType;
        this.stride = builder.stride;
//        this.kernelSize = builder.kernelSize;
    }

    @AllArgsConstructor
    public static class Builder extends Layer.Builder {
        private PoolingType poolingType;
        private int[] stride; // Default is 2. Down-sample by a factor of 2
//        private int[] kernelSize; // Same as filter size from the last conv layer

        @Override
        public Builder activation(String activationFunction) {
            throw new UnsupportedOperationException("SubsamplingLayer does not accept activation");
        }
        @Override
        public Builder weightInit(WeightInit weightInit) {
            throw new UnsupportedOperationException("SubsamplingLayer does not accept weight init");
        }
        @Override
        public Builder dropOut(double dropOut) {
            throw new UnsupportedOperationException("SubsamplingLayer does not accept dropout");
        }
        @Override
        @SuppressWarnings("unchecked")
        public SubsamplingLayer build() {
            return new SubsamplingLayer(this);
        }
    }

}

package org.deeplearning4j.nn.conf.layers;

import lombok.*;

import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.Distribution;
import org.deeplearning4j.nn.weights.WeightInit;

/**
 * Subsampling layer also referred to as pooling in convolution neural nets
 *
 *  Supports the following pooling types:
 *     MAX
 *     AVG
 *     NON
 * @author Adam Gibson
 */

@Data @NoArgsConstructor
@ToString(callSuper = true)
@EqualsAndHashCode(callSuper = true)
public class SubsamplingLayer extends Layer {

    protected PoolingType poolingType;
    protected int[] kernelSize; // Same as filter size from the last conv layer
    protected int[] stride; // Default is 2. Down-sample by a factor of 2
    protected int[] padding;

    public enum PoolingType {
        MAX, AVG, SUM, NONE
    }

    private SubsamplingLayer(Builder builder) {
        super(builder);
        this.poolingType = builder.poolingType;
        if(builder.kernelSize.length != 2)
            throw new IllegalArgumentException("Kernel size of should be rows x columns (a 2d array)");
        this.kernelSize = builder.kernelSize;
        if(builder.stride.length != 2)
            throw new IllegalArgumentException("Invalid stride, must be length 2");
        this.stride = builder.stride;
        this.padding = builder.padding;
    }

    @Override
    public SubsamplingLayer clone() {
        SubsamplingLayer clone = (SubsamplingLayer) super.clone();

        if(clone.kernelSize != null) clone.kernelSize = clone.kernelSize.clone();
        if(clone.stride != null) clone.stride = clone.stride.clone();
        if(clone.padding != null) clone.padding = clone.padding.clone();
        return clone;
    }

    @AllArgsConstructor
    public static class Builder extends Layer.Builder<Builder> {
        private PoolingType poolingType = PoolingType.MAX;
        private int[] kernelSize = new int[] {1, 1}; // Same as filter size from the last conv layer
        private int[] stride = new int[] {2, 2}; // Default is 2. Down-sample by a factor of 2
        private int[] padding = new int[] {0, 0};

        public Builder(PoolingType poolingType, int[] kernelSize, int[] stride) {
            this.poolingType = poolingType;
            this.kernelSize = kernelSize;
            this.stride = stride;
        }

        public Builder(PoolingType poolingType, int[] kernelSize) {
            this.poolingType = poolingType;
            this.kernelSize = kernelSize;
        }

        public Builder(int[] kernelSize, int[] stride, int[] padding) {
            this.kernelSize = kernelSize;
            this.stride = stride;
            this.padding = padding;
        }

        public Builder(int[] stride) {
            this(new int[]{1,1},stride);
        }

        public Builder(int[] kernelSize, int[] stride) {
            this.kernelSize = kernelSize;
            this.stride = stride;
        }

        public Builder(PoolingType poolingType) {
            this.poolingType = poolingType;
        }

        public Builder() {}

        @Override
        @SuppressWarnings("unchecked")
        public SubsamplingLayer build() {
            return new SubsamplingLayer(this);
        }

        public Builder poolingType(PoolingType poolingType){
            this.poolingType = poolingType;
            return this;
        }

        public Builder kernelSize(int... kernelSize){
            this.kernelSize = kernelSize;
            return this;
        }

        public Builder stride(int... stride){
            this.stride = stride;
            return this;
        }

        public Builder padding(int... padding){
            this.padding = padding;
            return this;
        }
    }

}

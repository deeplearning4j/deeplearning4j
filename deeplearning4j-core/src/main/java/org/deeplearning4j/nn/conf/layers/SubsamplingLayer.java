package org.deeplearning4j.nn.conf.layers;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

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
        this.kernelSize = builder.kernelSize;
        this.stride = builder.stride;
        this.padding = builder.padding;
    }

    @AllArgsConstructor
    public static class Builder extends Layer.Builder {
        private PoolingType poolingType = PoolingType.MAX;
        private int[] kernelSize = new int[] {2, 2}; // Same as filter size from the last conv layer
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

        public Builder(int[] kernelSize, int[] stride) {
            this.kernelSize = kernelSize;
            this.stride = stride;
        }

        public Builder(PoolingType poolingType) {
            this.poolingType = poolingType;
        }

        public Builder() {}

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

        public Builder poolingType(PoolingType poolingType){
            this.poolingType = poolingType;
            return this;
        }

        public Builder kernelSize(int[] kernelSize){
            this.kernelSize = kernelSize;
            return this;
        }

        public Builder stride(int[] stride){
            this.stride = stride;
            return this;
        }

        public Builder padding(int[] padding){
            this.padding = padding;
            return this;
        }

        @Override
        public Builder updater(Updater updater){
        	this.updater = updater;
        	return this;
        }
    }

}

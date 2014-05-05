package org.deeplearning4j.nn;

import org.apache.commons.math3.distribution.RealDistribution;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.nn.activation.ActivationFunction;
import org.jblas.DoubleMatrix;


/**
 * Down sampling hidden layer
 */
public class DownSamplingLayer extends HiddenLayer {

    private int[] stride;
    private int numFeatureMaps;
    private DoubleMatrix fmSize;


    private DownSamplingLayer() {}


    public int[] getStride() {
        return stride;
    }

    public void setStride(int[] stride) {
        this.stride = stride;
    }

    public int getNumFeatureMaps() {
        return numFeatureMaps;
    }

    public void setNumFeatureMaps(int numFeatureMaps) {
        this.numFeatureMaps = numFeatureMaps;
    }


    public DoubleMatrix getFmSize() {
        return fmSize;
    }

    public void setFmSize(DoubleMatrix fmSize) {
        this.fmSize = fmSize;
    }


    public static class Builder extends HiddenLayer.Builder {

        private int[] stride;
        private DoubleMatrix fmSize;
        private int numFeatureMaps;




        public Builder numFeatureMaps(int numFeatureMaps) {
            this.numFeatureMaps = numFeatureMaps;
            return this;

        }
        public Builder withStride(int[] stride) {
            this.stride = stride;
            return this;
        }

        public Builder withFmSize(DoubleMatrix fmSize) {
            this.fmSize = fmSize;
            return this;
        }


        @Override
        public Builder dist(RealDistribution dist) {
            super.dist(dist);
            return this;
        }

        @Override
        public Builder nIn(int nIn) {
            super.nIn(nIn);
            return this;
        }

        @Override
        public Builder nOut(int nOut) {
            super.nOut(nOut);
            return this;
        }

        @Override
        public Builder withWeights(DoubleMatrix W) {
            super.withWeights(W);
            return this;
        }

        @Override
        public Builder withRng(RandomGenerator gen) {
            super.withRng(gen);
            return this;
        }

        @Override
        public Builder withActivation(ActivationFunction function) {
            super.withActivation(function);
            return this;
        }

        @Override
        public Builder withBias(DoubleMatrix b) {
            super.withBias(b);
            return this;
        }

        @Override
        public HiddenLayer.Builder withInput(DoubleMatrix input) {
            return super.withInput(input);
        }

        @Override
        public DownSamplingLayer build() {
            DownSamplingLayer layer = new DownSamplingLayer();
            layer.fmSize = fmSize;
            layer.numFeatureMaps = numFeatureMaps;
            layer.stride = stride;
            layer.activationFunction = activationFunction;
            layer.b = b;
            layer.rng = rng;
            layer.W = W;
            layer.input = input;
            layer.dist = dist;
            return layer;
        }
    }

}

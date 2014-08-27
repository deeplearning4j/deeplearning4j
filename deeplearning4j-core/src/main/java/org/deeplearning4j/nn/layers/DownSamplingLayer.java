package org.deeplearning4j.nn.layers;

import org.apache.commons.math3.distribution.RealDistribution;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.linalg.api.activation.ActivationFunction;
import org.deeplearning4j.linalg.api.ndarray.INDArray;


/**
 * Down sampling hidden layer
 */
public class DownSamplingLayer extends HiddenLayer {

    private int[] stride;
    private int numFeatureMaps;
    private INDArray fmSize;
    private INDArray featureMap;

    private DownSamplingLayer() {}


    public INDArray getFeatureMap() {
        return featureMap;
    }

    public void setFeatureMap(INDArray featureMap) {
        this.featureMap = featureMap;
    }

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


    public INDArray getFmSize() {
        return fmSize;
    }

    public void setFmSize(INDArray fmSize) {
        this.fmSize = fmSize;
    }


    public static class Builder extends HiddenLayer.Builder {

        private int[] stride;
        private INDArray fmSize;
        private int numFeatureMaps;




        public Builder numFeatureMaps(int numFeatureMaps) {
            this.numFeatureMaps = numFeatureMaps;
            return this;

        }
        public Builder withStride(int[] stride) {
            this.stride = stride;
            return this;
        }

        public Builder withFmSize(INDArray fmSize) {
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
        public Builder withWeights(INDArray W) {
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
        public Builder withBias(INDArray b) {
            super.withBias(b);
            return this;
        }

        @Override
        public Builder withInput(INDArray input) {
             super.withInput(input);
            return this;
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

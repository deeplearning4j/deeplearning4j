package org.deeplearning4j.nn.layers;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;


/**
 * Down sampling hidden layer
 */
public class DownSamplingLayer extends Layer {

    private int[] stride;
    private int numFeatureMaps;
    private INDArray fmSize;
    private INDArray featureMap;

    public DownSamplingLayer(NeuralNetConfiguration conf, INDArray W, INDArray b, INDArray input) {
        super(conf, W, b, input);
    }


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


    public static class Builder extends Layer.Builder {

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
        public Builder withWeights(INDArray W) {
            super.withWeights(W);
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
            DownSamplingLayer layer = new DownSamplingLayer(conf,W,b,input);
            layer.fmSize = fmSize;
            layer.numFeatureMaps = numFeatureMaps;
            layer.stride = stride;
            return layer;
        }
    }

}

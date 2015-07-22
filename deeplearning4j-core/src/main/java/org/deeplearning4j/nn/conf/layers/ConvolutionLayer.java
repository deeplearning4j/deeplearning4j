package org.deeplearning4j.nn.conf.layers;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.convolution.Convolution;

/**
 * @author Adam Gibson
 */
@Data
@NoArgsConstructor
public class ConvolutionLayer extends Layer {

    private static final long serialVersionUID = 3073633667258683720L;
    private int[] filterSize; // Square filter
    private int filterDepth; // Depth of the each column of neurons
    private Convolution.Type convolutionType; // FULL / VALID / SAME

    private ConvolutionLayer(Builder builder) {
        this.filterSize = builder.filterSize;
        this.filterDepth = builder.filterDepth;
        this.convolutionType = builder.convolutionType;
        this.activationFunction = builder.activationFunction;
        this.weightInit = builder.weightInit;
        this.dropOut = builder.dropOut;
    }

    @AllArgsConstructor
    public static class Builder extends Layer.Builder {
        private int[] filterSize; // Square filter
        private int filterDepth; // Depth of the each column of neurons
        private Convolution.Type convolutionType; // FULL / VALID / SAME

        public Builder(int[] filterSize, int filterDepth) {
            this.filterSize = filterSize;
            this.filterDepth = filterDepth;
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
        public Builder dropOut(double dropOut) {
            throw new UnsupportedOperationException("ConvolutionLayer Layer does not accept dropout");
        }
        @Override
        @SuppressWarnings("unchecked")
        public ConvolutionLayer build() {
            return new ConvolutionLayer(this);
        }
    }
}

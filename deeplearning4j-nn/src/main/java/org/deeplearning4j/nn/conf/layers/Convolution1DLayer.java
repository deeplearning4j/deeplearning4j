package org.deeplearning4j.nn.conf.layers;

import lombok.*;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.util.*;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Collection;
import java.util.Map;

/**
 * 1D (temporal) convolutional layer. Currently, we just subclass off the
 * ConvolutionLayer and hard code the "width" dimension to 1. Also, this
 * layer accepts RNN InputTypes instead of CNN InputTypes.
 *
 * This approach treats a multivariate time series with L timesteps and
 * P variables as an L x 1 x P image (L rows high, 1 column wide, P
 * channels deep). The kernel should be H<L pixels high and W=1 pixels
 * wide.
 *
 * TODO: We will eventually want to NOT subclass off of ConvolutionLayer.
 *
 * @author dave@skymind.io
 */
@Data @NoArgsConstructor
@ToString(callSuper = true)
@EqualsAndHashCode(callSuper = true)
public class Convolution1DLayer extends ConvolutionLayer {

    private Convolution1DLayer(Builder builder) {
        super(builder);
    }

    @Override
    public org.deeplearning4j.nn.api.Layer instantiate(NeuralNetConfiguration conf, Collection<IterationListener> iterationListeners, int layerIndex, INDArray layerParamsView, boolean initializeParams) {
        org.deeplearning4j.util.LayerValidation.assertNInNOutSet("Convolution1DLayer", getLayerName(), layerIndex, getNIn(), getNOut());

        org.deeplearning4j.nn.layers.convolution.Convolution1DLayer ret
                = new org.deeplearning4j.nn.layers.convolution.Convolution1DLayer(conf);
        ret.setListeners(iterationListeners);
        ret.setIndex(layerIndex);
        ret.setParamsViewArray(layerParamsView);
        Map<String, INDArray> paramTable = initializer().init(conf, layerParamsView, initializeParams);
        ret.setParamTable(paramTable);
        ret.setConf(conf);
        return ret;
    }

    @Override
    public InputType getOutputType(int layerIndex, InputType inputType) {
        if (inputType == null || inputType.getType() != InputType.Type.RNN) {
            throw new IllegalStateException("Invalid input for 1D CNN layer (layer index = " + layerIndex +
                    ", layer name = \"" + getLayerName() + "\"): expect RNN input type with size > 0. Got: " + inputType);
        }

        return InputType.recurrent(nOut);
    }

    @Override
    public void setNIn(InputType inputType, boolean override){
        if (inputType == null || inputType.getType() != InputType.Type.RNN) {
            throw new IllegalStateException("Invalid input for 1D CNN layer (layer name = \"" + getLayerName() + "\"): expect RNN input type with size > 0. Got: " + inputType);
        }

        if (nIn <= 0 || override) {
            InputType.InputTypeRecurrent r = (InputType.InputTypeRecurrent) inputType;
            this.nIn = r.getSize();
        }
    }

    @Override
    public InputPreProcessor getPreProcessorForInputType(InputType inputType) {
        if(inputType == null ){
            throw new IllegalStateException("Invalid input for Convolution1D layer (layer name=\"" + getLayerName() + "\"): input is null");
        }

        return InputTypeUtil.getPreprocessorForInputTypeRnnLayers(inputType, getLayerName());
    }

    public static class Builder extends ConvolutionLayer.Builder {
        public Builder(int kernelSize, int stride, int padding) {
            this();
            this.kernelSize[0] = kernelSize;
            this.stride[0] = stride;
            this.padding[0] = padding;
        }

        public Builder(int kernelSize, int stride) {
            this();
            this.kernelSize[0] = kernelSize;
            this.stride[0] = stride;
        }

        public Builder(int kernelSize) {
            this();
            this.kernelSize[0] = kernelSize;
        }

        public Builder() {
            this.kernelSize = new int[] { 5, 1 };
            this.stride = new int[] { 1, 1 };
            this.padding = new int[] { 0, 0 };
        }

        /**
         * Size of the convolution
         * rows/columns
         * @param kernelSize the length of the kernel
         * @return
         */
        public Convolution1DLayer.Builder kernelSize(int kernelSize){
            this.kernelSize[0] = kernelSize;
            return this;
        }

        public Convolution1DLayer.Builder stride(int stride){
            this.stride[0] = stride;
            return this;
        }

        public Convolution1DLayer.Builder padding(int padding){
            this.padding[0] = padding;
            return this;
        }

        @Override
        @SuppressWarnings("unchecked")
        public Convolution1DLayer build() {
            ConvolutionUtils.validateCnnKernelStridePadding(kernelSize, stride, padding);

            return new Convolution1DLayer(this);
        }
    }
}

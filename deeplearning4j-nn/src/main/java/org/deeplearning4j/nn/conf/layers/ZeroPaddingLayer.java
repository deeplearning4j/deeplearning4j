package org.deeplearning4j.nn.conf.layers;

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.params.EmptyParamInitializer;
import org.deeplearning4j.optimize.api.IterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Arrays;
import java.util.Collection;
import java.util.Map;

/**
 * Zero padding layer for convolutional neural networks.
 * Allows padding to be done separately for top/bottom/left/right
 *
 * @author Alex Black
 */
@Data
@NoArgsConstructor
@EqualsAndHashCode(callSuper = true)
public class ZeroPaddingLayer extends Layer {

    private int[] padding;

    private ZeroPaddingLayer(Builder builder) {
        super(builder);
        this.padding = builder.padding;
    }

    @Override
    public org.deeplearning4j.nn.api.Layer instantiate(NeuralNetConfiguration conf,
                    Collection<IterationListener> iterationListeners, int layerIndex, INDArray layerParamsView,
                    boolean initializeParams) {
        org.deeplearning4j.nn.layers.convolution.ZeroPaddingLayer ret =
                        new org.deeplearning4j.nn.layers.convolution.ZeroPaddingLayer(conf);
        ret.setListeners(iterationListeners);
        ret.setIndex(layerIndex);
        Map<String, INDArray> paramTable = initializer().init(conf, layerParamsView, initializeParams);
        ret.setParamTable(paramTable);
        ret.setConf(conf);
        return ret;
    }

    @Override
    public ParamInitializer initializer() {
        return EmptyParamInitializer.getInstance();
    }

    @Override
    public InputType getOutputType(int layerIndex, InputType inputType) {
        int inH;
        int inW;
        int inDepth;
        if (inputType instanceof InputType.InputTypeConvolutional) {
            InputType.InputTypeConvolutional conv = (InputType.InputTypeConvolutional) inputType;
            inH = conv.getHeight();
            inW = conv.getWidth();
            inDepth = conv.getDepth();
        } else if (inputType instanceof InputType.InputTypeConvolutionalFlat) {
            InputType.InputTypeConvolutionalFlat conv = (InputType.InputTypeConvolutionalFlat) inputType;
            inH = conv.getHeight();
            inW = conv.getWidth();
            inDepth = conv.getDepth();
        } else {
            throw new IllegalStateException(
                            "Invalid input type: expected InputTypeConvolutional or InputTypeConvolutionalFlat."
                                            + " Got: " + inputType);
        }

        int outH = inH + padding[0] + padding[1];
        int outW = inW + padding[2] + padding[3];

        return InputType.convolutional(outH, outW, inDepth);
    }

    @Override
    public void setNIn(InputType inputType, boolean override) {
        //No op
    }

    @Override
    public InputPreProcessor getPreProcessorForInputType(InputType inputType) {
        if (inputType == null) {
            throw new IllegalStateException("Invalid input for ZeroPaddingLayer layer (layer name=\"" + getLayerName()
                            + "\"): input is null");
        }

        return InputTypeUtil.getPreProcessorForInputTypeCnnLayers(inputType, getLayerName());
    }

    @Override
    public double getL1ByParam(String paramName) {
        return 0;
    }

    @Override
    public double getL2ByParam(String paramName) {
        return 0;
    }

    @Override
    public double getLearningRateByParam(String paramName) {
        return learningRate;
    }

    @Override
    public boolean isPretrainParam(String paramName) {
        throw new UnsupportedOperationException("ZeroPaddingLayer does not contain parameters");
    }

    public static class Builder extends Layer.Builder<Builder> {

        private int[] padding = new int[] {0, 0, 0, 0}; //Padding: top, bottom, left, right

        /**
         *
         * @param padHeight Padding for both the top and bottom
         * @param padWidth  Padding for both the left and right
         */
        public Builder(int padHeight, int padWidth) {
            this(padHeight, padHeight, padWidth, padWidth);
        }

        public Builder(int padTop, int padBottom, int padLeft, int padRight) {
            this(new int[] {padTop, padBottom, padLeft, padRight});
        }

        public Builder(int[] padding) {
            this.padding = padding;
        }

        @Override
        @SuppressWarnings("unchecked")
        public ZeroPaddingLayer build() {
            for (int p : padding) {
                if (p < 0) {
                    throw new IllegalStateException(
                                    "Invalid zero padding layer config: padding [top, bottom, left, right]"
                                                    + " must be > 0 for all elements. Got: "
                                                    + Arrays.toString(padding));
                }
            }

            return new ZeroPaddingLayer(this);
        }
    }
}

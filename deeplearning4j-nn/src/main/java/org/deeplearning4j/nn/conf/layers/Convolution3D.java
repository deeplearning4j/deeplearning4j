package org.deeplearning4j.nn.conf.layers;

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import lombok.ToString;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.layers.convolution.Convolution3DLayer;
import org.deeplearning4j.nn.params.Convolution3DParamInitializer;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.util.Convolution3DUtils;
import org.deeplearning4j.util.ConvolutionUtils;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Collection;
import java.util.Map;

/**
 * 3D convolution layer configuration
 *
 * @author Max Pumperla
 */
@Data
@NoArgsConstructor
@ToString(callSuper = true)
@EqualsAndHashCode(callSuper = true)
public class Convolution3D extends ConvolutionLayer {

    private ConvolutionMode mode = ConvolutionMode.Same;  // in libnd4j: 0 - same mode, 1 - valid mode
    private boolean isNCDHW = true; // in libnd4j: 1 - NCDHW, 0 - NDHWC

    /**
     * 3-dimensional convolutional layer configuration
     * nIn in the input layer is the number of channels
     * nOut is the number of filters to be used in the net or in other words the depth
     * The builder specifies the filter/kernel size, the stride and padding
     * The pooling layer takes the kernel size
     */
    public Convolution3D(Builder builder) {
        super(builder);
        this.isNCDHW = builder.isNCDHW;
        this.convolutionMode = builder.convolutionMode;
    }

    public boolean hasBias() {
        return hasBias;
    }


    @Override
    public Convolution3D clone() {
        Convolution3D clone = (Convolution3D) super.clone();
        if (clone.kernelSize != null)
            clone.kernelSize = clone.kernelSize.clone();
        if (clone.stride != null)
            clone.stride = clone.stride.clone();
        if (clone.padding != null)
            clone.padding = clone.padding.clone();
        if (clone.dilation != null)
            clone.dilation = clone.dilation.clone();
        return clone;
    }

    @Override
    public Layer instantiate(NeuralNetConfiguration conf, Collection<IterationListener> iterationListeners,
                             int layerIndex, INDArray layerParamsView, boolean initializeParams) {
        LayerValidation.assertNInNOutSet("Convolution3D", getLayerName(), layerIndex, getNIn(), getNOut());

        Convolution3DLayer ret = new Convolution3DLayer(conf);
        ret.setListeners(iterationListeners);
        ret.setIndex(layerIndex);
        ret.setParamsViewArray(layerParamsView);
        Map<String, INDArray> paramTable = initializer().init(conf, layerParamsView, initializeParams);
        ret.setParamTable(paramTable);
        ret.setConf(conf);
        return ret;
    }

    @Override
    public ParamInitializer initializer() {
        return Convolution3DParamInitializer.getInstance();
    }

    @Override
    public InputType getOutputType(int layerIndex, InputType inputType) {
        if (inputType == null || inputType.getType() != InputType.Type.CNN3D) {
            throw new IllegalStateException("Invalid input for Convolution3D layer (layer name=\"" + getLayerName()
                    + "\"): Expected CNN3D input, got " + inputType);
        }
        return InputTypeUtil.getOutputTypeCnn3DLayers(inputType, kernelSize, stride, padding, dilation,
                convolutionMode, nOut, layerIndex, getLayerName(), Convolution3DLayer.class);
    }

    @Override
    public InputPreProcessor getPreProcessorForInputType(InputType inputType) {
        if (inputType == null) {
            throw new IllegalStateException("Invalid input for Convolution3D layer (layer name=\"" + getLayerName()
                    + "\"): input is null");
        }

        return InputTypeUtil.getPreProcessorForInputTypeCnn3DLayers(inputType, getLayerName());
    }


    @Override
    public void setNIn(InputType inputType, boolean override) {
        if (inputType == null || inputType.getType() != InputType.Type.CNN3D) {
            throw new IllegalStateException("Invalid input for Convolution 3D layer (layer name=\"" + getLayerName()
                    + "\"): Expected CNN3D input, got " + inputType);
        }

        if (nIn <= 0 || override) {
            InputType.InputTypeConvolutional3D c = (InputType.InputTypeConvolutional3D) inputType;
            this.nIn = c.getChannels();
        }
    }

    public static class Builder extends ConvolutionLayer.BaseConvBuilder<Builder> {

        private boolean isNCDHW = true; // in libnd4j: 1 - NCDHW, 0 - NDHWC
        protected int[] kernelSize = new int[] {2, 2, 2};
        protected int[] stride = new int[] {1, 1, 1};
        protected int[] padding = new int[] {0, 0, 0};
        protected int[] dilation = new int[]{0, 0, 0};


        public Builder(int[] kernelSize, int[] stride, int[] padding) {
            super(kernelSize, stride, padding);
        }

        public Builder(int[] kernelSize, int[] stride) {
            super(kernelSize, stride);
        }

        public Builder(int... kernelSize) {
            super(kernelSize);
        }

        public Builder() {
            super(new int[]{1, 1, 1},
                    new int[]{1, 1, 1},
                    new int[]{0, 0, 0},
                    new int[]{0, 0, 0});
        }


        public Builder kernelSize(int... kernelSize) {
            this.kernelSize = kernelSize;
            return this;
        }

        public Builder stride(int... stride) {
            this.stride = stride;
            return this;
        }

        public Builder padding(int... padding) {
            this.padding = padding;
            return this;
        }

        public Builder dilation(int... dilation) {
            this.dilation = dilation;
            return this;
        }

        public Builder convolutionMode(ConvolutionMode mode) {
            this.convolutionMode = mode;
            return this;
        }

        public Builder isNCDHW(boolean dataFormat) {
            this.isNCDHW = dataFormat;
            return this;
        }


        @Override
        @SuppressWarnings("unchecked")
        public Convolution3D build() {
            ConvolutionUtils.validateConvolutionModePadding(convolutionMode, padding);
            Convolution3DUtils.validateCnn3DKernelStridePadding(kernelSize, stride, padding);

            return new Convolution3D(this);
        }
    }

}
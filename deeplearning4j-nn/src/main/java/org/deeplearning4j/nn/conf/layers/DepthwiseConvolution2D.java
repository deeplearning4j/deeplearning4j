package org.deeplearning4j.nn.conf.layers;

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import lombok.ToString;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.layers.convolution.DepthwiseConvolution2DLayer;
import org.deeplearning4j.nn.params.DepthwiseConvolutionParamInitializer;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.util.ConvolutionUtils;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.*;

/**
 * 2D depth-wise convolution layer configuration.
 * <p>
 * Performs a channels-wise convolution, which
 * operates on each of the input maps separately. A channel multiplier is used to
 * specify the number of outputs per input map. This convolution
 * is carried out with the specified kernel sizes, stride and padding values.
 *
 * @author Max Pumperla
 */
@Data
@NoArgsConstructor
@ToString(callSuper = true)
@EqualsAndHashCode(callSuper = true)
public class DepthwiseConvolution2D extends ConvolutionLayer {

    int depthMultiplier;

    protected DepthwiseConvolution2D(Builder builder) {
        super(builder);
        this.hasBias = builder.hasBias;
        this.depthMultiplier = builder.depthMultiplier;
        this.convolutionMode = builder.convolutionMode;
        this.dilation = builder.dilation;
        if (builder.kernelSize.length != 2)
            throw new IllegalArgumentException("Kernel size of should be rows x columns (a 2d array)");
        this.kernelSize = builder.kernelSize;
        if (builder.stride.length != 2)
            throw new IllegalArgumentException("Stride should include stride for rows and columns (a 2d array)");
        this.stride = builder.stride;
        if (builder.padding.length != 2)
            throw new IllegalArgumentException("Padding should include padding for rows and columns (a 2d array)");
        this.padding = builder.padding;
        this.cudnnAlgoMode = builder.cudnnAlgoMode;
        this.cudnnFwdAlgo = builder.cudnnFwdAlgo;
        this.cudnnBwdFilterAlgo = builder.cudnnBwdFilterAlgo;
        this.cudnnBwdDataAlgo = builder.cudnnBwdDataAlgo;

        initializeConstraints(builder);
    }

    public boolean hasBias() {
        return hasBias;
    }

    @Override
    public DepthwiseConvolution2D clone() {
        DepthwiseConvolution2D clone = (DepthwiseConvolution2D) super.clone();
        if (clone.kernelSize != null)
            clone.kernelSize = clone.kernelSize.clone();
        if (clone.stride != null)
            clone.stride = clone.stride.clone();
        if (clone.padding != null)
            clone.padding = clone.padding.clone();
        return clone;
    }


    @Override
    public Layer instantiate(NeuralNetConfiguration conf, Collection<TrainingListener> trainingListeners,
                             int layerIndex, INDArray layerParamsView, boolean initializeParams) {
        LayerValidation.assertNInNOutSet(
                "DepthwiseConvolution2D", getLayerName(), layerIndex, getNIn(), getNOut());

        DepthwiseConvolution2DLayer ret = new DepthwiseConvolution2DLayer(conf);
        ret.setListeners(trainingListeners);
        ret.setIndex(layerIndex);
        ret.setParamsViewArray(layerParamsView);
        Map<String, INDArray> paramTable = initializer().init(conf, layerParamsView, initializeParams);
        ret.setParamTable(paramTable);
        ret.setConf(conf);

        return ret;
    }

    @Override
    public ParamInitializer initializer() {
        return DepthwiseConvolutionParamInitializer.getInstance();
    }

    @Override
    public InputType getOutputType(int layerIndex, InputType inputType) {
        if (inputType == null || inputType.getType() != InputType.Type.CNN) {
            throw new IllegalStateException("Invalid input for  depth-wise convolution layer (layer name=\""
                    + getLayerName() + "\"): Expected CNN input, got " + inputType);
        }


        return InputTypeUtil.getOutputTypeCnnLayers(inputType, kernelSize, stride, padding, dilation,
                convolutionMode, nOut, layerIndex, getLayerName(), DepthwiseConvolution2DLayer.class);
    }


    public static class Builder extends BaseConvBuilder<Builder> {

        public int depthMultiplier = 1;

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
            super();
        }

        /**
         * Set channels multiplier for depth-wise convolution
         *
         * @param depthMultiplier integer value, for each input map we get depthMultiplier
         *                        outputs in channels-wise step.
         * @return Builder
         */
        public Builder depthMultiplier(int depthMultiplier) {
            this.depthMultiplier = depthMultiplier;
            return this;
        }

        /**
         * Size of the convolution
         * rows/columns
         *
         * @param kernelSize the height and width of the
         *                   kernel
         * @return
         */
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

        @Override
        @SuppressWarnings("unchecked")
        public DepthwiseConvolution2D build() {
            ConvolutionUtils.validateConvolutionModePadding(convolutionMode, padding);
            ConvolutionUtils.validateCnnKernelStridePadding(kernelSize, stride, padding);

            return new DepthwiseConvolution2D(this);
        }
    }

}

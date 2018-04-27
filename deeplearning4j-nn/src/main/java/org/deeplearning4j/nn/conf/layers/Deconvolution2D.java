package org.deeplearning4j.nn.conf.layers;

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import lombok.ToString;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.layers.convolution.Deconvolution2DLayer;
import org.deeplearning4j.nn.params.DeconvolutionParamInitializer;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Collection;
import java.util.Map;

/**
 * 2D deconvolution layer configuration
 *
 * Deconvolutions are also known as transpose convolutions or fractionally strided convolutions.
 * In essence, deconvolutions swap forward and backward pass with regular 2D convolutions.
 *
 * See the paper by Matt Zeiler for details:
 * http://www.matthewzeiler.com/wp-content/uploads/2017/07/cvpr2010.pdf
 *
 * For an intuitive guide to convolution arithmetic and shapes, see:
 * https://arxiv.org/abs/1603.07285v1
 *
 * @author Max Pumperla
 */
@Data
@NoArgsConstructor
@ToString(callSuper = true)
@EqualsAndHashCode(callSuper = true)
public class Deconvolution2D extends ConvolutionLayer {

    /**
     * Deconvolution2D layer
     * nIn in the input layer is the number of channels
     * nOut is the number of filters to be used in the net or in other words the channels
     * The builder specifies the filter/kernel size, the stride and padding
     * The pooling layer takes the kernel size
     */
    protected Deconvolution2D(BaseConvBuilder<?> builder) {
        super(builder);
        initializeConstraints(builder);
    }

    public boolean hasBias(){
        return hasBias;
    }

    @Override
    public Deconvolution2D clone() {
        Deconvolution2D clone = (Deconvolution2D) super.clone();
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
        LayerValidation.assertNInNOutSet("Deconvolution2D", getLayerName(), layerIndex, getNIn(), getNOut());

        org.deeplearning4j.nn.layers.convolution.Deconvolution2DLayer ret =
                new org.deeplearning4j.nn.layers.convolution.Deconvolution2DLayer(conf);
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
        return DeconvolutionParamInitializer.getInstance();
    }

    @Override
    public InputType getOutputType(int layerIndex, InputType inputType) {
        if (inputType == null || inputType.getType() != InputType.Type.CNN) {
            throw new IllegalStateException("Invalid input for Convolution layer (layer name=\"" + getLayerName()
                    + "\"): Expected CNN input, got " + inputType);
        }

        return InputTypeUtil.getOutputTypeDeconvLayer(inputType, kernelSize, stride, padding, dilation,
                convolutionMode, nOut, layerIndex, getLayerName(), Deconvolution2DLayer.class);
    }

    public static class Builder extends BaseConvBuilder<Builder> {

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
         * Set the convolution mode for the Convolution layer.
         * See {@link ConvolutionMode} for more details
         *
         * @param convolutionMode    Convolution mode for layer
         */
        public Builder convolutionMode(ConvolutionMode convolutionMode) {
            if(convolutionMode == ConvolutionMode.Same){
                throw new IllegalArgumentException("SAME mode not yet supported for deconvolution layer");
            }
            return super.convolutionMode(convolutionMode);
        }

        /**
         * Size of the convolution
         * rows/columns
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
        public Deconvolution2D build() {
            return new Deconvolution2D(this);
        }
    }

}
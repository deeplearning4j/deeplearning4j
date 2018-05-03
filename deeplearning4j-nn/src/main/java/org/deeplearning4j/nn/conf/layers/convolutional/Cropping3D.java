package org.deeplearning4j.nn.conf.layers.convolutional;

import com.google.common.base.Preconditions;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import lombok.NonNull;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.InputTypeUtil;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.layers.NoParamLayer;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.nn.layers.convolution.Cropping2DLayer;
import org.deeplearning4j.nn.layers.convolution.Cropping3DLayer;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.util.ConvolutionUtils;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Arrays;
import java.util.Collection;
import java.util.Map;

/**
 * Cropping layer for convolutional (3d) neural networks.
 * Allows cropping to be done separately for upper and lower bounds of
 * depth, height and width dimensions.
 *
 * @author Max Pumperla
 */
@Data
@NoArgsConstructor
@EqualsAndHashCode(callSuper = true)
public class Cropping3D extends NoParamLayer {

    private int[] cropping;

    /**
     * @param cropDepth  Amount of cropping to apply to both depth boundaries of the input activations
     * @param cropHeight Amount of cropping to apply to both height boundaries of the input activations
     * @param cropWidth  Amount of cropping to apply to both width boundaries of the input activations
     */
    public Cropping3D(int cropDepth, int cropHeight, int cropWidth) {
        this(cropDepth, cropDepth, cropHeight, cropHeight, cropWidth, cropWidth);
    }

    /**
     * @param cropLeftD  Amount of cropping to apply to the left of the depth dimension
     * @param cropRightD Amount of cropping to apply to the right of the depth dimension
     * @param cropLeftH  Amount of cropping to apply to the left of the height dimension
     * @param cropRightH Amount of cropping to apply to the right of the height dimension
     * @param cropLeftW  Amount of cropping to apply to the left of the width dimension
     * @param cropRightW Amount of cropping to apply to the right of the width dimension
     */
    public Cropping3D(int cropLeftD, int cropRightD, int cropLeftH, int cropRightH, int cropLeftW, int cropRightW) {
        this(new Builder(cropLeftD, cropRightD, cropLeftH, cropRightH, cropLeftW, cropRightW));
    }

    public Cropping3D(int[] cropping) {
        this(new Builder(cropping));
    }

    protected Cropping3D(Builder builder) {
        super(builder);
        this.cropping = builder.cropping;
    }

    @Override
    public org.deeplearning4j.nn.api.Layer instantiate(NeuralNetConfiguration conf,
                                                       Collection<TrainingListener> iterationListeners,
                                                       int layerIndex, INDArray layerParamsView,
                                                       boolean initializeParams) {
        Cropping3DLayer ret = new Cropping3DLayer(conf);
        ret.setListeners(iterationListeners);
        ret.setIndex(layerIndex);
        Map<String, INDArray> paramTable = initializer().init(conf, layerParamsView, initializeParams);
        ret.setParamTable(paramTable);
        ret.setConf(conf);
        return ret;
    }

    @Override
    public InputType getOutputType(int layerIndex, InputType inputType) {
        if (inputType == null || inputType.getType() != InputType.Type.CNN3D) {
            throw new IllegalStateException("Invalid input for 3D cropping layer (layer index = " + layerIndex
                    + ", layer name = \"" + getLayerName() + "\"): expect CNN3D input type with size > 0. Got: "
                    + inputType);
        }
        InputType.InputTypeConvolutional3D c = (InputType.InputTypeConvolutional3D) inputType;
        return InputType.convolutional3D(
                c.getDepth()  - cropping[0] - cropping[1],
                c.getHeight()  - cropping[2] - cropping[3],
                c.getWidth()  - cropping[4] - cropping[5],
                c.getChannels());
    }

    @Override
    public InputPreProcessor getPreProcessorForInputType(InputType inputType) {
        Preconditions.checkArgument(inputType != null, "Invalid input for Cropping3D " +
                "layer (layer name=\"" + getLayerName() + "\"): InputType is null");
        return InputTypeUtil.getPreProcessorForInputTypeCnn3DLayers(inputType, getLayerName());
    }

    @Override
    public LayerMemoryReport getMemoryReport(InputType inputType) {
        return null;
    }


    public static class Builder extends Layer.Builder<Builder> {

        private int[] cropping = new int[]{0, 0, 0, 0, 0, 0};

        public Builder() {

        }

        /**
         * @param cropping Cropping amount, must be length 3 or 6 array, i.e. either
         *                 crop depth, crop height, crop width or
         *                 crop left depth, crop right depth, crop left height, crop right height, crop left width,
         *                 crop right width
         */
        public Builder(@NonNull int[] cropping) {
            Preconditions.checkArgument(cropping.length == 6 || cropping.length == 3,
                    "Either 3 or 6 cropping values, got "
                            + cropping.length + " values: " + Arrays.toString(cropping));
            if (cropping.length == 3) {
                this.cropping = new int[]{cropping[0], cropping[0], cropping[1], cropping[1], cropping[2], cropping[2]};
            } else {
                this.cropping = cropping;
            }
        }

        /**
         * @param cropDepth  Amount of cropping to apply to both depth boundaries of the input activations
         * @param cropHeight Amount of cropping to apply to both height boundaries of the input activations
         * @param cropWidth  Amount of cropping to apply to both width boundaries of the input activations
         */
        public Builder(int cropDepth, int cropHeight, int cropWidth) {
            this(cropDepth, cropDepth, cropHeight, cropHeight, cropWidth, cropWidth);
        }

        /**
         * @param cropLeftD  Amount of cropping to apply to the left of the depth dimension
         * @param cropRightD Amount of cropping to apply to the right of the depth dimension
         * @param cropLeftH  Amount of cropping to apply to the left of the height dimension
         * @param cropRightH Amount of cropping to apply to the right of the height dimension
         * @param cropLeftW  Amount of cropping to apply to the left of the width dimension
         * @param cropRightW Amount of cropping to apply to the right of the width dimension
         */
        public Builder(int cropLeftD, int cropRightD, int cropLeftH, int cropRightH, int cropLeftW, int cropRightW) {
            this.cropping = new int[]{cropLeftD, cropRightD, cropLeftH, cropRightH, cropLeftW, cropRightW};
            Preconditions.checkArgument(cropLeftD >= 0 && cropLeftH >= 0 && cropLeftW >= 0
                            && cropRightD >= 0 && cropRightH >= 0 && cropRightW >= 0,
                    "Invalid arguments: crop dimensions must be > 0. Got " + Arrays.toString(this.cropping));
        }

        public Cropping3D build() {
            return new Cropping3D(this);
        }
    }
}

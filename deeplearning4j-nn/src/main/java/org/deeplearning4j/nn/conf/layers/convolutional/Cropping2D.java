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
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.util.ConvolutionUtils;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Arrays;
import java.util.Collection;
import java.util.Map;

/**
 * Cropping layer for convolutional (2d) neural networks.
 * Allows cropping to be done separately for top/bottom/left/right
 *
 * @author Alex Black
 */
@Data
@NoArgsConstructor
@EqualsAndHashCode(callSuper = true)
public class Cropping2D extends NoParamLayer {

    private int[] cropping;

    /**
     * @param cropTopBottom Amount of cropping to apply to both the top and the bottom of the input activations
     * @param cropLeftRight Amount of cropping to apply to both the left and the right of the input activations
     */
    public Cropping2D(int cropTopBottom, int cropLeftRight) {
        this(cropTopBottom, cropTopBottom, cropLeftRight, cropLeftRight);
    }

    /**
     * @param cropTop    Amount of cropping to apply to the top of the input activations
     * @param cropBottom Amount of cropping to apply to the bottom of the input activations
     * @param cropLeft   Amount of cropping to apply to the left of the input activations
     * @param cropRight  Amount of cropping to apply to the right of the input activations
     */
    public Cropping2D(int cropTop, int cropBottom, int cropLeft, int cropRight) {
        this(new Builder(cropTop, cropBottom, cropLeft, cropRight));
    }

    public Cropping2D(int[] cropping) {
        this(new Builder(cropping));
    }

    protected Cropping2D(Builder builder) {
        super(builder);
        this.cropping = builder.cropping;
    }

    @Override
    public org.deeplearning4j.nn.api.Layer instantiate(NeuralNetConfiguration conf, Collection<TrainingListener> trainingListeners,
                                                       int layerIndex, INDArray layerParamsView, boolean initializeParams) {
        Cropping2DLayer ret = new Cropping2DLayer(conf);
        ret.setListeners(trainingListeners);
        ret.setIndex(layerIndex);
        Map<String, INDArray> paramTable = initializer().init(conf, layerParamsView, initializeParams);
        ret.setParamTable(paramTable);
        ret.setConf(conf);
        return ret;
    }

    @Override
    public InputType getOutputType(int layerIndex, InputType inputType) {
        int[] hwd = ConvolutionUtils.getHWDFromInputType(inputType);
        int outH = hwd[0] - cropping[0] - cropping[1];
        int outW = hwd[1] - cropping[2] - cropping[3];

        return InputType.convolutional(outH, outW, hwd[2]);
    }

    @Override
    public InputPreProcessor getPreProcessorForInputType(InputType inputType) {
        Preconditions.checkArgument(inputType != null, "Invalid input for Cropping2D layer (layer name=\""
                + getLayerName() + "\"): InputType is null");
        return InputTypeUtil.getPreProcessorForInputTypeCnnLayers(inputType, getLayerName());
    }

    @Override
    public LayerMemoryReport getMemoryReport(InputType inputType) {
        return null;
    }


    public static class Builder extends Layer.Builder<Builder> {

        private int[] cropping = new int[]{0, 0, 0, 0};

        public Builder() {

        }

        /**
         * @param cropping Cropping amount for top/bottom/left/right (in that order). Must be length 4 array.
         */
        public Builder(@NonNull int[] cropping) {
            Preconditions.checkArgument(cropping.length == 4 || cropping.length == 2,
                    "Either 2 or 4 cropping values,  i.e. (top/bottom. left/right) or (top, bottom," +
                            " left, right) must be provided. Got " + cropping.length + " values: " + Arrays.toString(cropping));
            if (cropping.length == 2) {
                this.cropping = new int[]{cropping[0], cropping[0], cropping[1], cropping[1]};
            } else {
                this.cropping = cropping;
            }
        }

        /**
         * @param cropTopBottom Amount of cropping to apply to both the top and the bottom of the input activations
         * @param cropLeftRight Amount of cropping to apply to both the left and the right of the input activations
         */
        public Builder(int cropTopBottom, int cropLeftRight) {
            this(cropTopBottom, cropTopBottom, cropLeftRight, cropLeftRight);
        }

        /**
         * @param cropTop    Amount of cropping to apply to the top of the input activations
         * @param cropBottom Amount of cropping to apply to the bottom of the input activations
         * @param cropLeft   Amount of cropping to apply to the left of the input activations
         * @param cropRight  Amount of cropping to apply to the right of the input activations
         */
        public Builder(int cropTop, int cropBottom, int cropLeft, int cropRight) {
            this.cropping = new int[]{cropTop, cropBottom, cropLeft, cropRight};
            Preconditions.checkArgument(cropTop >= 0 && cropBottom >= 0 && cropLeft >= 0 && cropRight >= 0,
                    "Invalid arguments: crop dimensions must be > 0. Got [t,b,l,r] = " + Arrays.toString(this.cropping));
        }

        public Cropping2D build() {
            return new Cropping2D(this);
        }
    }
}

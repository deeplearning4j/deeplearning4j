package org.deeplearning4j.nn.conf.layers.convolutional;

import com.google.common.base.Preconditions;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.InputTypeUtil;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.layers.NoParamLayer;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.optimize.api.IterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Arrays;
import java.util.Collection;

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

    public Cropping2D(int cropTopBottom, int cropLeftRight){
        this(cropTopBottom, cropTopBottom, cropLeftRight, cropLeftRight);
    }

    public Cropping2D(int cropTop, int cropBottom, int cropLeft, int cropRight){
        this(new Builder(cropTop, cropBottom, cropLeft, cropRight));
    }

    protected Cropping2D(Builder builder){
        super(builder);
        this.cropping = builder.cropping;
    }

    @Override
    public org.deeplearning4j.nn.api.Layer instantiate(NeuralNetConfiguration conf, Collection<IterationListener> iterationListeners, int layerIndex, INDArray layerParamsView, boolean initializeParams) {
        return null;
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

        int outH = inH - cropping[0] - cropping[1];
        int outW = inW - cropping[2] - cropping[3];

        return InputType.convolutional(outH, outW, inDepth);
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

        public Builder(){

        }

        public Builder(int cropTopBottom, int cropLeftRight){
            this(cropTopBottom, cropTopBottom, cropLeftRight, cropLeftRight);
        }

        public Builder(int cropTop, int cropBottom, int cropLeft, int cropRight){
            this.cropping = new int[]{cropTop, cropBottom, cropLeft, cropRight};
            Preconditions.checkArgument(cropTop >= 0 && cropBottom >= 0 && cropLeft >= 0 && cropRight >= 0,
                    "Invalid arguments: crop dimensions must be > 0. Got [t,b,l,r] = " + Arrays.toString(this.cropping));
        }

        public Cropping2D build(){
            return new Cropping2D(this);
        }
    }
}

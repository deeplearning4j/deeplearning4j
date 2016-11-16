package org.deeplearning4j.nn.conf.layers;

import lombok.*;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.params.ConvolutionParamInitializer;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.util.LayerValidation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.convolution.Convolution;

import java.util.Collection;
import java.util.Map;

/**
 * @author Adam Gibson
 */
@Data @NoArgsConstructor
@ToString(callSuper = true)
@EqualsAndHashCode(callSuper = true)
public class ConvolutionLayer extends FeedForwardLayer {
//    protected Convolution.Type convolutionType;
    protected ConvolutionMode convolutionMode = ConvolutionMode.Truncate;       //Default to truncate here - default for 0.6.0 and earlier networks on JSON deserialization
    protected int[] kernelSize; // Square filter
    protected int[] stride; // Default is 2. Down-sample by a factor of 2
    protected int[] padding;

    public enum AlgoMode { NO_WORKSPACE, PREFER_FASTEST }

    /** Defaults to "PREFER_FASTEST", but "NO_WORKSPACE" uses less memory. */
    protected AlgoMode cudnnAlgoMode = AlgoMode.PREFER_FASTEST;

    /**
    * ConvolutionLayer
    * nIn in the input layer is the number of channels
    * nOut is the number of filters to be used in the net or in other words the depth
    * The builder specifies the filter/kernel size, the stride and padding
    * The pooling layer takes the kernel size
    */
    private ConvolutionLayer(Builder builder) {
    	super(builder);
//        this.convolutionType = builder.convolutionType;
        this.convolutionMode = builder.convolutionMode;
        if(builder.kernelSize.length != 2)
            throw new IllegalArgumentException("Kernel size of should be rows x columns (a 2d array)");
        this.kernelSize = builder.kernelSize;
        if(builder.stride.length != 2)
            throw new IllegalArgumentException("Stride should include stride for rows and columns (a 2d array)");
        this.stride = builder.stride;
        if(builder.padding.length != 2)
            throw new IllegalArgumentException("Padding should include padding for rows and columns (a 2d array)");
        this.padding = builder.padding;
        this.cudnnAlgoMode = builder.cudnnAlgoMode;
    }

    @Override
    public ConvolutionLayer clone() {
        ConvolutionLayer clone = (ConvolutionLayer) super.clone();
        if(clone.kernelSize != null) clone.kernelSize = clone.kernelSize.clone();
        if(clone.stride != null) clone.stride = clone.stride.clone();
        if(clone.padding != null) clone.padding = clone.padding.clone();
        return clone;
    }

    @Override
    public Layer instantiate(NeuralNetConfiguration conf, Collection<IterationListener> iterationListeners, int layerIndex, INDArray layerParamsView, boolean initializeParams) {
        LayerValidation.assertNInNOutSet("ConvolutionLayer", getLayerName(), layerIndex, getNIn(), getNOut());

        org.deeplearning4j.nn.layers.convolution.ConvolutionLayer ret
                = new org.deeplearning4j.nn.layers.convolution.ConvolutionLayer(conf);
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
        return ConvolutionParamInitializer.getInstance();
    }

    @Override
    public InputType getOutputType(int layerIndex, InputType inputType) {
        if(inputType == null || inputType.getType() != InputType.Type.CNN){
            throw new IllegalStateException("Invalid input for Convolution layer (layer name=\"" + getLayerName() + "\"): Expected CNN input, got " + inputType);
        }

        return InputTypeUtil.getOutputTypeCnnLayers(inputType, kernelSize, stride, padding, convolutionMode, nOut,
                layerIndex, getLayerName(), ConvolutionLayer.class);
    }

    @Override
    public void setNIn(InputType inputType, boolean override){
        if(inputType == null || inputType.getType() != InputType.Type.CNN){
            throw new IllegalStateException("Invalid input for Convolution layer (layer name=\"" + getLayerName() + "\"): Expected CNN input, got " + inputType);
        }

        if(nIn <= 0 || override){
            InputType.InputTypeConvolutional c = (InputType.InputTypeConvolutional)inputType;
            this.nIn = c.getDepth();
        }
    }

    @Override
    public InputPreProcessor getPreProcessorForInputType(InputType inputType) {
        if(inputType == null ){
            throw new IllegalStateException("Invalid input for Convolution layer (layer name=\"" + getLayerName() + "\"): input is null");
        }

        return InputTypeUtil.getPreProcessorForInputTypeCnnLayers(inputType, getLayerName());
    }

    @Override
    public double getL1ByParam(String paramName) {
        switch (paramName){
            case ConvolutionParamInitializer.WEIGHT_KEY:
                return l1;
            case ConvolutionParamInitializer.BIAS_KEY:
                return 0.0;
            default:
                throw new IllegalArgumentException("Unknown parameter name: \"" + paramName + "\"");
        }
    }

    @Override
    public double getL2ByParam(String paramName) {
        switch (paramName){
            case ConvolutionParamInitializer.WEIGHT_KEY:
                return l2;
            case ConvolutionParamInitializer.BIAS_KEY:
                return 0.0;
            default:
                throw new IllegalArgumentException("Unknown parameter name: \"" + paramName + "\"");
        }
    }

    @Override
    public double getLearningRateByParam(String paramName) {
        switch (paramName){
            case ConvolutionParamInitializer.WEIGHT_KEY:
                return learningRate;
            case ConvolutionParamInitializer.BIAS_KEY:
                if(!Double.isNaN(biasLearningRate)){
                    //Bias learning rate has been explicitly set
                    return biasLearningRate;
                } else {
                    return learningRate;
                }
            default:
                throw new IllegalArgumentException("Unknown parameter name: \"" + paramName + "\"");
        }
    }

    @AllArgsConstructor
    public static class Builder extends FeedForwardLayer.Builder<Builder> {
//        private Convolution.Type convolutionType = Convolution.Type.VALID;
        private ConvolutionMode convolutionMode = null;
        private int[] kernelSize = new int[] {5,5};
        private int[] stride = new int[] {1,1};
        private int[] padding = new int[] {0, 0};
        private AlgoMode cudnnAlgoMode = AlgoMode.PREFER_FASTEST;


        public Builder(int[] kernelSize, int[] stride, int[] padding) {
            this.kernelSize = kernelSize;
            this.stride = stride;
            this.padding = padding;
        }

        public Builder(int[] kernelSize, int[] stride) {
            this.kernelSize = kernelSize;
            this.stride = stride;
        }

        public Builder(int... kernelSize) {
            this.kernelSize = kernelSize;
        }

        public Builder() {}

        /**
         * @deprecated Use {@link #convolutionMode}
         */
        @Deprecated
        public Builder convolutionType(Convolution.Type convolutionType) {
//            this.convolutionType = convolutionType;
            return this;
        }

        /**
         * Set the convolution mode for the Convolution layer.
         * See {@link ConvolutionMode} for more details
         *
         * @param convolutionMode    Convolution mode for layer
         */
        public Builder convolutionMode(ConvolutionMode convolutionMode){
            this.convolutionMode = convolutionMode;
            return this;
        }

        /**
         * Size of the convolution
         * rows/columns
         * @param kernelSize the height and width of the
         *                   kernel
         * @return
         */
        public Builder kernelSize(int... kernelSize){
            this.kernelSize = kernelSize;
            return this;
        }

        public Builder stride(int... stride){
            this.stride = stride;
            return this;
        }

        public Builder padding(int... padding){
            this.padding = padding;
            return this;
        }

        /** Defaults to "PREFER_FASTEST", but "NO_WORKSPACE" uses less memory. */
        public Builder cudnnAlgoMode(AlgoMode cudnnAlgoMode){
            this.cudnnAlgoMode = cudnnAlgoMode;
            return this;
        }

        @Override
        @SuppressWarnings("unchecked")
        public ConvolutionLayer build() {
            return new ConvolutionLayer(this);
        }
    }
}

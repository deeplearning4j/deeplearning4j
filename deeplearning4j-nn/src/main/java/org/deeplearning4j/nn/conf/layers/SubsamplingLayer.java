package org.deeplearning4j.nn.conf.layers;

import lombok.*;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.params.EmptyParamInitializer;
import org.deeplearning4j.optimize.api.IterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Collection;
import java.util.Map;

/**
 * Subsampling layer also referred to as pooling in convolution neural nets
 *
 *  Supports the following pooling types:
 *     MAX
 *     AVG
 *     NON
 * @author Adam Gibson
 */

@Data @NoArgsConstructor
@ToString(callSuper = true)
@EqualsAndHashCode(callSuper = true)
public class SubsamplingLayer extends Layer {

    protected ConvolutionMode convolutionMode = ConvolutionMode.Truncate;          //Default to truncate here - default for 0.6.0 and earlier networks on JSON deserialization
    protected org.deeplearning4j.nn.conf.layers.PoolingType poolingType;
    protected int[] kernelSize; // Same as filter size from the last conv layer
    protected int[] stride; // Default is 2. Down-sample by a factor of 2
    protected int[] padding;
    protected int pnorm;
    protected double eps;

    public enum PoolingType {
        MAX, AVG, SUM, PNORM, NONE;

        public org.deeplearning4j.nn.conf.layers.PoolingType toPoolingType(){
            switch (this){
                case MAX:
                    return org.deeplearning4j.nn.conf.layers.PoolingType.MAX;
                case AVG:
                    return org.deeplearning4j.nn.conf.layers.PoolingType.AVG;
                case SUM:
                    return org.deeplearning4j.nn.conf.layers.PoolingType.SUM;
                case PNORM:
                    return org.deeplearning4j.nn.conf.layers.PoolingType.PNORM;
                case NONE:
                    return org.deeplearning4j.nn.conf.layers.PoolingType.NONE;
            }
            throw new UnsupportedOperationException("Unknown/not supported pooling type: " + this);
        }
    }

    private SubsamplingLayer(Builder builder) {
        super(builder);
        this.poolingType = builder.poolingType;
        if(builder.kernelSize.length != 2)
            throw new IllegalArgumentException("Kernel size of should be rows x columns (a 2d array)");
        this.kernelSize = builder.kernelSize;
        if(builder.stride.length != 2)
            throw new IllegalArgumentException("Invalid stride, must be length 2");
        this.stride = builder.stride;
        this.padding = builder.padding;
        this.convolutionMode = builder.convolutionMode;
        this.pnorm = builder.pnorm;
        this.eps = builder.eps;
    }

    @Override
    public SubsamplingLayer clone() {
        SubsamplingLayer clone = (SubsamplingLayer) super.clone();

        if(clone.kernelSize != null) clone.kernelSize = clone.kernelSize.clone();
        if(clone.stride != null) clone.stride = clone.stride.clone();
        if(clone.padding != null) clone.padding = clone.padding.clone();
        return clone;
    }

    @Override
    public org.deeplearning4j.nn.api.Layer instantiate(NeuralNetConfiguration conf, Collection<IterationListener> iterationListeners, int layerIndex, INDArray layerParamsView, boolean initializeParams) {
        org.deeplearning4j.nn.layers.convolution.subsampling.SubsamplingLayer ret
                = new org.deeplearning4j.nn.layers.convolution.subsampling.SubsamplingLayer(conf);
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
        return EmptyParamInitializer.getInstance();
    }

    @Override
    public InputType getOutputType(int layerIndex, InputType inputType) {
        if(inputType == null || inputType.getType() != InputType.Type.CNN){
            throw new IllegalStateException("Invalid input for Subsampling layer (layer name=\"" + getLayerName() + "\"): Expected CNN input, got " + inputType);
        }

        return InputTypeUtil.getOutputTypeCnnLayers(inputType, kernelSize, stride, padding, convolutionMode,
                ((InputType.InputTypeConvolutional) inputType).getDepth(), layerIndex, getLayerName(), SubsamplingLayer.class);
    }

    @Override
    public void setNIn(InputType inputType, boolean override){
        //No op: subsampling layer doesn't have nIn value
    }

    @Override
    public InputPreProcessor getPreProcessorForInputType(InputType inputType) {
        if(inputType == null ){
            throw new IllegalStateException("Invalid input for Subsampling layer (layer name=\"" + getLayerName() + "\"): input is null");
        }

        return InputTypeUtil.getPreProcessorForInputTypeCnnLayers(inputType, getLayerName());
    }

    @Override
    public double getL1ByParam(String paramName) {
        //Not applicable
        return 0;
    }

    @Override
    public double getL2ByParam(String paramName) {
        //Not applicable
        return 0;
    }

    @Override
    public double getLearningRateByParam(String paramName) {
        //Not applicable
        return 0;
    }

    public int getPnorm() {
        return pnorm;
    }

    public double getEps() { return eps; }

    @AllArgsConstructor
    public static class Builder extends Layer.Builder<Builder> {
        private org.deeplearning4j.nn.conf.layers.PoolingType poolingType = org.deeplearning4j.nn.conf.layers.PoolingType.MAX;
        private int[] kernelSize = new int[] {1, 1}; // Same as filter size from the last conv layer
        private int[] stride = new int[] {2, 2}; // Default is 2. Down-sample by a factor of 2
        private int[] padding = new int[] {0, 0};
        private ConvolutionMode convolutionMode = null;
        private int pnorm;
        private double eps = 1e-8;

        public Builder(PoolingType poolingType, int[] kernelSize, int[] stride) {
            this.poolingType = poolingType.toPoolingType();
            this.kernelSize = kernelSize;
            this.stride = stride;
        }

        public Builder(PoolingType poolingType, int[] kernelSize) {
            this.poolingType = poolingType.toPoolingType();
            this.kernelSize = kernelSize;
        }

        public Builder(PoolingType poolingType, int[] kernelSize, int[] stride, int[] padding){
            this.poolingType = poolingType.toPoolingType();
            this.kernelSize = kernelSize;
            this.stride = stride;
            this.padding = padding;
        }

        public Builder(org.deeplearning4j.nn.conf.layers.PoolingType poolingType, int[] kernelSize) {
            this.poolingType = poolingType;
            this.kernelSize = kernelSize;
        }

        public Builder(org.deeplearning4j.nn.conf.layers.PoolingType poolingType, int[] kernelSize, int[] stride, int[] padding){
            this.poolingType = poolingType;
            this.kernelSize = kernelSize;
            this.stride = stride;
            this.padding = padding;
        }

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

        public Builder(PoolingType poolingType) {
            this.poolingType = poolingType.toPoolingType();
        }

        public Builder(org.deeplearning4j.nn.conf.layers.PoolingType poolingType){
            this.poolingType = poolingType;
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

        public Builder() {}

        @Override
        @SuppressWarnings("unchecked")
        public SubsamplingLayer build() {
            if(poolingType==org.deeplearning4j.nn.conf.layers.PoolingType.PNORM && pnorm <= 0) throw new IllegalStateException("Incorrect Subsampling config: p-norm must be set when using PoolingType.PNORM");

            return new SubsamplingLayer(this);
        }

        public Builder poolingType(PoolingType poolingType){
            this.poolingType = poolingType.toPoolingType();
            return this;
        }

        /**
         * Kernel size
         *
         * @param kernelSize    kernel size in height and width dimensions
         */
        public Builder kernelSize(int... kernelSize){
            if(kernelSize.length != 2) throw new IllegalArgumentException("Invalid input: must be length 2");
            this.kernelSize = kernelSize;
            return this;
        }

        /**
         * Stride
         *
         * @param stride    stride in height and width dimensions
         */
        public Builder stride(int... stride){
            if(stride.length != 2) throw new IllegalArgumentException("Invalid input: must be length 2");
            this.stride = stride;
            return this;
        }

        /**
         * Padding
         *
         * @param padding    padding in the height and width dimensions
         */
        public Builder padding(int... padding){
            if(padding.length != 2) throw new IllegalArgumentException("Invalid input: must be length 2");
            this.padding = padding;
            return this;
        }

        public Builder pnorm(int pnorm){
            if(pnorm <= 0) throw new IllegalArgumentException("Invalid input: p-norm value must be greater than 0");
            this.pnorm = pnorm;
            return this;
        }

        public Builder eps(double eps){
            if(eps <= 0) throw new IllegalArgumentException("Invalid input: epsilon for p-norm must be greater than 0");
            this.eps = eps;
            return this;
        }
    }

}

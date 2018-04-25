package org.deeplearning4j.nn.conf.layers;

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import lombok.ToString;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.nn.conf.memory.MemoryReport;
import org.deeplearning4j.nn.params.EmptyParamInitializer;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.util.ConvolutionUtils;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Collection;
import java.util.Map;

/**
 * Subsampling layer also referred to as pooling in convolution neural nets
 *
 *  Supports the following pooling types: MAX, AVG, SUM, PNORM, NONE
 * @author Adam Gibson
 */

@Data
@NoArgsConstructor
@ToString(callSuper = true)
@EqualsAndHashCode(callSuper = true)
public class SubsamplingLayer extends Layer {

    protected ConvolutionMode convolutionMode = ConvolutionMode.Truncate; //Default to truncate here - default for 0.6.0 and earlier networks on JSON deserialization
    protected org.deeplearning4j.nn.conf.layers.PoolingType poolingType;
    protected int[] kernelSize; // Same as filter size from the last conv layer
    protected int[] stride; // Default is 2. Down-sample by a factor of 2
    protected int[] padding;
    protected int[] dilation = new int[]{1,1};
    protected int pnorm;
    protected double eps;
    protected boolean cudnnAllowFallback = true;

    public enum PoolingType {
        MAX, AVG, SUM, PNORM;

        public org.deeplearning4j.nn.conf.layers.PoolingType toPoolingType() {
            switch (this) {
                case MAX:
                    return org.deeplearning4j.nn.conf.layers.PoolingType.MAX;
                case AVG:
                    return org.deeplearning4j.nn.conf.layers.PoolingType.AVG;
                case SUM:
                    return org.deeplearning4j.nn.conf.layers.PoolingType.SUM;
                case PNORM:
                    return org.deeplearning4j.nn.conf.layers.PoolingType.PNORM;
            }
            throw new UnsupportedOperationException("Unknown/not supported pooling type: " + this);
        }
    }

    protected SubsamplingLayer(BaseSubsamplingBuilder builder) {
        super(builder);
        this.poolingType = builder.poolingType;
        if (builder.kernelSize.length != 2)
            throw new IllegalArgumentException("Kernel size of should be rows x columns (a 2d array)");
        this.kernelSize = builder.kernelSize;
        if (builder.stride.length != 2)
            throw new IllegalArgumentException("Invalid stride, must be length 2");
        this.stride = builder.stride;
        this.padding = builder.padding;
        this.convolutionMode = builder.convolutionMode;
        if(builder instanceof Builder){
            this.dilation = ((Builder)builder).dilation;
        }
        this.pnorm = builder.pnorm;
        this.eps = builder.eps;
        this.cudnnAllowFallback = builder.cudnnAllowFallback;
    }

    @Override
    public SubsamplingLayer clone() {
        SubsamplingLayer clone = (SubsamplingLayer) super.clone();

        if (clone.kernelSize != null)
            clone.kernelSize = clone.kernelSize.clone();
        if (clone.stride != null)
            clone.stride = clone.stride.clone();
        if (clone.padding != null)
            clone.padding = clone.padding.clone();
        return clone;
    }

    @Override
    public org.deeplearning4j.nn.api.Layer instantiate(NeuralNetConfiguration conf,
                    Collection<TrainingListener> trainingListeners, int layerIndex, INDArray layerParamsView,
                    boolean initializeParams) {
        org.deeplearning4j.nn.layers.convolution.subsampling.SubsamplingLayer ret =
                        new org.deeplearning4j.nn.layers.convolution.subsampling.SubsamplingLayer(conf);
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
        return EmptyParamInitializer.getInstance();
    }

    @Override
    public InputType getOutputType(int layerIndex, InputType inputType) {
        if (inputType == null || inputType.getType() != InputType.Type.CNN) {
            throw new IllegalStateException("Invalid input for Subsampling layer (layer name=\"" + getLayerName()
                            + "\"): Expected CNN input, got " + inputType);
        }

        return InputTypeUtil.getOutputTypeCnnLayers(inputType, kernelSize, stride, padding, dilation, convolutionMode,
                        ((InputType.InputTypeConvolutional) inputType).getChannels(), layerIndex, getLayerName(),
                        SubsamplingLayer.class);
    }

    @Override
    public void setNIn(InputType inputType, boolean override) {
        //No op: subsampling layer doesn't have nIn value
    }

    @Override
    public InputPreProcessor getPreProcessorForInputType(InputType inputType) {
        if (inputType == null) {
            throw new IllegalStateException("Invalid input for Subsampling layer (layer name=\"" + getLayerName()
                            + "\"): input is null");
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
    public boolean isPretrainParam(String paramName) {
        throw new UnsupportedOperationException("SubsamplingLayer does not contain parameters");
    }

    @Override
    public LayerMemoryReport getMemoryReport(InputType inputType) {
        InputType.InputTypeConvolutional c = (InputType.InputTypeConvolutional) inputType;
        InputType.InputTypeConvolutional outputType = (InputType.InputTypeConvolutional) getOutputType(-1, inputType);
        int actElementsPerEx = outputType.arrayElementsPerExample();

        //TODO Subsampling helper memory use... (CuDNN etc)

        //During forward pass: im2col array + reduce. Reduce is counted as activations, so only im2col is working mem
        int im2colSizePerEx =
                        c.getChannels() * outputType.getHeight() * outputType.getWidth() * kernelSize[0] * kernelSize[1];

        //Current implementation does NOT cache im2col etc... which means: it's recalculated on each backward pass
        int trainingWorkingSizePerEx = im2colSizePerEx;
        if (getIDropout() != null) {
            //Dup on the input before dropout, but only for training
            trainingWorkingSizePerEx += inputType.arrayElementsPerExample();
        }

        return new LayerMemoryReport.Builder(layerName, SubsamplingLayer.class, inputType, outputType)
                        .standardMemory(0, 0) //No params
                        .workingMemory(0, im2colSizePerEx, 0, trainingWorkingSizePerEx)
                        .cacheMemory(MemoryReport.CACHE_MODE_ALL_ZEROS, MemoryReport.CACHE_MODE_ALL_ZEROS) //No caching
                        .build();
    }

    public int getPnorm() {
        return pnorm;
    }

    public double getEps() {
        return eps;
    }

    @NoArgsConstructor
    public static class Builder extends BaseSubsamplingBuilder<Builder> {

        private int[] dilation = new int[]{1,1};

        public Builder(PoolingType poolingType, int[] kernelSize, int[] stride) {
            super(poolingType, kernelSize, stride);
        }

        public Builder(PoolingType poolingType, int[] kernelSize) {
            super(poolingType, kernelSize);
        }

        public Builder(PoolingType poolingType, int[] kernelSize, int[] stride, int[] padding) {
            super(poolingType, kernelSize, stride, padding);
        }

        public Builder(org.deeplearning4j.nn.conf.layers.PoolingType poolingType, int[] kernelSize) {
            super(poolingType, kernelSize);
        }

        public Builder(org.deeplearning4j.nn.conf.layers.PoolingType poolingType, int[] kernelSize, int[] stride,
                        int[] padding) {
            super(poolingType, kernelSize, stride, padding);
        }

        public Builder(int[] kernelSize, int[] stride, int[] padding) {
            super(kernelSize, stride, padding);
        }

        public Builder(int[] kernelSize, int[] stride) {
            super(kernelSize, stride);
        }

        public Builder(int... kernelSize) {
            super(kernelSize);
        }

        public Builder(PoolingType poolingType) {
            super(poolingType);
        }

        public Builder(org.deeplearning4j.nn.conf.layers.PoolingType poolingType) {
            super(poolingType);
        }

        /**
         * Kernel size
         *
         * @param kernelSize    kernel size in height and width dimensions
         */
        public Builder kernelSize(int... kernelSize) {
            if (kernelSize.length != 2)
                throw new IllegalArgumentException("Invalid input: must be length 2");
            this.kernelSize = kernelSize;
            return this;
        }

        /**
         * Stride
         *
         * @param stride    stride in height and width dimensions
         */
        public Builder stride(int... stride) {
            if (stride.length != 2)
                throw new IllegalArgumentException("Invalid input: must be length 2");
            this.stride = stride;
            return this;
        }

        /**
         * Padding
         *
         * @param padding    padding in the height and width dimensions
         */
        public Builder padding(int... padding) {
            if (padding.length != 2)
                throw new IllegalArgumentException("Invalid input: must be length 2");
            this.padding = padding;
            return this;
        }

        /**
         * Kernel dilation. Default: {1, 1}, which is standard convolutions. Used for implementing dilated convolutions,
         * which are also known as atrous convolutions.<br>
         * NOTE: Kernel dilation is less common in practice for subsampling layers, compared to convolutional layers.
         *
         * For more details, see:
         * <a href="https://arxiv.org/abs/1511.07122">Yu and Koltun (2014)</a> and
         * <a href="https://arxiv.org/abs/1412.7062">Chen et al. (2014)</a>, as well as
         * <a href="http://deeplearning.net/software/theano/tutorial/conv_arithmetic.html#dilated-convolutions">
         *     http://deeplearning.net/software/theano/tutorial/conv_arithmetic.html#dilated-convolutions</a><br>
         *
         * @param dilation Dilation for kernel
         */
        public Builder dilation(int... dilation){
            this.dilation = dilation;
            return this;
        }


        @Override
        @SuppressWarnings("unchecked")
        public SubsamplingLayer build() {
            if (poolingType == org.deeplearning4j.nn.conf.layers.PoolingType.PNORM && pnorm <= 0)
                throw new IllegalStateException(
                                "Incorrect Subsampling config: p-norm must be set when using PoolingType.PNORM");
            ConvolutionUtils.validateConvolutionModePadding(convolutionMode, padding);
            ConvolutionUtils.validateCnnKernelStridePadding(kernelSize, stride, padding);

            return new SubsamplingLayer(this);
        }
    }

    @NoArgsConstructor
    protected static abstract class BaseSubsamplingBuilder<T extends BaseSubsamplingBuilder<T>>
                    extends Layer.Builder<T> {
        protected org.deeplearning4j.nn.conf.layers.PoolingType poolingType =
                        org.deeplearning4j.nn.conf.layers.PoolingType.MAX;
        protected int[] kernelSize = new int[] {1, 1}; // Same as filter size from the last conv layer
        protected int[] stride = new int[] {2, 2}; // Default is 2. Down-sample by a factor of 2
        protected int[] padding = new int[] {0, 0};
        protected ConvolutionMode convolutionMode = null;
        protected int pnorm;
        protected double eps = 1e-8;
        protected boolean cudnnAllowFallback = true;

        protected BaseSubsamplingBuilder(PoolingType poolingType, int[] kernelSize, int[] stride) {
            this.poolingType = poolingType.toPoolingType();
            this.kernelSize = kernelSize;
            this.stride = stride;
        }

        protected BaseSubsamplingBuilder(PoolingType poolingType, int[] kernelSize) {
            this.poolingType = poolingType.toPoolingType();
            this.kernelSize = kernelSize;
        }

        protected BaseSubsamplingBuilder(PoolingType poolingType, int[] kernelSize, int[] stride, int[] padding) {
            this.poolingType = poolingType.toPoolingType();
            this.kernelSize = kernelSize;
            this.stride = stride;
            this.padding = padding;
        }

        protected BaseSubsamplingBuilder(org.deeplearning4j.nn.conf.layers.PoolingType poolingType, int[] kernelSize) {
            this.poolingType = poolingType;
            this.kernelSize = kernelSize;
        }

        protected BaseSubsamplingBuilder(org.deeplearning4j.nn.conf.layers.PoolingType poolingType, int[] kernelSize,
                        int[] stride, int[] padding) {
            this.poolingType = poolingType;
            this.kernelSize = kernelSize;
            this.stride = stride;
            this.padding = padding;
        }

        protected BaseSubsamplingBuilder(int[] kernelSize, int[] stride, int[] padding) {
            this.kernelSize = kernelSize;
            this.stride = stride;
            this.padding = padding;
        }

        protected BaseSubsamplingBuilder(int[] kernelSize, int[] stride) {
            this.kernelSize = kernelSize;
            this.stride = stride;
        }

        protected BaseSubsamplingBuilder(int... kernelSize) {
            this.kernelSize = kernelSize;
        }

        protected BaseSubsamplingBuilder(PoolingType poolingType) {
            this.poolingType = poolingType.toPoolingType();
        }

        protected BaseSubsamplingBuilder(org.deeplearning4j.nn.conf.layers.PoolingType poolingType) {
            this.poolingType = poolingType;
        }

        /**
         * Set the convolution mode for the Convolution layer.
         * See {@link ConvolutionMode} for more details
         *
         * @param convolutionMode    Convolution mode for layer
         */
        public T convolutionMode(ConvolutionMode convolutionMode) {
            this.convolutionMode = convolutionMode;
            return (T) this;
        }

        public T poolingType(PoolingType poolingType) {
            this.poolingType = poolingType.toPoolingType();
            return (T) this;
        }

        public T pnorm(int pnorm) {
            if (pnorm <= 0)
                throw new IllegalArgumentException("Invalid input: p-norm value must be greater than 0");
            this.pnorm = pnorm;
            return (T) this;
        }

        public T eps(double eps) {
            if (eps <= 0)
                throw new IllegalArgumentException("Invalid input: epsilon for p-norm must be greater than 0");
            this.eps = eps;
            return (T) this;
        }

        /**
         * When using CuDNN and an error is encountered, should fallback to the non-CuDNN implementatation be allowed?
         * If set to false, an exception in CuDNN will be propagated back to the user. If false, the built-in (non-CuDNN)
         * implementation for ConvolutionLayer will be used
         *
         * @param allowFallback Whether fallback to non-CuDNN implementation should be used
         */
        public T cudnnAllowFallback(boolean allowFallback){
            this.cudnnAllowFallback = allowFallback;
            return (T) this;
        }
    }

}

/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.nn.conf.layers;

import lombok.*;
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
import org.deeplearning4j.util.ValidationUtils;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Collection;
import java.util.Map;

/**
 * Subsampling layer also referred to as pooling in convolution neural nets
 *
 * Supports the following pooling types: MAX, AVG, SUM, PNORM
 *
 * @author Adam Gibson
 */

@Data
@NoArgsConstructor
@ToString(callSuper = true)
@EqualsAndHashCode(callSuper = true)
public class SubsamplingLayer extends NoParamLayer {

    protected ConvolutionMode convolutionMode = ConvolutionMode.Truncate; //Default to truncate here - default for 0.6.0 and earlier networks on JSON deserialization
    protected org.deeplearning4j.nn.conf.layers.PoolingType poolingType;
    protected int[] kernelSize; // Same as filter size from the last conv layer
    protected int[] stride; // Default is 2. Down-sample by a factor of 2
    protected int[] padding;
    protected int[] dilation = new int[] {1, 1};
    protected int pnorm;
    protected double eps;
    protected boolean cudnnAllowFallback = true;
    /*
    Default here for JSON deserialization of 1.0.0-beta4 and earlier models. New models default to false via builder.
    This impacts average pooling only - whether the divisor should include or exclude padding along image edges.
    DL4J originally included padding in the count, versions after 1.0.0-beta4 will exclude it by default.
     */
    protected boolean avgPoolIncludePadInDivisor = true;

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
        if (builder.kernelSize.length != 2) {
            throw new IllegalArgumentException("Kernel size of should be rows x columns (a 2d array)");
        }
        this.kernelSize = builder.kernelSize;
        if (builder.stride.length != 2) {
            throw new IllegalArgumentException("Invalid stride, must be length 2");
        }
        this.stride = builder.stride;
        this.padding = builder.padding;
        this.convolutionMode = builder.convolutionMode;
        if (builder instanceof Builder) {
            this.dilation = ((Builder) builder).dilation;
        }
        this.pnorm = builder.pnorm;
        this.eps = builder.eps;
        this.cudnnAllowFallback = builder.cudnnAllowFallback;
        this.avgPoolIncludePadInDivisor = builder.avgPoolIncludePadInDivisor;
    }

    @Override
    public SubsamplingLayer clone() {
        SubsamplingLayer clone = (SubsamplingLayer) super.clone();

        if (clone.kernelSize != null) {
            clone.kernelSize = clone.kernelSize.clone();
        }
        if (clone.stride != null) {
            clone.stride = clone.stride.clone();
        }
        if (clone.padding != null) {
            clone.padding = clone.padding.clone();
        }
        if (clone.dilation != null) {
            clone.dilation = clone.dilation.clone();
        }
        return clone;
    }

    @Override
    public org.deeplearning4j.nn.api.Layer instantiate(NeuralNetConfiguration conf,
                                                       Collection<TrainingListener> trainingListeners, int layerIndex, INDArray layerParamsView,
                                                       boolean initializeParams, DataType networkDataType) {
        org.deeplearning4j.nn.layers.convolution.subsampling.SubsamplingLayer ret =
                        new org.deeplearning4j.nn.layers.convolution.subsampling.SubsamplingLayer(conf, networkDataType);
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
    public boolean isPretrainParam(String paramName) {
        throw new UnsupportedOperationException("SubsamplingLayer does not contain parameters");
    }

    @Override
    public LayerMemoryReport getMemoryReport(InputType inputType) {
        InputType.InputTypeConvolutional c = (InputType.InputTypeConvolutional) inputType;
        InputType.InputTypeConvolutional outputType = (InputType.InputTypeConvolutional) getOutputType(-1, inputType);
        val actElementsPerEx = outputType.arrayElementsPerExample();

        //TODO Subsampling helper memory use... (CuDNN etc)

        //During forward pass: im2col array + reduce. Reduce is counted as activations, so only im2col is working mem
        val im2colSizePerEx = c.getChannels() * outputType.getHeight() * outputType.getWidth() * kernelSize[0]
                        * kernelSize[1];

        //Current implementation does NOT cache im2col etc... which means: it's recalculated on each backward pass
        long trainingWorkingSizePerEx = im2colSizePerEx;
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
    @Getter
    @Setter
    public static class Builder extends BaseSubsamplingBuilder<Builder> {

        /**
         * Kernel dilation. Default: {1, 1}, which is standard convolutions. Used for implementing dilated convolutions,
         * which are also known as atrous convolutions.<br> NOTE: Kernel dilation is less common in practice for
         * subsampling layers, compared to convolutional layers.
         *
         * For more details, see:
         * <a href="https://arxiv.org/abs/1511.07122">Yu and Koltun (2014)</a> and
         * <a href="https://arxiv.org/abs/1412.7062">Chen et al. (2014)</a>, as well as
         * <a href="http://deeplearning.net/software/theano/tutorial/conv_arithmetic.html#dilated-convolutions">
         * http://deeplearning.net/software/theano/tutorial/conv_arithmetic.html#dilated-convolutions</a><br>
         *
         * Dilation for kernel
         */
        private int[] dilation = new int[] {1, 1};

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
         * @param kernelSize kernel size in height and width dimensions
         */
        public Builder kernelSize(int... kernelSize) {
            this.setKernelSize(kernelSize);
            return this;
        }

        /**
         * Stride
         *
         * @param stride stride in height and width dimensions
         */
        public Builder stride(int... stride) {
            this.setStride(stride);
            return this;
        }

        /**
         * Padding
         *
         * @param padding padding in the height and width dimensions
         */
        public Builder padding(int... padding) {
            this.setPadding(padding);
            return this;
        }

        /**
         * Kernel dilation. Default: {1, 1}, which is standard convolutions. Used for implementing dilated convolutions,
         * which are also known as atrous convolutions.<br> NOTE: Kernel dilation is less common in practice for
         * subsampling layers, compared to convolutional layers.
         *
         * For more details, see:
         * <a href="https://arxiv.org/abs/1511.07122">Yu and Koltun (2014)</a> and
         * <a href="https://arxiv.org/abs/1412.7062">Chen et al. (2014)</a>, as well as
         * <a href="http://deeplearning.net/software/theano/tutorial/conv_arithmetic.html#dilated-convolutions">
         * http://deeplearning.net/software/theano/tutorial/conv_arithmetic.html#dilated-convolutions</a><br>
         *
         * @param dilation Dilation for kernel
         */
        public Builder dilation(int... dilation) {
            this.setDilation(dilation);
            return this;
        }


        @Override
        @SuppressWarnings("unchecked")
        public SubsamplingLayer build() {
            if (poolingType == org.deeplearning4j.nn.conf.layers.PoolingType.PNORM && pnorm <= 0) {
                throw new IllegalStateException(
                                "Incorrect Subsampling config: p-norm must be set when using PoolingType.PNORM");
            }
            ConvolutionUtils.validateConvolutionModePadding(convolutionMode, padding);
            ConvolutionUtils.validateCnnKernelStridePadding(kernelSize, stride, padding);

            return new SubsamplingLayer(this);
        }

        @Override
        public void setKernelSize(int... kernelSize) {
            this.kernelSize = ValidationUtils.validate2NonNegative(kernelSize,false, "kernelSize");
        }

        @Override
        public void setStride(int... stride) {
            this.stride = ValidationUtils.validate2NonNegative(stride, false, "stride");
        }

        @Override
        public void setPadding(int... padding) {
            this.padding = ValidationUtils.validate2NonNegative(padding,false, "padding");
        }


        public void setDilation(int[] dilation) {
            this.dilation = ValidationUtils.validate2NonNegative(dilation, false, "dilation");
        }
    }

    @NoArgsConstructor
    @Getter
    @Setter
    protected static abstract class BaseSubsamplingBuilder<T extends BaseSubsamplingBuilder<T>>
                    extends Layer.Builder<T> {

        protected org.deeplearning4j.nn.conf.layers.PoolingType poolingType =
                        org.deeplearning4j.nn.conf.layers.PoolingType.MAX;

        protected int[] kernelSize = new int[] {1, 1}; // Same as filter size from the last conv layer
        protected int[] stride = new int[] {2, 2}; // Default is 2. Down-sample by a factor of 2
        protected int[] padding = new int[] {0, 0};

        /**
         * Set the convolution mode for the Convolution layer. See {@link ConvolutionMode} for more details
         *
         * Convolution mode for layer
         */
        protected ConvolutionMode convolutionMode = null;
        protected int pnorm;
        protected double eps = 1e-8;

        /**
         * When using CuDNN and an error is encountered, should fallback to the non-CuDNN implementatation be allowed?
         * If set to false, an exception in CuDNN will be propagated back to the user. If false, the built-in
         * (non-CuDNN) implementation for ConvolutionLayer will be used
         *
         * Whether fallback to non-CuDNN implementation should be used
         */
        protected boolean cudnnAllowFallback = true;
        protected boolean avgPoolIncludePadInDivisor = false;

        protected BaseSubsamplingBuilder(PoolingType poolingType, int[] kernelSize, int[] stride) {
            this.setPoolingType(poolingType.toPoolingType());
            this.setKernelSize(kernelSize);
            this.setStride(stride);
        }

        protected BaseSubsamplingBuilder(PoolingType poolingType, int[] kernelSize) {
            this.setPoolingType(poolingType.toPoolingType());
            this.setKernelSize(kernelSize);
        }

        protected BaseSubsamplingBuilder(PoolingType poolingType, int[] kernelSize, int[] stride, int[] padding) {
            this.setPoolingType(poolingType.toPoolingType());
            this.setKernelSize(kernelSize);
            this.setStride(stride);
            this.setPadding(padding);
        }

        protected BaseSubsamplingBuilder(org.deeplearning4j.nn.conf.layers.PoolingType poolingType, int[] kernelSize) {
            this.setPoolingType(poolingType);
            this.setKernelSize(kernelSize);
        }

        protected BaseSubsamplingBuilder(org.deeplearning4j.nn.conf.layers.PoolingType poolingType, int[] kernelSize,
                        int[] stride, int[] padding) {
            this.setPoolingType(poolingType);
            this.setKernelSize(kernelSize);
            this.setStride(stride);
            this.setPadding(padding);
        }

        protected BaseSubsamplingBuilder(int[] kernelSize, int[] stride, int[] padding) {
            this.setKernelSize(kernelSize);
            this.setStride(stride);
            this.setPadding(padding);
        }

        protected BaseSubsamplingBuilder(int[] kernelSize, int[] stride) {
            this.setKernelSize(kernelSize);
            this.setStride(stride);
        }

        protected BaseSubsamplingBuilder(int... kernelSize) {
            this.setKernelSize(kernelSize);
        }

        protected BaseSubsamplingBuilder(PoolingType poolingType) {
            this.setPoolingType(poolingType.toPoolingType());
        }

        protected BaseSubsamplingBuilder(org.deeplearning4j.nn.conf.layers.PoolingType poolingType) {
            this.setPoolingType(poolingType);
        }

        public void setPnorm(int pnorm){
            ValidationUtils.validateNonNegative(pnorm, "pnorm");
            this.pnorm = pnorm;
        }

        public void setEps(double eps){
            ValidationUtils.validateNonNegative(eps, "eps");
            this.eps = eps;
        }

        /**
         * Set the convolution mode for the Convolution layer. See {@link ConvolutionMode} for more details
         *
         * @param convolutionMode Convolution mode for layer
         */
        public T convolutionMode(ConvolutionMode convolutionMode) {
            this.setConvolutionMode(convolutionMode);
            return (T) this;
        }

        public T poolingType(PoolingType poolingType) {
            this.setPoolingType(poolingType.toPoolingType());
            return (T) this;
        }

        public T poolingType(org.deeplearning4j.nn.conf.layers.PoolingType poolingType){
            this.setPoolingType(poolingType);
            return (T) this;
        }

        public T pnorm(int pnorm) {
            this.setPnorm(pnorm);
            return (T) this;
        }

        public T eps(double eps) {
            this.setEps(eps);
            return (T) this;
        }

        /**
         * When using CuDNN or MKLDNN and an error is encountered, should fallback to the non-helper implementation be allowed?
         * If set to false, an exception in the helper will be propagated back to the user. If true, the built-in
         * (non-MKL/CuDNN) implementation for ConvolutionLayer will be used
         *
         * @deprecated Use {@link #helperAllowFallback(boolean)}
         *
         * @param allowFallback Whether fallback to non-CuDNN implementation should be used
         */
        @Deprecated
        public T cudnnAllowFallback(boolean allowFallback) {
            this.cudnnAllowFallback = allowFallback;
            return (T) this;
        }

        /**
         * When using CuDNN or MKLDNN and an error is encountered, should fallback to the non-helper implementation be allowed?
         * If set to false, an exception in the helper will be propagated back to the user. If true, the built-in
         * (non-MKL/CuDNN) implementation for SubsamplingLayer will be used
         *
         * @param allowFallback Whether fallback to non-CuDNN implementation should be used
         */
        public T helperAllowFallback(boolean allowFallback) {
            this.cudnnAllowFallback = allowFallback;
            return (T) this;
        }

        /**
         * When doing average pooling, should the padding values be included in the divisor or not?<br>
         * Not applicable for max and p-norm pooling.<br>
         * Users should not usually set this - instead, leave it as the default (false). It is included mainly for backward
         * compatibility of older models<br>
         * Consider the following 2x2 segment along the right side of the image:<br>
         * <pre>
         * [A, P]
         * [B, P]
         * </pre>
         * Where A and B are actual values, and P is padding (0).<br>
         * With avgPoolIncludePadInDivisor = true, we have: out = (A+B+0+0)/4<br>
         * With avgPoolIncludePadInDivisor = false, we have: out = (A+B+0+0)/2<br>
         * <br>
         * Earlier versions of DL4J originally included padding in the count, newer versions exclude it.<br>
         *
         * @param avgPoolIncludePadInDivisor Whether the divisor should include or exclude padding for average pooling
         */
        public T avgPoolIncludePadInDivisor(boolean avgPoolIncludePadInDivisor){
            this.avgPoolIncludePadInDivisor = avgPoolIncludePadInDivisor;
            return (T) this;
        }
    }

}

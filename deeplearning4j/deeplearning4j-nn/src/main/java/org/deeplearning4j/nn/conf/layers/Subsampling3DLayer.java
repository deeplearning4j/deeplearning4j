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
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.util.Convolution3DUtils;
import org.deeplearning4j.util.ConvolutionUtils;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Collection;
import java.util.Map;

/**
 * 3D subsampling / pooling layer for convolutional neural networks
 * <p>
 * Supports max and average pooling modes
 *
 * @author Max Pumperla
 */
@Data
@NoArgsConstructor
@ToString(callSuper = true)
@EqualsAndHashCode(callSuper = true)
public class Subsampling3DLayer extends NoParamLayer {

    protected ConvolutionMode convolutionMode = ConvolutionMode.Truncate;
    protected org.deeplearning4j.nn.conf.layers.PoolingType poolingType;
    protected int[] kernelSize;
    protected int[] stride;
    protected int[] padding;
    protected int[] dilation;
    protected boolean cudnnAllowFallback = true;

    public enum PoolingType {
        MAX, AVG;

        public org.deeplearning4j.nn.conf.layers.PoolingType toPoolingType() {
            switch (this) {
                case MAX:
                    return org.deeplearning4j.nn.conf.layers.PoolingType.MAX;
                case AVG:
                    return org.deeplearning4j.nn.conf.layers.PoolingType.AVG;
            }
            throw new UnsupportedOperationException("Unknown/not supported pooling type: " + this);
        }
    }

    protected Subsampling3DLayer(BaseSubsamplingBuilder builder) {
        super(builder);
        this.poolingType = builder.poolingType;
        if (builder.kernelSize.length != 3)
            throw new IllegalArgumentException("Kernel size must be length 3");
        this.kernelSize = builder.kernelSize;
        if (builder.stride.length != 3)
            throw new IllegalArgumentException("Invalid stride, must be length 3");
        this.stride = builder.stride;
        this.padding = builder.padding;
        this.dilation = builder.dilation;
        this.convolutionMode = builder.convolutionMode;
        this.cudnnAllowFallback = builder.cudnnAllowFallback;
    }

    @Override
    public Subsampling3DLayer clone() {
        Subsampling3DLayer clone = (Subsampling3DLayer) super.clone();

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
    public org.deeplearning4j.nn.api.Layer instantiate(NeuralNetConfiguration conf,
                                                       Collection<TrainingListener> iterationListeners,
                                                       int layerIndex, INDArray layerParamsView,
                                                       boolean initializeParams) {
        org.deeplearning4j.nn.layers.convolution.subsampling.Subsampling3DLayer ret =
                new org.deeplearning4j.nn.layers.convolution.subsampling.Subsampling3DLayer(conf);
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
        if (inputType == null || inputType.getType() != InputType.Type.CNN3D) {
            throw new IllegalStateException("Invalid input for Subsampling 3D layer (layer name=\"" + getLayerName()
                    + "\"): Expected CNN input, got " + inputType);
        }

        // FIXME: int cast
        return InputTypeUtil.getOutputTypeCnn3DLayers(inputType, kernelSize, stride, padding,
                new int[]{1, 1, 1}, // no dilation
                convolutionMode,
                (int) ((InputType.InputTypeConvolutional3D) inputType).getChannels(), layerIndex, getLayerName(),
                Subsampling3DLayer.class);
    }

    @Override
    public void setNIn(InputType inputType, boolean override) {
        //No op: subsampling layer doesn't have nIn value
    }

    @Override
    public InputPreProcessor getPreProcessorForInputType(InputType inputType) {
        if (inputType == null) {
            throw new IllegalStateException("Invalid input for Subsampling 3D layer (layer name=\"" + getLayerName()
                    + "\"): input is null");
        }

        return InputTypeUtil.getPreProcessorForInputTypeCnn3DLayers(inputType, getLayerName());
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
        InputType.InputTypeConvolutional3D c = (InputType.InputTypeConvolutional3D) inputType;
        InputType.InputTypeConvolutional3D outputType = (InputType.InputTypeConvolutional3D) getOutputType(-1, inputType);
        val actElementsPerEx = outputType.arrayElementsPerExample();


        //During forward pass: im2col array + reduce. Reduce is counted as activations, so only im2col is working mem
        val im2colSizePerEx =
                c.getChannels() * outputType.getHeight() * outputType.getWidth() * outputType.getDepth()
                        * kernelSize[0] * kernelSize[1];

        //Current implementation does NOT cache im2col etc... which means: it's recalculated on each backward pass
        long trainingWorkingSizePerEx = im2colSizePerEx;
        if (getIDropout() != null) {
            //Dup on the input before dropout, but only for training
            trainingWorkingSizePerEx += inputType.arrayElementsPerExample();
        }

        return new LayerMemoryReport.Builder(layerName, Subsampling3DLayer.class, inputType, outputType)
                .standardMemory(0, 0) //No params
                .workingMemory(0, im2colSizePerEx, 0, trainingWorkingSizePerEx)
                .cacheMemory(MemoryReport.CACHE_MODE_ALL_ZEROS, MemoryReport.CACHE_MODE_ALL_ZEROS) //No caching
                .build();
    }

    @NoArgsConstructor
    public static class Builder extends BaseSubsamplingBuilder<Builder> {


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
            if (kernelSize.length != 3)
                throw new IllegalArgumentException("Invalid input: must be length 3");
            this.kernelSize = kernelSize;
            return this;
        }

        /**
         * Stride
         *
         * @param stride stride in height and width dimensions
         */
        public Builder stride(int... stride) {
            if (stride.length != 3)
                throw new IllegalArgumentException("Invalid input: must be length 3");
            this.stride = stride;
            return this;
        }

        /**
         * Padding
         *
         * @param padding padding in the height and width dimensions
         */
        public Builder padding(int... padding) {
            if (padding.length != 3)
                throw new IllegalArgumentException("Invalid input: must be length 3");
            this.padding = padding;
            return this;
        }


        @Override
        @SuppressWarnings("unchecked")
        public Subsampling3DLayer build() {
            ConvolutionUtils.validateConvolutionModePadding(convolutionMode, padding);
            Convolution3DUtils.validateCnn3DKernelStridePadding(kernelSize, stride, padding);
            return new Subsampling3DLayer(this);
        }
    }

    @NoArgsConstructor
    protected static abstract class BaseSubsamplingBuilder<T extends BaseSubsamplingBuilder<T>>
            extends Layer.Builder<T> {
        protected org.deeplearning4j.nn.conf.layers.PoolingType poolingType =
                org.deeplearning4j.nn.conf.layers.PoolingType.MAX;
        protected int[] kernelSize = new int[]{1, 1, 1};
        protected int[] stride = new int[]{2, 2, 2};
        protected int[] padding = new int[]{0, 0, 0};
        protected int[] dilation = new int[]{1, 1, 1};
        protected ConvolutionMode convolutionMode = ConvolutionMode.Same;
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
         * @param convolutionMode Convolution mode for layer
         */
        public T convolutionMode(ConvolutionMode convolutionMode) {
            this.convolutionMode = convolutionMode;
            return (T) this;
        }

        public T poolingType(PoolingType poolingType) {
            this.poolingType = poolingType.toPoolingType();
            return (T) this;
        }

        public T dilation(int dDepth, int dHeight, int dWidth){
            this.dilation = new int[]{dDepth, dHeight, dWidth};
            return (T) this;
        }

        /**
         * When using CuDNN and an error is encountered, should fallback to the non-CuDNN implementatation be allowed?
         * If set to false, an exception in CuDNN will be propagated back to the user. If false, the built-in (non-CuDNN)
         * implementation for ConvolutionLayer will be used
         *
         * @param allowFallback Whether fallback to non-CuDNN implementation should be used
         */
        public T cudnnAllowFallback(boolean allowFallback) {
            this.cudnnAllowFallback = allowFallback;
            return (T) this;
        }
    }

}

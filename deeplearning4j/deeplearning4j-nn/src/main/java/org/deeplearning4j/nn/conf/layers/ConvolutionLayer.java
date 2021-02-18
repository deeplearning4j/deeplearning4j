/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.nn.conf.layers;

import lombok.*;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.nn.conf.memory.MemoryReport;
import org.deeplearning4j.nn.params.ConvolutionParamInitializer;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.util.ConvolutionUtils;
import org.deeplearning4j.util.ValidationUtils;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.shade.jackson.annotation.JsonIgnore;

import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;

@Data
@NoArgsConstructor
@ToString(callSuper = true)
@EqualsAndHashCode(callSuper = true)
public class ConvolutionLayer extends FeedForwardLayer {

    protected boolean hasBias = true;
    protected ConvolutionMode convolutionMode = ConvolutionMode.Truncate; //Default to truncate here - default for 0.6.0 and earlier networks on JSON deserialization
    protected int dilation[] = new int[] {1, 1};
    protected int[] kernelSize; // Square filter
    protected int[] stride; // Default is 2. Down-sample by a factor of 2
    protected int[] padding;
    protected boolean cudnnAllowFallback = true;
    protected CNN2DFormat cnn2dDataFormat = CNN2DFormat.NCHW; //default value for legacy serialization reasons
    @JsonIgnore
    @EqualsAndHashCode.Exclude
    private boolean defaultValueOverriden = false;

    /**
     * The "PREFER_FASTEST" mode will pick the fastest algorithm for the specified parameters from the {@link FwdAlgo},
     * {@link BwdFilterAlgo}, and {@link BwdDataAlgo} lists, but they may be very memory intensive, so if weird errors
     * occur when using cuDNN, please try the "NO_WORKSPACE" mode. Alternatively, it is possible to specify the
     * algorithm manually by setting the "USER_SPECIFIED" mode, but this is not recommended.
     * <p>
     * Note: Currently only supported with cuDNN.
     */
    public enum AlgoMode {
        NO_WORKSPACE, PREFER_FASTEST, USER_SPECIFIED
    }

    /**
     * The forward algorithm to use when {@link AlgoMode} is set to "USER_SPECIFIED".
     * <p>
     * Note: Currently only supported with cuDNN.
     */
    public enum FwdAlgo {
        IMPLICIT_GEMM, IMPLICIT_PRECOMP_GEMM, GEMM, DIRECT, FFT, FFT_TILING, WINOGRAD, WINOGRAD_NONFUSED, COUNT
    }

    /**
     * The backward filter algorithm to use when {@link AlgoMode} is set to "USER_SPECIFIED".
     * <p>
     * Note: Currently only supported with cuDNN.
     */
    public enum BwdFilterAlgo {
        ALGO_0, ALGO_1, FFT, ALGO_3, WINOGRAD, WINOGRAD_NONFUSED, FFT_TILING, COUNT
    }

    /**
     * The backward data algorithm to use when {@link AlgoMode} is set to "USER_SPECIFIED".
     * <p>
     * Note: Currently only supported with cuDNN.
     */
    public enum BwdDataAlgo {
        ALGO_0, ALGO_1, FFT, FFT_TILING, WINOGRAD, WINOGRAD_NONFUSED, COUNT
    }

    /**
     * Defaults to "PREFER_FASTEST", but "NO_WORKSPACE" uses less memory.
     */
    protected AlgoMode cudnnAlgoMode = AlgoMode.PREFER_FASTEST;
    protected FwdAlgo cudnnFwdAlgo;
    protected BwdFilterAlgo cudnnBwdFilterAlgo;
    protected BwdDataAlgo cudnnBwdDataAlgo;

    /**
     * ConvolutionLayer nIn in the input layer is the number of channels nOut is the number of filters to be used in the
     * net or in other words the channels The builder specifies the filter/kernel size, the stride and padding The
     * pooling layer takes the kernel size
     */
    protected ConvolutionLayer(BaseConvBuilder<?> builder) {
        super(builder);
        int dim = builder.convolutionDim;

        this.hasBias = builder.hasBias;
        this.convolutionMode = builder.convolutionMode;
        this.dilation = builder.dilation;
        if (builder.kernelSize.length != dim) {
            throw new IllegalArgumentException("Kernel argument should be a " + dim + "d array, got " + Arrays.toString(builder.kernelSize));
        }
        this.kernelSize = builder.kernelSize;
        if (builder.stride.length != dim) {
            throw new IllegalArgumentException("Strides argument should be a " + dim + "d array, got " + Arrays.toString(builder.stride));
        }
        this.stride = builder.stride;
        if (builder.padding.length != dim) {
            throw new IllegalArgumentException("Padding argument should be a " + dim + "d array, got " + Arrays.toString(builder.padding));
        }
        this.padding = builder.padding;
        if (builder.dilation.length != dim) {
            throw new IllegalArgumentException("Dilation argument should be a " + dim + "d array, got " + Arrays.toString(builder.dilation));
        }
        this.dilation = builder.dilation;
        this.cudnnAlgoMode = builder.cudnnAlgoMode;
        this.cudnnFwdAlgo = builder.cudnnFwdAlgo;
        this.cudnnBwdFilterAlgo = builder.cudnnBwdFilterAlgo;
        this.cudnnBwdDataAlgo = builder.cudnnBwdDataAlgo;
        this.cudnnAllowFallback = builder.cudnnAllowFallback;
        if(builder instanceof Builder) {
            this.cnn2dDataFormat = ((Builder)builder).dataFormat;
        }

        initializeConstraints(builder);
    }

    public boolean hasBias() {
        return hasBias;
    }

    @Override
    public ConvolutionLayer clone() {
        ConvolutionLayer clone = (ConvolutionLayer) super.clone();
        if (clone.kernelSize != null) {
            clone.kernelSize = clone.kernelSize.clone();
        }
        if (clone.stride != null) {
            clone.stride = clone.stride.clone();
        }
        if (clone.padding != null) {
            clone.padding = clone.padding.clone();
        }
        return clone;
    }

    @Override
    public Layer instantiate(NeuralNetConfiguration conf, Collection<TrainingListener> trainingListeners,
                             int layerIndex, INDArray layerParamsView, boolean initializeParams, DataType networkDataType) {
        LayerValidation.assertNInNOutSet("ConvolutionLayer", getLayerName(), layerIndex, getNIn(), getNOut());

        org.deeplearning4j.nn.layers.convolution.ConvolutionLayer ret =
                new org.deeplearning4j.nn.layers.convolution.ConvolutionLayer(conf, networkDataType);
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
        return ConvolutionParamInitializer.getInstance();
    }

    @Override
    public InputType getOutputType(int layerIndex, InputType inputType) {
        if (inputType == null || inputType.getType() != InputType.Type.CNN) {
            throw new IllegalStateException("Invalid input for Convolution layer (layer name=\"" + getLayerName()
                    + "\"): Expected CNN input, got " + inputType);
        }

        return InputTypeUtil.getOutputTypeCnnLayers(inputType, kernelSize, stride, padding, dilation, convolutionMode,
                nOut, layerIndex, getLayerName(), cnn2dDataFormat, ConvolutionLayer.class);
    }

    @Override
    public void setNIn(InputType inputType, boolean override) {
        if (inputType == null || inputType.getType() != InputType.Type.CNN) {
            throw new IllegalStateException("Invalid input for Convolution layer (layer name=\"" + getLayerName()
                    + "\"): Expected CNN input, got " + inputType);
        }

        if (!defaultValueOverriden || nIn <= 0 || override) {
            InputType.InputTypeConvolutional c = (InputType.InputTypeConvolutional) inputType;
            this.nIn = c.getChannels();
            this.cnn2dDataFormat = ((InputType.InputTypeConvolutional) inputType).getFormat();
        }

        if(cnn2dDataFormat == null || override)
            this.cnn2dDataFormat = ((InputType.InputTypeConvolutional) inputType).getFormat();
    }

    @Override
    public InputPreProcessor getPreProcessorForInputType(InputType inputType) {
        if (inputType == null) {
            throw new IllegalStateException("Invalid input for Convolution layer (layer name=\"" + getLayerName()
                    + "\"): input is null");
        }

        return InputTypeUtil.getPreProcessorForInputTypeCnnLayers(inputType, getLayerName());
    }

    @Override
    public LayerMemoryReport getMemoryReport(InputType inputType) {
        val paramSize = initializer().numParams(this);
        val updaterStateSize = (int) getIUpdater().stateSize(paramSize);

        InputType.InputTypeConvolutional c = (InputType.InputTypeConvolutional) inputType;
        InputType.InputTypeConvolutional outputType = (InputType.InputTypeConvolutional) getOutputType(-1, inputType);

        //TODO convolution helper memory use... (CuDNN etc)

        //During forward pass: im2col array, mmul (result activations), in-place broadcast add
        val im2colSizePerEx = c.getChannels() * outputType.getHeight() * outputType.getWidth() * kernelSize[0]
                * kernelSize[1];

        //During training: have im2col array, in-place gradient calculation, then epsilons...
        //But: im2col array may be cached...
        Map<CacheMode, Long> trainWorkingMemoryPerEx = new HashMap<>();
        Map<CacheMode, Long> cachedPerEx = new HashMap<>();

        //During backprop: im2col array for forward pass (possibly cached) + the epsilon6d array required to calculate
        // the 4d epsilons (equal size to input)
        //Note that the eps6d array is same size as im2col
        for (CacheMode cm : CacheMode.values()) {
            long trainWorkingSizePerEx;
            long cacheMemSizePerEx = 0;
            if (cm == CacheMode.NONE) {
                trainWorkingSizePerEx = 2 * im2colSizePerEx;
            } else {
                //im2col is cached, but epsNext2d/eps6d is not
                cacheMemSizePerEx = im2colSizePerEx;
                trainWorkingSizePerEx = im2colSizePerEx;
            }

            if (getIDropout() != null) {
                //Dup on the input before dropout, but only for training
                trainWorkingSizePerEx += inputType.arrayElementsPerExample();
            }

            trainWorkingMemoryPerEx.put(cm, trainWorkingSizePerEx);
            cachedPerEx.put(cm, cacheMemSizePerEx);
        }

        return new LayerMemoryReport.Builder(layerName, ConvolutionLayer.class, inputType, outputType)
                .standardMemory(paramSize, updaterStateSize)
                //im2col caching -> only variable size caching
                .workingMemory(0, im2colSizePerEx, MemoryReport.CACHE_MODE_ALL_ZEROS, trainWorkingMemoryPerEx)
                .cacheMemory(MemoryReport.CACHE_MODE_ALL_ZEROS, cachedPerEx).build();

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

        protected CNN2DFormat dataFormat = CNN2DFormat.NCHW;

        @Override
        protected boolean allowCausal() {
            //Causal convolution - allowed for 1D only
            return false;
        }

        /**
         * Size of the convolution rows/columns
         *
         * @param kernelSize the height and width of the kernel
         */
        public Builder kernelSize(int... kernelSize) {
            this.setKernelSize(kernelSize);
            return this;
        }

        public Builder stride(int... stride) {
            this.setStride(stride);
            return this;
        }

        public Builder padding(int... padding) {
            this.setPadding(padding);
            return this;
        }

        /**
         * Set the data format for the CNN activations - NCHW (channels first) or NHWC (channels last).
         * See {@link CNN2DFormat} for more details.<br>
         * Default: NCHW
         * @param format Format for activations (in and out)
         */
        public Builder dataFormat(CNN2DFormat format){
            this.dataFormat = format;
            return this;
        }

        @Override
        @SuppressWarnings("unchecked")
        public ConvolutionLayer build() {
            ConvolutionUtils.validateConvolutionModePadding(convolutionMode, padding);
            ConvolutionUtils.validateCnnKernelStridePadding(kernelSize, stride, padding);

            return new ConvolutionLayer(this);
        }

        /**
         * Set kernel size for 3D convolutions in (depth, height, width) order
         *
         * @param kernelSize kernel size
         */
        @Override
        public void setKernelSize(int... kernelSize) {
            this.kernelSize = ValidationUtils.validate2NonNegative(kernelSize, false, "kernelSize");
        }

        /**
         * Set stride size for 3D convolutions in (depth, height, width) order
         *
         * @param stride kernel size
         */
        @Override
        public void setStride(int... stride) {
            this.stride = ValidationUtils.validate2NonNegative(stride, false, "stride");
        }

        /**
         * Set padding size for 3D convolutions in (depth, height, width) order
         *
         * @param padding kernel size
         */
        @Override
        public void setPadding(int... padding) {
            this.padding = ValidationUtils.validate2NonNegative(padding, false, "padding");
        }

        /**
         * Set dilation size for 3D convolutions in (depth, height, width) order
         *
         * @param dilation kernel size
         */
        @Override
        public void setDilation(int... dilation) {
            this.dilation = ValidationUtils.validate2NonNegative(dilation, false, "dilation");
        }

        public void setDataFormat(CNN2DFormat dataFormat){
            this.dataFormat = dataFormat;
        }
    }

    @Getter
    @Setter
    public static abstract class BaseConvBuilder<T extends BaseConvBuilder<T>> extends FeedForwardLayer.Builder<T> {

        protected int convolutionDim = 2; // 2D convolution by default

        /**
         * If true (default): include bias parameters in the model. False: no bias.
         *
         */
        protected boolean hasBias = true;

        /**
         * Set the convolution mode for the Convolution layer. See {@link ConvolutionMode} for more details
         *
         */
        protected ConvolutionMode convolutionMode;

        /**
         * Kernel dilation. Default: {1, 1}, which is standard convolutions. Used for implementing dilated convolutions,
         * which are also known as atrous convolutions.
         * <p>
         * For more details, see:
         * <a href="https://arxiv.org/abs/1511.07122">Yu and Koltun (2014)</a> and
         * <a href="https://arxiv.org/abs/1412.7062">Chen et al. (2014)</a>, as well as
         * <a href="http://deeplearning.net/software/theano/tutorial/conv_arithmetic.html#dilated-convolutions">
         * http://deeplearning.net/software/theano/tutorial/conv_arithmetic.html#dilated-convolutions</a><br>
         *
         */
        protected int[] dilation = new int[] {1, 1};
        public int[] kernelSize = new int[] {5, 5};
        protected int[] stride = new int[] {1, 1};
        protected int[] padding = new int[] {0, 0};

        /**
         * Defaults to "PREFER_FASTEST", but "NO_WORKSPACE" uses less memory.
         */
        protected AlgoMode cudnnAlgoMode = null;
        protected FwdAlgo cudnnFwdAlgo;
        protected BwdFilterAlgo cudnnBwdFilterAlgo;
        protected BwdDataAlgo cudnnBwdDataAlgo;

        /**
         * When using CuDNN and an error is encountered, should fallback to the non-CuDNN implementatation be allowed?
         * If set to false, an exception in CuDNN will be propagated back to the user. If false, the built-in
         * (non-CuDNN) implementation for ConvolutionLayer will be used
         *
         */
        protected boolean cudnnAllowFallback = true;


        protected BaseConvBuilder(int[] kernelSize, int[] stride, int[] padding, int[] dilation, int dim) {
            this.setKernelSize(kernelSize);
            this.setStride(stride);
            this.setPadding(padding);
            this.setDilation(dilation);
            this.setConvolutionDim(dim);
        }

        protected BaseConvBuilder(int[] kernelSize, int[] stride, int[] padding, int[] dilation) {
            this.setKernelSize(kernelSize);
            this.setStride(stride);
            this.setPadding(padding);
            this.setDilation(dilation);
        }

        protected BaseConvBuilder(int[] kernelSize, int[] stride, int[] padding, int dim) {
            this.setKernelSize(kernelSize);
            this.setStride(stride);
            this.setPadding(padding);
            this.setConvolutionDim(dim);
        }

        protected BaseConvBuilder(int[] kernelSize, int[] stride, int[] padding) {
            this.setKernelSize(kernelSize);
            this.setStride(stride);
            this.setPadding(padding);
        }

        protected BaseConvBuilder(int[] kernelSize, int[] stride, int dim) {
            this.setKernelSize(kernelSize);
            this.setStride(stride);
            this.setConvolutionDim(dim);
        }


        protected BaseConvBuilder(int[] kernelSize, int[] stride) {
            this.setKernelSize(kernelSize);
            this.setStride(stride);
        }

        protected BaseConvBuilder(int dim, int... kernelSize) {
            this.setKernelSize(kernelSize);
            this.setConvolutionDim(dim);
        }


        protected BaseConvBuilder(int... kernelSize) {
            this.setKernelSize(kernelSize);
        }

        protected BaseConvBuilder() {}

        protected abstract boolean allowCausal();

        protected void setConvolutionMode(ConvolutionMode convolutionMode){
            Preconditions.checkState(allowCausal() || convolutionMode != ConvolutionMode.Causal, "Causal convolution mode can only be used with 1D" +
                    " convolutional neural network layers");
            this.convolutionMode = convolutionMode;
        }


        /**
         * If true (default): include bias parameters in the model. False: no bias.
         *
         * @param hasBias If true: include bias parameters in this model
         */
        public T hasBias(boolean hasBias) {
            this.setHasBias(hasBias);
            return (T) this;
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

        /**
         * Kernel dilation. Default: {1, 1}, which is standard convolutions. Used for implementing dilated convolutions,
         * which are also known as atrous convolutions.
         * <p>
         * For more details, see:
         * <a href="https://arxiv.org/abs/1511.07122">Yu and Koltun (2014)</a> and
         * <a href="https://arxiv.org/abs/1412.7062">Chen et al. (2014)</a>, as well as
         * <a href="http://deeplearning.net/software/theano/tutorial/conv_arithmetic.html#dilated-convolutions">
         * http://deeplearning.net/software/theano/tutorial/conv_arithmetic.html#dilated-convolutions</a><br>
         *
         * @param dilation Dilation for kernel
         */
        public T dilation(int... dilation) {
            this.setDilation(dilation);
            return (T) this;
        }

        public T kernelSize(int... kernelSize) {
            this.setKernelSize(kernelSize);
            return (T) this;
        }

        public T stride(int... stride) {
            this.setStride(stride);
            return (T) this;
        }

        public T padding(int... padding) {
            this.setPadding(padding);
            return (T) this;
        }

        /**
         * Defaults to "PREFER_FASTEST", but "NO_WORKSPACE" uses less memory.
         */
        public T cudnnAlgoMode(AlgoMode cudnnAlgoMode) {
            this.setCudnnAlgoMode(cudnnAlgoMode);
            return (T) this;
        }

        public T cudnnFwdMode(FwdAlgo cudnnFwdAlgo) {
            this.setCudnnFwdAlgo(cudnnFwdAlgo);
            return (T) this;
        }

        public T cudnnBwdFilterMode(BwdFilterAlgo cudnnBwdFilterAlgo) {
            this.setCudnnBwdFilterAlgo(cudnnBwdFilterAlgo);
            return (T) this;
        }

        public T cudnnBwdDataMode(BwdDataAlgo cudnnBwdDataAlgo) {
            this.setCudnnBwdDataAlgo(cudnnBwdDataAlgo);
            return (T) this;
        }

        /**
         * When using CuDNN and an error is encountered, should fallback to the non-CuDNN implementatation be allowed?
         * If set to false, an exception in CuDNN will be propagated back to the user. If true, the built-in
         * (non-CuDNN) implementation for ConvolutionLayer will be used
         *
         * @deprecated Use {@link #helperAllowFallback(boolean)}
         *
         * @param allowFallback Whether fallback to non-CuDNN implementation should be used
         */
        @Deprecated
        public T cudnnAllowFallback(boolean allowFallback) {
            this.setCudnnAllowFallback(allowFallback);
            return (T) this;
        }

        /**
         * When using CuDNN or MKLDNN and an error is encountered, should fallback to the non-helper implementation be allowed?
         * If set to false, an exception in the helper will be propagated back to the user. If true, the built-in
         * (non-MKL/CuDNN) implementation for ConvolutionLayer will be used
         *
         * @param allowFallback Whether fallback to non-CuDNN implementation should be used
         */
        public T helperAllowFallback(boolean allowFallback) {
            this.cudnnAllowFallback = allowFallback;
            return (T) this;
        }
    }
}

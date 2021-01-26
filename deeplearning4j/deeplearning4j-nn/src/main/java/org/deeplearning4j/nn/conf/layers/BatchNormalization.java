/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
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
import org.deeplearning4j.nn.api.layers.LayerConstraint;
import org.deeplearning4j.nn.conf.CNN2DFormat;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.nn.conf.memory.MemoryReport;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToCnnPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.RnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.params.BatchNormalizationParamInitializer;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.config.NoOp;
import org.nd4j.linalg.learning.regularization.Regularization;

import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.Map;

/**
 * Batch normalization layer<br> See: Ioffe and Szegedy, 2014, <i>Batch Normalization: Accelerating Deep Network
 * Training by Reducing Internal Covariate Shift</i>
 * <a href="https://arxiv.org/abs/1502.03167">https://arxiv.org/abs/1502.03167</a>
 */
@Data
@ToString(callSuper = true)
@EqualsAndHashCode(callSuper = true)
@Builder
public class BatchNormalization extends FeedForwardLayer {

    //Note: need to set defaults here in addition to builder, in case user uses no-op constructor...
    protected double decay = 0.9;
    protected double eps = 1e-5;
    protected boolean isMinibatch = true;
    protected double gamma = 1.0;
    protected double beta = 0.0;
    protected boolean lockGammaBeta = false;
    protected boolean cudnnAllowFallback = true;
    protected boolean useLogStd = false; //Default for deserialized models (1.0.0-beta3) and earlier: store variance as variance. Post 1.0.0-beta3: use log stdev instead
    protected CNN2DFormat cnn2DFormat = CNN2DFormat.NCHW;   //Default for deserialized models, 1.0.0-beta6 and earlier

    private BatchNormalization(Builder builder) {
        super(builder);
        this.decay = builder.decay;
        this.eps = builder.eps;
        this.isMinibatch = builder.isMinibatch;
        this.gamma = builder.gamma;
        this.beta = builder.beta;
        this.lockGammaBeta = builder.lockGammaBeta;
        this.cudnnAllowFallback = builder.cudnnAllowFallback;
        this.useLogStd = builder.useLogStd;
        this.cnn2DFormat = builder.cnn2DFormat;
        initializeConstraints(builder);
    }

    public BatchNormalization() {
        this(new Builder()); //Defaults from builder
    }

    @Override
    public BatchNormalization clone() {
        BatchNormalization clone = (BatchNormalization) super.clone();
        return clone;
    }

    @Override
    public Layer instantiate(NeuralNetConfiguration conf, Collection<TrainingListener> trainingListeners,
                             int layerIndex, INDArray layerParamsView, boolean initializeParams, DataType networkDataType) {
        LayerValidation.assertNOutSet("BatchNormalization", getLayerName(), layerIndex, getNOut());

        org.deeplearning4j.nn.layers.normalization.BatchNormalization ret =
                        new org.deeplearning4j.nn.layers.normalization.BatchNormalization(conf, networkDataType);
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
        return BatchNormalizationParamInitializer.getInstance();
    }


    @Override
    public InputType getOutputType(int layerIndex, InputType inputType) {
        if (inputType == null) {
            throw new IllegalStateException(
                            "Invalid input type: Batch norm layer expected input of type CNN, got null for layer \""
                                            + getLayerName() + "\"");
        }

        //Can handle CNN, flat CNN, CNN3D or FF input formats only
        switch (inputType.getType()) {
            case FF:
            case CNN:
            case CNNFlat:
            case CNN3D:
                return inputType; //OK
            default:
                throw new IllegalStateException(
                                "Invalid input type: Batch norm layer expected input of type CNN, CNN Flat or FF, got "
                                                + inputType + " for layer index " + layerIndex + ", layer name = "
                                                + getLayerName());
        }
    }

    @Override
    public void setNIn(InputType inputType, boolean override) {
        if (nIn <= 0 || override) {
            switch (inputType.getType()) {
                case FF:
                    nIn = ((InputType.InputTypeFeedForward) inputType).getSize();
                    break;
                case CNN:
                    nIn = ((InputType.InputTypeConvolutional) inputType).getChannels();
                    cnn2DFormat = ((InputType.InputTypeConvolutional) inputType).getFormat();
                    break;
                case CNN3D:
                    nIn = ((InputType.InputTypeConvolutional3D) inputType).getChannels();
                    break;
                case CNNFlat:
                    nIn = ((InputType.InputTypeConvolutionalFlat) inputType).getDepth();
                default:
                    throw new IllegalStateException(
                                    "Invalid input type: Batch norm layer expected input of type CNN, CNN Flat or FF, got "
                                                    + inputType + " for layer " + getLayerName() + "\"");
            }
            nOut = nIn;
        }
    }

    @Override
    public InputPreProcessor getPreProcessorForInputType(InputType inputType) {
        if (inputType.getType() == InputType.Type.CNNFlat) {
            InputType.InputTypeConvolutionalFlat i = (InputType.InputTypeConvolutionalFlat) inputType;
            return new FeedForwardToCnnPreProcessor(i.getHeight(), i.getWidth(), i.getDepth());
        } else if (inputType.getType() == InputType.Type.RNN) {
            return new RnnToFeedForwardPreProcessor();
        }

        return null;
    }

    @Override
    public List<Regularization> getRegularizationByParam(String paramName){
        //Don't regularize batch norm params: similar to biases in the sense that there are not many of them...
        return null;
    }

    @Override
    public IUpdater getUpdaterByParam(String paramName) {
        switch (paramName) {
            case BatchNormalizationParamInitializer.BETA:
            case BatchNormalizationParamInitializer.GAMMA:
                return iUpdater;
            case BatchNormalizationParamInitializer.GLOBAL_MEAN:
            case BatchNormalizationParamInitializer.GLOBAL_VAR:
            case BatchNormalizationParamInitializer.GLOBAL_LOG_STD:
                return new NoOp();
            default:
                throw new IllegalArgumentException("Unknown parameter: \"" + paramName + "\"");
        }
    }

    @Override
    public LayerMemoryReport getMemoryReport(InputType inputType) {
        InputType outputType = getOutputType(-1, inputType);

        //TODO CuDNN helper etc

        val numParams = initializer().numParams(this);
        int updaterStateSize = 0;

        for (String s : BatchNormalizationParamInitializer.getInstance().paramKeys(this)) {
            updaterStateSize += getUpdaterByParam(s).stateSize(nOut);
        }

        //During forward pass: working memory size approx. equal to 2x input size (copy ops, etc)
        val inferenceWorkingSize = 2 * inputType.arrayElementsPerExample();

        //During training: we calculate mean and variance... result is equal to nOut, and INDEPENDENT of minibatch size
        val trainWorkFixed = 2 * nOut;
        //During backprop: multiple working arrays... output size, 2 * output size (indep. of example size),
        val trainWorkingSizePerExample = inferenceWorkingSize //Inference during backprop
                        + (outputType.arrayElementsPerExample() + 2 * nOut); //Backprop gradient calculation

        return new LayerMemoryReport.Builder(layerName, BatchNormalization.class, inputType, outputType)
                        .standardMemory(numParams, updaterStateSize)
                        .workingMemory(0, 0, trainWorkFixed, trainWorkingSizePerExample) //No additional memory (beyond activations) for inference
                        .cacheMemory(MemoryReport.CACHE_MODE_ALL_ZEROS, MemoryReport.CACHE_MODE_ALL_ZEROS) //No caching
                        .build();
    }

    @Override
    public boolean isPretrainParam(String paramName) {
        return false; //No pretrain params in BN
    }

    @AllArgsConstructor
    @Getter
    @Setter
    public static class Builder extends FeedForwardLayer.Builder<Builder> {

        /**
         * At test time: we can use a global estimate of the mean and variance, calculated using a moving average of the
         * batch means/variances. This moving average is implemented as:<br> globalMeanEstimate = decay *
         * globalMeanEstimate + (1-decay) * batchMean<br> globalVarianceEstimate = decay * globalVarianceEstimate +
         * (1-decay) * batchVariance<br>
         *
         */
        protected double decay = 0.9;

        /**
         * Epsilon value for batch normalization; small floating point value added to variance (algorithm 1 in <a
         * href="https://arxiv.org/pdf/1502.03167v3.pdf">https://arxiv.org/pdf/1502.03167v3.pdf</a>) to reduce/avoid
         * underflow issues.<br> Default: 1e-5
         */
        protected double eps = 1e-5;

        /**
         * If doing minibatch training or not. Default: true. Under most circumstances, this should be set to true. If
         * doing full batch training (i.e., all examples in a single DataSet object - very small data sets) then this
         * should be set to false. Affects how global mean/variance estimates are calculated.
         *
         */
        protected boolean isMinibatch = true; // TODO auto set this if layer conf is batch

        /**
         * If set to true: lock the gamma and beta parameters to the values for each activation, specified by {@link
         * #gamma(double)} and {@link #beta(double)}. Default: false -> learn gamma and beta parameter values during
         * network training.
         *
         */
        protected boolean lockGammaBeta = false;

        /**
         * Used only when 'true' is passed to {@link #lockGammaBeta(boolean)}. Value is not used otherwise.<br> Default:
         * 1.0
         *
         */
        protected double gamma = 1.0;

        /**
         * Used only when 'true' is passed to {@link #lockGammaBeta(boolean)}. Value is not used otherwise.<br> Default:
         * 0.0
         *
         */
        protected double beta = 0.0;

        /**
         * Set constraints to be applied to the beta parameter of this batch normalisation layer. Default: no
         * constraints.<br> Constraints can be used to enforce certain conditions (non-negativity of parameters,
         * max-norm regularization, etc). These constraints are applied at each iteration, after the parameters have
         * been updated.
         *
         */
        protected List<LayerConstraint> betaConstraints;

        /**
         * Set constraints to be applied to the gamma parameter of this batch normalisation layer. Default: no
         * constraints.<br> Constraints can be used to enforce certain conditions (non-negativity of parameters,
         * max-norm regularization, etc). These constraints are applied at each iteration, after the parameters have
         * been updated.
         *
         */
        protected List<LayerConstraint> gammaConstraints;

        /**
         * When using CuDNN and an error is encountered, should fallback to the non-CuDNN implementatation be allowed?
         * If set to false, an exception in CuDNN will be propagated back to the user. If false, the built-in
         * (non-CuDNN) implementation for BatchNormalization will be used
         *
         */
        protected boolean cudnnAllowFallback = true;

        /**
         * How should the moving average of variance be stored? Two different parameterizations are supported.
         * useLogStd(false): equivalent to 1.0.0-beta3 and earlier. The variance "parameter" is stored directly as
         * variable<br> useLogStd(true): (Default) variance is stored as log10(stdev)<br> The motivation here is for
         * numerical stability (FP16 etc) and also distributed training: storing the variance directly can cause
         * numerical issues. For example, a standard deviation of 1e-3 (something that could be encountered in practice)
         * gives a variance of 1e-6, which can be problematic for 16-bit floating point
         */
        protected boolean useLogStd = true;

        protected CNN2DFormat cnn2DFormat = CNN2DFormat.NCHW;   //Default for deserialized models, 1.0.0-beta6 and earlier

        public Builder(double decay, boolean isMinibatch) {
            this.setDecay(decay);
            this.setMinibatch(isMinibatch);
        }

        public Builder(double gamma, double beta) {
            this.setGamma(gamma);
            this.setBeta(beta);
        }

        public Builder(double gamma, double beta, boolean lockGammaBeta) {
            this.setGamma(gamma);
            this.setBeta(beta);
            this.setLockGammaBeta(lockGammaBeta);
        }

        public Builder(boolean lockGammaBeta) {
            this.setLockGammaBeta(lockGammaBeta);
        }

        public Builder() {}

        /**
         * Set the input and output array data format. Defaults to NCHW format - i.e., channels first.
         * See {@link CNN2DFormat} for more details
         * @param format Format to use
         */
        public Builder dataFormat(CNN2DFormat format){
            this.cnn2DFormat = format;
            return this;
        }

        /**
         * If doing minibatch training or not. Default: true. Under most circumstances, this should be set to true. If
         * doing full batch training (i.e., all examples in a single DataSet object - very small data sets) then this
         * should be set to false. Affects how global mean/variance estimates are calculated.
         *
         * @param minibatch Minibatch parameter
         */
        public Builder minibatch(boolean minibatch) {
            this.setMinibatch(minibatch);
            return this;
        }

        /**
         * Used only when 'true' is passed to {@link #lockGammaBeta(boolean)}. Value is not used otherwise.<br> Default:
         * 1.0
         *
         * @param gamma Gamma parameter for all activations, used only with locked gamma/beta configuration mode
         */
        public Builder gamma(double gamma) {
            this.setGamma(gamma);
            return this;
        }

        /**
         * Used only when 'true' is passed to {@link #lockGammaBeta(boolean)}. Value is not used otherwise.<br> Default:
         * 0.0
         *
         * @param beta Beta parameter for all activations, used only with locked gamma/beta configuration mode
         */
        public Builder beta(double beta) {
            this.setBeta(beta);
            return this;
        }

        /**
         * Epsilon value for batch normalization; small floating point value added to variance (algorithm 1 in <a
         * href="https://arxiv.org/pdf/1502.03167v3.pdf">https://arxiv.org/pdf/1502.03167v3.pdf</a>) to reduce/avoid
         * underflow issues.<br> Default: 1e-5
         *
         * @param eps Epsilon values to use
         */
        public Builder eps(double eps) {
            this.setEps(eps);
            return this;
        }

        /**
         * At test time: we can use a global estimate of the mean and variance, calculated using a moving average of the
         * batch means/variances. This moving average is implemented as:<br> globalMeanEstimate = decay *
         * globalMeanEstimate + (1-decay) * batchMean<br> globalVarianceEstimate = decay * globalVarianceEstimate +
         * (1-decay) * batchVariance<br>
         *
         * @param decay Decay value to use for global stats calculation
         */
        public Builder decay(double decay) {
            this.setDecay(decay);
            return this;
        }

        /**
         * If set to true: lock the gamma and beta parameters to the values for each activation, specified by {@link
         * #gamma(double)} and {@link #beta(double)}. Default: false -> learn gamma and beta parameter values during
         * network training.
         *
         * @param lockGammaBeta If true: use fixed beta/gamma values. False: learn during
         */
        public Builder lockGammaBeta(boolean lockGammaBeta) {
            this.setLockGammaBeta(lockGammaBeta);
            return this;
        }

        /**
         * Set constraints to be applied to the beta parameter of this batch normalisation layer. Default: no
         * constraints.<br> Constraints can be used to enforce certain conditions (non-negativity of parameters,
         * max-norm regularization, etc). These constraints are applied at each iteration, after the parameters have
         * been updated.
         *
         * @param constraints Constraints to apply to the beta parameter of this layer
         */
        public Builder constrainBeta(LayerConstraint... constraints) {
            this.setBetaConstraints(Arrays.asList(constraints));
            return this;
        }

        /**
         * Set constraints to be applied to the gamma parameter of this batch normalisation layer. Default: no
         * constraints.<br> Constraints can be used to enforce certain conditions (non-negativity of parameters,
         * max-norm regularization, etc). These constraints are applied at each iteration, after the parameters have
         * been updated.
         *
         * @param constraints Constraints to apply to the gamma parameter of this layer
         */
        public Builder constrainGamma(LayerConstraint... constraints) {
            this.setGammaConstraints(Arrays.asList(constraints));
            return this;
        }

        /**
         * When using CuDNN and an error is encountered, should fallback to the non-CuDNN implementatation be allowed?
         * If set to false, an exception in CuDNN will be propagated back to the user. If true, the built-in
         * (non-CuDNN) implementation for BatchNormalization will be used
         *
         * @deprecated Use {@link #helperAllowFallback(boolean)}
         *
         * @param allowFallback Whether fallback to non-CuDNN implementation should be used
         */
        @Deprecated
        public Builder cudnnAllowFallback(boolean allowFallback) {
            this.setCudnnAllowFallback(allowFallback);
            return this;
        }

        /**
         * When using CuDNN or MKLDNN and an error is encountered, should fallback to the non-helper implementation be allowed?
         * If set to false, an exception in the helper will be propagated back to the user. If true, the built-in
         * (non-MKL/CuDNN) implementation for BatchNormalizationLayer will be used
         *
         * @param allowFallback Whether fallback to non-CuDNN implementation should be used
         */
        public Builder helperAllowFallback(boolean allowFallback) {
            this.cudnnAllowFallback = allowFallback;
            return this;
        }

        /**
         * How should the moving average of variance be stored? Two different parameterizations are supported.
         * useLogStd(false): equivalent to 1.0.0-beta3 and earlier. The variance "parameter" is stored directly as
         * variable<br> useLogStd(true): (Default) variance is stored as log10(stdev)<br> The motivation here is for
         * numerical stability (FP16 etc) and also distributed training: storing the variance directly can cause
         * numerical issues. For example, a standard deviation of 1e-3 (something that could be encountered in practice)
         * gives a variance of 1e-6, which can be problematic for 16-bit floating point
         */
        public Builder useLogStd(boolean useLogStd) {
            this.setUseLogStd(useLogStd);
            return this;
        }

        @Override
        public BatchNormalization build() {
            return new BatchNormalization(this);
        }
    }

}

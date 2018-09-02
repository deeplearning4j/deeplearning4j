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

package org.deeplearning4j.nn.transferlearning;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.api.layers.LayerConstraint;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.distribution.Distribution;
import org.deeplearning4j.nn.conf.dropout.Dropout;
import org.deeplearning4j.nn.conf.dropout.IDropout;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.stepfunctions.StepFunction;
import org.deeplearning4j.nn.conf.weightnoise.IWeightNoise;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.primitives.Optional;
import org.nd4j.shade.jackson.annotation.JsonInclude;
import org.nd4j.shade.jackson.annotation.JsonTypeInfo;
import org.nd4j.shade.jackson.core.JsonProcessingException;

import java.io.IOException;
import java.util.List;

/**
 * Configuration for fine tuning. Note that values set here will override values for all non-frozen layers
 *
 * @author Alex Black
 * @author Susan Eraly
 */
@JsonTypeInfo(use = JsonTypeInfo.Id.CLASS, include = JsonTypeInfo.As.PROPERTY, property = "type")
@JsonInclude(JsonInclude.Include.NON_NULL)
@NoArgsConstructor
@AllArgsConstructor
@Data
public class FineTuneConfiguration {

    protected IActivation activationFn;
    protected WeightInit weightInit;
    protected Double biasInit;
    protected Distribution dist;
    protected Double l1;
    protected Double l2;
    protected Double l1Bias;
    protected Double l2Bias;
    protected Optional<IDropout> dropout;
    protected Optional<IWeightNoise> weightNoise;
    protected IUpdater updater;
    protected IUpdater biasUpdater;
    protected Boolean miniBatch;
    protected Integer maxNumLineSearchIterations;
    protected Long seed;
    protected OptimizationAlgorithm optimizationAlgo;
    protected StepFunction stepFunction;
    protected Boolean minimize;
    protected Optional<GradientNormalization> gradientNormalization;
    protected Double gradientNormalizationThreshold;
    protected ConvolutionMode convolutionMode;
    protected ConvolutionLayer.AlgoMode cudnnAlgoMode;
    protected Optional<List<LayerConstraint>> constraints;

    protected Boolean pretrain;
    protected Boolean backprop;
    protected BackpropType backpropType;
    protected Integer tbpttFwdLength;
    protected Integer tbpttBackLength;

    protected WorkspaceMode trainingWorkspaceMode;
    protected WorkspaceMode inferenceWorkspaceMode;

    public static Builder builder() {
        return new Builder();
    }

    /*
     * Can't use Lombok @Builder annotation due to optionals (otherwise we have a bunch of ugly .x(Optional<T> value)
      * methods - lombok builder doesn't support excluding fields? :(
     * Note the use of optional here: gives us 3 states...
     * 1. Null: not set
     * 2. Optional (empty): set to null
     * 3. Optional (not empty): set to specific value
     *
     * Obviously, having null only makes sense for some things (dropout, etc) whereas null for other things doesn't
     * make sense
     */
    public static class Builder {
        private IActivation activation;
        private WeightInit weightInit;
        private Double biasInit;
        private Distribution dist;
        private Double l1;
        private Double l2;
        private Double l1Bias;
        private Double l2Bias;
        private Optional<IDropout> dropout;
        private Optional<IWeightNoise> weightNoise;
        private IUpdater updater;
        private IUpdater biasUpdater;
        private Boolean miniBatch;
        private Integer maxNumLineSearchIterations;
        private Long seed;
        private OptimizationAlgorithm optimizationAlgo;
        private StepFunction stepFunction;
        private Boolean minimize;
        private Optional<GradientNormalization> gradientNormalization;
        private Double gradientNormalizationThreshold;
        private ConvolutionMode convolutionMode;
        private ConvolutionLayer.AlgoMode cudnnAlgoMode;
        private Optional<List<LayerConstraint>> constraints;
        private Boolean pretrain;
        private Boolean backprop;
        private BackpropType backpropType;
        private Integer tbpttFwdLength;
        private Integer tbpttBackLength;
        private WorkspaceMode trainingWorkspaceMode;
        private WorkspaceMode inferenceWorkspaceMode;

        public Builder() {

        }

        /**
         * Activation function / neuron non-linearity
         */
        public Builder activation(IActivation activationFn) {
            this.activation = activationFn;
            return this;
        }

        /**
         * Activation function / neuron non-linearity
         */
        public Builder activation(Activation activation) {
            this.activation = activation.getActivationFunction();
            return this;
        }

        /**
         * Weight initialization scheme
         *
         * @see org.deeplearning4j.nn.weights.WeightInit
         */
        public Builder weightInit(WeightInit weightInit) {
            this.weightInit = weightInit;
            return this;
        }

        /**
         * Set weight initialization scheme to random sampling via the specified distribution.<br>
         * Equivalent to: {@code .weightInit(WeightInit.DISTRIBUTION).dist(distribution)}
         *
         * @param distribution Distribution to use for weight initialization
         */
        public Builder weightInit(Distribution distribution){
            weightInit(WeightInit.DISTRIBUTION);
            return dist(distribution);
        }

        /**
         * Constant for bias initialization. Default: 0.0
         *
         * @param biasInit Constant for bias initialization
         */
        public Builder biasInit(double biasInit) {
            this.biasInit = biasInit;
            return this;
        }

        /**
         * Distribution to sample initial weights from. Used in conjunction with .weightInit(WeightInit.DISTRIBUTION)
         *
         * @see #weightInit(Distribution)
         */
        public Builder dist(Distribution dist) {
            this.dist = dist;
            return this;
        }

        /**
         * L1 regularization coefficient for the weights
         */
        public Builder l1(double l1) {
            this.l1 = l1;
            return this;
        }

        /**
         * L2 regularization coefficient for the weights
         */
        public Builder l2(double l2) {
            this.l2 = l2;
            return this;
        }

        /**
         * L1 regularization coefficient for the bias parameters
         */
        public Builder l1Bias(double l1Bias) {
            this.l1Bias = l1Bias;
            return this;
        }

        /**
         * L2 regularization coefficient for the bias parameters
         */
        public Builder l2Bias(double l2Bias) {
            this.l2Bias = l2Bias;
            return this;
        }

        /**
         * Set the dropout
         *
         * @param dropout Dropout, such as {@link Dropout}, {@link org.deeplearning4j.nn.conf.dropout.GaussianDropout},
         *                {@link org.deeplearning4j.nn.conf.dropout.GaussianNoise} etc
         */
        public Builder dropout(IDropout dropout) {
            this.dropout = Optional.ofNullable(dropout);
            return this;
        }

        /**
         * Dropout probability. This is the probability of <it>retaining</it> each input activation value for a layer.
         * dropOut(x) will keep an input activation with probability x, and set to 0 with probability 1-x.<br>
         * dropOut(0.0) is a special value / special case - when set to 0.0., dropout is disabled (not applied). Note
         * that a dropout value of 1.0 is functionally equivalent to no dropout: i.e., 100% probability of retaining
         * each input activation.<br>
         * <p>
         * Note 1: Dropout is applied at training time only - and is automatically not applied at test time
         * (for evaluation, etc)<br>
         * Note 2: This sets the probability per-layer. Care should be taken when setting lower values for
         * complex networks (too much information may be lost with aggressive (very low) dropout values).<br>
         * Note 3: Frequently, dropout is not applied to (or, has higher retain probability for) input (first layer)
         * layers. Dropout is also often not applied to output layers. This needs to be handled MANUALLY by the user
         * - set .dropout(0) on those layers when using global dropout setting.<br>
         * Note 4: Implementation detail (most users can ignore): DL4J uses inverted dropout, as described here:
         * <a href="http://cs231n.github.io/neural-networks-2/">http://cs231n.github.io/neural-networks-2/</a>
         * </p>
         *
         * @param inputRetainProbability Dropout probability (probability of retaining each input activation value for a layer)
         * @see #dropout(IDropout)
         */
        public Builder dropOut(double inputRetainProbability){
            if(inputRetainProbability == 0.0){
                return dropout(null);
            }
            return dropout(new Dropout(inputRetainProbability));
        }

        /**
         * Set the weight noise (such as {@link org.deeplearning4j.nn.conf.weightnoise.DropConnect} and
         * {@link org.deeplearning4j.nn.conf.weightnoise.WeightNoise})
         *
         * @param weightNoise Weight noise instance to use
         */
        public Builder weightNoise(IWeightNoise weightNoise) {
            this.weightNoise = Optional.ofNullable(weightNoise);
            return this;
        }

        /**
         * Gradient updater configuration. For example, {@link org.nd4j.linalg.learning.config.Adam}
         * or {@link org.nd4j.linalg.learning.config.Nesterovs}
         *
         * @param updater Updater to use
         */
        public Builder updater(IUpdater updater) {
            this.updater = updater;
            return this;
        }

        /**
         * @deprecated Use {@link #updater(IUpdater)}
         */
        @Deprecated
        public Builder updater(Updater updater) {
            return updater(updater.getIUpdaterWithDefaultConfig());
        }

        /**
         * Gradient updater configuration, for the biases only. If not set, biases will use the updater as
         * set by {@link #updater(IUpdater)}
         *
         * @param biasUpdater Updater to use for bias parameters
         */
        public Builder biasUpdater(IUpdater biasUpdater) {
            this.biasUpdater = biasUpdater;
            return this;
        }

        /**
         * Whether scores and gradients should be divided by the minibatch size.<br>
         * Most users should leave this ast he default value of true.
         */
        public Builder miniBatch(boolean miniBatch) {
            this.miniBatch = miniBatch;
            return this;
        }

        public Builder maxNumLineSearchIterations(int maxNumLineSearchIterations) {
            this.maxNumLineSearchIterations = maxNumLineSearchIterations;
            return this;
        }

        /**
         * RNG seed for reproducibility
         * @param seed RNG seed to use
         */
        public Builder seed(long seed) {
            this.seed = seed;
            return this;
        }

        /**
         * RNG seed for reproducibility
         * @param seed RNG seed to use
         */
        public Builder seed(int seed){
            return seed((long)seed);
        }

        public Builder optimizationAlgo(OptimizationAlgorithm optimizationAlgo) {
            this.optimizationAlgo = optimizationAlgo;
            return this;
        }

        public Builder stepFunction(StepFunction stepFunction) {
            this.stepFunction = stepFunction;
            return this;
        }

        public Builder minimize(boolean minimize) {
            this.minimize = minimize;
            return this;
        }

        /**
         * Gradient normalization strategy. Used to specify gradient renormalization, gradient clipping etc.
         * See {@link GradientNormalization} for details
         *
         * @param gradientNormalization Type of normalization to use. Defaults to None.
         * @see GradientNormalization
         */
        public Builder gradientNormalization(GradientNormalization gradientNormalization) {
            this.gradientNormalization = Optional.ofNullable(gradientNormalization);
            return this;
        }

        /**
         * Threshold for gradient normalization, only used for GradientNormalization.ClipL2PerLayer,
         * GradientNormalization.ClipL2PerParamType, and GradientNormalization.ClipElementWiseAbsoluteValue<br>
         * Not used otherwise.<br>
         * L2 threshold for first two types of clipping, or absolute value threshold for last type of clipping
         */
        public Builder gradientNormalizationThreshold(double gradientNormalizationThreshold) {
            this.gradientNormalizationThreshold = gradientNormalizationThreshold;
            return this;
        }

        /**
         * Sets the convolution mode for convolutional layers, which impacts padding and output sizes.
         * See {@link ConvolutionMode} for details. Defaults to ConvolutionMode.TRUNCATE<br>
         * @param convolutionMode Convolution mode to use
         */
        public Builder convolutionMode(ConvolutionMode convolutionMode) {
            this.convolutionMode = convolutionMode;
            return this;
        }

        /**
         * Sets the cuDNN algo mode for convolutional layers, which impacts performance and memory usage of cuDNN.
         * See {@link ConvolutionLayer.AlgoMode} for details.  Defaults to "PREFER_FASTEST", but "NO_WORKSPACE" uses less memory.
         */
        public Builder cudnnAlgoMode(ConvolutionLayer.AlgoMode cudnnAlgoMode) {
            this.cudnnAlgoMode = cudnnAlgoMode;
            return this;
        }

        /**
         * Set constraints to be applied to all layers. Default: no constraints.<br>
         * Constraints can be used to enforce certain conditions (non-negativity of parameters, max-norm regularization,
         * etc). These constraints are applied at each iteration, after the parameters have been updated.
         *
         * @param constraints Constraints to apply to all parameters of all layers
         */
        public Builder constraints(List<LayerConstraint> constraints) {
            this.constraints = Optional.ofNullable(constraints);
            return this;
        }

        public Builder pretrain(boolean pretrain) {
            this.pretrain = pretrain;
            return this;
        }

        public Builder backprop(boolean backprop) {
            this.backprop = backprop;
            return this;
        }

        /**
         * The type of backprop. Default setting is used for most networks (MLP, CNN etc),
         * but optionally truncated BPTT can be used for training recurrent neural networks.
         * If using TruncatedBPTT make sure you set both tBPTTForwardLength() and tBPTTBackwardLength()
         *
         * @param backpropType Type of backprop. Default: BackpropType.Standard
         */
        public Builder backpropType(BackpropType backpropType) {
            this.backpropType = backpropType;
            return this;
        }

        /**
         * When doing truncated BPTT: how many steps of forward pass should we do
         * before doing (truncated) backprop?<br>
         * Only applicable when doing backpropType(BackpropType.TruncatedBPTT)<br>
         * Typically tBPTTForwardLength parameter is same as the tBPTTBackwardLength parameter,
         * but may be larger than it in some circumstances (but never smaller)<br>
         * Ideally your training data time series length should be divisible by this
         * This is the k1 parameter on pg23 of
         * <a href="http://www.cs.utoronto.ca/~ilya/pubs/ilya_sutskever_phd_thesis.pdf">http://www.cs.utoronto.ca/~ilya/pubs/ilya_sutskever_phd_thesis.pdf</a>
         *
         * @param tbpttFwdLength Forward length > 0, >= backwardLength
         */
        public Builder tbpttFwdLength(int tbpttFwdLength) {
            this.tbpttFwdLength = tbpttFwdLength;
            return this;
        }

        /**
         * When doing truncated BPTT: how many steps of backward should we do?<br>
         * Only applicable when doing backpropType(BackpropType.TruncatedBPTT)<br>
         * This is the k2 parameter on pg23 of
         * <a href="http://www.cs.utoronto.ca/~ilya/pubs/ilya_sutskever_phd_thesis.pdf">http://www.cs.utoronto.ca/~ilya/pubs/ilya_sutskever_phd_thesis.pdf</a>
         *
         * @param tbpttBackLength <= forwardLength
         */
        public Builder tbpttBackLength(int tbpttBackLength) {
            this.tbpttBackLength = tbpttBackLength;
            return this;
        }

        /**
         * This method defines Workspace mode being used during training:
         * NONE: workspace won't be used
         * ENABLED: workspaces will be used for training (reduced memory and better performance)
         *
         * @param trainingWorkspaceMode Workspace mode for training
         * @return Builder
         */
        public Builder trainingWorkspaceMode(WorkspaceMode trainingWorkspaceMode) {
            this.trainingWorkspaceMode = trainingWorkspaceMode;
            return this;
        }

        /**
         * This method defines Workspace mode being used during inference:<br>
         * NONE: workspace won't be used<br>
         * ENABLED: workspaces will be used for inference (reduced memory and better performance)
         *
         * @param inferenceWorkspaceMode Workspace mode for inference
         * @return Builder
         */
        public Builder inferenceWorkspaceMode(WorkspaceMode inferenceWorkspaceMode) {
            this.inferenceWorkspaceMode = inferenceWorkspaceMode;
            return this;
        }

        public FineTuneConfiguration build() {
            return new FineTuneConfiguration(activation, weightInit, biasInit, dist, l1, l2, l1Bias, l2Bias, dropout, weightNoise, updater, biasUpdater, miniBatch, maxNumLineSearchIterations, seed, optimizationAlgo, stepFunction, minimize, gradientNormalization, gradientNormalizationThreshold, convolutionMode, cudnnAlgoMode, constraints, pretrain, backprop, backpropType, tbpttFwdLength, tbpttBackLength, trainingWorkspaceMode, inferenceWorkspaceMode);
        }

        public String toString() {
            return "FineTuneConfiguration.Builder(activation=" + this.activation + ", weightInit=" + this.weightInit + ", biasInit=" + this.biasInit + ", dist=" + this.dist + ", l1=" + this.l1 + ", l2=" + this.l2 + ", l1Bias=" + this.l1Bias + ", l2Bias=" + this.l2Bias + ", dropout=" + this.dropout + ", weightNoise=" + this.weightNoise + ", updater=" + this.updater + ", biasUpdater=" + this.biasUpdater + ", miniBatch=" + this.miniBatch + ", maxNumLineSearchIterations=" + this.maxNumLineSearchIterations + ", seed=" + this.seed + ", optimizationAlgo=" + this.optimizationAlgo + ", stepFunction=" + this.stepFunction + ", minimize=" + this.minimize + ", gradientNormalization=" + this.gradientNormalization + ", gradientNormalizationThreshold=" + this.gradientNormalizationThreshold + ", convolutionMode=" + this.convolutionMode + ", cudnnAlgoMode=" + this.cudnnAlgoMode + ", constraints=" + this.constraints + ", pretrain=" + this.pretrain + ", backprop=" + this.backprop + ", backpropType=" + this.backpropType + ", tbpttFwdLength=" + this.tbpttFwdLength + ", tbpttBackLength=" + this.tbpttBackLength + ", trainingWorkspaceMode=" + this.trainingWorkspaceMode + ", inferenceWorkspaceMode=" + this.inferenceWorkspaceMode + ")";
        }
    }


    public NeuralNetConfiguration appliedNeuralNetConfiguration(NeuralNetConfiguration nnc) {
        applyToNeuralNetConfiguration(nnc);
        nnc = new NeuralNetConfiguration.Builder(nnc.clone()).build();
        return nnc;
    }

    public void applyToNeuralNetConfiguration(NeuralNetConfiguration nnc) {

        Layer l = nnc.getLayer();
        Updater originalUpdater = null;
        WeightInit origWeightInit = null;

        if (l != null) {
            if (dropout != null)
                l.setIDropout(dropout.orElse(null));
            if(constraints != null)
                l.setConstraints(constraints.orElse(null));
        }

        if (l != null && l instanceof BaseLayer) {
            BaseLayer bl = (BaseLayer) l;
            origWeightInit = bl.getWeightInit();
            if (activationFn != null)
                bl.setActivationFn(activationFn);
            if (weightInit != null)
                bl.setWeightInit(weightInit);
            if (biasInit != null)
                bl.setBiasInit(biasInit);
            if (dist != null)
                bl.setDist(dist);
            if (l1 != null)
                bl.setL1(l1);
            if (l2 != null)
                bl.setL2(l2);
            if (l1Bias != null)
                bl.setL1Bias(l1Bias);
            if (l2Bias != null)
                bl.setL2Bias(l2Bias);
            if (gradientNormalization != null)
                bl.setGradientNormalization(gradientNormalization.orElse(null));
            if (gradientNormalizationThreshold != null)
                bl.setGradientNormalizationThreshold(gradientNormalizationThreshold);
            if (updater != null){
                bl.setIUpdater(updater);
            }
            if (biasUpdater != null){
                bl.setBiasUpdater(biasUpdater);
            }
            if (weightNoise != null){
                bl.setWeightNoise(weightNoise.orElse(null));
            }
        }
        if (miniBatch != null)
            nnc.setMiniBatch(miniBatch);
        if (maxNumLineSearchIterations != null)
            nnc.setMaxNumLineSearchIterations(maxNumLineSearchIterations);
        if (seed != null)
            nnc.setSeed(seed);
        if (optimizationAlgo != null)
            nnc.setOptimizationAlgo(optimizationAlgo);
        if (stepFunction != null)
            nnc.setStepFunction(stepFunction);
        if (minimize != null)
            nnc.setMinimize(minimize);

        if (convolutionMode != null && l instanceof ConvolutionLayer) {
            ((ConvolutionLayer) l).setConvolutionMode(convolutionMode);
        }
        if (cudnnAlgoMode != null && l instanceof ConvolutionLayer) {
            ((ConvolutionLayer) l).setCudnnAlgoMode(cudnnAlgoMode);
        }
        if (convolutionMode != null && l instanceof SubsamplingLayer) {
            ((SubsamplingLayer) l).setConvolutionMode(convolutionMode);
        }

        //Check weight init. Remove dist if originally was DISTRIBUTION, and isn't now -> remove no longer needed distribution
        if (l != null && l instanceof BaseLayer && origWeightInit == WeightInit.DISTRIBUTION && weightInit != null
                        && weightInit != WeightInit.DISTRIBUTION) {
            ((BaseLayer) l).setDist(null);
        }

        //Perform validation
        if (l != null) {
            LayerValidation.generalValidation(l.getLayerName(), l, get(dropout), l2, l2Bias, l1, l1Bias,
                    dist, get(constraints), null, null);
        }
    }

    private static <T> T get(Optional<T> optional){
        if(optional == null){
            return null;
        }
        return optional.orElse(null);
    }

    public void applyToMultiLayerConfiguration(MultiLayerConfiguration conf) {
        if (pretrain != null)
            conf.setPretrain(pretrain);
        if (backprop != null)
            conf.setBackprop(backprop);
        if (backpropType != null)
            conf.setBackpropType(backpropType);
        if (tbpttFwdLength != null)
            conf.setTbpttFwdLength(tbpttFwdLength);
        if (tbpttBackLength != null)
            conf.setTbpttBackLength(tbpttBackLength);
    }

    public void applyToComputationGraphConfiguration(ComputationGraphConfiguration conf) {
        if (pretrain != null)
            conf.setPretrain(pretrain);
        if (backprop != null)
            conf.setBackprop(backprop);
        if (backpropType != null)
            conf.setBackpropType(backpropType);
        if (tbpttFwdLength != null)
            conf.setTbpttFwdLength(tbpttFwdLength);
        if (tbpttBackLength != null)
            conf.setTbpttBackLength(tbpttBackLength);
    }

    public NeuralNetConfiguration.Builder appliedNeuralNetConfigurationBuilder() {
        NeuralNetConfiguration.Builder confBuilder = new NeuralNetConfiguration.Builder();
        if (activationFn != null)
            confBuilder.setActivationFn(activationFn);
        if (weightInit != null)
            confBuilder.setWeightInit(weightInit);
        if (biasInit != null)
            confBuilder.setBiasInit(biasInit);
        if (dist != null)
            confBuilder.setDist(dist);
        if (l1 != null)
            confBuilder.setL1(l1);
        if (l2 != null)
            confBuilder.setL2(l2);
        if (l1Bias != null)
            confBuilder.setL1Bias(l1Bias);
        if (l2Bias != null)
            confBuilder.setL2Bias(l2Bias);
        if (dropout != null)
            confBuilder.setIdropOut(dropout.orElse(null));
        if (updater != null)
            confBuilder.updater(updater);
        if(biasUpdater != null)
            confBuilder.biasUpdater(biasUpdater);
        if (miniBatch != null)
            confBuilder.setMiniBatch(miniBatch);
        if (maxNumLineSearchIterations != null)
            confBuilder.setMaxNumLineSearchIterations(maxNumLineSearchIterations);
        if (seed != null)
            confBuilder.setSeed(seed);
        if (optimizationAlgo != null)
            confBuilder.setOptimizationAlgo(optimizationAlgo);
        if (stepFunction != null)
            confBuilder.setStepFunction(stepFunction);
        if (minimize != null)
            confBuilder.setMinimize(minimize);
        if (gradientNormalization != null)
            confBuilder.setGradientNormalization(gradientNormalization.orElse(null));
        if (gradientNormalizationThreshold != null)
            confBuilder.setGradientNormalizationThreshold(gradientNormalizationThreshold);
        if (trainingWorkspaceMode != null)
            confBuilder.trainingWorkspaceMode(trainingWorkspaceMode);
        if (inferenceWorkspaceMode != null)
            confBuilder.inferenceWorkspaceMode(inferenceWorkspaceMode);
        return confBuilder;
    }


    public String toJson() {
        try {
            return NeuralNetConfiguration.mapper().writeValueAsString(this);
        } catch (JsonProcessingException e) {
            throw new RuntimeException(e);
        }
    }

    public String toYaml() {
        try {
            return NeuralNetConfiguration.mapperYaml().writeValueAsString(this);
        } catch (JsonProcessingException e) {
            throw new RuntimeException(e);
        }
    }

    public static FineTuneConfiguration fromJson(String json) {
        try {
            return NeuralNetConfiguration.mapper().readValue(json, FineTuneConfiguration.class);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public static FineTuneConfiguration fromYaml(String yaml) {
        try {
            return NeuralNetConfiguration.mapperYaml().readValue(yaml, FineTuneConfiguration.class);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}

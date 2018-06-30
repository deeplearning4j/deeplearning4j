/*-
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.nn.conf.layers;

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.Distribution;
import org.deeplearning4j.nn.conf.weightnoise.IWeightNoise;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.learning.config.IUpdater;

import java.io.Serializable;

/**
 * A neural network layer.
 */
@Data
@EqualsAndHashCode(callSuper = true)
@NoArgsConstructor
public abstract class BaseLayer extends Layer implements Serializable, Cloneable {
    protected IActivation activationFn;
    protected WeightInit weightInit;
    protected double biasInit;
    protected Distribution dist;
    protected double l1;
    protected double l2;
    protected double l1Bias;
    protected double l2Bias;
    protected IUpdater iUpdater;
    protected IUpdater biasUpdater;
    protected IWeightNoise weightNoise;
    protected GradientNormalization gradientNormalization = GradientNormalization.None; //Clipping, rescale based on l2 norm, etc
    protected double gradientNormalizationThreshold = 1.0; //Threshold for l2 and element-wise gradient clipping


    public BaseLayer(Builder builder) {
        super(builder);
        this.layerName = builder.layerName;
        this.activationFn = builder.activationFn;
        this.weightInit = builder.weightInit;
        this.biasInit = builder.biasInit;
        this.dist = builder.dist;
        this.l1 = builder.l1;
        this.l2 = builder.l2;
        this.l1Bias = builder.l1Bias;
        this.l2Bias = builder.l2Bias;
        this.iUpdater = builder.iupdater;
        this.biasUpdater = builder.biasUpdater;
        this.gradientNormalization = builder.gradientNormalization;
        this.gradientNormalizationThreshold = builder.gradientNormalizationThreshold;
        this.weightNoise = builder.weightNoise;
    }

    /**
     * Reset the learning related configs of the layer to default. When instantiated with a global neural network configuration
     * the parameters specified in the neural network configuration will be used.
     * For internal use with the transfer learning API. Users should not have to call this method directly.
     */
    public void resetLayerDefaultConfig() {
        //clear the learning related params for all layers in the origConf and set to defaults
        this.setIUpdater(null);
        this.setWeightInit(null);
        this.setBiasInit(Double.NaN);
        this.setDist(null);
        this.setL1(Double.NaN);
        this.setL2(Double.NaN);
        this.setGradientNormalization(GradientNormalization.None);
        this.setGradientNormalizationThreshold(1.0);
        this.iUpdater = null;
        this.biasUpdater = null;
    }

    @Override
    public BaseLayer clone() {
        BaseLayer clone = (BaseLayer) super.clone();
        if (clone.dist != null)
            clone.dist = clone.dist.clone();
        return clone;
    }

    /**
     * Get the updater for the given parameter. Typically the same updater will be used for all updaters, but this
     * is not necessarily the case
     *
     * @param paramName    Parameter name
     * @return             IUpdater for the parameter
     */
    @Override
    public IUpdater getUpdaterByParam(String paramName) {
        if(biasUpdater != null && initializer().isBiasParam(this, paramName)){
            return biasUpdater;
        }
        return iUpdater;
    }

    @Override
    public GradientNormalization getGradientNormalization(String param){
        return gradientNormalization;
    }

    @SuppressWarnings("unchecked")
    public abstract static class Builder<T extends Builder<T>> extends Layer.Builder<T> {
        protected IActivation activationFn = null;
        protected WeightInit weightInit = null;
        protected double biasInit = Double.NaN;
        protected Distribution dist = null;
        protected double l1 = Double.NaN;
        protected double l2 = Double.NaN;
        protected double l1Bias = Double.NaN;
        protected double l2Bias = Double.NaN;
        protected IUpdater iupdater = null;
        protected IUpdater biasUpdater = null;
        protected GradientNormalization gradientNormalization = null;
        protected double gradientNormalizationThreshold = Double.NaN;
        protected IWeightNoise weightNoise;

        /**
         * Set the activation function for the layer. This overload can be used for custom {@link IActivation} instances
         *
         * @param activationFunction Activation function to use for the layer
         */
        public T activation(IActivation activationFunction) {
            this.activationFn = activationFunction;
            return (T) this;
        }

        /**
         * Set the activation function for the layer, from an {@link Activation} enumeration value.
         *
         * @param activation Activation function to use for the layer
         */
        public T activation(Activation activation) {
            return activation(activation.getActivationFunction());
        }

        /**
         * Weight initialization scheme to use, for initial weight values
         *
         * @see WeightInit
         */
        public T weightInit(WeightInit weightInit) {
            this.weightInit = weightInit;
            return (T) this;
        }

        /**
         * Set weight initialization scheme to random sampling via the specified distribution.
         * Equivalent to: {@code .weightInit(WeightInit.DISTRIBUTION).dist(distribution)}
         *
         * @param distribution Distribution to use for weight initialization
         */
        public T weightInit(Distribution distribution){
            weightInit(WeightInit.DISTRIBUTION);
            return dist(distribution);
        }

        /**
         * Bias initialization value, for layers with biases. Defaults to 0
         *
         * @param biasInit Value to use for initializing biases
         */
        public T biasInit(double biasInit) {
            this.biasInit = biasInit;
            return (T) this;
        }

        /**
         * Distribution to sample initial weights from. Used in conjunction with
         * .weightInit(WeightInit.DISTRIBUTION).
         */
        public T dist(Distribution dist) {
            this.dist = dist;
            return (T) this;
        }

        /**
         * L1 regularization coefficient (weights only). Use {@link #l1Bias(double)} to configure the l1 regularization
         * coefficient for the bias.
         */
        public T l1(double l1) {
            this.l1 = l1;
            return (T) this;
        }

        /**
         * L2 regularization coefficient (weights only). Use {@link #l2Bias(double)} to configure the l2 regularization
         * coefficient for the bias.
         */
        public T l2(double l2) {
            this.l2 = l2;
            return (T) this;
        }

        /**
         * L1 regularization coefficient for the bias. Default: 0. See also {@link #l1(double)}
         */
        public T l1Bias(double l1Bias) {
            this.l1Bias = l1Bias;
            return (T) this;
        }

        /**
         * L2 regularization coefficient for the bias. Default: 0. See also {@link #l2(double)}
         */
        public T l2Bias(double l2Bias) {
            this.l2Bias = l2Bias;
            return (T) this;
        }

        /**
         * Gradient updater. For example, SGD for standard stochastic gradient descent, NESTEROV for Nesterov momentum,
         * RSMPROP for RMSProp, etc.
         *
         * @see Updater
         */
        @Deprecated
        public T updater(Updater updater) {
            return updater(updater.getIUpdaterWithDefaultConfig());
        }

        /**
         * Gradient updater. For example, {@link org.nd4j.linalg.learning.config.Adam}
         * or {@link org.nd4j.linalg.learning.config.Nesterovs}
         *
         * @param updater Updater to use
         */
        public T updater(IUpdater updater) {
            this.iupdater = updater;
            return (T) this;
        }

        /**
         * Gradient updater configuration, for the biases only. If not set, biases will use the updater as
         * set by {@link #updater(IUpdater)}
         *
         * @param biasUpdater Updater to use for bias parameters
         */
        public T biasUpdater(IUpdater biasUpdater){
            this.biasUpdater = biasUpdater;
            return (T) this;
        }

        /**
         * Gradient normalization strategy. Used to specify gradient renormalization, gradient clipping etc.
         *
         * @param gradientNormalization Type of normalization to use. Defaults to None.
         * @see GradientNormalization
         */
        public T gradientNormalization(GradientNormalization gradientNormalization) {
            this.gradientNormalization = gradientNormalization;
            return (T) this;
        }

        /**
         * Threshold for gradient normalization, only used for GradientNormalization.ClipL2PerLayer,
         * GradientNormalization.ClipL2PerParamType, and GradientNormalization.ClipElementWiseAbsoluteValue<br>
         * Not used otherwise.<br>
         * L2 threshold for first two types of clipping, or absolute value threshold for last type of clipping.
         */
        public T gradientNormalizationThreshold(double threshold) {
            this.gradientNormalizationThreshold = threshold;
            return (T) this;
        }

        /**
         * Set the weight noise (such as {@link org.deeplearning4j.nn.conf.weightnoise.DropConnect} and
         * {@link org.deeplearning4j.nn.conf.weightnoise.WeightNoise}) for this layer
         *
         * @param weightNoise Weight noise instance to use
         */
        public T weightNoise(IWeightNoise weightNoise){
            this.weightNoise = weightNoise;
            return (T)this;
        }
    }
}

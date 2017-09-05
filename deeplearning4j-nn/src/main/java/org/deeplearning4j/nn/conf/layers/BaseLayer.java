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
import org.deeplearning4j.nn.conf.LearningRatePolicy;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.Distribution;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.learning.config.*;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;

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


        //Handle legacy config set by user:
        if(builder.updater != null && builder.iupdater == null){

            if(!Double.isNaN(builder.learningRate)){
                //User has done something like .updater(Updater.NESTEROVS).learningRate(0.2)
                switch(builder.updater){
                    case SGD:
                        iUpdater = new Sgd(builder.learningRate);
                        break;
                    case ADAM:
                        double aEps = Double.isNaN(builder.epsilon) ? Adam.DEFAULT_ADAM_EPSILON : builder.epsilon;
                        double aM = Double.isNaN(builder.adamMeanDecay) ? Adam.DEFAULT_ADAM_BETA1_MEAN_DECAY : builder.adamMeanDecay;
                        double aV = Double.isNaN(builder.adamVarDecay) ? Adam.DEFAULT_ADAM_BETA2_VAR_DECAY : builder.adamVarDecay;
                        iUpdater = new Adam(builder.learningRate, aM, aV, aEps);
                        break;
                    case ADAMAX:
                        double amEps = Double.isNaN(builder.epsilon) ? AdaMax.DEFAULT_ADAMAX_EPSILON : builder.epsilon;
                        double amM = Double.isNaN(builder.adamMeanDecay) ? AdaMax.DEFAULT_ADAMAX_BETA1_MEAN_DECAY : builder.adamMeanDecay;
                        double amV = Double.isNaN(builder.adamVarDecay) ? AdaMax.DEFAULT_ADAMAX_BETA2_VAR_DECAY : builder.adamVarDecay;
                        iUpdater = new AdaMax(builder.learningRate, amM, amV, amEps);
                        break;
                    case ADADELTA:
                        double adRho = Double.isNaN(builder.rho) ? AdaDelta.DEFAULT_ADADELTA_RHO : builder.rho;
                        double adEps = Double.isNaN(builder.epsilon) ? AdaDelta.DEFAULT_ADADELTA_EPSILON : builder.epsilon;
                        iUpdater = new AdaDelta(adRho, adEps);
                        break;
                    case NESTEROVS:
                        double nM = Double.isNaN(builder.momentum) ? Nesterovs.DEFAULT_NESTEROV_MOMENTUM : builder.momentum;
                        iUpdater = new Nesterovs(builder.learningRate, nM);
                        break;
                    case NADAM:
                        double naEps = Double.isNaN(builder.epsilon) ? Nadam.DEFAULT_NADAM_EPSILON : builder.epsilon;
                        double naM = Double.isNaN(builder.adamMeanDecay) ? Nadam.DEFAULT_NADAM_BETA1_MEAN_DECAY : builder.adamMeanDecay;
                        double naV = Double.isNaN(builder.adamVarDecay) ? Nadam.DEFAULT_NADAM_BETA2_VAR_DECAY : builder.adamVarDecay;
                        iUpdater = new Nadam(builder.learningRate, naM, naV, naEps);
                        break;
                    case ADAGRAD:
                        double agEps = Double.isNaN(builder.epsilon) ? AdaGrad.DEFAULT_ADAGRAD_EPSILON : builder.epsilon;
                        iUpdater = new AdaGrad(builder.learningRate, agEps);
                        break;
                    case RMSPROP:
                        double rEps = Double.isNaN(builder.epsilon) ? RmsProp.DEFAULT_RMSPROP_EPSILON : builder.epsilon;
                        double rRms = Double.isNaN(builder.rmsDecay) ? RmsProp.DEFAULT_RMSPROP_RMSDECAY : builder.rmsDecay;
                        iUpdater = new RmsProp(builder.learningRate, rRms, rEps);
                        break;
                    case NONE:
                        iUpdater = new NoOp();
                        break;
                    case CUSTOM:
                        //No op
                        break;
                }
            } else {
                //Use default learning rate
                this.iUpdater = builder.updater.getIUpdaterWithDefaultConfig();
            }
        }
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
    public IUpdater getIUpdaterByParam(String paramName) {
        if(biasUpdater != null && initializer().isBiasParam(paramName)){
            return biasUpdater;
        }
        return iUpdater;
    }

    @SuppressWarnings("unchecked")
    public abstract static class Builder<T extends Builder<T>> extends Layer.Builder<T> {
        protected IActivation activationFn = null;
        protected WeightInit weightInit = null;
        protected double biasInit = Double.NaN;
        protected Distribution dist = null;
        @Deprecated
        protected double learningRate = Double.NaN;
        @Deprecated
        protected double biasLearningRate = Double.NaN;
        @Deprecated
        protected Map<Integer, Double> learningRateSchedule = null;
        @Deprecated
        protected double momentum = Double.NaN;
        @Deprecated
        protected Map<Integer, Double> momentumAfter = null;
        protected double l1 = Double.NaN;
        protected double l2 = Double.NaN;
        protected double l1Bias = Double.NaN;
        protected double l2Bias = Double.NaN;
        @Deprecated
        protected Updater updater = null;
        protected IUpdater iupdater = null;
        protected IUpdater biasUpdater = null;
        @Deprecated
        protected double rho = Double.NaN;
        @Deprecated
        protected double epsilon = Double.NaN;
        @Deprecated
        protected double rmsDecay = Double.NaN;
        @Deprecated
        protected double adamMeanDecay = Double.NaN;
        @Deprecated
        protected double adamVarDecay = Double.NaN;
        protected GradientNormalization gradientNormalization = null;
        protected double gradientNormalizationThreshold = Double.NaN;
        @Deprecated
        protected LearningRatePolicy learningRatePolicy = null;


        /**
         * Layer activation function.
         * Typical values include:<br>
         * "relu" (rectified linear), "tanh", "sigmoid", "softmax",
         * "hardtanh", "leakyrelu", "maxout", "softsign", "softplus"
         * @deprecated Use {@link #activation(Activation)} or {@link @activation(IActivation)}
         */
        @Deprecated
        public T activation(String activationFunction) {
            return activation(Activation.fromString(activationFunction));
        }

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
         * Learning rate. Defaults to 1e-1
         */
        @Deprecated
        public T learningRate(double learningRate) {
            this.learningRate = learningRate;
            return (T) this;
        }

        /**
         * Bias learning rate. Set this to apply a different learning rate to the bias
         */
        @Deprecated
        public T biasLearningRate(double biasLearningRate) {
            this.biasLearningRate = biasLearningRate;
            return (T) this;
        }

        /**
         * Learning rate schedule. Map of the iteration to the learning rate to apply at that iteration.
         */
        @Deprecated
        public T learningRateSchedule(Map<Integer, Double> learningRateSchedule) {
            this.learningRateSchedule = learningRateSchedule;
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
         * Momentum rate.
         * @deprecated Use {@code .updater(new Nesterov(momentum))} instead
         */
        @Deprecated
        public T momentum(double momentum) {
            this.momentum = momentum;
            return (T) this;
        }

        /**
         * Momentum schedule. Map of the iteration to the momentum rate to apply at that iteration.
         * @deprecated Use {@code .updater(Nesterov.builder().momentumSchedule(schedule).build())} instead
         */
        @Deprecated
        public T momentumAfter(Map<Integer, Double> momentumAfter) {
            this.momentumAfter = momentumAfter;
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

        public T biasUpdater(IUpdater biasUpdater){
            this.biasUpdater = biasUpdater;
            return (T) this;
        }

        /**
         * Ada delta coefficient, rho. Only applies if using .updater(Updater.ADADELTA)
         *
         * @param rho
         * @deprecated use {@code .updater(new AdaDelta(rho,epsilon))} intead
         */
        @Deprecated
        public T rho(double rho) {
            this.rho = rho;
            return (T) this;
        }

        /**
         * Decay rate for RMSProp. Only applies if using .updater(Updater.RMSPROP)
         * @deprecated use {@code .updater(new RmsProp(rmsDecay))} instead
         */
        @Deprecated
        public T rmsDecay(double rmsDecay) {
            this.rmsDecay = rmsDecay;
            return (T) this;
        }

        /**
         * Epsilon value for updaters: Adam, RMSProp, Adagrad, Adadelta
         *
         * @param epsilon    Epsilon value to use
         * @deprecated Use use {@code .updater(Adam.builder().epsilon(epsilon).build())} or similar instead
         */
        @Deprecated
        public T epsilon(double epsilon) {
            this.epsilon = epsilon;
            return (T) this;
        }

        /**
         * Mean decay rate for Adam updater. Only applies if using .updater(Updater.ADAM)
         * @deprecated use {@code .updater(Adam.builder().beta1(adamMeanDecay).build())} intead
         */
        @Deprecated
        public T adamMeanDecay(double adamMeanDecay) {
            this.adamMeanDecay = adamMeanDecay;
            return (T) this;
        }

        /**
         * Variance decay rate for Adam updater. Only applies if using .updater(Updater.ADAM)
         * @deprecated use {@code .updater(Adam.builder().beta2(adamVarDecay).build())} intead
         */
        @Deprecated
        public T adamVarDecay(double adamVarDecay) {
            this.adamVarDecay = adamVarDecay;
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
         * Learning rate decay policy. Used to adapt learning rate based on policy.
         *
         * @param policy Type of policy to use. Defaults to None.
         * @see GradientNormalization
         */
        @Deprecated
        public T learningRateDecayPolicy(LearningRatePolicy policy) {
            this.learningRatePolicy = policy;
            return (T) this;
        }
    }
}

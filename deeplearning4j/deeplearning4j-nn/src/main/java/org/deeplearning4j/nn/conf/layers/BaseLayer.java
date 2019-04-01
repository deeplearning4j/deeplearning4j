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

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.Distribution;
import org.deeplearning4j.nn.conf.weightnoise.IWeightNoise;
import org.deeplearning4j.nn.weights.IWeightInit;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.nn.weights.WeightInitDistribution;
import org.deeplearning4j.nn.weightsharing.WeightPool;
import org.deeplearning4j.util.NetworkUtils;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.regularization.L1Regularization;
import org.nd4j.linalg.learning.regularization.L2Regularization;
import org.nd4j.linalg.learning.regularization.Regularization;
import org.nd4j.linalg.learning.regularization.WeightDecay;

/**
 * A neural network layer.
 */
@Data
@EqualsAndHashCode(callSuper = true)
@NoArgsConstructor
public abstract class BaseLayer extends Layer implements Serializable, Cloneable {

    protected IActivation activationFn;
    protected IWeightInit weightInitFn;
    protected double biasInit;
    protected double gainInit;
    protected List<Regularization> regularization;
    protected List<Regularization> regularizationBias;
    protected IUpdater iUpdater;
    protected IUpdater biasUpdater;
    protected IWeightNoise weightNoise;
    protected GradientNormalization gradientNormalization = GradientNormalization.None; //Clipping, rescale based on l2 norm, etc
    protected double gradientNormalizationThreshold = 1.0; //Threshold for l2 and element-wise gradient clipping

    protected String weightPoolId;

    public BaseLayer(Builder builder) {
        super(builder);
        this.layerName = builder.layerName;
        this.activationFn = builder.activationFn;
        this.weightInitFn = builder.weightInitFn;
        this.biasInit = builder.biasInit;
        this.gainInit = builder.gainInit;
        this.regularization = builder.regularization;
        this.regularizationBias = builder.regularizationBias;
        this.iUpdater = builder.iupdater;
        this.biasUpdater = builder.biasUpdater;
        this.gradientNormalization = builder.gradientNormalization;
        this.gradientNormalizationThreshold = builder.gradientNormalizationThreshold;
        this.weightNoise = builder.weightNoise;
        this.weightPoolId = WeightPool.getNewId();

    }

    /**
     * Reset the learning related configs of the layer to default. When instantiated with a global neural network
     * configuration the parameters specified in the neural network configuration will be used. For internal use with
     * the transfer learning API. Users should not have to call this method directly.
     */
    public void resetLayerDefaultConfig() {
        //clear the learning related params for all layers in the origConf and set to defaults
        this.setIUpdater(null);
        this.setWeightInitFn(null);
        this.setBiasInit(Double.NaN);
        this.setGainInit(Double.NaN);
        this.regularization = null;
        this.regularizationBias = null;
        this.setGradientNormalization(GradientNormalization.None);
        this.setGradientNormalizationThreshold(1.0);
        this.iUpdater = null;
        this.biasUpdater = null;
        this.weightPoolId = WeightPool.getNewId();
    }

    @Override
    public BaseLayer clone() {
        BaseLayer clone = (BaseLayer) super.clone();
        if (clone.iDropout != null) {
            clone.iDropout = clone.iDropout.clone();
        }
        if(regularization != null){
            //Regularization fields are _usually_ thread safe and immutable, but let's clone to be sure
            clone.regularization = new ArrayList<>(regularization.size());
            for(Regularization r : regularization){
                clone.regularization.add(r.clone());
            }
        }
        if(regularizationBias != null){
            clone.regularizationBias = new ArrayList<>(regularizationBias.size());
            for(Regularization r : regularizationBias){
                clone.regularizationBias.add(r.clone());
            }
        }
        return clone;
    }

    @Override
    public BaseLayer cloneAndShareWeights() {
        BaseLayer clone = clone();
        clone.weightPoolId = this.weightPoolId;
        return clone;
    }

    /**
     * Get the updater for the given parameter. Typically the same updater will be used for all updaters, but this is
     * not necessarily the case
     *
     * @param paramName Parameter name
     * @return IUpdater for the parameter
     */
    @Override
    public IUpdater getUpdaterByParam(String paramName) {
        if (biasUpdater != null && initializer().isBiasParam(this, paramName)) {
            return biasUpdater;
        }
        return iUpdater;
    }

    @Override
    public GradientNormalization getGradientNormalization() {
        return gradientNormalization;
    }

    @Override
    public List<Regularization> getRegularizationByParam(String paramName){
        if(initializer().isWeightParam(this, paramName)){
            return regularization;
        } else if(initializer().isBiasParam(this, paramName)){
            return regularizationBias;
        }
        return null;
    }


    @SuppressWarnings("unchecked")
    @Getter
    @Setter
    public abstract static class Builder<T extends Builder<T>> extends Layer.Builder<T> {

        /**
         * Set the activation function for the layer. This overload can be used for custom {@link IActivation}
         * instances
         *
         */
        protected IActivation activationFn = null;

        /**
         * Weight initialization scheme to use, for initial weight values
         *
         * @see IWeightInit
         */
        protected IWeightInit weightInitFn = null;

        /**
         * Bias initialization value, for layers with biases. Defaults to 0
         *
         */
        protected double biasInit = Double.NaN;

        /**
         * Gain initialization value, for layers with Layer Normalization. Defaults to 1
         *
         */
        protected double gainInit = Double.NaN;

        /**
         * Regularization for the parameters (excluding biases).
         */
        protected List<Regularization> regularization = new ArrayList<>();
        /**
         * Regularization for the bias parameters only
         */
        protected List<Regularization> regularizationBias = new ArrayList<>();

        /**
         * Gradient updater. For example, {@link org.nd4j.linalg.learning.config.Adam} or {@link
         * org.nd4j.linalg.learning.config.Nesterovs}
         *
         */
        protected IUpdater iupdater = null;

        /**
         * Gradient updater configuration, for the biases only. If not set, biases will use the updater as set by {@link
         * #updater(IUpdater)}
         *
         */
        protected IUpdater biasUpdater = null;

        /**
         * Gradient normalization strategy. Used to specify gradient renormalization, gradient clipping etc.
         *
         * @see GradientNormalization
         */
        protected GradientNormalization gradientNormalization = null;

        /**
         * Threshold for gradient normalization, only used for GradientNormalization.ClipL2PerLayer,
         * GradientNormalization.ClipL2PerParamType, and GradientNormalization.ClipElementWiseAbsoluteValue<br> Not used
         * otherwise.<br> L2 threshold for first two types of clipping, or absolute value threshold for last type of
         * clipping.
         */
        protected double gradientNormalizationThreshold = Double.NaN;

        /**
         * Set the weight noise (such as {@link org.deeplearning4j.nn.conf.weightnoise.DropConnect} and {@link
         * org.deeplearning4j.nn.conf.weightnoise.WeightNoise}) for this layer
         *
         */
        protected IWeightNoise weightNoise;

        /**
         * Set the activation function for the layer. This overload can be used for custom {@link IActivation}
         * instances
         *
         * @param activationFunction Activation function to use for the layer
         */
        public T activation(IActivation activationFunction) {
            this.setActivationFn(activationFunction);
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
         * @see IWeightInit
         */
        public T weightInit(IWeightInit weightInit) {
            this.setWeightInitFn(weightInit);
            return (T) this;
        }

        /**
         * Weight initialization scheme to use, for initial weight values
         *
         * @see WeightInit
         */
        public T weightInit(WeightInit weightInit) {
            if (weightInit == WeightInit.DISTRIBUTION) {
                throw new UnsupportedOperationException(
                                "Not supported!, Use weightInit(Distribution distribution) instead!");
            }

            this.setWeightInitFn(weightInit.getWeightInitFunction());
            return (T) this;
        }

        /**
         * Set weight initialization scheme to random sampling via the specified distribution. Equivalent to: {@code
         * .weightInit(new WeightInitDistribution(distribution))}
         *
         * @param distribution Distribution to use for weight initialization
         */
        public T weightInit(Distribution distribution) {
            return weightInit(new WeightInitDistribution(distribution));
        }

        /**
         * Bias initialization value, for layers with biases. Defaults to 0
         *
         * @param biasInit Value to use for initializing biases
         */
        public T biasInit(double biasInit) {
            this.setBiasInit(biasInit);
            return (T) this;
        }

        /**
         * Gain initialization value, for layers with Layer Normalization. Defaults to 1
         *
         * @param gainInit Value to use for initializing gain
         */
        public T gainInit(double gainInit) {
            this.gainInit = gainInit;
            return (T) this;
        }

        /**
         * Distribution to sample initial weights from. Equivalent to: {@code .weightInit(new
         * WeightInitDistribution(distribution))}
         */
        @Deprecated
        public T dist(Distribution dist) {
            return weightInit(dist);
        }

        /**
         * L1 regularization coefficient (weights only). Use {@link #l1Bias(double)} to configure the l1 regularization
         * coefficient for the bias.
         */
        public T l1(double l1) {
            //Check if existing L1 exists; if so, replace it
            NetworkUtils.removeInstances(this.regularization, L1Regularization.class);
            if(l1 > 0.0) {
                this.regularization.add(new L1Regularization(l1));
            }
            return (T) this;
        }

        /**
         * L2 regularization coefficient (weights only). Use {@link #l2Bias(double)} to configure the l2 regularization
         * coefficient for the bias.<br>
         * <b>Note</b>: Generally, {@link WeightDecay} (set via {@link #weightDecay(double,boolean)} should be preferred to
         * L2 regularization. See {@link WeightDecay} javadoc for further details.<br>
         */
        public T l2(double l2) {
            //Check if existing L2 exists; if so, replace it. Also remove weight decay - it doesn't make sense to use both
            NetworkUtils.removeInstances(this.regularization, L2Regularization.class);
            if(l2 > 0.0) {
                NetworkUtils.removeInstancesWithWarning(this.regularization, WeightDecay.class, "WeightDecay regularization removed: incompatible with added L2 regularization");
                this.regularization.add(new L2Regularization(l2));
            }
            return (T) this;
        }

        /**
         * L1 regularization coefficient for the bias. Default: 0. See also {@link #l1(double)}
         */
        public T l1Bias(double l1Bias) {
            NetworkUtils.removeInstances(this.regularizationBias, L1Regularization.class);
            if(l1Bias > 0.0) {
                this.regularizationBias.add(new L1Regularization(l1Bias));
            }
            return (T) this;
        }

        /**
         * L2 regularization coefficient for the bias. Default: 0. See also {@link #l2(double)}<br>
         * <b>Note</b>: Generally, {@link WeightDecay} (set via {@link #weightDecayBias(double,boolean)} should be preferred to
         * L2 regularization. See {@link WeightDecay} javadoc for further details.<br>
         */
        public T l2Bias(double l2Bias) {
            NetworkUtils.removeInstances(this.regularizationBias, L2Regularization.class);
            if(l2Bias > 0.0) {
                NetworkUtils.removeInstancesWithWarning(this.regularizationBias, WeightDecay.class, "WeightDecay regularization removed: incompatible with added L2 regularization");
                this.regularizationBias.add(new L2Regularization(l2Bias));
            }
            return (T) this;
        }

        /**
         * Add weight decay regularization for the network parameters (excluding biases).<br>
         * This applies weight decay <i>with</i> multiplying the learning rate - see {@link WeightDecay} for more details.<br>
         *
         * @param coefficient Weight decay regularization coefficient
         * @see #weightDecay(double, boolean)
         */
        public Builder weightDecay(double coefficient) {
            return weightDecay(coefficient, true);
        }

        /**
         * Add weight decay regularization for the network parameters (excluding biases). See {@link WeightDecay} for more details.<br>
         *
         * @param coefficient Weight decay regularization coefficient
         * @param applyLR     Whether the learning rate should be multiplied in when performing weight decay updates. See {@link WeightDecay} for more details.
         * @see #weightDecay(double, boolean)
         */
        public Builder weightDecay(double coefficient, boolean applyLR) {
            //Check if existing weight decay if it exists; if so, replace it. Also remove L2 - it doesn't make sense to use both
            NetworkUtils.removeInstances(this.regularization, WeightDecay.class);
            if(coefficient > 0.0) {
                NetworkUtils.removeInstancesWithWarning(this.regularization, L2Regularization.class, "L2 regularization removed: incompatible with added WeightDecay regularization");
                this.regularization.add(new WeightDecay(coefficient, applyLR));
            }
            return this;
        }

        /**
         * Weight decay for the biases only - see {@link #weightDecay(double)} for more details.
         * This applies weight decay <i>with</i> multiplying the learning rate.<br>
         *
         * @param coefficient Weight decay regularization coefficient
         * @see #weightDecayBias(double, boolean)
         */
        public Builder weightDecayBias(double coefficient) {
            return weightDecayBias(coefficient, true);
        }

        /**
         * Weight decay for the biases only - see {@link #weightDecay(double)} for more details<br>
         *
         * @param coefficient Weight decay regularization coefficient
         */
        public Builder weightDecayBias(double coefficient, boolean applyLR) {
            //Check if existing weight decay if it exists; if so, replace it. Also remove L2 - it doesn't make sense to use both
            NetworkUtils.removeInstances(this.regularizationBias, WeightDecay.class);
            if(coefficient > 0.0) {
                NetworkUtils.removeInstancesWithWarning(this.regularizationBias, L2Regularization.class, "L2 regularization removed: incompatible with added WeightDecay regularization");
                this.regularizationBias.add(new WeightDecay(coefficient, applyLR));
            }
            return this;
        }

        /**
         * Set the regularization for the parameters (excluding biases) - for example {@link WeightDecay}<br>
         *
         * @param regularization Regularization to apply for the network parameters/weights (excluding biases)
         */
        public Builder regularization(List<Regularization> regularization) {
            this.setRegularization(regularization);
            return this;
        }

        /**
         * Set the regularization for the biases only - for example {@link WeightDecay}<br>
         *
         * @param regularizationBias Regularization to apply for the network biases only
         */
        public Builder regularizationBias(List<Regularization> regularizationBias) {
            this.setRegularizationBias(regularizationBias);
            return this;
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
         * Gradient updater. For example, {@link org.nd4j.linalg.learning.config.Adam} or {@link
         * org.nd4j.linalg.learning.config.Nesterovs}
         *
         * @param updater Updater to use
         */
        public T updater(IUpdater updater) {
            this.setIupdater(updater);
            return (T) this;
        }

        /**
         * Gradient updater configuration, for the biases only. If not set, biases will use the updater as set by {@link
         * #updater(IUpdater)}
         *
         * @param biasUpdater Updater to use for bias parameters
         */
        public T biasUpdater(IUpdater biasUpdater) {
            this.setBiasUpdater(biasUpdater);
            return (T) this;
        }

        /**
         * Gradient normalization strategy. Used to specify gradient renormalization, gradient clipping etc.
         *
         * @param gradientNormalization Type of normalization to use. Defaults to None.
         * @see GradientNormalization
         */
        public T gradientNormalization(GradientNormalization gradientNormalization) {
            this.setGradientNormalization(gradientNormalization);
            return (T) this;
        }

        /**
         * Threshold for gradient normalization, only used for GradientNormalization.ClipL2PerLayer,
         * GradientNormalization.ClipL2PerParamType, and GradientNormalization.ClipElementWiseAbsoluteValue<br> Not used
         * otherwise.<br> L2 threshold for first two types of clipping, or absolute value threshold for last type of
         * clipping.
         */
        public T gradientNormalizationThreshold(double threshold) {
            this.setGradientNormalizationThreshold(threshold);
            return (T) this;
        }

        /**
         * Set the weight noise (such as {@link org.deeplearning4j.nn.conf.weightnoise.DropConnect} and {@link
         * org.deeplearning4j.nn.conf.weightnoise.WeightNoise}) for this layer
         *
         * @param weightNoise Weight noise instance to use
         */
        public T weightNoise(IWeightNoise weightNoise) {
            this.setWeightNoise(weightNoise);
            return (T) this;
        }
        
        
    }
}

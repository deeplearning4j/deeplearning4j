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
import lombok.NoArgsConstructor;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.distribution.Distribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.shade.jackson.annotation.JsonProperty;
import org.nd4j.shade.jackson.annotation.JsonSubTypes;
import org.nd4j.shade.jackson.annotation.JsonTypeInfo;
import org.nd4j.shade.jackson.annotation.JsonTypeInfo.As;
import org.nd4j.shade.jackson.annotation.JsonTypeInfo.Id;

import java.io.Serializable;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;

/**
 * A neural network layer.
 */
@JsonTypeInfo(use = Id.NAME, include = As.WRAPPER_OBJECT)
@JsonSubTypes(value = {
        @JsonSubTypes.Type(value = AutoEncoder.class, name = "autoEncoder"),
        @JsonSubTypes.Type(value = ConvolutionLayer.class, name = "convolution"),
        @JsonSubTypes.Type(value = Convolution1DLayer.class, name = "convolution1d"),
        @JsonSubTypes.Type(value = GravesLSTM.class, name = "gravesLSTM"),
        @JsonSubTypes.Type(value = GravesBidirectionalLSTM.class, name = "gravesBidirectionalLSTM"),
        @JsonSubTypes.Type(value = OutputLayer.class, name = "output"),
        @JsonSubTypes.Type(value = RnnOutputLayer.class, name = "rnnoutput"),
        @JsonSubTypes.Type(value = LossLayer.class, name = "loss"),
        @JsonSubTypes.Type(value = RBM.class, name = "RBM"),
        @JsonSubTypes.Type(value = DenseLayer.class, name = "dense"),
        @JsonSubTypes.Type(value = SubsamplingLayer.class, name = "subsampling"),
        @JsonSubTypes.Type(value = Subsampling1DLayer.class, name = "subsampling1d"),
        @JsonSubTypes.Type(value = BatchNormalization.class, name = "batchNormalization"),
        @JsonSubTypes.Type(value = LocalResponseNormalization.class, name = "localResponseNormalization"),
        @JsonSubTypes.Type(value = EmbeddingLayer.class, name = "embedding"),
        @JsonSubTypes.Type(value = ActivationLayer.class, name = "activation"),
        @JsonSubTypes.Type(value = VariationalAutoencoder.class, name = "VariationalAutoencoder"),
        @JsonSubTypes.Type(value = DropoutLayer.class, name = "dropout"),
        @JsonSubTypes.Type(value = GlobalPoolingLayer.class, name = "GlobalPooling"),
        @JsonSubTypes.Type(value = ZeroPaddingLayer.class, name = "zeroPadding")
})
@Data
@NoArgsConstructor
public abstract class Layer implements Serializable, Cloneable {
    protected String layerName;
    protected IActivation activationFn;
    protected WeightInit weightInit;
    protected double biasInit;
    protected Distribution dist;
    protected double learningRate;
    protected double biasLearningRate;
    //learning rate after n iterations
    protected Map<Integer, Double> learningRateSchedule;
    protected double momentum;
    //momentum after n iterations
    protected Map<Integer, Double> momentumSchedule;
    protected double l1;
    protected double l2;
    protected double l1Bias;
    protected double l2Bias;
    protected double dropOut;
    protected Updater updater;
    //adadelta - weight for how much to consider previous history
    protected double rho;
    //Epsilon value for adagrad and adadelta
    protected double epsilon;
    protected double rmsDecay;
    protected double adamMeanDecay;
    protected double adamVarDecay;
    protected GradientNormalization gradientNormalization = GradientNormalization.None; //Clipping, rescale based on l2 norm, etc
    protected double gradientNormalizationThreshold = 1.0; //Threshold for l2 and element-wise gradient clipping


    public Layer(Builder builder) {
        this.layerName = builder.layerName;
        this.activationFn = builder.activationFn;
        this.weightInit = builder.weightInit;
        this.biasInit = builder.biasInit;
        this.dist = builder.dist;
        this.learningRate = builder.learningRate;
        this.biasLearningRate = builder.biasLearningRate;
        this.learningRateSchedule = builder.learningRateSchedule;
        this.momentum = builder.momentum;
        this.momentumSchedule = builder.momentumAfter;
        this.l1 = builder.l1;
        this.l2 = builder.l2;
        this.l1Bias = builder.l1Bias;
        this.l2Bias = builder.l2Bias;
        this.dropOut = builder.dropOut;
        this.updater = builder.updater;
        this.rho = builder.rho;
        this.epsilon = builder.epsilon;
        this.rmsDecay = builder.rmsDecay;
        this.adamMeanDecay = builder.adamMeanDecay;
        this.adamVarDecay = builder.adamVarDecay;
        this.gradientNormalization = builder.gradientNormalization;
        this.gradientNormalizationThreshold = builder.gradientNormalizationThreshold;
    }

    /**
     * Reset the learning related configs of the layer to default. When instantiated with a global neural network configuration
     * the parameters specified in the neural network configuration will be used.
     * For internal use with the transfer learning API. Users should not have to call this method directly.
     */
    public void resetLayerDefaultConfig() {
        //clear the learning related params for all layers in the origConf and set to defaults
        this.setUpdater(null);
        this.setMomentum(Double.NaN);
        this.setWeightInit(null);
        this.setBiasInit(Double.NaN);
        this.setDist(null);
        this.setLearningRate(Double.NaN);
        this.setBiasLearningRate(Double.NaN);
        this.setLearningRateSchedule(null);
        this.setMomentumSchedule(null);
        this.setL1(Double.NaN);
        this.setL2(Double.NaN);
        this.setDropOut(Double.NaN);
        this.setRho(Double.NaN);
        this.setEpsilon(Double.NaN);
        this.setRmsDecay(Double.NaN);
        this.setAdamMeanDecay(Double.NaN);
        this.setAdamVarDecay(Double.NaN);
        this.setGradientNormalization(GradientNormalization.None);
        this.setGradientNormalizationThreshold(1.0);
    }

    @Override
    public Layer clone() {
        try {
            Layer clone = (Layer) super.clone();
            if (clone.dist != null)
                clone.dist = clone.dist.clone();
            if (clone.learningRateSchedule != null)
                clone.learningRateSchedule = new HashMap<>(clone.learningRateSchedule);
            if (clone.momentumSchedule != null)
                clone.momentumSchedule = new HashMap<>(clone.momentumSchedule);
            return clone;
        } catch (CloneNotSupportedException e) {
            throw new RuntimeException(e);
        }
    }

    public abstract org.deeplearning4j.nn.api.Layer instantiate(NeuralNetConfiguration conf,
                    Collection<IterationListener> iterationListeners, int layerIndex, INDArray layerParamsView,
                    boolean initializeParams);

    public abstract ParamInitializer initializer();

    /**
     * For a given type of input to this layer, what is the type of the output?
     *
     * @param layerIndex Index of the layer
     * @param inputType Type of input for the layer
     * @return Type of output from the layer
     * @throws IllegalStateException if input type is invalid for this layer
     */
    public abstract InputType getOutputType(int layerIndex, InputType inputType);

    /**
     * Set the nIn value (number of inputs, or input depth for CNNs) based on the given input type
     *
     * @param inputType Input type for this layer
     * @param override  If false: only set the nIn value if it's not already set. If true: set it regardless of whether it's
     *                  already set or not.
     * @throws IllegalStateException if input type is invalid for this layer
     */
    public abstract void setNIn(InputType inputType, boolean override);


    /**
     * For the given type of input to this layer, what preprocessor (if any) is required?<br>
     * Returns null if no preprocessor is required, otherwise returns an appropriate {@link InputPreProcessor}
     * for this layer, such as a {@link org.deeplearning4j.nn.conf.preprocessor.CnnToFeedForwardPreProcessor}
     *
     * @param inputType InputType to this layer
     * @return Null if no preprocessor is required, otherwise the type of preprocessor necessary for this layer/input combination
     * @throws IllegalStateException if input type is invalid for this layer
     */
    public abstract InputPreProcessor getPreProcessorForInputType(InputType inputType);

    /**
     * Get the L1 coefficient for the given parameter.
     * Different parameters may have different L1 values, even for a single .l1(x) configuration.
     * For example, biases generally aren't L1 regularized, even if weights are
     *
     * @param paramName    Parameter name
     * @return L1 value for that parameter
     */
    public abstract double getL1ByParam(String paramName);

    /**
     * Get the L2 coefficient for the given parameter.
     * Different parameters may have different L2 values, even for a single .l2(x) configuration.
     * For example, biases generally aren't L1 regularized, even if weights are
     *
     * @param paramName    Parameter name
     * @return L2 value for that parameter
     */
    public abstract double getL2ByParam(String paramName);

    /**
     * Get the (initial) learning rate coefficient for the given parameter.
     * Different parameters may be configured to have different learning rates, though commonly all parameters will
     * have the same learning rate
     *
     * @param paramName    Parameter name
     * @return Initial learning rate value for that parameter
     */
    public abstract double getLearningRateByParam(String paramName);

    /**
     * Get the updater for the given parameter. Typically the same updater will be used for all updaters, but this
     * is not necessarily the case
     *
     * @param paramName    Parameter name
     * @return             Updater for the parameter
     */
    public Updater getUpdaterByParam(String paramName) {
        return updater;
    }

    @SuppressWarnings("unchecked")
    public abstract static class Builder<T extends Builder<T>> {
        protected String layerName = null;
        protected IActivation activationFn = null;
        protected WeightInit weightInit = null;
        protected double biasInit = Double.NaN;
        protected Distribution dist = null;
        protected double learningRate = Double.NaN;
        protected double biasLearningRate = Double.NaN;
        protected Map<Integer, Double> learningRateSchedule = null;
        protected double momentum = Double.NaN;
        protected Map<Integer, Double> momentumAfter = null;
        protected double l1 = Double.NaN;
        protected double l2 = Double.NaN;
        protected double l1Bias = Double.NaN;
        protected double l2Bias = Double.NaN;
        protected double dropOut = Double.NaN;
        protected Updater updater = null;
        protected double rho = Double.NaN;
        protected double epsilon = Double.NaN;
        protected double rmsDecay = Double.NaN;
        protected double adamMeanDecay = Double.NaN;
        protected double adamVarDecay = Double.NaN;
        protected GradientNormalization gradientNormalization = null;
        protected double gradientNormalizationThreshold = Double.NaN;
        protected LearningRatePolicy learningRatePolicy = null;


        /**
         * Layer name assigns layer string name.
         * Allows easier differentiation between layers.
         */
        public T name(String layerName) {
            this.layerName = layerName;
            return (T) this;
        }


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

        public T activation(IActivation activationFunction) {
            this.activationFn = activationFunction;
            return (T) this;
        }

        public T activation(Activation activation) {
            return activation(activation.getActivationFunction());
        }

        /**
         * Weight initialization scheme.
         *
         * @see org.deeplearning4j.nn.weights.WeightInit
         */
        public T weightInit(WeightInit weightInit) {
            this.weightInit = weightInit;
            return (T) this;
        }

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
        public T learningRate(double learningRate) {
            this.learningRate = learningRate;
            return (T) this;
        }

        /**
         * Bias learning rate. Set this to apply a different learning rate to the bias
         */
        public T biasLearningRate(double biasLearningRate) {
            this.biasLearningRate = biasLearningRate;
            return (T) this;
        }

        /**
         * Learning rate schedule. Map of the iteration to the learning rate to apply at that iteration.
         */
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
         * Dropout. Value is probability of retaining an activation - thus 1.0 is equivalent to no dropout.
         * Note that 0.0 (the default) disables dropout.
         */
        public T dropOut(double dropOut) {
            this.dropOut = dropOut;
            return (T) this;
        }

        /**
         * Momentum rate.
         */
        public T momentum(double momentum) {
            this.momentum = momentum;
            return (T) this;
        }

        /**
         * Momentum schedule. Map of the iteration to the momentum rate to apply at that iteration.
         */
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
        public T updater(Updater updater) {
            this.updater = updater;
            return (T) this;
        }

        /**
         * Ada delta coefficient, rho. Only applies if using .updater(Updater.ADADELTA)
         *
         * @param rho
         */
        public T rho(double rho) {
            this.rho = rho;
            return (T) this;
        }

        /**
         * Decay rate for RMSProp. Only applies if using .updater(Updater.RMSPROP)
         */
        public T rmsDecay(double rmsDecay) {
            this.rmsDecay = rmsDecay;
            return (T) this;
        }

        /**
         * Epsilon value for updaters: Adagrad and Adadelta. Only used if using Updater.ADAGRAD or Updater.ADADELTA
         *
         * @param epsilon    Epsilon value to use for adagrad and adadelta
         */
        public T epsilon(double epsilon) {
            this.epsilon = epsilon;
            return (T) this;
        }

        /**
         * Mean decay rate for Adam updater. Only applies if using .updater(Updater.ADAM)
         */
        public T adamMeanDecay(double adamMeanDecay) {
            this.adamMeanDecay = adamMeanDecay;
            return (T) this;
        }

        /**
         * Variance decay rate for Adam updater. Only applies if using .updater(Updater.ADAM)
         */
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
        public T learningRateDecayPolicy(LearningRatePolicy policy) {
            this.learningRatePolicy = policy;
            return (T) this;
        }


        public abstract <E extends Layer> E build();
    }
}

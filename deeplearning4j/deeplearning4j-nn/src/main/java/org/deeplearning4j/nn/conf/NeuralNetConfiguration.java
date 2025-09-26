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

package org.deeplearning4j.nn.conf;

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.api.layers.LayerConstraint;
import org.deeplearning4j.nn.conf.distribution.Distribution;
import org.deeplearning4j.nn.conf.dropout.Dropout;
import org.deeplearning4j.nn.conf.dropout.IDropout;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.layers.misc.FrozenLayer;
import org.deeplearning4j.nn.conf.layers.misc.FrozenLayerWithBackprop;
import org.deeplearning4j.nn.conf.layers.recurrent.Bidirectional;
import org.deeplearning4j.nn.conf.layers.samediff.AbstractSameDiffLayer;
import org.deeplearning4j.nn.conf.layers.wrapper.BaseWrapperLayer;
import org.deeplearning4j.nn.conf.serde.JsonMappers;
import org.deeplearning4j.nn.conf.stepfunctions.StepFunction;
import org.deeplearning4j.nn.conf.weightnoise.IWeightNoise;
import org.deeplearning4j.nn.weights.IWeightInit;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.nn.weights.WeightInitDistribution;
import org.deeplearning4j.nn.weights.WeightInitXavier;
import org.deeplearning4j.util.NetworkUtils;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.ActivationSigmoid;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.learning.regularization.L1Regularization;
import org.nd4j.linalg.learning.regularization.L2Regularization;
import org.nd4j.linalg.learning.regularization.Regularization;
import org.nd4j.linalg.learning.regularization.WeightDecay;
import org.nd4j.shade.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.io.Serializable;
import java.util.*;


@Data
@NoArgsConstructor
@Slf4j
@EqualsAndHashCode(exclude = {"iterationCount", "epochCount"})
public class NeuralNetConfiguration implements Serializable, Cloneable {

    protected Layer layer;
    //batch size: primarily used for conv nets. Will be reinforced if set.
    protected boolean miniBatch = true;
    //number of line search iterations
    protected int maxNumLineSearchIterations;
    protected long seed;
    protected OptimizationAlgorithm optimizationAlgo;
    //gradient keys used for ensuring order when getting and setting the gradient
    protected List<String> variables = new ArrayList<>();
    //whether to constrain the gradient to unit norm or not
    protected StepFunction stepFunction;
    //minimize or maximize objective
    protected boolean minimize = true;

    // this field defines preOutput cache
    protected CacheMode cacheMode;

    protected DataType dataType = DataType.FLOAT;   //Default to float for deserialization of legacy format nets

    //Counter for the number of parameter updates so far for this layer.
    //Note that this is only used for pretrain layers (AE, VAE) - MultiLayerConfiguration and ComputationGraphConfiguration
    //contain counters for standard backprop training.
    // This is important for learning rate schedules, for example, and is stored here to ensure it is persisted
    // for Spark and model serialization
    protected int iterationCount = 0;

    //Counter for the number of epochs completed so far. Used for per-epoch schedules
    protected int epochCount = 0;


    /**
     * Creates and returns a deep copy of the configuration.
     */
    @Override
    public NeuralNetConfiguration clone() {
        try {
            NeuralNetConfiguration clone = (NeuralNetConfiguration) super.clone();
            if (clone.layer != null)
                clone.layer = clone.layer.clone();
            if (clone.stepFunction != null)
                clone.stepFunction = clone.stepFunction.clone();
            if (clone.variables != null)
                clone.variables = new ArrayList<>(clone.variables);
            return clone;
        } catch (CloneNotSupportedException e) {
            throw new RuntimeException(e);
        }
    }

    public List<String> variables() {
        return new ArrayList<>(variables);
    }

    public List<String> variables(boolean copy) {
        if (copy)
            return variables();
        return variables;
    }

    public void addVariable(String variable) {
        if (!variables.contains(variable)) {
            variables.add(variable);
        }
    }

    public void clearVariables() {
        variables.clear();
    }

    /**
     * Return this configuration as json
     *
     * @return this configuration represented as json
     */
    public String toYaml() {
        ObjectMapper mapper = mapperYaml();

        try {
            String ret = mapper.writeValueAsString(this);
            return ret;

        } catch (org.nd4j.shade.jackson.core.JsonProcessingException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Create a neural net configuration from json
     *
     * @param json the neural net configuration from json
     * @return
     */
    public static NeuralNetConfiguration fromYaml(String json) {
        ObjectMapper mapper = mapperYaml();
        try {
            NeuralNetConfiguration ret = mapper.readValue(json, NeuralNetConfiguration.class);
            return ret;
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Return this configuration as json
     *
     * @return this configuration represented as json
     */
    public String toJson() {
        ObjectMapper mapper = mapper();

        try {
            return mapper.writeValueAsString(this);
        } catch (org.nd4j.shade.jackson.core.JsonProcessingException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Create a neural net configuration from json
     *
     * @param json the neural net configuration from json
     * @return
     */
    public static NeuralNetConfiguration fromJson(String json) {
        ObjectMapper mapper = mapper();
        try {
            NeuralNetConfiguration ret = mapper.readValue(json, NeuralNetConfiguration.class);
            return ret;
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Object mapper for serialization of configurations
     *
     * @return
     */
    public static ObjectMapper mapperYaml() {
        return JsonMappers.getMapperYaml();
    }

    /**
     * Object mapper for serialization of configurations
     *
     * @return
     */
    public static ObjectMapper mapper() {
        return JsonMappers.getMapper();
    }

    /**
     * NeuralNetConfiguration builder, used as a starting point for creating a MultiLayerConfiguration or
     * ComputationGraphConfiguration.<br>
     * Note that values set here on the layer will be applied to all relevant layers - unless the value is overridden
     * on a layer's configuration
     */
    @Data
    public static class Builder implements Cloneable {
        protected IActivation activationFn = new ActivationSigmoid();
        protected IWeightInit weightInitFn = new WeightInitXavier();
        protected double biasInit = 0.0;
        protected double gainInit = 1.0;
        protected List<Regularization> regularization = new ArrayList<>();
        protected List<Regularization> regularizationBias = new ArrayList<>();
        protected IDropout idropOut;
        protected IWeightNoise weightNoise;
        protected IUpdater iUpdater = new Sgd();
        protected IUpdater biasUpdater = null;
        protected Layer layer;
        protected boolean miniBatch = true;
        protected int maxNumLineSearchIterations = 5;
        protected long seed = System.currentTimeMillis();
        protected OptimizationAlgorithm optimizationAlgo = OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT;
        protected StepFunction stepFunction = null;
        protected boolean minimize = true;
        protected GradientNormalization gradientNormalization = GradientNormalization.None;
        protected double gradientNormalizationThreshold = 1.0;
        protected List<LayerConstraint> allParamConstraints;
        protected List<LayerConstraint> weightConstraints;
        protected List<LayerConstraint> biasConstraints;

        protected WorkspaceMode trainingWorkspaceMode = WorkspaceMode.ENABLED;
        protected WorkspaceMode inferenceWorkspaceMode = WorkspaceMode.ENABLED;
        protected boolean setTWM = false;
        protected boolean setIWM = false;
        protected CacheMode cacheMode = CacheMode.NONE;
        protected DataType dataType = DataType.FLOAT;

        protected ConvolutionMode convolutionMode = ConvolutionMode.Truncate;
        protected ConvolutionLayer.AlgoMode cudnnAlgoMode = ConvolutionLayer.AlgoMode.PREFER_FASTEST;

        public Builder() {
            //
        }

        public Builder(NeuralNetConfiguration newConf) {
            if (newConf != null) {
                minimize = newConf.minimize;
                maxNumLineSearchIterations = newConf.maxNumLineSearchIterations;
                layer = newConf.layer;
                optimizationAlgo = newConf.optimizationAlgo;
                seed = newConf.seed;
                stepFunction = newConf.stepFunction;
                miniBatch = newConf.miniBatch;
            }
        }

        /**
         * Process input as minibatch vs full dataset.
         * Default set to true.
         */
        public Builder miniBatch(boolean miniBatch) {
            this.miniBatch = miniBatch;
            return this;
        }

        /**
         * This method defines Workspace mode being used during training:<br>
         * NONE: workspace won't be used<br>
         * ENABLED: workspaces will be used for training (reduced memory and better performance)
         *
         * @param workspaceMode Workspace mode for training
         * @return Builder
         */
        public Builder trainingWorkspaceMode(@NonNull WorkspaceMode workspaceMode) {
            this.trainingWorkspaceMode = workspaceMode;
            this.setTWM = true;
            return this;
        }

        /**
         * This method defines Workspace mode being used during inference:<br>
         * NONE: workspace won't be used<br>
         * ENABLED: workspaces will be used for inference (reduced memory and better performance)
         *
         * @param workspaceMode Workspace mode for inference
         * @return Builder
         */
        public Builder inferenceWorkspaceMode(@NonNull WorkspaceMode workspaceMode) {
            this.inferenceWorkspaceMode = workspaceMode;
            this.setIWM = true;
            return this;
        }

        /**
         * This method defines how/if preOutput cache is handled:
         * NONE: cache disabled (default value)
         * HOST: Host memory will be used
         * DEVICE: GPU memory will be used (on CPU backends effect will be the same as for HOST)
         *
         * @param cacheMode Cache mode to use
         * @return Builder
         */
        public Builder cacheMode(@NonNull CacheMode cacheMode) {
            this.cacheMode = cacheMode;
            return this;
        }

        /**
         * Objective function to minimize or maximize cost function
         * Default set to minimize true.
         */
        public Builder minimize(boolean minimize) {
            this.minimize = minimize;
            return this;
        }

        /**
         * Maximum number of line search iterations.
         * Only applies for line search optimizers: Line Search SGD, Conjugate Gradient, LBFGS
         * is NOT applicable for standard SGD
         *
         * @param maxNumLineSearchIterations > 0
         * @return
         */
        public Builder maxNumLineSearchIterations(int maxNumLineSearchIterations) {
            this.maxNumLineSearchIterations = maxNumLineSearchIterations;
            return this;
        }


        /**
         * Layer class.
         */
        public Builder layer(Layer layer) {
            this.layer = layer;
            return this;
        }

        /**
         * Step function to apply for back track line search.
         * Only applies for line search optimizers: Line Search SGD, Conjugate Gradient, LBFGS
         * Options: DefaultStepFunction (default), NegativeDefaultStepFunction
         * GradientStepFunction (for SGD), NegativeGradientStepFunction
         */
        @Deprecated
        public Builder stepFunction(StepFunction stepFunction) {
            this.stepFunction = stepFunction;
            return this;
        }

        /**
         * Create a ListBuilder (for creating a MultiLayerConfiguration)<br>
         * Usage:<br>
         * <pre>
         * {@code .list()
         * .layer(new DenseLayer.Builder()...build())
         * ...
         * .layer(new OutputLayer.Builder()...build())
         * }
         * </pre>
         */
        public ListBuilder list() {
            return new ListBuilder(this);
        }

        /**
         * Create a ListBuilder (for creating a MultiLayerConfiguration) with the specified layers<br>
         * Usage:<br>
         * <pre>
         * {@code .list(
         *      new DenseLayer.Builder()...build(),
         *      ...,
         *      new OutputLayer.Builder()...build())
         * }
         * </pre>
         *
         * @param layers The layer configurations for the network
         */
        public ListBuilder list(Layer... layers) {
            if (layers == null || layers.length == 0)
                throw new IllegalArgumentException("Cannot create network with no layers");
            Map<Integer, Builder> layerMap = new HashMap<>();
            for (int i = 0; i < layers.length; i++) {
                Builder b = this.clone();
                b.layer(layers[i]);
                layerMap.put(i, b);
            }
            return new ListBuilder(this, layerMap);

        }

        /**
         * Create a GraphBuilder (for creating a ComputationGraphConfiguration).
         */
        public ComputationGraphConfiguration.GraphBuilder graphBuilder() {
            return new ComputationGraphConfiguration.GraphBuilder(this);
        }

        /**
         * Random number generator seed. Used for reproducability between runs
         */
        public Builder seed(long seed) {
            this.seed = seed;
            Nd4j.getRandom().setSeed(seed);
            return this;
        }

        /**
         * Optimization algorithm to use. Most common: OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT
         *
         * @param optimizationAlgo Optimization algorithm to use when training
         */
        public Builder optimizationAlgo(OptimizationAlgorithm optimizationAlgo) {
            this.optimizationAlgo = optimizationAlgo;
            return this;
        }

        @Override
        public Builder clone() {
            try {
                Builder clone = (Builder) super.clone();
                if (clone.layer != null)
                    clone.layer = clone.layer.clone();
                if (clone.stepFunction != null)
                    clone.stepFunction = clone.stepFunction.clone();

                return clone;

            } catch (CloneNotSupportedException e) {
                throw new RuntimeException(e);
            }
        }

        /**
         * Activation function / neuron non-linearity<br>
         * Note: values set by this method will be applied to all applicable layers in the network, unless a different
         * value is explicitly set on a given layer. In other words: values set via this method are used as the default
         * value, and can be overridden on a per-layer basis.
         *
         * @see #activation(Activation)
         */
        public Builder activation(IActivation activationFunction) {
            this.activationFn = activationFunction;
            return this;
        }

        /**
         * Activation function / neuron non-linearity<br>
         * Note: values set by this method will be applied to all applicable layers in the network, unless a different
         * value is explicitly set on a given layer. In other words: values set via this method are used as the default
         * value, and can be overridden on a per-layer basis.
         */
        public Builder activation(Activation activation) {
            return activation(activation.getActivationFunction());
        }


        /**
         * Weight initialization scheme to use, for initial weight values
         * Note: values set by this method will be applied to all applicable layers in the network, unless a different
         * value is explicitly set on a given layer. In other words: values set via this method are used as the default
         * value, and can be overridden on a per-layer basis.
         *
         * @see IWeightInit
         */
        public Builder weightInit(IWeightInit weightInit) {
            this.weightInitFn = weightInit;
            return this;
        }

        /**
         * Weight initialization scheme to use, for initial weight values
         * Note: values set by this method will be applied to all applicable layers in the network, unless a different
         * value is explicitly set on a given layer. In other words: values set via this method are used as the default
         * value, and can be overridden on a per-layer basis.
         *
         * @see WeightInit
         */
        public Builder weightInit(WeightInit weightInit) {

            this.weightInitFn = weightInit.getWeightInitFunction();
            return this;
        }

        /**
         * Set weight initialization scheme to random sampling via the specified distribution.
         * Equivalent to: {@code .weightInit(new WeightInitDistribution(distribution))}
         * Note: values set by this method will be applied to all applicable layers in the network, unless a different
         * value is explicitly set on a given layer. In other words: values set via this method are used as the default
         * value, and can be overridden on a per-layer basis.
         *
         * @param distribution Distribution to use for weight initialization
         */
        public Builder weightInit(Distribution distribution){
            return weightInit(new WeightInitDistribution(distribution));
        }

        /**
         * Constant for bias initialization. Default: 0.0<br>
         * Note: values set by this method will be applied to all applicable layers in the network, unless a different
         * value is explicitly set on a given layer. In other words: values set via this method are used as the default
         * value, and can be overridden on a per-layer basis.
         *
         * @param biasInit Constant for bias initialization
         */
        public Builder biasInit(double biasInit) {
            this.biasInit = biasInit;
            return this;
        }

        /**
         * Distribution to sample initial weights from.
         * Equivalent to: {@code .weightInit(new WeightInitDistribution(distribution))}.<br>
         * Note: values set by this method will be applied to all applicable layers in the network, unless a different
         * value is explicitly set on a given layer. In other words: values set via this method are used as the default
         * value, and can be overridden on a per-layer basis.
         *
         * @see #weightInit(Distribution)
         * @deprecated Use {@link #weightInit(Distribution)}
         */
        @Deprecated
        public Builder dist(Distribution dist) {
            return weightInit(dist);
        }

        /**
         * L1 regularization coefficient for the weights (excluding biases).<br>
         * Note: values set by this method will be applied to all applicable layers in the network, unless a different
         * value is explicitly set on a given layer. In other words: values set via this method are used as the default
         * value, and can be overridden on a per-layer basis.
         */
        public Builder l1(double l1) {
            //Check if existing L1 exists; if so, replace it
            NetworkUtils.removeInstances(this.regularization, L1Regularization.class);
            if(l1 > 0.0) {
                this.regularization.add(new L1Regularization(l1));
            }
            return this;
        }

        /**
         * L2 regularization coefficient for the weights (excluding biases).<br>
         * <b>Note</b>: Generally, {@link WeightDecay} (set via {@link #weightDecay(double)} should be preferred to
         * L2 regularization. See {@link WeightDecay} javadoc for further details.<br>
         * Note: values set by this method will be applied to all applicable layers in the network, unless a different
         * value is explicitly set on a given layer. In other words: values set via this method are used as the default
         * value, and can be overridden on a per-layer basis.<br>
         * Note: L2 regularization and weight decay usually should not be used together; if any weight decay (or L2) has
         * been added for the biases, these will be removed first.
         *
         * @see #weightDecay(double, boolean)
         */
        public Builder l2(double l2) {
            //Check if existing L2 exists; if so, replace it. Also remove weight decay - it doesn't make sense to use both
            NetworkUtils.removeInstances(this.regularization, L2Regularization.class);
            if(l2 > 0.0) {
                NetworkUtils.removeInstancesWithWarning(this.regularization, WeightDecay.class, "WeightDecay regularization removed: incompatible with added L2 regularization");
                this.regularization.add(new L2Regularization(l2));
            }
            return this;
        }

        /**
         * L1 regularization coefficient for the bias.<br>
         * Note: values set by this method will be applied to all applicable layers in the network, unless a different
         * value is explicitly set on a given layer. In other words: values set via this method are used as the default
         * value, and can be overridden on a per-layer basis.
         */
        public Builder l1Bias(double l1Bias) {
            NetworkUtils.removeInstances(this.regularizationBias, L1Regularization.class);
            if(l1Bias > 0.0) {
                this.regularizationBias.add(new L1Regularization(l1Bias));
            }
            return this;
        }

        /**
         * L2 regularization coefficient for the bias.<br>
         * <b>Note</b>: Generally, {@link WeightDecay} (set via {@link #weightDecayBias(double,boolean)} should be preferred to
         * L2 regularization. See {@link WeightDecay} javadoc for further details.<br>
         * Note: values set by this method will be applied to all applicable layers in the network, unless a different
         * value is explicitly set on a given layer. In other words: values set via this method are used as the default
         * value, and can be overridden on a per-layer basis.<br>
         * Note: L2 regularization and weight decay usually should not be used together; if any weight decay (or L2) has
         * been added for the biases, these will be removed first.
         *
         * @see #weightDecayBias(double, boolean)
         */
        public Builder l2Bias(double l2Bias) {
            NetworkUtils.removeInstances(this.regularizationBias, L2Regularization.class);
            if(l2Bias > 0.0) {
                NetworkUtils.removeInstancesWithWarning(this.regularizationBias, WeightDecay.class, "L2 bias regularization removed: incompatible with added WeightDecay regularization");
                this.regularizationBias.add(new L2Regularization(l2Bias));
            }
            return this;
        }

        /**
         * Add weight decay regularization for the network parameters (excluding biases).<br>
         * This applies weight decay <i>with</i> multiplying the learning rate - see {@link WeightDecay} for more details.<br>
         * Note: values set by this method will be applied to all applicable layers in the network, unless a different
         * value is explicitly set on a given layer. In other words: values set via this method are used as the default
         * value, and can be overridden on a per-layer basis.<br>
         *
         * @param coefficient Weight decay regularization coefficient
         * @see #weightDecay(double, boolean)
         */
        public Builder weightDecay(double coefficient) {
            return weightDecay(coefficient, true);
        }

        /**
         * Add weight decay regularization for the network parameters (excluding biases). See {@link WeightDecay} for more details.<br>
         * Note: values set by this method will be applied to all applicable layers in the network, unless a different
         * value is explicitly set on a given layer. In other words: values set via this method are used as the default
         * value, and can be overridden on a per-layer basis.<br>
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
         * Note: values set by this method will be applied to all applicable layers in the network, unless a different
         * value is explicitly set on a given layer. In other words: values set via this method are used as the default
         * value, and can be overridden on a per-layer basis.<br>
         *
         * @param coefficient Weight decay regularization coefficient
         * @see #weightDecayBias(double, boolean)
         */
        public Builder weightDecayBias(double coefficient) {
            return weightDecayBias(coefficient, true);
        }

        /**
         * Weight decay for the biases only - see {@link #weightDecay(double)} for more details<br>
         * Note: values set by this method will be applied to all applicable layers in the network, unless a different
         * value is explicitly set on a given layer. In other words: values set via this method are used as the default
         * value, and can be overridden on a per-layer basis.<br>
         *
         * @param coefficient Weight decay regularization coefficient
         */
        public Builder weightDecayBias(double coefficient, boolean applyLR) {
            //Check if existing weight decay if it exists; if so, replace it. Also remove L2 - it doesn't make sense to use both
            NetworkUtils.removeInstances(this.regularizationBias, WeightDecay.class);
            if(coefficient > 0) {
                NetworkUtils.removeInstancesWithWarning(this.regularizationBias, L2Regularization.class, "L2 bias regularization removed: incompatible with added WeightDecay regularization");
                this.regularizationBias.add(new WeightDecay(coefficient, applyLR));
            }
            return this;
        }

        /**
         * Set the regularization for the parameters (excluding biases) - for example {@link WeightDecay}<br>
         * Note: values set by this method will be applied to all applicable layers in the network, unless a different
         * value is explicitly set on a given layer. In other words: values set via this method are used as the default
         * value, and can be overridden on a per-layer basis.<br>
         *
         * @param regularization Regularization to apply for the network parameters/weights (excluding biases)
         */
        public Builder regularization(List<Regularization> regularization) {
            this.regularization = regularization;
            return this;
        }

        /**
         * Set the regularization for the biases only - for example {@link WeightDecay}<br>
         * Note: values set by this method will be applied to all applicable layers in the network, unless a different
         * value is explicitly set on a given layer. In other words: values set via this method are used as the default
         * value, and can be overridden on a per-layer basis.<br>
         *
         * @param regularizationBias Regularization to apply for the network biases only
         */
        public Builder regularizationBias(List<Regularization> regularizationBias) {
            this.regularizationBias = regularizationBias;
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
         *<br>
         * Note: values set by this method will be applied to all applicable layers in the network, unless a different
         * value is explicitly set on a given layer. In other words: values set via this method are used as the default
         * value, and can be overridden on a per-layer basis.
         *
         * @param inputRetainProbability Dropout probability (probability of retaining each input activation value for a layer)
         * @see #dropOut(IDropout)
         */
        public Builder dropOut(double inputRetainProbability) {
            if(inputRetainProbability == 0.0){
                return dropOut(null);
            }
            return dropOut(new Dropout(inputRetainProbability));
        }

        /**
         * Set the dropout for all layers in this network<br>
         * Note: values set by this method will be applied to all applicable layers in the network, unless a different
         * value is explicitly set on a given layer. In other words: values set via this method are used as the default
         * value, and can be overridden on a per-layer basis.
         *
         * @param dropout Dropout, such as {@link Dropout}, {@link org.deeplearning4j.nn.conf.dropout.GaussianDropout},
         *                {@link org.deeplearning4j.nn.conf.dropout.GaussianNoise} etc
         * @return
         */
        public Builder dropOut(IDropout dropout){
            //Clone: Dropout is stateful usually - don't want to have the same instance shared in multiple places
            this.idropOut = (dropout == null ? null : dropout.clone());
            return this;
        }

        /**
         * Set the weight noise (such as {@link org.deeplearning4j.nn.conf.weightnoise.DropConnect} and
         * {@link org.deeplearning4j.nn.conf.weightnoise.WeightNoise}) for the layers in this network.<br>
         * Note: values set by this method will be applied to all applicable layers in the network, unless a different
         * value is explicitly set on a given layer. In other words: values set via this method are used as the default
         * value, and can be overridden on a per-layer basis.
         *
         * @param weightNoise Weight noise instance to use
         */
        public Builder weightNoise(IWeightNoise weightNoise){
            this.weightNoise = weightNoise;
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
         * Gradient updater configuration. For example, {@link org.nd4j.linalg.learning.config.Adam}
         * or {@link org.nd4j.linalg.learning.config.Nesterovs}<br>
         * Note: values set by this method will be applied to all applicable layers in the network, unless a different
         * value is explicitly set on a given layer. In other words: values set via this method are used as the default
         * value, and can be overridden on a per-layer basis.
         *
         * @param updater Updater to use
         */
        public Builder updater(IUpdater updater) {
            this.iUpdater = updater;
            return this;
        }

        /**
         * Gradient updater configuration, for the biases only. If not set, biases will use the updater as
         * set by {@link #updater(IUpdater)}<br>
         * Note: values set by this method will be applied to all applicable layers in the network, unless a different
         * value is explicitly set on a given layer. In other words: values set via this method are used as the default
         * value, and can be overridden on a per-layer basis.
         *
         * @param updater Updater to use for bias parameters
         */
        public Builder biasUpdater(IUpdater updater){
            this.biasUpdater = updater;
            return this;
        }

        /**
         * Gradient normalization strategy. Used to specify gradient renormalization, gradient clipping etc.
         * See {@link GradientNormalization} for details<br>
         * Note: values set by this method will be applied to all applicable layers in the network, unless a different
         * value is explicitly set on a given layer. In other words: values set via this method are used as the default
         * value, and can be overridden on a per-layer basis.
         *
         * @param gradientNormalization Type of normalization to use. Defaults to None.
         * @see GradientNormalization
         */
        public Builder gradientNormalization(GradientNormalization gradientNormalization) {
            this.gradientNormalization = gradientNormalization;
            return this;
        }

        /**
         * Threshold for gradient normalization, only used for GradientNormalization.ClipL2PerLayer,
         * GradientNormalization.ClipL2PerParamType, and GradientNormalization.ClipElementWiseAbsoluteValue<br>
         * Not used otherwise.<br>
         * L2 threshold for first two types of clipping, or absolute value threshold for last type of clipping.<br>
         * Note: values set by this method will be applied to all applicable layers in the network, unless a different
         * value is explicitly set on a given layer. In other words: values set via this method are used as the default
         * value, and can be overridden on a per-layer basis.
         */
        public Builder gradientNormalizationThreshold(double threshold) {
            this.gradientNormalizationThreshold = threshold;
            return this;
        }

        /**
         * Sets the convolution mode for convolutional layers, which impacts padding and output sizes.
         * See {@link ConvolutionMode} for details. Defaults to ConvolutionMode.TRUNCATE<br>
         * Note: values set by this method will be applied to all applicable layers in the network, unless a different
         * value is explicitly set on a given layer. In other words: values set via this method are used as the default
         * value, and can be overridden on a per-layer basis.
         * @param convolutionMode Convolution mode to use
         */
        public Builder convolutionMode(ConvolutionMode convolutionMode) {
            this.convolutionMode = convolutionMode;
            return this;
        }

        /**
         * Sets the cuDNN algo mode for convolutional layers, which impacts performance and memory usage of cuDNN.
         * See {@link ConvolutionLayer.AlgoMode} for details.  Defaults to "PREFER_FASTEST", but "NO_WORKSPACE" uses less memory.
         * <br>
         * Note: values set by this method will be applied to all applicable layers in the network, unless a different
         * value is explicitly set on a given layer. In other words: values set via this method are used as the default
         * value, and can be overridden on a per-layer basis.
         * @param cudnnAlgoMode cuDNN algo mode to use
         */
        public Builder cudnnAlgoMode(ConvolutionLayer.AlgoMode cudnnAlgoMode) {
            this.cudnnAlgoMode = cudnnAlgoMode;
            return this;
        }

        /**
         * Set constraints to be applied to all layers. Default: no constraints.<br>
         * Constraints can be used to enforce certain conditions (non-negativity of parameters, max-norm regularization,
         * etc). These constraints are applied at each iteration, after the parameters have been updated.<br>
         * Note: values set by this method will be applied to all applicable layers in the network, unless a different
         * value is explicitly set on a given layer. In other words: values set via this method are used as the default
         * value, and can be overridden on a per-layer basis.
         *
         * @param constraints Constraints to apply to all parameters of all layers
         */
        public Builder constrainAllParameters(LayerConstraint... constraints){
            this.allParamConstraints = Arrays.asList(constraints);
            return this;
        }

        /**
         * Set constraints to be applied to all layers. Default: no constraints.<br>
         * Constraints can be used to enforce certain conditions (non-negativity of parameters, max-norm regularization,
         * etc). These constraints are applied at each iteration, after the parameters have been updated.<br>
         * Note: values set by this method will be applied to all applicable layers in the network, unless a different
         * value is explicitly set on a given layer. In other words: values set via this method are used as the default
         * value, and can be overridden on a per-layer basis.
         *
         * @param constraints Constraints to apply to all bias parameters of all layers
         */
        public Builder constrainBias(LayerConstraint... constraints) {
            this.biasConstraints = Arrays.asList(constraints);
            return this;
        }

        /**
         * Set constraints to be applied to all layers. Default: no constraints.<br>
         * Constraints can be used to enforce certain conditions (non-negativity of parameters, max-norm regularization,
         * etc). These constraints are applied at each iteration, after the parameters have been updated.<br>
         * Note: values set by this method will be applied to all applicable layers in the network, unless a different
         * value is explicitly set on a given layer. In other words: values set via this method are used as the default
         * value, and can be overridden on a per-layer basis.
         *
         * @param constraints Constraints to apply to all weight parameters of all layers
         */
        public Builder constrainWeights(LayerConstraint... constraints) {
            this.weightConstraints = Arrays.asList(constraints);
            return this;
        }


        /**
         * Set the DataType for the network parameters and activations. Must be a floating point type: {@link DataType#DOUBLE},
         * {@link DataType#FLOAT} or {@link DataType#HALF}.<br>
         */
        public Builder dataType(@NonNull DataType dataType){
            Preconditions.checkState(dataType == DataType.DOUBLE || dataType == DataType.FLOAT || dataType == DataType.HALF,
                    "Data type must be a floating point type: one of DOUBLE, FLOAT, or HALF. Got datatype: %s", dataType);
            this.dataType = dataType;
            return this;
        }

        /**
         * Return a configuration based on this builder
         *
         * @return
         */
        public NeuralNetConfiguration build() {

            NeuralNetConfiguration conf = new NeuralNetConfiguration();
            conf.minimize = minimize;
            conf.maxNumLineSearchIterations = maxNumLineSearchIterations;
            conf.layer = layer;
            conf.optimizationAlgo = optimizationAlgo;
            conf.seed = seed;
            conf.stepFunction = stepFunction;
            conf.miniBatch = miniBatch;
            conf.cacheMode = this.cacheMode;
            conf.dataType = this.dataType;

            configureLayer(layer);
            if (layer instanceof FrozenLayer) {
                configureLayer(((FrozenLayer) layer).getLayer());
            }

            if (layer instanceof FrozenLayerWithBackprop) {
                configureLayer(((FrozenLayerWithBackprop) layer).getUnderlying());
            }

            return conf;
        }

        private void configureLayer(Layer layer) {
            String layerName;
            if (layer == null || layer.getLayerName() == null)
                layerName = "Layer not named";
            else
                layerName = layer.getLayerName();

            if(layer instanceof AbstractSameDiffLayer){
                AbstractSameDiffLayer sdl = (AbstractSameDiffLayer)layer;
                sdl.applyGlobalConfig(this);
            }

            if (layer != null) {
                copyConfigToLayer(layerName, layer);
            }

            if (layer instanceof FrozenLayer) {
                copyConfigToLayer(layerName, ((FrozenLayer) layer).getLayer());
            }

            if (layer instanceof FrozenLayerWithBackprop) {
                copyConfigToLayer(layerName, ((FrozenLayerWithBackprop) layer).getUnderlying());
            }

            if (layer instanceof Bidirectional) {
                Bidirectional b = (Bidirectional)layer;
                copyConfigToLayer(b.getFwd().getLayerName(), b.getFwd());
                copyConfigToLayer(b.getBwd().getLayerName(), b.getBwd());
            }

            if(layer instanceof BaseWrapperLayer){
                BaseWrapperLayer bwr = (BaseWrapperLayer)layer;
                configureLayer(bwr.getUnderlying());
            }

            if (layer instanceof ConvolutionLayer) {
                ConvolutionLayer cl = (ConvolutionLayer) layer;
                if (cl.getConvolutionMode() == null) {
                    cl.setConvolutionMode(convolutionMode);
                }
                if (cl.getCudnnAlgoMode() == null) {
                    cl.setCudnnAlgoMode(cudnnAlgoMode);
                }
            }
            if (layer instanceof SubsamplingLayer) {
                SubsamplingLayer sl = (SubsamplingLayer) layer;
                if (sl.getConvolutionMode() == null) {
                    sl.setConvolutionMode(convolutionMode);
                }
            }

            LayerValidation.generalValidation(layerName, layer, idropOut, regularization, regularizationBias,
                    allParamConstraints, weightConstraints, biasConstraints);
        }

        private void copyConfigToLayer(String layerName, Layer layer) {

            if (layer.getIDropout() == null) {
                //Dropout is stateful usually - don't want to have the same instance shared by multiple layers
                layer.setIDropout(idropOut == null ? null : idropOut.clone());
            }

            if (layer instanceof BaseLayer) {
                BaseLayer bLayer = (BaseLayer) layer;
                if (bLayer.getRegularization() == null || bLayer.getRegularization().isEmpty())
                    bLayer.setRegularization(new ArrayList<>(regularization));
                if (bLayer.getRegularizationBias() == null || bLayer.getRegularizationBias().isEmpty())
                    bLayer.setRegularizationBias(new ArrayList<>(regularizationBias));
                if (bLayer.getActivationFn() == null)
                    bLayer.setActivationFn(activationFn);
                if (bLayer.getWeightInitFn() == null)
                    bLayer.setWeightInitFn(weightInitFn);
                if (Double.isNaN(bLayer.getBiasInit()))
                    bLayer.setBiasInit(biasInit);
                if (Double.isNaN(bLayer.getGainInit()))
                    bLayer.setGainInit(gainInit);

                //Configure weight noise:
                if(weightNoise != null && ((BaseLayer) layer).getWeightNoise() == null){
                    ((BaseLayer) layer).setWeightNoise(weightNoise.clone());
                }

                //Configure updaters:
                if(iUpdater != null && bLayer.getIUpdater() == null){
                    bLayer.setIUpdater(iUpdater.clone());   //Clone the updater to avoid shared instances - in case of setLearningRate calls later
                }
                if(biasUpdater != null && bLayer.getBiasUpdater() == null){
                    bLayer.setBiasUpdater(biasUpdater.clone());     //Clone the updater to avoid shared instances - in case of setLearningRate calls later
                }

                if(bLayer.getIUpdater() == null && iUpdater == null && bLayer.initializer().numParams(bLayer) > 0){
                    //No updater set anywhere
                    IUpdater u = new Sgd();
                    bLayer.setIUpdater(u);
                    log.warn("*** No updater configuration is set for layer {} - defaulting to {} ***", layerName, u);
                }

                if (bLayer.getGradientNormalization() == null)
                    bLayer.setGradientNormalization(gradientNormalization);
                if (Double.isNaN(bLayer.getGradientNormalizationThreshold()))
                    bLayer.setGradientNormalizationThreshold(gradientNormalizationThreshold);
            }

            if (layer instanceof ActivationLayer){
                ActivationLayer al = (ActivationLayer)layer;
                if(al.getActivationFn() == null)
                    al.setActivationFn(activationFn);
            }
        }
    }

}

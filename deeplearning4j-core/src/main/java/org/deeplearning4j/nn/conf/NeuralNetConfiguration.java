/*
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

package org.deeplearning4j.nn.conf;

import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.MapperFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.fasterxml.jackson.databind.jsontype.NamedType;
import com.fasterxml.jackson.dataformat.yaml.YAMLFactory;

import lombok.Data;
import lombok.NoArgsConstructor;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.distribution.Distribution;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.stepfunctions.StepFunction;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.factory.Nd4j;

import java.io.IOException;
import java.io.Serializable;
import java.util.*;


/**
 * A Serializable configuration
 * for neural nets that covers per layer parameters
 *
 * @author Adam Gibson
 */
@Data
@NoArgsConstructor
public class NeuralNetConfiguration implements Serializable,Cloneable {

    protected Layer layer;
    //batch size: primarily used for conv nets. Will be reinforced if set.
    protected boolean miniBatch = true;
    protected int numIterations = 5;
    //number of line search iterations
    protected int maxNumLineSearchIterations = 5;
    protected long seed = System.currentTimeMillis();
    protected OptimizationAlgorithm optimizationAlgo = OptimizationAlgorithm.CONJUGATE_GRADIENT;
    //gradient keys used for ensuring order when getting and setting the gradient
    protected List<String> variables = new ArrayList<>();
    //whether to constrain the gradient to unit norm or not
    //adadelta - weight for how much to consider previous history
    protected StepFunction stepFunction;
    protected boolean useRegularization = false;
    protected boolean useDropConnect = false;
    @Deprecated
    protected boolean useSchedules = false;
    //minimize or maximize objective
    protected boolean minimize = true;
    // Graves LSTM & RNN
    @Deprecated
    private int timeSeriesLength = 1;
    protected Map<String,Double> learningRateByParam = new HashMap<>();
    protected Map<String,Double> l1ByParam = new HashMap<>();
    protected Map<String,Double> l2ByParam = new HashMap<>();
    protected LearningRatePolicy learningRatePolicy = LearningRatePolicy.None;
    protected double lrPolicyDecayRate;
    protected double lrPolicySteps;
    protected double lrPolicyPower;

    /**
     * Creates and returns a deep copy of the configuration.
     */
    @Override
    public NeuralNetConfiguration clone()  {
        try {
            NeuralNetConfiguration clone = (NeuralNetConfiguration) super.clone();
            if(clone.layer != null) clone.layer = clone.layer.clone();
            if(clone.stepFunction != null) clone.stepFunction = clone.stepFunction.clone();
            if(clone.variables != null ) clone.variables = new ArrayList<>(clone.variables);
            if(clone.learningRateByParam != null ) clone.learningRateByParam = new HashMap<>(clone.learningRateByParam);
            if(clone.l1ByParam != null ) clone.l1ByParam = new HashMap<>(clone.l1ByParam);
            if(clone.l2ByParam != null ) clone.l2ByParam = new HashMap<>(clone.l2ByParam);
            return clone;
        } catch (CloneNotSupportedException e) {
            throw new RuntimeException(e);
        }
    }

    public List<String> variables() {
        return new ArrayList<>(variables);
    }

    public void addVariable(String variable) {
        if(!variables.contains(variable)) {
            variables.add(variable);
            setLayerParamLR(variable);
        }
    }
    
    public void clearVariables(){
    	variables.clear();
    }


    public void setLayerParamLR(String variable){
        double lr = (variable.substring(0, 1) == DefaultParamInitializer.BIAS_KEY && !Double.isNaN(layer.getBiasLearningRate()))? layer.getBiasLearningRate(): layer.getLearningRate();
        double l1 = (variable.substring(0, 1) == DefaultParamInitializer.BIAS_KEY)? 0.0: layer.getL1();
        double l2 = (variable.substring(0, 1) == DefaultParamInitializer.BIAS_KEY)? 0.0: layer.getL2();
        learningRateByParam.put(variable, lr);
        l1ByParam.put(variable, l1);
        l2ByParam.put(variable, l2);

    }

    public double getLearningRateByParam(String variable){
        return learningRateByParam.get(variable);
    }

    public void setLearningRateByParam(String variable, double rate){
        learningRateByParam.put(variable, rate);
    }
    public double getL1ByParam(String variable ){
        return l1ByParam.get(variable);
    }

    public double getL2ByParam(String variable ){
        return l2ByParam.get(variable);
    }


    /**
     * Fluent interface for building a list of configurations
     */
    public static class ListBuilder extends MultiLayerConfiguration.Builder {
        private Map<Integer, Builder> layerwise;
        private Builder globalConfig;

        // Constructor
        public ListBuilder(Builder globalConfig, Map<Integer, Builder> layerMap) {
            this.globalConfig = globalConfig;
            this.layerwise = layerMap;
        }

        public ListBuilder(Builder globalConfig){
            this(globalConfig,new HashMap<Integer, Builder>());
        }


        public ListBuilder backprop(boolean backprop) {
            this.backprop = backprop;
            return this;
        }

        public ListBuilder pretrain(boolean pretrain) {
            this.pretrain = pretrain;
            return this;
        }

        public ListBuilder layer(int ind, Layer layer) {
            if(layerwise.containsKey(ind)){
                layerwise.get(ind).layer(layer);
            } else {
                layerwise.put(ind,globalConfig.clone().layer(layer));
            }
            return this;
        }

        public Map<Integer, Builder> getLayerwise() {
            return layerwise;
        }

        /**
         * Build the multi layer network
         * based on this neural network and
         * overr ridden parameters
         * @return the configuration to build
         */
        public MultiLayerConfiguration build() {
            List<NeuralNetConfiguration> list = new ArrayList<>();
            if(layerwise.size() == 0) throw new IllegalStateException("Invalid configuration: no layers defined");
            for(int i = 0; i < layerwise.size(); i++) {
                if(layerwise.get(i) == null){
                    throw new IllegalStateException("Invalid configuration: layer number " + i + " not specified. Expect layer "
                        + "numbers to be 0 to " + (layerwise.size()-1) + " inclusive (number of layers defined: " + layerwise.size() + ")");
                }
                if(layerwise.get(i).getLayer() == null) throw new IllegalStateException("Cannot construct network: Layer config for" +
                        "layer with index " + i + " is not defined)");
                list.add(layerwise.get(i).build());
            }
            return new MultiLayerConfiguration.Builder().backprop(backprop).inputPreProcessors(inputPreProcessors).
                    pretrain(pretrain).backpropType(backpropType).tBPTTForwardLength(tbpttFwdLength)
                    .tBPTTBackwardLength(tbpttBackLength)
                    .redistributeParams(redistributeParams)
                    .cnnInputSize(cnnInputSize)
                    .confs(list).build();
        }

    }
    /**
     * Return this configuration as json
     * @return this configuration represented as json
     */
    public String toYaml() {
        ObjectMapper mapper = mapperYaml();

        try {
            String ret =  mapper.writeValueAsString(this);
            return ret;

        } catch (com.fasterxml.jackson.core.JsonProcessingException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Create a neural net configuration from json
     * @param json the neural net configuration from json
     * @return
     */
    public static NeuralNetConfiguration fromYaml(String json) {
        ObjectMapper mapper = mapperYaml();
        try {
            NeuralNetConfiguration ret =  mapper.readValue(json, NeuralNetConfiguration.class);
            return ret;
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Return this configuration as json
     * @return this configuration represented as json
     */
    public String toJson() {
        ObjectMapper mapper = mapper();

        try {
            String ret =  mapper.writeValueAsString(this);
            return ret;

        } catch (com.fasterxml.jackson.core.JsonProcessingException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Create a neural net configuration from json
     * @param json the neural net configuration from json
     * @return
     */
    public static NeuralNetConfiguration fromJson(String json) {
        ObjectMapper mapper = mapper();
        try {
            NeuralNetConfiguration ret =  mapper.readValue(json, NeuralNetConfiguration.class);
            return ret;
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Object mapper for serialization of configurations
     * @return
     */
    public static ObjectMapper mapperYaml() {
        return mapperYaml;
    }

    private static final ObjectMapper mapperYaml = initMapperYaml();

    private static ObjectMapper initMapperYaml() {
        ObjectMapper ret = new ObjectMapper(new YAMLFactory());
        ret.configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);
        ret.configure(SerializationFeature.FAIL_ON_EMPTY_BEANS, false);
        ret.enable(SerializationFeature.INDENT_OUTPUT);
        return ret;
    }

    /**
     * Object mapper for serialization of configurations
     * @return
     */
    public static ObjectMapper mapper() {
        return mapper;
    }

    /**Reinitialize and return the Jackson/json ObjectMapper with additional named types.
     * This can be used to add additional subtypes at runtime (i.e., for JSON mapping with
     * types defined outside of the main DL4J codebase)
     */
    public static ObjectMapper reinitMapperWithSubtypes(Collection<NamedType> additionalTypes){
        mapper.registerSubtypes(additionalTypes.toArray(new NamedType[additionalTypes.size()]));
        //Recreate the mapper (via copy), as mapper won't use registered subtypes after first use
        mapper = mapper.copy();
        return mapper;
    }

    private static ObjectMapper mapper = initMapper();

    private static ObjectMapper initMapper() {
        ObjectMapper ret = new ObjectMapper();
        ret.configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);
        ret.configure(SerializationFeature.FAIL_ON_EMPTY_BEANS, false);
        ret.configure(MapperFeature.SORT_PROPERTIES_ALPHABETICALLY, true);
        ret.enable(SerializationFeature.INDENT_OUTPUT);
        return ret;
    }


    @Data
    public static class Builder implements Cloneable {
        protected String activationFunction = "sigmoid";
        protected WeightInit weightInit = WeightInit.XAVIER;
        protected double biasInit = 0.0;
        protected Distribution dist = new NormalDistribution(1e-3,1);
        protected double learningRate = 1e-1;
        protected double biasLearningRate;
        protected Map<Integer, Double> learningRateSchedule = new HashMap<>();
        protected double lrScoreBasedDecay;
        protected double momentum = 0.5;
        protected Map<Integer, Double> momentumSchedule = new HashMap<>();
        protected double l1 = 0.0;
        protected double l2 = 0.0;
        protected double dropOut = 0;
        protected Updater updater = Updater.SGD;
        protected double rho;
        protected double rmsDecay = 0.95;
        protected double adamMeanDecay = 0.9;
        protected double adamVarDecay = 0.999;
        protected Layer layer;
        protected boolean miniBatch = true;
        protected int numIterations = 5;
        protected int maxNumLineSearchIterations = 5;
        protected long seed = System.currentTimeMillis();
        protected boolean useRegularization = false;
        @Deprecated
        protected boolean useSchedules = false;
        protected OptimizationAlgorithm optimizationAlgo = OptimizationAlgorithm.CONJUGATE_GRADIENT;
        @Deprecated
        protected boolean constrainGradientToUnitNorm = false;
        protected StepFunction stepFunction = null;
        protected boolean useDropConnect = false;
        protected boolean minimize = true;
        @Deprecated
        protected int timeSeriesLength = 1;
        protected GradientNormalization gradientNormalization = GradientNormalization.None;
        protected double gradientNormalizationThreshold = 1.0;
        protected LearningRatePolicy learningRatePolicy = LearningRatePolicy.None;
        protected double lrPolicyDecayRate;
        protected double lrPolicySteps;
        protected double lrPolicyPower;

        /**Deprecated.
         +         * Time series length
         +         * @param timeSeriesLength
         +         * @return
         +         */
        @Deprecated
        public Builder timeSeriesLength(int timeSeriesLength) {
            this.timeSeriesLength = timeSeriesLength;
            return this;
        }

        /** Process input as minibatch vs full dataset.
         * Default set to true. */
        public Builder miniBatch(boolean miniBatch) {
            this.miniBatch = miniBatch;
            return this;
        }

        /**
         * Use drop connect: multiply the coefficients
         * by a binomial sampling wrt the dropout probability
         * @param useDropConnect whether to use drop connect or not
         * @return the
         */
        public Builder useDropConnect(boolean useDropConnect) {
            this.useDropConnect = useDropConnect;
            return this;
        }

        /** Objective function to minimize or maximize cost function
         * Default set to minimize true. */
        public Builder minimize(boolean minimize) {
            this.minimize = minimize;
            return this;
        }

        /** Maximum number of line search iterations.
         * Only applies for line search optimizers: Line Search SGD, Conjugate Gradient, LBFGS
         * is NOT applicable for standard SGD
         * @param maxNumLineSearchIterations > 0
         * @return
         */
        public Builder maxNumLineSearchIterations(int maxNumLineSearchIterations) {
            this.maxNumLineSearchIterations = maxNumLineSearchIterations;
            return this;
        }



        /** Layer class. */
        public Builder layer(Layer layer) {
            this.layer = layer;
            return this;
        }

        /** Step function to apply for back track line search.
         * Only applies for line search optimizers: Line Search SGD, Conjugate Gradient, LBFGS
         * Options: DefaultStepFunction (default), NegativeDefaultStepFunction
         * GradientStepFunction (for SGD), NegativeGradientStepFunction */
        public Builder stepFunction(StepFunction stepFunction) {
            this.stepFunction = stepFunction;
            return this;
        }

        /** <b>Deprecated</b><br>
         * Create a ListBuilder (for creating a MultiLayerConfiguration) with the specified number of layers, not including input.
         * @param size number of layers in the network
         * @deprecated Manually specifying number of layers in  is not necessary. Use {@link #list()} or {@link #list(Layer...)} methods.
         * */
        public ListBuilder list(int size) {
            return list();
        }

        /**Create a ListBuilder (for creating a MultiLayerConfiguration)<br>
         * Usage:<br>
         * <pre>
         * {@code .list()
         * .layer(0,new DenseLayer.Builder()...build())
         * ...
         * .layer(n,new OutputLayer.Builder()...build())
         * }
         * </pre>
         * */
        public ListBuilder list(){
            return new ListBuilder(this);
        }

        /**Create a ListBuilder (for creating a MultiLayerConfiguration) with the specified layers<br>
         * Usage:<br>
         * <pre>
         * {@code .list(
         *      new DenseLayer.Builder()...build(),
         *      ...,
         *      new OutputLayer.Builder()...build())
         * }
         * </pre>
         * @param layers The layer configurations for the network
         */
        public ListBuilder list(Layer... layers){
            if(layers == null || layers.length == 0) throw new IllegalArgumentException("Cannot create network with no layers");
            Map<Integer, Builder> layerMap = new HashMap<>();
            for(int i = 0; i < layers.length; i++) {
                NeuralNetConfiguration.Builder b = this.clone();
                b.layer(layers[i]);
                layerMap.put(i, b);
            }
            return new ListBuilder(this,layerMap);

        }

        public ComputationGraphConfiguration.GraphBuilder graphBuilder(){
            return new ComputationGraphConfiguration.GraphBuilder(this);
        }

        /** Number of optimization iterations. */
        public Builder iterations(int numIterations) {
            this.numIterations = numIterations;
            return this;
        }

        /** Random number generator seed. Used for reproducability between runs */
        public Builder seed(int seed) {
            this.seed = (long) seed;
            Nd4j.getRandom().setSeed(seed);
            return this;
        }

        /** Random number generator seed. Used for reproducability between runs */
        public Builder seed(long seed) {
            this.seed = seed;
            Nd4j.getRandom().setSeed(seed);
            return this;
        }

        public Builder optimizationAlgo(OptimizationAlgorithm optimizationAlgo) {
            this.optimizationAlgo = optimizationAlgo;
            return this;
        }

        /** Deprecated. Use .gradientNormalization(GradientNormalization) instead
         * @see org.deeplearning4j.nn.conf.GradientNormalization
         */
        @Deprecated
        public Builder constrainGradientToUnitNorm(boolean constrainGradientToUnitNorm) {
            this.constrainGradientToUnitNorm = constrainGradientToUnitNorm;
            return this;
        }

        /** Whether to use regularization (l1, l2, dropout, etc */
        public Builder regularization(boolean useRegularization) {
            this.useRegularization = useRegularization;
            return this;
        }

        /** Whether to use schedules, learningRateSchedule and momentumSchedule*/
        @Deprecated
        public Builder schedules(boolean schedules) {
            this.useSchedules = schedules;
            return this;
        }

        @Override
        public Builder clone() {
            try {
                Builder clone = (Builder) super.clone();
                if(clone.layer != null) clone.layer = clone.layer.clone();
                if(clone.stepFunction != null) clone.stepFunction = clone.stepFunction.clone();

                return clone;

            } catch (CloneNotSupportedException e) {
                throw new RuntimeException(e);
            }
        }

        /**Activation function / neuron non-linearity
         * Typical values include:<br>
         * "relu" (rectified linear), "tanh", "sigmoid", "softmax",
         * "hardtanh", "leakyrelu", "maxout", "softsign", "softplus"
         */
        public Builder activation(String activationFunction) {
            this.activationFunction = activationFunction;
            return this;
        }

        /** Weight initialization scheme.
         * @see org.deeplearning4j.nn.weights.WeightInit
         */
        public Builder weightInit(WeightInit weightInit) {
            this.weightInit = weightInit;
            return this;
            }

        public Builder biasInit(double biasInit) {
            this.biasInit = biasInit;
            return this;
        }

        /** Distribution to sample initial weights from. Used in conjunction with
         * .weightInit(WeightInit.DISTRIBUTION).
         */
        public Builder dist(Distribution dist) {
            this.dist = dist;
            return this;
        }

        /** Learning rate. Defaults to 1e-1*/
        public Builder learningRate(double learningRate) {
            this.learningRate = learningRate;
            return this;
        }

        /** Bias learning rate. Set this to apply a different learning rate to the bias*/
        public Builder biasLearningRate(double biasLearningRate){
            this.biasLearningRate = biasLearningRate;
            return this;
        }

        /** Learning rate schedule. Map of the iteration to the learning rate to apply at that iteration. */
        public Builder learningRateSchedule(Map<Integer, Double> learningRateSchedule) {
            this.learningRateSchedule = learningRateSchedule;
            return this;
        }

        /** Rate to decrease learningRate by when the score stops improving.
         * Learning rate is multiplied by this rate so ideally keep between 0 and 1. */
        public Builder learningRateScoreBasedDecayRate(double lrScoreBasedDecay) {
            this.lrScoreBasedDecay = lrScoreBasedDecay;
            return this;
        }

        /** L1 regularization coefficient.*/
        public Builder l1(double l1) {
            this.l1 = l1;
            return this;
        }

        /** L2 regularization coefficient. */
        public Builder l2(double l2) {
            this.l2 = l2;
            return this;
        }

        public Builder dropOut(double dropOut) {
            this.dropOut = dropOut;
            return this;
        }

        /** Momentum rate. */
        public Builder momentum(double momentum) {
            this.momentum = momentum;
            return this;
        }

        /** Momentum schedule. Map of the iteration to the momentum rate to apply at that iteration. */
        public Builder momentumAfter(Map<Integer, Double> momentumAfter) {
            this.momentumSchedule = momentumAfter;
            return this;
        }

        /** Gradient updater. For example, Updater.SGD for standard stochastic gradient descent,
         * Updater.NESTEROV for Nesterov momentum, Updater.RSMPROP for RMSProp, etc.
         * @see org.deeplearning4j.nn.conf.Updater
         */
        public Builder updater(Updater updater) {
            this.updater = updater;
            return this;
        }

        /**
         * Ada delta coefficient
         * @param rho
         * @return
         */
        public Builder rho(double rho) {
            this.rho = rho;
            return this;
        }

        /** Decay rate for RMSProp. Only applies if using .updater(Updater.RMSPROP)
         */
        public Builder rmsDecay(double rmsDecay) {
            this.rmsDecay = rmsDecay;
            return this;
        }

        /** Mean decay rate for Adam updater. Only applies if using .updater(Updater.ADAM) */
        public Builder adamMeanDecay(double adamMeanDecay) {
            this.adamMeanDecay = adamMeanDecay;
            return this;
        }

        /** Variance decay rate for Adam updater. Only applies if using .updater(Updater.ADAM) */
        public Builder adamVarDecay(double adamVarDecay) {
            this.adamVarDecay = adamVarDecay;
            return this;
        }

        /** Gradient normalization strategy. Used to specify gradient renormalization, gradient clipping etc.
         * @param gradientNormalization Type of normalization to use. Defaults to None.
         * @see org.deeplearning4j.nn.conf.GradientNormalization
         */
        public Builder gradientNormalization(GradientNormalization gradientNormalization){
            this.gradientNormalization = gradientNormalization;
            return this;
        }

        /** Threshold for gradient normalization, only used for GradientNormalization.ClipL2PerLayer,
         * GradientNormalization.ClipL2PerParamType, and GradientNormalization.ClipElementWiseAbsoluteValue<br>
         * Not used otherwise.<br>
         * L2 threshold for first two types of clipping, or absolute value threshold for last type of clipping.
         */
        public Builder gradientNormalizationThreshold(double threshold){
            this.gradientNormalizationThreshold = threshold;
            return this;
        }

        /** Learning rate decay policy. Used to adapt learning rate based on policy.
         * @param policy Type of policy to use. Defaults to None.
         */
        public Builder learningRateDecayPolicy(LearningRatePolicy policy){
            this.learningRatePolicy = policy;
            return this;
        }

        /** Set the decay rate for the learning rate decay policy.
         * @param lrPolicyDecayRate rate.
         */
        public Builder lrPolicyDecayRate(double lrPolicyDecayRate){
            this.lrPolicyDecayRate = lrPolicyDecayRate;
            return this;
        }

        /** Set the number of steps used for learning decay rate steps policy.
         * @param lrPolicySteps number of steps
         */
        public Builder lrPolicySteps(double lrPolicySteps){
            this.lrPolicySteps = lrPolicySteps;
            return this;
        }

        /** Set the power used for learning rate inverse policy.
         * @param lrPolicyPower power
         */
        public Builder lrPolicyPower(double lrPolicyPower){
            this.lrPolicyPower = lrPolicyPower;
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
            conf.numIterations = numIterations;
            conf.useRegularization = useRegularization;
            conf.useSchedules = useSchedules;
            conf.optimizationAlgo = optimizationAlgo;
            conf.seed = seed;
            conf.timeSeriesLength = timeSeriesLength;
            conf.stepFunction = stepFunction;
            conf.useDropConnect = useDropConnect;
            conf.miniBatch = miniBatch;
            conf.learningRatePolicy = learningRatePolicy;
            conf.lrPolicyDecayRate = lrPolicyDecayRate;
            conf.lrPolicySteps = lrPolicySteps;
            conf.lrPolicyPower = lrPolicyPower;

            if(layer != null ) {
                if (Double.isNaN(layer.getLearningRate())) layer.setLearningRate(learningRate);
                if (Double.isNaN(layer.getBiasLearningRate())) layer.setBiasLearningRate(learningRate);
                if (layer.getLearningRateSchedule() == null) layer.setLearningRateSchedule(learningRateSchedule);
                if (Double.isNaN(layer.getLrScoreBasedDecay())) layer.setLrScoreBasedDecay(lrScoreBasedDecay);
                if (Double.isNaN(layer.getL1())) layer.setL1(l1);
                if (Double.isNaN(layer.getL2())) layer.setL2(l2);
                if (layer.getActivationFunction() == null) layer.setActivationFunction(activationFunction);
                if (layer.getWeightInit() == null) layer.setWeightInit(weightInit);
                if (Double.isNaN(layer.getBiasInit())) layer.setBiasInit(biasInit);
                if (layer.getDist() == null) layer.setDist(dist);
                if (Double.isNaN(layer.getDropOut())) layer.setDropOut(dropOut);
                if (layer.getUpdater() == null) layer.setUpdater(updater);
                if (Double.isNaN(layer.getMomentum())) layer.setMomentum(momentum);
                if (layer.getMomentumSchedule() == null) layer.setMomentumSchedule(momentumSchedule);
                if (Double.isNaN(layer.getRho())) layer.setRho(rho);
                if (Double.isNaN(layer.getRmsDecay())) layer.setRmsDecay(rmsDecay);
                if (Double.isNaN(layer.getAdamMeanDecay())) layer.setAdamMeanDecay(adamMeanDecay);
                if (Double.isNaN(layer.getAdamVarDecay())) layer.setAdamVarDecay(adamVarDecay);
                if (layer.getGradientNormalization() == null) layer.setGradientNormalization(gradientNormalization);
                if (Double.isNaN(layer.getGradientNormalizationThreshold()))
                    layer.setGradientNormalizationThreshold(gradientNormalizationThreshold);
            }

            return conf;
        }

    }
}

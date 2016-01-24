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
    protected boolean useSchedules = false;
    //minimize or maximize objective
    protected boolean minimize = true;
    // Graves LSTM & RNN
    @Deprecated
    private int timeSeriesLength = 1;

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
            return clone;
        } catch (CloneNotSupportedException e) {
            throw new RuntimeException(e);
        }
    }

    public List<String> variables() {
        return new ArrayList<>(variables);
    }

    public void addVariable(String variable) {
        if(!variables.contains(variable))
            variables.add(variable);
    }
    
    public void clearVariables(){
    	variables.clear();
    }

    /**
     * Fluent interface for building a list of configurations
     */
    public static class ListBuilder extends MultiLayerConfiguration.Builder {
        private Map<Integer, Builder> layerwise;

        // Constructor
        public ListBuilder(Map<Integer, Builder> layerMap) {
            this.layerwise = layerMap;
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
            if (layerwise.get(0) == null && ind != 0) {
                throw new IllegalArgumentException("LayerZeroIndexError: Layer index must start from 0");
            }
            if (layerwise.size() < ind + 1) {
                throw new IllegalArgumentException("IndexOutOfBoundsError: Layer index exceeds listed size");
            }

            Builder builderWithLayer = layerwise.get(ind).layer(layer);
            layerwise.put(ind, builderWithLayer);
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
            for(int i = 0; i < layerwise.size(); i++) {
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
        private double learningRate = 1e-1;
        private Map<Integer, Double> learningRateAfter = new HashMap<>();
        private double lrScoreBasedDecay;
        private double momentum = 0.5;
        private Map<Integer, Double> momentumAfter = new HashMap<>();
        private double l1 = 0.0;
        private double l2 = 0.0;
        protected double dropOut = 0;
        protected Updater updater = Updater.SGD;
        private double rho;
        private double rmsDecay = 0.95;
        private double adamMeanDecay = 0.9;
        private double adamVarDecay = 0.999;
        private Layer layer;
        private boolean miniBatch = true;
        private int numIterations = 5;
        private int maxNumLineSearchIterations = 5;
        private long seed = System.currentTimeMillis();
        private boolean useRegularization = false;
        private boolean useSchedules = false;
        private OptimizationAlgorithm optimizationAlgo = OptimizationAlgorithm.CONJUGATE_GRADIENT;
        @Deprecated
        private boolean constrainGradientToUnitNorm = false;
        private StepFunction stepFunction = null;
        private boolean useDropConnect = false;
        private boolean minimize = true;
        @Deprecated
        private int timeSeriesLength = 1;
        private GradientNormalization gradientNormalization = GradientNormalization.None;
        private double gradientNormalizationThreshold = 1.0;



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

        /** Number of layers not including input. */
        public ListBuilder list(int size) {
            Map<Integer, Builder> layerMap = new HashMap<>();
            for(int i = 0; i < size; i++)
                layerMap.put(i, clone());
            return new ListBuilder(layerMap);
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

        /** Whether to use schedules, learningRateAfter and momentumAfter*/
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

        /** Learning rate schedule. Map of the iteration to the learning rate to apply at that iteration. */
        public Builder learningRateAfter(Map<Integer, Double> learningRateAfter) {
            this.learningRateAfter = learningRateAfter;
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
            this.momentumAfter = momentumAfter;
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

        /**
         * Return a configuration based on this builder
         *
         * @return
         */
        public NeuralNetConfiguration build() {
            if (layer == null)
                throw new IllegalStateException("No layer defined.");

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


            if(Double.isNaN(layer.getLearningRate())) layer.setLearningRate(learningRate);
            if(layer.getLearningRateAfter() == null) layer.setLearningRateAfter(learningRateAfter);
            if(Double.isNaN(layer.getLrScoreBasedDecay())) layer.setLrScoreBasedDecay(lrScoreBasedDecay);
            if(Double.isNaN(layer.getL1())) layer.setL1(l1);
            if(Double.isNaN(layer.getL2())) layer.setL2(l2);
            if(layer.getActivationFunction() == null) layer.setActivationFunction(activationFunction);
            if(layer.getWeightInit() == null) layer.setWeightInit(weightInit);
            if(Double.isNaN(layer.getBiasInit())) layer.setBiasInit(biasInit);
            if(layer.getDist() == null) layer.setDist(dist);
            if(Double.isNaN(layer.getDropOut())) layer.setDropOut(dropOut);
            if(layer.getUpdater() == null) layer.setUpdater(updater);
            if(Double.isNaN(layer.getMomentum())) layer.setMomentum(momentum);
            if(layer.getMomentumAfter() == null) layer.setMomentumAfter(momentumAfter);
            if(Double.isNaN(layer.getRho())) layer.setRho(rho);
            if(Double.isNaN(layer.getRmsDecay())) layer.setRmsDecay(rmsDecay);
            if(Double.isNaN(layer.getAdamMeanDecay())) layer.setAdamMeanDecay(adamMeanDecay);
            if(Double.isNaN(layer.getAdamVarDecay())) layer.setAdamVarDecay(adamVarDecay);
            if(layer.getGradientNormalization() == null) layer.setGradientNormalization(gradientNormalization);
            if(Double.isNaN(layer.getGradientNormalizationThreshold())) layer.setGradientNormalizationThreshold(gradientNormalizationThreshold);

            return conf;
        }

    }
}

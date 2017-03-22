/*-
 *  * Copyright 2016 Skymind,Inc.
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
 */

package org.deeplearning4j.arbiter;

import lombok.AllArgsConstructor;
import lombok.EqualsAndHashCode;
import org.deeplearning4j.arbiter.layers.LayerSpace;
import org.deeplearning4j.arbiter.optimize.parameter.FixedValue;
import org.deeplearning4j.arbiter.optimize.api.ParameterSpace;
import org.deeplearning4j.arbiter.optimize.serde.jackson.JsonMapper;
import org.deeplearning4j.arbiter.optimize.serde.jackson.YamlMapper;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.distribution.Distribution;
import org.deeplearning4j.nn.conf.stepfunctions.StepFunction;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.shade.jackson.annotation.JsonTypeInfo;
import org.nd4j.shade.jackson.core.JsonProcessingException;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * This is an abstract ParameterSpace for both MultiLayerNetworks (MultiLayerSpace) and ComputationGraph (ComputationGraphSpace)
 * <p>
 * Functionality here should match {@link org.deeplearning4j.nn.conf.NeuralNetConfiguration.Builder}
 *
 * @param <T> Type of network (MultiLayerNetwork or ComputationGraph)
 * @author Alex Black
 */
@EqualsAndHashCode
@JsonTypeInfo(use = JsonTypeInfo.Id.CLASS, include = JsonTypeInfo.As.PROPERTY, property = "type")
public abstract class BaseNetworkSpace<T> implements ParameterSpace<T> {

    protected ParameterSpace<Boolean> useDropConnect;
    protected ParameterSpace<Integer> iterations;
    protected Long seed;
    protected ParameterSpace<OptimizationAlgorithm> optimizationAlgo;
    protected ParameterSpace<Boolean> regularization;
    protected ParameterSpace<Boolean> schedules;
    protected ParameterSpace<String> activationFunction;
    protected ParameterSpace<Double> biasInit;
    protected ParameterSpace<WeightInit> weightInit;
    protected ParameterSpace<Distribution> dist;
    protected ParameterSpace<Double> learningRate;
    protected ParameterSpace<Double> biasLearningRate;
    protected ParameterSpace<Map<Integer, Double>> learningRateAfter;
    protected ParameterSpace<Double> lrScoreBasedDecay;
    protected ParameterSpace<LearningRatePolicy> learningRateDecayPolicy;
    protected ParameterSpace<Map<Integer, Double>> learningRateSchedule;
    protected ParameterSpace<Double> lrPolicyDecayRate;
    protected ParameterSpace<Double> lrPolicyPower;
    protected ParameterSpace<Double> lrPolicySteps;
    protected ParameterSpace<Integer> maxNumLineSearchIterations;
    protected ParameterSpace<Boolean> miniBatch;
    protected ParameterSpace<Boolean> minimize;
    protected ParameterSpace<StepFunction> stepFunction;
    protected ParameterSpace<Double> l1;
    protected ParameterSpace<Double> l2;
    protected ParameterSpace<Double> dropOut;
    protected ParameterSpace<Double> momentum;
    protected ParameterSpace<Map<Integer, Double>> momentumAfter;
    protected ParameterSpace<Updater> updater;
    protected ParameterSpace<Double> epsilon;
    protected ParameterSpace<Double> rho;
    protected ParameterSpace<Double> rmsDecay;
    protected ParameterSpace<Double> adamMeanDecay;
    protected ParameterSpace<Double> adamVarDecay;
    protected ParameterSpace<GradientNormalization> gradientNormalization;
    protected ParameterSpace<Double> gradientNormalizationThreshold;
    protected ParameterSpace<int[]> cnnInputSize;


    protected List<LayerConf> layerSpaces = new ArrayList<>();

    //NeuralNetConfiguration.ListBuilder/MultiLayerConfiguration.Builder<T> options:
    protected ParameterSpace<Boolean> backprop;
    protected ParameterSpace<Boolean> pretrain;
    protected ParameterSpace<BackpropType> backpropType;
    protected ParameterSpace<Integer> tbpttFwdLength;
    protected ParameterSpace<Integer> tbpttBwdLength;
    protected ParameterSpace<ConvolutionMode> convolutionMode;

    protected int numEpochs = 1;


    static {
        JsonMapper.getMapper().registerSubtypes(ComputationGraphSpace.class,MultiLayerSpace.class);
        YamlMapper.getMapper().registerSubtypes(ComputationGraphSpace.class,MultiLayerSpace.class);
    }

    @SuppressWarnings("unchecked")
    protected BaseNetworkSpace(Builder builder) {
        this.useDropConnect = builder.useDropConnect;
        this.iterations = builder.iterations;
        this.seed = builder.seed;
        this.optimizationAlgo = builder.optimizationAlgo;
        this.regularization = builder.regularization;
        this.schedules = builder.schedules;
        this.activationFunction = builder.activationFunction;
        this.biasInit = builder.biasInit;
        this.weightInit = builder.weightInit;
        this.dist = builder.dist;
        this.learningRate = builder.learningRate;
        this.biasLearningRate = builder.biasLearningRate;
        this.learningRateAfter = builder.learningRateAfter;
        this.lrScoreBasedDecay = builder.lrScoreBasedDecay;
        this.learningRateDecayPolicy = builder.learningRateDecayPolicy;
        this.learningRateSchedule = builder.learningRateSchedule;
        this.lrPolicyDecayRate = builder.lrPolicyDecayRate;
        this.lrPolicyPower = builder.lrPolicyPower;
        this.lrPolicySteps = builder.lrPolicySteps;
        this.maxNumLineSearchIterations = builder.maxNumLineSearchIterations;
        this.miniBatch = builder.miniBatch;
        this.minimize = builder.minimize;
        this.stepFunction = builder.stepFunction;
        this.l1 = builder.l1;
        this.l2 = builder.l2;
        this.dropOut = builder.dropOut;
        this.momentum = builder.momentum;
        this.momentumAfter = builder.momentumAfter;
        this.updater = builder.updater;
        this.epsilon = builder.epsilon;
        this.rho = builder.rho;
        this.rmsDecay = builder.rmsDecay;
        this.gradientNormalization = builder.gradientNormalization;
        this.gradientNormalizationThreshold = builder.gradientNormalizationThreshold;
        this.adamMeanDecay = builder.adamMeanDecay;
        this.adamVarDecay = builder.adamVarDecay;
        this.convolutionMode = builder.convolutionMode;


        this.backprop = builder.backprop;
        this.pretrain = builder.pretrain;
        this.backpropType = builder.backpropType;
        this.tbpttFwdLength = builder.tbpttFwdLength;
        this.tbpttBwdLength = builder.tbpttBwdLength;
        this.cnnInputSize = builder.cnnInputSize;

        this.numEpochs = builder.numEpochs;
    }

    protected BaseNetworkSpace(){
        //Default constructor for Jackson json/yaml serialization
    }


    protected NeuralNetConfiguration.Builder randomGlobalConf(double[] values) {
        //Create MultiLayerConfiguration...
        NeuralNetConfiguration.Builder builder = new NeuralNetConfiguration.Builder();
        if (useDropConnect != null) builder.useDropConnect(useDropConnect.getValue(values));
        if (iterations != null) builder.iterations(iterations.getValue(values));
        if (seed != null) builder.seed(seed);
        if (optimizationAlgo != null) builder.optimizationAlgo(optimizationAlgo.getValue(values));
        if (regularization != null) builder.regularization(regularization.getValue(values));
        // if(schedules != null) builder.learningRateSchedule(schedules.getValue(values));
        if (activationFunction != null) builder.activation(activationFunction.getValue(values));
        if (biasInit != null) builder.biasInit(biasInit.getValue(values));
        if (weightInit != null) builder.weightInit(weightInit.getValue(values));
        if (dist != null) builder.dist(dist.getValue(values));
        if (learningRate != null) builder.learningRate(learningRate.getValue(values));
        if (biasLearningRate != null) builder.biasLearningRate(biasLearningRate.getValue(values));
        if (learningRateAfter != null) builder.learningRateSchedule(learningRateAfter.getValue(values));
        if (lrScoreBasedDecay != null) builder.learningRateScoreBasedDecayRate(lrScoreBasedDecay.getValue(values));
        if (learningRateDecayPolicy != null) builder.learningRateDecayPolicy(learningRateDecayPolicy.getValue(values));
        if (learningRateSchedule != null) builder.learningRateSchedule(learningRateSchedule.getValue(values));
        if (lrPolicyDecayRate != null) builder.lrPolicyDecayRate(lrPolicyDecayRate.getValue(values));
        if (lrPolicyPower != null) builder.lrPolicyPower(lrPolicyPower.getValue(values));
        if (lrPolicySteps != null) builder.lrPolicySteps(lrPolicySteps.getValue(values));
        if (maxNumLineSearchIterations != null)
            builder.maxNumLineSearchIterations(maxNumLineSearchIterations.getValue(values));
        if (miniBatch != null) builder.miniBatch(miniBatch.getValue(values));
        if (minimize != null) builder.minimize(minimize.getValue(values));
        if (stepFunction != null) builder.stepFunction(stepFunction.getValue(values));
        if (l1 != null) builder.l1(l1.getValue(values));
        if (l2 != null) builder.l2(l2.getValue(values));
        if (dropOut != null) builder.dropOut(dropOut.getValue(values));
        if (momentum != null) builder.momentum(momentum.getValue(values));
        if (momentumAfter != null) builder.momentumAfter(momentumAfter.getValue(values));
        if (updater != null) builder.updater(updater.getValue(values));
        if (epsilon != null) builder.epsilon(epsilon.getValue(values));
        if (rho != null) builder.rho(rho.getValue(values));
        if (rmsDecay != null) builder.rmsDecay(rmsDecay.getValue(values));
        if (gradientNormalization != null) builder.gradientNormalization(gradientNormalization.getValue(values));
        if (gradientNormalizationThreshold != null)
            builder.gradientNormalizationThreshold(gradientNormalizationThreshold.getValue(values));
        if (adamMeanDecay != null) builder.adamMeanDecay(adamMeanDecay.getValue(values));
        if (adamVarDecay != null) builder.adamVarDecay(adamVarDecay.getValue(values));
        if (convolutionMode != null) builder.convolutionMode(convolutionMode.getValue(values));

        return builder;
    }

    @Override
    public List<ParameterSpace> collectLeaves() {
        List<ParameterSpace> list = new ArrayList<>();
        if (useDropConnect != null) list.addAll(useDropConnect.collectLeaves());
        if (iterations != null) list.addAll(iterations.collectLeaves());
        if (optimizationAlgo != null) list.addAll(optimizationAlgo.collectLeaves());
        if (regularization != null) list.addAll(regularization.collectLeaves());
        if (schedules != null) list.addAll(schedules.collectLeaves());
        if (activationFunction != null) list.addAll(activationFunction.collectLeaves());
        if (biasInit != null) list.addAll(biasInit.collectLeaves());
        if (weightInit != null) list.addAll(weightInit.collectLeaves());
        if (dist != null) list.addAll(dist.collectLeaves());
        if (learningRate != null) list.addAll(learningRate.collectLeaves());
        if (biasLearningRate != null) list.addAll(biasLearningRate.collectLeaves());
        if (learningRateAfter != null) list.addAll(learningRateAfter.collectLeaves());
        if (lrScoreBasedDecay != null) list.addAll(lrScoreBasedDecay.collectLeaves());
        if (learningRateDecayPolicy != null) list.addAll(learningRateDecayPolicy.collectLeaves());
        if (learningRateSchedule != null) list.addAll(learningRateSchedule.collectLeaves());
        if (lrPolicyDecayRate != null) list.addAll(lrPolicyDecayRate.collectLeaves());
        if (lrPolicyPower != null) list.addAll(lrPolicyPower.collectLeaves());
        if (lrPolicySteps != null) list.addAll(lrPolicySteps.collectLeaves());
        if (maxNumLineSearchIterations != null) list.addAll(maxNumLineSearchIterations.collectLeaves());
        if (miniBatch != null) list.addAll(miniBatch.collectLeaves());
        if (minimize != null) list.addAll(minimize.collectLeaves());
        if (stepFunction != null) list.addAll(minimize.collectLeaves());
        if (l1 != null) list.addAll(l1.collectLeaves());
        if (l2 != null) list.addAll(l2.collectLeaves());
        if (dropOut != null) list.addAll(dropOut.collectLeaves());
        if (momentum != null) list.addAll(momentum.collectLeaves());
        if (momentumAfter != null) list.addAll(momentumAfter.collectLeaves());
        if (updater != null) list.addAll(updater.collectLeaves());
        if (epsilon != null) list.addAll(epsilon.collectLeaves());
        if (rho != null) list.addAll(rho.collectLeaves());
        if (rmsDecay != null) list.addAll(rmsDecay.collectLeaves());
        if (gradientNormalization != null) list.addAll(gradientNormalization.collectLeaves());
        if (gradientNormalizationThreshold != null) list.addAll(gradientNormalizationThreshold.collectLeaves());
        if (cnnInputSize != null) list.addAll(cnnInputSize.collectLeaves());
        if (adamMeanDecay != null) list.addAll(adamMeanDecay.collectLeaves());
        if (adamVarDecay != null) list.addAll(adamVarDecay.collectLeaves());
        if (convolutionMode != null) list.add(convolutionMode);

        if (backprop != null) list.add(backprop);
        if (pretrain != null) list.add(pretrain);
        if (backpropType != null) list.add(backpropType);
        if (tbpttBwdLength != null) list.add(tbpttBwdLength);
        if (tbpttFwdLength != null) list.add(tbpttFwdLength);
        return list;
    }


    @Override
    public boolean isLeaf() {
        return false;
    }

    @Override
    public void setIndices(int... indices) {
        throw new UnsupportedOperationException("Cannot set indices for non leaf");
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        if (useDropConnect != null) sb.append("useDropConnect: ").append(useDropConnect).append("\n");
        if (iterations != null) sb.append("iterations: ").append(iterations).append("\n");
        if (seed != null) sb.append("seed: ").append(seed).append("\n");
        if (optimizationAlgo != null) sb.append("optimizationAlgo: ").append(optimizationAlgo).append("\n");
        if (regularization != null) sb.append("regularization: ").append(regularization).append("\n");
        if (schedules != null) sb.append("schedules: ").append(schedules).append("\n");
        if (activationFunction != null) sb.append("activationFunction: ").append(activationFunction).append("\n");
        if (weightInit != null) sb.append("weightInit: ").append(weightInit).append("\n");
        if (dist != null) sb.append("dist: ").append(dist).append("\n");
        if (learningRate != null) sb.append("learningRate: ").append(learningRate).append("\n");
        if (biasLearningRate != null) sb.append("biasLearningRate: ").append(biasLearningRate).append("\n");
        if (learningRateAfter != null) sb.append("learningRateAfter: ").append(learningRateAfter).append("\n");
        if (lrScoreBasedDecay != null) sb.append("lrScoreBasedDecay: ").append(lrScoreBasedDecay).append("\n");
        if (learningRateDecayPolicy != null)
            sb.append("learningRateDecayPolicy: ").append(learningRateDecayPolicy).append("\n");
        if (learningRateSchedule != null) sb.append("learningRateSchedule: ").append(learningRateSchedule).append("\n");
        if (lrPolicyDecayRate != null) sb.append("lrPolicyDecayRate: ").append(lrPolicyDecayRate).append("\n");
        if (lrPolicyPower != null) sb.append("lrPolicyPower: ").append(lrPolicyPower).append("\n");
        if (lrPolicySteps != null) sb.append("lrPolicySteps: ").append(lrPolicySteps).append("\n");
        if (maxNumLineSearchIterations != null)
            sb.append("maxNumLineSearchIterations: ").append(maxNumLineSearchIterations).append("\n");
        if (miniBatch != null) sb.append("miniBatch: ").append(miniBatch).append("\n");
        if (minimize != null) sb.append("minimize: ").append(minimize).append("\n");
        if (stepFunction != null) sb.append("stepFunction: ").append(stepFunction).append("\n");
        if (l1 != null) sb.append("l1: ").append(l1).append("\n");
        if (l2 != null) sb.append("l2: ").append(l2).append("\n");
        if (dropOut != null) sb.append("dropOut: ").append(dropOut).append("\n");
        if (momentum != null) sb.append("momentum: ").append(momentum).append("\n");
        if (momentumAfter != null) sb.append("momentumAfter: ").append(momentumAfter).append("\n");
        if (updater != null) sb.append("updater: ").append(updater).append("\n");
        if (epsilon != null) sb.append("epsilon: ").append(epsilon).append("\n");
        if (rho != null) sb.append("rho: ").append(rho).append("\n");
        if (rmsDecay != null) sb.append("rmsDecay: ").append(rmsDecay).append("\n");
        if (gradientNormalization != null)
            sb.append("gradientNormalization: ").append(gradientNormalization).append("\n");
        if (gradientNormalizationThreshold != null)
            sb.append("gradientNormalizationThreshold: ").append(gradientNormalizationThreshold).append("\n");
        if (backprop != null) sb.append("backprop: ").append(backprop).append("\n");
        if (pretrain != null) sb.append("pretrain: ").append(pretrain).append("\n");
        if (backpropType != null) sb.append("backpropType: ").append(backpropType).append("\n");
        if (tbpttFwdLength != null) sb.append("tbpttFwdLength: ").append(tbpttFwdLength).append("\n");
        if (tbpttBwdLength != null) sb.append("tbpttBwdLength: ").append(tbpttBwdLength).append("\n");
        if (cnnInputSize != null) sb.append("cnnInputSize: ").append(cnnInputSize).append("\n");
        if (adamMeanDecay != null) sb.append("adamMeanDecay: ").append(adamMeanDecay).append("\n");
        if (adamVarDecay != null) sb.append("adamVarDecay: ").append(adamVarDecay).append("\n");
        if (convolutionMode != null) sb.append("convolutionMode: ").append(convolutionMode).append("\n");

        int i = 0;
        for (LayerConf conf : layerSpaces) {

            sb.append("Layer config ").append(i++).append(": (Number layers:").append(conf.numLayers)
                    .append(", duplicate: ").append(conf.duplicateConfig).append("), ")
                    .append(conf.layerSpace.toString()).append("\n");
        }


        return sb.toString();
    }

    @AllArgsConstructor
    private static class LayerConf {
        private final LayerSpace<?> layerSpace;
        private final ParameterSpace<Integer> numLayers;
        private final boolean duplicateConfig;

    }

    @SuppressWarnings("unchecked")
    protected abstract static class Builder<T extends Builder<T>> {

        private ParameterSpace<String> activationFunction;
        private ParameterSpace<WeightInit> weightInit;
        private ParameterSpace<Double> biasInit;
        private ParameterSpace<Boolean> useDropConnect;
        private ParameterSpace<Integer> iterations;
        private Long seed;
        private ParameterSpace<OptimizationAlgorithm> optimizationAlgo;
        private ParameterSpace<Boolean> regularization;
        private ParameterSpace<Boolean> schedules;
        private ParameterSpace<Distribution> dist;
        private ParameterSpace<Double> learningRate;
        private ParameterSpace<Double> biasLearningRate;
        private ParameterSpace<Map<Integer, Double>> learningRateAfter;
        private ParameterSpace<Double> lrScoreBasedDecay;
        private ParameterSpace<LearningRatePolicy> learningRateDecayPolicy;
        private ParameterSpace<Map<Integer, Double>> learningRateSchedule;
        private ParameterSpace<Double> lrPolicyDecayRate;
        private ParameterSpace<Double> lrPolicyPower;
        private ParameterSpace<Double> lrPolicySteps;
        private ParameterSpace<Integer> maxNumLineSearchIterations;
        private ParameterSpace<Boolean> miniBatch;
        private ParameterSpace<Boolean> minimize;
        private ParameterSpace<StepFunction> stepFunction;
        private ParameterSpace<Double> l1;
        private ParameterSpace<Double> l2;
        private ParameterSpace<Double> dropOut;
        private ParameterSpace<Double> momentum;
        private ParameterSpace<Map<Integer, Double>> momentumAfter;
        private ParameterSpace<Updater> updater;
        private ParameterSpace<Double> epsilon;
        private ParameterSpace<Double> rho;
        private ParameterSpace<Double> rmsDecay;
        private ParameterSpace<GradientNormalization> gradientNormalization;
        private ParameterSpace<Double> gradientNormalizationThreshold;
        private ParameterSpace<int[]> cnnInputSize;
        private ParameterSpace<Double> adamMeanDecay;
        private ParameterSpace<Double> adamVarDecay;

        //NeuralNetConfiguration.ListBuilder/MultiLayerConfiguration.Builder<T> options:
        private ParameterSpace<Boolean> backprop;
        private ParameterSpace<Boolean> pretrain;
        private ParameterSpace<BackpropType> backpropType;
        private ParameterSpace<Integer> tbpttFwdLength;
        private ParameterSpace<Integer> tbpttBwdLength;
        private ParameterSpace<ConvolutionMode> convolutionMode;

        //Early stopping configuration / (fixed) number of epochs:
        private EarlyStoppingConfiguration earlyStoppingConfiguration;
        private int numEpochs = 1;


        public T useDropConnect(boolean useDropConnect) {
            return useDropConnect(new FixedValue<>(useDropConnect));
        }

        public T useDropConnect(ParameterSpace<Boolean> parameterSpace) {
            this.useDropConnect = parameterSpace;
            return (T) this;
        }

        public T iterations(int iterations) {
            return iterations(new FixedValue<>(iterations));
        }

        public T iterations(ParameterSpace<Integer> parameterSpace) {
            this.iterations = parameterSpace;
            return (T) this;
        }

        public T seed(long seed) {
            this.seed = seed;
            return (T) this;
        }

        public T optimizationAlgo(OptimizationAlgorithm optimizationAlgorithm) {
            return optimizationAlgo(new FixedValue<>(optimizationAlgorithm));
        }

        public T optimizationAlgo(ParameterSpace<OptimizationAlgorithm> parameterSpace) {
            this.optimizationAlgo = parameterSpace;
            return (T) this;
        }

        public T regularization(boolean useRegularization) {
            return regularization(new FixedValue<>(useRegularization));
        }

        public T regularization(ParameterSpace<Boolean> parameterSpace) {
            this.regularization = parameterSpace;
            return (T) this;
        }

        public T schedules(boolean schedules) {
            return schedules(new FixedValue<>(schedules));
        }

        public T schedules(ParameterSpace<Boolean> schedules) {
            this.schedules = schedules;
            return (T) this;
        }

        public T activation(String activationFunction) {
            return activation(new FixedValue<>(activationFunction));
        }

        public T activation(ParameterSpace<String> activationFunction) {
            this.activationFunction = activationFunction;
            return (T) this;
        }

        public T weightInit(WeightInit weightInit) {
            return weightInit(new FixedValue<>(weightInit));
        }

        public T weightInit(ParameterSpace<WeightInit> weightInit) {
            this.weightInit = weightInit;
            return (T) this;
        }

        public T dist(Distribution dist) {
            return dist(new FixedValue<>(dist));
        }

        public T dist(ParameterSpace<Distribution> dist) {
            this.dist = dist;
            return (T) this;
        }

        public T learningRate(double learningRate) {
            return learningRate(new FixedValue<>(learningRate));
        }

        public T learningRate(ParameterSpace<Double> learningRate) {
            this.learningRate = learningRate;
            return (T) this;
        }

        public T biasLearningRate(double learningRate) {
            return biasLearningRate(new FixedValue<>(learningRate));
        }

        public T biasLearningRate(ParameterSpace<Double> biasLearningRate) {
            this.biasLearningRate = biasLearningRate;
            return (T) this;
        }

        public T learningRateAfter(Map<Integer, Double> learningRateAfter) {
            return learningRateAfter(new FixedValue<>(learningRateAfter));
        }

        public T learningRateAfter(ParameterSpace<Map<Integer, Double>> learningRateAfter) {
            this.learningRateAfter = learningRateAfter;
            return (T) this;
        }

        public T learningRateScoreBasedDecayRate(double lrScoreBasedDecay) {
            return learningRateScoreBasedDecayRate(new FixedValue<>(lrScoreBasedDecay));
        }

        public T learningRateScoreBasedDecayRate(ParameterSpace<Double> lrScoreBasedDecay) {
            this.lrScoreBasedDecay = lrScoreBasedDecay;
            return (T) this;
        }

        public T learningRateDecayPolicy(LearningRatePolicy learningRatePolicy) {
            return learningRateDecayPolicy(new FixedValue<>(learningRatePolicy));
        }

        public T learningRateDecayPolicy(ParameterSpace<LearningRatePolicy> learningRateDecayPolicy) {
            this.learningRateDecayPolicy = learningRateDecayPolicy;
            return (T) this;
        }

        public T learningRateSchedule(Map<Integer, Double> learningRateSchedule) {
            return learningRateSchedule(new FixedValue<>(learningRateSchedule));
        }

        public T learningRateSchedule(ParameterSpace<Map<Integer, Double>> learningRateSchedule) {
            this.learningRateSchedule = learningRateSchedule;
            return (T) this;
        }

        public T lrPolicyDecayRate(double lrPolicyDecayRate) {
            return lrPolicyDecayRate(new FixedValue<>(lrPolicyDecayRate));
        }

        public T lrPolicyDecayRate(ParameterSpace<Double> lrPolicyDecayRate) {
            this.lrPolicyDecayRate = lrPolicyDecayRate;
            return (T) this;
        }

        public T lrPolicyPower(double lrPolicyPower) {
            return lrPolicyPower(new FixedValue<>(lrPolicyPower));
        }

        public T lrPolicyPower(ParameterSpace<Double> lrPolicyPower) {
            this.lrPolicyPower = lrPolicyPower;
            return (T) this;
        }

        public T lrPolicySteps(double lrPolicySteps) {
            return lrPolicySteps(new FixedValue<>(lrPolicySteps));
        }

        public T lrPolicySteps(ParameterSpace<Double> lrPolicySteps) {
            this.lrPolicySteps = lrPolicySteps;
            return (T) this;
        }

        public T maxNumLineSearchIterations(int maxNumLineSearchIterations) {
            return maxNumLineSearchIterations(new FixedValue<>(maxNumLineSearchIterations));
        }

        public T maxNumLineSearchIterations(ParameterSpace<Integer> maxNumLineSearchIterations) {
            this.maxNumLineSearchIterations = maxNumLineSearchIterations;
            return (T) this;
        }

        public T miniBatch(boolean minibatch) {
            return miniBatch(new FixedValue<>(minibatch));
        }

        public T miniBatch(ParameterSpace<Boolean> miniBatch) {
            this.miniBatch = miniBatch;
            return (T) this;
        }

        public T minimize(boolean minimize) {
            return minimize(new FixedValue<>(minimize));
        }

        public T minimize(ParameterSpace<Boolean> minimize) {
            this.minimize = minimize;
            return (T) this;
        }

        public T stepFunction(StepFunction stepFunction) {
            return stepFunction(new FixedValue<>(stepFunction));
        }

        public T stepFunction(ParameterSpace<StepFunction> stepFunction) {
            this.stepFunction = stepFunction;
            return (T) this;
        }

        public T l1(double l1) {
            return l1(new FixedValue<>(l1));
        }

        public T l1(ParameterSpace<Double> l1) {
            this.l1 = l1;
            return (T) this;
        }

        public T l2(double l2) {
            return l2(new FixedValue<>(l2));
        }

        public T l2(ParameterSpace<Double> l2) {
            this.l2 = l2;
            return (T) this;
        }

        public T dropOut(double dropOut) {
            return dropOut(new FixedValue<>(dropOut));
        }

        public T dropOut(ParameterSpace<Double> dropOut) {
            this.dropOut = dropOut;
            return (T) this;
        }

        public T momentum(double momentum) {
            return momentum(new FixedValue<>(momentum));
        }

        public T momentum(ParameterSpace<Double> momentum) {
            this.momentum = momentum;
            return (T) this;
        }

        public T momentumAfter(Map<Integer, Double> momentumAfter) {
            return momentumAfter(new FixedValue<>(momentumAfter));
        }

        public T momentumAfter(ParameterSpace<Map<Integer, Double>> momentumAfter) {
            this.momentumAfter = momentumAfter;
            return (T) this;
        }

        public T updater(Updater updater) {
            return updater(new FixedValue<>(updater));
        }

        public T updater(ParameterSpace<Updater> updater) {
            this.updater = updater;
            return (T) this;
        }

        public T epsilon(double epsilon){
            return epsilon(new FixedValue<>(epsilon));
        }

        public T epsilon(ParameterSpace<Double> epsilon){
            this.epsilon = epsilon;
            return (T) this;
        }

        public T rho(double rho) {
            return rho(new FixedValue<>(rho));
        }

        public T rho(ParameterSpace<Double> rho) {
            this.rho = rho;
            return (T) this;
        }

        public T rmsDecay(double rmsDecay) {
            return rmsDecay(new FixedValue<>(rmsDecay));
        }

        public T rmsDecay(ParameterSpace<Double> rmsDecay) {
            this.rmsDecay = rmsDecay;
            return (T) this;
        }

        public T gradientNormalization(GradientNormalization gradientNormalization) {
            return gradientNormalization(new FixedValue<>(gradientNormalization));
        }

        public T gradientNormalization(ParameterSpace<GradientNormalization> gradientNormalization) {
            this.gradientNormalization = gradientNormalization;
            return (T) this;
        }

        public T gradientNormalizationThreshold(double threshold) {
            return gradientNormalizationThreshold(new FixedValue<>(threshold));
        }

        public T gradientNormalizationThreshold(ParameterSpace<Double> gradientNormalizationThreshold) {
            this.gradientNormalizationThreshold = gradientNormalizationThreshold;
            return (T) this;
        }

        public T backprop(boolean backprop) {
            return backprop(new FixedValue<>(backprop));
        }

        public T backprop(ParameterSpace<Boolean> backprop) {
            this.backprop = backprop;
            return (T) this;
        }

        public T pretrain(boolean pretrain) {
            return pretrain(new FixedValue<>(pretrain));
        }

        public T pretrain(ParameterSpace<Boolean> pretrain) {
            this.pretrain = pretrain;
            return (T) this;
        }

        public T backpropType(BackpropType backpropType) {
            return backpropType(new FixedValue<>(backpropType));
        }

        public T backpropType(ParameterSpace<BackpropType> backpropType) {
            this.backpropType = backpropType;
            return (T) this;
        }

        public T tbpttFwdLength(int tbpttFwdLength) {
            return tbpttFwdLength(new FixedValue<>(tbpttFwdLength));
        }

        public T tbpttFwdLength(ParameterSpace<Integer> tbpttFwdLength) {
            this.tbpttFwdLength = tbpttFwdLength;
            return (T) this;
        }

        public T tbpttBwdLength(int tbpttBwdLength) {
            return tbpttBwdLength(new FixedValue<>(tbpttBwdLength));
        }

        public T tbpttBwdLength(ParameterSpace<Integer> tbpttBwdLength) {
            this.tbpttBwdLength = tbpttBwdLength;
            return (T) this;
        }

        public T convolutionMode(ConvolutionMode convolutionMode){
            return convolutionMode(new FixedValue<ConvolutionMode>(convolutionMode));
        }

        public T convolutionMode(ParameterSpace<ConvolutionMode> convolutionMode){
            this.convolutionMode = convolutionMode;
            return (T) this;
        }

        /**
         * Fixed number of training epochs. Default: 1
         * Note if both EarlyStoppingConfiguration and number of epochs is present, early stopping will be used in preference.
         */
        public T numEpochs(int numEpochs) {
            this.numEpochs = numEpochs;
            return (T) this;
        }

        public T biasInit(double biasInit) {
            return biasInit(new FixedValue<>(biasInit));
        }

        public T biasInit(ParameterSpace<Double> biasInit) {
            this.biasInit = biasInit;
            return (T) this;
        }

        public T adamMeanDecay(double adamMeanDecay) {
            return adamMeanDecay(new FixedValue<>(adamMeanDecay));
        }

        public T adamMeanDecay(ParameterSpace<Double> adamMeanDecay) {
            this.adamMeanDecay = adamMeanDecay;
            return (T) this;
        }

        public T adamVarDecay(double adamVarDecay) {
            return adamVarDecay(new FixedValue<>(adamVarDecay));
        }

        public T adamVarDecay(ParameterSpace<Double> adamVarDecay) {
            this.adamVarDecay = adamVarDecay;
            return (T) this;
        }


        public abstract <E extends BaseNetworkSpace> E build();
    }

    /**
     * Return a json configuration of this configuration space.
     *
     * @return
     */
    public String toJson() {
        try {
            return JsonMapper.getMapper().writeValueAsString(this);
        } catch (JsonProcessingException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Return a yaml configuration of this configuration space.
     *
     * @return
     */
    public String toYaml() {
        try {
            return YamlMapper.getMapper().writeValueAsString(this);
        } catch (JsonProcessingException e) {
            throw new RuntimeException(e);
        }
    }

}

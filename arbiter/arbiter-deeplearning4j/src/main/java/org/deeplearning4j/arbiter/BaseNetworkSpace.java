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

package org.deeplearning4j.arbiter;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import org.deeplearning4j.arbiter.adapter.ActivationParameterSpaceAdapter;
import org.deeplearning4j.arbiter.conf.dropout.DropoutSpace;
import org.deeplearning4j.arbiter.layers.LayerSpace;
import org.deeplearning4j.arbiter.optimize.api.AbstractParameterSpace;
import org.deeplearning4j.arbiter.optimize.api.ParameterSpace;
import org.deeplearning4j.arbiter.optimize.parameter.FixedValue;
import org.deeplearning4j.arbiter.optimize.serde.jackson.JsonMapper;
import org.deeplearning4j.arbiter.optimize.serde.jackson.YamlMapper;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.api.layers.LayerConstraint;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.distribution.Distribution;
import org.deeplearning4j.nn.conf.dropout.Dropout;
import org.deeplearning4j.nn.conf.dropout.IDropout;
import org.deeplearning4j.nn.conf.stepfunctions.StepFunction;
import org.deeplearning4j.nn.conf.weightnoise.IWeightNoise;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.shade.jackson.annotation.JsonTypeInfo;
import org.nd4j.shade.jackson.core.JsonProcessingException;

import java.util.ArrayList;
import java.util.Arrays;
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
@EqualsAndHashCode(callSuper = false)
@JsonTypeInfo(use = JsonTypeInfo.Id.CLASS, include = JsonTypeInfo.As.PROPERTY, property = "@class")
@Data
public abstract class BaseNetworkSpace<T> extends AbstractParameterSpace<T> {

    protected Long seed;
    protected ParameterSpace<OptimizationAlgorithm> optimizationAlgo;
    protected ParameterSpace<IActivation> activationFunction;
    protected ParameterSpace<Double> biasInit;
    protected ParameterSpace<WeightInit> weightInit;
    protected ParameterSpace<Distribution> dist;
    protected ParameterSpace<Integer> maxNumLineSearchIterations;
    protected ParameterSpace<Boolean> miniBatch;
    protected ParameterSpace<Boolean> minimize;
    protected ParameterSpace<StepFunction> stepFunction;
    protected ParameterSpace<Double> l1;
    protected ParameterSpace<Double> l2;
    protected ParameterSpace<Double> l1Bias;
    protected ParameterSpace<Double> l2Bias;
    protected ParameterSpace<IUpdater> updater;
    protected ParameterSpace<IUpdater> biasUpdater;
    protected ParameterSpace<IWeightNoise> weightNoise;
    private ParameterSpace<IDropout> dropout;
    protected ParameterSpace<GradientNormalization> gradientNormalization;
    protected ParameterSpace<Double> gradientNormalizationThreshold;
    protected ParameterSpace<ConvolutionMode> convolutionMode;

    protected List<LayerConf> layerSpaces = new ArrayList<>();

    //NeuralNetConfiguration.ListBuilder/MultiLayerConfiguration.Builder<T> options:
    protected ParameterSpace<Boolean> backprop;
    protected ParameterSpace<Boolean> pretrain;
    protected ParameterSpace<BackpropType> backpropType;
    protected ParameterSpace<Integer> tbpttFwdLength;
    protected ParameterSpace<Integer> tbpttBwdLength;

    protected ParameterSpace<List<LayerConstraint>> allParamConstraints;
    protected ParameterSpace<List<LayerConstraint>> weightConstraints;
    protected ParameterSpace<List<LayerConstraint>> biasConstraints;

    protected int numEpochs = 1;


    static {
        JsonMapper.getMapper().registerSubtypes(ComputationGraphSpace.class, MultiLayerSpace.class);
        YamlMapper.getMapper().registerSubtypes(ComputationGraphSpace.class, MultiLayerSpace.class);
    }

    @SuppressWarnings("unchecked")
    protected BaseNetworkSpace(Builder builder) {
        this.seed = builder.seed;
        this.optimizationAlgo = builder.optimizationAlgo;
        this.activationFunction = builder.activationFunction;
        this.biasInit = builder.biasInit;
        this.weightInit = builder.weightInit;
        this.dist = builder.dist;
        this.maxNumLineSearchIterations = builder.maxNumLineSearchIterations;
        this.miniBatch = builder.miniBatch;
        this.minimize = builder.minimize;
        this.stepFunction = builder.stepFunction;
        this.l1 = builder.l1;
        this.l2 = builder.l2;
        this.l1Bias = builder.l1Bias;
        this.l2Bias = builder.l2Bias;
        this.updater = builder.updater;
        this.biasUpdater = builder.biasUpdater;
        this.weightNoise = builder.weightNoise;
        this.dropout = builder.dropout;
        this.gradientNormalization = builder.gradientNormalization;
        this.gradientNormalizationThreshold = builder.gradientNormalizationThreshold;
        this.convolutionMode = builder.convolutionMode;
        this.allParamConstraints = builder.allParamConstraints;
        this.weightConstraints = builder.weightConstraints;
        this.biasConstraints = builder.biasConstraints;


        this.backprop = builder.backprop;
        this.pretrain = builder.pretrain;
        this.backpropType = builder.backpropType;
        this.tbpttFwdLength = builder.tbpttFwdLength;
        this.tbpttBwdLength = builder.tbpttBwdLength;

        this.numEpochs = builder.numEpochs;
    }

    protected BaseNetworkSpace() {
        //Default constructor for Jackson json/yaml serialization
    }


    protected NeuralNetConfiguration.Builder randomGlobalConf(double[] values) {
        //Create MultiLayerConfiguration...
        NeuralNetConfiguration.Builder builder = new NeuralNetConfiguration.Builder();
        if (seed != null)
            builder.seed(seed);
        if (optimizationAlgo != null)
            builder.optimizationAlgo(optimizationAlgo.getValue(values));
        if (activationFunction != null)
            builder.activation(activationFunction.getValue(values));
        if (biasInit != null)
            builder.biasInit(biasInit.getValue(values));
        if (weightInit != null)
            builder.weightInit(weightInit.getValue(values));
        if (dist != null)
            builder.dist(dist.getValue(values));
        if (maxNumLineSearchIterations != null)
            builder.maxNumLineSearchIterations(maxNumLineSearchIterations.getValue(values));
        if (miniBatch != null)
            builder.miniBatch(miniBatch.getValue(values));
        if (minimize != null)
            builder.minimize(minimize.getValue(values));
        if (stepFunction != null)
            builder.stepFunction(stepFunction.getValue(values));
        if (l1 != null)
            builder.l1(l1.getValue(values));
        if (l2 != null)
            builder.l2(l2.getValue(values));
        if (l1Bias != null)
            builder.l1Bias(l1Bias.getValue(values));
        if (l2Bias != null)
            builder.l2(l2Bias.getValue(values));
        if (updater != null)
            builder.updater(updater.getValue(values));
        if (biasUpdater != null)
            builder.biasUpdater(biasUpdater.getValue(values));
        if (weightNoise != null)
            builder.weightNoise(weightNoise.getValue(values));
        if (dropout != null)
            builder.dropOut(dropout.getValue(values));
        if (gradientNormalization != null)
            builder.gradientNormalization(gradientNormalization.getValue(values));
        if (gradientNormalizationThreshold != null)
            builder.gradientNormalizationThreshold(gradientNormalizationThreshold.getValue(values));
        if (convolutionMode != null)
            builder.convolutionMode(convolutionMode.getValue(values));
        if (allParamConstraints != null){
            List<LayerConstraint> c = allParamConstraints.getValue(values);
            if(c != null){
                builder.constrainAllParameters(c.toArray(new LayerConstraint[c.size()]));
            }
        }
        if (weightConstraints != null){
            List<LayerConstraint> c = weightConstraints.getValue(values);
            if(c != null){
                builder.constrainWeights(c.toArray(new LayerConstraint[c.size()]));
            }
        }
        if (biasConstraints != null){
            List<LayerConstraint> c = biasConstraints.getValue(values);
            if(c != null){
                builder.constrainBias(c.toArray(new LayerConstraint[c.size()]));
            }
        }

        return builder;
    }

    @Override
    public List<ParameterSpace> collectLeaves() {
        Map<String, ParameterSpace> global = getNestedSpaces();
        List<ParameterSpace> list = new ArrayList<>();
        list.addAll(global.values());

        //Note: Results on previous line does NOT include the LayerSpaces, therefore we need to add these manually...
        //This is because the type is a list, not a ParameterSpace

        for (LayerConf layerConf : layerSpaces) {
            LayerSpace ls = layerConf.getLayerSpace();
            list.addAll(ls.collectLeaves());
        }

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

        for (Map.Entry<String, ParameterSpace> e : getNestedSpaces().entrySet()) {
            sb.append(e.getKey()).append(": ").append(e.getValue()).append("\n");
        }

        int i = 0;
        for (LayerConf conf : layerSpaces) {

            sb.append("Layer config ").append(i++).append(": (Number layers:").append(conf.numLayers)
                            .append(", duplicate: ").append(conf.duplicateConfig).append("), ")
                            .append(conf.layerSpace.toString()).append("\n");
        }


        return sb.toString();
    }

    @AllArgsConstructor
    @Data
    @NoArgsConstructor
    public static class LayerConf {
        protected LayerSpace<?> layerSpace;
        protected String layerName;
        protected String[] inputs;
        protected ParameterSpace<Integer> numLayers;
        protected boolean duplicateConfig;
        protected InputPreProcessor preProcessor;
    }

    @SuppressWarnings("unchecked")
    protected abstract static class Builder<T extends Builder<T>> {
        private Long seed;
        private ParameterSpace<OptimizationAlgorithm> optimizationAlgo;
        private ParameterSpace<IActivation> activationFunction;
        private ParameterSpace<Double> biasInit;
        private ParameterSpace<WeightInit> weightInit;
        private ParameterSpace<Distribution> dist;
        private ParameterSpace<Integer> maxNumLineSearchIterations;
        private ParameterSpace<Boolean> miniBatch;
        private ParameterSpace<Boolean> minimize;
        private ParameterSpace<StepFunction> stepFunction;
        private ParameterSpace<Double> l1;
        private ParameterSpace<Double> l2;
        private ParameterSpace<Double> l1Bias;
        private ParameterSpace<Double> l2Bias;
        private ParameterSpace<IUpdater> updater;
        private ParameterSpace<IUpdater> biasUpdater;
        private ParameterSpace<IWeightNoise> weightNoise;
        private ParameterSpace<IDropout> dropout;
        private ParameterSpace<GradientNormalization> gradientNormalization;
        private ParameterSpace<Double> gradientNormalizationThreshold;
        private ParameterSpace<ConvolutionMode> convolutionMode;

        private ParameterSpace<List<LayerConstraint>> allParamConstraints;
        private ParameterSpace<List<LayerConstraint>> weightConstraints;
        private ParameterSpace<List<LayerConstraint>> biasConstraints;

        //NeuralNetConfiguration.ListBuilder/MultiLayerConfiguration.Builder<T> options:
        private ParameterSpace<Boolean> backprop;
        private ParameterSpace<Boolean> pretrain;
        private ParameterSpace<BackpropType> backpropType;
        private ParameterSpace<Integer> tbpttFwdLength;
        private ParameterSpace<Integer> tbpttBwdLength;

        //Early stopping configuration / (fixed) number of epochs:
        private EarlyStoppingConfiguration earlyStoppingConfiguration;
        private int numEpochs = 1;

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


        public T activation(Activation activationFunction) {
            return activation(new FixedValue<>(activationFunction));
        }

        public T activation(ParameterSpace<Activation> activationFunction) {
            return activationFn(new ActivationParameterSpaceAdapter(activationFunction));
        }

        public T activationFn(ParameterSpace<IActivation> activationFunction) {
            this.activationFunction = activationFunction;
            return (T) this;
        }

        public T biasInit(double biasInit){
            return biasInit(new FixedValue<>(biasInit));
        }

        public T biasInit(ParameterSpace<Double> biasInit){
            this.biasInit = biasInit;
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
        public T l1Bias(double l1Bias) {
            return l1Bias(new FixedValue<>(l1Bias));
        }

        public T l1Bias(ParameterSpace<Double> l1Bias) {
            this.l1Bias = l1Bias;
            return (T) this;
        }

        public T l2Bias(double l2Bias) {
            return l2Bias(new FixedValue<>(l2Bias));
        }

        public T l2Bias(ParameterSpace<Double> l2Bias) {
            this.l2Bias = l2Bias;
            return (T) this;
        }

        public T updater(IUpdater updater){
            return updater(new FixedValue<>(updater));
        }

        public T updater(ParameterSpace<IUpdater> updater) {
            this.updater = updater;
            return (T) this;
        }

        public T biasUpdater(IUpdater biasUpdater){
            return biasUpdater(new FixedValue<>(biasUpdater));
        }

        public T biasUpdater(ParameterSpace<IUpdater> biasUpdater){
            this.biasUpdater = biasUpdater;
            return (T)this;
        }

        public T weightNoise(IWeightNoise weightNoise){
            return weightNoise(new FixedValue<>(weightNoise));
        }

        public T weightNoise(ParameterSpace<IWeightNoise> weightNoise){
            this.weightNoise = weightNoise;
            return (T) this;
        }

        public T dropOut(double dropout){
            return idropOut(new Dropout(dropout));
        }

        public T dropOut(ParameterSpace<Double> dropOut){
            return idropOut(new DropoutSpace(dropOut));
        }

        public T idropOut(IDropout idropOut){
            return idropOut(new FixedValue<>(idropOut));
        }

        public T idropOut(ParameterSpace<IDropout> idropOut){
            this.dropout = idropOut;
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

        public T convolutionMode(ConvolutionMode convolutionMode) {
            return convolutionMode(new FixedValue<ConvolutionMode>(convolutionMode));
        }

        public T convolutionMode(ParameterSpace<ConvolutionMode> convolutionMode) {
            this.convolutionMode = convolutionMode;
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

        public T constrainWeights(LayerConstraint... constraints){
            return constrainWeights(new FixedValue<List<LayerConstraint>>(Arrays.asList(constraints)));
        }

        public T constrainWeights(ParameterSpace<List<LayerConstraint>> constraints){
            this.weightConstraints = constraints;
            return (T) this;
        }

        public T constrainBias(LayerConstraint... constraints){
            return constrainBias(new FixedValue<List<LayerConstraint>>(Arrays.asList(constraints)));
        }

        public T constrainBias(ParameterSpace<List<LayerConstraint>> constraints){
            this.biasConstraints = constraints;
            return (T) this;
        }

        public T constrainAllParams(LayerConstraint... constraints){
            return constrainAllParams(new FixedValue<List<LayerConstraint>>(Arrays.asList(constraints)));
        }

        public T constrainAllParams(ParameterSpace<List<LayerConstraint>> constraints){
            this.allParamConstraints = constraints;
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

package org.arbiter.deeplearning4j;

import lombok.AllArgsConstructor;
import org.arbiter.deeplearning4j.layers.LayerSpace;
import org.arbiter.optimize.api.ModelParameterSpace;
import org.arbiter.optimize.parameter.FixedValue;
import org.arbiter.optimize.parameter.ParameterSpace;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.distribution.Distribution;
import org.deeplearning4j.nn.weights.WeightInit;
import sun.plugin.javascript.navig4.Layer;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class MultiLayerSpace implements ModelParameterSpace<MultiLayerConfiguration> {

    private ParameterSpace<Boolean> useDropConnect;
    private ParameterSpace<Integer> iterations;
    private Long seed;
    private ParameterSpace<OptimizationAlgorithm> optimizationAlgo;
    private ParameterSpace<Boolean> regularization;
    private ParameterSpace<Boolean> schedules;
    private ParameterSpace<String> activationFunction;
    private ParameterSpace<WeightInit> weightInit;
    private ParameterSpace<Distribution> dist;
    private ParameterSpace<Double> learningRate;
    private ParameterSpace<Map<Integer,Double>> learningRateAfter;
    private ParameterSpace<Double> lrScoreBasedDecay;
    private ParameterSpace<Double> l1;
    private ParameterSpace<Double> l2;
    private ParameterSpace<Double> dropOut;
    private ParameterSpace<Double> momentum;
    private ParameterSpace<Map<Integer,Double>> momentumAfter;
    private ParameterSpace<Updater> updater;
    private ParameterSpace<Double> rho;
    private ParameterSpace<Double> rmsDecay;
    private ParameterSpace<GradientNormalization> gradientNormalization;
    private ParameterSpace<Double> gradientNormalizationThreshold;

    private List<LayerConf> layerSpaces = new ArrayList<>();

    //NeuralNetConfiguration.ListBuilder/MultiLayerConfiguration.Builder options:
    private ParameterSpace<Boolean> backprop;
    private ParameterSpace<Boolean> pretrain;
    private ParameterSpace<BackpropType> backpropType;
    private ParameterSpace<Integer> tbpttFwdLength;
    private ParameterSpace<Integer> tbpttBwdLength;

    private MultiLayerSpace(Builder builder){
        this.useDropConnect = builder.useDropConnect;
        this.iterations = builder.iterations;
        this.seed = builder.seed;
        this.optimizationAlgo = builder.optimizationAlgo;
        this.regularization = builder.regularization;
        this.schedules = builder.schedules;
        this.activationFunction = builder.activationFunction;
        this.weightInit = builder.weightInit;
        this.dist = builder.dist;
        this.learningRate = builder.learningRate;
        this.learningRateAfter = builder.learningRateAfter;
        this.lrScoreBasedDecay = builder.lrScoreBasedDecay;
        this.l1 = builder.l1;
        this.l2 = builder.l2;
        this.dropOut = builder.dropOut;
        this.momentum = builder.momentum;
        this.momentumAfter = builder.momentumAfter;
        this.updater = builder.updater;
        this.rho = builder.rho;
        this.rmsDecay = builder.rmsDecay;
        this.gradientNormalization = builder.gradientNormalization;
        this.gradientNormalizationThreshold = builder.gradientNormalizationThreshold;
        this.layerSpaces = builder.layerSpaces;

        this.backprop = builder.backprop;
        this.pretrain = builder.pretrain;
        this.backpropType = builder.backpropType;
        this.tbpttFwdLength = builder.tbpttFwdLength;
        this.tbpttBwdLength = builder.tbpttBwdLength;
    }



    @Override
    public MultiLayerConfiguration randomCandidate() {

        //First: create layer configs
        List<org.deeplearning4j.nn.conf.layers.Layer> layers = new ArrayList<>();
        for(LayerConf c : layerSpaces){
            int n = c.numLayers.randomValue();
            if(c.duplicateConfig){
                //Generate N identical configs
                org.deeplearning4j.nn.conf.layers.Layer l = c.layerSpace.randomLayer();
                for( int i=0; i<n; i++ ){
                    layers.add(l.clone());
                }
            } else {
                //Generate N indepedent configs
                for( int i=0; i<n; i++ ){
                    layers.add(c.layerSpace.randomLayer());
                }
            }
        }

        //Create MultiLayerConfiguration...
        NeuralNetConfiguration.Builder builder = new NeuralNetConfiguration.Builder();
        if(useDropConnect != null) builder.useDropConnect(useDropConnect.randomValue());
        if(iterations != null) builder.iterations(iterations.randomValue());
        if(seed != null) builder.seed(seed);
        if(optimizationAlgo != null) builder.optimizationAlgo(optimizationAlgo.randomValue());
        if(regularization != null) builder.regularization(regularization.randomValue());
        if(schedules != null) builder.schedules(schedules.randomValue());
        if(activationFunction != null) builder.activation(activationFunction.randomValue());
        if(weightInit != null) builder.weightInit(weightInit.randomValue());
        if(dist != null) builder.dist(dist.randomValue());
        if(learningRate != null) builder.learningRate(learningRate.randomValue());
        if(learningRateAfter != null) builder.learningRateAfter(learningRateAfter.randomValue());
        if(lrScoreBasedDecay != null) builder.learningRateScoreBasedDecayRate(lrScoreBasedDecay.randomValue());
        if(l1 != null) builder.l1(l1.randomValue());
        if(l2 != null) builder.l2(l2.randomValue());
        if(dropOut != null) builder.dropOut(dropOut.randomValue());
        if(momentum != null) builder.momentum(momentum.randomValue());
        if(momentumAfter != null) builder.momentumAfter(momentumAfter.randomValue());
        if(updater != null) builder.updater(updater.randomValue());
        if(rho != null) builder.rho(rho.randomValue());
        if(rmsDecay != null) builder.rmsDecay(rmsDecay.randomValue());
        if(gradientNormalization != null) builder.gradientNormalization(gradientNormalization.randomValue());
        if(gradientNormalizationThreshold != null) builder.gradientNormalizationThreshold(gradientNormalizationThreshold.randomValue());

        NeuralNetConfiguration.ListBuilder listBuilder = builder.list(layers.size());
        for( int i=0; i<layers.size(); i++ ){
            listBuilder.layer(i,layers.get(i));
        }

        if(backprop != null) listBuilder.backprop(backprop.randomValue());
        if(pretrain != null) listBuilder.pretrain(pretrain.randomValue());
        if(backpropType != null) listBuilder.backpropType(backpropType.randomValue());
        if(tbpttFwdLength != null) listBuilder.tBPTTForwardLength(tbpttFwdLength.randomValue());
        if(tbpttBwdLength != null) listBuilder.tBPTTBackwardLength(tbpttBwdLength.randomValue());

        return listBuilder.build();
    }

    @AllArgsConstructor
    private static class LayerConf {
        private final LayerSpace<?> layerSpace;
        private final ParameterSpace<Integer> numLayers;
        private final boolean duplicateConfig;

    }

    public static class Builder {

        private ParameterSpace<Boolean> useDropConnect;
        private ParameterSpace<Integer> iterations;
        private Long seed;
        private ParameterSpace<OptimizationAlgorithm> optimizationAlgo;
        private ParameterSpace<Boolean> regularization;
        private ParameterSpace<Boolean> schedules;
        private ParameterSpace<String> activationFunction;
        private ParameterSpace<WeightInit> weightInit;
        private ParameterSpace<Distribution> dist;
        private ParameterSpace<Double> learningRate;
        private ParameterSpace<Map<Integer,Double>> learningRateAfter;
        private ParameterSpace<Double> lrScoreBasedDecay;
        private ParameterSpace<Double> l1;
        private ParameterSpace<Double> l2;
        private ParameterSpace<Double> dropOut;
        private ParameterSpace<Double> momentum;
        private ParameterSpace<Map<Integer,Double>> momentumAfter;
        private ParameterSpace<Updater> updater;
        private ParameterSpace<Double> rho;
        private ParameterSpace<Double> rmsDecay;
        private ParameterSpace<GradientNormalization> gradientNormalization;
        private ParameterSpace<Double> gradientNormalizationThreshold;

        private List<LayerConf> layerSpaces = new ArrayList<>();

        //NeuralNetConfiguration.ListBuilder/MultiLayerConfiguration.Builder options:
        private ParameterSpace<Boolean> backprop;
        private ParameterSpace<Boolean> pretrain;
        private ParameterSpace<BackpropType> backpropType;
        private ParameterSpace<Integer> tbpttFwdLength;
        private ParameterSpace<Integer> tbpttBwdLength;


        public Builder useDropConnect(boolean useDropConnect){
            return useDropConnect(new FixedValue<Boolean>(useDropConnect));
        }

        public Builder useDropConnect(ParameterSpace<Boolean> parameterSpace){
            this.useDropConnect = parameterSpace;
            return this;
        }

        public Builder iterations(int iterations){
            return iterations(new FixedValue<Integer>(iterations));
        }

        public Builder iterations(ParameterSpace<Integer> parameterSpace){
            this.iterations = parameterSpace;
            return this;
        }

        public Builder seed(long seed){
            this.seed = seed;
            return this;
        }

        public Builder optimizationAlgo(OptimizationAlgorithm optimizationAlgorithm){
            return optimizationAlgo(new FixedValue<OptimizationAlgorithm>(optimizationAlgorithm));
        }

        public Builder optimizationAlgo(ParameterSpace<OptimizationAlgorithm> parameterSpace){
            this.optimizationAlgo = parameterSpace;
            return this;
        }

        public Builder regularization(boolean useRegularization){
            return regularization(new FixedValue<Boolean>(useRegularization));
        }

        public Builder regularization(ParameterSpace<Boolean> parameterSpace){
            this.regularization = parameterSpace;
            return this;
        }

        public Builder schedules(boolean schedules){
            return schedules(new FixedValue<Boolean>(schedules));
        }

        public Builder schedules(ParameterSpace<Boolean> schedules){
            this.schedules = schedules;
            return this;
        }

        public Builder activation(String activationFunction){
            return activation(new FixedValue<String>(activationFunction));
        }

        public Builder activation(ParameterSpace<String> activationFunction){
            this.activationFunction = activationFunction;
            return this;
        }

        public Builder weightInit(WeightInit weightInit){
            return weightInit(new FixedValue<WeightInit>(weightInit));
        }

        public Builder weightInit(ParameterSpace<WeightInit> weightInit){
            this.weightInit = weightInit;
            return this;
        }

        public Builder dist(Distribution dist){
            return dist(new FixedValue<Distribution>(dist));
        }

        public Builder dist(ParameterSpace<Distribution> dist){
            this.dist = dist;
            return this;
        }

        public Builder learningRate(double learningRate){
            return learningRate(new FixedValue<Double>(learningRate));
        }

        public Builder learningRate(ParameterSpace<Double> learningRate){
            this.learningRate = learningRate;
            return this;
        }

        public Builder learningRateAfter(Map<Integer,Double> learningRateAfter){
            return learningRateAfter(new FixedValue<Map<Integer, Double>>(learningRateAfter));
        }

        public Builder learningRateAfter(ParameterSpace<Map<Integer,Double>> learningRateAfter ){
            this.learningRateAfter = learningRateAfter;
            return this;
        }

        public Builder learningRateScoreBasedDecayRate(double lrScoreBasedDecay){
            return learningRateScoreBasedDecayRate(new FixedValue<Double>(lrScoreBasedDecay));
        }

        public Builder learningRateScoreBasedDecayRate(ParameterSpace<Double> lrScoreBasedDecay){
            this.lrScoreBasedDecay = lrScoreBasedDecay;
            return this;
        }

        public Builder l1(double l1){
            return l1(new FixedValue<Double>(l1));
        }

        public Builder l1(ParameterSpace<Double> l1){
            this.l1 = l1;
            return this;
        }

        public Builder l2(double l2){
            return l2(new FixedValue<Double>(l2));
        }

        public Builder l2(ParameterSpace<Double> l2){
            this.l2 = l2;
            return this;
        }

        public Builder dropOut(double dropOut){
            return dropOut(new FixedValue<Double>(dropOut));
        }

        public Builder dropOut(ParameterSpace<Double> dropOut){
            this.dropOut = dropOut;
            return this;
        }

        public Builder momentum(double momentum){
            return momentum(new FixedValue<Double>(momentum));
        }

        public Builder momentum(ParameterSpace<Double> momentum){
            this.momentum = momentum;
            return this;
        }

        public Builder momentumAfter(Map<Integer,Double> momentumAfter){
            return momentumAfter(new FixedValue<Map<Integer, Double>>(momentumAfter));
        }

        public Builder momentumAfter(ParameterSpace<Map<Integer,Double>> momentumAfter){
            this.momentumAfter = momentumAfter;
            return this;
        }

        public Builder updater(Updater updater){
            return updater(new FixedValue<Updater>(updater));
        }

        public Builder updater(ParameterSpace<Updater> updater){
            this.updater = updater;
            return this;
        }

        public Builder rho(double rho){
            return rho(new FixedValue<Double>(rho));
        }

        public Builder rho(ParameterSpace<Double> rho){
            this.rho = rho;
            return this;
        }

        public Builder rmsDecay(double rmsDecay){
            return rmsDecay(new FixedValue<Double>(rmsDecay));
        }

        public Builder rmsDecay(ParameterSpace<Double> rmsDecay){
            this.rmsDecay = rmsDecay;
            return this;
        }

        public Builder gradientNormalization(GradientNormalization gradientNormalization){
            return gradientNormalization(new FixedValue<GradientNormalization>(gradientNormalization));
        }

        public Builder gradientNormalization(ParameterSpace<GradientNormalization> gradientNormalization){
            this.gradientNormalization = gradientNormalization;
            return this;
        }

        public Builder gradientNormalizationThreshold(double threshold){
            return gradientNormalizationThreshold(new FixedValue<Double>(threshold));
        }

        public Builder gradientNormalizationThreshold(ParameterSpace<Double> gradientNormalizationThreshold){
            this.gradientNormalizationThreshold = gradientNormalizationThreshold;
            return this;
        }

        public Builder backprop(boolean backprop){
            return backprop(new FixedValue<Boolean>(backprop));
        }

        public Builder backprop(ParameterSpace<Boolean> backprop){
            this.backprop = backprop;
            return this;
        }

        public Builder pretrain(boolean pretrain){
            return pretrain(new FixedValue<Boolean>(pretrain));
        }

        public Builder pretrain(ParameterSpace<Boolean> pretrain){
            this.pretrain = pretrain;
            return this;
        }

        public Builder backpropType(BackpropType backpropType){
            return backpropType(new FixedValue<BackpropType>(backpropType));
        }

        public Builder backpropType(ParameterSpace<BackpropType> backpropType){
            this.backpropType = backpropType;
            return this;
        }

        public Builder tbpttFwdLength(int tbpttFwdLength){
            return tbpttFwdLength(new FixedValue<Integer>(tbpttFwdLength));
        }

        public Builder tbpttFwdLength(ParameterSpace<Integer> tbpttFwdLength){
            this.tbpttBwdLength = tbpttFwdLength;
            return this;
        }

        public Builder tbpttBwdLength(int tbpttBwdLength){
            return tbpttBwdLength(new FixedValue<Integer>(tbpttBwdLength));
        }

        public Builder tbpttBwdLength(ParameterSpace<Integer> tbpttBwdLength){
            this.tbpttBwdLength = tbpttBwdLength;
            return this;
        }


        public Builder addLayer(LayerSpace<?> layerSpace){
            return addLayer(layerSpace,new FixedValue<Integer>(1),true);
        }



        /**
         * @param layerSpace
         * @param numLayersDistribution Distribution for number of layers to generate
         * @param duplicateConfig Only used if more than 1 layer can be generated. If true: generate N identical (stacked) layers.
         *                        If false: generate N independent layers
         */
        public Builder addLayer(LayerSpace<? extends org.deeplearning4j.nn.conf.layers.Layer> layerSpace,
                                ParameterSpace<Integer> numLayersDistribution, boolean duplicateConfig){
            layerSpaces.add(new LayerConf(layerSpace,numLayersDistribution,duplicateConfig));
            return this;
        }

        public MultiLayerSpace build(){
            return new MultiLayerSpace(this);
        }
    }

}

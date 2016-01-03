package org.arbiter.deeplearning4j;

import lombok.AllArgsConstructor;
import org.arbiter.deeplearning4j.layers.LayerSpace;
import org.arbiter.optimize.api.ModelParameterSpace;
import org.arbiter.optimize.parameter.FixedValue;
import org.arbiter.optimize.parameter.ParameterSpace;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.distribution.Distribution;
import org.deeplearning4j.nn.conf.layers.FeedForwardLayer;
import org.deeplearning4j.nn.weights.WeightInit;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

//public class MultiLayerSpace implements ModelParameterSpace<MultiLayerConfiguration> {
public class MultiLayerSpace implements ModelParameterSpace<DL4JConfiguration> {

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
    private ParameterSpace<int[]> cnnInputSize;

    private List<LayerConf> layerSpaces = new ArrayList<>();

    //NeuralNetConfiguration.ListBuilder/MultiLayerConfiguration.Builder options:
    private ParameterSpace<Boolean> backprop;
    private ParameterSpace<Boolean> pretrain;
    private ParameterSpace<BackpropType> backpropType;
    private ParameterSpace<Integer> tbpttFwdLength;
    private ParameterSpace<Integer> tbpttBwdLength;

    //Early stopping configuration / (fixed) number of epochs:
    private EarlyStoppingConfiguration earlyStoppingConfiguration;
    private int numEpochs = 1;

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
        this.cnnInputSize = builder.cnnInputSize;

        this.earlyStoppingConfiguration = builder.earlyStoppingConfiguration;
        this.numEpochs = builder.numEpochs;
    }



    @Override
    public DL4JConfiguration randomCandidate() {

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


        //Set nIn based on nOut of previous layer.
        //TODO This won't work for all cases (at minimum: cast is an issue)
        int lastNOut = ((FeedForwardLayer)layers.get(0)).getNOut();
        for( int i=1; i<layers.size(); i++ ){
            FeedForwardLayer ffl = (FeedForwardLayer)layers.get(i);
            ffl.setNIn(lastNOut);
            lastNOut = ffl.getNOut();
        }

        NeuralNetConfiguration.ListBuilder listBuilder = builder.list(layers.size());
        for( int i=0; i<layers.size(); i++ ){
            listBuilder.layer(i,layers.get(i));
        }

        if(backprop != null) listBuilder.backprop(backprop.randomValue());
        if(pretrain != null) listBuilder.pretrain(pretrain.randomValue());
        if(backpropType != null) listBuilder.backpropType(backpropType.randomValue());
        if(tbpttFwdLength != null) listBuilder.tBPTTForwardLength(tbpttFwdLength.randomValue());
        if(tbpttBwdLength != null) listBuilder.tBPTTBackwardLength(tbpttBwdLength.randomValue());
        if(cnnInputSize != null) listBuilder.cnnInputSize(cnnInputSize.randomValue());

        MultiLayerConfiguration configuration = listBuilder.build();
        return new DL4JConfiguration(configuration,earlyStoppingConfiguration,numEpochs);
    }

    @Override
    public String toString(){
        StringBuilder sb = new StringBuilder();
        if(useDropConnect != null) sb.append("useDropConnect: ").append(useDropConnect).append("\n");
        if(iterations != null) sb.append("iterations: ").append(iterations).append("\n");
        if(seed != null) sb.append("seed: ").append(seed).append("\n");
        if(optimizationAlgo != null) sb.append("optimizationAlgo: ").append(optimizationAlgo).append("\n");
        if(regularization != null) sb.append("regularization: ").append(regularization).append("\n");
        if(schedules != null) sb.append("schedules: ").append(schedules).append("\n");
        if(activationFunction != null) sb.append("activationFunction: ").append(activationFunction).append("\n");
        if(weightInit != null) sb.append("weightInit: ").append(weightInit).append("\n");
        if(dist != null) sb.append("dist: ").append(dist).append("\n");
        if(learningRate != null) sb.append("learningRate: ").append(learningRate).append("\n");
        if(learningRateAfter != null) sb.append("learningRateAfter: ").append(learningRateAfter).append("\n");
        if(lrScoreBasedDecay != null) sb.append("lrScoreBasedDecay: ").append(lrScoreBasedDecay).append("\n");
        if(l1 != null) sb.append("l1: ").append(l1).append("\n");
        if(l2 != null) sb.append("l2: ").append(l2).append("\n");
        if(dropOut != null) sb.append("dropOut: ").append(dropOut).append("\n");
        if(momentum != null) sb.append("momentum: ").append(momentum).append("\n");
        if(momentumAfter != null) sb.append("momentumAfter: ").append(momentumAfter).append("\n");
        if(updater != null) sb.append("updater: ").append(updater).append("\n");
        if(rho != null) sb.append("rho: ").append(rho).append("\n");
        if(rmsDecay != null) sb.append("rmsDecay: ").append(rmsDecay).append("\n");
        if(gradientNormalization != null) sb.append("gradientNormalization: ").append(gradientNormalization).append("\n");
        if(gradientNormalizationThreshold != null) sb.append("gradientNormalizationThreshold: ").append(gradientNormalizationThreshold).append("\n");
        if(backprop != null) sb.append("backprop: ").append(backprop).append("\n");
        if(pretrain != null) sb.append("pretrain: ").append(pretrain).append("\n");
        if(backpropType != null) sb.append("backpropType: ").append(backpropType).append("\n");
        if(tbpttFwdLength != null) sb.append("tbpttFwdLength: ").append(tbpttFwdLength).append("\n");
        if(tbpttBwdLength != null) sb.append("tbpttBwdLength: ").append(tbpttBwdLength).append("\n");
        if(cnnInputSize != null) sb.append("cnnInputSize: ").append(cnnInputSize).append("\n");

        int i=0;
        for(LayerConf conf : layerSpaces){

            sb.append("Layer config ").append(i++).append(": (Number layers:").append(conf.numLayers)
                    .append(", duplicate: ").append(conf.duplicateConfig).append("), ")
                    .append(conf.layerSpace.toString()).append("\n");
        }

        if(earlyStoppingConfiguration != null){
            sb.append("Early stopping configuration:").append(earlyStoppingConfiguration.toString()).append("\n");
        } else {
            sb.append("Training # epochs:").append(numEpochs).append("\n");
        }

        return sb.toString();
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
        private ParameterSpace<int[]> cnnInputSize;

        private List<LayerConf> layerSpaces = new ArrayList<>();

        //NeuralNetConfiguration.ListBuilder/MultiLayerConfiguration.Builder options:
        private ParameterSpace<Boolean> backprop;
        private ParameterSpace<Boolean> pretrain;
        private ParameterSpace<BackpropType> backpropType;
        private ParameterSpace<Integer> tbpttFwdLength;
        private ParameterSpace<Integer> tbpttBwdLength;

        //Early stopping configuration / (fixed) number of epochs:
        private EarlyStoppingConfiguration earlyStoppingConfiguration;
        private int numEpochs = 1;


        public Builder useDropConnect(boolean useDropConnect){
            return useDropConnect(new FixedValue<>(useDropConnect));
        }

        public Builder useDropConnect(ParameterSpace<Boolean> parameterSpace){
            this.useDropConnect = parameterSpace;
            return this;
        }

        public Builder iterations(int iterations){
            return iterations(new FixedValue<>(iterations));
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
            return optimizationAlgo(new FixedValue<>(optimizationAlgorithm));
        }

        public Builder optimizationAlgo(ParameterSpace<OptimizationAlgorithm> parameterSpace){
            this.optimizationAlgo = parameterSpace;
            return this;
        }

        public Builder regularization(boolean useRegularization){
            return regularization(new FixedValue<>(useRegularization));
        }

        public Builder regularization(ParameterSpace<Boolean> parameterSpace){
            this.regularization = parameterSpace;
            return this;
        }

        public Builder schedules(boolean schedules){
            return schedules(new FixedValue<>(schedules));
        }

        public Builder schedules(ParameterSpace<Boolean> schedules){
            this.schedules = schedules;
            return this;
        }

        public Builder activation(String activationFunction){
            return activation(new FixedValue<>(activationFunction));
        }

        public Builder activation(ParameterSpace<String> activationFunction){
            this.activationFunction = activationFunction;
            return this;
        }

        public Builder weightInit(WeightInit weightInit){
            return weightInit(new FixedValue<>(weightInit));
        }

        public Builder weightInit(ParameterSpace<WeightInit> weightInit){
            this.weightInit = weightInit;
            return this;
        }

        public Builder dist(Distribution dist){
            return dist(new FixedValue<>(dist));
        }

        public Builder dist(ParameterSpace<Distribution> dist){
            this.dist = dist;
            return this;
        }

        public Builder learningRate(double learningRate){
            return learningRate(new FixedValue<>(learningRate));
        }

        public Builder learningRate(ParameterSpace<Double> learningRate){
            this.learningRate = learningRate;
            return this;
        }

        public Builder learningRateAfter(Map<Integer,Double> learningRateAfter){
            return learningRateAfter(new FixedValue<>(learningRateAfter));
        }

        public Builder learningRateAfter(ParameterSpace<Map<Integer,Double>> learningRateAfter ){
            this.learningRateAfter = learningRateAfter;
            return this;
        }

        public Builder learningRateScoreBasedDecayRate(double lrScoreBasedDecay){
            return learningRateScoreBasedDecayRate(new FixedValue<>(lrScoreBasedDecay));
        }

        public Builder learningRateScoreBasedDecayRate(ParameterSpace<Double> lrScoreBasedDecay){
            this.lrScoreBasedDecay = lrScoreBasedDecay;
            return this;
        }

        public Builder l1(double l1){
            return l1(new FixedValue<>(l1));
        }

        public Builder l1(ParameterSpace<Double> l1){
            this.l1 = l1;
            return this;
        }

        public Builder l2(double l2){
            return l2(new FixedValue<>(l2));
        }

        public Builder l2(ParameterSpace<Double> l2){
            this.l2 = l2;
            return this;
        }

        public Builder dropOut(double dropOut){
            return dropOut(new FixedValue<>(dropOut));
        }

        public Builder dropOut(ParameterSpace<Double> dropOut){
            this.dropOut = dropOut;
            return this;
        }

        public Builder momentum(double momentum){
            return momentum(new FixedValue<>(momentum));
        }

        public Builder momentum(ParameterSpace<Double> momentum){
            this.momentum = momentum;
            return this;
        }

        public Builder momentumAfter(Map<Integer,Double> momentumAfter){
            return momentumAfter(new FixedValue<>(momentumAfter));
        }

        public Builder momentumAfter(ParameterSpace<Map<Integer,Double>> momentumAfter){
            this.momentumAfter = momentumAfter;
            return this;
        }

        public Builder updater(Updater updater){
            return updater(new FixedValue<>(updater));
        }

        public Builder updater(ParameterSpace<Updater> updater){
            this.updater = updater;
            return this;
        }

        public Builder rho(double rho){
            return rho(new FixedValue<>(rho));
        }

        public Builder rho(ParameterSpace<Double> rho){
            this.rho = rho;
            return this;
        }

        public Builder rmsDecay(double rmsDecay){
            return rmsDecay(new FixedValue<>(rmsDecay));
        }

        public Builder rmsDecay(ParameterSpace<Double> rmsDecay){
            this.rmsDecay = rmsDecay;
            return this;
        }

        public Builder gradientNormalization(GradientNormalization gradientNormalization){
            return gradientNormalization(new FixedValue<>(gradientNormalization));
        }

        public Builder gradientNormalization(ParameterSpace<GradientNormalization> gradientNormalization){
            this.gradientNormalization = gradientNormalization;
            return this;
        }

        public Builder gradientNormalizationThreshold(double threshold){
            return gradientNormalizationThreshold(new FixedValue<>(threshold));
        }

        public Builder gradientNormalizationThreshold(ParameterSpace<Double> gradientNormalizationThreshold){
            this.gradientNormalizationThreshold = gradientNormalizationThreshold;
            return this;
        }

        public Builder backprop(boolean backprop){
            return backprop(new FixedValue<>(backprop));
        }

        public Builder backprop(ParameterSpace<Boolean> backprop){
            this.backprop = backprop;
            return this;
        }

        public Builder pretrain(boolean pretrain){
            return pretrain(new FixedValue<>(pretrain));
        }

        public Builder pretrain(ParameterSpace<Boolean> pretrain){
            this.pretrain = pretrain;
            return this;
        }

        public Builder backpropType(BackpropType backpropType){
            return backpropType(new FixedValue<>(backpropType));
        }

        public Builder backpropType(ParameterSpace<BackpropType> backpropType){
            this.backpropType = backpropType;
            return this;
        }

        public Builder tbpttFwdLength(int tbpttFwdLength){
            return tbpttFwdLength(new FixedValue<>(tbpttFwdLength));
        }

        public Builder tbpttFwdLength(ParameterSpace<Integer> tbpttFwdLength){
            this.tbpttBwdLength = tbpttFwdLength;
            return this;
        }

        public Builder tbpttBwdLength(int tbpttBwdLength){
            return tbpttBwdLength(new FixedValue<>(tbpttBwdLength));
        }

        public Builder tbpttBwdLength(ParameterSpace<Integer> tbpttBwdLength){
            this.tbpttBwdLength = tbpttBwdLength;
            return this;
        }

        public Builder cnnInputSize(int height, int width, int depth){
            return cnnInputSize(new FixedValue<>(new int[]{height, width, depth}));
        }

        public Builder cnnInputSize(ParameterSpace<int[]> cnnInputSize){
            this.cnnInputSize = cnnInputSize;
            return this;
        }


        public Builder addLayer(LayerSpace<?> layerSpace){
            return addLayer(layerSpace,new FixedValue<>(1),true);
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

        /** Early stopping configuration (optional). Note if both EarlyStoppingConfiguration and number of epochs is
         * present, early stopping will be used in preference.
         */
        public Builder earlyStoppingConfiguration(EarlyStoppingConfiguration earlyStoppingConfiguration){
            this.earlyStoppingConfiguration = earlyStoppingConfiguration;
            return this;
        }

        /** Fixed number of training epochs. Default: 1
         * Note if both EarlyStoppingConfiguration and number of epochs is present, early stopping will be used in preference.
         */
        public Builder numEpochs(int numEpochs){
            this.numEpochs = numEpochs;
            return this;
        }

        public MultiLayerSpace build(){
            return new MultiLayerSpace(this);
        }
    }

}

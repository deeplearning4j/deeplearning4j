package org.arbiter.deeplearning4j;

import org.arbiter.optimize.api.ModelParameterSpace;
import org.arbiter.optimize.parameter.FixedValue;
import org.arbiter.optimize.parameter.ParameterSpace;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.Distribution;
import org.deeplearning4j.nn.weights.WeightInit;

import java.util.Map;

/**
 * Created by Alex on 26/12/2015.
 */
public class MultiLayerSpace implements ModelParameterSpace<MultiLayerConfiguration> {


    @Override
    public MultiLayerConfiguration randomCandidate() {
        throw new UnsupportedOperationException("Not yet implemented");
    }

    public static class Builder {

        private ParameterSpace<Boolean> useDropConnect;
        private ParameterSpace<Integer> iterations;
        private long seed;
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
            return momentumAfter(new FixedValue<Map<Integer,Double>>(momentumAfter));
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
    }

}

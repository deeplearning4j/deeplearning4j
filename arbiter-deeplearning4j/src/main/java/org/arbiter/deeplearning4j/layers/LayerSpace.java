package org.arbiter.deeplearning4j.layers;

import org.arbiter.optimize.parameter.FixedValue;
import org.arbiter.optimize.parameter.ParameterSpace;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.Distribution;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.weights.WeightInit;

import java.util.Map;

/**
 * Created by Alex on 26/12/2015.
 */
public abstract class LayerSpace<L extends Layer> {

    protected ParameterSpace<String> activationFunction;
    protected ParameterSpace<WeightInit> weightInit;
    protected ParameterSpace<Double> biasInit;
    protected ParameterSpace<Distribution> dist;
    protected ParameterSpace<Double> learningRate;
    protected ParameterSpace<Map<Integer,Double>> learningRateAfter;
    protected ParameterSpace<Double> lrScoreBasedDecay;
    protected ParameterSpace<Double> l1;
    protected ParameterSpace<Double> l2;
    protected ParameterSpace<Double> dropOut;
    protected ParameterSpace<Double> momentum;
    protected ParameterSpace<Map<Integer,Double>> momentumAfter;
    protected ParameterSpace<Updater> updater;
    protected ParameterSpace<Double> rho;
    protected ParameterSpace<Double> rmsDecay;
    protected ParameterSpace<GradientNormalization> gradientNormalization;
    protected ParameterSpace<Double> gradientNormalizationThreshold;

    @SuppressWarnings("unchecked")
    protected LayerSpace(Builder builder){
        this.activationFunction = builder.activationFunction;
        this.weightInit = builder.weightInit;
        this.biasInit = builder.biasInit;
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
    }

    public abstract L randomLayer();

    protected void setLayerOptionsBuilder(Layer.Builder builder){
        if(activationFunction != null) builder.activation(activationFunction.randomValue());
        if(weightInit != null) builder.weightInit(weightInit.randomValue());
        if(biasInit != null) builder.biasInit(biasInit.randomValue());
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
    }



//    public abstract static class Builder<T extends Builder<T>> {
    @SuppressWarnings("unchecked")
    public abstract static class Builder<T> {
        protected ParameterSpace<String> activationFunction;
        protected ParameterSpace<WeightInit> weightInit;
        protected ParameterSpace<Double> biasInit;
        protected ParameterSpace<Distribution> dist;
        protected ParameterSpace<Double> learningRate;
        protected ParameterSpace<Map<Integer,Double>> learningRateAfter;
        protected ParameterSpace<Double> lrScoreBasedDecay;
        protected ParameterSpace<Double> l1;
        protected ParameterSpace<Double> l2;
        protected ParameterSpace<Double> dropOut;
        protected ParameterSpace<Double> momentum;
        protected ParameterSpace<Map<Integer,Double>> momentumAfter;
        protected ParameterSpace<Updater> updater;
        protected ParameterSpace<Double> rho;
        protected ParameterSpace<Double> rmsDecay;
        protected ParameterSpace<GradientNormalization> gradientNormalization;
        protected ParameterSpace<Double> gradientNormalizationThreshold;


        public T activation(String activationFunction){
            return activation(new FixedValue<String>(activationFunction));
        }

        public T activation(ParameterSpace<String> activationFunction){
            this.activationFunction = activationFunction;
            return (T)this;
        }

        public T weightInit(WeightInit weightInit){
            return (T)weightInit(new FixedValue<WeightInit>(weightInit));
        }

        public T weightInit(ParameterSpace<WeightInit> weightInit){
            this.weightInit = weightInit;
            return (T)this;
        }

        public T dist(Distribution dist){
            return dist(new FixedValue<Distribution>(dist));
        }

        public T dist(ParameterSpace<Distribution> dist){
            this.dist = dist;
            return (T)this;
        }

        public T learningRate(double learningRate){
            return learningRate(new FixedValue<Double>(learningRate));
        }

        public T learningRate(ParameterSpace<Double> learningRate){
            this.learningRate = learningRate;
            return (T)this;
        }

        public T learningRateAfter(Map<Integer,Double> learningRateAfter){
            return learningRateAfter(new FixedValue<Map<Integer, Double>>(learningRateAfter));
        }

        public T learningRateAfter(ParameterSpace<Map<Integer,Double>> learningRateAfter ){
            this.learningRateAfter = learningRateAfter;
            return (T)this;
        }

        public T learningRateScoreBasedDecayRate(double lrScoreBasedDecay){
            return learningRateScoreBasedDecayRate(new FixedValue<Double>(lrScoreBasedDecay));
        }

        public T learningRateScoreBasedDecayRate(ParameterSpace<Double> lrScoreBasedDecay){
            this.lrScoreBasedDecay = lrScoreBasedDecay;
            return (T)this;
        }

        public T l1(double l1){
            return l1(new FixedValue<Double>(l1));
        }

        public T l1(ParameterSpace<Double> l1){
            this.l1 = l1;
            return (T)this;
        }

        public T l2(double l2){
            return l2(new FixedValue<Double>(l2));
        }

        public T l2(ParameterSpace<Double> l2){
            this.l2 = l2;
            return (T)this;
        }

        public T dropOut(double dropOut){
            return dropOut(new FixedValue<Double>(dropOut));
        }

        public T dropOut(ParameterSpace<Double> dropOut){
            this.dropOut = dropOut;
            return (T)this;
        }

        public T momentum(double momentum){
            return momentum(new FixedValue<Double>(momentum));
        }

        public T momentum(ParameterSpace<Double> momentum){
            this.momentum = momentum;
            return (T)this;
        }

        public T momentumAfter(Map<Integer,Double> momentumAfter){
            return momentumAfter(new FixedValue<Map<Integer,Double>>(momentumAfter));
        }

        public T momentumAfter(ParameterSpace<Map<Integer,Double>> momentumAfter){
            this.momentumAfter = momentumAfter;
            return (T)this;
        }

        public T updater(Updater updater){
            return updater(new FixedValue<Updater>(updater));
        }

        public T updater(ParameterSpace<Updater> updater){
            this.updater = updater;
            return (T)this;
        }

        public T rho(double rho){
            return rho(new FixedValue<Double>(rho));
        }

        public T rho(ParameterSpace<Double> rho){
            this.rho = rho;
            return (T)this;
        }

        public T rmsDecay(double rmsDecay){
            return rmsDecay(new FixedValue<Double>(rmsDecay));
        }

        public T rmsDecay(ParameterSpace<Double> rmsDecay){
            this.rmsDecay = rmsDecay;
            return (T)this;
        }

        public T gradientNormalization(GradientNormalization gradientNormalization){
            return gradientNormalization(new FixedValue<GradientNormalization>(gradientNormalization));
        }

        public T gradientNormalization(ParameterSpace<GradientNormalization> gradientNormalization){
            this.gradientNormalization = gradientNormalization;
            return (T)this;
        }

        public T gradientNormalizationThreshold(double threshold){
            return gradientNormalizationThreshold(new FixedValue<Double>(threshold));
        }

        public T gradientNormalizationThreshold(ParameterSpace<Double> gradientNormalizationThreshold){
            this.gradientNormalizationThreshold = gradientNormalizationThreshold;
            return (T)this;
        }

        public abstract <E extends LayerSpace> E build();
    }

}

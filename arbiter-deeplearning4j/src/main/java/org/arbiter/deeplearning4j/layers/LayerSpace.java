/*
 *
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
 *
 */
package org.arbiter.deeplearning4j.layers;

import org.arbiter.optimize.parameter.FixedValue;
import org.arbiter.optimize.api.ParameterSpace;
import org.arbiter.util.CollectionUtils;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.Distribution;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.weights.WeightInit;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Created by Alex on 26/12/2015.
 */
public abstract class LayerSpace<L extends Layer> implements ParameterSpace<L> {

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

    private int numParameters;

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

        numParameters = CollectionUtils.countUnique(collectLeaves());
    }

//    public abstract L randomLayer();
    
    public List<ParameterSpace> collectLeaves(){
        List<ParameterSpace> list = new ArrayList<>();
        if(activationFunction != null ) list.addAll(activationFunction.collectLeaves());
        if(weightInit != null ) list.addAll(weightInit.collectLeaves());
        if(biasInit != null ) list.addAll(biasInit.collectLeaves());
        if(dist != null ) list.addAll(dist.collectLeaves());
        if(learningRate != null ) list.addAll(learningRate.collectLeaves());
        if(learningRateAfter != null ) list.addAll(learningRateAfter.collectLeaves());
        if(lrScoreBasedDecay != null ) list.addAll(lrScoreBasedDecay.collectLeaves());
        if(l1 != null ) list.addAll(l1.collectLeaves());
        if(l2 != null ) list.addAll(l2.collectLeaves());
        if(dropOut != null ) list.addAll(dropOut.collectLeaves());
        if(momentum != null ) list.addAll(momentum.collectLeaves());
        if(momentumAfter != null ) list.addAll(momentumAfter.collectLeaves());
        if(updater != null ) list.addAll(updater.collectLeaves());
        if(rho != null ) list.addAll(rho.collectLeaves());
        if(rmsDecay != null ) list.addAll(rmsDecay.collectLeaves());
        if(gradientNormalization != null ) list.addAll(gradientNormalization.collectLeaves());
        if(gradientNormalizationThreshold != null ) list.addAll(gradientNormalizationThreshold.collectLeaves());
        return list;
    }

    @Override
    public int numParameters() {
        return numParameters;
    }

    @Override
    public boolean isLeaf() {
        return false;
    }

    @Override
    public void setIndices(int... indices) {
        throw new UnsupportedOperationException("Cannot set indices for non-leaf parameter space");
    }
    

    protected void setLayerOptionsBuilder(Layer.Builder builder, double[] values){
        if(activationFunction != null) builder.activation(activationFunction.getValue(values));
        if(weightInit != null) builder.weightInit(weightInit.getValue(values));
        if(biasInit != null) builder.biasInit(biasInit.getValue(values));
        if(dist != null) builder.dist(dist.getValue(values));
        if(learningRate != null) builder.learningRate(learningRate.getValue(values));
        if(learningRateAfter != null) builder.learningRateAfter(learningRateAfter.getValue(values));
        if(lrScoreBasedDecay != null) builder.learningRateScoreBasedDecayRate(lrScoreBasedDecay.getValue(values));
        if(l1 != null) builder.l1(l1.getValue(values));
        if(l2 != null) builder.l2(l2.getValue(values));
        if(dropOut != null) builder.dropOut(dropOut.getValue(values));
        if(momentum != null) builder.momentum(momentum.getValue(values));
        if(momentumAfter != null) builder.momentumAfter(momentumAfter.getValue(values));
        if(updater != null) builder.updater(updater.getValue(values));
        if(rho != null) builder.rho(rho.getValue(values));
        if(rmsDecay != null) builder.rmsDecay(rmsDecay.getValue(values));
        if(gradientNormalization != null) builder.gradientNormalization(gradientNormalization.getValue(values));
        if(gradientNormalizationThreshold != null) builder.gradientNormalizationThreshold(gradientNormalizationThreshold.getValue(values));
    }


    @Override
    public String toString() {
        return toString(", ");
    }

    protected String toString(String delim){
        StringBuilder sb = new StringBuilder();
        if(activationFunction != null) sb.append("activationFunction: ").append(activationFunction).append(delim);
        if(weightInit != null) sb.append("weightInit: ").append(weightInit).append(delim);
        if(biasInit != null) sb.append("biasInit: ").append(biasInit).append(delim);
        if(dist != null) sb.append("dist: ").append(dist).append(delim);
        if(learningRate != null) sb.append("learningRate: ").append(learningRate).append(delim);
        if(learningRateAfter != null) sb.append("learningRateAfter: ").append(learningRateAfter).append(delim);
        if(lrScoreBasedDecay != null) sb.append("lrScoreBasedDecay: ").append(lrScoreBasedDecay).append(delim);
        if(l1 != null) sb.append("l1: ").append(l1).append(delim);
        if(l2 != null) sb.append("l2: ").append(l2).append(delim);
        if(dropOut != null) sb.append("dropOut: ").append(dropOut).append(delim);
        if(momentum != null) sb.append("momentum: ").append(momentum).append(delim);
        if(momentumAfter != null) sb.append("momentumAfter: ").append(momentumAfter).append(delim);
        if(updater != null) sb.append("updater: ").append(updater).append(delim);
        if(rho != null) sb.append("rho: ").append(rho).append(delim);
        if(rmsDecay != null) sb.append("rmsDecay: ").append(rmsDecay).append(delim);
        if(gradientNormalization != null) sb.append("gradientNormalization: ").append(gradientNormalization).append(delim);
        if(gradientNormalizationThreshold != null) sb.append("gradientNormalizationThreshold").append(gradientNormalizationThreshold);
        String s = sb.toString();

        if(s.endsWith(delim)){
            //Remove final delimiter
            int last = s.lastIndexOf(delim);
            return s.substring(0,last);
        } else return s;
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

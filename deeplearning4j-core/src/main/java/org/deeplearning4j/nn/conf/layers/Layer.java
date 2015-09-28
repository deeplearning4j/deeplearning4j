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

package org.deeplearning4j.nn.conf.layers;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;

import com.fasterxml.jackson.annotation.JsonSubTypes;
import com.fasterxml.jackson.annotation.JsonTypeInfo;
import com.fasterxml.jackson.annotation.JsonTypeInfo.As;
import com.fasterxml.jackson.annotation.JsonTypeInfo.Id;

import lombok.Data;
import lombok.NoArgsConstructor;

import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.Distribution;
import org.deeplearning4j.nn.weights.WeightInit;

/**
 * A neural network layer.
 */
@JsonTypeInfo(use=Id.NAME, include=As.WRAPPER_OBJECT)
@JsonSubTypes(value={
        @JsonSubTypes.Type(value = AutoEncoder.class, name = "autoEncoder"),
        @JsonSubTypes.Type(value = ConvolutionLayer.class, name = "convolution"),
        @JsonSubTypes.Type(value = ImageLSTM.class, name = "imageLSTM"),
        @JsonSubTypes.Type(value = GravesLSTM.class, name = "gravesLSTM"),
        @JsonSubTypes.Type(value = GRU.class, name = "gru"),
        @JsonSubTypes.Type(value = OutputLayer.class, name = "output"),
        @JsonSubTypes.Type(value = RnnOutputLayer.class, name = "rnnoutput"),
        @JsonSubTypes.Type(value = RBM.class, name = "RBM"),
        @JsonSubTypes.Type(value = DenseLayer.class, name = "dense"),
        @JsonSubTypes.Type(value = RecursiveAutoEncoder.class, name = "recursiveAutoEncoder"),
        @JsonSubTypes.Type(value = SubsamplingLayer.class, name = "subsampling"),
        })
@Data
@NoArgsConstructor
public abstract class Layer implements Serializable, Cloneable {
    protected String activationFunction;
    protected WeightInit weightInit;
    protected double biasInit;
    protected Distribution dist;
    protected double learningRate;
    protected double momentum;
    //momentum after n iterations
    protected Map<Integer,Double> momentumAfter;
    protected double l1;
    protected double l2;
    protected double dropOut;
    protected Updater updater;
    //adadelta - weight for how much to consider previous history
    protected double rho;
    protected double rmsDecay;

    public Layer(Builder builder) {
    	this.activationFunction = builder.activationFunction;
    	this.weightInit = builder.weightInit;
        this.biasInit = builder.biasInit;
    	this.dist = builder.dist;
        this.learningRate = builder.learningRate;
        this.momentum = builder.momentum;
        this.momentumAfter = builder.momentumAfter;
        this.l1 = builder.l1;
        this.l2 = builder.l2;
        this.dropOut = builder.dropOut;
        this.updater = builder.updater;
        this.rho = builder.rho;
        this.rmsDecay = builder.rmsDecay;
    }

    @Override
    public Layer clone() {
        try {
            Layer clone = (Layer) super.clone();
            if(clone.dist != null) clone.dist = clone.dist.clone();
            if(clone.momentumAfter != null) clone.momentumAfter = new HashMap<>(clone.momentumAfter);
            return clone;
        } catch (CloneNotSupportedException e) {
            throw new RuntimeException(e);
        }
    }


    public abstract static class Builder<T extends Builder<T>> {
        protected String activationFunction = null;
        protected WeightInit weightInit = null;
        protected double biasInit = Double.NaN;
        protected Distribution dist = null;
        protected double learningRate = Double.NaN;
        protected double momentum = Double.NaN;
        protected Map<Integer,Double> momentumAfter = null;
        protected double l1 = Double.NaN;
        protected double l2 = Double.NaN;
        protected double dropOut = Double.NaN;
        protected Updater updater = Updater.ADAGRAD;
        protected double rho = Double.NaN;
        protected double rmsDecay = Double.NaN;

        public T activation(String activationFunction) {
            this.activationFunction = activationFunction;
            return (T) this;
        }

        public T weightInit(WeightInit weightInit) {
            this.weightInit = weightInit;
            return (T) this;
        }

        public T biasInit(double biasInit) {
            this.biasInit = biasInit;
            return (T) this;
        }

        /** Distribution to sample initial weights from. Used in conjunction with
         * .weightInit(WeightInit.DISTRIBUTION)
         */
        public T dist(Distribution dist){
        	this.dist = dist;
        	return (T) this;
        }

        public T learningRate(double learningRate){
            this.learningRate = learningRate;
            return (T)this;
        }

        public T l1(double l1){
            this.l1 = l1;
            return (T)this;
        }
        public T l2(double l2){
            this.l2 = l2;
            return (T)this;
        }

        public T dropOut(double dropOut) {
            this.dropOut = dropOut;
            return (T) this;
        }


        public T momentum(double momentum) {
            this.momentum = momentum;
        }

        public Builder momentumAfter(Map<Integer, Double> momentumAfter) {
            this.momentumAfter = momentumAfter;
            return this;
        }

        public T updater(Updater updater){
            this.updater = updater;
            return (T) this;
        }

        public Builder rho(double rho) {
            this.rho = rho;
            return this;
        }

        public Builder rmsDecay(double rmsDecay) {
            this.rmsDecay = rmsDecay;
            return this;
        }

        public abstract <E extends Layer> E build();
    }
}

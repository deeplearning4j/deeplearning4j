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

import com.fasterxml.jackson.annotation.JsonSubTypes;
import com.fasterxml.jackson.annotation.JsonTypeInfo;
import com.fasterxml.jackson.annotation.JsonTypeInfo.As;
import com.fasterxml.jackson.annotation.JsonTypeInfo.Id;

import lombok.Data;
import lombok.NoArgsConstructor;

import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.Distribution;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.api.ndarray.INDArray;

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
    protected Distribution dist;
    protected double dropOut;
    protected Updater updater;
    protected double l1;
    protected double l2;
    protected double lr;
    
    public Layer(Builder builder){
    	this.activationFunction = builder.activationFunction;
    	this.weightInit = builder.weightInit;
    	this.dist = builder.dist;
    	this.dropOut = builder.dropOut;
    	this.updater = builder.updater;
        this.l1 = builder.l1;
        this.l2 = builder.l2;
        this.lr = builder.lr;
    }

    @Override
    public Layer clone() {
        try {
            Layer clone = (Layer) super.clone();
            if(clone.dist != null) clone.dist = clone.dist.clone();
            return clone;
        } catch (CloneNotSupportedException e) {
            throw new RuntimeException(e);
        }
    }


    public abstract static class Builder<T extends Builder<T>> {
        protected String activationFunction = "sigmoid";
        protected WeightInit weightInit = WeightInit.VI;
        protected Distribution dist = new NormalDistribution(1e-3,1);
        protected double dropOut = 0;
        protected Updater updater = Updater.ADAGRAD;
        protected double l1 = Double.NaN;
        protected double l2 = Double.NaN;
        protected double lr = Double.NaN;

        public T activation(String activationFunction) {
            this.activationFunction = activationFunction;
            return (T) this;
        }

        public T weightInit(WeightInit weightInit) {
            this.weightInit = weightInit;
            return (T) this;
        }
        
        /** Distribution to sample initial weights from. Used in conjunction with
         * .weightInit(WeightInit.DISTRIBUTION)
         */
        public T dist(Distribution dist){
        	this.dist = dist;
        	return (T) this;
        }

        public T dropOut(double dropOut) {
            this.dropOut = dropOut;
            return (T) this;
        }
        
        public T updater(Updater updater){
        	this.updater = updater;
        	return (T) this;
        }

        public T l1(double l1){
            this.l1 = l1;
            return (T)this;
        }
        public T l2(double l2){
            this.l2 = l2;
            return (T)this;
        }

        public T learningRate(double lr){
            this.lr = lr;
            return (T)this;
        }

        public abstract <E extends Layer> E build();
    }
}

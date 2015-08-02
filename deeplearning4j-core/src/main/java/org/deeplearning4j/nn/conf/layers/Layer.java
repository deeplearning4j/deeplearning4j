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

import org.deeplearning4j.nn.conf.distribution.Distribution;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.weights.WeightInit;

/**
 * A neural network layer.
 */
@JsonTypeInfo(use=Id.NAME, include=As.WRAPPER_OBJECT)
@JsonSubTypes(value={
        @JsonSubTypes.Type(value = AutoEncoder.class, name = "autoEncoder"),
        @JsonSubTypes.Type(value = ConvolutionDownSampleLayer.class, name = "convolutionDownSample"),
        @JsonSubTypes.Type(value = LSTM.class, name = "LSTM"),
        @JsonSubTypes.Type(value = OutputLayer.class, name = "output"),
        @JsonSubTypes.Type(value = RBM.class, name = "RBM"),
        @JsonSubTypes.Type(value = RecursiveAutoEncoder.class, name = "recursiveAutoEncoder"),
        })
@Data
@NoArgsConstructor
public abstract class Layer implements Serializable {
    private static final long serialVersionUID = 492217000569721428L;
    protected String activationFunction;
    protected WeightInit weightInit;
    protected Distribution dist;
    protected double dropOut;
    
    public Layer(Builder builder){
    	this.activationFunction = builder.activationFunction;
    	this.weightInit = builder.weightInit;
    	this.dist = builder.dist;
    	this.dropOut = builder.dropOut;
    }

    public abstract static class Builder {
        protected String activationFunction;
        protected WeightInit weightInit;
        protected Distribution dist;
        protected double dropOut = Double.NaN;	//Use in place of null = "not set" for primitives

        public Builder activation(String activationFunction) {
            this.activationFunction = activationFunction;
            return this;
        }

        public Builder weightInit(WeightInit weightInit) {
            this.weightInit = weightInit;
            return this;
        }
        
        /** Distribution to sample initial weights from. Used in conjunction with
         * .weightInit(WeightInit.DISTRIBUTION)
         */
        public Builder dist(Distribution dist){
        	this.dist = dist;
        	return this;
        }

        public Builder dropOut(double dropOut) {
            this.dropOut = dropOut;
            return this;
        }

        public abstract <E extends Layer> E build();
    }
}

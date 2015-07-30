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

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import org.deeplearning4j.nn.conf.distribution.Distribution;
import org.deeplearning4j.nn.weights.WeightInit;

/**
 *  Autoencoder.
 * Add Gaussian noise to input and learn
 * a reconstruction function.
 *
 */
@Data @NoArgsConstructor
public class AutoEncoder extends BasePretrainNetwork {
    private static final long serialVersionUID = -7624965662728637504L;
    protected double corruptionLevel;
    protected double sparsity;

    // Builder
    private AutoEncoder(Builder builder) {
    	super(builder);
        this.corruptionLevel = builder.corruptionLevel;
        this.sparsity = builder.sparsity;
    }

    @AllArgsConstructor
    public static class Builder extends FeedForwardLayer.Builder {
        private double corruptionLevel = Double.NaN;
        private double sparsity = Double.NaN;

        public Builder(double corruptionLevel) {
            this.corruptionLevel = corruptionLevel;
        }
        
        public Builder corruptionLevel(double corruptionLevel){
        	this.corruptionLevel = corruptionLevel;
        	return this;
        }
        
        public Builder sparsity(double sparsity){
        	this.sparsity = sparsity;
        	return this;
        }

        @Override
        public Builder nIn(int nIn) {
            this.nIn = nIn;
            return this;
        }

        @Override
        public Builder nOut(int nOut) {
            this.nOut = nOut;
            return this;
        }

        @Override
        public Builder activation(String activationFunction) {
            this.activationFunction = activationFunction;
            return this;
        }

        @Override
        public Builder weightInit(WeightInit weightInit) {
            this.weightInit = weightInit;
            return this;
        }

        @Override
        public Builder dropOut(double dropOut) {
            this.dropOut = dropOut;
            return this;
        }

        @Override
        @SuppressWarnings("unchecked")
        public AutoEncoder build() {
            return new AutoEncoder(this);
        }
        
        @Override
        public Builder dist(Distribution dist){
        	this.dist = dist;
        	return this;
        }
    }
}

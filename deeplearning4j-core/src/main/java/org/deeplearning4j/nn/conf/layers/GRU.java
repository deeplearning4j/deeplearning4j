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

import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.Distribution;
import org.deeplearning4j.nn.weights.WeightInit;

/** Gated Recurrent Unit RNN Layer.<br>
 * The GRU was recently proposed by Cho et al. 2014 - http://arxiv.org/abs/1406.1078<br>
 * It is similar to the LSTM architecture in that both use a gating structure within each unit
 * to attempt to capture long-term dependencies and deal with the vanishing gradient problem.
 * A GRU layer contains fewer parameters than an equivalent size LSTM layer, and some research
 * (such as http://arxiv.org/abs/1412.3555) suggests it may outperform LSTM layers (given an
 * equal number of parameters) in some cases.
 * @author Alex Black
 */
@Data @NoArgsConstructor
public class GRU extends FeedForwardLayer {

    private GRU(Builder builder) {
    	super(builder);
    }

    @AllArgsConstructor
    public static class Builder extends FeedForwardLayer.Builder {

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
        public Builder dist(Distribution dist){
        	super.dist(dist);
        	return this;
        }
        
        @Override
        public Builder dropOut(double dropOut) {
            this.dropOut = dropOut;
            return this;
        }
        
        @Override
        public Builder updater(Updater updater){
        	this.updater = updater;
        	return this;
        }
        
        @Override
        @SuppressWarnings("unchecked")
        public GRU build() {
            return new GRU(this);
        }
    }

}

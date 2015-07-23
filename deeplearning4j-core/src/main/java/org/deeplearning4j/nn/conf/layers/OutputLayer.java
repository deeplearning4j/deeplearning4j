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
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

/**
 * Output layer with different objective co-occurrences for different objectives.
 * This includes classification as well as prediction
 *
 */
@Data
@NoArgsConstructor
public class OutputLayer extends FeedForwardLayer {

    private static final long serialVersionUID = 8554480736972510788L;
    protected LossFunction lossFunction;

    private OutputLayer(Builder builder) {
        this.lossFunction = builder.lossFunction;
        this.nIn = builder.nIn;
        this.nOut = builder.nOut;
        this.activationFunction = builder.activationFunction;
        this.weightInit = builder.weightInit;
        this.dropOut = builder.dropOut;
    }

    @AllArgsConstructor
    public static class Builder extends FeedForwardLayer.Builder {
        private LossFunction lossFunction;

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
        public OutputLayer build() {
            return new OutputLayer(this);
        }
    }
}


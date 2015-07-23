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

/**
 * Restricted Boltzmann Machine.
 *
 * Markov chain with gibbs sampling.
 *
 * Supports the following visible units:
 *
 *     binary
 *     gaussian
 *     softmax
 *     linear
 *
 * Supports the following hidden units:
 *     rectified
 *     binary
 *     gaussian
 *     softmax
 *     linear
 *
 * Based on Hinton et al.'s work
 *
 * Great reference:
 * http://www.iro.umontreal.ca/~lisa/publications2/index.php/publications/show/239
 *
 */
@Data
@NoArgsConstructor
public class RBM extends BasePretrainNetwork {

    private static final long serialVersionUID = 485040309806445606L;
    protected int ind;
    protected HiddenUnit hiddenUnit;
    protected VisibleUnit visibleUnit;
    protected int k;

    public enum VisibleUnit {
        BINARY, GAUSSIAN, SOFTMAX, LINEAR
    }
    public enum HiddenUnit {
        RECTIFIED, BINARY, GAUSSIAN, SOFTMAX
    }

    // Builder
    private RBM(Builder builder) {
        this.hiddenUnit = builder.hiddenUnit;
        this.visibleUnit = builder.visibleUnit;
        this.k = builder.k;
        this.nIn = builder.nIn;
        this.nOut = builder.nOut;
        this.weightInit = builder.weightInit;
        this.dropOut = builder.dropOut;
    }

    @AllArgsConstructor @NoArgsConstructor
    public static class Builder extends FeedForwardLayer.Builder {
        private HiddenUnit hiddenUnit;
        private VisibleUnit visibleUnit;
        private int k;

        public Builder(HiddenUnit hiddenUnit, VisibleUnit visibleUnit) {
            this.hiddenUnit = hiddenUnit;
            this.visibleUnit = visibleUnit;
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
            throw new UnsupportedOperationException("RBM does not accept activation");
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
        public RBM build() {
            return new RBM(this);
        }
    }
}
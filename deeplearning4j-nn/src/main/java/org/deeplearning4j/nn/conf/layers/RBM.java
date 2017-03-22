/*-
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

import lombok.*;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.params.PretrainParamInitializer;
import org.deeplearning4j.optimize.api.IterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Collection;
import java.util.Map;

/**
 * Restricted Boltzmann Machine.
 *
 * Markov chain with gibbs sampling.
 *
 * Supports the following visible units:
 *     BINARY
 *     GAUSSIAN
 *     SOFTMAX
 *     LINEAR
 *
 * Supports the following hidden units:
 *     RECTIFIED
 *     BINARY
 *     GAUSSIAN
 *     SOFTMAX
 *
 * Based on Hinton et al.'s work
 *
 * Great reference:
 * http://www.iro.umontreal.ca/~lisa/publications2/index.php/publications/show/239
 *
 */

@Data
@NoArgsConstructor
@ToString(callSuper = true)
@EqualsAndHashCode(callSuper = true)
public class RBM extends BasePretrainNetwork {
    protected HiddenUnit hiddenUnit;
    protected VisibleUnit visibleUnit;
    protected int k; // gibbs sampling steps standard is 1 and includes propup and propdown
    protected double sparsity;

    @Override
    public Layer instantiate(NeuralNetConfiguration conf, Collection<IterationListener> iterationListeners,
                    int layerIndex, INDArray layerParamsView, boolean initializeParams) {
        org.deeplearning4j.nn.layers.feedforward.rbm.RBM ret =
                        new org.deeplearning4j.nn.layers.feedforward.rbm.RBM(conf);
        ret.setListeners(iterationListeners);
        ret.setIndex(layerIndex);
        ret.setParamsViewArray(layerParamsView);
        Map<String, INDArray> paramTable = initializer().init(conf, layerParamsView, initializeParams);
        ret.setParamTable(paramTable);
        ret.setConf(conf);
        return ret;
    }

    @Override
    public ParamInitializer initializer() {
        return PretrainParamInitializer.getInstance();
    }

    public enum VisibleUnit {
        BINARY, GAUSSIAN, SOFTMAX, LINEAR, IDENTITY
    }
    public enum HiddenUnit {
        RECTIFIED, BINARY, GAUSSIAN, SOFTMAX, IDENTITY
    }

    private RBM(Builder builder) {
        super(builder);
        this.hiddenUnit = builder.hiddenUnit;
        this.visibleUnit = builder.visibleUnit;
        this.k = builder.k;
        this.sparsity = builder.sparsity;
    }

    @AllArgsConstructor
    public static class Builder extends BasePretrainNetwork.Builder<Builder> {
        private HiddenUnit hiddenUnit = HiddenUnit.BINARY;
        private VisibleUnit visibleUnit = VisibleUnit.BINARY;
        private int k = 1;
        private double sparsity = 0f;

        public Builder(HiddenUnit hiddenUnit, VisibleUnit visibleUnit) {
            this.hiddenUnit = hiddenUnit;
            this.visibleUnit = visibleUnit;
        }

        public Builder() {}

        @Override
        @SuppressWarnings("unchecked")
        public RBM build() {
            return new RBM(this);
        }

        // convergence iterations
        public Builder k(int k) {
            this.k = k;
            return this;
        }

        public Builder hiddenUnit(HiddenUnit hiddenUnit) {
            this.hiddenUnit = hiddenUnit;
            return this;
        }

        public Builder visibleUnit(VisibleUnit visibleUnit) {
            this.visibleUnit = visibleUnit;
            return this;
        }

        public Builder sparsity(double sparsity) {
            this.sparsity = sparsity;
            return this;
        }


    }
}

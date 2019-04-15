/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.nn.conf.layers;

import lombok.*;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.nn.conf.memory.MemoryReport;
import org.deeplearning4j.nn.params.EmptyParamInitializer;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.regularization.Regularization;

import java.util.Collection;
import java.util.List;
import java.util.Map;

/**
 * Local response normalization layer<br> See section 3.3 of <a href="http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf">http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf</a>
 */
@Data
@NoArgsConstructor
@ToString(callSuper = true)
@EqualsAndHashCode(callSuper = true)
public class LocalResponseNormalization extends Layer {

    // Defaults as per http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf
    //Set defaults here as well as in builder, in case users use no-arg constructor instead of builder
    protected double n = 5; // # adjacent kernal maps
    protected double k = 2; // constant (e.g. scale)
    protected double beta = 0.75; // decay rate
    protected double alpha = 1e-4; // decay rate
    protected boolean cudnnAllowFallback = true;

    private LocalResponseNormalization(Builder builder) {
        super(builder);
        this.k = builder.k;
        this.n = builder.n;
        this.alpha = builder.alpha;
        this.beta = builder.beta;
        this.cudnnAllowFallback = builder.cudnnAllowFallback;
    }

    @Override
    public LocalResponseNormalization clone() {
        LocalResponseNormalization clone = (LocalResponseNormalization) super.clone();
        return clone;
    }

    @Override
    public org.deeplearning4j.nn.api.Layer instantiate(NeuralNetConfiguration conf,
                                                       Collection<TrainingListener> trainingListeners, int layerIndex, INDArray layerParamsView,
                                                       boolean initializeParams, DataType networkDataType) {
        org.deeplearning4j.nn.layers.normalization.LocalResponseNormalization ret =
                        new org.deeplearning4j.nn.layers.normalization.LocalResponseNormalization(conf, networkDataType);
        ret.setListeners(trainingListeners);
        ret.setIndex(layerIndex);
        ret.setParamsViewArray(layerParamsView);
        Map<String, INDArray> paramTable = initializer().init(conf, layerParamsView, initializeParams);
        ret.setParamTable(paramTable);
        ret.setConf(conf);
        return ret;
    }

    @Override
    public ParamInitializer initializer() {
        return EmptyParamInitializer.getInstance();
    }


    @Override
    public InputType getOutputType(int layerIndex, InputType inputType) {
        if (inputType == null || inputType.getType() != InputType.Type.CNN) {
            throw new IllegalStateException(
                            "Invalid input type for LRN layer (layer index = " + layerIndex + ", layer name = \""
                                            + getLayerName() + "\"): Expected input of type CNN, got " + inputType);
        }
        return inputType;
    }

    @Override
    public void setNIn(InputType inputType, boolean override) {
        //No op
    }

    @Override
    public InputPreProcessor getPreProcessorForInputType(InputType inputType) {
        if (inputType == null) {
            throw new IllegalStateException(
                            "Invalid input type for LRN layer (layer name = \"" + getLayerName() + "\"): null");
        }

        return InputTypeUtil.getPreProcessorForInputTypeCnnLayers(inputType, getLayerName());
    }

    @Override
    public List<Regularization> getRegularizationByParam(String paramName) {
        return null;
    }

    @Override
    public boolean isPretrainParam(String paramName) {
        return false; //No params in LRN
    }

    @Override
    public GradientNormalization getGradientNormalization() {
        return GradientNormalization.None;
    }

    @Override
    public double getGradientNormalizationThreshold() {
        return 0;
    }

    @Override
    public LayerMemoryReport getMemoryReport(InputType inputType) {
        val actElementsPerEx = inputType.arrayElementsPerExample();

        //Forward pass: 3x input size as working memory, in addition to output activations
        //Backward pass: 2x input size as working memory, in addition to epsilons

        return new LayerMemoryReport.Builder(layerName, DenseLayer.class, inputType, inputType).standardMemory(0, 0)
                        .workingMemory(0, 2 * actElementsPerEx, 0, 3 * actElementsPerEx)
                        .cacheMemory(MemoryReport.CACHE_MODE_ALL_ZEROS, MemoryReport.CACHE_MODE_ALL_ZEROS) //No caching in DenseLayer
                        .build();
    }

    @AllArgsConstructor
    @Getter
    @Setter
    public static class Builder extends Layer.Builder<Builder> {

        // defaults based on AlexNet model

        /**
         * LRN scaling constant k. Default: 2
         *
         */
        private double k = 2;

        /**
         * Number of adjacent kernel maps to use when doing LRN. default: 5
         *
         */
        private double n = 5;

        /**
         * LRN scaling constant alpha. Default: 1e-4
         *
         */
        private double alpha = 1e-4;

        /**
         * Scaling constant beta. Default: 0.75
         *
         */
        private double beta = 0.75;

        /**
         * When using CuDNN and an error is encountered, should fallback to the non-CuDNN implementatation be allowed?
         * If set to false, an exception in CuDNN will be propagated back to the user. If false, the built-in
         * (non-CuDNN) implementation for BatchNormalization will be used
         *
         */
        protected boolean cudnnAllowFallback = true;

        public Builder(double k, double n, double alpha, double beta) {
            this(k, n, alpha, beta, true);
        }

        public Builder(double k, double alpha, double beta) {
            this.setK(k);
            this.setAlpha(alpha);
            this.setBeta(beta);
        }

        public Builder() {}

        /**
         * LRN scaling constant k. Default: 2
         *
         * @param k Scaling constant
         */
        public Builder k(double k) {
            this.setK(k);
            return this;
        }

        /**
         * Number of adjacent kernel maps to use when doing LRN. default: 5
         *
         * @param n Number of adjacent kernel maps
         */
        public Builder n(double n) {
            this.setN(n);
            return this;
        }

        /**
         * LRN scaling constant alpha. Default: 1e-4
         *
         * @param alpha Scaling constant
         */
        public Builder alpha(double alpha) {
            this.setAlpha(alpha);
            return this;
        }

        /**
         * Scaling constant beta. Default: 0.75
         *
         * @param beta Scaling constant
         */
        public Builder beta(double beta) {
            this.setBeta(beta);
            return this;
        }

        /**
         * When using CuDNN and an error is encountered, should fallback to the non-CuDNN implementatation be allowed?
         * If set to false, an exception in CuDNN will be propagated back to the user. If false, the built-in
         * (non-CuDNN) implementation for BatchNormalization will be used
         *
         * @param allowFallback Whether fallback to non-CuDNN implementation should be used
         */
        public Builder cudnnAllowFallback(boolean allowFallback) {
            this.setCudnnAllowFallback(allowFallback);
            return this;
        }

        @Override
        public LocalResponseNormalization build() {
            return new LocalResponseNormalization(this);
        }
    }

}

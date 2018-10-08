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
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Collection;
import java.util.Map;

/**
 * Local response normalization layer<br>
 * See section 3.3 of <a href="http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf">http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf</a>
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

    private LocalResponseNormalization(Builder builder) {
        super(builder);
        this.k = builder.k;
        this.n = builder.n;
        this.alpha = builder.alpha;
        this.beta = builder.beta;
    }

    @Override
    public LocalResponseNormalization clone() {
        LocalResponseNormalization clone = (LocalResponseNormalization) super.clone();
        return clone;
    }

    @Override
    public org.deeplearning4j.nn.api.Layer instantiate(NeuralNetConfiguration conf,
                    Collection<TrainingListener> trainingListeners, int layerIndex, INDArray layerParamsView,
                    boolean initializeParams) {
        org.deeplearning4j.nn.layers.normalization.LocalResponseNormalization ret =
                        new org.deeplearning4j.nn.layers.normalization.LocalResponseNormalization(conf);
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
    public boolean isPretrain() {
        return false;
    }

    @Override
    public double getL1ByParam(String paramName) {
        //Not applicable
        return 0;
    }

    @Override
    public double getL2ByParam(String paramName) {
        //Not applicable
        return 0;
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
    public static class Builder extends Layer.Builder<Builder> {
        // defaults based on AlexNet model
        private double k = 2;
        private double n = 5;
        private double alpha = 1e-4;
        private double beta = 0.75;

        public Builder(double k, double alpha, double beta) {
            this.k = k;
            this.alpha = alpha;
            this.beta = beta;
        }

        public Builder() {}

        /**
         * LRN scaling constant k. Default: 2
         *
         * @param k Scaling constant
         */
        public Builder k(double k) {
            this.k = k;
            return this;
        }

        /**
         * Number of adjacent kernel maps to use when doing LRN. default: 5
         *
         * @param n    Number of adjacent kernel maps
         */
        public Builder n(double n) {
            this.n = n;
            return this;
        }

        /**
         * LRN scaling constant alpha. Default: 1e-4
         *
         * @param alpha    Scaling constant
         */
        public Builder alpha(double alpha) {
            this.alpha = alpha;
            return this;
        }

        /**
         * Scaling constant beta. Default: 0.75
         *
         * @param beta    Scaling constant
         */
        public Builder beta(double beta) {
            this.beta = beta;
            return this;
        }

        @Override
        public LocalResponseNormalization build() {
            return new LocalResponseNormalization(this);
        }
    }

}

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
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.nn.conf.memory.MemoryReport;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.nn.params.PReLUParamInitializer;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Collection;
import java.util.Map;

/**
 * Parametrized Rectified Linear Unit (PReLU)
 * <p>
 * {@code f(x) = alpha * x for x < 0, f(x) = x for x >= 0}
 * <p>
 * alpha has the same shape as x and is a learned parameter.
 *
 * @author Max Pumperla
 */
@Data
@NoArgsConstructor
@ToString(callSuper = true)
@EqualsAndHashCode(callSuper = true)
public class PReLULayer extends BaseLayer {

    private long[] inputShape = null;
    private long[] sharedAxes = null;

    private int nIn;
    private int nOut;

    private PReLULayer(Builder builder) {
        super(builder);
        this.inputShape = builder.inputShape;
        this.sharedAxes = builder.sharedAxes;
        initializeConstraints(builder);
    }

    @Override
    public Layer instantiate(NeuralNetConfiguration conf, Collection<TrainingListener> trainingListeners,
                    int layerIndex, INDArray layerParamsView, boolean initializeParams) {
        org.deeplearning4j.nn.layers.feedforward.PReLU ret =
                        new org.deeplearning4j.nn.layers.feedforward.PReLU(conf);
        ret.setListeners(trainingListeners);
        ret.setIndex(layerIndex);
        ret.setParamsViewArray(layerParamsView);
        Map<String, INDArray> paramTable = initializer().init(conf, layerParamsView, initializeParams);
        ret.setParamTable(paramTable);
        ret.setConf(conf);
        return ret;
    }

    @Override
    public InputType getOutputType(int layerIndex, InputType inputType) {
        if (inputType == null)
            throw new IllegalStateException("Invalid input type: null for layer name \"" + getLayerName() + "\"");
        return inputType;
    }

    @Override
    public void setNIn(InputType inputType, boolean override) {
        // not needed
    }

    @Override
    public InputPreProcessor getPreProcessorForInputType(InputType inputType) {
        // None needed
        return null;
    }

    @Override
    public boolean isPretrain() {
        return false;
    }

    @Override
    public double getL1ByParam(String paramName) {
        switch (paramName) {
            case DefaultParamInitializer.WEIGHT_KEY:
                return l1;
            default:
                throw new IllegalStateException("Unknown parameter: \"" + paramName + "\"");
        }
    }

    @Override
    public double getL2ByParam(String paramName) {
        switch (paramName) {
            case DefaultParamInitializer.WEIGHT_KEY:
                return l2;
            default:
                throw new IllegalStateException("Unknown parameter: \"" + paramName + "\"");
        }
    }

    @Override
    public boolean isPretrainParam(String paramName) {
        return false;
    }

    @Override
    public ParamInitializer initializer() {
        return PReLUParamInitializer.getInstance(inputShape, sharedAxes);
    }

    @Override
    public LayerMemoryReport getMemoryReport(InputType inputType) {
        InputType outputType = getOutputType(-1, inputType);

        val numParams = initializer().numParams(this);
        val updaterStateSize = (int) getIUpdater().stateSize(numParams);

        return new LayerMemoryReport.Builder(layerName, PReLULayer.class, inputType, outputType)
                        .standardMemory(numParams, updaterStateSize)
                        .workingMemory(0, 0, 0, 0)
                        .cacheMemory(MemoryReport.CACHE_MODE_ALL_ZEROS, MemoryReport.CACHE_MODE_ALL_ZEROS)
                        .build();
    }

    @NoArgsConstructor
    public static class Builder extends FeedForwardLayer.Builder<PReLULayer.Builder> {

        private long[] inputShape = null;
        private long[] sharedAxes = null;

        /**
         * Explicitly set input shape of incoming activations so that parameters
         * can be initialized properly. This explicitly excludes the mini-batch
         * dimension.
         *
         * @param shape shape of input data
         * @return
         */
        public Builder inputShape(long... shape){
            this.inputShape = shape;
            return this;
        }

        /**
         * Set the broadcasting axes of PReLU's alpha parameter.
         *
         * For instance, given input data of shape
         * [mb, channels, height, width], setting axes to [2,3] will
         * set alpha to shape [channels, 1, 1] and broadcast alpha across
         * height and width dimensions of each channel.
         *
         * @param axes shared/broadcasting axes
         * @return Builder
         */
        public Builder sharedAxes(long... axes) {
            this.sharedAxes = axes;
            return this;
        }

        @Override
        @SuppressWarnings("unchecked")
        public PReLULayer build() {
            return new PReLULayer(this);
        }
    }

}

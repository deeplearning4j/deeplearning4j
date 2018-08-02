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

import com.google.common.base.Preconditions;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.nn.conf.memory.MemoryReport;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.util.ConvolutionUtils;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Arrays;
import java.util.Collection;
import java.util.Map;

/**
 * Zero padding layer for convolutional neural networks.
 * Allows padding to be done separately for top/bottom/left/right
 *
 * @author Alex Black
 */
@Data
@NoArgsConstructor
@EqualsAndHashCode(callSuper = true)
public class ZeroPaddingLayer extends NoParamLayer {

    private int[] padding;

    public ZeroPaddingLayer(int padTopBottom, int padLeftRight){
        this(new Builder(padTopBottom, padLeftRight));
    }

    public ZeroPaddingLayer(int padTop, int padBottom, int padLeft, int padRight){
        this(new Builder(padTop, padBottom, padLeft, padRight));
    }

    private ZeroPaddingLayer(Builder builder) {
        super(builder);
        if(builder.padding == null || builder.padding.length != 4){
            throw new IllegalArgumentException("Invalid padding values: must have exactly 4 values [top, bottom, left, right]." +
                    " Got: " + (builder.padding == null ? null : Arrays.toString(builder.padding)));
        }

        this.padding = builder.padding;
    }

    @Override
    public org.deeplearning4j.nn.api.Layer instantiate(NeuralNetConfiguration conf,
                    Collection<TrainingListener> trainingListeners, int layerIndex, INDArray layerParamsView,
                    boolean initializeParams) {
        org.deeplearning4j.nn.layers.convolution.ZeroPaddingLayer ret =
                        new org.deeplearning4j.nn.layers.convolution.ZeroPaddingLayer(conf);
        ret.setListeners(trainingListeners);
        ret.setIndex(layerIndex);
        Map<String, INDArray> paramTable = initializer().init(conf, layerParamsView, initializeParams);
        ret.setParamTable(paramTable);
        ret.setConf(conf);
        return ret;
    }

    @Override
    public InputType getOutputType(int layerIndex, InputType inputType) {
        int[] hwd = ConvolutionUtils.getHWDFromInputType(inputType);
        int outH = hwd[0] + padding[0] + padding[1];
        int outW = hwd[1] + padding[2] + padding[3];

        return InputType.convolutional(outH, outW, hwd[2]);
    }

    @Override
    public InputPreProcessor getPreProcessorForInputType(InputType inputType) {
        Preconditions.checkArgument(inputType != null, "Invalid input for ZeroPaddingLayer layer (layer name=\""
                + getLayerName() + "\"): InputType is null");
        return InputTypeUtil.getPreProcessorForInputTypeCnnLayers(inputType, getLayerName());
    }

    @Override
    public LayerMemoryReport getMemoryReport(InputType inputType) {
        InputType outputType = getOutputType(-1, inputType);

        return new LayerMemoryReport.Builder(layerName, ZeroPaddingLayer.class, inputType, outputType)
                        .standardMemory(0, 0) //No params
                        //Inference and training is same - just output activations, no working memory in addition to that
                        .workingMemory(0, 0, MemoryReport.CACHE_MODE_ALL_ZEROS, MemoryReport.CACHE_MODE_ALL_ZEROS)
                        .cacheMemory(MemoryReport.CACHE_MODE_ALL_ZEROS, MemoryReport.CACHE_MODE_ALL_ZEROS) //No caching
                        .build();
    }

    public static class Builder extends Layer.Builder<Builder> {

        private int[] padding = new int[] {0, 0, 0, 0}; //Padding: top, bottom, left, right

        /**
         *
         * @param padHeight Padding for both the top and bottom
         * @param padWidth  Padding for both the left and right
         */
        public Builder(int padHeight, int padWidth) {
            this(padHeight, padHeight, padWidth, padWidth);
        }

        public Builder(int padTop, int padBottom, int padLeft, int padRight) {
            this(new int[] {padTop, padBottom, padLeft, padRight});
        }

        public Builder(int[] padding) {
            if(padding.length == 2){
                padding = new int[]{padding[0], padding[0], padding[1], padding[1]};
            } else if(padding.length != 4){
                throw new IllegalArgumentException("Padding must have exactly 2 or 4 values - got " + Arrays.toString(padding));
            }
            this.padding = padding;
        }

        @Override
        @SuppressWarnings("unchecked")
        public ZeroPaddingLayer build() {
            for (int p : padding) {
                if (p < 0) {
                    throw new IllegalStateException(
                                    "Invalid zero padding layer config: padding [top, bottom, left, right]"
                                                    + " must be > 0 for all elements. Got: "
                                                    + Arrays.toString(padding));
                }
            }

            return new ZeroPaddingLayer(this);
        }
    }
}

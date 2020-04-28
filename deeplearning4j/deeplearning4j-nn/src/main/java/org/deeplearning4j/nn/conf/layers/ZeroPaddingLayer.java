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
import org.deeplearning4j.nn.conf.CNN2DFormat;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.nn.conf.memory.MemoryReport;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.util.ConvolutionUtils;
import org.deeplearning4j.util.ValidationUtils;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Arrays;
import java.util.Collection;
import java.util.Map;

/**
 * Zero padding layer for convolutional neural networks (2D CNNs). Allows padding to be done separately for
 * top/bottom/left/right
 *
 * @author Alex Black
 */
@Data
@NoArgsConstructor
@EqualsAndHashCode(callSuper = true)
public class ZeroPaddingLayer extends NoParamLayer {

    private int[] padding;
    private CNN2DFormat dataFormat = CNN2DFormat.NCHW;

    public ZeroPaddingLayer(int padTopBottom, int padLeftRight) {
        this(new Builder(padTopBottom, padLeftRight));
    }

    public ZeroPaddingLayer(int padTop, int padBottom, int padLeft, int padRight) {
        this(new Builder(padTop, padBottom, padLeft, padRight));
    }

    private ZeroPaddingLayer(Builder builder) {
        super(builder);
        if (builder.padding == null || builder.padding.length != 4) {
            throw new IllegalArgumentException(
                            "Invalid padding values: must have exactly 4 values [top, bottom, left, right]." + " Got: "
                                            + (builder.padding == null ? null : Arrays.toString(builder.padding)));
        }

        this.padding = builder.padding;
        this.dataFormat = builder.cnn2DFormat;
    }

    @Override
    public org.deeplearning4j.nn.api.Layer instantiate(NeuralNetConfiguration conf,
                                                       Collection<TrainingListener> trainingListeners, int layerIndex, INDArray layerParamsView,
                                                       boolean initializeParams, DataType networkDataType) {
        org.deeplearning4j.nn.layers.convolution.ZeroPaddingLayer ret =
                        new org.deeplearning4j.nn.layers.convolution.ZeroPaddingLayer(conf, networkDataType);
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

        InputType.InputTypeConvolutional c = (InputType.InputTypeConvolutional)inputType;

        return InputType.convolutional(outH, outW, hwd[2], c.getFormat());
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

    @Override
    public void setNIn(InputType inputType, boolean override) {
        InputType.InputTypeConvolutional c = (InputType.InputTypeConvolutional)inputType;
        this.dataFormat = c.getFormat();
    }

    @Getter
    @Setter
    public static class Builder extends Layer.Builder<Builder> {

        /**
         * Padding value for top, bottom, left, and right. Must be length 4 array
         */
        @Setter(AccessLevel.NONE)
        private int[] padding = new int[] {0, 0, 0, 0}; //Padding: top, bottom, left, right

        private CNN2DFormat cnn2DFormat = CNN2DFormat.NCHW;

        /**
         * Set the data format for the CNN activations - NCHW (channels first) or NHWC (channels last).
         * See {@link CNN2DFormat} for more details.<br>
         * Default: NCHW
         * @param format Format for activations (in and out)
         */
        public Builder dataFormat(CNN2DFormat format){
            this.cnn2DFormat = format;
            return this;
        }

        /**
         * @param padding Padding value for top, bottom, left, and right. Must be length 4 array
         */
        public void setPadding(int... padding) {
            this.padding = ValidationUtils.validate4NonNegative(padding, "padding");
        }

        /**
         * @param padHeight Padding for both the top and bottom
         * @param padWidth Padding for both the left and right
         */
        public Builder(int padHeight, int padWidth) {
            this(padHeight, padHeight, padWidth, padWidth);
        }

        /**
         * @param padTop Top padding value
         * @param padBottom Bottom padding value
         * @param padLeft Left padding value
         * @param padRight Right padding value
         */
        public Builder(int padTop, int padBottom, int padLeft, int padRight) {
            this(new int[] {padTop, padBottom, padLeft, padRight});
        }

        /**
         * @param padding Must be a length 1 array with values [paddingAll], a length 2 array with values
         * [padTopBottom, padLeftRight], or a length 4 array with
         * values [padTop, padBottom, padLeft, padRight]
         */
        public Builder(int[] padding) {
            this.setPadding(padding);
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

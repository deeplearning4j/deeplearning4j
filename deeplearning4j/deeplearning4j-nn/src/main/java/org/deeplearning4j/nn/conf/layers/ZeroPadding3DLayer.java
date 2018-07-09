/*-
 *
 *  * Copyright 2017 Skymind,Inc.
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

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.nn.conf.memory.MemoryReport;
import org.deeplearning4j.nn.params.EmptyParamInitializer;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Arrays;
import java.util.Collection;
import java.util.Map;

/**
 * Zero padding 3D layer for convolutional neural networks.
 * Allows padding to be done separately for "left" and "right"
 * in all three spatial dimensions.
 *
 * @author Max Pumperla
 */
@Data
@NoArgsConstructor
@EqualsAndHashCode(callSuper = true)
public class ZeroPadding3DLayer extends NoParamLayer {

    private int[] padding; // [padLeftD, padRightD, padLeftH, padRightH, padLeftW, padRightW]

    private ZeroPadding3DLayer(Builder builder) {
        super(builder);
        this.padding = builder.padding;
    }

    @Override
    public org.deeplearning4j.nn.api.Layer instantiate(NeuralNetConfiguration conf,
                                                       Collection<TrainingListener> iterationListeners,
                                                       int layerIndex, INDArray layerParamsView,
                                                       boolean initializeParams) {
        org.deeplearning4j.nn.layers.convolution.ZeroPadding3DLayer ret =
                new org.deeplearning4j.nn.layers.convolution.ZeroPadding3DLayer(conf);
        ret.setListeners(iterationListeners);
        ret.setIndex(layerIndex);
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
        if (inputType == null || inputType.getType() != InputType.Type.CNN3D) {
            throw new IllegalStateException("Invalid input for 3D CNN layer (layer index = " + layerIndex
                    + ", layer name = \"" + getLayerName() + "\"): expect CNN3D input type with size > 0. Got: "
                    + inputType);
        }
        InputType.InputTypeConvolutional3D c = (InputType.InputTypeConvolutional3D) inputType;
        return InputType.convolutional3D(
                c.getDepth() + padding[0] + padding[1],
                c.getHeight() + padding[2] + padding[3],
                c.getWidth() + padding[4] + padding[5],
                c.getChannels());
    }

    @Override
    public void setNIn(InputType inputType, boolean override) {
        //No op
    }

    @Override
    public InputPreProcessor getPreProcessorForInputType(InputType inputType) {
        if (inputType == null) {
            throw new IllegalStateException("Invalid input for ZeroPadding3DLayer layer (layer name=\"" + getLayerName()
                    + "\"): input is null");
        }

        return InputTypeUtil.getPreProcessorForInputTypeCnn3DLayers(inputType, getLayerName());
    }

    @Override
    public double getL1ByParam(String paramName) {
        return 0;
    }

    @Override
    public double getL2ByParam(String paramName) {
        return 0;
    }

    @Override
    public boolean isPretrainParam(String paramName) {
        throw new UnsupportedOperationException("ZeroPadding3DLayer does not contain parameters");
    }

    @Override
    public LayerMemoryReport getMemoryReport(InputType inputType) {
        InputType outputType = getOutputType(-1, inputType);

        return new LayerMemoryReport.Builder(layerName, ZeroPadding3DLayer.class, inputType, outputType)
                .standardMemory(0, 0) //No params
                .workingMemory(0, 0, MemoryReport.CACHE_MODE_ALL_ZEROS,
                        MemoryReport.CACHE_MODE_ALL_ZEROS)
                .cacheMemory(MemoryReport.CACHE_MODE_ALL_ZEROS, MemoryReport.CACHE_MODE_ALL_ZEROS) //No caching
                .build();
    }

    public static class Builder extends Layer.Builder<Builder> {

        private int[] padding = new int[]{0, 0, 0, 0, 0, 0}; // [padLeftD, padRightD, padLeftH, padRightH, padLeftW, padRightW]

        /**
         * @param padding Padding for both the left and right in all three spatial dimensions
         */
        public Builder(int padding) {
            this(padding, padding, padding, padding, padding, padding);
        }


        /**
         * Use same padding for left and right boundaries in depth, height and width.
         *
         * @param padDepth padding used for both depth boundaries
         * @param padHeight padding used for both height boundaries
         * @param padWidth padding used for both width boudaries
         */
        public Builder(int padDepth, int padHeight, int padWidth) {
            this(padDepth, padDepth, padHeight, padHeight, padWidth, padWidth);
        }

        /**
         * Explicit padding of left and right boundaries in depth, height and width dimensions
         *
         * @param padLeftD Depth padding left
         * @param padRightD Depth padding right
         * @param padLeftH Height padding left
         * @param padRightH Height padding right
         * @param padLeftW Width padding left
         * @param padRightW Width padding right
         */
        public Builder(int padLeftD, int padRightD,
                       int padLeftH, int padRightH,
                       int padLeftW, int padRightW) {
            this(new int[]{padLeftD, padRightD, padLeftH, padRightH, padLeftW, padRightW});
        }

        public Builder(int[] padding) {
            if (padding.length == 3) {
                this.padding = new int[]{padding[0], padding[0], padding[1], padding[1], padding[2], padding[2]};
            } else if (padding.length == 6) {
                this.padding = padding;
            } else if (padding.length == 1) {
                this.padding = new int[]{padding[0], padding[0], padding[0], padding[0], padding[0], padding[0]};
            } else {
                throw new IllegalStateException("Padding length has to be either 1, 3 or 6, got " + padding.length);
            }
        }

        @Override
        @SuppressWarnings("unchecked")
        public ZeroPadding3DLayer build() {
            for (int p : padding) {
                if (p < 0)
                    throw new IllegalStateException("Invalid zero padding layer config: padding [left, right]"
                            + " must be > 0 for all elements. Got: " + Arrays.toString(padding));
            }
            return new ZeroPadding3DLayer(this);
        }
    }
}

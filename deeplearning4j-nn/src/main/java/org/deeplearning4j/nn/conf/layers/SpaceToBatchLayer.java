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
import lombok.ToString;
import org.deeplearning4j.nn.api.ParamInitializer;
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
 * Space to batch utility layer configuration for convolutional input types.
 * <p>
 * Does a 2-dimensional space to batch operation, i.e. ransforms data from a tensor from 2 spatial dimensions
 * into batch dimension according to the "blocks" specified (a vector of length 2). Afterwards the spatial
 * dimensions are optionally padded, as specified in "padding", a tensor of dim (2, 2), denoting the padding range.
 * <p>
 * Example:
 * input:         [[[[1], [2]], [[3], [4]]]]
 * input shape:   [1, 2, 2, 1]
 * blocks:        [2, 2]
 * padding:       [[0, 0], [0, 0]]
 * <p>
 * output:        [[[[1]]], [[[2]]], [[[3]]], [[[4]]]]
 * output shape:  [4, 1, 1, 1]
 *
 * @author Max Pumperla
 */

@Data
@NoArgsConstructor
@ToString(callSuper = true)
@EqualsAndHashCode(callSuper = true)
public class SpaceToBatchLayer extends Layer {

    // TODO: throw error when block and padding dims don't match

    protected int[] blocks;
    protected int[][] padding;


    protected SpaceToBatchLayer(Builder builder) {
        super(builder);
        this.blocks = builder.blocks;
        this.padding = builder.padding;
    }

    @Override
    public SpaceToBatchLayer clone() {
        return (SpaceToBatchLayer) super.clone();
    }

    @Override
    public org.deeplearning4j.nn.api.Layer instantiate(NeuralNetConfiguration conf,
                                                       Collection<TrainingListener> trainingListeners,
                                                       int layerIndex, INDArray layerParamsView,
                                                       boolean initializeParams) {
        org.deeplearning4j.nn.layers.convolution.SpaceToBatch ret =
                new org.deeplearning4j.nn.layers.convolution.SpaceToBatch(conf);
        ret.setListeners(trainingListeners);
        ret.setIndex(layerIndex);
        ret.setParamsViewArray(layerParamsView);
        Map<String, INDArray> paramTable = initializer().init(conf, layerParamsView, initializeParams);
        ret.setParamTable(paramTable);
        ret.setConf(conf);
        return ret;
    }

    @Override
    public LayerMemoryReport getMemoryReport(InputType inputType) {
        InputType.InputTypeConvolutional c = (InputType.InputTypeConvolutional) inputType;
        InputType.InputTypeConvolutional outputType = (InputType.InputTypeConvolutional) getOutputType(-1, inputType);

        return new LayerMemoryReport.Builder(layerName, SpaceToBatchLayer.class, inputType, outputType)
                .standardMemory(0, 0) //No params
                .cacheMemory(MemoryReport.CACHE_MODE_ALL_ZEROS, MemoryReport.CACHE_MODE_ALL_ZEROS) //No caching
                .build();
    }

    @Override
    public InputType getOutputType(int layerIndex, InputType inputType) {
        if (inputType == null || inputType.getType() != InputType.Type.CNN) {
            throw new IllegalStateException("Invalid input for Subsampling layer (layer name=\"" + getLayerName()
                    + "\"): Expected CNN input, got " + inputType);
        }
        InputType.InputTypeConvolutional i = (InputType.InputTypeConvolutional) inputType;
        return InputType.convolutional(
                (i.getHeight() + padding[0][0] + padding[0][1]) / blocks[0],
                (i.getWidth()+ padding[1][0] + padding[1][1]) / blocks[1],
                i.getChannels()
        );
    }

    @Override
    public ParamInitializer initializer() {
        return EmptyParamInitializer.getInstance();
    }


    @Override
    public void setNIn(InputType inputType, boolean override) {
        //No op: space to batch layer doesn't have nIn value
    }

    @Override
    public InputPreProcessor getPreProcessorForInputType(InputType inputType) {
        if (inputType == null) {
            throw new IllegalStateException("Invalid input for space to batch layer (layer name=\"" + getLayerName()
                    + "\"): input is null");
        }
        return InputTypeUtil.getPreProcessorForInputTypeCnnLayers(inputType, getLayerName());
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
        throw new UnsupportedOperationException("SpaceToBatchLayer does not contain parameters");
    }


    @NoArgsConstructor
    public static class Builder<T extends Builder<T>> extends Layer.Builder<T>{
        protected int[] blocks;
        protected int[][] padding;

        public Builder(int[] blocks) {
            this.blocks = blocks;
            this.padding = new int[][]{{0, 0}, {0, 0}};
        }

        public Builder(int[] blocks, int[][] padding) {
            this.blocks = blocks;
            this.padding = padding;
        }

        public T blocks(int[] blocks) {
            this.blocks = blocks;
            return (T) this;
        }

        public T padding(int[][] padding) {
            this.padding = padding;
            return (T) this;
        }

        @Override
        public T name(String layerName) {
            this.layerName = layerName;
            return (T) this;
        }

        @Override
        @SuppressWarnings("unchecked")
        public SpaceToBatchLayer build() {
            return new SpaceToBatchLayer(this);
        }
    }

}

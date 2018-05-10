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
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.nn.conf.memory.MemoryReport;
import org.deeplearning4j.nn.conf.serde.legacyformat.LegacyIntArrayDeserializer;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.shade.jackson.databind.annotation.JsonDeserialize;

import java.util.Collection;
import java.util.Map;

/**
 * Upsampling 2D layer
 *
 * @author Max Pumperla
 */

@Data
@NoArgsConstructor
@ToString(callSuper = true)
@EqualsAndHashCode(callSuper = true)
public class Upsampling2D extends BaseUpsamplingLayer {

    @JsonDeserialize(using= LegacyIntArrayDeserializer.class)
    protected int[] size;

    protected Upsampling2D(UpsamplingBuilder builder) {
        super(builder);
        this.size = builder.size;
    }

    @Override
    public Upsampling2D clone() {
        Upsampling2D clone = (Upsampling2D) super.clone();
        return clone;
    }

    @Override
    public org.deeplearning4j.nn.api.Layer instantiate(NeuralNetConfiguration conf,
                                                       Collection<TrainingListener> trainingListeners, int layerIndex, INDArray layerParamsView,
                                                       boolean initializeParams) {
        org.deeplearning4j.nn.layers.convolution.upsampling.Upsampling2D ret =
                new org.deeplearning4j.nn.layers.convolution.upsampling.Upsampling2D(conf);
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
        if (inputType == null || inputType.getType() != InputType.Type.CNN) {
            throw new IllegalStateException("Invalid input for Upsampling 2D layer (layer name=\"" + getLayerName()
                    + "\"): Expected CNN input, got " + inputType);
        }
        InputType.InputTypeConvolutional i = (InputType.InputTypeConvolutional) inputType;
        int inHeight = i.getHeight();
        int inWidth = i.getWidth();
        int inDepth = i.getChannels();

        return InputType.convolutional(size[0] * inHeight, size[1] * inWidth, inDepth);
    }

    @Override
    public InputPreProcessor getPreProcessorForInputType(InputType inputType) {
        if (inputType == null) {
            throw new IllegalStateException("Invalid input for Upsampling 2D layer (layer name=\"" + getLayerName()
                    + "\"): input is null");
        }
        return InputTypeUtil.getPreProcessorForInputTypeCnnLayers(inputType, getLayerName());
    }

    @Override
    public LayerMemoryReport getMemoryReport(InputType inputType) {
        InputType.InputTypeConvolutional c = (InputType.InputTypeConvolutional) inputType;
        InputType.InputTypeConvolutional outputType = (InputType.InputTypeConvolutional) getOutputType(-1, inputType);

        // During forward pass: im2col array + reduce. Reduce is counted as activations, so only im2col is working mem
        int im2colSizePerEx = c.getChannels() * outputType.getHeight() * outputType.getWidth()
                * size[0] * size[1] * size[2];

        // Current implementation does NOT cache im2col etc... which means: it's recalculated on each backward pass
        int trainingWorkingSizePerEx = im2colSizePerEx;
        if (getIDropout() != null) {
            //Dup on the input before dropout, but only for training
            trainingWorkingSizePerEx += inputType.arrayElementsPerExample();
        }

        return new LayerMemoryReport.Builder(layerName, Upsampling2D.class, inputType, outputType)
                .standardMemory(0, 0) //No params
                .workingMemory(0, im2colSizePerEx, 0, trainingWorkingSizePerEx)
                .cacheMemory(MemoryReport.CACHE_MODE_ALL_ZEROS, MemoryReport.CACHE_MODE_ALL_ZEROS) //No caching
                .build();
    }


    @NoArgsConstructor
    public static class Builder extends UpsamplingBuilder<Builder> {

        public Builder(int size) {
            super(new int[]{size, size});
        }

        /**
         * Upsampling size int, used for both height and width
         *
         * @param size upsampling size in height and width dimensions
         */
        public Builder size(int size) {

            this.size = new int[]{size, size};
            return this;
        }


        /**
         * Upsampling size array
         *
         * @param size upsampling size in height and width dimensions
         */
        public Builder size(int[] size) {
            Preconditions.checkArgument(size.length == 2);

            this.size = size;
            return this;
        }

        @Override
        @SuppressWarnings("unchecked")
        public Upsampling2D build() {
            return new Upsampling2D(this);
        }
    }

}

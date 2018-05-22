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

import lombok.*;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.nn.conf.memory.MemoryReport;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Collection;
import java.util.Map;

/**
 * Upsampling 3D layer
 *
 * @author Max Pumperla
 */

@Data
@NoArgsConstructor
@ToString(callSuper = true)
@EqualsAndHashCode(callSuper = true)
public class Upsampling3D extends BaseUpsamplingLayer {

    protected int[] size;

    protected Upsampling3D(UpsamplingBuilder builder) {
        super(builder);
        this.size = builder.size;
    }

    @Override
    public Upsampling3D clone() {
        return (Upsampling3D) super.clone();
    }

    @Override
    public org.deeplearning4j.nn.api.Layer instantiate(NeuralNetConfiguration conf,
                                                       Collection<TrainingListener> iterationListeners,
                                                       int layerIndex, INDArray layerParamsView,
                                                       boolean initializeParams) {
        org.deeplearning4j.nn.layers.convolution.upsampling.Upsampling3D ret =
                new org.deeplearning4j.nn.layers.convolution.upsampling.Upsampling3D(conf);
        ret.setListeners(iterationListeners);
        ret.setIndex(layerIndex);
        ret.setParamsViewArray(layerParamsView);
        Map<String, INDArray> paramTable = initializer().init(conf, layerParamsView, initializeParams);
        ret.setParamTable(paramTable);
        ret.setConf(conf);
        return ret;
    }

    @Override
    public InputType getOutputType(int layerIndex, InputType inputType) {
        if (inputType == null || inputType.getType() != InputType.Type.CNN3D) {
            throw new IllegalStateException("Invalid input for Upsampling 3D layer (layer name=\"" + getLayerName()
                    + "\"): Expected CNN3D input, got " + inputType);
        }
        InputType.InputTypeConvolutional3D i = (InputType.InputTypeConvolutional3D) inputType;

        // FIXME: int cast
        int inHeight = (int) i.getHeight();
        int inWidth = (int) i.getWidth();
        int inDepth = (int) i.getDepth();
        int inChannels = (int) i.getChannels();

        return InputType.convolutional3D(
                size[0] * inDepth,size[1] * inHeight, size[2] * inWidth,  inChannels);
    }

    @Override
    public InputPreProcessor getPreProcessorForInputType(InputType inputType) {
        if (inputType == null) {
            throw new IllegalStateException("Invalid input for Upsampling 3D layer (layer name=\"" + getLayerName()
                    + "\"): input is null");
        }
        return InputTypeUtil.getPreProcessorForInputTypeCnn3DLayers(inputType, getLayerName());
    }

    @Override
    public LayerMemoryReport getMemoryReport(InputType inputType) {
        InputType.InputTypeConvolutional3D c = (InputType.InputTypeConvolutional3D) inputType;
        InputType.InputTypeConvolutional3D outputType =
                (InputType.InputTypeConvolutional3D) getOutputType(-1, inputType);

        // During forward pass: im2col array + reduce. Reduce is counted as activations, so only im2col is working mem
        val im2colSizePerEx = c.getChannels() & outputType.getDepth() * outputType.getHeight()
                * outputType.getWidth() * size[0] * size[1] * size[2];

        // Current implementation does NOT cache im2col etc... which means: it's recalculated on each backward pass
        long trainingWorkingSizePerEx = im2colSizePerEx;
        if (getIDropout() != null) {
            //Dup on the input before dropout, but only for training
            trainingWorkingSizePerEx += inputType.arrayElementsPerExample();
        }

        return new LayerMemoryReport.Builder(layerName, Upsampling3D.class, inputType, outputType)
                .standardMemory(0, 0) //No params
                .workingMemory(0, im2colSizePerEx, 0, trainingWorkingSizePerEx)
                .cacheMemory(MemoryReport.CACHE_MODE_ALL_ZEROS, MemoryReport.CACHE_MODE_ALL_ZEROS) //No caching
                .build();
    }


    @NoArgsConstructor
    public static class Builder extends UpsamplingBuilder<Builder> {

        public Builder(int size) {
            super(new int[] {size, size, size});
        }

        /**
         * Upsampling size as int, so same upsampling size is used for depth, width and height
         *
         * @param size upsampling size in height, width and depth dimensions
         */
        public Builder size(int size) {

            this.size = new int[] {size, size, size};
            return this;
        }

        /**
         * Upsampling size as int, so same upsampling size is used for depth, width and height
         *
         * @param size upsampling size in height, width and depth dimensions
         */
        public Builder size(int[] size) {
            Preconditions.checkArgument(size.length == 3);
            this.size = size;
            return this;
        }

        @Override
        @SuppressWarnings("unchecked")
        public Upsampling3D build() {
            return new Upsampling3D(this);
        }
    }

}

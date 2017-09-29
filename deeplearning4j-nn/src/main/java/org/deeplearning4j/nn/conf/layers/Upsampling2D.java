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
import org.deeplearning4j.optimize.api.IterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Arrays;
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

    protected int size;

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
    public org.deeplearning4j.nn.api.Layer instantiate(Collection<IterationListener> iterationListeners,
                                                       String name, int layerIndex, int numInputs, INDArray layerParamsView,
                                                       boolean initializeParams) {
        org.deeplearning4j.nn.layers.convolution.upsampling.Upsampling2D ret =
                        new org.deeplearning4j.nn.layers.convolution.upsampling.Upsampling2D(this);
        ret.setIndex(layerIndex);
        ret.setParamsViewArray(layerParamsView);
        Map<String, INDArray> paramTable = initializer().init(this, layerParamsView, initializeParams);
        ret.setParamTable(paramTable);
        ret.setConf(this);
        return ret;
    }

    @Override
    public InputType[] getOutputType(int layerIndex, InputType... inputType) {
        if (preProcessor != null) {
            inputType = preProcessor.getOutputType(inputType);
        }
        if (inputType == null || inputType[0].getType() != InputType.Type.CNN) {
            throw new IllegalStateException("Invalid input for Subsampling layer (layer name=\"" + getLayerName()
                            + "\"): Expected CNN input, got " + (inputType == null ? null : inputType[0]));
        }
        InputType.InputTypeConvolutional i = (InputType.InputTypeConvolutional) inputType[0];
        int inHeight = i.getHeight();
        int inWidth = i.getWidth();
        int inDepth = i.getDepth();

        return new InputType[]{InputType.convolutional(size * inHeight, size * inWidth, inDepth)};
    }

    @Override
    public InputPreProcessor getPreProcessorForInputType(InputType inputType) {
        if (inputType == null) {
            throw new IllegalStateException("Invalid input for Upsampling layer (layer name=\"" + getLayerName()
                            + "\"): input is null");
        }
        return InputTypeUtil.getPreProcessorForInputTypeCnnLayers(inputType, getLayerName());
    }

    @Override
    public LayerMemoryReport getMemoryReport(InputType... inputTypes) {
        if(inputTypes == null || inputTypes.length != 1){
            throw new IllegalArgumentException("Expected 1 input type: got " + (inputTypes == null ? null : Arrays.toString(inputTypes)));
        }
        InputType inputType = inputTypes[0];

        InputType.InputTypeConvolutional c = (InputType.InputTypeConvolutional) inputType;
        InputType.InputTypeConvolutional outputType = (InputType.InputTypeConvolutional) getOutputType(-1, inputType)[0];

        // During forward pass: im2col array + reduce. Reduce is counted as activations, so only im2col is working mem
        int im2colSizePerEx = c.getDepth() * outputType.getHeight() * outputType.getWidth() * size;

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
            super(size);
        }

        /**
         * Upsampling size
         *
         * @param size    upsampling size in height and width dimensions
         */
        public Builder size(int size) {

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

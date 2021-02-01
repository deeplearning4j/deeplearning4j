/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.nn.conf.layers.convolutional;

import lombok.*;
import org.deeplearning4j.nn.conf.CNN2DFormat;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.InputTypeUtil;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.layers.NoParamLayer;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.nn.layers.convolution.Cropping2DLayer;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.util.ConvolutionUtils;
import org.deeplearning4j.util.ValidationUtils;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Collection;
import java.util.Map;

/**
 * Cropping layer for convolutional (2d) neural networks. Allows cropping to be done separately for
 * top/bottom/left/right
 *
 * @author Alex Black
 */
@Data
@NoArgsConstructor
@EqualsAndHashCode(callSuper = true)
public class Cropping2D extends NoParamLayer {

    private int[] cropping;
    private CNN2DFormat dataFormat = CNN2DFormat.NCHW;

    /**
     * @param cropTopBottom Amount of cropping to apply to both the top and the bottom of the input activations
     * @param cropLeftRight Amount of cropping to apply to both the left and the right of the input activations
     */
    public Cropping2D(int cropTopBottom, int cropLeftRight) {
        this(cropTopBottom, cropTopBottom, cropLeftRight, cropLeftRight);
    }

    public Cropping2D(CNN2DFormat dataFormat, int cropTopBottom, int cropLeftRight) {
        this(dataFormat, cropTopBottom, cropTopBottom, cropLeftRight, cropLeftRight);
    }

    /**
     * @param cropTop Amount of cropping to apply to the top of the input activations
     * @param cropBottom Amount of cropping to apply to the bottom of the input activations
     * @param cropLeft Amount of cropping to apply to the left of the input activations
     * @param cropRight Amount of cropping to apply to the right of the input activations
     */
    public Cropping2D(int cropTop, int cropBottom, int cropLeft, int cropRight) {
        this(CNN2DFormat.NCHW, cropTop, cropBottom, cropLeft, cropRight);
    }

    public Cropping2D(CNN2DFormat format, int cropTop, int cropBottom, int cropLeft, int cropRight) {
        this(new Builder(cropTop, cropBottom, cropLeft, cropRight).dataFormat(format));
    }

    /**
     * @param cropping Cropping as either a length 2 array, with values {@code [cropTopBottom, cropLeftRight]}, or as a
     * length 4 array, with values {@code [cropTop, cropBottom, cropLeft, cropRight]}
     */
    public Cropping2D(int[] cropping) {
        this(new Builder(cropping));
    }

    protected Cropping2D(Builder builder) {
        super(builder);
        this.cropping = builder.cropping;
        this.dataFormat = builder.cnn2DFormat;
    }

    @Override
    public org.deeplearning4j.nn.api.Layer instantiate(NeuralNetConfiguration conf,
                                                       Collection<TrainingListener> trainingListeners, int layerIndex, INDArray layerParamsView,
                                                       boolean initializeParams, DataType networkDataType) {
        Cropping2DLayer ret = new Cropping2DLayer(conf, networkDataType);
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
        int outH = hwd[0] - cropping[0] - cropping[1];
        int outW = hwd[1] - cropping[2] - cropping[3];

        InputType.InputTypeConvolutional c = (InputType.InputTypeConvolutional)inputType;

        return InputType.convolutional(outH, outW, hwd[2], c.getFormat());
    }

    @Override
    public InputPreProcessor getPreProcessorForInputType(InputType inputType) {
        Preconditions.checkArgument(inputType != null, "Invalid input for Cropping2D layer (layer name=\""
                        + getLayerName() + "\"): InputType is null");
        return InputTypeUtil.getPreProcessorForInputTypeCnnLayers(inputType, getLayerName());
    }

    @Override
    public LayerMemoryReport getMemoryReport(InputType inputType) {
        return null;
    }

    @Override
    public void setNIn(InputType inputType, boolean override) {
        this.dataFormat = ((InputType.InputTypeConvolutional)inputType).getFormat();
    }

    @Getter
    @Setter
    public static class Builder extends Layer.Builder<Builder> {

        /**
         * Cropping amount for top/bottom/left/right (in that order). A length 4 array.
         */
        @Setter(AccessLevel.NONE)
        private int[] cropping = new int[] {0, 0, 0, 0};

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
         * @param cropping Cropping amount for top/bottom/left/right (in that order). Must be length 1, 2, or 4 array.
         */
        public void setCropping(int... cropping) {
            this.cropping = ValidationUtils.validate4NonNegative(cropping, "cropping");
        }

        public Builder() {

        }

        /**
         * @param cropping Cropping amount for top/bottom/left/right (in that order). Must be length 4 array.
         */
        public Builder(@NonNull int[] cropping) {
            this.setCropping(cropping);
        }

        /**
         * @param cropTopBottom Amount of cropping to apply to both the top and the bottom of the input activations
         * @param cropLeftRight Amount of cropping to apply to both the left and the right of the input activations
         */
        public Builder(int cropTopBottom, int cropLeftRight) {
            this(cropTopBottom, cropTopBottom, cropLeftRight, cropLeftRight);
        }

        /**
         * @param cropTop Amount of cropping to apply to the top of the input activations
         * @param cropBottom Amount of cropping to apply to the bottom of the input activations
         * @param cropLeft Amount of cropping to apply to the left of the input activations
         * @param cropRight Amount of cropping to apply to the right of the input activations
         */
        public Builder(int cropTop, int cropBottom, int cropLeft, int cropRight) {
            this.setCropping(new int[] {cropTop, cropBottom, cropLeft, cropRight});
        }

        public Cropping2D build() {
            return new Cropping2D(this);
        }
    }
}

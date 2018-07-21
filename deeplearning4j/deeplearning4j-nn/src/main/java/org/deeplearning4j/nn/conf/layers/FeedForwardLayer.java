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

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import lombok.ToString;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.preprocessor.CnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.RnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.params.DefaultParamInitializer;

/**
 * Created by jeffreytang on 7/21/15.
 */
@Data
@NoArgsConstructor
@ToString(callSuper = true)
@EqualsAndHashCode(callSuper = true)
public abstract class FeedForwardLayer extends BaseLayer {
    protected long nIn;
    protected long nOut;

    public FeedForwardLayer(Builder builder) {
        super(builder);
        this.nIn = builder.nIn;
        this.nOut = builder.nOut;
    }


    @Override
    public InputType getOutputType(int layerIndex, InputType inputType) {
        if (inputType == null || (inputType.getType() != InputType.Type.FF
                        && inputType.getType() != InputType.Type.CNNFlat)) {
            throw new IllegalStateException("Invalid input type (layer index = " + layerIndex + ", layer name=\""
                            + getLayerName() + "\"): expected FeedForward input type. Got: " + inputType);
        }

        return InputType.feedForward(nOut);
    }

    @Override
    public void setNIn(InputType inputType, boolean override) {
        if (inputType == null || (inputType.getType() != InputType.Type.FF
                        && inputType.getType() != InputType.Type.CNNFlat)) {
            throw new IllegalStateException("Invalid input type (layer name=\"" + getLayerName()
                            + "\"): expected FeedForward input type. Got: " + inputType);
        }

        if (nIn <= 0 || override) {
            if (inputType.getType() == InputType.Type.FF) {
                InputType.InputTypeFeedForward f = (InputType.InputTypeFeedForward) inputType;
                this.nIn = f.getSize();
            } else {
                InputType.InputTypeConvolutionalFlat f = (InputType.InputTypeConvolutionalFlat) inputType;
                this.nIn = f.getFlattenedSize();
            }
        }
    }

    @Override
    public InputPreProcessor getPreProcessorForInputType(InputType inputType) {
        if (inputType == null) {
            throw new IllegalStateException(
                            "Invalid input for layer (layer name = \"" + getLayerName() + "\"): input type is null");
        }

        switch (inputType.getType()) {
            case FF:
            case CNNFlat:
                //FF -> FF and CNN (flattened format) -> FF: no preprocessor necessary
                return null;
            case RNN:
                //RNN -> FF
                return new RnnToFeedForwardPreProcessor();
            case CNN:
                //CNN -> FF
                InputType.InputTypeConvolutional c = (InputType.InputTypeConvolutional) inputType;
                return new CnnToFeedForwardPreProcessor(c.getHeight(), c.getWidth(), c.getChannels());
            default:
                throw new RuntimeException("Unknown input type: " + inputType);
        }
    }

    @Override
    public double getL1ByParam(String paramName) {
        switch (paramName) {
            case DefaultParamInitializer.WEIGHT_KEY:
                return l1;
            case DefaultParamInitializer.BIAS_KEY:
                return l1Bias;
            default:
                throw new IllegalStateException("Unknown parameter: \"" + paramName + "\"");
        }
    }

    @Override
    public double getL2ByParam(String paramName) {
        switch (paramName) {
            case DefaultParamInitializer.WEIGHT_KEY:
                return l2;
            case DefaultParamInitializer.BIAS_KEY:
                return l2Bias;
            default:
                throw new IllegalStateException("Unknown parameter: \"" + paramName + "\"");
        }
    }

    @Override
    public boolean isPretrainParam(String paramName) {
        return false; //No pretrain params in standard FF layers
    }

    @Override
    public boolean isPretrain() {
        return false;
    }

    public abstract static class Builder<T extends Builder<T>> extends BaseLayer.Builder<T> {
        protected int nIn = 0;
        protected int nOut = 0;

        /**
         * Number of inputs for the layer (usually the size of the last layer). <br>
         * Note that for Convolutional layers, this is the input channels, otherwise is the previous layer size.
         *
         * @param nIn Number of inputs for the layer
         */
        public T nIn(int nIn) {
            this.nIn = nIn;
            return (T) this;
        }

        /**
         * Number of inputs for the layer (usually the size of the last layer). <br>
         * Note that for Convolutional layers, this is the input channels, otherwise is the previous layer size.
         *
         * @param nIn Number of inputs for the layer
         */
        public T nIn(long nIn) {
            // FIXME: int cast
            this.nIn = (int) nIn;
            return (T) this;
        }

        /**
         * Number of outputs - used to set the layer size (number of units/nodes for the current layer).
         * Note that this is equivalent to {@link #units(int)}
         *
         * @param nOut Number of outputs / layer size
         */
        public T nOut(int nOut) {
            this.nOut = nOut;
            return (T) this;
        }

        /**
         * Number of outputs - used to set the layer size (number of units/nodes for the current layer).
         * Note that this is equivalent to {@link #units(int)}
         *
         * @param nOut Number of outputs / layer size
         */
        public T nOut(long nOut) {
            this.nOut = (int) nOut;
            return (T) this;
        }

        /**
         * Set the number of units / layer size for this layer.<br>
         * This is equivalent to {@link #nOut(int)}
         *
         * @param units Size of the layer (number of units) / nOut
         * @see #nOut(int)
         */
        public T units(int units){
            return nOut(units);
        }
    }
}

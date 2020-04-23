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
import org.deeplearning4j.nn.api.layers.LayerConstraint;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.RNNFormat;
import org.deeplearning4j.nn.conf.distribution.Distribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.weights.IWeightInit;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.nn.weights.WeightInitDistribution;

import java.util.Arrays;
import java.util.List;

@Data
@NoArgsConstructor
@ToString(callSuper = true)
@EqualsAndHashCode(callSuper = true)
public abstract class BaseRecurrentLayer extends FeedForwardLayer {

    protected IWeightInit weightInitFnRecurrent;
    protected RNNFormat rnnDataFormat = RNNFormat.NCW;

    protected BaseRecurrentLayer(Builder builder) {
        super(builder);
        this.weightInitFnRecurrent = builder.weightInitFnRecurrent;
        this.rnnDataFormat = builder.rnnDataFormat;
    }

    @Override
    public InputType getOutputType(int layerIndex, InputType inputType) {
        if (inputType == null || inputType.getType() != InputType.Type.RNN) {
            throw new IllegalStateException("Invalid input for RNN layer (layer index = " + layerIndex
                            + ", layer name = \"" + getLayerName() + "\"): expect RNN input type with size > 0. Got: "
                            + inputType);
        }

        InputType.InputTypeRecurrent itr = (InputType.InputTypeRecurrent) inputType;

        return InputType.recurrent(nOut, itr.getTimeSeriesLength(), itr.getFormat());
    }

    @Override
    public void setNIn(InputType inputType, boolean override) {
        if (inputType == null || inputType.getType() != InputType.Type.RNN) {
            throw new IllegalStateException("Invalid input for RNN layer (layer name = \"" + getLayerName()
                            + "\"): expect RNN input type with size > 0. Got: " + inputType);
        }

        if (nIn <= 0 || override) {
            InputType.InputTypeRecurrent r = (InputType.InputTypeRecurrent) inputType;
            this.nIn = r.getSize();
            this.rnnDataFormat = r.getFormat();
        }
    }

    @Override
    public InputPreProcessor getPreProcessorForInputType(InputType inputType) {
        return InputTypeUtil.getPreprocessorForInputTypeRnnLayers(inputType, rnnDataFormat,getLayerName());
    }

    @NoArgsConstructor
    @Getter
    @Setter
    public static abstract class Builder<T extends Builder<T>> extends FeedForwardLayer.Builder<T> {

        /**
         * Set the format of data expected by the RNN. NCW = [miniBatchSize, size, timeSeriesLength],
         * NWC = [miniBatchSize, timeSeriesLength, size]. Defaults to NCW.
         */
        protected RNNFormat rnnDataFormat = RNNFormat.NCW;

        /**
         * Set constraints to be applied to the RNN recurrent weight parameters of this layer. Default: no
         * constraints.<br> Constraints can be used to enforce certain conditions (non-negativity of parameters,
         * max-norm regularization, etc). These constraints are applied at each iteration, after the parameters have
         * been updated.
         */
        protected List<LayerConstraint> recurrentConstraints;

        /**
         * Set constraints to be applied to the RNN input weight parameters of this layer. Default: no constraints.<br>
         * Constraints can be used to enforce certain conditions (non-negativity of parameters, max-norm regularization,
         * etc). These constraints are applied at each iteration, after the parameters have been updated.
         *
         */
        protected List<LayerConstraint> inputWeightConstraints;

        /**
         * Set the weight initialization for the recurrent weights. Not that if this is not set explicitly, the same
         * weight initialization as the layer input weights is also used for the recurrent weights.
         *
         */
        protected IWeightInit weightInitFnRecurrent;

        /**
         * Set constraints to be applied to the RNN recurrent weight parameters of this layer. Default: no
         * constraints.<br> Constraints can be used to enforce certain conditions (non-negativity of parameters,
         * max-norm regularization, etc). These constraints are applied at each iteration, after the parameters have
         * been updated.
         *
         * @param constraints Constraints to apply to the recurrent weight parameters of this layer
         */
        public T constrainRecurrent(LayerConstraint... constraints) {
            this.setRecurrentConstraints(Arrays.asList(constraints));
            return (T) this;
        }

        /**
         * Set constraints to be applied to the RNN input weight parameters of this layer. Default: no constraints.<br>
         * Constraints can be used to enforce certain conditions (non-negativity of parameters, max-norm regularization,
         * etc). These constraints are applied at each iteration, after the parameters have been updated.
         *
         * @param constraints Constraints to apply to the input weight parameters of this layer
         */
        public T constrainInputWeights(LayerConstraint... constraints) {
            this.setInputWeightConstraints(Arrays.asList(constraints));
            return (T) this;
        }

        /**
         * Set the weight initialization for the recurrent weights. Not that if this is not set explicitly, the same
         * weight initialization as the layer input weights is also used for the recurrent weights.
         *
         * @param weightInit Weight initialization for the recurrent weights only.
         */
        public T weightInitRecurrent(IWeightInit weightInit) {
            this.setWeightInitFnRecurrent(weightInit);
            return (T) this;
        }

        /**
         * Set the weight initialization for the recurrent weights. Not that if this is not set explicitly, the same
         * weight initialization as the layer input weights is also used for the recurrent weights.
         *
         * @param weightInit Weight initialization for the recurrent weights only.
         */
        public T weightInitRecurrent(WeightInit weightInit) {
            if (weightInit == WeightInit.DISTRIBUTION) {
                throw new UnsupportedOperationException(
                                "Not supported!, Use weightInit(Distribution distribution) instead!");
            }

            this.setWeightInitFnRecurrent(weightInit.getWeightInitFunction());
            return (T) this;
        }

        /**
         * Set the weight initialization for the recurrent weights, based on the specified distribution. Not that if
         * this is not set explicitly, the same weight initialization as the layer input weights is also used for the
         * recurrent weights.
         *
         * @param dist Distribution to use for initializing the recurrent weights
         */
        public T weightInitRecurrent(Distribution dist) {
            this.setWeightInitFnRecurrent(new WeightInitDistribution(dist));
            return (T) this;
        }

        public T dataFormat(RNNFormat rnnDataFormat){
            this.rnnDataFormat = rnnDataFormat;
            return (T)this;
        }
    }
}

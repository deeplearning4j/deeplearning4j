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
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.nn.conf.memory.MemoryReport;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.nn.weights.IWeightInit;
import org.deeplearning4j.nn.weights.embeddings.ArrayEmbeddingInitializer;
import org.deeplearning4j.nn.weights.embeddings.EmbeddingInitializer;
import org.deeplearning4j.nn.weights.embeddings.WeightInitEmbedding;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.activations.impl.ActivationIdentity;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Collection;
import java.util.Map;

/**
 * Embedding layer for sequences: feed-forward layer that expects fixed-length number (inputLength) of integers/indices
 * per example as input, ranged from 0 to numClasses - 1. This input thus has shape [numExamples, inputLength] or shape
 * [numExamples, 1, inputLength].<br> The output of this layer is 3D (sequence/time series), namely of shape
 * [numExamples, nOut, inputLength].
 * <b>Note</b>: can only be used as the first layer for a network<br>
 * <b>Note 2</b>: For a given example index i, the output is activationFunction(weights.getRow(i) + bias), hence the
 * weight rows can be considered a vector/embedding of each index.<br> Note also that embedding layer has an activation
 * function (set to IDENTITY to disable) and optional bias (which is disabled by default)
 *
 * @author Max Pumperla
 */
@Data
@NoArgsConstructor
@ToString(callSuper = true)
@EqualsAndHashCode(callSuper = true)
public class EmbeddingSequenceLayer extends FeedForwardLayer {

    private int inputLength = 1; // By default only use one index to embed
    private boolean hasBias = false;
    private boolean inferInputLength = false; // use input length as provided by input data

    private EmbeddingSequenceLayer(Builder builder) {
        super(builder);
        this.hasBias = builder.hasBias;
        this.inputLength = builder.inputLength;
        this.inferInputLength = builder.inferInputLength;
        initializeConstraints(builder);
    }

    @Override
    public Layer instantiate(NeuralNetConfiguration conf, Collection<TrainingListener> trainingListeners,
                             int layerIndex, INDArray layerParamsView, boolean initializeParams, DataType networkDataType) {
        org.deeplearning4j.nn.layers.feedforward.embedding.EmbeddingSequenceLayer ret =
                        new org.deeplearning4j.nn.layers.feedforward.embedding.EmbeddingSequenceLayer(conf, networkDataType);
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
        if (inputType == null || (inputType.getType() != InputType.Type.FF && inputType.getType() != InputType.Type.RNN)) {
            throw new IllegalStateException("Invalid input for Embedding layer (layer index = " + layerIndex
                            + ", layer name = \"" + getLayerName() + "\"): expect FF/RNN input type. Got: " + inputType);
        }
        return InputType.recurrent(nOut, inputLength);
    }

    @Override
    public ParamInitializer initializer() {
        return DefaultParamInitializer.getInstance();
    }

    @Override
    public LayerMemoryReport getMemoryReport(InputType inputType) {
        InputType outputType = getOutputType(-1, inputType);

        val actElementsPerEx = outputType.arrayElementsPerExample();
        val numParams = initializer().numParams(this);
        val updaterStateSize = (int) getIUpdater().stateSize(numParams);

        return new LayerMemoryReport.Builder(layerName, EmbeddingSequenceLayer.class, inputType, outputType)
                        .standardMemory(numParams, updaterStateSize).workingMemory(0, 0, 0, actElementsPerEx)
                        .cacheMemory(MemoryReport.CACHE_MODE_ALL_ZEROS, MemoryReport.CACHE_MODE_ALL_ZEROS) //No caching
                        .build();
    }

    public boolean hasBias() {
        return hasBias;
    }

    @Override
    public InputPreProcessor getPreProcessorForInputType(InputType inputType) {
        if (inputType == null) {
            throw new IllegalStateException(
                    "Invalid input for layer (layer name = \"" + getLayerName() + "\"): input type is null");
        }

        if(inputType.getType() == InputType.Type.RNN){
            return null;
        }
        return super.getPreProcessorForInputType(inputType);
    }

    @Override
    public void setNIn(InputType inputType, boolean override) {
        if(inputType.getType() == InputType.Type.RNN){
            if (nIn <= 0 || override) {
                InputType.InputTypeRecurrent f = (InputType.InputTypeRecurrent) inputType;
                this.nIn = f.getSize();
            }
        } else {
            super.setNIn(inputType, override);
        }

    }

    @Getter
    @Setter
    public static class Builder extends FeedForwardLayer.Builder<Builder> {

        public Builder(){
            //Default to Identity activation - i.e., don't inherit.
            //For example, if user sets ReLU as global default, they very likely don't intend to use it for Embedding layer also
            this.activationFn = new ActivationIdentity();
        }

        /**
         * If true: include bias parameters in the layer. False (default): no bias.
         *
         */
        private boolean hasBias = false;

        /**
         * Set input sequence length for this embedding layer.
         *
         */
        private int inputLength = 1;

        /**
         * Set input sequence inference mode for embedding layer.
         *
         */
        private boolean inferInputLength = true;

        /**
         * If true: include bias parameters in the layer. False (default): no bias.
         *
         * @param hasBias If true: include bias parameters in this layer
         */
        public Builder hasBias(boolean hasBias) {
            this.setHasBias(hasBias);
            return this;
        }

        /**
         * Set input sequence length for this embedding layer.
         *
         * @param inputLength input sequence length
         * @return Builder
         */
        public Builder inputLength(int inputLength) {
            this.setInputLength(inputLength);
            return this;
        }


        /**
         * Set input sequence inference mode for embedding layer.
         *
         * @param inferInputLength whether to infer input length
         * @return Builder
         */
        public Builder inferInputLength(boolean inferInputLength) {
            this.setInferInputLength(inferInputLength);
            return this;
        }

        @Override
        public Builder weightInit(IWeightInit weightInit) {
            this.setWeightInitFn(weightInit);
            return this;
        }

        @Override
        public void setWeightInitFn(IWeightInit weightInit){
            if(weightInit instanceof WeightInitEmbedding){
                long[] shape = ((WeightInitEmbedding) weightInit).shape();
                nIn(shape[0]);
                nOut(shape[1]);
            }
            this.weightInitFn = weightInit;
        }

        /**
         * Initialize the embedding layer using the specified EmbeddingInitializer - such as a Word2Vec instance
         *
         * @param embeddingInitializer Source of the embedding layer weights
         */
        public Builder weightInit(EmbeddingInitializer embeddingInitializer){
            return weightInit(new WeightInitEmbedding(embeddingInitializer));
        }

        /**
         * Initialize the embedding layer using values from the specified array. Note that the array should have shape
         * [vocabSize, vectorSize]. After copying values from the array to initialize the network parameters, the input
         * array will be discarded (so that, if necessary, it can be garbage collected)
         *
         * @param vectors Vectors to initialize the embedding layer with
         */
        public Builder weightInit(INDArray vectors){
            return weightInit(new ArrayEmbeddingInitializer(vectors));
        }

        @Override
        @SuppressWarnings("unchecked")
        public EmbeddingSequenceLayer build() {
            return new EmbeddingSequenceLayer(this);
        }
    }
}

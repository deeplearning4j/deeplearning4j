package org.deeplearning4j.nn.conf.layers;

import lombok.*;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.nn.conf.memory.MemoryReport;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Collection;
import java.util.Map;

/**
 * Embedding layer for sequences: feed-forward layer that expects fixed-length number (inputLength) of integers/indices
 * per example as input, ranged from 0 to numClasses - 1. This input thus has shape [numExamples, inputLength].
 * The output of this layer is 3D, namely of shape [numExamples, nOut, inputLength].
 * <b>Note</b>: can only be used as the first layer for a network<br>
 * <b>Note 2</b>: For a given example index i, the output is activationFunction(weights.getRow(i) + bias), hence the
 * weight rows can be considered a vector/embedding of each index.
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
                             int layerIndex, INDArray layerParamsView, boolean initializeParams) {
        org.deeplearning4j.nn.layers.feedforward.embedding.EmbeddingSequenceLayer ret =
                new org.deeplearning4j.nn.layers.feedforward.embedding.EmbeddingSequenceLayer(conf);
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
        if (inputType == null || inputType.getType() != InputType.Type.FF) {
            throw new IllegalStateException("Invalid input for Embedding layer (layer index = " + layerIndex
                    + ", layer name = \"" + getLayerName() + "\"): expect FFN input type. Got: "
                    + inputType);
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
                .standardMemory(numParams, updaterStateSize)
                .workingMemory(0, 0, 0, actElementsPerEx)
                .cacheMemory(MemoryReport.CACHE_MODE_ALL_ZEROS, MemoryReport.CACHE_MODE_ALL_ZEROS) //No caching
                .build();
    }

    public boolean hasBias() {
        return hasBias;
    }

    @NoArgsConstructor
    public static class Builder extends FeedForwardLayer.Builder<Builder> {

        private boolean hasBias = false;
        private int inputLength = 1;
        private boolean inferInputLength = true;

        /**
         * If true: include bias parameters in the layer. False (default): no bias.
         *
         * @param hasBias If true: include bias parameters in this layer
         */
        public Builder hasBias(boolean hasBias) {
            this.hasBias = hasBias;
            return this;
        }

        /**
         * Set input sequence length for this embedding layer.
         *
         * @param inputLength input sequence length
         * @return Builder
         */
        public Builder inputLength(int inputLength) {
            this.inputLength = inputLength;
            return this;
        }


        /**
         * Set input sequence inference mode for embedding layer.
         *
         * @param inferInputLength whether to infer input length
         * @return Builder
         */
        public Builder inferInputLength(boolean inferInputLength) {
            this.inferInputLength = inferInputLength;
            return this;
        }

        @Override
        @SuppressWarnings("unchecked")
        public EmbeddingSequenceLayer build() {
            return new EmbeddingSequenceLayer(this);
        }
    }
}

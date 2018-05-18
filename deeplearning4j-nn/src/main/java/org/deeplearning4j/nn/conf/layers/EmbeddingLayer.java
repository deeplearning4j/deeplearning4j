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
 * Embedding layer: feed-forward layer that expects single integers per example as input (class numbers, in range 0 to numClass-1)
 * as input. This input has shape [numExamples,1] instead of [numExamples,numClasses] for the equivalent one-hot representation.
 * Mathematically, EmbeddingLayer is equivalent to using a DenseLayer with a one-hot representation for the input; however,
 * it can be much more efficient with a large number of classes (as a dense layer + one-hot input does a matrix multiply
 * with all but one value being zero).<br>
 * <b>Note</b>: can only be used as the first layer for a network<br>
 * <b>Note 2</b>: For a given example index i, the output is activationFunction(weights.getRow(i) + bias), hence the
 * weight rows can be considered a vector/embedding for each example.
 *
 * @author Alex Black
 */
@Data
@NoArgsConstructor
@ToString(callSuper = true)
@EqualsAndHashCode(callSuper = true)
public class EmbeddingLayer extends FeedForwardLayer {
    private boolean hasBias = true; //Default for pre-0.9.2 implementations

    private EmbeddingLayer(Builder builder) {
        super(builder);
        this.hasBias = builder.hasBias;
        initializeConstraints(builder);
    }

    @Override
    public Layer instantiate(NeuralNetConfiguration conf, Collection<TrainingListener> trainingListeners,
                    int layerIndex, INDArray layerParamsView, boolean initializeParams) {
        org.deeplearning4j.nn.layers.feedforward.embedding.EmbeddingLayer ret =
                        new org.deeplearning4j.nn.layers.feedforward.embedding.EmbeddingLayer(conf);
        ret.setListeners(trainingListeners);
        ret.setIndex(layerIndex);
        ret.setParamsViewArray(layerParamsView);
        Map<String, INDArray> paramTable = initializer().init(conf, layerParamsView, initializeParams);
        ret.setParamTable(paramTable);
        ret.setConf(conf);
        return ret;
    }

    @Override
    public ParamInitializer initializer() {
        return DefaultParamInitializer.getInstance();
    }

    @Override
    public LayerMemoryReport getMemoryReport(InputType inputType) {
        //Basically a dense layer, but no dropout is possible here, and no epsilons
        InputType outputType = getOutputType(-1, inputType);

        val actElementsPerEx = outputType.arrayElementsPerExample();
        int numParams = initializer().numParams(this);
        int updaterStateSize = (int) getIUpdater().stateSize(numParams);

        //Embedding layer does not use caching.
        //Inference: no working memory - just activations (pullRows)
        //Training: preout op, the only in-place ops on epsilon (from layer above) + assign ops

        return new LayerMemoryReport.Builder(layerName, EmbeddingLayer.class, inputType, outputType)
                        .standardMemory(numParams, updaterStateSize).workingMemory(0, 0, 0, actElementsPerEx)
                        .cacheMemory(MemoryReport.CACHE_MODE_ALL_ZEROS, MemoryReport.CACHE_MODE_ALL_ZEROS) //No caching
                        .build();
    }

    public boolean hasBias(){
        return hasBias;
    }

    @NoArgsConstructor
    public static class Builder extends FeedForwardLayer.Builder<Builder> {

        private boolean hasBias = false;

        /**
         * If true: include bias parameters in the layer. False (default): no bias.
         *
         * @param hasBias If true: include bias parameters in this layer
         */
        public Builder hasBias(boolean hasBias){
            this.hasBias = hasBias;
            return this;
        }

        @Override
        @SuppressWarnings("unchecked")
        public EmbeddingLayer build() {
            return new EmbeddingLayer(this);
        }
    }
}

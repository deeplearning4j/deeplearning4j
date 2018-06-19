package org.deeplearning4j.nn.conf.layers.misc;

import lombok.*;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.FeedForwardLayer;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.nn.conf.memory.MemoryReport;
import org.deeplearning4j.nn.params.EmptyParamInitializer;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Collection;
import java.util.Map;

/**
 * RepeatVector layer configuration.
 *
 * RepeatVector takes a mini-batch of vectors of shape (mb, length) and a repeat factor n and outputs
 * a 3D tensor of shape (mb, n, length) in which x is repeated n times.
 *
 * @author Max Pumperla
 */
@Data
@NoArgsConstructor
@ToString(callSuper = true)
@EqualsAndHashCode(callSuper = true)
public class RepeatVector extends FeedForwardLayer {

    private int n = 1;

    protected RepeatVector(Builder builder) {
        super(builder);
        this.n = builder.n;
    }

    @Override
    public RepeatVector clone() {
        return  (RepeatVector) super.clone();
    }

    @Override
    public ParamInitializer initializer() {
        return EmptyParamInitializer.getInstance();
    }

    @Override
    public org.deeplearning4j.nn.api.Layer instantiate(NeuralNetConfiguration conf,
                                                       Collection<TrainingListener> trainingListeners,
                                                       int layerIndex, INDArray layerParamsView,
                                                       boolean initializeParams) {
        org.deeplearning4j.nn.layers.RepeatVector ret =
                new org.deeplearning4j.nn.layers.RepeatVector(conf);
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
            throw new IllegalStateException("Invalid input for RepeatVector layer (layer name=\"" + getLayerName()
                    + "\"): Expected FF input, got " + inputType);
        }
        InputType.InputTypeFeedForward ffInput = (InputType.InputTypeFeedForward) inputType;
        return InputType.recurrent(ffInput.getSize(), n);
    }

    @Override
    public LayerMemoryReport getMemoryReport(InputType inputType) {
        InputType outputType = getOutputType(-1, inputType);

        return new LayerMemoryReport.Builder(layerName, RepeatVector.class, inputType, outputType)
                .standardMemory(0, 0)
                .workingMemory(0, 0, 0, 0)
                .cacheMemory(MemoryReport.CACHE_MODE_ALL_ZEROS, MemoryReport.CACHE_MODE_ALL_ZEROS)
                .build();
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
        throw new UnsupportedOperationException("UpsamplingLayer does not contain parameters");
    }


    @NoArgsConstructor
    public static class Builder<T extends Builder<T>> extends FeedForwardLayer.Builder<T> {

        private int n = 1; // no repetition by default

        public Builder(int n) {
            this.n = n;
        }

        /**
         * Set repetition factor for RepeatVector layer
         *
         * @param n upsampling size in height and width dimensions
         */
        public Builder repetitionFactor(int n) {
            this.n = n;
            return this;
        }

        @Override
        @SuppressWarnings("unchecked")
        public RepeatVector build() {
            return new RepeatVector(this);
        }
    }
}

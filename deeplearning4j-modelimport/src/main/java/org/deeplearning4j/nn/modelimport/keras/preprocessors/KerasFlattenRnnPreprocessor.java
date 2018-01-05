package org.deeplearning4j.nn.modelimport.keras.preprocessors;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.inputs.InvalidInputTypeException;
import org.deeplearning4j.nn.conf.preprocessor.BaseInputPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.CnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.RnnToFeedForwardPreProcessor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.shade.jackson.annotation.JsonCreator;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.util.Arrays;

/**
 * Preprocessor to flatten input of RNN type
 *
 * @author Max Pumperla
 */
@Slf4j
public class KerasFlattenRnnPreprocessor extends BaseInputPreProcessor {

    int tsLength;
    int depth;

    public KerasFlattenRnnPreprocessor(int depth, int tsLength) {
        super();
        this.tsLength = Math.abs(tsLength);
        this.depth = depth;
    }

    @Override
    public INDArray preProcess(INDArray input, int miniBatchSize) {
        INDArray output = input.dup('c');
        output.reshape(input.size(0), depth * tsLength);
        return output;
    }

    @Override
    public INDArray backprop(INDArray epsilons, int miniBatchSize) {
        return epsilons.dup().reshape(miniBatchSize, depth, tsLength);
    }

    @Override
    public KerasFlattenRnnPreprocessor clone() {
        return (KerasFlattenRnnPreprocessor) super.clone();
    }

    @Override
    public InputType getOutputType(InputType inputType) throws InvalidInputTypeException {

        return InputType.feedForward(depth * tsLength);

    }
}

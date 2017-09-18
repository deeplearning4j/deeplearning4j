package org.deeplearning4j.nn.conf.preprocessor.custom;

import lombok.EqualsAndHashCode;
import org.deeplearning4j.nn.api.MaskState;
import org.deeplearning4j.nn.api.activations.Activations;
import org.deeplearning4j.nn.api.gradients.Gradients;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.Pair;

/**
 * Created by Alex on 09/09/2016.
 */
@EqualsAndHashCode
public class MyCustomPreprocessor implements InputPreProcessor {

    @Override
    public Activations preProcess(Activations input, int miniBatchSize, boolean train) {
        input.get(0).add(1.0);
        return input;
    }

    @Override
    public Gradients backprop(Gradients output, int miniBatchSize, boolean train) {
        return output;
    }

    @Override
    public InputPreProcessor clone() {
        return new MyCustomPreprocessor();
    }

    @Override
    public InputType getOutputType(InputType inputType) {
        return inputType;
    }


    public Pair<INDArray, MaskState> feedForwardMaskArray(INDArray maskArray, MaskState currentMaskState,
                    int minibatchSize) {
        throw new UnsupportedOperationException();
    }
}

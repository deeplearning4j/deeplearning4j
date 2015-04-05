package org.deeplearning4j.nn.conf.preprocessor;

import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Composable input pre processor
 * @author Adam Gibson
 */
public class ComposableInputPreProcessor implements InputPreProcessor {
    private InputPreProcessor[] preProcessors;

    public ComposableInputPreProcessor(InputPreProcessor...preProcessors) {
        this.preProcessors  = preProcessors;
    }
    @Override
    public INDArray preProcess(INDArray input) {
        for(InputPreProcessor preProcessor : preProcessors)
        input = preProcessor.preProcess(input);
        return input;
    }
}

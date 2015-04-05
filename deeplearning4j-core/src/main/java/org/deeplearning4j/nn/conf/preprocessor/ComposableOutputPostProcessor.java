package org.deeplearning4j.nn.conf.preprocessor;

import org.deeplearning4j.nn.conf.OutputPreProcessor;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Composable output post processor
 *
 * @author Adam Gibson
 */
public class ComposableOutputPostProcessor implements OutputPreProcessor {
    private OutputPreProcessor[] outputPreProcessors;

    public ComposableOutputPostProcessor(OutputPreProcessor...outputPreProcessors) {
        this.outputPreProcessors = outputPreProcessors;
    }
    @Override
    public INDArray preProcess(INDArray output) {
        for(OutputPreProcessor outputPreProcessor : outputPreProcessors)
          output = outputPreProcessor.preProcess(output);
        return output;
    }
}

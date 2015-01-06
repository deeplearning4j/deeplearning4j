package org.deeplearning4j.nn.conf.preprocessor;

import org.deeplearning4j.nn.conf.OutputPreProcessor;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Aggregate output pre processor
 * @author Adam Gibson
 */
public class AggregatePreProcessor implements OutputPreProcessor {
    private OutputPreProcessor[] preProcessor;

    public AggregatePreProcessor(OutputPreProcessor[] preProcessor) {
        this.preProcessor = preProcessor;
    }

    @Override
    public INDArray preProcess(INDArray output) {
        for(OutputPreProcessor outputPreProcessor : preProcessor)
            output = outputPreProcessor.preProcess(output);
        return output;
    }
}

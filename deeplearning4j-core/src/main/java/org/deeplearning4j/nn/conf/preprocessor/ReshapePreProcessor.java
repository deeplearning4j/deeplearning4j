package org.deeplearning4j.nn.conf.preprocessor;

import org.deeplearning4j.nn.conf.OutputPreProcessor;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Reshape post processor
 * @author Adam Gibson
 */
public class ReshapePreProcessor implements OutputPreProcessor {
    private int[] shape;

    public ReshapePreProcessor(int...shape) {
        this.shape = shape;
    }

    @Override
    public INDArray preProcess(INDArray output) {
        return output.reshape(shape);
    }
}

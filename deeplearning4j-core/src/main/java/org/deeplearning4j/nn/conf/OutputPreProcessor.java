package org.deeplearning4j.nn.conf;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Output pre processor's handle layer to layer interactions
 * to ensure things like proper shape for input among other things.
 *
 * @author Adam Gibson
 */
public interface OutputPreProcessor {
    /**
     * Used for handling pre processing of layer output.
     * The typical use case is for handling reshaping of output
     * in to shapes proper for the next layer of input.
     * @param output the output to pre process
     * @return the pre processed output
     */
    INDArray preProcess(INDArray output);

}

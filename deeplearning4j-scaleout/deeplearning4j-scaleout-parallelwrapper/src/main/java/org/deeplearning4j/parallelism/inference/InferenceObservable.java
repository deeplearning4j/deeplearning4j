package org.deeplearning4j.parallelism.inference;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * @author raver119@gmail.com
 */
public interface InferenceObservable {

    INDArray[] getInput();

    void setOutput(INDArray... output);
}

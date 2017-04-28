package org.deeplearning4j.parallelism.inference;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Observer;

/**
 * @author raver119@gmail.com
 */
public interface InferenceObservable {

    INDArray[] getInput();

    void setInput(INDArray... input);

    void setOutput(INDArray... output);

    void addObserver(Observer observer);

    INDArray[] getOutput();
}

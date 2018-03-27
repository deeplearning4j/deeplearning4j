package org.deeplearning4j.parallelism.inference;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.List;
import java.util.Observer;

/**
 * @author raver119@gmail.com
 */
public interface InferenceObservable {

    List<INDArray[]> getInputBatches();

    void addInput(INDArray... input);

    void setOutputBatches(List<INDArray[]> output);

    void setOutputException(Exception e);

    void addObserver(Observer observer);

    INDArray[] getOutput();
}

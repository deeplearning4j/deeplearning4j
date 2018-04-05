package org.deeplearning4j.parallelism.inference;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.Pair;

import java.util.List;
import java.util.Observer;

/**
 * @author raver119@gmail.com
 */
public interface InferenceObservable {

    /**
     * Get input batches - and their associated input mask arrays, if any<br>
     * Note that usually the returned list will be of size 1 - however, in the batched case, not all inputs
     * can actually be batched (variable size inputs to fully convolutional net, for example). In these "can't batch"
     * cases, multiple input batches will be returned, to be processed
     *
     * @return List of pairs of input arrays and input mask arrays. Input mask arrays may be null.
     */
    List<Pair<INDArray[],INDArray[]>> getInputBatches();

    void addInput(INDArray... input);

    void addInput(INDArray[] input, INDArray[] inputMasks);

    void setOutputBatches(List<INDArray[]> output);

    void setOutputException(Exception e);

    void addObserver(Observer observer);

    INDArray[] getOutput();
}

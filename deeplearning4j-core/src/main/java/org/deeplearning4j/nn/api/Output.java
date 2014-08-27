package org.deeplearning4j.nn.api;


import org.deeplearning4j.linalg.api.ndarray.INDArray;

/**
 * Interface for outputting  a value
 * relative to an output
 */
public interface Output {

    INDArray output(INDArray input);


}

package org.nd4j.linalg.api.ops;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * A loss function for computing
 * the delta between two arrays
 *
 * @author Adam Gibson
 */
public interface LossFunction extends Accumulation {
    /**
     * The true
     * @return
     */
    INDArray input();

    /**
     * The guess
     * @return
     */
    INDArray output();

}

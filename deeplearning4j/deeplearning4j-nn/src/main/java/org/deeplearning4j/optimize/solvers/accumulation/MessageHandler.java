package org.deeplearning4j.optimize.solvers.accumulation;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.Serializable;

/**
 * This interface describes communication primitive for GradientsAccumulator
 *
 * PLEASE NOTE: All implementations of this interface must be thread-safe.
 *
 * @author raver119@gmail.com
 */
public interface MessageHandler extends Serializable {

    /**
     * This method does initial configuration of given MessageHandler instance
     * @param accumulator
     */
    void initialize(GradientsAccumulator accumulator);

    /**
     * This method does broadcast of given update message across network
     *
     * @param updates
     * @return TRUE if something was sent, FALSE otherwise
     */
    boolean broadcastUpdates(INDArray updates);
}

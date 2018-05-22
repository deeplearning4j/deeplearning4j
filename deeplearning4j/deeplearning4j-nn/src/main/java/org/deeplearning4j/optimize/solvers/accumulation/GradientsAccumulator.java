package org.deeplearning4j.optimize.solvers.accumulation;

import org.deeplearning4j.optimize.api.StepFunction;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.Serializable;
import java.util.Queue;

/**
 * @author raver119@gmail.com
 */
public interface GradientsAccumulator extends Serializable {

    /**
     * This method allows to pass external updates to accumulator, they will be populated across all workers using this GradientsAccumulator instance
     *
     * @param source
     */
    void setExternalSource(Queue<INDArray> source);

    /**
     * This method applies accumulated updates via given StepFunction
     *
     * @param function
     * @param params
     */
    void applyUpdate(StepFunction function, INDArray params, INDArray updates);

    /**
     * This method applies accumulated updates via given StepFunction
     *
     * @param function
     * @param params
     */
    void applyUpdate(StepFunction function, INDArray params, INDArray updates, double alpha);

    /**
     * This method accepts updates suitable for StepFunction, and accumulates/propagates it across all workers
     *
     * @param array
     */
    void storeUpdate(INDArray array);

    /**
     * This method accepts updates suitable for StepFunction and puts them to the queue, which is used in backpropagation loop
     *
     * PLEASE NOTE: array is expected to be ready for use and match params dimensionality
     *
     * @param array
     */
    void receiveUpdate(INDArray array);

    /**
     * This method resets all accumulated updates (if any)
     */
    void reset();

    /**
     * This method does initialization of given worker wrt Thread-Device Affinity
     */
    void touch();
}

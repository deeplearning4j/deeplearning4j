
package org.nd4j.linalg.learning;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.Serializable;

/**
 * Gradient modifications:
 * Calculates an update and tracks related
 * information for gradient changes over time
 * for handling updates.
 *
 * @author Adam Gibson
 */
public interface GradientUpdater extends Serializable {

    /**
     * For a give input size (length) array, how big is the internal state?
     * Typically 0, 1 or 2x the input size, depending on the type of updater
     *
     * @param inputSize Length of the input array
     * @return Number of elements in the internal state
     */
    int stateSizeForInputSize(int inputSize);

    /**
     * For the internal updater state (if any): set this to use the provided array.
     * Used during initialization, and when restoring the updater state (after serialization, for example)
     *  @param viewArray    Array (that is a view of a larger array) to use for the state.
     * @param gradientShape
     * @param gradientOrder
     * @param initialize   If true: the updater must initialize the view array. If false: no change to view array contents
     */
    void setStateViewArray(INDArray viewArray, int[] gradientShape, char gradientOrder, boolean initialize);

    /**
     * update(learningRate,momentum)
     *
     * @param args
     */
    void update(Object... args);

    /**
     * Modify the gradient
     * to be an update
     *
     * @param gradient  the gradient to modify
     * @param iteration
     * @return the modified gradient
     */
    INDArray getGradient(INDArray gradient, int iteration);

    /**
     * Get a GradientUpdaterAggregator. The aggregator is used (typically in distributed learning scenarios) to combine
     * separate GradientUpdater instances for different networks (usually by averaging).
     *
     * @param addThis If true: return a GradientUpdaterAggregator with the GradientUpdater already added.
     *                If false: return an empty (uninitialized) GradientUpdaterAggregator
     * @deprecated Use Updater view array functionality instead
     */
    @Deprecated
    GradientUpdaterAggregator getAggregator(boolean addThis);
}

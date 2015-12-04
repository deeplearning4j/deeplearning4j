package org.nd4j.linalg.learning;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.Serializable;

/**
 * Gradient modifications:
 * Calculates an update and tracks related
 * information for gradient changes over time
 * for handling updates.
 *
 *
 * @author Adam Gibson
 */
public interface GradientUpdater extends Serializable {


    /**
     * update(learningRate,momentum)
     * @param args
     */
    void update(Object...args);

    /**
     * Modify the gradient
     * to be an update
     * @param gradient the gradient to modify
     * @param iteration
     * @return the modified gradient
     */
    INDArray getGradient(INDArray gradient, int iteration);

    /** Get a GradientUpdaterAggregator. The aggregator is used (typically in distributed learning scenarios) to combine
     * separate GradientUpdater instances for different networks (usually by averaging).
     * @param addThis If true: return a GradientUpdaterAggregator with the GradientUpdater already added.
     *                If false: return an empty (uninitialized) GradientUpdaterAggregator
     */
    GradientUpdaterAggregator getAggregator(boolean addThis);
}

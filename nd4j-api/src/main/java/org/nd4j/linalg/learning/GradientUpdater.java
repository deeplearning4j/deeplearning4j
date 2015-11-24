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

    /** Given this updater, combine the current state (if any) with the state of the other updaters.
     * For example, average the internal state of the updaters. Typically used in distributed learning scenarios
     * @param updaters Other updaters to combine with this one
     */
    void combineUpdaters(GradientUpdater... updaters);
}

package org.deeplearning4j.nn.api;

import org.deeplearning4j.nn.gradient.Gradient;

import java.io.Serializable;

/**
 * Update the model
 *
 * @author Adam Gibson
 */
public interface Updater extends Serializable {
    /**
     * Updater: updates the model
     *
     * @param layer
     * @param gradient
     * @param iteration
     */
    void update(Layer layer, Gradient gradient, int iteration, int miniBatchSize);

    /** Given this updater, combine the current state (if any) with the state of the other updaters.
     * For example, average the internal state of the updaters.
      * @param other Other updaters to combine with this one
     */
    void combineUpdaters(Updater... other);

}

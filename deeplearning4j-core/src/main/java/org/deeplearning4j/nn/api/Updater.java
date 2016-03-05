package org.deeplearning4j.nn.api;

import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.updater.aggregate.UpdaterAggregator;

import java.io.Serializable;

/**
 * Update the model
 *
 * @author Adam Gibson
 */
public interface Updater extends Serializable, Cloneable {
    /**
     * Updater: updates the model
     *
     * @param layer
     * @param gradient
     * @param iteration
     */
    void update(Layer layer, Gradient gradient, int iteration, int miniBatchSize);

    /** Given this updater, get an UpdaterAggregator that can be used to combine the current state (if any)
     * of this updater with the state of other updaters.
     * Typically used only in distributed learning scenarios (where each copy of the network has an updater
     * that needs to be combined)
      * @param addThis whether to add the Updater to the UpdaterAggregator, or just return an empy aggregator
     */
    UpdaterAggregator getAggregator(boolean addThis);

    Updater clone();
}

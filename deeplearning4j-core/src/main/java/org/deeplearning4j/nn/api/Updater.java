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

}

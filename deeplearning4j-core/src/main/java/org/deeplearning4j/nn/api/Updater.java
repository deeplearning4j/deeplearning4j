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
     * @param layer
     * @param gradient
     * @param  iteration
     */
    void update(Layer layer,Gradient gradient,int iteration);

    /**
     * Apply update for a particular layer
     * @param layer the layer to apply the update for
     * @param gradient the gradient to apply
     */
    void applyUpdate(Layer layer,Gradient gradient);

}

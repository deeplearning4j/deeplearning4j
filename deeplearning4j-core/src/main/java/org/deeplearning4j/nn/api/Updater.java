package org.deeplearning4j.nn.api;

import org.deeplearning4j.nn.gradient.Gradient;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.Serializable;

/**
 * Update the model
 *
 * @author Adam Gibson
 */
public interface Updater extends Serializable, Cloneable {

    /**
     * Set the internal (historical) state view array for this updater
     *
     * @param layer      Layer that this updater belongs to
     * @param viewArray  View array
     * @param initialize Whether to initialize the array or not
     */
    void setStateViewArray(Layer layer, INDArray viewArray, boolean initialize);

    /**
     * @return the view array for this updater
     */
    INDArray getStateViewArray();

    /**
     * Calculate and return the state size for this updater (for the given layer).
     * How many parameters/values does this updater have?
     *
     * @param layer Layer that this updater belongs to
     * @return number of parameters/values in the updater state
     */
    int stateSizeForLayer(Layer layer);

    /**
     * Updater: updates the model
     *
     * @param layer
     * @param gradient
     * @param iteration
     */
    void update(Layer layer, Gradient gradient, int iteration, int miniBatchSize);

    Updater clone();
}

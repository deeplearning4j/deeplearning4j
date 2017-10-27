package org.deeplearning4j.nn.api;

import org.deeplearning4j.nn.gradient.Gradient;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.Serializable;

/**
 * Update the model
 *
 * @author Adam Gibson
 */
public interface Updater extends Serializable {

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
     * Updater: updates the model
     *
     * @param layer      Layer (model) to update. May be a single Layer, a MultiLayerNetwork or ComputationGraph
     * @param gradient   Gradient for model
     * @param iteration  Current iteration count
     * @param epoch      Current epoch count
     * @param isPretrain True if currently doing layerwise pretraining, false otherwise
     */
    void update(Layer layer, Gradient gradient, int iteration, int epoch, int miniBatchSize, boolean isPretrain);
}

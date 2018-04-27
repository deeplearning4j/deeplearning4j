package org.deeplearning4j.nn.api;

import org.deeplearning4j.nn.gradient.Gradient;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;

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
     * @param layer
     * @param gradient
     * @param iteration
     */
    void update(Layer layer, Gradient gradient, int iteration, int epoch, int miniBatchSize, LayerWorkspaceMgr workspaceMgr);
}

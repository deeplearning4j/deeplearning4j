package org.deeplearning4j.nn.api;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Map;

/**
 * Trainable: an interface common to Layers and GraphVertices that have trainable parameters
 *
 * @author Alex Black
 */
public interface Trainable {

    /**
     * @return Training configuration
     */
    TrainingConfig getConfig();

    /**
     * @return Number of parameters
     */
    int numParams();

    /**
     * @return 1d parameter vector
     */
    INDArray params();

    /**
     * @param backpropOnly If true: return only parameters that are not exclusively used for layerwise pretraining
     * @return Parameter table
     */
    Map<String,INDArray> paramTable(boolean backpropOnly);

    /**
     * @return 1D gradients view array
     */
    INDArray getGradientsViewArray();

}

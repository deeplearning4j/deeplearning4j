package org.deeplearning4j.nn.gradient;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.Serializable;
import java.util.Map;

/**
 * Generic gradient
 *
 * @author Adam Gibson
 */
public interface Gradient extends Serializable {

    /**
     * Gradient look up table
     * @return the gradient look up table
     */
    Map<String,INDArray> gradientLookupTable();

    /**
     * The ful gradient as one flat vector
     * @return
     */
    INDArray gradient();

    /**
     * Clear residual parameters (useful for returning a gradient and then clearing old objects)
     */
    void clear();

}

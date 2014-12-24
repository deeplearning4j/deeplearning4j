package org.deeplearning4j.nn.gradient;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.Serializable;

/**
 * Generic gradient
 *
 * @author Adam Gibson
 */
public interface Gradient extends Serializable {

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

package org.deeplearning4j.nn.gradient;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.Serializable;
import java.util.List;
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
    Map<String,INDArray> gradientForVariable();
    /**
     * The full gradient as one flat vector
     * @return
     */
    INDArray gradient(List<String> order);

    /**
     * The full gradient as one flat vector
     * @return
     */
    INDArray gradient();

    /**
     * Clear residual parameters (useful for returning a gradient and then clearing old objects)
     */
    void clear();

    /**
     * The gradient for the given variable
     * @param variable the variable to get the gradient for
     * @return the gradient for the given variable or null
     */
    INDArray getGradientFor(String variable);


}

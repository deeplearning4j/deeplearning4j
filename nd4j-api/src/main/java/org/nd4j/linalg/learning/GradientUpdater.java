package org.nd4j.linalg.learning;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.Serializable;

/**
 * Gradient modifications:
 * Calculates an update and tracks related
 * information for gradient changes over time
 * for handling updates.
 *
 *
 * @author Adam Gibson
 */
public interface GradientUpdater extends Serializable {


    /**
     * Modify the gradient
     * to be an update
     * @param gradient the gradient to modify
     * @param iteration
     * @return the modified gradient
     */
    INDArray getGradient(INDArray gradient, int iteration);

    /**
     * Update learningRate
     * @param learningRate
     */
    void setLearningRate(double learningRate);

    /**
     * Update momentum
     * @param momentum
     */
    void setMomentum(double momentum);

}

package org.nd4j.linalg.activations;

/**
 * Interface for loss functions
 * Implement this for an element-wise custom activation function
 * Use ActivationLayer for custom activations with learn-able parameters
 * @author Susan Eraly
 */

import org.apache.commons.math3.util.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.Serializable;

public interface IActivation extends Serializable {

    //parameter array is a row vector
    //      def vals,
    //      initialization scheme
    //      update scheme
    //learnable
    //shared in a layer or specific to a unit

    /**
     * Computes the activation for each element in the ndarray and returns an ndarray of the same size
     *
     * @param in          Input ndarray to transform
     * @param training    Activation functions can behave differently during training and test
     * @return Transformed ndarray, same size as in
     */
    INDArray computeActivation(INDArray in,boolean training);

    /**
     * Computes the gradient of the activation with respect to the input array
     *
     * @param in          Input ndarray to calculate the gradient of
     * @return Gradient of the activation wrt in
     * Of the same size as in for an elementwise transform
     * Different shape if not an element wise transform like softmax
     */
    INDArray computeGradient(INDArray in);

    Pair<INDArray, INDArray> computeGradientAndActivation(INDArray in);


}

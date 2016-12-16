package org.nd4j.linalg.activations;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.Serializable;
import java.util.List;

/**
 * Interface for implementing custom activation functions
 * @author Susan Eraly
 */
public interface IActivation extends Serializable {

    /**
     * Carry out activation function on "in" and write in place to "activation".
     * "in" and "activation" will be of the same shape
     * Can support separate behaviour during test
     * @param in
     * @param activation
     * @param training
     */
    void setActivation(INDArray in, INDArray activation, boolean training);

    /**
     * Value of the partial derivative of the activation function at "in" with respect to the input (linear transformation with weights and biases, sometimes referred to as the preout)
     * Write in place to "gradient"
     * "in" and "gradient" will be of the same shape
     * @param in
     * @param gradient
     */
    void setGradient(INDArray in, INDArray gradient);

    /**
     * Do setActivation and setGradient in one go
     * @param in
     * @param activation
     * @param gradient
     */
    void setActivationAndGradient(INDArray in, INDArray activation, INDArray gradient);

    /**
     * Only required if activation function has learnable params
     * Should really be a static method
     *
     * @return The total number of learnable params for this activation function.
     */
    int getNumParams();

    /**
     * Only required if activation function has learnable params
     * Should really be a static method
     *
     * dl4j terminology around learnable parameters:
     *
     *  shared - if all activation nodes in a layer share a parameter
     *  shaded - if the parameter is duplicated for each activation node
     *
     *  If not shared or sharded, they will be shared across certain axes
     *  which if not specified will default to channels for images and features for time series
     *
     * @return A boolean array the length of the number of learnable parameters where each entry specifies if a parameter
     * is shared or not. The ordering here is expected to be consistent across isSharedParam and isShardedParam
     * eg. numofParams = 2; isSharedParam() returns [true, false] indicating the first param is shared and the second is not.
     */
    boolean [] isSharedParam();

    boolean[] isShardedParam();

    double [] getDefaultParamVals();

    INDArray initParam(int paramIndex, int [] ofShape);

    void setParams(double [] paramsShared, List<INDArray> paramsSharded);

    void setGradientParam(INDArray in, int paramIndex, INDArray gradient);

    int [] getShardAcrossDim();

}

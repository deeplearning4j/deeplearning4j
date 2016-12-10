package org.nd4j.linalg.activations;

/**
 * Interface for loss functions
 * Created by susaneraly on 12/1/16.
 */

import org.apache.commons.math3.util.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.Serializable;

public interface IActivation extends Serializable {

    /*
        Computes the transformation,
         from INDArray in (preout, after linear transformation of activations from previous layer)
                to activation
         INDArray in is the size of the layer

         Different behaviour for training and testing possibly
     */
    INDArray computeActivation(INDArray in, boolean training);


    /*
        Computes the gradient wrt to the preout
        d(activation)/d(preout)
     */
    INDArray computeGradient(INDArray in);

    Pair<INDArray, INDArray> computeGradientAndActivation(INDArray in);

    /*
        Number of params in activation function
     */
    int numParams();

    /*
        Set the parameters
            Different behaviour after initialization?
     */
    void setParamsViewArray(INDArray params, boolean initialize);

    /*
        If the params are learn-able they will be updated during backprop
        This will return dC/dparam which will be used by the updater to update the params with setParamsViewArray

        dC/dparam will be of size numParamsx1

        dC/dparam = (dC/dactivation)*(dactivation/dparam)
        dC/dactivation should be found as part of back prop to update the weights too

     */
    void setBackpropViewArray(INDArray params);

    String toString();
}

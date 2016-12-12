package org.nd4j.linalg.activations;

/**
 * Interface for loss functions
 * Created by susaneraly on 12/1/16.
 */

import org.apache.commons.math3.util.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.Serializable;

public interface IActivation extends Serializable {

    int getNumParams(); //should return the number of params, initialized and saved with the layer

    void setNumParams(int numParams); //should return the number of params, initialized and saved with the layer

    INDArray computeActivation(INDArray in,boolean training);

    INDArray computeGradient(INDArray in);

    Pair<INDArray, INDArray> computeGradientAndActivation(INDArray in);

    String toString();

    boolean initCalled();

    void setParamsViewArray(INDArray paramView);

    void setBackpropViewArray(INDArray in, INDArray gradientView);


}

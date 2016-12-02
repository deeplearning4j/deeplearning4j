package org.nd4j.linalg.activations;

/**
 * Interface for loss functions
 * Created by susaneraly on 12/1/16.
 */

import org.apache.commons.math3.util.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.Serializable;

public interface IActivation extends Serializable {

    INDArray computeActivation(INDArray in);

    INDArray computeGradient(INDArray in);

    Pair<INDArray, INDArray> computeGradientAndActivation(INDArray in);

}

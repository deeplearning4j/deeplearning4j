package org.nd4j.linalg.activations;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.Serializable;
import java.util.List;

/**
 * Interface for implementing custom activation functions
 * @author Susan Eraly
 */
public interface IActivation extends Serializable {

    void setActivation(INDArray in, INDArray activation, boolean training);

    void setGradient(INDArray in, INDArray gradient);

    void setActivationAndGradient(INDArray in, INDArray activation, INDArray gradient);

    int getNumParams();

    boolean [] isSharedParam();

    boolean[] isShardedParam();

    double [] getDefaultParamVals();

    INDArray initParam(int paramIndex, int [] ofShape);

    void setParams(double [] paramsShared, List<INDArray> paramsSharded);

    void setGradientParam(INDArray in, int paramIndex, INDArray gradient);

    int [] getShardAcrossDim();

}

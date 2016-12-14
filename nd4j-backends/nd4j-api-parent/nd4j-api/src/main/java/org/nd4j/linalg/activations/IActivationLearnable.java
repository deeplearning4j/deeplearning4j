package org.nd4j.linalg.activations;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.List;

/**
 * Created by susaneraly on 12/13/16.
 */
public interface IActivationLearnable extends IActivation {

    int getNumParams();

    boolean [] isSharedParam();

    boolean[] isShardedParam();

    double [] getDefaultParamVals();

    INDArray initParam(int paramIndex, int [] ofShape);

    void setParams(double [] paramsShared, List<INDArray> paramsSharded);

    void setGradientParam(INDArray in, int paramIndex, INDArray gradient);

    int [] getShardAcrossDim();

}

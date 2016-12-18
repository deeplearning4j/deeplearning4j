package org.nd4j.linalg.activations;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Created by Alex on 18/12/2016.
 */
public abstract class BaseActivationFunction implements IActivation {

    @Override
    public int numParams(int inputSize){
        return 0;
    }

    @Override
    public void setParametersViewArray(INDArray viewArray){
        //No op
    }

    @Override
    public INDArray getParametersViewArray(){
        return null;
    }

}

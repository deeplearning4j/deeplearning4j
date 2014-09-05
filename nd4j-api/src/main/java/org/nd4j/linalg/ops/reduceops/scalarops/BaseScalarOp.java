package org.nd4j.linalg.ops.reduceops.scalarops;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 *
 * Abstract class for scalar operations
 * @author Adam Gibson
 */
public abstract class BaseScalarOp implements ScalarOp {

    protected float startingValue;


    public BaseScalarOp(float startingValue) {
        this.startingValue = startingValue;
    }



    @Override
    public Float apply(INDArray input) {
        INDArray doNDArray = input.isVector() ? input : input.linearView();
        float start = startingValue;
        for(int i = 0; i < doNDArray.length(); i++)
            start = accumulate(doNDArray,i,start);
        return start;
    }

    public abstract float accumulate(INDArray arr,int i,float soFar);


}

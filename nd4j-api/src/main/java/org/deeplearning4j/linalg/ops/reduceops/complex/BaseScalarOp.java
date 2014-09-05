package org.deeplearning4j.linalg.ops.reduceops.complex;

import org.deeplearning4j.linalg.api.complex.IComplexNDArray;
import org.deeplearning4j.linalg.api.complex.IComplexNumber;

/**
 *
 * Abstract class for scalar operations
 * @author Adam Gibson
 */
public abstract class BaseScalarOp implements ScalarOp {

    protected IComplexNumber startingValue;


    public BaseScalarOp(IComplexNumber startingValue) {
        this.startingValue = startingValue;
    }


    @Override
    public IComplexNumber apply(IComplexNDArray input) {
        IComplexNDArray doNDArray = input.isVector() ? input : input.linearView();
        IComplexNumber start = startingValue;
        for(int i = 0; i < doNDArray.length(); i++)
            start = accumulate(doNDArray,i,start);
        return start;
    }

    public abstract IComplexNumber accumulate(IComplexNDArray arr,int i,IComplexNumber soFar);


}

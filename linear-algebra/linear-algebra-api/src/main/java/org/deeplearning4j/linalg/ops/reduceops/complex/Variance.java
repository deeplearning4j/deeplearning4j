package org.deeplearning4j.linalg.ops.reduceops.complex;

import org.apache.commons.math3.stat.StatUtils;
import org.deeplearning4j.linalg.api.complex.IComplexNDArray;
import org.deeplearning4j.linalg.api.complex.IComplexNumber;
import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.factory.NDArrays;

/**
 * Return the variance of an ndarray
 *
 * @author Adam Gibson
 */
public class Variance extends BaseScalarOp {

    public Variance() {
        super(NDArrays.createDouble(0,0));
    }




    public IComplexNumber var(IComplexNDArray arr) {
        IComplexNumber mean = new Mean().apply(arr);
        return NDArrays.createDouble(StatUtils.variance(arr.data(), mean.absoluteValue().doubleValue()),0);
    }



    @Override
    public IComplexNumber apply(IComplexNDArray input) {
        return  var(input);
    }

    @Override
    public IComplexNumber accumulate(IComplexNDArray arr, int i, IComplexNumber soFar) {
        return NDArrays.createDouble(0,0);
    }
}

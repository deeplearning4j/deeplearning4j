package org.deeplearning4j.spark.impl.common;

import org.apache.spark.AccumulatorParam;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Accumulator for addition of parameters
 * @author Adam Gibson
 */
public class ParamAccumulator implements AccumulatorParam<INDArray> {
    @Override
    public INDArray addAccumulator(INDArray indArray, INDArray t1) {
        return indArray.add(t1);
    }

    @Override
    public INDArray addInPlace(INDArray indArray, INDArray r1) {
        return indArray.addi(r1);
    }

    @Override
    public INDArray zero(INDArray indArray) {
        return Nd4j.zeros(indArray.shape());
    }
}

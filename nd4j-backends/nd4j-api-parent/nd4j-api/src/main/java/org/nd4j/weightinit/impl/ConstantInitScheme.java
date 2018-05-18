package org.nd4j.weightinit.impl;

import lombok.Builder;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.weightinit.BaseWeightInitScheme;
import org.nd4j.weightinit.WeightInit;

/**
 * Initialize the weight to zero.
 * @author Adam Gibson
 */
public class ConstantInitScheme extends BaseWeightInitScheme {
    private double constant;

    @Builder
    public ConstantInitScheme(char order,double constant) {
        super(order);
        this.constant = constant;
    }

    @Override
    public INDArray doCreate(long[] shape, INDArray paramsView) {
        return Nd4j.valueArrayOf(shape,constant);
    }


    @Override
    public WeightInit type() {
        return WeightInit.ZERO;
    }
}

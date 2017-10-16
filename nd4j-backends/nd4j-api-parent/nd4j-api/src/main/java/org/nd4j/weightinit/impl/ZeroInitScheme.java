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
public class ZeroInitScheme extends BaseWeightInitScheme {

    @Builder
    public ZeroInitScheme(char order) {
        super(order);
    }

    @Override
    public INDArray doCreate(int[] shape, INDArray paramsView) {
       return Nd4j.createUninitialized(shape, order()).assign(0.0);
    }


    @Override
    public WeightInit type() {
        return WeightInit.ZERO;
    }
}

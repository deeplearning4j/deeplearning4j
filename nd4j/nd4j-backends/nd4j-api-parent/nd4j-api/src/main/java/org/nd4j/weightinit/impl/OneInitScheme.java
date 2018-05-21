package org.nd4j.weightinit.impl;

import lombok.Builder;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.distribution.Distribution;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.weightinit.BaseWeightInitScheme;
import org.nd4j.weightinit.WeightInit;

/**
 * Initialize the weight to one.
 * @author Adam Gibson
 */
public class OneInitScheme extends BaseWeightInitScheme {

    @Builder
    public OneInitScheme(char order) {
        super(order);
    }

    @Override
    public INDArray doCreate(long[] shape, INDArray paramsView) {
       return Nd4j.createUninitialized(shape, order()).assign(1.0);
    }


    @Override
    public WeightInit type() {
        return WeightInit.ONES;
    }
}

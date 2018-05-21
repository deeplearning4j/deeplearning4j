package org.nd4j.weightinit.impl;

import lombok.Builder;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.distribution.Distribution;
import org.nd4j.weightinit.BaseWeightInitScheme;
import org.nd4j.weightinit.WeightInit;

/**
 * Initialize the weights based on a given {@link Distribution}
 * @author Adam Gibson
 */
public class DistributionInitScheme  extends BaseWeightInitScheme {
    private Distribution distribution;

    @Builder
    public DistributionInitScheme(char order, Distribution distribution) {
        super(order);
        this.distribution = distribution;
    }

    @Override
    public INDArray doCreate(long[] shape, INDArray paramsView) {
        return distribution.sample(shape);
    }


    @Override
    public WeightInit type() {
        return WeightInit.DISTRIBUTION;
    }
}

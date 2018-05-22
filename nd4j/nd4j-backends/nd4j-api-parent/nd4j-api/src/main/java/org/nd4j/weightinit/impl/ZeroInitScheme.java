package org.nd4j.weightinit.impl;

import lombok.Builder;
import lombok.NoArgsConstructor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.weightinit.BaseWeightInitScheme;
import org.nd4j.weightinit.WeightInit;

/**
 * Initialize the weight to zero.
 * @author Adam Gibson
 */
@NoArgsConstructor
public class ZeroInitScheme extends BaseWeightInitScheme {

    @Builder
    public ZeroInitScheme(char order) {
        super(order);
    }

    @Override
    public INDArray doCreate(long[] shape, INDArray paramsView) {
        if(shape == null) {
            throw new ND4JIllegalStateException("Shape must not be null!");
        }
        return Nd4j.createUninitialized(shape, order()).assign(0.0);
    }


    @Override
    public WeightInit type() {
        return WeightInit.ZERO;
    }
}

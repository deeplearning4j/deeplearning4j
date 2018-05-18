package org.nd4j.weightinit.impl;

import lombok.Builder;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.weightinit.BaseWeightInitScheme;
import org.nd4j.weightinit.WeightInit;

import java.util.Arrays;

/**
 * Initialize the weight to one.
 * @author Adam Gibson
 */
public class IdentityInitScheme extends BaseWeightInitScheme {

    @Builder
    public IdentityInitScheme(char order) {
        super(order);
    }

    @Override
    public INDArray doCreate(long[] shape, INDArray paramsView) {
        if(shape.length != 2 || shape[0] != shape[1]){
            throw new IllegalStateException("Cannot use IDENTITY init with parameters of shape "
                    + Arrays.toString(shape) + ": weights must be a square matrix for identity");
        }
        if(order() == Nd4j.order()){
            return Nd4j.eye(shape[0]);
        } else {
            return  Nd4j.createUninitialized(shape, order()).assign(Nd4j.eye(shape[0]));
        }
    }


    @Override
    public WeightInit type() {
        return WeightInit.IDENTITY;
    }
}

package org.nd4j.weightinit.impl;

import lombok.Builder;
import org.apache.commons.math3.util.FastMath;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.weightinit.BaseWeightInitScheme;
import org.nd4j.weightinit.WeightInit;

/**
 * Initialize the weight to:
 * U(-sqrt(6/fanIn), sqrt(6/fanIn)
 *
 * @author Adam Gibson
 */
public class ReluUniformInitScheme extends BaseWeightInitScheme {

    private double fanIn;

    @Builder
    public ReluUniformInitScheme(char order, double fanIn) {
        super(order);
        this.fanIn = fanIn;
    }

    @Override
    public INDArray doCreate(long[] shape, INDArray paramsView) {
        double u = Math.sqrt(6.0 / fanIn);
        return Nd4j.rand(shape, Nd4j.getDistributions().createUniform(-u, u)); //U(-sqrt(6/fanIn), sqrt(6/fanIn)
    }


    @Override
    public WeightInit type() {
        return WeightInit.RELU_UNIFORM;
    }
}

package org.nd4j.weightinit.impl;

import lombok.Builder;
import org.apache.commons.math3.util.FastMath;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.weightinit.BaseWeightInitScheme;
import org.nd4j.weightinit.WeightInit;

/**
 * Initialize the weight to:
 * randn(shape) //N(0, 2/nIn);
 * @author Adam Gibson
 */
public class VarScalingNormalUniformFanOutInitScheme extends BaseWeightInitScheme {

    private double fanOut;

    @Builder
    public VarScalingNormalUniformFanOutInitScheme(char order, double fanOut) {
        super(order);
        this.fanOut = fanOut;
    }

    @Override
    public INDArray doCreate(long[] shape, INDArray paramsView) {
        double scalingFanOut = 3.0 / Math.sqrt(fanOut);
        return Nd4j.rand(shape, Nd4j.getDistributions().createUniform(-scalingFanOut, scalingFanOut));
    }


    @Override
    public WeightInit type() {
        return WeightInit.VAR_SCALING_UNIFORM_FAN_OUT;
    }
}

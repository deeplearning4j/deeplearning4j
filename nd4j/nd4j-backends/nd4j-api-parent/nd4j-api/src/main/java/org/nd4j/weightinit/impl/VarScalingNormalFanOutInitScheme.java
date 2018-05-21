package org.nd4j.weightinit.impl;

import lombok.Builder;
import org.apache.commons.math3.util.FastMath;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.weightinit.BaseWeightInitScheme;
import org.nd4j.weightinit.WeightInit;

/**
 * Initialize the weight to:
 *
 * U / sqrt(fanout)
 * @author Adam Gibson
 */
public class VarScalingNormalFanOutInitScheme extends BaseWeightInitScheme {

    private double fanOut;

    @Builder
    public VarScalingNormalFanOutInitScheme(char order, double fanOut) {
        super(order);
        this.fanOut = fanOut;
    }

    @Override
    public INDArray doCreate(long[] shape, INDArray paramsView) {
        return    Nd4j.randn(order(), shape).divi(FastMath.sqrt(fanOut));
    }


    @Override
    public WeightInit type() {
        return WeightInit.VAR_SCALING_NORMAL_FAN_OUT;
    }
}

package org.nd4j.weightinit.impl;

import lombok.Builder;
import org.apache.commons.math3.util.FastMath;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.weightinit.BaseWeightInitScheme;
import org.nd4j.weightinit.WeightInit;

/**
 * Initialize the weight to:
 * U / fanIn
 * @author Adam Gibson
 */
public class VarScalingNormalFanInInitScheme extends BaseWeightInitScheme {

    private double fanIn;

    @Builder
    public VarScalingNormalFanInInitScheme(char order, double fanIn) {
        super(order);
        this.fanIn = fanIn;
    }

    @Override
    public INDArray doCreate(long[] shape, INDArray paramsView) {
        return    Nd4j.randn(order(), shape).divi(FastMath.sqrt(fanIn));
    }


    @Override
    public WeightInit type() {
        return WeightInit.VAR_SCALING_NORMAL_FAN_IN;
    }
}

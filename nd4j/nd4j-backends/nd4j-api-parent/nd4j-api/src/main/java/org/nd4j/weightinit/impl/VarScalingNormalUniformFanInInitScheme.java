package org.nd4j.weightinit.impl;

import lombok.Builder;
import org.apache.commons.math3.util.FastMath;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.weightinit.BaseWeightInitScheme;
import org.nd4j.weightinit.WeightInit;

/**
 * Initialize the weight to:
 * range = = 3.0 / Math.sqrt(fanIn)
 * U(-range,range)
 * @author Adam Gibson
 */
public class VarScalingNormalUniformFanInInitScheme extends BaseWeightInitScheme {

    private double fanIn;

    @Builder
    public VarScalingNormalUniformFanInInitScheme(char order, double fanIn) {
        super(order);
        this.fanIn = fanIn;
    }

    @Override
    public INDArray doCreate(long[] shape, INDArray paramsView) {
        double scalingFanIn = 3.0 / Math.sqrt(fanIn);
        return Nd4j.rand(shape, Nd4j.getDistributions().createUniform(-scalingFanIn, scalingFanIn));
    }


    @Override
    public WeightInit type() {
        return WeightInit.VAR_SCALING_UNIFORM_FAN_IN;
    }
}

package org.nd4j.weightinit.impl;

import lombok.Builder;
import org.apache.commons.math3.util.FastMath;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.weightinit.BaseWeightInitScheme;
import org.nd4j.weightinit.WeightInit;

/**
 * Initialize the weight to:
 * range = = 3.0 / Math.sqrt((fanIn + fanOut) / 2)
 * U(-range,range)
 * @author Adam Gibson
 */
public class VarScalingUniformFanAvgInitScheme extends BaseWeightInitScheme {

    private double fanIn;
    private double fanOut;

    @Builder
    public VarScalingUniformFanAvgInitScheme(char order, double fanIn, double fanOut) {
        super(order);
        this.fanIn = fanIn;
        this.fanOut = fanOut;
    }


    @Override
    public INDArray doCreate(long[] shape, INDArray paramsView) {
        double scalingFanAvg = 3.0 / Math.sqrt((fanIn + fanOut) / 2);
        return Nd4j.rand(shape, Nd4j.getDistributions().createUniform(-scalingFanAvg, scalingFanAvg));
    }


    @Override
    public WeightInit type() {
        return WeightInit.VAR_SCALING_UNIFORM_FAN_AVG;
    }
}

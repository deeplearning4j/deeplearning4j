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
public class XavierFanInInitScheme extends BaseWeightInitScheme {

    private double fanIn;

    @Builder
    public XavierFanInInitScheme(char order, double fanIn) {
        super(order);
        this.fanIn = fanIn;
    }

    @Override
    public INDArray doCreate(long[] shape, INDArray paramsView) {
        return   Nd4j.randn(order(), shape).divi(FastMath.sqrt(fanIn));
    }


    @Override
    public WeightInit type() {
        return WeightInit.XAVIER_FAN_IN;
    }
}

package org.nd4j.weightinit.impl;

import lombok.Builder;
import org.apache.commons.math3.util.FastMath;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.weightinit.BaseWeightInitScheme;
import org.nd4j.weightinit.WeightInit;

/**
 * Initialize the weight to:
 * range =  4.0 * Math.sqrt(6.0 / (fanIn + fanOut))
 * U(-range,range)
 * @author Adam Gibson
 */
public class SigmoidUniformInitScheme extends BaseWeightInitScheme {

    private double fanIn;
    private double fanOut;

    @Builder
    public SigmoidUniformInitScheme(char order, double fanIn,double fanOut) {
        super(order);
        this.fanIn = fanIn;
        this.fanOut = fanOut;
    }


    @Override
    public INDArray doCreate(long[] shape, INDArray paramsView) {
        double r = 4.0 * Math.sqrt(6.0 / (fanIn + fanOut));
        return Nd4j.rand(shape, Nd4j.getDistributions().createUniform(-r, r));
    }


    @Override
    public WeightInit type() {
        return WeightInit.SIGMOID_UNIFORM;
    }
}

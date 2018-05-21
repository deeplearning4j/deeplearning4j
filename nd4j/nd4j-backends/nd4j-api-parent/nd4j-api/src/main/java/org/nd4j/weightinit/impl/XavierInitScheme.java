package org.nd4j.weightinit.impl;

import lombok.Builder;
import org.apache.commons.math3.util.FastMath;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.weightinit.BaseWeightInitScheme;
import org.nd4j.weightinit.WeightInit;

/**
 * Initialize the weight to:
 * range = = U * sqrt(2.0 / (fanIn + fanOut)
 * @author Adam Gibson
 */
public class XavierInitScheme extends BaseWeightInitScheme {

    private double fanIn;
    private double fanOut;

    @Builder
    public XavierInitScheme(char order, double fanIn, double fanOut) {
        super(order);
        this.fanIn = fanIn;
        this.fanOut = fanOut;
    }


    @Override
    public INDArray doCreate(long[] shape, INDArray paramsView) {
        return Nd4j.randn(order(), shape).muli(FastMath.sqrt(2.0 / (fanIn + fanOut)));
    }


    @Override
    public WeightInit type() {
        return WeightInit.XAVIER;
    }
}

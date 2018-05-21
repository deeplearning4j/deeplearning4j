package org.nd4j.weightinit.impl;

import lombok.Builder;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.weightinit.BaseWeightInitScheme;
import org.nd4j.weightinit.WeightInit;

/**
 * Initialize the weight to:
 * //As per Glorot and Bengio 2010: Uniform distribution U(-s,s) with s = sqrt(6/(fanIn + fanOut))
 * Eq 16: http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
 */
public class XavierUniformInitScheme extends BaseWeightInitScheme {

    private double fanIn;
    private double fanOut;

    @Builder
    public XavierUniformInitScheme(char order, double fanIn, double fanOut) {
        super(order);
        this.fanIn = fanIn;
        this.fanOut = fanOut;
    }


    @Override
    public INDArray doCreate(long[] shape, INDArray paramsView) {
        //As per Glorot and Bengio 2010: Uniform distribution U(-s,s) with s = sqrt(6/(fanIn + fanOut))
        //Eq 16: http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
        double s = Math.sqrt(6.0) / Math.sqrt(fanIn + fanOut);
        return Nd4j.rand(shape, Nd4j.getDistributions().createUniform(-s, s));
    }


    @Override
    public WeightInit type() {
        return WeightInit.XAVIER_UNIFORM;
    }
}

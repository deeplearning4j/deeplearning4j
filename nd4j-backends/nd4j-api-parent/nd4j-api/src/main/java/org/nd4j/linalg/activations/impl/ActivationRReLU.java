package org.nd4j.linalg.activations.impl;

import org.apache.commons.math3.util.Pair;
import org.nd4j.linalg.activations.BaseActivationFunction;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.RectifedLinear;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;

/**
 * Created by susaneraly on 12/10/16.
 */
public class ActivationRReLU extends BaseActivationFunction {

    public static final double DEFAULT_L = 1.0 / 8;
    public static final double DEFAULT_U = 1.0 / 3;

    private double l,u;
    private transient INDArray alpha; //don't need to write to json, when streaming

    public ActivationRReLU() {
        this(DEFAULT_L, DEFAULT_U);
    }

    public ActivationRReLU(double l, double u) {
        if(l > u){
            throw new IllegalArgumentException("Cannot have lower value (" + l + ") greater than upper (" + u + ")");
        }
        this.l = l;
        this.u = u;
    }

    @Override
    public INDArray getActivation(INDArray in, boolean training) {
        if (training) {
            this.alpha = Nd4j.rand(in.shape(), l, u, Nd4j.getRandom());
            INDArray inTimesAlpha = in.mul(alpha);
            BooleanIndexing.replaceWhere(in, inTimesAlpha, Conditions.lessThan(0));
        } else {
            this.alpha = null;
            double a = 0.5*(l+u);
            return Nd4j.getExecutioner().execAndReturn(new RectifedLinear(in, a));
        }

        return in;
    }

    @Override
    public Pair<INDArray,INDArray> backprop(INDArray in, INDArray epsilon) {

        INDArray dLdz = Nd4j.ones(in.shape());
        BooleanIndexing.replaceWhere(dLdz, alpha, Conditions.greaterThanOrEqual(0.0));
        dLdz.muli(epsilon);

        return new Pair<>(dLdz, null);
    }

    @Override
    public String toString() {
        return "rrelu(l=" + l + ", u=" + u + ")";
    }

}

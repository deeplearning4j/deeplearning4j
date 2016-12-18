package org.nd4j.linalg.activations.impl;

import org.apache.commons.math3.util.Pair;
import org.nd4j.linalg.activations.BaseActivationFunction;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.Sigmoid;
import org.nd4j.linalg.api.ops.impl.transforms.SigmoidDerivative;
import org.nd4j.linalg.api.ops.impl.transforms.SoftSign;
import org.nd4j.linalg.api.ops.impl.transforms.SoftSignDerivative;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Created by susaneraly on 12/17/16.
 */
public class ActivationSoftSign extends BaseActivationFunction {

    @Override
    public INDArray getActivation(INDArray in, boolean training) {
        Nd4j.getExecutioner().execAndReturn(new SoftSign(in));
        return in;
    }

    @Override
    public Pair<INDArray,INDArray> backprop(INDArray in, INDArray epsilon) {
        INDArray dLdz = Nd4j.getExecutioner().execAndReturn(new SoftSignDerivative(in));
        dLdz.muli(epsilon);
        return new Pair<>(dLdz, null);
    }

    @Override
    public String toString() {
        return "softsign";
    }
}

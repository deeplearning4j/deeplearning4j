package org.nd4j.linalg.activations.impl;

import lombok.EqualsAndHashCode;
import lombok.Getter;
import org.nd4j.linalg.api.ops.impl.transforms.gradient.LeakyReLUDerivative;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.linalg.activations.BaseActivationFunction;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.LeakyReLU;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Leaky RELU
 * f(x) = max(0, x) + alpha * min(0, x)
 * alpha defaults to 0.01
 */
@EqualsAndHashCode
@Getter
public class ActivationLReLU extends BaseActivationFunction {
    public static final double DEFAULT_ALPHA = 0.01;

    private double alpha = DEFAULT_ALPHA;

    public ActivationLReLU() {
        this(DEFAULT_ALPHA);
    }

    public ActivationLReLU(double alpha) {
        this.alpha = alpha;
    }

    @Override
    public INDArray getActivation(INDArray in, boolean training) {
        Nd4j.getExecutioner().execAndReturn(new LeakyReLU(in, alpha));
        return in;
    }

    @Override
    public Pair<INDArray, INDArray> backprop(INDArray in, INDArray epsilon) {
        assertShape(in, epsilon);
        INDArray dLdz = Nd4j.getExecutioner().execAndReturn(new LeakyReLUDerivative(in, alpha));
        dLdz.muli(epsilon);
        return new Pair<>(dLdz, null);
    }

    @Override
    public String toString() {
        return "leakyrelu(a=" + alpha + ")";
    }
}

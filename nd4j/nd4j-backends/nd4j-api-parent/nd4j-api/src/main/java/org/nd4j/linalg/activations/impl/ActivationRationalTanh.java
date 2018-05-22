package org.nd4j.linalg.activations.impl;

import lombok.EqualsAndHashCode;
import lombok.Getter;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.linalg.activations.BaseActivationFunction;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.RationalTanh;
import org.nd4j.linalg.api.ops.impl.transforms.gradient.RationalTanhDerivative;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Rational tanh approximation
 * From https://arxiv.org/pdf/1508.01292v3
 *
 * f(x) = 1.7159 * tanh(2x/3)
 * where tanh is approximated as follows,
 * tanh(y) ~ sgn(y) * { 1 - 1/(1+|y|+y^2+1.41645*y^4)}
 *
 * Underlying implementation is in native code
 */
@EqualsAndHashCode
@Getter
public class ActivationRationalTanh extends BaseActivationFunction {

    @Override
    public INDArray getActivation(INDArray in, boolean training) {
        Nd4j.getExecutioner().execAndReturn(new RationalTanh(in));
        return in;
    }

    @Override
    public Pair<INDArray, INDArray> backprop(INDArray in, INDArray epsilon) {
        INDArray dLdz = Nd4j.getExecutioner().execAndReturn(new RationalTanhDerivative(in));
        dLdz.muli(epsilon);
        return new Pair<>(dLdz, null);
    }

    @Override
    public String toString() {
        return "rationaltanh";
    }
}

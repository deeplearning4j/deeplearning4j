package org.nd4j.linalg.activations.impl;

import lombok.EqualsAndHashCode;
import lombok.Getter;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.linalg.activations.BaseActivationFunction;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.HardSigmoid;
import org.nd4j.linalg.api.ops.impl.transforms.gradient.HardSigmoidDerivative;
import org.nd4j.linalg.factory.Nd4j;

/**
 * f(x) = min(1, max(0, 0.2*x + 0.5))
 */
@EqualsAndHashCode
@Getter
public class ActivationHardSigmoid extends BaseActivationFunction {

    @Override
    public INDArray getActivation(INDArray in, boolean training) {
        Nd4j.getExecutioner().execAndReturn(new HardSigmoid(in));
        return in;
    }

    @Override
    public Pair<INDArray, INDArray> backprop(INDArray in, INDArray epsilon) {
        INDArray dLdz = Nd4j.getExecutioner().execAndReturn(new HardSigmoidDerivative(in));
        dLdz.muli(epsilon);
        return new Pair<>(dLdz, null);
    }

    @Override
    public String toString() {
        return "hardsigmoid";
    }
}

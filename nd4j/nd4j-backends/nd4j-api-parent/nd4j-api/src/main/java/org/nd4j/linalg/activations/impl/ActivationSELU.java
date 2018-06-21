package org.nd4j.linalg.activations.impl;

import lombok.EqualsAndHashCode;
import lombok.Getter;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.linalg.activations.BaseActivationFunction;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.SELU;
import org.nd4j.linalg.api.ops.impl.transforms.gradient.SELUDerivative;
import org.nd4j.linalg.factory.Nd4j;

/**
 * https://arxiv.org/pdf/1706.02515.pdf
 */
@EqualsAndHashCode
@Getter
public class ActivationSELU extends BaseActivationFunction {

    @Override
    public INDArray getActivation(INDArray in, boolean training) {
        Nd4j.getExecutioner().execAndReturn(new SELU(in));
        return in;
    }

    @Override
    public Pair<INDArray, INDArray> backprop(INDArray in, INDArray epsilon) {
        assertShape(in, epsilon);
        INDArray dLdz = Nd4j.getExecutioner().execAndReturn(new SELUDerivative(in));
        dLdz.muli(epsilon);
        return new Pair<>(dLdz, null);
    }

    @Override
    public String toString() {
        return "selu";
    }

}

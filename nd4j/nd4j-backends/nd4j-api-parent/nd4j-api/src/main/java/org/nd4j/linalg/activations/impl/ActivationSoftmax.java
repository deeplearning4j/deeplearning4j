package org.nd4j.linalg.activations.impl;

import lombok.EqualsAndHashCode;
import lombok.Getter;
import org.nd4j.linalg.activations.BaseActivationFunction;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.OldSoftMax;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;

/**
 * f_i(x) = exp(x_i - shift) / sum_j exp(x_j - shift)
 * where shift = max_i(x_i)
 */
@EqualsAndHashCode
@Getter
public class ActivationSoftmax extends BaseActivationFunction {

    @Override
    public INDArray getActivation(INDArray in, boolean training) {
        Nd4j.getExecutioner().execAndReturn(new OldSoftMax(in));
        return in;
    }

    @Override
    public Pair<INDArray, INDArray> backprop(INDArray in, INDArray epsilon) {
        assertShape(in, epsilon);
        INDArray out = Nd4j.getExecutioner().execAndReturn(new OldSoftMax(in));
        INDArray x = out.mul(epsilon).sum(1);
        INDArray dLdz = out.mul(epsilon.subColumnVector(x));
        return new Pair<>(dLdz, null);
    }

    @Override
    public String toString() {
        return "softmax";
    }

}

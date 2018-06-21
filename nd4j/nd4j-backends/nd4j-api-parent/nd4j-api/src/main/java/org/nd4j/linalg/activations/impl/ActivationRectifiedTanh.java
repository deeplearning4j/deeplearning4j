package org.nd4j.linalg.activations.impl;

import lombok.EqualsAndHashCode;
import lombok.Getter;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.linalg.activations.BaseActivationFunction;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.RectifiedTanh;
import org.nd4j.linalg.api.ops.impl.transforms.gradient.RectifiedTanhDerivative;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Rectified tanh
 *
 * Essentially max(0, tanh(x))
 *
 * Underlying implementation is in native code
 */
@EqualsAndHashCode
@Getter
public class ActivationRectifiedTanh extends BaseActivationFunction {

    @Override
    public INDArray getActivation(INDArray in, boolean training) {
        Nd4j.getExecutioner().execAndReturn(new RectifiedTanh(in));
        return in;
    }

    @Override
    public Pair<INDArray, INDArray> backprop(INDArray in, INDArray epsilon) {
        assertShape(in, epsilon);
        INDArray dLdz = Nd4j.getExecutioner().execAndReturn(new RectifiedTanhDerivative(in));
        dLdz.muli(epsilon);
        return new Pair<>(dLdz, null);
    }

    @Override
    public String toString() {
        return "rectifiedtanh";
    }
}

package org.nd4j.linalg.activations.impl;

import lombok.EqualsAndHashCode;
import lombok.Getter;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.linalg.activations.BaseActivationFunction;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * f(x) = x
 */
@EqualsAndHashCode
@Getter
public class ActivationIdentity extends BaseActivationFunction {

    @Override
    public INDArray getActivation(INDArray in, boolean training) {
        //no op
        return in;
    }

    @Override
    public Pair<INDArray, INDArray> backprop(INDArray in, INDArray epsilon) {
        assertShape(in, epsilon);
        return new Pair<>(epsilon, null);
    }

    @Override
    public String toString() {
        return "identity";
    }

}

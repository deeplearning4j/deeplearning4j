package org.nd4j.linalg.activations.impl;

import lombok.EqualsAndHashCode;
import lombok.Getter;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.linalg.activations.BaseActivationFunction;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.HardTanh;
import org.nd4j.linalg.api.ops.impl.transforms.gradient.HardTanhDerivative;
import org.nd4j.linalg.factory.Nd4j;

/**
 *        ⎧  1, if x >  1
 f(x) =   ⎨ -1, if x < -1
          ⎩  x, otherwise
 */
@EqualsAndHashCode
@Getter
public class ActivationHardTanH extends BaseActivationFunction {

    @Override
    public INDArray getActivation(INDArray in, boolean training) {
        Nd4j.getExecutioner().execAndReturn(new HardTanh(in));
        return in;
    }

    @Override
    public Pair<INDArray, INDArray> backprop(INDArray in, INDArray epsilon) {
        assertShape(in, epsilon);
        INDArray dLdz = Nd4j.getExecutioner().execAndReturn(new HardTanhDerivative(in));
        dLdz.muli(epsilon);
        return new Pair<>(dLdz, null);
    }

    @Override
    public String toString() {
        return "hardtanh";
    }
}

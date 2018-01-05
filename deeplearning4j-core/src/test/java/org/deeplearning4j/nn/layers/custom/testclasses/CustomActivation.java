package org.deeplearning4j.nn.layers.custom.testclasses;

import lombok.EqualsAndHashCode;
import org.nd4j.linalg.activations.BaseActivationFunction;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.Pair;

/**
 * Created by Alex on 19/12/2016.
 */
@EqualsAndHashCode
public class CustomActivation extends BaseActivationFunction implements IActivation {
    @Override
    public INDArray getActivation(INDArray in, boolean training) {
        return in;
    }

    @Override
    public Pair<INDArray, INDArray> backprop(INDArray in, INDArray epsilon) {
        return new Pair<>(in.muli(epsilon), null);
    }
}

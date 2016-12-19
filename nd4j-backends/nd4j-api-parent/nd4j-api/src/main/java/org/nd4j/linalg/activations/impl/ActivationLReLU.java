package org.nd4j.linalg.activations.impl;

import lombok.EqualsAndHashCode;
import lombok.Getter;
import org.apache.commons.math3.util.Pair;
import org.nd4j.linalg.activations.BaseActivationFunction;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.LeakyReLU;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.serde.RowVectorDeserializer;
import org.nd4j.linalg.lossfunctions.serde.RowVectorSerializer;
import org.nd4j.shade.jackson.annotation.JsonInclude;
import org.nd4j.shade.jackson.databind.annotation.JsonDeserialize;
import org.nd4j.shade.jackson.databind.annotation.JsonSerialize;

/**
 * Leaky RELU
 * f(x) = max(0, x) + alpha * min(0, x)
 * alpha defaults to 0.01
 */
@EqualsAndHashCode
@Getter
public class ActivationLReLU extends BaseActivationFunction {
    private double alpha;

    public ActivationLReLU() {
        this.alpha = 0.01;
    }

    public ActivationLReLU (double alpha) {
        this.alpha = alpha;
    }

    @Override
    public INDArray getActivation(INDArray in, boolean training) {
        Nd4j.getExecutioner().execAndReturn(new LeakyReLU(in,alpha));
        return in;
    }

    @Override
    public Pair<INDArray,INDArray> backprop(INDArray in, INDArray epsilon) {
        INDArray dLdz = Nd4j.getExecutioner().execAndReturn(new LeakyReLU(in,alpha).derivative());
        dLdz.muli(epsilon);
        return new Pair<>(dLdz, null);
    }

    @Override
    public String toString() {
        return "leakyrelu(a=" + alpha + ")";
    }
}

package org.nd4j.linalg.api.ops.impl.layers.recurrent.weights;

import lombok.Builder;
import lombok.Data;
import lombok.EqualsAndHashCode;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.SRU;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.SRUCell;

/**
 * The weight configuration of a SRU layer.  For {@link SRU} and {@link SRUCell}.
 *
 */
@EqualsAndHashCode(callSuper = true)
@Data
@Builder
public class SRUWeights extends RNNWeights {

    /**
     * Weights, with shape [inSize, 3*inSize].
     */
    private SDVariable weights;

    private INDArray iWeights;

    /**
     * Biases, with shape [2*inSize].
     */
    private SDVariable bias;

    private INDArray iBias;

    @Override
    public SDVariable[] args() {
        return new SDVariable[]{weights, bias};
    }

    @Override
    public INDArray[] arrayArgs() {
        return new INDArray[]{iWeights, iBias};
    }
}

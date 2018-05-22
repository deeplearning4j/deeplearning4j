package org.nd4j.linalg.lossfunctions.impl;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Negative log likelihood loss function
 * <p>
 * In practice, this is implemented as an alias for {@link LossMCXENT} due to the mathematical equivalence
 */
public class LossNegativeLogLikelihood extends LossMCXENT {

    public LossNegativeLogLikelihood() {}

    public LossNegativeLogLikelihood(INDArray weights) {
        super(weights);
    }

    /**
     * The opName of this function
     *
     * @return
     */
    @Override
    public String name() {
        return toString();
    }

    @Override
    public String opName() {
        return "lossnegativeloglikelihood";
    }

    @Override
    public String toString() {
        return "LossNegativeLogLikelihood()";
    }
}

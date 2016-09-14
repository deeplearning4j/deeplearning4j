package org.nd4j.linalg.lossfunctions.impl;

/**
 * Negative log likelihood loss function
 * <p>
 * In practice, this is implemented as an alias for {@link LossMCXENT} due to the mathematical equivalence
 */
public class LossNegativeLogLikelihood extends LossMCXENT {


    @Override
    public String toString() {
        return "LossNegativeLogLikelihood()";
    }
}

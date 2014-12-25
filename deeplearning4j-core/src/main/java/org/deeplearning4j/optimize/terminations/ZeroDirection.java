package org.deeplearning4j.optimize.terminations;

import org.deeplearning4j.optimize.api.TerminationCondition;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;

/**
 * Absolute magnitude of gradient is 0
 * @author Adam Gibson
 */
public class ZeroDirection implements TerminationCondition {
    @Override
    public boolean terminate(double cost, double oldCost, Object[] otherParams) {
        INDArray gradient = (INDArray) otherParams[0];
        return Transforms.abs(gradient).sum(Integer.MAX_VALUE).getDouble(0) == 0;
    }
}

package org.deeplearning4j.optimize.terminations;

import org.deeplearning4j.optimize.api.TerminationCondition;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Terminate if the norm2 of the gradient is < a certain tolerance
 */
public class Norm2Termination implements TerminationCondition {
    private double gradientTolerance = 1e-3;

    public Norm2Termination(double gradientTolerance) {
        this.gradientTolerance = gradientTolerance;
    }

    @Override
    public boolean terminate(double cost, double oldCost, Object[] otherParams) {
        INDArray line = (INDArray) otherParams[0];
        double norm2 = line.norm2(Integer.MAX_VALUE).getDouble(0);
        return norm2 < gradientTolerance;
    }
}

package org.deeplearning4j.optimize.stepfunctions;

import org.deeplearning4j.optimize.api.StepFunction;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Created by agibsonccc on 11/1/14.
 */
public class NegativeDefaultStepFunction implements StepFunction {
    @Override
    public void step(INDArray x, INDArray line, Object[] params) {
        double alam = (double) params[0];
        double oldAlam = (double) params[1];
        x.subi(line.mul(alam - oldAlam));
    }

    @Override
    public void step(INDArray x, INDArray line) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void step() {
       throw new UnsupportedOperationException();
    }
}

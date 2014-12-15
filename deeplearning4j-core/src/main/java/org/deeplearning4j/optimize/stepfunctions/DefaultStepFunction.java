package org.deeplearning4j.optimize.stepfunctions;

import org.deeplearning4j.optimize.api.StepFunction;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Default step function
 * @author Adam Gibson
 */
public class DefaultStepFunction implements StepFunction {
    @Override
    public void step(INDArray x, INDArray line, Object[] params) {
        double alam = (double) params[0];
        double oldAlam = (double) params[1];
        x.addi(line.muli(alam - oldAlam));
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
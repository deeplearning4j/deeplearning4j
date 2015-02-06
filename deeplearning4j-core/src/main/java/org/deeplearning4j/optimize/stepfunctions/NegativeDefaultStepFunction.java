package org.deeplearning4j.optimize.stepfunctions;

import org.deeplearning4j.optimize.api.StepFunction;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Inverse step function
 * @author Adam Gibson
 */
public class NegativeDefaultStepFunction implements StepFunction {
    @Override
    public void step(INDArray x, INDArray line, Object[] params) {
        double alam = (double) params[0];
        double oldAlam = (double) params[1];
        Nd4j.getBlasWrapper().axpy(alam - oldAlam,line,x);
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

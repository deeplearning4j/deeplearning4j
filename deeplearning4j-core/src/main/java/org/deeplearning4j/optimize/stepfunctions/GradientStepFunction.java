package org.deeplearning4j.optimize.stepfunctions;

import org.deeplearning4j.optimize.api.StepFunction;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Normal gradient step function
 * @author Adam Gibson
 */
public class GradientStepFunction implements StepFunction {
    @Override
    public void step(INDArray x, INDArray line, Object[] params) {
        x.addi(line);
    }

    @Override
    public void step(INDArray x, INDArray line) {
        x.addi(line);
    }

    @Override
    public void step() {

    }
}

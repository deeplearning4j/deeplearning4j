package org.deeplearning4j.optimize.stepfunctions;

import org.deeplearning4j.optimize.api.StepFunction;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Default step function
 * @author Adam Gibson
 */
public class DefaultStepFunction implements StepFunction {
    @Override
    public void step(INDArray x, INDArray line, Object[] params) {
        double alam = (double) params[0];
        double oldAlam = (double) params[1];
        if(x.data().dataType() == DataBuffer.DOUBLE) {
            Nd4j.getBlasWrapper().axpy(alam - oldAlam, x, line);
        }
        else {
            float diff = (float) (alam - oldAlam);
            Nd4j.getBlasWrapper().axpy(diff,x,line);

        }
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
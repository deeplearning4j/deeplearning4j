package org.nd4j.linalg.benchmark.linearview.getput;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.benchmark.api.OpRunner;
import org.nd4j.linalg.factory.Nd4j;

/**
 * @author Adam Gibson
 */
public class GetPutScalarLinearViewOpRunner implements OpRunner {
    INDArray arr = Nd4j.create(10000).linearView();
    @Override
    public void runOp() {
        arr.putScalar(1,0);
    }
}

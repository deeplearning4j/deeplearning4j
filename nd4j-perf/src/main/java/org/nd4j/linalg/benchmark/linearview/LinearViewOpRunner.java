package org.nd4j.linalg.benchmark.linearview;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.benchmark.api.OpRunner;
import org.nd4j.linalg.factory.Nd4j;

/**
 * @author Adam Gibson
 */
public class LinearViewOpRunner implements OpRunner {
    INDArray arr = Nd4j.create(1000000);


    @Override
    public void runOp() {
        arr.resetLinearView();
    }
}

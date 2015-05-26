package org.nd4j.linalg.benchmark.scalar;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.benchmark.api.OpRunner;
import org.nd4j.linalg.factory.Nd4j;

/**
 * @author Adam Gibson
 */
public class ScalarOpRunner implements OpRunner {
    INDArray arr = Nd4j.create(10000000);


    @Override
    public void runOp() {
        arr.addi(1);
    }
}

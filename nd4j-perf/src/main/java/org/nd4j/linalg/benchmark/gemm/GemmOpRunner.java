package org.nd4j.linalg.benchmark.gemm;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.benchmark.api.OpRunner;
import org.nd4j.linalg.factory.Nd4j;

/**
 * @author Adam Gibson
 */
public class GemmOpRunner implements OpRunner {
    INDArray arr = Nd4j.create(1000000);
    INDArray arr2 = arr.transpose();

    @Override
    public void runOp() {
        arr.mmul(arr2);
    }
}

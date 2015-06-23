package org.nd4j.linalg.benchmark.accum;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.benchmark.api.OpRunner;
import org.nd4j.linalg.factory.Nd4j;

/**
 * @author Adam Gibson
 */
public class SumOpRunner implements OpRunner {
    INDArray arr = Nd4j.create(100000);


    @Override
    public void runOp() {
       arr.sum(Integer.MAX_VALUE);
    }



}

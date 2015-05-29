package org.nd4j.linalg.benchmark.addirowvector;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.benchmark.api.OpRunner;
import org.nd4j.linalg.factory.Nd4j;

/**
 * @author Adam Gibson
 */
public class AddiRowVectorOpRunner implements OpRunner {
    INDArray arr = Nd4j.create(10000,10000);
    INDArray dup = arr.getRow(0);


    @Override
    public void runOp() {
        arr.addiRowVector(dup);
    }



}

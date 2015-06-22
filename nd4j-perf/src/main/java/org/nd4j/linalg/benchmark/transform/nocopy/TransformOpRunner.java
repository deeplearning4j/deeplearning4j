package org.nd4j.linalg.benchmark.transform.nocopy;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.benchmark.api.OpRunner;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

/**
 * @author Adam Gibson
 */
public class TransformOpRunner implements OpRunner {
    INDArray arr = Nd4j.create(10000000);


    @Override
    public void runOp() {
        Transforms.sigmoid(arr,false);
    }
}

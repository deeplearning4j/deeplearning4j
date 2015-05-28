package org.nd4j.linalg.benchmark.linearview;

import org.nd4j.linalg.benchmark.api.OpRunner;
import org.nd4j.linalg.factory.Nd4j;

/**
 * @author Adam Gibson
 */
public class LinearViewOpRunner implements OpRunner {
    int num = 1000000;


    @Override
    public void runOp() {
        Nd4j.create(1000000).reshape(2,num / 2).resetLinearView();
    }
}

package org.deeplearning4j.rl4j.support;

import lombok.Getter;
import org.deeplearning4j.rl4j.observation.CircularFifoObservationPool;
import org.nd4j.linalg.api.ndarray.INDArray;

public class TestObservationPool extends CircularFifoObservationPool {

    @Getter
    private int addCount = 0;

    public TestObservationPool(int poolSize) {
        super(poolSize);
    }

    @Override
    public void add(INDArray observation) {
        super.add(observation);
        ++addCount;
    }
}

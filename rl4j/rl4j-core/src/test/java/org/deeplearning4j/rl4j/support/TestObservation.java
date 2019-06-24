package org.deeplearning4j.rl4j.support;

import org.deeplearning4j.rl4j.observation.Observation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class TestObservation implements Observation {

    private final int observation;

    public TestObservation(int observation) {
        this.observation = observation;
    }

    @Override
    public INDArray toNDArray() {
        INDArray result = Nd4j.create(new double[][] {
                new double[] { observation, (double)observation / 10.0 },
                new double[] { (double)observation / 100.0, (double)observation / 1000.0 }
        });
        return result;
    }
}

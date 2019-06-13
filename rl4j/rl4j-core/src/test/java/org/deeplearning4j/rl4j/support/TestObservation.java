package org.deeplearning4j.rl4j.support;

import org.deeplearning4j.rl4j.space.Encodable;

public class TestObservation implements Encodable {

    private final int observation;

    public TestObservation(int observation) {
        this.observation = observation;
    }

    @Override
    public double[] toArray() {
        return new double[] { observation, (double)observation / 10.0 };
    }
}

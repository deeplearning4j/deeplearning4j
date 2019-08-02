package org.deeplearning4j.rl4j.support;

import org.deeplearning4j.rl4j.space.Encodable;

public class MockEncodable implements Encodable {

    private final int value;

    public MockEncodable(int value) {

        this.value = value;
    }

    @Override
    public double[] toArray() {
        return new double[] { value };
    }
}

package org.deeplearning4j.rl4j.support;

import lombok.Builder;
import org.deeplearning4j.rl4j.learning.IHistoryProcessor;
import org.deeplearning4j.rl4j.observation.Observation;
import org.deeplearning4j.rl4j.observation.VoidObservation;
import org.deeplearning4j.rl4j.observation.transforms.PassthroughTransform;

public class TestObservationTransform extends PassthroughTransform {

    private final TestHistoryProcessor hp;
    private final int addCountBeforeReady;

    public TestObservationTransform(TestHistoryProcessor hp, int addCountBeforeReady) {
        this.hp = hp;
        this.addCountBeforeReady = addCountBeforeReady;
    }

    @Override
    protected Observation handle(Observation input) {
        return input;
    }

    @Override
    protected boolean getIsReady() {
        return hp.getAddCount() >= addCountBeforeReady;
    }
}

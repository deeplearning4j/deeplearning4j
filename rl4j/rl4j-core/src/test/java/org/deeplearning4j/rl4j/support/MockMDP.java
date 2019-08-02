package org.deeplearning4j.rl4j.support;

import org.deeplearning4j.gym.StepReply;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.space.ObservationSpace;

import java.util.ArrayList;
import java.util.List;

public class MockMDP implements MDP<MockEncodable, Integer, DiscreteSpace> {

    private final DiscreteSpace actionSpace;
    private int currentObsValue = 0;
    private final ObservationSpace observationSpace;

    public final List<Integer> actions = new ArrayList<>();

    public MockMDP(ObservationSpace observationSpace) {
        actionSpace = new DiscreteSpace(5);
        this.observationSpace = observationSpace;
    }

    @Override
    public ObservationSpace getObservationSpace() {
        return observationSpace;
    }

    @Override
    public DiscreteSpace getActionSpace() {
        return actionSpace;
    }

    @Override
    public MockEncodable reset() {
        currentObsValue = 0;
        return new MockEncodable(currentObsValue++);
    }

    @Override
    public void close() {

    }

    @Override
    public StepReply<MockEncodable> step(Integer action) {
        actions.add(action);
        return new StepReply<>(new MockEncodable(currentObsValue), (double) currentObsValue++, isDone(), null);
    }

    @Override
    public boolean isDone() {
        return false;
    }

    @Override
    public MDP newInstance() {
        return null;
    }
}

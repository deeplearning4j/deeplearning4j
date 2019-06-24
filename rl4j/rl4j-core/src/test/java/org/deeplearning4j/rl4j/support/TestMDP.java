package org.deeplearning4j.rl4j.support;

import org.deeplearning4j.gym.StepReply;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.observation.Observation;
import org.deeplearning4j.rl4j.space.ActionSpace;
import org.deeplearning4j.rl4j.space.ObservationSpace;

public class TestMDP implements MDP<Observation, Integer, ActionSpace<Integer>> {

    private int currentStep = 0;
    private int lastStep;

    public TestMDP(int lastStep) {
        this.lastStep = lastStep;
    }

    @Override
    public ObservationSpace<Observation> getObservationSpace() {
        return new TestObservationSpace();
    }

    @Override
    public ActionSpace<Integer> getActionSpace() {
        return new TestActionSpace();
    }

    @Override
    public Observation reset() {
        currentStep = 0;
        return new TestObservation(currentStep++);
    }

    @Override
    public void close() {

    }

    @Override
    public StepReply<Observation> step(Integer a) {
        return new StepReply<>(new TestObservation(currentStep), (double) currentStep++, isDone(), null);
    }

    @Override
    public boolean isDone() {
        return currentStep == lastStep;
    }

    @Override
    public MDP<Observation, Integer, ActionSpace<Integer>> newInstance() {
        return null;
    }
}

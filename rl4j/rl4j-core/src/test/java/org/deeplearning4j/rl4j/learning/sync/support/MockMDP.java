package org.deeplearning4j.rl4j.learning.sync.support;

import org.deeplearning4j.gym.StepReply;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.space.ObservationSpace;
import org.nd4j.linalg.api.ndarray.INDArray;

public class MockMDP implements MDP<Object, Integer, DiscreteSpace> {

    private final int maxSteps;
    private final DiscreteSpace actionSpace = new DiscreteSpace(1);
    private final MockObservationSpace observationSpace = new MockObservationSpace();

    private int currentStep = 0;

    public MockMDP(int maxSteps) {

        this.maxSteps = maxSteps;
    }

    @Override
    public ObservationSpace<Object> getObservationSpace() {
        return observationSpace;
    }

    @Override
    public DiscreteSpace getActionSpace() {
        return actionSpace;
    }

    @Override
    public Object reset() {
        return null;
    }

    @Override
    public void close() {

    }

    @Override
    public StepReply<Object> step(Integer integer) {
        return new StepReply<Object>(null, 1.0, isDone(), null);
    }

    @Override
    public boolean isDone() {
        return currentStep >= maxSteps;
    }

    @Override
    public MDP<Object, Integer, DiscreteSpace> newInstance() {
        return null;
    }

    private static class MockObservationSpace implements ObservationSpace {

        @Override
        public String getName() {
            return null;
        }

        @Override
        public int[] getShape() {
            return new int[0];
        }

        @Override
        public INDArray getLow() {
            return null;
        }

        @Override
        public INDArray getHigh() {
            return null;
        }
    }
}

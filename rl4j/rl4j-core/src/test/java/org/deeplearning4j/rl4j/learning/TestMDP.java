package org.deeplearning4j.rl4j.learning;

import org.deeplearning4j.gym.StepReply;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.space.ActionSpace;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.space.Encodable;
import org.deeplearning4j.rl4j.space.ObservationSpace;
import org.nd4j.linalg.api.ndarray.INDArray;

public class TestMDP implements MDP<TestMDP.TestObservation, Integer, DiscreteSpace> {
    public ObservationSpace<TestObservation> observationSpace = new TestObservationSpace();
    public DiscreteSpace actionSpace = new DiscreteSpace(2);

    private int count;

    @Override
    public ObservationSpace<TestObservation> getObservationSpace() {
        return observationSpace;
    }

    @Override
    public DiscreteSpace getActionSpace() {
        return actionSpace;
    }

    @Override
    public TestObservation reset() {
        return new TestObservation();
    }

    @Override
    public void close() {

    }

    @Override
    public StepReply<TestObservation> step(Integer action) {
        boolean isDone = (action == 1);
        return new StepReply<>(new TestObservation(), 1.0, isDone, null);
    }

    @Override
    public boolean isDone() {
        return ++count == 10;
    }

    @Override
    public MDP<TestObservation, Integer, DiscreteSpace> newInstance() {
        return null;
    }

    public static class TestObservation implements Encodable {

        @Override
        public double[] toArray() {
            return new double[] { 0 };
        }
    }

    public static class TestActionSpace implements ActionSpace<Integer> {

        @Override
        public Integer randomAction() {
            return 0;
        }

        @Override
        public void setSeed(int seed) {

        }

        @Override
        public Object encode(Integer action) {
            return null;
        }

        @Override
        public int getSize() {
            return 0;
        }

        @Override
        public Integer noOp() {
            return -1;
        }
    }

    public static class TestObservationSpace implements ObservationSpace<TestObservation> {

        @Override
        public String getName() {
            return null;
        }

        @Override
        public int[] getShape() {
            return new int[] { 1, 1 };
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

package org.deeplearning4j.rl4j.util;

import lombok.Getter;
import org.deeplearning4j.gym.StepReply;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.observation.Observation;
import org.deeplearning4j.rl4j.space.ActionSpace;
import org.deeplearning4j.rl4j.space.Encodable;
import org.deeplearning4j.rl4j.space.ObservationSpace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class LegacyMDPWrapper<O extends Encodable, A, AS extends ActionSpace<A>> implements MDP<Observation, A, AS> {

    @Getter
    private final MDP<O, A, AS> wrappedMDP;
    @Getter
    private final WrapperObservationSpace observationSpace;

    public LegacyMDPWrapper(MDP<O, A, AS> wrappedMDP) {
        this.wrappedMDP = wrappedMDP;
        this.observationSpace = new WrapperObservationSpace(wrappedMDP.getObservationSpace().getShape());
    }

    @Override
    public AS getActionSpace() {
        return wrappedMDP.getActionSpace();
    }

    @Override
    public Observation reset() {
        return new Observation(getInput(wrappedMDP.reset()));
    }

    @Override
    public void close() {
        wrappedMDP.close();
    }

    @Override
    public StepReply<Observation> step(A a) {
        StepReply<O> rawStepReply = wrappedMDP.step(a);
        Observation observation = new Observation(getInput(rawStepReply.getObservation()));
        return new StepReply<Observation>(observation, rawStepReply.getReward(), rawStepReply.isDone(), rawStepReply.getInfo());
    }

    @Override
    public boolean isDone() {
        return wrappedMDP.isDone();
    }

    @Override
    public MDP<Observation, A, AS> newInstance() {
        return new LegacyMDPWrapper<O, A, AS>(wrappedMDP.newInstance());
    }

    private INDArray getInput(O obs) {
        INDArray arr = Nd4j.create(obs.toArray());
        int[] shape = observationSpace.getShape();
        if (shape.length == 1)
            return arr.reshape(new long[] {1, arr.length()});
        else
            return arr.reshape(shape);
    }

    public static class WrapperObservationSpace implements ObservationSpace<Observation> {

        @Getter
        private final int[] shape;

        public WrapperObservationSpace(int[] shape) {

            this.shape = shape;
        }

        @Override
        public String getName() {
            return null;
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

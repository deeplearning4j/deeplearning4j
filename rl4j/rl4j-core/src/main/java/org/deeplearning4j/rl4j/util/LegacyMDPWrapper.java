package org.deeplearning4j.rl4j.util;

import lombok.Getter;
import org.deeplearning4j.gym.StepReply;
import org.deeplearning4j.rl4j.learning.IHistoryProcessor;
import org.deeplearning4j.rl4j.learning.ILearning;
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
    private final ILearning learning;
    private int skipFrame;

    private int step = 0;

    public LegacyMDPWrapper(MDP<O, A, AS> wrappedMDP, ILearning learning) {
        this.wrappedMDP = wrappedMDP;
        this.observationSpace = new WrapperObservationSpace(wrappedMDP.getObservationSpace().getShape());
        this.learning = learning;
    }

    @Override
    public AS getActionSpace() {
        return wrappedMDP.getActionSpace();
    }

    @Override
    public Observation reset() {
        INDArray rawObservation = getInput(wrappedMDP.reset());

        IHistoryProcessor historyProcessor = learning.getHistoryProcessor();
        if(historyProcessor != null) {
            historyProcessor.record(rawObservation.dup());
            rawObservation.muli(1.0 / historyProcessor.getScale());
        }

        Observation observation = new Observation(new INDArray[] { rawObservation });

        if(historyProcessor != null) {
            skipFrame = historyProcessor.getConf().getSkipFrame();
            historyProcessor.add(rawObservation);
        }
        step = 0;

        return observation;
    }

    @Override
    public void close() {
        wrappedMDP.close();
    }

    @Override
    public StepReply<Observation> step(A a) {
        IHistoryProcessor historyProcessor = learning.getHistoryProcessor();

        StepReply<O> rawStepReply = wrappedMDP.step(a);
        INDArray rawObservation = getInput(rawStepReply.getObservation());

        ++step;

        int requiredFrame = 0;
        if(historyProcessor != null) {
            historyProcessor.record(rawObservation.dup());
            rawObservation.muli(1.0 / historyProcessor.getScale());

            requiredFrame = skipFrame * (historyProcessor.getConf().getHistoryLength() - 1);
            if ((learning.getStepCounter() % skipFrame == 0 && step >= requiredFrame)
            || (step % skipFrame == 0 && step < requiredFrame )){
                historyProcessor.add(rawObservation);
            }
        }

        Observation observation;
        if(historyProcessor != null && step >= requiredFrame) {
            observation = new Observation(historyProcessor.getHistory());
        }
        else {
            observation = new Observation(new INDArray[] { rawObservation });
        }

        return new StepReply<Observation>(observation, rawStepReply.getReward(), rawStepReply.isDone(), rawStepReply.getInfo());
    }

    @Override
    public boolean isDone() {
        return wrappedMDP.isDone();
    }

    @Override
    public MDP<Observation, A, AS> newInstance() {
        return new LegacyMDPWrapper<O, A, AS>(wrappedMDP.newInstance(), learning);
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

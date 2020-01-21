package org.deeplearning4j.rl4j.util;

import lombok.Getter;
import org.deeplearning4j.gym.StepReply;
import org.deeplearning4j.rl4j.learning.EpochStepCounter;
import org.deeplearning4j.rl4j.learning.IHistoryProcessor;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.observation.Observation;
import org.deeplearning4j.rl4j.space.ActionSpace;
import org.deeplearning4j.rl4j.space.Encodable;
import org.deeplearning4j.rl4j.space.ObservationSpace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class LegacyMDPWrapper<O, A, AS extends ActionSpace<A>> implements MDP<Observation, A, AS> {

    @Getter
    private final MDP<O, A, AS> wrappedMDP;
    @Getter
    private final WrapperObservationSpace observationSpace;
    private IHistoryProcessor historyProcessor;
    private final EpochStepCounter epochStepCounter;

    private int skipFrame = 1;
    private int requiredFrame = 0;

    public LegacyMDPWrapper(MDP<O, A, AS> wrappedMDP, IHistoryProcessor historyProcessor, EpochStepCounter epochStepCounter) {
        this.wrappedMDP = wrappedMDP;
        this.observationSpace = new WrapperObservationSpace(wrappedMDP.getObservationSpace().getShape());
        this.historyProcessor = historyProcessor;
        this.epochStepCounter = epochStepCounter;
    }

    private IHistoryProcessor getHistoryProcessor() {
        if(historyProcessor != null) {
            return historyProcessor;
        }
        
        return null;
    }
    public void setHistoryProcessor(IHistoryProcessor historyProcessor) {
        this.historyProcessor = historyProcessor;
    }

    @Override
    public AS getActionSpace() {
        return wrappedMDP.getActionSpace();
    }

    @Override
    public Observation reset() {
        INDArray rawObservation = getInput(wrappedMDP.reset());

        IHistoryProcessor historyProcessor = getHistoryProcessor();
        if(historyProcessor != null) {
            historyProcessor.record(rawObservation);
        }

        Observation observation = new Observation(new INDArray[] { rawObservation }, false);

        if(historyProcessor != null) {
            skipFrame = historyProcessor.getConf().getSkipFrame();
            requiredFrame = skipFrame * (historyProcessor.getConf().getHistoryLength() - 1);

            historyProcessor.add(rawObservation);
        }

        observation.setSkipped(skipFrame != 0);

        return observation;
    }

    @Override
    public StepReply<Observation> step(A a) {
        IHistoryProcessor historyProcessor = getHistoryProcessor();

        StepReply<O> rawStepReply = wrappedMDP.step(a);
        INDArray rawObservation = getInput(rawStepReply.getObservation());

        int stepOfObservation = epochStepCounter.getCurrentEpochStep() + 1;

        if(historyProcessor != null) {
            historyProcessor.record(rawObservation);

            if (stepOfObservation % skipFrame == 0) {
                historyProcessor.add(rawObservation);
            }
        }

        Observation observation;
        if(historyProcessor != null && stepOfObservation >= requiredFrame) {
            observation = new Observation(historyProcessor.getHistory(), true);
            observation.getData().muli(1.0 / historyProcessor.getScale());
        }
        else {
            observation = new Observation(new INDArray[] { rawObservation }, false);
        }

        if(stepOfObservation % skipFrame != 0 || stepOfObservation < requiredFrame) {
            observation.setSkipped(true);
        }

        return new StepReply<Observation>(observation, rawStepReply.getReward(), rawStepReply.isDone(), rawStepReply.getInfo());
    }

    @Override
    public void close() {
        wrappedMDP.close();
    }

    @Override
    public boolean isDone() {
        return wrappedMDP.isDone();
    }

    @Override
    public MDP<Observation, A, AS> newInstance() {
        return new LegacyMDPWrapper<O, A, AS>(wrappedMDP.newInstance(), historyProcessor, epochStepCounter);
    }

    private INDArray getInput(O obs) {
        INDArray arr = Nd4j.create(((Encodable)obs).toArray());
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

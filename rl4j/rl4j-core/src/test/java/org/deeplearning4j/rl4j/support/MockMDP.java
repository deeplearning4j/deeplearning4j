package org.deeplearning4j.rl4j.support;

import org.deeplearning4j.gym.StepReply;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.observation.transform.TransformProcess;
import org.deeplearning4j.rl4j.observation.transform.filter.UniformSkippingFilter;
import org.deeplearning4j.rl4j.observation.transform.legacy.EncodableToINDArrayTransform;
import org.deeplearning4j.rl4j.observation.transform.operation.HistoryMergeTransform;
import org.deeplearning4j.rl4j.observation.transform.operation.SimpleNormalizationTransform;
import org.deeplearning4j.rl4j.observation.transform.operation.historymerge.CircularFifoStore;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.space.ObservationSpace;
import org.nd4j.linalg.api.rng.Random;

import java.util.ArrayList;
import java.util.List;

public class MockMDP implements MDP<MockEncodable, Integer, DiscreteSpace> {

    private final DiscreteSpace actionSpace;
    private final int stepsUntilDone;
    private int currentObsValue = 0;
    private final ObservationSpace observationSpace;

    public final List<Integer> actions = new ArrayList<>();
    private int step = 0;
    public int resetCount = 0;

    public MockMDP(ObservationSpace observationSpace, int stepsUntilDone, DiscreteSpace actionSpace) {
        this.stepsUntilDone = stepsUntilDone;
        this.actionSpace = actionSpace;
        this.observationSpace = observationSpace;
    }

    public MockMDP(ObservationSpace observationSpace, int stepsUntilDone, Random rnd) {
        this(observationSpace, stepsUntilDone, new DiscreteSpace(5, rnd));
    }

    public MockMDP(ObservationSpace observationSpace) {
        this(observationSpace, Integer.MAX_VALUE, new DiscreteSpace(5));
    }

    public MockMDP(ObservationSpace observationSpace, Random rnd) {
        this(observationSpace, Integer.MAX_VALUE, new DiscreteSpace(5, rnd));
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
        ++resetCount;
        currentObsValue = 0;
        step = 0;
        return new MockEncodable(currentObsValue++);
    }

    @Override
    public void close() {

    }

    @Override
    public StepReply<MockEncodable> step(Integer action) {
        actions.add(action);
        ++step;
        return new StepReply<>(new MockEncodable(currentObsValue), (double) currentObsValue++, isDone(), null);
    }

    @Override
    public boolean isDone() {
        return step >= stepsUntilDone;
    }

    @Override
    public MDP newInstance() {
        return null;
    }

    public static TransformProcess buildTransformProcess(int[] shape, int skipFrame, int historyLength) {
        return TransformProcess.builder()
                .filter(new UniformSkippingFilter(skipFrame))
                .transform("data", new EncodableToINDArrayTransform(shape))
                .transform("data", new SimpleNormalizationTransform(0.0, 255.0))
                .transform("data", HistoryMergeTransform.builder()
                        .elementStore(new CircularFifoStore(historyLength))
                        .build())
                .build("data");
    }

}

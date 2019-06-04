package org.deeplearning4j.rl4j.policy;

import org.deeplearning4j.rl4j.learning.StepCountable;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.space.Encodable;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Random;

public class TestEpsGreedyPolicy<O extends Encodable> extends EpsGreedy<O, Integer, DiscreteSpace> {
    private final int[] actionList;
    private int actionListIdx;

    public TestEpsGreedyPolicy(int[] actionList) {
        super(null, null, 0, 0, null, 0.0f, null);
        this.actionList = actionList;
    }

    public TestEpsGreedyPolicy(int[] actionList, StepCountable learning) {
        super(null, null, 0, 0, null, 0.0f, learning);
        this.actionList = actionList;
    }

    @Override
    public Integer nextAction(INDArray input) {
        return actionList[actionListIdx++];
    }
}

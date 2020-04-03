package org.deeplearning4j.rl4j.support;

import org.deeplearning4j.rl4j.learning.IHistoryProcessor;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.observation.Observation;
import org.deeplearning4j.rl4j.policy.IPolicy;
import org.deeplearning4j.rl4j.space.ActionSpace;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.ArrayList;
import java.util.List;

public class MockPolicy implements IPolicy<Integer> {

    public int playCallCount = 0;
    public List<INDArray> actionInputs = new ArrayList<INDArray>();

    @Override
    public <O, AS extends ActionSpace<Integer>> double play(MDP<O, Integer, AS> mdp, IHistoryProcessor hp) {
        ++playCallCount;
        return 0;
    }

    @Override
    public Integer nextAction(INDArray input) {
        actionInputs.add(input);
        return input.getInt(0, 0, 0);
    }

    @Override
    public Integer nextAction(Observation observation) {
        return nextAction(observation.getData());
    }

    @Override
    public void reset() {

    }
}

package org.deeplearning4j.rl4j.support;

import org.deeplearning4j.rl4j.learning.IHistoryProcessor;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.policy.IPolicy;
import org.deeplearning4j.rl4j.space.ActionSpace;

public class MockPolicy implements IPolicy<MockEncodable, Integer> {

    public int playCallCount = 0;

    @Override
    public <AS extends ActionSpace<Integer>> double play(MDP<MockEncodable, Integer, AS> mdp, IHistoryProcessor hp) {
        ++playCallCount;
        return 0;
    }
}

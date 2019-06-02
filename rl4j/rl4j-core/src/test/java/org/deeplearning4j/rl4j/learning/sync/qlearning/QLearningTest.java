package org.deeplearning4j.rl4j.learning.sync.qlearning;

import org.deeplearning4j.rl4j.learning.TestHistoryProcessor;
import org.deeplearning4j.rl4j.learning.TestMDP;
import org.deeplearning4j.rl4j.learning.sync.qlearning.discrete.QLearningDiscrete;
import org.deeplearning4j.rl4j.policy.TestEpsGreedyPolicy;
import org.junit.Test;

import java.io.IOException;

import static org.junit.Assert.assertEquals;

public class QLearningTest {

    @Test
    public void trainEpoch_WithHistoryProcessor_ShouldCallRecord() throws IOException {
        // Arrange
        TestHistoryProcessor hp = new TestHistoryProcessor();
        TestMDP mdp = new TestMDP();
        TestQLearning sut = new TestQLearning(mdp, hp);
        TestEpsGreedyPolicy policy = new TestEpsGreedyPolicy(new int[] { 0, 0, 0, 1 }, sut);
        sut.setPolicy(policy);


        // Act
        for(int i = 0; i < 4; ++i){
            sut.testTrainEpoch();
        }

        assertEquals(108, hp.recordCallCount);

    }
}

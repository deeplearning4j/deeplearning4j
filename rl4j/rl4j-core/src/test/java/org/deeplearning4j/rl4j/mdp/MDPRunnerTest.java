package org.deeplearning4j.rl4j.mdp;

import org.deeplearning4j.rl4j.learning.Learning;
import org.deeplearning4j.rl4j.learning.TestHistoryProcessor;
import org.deeplearning4j.rl4j.learning.TestMDP;
import org.junit.Test;
import org.deeplearning4j.rl4j.mdp.MDPRunner;
import org.deeplearning4j.rl4j.mdp.IMDPRunner;

import java.io.IOException;

import static org.junit.Assert.assertEquals;

public class MDPRunnerTest {
    @Test
    public void initMdp_NoHistoryProcessor_RewardShouldBe0() throws IOException {

        // Arrange
        TestMDP mdp = new TestMDP();
        IMDPRunner sut = new MDPRunner();

        // Act
        Learning.InitMdp<TestMDP.TestObservation> result = sut.initMdp(mdp);

        assertEquals(0.0, result.getReward(), 0.1);
    }
}

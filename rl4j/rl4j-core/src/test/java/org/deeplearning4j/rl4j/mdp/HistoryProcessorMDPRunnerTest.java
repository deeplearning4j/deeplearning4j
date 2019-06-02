package org.deeplearning4j.rl4j.mdp;

import org.deeplearning4j.rl4j.learning.Learning;
import org.deeplearning4j.rl4j.learning.TestHistoryProcessor;
import org.deeplearning4j.rl4j.learning.TestMDP;
import org.junit.Test;
import org.deeplearning4j.rl4j.mdp.HistoryProcessorMDPRunner;
import org.deeplearning4j.rl4j.mdp.IMDPRunner;

import java.io.IOException;

import static org.junit.Assert.assertEquals;

public class HistoryProcessorMDPRunnerTest {
    @Test
    public void initMdp_WithHistoryProcessor_ShouldCallRecord() {

        // Arrange
        TestHistoryProcessor hp = new TestHistoryProcessor();
        TestMDP mdp = new TestMDP();
        IMDPRunner sut = new HistoryProcessorMDPRunner(hp);

        // Act
        sut.initMdp(mdp);

        assertEquals(27, hp.recordCallCount);
    }

    @Test
    public void initMdp_WithHistoryProcessor_ShouldCallAdd() {

        // Arrange
        TestHistoryProcessor hp = new TestHistoryProcessor();
        TestMDP mdp = new TestMDP();
        IMDPRunner sut = new HistoryProcessorMDPRunner(hp);

        // Act
        sut.initMdp(mdp);

        assertEquals(9, hp.addCallCount);
    }

    @Test
    public void initMdp_WithHistoryProcessor_RewardShouldBe27() throws IOException {

        // Arrange
        TestHistoryProcessor hp = new TestHistoryProcessor();
        TestMDP mdp = new TestMDP();
        IMDPRunner sut = new HistoryProcessorMDPRunner(hp);

        // Act
        Learning.InitMdp<TestMDP.TestObservation> result = sut.initMdp(mdp);

        assertEquals(27.0, result.getReward(), 0.1);
    }

}

package org.deeplearning4j.rl4j.learning;

import org.junit.Test;

import java.io.IOException;

import static org.junit.Assert.assertEquals;


public class LearningTest {

    @Test
    public void initMdp_WithHistoryProcessor_ShouldCallRecord() {

        // Arrange
        TestHistoryProcessor hp = new TestHistoryProcessor();
        TestMDP mdp = new TestMDP();

        // Act
        Learning.initMdp(mdp, hp);

        assertEquals(27, hp.recordCallCount);
    }

    @Test
    public void initMdp_WithHistoryProcessor_ShouldCallAdd() {

        // Arrange
        TestHistoryProcessor hp = new TestHistoryProcessor();
        TestMDP mdp = new TestMDP();

        // Act
        Learning.initMdp(mdp, hp);

        assertEquals(9, hp.addCallCount);
    }

    @Test
    public void initMdp_WithHistoryProcessor_RewardShouldBe27() throws IOException {

        // Arrange
        TestHistoryProcessor hp = new TestHistoryProcessor();
        TestMDP mdp = new TestMDP();

        // Act
        Learning.InitMdp<TestMDP.TestObservation> result = Learning.initMdp(mdp, hp);

        assertEquals(27.0, result.getReward(), 0.1);
    }

    @Test
    public void initMdp_NoHistoryProcessor_RewardShouldBe0() throws IOException {

        // Arrange
        TestHistoryProcessor hp = null;
        TestMDP mdp = new TestMDP();

        // Act
        Learning.InitMdp<TestMDP.TestObservation> result = Learning.initMdp(mdp, hp);

        assertEquals(0.0, result.getReward(), 0.1);
    }


}

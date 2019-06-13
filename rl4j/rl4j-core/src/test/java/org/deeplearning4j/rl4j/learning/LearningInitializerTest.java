package org.deeplearning4j.rl4j.learning;

import org.deeplearning4j.rl4j.space.ActionSpace;
import org.deeplearning4j.rl4j.support.TestHistoryProcessor;
import org.deeplearning4j.rl4j.support.TestMDP;
import org.deeplearning4j.rl4j.support.TestObservation;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class LearningInitializerTest {
    @Test
    public void LearningInitializer_init_ShouldBeAll0() {
        // Arrange
        ILearningInitializer<TestObservation, Integer, ActionSpace<Integer>> sut = new LearningInitializer<TestObservation, Integer, ActionSpace<Integer>>();
        TestMDP mdp = new TestMDP();

        // Act
        Learning.InitMdp<TestObservation> result = sut.initMdp(mdp);

        // Assert
        assertEquals(0.0, result.getLastObs().toArray()[0], 0.0);
        assertEquals(0.0, result.getReward(), 0.0);
        assertEquals(0, result.getSteps());
    }
}

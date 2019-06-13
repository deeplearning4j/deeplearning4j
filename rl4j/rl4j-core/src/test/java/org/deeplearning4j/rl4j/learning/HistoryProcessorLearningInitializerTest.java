package org.deeplearning4j.rl4j.learning;

import org.deeplearning4j.rl4j.space.ActionSpace;
import org.deeplearning4j.rl4j.support.TestHistoryProcessor;
import org.deeplearning4j.rl4j.support.TestMDP;
import org.deeplearning4j.rl4j.support.TestObservation;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class HistoryProcessorLearningInitializerTest {
    @Test
    public void HistoryProcessorLearningInitializer_init_0HistoryLength() {
        // Arrange
        TestHistoryProcessor hp = new TestHistoryProcessor(0, 1);
        ILearningInitializer<TestObservation, Integer, ActionSpace<Integer>> sut = new org.deeplearning4j.rl4j.learning.HistoryProcessorLearningInitializer<TestObservation, Integer, ActionSpace<Integer>>(hp);
        TestMDP mdp = new TestMDP();

        // Act
        Learning.InitMdp<TestObservation> result = sut.initMdp(mdp);

        // Assert
        assertEquals(0.0, result.getLastObs().toArray()[0], 0.0);
        assertEquals(0.0, result.getReward(), 0.0);
        assertEquals(0, result.getSteps());
    }

    @Test
    public void HistoryProcessorLearningInitializer_init_NoSkip5HistoryLen() {
        // Arrange
        TestHistoryProcessor hp = new TestHistoryProcessor(5, 1);
        ILearningInitializer<TestObservation, Integer, ActionSpace<Integer>> sut = new org.deeplearning4j.rl4j.learning.HistoryProcessorLearningInitializer<TestObservation, Integer, ActionSpace<Integer>>(hp);
        TestMDP mdp = new TestMDP();

        // Act
        Learning.InitMdp<TestObservation> result = sut.initMdp(mdp);

        // Assert
        assertEquals(4.0, result.getLastObs().toArray()[0], 0.0);
        assertEquals(10.0, result.getReward(), 0.0);
        assertEquals(4, result.getSteps());
    }

    @Test
    public void HistoryProcessorLearningInitializer_init_2Skip6HistoryLen() {
        // Arrange
        TestHistoryProcessor hp = new TestHistoryProcessor(6, 2);
        ILearningInitializer<TestObservation, Integer, ActionSpace<Integer>> sut = new org.deeplearning4j.rl4j.learning.HistoryProcessorLearningInitializer<TestObservation, Integer, ActionSpace<Integer>>(hp);
        TestMDP mdp = new TestMDP();

        // Act
        Learning.InitMdp<TestObservation> result = sut.initMdp(mdp);

        // Assert
        assertEquals(10.0, result.getLastObs().toArray()[0], 0.0);
        assertEquals(55.0, result.getReward(), 0.0);
        assertEquals(10, result.getSteps());
        assertEquals(10, hp.getRecordCount());
        assertEquals(5, hp.getAddCount());
    }

}

package org.deeplearning4j.rl4j.learning.sync.qlearning;

import org.deeplearning4j.rl4j.learning.Learning;
import org.deeplearning4j.rl4j.learning.TestHistoryProcessor;
import org.deeplearning4j.rl4j.learning.TestMDP;
import org.deeplearning4j.rl4j.policy.TestEpsGreedyPolicy;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;

import java.io.IOException;

import static org.junit.Assert.assertEquals;

public class QLearningDiscreteTest {

    @Rule
    public ExpectedException thrown = ExpectedException.none();


    @Test
    public void trainStep_WithHistoryProcessor_ShouldCallRecord() throws IOException {

        // Arrange
        TestHistoryProcessor hp = new TestHistoryProcessor();
        TestMDP mdp = new TestMDP();
        TestQLearningDiscrete sut = new TestQLearningDiscrete(mdp, hp);
        TestEpsGreedyPolicy policy = new TestEpsGreedyPolicy(new int[] { 0, 0, 0, 1 });
        sut.setEgPolicy(policy);

        // Act
        for(int i = 0; i < 4; ++i){
            sut.testTrainStep(new TestMDP.TestObservation());
            sut.setStepCounter(i);
        }

        assertEquals(4, hp.recordCallCount);
    }

    @Test
    public void trainStep_WithHistoryProcessor_ShouldCallAdd() throws IOException {

        // Arrange
        TestHistoryProcessor hp = new TestHistoryProcessor();
        TestMDP mdp = new TestMDP();
        TestQLearningDiscrete sut = new TestQLearningDiscrete(mdp, hp);
        TestEpsGreedyPolicy policy = new TestEpsGreedyPolicy(new int[] { 0, 0, 0, 1 });
        sut.setEgPolicy(policy);

        // Act
        for(int i = 0; i < 4; ++i){
            sut.testTrainStep(new TestMDP.TestObservation());
            sut.setStepCounter(i);
        }

        assertEquals(3, hp.addCallCount);
    }

    @Test
    public void trainStep_WithHistoryProcessor_ShouldCallGetHistory() throws IOException {

        // Arrange
        TestHistoryProcessor hp = new TestHistoryProcessor();
        TestMDP mdp = new TestMDP();
        TestQLearningDiscrete sut = new TestQLearningDiscrete(mdp, hp);
        TestEpsGreedyPolicy policy = new TestEpsGreedyPolicy(new int[] { 0, 0, 0, 1 });
        sut.setEgPolicy(policy);

        // Act
        for(int i = 0; i < 4; ++i){
            sut.testTrainStep(new TestMDP.TestObservation());
            sut.setStepCounter(i);
        }

        assertEquals(3, hp.getHistoryCallCount);
    }

    @Test
    public void trainStep_WithHistoryProcessor_ShouldCallGetScale() throws IOException {

        // Arrange
        TestHistoryProcessor hp = new TestHistoryProcessor();
        TestMDP mdp = new TestMDP();
        TestQLearningDiscrete sut = new TestQLearningDiscrete(mdp, hp);
        TestEpsGreedyPolicy policy = new TestEpsGreedyPolicy(new int[] { 0, 0, 0, 1 });
        sut.setEgPolicy(policy);

        // Act
        for(int i = 0; i < 4; ++i){
            sut.testTrainStep(new TestMDP.TestObservation());
            sut.setStepCounter(i);
        }

        assertEquals(2, hp.getScaleCallCount);
    }
}

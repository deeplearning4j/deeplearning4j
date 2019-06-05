package org.deeplearning4j.rl4j.learning.sync.qlearning.discrete;

import org.deeplearning4j.rl4j.learning.TestHistoryProcessor;
import org.deeplearning4j.rl4j.learning.TestMDP;
import org.deeplearning4j.rl4j.learning.sync.qlearning.QLearning;
import org.deeplearning4j.rl4j.policy.TestEpsGreedyPolicy;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;

import java.io.IOException;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

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
        sut.preEpoch();
        for(int i = 0; i < 4; ++i){
            sut.setStepCounter(i);
            sut.testTrainStep(new TestMDP.TestObservation());
        }
        sut.postEpoch();

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
        sut.preEpoch();
        for(int i = 0; i < 4; ++i){
            sut.setStepCounter(i);
            sut.testTrainStep(new TestMDP.TestObservation());
        }
        sut.postEpoch();

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
        sut.preEpoch();
        for(int i = 0; i < 4; ++i){
            sut.setStepCounter(i);
            sut.testTrainStep(new TestMDP.TestObservation());
        }
        sut.postEpoch();

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
        sut.preEpoch();
        for(int i = 0; i < 4; ++i){
            sut.setStepCounter(i);
            sut.testTrainStep(new TestMDP.TestObservation());
        }
        sut.postEpoch();

        assertEquals(2, hp.getScaleCallCount);
    }

    @Test
    public void trainStep_WithHistoryProcessor_ShouldHaveExpectedBatches() throws IOException {

        // Arrange
        TestHistoryProcessor hp = new TestHistoryProcessor();
        TestMDP mdp = new TestMDP();
        TestQLearningDiscrete sut = new TestQLearningDiscrete(mdp, hp);
        TestEpsGreedyPolicy policy = new TestEpsGreedyPolicy(new int[] { 0, 0, 0, 1 });
        sut.setEgPolicy(policy);

        // Act
        sut.preEpoch();
        for(int i = 0; i < 4; ++i){
            sut.setStepCounter(i);
            sut.testTrainStep(new TestMDP.TestObservation());
        }
        sut.postEpoch();

        TestQLearningDiscrete.TestDQN testDQN = (TestQLearningDiscrete.TestDQN)sut.getCurrentDQN();

        assertEquals(2, testDQN.batches.size());
        assertEquals(1.0, testDQN.batches.get(0).getDouble(0, 0), 0.0);
        assertEquals(2.0, testDQN.batches.get(1).getDouble(0, 0), 0.0);
    }

    @Test
    public void trainStep_NoHistoryProcessor_ShouldHaveExpectedBatches() throws IOException {

        // Arrange
        TestHistoryProcessor hp = null;
        TestMDP mdp = new TestMDP();
        TestQLearningDiscrete sut = new TestQLearningDiscrete(mdp, hp);
        TestEpsGreedyPolicy policy = new TestEpsGreedyPolicy(new int[] { 0, 0, 0, 1 });
        sut.setEgPolicy(policy);

        // Act
        sut.preEpoch();
        for(int i = 0; i < 4; ++i){
            sut.setStepCounter(i);
            sut.testTrainStep(new TestMDP.TestObservation((double)i));
        }
        sut.postEpoch();

        TestQLearningDiscrete.TestDQN testDQN = (TestQLearningDiscrete.TestDQN)sut.getCurrentDQN();

        assertEquals(4, testDQN.batches.size());
        assertEquals(0.0, testDQN.batches.get(0).getDouble(0, 0), 0.0);
        assertEquals(1.0, testDQN.batches.get(1).getDouble(0, 0), 0.0);
        assertEquals(2.0, testDQN.batches.get(2).getDouble(0, 0), 0.0);
        assertEquals(3.0, testDQN.batches.get(3).getDouble(0, 0), 0.0);

    }

    @Test
    public void trainStep_WithHistoryProcessor_CheckLastAction() throws IOException {

        // Arrange
        TestHistoryProcessor hp = new TestHistoryProcessor(2);
        TestMDP mdp = new TestMDP();
        TestQLearningDiscrete sut = new TestQLearningDiscrete(mdp, hp);
        TestEpsGreedyPolicy policy = new TestEpsGreedyPolicy(new int[] { 4, 3, 2, 1 });
        sut.setEgPolicy(policy);

        sut.preEpoch();
        assertEquals(0, sut.getLastAction());

        sut.setStepCounter(0);
        sut.testTrainStep(new TestMDP.TestObservation());
        assertEquals(4, sut.getLastAction());

        sut.setStepCounter(1);
        sut.testTrainStep(new TestMDP.TestObservation());
        assertEquals(4, sut.getLastAction());

        sut.setStepCounter(2);
        sut.testTrainStep(new TestMDP.TestObservation());
        assertEquals(3, sut.getLastAction());

        sut.setStepCounter(3);
        sut.testTrainStep(new TestMDP.TestObservation());
        assertEquals(3, sut.getLastAction());

        sut.setStepCounter(4);
        sut.testTrainStep(new TestMDP.TestObservation());
        assertEquals(2, sut.getLastAction());

        sut.setStepCounter(5);
        sut.testTrainStep(new TestMDP.TestObservation());
        assertEquals(2, sut.getLastAction());

        sut.setStepCounter(6);
        sut.testTrainStep(new TestMDP.TestObservation());
        assertEquals(1, sut.getLastAction());

        sut.setStepCounter(7);
        sut.testTrainStep(new TestMDP.TestObservation());
        assertEquals(1, sut.getLastAction());
    }

    @Test
    public void trainStep_NoHistoryProcessor_CheckLastAction() throws IOException {

        // Arrange
        TestHistoryProcessor hp = null;
        TestMDP mdp = new TestMDP();
        TestQLearningDiscrete sut = new TestQLearningDiscrete(mdp, hp);
        TestEpsGreedyPolicy policy = new TestEpsGreedyPolicy(new int[] { 4, 3, 2, 1 });
        sut.setEgPolicy(policy);

        sut.preEpoch();
        assertEquals(0, sut.getLastAction());

        sut.setStepCounter(0);
        sut.testTrainStep(new TestMDP.TestObservation());
        assertEquals(4, sut.getLastAction());

        sut.setStepCounter(1);
        sut.testTrainStep(new TestMDP.TestObservation());
        assertEquals(3, sut.getLastAction());

        sut.setStepCounter(2);
        sut.testTrainStep(new TestMDP.TestObservation());
        assertEquals(2, sut.getLastAction());

        sut.setStepCounter(3);
        sut.testTrainStep(new TestMDP.TestObservation());
        assertEquals(1, sut.getLastAction());

    }

    @Test
    public void trainStep_WithHistoryProcessor_CheckMaxQ() throws IOException {

        // Arrange
        TestHistoryProcessor hp = new TestHistoryProcessor(2);
        TestMDP mdp = new TestMDP();
        TestQLearningDiscrete sut = new TestQLearningDiscrete(mdp, hp);
        TestEpsGreedyPolicy policy = new TestEpsGreedyPolicy(new int[] { 4, 3, 2, 1 });
        sut.setEgPolicy(policy);

        sut.preEpoch();
        assertEquals(0, sut.getLastAction());

        sut.setStepCounter(0);
        QLearning.QLStepReturn<TestMDP.TestObservation> result =  sut.testTrainStep(new TestMDP.TestObservation());
        assertEquals(1.0, result.getMaxQ(), 0.0);

        sut.setStepCounter(1);
        result =  sut.testTrainStep(new TestMDP.TestObservation());
        assertTrue(Double.isNaN(result.getMaxQ()));

        sut.setStepCounter(2);
        result =  sut.testTrainStep(new TestMDP.TestObservation());
        assertEquals(2.0, result.getMaxQ(), 0.0);

        sut.setStepCounter(3);
        result =  sut.testTrainStep(new TestMDP.TestObservation());
        assertTrue(Double.isNaN(result.getMaxQ()));

        sut.setStepCounter(4);
        result =  sut.testTrainStep(new TestMDP.TestObservation());
        assertEquals(3.0, result.getMaxQ(), 0.0);

        sut.setStepCounter(5);
        result =  sut.testTrainStep(new TestMDP.TestObservation());
        assertTrue(Double.isNaN(result.getMaxQ()));

        sut.setStepCounter(6);
        result =  sut.testTrainStep(new TestMDP.TestObservation());
        assertEquals(4.0, result.getMaxQ(), 0.0);

        sut.setStepCounter(7);
        result =  sut.testTrainStep(new TestMDP.TestObservation());
        assertTrue(Double.isNaN(result.getMaxQ()));
    }

    @Test
    public void trainStep_NoHistoryProcessor_CheckMaxQ() throws IOException {

        // Arrange
        TestHistoryProcessor hp = null;
        TestMDP mdp = new TestMDP();
        TestQLearningDiscrete sut = new TestQLearningDiscrete(mdp, hp);
        TestEpsGreedyPolicy policy = new TestEpsGreedyPolicy(new int[] { 4, 3, 2, 1 });
        sut.setEgPolicy(policy);

        sut.preEpoch();
        assertEquals(0, sut.getLastAction());

        sut.setStepCounter(0);
        QLearning.QLStepReturn<TestMDP.TestObservation> result =  sut.testTrainStep(new TestMDP.TestObservation());
        assertEquals(1.0, result.getMaxQ(), 0.0);

        sut.setStepCounter(1);
        result =  sut.testTrainStep(new TestMDP.TestObservation());
        assertEquals(2.0, result.getMaxQ(), 0.0);

        sut.setStepCounter(2);
        result =  sut.testTrainStep(new TestMDP.TestObservation());
        assertEquals(3.0, result.getMaxQ(), 0.0);

        sut.setStepCounter(3);
        result =  sut.testTrainStep(new TestMDP.TestObservation());
        assertEquals(4.0, result.getMaxQ(), 0.0);
    }
}

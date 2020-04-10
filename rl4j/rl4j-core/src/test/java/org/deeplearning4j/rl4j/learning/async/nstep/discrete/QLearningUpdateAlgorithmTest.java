package org.deeplearning4j.rl4j.learning.async.nstep.discrete;

import org.deeplearning4j.rl4j.experience.StateActionPair;
import org.deeplearning4j.rl4j.learning.async.UpdateAlgorithm;
import org.deeplearning4j.rl4j.observation.Observation;
import org.deeplearning4j.rl4j.support.MockAsyncGlobal;
import org.deeplearning4j.rl4j.support.MockDQN;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;

public class QLearningUpdateAlgorithmTest {

    @Test
    public void when_isTerminal_expect_initRewardIs0() {
        // Arrange
        MockDQN dqnMock = new MockDQN();
        MockAsyncGlobal asyncGlobalMock = new MockAsyncGlobal(dqnMock);
        UpdateAlgorithm sut = new QLearningUpdateAlgorithm(asyncGlobalMock, new int[] { 1 }, 1, -1, 1.0);
        List<StateActionPair<Integer>> experience = new ArrayList<StateActionPair<Integer>>() {
            {
                add(new StateActionPair<Integer>(new Observation(Nd4j.zeros(1)), 0, 0.0, true));
            }
        };

        // Act
        sut.computeGradients(dqnMock, experience);

        // Assert
        assertEquals(0.0, dqnMock.gradientParams.get(0).getRight().getDouble(0), 0.00001);
    }

    @Test
    public void when_terminalAndNoTargetUpdate_expect_initRewardWithMaxQFromCurrent() {
        // Arrange
        MockDQN globalDQNMock = new MockDQN();
        MockAsyncGlobal asyncGlobalMock = new MockAsyncGlobal(globalDQNMock);
        UpdateAlgorithm sut = new QLearningUpdateAlgorithm(asyncGlobalMock, new int[] { 2 }, 2, -1, 1.0);
        List<StateActionPair<Integer>> experience = new ArrayList<StateActionPair<Integer>>() {
            {
                add(new StateActionPair<Integer>(new Observation(Nd4j.create(new double[] { -123.0, -234.0 })), 0, 0.0, false));
            }
        };
        MockDQN dqnMock = new MockDQN();

        // Act
        sut.computeGradients(dqnMock, experience);

        // Assert
        assertEquals(2, dqnMock.outputAllParams.size());
        assertEquals(-123.0, dqnMock.outputAllParams.get(0).getDouble(0, 0), 0.00001);
        assertEquals(234.0, dqnMock.gradientParams.get(0).getRight().getDouble(0), 0.00001);
    }

    @Test
    public void when_terminalWithTargetUpdate_expect_initRewardWithMaxQFromGlobal() {
        // Arrange
        MockDQN globalDQNMock = new MockDQN();
        MockAsyncGlobal asyncGlobalMock = new MockAsyncGlobal(globalDQNMock);
        UpdateAlgorithm sut = new QLearningUpdateAlgorithm(asyncGlobalMock, new int[] { 2 }, 2, 1, 1.0);
        List<StateActionPair<Integer>> experience = new ArrayList<StateActionPair<Integer>>() {
            {
                add(new StateActionPair<Integer>(new Observation(Nd4j.create(new double[] { -123.0, -234.0 })), 0, 0.0, false));
            }
        };
        MockDQN dqnMock = new MockDQN();

        // Act
        sut.computeGradients(dqnMock, experience);

        // Assert
        assertEquals(1, globalDQNMock.outputAllParams.size());
        assertEquals(-123.0, globalDQNMock.outputAllParams.get(0).getDouble(0, 0), 0.00001);
        assertEquals(234.0, dqnMock.gradientParams.get(0).getRight().getDouble(0), 0.00001);
    }

    @Test
    public void when_callingWithMultipleExperiences_expect_gradientsAreValid() {
        // Arrange
        double gamma = 0.9;
        MockDQN globalDQNMock = new MockDQN();
        MockAsyncGlobal asyncGlobalMock = new MockAsyncGlobal(globalDQNMock);
        UpdateAlgorithm sut = new QLearningUpdateAlgorithm(asyncGlobalMock, new int[] { 2 }, 2, 1, gamma);
        List<StateActionPair<Integer>> experience = new ArrayList<StateActionPair<Integer>>() {
            {
                add(new StateActionPair<Integer>(new Observation(Nd4j.create(new double[] { -1.1, -1.2 })), 0, 1.0, false));
                add(new StateActionPair<Integer>(new Observation(Nd4j.create(new double[] { -2.1, -2.2 })), 1, 2.0, true));
            }
        };
        MockDQN dqnMock = new MockDQN();

        // Act
        sut.computeGradients(dqnMock, experience);

        // Assert
        // input side -- should be a stack of observations
        INDArray input = dqnMock.gradientParams.get(0).getLeft();
        assertEquals(-1.1, input.getDouble(0, 0), 0.00001);
        assertEquals(-1.2, input.getDouble(0, 1), 0.00001);
        assertEquals(-2.1, input.getDouble(1, 0), 0.00001);
        assertEquals(-2.2, input.getDouble(1, 1), 0.00001);

        // target side
        INDArray target = dqnMock.gradientParams.get(0).getRight();
        assertEquals(1.0 + gamma * 2.0, target.getDouble(0, 0), 0.00001);
        assertEquals(1.2, target.getDouble(0, 1), 0.00001);
        assertEquals(2.1, target.getDouble(1, 0), 0.00001);
        assertEquals(2.0, target.getDouble(1, 1), 0.00001);
    }
}

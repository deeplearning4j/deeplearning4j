package org.deeplearning4j.rl4j.learning.async.nstep.discrete;

import org.deeplearning4j.rl4j.learning.IHistoryProcessor;
import org.deeplearning4j.rl4j.learning.async.MiniTrans;
import org.deeplearning4j.rl4j.support.*;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Stack;

import static org.junit.Assert.assertEquals;

public class AsyncNStepQLearningThreadDiscreteTest {

    @Test
    public void refac_calcGradient() {
        // Arrange
        double gamma = 0.9;
        MockObservationSpace observationSpace = new MockObservationSpace();
        MockMDP mdpMock = new MockMDP(observationSpace);
        AsyncNStepQLearningDiscrete.AsyncNStepQLConfiguration config = new AsyncNStepQLearningDiscrete.AsyncNStepQLConfiguration(0, 0, 0, 0, 0, 0, 0, 0, gamma, 0, 0, 0);
        MockDQN dqnMock = new MockDQN();
        IHistoryProcessor.Configuration hpConf = new IHistoryProcessor.Configuration(5, 1, 1, 1, 1, 0, 0, 2);
        MockAsyncGlobal asyncGlobalMock = new MockAsyncGlobal(dqnMock);
        AsyncNStepQLearningThreadDiscrete sut = new AsyncNStepQLearningThreadDiscrete<MockEncodable>(mdpMock, asyncGlobalMock, config, null, 0, 0);
        MockHistoryProcessor hpMock = new MockHistoryProcessor(hpConf);
        sut.setHistoryProcessor(hpMock);

        double[][] minitransObs = new double[][] {
                new double[] { 0.0, 2.0, 4.0, 6.0, 8.0 },
                new double[] { 2.0, 4.0, 6.0, 8.0, 10.0 },
                new double[] { 4.0, 6.0, 8.0, 10.0, 12.0 },
        };
        double[] outputs = new double[] { 1.0, 2.0, 3.0 };
        double[] rewards = new double[] { 0.0, 0.0, 3.0 };

        Stack<MiniTrans<Integer>> minitransList = new Stack<MiniTrans<Integer>>();
        for(int i = 0; i < 3; ++i) {
            INDArray obs = Nd4j.create(minitransObs[i]).reshape(5, 1, 1);
            INDArray[] output = new INDArray[] {
                    Nd4j.zeros(5)
            };
            output[0].putScalar(i, outputs[i]);
            minitransList.push(new MiniTrans<Integer>(obs, i, output, rewards[i]));
        }
        minitransList.push(new MiniTrans<Integer>(null, 0, null, 4.0)); // The special batch-ending MiniTrans

        // Act
        sut.calcGradient(dqnMock, minitransList);

        // Assert
        assertEquals(1, dqnMock.gradientParams.size());
        INDArray input = dqnMock.gradientParams.get(0).getFirst();
        INDArray labels = dqnMock.gradientParams.get(0).getSecond();

        assertEquals(minitransObs.length, input.shape()[0]);
        for(int i = 0; i < minitransObs.length; ++i) {
            double[] expectedRow = minitransObs[i];
            assertEquals(expectedRow.length, input.shape()[1]);
            for(int j = 0; j < expectedRow.length; ++j) {
                assertEquals(expectedRow[j], input.getDouble(i, j), 0.00001);
            }
        }

        double latestReward = (gamma * 4.0) + 3.0;
        double[][] expectedLabels = new double[][] {
                new double[] { gamma * gamma * latestReward, 0.0, 0.0, 0.0, 0.0 },
                new double[] { 0.0, gamma * latestReward, 0.0, 0.0, 0.0 },
                new double[] { 0.0, 0.0, latestReward, 0.0, 0.0 },
        };
        assertEquals(minitransObs.length, labels.shape()[0]);
        for(int i = 0; i < minitransObs.length; ++i) {
            double[] expectedRow = expectedLabels[i];
            assertEquals(expectedRow.length, labels.shape()[1]);
            for(int j = 0; j < expectedRow.length; ++j) {
                assertEquals(expectedRow[j], labels.getDouble(i, j), 0.00001);
            }
        }
    }
}

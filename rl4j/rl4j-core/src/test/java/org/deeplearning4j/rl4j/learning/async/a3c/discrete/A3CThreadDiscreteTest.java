package org.deeplearning4j.rl4j.learning.async.a3c.discrete;

import org.deeplearning4j.nn.api.NeuralNetwork;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.rl4j.learning.IHistoryProcessor;
import org.deeplearning4j.rl4j.learning.async.MiniTrans;
import org.deeplearning4j.rl4j.learning.async.nstep.discrete.AsyncNStepQLearningDiscrete;
import org.deeplearning4j.rl4j.learning.async.nstep.discrete.AsyncNStepQLearningThreadDiscrete;
import org.deeplearning4j.rl4j.network.NeuralNet;
import org.deeplearning4j.rl4j.network.ac.IActorCritic;
import org.deeplearning4j.rl4j.support.*;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;

import java.io.IOException;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Stack;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

public class A3CThreadDiscreteTest {

    @Test
    public void refac_calcGradient() {
        // Arrange
        double gamma = 0.9;
        MockObservationSpace observationSpace = new MockObservationSpace();
        MockMDP mdpMock = new MockMDP(observationSpace);
        A3CDiscrete.A3CConfiguration config = new A3CDiscrete.A3CConfiguration(0, 0, 0, 0, 0, 0, 0, gamma, 0);
        MockActorCritic actorCriticMock = new MockActorCritic();
        IHistoryProcessor.Configuration hpConf = new IHistoryProcessor.Configuration(5, 1, 1, 1, 1, 0, 0, 2);
        MockAsyncGlobal<IActorCritic> asyncGlobalMock = new MockAsyncGlobal<IActorCritic>(actorCriticMock);
        A3CThreadDiscrete sut = new A3CThreadDiscrete<MockEncodable>(mdpMock, asyncGlobalMock, config, 0, null, 0);
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
        sut.calcGradient(actorCriticMock, minitransList);

        // Assert
        assertEquals(1, actorCriticMock.gradientParams.size());
        INDArray input = actorCriticMock.gradientParams.get(0).getFirst();
        INDArray[] labels = actorCriticMock.gradientParams.get(0).getSecond();

        assertEquals(minitransObs.length, input.shape()[0]);
        for(int i = 0; i < minitransObs.length; ++i) {
            double[] expectedRow = minitransObs[i];
            assertEquals(expectedRow.length, input.shape()[1]);
            for(int j = 0; j < expectedRow.length; ++j) {
                assertEquals(expectedRow[j], input.getDouble(i, j, 1, 1), 0.00001);
            }
        }

        double latestReward = (gamma * 4.0) + 3.0;
        double[] expectedLabels0 = new double[] { gamma * gamma * latestReward, gamma * latestReward, latestReward };
        for(int i = 0; i < expectedLabels0.length; ++i) {
            assertEquals(expectedLabels0[i], labels[0].getDouble(i), 0.00001);
        }
        double[][] expectedLabels1 = new double[][] {
                new double[] { 4.346, 0.0, 0.0, 0.0, 0.0 },
                new double[] { 0.0, gamma * latestReward, 0.0, 0.0, 0.0 },
                new double[] { 0.0, 0.0, latestReward, 0.0, 0.0 },
        };

        assertArrayEquals(new long[] { expectedLabels0.length, 1 }, labels[0].shape());

        for(int i = 0; i < expectedLabels1.length; ++i) {
            double[] expectedRow = expectedLabels1[i];
            assertEquals(expectedRow.length, labels[1].shape()[1]);
            for(int j = 0; j < expectedRow.length; ++j) {
                assertEquals(expectedRow[j], labels[1].getDouble(i, j), 0.00001);
            }
        }

    }

    public class MockActorCritic implements IActorCritic {

        public final List<Pair<INDArray, INDArray[]>> gradientParams = new ArrayList<>();

        @Override
        public NeuralNetwork[] getNeuralNetworks() {
            return new NeuralNetwork[0];
        }

        @Override
        public boolean isRecurrent() {
            return false;
        }

        @Override
        public void reset() {

        }

        @Override
        public void fit(INDArray input, INDArray[] labels) {

        }

        @Override
        public INDArray[] outputAll(INDArray batch) {
            return new INDArray[0];
        }

        @Override
        public IActorCritic clone() {
            return this;
        }

        @Override
        public void copy(NeuralNet from) {

        }

        @Override
        public void copy(IActorCritic from) {

        }

        @Override
        public Gradient[] gradient(INDArray input, INDArray[] labels) {
            gradientParams.add(new Pair<INDArray, INDArray[]>(input, labels));
            return new Gradient[0];
        }

        @Override
        public void applyGradient(Gradient[] gradient, int batchSize) {

        }

        @Override
        public void save(OutputStream streamValue, OutputStream streamPolicy) throws IOException {

        }

        @Override
        public void save(String pathValue, String pathPolicy) throws IOException {

        }

        @Override
        public double getLatestScore() {
            return 0;
        }

        @Override
        public void save(OutputStream os) throws IOException {

        }

        @Override
        public void save(String filename) throws IOException {

        }
    }
}

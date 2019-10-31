package org.deeplearning4j.rl4j.learning.async;

import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.rl4j.learning.IHistoryProcessor;
import org.deeplearning4j.rl4j.learning.Learning;
import org.deeplearning4j.rl4j.learning.listener.TrainingListenerList;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.policy.IPolicy;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.support.*;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Stack;

import static org.junit.Assert.assertEquals;

public class AsyncThreadDiscreteTest {

    @Test
    public void refac_AsyncThreadDiscrete_trainSubEpoch() {
        // Arrange
        MockNeuralNet nnMock = new MockNeuralNet();
        MockAsyncGlobal asyncGlobalMock = new MockAsyncGlobal(nnMock);
        MockObservationSpace observationSpace = new MockObservationSpace();
        MockMDP mdpMock = new MockMDP(observationSpace);
        TrainingListenerList listeners = new TrainingListenerList();
        MockPolicy policyMock = new MockPolicy();
        MockAsyncConfiguration config = new MockAsyncConfiguration(5, 10, 0, 0, 0, 5,0, 0, 0, 0);
        IHistoryProcessor.Configuration hpConf = new IHistoryProcessor.Configuration(5, 4, 4, 4, 4, 0, 0, 2);
        MockHistoryProcessor hpMock = new MockHistoryProcessor(hpConf);
        TestAsyncThreadDiscrete sut = new TestAsyncThreadDiscrete(asyncGlobalMock, mdpMock, listeners, 0, 0, policyMock, config, hpMock);
        MockEncodable obs = new MockEncodable(123);

        hpMock.add(Learning.getInput(mdpMock, new MockEncodable(1)));
        hpMock.add(Learning.getInput(mdpMock, new MockEncodable(2)));
        hpMock.add(Learning.getInput(mdpMock, new MockEncodable(3)));
        hpMock.add(Learning.getInput(mdpMock, new MockEncodable(4)));
        hpMock.add(Learning.getInput(mdpMock, new MockEncodable(5)));

        // Act
        AsyncThread.SubEpochReturn<MockEncodable> result = sut.trainSubEpoch(obs, 2);

        // Assert
        assertEquals(4, result.getSteps());
        assertEquals(6.0, result.getReward(), 0.00001);
        assertEquals(0.0, result.getScore(), 0.00001);
        assertEquals(3.0, result.getLastObs().toArray()[0], 0.00001);
        assertEquals(1, asyncGlobalMock.enqueueCallCount);

        // HistoryProcessor
        assertEquals(10, hpMock.addCallCount);
        double[] expectedRecordValues = new double[] { 123.0, 0.0, 1.0, 2.0, 3.0 };
        assertEquals(expectedRecordValues.length, hpMock.recordCalls.size());
        for(int i = 0; i < expectedRecordValues.length; ++i) {
            assertEquals(expectedRecordValues[i], hpMock.recordCalls.get(i).getDouble(0), 0.00001);
        }

        // Policy
        double[][] expectedPolicyInputs = new double[][] {
                new double[] { 2.0, 3.0, 4.0, 5.0, 123.0 },
                new double[] { 3.0, 4.0, 5.0, 123.0, 0.0 },
                new double[] { 4.0, 5.0, 123.0, 0.0, 1.0 },
                new double[] { 5.0, 123.0, 0.0, 1.0, 2.0 },
        };
        assertEquals(expectedPolicyInputs.length, policyMock.actionInputs.size());
        for(int i = 0; i < expectedPolicyInputs.length; ++i) {
            double[] expectedRow = expectedPolicyInputs[i];
            INDArray input = policyMock.actionInputs.get(i);
            assertEquals(expectedRow.length, input.shape()[0]);
            for(int j = 0; j < expectedRow.length; ++j) {
                assertEquals(expectedRow[j], 255.0 * input.getDouble(j), 0.00001);
            }
        }

        // NeuralNetwork
        assertEquals(1, nnMock.copyCallCount);
        double[][] expectedNNInputs = new double[][] {
                new double[] { 2.0, 3.0, 4.0, 5.0, 123.0 },
                new double[] { 3.0, 4.0, 5.0, 123.0, 0.0 },
                new double[] { 4.0, 5.0, 123.0, 0.0, 1.0 },
                new double[] { 5.0, 123.0, 0.0, 1.0, 2.0 },
                new double[] { 123.0, 0.0, 1.0, 2.0, 3.0 },
        };
        assertEquals(expectedNNInputs.length, nnMock.outputAllInputs.size());
        for(int i = 0; i < expectedNNInputs.length; ++i) {
            double[] expectedRow = expectedNNInputs[i];
            INDArray input = nnMock.outputAllInputs.get(i);
            assertEquals(expectedRow.length, input.shape()[0]);
            for(int j = 0; j < expectedRow.length; ++j) {
                assertEquals(expectedRow[j], 255.0 * input.getDouble(j), 0.00001);
            }
        }

    }

    public static class TestAsyncThreadDiscrete extends AsyncThreadDiscrete<MockEncodable, MockNeuralNet> {

        private final IAsyncGlobal<MockNeuralNet> asyncGlobal;
        private final MockPolicy policy;
        private final MockAsyncConfiguration config;

        public TestAsyncThreadDiscrete(IAsyncGlobal<MockNeuralNet> asyncGlobal, MDP<MockEncodable, Integer, DiscreteSpace> mdp,
                                       TrainingListenerList listeners, int threadNumber, int deviceNum, MockPolicy policy,
                                       MockAsyncConfiguration config, IHistoryProcessor hp) {
            super(asyncGlobal, mdp, listeners, threadNumber, deviceNum);
            this.asyncGlobal = asyncGlobal;
            this.policy = policy;
            this.config = config;
            setHistoryProcessor(hp);
        }

        @Override
        public Gradient[] calcGradient(MockNeuralNet mockNeuralNet, Stack<MiniTrans<Integer>> rewards) {
            return new Gradient[0];
        }

        @Override
        protected IAsyncGlobal<MockNeuralNet> getAsyncGlobal() {
            return asyncGlobal;
        }

        @Override
        protected AsyncConfiguration getConf() {
            return config;
        }

        @Override
        protected IPolicy<MockEncodable, Integer> getPolicy(MockNeuralNet net) {
            return policy;
        }
    }
}

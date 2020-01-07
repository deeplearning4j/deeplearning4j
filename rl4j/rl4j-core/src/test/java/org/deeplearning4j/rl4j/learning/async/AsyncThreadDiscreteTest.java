package org.deeplearning4j.rl4j.learning.async;

import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.rl4j.learning.IHistoryProcessor;
import org.deeplearning4j.rl4j.learning.listener.TrainingListenerList;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.observation.Observation;
import org.deeplearning4j.rl4j.policy.IPolicy;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.support.*;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.ArrayList;
import java.util.List;
import java.util.Stack;

import static org.junit.Assert.assertEquals;

public class AsyncThreadDiscreteTest {

    @Test
    public void refac_AsyncThreadDiscrete_trainSubEpoch() {
        // Arrange
        int numEpochs = 1;
        MockNeuralNet nnMock = new MockNeuralNet();
        IHistoryProcessor.Configuration hpConf = new IHistoryProcessor.Configuration(5, 4, 4, 4, 4, 0, 0, 2);
        MockHistoryProcessor hpMock = new MockHistoryProcessor(hpConf);
        MockAsyncGlobal asyncGlobalMock = new MockAsyncGlobal(nnMock);
        asyncGlobalMock.setMaxLoops(hpConf.getSkipFrame() * numEpochs);
        MockObservationSpace observationSpace = new MockObservationSpace();
        MockMDP mdpMock = new MockMDP(observationSpace);
        TrainingListenerList listeners = new TrainingListenerList();
        MockPolicy policyMock = new MockPolicy();
        MockAsyncConfiguration config = new MockAsyncConfiguration(5, 100, 0, 0, 2, 5,0, 0, 0, 0);
        TestAsyncThreadDiscrete sut = new TestAsyncThreadDiscrete(asyncGlobalMock, mdpMock, listeners, 0, 0, policyMock, config, hpMock);

        // Act
        sut.run();

        // Assert
        assertEquals(2, sut.trainSubEpochResults.size());
        double[][] expectedLastObservations = new double[][] {
            new double[] { 4.0, 6.0, 8.0, 10.0, 12.0 },
            new double[] { 8.0, 10.0, 12.0, 14.0, 16.0 },
        };
        double[] expectedSubEpochReturnRewards = new double[] { 42.0, 58.0 };
        for(int i = 0; i < 2; ++i) {
            AsyncThread.SubEpochReturn result = sut.trainSubEpochResults.get(i);
            assertEquals(4, result.getSteps());
            assertEquals(expectedSubEpochReturnRewards[i], result.getReward(), 0.00001);
            assertEquals(0.0, result.getScore(), 0.00001);

            double[] expectedLastObservation = expectedLastObservations[i];
            assertEquals(expectedLastObservation.length, result.getLastObs().getData().shape()[1]);
            for(int j = 0; j < expectedLastObservation.length; ++j) {
                assertEquals(expectedLastObservation[j], 255.0 * result.getLastObs().getData().getDouble(j), 0.00001);
            }
        }
        assertEquals(2, asyncGlobalMock.enqueueCallCount);

        // HistoryProcessor
        double[] expectedAddValues = new double[] { 0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0 };
        assertEquals(expectedAddValues.length, hpMock.addCalls.size());
        for(int i = 0; i < expectedAddValues.length; ++i) {
            assertEquals(expectedAddValues[i], hpMock.addCalls.get(i).getDouble(0), 0.00001);
        }

        double[] expectedRecordValues = new double[] { 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, };
        assertEquals(expectedRecordValues.length, hpMock.recordCalls.size());
        for(int i = 0; i < expectedRecordValues.length; ++i) {
            assertEquals(expectedRecordValues[i], hpMock.recordCalls.get(i).getDouble(0), 0.00001);
        }

        // Policy
        double[][] expectedPolicyInputs = new double[][] {
                new double[] { 0.0, 2.0, 4.0, 6.0, 8.0 },
                new double[] { 2.0, 4.0, 6.0, 8.0, 10.0 },
                new double[] { 4.0, 6.0, 8.0, 10.0, 12.0 },
                new double[] { 6.0, 8.0, 10.0, 12.0, 14.0 },
        };
        assertEquals(expectedPolicyInputs.length, policyMock.actionInputs.size());
        for(int i = 0; i < expectedPolicyInputs.length; ++i) {
            double[] expectedRow = expectedPolicyInputs[i];
            INDArray input = policyMock.actionInputs.get(i);
            assertEquals(expectedRow.length, input.shape()[1]);
            for(int j = 0; j < expectedRow.length; ++j) {
                assertEquals(expectedRow[j], 255.0 * input.getDouble(j), 0.00001);
            }
        }

        // NeuralNetwork
        assertEquals(2, nnMock.copyCallCount);
        double[][] expectedNNInputs = new double[][] {
                new double[] { 0.0, 2.0, 4.0, 6.0, 8.0 },
                new double[] { 2.0, 4.0, 6.0, 8.0, 10.0 },
                new double[] { 4.0, 6.0, 8.0, 10.0, 12.0 }, // FIXME: This one comes from the computation of output of the last minitrans
                new double[] { 4.0, 6.0, 8.0, 10.0, 12.0 },
                new double[] { 6.0, 8.0, 10.0, 12.0, 14.0 },
                new double[] { 8.0, 10.0, 12.0, 14.0, 16.0 }, // FIXME: This one comes from the computation of output of the last minitrans
        };
        assertEquals(expectedNNInputs.length, nnMock.outputAllInputs.size());
        for(int i = 0; i < expectedNNInputs.length; ++i) {
            double[] expectedRow = expectedNNInputs[i];
            INDArray input = nnMock.outputAllInputs.get(i);
            assertEquals(expectedRow.length, input.shape()[1]);
            for(int j = 0; j < expectedRow.length; ++j) {
                assertEquals(expectedRow[j], 255.0 * input.getDouble(j), 0.00001);
            }
        }

        int arrayIdx = 0;
        double[][][] expectedMinitransObs = new double[][][] {
            new double[][] {
                    new double[] { 0.0, 2.0, 4.0, 6.0, 8.0 },
                    new double[] { 2.0, 4.0, 6.0, 8.0, 10.0 },
                    new double[] { 4.0, 6.0, 8.0, 10.0, 12.0 }, // FIXME: The last minitrans contains the next observation
            },
            new double[][] {
                    new double[] { 4.0, 6.0, 8.0, 10.0, 12.0 },
                    new double[] { 6.0, 8.0, 10.0, 12.0, 14.0 },
                    new double[] { 8.0, 10.0, 12.0, 14.0, 16.0 }, // FIXME: The last minitrans contains the next observation
            }
        };
        double[] expectedOutputs = new double[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
        double[] expectedRewards = new double[] { 0.0, 0.0, 3.0, 0.0, 0.0, 6.0 };

        assertEquals(2, sut.rewards.size());
        for(int rewardIdx = 0; rewardIdx < 2; ++rewardIdx) {
            Stack<MiniTrans<Integer>> miniTransStack = sut.rewards.get(rewardIdx);

            for (int i = 0; i < expectedMinitransObs[rewardIdx].length; ++i) {
                MiniTrans minitrans = miniTransStack.get(i);

                // Observation
                double[] expectedRow = expectedMinitransObs[rewardIdx][i];
                INDArray realRewards = minitrans.getObs();
                assertEquals(expectedRow.length, realRewards.shape()[1]);
                for (int j = 0; j < expectedRow.length; ++j) {
                    assertEquals("row: "+ i + " col: " + j, expectedRow[j], 255.0 * realRewards.getDouble(j), 0.00001);
                }

                assertEquals(expectedOutputs[arrayIdx], minitrans.getOutput()[0].getDouble(0), 0.00001);
                assertEquals(expectedRewards[arrayIdx], minitrans.getReward(), 0.00001);
                ++arrayIdx;
            }
        }
    }

    public static class TestAsyncThreadDiscrete extends AsyncThreadDiscrete<MockEncodable, MockNeuralNet> {

        private final MockAsyncGlobal asyncGlobal;
        private final MockPolicy policy;
        private final MockAsyncConfiguration config;

        public final List<SubEpochReturn> trainSubEpochResults = new ArrayList<SubEpochReturn>();
        public final List<Stack<MiniTrans<Integer>>> rewards = new ArrayList<Stack<MiniTrans<Integer>>>();

        public TestAsyncThreadDiscrete(MockAsyncGlobal asyncGlobal, MDP<MockEncodable, Integer, DiscreteSpace> mdp,
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
            this.rewards.add(rewards);
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

        @Override
        public SubEpochReturn trainSubEpoch(Observation sObs, int nstep) {
            asyncGlobal.increaseCurrentLoop();
            SubEpochReturn result = super.trainSubEpoch(sObs, nstep);
            trainSubEpochResults.add(result);
            return result;
        }
    }
}

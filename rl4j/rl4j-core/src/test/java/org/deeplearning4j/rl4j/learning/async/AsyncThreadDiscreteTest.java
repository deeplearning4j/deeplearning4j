/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
 * Copyright (c) 2020 Konduit K.K.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.rl4j.learning.async;

import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.rl4j.experience.ExperienceHandler;
import org.deeplearning4j.rl4j.experience.StateActionPair;
import org.deeplearning4j.rl4j.learning.IHistoryProcessor;
import org.deeplearning4j.rl4j.learning.configuration.IAsyncLearningConfiguration;
import org.deeplearning4j.rl4j.learning.listener.TrainingListenerList;
import org.deeplearning4j.rl4j.learning.sync.Transition;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.observation.Observation;
import org.deeplearning4j.rl4j.policy.IPolicy;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.support.*;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.ArrayList;
import java.util.List;

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
        MockAsyncConfiguration config = new MockAsyncConfiguration(5L, 100, 0,0, 0, 0, 0, 0, 2, 5);
        MockExperienceHandler experienceHandlerMock = new MockExperienceHandler();
        MockUpdateAlgorithm updateAlgorithmMock = new MockUpdateAlgorithm();
        TestAsyncThreadDiscrete sut = new TestAsyncThreadDiscrete(asyncGlobalMock, mdpMock, listeners, 0, 0, policyMock, config, hpMock, experienceHandlerMock, updateAlgorithmMock);
        sut.getLegacyMDPWrapper().setTransformProcess(MockMDP.buildTransformProcess(observationSpace.getShape(), hpConf.getSkipFrame(), hpConf.getHistoryLength()));

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

        // ExperienceHandler
        double[][] expectedExperienceHandlerInputs = new double[][] {
                new double[] { 0.0, 2.0, 4.0, 6.0, 8.0 },
                new double[] { 2.0, 4.0, 6.0, 8.0, 10.0 },
                new double[] { 4.0, 6.0, 8.0, 10.0, 12.0 },
                new double[] { 6.0, 8.0, 10.0, 12.0, 14.0 },
        };
        assertEquals(expectedExperienceHandlerInputs.length, experienceHandlerMock.addExperienceArgs.size());
        for(int i = 0; i < expectedExperienceHandlerInputs.length; ++i) {
            double[] expectedRow = expectedExperienceHandlerInputs[i];
            INDArray input = experienceHandlerMock.addExperienceArgs.get(i).getObservation().getData();
            assertEquals(expectedRow.length, input.shape()[1]);
            for(int j = 0; j < expectedRow.length; ++j) {
                assertEquals(expectedRow[j], 255.0 * input.getDouble(j), 0.00001);
            }
        }
    }

    public static class TestAsyncThreadDiscrete extends AsyncThreadDiscrete<MockEncodable, MockNeuralNet> {

        private final MockAsyncGlobal asyncGlobal;
        private final MockPolicy policy;
        private final MockAsyncConfiguration config;

        public final List<SubEpochReturn> trainSubEpochResults = new ArrayList<SubEpochReturn>();

        public TestAsyncThreadDiscrete(MockAsyncGlobal asyncGlobal, MDP<MockEncodable, Integer, DiscreteSpace> mdp,
                                       TrainingListenerList listeners, int threadNumber, int deviceNum, MockPolicy policy,
                                       MockAsyncConfiguration config, IHistoryProcessor hp,
                                       ExperienceHandler<Integer, Transition<Integer>> experienceHandler,
                                       UpdateAlgorithm<MockNeuralNet> updateAlgorithm) {
            super(asyncGlobal, mdp, listeners, threadNumber, deviceNum);
            this.asyncGlobal = asyncGlobal;
            this.policy = policy;
            this.config = config;
            setHistoryProcessor(hp);
            setExperienceHandler(experienceHandler);
            setUpdateAlgorithm(updateAlgorithm);
        }

        @Override
        protected IAsyncGlobal<MockNeuralNet> getAsyncGlobal() {
            return asyncGlobal;
        }

        @Override
        protected IAsyncLearningConfiguration getConf() {
            return config;
        }

        @Override
        protected IPolicy<MockEncodable, Integer> getPolicy(MockNeuralNet net) {
            return policy;
        }

        @Override
        protected UpdateAlgorithm<MockNeuralNet> buildUpdateAlgorithm() {
            return null;
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

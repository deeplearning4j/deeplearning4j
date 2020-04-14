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

package org.deeplearning4j.rl4j.learning.sync.qlearning.discrete;

import org.deeplearning4j.rl4j.experience.ExperienceHandler;
import org.deeplearning4j.rl4j.experience.StateActionPair;
import org.deeplearning4j.rl4j.learning.IHistoryProcessor;
import org.deeplearning4j.rl4j.learning.configuration.QLearningConfiguration;
import org.deeplearning4j.rl4j.learning.sync.IExpReplay;
import org.deeplearning4j.rl4j.learning.sync.Transition;
import org.deeplearning4j.rl4j.learning.sync.qlearning.QLearning;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.network.dqn.IDQN;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.support.*;
import org.deeplearning4j.rl4j.util.DataManagerTrainingListener;
import org.deeplearning4j.rl4j.util.IDataManager;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.*;

public class QLearningDiscreteTest {
    @Test
    public void refac_QLearningDiscrete_trainStep() {
        // Arrange
        MockObservationSpace observationSpace = new MockObservationSpace();
        MockDQN dqn = new MockDQN();
        MockRandom random = new MockRandom(new double[]{
                0.7309677600860596,
                0.8314409852027893,
                0.2405363917350769,
                0.6063451766967773,
                0.6374173760414124,
                0.3090505599975586,
                0.5504369735717773,
                0.11700659990310669
        },
                new int[]{1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4});
        MockMDP mdp = new MockMDP(observationSpace, random);

        int initStepCount = 8;

        QLearningConfiguration conf = QLearningConfiguration.builder()
                .seed(0L)
                .maxEpochStep(24)
                .maxStep(0)
                .expRepMaxSize(5).batchSize(1).targetDqnUpdateFreq(1000)
                .updateStart(initStepCount)
                .rewardFactor(1.0)
                .gamma(0)
                .errorClamp(0)
                .minEpsilon(0)
                .epsilonNbStep(0)
                .doubleDQN(true)
                .build();

        MockDataManager dataManager = new MockDataManager(false);
        MockExperienceHandler experienceHandler = new MockExperienceHandler();
        TestQLearningDiscrete sut = new TestQLearningDiscrete(mdp, dqn, conf, dataManager, experienceHandler, 10, random);
        IHistoryProcessor.Configuration hpConf = new IHistoryProcessor.Configuration(5, 4, 4, 4, 4, 0, 0, 2);
        MockHistoryProcessor hp = new MockHistoryProcessor(hpConf);
        sut.setHistoryProcessor(hp);
        sut.getLegacyMDPWrapper().setTransformProcess(MockMDP.buildTransformProcess(observationSpace.getShape(), hpConf.getSkipFrame(), hpConf.getHistoryLength()));
        List<QLearning.QLStepReturn<MockEncodable>> results = new ArrayList<>();

        // Act
        IDataManager.StatEntry result = sut.trainEpoch();

        // Assert
        // HistoryProcessor calls
        double[] expectedRecords = new double[]{0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
        assertEquals(expectedRecords.length, hp.recordCalls.size());
        for (int i = 0; i < expectedRecords.length; ++i) {
            assertEquals(expectedRecords[i], hp.recordCalls.get(i).getDouble(0), 0.0001);
        }
        assertEquals(0, hp.startMonitorCallCount);
        assertEquals(0, hp.stopMonitorCallCount);

        // DQN calls
        assertEquals(1, dqn.fitParams.size());
        assertEquals(123.0, dqn.fitParams.get(0).getFirst().getDouble(0), 0.001);
        assertEquals(234.0, dqn.fitParams.get(0).getSecond().getDouble(0), 0.001);
        assertEquals(14, dqn.outputParams.size());
        double[][] expectedDQNOutput = new double[][]{
                new double[]{0.0, 2.0, 4.0, 6.0, 8.0},
                new double[]{2.0, 4.0, 6.0, 8.0, 10.0},
                new double[]{2.0, 4.0, 6.0, 8.0, 10.0},
                new double[]{4.0, 6.0, 8.0, 10.0, 12.0},
                new double[]{6.0, 8.0, 10.0, 12.0, 14.0},
                new double[]{6.0, 8.0, 10.0, 12.0, 14.0},
                new double[]{8.0, 10.0, 12.0, 14.0, 16.0},
                new double[]{8.0, 10.0, 12.0, 14.0, 16.0},
                new double[]{10.0, 12.0, 14.0, 16.0, 18.0},
                new double[]{10.0, 12.0, 14.0, 16.0, 18.0},
                new double[]{12.0, 14.0, 16.0, 18.0, 20.0},
                new double[]{12.0, 14.0, 16.0, 18.0, 20.0},
                new double[]{14.0, 16.0, 18.0, 20.0, 22.0},
                new double[]{14.0, 16.0, 18.0, 20.0, 22.0},
        };
        for (int i = 0; i < expectedDQNOutput.length; ++i) {
            INDArray outputParam = dqn.outputParams.get(i);

            assertEquals(5, outputParam.shape()[1]);
            assertEquals(1, outputParam.shape()[2]);

            double[] expectedRow = expectedDQNOutput[i];
            for (int j = 0; j < expectedRow.length; ++j) {
                assertEquals("row: " + i + " col: " + j, expectedRow[j], 255.0 * outputParam.getDouble(j), 0.00001);
            }
        }

        // MDP calls
        assertArrayEquals(new Integer[]{0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 4, 4, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4}, mdp.actions.toArray());

        // ExperienceHandler calls
        double[] expectedTrRewards = new double[] { 9.0, 21.0, 25.0, 29.0, 33.0, 37.0, 41.0 };
        int[] expectedTrActions = new int[] { 1, 4, 2, 4, 4, 4, 4, 4 };
        double[] expectedTrNextObservation = new double[] { 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0 };
        double[][] expectedTrObservations = new double[][] {
                new double[] { 0.0, 2.0, 4.0, 6.0, 8.0 },
                new double[] { 2.0, 4.0, 6.0, 8.0, 10.0 },
                new double[] { 4.0, 6.0, 8.0, 10.0, 12.0 },
                new double[] { 6.0, 8.0, 10.0, 12.0, 14.0 },
                new double[] { 8.0, 10.0, 12.0, 14.0, 16.0 },
                new double[] { 10.0, 12.0, 14.0, 16.0, 18.0 },
                new double[] { 12.0, 14.0, 16.0, 18.0, 20.0 },
                new double[] { 14.0, 16.0, 18.0, 20.0, 22.0 },
        };

        assertEquals(expectedTrObservations.length, experienceHandler.addExperienceArgs.size());
        for(int i = 0; i < expectedTrRewards.length; ++i) {
            StateActionPair<Integer> stateActionPair = experienceHandler.addExperienceArgs.get(i);
            assertEquals(expectedTrRewards[i], stateActionPair.getReward(), 0.0001);
            assertEquals((int)expectedTrActions[i], (int)stateActionPair.getAction());
            for(int j = 0; j < expectedTrObservations[i].length; ++j) {
                assertEquals("row: "+ i + " col: " + j, expectedTrObservations[i][j], 255.0 * stateActionPair.getObservation().getData().getDouble(0, j, 0), 0.0001);
            }
        }
        assertEquals(expectedTrNextObservation[expectedTrNextObservation.length - 1], 255.0 * experienceHandler.finalObservation.getData().getDouble(0), 0.0001);

        // trainEpoch result
        assertEquals(initStepCount + 16, result.getStepCounter());
        assertEquals(300.0, result.getReward(), 0.00001);
        assertTrue(dqn.hasBeenReset);
        assertTrue(((MockDQN) sut.getTargetQNetwork()).hasBeenReset);
    }

    public static class TestQLearningDiscrete extends QLearningDiscrete<MockEncodable> {
        public TestQLearningDiscrete(MDP<MockEncodable, Integer, DiscreteSpace> mdp, IDQN dqn,
                                     QLearningConfiguration conf, IDataManager dataManager, ExperienceHandler<Integer, Transition<Integer>> experienceHandler,
                                     int epsilonNbStep, Random rnd) {
            super(mdp, dqn, conf, epsilonNbStep, rnd);
            addListener(new DataManagerTrainingListener(dataManager));
            setExperienceHandler(experienceHandler);
        }

        @Override
        protected DataSet setTarget(List<Transition<Integer>> transitions) {
            return new org.nd4j.linalg.dataset.DataSet(Nd4j.create(new double[] { 123.0 }), Nd4j.create(new double[] { 234.0 }));
        }

        @Override
        public IDataManager.StatEntry trainEpoch() {
            return super.trainEpoch();
        }
    }
}

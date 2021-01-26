/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.rl4j.learning.sync;

import org.deeplearning4j.rl4j.experience.StateActionRewardState;
import org.deeplearning4j.rl4j.observation.Observation;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertEquals;

public class StateActionRewardStateTest {
    @Test
    public void when_callingCtorWithoutHistory_expect_2DObservationAndNextObservation() {
        // Arrange
        double[] obs = new double[] { 1.0, 2.0, 3.0 };
        Observation observation = buildObservation(obs);

        double[] nextObs = new double[] { 10.0, 20.0, 30.0 };
        Observation nextObservation = buildObservation(nextObs);

        // Act
        StateActionRewardState stateActionRewardState = buildTransition(observation, 123, 234.0, nextObservation);

        // Assert
        double[][] expectedObservation = new double[][] { obs };
        assertExpected(expectedObservation, stateActionRewardState.getObservation().getChannelData(0));

        double[][] expectedNextObservation = new double[][] { nextObs };
        assertExpected(expectedNextObservation, stateActionRewardState.getNextObservation().getChannelData(0));

        assertEquals(123, stateActionRewardState.getAction());
        assertEquals(234.0, stateActionRewardState.getReward(), 0.0001);
    }

    @Test
    public void when_callingCtorWithHistory_expect_ObservationAndNextWithHistory() {
        // Arrange
        double[][] obs = new double[][] {
                { 0.0, 1.0, 2.0 },
                { 3.0, 4.0, 5.0 },
                { 6.0, 7.0, 8.0 },
        };
        Observation observation = buildObservation(obs);

        double[][] nextObs = new double[][] {
                { 10.0, 11.0, 12.0 },
                { 0.0, 1.0, 2.0 },
                { 3.0, 4.0, 5.0 },
        };
        Observation nextObservation = buildObservation(nextObs);

        // Act
        StateActionRewardState stateActionRewardState = buildTransition(observation, 123, 234.0, nextObservation);

        // Assert
        assertExpected(obs, stateActionRewardState.getObservation().getChannelData(0));

        assertExpected(nextObs, stateActionRewardState.getNextObservation().getChannelData(0));

        assertEquals(123, stateActionRewardState.getAction());
        assertEquals(234.0, stateActionRewardState.getReward(), 0.0001);
    }

    private Observation buildObservation(double[][] obs) {
        INDArray[] history = new INDArray[] {
                Nd4j.create(obs[0]).reshape(1, 3),
                Nd4j.create(obs[1]).reshape(1, 3),
                Nd4j.create(obs[2]).reshape(1, 3),
        };
        return new Observation(Nd4j.concat(0, history));
    }

    private Observation buildObservation(double[] obs) {
        return new Observation(Nd4j.create(obs).reshape(1, 3));
    }

    private Observation buildNextObservation(double[][] obs, double[] nextObs) {
        INDArray[] nextHistory = new INDArray[] {
                Nd4j.create(nextObs).reshape(1, 3),
                Nd4j.create(obs[0]).reshape(1, 3),
                Nd4j.create(obs[1]).reshape(1, 3),
        };
        return new Observation(Nd4j.concat(0, nextHistory));
    }

    private StateActionRewardState buildTransition(Observation observation, int action, double reward, Observation nextObservation) {
        StateActionRewardState result = new StateActionRewardState(observation, action, reward, false);
        result.setNextObservation(nextObservation);

        return result;
    }

    private void assertExpected(double[] expected, INDArray actual) {
        long[] shape = actual.shape();
        assertEquals(2, shape.length);
        assertEquals(1, shape[0]);
        assertEquals(expected.length, shape[1]);
        for(int i = 0; i < expected.length; ++i) {
            assertEquals(expected[i], actual.getDouble(0, i), 0.0001);
        }
    }

    private void assertExpected(double[][] expected, INDArray actual) {
        long[] shape = actual.shape();
        assertEquals(2, shape.length);
        assertEquals(expected.length, shape[0]);
        assertEquals(expected[0].length, shape[1]);

        for(int i = 0; i < expected.length; ++i) {
            double[] expectedLine = expected[i];
            for(int j = 0; j < expectedLine.length; ++j) {
                assertEquals(expectedLine[j], actual.getDouble(i, j), 0.0001);
            }
        }
    }

    private void assertExpected(double[][][] expected, INDArray actual) {
        long[] shape = actual.shape();
        assertEquals(3, shape.length);
        assertEquals(expected.length, shape[0]);
        assertEquals(expected[0].length, shape[1]);
        assertEquals(expected[0][0].length, shape[2]);

        for(int i = 0; i < expected.length; ++i) {
            double[][] expected2D = expected[i];
            for(int j = 0; j < expected2D.length; ++j) {
                double[] expectedLine = expected2D[j];
                for (int k = 0; k < expectedLine.length; ++k) {
                    assertEquals(expectedLine[k], actual.getDouble(i, j, k), 0.0001);
                }
            }
        }

    }
}

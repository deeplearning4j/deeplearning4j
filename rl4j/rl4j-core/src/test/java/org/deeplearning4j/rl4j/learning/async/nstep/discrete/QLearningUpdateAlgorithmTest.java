/*******************************************************************************
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

package org.deeplearning4j.rl4j.learning.async.nstep.discrete;

import org.deeplearning4j.rl4j.experience.StateActionPair;
import org.deeplearning4j.rl4j.learning.async.AsyncGlobal;
import org.deeplearning4j.rl4j.learning.async.UpdateAlgorithm;
import org.deeplearning4j.rl4j.observation.Observation;
import org.deeplearning4j.rl4j.support.MockDQN;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.Mock;
import org.mockito.junit.MockitoJUnitRunner;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;

@RunWith(MockitoJUnitRunner.class)
public class QLearningUpdateAlgorithmTest {

    @Mock
    AsyncGlobal mockAsyncGlobal;

    @Test
    public void when_isTerminal_expect_initRewardIs0() {
        // Arrange
        MockDQN dqnMock = new MockDQN();
        UpdateAlgorithm sut = new QLearningUpdateAlgorithm(new int[] { 1 }, 1, 1.0);
        final Observation observation = new Observation(Nd4j.zeros(1));
        List<StateActionPair<Integer>> experience = new ArrayList<StateActionPair<Integer>>() {
            {
                add(new StateActionPair<Integer>(observation, 0, 0.0, true));
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
        UpdateAlgorithm sut = new QLearningUpdateAlgorithm(new int[] { 2 }, 2, 1.0);
        final Observation observation = new Observation(Nd4j.create(new double[] { -123.0, -234.0 }));
        List<StateActionPair<Integer>> experience = new ArrayList<StateActionPair<Integer>>() {
            {
                add(new StateActionPair<Integer>(observation, 0, 0.0, false));
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
    public void when_callingWithMultipleExperiences_expect_gradientsAreValid() {
        // Arrange
        double gamma = 0.9;
        UpdateAlgorithm sut = new QLearningUpdateAlgorithm(new int[] { 2 }, 2, gamma);
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

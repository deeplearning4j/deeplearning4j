/*
 *  ******************************************************************************
 *  *
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

package org.deeplearning4j.rl4j.learning.async.nstep.discrete;

import org.deeplearning4j.rl4j.experience.StateActionReward;
import org.deeplearning4j.rl4j.learning.async.AsyncGlobal;
import org.deeplearning4j.rl4j.learning.async.UpdateAlgorithm;
import org.deeplearning4j.rl4j.network.dqn.IDQN;
import org.deeplearning4j.rl4j.observation.Observation;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.junit.MockitoJUnitRunner;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.argThat;
import static org.mockito.Mockito.*;

@RunWith(MockitoJUnitRunner.class)
public class QLearningUpdateAlgorithmTest {

    @Mock
    AsyncGlobal mockAsyncGlobal;

    @Mock
    IDQN dqnMock;

    private UpdateAlgorithm sut;

    private void setup(double gamma) {
        // mock a neural net output -- just invert the sign of the input
        when(dqnMock.outputAll(any(INDArray.class))).thenAnswer(invocation -> new INDArray[] { invocation.getArgument(0, INDArray.class).mul(-1.0) });

        sut = new QLearningUpdateAlgorithm(2, gamma);
    }

    @Test
    public void when_isTerminal_expect_initRewardIs0() {
        // Arrange
        setup(1.0);

        final Observation observation = new Observation(Nd4j.zeros(1, 2));
        List<StateActionReward<Integer>> experience = new ArrayList<StateActionReward<Integer>>() {
            {
                add(new StateActionReward<Integer>(observation, 0, 0.0, true));
            }
        };

        // Act
        sut.computeGradients(dqnMock, experience);

        // Assert
        verify(dqnMock, times(1)).gradient(any(INDArray.class), argThat((INDArray x) -> x.getDouble(0) == 0.0));
    }

    @Test
    public void when_terminalAndNoTargetUpdate_expect_initRewardWithMaxQFromCurrent() {
        // Arrange
        setup(1.0);

        final Observation observation = new Observation(Nd4j.create(new double[] { -123.0, -234.0 }).reshape(1, 2));
        List<StateActionReward<Integer>> experience = new ArrayList<StateActionReward<Integer>>() {
            {
                add(new StateActionReward<Integer>(observation, 0, 0.0, false));
            }
        };

        // Act
        sut.computeGradients(dqnMock, experience);

        // Assert
        ArgumentCaptor<INDArray> argument = ArgumentCaptor.forClass(INDArray.class);

        verify(dqnMock, times(2)).outputAll(argument.capture());
        List<INDArray> values = argument.getAllValues();
        assertEquals(-123.0, values.get(0).getDouble(0, 0), 0.00001);
        assertEquals(-123.0, values.get(1).getDouble(0, 0), 0.00001);

        verify(dqnMock, times(1)).gradient(any(INDArray.class), argThat((INDArray x) -> x.getDouble(0) == 234.0));
    }

    @Test
    public void when_callingWithMultipleExperiences_expect_gradientsAreValid() {
        // Arrange
        double gamma = 0.9;
        setup(gamma);

        List<StateActionReward<Integer>> experience = new ArrayList<StateActionReward<Integer>>() {
            {
                add(new StateActionReward<Integer>(new Observation(Nd4j.create(new double[] { -1.1, -1.2 }).reshape(1, 2)), 0, 1.0, false));
                add(new StateActionReward<Integer>(new Observation(Nd4j.create(new double[] { -2.1, -2.2 }).reshape(1, 2)), 1, 2.0, true));
            }
        };

        // Act
        sut.computeGradients(dqnMock, experience);

        // Assert
        ArgumentCaptor<INDArray> features = ArgumentCaptor.forClass(INDArray.class);
        ArgumentCaptor<INDArray> targets = ArgumentCaptor.forClass(INDArray.class);
        verify(dqnMock, times(1)).gradient(features.capture(), targets.capture());

        // input side -- should be a stack of observations
        INDArray featuresValues = features.getValue();
        assertEquals(-1.1, featuresValues.getDouble(0, 0), 0.00001);
        assertEquals(-1.2, featuresValues.getDouble(0, 1), 0.00001);
        assertEquals(-2.1, featuresValues.getDouble(1, 0), 0.00001);
        assertEquals(-2.2, featuresValues.getDouble(1, 1), 0.00001);

        // target side
        INDArray targetsValues = targets.getValue();
        assertEquals(1.0 + gamma * 2.0, targetsValues.getDouble(0, 0), 0.00001);
        assertEquals(1.2, targetsValues.getDouble(0, 1), 0.00001);
        assertEquals(2.1, targetsValues.getDouble(1, 0), 0.00001);
        assertEquals(2.0, targetsValues.getDouble(1, 1), 0.00001);
    }
}

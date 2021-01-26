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

package org.deeplearning4j.rl4j.agent.learning.algorithm.nstepqlearning;

import org.deeplearning4j.rl4j.agent.learning.update.Features;
import org.deeplearning4j.rl4j.experience.StateActionReward;
import org.deeplearning4j.rl4j.network.CommonOutputNames;
import org.deeplearning4j.rl4j.network.IOutputNeuralNet;
import org.deeplearning4j.rl4j.network.NeuralNetOutput;
import org.deeplearning4j.rl4j.observation.Observation;
import org.junit.Test;
import org.mockito.ArgumentCaptor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.mockito.Mockito.*;

public class RecurrentNStepQLearningHelperTest {

    private final RecurrentNStepQLearningHelper sut = new RecurrentNStepQLearningHelper(3);

    @Test
    public void when_callingCreateValueLabels_expect_INDArrayWithCorrectShape() {
        // Arrange

        // Act
        INDArray result = sut.createLabels(4);

        // Assert
        assertArrayEquals(new long[] { 1, 3, 4 }, result.shape());
    }

    @Test
    public void when_callingGetExpectedQValues_expect_INDArrayWithCorrectShape() {
        // Arrange
        INDArray allExpectedQValues = Nd4j.create(new double[] { 1.1, 1.2, 2.1, 2.2 }).reshape(1, 2, 2);

        // Act
        INDArray result = sut.getExpectedQValues(allExpectedQValues, 1);

        // Assert
        assertEquals(1.2, result.getDouble(0), 0.00001);
        assertEquals(2.2, result.getDouble(1), 0.00001);
    }

    @Test
    public void when_callingSetLabels_expect_INDArrayWithCorrectShape() {
        // Arrange
        INDArray labels = Nd4j.zeros(1, 2, 2);
        INDArray data = Nd4j.create(new double[] { 1.1, 1.2 });

        // Act
        sut.setLabels(labels, 1, data);

        // Assert
        assertEquals(0.0, labels.getDouble(0, 0, 0), 0.00001);
        assertEquals(0.0, labels.getDouble(0, 1, 0), 0.00001);
        assertEquals(1.1, labels.getDouble(0, 0, 1), 0.00001);
        assertEquals(1.2, labels.getDouble(0, 1, 1), 0.00001);
    }

    @Test
    public void when_callingGetTargetExpectedQValuesOfLast_expect_INDArrayWithCorrectShape() {
        // Arrange
        List<StateActionReward<Integer>> experience = new ArrayList<StateActionReward<Integer>>() {
            {
                add(new StateActionReward<Integer>(new Observation(Nd4j.create(new double[] { 1.1, 1.2 }).reshape(1, 2, 1)), 0, 1.0, false));
                add(new StateActionReward<Integer>(new Observation(Nd4j.create(new double[] { 2.1, 2.2 }).reshape(1, 2, 1)), 1, 2.0, false));
            }
        };
        INDArray featuresData = Nd4j.create(new double[] { 1.0, 2.0, 3.0, 4.0 }).reshape(1, 2, 2);
        Features features = new Features(new INDArray[] { featuresData });
        IOutputNeuralNet targetMock = mock(IOutputNeuralNet.class);

        final NeuralNetOutput neuralNetOutput = new NeuralNetOutput();
        neuralNetOutput.put(CommonOutputNames.QValues, Nd4j.create(new double[] { -4.1, -4.2 }).reshape(1, 1, 2));
        when(targetMock.output(any(Features.class))).thenReturn(neuralNetOutput);

        // Act
        INDArray result = sut.getTargetExpectedQValuesOfLast(targetMock, experience, features);

        // Assert
        ArgumentCaptor<Features> captor = ArgumentCaptor.forClass(Features.class);
        verify(targetMock, times(1)).output(captor.capture());
        INDArray array = captor.getValue().get(0);
        assertEquals(1.0, array.getDouble(0, 0, 0), 0.00001);
        assertEquals(2.0, array.getDouble(0, 0, 1), 0.00001);
        assertEquals(3.0, array.getDouble(0, 1, 0), 0.00001);
        assertEquals(4.0, array.getDouble(0, 1, 1), 0.00001);

        assertEquals(-4.2, result.getDouble(0, 0, 0), 0.00001);
    }
}

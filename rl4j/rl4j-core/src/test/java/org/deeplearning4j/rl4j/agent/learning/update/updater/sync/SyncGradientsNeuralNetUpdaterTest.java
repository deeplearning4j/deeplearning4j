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

package org.deeplearning4j.rl4j.agent.learning.update.updater.sync;

import org.deeplearning4j.rl4j.agent.learning.update.Gradients;
import org.deeplearning4j.rl4j.agent.learning.update.updater.NeuralNetUpdaterConfiguration;
import org.deeplearning4j.rl4j.network.ITrainableNeuralNet;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.Mock;
import org.mockito.junit.MockitoJUnitRunner;

import static org.mockito.Mockito.*;

@RunWith(MockitoJUnitRunner.class)
public class SyncGradientsNeuralNetUpdaterTest {

    @Mock
    ITrainableNeuralNet currentMock;

    @Mock
    ITrainableNeuralNet targetMock;

    @Test
    public void when_callingUpdate_expect_currentUpdatedAndtargetNotChanged() {
        // Arrange
        NeuralNetUpdaterConfiguration configuration = NeuralNetUpdaterConfiguration.builder()
                .build();
        SyncGradientsNeuralNetUpdater sut = new SyncGradientsNeuralNetUpdater(currentMock, targetMock, configuration);
        Gradients gradients = new Gradients(10);

        // Act
        sut.update(gradients);

        // Assert
        verify(currentMock, times(1)).applyGradients(gradients);
        verify(targetMock, never()).applyGradients(any());
    }

    @Test
    public void when_callingUpdate_expect_targetUpdatedFromCurrentAtFrequency() {
        // Arrange
        NeuralNetUpdaterConfiguration configuration = NeuralNetUpdaterConfiguration.builder()
                .targetUpdateFrequency(3)
                .build();
        SyncGradientsNeuralNetUpdater sut = new SyncGradientsNeuralNetUpdater(currentMock, targetMock, configuration);
        Gradients gradients = new Gradients(10);

        // Act
        sut.update(gradients);
        sut.update(gradients);
        sut.update(gradients);

        // Assert
        verify(currentMock, never()).copyFrom(any());
        verify(targetMock, times(1)).copyFrom(currentMock);
    }
}

/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
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

import org.deeplearning4j.rl4j.agent.learning.update.FeaturesLabels;
import org.deeplearning4j.rl4j.agent.learning.update.updater.NeuralNetUpdaterConfiguration;
import org.deeplearning4j.rl4j.network.ITrainableNeuralNet;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.Mock;
import org.mockito.junit.MockitoJUnitRunner;

import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;
import static org.mockito.Mockito.*;

@RunWith(MockitoJUnitRunner.class)
public class SyncLabelsNeuralNetUpdaterTest {

    @Mock
    ITrainableNeuralNet threadCurrentMock;

    @Mock
    ITrainableNeuralNet targetMock;

    @Test
    public void when_callingUpdateWithTargetUpdateFrequencyAt0_expect_Exception() {
        // Arrange
        NeuralNetUpdaterConfiguration configuration = NeuralNetUpdaterConfiguration.builder()
                .targetUpdateFrequency(0)
                .build();
        try {
            SyncLabelsNeuralNetUpdater sut = new SyncLabelsNeuralNetUpdater(threadCurrentMock, targetMock, configuration);
            fail("IllegalArgumentException should have been thrown");
        } catch (IllegalArgumentException exception) {
            String expectedMessage = "Configuration: targetUpdateFrequency must be greater than 0, got:  [0]";
            String actualMessage = exception.getMessage();

            assertTrue(actualMessage.contains(expectedMessage));
        }

    }

    @Test
    public void when_callingUpdate_expect_gradientsComputedFromThreadCurrentAndAppliedOnGlobalCurrent() {
        // Arrange
        NeuralNetUpdaterConfiguration configuration = NeuralNetUpdaterConfiguration.builder()
                .build();
        SyncLabelsNeuralNetUpdater sut = new SyncLabelsNeuralNetUpdater(threadCurrentMock, targetMock, configuration);
        FeaturesLabels featureLabels = new FeaturesLabels(null);

        // Act
        sut.update(featureLabels);

        // Assert
        verify(threadCurrentMock, times(1)).fit(featureLabels);
        verify(targetMock, never()).fit(any());
    }

    @Test
    public void when_callingUpdate_expect_targetUpdatedFromGlobalCurrentAtFrequency() {
        // Arrange
        NeuralNetUpdaterConfiguration configuration = NeuralNetUpdaterConfiguration.builder()
                .targetUpdateFrequency(3)
                .build();
        SyncLabelsNeuralNetUpdater sut = new SyncLabelsNeuralNetUpdater(threadCurrentMock, targetMock, configuration);
        FeaturesLabels featureLabels = new FeaturesLabels(null);

        // Act
        sut.update(featureLabels);
        sut.update(featureLabels);
        sut.update(featureLabels);

        // Assert
        verify(threadCurrentMock, never()).copyFrom(any());
        verify(targetMock, times(1)).copyFrom(threadCurrentMock);
    }
}

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

package org.deeplearning4j.rl4j.agent.learning.update.updater.async;

import org.deeplearning4j.rl4j.agent.learning.update.Gradients;
import org.deeplearning4j.rl4j.agent.learning.update.updater.NeuralNetUpdaterConfiguration;
import org.deeplearning4j.rl4j.network.ITrainableNeuralNet;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.runner.RunWith;
import org.mockito.Mock;
import org.mockito.junit.MockitoJUnitRunner;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

@RunWith(MockitoJUnitRunner.class)
@Tag(TagNames.FILE_IO)
@NativeTag
public class AsyncSharedNetworksUpdateHandlerTest {

    @Mock
    ITrainableNeuralNet globalCurrentMock;

    @Mock
    ITrainableNeuralNet targetMock;

    @Test
    public void when_handleGradientsIsCalledWithoutTarget_expect_gradientsAppliedOnGlobalCurrent() {
        // Arrange
        NeuralNetUpdaterConfiguration configuration = NeuralNetUpdaterConfiguration.builder()
                .build();
        AsyncSharedNetworksUpdateHandler sut = new AsyncSharedNetworksUpdateHandler(globalCurrentMock, configuration);
        Gradients gradients = new Gradients(10);

        // Act
        sut.handleGradients(gradients);

        // Assert
        verify(globalCurrentMock, times(1)).applyGradients(gradients);
    }

    @Test
    public void when_handleGradientsIsCalledWithTarget_expect_gradientsAppliedOnGlobalCurrentAndTargetUpdated() {
        // Arrange
        NeuralNetUpdaterConfiguration configuration = NeuralNetUpdaterConfiguration.builder()
                .targetUpdateFrequency(2)
                .build();
        AsyncSharedNetworksUpdateHandler sut = new AsyncSharedNetworksUpdateHandler(globalCurrentMock, targetMock, configuration);
        Gradients gradients = new Gradients(10);

        // Act
        sut.handleGradients(gradients);
        sut.handleGradients(gradients);

        // Assert
        verify(globalCurrentMock, times(2)).applyGradients(gradients);
        verify(targetMock, times(1)).copyFrom(globalCurrentMock);
    }

    @Test
    public void when_configurationHasInvalidFrequency_expect_Exception() {
        try {
            NeuralNetUpdaterConfiguration configuration = NeuralNetUpdaterConfiguration.builder()
                    .targetUpdateFrequency(0)
                    .build();
            AsyncSharedNetworksUpdateHandler sut = new AsyncSharedNetworksUpdateHandler(globalCurrentMock, targetMock, configuration);

            fail("NullPointerException should have been thrown");
        } catch (IllegalArgumentException exception) {
            String expectedMessage = "Configuration: targetUpdateFrequency must be greater than 0, got:  [0]";
            String actualMessage = exception.getMessage();

            assertTrue(actualMessage.contains(expectedMessage));
        }
    }

}

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

package org.deeplearning4j.rl4j.network;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.rl4j.observation.Observation;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.junit.MockitoJUnitRunner;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;
import static org.mockito.Mockito.*;

@RunWith(MockitoJUnitRunner.class)
public class NetworkHelperTest {

    @Test
    public void when_callingBuildHandlerWithMapper_expect_correctlyBuiltNetworkHandler() {
        // Arrange
        final List<String> networkInputs = Arrays.asList("INPUT1", "INPUT2", "INPUT3");
        ComputationGraphConfiguration configurationMock = mock(ComputationGraphConfiguration.class);
        when(configurationMock.getNetworkInputs()).thenReturn(networkInputs);

        ComputationGraph modelMock = mock(ComputationGraph.class);
        when(modelMock.getConfiguration()).thenReturn(configurationMock);

        ChannelToNetworkInputMapper.NetworkInputToChannelBinding[] networkInputsToChannelNameMap = new ChannelToNetworkInputMapper.NetworkInputToChannelBinding[] {
                ChannelToNetworkInputMapper.NetworkInputToChannelBinding.map("INPUT1", "CN2"),
                ChannelToNetworkInputMapper.NetworkInputToChannelBinding.map("INPUT2", "CN3"),
                ChannelToNetworkInputMapper.NetworkInputToChannelBinding.map("INPUT3", "CN1"),
        };
        String[] channelNames = new String[] { "CN1", "CN2", "CN3" };
        String[] labelNames = new String[] { "LN1", "LN2", "LN3" };
        NetworkHelper sut = new NetworkHelper();

        // Act
        INetworkHandler handler = sut.buildHandler(modelMock, networkInputsToChannelNameMap, channelNames, labelNames, "GRADIENT");

        // Assert
        INDArray channel1 = Nd4j.rand(1, 2);
        INDArray channel2 = Nd4j.rand(1, 2);
        INDArray channel3 = Nd4j.rand(1, 2);
        Observation observation = new Observation(new INDArray[] { channel1, channel2, channel3});
        handler.stepOutput(observation);

        verify(modelMock, times(1)).output(channel2, channel3, channel1);
    }

    @Test
    public void when_callingBuildHandlerWithComputationGraphAndEmptyChannelName_expect_networkHandlerWithFirstInputBoundToFirstChannel() {
        // Arrange
        ComputationGraph modelMock = mock(ComputationGraph.class);

        String[] channelNames = new String[] { "CN1", "CN2", "CN3" };
        String[] labelNames = new String[] { "LN1", "LN2", "LN3" };
        NetworkHelper sut = new NetworkHelper();

        // Act
        INetworkHandler handler = sut.buildHandler(modelMock, "", channelNames, labelNames, "GRADIENT");

        // Assert
        INDArray channel1 = Nd4j.rand(1, 2);
        INDArray channel2 = Nd4j.rand(1, 2);
        INDArray channel3 = Nd4j.rand(1, 2);
        Observation observation = new Observation(new INDArray[] { channel1, channel2, channel3});
        handler.stepOutput(observation);

        verify(modelMock, times(1)).output(channel1);
    }

    @Test
    public void when_callingBuildHandlerWithMLNAndEmptyChannelName_expect_networkHandlerWithFirstInputBoundToFirstChannel() {
        // Arrange
        MultiLayerNetwork modelMock = mock(MultiLayerNetwork.class);

        String[] channelNames = new String[] { "CN1", "CN2", "CN3" };
        NetworkHelper sut = new NetworkHelper();

        // Act
        INetworkHandler handler = sut.buildHandler(modelMock, "", channelNames, "LABEL", "GRADIENT");

        // Assert
        INDArray channel1 = Nd4j.rand(1, 2);
        INDArray channel2 = Nd4j.rand(1, 2);
        INDArray channel3 = Nd4j.rand(1, 2);
        Observation observation = new Observation(new INDArray[] { channel1, channel2, channel3});
        handler.stepOutput(observation);

        verify(modelMock, times(1)).output(channel1);
    }

    @Test
    public void when_callingBuildHandlerWithComputationGraphAndNullChannelNames_expect_networkHandlerWithFirstInputBoundToFirstChannel() {
        // Arrange
        ComputationGraph modelMock = mock(ComputationGraph.class);

        String[] labelNames = new String[] { "LN1", "LN2", "LN3" };
        NetworkHelper sut = new NetworkHelper();

        // Act
        INetworkHandler handler = sut.buildHandler(modelMock, "CN2", null, labelNames, "GRADIENT");

        // Assert
        INDArray channel1 = Nd4j.rand(1, 2);
        INDArray channel2 = Nd4j.rand(1, 2);
        INDArray channel3 = Nd4j.rand(1, 2);
        Observation observation = new Observation(new INDArray[] { channel1, channel2, channel3});
        handler.stepOutput(observation);

        verify(modelMock, times(1)).output(channel1);
    }

    @Test
    public void when_callingBuildHandlerWithMLNAndNullChannelNames_expect_networkHandlerWithFirstInputBoundToFirstChannel() {
        // Arrange
        MultiLayerNetwork modelMock = mock(MultiLayerNetwork.class);

        NetworkHelper sut = new NetworkHelper();

        // Act
        INetworkHandler handler = sut.buildHandler(modelMock, "CN2", null, "LABEL", "GRADIENT");

        // Assert
        INDArray channel1 = Nd4j.rand(1, 2);
        INDArray channel2 = Nd4j.rand(1, 2);
        INDArray channel3 = Nd4j.rand(1, 2);
        Observation observation = new Observation(new INDArray[] { channel1, channel2, channel3});
        handler.stepOutput(observation);

        verify(modelMock, times(1)).output(channel1);
    }

    @Test
    public void when_callingBuildHandlerWithComputationGraphAndEmptyChannelNames_expect_networkHandlerWithFirstInputBoundToFirstChannel() {
        // Arrange
        ComputationGraph modelMock = mock(ComputationGraph.class);

        String[] labelNames = new String[] { "LN1", "LN2", "LN3" };
        NetworkHelper sut = new NetworkHelper();

        // Act
        INetworkHandler handler = sut.buildHandler(modelMock, "CN2", new String[0], labelNames, "GRADIENT");

        // Assert
        INDArray channel1 = Nd4j.rand(1, 2);
        INDArray channel2 = Nd4j.rand(1, 2);
        INDArray channel3 = Nd4j.rand(1, 2);
        Observation observation = new Observation(new INDArray[] { channel1, channel2, channel3});
        handler.stepOutput(observation);

        verify(modelMock, times(1)).output(channel1);
    }

    @Test
    public void when_callingBuildHandlerWithMLNAndEmptyChannelNames_expect_networkHandlerWithFirstInputBoundToFirstChannel() {
        // Arrange
        MultiLayerNetwork modelMock = mock(MultiLayerNetwork.class);

        NetworkHelper sut = new NetworkHelper();

        // Act
        INetworkHandler handler = sut.buildHandler(modelMock, "CN2", new String[0], "LABEL", "GRADIENT");

        // Assert
        INDArray channel1 = Nd4j.rand(1, 2);
        INDArray channel2 = Nd4j.rand(1, 2);
        INDArray channel3 = Nd4j.rand(1, 2);
        Observation observation = new Observation(new INDArray[] { channel1, channel2, channel3});
        handler.stepOutput(observation);

        verify(modelMock, times(1)).output(channel1);
    }

    @Test
    public void when_callingBuildHandlerWithComputationGraphAndSpecificChannelName_expect_networkHandlerWithFirstInputBoundToThatChannel() {
        // Arrange
        ComputationGraph modelMock = mock(ComputationGraph.class);

        String[] channelNames = new String[] { "CN1", "CN2", "CN3" };
        String[] labelNames = new String[] { "LN1", "LN2", "LN3" };
        NetworkHelper sut = new NetworkHelper();

        // Act
        INetworkHandler handler = sut.buildHandler(modelMock, "CN2", channelNames, labelNames, "GRADIENT");

        // Assert
        INDArray channel1 = Nd4j.rand(1, 2);
        INDArray channel2 = Nd4j.rand(1, 2);
        INDArray channel3 = Nd4j.rand(1, 2);
        Observation observation = new Observation(new INDArray[] { channel1, channel2, channel3});
        handler.stepOutput(observation);

        verify(modelMock, times(1)).output(channel2);
    }

    @Test
    public void when_callingBuildHandlerWithMLNAndSpecificChannelName_expect_networkHandlerWithFirstInputBoundToThatChannel() {
        // Arrange
        MultiLayerNetwork modelMock = mock(MultiLayerNetwork.class);

        String[] channelNames = new String[] { "CN1", "CN2", "CN3" };
        NetworkHelper sut = new NetworkHelper();

        // Act
        INetworkHandler handler = sut.buildHandler(modelMock, "CN2", channelNames, "LABEL", "GRADIENT");

        // Assert
        INDArray channel1 = Nd4j.rand(1, 2);
        INDArray channel2 = Nd4j.rand(1, 2);
        INDArray channel3 = Nd4j.rand(1, 2);
        Observation observation = new Observation(new INDArray[] { channel1, channel2, channel3});
        handler.stepOutput(observation);

        verify(modelMock, times(1)).output(channel2);
    }

    @Test
    public void when_callingBuildHandlerWithComputationGraphAndUnknownChannelName_expect_networkHandlerWithFirstInputBoundToThatChannel() {
        try {
            // Arrange
            ComputationGraph modelMock = mock(ComputationGraph.class);

            String[] channelNames = new String[]{"CN1", "CN2", "CN3"};
            String[] labelNames = new String[]{"LN1", "LN2", "LN3"};
            NetworkHelper sut = new NetworkHelper();

            // Act
            INetworkHandler handler = sut.buildHandler(modelMock, "UNKNOWN", channelNames, labelNames, "GRADIENT");

            // Assert
            INDArray channel1 = Nd4j.rand(1, 2);
            INDArray channel2 = Nd4j.rand(1, 2);
            INDArray channel3 = Nd4j.rand(1, 2);
            Observation observation = new Observation(new INDArray[]{channel1, channel2, channel3});
            handler.stepOutput(observation);
            fail("IllegalArgumentException should have been thrown");
        } catch (IllegalArgumentException exception) {
            String expectedMessage = "The channel 'UNKNOWN' was not found in channelNames.";
            String actualMessage = exception.getMessage();

            assertTrue(actualMessage.contains(expectedMessage));
        }
    }


    @Test
    public void when_callingBuildHandlerWithMLNAndUnknownChannelName_expect_networkHandlerWithFirstInputBoundToThatChannel() {
        try {
            // Arrange
            MultiLayerNetwork modelMock = mock(MultiLayerNetwork.class);

            String[] channelNames = new String[] { "CN1", "CN2", "CN3" };
            NetworkHelper sut = new NetworkHelper();

            // Act
            INetworkHandler handler = sut.buildHandler(modelMock, "UNKNOWN", channelNames, "LABEL", "GRADIENT");

            // Assert
            INDArray channel1 = Nd4j.rand(1, 2);
            INDArray channel2 = Nd4j.rand(1, 2);
            INDArray channel3 = Nd4j.rand(1, 2);
            Observation observation = new Observation(new INDArray[] { channel1, channel2, channel3});
            handler.stepOutput(observation);
            fail("IllegalArgumentException should have been thrown");
        } catch (IllegalArgumentException exception) {
            String expectedMessage = "The channel 'UNKNOWN' was not found in channelNames.";
            String actualMessage = exception.getMessage();

            assertTrue(actualMessage.contains(expectedMessage));
        }
    }

}

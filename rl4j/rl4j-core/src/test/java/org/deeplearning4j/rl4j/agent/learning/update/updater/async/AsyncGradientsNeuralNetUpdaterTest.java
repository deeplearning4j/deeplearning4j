package org.deeplearning4j.rl4j.agent.learning.update.updater.async;

import org.deeplearning4j.rl4j.agent.learning.update.Gradients;
import org.deeplearning4j.rl4j.agent.learning.update.updater.NeuralNetUpdaterConfiguration;
import org.deeplearning4j.rl4j.network.ITrainableNeuralNet;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.Mock;
import org.mockito.junit.MockitoJUnitRunner;

import static org.mockito.Mockito.*;

@RunWith(MockitoJUnitRunner.class)
public class AsyncGradientsNeuralNetUpdaterTest {

    @Mock
    ITrainableNeuralNet threadCurrentMock;

    @Mock
    ITrainableNeuralNet globalCurrentMock;

    @Mock
    AsyncSharedNetworksUpdateHandler asyncSharedNetworksUpdateHandlerMock;

    @Test
    public void when_callingUpdate_expect_handlerCalledAndThreadCurrentUpdated() {
        // Arrange
        NeuralNetUpdaterConfiguration configuration = NeuralNetUpdaterConfiguration.builder()
                .targetUpdateFrequency(2)
                .build();
        AsyncGradientsNeuralNetUpdater sut = new AsyncGradientsNeuralNetUpdater(threadCurrentMock, asyncSharedNetworksUpdateHandlerMock);
        Gradients gradients = new Gradients(10);

        // Act
        sut.update(gradients);

        // Assert
        verify(asyncSharedNetworksUpdateHandlerMock, times(1)).handleGradients(gradients);
        verify(threadCurrentMock, never()).copyFrom(globalCurrentMock);
    }

    @Test
    public void when_synchronizeCurrentIsCalled_expect_synchronizeThreadCurrentWithGlobal() {
        // Arrange
        NeuralNetUpdaterConfiguration configuration = NeuralNetUpdaterConfiguration.builder()
                .build();
        AsyncGradientsNeuralNetUpdater sut = new AsyncGradientsNeuralNetUpdater(threadCurrentMock, asyncSharedNetworksUpdateHandlerMock);
        when(asyncSharedNetworksUpdateHandlerMock.getGlobalCurrent()).thenReturn(globalCurrentMock);

        // Act
        sut.synchronizeCurrent();

        // Assert
        verify(threadCurrentMock, times(1)).copyFrom(globalCurrentMock);
    }
}

package org.deeplearning4j.rl4j.agent.learning.update.updater.async;

import org.deeplearning4j.rl4j.agent.learning.update.FeaturesLabels;
import org.deeplearning4j.rl4j.agent.learning.update.Gradients;
import org.deeplearning4j.rl4j.agent.learning.update.updater.NeuralNetUpdaterConfiguration;
import org.deeplearning4j.rl4j.network.ITrainableNeuralNet;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.Mock;
import org.mockito.junit.MockitoJUnitRunner;

import static org.mockito.Mockito.*;

@RunWith(MockitoJUnitRunner.class)
public class AsyncLabelsNeuralNetUpdaterTest {

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
        AsyncLabelsNeuralNetUpdater sut = new AsyncLabelsNeuralNetUpdater(threadCurrentMock, asyncSharedNetworksUpdateHandlerMock);
        FeaturesLabels featureLabels = new FeaturesLabels(null);
        Gradients gradients = new Gradients(10);
        when(threadCurrentMock.computeGradients(featureLabels)).thenReturn(gradients);

        // Act
        sut.update(featureLabels);

        // Assert
        verify(threadCurrentMock, times(1)).computeGradients(featureLabels);
        verify(asyncSharedNetworksUpdateHandlerMock, times(1)).handleGradients(gradients);
        verify(threadCurrentMock, times(0)).copyFrom(any());
    }

    @Test
    public void when_synchronizeCurrentIsCalled_expect_synchronizeThreadCurrentWithGlobal() {
        // Arrange
        NeuralNetUpdaterConfiguration configuration = NeuralNetUpdaterConfiguration.builder()
                .build();
        AsyncLabelsNeuralNetUpdater sut = new AsyncLabelsNeuralNetUpdater(threadCurrentMock, asyncSharedNetworksUpdateHandlerMock);
        when(asyncSharedNetworksUpdateHandlerMock.getGlobalCurrent()).thenReturn(globalCurrentMock);

        // Act
        sut.synchronizeCurrent();

        // Assert
        verify(threadCurrentMock, times(1)).copyFrom(globalCurrentMock);
    }
}

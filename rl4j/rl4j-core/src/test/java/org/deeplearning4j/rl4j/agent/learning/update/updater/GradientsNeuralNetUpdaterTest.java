package org.deeplearning4j.rl4j.agent.learning.update.updater;

import org.deeplearning4j.rl4j.agent.learning.update.Gradients;
import org.deeplearning4j.rl4j.network.ITrainableNeuralNet;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.Mock;
import org.mockito.junit.MockitoJUnitRunner;

import static org.mockito.Mockito.*;

@RunWith(MockitoJUnitRunner.class)
public class GradientsNeuralNetUpdaterTest {

    @Mock
    ITrainableNeuralNet currentMock;

    @Mock
    ITrainableNeuralNet targetMock;

    @Test
    public void when_callingUpdate_expect_currentUpdatedAndtargetNotChanged() {
        // Arrange
        GradientsNeuralNetUpdater.Configuration configuration = GradientsNeuralNetUpdater.Configuration.builder()
                .build();
        GradientsNeuralNetUpdater sut = new GradientsNeuralNetUpdater(currentMock, targetMock, configuration);
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
        GradientsNeuralNetUpdater.Configuration configuration = GradientsNeuralNetUpdater.Configuration.builder()
                .targetUpdateFrequency(3)
                .build();
        GradientsNeuralNetUpdater sut = new GradientsNeuralNetUpdater(currentMock, targetMock, configuration);
        Gradients gradients = new Gradients(10);

        // Act
        sut.update(gradients);
        sut.update(gradients);
        sut.update(gradients);

        // Assert
        verify(currentMock, never()).copy(any());
        verify(targetMock, times(1)).copy(currentMock);
    }

}

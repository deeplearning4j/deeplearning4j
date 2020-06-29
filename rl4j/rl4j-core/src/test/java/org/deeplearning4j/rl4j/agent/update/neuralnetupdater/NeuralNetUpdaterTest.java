package org.deeplearning4j.rl4j.agent.update.neuralnetupdater;

import org.deeplearning4j.rl4j.network.ITrainableNeuralNet;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.Mock;
import org.mockito.junit.MockitoJUnitRunner;
import org.nd4j.linalg.dataset.api.DataSet;

import static org.mockito.Mockito.*;

@RunWith(MockitoJUnitRunner.class)
public class NeuralNetUpdaterTest {

    @Mock
    ITrainableNeuralNet currentMock;

    @Mock
    ITrainableNeuralNet targetMock;

    @Test
    public void when_callingUpdate_expect_currentUpdatedAndtargetNotChanged() {
        // Arrange
        NeuralNetUpdater sut = new NeuralNetUpdater(currentMock, targetMock, Integer.MAX_VALUE);
        DataSet featureLabels = new org.nd4j.linalg.dataset.DataSet();

        // Act
        sut.update(featureLabels);

        // Assert
        verify(currentMock, times(1)).fit(featureLabels);
        verify(targetMock, never()).fit(any());
    }

    @Test
    public void when_callingUpdate_expect_targetUpdatedFromCurrentAtFrequency() {
        // Arrange
        NeuralNetUpdater sut = new NeuralNetUpdater(currentMock, targetMock, 3);
        DataSet featureLabels = new org.nd4j.linalg.dataset.DataSet();

        // Act
        sut.update(featureLabels);
        sut.update(featureLabels);
        sut.update(featureLabels);

        // Assert
        verify(currentMock, never()).copy(any());
        verify(targetMock, times(1)).copy(currentMock);
    }

}

package org.deeplearning4j.rl4j.agent.learning.update.updater;

import org.deeplearning4j.rl4j.agent.learning.update.FeaturesLabels;
import org.deeplearning4j.rl4j.network.ITrainableNeuralNet;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.Mock;
import org.mockito.junit.MockitoJUnitRunner;

import static org.mockito.Mockito.*;

@RunWith(MockitoJUnitRunner.class)
public class LabelsNeuralNetUpdaterTest {

    @Mock
    ITrainableNeuralNet currentMock;

    @Mock
    ITrainableNeuralNet targetMock;

    @Test
    public void when_callingUpdate_expect_currentUpdatedAndtargetNotChanged() {
        // Arrange
        LabelsNeuralNetUpdater sut = new LabelsNeuralNetUpdater(currentMock, targetMock, Integer.MAX_VALUE);
        FeaturesLabels featureLabels = new FeaturesLabels(null);

        // Act
        sut.update(featureLabels);

        // Assert
        verify(currentMock, times(1)).fit(featureLabels);
        verify(targetMock, never()).fit(any());
    }

    @Test
    public void when_callingUpdate_expect_targetUpdatedFromCurrentAtFrequency() {
        // Arrange
        LabelsNeuralNetUpdater sut = new LabelsNeuralNetUpdater(currentMock, targetMock, 3);
        FeaturesLabels featureLabels = new FeaturesLabels(null);

        // Act
        sut.update(featureLabels);
        sut.update(featureLabels);
        sut.update(featureLabels);

        // Assert
        verify(currentMock, never()).copy(any());
        verify(targetMock, times(1)).copy(currentMock);
    }

}

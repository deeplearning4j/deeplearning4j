package org.deeplearning4j.rl4j.agent.learning.update.updater;

import org.deeplearning4j.rl4j.agent.learning.update.FeaturesLabels;
import org.deeplearning4j.rl4j.network.ITrainableNeuralNet;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.Mock;
import org.mockito.junit.MockitoJUnitRunner;

import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;
import static org.mockito.Mockito.*;

@RunWith(MockitoJUnitRunner.class)
public class LabelsNeuralNetUpdaterTest {

    @Mock
    ITrainableNeuralNet currentMock;

    @Mock
    ITrainableNeuralNet targetMock;

    @Test
    public void when_callingUpdateWithTargetUpdateFrequencyAt0_expect_Exception() {
        // Arrange
        LabelsNeuralNetUpdater.Configuration configuration = LabelsNeuralNetUpdater.Configuration.builder()
                .targetUpdateFrequency(0)
                .build();
        try {
            LabelsNeuralNetUpdater sut = new LabelsNeuralNetUpdater(currentMock, targetMock, configuration);
            fail("IllegalArgumentException should have been thrown");
        } catch (IllegalArgumentException exception) {
            String expectedMessage = "Configuration: targetUpdateFrequency must be greater than 0, got:  [0]";
            String actualMessage = exception.getMessage();

            assertTrue(actualMessage.contains(expectedMessage));
        }

    }

    @Test
    public void when_callingUpdate_expect_currentUpdatedAndTargetNotChanged() {
        // Arrange
        LabelsNeuralNetUpdater.Configuration configuration = LabelsNeuralNetUpdater.Configuration.builder()
                .build();
        LabelsNeuralNetUpdater sut = new LabelsNeuralNetUpdater(currentMock, targetMock, configuration);
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
        LabelsNeuralNetUpdater.Configuration configuration = LabelsNeuralNetUpdater.Configuration.builder()
                .targetUpdateFrequency(3)
                .build();
        LabelsNeuralNetUpdater sut = new LabelsNeuralNetUpdater(currentMock, targetMock, configuration);
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

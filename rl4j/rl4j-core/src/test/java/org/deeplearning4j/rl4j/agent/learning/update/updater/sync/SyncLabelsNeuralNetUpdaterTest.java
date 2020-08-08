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

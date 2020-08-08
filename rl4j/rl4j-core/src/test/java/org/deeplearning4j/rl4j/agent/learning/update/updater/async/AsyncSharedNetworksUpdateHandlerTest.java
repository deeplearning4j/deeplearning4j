package org.deeplearning4j.rl4j.agent.learning.update.updater.async;

import org.deeplearning4j.rl4j.agent.learning.update.Gradients;
import org.deeplearning4j.rl4j.agent.learning.update.updater.NeuralNetUpdaterConfiguration;
import org.deeplearning4j.rl4j.network.ITrainableNeuralNet;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.Mock;
import org.mockito.junit.MockitoJUnitRunner;

import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

@RunWith(MockitoJUnitRunner.class)
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

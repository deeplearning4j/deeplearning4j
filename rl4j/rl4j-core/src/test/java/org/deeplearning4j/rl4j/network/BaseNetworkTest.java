package org.deeplearning4j.rl4j.network;

import org.deeplearning4j.rl4j.agent.learning.update.FeaturesLabels;
import org.deeplearning4j.rl4j.agent.learning.update.Gradients;
import org.deeplearning4j.rl4j.observation.Observation;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.Mock;
import org.mockito.Mockito;
import org.mockito.junit.MockitoJUnitRunner;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertEquals;
import static org.mockito.Mockito.*;

@RunWith(MockitoJUnitRunner.class)
public class BaseNetworkTest {

    @Mock
    INetworkHandler handlerMock;

    @Mock
    NeuralNetOutput neuralNetOutputMock;

    private BaseNetwork<BaseNetwork> sut;

    public void setup(boolean setupRecurrent) {
        when(handlerMock.isRecurrent()).thenReturn(setupRecurrent);
        sut = mock(BaseNetwork.class, Mockito.withSettings()
                .useConstructor(handlerMock)
                .defaultAnswer(Mockito.CALLS_REAL_METHODS));
        when(sut.packageResult(any())).thenReturn(neuralNetOutputMock);
    }

    @Test
    public void when_callingIsRecurrent_expect_handlerIsCalled() {
        // Arrange
        setup(false);

        // Act
        sut.isRecurrent();

        // Assert
        verify(handlerMock, times(1)).isRecurrent();
    }

    @Test
    public void when_callingFit_expect_handlerIsCalled() {
        // Arrange
        setup(false);
        FeaturesLabels featuresLabels = new FeaturesLabels(null);

        // Act
        sut.fit(featuresLabels);

        // Assert
        verify(handlerMock, times(1)).performFit(featuresLabels);
    }

    @Test
    public void when_callingComputeGradients_expect_handlerComputeGradientsIsNotifiedAndResponseIsFilled() {
        // Arrange
        setup(false);
        FeaturesLabels featuresLabels = new FeaturesLabels(Nd4j.create(12, 1));
        Gradients gradientsMock = mock(Gradients.class);

        // Act
        Gradients response = sut.computeGradients(featuresLabels);

        // Assert
        verify(handlerMock, times(1)).performGradientsComputation(featuresLabels);
        verify(handlerMock, times(1)).notifyGradientCalculation();
        verify(handlerMock, times(1)).fillGradientsResponse(response);
        assertEquals(response.getBatchSize(), 12);
    }

    @Test
    public void when_callingApplyGradients_expect_handlerAppliesGradientAndIsNotified() {
        // Arrange
        setup(false);
        Gradients gradientsMock = mock(Gradients.class);
        when(gradientsMock.getBatchSize()).thenReturn(12L);

        // Act
        sut.applyGradients(gradientsMock);

        // Assert
        verify(handlerMock, times(1)).applyGradient(gradientsMock, 12L);
        verify(handlerMock, times(1)).notifyIterationDone();
    }

    @Test
    public void when_callingOutputOnNonRecurrentNetworkAndNotInCache_expect_nonRecurrentOutputIsReturned() {
        // Arrange
        setup(false);
        Observation observation = new Observation(Nd4j.rand(1, 2));
        INDArray[] batchOutputResult = new INDArray[] { Nd4j.rand(1, 2) };
        when(handlerMock.batchOutput(observation.getData())).thenReturn(batchOutputResult);

        // Act
        sut.output(observation);

        // Assert
        verify(handlerMock, times(1)).batchOutput(observation.getData());
        verify(sut, times(1)).packageResult(batchOutputResult);
    }

    @Test
    public void when_callingOutputOnRecurrentNetworkAndNotInCache_expect_nonRecurrentOutputIsReturned() {
        // Arrange
        setup(true);
        Observation observation = new Observation(Nd4j.rand(1, 2));
        INDArray[] batchOutputResult = new INDArray[] { Nd4j.rand(1, 2) };
        when(handlerMock.recurrentStepOutput(observation)).thenReturn(batchOutputResult);

        // Act
        sut.output(observation);

        // Assert
        verify(handlerMock, times(1)).recurrentStepOutput(observation);
        verify(sut, times(1)).packageResult(batchOutputResult);
    }

    @Test
    public void when_callingOutput_expect_nonRecurrentOutputIsReturned() {
        // Arrange
        setup(false);
        INDArray batch = Nd4j.rand(1, 2);
        INDArray[] batchOutputResult = new INDArray[] { Nd4j.rand(1, 2) };
        when(handlerMock.batchOutput(batch)).thenReturn(batchOutputResult);

        // Act
        sut.output(batch);

        // Assert
        verify(handlerMock, times(1)).batchOutput(batch);
        verify(sut, times(1)).packageResult(batchOutputResult);
    }

    @Test
    public void when_callingResetOnNonRecurrent_expect_handlerNotCalled() {
        // Arrange
        setup(false);

        // Act
        sut.reset();

        // Assert
        verify(handlerMock, never()).resetState();
    }

    @Test
    public void when_callingResetOnRecurrent_expect_handlerIsCalled() {
        // Arrange
        setup(true);

        // Act
        sut.reset();

        // Assert
        verify(handlerMock, times(1)).resetState();
    }

    @Test
    public void when_callingCopyFrom_expect_handlerIsCalled() {
        // Arrange
        setup(false);

        // Act
        sut.copyFrom(sut);

        // Assert
        verify(handlerMock, times(1)).copyFrom(handlerMock);
    }

    @Test
    public void when_callingFit_expect_CacheInvalidated() {
        // Arrange
        setup(false);
        Observation observation = new Observation(Nd4j.rand(1, 2));
        INDArray[] batchOutputResult = new INDArray[] { Nd4j.rand(1, 2) };
        when(handlerMock.batchOutput(observation.getData())).thenReturn(batchOutputResult);

        // Act
        sut.output(observation);
        sut.fit(null);
        sut.output(observation);

        // Assert
        // Note: calling batchOutput twice means BaseNetwork.fit() has cleared the cache
        verify(handlerMock, times(2)).batchOutput(observation.getData());
    }

    @Test
    public void when_callingApplyGradients_expect_CacheInvalidated() {
        // Arrange
        setup(false);
        Observation observation = new Observation(Nd4j.rand(1, 2));
        INDArray[] batchOutputResult = new INDArray[] { Nd4j.rand(1, 2) };
        when(handlerMock.batchOutput(observation.getData())).thenReturn(batchOutputResult);

        // Act
        sut.output(observation);
        sut.fit(null);
        sut.output(observation);

        // Assert
        // Note: calling batchOutput twice means BaseNetwork.fit() has cleared the cache
        verify(handlerMock, times(2)).batchOutput(observation.getData());
    }

    @Test
    public void when_callingOutputWithoutClearingCache_expect_CacheInvalidated() {
        // Arrange
        setup(false);
        Gradients gradientsMock = mock(Gradients.class);
        when(gradientsMock.getBatchSize()).thenReturn(12L);
        Observation observation = new Observation(Nd4j.rand(1, 2));
        INDArray[] batchOutputResult = new INDArray[] { Nd4j.rand(1, 2) };
        when(handlerMock.batchOutput(observation.getData())).thenReturn(batchOutputResult);


        // Act
        sut.output(observation);
        sut.applyGradients(gradientsMock);
        sut.output(observation);

        // Assert
        // Note: calling batchOutput twice means BaseNetwork.applyGradients() has cleared the cache
        verify(handlerMock, times(2)).batchOutput(observation.getData());
    }

    @Test
    public void when_callingReset_expect_CacheInvalidated() {
        // Arrange
        setup(false);
        Observation observation = new Observation(Nd4j.rand(1, 2));
        INDArray[] batchOutputResult = new INDArray[] { Nd4j.rand(1, 2) };
        when(handlerMock.batchOutput(observation.getData())).thenReturn(batchOutputResult);


        // Act
        sut.output(observation);
        sut.reset();
        sut.output(observation);

        // Assert
        // Note: calling batchOutput twice means BaseNetwork.reset() has cleared the cache
        verify(handlerMock, times(2)).batchOutput(observation.getData());
    }

    @Test
    public void when_callingCopyFrom_expect_CacheInvalidated() {
        // Arrange
        setup(false);
        Observation observation = new Observation(Nd4j.rand(1, 2));
        INDArray[] batchOutputResult = new INDArray[] { Nd4j.rand(1, 2) };
        when(handlerMock.batchOutput(observation.getData())).thenReturn(batchOutputResult);


        // Act
        sut.output(observation);
        sut.copyFrom(sut);
        sut.output(observation);

        // Assert
        // Note: calling batchOutput twice means BaseNetwork.reset() has cleared the cache
        verify(handlerMock, times(2)).batchOutput(observation.getData());
    }

}

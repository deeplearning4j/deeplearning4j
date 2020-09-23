package org.deeplearning4j.rl4j.network;

import org.deeplearning4j.nn.api.Updater;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.recurrent.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.rl4j.agent.learning.update.FeaturesLabels;
import org.deeplearning4j.rl4j.agent.learning.update.Gradients;
import org.deeplearning4j.rl4j.observation.Observation;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.ArgumentCaptor;
import org.mockito.junit.MockitoJUnitRunner;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.Collection;

import static org.junit.Assert.*;
import static org.mockito.Mockito.*;

@RunWith(MockitoJUnitRunner.class)
public class MultiLayerNetworkHandlerTest {

    private static final String LABEL_NAME = "TEST_LABEL";
    private static final String GRADIENT_NAME = "TEST_GRADIENT";

    private MultiLayerNetwork modelMock;
    private TrainingListener trainingListenerMock;
    private MultiLayerConfiguration configurationMock;

    private MultiLayerNetworkHandler sut;

    public void setup(boolean setupRecurrent) {
        modelMock = mock(MultiLayerNetwork.class);
        trainingListenerMock = mock(TrainingListener.class);

        configurationMock = mock(MultiLayerConfiguration.class);
        when(configurationMock.getIterationCount()).thenReturn(123);
        when(configurationMock.getEpochCount()).thenReturn(234);
        when(modelMock.getLayerWiseConfigurations()).thenReturn(configurationMock);

        if(setupRecurrent) {
            when(modelMock.getOutputLayer()).thenReturn(new RnnOutputLayer(null, null));
        }

        sut = new MultiLayerNetworkHandler(modelMock, LABEL_NAME, GRADIENT_NAME);
    }

    @Test
    public void when_callingNotifyGradientCalculation_expect_listenersNotified() {
        // Arrange
        setup(false);
        final Collection<TrainingListener> listeners = new ArrayList<TrainingListener>() {{
            add(trainingListenerMock);
        }};
        when(modelMock.getListeners()).thenReturn(listeners);

        // Act
        sut.notifyGradientCalculation();

        // Assert
        verify(trainingListenerMock, times(1)).onGradientCalculation(modelMock);
    }

    @Test
    public void when_callingNotifyIterationDone_expect_listenersNotified() {
        // Arrange
        setup(false);
        final Collection<TrainingListener> listeners = new ArrayList<TrainingListener>() {{
            add(trainingListenerMock);
        }};
        when(modelMock.getListeners()).thenReturn(listeners);

        // Act
        sut.notifyIterationDone();

        // Assert
        verify(trainingListenerMock, times(1)).iterationDone(modelMock, 123, 234);
    }

    @Test
    public void when_callingPerformFit_expect_fitCalledOnModelWithCorrectLabels() {
        // Arrange
        setup(false);
        INDArray features = Nd4j.rand(1, 2);
        INDArray labels = Nd4j.rand(1, 2);
        FeaturesLabels featuresLabels = new FeaturesLabels(features);
        featuresLabels.putLabels("TEST_LABEL", labels);

        // Act
        sut.performFit(featuresLabels);

        // Assert
        ArgumentCaptor<INDArray> featuresCaptor = ArgumentCaptor.forClass(INDArray.class);
        ArgumentCaptor<INDArray> labelsCaptor = ArgumentCaptor.forClass(INDArray.class);
        verify(modelMock, times(1)).fit(featuresCaptor.capture(), labelsCaptor.capture());
        INDArray featuresArg = featuresCaptor.getValue();
        assertSame(featuresArg, features);
        INDArray labelsArg = labelsCaptor.getValue();
        assertSame(labelsArg, labels);
    }

    @Test
    public void when_callingperformGradientsComputation_expect_modelCalledWithCorrectFeaturesLabels() {
        // Arrange
        setup(false);
        INDArray features = Nd4j.rand(1, 2);
        INDArray labels = Nd4j.rand(1, 2);
        FeaturesLabels featuresLabels = new FeaturesLabels(features);
        featuresLabels.putLabels("TEST_LABEL", labels);

        // Act
        sut.performGradientsComputation(featuresLabels);

        // Assert
        verify(modelMock, times(1)).setInput(features);

        ArgumentCaptor<INDArray> labelsCaptor = ArgumentCaptor.forClass(INDArray.class);
        verify(modelMock, times(1)).setLabels(labelsCaptor.capture());
        Object debug = labelsCaptor.getAllValues();
        INDArray labelsArg = labelsCaptor.getValue();
        assertSame(labels, labelsArg);

        verify(modelMock, times(1)).computeGradientAndScore();
    }

    @Test
    public void when_callingFillGradientsResponse_expect_gradientIsCorrectlyFilled() {
        // Arrange
        setup(false);
        Gradients gradientsMock = mock(Gradients.class);

        final Gradient gradient = mock(Gradient.class);
        when(modelMock.gradient()).thenReturn(gradient);

        // Act
        sut.fillGradientsResponse(gradientsMock);

        // Assert
        verify(gradientsMock, times(1)).putGradient(GRADIENT_NAME, gradient);
    }

    @Test
    public void when_callingApplyGradient_expect_correctGradientAppliedAndIterationUpdated() {
        // Arrange
        setup(false);
        Gradients gradientsMock = mock(Gradients.class);
        final Gradient gradient = mock(Gradient.class);
        INDArray gradientGradient = Nd4j.rand(1, 2);
        when(gradient.gradient()).thenReturn(gradientGradient);
        when(gradientsMock.getGradient(GRADIENT_NAME)).thenReturn(gradient);
        Updater updaterMock = mock(Updater.class);
        when(modelMock.getUpdater()).thenReturn(updaterMock);
        INDArray paramsMock = mock(INDArray.class);
        when(modelMock.params()).thenReturn(paramsMock);

        // Act
        sut.applyGradient(gradientsMock, 345);

        // Assert
        verify(gradientsMock, times(1)).getGradient(GRADIENT_NAME);
        verify(updaterMock, times(1)).update(eq(modelMock), eq(gradient), eq(123), eq(234), eq(345), any());
        verify(paramsMock, times(1)).subi(gradientGradient);
        verify(configurationMock, times(1)).setIterationCount(124);
    }

    @Test
    public void when_callingRecurrentStepOutput_expect_recurrentStepCalledWithObservationData() {
        // Arrange
        setup(false);
        Observation observationMock = mock(Observation.class);
        INDArray observationData = Nd4j.rand(1, 2);
        when(observationMock.getData()).thenReturn(observationData);

        // Act
        sut.recurrentStepOutput(observationMock);

        // Assert
        verify(modelMock, times(1)).rnnTimeStep(observationData);
    }

    @Test
    public void when_callingBatchOutput_expect_outputCalledWithBatch() {
        // Arrange
        setup(false);
        INDArray batch = Nd4j.rand(1, 2);

        // Act
        sut.batchOutput(batch);

        // Assert
        verify(modelMock, times(1)).output(batch);
    }

    @Test
    public void when_callingResetState_expect_modelStateIsCleared() {
        // Arrange
        setup(false);

        // Act
        sut.resetState();

        // Assert
        verify(modelMock, times(1)).rnnClearPreviousState();
    }

    @Test
    public void when_callingClone_expect_handlerAndModelIsCloned() throws Exception {
        // Arrange
        setup(false);
        when(modelMock.clone()).thenReturn(modelMock);

        // Act
        MultiLayerNetworkHandler result = (MultiLayerNetworkHandler)sut.clone();

        // Assert
        assertNotSame(sut, result);

        verify(modelMock, times(1)).clone();

        Field privateField = MultiLayerNetworkHandler.class.getDeclaredField("labelName");
        privateField.setAccessible(true);
        String cloneLabelNames = (String)privateField.get(sut);
        assertEquals(cloneLabelNames, LABEL_NAME);

        privateField = MultiLayerNetworkHandler.class.getDeclaredField("gradientName");
        privateField.setAccessible(true);
        String cloneGradientName = (String)privateField.get(sut);
        assertEquals(cloneGradientName, GRADIENT_NAME);
    }

    @Test
    public void when_callingCopyFrom_expect_modelParamsAreCopiedToModel() {
        // Arrange
        setup(false);
        INDArray params = Nd4j.rand(1, 2);
        when(modelMock.params()).thenReturn(params);
        MultiLayerNetworkHandler from = new MultiLayerNetworkHandler(modelMock, null, null);

        // Act
        sut.copyFrom(from);

        // Assert
        verify(modelMock, times(1)).setParams(params);
    }

    @Test
    public void when_modelIsNotRecurrent_expect_isRecurrentFalse() {
        // Arrange
        setup(false);

        // Act
        boolean isRecurrent = sut.isRecurrent();

        // Assert
        assertFalse(isRecurrent);
    }

    @Test
    public void when_modelIsRecurrent_expect_isRecurrentTrue() {
        // Arrange
        setup(true);

        // Act
        boolean isRecurrent = sut.isRecurrent();

        // Assert
        assertTrue(isRecurrent);
    }
}
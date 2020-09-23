package org.deeplearning4j.rl4j.network;

import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.rl4j.agent.learning.update.FeaturesLabels;
import org.deeplearning4j.rl4j.agent.learning.update.Gradients;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.junit.MockitoJUnitRunner;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertNotSame;
import static org.junit.Assert.assertSame;
import static org.mockito.Mockito.*;
import static org.mockito.Mockito.times;

@RunWith(MockitoJUnitRunner.class)
public class ActorCriticNetworkTest {

    @Test
    public void when_callingCtorWithCG_expect_handlerUsesCorrectLabelAndGradientNames() {
        // Arrange
        ComputationGraph modelMock = mock(ComputationGraph.class);
        FeaturesLabels featuresLabelsMock = mock(FeaturesLabels.class);
        Gradient gradientMock = mock(Gradient.class);
        when(modelMock.gradient()).thenReturn(gradientMock);

        // Act
        ActorCriticNetwork sut = new ActorCriticNetwork(modelMock);
        Gradients results = sut.computeGradients(featuresLabelsMock);

        // Assert
        verify(featuresLabelsMock, times(1)).getLabels(CommonLabelNames.ActorCritic.Value);
        verify(featuresLabelsMock, times(1)).getLabels(CommonLabelNames.ActorCritic.Policy);
        assertSame(gradientMock, results.getGradient(CommonGradientNames.ActorCritic.Combined));
    }

    @Test
    public void when_callingCtorWithSeparateMLN_expect_handlerUsesCorrectLabelAndGradientNames() {
        // Arrange
        MultiLayerNetwork valueMock = mock(MultiLayerNetwork.class);
        Gradient valueGradientMock = mock(Gradient.class);
        when(valueMock.gradient()).thenReturn(valueGradientMock);

        MultiLayerNetwork policyMock = mock(MultiLayerNetwork.class);
        Gradient policyGradientMock = mock(Gradient.class);
        when(policyMock.gradient()).thenReturn(policyGradientMock);

        FeaturesLabels featuresLabelsMock = mock(FeaturesLabels.class);


        // Act
        ActorCriticNetwork sut = new ActorCriticNetwork(valueMock, policyMock);
        Gradients results = sut.computeGradients(featuresLabelsMock);

        // Assert
        verify(featuresLabelsMock, times(1)).getLabels(CommonLabelNames.ActorCritic.Value);
        verify(featuresLabelsMock, times(1)).getLabels(CommonLabelNames.ActorCritic.Policy);
        assertSame(valueGradientMock, results.getGradient(CommonGradientNames.ActorCritic.Value));
        assertSame(policyGradientMock, results.getGradient(CommonGradientNames.ActorCritic.Policy));
    }

    @Test
    public void when_callingCtorWithSeparateMLNAndCG_expect_handlerUsesCorrectLabelAndGradientNames() {
        // Arrange
        MultiLayerNetwork valueMock = mock(MultiLayerNetwork.class);
        Gradient valueGradientMock = mock(Gradient.class);
        when(valueMock.gradient()).thenReturn(valueGradientMock);

        ComputationGraph policyMock = mock(ComputationGraph.class);
        Gradient policyGradientMock = mock(Gradient.class);
        when(policyMock.gradient()).thenReturn(policyGradientMock);

        FeaturesLabels featuresLabelsMock = mock(FeaturesLabels.class);


        // Act
        ActorCriticNetwork sut = new ActorCriticNetwork(valueMock, policyMock);
        Gradients results = sut.computeGradients(featuresLabelsMock);

        // Assert
        verify(featuresLabelsMock, times(1)).getLabels(CommonLabelNames.ActorCritic.Value);
        verify(featuresLabelsMock, times(1)).getLabels(CommonLabelNames.ActorCritic.Policy);
        assertSame(valueGradientMock, results.getGradient(CommonGradientNames.ActorCritic.Value));
        assertSame(policyGradientMock, results.getGradient(CommonGradientNames.ActorCritic.Policy));
    }

    @Test
    public void when_callingCtorWithSeparateCGAndMLN_expect_handlerUsesCorrectLabelAndGradientNames() {
        // Arrange
        ComputationGraph valueMock = mock(ComputationGraph.class);
        Gradient valueGradientMock = mock(Gradient.class);
        when(valueMock.gradient()).thenReturn(valueGradientMock);

        MultiLayerNetwork policyMock = mock(MultiLayerNetwork.class);
        Gradient policyGradientMock = mock(Gradient.class);
        when(policyMock.gradient()).thenReturn(policyGradientMock);

        FeaturesLabels featuresLabelsMock = mock(FeaturesLabels.class);


        // Act
        ActorCriticNetwork sut = new ActorCriticNetwork(valueMock, policyMock);
        Gradients results = sut.computeGradients(featuresLabelsMock);

        // Assert
        verify(featuresLabelsMock, times(1)).getLabels(CommonLabelNames.ActorCritic.Value);
        verify(featuresLabelsMock, times(1)).getLabels(CommonLabelNames.ActorCritic.Policy);
        assertSame(valueGradientMock, results.getGradient(CommonGradientNames.ActorCritic.Value));
        assertSame(policyGradientMock, results.getGradient(CommonGradientNames.ActorCritic.Policy));
    }

    @Test
    public void when_callingCtorWithSeparateCG_expect_handlerUsesCorrectLabelAndGradientNames() {
        // Arrange
        ComputationGraph valueMock = mock(ComputationGraph.class);
        Gradient valueGradientMock = mock(Gradient.class);
        when(valueMock.gradient()).thenReturn(valueGradientMock);

        ComputationGraph policyMock = mock(ComputationGraph.class);
        Gradient policyGradientMock = mock(Gradient.class);
        when(policyMock.gradient()).thenReturn(policyGradientMock);

        FeaturesLabels featuresLabelsMock = mock(FeaturesLabels.class);


        // Act
        ActorCriticNetwork sut = new ActorCriticNetwork(valueMock, policyMock);
        Gradients results = sut.computeGradients(featuresLabelsMock);

        // Assert
        verify(featuresLabelsMock, times(1)).getLabels(CommonLabelNames.ActorCritic.Value);
        verify(featuresLabelsMock, times(1)).getLabels(CommonLabelNames.ActorCritic.Policy);
        assertSame(valueGradientMock, results.getGradient(CommonGradientNames.ActorCritic.Value));
        assertSame(policyGradientMock, results.getGradient(CommonGradientNames.ActorCritic.Policy));
    }

    @Test
    public void when_callingOutput_expect_resultHasCorrectNames() {
        // Arrange
        ComputationGraph modelMock = mock(ComputationGraph.class);
        INDArray batch = Nd4j.rand(1, 2);
        INDArray outputValue = Nd4j.rand(1, 2);
        INDArray outputPolicy = Nd4j.rand(1, 2);
        when(modelMock.output(batch)).thenReturn(new INDArray[] { outputValue, outputPolicy });

        // Act
        ActorCriticNetwork sut = new ActorCriticNetwork(modelMock);
        NeuralNetOutput result = sut.output(batch);

        // Assert
        assertSame(outputValue, result.get(CommonOutputNames.ActorCritic.Value));
        assertSame(outputPolicy, result.get(CommonOutputNames.ActorCritic.Policy));
    }

    @Test
    public void when_callingClone_expect_clonedActorCriticNetwork() {
        // Arrange
        ComputationGraph modelMock = mock(ComputationGraph.class);
        when(modelMock.clone()).thenReturn(modelMock);

        // Act
        ActorCriticNetwork sut = new ActorCriticNetwork(modelMock);
        ActorCriticNetwork clone = sut.clone();

        // Assert
        assertNotSame(sut, clone);
        assertNotSame(sut.getNetworkHandler(), clone.getNetworkHandler());
    }

}

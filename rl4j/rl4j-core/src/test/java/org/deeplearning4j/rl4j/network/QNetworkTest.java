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

@RunWith(MockitoJUnitRunner.class)
public class QNetworkTest {

    @Test
    public void when_callingCtorWithMLN_expect_handlerUsesCorrectLabelAndGradientNames() {
        // Arrange
        MultiLayerNetwork modelMock = mock(MultiLayerNetwork.class);
        FeaturesLabels featuresLabelsMock = mock(FeaturesLabels.class);
        Gradient gradientMock = mock(Gradient.class);
        when(modelMock.gradient()).thenReturn(gradientMock);

        // Act
        QNetwork sut = new QNetwork(modelMock);
        Gradients results = sut.computeGradients(featuresLabelsMock);

        // Assert
        verify(featuresLabelsMock, times(1)).getLabels(CommonLabelNames.QValues);
        assertSame(gradientMock, results.getGradient(CommonGradientNames.QValues));
    }

    @Test
    public void when_callingCtorWithCG_expect_handlerUsesCorrectLabelAndGradientNames() {
        // Arrange
        ComputationGraph modelMock = mock(ComputationGraph.class);
        FeaturesLabels featuresLabelsMock = mock(FeaturesLabels.class);
        Gradient gradientMock = mock(Gradient.class);
        when(modelMock.gradient()).thenReturn(gradientMock);

        // Act
        QNetwork sut = new QNetwork(modelMock);
        Gradients results = sut.computeGradients(featuresLabelsMock);

        // Assert
        verify(featuresLabelsMock, times(1)).getLabels(CommonLabelNames.QValues);
        assertSame(gradientMock, results.getGradient(CommonGradientNames.QValues));
    }

    @Test
    public void when_callingOutput_expect_resultHasCorrectNames() {
        // Arrange
        ComputationGraph modelMock = mock(ComputationGraph.class);
        INDArray batch = Nd4j.rand(1, 2);
        INDArray output = Nd4j.rand(1, 2);
        when(modelMock.output(batch)).thenReturn(new INDArray[] { output });

        // Act
        QNetwork sut = new QNetwork(modelMock);
        NeuralNetOutput result = sut.output(batch);

        // Assert
        assertSame(output, result.get(CommonOutputNames.QValues));
    }

    @Test
    public void when_callingClone_expect_clonedQNetwork() {
        // Arrange
        ComputationGraph modelMock = mock(ComputationGraph.class);
        when(modelMock.clone()).thenReturn(modelMock);

        // Act
        QNetwork sut = new QNetwork(modelMock);
        QNetwork clone = sut.clone();

        // Assert
        assertNotSame(sut, clone);
        assertNotSame(sut.getNetworkHandler(), clone.getNetworkHandler());
    }

}

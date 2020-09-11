package org.deeplearning4j.rl4j.agent.learning.algorithm.nstepqlearning;

import org.deeplearning4j.rl4j.agent.learning.update.FeaturesLabels;
import org.deeplearning4j.rl4j.agent.learning.update.Gradients;
import org.deeplearning4j.rl4j.experience.StateActionPair;
import org.deeplearning4j.rl4j.network.*;
import org.deeplearning4j.rl4j.observation.Observation;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.junit.MockitoJUnitRunner;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.mockito.Mockito.*;

@RunWith(MockitoJUnitRunner.class)
public class RecurrentNStepQLearningTest {

    private static final int ACTION_SPACE_SIZE = 2;

    @Mock
    ITrainableNeuralNet threadCurrentMock;

    @Mock
    IOutputNeuralNet targetMock;

    NStepQLearning sut;

    private void setup(double gamma) {
        when(threadCurrentMock.output(any(INDArray.class))).thenAnswer(invocation -> {
            NeuralNetOutput result = new NeuralNetOutput();
            result.put(CommonOutputNames.QValues, invocation.getArgument(0, INDArray.class).mul(-1.0));
            return result;
        });
        when(targetMock.output(any(INDArray.class))).thenAnswer(invocation -> {
            NeuralNetOutput result = new NeuralNetOutput();
            result.put(CommonOutputNames.QValues, invocation.getArgument(0, INDArray.class).mul(-2.0));
            return result;
        });
        when(threadCurrentMock.isRecurrent()).thenReturn(true);

        NStepQLearning.Configuration configuration = NStepQLearning.Configuration.builder()
            .gamma(gamma)
            .build();
        sut = new NStepQLearning(threadCurrentMock, targetMock, ACTION_SPACE_SIZE, configuration);
    }

    @Test
    public void when_isTerminal_expect_initRewardIs0() {
        // Arrange
        int action = 0;
        setup(1.0);

        final Observation observation = new Observation(Nd4j.zeros(1, 2, 1));
        List<StateActionPair<Integer>> experience = new ArrayList<StateActionPair<Integer>>() {
            {
                add(new StateActionPair<Integer>(observation, action, 0.0, true));
            }
        };

        // Act
        Gradients result = sut.compute(experience);

        // Assert
        ArgumentCaptor<FeaturesLabels> argument = ArgumentCaptor.forClass(FeaturesLabels.class);
        verify(threadCurrentMock, times(1)).computeGradients(argument.capture());

        FeaturesLabels featuresLabels = argument.getValue();
        assertEquals(0.0, featuresLabels.getLabels(CommonLabelNames.QValues).getDouble(0, 0), 0.000001);
    }

    @Test
    public void when_notTerminal_expect_initRewardWithMaxQFromTarget() {
        // Arrange
        int action = 0;
        setup(1.0);

        final Observation observation = new Observation(Nd4j.create(new double[] { -123.0, -234.0 }).reshape(1, 2, 1));
        List<StateActionPair<Integer>> experience = new ArrayList<StateActionPair<Integer>>() {
            {
                add(new StateActionPair<Integer>(observation, action, 0.0, false));
            }
        };

        // Act
        Gradients result = sut.compute(experience);

        // Assert
        ArgumentCaptor<FeaturesLabels> argument = ArgumentCaptor.forClass(FeaturesLabels.class);
        verify(threadCurrentMock, times(1)).computeGradients(argument.capture());

        FeaturesLabels featuresLabels = argument.getValue();
        assertEquals(-2.0 * observation.getData().getDouble(0, 1, 0), featuresLabels.getLabels(CommonLabelNames.QValues).getDouble(0), 0.000001);
    }

    @Test
    public void when_callingWithMultipleExperiences_expect_gradientsAreValid() {
        // Arrange
        double gamma = 0.9;
        setup(gamma);

        List<StateActionPair<Integer>> experience = new ArrayList<StateActionPair<Integer>>() {
            {
                add(new StateActionPair<Integer>(new Observation(Nd4j.create(new double[] { -1.1, -1.2 }).reshape(1, 2, 1)), 0, 1.0, false));
                add(new StateActionPair<Integer>(new Observation(Nd4j.create(new double[] { -2.1, -2.2 }).reshape(1, 2, 1)), 1, 2.0, true));
            }
        };

        // Act
        sut.compute(experience);

        // Assert
        ArgumentCaptor<FeaturesLabels> argument = ArgumentCaptor.forClass(FeaturesLabels.class);
        verify(threadCurrentMock, times(1)).computeGradients(argument.capture());

        // input side -- should be a stack of observations
        INDArray featuresValues = argument.getValue().getFeatures();
        assertEquals(-1.1, featuresValues.getDouble(0, 0, 0), 0.00001);
        assertEquals(-1.2, featuresValues.getDouble(0, 1, 0), 0.00001);
        assertEquals(-2.1, featuresValues.getDouble(0, 0, 1), 0.00001);
        assertEquals(-2.2, featuresValues.getDouble(0, 1, 1), 0.00001);

        // target side
        INDArray labels = argument.getValue().getLabels(CommonLabelNames.QValues);
        assertEquals(1.0 + gamma * 2.0, labels.getDouble(0, 0, 0), 0.00001);
        assertEquals(1.2, labels.getDouble(0, 1, 0), 0.00001);
        assertEquals(2.1, labels.getDouble(0, 0, 1), 0.00001);
        assertEquals(2.0, labels.getDouble(0, 1, 1), 0.00001);
    }
}

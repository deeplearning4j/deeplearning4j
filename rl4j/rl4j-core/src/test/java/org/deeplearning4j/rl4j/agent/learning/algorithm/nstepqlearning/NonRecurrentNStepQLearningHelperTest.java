package org.deeplearning4j.rl4j.agent.learning.algorithm.nstepqlearning;

import org.deeplearning4j.rl4j.experience.StateActionPair;
import org.deeplearning4j.rl4j.network.CommonOutputNames;
import org.deeplearning4j.rl4j.network.IOutputNeuralNet;
import org.deeplearning4j.rl4j.network.NeuralNet;
import org.deeplearning4j.rl4j.network.NeuralNetOutput;
import org.deeplearning4j.rl4j.observation.Observation;
import org.junit.Test;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.*;
import static org.mockito.Mockito.*;

public class NonRecurrentNStepQLearningHelperTest {

    private final NonRecurrentNStepQLearningHelper sut = new NonRecurrentNStepQLearningHelper(3);

    @Test
    public void when_callingCreateFeatures_expect_INDArrayWithCorrectShape() {
        // Arrange
        List<StateActionPair<Integer>> experience = new ArrayList<StateActionPair<Integer>>() {
            {
                add(new StateActionPair<Integer>(new Observation(Nd4j.create(new double[] { 1.1, 1.2 }).reshape(1, 2)), 0, 1.0, false));
                add(new StateActionPair<Integer>(new Observation(Nd4j.create(new double[] { 2.1, 2.2 }).reshape(1, 2)), 1, 2.0, false));
                add(new StateActionPair<Integer>(new Observation(Nd4j.create(new double[] { 3.1, 3.2 }).reshape(1, 2)), 2, 3.0, false));
                add(new StateActionPair<Integer>(new Observation(Nd4j.create(new double[] { 4.1, 4.2 }).reshape(1, 2)), 3, 4.0, false));
            }
        };

        // Act
        INDArray result = sut.createFeatures(experience);

        // Assert
        assertArrayEquals(new long[] { 4, 2 }, result.shape());
        assertEquals(1.1, result.getDouble(0, 0), 0.00001);
        assertEquals(1.2, result.getDouble(0, 1), 0.00001);
        assertEquals(2.1, result.getDouble(1, 0), 0.00001);
        assertEquals(2.2, result.getDouble(1, 1), 0.00001);
        assertEquals(3.1, result.getDouble(2, 0), 0.00001);
        assertEquals(3.2, result.getDouble(2, 1), 0.00001);
        assertEquals(4.1, result.getDouble(3, 0), 0.00001);
        assertEquals(4.2, result.getDouble(3, 1), 0.00001);
    }

    @Test
    public void when_callingCreateValueLabels_expect_INDArrayWithCorrectShape() {
        // Arrange

        // Act
        INDArray result = sut.createLabels(4);

        // Assert
        assertArrayEquals(new long[] { 4, 3 }, result.shape());
    }

    @Test
    public void when_callingGetExpectedQValues_expect_INDArrayWithCorrectShape() {
        // Arrange
        INDArray allExpectedQValues = Nd4j.create(new double[] { 1.1, 1.2, 2.1, 2.2 }).reshape(2,2);

        // Act
        INDArray result = sut.getExpectedQValues(allExpectedQValues, 1);

        // Assert
        assertEquals(2.1, result.getDouble(0), 0.00001);
        assertEquals(2.2, result.getDouble(1), 0.00001);
    }

    @Test
    public void when_callingSetLabels_expect_INDArrayWithCorrectShape() {
        // Arrange
        INDArray labels = Nd4j.zeros(2, 2);
        INDArray data = Nd4j.create(new double[] { 1.1, 1.2 });

        // Act
        sut.setLabels(labels, 1, data);

        // Assert
        assertEquals(0.0, labels.getDouble(0, 0), 0.00001);
        assertEquals(0.0, labels.getDouble(0, 1), 0.00001);
        assertEquals(1.1, labels.getDouble(1, 0), 0.00001);
        assertEquals(1.2, labels.getDouble(1, 1), 0.00001);
    }

    @Test
    public void when_callingGetTargetExpectedQValuesOfLast_expect_INDArrayWithCorrectShape() {
        // Arrange
        IOutputNeuralNet targetMock = mock(IOutputNeuralNet.class);
        List<StateActionPair<Integer>> experience = new ArrayList<StateActionPair<Integer>>() {
            {
                add(new StateActionPair<Integer>(new Observation(Nd4j.create(new double[] { 1.1, 1.2 }).reshape(1, 2)), 0, 1.0, false));
                add(new StateActionPair<Integer>(new Observation(Nd4j.create(new double[] { 2.1, 2.2 }).reshape(1, 2)), 1, 2.0, false));
                add(new StateActionPair<Integer>(new Observation(Nd4j.create(new double[] { 3.1, 3.2 }).reshape(1, 2)), 2, 3.0, false));
                add(new StateActionPair<Integer>(new Observation(Nd4j.create(new double[] { 4.1, 4.2 }).reshape(1, 2)), 3, 4.0, false));
            }
        };
        final NeuralNetOutput neuralNetOutput = new NeuralNetOutput();
        neuralNetOutput.put(CommonOutputNames.QValues, Nd4j.create(new double[] { -4.1, -4.2 }).reshape(1, 2));
        when(targetMock.output(any(Observation.class))).thenReturn(neuralNetOutput);

        // Act
        INDArray result = sut.getTargetExpectedQValuesOfLast(targetMock, experience, null);

        // Assert
        ArgumentCaptor<Observation> observationCaptor = ArgumentCaptor.forClass(Observation.class);
        verify(targetMock, times(1)).output(observationCaptor.capture());
        Observation observation = observationCaptor.getValue();
        assertEquals(4.1, observation.getData().getDouble(0), 0.00001);
        assertEquals(4.2, observation.getData().getDouble(1), 0.00001);

        assertEquals(-4.1, result.getDouble(0), 0.00001);
        assertEquals(-4.2, result.getDouble(1), 0.00001);
    }
}

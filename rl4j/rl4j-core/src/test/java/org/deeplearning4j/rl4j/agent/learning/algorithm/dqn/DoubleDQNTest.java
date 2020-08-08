package org.deeplearning4j.rl4j.agent.learning.algorithm.dqn;

import org.deeplearning4j.rl4j.agent.learning.update.FeaturesLabels;
import org.deeplearning4j.rl4j.learning.sync.Transition;
import org.deeplearning4j.rl4j.network.CommonLabelNames;
import org.deeplearning4j.rl4j.network.CommonOutputNames;
import org.deeplearning4j.rl4j.network.IOutputNeuralNet;
import org.deeplearning4j.rl4j.network.NeuralNetOutput;
import org.deeplearning4j.rl4j.observation.Observation;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.Mock;
import org.mockito.junit.MockitoJUnitRunner;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.when;

@RunWith(MockitoJUnitRunner.class)
public class DoubleDQNTest {

    @Mock
    IOutputNeuralNet qNetworkMock;

    @Mock
    IOutputNeuralNet targetQNetworkMock;

    private final BaseTransitionTDAlgorithm.Configuration configuration = BaseTransitionTDAlgorithm.Configuration.builder()
            .gamma(0.5)
            .build();

    @Before
    public void setup() {
        when(qNetworkMock.output(any(INDArray.class))).thenAnswer(i -> {
            NeuralNetOutput result = new NeuralNetOutput();
            result.put(CommonOutputNames.QValues, i.getArgument(0, INDArray.class));
            return result;
        });
    }

    @Test
    public void when_isTerminal_expect_rewardValueAtIdx0() {

        // Assemble
        when(targetQNetworkMock.output(any(INDArray.class))).thenAnswer(i -> {
            NeuralNetOutput result = new NeuralNetOutput();
            result.put(CommonOutputNames.QValues, i.getArgument(0, INDArray.class));
            return result;
        });

        List<Transition<Integer>> transitions = new ArrayList<Transition<Integer>>() {
            {
                add(builtTransition(buildObservation(new double[]{1.1, 2.2}),
                        0, 1.0, true, buildObservation(new double[]{11.0, 22.0})));
            }
        };

        org.deeplearning4j.rl4j.agent.learning.algorithm.dqn.DoubleDQN sut = new org.deeplearning4j.rl4j.agent.learning.algorithm.dqn.DoubleDQN(qNetworkMock, targetQNetworkMock, configuration);

        // Act
        FeaturesLabels result = sut.compute(transitions);

        // Assert
        INDArray evaluatedQValues = result.getLabels(CommonLabelNames.QValues);
        assertEquals(1.0, evaluatedQValues.getDouble(0, 0), 0.0001);
        assertEquals(2.2, evaluatedQValues.getDouble(0, 1), 0.0001);
    }

    @Test
    public void when_isNotTerminal_expect_rewardPlusEstimatedQValue() {

        // Assemble
        when(targetQNetworkMock.output(any(INDArray.class))).thenAnswer(i -> {
            NeuralNetOutput result = new NeuralNetOutput();
            result.put(CommonOutputNames.QValues, i.getArgument(0, INDArray.class).mul(-1.0));
            return result;
        });

        List<Transition<Integer>> transitions = new ArrayList<Transition<Integer>>() {
            {
                add(builtTransition(buildObservation(new double[]{1.1, 2.2}),
                        0, 1.0, false, buildObservation(new double[]{11.0, 22.0})));
            }
        };

        org.deeplearning4j.rl4j.agent.learning.algorithm.dqn.DoubleDQN sut = new org.deeplearning4j.rl4j.agent.learning.algorithm.dqn.DoubleDQN(qNetworkMock, targetQNetworkMock, configuration);

        // Act
        FeaturesLabels result = sut.compute(transitions);

        // Assert
        INDArray evaluatedQValues = result.getLabels(CommonLabelNames.QValues);
        assertEquals(1.0 + 0.5 * -22.0, evaluatedQValues.getDouble(0, 0), 0.0001);
        assertEquals(2.2, evaluatedQValues.getDouble(0, 1), 0.0001);
    }

    @Test
    public void when_batchHasMoreThanOne_expect_everySampleEvaluated() {

        // Assemble
        when(targetQNetworkMock.output(any(INDArray.class))).thenAnswer(i -> {
            NeuralNetOutput result = new NeuralNetOutput();
            result.put(CommonOutputNames.QValues, i.getArgument(0, INDArray.class).mul(-1.0));
            return result;
        });

        List<Transition<Integer>> transitions = new ArrayList<Transition<Integer>>() {
            {
                add(builtTransition(buildObservation(new double[]{1.1, 2.2}),
                        0, 1.0, false, buildObservation(new double[]{11.0, 22.0})));
                add(builtTransition(buildObservation(new double[]{3.3, 4.4}),
                        1, 2.0, false, buildObservation(new double[]{33.0, 44.0})));
                add(builtTransition(buildObservation(new double[]{5.5, 6.6}),
                        0, 3.0, true, buildObservation(new double[]{55.0, 66.0})));
            }
        };

        org.deeplearning4j.rl4j.agent.learning.algorithm.dqn.DoubleDQN sut = new DoubleDQN(qNetworkMock, targetQNetworkMock, configuration);

        // Act
        FeaturesLabels result = sut.compute(transitions);

        // Assert
        INDArray evaluatedQValues = result.getLabels(CommonLabelNames.QValues);
        assertEquals(1.0 + 0.5 * -22.0, evaluatedQValues.getDouble(0, 0), 0.0001);
        assertEquals(2.2, evaluatedQValues.getDouble(0, 1), 0.0001);

        assertEquals(3.3, evaluatedQValues.getDouble(1, 0), 0.0001);
        assertEquals(2.0 + 0.5 * -44.0, evaluatedQValues.getDouble(1, 1), 0.0001);

        assertEquals(3.0, evaluatedQValues.getDouble(2, 0), 0.0001); // terminal: reward only
        assertEquals(6.6, evaluatedQValues.getDouble(2, 1), 0.0001);

    }

    private Observation buildObservation(double[] data) {
        return new Observation(Nd4j.create(data).reshape(1, 2));
    }

    private Transition<Integer> builtTransition(Observation observation, Integer action, double reward, boolean isTerminal, Observation nextObservation) {
        Transition<Integer> result = new Transition<Integer>(observation, action, reward, isTerminal);
        result.setNextObservation(nextObservation);

        return result;
    }
}

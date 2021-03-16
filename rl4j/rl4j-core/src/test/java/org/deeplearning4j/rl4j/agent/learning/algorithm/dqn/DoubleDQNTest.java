/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.rl4j.agent.learning.algorithm.dqn;

import org.deeplearning4j.rl4j.agent.learning.update.Features;
import org.deeplearning4j.rl4j.agent.learning.update.FeaturesLabels;
import org.deeplearning4j.rl4j.experience.StateActionRewardState;
import org.deeplearning4j.rl4j.network.CommonLabelNames;
import org.deeplearning4j.rl4j.network.CommonOutputNames;
import org.deeplearning4j.rl4j.network.IOutputNeuralNet;
import org.deeplearning4j.rl4j.network.NeuralNetOutput;
import org.deeplearning4j.rl4j.observation.Observation;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.runner.RunWith;
import org.mockito.Mock;
import org.mockito.junit.MockitoJUnitRunner;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
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

    @BeforeEach
    public void setup() {
        when(qNetworkMock.output(any(Features.class))).thenAnswer(i -> {
            NeuralNetOutput result = new NeuralNetOutput();
            result.put(CommonOutputNames.QValues, i.getArgument(0, Features.class).get(0));
            return result;
        });
    }

    @Test
    public void when_isTerminal_expect_rewardValueAtIdx0() {

        // Assemble
        when(targetQNetworkMock.output(any(Features.class))).thenAnswer(i -> {
            NeuralNetOutput result = new NeuralNetOutput();
            result.put(CommonOutputNames.QValues, i.getArgument(0, Features.class).get(0));
            return result;
        });

        List<StateActionRewardState<Integer>> stateActionRewardStates = new ArrayList<StateActionRewardState<Integer>>() {
            {
                add(builtTransition(buildObservation(new double[]{1.1, 2.2}),
                        0, 1.0, true, buildObservation(new double[]{11.0, 22.0})));
            }
        };

        org.deeplearning4j.rl4j.agent.learning.algorithm.dqn.DoubleDQN sut = new org.deeplearning4j.rl4j.agent.learning.algorithm.dqn.DoubleDQN(qNetworkMock, targetQNetworkMock, configuration);

        // Act
        FeaturesLabels result = sut.compute(stateActionRewardStates);

        // Assert
        INDArray evaluatedQValues = result.getLabels(CommonLabelNames.QValues);
        assertEquals(1.0, evaluatedQValues.getDouble(0, 0), 0.0001);
        assertEquals(2.2, evaluatedQValues.getDouble(0, 1), 0.0001);
    }

    @Test
    public void when_isNotTerminal_expect_rewardPlusEstimatedQValue() {

        // Assemble
        when(targetQNetworkMock.output(any(Features.class))).thenAnswer(i -> {
            NeuralNetOutput result = new NeuralNetOutput();
            result.put(CommonOutputNames.QValues, i.getArgument(0, Features.class).get(0).mul(-1.0));
            return result;
        });

        List<StateActionRewardState<Integer>> stateActionRewardStates = new ArrayList<StateActionRewardState<Integer>>() {
            {
                add(builtTransition(buildObservation(new double[]{1.1, 2.2}),
                        0, 1.0, false, buildObservation(new double[]{11.0, 22.0})));
            }
        };

        DoubleDQN sut = new DoubleDQN(qNetworkMock, targetQNetworkMock, configuration);

        // Act
        FeaturesLabels result = sut.compute(stateActionRewardStates);

        // Assert
        INDArray evaluatedQValues = result.getLabels(CommonLabelNames.QValues);
        assertEquals(1.0 + 0.5 * -22.0, evaluatedQValues.getDouble(0, 0), 0.0001);
        assertEquals(2.2, evaluatedQValues.getDouble(0, 1), 0.0001);
    }

    @Test
    public void when_batchHasMoreThanOne_expect_everySampleEvaluated() {

        // Assemble
        when(targetQNetworkMock.output(any(Features.class))).thenAnswer(i -> {
            NeuralNetOutput result = new NeuralNetOutput();
            result.put(CommonOutputNames.QValues, i.getArgument(0, Features.class).get(0).mul(-1.0));
            return result;
        });

        List<StateActionRewardState<Integer>> stateActionRewardStates = new ArrayList<StateActionRewardState<Integer>>() {
            {
                add(builtTransition(buildObservation(new double[]{1.1, 2.2}),
                        0, 1.0, false, buildObservation(new double[]{11.0, 22.0})));
                add(builtTransition(buildObservation(new double[]{3.3, 4.4}),
                        1, 2.0, false, buildObservation(new double[]{33.0, 44.0})));
                add(builtTransition(buildObservation(new double[]{5.5, 6.6}),
                        0, 3.0, true, buildObservation(new double[]{55.0, 66.0})));
            }
        };

        DoubleDQN sut = new DoubleDQN(qNetworkMock, targetQNetworkMock, configuration);

        // Act
        FeaturesLabels result = sut.compute(stateActionRewardStates);

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

    private StateActionRewardState<Integer> builtTransition(Observation observation, Integer action, double reward, boolean isTerminal, Observation nextObservation) {
        StateActionRewardState<Integer> result = new StateActionRewardState<Integer>(observation, action, reward, isTerminal);
        result.setNextObservation(nextObservation);

        return result;
    }
}

/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.rl4j.learning.sync.qlearning.discrete;

import org.deeplearning4j.gym.StepReply;
import org.deeplearning4j.rl4j.agent.learning.behavior.ILearningBehavior;
import org.deeplearning4j.rl4j.learning.IHistoryProcessor;
import org.deeplearning4j.rl4j.learning.configuration.QLearningConfiguration;
import org.deeplearning4j.rl4j.learning.sync.qlearning.QLearning;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.network.CommonOutputNames;
import org.deeplearning4j.rl4j.network.NeuralNetOutput;
import org.deeplearning4j.rl4j.network.dqn.IDQN;
import org.deeplearning4j.rl4j.space.Encodable;
import org.deeplearning4j.rl4j.observation.Observation;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.space.ObservationSpace;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.Mock;
import org.mockito.Mockito;
import org.mockito.junit.MockitoJUnitRunner;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

@RunWith(MockitoJUnitRunner.class)
public class QLearningDiscreteTest {

    QLearningDiscrete<Encodable> qLearningDiscrete;

    @Mock
    IHistoryProcessor mockHistoryProcessor;

    @Mock
    IHistoryProcessor.Configuration mockHistoryConfiguration;

    @Mock
    MDP<Encodable, Integer, DiscreteSpace> mockMDP;

    @Mock
    DiscreteSpace mockActionSpace;

    @Mock
    ObservationSpace<Encodable> mockObservationSpace;

    @Mock
    IDQN mockDQN;

    @Mock
    QLearningConfiguration mockQlearningConfiguration;

    @Mock
    ILearningBehavior<Integer> learningBehavior;

    // HWC
    int[] observationShape = new int[]{3, 10, 10};
    int totalObservationSize = 1;

    private void setupMDPMocks() {

        when(mockObservationSpace.getShape()).thenReturn(observationShape);

        when(mockMDP.getObservationSpace()).thenReturn(mockObservationSpace);
        when(mockMDP.getActionSpace()).thenReturn(mockActionSpace);

        int dataLength = 1;
        for (int d : observationShape) {
            dataLength *= d;
        }
    }


    private void mockTestContext(int maxSteps, int updateStart, int batchSize, double rewardFactor, int maxExperienceReplay, ILearningBehavior<Integer> learningBehavior) {
        when(mockQlearningConfiguration.getBatchSize()).thenReturn(batchSize);
        when(mockQlearningConfiguration.getRewardFactor()).thenReturn(rewardFactor);
        when(mockQlearningConfiguration.getExpRepMaxSize()).thenReturn(maxExperienceReplay);
        when(mockQlearningConfiguration.getSeed()).thenReturn(123L);
        when(mockQlearningConfiguration.getTargetDqnUpdateFreq()).thenReturn(1);
        when(mockDQN.clone()).thenReturn(mockDQN);

        if(learningBehavior != null) {
            qLearningDiscrete = mock(
                    QLearningDiscrete.class,
                    Mockito.withSettings()
                            .useConstructor(mockMDP, mockDQN, mockQlearningConfiguration, 0, learningBehavior, Nd4j.getRandom())
                            .defaultAnswer(Mockito.CALLS_REAL_METHODS)
            );
        }
        else {
            qLearningDiscrete = mock(
                    QLearningDiscrete.class,
                    Mockito.withSettings()
                            .useConstructor(mockMDP, mockDQN, mockQlearningConfiguration, 0)
                            .defaultAnswer(Mockito.CALLS_REAL_METHODS)
            );
        }
    }

    private void mockHistoryProcessor(int skipFrames) {
        when(mockHistoryConfiguration.getRescaledHeight()).thenReturn(observationShape[1]);
        when(mockHistoryConfiguration.getRescaledWidth()).thenReturn(observationShape[2]);

        when(mockHistoryConfiguration.getOffsetX()).thenReturn(0);
        when(mockHistoryConfiguration.getOffsetY()).thenReturn(0);

        when(mockHistoryConfiguration.getCroppingHeight()).thenReturn(observationShape[1]);
        when(mockHistoryConfiguration.getCroppingWidth()).thenReturn(observationShape[2]);
        when(mockHistoryConfiguration.getSkipFrame()).thenReturn(skipFrames);
        when(mockHistoryConfiguration.getHistoryLength()).thenReturn(1);
        when(mockHistoryProcessor.getConf()).thenReturn(mockHistoryConfiguration);

        qLearningDiscrete.setHistoryProcessor(mockHistoryProcessor);
    }

    @Before
    public void setup() {
        setupMDPMocks();

        for (int i : observationShape) {
            totalObservationSize *= i;
        }

    }

    @Test
    public void when_singleTrainStep_expect_correctValues() {

        // Arrange
        mockTestContext(100,0,2,1.0, 10, null);

        // An example observation and 2 Q values output (2 actions)
        Observation observation = new Observation(Nd4j.zeros(observationShape));
        NeuralNetOutput netOutputResult = new NeuralNetOutput();
        netOutputResult.put(CommonOutputNames.QValues, Nd4j.create(new float[] {1.0f, 0.5f}));
        when(mockDQN.output(eq(observation))).thenReturn(netOutputResult);

        when(mockMDP.step(anyInt())).thenReturn(new StepReply<>(new Observation(Nd4j.zeros(observationShape)), 0, false, null));

        // Act
        QLearning.QLStepReturn<Observation> stepReturn = qLearningDiscrete.trainStep(observation);

        // Assert
        assertEquals(1.0, stepReturn.getMaxQ(), 1e-5);

        StepReply<Observation> stepReply = stepReturn.getStepReply();

        assertEquals(0, stepReply.getReward(), 1e-5);
        assertFalse(stepReply.isDone());
        assertFalse(stepReply.getObservation().isSkipped());
        assertEquals(observation.getData().reshape(observationShape), stepReply.getObservation().getData().reshape(observationShape));

    }

    @Test
    public void when_singleTrainStepSkippedFrames_expect_correctValues() {
        // Arrange
        mockTestContext(100,0,2,1.0, 10, learningBehavior);

        Observation skippedObservation = Observation.SkippedObservation;
        Observation nextObservation = new Observation(Nd4j.zeros(observationShape));

        when(mockMDP.step(anyInt())).thenReturn(new StepReply<>(nextObservation, 0, false, null));

        // Act
        QLearning.QLStepReturn<Observation> stepReturn = qLearningDiscrete.trainStep(skippedObservation);

        // Assert
        assertEquals(Double.NaN, stepReturn.getMaxQ(), 1e-5);

        StepReply<Observation> stepReply = stepReturn.getStepReply();

        assertEquals(0, stepReply.getReward(), 1e-5);
        assertFalse(stepReply.isDone());
        assertFalse(stepReply.getObservation().isSkipped());

        verify(learningBehavior, never()).handleNewExperience(any(Observation.class), any(Integer.class), any(Double.class), any(Boolean.class));
        verify(mockDQN, never()).output(any(INDArray.class));

    }

    //TODO: there are much more test cases here that can be improved upon

}

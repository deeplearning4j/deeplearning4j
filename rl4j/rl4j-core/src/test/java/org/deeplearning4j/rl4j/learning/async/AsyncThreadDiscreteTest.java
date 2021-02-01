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

package org.deeplearning4j.rl4j.learning.async;

import org.deeplearning4j.gym.StepReply;
import org.deeplearning4j.rl4j.learning.configuration.IAsyncLearningConfiguration;
import org.deeplearning4j.rl4j.learning.listener.TrainingListenerList;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.network.NeuralNet;
import org.deeplearning4j.rl4j.observation.Observation;
import org.deeplearning4j.rl4j.policy.Policy;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.space.Encodable;
import org.deeplearning4j.rl4j.space.ObservationSpace;
import org.deeplearning4j.rl4j.util.LegacyMDPWrapper;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.Mock;
import org.mockito.Mockito;
import org.mockito.junit.MockitoJUnitRunner;
import org.nd4j.linalg.factory.Nd4j;

import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

@RunWith(MockitoJUnitRunner.class)
public class AsyncThreadDiscreteTest {


    AsyncThreadDiscrete<Encodable, NeuralNet> asyncThreadDiscrete;

    @Mock
    IAsyncLearningConfiguration mockAsyncConfiguration;

    @Mock
    UpdateAlgorithm<NeuralNet> mockUpdateAlgorithm;

    @Mock
    IAsyncGlobal<NeuralNet> mockAsyncGlobal;

    @Mock
    Policy<Integer> mockGlobalCurrentPolicy;

    @Mock
    NeuralNet mockGlobalTargetNetwork;

    @Mock
    MDP<Encodable, Integer, DiscreteSpace> mockMDP;

    @Mock
    LegacyMDPWrapper<Encodable, Integer, DiscreteSpace> mockLegacyMDPWrapper;

    @Mock
    DiscreteSpace mockActionSpace;

    @Mock
    ObservationSpace<Encodable> mockObservationSpace;

    @Mock
    TrainingListenerList mockTrainingListenerList;

    @Mock
    Observation mockObservation;

    int[] observationShape = new int[]{3, 10, 10};
    int actionSize = 4;

    private void setupMDPMocks() {

        when(mockActionSpace.noOp()).thenReturn(0);
        when(mockMDP.getActionSpace()).thenReturn(mockActionSpace);

        when(mockObservationSpace.getShape()).thenReturn(observationShape);
        when(mockMDP.getObservationSpace()).thenReturn(mockObservationSpace);

    }

    private void setupNNMocks() {
        when(mockAsyncGlobal.getTarget()).thenReturn(mockGlobalTargetNetwork);
        when(mockGlobalTargetNetwork.clone()).thenReturn(mockGlobalTargetNetwork);
    }

    @Before
    public void setup() {

        setupMDPMocks();
        setupNNMocks();

        asyncThreadDiscrete = mock(AsyncThreadDiscrete.class, Mockito.withSettings()
                .useConstructor(mockAsyncGlobal, mockMDP, mockTrainingListenerList, 0, 0)
                .defaultAnswer(Mockito.CALLS_REAL_METHODS));

        asyncThreadDiscrete.setUpdateAlgorithm(mockUpdateAlgorithm);

        when(asyncThreadDiscrete.getConfiguration()).thenReturn(mockAsyncConfiguration);
        when(mockAsyncConfiguration.getRewardFactor()).thenReturn(1.0);
        when(asyncThreadDiscrete.getAsyncGlobal()).thenReturn(mockAsyncGlobal);
        when(asyncThreadDiscrete.getPolicy(eq(mockGlobalTargetNetwork))).thenReturn(mockGlobalCurrentPolicy);

        when(mockGlobalCurrentPolicy.nextAction(any(Observation.class))).thenReturn(0);

        when(asyncThreadDiscrete.getLegacyMDPWrapper()).thenReturn(mockLegacyMDPWrapper);

    }

    @Test
    public void when_episodeCompletes_expect_stepsToBeInLineWithEpisodeLenth() {

        // Arrange
        int episodeRemaining = 5;
        int remainingTrainingSteps = 10;

        // return done after 4 steps (the episode finishes before nsteps)
        when(mockMDP.isDone()).thenAnswer(invocation ->
                asyncThreadDiscrete.getStepCount() == episodeRemaining
        );

        when(mockLegacyMDPWrapper.step(0)).thenReturn(new StepReply<>(mockObservation, 0.0, false, null));

        // Act
        AsyncThread.SubEpochReturn subEpochReturn = asyncThreadDiscrete.trainSubEpoch(mockObservation, remainingTrainingSteps);

        // Assert
        assertTrue(subEpochReturn.isEpisodeComplete());
        assertEquals(5, subEpochReturn.getSteps());
    }

    @Test
    public void when_episodeCompletesDueToMaxStepsReached_expect_isEpisodeComplete() {

        // Arrange
        int remainingTrainingSteps = 50;

        // Episode does not complete due to MDP
        when(mockMDP.isDone()).thenReturn(false);

        when(mockLegacyMDPWrapper.step(0)).thenReturn(new StepReply<>(mockObservation, 0.0, false, null));

        when(mockAsyncConfiguration.getMaxEpochStep()).thenReturn(50);

        // Act
        AsyncThread.SubEpochReturn subEpochReturn = asyncThreadDiscrete.trainSubEpoch(mockObservation, remainingTrainingSteps);

        // Assert
        assertTrue(subEpochReturn.isEpisodeComplete());
        assertEquals(50, subEpochReturn.getSteps());

    }

    @Test
    public void when_episodeLongerThanNsteps_expect_returnNStepLength() {

        // Arrange
        int episodeRemaining = 5;
        int remainingTrainingSteps = 4;

        // return done after 4 steps (the episode finishes before nsteps)
        when(mockMDP.isDone()).thenAnswer(invocation ->
                asyncThreadDiscrete.getStepCount() == episodeRemaining
        );

        when(mockLegacyMDPWrapper.step(0)).thenReturn(new StepReply<>(mockObservation, 0.0, false, null));

        // Act
        AsyncThread.SubEpochReturn subEpochReturn = asyncThreadDiscrete.trainSubEpoch(mockObservation, remainingTrainingSteps);

        // Assert
        assertFalse(subEpochReturn.isEpisodeComplete());
        assertEquals(remainingTrainingSteps, subEpochReturn.getSteps());
    }

    @Test
    public void when_framesAreSkipped_expect_proportionateStepCounterUpdates() {
        int skipFrames = 2;
        int remainingTrainingSteps = 10;

        // Episode does not complete due to MDP
        when(mockMDP.isDone()).thenReturn(false);

        AtomicInteger stepCount = new AtomicInteger();

        // Use skipFrames to return if observations are skipped or not
        when(mockLegacyMDPWrapper.step(anyInt())).thenAnswer(invocationOnMock -> {

            boolean isSkipped = stepCount.incrementAndGet() % skipFrames != 0;

            Observation mockObs = isSkipped ? Observation.SkippedObservation : new Observation(Nd4j.create(observationShape));
            return new StepReply<>(mockObs, 0.0, false, null);
        });


        // Act
        AsyncThread.SubEpochReturn subEpochReturn = asyncThreadDiscrete.trainSubEpoch(mockObservation, remainingTrainingSteps);

        // Assert
        assertFalse(subEpochReturn.isEpisodeComplete());
        assertEquals(remainingTrainingSteps, subEpochReturn.getSteps());
        assertEquals((remainingTrainingSteps - 1) * skipFrames + 1, stepCount.get());
    }

    @Test
    public void when_preEpisodeCalled_expect_experienceHandlerReset() {

        // Arrange
        int trainingSteps = 100;
        for (int i = 0; i < trainingSteps; i++) {
            asyncThreadDiscrete.getExperienceHandler().addExperience(mockObservation, 0, 0.0, false);
        }

        int experienceHandlerSizeBeforeReset = asyncThreadDiscrete.getExperienceHandler().getTrainingBatchSize();

        // Act
        asyncThreadDiscrete.preEpisode();

        // Assert
        assertEquals(100, experienceHandlerSizeBeforeReset);
        assertEquals(0, asyncThreadDiscrete.getExperienceHandler().getTrainingBatchSize());


    }

}

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

import org.deeplearning4j.rl4j.learning.configuration.IAsyncLearningConfiguration;
import org.deeplearning4j.rl4j.learning.listener.TrainingListenerList;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.network.NeuralNet;
import org.deeplearning4j.rl4j.observation.Observation;
import org.deeplearning4j.rl4j.space.ActionSpace;
import org.deeplearning4j.rl4j.space.Box;
import org.deeplearning4j.rl4j.space.ObservationSpace;
import org.deeplearning4j.rl4j.util.IDataManager;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.Mock;
import org.mockito.Mockito;
import org.mockito.junit.MockitoJUnitRunner;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.shade.guava.base.Preconditions;

import static org.junit.Assert.assertEquals;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

@RunWith(MockitoJUnitRunner.class)
public class AsyncThreadTest {

    @Mock
    ActionSpace<INDArray> mockActionSpace;

    @Mock
    ObservationSpace<Box> mockObservationSpace;

    @Mock
    IAsyncLearningConfiguration mockAsyncConfiguration;

    @Mock
    NeuralNet mockNeuralNet;

    @Mock
    IAsyncGlobal<NeuralNet> mockAsyncGlobal;

    @Mock
    MDP<Box, INDArray, ActionSpace<INDArray>> mockMDP;

    @Mock
    TrainingListenerList mockTrainingListeners;

    int[] observationShape = new int[]{3, 10, 10};
    int actionSize = 4;

    AsyncThread<Box, INDArray, ActionSpace<INDArray>, NeuralNet> thread;

    @Before
    public void setup() {
        setupMDPMocks();
        setupThreadMocks();
    }

    private void setupThreadMocks() {

        thread = mock(AsyncThread.class, Mockito.withSettings()
                .useConstructor(mockMDP, mockTrainingListeners, 0, 0)
                .defaultAnswer(Mockito.CALLS_REAL_METHODS));

        when(thread.getAsyncGlobal()).thenReturn(mockAsyncGlobal);
        when(thread.getCurrent()).thenReturn(mockNeuralNet);
    }

    private void setupMDPMocks() {

        when(mockObservationSpace.getShape()).thenReturn(observationShape);
        when(mockActionSpace.noOp()).thenReturn(Nd4j.zeros(actionSize));

        when(mockMDP.getObservationSpace()).thenReturn(mockObservationSpace);
        when(mockMDP.getActionSpace()).thenReturn(mockActionSpace);

        int dataLength = 1;
        for (int d : observationShape) {
            dataLength *= d;
        }

        when(mockMDP.reset()).thenReturn(new Box(new double[dataLength]));
    }

    private void mockTrainingListeners() {
        mockTrainingListeners(false, false);
    }

    private void mockTrainingListeners(boolean stopOnNotifyNewEpoch, boolean stopOnNotifyEpochTrainingResult) {
        when(mockTrainingListeners.notifyNewEpoch(eq(thread))).thenReturn(!stopOnNotifyNewEpoch);
        when(mockTrainingListeners.notifyEpochTrainingResult(eq(thread), any(IDataManager.StatEntry.class))).thenReturn(!stopOnNotifyEpochTrainingResult);
    }

    private void mockTrainingContext() {
        mockTrainingContext(1000, 100, 10);
    }

    private void mockTrainingContext(int maxSteps, int maxStepsPerEpisode, int nstep) {

        // Some conditions of this test harness
        Preconditions.checkArgument(maxStepsPerEpisode >= nstep, "episodeLength must be greater than or equal to nstep");
        Preconditions.checkArgument(maxStepsPerEpisode % nstep == 0, "episodeLength must be a multiple of nstep");

        Observation mockObs = new Observation(Nd4j.zeros(observationShape));

        when(mockAsyncConfiguration.getMaxEpochStep()).thenReturn(maxStepsPerEpisode);
        when(mockAsyncConfiguration.getNStep()).thenReturn(nstep);
        when(thread.getConfiguration()).thenReturn(mockAsyncConfiguration);

        // if we hit the max step count
        when(mockAsyncGlobal.isTrainingComplete()).thenAnswer(invocation -> thread.getStepCount() >= maxSteps);

        when(thread.trainSubEpoch(any(Observation.class), anyInt())).thenAnswer(invocationOnMock -> {
            int steps = invocationOnMock.getArgument(1);
            thread.stepCount += steps;
            thread.currentEpisodeStepCount += steps;
            boolean isEpisodeComplete = thread.getCurrentEpisodeStepCount() % maxStepsPerEpisode == 0;
            return new AsyncThread.SubEpochReturn(steps, mockObs, 0.0, 0.0, isEpisodeComplete);
        });
    }

    @Test
    public void when_episodeComplete_expect_neuralNetworkReset() {

        // Arrange
        mockTrainingContext(100, 10, 10);
        mockTrainingListeners();

        // Act
        thread.run();

        // Assert
        verify(mockNeuralNet, times(10)).reset(); // there are 10 episodes so the network should be reset between each
        assertEquals(10, thread.getEpochCount()); // We are performing a training iteration every 10 steps, so there should be 10 epochs
        assertEquals(10, thread.getEpisodeCount()); // There should be 10 completed episodes
        assertEquals(100, thread.getStepCount()); // 100 steps overall
    }

    @Test
    public void when_notifyNewEpochReturnsStop_expect_threadStopped() {
        // Arrange
        mockTrainingContext();
        mockTrainingListeners(true, false);

        // Act
        thread.run();

        // Assert
        assertEquals(0, thread.getEpochCount());
        assertEquals(1, thread.getEpisodeCount());
        assertEquals(0, thread.getStepCount());
    }

    @Test
    public void when_notifyEpochTrainingResultReturnsStop_expect_threadStopped() {
        // Arrange
        mockTrainingContext();
        mockTrainingListeners(false, true);

        // Act
        thread.run();

        // Assert
        assertEquals(1, thread.getEpochCount());
        assertEquals(1, thread.getEpisodeCount());
        assertEquals(10, thread.getStepCount()); // one epoch is by default 10 steps
    }

    @Test
    public void when_run_expect_preAndPostEpisodeCalled() {
        // Arrange
        mockTrainingContext(100, 10, 5);
        mockTrainingListeners(false, false);

        // Act
        thread.run();

        // Assert
        assertEquals(20, thread.getEpochCount());
        assertEquals(10, thread.getEpisodeCount());
        assertEquals(100, thread.getStepCount());

        verify(thread, times(10)).preEpisode(); // over 100 steps there will be 10 episodes
        verify(thread, times(10)).postEpisode();
    }

    @Test
    public void when_run_expect_trainSubEpochCalledAndResultPassedToListeners() {
        // Arrange
        mockTrainingContext(100, 10, 5);
        mockTrainingListeners(false, false);

        // Act
        thread.run();

        // Assert
        assertEquals(20, thread.getEpochCount());
        assertEquals(10, thread.getEpisodeCount());
        assertEquals(100, thread.getStepCount());

        // Over 100 steps there will be 20 training iterations, so there will be 20 calls to notifyEpochTrainingResult
        verify(mockTrainingListeners, times(20)).notifyEpochTrainingResult(eq(thread), any(IDataManager.StatEntry.class));
    }

    @Test
    public void when_run_expect_trainSubEpochCalled() {
        // Arrange
        mockTrainingContext(100, 10, 5);
        mockTrainingListeners(false, false);

        // Act
        thread.run();

        // Assert
        assertEquals(20, thread.getEpochCount());
        assertEquals(10, thread.getEpisodeCount());
        assertEquals(100, thread.getStepCount());

        // There should be 20 calls to trainsubepoch with 5 steps per epoch
        verify(thread, times(20)).trainSubEpoch(any(Observation.class), eq(5));
    }

    @Test
    public void when_remainingEpisodeLengthSmallerThanNSteps_expect_trainSubEpochCalledWithMinimumValue() {

        int currentEpisodeSteps = 95;
        mockTrainingContext(1000, 100, 10);
        mockTrainingListeners(false, true);

        // want to mock that we are 95 steps into the episode
        doAnswer(invocationOnMock -> {
            for (int i = 0; i < currentEpisodeSteps; i++) {
                thread.incrementSteps();
            }
            return null;
        }).when(thread).preEpisode();

        mockTrainingListeners(false, true);

        // Act
        thread.run();

        // Assert
        assertEquals(1, thread.getEpochCount());
        assertEquals(1, thread.getEpisodeCount());
        assertEquals(100, thread.getStepCount());

        // There should be 1 call to trainsubepoch with 5 steps as this is the remaining episode steps
        verify(thread, times(1)).trainSubEpoch(any(Observation.class), eq(5));
    }

}

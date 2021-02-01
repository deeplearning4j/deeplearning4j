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

package org.deeplearning4j.rl4j.agent;

import org.deeplearning4j.rl4j.agent.learning.behavior.LearningBehavior;
import org.deeplearning4j.rl4j.environment.Environment;
import org.deeplearning4j.rl4j.environment.IntegerActionSchema;
import org.deeplearning4j.rl4j.environment.Schema;
import org.deeplearning4j.rl4j.environment.StepResult;
import org.deeplearning4j.rl4j.observation.Observation;
import org.deeplearning4j.rl4j.observation.transform.TransformProcess;
import org.deeplearning4j.rl4j.policy.IPolicy;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.invocation.InvocationOnMock;
import org.mockito.junit.MockitoJUnitRunner;
import org.mockito.stubbing.Answer;
import org.nd4j.linalg.factory.Nd4j;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;
import static org.junit.Assert.*;

@RunWith(MockitoJUnitRunner.class)
public class AgentLearnerTest {

    @Mock
    Environment<Integer> environmentMock;

    @Mock
    TransformProcess transformProcessMock;

    @Mock
    IPolicy<Integer> policyMock;

    @Mock
    LearningBehavior<Integer, Object> learningBehaviorMock;

    @Test
    public void when_episodeIsStarted_expect_learningBehaviorHandleEpisodeStartCalled() {
        // Arrange
        AgentLearner.Configuration configuration = AgentLearner.Configuration.builder()
                .maxEpisodeSteps(3)
                .build();
        AgentLearner<Integer> sut = new AgentLearner(environmentMock, transformProcessMock, policyMock, configuration, null, learningBehaviorMock);

        Schema schema = new Schema(new IntegerActionSchema(0, -1));
        when(environmentMock.reset()).thenReturn(new HashMap<>());
        when(environmentMock.getSchema()).thenReturn(schema);
        StepResult stepResult = new StepResult(new HashMap<>(), 234.0, false);
        when(environmentMock.step(any(Integer.class))).thenReturn(stepResult);

        when(transformProcessMock.transform(any(Map.class), anyInt(), anyBoolean())).thenReturn(new Observation(Nd4j.create(new double[] { 123.0 })));

        when(policyMock.nextAction(any(Observation.class))).thenReturn(123);

        // Act
        sut.run();

        // Assert
        verify(learningBehaviorMock, times(1)).handleEpisodeStart();
    }

    @Test
    public void when_runIsCalled_expect_experienceHandledWithLearningBehavior() {
        // Arrange
        AgentLearner.Configuration configuration = AgentLearner.Configuration.builder()
                .maxEpisodeSteps(4)
                .build();
        AgentLearner<Integer> sut = new AgentLearner(environmentMock, transformProcessMock, policyMock, configuration, null, learningBehaviorMock);

        Schema schema = new Schema(new IntegerActionSchema(0, -1));
        when(environmentMock.getSchema()).thenReturn(schema);
        when(environmentMock.reset()).thenReturn(new HashMap<>());

        double[] reward = new double[] { 0.0 };
        when(environmentMock.step(any(Integer.class)))
                .thenAnswer(a -> new StepResult(new HashMap<>(), ++reward[0], reward[0] == 4.0));

        when(environmentMock.isEpisodeFinished()).thenAnswer(x -> reward[0] == 4.0);

        when(transformProcessMock.transform(any(Map.class), anyInt(), anyBoolean()))
                .thenAnswer(new Answer<Observation>() {
                    public Observation answer(InvocationOnMock invocation) throws Throwable {
                        int step = (int)invocation.getArgument(1);
                        boolean isTerminal = (boolean)invocation.getArgument(2);
                        return (step % 2 == 0 || isTerminal)
                                ? new Observation(Nd4j.create(new double[] { step * 1.1 }))
                                : Observation.SkippedObservation;
                    }
                });

        when(policyMock.nextAction(any(Observation.class))).thenAnswer(x -> (int)reward[0]);

        // Act
        sut.run();

        // Assert
        ArgumentCaptor<Observation> observationCaptor = ArgumentCaptor.forClass(Observation.class);
        ArgumentCaptor<Integer> actionCaptor = ArgumentCaptor.forClass(Integer.class);
        ArgumentCaptor<Double> rewardCaptor = ArgumentCaptor.forClass(Double.class);
        ArgumentCaptor<Boolean> isTerminalCaptor = ArgumentCaptor.forClass(Boolean.class);

        verify(learningBehaviorMock, times(2)).handleNewExperience(observationCaptor.capture(), actionCaptor.capture(), rewardCaptor.capture(), isTerminalCaptor.capture());
        List<Observation> observations = observationCaptor.getAllValues();
        List<Integer> actions = actionCaptor.getAllValues();
        List<Double> rewards = rewardCaptor.getAllValues();
        List<Boolean> isTerminalList = isTerminalCaptor.getAllValues();

        assertEquals(0.0, observations.get(0).getData().getDouble(0), 0.00001);
        assertEquals(0, (int)actions.get(0));
        assertEquals(0.0 + 1.0, rewards.get(0), 0.00001);
        assertFalse(isTerminalList.get(0));

        assertEquals(2.2, observations.get(1).getData().getDouble(0), 0.00001);
        assertEquals(2, (int)actions.get(1));
        assertEquals(2.0 + 3.0, rewards.get(1), 0.00001);
        assertFalse(isTerminalList.get(1));

        ArgumentCaptor<Observation> finalObservationCaptor = ArgumentCaptor.forClass(Observation.class);
        verify(learningBehaviorMock, times(1)).handleEpisodeEnd(finalObservationCaptor.capture());
        assertEquals(4.4, finalObservationCaptor.getValue().getData().getDouble(0), 0.00001);
    }

    @Test
    public void when_runIsCalledMultipleTimes_expect_rewardSentToLearningBehaviorToBeCorrect() {
        // Arrange
        AgentLearner.Configuration configuration = AgentLearner.Configuration.builder()
                .maxEpisodeSteps(4)
                .build();
        AgentLearner<Integer> sut = new AgentLearner(environmentMock, transformProcessMock, policyMock, configuration, null, learningBehaviorMock);

        Schema schema = new Schema(new IntegerActionSchema(0, -1));
        when(environmentMock.getSchema()).thenReturn(schema);
        when(environmentMock.reset()).thenReturn(new HashMap<>());

        double[] reward = new double[] { 0.0 };
        when(environmentMock.step(any(Integer.class)))
                .thenAnswer(a -> new StepResult(new HashMap<>(), ++reward[0], reward[0] == 4.0));

        when(environmentMock.isEpisodeFinished()).thenAnswer(x -> reward[0] == 4.0);

        when(transformProcessMock.transform(any(Map.class), anyInt(), anyBoolean()))
                .thenAnswer(new Answer<Observation>() {
                    public Observation answer(InvocationOnMock invocation) throws Throwable {
                        int step = (int)invocation.getArgument(1);
                        boolean isTerminal = (boolean)invocation.getArgument(2);
                        return (step % 2 == 0 || isTerminal)
                                ? new Observation(Nd4j.create(new double[] { step * 1.1 }))
                                : Observation.SkippedObservation;
                    }
                });

        when(policyMock.nextAction(any(Observation.class))).thenAnswer(x -> (int)reward[0]);

        // Act
        sut.run();
        reward[0] = 0.0;
        sut.run();

        // Assert
        ArgumentCaptor<Double> rewardCaptor = ArgumentCaptor.forClass(Double.class);

        verify(learningBehaviorMock, times(4)).handleNewExperience(any(Observation.class), any(Integer.class), rewardCaptor.capture(), any(Boolean.class));
        List<Double> rewards = rewardCaptor.getAllValues();

        // rewardAtLastExperience at the end of 1st call to .run() should not leak into 2nd call.
        assertEquals(0.0 + 1.0, rewards.get(2), 0.00001);
        assertEquals(2.0 + 3.0, rewards.get(3), 0.00001);
    }

    @Test
    public void when_aStepWillBeTaken_expect_learningBehaviorNotified() {
        // Arrange
        AgentLearner.Configuration configuration = AgentLearner.Configuration.builder()
                .maxEpisodeSteps(1)
                .build();
        AgentLearner<Integer> sut = new AgentLearner(environmentMock, transformProcessMock, policyMock, configuration, null, learningBehaviorMock);

        Schema schema = new Schema(new IntegerActionSchema(0, -1));
        when(environmentMock.reset()).thenReturn(new HashMap<>());
        when(environmentMock.getSchema()).thenReturn(schema);
        StepResult stepResult = new StepResult(new HashMap<>(), 234.0, false);
        when(environmentMock.step(any(Integer.class))).thenReturn(stepResult);

        when(transformProcessMock.transform(any(Map.class), anyInt(), anyBoolean())).thenReturn(new Observation(Nd4j.create(new double[] { 123.0 })));

        when(policyMock.nextAction(any(Observation.class))).thenReturn(123);

        // Act
        sut.run();

        // Assert
        verify(learningBehaviorMock, times(1)).notifyBeforeStep();
    }

}
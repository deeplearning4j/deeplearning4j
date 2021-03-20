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

package org.deeplearning4j.rl4j.agent;

import org.deeplearning4j.rl4j.agent.listener.AgentListener;
import org.deeplearning4j.rl4j.environment.*;
import org.deeplearning4j.rl4j.observation.Observation;
import org.deeplearning4j.rl4j.observation.transform.TransformProcess;
import org.deeplearning4j.rl4j.policy.IPolicy;

import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.extension.ExtendWith;
//import org.junit.platform.runner.JUnitPlatform;
import org.junit.runner.RunWith;
import org.mockito.*;
import org.mockito.exceptions.base.MockitoException;
import org.mockito.junit.*;
import org.mockito.junit.jupiter.MockitoExtension;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.factory.Nd4j;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.mockito.Mockito.*;

//@RunWith(JUnitPlatform.class)
@ExtendWith(MockitoExtension.class)
@Disabled("Mockito")
@Tag(TagNames.FILE_IO)
@NativeTag
public class AgentTest {
    @Mock Environment environmentMock;
    @Mock TransformProcess transformProcessMock;
    @Mock IPolicy policyMock;
    @Mock AgentListener listenerMock;



    @Test
    public void when_buildingWithNullEnvironment_expect_exception() {
        try {
            new Agent(null, null, null, null, null);
            fail("NullPointerException should have been thrown");
        } catch (NullPointerException exception) {
            String expectedMessage = "environment is marked non-null but is null";
            String actualMessage = exception.getMessage();

            assertTrue(actualMessage.contains(expectedMessage));
        }
    }

    @Test
    public void when_buildingWithNullTransformProcess_expect_exception() {
        try {
            new Agent(environmentMock, null, null, null, null);
            fail("NullPointerException should have been thrown");
        } catch (NullPointerException exception) {
            String expectedMessage = "transformProcess is marked non-null but is null";
            String actualMessage = exception.getMessage();

            assertTrue(actualMessage.contains(expectedMessage));
        }
    }

    @Test
    public void when_buildingWithNullPolicy_expect_exception() {
        try {
            new Agent(environmentMock, transformProcessMock, null, null, null);
            fail("NullPointerException should have been thrown");
        } catch (NullPointerException exception) {
            String expectedMessage = "policy is marked non-null but is null";
            String actualMessage = exception.getMessage();

            assertTrue(actualMessage.contains(expectedMessage));
        }
    }

    @Test
    public void when_buildingWithNullConfiguration_expect_exception() {
        try {
            new Agent(environmentMock, transformProcessMock, policyMock, null, null);
            fail("NullPointerException should have been thrown");
        } catch (NullPointerException exception) {
            String expectedMessage = "configuration is marked non-null but is null";
            String actualMessage = exception.getMessage();

            assertTrue(actualMessage.contains(expectedMessage));
        }
    }

    @Test
    public void when_buildingWithInvalidMaxSteps_expect_exception() {
        try {
            Agent.Configuration configuration = Agent.Configuration.builder()
                .maxEpisodeSteps(0)
                .build();
            new Agent(environmentMock, transformProcessMock, policyMock, configuration, null);
            fail("IllegalArgumentException should have been thrown");
        } catch (IllegalArgumentException exception) {
            String expectedMessage = "Configuration: maxEpisodeSteps must be null (no maximum) or greater than 0, got [0]";
            String actualMessage = exception.getMessage();

            assertTrue(actualMessage.contains(expectedMessage));
        }
    }

    @Test
    public void when_buildingWithId_expect_idSetInAgent() {
        // Arrange
        Agent.Configuration configuration = Agent.Configuration.builder().build();
        Agent sut = new Agent(environmentMock, transformProcessMock, policyMock, configuration, "TestAgent");

        // Assert
        assertEquals("TestAgent", sut.getId());
    }

    @Test
    public void when_runIsCalled_expect_agentIsReset() {
        // Arrange
        Map<String, Object> envResetResult = new HashMap<>();
        Schema schema = new Schema(new IntegerActionSchema(0, -1));
        when(environmentMock.reset()).thenReturn(envResetResult);
        when(environmentMock.getSchema()).thenReturn(schema);

        when(transformProcessMock.transform(any(Map.class), anyInt(), anyBoolean())).thenReturn(new Observation(Nd4j.create(new double[] { 123.0 })));
        when(policyMock.nextAction(any(Observation.class))).thenReturn(1);

        Agent.Configuration configuration = Agent.Configuration.builder().build();
        Agent sut = new Agent(environmentMock, transformProcessMock, policyMock, configuration, null);

        when(listenerMock.onBeforeStep(any(Agent.class), any(Observation.class), anyInt())).thenReturn(AgentListener.ListenerResponse.STOP);
        sut.addListener(listenerMock);

        // Act
        sut.run();

        // Assert
        assertEquals(0, sut.getEpisodeStepCount());
        verify(transformProcessMock).transform(envResetResult, 0, false);
        verify(policyMock, times(1)).reset();
        assertEquals(0.0, sut.getReward(), 0.00001);
        verify(environmentMock, times(1)).reset();
    }

    @Test
    public void when_runIsCalled_expect_onBeforeAndAfterEpisodeCalled() {
        // Arrange
        Map<String, Object> envResetResult = new HashMap<>();
        Schema schema = new Schema(new IntegerActionSchema(0, -1));
        when(environmentMock.reset()).thenReturn(envResetResult);
        when(environmentMock.getSchema()).thenReturn(schema);

        when(transformProcessMock.transform(any(Map.class), anyInt(), anyBoolean())).thenReturn(new Observation(Nd4j.create(new double[] { 123.0 })));
        when(environmentMock.isEpisodeFinished()).thenReturn(true);

        Agent.Configuration configuration = Agent.Configuration.builder().build();
        Agent sut = new Agent(environmentMock, transformProcessMock, policyMock, configuration, null);
        Agent spy = Mockito.spy(sut);

        // Act
        spy.run();

        // Assert
        verify(spy, times(1)).onBeforeEpisode();
        verify(spy, times(1)).onAfterEpisode();
    }

    @Test
    public void when_onBeforeEpisodeReturnsStop_expect_performStepAndOnAfterEpisodeNotCalled() {
        // Arrange
        Map<String, Object> envResetResult = new HashMap<>();
        Schema schema = new Schema(new IntegerActionSchema(0, -1));
        when(environmentMock.reset()).thenReturn(envResetResult);
        when(environmentMock.getSchema()).thenReturn(schema);

        when(transformProcessMock.transform(any(Map.class), anyInt(), anyBoolean())).thenReturn(new Observation(Nd4j.create(new double[] { 123.0 })));

        Agent.Configuration configuration = Agent.Configuration.builder().build();
        Agent sut = new Agent(environmentMock, transformProcessMock, policyMock, configuration, null);

        when(listenerMock.onBeforeEpisode(any(Agent.class))).thenReturn(AgentListener.ListenerResponse.STOP);
        sut.addListener(listenerMock);

        Agent spy = Mockito.spy(sut);

        // Act
        spy.run();

        // Assert
        verify(spy, times(1)).onBeforeEpisode();
        verify(spy, never()).performStep();
        verify(spy, never()).onAfterStep(any(StepResult.class));
        verify(spy, never()).onAfterEpisode();
    }

    @Test
    public void when_runIsCalledWithoutMaxStep_expect_agentRunUntilEpisodeIsFinished() {
        // Arrange
        Map<String, Object> envResetResult = new HashMap<>();
        Schema schema = new Schema(new IntegerActionSchema(0, -1));
        when(environmentMock.reset()).thenReturn(envResetResult);
        when(environmentMock.getSchema()).thenReturn(schema);

        when(transformProcessMock.transform(any(Map.class), anyInt(), anyBoolean())).thenReturn(new Observation(Nd4j.create(new double[] { 123.0 })));

        Agent.Configuration configuration = Agent.Configuration.builder().build();
        Agent sut = new Agent(environmentMock, transformProcessMock, policyMock, configuration, null);

        final Agent spy = Mockito.spy(sut);

        doAnswer(invocation -> {
            ((Agent)invocation.getMock()).incrementEpisodeStepCount();
            return null;
        }).when(spy).performStep();
        when(environmentMock.isEpisodeFinished()).thenAnswer(invocation -> spy.getEpisodeStepCount() >= 5 );

        // Act
        spy.run();

        // Assert
        verify(spy, times(1)).onBeforeEpisode();
        verify(spy, times(5)).performStep();
        verify(spy, times(1)).onAfterEpisode();
    }

    @Test
    public void when_maxStepsIsReachedBeforeEposideEnds_expect_runTerminated() {
        // Arrange
        Map<String, Object> envResetResult = new HashMap<>();
        Schema schema = new Schema(new IntegerActionSchema(0, -1));
        when(environmentMock.reset()).thenReturn(envResetResult);
        when(environmentMock.getSchema()).thenReturn(schema);

        when(transformProcessMock.transform(any(Map.class), anyInt(), anyBoolean())).thenReturn(new Observation(Nd4j.create(new double[] { 123.0 })));

        Agent.Configuration configuration = Agent.Configuration.builder()
                .maxEpisodeSteps(3)
                .build();
        Agent sut = new Agent(environmentMock, transformProcessMock, policyMock, configuration, null);

        final Agent spy = Mockito.spy(sut);

        doAnswer(invocation -> {
            ((Agent)invocation.getMock()).incrementEpisodeStepCount();
            return null;
        }).when(spy).performStep();

        // Act
        spy.run();

        // Assert
        verify(spy, times(1)).onBeforeEpisode();
        verify(spy, times(3)).performStep();
        verify(spy, times(1)).onAfterEpisode();
    }

    @Test
    public void when_initialObservationsAreSkipped_expect_performNoOpAction() {
        // Arrange
        Map<String, Object> envResetResult = new HashMap<>();
        Schema schema = new Schema(new IntegerActionSchema(0, -1));
        when(environmentMock.reset()).thenReturn(envResetResult);
        when(environmentMock.getSchema()).thenReturn(schema);

        when(transformProcessMock.transform(any(Map.class), anyInt(), anyBoolean())).thenReturn(Observation.SkippedObservation);

        Agent.Configuration configuration = Agent.Configuration.builder().build();
        Agent sut = new Agent(environmentMock, transformProcessMock, policyMock, configuration, null);

        when(listenerMock.onBeforeStep(any(Agent.class), any(Observation.class), any())).thenReturn(AgentListener.ListenerResponse.STOP);
        sut.addListener(listenerMock);

        Agent spy = Mockito.spy(sut);

        // Act
        spy.run();

        // Assert
        verify(listenerMock).onBeforeStep(any(), any(), eq(-1));
    }

    @Test
    public void when_initialObservationsAreSkipped_expect_performNoOpActionAnd() {
        // Arrange
        Map<String, Object> envResetResult = new HashMap<>();
        Schema schema = new Schema(new IntegerActionSchema(0, -1));
        when(environmentMock.reset()).thenReturn(envResetResult);
        when(environmentMock.getSchema()).thenReturn(schema);

        when(transformProcessMock.transform(any(Map.class), anyInt(), anyBoolean())).thenReturn(Observation.SkippedObservation);

        Agent.Configuration configuration = Agent.Configuration.builder().build();
        Agent sut = new Agent(environmentMock, transformProcessMock, policyMock, configuration, null);

        when(listenerMock.onBeforeStep(any(Agent.class), any(Observation.class), any())).thenReturn(AgentListener.ListenerResponse.STOP);
        sut.addListener(listenerMock);

        Agent spy = Mockito.spy(sut);

        // Act
        spy.run();

        // Assert
        verify(listenerMock).onBeforeStep(any(), any(), eq(-1));
    }

    @Test
    public void when_observationsIsSkipped_expect_performLastAction() {
        // Arrange
        Map<String, Object> envResetResult = new HashMap<>();
        Schema schema = new Schema(new IntegerActionSchema(0, -1));
        when(environmentMock.reset()).thenReturn(envResetResult);
        when(environmentMock.step(any(Integer.class))).thenReturn(new StepResult(envResetResult, 0.0, false));
        when(environmentMock.getSchema()).thenReturn(schema);

        when(policyMock.nextAction(any(Observation.class)))
                .thenAnswer(invocation -> (int)((Observation)invocation.getArgument(0)).getData().getDouble(0));

        Agent.Configuration configuration = Agent.Configuration.builder()
                .maxEpisodeSteps(3)
                .build();
        Agent sut = new Agent(environmentMock, transformProcessMock, policyMock, configuration, null);

        Agent spy = Mockito.spy(sut);

        when(transformProcessMock.transform(any(Map.class), anyInt(), anyBoolean()))
                .thenAnswer(invocation -> {
                    int stepNumber = (int)invocation.getArgument(1);
                    return stepNumber  % 2 == 1 ? Observation.SkippedObservation
                            : new Observation(Nd4j.create(new double[] {  stepNumber }));
                });

        sut.addListener(listenerMock);

        // Act
        spy.run();

        // Assert
        verify(policyMock, times(2)).nextAction(any(Observation.class));

        ArgumentCaptor<Agent> agentCaptor = ArgumentCaptor.forClass(Agent.class);
        ArgumentCaptor<Observation> observationCaptor = ArgumentCaptor.forClass(Observation.class);
        ArgumentCaptor<Integer> actionCaptor = ArgumentCaptor.forClass(Integer.class);
        verify(listenerMock, times(3)).onBeforeStep(agentCaptor.capture(), observationCaptor.capture(), actionCaptor.capture());
        List<Integer> capturedActions = actionCaptor.getAllValues();
        assertEquals(0, (int)capturedActions.get(0));
        assertEquals(0, (int)capturedActions.get(1));
        assertEquals(2, (int)capturedActions.get(2));
    }

    @Test
    public void when_onBeforeStepReturnsStop_expect_performStepAndOnAfterEpisodeNotCalled() {
        // Arrange
        Schema schema = new Schema(new IntegerActionSchema(0, -1));
        when(environmentMock.reset()).thenReturn(new HashMap<>());
        when(environmentMock.getSchema()).thenReturn(schema);

        when(transformProcessMock.transform(any(Map.class), anyInt(), anyBoolean())).thenReturn(new Observation(Nd4j.create(new double[] { 123.0 })));

        Agent.Configuration configuration = Agent.Configuration.builder().build();
        Agent sut = new Agent(environmentMock, transformProcessMock, policyMock, configuration, null);

        when(listenerMock.onBeforeStep(any(Agent.class), any(Observation.class), any())).thenReturn(AgentListener.ListenerResponse.STOP);
        sut.addListener(listenerMock);

        Agent spy = Mockito.spy(sut);

        // Act
        spy.run();

        // Assert
        verify(spy, times(1)).onBeforeEpisode();
        verify(spy, times(1)).onBeforeStep();
        verify(spy, never()).act(any());
        verify(spy, never()).onAfterStep(any(StepResult.class));
        verify(spy, never()).onAfterEpisode();
    }

    @Test
    public void when_observationIsNotSkipped_expect_policyActionIsSentToEnvironment() {
        // Arrange
        Schema schema = new Schema(new IntegerActionSchema(0, -1));
        when(environmentMock.reset()).thenReturn(new HashMap<>());
        when(environmentMock.getSchema()).thenReturn(schema);
        when(environmentMock.step(any(Integer.class))).thenReturn(new StepResult(new HashMap<>(), 0.0, false));

        when(transformProcessMock.transform(any(Map.class), anyInt(), anyBoolean())).thenReturn(new Observation(Nd4j.create(new double[] { 123.0 })));

        when(policyMock.nextAction(any(Observation.class))).thenReturn(123);

        Agent.Configuration configuration = Agent.Configuration.builder()
                .maxEpisodeSteps(1)
                .build();
        Agent sut = new Agent(environmentMock, transformProcessMock, policyMock, configuration, null);

        // Act
        sut.run();

        // Assert
        verify(environmentMock, times(1)).step(123);
    }

    @Test
    public void when_stepResultIsReceived_expect_observationAndRewardUpdated() {
        // Arrange
        Schema schema = new Schema(new IntegerActionSchema(0, -1));
        when(environmentMock.reset()).thenReturn(new HashMap<>());
        when(environmentMock.getSchema()).thenReturn(schema);
        when(environmentMock.step(any(Integer.class))).thenReturn(new StepResult(new HashMap<>(), 234.0, false));

        when(transformProcessMock.transform(any(Map.class), anyInt(), anyBoolean())).thenReturn(new Observation(Nd4j.create(new double[] { 123.0 })));

        when(policyMock.nextAction(any(Observation.class))).thenReturn(123);

        Agent.Configuration configuration = Agent.Configuration.builder()
                .maxEpisodeSteps(1)
                .build();
        Agent sut = new Agent(environmentMock, transformProcessMock, policyMock, configuration, null);

        // Act
        sut.run();

        // Assert
        assertEquals(123.0, sut.getObservation().getData().getDouble(0), 0.00001);
        assertEquals(234.0, sut.getReward(), 0.00001);
    }

    @Test
    public void when_stepIsDone_expect_onAfterStepAndWithStepResult() {
        // Arrange
        Schema schema = new Schema(new IntegerActionSchema(0, -1));
        when(environmentMock.reset()).thenReturn(new HashMap<>());
        when(environmentMock.getSchema()).thenReturn(schema);
        StepResult stepResult = new StepResult(new HashMap<>(), 234.0, false);
        when(environmentMock.step(any(Integer.class))).thenReturn(stepResult);

        when(transformProcessMock.transform(any(Map.class), anyInt(), anyBoolean())).thenReturn(new Observation(Nd4j.create(new double[] { 123.0 })));

        when(policyMock.nextAction(any(Observation.class))).thenReturn(123);

        Agent.Configuration configuration = Agent.Configuration.builder()
                .maxEpisodeSteps(1)
                .build();
        Agent sut = new Agent(environmentMock, transformProcessMock, policyMock, configuration, null);
        Agent spy = Mockito.spy(sut);

        // Act
        spy.run();

        // Assert
        verify(spy).onAfterStep(stepResult);
    }

    @Test
    public void when_onAfterStepReturnsStop_expect_onAfterEpisodeNotCalled() {
        // Arrange
        Schema schema = new Schema(new IntegerActionSchema(0, -1));
        when(environmentMock.reset()).thenReturn(new HashMap<>());
        when(environmentMock.getSchema()).thenReturn(schema);
        StepResult stepResult = new StepResult(new HashMap<>(), 234.0, false);
        when(environmentMock.step(any(Integer.class))).thenReturn(stepResult);

        when(transformProcessMock.transform(any(Map.class), anyInt(), anyBoolean())).thenReturn(new Observation(Nd4j.create(new double[] { 123.0 })));

        when(policyMock.nextAction(any(Observation.class))).thenReturn(123);

        Agent.Configuration configuration = Agent.Configuration.builder()
                .maxEpisodeSteps(1)
                .build();
        Agent sut = new Agent(environmentMock, transformProcessMock, policyMock, configuration, null);
        when(listenerMock.onAfterStep(any(Agent.class), any(StepResult.class))).thenReturn(AgentListener.ListenerResponse.STOP);
        sut.addListener(listenerMock);

        Agent spy = Mockito.spy(sut);

        // Act
        spy.run();

        // Assert
        verify(spy, never()).onAfterEpisode();
    }

    @Test
    public void when_runIsCalled_expect_onAfterEpisodeIsCalled() {
        // Arrange
        Schema schema = new Schema(new IntegerActionSchema(0, -1));
        when(environmentMock.reset()).thenReturn(new HashMap<>());
        when(environmentMock.getSchema()).thenReturn(schema);
        StepResult stepResult = new StepResult(new HashMap<>(), 234.0, false);
        when(environmentMock.step(any(Integer.class))).thenReturn(stepResult);

        when(transformProcessMock.transform(any(Map.class), anyInt(), anyBoolean())).thenReturn(new Observation(Nd4j.create(new double[] { 123.0 })));

        when(policyMock.nextAction(any(Observation.class))).thenReturn(123);

        Agent.Configuration configuration = Agent.Configuration.builder()
                .maxEpisodeSteps(1)
                .build();
        Agent sut = new Agent(environmentMock, transformProcessMock, policyMock, configuration, null);
        sut.addListener(listenerMock);
        Agent spy = Mockito.spy(sut);

        // Act
        spy.run();

        // Assert
        verify(spy, times(1)).onAfterEpisode();
        verify(listenerMock, times(1)).onAfterEpisode(any());
    }
}

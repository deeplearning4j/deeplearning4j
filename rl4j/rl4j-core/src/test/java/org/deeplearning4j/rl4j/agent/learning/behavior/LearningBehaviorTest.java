package org.deeplearning4j.rl4j.agent.learning.behavior;

import org.deeplearning4j.rl4j.agent.learning.behavior.LearningBehavior;
import org.deeplearning4j.rl4j.agent.learning.update.IUpdateRule;
import org.deeplearning4j.rl4j.experience.ExperienceHandler;
import org.deeplearning4j.rl4j.observation.Observation;
import org.junit.Before;
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
import static org.junit.Assert.assertFalse;
import static org.mockito.Mockito.*;

@RunWith(MockitoJUnitRunner.class)
public class LearningBehaviorTest {

    @Mock
    ExperienceHandler<Integer, Object> experienceHandlerMock;

    @Mock
    IUpdateRule<Object> updateRuleMock;

    LearningBehavior<Integer, Object> sut;

    @Before
    public void setup() {
        sut = LearningBehavior.<Integer, Object>builder()
            .experienceHandler(experienceHandlerMock)
            .updateRule(updateRuleMock)
            .build();
    }

    @Test
    public void when_callingHandleEpisodeStart_expect_experienceHandlerResetCalled() {
        // Arrange
        LearningBehavior<Integer, Object> sut = LearningBehavior.<Integer, Object>builder()
                .experienceHandler(experienceHandlerMock)
                .updateRule(updateRuleMock)
                .build();

        // Act
        sut.handleEpisodeStart();

        // Assert
        verify(experienceHandlerMock, times(1)).reset();
    }

    @Test
    public void when_callingHandleNewExperience_expect_experienceHandlerAddExperienceCalled() {
        // Arrange
        INDArray observationData = Nd4j.rand(1, 1);
        when(experienceHandlerMock.isTrainingBatchReady()).thenReturn(false);

        // Act
        sut.handleNewExperience(new Observation(observationData), 1, 2.0, false);

        // Assert
        ArgumentCaptor<Observation> observationCaptor = ArgumentCaptor.forClass(Observation.class);
        ArgumentCaptor<Integer> actionCaptor = ArgumentCaptor.forClass(Integer.class);
        ArgumentCaptor<Double> rewardCaptor = ArgumentCaptor.forClass(Double.class);
        ArgumentCaptor<Boolean> isTerminatedCaptor = ArgumentCaptor.forClass(Boolean.class);
        verify(experienceHandlerMock, times(1)).addExperience(observationCaptor.capture(), actionCaptor.capture(), rewardCaptor.capture(), isTerminatedCaptor.capture());

        assertEquals(observationData.getDouble(0, 0), observationCaptor.getValue().getData().getDouble(0, 0), 0.00001);
        assertEquals(1, (int)actionCaptor.getValue());
        assertEquals(2.0, (double)rewardCaptor.getValue(), 0.00001);
        assertFalse(isTerminatedCaptor.getValue());

        verify(updateRuleMock, never()).update(any(List.class));
    }

    @Test
    public void when_callingHandleNewExperienceAndTrainingBatchIsReady_expect_updateRuleUpdateWithTrainingBatch() {
        // Arrange
        INDArray observationData = Nd4j.rand(1, 1);
        when(experienceHandlerMock.isTrainingBatchReady()).thenReturn(true);
        List<Object> trainingBatch = new ArrayList<Object>();
        when(experienceHandlerMock.generateTrainingBatch()).thenReturn(trainingBatch);

        // Act
        sut.handleNewExperience(new Observation(observationData), 1, 2.0, false);

        // Assert
        verify(updateRuleMock, times(1)).update(trainingBatch);
    }

    @Test
    public void when_callingHandleEpisodeEnd_expect_experienceHandlerSetFinalObservationCalled() {
        // Arrange
        INDArray observationData = Nd4j.rand(1, 1);
        when(experienceHandlerMock.isTrainingBatchReady()).thenReturn(false);

        // Act
        sut.handleEpisodeEnd(new Observation(observationData));

        // Assert
        ArgumentCaptor<Observation> observationCaptor = ArgumentCaptor.forClass(Observation.class);
        verify(experienceHandlerMock, times(1)).setFinalObservation(observationCaptor.capture());

        assertEquals(observationData.getDouble(0, 0), observationCaptor.getValue().getData().getDouble(0, 0), 0.00001);

        verify(updateRuleMock, never()).update(any(List.class));
    }

    @Test
    public void when_callingHandleEpisodeEndAndTrainingBatchIsNotEmpty_expect_updateRuleUpdateWithTrainingBatch() {
        // Arrange
        INDArray observationData = Nd4j.rand(1, 1);
        when(experienceHandlerMock.isTrainingBatchReady()).thenReturn(true);
        List<Object> trainingBatch = new ArrayList<Object>();
        when(experienceHandlerMock.generateTrainingBatch()).thenReturn(trainingBatch);

        // Act
        sut.handleEpisodeEnd(new Observation(observationData));

        // Assert
        ArgumentCaptor<Observation> observationCaptor = ArgumentCaptor.forClass(Observation.class);
        verify(experienceHandlerMock, times(1)).setFinalObservation(observationCaptor.capture());

        assertEquals(observationData.getDouble(0, 0), observationCaptor.getValue().getData().getDouble(0, 0), 0.00001);

        verify(updateRuleMock, times(1)).update(trainingBatch);
    }
}

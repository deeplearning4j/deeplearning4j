package org.deeplearning4j.rl4j.experience;

import org.deeplearning4j.rl4j.learning.sync.IExpReplay;
import org.deeplearning4j.rl4j.learning.sync.Transition;
import org.deeplearning4j.rl4j.observation.Observation;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.junit.MockitoJUnitRunner;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;

import static org.junit.Assert.*;
import static org.mockito.Mockito.*;

@RunWith(MockitoJUnitRunner.class)
public class ReplayMemoryExperienceHandlerTest {

    @Mock
    IExpReplay<Integer> expReplayMock;

    private ReplayMemoryExperienceHandler.Configuration buildConfiguration() {
        return ReplayMemoryExperienceHandler.Configuration.builder()
                .maxReplayMemorySize(10)
                .batchSize(5)
                .build();
    }

    @Test
    public void when_addingFirstExperience_expect_notAddedToStoreBeforeNextObservationIsAdded() {
        // Arrange
        when(expReplayMock.getDesignatedBatchSize()).thenReturn(10);

        ReplayMemoryExperienceHandler sut = new ReplayMemoryExperienceHandler(expReplayMock);

        // Act
        sut.addExperience(new Observation(Nd4j.create(new double[] { 1.0 })), 1, 1.0, false);
        boolean isStoreCalledAfterFirstAdd = mockingDetails(expReplayMock).getInvocations().stream().anyMatch(x -> x.getMethod().getName() == "store");
        sut.addExperience(new Observation(Nd4j.create(new double[] { 2.0 })), 2, 2.0, false);
        boolean isStoreCalledAfterSecondAdd = mockingDetails(expReplayMock).getInvocations().stream().anyMatch(x -> x.getMethod().getName() == "store");

        // Assert
        assertFalse(isStoreCalledAfterFirstAdd);
        assertTrue(isStoreCalledAfterSecondAdd);
    }

    @Test
    public void when_addingExperience_expect_transitionsAreCorrect() {
        // Arrange
        ReplayMemoryExperienceHandler sut = new ReplayMemoryExperienceHandler(expReplayMock);

        // Act
        sut.addExperience(new Observation(Nd4j.create(new double[] { 1.0 })), 1, 1.0, false);
        sut.addExperience(new Observation(Nd4j.create(new double[] { 2.0 })), 2, 2.0, false);
        sut.setFinalObservation(new Observation(Nd4j.create(new double[] { 3.0 })));

        // Assert
        ArgumentCaptor<Transition<Integer>> argument = ArgumentCaptor.forClass(Transition.class);
        verify(expReplayMock, times(2)).store(argument.capture());
        List<Transition<Integer>> transitions = argument.getAllValues();

        assertEquals(1.0, transitions.get(0).getObservation().getData().getDouble(0), 0.00001);
        assertEquals(1, (int)transitions.get(0).getAction());
        assertEquals(1.0, transitions.get(0).getReward(), 0.00001);
        assertEquals(2.0, transitions.get(0).getNextObservation().getDouble(0), 0.00001);

        assertEquals(2.0, transitions.get(1).getObservation().getData().getDouble(0), 0.00001);
        assertEquals(2, (int)transitions.get(1).getAction());
        assertEquals(2.0, transitions.get(1).getReward(), 0.00001);
        assertEquals(3.0, transitions.get(1).getNextObservation().getDouble(0), 0.00001);

    }

    @Test
    public void when_settingFinalObservation_expect_nextAddedExperienceDoNotUsePreviousObservation() {
        // Arrange
        ReplayMemoryExperienceHandler sut = new ReplayMemoryExperienceHandler(expReplayMock);

        // Act
        sut.addExperience(new Observation(Nd4j.create(new double[] { 1.0 })), 1, 1.0, false);
        sut.setFinalObservation(new Observation(Nd4j.create(new double[] { 2.0 })));
        sut.addExperience(new Observation(Nd4j.create(new double[] { 3.0 })), 3, 3.0, false);

        // Assert
        ArgumentCaptor<Transition<Integer>> argument = ArgumentCaptor.forClass(Transition.class);
        verify(expReplayMock, times(1)).store(argument.capture());
        Transition<Integer> transition = argument.getValue();

        assertEquals(1, (int)transition.getAction());
    }

    @Test
    public void when_addingExperience_expect_getTrainingBatchSizeReturnSize() {
        // Arrange
        ReplayMemoryExperienceHandler sut = new ReplayMemoryExperienceHandler(buildConfiguration(), Nd4j.getRandom());
        sut.addExperience(new Observation(Nd4j.create(new double[] { 1.0 })), 1, 1.0, false);
        sut.addExperience(new Observation(Nd4j.create(new double[] { 2.0 })), 2, 2.0, false);
        sut.setFinalObservation(new Observation(Nd4j.create(new double[] { 3.0 })));

        // Act
        int size = sut.getTrainingBatchSize();

        // Assert
        assertEquals(2, size);
    }

    @Test
    public void when_experienceSizeIsSmallerThanBatchSize_expect_TrainingBatchIsNotReady() {
        // Arrange
        ReplayMemoryExperienceHandler sut = new ReplayMemoryExperienceHandler(buildConfiguration(), Nd4j.getRandom());
        sut.addExperience(new Observation(Nd4j.create(new double[] { 1.0 })), 1, 1.0, false);
        sut.addExperience(new Observation(Nd4j.create(new double[] { 2.0 })), 2, 2.0, false);
        sut.setFinalObservation(new Observation(Nd4j.create(new double[] { 3.0 })));

        // Act

        // Assert
        assertFalse(sut.isTrainingBatchReady());
    }

    @Test
    public void when_experienceSizeIsGreaterOrEqualToBatchSize_expect_TrainingBatchIsReady() {
        // Arrange
        ReplayMemoryExperienceHandler sut = new ReplayMemoryExperienceHandler(buildConfiguration(), Nd4j.getRandom());
        sut.addExperience(new Observation(Nd4j.create(new double[] { 1.0 })), 1, 1.0, false);
        sut.addExperience(new Observation(Nd4j.create(new double[] { 2.0 })), 2, 2.0, false);
        sut.addExperience(new Observation(Nd4j.create(new double[] { 3.0 })), 3, 3.0, false);
        sut.addExperience(new Observation(Nd4j.create(new double[] { 4.0 })), 4, 4.0, false);
        sut.addExperience(new Observation(Nd4j.create(new double[] { 5.0 })), 5, 5.0, false);
        sut.setFinalObservation(new Observation(Nd4j.create(new double[] { 6.0 })));

        // Act

        // Assert
        assertTrue(sut.isTrainingBatchReady());
    }

}

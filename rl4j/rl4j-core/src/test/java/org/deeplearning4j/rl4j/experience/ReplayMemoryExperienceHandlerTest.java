package org.deeplearning4j.rl4j.experience;

import org.deeplearning4j.rl4j.learning.sync.IExpReplay;
import org.deeplearning4j.rl4j.learning.sync.Transition;
import org.deeplearning4j.rl4j.observation.Observation;
import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;

public class ReplayMemoryExperienceHandlerTest {
    @Test
    public void when_addingFirstExperience_expect_notAddedToStoreBeforeNextObservationIsAdded() {
        // Arrange
        TestExpReplay expReplayMock = new TestExpReplay();
        ReplayMemoryExperienceHandler sut = new ReplayMemoryExperienceHandler(expReplayMock);

        // Act
        sut.addExperience(new Observation(Nd4j.create(new double[] { 1.0 })), 1, 1.0, false);
        int numStoredTransitions = expReplayMock.addedTransitions.size();
        sut.addExperience(new Observation(Nd4j.create(new double[] { 2.0 })), 2, 2.0, false);

        // Assert
        assertEquals(0, numStoredTransitions);
        assertEquals(1, expReplayMock.addedTransitions.size());
    }

    @Test
    public void when_addingExperience_expect_transitionsAreCorrect() {
        // Arrange
        TestExpReplay expReplayMock = new TestExpReplay();
        ReplayMemoryExperienceHandler sut = new ReplayMemoryExperienceHandler(expReplayMock);

        // Act
        sut.addExperience(new Observation(Nd4j.create(new double[] { 1.0 })), 1, 1.0, false);
        sut.addExperience(new Observation(Nd4j.create(new double[] { 2.0 })), 2, 2.0, false);
        sut.setFinalObservation(new Observation(Nd4j.create(new double[] { 3.0 })));

        // Assert
        assertEquals(2, expReplayMock.addedTransitions.size());

        assertEquals(1.0, expReplayMock.addedTransitions.get(0).getObservation().getData().getDouble(0), 0.00001);
        assertEquals(1, (int)expReplayMock.addedTransitions.get(0).getAction());
        assertEquals(1.0, expReplayMock.addedTransitions.get(0).getReward(), 0.00001);
        assertEquals(2.0, expReplayMock.addedTransitions.get(0).getNextObservation().getDouble(0), 0.00001);

        assertEquals(2.0, expReplayMock.addedTransitions.get(1).getObservation().getData().getDouble(0), 0.00001);
        assertEquals(2, (int)expReplayMock.addedTransitions.get(1).getAction());
        assertEquals(2.0, expReplayMock.addedTransitions.get(1).getReward(), 0.00001);
        assertEquals(3.0, expReplayMock.addedTransitions.get(1).getNextObservation().getDouble(0), 0.00001);

    }

    @Test
    public void when_settingFinalObservation_expect_nextAddedExperienceDoNotUsePreviousObservation() {
        // Arrange
        TestExpReplay expReplayMock = new TestExpReplay();
        ReplayMemoryExperienceHandler sut = new ReplayMemoryExperienceHandler(expReplayMock);

        // Act
        sut.addExperience(new Observation(Nd4j.create(new double[] { 1.0 })), 1, 1.0, false);
        sut.setFinalObservation(new Observation(Nd4j.create(new double[] { 2.0 })));
        sut.addExperience(new Observation(Nd4j.create(new double[] { 3.0 })), 3, 3.0, false);

        // Assert
        assertEquals(1, expReplayMock.addedTransitions.size());
        assertEquals(1, (int)expReplayMock.addedTransitions.get(0).getAction());
    }

    @Test
    public void when_addingExperience_expect_getTrainingBatchSizeReturnSize() {
        // Arrange
        TestExpReplay expReplayMock = new TestExpReplay();
        ReplayMemoryExperienceHandler sut = new ReplayMemoryExperienceHandler(expReplayMock);
        sut.addExperience(new Observation(Nd4j.create(new double[] { 1.0 })), 1, 1.0, false);
        sut.addExperience(new Observation(Nd4j.create(new double[] { 2.0 })), 2, 2.0, false);
        sut.setFinalObservation(new Observation(Nd4j.create(new double[] { 3.0 })));

        // Act
        int size = sut.getTrainingBatchSize();
        // Assert
        assertEquals(2, size);
    }

    private static class TestExpReplay implements IExpReplay<Integer> {

        public final List<Transition<Integer>> addedTransitions = new ArrayList<>();

        @Override
        public ArrayList<Transition<Integer>> getBatch() {
            return null;
        }

        @Override
        public void store(Transition<Integer> transition) {
            addedTransitions.add(transition);
        }

        @Override
        public int getBatchSize() {
            return addedTransitions.size();
        }
    }
}

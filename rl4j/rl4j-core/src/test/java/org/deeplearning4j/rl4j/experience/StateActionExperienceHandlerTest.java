package org.deeplearning4j.rl4j.experience;

import org.deeplearning4j.rl4j.observation.Observation;
import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;

import static org.junit.Assert.*;

public class StateActionExperienceHandlerTest {

    @Test
    public void when_addingExperience_expect_generateTrainingBatchReturnsIt() {
        // Arrange
        StateActionExperienceHandler sut = new StateActionExperienceHandler();
        sut.reset();
        Observation observation = new Observation(Nd4j.zeros(1));
        sut.addExperience(observation, 123, 234.0, true);

        // Act
        List<StateActionPair<Integer>> result = sut.generateTrainingBatch();

        // Assert
        assertEquals(1, result.size());
        assertSame(observation, result.get(0).getObservation());
        assertEquals(123, (int)result.get(0).getAction());
        assertEquals(234.0, result.get(0).getReward(), 0.00001);
        assertTrue(result.get(0).isTerminal());
    }

    @Test
    public void when_addingMultipleExperiences_expect_generateTrainingBatchReturnsItInSameOrder() {
        // Arrange
        StateActionExperienceHandler sut = new StateActionExperienceHandler();
        sut.reset();
        sut.addExperience(null, 1, 1.0, false);
        sut.addExperience(null, 2, 2.0, false);
        sut.addExperience(null, 3, 3.0, false);

        // Act
        List<StateActionPair<Integer>> result = sut.generateTrainingBatch();

        // Assert
        assertEquals(3, result.size());
        assertEquals(1, (int)result.get(0).getAction());
        assertEquals(2, (int)result.get(1).getAction());
        assertEquals(3, (int)result.get(2).getAction());
    }

    @Test
    public void when_gettingExperience_expect_experienceStoreIsCleared() {
        // Arrange
        StateActionExperienceHandler sut = new StateActionExperienceHandler();
        sut.reset();
        sut.addExperience(null, 1, 1.0, false);

        // Act
        List<StateActionPair<Integer>> firstResult = sut.generateTrainingBatch();
        List<StateActionPair<Integer>> secondResult = sut.generateTrainingBatch();

        // Assert
        assertEquals(1, firstResult.size());
        assertEquals(0, secondResult.size());
    }

    @Test
    public void when_addingExperience_expect_getTrainingBatchSizeReturnSize() {
        // Arrange
        StateActionExperienceHandler sut = new StateActionExperienceHandler();
        sut.reset();
        sut.addExperience(null, 1, 1.0, false);
        sut.addExperience(null, 2, 2.0, false);
        sut.addExperience(null, 3, 3.0, false);

        // Act
        int size = sut.getTrainingBatchSize();

        // Assert
        assertEquals(3, size);
    }
}

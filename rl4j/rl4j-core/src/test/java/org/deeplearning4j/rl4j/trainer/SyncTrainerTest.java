package org.deeplearning4j.rl4j.trainer;

import org.apache.commons.lang3.builder.Builder;
import org.deeplearning4j.rl4j.agent.IAgentLearner;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.Mock;
import org.mockito.junit.MockitoJUnitRunner;

import java.util.function.Predicate;

import static org.junit.Assert.assertEquals;
import static org.mockito.Mockito.*;

@RunWith(MockitoJUnitRunner.class)
public class SyncTrainerTest {

    @Mock
    IAgentLearner agentLearnerMock;

    @Mock
    Builder<IAgentLearner> agentLearnerBuilder;

    SyncTrainer sut;

    public void setup(Predicate<SyncTrainer> stoppingCondition) {
        when(agentLearnerBuilder.build()).thenReturn(agentLearnerMock);
        when(agentLearnerMock.getEpisodeStepCount()).thenReturn(10);

        sut = new SyncTrainer(agentLearnerBuilder, stoppingCondition);
    }

    @Test
    public void when_training_expect_stoppingConditionWillStopTraining() {
        // Arrange
        Predicate<SyncTrainer> stoppingCondition = t -> t.getEpisodeCount() >= 5; // Stop after 5 episodes
        setup(stoppingCondition);

        // Act
        sut.train();

        // Assert
        assertEquals(5, sut.getEpisodeCount());
    }

    @Test
    public void when_training_expect_agentIsRun() {
        // Arrange
        Predicate<SyncTrainer> stoppingCondition = t -> t.getEpisodeCount() >= 5; // Stop after 5 episodes
        setup(stoppingCondition);

        // Act
        sut.train();

        // Assert
        verify(agentLearnerMock, times(5)).run();
    }

    @Test
    public void when_training_expect_countsAreReset() {
        // Arrange
        Predicate<SyncTrainer> stoppingCondition = t -> t.getEpisodeCount() >= 5; // Stop after 5 episodes
        setup(stoppingCondition);

        // Act
        sut.train();
        sut.train();

        // Assert
        assertEquals(5, sut.getEpisodeCount());
        assertEquals(50, sut.getStepCount());
    }

}

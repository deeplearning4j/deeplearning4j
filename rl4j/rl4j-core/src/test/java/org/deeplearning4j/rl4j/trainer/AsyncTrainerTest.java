package org.deeplearning4j.rl4j.trainer;

import org.apache.commons.lang3.builder.Builder;
import org.deeplearning4j.rl4j.agent.IAgentLearner;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.Mock;
import org.mockito.junit.MockitoJUnitRunner;

import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Predicate;

import static org.junit.Assert.*;
import static org.mockito.Mockito.*;

@RunWith(MockitoJUnitRunner.class)
public class AsyncTrainerTest {

    @Mock
    Builder<IAgentLearner<Integer>> agentLearnerBuilderMock;

    @Mock
    Predicate<AsyncTrainer<Integer>> stoppingConditionMock;

    @Mock
    IAgentLearner<Integer> agentLearnerMock;

    @Before
    public void setup() {
        when(agentLearnerBuilderMock.build()).thenReturn(agentLearnerMock);
        when(agentLearnerMock.getEpisodeStepCount()).thenReturn(100);
    }

    @Test
    public void when_ctorIsCalledWithInvalidNumberOfThreads_expect_Exception() {
        try {
            AsyncTrainer sut = new AsyncTrainer(agentLearnerBuilderMock, stoppingConditionMock, 0);
            fail("IllegalArgumentException should have been thrown");
        } catch (IllegalArgumentException exception) {
            String expectedMessage = "numThreads must be greater than 0, got:  [0]";
            String actualMessage = exception.getMessage();

            assertTrue(actualMessage.contains(expectedMessage));
        }
    }

    @Test
    public void when_runningWith2Threads_expect_2AgentLearnerCreated() {
        // Arrange
        Predicate<AsyncTrainer<Integer>> stoppingCondition = t -> true;
        AsyncTrainer sut = new AsyncTrainer(agentLearnerBuilderMock, stoppingCondition, 2);

        // Act
        sut.train();

        // Assert
        verify(agentLearnerBuilderMock, times(2)).build();
    }

    @Test
    public void when_stoppingConditionTriggered_expect_agentLearnersStopsAndCountersAreCorrect() {
        // Arrange
        AtomicInteger stoppingConditionHitCount = new AtomicInteger(0);
        Predicate<AsyncTrainer<Integer>> stoppingCondition = t -> stoppingConditionHitCount.incrementAndGet() >= 5;
        AsyncTrainer<Integer> sut = new AsyncTrainer<Integer>(agentLearnerBuilderMock, stoppingCondition, 2);

        // Act
        sut.train();

        // Assert
        assertEquals(6, stoppingConditionHitCount.get());
        assertEquals(6, sut.getEpisodeCount());
        assertEquals(600, sut.getStepCount());
    }

    @Test
    public void when_training_expect_countsAreReset() {
        // Arrange
        AtomicInteger stoppingConditionHitCount = new AtomicInteger(0);
        Predicate<AsyncTrainer<Integer>> stoppingCondition = t -> stoppingConditionHitCount.incrementAndGet() >= 5;
        AsyncTrainer<Integer> sut = new AsyncTrainer<Integer>(agentLearnerBuilderMock, stoppingCondition, 2);

        // Act
        sut.train();
        stoppingConditionHitCount.set(0);
        sut.train();

        // Assert
        assertEquals(6, sut.getEpisodeCount());
        assertEquals(600, sut.getStepCount());
    }
}

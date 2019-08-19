package org.deeplearning4j.rl4j.learning.listener;

import org.deeplearning4j.rl4j.support.MockTrainingListener;
import org.junit.Test;

import static org.junit.Assert.*;

public class TrainingListenerListTest {
    @Test
    public void when_listIsEmpty_expect_notifyReturnTrue() {
        // Arrange
        TrainingListenerList sut = new TrainingListenerList();

        // Act
        boolean resultTrainingStarted = sut.notifyTrainingStarted(null);
        boolean resultNewEpoch = sut.notifyNewEpoch(null);
        boolean resultEpochFinished = sut.notifyEpochTrainingResult(null);

        // Assert
        assertTrue(resultTrainingStarted);
        assertTrue(resultNewEpoch);
        assertTrue(resultEpochFinished);
    }

    @Test
    public void when_firstListerStops_expect_othersListnersNotCalled() {
        // Arrange
        MockTrainingListener listener1 = new MockTrainingListener();
        listener1.setRemainingTrainingStartCallCount(0);
        listener1.setRemainingOnNewEpochCallCount(0);
        listener1.setRemainingOnEpochTrainingResult(0);
        MockTrainingListener listener2 = new MockTrainingListener();
        TrainingListenerList sut = new TrainingListenerList();
        sut.add(listener1);
        sut.add(listener2);

        // Act
        sut.notifyTrainingStarted(null);
        sut.notifyNewEpoch(null);
        sut.notifyEpochTrainingResult(new EpochTrainingResultEvent(0, 0, null));
        sut.notifyTrainingFinished(null);

        // Assert
        assertEquals(1, listener1.onTrainingStartCallCount);
        assertEquals(0, listener2.onTrainingStartCallCount);

        assertEquals(1, listener1.onNewEpochCallCount);
        assertEquals(0, listener2.onNewEpochCallCount);

        assertEquals(1, listener1.onEpochTrainingResultCallCount);
        assertEquals(0, listener2.onEpochTrainingResultCallCount);

        assertEquals(1, listener1.onTrainingEndCallCount);
        assertEquals(1, listener2.onTrainingEndCallCount);
    }

    @Test
    public void when_allListenersContinue_expect_listReturnsTrue() {
        // Arrange
        MockTrainingListener listener1 = new MockTrainingListener();
        MockTrainingListener listener2 = new MockTrainingListener();
        TrainingListenerList sut = new TrainingListenerList();
        sut.add(listener1);
        sut.add(listener2);

        // Act
        boolean resultTrainingStarted = sut.notifyTrainingStarted(null);
        boolean resultNewEpoch = sut.notifyNewEpoch(null);
        boolean resultEpochTrainingResult = sut.notifyEpochTrainingResult(new EpochTrainingResultEvent(0, 0, null) );

        // Assert
        assertTrue(resultTrainingStarted);
        assertTrue(resultNewEpoch);
        assertTrue(resultEpochTrainingResult);
    }
}

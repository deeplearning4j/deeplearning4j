package org.deeplearning4j.rl4j.learning.async.listener;

import org.deeplearning4j.rl4j.learning.listener.EpochTrainingResultEvent;
import org.deeplearning4j.rl4j.learning.listener.TrainingEvent;
import org.deeplearning4j.rl4j.learning.listener.TrainingListener;
import org.deeplearning4j.rl4j.learning.listener.TrainingListenerList;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class AsyncTrainingListenerListTest {
    @Test
    public void when_listIsEmpty_expect_notifyTrainingStartedReturnTrue() {
        // Arrange
        TrainingListenerList sut = new TrainingListenerList();

        // Act
        boolean resultTrainingStarted = sut.notifyTrainingStarted(null);
        boolean resultNewEpoch = sut.notifyNewEpoch(null);
        boolean resultEpochTrainingResult = sut.notifyEpochTrainingResult(null);

        // Assert
        assertTrue(resultTrainingStarted);
        assertTrue(resultNewEpoch);
        assertTrue(resultEpochTrainingResult);
    }

    @Test
    public void when_firstListerStops_expect_othersListnersNotCalled() {
        // Arrange
        MockTrainingListener listener1 = new MockTrainingListener();
        listener1.onTrainingResultResponse = TrainingListener.ListenerResponse.STOP;
        MockTrainingListener listener2 = new MockTrainingListener();
        TrainingListenerList sut = new TrainingListenerList();
        sut.add(listener1);
        sut.add(listener2);

        // Act
        sut.notifyEpochTrainingResult(null);

        // Assert
        assertEquals(1, listener1.onEpochTrainingResultCallCount);
        assertEquals(0, listener2.onEpochTrainingResultCallCount);
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
        boolean resultTrainingProgress = sut.notifyEpochTrainingResult(null);

        // Assert
        assertTrue(resultTrainingProgress);
    }

    private static class MockTrainingListener implements TrainingListener {

        public int onEpochTrainingResultCallCount = 0;
        public ListenerResponse onTrainingResultResponse = ListenerResponse.CONTINUE;

        @Override
        public ListenerResponse onTrainingStart(TrainingEvent event) {
            return ListenerResponse.CONTINUE;
        }

        @Override
        public void onTrainingEnd(TrainingEvent event) {

        }

        @Override
        public ListenerResponse onNewEpoch(TrainingEvent event) {
            return ListenerResponse.CONTINUE;
        }

        @Override
        public ListenerResponse onEpochTrainingResult(EpochTrainingResultEvent event) {
            ++onEpochTrainingResultCallCount;
            return onTrainingResultResponse;
        }
    }

}

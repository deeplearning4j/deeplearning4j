package org.deeplearning4j.rl4j.learning.async.listener;

import org.deeplearning4j.rl4j.learning.listener.TrainingListener;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class AsyncTrainingListenerListTest {
    @Test
    public void when_listIsEmpty_expect_notifyTrainingStartedReturnTrue() {
        // Arrange
        AsyncTrainingListenerList sut = new AsyncTrainingListenerList();

        // Act
        boolean resultTrainingStarted = sut.notifyTrainingStarted(null);
        boolean resultEpochStarted = sut.notifyEpochStarted(null);
        boolean resultEpochFinished = sut.notifyEpochFinished(null);

        // Assert
        assertTrue(resultTrainingStarted);
        assertTrue(resultEpochStarted);
        assertTrue(resultEpochFinished);
    }

    @Test
    public void when_firstListerStops_expect_othersListnersNotCalled() {
        // Arrange
        MockAsyncTrainingListener listener1 = new MockAsyncTrainingListener();
        listener1.onTrainingProgressResponse = TrainingListener.ListenerResponse.STOP;
        MockAsyncTrainingListener listener2 = new MockAsyncTrainingListener();
        AsyncTrainingListenerList sut = new AsyncTrainingListenerList();
        sut.add(listener1);
        sut.add(listener2);

        // Act
        sut.notifyTrainingProgress(null);

        // Assert
        assertEquals(1, listener1.onTrainingProgressCallCount);
        assertEquals(0, listener2.onTrainingProgressCallCount);
    }

    @Test
    public void when_allListenersContinue_expect_listReturnsTrue() {
        // Arrange
        MockAsyncTrainingListener listener1 = new MockAsyncTrainingListener();
        MockAsyncTrainingListener listener2 = new MockAsyncTrainingListener();
        AsyncTrainingListenerList sut = new AsyncTrainingListenerList();
        sut.add(listener1);
        sut.add(listener2);

        // Act
        boolean resultTrainingProgress = sut.notifyTrainingProgress(null);

        // Assert
        assertTrue(resultTrainingProgress);
    }

    private static class MockAsyncTrainingListener implements AsyncTrainingListener {

        public int onTrainingProgressCallCount = 0;
        public ListenerResponse onTrainingProgressResponse = ListenerResponse.CONTINUE;

        @Override
        public ListenerResponse onTrainingProgress(AsyncTrainingEvent event) {
            ++onTrainingProgressCallCount;
            return onTrainingProgressResponse;
        }

        @Override
        public ListenerResponse onTrainingStart(AsyncTrainingEvent event) {
            return null;
        }

        @Override
        public void onTrainingEnd() {

        }

        @Override
        public ListenerResponse onEpochStart(AsyncTrainingEpochEvent event) {
            return null;
        }

        @Override
        public ListenerResponse onEpochEnd(AsyncTrainingEpochEndEvent event) {
            return null;
        }
    }

}

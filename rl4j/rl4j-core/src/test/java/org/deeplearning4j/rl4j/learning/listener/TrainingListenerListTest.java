package org.deeplearning4j.rl4j.learning.listener;

import org.deeplearning4j.rl4j.util.IDataManager;
import org.junit.Test;

import static org.junit.Assert.*;

public class TrainingListenerListTest {
    @Test
    public void when_listIsEmpty_expect_notifyReturnTrue() {
        // Arrange
        TestTrainingListenerList sut = new TestTrainingListenerList();

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
        MockTrainingListener listener1 = new MockTrainingListener();
        listener1.onTrainingStartResponse = TrainingListener.ListenerResponse.STOP;
        listener1.onEpochStartResponse = TrainingListener.ListenerResponse.STOP;
        listener1.onEpochEndResponse = TrainingListener.ListenerResponse.STOP;
        MockTrainingListener listener2 = new MockTrainingListener();
        TestTrainingListenerList sut = new TestTrainingListenerList();
        sut.add(listener1);
        sut.add(listener2);

        // Act
        sut.notifyTrainingStarted(null);
        sut.notifyEpochStarted(null);
        sut.notifyEpochFinished(null);
        sut.notifyTrainingFinished();

        // Assert
        assertEquals(1, listener1.onTrainingStartCallCount);
        assertEquals(0, listener2.onTrainingStartCallCount);

        assertEquals(1, listener1.onEpochStartCallCount);
        assertEquals(0, listener2.onEpochStartCallCount);

        assertEquals(1, listener1.onEpochEndCallCount);
        assertEquals(0, listener2.onEpochEndCallCount);

        assertEquals(1, listener1.onTrainingEndCallCount);
        assertEquals(1, listener2.onTrainingEndCallCount);
    }

    @Test
    public void when_allListenersContinue_expect_listReturnsTrue() {
        // Arrange
        MockTrainingListener listener1 = new MockTrainingListener();
        MockTrainingListener listener2 = new MockTrainingListener();
        TestTrainingListenerList sut = new TestTrainingListenerList();
        sut.add(listener1);
        sut.add(listener2);

        // Act
        boolean resultTrainingStarted = sut.notifyTrainingStarted(null);
        boolean resultEpochStarted = sut.notifyEpochStarted(null);
        boolean resultEpochFinished = sut.notifyEpochFinished(null);

        // Assert
        assertTrue(resultTrainingStarted);
        assertTrue(resultEpochStarted);
        assertTrue(resultEpochFinished);
    }

    private static class MockTrainingEvent implements TrainingEvent {
    }

    private static class MockTrainingEpochEndEvent implements TrainingEpochEndEvent {
        @Override
        public IDataManager.StatEntry getStatEntry() {
            return null;
        }
    }

    private static class MockTrainingListener implements TrainingListener<MockTrainingEvent, MockTrainingEvent, MockTrainingEpochEndEvent> {

        public int onTrainingStartCallCount = 0;
        public int onTrainingEndCallCount = 0;
        public int onEpochStartCallCount = 0;
        public int onEpochEndCallCount = 0;

        public ListenerResponse onTrainingStartResponse = ListenerResponse.CONTINUE;
        public ListenerResponse onEpochStartResponse = ListenerResponse.CONTINUE;
        public ListenerResponse onEpochEndResponse = ListenerResponse.CONTINUE;

        @Override
        public ListenerResponse onTrainingStart(MockTrainingEvent event) {
            ++onTrainingStartCallCount;
            return onTrainingStartResponse;
        }

        @Override
        public void onTrainingEnd() {
            ++onTrainingEndCallCount;
        }

        @Override
        public ListenerResponse onEpochStart(MockTrainingEvent event) {
            ++onEpochStartCallCount;
            return onEpochStartResponse;
        }

        @Override
        public ListenerResponse onEpochEnd(MockTrainingEpochEndEvent event) {
            ++onEpochEndCallCount;
            return onEpochEndResponse;
        }
    }

    private static class TestTrainingListenerList extends TrainingListenerList<MockTrainingEvent, MockTrainingEvent, MockTrainingEpochEndEvent, MockTrainingListener> {

    }
}

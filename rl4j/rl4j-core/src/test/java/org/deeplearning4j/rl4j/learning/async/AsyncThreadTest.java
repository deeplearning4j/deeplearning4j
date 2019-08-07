package org.deeplearning4j.rl4j.learning.async;

import org.deeplearning4j.rl4j.learning.async.listener.AsyncTrainingListenerList;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.network.NeuralNet;
import org.deeplearning4j.rl4j.policy.Policy;
import org.deeplearning4j.rl4j.space.Encodable;
import org.deeplearning4j.rl4j.support.*;
import org.deeplearning4j.rl4j.util.IDataManager;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class AsyncThreadTest {

    @Test
    public void when_newEpochStarted_expect_neuralNetworkReset() {
        // Arrange
        TestContext context = new TestContext();
        context.listener.setRemainingEpochStartCallCount(5);

        // Act
        context.sut.run();

        // Assert
        assertEquals(5, context.neuralNet.resetCallCount);
    }

    @Test
    public void when_epochStartedReturnsStop_expect_threadStopped() {
        // Arrange
        TestContext context = new TestContext();
        context.listener.setRemainingEpochStartCallCount(1);

        // Act
        context.sut.run();

        // Assert
        assertEquals(2, context.listener.onEpochStartCallCount);
        assertEquals(1, context.listener.onEpochEndCallCount);
    }

    @Test
    public void when_epochEndReturnsStop_expect_threadStopped() {
        // Arrange
        TestContext context = new TestContext();
        context.listener.setRemainingEpochEndCallCount(1);

        // Act
        context.sut.run();

        // Assert
        assertEquals(2, context.listener.onEpochStartCallCount);
        assertEquals(2, context.listener.onEpochEndCallCount);
    }

    @Test
    public void when_run_expect_preAndPostEpochCalled() {
        // Arrange
        TestContext context = new TestContext();

        // Act
        context.sut.run();

        // Assert
        assertEquals(5, context.sut.preEpochCallCount);
        assertEquals(5, context.sut.postEpochCallCount);
    }

    @Test
    public void when_run_expect_trainSubEpochCalledAndResultPassedToListeners() {
        // Arrange
        TestContext context = new TestContext();

        // Act
        context.sut.run();

        // Assert
        assertEquals(4, context.listener.statEntries.size());
        int[] expectedStepCounter = new int[] { 2, 4, 6, 8 };
        for(int i = 0; i < 4; ++i) {
            IDataManager.StatEntry statEntry = context.listener.statEntries.get(i);
            assertEquals(expectedStepCounter[i], statEntry.getStepCounter());
            assertEquals(i, statEntry.getEpochCounter());
            assertEquals(2.0, statEntry.getReward(), 0.0001);
        }
    }

    private static class TestContext {
        public final MockAsyncGlobal asyncGlobal = new MockAsyncGlobal();
        public final MockNeuralNet neuralNet = new MockNeuralNet();
        public final MockObservationSpace observationSpace = new MockObservationSpace();
        public final MockMDP mdp = new MockMDP(observationSpace);
        public final MockAsyncConfiguration config = new MockAsyncConfiguration(5, 2);
        public final AsyncTrainingListenerList listeners = new AsyncTrainingListenerList();
        public final MockAsyncTrainingListener listener = new MockAsyncTrainingListener();
        public final MockAsyncThread sut = new MockAsyncThread(asyncGlobal, 0, neuralNet, mdp, config, listeners);

        public TestContext() {
            asyncGlobal.setMaxLoops(10);
            listeners.add(listener);
        }
    }

    public static class MockAsyncThread extends AsyncThread {

        public int preEpochCallCount = 0;
        public int postEpochCallCount = 0;


        IAsyncGlobal asyncGlobal;
        private final MockNeuralNet neuralNet;
        private final AsyncConfiguration conf;

        public MockAsyncThread(IAsyncGlobal asyncGlobal, int threadNumber, MockNeuralNet neuralNet, MDP mdp, AsyncConfiguration conf, AsyncTrainingListenerList listeners) {
            super(asyncGlobal, mdp, listeners, threadNumber);

            this.asyncGlobal = asyncGlobal;
            this.neuralNet = neuralNet;
            this.conf = conf;
        }

        @Override
        protected void preEpoch() {
            ++preEpochCallCount;
            super.preEpoch();
        }

        @Override
        protected void postEpoch() {
            ++postEpochCallCount;
            super.postEpoch();
        }

        @Override
        protected NeuralNet getCurrent() {
            return neuralNet;
        }

        @Override
        protected int getThreadNumber() {
            return 0;
        }

        @Override
        protected IAsyncGlobal getAsyncGlobal() {
            return asyncGlobal;
        }

        @Override
        protected AsyncConfiguration getConf() {
            return conf;
        }

        @Override
        protected Policy getPolicy(NeuralNet net) {
            return null;
        }

        @Override
        protected SubEpochReturn trainSubEpoch(Encodable obs, int nstep) {
            return new SubEpochReturn(1, null, 1.0, 1.0);
        }
    }



}

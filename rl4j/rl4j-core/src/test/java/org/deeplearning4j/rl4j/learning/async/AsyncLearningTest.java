package org.deeplearning4j.rl4j.learning.async;

import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.policy.IPolicy;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.support.*;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class AsyncLearningTest {

    @Test
    public void when_training_expect_AsyncGlobalStarted() {
        // Arrange
        TestContext context = new TestContext();
        context.asyncGlobal.setMaxLoops(1);

        // Act
        context.sut.train();

        // Assert
        assertTrue(context.asyncGlobal.hasBeenStarted);
        assertTrue(context.asyncGlobal.hasBeenTerminated);
    }

    @Test
    public void when_trainStartReturnsStop_expect_noTraining() {
        // Arrange
        TestContext context = new TestContext();
        context.listener.canStartTraining = false;
        // Act
        context.sut.train();

        // Assert
        assertEquals(1, context.listener.onTrainingStartCallCount);
        assertEquals(0, context.listener.onTrainingProgressCallCount);
        assertEquals(1, context.listener.onTrainingEndCallCount);
        assertEquals(0, context.policy.playCallCount);
        assertTrue(context.asyncGlobal.hasBeenTerminated);
    }

    @Test
    public void when_progressReturnsStop_expect_trainingStop() {
        // Arrange
        TestContext context = new TestContext();
        context.sut.setProgressEventInterval(1);
        context.listener.setRemainingTrainingProgressCallCount(1);

        // Act
        context.sut.train();

        // Assert
        assertEquals(1, context.listener.onTrainingStartCallCount);
        assertEquals(2, context.listener.onTrainingProgressCallCount);
        assertEquals(1, context.listener.onTrainingEndCallCount);
        assertEquals(2, context.policy.playCallCount);
        assertTrue(context.asyncGlobal.hasBeenTerminated);
    }

    @Test
    public void when_asyncGlobalIsTerminated_expect_trainingStop() {
        // Arrange
        TestContext context = new TestContext();
        context.asyncGlobal.setNumLoopsStopRunning(5);
        context.sut.setProgressEventInterval(1);

        // Act
        context.sut.train();

        // Assert
        assertEquals(1, context.listener.onTrainingStartCallCount);
        assertEquals(4, context.listener.onTrainingProgressCallCount);
        assertEquals(1, context.listener.onTrainingEndCallCount);
        assertEquals(4, context.policy.playCallCount);
        assertTrue(context.asyncGlobal.hasBeenTerminated);
    }

    @Test
    public void when_trainingIsComplete_expect_trainingStop() {
        // Arrange
        TestContext context = new TestContext();
        context.asyncGlobal.setMaxLoops(10);

        // Act
        context.sut.train();

        // Assert
        assertEquals(1, context.listener.onTrainingStartCallCount);
        assertEquals(9, context.listener.onTrainingProgressCallCount);
        assertEquals(1, context.listener.onTrainingEndCallCount);
        assertEquals(9, context.policy.playCallCount);
        assertTrue(context.asyncGlobal.hasBeenTerminated);
    }

    public static class TestContext {
        public MockAsyncConfiguration conf = new MockAsyncConfiguration(20, 10);
        public MockAsyncGlobal asyncGlobal = new MockAsyncGlobal();
        public MockPolicy policy = new MockPolicy();
        public TestAsyncLearning sut = new TestAsyncLearning(conf, asyncGlobal, policy);
        public MockAsyncTrainingListener listener = new MockAsyncTrainingListener();

        public TestContext() {

            sut.setProgressEventInterval(1);
            sut.addListener(listener);
        }
    }

    public static class TestAsyncLearning extends AsyncLearning<MockEncodable, Integer, DiscreteSpace, MockNeuralNet> {
        private final AsyncConfiguration conf;
        private final IAsyncGlobal asyncGlobal;
        private final IPolicy<MockEncodable, Integer> policy;

        public TestAsyncLearning(AsyncConfiguration conf, IAsyncGlobal asyncGlobal, IPolicy<MockEncodable, Integer> policy) {
            super(conf);
            this.conf = conf;
            this.asyncGlobal = asyncGlobal;
            this.policy = policy;
        }

        @Override
        public IPolicy getPolicy() {
            return policy;
        }

        @Override
        public AsyncConfiguration getConfiguration() {
            return conf;
        }

        @Override
        public MDP getMdp() {
            return null;
        }

        @Override
        protected AsyncThread newThread(int i) {
            return null;
        }

        @Override
        protected IAsyncGlobal getAsyncGlobal() {
            return asyncGlobal;
        }

        @Override
        public MockNeuralNet getNeuralNet() {
            return null;
        }
    }

}

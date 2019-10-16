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
        context.listener.setRemainingTrainingStartCallCount(0);
        // Act
        context.sut.train();

        // Assert
        assertEquals(1, context.listener.onTrainingStartCallCount);
        assertEquals(1, context.listener.onTrainingEndCallCount);
        assertEquals(0, context.policy.playCallCount);
        assertTrue(context.asyncGlobal.hasBeenTerminated);
    }

    @Test
    public void when_trainingIsComplete_expect_trainingStop() {
        // Arrange
        TestContext context = new TestContext();

        // Act
        context.sut.train();

        // Assert
        assertEquals(1, context.listener.onTrainingStartCallCount);
        assertEquals(1, context.listener.onTrainingEndCallCount);
        assertTrue(context.asyncGlobal.hasBeenTerminated);
    }

    @Test
    public void when_training_expect_onTrainingProgressCalled() {
        // Arrange
        TestContext context = new TestContext();

        // Act
        context.sut.train();

        // Assert
        assertEquals(1, context.listener.onTrainingProgressCallCount);
    }


    public static class TestContext {
        public final MockAsyncConfiguration conf = new MockAsyncConfiguration(1, 1);
        public final MockAsyncGlobal asyncGlobal = new MockAsyncGlobal();
        public final MockPolicy policy = new MockPolicy();
        public final TestAsyncLearning sut = new TestAsyncLearning(conf, asyncGlobal, policy);
        public final MockTrainingListener listener = new MockTrainingListener();

        public TestContext() {
            sut.addListener(listener);
            asyncGlobal.setMaxLoops(1);
            sut.setProgressMonitorFrequency(1);
        }
    }

    public static class TestAsyncLearning extends AsyncLearning<MockEncodable, Integer, DiscreteSpace, MockNeuralNet> {
        private final AsyncConfiguration conf;
        private final IAsyncGlobal asyncGlobal;
        private final IPolicy<MockEncodable, Integer> policy;

        public TestAsyncLearning(AsyncConfiguration conf, IAsyncGlobal asyncGlobal, IPolicy<MockEncodable, Integer> policy) {
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
        protected AsyncThread newThread(int i, int deviceAffinity) {
            return null;
        }

        @Override
        public MDP getMdp() {
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

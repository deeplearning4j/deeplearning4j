package org.deeplearning4j.rl4j.learning.sync;

import lombok.Getter;
import org.deeplearning4j.rl4j.learning.sync.qlearning.QLearning;
import org.deeplearning4j.rl4j.learning.sync.support.MockStatEntry;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.network.NeuralNet;
import org.deeplearning4j.rl4j.policy.IPolicy;
import org.deeplearning4j.rl4j.support.MockTrainingListener;
import org.deeplearning4j.rl4j.util.IDataManager;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class SyncLearningTest {

    @Test
    public void when_training_expect_listenersToBeCalled() {
        // Arrange
        QLearning.QLConfiguration lconfig = QLearning.QLConfiguration.builder().maxStep(10).build();
        MockTrainingListener listener = new MockTrainingListener();
        MockSyncLearning sut = new MockSyncLearning(lconfig);
        sut.addListener(listener);

        // Act
        sut.train();

        assertEquals(1, listener.onTrainingStartCallCount);
        assertEquals(10, listener.onNewEpochCallCount);
        assertEquals(10, listener.onEpochTrainingResultCallCount);
        assertEquals(1, listener.onTrainingEndCallCount);
    }

    @Test
    public void when_trainingStartCanContinueFalse_expect_trainingStopped() {
        // Arrange
        QLearning.QLConfiguration lconfig = QLearning.QLConfiguration.builder().maxStep(10).build();
        MockTrainingListener listener = new MockTrainingListener();
        MockSyncLearning sut = new MockSyncLearning(lconfig);
        sut.addListener(listener);
        listener.setRemainingTrainingStartCallCount(0);

        // Act
        sut.train();

        assertEquals(1, listener.onTrainingStartCallCount);
        assertEquals(0, listener.onNewEpochCallCount);
        assertEquals(0, listener.onEpochTrainingResultCallCount);
        assertEquals(1, listener.onTrainingEndCallCount);
    }

    @Test
    public void when_newEpochCanContinueFalse_expect_trainingStopped() {
        // Arrange
        QLearning.QLConfiguration lconfig = QLearning.QLConfiguration.builder().maxStep(10).build();
        MockTrainingListener listener = new MockTrainingListener();
        MockSyncLearning sut = new MockSyncLearning(lconfig);
        sut.addListener(listener);
        listener.setRemainingOnNewEpochCallCount(2);

        // Act
        sut.train();

        assertEquals(1, listener.onTrainingStartCallCount);
        assertEquals(3, listener.onNewEpochCallCount);
        assertEquals(2, listener.onEpochTrainingResultCallCount);
        assertEquals(1, listener.onTrainingEndCallCount);
    }

    @Test
    public void when_epochTrainingResultCanContinueFalse_expect_trainingStopped() {
        // Arrange
        QLearning.QLConfiguration lconfig = QLearning.QLConfiguration.builder().maxStep(10).build();
        MockTrainingListener listener = new MockTrainingListener();
        MockSyncLearning sut = new MockSyncLearning(lconfig);
        sut.addListener(listener);
        listener.setRemainingOnEpochTrainingResult(2);

        // Act
        sut.train();

        assertEquals(1, listener.onTrainingStartCallCount);
        assertEquals(3, listener.onNewEpochCallCount);
        assertEquals(3, listener.onEpochTrainingResultCallCount);
        assertEquals(1, listener.onTrainingEndCallCount);
    }

    public static class MockSyncLearning extends SyncLearning {

        private final LConfiguration conf;

        @Getter
        private int currentEpochStep = 0;

        public MockSyncLearning(LConfiguration conf) {
            this.conf = conf;
        }

        @Override
        protected void preEpoch() { currentEpochStep = 0;  }

        @Override
        protected void postEpoch() { }

        @Override
        protected IDataManager.StatEntry trainEpoch() {
            setStepCounter(getStepCounter() + 1);
            return new MockStatEntry(getCurrentEpochStep(), getStepCounter(), 1.0);
        }

        @Override
        public NeuralNet getNeuralNet() {
            return null;
        }

        @Override
        public IPolicy getPolicy() {
            return null;
        }

        @Override
        public LConfiguration getConfiguration() {
            return conf;
        }

        @Override
        public MDP getMdp() {
            return null;
        }
    }
}

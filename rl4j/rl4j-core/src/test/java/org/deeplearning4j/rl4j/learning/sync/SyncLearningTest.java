package org.deeplearning4j.rl4j.learning.sync;

import org.deeplearning4j.rl4j.learning.sync.qlearning.QLearning;
import org.deeplearning4j.rl4j.learning.sync.support.MockStatEntry;
import org.deeplearning4j.rl4j.learning.sync.support.MockSyncTrainingListener;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.network.NeuralNet;
import org.deeplearning4j.rl4j.policy.Policy;
import org.deeplearning4j.rl4j.support.MockDataManager;
import org.deeplearning4j.rl4j.util.DataManagerSyncTrainingListener;
import org.deeplearning4j.rl4j.util.IDataManager;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class SyncLearningTest {

    // Test to help refac -- will be removed
    @Test
    public void refac_checkDataManagerCallsRemainTheSame() {
        // Arrange
        QLearning.QLConfiguration lconfig = QLearning.QLConfiguration.builder().maxStep(10).build();
        MockDataManager dataManager = new MockDataManager(false);
        MockSyncLearning sut = new MockSyncLearning(lconfig, dataManager);

        // Act
        sut.train();

        assertEquals(10, dataManager.statEntries.size());
        for(int i = 0; i < 10; ++i) {
            IDataManager.StatEntry entry = dataManager.statEntries.get(i);
            assertEquals(i, entry.getEpochCounter());
            assertEquals(i+1, entry.getStepCounter());
            assertEquals(1.0, entry.getReward(), 0.0);

        }
        assertEquals(0, dataManager.isSaveDataCallCount);
        assertEquals(0, dataManager.getVideoDirCallCount);
        assertEquals(11, dataManager.writeInfoCallCount);
        assertEquals(1, dataManager.saveCallCount);
    }

    @Test
    public void when_training_expect_listenersToBeCalled() {
        // Arrange
        QLearning.QLConfiguration lconfig = QLearning.QLConfiguration.builder().maxStep(10).build();
        MockSyncTrainingListener listener = new MockSyncTrainingListener();
        MockSyncLearning sut = new MockSyncLearning(lconfig);
        sut.addListener(listener);

        // Act
        sut.train();

        assertEquals(1, listener.onTrainingStartCallCount);
        assertEquals(10, listener.onEpochStartCallCount);
        assertEquals(10, listener.onEpochEndStartCallCount);
        assertEquals(1, listener.onTrainingEndCallCount);
    }

    @Test
    public void when_trainingStartCanContinueFalse_expect_trainingStopped() {
        // Arrange
        QLearning.QLConfiguration lconfig = QLearning.QLConfiguration.builder().maxStep(10).build();
        MockSyncTrainingListener listener = new MockSyncTrainingListener();
        MockSyncLearning sut = new MockSyncLearning(lconfig);
        sut.addListener(listener);
        listener.trainingStartCanContinue = false;

        // Act
        sut.train();

        assertEquals(1, listener.onTrainingStartCallCount);
        assertEquals(0, listener.onEpochStartCallCount);
        assertEquals(0, listener.onEpochEndStartCallCount);
        assertEquals(1, listener.onTrainingEndCallCount);
    }

    @Test
    public void when_epochStartCanContinueFalse_expect_trainingStopped() {
        // Arrange
        QLearning.QLConfiguration lconfig = QLearning.QLConfiguration.builder().maxStep(10).build();
        MockSyncTrainingListener listener = new MockSyncTrainingListener();
        MockSyncLearning sut = new MockSyncLearning(lconfig);
        sut.addListener(listener);
        listener.nbStepsEpochStartCanContinue = 3;

        // Act
        sut.train();

        assertEquals(1, listener.onTrainingStartCallCount);
        assertEquals(3, listener.onEpochStartCallCount);
        assertEquals(2, listener.onEpochEndStartCallCount);
        assertEquals(1, listener.onTrainingEndCallCount);
    }

    @Test
    public void when_epochEndCanContinueFalse_expect_trainingStopped() {
        // Arrange
        QLearning.QLConfiguration lconfig = QLearning.QLConfiguration.builder().maxStep(10).build();
        MockSyncTrainingListener listener = new MockSyncTrainingListener();
        MockSyncLearning sut = new MockSyncLearning(lconfig);
        sut.addListener(listener);
        listener.nbStepsEpochEndCanContinue = 3;

        // Act
        sut.train();

        assertEquals(1, listener.onTrainingStartCallCount);
        assertEquals(3, listener.onEpochStartCallCount);
        assertEquals(3, listener.onEpochEndStartCallCount);
        assertEquals(1, listener.onTrainingEndCallCount);
    }

    public static class MockSyncLearning extends SyncLearning {

        private LConfiguration conf;

        public MockSyncLearning(LConfiguration conf, IDataManager dataManager) {
            super(conf);
            addListener(DataManagerSyncTrainingListener.builder(dataManager).build());
            this.conf = conf;
        }

        public MockSyncLearning(LConfiguration conf) {
            super(conf);
            this.conf = conf;
        }

        @Override
        protected void preEpoch() { }

        @Override
        protected void postEpoch() { }

        @Override
        protected IDataManager.StatEntry trainEpoch() {
            setStepCounter(getStepCounter() + 1);
            return new MockStatEntry(getEpochCounter(), getStepCounter(), 1.0);
        }

        @Override
        public NeuralNet getNeuralNet() {
            return null;
        }

        @Override
        public Policy getPolicy() {
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

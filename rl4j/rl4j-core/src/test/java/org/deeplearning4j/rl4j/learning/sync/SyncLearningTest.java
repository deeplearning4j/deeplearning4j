package org.deeplearning4j.rl4j.learning.sync;

import lombok.AllArgsConstructor;
import lombok.Value;
import org.deeplearning4j.rl4j.learning.ILearning;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.network.NeuralNet;
import org.deeplearning4j.rl4j.policy.Policy;
import org.deeplearning4j.rl4j.support.MockDataManager;
import org.deeplearning4j.rl4j.util.IDataManager;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class SyncLearningTest {

    @Test
    public void refac_checkDataManagerCallsRemainTheSame() {
        // Arrange
        MockLConfiguration lconfig = new MockLConfiguration(10);
        MockDataManager dataManager = new MockDataManager(false);
        MockSyncLearning sut = new MockSyncLearning(lconfig, dataManager, 2);

        // Act
        sut.train();

        assertEquals(10, dataManager.statEntries.size());
        for(int i = 0; i < 10; ++i) {
            IDataManager.StatEntry entry = dataManager.statEntries.get(i);
            assertEquals(2, entry.getEpochCounter());
            assertEquals(i+1, entry.getStepCounter());
            assertEquals(1.0, entry.getReward(), 0.0);

        }
        assertEquals(0, dataManager.isSaveDataCallCount);
        assertEquals(0, dataManager.getVideoDirCallCount);
        assertEquals(11, dataManager.writeInfoCallCount);
        assertEquals(1, dataManager.saveCallCount);
    }

    public static class MockSyncLearning extends SyncLearning {

        private final IDataManager dataManager;
        private LConfiguration conf;
        private final int epochSteps;

        public MockSyncLearning(LConfiguration conf, IDataManager dataManager, int epochSteps) {
            super(conf);
            this.dataManager = dataManager;
            this.conf = conf;
            this.epochSteps = epochSteps;
        }

        @Override
        protected void preEpoch() {

        }

        @Override
        protected void postEpoch() {

        }

        @Override
        protected IDataManager.StatEntry trainEpoch() {
            setStepCounter(getStepCounter() + 1);
            return new MockStatEntry(epochSteps, getStepCounter(), 1.0);
        }

        @Override
        protected IDataManager getDataManager() {
            return dataManager;
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

    public static class MockLConfiguration implements ILearning.LConfiguration {

        private final int maxStep;

        public MockLConfiguration(int maxStep) {
            this.maxStep = maxStep;
        }

        @Override
        public int getSeed() {
            return 0;
        }

        @Override
        public int getMaxEpochStep() {
            return 0;
        }

        @Override
        public int getMaxStep() {
            return maxStep;
        }

        @Override
        public double getGamma() {
            return 0;
        }
    }

    @AllArgsConstructor
    @Value
    public static class MockStatEntry implements IDataManager.StatEntry {
        int epochCounter;
        int stepCounter;
        double reward;
    }
}

package org.deeplearning4j.rl4j.support;

import lombok.Getter;
import org.deeplearning4j.rl4j.learning.sync.SyncLearning;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.policy.Policy;
import org.deeplearning4j.rl4j.util.DataManager;
import org.deeplearning4j.rl4j.util.IDataManager;

import org.deeplearning4j.rl4j.learning.sync.DataManagerSyncLearningEpochListener;

public class TestSyncLearning extends SyncLearning<TestObservation, Integer, TestActionSpace, TestNeuralNet> {

    private final LConfiguration conf;

    @Getter
    private int preEpochCount = 0;

    @Getter
    private int postEpochCount = 0;

    public TestSyncLearning(LConfiguration conf, IDataManager dataManager) {
        super(conf);
        this.conf = conf;
        if(dataManager != null) {
            addEpochListener(new DataManagerSyncLearningEpochListener(dataManager));
        }
    }

    @Override
    protected void preEpoch() {
        ++preEpochCount;
    }

    @Override
    protected void postEpoch() {
        ++postEpochCount;
    }

    @Override
    protected DataManager.StatEntry trainEpoch() {
        setStepCounter(getStepCounter() + 1);
        return new StatEntry();
    }

    @Override
    public TestNeuralNet getNeuralNet() {
        return null;
    }

    @Override
    public Policy<TestObservation, Integer> getPolicy() {
        return null;
    }

    @Override
    public LConfiguration getConfiguration() {
        return conf;
    }

    @Override
    public MDP<TestObservation, Integer, TestActionSpace> getMdp() {
        return null;
    }

    public static class StatEntry implements DataManager.StatEntry {

        @Override
        public int getEpochCounter() {
            return 0;
        }

        @Override
        public int getStepCounter() {
            return 0;
        }

        @Override
        public double getReward() {
            return 0;
        }
    }
}

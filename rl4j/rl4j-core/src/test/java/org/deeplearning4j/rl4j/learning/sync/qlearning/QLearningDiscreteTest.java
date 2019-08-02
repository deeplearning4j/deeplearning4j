package org.deeplearning4j.rl4j.learning.sync.qlearning;

import org.deeplearning4j.rl4j.learning.IHistoryProcessor;
import org.deeplearning4j.rl4j.learning.sync.qlearning.discrete.QLearningDiscrete;
import org.deeplearning4j.rl4j.learning.sync.support.MockDQN;
import org.deeplearning4j.rl4j.learning.sync.support.MockMDP;
import org.deeplearning4j.rl4j.learning.sync.support.MockStatEntry;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.support.MockDataManager;
import org.deeplearning4j.rl4j.support.MockHistoryProcessor;
import org.deeplearning4j.rl4j.util.DataManagerSyncTrainingListener;
import org.deeplearning4j.rl4j.util.IDataManager;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class QLearningDiscreteTest {
    @Test
    public void refac_checkDataManagerCallsRemainTheSame() {
        // Arrange
        QLearning.QLConfiguration lconfig = QLearning.QLConfiguration.builder()
                .maxStep(10)
                .expRepMaxSize(1)
                .build();
        MockDataManager dataManager = new MockDataManager(true);
        MockQLearningDiscrete sut = new MockQLearningDiscrete(10, lconfig, dataManager, 2, 3);
        IHistoryProcessor.Configuration hpConfig = IHistoryProcessor.Configuration.builder()
                .build();
        sut.setHistoryProcessor(new MockHistoryProcessor(hpConfig));

        // Act
        sut.train();

        assertEquals(10, dataManager.statEntries.size());
        for(int i = 0; i < 10; ++i) {
            IDataManager.StatEntry entry = dataManager.statEntries.get(i);
            assertEquals(i, entry.getEpochCounter());
            assertEquals(i+1, entry.getStepCounter());
            assertEquals(1.0, entry.getReward(), 0.0);

        }
        assertEquals(4, dataManager.isSaveDataCallCount);
        assertEquals(4, dataManager.getVideoDirCallCount);
        assertEquals(11, dataManager.writeInfoCallCount);
        assertEquals(5, dataManager.saveCallCount);
    }

    public static class MockQLearningDiscrete extends QLearningDiscrete {

        public MockQLearningDiscrete(int maxSteps, QLConfiguration conf,
                                IDataManager dataManager, int saveFrequency, int monitorFrequency) {
            super(new MockMDP(maxSteps), new MockDQN(), conf, 2);
            addListener(DataManagerSyncTrainingListener.builder(dataManager)
                    .saveFrequency(saveFrequency)
                    .monitorFrequency(monitorFrequency)
                    .build());
        }

        @Override
        protected IDataManager.StatEntry trainEpoch() {
            setStepCounter(getStepCounter() + 1);
            return new MockStatEntry(getEpochCounter(), getStepCounter(), 1.0);
        }
    }
}

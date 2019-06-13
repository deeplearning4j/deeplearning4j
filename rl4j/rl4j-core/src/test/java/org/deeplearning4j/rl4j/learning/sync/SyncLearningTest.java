package org.deeplearning4j.rl4j.learning.sync;

import org.deeplearning4j.rl4j.learning.sync.qlearning.QLearning;
import org.deeplearning4j.rl4j.support.TestDataManager;
import org.deeplearning4j.rl4j.support.TestSyncLearning;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;

import static org.junit.Assert.assertEquals;

public class SyncLearningTest {

    @Rule
    public ExpectedException thrown= ExpectedException.none();

    @Test
    public void SyncLearning_train_WithDataManager() {
        // Arrange
        QLearning.QLConfiguration conf = QLearning.QLConfiguration.builder()
            .maxStep(5)
            .build();
        TestDataManager dataManager = new TestDataManager(false);
        TestSyncLearning sut = new TestSyncLearning(conf, dataManager);

        // Act
        sut.train();

        // Assert
        assertEquals(6, dataManager.getWriteInfoCount());
        assertEquals(5, dataManager.getAppendStatCount());
        assertEquals(1, dataManager.getSaveCount()); // FIXME: Make save freq a parameter and test if it is called as expected

        assertEquals(5, sut.getPreEpochCount());
        assertEquals(5, sut.getPostEpochCount());
    }

    @Test
    public void SyncLearning_train_NoDataManager() {
        // Arrange
        QLearning.QLConfiguration conf = QLearning.QLConfiguration.builder()
                .maxStep(5)
                .build();
        TestSyncLearning sut = new TestSyncLearning(conf, null);

        // Act
        sut.train();

        assertEquals(5, sut.getPreEpochCount());
        assertEquals(5, sut.getPostEpochCount());
    }

}

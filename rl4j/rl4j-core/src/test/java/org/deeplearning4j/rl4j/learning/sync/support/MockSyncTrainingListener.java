package org.deeplearning4j.rl4j.learning.sync.support;

import org.deeplearning4j.rl4j.learning.sync.listener.SyncTrainingEpochEndEvent;
import org.deeplearning4j.rl4j.learning.sync.listener.SyncTrainingEvent;
import org.deeplearning4j.rl4j.learning.sync.listener.SyncTrainingListener;

public class MockSyncTrainingListener implements SyncTrainingListener {

    public int onTrainingStartCallCount = 0;
    public int onTrainingEndCallCount = 0;
    public int onEpochStartCallCount = 0;
    public int onEpochEndStartCallCount = 0;

    public boolean trainingStartCanContinue = true;
    public int nbStepsEpochStartCanContinue = Integer.MAX_VALUE;
    public int nbStepsEpochEndCanContinue = Integer.MAX_VALUE;

    @Override
    public void onTrainingStart(SyncTrainingEvent event) {
        ++onTrainingStartCallCount;
        event.setCanContinue(trainingStartCanContinue);
    }

    @Override
    public void onTrainingEnd() {
        ++onTrainingEndCallCount;
    }

    @Override
    public void onEpochStart(SyncTrainingEvent event) {
        ++onEpochStartCallCount;
        if(onEpochStartCallCount >= nbStepsEpochStartCanContinue) {
            event.setCanContinue(false);
        }
    }

    @Override
    public void onEpochEnd(SyncTrainingEpochEndEvent event) {
        ++onEpochEndStartCallCount;
        if(onEpochEndStartCallCount >= nbStepsEpochEndCanContinue) {
            event.setCanContinue(false);
        }
    }
}

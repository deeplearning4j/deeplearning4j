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
    public ListenerResponse onTrainingStart(SyncTrainingEvent event) {
        ++onTrainingStartCallCount;
        return trainingStartCanContinue ? ListenerResponse.CONTINUE : ListenerResponse.STOP;
    }

    @Override
    public void onTrainingEnd() {
        ++onTrainingEndCallCount;
    }

    @Override
    public ListenerResponse onEpochStart(SyncTrainingEvent event) {
        ++onEpochStartCallCount;
        if(onEpochStartCallCount >= nbStepsEpochStartCanContinue) {
            return ListenerResponse.STOP;
        }
        return ListenerResponse.CONTINUE;
    }

    @Override
    public ListenerResponse onEpochEnd(SyncTrainingEpochEndEvent event) {
        ++onEpochEndStartCallCount;
        if(onEpochEndStartCallCount >= nbStepsEpochEndCanContinue) {
            return ListenerResponse.STOP;
        }
        return ListenerResponse.CONTINUE;
    }
}

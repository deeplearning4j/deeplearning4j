package org.deeplearning4j.rl4j.support;

import lombok.Setter;
import org.deeplearning4j.rl4j.learning.async.listener.AsyncTrainingEpochEndEvent;
import org.deeplearning4j.rl4j.learning.async.listener.AsyncTrainingEpochEvent;
import org.deeplearning4j.rl4j.learning.async.listener.AsyncTrainingEvent;
import org.deeplearning4j.rl4j.learning.async.listener.AsyncTrainingListener;
import org.deeplearning4j.rl4j.util.IDataManager;

import java.util.ArrayList;
import java.util.List;

public class MockAsyncTrainingListener implements AsyncTrainingListener {

    public int onTrainingStartCallCount = 0;
    public int onTrainingProgressCallCount = 0;
    public int onTrainingEndCallCount = 0;
    public int onEpochStartCallCount = 0;
    public int onEpochEndCallCount = 0;

    public boolean canStartTraining = true;

    @Setter
    private int remainingTrainingProgressCallCount = Integer.MAX_VALUE;
    @Setter
    private int remainingEpochStartCallCount = Integer.MAX_VALUE;
    @Setter
    private int remainingEpochEndCallCount = Integer.MAX_VALUE;

    public List<IDataManager.StatEntry> statEntries = new ArrayList<>();


    @Override
    public ListenerResponse onTrainingStart(AsyncTrainingEvent event) {
        ++onTrainingStartCallCount;
        return canStartTraining ? ListenerResponse.CONTINUE : ListenerResponse.STOP;
    }

    @Override
    public ListenerResponse onTrainingProgress(AsyncTrainingEvent event) {
        ++onTrainingProgressCallCount;
        --remainingTrainingProgressCallCount;
        return remainingTrainingProgressCallCount < 0 ? ListenerResponse.STOP : ListenerResponse.CONTINUE;
    }

    @Override
    public void onTrainingEnd() {
        ++onTrainingEndCallCount;
    }

    @Override
    public ListenerResponse onEpochStart(AsyncTrainingEpochEvent event) {
        ++onEpochStartCallCount;
        --remainingEpochStartCallCount;
        return remainingEpochStartCallCount < 0 ? ListenerResponse.STOP : ListenerResponse.CONTINUE;
    }

    @Override
    public ListenerResponse onEpochEnd(AsyncTrainingEpochEndEvent event) {
        ++onEpochEndCallCount;
        --remainingEpochEndCallCount;
        statEntries.add(event.getStatEntry());
        return remainingEpochEndCallCount < 0 ? ListenerResponse.STOP : ListenerResponse.CONTINUE;
    }
}
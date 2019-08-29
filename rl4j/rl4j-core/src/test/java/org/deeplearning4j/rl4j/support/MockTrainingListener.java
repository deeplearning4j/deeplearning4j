package org.deeplearning4j.rl4j.support;

import lombok.Setter;
import org.deeplearning4j.rl4j.learning.listener.*;
import org.deeplearning4j.rl4j.util.IDataManager;

import java.util.ArrayList;
import java.util.List;

public class MockTrainingListener implements TrainingListener {

    public int onTrainingStartCallCount = 0;
    public int onTrainingEndCallCount = 0;
    public int onNewEpochCallCount = 0;
    public int onEpochTrainingResultCallCount = 0;
    public int onTrainingProgressCallCount = 0;

    @Setter
    private int remainingTrainingStartCallCount = Integer.MAX_VALUE;
    @Setter
    private int remainingOnNewEpochCallCount = Integer.MAX_VALUE;
    @Setter
    private int remainingOnEpochTrainingResult = Integer.MAX_VALUE;
    @Setter
    private int remainingonTrainingProgressCallCount = Integer.MAX_VALUE;

    public final List<IDataManager.StatEntry> statEntries = new ArrayList<>();


    @Override
    public ListenerResponse onTrainingStart(ITrainingEvent event) {
        ++onTrainingStartCallCount;
        --remainingTrainingStartCallCount;
        return remainingTrainingStartCallCount < 0 ? ListenerResponse.STOP : ListenerResponse.CONTINUE;
    }

    @Override
    public ListenerResponse onNewEpoch(IEpochTrainingEvent event) {
        ++onNewEpochCallCount;
        --remainingOnNewEpochCallCount;
        return remainingOnNewEpochCallCount < 0 ? ListenerResponse.STOP : ListenerResponse.CONTINUE;
    }

    @Override
    public ListenerResponse onEpochTrainingResult(IEpochTrainingResultEvent event) {
        ++onEpochTrainingResultCallCount;
        --remainingOnEpochTrainingResult;
        statEntries.add(event.getStatEntry());
        return remainingOnEpochTrainingResult < 0 ? ListenerResponse.STOP : ListenerResponse.CONTINUE;
    }

    @Override
    public ListenerResponse onTrainingProgress(ITrainingProgressEvent event) {
        ++onTrainingProgressCallCount;
        --remainingonTrainingProgressCallCount;
        return remainingonTrainingProgressCallCount < 0 ? ListenerResponse.STOP : ListenerResponse.CONTINUE;
    }

    @Override
    public void onTrainingEnd(ITrainingEvent event) {
        ++onTrainingEndCallCount;
    }
}
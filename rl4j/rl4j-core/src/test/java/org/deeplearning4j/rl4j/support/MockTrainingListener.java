package org.deeplearning4j.rl4j.support;

import lombok.Setter;
import org.deeplearning4j.rl4j.learning.IEpochTrainer;
import org.deeplearning4j.rl4j.learning.ILearning;
import org.deeplearning4j.rl4j.learning.listener.*;
import org.deeplearning4j.rl4j.util.IDataManager;

import java.util.ArrayList;
import java.util.List;

public class MockTrainingListener implements TrainingListener {

    private final MockAsyncGlobal asyncGlobal;
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

    public MockTrainingListener() {
        this(null);
    }

    public MockTrainingListener(MockAsyncGlobal asyncGlobal) {
        this.asyncGlobal = asyncGlobal;
    }


    @Override
    public ListenerResponse onTrainingStart() {
        ++onTrainingStartCallCount;
        --remainingTrainingStartCallCount;
        return remainingTrainingStartCallCount < 0 ? ListenerResponse.STOP : ListenerResponse.CONTINUE;
    }

    @Override
    public ListenerResponse onNewEpoch(IEpochTrainer trainer) {
        ++onNewEpochCallCount;
        --remainingOnNewEpochCallCount;
        return remainingOnNewEpochCallCount < 0 ? ListenerResponse.STOP : ListenerResponse.CONTINUE;
    }

    @Override
    public ListenerResponse onEpochTrainingResult(IEpochTrainer trainer, IDataManager.StatEntry statEntry) {
        ++onEpochTrainingResultCallCount;
        --remainingOnEpochTrainingResult;
        statEntries.add(statEntry);
        return remainingOnEpochTrainingResult < 0 ? ListenerResponse.STOP : ListenerResponse.CONTINUE;
    }

    @Override
    public ListenerResponse onTrainingProgress(ILearning learning) {
        ++onTrainingProgressCallCount;
        --remainingonTrainingProgressCallCount;
        if(asyncGlobal != null) {
            asyncGlobal.increaseCurrentLoop();
        }
        return remainingonTrainingProgressCallCount < 0 ? ListenerResponse.STOP : ListenerResponse.CONTINUE;
    }

    @Override
    public void onTrainingEnd() {
        ++onTrainingEndCallCount;
    }
}
package org.deeplearning4j.rl4j.learning.sync.listener;

import lombok.Getter;
import org.deeplearning4j.rl4j.learning.Learning;
import org.deeplearning4j.rl4j.util.IDataManager;

public class SyncTrainingEpochEndEvent extends SyncTrainingEvent {
    @Getter
    private final IDataManager.StatEntry statEntry;

    public SyncTrainingEpochEndEvent(Learning learning, IDataManager.StatEntry statEntry) {
        super(learning);
        this.statEntry = statEntry;
    }
}

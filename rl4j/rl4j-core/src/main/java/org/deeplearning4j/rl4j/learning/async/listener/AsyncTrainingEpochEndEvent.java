package org.deeplearning4j.rl4j.learning.async.listener;

import lombok.Getter;
import org.deeplearning4j.rl4j.learning.async.AsyncThread;
import org.deeplearning4j.rl4j.learning.listener.TrainingEpochEndEvent;
import org.deeplearning4j.rl4j.learning.listener.TrainingListenerList;
import org.deeplearning4j.rl4j.util.IDataManager;

/**
 * The definition of the event sent by {@link TrainingListenerList#notifyEpochFinished(TrainingEpochEndEvent)}
 * in the context of async training
 */
public class AsyncTrainingEpochEndEvent implements TrainingEpochEndEvent {

    /**
     * The source of the event
     */
    @Getter
    private final AsyncThread asyncThread;

    /**
     * The stats of the epoch training
     */
    @Getter
    private final IDataManager.StatEntry statEntry;

    /**
     * @param asyncThread The source of the event
     * @param statEntry The stats of the epoch training
     */
    public AsyncTrainingEpochEndEvent(AsyncThread asyncThread, IDataManager.StatEntry statEntry) {
        this.asyncThread = asyncThread;
        this.statEntry = statEntry;
    }
}

package org.deeplearning4j.rl4j.learning.listener;

import org.deeplearning4j.rl4j.util.IDataManager;

/**
 * The definition of the event sent by {@link TrainingListenerList#notifyEpochFinished(TrainingEpochEndEvent)}
 */
public interface TrainingEpochEndEvent extends TrainingEvent {
    /**
     * @return The {@link org.deeplearning4j.rl4j.util.IDataManager.StatEntry}
     */
    IDataManager.StatEntry getStatEntry();
}

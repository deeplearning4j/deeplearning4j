package org.deeplearning4j.rl4j.learning.sync.listener;

import org.deeplearning4j.rl4j.util.IDataManager;

/**
 * A listener interface to use with a descendant of {@link org.deeplearning4j.rl4j.learning.sync.SyncLearning}
 */
public interface SyncTrainingListener {
    /**
     * Called once when the training starts. The training can be aborted by calling event.setCanContinue(false)
     * @param event
     */
    void onTrainingStart(SyncTrainingEvent event);

    /**
     * Called once when the training has finished. This method called even when the training has been aborted.
     */
    void onTrainingEnd();

    /**
     * Called before the start of every epoch. The training can be aborted by calling event.setCanContinue(false)
     * @param event
     */
    void onEpochStart(SyncTrainingEvent event);

    /**
     * Called after the end of every epoch. The training can be aborted by calling event.setCanContinue(false)
     * @param event
     */
    void onEpochEnd(SyncTrainingEpochEndEvent event);
}

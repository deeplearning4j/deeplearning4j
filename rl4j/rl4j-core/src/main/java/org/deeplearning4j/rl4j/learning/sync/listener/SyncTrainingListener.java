package org.deeplearning4j.rl4j.learning.sync.listener;

/**
 * A listener interface to use with a descendant of {@link org.deeplearning4j.rl4j.learning.sync.SyncLearning}
 */
public interface SyncTrainingListener {

    public enum ListenerResponse {
        /**
         * Tell SyncLearning to continue calling the listeners and the training.
         */
        CONTINUE,

        /**
         * Tell SyncLearning to stop calling the listeners and terminate the training.
         */
        STOP,
    }

    /**
     * Called once when the training starts.
     * @param event
     * @return A ListenerResponse telling the source of the event if it should go on or cancel the training.
     */
    ListenerResponse onTrainingStart(SyncTrainingEvent event);

    /**
     * Called once when the training has finished. This method is called even when the training has been aborted.
     */
    void onTrainingEnd();

    /**
     * Called before the start of every epoch.
     * @param event
     * @return A ListenerResponse telling the source of the event if it should continue or stop the training.
     */
    ListenerResponse onEpochStart(SyncTrainingEvent event);

    /**
     * Called after the end of every epoch.
     * @param event
     * @return A ListenerResponse telling the source of the event if it should continue or stop the training.
     */
    ListenerResponse onEpochEnd(SyncTrainingEpochEndEvent event);
}

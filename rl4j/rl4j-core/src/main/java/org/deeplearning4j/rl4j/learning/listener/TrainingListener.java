package org.deeplearning4j.rl4j.learning.listener;

/**
 * The base definition of all training event listeners
 */
public interface TrainingListener {
    enum ListenerResponse {
        /**
         * Tell the learning process to continue calling the listeners and the training.
         */
        CONTINUE,

        /**
         * Tell the learning process to stop calling the listeners and terminate the training.
         */
        STOP,
    }

    /**
     * Called once when the training starts.
     * @return A ListenerResponse telling the source of the event if it should go on or cancel the training.
     */
    ListenerResponse onTrainingStart(TrainingEvent event);

    /**
     * Called once when the training has finished. This method is called even when the training has been aborted.
     */
    void onTrainingEnd(TrainingEvent event);

    /**
     * Called before the start of every epoch.
     * @return A ListenerResponse telling the source of the event if it should continue or stop the training.
     */
    ListenerResponse onNewEpoch(TrainingEvent event);

    /**
     * Called when an epoch has been completed
     * @param event A EpochTrainingResultEvent containing the results of the epoch training
     * @return A ListenerResponse telling the source of the event if it should continue or stop the training.
     */
    ListenerResponse onEpochTrainingResult(EpochTrainingResultEvent event);

}

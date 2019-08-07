package org.deeplearning4j.rl4j.learning.listener;

/**
 * The base definition of all training event listeners
 * @param <E> The type of the event that will be used by onTrainingStart()
 * @param <ESTART> The type of the event that will be used by onEpochStart()
 * @param <EEND> The type of the event that will be used by onEpochEnd()
 */
public interface TrainingListener<E extends TrainingEvent, ESTART extends TrainingEvent, EEND extends TrainingEpochEndEvent> {
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
     * @param event
     * @return A ListenerResponse telling the source of the event if it should go on or cancel the training.
     */
    ListenerResponse onTrainingStart(E event);

    /**
     * Called once when the training has finished. This method is called even when the training has been aborted.
     */
    void onTrainingEnd();

    /**
     * Called before the start of every epoch.
     * @param event
     * @return A ListenerResponse telling the source of the event if it should continue or stop the training.
     */
    ListenerResponse onEpochStart(ESTART event);

    /**
     * Called after the end of every epoch.
     * @param event
     * @return A ListenerResponse telling the source of the event if it should continue or stop the training.
     */
    ListenerResponse onEpochEnd(EEND event);
}

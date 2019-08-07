package org.deeplearning4j.rl4j.learning.async.listener;

import org.deeplearning4j.rl4j.learning.listener.TrainingListener;
import org.deeplearning4j.rl4j.learning.listener.TrainingListenerList;

/**
 * The base logic to notify async training listeners with the different training events.
 */
public class AsyncTrainingListenerList extends TrainingListenerList<AsyncTrainingEvent, AsyncTrainingEpochEvent, AsyncTrainingEpochEndEvent, AsyncTrainingListener> {

    /**
     * Notify the listeners of the progress at regular intervals. Will stop early if a listener returns {@link org.deeplearning4j.rl4j.learning.listener.TrainingListener.ListenerResponse#STOP}
     * @param event to be passed to the listeners
     * @return whether or not the source training should be stopped
     */
    public boolean notifyTrainingProgress(AsyncTrainingEvent event) {
        for (AsyncTrainingListener listener : listeners) {
            if (listener.onTrainingProgress(event) == TrainingListener.ListenerResponse.STOP) {
                return false;
            }
        }

        return true;
    }

}

package org.deeplearning4j.rl4j.learning.listener;

import java.util.ArrayList;
import java.util.List;

/**
 * The base logic to notify training listeners with the different training events.
 */
public class TrainingListenerList {
    protected final List<TrainingListener> listeners = new ArrayList<>();

    /**
     * Add a listener at the end of the list
     * @param listener The listener to be added
     */
    public void add(TrainingListener listener) {
        listeners.add(listener);
    }

    /**
     * Notify the listeners that the training has started. Will stop early if a listener returns {@link org.deeplearning4j.rl4j.learning.listener.TrainingListener.ListenerResponse#STOP}
     * @return whether or not the source training should be stopped
     */
    public boolean notifyTrainingStarted(TrainingEvent event) {
        for (TrainingListener listener : listeners) {
            if (listener.onTrainingStart(event) == TrainingListener.ListenerResponse.STOP) {
                return false;
            }
        }

        return true;
    }

    /**
     * Notify the listeners that the training has finished.
     */
    public void notifyTrainingFinished(TrainingEvent event) {
        for (TrainingListener listener : listeners) {
            listener.onTrainingEnd(event);
        }
    }

    /**
     * Notify the listeners that a new epoch has started. Will stop early if a listener returns {@link org.deeplearning4j.rl4j.learning.listener.TrainingListener.ListenerResponse#STOP}
     * @return whether or not the source training should be stopped
     */
    public boolean notifyNewEpoch(TrainingEvent event) {
        for (TrainingListener listener : listeners) {
            if (listener.onNewEpoch(event) == TrainingListener.ListenerResponse.STOP) {
                return false;
            }
        }

        return true;
    }

    /**
     * Notify the listeners that an epoch has been completed and the training results are available. Will stop early if a listener returns {@link org.deeplearning4j.rl4j.learning.listener.TrainingListener.ListenerResponse#STOP}
     * @return whether or not the source training should be stopped
     */
    public boolean notifyEpochTrainingResult(EpochTrainingResultEvent event) {
        for (TrainingListener listener : listeners) {
            if (listener.onEpochTrainingResult(event) == TrainingListener.ListenerResponse.STOP) {
                return false;
            }
        }

        return true;
    }

}

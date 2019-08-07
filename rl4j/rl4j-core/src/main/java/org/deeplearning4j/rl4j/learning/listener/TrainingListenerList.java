package org.deeplearning4j.rl4j.learning.listener;

import java.util.ArrayList;
import java.util.List;

/**
 * The base logic to notify training listeners with the different training events.
 * @param <E> The type of the event that will be used by onTrainingStart()
 * @param <ESTART> The type of the event that will be used by onEpochStart()
 * @param <EEND> The type of the event that will be used by onEpochEnd()
 * @param <L> The type of the event listeners
 */
public class TrainingListenerList<E extends TrainingEvent, ESTART extends TrainingEvent, EEND extends TrainingEpochEndEvent, L extends TrainingListener<E, ESTART, EEND>> {
    protected final List<L> listeners = new ArrayList<>();

    /**
     * Add a listener at the end of the list
     * @param listener The listener to be added
     */
    public void add(L listener) {
        listeners.add(listener);
    }

    /**
     * Notify the listeners that the training has started. Will stop early if a listener returns {@link org.deeplearning4j.rl4j.learning.listener.TrainingListener.ListenerResponse#STOP}
     * @param event to be passed to the listeners
     * @return whether or not the source training should be stopped
     */
    public boolean notifyTrainingStarted(E event) {
        for (L listener : listeners) {
            if (listener.onTrainingStart(event) == TrainingListener.ListenerResponse.STOP) {
                return false;
            }
        }

        return true;
    }

    /**
     * Notify the listeners that the training has finished.
     */
    public void notifyTrainingFinished() {
        for (L listener : listeners) {
            listener.onTrainingEnd();
        }
    }

    /**
     * Notify the listeners that a new epoch has started. Will stop early if a listener returns {@link org.deeplearning4j.rl4j.learning.listener.TrainingListener.ListenerResponse#STOP}
     * @param event to be passed to the listeners
     * @return whether or not the source training should be stopped
     */
    public boolean notifyEpochStarted(ESTART event) {
        for (L listener : listeners) {
            if (listener.onEpochStart(event) == TrainingListener.ListenerResponse.STOP) {
                return false;
            }
        }

        return true;
    }

    /**
     * Notify the listeners that an epoch has finished. Will stop early if a listener returns {@link org.deeplearning4j.rl4j.learning.listener.TrainingListener.ListenerResponse#STOP}
     * @param event to be passed to the listeners
     * @return whether or not the source training should be stopped
     */
    public boolean notifyEpochFinished(EEND event) {
        for (L listener : listeners) {
            if (listener.onEpochEnd(event) == TrainingListener.ListenerResponse.STOP) {
                return false;
            }
        }

        return true;
    }
}

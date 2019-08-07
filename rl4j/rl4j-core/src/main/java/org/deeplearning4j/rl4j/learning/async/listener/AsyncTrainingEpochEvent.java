package org.deeplearning4j.rl4j.learning.async.listener;

import lombok.Getter;
import org.deeplearning4j.rl4j.learning.async.AsyncThread;
import org.deeplearning4j.rl4j.learning.listener.TrainingEvent;
import org.deeplearning4j.rl4j.learning.listener.TrainingListenerList;

/**
 * Events that are triggered by AsyncThread <br>
 * (see {@link TrainingListenerList#notifyEpochStarted(TrainingEvent)}
 * and {@link TrainingListenerList#notifyEpochStarted(TrainingEvent)})
 */
public class AsyncTrainingEpochEvent implements TrainingEvent {

    /**
     * The source of the event
     */
    @Getter
    private final AsyncThread asyncThread;

    /**
     * @param asyncThread The source of the event
     */
    public AsyncTrainingEpochEvent(AsyncThread asyncThread) {
        this.asyncThread = asyncThread;
    }
}

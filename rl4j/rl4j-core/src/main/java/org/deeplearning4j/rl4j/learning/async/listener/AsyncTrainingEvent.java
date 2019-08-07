package org.deeplearning4j.rl4j.learning.async.listener;

import lombok.Getter;
import org.deeplearning4j.rl4j.learning.Learning;
import org.deeplearning4j.rl4j.learning.listener.TrainingEvent;
import org.deeplearning4j.rl4j.learning.listener.TrainingListenerList;

/**
 * Events that are triggered by AsyncLearning <br>
 * (see {@link TrainingListenerList#notifyTrainingStarted(TrainingEvent)},
 * {@link AsyncTrainingListenerList#notifyTrainingProgress(AsyncTrainingEvent)} and ({@link TrainingListenerList#notifyTrainingFinished()})
 */
public class AsyncTrainingEvent implements TrainingEvent {

    /**
     * The source of the event
     */
    @Getter
    private final Learning learning;

    /**
     * @param learning The source of the event
     */
    public AsyncTrainingEvent(Learning learning) {
        this.learning = learning;
    }
}

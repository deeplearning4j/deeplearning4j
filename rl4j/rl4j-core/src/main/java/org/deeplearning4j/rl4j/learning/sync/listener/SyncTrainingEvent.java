package org.deeplearning4j.rl4j.learning.sync.listener;

import lombok.Getter;
import lombok.Setter;
import org.deeplearning4j.rl4j.learning.Learning;

/**
 * SyncTrainingEvent are passed as parameters to the events of SyncTrainingListener
 */
public class SyncTrainingEvent {

    /**
     * The source of the event
     */
    @Getter
    private final Learning learning;

    public SyncTrainingEvent(Learning learning) {
        this.learning = learning;
    }
}

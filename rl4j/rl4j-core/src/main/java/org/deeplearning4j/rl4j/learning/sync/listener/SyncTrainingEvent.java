package org.deeplearning4j.rl4j.learning.sync.listener;

import lombok.Getter;
import lombok.Setter;
import org.deeplearning4j.rl4j.learning.Learning;

public class SyncTrainingEvent {

    @Getter
    private final Learning learning;

    @Getter @Setter
    private Boolean canContinue = true;

    public SyncTrainingEvent(Learning learning) {
        this.learning = learning;
    }
}

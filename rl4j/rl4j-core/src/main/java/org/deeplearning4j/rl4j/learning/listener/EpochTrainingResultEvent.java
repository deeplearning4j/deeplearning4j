package org.deeplearning4j.rl4j.learning.listener;

import lombok.*;
import org.deeplearning4j.rl4j.util.IDataManager;

@Data
@AllArgsConstructor
public class EpochTrainingResultEvent extends TrainingEvent {
    private IDataManager.StatEntry statEntry;
    private int epochCount;
    private int stepNum;
}

package org.deeplearning4j.rl4j.learning.listener;

import lombok.AllArgsConstructor;
import lombok.Getter;
import org.deeplearning4j.rl4j.learning.ILearning;

@AllArgsConstructor
public class TrainingProgressEvent implements ITrainingProgressEvent {
    @Getter
    private ILearning learning;
}

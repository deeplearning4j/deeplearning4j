package org.deeplearning4j.rl4j.learning.listener;

import org.deeplearning4j.rl4j.learning.ILearning;

public interface ITrainingProgressEvent extends ITrainingEvent{
    ILearning getLearning();
}

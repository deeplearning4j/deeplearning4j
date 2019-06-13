package org.deeplearning4j.rl4j.learning.async;

import org.deeplearning4j.rl4j.learning.ILearning;

public interface AsyncTrainingListener {
    void onTrainingProgress(ILearning learning);
}

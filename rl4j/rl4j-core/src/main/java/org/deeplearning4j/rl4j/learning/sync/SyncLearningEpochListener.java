package org.deeplearning4j.rl4j.learning.sync;

import org.deeplearning4j.rl4j.learning.ILearning;
import org.deeplearning4j.rl4j.util.DataManager;

public interface SyncLearningEpochListener {

    void onTrainingStarted(ILearning learning);
    void onBeforeEpoch(ILearning learning, int currentEpoch, int currentStep);
    void onAfterEpoch(ILearning learning, DataManager.StatEntry statEntry, int currentEpoch, int currentStep);
}

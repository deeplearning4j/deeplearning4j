package org.deeplearning4j.rl4j.learning.async.listener;

import org.deeplearning4j.rl4j.learning.listener.TrainingListener;

/**
 * The base definition of all async training event listeners
 */
public interface AsyncTrainingListener extends TrainingListener<AsyncTrainingEvent, AsyncTrainingEpochEvent, AsyncTrainingEpochEndEvent> {
    ListenerResponse onTrainingProgress(AsyncTrainingEvent event);
}

package org.deeplearning4j.rl4j.learning.sync.listener;

import org.deeplearning4j.rl4j.learning.listener.TrainingListener;

/**
 * The base definition of all sync training event listeners
 */
public interface SyncTrainingListener extends TrainingListener<SyncTrainingEvent, SyncTrainingEvent, SyncTrainingEpochEndEvent> {
}

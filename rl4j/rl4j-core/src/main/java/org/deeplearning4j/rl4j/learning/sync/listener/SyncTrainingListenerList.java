package org.deeplearning4j.rl4j.learning.sync.listener;

import org.deeplearning4j.rl4j.learning.listener.TrainingListener;
import org.deeplearning4j.rl4j.learning.listener.TrainingListenerList;

/**
 * The base logic to notify sync training listeners with the different training events.
 */
public class SyncTrainingListenerList extends TrainingListenerList<SyncTrainingEvent, SyncTrainingEvent, SyncTrainingEpochEndEvent, TrainingListener<SyncTrainingEvent, SyncTrainingEvent, SyncTrainingEpochEndEvent>> {
}

package org.deeplearning4j.ui.storage.impl;

import lombok.AllArgsConstructor;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.api.storage.StatsStorageEvent;
import org.deeplearning4j.api.storage.StatsStorageListener;
import org.nd4j.linalg.primitives.Pair;

import java.util.Queue;

/**
 * A very simple {@link StatsStorageListener}, that adds the {@link StatsStorageEvent} instances and the specified
 * {@link StatsStorage} instance (i.e., the source) to the specified queue for later processing.
 *
 * @author Alex Black
 */
@AllArgsConstructor
public class QueuePairStatsStorageListener implements StatsStorageListener {

    private final StatsStorage statsStorage;
    private final Queue<Pair<StatsStorage, StatsStorageEvent>> queue;

    @Override
    public void notify(StatsStorageEvent event) {
        queue.add(new Pair<>(statsStorage, event));
    }
}

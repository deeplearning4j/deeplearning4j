package org.deeplearning4j.ui.storage.impl;

import lombok.AllArgsConstructor;
import org.deeplearning4j.api.storage.StatsStorageEvent;
import org.deeplearning4j.api.storage.StatsStorageListener;

import java.util.Queue;

/**
 * A very simple {@link StatsStorageListener}, that adds the {@link StatsStorageEvent} instances to a provided queue
 * for later processing.
 *
 * @author Alex Black
 */
@AllArgsConstructor
public class QueueStatsStorageListener implements StatsStorageListener {

    private final Queue<StatsStorageEvent> queue;

    @Override
    public void notify(StatsStorageEvent event) {
        queue.add(event);
    }
}

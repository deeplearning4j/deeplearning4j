package org.deeplearning4j.api.storage;

import lombok.AllArgsConstructor;
import lombok.Data;

/**
 * StatsStorageEvent: use with {@link StatsStorageListener} to specify when the state of the {@link StatsStorage}
 * implementation changes.<br>
 * Note that depending on the {@link StatsStorageListener.EventType}, some of the
 * field may be null.
 *
 * @author Alex Black
 */
@AllArgsConstructor
@Data
public class StatsStorageEvent {
    private final StatsStorage statsStorage;
    private final StatsStorageListener.EventType eventType;
    private final String sessionID;
    private final String typeID;
    private final String workerID;
    private final long timestamp;
}

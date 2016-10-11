package org.deeplearning4j.api.storage;

import org.deeplearning4j.api.storage.StatsStorage;

/**
 * A listener interface, so that classes can be notified of changes to a {@link StatsStorage}
 * implementation
 *
 * @author Alex Black
 */
public interface StatsStorageListener {

    enum EventType {
        NewSessionID,
        NewTypeID,
        NewWorkerID,
        PostMetaData,
        PostStaticInfo,
        PostUpdate
    }

    void notify(StatsStorageEvent event);

}

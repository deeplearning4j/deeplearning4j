package org.deeplearning4j.ui.storage;

/**
 * A listener interface, so that classes can be notified of changes to a {@link org.deeplearning4j.ui.storage.StatsStorage}
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

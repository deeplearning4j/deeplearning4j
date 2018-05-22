package org.deeplearning4j.api.storage;

/**
 * A listener interface, so that classes can be notified of changes to a {@link StatsStorage}
 * implementation
 *
 * @author Alex Black
 */
public interface StatsStorageListener {

    enum EventType {
        NewSessionID, NewTypeID, NewWorkerID, PostMetaData, PostStaticInfo, PostUpdate
    }

    /**
     * Notify will be called whenever an event (new information posted, etc) occurs.
     * Processing these events should ideally be done asynchronously.
     *
     * @param event    Event that occurred
     */
    void notify(StatsStorageEvent event);

}

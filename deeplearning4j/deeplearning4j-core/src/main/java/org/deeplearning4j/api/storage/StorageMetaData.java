package org.deeplearning4j.api.storage;

import java.io.Serializable;

/**
 * StorageMetaData: contains metadata (such at types, and arbitrary custom serializable data) for storage
 *
 * @author Alex Black
 */
public interface StorageMetaData extends Persistable {

    /**
     * Timestamp for the metadata
     */
    long getTimeStamp();

    /**
     * Session ID for the metadata
     */
    String getSessionID();

    /**
     * Type ID for the metadata
     */
    String getTypeID();

    /**
     * Worker ID for the metadata
     */
    String getWorkerID();

    /**
     * Full class name for the initialization information that will be posted. Is expected to implement {@link Persistable}.
     */
    String getInitTypeClass();

    /**
     * Full class name for the update information that will be posted. Is expected to implement {@link Persistable}.
     */
    String getUpdateTypeClass();

    /**
     * Get extra metadata, if any
     */
    Serializable getExtraMetaData();

}

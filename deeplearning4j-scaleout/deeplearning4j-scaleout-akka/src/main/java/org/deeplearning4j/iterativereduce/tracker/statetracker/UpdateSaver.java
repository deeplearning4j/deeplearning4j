package org.deeplearning4j.iterativereduce.tracker.statetracker;

import java.io.Serializable;

/**
 * Saves an update and handles interfacing
 * with the state tracker for saving and loading
 * updates.
 */
public interface UpdateSaver<E> extends Serializable {
    /**
     * Load the serialized element based on the passed in id
     * This will remove any reference to the specified id
     * and return the object
     * @param id the id of the object to retrieve
     * @return the laoded object
     * @throws Exception
     */
    E load(String id) throws Exception;

    /**
     * Saves the passed in object for later loading
     * @param id the id of the object to save
     * @param save the object to save
     */
    void save(String id,E save) throws Exception;

    /**
     * Cleans up the persistence layer.
     * This will usually be used to clear up left over files from updates
     */
    void cleanup();
}

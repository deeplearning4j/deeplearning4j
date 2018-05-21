package org.nd4j.parameterserver.updater.storage;

import org.nd4j.aeron.ipc.NDArrayMessage;

/**
 * An interface for storing parameter server updates.
 * This is used by an {@link org.nd4j.parameterserver.updater.ParameterServerUpdater}
 * to handle storage of ndarrays
 *
 * @author Adam Gibson
 */
public interface UpdateStorage {

    /**
     * Add an ndarray to the storage
     * @param array the array to add
     */
    void addUpdate(NDArrayMessage array);

    /**
     * The number of updates added
     * to the update storage
     * @return
     */
    int numUpdates();

    /**
     * Clear the array storage
     */
    void clear();

    /**
     * Get the update at the specified index
     * @param index the update to get
     * @return the update at the specified index
     */
    NDArrayMessage getUpdate(int index);

    /**
     * Close the database
     */
    void close();

}

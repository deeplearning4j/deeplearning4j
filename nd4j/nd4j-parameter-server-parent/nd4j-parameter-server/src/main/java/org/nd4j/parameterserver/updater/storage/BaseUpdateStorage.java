package org.nd4j.parameterserver.updater.storage;


import org.nd4j.aeron.ipc.NDArrayMessage;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Base class for common logic in update storage
 *
 * @author Adam Gibson
 */
public abstract class BaseUpdateStorage implements UpdateStorage {
    /**
     * Get the update at the specified index
     *
     * @param index the update to get
     * @return the update at the specified index
     */
    @Override
    public NDArrayMessage getUpdate(int index) {
        if (index >= numUpdates())
            throw new IndexOutOfBoundsException(
                            "Index passed in " + index + " was >= current number of updates " + numUpdates());
        return doGetUpdate(index);
    }

    /**
     * A method for actually performing the implementation
     * of retrieving the ndarray
     * @param index the index of the {@link INDArray} to get
     * @return the ndarray at the specified index
     */
    public abstract NDArrayMessage doGetUpdate(int index);

    /**
     * Close the database
     */
    @Override
    public void close() {
        //default no op
    }
}

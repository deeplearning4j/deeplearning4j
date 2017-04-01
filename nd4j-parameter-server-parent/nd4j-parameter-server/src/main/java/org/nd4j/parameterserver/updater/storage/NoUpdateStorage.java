package org.nd4j.parameterserver.updater.storage;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.aeron.ipc.NDArrayMessage;

import java.util.concurrent.atomic.AtomicInteger;

/**
 * Update storage that only stores update counts
 *
 * @author Adam Gibson
 */
@Slf4j
public class NoUpdateStorage extends BaseUpdateStorage {
    private AtomicInteger updateCount = new AtomicInteger(0);

    /**
     * Add an ndarray to the storage
     *
     * @param array the array to add
     */
    @Override
    public void addUpdate(NDArrayMessage array) {
        log.info("Adding array " + updateCount.get());
        updateCount.incrementAndGet();
    }

    /**
     * The number of updates added
     * to the update storage
     *
     * @return
     */
    @Override
    public int numUpdates() {
        return updateCount.get();
    }

    /**
     * Clear the array storage
     */
    @Override
    public void clear() {
        updateCount.set(0);
    }

    /**
     * A method for actually performing the implementation
     * of retrieving the ndarray
     *
     * @param index the index of the {@link INDArray} to get
     * @return the ndarray at the specified index
     */
    @Override
    public NDArrayMessage doGetUpdate(int index) {
        throw new UnsupportedOperationException("Nothing is being stored in this implementation");
    }
}

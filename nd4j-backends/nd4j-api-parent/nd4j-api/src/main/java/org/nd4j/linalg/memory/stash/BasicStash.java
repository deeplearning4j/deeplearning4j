package org.nd4j.linalg.memory.stash;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * @author raver119@gmail.com
 */
public abstract class BasicStash<T extends Object> implements Stash<T> {
    protected Map<T, INDArray> stash = new ConcurrentHashMap<>();

    protected BasicStash() {

    }

    @Override
    public boolean checkIfExists(T key) {
        /*
            Just checkin'
         */
        return false;
    }

    @Override
    public void put(T key, INDArray object) {
        /*
            Basically we want to get DataBuffer here, and store it here together with shape
            Special case here is GPU: we want to synchronize HOST memory, and store only HOST memory.
         */
    }

    @Override
    public INDArray get(T key) {
        /*
            We want to restore INDArray here, In case of GPU backend - we want to ensure data is replicated to device.
         */
        return null;
    }

    @Override
    public void purge() {
        /*
            We want to purge all stored stuff here.
         */
    }
}

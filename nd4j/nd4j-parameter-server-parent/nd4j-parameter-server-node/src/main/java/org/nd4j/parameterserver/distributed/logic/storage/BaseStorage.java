package org.nd4j.parameterserver.distributed.logic.storage;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.parameterserver.distributed.logic.Storage;

import java.util.concurrent.ConcurrentHashMap;

/**
 * @author raver119@gmail.com
 */
@Slf4j
public abstract class BaseStorage implements Storage {

    private ConcurrentHashMap<Integer, INDArray> storage = new ConcurrentHashMap<>();


    @Override
    public INDArray getArray(@NonNull Integer key) {
        return storage.get(key);
    }

    @Override
    public void setArray(@NonNull Integer key, @NonNull INDArray array) {
        storage.put(key, array);
    }

    @Override
    public boolean arrayExists(@NonNull Integer key) {
        return storage.containsKey(key);
    }

    @Override
    public void shutdown() {
        storage.clear();
    }
}

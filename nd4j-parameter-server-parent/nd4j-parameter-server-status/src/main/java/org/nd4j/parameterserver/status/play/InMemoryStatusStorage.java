package org.nd4j.parameterserver.status.play;


import org.nd4j.parameterserver.model.SubscriberState;

import java.util.HashMap;
import java.util.Map;

/**
 * In memory status storage
 * for parameter server subscribers
 * @author Adam Gibson
 */
public class InMemoryStatusStorage extends BaseStatusStorage {

    /**
     * Create the storage map
     *
     * @return
     */
    @Override
    public Map<Integer, Long> createUpdatedMap() {
        return new HashMap<>();
    }

    @Override
    public Map<Integer, SubscriberState> createMap() {
        return new HashMap<>();
    }
}

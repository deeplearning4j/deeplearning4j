package org.nd4j.parameterserver.status.play;

import io.aeron.driver.MediaDriver;
import org.nd4j.parameterserver.ParameterServerSubscriber;
import org.nd4j.parameterserver.model.SubscriberState;

import java.util.HashMap;
import java.util.Map;

/**
 * In memory status storage
 * for parameter server subscribers
 * @author Adam Gibson
 */
public class InMemoryStatusStorage implements StatusStorage {
    private Map<Integer,SubscriberState> stateMap = new HashMap<>();
    /**
     * Get the state given an id.
     * The integer represents a stream id
     * for a given {@link ParameterServerSubscriber}.
     * <p>
     * A {@link SubscriberState} is supposed to be 1 to 1 mapping
     * for a stream and a {@link MediaDriver}.
     *
     * @param id the id of the state to get
     * @return the subscriber state for the given id or none
     * if it doesn't exist
     */
    @Override
    public SubscriberState getState(int id) {
        if(!stateMap.containsKey(id))
            return SubscriberState.empty();
        return stateMap.get(id);
    }

    /**
     * Update the state for storage
     *
     * @param subscriberState the subscriber state to update
     */
    @Override
    public void updateState(SubscriberState subscriberState) {
        stateMap.put(subscriberState.getStreamId(),subscriberState);
    }
}

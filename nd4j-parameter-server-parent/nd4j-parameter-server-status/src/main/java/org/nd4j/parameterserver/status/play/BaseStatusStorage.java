package org.nd4j.parameterserver.status.play;

import io.aeron.driver.MediaDriver;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.parameterserver.ParameterServerSubscriber;
import org.nd4j.parameterserver.model.SubscriberState;

import java.util.*;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

/**
 * Base status storage for storage logic
 * and scheduling of ejection of
 * instances indicating
 * failure
 *
 * @author Adam Gibson
 */
@Slf4j
public abstract class BaseStatusStorage implements StatusStorage {
    protected Map<Integer, SubscriberState> statusStorageMap = createMap();
    private ScheduledExecutorService executorService;
    protected Map<Integer, Long> updated;
    private long heartBeatEjectionMilliSeconds = 1000;
    private long checkInterval = 1000;

    public BaseStatusStorage() {
        this(1000, 1000);
    }

    /**
     * The list of state ids
     * for the given {@link SubscriberState}
     *
     * @return the list of ids for the given state
     */
    @Override
    public List<Integer> ids() {
        return new ArrayList<>(statusStorageMap.keySet());
    }

    /**
     * Returns the number of states
     * held by this storage
     *
     * @return
     */
    @Override
    public int numStates() {
        return statusStorageMap.size();
    }

    /**
     *
     * @param heartBeatEjectionMilliSeconds the amount of time before
     *                                      ejecting a given subscriber as failed
     * @param checkInterval the interval to check for
     */
    public BaseStatusStorage(long heartBeatEjectionMilliSeconds, long checkInterval) {
        this.heartBeatEjectionMilliSeconds = heartBeatEjectionMilliSeconds;
        this.checkInterval = checkInterval;
        init();
    }


    private void init() {
        updated = createUpdatedMap();
        executorService = Executors.newScheduledThreadPool(1);
        //eject values that haven't checked in in a while
        executorService.scheduleAtFixedRate(new Runnable() {
            @Override
            public void run() {
                long curr = System.currentTimeMillis();
                Set<Integer> remove = new HashSet<>();
                for (Map.Entry<Integer, Long> entry : updated.entrySet()) {
                    long val = entry.getValue();
                    long diff = Math.abs(curr - val);
                    if (diff > heartBeatEjectionMilliSeconds) {
                        remove.add(entry.getKey());
                    }
                }

                if (!remove.isEmpty())
                    log.info("Removing " + remove.size() + " entries");
                //purge removed values
                for (Integer i : remove) {
                    updated.remove(i);
                    statusStorageMap.remove(i);
                }

            }
        }, 30000, checkInterval, TimeUnit.MILLISECONDS);
    }


    /**
     * Create the storage map
     * @return
     */
    public abstract Map<Integer, Long> createUpdatedMap();

    /**
     * Create the storage map
     * @return
     */
    public abstract Map<Integer, SubscriberState> createMap();

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
        if (!statusStorageMap.containsKey(id))
            return SubscriberState.empty();
        return statusStorageMap.get(id);
    }

    /**
     * Update the state for storage
     *
     * @param subscriberState the subscriber state to update
     */
    @Override
    public void updateState(SubscriberState subscriberState) {
        updated.put(subscriberState.getStreamId(), System.currentTimeMillis());
        statusStorageMap.put(subscriberState.getStreamId(), subscriberState);
    }

}

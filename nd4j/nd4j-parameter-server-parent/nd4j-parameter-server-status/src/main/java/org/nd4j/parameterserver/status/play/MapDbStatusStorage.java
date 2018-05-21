package org.nd4j.parameterserver.status.play;

import io.aeron.driver.MediaDriver;
import org.jetbrains.annotations.NotNull;
import org.mapdb.*;
import org.nd4j.parameterserver.ParameterServerSubscriber;
import org.nd4j.parameterserver.model.SubscriberState;

import java.io.File;
import java.io.IOException;
import java.util.Map;

/**
 * MapDB status storage
 *
 * @author Adam Gibson
 */
public class MapDbStatusStorage extends BaseStatusStorage {
    private DB db;
    private File storageFile;

    /**
     * @param heartBeatEjectionMilliSeconds the amount of time before
     *                                      ejecting a given subscriber as failed
     * @param checkInterval                 the interval to check for
     */
    public MapDbStatusStorage(long heartBeatEjectionMilliSeconds, long checkInterval) {
        super(heartBeatEjectionMilliSeconds, checkInterval);
    }

    public MapDbStatusStorage() {
        this(1000, 1000);
    }

    /**
     * Create the storage map
     *
     * @return
     */
    @Override
    public Map<Integer, Long> createUpdatedMap() {
        if (storageFile == null) {
            //In-Memory Stats Storage
            db = DBMaker.memoryDB().make();
        } else {
            db = DBMaker.fileDB(storageFile).closeOnJvmShutdown().transactionEnable() //Default to Write Ahead Log - lower performance, but has crash protection
                            .make();
        }

        updated = db.hashMap("updated").keySerializer(Serializer.INTEGER).valueSerializer(Serializer.LONG)
                        .createOrOpen();
        return updated;
    }



    @Override
    public Map<Integer, SubscriberState> createMap() {
        if (storageFile == null) {
            //In-Memory Stats Storage
            db = DBMaker.memoryDB().make();
        } else {
            db = DBMaker.fileDB(storageFile).closeOnJvmShutdown().transactionEnable() //Default to Write Ahead Log - lower performance, but has crash protection
                            .make();
        }

        statusStorageMap = db.hashMap("statusStorageMap").keySerializer(Serializer.INTEGER)
                        .valueSerializer(new StatusStorageSerializer()).createOrOpen();
        return statusStorageMap;
    }

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



    private class StatusStorageSerializer implements Serializer<SubscriberState> {

        @Override
        public void serialize(@NotNull DataOutput2 out, @NotNull SubscriberState value) throws IOException {
            value.write(out);
        }

        @Override
        public SubscriberState deserialize(@NotNull DataInput2 input, int available) throws IOException {
            return SubscriberState.read(input);
        }

        @Override
        public int compare(SubscriberState p1, SubscriberState p2) {
            return p1.compareTo(p2);
        }
    }
}

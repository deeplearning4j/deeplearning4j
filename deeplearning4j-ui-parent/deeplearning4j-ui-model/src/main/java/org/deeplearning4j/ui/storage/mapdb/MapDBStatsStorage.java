package org.deeplearning4j.ui.storage.mapdb;

import lombok.Data;
import org.deeplearning4j.api.storage.*;
import org.deeplearning4j.ui.storage.BaseCollectionStatsStorage;
import org.jetbrains.annotations.NotNull;
import org.mapdb.*;

import java.io.File;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.List;
import java.util.Map;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

/**
 * An implementation of the {@link StatsStorage} interface, backed by MapDB (in-memory or file).<br>
 * See also {@link org.deeplearning4j.ui.storage.InMemoryStatsStorage} and {@link org.deeplearning4j.ui.storage.FileStatsStorage}
 *
 * @author Alex Black
 */
public class MapDBStatsStorage extends BaseCollectionStatsStorage {

    private static final String COMPOSITE_KEY_HEADER = "&&&";
    private static final String COMPOSITE_KEY_SEPARATOR = "@@@";

    private boolean isClosed = false;
    private DB db;
    private Lock updateMapLock = new ReentrantLock(true);

    private Map<String, Integer> classToInteger; //For storage
    private Map<Integer, String> integerToClass; //For storage
    private Atomic.Integer classCounter;

    public MapDBStatsStorage() {
        this(new Builder());
    }

    public MapDBStatsStorage(File f) {
        this(new Builder().file(f));
    }

    private MapDBStatsStorage(Builder builder) {
        File f = builder.getFile();

        if (f == null) {
            //In-Memory Stats Storage
            db = DBMaker.memoryDB().make();
        } else {
            db = DBMaker.fileDB(f).closeOnJvmShutdown().transactionEnable() //Default to Write Ahead Log - lower performance, but has crash protection
                            .make();
        }

        //Initialize/open the required maps/lists
        sessionIDs = db.hashSet("sessionIDs", Serializer.STRING).createOrOpen();
        storageMetaData = db.hashMap("storageMetaData").keySerializer(new SessionTypeIdSerializer())
                        .valueSerializer(new PersistableSerializer<StorageMetaData>()).createOrOpen();
        staticInfo = db.hashMap("staticInfo").keySerializer(new SessionTypeWorkerIdSerializer())
                        .valueSerializer(new PersistableSerializer<>()).createOrOpen();

        classToInteger = db.hashMap("classToInteger").keySerializer(Serializer.STRING)
                        .valueSerializer(Serializer.INTEGER).createOrOpen();

        integerToClass = db.hashMap("integerToClass").keySerializer(Serializer.INTEGER)
                        .valueSerializer(Serializer.STRING).createOrOpen();

        classCounter = db.atomicInteger("classCounter").createOrOpen();

        //Load up any saved update maps to the update map...
        for (String s : db.getAllNames()) {
            if (s.startsWith(COMPOSITE_KEY_HEADER)) {
                Map<Long, Persistable> m = db.hashMap(s).keySerializer(Serializer.LONG)
                                .valueSerializer(new PersistableSerializer<>()).open();
                String[] arr = s.split(COMPOSITE_KEY_SEPARATOR);
                arr[0] = arr[0].substring(COMPOSITE_KEY_HEADER.length()); //Remove header...
                SessionTypeWorkerId id = new SessionTypeWorkerId(arr[0], arr[1], arr[2]);
                updates.put(id, m);
            }
        }
    }

    @Override
    protected Map<Long, Persistable> getUpdateMap(String sessionID, String typeID, String workerID,
                    boolean createIfRequired) {
        SessionTypeWorkerId id = new SessionTypeWorkerId(sessionID, typeID, workerID);
        if (updates.containsKey(id)) {
            return updates.get(id);
        }
        if (!createIfRequired) {
            return null;
        }
        String compositeKey = COMPOSITE_KEY_HEADER + sessionID + COMPOSITE_KEY_SEPARATOR + typeID
                        + COMPOSITE_KEY_SEPARATOR + workerID;

        Map<Long, Persistable> updateMap;
        updateMapLock.lock();
        try {
            //Try again, in case another thread created it before lock was acquired in this thread
            if (updates.containsKey(id)) {
                return updates.get(id);
            }
            updateMap = db.hashMap(compositeKey).keySerializer(Serializer.LONG)
                            .valueSerializer(new PersistableSerializer<>()).createOrOpen();
            updates.put(id, updateMap);
        } finally {
            updateMapLock.unlock();
        }

        return updateMap;
    }



    @Override
    public void close() {
        db.commit(); //For write ahead log: need to ensure that we persist all data to disk...
        db.close();
        isClosed = true;
    }

    @Override
    public boolean isClosed() {
        return isClosed;
    }

    // ----- Store new info -----

    @Override
    public void putStaticInfo(Persistable staticInfo) {
        List<StatsStorageEvent> sses = checkStorageEvents(staticInfo);
        if (!sessionIDs.contains(staticInfo.getSessionID())) {
            sessionIDs.add(staticInfo.getSessionID());
        }
        SessionTypeWorkerId id = new SessionTypeWorkerId(staticInfo.getSessionID(), staticInfo.getTypeID(),
                        staticInfo.getWorkerID());

        this.staticInfo.put(id, staticInfo);
        db.commit(); //For write ahead log: need to ensure that we persist all data to disk...
        StatsStorageEvent sse = null;
        if (!listeners.isEmpty())
            sse = new StatsStorageEvent(this, StatsStorageListener.EventType.PostStaticInfo, staticInfo.getSessionID(),
                            staticInfo.getTypeID(), staticInfo.getWorkerID(), staticInfo.getTimeStamp());
        for (StatsStorageListener l : listeners) {
            l.notify(sse);
        }

        notifyListeners(sses);
    }

    @Override
    public void putUpdate(Persistable update) {
        List<StatsStorageEvent> sses = checkStorageEvents(update);
        Map<Long, Persistable> updateMap =
                        getUpdateMap(update.getSessionID(), update.getTypeID(), update.getWorkerID(), true);
        updateMap.put(update.getTimeStamp(), update);
        db.commit(); //For write ahead log: need to ensure that we persist all data to disk...

        StatsStorageEvent sse = null;
        if (!listeners.isEmpty())
            sse = new StatsStorageEvent(this, StatsStorageListener.EventType.PostUpdate, update.getSessionID(),
                            update.getTypeID(), update.getWorkerID(), update.getTimeStamp());
        for (StatsStorageListener l : listeners) {
            l.notify(sse);
        }

        notifyListeners(sses);
    }

    @Override
    public void putStorageMetaData(StorageMetaData storageMetaData) {
        List<StatsStorageEvent> sses = checkStorageEvents(storageMetaData);
        SessionTypeId id = new SessionTypeId(storageMetaData.getSessionID(), storageMetaData.getTypeID());
        this.storageMetaData.put(id, storageMetaData);
        db.commit(); //For write ahead log: need to ensure that we persist all data to disk...

        StatsStorageEvent sse = null;
        if (!listeners.isEmpty())
            sse = new StatsStorageEvent(this, StatsStorageListener.EventType.PostMetaData,
                            storageMetaData.getSessionID(), storageMetaData.getTypeID(), storageMetaData.getWorkerID(),
                            storageMetaData.getTimeStamp());
        for (StatsStorageListener l : listeners) {
            l.notify(sse);
        }

        notifyListeners(sses);
    }


    @Data
    public static class Builder {

        private File file;
        private boolean useWriteAheadLog = true;

        public Builder() {
            this(null);
        }

        public Builder(File file) {
            this.file = file;
        }

        public Builder file(File file) {
            this.file = file;
            return this;
        }

        public Builder useWriteAheadLog(boolean useWriteAheadLog) {
            this.useWriteAheadLog = useWriteAheadLog;
            return this;
        }

        public MapDBStatsStorage build() {
            return new MapDBStatsStorage(this);
        }

    }


    private int getIntForClass(Class<?> c) {
        String str = c.getName();
        if (classToInteger.containsKey(str)) {
            return classToInteger.get(str);
        }
        int idx = classCounter.getAndIncrement();
        classToInteger.put(str, idx);
        integerToClass.put(idx, str);
        db.commit();
        return idx;
    }

    private String getClassForInt(int integer) {
        String c = integerToClass.get(integer);
        if (c == null)
            throw new RuntimeException("Unknown class index: " + integer); //Should never happen
        return c;
    }

    //Simple serializer, based on MapDB's SerializerJava
    private static class SessionTypeWorkerIdSerializer implements Serializer<SessionTypeWorkerId> {
        @Override
        public void serialize(@NotNull DataOutput2 out, @NotNull SessionTypeWorkerId value) throws IOException {
            ObjectOutputStream out2 = new ObjectOutputStream(out);
            out2.writeObject(value);
            out2.flush();
        }

        @Override
        public SessionTypeWorkerId deserialize(@NotNull DataInput2 in, int available) throws IOException {
            try {
                ObjectInputStream in2 = new ObjectInputStream(new DataInput2.DataInputToStream(in));
                return (SessionTypeWorkerId) in2.readObject();
            } catch (ClassNotFoundException e) {
                throw new IOException(e);
            }
        }

        @Override
        public int compare(SessionTypeWorkerId w1, SessionTypeWorkerId w2) {
            return w1.compareTo(w2);
        }
    }

    //Simple serializer, based on MapDB's SerializerJava
    private static class SessionTypeIdSerializer implements Serializer<SessionTypeId> {
        @Override
        public void serialize(@NotNull DataOutput2 out, @NotNull SessionTypeId value) throws IOException {
            ObjectOutputStream out2 = new ObjectOutputStream(out);
            out2.writeObject(value);
            out2.flush();
        }

        @Override
        public SessionTypeId deserialize(@NotNull DataInput2 in, int available) throws IOException {
            try {
                ObjectInputStream in2 = new ObjectInputStream(new DataInput2.DataInputToStream(in));
                return (SessionTypeId) in2.readObject();
            } catch (ClassNotFoundException e) {
                throw new IOException(e);
            }
        }

        @Override
        public int compare(SessionTypeId w1, SessionTypeId w2) {
            return w1.compareTo(w2);
        }
    }

    private class PersistableSerializer<T extends Persistable> implements Serializer<T> {

        @Override
        public void serialize(@NotNull DataOutput2 out, @NotNull Persistable value) throws IOException {
            //Persistable values can't be decoded in isolation, i.e., without knowing the type
            //So, we'll first write an integer representing the class name, so we can decode it later...
            int classIdx = getIntForClass(value.getClass());
            out.writeInt(classIdx);
            value.encode(out);
        }

        @Override
        public T deserialize(@NotNull DataInput2 input, int available) throws IOException {
            int classIdx = input.readInt();
            String className = getClassForInt(classIdx);
            Class<?> clazz;
            try {
                clazz = Class.forName(className);
            } catch (ClassNotFoundException e) {
                throw new RuntimeException(e); //Shouldn't normally happen...
            }
            Persistable p;
            try {
                p = (Persistable) clazz.newInstance();
            } catch (InstantiationException | IllegalAccessException e) {
                throw new RuntimeException(e);
            }
            int remainingLength = available - 4; //-4 for int class index
            byte[] temp = new byte[remainingLength];
            input.readFully(temp);
            p.decode(temp);
            return (T) p;
        }

        @Override
        public int compare(Persistable p1, Persistable p2) {
            int c = p1.getSessionID().compareTo(p2.getSessionID());
            if (c != 0)
                return c;
            c = p1.getTypeID().compareTo(p2.getTypeID());
            if (c != 0)
                return c;
            return p1.getWorkerID().compareTo(p2.getWorkerID());
        }
    }

}

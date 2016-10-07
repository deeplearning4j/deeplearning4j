package org.deeplearning4j.ui.storage.mapdb;

import lombok.AllArgsConstructor;
import lombok.Data;
import org.deeplearning4j.ui.storage.StatsStorage;
import org.deeplearning4j.ui.stats.storage.StatsStorageListener;
import org.jetbrains.annotations.NotNull;
import org.mapdb.*;

import java.io.*;
import java.util.*;

/**
 * Created by Alex on 02/10/2016.
 */
public class MapDBStatsStorage implements StatsStorage {

    private static final String COMPOSITE_KEY_HEADER = "&&&";
    private static final String COMPOSITE_KEY_SEPARATOR = "@@@";

    private boolean isClosed = false;
    private DB db;

    private Set<String> sessionIDs;
    private Map<String, SessionMetaData> sessionMetaData;
    private Map<SessionTypeWorkerId, byte[]> staticInfo;

    private Map<SessionTypeWorkerId, Map<Long, byte[]>> updates = new HashMap<>();

    private List<StatsStorageListener> listeners = new ArrayList<>();

    public MapDBStatsStorage() {
        this(new Builder());
    }

    public MapDBStatsStorage(Builder builder) {
        File f = builder.getFile();

        if (f == null) {
            //In-Memory Stats Storage
            db = DBMaker
                    .memoryDB()
                    .make();
        } else {
            db = DBMaker
                    .fileDB(f)
                    .closeOnJvmShutdown()
                    .transactionEnable()    //Default to Write Ahead Log - lower performance, but has crash protection
                    .make();
        }

        //Initialize/open the required maps/lists
        sessionIDs = db.hashSet("sessionIDs", Serializer.STRING).createOrOpen();
        sessionMetaData = db.hashMap("sessionMetaData")
                .keySerializer(Serializer.STRING)
                .valueSerializer(new SessionMetaDataSerializer())
                .createOrOpen();
        staticInfo = db.hashMap("staticInfo")
                .keySerializer(new SessionWorkerIdSerializer())
                .valueSerializer(Serializer.BYTE_ARRAY)
                .createOrOpen();

        //Load up any saved update maps to the update map...
        for(String s : db.getAllNames()){
            if(s.startsWith(COMPOSITE_KEY_HEADER)){
                Map<Long,byte[]> m = db.hashMap(s)
                        .keySerializer(Serializer.LONG)
                        .valueSerializer(Serializer.BYTE_ARRAY)
                        .open();
                String[] arr = s.split(COMPOSITE_KEY_SEPARATOR);
                arr[0] = arr[0].substring(COMPOSITE_KEY_HEADER.length());   //Remove header...
                SessionTypeWorkerId id = new SessionTypeWorkerId(arr[0], arr[1]);
                updates.put(id, m);
            }
        }
    }

    private synchronized Map<Long, byte[]> getUpdateMap(String sessionID, String workerID, boolean createIfRequired) {
        SessionTypeWorkerId id = new SessionTypeWorkerId(sessionID, workerID);
        if (updates.containsKey(id)) {
            return updates.get(id);
        }
        if(!createIfRequired){
            return null;
        }
        String compositeKey = COMPOSITE_KEY_HEADER + sessionID + COMPOSITE_KEY_SEPARATOR + workerID;
        Map<Long, byte[]> updateMap = db.hashMap(compositeKey)
                .keySerializer(Serializer.LONG)
                .valueSerializer(Serializer.BYTE_ARRAY)
                .createOrOpen();
        updates.put(id, updateMap);
        return updateMap;
    }

    private void logIDs(String sessionId, String workerID) {
        if (!sessionIDs.contains(sessionId)) {
            sessionIDs.add(sessionId);
            for (StatsStorageListener l : listeners) {
                l.notifyNewSession(sessionId);
                l.notifyNewWorkerID(sessionId, workerID);   //Must also be a new worker ID...
            }
        } else {
            SessionTypeWorkerId id = new SessionTypeWorkerId(sessionId, workerID);
            if (getUpdateMap(sessionId,workerID,false) == null && !staticInfo.containsKey(id)) {
                for (StatsStorageListener l : listeners) {
                    l.notifyNewWorkerID(sessionId, workerID);
                }
            }
        }
    }

    @Override
    public void close() {
        db.commit();    //For write ahead log: need to ensure that we persist all data to disk...
        db.close();
        isClosed = true;
    }

    @Override
    public boolean isClosed() {
        return isClosed;
    }

    @Override
    public List<String> listSessionIDs() {
        return new ArrayList<>(sessionIDs);
    }

    @Override
    public boolean sessionExists(String sessionID) {
        return sessionIDs.contains(sessionID);
    }

    @Override
    public byte[] getStaticInfo(String sessionID, String workerID) {
        SessionTypeWorkerId id = new SessionTypeWorkerId(sessionID, workerID);
        return staticInfo.get(id);
    }

    @Override
    public List<String> listWorkerIDsForSession(String sessionID) {
        List<String> out = new ArrayList<>();
        for (SessionTypeWorkerId ids : staticInfo.keySet()) {
            if (sessionID.equals(ids.getSessionID())) {
                out.add(ids.getWorkerID());
            }
        }
        return out;
    }

    @Override
    public int getNumUpdateRecordsFor(String sessionID) {
        int count = 0;
        for( SessionTypeWorkerId id : updates.keySet() ){
            if(sessionID.equals(id.getSessionID())){
                Map<Long,byte[]> map = updates.get(id);
                if(map != null) count += map.size();
            }
        }
        return count;
    }

    @Override
    public int getNumUpdateRecordsFor(String sessionID, String workerID) {
        SessionTypeWorkerId id = new SessionTypeWorkerId(sessionID, workerID);
        Map<Long,byte[]> map = updates.get(id);
        if(map != null) return map.size();
        return 0;
    }

    @Override
    public UpdateRecord getLatestUpdate(String sessionID, String workerID) {
        SessionTypeWorkerId id = new SessionTypeWorkerId(sessionID, workerID);
        Map<Long,byte[]> map = updates.get(id);
        if(map == null) return null;
        long max = Long.MIN_VALUE;
        for(Long l : map.keySet()){
            max = Math.max(max, l);
        }
        return new UpdateRecord(sessionID, workerID, max, map.get(max));
    }

    @Override
    public UpdateRecord getUpdate(String sessionID, String workerID, long timestamp) {
        SessionTypeWorkerId id = new SessionTypeWorkerId(sessionID, workerID);
        Map<Long,byte[]> map = updates.get(id);
        if(map == null) return null;

        return new UpdateRecord(sessionID, workerID, timestamp, map.get(timestamp));
    }

    @Override
    public List<UpdateRecord> getLatestUpdateAllWorkers(String sessionID) {
        List<UpdateRecord> list = new ArrayList<>();

        for( SessionTypeWorkerId id : updates.keySet() ){
            if(sessionID.equals(id.getSessionID())){
                UpdateRecord r = getLatestUpdate(sessionID, id.workerID);
                if(r != null){
                    list.add(r);
                }
            }
        }

        return list;
    }

    @Override
    public List<UpdateRecord> getAllUpdatesAfter(String sessionID, String workerID, long timestamp) {
        List<UpdateRecord> list = new ArrayList<>();

        Map<Long,byte[]> map = getUpdateMap(sessionID, workerID, false);
        if(map == null) return list;

        for(Long time : map.keySet()){
            if(time > timestamp){
                list.add(new UpdateRecord(sessionID, workerID, time, map.get(time)));
            }
        }

        Collections.sort(list);

        return list;
    }

    @Override
    public SessionMetaData getSessionMetaData(String sessionID) {
        return this.sessionMetaData.get(sessionID);
    }

    // ----- Store new info -----

    @Override
    public void putStaticInfo(String sessionID, String workerID, byte[] staticInfo) {
        logIDs(sessionID, workerID);
        SessionTypeWorkerId id = new SessionTypeWorkerId(sessionID, workerID);

        this.staticInfo.put(id, staticInfo);
        db.commit();    //For write ahead log: need to ensure that we persist all data to disk...

        for (StatsStorageListener l : listeners) {
            l.notifyStaticInfo(sessionID, workerID);
        }
    }

    @Override
    public void putUpdate(String sessionID, String workerID, long timestamp, byte[] update) {
        logIDs(sessionID, workerID);

        Map<Long, byte[]> updateMap = getUpdateMap(sessionID, workerID, true);
        updateMap.put(timestamp, update);
        db.commit();    //For write ahead log: need to ensure that we persist all data to disk...

        for (StatsStorageListener l : listeners) {
            l.notifyStatusUpdate(sessionID, workerID, timestamp);
        }
    }

    @Override
    public void putSessionMetaData(String sessionID, String staticInfoClass, String updateClass, Serializable otherMetaData) {
        this.sessionMetaData.put(sessionID, new SessionMetaData(sessionID, staticInfoClass, updateClass, otherMetaData));
        for(StatsStorageListener l : listeners){
            l.notifySessionMetaData(sessionID);
        }
    }


    // ----- Listeners -----

    @Override
    public void registerStatsStorageListener(StatsStorageListener listener) {
        if (!this.listeners.contains(listener)) {
            this.listeners.add(listener);
        }
    }

    @Override
    public void deregisterStatsStorageListener(StatsStorageListener listener) {
        this.listeners.remove(listener);
    }

    @Override
    public void removeAllListeners() {
        this.listeners.clear();
    }

    @Override
    public List<StatsStorageListener> getListeners() {
        return new ArrayList<>(listeners);
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

    @AllArgsConstructor
    @Data
    public static class SessionTypeWorkerId implements Serializable, Comparable<SessionTypeWorkerId> {
        private final String sessionID;
        private final String typeID;
        private final String workerID;

        @Override
        public int compareTo(SessionTypeWorkerId o) {
            int c = sessionID.compareTo(o.sessionID);
            if (c != 0) return c;
            c = typeID.compareTo(o.typeID);
            if (c != 0) return c;
            return workerID.compareTo(workerID);
        }

        @Override
        public String toString(){
            return "(" + sessionID + "," + typeID + "," + workerID + ")";
        }
    }

    //Simple serializer, based on MapDB's SerializerJava
    private static class SessionWorkerIdSerializer implements Serializer<SessionTypeWorkerId> {
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
        public int compare(SessionTypeWorkerId w1, SessionTypeWorkerId w2){
            return w1.compareTo(w2);
        }
    }

    private static class SessionMetaDataSerializer implements Serializer<SessionMetaData>{

        @Override
        public void serialize(@NotNull DataOutput2 out, @NotNull SessionMetaData value) throws IOException {
            ObjectOutputStream out2 = new ObjectOutputStream(out);
            out2.writeObject(value);
            out2.flush();
        }

        @Override
        public SessionMetaData deserialize(@NotNull DataInput2 in, int available) throws IOException {
            try {
                ObjectInputStream in2 = new ObjectInputStream(new DataInput2.DataInputToStream(in));
                return (SessionMetaData) in2.readObject();
            } catch (ClassNotFoundException e) {
                throw new IOException(e);
            }
        }

        @Override
        public int compare(SessionMetaData m1, SessionMetaData m2){
            return m1.compareTo(m2);
        }
    }
}

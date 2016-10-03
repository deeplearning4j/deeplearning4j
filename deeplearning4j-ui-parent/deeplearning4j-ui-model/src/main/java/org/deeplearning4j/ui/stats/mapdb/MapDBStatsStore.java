package org.deeplearning4j.ui.stats.mapdb;

import lombok.AllArgsConstructor;
import lombok.Data;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.ui.stats.storage.StatsStorage;
import org.deeplearning4j.ui.stats.storage.StatsStorageListener;
import org.jetbrains.annotations.NotNull;
import org.mapdb.*;

import java.io.*;
import java.util.*;

/**
 * Created by Alex on 02/10/2016.
 */
public class MapDBStatsStore implements StatsStorage {

    private DB db;

    private Set<String> sessionIDs;
    private Map<SessionWorkerId, byte[]> staticInfo;

    private Map<SessionWorkerId, Map<Long, byte[]>> updates = new HashMap<>();

    private List<StatsStorageListener> listeners = new ArrayList<>();

    public MapDBStatsStore() {
        this(new Builder());
    }

    public MapDBStatsStore(Builder builder) {
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
        staticInfo = db.hashMap("staticInfo")
                .keySerializer(new SessionWorkerIdSerializer())
                .valueSerializer(Serializer.BYTE_ARRAY)
                .createOrOpen();


    }

    private synchronized Map<Long, byte[]> getUpdateMap(String sessionID, String workerID, boolean createIfRequired) {
        SessionWorkerId id = new SessionWorkerId(sessionID, workerID);
        if (updates.containsKey(id)) {
            return updates.get(id);
        }
        if(!createIfRequired){
            return null;
        }
        String compositeKey = "sID_" + sessionID + "-wID_" + workerID;
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
            SessionWorkerId id = new SessionWorkerId(sessionId, workerID);
            if (!updates.containsKey(id)) {
                for (StatsStorageListener l : listeners) {
                    l.notifyNewWorkerID(sessionId, workerID);
                }
            }
        }
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
        SessionWorkerId id = new SessionWorkerId(sessionID, workerID);
        return staticInfo.get(id);
    }

    @Override
    public List<String> listWorkerIDsForSession(String sessionID) {
        List<String> out = new ArrayList<>();
        for (SessionWorkerId ids : staticInfo.keySet()) {
            if (sessionID.equals(ids.getSessionID())) {
                out.add(ids.getWorkerID());
            }
        }
        return out;
    }

    @Override
    public int getNumUpdateRecordsFor(String sessionID) {
        int count = 0;
        for( SessionWorkerId id : updates.keySet() ){
            if(sessionID.equals(id.getSessionID())){
                Map<Long,byte[]> map = updates.get(id);
                if(map != null) count += map.size();
            }
        }
        return count;
    }

    @Override
    public int getNumUpdateRecordsFor(String sessionID, String workerID) {
        SessionWorkerId id = new SessionWorkerId(sessionID, workerID);
        Map<Long,byte[]> map = updates.get(id);
        if(map != null) return map.size();
        return 0;
    }

    @Override
    public Pair<Long, byte[]> getLatestUpdate(String sessionID, String workerID) {
        SessionWorkerId id = new SessionWorkerId(sessionID, workerID);
        Map<Long,byte[]> map = updates.get(id);
        if(map == null) return null;
        long max = Long.MIN_VALUE;
        for(Long l : map.keySet()){
            max = Math.max(max, l);
        }
        return new Pair<>(max,map.get(max));
    }

    @Override
    public byte[] getUpdate(String sessionID, String workerID, long timestamp) {
        SessionWorkerId id = new SessionWorkerId(sessionID, workerID);
        Map<Long,byte[]> map = updates.get(id);
        if(map == null) return null;
        return map.get(timestamp);
    }

    @Override
    public List<UpdateRecord> getLatestUpdateAllWorkers(String sessionID) {
        List<UpdateRecord> list = new ArrayList<>();

        for( SessionWorkerId id : updates.keySet() ){
            if(sessionID.equals(id.getSessionID())){
                Pair<Long,byte[]> p = getLatestUpdate(sessionID, id.workerID);
                if(p != null){
                    list.add(new UpdateRecord(sessionID, id.workerID, p.getFirst(), p.getSecond()));
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

        return list;
    }

    // ----- Store new info -----

    @Override
    public void putStaticInfo(String sessionID, String workerID, byte[] staticInfo) {
        logIDs(sessionID, workerID);
        SessionWorkerId id = new SessionWorkerId(sessionID, workerID);

        this.staticInfo.put(id, staticInfo);

        for (StatsStorageListener l : listeners) {
            l.notifyStaticInfo(sessionID, workerID);
        }
    }

    @Override
    public void putUpdate(String sessionID, String workerID, long timestamp, byte[] update) {
        logIDs(sessionID, workerID);

        Map<Long, byte[]> updateMap = getUpdateMap(sessionID, workerID, true);
        updateMap.put(timestamp, update);

        for (StatsStorageListener l : listeners) {
            l.notifyStatusUpdate(sessionID, workerID, timestamp);
        }
    }


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

        public MapDBStatsStore build() {
            return new MapDBStatsStore(this);
        }

    }

    @AllArgsConstructor
    @Data
    public static class SessionWorkerId implements Serializable, Comparable<SessionWorkerId> {
        private final String sessionID;
        private final String workerID;

        @Override
        public int compareTo(SessionWorkerId o) {
            int c = sessionID.compareTo(o.sessionID);
            if (c != 0) return c;
            return workerID.compareTo(workerID);
        }
    }

    //Simple serializer, based on MapDB's SerializerJava
    private static class SessionWorkerIdSerializer implements Serializer<SessionWorkerId> {
        @Override
        public void serialize(@NotNull DataOutput2 out, @NotNull SessionWorkerId value) throws IOException {
            ObjectOutputStream out2 = new ObjectOutputStream((OutputStream) out);
            out2.writeObject(value);
            out2.flush();
        }

        @Override
        public SessionWorkerId deserialize(@NotNull DataInput2 in, int available) throws IOException {
            try {
                ObjectInputStream in2 = new ObjectInputStream(new DataInput2.DataInputToStream(in));
                return (SessionWorkerId) in2.readObject();
            } catch (ClassNotFoundException e) {
                throw new IOException(e);
            }
        }

        @Override
        public int compare(SessionWorkerId w1, SessionWorkerId w2){
            return w1.compareTo(w2);
        }
    }
}

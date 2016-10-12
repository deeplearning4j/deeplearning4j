package org.deeplearning4j.ui.storage.mapdb;

import lombok.AllArgsConstructor;
import lombok.Data;
import org.deeplearning4j.api.storage.*;
import org.deeplearning4j.ui.stats.impl.SbeUtil;
import org.jetbrains.annotations.NotNull;
import org.mapdb.*;

import java.io.*;
import java.util.*;

/**
 * An implementation of the {@link StatsStorage} interface, backed by MapDB
 *
 * @author Alex Black
 */
public class MapDBStatsStorage implements StatsStorage {

    private static final String COMPOSITE_KEY_HEADER = "&&&";
    private static final String COMPOSITE_KEY_SEPARATOR = "@@@";

    private boolean isClosed = false;
    private DB db;

    private Set<String> sessionIDs;
    private Map<SessionTypeId, StorageMetaData> storageMetaData;
    private Map<SessionTypeWorkerId, Persistable> staticInfo;

    private Map<SessionTypeWorkerId, Map<Long, Persistable>> updates = new HashMap<>();

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
        storageMetaData = db.hashMap("storageMetaData")
                .keySerializer(new SessionTypeIdSerializer())
                .valueSerializer(new PersistableSerializer<StorageMetaData>())
                .createOrOpen();
        staticInfo = db.hashMap("staticInfo")
                .keySerializer(new SessionTypeWorkerIdSerializer())
                .valueSerializer(new PersistableSerializer<>())
                .createOrOpen();

        //Load up any saved update maps to the update map...
        for (String s : db.getAllNames()) {
            if (s.startsWith(COMPOSITE_KEY_HEADER)) {
                Map<Long, Persistable> m = db.hashMap(s)
                        .keySerializer(Serializer.LONG)
                        .valueSerializer(new PersistableSerializer<>())
                        .open();
                String[] arr = s.split(COMPOSITE_KEY_SEPARATOR);
                arr[0] = arr[0].substring(COMPOSITE_KEY_HEADER.length());   //Remove header...
                SessionTypeWorkerId id = new SessionTypeWorkerId(arr[0], arr[1], arr[2]);
                updates.put(id, m);
            }
        }
    }

    private synchronized Map<Long, Persistable> getUpdateMap(String sessionID, String typeID, String workerID, boolean createIfRequired) {
        SessionTypeWorkerId id = new SessionTypeWorkerId(sessionID, typeID, workerID);
        if (updates.containsKey(id)) {
            return updates.get(id);
        }
        if (!createIfRequired) {
            return null;
        }
        String compositeKey = COMPOSITE_KEY_HEADER + sessionID + COMPOSITE_KEY_SEPARATOR + typeID + COMPOSITE_KEY_SEPARATOR + workerID;
        Map<Long, Persistable> updateMap = db.hashMap(compositeKey)
                .keySerializer(Serializer.LONG)
                .valueSerializer(new PersistableSerializer<>())
                .createOrOpen();
        updates.put(id, updateMap);
        return updateMap;
    }

    //Return any relevant storage events
    //We want to return these so they can be logged later. Can't be logged immediately, as this may case a race
    //condition with whatever is receiving the events: i.e., might get the event before the contents are actually
    //available in the DB
    private List<StatsStorageEvent> checkStorageEvents(Persistable p) {
        if (listeners.size() == 0) return null;

        int count = 0;
        StatsStorageEvent newSID = null;
        StatsStorageEvent newTID = null;
        StatsStorageEvent newWID = null;

        //Is this a new session ID?
        if (!sessionIDs.contains(p.getSessionID())) {
            newSID = new StatsStorageEvent(StatsStorageListener.EventType.NewSessionID,
                    p.getSessionID(), p.getTypeID(), p.getWorkerID(), p.getTimeStamp());
            count++;
        }

        //Check for new type and worker IDs
        //TODO probably more efficient way to do this
        boolean foundTypeId = false;
        boolean foundWorkerId = false;
        String typeId = p.getTypeID();
        String wid = p.getWorkerID();
        for (SessionTypeId ts : storageMetaData.keySet()) {
            if (typeId.equals(ts.getTypeID())) {
                foundTypeId = true;
                break;
            }
        }
        for (SessionTypeWorkerId stw : staticInfo.keySet()) {
            if (!foundTypeId && typeId.equals(stw.getTypeID())) {
                foundTypeId = true;
            }
            if (!foundWorkerId && wid.equals(stw.getWorkerID())) {
                foundWorkerId = true;
            }
            if (foundTypeId && foundWorkerId) break;
        }
        if (!foundTypeId || !foundWorkerId) {
            for (SessionTypeWorkerId stw : updates.keySet()) {
                if (!foundTypeId && typeId.equals(stw.getTypeID())) {
                    foundTypeId = true;
                }
                if (!foundWorkerId && wid.equals(stw.getWorkerID())) {
                    foundWorkerId = true;
                }
                if (foundTypeId && foundWorkerId) break;
            }
        }
        if (!foundTypeId) {
            //New type ID
            newTID = new StatsStorageEvent(StatsStorageListener.EventType.NewTypeID,
                    p.getSessionID(), p.getTypeID(), p.getWorkerID(), p.getTimeStamp());
            count++;
        }
        if (!foundWorkerId) {
            //New worker ID
            newWID = new StatsStorageEvent(StatsStorageListener.EventType.NewWorkerID,
                    p.getSessionID(), p.getTypeID(), p.getWorkerID(), p.getTimeStamp());
            count++;
        }
        if (count == 0) return null;
        List<StatsStorageEvent> sses = new ArrayList<>(count);
        if (newSID != null) sses.add(newSID);
        if (newTID != null) sses.add(newTID);
        if (newWID != null) sses.add(newWID);
        return sses;
    }

    private void notifyListeners(List<StatsStorageEvent> sses) {
        if (sses == null || sses.size() == 0 || listeners.size() == 0) return;
        for (StatsStorageListener l : listeners) {
            for (StatsStorageEvent e : sses) {
                l.notify(e);
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
    public Persistable getStaticInfo(String sessionID, String typeID, String workerID) {
        SessionTypeWorkerId id = new SessionTypeWorkerId(sessionID, typeID, workerID);
        return staticInfo.get(id);
    }

    @Override
    public List<String> listTypeIDsForSession(String sessionID) {
        Set<String> typeIDs = new HashSet<>();
        for (SessionTypeId st : storageMetaData.keySet()) {
            if (!sessionID.equals(st.getSessionID())) continue;
            typeIDs.add(st.getTypeID());
        }

        for (SessionTypeWorkerId stw : staticInfo.keySet()) {
            if (!sessionID.equals(stw.getSessionID())) continue;
            typeIDs.add(stw.getTypeID());
        }
        for (SessionTypeWorkerId stw : updates.keySet()) {
            if (!sessionID.equals(stw.getSessionID())) continue;
            typeIDs.add(stw.getTypeID());
        }

        return new ArrayList<>(typeIDs);
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
        for (SessionTypeWorkerId id : updates.keySet()) {
            if (sessionID.equals(id.getSessionID())) {
                Map<Long, Persistable> map = updates.get(id);
                if (map != null) count += map.size();
            }
        }
        return count;
    }

    @Override
    public int getNumUpdateRecordsFor(String sessionID, String typeID, String workerID) {
        SessionTypeWorkerId id = new SessionTypeWorkerId(sessionID, typeID, workerID);
        Map<Long, Persistable> map = updates.get(id);
        if (map != null) return map.size();
        return 0;
    }

    @Override
    public Persistable getLatestUpdate(String sessionID, String typeID, String workerID) {
        SessionTypeWorkerId id = new SessionTypeWorkerId(sessionID, typeID, workerID);
        Map<Long, Persistable> map = updates.get(id);
        if (map == null || map.isEmpty()) return null;
        long maxTime = Long.MIN_VALUE;
        for (Long l : map.keySet()) {
            maxTime = Math.max(maxTime, l);
        }
        return map.get(maxTime);
    }

    @Override
    public Persistable getUpdate(String sessionID, String typeID, String workerID, long timestamp) {
        SessionTypeWorkerId id = new SessionTypeWorkerId(sessionID, typeID, workerID);
        Map<Long, Persistable> map = updates.get(id);
        if (map == null) return null;

        return map.get(timestamp);
    }

    @Override
    public List<Persistable> getLatestUpdateAllWorkers(String sessionID, String typeID) {
        List<Persistable> list = new ArrayList<>();

        for (SessionTypeWorkerId id : updates.keySet()) {
            if (sessionID.equals(id.getSessionID()) && typeID.equals(id.getTypeID())) {
                Persistable p = getLatestUpdate(sessionID, typeID, id.workerID);
                if (p != null) {
                    list.add(p);
                }
            }
        }

        return list;
    }

    @Override
    public List<Persistable> getAllUpdatesAfter(String sessionID, String typeID, String workerID, long timestamp) {
        List<Persistable> list = new ArrayList<>();

        Map<Long, Persistable> map = getUpdateMap(sessionID, typeID, workerID, false);
        if (map == null) return list;

        for (Long time : map.keySet()) {
            if (time > timestamp) {
                list.add(map.get(time));
            }
        }

        Collections.sort(list, new Comparator<Persistable>() {
            @Override
            public int compare(Persistable o1, Persistable o2) {
                return Long.compare(o1.getTimeStamp(), o2.getTimeStamp());
            }
        });

        return list;
    }

    @Override
    public StorageMetaData getStorageMetaData(String sessionID, String typeID) {
        return this.storageMetaData.get(new SessionTypeId(sessionID, typeID));
    }

    // ----- Store new info -----

    @Override
    public void putStaticInfo(Persistable staticInfo) {
        List<StatsStorageEvent> sses = checkStorageEvents(staticInfo);
        if(!sessionIDs.contains(staticInfo.getSessionID())){
            sessionIDs.add(staticInfo.getSessionID());
        }
        SessionTypeWorkerId id = new SessionTypeWorkerId(staticInfo.getSessionID(), staticInfo.getTypeID(), staticInfo.getWorkerID());

        this.staticInfo.put(id, staticInfo);
        db.commit();    //For write ahead log: need to ensure that we persist all data to disk...
        StatsStorageEvent sse = null;
        if (listeners.size() > 0) sse = new StatsStorageEvent(StatsStorageListener.EventType.PostStaticInfo,
                staticInfo.getSessionID(), staticInfo.getTypeID(), staticInfo.getWorkerID(), staticInfo.getTimeStamp());
        for (StatsStorageListener l : listeners) {
            l.notify(sse);
        }

        notifyListeners(sses);
    }

    @Override
    public void putStaticInfo(Collection<? extends Persistable> staticInfo) {
        for(Persistable p : staticInfo){
            putStaticInfo(p);
        }
    }

    @Override
    public void putUpdate(Persistable update) {
        List<StatsStorageEvent> sses = checkStorageEvents(update);
        Map<Long, Persistable> updateMap = getUpdateMap(update.getSessionID(), update.getTypeID(), update.getWorkerID(), true);
        updateMap.put(update.getTimeStamp(), update);
        db.commit();    //For write ahead log: need to ensure that we persist all data to disk...

        StatsStorageEvent sse = null;
        if (listeners.size() > 0) sse = new StatsStorageEvent(StatsStorageListener.EventType.PostUpdate,
                update.getSessionID(), update.getTypeID(), update.getWorkerID(), update.getTimeStamp());
        for (StatsStorageListener l : listeners) {
            l.notify(sse);
        }

        notifyListeners(sses);
    }

    @Override
    public void putUpdate(Collection<? extends Persistable> updates) {
        for(Persistable p : updates){
            putUpdate(p);
        }
    }

    @Override
    public void putStorageMetaData(StorageMetaData storageMetaData) {
        List<StatsStorageEvent> sses = checkStorageEvents(storageMetaData);
        SessionTypeId id = new SessionTypeId(storageMetaData.getSessionID(), storageMetaData.getTypeID());
        this.storageMetaData.put(id, storageMetaData);

        StatsStorageEvent sse = null;
        if (listeners.size() > 0) sse = new StatsStorageEvent(StatsStorageListener.EventType.PostMetaData,
                storageMetaData.getSessionID(), storageMetaData.getTypeID(), storageMetaData.getWorkerID(), storageMetaData.getTimeStamp());
        for (StatsStorageListener l : listeners) {
            l.notify(sse);
        }

        notifyListeners(sses);
    }

    @Override
    public void putStorageMetaData(Collection<? extends StorageMetaData> storageMetaData) {
        for(StorageMetaData m : storageMetaData){
            putStorageMetaData(m);
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

    @Data
    public static class SessionTypeWorkerId implements Serializable, Comparable<SessionTypeWorkerId> {
        private final String sessionID;
        private final String typeID;
        private final String workerID;

        public SessionTypeWorkerId(String sessionID, String typeID, String workerID) {
            this.sessionID = sessionID;
            this.typeID = typeID;
            this.workerID = workerID;
        }

        @Override
        public int compareTo(SessionTypeWorkerId o) {
            int c = sessionID.compareTo(o.sessionID);
            if (c != 0) return c;
            c = typeID.compareTo(o.typeID);
            if (c != 0) return c;
            return workerID.compareTo(workerID);
        }

        @Override
        public String toString() {
            return "(" + sessionID + "," + typeID + "," + workerID + ")";
        }
    }

    @AllArgsConstructor
    @Data
    public static class SessionTypeId implements Serializable, Comparable<SessionTypeId> {
        private final String sessionID;
        private final String typeID;

        @Override
        public int compareTo(SessionTypeId o) {
            int c = sessionID.compareTo(o.sessionID);
            if (c != 0) return c;
            return typeID.compareTo(o.typeID);
        }

        @Override
        public String toString() {
            return "(" + sessionID + "," + typeID + ")";
        }
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

    private static class PersistableSerializer<T extends Persistable> implements Serializer<T> {

        @Override
        public void serialize(@NotNull DataOutput2 out, @NotNull Persistable value) throws IOException {
            //Persistable values can't be decoded in isolation, i.e., without knowing the type
            //So, we'll first write the class name, so we can decode it later...
            String className = value.getClass().getName();
            int length = className.length();
            out.writeInt(length);
            byte[] b = SbeUtil.toBytes(true, className);
            out.write(b);
            value.encode(out);
        }

        @Override
        public T deserialize(@NotNull DataInput2 input, int available) throws IOException {
            int length = input.readInt();
            byte[] classNameAsBytes = new byte[length];
            input.readFully(classNameAsBytes);
            String className = new String(classNameAsBytes, "UTF-8");
            Class<?> clazz;
            try {
                clazz = Class.forName(className);
            } catch (ClassNotFoundException e) {
                throw new RuntimeException(e);  //Shouldn't normally happen...
            }
            Persistable p;
            try {
                p = (Persistable) clazz.newInstance();
            } catch (InstantiationException | IllegalAccessException e) {
                throw new RuntimeException(e);
            }
            int remainingLength = available - length - 4;   //-4 for int length value
            byte[] temp = new byte[remainingLength];
            input.readFully(temp);
            p.decode(temp);
            return (T) p;
        }

        @Override
        public int compare(Persistable p1, Persistable p2) {
            int c = p1.getSessionID().compareTo(p2.getSessionID());
            if (c != 0) return c;
            c = p1.getTypeID().compareTo(p2.getTypeID());
            if (c != 0) return c;
            return p1.getWorkerID().compareTo(p2.getWorkerID());
        }
    }

}

/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.ui.storage;

import lombok.AllArgsConstructor;
import lombok.Data;
import org.deeplearning4j.api.storage.*;

import java.io.Serializable;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * An implementation of the {@link StatsStorage} interface, backed by MapDB
 *
 * @author Alex Black
 */
public abstract class BaseCollectionStatsStorage implements StatsStorage {

    protected Set<String> sessionIDs;
    protected Map<SessionTypeId, StorageMetaData> storageMetaData;
    protected Map<SessionTypeWorkerId, Persistable> staticInfo;

    protected Map<SessionTypeWorkerId, Map<Long, Persistable>> updates = new ConcurrentHashMap<>();

    protected List<StatsStorageListener> listeners = new ArrayList<>();

    protected BaseCollectionStatsStorage() {

    }

    protected abstract Map<Long, Persistable> getUpdateMap(String sessionID, String typeID, String workerID,
                    boolean createIfRequired);

    //Return any relevant storage events
    //We want to return these so they can be logged later. Can't be logged immediately, as this may case a race
    //condition with whatever is receiving the events: i.e., might get the event before the contents are actually
    //available in the DB
    protected List<StatsStorageEvent> checkStorageEvents(Persistable p) {
        if (listeners.isEmpty())
            return null;

        int count = 0;
        StatsStorageEvent newSID = null;
        StatsStorageEvent newTID = null;
        StatsStorageEvent newWID = null;

        //Is this a new session ID?
        if (!sessionIDs.contains(p.getSessionID())) {
            newSID = new StatsStorageEvent(this, StatsStorageListener.EventType.NewSessionID, p.getSessionID(),
                            p.getTypeID(), p.getWorkerID(), p.getTimeStamp());
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
            if (foundTypeId && foundWorkerId)
                break;
        }
        if (!foundTypeId || !foundWorkerId) {
            for (SessionTypeWorkerId stw : updates.keySet()) {
                if (!foundTypeId && typeId.equals(stw.getTypeID())) {
                    foundTypeId = true;
                }
                if (!foundWorkerId && wid.equals(stw.getWorkerID())) {
                    foundWorkerId = true;
                }
                if (foundTypeId && foundWorkerId)
                    break;
            }
        }
        if (!foundTypeId) {
            //New type ID
            newTID = new StatsStorageEvent(this, StatsStorageListener.EventType.NewTypeID, p.getSessionID(),
                            p.getTypeID(), p.getWorkerID(), p.getTimeStamp());
            count++;
        }
        if (!foundWorkerId) {
            //New worker ID
            newWID = new StatsStorageEvent(this, StatsStorageListener.EventType.NewWorkerID, p.getSessionID(),
                            p.getTypeID(), p.getWorkerID(), p.getTimeStamp());
            count++;
        }
        if (count == 0)
            return null;
        List<StatsStorageEvent> sses = new ArrayList<>(count);
        if (newSID != null)
            sses.add(newSID);
        if (newTID != null)
            sses.add(newTID);
        if (newWID != null)
            sses.add(newWID);
        return sses;
    }

    protected void notifyListeners(List<StatsStorageEvent> sses) {
        if (sses == null || sses.isEmpty() || listeners.isEmpty())
            return;
        for (StatsStorageListener l : listeners) {
            for (StatsStorageEvent e : sses) {
                l.notify(e);
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
    public Persistable getStaticInfo(String sessionID, String typeID, String workerID) {
        SessionTypeWorkerId id = new SessionTypeWorkerId(sessionID, typeID, workerID);
        return staticInfo.get(id);
    }

    @Override
    public List<Persistable> getAllStaticInfos(String sessionID, String typeID) {
        List<Persistable> out = new ArrayList<>();
        for (SessionTypeWorkerId key : staticInfo.keySet()) {
            if (sessionID.equals(key.getSessionID()) && typeID.equals(key.getTypeID())) {
                out.add(staticInfo.get(key));
            }
        }
        return out;
    }

    @Override
    public List<String> listTypeIDsForSession(String sessionID) {
        Set<String> typeIDs = new HashSet<>();
        for (SessionTypeId st : storageMetaData.keySet()) {
            if (!sessionID.equals(st.getSessionID()))
                continue;
            typeIDs.add(st.getTypeID());
        }

        for (SessionTypeWorkerId stw : staticInfo.keySet()) {
            if (!sessionID.equals(stw.getSessionID()))
                continue;
            typeIDs.add(stw.getTypeID());
        }
        for (SessionTypeWorkerId stw : updates.keySet()) {
            if (!sessionID.equals(stw.getSessionID()))
                continue;
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
    public List<String> listWorkerIDsForSessionAndType(String sessionID, String typeID) {
        List<String> out = new ArrayList<>();
        for (SessionTypeWorkerId ids : staticInfo.keySet()) {
            if (sessionID.equals(ids.getSessionID()) && typeID.equals(ids.getTypeID())) {
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
                if (map != null)
                    count += map.size();
            }
        }
        return count;
    }

    @Override
    public int getNumUpdateRecordsFor(String sessionID, String typeID, String workerID) {
        SessionTypeWorkerId id = new SessionTypeWorkerId(sessionID, typeID, workerID);
        Map<Long, Persistable> map = updates.get(id);
        if (map != null)
            return map.size();
        return 0;
    }

    @Override
    public Persistable getLatestUpdate(String sessionID, String typeID, String workerID) {
        SessionTypeWorkerId id = new SessionTypeWorkerId(sessionID, typeID, workerID);
        Map<Long, Persistable> map = updates.get(id);
        if (map == null || map.isEmpty())
            return null;
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
        if (map == null)
            return null;

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
        if (map == null)
            return list;

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
    public List<Persistable> getAllUpdatesAfter(String sessionID, String typeID, long timestamp) {
        List<Persistable> list = new ArrayList<>();

        for (SessionTypeWorkerId stw : staticInfo.keySet()) {
            if (stw.getSessionID().equals(sessionID) && stw.getTypeID().equals(typeID)) {
                Map<Long, Persistable> u = updates.get(stw);
                if (u == null)
                    continue;
                for (long l : u.keySet()) {
                    if (l > timestamp) {
                        list.add(u.get(l));
                    }
                }
            }
        }

        //Sort by time stamp
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

    @Override
    public long[] getAllUpdateTimes(String sessionID, String typeID, String workerID) {
        SessionTypeWorkerId stw = new SessionTypeWorkerId(sessionID, typeID, workerID);
        Map<Long,Persistable> m = updates.get(stw);
        if(m == null){
            return new long[0];
        }

        long[] ret = new long[m.size()];
        int i=0;
        for(Long l : m.keySet()){
            ret[i++] = l;
        }
        Arrays.sort(ret);
        return ret;
    }

    @Override
    public List<Persistable> getUpdates(String sessionID, String typeID, String workerID, long[] timestamps) {
        SessionTypeWorkerId stw = new SessionTypeWorkerId(sessionID, typeID, workerID);
        Map<Long,Persistable> m = updates.get(stw);
        if(m == null){
            return Collections.emptyList();
        }

        List<Persistable> ret = new ArrayList<>(timestamps.length);
        for(long l : timestamps){
            Persistable p = m.get(l);
            if(p != null){
                ret.add(p);
            }
        }
        return ret;
    }

    // ----- Store new info -----

    @Override
    public abstract void putStaticInfo(Persistable staticInfo);

    @Override
    public void putStaticInfo(Collection<? extends Persistable> staticInfo) {
        for (Persistable p : staticInfo) {
            putStaticInfo(p);
        }
    }

    @Override
    public abstract void putUpdate(Persistable update);

    @Override
    public void putUpdate(Collection<? extends Persistable> updates) {
        for (Persistable p : updates) {
            putUpdate(p);
        }
    }

    @Override
    public abstract void putStorageMetaData(StorageMetaData storageMetaData);

    @Override
    public void putStorageMetaData(Collection<? extends StorageMetaData> storageMetaData) {
        for (StorageMetaData m : storageMetaData) {
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
            if (c != 0)
                return c;
            c = typeID.compareTo(o.typeID);
            if (c != 0)
                return c;
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
            if (c != 0)
                return c;
            return typeID.compareTo(o.typeID);
        }

        @Override
        public String toString() {
            return "(" + sessionID + "," + typeID + ")";
        }
    }
}

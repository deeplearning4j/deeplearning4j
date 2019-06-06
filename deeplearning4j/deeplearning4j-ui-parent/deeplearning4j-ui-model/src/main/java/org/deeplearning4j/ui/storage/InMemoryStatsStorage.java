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

import org.deeplearning4j.api.storage.Persistable;
import org.deeplearning4j.api.storage.StatsStorageEvent;
import org.deeplearning4j.api.storage.StatsStorageListener;
import org.deeplearning4j.api.storage.StorageMetaData;
import org.deeplearning4j.ui.storage.mapdb.MapDBStatsStorage;

import java.io.IOException;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * A StatsStorage implementation that stores all data in memory. If persistence is required for the UI information,
 * use {@link FileStatsStorage} or {@link MapDBStatsStorage}.
 *
 * @author Alex Black
 */
public class InMemoryStatsStorage extends BaseCollectionStatsStorage {
    private final String uid;

    public InMemoryStatsStorage() {
        super();
        String str = UUID.randomUUID().toString();
        uid = str.substring(0, Math.min(str.length(), 8));

        sessionIDs = Collections.synchronizedSet(new HashSet<String>());
        storageMetaData = new ConcurrentHashMap<>();
        staticInfo = new ConcurrentHashMap<>();
    }


    @Override
    protected synchronized Map<Long, Persistable> getUpdateMap(String sessionID, String typeID, String workerID,
                    boolean createIfRequired) {
        SessionTypeWorkerId id = new SessionTypeWorkerId(sessionID, typeID, workerID);
        if (updates.containsKey(id)) {
            return updates.get(id);
        }
        if (!createIfRequired) {
            return null;
        }
        Map<Long, Persistable> updateMap = new ConcurrentHashMap<>();
        updates.put(id, updateMap);
        return updateMap;
    }

    @Override
    public void putStaticInfo(Persistable staticInfo) {
        List<StatsStorageEvent> sses = checkStorageEvents(staticInfo);
        if (!sessionIDs.contains(staticInfo.getSessionID())) {
            sessionIDs.add(staticInfo.getSessionID());
        }
        SessionTypeWorkerId id = new SessionTypeWorkerId(staticInfo.getSessionID(), staticInfo.getTypeID(),
                        staticInfo.getWorkerID());

        this.staticInfo.put(id, staticInfo);
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


    @Override
    public void close() throws IOException {
        //No op
    }

    @Override
    public boolean isClosed() {
        return false;
    }


    @Override
    public String toString() {
        return "InMemoryStatsStorage(uid=" + uid + ")";
    }
}

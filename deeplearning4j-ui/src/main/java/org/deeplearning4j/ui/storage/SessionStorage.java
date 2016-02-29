package org.deeplearning4j.ui.storage;

import com.google.common.collect.HashBasedTable;
import com.google.common.collect.Table;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.ui.storage.def.ObjectType;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * @author raver119@gmail.com
 */
public class SessionStorage {
    private static final SessionStorage INSTANCE = new SessionStorage();

    private Table<String, ObjectType, Object> storage = HashBasedTable.create();
    private ConcurrentHashMap<String, AtomicLong> accessTime = new ConcurrentHashMap<>();

    private ReentrantReadWriteLock singleLock = new ReentrantReadWriteLock();

    private SessionStorage() {
        ;
    }

    public static SessionStorage getInstance() {
        return INSTANCE;
    }

    public Object getObject(String sessionId, ObjectType type) {
        try {
            singleLock.readLock().lock();

            if (!accessTime.containsKey(sessionId)) {
                accessTime.put(sessionId, new AtomicLong(System.currentTimeMillis()));
            }
            accessTime.get(sessionId).set(System.currentTimeMillis());

            return storage.get(sessionId, type);
        } finally {
            singleLock.readLock().unlock();
        }
    }

    public void putObject(String sessionId, ObjectType type, Object object) {
        try {
            singleLock.writeLock().lock();

            if (!accessTime.containsKey(sessionId)) {
                accessTime.put(sessionId, new AtomicLong(System.currentTimeMillis()));
            }
            accessTime.get(sessionId).set(System.currentTimeMillis());

            truncateUnused();

            storage.put(sessionId, type, object);
        } finally {
            singleLock.writeLock().unlock();
        }
    }

    /**
     * This method removes all references that were not used within some timeframe
     */
    protected void truncateUnused() {
        List<String> sessions = Collections.list(accessTime.keys());
        List<Pair<String, ObjectType>> removals = new ArrayList<>();

        for (String session: sessions) {
            long time = accessTime.get(session).get();
            if (time < System.currentTimeMillis() - (30 * 60 * 1000L)) {
                accessTime.remove(session);
                try {
                    singleLock.readLock().lock();


                    Map<ObjectType, Object> map = storage.row(session);
                    for (ObjectType type: map.keySet()) {
                        removals.add(Pair.makePair(session, type));
                    }
                } finally {
                    singleLock.readLock().unlock();
                }


            }
        }

        try {
            singleLock.writeLock().lock();

            for (Pair<String, ObjectType> objects : removals) {
                storage.remove(objects.getFirst(), objects.getSecond());
            }
        } finally {
            singleLock.writeLock().unlock();
        }
    }

    public Map<String, List<ObjectType>> getSessions() {
        Map<String, List<ObjectType>> result = new ConcurrentHashMap<>();
        try {
            singleLock.readLock().lock();

            Set<String> sessions = storage.rowKeySet();
            for (String session : sessions) {
                Map<ObjectType, Object> map = storage.row(session);
                for (ObjectType type : map.keySet()) {
                    if (!result.containsKey(session)) {
                        result.put(session, new ArrayList<ObjectType>());
                    }
                    result.get(session).add(type);
                }
            }

            return result;
        } finally {
            singleLock.readLock().unlock();
        }
    }

    public List<String> getSessions(ObjectType type) {
        List<String> results = new ArrayList<>();
        try {
            singleLock.readLock().lock();

            Map<String, Object> map = storage.column(type);
            for (String session: map.keySet()) {
                results.add(session);
            }
        } finally {
            singleLock.readLock().unlock();
        }

        return results;
    }

    public Map<ObjectType, List<String>> getEvents() {
        Map<ObjectType, List<String>> result = new ConcurrentHashMap<>();
        try {
            singleLock.readLock().lock();

            Set<ObjectType> events = storage.columnKeySet();
            for (ObjectType type: events) {
                Map<String, Object> map = storage.column(type);
                for (String session: map.keySet()) {
                    if (!result.containsKey(type)) {
                        result.put(type, new ArrayList<String>());
                    }
                    result.get(type).add(session);
                }
            }

            return result;
        } finally {
            singleLock.readLock().unlock();
        }
    }
}

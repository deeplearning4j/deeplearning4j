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

    private Table<Integer, ObjectType, Object> storage = HashBasedTable.create();
    private ConcurrentHashMap<Integer, AtomicLong> accessTime = new ConcurrentHashMap<>();

    private ReentrantReadWriteLock singleLock = new ReentrantReadWriteLock();

    private SessionStorage() {
        ;
    }

    public static SessionStorage getInstance() {
        return INSTANCE;
    }

    public Object getObject(Integer sessionId, ObjectType type) {
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

    public void putObject(Integer sessionId, ObjectType type, Object object) {
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
        List<Integer> sessions = Collections.list(accessTime.keys());
        List<Pair<Integer, ObjectType>> removals = new ArrayList<>();

        for (Integer session: sessions) {
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

            for (Pair<Integer, ObjectType> objects : removals) {
                storage.remove(objects.getFirst(), objects.getSecond());
            }
        } finally {
            singleLock.writeLock().unlock();
        }
    }

    public Map<Integer, List<ObjectType>> getSessions() {
        Map<Integer, List<ObjectType>> result = new ConcurrentHashMap<>();
        try {
            singleLock.readLock().lock();

            Set<Integer> sessions = storage.rowKeySet();
            for (Integer session : sessions) {
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

    public List<Integer> getSessions(ObjectType type) {
        List<Integer> results = new ArrayList<>();
        try {
            singleLock.readLock().lock();

            Map<Integer, Object> map = storage.column(type);
            for (Integer session: map.keySet()) {
                results.add(session);
            }
        } finally {
            singleLock.readLock().unlock();
        }

        return results;
    }

    public Map<ObjectType, List<Integer>> getEvents() {
        Map<ObjectType, List<Integer>> result = new ConcurrentHashMap<>();
        try {
            singleLock.readLock().lock();

            Set<ObjectType> events = storage.columnKeySet();
            for (ObjectType type: events) {
                Map<Integer, Object> map = storage.column(type);
                for (Integer session: map.keySet()) {
                    if (!result.containsKey(type)) {
                        result.put(type, new ArrayList<Integer>());
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

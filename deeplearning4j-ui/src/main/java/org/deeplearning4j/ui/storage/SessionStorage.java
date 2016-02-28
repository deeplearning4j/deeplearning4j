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

    private Table<Long, ObjectType, Object> storage = HashBasedTable.create();
    private ConcurrentHashMap<Long, AtomicLong> accessTime = new ConcurrentHashMap<>();

    private ReentrantReadWriteLock singleLock = new ReentrantReadWriteLock();

    private SessionStorage() {
        ;
    }

    public static SessionStorage getInstance() {
        return INSTANCE;
    }

    public Object getObject(Long sessionId, ObjectType type) {
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

    public void putObject(Long sessionId, ObjectType type, Object object) {
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
        List<Long> sessions = Collections.list(accessTime.keys());
        List<Pair<Long, ObjectType>> removals = new ArrayList<>();

        for (Long session: sessions) {
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

            for (Pair<Long, ObjectType> objects : removals) {
                storage.remove(objects.getFirst(), objects.getSecond());
            }
        } finally {
            singleLock.writeLock().unlock();
        }
    }
}

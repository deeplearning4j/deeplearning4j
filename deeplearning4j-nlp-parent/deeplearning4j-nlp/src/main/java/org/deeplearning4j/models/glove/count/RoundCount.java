package org.deeplearning4j.models.glove.count;

import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * Simple circular counter, that circulates within 0...Limit, both inclusive
 *
 * @author raver119@gmail.com
 */
public class RoundCount {

    private int limit = 0;
    private int lower = 0;
    private int value = 0;

    private ReentrantReadWriteLock lock = new ReentrantReadWriteLock();

    /**
     * Creates new RoundCount instance.
     *
     * @param limit Maximum top value for this counter. Inclusive.
     */
    public RoundCount(int limit) {
        this.limit = limit;
    }

    /**
     * Creates new RoundCount instance.
     *
     * @param lower - Minimum value for this counter. Inclusive
     * @param top - Maximum value for this counter. Inclusive.
     */
    public RoundCount(int lower, int top) {
        this.limit = top;
        this.lower = lower;
    }

    public int previous() {
        try {
            lock.readLock().lock();
            if (value == lower)
                return limit;
            else
                return value - 1;
        } finally {
            lock.readLock().unlock();
        }
    }

    public int get() {
        try {
            lock.readLock().lock();
            return value;
        } finally {
            lock.readLock().unlock();
        }
    }

    public void tick() {
        try {
            lock.writeLock().lock();
            if (value == limit)
                value = lower;
            else
                value++;
        } finally {
            lock.writeLock().unlock();
        }
    }
}

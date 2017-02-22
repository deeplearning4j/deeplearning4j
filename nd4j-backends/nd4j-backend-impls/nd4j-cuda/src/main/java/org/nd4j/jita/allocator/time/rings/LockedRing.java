package org.nd4j.jita.allocator.time.rings;

import org.nd4j.jita.allocator.time.Ring;

import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * @author raver119@gmail.com
 */
public class LockedRing implements Ring {

    private final float[] ring;
    private final AtomicInteger position = new AtomicInteger(0);

    private ReentrantReadWriteLock lock = new ReentrantReadWriteLock();

    /**
     * Builds new BasicRing with specified number of elements stored
     *
     * @param ringLength
     */
    public LockedRing(int ringLength) {
        this.ring = new float[ringLength];
    }

    public float getAverage() {
        try {
            lock.readLock().lock();

            float rates = 0.0f;
            int x = 0;
            int existing = 0;
            for (x = 0; x < ring.length; x++) {
                rates += ring[x];
                if (ring[x] > 0) {
                    existing++;
                }
            }
            if (existing > 0) {
                return rates / existing;
            } else {
                return 0.0f;
            }
        } finally {
            lock.readLock().unlock();
        }
    }

    public void store(double rate) {
        store((float) rate);
    }

    public void store(float rate) {
        try {
            lock.writeLock().lock();

            int pos = position.getAndIncrement();
            if (pos >= ring.length) {
                pos = 0;
                position.set(0);
            }
            ring[pos] = rate;
        } finally {
            lock.writeLock().unlock();
        }
    }
}

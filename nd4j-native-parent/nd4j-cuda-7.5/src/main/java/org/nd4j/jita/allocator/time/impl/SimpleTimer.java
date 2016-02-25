package org.nd4j.jita.allocator.time.impl;

import org.nd4j.jita.allocator.time.RateTimer;

import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * @author raver119@gmail.com
 */
public class SimpleTimer implements RateTimer {
    protected volatile long timeframe;
    protected final AtomicLong latestEvent = new AtomicLong(0);
    protected volatile long[] buckets;
    protected final ReentrantReadWriteLock lock = new ReentrantReadWriteLock();

    public SimpleTimer(long timeframe, TimeUnit timeUnit) {
        this.timeframe = TimeUnit.MILLISECONDS.convert(timeframe, timeUnit);

        int bucketsize = (int) TimeUnit.SECONDS.convert(timeframe, timeUnit);
        this.buckets = new long[bucketsize];
    }

    /**
     * This method notifies timer about event
     */
    @Override
    public void triggerEvent() {
        // delete all outdated events
        try {
            lock.writeLock().lock();
            long currentTime = System.currentTimeMillis();
            if (latestEvent.get() == 0) this.latestEvent.set(currentTime);

            actualizeCounts(currentTime);
            int currentBin = (int) TimeUnit.SECONDS.convert(currentTime, TimeUnit.MILLISECONDS) % buckets.length;

            buckets[currentBin]++;

            // nullify next bin
            if (currentBin == buckets.length - 1) buckets[0] = 0;
            else buckets[currentBin+1] = 0;

            // set new time
            this.latestEvent.set(currentTime);
        } finally {
            lock.writeLock().unlock();
        }
    }

    protected void actualizeCounts(long currentTime) {
        int currentBin = (int) TimeUnit.SECONDS.convert(currentTime, TimeUnit.MILLISECONDS) % buckets.length;

        long lastTime = latestEvent.get();
        int expiredBinsNum = (int) TimeUnit.SECONDS.convert(currentTime - lastTime, TimeUnit.MILLISECONDS);

        if (expiredBinsNum > 0 && expiredBinsNum < buckets.length) {
            for (int x = 1; x <= expiredBinsNum; x++) {
                int position = currentBin + x;
                if (position >= buckets.length)
                    position -= buckets.length;
                buckets[position] = 0;
            }
        } else if (expiredBinsNum >= buckets.length) {
            // nullify everything, counter is really outdated
            for (int x = 0; x< buckets.length; x++)
                buckets[x] = 0;
        } else {
            // do nothing here probably
            ;
        }
    }

    /**
     * This method returns average frequency of events happened within predefined timeframe
     *
     * @return
     */
    @Override
    public double getFrequencyOfEvents() {
        return getNumberOfEvents() / (double)TimeUnit.SECONDS.convert(timeframe, TimeUnit.MILLISECONDS);
    }

    protected long sumCounts() {
        long result = 0;
        for (int x = 0; x < buckets.length; x++)
            result += buckets[x];

        return result;
    }

    /**
     * This method returns total number of events happened withing predefined timeframe
     *
     * @return
     */
    @Override
    public long getNumberOfEvents() {
        try {
            lock.readLock().lock();
            long currentTime = System.currentTimeMillis();
            actualizeCounts(currentTime);
            return sumCounts();
        } finally {
            lock.readLock().unlock();
        }
    }
}

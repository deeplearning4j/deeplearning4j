package org.nd4j.jita.allocator.time.impl;

import org.nd4j.jita.allocator.time.DecayingTimer;

import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;

/**
 * This is simple implementation of DecayingTimer, it doesn't store any actual information for number of events happened.
 * Just a fact: there were events, or there were no events
 *
 * @author raver119@gmail.com
 */
public class BinaryTimer implements DecayingTimer {
    private AtomicLong timer;
    private long timeframeMilliseconds;

    public BinaryTimer(long timeframe, TimeUnit timeUnit) {
        timer = new AtomicLong(System.currentTimeMillis());

        timeframeMilliseconds = TimeUnit.MILLISECONDS.convert(timeframe, timeUnit);
    }

    /**
     * This method notifies timer about event
     */
    @Override
    public void triggerEvent() {
        timer.set(System.currentTimeMillis());
    }

    /**
     * This method returns average frequency of events happened within predefined timeframe
     *
     * @return
     */
    @Override
    public double getFrequencyOfEvents() {
        if (isAlive()) {
            return 1;
        } else {
            return 0;
        }
    }

    /**
     * This method returns total number of events happened withing predefined timeframe
     *
     * @return
     */
    @Override
    public long getNumberOfEvents() {
        if (isAlive()) {
            return 1;
        } else {
            return 0;
        }
    }

    protected boolean isAlive() {
        long currentTime = System.currentTimeMillis();

        if (currentTime - timer.get() > timeframeMilliseconds) {
            return false;
        }

        return true;
    }
}

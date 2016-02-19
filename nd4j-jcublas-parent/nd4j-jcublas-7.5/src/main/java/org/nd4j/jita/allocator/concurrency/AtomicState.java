package org.nd4j.jita.allocator.concurrency;

import org.nd4j.jita.allocator.enums.AccessState;

import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/**
 *
 * Thread-safe atomic Tick/Tack/Toe implementation.
 *
 * TODO: add more explanations here
 *
 * @author raver119@gmail.com
 */
public class AtomicState {

    protected final AtomicInteger currentState;

    protected final AtomicLong tickRequests = new AtomicLong(0);
    protected final AtomicLong tackRequests = new AtomicLong(0);

    protected final AtomicLong waitingTicks = new AtomicLong(0);

    protected final AtomicBoolean isToeWaiting = new AtomicBoolean(false);
    protected final AtomicBoolean isToeScheduled = new AtomicBoolean(false);

    protected final AtomicLong toeThread = new AtomicLong(0);

    public AtomicState() {
        this(AccessState.TACK);
    }

    public AtomicState(AccessState initialStatus) {
        currentState = new AtomicInteger(initialStatus.ordinal());
    }

    /**
     * This method requests to change state to Tick.
     *
     * PLEASE NOTE: this method is blocking, if memory is in Toe state
     */
    public void requestTick() {
        requestTick(10, TimeUnit.SECONDS);
    }

    /**
     * This method requests to change state to Tick.
     *
     * PLEASE NOTE: this method is blocking, if memory is in Toe state.
     * PLEASE NOTE: if Tick can't be acquired within specified timeframe, exception will be thrown
     *
     * @param time
     * @param timeUnit
     */
    public void requestTick(long time, TimeUnit timeUnit) {
        long timeframeMs = TimeUnit.MILLISECONDS.convert(time, timeUnit);
        long currentTime = System.currentTimeMillis();
        boolean isWaiting = false;

        // if we have Toe request queued - we' have to wait till it finishes.
        try {
            while (isToeScheduled.get() || isToeWaiting.get() || getCurrentState() == AccessState.TOE) {
                if (!isWaiting) {
                    isWaiting = true;
                    waitingTicks.incrementAndGet();
                }
                Thread.sleep(50);
            }

            currentState.set(AccessState.TICK.ordinal());
            waitingTicks.decrementAndGet();
            tickRequests.incrementAndGet();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * This method requests to change state to Tack
     *
     *
     */
    public void requestTack() {
        currentState.set(AccessState.TACK.ordinal());
        tackRequests.incrementAndGet();
    }

    /**
     *
     * This method requests to change state to Toe
     *
     *
     * PLEASE NOTE: this method is blocking, untill all Tick requests are brought down to Tack state;
     *
     */
    public void requestToe() {
        isToeWaiting.set(true);
        try {

            while (getCurrentState() != AccessState.TACK) {
                Thread.sleep(10);
            }
            currentState.set(AccessState.TOE.ordinal());

            toeThread.set(Thread.currentThread().getId());

            isToeWaiting.set(false);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * This method requests to change state to Toe
     *
     * PLEASE NOTE: this method is non-blocking, if Toe request is impossible atm, it will return false.
     *
     * @return TRUE, if Toe state entered, FALSE otherwise
     */
    public boolean tryRequestToe() {
        scheduleToe();
        if (isToeWaiting.get() || getCurrentState() == AccessState.TOE) {
            //System.out.println("discarding TOE");
            discardScheduledToe();
            return false;
        } else {
            //System.out.println("requesting TOE");
            discardScheduledToe();
            requestToe();
            return true;
        }
    }

    /**
     * This method requests release Toe status back to Tack.
     *
     * PLEASE NOTE: only the thread originally entered Toe state is able to release it.
     */
    public void releaseToe() {
        if (getCurrentState() == AccessState.TOE) {
            if (toeThread.get() == Thread.currentThread().getId()) {
                tickRequests.set(0);
                tackRequests.set(0);

                currentState.set(AccessState.TACK.ordinal());
            } else throw new IllegalStateException("releaseToe() is called from different thread.");
        } else throw new IllegalStateException("Object is NOT in Toe state!");
    }

    /**
     * This method returns the current memory state
     *
     * @return
     */
    public AccessState getCurrentState() {
        if (AccessState.values()[currentState.get()] == AccessState.TOE) {
            return AccessState.TOE;
        } else {
            if (tickRequests.get() <= tackRequests.get()) {

                // TODO: looks like this piece of code should be locked :/
                tickRequests.set(0);
                tackRequests.set(0);

                return AccessState.TACK;
            } else return AccessState.TICK;
        }
    }

    /**
     * This methods
     *
     * @return number of WAITING tick requests, if they are really WAITING. If state isn't Toe, return value will always be 0.
     */
    public long getWaitingTickRequests() {
        return waitingTicks.get();
    }

    /**
     * This method returns number of current Tick sessions
     *
     * @return
     */
    public long getTickRequests() {
        return tickRequests.get();
    }

    /**
     * This method returns number of current Tack sessions
     * @return
     */
    public long getTackRequests() {
        return tackRequests.get();
    }

    /**
     * This method checks, if Toe state can be entered.
     *
     * @return True if Toe is available, false otherwise
     */
    public boolean isToeAvailable() {
        return getCurrentState() == AccessState.TACK;
    }

    /**
     * This method schedules Toe state entry, but doesn't enters it.
     */
    public void scheduleToe() {
        isToeScheduled.set(true);
    }

    /**
     * This method discards scheduled Toe state entry, but doesn't exits currently entered Toe state, if that's the case.
     */
    public void discardScheduledToe() {
        if (isToeScheduled.get()) {
            isToeScheduled.set(false);
        }
    }
}

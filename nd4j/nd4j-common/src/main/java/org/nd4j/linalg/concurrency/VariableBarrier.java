package org.nd4j.linalg.concurrency;

import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.LockSupport;

/**
 * This class is implementation of Cyclic Barrier, but with VARIABLE number of participants on each cycle
 *
 * @author raver119@gmail.com
 */
public class VariableBarrier {
    protected AtomicInteger consumers = new AtomicInteger(0);

    protected AtomicInteger barrier = new AtomicInteger(0);

    public VariableBarrier() {
        // no-op at this moment
    }

    /**
     * This method blocks untill all consumers are at this point
     */
    public void synchronizedBlock() throws InterruptedException {
        // we should ensure, that number of consumers already defined
        while (consumers.get() < 0)
            Thread.sleep(10);

        //incrementing barrier counter
        barrier.incrementAndGet();

        // now we block until all consumers are here
        while (barrier.get() != consumers.get())
            Thread.sleep(10);

    }

    /**
     * This method notifies that all consumers stepped out of this point
     */
    public void desynchronizedBlock() throws InterruptedException {
        // leaving barrier
        barrier.decrementAndGet();

        while (barrier.get() != 0)
            Thread.sleep(10);
    }

    /**
     * This method specifies how many consumers will be active on next cycle
     * @param numberOfConsumers
     */
    public void registerConsumers(int numberOfConsumers) {

        while (barrier.get() != 0)
            LockSupport.parkNanos(1000);

        this.consumers.set(numberOfConsumers);
    }


    protected int getNumberOfConsumers() {
        return this.consumers.get();
    }
}

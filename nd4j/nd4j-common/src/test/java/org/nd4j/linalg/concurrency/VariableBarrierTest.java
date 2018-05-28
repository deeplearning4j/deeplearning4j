package org.nd4j.linalg.concurrency;

import lombok.NonNull;
import lombok.val;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.primitives.AtomicBoolean;

import java.util.Queue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.locks.LockSupport;

import static org.junit.Assert.*;

public class VariableBarrierTest {

    @Before
    public void setUp() throws Exception {
    }

    @After
    public void tearDown() throws Exception {
    }

    @Test (timeout = 1000L)
    public void getNumberOfConsumers() {
        val barrier = new VariableBarrier();
        barrier.registerConsumers(3);

        assertEquals(3, barrier.getNumberOfConsumers());
    }


    @Test
    public void testVariableBarrier_1() {


    }




    protected static class WorkerThread extends Thread implements Runnable {
        protected Queue<Integer> queue = new LinkedBlockingQueue<>(1);
        protected AtomicBoolean shouldWork = new AtomicBoolean(true);
        protected final VariableBarrier barrier;
        protected final long time;

        protected WorkerThread(@NonNull VariableBarrier barrier, long time) {
            this.barrier = barrier;
            this.time = time;
        }

        public void run() {
            while (shouldWork.get()) {
                try {
                    // we're entering synchronous block
                    barrier.synchronizedBlock();

                    // doing something important
                    LockSupport.parkNanos(time);

                    // leaving synchronized block
                    barrier.desynchronizedBlock();
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
            }
        }

        public void shutdown() {
            shouldWork.set(false);
        }
    }
}
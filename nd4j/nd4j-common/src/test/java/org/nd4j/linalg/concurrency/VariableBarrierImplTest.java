package org.nd4j.linalg.concurrency;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.primitives.AtomicBoolean;

import java.util.Queue;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.locks.LockSupport;

import static org.junit.Assert.*;

@Slf4j
public class VariableBarrierImplTest {

    @Before
    public void setUp() throws Exception {
    }

    @After
    public void tearDown() throws Exception {
    }

    /**
     * This test checks for VariableBarrierImpl WITHOUT tail sync
     *
     * @throws Exception
     */
    @Test (timeout = 30000L)
    public void testVariableBarrier_1() throws Exception {

        val testSize = 100;
        val workersOptions = new int[] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        val workloads = new long[] {10, 1000, 10000, 100000, 1000000};

        for (val workers: workersOptions) {
            for (val workload: workloads) {
                log.info("Trying {} workers with {} ns workloads", workers, workload);
                val zoo = new WorkerThread[workers];
                val barrier = new VariableBarrierImpl();
                val queue = new ArrayBlockingQueue<Integer>(testSize + 1);

                // creating our initial workers
                for (int z = 0; z < workers; z++) {
                    zoo[z] = new WorkerThread(z, barrier, workload, queue);

                    // as soon as we start - all threads just block on queue.take(), waiting for next dataset
                    zoo[z].start();
                }

                // now we imitate our PW flow
                for (int e = 0; e < testSize; e++) {
                    // this is simple counter for interleaved fit
                    val pos = e % workers;

                    // blocking feed, won't advance unless there's some space in queue
                    zoo[pos].feedQueue(e);

                    // check if we're on last step
                    if (pos == workers - 1) {
                        barrier.registerConsumers(workers);
                    }
                }

                // notify barrier that there's no need in synchronization anymore
                barrier.bypassEverything();

                // finalizing process
                for (int z = 0; z < workers; z++) {
                    // setting shutdown flag
                    zoo[z].shutdown();

                    // waiting for thread to actually exit
                    zoo[z].join();
                }


                assertEquals(testSize, queue.size());
            }
        }
    }

    /**
     * This test checks for VariableBarrierImpl WITH tail sync
     *
     * @throws Exception
     */
    @Test (timeout = 30000L)
    public void testVariableBarrier_2() throws Exception {

        val testSize = 100;
        val workersOptions = new int[] {3, 4, 5, 6, 7, 8, 9, 10};
        val workloads = new long[] {10, 1000, 10000, 100000, 1000000};

        for (val workers: workersOptions) {
            for (val workload: workloads) {
                log.info("Trying {} workers with {} ns workloads", workers, workload);
                val zoo = new WorkerThread[workers];
                val barrier = new VariableBarrierImpl();
                val queue = new ArrayBlockingQueue<Integer>(testSize + 1);
                int consumers = 0;

                // creating our initial workers
                for (int z = 0; z < workers; z++) {
                    zoo[z] = new WorkerThread(z, barrier, workload, queue);

                    // as soon as we start - all threads just block on queue.take(), waiting for next dataset
                    zoo[z].start();
                }

                // now we imitate our PW flow
                for (int e = 0; e < testSize; e++) {
                    // this is simple counter for interleaved fit
                    val pos = e % workers;
                    consumers = pos + 1;

                    // blocking feed, won't advance unless there's some space in queue
                    zoo[pos].feedQueue(e);

                    // check if we're on last step
                    if (pos == workers - 1) {
                        barrier.registerConsumers(workers);
                    }
                }

                // notifying about last consumers left running
                if (consumers != workers) {
                    barrier.registerConsumers(consumers);
                }

                // finalizing process
                for (int z = 0; z < workers; z++) {
                    // setting shutdown flag
                    zoo[z].shutdown();

                    // waiting for thread to actually exit
                    zoo[z].join();
                }

                //barrier.checkForException();

                assertEquals(testSize, queue.size());
            }
        }
    }


    protected static class WorkerThread extends Thread implements Runnable {
        protected BlockingQueue<Integer> queue = new LinkedBlockingQueue<>(1);
        protected AtomicBoolean shouldWork = new AtomicBoolean(true);
        protected final VariableBarrier barrier;
        protected final long time;
        protected final Queue<Integer> outerQueue;

        protected WorkerThread(int id, @NonNull VariableBarrier barrier, long time, Queue<Integer> queue) {
            this.barrier = barrier;
            this.time = time;
            this.outerQueue = queue;

            this.setName("Worker thread " + id);
        }

        public void feedQueue(Integer value) {
            try {
                this.queue.put(value);
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }

        public void run() {
            while (shouldWork.get()) {
                try {
                    // taking "DataSet" here
                    val ds = queue.poll(25, TimeUnit.MILLISECONDS);

                    if (ds != null) {
                        // we're entering synchronous block
                        barrier.synchronizedBlock();

                        // stroing proof of work
                        outerQueue.add(ds);

                        // kind of doing something important here
                        LockSupport.parkNanos(time);

                        // leaving synchronized block
                        barrier.desynchronizedBlock();
                    }
                } catch (InterruptedException e) {
                    // noop
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
            }
        }

        public void shutdown() {
            // we dont want early termination with non-empty queues here
            while (!this.queue.isEmpty())
                LockSupport.parkNanos(100);

            shouldWork.set(false);
        }
    }
}
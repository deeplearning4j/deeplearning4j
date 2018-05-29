package org.nd4j.linalg.concurrency;

import lombok.NonNull;
import lombok.experimental.var;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.apache.commons.lang3.RandomUtils;
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

    @Test
    public void testPlanMapping_1() {
        val barrier = new VariableBarrierImpl(true);

        int[] array = {3, 3, 4};
        barrier.updatePlan(array);

        assertEquals(3, barrier.getConsumersForIteration(0));
        assertEquals(3, barrier.getConsumersForIteration(1));
        assertEquals(3, barrier.getConsumersForIteration(2));
        assertEquals(1, barrier.getConsumersForIteration(3));
        assertEquals(0, barrier.getConsumersForIteration(4));
        assertEquals(0, barrier.getConsumersForIteration(5));
        assertEquals(0, barrier.getConsumersForIteration(6));
    }


    @Test
    public void testPlanMapping_2() {
        val barrier = new VariableBarrierImpl(true);

        int[] array = {1, 2, 3, 4};
        barrier.updatePlan(array);

        assertEquals(4, barrier.getConsumersForIteration(0));
        assertEquals(3, barrier.getConsumersForIteration(1));
        assertEquals(2, barrier.getConsumersForIteration(2));
        assertEquals(1, barrier.getConsumersForIteration(3));
        assertEquals(0, barrier.getConsumersForIteration(4));
    }

    @Test
    public void testPlanMapping_3() {
        val barrier = new VariableBarrierImpl(true);

        int[] array = {2, 17};
        barrier.updatePlan(array);

        assertEquals(2, barrier.getConsumersForIteration(0));
        assertEquals(2, barrier.getConsumersForIteration(1));
        assertEquals(1, barrier.getConsumersForIteration(2));
        assertEquals(1, barrier.getConsumersForIteration(3));
        assertEquals(1, barrier.getConsumersForIteration(4));
        assertEquals(1, barrier.getConsumersForIteration(5));
        assertEquals(0, barrier.getConsumersForIteration(17));
        assertEquals(0, barrier.getConsumersForIteration(18));
    }



    /**
     * This test checks for VariableBarrierImpl WITHOUT tail sync
     *
     * @throws Exception
     */
    @Test (timeout = 45000L)
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
    @Test (timeout = 45000L)
    public void testVariableBarrier_2() throws Exception {

        val testSize = 100;
        val workersOptions = new int[] {1, 23, 4, 5, 6, 7, 8, 9, 10};
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


    /**
     * This test checks for VariableBarrierImpl WITH tail sync and WITH workload/workers within main thread
     *
     * @throws Exception
     */
    @Test (timeout = 45000L)
    public void testVariableBarrier_3() throws Exception {

        val testSize = 100;
        val workersOptions = new int[] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
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

                    // we mimic ETL pressure this way
                    LockSupport.parkNanos(workload);
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


    /**
     * This test checks for VariableBarrierImpl WITH tail sync and WITH workload/workers within main thread.
     * On top of that: this test uses workers with variable workload 0...workload
     * On top of that: this test mimics TBPTT with variable sequence length scenario
     *
     * @throws Exception
     */
    @Test //(timeout = 45000L)
    public void testVariableBarrier_5() throws Exception {

        val testSize = 113;
        val workersOptions = new int[] {2, 3, 4, 5, 6, 7, 8, 9, 10};
        val workloads = new long[] {100, 1000, 10000, 100000, 1000000};

        for (val workers: workersOptions) {
            for (val workload: workloads) {
                log.info("Trying {} workers with {} ns workloads", workers, workload);
                val zoo = new TruncatedWorkerThread[workers];
                val barrier = new VariableBarrierImpl(true);
                val queue = new ArrayBlockingQueue<Integer>(testSize + 1);
                int consumers = 0;

                // creating our initial workers
                for (int z = 0; z < workers; z++) {
                    zoo[z] = new TruncatedWorkerThread(z, barrier, workload, queue, true);

                    // as soon as we start - all threads just block on queue.take(), waiting for next dataset
                    zoo[z].start();
                }

                int[] plan = new int[workers];

                // now we imitate our PW flow
                for (int e = 0; e < testSize; e++) {
                    // this is simple counter for interleaved fit
                    val pos = e % workers;
                    consumers = pos + 1;

                    val seriesLength = RandomUtils.nextInt(1, 5);
                    plan[pos] = seriesLength;

                    // blocking feed, won't advance unless there's some space in queue
                    zoo[pos].feedQueue(seriesLength);
                    //log.info("Feeding with {} time steps", seriesLength);

                    // check if we're on last step
                    if (pos == workers - 1) {
                        barrier.registerConsumers(plan);

                        barrier.blockMainThread();

                        plan = new int[workers];
                    }

                    // we mimic ETL pressure this way
                    LockSupport.parkNanos(workload);
                }

                // notifying about last consumers left running
                if (consumers != workers) {
                    barrier.registerConsumers(plan);
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

    /**
     * This test checks for VariableBarrierImpl WITH tail sync and WITH workload/workers within main thread.
     * On top of that: this test uses workers with variable workload 0...workload
     *
     * @throws Exception
     */
    @Test (timeout = 45000L)
    public void testVariableBarrier_4() throws Exception {

        val testSize = 100;
        val workersOptions = new int[] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
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
                    zoo[z] = new WorkerThread(z, barrier, workload, queue, true);

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

                    // we mimic ETL pressure this way
                    LockSupport.parkNanos(workload);
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

    /**
     * This class simulates TBPTT with variable sequence length scenario
     */
    protected static class TruncatedWorkerThread extends WorkerThread {
        protected TruncatedWorkerThread(int id, @NonNull VariableBarrier barrier, long time, Queue<Integer> queue, boolean variableWorkload) {
            super(id, barrier, time, queue, variableWorkload);
        }

        @Override
        public void run() {
            while (shouldWork.get()) {
                try {
                    // taking "DataSet" here
                    val ds = queue.poll(25, TimeUnit.MILLISECONDS);

                    if (ds != null) {
                        // simulating variable sequences etc here
                        val sequenceLength = ds;
                        //log.info("{} TS length: {}", Thread.currentThread().getName(), sequenceLength);
                        for (int e = 0; e < sequenceLength; e++) {
                            // kind of doing something important here
                            if (variableWorkload)
                                LockSupport.parkNanos(RandomUtils.nextInt(0, (int) time));
                            else
                                LockSupport.parkNanos(time);

                            // we're entering synchronous block
                            barrier.synchronizedBlock();

                            // storing proof of work, but only once
                            if (e == 0)
                                outerQueue.add(ds);


                            // leaving synchronized block
                            barrier.desynchronizedBlock();
                        }
                    }
                } catch (InterruptedException e) {
                    // noop
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
            }
        }
    }

    protected static class WorkerThread extends Thread implements Runnable {
        protected BlockingQueue<Integer> queue = new LinkedBlockingQueue<>(1);
        protected AtomicBoolean shouldWork = new AtomicBoolean(true);
        protected final VariableBarrier barrier;
        protected final long time;
        protected final Queue<Integer> outerQueue;
        protected final boolean variableWorkload;

        protected WorkerThread(int id, @NonNull VariableBarrier barrier, long time, Queue<Integer> queue) {
            this(id, barrier, time, queue, false);
        }

        protected WorkerThread(int id, @NonNull VariableBarrier barrier, long time, Queue<Integer> queue, boolean variableWorkload) {
            this.barrier = barrier;
            this.time = time;
            this.outerQueue = queue;
            this.variableWorkload = variableWorkload;

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

                        // storing proof of work
                        outerQueue.add(ds);

                        // kind of doing something important here
                        if (variableWorkload)
                            LockSupport.parkNanos(RandomUtils.nextInt(0, (int) time));
                        else
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
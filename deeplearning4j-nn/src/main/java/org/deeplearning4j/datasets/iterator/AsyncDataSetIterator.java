package org.deeplearning4j.datasets.iterator;


import org.nd4j.linalg.api.ops.executioner.GridExecutioner;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ConcurrentModificationException;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingDeque;
import java.util.concurrent.Semaphore;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * AsyncDataSetIterator takes an existing DataSetIterator and loads one or more DataSet objects
 * from it using a separate thread.
 * For data sets where DataSetIterator.next() is long running (limited by disk read or processing time
 * for example) this may improve performance by loading the next DataSet asynchronously (i.e., while
 * training is continuing on the previous DataSet). Obviously this may use additional memory.<br>
 * Note however that due to asynchronous loading of data, next(int) is not supported.
 * <p>
 * PLEASE NOTE: If used together with CUDA backend, please use it with caution.
 *
 * @author Alex Black
 * @author raver119@gmail.com
 */
public class AsyncDataSetIterator implements DataSetIterator {
    private final DataSetIterator baseIterator;
    private final BlockingQueue<DataSet> blockingQueue;
    private Thread thread;
    private IteratorRunnable runnable;

    protected static final Logger logger = LoggerFactory.getLogger(AsyncDataSetIterator.class);

    /**
     * Create an AsyncDataSetIterator with a queue size of 1 (i.e., only load a
     * single additional DataSet)
     *
     * @param baseIterator The DataSetIterator to load data from asynchronously
     */
    public AsyncDataSetIterator(DataSetIterator baseIterator) {
        this(baseIterator, 8);
    }

    /**
     * Create an AsyncDataSetIterator with a specified queue size.
     *
     * @param baseIterator The DataSetIterator to load data from asynchronously
     * @param queueSize    size of the queue (max number of elements to load into queue)
     */
    public AsyncDataSetIterator(DataSetIterator baseIterator, int queueSize) {
        if (queueSize <= 0)
            throw new IllegalArgumentException("Queue size must be > 0");
        if (queueSize < 2)
            queueSize = 2;

        this.baseIterator = baseIterator;
        if (this.baseIterator.resetSupported()) this.baseIterator.reset();
        blockingQueue = new LinkedBlockingDeque<>(queueSize);
        runnable = new IteratorRunnable(baseIterator.hasNext());
        thread = runnable;

        /**
         * We want to ensure, that background thread will have the same thread->device affinity, as master thread
         */
        Integer deviceId = Nd4j.getAffinityManager().getDeviceForCurrentThread();
        Nd4j.getAffinityManager().attachThreadToDevice(thread, deviceId);

        thread.setDaemon(true);
        thread.start();
    }


    @Override
    public DataSet next(int num) {
        // TODO: why isn't supported? We could just check queue size
        throw new UnsupportedOperationException("Next(int) not supported for AsyncDataSetIterator");
    }

    @Override
    public int totalExamples() {
        return baseIterator.totalExamples();
    }

    @Override
    public int inputColumns() {
        return baseIterator.inputColumns();
    }

    @Override
    public int totalOutcomes() {
        return baseIterator.totalOutcomes();
    }

    @Override
    public boolean resetSupported() {
        return baseIterator.resetSupported();
    }

    @Override
    public boolean asyncSupported() {
        return false;
    }

    @Override
    public synchronized void reset() {
        if (!resetSupported())
            throw new UnsupportedOperationException("Cannot reset Async iterator wrapping iterator that does not support reset");
        //Complication here: runnable could be blocking on either baseIterator.next() or blockingQueue.put()
        runnable.killRunnable = true;
        if (runnable.isAlive.get()) {
            thread.interrupt();
        }
        //Wait for runnable to exit, but should only have to wait very short period of time
        //This probably isn't necessary, but is included as a safeguard against race conditions
        try {
            runnable.runCompletedSemaphore.tryAcquire(5, TimeUnit.SECONDS);
        } catch (InterruptedException e) {
        }

        //Clear the queue, reset the base iterator, set up a new thread
        blockingQueue.clear();
        baseIterator.reset();
        runnable = new IteratorRunnable(baseIterator.hasNext());
        thread = runnable;

        Integer deviceId = Nd4j.getAffinityManager().getDeviceForCurrentThread();
        Nd4j.getAffinityManager().attachThreadToDevice(thread, deviceId);

        thread.setDaemon(true);
        thread.start();
    }

    @Override
    public int batch() {
        return baseIterator.batch();
    }

    @Override
    public int cursor() {
        return baseIterator.cursor();
    }

    @Override
    public int numExamples() {
        return baseIterator.numExamples();
    }

    @Override
    public void setPreProcessor(DataSetPreProcessor preProcessor) {
        baseIterator.setPreProcessor(preProcessor);
    }

    @Override
    public DataSetPreProcessor getPreProcessor() {
        return baseIterator.getPreProcessor();
    }

    @Override
    public List<String> getLabels() {
        return baseIterator.getLabels();
    }

    @Override
    public synchronized boolean hasNext() {
        if (!blockingQueue.isEmpty()) {
            return true;
        }

        if (runnable.isAlive.get()) {
            //Empty blocking queue, but runnable is alive
            //(a) runnable is blocking on baseIterator.next()
            //(b) runnable is blocking on blockingQueue.put()
            //either way: there's at least 1 more element to come

            // this is fix for possible race condition within runnable cycle
            return runnable.hasLatch();
        } else {
            if (!runnable.killRunnable && runnable.exception != null) {
                throw runnable.exception;   //Something went wrong
            }
            //Runnable has exited, presumably because it has fetched all elements
            return runnable.hasLatch();
        }
    }

    @Override
    public synchronized DataSet next() {
        if (!hasNext()) {
            throw new NoSuchElementException();
        }
        //If base iterator threw an unchecked exception: rethrow it now
        if (runnable.exception != null) {
            throw runnable.exception;
        }

        if (!blockingQueue.isEmpty()) {
            runnable.feeder.decrementAndGet();
            return blockingQueue.poll();    //non-blocking, but returns null if empty
        }

        //Blocking queue is empty, but more to come
        //Possible reasons:
        // (a) runnable died (already handled - runnable.exception != null)
        // (b) baseIterator.next() hasn't returned yet -> wait for it
        try {
            //Normally: just do blockingQueue.take(), but can't do that here
            //Reason: what if baseIterator.next() throws an exception after
            // blockingQueue.take() is called? In this case, next() will never return
            while (runnable.exception == null) {
                DataSet ds = blockingQueue.poll(2, TimeUnit.SECONDS);
                if (ds != null) {
                    runnable.feeder.decrementAndGet();
                    return ds;
                }
                if (runnable.killRunnable) {
                    //should never happen
                    throw new ConcurrentModificationException("Reset while next() is waiting for element?");
                }
                if (!runnable.isAlive.get() && blockingQueue.isEmpty()) {
                    if (runnable.exception != null)
                        throw new RuntimeException("Exception thrown in base iterator", runnable.exception);
                    throw new IllegalStateException("Unexpected state occurred for AsyncDataSetIterator: runnable died or no data available");
                }
            }
            //exception thrown while getting data from base iterator
            throw runnable.exception;
        } catch (InterruptedException e) {
            throw new RuntimeException(e);  //Shouldn't happen under normal circumstances
        }
    }

    /**
     * Shut down the async data set iterator thread
     * This is not typically necessary if using a single AsyncDataSetIterator
     * (thread is a daemon thread and so shouldn't block the JVM from exiting)
     * Behaviour of next(), hasNext() etc methods after shutdown of async iterator is undefined
     */
    public void shutdown() {
        if (thread != null && thread.isAlive()) {
            runnable.killRunnable = true;
            thread.interrupt();
            thread = null;
        }
    }

    private class IteratorRunnable extends Thread implements Runnable {
        private volatile boolean killRunnable = false;
        private volatile AtomicBoolean isAlive = new AtomicBoolean(true);
        private volatile RuntimeException exception;
        private Semaphore runCompletedSemaphore = new Semaphore(0);
        private ReentrantReadWriteLock lock = new ReentrantReadWriteLock();
        private AtomicLong feeder = new AtomicLong(0);

        public IteratorRunnable(boolean hasNext) {
            this.isAlive.set(hasNext);
            this.setName("AsyncIterator thread");
            this.setDaemon(true);
        }

        public boolean hasLatch() {
            /*
            This method was added to address possible race condition within runnable loop.
            Idea is simple: in 99% of cases semaphore won't lock in hasLatch calls, since method is called ONLY if there's nothing in queue,
            and if it's already locked within main runnable loop - we get fast TRUE.
         */

            // this is added just to avoid expensive lock
            if (feeder.get() > 0 || !blockingQueue.isEmpty())
                return true;

            try {
                lock.readLock().lock();
                boolean result = baseIterator.hasNext() || feeder.get() != 0 || !blockingQueue.isEmpty();
                if (!isAlive.get())
                    return result;
                else while (isAlive.get()) {
                    // in normal scenario this cycle is possible to hit into feeder state, since readLock is taken
                    result = feeder.get() != 0 || !blockingQueue.isEmpty() || baseIterator.hasNext();
                    if (result) return true;
                }
                return result;
            } finally {
                lock.readLock().unlock();
            }
        }

        @Override
        public void run() {
            try {
                while (!killRunnable && baseIterator.hasNext()) {
                    feeder.incrementAndGet();
                    lock.writeLock().lock();
                    DataSet ds = baseIterator.next();

                    if (Nd4j.getExecutioner() instanceof GridExecutioner)
                        ((GridExecutioner) Nd4j.getExecutioner()).flushQueueBlocking();

                    // feeder is temporary state variable, that shows if we have something between backend iterator and buffer

                    lock.writeLock().unlock();

                    blockingQueue.put(ds);
                }
                isAlive.set(false);
            } catch (InterruptedException e) {
                //thread.interrupt() while put(DataSet) was blocking
                if (killRunnable) {
                    return;
                } else
                    exception = new RuntimeException("Runnable interrupted unexpectedly", e); //Something else interrupted
            } catch (RuntimeException e) {
                exception = e;
                if (lock.writeLock().isHeldByCurrentThread()) {
                    lock.writeLock().unlock();
                }
            } finally {
                isAlive.set(false);
                runCompletedSemaphore.release();
            }
        }
    }

    @Override
    public void remove() {
    }

}

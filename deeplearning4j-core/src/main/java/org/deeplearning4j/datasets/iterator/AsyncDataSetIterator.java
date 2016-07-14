package org.deeplearning4j.datasets.iterator;


import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ConcurrentModificationException;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingDeque;
import java.util.concurrent.Semaphore;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

import org.nd4j.linalg.dataset.DataSet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 * AsyncDataSetIterator takes an existing DataSetIterator and loads one or more DataSet objects
 * from it using a separate thread.
 * For data sets where DataSetIterator.next() is long running (limited by disk read or processing time
 * for example) this may improve performance by loading the next DataSet asynchronously (i.e., while
 * training is continuing on the previous DataSet). Obviously this may use additional memory.<br>
 * Note however that due to asynchronous loading of data, next(int) is not supported.
 *
 * PLEASE NOTE: If used together with CUDA backend, please use it with caution.
 *
 * @author Alex Black
 */
public class AsyncDataSetIterator implements DataSetIterator {
    private final DataSetIterator baseIterator;
    private final BlockingQueue<DataSet> blockingQueue;
    private Thread thread;
    private IteratorRunnable runnable;

    protected static final Logger logger = LoggerFactory.getLogger(AsyncDataSetIterator.class);

    /**
     *
     * Create an AsyncDataSetIterator with a queue size of 1 (i.e., only load a
     * single additional DataSet)
     * @param baseIterator The DataSetIterator to load data from asynchronously
     */
    public AsyncDataSetIterator(DataSetIterator baseIterator){
        this(baseIterator,1);
    }

    /** Create an AsyncDataSetIterator with a specified queue size.
     * @param baseIterator The DataSetIterator to load data from asynchronously
     * @param queueSize size of the queue (max number of elements to load into queue)
     */
    public AsyncDataSetIterator(DataSetIterator baseIterator, int queueSize) {
        if(queueSize <= 0)
            throw new IllegalArgumentException("Queue size must be > 0");
        this.baseIterator = baseIterator;
        this.baseIterator.reset();
        blockingQueue = new LinkedBlockingDeque<>(queueSize);
        runnable = new IteratorRunnable(baseIterator.hasNext());
        thread = new Thread(runnable);

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
    public synchronized void reset() {
        //Complication here: runnable could be blocking on either baseIterator.next() or blockingQueue.put()
        runnable.killRunnable = true;
        if(runnable.isAlive.get()) {
            thread.interrupt();
        }
        //Wait for runnable to exit, but should only have to wait very short period of time
        //This probably isn't necessary, but is included as a safeguard against race conditions
        try{
            runnable.runCompletedSemaphore.tryAcquire(5, TimeUnit.SECONDS);
        } catch( InterruptedException e ){ }

        //Clear the queue, reset the base iterator, set up a new thread
        blockingQueue.clear();
        baseIterator.reset();
        runnable = new IteratorRunnable(baseIterator.hasNext());
        thread = new Thread(runnable);

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
    public synchronized  boolean hasNext() {
        if(!blockingQueue.isEmpty()) {
            logger.debug("Branch A");
            return true;
        }

        if(runnable.isAlive.get()) {
            //Empty blocking queue, but runnable is alive
            //(a) runnable is blocking on baseIterator.next()
            //(b) runnable is blocking on blockingQueue.put()
            //either way: there's at least 1 more element to come

            // this is fix for possible race condition within runnable cycle
            logger.debug("Branch B");
            return runnable.hasLatch();
        } else {
            if(!runnable.killRunnable && runnable.exception != null ) {
                throw runnable.exception;   //Something went wrong
            }
            //Runnable has exited, presumably because it has fetched all elements
            logger.debug("Branch C");
            return !blockingQueue.isEmpty();
        }
    }

    @Override
    public synchronized DataSet next() {
        if(!hasNext()) {
            throw new NoSuchElementException();
        }
        //If base iterator threw an unchecked exception: rethrow it now
        if(runnable.exception != null) {
            throw runnable.exception;
        }

        if(!blockingQueue.isEmpty()){
            return blockingQueue.poll();    //non-blocking, but returns null if empty
        }

        //Blocking queue is empty, but more to come
        //Possible reasons:
        // (a) runnable died (already handled - runnable.exception != null)
        // (b) baseIterator.next() hasn't returned yet -> wait for it
        try{
            //Normally: just do blockingQueue.take(), but can't do that here
            //Reason: what if baseIterator.next() throws an exception after
            // blockingQueue.take() is called? In this case, next() will never return
            while(runnable.exception == null ){
                DataSet ds = blockingQueue.poll(5,TimeUnit.SECONDS);
                if(ds != null) {
                    return ds;
                }
                if(runnable.killRunnable){
                    //should never happen
                    throw new ConcurrentModificationException("Reset while next() is waiting for element?");
                }
                if(!runnable.isAlive.get() && blockingQueue.isEmpty()){
                    throw new IllegalStateException("Unexpected state occurred for AsyncDataSetIterator: runnable died or no data available");
                }
            }
            //exception thrown while getting data from base iterator
            throw runnable.exception;
        }catch(InterruptedException e ){
            throw new RuntimeException(e);  //Shouldn't happen under normal circumstances
        }
    }

    /**
     *
     * Shut down the async data set iterator thread
     * This is not typically necessary if using a single AsyncDataSetIterator
     * (thread is a daemon thread and so shouldn't block the JVM from exiting)
     * Behaviour of next(), hasNext() etc methods after shutdown of async iterator is undefined
     */
    public void shutdown() {
        if(thread.isAlive()) {
            runnable.killRunnable = true;
            thread.interrupt();
        }
    }

    private class IteratorRunnable implements Runnable {
        private volatile boolean killRunnable = false;
        private volatile AtomicBoolean isAlive = new AtomicBoolean(true);
        private volatile RuntimeException exception;
        private Semaphore runCompletedSemaphore = new Semaphore(0);
        private Semaphore back = new Semaphore(1);

        public IteratorRunnable(boolean hasNext){
            this.isAlive.set(hasNext);
        }

        public boolean hasLatch() {
            /*
            This method was added to address possible race condition within runnable loop.
            Idea is simple: in 99% of cases semaphore won't lock in hasLatch calls, since method is called ONLY if there's nothing in queue,
            and if it's already locked within main runnable loop - we get fast TRUE.
         */
            logger.info("back permits: {}", back.availablePermits() );
            if (back.tryAcquire()) {
                boolean result = baseIterator.hasNext();
                logger.info("got result: {}", result);
                back.release();
                return result;
            } else {
                // if we're here, then at the request moment, we were inside runnable loop, and inside iterator there was something available as next
                return true;
            }
        }

        @Override
        public void run() {
            try {
                while (!killRunnable && baseIterator.hasNext()) {
                    try {
                        back.acquire();
                        blockingQueue.put(baseIterator.next());
                    } finally {
                        logger.info("Releasing");
                        back.release();
                    }
                }
            } catch( InterruptedException e ){
                //thread.interrupt() while put(DataSet) was blocking
                if(killRunnable) {
                    return;
                }
                else exception = new RuntimeException("Runnable interrupted unexpectedly",e); //Something else interrupted
            } catch(RuntimeException e ) {
                exception = e;
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

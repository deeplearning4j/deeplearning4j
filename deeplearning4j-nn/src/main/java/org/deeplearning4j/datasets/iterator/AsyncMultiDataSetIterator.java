package org.deeplearning4j.datasets.iterator;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.AllocationPolicy;
import org.nd4j.linalg.api.memory.enums.ResetPolicy;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Async prefetching iterator wrapper for MultiDataSetIterator implementations
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class AsyncMultiDataSetIterator implements MultiDataSetIterator {
    private MultiDataSetIterator backedIterator;

    private MultiDataSet terminator = new org.nd4j.linalg.dataset.MultiDataSet();
    private MultiDataSet nextElement = null;
    private BlockingQueue<MultiDataSet> buffer;
    private MemoryWorkspace workspace;
    private AsyncPrefetchThread thread;
    private AtomicBoolean shouldWork = new AtomicBoolean(true);
    private volatile RuntimeException throwable = null;


    public AsyncMultiDataSetIterator(MultiDataSetIterator baseIterator) {
        this(baseIterator, 8);
    }

    public AsyncMultiDataSetIterator(MultiDataSetIterator iterator, int queueSize, BlockingQueue<MultiDataSet> queue) {
        this(iterator, queueSize, queue, true);
    }

    public AsyncMultiDataSetIterator(MultiDataSetIterator baseIterator, int queueSize) {
        this(baseIterator, queueSize, new LinkedBlockingQueue<MultiDataSet>(queueSize));
    }

    public AsyncMultiDataSetIterator(MultiDataSetIterator baseIterator, int queueSize, boolean useWorkspace) {
        this(baseIterator, queueSize, new LinkedBlockingQueue<MultiDataSet>(queueSize), useWorkspace);
    }

    public AsyncMultiDataSetIterator(MultiDataSetIterator iterator, int queueSize, BlockingQueue<MultiDataSet> queue, boolean useWorkspace) {
        if (queueSize <= 0)
            throw new IllegalArgumentException("Queue size must be > 0");
        if (queueSize < 4)
            queueSize = 4;

        if (iterator.resetSupported() && useWorkspace) {
            iterator.reset();

            MultiDataSet ds = iterator.next();

            long initSize = Math.max(ds.getMemoryFootprint() * queueSize, 10 * 1024L * 1024L);

            WorkspaceConfiguration configuration = WorkspaceConfiguration.builder()
                    .initialSize(initSize)
                    .overallocationLimit(2.0)
                    .policyReset(ResetPolicy.ENDOFBUFFER_REACHED)
                    .policyAllocation(AllocationPolicy.OVERALLOCATE)
                    .build();

            MemoryWorkspace workspace = Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread(configuration, "AMDSI_ITER-" + java.util.UUID.randomUUID().toString());
            this.workspace = workspace;
        } else workspace = null;

        this.buffer = queue;
        this.backedIterator = iterator;

        if (iterator.resetSupported())
            this.backedIterator.reset();

        this.thread = new AsyncPrefetchThread(buffer, iterator, terminator, workspace);

        /**
         * We want to ensure, that background thread will have the same thread->device affinity, as master thread
         */
        Integer deviceId = Nd4j.getAffinityManager().getDeviceForCurrentThread();
        Nd4j.getAffinityManager().attachThreadToDevice(thread, deviceId);

        thread.setDaemon(true);
        thread.start();
    }

    /**
     * Like the standard next method but allows a
     * customizable number of examples returned
     *
     * @param num the number of examples
     * @return the next data applyTransformToDestination
     */
    @Override
    public MultiDataSet next(int num) {
        throw new UnsupportedOperationException();
    }

    /**
     * Set the preprocessor to be applied to each MultiDataSet, before each MultiDataSet is returned.
     *
     * @param preProcessor MultiDataSetPreProcessor. May be null.
     */
    @Override
    public void setPreProcessor(MultiDataSetPreProcessor preProcessor) {
        backedIterator.setPreProcessor(preProcessor);
    }

    /**
     * Is resetting supported by this DataSetIterator? Many DataSetIterators do support resetting,
     * but some don't
     *
     * @return true if reset method is supported; false otherwise
     */
    @Override
    public boolean resetSupported() {
        return backedIterator.resetSupported();
    }

    /**
     * Does this DataSetIterator support asynchronous prefetching of multiple DataSet objects?
     * Most DataSetIterators do, but in some cases it may not make sense to wrap this iterator in an
     * iterator that does asynchronous prefetching. For example, it would not make sense to use asynchronous
     * prefetching for the following types of iterators:
     * (a) Iterators that store their full contents in memory already
     * (b) Iterators that re-use features/labels arrays (as future next() calls will overwrite past contents)
     * (c) Iterators that already implement some level of asynchronous prefetching
     * (d) Iterators that may return different data depending on when the next() method is called
     *
     * @return true if asynchronous prefetching from this iterator is OK; false if asynchronous prefetching should not
     * be used with this iterator
     */
    @Override
    public boolean asyncSupported() {
        return false;
    }

    /**
     * Resets the iterator back to the beginning
     */
    @Override
    public void reset() {
        buffer.clear();


        if (thread != null)
            thread.interrupt();
        try {
            // Shutdown() should be a synchronous operation since the iterator is reset after shutdown() is
            // called in AsyncLabelAwareIterator.reset().
            if (thread != null)
                thread.join();
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }
        buffer.clear();

        backedIterator.reset();
        shouldWork.set(true);
        this.thread = new AsyncPrefetchThread(buffer, backedIterator, terminator, workspace);

        /**
         * We want to ensure, that background thread will have the same thread->device affinity, as master thread
         */
        Integer deviceId = Nd4j.getAffinityManager().getDeviceForCurrentThread();
        Nd4j.getAffinityManager().attachThreadToDevice(thread, deviceId);

        thread.setDaemon(true);
        thread.start();

        nextElement = null;
    }


    public void shutdown(){
        buffer.clear();


        if (thread != null)
            thread.interrupt();
        try {
            // Shutdown() should be a synchronous operation since the iterator is reset after shutdown() is
            // called in AsyncLabelAwareIterator.reset().
            if (thread != null)
                thread.join();
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }
        buffer.clear();

        if (this.workspace != null)
            Nd4j.getWorkspaceManager().destroyWorkspace(workspace);
    }


    /**
     * Returns {@code true} if the iteration has more elements.
     * (In other words, returns {@code true} if {@link #next} would
     * return an element rather than throwing an exception.)
     *
     * @return {@code true} if the iteration has more elements
     */
    @Override
    public boolean hasNext() {
        if (throwable != null)
            throw throwable;

        try {
            if (nextElement != null && nextElement != terminator) {
                return true;
            } else if(nextElement == terminator)
                return false;


            nextElement = buffer.take();

            if (nextElement == terminator)
                return false;

            return true;
        } catch (Exception e) {
            log.error("Premature end of loop!");
            throw new RuntimeException(e);
        }
    }

    /**
     * Returns the next element in the iteration.
     *
     * @return the next element in the iteration
     */
    @Override
    public MultiDataSet next() {
        if (throwable != null)
            throw throwable;

        MultiDataSet temp = nextElement;
        nextElement = null;
        return temp;
    }

    /**
     * Removes from the underlying collection the last element returned
     * by this iterator (optional operation).  This method can be called
     * only once per call to {@link #next}.  The behavior of an iterator
     * is unspecified if the underlying collection is modified while the
     * iteration is in progress in any way other than by calling this
     * method.
     *
     * @throws UnsupportedOperationException if the {@code remove}
     *                                       operation is not supported by this iterator
     * @throws IllegalStateException         if the {@code next} method has not
     *                                       yet been called, or the {@code remove} method has already
     *                                       been called after the last call to the {@code next}
     *                                       method
     * @implSpec The default implementation throws an instance of
     * {@link UnsupportedOperationException} and performs no other action.
     */
    @Override
    public void remove() {

    }

    protected class AsyncPrefetchThread extends Thread implements Runnable {
        private BlockingQueue<MultiDataSet> queue;
        private MultiDataSetIterator iterator;
        private MultiDataSet terminator;
        private MemoryWorkspace workspace;

        protected AsyncPrefetchThread(@NonNull BlockingQueue<MultiDataSet> queue, @NonNull MultiDataSetIterator iterator, @NonNull MultiDataSet terminator, MemoryWorkspace workspace) {
            this.queue = queue;
            this.iterator = iterator;
            this.terminator = terminator;
            this.workspace = workspace;

            this.setDaemon(true);
            this.setName("AMDSI prefetch thread");
        }

        @Override
        public void run() {
            try {
                while (iterator.hasNext() && shouldWork.get()) {
                    MultiDataSet smth = null;

                    if (workspace != null) {
                        try (MemoryWorkspace ws = workspace.notifyScopeEntered()) {
                            smth = iterator.next();
                        }
                    } else smth = iterator.next();

                    if (smth != null)
                        queue.put(smth);
                }
                queue.put(terminator);
            } catch (InterruptedException e) {
                // do nothing
                shouldWork.set(false);
            } catch (RuntimeException e) {
                throwable = e;
            } catch (Exception e) {
                throwable = new RuntimeException(e);
            }
        }
    }
}

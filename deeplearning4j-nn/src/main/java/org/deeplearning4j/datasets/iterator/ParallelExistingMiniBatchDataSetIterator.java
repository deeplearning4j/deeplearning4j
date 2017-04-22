package org.deeplearning4j.datasets.iterator;


import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.LockSupport;

/**
 * Read in existing mini batches created
 * by the mini batch file datasetiterator.
 *
 * @author Adam Gibson
 */
@Slf4j
public class ParallelExistingMiniBatchDataSetIterator implements DataSetIterator {

    public static final String DEFAULT_PATTERN = "dataset-%d.bin";

    private final AtomicInteger currIdx = new AtomicInteger(0);
    private File rootDir;
    private int totalBatches = -1;
    private DataSetPreProcessor dataSetPreProcessor;
    private final String pattern;

    private int numThreads = 2;
    private ThreadPoolExecutor executor;
    private final Future<DataSet> terminator = new DummyFuture();
    private Future<DataSet> nextElement = null;
    private BlockingQueue<Future<DataSet>> buffer;
    private AtomicBoolean shouldWork = new AtomicBoolean(true);
    private AsyncDispatcherThread thread;
    private int bufferSize;

    /**
     * Create with the given root directory, using the default filename pattern {@link #DEFAULT_PATTERN}
     * @param rootDir the root directory to use
     */
    public ParallelExistingMiniBatchDataSetIterator(File rootDir) {
        this(rootDir, DEFAULT_PATTERN);
    }

    public ParallelExistingMiniBatchDataSetIterator(File rootDir, int numThreads) {
        this(rootDir, DEFAULT_PATTERN, numThreads);
    }

    /**
     *
     * @param rootDir    The root directory to use
     * @param pattern    The filename pattern to use. Used with {@code String.format(pattern,idx)}, where idx is an
     *                   integer, starting at 0.
     */
    public ParallelExistingMiniBatchDataSetIterator(File rootDir, String pattern) {
        this(rootDir, pattern, 2);
    }

    public ParallelExistingMiniBatchDataSetIterator(File rootDir, String pattern, int numThreads) {
        this(rootDir, pattern, numThreads, 8);
    }

    public ParallelExistingMiniBatchDataSetIterator(File rootDir, String pattern, int numThreads, int bufferSize) {
        if (numThreads < 2)
            numThreads = 2;

        this.numThreads = numThreads;
        this.rootDir = rootDir;
        totalBatches = rootDir.list().length;
        this.pattern = pattern;
        this.bufferSize = bufferSize;

        this.buffer = new LinkedBlockingQueue<>(this.bufferSize);

        this.executor = (ThreadPoolExecutor) Executors.newFixedThreadPool(numThreads, new ThreadFactory() {
            @Override
            public Thread newThread(Runnable r) {
                Thread t = Executors.defaultThreadFactory().newThread(r);
                // PEMBDSI. I like it.
                t.setName("PEMBDSI pool thread ");
                t.setDaemon(true);
                return t;
            }
        });

        this.thread = new AsyncDispatcherThread();
        this.thread.start();
    }

    @Override
    public DataSet next(int num) {
        throw new UnsupportedOperationException("Unable to load custom number of examples");
    }

    @Override
    public int totalExamples() {
        throw new UnsupportedOperationException();

    }

    @Override
    public int inputColumns() {
        throw new UnsupportedOperationException();
    }

    @Override
    public int totalOutcomes() {
        throw new UnsupportedOperationException();
    }

    @Override
    public boolean resetSupported() {
        return true;
    }

    @Override
    public boolean asyncSupported() {
        return true;
    }

    @Override
    public void reset() {
        shouldWork.set(false);
        thread.interrupt();
        thread.shutdown();
        buffer.clear();
        nextElement = null;
        currIdx.set(0);
        shouldWork.set(true);
        this.thread = new AsyncDispatcherThread();
        this.thread.start();
    }

    @Override
    public int batch() {
        throw new UnsupportedOperationException();
    }

    @Override
    public int cursor() {
        return currIdx.get();
    }

    @Override
    public int numExamples() {
        throw new UnsupportedOperationException();
    }

    @Override
    public void setPreProcessor(DataSetPreProcessor preProcessor) {
        this.dataSetPreProcessor = preProcessor;
    }

    @Override
    public DataSetPreProcessor getPreProcessor() {
        return dataSetPreProcessor;
    }

    @Override
    public List<String> getLabels() {
        return null;
    }

    @Override
    public boolean hasNext() {
        try {
            if (nextElement != null && nextElement != terminator) {
                return true;
            } else if (nextElement == terminator) {
                return false;
            }

            nextElement = buffer.take();

            if (nextElement == terminator)
                return false;

            return true;
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        //return currIdx.get() < totalBatches;
    }


    @Override
    public DataSet next() {
        Future<DataSet> tmp = nextElement;
        nextElement = null;

        try {
            DataSet ds = tmp.get();

            if (dataSetPreProcessor != null)
                dataSetPreProcessor.preProcess(ds);

            if (ds.getFeatures().isAttached()) {
                if (Nd4j.getMemoryManager().getCurrentWorkspace() == null) {
                    ds.detach();
                } else {
                    ds.migrate();
                }
            }

            return ds;
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        /*
        try {
            DataSet ret = read(currIdx.getAndIncrement());
            if (dataSetPreProcessor != null)
                dataSetPreProcessor.preProcess(ret);

            return ret;
        } catch (IOException e) {
            throw new IllegalStateException("Unable to read dataset");
        }
        */
    }

    @Override
    public void remove() {
        //no opt;
    }

    private DataSet read(int idx) throws IOException {
        File path = new File(rootDir, String.format(pattern, idx));
        DataSet d = new DataSet();
        d.load(path);
        return d;
    }

    private DataSet read(File file) throws IOException {
        DataSet d = new DataSet();
        d.load(file);
        return d;
    }

    private class DummyFuture implements Future<DataSet> {

        @Override
        public boolean cancel(boolean mayInterruptIfRunning) {
            return false;
        }

        @Override
        public boolean isCancelled() {
            return false;
        }

        @Override
        public boolean isDone() {
            return true;
        }

        @Override
        public DataSet get() throws InterruptedException, ExecutionException {
            return null;
        }

        @Override
        public DataSet get(long timeout, TimeUnit unit) throws InterruptedException, ExecutionException, TimeoutException {
            return null;
        }
    }

    private class ReadCallable implements Callable<DataSet> {
        private File file;

        public ReadCallable(@NonNull File file) {
            this.file = file;
        }

        @Override
        public DataSet call() throws Exception {
            // TODO: workspaces? huh...
            DataSet ds = null;
            try {
                ds = read(file);
            } catch (Exception e) {
                e.printStackTrace();
                throw new RuntimeException(e);
            }

            return ds;
        }
    }


    private class AsyncDispatcherThread extends Thread implements Runnable {
        private AtomicBoolean isStopped = new AtomicBoolean(false);
        private RuntimeException exception;

        @Override
        public void run() {
            while (shouldWork.get() && currIdx.get() < totalBatches) {
                try {
                    File path = new File(rootDir, String.format(pattern, currIdx.getAndIncrement()));
                    Future<DataSet> ds = executor.submit(new ReadCallable(path));
                    buffer.put(ds);

                    if (currIdx.get() >= totalBatches)
                        buffer.put(terminator);
                } catch (InterruptedException e) {
                    shouldWork.set(false);
                } catch (Exception e) {
                    shouldWork.set(false);
                    e.printStackTrace();
                    this.exception = new RuntimeException(e);
                }
            }


            isStopped.set(true);
        }

        public void shutdown() {
            shouldWork.set(false);
            while (!isStopped.get())
                LockSupport.parkNanos(1000L);
        }
    }
}

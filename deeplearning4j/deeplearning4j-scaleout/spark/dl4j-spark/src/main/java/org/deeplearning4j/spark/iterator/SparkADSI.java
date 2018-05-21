package org.deeplearning4j.spark.iterator;

import lombok.extern.slf4j.Slf4j;
import org.apache.spark.TaskContext;
import org.apache.spark.TaskContextHelper;
import org.deeplearning4j.datasets.iterator.AsyncDataSetIterator;
import org.deeplearning4j.datasets.iterator.callbacks.DataSetCallback;
import org.deeplearning4j.datasets.iterator.callbacks.DefaultCallback;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;

/**
 * Spark version of AsyncDataSetIterator, made separate to propagate Spark TaskContext to new background thread, for Spark block locks compatibility
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class SparkADSI extends AsyncDataSetIterator {
    protected TaskContext context;

    protected SparkADSI() {
        super();
    }

    public SparkADSI(DataSetIterator baseIterator) {
        this(baseIterator, 8);
    }

    public SparkADSI(DataSetIterator iterator, int queueSize, BlockingQueue<DataSet> queue) {
        this(iterator, queueSize, queue, true);
    }

    public SparkADSI(DataSetIterator baseIterator, int queueSize) {
        this(baseIterator, queueSize, new LinkedBlockingQueue<DataSet>(queueSize));
    }

    public SparkADSI(DataSetIterator baseIterator, int queueSize, boolean useWorkspace) {
        this(baseIterator, queueSize, new LinkedBlockingQueue<DataSet>(queueSize), useWorkspace);
    }

    public SparkADSI(DataSetIterator baseIterator, int queueSize, boolean useWorkspace, Integer deviceId) {
        this(baseIterator, queueSize, new LinkedBlockingQueue<DataSet>(queueSize), useWorkspace, new DefaultCallback(),
                        deviceId);
    }

    public SparkADSI(DataSetIterator baseIterator, int queueSize, boolean useWorkspace, DataSetCallback callback) {
        this(baseIterator, queueSize, new LinkedBlockingQueue<DataSet>(queueSize), useWorkspace, callback);
    }

    public SparkADSI(DataSetIterator iterator, int queueSize, BlockingQueue<DataSet> queue, boolean useWorkspace) {
        this(iterator, queueSize, queue, useWorkspace, new DefaultCallback());
    }

    public SparkADSI(DataSetIterator iterator, int queueSize, BlockingQueue<DataSet> queue, boolean useWorkspace,
                    DataSetCallback callback) {
        this(iterator, queueSize, queue, useWorkspace, callback, Nd4j.getAffinityManager().getDeviceForCurrentThread());
    }

    public SparkADSI(DataSetIterator iterator, int queueSize, BlockingQueue<DataSet> queue, boolean useWorkspace,
                    DataSetCallback callback, Integer deviceId) {
        this();

        if (queueSize < 2)
            queueSize = 2;

        this.deviceId = deviceId;
        this.callback = callback;
        this.useWorkspace = useWorkspace;
        this.buffer = queue;
        this.prefetchSize = queueSize;
        this.backedIterator = iterator;
        this.workspaceId = "SADSI_ITER-" + java.util.UUID.randomUUID().toString();

        if (iterator.resetSupported())
            this.backedIterator.reset();

        context = TaskContext.get();

        this.thread = new SparkPrefetchThread(buffer, iterator, terminator, null);

        /**
         * We want to ensure, that background thread will have the same thread->device affinity, as master thread
         */

        Nd4j.getAffinityManager().attachThreadToDevice(thread, deviceId);
        thread.setDaemon(true);
        thread.start();
    }

    @Override
    protected void externalCall() {
        TaskContextHelper.setTaskContext(context);

    }

    public class SparkPrefetchThread extends AsyncPrefetchThread {

        protected SparkPrefetchThread(BlockingQueue<DataSet> queue, DataSetIterator iterator, DataSet terminator,
                        MemoryWorkspace workspace) {
            super(queue, iterator, terminator, workspace);
        }


    }
}

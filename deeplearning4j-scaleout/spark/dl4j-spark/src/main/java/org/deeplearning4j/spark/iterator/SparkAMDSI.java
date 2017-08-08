package org.deeplearning4j.spark.iterator;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.apache.spark.TaskContext;
import org.apache.spark.TaskContextHelper;
import org.deeplearning4j.datasets.iterator.AsyncMultiDataSetIterator;
import org.deeplearning4j.datasets.iterator.callbacks.DataSetCallback;
import org.deeplearning4j.datasets.iterator.callbacks.DefaultCallback;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;

/**
 * Spark version of AsyncMultiDataSetIterator, made separate to propagate Spark TaskContext to new background thread, for Spark block locks compatibility
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class SparkAMDSI extends AsyncMultiDataSetIterator {
    protected TaskContext context;

    protected SparkAMDSI() {
        super();
    }

    public SparkAMDSI(MultiDataSetIterator baseIterator) {
        this(baseIterator, 8);
    }

    public SparkAMDSI(MultiDataSetIterator iterator, int queueSize, BlockingQueue<MultiDataSet> queue) {
        this(iterator, queueSize, queue, true);
    }

    public SparkAMDSI(MultiDataSetIterator baseIterator, int queueSize) {
        this(baseIterator, queueSize, new LinkedBlockingQueue<MultiDataSet>(queueSize));
    }

    public SparkAMDSI(MultiDataSetIterator baseIterator, int queueSize, boolean useWorkspace) {
        this(baseIterator, queueSize, new LinkedBlockingQueue<MultiDataSet>(queueSize), useWorkspace);
    }

    public SparkAMDSI(MultiDataSetIterator baseIterator, int queueSize, boolean useWorkspace, Integer deviceId) {
        this(baseIterator, queueSize, new LinkedBlockingQueue<MultiDataSet>(queueSize), useWorkspace,
                        new DefaultCallback(), deviceId);
    }

    public SparkAMDSI(MultiDataSetIterator baseIterator, int queueSize, boolean useWorkspace,
                    DataSetCallback callback) {
        this(baseIterator, queueSize, new LinkedBlockingQueue<MultiDataSet>(queueSize), useWorkspace, callback);
    }

    public SparkAMDSI(MultiDataSetIterator iterator, int queueSize, BlockingQueue<MultiDataSet> queue,
                    boolean useWorkspace) {
        this(iterator, queueSize, queue, useWorkspace, null);
    }

    public SparkAMDSI(MultiDataSetIterator iterator, int queueSize, BlockingQueue<MultiDataSet> queue,
                    boolean useWorkspace, DataSetCallback callback) {
        this(iterator, queueSize, queue, useWorkspace, callback, Nd4j.getAffinityManager().getDeviceForCurrentThread());
    }

    public SparkAMDSI(MultiDataSetIterator iterator, int queueSize, BlockingQueue<MultiDataSet> queue,
                    boolean useWorkspace, DataSetCallback callback, Integer deviceId) {
        this();

        if (queueSize < 2)
            queueSize = 2;

        this.callback = callback;
        this.buffer = queue;
        this.backedIterator = iterator;
        this.useWorkspaces = useWorkspace;
        this.prefetchSize = queueSize;
        this.workspaceId = "SAMDSI_ITER-" + java.util.UUID.randomUUID().toString();
        this.deviceId = deviceId;

        if (iterator.resetSupported())
            this.backedIterator.reset();

        this.thread = new SparkPrefetchThread(buffer, iterator, terminator);

        context = TaskContext.get();

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

    protected class SparkPrefetchThread extends AsyncPrefetchThread {

        protected SparkPrefetchThread(@NonNull BlockingQueue<MultiDataSet> queue,
                        @NonNull MultiDataSetIterator iterator, @NonNull MultiDataSet terminator) {
            super(queue, iterator, terminator);
        }
    }
}

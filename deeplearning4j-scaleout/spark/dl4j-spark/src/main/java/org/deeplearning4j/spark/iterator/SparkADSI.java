package org.deeplearning4j.spark.iterator;

import lombok.extern.slf4j.Slf4j;
import org.apache.spark.TaskContext;
import org.apache.spark.TaskContextHelper;
import org.deeplearning4j.datasets.iterator.AsyncDataSetIterator;
import org.deeplearning4j.datasets.iterator.callbacks.DataSetCallback;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import java.lang.reflect.Field;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.util.concurrent.BlockingQueue;

/**
 * @author raver119@gmail.com
 */
@Slf4j
public class SparkADSI extends AsyncDataSetIterator {
    protected TaskContext context;

    public SparkADSI(DataSetIterator iterator, int queueSize, BlockingQueue<DataSet> queue, boolean useWorkspace, DataSetCallback callback, Integer deviceId) {
        super(iterator, queueSize, queue, useWorkspace, callback, deviceId);

        log.info("Spark ADSI");

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
        log.info("Pushing fake context");
        TaskContextHelper.setTaskContext(context);

    }

    public class SparkPrefetchThread extends AsyncPrefetchThread {

        protected SparkPrefetchThread(BlockingQueue<DataSet> queue, DataSetIterator iterator, DataSet terminator, MemoryWorkspace workspace) {
            super(queue, iterator, terminator, workspace);
        }


    }
}

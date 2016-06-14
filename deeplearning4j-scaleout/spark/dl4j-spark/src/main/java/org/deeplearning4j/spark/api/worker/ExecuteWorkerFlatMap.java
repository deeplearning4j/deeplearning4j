package org.deeplearning4j.spark.api.worker;

import org.apache.spark.api.java.function.FlatMapFunction;
import org.deeplearning4j.datasets.iterator.AsyncDataSetIterator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.spark.api.TrainingResult;
import org.deeplearning4j.spark.api.TrainingWorker;
import org.deeplearning4j.spark.api.WorkerConfiguration;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.util.Collections;
import java.util.Iterator;

/**
 * Created by Alex on 14/06/2016.
 */
public class ExecuteWorkerFlatMap<R extends TrainingResult> implements FlatMapFunction<Iterator<DataSet>, R> {

    private final TrainingWorker<R> worker;

    public ExecuteWorkerFlatMap(TrainingWorker<R> worker){
        this.worker = worker;
    }

    @Override
    public Iterable<R> call(Iterator<DataSet> dataSetIterator) throws Exception {
        WorkerConfiguration dataConfig = worker.getDataConfiguration();
        int batchSize = dataConfig.getBatchSizePerWorker();
        final int prefetchCount = dataConfig.getPrefetchNumBatches();

        DataSetIterator batchedIterator = new IteratorDataSetIterator(dataSetIterator, batchSize);
        if(prefetchCount > 0){
            batchedIterator = new AsyncDataSetIterator(batchedIterator, prefetchCount);
        }

        try {
            MultiLayerNetwork net = worker.getInitialModel();

            while (batchedIterator.hasNext()) {
                DataSet next = batchedIterator.next();
                R result = worker.processMinibatch(next, net, batchedIterator.hasNext());
                if(result != null){
                    //Terminate training immediately
                    return Collections.singletonList(result);
                }
            }

            //For some reason, we didn't return already. Normally this shouldn't happen
            return Collections.singletonList(worker.getFinalResult(net));
        } finally {
            //Make sure we shut down the async thread properly...
            if(batchedIterator instanceof AsyncDataSetIterator){
                ((AsyncDataSetIterator)batchedIterator).shutdown();
            }
        }


    }
}

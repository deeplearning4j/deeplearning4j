package org.deeplearning4j.spark.api.worker;

import org.apache.spark.api.java.function.FlatMapFunction;
import org.deeplearning4j.spark.api.TrainingResult;
import org.deeplearning4j.spark.api.TrainingWorker;
import org.deeplearning4j.spark.api.WorkerConfiguration;
import org.deeplearning4j.spark.iterator.PathSparkMultiDataSetIterator;
import org.nd4j.linalg.dataset.api.MultiDataSet;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/**
 * A FlatMapFunction for executing training on serialized DataSet objects, that can be loaded from a path (local or HDFS)
 * that is specified as a String
 * Used in both SparkDl4jMultiLayer and SparkComputationGraph implementations
 *
 * @author Alex Black
 */
public class ExecuteWorkerPathMDSFlatMap<R extends TrainingResult> implements FlatMapFunction<Iterator<String>, R> {
    private final FlatMapFunction<Iterator<MultiDataSet>, R> workerFlatMap;
    private final int maxMinibatches;

    public ExecuteWorkerPathMDSFlatMap(TrainingWorker<R> worker){
        this.workerFlatMap = new ExecuteWorkerMultiDataSetFlatMap<>(worker);
        WorkerConfiguration conf = worker.getDataConfiguration();
        this.maxMinibatches = (conf.getMaxBatchesPerWorker() > 0 ? conf.getMaxBatchesPerWorker() : Integer.MAX_VALUE);
    }

    @Override
    public Iterable<R> call(Iterator<String> iter) throws Exception {
        List<String> list = new ArrayList<>(maxMinibatches);
        int count = 0;
        while(iter.hasNext() && count++ < maxMinibatches){
            list.add(iter.next());
        }

        return workerFlatMap.call(new PathSparkMultiDataSetIterator(list.iterator()));
    }
}

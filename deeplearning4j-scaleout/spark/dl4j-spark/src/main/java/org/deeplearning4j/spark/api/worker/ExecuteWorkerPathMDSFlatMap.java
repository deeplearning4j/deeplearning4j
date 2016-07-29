package org.deeplearning4j.spark.api.worker;

import org.apache.spark.api.java.function.FlatMapFunction;
import org.deeplearning4j.spark.api.TrainingResult;
import org.deeplearning4j.spark.api.TrainingWorker;
import org.deeplearning4j.spark.iterator.PathSparkMultiDataSetIterator;
import org.nd4j.linalg.dataset.api.MultiDataSet;

import java.util.Iterator;

/**
 * A FlatMapFunction for executing training on serialized DataSet objects, that can be loaded from a path (local or HDFS)
 * that is specified as a String
 * Used in both SparkDl4jMultiLayer and SparkComputationGraph implementations
 *
 * @author Alex Black
 */
public class ExecuteWorkerPathMDSFlatMap<R extends TrainingResult> implements FlatMapFunction<Iterator<String>, R> {
    private final FlatMapFunction<Iterator<MultiDataSet>, R> workerFlatMap;

    public ExecuteWorkerPathMDSFlatMap(TrainingWorker<R> worker){
        this.workerFlatMap = new ExecuteWorkerMultiDataSetFlatMap<>(worker);
    }

    @Override
    public Iterable<R> call(Iterator<String> iter) throws Exception {
        return workerFlatMap.call(new PathSparkMultiDataSetIterator(iter));
    }
}

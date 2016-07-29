package org.deeplearning4j.spark.api.worker;

import org.apache.spark.api.java.function.FlatMapFunction;
import org.deeplearning4j.spark.api.TrainingResult;
import org.deeplearning4j.spark.api.TrainingWorker;
import org.deeplearning4j.spark.iterator.PathSparkDataSetIterator;
import org.nd4j.linalg.dataset.DataSet;

import java.util.Iterator;

/**
 * A FlatMapFunction for executing training on serialized DataSet objects, that can be loaded from a path (local or HDFS)
 * that is specified as a String
 * Used in both SparkDl4jMultiLayer and SparkComputationGraph implementations
 *
 * @author Alex Black
 */
public class ExecuteWorkerPathFlatMap<R extends TrainingResult> implements FlatMapFunction<Iterator<String>, R> {
    private final FlatMapFunction<Iterator<DataSet>, R> workerFlatMap;

    public ExecuteWorkerPathFlatMap(TrainingWorker<R> worker){
        this.workerFlatMap = new ExecuteWorkerFlatMap<>(worker);
    }

    @Override
    public Iterable<R> call(Iterator<String> iter) throws Exception {
        return workerFlatMap.call(new PathSparkDataSetIterator(iter));
    }
}

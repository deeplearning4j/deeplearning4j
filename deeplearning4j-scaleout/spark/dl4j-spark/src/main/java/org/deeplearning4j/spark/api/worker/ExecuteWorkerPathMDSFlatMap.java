package org.deeplearning4j.spark.api.worker;

import org.datavec.spark.functions.FlatMapFunctionAdapter;
import org.datavec.spark.transform.BaseFlatMapFunctionAdaptee;
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
public class ExecuteWorkerPathMDSFlatMap<R extends TrainingResult>
                extends BaseFlatMapFunctionAdaptee<Iterator<String>, R> {

    public ExecuteWorkerPathMDSFlatMap(TrainingWorker<R> worker) {
        super(new ExecuteWorkerPathMDSFlatMapAdapter<>(worker));
    }
}


/**
 * A FlatMapFunction for executing training on serialized DataSet objects, that can be loaded from a path (local or HDFS)
 * that is specified as a String
 * Used in both SparkDl4jMultiLayer and SparkComputationGraph implementations
 *
 * @author Alex Black
 */
class ExecuteWorkerPathMDSFlatMapAdapter<R extends TrainingResult>
                implements FlatMapFunctionAdapter<Iterator<String>, R> {
    private final FlatMapFunctionAdapter<Iterator<MultiDataSet>, R> workerFlatMap;
    private final int maxDataSetObjects;

    public ExecuteWorkerPathMDSFlatMapAdapter(TrainingWorker<R> worker) {
        this.workerFlatMap = new ExecuteWorkerMultiDataSetFlatMapAdapter<>(worker);

        //How many dataset objects of size 'dataSetObjectNumExamples' should we load?
        //Only pass on the required number, not all of them (to avoid async preloading data that won't be used)
        //Most of the time we'll get exactly the number we want, but this isn't guaranteed all the time for all
        // splitting strategies
        WorkerConfiguration conf = worker.getDataConfiguration();
        int dataSetObjectNumExamples = conf.getDataSetObjectSizeExamples();
        int workerMinibatchSize = conf.getBatchSizePerWorker();
        int maxMinibatches = (conf.getMaxBatchesPerWorker() > 0 ? conf.getMaxBatchesPerWorker() : Integer.MAX_VALUE);

        if (maxMinibatches == Integer.MAX_VALUE) {
            maxDataSetObjects = Integer.MAX_VALUE;
        } else {
            //Required: total number of examples / examples per dataset object
            maxDataSetObjects =
                            (int) Math.ceil(maxMinibatches * workerMinibatchSize / ((double) dataSetObjectNumExamples));
        }
    }

    @Override
    public Iterable<R> call(Iterator<String> iter) throws Exception {
        List<String> list = new ArrayList<>();
        int count = 0;
        while (iter.hasNext() && count++ < maxDataSetObjects) {
            list.add(iter.next());
        }

        return workerFlatMap.call(new PathSparkMultiDataSetIterator(list.iterator()));
    }
}

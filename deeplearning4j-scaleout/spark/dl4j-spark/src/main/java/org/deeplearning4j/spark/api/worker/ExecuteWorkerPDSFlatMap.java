package org.deeplearning4j.spark.api.worker;

import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.input.PortableDataStream;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.datasets.iterator.AsyncDataSetIterator;
import org.deeplearning4j.datasets.iterator.IteratorDataSetIterator;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.spark.api.TrainingResult;
import org.deeplearning4j.spark.api.TrainingWorker;
import org.deeplearning4j.spark.api.WorkerConfiguration;
import org.deeplearning4j.spark.api.stats.SparkTrainingStats;
import org.deeplearning4j.spark.api.stats.StatsCalculationHelper;
import org.deeplearning4j.spark.canova.iterator.PortableDataStreamDataSetIterator;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.util.Collections;
import java.util.Iterator;

/**
 * A FlatMapFunction for executing training on serialized DataSet objects, that can be loaded using a PortableDataStream
 * Used in both SparkDl4jMultiLayer and SparkComputationGraph implementations
 *
 * @author Alex Black
 */
public class ExecuteWorkerPDSFlatMap<R extends TrainingResult> implements FlatMapFunction<Iterator<PortableDataStream>, R> {
    private final FlatMapFunction<Iterator<DataSet>, R> workerFlatMap;

    public ExecuteWorkerPDSFlatMap(TrainingWorker<R> worker){
        this.workerFlatMap = new ExecuteWorkerFlatMap<>(worker);
    }

    @Override
    public Iterable<R> call(Iterator<PortableDataStream> iter) throws Exception {
        return workerFlatMap.call(new PortableDataStreamDataSetIterator(iter));
    }
}
